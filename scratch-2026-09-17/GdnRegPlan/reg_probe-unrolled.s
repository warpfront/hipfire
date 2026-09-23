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
	s_load_b32 s3, s[0:1], 0x10
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_mov_b32 s0, 0x3c23d70a
	v_fma_f32 v33, 0x3c23d70a, v9, 0
	v_fmaak_f32 v34, s0, v9, 0x38d1b717
	v_fmaak_f32 v35, s0, v9, 0x3951b717
	v_fmaak_f32 v36, s0, v9, 0x399d4951
	v_fmaak_f32 v37, s0, v9, 0x39d1b717
	v_fmaak_f32 v38, s0, v9, 0x3a03126e
	v_fmaak_f32 v39, s0, v9, 0x3a1d4951
	v_fmaak_f32 v40, s0, v9, 0x3a378034
	v_fmaak_f32 v64, s0, v9, 0x3a51b717
	v_fmaak_f32 v56, s0, v9, 0x3a6bedfa
	v_fmaak_f32 v48, s0, v9, 0x3a83126e
	v_fmaak_f32 v32, s0, v9, 0x3a902de0
	v_fmaak_f32 v24, s0, v9, 0x3a9d4951
	v_fmaak_f32 v16, s0, v9, 0x3aaa64c3
	v_fmaak_f32 v8, s0, v9, 0x3ab78034
	s_barrier_wait -1
	s_wait_kmcnt 0x0
	s_cmp_lt_i32 s3, 1
	global_inv scope:SCOPE_SE
	s_cbranch_scc1 .LBB4_11
; %bb.7:
	s_mov_b32 s0, 0x3a83126f
	v_dual_mov_b32 v58, v35 :: v_dual_and_b32 v19, 31, v0
	s_wait_alu depctr_sa_sdst(0)
	v_fma_mixlo_f16 v109, v1, s0, 0
	v_mbcnt_lo_u32_b32 v1, -1, 0
	v_fma_mixhi_f16 v109, v3, s0, 0
	v_fma_mixlo_f16 v106, v4, s0, 0
	v_fma_mixhi_f16 v105, v2, s0, 0
	v_lshrrev_b32_e32 v2, 1, v0
	v_xor_b32_e32 v3, 16, v1
	v_bfi_b32 v4, v1, 0, 32
	v_fma_mixhi_f16 v111, v14, s0, 0
	v_fma_mixlo_f16 v105, v9, s0, 0
	v_dual_mov_b32 v60, v37 :: v_dual_and_b32 v145, 8, v2
	s_delay_alu instid0(VALU_DEP_4)
	v_cmp_lt_u32_e32 vcc_lo, v3, v4
	v_fma_mixlo_f16 v110, v5, s0, 0
	v_lshlrev_b32_e32 v14, 7, v0
	v_fma_mixhi_f16 v106, v6, s0, 0
	v_fma_mixhi_f16 v110, v7, s0, 0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v1, v1, v3, vcc_lo
	v_fma_mixlo_f16 v107, v17, s0, 0
	v_fma_mixlo_f16 v111, v18, s0, 0
	v_fma_mixhi_f16 v107, v13, s0, 0
	v_fma_mixlo_f16 v108, v15, s0, 0
	v_fma_mixlo_f16 v112, v11, s0, 0
	v_fma_mixhi_f16 v108, v12, s0, 0
	v_fma_mixhi_f16 v112, v10, s0, 0
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
	v_dual_mov_b32 v21, v56 :: v_dual_mov_b32 v164, 0x3b03126f
	v_mov_b32_e32 v23, v32
	v_mov_b32_e32 v9, v39
	v_mov_b32_e32 v11, v64
	v_mov_b32_e32 v13, v48
	v_mov_b32_e32 v15, v24
	v_mov_b32_e32 v1, v40
	v_mov_b32_e32 v3, v56
	v_mov_b32_e32 v5, v32
	v_mov_b32_e32 v7, v16
	v_add_nc_u32_e32 v165, 0x1800, v147
	v_or_b32_e32 v166, 0x2000, v147
	v_add_nc_u32_e32 v167, 0xfffff800, v147
	s_mov_b32 s6, 0
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
	s_mov_b32 s7, 0
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
	v_dual_add_f32 v175, 0x37fba882, v72 :: v_dual_add_f32 v174, 0x37fba882, v71
	v_dual_add_f32 v173, 0x37fba882, v70 :: v_dual_add_f32 v172, 0x37fba882, v69
	v_dual_add_f32 v171, 0x37fba882, v68 :: v_dual_add_f32 v170, 0x37fba882, v67
	v_dual_add_f32 v169, 0x37fba882, v66 :: v_dual_add_f32 v168, 0x37fba882, v65
	v_dual_add_f32 v183, 0x3827c5ac, v72 :: v_dual_add_f32 v182, 0x3827c5ac, v71
	v_dual_add_f32 v181, 0x3827c5ac, v70 :: v_dual_add_f32 v180, 0x3827c5ac, v69
	v_dual_add_f32 v179, 0x3827c5ac, v68 :: v_dual_add_f32 v178, 0x3827c5ac, v67
	v_dual_add_f32 v177, 0x3827c5ac, v66 :: v_dual_add_f32 v176, 0x3827c5ac, v65
	v_wmma_f32_16x16x16_f16 v[73:80], v[105:108], v[113:116], v[73:80]
	v_wmma_f32_16x16x16_f16 v[81:88], v[105:108], v[113:116], v[81:88]
	v_wmma_f32_16x16x16_f16 v[168:175], v[105:108], v[113:116], v[168:175]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[176:183], v[105:108], v[113:116], v[176:183]
	v_wmma_f32_16x16x16_f16 v[73:80], v[105:108], v[117:120], v[73:80]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[81:88], v[105:108], v[117:120], v[81:88]
	v_wmma_f32_16x16x16_f16 v[168:175], v[105:108], v[117:120], v[168:175]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[176:183], v[105:108], v[117:120], v[176:183]
	v_wmma_f32_16x16x16_f16 v[73:80], v[105:108], v[121:124], v[73:80]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[81:88], v[105:108], v[121:124], v[81:88]
	v_wmma_f32_16x16x16_f16 v[168:175], v[105:108], v[121:124], v[168:175]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[176:183], v[105:108], v[121:124], v[176:183]
	v_wmma_f32_16x16x16_f16 v[73:80], v[105:108], v[125:128], v[73:80]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[81:88], v[105:108], v[125:128], v[81:88]
	v_wmma_f32_16x16x16_f16 v[168:175], v[105:108], v[125:128], v[168:175]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[176:183], v[105:108], v[125:128], v[176:183]
	v_wmma_f32_16x16x16_f16 v[73:80], v[105:108], v[129:132], v[73:80]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[81:88], v[105:108], v[129:132], v[81:88]
	v_wmma_f32_16x16x16_f16 v[168:175], v[105:108], v[129:132], v[168:175]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[176:183], v[105:108], v[129:132], v[176:183]
	v_wmma_f32_16x16x16_f16 v[73:80], v[105:108], v[133:136], v[73:80]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[81:88], v[105:108], v[133:136], v[81:88]
	v_wmma_f32_16x16x16_f16 v[168:175], v[105:108], v[133:136], v[168:175]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[176:183], v[105:108], v[133:136], v[176:183]
	v_wmma_f32_16x16x16_f16 v[73:80], v[105:108], v[137:140], v[73:80]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[81:88], v[105:108], v[137:140], v[81:88]
	v_wmma_f32_16x16x16_f16 v[168:175], v[105:108], v[137:140], v[168:175]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[176:183], v[105:108], v[137:140], v[176:183]
	v_wmma_f32_16x16x16_f16 v[73:80], v[105:108], v[141:144], v[73:80]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[81:88], v[105:108], v[141:144], v[81:88]
	v_wmma_f32_16x16x16_f16 v[168:175], v[105:108], v[141:144], v[168:175]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[176:183], v[105:108], v[141:144], v[176:183]
	v_dual_mul_f32 v97, 0x3727c5ac, v73 :: v_dual_mul_f32 v98, 0x3727c5ac, v74
	v_dual_mul_f32 v99, 0x3727c5ac, v75 :: v_dual_mul_f32 v100, 0x3727c5ac, v76
	v_dual_mul_f32 v101, 0x3727c5ac, v77 :: v_dual_mul_f32 v102, 0x3727c5ac, v78
	v_dual_mul_f32 v103, 0x3727c5ac, v79 :: v_dual_mul_f32 v104, 0x3727c5ac, v80
	v_dual_mul_f32 v89, 0x3727c5ac, v81 :: v_dual_mul_f32 v90, 0x3727c5ac, v82
	v_dual_mul_f32 v91, 0x3727c5ac, v83 :: v_dual_mul_f32 v92, 0x3727c5ac, v84
	v_dual_mul_f32 v93, 0x3727c5ac, v85 :: v_dual_mul_f32 v94, 0x3727c5ac, v86
	v_dual_mul_f32 v95, 0x3727c5ac, v87 :: v_dual_mul_f32 v96, 0x3727c5ac, v88
	v_dual_mul_f32 v81, 0x3727c5ac, v168 :: v_dual_mul_f32 v82, 0x3727c5ac, v169
	v_dual_mul_f32 v83, 0x3727c5ac, v170 :: v_dual_mul_f32 v84, 0x3727c5ac, v171
	v_dual_mul_f32 v85, 0x3727c5ac, v172 :: v_dual_mul_f32 v86, 0x3727c5ac, v173
	v_dual_mul_f32 v87, 0x3727c5ac, v174 :: v_dual_mul_f32 v88, 0x3727c5ac, v175
	v_dual_mul_f32 v73, 0x3727c5ac, v176 :: v_dual_mul_f32 v74, 0x3727c5ac, v177
	v_dual_mul_f32 v75, 0x3727c5ac, v178 :: v_dual_mul_f32 v76, 0x3727c5ac, v179
	v_dual_mul_f32 v77, 0x3727c5ac, v180 :: v_dual_mul_f32 v78, 0x3727c5ac, v181
	v_dual_mul_f32 v79, 0x3727c5ac, v182 :: v_dual_mul_f32 v80, 0x3727c5ac, v183
.LBB4_9:                                ;   Parent Loop BB4_8 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_wait_alu depctr_sa_sdst(0)
	v_cmp_eq_u32_e64 s1, s7, v147
	v_cmp_eq_u32_e64 s2, s7, v166
	v_cmp_eq_u32_e32 vcc_lo, s7, v167
	v_cmp_eq_u32_e64 s0, s7, v165
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v168, 0, v97, s1
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v168, v168, v81, s2
	ds_bpermute_b32 v169, v148, v168
	s_wait_dscnt 0x0
	v_dual_add_f32 v172, v168, v169 :: v_dual_add_nc_u32 v169, s7, v146
	v_lshlrev_b32_e32 v168, 2, v145
	s_addk_co_i32 s7, 0x2000
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s7, 0x4000
	ds_load_2addr_b32 v[170:171], v169 offset1:32
	s_wait_dscnt 0x0
	ds_bpermute_b32 v173, v168, v170
	s_wait_dscnt 0x0
	v_fma_f32 v97, -v172, v173, v97
	ds_bpermute_b32 v173, v149, v170
	s_wait_dscnt 0x0
	v_fma_f32 v98, -v172, v173, v98
	ds_bpermute_b32 v173, v150, v170
	s_wait_dscnt 0x0
	v_fma_f32 v99, -v172, v173, v99
	ds_bpermute_b32 v173, v151, v170
	s_wait_dscnt 0x0
	v_fma_f32 v100, -v172, v173, v100
	ds_bpermute_b32 v173, v152, v170
	s_wait_dscnt 0x0
	v_fma_f32 v101, -v172, v173, v101
	ds_bpermute_b32 v173, v153, v170
	s_wait_dscnt 0x0
	v_fma_f32 v102, -v172, v173, v102
	ds_bpermute_b32 v173, v154, v170
	s_wait_dscnt 0x0
	v_fma_f32 v103, -v172, v173, v103
	ds_bpermute_b32 v173, v155, v170
	s_wait_dscnt 0x0
	v_fma_f32 v104, -v172, v173, v104
	ds_bpermute_b32 v173, v156, v170
	s_wait_dscnt 0x0
	v_fma_f32 v89, -v172, v173, v89
	ds_bpermute_b32 v173, v157, v170
	s_wait_dscnt 0x0
	v_fma_f32 v90, -v172, v173, v90
	ds_bpermute_b32 v173, v158, v170
	s_wait_dscnt 0x0
	v_fma_f32 v91, -v172, v173, v91
	ds_bpermute_b32 v173, v159, v170
	s_wait_dscnt 0x0
	v_fma_f32 v92, -v172, v173, v92
	ds_bpermute_b32 v173, v160, v170
	s_wait_dscnt 0x0
	v_fma_f32 v93, -v172, v173, v93
	ds_bpermute_b32 v173, v161, v170
	s_wait_dscnt 0x0
	v_fma_f32 v94, -v172, v173, v94
	ds_bpermute_b32 v173, v162, v170
	ds_bpermute_b32 v170, v163, v170
	s_wait_dscnt 0x1
	v_fma_f32 v95, -v172, v173, v95
	s_wait_dscnt 0x0
	v_fma_f32 v96, -v172, v170, v96
	ds_bpermute_b32 v170, v168, v171
	s_wait_dscnt 0x0
	v_fma_f32 v81, -v172, v170, v81
	ds_bpermute_b32 v170, v149, v171
	s_wait_dscnt 0x0
	v_fma_f32 v82, -v172, v170, v82
	ds_bpermute_b32 v170, v150, v171
	s_wait_dscnt 0x0
	v_fma_f32 v83, -v172, v170, v83
	ds_bpermute_b32 v170, v151, v171
	s_wait_dscnt 0x0
	v_fma_f32 v84, -v172, v170, v84
	ds_bpermute_b32 v170, v152, v171
	s_wait_dscnt 0x0
	v_fma_f32 v85, -v172, v170, v85
	ds_bpermute_b32 v170, v153, v171
	s_wait_dscnt 0x0
	v_fma_f32 v86, -v172, v170, v86
	ds_bpermute_b32 v170, v154, v171
	s_wait_dscnt 0x0
	v_fma_f32 v87, -v172, v170, v87
	ds_bpermute_b32 v170, v155, v171
	s_wait_dscnt 0x0
	v_fma_f32 v88, -v172, v170, v88
	ds_bpermute_b32 v170, v156, v171
	s_wait_dscnt 0x0
	v_fma_f32 v73, -v172, v170, v73
	ds_bpermute_b32 v170, v157, v171
	s_wait_dscnt 0x0
	v_fma_f32 v74, -v172, v170, v74
	ds_bpermute_b32 v170, v158, v171
	s_wait_dscnt 0x0
	v_fma_f32 v75, -v172, v170, v75
	ds_bpermute_b32 v170, v159, v171
	s_wait_dscnt 0x0
	v_fma_f32 v76, -v172, v170, v76
	ds_bpermute_b32 v170, v160, v171
	s_wait_dscnt 0x0
	v_fma_f32 v77, -v172, v170, v77
	ds_bpermute_b32 v170, v161, v171
	s_wait_dscnt 0x0
	v_fma_f32 v78, -v172, v170, v78
	ds_bpermute_b32 v170, v162, v171
	s_wait_dscnt 0x0
	v_fma_f32 v79, -v172, v170, v79
	ds_bpermute_b32 v170, v163, v171
	s_wait_dscnt 0x0
	v_fma_f32 v80, -v172, v170, v80
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v170, 0, v98, s1
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v170, v170, v82, s2
	ds_bpermute_b32 v171, v148, v170
	s_wait_dscnt 0x0
	v_add_f32_e32 v172, v170, v171
	ds_load_2addr_b32 v[170:171], v169 offset0:64 offset1:96
	s_wait_dscnt 0x0
	ds_bpermute_b32 v173, v168, v170
	s_wait_dscnt 0x0
	v_fma_f32 v97, -v172, v173, v97
	ds_bpermute_b32 v173, v149, v170
	s_wait_dscnt 0x0
	v_fma_f32 v98, -v172, v173, v98
	ds_bpermute_b32 v173, v150, v170
	s_wait_dscnt 0x0
	v_fma_f32 v99, -v172, v173, v99
	ds_bpermute_b32 v173, v151, v170
	s_wait_dscnt 0x0
	v_fma_f32 v100, -v172, v173, v100
	ds_bpermute_b32 v173, v152, v170
	s_wait_dscnt 0x0
	v_fma_f32 v101, -v172, v173, v101
	ds_bpermute_b32 v173, v153, v170
	s_wait_dscnt 0x0
	v_fma_f32 v102, -v172, v173, v102
	ds_bpermute_b32 v173, v154, v170
	s_wait_dscnt 0x0
	v_fma_f32 v103, -v172, v173, v103
	ds_bpermute_b32 v173, v155, v170
	s_wait_dscnt 0x0
	v_fma_f32 v104, -v172, v173, v104
	ds_bpermute_b32 v173, v156, v170
	s_wait_dscnt 0x0
	v_fma_f32 v89, -v172, v173, v89
	ds_bpermute_b32 v173, v157, v170
	s_wait_dscnt 0x0
	v_fma_f32 v90, -v172, v173, v90
	ds_bpermute_b32 v173, v158, v170
	s_wait_dscnt 0x0
	v_fma_f32 v91, -v172, v173, v91
	ds_bpermute_b32 v173, v159, v170
	s_wait_dscnt 0x0
	v_fma_f32 v92, -v172, v173, v92
	ds_bpermute_b32 v173, v160, v170
	s_wait_dscnt 0x0
	v_fma_f32 v93, -v172, v173, v93
	ds_bpermute_b32 v173, v161, v170
	s_wait_dscnt 0x0
	v_fma_f32 v94, -v172, v173, v94
	ds_bpermute_b32 v173, v162, v170
	ds_bpermute_b32 v170, v163, v170
	s_wait_dscnt 0x1
	v_fma_f32 v95, -v172, v173, v95
	s_wait_dscnt 0x0
	v_fma_f32 v96, -v172, v170, v96
	ds_bpermute_b32 v170, v168, v171
	s_wait_dscnt 0x0
	v_fma_f32 v81, -v172, v170, v81
	ds_bpermute_b32 v170, v149, v171
	s_wait_dscnt 0x0
	v_fma_f32 v82, -v172, v170, v82
	ds_bpermute_b32 v170, v150, v171
	s_wait_dscnt 0x0
	v_fma_f32 v83, -v172, v170, v83
	ds_bpermute_b32 v170, v151, v171
	s_wait_dscnt 0x0
	v_fma_f32 v84, -v172, v170, v84
	ds_bpermute_b32 v170, v152, v171
	s_wait_dscnt 0x0
	v_fma_f32 v85, -v172, v170, v85
	ds_bpermute_b32 v170, v153, v171
	s_wait_dscnt 0x0
	v_fma_f32 v86, -v172, v170, v86
	ds_bpermute_b32 v170, v154, v171
	s_wait_dscnt 0x0
	v_fma_f32 v87, -v172, v170, v87
	ds_bpermute_b32 v170, v155, v171
	s_wait_dscnt 0x0
	v_fma_f32 v88, -v172, v170, v88
	ds_bpermute_b32 v170, v156, v171
	s_wait_dscnt 0x0
	v_fma_f32 v73, -v172, v170, v73
	ds_bpermute_b32 v170, v157, v171
	s_wait_dscnt 0x0
	v_fma_f32 v74, -v172, v170, v74
	ds_bpermute_b32 v170, v158, v171
	s_wait_dscnt 0x0
	v_fma_f32 v75, -v172, v170, v75
	ds_bpermute_b32 v170, v159, v171
	s_wait_dscnt 0x0
	v_fma_f32 v76, -v172, v170, v76
	ds_bpermute_b32 v170, v160, v171
	s_wait_dscnt 0x0
	v_fma_f32 v77, -v172, v170, v77
	ds_bpermute_b32 v170, v161, v171
	s_wait_dscnt 0x0
	v_fma_f32 v78, -v172, v170, v78
	ds_bpermute_b32 v170, v162, v171
	s_wait_dscnt 0x0
	v_fma_f32 v79, -v172, v170, v79
	ds_bpermute_b32 v170, v163, v171
	s_wait_dscnt 0x0
	v_fma_f32 v80, -v172, v170, v80
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v170, 0, v99, s1
	ds_load_2addr_b32 v[184:185], v169 offset0:128 offset1:160
	v_cndmask_b32_e64 v170, v170, v83, s2
	ds_bpermute_b32 v171, v148, v170
	s_wait_dscnt 0x0
	v_add_f32_e32 v186, v170, v171
	ds_bpermute_b32 v170, v168, v184
	s_wait_dscnt 0x0
	v_fma_f32 v97, -v186, v170, v97
	ds_bpermute_b32 v170, v149, v184
	s_wait_dscnt 0x0
	v_fma_f32 v98, -v186, v170, v98
	ds_bpermute_b32 v170, v150, v184
	s_wait_dscnt 0x0
	v_fma_f32 v99, -v186, v170, v99
	ds_bpermute_b32 v170, v151, v184
	s_wait_dscnt 0x0
	v_fma_f32 v100, -v186, v170, v100
	ds_bpermute_b32 v170, v152, v184
	s_wait_dscnt 0x0
	v_fma_f32 v101, -v186, v170, v101
	ds_bpermute_b32 v170, v153, v184
	s_wait_dscnt 0x0
	v_fma_f32 v102, -v186, v170, v102
	ds_bpermute_b32 v170, v154, v184
	s_wait_dscnt 0x0
	v_fma_f32 v103, -v186, v170, v103
	ds_bpermute_b32 v170, v155, v184
	s_wait_dscnt 0x0
	v_fma_f32 v104, -v186, v170, v104
	ds_bpermute_b32 v170, v156, v184
	s_wait_dscnt 0x0
	v_fma_f32 v89, -v186, v170, v89
	ds_bpermute_b32 v170, v157, v184
	s_wait_dscnt 0x0
	v_fma_f32 v90, -v186, v170, v90
	ds_bpermute_b32 v170, v158, v184
	s_wait_dscnt 0x0
	v_fma_f32 v91, -v186, v170, v91
	ds_bpermute_b32 v170, v159, v184
	s_wait_dscnt 0x0
	v_fma_f32 v92, -v186, v170, v92
	ds_bpermute_b32 v170, v160, v184
	s_wait_dscnt 0x0
	v_fma_f32 v93, -v186, v170, v93
	ds_bpermute_b32 v170, v161, v184
	s_wait_dscnt 0x0
	v_fma_f32 v94, -v186, v170, v94
	ds_bpermute_b32 v170, v162, v184
	s_wait_dscnt 0x0
	v_fma_f32 v95, -v186, v170, v95
	ds_bpermute_b32 v170, v163, v184
	s_wait_dscnt 0x0
	v_fma_f32 v96, -v186, v170, v96
	ds_bpermute_b32 v170, v168, v185
	s_wait_dscnt 0x0
	v_fma_f32 v170, -v186, v170, v81
	ds_bpermute_b32 v81, v149, v185
	s_wait_dscnt 0x0
	v_fma_f32 v171, -v186, v81, v82
	ds_bpermute_b32 v81, v150, v185
	s_wait_dscnt 0x0
	v_fma_f32 v172, -v186, v81, v83
	ds_bpermute_b32 v81, v151, v185
	s_wait_dscnt 0x0
	v_fma_f32 v173, -v186, v81, v84
	ds_bpermute_b32 v81, v152, v185
	s_wait_dscnt 0x0
	v_fma_f32 v174, -v186, v81, v85
	ds_bpermute_b32 v81, v153, v185
	s_wait_dscnt 0x0
	v_fma_f32 v175, -v186, v81, v86
	ds_bpermute_b32 v81, v154, v185
	s_wait_dscnt 0x0
	v_fma_f32 v176, -v186, v81, v87
	ds_bpermute_b32 v81, v155, v185
	s_wait_dscnt 0x0
	v_fma_f32 v177, -v186, v81, v88
	ds_bpermute_b32 v81, v156, v185
	s_wait_dscnt 0x0
	v_fma_f32 v178, -v186, v81, v73
	ds_bpermute_b32 v73, v157, v185
	s_wait_dscnt 0x0
	v_fma_f32 v179, -v186, v73, v74
	ds_bpermute_b32 v73, v158, v185
	s_wait_dscnt 0x0
	v_fma_f32 v180, -v186, v73, v75
	ds_bpermute_b32 v73, v159, v185
	s_wait_dscnt 0x0
	v_fma_f32 v181, -v186, v73, v76
	ds_bpermute_b32 v73, v160, v185
	s_wait_dscnt 0x0
	v_fma_f32 v182, -v186, v73, v77
	ds_bpermute_b32 v73, v161, v185
	s_wait_dscnt 0x0
	v_fma_f32 v183, -v186, v73, v78
	ds_bpermute_b32 v73, v162, v185
	s_wait_dscnt 0x0
	v_fma_f32 v184, -v186, v73, v79
	ds_bpermute_b32 v73, v163, v185
	s_wait_dscnt 0x0
	v_fma_f32 v185, -v186, v73, v80
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v73, 0, v100, s1
	ds_load_2addr_b32 v[186:187], v169 offset0:192 offset1:224
	v_cndmask_b32_e64 v73, v73, v173, s2
	ds_bpermute_b32 v74, v148, v73
	s_wait_dscnt 0x1
	ds_bpermute_b32 v75, v150, v186
	ds_bpermute_b32 v76, v151, v186
	ds_bpermute_b32 v77, v152, v186
	ds_bpermute_b32 v78, v153, v186
	ds_bpermute_b32 v79, v154, v186
	ds_bpermute_b32 v80, v155, v186
	ds_bpermute_b32 v81, v156, v186
	ds_bpermute_b32 v82, v157, v186
	ds_bpermute_b32 v83, v158, v186
	s_wait_dscnt 0x9
	v_add_f32_e32 v188, v73, v74
	ds_bpermute_b32 v73, v168, v186
	ds_bpermute_b32 v74, v149, v186
	ds_bpermute_b32 v84, v159, v186
	ds_bpermute_b32 v85, v160, v186
	ds_bpermute_b32 v86, v161, v186
	ds_bpermute_b32 v87, v162, v186
	ds_bpermute_b32 v88, v163, v186
	v_add_nc_u32_e32 v186, 0x1800, v169
	s_wait_dscnt 0xf
	v_fma_f32 v75, -v188, v75, v99
	s_wait_dscnt 0xe
	v_fma_f32 v76, -v188, v76, v100
	s_wait_dscnt 0xd
	v_fma_f32 v77, -v188, v77, v101
	s_wait_dscnt 0xc
	v_fma_f32 v78, -v188, v78, v102
	s_wait_dscnt 0xb
	v_fma_f32 v79, -v188, v79, v103
	s_wait_dscnt 0xa
	v_fma_f32 v80, -v188, v80, v104
	s_wait_dscnt 0x9
	v_fma_f32 v81, -v188, v81, v89
	s_wait_dscnt 0x8
	v_fma_f32 v82, -v188, v82, v90
	s_wait_dscnt 0x7
	v_fma_f32 v83, -v188, v83, v91
	ds_bpermute_b32 v89, v168, v187
	s_wait_dscnt 0x7
	v_fma_f32 v73, -v188, v73, v97
	s_wait_dscnt 0x6
	v_fma_f32 v74, -v188, v74, v98
	s_wait_dscnt 0x5
	v_fma_f32 v84, -v188, v84, v92
	s_wait_dscnt 0x4
	v_fma_f32 v85, -v188, v85, v93
	s_wait_dscnt 0x3
	v_fma_f32 v86, -v188, v86, v94
	s_wait_dscnt 0x2
	v_fma_f32 v87, -v188, v87, v95
	s_wait_dscnt 0x1
	v_fma_f32 v88, -v188, v88, v96
	ds_bpermute_b32 v90, v149, v187
	ds_bpermute_b32 v91, v150, v187
	ds_bpermute_b32 v92, v151, v187
	ds_bpermute_b32 v93, v152, v187
	ds_bpermute_b32 v94, v153, v187
	ds_bpermute_b32 v95, v154, v187
	ds_bpermute_b32 v96, v155, v187
	ds_bpermute_b32 v97, v156, v187
	ds_bpermute_b32 v98, v157, v187
	ds_bpermute_b32 v99, v158, v187
	ds_bpermute_b32 v100, v159, v187
	ds_bpermute_b32 v101, v160, v187
	ds_bpermute_b32 v102, v161, v187
	ds_bpermute_b32 v103, v162, v187
	ds_bpermute_b32 v104, v163, v187
	s_wait_dscnt 0xf
	v_fma_f32 v89, -v188, v89, v170
	s_wait_dscnt 0xe
	v_fma_f32 v90, -v188, v90, v171
	s_wait_dscnt 0xd
	v_fma_f32 v91, -v188, v91, v172
	s_wait_dscnt 0xc
	v_fma_f32 v92, -v188, v92, v173
	s_wait_dscnt 0xb
	v_fma_f32 v93, -v188, v93, v174
	s_wait_dscnt 0xa
	v_fma_f32 v94, -v188, v94, v175
	s_wait_dscnt 0x9
	v_fma_f32 v95, -v188, v95, v176
	s_wait_dscnt 0x8
	v_fma_f32 v96, -v188, v96, v177
	s_wait_dscnt 0x7
	v_fma_f32 v97, -v188, v97, v178
	s_wait_dscnt 0x6
	v_fma_f32 v98, -v188, v98, v179
	s_wait_dscnt 0x5
	v_fma_f32 v99, -v188, v99, v180
	s_wait_dscnt 0x4
	v_fma_f32 v100, -v188, v100, v181
	s_wait_dscnt 0x3
	v_fma_f32 v101, -v188, v101, v182
	s_wait_dscnt 0x2
	v_fma_f32 v102, -v188, v102, v183
	s_wait_dscnt 0x1
	v_fma_f32 v103, -v188, v103, v184
	s_wait_dscnt 0x0
	v_fma_f32 v104, -v188, v104, v185
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v170, 0, v77, s1
	v_add_nc_u32_e32 v173, 0x400, v169
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e64 v170, v170, v93, s2
	ds_bpermute_b32 v171, v148, v170
	s_wait_dscnt 0x0
	v_add_f32_e32 v172, v170, v171
	ds_load_2addr_b32 v[170:171], v173 offset1:32
	s_wait_dscnt 0x0
	ds_bpermute_b32 v174, v168, v170
	s_wait_dscnt 0x0
	v_fma_f32 v73, -v172, v174, v73
	ds_bpermute_b32 v174, v149, v170
	s_wait_dscnt 0x0
	v_fma_f32 v74, -v172, v174, v74
	ds_bpermute_b32 v174, v150, v170
	s_wait_dscnt 0x0
	v_fma_f32 v75, -v172, v174, v75
	ds_bpermute_b32 v174, v151, v170
	s_wait_dscnt 0x0
	v_fma_f32 v76, -v172, v174, v76
	ds_bpermute_b32 v174, v152, v170
	s_wait_dscnt 0x0
	v_fma_f32 v77, -v172, v174, v77
	ds_bpermute_b32 v174, v153, v170
	s_wait_dscnt 0x0
	v_fma_f32 v78, -v172, v174, v78
	ds_bpermute_b32 v174, v154, v170
	s_wait_dscnt 0x0
	v_fma_f32 v79, -v172, v174, v79
	ds_bpermute_b32 v174, v155, v170
	s_wait_dscnt 0x0
	v_fma_f32 v80, -v172, v174, v80
	ds_bpermute_b32 v174, v156, v170
	s_wait_dscnt 0x0
	v_fma_f32 v81, -v172, v174, v81
	ds_bpermute_b32 v174, v157, v170
	s_wait_dscnt 0x0
	v_fma_f32 v82, -v172, v174, v82
	ds_bpermute_b32 v174, v158, v170
	s_wait_dscnt 0x0
	v_fma_f32 v83, -v172, v174, v83
	ds_bpermute_b32 v174, v159, v170
	s_wait_dscnt 0x0
	v_fma_f32 v84, -v172, v174, v84
	ds_bpermute_b32 v174, v160, v170
	s_wait_dscnt 0x0
	v_fma_f32 v85, -v172, v174, v85
	ds_bpermute_b32 v174, v161, v170
	s_wait_dscnt 0x0
	v_fma_f32 v86, -v172, v174, v86
	ds_bpermute_b32 v174, v162, v170
	ds_bpermute_b32 v170, v163, v170
	s_wait_dscnt 0x1
	v_fma_f32 v87, -v172, v174, v87
	s_wait_dscnt 0x0
	v_fma_f32 v88, -v172, v170, v88
	ds_bpermute_b32 v170, v168, v171
	s_wait_dscnt 0x0
	v_fma_f32 v89, -v172, v170, v89
	ds_bpermute_b32 v170, v149, v171
	s_wait_dscnt 0x0
	v_fma_f32 v90, -v172, v170, v90
	ds_bpermute_b32 v170, v150, v171
	s_wait_dscnt 0x0
	v_fma_f32 v91, -v172, v170, v91
	ds_bpermute_b32 v170, v151, v171
	s_wait_dscnt 0x0
	v_fma_f32 v92, -v172, v170, v92
	ds_bpermute_b32 v170, v152, v171
	s_wait_dscnt 0x0
	v_fma_f32 v93, -v172, v170, v93
	ds_bpermute_b32 v170, v153, v171
	s_wait_dscnt 0x0
	v_fma_f32 v94, -v172, v170, v94
	ds_bpermute_b32 v170, v154, v171
	s_wait_dscnt 0x0
	v_fma_f32 v95, -v172, v170, v95
	ds_bpermute_b32 v170, v155, v171
	s_wait_dscnt 0x0
	v_fma_f32 v96, -v172, v170, v96
	ds_bpermute_b32 v170, v156, v171
	s_wait_dscnt 0x0
	v_fma_f32 v97, -v172, v170, v97
	ds_bpermute_b32 v170, v157, v171
	s_wait_dscnt 0x0
	v_fma_f32 v98, -v172, v170, v98
	ds_bpermute_b32 v170, v158, v171
	s_wait_dscnt 0x0
	v_fma_f32 v99, -v172, v170, v99
	ds_bpermute_b32 v170, v159, v171
	s_wait_dscnt 0x0
	v_fma_f32 v100, -v172, v170, v100
	ds_bpermute_b32 v170, v160, v171
	s_wait_dscnt 0x0
	v_fma_f32 v101, -v172, v170, v101
	ds_bpermute_b32 v170, v161, v171
	s_wait_dscnt 0x0
	v_fma_f32 v102, -v172, v170, v102
	ds_bpermute_b32 v170, v162, v171
	s_wait_dscnt 0x0
	v_fma_f32 v103, -v172, v170, v103
	ds_bpermute_b32 v170, v163, v171
	s_wait_dscnt 0x0
	v_fma_f32 v104, -v172, v170, v104
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v170, 0, v78, s1
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v170, v170, v94, s2
	ds_bpermute_b32 v171, v148, v170
	s_wait_dscnt 0x0
	v_add_f32_e32 v172, v170, v171
	ds_load_2addr_b32 v[170:171], v173 offset0:64 offset1:96
	s_wait_dscnt 0x0
	ds_bpermute_b32 v174, v168, v170
	s_wait_dscnt 0x0
	v_fma_f32 v73, -v172, v174, v73
	ds_bpermute_b32 v174, v149, v170
	s_wait_dscnt 0x0
	v_fma_f32 v74, -v172, v174, v74
	ds_bpermute_b32 v174, v150, v170
	s_wait_dscnt 0x0
	v_fma_f32 v75, -v172, v174, v75
	ds_bpermute_b32 v174, v151, v170
	s_wait_dscnt 0x0
	v_fma_f32 v76, -v172, v174, v76
	ds_bpermute_b32 v174, v152, v170
	s_wait_dscnt 0x0
	v_fma_f32 v77, -v172, v174, v77
	ds_bpermute_b32 v174, v153, v170
	s_wait_dscnt 0x0
	v_fma_f32 v78, -v172, v174, v78
	ds_bpermute_b32 v174, v154, v170
	s_wait_dscnt 0x0
	v_fma_f32 v79, -v172, v174, v79
	ds_bpermute_b32 v174, v155, v170
	s_wait_dscnt 0x0
	v_fma_f32 v80, -v172, v174, v80
	ds_bpermute_b32 v174, v156, v170
	s_wait_dscnt 0x0
	v_fma_f32 v81, -v172, v174, v81
	ds_bpermute_b32 v174, v157, v170
	s_wait_dscnt 0x0
	v_fma_f32 v82, -v172, v174, v82
	ds_bpermute_b32 v174, v158, v170
	s_wait_dscnt 0x0
	v_fma_f32 v83, -v172, v174, v83
	ds_bpermute_b32 v174, v159, v170
	s_wait_dscnt 0x0
	v_fma_f32 v84, -v172, v174, v84
	ds_bpermute_b32 v174, v160, v170
	s_wait_dscnt 0x0
	v_fma_f32 v85, -v172, v174, v85
	ds_bpermute_b32 v174, v161, v170
	s_wait_dscnt 0x0
	v_fma_f32 v86, -v172, v174, v86
	ds_bpermute_b32 v174, v162, v170
	ds_bpermute_b32 v170, v163, v170
	s_wait_dscnt 0x1
	v_fma_f32 v87, -v172, v174, v87
	s_wait_dscnt 0x0
	v_fma_f32 v88, -v172, v170, v88
	ds_bpermute_b32 v170, v168, v171
	s_wait_dscnt 0x0
	v_fma_f32 v89, -v172, v170, v89
	ds_bpermute_b32 v170, v149, v171
	s_wait_dscnt 0x0
	v_fma_f32 v90, -v172, v170, v90
	ds_bpermute_b32 v170, v150, v171
	s_wait_dscnt 0x0
	v_fma_f32 v91, -v172, v170, v91
	ds_bpermute_b32 v170, v151, v171
	s_wait_dscnt 0x0
	v_fma_f32 v92, -v172, v170, v92
	ds_bpermute_b32 v170, v152, v171
	s_wait_dscnt 0x0
	v_fma_f32 v93, -v172, v170, v93
	ds_bpermute_b32 v170, v153, v171
	s_wait_dscnt 0x0
	v_fma_f32 v94, -v172, v170, v94
	ds_bpermute_b32 v170, v154, v171
	s_wait_dscnt 0x0
	v_fma_f32 v95, -v172, v170, v95
	ds_bpermute_b32 v170, v155, v171
	s_wait_dscnt 0x0
	v_fma_f32 v96, -v172, v170, v96
	ds_bpermute_b32 v170, v156, v171
	s_wait_dscnt 0x0
	v_fma_f32 v97, -v172, v170, v97
	ds_bpermute_b32 v170, v157, v171
	s_wait_dscnt 0x0
	v_fma_f32 v98, -v172, v170, v98
	ds_bpermute_b32 v170, v158, v171
	s_wait_dscnt 0x0
	v_fma_f32 v99, -v172, v170, v99
	ds_bpermute_b32 v170, v159, v171
	s_wait_dscnt 0x0
	v_fma_f32 v100, -v172, v170, v100
	ds_bpermute_b32 v170, v160, v171
	s_wait_dscnt 0x0
	v_fma_f32 v101, -v172, v170, v101
	ds_bpermute_b32 v170, v161, v171
	s_wait_dscnt 0x0
	v_fma_f32 v102, -v172, v170, v102
	ds_bpermute_b32 v170, v162, v171
	s_wait_dscnt 0x0
	v_fma_f32 v103, -v172, v170, v103
	ds_bpermute_b32 v170, v163, v171
	s_wait_dscnt 0x0
	v_fma_f32 v104, -v172, v170, v104
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v170, 0, v79, s1
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v170, v170, v95, s2
	ds_bpermute_b32 v171, v148, v170
	s_wait_dscnt 0x0
	v_add_f32_e32 v172, v170, v171
	ds_load_2addr_b32 v[170:171], v173 offset0:128 offset1:160
	s_wait_dscnt 0x0
	ds_bpermute_b32 v174, v168, v170
	s_wait_dscnt 0x0
	v_fma_f32 v73, -v172, v174, v73
	ds_bpermute_b32 v174, v149, v170
	s_wait_dscnt 0x0
	v_fma_f32 v74, -v172, v174, v74
	ds_bpermute_b32 v174, v150, v170
	s_wait_dscnt 0x0
	v_fma_f32 v75, -v172, v174, v75
	ds_bpermute_b32 v174, v151, v170
	s_wait_dscnt 0x0
	v_fma_f32 v76, -v172, v174, v76
	ds_bpermute_b32 v174, v152, v170
	s_wait_dscnt 0x0
	v_fma_f32 v77, -v172, v174, v77
	ds_bpermute_b32 v174, v153, v170
	s_wait_dscnt 0x0
	v_fma_f32 v78, -v172, v174, v78
	ds_bpermute_b32 v174, v154, v170
	s_wait_dscnt 0x0
	v_fma_f32 v79, -v172, v174, v79
	ds_bpermute_b32 v174, v155, v170
	s_wait_dscnt 0x0
	v_fma_f32 v80, -v172, v174, v80
	ds_bpermute_b32 v174, v156, v170
	s_wait_dscnt 0x0
	v_fma_f32 v81, -v172, v174, v81
	ds_bpermute_b32 v174, v157, v170
	s_wait_dscnt 0x0
	v_fma_f32 v82, -v172, v174, v82
	ds_bpermute_b32 v174, v158, v170
	s_wait_dscnt 0x0
	v_fma_f32 v83, -v172, v174, v83
	ds_bpermute_b32 v174, v159, v170
	s_wait_dscnt 0x0
	v_fma_f32 v84, -v172, v174, v84
	ds_bpermute_b32 v174, v160, v170
	s_wait_dscnt 0x0
	v_fma_f32 v85, -v172, v174, v85
	ds_bpermute_b32 v174, v161, v170
	s_wait_dscnt 0x0
	v_fma_f32 v86, -v172, v174, v86
	ds_bpermute_b32 v174, v162, v170
	ds_bpermute_b32 v170, v163, v170
	s_wait_dscnt 0x1
	v_fma_f32 v87, -v172, v174, v87
	s_wait_dscnt 0x0
	v_fma_f32 v88, -v172, v170, v88
	ds_bpermute_b32 v170, v168, v171
	s_wait_dscnt 0x0
	v_fma_f32 v89, -v172, v170, v89
	ds_bpermute_b32 v170, v149, v171
	s_wait_dscnt 0x0
	v_fma_f32 v90, -v172, v170, v90
	ds_bpermute_b32 v170, v150, v171
	s_wait_dscnt 0x0
	v_fma_f32 v91, -v172, v170, v91
	ds_bpermute_b32 v170, v151, v171
	s_wait_dscnt 0x0
	v_fma_f32 v92, -v172, v170, v92
	ds_bpermute_b32 v170, v152, v171
	s_wait_dscnt 0x0
	v_fma_f32 v93, -v172, v170, v93
	ds_bpermute_b32 v170, v153, v171
	s_wait_dscnt 0x0
	v_fma_f32 v94, -v172, v170, v94
	ds_bpermute_b32 v170, v154, v171
	s_wait_dscnt 0x0
	v_fma_f32 v95, -v172, v170, v95
	ds_bpermute_b32 v170, v155, v171
	s_wait_dscnt 0x0
	v_fma_f32 v96, -v172, v170, v96
	ds_bpermute_b32 v170, v156, v171
	s_wait_dscnt 0x0
	v_fma_f32 v97, -v172, v170, v97
	ds_bpermute_b32 v170, v157, v171
	s_wait_dscnt 0x0
	v_fma_f32 v98, -v172, v170, v98
	ds_bpermute_b32 v170, v158, v171
	s_wait_dscnt 0x0
	v_fma_f32 v99, -v172, v170, v99
	ds_bpermute_b32 v170, v159, v171
	s_wait_dscnt 0x0
	v_fma_f32 v100, -v172, v170, v100
	ds_bpermute_b32 v170, v160, v171
	s_wait_dscnt 0x0
	v_fma_f32 v101, -v172, v170, v101
	ds_bpermute_b32 v170, v161, v171
	s_wait_dscnt 0x0
	v_fma_f32 v102, -v172, v170, v102
	ds_bpermute_b32 v170, v162, v171
	s_wait_dscnt 0x0
	v_fma_f32 v103, -v172, v170, v103
	ds_bpermute_b32 v170, v163, v171
	s_wait_dscnt 0x0
	v_fma_f32 v104, -v172, v170, v104
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v170, 0, v80, s1
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v170, v170, v96, s2
	ds_bpermute_b32 v171, v148, v170
	s_wait_dscnt 0x0
	v_add_f32_e32 v172, v170, v171
	ds_load_2addr_b32 v[170:171], v173 offset0:192 offset1:224
	s_wait_dscnt 0x0
	ds_bpermute_b32 v173, v168, v170
	s_wait_dscnt 0x0
	v_fma_f32 v73, -v172, v173, v73
	ds_bpermute_b32 v173, v149, v170
	s_wait_dscnt 0x0
	v_fma_f32 v74, -v172, v173, v74
	ds_bpermute_b32 v173, v150, v170
	s_wait_dscnt 0x0
	v_fma_f32 v75, -v172, v173, v75
	ds_bpermute_b32 v173, v151, v170
	s_wait_dscnt 0x0
	v_fma_f32 v76, -v172, v173, v76
	ds_bpermute_b32 v173, v152, v170
	s_wait_dscnt 0x0
	v_fma_f32 v77, -v172, v173, v77
	ds_bpermute_b32 v173, v153, v170
	s_wait_dscnt 0x0
	v_fma_f32 v78, -v172, v173, v78
	ds_bpermute_b32 v173, v154, v170
	s_wait_dscnt 0x0
	v_fma_f32 v79, -v172, v173, v79
	ds_bpermute_b32 v173, v155, v170
	s_wait_dscnt 0x0
	v_fma_f32 v80, -v172, v173, v80
	ds_bpermute_b32 v173, v156, v170
	s_wait_dscnt 0x0
	v_fma_f32 v81, -v172, v173, v81
	ds_bpermute_b32 v173, v157, v170
	s_wait_dscnt 0x0
	v_fma_f32 v82, -v172, v173, v82
	ds_bpermute_b32 v173, v158, v170
	s_wait_dscnt 0x0
	v_fma_f32 v83, -v172, v173, v83
	ds_bpermute_b32 v173, v159, v170
	s_wait_dscnt 0x0
	v_fma_f32 v84, -v172, v173, v84
	ds_bpermute_b32 v173, v160, v170
	s_wait_dscnt 0x0
	v_fma_f32 v85, -v172, v173, v85
	ds_bpermute_b32 v173, v161, v170
	s_wait_dscnt 0x0
	v_fma_f32 v86, -v172, v173, v86
	ds_bpermute_b32 v173, v162, v170
	ds_bpermute_b32 v170, v163, v170
	s_wait_dscnt 0x1
	v_fma_f32 v87, -v172, v173, v87
	s_wait_dscnt 0x0
	v_fma_f32 v88, -v172, v170, v88
	ds_bpermute_b32 v170, v168, v171
	v_add_nc_u32_e32 v173, 0x800, v169
	s_wait_dscnt 0x0
	v_fma_f32 v89, -v172, v170, v89
	ds_bpermute_b32 v170, v149, v171
	s_wait_dscnt 0x0
	v_fma_f32 v90, -v172, v170, v90
	ds_bpermute_b32 v170, v150, v171
	s_wait_dscnt 0x0
	v_fma_f32 v91, -v172, v170, v91
	ds_bpermute_b32 v170, v151, v171
	s_wait_dscnt 0x0
	v_fma_f32 v92, -v172, v170, v92
	ds_bpermute_b32 v170, v152, v171
	s_wait_dscnt 0x0
	v_fma_f32 v93, -v172, v170, v93
	ds_bpermute_b32 v170, v153, v171
	s_wait_dscnt 0x0
	v_fma_f32 v94, -v172, v170, v94
	ds_bpermute_b32 v170, v154, v171
	s_wait_dscnt 0x0
	v_fma_f32 v95, -v172, v170, v95
	ds_bpermute_b32 v170, v155, v171
	s_wait_dscnt 0x0
	v_fma_f32 v96, -v172, v170, v96
	ds_bpermute_b32 v170, v156, v171
	s_wait_dscnt 0x0
	v_fma_f32 v97, -v172, v170, v97
	ds_bpermute_b32 v170, v157, v171
	s_wait_dscnt 0x0
	v_fma_f32 v98, -v172, v170, v98
	ds_bpermute_b32 v170, v158, v171
	s_wait_dscnt 0x0
	v_fma_f32 v99, -v172, v170, v99
	ds_bpermute_b32 v170, v159, v171
	s_wait_dscnt 0x0
	v_fma_f32 v100, -v172, v170, v100
	ds_bpermute_b32 v170, v160, v171
	s_wait_dscnt 0x0
	v_fma_f32 v101, -v172, v170, v101
	ds_bpermute_b32 v170, v161, v171
	s_wait_dscnt 0x0
	v_fma_f32 v102, -v172, v170, v102
	ds_bpermute_b32 v170, v162, v171
	s_wait_dscnt 0x0
	v_fma_f32 v103, -v172, v170, v103
	ds_bpermute_b32 v170, v163, v171
	s_wait_dscnt 0x0
	v_fma_f32 v104, -v172, v170, v104
	;;#ASMSTART
	;;#ASMEND
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v170, 0, v73, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v170, v170, v89, s0
	ds_bpermute_b32 v171, v148, v170
	s_wait_dscnt 0x0
	v_add_f32_e32 v172, v170, v171
	ds_load_2addr_b32 v[170:171], v173 offset1:32
	s_wait_dscnt 0x0
	ds_bpermute_b32 v174, v168, v170
	s_wait_dscnt 0x0
	v_fma_f32 v73, -v172, v174, v73
	ds_bpermute_b32 v174, v149, v170
	s_wait_dscnt 0x0
	v_fma_f32 v74, -v172, v174, v74
	ds_bpermute_b32 v174, v150, v170
	s_wait_dscnt 0x0
	v_fma_f32 v75, -v172, v174, v75
	ds_bpermute_b32 v174, v151, v170
	s_wait_dscnt 0x0
	v_fma_f32 v76, -v172, v174, v76
	ds_bpermute_b32 v174, v152, v170
	s_wait_dscnt 0x0
	v_fma_f32 v77, -v172, v174, v77
	ds_bpermute_b32 v174, v153, v170
	s_wait_dscnt 0x0
	v_fma_f32 v78, -v172, v174, v78
	ds_bpermute_b32 v174, v154, v170
	s_wait_dscnt 0x0
	v_fma_f32 v79, -v172, v174, v79
	ds_bpermute_b32 v174, v155, v170
	s_wait_dscnt 0x0
	v_fma_f32 v80, -v172, v174, v80
	ds_bpermute_b32 v174, v156, v170
	s_wait_dscnt 0x0
	v_fma_f32 v81, -v172, v174, v81
	ds_bpermute_b32 v174, v157, v170
	s_wait_dscnt 0x0
	v_fma_f32 v82, -v172, v174, v82
	ds_bpermute_b32 v174, v158, v170
	s_wait_dscnt 0x0
	v_fma_f32 v83, -v172, v174, v83
	ds_bpermute_b32 v174, v159, v170
	s_wait_dscnt 0x0
	v_fma_f32 v84, -v172, v174, v84
	ds_bpermute_b32 v174, v160, v170
	s_wait_dscnt 0x0
	v_fma_f32 v85, -v172, v174, v85
	ds_bpermute_b32 v174, v161, v170
	s_wait_dscnt 0x0
	v_fma_f32 v86, -v172, v174, v86
	ds_bpermute_b32 v174, v162, v170
	ds_bpermute_b32 v170, v163, v170
	s_wait_dscnt 0x1
	v_fma_f32 v87, -v172, v174, v87
	s_wait_dscnt 0x0
	v_fma_f32 v88, -v172, v170, v88
	ds_bpermute_b32 v170, v168, v171
	s_wait_dscnt 0x0
	v_fma_f32 v89, -v172, v170, v89
	ds_bpermute_b32 v170, v149, v171
	s_wait_dscnt 0x0
	v_fma_f32 v90, -v172, v170, v90
	ds_bpermute_b32 v170, v150, v171
	s_wait_dscnt 0x0
	v_fma_f32 v91, -v172, v170, v91
	ds_bpermute_b32 v170, v151, v171
	s_wait_dscnt 0x0
	v_fma_f32 v92, -v172, v170, v92
	ds_bpermute_b32 v170, v152, v171
	s_wait_dscnt 0x0
	v_fma_f32 v93, -v172, v170, v93
	ds_bpermute_b32 v170, v153, v171
	s_wait_dscnt 0x0
	v_fma_f32 v94, -v172, v170, v94
	ds_bpermute_b32 v170, v154, v171
	s_wait_dscnt 0x0
	v_fma_f32 v95, -v172, v170, v95
	ds_bpermute_b32 v170, v155, v171
	s_wait_dscnt 0x0
	v_fma_f32 v96, -v172, v170, v96
	ds_bpermute_b32 v170, v156, v171
	s_wait_dscnt 0x0
	v_fma_f32 v97, -v172, v170, v97
	ds_bpermute_b32 v170, v157, v171
	s_wait_dscnt 0x0
	v_fma_f32 v98, -v172, v170, v98
	ds_bpermute_b32 v170, v158, v171
	s_wait_dscnt 0x0
	v_fma_f32 v99, -v172, v170, v99
	ds_bpermute_b32 v170, v159, v171
	s_wait_dscnt 0x0
	v_fma_f32 v100, -v172, v170, v100
	ds_bpermute_b32 v170, v160, v171
	s_wait_dscnt 0x0
	v_fma_f32 v101, -v172, v170, v101
	ds_bpermute_b32 v170, v161, v171
	s_wait_dscnt 0x0
	v_fma_f32 v102, -v172, v170, v102
	ds_bpermute_b32 v170, v162, v171
	s_wait_dscnt 0x0
	v_fma_f32 v103, -v172, v170, v103
	ds_bpermute_b32 v170, v163, v171
	s_wait_dscnt 0x0
	v_fma_f32 v104, -v172, v170, v104
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e32 v170, 0, v74, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v170, v170, v90, s0
	ds_bpermute_b32 v171, v148, v170
	s_wait_dscnt 0x0
	v_add_f32_e32 v172, v170, v171
	ds_load_2addr_b32 v[170:171], v173 offset0:64 offset1:96
	s_wait_dscnt 0x0
	ds_bpermute_b32 v174, v168, v170
	s_wait_dscnt 0x0
	v_fma_f32 v73, -v172, v174, v73
	ds_bpermute_b32 v174, v149, v170
	s_wait_dscnt 0x0
	v_fma_f32 v74, -v172, v174, v74
	ds_bpermute_b32 v174, v150, v170
	s_wait_dscnt 0x0
	v_fma_f32 v75, -v172, v174, v75
	ds_bpermute_b32 v174, v151, v170
	s_wait_dscnt 0x0
	v_fma_f32 v76, -v172, v174, v76
	ds_bpermute_b32 v174, v152, v170
	s_wait_dscnt 0x0
	v_fma_f32 v77, -v172, v174, v77
	ds_bpermute_b32 v174, v153, v170
	s_wait_dscnt 0x0
	v_fma_f32 v78, -v172, v174, v78
	ds_bpermute_b32 v174, v154, v170
	s_wait_dscnt 0x0
	v_fma_f32 v79, -v172, v174, v79
	ds_bpermute_b32 v174, v155, v170
	s_wait_dscnt 0x0
	v_fma_f32 v80, -v172, v174, v80
	ds_bpermute_b32 v174, v156, v170
	s_wait_dscnt 0x0
	v_fma_f32 v81, -v172, v174, v81
	ds_bpermute_b32 v174, v157, v170
	s_wait_dscnt 0x0
	v_fma_f32 v82, -v172, v174, v82
	ds_bpermute_b32 v174, v158, v170
	s_wait_dscnt 0x0
	v_fma_f32 v83, -v172, v174, v83
	ds_bpermute_b32 v174, v159, v170
	s_wait_dscnt 0x0
	v_fma_f32 v84, -v172, v174, v84
	ds_bpermute_b32 v174, v160, v170
	s_wait_dscnt 0x0
	v_fma_f32 v85, -v172, v174, v85
	ds_bpermute_b32 v174, v161, v170
	s_wait_dscnt 0x0
	v_fma_f32 v86, -v172, v174, v86
	ds_bpermute_b32 v174, v162, v170
	ds_bpermute_b32 v170, v163, v170
	s_wait_dscnt 0x1
	v_fma_f32 v87, -v172, v174, v87
	s_wait_dscnt 0x0
	v_fma_f32 v88, -v172, v170, v88
	ds_bpermute_b32 v170, v168, v171
	s_wait_dscnt 0x0
	v_fma_f32 v89, -v172, v170, v89
	ds_bpermute_b32 v170, v149, v171
	s_wait_dscnt 0x0
	v_fma_f32 v90, -v172, v170, v90
	ds_bpermute_b32 v170, v150, v171
	s_wait_dscnt 0x0
	v_fma_f32 v91, -v172, v170, v91
	ds_bpermute_b32 v170, v151, v171
	s_wait_dscnt 0x0
	v_fma_f32 v92, -v172, v170, v92
	ds_bpermute_b32 v170, v152, v171
	s_wait_dscnt 0x0
	v_fma_f32 v93, -v172, v170, v93
	ds_bpermute_b32 v170, v153, v171
	s_wait_dscnt 0x0
	v_fma_f32 v94, -v172, v170, v94
	ds_bpermute_b32 v170, v154, v171
	s_wait_dscnt 0x0
	v_fma_f32 v95, -v172, v170, v95
	ds_bpermute_b32 v170, v155, v171
	s_wait_dscnt 0x0
	v_fma_f32 v96, -v172, v170, v96
	ds_bpermute_b32 v170, v156, v171
	s_wait_dscnt 0x0
	v_fma_f32 v97, -v172, v170, v97
	ds_bpermute_b32 v170, v157, v171
	s_wait_dscnt 0x0
	v_fma_f32 v98, -v172, v170, v98
	ds_bpermute_b32 v170, v158, v171
	s_wait_dscnt 0x0
	v_fma_f32 v99, -v172, v170, v99
	ds_bpermute_b32 v170, v159, v171
	s_wait_dscnt 0x0
	v_fma_f32 v100, -v172, v170, v100
	ds_bpermute_b32 v170, v160, v171
	s_wait_dscnt 0x0
	v_fma_f32 v101, -v172, v170, v101
	ds_bpermute_b32 v170, v161, v171
	s_wait_dscnt 0x0
	v_fma_f32 v102, -v172, v170, v102
	ds_bpermute_b32 v170, v162, v171
	s_wait_dscnt 0x0
	v_fma_f32 v103, -v172, v170, v103
	ds_bpermute_b32 v170, v163, v171
	s_wait_dscnt 0x0
	v_fma_f32 v104, -v172, v170, v104
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e32 v170, 0, v75, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v170, v170, v91, s0
	ds_bpermute_b32 v171, v148, v170
	s_wait_dscnt 0x0
	v_add_f32_e32 v172, v170, v171
	ds_load_2addr_b32 v[170:171], v173 offset0:128 offset1:160
	s_wait_dscnt 0x0
	ds_bpermute_b32 v174, v168, v170
	s_wait_dscnt 0x0
	v_fma_f32 v73, -v172, v174, v73
	ds_bpermute_b32 v174, v149, v170
	s_wait_dscnt 0x0
	v_fma_f32 v74, -v172, v174, v74
	ds_bpermute_b32 v174, v150, v170
	s_wait_dscnt 0x0
	v_fma_f32 v75, -v172, v174, v75
	ds_bpermute_b32 v174, v151, v170
	s_wait_dscnt 0x0
	v_fma_f32 v76, -v172, v174, v76
	ds_bpermute_b32 v174, v152, v170
	s_wait_dscnt 0x0
	v_fma_f32 v77, -v172, v174, v77
	ds_bpermute_b32 v174, v153, v170
	s_wait_dscnt 0x0
	v_fma_f32 v78, -v172, v174, v78
	ds_bpermute_b32 v174, v154, v170
	s_wait_dscnt 0x0
	v_fma_f32 v79, -v172, v174, v79
	ds_bpermute_b32 v174, v155, v170
	s_wait_dscnt 0x0
	v_fma_f32 v80, -v172, v174, v80
	ds_bpermute_b32 v174, v156, v170
	s_wait_dscnt 0x0
	v_fma_f32 v81, -v172, v174, v81
	ds_bpermute_b32 v174, v157, v170
	s_wait_dscnt 0x0
	v_fma_f32 v82, -v172, v174, v82
	ds_bpermute_b32 v174, v158, v170
	s_wait_dscnt 0x0
	v_fma_f32 v83, -v172, v174, v83
	ds_bpermute_b32 v174, v159, v170
	s_wait_dscnt 0x0
	v_fma_f32 v84, -v172, v174, v84
	ds_bpermute_b32 v174, v160, v170
	s_wait_dscnt 0x0
	v_fma_f32 v85, -v172, v174, v85
	ds_bpermute_b32 v174, v161, v170
	s_wait_dscnt 0x0
	v_fma_f32 v86, -v172, v174, v86
	ds_bpermute_b32 v174, v162, v170
	ds_bpermute_b32 v170, v163, v170
	s_wait_dscnt 0x1
	v_fma_f32 v87, -v172, v174, v87
	s_wait_dscnt 0x0
	v_fma_f32 v88, -v172, v170, v88
	ds_bpermute_b32 v170, v168, v171
	s_wait_dscnt 0x0
	v_fma_f32 v89, -v172, v170, v89
	ds_bpermute_b32 v170, v149, v171
	s_wait_dscnt 0x0
	v_fma_f32 v90, -v172, v170, v90
	ds_bpermute_b32 v170, v150, v171
	s_wait_dscnt 0x0
	v_fma_f32 v91, -v172, v170, v91
	ds_bpermute_b32 v170, v151, v171
	s_wait_dscnt 0x0
	v_fma_f32 v92, -v172, v170, v92
	ds_bpermute_b32 v170, v152, v171
	s_wait_dscnt 0x0
	v_fma_f32 v93, -v172, v170, v93
	ds_bpermute_b32 v170, v153, v171
	s_wait_dscnt 0x0
	v_fma_f32 v94, -v172, v170, v94
	ds_bpermute_b32 v170, v154, v171
	s_wait_dscnt 0x0
	v_fma_f32 v95, -v172, v170, v95
	ds_bpermute_b32 v170, v155, v171
	s_wait_dscnt 0x0
	v_fma_f32 v96, -v172, v170, v96
	ds_bpermute_b32 v170, v156, v171
	s_wait_dscnt 0x0
	v_fma_f32 v97, -v172, v170, v97
	ds_bpermute_b32 v170, v157, v171
	s_wait_dscnt 0x0
	v_fma_f32 v98, -v172, v170, v98
	ds_bpermute_b32 v170, v158, v171
	s_wait_dscnt 0x0
	v_fma_f32 v99, -v172, v170, v99
	ds_bpermute_b32 v170, v159, v171
	s_wait_dscnt 0x0
	v_fma_f32 v100, -v172, v170, v100
	ds_bpermute_b32 v170, v160, v171
	s_wait_dscnt 0x0
	v_fma_f32 v101, -v172, v170, v101
	ds_bpermute_b32 v170, v161, v171
	s_wait_dscnt 0x0
	v_fma_f32 v102, -v172, v170, v102
	ds_bpermute_b32 v170, v162, v171
	s_wait_dscnt 0x0
	v_fma_f32 v103, -v172, v170, v103
	ds_bpermute_b32 v170, v163, v171
	s_wait_dscnt 0x0
	v_fma_f32 v104, -v172, v170, v104
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e32 v170, 0, v76, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v170, v170, v92, s0
	ds_bpermute_b32 v171, v148, v170
	s_wait_dscnt 0x0
	v_add_f32_e32 v172, v170, v171
	ds_load_2addr_b32 v[170:171], v173 offset0:192 offset1:224
	s_wait_dscnt 0x0
	ds_bpermute_b32 v173, v168, v170
	s_wait_dscnt 0x0
	v_fma_f32 v73, -v172, v173, v73
	ds_bpermute_b32 v173, v149, v170
	s_wait_dscnt 0x0
	v_fma_f32 v74, -v172, v173, v74
	ds_bpermute_b32 v173, v150, v170
	s_wait_dscnt 0x0
	v_fma_f32 v75, -v172, v173, v75
	ds_bpermute_b32 v173, v151, v170
	s_wait_dscnt 0x0
	v_fma_f32 v76, -v172, v173, v76
	ds_bpermute_b32 v173, v152, v170
	s_wait_dscnt 0x0
	v_fma_f32 v77, -v172, v173, v77
	ds_bpermute_b32 v173, v153, v170
	s_wait_dscnt 0x0
	v_fma_f32 v78, -v172, v173, v78
	ds_bpermute_b32 v173, v154, v170
	s_wait_dscnt 0x0
	v_fma_f32 v79, -v172, v173, v79
	ds_bpermute_b32 v173, v155, v170
	s_wait_dscnt 0x0
	v_fma_f32 v80, -v172, v173, v80
	ds_bpermute_b32 v173, v156, v170
	s_wait_dscnt 0x0
	v_fma_f32 v81, -v172, v173, v81
	ds_bpermute_b32 v173, v157, v170
	s_wait_dscnt 0x0
	v_fma_f32 v82, -v172, v173, v82
	ds_bpermute_b32 v173, v158, v170
	s_wait_dscnt 0x0
	v_fma_f32 v83, -v172, v173, v83
	ds_bpermute_b32 v173, v159, v170
	s_wait_dscnt 0x0
	v_fma_f32 v84, -v172, v173, v84
	ds_bpermute_b32 v173, v160, v170
	s_wait_dscnt 0x0
	v_fma_f32 v85, -v172, v173, v85
	ds_bpermute_b32 v173, v161, v170
	s_wait_dscnt 0x0
	v_fma_f32 v86, -v172, v173, v86
	ds_bpermute_b32 v173, v162, v170
	ds_bpermute_b32 v170, v163, v170
	s_wait_dscnt 0x1
	v_fma_f32 v87, -v172, v173, v87
	s_wait_dscnt 0x0
	v_fma_f32 v88, -v172, v170, v88
	ds_bpermute_b32 v170, v168, v171
	v_add_nc_u32_e32 v173, 0xc00, v169
	s_wait_dscnt 0x0
	v_fma_f32 v89, -v172, v170, v89
	ds_bpermute_b32 v170, v149, v171
	s_wait_dscnt 0x0
	v_fma_f32 v90, -v172, v170, v90
	ds_bpermute_b32 v170, v150, v171
	s_wait_dscnt 0x0
	v_fma_f32 v91, -v172, v170, v91
	ds_bpermute_b32 v170, v151, v171
	s_wait_dscnt 0x0
	v_fma_f32 v92, -v172, v170, v92
	ds_bpermute_b32 v170, v152, v171
	s_wait_dscnt 0x0
	v_fma_f32 v93, -v172, v170, v93
	ds_bpermute_b32 v170, v153, v171
	s_wait_dscnt 0x0
	v_fma_f32 v94, -v172, v170, v94
	ds_bpermute_b32 v170, v154, v171
	s_wait_dscnt 0x0
	v_fma_f32 v95, -v172, v170, v95
	ds_bpermute_b32 v170, v155, v171
	s_wait_dscnt 0x0
	v_fma_f32 v96, -v172, v170, v96
	ds_bpermute_b32 v170, v156, v171
	s_wait_dscnt 0x0
	v_fma_f32 v97, -v172, v170, v97
	ds_bpermute_b32 v170, v157, v171
	s_wait_dscnt 0x0
	v_fma_f32 v98, -v172, v170, v98
	ds_bpermute_b32 v170, v158, v171
	s_wait_dscnt 0x0
	v_fma_f32 v99, -v172, v170, v99
	ds_bpermute_b32 v170, v159, v171
	s_wait_dscnt 0x0
	v_fma_f32 v100, -v172, v170, v100
	ds_bpermute_b32 v170, v160, v171
	s_wait_dscnt 0x0
	v_fma_f32 v101, -v172, v170, v101
	ds_bpermute_b32 v170, v161, v171
	s_wait_dscnt 0x0
	v_fma_f32 v102, -v172, v170, v102
	ds_bpermute_b32 v170, v162, v171
	s_wait_dscnt 0x0
	v_fma_f32 v103, -v172, v170, v103
	ds_bpermute_b32 v170, v163, v171
	s_wait_dscnt 0x0
	v_fma_f32 v104, -v172, v170, v104
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e32 v170, 0, v77, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v170, v170, v93, s0
	ds_bpermute_b32 v171, v148, v170
	s_wait_dscnt 0x0
	v_add_f32_e32 v172, v170, v171
	ds_load_2addr_b32 v[170:171], v173 offset1:32
	s_wait_dscnt 0x0
	ds_bpermute_b32 v174, v168, v170
	s_wait_dscnt 0x0
	v_fma_f32 v73, -v172, v174, v73
	ds_bpermute_b32 v174, v149, v170
	s_wait_dscnt 0x0
	v_fma_f32 v74, -v172, v174, v74
	ds_bpermute_b32 v174, v150, v170
	s_wait_dscnt 0x0
	v_fma_f32 v75, -v172, v174, v75
	ds_bpermute_b32 v174, v151, v170
	s_wait_dscnt 0x0
	v_fma_f32 v76, -v172, v174, v76
	ds_bpermute_b32 v174, v152, v170
	s_wait_dscnt 0x0
	v_fma_f32 v77, -v172, v174, v77
	ds_bpermute_b32 v174, v153, v170
	s_wait_dscnt 0x0
	v_fma_f32 v78, -v172, v174, v78
	ds_bpermute_b32 v174, v154, v170
	s_wait_dscnt 0x0
	v_fma_f32 v79, -v172, v174, v79
	ds_bpermute_b32 v174, v155, v170
	s_wait_dscnt 0x0
	v_fma_f32 v80, -v172, v174, v80
	ds_bpermute_b32 v174, v156, v170
	s_wait_dscnt 0x0
	v_fma_f32 v81, -v172, v174, v81
	ds_bpermute_b32 v174, v157, v170
	s_wait_dscnt 0x0
	v_fma_f32 v82, -v172, v174, v82
	ds_bpermute_b32 v174, v158, v170
	s_wait_dscnt 0x0
	v_fma_f32 v83, -v172, v174, v83
	ds_bpermute_b32 v174, v159, v170
	s_wait_dscnt 0x0
	v_fma_f32 v84, -v172, v174, v84
	ds_bpermute_b32 v174, v160, v170
	s_wait_dscnt 0x0
	v_fma_f32 v85, -v172, v174, v85
	ds_bpermute_b32 v174, v161, v170
	s_wait_dscnt 0x0
	v_fma_f32 v86, -v172, v174, v86
	ds_bpermute_b32 v174, v162, v170
	ds_bpermute_b32 v170, v163, v170
	s_wait_dscnt 0x1
	v_fma_f32 v87, -v172, v174, v87
	s_wait_dscnt 0x0
	v_fma_f32 v88, -v172, v170, v88
	ds_bpermute_b32 v170, v168, v171
	s_wait_dscnt 0x0
	v_fma_f32 v89, -v172, v170, v89
	ds_bpermute_b32 v170, v149, v171
	s_wait_dscnt 0x0
	v_fma_f32 v90, -v172, v170, v90
	ds_bpermute_b32 v170, v150, v171
	s_wait_dscnt 0x0
	v_fma_f32 v91, -v172, v170, v91
	ds_bpermute_b32 v170, v151, v171
	s_wait_dscnt 0x0
	v_fma_f32 v92, -v172, v170, v92
	ds_bpermute_b32 v170, v152, v171
	s_wait_dscnt 0x0
	v_fma_f32 v93, -v172, v170, v93
	ds_bpermute_b32 v170, v153, v171
	s_wait_dscnt 0x0
	v_fma_f32 v94, -v172, v170, v94
	ds_bpermute_b32 v170, v154, v171
	s_wait_dscnt 0x0
	v_fma_f32 v95, -v172, v170, v95
	ds_bpermute_b32 v170, v155, v171
	s_wait_dscnt 0x0
	v_fma_f32 v96, -v172, v170, v96
	ds_bpermute_b32 v170, v156, v171
	s_wait_dscnt 0x0
	v_fma_f32 v97, -v172, v170, v97
	ds_bpermute_b32 v170, v157, v171
	s_wait_dscnt 0x0
	v_fma_f32 v98, -v172, v170, v98
	ds_bpermute_b32 v170, v158, v171
	s_wait_dscnt 0x0
	v_fma_f32 v99, -v172, v170, v99
	ds_bpermute_b32 v170, v159, v171
	s_wait_dscnt 0x0
	v_fma_f32 v100, -v172, v170, v100
	ds_bpermute_b32 v170, v160, v171
	s_wait_dscnt 0x0
	v_fma_f32 v101, -v172, v170, v101
	ds_bpermute_b32 v170, v161, v171
	s_wait_dscnt 0x0
	v_fma_f32 v102, -v172, v170, v102
	ds_bpermute_b32 v170, v162, v171
	s_wait_dscnt 0x0
	v_fma_f32 v103, -v172, v170, v103
	ds_bpermute_b32 v170, v163, v171
	s_wait_dscnt 0x0
	v_fma_f32 v104, -v172, v170, v104
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e32 v170, 0, v78, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v170, v170, v94, s0
	ds_bpermute_b32 v171, v148, v170
	s_wait_dscnt 0x0
	v_add_f32_e32 v172, v170, v171
	ds_load_2addr_b32 v[170:171], v173 offset0:64 offset1:96
	s_wait_dscnt 0x0
	ds_bpermute_b32 v174, v168, v170
	s_wait_dscnt 0x0
	v_fma_f32 v73, -v172, v174, v73
	ds_bpermute_b32 v174, v149, v170
	s_wait_dscnt 0x0
	v_fma_f32 v74, -v172, v174, v74
	ds_bpermute_b32 v174, v150, v170
	s_wait_dscnt 0x0
	v_fma_f32 v75, -v172, v174, v75
	ds_bpermute_b32 v174, v151, v170
	s_wait_dscnt 0x0
	v_fma_f32 v76, -v172, v174, v76
	ds_bpermute_b32 v174, v152, v170
	s_wait_dscnt 0x0
	v_fma_f32 v77, -v172, v174, v77
	ds_bpermute_b32 v174, v153, v170
	s_wait_dscnt 0x0
	v_fma_f32 v78, -v172, v174, v78
	ds_bpermute_b32 v174, v154, v170
	s_wait_dscnt 0x0
	v_fma_f32 v79, -v172, v174, v79
	ds_bpermute_b32 v174, v155, v170
	s_wait_dscnt 0x0
	v_fma_f32 v80, -v172, v174, v80
	ds_bpermute_b32 v174, v156, v170
	s_wait_dscnt 0x0
	v_fma_f32 v81, -v172, v174, v81
	ds_bpermute_b32 v174, v157, v170
	s_wait_dscnt 0x0
	v_fma_f32 v82, -v172, v174, v82
	ds_bpermute_b32 v174, v158, v170
	s_wait_dscnt 0x0
	v_fma_f32 v83, -v172, v174, v83
	ds_bpermute_b32 v174, v159, v170
	s_wait_dscnt 0x0
	v_fma_f32 v84, -v172, v174, v84
	ds_bpermute_b32 v174, v160, v170
	s_wait_dscnt 0x0
	v_fma_f32 v85, -v172, v174, v85
	ds_bpermute_b32 v174, v161, v170
	s_wait_dscnt 0x0
	v_fma_f32 v86, -v172, v174, v86
	ds_bpermute_b32 v174, v162, v170
	ds_bpermute_b32 v170, v163, v170
	s_wait_dscnt 0x1
	v_fma_f32 v87, -v172, v174, v87
	s_wait_dscnt 0x0
	v_fma_f32 v88, -v172, v170, v88
	ds_bpermute_b32 v170, v168, v171
	s_wait_dscnt 0x0
	v_fma_f32 v89, -v172, v170, v89
	ds_bpermute_b32 v170, v149, v171
	s_wait_dscnt 0x0
	v_fma_f32 v90, -v172, v170, v90
	ds_bpermute_b32 v170, v150, v171
	s_wait_dscnt 0x0
	v_fma_f32 v91, -v172, v170, v91
	ds_bpermute_b32 v170, v151, v171
	s_wait_dscnt 0x0
	v_fma_f32 v92, -v172, v170, v92
	ds_bpermute_b32 v170, v152, v171
	s_wait_dscnt 0x0
	v_fma_f32 v93, -v172, v170, v93
	ds_bpermute_b32 v170, v153, v171
	s_wait_dscnt 0x0
	v_fma_f32 v94, -v172, v170, v94
	ds_bpermute_b32 v170, v154, v171
	s_wait_dscnt 0x0
	v_fma_f32 v95, -v172, v170, v95
	ds_bpermute_b32 v170, v155, v171
	s_wait_dscnt 0x0
	v_fma_f32 v96, -v172, v170, v96
	ds_bpermute_b32 v170, v156, v171
	s_wait_dscnt 0x0
	v_fma_f32 v97, -v172, v170, v97
	ds_bpermute_b32 v170, v157, v171
	s_wait_dscnt 0x0
	v_fma_f32 v98, -v172, v170, v98
	ds_bpermute_b32 v170, v158, v171
	s_wait_dscnt 0x0
	v_fma_f32 v99, -v172, v170, v99
	ds_bpermute_b32 v170, v159, v171
	s_wait_dscnt 0x0
	v_fma_f32 v100, -v172, v170, v100
	ds_bpermute_b32 v170, v160, v171
	s_wait_dscnt 0x0
	v_fma_f32 v101, -v172, v170, v101
	ds_bpermute_b32 v170, v161, v171
	s_wait_dscnt 0x0
	v_fma_f32 v102, -v172, v170, v102
	ds_bpermute_b32 v170, v162, v171
	s_wait_dscnt 0x0
	v_fma_f32 v103, -v172, v170, v103
	ds_bpermute_b32 v170, v163, v171
	s_wait_dscnt 0x0
	v_fma_f32 v104, -v172, v170, v104
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e32 v170, 0, v79, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v170, v170, v95, s0
	ds_bpermute_b32 v171, v148, v170
	s_wait_dscnt 0x0
	v_add_f32_e32 v172, v170, v171
	ds_load_2addr_b32 v[170:171], v173 offset0:128 offset1:160
	s_wait_dscnt 0x0
	ds_bpermute_b32 v174, v168, v170
	s_wait_dscnt 0x0
	v_fma_f32 v73, -v172, v174, v73
	ds_bpermute_b32 v174, v149, v170
	s_wait_dscnt 0x0
	v_fma_f32 v74, -v172, v174, v74
	ds_bpermute_b32 v174, v150, v170
	s_wait_dscnt 0x0
	v_fma_f32 v75, -v172, v174, v75
	ds_bpermute_b32 v174, v151, v170
	s_wait_dscnt 0x0
	v_fma_f32 v76, -v172, v174, v76
	ds_bpermute_b32 v174, v152, v170
	s_wait_dscnt 0x0
	v_fma_f32 v77, -v172, v174, v77
	ds_bpermute_b32 v174, v153, v170
	s_wait_dscnt 0x0
	v_fma_f32 v78, -v172, v174, v78
	ds_bpermute_b32 v174, v154, v170
	s_wait_dscnt 0x0
	v_fma_f32 v79, -v172, v174, v79
	ds_bpermute_b32 v174, v155, v170
	s_wait_dscnt 0x0
	v_fma_f32 v80, -v172, v174, v80
	ds_bpermute_b32 v174, v156, v170
	s_wait_dscnt 0x0
	v_fma_f32 v81, -v172, v174, v81
	ds_bpermute_b32 v174, v157, v170
	s_wait_dscnt 0x0
	v_fma_f32 v82, -v172, v174, v82
	ds_bpermute_b32 v174, v158, v170
	s_wait_dscnt 0x0
	v_fma_f32 v83, -v172, v174, v83
	ds_bpermute_b32 v174, v159, v170
	s_wait_dscnt 0x0
	v_fma_f32 v84, -v172, v174, v84
	ds_bpermute_b32 v174, v160, v170
	s_wait_dscnt 0x0
	v_fma_f32 v85, -v172, v174, v85
	ds_bpermute_b32 v174, v161, v170
	s_wait_dscnt 0x0
	v_fma_f32 v86, -v172, v174, v86
	ds_bpermute_b32 v174, v162, v170
	ds_bpermute_b32 v170, v163, v170
	s_wait_dscnt 0x1
	v_fma_f32 v87, -v172, v174, v87
	s_wait_dscnt 0x0
	v_fma_f32 v88, -v172, v170, v88
	ds_bpermute_b32 v170, v168, v171
	s_wait_dscnt 0x0
	v_fma_f32 v89, -v172, v170, v89
	ds_bpermute_b32 v170, v149, v171
	s_wait_dscnt 0x0
	v_fma_f32 v90, -v172, v170, v90
	ds_bpermute_b32 v170, v150, v171
	s_wait_dscnt 0x0
	v_fma_f32 v91, -v172, v170, v91
	ds_bpermute_b32 v170, v151, v171
	s_wait_dscnt 0x0
	v_fma_f32 v92, -v172, v170, v92
	ds_bpermute_b32 v170, v152, v171
	s_wait_dscnt 0x0
	v_fma_f32 v93, -v172, v170, v93
	ds_bpermute_b32 v170, v153, v171
	s_wait_dscnt 0x0
	v_fma_f32 v94, -v172, v170, v94
	ds_bpermute_b32 v170, v154, v171
	s_wait_dscnt 0x0
	v_fma_f32 v95, -v172, v170, v95
	ds_bpermute_b32 v170, v155, v171
	s_wait_dscnt 0x0
	v_fma_f32 v96, -v172, v170, v96
	ds_bpermute_b32 v170, v156, v171
	s_wait_dscnt 0x0
	v_fma_f32 v97, -v172, v170, v97
	ds_bpermute_b32 v170, v157, v171
	s_wait_dscnt 0x0
	v_fma_f32 v98, -v172, v170, v98
	ds_bpermute_b32 v170, v158, v171
	s_wait_dscnt 0x0
	v_fma_f32 v99, -v172, v170, v99
	ds_bpermute_b32 v170, v159, v171
	s_wait_dscnt 0x0
	v_fma_f32 v100, -v172, v170, v100
	ds_bpermute_b32 v170, v160, v171
	s_wait_dscnt 0x0
	v_fma_f32 v101, -v172, v170, v101
	ds_bpermute_b32 v170, v161, v171
	s_wait_dscnt 0x0
	v_fma_f32 v102, -v172, v170, v102
	ds_bpermute_b32 v170, v162, v171
	s_wait_dscnt 0x0
	v_fma_f32 v103, -v172, v170, v103
	ds_bpermute_b32 v170, v163, v171
	s_wait_dscnt 0x0
	v_fma_f32 v104, -v172, v170, v104
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e32 v170, 0, v80, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v170, v170, v96, s0
	ds_bpermute_b32 v171, v148, v170
	s_wait_dscnt 0x0
	v_add_f32_e32 v172, v170, v171
	ds_load_2addr_b32 v[170:171], v173 offset0:192 offset1:224
	s_wait_dscnt 0x0
	ds_bpermute_b32 v173, v168, v170
	s_wait_dscnt 0x0
	v_fma_f32 v73, -v172, v173, v73
	ds_bpermute_b32 v173, v149, v170
	s_wait_dscnt 0x0
	v_fma_f32 v74, -v172, v173, v74
	ds_bpermute_b32 v173, v150, v170
	s_wait_dscnt 0x0
	v_fma_f32 v75, -v172, v173, v75
	ds_bpermute_b32 v173, v151, v170
	s_wait_dscnt 0x0
	v_fma_f32 v76, -v172, v173, v76
	ds_bpermute_b32 v173, v152, v170
	s_wait_dscnt 0x0
	v_fma_f32 v77, -v172, v173, v77
	ds_bpermute_b32 v173, v153, v170
	s_wait_dscnt 0x0
	v_fma_f32 v78, -v172, v173, v78
	ds_bpermute_b32 v173, v154, v170
	s_wait_dscnt 0x0
	v_fma_f32 v79, -v172, v173, v79
	ds_bpermute_b32 v173, v155, v170
	s_wait_dscnt 0x0
	v_fma_f32 v80, -v172, v173, v80
	ds_bpermute_b32 v173, v156, v170
	s_wait_dscnt 0x0
	v_fma_f32 v81, -v172, v173, v81
	ds_bpermute_b32 v173, v157, v170
	s_wait_dscnt 0x0
	v_fma_f32 v82, -v172, v173, v82
	ds_bpermute_b32 v173, v158, v170
	s_wait_dscnt 0x0
	v_fma_f32 v83, -v172, v173, v83
	ds_bpermute_b32 v173, v159, v170
	s_wait_dscnt 0x0
	v_fma_f32 v84, -v172, v173, v84
	ds_bpermute_b32 v173, v160, v170
	s_wait_dscnt 0x0
	v_fma_f32 v85, -v172, v173, v85
	ds_bpermute_b32 v173, v161, v170
	s_wait_dscnt 0x0
	v_fma_f32 v86, -v172, v173, v86
	ds_bpermute_b32 v173, v162, v170
	ds_bpermute_b32 v170, v163, v170
	s_wait_dscnt 0x1
	v_fma_f32 v87, -v172, v173, v87
	s_wait_dscnt 0x0
	v_fma_f32 v88, -v172, v170, v88
	ds_bpermute_b32 v170, v168, v171
	v_add_nc_u32_e32 v173, 0x1000, v169
	s_wait_dscnt 0x0
	v_fma_f32 v89, -v172, v170, v89
	ds_bpermute_b32 v170, v149, v171
	s_wait_dscnt 0x0
	v_fma_f32 v90, -v172, v170, v90
	ds_bpermute_b32 v170, v150, v171
	s_wait_dscnt 0x0
	v_fma_f32 v91, -v172, v170, v91
	ds_bpermute_b32 v170, v151, v171
	s_wait_dscnt 0x0
	v_fma_f32 v92, -v172, v170, v92
	ds_bpermute_b32 v170, v152, v171
	s_wait_dscnt 0x0
	v_fma_f32 v93, -v172, v170, v93
	ds_bpermute_b32 v170, v153, v171
	s_wait_dscnt 0x0
	v_fma_f32 v94, -v172, v170, v94
	ds_bpermute_b32 v170, v154, v171
	s_wait_dscnt 0x0
	v_fma_f32 v95, -v172, v170, v95
	ds_bpermute_b32 v170, v155, v171
	s_wait_dscnt 0x0
	v_fma_f32 v96, -v172, v170, v96
	ds_bpermute_b32 v170, v156, v171
	s_wait_dscnt 0x0
	v_fma_f32 v97, -v172, v170, v97
	ds_bpermute_b32 v170, v157, v171
	s_wait_dscnt 0x0
	v_fma_f32 v98, -v172, v170, v98
	ds_bpermute_b32 v170, v158, v171
	s_wait_dscnt 0x0
	v_fma_f32 v99, -v172, v170, v99
	ds_bpermute_b32 v170, v159, v171
	s_wait_dscnt 0x0
	v_fma_f32 v100, -v172, v170, v100
	ds_bpermute_b32 v170, v160, v171
	s_wait_dscnt 0x0
	v_fma_f32 v101, -v172, v170, v101
	ds_bpermute_b32 v170, v161, v171
	s_wait_dscnt 0x0
	v_fma_f32 v102, -v172, v170, v102
	ds_bpermute_b32 v170, v162, v171
	s_wait_dscnt 0x0
	v_fma_f32 v103, -v172, v170, v103
	ds_bpermute_b32 v170, v163, v171
	s_wait_dscnt 0x0
	v_fma_f32 v104, -v172, v170, v104
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v170, 0, v81, s1
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v170, v170, v97, s2
	ds_bpermute_b32 v171, v148, v170
	s_wait_dscnt 0x0
	v_add_f32_e32 v172, v170, v171
	ds_load_2addr_b32 v[170:171], v173 offset1:32
	s_wait_dscnt 0x0
	ds_bpermute_b32 v174, v168, v170
	s_wait_dscnt 0x0
	v_fma_f32 v73, -v172, v174, v73
	ds_bpermute_b32 v174, v149, v170
	s_wait_dscnt 0x0
	v_fma_f32 v74, -v172, v174, v74
	ds_bpermute_b32 v174, v150, v170
	s_wait_dscnt 0x0
	v_fma_f32 v75, -v172, v174, v75
	ds_bpermute_b32 v174, v151, v170
	s_wait_dscnt 0x0
	v_fma_f32 v76, -v172, v174, v76
	ds_bpermute_b32 v174, v152, v170
	s_wait_dscnt 0x0
	v_fma_f32 v77, -v172, v174, v77
	ds_bpermute_b32 v174, v153, v170
	s_wait_dscnt 0x0
	v_fma_f32 v78, -v172, v174, v78
	ds_bpermute_b32 v174, v154, v170
	s_wait_dscnt 0x0
	v_fma_f32 v79, -v172, v174, v79
	ds_bpermute_b32 v174, v155, v170
	s_wait_dscnt 0x0
	v_fma_f32 v80, -v172, v174, v80
	ds_bpermute_b32 v174, v156, v170
	s_wait_dscnt 0x0
	v_fma_f32 v81, -v172, v174, v81
	ds_bpermute_b32 v174, v157, v170
	s_wait_dscnt 0x0
	v_fma_f32 v82, -v172, v174, v82
	ds_bpermute_b32 v174, v158, v170
	s_wait_dscnt 0x0
	v_fma_f32 v83, -v172, v174, v83
	ds_bpermute_b32 v174, v159, v170
	s_wait_dscnt 0x0
	v_fma_f32 v84, -v172, v174, v84
	ds_bpermute_b32 v174, v160, v170
	s_wait_dscnt 0x0
	v_fma_f32 v85, -v172, v174, v85
	ds_bpermute_b32 v174, v161, v170
	s_wait_dscnt 0x0
	v_fma_f32 v86, -v172, v174, v86
	ds_bpermute_b32 v174, v162, v170
	ds_bpermute_b32 v170, v163, v170
	s_wait_dscnt 0x1
	v_fma_f32 v87, -v172, v174, v87
	s_wait_dscnt 0x0
	v_fma_f32 v88, -v172, v170, v88
	ds_bpermute_b32 v170, v168, v171
	s_wait_dscnt 0x0
	v_fma_f32 v89, -v172, v170, v89
	ds_bpermute_b32 v170, v149, v171
	s_wait_dscnt 0x0
	v_fma_f32 v90, -v172, v170, v90
	ds_bpermute_b32 v170, v150, v171
	s_wait_dscnt 0x0
	v_fma_f32 v91, -v172, v170, v91
	ds_bpermute_b32 v170, v151, v171
	s_wait_dscnt 0x0
	v_fma_f32 v92, -v172, v170, v92
	ds_bpermute_b32 v170, v152, v171
	s_wait_dscnt 0x0
	v_fma_f32 v93, -v172, v170, v93
	ds_bpermute_b32 v170, v153, v171
	s_wait_dscnt 0x0
	v_fma_f32 v94, -v172, v170, v94
	ds_bpermute_b32 v170, v154, v171
	s_wait_dscnt 0x0
	v_fma_f32 v95, -v172, v170, v95
	ds_bpermute_b32 v170, v155, v171
	s_wait_dscnt 0x0
	v_fma_f32 v96, -v172, v170, v96
	ds_bpermute_b32 v170, v156, v171
	s_wait_dscnt 0x0
	v_fma_f32 v97, -v172, v170, v97
	ds_bpermute_b32 v170, v157, v171
	s_wait_dscnt 0x0
	v_fma_f32 v98, -v172, v170, v98
	ds_bpermute_b32 v170, v158, v171
	s_wait_dscnt 0x0
	v_fma_f32 v99, -v172, v170, v99
	ds_bpermute_b32 v170, v159, v171
	s_wait_dscnt 0x0
	v_fma_f32 v100, -v172, v170, v100
	ds_bpermute_b32 v170, v160, v171
	s_wait_dscnt 0x0
	v_fma_f32 v101, -v172, v170, v101
	ds_bpermute_b32 v170, v161, v171
	s_wait_dscnt 0x0
	v_fma_f32 v102, -v172, v170, v102
	ds_bpermute_b32 v170, v162, v171
	s_wait_dscnt 0x0
	v_fma_f32 v103, -v172, v170, v103
	ds_bpermute_b32 v170, v163, v171
	s_wait_dscnt 0x0
	v_fma_f32 v104, -v172, v170, v104
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v170, 0, v82, s1
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v170, v170, v98, s2
	ds_bpermute_b32 v171, v148, v170
	s_wait_dscnt 0x0
	v_add_f32_e32 v172, v170, v171
	ds_load_2addr_b32 v[170:171], v173 offset0:64 offset1:96
	s_wait_dscnt 0x0
	ds_bpermute_b32 v174, v168, v170
	s_wait_dscnt 0x0
	v_fma_f32 v73, -v172, v174, v73
	ds_bpermute_b32 v174, v149, v170
	s_wait_dscnt 0x0
	v_fma_f32 v74, -v172, v174, v74
	ds_bpermute_b32 v174, v150, v170
	s_wait_dscnt 0x0
	v_fma_f32 v75, -v172, v174, v75
	ds_bpermute_b32 v174, v151, v170
	s_wait_dscnt 0x0
	v_fma_f32 v76, -v172, v174, v76
	ds_bpermute_b32 v174, v152, v170
	s_wait_dscnt 0x0
	v_fma_f32 v77, -v172, v174, v77
	ds_bpermute_b32 v174, v153, v170
	s_wait_dscnt 0x0
	v_fma_f32 v78, -v172, v174, v78
	ds_bpermute_b32 v174, v154, v170
	s_wait_dscnt 0x0
	v_fma_f32 v79, -v172, v174, v79
	ds_bpermute_b32 v174, v155, v170
	s_wait_dscnt 0x0
	v_fma_f32 v80, -v172, v174, v80
	ds_bpermute_b32 v174, v156, v170
	s_wait_dscnt 0x0
	v_fma_f32 v81, -v172, v174, v81
	ds_bpermute_b32 v174, v157, v170
	s_wait_dscnt 0x0
	v_fma_f32 v82, -v172, v174, v82
	ds_bpermute_b32 v174, v158, v170
	s_wait_dscnt 0x0
	v_fma_f32 v83, -v172, v174, v83
	ds_bpermute_b32 v174, v159, v170
	s_wait_dscnt 0x0
	v_fma_f32 v84, -v172, v174, v84
	ds_bpermute_b32 v174, v160, v170
	s_wait_dscnt 0x0
	v_fma_f32 v85, -v172, v174, v85
	ds_bpermute_b32 v174, v161, v170
	s_wait_dscnt 0x0
	v_fma_f32 v86, -v172, v174, v86
	ds_bpermute_b32 v174, v162, v170
	ds_bpermute_b32 v170, v163, v170
	s_wait_dscnt 0x1
	v_fma_f32 v87, -v172, v174, v87
	s_wait_dscnt 0x0
	v_fma_f32 v88, -v172, v170, v88
	ds_bpermute_b32 v170, v168, v171
	s_wait_dscnt 0x0
	v_fma_f32 v89, -v172, v170, v89
	ds_bpermute_b32 v170, v149, v171
	s_wait_dscnt 0x0
	v_fma_f32 v90, -v172, v170, v90
	ds_bpermute_b32 v170, v150, v171
	s_wait_dscnt 0x0
	v_fma_f32 v91, -v172, v170, v91
	ds_bpermute_b32 v170, v151, v171
	s_wait_dscnt 0x0
	v_fma_f32 v92, -v172, v170, v92
	ds_bpermute_b32 v170, v152, v171
	s_wait_dscnt 0x0
	v_fma_f32 v93, -v172, v170, v93
	ds_bpermute_b32 v170, v153, v171
	s_wait_dscnt 0x0
	v_fma_f32 v94, -v172, v170, v94
	ds_bpermute_b32 v170, v154, v171
	s_wait_dscnt 0x0
	v_fma_f32 v95, -v172, v170, v95
	ds_bpermute_b32 v170, v155, v171
	s_wait_dscnt 0x0
	v_fma_f32 v96, -v172, v170, v96
	ds_bpermute_b32 v170, v156, v171
	s_wait_dscnt 0x0
	v_fma_f32 v97, -v172, v170, v97
	ds_bpermute_b32 v170, v157, v171
	s_wait_dscnt 0x0
	v_fma_f32 v98, -v172, v170, v98
	ds_bpermute_b32 v170, v158, v171
	s_wait_dscnt 0x0
	v_fma_f32 v99, -v172, v170, v99
	ds_bpermute_b32 v170, v159, v171
	s_wait_dscnt 0x0
	v_fma_f32 v100, -v172, v170, v100
	ds_bpermute_b32 v170, v160, v171
	s_wait_dscnt 0x0
	v_fma_f32 v101, -v172, v170, v101
	ds_bpermute_b32 v170, v161, v171
	s_wait_dscnt 0x0
	v_fma_f32 v102, -v172, v170, v102
	ds_bpermute_b32 v170, v162, v171
	s_wait_dscnt 0x0
	v_fma_f32 v103, -v172, v170, v103
	ds_bpermute_b32 v170, v163, v171
	s_wait_dscnt 0x0
	v_fma_f32 v104, -v172, v170, v104
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v170, 0, v83, s1
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v170, v170, v99, s2
	ds_bpermute_b32 v171, v148, v170
	s_wait_dscnt 0x0
	v_add_f32_e32 v172, v170, v171
	ds_load_2addr_b32 v[170:171], v173 offset0:128 offset1:160
	s_wait_dscnt 0x0
	ds_bpermute_b32 v174, v168, v170
	s_wait_dscnt 0x0
	v_fma_f32 v73, -v172, v174, v73
	ds_bpermute_b32 v174, v149, v170
	s_wait_dscnt 0x0
	v_fma_f32 v74, -v172, v174, v74
	ds_bpermute_b32 v174, v150, v170
	s_wait_dscnt 0x0
	v_fma_f32 v75, -v172, v174, v75
	ds_bpermute_b32 v174, v151, v170
	s_wait_dscnt 0x0
	v_fma_f32 v76, -v172, v174, v76
	ds_bpermute_b32 v174, v152, v170
	s_wait_dscnt 0x0
	v_fma_f32 v77, -v172, v174, v77
	ds_bpermute_b32 v174, v153, v170
	s_wait_dscnt 0x0
	v_fma_f32 v78, -v172, v174, v78
	ds_bpermute_b32 v174, v154, v170
	s_wait_dscnt 0x0
	v_fma_f32 v79, -v172, v174, v79
	ds_bpermute_b32 v174, v155, v170
	s_wait_dscnt 0x0
	v_fma_f32 v80, -v172, v174, v80
	ds_bpermute_b32 v174, v156, v170
	s_wait_dscnt 0x0
	v_fma_f32 v81, -v172, v174, v81
	ds_bpermute_b32 v174, v157, v170
	s_wait_dscnt 0x0
	v_fma_f32 v82, -v172, v174, v82
	ds_bpermute_b32 v174, v158, v170
	s_wait_dscnt 0x0
	v_fma_f32 v83, -v172, v174, v83
	ds_bpermute_b32 v174, v159, v170
	s_wait_dscnt 0x0
	v_fma_f32 v84, -v172, v174, v84
	ds_bpermute_b32 v174, v160, v170
	s_wait_dscnt 0x0
	v_fma_f32 v85, -v172, v174, v85
	ds_bpermute_b32 v174, v161, v170
	s_wait_dscnt 0x0
	v_fma_f32 v86, -v172, v174, v86
	ds_bpermute_b32 v174, v162, v170
	ds_bpermute_b32 v170, v163, v170
	s_wait_dscnt 0x1
	v_fma_f32 v87, -v172, v174, v87
	s_wait_dscnt 0x0
	v_fma_f32 v88, -v172, v170, v88
	ds_bpermute_b32 v170, v168, v171
	s_wait_dscnt 0x0
	v_fma_f32 v89, -v172, v170, v89
	ds_bpermute_b32 v170, v149, v171
	s_wait_dscnt 0x0
	v_fma_f32 v90, -v172, v170, v90
	ds_bpermute_b32 v170, v150, v171
	s_wait_dscnt 0x0
	v_fma_f32 v91, -v172, v170, v91
	ds_bpermute_b32 v170, v151, v171
	s_wait_dscnt 0x0
	v_fma_f32 v92, -v172, v170, v92
	ds_bpermute_b32 v170, v152, v171
	s_wait_dscnt 0x0
	v_fma_f32 v93, -v172, v170, v93
	ds_bpermute_b32 v170, v153, v171
	s_wait_dscnt 0x0
	v_fma_f32 v94, -v172, v170, v94
	ds_bpermute_b32 v170, v154, v171
	s_wait_dscnt 0x0
	v_fma_f32 v95, -v172, v170, v95
	ds_bpermute_b32 v170, v155, v171
	s_wait_dscnt 0x0
	v_fma_f32 v96, -v172, v170, v96
	ds_bpermute_b32 v170, v156, v171
	s_wait_dscnt 0x0
	v_fma_f32 v97, -v172, v170, v97
	ds_bpermute_b32 v170, v157, v171
	s_wait_dscnt 0x0
	v_fma_f32 v98, -v172, v170, v98
	ds_bpermute_b32 v170, v158, v171
	s_wait_dscnt 0x0
	v_fma_f32 v99, -v172, v170, v99
	ds_bpermute_b32 v170, v159, v171
	s_wait_dscnt 0x0
	v_fma_f32 v100, -v172, v170, v100
	ds_bpermute_b32 v170, v160, v171
	s_wait_dscnt 0x0
	v_fma_f32 v101, -v172, v170, v101
	ds_bpermute_b32 v170, v161, v171
	s_wait_dscnt 0x0
	v_fma_f32 v102, -v172, v170, v102
	ds_bpermute_b32 v170, v162, v171
	s_wait_dscnt 0x0
	v_fma_f32 v103, -v172, v170, v103
	ds_bpermute_b32 v170, v163, v171
	s_wait_dscnt 0x0
	v_fma_f32 v104, -v172, v170, v104
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v170, 0, v84, s1
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v170, v170, v100, s2
	ds_bpermute_b32 v171, v148, v170
	s_wait_dscnt 0x0
	v_add_f32_e32 v172, v170, v171
	ds_load_2addr_b32 v[170:171], v173 offset0:192 offset1:224
	s_wait_dscnt 0x0
	ds_bpermute_b32 v173, v168, v170
	s_wait_dscnt 0x0
	v_fma_f32 v73, -v172, v173, v73
	ds_bpermute_b32 v173, v149, v170
	s_wait_dscnt 0x0
	v_fma_f32 v74, -v172, v173, v74
	ds_bpermute_b32 v173, v150, v170
	s_wait_dscnt 0x0
	v_fma_f32 v75, -v172, v173, v75
	ds_bpermute_b32 v173, v151, v170
	s_wait_dscnt 0x0
	v_fma_f32 v76, -v172, v173, v76
	ds_bpermute_b32 v173, v152, v170
	s_wait_dscnt 0x0
	v_fma_f32 v77, -v172, v173, v77
	ds_bpermute_b32 v173, v153, v170
	s_wait_dscnt 0x0
	v_fma_f32 v78, -v172, v173, v78
	ds_bpermute_b32 v173, v154, v170
	s_wait_dscnt 0x0
	v_fma_f32 v79, -v172, v173, v79
	ds_bpermute_b32 v173, v155, v170
	s_wait_dscnt 0x0
	v_fma_f32 v80, -v172, v173, v80
	ds_bpermute_b32 v173, v156, v170
	s_wait_dscnt 0x0
	v_fma_f32 v81, -v172, v173, v81
	ds_bpermute_b32 v173, v157, v170
	s_wait_dscnt 0x0
	v_fma_f32 v82, -v172, v173, v82
	ds_bpermute_b32 v173, v158, v170
	s_wait_dscnt 0x0
	v_fma_f32 v83, -v172, v173, v83
	ds_bpermute_b32 v173, v159, v170
	s_wait_dscnt 0x0
	v_fma_f32 v84, -v172, v173, v84
	ds_bpermute_b32 v173, v160, v170
	s_wait_dscnt 0x0
	v_fma_f32 v85, -v172, v173, v85
	ds_bpermute_b32 v173, v161, v170
	s_wait_dscnt 0x0
	v_fma_f32 v86, -v172, v173, v86
	ds_bpermute_b32 v173, v162, v170
	ds_bpermute_b32 v170, v163, v170
	s_wait_dscnt 0x1
	v_fma_f32 v87, -v172, v173, v87
	s_wait_dscnt 0x0
	v_fma_f32 v88, -v172, v170, v88
	ds_bpermute_b32 v170, v168, v171
	v_add_nc_u32_e32 v173, 0x1400, v169
	v_add_nc_u32_e32 v169, 0x1c00, v169
	s_wait_dscnt 0x0
	v_fma_f32 v89, -v172, v170, v89
	ds_bpermute_b32 v170, v149, v171
	s_wait_dscnt 0x0
	v_fma_f32 v90, -v172, v170, v90
	ds_bpermute_b32 v170, v150, v171
	s_wait_dscnt 0x0
	v_fma_f32 v91, -v172, v170, v91
	ds_bpermute_b32 v170, v151, v171
	s_wait_dscnt 0x0
	v_fma_f32 v92, -v172, v170, v92
	ds_bpermute_b32 v170, v152, v171
	s_wait_dscnt 0x0
	v_fma_f32 v93, -v172, v170, v93
	ds_bpermute_b32 v170, v153, v171
	s_wait_dscnt 0x0
	v_fma_f32 v94, -v172, v170, v94
	ds_bpermute_b32 v170, v154, v171
	s_wait_dscnt 0x0
	v_fma_f32 v95, -v172, v170, v95
	ds_bpermute_b32 v170, v155, v171
	s_wait_dscnt 0x0
	v_fma_f32 v96, -v172, v170, v96
	ds_bpermute_b32 v170, v156, v171
	s_wait_dscnt 0x0
	v_fma_f32 v97, -v172, v170, v97
	ds_bpermute_b32 v170, v157, v171
	s_wait_dscnt 0x0
	v_fma_f32 v98, -v172, v170, v98
	ds_bpermute_b32 v170, v158, v171
	s_wait_dscnt 0x0
	v_fma_f32 v99, -v172, v170, v99
	ds_bpermute_b32 v170, v159, v171
	s_wait_dscnt 0x0
	v_fma_f32 v100, -v172, v170, v100
	ds_bpermute_b32 v170, v160, v171
	s_wait_dscnt 0x0
	v_fma_f32 v101, -v172, v170, v101
	ds_bpermute_b32 v170, v161, v171
	s_wait_dscnt 0x0
	v_fma_f32 v102, -v172, v170, v102
	ds_bpermute_b32 v170, v162, v171
	s_wait_dscnt 0x0
	v_fma_f32 v103, -v172, v170, v103
	ds_bpermute_b32 v170, v163, v171
	s_wait_dscnt 0x0
	v_fma_f32 v104, -v172, v170, v104
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v170, 0, v85, s1
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v170, v170, v101, s2
	ds_bpermute_b32 v171, v148, v170
	s_wait_dscnt 0x0
	v_add_f32_e32 v172, v170, v171
	ds_load_2addr_b32 v[170:171], v173 offset1:32
	s_wait_dscnt 0x0
	ds_bpermute_b32 v174, v168, v170
	s_wait_dscnt 0x0
	v_fma_f32 v73, -v172, v174, v73
	ds_bpermute_b32 v174, v149, v170
	s_wait_dscnt 0x0
	v_fma_f32 v74, -v172, v174, v74
	ds_bpermute_b32 v174, v150, v170
	s_wait_dscnt 0x0
	v_fma_f32 v75, -v172, v174, v75
	ds_bpermute_b32 v174, v151, v170
	s_wait_dscnt 0x0
	v_fma_f32 v76, -v172, v174, v76
	ds_bpermute_b32 v174, v152, v170
	s_wait_dscnt 0x0
	v_fma_f32 v77, -v172, v174, v77
	ds_bpermute_b32 v174, v153, v170
	s_wait_dscnt 0x0
	v_fma_f32 v78, -v172, v174, v78
	ds_bpermute_b32 v174, v154, v170
	s_wait_dscnt 0x0
	v_fma_f32 v79, -v172, v174, v79
	ds_bpermute_b32 v174, v155, v170
	s_wait_dscnt 0x0
	v_fma_f32 v80, -v172, v174, v80
	ds_bpermute_b32 v174, v156, v170
	s_wait_dscnt 0x0
	v_fma_f32 v81, -v172, v174, v81
	ds_bpermute_b32 v174, v157, v170
	s_wait_dscnt 0x0
	v_fma_f32 v82, -v172, v174, v82
	ds_bpermute_b32 v174, v158, v170
	s_wait_dscnt 0x0
	v_fma_f32 v83, -v172, v174, v83
	ds_bpermute_b32 v174, v159, v170
	s_wait_dscnt 0x0
	v_fma_f32 v84, -v172, v174, v84
	ds_bpermute_b32 v174, v160, v170
	s_wait_dscnt 0x0
	v_fma_f32 v85, -v172, v174, v85
	ds_bpermute_b32 v174, v161, v170
	s_wait_dscnt 0x0
	v_fma_f32 v86, -v172, v174, v86
	ds_bpermute_b32 v174, v162, v170
	ds_bpermute_b32 v170, v163, v170
	s_wait_dscnt 0x1
	v_fma_f32 v87, -v172, v174, v87
	s_wait_dscnt 0x0
	v_fma_f32 v88, -v172, v170, v88
	ds_bpermute_b32 v170, v168, v171
	s_wait_dscnt 0x0
	v_fma_f32 v89, -v172, v170, v89
	ds_bpermute_b32 v170, v149, v171
	s_wait_dscnt 0x0
	v_fma_f32 v90, -v172, v170, v90
	ds_bpermute_b32 v170, v150, v171
	s_wait_dscnt 0x0
	v_fma_f32 v91, -v172, v170, v91
	ds_bpermute_b32 v170, v151, v171
	s_wait_dscnt 0x0
	v_fma_f32 v92, -v172, v170, v92
	ds_bpermute_b32 v170, v152, v171
	s_wait_dscnt 0x0
	v_fma_f32 v93, -v172, v170, v93
	ds_bpermute_b32 v170, v153, v171
	s_wait_dscnt 0x0
	v_fma_f32 v94, -v172, v170, v94
	ds_bpermute_b32 v170, v154, v171
	s_wait_dscnt 0x0
	v_fma_f32 v95, -v172, v170, v95
	ds_bpermute_b32 v170, v155, v171
	s_wait_dscnt 0x0
	v_fma_f32 v96, -v172, v170, v96
	ds_bpermute_b32 v170, v156, v171
	s_wait_dscnt 0x0
	v_fma_f32 v97, -v172, v170, v97
	ds_bpermute_b32 v170, v157, v171
	s_wait_dscnt 0x0
	v_fma_f32 v98, -v172, v170, v98
	ds_bpermute_b32 v170, v158, v171
	s_wait_dscnt 0x0
	v_fma_f32 v99, -v172, v170, v99
	ds_bpermute_b32 v170, v159, v171
	s_wait_dscnt 0x0
	v_fma_f32 v100, -v172, v170, v100
	ds_bpermute_b32 v170, v160, v171
	s_wait_dscnt 0x0
	v_fma_f32 v101, -v172, v170, v101
	ds_bpermute_b32 v170, v161, v171
	s_wait_dscnt 0x0
	v_fma_f32 v102, -v172, v170, v102
	ds_bpermute_b32 v170, v162, v171
	s_wait_dscnt 0x0
	v_fma_f32 v103, -v172, v170, v103
	ds_bpermute_b32 v170, v163, v171
	s_wait_dscnt 0x0
	v_fma_f32 v104, -v172, v170, v104
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v170, 0, v86, s1
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v170, v170, v102, s2
	ds_bpermute_b32 v171, v148, v170
	s_wait_dscnt 0x0
	v_add_f32_e32 v172, v170, v171
	ds_load_2addr_b32 v[170:171], v173 offset0:64 offset1:96
	s_wait_dscnt 0x0
	ds_bpermute_b32 v174, v168, v170
	s_wait_dscnt 0x0
	v_fma_f32 v73, -v172, v174, v73
	ds_bpermute_b32 v174, v149, v170
	s_wait_dscnt 0x0
	v_fma_f32 v74, -v172, v174, v74
	ds_bpermute_b32 v174, v150, v170
	s_wait_dscnt 0x0
	v_fma_f32 v75, -v172, v174, v75
	ds_bpermute_b32 v174, v151, v170
	s_wait_dscnt 0x0
	v_fma_f32 v76, -v172, v174, v76
	ds_bpermute_b32 v174, v152, v170
	s_wait_dscnt 0x0
	v_fma_f32 v77, -v172, v174, v77
	ds_bpermute_b32 v174, v153, v170
	s_wait_dscnt 0x0
	v_fma_f32 v78, -v172, v174, v78
	ds_bpermute_b32 v174, v154, v170
	s_wait_dscnt 0x0
	v_fma_f32 v79, -v172, v174, v79
	ds_bpermute_b32 v174, v155, v170
	s_wait_dscnt 0x0
	v_fma_f32 v80, -v172, v174, v80
	ds_bpermute_b32 v174, v156, v170
	s_wait_dscnt 0x0
	v_fma_f32 v81, -v172, v174, v81
	ds_bpermute_b32 v174, v157, v170
	s_wait_dscnt 0x0
	v_fma_f32 v82, -v172, v174, v82
	ds_bpermute_b32 v174, v158, v170
	s_wait_dscnt 0x0
	v_fma_f32 v83, -v172, v174, v83
	ds_bpermute_b32 v174, v159, v170
	s_wait_dscnt 0x0
	v_fma_f32 v84, -v172, v174, v84
	ds_bpermute_b32 v174, v160, v170
	s_wait_dscnt 0x0
	v_fma_f32 v85, -v172, v174, v85
	ds_bpermute_b32 v174, v161, v170
	s_wait_dscnt 0x0
	v_fma_f32 v86, -v172, v174, v86
	ds_bpermute_b32 v174, v162, v170
	ds_bpermute_b32 v170, v163, v170
	s_wait_dscnt 0x1
	v_fma_f32 v87, -v172, v174, v87
	s_wait_dscnt 0x0
	v_fma_f32 v88, -v172, v170, v88
	ds_bpermute_b32 v170, v168, v171
	s_wait_dscnt 0x0
	v_fma_f32 v89, -v172, v170, v89
	ds_bpermute_b32 v170, v149, v171
	s_wait_dscnt 0x0
	v_fma_f32 v90, -v172, v170, v90
	ds_bpermute_b32 v170, v150, v171
	s_wait_dscnt 0x0
	v_fma_f32 v91, -v172, v170, v91
	ds_bpermute_b32 v170, v151, v171
	s_wait_dscnt 0x0
	v_fma_f32 v92, -v172, v170, v92
	ds_bpermute_b32 v170, v152, v171
	s_wait_dscnt 0x0
	v_fma_f32 v93, -v172, v170, v93
	ds_bpermute_b32 v170, v153, v171
	s_wait_dscnt 0x0
	v_fma_f32 v94, -v172, v170, v94
	ds_bpermute_b32 v170, v154, v171
	s_wait_dscnt 0x0
	v_fma_f32 v95, -v172, v170, v95
	ds_bpermute_b32 v170, v155, v171
	s_wait_dscnt 0x0
	v_fma_f32 v96, -v172, v170, v96
	ds_bpermute_b32 v170, v156, v171
	s_wait_dscnt 0x0
	v_fma_f32 v97, -v172, v170, v97
	ds_bpermute_b32 v170, v157, v171
	s_wait_dscnt 0x0
	v_fma_f32 v98, -v172, v170, v98
	ds_bpermute_b32 v170, v158, v171
	s_wait_dscnt 0x0
	v_fma_f32 v99, -v172, v170, v99
	ds_bpermute_b32 v170, v159, v171
	s_wait_dscnt 0x0
	v_fma_f32 v100, -v172, v170, v100
	ds_bpermute_b32 v170, v160, v171
	s_wait_dscnt 0x0
	v_fma_f32 v101, -v172, v170, v101
	ds_bpermute_b32 v170, v161, v171
	s_wait_dscnt 0x0
	v_fma_f32 v102, -v172, v170, v102
	ds_bpermute_b32 v170, v162, v171
	s_wait_dscnt 0x0
	v_fma_f32 v103, -v172, v170, v103
	ds_bpermute_b32 v170, v163, v171
	s_wait_dscnt 0x0
	v_fma_f32 v104, -v172, v170, v104
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v170, 0, v87, s1
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v170, v170, v103, s2
	ds_bpermute_b32 v171, v148, v170
	s_wait_dscnt 0x0
	v_add_f32_e32 v172, v170, v171
	ds_load_2addr_b32 v[170:171], v173 offset0:128 offset1:160
	s_wait_dscnt 0x0
	ds_bpermute_b32 v174, v168, v170
	s_wait_dscnt 0x0
	v_fma_f32 v73, -v172, v174, v73
	ds_bpermute_b32 v174, v149, v170
	s_wait_dscnt 0x0
	v_fma_f32 v74, -v172, v174, v74
	ds_bpermute_b32 v174, v150, v170
	s_wait_dscnt 0x0
	v_fma_f32 v75, -v172, v174, v75
	ds_bpermute_b32 v174, v151, v170
	s_wait_dscnt 0x0
	v_fma_f32 v76, -v172, v174, v76
	ds_bpermute_b32 v174, v152, v170
	s_wait_dscnt 0x0
	v_fma_f32 v77, -v172, v174, v77
	ds_bpermute_b32 v174, v153, v170
	s_wait_dscnt 0x0
	v_fma_f32 v78, -v172, v174, v78
	ds_bpermute_b32 v174, v154, v170
	s_wait_dscnt 0x0
	v_fma_f32 v79, -v172, v174, v79
	ds_bpermute_b32 v174, v155, v170
	s_wait_dscnt 0x0
	v_fma_f32 v80, -v172, v174, v80
	ds_bpermute_b32 v174, v156, v170
	s_wait_dscnt 0x0
	v_fma_f32 v81, -v172, v174, v81
	ds_bpermute_b32 v174, v157, v170
	s_wait_dscnt 0x0
	v_fma_f32 v82, -v172, v174, v82
	ds_bpermute_b32 v174, v158, v170
	s_wait_dscnt 0x0
	v_fma_f32 v83, -v172, v174, v83
	ds_bpermute_b32 v174, v159, v170
	s_wait_dscnt 0x0
	v_fma_f32 v84, -v172, v174, v84
	ds_bpermute_b32 v174, v160, v170
	s_wait_dscnt 0x0
	v_fma_f32 v85, -v172, v174, v85
	ds_bpermute_b32 v174, v161, v170
	s_wait_dscnt 0x0
	v_fma_f32 v86, -v172, v174, v86
	ds_bpermute_b32 v174, v162, v170
	ds_bpermute_b32 v170, v163, v170
	s_wait_dscnt 0x1
	v_fma_f32 v87, -v172, v174, v87
	s_wait_dscnt 0x0
	v_fma_f32 v88, -v172, v170, v88
	ds_bpermute_b32 v170, v168, v171
	s_wait_dscnt 0x0
	v_fma_f32 v89, -v172, v170, v89
	ds_bpermute_b32 v170, v149, v171
	s_wait_dscnt 0x0
	v_fma_f32 v90, -v172, v170, v90
	ds_bpermute_b32 v170, v150, v171
	s_wait_dscnt 0x0
	v_fma_f32 v91, -v172, v170, v91
	ds_bpermute_b32 v170, v151, v171
	s_wait_dscnt 0x0
	v_fma_f32 v92, -v172, v170, v92
	ds_bpermute_b32 v170, v152, v171
	s_wait_dscnt 0x0
	v_fma_f32 v93, -v172, v170, v93
	ds_bpermute_b32 v170, v153, v171
	s_wait_dscnt 0x0
	v_fma_f32 v94, -v172, v170, v94
	ds_bpermute_b32 v170, v154, v171
	s_wait_dscnt 0x0
	v_fma_f32 v95, -v172, v170, v95
	ds_bpermute_b32 v170, v155, v171
	s_wait_dscnt 0x0
	v_fma_f32 v96, -v172, v170, v96
	ds_bpermute_b32 v170, v156, v171
	s_wait_dscnt 0x0
	v_fma_f32 v97, -v172, v170, v97
	ds_bpermute_b32 v170, v157, v171
	s_wait_dscnt 0x0
	v_fma_f32 v98, -v172, v170, v98
	ds_bpermute_b32 v170, v158, v171
	s_wait_dscnt 0x0
	v_fma_f32 v99, -v172, v170, v99
	ds_bpermute_b32 v170, v159, v171
	s_wait_dscnt 0x0
	v_fma_f32 v100, -v172, v170, v100
	ds_bpermute_b32 v170, v160, v171
	s_wait_dscnt 0x0
	v_fma_f32 v101, -v172, v170, v101
	ds_bpermute_b32 v170, v161, v171
	s_wait_dscnt 0x0
	v_fma_f32 v102, -v172, v170, v102
	ds_bpermute_b32 v170, v162, v171
	s_wait_dscnt 0x0
	v_fma_f32 v103, -v172, v170, v103
	ds_bpermute_b32 v170, v163, v171
	s_wait_dscnt 0x0
	v_fma_f32 v104, -v172, v170, v104
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v170, 0, v88, s1
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v170, v170, v104, s2
	ds_bpermute_b32 v171, v148, v170
	s_wait_dscnt 0x0
	v_add_f32_e32 v172, v170, v171
	ds_load_2addr_b32 v[170:171], v173 offset0:192 offset1:224
	s_wait_dscnt 0x0
	ds_bpermute_b32 v173, v168, v170
	s_wait_dscnt 0x0
	v_fma_f32 v73, -v172, v173, v73
	ds_bpermute_b32 v173, v149, v170
	s_wait_dscnt 0x0
	v_fma_f32 v74, -v172, v173, v74
	ds_bpermute_b32 v173, v150, v170
	s_wait_dscnt 0x0
	v_fma_f32 v75, -v172, v173, v75
	ds_bpermute_b32 v173, v151, v170
	s_wait_dscnt 0x0
	v_fma_f32 v76, -v172, v173, v76
	ds_bpermute_b32 v173, v152, v170
	s_wait_dscnt 0x0
	v_fma_f32 v77, -v172, v173, v77
	ds_bpermute_b32 v173, v153, v170
	s_wait_dscnt 0x0
	v_fma_f32 v78, -v172, v173, v78
	ds_bpermute_b32 v173, v154, v170
	s_wait_dscnt 0x0
	v_fma_f32 v79, -v172, v173, v79
	ds_bpermute_b32 v173, v155, v170
	s_wait_dscnt 0x0
	v_fma_f32 v80, -v172, v173, v80
	ds_bpermute_b32 v173, v156, v170
	s_wait_dscnt 0x0
	v_fma_f32 v81, -v172, v173, v81
	ds_bpermute_b32 v173, v157, v170
	s_wait_dscnt 0x0
	v_fma_f32 v82, -v172, v173, v82
	ds_bpermute_b32 v173, v158, v170
	s_wait_dscnt 0x0
	v_fma_f32 v83, -v172, v173, v83
	ds_bpermute_b32 v173, v159, v170
	s_wait_dscnt 0x0
	v_fma_f32 v84, -v172, v173, v84
	ds_bpermute_b32 v173, v160, v170
	s_wait_dscnt 0x0
	v_fma_f32 v85, -v172, v173, v85
	ds_bpermute_b32 v173, v161, v170
	s_wait_dscnt 0x0
	v_fma_f32 v86, -v172, v173, v86
	ds_bpermute_b32 v173, v162, v170
	ds_bpermute_b32 v170, v163, v170
	s_wait_dscnt 0x1
	v_fma_f32 v87, -v172, v173, v87
	s_wait_dscnt 0x0
	v_fma_f32 v88, -v172, v170, v88
	ds_bpermute_b32 v170, v168, v171
	s_wait_dscnt 0x0
	v_fma_f32 v89, -v172, v170, v89
	ds_bpermute_b32 v170, v149, v171
	s_wait_dscnt 0x0
	v_fma_f32 v90, -v172, v170, v90
	ds_bpermute_b32 v170, v150, v171
	s_wait_dscnt 0x0
	v_fma_f32 v91, -v172, v170, v91
	ds_bpermute_b32 v170, v151, v171
	s_wait_dscnt 0x0
	v_fma_f32 v92, -v172, v170, v92
	ds_bpermute_b32 v170, v152, v171
	s_wait_dscnt 0x0
	v_fma_f32 v93, -v172, v170, v93
	ds_bpermute_b32 v170, v153, v171
	s_wait_dscnt 0x0
	v_fma_f32 v94, -v172, v170, v94
	ds_bpermute_b32 v170, v154, v171
	s_wait_dscnt 0x0
	v_fma_f32 v95, -v172, v170, v95
	ds_bpermute_b32 v170, v155, v171
	s_wait_dscnt 0x0
	v_fma_f32 v96, -v172, v170, v96
	ds_bpermute_b32 v170, v156, v171
	s_wait_dscnt 0x0
	v_fma_f32 v97, -v172, v170, v97
	ds_bpermute_b32 v170, v157, v171
	s_wait_dscnt 0x0
	v_fma_f32 v98, -v172, v170, v98
	ds_bpermute_b32 v170, v158, v171
	s_wait_dscnt 0x0
	v_fma_f32 v99, -v172, v170, v99
	ds_bpermute_b32 v170, v159, v171
	s_wait_dscnt 0x0
	v_fma_f32 v100, -v172, v170, v100
	ds_bpermute_b32 v170, v160, v171
	s_wait_dscnt 0x0
	v_fma_f32 v101, -v172, v170, v101
	ds_bpermute_b32 v170, v161, v171
	s_wait_dscnt 0x0
	v_fma_f32 v102, -v172, v170, v102
	ds_bpermute_b32 v170, v162, v171
	s_wait_dscnt 0x0
	v_fma_f32 v103, -v172, v170, v103
	ds_bpermute_b32 v170, v163, v171
	s_wait_dscnt 0x0
	v_fma_f32 v104, -v172, v170, v104
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e32 v170, 0, v81, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v170, v170, v97, s0
	ds_bpermute_b32 v171, v148, v170
	s_wait_dscnt 0x0
	v_add_f32_e32 v172, v170, v171
	ds_load_2addr_b32 v[170:171], v186 offset1:32
	s_wait_dscnt 0x0
	ds_bpermute_b32 v173, v168, v170
	s_wait_dscnt 0x0
	v_fma_f32 v73, -v172, v173, v73
	ds_bpermute_b32 v173, v149, v170
	s_wait_dscnt 0x0
	v_fma_f32 v74, -v172, v173, v74
	ds_bpermute_b32 v173, v150, v170
	s_wait_dscnt 0x0
	v_fma_f32 v75, -v172, v173, v75
	ds_bpermute_b32 v173, v151, v170
	s_wait_dscnt 0x0
	v_fma_f32 v76, -v172, v173, v76
	ds_bpermute_b32 v173, v152, v170
	s_wait_dscnt 0x0
	v_fma_f32 v77, -v172, v173, v77
	ds_bpermute_b32 v173, v153, v170
	s_wait_dscnt 0x0
	v_fma_f32 v78, -v172, v173, v78
	ds_bpermute_b32 v173, v154, v170
	s_wait_dscnt 0x0
	v_fma_f32 v79, -v172, v173, v79
	ds_bpermute_b32 v173, v155, v170
	s_wait_dscnt 0x0
	v_fma_f32 v80, -v172, v173, v80
	ds_bpermute_b32 v173, v156, v170
	s_wait_dscnt 0x0
	v_fma_f32 v81, -v172, v173, v81
	ds_bpermute_b32 v173, v157, v170
	s_wait_dscnt 0x0
	v_fma_f32 v82, -v172, v173, v82
	ds_bpermute_b32 v173, v158, v170
	s_wait_dscnt 0x0
	v_fma_f32 v83, -v172, v173, v83
	ds_bpermute_b32 v173, v159, v170
	s_wait_dscnt 0x0
	v_fma_f32 v84, -v172, v173, v84
	ds_bpermute_b32 v173, v160, v170
	s_wait_dscnt 0x0
	v_fma_f32 v85, -v172, v173, v85
	ds_bpermute_b32 v173, v161, v170
	s_wait_dscnt 0x0
	v_fma_f32 v86, -v172, v173, v86
	ds_bpermute_b32 v173, v162, v170
	ds_bpermute_b32 v170, v163, v170
	s_wait_dscnt 0x1
	v_fma_f32 v87, -v172, v173, v87
	s_wait_dscnt 0x0
	v_fma_f32 v88, -v172, v170, v88
	ds_bpermute_b32 v170, v168, v171
	s_wait_dscnt 0x0
	v_fma_f32 v89, -v172, v170, v89
	ds_bpermute_b32 v170, v149, v171
	s_wait_dscnt 0x0
	v_fma_f32 v90, -v172, v170, v90
	ds_bpermute_b32 v170, v150, v171
	s_wait_dscnt 0x0
	v_fma_f32 v91, -v172, v170, v91
	ds_bpermute_b32 v170, v151, v171
	s_wait_dscnt 0x0
	v_fma_f32 v92, -v172, v170, v92
	ds_bpermute_b32 v170, v152, v171
	s_wait_dscnt 0x0
	v_fma_f32 v93, -v172, v170, v93
	ds_bpermute_b32 v170, v153, v171
	s_wait_dscnt 0x0
	v_fma_f32 v94, -v172, v170, v94
	ds_bpermute_b32 v170, v154, v171
	s_wait_dscnt 0x0
	v_fma_f32 v95, -v172, v170, v95
	ds_bpermute_b32 v170, v155, v171
	s_wait_dscnt 0x0
	v_fma_f32 v96, -v172, v170, v96
	ds_bpermute_b32 v170, v156, v171
	s_wait_dscnt 0x0
	v_fma_f32 v97, -v172, v170, v97
	ds_bpermute_b32 v170, v157, v171
	s_wait_dscnt 0x0
	v_fma_f32 v98, -v172, v170, v98
	ds_bpermute_b32 v170, v158, v171
	s_wait_dscnt 0x0
	v_fma_f32 v99, -v172, v170, v99
	ds_bpermute_b32 v170, v159, v171
	s_wait_dscnt 0x0
	v_fma_f32 v100, -v172, v170, v100
	ds_bpermute_b32 v170, v160, v171
	s_wait_dscnt 0x0
	v_fma_f32 v101, -v172, v170, v101
	ds_bpermute_b32 v170, v161, v171
	s_wait_dscnt 0x0
	v_fma_f32 v102, -v172, v170, v102
	ds_bpermute_b32 v170, v162, v171
	s_wait_dscnt 0x0
	v_fma_f32 v103, -v172, v170, v103
	ds_bpermute_b32 v170, v163, v171
	s_wait_dscnt 0x0
	v_fma_f32 v104, -v172, v170, v104
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e32 v170, 0, v82, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v170, v170, v98, s0
	ds_bpermute_b32 v171, v148, v170
	s_wait_dscnt 0x0
	v_add_f32_e32 v172, v170, v171
	ds_load_2addr_b32 v[170:171], v186 offset0:64 offset1:96
	s_wait_dscnt 0x0
	ds_bpermute_b32 v173, v168, v170
	s_wait_dscnt 0x0
	v_fma_f32 v73, -v172, v173, v73
	ds_bpermute_b32 v173, v149, v170
	s_wait_dscnt 0x0
	v_fma_f32 v74, -v172, v173, v74
	ds_bpermute_b32 v173, v150, v170
	s_wait_dscnt 0x0
	v_fma_f32 v75, -v172, v173, v75
	ds_bpermute_b32 v173, v151, v170
	s_wait_dscnt 0x0
	v_fma_f32 v76, -v172, v173, v76
	ds_bpermute_b32 v173, v152, v170
	s_wait_dscnt 0x0
	v_fma_f32 v77, -v172, v173, v77
	ds_bpermute_b32 v173, v153, v170
	s_wait_dscnt 0x0
	v_fma_f32 v78, -v172, v173, v78
	ds_bpermute_b32 v173, v154, v170
	s_wait_dscnt 0x0
	v_fma_f32 v79, -v172, v173, v79
	ds_bpermute_b32 v173, v155, v170
	s_wait_dscnt 0x0
	v_fma_f32 v80, -v172, v173, v80
	ds_bpermute_b32 v173, v156, v170
	s_wait_dscnt 0x0
	v_fma_f32 v81, -v172, v173, v81
	ds_bpermute_b32 v173, v157, v170
	s_wait_dscnt 0x0
	v_fma_f32 v82, -v172, v173, v82
	ds_bpermute_b32 v173, v158, v170
	s_wait_dscnt 0x0
	v_fma_f32 v83, -v172, v173, v83
	ds_bpermute_b32 v173, v159, v170
	s_wait_dscnt 0x0
	v_fma_f32 v84, -v172, v173, v84
	ds_bpermute_b32 v173, v160, v170
	s_wait_dscnt 0x0
	v_fma_f32 v85, -v172, v173, v85
	ds_bpermute_b32 v173, v161, v170
	s_wait_dscnt 0x0
	v_fma_f32 v86, -v172, v173, v86
	ds_bpermute_b32 v173, v162, v170
	ds_bpermute_b32 v170, v163, v170
	s_wait_dscnt 0x1
	v_fma_f32 v87, -v172, v173, v87
	s_wait_dscnt 0x0
	v_fma_f32 v88, -v172, v170, v88
	ds_bpermute_b32 v170, v168, v171
	s_wait_dscnt 0x0
	v_fma_f32 v89, -v172, v170, v89
	ds_bpermute_b32 v170, v149, v171
	s_wait_dscnt 0x0
	v_fma_f32 v90, -v172, v170, v90
	ds_bpermute_b32 v170, v150, v171
	s_wait_dscnt 0x0
	v_fma_f32 v91, -v172, v170, v91
	ds_bpermute_b32 v170, v151, v171
	s_wait_dscnt 0x0
	v_fma_f32 v92, -v172, v170, v92
	ds_bpermute_b32 v170, v152, v171
	s_wait_dscnt 0x0
	v_fma_f32 v93, -v172, v170, v93
	ds_bpermute_b32 v170, v153, v171
	s_wait_dscnt 0x0
	v_fma_f32 v94, -v172, v170, v94
	ds_bpermute_b32 v170, v154, v171
	s_wait_dscnt 0x0
	v_fma_f32 v95, -v172, v170, v95
	ds_bpermute_b32 v170, v155, v171
	s_wait_dscnt 0x0
	v_fma_f32 v96, -v172, v170, v96
	ds_bpermute_b32 v170, v156, v171
	s_wait_dscnt 0x0
	v_fma_f32 v97, -v172, v170, v97
	ds_bpermute_b32 v170, v157, v171
	s_wait_dscnt 0x0
	v_fma_f32 v98, -v172, v170, v98
	ds_bpermute_b32 v170, v158, v171
	s_wait_dscnt 0x0
	v_fma_f32 v99, -v172, v170, v99
	ds_bpermute_b32 v170, v159, v171
	s_wait_dscnt 0x0
	v_fma_f32 v100, -v172, v170, v100
	ds_bpermute_b32 v170, v160, v171
	s_wait_dscnt 0x0
	v_fma_f32 v101, -v172, v170, v101
	ds_bpermute_b32 v170, v161, v171
	s_wait_dscnt 0x0
	v_fma_f32 v102, -v172, v170, v102
	ds_bpermute_b32 v170, v162, v171
	s_wait_dscnt 0x0
	v_fma_f32 v103, -v172, v170, v103
	ds_bpermute_b32 v170, v163, v171
	s_wait_dscnt 0x0
	v_fma_f32 v104, -v172, v170, v104
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e32 v170, 0, v83, vcc_lo
	ds_load_2addr_b32 v[184:185], v186 offset0:128 offset1:160
	v_cndmask_b32_e64 v170, v170, v99, s0
	ds_bpermute_b32 v171, v148, v170
	s_wait_dscnt 0x0
	v_add_f32_e32 v187, v170, v171
	ds_bpermute_b32 v170, v168, v184
	s_wait_dscnt 0x0
	v_fma_f32 v73, -v187, v170, v73
	ds_bpermute_b32 v170, v149, v184
	s_wait_dscnt 0x0
	v_fma_f32 v74, -v187, v170, v74
	ds_bpermute_b32 v170, v150, v184
	s_wait_dscnt 0x0
	v_fma_f32 v75, -v187, v170, v75
	ds_bpermute_b32 v170, v151, v184
	s_wait_dscnt 0x0
	v_fma_f32 v76, -v187, v170, v76
	ds_bpermute_b32 v170, v152, v184
	s_wait_dscnt 0x0
	v_fma_f32 v77, -v187, v170, v77
	ds_bpermute_b32 v170, v153, v184
	s_wait_dscnt 0x0
	v_fma_f32 v78, -v187, v170, v78
	ds_bpermute_b32 v170, v154, v184
	s_wait_dscnt 0x0
	v_fma_f32 v79, -v187, v170, v79
	ds_bpermute_b32 v170, v155, v184
	s_wait_dscnt 0x0
	v_fma_f32 v80, -v187, v170, v80
	ds_bpermute_b32 v170, v156, v184
	s_wait_dscnt 0x0
	v_fma_f32 v81, -v187, v170, v81
	ds_bpermute_b32 v170, v157, v184
	s_wait_dscnt 0x0
	v_fma_f32 v82, -v187, v170, v82
	ds_bpermute_b32 v170, v158, v184
	s_wait_dscnt 0x0
	v_fma_f32 v83, -v187, v170, v83
	ds_bpermute_b32 v170, v159, v184
	s_wait_dscnt 0x0
	v_fma_f32 v84, -v187, v170, v84
	ds_bpermute_b32 v170, v160, v184
	s_wait_dscnt 0x0
	v_fma_f32 v85, -v187, v170, v85
	ds_bpermute_b32 v170, v161, v184
	s_wait_dscnt 0x0
	v_fma_f32 v86, -v187, v170, v86
	ds_bpermute_b32 v170, v162, v184
	s_wait_dscnt 0x0
	v_fma_f32 v87, -v187, v170, v87
	ds_bpermute_b32 v170, v163, v184
	s_wait_dscnt 0x0
	v_fma_f32 v88, -v187, v170, v88
	ds_bpermute_b32 v170, v168, v185
	s_wait_dscnt 0x0
	v_fma_f32 v170, -v187, v170, v89
	ds_bpermute_b32 v89, v149, v185
	s_wait_dscnt 0x0
	v_fma_f32 v171, -v187, v89, v90
	ds_bpermute_b32 v89, v150, v185
	s_wait_dscnt 0x0
	v_fma_f32 v172, -v187, v89, v91
	ds_bpermute_b32 v89, v151, v185
	s_wait_dscnt 0x0
	v_fma_f32 v173, -v187, v89, v92
	ds_bpermute_b32 v89, v152, v185
	s_wait_dscnt 0x0
	v_fma_f32 v174, -v187, v89, v93
	ds_bpermute_b32 v89, v153, v185
	s_wait_dscnt 0x0
	v_fma_f32 v175, -v187, v89, v94
	ds_bpermute_b32 v89, v154, v185
	s_wait_dscnt 0x0
	v_fma_f32 v176, -v187, v89, v95
	ds_bpermute_b32 v89, v155, v185
	s_wait_dscnt 0x0
	v_fma_f32 v177, -v187, v89, v96
	ds_bpermute_b32 v89, v156, v185
	s_wait_dscnt 0x0
	v_fma_f32 v178, -v187, v89, v97
	ds_bpermute_b32 v89, v157, v185
	s_wait_dscnt 0x0
	v_fma_f32 v179, -v187, v89, v98
	ds_bpermute_b32 v89, v158, v185
	s_wait_dscnt 0x0
	v_fma_f32 v180, -v187, v89, v99
	ds_bpermute_b32 v89, v159, v185
	s_wait_dscnt 0x0
	v_fma_f32 v181, -v187, v89, v100
	ds_bpermute_b32 v89, v160, v185
	s_wait_dscnt 0x0
	v_fma_f32 v182, -v187, v89, v101
	ds_bpermute_b32 v89, v161, v185
	s_wait_dscnt 0x0
	v_fma_f32 v183, -v187, v89, v102
	ds_bpermute_b32 v89, v162, v185
	s_wait_dscnt 0x0
	v_fma_f32 v184, -v187, v89, v103
	ds_bpermute_b32 v89, v163, v185
	s_wait_dscnt 0x0
	v_fma_f32 v185, -v187, v89, v104
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e32 v89, 0, v84, vcc_lo
	ds_load_2addr_b32 v[186:187], v186 offset0:192 offset1:224
	v_cndmask_b32_e64 v89, v89, v181, s0
	ds_bpermute_b32 v90, v148, v89
	s_wait_dscnt 0x0
	v_add_f32_e32 v188, v89, v90
	ds_bpermute_b32 v89, v168, v186
	s_wait_dscnt 0x0
	v_fma_f32 v97, -v188, v89, v73
	ds_bpermute_b32 v73, v149, v186
	s_wait_dscnt 0x0
	v_fma_f32 v98, -v188, v73, v74
	ds_bpermute_b32 v73, v150, v186
	ds_bpermute_b32 v74, v157, v187
	s_wait_dscnt 0x1
	v_fma_f32 v99, -v188, v73, v75
	ds_bpermute_b32 v73, v151, v186
	ds_bpermute_b32 v75, v158, v187
	s_wait_dscnt 0x2
	v_fma_f32 v74, -v188, v74, v179
	s_wait_dscnt 0x1
	v_fma_f32 v100, -v188, v73, v76
	ds_bpermute_b32 v73, v152, v186
	ds_bpermute_b32 v76, v159, v187
	s_wait_dscnt 0x2
	v_fma_f32 v75, -v188, v75, v180
	s_wait_dscnt 0x1
	v_fma_f32 v101, -v188, v73, v77
	ds_bpermute_b32 v73, v153, v186
	ds_bpermute_b32 v77, v160, v187
	s_wait_dscnt 0x2
	v_fma_f32 v76, -v188, v76, v181
	s_wait_dscnt 0x1
	v_fma_f32 v102, -v188, v73, v78
	ds_bpermute_b32 v73, v154, v186
	ds_bpermute_b32 v78, v161, v187
	s_wait_dscnt 0x2
	v_fma_f32 v77, -v188, v77, v182
	s_wait_dscnt 0x1
	v_fma_f32 v103, -v188, v73, v79
	ds_bpermute_b32 v73, v155, v186
	ds_bpermute_b32 v79, v162, v187
	s_wait_dscnt 0x2
	v_fma_f32 v78, -v188, v78, v183
	s_wait_dscnt 0x1
	v_fma_f32 v104, -v188, v73, v80
	ds_bpermute_b32 v73, v156, v186
	ds_bpermute_b32 v80, v163, v187
	s_wait_dscnt 0x2
	v_fma_f32 v79, -v188, v79, v184
	s_wait_dscnt 0x1
	v_fma_f32 v89, -v188, v73, v81
	ds_bpermute_b32 v73, v157, v186
	s_wait_dscnt 0x1
	v_fma_f32 v80, -v188, v80, v185
	s_wait_dscnt 0x0
	v_fma_f32 v90, -v188, v73, v82
	ds_bpermute_b32 v73, v158, v186
	s_wait_dscnt 0x0
	v_fma_f32 v91, -v188, v73, v83
	ds_bpermute_b32 v73, v159, v186
	s_wait_dscnt 0x0
	v_fma_f32 v92, -v188, v73, v84
	ds_bpermute_b32 v73, v160, v186
	s_wait_dscnt 0x0
	v_fma_f32 v93, -v188, v73, v85
	ds_bpermute_b32 v73, v161, v186
	s_wait_dscnt 0x0
	v_fma_f32 v94, -v188, v73, v86
	ds_bpermute_b32 v73, v162, v186
	s_wait_dscnt 0x0
	v_fma_f32 v95, -v188, v73, v87
	ds_bpermute_b32 v73, v163, v186
	s_wait_dscnt 0x0
	v_fma_f32 v96, -v188, v73, v88
	ds_bpermute_b32 v73, v168, v187
	s_wait_dscnt 0x0
	v_fma_f32 v81, -v188, v73, v170
	ds_bpermute_b32 v73, v149, v187
	s_wait_dscnt 0x0
	v_fma_f32 v82, -v188, v73, v171
	ds_bpermute_b32 v73, v150, v187
	s_wait_dscnt 0x0
	v_fma_f32 v83, -v188, v73, v172
	ds_bpermute_b32 v73, v151, v187
	s_wait_dscnt 0x0
	v_fma_f32 v84, -v188, v73, v173
	ds_bpermute_b32 v73, v152, v187
	s_wait_dscnt 0x0
	v_fma_f32 v85, -v188, v73, v174
	ds_bpermute_b32 v73, v153, v187
	s_wait_dscnt 0x0
	v_fma_f32 v86, -v188, v73, v175
	ds_bpermute_b32 v73, v154, v187
	s_wait_dscnt 0x0
	v_fma_f32 v87, -v188, v73, v176
	ds_bpermute_b32 v73, v155, v187
	s_wait_dscnt 0x0
	v_fma_f32 v88, -v188, v73, v177
	ds_bpermute_b32 v73, v156, v187
	s_wait_dscnt 0x0
	v_fma_f32 v73, -v188, v73, v178
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e32 v170, 0, v93, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v170, v170, v77, s0
	ds_bpermute_b32 v171, v148, v170
	s_wait_dscnt 0x0
	v_add_f32_e32 v170, v170, v171
	ds_load_2addr_b32 v[171:172], v169 offset1:32
	s_wait_dscnt 0x0
	ds_bpermute_b32 v173, v168, v171
	s_wait_dscnt 0x0
	v_fma_f32 v97, -v170, v173, v97
	ds_bpermute_b32 v173, v149, v171
	s_wait_dscnt 0x0
	v_fma_f32 v98, -v170, v173, v98
	ds_bpermute_b32 v173, v150, v171
	s_wait_dscnt 0x0
	v_fma_f32 v99, -v170, v173, v99
	ds_bpermute_b32 v173, v151, v171
	s_wait_dscnt 0x0
	v_fma_f32 v100, -v170, v173, v100
	ds_bpermute_b32 v173, v152, v171
	s_wait_dscnt 0x0
	v_fma_f32 v101, -v170, v173, v101
	ds_bpermute_b32 v173, v153, v171
	s_wait_dscnt 0x0
	v_fma_f32 v102, -v170, v173, v102
	ds_bpermute_b32 v173, v154, v171
	s_wait_dscnt 0x0
	v_fma_f32 v103, -v170, v173, v103
	ds_bpermute_b32 v173, v155, v171
	s_wait_dscnt 0x0
	v_fma_f32 v104, -v170, v173, v104
	ds_bpermute_b32 v173, v156, v171
	s_wait_dscnt 0x0
	v_fma_f32 v89, -v170, v173, v89
	ds_bpermute_b32 v173, v157, v171
	s_wait_dscnt 0x0
	v_fma_f32 v90, -v170, v173, v90
	ds_bpermute_b32 v173, v158, v171
	s_wait_dscnt 0x0
	v_fma_f32 v91, -v170, v173, v91
	ds_bpermute_b32 v173, v159, v171
	s_wait_dscnt 0x0
	v_fma_f32 v92, -v170, v173, v92
	ds_bpermute_b32 v173, v160, v171
	s_wait_dscnt 0x0
	v_fma_f32 v93, -v170, v173, v93
	ds_bpermute_b32 v173, v161, v171
	s_wait_dscnt 0x0
	v_fma_f32 v94, -v170, v173, v94
	ds_bpermute_b32 v173, v162, v171
	ds_bpermute_b32 v171, v163, v171
	s_wait_dscnt 0x1
	v_fma_f32 v95, -v170, v173, v95
	s_wait_dscnt 0x0
	v_fma_f32 v96, -v170, v171, v96
	ds_bpermute_b32 v171, v168, v172
	s_wait_dscnt 0x0
	v_fma_f32 v81, -v170, v171, v81
	ds_bpermute_b32 v171, v149, v172
	s_wait_dscnt 0x0
	v_fma_f32 v82, -v170, v171, v82
	ds_bpermute_b32 v171, v150, v172
	s_wait_dscnt 0x0
	v_fma_f32 v83, -v170, v171, v83
	ds_bpermute_b32 v171, v151, v172
	s_wait_dscnt 0x0
	v_fma_f32 v84, -v170, v171, v84
	ds_bpermute_b32 v171, v152, v172
	s_wait_dscnt 0x0
	v_fma_f32 v85, -v170, v171, v85
	ds_bpermute_b32 v171, v153, v172
	s_wait_dscnt 0x0
	v_fma_f32 v86, -v170, v171, v86
	ds_bpermute_b32 v171, v154, v172
	s_wait_dscnt 0x0
	v_fma_f32 v87, -v170, v171, v87
	ds_bpermute_b32 v171, v155, v172
	s_wait_dscnt 0x0
	v_fma_f32 v88, -v170, v171, v88
	ds_bpermute_b32 v171, v156, v172
	s_wait_dscnt 0x0
	v_fma_f32 v73, -v170, v171, v73
	ds_bpermute_b32 v171, v157, v172
	s_wait_dscnt 0x0
	v_fma_f32 v74, -v170, v171, v74
	ds_bpermute_b32 v171, v158, v172
	s_wait_dscnt 0x0
	v_fma_f32 v75, -v170, v171, v75
	ds_bpermute_b32 v171, v159, v172
	s_wait_dscnt 0x0
	v_fma_f32 v76, -v170, v171, v76
	ds_bpermute_b32 v171, v160, v172
	s_wait_dscnt 0x0
	v_fma_f32 v77, -v170, v171, v77
	ds_bpermute_b32 v171, v161, v172
	s_wait_dscnt 0x0
	v_fma_f32 v78, -v170, v171, v78
	ds_bpermute_b32 v171, v162, v172
	s_wait_dscnt 0x0
	v_fma_f32 v79, -v170, v171, v79
	ds_bpermute_b32 v171, v163, v172
	s_wait_dscnt 0x0
	v_fma_f32 v80, -v170, v171, v80
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e32 v170, 0, v94, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v170, v170, v78, s0
	ds_bpermute_b32 v171, v148, v170
	s_wait_dscnt 0x0
	v_add_f32_e32 v172, v170, v171
	ds_load_2addr_b32 v[170:171], v169 offset0:64 offset1:96
	s_wait_dscnt 0x0
	ds_bpermute_b32 v173, v168, v170
	s_wait_dscnt 0x0
	v_fma_f32 v97, -v172, v173, v97
	ds_bpermute_b32 v173, v149, v170
	s_wait_dscnt 0x0
	v_fma_f32 v98, -v172, v173, v98
	ds_bpermute_b32 v173, v150, v170
	s_wait_dscnt 0x0
	v_fma_f32 v99, -v172, v173, v99
	ds_bpermute_b32 v173, v151, v170
	s_wait_dscnt 0x0
	v_fma_f32 v100, -v172, v173, v100
	ds_bpermute_b32 v173, v152, v170
	s_wait_dscnt 0x0
	v_fma_f32 v101, -v172, v173, v101
	ds_bpermute_b32 v173, v153, v170
	s_wait_dscnt 0x0
	v_fma_f32 v102, -v172, v173, v102
	ds_bpermute_b32 v173, v154, v170
	s_wait_dscnt 0x0
	v_fma_f32 v103, -v172, v173, v103
	ds_bpermute_b32 v173, v155, v170
	s_wait_dscnt 0x0
	v_fma_f32 v104, -v172, v173, v104
	ds_bpermute_b32 v173, v156, v170
	s_wait_dscnt 0x0
	v_fma_f32 v89, -v172, v173, v89
	ds_bpermute_b32 v173, v157, v170
	s_wait_dscnt 0x0
	v_fma_f32 v90, -v172, v173, v90
	ds_bpermute_b32 v173, v158, v170
	s_wait_dscnt 0x0
	v_fma_f32 v91, -v172, v173, v91
	ds_bpermute_b32 v173, v159, v170
	s_wait_dscnt 0x0
	v_fma_f32 v92, -v172, v173, v92
	ds_bpermute_b32 v173, v160, v170
	s_wait_dscnt 0x0
	v_fma_f32 v93, -v172, v173, v93
	ds_bpermute_b32 v173, v161, v170
	s_wait_dscnt 0x0
	v_fma_f32 v94, -v172, v173, v94
	ds_bpermute_b32 v173, v162, v170
	ds_bpermute_b32 v170, v163, v170
	s_wait_dscnt 0x1
	v_fma_f32 v95, -v172, v173, v95
	s_wait_dscnt 0x0
	v_fma_f32 v96, -v172, v170, v96
	ds_bpermute_b32 v170, v168, v171
	s_wait_dscnt 0x0
	v_fma_f32 v81, -v172, v170, v81
	ds_bpermute_b32 v170, v149, v171
	s_wait_dscnt 0x0
	v_fma_f32 v82, -v172, v170, v82
	ds_bpermute_b32 v170, v150, v171
	s_wait_dscnt 0x0
	v_fma_f32 v83, -v172, v170, v83
	ds_bpermute_b32 v170, v151, v171
	s_wait_dscnt 0x0
	v_fma_f32 v84, -v172, v170, v84
	ds_bpermute_b32 v170, v152, v171
	s_wait_dscnt 0x0
	v_fma_f32 v85, -v172, v170, v85
	ds_bpermute_b32 v170, v153, v171
	s_wait_dscnt 0x0
	v_fma_f32 v86, -v172, v170, v86
	ds_bpermute_b32 v170, v154, v171
	s_wait_dscnt 0x0
	v_fma_f32 v87, -v172, v170, v87
	ds_bpermute_b32 v170, v155, v171
	s_wait_dscnt 0x0
	v_fma_f32 v88, -v172, v170, v88
	ds_bpermute_b32 v170, v156, v171
	s_wait_dscnt 0x0
	v_fma_f32 v73, -v172, v170, v73
	ds_bpermute_b32 v170, v157, v171
	s_wait_dscnt 0x0
	v_fma_f32 v74, -v172, v170, v74
	ds_bpermute_b32 v170, v158, v171
	s_wait_dscnt 0x0
	v_fma_f32 v75, -v172, v170, v75
	ds_bpermute_b32 v170, v159, v171
	s_wait_dscnt 0x0
	v_fma_f32 v76, -v172, v170, v76
	ds_bpermute_b32 v170, v160, v171
	s_wait_dscnt 0x0
	v_fma_f32 v77, -v172, v170, v77
	ds_bpermute_b32 v170, v161, v171
	s_wait_dscnt 0x0
	v_fma_f32 v78, -v172, v170, v78
	ds_bpermute_b32 v170, v162, v171
	s_wait_dscnt 0x0
	v_fma_f32 v79, -v172, v170, v79
	ds_bpermute_b32 v170, v163, v171
	s_wait_dscnt 0x0
	v_fma_f32 v80, -v172, v170, v80
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e32 v170, 0, v95, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v170, v170, v79, s0
	ds_bpermute_b32 v171, v148, v170
	s_wait_dscnt 0x0
	v_add_f32_e32 v172, v170, v171
	ds_load_2addr_b32 v[170:171], v169 offset0:128 offset1:160
	s_wait_dscnt 0x0
	ds_bpermute_b32 v173, v168, v170
	s_wait_dscnt 0x0
	v_fma_f32 v97, -v172, v173, v97
	ds_bpermute_b32 v173, v149, v170
	s_wait_dscnt 0x0
	v_fma_f32 v98, -v172, v173, v98
	ds_bpermute_b32 v173, v150, v170
	s_wait_dscnt 0x0
	v_fma_f32 v99, -v172, v173, v99
	ds_bpermute_b32 v173, v151, v170
	s_wait_dscnt 0x0
	v_fma_f32 v100, -v172, v173, v100
	ds_bpermute_b32 v173, v152, v170
	s_wait_dscnt 0x0
	v_fma_f32 v101, -v172, v173, v101
	ds_bpermute_b32 v173, v153, v170
	s_wait_dscnt 0x0
	v_fma_f32 v102, -v172, v173, v102
	ds_bpermute_b32 v173, v154, v170
	s_wait_dscnt 0x0
	v_fma_f32 v103, -v172, v173, v103
	ds_bpermute_b32 v173, v155, v170
	s_wait_dscnt 0x0
	v_fma_f32 v104, -v172, v173, v104
	ds_bpermute_b32 v173, v156, v170
	s_wait_dscnt 0x0
	v_fma_f32 v89, -v172, v173, v89
	ds_bpermute_b32 v173, v157, v170
	s_wait_dscnt 0x0
	v_fma_f32 v90, -v172, v173, v90
	ds_bpermute_b32 v173, v158, v170
	s_wait_dscnt 0x0
	v_fma_f32 v91, -v172, v173, v91
	ds_bpermute_b32 v173, v159, v170
	s_wait_dscnt 0x0
	v_fma_f32 v92, -v172, v173, v92
	ds_bpermute_b32 v173, v160, v170
	s_wait_dscnt 0x0
	v_fma_f32 v93, -v172, v173, v93
	ds_bpermute_b32 v173, v161, v170
	s_wait_dscnt 0x0
	v_fma_f32 v94, -v172, v173, v94
	ds_bpermute_b32 v173, v162, v170
	ds_bpermute_b32 v170, v163, v170
	s_wait_dscnt 0x1
	v_fma_f32 v95, -v172, v173, v95
	s_wait_dscnt 0x0
	v_fma_f32 v96, -v172, v170, v96
	ds_bpermute_b32 v170, v168, v171
	s_wait_dscnt 0x0
	v_fma_f32 v81, -v172, v170, v81
	ds_bpermute_b32 v170, v149, v171
	s_wait_dscnt 0x0
	v_fma_f32 v82, -v172, v170, v82
	ds_bpermute_b32 v170, v150, v171
	s_wait_dscnt 0x0
	v_fma_f32 v83, -v172, v170, v83
	ds_bpermute_b32 v170, v151, v171
	s_wait_dscnt 0x0
	v_fma_f32 v84, -v172, v170, v84
	ds_bpermute_b32 v170, v152, v171
	s_wait_dscnt 0x0
	v_fma_f32 v85, -v172, v170, v85
	ds_bpermute_b32 v170, v153, v171
	s_wait_dscnt 0x0
	v_fma_f32 v86, -v172, v170, v86
	ds_bpermute_b32 v170, v154, v171
	s_wait_dscnt 0x0
	v_fma_f32 v87, -v172, v170, v87
	ds_bpermute_b32 v170, v155, v171
	s_wait_dscnt 0x0
	v_fma_f32 v88, -v172, v170, v88
	ds_bpermute_b32 v170, v156, v171
	s_wait_dscnt 0x0
	v_fma_f32 v73, -v172, v170, v73
	ds_bpermute_b32 v170, v157, v171
	s_wait_dscnt 0x0
	v_fma_f32 v74, -v172, v170, v74
	ds_bpermute_b32 v170, v158, v171
	s_wait_dscnt 0x0
	v_fma_f32 v75, -v172, v170, v75
	ds_bpermute_b32 v170, v159, v171
	s_wait_dscnt 0x0
	v_fma_f32 v76, -v172, v170, v76
	ds_bpermute_b32 v170, v160, v171
	s_wait_dscnt 0x0
	v_fma_f32 v77, -v172, v170, v77
	ds_bpermute_b32 v170, v161, v171
	s_wait_dscnt 0x0
	v_fma_f32 v78, -v172, v170, v78
	ds_bpermute_b32 v170, v162, v171
	s_wait_dscnt 0x0
	v_fma_f32 v79, -v172, v170, v79
	ds_bpermute_b32 v170, v163, v171
	s_wait_dscnt 0x0
	v_fma_f32 v80, -v172, v170, v80
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e32 v170, 0, v96, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v170, v170, v80, s0
	ds_bpermute_b32 v171, v148, v170
	s_wait_dscnt 0x0
	v_add_f32_e32 v171, v170, v171
	ds_load_2addr_b32 v[169:170], v169 offset0:192 offset1:224
	s_wait_dscnt 0x0
	ds_bpermute_b32 v172, v168, v169
	ds_bpermute_b32 v168, v168, v170
	s_wait_dscnt 0x1
	v_fma_f32 v97, -v171, v172, v97
	s_wait_dscnt 0x0
	v_fma_f32 v81, -v171, v168, v81
	ds_bpermute_b32 v168, v149, v170
	ds_bpermute_b32 v172, v149, v169
	s_wait_dscnt 0x1
	v_fma_f32 v82, -v171, v168, v82
	ds_bpermute_b32 v168, v150, v170
	s_wait_dscnt 0x1
	v_fma_f32 v98, -v171, v172, v98
	ds_bpermute_b32 v172, v150, v169
	s_wait_dscnt 0x1
	v_fma_f32 v83, -v171, v168, v83
	ds_bpermute_b32 v168, v151, v170
	s_wait_dscnt 0x1
	v_fma_f32 v99, -v171, v172, v99
	ds_bpermute_b32 v172, v151, v169
	s_wait_dscnt 0x1
	v_fma_f32 v84, -v171, v168, v84
	ds_bpermute_b32 v168, v152, v170
	s_wait_dscnt 0x1
	v_fma_f32 v100, -v171, v172, v100
	ds_bpermute_b32 v172, v152, v169
	s_wait_dscnt 0x1
	v_fma_f32 v85, -v171, v168, v85
	ds_bpermute_b32 v168, v153, v170
	s_wait_dscnt 0x1
	v_fma_f32 v101, -v171, v172, v101
	ds_bpermute_b32 v172, v153, v169
	s_wait_dscnt 0x1
	v_fma_f32 v86, -v171, v168, v86
	ds_bpermute_b32 v168, v154, v170
	s_wait_dscnt 0x1
	v_fma_f32 v102, -v171, v172, v102
	ds_bpermute_b32 v172, v154, v169
	s_wait_dscnt 0x1
	v_fma_f32 v87, -v171, v168, v87
	ds_bpermute_b32 v168, v155, v170
	s_wait_dscnt 0x1
	v_fma_f32 v103, -v171, v172, v103
	ds_bpermute_b32 v172, v155, v169
	s_wait_dscnt 0x1
	v_fma_f32 v88, -v171, v168, v88
	ds_bpermute_b32 v168, v156, v170
	s_wait_dscnt 0x1
	v_fma_f32 v104, -v171, v172, v104
	ds_bpermute_b32 v172, v156, v169
	s_wait_dscnt 0x1
	v_fma_f32 v73, -v171, v168, v73
	ds_bpermute_b32 v168, v157, v170
	s_wait_dscnt 0x1
	v_fma_f32 v89, -v171, v172, v89
	ds_bpermute_b32 v172, v157, v169
	s_wait_dscnt 0x1
	v_fma_f32 v74, -v171, v168, v74
	ds_bpermute_b32 v168, v158, v170
	s_wait_dscnt 0x1
	v_fma_f32 v90, -v171, v172, v90
	ds_bpermute_b32 v172, v158, v169
	s_wait_dscnt 0x1
	v_fma_f32 v75, -v171, v168, v75
	ds_bpermute_b32 v168, v159, v170
	s_wait_dscnt 0x1
	v_fma_f32 v91, -v171, v172, v91
	ds_bpermute_b32 v172, v159, v169
	s_wait_dscnt 0x1
	v_fma_f32 v76, -v171, v168, v76
	ds_bpermute_b32 v168, v160, v170
	s_wait_dscnt 0x1
	v_fma_f32 v92, -v171, v172, v92
	ds_bpermute_b32 v172, v160, v169
	s_wait_dscnt 0x1
	v_fma_f32 v77, -v171, v168, v77
	ds_bpermute_b32 v168, v161, v170
	s_wait_dscnt 0x1
	v_fma_f32 v93, -v171, v172, v93
	ds_bpermute_b32 v172, v161, v169
	s_wait_dscnt 0x1
	v_fma_f32 v78, -v171, v168, v78
	ds_bpermute_b32 v168, v162, v170
	s_wait_dscnt 0x1
	v_fma_f32 v94, -v171, v172, v94
	ds_bpermute_b32 v172, v162, v169
	ds_bpermute_b32 v169, v163, v169
	s_wait_dscnt 0x2
	v_fma_f32 v79, -v171, v168, v79
	ds_bpermute_b32 v168, v163, v170
	s_wait_dscnt 0x2
	v_fma_f32 v95, -v171, v172, v95
	s_wait_dscnt 0x1
	v_fma_f32 v96, -v171, v169, v96
	s_wait_dscnt 0x0
	v_fma_f32 v80, -v171, v168, v80
	;;#ASMSTART
	;;#ASMEND
	s_cbranch_scc0 .LBB4_9
; %bb.10:                               ;   in Loop: Header=BB4_8 Depth=1
	v_wmma_f32_16x16x16_f16 v[168:175], v[109:112], v[113:116], v[65:72]
	v_cvt_f16_f32_e32 v104.h, v104
	v_cvt_f16_f32_e32 v104.l, v103
	v_cvt_f16_f32_e32 v103.h, v102
	s_delay_alu instid0(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[168:175], v[109:112], v[117:120], v[168:175]
	v_cvt_f16_f32_e32 v103.l, v101
	v_cvt_f16_f32_e32 v102.h, v100
	v_cvt_f16_f32_e32 v102.l, v99
	v_cvt_f16_f32_e32 v101.h, v98
	v_wmma_f32_16x16x16_f16 v[168:175], v[109:112], v[121:124], v[168:175]
	v_cvt_f16_f32_e32 v101.l, v97
	v_cvt_f16_f32_e32 v96.h, v96
	v_cvt_f16_f32_e32 v96.l, v95
	v_cvt_f16_f32_e32 v95.l, v93
	v_wmma_f32_16x16x16_f16 v[168:175], v[109:112], v[125:128], v[168:175]
	v_cvt_f16_f32_e32 v93.h, v90
	v_cvt_f16_f32_e32 v93.l, v89
	v_cvt_f16_f32_e32 v100.h, v88
	v_cvt_f16_f32_e32 v100.l, v87
	v_wmma_f32_16x16x16_f16 v[168:175], v[109:112], v[129:132], v[168:175]
	v_cvt_f16_f32_e32 v99.h, v86
	v_cvt_f16_f32_e32 v99.l, v85
	v_cvt_f16_f32_e32 v98.h, v84
	v_cvt_f16_f32_e32 v98.l, v83
	v_wmma_f32_16x16x16_f16 v[168:175], v[109:112], v[133:136], v[168:175]
	v_dual_add_f32 v90, v72, v72 :: v_dual_add_f32 v89, v71, v71
	v_dual_add_f32 v88, v70, v70 :: v_dual_add_f32 v87, v69, v69
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_wmma_f32_16x16x16_f16 v[168:175], v[109:112], v[137:140], v[168:175]
	v_dual_add_f32 v86, v68, v68 :: v_dual_add_f32 v85, v67, v67
	v_dual_add_f32 v84, v66, v66 :: v_dual_add_f32 v83, v65, v65
	v_wmma_f32_16x16x16_f16 v[168:175], v[109:112], v[141:144], v[168:175]
	v_cvt_f16_f32_e32 v95.h, v94
	v_cvt_f16_f32_e32 v94.h, v92
	v_cvt_f16_f32_e32 v94.l, v91
	v_wmma_f32_16x16x16_f16 v[83:90], v[109:112], v[113:116], v[83:90]
	v_wmma_f32_16x16x16_f16 v[168:175], v[105:108], v[101:104], v[168:175]
	v_cvt_f16_f32_e32 v97.h, v82
	v_cvt_f16_f32_e32 v97.l, v81
	v_cvt_f16_f32_e64 v179.h, v80
	v_wmma_f32_16x16x16_f16 v[83:90], v[109:112], v[117:120], v[83:90]
	v_wmma_f32_16x16x16_f16 v[168:175], v[105:108], v[93:96], v[168:175]
	v_cvt_f16_f32_e64 v179.l, v79
	v_cvt_f16_f32_e64 v178.h, v78
	v_cvt_f16_f32_e64 v178.l, v77
	v_cvt_f16_f32_e64 v177.h, v76
	v_wmma_f32_16x16x16_f16 v[168:175], v[105:108], v[97:100], v[168:175]
	v_cvt_f16_f32_e64 v177.l, v75
	v_cvt_f16_f32_e64 v176.h, v74
	v_wmma_f32_16x16x16_f16 v[83:90], v[109:112], v[121:124], v[83:90]
	v_cvt_f16_f32_e64 v176.l, v73
	v_dual_mul_f32 v80, 0x40400000, v72 :: v_dual_mul_f32 v79, 0x40400000, v71
	v_dual_mul_f32 v77, 0x40400000, v69 :: v_dual_mul_f32 v76, 0x40400000, v68
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[83:90], v[109:112], v[125:128], v[83:90]
	v_wmma_f32_16x16x16_f16 v[168:175], v[105:108], v[176:179], v[168:175]
	v_dual_mul_f32 v78, 0x40400000, v70 :: v_dual_mul_f32 v75, 0x40400000, v67
	v_mul_f32_e32 v72, 4.0, v72
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[83:90], v[109:112], v[129:132], v[83:90]
	v_dual_add_f32 v73, 0, v168 :: v_dual_mul_f32 v74, 0x40400000, v66
	v_dual_mul_f32 v70, 4.0, v70 :: v_dual_mul_f32 v71, 4.0, v71
	v_mul_f32_e32 v66, 4.0, v66
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[83:90], v[109:112], v[133:136], v[83:90]
	v_add_f32_e32 v81, v73, v169
	v_dual_mul_f32 v73, 0x40400000, v65 :: v_dual_mul_f32 v68, 4.0, v68
	v_dual_mul_f32 v69, 4.0, v69 :: v_dual_mul_f32 v38, 0x3f7d70a4, v38
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[83:90], v[109:112], v[137:140], v[83:90]
	v_add_f32_e32 v81, v81, v170
	s_delay_alu instid0(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[73:80], v[109:112], v[113:116], v[73:80]
	v_mul_f32_e32 v40, 0x3f7d70a4, v40
	v_mul_f32_e32 v34, 0x3f7d70a4, v34
	v_wmma_f32_16x16x16_f16 v[83:90], v[109:112], v[141:144], v[83:90]
	v_add_f32_e32 v81, v81, v171
	v_wmma_f32_16x16x16_f16 v[73:80], v[109:112], v[117:120], v[73:80]
	v_dual_mul_f32 v67, 4.0, v67 :: v_dual_mul_f32 v36, 0x3f7d70a4, v36
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[83:90], v[105:108], v[101:104], v[83:90]
	v_add_f32_e32 v81, v81, v172
	s_delay_alu instid0(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[73:80], v[109:112], v[121:124], v[73:80]
	v_mul_f32_e32 v64, 0x3f7d70a4, v64
	v_mul_f32_e32 v60, 0x3f7d70a4, v60
	v_wmma_f32_16x16x16_f16 v[83:90], v[105:108], v[93:96], v[83:90]
	v_add_f32_e32 v81, v81, v173
	v_wmma_f32_16x16x16_f16 v[73:80], v[109:112], v[125:128], v[73:80]
	v_dual_mul_f32 v65, 4.0, v65 :: v_dual_mul_f32 v62, 0x3f7d70a4, v62
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[83:90], v[105:108], v[97:100], v[83:90]
	v_add_f32_e32 v81, v81, v174
	s_delay_alu instid0(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[73:80], v[109:112], v[129:132], v[73:80]
	v_mul_f32_e32 v56, 0x3f7d70a4, v56
	v_mul_f32_e32 v50, 0x3f7d70a4, v50
	v_wmma_f32_16x16x16_f16 v[83:90], v[105:108], v[176:179], v[83:90]
	v_add_f32_e32 v81, v81, v175
	v_wmma_f32_16x16x16_f16 v[73:80], v[109:112], v[133:136], v[73:80]
	v_wmma_f32_16x16x16_f16 v[65:72], v[109:112], v[113:116], v[65:72]
	v_dual_mul_f32 v46, 0x3f7d70a4, v46 :: v_dual_mul_f32 v39, 0x3f7d70a4, v39
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_mul_f32 v58, 0x3f7d70a4, v58 :: v_dual_add_f32 v81, v81, v83
	v_wmma_f32_16x16x16_f16 v[73:80], v[109:112], v[137:140], v[73:80]
	v_mul_f32_e32 v32, 0x3f7d70a4, v32
	v_wmma_f32_16x16x16_f16 v[65:72], v[109:112], v[117:120], v[65:72]
	v_dual_mul_f32 v37, 0x3f7d70a4, v37 :: v_dual_mul_f32 v54, 0x3f7d70a4, v54
	v_add_f32_e32 v81, v81, v84
	v_wmma_f32_16x16x16_f16 v[73:80], v[109:112], v[141:144], v[73:80]
	v_mul_f32_e32 v28, 0x3f7d70a4, v28
	v_wmma_f32_16x16x16_f16 v[65:72], v[109:112], v[121:124], v[65:72]
	v_dual_mul_f32 v35, 0x3f7d70a4, v35 :: v_dual_mul_f32 v52, 0x3f7d70a4, v52
	v_add_f32_e32 v81, v81, v85
	v_wmma_f32_16x16x16_f16 v[73:80], v[105:108], v[101:104], v[73:80]
	v_mul_f32_e32 v24, 0x3f7d70a4, v24
	v_wmma_f32_16x16x16_f16 v[65:72], v[109:112], v[125:128], v[65:72]
	v_dual_mul_f32 v33, 0x3f7d70a4, v33 :: v_dual_mul_f32 v48, 0x3f7d70a4, v48
	v_add_f32_e32 v81, v81, v86
	v_wmma_f32_16x16x16_f16 v[73:80], v[105:108], v[93:96], v[73:80]
	v_mul_f32_e32 v18, 0x3f7d70a4, v18
	v_wmma_f32_16x16x16_f16 v[65:72], v[109:112], v[129:132], v[65:72]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_mul_f32 v16, 0x3f7d70a4, v16 :: v_dual_add_f32 v81, v81, v87
	v_wmma_f32_16x16x16_f16 v[73:80], v[105:108], v[97:100], v[73:80]
	v_dual_mul_f32 v63, 0x3f7d70a4, v63 :: v_dual_mul_f32 v44, 0x3f7d70a4, v44
	v_mul_f32_e32 v12, 0x3f7d70a4, v12
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_add_f32_e32 v81, v81, v88
	v_wmma_f32_16x16x16_f16 v[73:80], v[105:108], v[176:179], v[73:80]
	v_wmma_f32_16x16x16_f16 v[65:72], v[109:112], v[133:136], v[65:72]
	v_dual_mul_f32 v61, 0x3f7d70a4, v61 :: v_dual_mul_f32 v42, 0x3f7d70a4, v42
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_add_f32 v81, v81, v89 :: v_dual_mul_f32 v6, 0x3f7d70a4, v6
	v_wmma_f32_16x16x16_f16 v[65:72], v[109:112], v[137:140], v[65:72]
	v_dual_mul_f32 v59, 0x3f7d70a4, v59 :: v_dual_mul_f32 v30, 0x3f7d70a4, v30
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_f32_e32 v81, v81, v90
	v_dual_mul_f32 v57, 0x3f7d70a4, v57 :: v_dual_mul_f32 v26, 0x3f7d70a4, v26
	v_wmma_f32_16x16x16_f16 v[65:72], v[109:112], v[141:144], v[65:72]
	v_dual_mul_f32 v55, 0x3f7d70a4, v55 :: v_dual_mul_f32 v22, 0x3f7d70a4, v22
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_f32_e32 v73, v81, v73
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
	v_wmma_f32_16x16x16_f16 v[65:72], v[105:108], v[97:100], v[65:72]
	v_dual_mul_f32 v43, 0x3f7d70a4, v43 :: v_dual_mul_f32 v2, 0x3f7d70a4, v2
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_f32_e32 v73, v73, v76
	v_mul_f32_e32 v41, 0x3f7d70a4, v41
	v_wmma_f32_16x16x16_f16 v[65:72], v[105:108], v[176:179], v[65:72]
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
	v_wmma_f32_16x16x16_f16 v[33:40], v[105:108], v[97:100], v[33:40]
	v_wmma_f32_16x16x16_f16 v[57:64], v[105:108], v[97:100], v[57:64]
	v_add_f32_e32 v65, v65, v69
	v_wmma_f32_16x16x16_f16 v[49:56], v[105:108], v[97:100], v[49:56]
	v_wmma_f32_16x16x16_f16 v[41:48], v[105:108], v[97:100], v[41:48]
	v_wmma_f32_16x16x16_f16 v[25:32], v[105:108], v[97:100], v[25:32]
	v_wmma_f32_16x16x16_f16 v[17:24], v[105:108], v[97:100], v[17:24]
	v_add_f32_e32 v65, v65, v70
	v_wmma_f32_16x16x16_f16 v[9:16], v[105:108], v[97:100], v[9:16]
	v_wmma_f32_16x16x16_f16 v[1:8], v[105:108], v[97:100], v[1:8]
	v_wmma_f32_16x16x16_f16 v[33:40], v[105:108], v[176:179], v[33:40]
	v_wmma_f32_16x16x16_f16 v[57:64], v[105:108], v[176:179], v[57:64]
	v_add_f32_e32 v65, v65, v71
	v_wmma_f32_16x16x16_f16 v[49:56], v[105:108], v[176:179], v[49:56]
	v_wmma_f32_16x16x16_f16 v[41:48], v[105:108], v[176:179], v[41:48]
	v_wmma_f32_16x16x16_f16 v[25:32], v[105:108], v[176:179], v[25:32]
	v_wmma_f32_16x16x16_f16 v[17:24], v[105:108], v[176:179], v[17:24]
	v_add_f32_e32 v65, v65, v72
	v_wmma_f32_16x16x16_f16 v[9:16], v[105:108], v[176:179], v[9:16]
	v_wmma_f32_16x16x16_f16 v[1:8], v[105:108], v[176:179], v[1:8]
	s_add_co_i32 s6, s6, 1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_cmp_lt_f32_e32 vcc_lo, 0x60ad78ec, v65
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s6, s3
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v65, 0x3a83126f, v164, vcc_lo
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
		.amdhsa_next_free_vgpr 189
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
	.set .L_Z5probeILi8ELb0ELb1EEvPfPKfi.num_vgpr, 189
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
; codeLenInByte = 27256
; TotalNumSgprs: 10
; NumVgprs: 189
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 23
; NumSGPRsForWavesPerEU: 10
; NumVGPRsForWavesPerEU: 189
; Occupancy: 8
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
	s_load_b32 s3, s[0:1], 0x10
	s_mov_b32 s0, 0x3c23d70a
	s_mov_b32 s1, 0x38d1b717
	s_mov_b32 s2, 0x399d4951
	s_wait_loadcnt 0xf
	v_fma_mixhi_f16 v45, v10, s0, s1
	s_mov_b32 s1, 0x3951b717
	s_wait_alu depctr_sa_sdst(0)
	v_fma_mixlo_f16 v54, v10, s0, s2
	v_fma_mixhi_f16 v53, v10, s0, s1
	s_mov_b32 s1, 0x39d1b717
	s_mov_b32 s2, 0x3a03126e
	s_wait_alu depctr_sa_sdst(0)
	v_fma_mixhi_f16 v54, v10, s0, s1
	s_mov_b32 s1, 0x3a1d4951
	s_wait_loadcnt_dscnt 0x0
	s_wait_alu depctr_sa_sdst(0)
	v_fma_mixhi_f16 v55, v10, s0, s1
	s_mov_b32 s1, 0x3a51b717
	s_barrier_signal -1
	s_wait_alu depctr_sa_sdst(0)
	v_fma_mixhi_f16 v56, v10, s0, s1
	s_mov_b32 s1, 0x3a6bedfa
	v_fma_mixlo_f16 v55, v10, s0, s2
	s_wait_alu depctr_sa_sdst(0)
	v_fma_mixlo_f16 v60, v10, s0, s1
	s_mov_b32 s1, 0x3a83126e
	s_mov_b32 s2, 0x3a378034
	s_wait_alu depctr_sa_sdst(0)
	v_fma_mixlo_f16 v52, v10, s0, s1
	s_mov_b32 s1, 0x3a9d4951
	v_fma_mixlo_f16 v56, v10, s0, s2
	s_mov_b32 s2, 0x3a902de0
	s_wait_alu depctr_sa_sdst(0)
	v_fma_mixhi_f16 v64, v10, s0, s1
	s_mov_b32 s1, 0x3aaa64c3
	v_fma_mixhi_f16 v52, v10, s0, s2
	s_wait_alu depctr_sa_sdst(0)
	v_fma_mixhi_f16 v68, v10, s0, s1
	s_mov_b32 s1, 0x3ab78034
	v_fma_mixlo_f16 v45, v10, s0, 0
	v_mov_b16_e32 v53.l, v45.h
	v_mov_b16_e32 v46.l, v53.h
	v_mov_b16_e32 v46.h, v54.l
	v_mov_b16_e32 v47.l, v54.h
	v_mov_b16_e32 v47.h, v55.l
	v_mov_b16_e32 v60.h, v52.l
	v_mov_b16_e32 v64.l, v52.h
	v_mov_b16_e32 v48.l, v55.h
	v_mov_b16_e32 v48.h, v56.l
	v_mov_b16_e32 v72.l, v56.h
	v_mov_b16_e32 v72.h, v60.l
	v_mov_b16_e32 v68.l, v64.h
	s_wait_alu depctr_sa_sdst(0)
	v_fma_mixhi_f16 v44, v10, s0, s1
	v_mov_b16_e32 v44.l, v68.h
	s_mov_b32 s6, 0
	s_wait_kmcnt 0x0
	s_cmp_lt_i32 s3, 1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_cbranch_scc1 .LBB5_29
; %bb.25:
	v_mbcnt_lo_u32_b32 v2, -1, 0
	s_mov_b32 s0, 0x3a83126f
	v_dual_mov_b32 v70, v47 :: v_dual_and_b32 v1, 31, v0
	s_wait_alu depctr_sa_sdst(0)
	v_fma_mixlo_f16 v77, v3, s0, 0
	v_fma_mixhi_f16 v73, v4, s0, 0
	v_fma_mixhi_f16 v77, v5, s0, 0
	v_lshrrev_b32_e32 v3, 1, v0
	v_xor_b32_e32 v4, 16, v2
	v_bfi_b32 v5, v2, 0, 32
	v_fma_mixhi_f16 v78, v9, s0, 0
	v_dual_mov_b32 v58, v55 :: v_dual_lshlrev_b32 v9, 7, v0
	v_fma_mixlo_f16 v73, v10, s0, 0
	s_delay_alu instid0(VALU_DEP_4)
	v_cmp_lt_u32_e32 vcc_lo, v4, v5
	v_and_b32_e32 v81, 8, v3
	v_fma_mixlo_f16 v74, v6, s0, 0
	v_fma_mixlo_f16 v78, v7, s0, 0
	v_fma_mixhi_f16 v74, v8, s0, 0
	v_fma_mixhi_f16 v75, v14, s0, 0
	v_fma_mixhi_f16 v79, v15, s0, 0
	v_fma_mixlo_f16 v76, v16, s0, 0
	v_fma_mixlo_f16 v80, v12, s0, 0
	v_fma_mixhi_f16 v76, v13, s0, 0
	v_fma_mixhi_f16 v80, v11, s0, 0
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
	v_dual_mov_b32 v50, v48 :: v_dual_and_b32 v83, 0x800, v9
	v_or_b32_e32 v9, 20, v81
	v_or_b32_e32 v14, 21, v81
	v_or_b32_e32 v15, 22, v81
	v_or_b32_e32 v16, 23, v81
	v_fma_mixlo_f16 v75, v17, s0, 0
	v_fma_mixlo_f16 v79, v18, s0, 0
	v_dual_mov_b32 v69, v46 :: v_dual_add_nc_u32 v84, 0x1800, v83
	v_or_b32_e32 v85, 0x2000, v83
	v_dual_mov_b32 v71, v48 :: v_dual_add_nc_u32 v86, 0xfffff800, v83
	v_dual_mov_b32 v62, v56 :: v_dual_lshlrev_b32 v87, 2, v2
	v_dual_mov_b32 v103, 0x3b03126f :: v_dual_lshlrev_b32 v88, 2, v1
	v_dual_mov_b32 v66, v72 :: v_dual_lshlrev_b32 v89, 2, v3
	v_dual_mov_b32 v57, v54 :: v_dual_lshlrev_b32 v90, 2, v4
	v_dual_mov_b32 v42, v60 :: v_dual_lshlrev_b32 v91, 2, v5
	v_dual_mov_b32 v59, v56 :: v_dual_lshlrev_b32 v92, 2, v6
	v_lshlrev_b32_e32 v93, 2, v7
	v_dual_mov_b32 v49, v47 :: v_dual_lshlrev_b32 v94, 2, v8
	v_lshlrev_b32_e32 v95, 2, v10
	v_dual_mov_b32 v51, v72 :: v_dual_lshlrev_b32 v96, 2, v11
	v_lshlrev_b32_e32 v97, 2, v12
	v_dual_mov_b32 v61, v55 :: v_dual_lshlrev_b32 v98, 2, v13
	v_lshlrev_b32_e32 v99, 2, v9
	v_dual_mov_b32 v63, v60 :: v_dual_lshlrev_b32 v100, 2, v14
	v_lshlrev_b32_e32 v101, 2, v15
	v_dual_mov_b32 v65, v48 :: v_dual_lshlrev_b32 v102, 2, v16
	v_mov_b32_e32 v67, v52
	v_mov_b32_e32 v41, v56
	v_mov_b32_e32 v43, v64
	s_mov_b32 s7, 0x3f7d70a4
.LBB5_26:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB5_27 Depth 2
	v_wmma_f32_16x16x16_f16 v[1:8], v[73:76], v[77:80], 0
	s_mov_b32 s8, 0
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
	v_dual_add_f32 v111, 0x37fba882, v8 :: v_dual_add_f32 v110, 0x37fba882, v7
	v_dual_add_f32 v109, 0x37fba882, v6 :: v_dual_add_f32 v108, 0x37fba882, v5
	v_dual_add_f32 v107, 0x37fba882, v4 :: v_dual_add_f32 v106, 0x37fba882, v3
	v_dual_add_f32 v105, 0x37fba882, v2 :: v_dual_add_f32 v104, 0x37fba882, v1
	v_dual_add_f32 v119, 0x3827c5ac, v8 :: v_dual_add_f32 v118, 0x3827c5ac, v7
	v_dual_add_f32 v117, 0x3827c5ac, v6 :: v_dual_add_f32 v116, 0x3827c5ac, v5
	v_dual_add_f32 v115, 0x3827c5ac, v4 :: v_dual_add_f32 v114, 0x3827c5ac, v3
	v_dual_add_f32 v113, 0x3827c5ac, v2 :: v_dual_add_f32 v112, 0x3827c5ac, v1
	v_wmma_f32_16x16x16_f16 v[9:16], v[73:76], v[45:48], v[9:16]
	v_wmma_f32_16x16x16_f16 v[17:24], v[73:76], v[45:48], v[17:24]
	v_wmma_f32_16x16x16_f16 v[104:111], v[73:76], v[45:48], v[104:111]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[112:119], v[73:76], v[45:48], v[112:119]
	v_wmma_f32_16x16x16_f16 v[9:16], v[73:76], v[53:56], v[9:16]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[17:24], v[73:76], v[53:56], v[17:24]
	v_wmma_f32_16x16x16_f16 v[104:111], v[73:76], v[53:56], v[104:111]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[112:119], v[73:76], v[53:56], v[112:119]
	v_wmma_f32_16x16x16_f16 v[9:16], v[73:76], v[69:72], v[9:16]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[17:24], v[73:76], v[69:72], v[17:24]
	v_wmma_f32_16x16x16_f16 v[104:111], v[73:76], v[69:72], v[104:111]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[112:119], v[73:76], v[69:72], v[112:119]
	v_wmma_f32_16x16x16_f16 v[9:16], v[73:76], v[57:60], v[9:16]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[17:24], v[73:76], v[57:60], v[17:24]
	v_wmma_f32_16x16x16_f16 v[104:111], v[73:76], v[57:60], v[104:111]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[112:119], v[73:76], v[57:60], v[112:119]
	v_wmma_f32_16x16x16_f16 v[9:16], v[73:76], v[49:52], v[9:16]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[17:24], v[73:76], v[49:52], v[17:24]
	v_wmma_f32_16x16x16_f16 v[104:111], v[73:76], v[49:52], v[104:111]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[112:119], v[73:76], v[49:52], v[112:119]
	v_wmma_f32_16x16x16_f16 v[9:16], v[73:76], v[61:64], v[9:16]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[17:24], v[73:76], v[61:64], v[17:24]
	v_wmma_f32_16x16x16_f16 v[104:111], v[73:76], v[61:64], v[104:111]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[112:119], v[73:76], v[61:64], v[112:119]
	v_wmma_f32_16x16x16_f16 v[9:16], v[73:76], v[65:68], v[9:16]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[17:24], v[73:76], v[65:68], v[17:24]
	v_wmma_f32_16x16x16_f16 v[104:111], v[73:76], v[65:68], v[104:111]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[112:119], v[73:76], v[65:68], v[112:119]
	v_wmma_f32_16x16x16_f16 v[9:16], v[73:76], v[41:44], v[9:16]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[17:24], v[73:76], v[41:44], v[17:24]
	v_wmma_f32_16x16x16_f16 v[104:111], v[73:76], v[41:44], v[104:111]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[112:119], v[73:76], v[41:44], v[112:119]
	v_dual_mul_f32 v33, 0x3727c5ac, v9 :: v_dual_mul_f32 v34, 0x3727c5ac, v10
	v_dual_mul_f32 v35, 0x3727c5ac, v11 :: v_dual_mul_f32 v36, 0x3727c5ac, v12
	v_dual_mul_f32 v37, 0x3727c5ac, v13 :: v_dual_mul_f32 v38, 0x3727c5ac, v14
	v_dual_mul_f32 v39, 0x3727c5ac, v15 :: v_dual_mul_f32 v40, 0x3727c5ac, v16
	v_dual_mul_f32 v25, 0x3727c5ac, v17 :: v_dual_mul_f32 v26, 0x3727c5ac, v18
	v_dual_mul_f32 v27, 0x3727c5ac, v19 :: v_dual_mul_f32 v28, 0x3727c5ac, v20
	v_dual_mul_f32 v29, 0x3727c5ac, v21 :: v_dual_mul_f32 v30, 0x3727c5ac, v22
	v_dual_mul_f32 v31, 0x3727c5ac, v23 :: v_dual_mul_f32 v32, 0x3727c5ac, v24
	v_dual_mul_f32 v17, 0x3727c5ac, v104 :: v_dual_mul_f32 v18, 0x3727c5ac, v105
	v_dual_mul_f32 v19, 0x3727c5ac, v106 :: v_dual_mul_f32 v20, 0x3727c5ac, v107
	v_dual_mul_f32 v21, 0x3727c5ac, v108 :: v_dual_mul_f32 v22, 0x3727c5ac, v109
	v_dual_mul_f32 v23, 0x3727c5ac, v110 :: v_dual_mul_f32 v24, 0x3727c5ac, v111
	v_dual_mul_f32 v9, 0x3727c5ac, v112 :: v_dual_mul_f32 v10, 0x3727c5ac, v113
	v_dual_mul_f32 v11, 0x3727c5ac, v114 :: v_dual_mul_f32 v12, 0x3727c5ac, v115
	v_dual_mul_f32 v13, 0x3727c5ac, v116 :: v_dual_mul_f32 v14, 0x3727c5ac, v117
	v_dual_mul_f32 v15, 0x3727c5ac, v118 :: v_dual_mul_f32 v16, 0x3727c5ac, v119
.LBB5_27:                               ;   Parent Loop BB5_26 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_wait_alu depctr_sa_sdst(0)
	v_cmp_eq_u32_e64 s1, s8, v83
	v_cmp_eq_u32_e64 s2, s8, v85
	v_cmp_eq_u32_e32 vcc_lo, s8, v86
	v_cmp_eq_u32_e64 s0, s8, v84
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v104, 0, v33, s1
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v104, v104, v17, s2
	ds_bpermute_b32 v105, v87, v104
	s_wait_dscnt 0x0
	v_dual_add_f32 v108, v104, v105 :: v_dual_add_nc_u32 v105, s8, v82
	v_lshlrev_b32_e32 v104, 2, v81
	s_addk_co_i32 s8, 0x2000
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s8, 0x4000
	ds_load_2addr_b32 v[106:107], v105 offset1:32
	s_wait_dscnt 0x0
	ds_bpermute_b32 v109, v104, v106
	s_wait_dscnt 0x0
	v_fma_f32 v33, -v108, v109, v33
	ds_bpermute_b32 v109, v88, v106
	s_wait_dscnt 0x0
	v_fma_f32 v34, -v108, v109, v34
	ds_bpermute_b32 v109, v89, v106
	s_wait_dscnt 0x0
	v_fma_f32 v35, -v108, v109, v35
	ds_bpermute_b32 v109, v90, v106
	s_wait_dscnt 0x0
	v_fma_f32 v36, -v108, v109, v36
	ds_bpermute_b32 v109, v91, v106
	s_wait_dscnt 0x0
	v_fma_f32 v37, -v108, v109, v37
	ds_bpermute_b32 v109, v92, v106
	s_wait_dscnt 0x0
	v_fma_f32 v38, -v108, v109, v38
	ds_bpermute_b32 v109, v93, v106
	s_wait_dscnt 0x0
	v_fma_f32 v39, -v108, v109, v39
	ds_bpermute_b32 v109, v94, v106
	s_wait_dscnt 0x0
	v_fma_f32 v40, -v108, v109, v40
	ds_bpermute_b32 v109, v95, v106
	s_wait_dscnt 0x0
	v_fma_f32 v25, -v108, v109, v25
	ds_bpermute_b32 v109, v96, v106
	s_wait_dscnt 0x0
	v_fma_f32 v26, -v108, v109, v26
	ds_bpermute_b32 v109, v97, v106
	s_wait_dscnt 0x0
	v_fma_f32 v27, -v108, v109, v27
	ds_bpermute_b32 v109, v98, v106
	s_wait_dscnt 0x0
	v_fma_f32 v28, -v108, v109, v28
	ds_bpermute_b32 v109, v99, v106
	s_wait_dscnt 0x0
	v_fma_f32 v29, -v108, v109, v29
	ds_bpermute_b32 v109, v100, v106
	s_wait_dscnt 0x0
	v_fma_f32 v30, -v108, v109, v30
	ds_bpermute_b32 v109, v101, v106
	ds_bpermute_b32 v106, v102, v106
	s_wait_dscnt 0x1
	v_fma_f32 v31, -v108, v109, v31
	s_wait_dscnt 0x0
	v_fma_f32 v32, -v108, v106, v32
	ds_bpermute_b32 v106, v104, v107
	s_wait_dscnt 0x0
	v_fma_f32 v17, -v108, v106, v17
	ds_bpermute_b32 v106, v88, v107
	s_wait_dscnt 0x0
	v_fma_f32 v18, -v108, v106, v18
	ds_bpermute_b32 v106, v89, v107
	s_wait_dscnt 0x0
	v_fma_f32 v19, -v108, v106, v19
	ds_bpermute_b32 v106, v90, v107
	s_wait_dscnt 0x0
	v_fma_f32 v20, -v108, v106, v20
	ds_bpermute_b32 v106, v91, v107
	s_wait_dscnt 0x0
	v_fma_f32 v21, -v108, v106, v21
	ds_bpermute_b32 v106, v92, v107
	s_wait_dscnt 0x0
	v_fma_f32 v22, -v108, v106, v22
	ds_bpermute_b32 v106, v93, v107
	s_wait_dscnt 0x0
	v_fma_f32 v23, -v108, v106, v23
	ds_bpermute_b32 v106, v94, v107
	s_wait_dscnt 0x0
	v_fma_f32 v24, -v108, v106, v24
	ds_bpermute_b32 v106, v95, v107
	s_wait_dscnt 0x0
	v_fma_f32 v9, -v108, v106, v9
	ds_bpermute_b32 v106, v96, v107
	s_wait_dscnt 0x0
	v_fma_f32 v10, -v108, v106, v10
	ds_bpermute_b32 v106, v97, v107
	s_wait_dscnt 0x0
	v_fma_f32 v11, -v108, v106, v11
	ds_bpermute_b32 v106, v98, v107
	s_wait_dscnt 0x0
	v_fma_f32 v12, -v108, v106, v12
	ds_bpermute_b32 v106, v99, v107
	s_wait_dscnt 0x0
	v_fma_f32 v13, -v108, v106, v13
	ds_bpermute_b32 v106, v100, v107
	s_wait_dscnt 0x0
	v_fma_f32 v14, -v108, v106, v14
	ds_bpermute_b32 v106, v101, v107
	s_wait_dscnt 0x0
	v_fma_f32 v15, -v108, v106, v15
	ds_bpermute_b32 v106, v102, v107
	s_wait_dscnt 0x0
	v_fma_f32 v16, -v108, v106, v16
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v106, 0, v34, s1
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v106, v106, v18, s2
	ds_bpermute_b32 v107, v87, v106
	s_wait_dscnt 0x0
	v_add_f32_e32 v108, v106, v107
	ds_load_2addr_b32 v[106:107], v105 offset0:64 offset1:96
	s_wait_dscnt 0x0
	ds_bpermute_b32 v109, v104, v106
	s_wait_dscnt 0x0
	v_fma_f32 v33, -v108, v109, v33
	ds_bpermute_b32 v109, v88, v106
	s_wait_dscnt 0x0
	v_fma_f32 v34, -v108, v109, v34
	ds_bpermute_b32 v109, v89, v106
	s_wait_dscnt 0x0
	v_fma_f32 v35, -v108, v109, v35
	ds_bpermute_b32 v109, v90, v106
	s_wait_dscnt 0x0
	v_fma_f32 v36, -v108, v109, v36
	ds_bpermute_b32 v109, v91, v106
	s_wait_dscnt 0x0
	v_fma_f32 v37, -v108, v109, v37
	ds_bpermute_b32 v109, v92, v106
	s_wait_dscnt 0x0
	v_fma_f32 v38, -v108, v109, v38
	ds_bpermute_b32 v109, v93, v106
	s_wait_dscnt 0x0
	v_fma_f32 v39, -v108, v109, v39
	ds_bpermute_b32 v109, v94, v106
	s_wait_dscnt 0x0
	v_fma_f32 v40, -v108, v109, v40
	ds_bpermute_b32 v109, v95, v106
	s_wait_dscnt 0x0
	v_fma_f32 v25, -v108, v109, v25
	ds_bpermute_b32 v109, v96, v106
	s_wait_dscnt 0x0
	v_fma_f32 v26, -v108, v109, v26
	ds_bpermute_b32 v109, v97, v106
	s_wait_dscnt 0x0
	v_fma_f32 v27, -v108, v109, v27
	ds_bpermute_b32 v109, v98, v106
	s_wait_dscnt 0x0
	v_fma_f32 v28, -v108, v109, v28
	ds_bpermute_b32 v109, v99, v106
	s_wait_dscnt 0x0
	v_fma_f32 v29, -v108, v109, v29
	ds_bpermute_b32 v109, v100, v106
	s_wait_dscnt 0x0
	v_fma_f32 v30, -v108, v109, v30
	ds_bpermute_b32 v109, v101, v106
	ds_bpermute_b32 v106, v102, v106
	s_wait_dscnt 0x1
	v_fma_f32 v31, -v108, v109, v31
	s_wait_dscnt 0x0
	v_fma_f32 v32, -v108, v106, v32
	ds_bpermute_b32 v106, v104, v107
	s_wait_dscnt 0x0
	v_fma_f32 v17, -v108, v106, v17
	ds_bpermute_b32 v106, v88, v107
	s_wait_dscnt 0x0
	v_fma_f32 v18, -v108, v106, v18
	ds_bpermute_b32 v106, v89, v107
	s_wait_dscnt 0x0
	v_fma_f32 v19, -v108, v106, v19
	ds_bpermute_b32 v106, v90, v107
	s_wait_dscnt 0x0
	v_fma_f32 v20, -v108, v106, v20
	ds_bpermute_b32 v106, v91, v107
	s_wait_dscnt 0x0
	v_fma_f32 v21, -v108, v106, v21
	ds_bpermute_b32 v106, v92, v107
	s_wait_dscnt 0x0
	v_fma_f32 v22, -v108, v106, v22
	ds_bpermute_b32 v106, v93, v107
	s_wait_dscnt 0x0
	v_fma_f32 v23, -v108, v106, v23
	ds_bpermute_b32 v106, v94, v107
	s_wait_dscnt 0x0
	v_fma_f32 v24, -v108, v106, v24
	ds_bpermute_b32 v106, v95, v107
	s_wait_dscnt 0x0
	v_fma_f32 v9, -v108, v106, v9
	ds_bpermute_b32 v106, v96, v107
	s_wait_dscnt 0x0
	v_fma_f32 v10, -v108, v106, v10
	ds_bpermute_b32 v106, v97, v107
	s_wait_dscnt 0x0
	v_fma_f32 v11, -v108, v106, v11
	ds_bpermute_b32 v106, v98, v107
	s_wait_dscnt 0x0
	v_fma_f32 v12, -v108, v106, v12
	ds_bpermute_b32 v106, v99, v107
	s_wait_dscnt 0x0
	v_fma_f32 v13, -v108, v106, v13
	ds_bpermute_b32 v106, v100, v107
	s_wait_dscnt 0x0
	v_fma_f32 v14, -v108, v106, v14
	ds_bpermute_b32 v106, v101, v107
	s_wait_dscnt 0x0
	v_fma_f32 v15, -v108, v106, v15
	ds_bpermute_b32 v106, v102, v107
	s_wait_dscnt 0x0
	v_fma_f32 v16, -v108, v106, v16
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v106, 0, v35, s1
	ds_load_2addr_b32 v[120:121], v105 offset0:128 offset1:160
	v_cndmask_b32_e64 v106, v106, v19, s2
	ds_bpermute_b32 v107, v87, v106
	s_wait_dscnt 0x0
	v_add_f32_e32 v122, v106, v107
	ds_bpermute_b32 v106, v104, v120
	s_wait_dscnt 0x0
	v_fma_f32 v33, -v122, v106, v33
	ds_bpermute_b32 v106, v88, v120
	s_wait_dscnt 0x0
	v_fma_f32 v34, -v122, v106, v34
	ds_bpermute_b32 v106, v89, v120
	s_wait_dscnt 0x0
	v_fma_f32 v35, -v122, v106, v35
	ds_bpermute_b32 v106, v90, v120
	s_wait_dscnt 0x0
	v_fma_f32 v36, -v122, v106, v36
	ds_bpermute_b32 v106, v91, v120
	s_wait_dscnt 0x0
	v_fma_f32 v37, -v122, v106, v37
	ds_bpermute_b32 v106, v92, v120
	s_wait_dscnt 0x0
	v_fma_f32 v38, -v122, v106, v38
	ds_bpermute_b32 v106, v93, v120
	s_wait_dscnt 0x0
	v_fma_f32 v39, -v122, v106, v39
	ds_bpermute_b32 v106, v94, v120
	s_wait_dscnt 0x0
	v_fma_f32 v40, -v122, v106, v40
	ds_bpermute_b32 v106, v95, v120
	s_wait_dscnt 0x0
	v_fma_f32 v25, -v122, v106, v25
	ds_bpermute_b32 v106, v96, v120
	s_wait_dscnt 0x0
	v_fma_f32 v26, -v122, v106, v26
	ds_bpermute_b32 v106, v97, v120
	s_wait_dscnt 0x0
	v_fma_f32 v27, -v122, v106, v27
	ds_bpermute_b32 v106, v98, v120
	s_wait_dscnt 0x0
	v_fma_f32 v28, -v122, v106, v28
	ds_bpermute_b32 v106, v99, v120
	s_wait_dscnt 0x0
	v_fma_f32 v29, -v122, v106, v29
	ds_bpermute_b32 v106, v100, v120
	s_wait_dscnt 0x0
	v_fma_f32 v30, -v122, v106, v30
	ds_bpermute_b32 v106, v101, v120
	s_wait_dscnt 0x0
	v_fma_f32 v31, -v122, v106, v31
	ds_bpermute_b32 v106, v102, v120
	s_wait_dscnt 0x0
	v_fma_f32 v32, -v122, v106, v32
	ds_bpermute_b32 v106, v104, v121
	s_wait_dscnt 0x0
	v_fma_f32 v106, -v122, v106, v17
	ds_bpermute_b32 v17, v88, v121
	s_wait_dscnt 0x0
	v_fma_f32 v107, -v122, v17, v18
	ds_bpermute_b32 v17, v89, v121
	s_wait_dscnt 0x0
	v_fma_f32 v108, -v122, v17, v19
	ds_bpermute_b32 v17, v90, v121
	s_wait_dscnt 0x0
	v_fma_f32 v109, -v122, v17, v20
	ds_bpermute_b32 v17, v91, v121
	s_wait_dscnt 0x0
	v_fma_f32 v110, -v122, v17, v21
	ds_bpermute_b32 v17, v92, v121
	s_wait_dscnt 0x0
	v_fma_f32 v111, -v122, v17, v22
	ds_bpermute_b32 v17, v93, v121
	s_wait_dscnt 0x0
	v_fma_f32 v112, -v122, v17, v23
	ds_bpermute_b32 v17, v94, v121
	s_wait_dscnt 0x0
	v_fma_f32 v113, -v122, v17, v24
	ds_bpermute_b32 v17, v95, v121
	s_wait_dscnt 0x0
	v_fma_f32 v114, -v122, v17, v9
	ds_bpermute_b32 v9, v96, v121
	s_wait_dscnt 0x0
	v_fma_f32 v115, -v122, v9, v10
	ds_bpermute_b32 v9, v97, v121
	s_wait_dscnt 0x0
	v_fma_f32 v116, -v122, v9, v11
	ds_bpermute_b32 v9, v98, v121
	s_wait_dscnt 0x0
	v_fma_f32 v117, -v122, v9, v12
	ds_bpermute_b32 v9, v99, v121
	s_wait_dscnt 0x0
	v_fma_f32 v118, -v122, v9, v13
	ds_bpermute_b32 v9, v100, v121
	s_wait_dscnt 0x0
	v_fma_f32 v119, -v122, v9, v14
	ds_bpermute_b32 v9, v101, v121
	s_wait_dscnt 0x0
	v_fma_f32 v120, -v122, v9, v15
	ds_bpermute_b32 v9, v102, v121
	s_wait_dscnt 0x0
	v_fma_f32 v121, -v122, v9, v16
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v9, 0, v36, s1
	ds_load_2addr_b32 v[122:123], v105 offset0:192 offset1:224
	v_cndmask_b32_e64 v9, v9, v109, s2
	ds_bpermute_b32 v10, v87, v9
	s_wait_dscnt 0x1
	ds_bpermute_b32 v11, v89, v122
	ds_bpermute_b32 v12, v90, v122
	ds_bpermute_b32 v13, v91, v122
	ds_bpermute_b32 v14, v92, v122
	ds_bpermute_b32 v15, v93, v122
	ds_bpermute_b32 v16, v94, v122
	ds_bpermute_b32 v17, v95, v122
	ds_bpermute_b32 v18, v96, v122
	ds_bpermute_b32 v19, v97, v122
	s_wait_dscnt 0x9
	v_add_f32_e32 v124, v9, v10
	ds_bpermute_b32 v9, v104, v122
	ds_bpermute_b32 v10, v88, v122
	ds_bpermute_b32 v20, v98, v122
	ds_bpermute_b32 v21, v99, v122
	ds_bpermute_b32 v22, v100, v122
	ds_bpermute_b32 v23, v101, v122
	ds_bpermute_b32 v24, v102, v122
	v_add_nc_u32_e32 v122, 0x1800, v105
	s_wait_dscnt 0xf
	v_fma_f32 v11, -v124, v11, v35
	s_wait_dscnt 0xe
	v_fma_f32 v12, -v124, v12, v36
	s_wait_dscnt 0xd
	v_fma_f32 v13, -v124, v13, v37
	s_wait_dscnt 0xc
	v_fma_f32 v14, -v124, v14, v38
	s_wait_dscnt 0xb
	v_fma_f32 v15, -v124, v15, v39
	s_wait_dscnt 0xa
	v_fma_f32 v16, -v124, v16, v40
	s_wait_dscnt 0x9
	v_fma_f32 v17, -v124, v17, v25
	s_wait_dscnt 0x8
	v_fma_f32 v18, -v124, v18, v26
	s_wait_dscnt 0x7
	v_fma_f32 v19, -v124, v19, v27
	ds_bpermute_b32 v25, v104, v123
	s_wait_dscnt 0x7
	v_fma_f32 v9, -v124, v9, v33
	s_wait_dscnt 0x6
	v_fma_f32 v10, -v124, v10, v34
	s_wait_dscnt 0x5
	v_fma_f32 v20, -v124, v20, v28
	s_wait_dscnt 0x4
	v_fma_f32 v21, -v124, v21, v29
	s_wait_dscnt 0x3
	v_fma_f32 v22, -v124, v22, v30
	s_wait_dscnt 0x2
	v_fma_f32 v23, -v124, v23, v31
	s_wait_dscnt 0x1
	v_fma_f32 v24, -v124, v24, v32
	ds_bpermute_b32 v26, v88, v123
	ds_bpermute_b32 v27, v89, v123
	ds_bpermute_b32 v28, v90, v123
	ds_bpermute_b32 v29, v91, v123
	ds_bpermute_b32 v30, v92, v123
	ds_bpermute_b32 v31, v93, v123
	ds_bpermute_b32 v32, v94, v123
	ds_bpermute_b32 v33, v95, v123
	ds_bpermute_b32 v34, v96, v123
	ds_bpermute_b32 v35, v97, v123
	ds_bpermute_b32 v36, v98, v123
	ds_bpermute_b32 v37, v99, v123
	ds_bpermute_b32 v38, v100, v123
	ds_bpermute_b32 v39, v101, v123
	ds_bpermute_b32 v40, v102, v123
	s_wait_dscnt 0xf
	v_fma_f32 v25, -v124, v25, v106
	s_wait_dscnt 0xe
	v_fma_f32 v26, -v124, v26, v107
	s_wait_dscnt 0xd
	v_fma_f32 v27, -v124, v27, v108
	s_wait_dscnt 0xc
	v_fma_f32 v28, -v124, v28, v109
	s_wait_dscnt 0xb
	v_fma_f32 v29, -v124, v29, v110
	s_wait_dscnt 0xa
	v_fma_f32 v30, -v124, v30, v111
	s_wait_dscnt 0x9
	v_fma_f32 v31, -v124, v31, v112
	s_wait_dscnt 0x8
	v_fma_f32 v32, -v124, v32, v113
	s_wait_dscnt 0x7
	v_fma_f32 v33, -v124, v33, v114
	s_wait_dscnt 0x6
	v_fma_f32 v34, -v124, v34, v115
	s_wait_dscnt 0x5
	v_fma_f32 v35, -v124, v35, v116
	s_wait_dscnt 0x4
	v_fma_f32 v36, -v124, v36, v117
	s_wait_dscnt 0x3
	v_fma_f32 v37, -v124, v37, v118
	s_wait_dscnt 0x2
	v_fma_f32 v38, -v124, v38, v119
	s_wait_dscnt 0x1
	v_fma_f32 v39, -v124, v39, v120
	s_wait_dscnt 0x0
	v_fma_f32 v40, -v124, v40, v121
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v106, 0, v13, s1
	v_add_nc_u32_e32 v109, 0x400, v105
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e64 v106, v106, v29, s2
	ds_bpermute_b32 v107, v87, v106
	s_wait_dscnt 0x0
	v_add_f32_e32 v108, v106, v107
	ds_load_2addr_b32 v[106:107], v109 offset1:32
	s_wait_dscnt 0x0
	ds_bpermute_b32 v110, v104, v106
	s_wait_dscnt 0x0
	v_fma_f32 v9, -v108, v110, v9
	ds_bpermute_b32 v110, v88, v106
	s_wait_dscnt 0x0
	v_fma_f32 v10, -v108, v110, v10
	ds_bpermute_b32 v110, v89, v106
	s_wait_dscnt 0x0
	v_fma_f32 v11, -v108, v110, v11
	ds_bpermute_b32 v110, v90, v106
	s_wait_dscnt 0x0
	v_fma_f32 v12, -v108, v110, v12
	ds_bpermute_b32 v110, v91, v106
	s_wait_dscnt 0x0
	v_fma_f32 v13, -v108, v110, v13
	ds_bpermute_b32 v110, v92, v106
	s_wait_dscnt 0x0
	v_fma_f32 v14, -v108, v110, v14
	ds_bpermute_b32 v110, v93, v106
	s_wait_dscnt 0x0
	v_fma_f32 v15, -v108, v110, v15
	ds_bpermute_b32 v110, v94, v106
	s_wait_dscnt 0x0
	v_fma_f32 v16, -v108, v110, v16
	ds_bpermute_b32 v110, v95, v106
	s_wait_dscnt 0x0
	v_fma_f32 v17, -v108, v110, v17
	ds_bpermute_b32 v110, v96, v106
	s_wait_dscnt 0x0
	v_fma_f32 v18, -v108, v110, v18
	ds_bpermute_b32 v110, v97, v106
	s_wait_dscnt 0x0
	v_fma_f32 v19, -v108, v110, v19
	ds_bpermute_b32 v110, v98, v106
	s_wait_dscnt 0x0
	v_fma_f32 v20, -v108, v110, v20
	ds_bpermute_b32 v110, v99, v106
	s_wait_dscnt 0x0
	v_fma_f32 v21, -v108, v110, v21
	ds_bpermute_b32 v110, v100, v106
	s_wait_dscnt 0x0
	v_fma_f32 v22, -v108, v110, v22
	ds_bpermute_b32 v110, v101, v106
	ds_bpermute_b32 v106, v102, v106
	s_wait_dscnt 0x1
	v_fma_f32 v23, -v108, v110, v23
	s_wait_dscnt 0x0
	v_fma_f32 v24, -v108, v106, v24
	ds_bpermute_b32 v106, v104, v107
	s_wait_dscnt 0x0
	v_fma_f32 v25, -v108, v106, v25
	ds_bpermute_b32 v106, v88, v107
	s_wait_dscnt 0x0
	v_fma_f32 v26, -v108, v106, v26
	ds_bpermute_b32 v106, v89, v107
	s_wait_dscnt 0x0
	v_fma_f32 v27, -v108, v106, v27
	ds_bpermute_b32 v106, v90, v107
	s_wait_dscnt 0x0
	v_fma_f32 v28, -v108, v106, v28
	ds_bpermute_b32 v106, v91, v107
	s_wait_dscnt 0x0
	v_fma_f32 v29, -v108, v106, v29
	ds_bpermute_b32 v106, v92, v107
	s_wait_dscnt 0x0
	v_fma_f32 v30, -v108, v106, v30
	ds_bpermute_b32 v106, v93, v107
	s_wait_dscnt 0x0
	v_fma_f32 v31, -v108, v106, v31
	ds_bpermute_b32 v106, v94, v107
	s_wait_dscnt 0x0
	v_fma_f32 v32, -v108, v106, v32
	ds_bpermute_b32 v106, v95, v107
	s_wait_dscnt 0x0
	v_fma_f32 v33, -v108, v106, v33
	ds_bpermute_b32 v106, v96, v107
	s_wait_dscnt 0x0
	v_fma_f32 v34, -v108, v106, v34
	ds_bpermute_b32 v106, v97, v107
	s_wait_dscnt 0x0
	v_fma_f32 v35, -v108, v106, v35
	ds_bpermute_b32 v106, v98, v107
	s_wait_dscnt 0x0
	v_fma_f32 v36, -v108, v106, v36
	ds_bpermute_b32 v106, v99, v107
	s_wait_dscnt 0x0
	v_fma_f32 v37, -v108, v106, v37
	ds_bpermute_b32 v106, v100, v107
	s_wait_dscnt 0x0
	v_fma_f32 v38, -v108, v106, v38
	ds_bpermute_b32 v106, v101, v107
	s_wait_dscnt 0x0
	v_fma_f32 v39, -v108, v106, v39
	ds_bpermute_b32 v106, v102, v107
	s_wait_dscnt 0x0
	v_fma_f32 v40, -v108, v106, v40
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v106, 0, v14, s1
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v106, v106, v30, s2
	ds_bpermute_b32 v107, v87, v106
	s_wait_dscnt 0x0
	v_add_f32_e32 v108, v106, v107
	ds_load_2addr_b32 v[106:107], v109 offset0:64 offset1:96
	s_wait_dscnt 0x0
	ds_bpermute_b32 v110, v104, v106
	s_wait_dscnt 0x0
	v_fma_f32 v9, -v108, v110, v9
	ds_bpermute_b32 v110, v88, v106
	s_wait_dscnt 0x0
	v_fma_f32 v10, -v108, v110, v10
	ds_bpermute_b32 v110, v89, v106
	s_wait_dscnt 0x0
	v_fma_f32 v11, -v108, v110, v11
	ds_bpermute_b32 v110, v90, v106
	s_wait_dscnt 0x0
	v_fma_f32 v12, -v108, v110, v12
	ds_bpermute_b32 v110, v91, v106
	s_wait_dscnt 0x0
	v_fma_f32 v13, -v108, v110, v13
	ds_bpermute_b32 v110, v92, v106
	s_wait_dscnt 0x0
	v_fma_f32 v14, -v108, v110, v14
	ds_bpermute_b32 v110, v93, v106
	s_wait_dscnt 0x0
	v_fma_f32 v15, -v108, v110, v15
	ds_bpermute_b32 v110, v94, v106
	s_wait_dscnt 0x0
	v_fma_f32 v16, -v108, v110, v16
	ds_bpermute_b32 v110, v95, v106
	s_wait_dscnt 0x0
	v_fma_f32 v17, -v108, v110, v17
	ds_bpermute_b32 v110, v96, v106
	s_wait_dscnt 0x0
	v_fma_f32 v18, -v108, v110, v18
	ds_bpermute_b32 v110, v97, v106
	s_wait_dscnt 0x0
	v_fma_f32 v19, -v108, v110, v19
	ds_bpermute_b32 v110, v98, v106
	s_wait_dscnt 0x0
	v_fma_f32 v20, -v108, v110, v20
	ds_bpermute_b32 v110, v99, v106
	s_wait_dscnt 0x0
	v_fma_f32 v21, -v108, v110, v21
	ds_bpermute_b32 v110, v100, v106
	s_wait_dscnt 0x0
	v_fma_f32 v22, -v108, v110, v22
	ds_bpermute_b32 v110, v101, v106
	ds_bpermute_b32 v106, v102, v106
	s_wait_dscnt 0x1
	v_fma_f32 v23, -v108, v110, v23
	s_wait_dscnt 0x0
	v_fma_f32 v24, -v108, v106, v24
	ds_bpermute_b32 v106, v104, v107
	s_wait_dscnt 0x0
	v_fma_f32 v25, -v108, v106, v25
	ds_bpermute_b32 v106, v88, v107
	s_wait_dscnt 0x0
	v_fma_f32 v26, -v108, v106, v26
	ds_bpermute_b32 v106, v89, v107
	s_wait_dscnt 0x0
	v_fma_f32 v27, -v108, v106, v27
	ds_bpermute_b32 v106, v90, v107
	s_wait_dscnt 0x0
	v_fma_f32 v28, -v108, v106, v28
	ds_bpermute_b32 v106, v91, v107
	s_wait_dscnt 0x0
	v_fma_f32 v29, -v108, v106, v29
	ds_bpermute_b32 v106, v92, v107
	s_wait_dscnt 0x0
	v_fma_f32 v30, -v108, v106, v30
	ds_bpermute_b32 v106, v93, v107
	s_wait_dscnt 0x0
	v_fma_f32 v31, -v108, v106, v31
	ds_bpermute_b32 v106, v94, v107
	s_wait_dscnt 0x0
	v_fma_f32 v32, -v108, v106, v32
	ds_bpermute_b32 v106, v95, v107
	s_wait_dscnt 0x0
	v_fma_f32 v33, -v108, v106, v33
	ds_bpermute_b32 v106, v96, v107
	s_wait_dscnt 0x0
	v_fma_f32 v34, -v108, v106, v34
	ds_bpermute_b32 v106, v97, v107
	s_wait_dscnt 0x0
	v_fma_f32 v35, -v108, v106, v35
	ds_bpermute_b32 v106, v98, v107
	s_wait_dscnt 0x0
	v_fma_f32 v36, -v108, v106, v36
	ds_bpermute_b32 v106, v99, v107
	s_wait_dscnt 0x0
	v_fma_f32 v37, -v108, v106, v37
	ds_bpermute_b32 v106, v100, v107
	s_wait_dscnt 0x0
	v_fma_f32 v38, -v108, v106, v38
	ds_bpermute_b32 v106, v101, v107
	s_wait_dscnt 0x0
	v_fma_f32 v39, -v108, v106, v39
	ds_bpermute_b32 v106, v102, v107
	s_wait_dscnt 0x0
	v_fma_f32 v40, -v108, v106, v40
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v106, 0, v15, s1
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v106, v106, v31, s2
	ds_bpermute_b32 v107, v87, v106
	s_wait_dscnt 0x0
	v_add_f32_e32 v108, v106, v107
	ds_load_2addr_b32 v[106:107], v109 offset0:128 offset1:160
	s_wait_dscnt 0x0
	ds_bpermute_b32 v110, v104, v106
	s_wait_dscnt 0x0
	v_fma_f32 v9, -v108, v110, v9
	ds_bpermute_b32 v110, v88, v106
	s_wait_dscnt 0x0
	v_fma_f32 v10, -v108, v110, v10
	ds_bpermute_b32 v110, v89, v106
	s_wait_dscnt 0x0
	v_fma_f32 v11, -v108, v110, v11
	ds_bpermute_b32 v110, v90, v106
	s_wait_dscnt 0x0
	v_fma_f32 v12, -v108, v110, v12
	ds_bpermute_b32 v110, v91, v106
	s_wait_dscnt 0x0
	v_fma_f32 v13, -v108, v110, v13
	ds_bpermute_b32 v110, v92, v106
	s_wait_dscnt 0x0
	v_fma_f32 v14, -v108, v110, v14
	ds_bpermute_b32 v110, v93, v106
	s_wait_dscnt 0x0
	v_fma_f32 v15, -v108, v110, v15
	ds_bpermute_b32 v110, v94, v106
	s_wait_dscnt 0x0
	v_fma_f32 v16, -v108, v110, v16
	ds_bpermute_b32 v110, v95, v106
	s_wait_dscnt 0x0
	v_fma_f32 v17, -v108, v110, v17
	ds_bpermute_b32 v110, v96, v106
	s_wait_dscnt 0x0
	v_fma_f32 v18, -v108, v110, v18
	ds_bpermute_b32 v110, v97, v106
	s_wait_dscnt 0x0
	v_fma_f32 v19, -v108, v110, v19
	ds_bpermute_b32 v110, v98, v106
	s_wait_dscnt 0x0
	v_fma_f32 v20, -v108, v110, v20
	ds_bpermute_b32 v110, v99, v106
	s_wait_dscnt 0x0
	v_fma_f32 v21, -v108, v110, v21
	ds_bpermute_b32 v110, v100, v106
	s_wait_dscnt 0x0
	v_fma_f32 v22, -v108, v110, v22
	ds_bpermute_b32 v110, v101, v106
	ds_bpermute_b32 v106, v102, v106
	s_wait_dscnt 0x1
	v_fma_f32 v23, -v108, v110, v23
	s_wait_dscnt 0x0
	v_fma_f32 v24, -v108, v106, v24
	ds_bpermute_b32 v106, v104, v107
	s_wait_dscnt 0x0
	v_fma_f32 v25, -v108, v106, v25
	ds_bpermute_b32 v106, v88, v107
	s_wait_dscnt 0x0
	v_fma_f32 v26, -v108, v106, v26
	ds_bpermute_b32 v106, v89, v107
	s_wait_dscnt 0x0
	v_fma_f32 v27, -v108, v106, v27
	ds_bpermute_b32 v106, v90, v107
	s_wait_dscnt 0x0
	v_fma_f32 v28, -v108, v106, v28
	ds_bpermute_b32 v106, v91, v107
	s_wait_dscnt 0x0
	v_fma_f32 v29, -v108, v106, v29
	ds_bpermute_b32 v106, v92, v107
	s_wait_dscnt 0x0
	v_fma_f32 v30, -v108, v106, v30
	ds_bpermute_b32 v106, v93, v107
	s_wait_dscnt 0x0
	v_fma_f32 v31, -v108, v106, v31
	ds_bpermute_b32 v106, v94, v107
	s_wait_dscnt 0x0
	v_fma_f32 v32, -v108, v106, v32
	ds_bpermute_b32 v106, v95, v107
	s_wait_dscnt 0x0
	v_fma_f32 v33, -v108, v106, v33
	ds_bpermute_b32 v106, v96, v107
	s_wait_dscnt 0x0
	v_fma_f32 v34, -v108, v106, v34
	ds_bpermute_b32 v106, v97, v107
	s_wait_dscnt 0x0
	v_fma_f32 v35, -v108, v106, v35
	ds_bpermute_b32 v106, v98, v107
	s_wait_dscnt 0x0
	v_fma_f32 v36, -v108, v106, v36
	ds_bpermute_b32 v106, v99, v107
	s_wait_dscnt 0x0
	v_fma_f32 v37, -v108, v106, v37
	ds_bpermute_b32 v106, v100, v107
	s_wait_dscnt 0x0
	v_fma_f32 v38, -v108, v106, v38
	ds_bpermute_b32 v106, v101, v107
	s_wait_dscnt 0x0
	v_fma_f32 v39, -v108, v106, v39
	ds_bpermute_b32 v106, v102, v107
	s_wait_dscnt 0x0
	v_fma_f32 v40, -v108, v106, v40
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v106, 0, v16, s1
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v106, v106, v32, s2
	ds_bpermute_b32 v107, v87, v106
	s_wait_dscnt 0x0
	v_add_f32_e32 v108, v106, v107
	ds_load_2addr_b32 v[106:107], v109 offset0:192 offset1:224
	s_wait_dscnt 0x0
	ds_bpermute_b32 v109, v104, v106
	s_wait_dscnt 0x0
	v_fma_f32 v9, -v108, v109, v9
	ds_bpermute_b32 v109, v88, v106
	s_wait_dscnt 0x0
	v_fma_f32 v10, -v108, v109, v10
	ds_bpermute_b32 v109, v89, v106
	s_wait_dscnt 0x0
	v_fma_f32 v11, -v108, v109, v11
	ds_bpermute_b32 v109, v90, v106
	s_wait_dscnt 0x0
	v_fma_f32 v12, -v108, v109, v12
	ds_bpermute_b32 v109, v91, v106
	s_wait_dscnt 0x0
	v_fma_f32 v13, -v108, v109, v13
	ds_bpermute_b32 v109, v92, v106
	s_wait_dscnt 0x0
	v_fma_f32 v14, -v108, v109, v14
	ds_bpermute_b32 v109, v93, v106
	s_wait_dscnt 0x0
	v_fma_f32 v15, -v108, v109, v15
	ds_bpermute_b32 v109, v94, v106
	s_wait_dscnt 0x0
	v_fma_f32 v16, -v108, v109, v16
	ds_bpermute_b32 v109, v95, v106
	s_wait_dscnt 0x0
	v_fma_f32 v17, -v108, v109, v17
	ds_bpermute_b32 v109, v96, v106
	s_wait_dscnt 0x0
	v_fma_f32 v18, -v108, v109, v18
	ds_bpermute_b32 v109, v97, v106
	s_wait_dscnt 0x0
	v_fma_f32 v19, -v108, v109, v19
	ds_bpermute_b32 v109, v98, v106
	s_wait_dscnt 0x0
	v_fma_f32 v20, -v108, v109, v20
	ds_bpermute_b32 v109, v99, v106
	s_wait_dscnt 0x0
	v_fma_f32 v21, -v108, v109, v21
	ds_bpermute_b32 v109, v100, v106
	s_wait_dscnt 0x0
	v_fma_f32 v22, -v108, v109, v22
	ds_bpermute_b32 v109, v101, v106
	ds_bpermute_b32 v106, v102, v106
	s_wait_dscnt 0x1
	v_fma_f32 v23, -v108, v109, v23
	s_wait_dscnt 0x0
	v_fma_f32 v24, -v108, v106, v24
	ds_bpermute_b32 v106, v104, v107
	v_add_nc_u32_e32 v109, 0x800, v105
	s_wait_dscnt 0x0
	v_fma_f32 v25, -v108, v106, v25
	ds_bpermute_b32 v106, v88, v107
	s_wait_dscnt 0x0
	v_fma_f32 v26, -v108, v106, v26
	ds_bpermute_b32 v106, v89, v107
	s_wait_dscnt 0x0
	v_fma_f32 v27, -v108, v106, v27
	ds_bpermute_b32 v106, v90, v107
	s_wait_dscnt 0x0
	v_fma_f32 v28, -v108, v106, v28
	ds_bpermute_b32 v106, v91, v107
	s_wait_dscnt 0x0
	v_fma_f32 v29, -v108, v106, v29
	ds_bpermute_b32 v106, v92, v107
	s_wait_dscnt 0x0
	v_fma_f32 v30, -v108, v106, v30
	ds_bpermute_b32 v106, v93, v107
	s_wait_dscnt 0x0
	v_fma_f32 v31, -v108, v106, v31
	ds_bpermute_b32 v106, v94, v107
	s_wait_dscnt 0x0
	v_fma_f32 v32, -v108, v106, v32
	ds_bpermute_b32 v106, v95, v107
	s_wait_dscnt 0x0
	v_fma_f32 v33, -v108, v106, v33
	ds_bpermute_b32 v106, v96, v107
	s_wait_dscnt 0x0
	v_fma_f32 v34, -v108, v106, v34
	ds_bpermute_b32 v106, v97, v107
	s_wait_dscnt 0x0
	v_fma_f32 v35, -v108, v106, v35
	ds_bpermute_b32 v106, v98, v107
	s_wait_dscnt 0x0
	v_fma_f32 v36, -v108, v106, v36
	ds_bpermute_b32 v106, v99, v107
	s_wait_dscnt 0x0
	v_fma_f32 v37, -v108, v106, v37
	ds_bpermute_b32 v106, v100, v107
	s_wait_dscnt 0x0
	v_fma_f32 v38, -v108, v106, v38
	ds_bpermute_b32 v106, v101, v107
	s_wait_dscnt 0x0
	v_fma_f32 v39, -v108, v106, v39
	ds_bpermute_b32 v106, v102, v107
	s_wait_dscnt 0x0
	v_fma_f32 v40, -v108, v106, v40
	;;#ASMSTART
	;;#ASMEND
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v106, 0, v9, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v106, v106, v25, s0
	ds_bpermute_b32 v107, v87, v106
	s_wait_dscnt 0x0
	v_add_f32_e32 v108, v106, v107
	ds_load_2addr_b32 v[106:107], v109 offset1:32
	s_wait_dscnt 0x0
	ds_bpermute_b32 v110, v104, v106
	s_wait_dscnt 0x0
	v_fma_f32 v9, -v108, v110, v9
	ds_bpermute_b32 v110, v88, v106
	s_wait_dscnt 0x0
	v_fma_f32 v10, -v108, v110, v10
	ds_bpermute_b32 v110, v89, v106
	s_wait_dscnt 0x0
	v_fma_f32 v11, -v108, v110, v11
	ds_bpermute_b32 v110, v90, v106
	s_wait_dscnt 0x0
	v_fma_f32 v12, -v108, v110, v12
	ds_bpermute_b32 v110, v91, v106
	s_wait_dscnt 0x0
	v_fma_f32 v13, -v108, v110, v13
	ds_bpermute_b32 v110, v92, v106
	s_wait_dscnt 0x0
	v_fma_f32 v14, -v108, v110, v14
	ds_bpermute_b32 v110, v93, v106
	s_wait_dscnt 0x0
	v_fma_f32 v15, -v108, v110, v15
	ds_bpermute_b32 v110, v94, v106
	s_wait_dscnt 0x0
	v_fma_f32 v16, -v108, v110, v16
	ds_bpermute_b32 v110, v95, v106
	s_wait_dscnt 0x0
	v_fma_f32 v17, -v108, v110, v17
	ds_bpermute_b32 v110, v96, v106
	s_wait_dscnt 0x0
	v_fma_f32 v18, -v108, v110, v18
	ds_bpermute_b32 v110, v97, v106
	s_wait_dscnt 0x0
	v_fma_f32 v19, -v108, v110, v19
	ds_bpermute_b32 v110, v98, v106
	s_wait_dscnt 0x0
	v_fma_f32 v20, -v108, v110, v20
	ds_bpermute_b32 v110, v99, v106
	s_wait_dscnt 0x0
	v_fma_f32 v21, -v108, v110, v21
	ds_bpermute_b32 v110, v100, v106
	s_wait_dscnt 0x0
	v_fma_f32 v22, -v108, v110, v22
	ds_bpermute_b32 v110, v101, v106
	ds_bpermute_b32 v106, v102, v106
	s_wait_dscnt 0x1
	v_fma_f32 v23, -v108, v110, v23
	s_wait_dscnt 0x0
	v_fma_f32 v24, -v108, v106, v24
	ds_bpermute_b32 v106, v104, v107
	s_wait_dscnt 0x0
	v_fma_f32 v25, -v108, v106, v25
	ds_bpermute_b32 v106, v88, v107
	s_wait_dscnt 0x0
	v_fma_f32 v26, -v108, v106, v26
	ds_bpermute_b32 v106, v89, v107
	s_wait_dscnt 0x0
	v_fma_f32 v27, -v108, v106, v27
	ds_bpermute_b32 v106, v90, v107
	s_wait_dscnt 0x0
	v_fma_f32 v28, -v108, v106, v28
	ds_bpermute_b32 v106, v91, v107
	s_wait_dscnt 0x0
	v_fma_f32 v29, -v108, v106, v29
	ds_bpermute_b32 v106, v92, v107
	s_wait_dscnt 0x0
	v_fma_f32 v30, -v108, v106, v30
	ds_bpermute_b32 v106, v93, v107
	s_wait_dscnt 0x0
	v_fma_f32 v31, -v108, v106, v31
	ds_bpermute_b32 v106, v94, v107
	s_wait_dscnt 0x0
	v_fma_f32 v32, -v108, v106, v32
	ds_bpermute_b32 v106, v95, v107
	s_wait_dscnt 0x0
	v_fma_f32 v33, -v108, v106, v33
	ds_bpermute_b32 v106, v96, v107
	s_wait_dscnt 0x0
	v_fma_f32 v34, -v108, v106, v34
	ds_bpermute_b32 v106, v97, v107
	s_wait_dscnt 0x0
	v_fma_f32 v35, -v108, v106, v35
	ds_bpermute_b32 v106, v98, v107
	s_wait_dscnt 0x0
	v_fma_f32 v36, -v108, v106, v36
	ds_bpermute_b32 v106, v99, v107
	s_wait_dscnt 0x0
	v_fma_f32 v37, -v108, v106, v37
	ds_bpermute_b32 v106, v100, v107
	s_wait_dscnt 0x0
	v_fma_f32 v38, -v108, v106, v38
	ds_bpermute_b32 v106, v101, v107
	s_wait_dscnt 0x0
	v_fma_f32 v39, -v108, v106, v39
	ds_bpermute_b32 v106, v102, v107
	s_wait_dscnt 0x0
	v_fma_f32 v40, -v108, v106, v40
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e32 v106, 0, v10, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v106, v106, v26, s0
	ds_bpermute_b32 v107, v87, v106
	s_wait_dscnt 0x0
	v_add_f32_e32 v108, v106, v107
	ds_load_2addr_b32 v[106:107], v109 offset0:64 offset1:96
	s_wait_dscnt 0x0
	ds_bpermute_b32 v110, v104, v106
	s_wait_dscnt 0x0
	v_fma_f32 v9, -v108, v110, v9
	ds_bpermute_b32 v110, v88, v106
	s_wait_dscnt 0x0
	v_fma_f32 v10, -v108, v110, v10
	ds_bpermute_b32 v110, v89, v106
	s_wait_dscnt 0x0
	v_fma_f32 v11, -v108, v110, v11
	ds_bpermute_b32 v110, v90, v106
	s_wait_dscnt 0x0
	v_fma_f32 v12, -v108, v110, v12
	ds_bpermute_b32 v110, v91, v106
	s_wait_dscnt 0x0
	v_fma_f32 v13, -v108, v110, v13
	ds_bpermute_b32 v110, v92, v106
	s_wait_dscnt 0x0
	v_fma_f32 v14, -v108, v110, v14
	ds_bpermute_b32 v110, v93, v106
	s_wait_dscnt 0x0
	v_fma_f32 v15, -v108, v110, v15
	ds_bpermute_b32 v110, v94, v106
	s_wait_dscnt 0x0
	v_fma_f32 v16, -v108, v110, v16
	ds_bpermute_b32 v110, v95, v106
	s_wait_dscnt 0x0
	v_fma_f32 v17, -v108, v110, v17
	ds_bpermute_b32 v110, v96, v106
	s_wait_dscnt 0x0
	v_fma_f32 v18, -v108, v110, v18
	ds_bpermute_b32 v110, v97, v106
	s_wait_dscnt 0x0
	v_fma_f32 v19, -v108, v110, v19
	ds_bpermute_b32 v110, v98, v106
	s_wait_dscnt 0x0
	v_fma_f32 v20, -v108, v110, v20
	ds_bpermute_b32 v110, v99, v106
	s_wait_dscnt 0x0
	v_fma_f32 v21, -v108, v110, v21
	ds_bpermute_b32 v110, v100, v106
	s_wait_dscnt 0x0
	v_fma_f32 v22, -v108, v110, v22
	ds_bpermute_b32 v110, v101, v106
	ds_bpermute_b32 v106, v102, v106
	s_wait_dscnt 0x1
	v_fma_f32 v23, -v108, v110, v23
	s_wait_dscnt 0x0
	v_fma_f32 v24, -v108, v106, v24
	ds_bpermute_b32 v106, v104, v107
	s_wait_dscnt 0x0
	v_fma_f32 v25, -v108, v106, v25
	ds_bpermute_b32 v106, v88, v107
	s_wait_dscnt 0x0
	v_fma_f32 v26, -v108, v106, v26
	ds_bpermute_b32 v106, v89, v107
	s_wait_dscnt 0x0
	v_fma_f32 v27, -v108, v106, v27
	ds_bpermute_b32 v106, v90, v107
	s_wait_dscnt 0x0
	v_fma_f32 v28, -v108, v106, v28
	ds_bpermute_b32 v106, v91, v107
	s_wait_dscnt 0x0
	v_fma_f32 v29, -v108, v106, v29
	ds_bpermute_b32 v106, v92, v107
	s_wait_dscnt 0x0
	v_fma_f32 v30, -v108, v106, v30
	ds_bpermute_b32 v106, v93, v107
	s_wait_dscnt 0x0
	v_fma_f32 v31, -v108, v106, v31
	ds_bpermute_b32 v106, v94, v107
	s_wait_dscnt 0x0
	v_fma_f32 v32, -v108, v106, v32
	ds_bpermute_b32 v106, v95, v107
	s_wait_dscnt 0x0
	v_fma_f32 v33, -v108, v106, v33
	ds_bpermute_b32 v106, v96, v107
	s_wait_dscnt 0x0
	v_fma_f32 v34, -v108, v106, v34
	ds_bpermute_b32 v106, v97, v107
	s_wait_dscnt 0x0
	v_fma_f32 v35, -v108, v106, v35
	ds_bpermute_b32 v106, v98, v107
	s_wait_dscnt 0x0
	v_fma_f32 v36, -v108, v106, v36
	ds_bpermute_b32 v106, v99, v107
	s_wait_dscnt 0x0
	v_fma_f32 v37, -v108, v106, v37
	ds_bpermute_b32 v106, v100, v107
	s_wait_dscnt 0x0
	v_fma_f32 v38, -v108, v106, v38
	ds_bpermute_b32 v106, v101, v107
	s_wait_dscnt 0x0
	v_fma_f32 v39, -v108, v106, v39
	ds_bpermute_b32 v106, v102, v107
	s_wait_dscnt 0x0
	v_fma_f32 v40, -v108, v106, v40
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e32 v106, 0, v11, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v106, v106, v27, s0
	ds_bpermute_b32 v107, v87, v106
	s_wait_dscnt 0x0
	v_add_f32_e32 v108, v106, v107
	ds_load_2addr_b32 v[106:107], v109 offset0:128 offset1:160
	s_wait_dscnt 0x0
	ds_bpermute_b32 v110, v104, v106
	s_wait_dscnt 0x0
	v_fma_f32 v9, -v108, v110, v9
	ds_bpermute_b32 v110, v88, v106
	s_wait_dscnt 0x0
	v_fma_f32 v10, -v108, v110, v10
	ds_bpermute_b32 v110, v89, v106
	s_wait_dscnt 0x0
	v_fma_f32 v11, -v108, v110, v11
	ds_bpermute_b32 v110, v90, v106
	s_wait_dscnt 0x0
	v_fma_f32 v12, -v108, v110, v12
	ds_bpermute_b32 v110, v91, v106
	s_wait_dscnt 0x0
	v_fma_f32 v13, -v108, v110, v13
	ds_bpermute_b32 v110, v92, v106
	s_wait_dscnt 0x0
	v_fma_f32 v14, -v108, v110, v14
	ds_bpermute_b32 v110, v93, v106
	s_wait_dscnt 0x0
	v_fma_f32 v15, -v108, v110, v15
	ds_bpermute_b32 v110, v94, v106
	s_wait_dscnt 0x0
	v_fma_f32 v16, -v108, v110, v16
	ds_bpermute_b32 v110, v95, v106
	s_wait_dscnt 0x0
	v_fma_f32 v17, -v108, v110, v17
	ds_bpermute_b32 v110, v96, v106
	s_wait_dscnt 0x0
	v_fma_f32 v18, -v108, v110, v18
	ds_bpermute_b32 v110, v97, v106
	s_wait_dscnt 0x0
	v_fma_f32 v19, -v108, v110, v19
	ds_bpermute_b32 v110, v98, v106
	s_wait_dscnt 0x0
	v_fma_f32 v20, -v108, v110, v20
	ds_bpermute_b32 v110, v99, v106
	s_wait_dscnt 0x0
	v_fma_f32 v21, -v108, v110, v21
	ds_bpermute_b32 v110, v100, v106
	s_wait_dscnt 0x0
	v_fma_f32 v22, -v108, v110, v22
	ds_bpermute_b32 v110, v101, v106
	ds_bpermute_b32 v106, v102, v106
	s_wait_dscnt 0x1
	v_fma_f32 v23, -v108, v110, v23
	s_wait_dscnt 0x0
	v_fma_f32 v24, -v108, v106, v24
	ds_bpermute_b32 v106, v104, v107
	s_wait_dscnt 0x0
	v_fma_f32 v25, -v108, v106, v25
	ds_bpermute_b32 v106, v88, v107
	s_wait_dscnt 0x0
	v_fma_f32 v26, -v108, v106, v26
	ds_bpermute_b32 v106, v89, v107
	s_wait_dscnt 0x0
	v_fma_f32 v27, -v108, v106, v27
	ds_bpermute_b32 v106, v90, v107
	s_wait_dscnt 0x0
	v_fma_f32 v28, -v108, v106, v28
	ds_bpermute_b32 v106, v91, v107
	s_wait_dscnt 0x0
	v_fma_f32 v29, -v108, v106, v29
	ds_bpermute_b32 v106, v92, v107
	s_wait_dscnt 0x0
	v_fma_f32 v30, -v108, v106, v30
	ds_bpermute_b32 v106, v93, v107
	s_wait_dscnt 0x0
	v_fma_f32 v31, -v108, v106, v31
	ds_bpermute_b32 v106, v94, v107
	s_wait_dscnt 0x0
	v_fma_f32 v32, -v108, v106, v32
	ds_bpermute_b32 v106, v95, v107
	s_wait_dscnt 0x0
	v_fma_f32 v33, -v108, v106, v33
	ds_bpermute_b32 v106, v96, v107
	s_wait_dscnt 0x0
	v_fma_f32 v34, -v108, v106, v34
	ds_bpermute_b32 v106, v97, v107
	s_wait_dscnt 0x0
	v_fma_f32 v35, -v108, v106, v35
	ds_bpermute_b32 v106, v98, v107
	s_wait_dscnt 0x0
	v_fma_f32 v36, -v108, v106, v36
	ds_bpermute_b32 v106, v99, v107
	s_wait_dscnt 0x0
	v_fma_f32 v37, -v108, v106, v37
	ds_bpermute_b32 v106, v100, v107
	s_wait_dscnt 0x0
	v_fma_f32 v38, -v108, v106, v38
	ds_bpermute_b32 v106, v101, v107
	s_wait_dscnt 0x0
	v_fma_f32 v39, -v108, v106, v39
	ds_bpermute_b32 v106, v102, v107
	s_wait_dscnt 0x0
	v_fma_f32 v40, -v108, v106, v40
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e32 v106, 0, v12, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v106, v106, v28, s0
	ds_bpermute_b32 v107, v87, v106
	s_wait_dscnt 0x0
	v_add_f32_e32 v108, v106, v107
	ds_load_2addr_b32 v[106:107], v109 offset0:192 offset1:224
	s_wait_dscnt 0x0
	ds_bpermute_b32 v109, v104, v106
	s_wait_dscnt 0x0
	v_fma_f32 v9, -v108, v109, v9
	ds_bpermute_b32 v109, v88, v106
	s_wait_dscnt 0x0
	v_fma_f32 v10, -v108, v109, v10
	ds_bpermute_b32 v109, v89, v106
	s_wait_dscnt 0x0
	v_fma_f32 v11, -v108, v109, v11
	ds_bpermute_b32 v109, v90, v106
	s_wait_dscnt 0x0
	v_fma_f32 v12, -v108, v109, v12
	ds_bpermute_b32 v109, v91, v106
	s_wait_dscnt 0x0
	v_fma_f32 v13, -v108, v109, v13
	ds_bpermute_b32 v109, v92, v106
	s_wait_dscnt 0x0
	v_fma_f32 v14, -v108, v109, v14
	ds_bpermute_b32 v109, v93, v106
	s_wait_dscnt 0x0
	v_fma_f32 v15, -v108, v109, v15
	ds_bpermute_b32 v109, v94, v106
	s_wait_dscnt 0x0
	v_fma_f32 v16, -v108, v109, v16
	ds_bpermute_b32 v109, v95, v106
	s_wait_dscnt 0x0
	v_fma_f32 v17, -v108, v109, v17
	ds_bpermute_b32 v109, v96, v106
	s_wait_dscnt 0x0
	v_fma_f32 v18, -v108, v109, v18
	ds_bpermute_b32 v109, v97, v106
	s_wait_dscnt 0x0
	v_fma_f32 v19, -v108, v109, v19
	ds_bpermute_b32 v109, v98, v106
	s_wait_dscnt 0x0
	v_fma_f32 v20, -v108, v109, v20
	ds_bpermute_b32 v109, v99, v106
	s_wait_dscnt 0x0
	v_fma_f32 v21, -v108, v109, v21
	ds_bpermute_b32 v109, v100, v106
	s_wait_dscnt 0x0
	v_fma_f32 v22, -v108, v109, v22
	ds_bpermute_b32 v109, v101, v106
	ds_bpermute_b32 v106, v102, v106
	s_wait_dscnt 0x1
	v_fma_f32 v23, -v108, v109, v23
	s_wait_dscnt 0x0
	v_fma_f32 v24, -v108, v106, v24
	ds_bpermute_b32 v106, v104, v107
	v_add_nc_u32_e32 v109, 0xc00, v105
	s_wait_dscnt 0x0
	v_fma_f32 v25, -v108, v106, v25
	ds_bpermute_b32 v106, v88, v107
	s_wait_dscnt 0x0
	v_fma_f32 v26, -v108, v106, v26
	ds_bpermute_b32 v106, v89, v107
	s_wait_dscnt 0x0
	v_fma_f32 v27, -v108, v106, v27
	ds_bpermute_b32 v106, v90, v107
	s_wait_dscnt 0x0
	v_fma_f32 v28, -v108, v106, v28
	ds_bpermute_b32 v106, v91, v107
	s_wait_dscnt 0x0
	v_fma_f32 v29, -v108, v106, v29
	ds_bpermute_b32 v106, v92, v107
	s_wait_dscnt 0x0
	v_fma_f32 v30, -v108, v106, v30
	ds_bpermute_b32 v106, v93, v107
	s_wait_dscnt 0x0
	v_fma_f32 v31, -v108, v106, v31
	ds_bpermute_b32 v106, v94, v107
	s_wait_dscnt 0x0
	v_fma_f32 v32, -v108, v106, v32
	ds_bpermute_b32 v106, v95, v107
	s_wait_dscnt 0x0
	v_fma_f32 v33, -v108, v106, v33
	ds_bpermute_b32 v106, v96, v107
	s_wait_dscnt 0x0
	v_fma_f32 v34, -v108, v106, v34
	ds_bpermute_b32 v106, v97, v107
	s_wait_dscnt 0x0
	v_fma_f32 v35, -v108, v106, v35
	ds_bpermute_b32 v106, v98, v107
	s_wait_dscnt 0x0
	v_fma_f32 v36, -v108, v106, v36
	ds_bpermute_b32 v106, v99, v107
	s_wait_dscnt 0x0
	v_fma_f32 v37, -v108, v106, v37
	ds_bpermute_b32 v106, v100, v107
	s_wait_dscnt 0x0
	v_fma_f32 v38, -v108, v106, v38
	ds_bpermute_b32 v106, v101, v107
	s_wait_dscnt 0x0
	v_fma_f32 v39, -v108, v106, v39
	ds_bpermute_b32 v106, v102, v107
	s_wait_dscnt 0x0
	v_fma_f32 v40, -v108, v106, v40
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e32 v106, 0, v13, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v106, v106, v29, s0
	ds_bpermute_b32 v107, v87, v106
	s_wait_dscnt 0x0
	v_add_f32_e32 v108, v106, v107
	ds_load_2addr_b32 v[106:107], v109 offset1:32
	s_wait_dscnt 0x0
	ds_bpermute_b32 v110, v104, v106
	s_wait_dscnt 0x0
	v_fma_f32 v9, -v108, v110, v9
	ds_bpermute_b32 v110, v88, v106
	s_wait_dscnt 0x0
	v_fma_f32 v10, -v108, v110, v10
	ds_bpermute_b32 v110, v89, v106
	s_wait_dscnt 0x0
	v_fma_f32 v11, -v108, v110, v11
	ds_bpermute_b32 v110, v90, v106
	s_wait_dscnt 0x0
	v_fma_f32 v12, -v108, v110, v12
	ds_bpermute_b32 v110, v91, v106
	s_wait_dscnt 0x0
	v_fma_f32 v13, -v108, v110, v13
	ds_bpermute_b32 v110, v92, v106
	s_wait_dscnt 0x0
	v_fma_f32 v14, -v108, v110, v14
	ds_bpermute_b32 v110, v93, v106
	s_wait_dscnt 0x0
	v_fma_f32 v15, -v108, v110, v15
	ds_bpermute_b32 v110, v94, v106
	s_wait_dscnt 0x0
	v_fma_f32 v16, -v108, v110, v16
	ds_bpermute_b32 v110, v95, v106
	s_wait_dscnt 0x0
	v_fma_f32 v17, -v108, v110, v17
	ds_bpermute_b32 v110, v96, v106
	s_wait_dscnt 0x0
	v_fma_f32 v18, -v108, v110, v18
	ds_bpermute_b32 v110, v97, v106
	s_wait_dscnt 0x0
	v_fma_f32 v19, -v108, v110, v19
	ds_bpermute_b32 v110, v98, v106
	s_wait_dscnt 0x0
	v_fma_f32 v20, -v108, v110, v20
	ds_bpermute_b32 v110, v99, v106
	s_wait_dscnt 0x0
	v_fma_f32 v21, -v108, v110, v21
	ds_bpermute_b32 v110, v100, v106
	s_wait_dscnt 0x0
	v_fma_f32 v22, -v108, v110, v22
	ds_bpermute_b32 v110, v101, v106
	ds_bpermute_b32 v106, v102, v106
	s_wait_dscnt 0x1
	v_fma_f32 v23, -v108, v110, v23
	s_wait_dscnt 0x0
	v_fma_f32 v24, -v108, v106, v24
	ds_bpermute_b32 v106, v104, v107
	s_wait_dscnt 0x0
	v_fma_f32 v25, -v108, v106, v25
	ds_bpermute_b32 v106, v88, v107
	s_wait_dscnt 0x0
	v_fma_f32 v26, -v108, v106, v26
	ds_bpermute_b32 v106, v89, v107
	s_wait_dscnt 0x0
	v_fma_f32 v27, -v108, v106, v27
	ds_bpermute_b32 v106, v90, v107
	s_wait_dscnt 0x0
	v_fma_f32 v28, -v108, v106, v28
	ds_bpermute_b32 v106, v91, v107
	s_wait_dscnt 0x0
	v_fma_f32 v29, -v108, v106, v29
	ds_bpermute_b32 v106, v92, v107
	s_wait_dscnt 0x0
	v_fma_f32 v30, -v108, v106, v30
	ds_bpermute_b32 v106, v93, v107
	s_wait_dscnt 0x0
	v_fma_f32 v31, -v108, v106, v31
	ds_bpermute_b32 v106, v94, v107
	s_wait_dscnt 0x0
	v_fma_f32 v32, -v108, v106, v32
	ds_bpermute_b32 v106, v95, v107
	s_wait_dscnt 0x0
	v_fma_f32 v33, -v108, v106, v33
	ds_bpermute_b32 v106, v96, v107
	s_wait_dscnt 0x0
	v_fma_f32 v34, -v108, v106, v34
	ds_bpermute_b32 v106, v97, v107
	s_wait_dscnt 0x0
	v_fma_f32 v35, -v108, v106, v35
	ds_bpermute_b32 v106, v98, v107
	s_wait_dscnt 0x0
	v_fma_f32 v36, -v108, v106, v36
	ds_bpermute_b32 v106, v99, v107
	s_wait_dscnt 0x0
	v_fma_f32 v37, -v108, v106, v37
	ds_bpermute_b32 v106, v100, v107
	s_wait_dscnt 0x0
	v_fma_f32 v38, -v108, v106, v38
	ds_bpermute_b32 v106, v101, v107
	s_wait_dscnt 0x0
	v_fma_f32 v39, -v108, v106, v39
	ds_bpermute_b32 v106, v102, v107
	s_wait_dscnt 0x0
	v_fma_f32 v40, -v108, v106, v40
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e32 v106, 0, v14, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v106, v106, v30, s0
	ds_bpermute_b32 v107, v87, v106
	s_wait_dscnt 0x0
	v_add_f32_e32 v108, v106, v107
	ds_load_2addr_b32 v[106:107], v109 offset0:64 offset1:96
	s_wait_dscnt 0x0
	ds_bpermute_b32 v110, v104, v106
	s_wait_dscnt 0x0
	v_fma_f32 v9, -v108, v110, v9
	ds_bpermute_b32 v110, v88, v106
	s_wait_dscnt 0x0
	v_fma_f32 v10, -v108, v110, v10
	ds_bpermute_b32 v110, v89, v106
	s_wait_dscnt 0x0
	v_fma_f32 v11, -v108, v110, v11
	ds_bpermute_b32 v110, v90, v106
	s_wait_dscnt 0x0
	v_fma_f32 v12, -v108, v110, v12
	ds_bpermute_b32 v110, v91, v106
	s_wait_dscnt 0x0
	v_fma_f32 v13, -v108, v110, v13
	ds_bpermute_b32 v110, v92, v106
	s_wait_dscnt 0x0
	v_fma_f32 v14, -v108, v110, v14
	ds_bpermute_b32 v110, v93, v106
	s_wait_dscnt 0x0
	v_fma_f32 v15, -v108, v110, v15
	ds_bpermute_b32 v110, v94, v106
	s_wait_dscnt 0x0
	v_fma_f32 v16, -v108, v110, v16
	ds_bpermute_b32 v110, v95, v106
	s_wait_dscnt 0x0
	v_fma_f32 v17, -v108, v110, v17
	ds_bpermute_b32 v110, v96, v106
	s_wait_dscnt 0x0
	v_fma_f32 v18, -v108, v110, v18
	ds_bpermute_b32 v110, v97, v106
	s_wait_dscnt 0x0
	v_fma_f32 v19, -v108, v110, v19
	ds_bpermute_b32 v110, v98, v106
	s_wait_dscnt 0x0
	v_fma_f32 v20, -v108, v110, v20
	ds_bpermute_b32 v110, v99, v106
	s_wait_dscnt 0x0
	v_fma_f32 v21, -v108, v110, v21
	ds_bpermute_b32 v110, v100, v106
	s_wait_dscnt 0x0
	v_fma_f32 v22, -v108, v110, v22
	ds_bpermute_b32 v110, v101, v106
	ds_bpermute_b32 v106, v102, v106
	s_wait_dscnt 0x1
	v_fma_f32 v23, -v108, v110, v23
	s_wait_dscnt 0x0
	v_fma_f32 v24, -v108, v106, v24
	ds_bpermute_b32 v106, v104, v107
	s_wait_dscnt 0x0
	v_fma_f32 v25, -v108, v106, v25
	ds_bpermute_b32 v106, v88, v107
	s_wait_dscnt 0x0
	v_fma_f32 v26, -v108, v106, v26
	ds_bpermute_b32 v106, v89, v107
	s_wait_dscnt 0x0
	v_fma_f32 v27, -v108, v106, v27
	ds_bpermute_b32 v106, v90, v107
	s_wait_dscnt 0x0
	v_fma_f32 v28, -v108, v106, v28
	ds_bpermute_b32 v106, v91, v107
	s_wait_dscnt 0x0
	v_fma_f32 v29, -v108, v106, v29
	ds_bpermute_b32 v106, v92, v107
	s_wait_dscnt 0x0
	v_fma_f32 v30, -v108, v106, v30
	ds_bpermute_b32 v106, v93, v107
	s_wait_dscnt 0x0
	v_fma_f32 v31, -v108, v106, v31
	ds_bpermute_b32 v106, v94, v107
	s_wait_dscnt 0x0
	v_fma_f32 v32, -v108, v106, v32
	ds_bpermute_b32 v106, v95, v107
	s_wait_dscnt 0x0
	v_fma_f32 v33, -v108, v106, v33
	ds_bpermute_b32 v106, v96, v107
	s_wait_dscnt 0x0
	v_fma_f32 v34, -v108, v106, v34
	ds_bpermute_b32 v106, v97, v107
	s_wait_dscnt 0x0
	v_fma_f32 v35, -v108, v106, v35
	ds_bpermute_b32 v106, v98, v107
	s_wait_dscnt 0x0
	v_fma_f32 v36, -v108, v106, v36
	ds_bpermute_b32 v106, v99, v107
	s_wait_dscnt 0x0
	v_fma_f32 v37, -v108, v106, v37
	ds_bpermute_b32 v106, v100, v107
	s_wait_dscnt 0x0
	v_fma_f32 v38, -v108, v106, v38
	ds_bpermute_b32 v106, v101, v107
	s_wait_dscnt 0x0
	v_fma_f32 v39, -v108, v106, v39
	ds_bpermute_b32 v106, v102, v107
	s_wait_dscnt 0x0
	v_fma_f32 v40, -v108, v106, v40
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e32 v106, 0, v15, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v106, v106, v31, s0
	ds_bpermute_b32 v107, v87, v106
	s_wait_dscnt 0x0
	v_add_f32_e32 v108, v106, v107
	ds_load_2addr_b32 v[106:107], v109 offset0:128 offset1:160
	s_wait_dscnt 0x0
	ds_bpermute_b32 v110, v104, v106
	s_wait_dscnt 0x0
	v_fma_f32 v9, -v108, v110, v9
	ds_bpermute_b32 v110, v88, v106
	s_wait_dscnt 0x0
	v_fma_f32 v10, -v108, v110, v10
	ds_bpermute_b32 v110, v89, v106
	s_wait_dscnt 0x0
	v_fma_f32 v11, -v108, v110, v11
	ds_bpermute_b32 v110, v90, v106
	s_wait_dscnt 0x0
	v_fma_f32 v12, -v108, v110, v12
	ds_bpermute_b32 v110, v91, v106
	s_wait_dscnt 0x0
	v_fma_f32 v13, -v108, v110, v13
	ds_bpermute_b32 v110, v92, v106
	s_wait_dscnt 0x0
	v_fma_f32 v14, -v108, v110, v14
	ds_bpermute_b32 v110, v93, v106
	s_wait_dscnt 0x0
	v_fma_f32 v15, -v108, v110, v15
	ds_bpermute_b32 v110, v94, v106
	s_wait_dscnt 0x0
	v_fma_f32 v16, -v108, v110, v16
	ds_bpermute_b32 v110, v95, v106
	s_wait_dscnt 0x0
	v_fma_f32 v17, -v108, v110, v17
	ds_bpermute_b32 v110, v96, v106
	s_wait_dscnt 0x0
	v_fma_f32 v18, -v108, v110, v18
	ds_bpermute_b32 v110, v97, v106
	s_wait_dscnt 0x0
	v_fma_f32 v19, -v108, v110, v19
	ds_bpermute_b32 v110, v98, v106
	s_wait_dscnt 0x0
	v_fma_f32 v20, -v108, v110, v20
	ds_bpermute_b32 v110, v99, v106
	s_wait_dscnt 0x0
	v_fma_f32 v21, -v108, v110, v21
	ds_bpermute_b32 v110, v100, v106
	s_wait_dscnt 0x0
	v_fma_f32 v22, -v108, v110, v22
	ds_bpermute_b32 v110, v101, v106
	ds_bpermute_b32 v106, v102, v106
	s_wait_dscnt 0x1
	v_fma_f32 v23, -v108, v110, v23
	s_wait_dscnt 0x0
	v_fma_f32 v24, -v108, v106, v24
	ds_bpermute_b32 v106, v104, v107
	s_wait_dscnt 0x0
	v_fma_f32 v25, -v108, v106, v25
	ds_bpermute_b32 v106, v88, v107
	s_wait_dscnt 0x0
	v_fma_f32 v26, -v108, v106, v26
	ds_bpermute_b32 v106, v89, v107
	s_wait_dscnt 0x0
	v_fma_f32 v27, -v108, v106, v27
	ds_bpermute_b32 v106, v90, v107
	s_wait_dscnt 0x0
	v_fma_f32 v28, -v108, v106, v28
	ds_bpermute_b32 v106, v91, v107
	s_wait_dscnt 0x0
	v_fma_f32 v29, -v108, v106, v29
	ds_bpermute_b32 v106, v92, v107
	s_wait_dscnt 0x0
	v_fma_f32 v30, -v108, v106, v30
	ds_bpermute_b32 v106, v93, v107
	s_wait_dscnt 0x0
	v_fma_f32 v31, -v108, v106, v31
	ds_bpermute_b32 v106, v94, v107
	s_wait_dscnt 0x0
	v_fma_f32 v32, -v108, v106, v32
	ds_bpermute_b32 v106, v95, v107
	s_wait_dscnt 0x0
	v_fma_f32 v33, -v108, v106, v33
	ds_bpermute_b32 v106, v96, v107
	s_wait_dscnt 0x0
	v_fma_f32 v34, -v108, v106, v34
	ds_bpermute_b32 v106, v97, v107
	s_wait_dscnt 0x0
	v_fma_f32 v35, -v108, v106, v35
	ds_bpermute_b32 v106, v98, v107
	s_wait_dscnt 0x0
	v_fma_f32 v36, -v108, v106, v36
	ds_bpermute_b32 v106, v99, v107
	s_wait_dscnt 0x0
	v_fma_f32 v37, -v108, v106, v37
	ds_bpermute_b32 v106, v100, v107
	s_wait_dscnt 0x0
	v_fma_f32 v38, -v108, v106, v38
	ds_bpermute_b32 v106, v101, v107
	s_wait_dscnt 0x0
	v_fma_f32 v39, -v108, v106, v39
	ds_bpermute_b32 v106, v102, v107
	s_wait_dscnt 0x0
	v_fma_f32 v40, -v108, v106, v40
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e32 v106, 0, v16, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v106, v106, v32, s0
	ds_bpermute_b32 v107, v87, v106
	s_wait_dscnt 0x0
	v_add_f32_e32 v108, v106, v107
	ds_load_2addr_b32 v[106:107], v109 offset0:192 offset1:224
	s_wait_dscnt 0x0
	ds_bpermute_b32 v109, v104, v106
	s_wait_dscnt 0x0
	v_fma_f32 v9, -v108, v109, v9
	ds_bpermute_b32 v109, v88, v106
	s_wait_dscnt 0x0
	v_fma_f32 v10, -v108, v109, v10
	ds_bpermute_b32 v109, v89, v106
	s_wait_dscnt 0x0
	v_fma_f32 v11, -v108, v109, v11
	ds_bpermute_b32 v109, v90, v106
	s_wait_dscnt 0x0
	v_fma_f32 v12, -v108, v109, v12
	ds_bpermute_b32 v109, v91, v106
	s_wait_dscnt 0x0
	v_fma_f32 v13, -v108, v109, v13
	ds_bpermute_b32 v109, v92, v106
	s_wait_dscnt 0x0
	v_fma_f32 v14, -v108, v109, v14
	ds_bpermute_b32 v109, v93, v106
	s_wait_dscnt 0x0
	v_fma_f32 v15, -v108, v109, v15
	ds_bpermute_b32 v109, v94, v106
	s_wait_dscnt 0x0
	v_fma_f32 v16, -v108, v109, v16
	ds_bpermute_b32 v109, v95, v106
	s_wait_dscnt 0x0
	v_fma_f32 v17, -v108, v109, v17
	ds_bpermute_b32 v109, v96, v106
	s_wait_dscnt 0x0
	v_fma_f32 v18, -v108, v109, v18
	ds_bpermute_b32 v109, v97, v106
	s_wait_dscnt 0x0
	v_fma_f32 v19, -v108, v109, v19
	ds_bpermute_b32 v109, v98, v106
	s_wait_dscnt 0x0
	v_fma_f32 v20, -v108, v109, v20
	ds_bpermute_b32 v109, v99, v106
	s_wait_dscnt 0x0
	v_fma_f32 v21, -v108, v109, v21
	ds_bpermute_b32 v109, v100, v106
	s_wait_dscnt 0x0
	v_fma_f32 v22, -v108, v109, v22
	ds_bpermute_b32 v109, v101, v106
	ds_bpermute_b32 v106, v102, v106
	s_wait_dscnt 0x1
	v_fma_f32 v23, -v108, v109, v23
	s_wait_dscnt 0x0
	v_fma_f32 v24, -v108, v106, v24
	ds_bpermute_b32 v106, v104, v107
	v_add_nc_u32_e32 v109, 0x1000, v105
	s_wait_dscnt 0x0
	v_fma_f32 v25, -v108, v106, v25
	ds_bpermute_b32 v106, v88, v107
	s_wait_dscnt 0x0
	v_fma_f32 v26, -v108, v106, v26
	ds_bpermute_b32 v106, v89, v107
	s_wait_dscnt 0x0
	v_fma_f32 v27, -v108, v106, v27
	ds_bpermute_b32 v106, v90, v107
	s_wait_dscnt 0x0
	v_fma_f32 v28, -v108, v106, v28
	ds_bpermute_b32 v106, v91, v107
	s_wait_dscnt 0x0
	v_fma_f32 v29, -v108, v106, v29
	ds_bpermute_b32 v106, v92, v107
	s_wait_dscnt 0x0
	v_fma_f32 v30, -v108, v106, v30
	ds_bpermute_b32 v106, v93, v107
	s_wait_dscnt 0x0
	v_fma_f32 v31, -v108, v106, v31
	ds_bpermute_b32 v106, v94, v107
	s_wait_dscnt 0x0
	v_fma_f32 v32, -v108, v106, v32
	ds_bpermute_b32 v106, v95, v107
	s_wait_dscnt 0x0
	v_fma_f32 v33, -v108, v106, v33
	ds_bpermute_b32 v106, v96, v107
	s_wait_dscnt 0x0
	v_fma_f32 v34, -v108, v106, v34
	ds_bpermute_b32 v106, v97, v107
	s_wait_dscnt 0x0
	v_fma_f32 v35, -v108, v106, v35
	ds_bpermute_b32 v106, v98, v107
	s_wait_dscnt 0x0
	v_fma_f32 v36, -v108, v106, v36
	ds_bpermute_b32 v106, v99, v107
	s_wait_dscnt 0x0
	v_fma_f32 v37, -v108, v106, v37
	ds_bpermute_b32 v106, v100, v107
	s_wait_dscnt 0x0
	v_fma_f32 v38, -v108, v106, v38
	ds_bpermute_b32 v106, v101, v107
	s_wait_dscnt 0x0
	v_fma_f32 v39, -v108, v106, v39
	ds_bpermute_b32 v106, v102, v107
	s_wait_dscnt 0x0
	v_fma_f32 v40, -v108, v106, v40
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v106, 0, v17, s1
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v106, v106, v33, s2
	ds_bpermute_b32 v107, v87, v106
	s_wait_dscnt 0x0
	v_add_f32_e32 v108, v106, v107
	ds_load_2addr_b32 v[106:107], v109 offset1:32
	s_wait_dscnt 0x0
	ds_bpermute_b32 v110, v104, v106
	s_wait_dscnt 0x0
	v_fma_f32 v9, -v108, v110, v9
	ds_bpermute_b32 v110, v88, v106
	s_wait_dscnt 0x0
	v_fma_f32 v10, -v108, v110, v10
	ds_bpermute_b32 v110, v89, v106
	s_wait_dscnt 0x0
	v_fma_f32 v11, -v108, v110, v11
	ds_bpermute_b32 v110, v90, v106
	s_wait_dscnt 0x0
	v_fma_f32 v12, -v108, v110, v12
	ds_bpermute_b32 v110, v91, v106
	s_wait_dscnt 0x0
	v_fma_f32 v13, -v108, v110, v13
	ds_bpermute_b32 v110, v92, v106
	s_wait_dscnt 0x0
	v_fma_f32 v14, -v108, v110, v14
	ds_bpermute_b32 v110, v93, v106
	s_wait_dscnt 0x0
	v_fma_f32 v15, -v108, v110, v15
	ds_bpermute_b32 v110, v94, v106
	s_wait_dscnt 0x0
	v_fma_f32 v16, -v108, v110, v16
	ds_bpermute_b32 v110, v95, v106
	s_wait_dscnt 0x0
	v_fma_f32 v17, -v108, v110, v17
	ds_bpermute_b32 v110, v96, v106
	s_wait_dscnt 0x0
	v_fma_f32 v18, -v108, v110, v18
	ds_bpermute_b32 v110, v97, v106
	s_wait_dscnt 0x0
	v_fma_f32 v19, -v108, v110, v19
	ds_bpermute_b32 v110, v98, v106
	s_wait_dscnt 0x0
	v_fma_f32 v20, -v108, v110, v20
	ds_bpermute_b32 v110, v99, v106
	s_wait_dscnt 0x0
	v_fma_f32 v21, -v108, v110, v21
	ds_bpermute_b32 v110, v100, v106
	s_wait_dscnt 0x0
	v_fma_f32 v22, -v108, v110, v22
	ds_bpermute_b32 v110, v101, v106
	ds_bpermute_b32 v106, v102, v106
	s_wait_dscnt 0x1
	v_fma_f32 v23, -v108, v110, v23
	s_wait_dscnt 0x0
	v_fma_f32 v24, -v108, v106, v24
	ds_bpermute_b32 v106, v104, v107
	s_wait_dscnt 0x0
	v_fma_f32 v25, -v108, v106, v25
	ds_bpermute_b32 v106, v88, v107
	s_wait_dscnt 0x0
	v_fma_f32 v26, -v108, v106, v26
	ds_bpermute_b32 v106, v89, v107
	s_wait_dscnt 0x0
	v_fma_f32 v27, -v108, v106, v27
	ds_bpermute_b32 v106, v90, v107
	s_wait_dscnt 0x0
	v_fma_f32 v28, -v108, v106, v28
	ds_bpermute_b32 v106, v91, v107
	s_wait_dscnt 0x0
	v_fma_f32 v29, -v108, v106, v29
	ds_bpermute_b32 v106, v92, v107
	s_wait_dscnt 0x0
	v_fma_f32 v30, -v108, v106, v30
	ds_bpermute_b32 v106, v93, v107
	s_wait_dscnt 0x0
	v_fma_f32 v31, -v108, v106, v31
	ds_bpermute_b32 v106, v94, v107
	s_wait_dscnt 0x0
	v_fma_f32 v32, -v108, v106, v32
	ds_bpermute_b32 v106, v95, v107
	s_wait_dscnt 0x0
	v_fma_f32 v33, -v108, v106, v33
	ds_bpermute_b32 v106, v96, v107
	s_wait_dscnt 0x0
	v_fma_f32 v34, -v108, v106, v34
	ds_bpermute_b32 v106, v97, v107
	s_wait_dscnt 0x0
	v_fma_f32 v35, -v108, v106, v35
	ds_bpermute_b32 v106, v98, v107
	s_wait_dscnt 0x0
	v_fma_f32 v36, -v108, v106, v36
	ds_bpermute_b32 v106, v99, v107
	s_wait_dscnt 0x0
	v_fma_f32 v37, -v108, v106, v37
	ds_bpermute_b32 v106, v100, v107
	s_wait_dscnt 0x0
	v_fma_f32 v38, -v108, v106, v38
	ds_bpermute_b32 v106, v101, v107
	s_wait_dscnt 0x0
	v_fma_f32 v39, -v108, v106, v39
	ds_bpermute_b32 v106, v102, v107
	s_wait_dscnt 0x0
	v_fma_f32 v40, -v108, v106, v40
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v106, 0, v18, s1
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v106, v106, v34, s2
	ds_bpermute_b32 v107, v87, v106
	s_wait_dscnt 0x0
	v_add_f32_e32 v108, v106, v107
	ds_load_2addr_b32 v[106:107], v109 offset0:64 offset1:96
	s_wait_dscnt 0x0
	ds_bpermute_b32 v110, v104, v106
	s_wait_dscnt 0x0
	v_fma_f32 v9, -v108, v110, v9
	ds_bpermute_b32 v110, v88, v106
	s_wait_dscnt 0x0
	v_fma_f32 v10, -v108, v110, v10
	ds_bpermute_b32 v110, v89, v106
	s_wait_dscnt 0x0
	v_fma_f32 v11, -v108, v110, v11
	ds_bpermute_b32 v110, v90, v106
	s_wait_dscnt 0x0
	v_fma_f32 v12, -v108, v110, v12
	ds_bpermute_b32 v110, v91, v106
	s_wait_dscnt 0x0
	v_fma_f32 v13, -v108, v110, v13
	ds_bpermute_b32 v110, v92, v106
	s_wait_dscnt 0x0
	v_fma_f32 v14, -v108, v110, v14
	ds_bpermute_b32 v110, v93, v106
	s_wait_dscnt 0x0
	v_fma_f32 v15, -v108, v110, v15
	ds_bpermute_b32 v110, v94, v106
	s_wait_dscnt 0x0
	v_fma_f32 v16, -v108, v110, v16
	ds_bpermute_b32 v110, v95, v106
	s_wait_dscnt 0x0
	v_fma_f32 v17, -v108, v110, v17
	ds_bpermute_b32 v110, v96, v106
	s_wait_dscnt 0x0
	v_fma_f32 v18, -v108, v110, v18
	ds_bpermute_b32 v110, v97, v106
	s_wait_dscnt 0x0
	v_fma_f32 v19, -v108, v110, v19
	ds_bpermute_b32 v110, v98, v106
	s_wait_dscnt 0x0
	v_fma_f32 v20, -v108, v110, v20
	ds_bpermute_b32 v110, v99, v106
	s_wait_dscnt 0x0
	v_fma_f32 v21, -v108, v110, v21
	ds_bpermute_b32 v110, v100, v106
	s_wait_dscnt 0x0
	v_fma_f32 v22, -v108, v110, v22
	ds_bpermute_b32 v110, v101, v106
	ds_bpermute_b32 v106, v102, v106
	s_wait_dscnt 0x1
	v_fma_f32 v23, -v108, v110, v23
	s_wait_dscnt 0x0
	v_fma_f32 v24, -v108, v106, v24
	ds_bpermute_b32 v106, v104, v107
	s_wait_dscnt 0x0
	v_fma_f32 v25, -v108, v106, v25
	ds_bpermute_b32 v106, v88, v107
	s_wait_dscnt 0x0
	v_fma_f32 v26, -v108, v106, v26
	ds_bpermute_b32 v106, v89, v107
	s_wait_dscnt 0x0
	v_fma_f32 v27, -v108, v106, v27
	ds_bpermute_b32 v106, v90, v107
	s_wait_dscnt 0x0
	v_fma_f32 v28, -v108, v106, v28
	ds_bpermute_b32 v106, v91, v107
	s_wait_dscnt 0x0
	v_fma_f32 v29, -v108, v106, v29
	ds_bpermute_b32 v106, v92, v107
	s_wait_dscnt 0x0
	v_fma_f32 v30, -v108, v106, v30
	ds_bpermute_b32 v106, v93, v107
	s_wait_dscnt 0x0
	v_fma_f32 v31, -v108, v106, v31
	ds_bpermute_b32 v106, v94, v107
	s_wait_dscnt 0x0
	v_fma_f32 v32, -v108, v106, v32
	ds_bpermute_b32 v106, v95, v107
	s_wait_dscnt 0x0
	v_fma_f32 v33, -v108, v106, v33
	ds_bpermute_b32 v106, v96, v107
	s_wait_dscnt 0x0
	v_fma_f32 v34, -v108, v106, v34
	ds_bpermute_b32 v106, v97, v107
	s_wait_dscnt 0x0
	v_fma_f32 v35, -v108, v106, v35
	ds_bpermute_b32 v106, v98, v107
	s_wait_dscnt 0x0
	v_fma_f32 v36, -v108, v106, v36
	ds_bpermute_b32 v106, v99, v107
	s_wait_dscnt 0x0
	v_fma_f32 v37, -v108, v106, v37
	ds_bpermute_b32 v106, v100, v107
	s_wait_dscnt 0x0
	v_fma_f32 v38, -v108, v106, v38
	ds_bpermute_b32 v106, v101, v107
	s_wait_dscnt 0x0
	v_fma_f32 v39, -v108, v106, v39
	ds_bpermute_b32 v106, v102, v107
	s_wait_dscnt 0x0
	v_fma_f32 v40, -v108, v106, v40
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v106, 0, v19, s1
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v106, v106, v35, s2
	ds_bpermute_b32 v107, v87, v106
	s_wait_dscnt 0x0
	v_add_f32_e32 v108, v106, v107
	ds_load_2addr_b32 v[106:107], v109 offset0:128 offset1:160
	s_wait_dscnt 0x0
	ds_bpermute_b32 v110, v104, v106
	s_wait_dscnt 0x0
	v_fma_f32 v9, -v108, v110, v9
	ds_bpermute_b32 v110, v88, v106
	s_wait_dscnt 0x0
	v_fma_f32 v10, -v108, v110, v10
	ds_bpermute_b32 v110, v89, v106
	s_wait_dscnt 0x0
	v_fma_f32 v11, -v108, v110, v11
	ds_bpermute_b32 v110, v90, v106
	s_wait_dscnt 0x0
	v_fma_f32 v12, -v108, v110, v12
	ds_bpermute_b32 v110, v91, v106
	s_wait_dscnt 0x0
	v_fma_f32 v13, -v108, v110, v13
	ds_bpermute_b32 v110, v92, v106
	s_wait_dscnt 0x0
	v_fma_f32 v14, -v108, v110, v14
	ds_bpermute_b32 v110, v93, v106
	s_wait_dscnt 0x0
	v_fma_f32 v15, -v108, v110, v15
	ds_bpermute_b32 v110, v94, v106
	s_wait_dscnt 0x0
	v_fma_f32 v16, -v108, v110, v16
	ds_bpermute_b32 v110, v95, v106
	s_wait_dscnt 0x0
	v_fma_f32 v17, -v108, v110, v17
	ds_bpermute_b32 v110, v96, v106
	s_wait_dscnt 0x0
	v_fma_f32 v18, -v108, v110, v18
	ds_bpermute_b32 v110, v97, v106
	s_wait_dscnt 0x0
	v_fma_f32 v19, -v108, v110, v19
	ds_bpermute_b32 v110, v98, v106
	s_wait_dscnt 0x0
	v_fma_f32 v20, -v108, v110, v20
	ds_bpermute_b32 v110, v99, v106
	s_wait_dscnt 0x0
	v_fma_f32 v21, -v108, v110, v21
	ds_bpermute_b32 v110, v100, v106
	s_wait_dscnt 0x0
	v_fma_f32 v22, -v108, v110, v22
	ds_bpermute_b32 v110, v101, v106
	ds_bpermute_b32 v106, v102, v106
	s_wait_dscnt 0x1
	v_fma_f32 v23, -v108, v110, v23
	s_wait_dscnt 0x0
	v_fma_f32 v24, -v108, v106, v24
	ds_bpermute_b32 v106, v104, v107
	s_wait_dscnt 0x0
	v_fma_f32 v25, -v108, v106, v25
	ds_bpermute_b32 v106, v88, v107
	s_wait_dscnt 0x0
	v_fma_f32 v26, -v108, v106, v26
	ds_bpermute_b32 v106, v89, v107
	s_wait_dscnt 0x0
	v_fma_f32 v27, -v108, v106, v27
	ds_bpermute_b32 v106, v90, v107
	s_wait_dscnt 0x0
	v_fma_f32 v28, -v108, v106, v28
	ds_bpermute_b32 v106, v91, v107
	s_wait_dscnt 0x0
	v_fma_f32 v29, -v108, v106, v29
	ds_bpermute_b32 v106, v92, v107
	s_wait_dscnt 0x0
	v_fma_f32 v30, -v108, v106, v30
	ds_bpermute_b32 v106, v93, v107
	s_wait_dscnt 0x0
	v_fma_f32 v31, -v108, v106, v31
	ds_bpermute_b32 v106, v94, v107
	s_wait_dscnt 0x0
	v_fma_f32 v32, -v108, v106, v32
	ds_bpermute_b32 v106, v95, v107
	s_wait_dscnt 0x0
	v_fma_f32 v33, -v108, v106, v33
	ds_bpermute_b32 v106, v96, v107
	s_wait_dscnt 0x0
	v_fma_f32 v34, -v108, v106, v34
	ds_bpermute_b32 v106, v97, v107
	s_wait_dscnt 0x0
	v_fma_f32 v35, -v108, v106, v35
	ds_bpermute_b32 v106, v98, v107
	s_wait_dscnt 0x0
	v_fma_f32 v36, -v108, v106, v36
	ds_bpermute_b32 v106, v99, v107
	s_wait_dscnt 0x0
	v_fma_f32 v37, -v108, v106, v37
	ds_bpermute_b32 v106, v100, v107
	s_wait_dscnt 0x0
	v_fma_f32 v38, -v108, v106, v38
	ds_bpermute_b32 v106, v101, v107
	s_wait_dscnt 0x0
	v_fma_f32 v39, -v108, v106, v39
	ds_bpermute_b32 v106, v102, v107
	s_wait_dscnt 0x0
	v_fma_f32 v40, -v108, v106, v40
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v106, 0, v20, s1
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v106, v106, v36, s2
	ds_bpermute_b32 v107, v87, v106
	s_wait_dscnt 0x0
	v_add_f32_e32 v108, v106, v107
	ds_load_2addr_b32 v[106:107], v109 offset0:192 offset1:224
	s_wait_dscnt 0x0
	ds_bpermute_b32 v109, v104, v106
	s_wait_dscnt 0x0
	v_fma_f32 v9, -v108, v109, v9
	ds_bpermute_b32 v109, v88, v106
	s_wait_dscnt 0x0
	v_fma_f32 v10, -v108, v109, v10
	ds_bpermute_b32 v109, v89, v106
	s_wait_dscnt 0x0
	v_fma_f32 v11, -v108, v109, v11
	ds_bpermute_b32 v109, v90, v106
	s_wait_dscnt 0x0
	v_fma_f32 v12, -v108, v109, v12
	ds_bpermute_b32 v109, v91, v106
	s_wait_dscnt 0x0
	v_fma_f32 v13, -v108, v109, v13
	ds_bpermute_b32 v109, v92, v106
	s_wait_dscnt 0x0
	v_fma_f32 v14, -v108, v109, v14
	ds_bpermute_b32 v109, v93, v106
	s_wait_dscnt 0x0
	v_fma_f32 v15, -v108, v109, v15
	ds_bpermute_b32 v109, v94, v106
	s_wait_dscnt 0x0
	v_fma_f32 v16, -v108, v109, v16
	ds_bpermute_b32 v109, v95, v106
	s_wait_dscnt 0x0
	v_fma_f32 v17, -v108, v109, v17
	ds_bpermute_b32 v109, v96, v106
	s_wait_dscnt 0x0
	v_fma_f32 v18, -v108, v109, v18
	ds_bpermute_b32 v109, v97, v106
	s_wait_dscnt 0x0
	v_fma_f32 v19, -v108, v109, v19
	ds_bpermute_b32 v109, v98, v106
	s_wait_dscnt 0x0
	v_fma_f32 v20, -v108, v109, v20
	ds_bpermute_b32 v109, v99, v106
	s_wait_dscnt 0x0
	v_fma_f32 v21, -v108, v109, v21
	ds_bpermute_b32 v109, v100, v106
	s_wait_dscnt 0x0
	v_fma_f32 v22, -v108, v109, v22
	ds_bpermute_b32 v109, v101, v106
	ds_bpermute_b32 v106, v102, v106
	s_wait_dscnt 0x1
	v_fma_f32 v23, -v108, v109, v23
	s_wait_dscnt 0x0
	v_fma_f32 v24, -v108, v106, v24
	ds_bpermute_b32 v106, v104, v107
	v_add_nc_u32_e32 v109, 0x1400, v105
	v_add_nc_u32_e32 v105, 0x1c00, v105
	s_wait_dscnt 0x0
	v_fma_f32 v25, -v108, v106, v25
	ds_bpermute_b32 v106, v88, v107
	s_wait_dscnt 0x0
	v_fma_f32 v26, -v108, v106, v26
	ds_bpermute_b32 v106, v89, v107
	s_wait_dscnt 0x0
	v_fma_f32 v27, -v108, v106, v27
	ds_bpermute_b32 v106, v90, v107
	s_wait_dscnt 0x0
	v_fma_f32 v28, -v108, v106, v28
	ds_bpermute_b32 v106, v91, v107
	s_wait_dscnt 0x0
	v_fma_f32 v29, -v108, v106, v29
	ds_bpermute_b32 v106, v92, v107
	s_wait_dscnt 0x0
	v_fma_f32 v30, -v108, v106, v30
	ds_bpermute_b32 v106, v93, v107
	s_wait_dscnt 0x0
	v_fma_f32 v31, -v108, v106, v31
	ds_bpermute_b32 v106, v94, v107
	s_wait_dscnt 0x0
	v_fma_f32 v32, -v108, v106, v32
	ds_bpermute_b32 v106, v95, v107
	s_wait_dscnt 0x0
	v_fma_f32 v33, -v108, v106, v33
	ds_bpermute_b32 v106, v96, v107
	s_wait_dscnt 0x0
	v_fma_f32 v34, -v108, v106, v34
	ds_bpermute_b32 v106, v97, v107
	s_wait_dscnt 0x0
	v_fma_f32 v35, -v108, v106, v35
	ds_bpermute_b32 v106, v98, v107
	s_wait_dscnt 0x0
	v_fma_f32 v36, -v108, v106, v36
	ds_bpermute_b32 v106, v99, v107
	s_wait_dscnt 0x0
	v_fma_f32 v37, -v108, v106, v37
	ds_bpermute_b32 v106, v100, v107
	s_wait_dscnt 0x0
	v_fma_f32 v38, -v108, v106, v38
	ds_bpermute_b32 v106, v101, v107
	s_wait_dscnt 0x0
	v_fma_f32 v39, -v108, v106, v39
	ds_bpermute_b32 v106, v102, v107
	s_wait_dscnt 0x0
	v_fma_f32 v40, -v108, v106, v40
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v106, 0, v21, s1
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v106, v106, v37, s2
	ds_bpermute_b32 v107, v87, v106
	s_wait_dscnt 0x0
	v_add_f32_e32 v108, v106, v107
	ds_load_2addr_b32 v[106:107], v109 offset1:32
	s_wait_dscnt 0x0
	ds_bpermute_b32 v110, v104, v106
	s_wait_dscnt 0x0
	v_fma_f32 v9, -v108, v110, v9
	ds_bpermute_b32 v110, v88, v106
	s_wait_dscnt 0x0
	v_fma_f32 v10, -v108, v110, v10
	ds_bpermute_b32 v110, v89, v106
	s_wait_dscnt 0x0
	v_fma_f32 v11, -v108, v110, v11
	ds_bpermute_b32 v110, v90, v106
	s_wait_dscnt 0x0
	v_fma_f32 v12, -v108, v110, v12
	ds_bpermute_b32 v110, v91, v106
	s_wait_dscnt 0x0
	v_fma_f32 v13, -v108, v110, v13
	ds_bpermute_b32 v110, v92, v106
	s_wait_dscnt 0x0
	v_fma_f32 v14, -v108, v110, v14
	ds_bpermute_b32 v110, v93, v106
	s_wait_dscnt 0x0
	v_fma_f32 v15, -v108, v110, v15
	ds_bpermute_b32 v110, v94, v106
	s_wait_dscnt 0x0
	v_fma_f32 v16, -v108, v110, v16
	ds_bpermute_b32 v110, v95, v106
	s_wait_dscnt 0x0
	v_fma_f32 v17, -v108, v110, v17
	ds_bpermute_b32 v110, v96, v106
	s_wait_dscnt 0x0
	v_fma_f32 v18, -v108, v110, v18
	ds_bpermute_b32 v110, v97, v106
	s_wait_dscnt 0x0
	v_fma_f32 v19, -v108, v110, v19
	ds_bpermute_b32 v110, v98, v106
	s_wait_dscnt 0x0
	v_fma_f32 v20, -v108, v110, v20
	ds_bpermute_b32 v110, v99, v106
	s_wait_dscnt 0x0
	v_fma_f32 v21, -v108, v110, v21
	ds_bpermute_b32 v110, v100, v106
	s_wait_dscnt 0x0
	v_fma_f32 v22, -v108, v110, v22
	ds_bpermute_b32 v110, v101, v106
	ds_bpermute_b32 v106, v102, v106
	s_wait_dscnt 0x1
	v_fma_f32 v23, -v108, v110, v23
	s_wait_dscnt 0x0
	v_fma_f32 v24, -v108, v106, v24
	ds_bpermute_b32 v106, v104, v107
	s_wait_dscnt 0x0
	v_fma_f32 v25, -v108, v106, v25
	ds_bpermute_b32 v106, v88, v107
	s_wait_dscnt 0x0
	v_fma_f32 v26, -v108, v106, v26
	ds_bpermute_b32 v106, v89, v107
	s_wait_dscnt 0x0
	v_fma_f32 v27, -v108, v106, v27
	ds_bpermute_b32 v106, v90, v107
	s_wait_dscnt 0x0
	v_fma_f32 v28, -v108, v106, v28
	ds_bpermute_b32 v106, v91, v107
	s_wait_dscnt 0x0
	v_fma_f32 v29, -v108, v106, v29
	ds_bpermute_b32 v106, v92, v107
	s_wait_dscnt 0x0
	v_fma_f32 v30, -v108, v106, v30
	ds_bpermute_b32 v106, v93, v107
	s_wait_dscnt 0x0
	v_fma_f32 v31, -v108, v106, v31
	ds_bpermute_b32 v106, v94, v107
	s_wait_dscnt 0x0
	v_fma_f32 v32, -v108, v106, v32
	ds_bpermute_b32 v106, v95, v107
	s_wait_dscnt 0x0
	v_fma_f32 v33, -v108, v106, v33
	ds_bpermute_b32 v106, v96, v107
	s_wait_dscnt 0x0
	v_fma_f32 v34, -v108, v106, v34
	ds_bpermute_b32 v106, v97, v107
	s_wait_dscnt 0x0
	v_fma_f32 v35, -v108, v106, v35
	ds_bpermute_b32 v106, v98, v107
	s_wait_dscnt 0x0
	v_fma_f32 v36, -v108, v106, v36
	ds_bpermute_b32 v106, v99, v107
	s_wait_dscnt 0x0
	v_fma_f32 v37, -v108, v106, v37
	ds_bpermute_b32 v106, v100, v107
	s_wait_dscnt 0x0
	v_fma_f32 v38, -v108, v106, v38
	ds_bpermute_b32 v106, v101, v107
	s_wait_dscnt 0x0
	v_fma_f32 v39, -v108, v106, v39
	ds_bpermute_b32 v106, v102, v107
	s_wait_dscnt 0x0
	v_fma_f32 v40, -v108, v106, v40
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v106, 0, v22, s1
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v106, v106, v38, s2
	ds_bpermute_b32 v107, v87, v106
	s_wait_dscnt 0x0
	v_add_f32_e32 v108, v106, v107
	ds_load_2addr_b32 v[106:107], v109 offset0:64 offset1:96
	s_wait_dscnt 0x0
	ds_bpermute_b32 v110, v104, v106
	s_wait_dscnt 0x0
	v_fma_f32 v9, -v108, v110, v9
	ds_bpermute_b32 v110, v88, v106
	s_wait_dscnt 0x0
	v_fma_f32 v10, -v108, v110, v10
	ds_bpermute_b32 v110, v89, v106
	s_wait_dscnt 0x0
	v_fma_f32 v11, -v108, v110, v11
	ds_bpermute_b32 v110, v90, v106
	s_wait_dscnt 0x0
	v_fma_f32 v12, -v108, v110, v12
	ds_bpermute_b32 v110, v91, v106
	s_wait_dscnt 0x0
	v_fma_f32 v13, -v108, v110, v13
	ds_bpermute_b32 v110, v92, v106
	s_wait_dscnt 0x0
	v_fma_f32 v14, -v108, v110, v14
	ds_bpermute_b32 v110, v93, v106
	s_wait_dscnt 0x0
	v_fma_f32 v15, -v108, v110, v15
	ds_bpermute_b32 v110, v94, v106
	s_wait_dscnt 0x0
	v_fma_f32 v16, -v108, v110, v16
	ds_bpermute_b32 v110, v95, v106
	s_wait_dscnt 0x0
	v_fma_f32 v17, -v108, v110, v17
	ds_bpermute_b32 v110, v96, v106
	s_wait_dscnt 0x0
	v_fma_f32 v18, -v108, v110, v18
	ds_bpermute_b32 v110, v97, v106
	s_wait_dscnt 0x0
	v_fma_f32 v19, -v108, v110, v19
	ds_bpermute_b32 v110, v98, v106
	s_wait_dscnt 0x0
	v_fma_f32 v20, -v108, v110, v20
	ds_bpermute_b32 v110, v99, v106
	s_wait_dscnt 0x0
	v_fma_f32 v21, -v108, v110, v21
	ds_bpermute_b32 v110, v100, v106
	s_wait_dscnt 0x0
	v_fma_f32 v22, -v108, v110, v22
	ds_bpermute_b32 v110, v101, v106
	ds_bpermute_b32 v106, v102, v106
	s_wait_dscnt 0x1
	v_fma_f32 v23, -v108, v110, v23
	s_wait_dscnt 0x0
	v_fma_f32 v24, -v108, v106, v24
	ds_bpermute_b32 v106, v104, v107
	s_wait_dscnt 0x0
	v_fma_f32 v25, -v108, v106, v25
	ds_bpermute_b32 v106, v88, v107
	s_wait_dscnt 0x0
	v_fma_f32 v26, -v108, v106, v26
	ds_bpermute_b32 v106, v89, v107
	s_wait_dscnt 0x0
	v_fma_f32 v27, -v108, v106, v27
	ds_bpermute_b32 v106, v90, v107
	s_wait_dscnt 0x0
	v_fma_f32 v28, -v108, v106, v28
	ds_bpermute_b32 v106, v91, v107
	s_wait_dscnt 0x0
	v_fma_f32 v29, -v108, v106, v29
	ds_bpermute_b32 v106, v92, v107
	s_wait_dscnt 0x0
	v_fma_f32 v30, -v108, v106, v30
	ds_bpermute_b32 v106, v93, v107
	s_wait_dscnt 0x0
	v_fma_f32 v31, -v108, v106, v31
	ds_bpermute_b32 v106, v94, v107
	s_wait_dscnt 0x0
	v_fma_f32 v32, -v108, v106, v32
	ds_bpermute_b32 v106, v95, v107
	s_wait_dscnt 0x0
	v_fma_f32 v33, -v108, v106, v33
	ds_bpermute_b32 v106, v96, v107
	s_wait_dscnt 0x0
	v_fma_f32 v34, -v108, v106, v34
	ds_bpermute_b32 v106, v97, v107
	s_wait_dscnt 0x0
	v_fma_f32 v35, -v108, v106, v35
	ds_bpermute_b32 v106, v98, v107
	s_wait_dscnt 0x0
	v_fma_f32 v36, -v108, v106, v36
	ds_bpermute_b32 v106, v99, v107
	s_wait_dscnt 0x0
	v_fma_f32 v37, -v108, v106, v37
	ds_bpermute_b32 v106, v100, v107
	s_wait_dscnt 0x0
	v_fma_f32 v38, -v108, v106, v38
	ds_bpermute_b32 v106, v101, v107
	s_wait_dscnt 0x0
	v_fma_f32 v39, -v108, v106, v39
	ds_bpermute_b32 v106, v102, v107
	s_wait_dscnt 0x0
	v_fma_f32 v40, -v108, v106, v40
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v106, 0, v23, s1
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v106, v106, v39, s2
	ds_bpermute_b32 v107, v87, v106
	s_wait_dscnt 0x0
	v_add_f32_e32 v108, v106, v107
	ds_load_2addr_b32 v[106:107], v109 offset0:128 offset1:160
	s_wait_dscnt 0x0
	ds_bpermute_b32 v110, v104, v106
	s_wait_dscnt 0x0
	v_fma_f32 v9, -v108, v110, v9
	ds_bpermute_b32 v110, v88, v106
	s_wait_dscnt 0x0
	v_fma_f32 v10, -v108, v110, v10
	ds_bpermute_b32 v110, v89, v106
	s_wait_dscnt 0x0
	v_fma_f32 v11, -v108, v110, v11
	ds_bpermute_b32 v110, v90, v106
	s_wait_dscnt 0x0
	v_fma_f32 v12, -v108, v110, v12
	ds_bpermute_b32 v110, v91, v106
	s_wait_dscnt 0x0
	v_fma_f32 v13, -v108, v110, v13
	ds_bpermute_b32 v110, v92, v106
	s_wait_dscnt 0x0
	v_fma_f32 v14, -v108, v110, v14
	ds_bpermute_b32 v110, v93, v106
	s_wait_dscnt 0x0
	v_fma_f32 v15, -v108, v110, v15
	ds_bpermute_b32 v110, v94, v106
	s_wait_dscnt 0x0
	v_fma_f32 v16, -v108, v110, v16
	ds_bpermute_b32 v110, v95, v106
	s_wait_dscnt 0x0
	v_fma_f32 v17, -v108, v110, v17
	ds_bpermute_b32 v110, v96, v106
	s_wait_dscnt 0x0
	v_fma_f32 v18, -v108, v110, v18
	ds_bpermute_b32 v110, v97, v106
	s_wait_dscnt 0x0
	v_fma_f32 v19, -v108, v110, v19
	ds_bpermute_b32 v110, v98, v106
	s_wait_dscnt 0x0
	v_fma_f32 v20, -v108, v110, v20
	ds_bpermute_b32 v110, v99, v106
	s_wait_dscnt 0x0
	v_fma_f32 v21, -v108, v110, v21
	ds_bpermute_b32 v110, v100, v106
	s_wait_dscnt 0x0
	v_fma_f32 v22, -v108, v110, v22
	ds_bpermute_b32 v110, v101, v106
	ds_bpermute_b32 v106, v102, v106
	s_wait_dscnt 0x1
	v_fma_f32 v23, -v108, v110, v23
	s_wait_dscnt 0x0
	v_fma_f32 v24, -v108, v106, v24
	ds_bpermute_b32 v106, v104, v107
	s_wait_dscnt 0x0
	v_fma_f32 v25, -v108, v106, v25
	ds_bpermute_b32 v106, v88, v107
	s_wait_dscnt 0x0
	v_fma_f32 v26, -v108, v106, v26
	ds_bpermute_b32 v106, v89, v107
	s_wait_dscnt 0x0
	v_fma_f32 v27, -v108, v106, v27
	ds_bpermute_b32 v106, v90, v107
	s_wait_dscnt 0x0
	v_fma_f32 v28, -v108, v106, v28
	ds_bpermute_b32 v106, v91, v107
	s_wait_dscnt 0x0
	v_fma_f32 v29, -v108, v106, v29
	ds_bpermute_b32 v106, v92, v107
	s_wait_dscnt 0x0
	v_fma_f32 v30, -v108, v106, v30
	ds_bpermute_b32 v106, v93, v107
	s_wait_dscnt 0x0
	v_fma_f32 v31, -v108, v106, v31
	ds_bpermute_b32 v106, v94, v107
	s_wait_dscnt 0x0
	v_fma_f32 v32, -v108, v106, v32
	ds_bpermute_b32 v106, v95, v107
	s_wait_dscnt 0x0
	v_fma_f32 v33, -v108, v106, v33
	ds_bpermute_b32 v106, v96, v107
	s_wait_dscnt 0x0
	v_fma_f32 v34, -v108, v106, v34
	ds_bpermute_b32 v106, v97, v107
	s_wait_dscnt 0x0
	v_fma_f32 v35, -v108, v106, v35
	ds_bpermute_b32 v106, v98, v107
	s_wait_dscnt 0x0
	v_fma_f32 v36, -v108, v106, v36
	ds_bpermute_b32 v106, v99, v107
	s_wait_dscnt 0x0
	v_fma_f32 v37, -v108, v106, v37
	ds_bpermute_b32 v106, v100, v107
	s_wait_dscnt 0x0
	v_fma_f32 v38, -v108, v106, v38
	ds_bpermute_b32 v106, v101, v107
	s_wait_dscnt 0x0
	v_fma_f32 v39, -v108, v106, v39
	ds_bpermute_b32 v106, v102, v107
	s_wait_dscnt 0x0
	v_fma_f32 v40, -v108, v106, v40
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v106, 0, v24, s1
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v106, v106, v40, s2
	ds_bpermute_b32 v107, v87, v106
	s_wait_dscnt 0x0
	v_add_f32_e32 v108, v106, v107
	ds_load_2addr_b32 v[106:107], v109 offset0:192 offset1:224
	s_wait_dscnt 0x0
	ds_bpermute_b32 v109, v104, v106
	s_wait_dscnt 0x0
	v_fma_f32 v9, -v108, v109, v9
	ds_bpermute_b32 v109, v88, v106
	s_wait_dscnt 0x0
	v_fma_f32 v10, -v108, v109, v10
	ds_bpermute_b32 v109, v89, v106
	s_wait_dscnt 0x0
	v_fma_f32 v11, -v108, v109, v11
	ds_bpermute_b32 v109, v90, v106
	s_wait_dscnt 0x0
	v_fma_f32 v12, -v108, v109, v12
	ds_bpermute_b32 v109, v91, v106
	s_wait_dscnt 0x0
	v_fma_f32 v13, -v108, v109, v13
	ds_bpermute_b32 v109, v92, v106
	s_wait_dscnt 0x0
	v_fma_f32 v14, -v108, v109, v14
	ds_bpermute_b32 v109, v93, v106
	s_wait_dscnt 0x0
	v_fma_f32 v15, -v108, v109, v15
	ds_bpermute_b32 v109, v94, v106
	s_wait_dscnt 0x0
	v_fma_f32 v16, -v108, v109, v16
	ds_bpermute_b32 v109, v95, v106
	s_wait_dscnt 0x0
	v_fma_f32 v17, -v108, v109, v17
	ds_bpermute_b32 v109, v96, v106
	s_wait_dscnt 0x0
	v_fma_f32 v18, -v108, v109, v18
	ds_bpermute_b32 v109, v97, v106
	s_wait_dscnt 0x0
	v_fma_f32 v19, -v108, v109, v19
	ds_bpermute_b32 v109, v98, v106
	s_wait_dscnt 0x0
	v_fma_f32 v20, -v108, v109, v20
	ds_bpermute_b32 v109, v99, v106
	s_wait_dscnt 0x0
	v_fma_f32 v21, -v108, v109, v21
	ds_bpermute_b32 v109, v100, v106
	s_wait_dscnt 0x0
	v_fma_f32 v22, -v108, v109, v22
	ds_bpermute_b32 v109, v101, v106
	ds_bpermute_b32 v106, v102, v106
	s_wait_dscnt 0x1
	v_fma_f32 v23, -v108, v109, v23
	s_wait_dscnt 0x0
	v_fma_f32 v24, -v108, v106, v24
	ds_bpermute_b32 v106, v104, v107
	s_wait_dscnt 0x0
	v_fma_f32 v25, -v108, v106, v25
	ds_bpermute_b32 v106, v88, v107
	s_wait_dscnt 0x0
	v_fma_f32 v26, -v108, v106, v26
	ds_bpermute_b32 v106, v89, v107
	s_wait_dscnt 0x0
	v_fma_f32 v27, -v108, v106, v27
	ds_bpermute_b32 v106, v90, v107
	s_wait_dscnt 0x0
	v_fma_f32 v28, -v108, v106, v28
	ds_bpermute_b32 v106, v91, v107
	s_wait_dscnt 0x0
	v_fma_f32 v29, -v108, v106, v29
	ds_bpermute_b32 v106, v92, v107
	s_wait_dscnt 0x0
	v_fma_f32 v30, -v108, v106, v30
	ds_bpermute_b32 v106, v93, v107
	s_wait_dscnt 0x0
	v_fma_f32 v31, -v108, v106, v31
	ds_bpermute_b32 v106, v94, v107
	s_wait_dscnt 0x0
	v_fma_f32 v32, -v108, v106, v32
	ds_bpermute_b32 v106, v95, v107
	s_wait_dscnt 0x0
	v_fma_f32 v33, -v108, v106, v33
	ds_bpermute_b32 v106, v96, v107
	s_wait_dscnt 0x0
	v_fma_f32 v34, -v108, v106, v34
	ds_bpermute_b32 v106, v97, v107
	s_wait_dscnt 0x0
	v_fma_f32 v35, -v108, v106, v35
	ds_bpermute_b32 v106, v98, v107
	s_wait_dscnt 0x0
	v_fma_f32 v36, -v108, v106, v36
	ds_bpermute_b32 v106, v99, v107
	s_wait_dscnt 0x0
	v_fma_f32 v37, -v108, v106, v37
	ds_bpermute_b32 v106, v100, v107
	s_wait_dscnt 0x0
	v_fma_f32 v38, -v108, v106, v38
	ds_bpermute_b32 v106, v101, v107
	s_wait_dscnt 0x0
	v_fma_f32 v39, -v108, v106, v39
	ds_bpermute_b32 v106, v102, v107
	s_wait_dscnt 0x0
	v_fma_f32 v40, -v108, v106, v40
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e32 v106, 0, v17, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v106, v106, v33, s0
	ds_bpermute_b32 v107, v87, v106
	s_wait_dscnt 0x0
	v_add_f32_e32 v108, v106, v107
	ds_load_2addr_b32 v[106:107], v122 offset1:32
	s_wait_dscnt 0x0
	ds_bpermute_b32 v109, v104, v106
	s_wait_dscnt 0x0
	v_fma_f32 v9, -v108, v109, v9
	ds_bpermute_b32 v109, v88, v106
	s_wait_dscnt 0x0
	v_fma_f32 v10, -v108, v109, v10
	ds_bpermute_b32 v109, v89, v106
	s_wait_dscnt 0x0
	v_fma_f32 v11, -v108, v109, v11
	ds_bpermute_b32 v109, v90, v106
	s_wait_dscnt 0x0
	v_fma_f32 v12, -v108, v109, v12
	ds_bpermute_b32 v109, v91, v106
	s_wait_dscnt 0x0
	v_fma_f32 v13, -v108, v109, v13
	ds_bpermute_b32 v109, v92, v106
	s_wait_dscnt 0x0
	v_fma_f32 v14, -v108, v109, v14
	ds_bpermute_b32 v109, v93, v106
	s_wait_dscnt 0x0
	v_fma_f32 v15, -v108, v109, v15
	ds_bpermute_b32 v109, v94, v106
	s_wait_dscnt 0x0
	v_fma_f32 v16, -v108, v109, v16
	ds_bpermute_b32 v109, v95, v106
	s_wait_dscnt 0x0
	v_fma_f32 v17, -v108, v109, v17
	ds_bpermute_b32 v109, v96, v106
	s_wait_dscnt 0x0
	v_fma_f32 v18, -v108, v109, v18
	ds_bpermute_b32 v109, v97, v106
	s_wait_dscnt 0x0
	v_fma_f32 v19, -v108, v109, v19
	ds_bpermute_b32 v109, v98, v106
	s_wait_dscnt 0x0
	v_fma_f32 v20, -v108, v109, v20
	ds_bpermute_b32 v109, v99, v106
	s_wait_dscnt 0x0
	v_fma_f32 v21, -v108, v109, v21
	ds_bpermute_b32 v109, v100, v106
	s_wait_dscnt 0x0
	v_fma_f32 v22, -v108, v109, v22
	ds_bpermute_b32 v109, v101, v106
	ds_bpermute_b32 v106, v102, v106
	s_wait_dscnt 0x1
	v_fma_f32 v23, -v108, v109, v23
	s_wait_dscnt 0x0
	v_fma_f32 v24, -v108, v106, v24
	ds_bpermute_b32 v106, v104, v107
	s_wait_dscnt 0x0
	v_fma_f32 v25, -v108, v106, v25
	ds_bpermute_b32 v106, v88, v107
	s_wait_dscnt 0x0
	v_fma_f32 v26, -v108, v106, v26
	ds_bpermute_b32 v106, v89, v107
	s_wait_dscnt 0x0
	v_fma_f32 v27, -v108, v106, v27
	ds_bpermute_b32 v106, v90, v107
	s_wait_dscnt 0x0
	v_fma_f32 v28, -v108, v106, v28
	ds_bpermute_b32 v106, v91, v107
	s_wait_dscnt 0x0
	v_fma_f32 v29, -v108, v106, v29
	ds_bpermute_b32 v106, v92, v107
	s_wait_dscnt 0x0
	v_fma_f32 v30, -v108, v106, v30
	ds_bpermute_b32 v106, v93, v107
	s_wait_dscnt 0x0
	v_fma_f32 v31, -v108, v106, v31
	ds_bpermute_b32 v106, v94, v107
	s_wait_dscnt 0x0
	v_fma_f32 v32, -v108, v106, v32
	ds_bpermute_b32 v106, v95, v107
	s_wait_dscnt 0x0
	v_fma_f32 v33, -v108, v106, v33
	ds_bpermute_b32 v106, v96, v107
	s_wait_dscnt 0x0
	v_fma_f32 v34, -v108, v106, v34
	ds_bpermute_b32 v106, v97, v107
	s_wait_dscnt 0x0
	v_fma_f32 v35, -v108, v106, v35
	ds_bpermute_b32 v106, v98, v107
	s_wait_dscnt 0x0
	v_fma_f32 v36, -v108, v106, v36
	ds_bpermute_b32 v106, v99, v107
	s_wait_dscnt 0x0
	v_fma_f32 v37, -v108, v106, v37
	ds_bpermute_b32 v106, v100, v107
	s_wait_dscnt 0x0
	v_fma_f32 v38, -v108, v106, v38
	ds_bpermute_b32 v106, v101, v107
	s_wait_dscnt 0x0
	v_fma_f32 v39, -v108, v106, v39
	ds_bpermute_b32 v106, v102, v107
	s_wait_dscnt 0x0
	v_fma_f32 v40, -v108, v106, v40
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e32 v106, 0, v18, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v106, v106, v34, s0
	ds_bpermute_b32 v107, v87, v106
	s_wait_dscnt 0x0
	v_add_f32_e32 v108, v106, v107
	ds_load_2addr_b32 v[106:107], v122 offset0:64 offset1:96
	s_wait_dscnt 0x0
	ds_bpermute_b32 v109, v104, v106
	s_wait_dscnt 0x0
	v_fma_f32 v9, -v108, v109, v9
	ds_bpermute_b32 v109, v88, v106
	s_wait_dscnt 0x0
	v_fma_f32 v10, -v108, v109, v10
	ds_bpermute_b32 v109, v89, v106
	s_wait_dscnt 0x0
	v_fma_f32 v11, -v108, v109, v11
	ds_bpermute_b32 v109, v90, v106
	s_wait_dscnt 0x0
	v_fma_f32 v12, -v108, v109, v12
	ds_bpermute_b32 v109, v91, v106
	s_wait_dscnt 0x0
	v_fma_f32 v13, -v108, v109, v13
	ds_bpermute_b32 v109, v92, v106
	s_wait_dscnt 0x0
	v_fma_f32 v14, -v108, v109, v14
	ds_bpermute_b32 v109, v93, v106
	s_wait_dscnt 0x0
	v_fma_f32 v15, -v108, v109, v15
	ds_bpermute_b32 v109, v94, v106
	s_wait_dscnt 0x0
	v_fma_f32 v16, -v108, v109, v16
	ds_bpermute_b32 v109, v95, v106
	s_wait_dscnt 0x0
	v_fma_f32 v17, -v108, v109, v17
	ds_bpermute_b32 v109, v96, v106
	s_wait_dscnt 0x0
	v_fma_f32 v18, -v108, v109, v18
	ds_bpermute_b32 v109, v97, v106
	s_wait_dscnt 0x0
	v_fma_f32 v19, -v108, v109, v19
	ds_bpermute_b32 v109, v98, v106
	s_wait_dscnt 0x0
	v_fma_f32 v20, -v108, v109, v20
	ds_bpermute_b32 v109, v99, v106
	s_wait_dscnt 0x0
	v_fma_f32 v21, -v108, v109, v21
	ds_bpermute_b32 v109, v100, v106
	s_wait_dscnt 0x0
	v_fma_f32 v22, -v108, v109, v22
	ds_bpermute_b32 v109, v101, v106
	ds_bpermute_b32 v106, v102, v106
	s_wait_dscnt 0x1
	v_fma_f32 v23, -v108, v109, v23
	s_wait_dscnt 0x0
	v_fma_f32 v24, -v108, v106, v24
	ds_bpermute_b32 v106, v104, v107
	s_wait_dscnt 0x0
	v_fma_f32 v25, -v108, v106, v25
	ds_bpermute_b32 v106, v88, v107
	s_wait_dscnt 0x0
	v_fma_f32 v26, -v108, v106, v26
	ds_bpermute_b32 v106, v89, v107
	s_wait_dscnt 0x0
	v_fma_f32 v27, -v108, v106, v27
	ds_bpermute_b32 v106, v90, v107
	s_wait_dscnt 0x0
	v_fma_f32 v28, -v108, v106, v28
	ds_bpermute_b32 v106, v91, v107
	s_wait_dscnt 0x0
	v_fma_f32 v29, -v108, v106, v29
	ds_bpermute_b32 v106, v92, v107
	s_wait_dscnt 0x0
	v_fma_f32 v30, -v108, v106, v30
	ds_bpermute_b32 v106, v93, v107
	s_wait_dscnt 0x0
	v_fma_f32 v31, -v108, v106, v31
	ds_bpermute_b32 v106, v94, v107
	s_wait_dscnt 0x0
	v_fma_f32 v32, -v108, v106, v32
	ds_bpermute_b32 v106, v95, v107
	s_wait_dscnt 0x0
	v_fma_f32 v33, -v108, v106, v33
	ds_bpermute_b32 v106, v96, v107
	s_wait_dscnt 0x0
	v_fma_f32 v34, -v108, v106, v34
	ds_bpermute_b32 v106, v97, v107
	s_wait_dscnt 0x0
	v_fma_f32 v35, -v108, v106, v35
	ds_bpermute_b32 v106, v98, v107
	s_wait_dscnt 0x0
	v_fma_f32 v36, -v108, v106, v36
	ds_bpermute_b32 v106, v99, v107
	s_wait_dscnt 0x0
	v_fma_f32 v37, -v108, v106, v37
	ds_bpermute_b32 v106, v100, v107
	s_wait_dscnt 0x0
	v_fma_f32 v38, -v108, v106, v38
	ds_bpermute_b32 v106, v101, v107
	s_wait_dscnt 0x0
	v_fma_f32 v39, -v108, v106, v39
	ds_bpermute_b32 v106, v102, v107
	s_wait_dscnt 0x0
	v_fma_f32 v40, -v108, v106, v40
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e32 v106, 0, v19, vcc_lo
	ds_load_2addr_b32 v[120:121], v122 offset0:128 offset1:160
	v_cndmask_b32_e64 v106, v106, v35, s0
	ds_bpermute_b32 v107, v87, v106
	s_wait_dscnt 0x0
	v_add_f32_e32 v123, v106, v107
	ds_bpermute_b32 v106, v104, v120
	s_wait_dscnt 0x0
	v_fma_f32 v9, -v123, v106, v9
	ds_bpermute_b32 v106, v88, v120
	s_wait_dscnt 0x0
	v_fma_f32 v10, -v123, v106, v10
	ds_bpermute_b32 v106, v89, v120
	s_wait_dscnt 0x0
	v_fma_f32 v11, -v123, v106, v11
	ds_bpermute_b32 v106, v90, v120
	s_wait_dscnt 0x0
	v_fma_f32 v12, -v123, v106, v12
	ds_bpermute_b32 v106, v91, v120
	s_wait_dscnt 0x0
	v_fma_f32 v13, -v123, v106, v13
	ds_bpermute_b32 v106, v92, v120
	s_wait_dscnt 0x0
	v_fma_f32 v14, -v123, v106, v14
	ds_bpermute_b32 v106, v93, v120
	s_wait_dscnt 0x0
	v_fma_f32 v15, -v123, v106, v15
	ds_bpermute_b32 v106, v94, v120
	s_wait_dscnt 0x0
	v_fma_f32 v16, -v123, v106, v16
	ds_bpermute_b32 v106, v95, v120
	s_wait_dscnt 0x0
	v_fma_f32 v17, -v123, v106, v17
	ds_bpermute_b32 v106, v96, v120
	s_wait_dscnt 0x0
	v_fma_f32 v18, -v123, v106, v18
	ds_bpermute_b32 v106, v97, v120
	s_wait_dscnt 0x0
	v_fma_f32 v19, -v123, v106, v19
	ds_bpermute_b32 v106, v98, v120
	s_wait_dscnt 0x0
	v_fma_f32 v20, -v123, v106, v20
	ds_bpermute_b32 v106, v99, v120
	s_wait_dscnt 0x0
	v_fma_f32 v21, -v123, v106, v21
	ds_bpermute_b32 v106, v100, v120
	s_wait_dscnt 0x0
	v_fma_f32 v22, -v123, v106, v22
	ds_bpermute_b32 v106, v101, v120
	s_wait_dscnt 0x0
	v_fma_f32 v23, -v123, v106, v23
	ds_bpermute_b32 v106, v102, v120
	s_wait_dscnt 0x0
	v_fma_f32 v24, -v123, v106, v24
	ds_bpermute_b32 v106, v104, v121
	s_wait_dscnt 0x0
	v_fma_f32 v106, -v123, v106, v25
	ds_bpermute_b32 v25, v88, v121
	s_wait_dscnt 0x0
	v_fma_f32 v107, -v123, v25, v26
	ds_bpermute_b32 v25, v89, v121
	s_wait_dscnt 0x0
	v_fma_f32 v108, -v123, v25, v27
	ds_bpermute_b32 v25, v90, v121
	s_wait_dscnt 0x0
	v_fma_f32 v109, -v123, v25, v28
	ds_bpermute_b32 v25, v91, v121
	s_wait_dscnt 0x0
	v_fma_f32 v110, -v123, v25, v29
	ds_bpermute_b32 v25, v92, v121
	s_wait_dscnt 0x0
	v_fma_f32 v111, -v123, v25, v30
	ds_bpermute_b32 v25, v93, v121
	s_wait_dscnt 0x0
	v_fma_f32 v112, -v123, v25, v31
	ds_bpermute_b32 v25, v94, v121
	s_wait_dscnt 0x0
	v_fma_f32 v113, -v123, v25, v32
	ds_bpermute_b32 v25, v95, v121
	s_wait_dscnt 0x0
	v_fma_f32 v114, -v123, v25, v33
	ds_bpermute_b32 v25, v96, v121
	s_wait_dscnt 0x0
	v_fma_f32 v115, -v123, v25, v34
	ds_bpermute_b32 v25, v97, v121
	s_wait_dscnt 0x0
	v_fma_f32 v116, -v123, v25, v35
	ds_bpermute_b32 v25, v98, v121
	s_wait_dscnt 0x0
	v_fma_f32 v117, -v123, v25, v36
	ds_bpermute_b32 v25, v99, v121
	s_wait_dscnt 0x0
	v_fma_f32 v118, -v123, v25, v37
	ds_bpermute_b32 v25, v100, v121
	s_wait_dscnt 0x0
	v_fma_f32 v119, -v123, v25, v38
	ds_bpermute_b32 v25, v101, v121
	s_wait_dscnt 0x0
	v_fma_f32 v120, -v123, v25, v39
	ds_bpermute_b32 v25, v102, v121
	s_wait_dscnt 0x0
	v_fma_f32 v121, -v123, v25, v40
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e32 v25, 0, v20, vcc_lo
	ds_load_2addr_b32 v[122:123], v122 offset0:192 offset1:224
	v_cndmask_b32_e64 v25, v25, v117, s0
	ds_bpermute_b32 v26, v87, v25
	s_wait_dscnt 0x0
	v_add_f32_e32 v124, v25, v26
	ds_bpermute_b32 v25, v104, v122
	s_wait_dscnt 0x0
	v_fma_f32 v33, -v124, v25, v9
	ds_bpermute_b32 v9, v88, v122
	s_wait_dscnt 0x0
	v_fma_f32 v34, -v124, v9, v10
	ds_bpermute_b32 v9, v89, v122
	ds_bpermute_b32 v10, v96, v123
	s_wait_dscnt 0x1
	v_fma_f32 v35, -v124, v9, v11
	ds_bpermute_b32 v9, v90, v122
	ds_bpermute_b32 v11, v97, v123
	s_wait_dscnt 0x2
	v_fma_f32 v10, -v124, v10, v115
	s_wait_dscnt 0x1
	v_fma_f32 v36, -v124, v9, v12
	ds_bpermute_b32 v9, v91, v122
	ds_bpermute_b32 v12, v98, v123
	s_wait_dscnt 0x2
	v_fma_f32 v11, -v124, v11, v116
	s_wait_dscnt 0x1
	v_fma_f32 v37, -v124, v9, v13
	ds_bpermute_b32 v9, v92, v122
	ds_bpermute_b32 v13, v99, v123
	s_wait_dscnt 0x2
	v_fma_f32 v12, -v124, v12, v117
	s_wait_dscnt 0x1
	v_fma_f32 v38, -v124, v9, v14
	ds_bpermute_b32 v9, v93, v122
	ds_bpermute_b32 v14, v100, v123
	s_wait_dscnt 0x2
	v_fma_f32 v13, -v124, v13, v118
	s_wait_dscnt 0x1
	v_fma_f32 v39, -v124, v9, v15
	ds_bpermute_b32 v9, v94, v122
	ds_bpermute_b32 v15, v101, v123
	s_wait_dscnt 0x2
	v_fma_f32 v14, -v124, v14, v119
	s_wait_dscnt 0x1
	v_fma_f32 v40, -v124, v9, v16
	ds_bpermute_b32 v9, v95, v122
	ds_bpermute_b32 v16, v102, v123
	s_wait_dscnt 0x2
	v_fma_f32 v15, -v124, v15, v120
	s_wait_dscnt 0x1
	v_fma_f32 v25, -v124, v9, v17
	ds_bpermute_b32 v9, v96, v122
	s_wait_dscnt 0x1
	v_fma_f32 v16, -v124, v16, v121
	s_wait_dscnt 0x0
	v_fma_f32 v26, -v124, v9, v18
	ds_bpermute_b32 v9, v97, v122
	s_wait_dscnt 0x0
	v_fma_f32 v27, -v124, v9, v19
	ds_bpermute_b32 v9, v98, v122
	s_wait_dscnt 0x0
	v_fma_f32 v28, -v124, v9, v20
	ds_bpermute_b32 v9, v99, v122
	s_wait_dscnt 0x0
	v_fma_f32 v29, -v124, v9, v21
	ds_bpermute_b32 v9, v100, v122
	s_wait_dscnt 0x0
	v_fma_f32 v30, -v124, v9, v22
	ds_bpermute_b32 v9, v101, v122
	s_wait_dscnt 0x0
	v_fma_f32 v31, -v124, v9, v23
	ds_bpermute_b32 v9, v102, v122
	s_wait_dscnt 0x0
	v_fma_f32 v32, -v124, v9, v24
	ds_bpermute_b32 v9, v104, v123
	s_wait_dscnt 0x0
	v_fma_f32 v17, -v124, v9, v106
	ds_bpermute_b32 v9, v88, v123
	s_wait_dscnt 0x0
	v_fma_f32 v18, -v124, v9, v107
	ds_bpermute_b32 v9, v89, v123
	s_wait_dscnt 0x0
	v_fma_f32 v19, -v124, v9, v108
	ds_bpermute_b32 v9, v90, v123
	s_wait_dscnt 0x0
	v_fma_f32 v20, -v124, v9, v109
	ds_bpermute_b32 v9, v91, v123
	s_wait_dscnt 0x0
	v_fma_f32 v21, -v124, v9, v110
	ds_bpermute_b32 v9, v92, v123
	s_wait_dscnt 0x0
	v_fma_f32 v22, -v124, v9, v111
	ds_bpermute_b32 v9, v93, v123
	s_wait_dscnt 0x0
	v_fma_f32 v23, -v124, v9, v112
	ds_bpermute_b32 v9, v94, v123
	s_wait_dscnt 0x0
	v_fma_f32 v24, -v124, v9, v113
	ds_bpermute_b32 v9, v95, v123
	s_wait_dscnt 0x0
	v_fma_f32 v9, -v124, v9, v114
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e32 v106, 0, v29, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v106, v106, v13, s0
	ds_bpermute_b32 v107, v87, v106
	s_wait_dscnt 0x0
	v_add_f32_e32 v106, v106, v107
	ds_load_2addr_b32 v[107:108], v105 offset1:32
	s_wait_dscnt 0x0
	ds_bpermute_b32 v109, v104, v107
	s_wait_dscnt 0x0
	v_fma_f32 v33, -v106, v109, v33
	ds_bpermute_b32 v109, v88, v107
	s_wait_dscnt 0x0
	v_fma_f32 v34, -v106, v109, v34
	ds_bpermute_b32 v109, v89, v107
	s_wait_dscnt 0x0
	v_fma_f32 v35, -v106, v109, v35
	ds_bpermute_b32 v109, v90, v107
	s_wait_dscnt 0x0
	v_fma_f32 v36, -v106, v109, v36
	ds_bpermute_b32 v109, v91, v107
	s_wait_dscnt 0x0
	v_fma_f32 v37, -v106, v109, v37
	ds_bpermute_b32 v109, v92, v107
	s_wait_dscnt 0x0
	v_fma_f32 v38, -v106, v109, v38
	ds_bpermute_b32 v109, v93, v107
	s_wait_dscnt 0x0
	v_fma_f32 v39, -v106, v109, v39
	ds_bpermute_b32 v109, v94, v107
	s_wait_dscnt 0x0
	v_fma_f32 v40, -v106, v109, v40
	ds_bpermute_b32 v109, v95, v107
	s_wait_dscnt 0x0
	v_fma_f32 v25, -v106, v109, v25
	ds_bpermute_b32 v109, v96, v107
	s_wait_dscnt 0x0
	v_fma_f32 v26, -v106, v109, v26
	ds_bpermute_b32 v109, v97, v107
	s_wait_dscnt 0x0
	v_fma_f32 v27, -v106, v109, v27
	ds_bpermute_b32 v109, v98, v107
	s_wait_dscnt 0x0
	v_fma_f32 v28, -v106, v109, v28
	ds_bpermute_b32 v109, v99, v107
	s_wait_dscnt 0x0
	v_fma_f32 v29, -v106, v109, v29
	ds_bpermute_b32 v109, v100, v107
	s_wait_dscnt 0x0
	v_fma_f32 v30, -v106, v109, v30
	ds_bpermute_b32 v109, v101, v107
	ds_bpermute_b32 v107, v102, v107
	s_wait_dscnt 0x1
	v_fma_f32 v31, -v106, v109, v31
	s_wait_dscnt 0x0
	v_fma_f32 v32, -v106, v107, v32
	ds_bpermute_b32 v107, v104, v108
	s_wait_dscnt 0x0
	v_fma_f32 v17, -v106, v107, v17
	ds_bpermute_b32 v107, v88, v108
	s_wait_dscnt 0x0
	v_fma_f32 v18, -v106, v107, v18
	ds_bpermute_b32 v107, v89, v108
	s_wait_dscnt 0x0
	v_fma_f32 v19, -v106, v107, v19
	ds_bpermute_b32 v107, v90, v108
	s_wait_dscnt 0x0
	v_fma_f32 v20, -v106, v107, v20
	ds_bpermute_b32 v107, v91, v108
	s_wait_dscnt 0x0
	v_fma_f32 v21, -v106, v107, v21
	ds_bpermute_b32 v107, v92, v108
	s_wait_dscnt 0x0
	v_fma_f32 v22, -v106, v107, v22
	ds_bpermute_b32 v107, v93, v108
	s_wait_dscnt 0x0
	v_fma_f32 v23, -v106, v107, v23
	ds_bpermute_b32 v107, v94, v108
	s_wait_dscnt 0x0
	v_fma_f32 v24, -v106, v107, v24
	ds_bpermute_b32 v107, v95, v108
	s_wait_dscnt 0x0
	v_fma_f32 v9, -v106, v107, v9
	ds_bpermute_b32 v107, v96, v108
	s_wait_dscnt 0x0
	v_fma_f32 v10, -v106, v107, v10
	ds_bpermute_b32 v107, v97, v108
	s_wait_dscnt 0x0
	v_fma_f32 v11, -v106, v107, v11
	ds_bpermute_b32 v107, v98, v108
	s_wait_dscnt 0x0
	v_fma_f32 v12, -v106, v107, v12
	ds_bpermute_b32 v107, v99, v108
	s_wait_dscnt 0x0
	v_fma_f32 v13, -v106, v107, v13
	ds_bpermute_b32 v107, v100, v108
	s_wait_dscnt 0x0
	v_fma_f32 v14, -v106, v107, v14
	ds_bpermute_b32 v107, v101, v108
	s_wait_dscnt 0x0
	v_fma_f32 v15, -v106, v107, v15
	ds_bpermute_b32 v107, v102, v108
	s_wait_dscnt 0x0
	v_fma_f32 v16, -v106, v107, v16
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e32 v106, 0, v30, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v106, v106, v14, s0
	ds_bpermute_b32 v107, v87, v106
	s_wait_dscnt 0x0
	v_add_f32_e32 v108, v106, v107
	ds_load_2addr_b32 v[106:107], v105 offset0:64 offset1:96
	s_wait_dscnt 0x0
	ds_bpermute_b32 v109, v104, v106
	s_wait_dscnt 0x0
	v_fma_f32 v33, -v108, v109, v33
	ds_bpermute_b32 v109, v88, v106
	s_wait_dscnt 0x0
	v_fma_f32 v34, -v108, v109, v34
	ds_bpermute_b32 v109, v89, v106
	s_wait_dscnt 0x0
	v_fma_f32 v35, -v108, v109, v35
	ds_bpermute_b32 v109, v90, v106
	s_wait_dscnt 0x0
	v_fma_f32 v36, -v108, v109, v36
	ds_bpermute_b32 v109, v91, v106
	s_wait_dscnt 0x0
	v_fma_f32 v37, -v108, v109, v37
	ds_bpermute_b32 v109, v92, v106
	s_wait_dscnt 0x0
	v_fma_f32 v38, -v108, v109, v38
	ds_bpermute_b32 v109, v93, v106
	s_wait_dscnt 0x0
	v_fma_f32 v39, -v108, v109, v39
	ds_bpermute_b32 v109, v94, v106
	s_wait_dscnt 0x0
	v_fma_f32 v40, -v108, v109, v40
	ds_bpermute_b32 v109, v95, v106
	s_wait_dscnt 0x0
	v_fma_f32 v25, -v108, v109, v25
	ds_bpermute_b32 v109, v96, v106
	s_wait_dscnt 0x0
	v_fma_f32 v26, -v108, v109, v26
	ds_bpermute_b32 v109, v97, v106
	s_wait_dscnt 0x0
	v_fma_f32 v27, -v108, v109, v27
	ds_bpermute_b32 v109, v98, v106
	s_wait_dscnt 0x0
	v_fma_f32 v28, -v108, v109, v28
	ds_bpermute_b32 v109, v99, v106
	s_wait_dscnt 0x0
	v_fma_f32 v29, -v108, v109, v29
	ds_bpermute_b32 v109, v100, v106
	s_wait_dscnt 0x0
	v_fma_f32 v30, -v108, v109, v30
	ds_bpermute_b32 v109, v101, v106
	ds_bpermute_b32 v106, v102, v106
	s_wait_dscnt 0x1
	v_fma_f32 v31, -v108, v109, v31
	s_wait_dscnt 0x0
	v_fma_f32 v32, -v108, v106, v32
	ds_bpermute_b32 v106, v104, v107
	s_wait_dscnt 0x0
	v_fma_f32 v17, -v108, v106, v17
	ds_bpermute_b32 v106, v88, v107
	s_wait_dscnt 0x0
	v_fma_f32 v18, -v108, v106, v18
	ds_bpermute_b32 v106, v89, v107
	s_wait_dscnt 0x0
	v_fma_f32 v19, -v108, v106, v19
	ds_bpermute_b32 v106, v90, v107
	s_wait_dscnt 0x0
	v_fma_f32 v20, -v108, v106, v20
	ds_bpermute_b32 v106, v91, v107
	s_wait_dscnt 0x0
	v_fma_f32 v21, -v108, v106, v21
	ds_bpermute_b32 v106, v92, v107
	s_wait_dscnt 0x0
	v_fma_f32 v22, -v108, v106, v22
	ds_bpermute_b32 v106, v93, v107
	s_wait_dscnt 0x0
	v_fma_f32 v23, -v108, v106, v23
	ds_bpermute_b32 v106, v94, v107
	s_wait_dscnt 0x0
	v_fma_f32 v24, -v108, v106, v24
	ds_bpermute_b32 v106, v95, v107
	s_wait_dscnt 0x0
	v_fma_f32 v9, -v108, v106, v9
	ds_bpermute_b32 v106, v96, v107
	s_wait_dscnt 0x0
	v_fma_f32 v10, -v108, v106, v10
	ds_bpermute_b32 v106, v97, v107
	s_wait_dscnt 0x0
	v_fma_f32 v11, -v108, v106, v11
	ds_bpermute_b32 v106, v98, v107
	s_wait_dscnt 0x0
	v_fma_f32 v12, -v108, v106, v12
	ds_bpermute_b32 v106, v99, v107
	s_wait_dscnt 0x0
	v_fma_f32 v13, -v108, v106, v13
	ds_bpermute_b32 v106, v100, v107
	s_wait_dscnt 0x0
	v_fma_f32 v14, -v108, v106, v14
	ds_bpermute_b32 v106, v101, v107
	s_wait_dscnt 0x0
	v_fma_f32 v15, -v108, v106, v15
	ds_bpermute_b32 v106, v102, v107
	s_wait_dscnt 0x0
	v_fma_f32 v16, -v108, v106, v16
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e32 v106, 0, v31, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v106, v106, v15, s0
	ds_bpermute_b32 v107, v87, v106
	s_wait_dscnt 0x0
	v_add_f32_e32 v108, v106, v107
	ds_load_2addr_b32 v[106:107], v105 offset0:128 offset1:160
	s_wait_dscnt 0x0
	ds_bpermute_b32 v109, v104, v106
	s_wait_dscnt 0x0
	v_fma_f32 v33, -v108, v109, v33
	ds_bpermute_b32 v109, v88, v106
	s_wait_dscnt 0x0
	v_fma_f32 v34, -v108, v109, v34
	ds_bpermute_b32 v109, v89, v106
	s_wait_dscnt 0x0
	v_fma_f32 v35, -v108, v109, v35
	ds_bpermute_b32 v109, v90, v106
	s_wait_dscnt 0x0
	v_fma_f32 v36, -v108, v109, v36
	ds_bpermute_b32 v109, v91, v106
	s_wait_dscnt 0x0
	v_fma_f32 v37, -v108, v109, v37
	ds_bpermute_b32 v109, v92, v106
	s_wait_dscnt 0x0
	v_fma_f32 v38, -v108, v109, v38
	ds_bpermute_b32 v109, v93, v106
	s_wait_dscnt 0x0
	v_fma_f32 v39, -v108, v109, v39
	ds_bpermute_b32 v109, v94, v106
	s_wait_dscnt 0x0
	v_fma_f32 v40, -v108, v109, v40
	ds_bpermute_b32 v109, v95, v106
	s_wait_dscnt 0x0
	v_fma_f32 v25, -v108, v109, v25
	ds_bpermute_b32 v109, v96, v106
	s_wait_dscnt 0x0
	v_fma_f32 v26, -v108, v109, v26
	ds_bpermute_b32 v109, v97, v106
	s_wait_dscnt 0x0
	v_fma_f32 v27, -v108, v109, v27
	ds_bpermute_b32 v109, v98, v106
	s_wait_dscnt 0x0
	v_fma_f32 v28, -v108, v109, v28
	ds_bpermute_b32 v109, v99, v106
	s_wait_dscnt 0x0
	v_fma_f32 v29, -v108, v109, v29
	ds_bpermute_b32 v109, v100, v106
	s_wait_dscnt 0x0
	v_fma_f32 v30, -v108, v109, v30
	ds_bpermute_b32 v109, v101, v106
	ds_bpermute_b32 v106, v102, v106
	s_wait_dscnt 0x1
	v_fma_f32 v31, -v108, v109, v31
	s_wait_dscnt 0x0
	v_fma_f32 v32, -v108, v106, v32
	ds_bpermute_b32 v106, v104, v107
	s_wait_dscnt 0x0
	v_fma_f32 v17, -v108, v106, v17
	ds_bpermute_b32 v106, v88, v107
	s_wait_dscnt 0x0
	v_fma_f32 v18, -v108, v106, v18
	ds_bpermute_b32 v106, v89, v107
	s_wait_dscnt 0x0
	v_fma_f32 v19, -v108, v106, v19
	ds_bpermute_b32 v106, v90, v107
	s_wait_dscnt 0x0
	v_fma_f32 v20, -v108, v106, v20
	ds_bpermute_b32 v106, v91, v107
	s_wait_dscnt 0x0
	v_fma_f32 v21, -v108, v106, v21
	ds_bpermute_b32 v106, v92, v107
	s_wait_dscnt 0x0
	v_fma_f32 v22, -v108, v106, v22
	ds_bpermute_b32 v106, v93, v107
	s_wait_dscnt 0x0
	v_fma_f32 v23, -v108, v106, v23
	ds_bpermute_b32 v106, v94, v107
	s_wait_dscnt 0x0
	v_fma_f32 v24, -v108, v106, v24
	ds_bpermute_b32 v106, v95, v107
	s_wait_dscnt 0x0
	v_fma_f32 v9, -v108, v106, v9
	ds_bpermute_b32 v106, v96, v107
	s_wait_dscnt 0x0
	v_fma_f32 v10, -v108, v106, v10
	ds_bpermute_b32 v106, v97, v107
	s_wait_dscnt 0x0
	v_fma_f32 v11, -v108, v106, v11
	ds_bpermute_b32 v106, v98, v107
	s_wait_dscnt 0x0
	v_fma_f32 v12, -v108, v106, v12
	ds_bpermute_b32 v106, v99, v107
	s_wait_dscnt 0x0
	v_fma_f32 v13, -v108, v106, v13
	ds_bpermute_b32 v106, v100, v107
	s_wait_dscnt 0x0
	v_fma_f32 v14, -v108, v106, v14
	ds_bpermute_b32 v106, v101, v107
	s_wait_dscnt 0x0
	v_fma_f32 v15, -v108, v106, v15
	ds_bpermute_b32 v106, v102, v107
	s_wait_dscnt 0x0
	v_fma_f32 v16, -v108, v106, v16
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e32 v106, 0, v32, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v106, v106, v16, s0
	ds_bpermute_b32 v107, v87, v106
	s_wait_dscnt 0x0
	v_add_f32_e32 v107, v106, v107
	ds_load_2addr_b32 v[105:106], v105 offset0:192 offset1:224
	s_wait_dscnt 0x0
	ds_bpermute_b32 v108, v104, v105
	ds_bpermute_b32 v104, v104, v106
	s_wait_dscnt 0x1
	v_fma_f32 v33, -v107, v108, v33
	s_wait_dscnt 0x0
	v_fma_f32 v17, -v107, v104, v17
	ds_bpermute_b32 v104, v88, v106
	ds_bpermute_b32 v108, v88, v105
	s_wait_dscnt 0x1
	v_fma_f32 v18, -v107, v104, v18
	ds_bpermute_b32 v104, v89, v106
	s_wait_dscnt 0x1
	v_fma_f32 v34, -v107, v108, v34
	ds_bpermute_b32 v108, v89, v105
	s_wait_dscnt 0x1
	v_fma_f32 v19, -v107, v104, v19
	ds_bpermute_b32 v104, v90, v106
	s_wait_dscnt 0x1
	v_fma_f32 v35, -v107, v108, v35
	ds_bpermute_b32 v108, v90, v105
	s_wait_dscnt 0x1
	v_fma_f32 v20, -v107, v104, v20
	ds_bpermute_b32 v104, v91, v106
	s_wait_dscnt 0x1
	v_fma_f32 v36, -v107, v108, v36
	ds_bpermute_b32 v108, v91, v105
	s_wait_dscnt 0x1
	v_fma_f32 v21, -v107, v104, v21
	ds_bpermute_b32 v104, v92, v106
	s_wait_dscnt 0x1
	v_fma_f32 v37, -v107, v108, v37
	ds_bpermute_b32 v108, v92, v105
	s_wait_dscnt 0x1
	v_fma_f32 v22, -v107, v104, v22
	ds_bpermute_b32 v104, v93, v106
	s_wait_dscnt 0x1
	v_fma_f32 v38, -v107, v108, v38
	ds_bpermute_b32 v108, v93, v105
	s_wait_dscnt 0x1
	v_fma_f32 v23, -v107, v104, v23
	ds_bpermute_b32 v104, v94, v106
	s_wait_dscnt 0x1
	v_fma_f32 v39, -v107, v108, v39
	ds_bpermute_b32 v108, v94, v105
	s_wait_dscnt 0x1
	v_fma_f32 v24, -v107, v104, v24
	ds_bpermute_b32 v104, v95, v106
	s_wait_dscnt 0x1
	v_fma_f32 v40, -v107, v108, v40
	ds_bpermute_b32 v108, v95, v105
	s_wait_dscnt 0x1
	v_fma_f32 v9, -v107, v104, v9
	ds_bpermute_b32 v104, v96, v106
	s_wait_dscnt 0x1
	v_fma_f32 v25, -v107, v108, v25
	ds_bpermute_b32 v108, v96, v105
	s_wait_dscnt 0x1
	v_fma_f32 v10, -v107, v104, v10
	ds_bpermute_b32 v104, v97, v106
	s_wait_dscnt 0x1
	v_fma_f32 v26, -v107, v108, v26
	ds_bpermute_b32 v108, v97, v105
	s_wait_dscnt 0x1
	v_fma_f32 v11, -v107, v104, v11
	ds_bpermute_b32 v104, v98, v106
	s_wait_dscnt 0x1
	v_fma_f32 v27, -v107, v108, v27
	ds_bpermute_b32 v108, v98, v105
	s_wait_dscnt 0x1
	v_fma_f32 v12, -v107, v104, v12
	ds_bpermute_b32 v104, v99, v106
	s_wait_dscnt 0x1
	v_fma_f32 v28, -v107, v108, v28
	ds_bpermute_b32 v108, v99, v105
	s_wait_dscnt 0x1
	v_fma_f32 v13, -v107, v104, v13
	ds_bpermute_b32 v104, v100, v106
	s_wait_dscnt 0x1
	v_fma_f32 v29, -v107, v108, v29
	ds_bpermute_b32 v108, v100, v105
	s_wait_dscnt 0x1
	v_fma_f32 v14, -v107, v104, v14
	ds_bpermute_b32 v104, v101, v106
	s_wait_dscnt 0x1
	v_fma_f32 v30, -v107, v108, v30
	ds_bpermute_b32 v108, v101, v105
	ds_bpermute_b32 v105, v102, v105
	s_wait_dscnt 0x2
	v_fma_f32 v15, -v107, v104, v15
	ds_bpermute_b32 v104, v102, v106
	s_wait_dscnt 0x2
	v_fma_f32 v31, -v107, v108, v31
	s_wait_dscnt 0x1
	v_fma_f32 v32, -v107, v105, v32
	s_wait_dscnt 0x0
	v_fma_f32 v16, -v107, v104, v16
	;;#ASMSTART
	;;#ASMEND
	s_cbranch_scc0 .LBB5_27
; %bb.28:                               ;   in Loop: Header=BB5_26 Depth=1
	v_wmma_f32_16x16x16_f16 v[104:111], v[77:80], v[45:48], v[1:8]
	v_cvt_f16_f32_e32 v40.h, v40
	v_cvt_f16_f32_e32 v40.l, v39
	v_cvt_f16_f32_e32 v39.h, v38
	s_delay_alu instid0(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[104:111], v[77:80], v[53:56], v[104:111]
	v_cvt_f16_f32_e32 v39.l, v37
	v_cvt_f16_f32_e32 v38.h, v36
	v_cvt_f16_f32_e32 v38.l, v35
	v_cvt_f16_f32_e32 v37.h, v34
	v_wmma_f32_16x16x16_f16 v[104:111], v[77:80], v[69:72], v[104:111]
	v_cvt_f16_f32_e32 v37.l, v33
	v_dual_add_f32 v119, v8, v8 :: v_dual_add_f32 v118, v7, v7
	v_dual_add_f32 v117, v6, v6 :: v_dual_add_f32 v116, v5, v5
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_wmma_f32_16x16x16_f16 v[104:111], v[77:80], v[57:60], v[104:111]
	v_dual_add_f32 v115, v4, v4 :: v_dual_add_f32 v114, v3, v3
	v_dual_add_f32 v113, v2, v2 :: v_dual_add_f32 v112, v1, v1
	v_wmma_f32_16x16x16_f16 v[104:111], v[77:80], v[49:52], v[104:111]
	v_cvt_f16_f32_e32 v32.h, v32
	v_cvt_f16_f32_e32 v32.l, v31
	v_cvt_f16_f32_e32 v31.h, v30
	v_cvt_f16_f32_e32 v31.l, v29
	v_wmma_f32_16x16x16_f16 v[104:111], v[77:80], v[61:64], v[104:111]
	v_cvt_f16_f32_e32 v30.h, v28
	v_cvt_f16_f32_e32 v30.l, v27
	v_cvt_f16_f32_e32 v29.h, v26
	v_cvt_f16_f32_e32 v29.l, v25
	v_wmma_f32_16x16x16_f16 v[104:111], v[77:80], v[65:68], v[104:111]
	v_wmma_f32_16x16x16_f16 v[112:119], v[77:80], v[45:48], v[112:119]
	v_cvt_f16_f32_e32 v24.h, v24
	v_cvt_f16_f32_e32 v24.l, v23
	v_cvt_f16_f32_e32 v23.h, v22
	v_wmma_f32_16x16x16_f16 v[104:111], v[77:80], v[41:44], v[104:111]
	v_cvt_f16_f32_e32 v23.l, v21
	v_cvt_f16_f32_e32 v22.h, v20
	v_cvt_f16_f32_e32 v22.l, v19
	v_cvt_f16_f32_e32 v21.h, v18
	v_wmma_f32_16x16x16_f16 v[104:111], v[73:76], v[37:40], v[104:111]
	v_cvt_f16_f32_e32 v21.l, v17
	v_wmma_f32_16x16x16_f16 v[112:119], v[77:80], v[53:56], v[112:119]
	v_cvt_f16_f32_e32 v16.h, v16
	v_cvt_f16_f32_e32 v16.l, v15
	v_wmma_f32_16x16x16_f16 v[104:111], v[73:76], v[29:32], v[104:111]
	v_cvt_f16_f32_e32 v15.h, v14
	v_cvt_f16_f32_e32 v15.l, v13
	v_cvt_f16_f32_e32 v14.h, v12
	v_cvt_f16_f32_e32 v14.l, v11
	v_wmma_f32_16x16x16_f16 v[104:111], v[73:76], v[21:24], v[104:111]
	v_cvt_f16_f32_e32 v13.h, v10
	v_wmma_f32_16x16x16_f16 v[112:119], v[77:80], v[69:72], v[112:119]
	v_cvt_f16_f32_e32 v13.l, v9
	v_dual_mul_f32 v127, 0x40400000, v8 :: v_dual_mul_f32 v126, 0x40400000, v7
	v_dual_mul_f32 v125, 0x40400000, v6 :: v_dual_mul_f32 v122, 0x40400000, v3
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[112:119], v[77:80], v[57:60], v[112:119]
	v_wmma_f32_16x16x16_f16 v[104:111], v[73:76], v[13:16], v[104:111]
	v_dual_mul_f32 v124, 0x40400000, v5 :: v_dual_mul_f32 v123, 0x40400000, v4
	v_mul_f32_e32 v120, 0x40400000, v1
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[112:119], v[77:80], v[49:52], v[112:119]
	v_add_f32_e32 v9, 0, v104
	v_dual_mul_f32 v8, 4.0, v8 :: v_dual_mul_f32 v121, 0x40400000, v2
	v_mul_f32_e32 v4, 4.0, v4
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[112:119], v[77:80], v[61:64], v[112:119]
	v_add_f32_e32 v9, v9, v105
	v_dual_mul_f32 v7, 4.0, v7 :: v_dual_mul_f32 v6, 4.0, v6
	v_wmma_f32_16x16x16_f16 v[120:127], v[77:80], v[45:48], v[120:127]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[112:119], v[77:80], v[65:68], v[112:119]
	v_add_f32_e32 v9, v9, v106
	v_dual_mul_f32 v2, 4.0, v2 :: v_dual_mul_f32 v5, 4.0, v5
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[120:127], v[77:80], v[53:56], v[120:127]
	v_wmma_f32_16x16x16_f16 v[112:119], v[77:80], v[41:44], v[112:119]
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v9, v9, v107
	v_mul_f32_e32 v3, 4.0, v3
	v_mul_f32_e32 v1, 4.0, v1
	v_wmma_f32_16x16x16_f16 v[120:127], v[77:80], v[69:72], v[120:127]
	v_wmma_f32_16x16x16_f16 v[112:119], v[73:76], v[37:40], v[112:119]
	v_add_f32_e32 v9, v9, v108
	v_fma_mix_f32 v108, v55, s7, neg(0) op_sel_hi:[1,0,0]
	v_wmma_f32_16x16x16_f16 v[1:8], v[77:80], v[45:48], v[1:8]
	v_wmma_f32_16x16x16_f16 v[120:127], v[77:80], v[57:60], v[120:127]
	v_wmma_f32_16x16x16_f16 v[112:119], v[73:76], v[29:32], v[112:119]
	v_add_f32_e32 v9, v9, v109
	v_fma_mix_f32 v109, v55, s7, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_wmma_f32_16x16x16_f16 v[1:8], v[77:80], v[53:56], v[1:8]
	v_wmma_f32_16x16x16_f16 v[120:127], v[77:80], v[49:52], v[120:127]
	v_wmma_f32_16x16x16_f16 v[112:119], v[73:76], v[21:24], v[112:119]
	v_add_f32_e32 v9, v9, v110
	v_fma_mix_f32 v110, v56, s7, neg(0) op_sel_hi:[1,0,0]
	v_wmma_f32_16x16x16_f16 v[1:8], v[77:80], v[69:72], v[1:8]
	v_wmma_f32_16x16x16_f16 v[120:127], v[77:80], v[61:64], v[120:127]
	v_wmma_f32_16x16x16_f16 v[112:119], v[73:76], v[13:16], v[112:119]
	v_add_f32_e32 v9, v9, v111
	v_fma_mix_f32 v111, v56, s7, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_wmma_f32_16x16x16_f16 v[1:8], v[77:80], v[57:60], v[1:8]
	v_wmma_f32_16x16x16_f16 v[120:127], v[77:80], v[65:68], v[120:127]
	v_fma_mix_f32 v107, v54, s7, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_add_f32_e32 v9, v9, v112
	v_fma_mix_f32 v106, v54, s7, neg(0) op_sel_hi:[1,0,0]
	v_wmma_f32_16x16x16_f16 v[1:8], v[77:80], v[49:52], v[1:8]
	v_wmma_f32_16x16x16_f16 v[120:127], v[77:80], v[41:44], v[120:127]
	v_fma_mix_f32 v105, v53, s7, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_add_f32_e32 v9, v9, v113
	v_fma_mix_f32 v104, v53, s7, neg(0) op_sel_hi:[1,0,0]
	v_wmma_f32_16x16x16_f16 v[1:8], v[77:80], v[61:64], v[1:8]
	v_wmma_f32_16x16x16_f16 v[120:127], v[73:76], v[37:40], v[120:127]
	v_fma_mix_f32 v113, v57, s7, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_add_f32_e32 v9, v9, v114
	v_wmma_f32_16x16x16_f16 v[104:111], v[73:76], v[37:40], v[104:111]
	v_wmma_f32_16x16x16_f16 v[1:8], v[77:80], v[65:68], v[1:8]
	v_wmma_f32_16x16x16_f16 v[120:127], v[73:76], v[29:32], v[120:127]
	v_fma_mix_f32 v114, v58, s7, neg(0) op_sel_hi:[1,0,0]
	v_add_f32_e32 v9, v9, v115
	v_wmma_f32_16x16x16_f16 v[104:111], v[73:76], v[29:32], v[104:111]
	v_wmma_f32_16x16x16_f16 v[1:8], v[77:80], v[41:44], v[1:8]
	v_wmma_f32_16x16x16_f16 v[120:127], v[73:76], v[21:24], v[120:127]
	v_fma_mix_f32 v115, v58, s7, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_add_f32_e32 v9, v9, v116
	v_wmma_f32_16x16x16_f16 v[104:111], v[73:76], v[21:24], v[104:111]
	v_wmma_f32_16x16x16_f16 v[1:8], v[73:76], v[37:40], v[1:8]
	v_wmma_f32_16x16x16_f16 v[120:127], v[73:76], v[13:16], v[120:127]
	v_fma_mix_f32 v116, v59, s7, neg(0) op_sel_hi:[1,0,0]
	v_add_f32_e32 v9, v9, v117
	v_wmma_f32_16x16x16_f16 v[104:111], v[73:76], v[13:16], v[104:111]
	v_wmma_f32_16x16x16_f16 v[1:8], v[73:76], v[29:32], v[1:8]
	v_fma_mix_f32 v117, v59, s7, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v112, v57, s7, neg(0) op_sel_hi:[1,0,0]
	v_add_f32_e32 v9, v9, v118
	v_fma_mix_f32 v118, v60, s7, neg(0) op_sel_hi:[1,0,0]
	v_wmma_f32_16x16x16_f16 v[1:8], v[73:76], v[21:24], v[1:8]
	v_cvt_f16_f32_e32 v53.h, v105
	v_cvt_f16_f32_e32 v53.l, v104
	v_add_f32_e32 v9, v9, v119
	v_fma_mix_f32 v119, v60, s7, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_wmma_f32_16x16x16_f16 v[1:8], v[73:76], v[13:16], v[1:8]
	v_cvt_f16_f32_e32 v54.h, v107
	v_cvt_f16_f32_e32 v54.l, v106
	v_add_f32_e32 v9, v9, v120
	v_cvt_f16_f32_e32 v55.h, v109
	v_cvt_f16_f32_e32 v55.l, v108
	v_cvt_f16_f32_e32 v56.h, v111
	v_cvt_f16_f32_e32 v56.l, v110
	v_add_f32_e32 v9, v9, v121
	v_fma_mix_f32 v111, v52, s7, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v110, v52, s7, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v109, v51, s7, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v108, v51, s7, neg(0) op_sel_hi:[1,0,0]
	v_add_f32_e32 v9, v9, v122
	v_fma_mix_f32 v107, v50, s7, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v106, v50, s7, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v105, v49, s7, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v104, v49, s7, neg(0) op_sel_hi:[1,0,0]
	v_add_f32_e32 v9, v9, v123
	v_wmma_f32_16x16x16_f16 v[112:119], v[73:76], v[37:40], v[112:119]
	v_fma_mix_f32 v135, v48, s7, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v134, v48, s7, neg(0) op_sel_hi:[1,0,0]
	v_wmma_f32_16x16x16_f16 v[104:111], v[73:76], v[37:40], v[104:111]
	v_add_f32_e32 v9, v9, v124
	v_wmma_f32_16x16x16_f16 v[112:119], v[73:76], v[29:32], v[112:119]
	v_fma_mix_f32 v133, v47, s7, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v132, v47, s7, neg(0) op_sel_hi:[1,0,0]
	v_wmma_f32_16x16x16_f16 v[104:111], v[73:76], v[29:32], v[104:111]
	v_add_f32_e32 v9, v9, v125
	v_wmma_f32_16x16x16_f16 v[112:119], v[73:76], v[21:24], v[112:119]
	v_fma_mix_f32 v131, v46, s7, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v130, v46, s7, neg(0) op_sel_hi:[1,0,0]
	v_wmma_f32_16x16x16_f16 v[104:111], v[73:76], v[21:24], v[104:111]
	v_add_f32_e32 v9, v9, v126
	v_wmma_f32_16x16x16_f16 v[112:119], v[73:76], v[13:16], v[112:119]
	v_fma_mix_f32 v129, v45, s7, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v128, v45, s7, neg(0) op_sel_hi:[1,0,0]
	v_wmma_f32_16x16x16_f16 v[104:111], v[73:76], v[13:16], v[104:111]
	v_add_f32_e32 v9, v9, v127
	v_fma_mix_f32 v143, v72, s7, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v142, v72, s7, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v141, v71, s7, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v140, v71, s7, neg(0) op_sel_hi:[1,0,0]
	v_add_f32_e32 v1, v9, v1
	v_fma_mix_f32 v139, v70, s7, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v138, v70, s7, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v137, v69, s7, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v136, v69, s7, neg(0) op_sel_hi:[1,0,0]
	v_add_f32_e32 v1, v1, v2
	v_cvt_f16_f32_e32 v58.h, v115
	v_cvt_f16_f32_e32 v58.l, v114
	v_fma_mix_f32 v127, v64, s7, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v126, v64, s7, neg(0) op_sel_hi:[1,0,0]
	v_add_f32_e32 v1, v1, v3
	v_fma_mix_f32 v125, v63, s7, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v124, v63, s7, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v123, v62, s7, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v122, v62, s7, neg(0) op_sel_hi:[1,0,0]
	v_add_f32_e32 v1, v1, v4
	v_fma_mix_f32 v121, v61, s7, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v120, v61, s7, neg(0) op_sel_hi:[1,0,0]
	v_cvt_f16_f32_e32 v59.h, v117
	v_cvt_f16_f32_e32 v59.l, v116
	v_add_f32_e32 v1, v1, v5
	v_cvt_f16_f32_e32 v60.h, v119
	v_cvt_f16_f32_e32 v60.l, v118
	v_cvt_f16_f32_e32 v57.h, v113
	v_cvt_f16_f32_e32 v57.l, v112
	v_cvt_f16_f32_e32 v49.h, v105
	v_cvt_f16_f32_e32 v49.l, v104
	v_cvt_f16_f32_e32 v50.h, v107
	v_cvt_f16_f32_e32 v50.l, v106
	v_cvt_f16_f32_e32 v51.h, v109
	v_fma_mix_f32 v119, v68, s7, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v118, v68, s7, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v117, v67, s7, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v116, v67, s7, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v115, v66, s7, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v114, v66, s7, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v113, v65, s7, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v112, v65, s7, neg(0) op_sel_hi:[1,0,0]
	v_cvt_f16_f32_e32 v51.l, v108
	v_cvt_f16_f32_e32 v52.h, v111
	v_cvt_f16_f32_e32 v52.l, v110
	v_fma_mix_f32 v111, v44, s7, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v110, v44, s7, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v109, v43, s7, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v108, v43, s7, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v107, v42, s7, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v106, v42, s7, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v105, v41, s7, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v104, v41, s7, neg(0) op_sel_hi:[1,0,0]
	v_add_f32_e32 v1, v1, v6
	v_wmma_f32_16x16x16_f16 v[128:135], v[73:76], v[37:40], v[128:135]
	v_wmma_f32_16x16x16_f16 v[136:143], v[73:76], v[37:40], v[136:143]
	v_wmma_f32_16x16x16_f16 v[120:127], v[73:76], v[37:40], v[120:127]
	v_wmma_f32_16x16x16_f16 v[112:119], v[73:76], v[37:40], v[112:119]
	v_wmma_f32_16x16x16_f16 v[104:111], v[73:76], v[37:40], v[104:111]
	v_add_f32_e32 v1, v1, v7
	v_wmma_f32_16x16x16_f16 v[128:135], v[73:76], v[29:32], v[128:135]
	v_wmma_f32_16x16x16_f16 v[136:143], v[73:76], v[29:32], v[136:143]
	v_wmma_f32_16x16x16_f16 v[120:127], v[73:76], v[29:32], v[120:127]
	v_wmma_f32_16x16x16_f16 v[112:119], v[73:76], v[29:32], v[112:119]
	v_wmma_f32_16x16x16_f16 v[104:111], v[73:76], v[29:32], v[104:111]
	v_add_f32_e32 v1, v1, v8
	v_wmma_f32_16x16x16_f16 v[128:135], v[73:76], v[21:24], v[128:135]
	v_wmma_f32_16x16x16_f16 v[136:143], v[73:76], v[21:24], v[136:143]
	v_wmma_f32_16x16x16_f16 v[120:127], v[73:76], v[21:24], v[120:127]
	v_wmma_f32_16x16x16_f16 v[112:119], v[73:76], v[21:24], v[112:119]
	v_wmma_f32_16x16x16_f16 v[104:111], v[73:76], v[21:24], v[104:111]
	v_cmp_lt_f32_e32 vcc_lo, 0x60ad78ec, v1
	v_wmma_f32_16x16x16_f16 v[128:135], v[73:76], v[13:16], v[128:135]
	v_wmma_f32_16x16x16_f16 v[136:143], v[73:76], v[13:16], v[136:143]
	v_wmma_f32_16x16x16_f16 v[120:127], v[73:76], v[13:16], v[120:127]
	v_wmma_f32_16x16x16_f16 v[112:119], v[73:76], v[13:16], v[112:119]
	v_wmma_f32_16x16x16_f16 v[104:111], v[73:76], v[13:16], v[104:111]
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v1, 0x3a83126f, v103, vcc_lo
	v_cvt_f16_f32_e64 v45.h, v129
	v_cvt_f16_f32_e64 v45.l, v128
	v_cvt_f16_f32_e64 v46.h, v131
	v_cvt_f16_f32_e64 v46.l, v130
	v_cvt_f16_f32_e64 v47.h, v133
	v_cvt_f16_f32_e64 v47.l, v132
	v_cvt_f16_f32_e64 v48.h, v135
	v_cvt_f16_f32_e64 v48.l, v134
	v_cvt_f16_f32_e64 v69.h, v137
	v_cvt_f16_f32_e64 v69.l, v136
	v_cvt_f16_f32_e64 v70.h, v139
	v_cvt_f16_f32_e64 v70.l, v138
	v_cvt_f16_f32_e64 v71.h, v141
	v_cvt_f16_f32_e64 v71.l, v140
	v_cvt_f16_f32_e64 v72.h, v143
	v_cvt_f16_f32_e64 v72.l, v142
	v_cvt_f16_f32_e32 v61.h, v121
	v_cvt_f16_f32_e32 v61.l, v120
	v_cvt_f16_f32_e32 v62.h, v123
	v_cvt_f16_f32_e32 v62.l, v122
	v_cvt_f16_f32_e32 v63.h, v125
	v_cvt_f16_f32_e32 v63.l, v124
	v_cvt_f16_f32_e32 v64.h, v127
	v_cvt_f16_f32_e32 v64.l, v126
	v_cvt_f16_f32_e32 v65.h, v113
	v_cvt_f16_f32_e32 v65.l, v112
	v_cvt_f16_f32_e32 v66.h, v115
	v_cvt_f16_f32_e32 v66.l, v114
	v_cvt_f16_f32_e32 v67.h, v117
	v_cvt_f16_f32_e32 v67.l, v116
	v_cvt_f16_f32_e32 v68.h, v119
	v_cvt_f16_f32_e32 v68.l, v118
	v_cvt_f16_f32_e32 v42.h, v107
	v_cvt_f16_f32_e32 v42.l, v106
	v_cvt_f16_f32_e32 v43.h, v109
	v_cvt_f16_f32_e32 v43.l, v108
	v_cvt_f16_f32_e32 v44.h, v111
	v_cvt_f16_f32_e32 v44.l, v110
	v_cvt_f16_f32_e32 v41.h, v105
	v_cvt_f16_f32_e32 v41.l, v104
	v_cvt_f16_f32_e32 v77.l, v1
	s_add_co_i32 s6, s6, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s6, s3
	s_cbranch_scc0 .LBB5_26
	s_branch .LBB5_30
.LBB5_29:
	v_dual_mov_b32 v43, v64 :: v_dual_mov_b32 v42, v60
	v_dual_mov_b32 v41, v56 :: v_dual_mov_b32 v66, v72
	v_dual_mov_b32 v67, v52 :: v_dual_mov_b32 v62, v56
	v_dual_mov_b32 v65, v48 :: v_dual_mov_b32 v50, v48
	v_dual_mov_b32 v63, v60 :: v_dual_mov_b32 v58, v55
	v_dual_mov_b32 v61, v55 :: v_dual_mov_b32 v70, v47
	v_mov_b32_e32 v51, v72
	v_mov_b32_e32 v49, v47
	v_mov_b32_e32 v59, v56
	v_mov_b32_e32 v57, v54
	v_mov_b32_e32 v71, v48
	v_mov_b32_e32 v69, v46
.LBB5_30:
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
		.amdhsa_next_free_vgpr 144
		.amdhsa_next_free_sgpr 9
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
	.set .L_Z5probeILi8ELb1ELb1EEvPfPKfi.num_vgpr, 144
	.set .L_Z5probeILi8ELb1ELb1EEvPfPKfi.num_agpr, 0
	.set .L_Z5probeILi8ELb1ELb1EEvPfPKfi.numbered_sgpr, 9
	.set .L_Z5probeILi8ELb1ELb1EEvPfPKfi.num_named_barrier, 0
	.set .L_Z5probeILi8ELb1ELb1EEvPfPKfi.private_seg_size, 0
	.set .L_Z5probeILi8ELb1ELb1EEvPfPKfi.uses_vcc, 1
	.set .L_Z5probeILi8ELb1ELb1EEvPfPKfi.uses_flat_scratch, 0
	.set .L_Z5probeILi8ELb1ELb1EEvPfPKfi.has_dyn_sized_stack, 0
	.set .L_Z5probeILi8ELb1ELb1EEvPfPKfi.has_recursion, 0
	.set .L_Z5probeILi8ELb1ELb1EEvPfPKfi.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 27792
; TotalNumSgprs: 11
; NumVgprs: 144
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 17
; NumSGPRsForWavesPerEU: 11
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
	s_load_b32 s3, s[0:1], 0x10
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_mov_b32 s0, 0x3c23d70a
	v_fma_f32 v17, 0x3c23d70a, v11, 0
	v_fmaak_f32 v18, s0, v11, 0x38d1b717
	v_fmaak_f32 v19, s0, v11, 0x3951b717
	v_fmaak_f32 v20, s0, v11, 0x399d4951
	v_fmaak_f32 v21, s0, v11, 0x39d1b717
	v_fmaak_f32 v22, s0, v11, 0x3a03126e
	v_fmaak_f32 v23, s0, v11, 0x3a1d4951
	v_fmaak_f32 v24, s0, v11, 0x3a378034
	v_fmaak_f32 v32, s0, v11, 0x3a51b717
	v_fmaak_f32 v16, s0, v11, 0x3a6bedfa
	v_fmaak_f32 v8, s0, v11, 0x3a83126e
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_kmcnt 0x0
	s_cmp_lt_i32 s3, 1
	s_cbranch_scc1 .LBB6_29
; %bb.25:
	v_mbcnt_lo_u32_b32 v2, -1, 0
	s_mov_b32 s0, 0x3a83126f
	v_and_b32_e32 v1, 31, v0
	s_wait_alu depctr_sa_sdst(0)
	v_fma_mixlo_f16 v77, v3, s0, 0
	v_fma_mixhi_f16 v73, v4, s0, 0
	v_fma_mixhi_f16 v77, v5, s0, 0
	v_lshrrev_b32_e32 v3, 1, v0
	v_xor_b32_e32 v4, 16, v2
	v_bfi_b32 v5, v2, 0, 32
	v_fma_mixhi_f16 v78, v10, s0, 0
	v_fma_mixlo_f16 v76, v26, s0, 0
	v_dual_mov_b32 v119, 0x3b03126f :: v_dual_lshlrev_b32 v10, 7, v0
	s_delay_alu instid0(VALU_DEP_4)
	v_cmp_lt_u32_e32 vcc_lo, v4, v5
	v_and_b32_e32 v97, 8, v3
	v_fma_mixlo_f16 v73, v11, s0, 0
	v_fma_mixlo_f16 v74, v6, s0, 0
	v_fma_mixlo_f16 v78, v7, s0, 0
	v_cndmask_b32_e32 v2, v2, v4, vcc_lo
	v_or_b32_e32 v26, 23, v97
	v_fma_mixhi_f16 v74, v9, s0, 0
	v_fma_mixhi_f16 v75, v15, s0, 0
	v_fma_mixhi_f16 v79, v25, s0, 0
	v_fma_mixlo_f16 v80, v13, s0, 0
	v_fma_mixhi_f16 v76, v14, s0, 0
	v_fma_mixhi_f16 v80, v12, s0, 0
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
	v_and_b32_e32 v99, 0x800, v10
	v_or_b32_e32 v10, 20, v97
	v_or_b32_e32 v15, 21, v97
	v_or_b32_e32 v25, 22, v97
	v_lshlrev_b32_e32 v118, 2, v26
	v_mov_b32_e32 v26, v19
	v_lshl_add_u32 v98, v1, 2, 0
	v_or_b32_e32 v1, 1, v97
	v_fma_mixlo_f16 v75, v27, s0, 0
	v_fma_mixlo_f16 v79, v28, s0, 0
	v_dual_mov_b32 v30, v23 :: v_dual_lshlrev_b32 v103, 2, v2
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_mov_b32 v29, v22 :: v_dual_lshlrev_b32 v104, 2, v1
	v_lshlrev_b32_e32 v105, 2, v3
	v_dual_mov_b32 v31, v24 :: v_dual_lshlrev_b32 v106, 2, v4
	v_lshlrev_b32_e32 v107, 2, v5
	v_lshlrev_b32_e32 v108, 2, v6
	v_lshlrev_b32_e32 v109, 2, v7
	v_lshlrev_b32_e32 v110, 2, v9
	v_dual_mov_b32 v2, v21 :: v_dual_lshlrev_b32 v111, 2, v11
	v_lshlrev_b32_e32 v112, 2, v12
	v_dual_mov_b32 v4, v23 :: v_dual_lshlrev_b32 v113, 2, v13
	v_lshlrev_b32_e32 v114, 2, v14
	v_dual_mov_b32 v6, v32 :: v_dual_lshlrev_b32 v115, 2, v10
	v_dual_mov_b32 v1, v20 :: v_dual_lshlrev_b32 v116, 2, v15
	v_lshlrev_b32_e32 v117, 2, v25
	v_mov_b32_e32 v25, v18
	v_dual_mov_b32 v27, v20 :: v_dual_mov_b32 v28, v21
	v_add_nc_u32_e32 v100, 0x1800, v99
	v_or_b32_e32 v101, 0x2000, v99
	v_dual_mov_b32 v9, v19 :: v_dual_add_nc_u32 v102, 0xfffff800, v99
	v_dual_mov_b32 v10, v20 :: v_dual_mov_b32 v11, v21
	v_dual_mov_b32 v12, v22 :: v_dual_mov_b32 v13, v23
	v_dual_mov_b32 v14, v24 :: v_dual_mov_b32 v15, v32
	v_mov_b32_e32 v3, v22
	v_mov_b32_e32 v5, v24
	v_mov_b32_e32 v7, v16
	s_mov_b32 s6, 0
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
	s_mov_b32 s7, 0
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
	v_dual_add_f32 v127, 0x37fba882, v40 :: v_dual_add_f32 v126, 0x37fba882, v39
	v_dual_add_f32 v125, 0x37fba882, v38 :: v_dual_add_f32 v124, 0x37fba882, v37
	v_dual_add_f32 v123, 0x37fba882, v36 :: v_dual_add_f32 v122, 0x37fba882, v35
	v_dual_add_f32 v121, 0x37fba882, v34 :: v_dual_add_f32 v120, 0x37fba882, v33
	v_dual_add_f32 v135, 0x3827c5ac, v40 :: v_dual_add_f32 v134, 0x3827c5ac, v39
	v_dual_add_f32 v133, 0x3827c5ac, v38 :: v_dual_add_f32 v132, 0x3827c5ac, v37
	v_dual_add_f32 v131, 0x3827c5ac, v36 :: v_dual_add_f32 v130, 0x3827c5ac, v35
	v_dual_add_f32 v129, 0x3827c5ac, v34 :: v_dual_add_f32 v128, 0x3827c5ac, v33
	v_wmma_f32_16x16x16_f16 v[41:48], v[73:76], v[81:84], v[41:48]
	v_wmma_f32_16x16x16_f16 v[49:56], v[73:76], v[81:84], v[49:56]
	v_wmma_f32_16x16x16_f16 v[120:127], v[73:76], v[81:84], v[120:127]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[128:135], v[73:76], v[81:84], v[128:135]
	v_wmma_f32_16x16x16_f16 v[41:48], v[73:76], v[85:88], v[41:48]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[49:56], v[73:76], v[85:88], v[49:56]
	v_wmma_f32_16x16x16_f16 v[120:127], v[73:76], v[85:88], v[120:127]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[128:135], v[73:76], v[85:88], v[128:135]
	v_wmma_f32_16x16x16_f16 v[41:48], v[73:76], v[89:92], v[41:48]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[49:56], v[73:76], v[89:92], v[49:56]
	v_wmma_f32_16x16x16_f16 v[120:127], v[73:76], v[89:92], v[120:127]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[128:135], v[73:76], v[89:92], v[128:135]
	v_wmma_f32_16x16x16_f16 v[41:48], v[73:76], v[93:96], v[41:48]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[49:56], v[73:76], v[93:96], v[49:56]
	v_wmma_f32_16x16x16_f16 v[120:127], v[73:76], v[93:96], v[120:127]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[128:135], v[73:76], v[93:96], v[128:135]
	v_dual_mul_f32 v65, 0x3727c5ac, v41 :: v_dual_mul_f32 v66, 0x3727c5ac, v42
	v_dual_mul_f32 v67, 0x3727c5ac, v43 :: v_dual_mul_f32 v68, 0x3727c5ac, v44
	v_dual_mul_f32 v69, 0x3727c5ac, v45 :: v_dual_mul_f32 v70, 0x3727c5ac, v46
	v_dual_mul_f32 v71, 0x3727c5ac, v47 :: v_dual_mul_f32 v72, 0x3727c5ac, v48
	v_dual_mul_f32 v57, 0x3727c5ac, v49 :: v_dual_mul_f32 v58, 0x3727c5ac, v50
	v_dual_mul_f32 v59, 0x3727c5ac, v51 :: v_dual_mul_f32 v60, 0x3727c5ac, v52
	v_dual_mul_f32 v61, 0x3727c5ac, v53 :: v_dual_mul_f32 v62, 0x3727c5ac, v54
	v_dual_mul_f32 v63, 0x3727c5ac, v55 :: v_dual_mul_f32 v64, 0x3727c5ac, v56
	v_dual_mul_f32 v49, 0x3727c5ac, v120 :: v_dual_mul_f32 v50, 0x3727c5ac, v121
	v_dual_mul_f32 v51, 0x3727c5ac, v122 :: v_dual_mul_f32 v52, 0x3727c5ac, v123
	v_dual_mul_f32 v53, 0x3727c5ac, v124 :: v_dual_mul_f32 v54, 0x3727c5ac, v125
	v_dual_mul_f32 v55, 0x3727c5ac, v126 :: v_dual_mul_f32 v56, 0x3727c5ac, v127
	v_dual_mul_f32 v41, 0x3727c5ac, v128 :: v_dual_mul_f32 v42, 0x3727c5ac, v129
	v_dual_mul_f32 v43, 0x3727c5ac, v130 :: v_dual_mul_f32 v44, 0x3727c5ac, v131
	v_dual_mul_f32 v45, 0x3727c5ac, v132 :: v_dual_mul_f32 v46, 0x3727c5ac, v133
	v_dual_mul_f32 v47, 0x3727c5ac, v134 :: v_dual_mul_f32 v48, 0x3727c5ac, v135
.LBB6_27:                               ;   Parent Loop BB6_26 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_wait_alu depctr_sa_sdst(0)
	v_cmp_eq_u32_e64 s1, s7, v99
	v_cmp_eq_u32_e64 s2, s7, v101
	v_cmp_eq_u32_e32 vcc_lo, s7, v102
	v_cmp_eq_u32_e64 s0, s7, v100
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v120, 0, v65, s1
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v120, v120, v49, s2
	ds_bpermute_b32 v121, v103, v120
	s_wait_dscnt 0x0
	v_dual_add_f32 v124, v120, v121 :: v_dual_add_nc_u32 v121, s7, v98
	v_lshlrev_b32_e32 v120, 2, v97
	s_addk_co_i32 s7, 0x2000
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s7, 0x4000
	ds_load_2addr_b32 v[122:123], v121 offset1:32
	s_wait_dscnt 0x0
	ds_bpermute_b32 v125, v120, v122
	s_wait_dscnt 0x0
	v_fma_f32 v65, -v124, v125, v65
	ds_bpermute_b32 v125, v104, v122
	s_wait_dscnt 0x0
	v_fma_f32 v66, -v124, v125, v66
	ds_bpermute_b32 v125, v105, v122
	s_wait_dscnt 0x0
	v_fma_f32 v67, -v124, v125, v67
	ds_bpermute_b32 v125, v106, v122
	s_wait_dscnt 0x0
	v_fma_f32 v68, -v124, v125, v68
	ds_bpermute_b32 v125, v107, v122
	s_wait_dscnt 0x0
	v_fma_f32 v69, -v124, v125, v69
	ds_bpermute_b32 v125, v108, v122
	s_wait_dscnt 0x0
	v_fma_f32 v70, -v124, v125, v70
	ds_bpermute_b32 v125, v109, v122
	s_wait_dscnt 0x0
	v_fma_f32 v71, -v124, v125, v71
	ds_bpermute_b32 v125, v110, v122
	s_wait_dscnt 0x0
	v_fma_f32 v72, -v124, v125, v72
	ds_bpermute_b32 v125, v111, v122
	s_wait_dscnt 0x0
	v_fma_f32 v57, -v124, v125, v57
	ds_bpermute_b32 v125, v112, v122
	s_wait_dscnt 0x0
	v_fma_f32 v58, -v124, v125, v58
	ds_bpermute_b32 v125, v113, v122
	s_wait_dscnt 0x0
	v_fma_f32 v59, -v124, v125, v59
	ds_bpermute_b32 v125, v114, v122
	s_wait_dscnt 0x0
	v_fma_f32 v60, -v124, v125, v60
	ds_bpermute_b32 v125, v115, v122
	s_wait_dscnt 0x0
	v_fma_f32 v61, -v124, v125, v61
	ds_bpermute_b32 v125, v116, v122
	s_wait_dscnt 0x0
	v_fma_f32 v62, -v124, v125, v62
	ds_bpermute_b32 v125, v117, v122
	ds_bpermute_b32 v122, v118, v122
	s_wait_dscnt 0x1
	v_fma_f32 v63, -v124, v125, v63
	s_wait_dscnt 0x0
	v_fma_f32 v64, -v124, v122, v64
	ds_bpermute_b32 v122, v120, v123
	s_wait_dscnt 0x0
	v_fma_f32 v49, -v124, v122, v49
	ds_bpermute_b32 v122, v104, v123
	s_wait_dscnt 0x0
	v_fma_f32 v50, -v124, v122, v50
	ds_bpermute_b32 v122, v105, v123
	s_wait_dscnt 0x0
	v_fma_f32 v51, -v124, v122, v51
	ds_bpermute_b32 v122, v106, v123
	s_wait_dscnt 0x0
	v_fma_f32 v52, -v124, v122, v52
	ds_bpermute_b32 v122, v107, v123
	s_wait_dscnt 0x0
	v_fma_f32 v53, -v124, v122, v53
	ds_bpermute_b32 v122, v108, v123
	s_wait_dscnt 0x0
	v_fma_f32 v54, -v124, v122, v54
	ds_bpermute_b32 v122, v109, v123
	s_wait_dscnt 0x0
	v_fma_f32 v55, -v124, v122, v55
	ds_bpermute_b32 v122, v110, v123
	s_wait_dscnt 0x0
	v_fma_f32 v56, -v124, v122, v56
	ds_bpermute_b32 v122, v111, v123
	s_wait_dscnt 0x0
	v_fma_f32 v41, -v124, v122, v41
	ds_bpermute_b32 v122, v112, v123
	s_wait_dscnt 0x0
	v_fma_f32 v42, -v124, v122, v42
	ds_bpermute_b32 v122, v113, v123
	s_wait_dscnt 0x0
	v_fma_f32 v43, -v124, v122, v43
	ds_bpermute_b32 v122, v114, v123
	s_wait_dscnt 0x0
	v_fma_f32 v44, -v124, v122, v44
	ds_bpermute_b32 v122, v115, v123
	s_wait_dscnt 0x0
	v_fma_f32 v45, -v124, v122, v45
	ds_bpermute_b32 v122, v116, v123
	s_wait_dscnt 0x0
	v_fma_f32 v46, -v124, v122, v46
	ds_bpermute_b32 v122, v117, v123
	s_wait_dscnt 0x0
	v_fma_f32 v47, -v124, v122, v47
	ds_bpermute_b32 v122, v118, v123
	s_wait_dscnt 0x0
	v_fma_f32 v48, -v124, v122, v48
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v122, 0, v66, s1
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v122, v122, v50, s2
	ds_bpermute_b32 v123, v103, v122
	s_wait_dscnt 0x0
	v_add_f32_e32 v124, v122, v123
	ds_load_2addr_b32 v[122:123], v121 offset0:64 offset1:96
	s_wait_dscnt 0x0
	ds_bpermute_b32 v125, v120, v122
	s_wait_dscnt 0x0
	v_fma_f32 v65, -v124, v125, v65
	ds_bpermute_b32 v125, v104, v122
	s_wait_dscnt 0x0
	v_fma_f32 v66, -v124, v125, v66
	ds_bpermute_b32 v125, v105, v122
	s_wait_dscnt 0x0
	v_fma_f32 v67, -v124, v125, v67
	ds_bpermute_b32 v125, v106, v122
	s_wait_dscnt 0x0
	v_fma_f32 v68, -v124, v125, v68
	ds_bpermute_b32 v125, v107, v122
	s_wait_dscnt 0x0
	v_fma_f32 v69, -v124, v125, v69
	ds_bpermute_b32 v125, v108, v122
	s_wait_dscnt 0x0
	v_fma_f32 v70, -v124, v125, v70
	ds_bpermute_b32 v125, v109, v122
	s_wait_dscnt 0x0
	v_fma_f32 v71, -v124, v125, v71
	ds_bpermute_b32 v125, v110, v122
	s_wait_dscnt 0x0
	v_fma_f32 v72, -v124, v125, v72
	ds_bpermute_b32 v125, v111, v122
	s_wait_dscnt 0x0
	v_fma_f32 v57, -v124, v125, v57
	ds_bpermute_b32 v125, v112, v122
	s_wait_dscnt 0x0
	v_fma_f32 v58, -v124, v125, v58
	ds_bpermute_b32 v125, v113, v122
	s_wait_dscnt 0x0
	v_fma_f32 v59, -v124, v125, v59
	ds_bpermute_b32 v125, v114, v122
	s_wait_dscnt 0x0
	v_fma_f32 v60, -v124, v125, v60
	ds_bpermute_b32 v125, v115, v122
	s_wait_dscnt 0x0
	v_fma_f32 v61, -v124, v125, v61
	ds_bpermute_b32 v125, v116, v122
	s_wait_dscnt 0x0
	v_fma_f32 v62, -v124, v125, v62
	ds_bpermute_b32 v125, v117, v122
	ds_bpermute_b32 v122, v118, v122
	s_wait_dscnt 0x1
	v_fma_f32 v63, -v124, v125, v63
	s_wait_dscnt 0x0
	v_fma_f32 v64, -v124, v122, v64
	ds_bpermute_b32 v122, v120, v123
	s_wait_dscnt 0x0
	v_fma_f32 v49, -v124, v122, v49
	ds_bpermute_b32 v122, v104, v123
	s_wait_dscnt 0x0
	v_fma_f32 v50, -v124, v122, v50
	ds_bpermute_b32 v122, v105, v123
	s_wait_dscnt 0x0
	v_fma_f32 v51, -v124, v122, v51
	ds_bpermute_b32 v122, v106, v123
	s_wait_dscnt 0x0
	v_fma_f32 v52, -v124, v122, v52
	ds_bpermute_b32 v122, v107, v123
	s_wait_dscnt 0x0
	v_fma_f32 v53, -v124, v122, v53
	ds_bpermute_b32 v122, v108, v123
	s_wait_dscnt 0x0
	v_fma_f32 v54, -v124, v122, v54
	ds_bpermute_b32 v122, v109, v123
	s_wait_dscnt 0x0
	v_fma_f32 v55, -v124, v122, v55
	ds_bpermute_b32 v122, v110, v123
	s_wait_dscnt 0x0
	v_fma_f32 v56, -v124, v122, v56
	ds_bpermute_b32 v122, v111, v123
	s_wait_dscnt 0x0
	v_fma_f32 v41, -v124, v122, v41
	ds_bpermute_b32 v122, v112, v123
	s_wait_dscnt 0x0
	v_fma_f32 v42, -v124, v122, v42
	ds_bpermute_b32 v122, v113, v123
	s_wait_dscnt 0x0
	v_fma_f32 v43, -v124, v122, v43
	ds_bpermute_b32 v122, v114, v123
	s_wait_dscnt 0x0
	v_fma_f32 v44, -v124, v122, v44
	ds_bpermute_b32 v122, v115, v123
	s_wait_dscnt 0x0
	v_fma_f32 v45, -v124, v122, v45
	ds_bpermute_b32 v122, v116, v123
	s_wait_dscnt 0x0
	v_fma_f32 v46, -v124, v122, v46
	ds_bpermute_b32 v122, v117, v123
	s_wait_dscnt 0x0
	v_fma_f32 v47, -v124, v122, v47
	ds_bpermute_b32 v122, v118, v123
	s_wait_dscnt 0x0
	v_fma_f32 v48, -v124, v122, v48
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v122, 0, v67, s1
	ds_load_2addr_b32 v[136:137], v121 offset0:128 offset1:160
	v_cndmask_b32_e64 v122, v122, v51, s2
	ds_bpermute_b32 v123, v103, v122
	s_wait_dscnt 0x0
	v_add_f32_e32 v138, v122, v123
	ds_bpermute_b32 v122, v120, v136
	s_wait_dscnt 0x0
	v_fma_f32 v65, -v138, v122, v65
	ds_bpermute_b32 v122, v104, v136
	s_wait_dscnt 0x0
	v_fma_f32 v66, -v138, v122, v66
	ds_bpermute_b32 v122, v105, v136
	s_wait_dscnt 0x0
	v_fma_f32 v67, -v138, v122, v67
	ds_bpermute_b32 v122, v106, v136
	s_wait_dscnt 0x0
	v_fma_f32 v68, -v138, v122, v68
	ds_bpermute_b32 v122, v107, v136
	s_wait_dscnt 0x0
	v_fma_f32 v69, -v138, v122, v69
	ds_bpermute_b32 v122, v108, v136
	s_wait_dscnt 0x0
	v_fma_f32 v70, -v138, v122, v70
	ds_bpermute_b32 v122, v109, v136
	s_wait_dscnt 0x0
	v_fma_f32 v71, -v138, v122, v71
	ds_bpermute_b32 v122, v110, v136
	s_wait_dscnt 0x0
	v_fma_f32 v72, -v138, v122, v72
	ds_bpermute_b32 v122, v111, v136
	s_wait_dscnt 0x0
	v_fma_f32 v57, -v138, v122, v57
	ds_bpermute_b32 v122, v112, v136
	s_wait_dscnt 0x0
	v_fma_f32 v58, -v138, v122, v58
	ds_bpermute_b32 v122, v113, v136
	s_wait_dscnt 0x0
	v_fma_f32 v59, -v138, v122, v59
	ds_bpermute_b32 v122, v114, v136
	s_wait_dscnt 0x0
	v_fma_f32 v60, -v138, v122, v60
	ds_bpermute_b32 v122, v115, v136
	s_wait_dscnt 0x0
	v_fma_f32 v61, -v138, v122, v61
	ds_bpermute_b32 v122, v116, v136
	s_wait_dscnt 0x0
	v_fma_f32 v62, -v138, v122, v62
	ds_bpermute_b32 v122, v117, v136
	s_wait_dscnt 0x0
	v_fma_f32 v63, -v138, v122, v63
	ds_bpermute_b32 v122, v118, v136
	s_wait_dscnt 0x0
	v_fma_f32 v64, -v138, v122, v64
	ds_bpermute_b32 v122, v120, v137
	s_wait_dscnt 0x0
	v_fma_f32 v122, -v138, v122, v49
	ds_bpermute_b32 v49, v104, v137
	s_wait_dscnt 0x0
	v_fma_f32 v123, -v138, v49, v50
	ds_bpermute_b32 v49, v105, v137
	s_wait_dscnt 0x0
	v_fma_f32 v124, -v138, v49, v51
	ds_bpermute_b32 v49, v106, v137
	s_wait_dscnt 0x0
	v_fma_f32 v125, -v138, v49, v52
	ds_bpermute_b32 v49, v107, v137
	s_wait_dscnt 0x0
	v_fma_f32 v126, -v138, v49, v53
	ds_bpermute_b32 v49, v108, v137
	s_wait_dscnt 0x0
	v_fma_f32 v127, -v138, v49, v54
	ds_bpermute_b32 v49, v109, v137
	s_wait_dscnt 0x0
	v_fma_f32 v128, -v138, v49, v55
	ds_bpermute_b32 v49, v110, v137
	s_wait_dscnt 0x0
	v_fma_f32 v129, -v138, v49, v56
	ds_bpermute_b32 v49, v111, v137
	s_wait_dscnt 0x0
	v_fma_f32 v130, -v138, v49, v41
	ds_bpermute_b32 v41, v112, v137
	s_wait_dscnt 0x0
	v_fma_f32 v131, -v138, v41, v42
	ds_bpermute_b32 v41, v113, v137
	s_wait_dscnt 0x0
	v_fma_f32 v132, -v138, v41, v43
	ds_bpermute_b32 v41, v114, v137
	s_wait_dscnt 0x0
	v_fma_f32 v133, -v138, v41, v44
	ds_bpermute_b32 v41, v115, v137
	s_wait_dscnt 0x0
	v_fma_f32 v134, -v138, v41, v45
	ds_bpermute_b32 v41, v116, v137
	s_wait_dscnt 0x0
	v_fma_f32 v135, -v138, v41, v46
	ds_bpermute_b32 v41, v117, v137
	s_wait_dscnt 0x0
	v_fma_f32 v136, -v138, v41, v47
	ds_bpermute_b32 v41, v118, v137
	s_wait_dscnt 0x0
	v_fma_f32 v137, -v138, v41, v48
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v41, 0, v68, s1
	ds_load_2addr_b32 v[138:139], v121 offset0:192 offset1:224
	v_cndmask_b32_e64 v41, v41, v125, s2
	ds_bpermute_b32 v42, v103, v41
	s_wait_dscnt 0x1
	ds_bpermute_b32 v43, v105, v138
	ds_bpermute_b32 v44, v106, v138
	ds_bpermute_b32 v45, v107, v138
	ds_bpermute_b32 v46, v108, v138
	ds_bpermute_b32 v47, v109, v138
	ds_bpermute_b32 v48, v110, v138
	ds_bpermute_b32 v49, v111, v138
	ds_bpermute_b32 v50, v112, v138
	ds_bpermute_b32 v51, v113, v138
	s_wait_dscnt 0x9
	v_add_f32_e32 v140, v41, v42
	ds_bpermute_b32 v41, v120, v138
	ds_bpermute_b32 v42, v104, v138
	ds_bpermute_b32 v52, v114, v138
	ds_bpermute_b32 v53, v115, v138
	ds_bpermute_b32 v54, v116, v138
	ds_bpermute_b32 v55, v117, v138
	ds_bpermute_b32 v56, v118, v138
	v_add_nc_u32_e32 v138, 0x1800, v121
	s_wait_dscnt 0xf
	v_fma_f32 v43, -v140, v43, v67
	s_wait_dscnt 0xe
	v_fma_f32 v44, -v140, v44, v68
	s_wait_dscnt 0xd
	v_fma_f32 v45, -v140, v45, v69
	s_wait_dscnt 0xc
	v_fma_f32 v46, -v140, v46, v70
	s_wait_dscnt 0xb
	v_fma_f32 v47, -v140, v47, v71
	s_wait_dscnt 0xa
	v_fma_f32 v48, -v140, v48, v72
	s_wait_dscnt 0x9
	v_fma_f32 v49, -v140, v49, v57
	s_wait_dscnt 0x8
	v_fma_f32 v50, -v140, v50, v58
	s_wait_dscnt 0x7
	v_fma_f32 v51, -v140, v51, v59
	ds_bpermute_b32 v57, v120, v139
	s_wait_dscnt 0x7
	v_fma_f32 v41, -v140, v41, v65
	s_wait_dscnt 0x6
	v_fma_f32 v42, -v140, v42, v66
	s_wait_dscnt 0x5
	v_fma_f32 v52, -v140, v52, v60
	s_wait_dscnt 0x4
	v_fma_f32 v53, -v140, v53, v61
	s_wait_dscnt 0x3
	v_fma_f32 v54, -v140, v54, v62
	s_wait_dscnt 0x2
	v_fma_f32 v55, -v140, v55, v63
	s_wait_dscnt 0x1
	v_fma_f32 v56, -v140, v56, v64
	ds_bpermute_b32 v58, v104, v139
	ds_bpermute_b32 v59, v105, v139
	ds_bpermute_b32 v60, v106, v139
	ds_bpermute_b32 v61, v107, v139
	ds_bpermute_b32 v62, v108, v139
	ds_bpermute_b32 v63, v109, v139
	ds_bpermute_b32 v64, v110, v139
	ds_bpermute_b32 v65, v111, v139
	ds_bpermute_b32 v66, v112, v139
	ds_bpermute_b32 v67, v113, v139
	ds_bpermute_b32 v68, v114, v139
	ds_bpermute_b32 v69, v115, v139
	ds_bpermute_b32 v70, v116, v139
	ds_bpermute_b32 v71, v117, v139
	ds_bpermute_b32 v72, v118, v139
	s_wait_dscnt 0xf
	v_fma_f32 v57, -v140, v57, v122
	s_wait_dscnt 0xe
	v_fma_f32 v58, -v140, v58, v123
	s_wait_dscnt 0xd
	v_fma_f32 v59, -v140, v59, v124
	s_wait_dscnt 0xc
	v_fma_f32 v60, -v140, v60, v125
	s_wait_dscnt 0xb
	v_fma_f32 v61, -v140, v61, v126
	s_wait_dscnt 0xa
	v_fma_f32 v62, -v140, v62, v127
	s_wait_dscnt 0x9
	v_fma_f32 v63, -v140, v63, v128
	s_wait_dscnt 0x8
	v_fma_f32 v64, -v140, v64, v129
	s_wait_dscnt 0x7
	v_fma_f32 v65, -v140, v65, v130
	s_wait_dscnt 0x6
	v_fma_f32 v66, -v140, v66, v131
	s_wait_dscnt 0x5
	v_fma_f32 v67, -v140, v67, v132
	s_wait_dscnt 0x4
	v_fma_f32 v68, -v140, v68, v133
	s_wait_dscnt 0x3
	v_fma_f32 v69, -v140, v69, v134
	s_wait_dscnt 0x2
	v_fma_f32 v70, -v140, v70, v135
	s_wait_dscnt 0x1
	v_fma_f32 v71, -v140, v71, v136
	s_wait_dscnt 0x0
	v_fma_f32 v72, -v140, v72, v137
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v122, 0, v45, s1
	v_add_nc_u32_e32 v125, 0x400, v121
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e64 v122, v122, v61, s2
	ds_bpermute_b32 v123, v103, v122
	s_wait_dscnt 0x0
	v_add_f32_e32 v124, v122, v123
	ds_load_2addr_b32 v[122:123], v125 offset1:32
	s_wait_dscnt 0x0
	ds_bpermute_b32 v126, v120, v122
	s_wait_dscnt 0x0
	v_fma_f32 v41, -v124, v126, v41
	ds_bpermute_b32 v126, v104, v122
	s_wait_dscnt 0x0
	v_fma_f32 v42, -v124, v126, v42
	ds_bpermute_b32 v126, v105, v122
	s_wait_dscnt 0x0
	v_fma_f32 v43, -v124, v126, v43
	ds_bpermute_b32 v126, v106, v122
	s_wait_dscnt 0x0
	v_fma_f32 v44, -v124, v126, v44
	ds_bpermute_b32 v126, v107, v122
	s_wait_dscnt 0x0
	v_fma_f32 v45, -v124, v126, v45
	ds_bpermute_b32 v126, v108, v122
	s_wait_dscnt 0x0
	v_fma_f32 v46, -v124, v126, v46
	ds_bpermute_b32 v126, v109, v122
	s_wait_dscnt 0x0
	v_fma_f32 v47, -v124, v126, v47
	ds_bpermute_b32 v126, v110, v122
	s_wait_dscnt 0x0
	v_fma_f32 v48, -v124, v126, v48
	ds_bpermute_b32 v126, v111, v122
	s_wait_dscnt 0x0
	v_fma_f32 v49, -v124, v126, v49
	ds_bpermute_b32 v126, v112, v122
	s_wait_dscnt 0x0
	v_fma_f32 v50, -v124, v126, v50
	ds_bpermute_b32 v126, v113, v122
	s_wait_dscnt 0x0
	v_fma_f32 v51, -v124, v126, v51
	ds_bpermute_b32 v126, v114, v122
	s_wait_dscnt 0x0
	v_fma_f32 v52, -v124, v126, v52
	ds_bpermute_b32 v126, v115, v122
	s_wait_dscnt 0x0
	v_fma_f32 v53, -v124, v126, v53
	ds_bpermute_b32 v126, v116, v122
	s_wait_dscnt 0x0
	v_fma_f32 v54, -v124, v126, v54
	ds_bpermute_b32 v126, v117, v122
	ds_bpermute_b32 v122, v118, v122
	s_wait_dscnt 0x1
	v_fma_f32 v55, -v124, v126, v55
	s_wait_dscnt 0x0
	v_fma_f32 v56, -v124, v122, v56
	ds_bpermute_b32 v122, v120, v123
	s_wait_dscnt 0x0
	v_fma_f32 v57, -v124, v122, v57
	ds_bpermute_b32 v122, v104, v123
	s_wait_dscnt 0x0
	v_fma_f32 v58, -v124, v122, v58
	ds_bpermute_b32 v122, v105, v123
	s_wait_dscnt 0x0
	v_fma_f32 v59, -v124, v122, v59
	ds_bpermute_b32 v122, v106, v123
	s_wait_dscnt 0x0
	v_fma_f32 v60, -v124, v122, v60
	ds_bpermute_b32 v122, v107, v123
	s_wait_dscnt 0x0
	v_fma_f32 v61, -v124, v122, v61
	ds_bpermute_b32 v122, v108, v123
	s_wait_dscnt 0x0
	v_fma_f32 v62, -v124, v122, v62
	ds_bpermute_b32 v122, v109, v123
	s_wait_dscnt 0x0
	v_fma_f32 v63, -v124, v122, v63
	ds_bpermute_b32 v122, v110, v123
	s_wait_dscnt 0x0
	v_fma_f32 v64, -v124, v122, v64
	ds_bpermute_b32 v122, v111, v123
	s_wait_dscnt 0x0
	v_fma_f32 v65, -v124, v122, v65
	ds_bpermute_b32 v122, v112, v123
	s_wait_dscnt 0x0
	v_fma_f32 v66, -v124, v122, v66
	ds_bpermute_b32 v122, v113, v123
	s_wait_dscnt 0x0
	v_fma_f32 v67, -v124, v122, v67
	ds_bpermute_b32 v122, v114, v123
	s_wait_dscnt 0x0
	v_fma_f32 v68, -v124, v122, v68
	ds_bpermute_b32 v122, v115, v123
	s_wait_dscnt 0x0
	v_fma_f32 v69, -v124, v122, v69
	ds_bpermute_b32 v122, v116, v123
	s_wait_dscnt 0x0
	v_fma_f32 v70, -v124, v122, v70
	ds_bpermute_b32 v122, v117, v123
	s_wait_dscnt 0x0
	v_fma_f32 v71, -v124, v122, v71
	ds_bpermute_b32 v122, v118, v123
	s_wait_dscnt 0x0
	v_fma_f32 v72, -v124, v122, v72
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v122, 0, v46, s1
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v122, v122, v62, s2
	ds_bpermute_b32 v123, v103, v122
	s_wait_dscnt 0x0
	v_add_f32_e32 v124, v122, v123
	ds_load_2addr_b32 v[122:123], v125 offset0:64 offset1:96
	s_wait_dscnt 0x0
	ds_bpermute_b32 v126, v120, v122
	s_wait_dscnt 0x0
	v_fma_f32 v41, -v124, v126, v41
	ds_bpermute_b32 v126, v104, v122
	s_wait_dscnt 0x0
	v_fma_f32 v42, -v124, v126, v42
	ds_bpermute_b32 v126, v105, v122
	s_wait_dscnt 0x0
	v_fma_f32 v43, -v124, v126, v43
	ds_bpermute_b32 v126, v106, v122
	s_wait_dscnt 0x0
	v_fma_f32 v44, -v124, v126, v44
	ds_bpermute_b32 v126, v107, v122
	s_wait_dscnt 0x0
	v_fma_f32 v45, -v124, v126, v45
	ds_bpermute_b32 v126, v108, v122
	s_wait_dscnt 0x0
	v_fma_f32 v46, -v124, v126, v46
	ds_bpermute_b32 v126, v109, v122
	s_wait_dscnt 0x0
	v_fma_f32 v47, -v124, v126, v47
	ds_bpermute_b32 v126, v110, v122
	s_wait_dscnt 0x0
	v_fma_f32 v48, -v124, v126, v48
	ds_bpermute_b32 v126, v111, v122
	s_wait_dscnt 0x0
	v_fma_f32 v49, -v124, v126, v49
	ds_bpermute_b32 v126, v112, v122
	s_wait_dscnt 0x0
	v_fma_f32 v50, -v124, v126, v50
	ds_bpermute_b32 v126, v113, v122
	s_wait_dscnt 0x0
	v_fma_f32 v51, -v124, v126, v51
	ds_bpermute_b32 v126, v114, v122
	s_wait_dscnt 0x0
	v_fma_f32 v52, -v124, v126, v52
	ds_bpermute_b32 v126, v115, v122
	s_wait_dscnt 0x0
	v_fma_f32 v53, -v124, v126, v53
	ds_bpermute_b32 v126, v116, v122
	s_wait_dscnt 0x0
	v_fma_f32 v54, -v124, v126, v54
	ds_bpermute_b32 v126, v117, v122
	ds_bpermute_b32 v122, v118, v122
	s_wait_dscnt 0x1
	v_fma_f32 v55, -v124, v126, v55
	s_wait_dscnt 0x0
	v_fma_f32 v56, -v124, v122, v56
	ds_bpermute_b32 v122, v120, v123
	s_wait_dscnt 0x0
	v_fma_f32 v57, -v124, v122, v57
	ds_bpermute_b32 v122, v104, v123
	s_wait_dscnt 0x0
	v_fma_f32 v58, -v124, v122, v58
	ds_bpermute_b32 v122, v105, v123
	s_wait_dscnt 0x0
	v_fma_f32 v59, -v124, v122, v59
	ds_bpermute_b32 v122, v106, v123
	s_wait_dscnt 0x0
	v_fma_f32 v60, -v124, v122, v60
	ds_bpermute_b32 v122, v107, v123
	s_wait_dscnt 0x0
	v_fma_f32 v61, -v124, v122, v61
	ds_bpermute_b32 v122, v108, v123
	s_wait_dscnt 0x0
	v_fma_f32 v62, -v124, v122, v62
	ds_bpermute_b32 v122, v109, v123
	s_wait_dscnt 0x0
	v_fma_f32 v63, -v124, v122, v63
	ds_bpermute_b32 v122, v110, v123
	s_wait_dscnt 0x0
	v_fma_f32 v64, -v124, v122, v64
	ds_bpermute_b32 v122, v111, v123
	s_wait_dscnt 0x0
	v_fma_f32 v65, -v124, v122, v65
	ds_bpermute_b32 v122, v112, v123
	s_wait_dscnt 0x0
	v_fma_f32 v66, -v124, v122, v66
	ds_bpermute_b32 v122, v113, v123
	s_wait_dscnt 0x0
	v_fma_f32 v67, -v124, v122, v67
	ds_bpermute_b32 v122, v114, v123
	s_wait_dscnt 0x0
	v_fma_f32 v68, -v124, v122, v68
	ds_bpermute_b32 v122, v115, v123
	s_wait_dscnt 0x0
	v_fma_f32 v69, -v124, v122, v69
	ds_bpermute_b32 v122, v116, v123
	s_wait_dscnt 0x0
	v_fma_f32 v70, -v124, v122, v70
	ds_bpermute_b32 v122, v117, v123
	s_wait_dscnt 0x0
	v_fma_f32 v71, -v124, v122, v71
	ds_bpermute_b32 v122, v118, v123
	s_wait_dscnt 0x0
	v_fma_f32 v72, -v124, v122, v72
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v122, 0, v47, s1
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v122, v122, v63, s2
	ds_bpermute_b32 v123, v103, v122
	s_wait_dscnt 0x0
	v_add_f32_e32 v124, v122, v123
	ds_load_2addr_b32 v[122:123], v125 offset0:128 offset1:160
	s_wait_dscnt 0x0
	ds_bpermute_b32 v126, v120, v122
	s_wait_dscnt 0x0
	v_fma_f32 v41, -v124, v126, v41
	ds_bpermute_b32 v126, v104, v122
	s_wait_dscnt 0x0
	v_fma_f32 v42, -v124, v126, v42
	ds_bpermute_b32 v126, v105, v122
	s_wait_dscnt 0x0
	v_fma_f32 v43, -v124, v126, v43
	ds_bpermute_b32 v126, v106, v122
	s_wait_dscnt 0x0
	v_fma_f32 v44, -v124, v126, v44
	ds_bpermute_b32 v126, v107, v122
	s_wait_dscnt 0x0
	v_fma_f32 v45, -v124, v126, v45
	ds_bpermute_b32 v126, v108, v122
	s_wait_dscnt 0x0
	v_fma_f32 v46, -v124, v126, v46
	ds_bpermute_b32 v126, v109, v122
	s_wait_dscnt 0x0
	v_fma_f32 v47, -v124, v126, v47
	ds_bpermute_b32 v126, v110, v122
	s_wait_dscnt 0x0
	v_fma_f32 v48, -v124, v126, v48
	ds_bpermute_b32 v126, v111, v122
	s_wait_dscnt 0x0
	v_fma_f32 v49, -v124, v126, v49
	ds_bpermute_b32 v126, v112, v122
	s_wait_dscnt 0x0
	v_fma_f32 v50, -v124, v126, v50
	ds_bpermute_b32 v126, v113, v122
	s_wait_dscnt 0x0
	v_fma_f32 v51, -v124, v126, v51
	ds_bpermute_b32 v126, v114, v122
	s_wait_dscnt 0x0
	v_fma_f32 v52, -v124, v126, v52
	ds_bpermute_b32 v126, v115, v122
	s_wait_dscnt 0x0
	v_fma_f32 v53, -v124, v126, v53
	ds_bpermute_b32 v126, v116, v122
	s_wait_dscnt 0x0
	v_fma_f32 v54, -v124, v126, v54
	ds_bpermute_b32 v126, v117, v122
	ds_bpermute_b32 v122, v118, v122
	s_wait_dscnt 0x1
	v_fma_f32 v55, -v124, v126, v55
	s_wait_dscnt 0x0
	v_fma_f32 v56, -v124, v122, v56
	ds_bpermute_b32 v122, v120, v123
	s_wait_dscnt 0x0
	v_fma_f32 v57, -v124, v122, v57
	ds_bpermute_b32 v122, v104, v123
	s_wait_dscnt 0x0
	v_fma_f32 v58, -v124, v122, v58
	ds_bpermute_b32 v122, v105, v123
	s_wait_dscnt 0x0
	v_fma_f32 v59, -v124, v122, v59
	ds_bpermute_b32 v122, v106, v123
	s_wait_dscnt 0x0
	v_fma_f32 v60, -v124, v122, v60
	ds_bpermute_b32 v122, v107, v123
	s_wait_dscnt 0x0
	v_fma_f32 v61, -v124, v122, v61
	ds_bpermute_b32 v122, v108, v123
	s_wait_dscnt 0x0
	v_fma_f32 v62, -v124, v122, v62
	ds_bpermute_b32 v122, v109, v123
	s_wait_dscnt 0x0
	v_fma_f32 v63, -v124, v122, v63
	ds_bpermute_b32 v122, v110, v123
	s_wait_dscnt 0x0
	v_fma_f32 v64, -v124, v122, v64
	ds_bpermute_b32 v122, v111, v123
	s_wait_dscnt 0x0
	v_fma_f32 v65, -v124, v122, v65
	ds_bpermute_b32 v122, v112, v123
	s_wait_dscnt 0x0
	v_fma_f32 v66, -v124, v122, v66
	ds_bpermute_b32 v122, v113, v123
	s_wait_dscnt 0x0
	v_fma_f32 v67, -v124, v122, v67
	ds_bpermute_b32 v122, v114, v123
	s_wait_dscnt 0x0
	v_fma_f32 v68, -v124, v122, v68
	ds_bpermute_b32 v122, v115, v123
	s_wait_dscnt 0x0
	v_fma_f32 v69, -v124, v122, v69
	ds_bpermute_b32 v122, v116, v123
	s_wait_dscnt 0x0
	v_fma_f32 v70, -v124, v122, v70
	ds_bpermute_b32 v122, v117, v123
	s_wait_dscnt 0x0
	v_fma_f32 v71, -v124, v122, v71
	ds_bpermute_b32 v122, v118, v123
	s_wait_dscnt 0x0
	v_fma_f32 v72, -v124, v122, v72
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v122, 0, v48, s1
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v122, v122, v64, s2
	ds_bpermute_b32 v123, v103, v122
	s_wait_dscnt 0x0
	v_add_f32_e32 v124, v122, v123
	ds_load_2addr_b32 v[122:123], v125 offset0:192 offset1:224
	s_wait_dscnt 0x0
	ds_bpermute_b32 v125, v120, v122
	s_wait_dscnt 0x0
	v_fma_f32 v41, -v124, v125, v41
	ds_bpermute_b32 v125, v104, v122
	s_wait_dscnt 0x0
	v_fma_f32 v42, -v124, v125, v42
	ds_bpermute_b32 v125, v105, v122
	s_wait_dscnt 0x0
	v_fma_f32 v43, -v124, v125, v43
	ds_bpermute_b32 v125, v106, v122
	s_wait_dscnt 0x0
	v_fma_f32 v44, -v124, v125, v44
	ds_bpermute_b32 v125, v107, v122
	s_wait_dscnt 0x0
	v_fma_f32 v45, -v124, v125, v45
	ds_bpermute_b32 v125, v108, v122
	s_wait_dscnt 0x0
	v_fma_f32 v46, -v124, v125, v46
	ds_bpermute_b32 v125, v109, v122
	s_wait_dscnt 0x0
	v_fma_f32 v47, -v124, v125, v47
	ds_bpermute_b32 v125, v110, v122
	s_wait_dscnt 0x0
	v_fma_f32 v48, -v124, v125, v48
	ds_bpermute_b32 v125, v111, v122
	s_wait_dscnt 0x0
	v_fma_f32 v49, -v124, v125, v49
	ds_bpermute_b32 v125, v112, v122
	s_wait_dscnt 0x0
	v_fma_f32 v50, -v124, v125, v50
	ds_bpermute_b32 v125, v113, v122
	s_wait_dscnt 0x0
	v_fma_f32 v51, -v124, v125, v51
	ds_bpermute_b32 v125, v114, v122
	s_wait_dscnt 0x0
	v_fma_f32 v52, -v124, v125, v52
	ds_bpermute_b32 v125, v115, v122
	s_wait_dscnt 0x0
	v_fma_f32 v53, -v124, v125, v53
	ds_bpermute_b32 v125, v116, v122
	s_wait_dscnt 0x0
	v_fma_f32 v54, -v124, v125, v54
	ds_bpermute_b32 v125, v117, v122
	ds_bpermute_b32 v122, v118, v122
	s_wait_dscnt 0x1
	v_fma_f32 v55, -v124, v125, v55
	s_wait_dscnt 0x0
	v_fma_f32 v56, -v124, v122, v56
	ds_bpermute_b32 v122, v120, v123
	v_add_nc_u32_e32 v125, 0x800, v121
	s_wait_dscnt 0x0
	v_fma_f32 v57, -v124, v122, v57
	ds_bpermute_b32 v122, v104, v123
	s_wait_dscnt 0x0
	v_fma_f32 v58, -v124, v122, v58
	ds_bpermute_b32 v122, v105, v123
	s_wait_dscnt 0x0
	v_fma_f32 v59, -v124, v122, v59
	ds_bpermute_b32 v122, v106, v123
	s_wait_dscnt 0x0
	v_fma_f32 v60, -v124, v122, v60
	ds_bpermute_b32 v122, v107, v123
	s_wait_dscnt 0x0
	v_fma_f32 v61, -v124, v122, v61
	ds_bpermute_b32 v122, v108, v123
	s_wait_dscnt 0x0
	v_fma_f32 v62, -v124, v122, v62
	ds_bpermute_b32 v122, v109, v123
	s_wait_dscnt 0x0
	v_fma_f32 v63, -v124, v122, v63
	ds_bpermute_b32 v122, v110, v123
	s_wait_dscnt 0x0
	v_fma_f32 v64, -v124, v122, v64
	ds_bpermute_b32 v122, v111, v123
	s_wait_dscnt 0x0
	v_fma_f32 v65, -v124, v122, v65
	ds_bpermute_b32 v122, v112, v123
	s_wait_dscnt 0x0
	v_fma_f32 v66, -v124, v122, v66
	ds_bpermute_b32 v122, v113, v123
	s_wait_dscnt 0x0
	v_fma_f32 v67, -v124, v122, v67
	ds_bpermute_b32 v122, v114, v123
	s_wait_dscnt 0x0
	v_fma_f32 v68, -v124, v122, v68
	ds_bpermute_b32 v122, v115, v123
	s_wait_dscnt 0x0
	v_fma_f32 v69, -v124, v122, v69
	ds_bpermute_b32 v122, v116, v123
	s_wait_dscnt 0x0
	v_fma_f32 v70, -v124, v122, v70
	ds_bpermute_b32 v122, v117, v123
	s_wait_dscnt 0x0
	v_fma_f32 v71, -v124, v122, v71
	ds_bpermute_b32 v122, v118, v123
	s_wait_dscnt 0x0
	v_fma_f32 v72, -v124, v122, v72
	;;#ASMSTART
	;;#ASMEND
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v122, 0, v41, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v122, v122, v57, s0
	ds_bpermute_b32 v123, v103, v122
	s_wait_dscnt 0x0
	v_add_f32_e32 v124, v122, v123
	ds_load_2addr_b32 v[122:123], v125 offset1:32
	s_wait_dscnt 0x0
	ds_bpermute_b32 v126, v120, v122
	s_wait_dscnt 0x0
	v_fma_f32 v41, -v124, v126, v41
	ds_bpermute_b32 v126, v104, v122
	s_wait_dscnt 0x0
	v_fma_f32 v42, -v124, v126, v42
	ds_bpermute_b32 v126, v105, v122
	s_wait_dscnt 0x0
	v_fma_f32 v43, -v124, v126, v43
	ds_bpermute_b32 v126, v106, v122
	s_wait_dscnt 0x0
	v_fma_f32 v44, -v124, v126, v44
	ds_bpermute_b32 v126, v107, v122
	s_wait_dscnt 0x0
	v_fma_f32 v45, -v124, v126, v45
	ds_bpermute_b32 v126, v108, v122
	s_wait_dscnt 0x0
	v_fma_f32 v46, -v124, v126, v46
	ds_bpermute_b32 v126, v109, v122
	s_wait_dscnt 0x0
	v_fma_f32 v47, -v124, v126, v47
	ds_bpermute_b32 v126, v110, v122
	s_wait_dscnt 0x0
	v_fma_f32 v48, -v124, v126, v48
	ds_bpermute_b32 v126, v111, v122
	s_wait_dscnt 0x0
	v_fma_f32 v49, -v124, v126, v49
	ds_bpermute_b32 v126, v112, v122
	s_wait_dscnt 0x0
	v_fma_f32 v50, -v124, v126, v50
	ds_bpermute_b32 v126, v113, v122
	s_wait_dscnt 0x0
	v_fma_f32 v51, -v124, v126, v51
	ds_bpermute_b32 v126, v114, v122
	s_wait_dscnt 0x0
	v_fma_f32 v52, -v124, v126, v52
	ds_bpermute_b32 v126, v115, v122
	s_wait_dscnt 0x0
	v_fma_f32 v53, -v124, v126, v53
	ds_bpermute_b32 v126, v116, v122
	s_wait_dscnt 0x0
	v_fma_f32 v54, -v124, v126, v54
	ds_bpermute_b32 v126, v117, v122
	ds_bpermute_b32 v122, v118, v122
	s_wait_dscnt 0x1
	v_fma_f32 v55, -v124, v126, v55
	s_wait_dscnt 0x0
	v_fma_f32 v56, -v124, v122, v56
	ds_bpermute_b32 v122, v120, v123
	s_wait_dscnt 0x0
	v_fma_f32 v57, -v124, v122, v57
	ds_bpermute_b32 v122, v104, v123
	s_wait_dscnt 0x0
	v_fma_f32 v58, -v124, v122, v58
	ds_bpermute_b32 v122, v105, v123
	s_wait_dscnt 0x0
	v_fma_f32 v59, -v124, v122, v59
	ds_bpermute_b32 v122, v106, v123
	s_wait_dscnt 0x0
	v_fma_f32 v60, -v124, v122, v60
	ds_bpermute_b32 v122, v107, v123
	s_wait_dscnt 0x0
	v_fma_f32 v61, -v124, v122, v61
	ds_bpermute_b32 v122, v108, v123
	s_wait_dscnt 0x0
	v_fma_f32 v62, -v124, v122, v62
	ds_bpermute_b32 v122, v109, v123
	s_wait_dscnt 0x0
	v_fma_f32 v63, -v124, v122, v63
	ds_bpermute_b32 v122, v110, v123
	s_wait_dscnt 0x0
	v_fma_f32 v64, -v124, v122, v64
	ds_bpermute_b32 v122, v111, v123
	s_wait_dscnt 0x0
	v_fma_f32 v65, -v124, v122, v65
	ds_bpermute_b32 v122, v112, v123
	s_wait_dscnt 0x0
	v_fma_f32 v66, -v124, v122, v66
	ds_bpermute_b32 v122, v113, v123
	s_wait_dscnt 0x0
	v_fma_f32 v67, -v124, v122, v67
	ds_bpermute_b32 v122, v114, v123
	s_wait_dscnt 0x0
	v_fma_f32 v68, -v124, v122, v68
	ds_bpermute_b32 v122, v115, v123
	s_wait_dscnt 0x0
	v_fma_f32 v69, -v124, v122, v69
	ds_bpermute_b32 v122, v116, v123
	s_wait_dscnt 0x0
	v_fma_f32 v70, -v124, v122, v70
	ds_bpermute_b32 v122, v117, v123
	s_wait_dscnt 0x0
	v_fma_f32 v71, -v124, v122, v71
	ds_bpermute_b32 v122, v118, v123
	s_wait_dscnt 0x0
	v_fma_f32 v72, -v124, v122, v72
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e32 v122, 0, v42, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v122, v122, v58, s0
	ds_bpermute_b32 v123, v103, v122
	s_wait_dscnt 0x0
	v_add_f32_e32 v124, v122, v123
	ds_load_2addr_b32 v[122:123], v125 offset0:64 offset1:96
	s_wait_dscnt 0x0
	ds_bpermute_b32 v126, v120, v122
	s_wait_dscnt 0x0
	v_fma_f32 v41, -v124, v126, v41
	ds_bpermute_b32 v126, v104, v122
	s_wait_dscnt 0x0
	v_fma_f32 v42, -v124, v126, v42
	ds_bpermute_b32 v126, v105, v122
	s_wait_dscnt 0x0
	v_fma_f32 v43, -v124, v126, v43
	ds_bpermute_b32 v126, v106, v122
	s_wait_dscnt 0x0
	v_fma_f32 v44, -v124, v126, v44
	ds_bpermute_b32 v126, v107, v122
	s_wait_dscnt 0x0
	v_fma_f32 v45, -v124, v126, v45
	ds_bpermute_b32 v126, v108, v122
	s_wait_dscnt 0x0
	v_fma_f32 v46, -v124, v126, v46
	ds_bpermute_b32 v126, v109, v122
	s_wait_dscnt 0x0
	v_fma_f32 v47, -v124, v126, v47
	ds_bpermute_b32 v126, v110, v122
	s_wait_dscnt 0x0
	v_fma_f32 v48, -v124, v126, v48
	ds_bpermute_b32 v126, v111, v122
	s_wait_dscnt 0x0
	v_fma_f32 v49, -v124, v126, v49
	ds_bpermute_b32 v126, v112, v122
	s_wait_dscnt 0x0
	v_fma_f32 v50, -v124, v126, v50
	ds_bpermute_b32 v126, v113, v122
	s_wait_dscnt 0x0
	v_fma_f32 v51, -v124, v126, v51
	ds_bpermute_b32 v126, v114, v122
	s_wait_dscnt 0x0
	v_fma_f32 v52, -v124, v126, v52
	ds_bpermute_b32 v126, v115, v122
	s_wait_dscnt 0x0
	v_fma_f32 v53, -v124, v126, v53
	ds_bpermute_b32 v126, v116, v122
	s_wait_dscnt 0x0
	v_fma_f32 v54, -v124, v126, v54
	ds_bpermute_b32 v126, v117, v122
	ds_bpermute_b32 v122, v118, v122
	s_wait_dscnt 0x1
	v_fma_f32 v55, -v124, v126, v55
	s_wait_dscnt 0x0
	v_fma_f32 v56, -v124, v122, v56
	ds_bpermute_b32 v122, v120, v123
	s_wait_dscnt 0x0
	v_fma_f32 v57, -v124, v122, v57
	ds_bpermute_b32 v122, v104, v123
	s_wait_dscnt 0x0
	v_fma_f32 v58, -v124, v122, v58
	ds_bpermute_b32 v122, v105, v123
	s_wait_dscnt 0x0
	v_fma_f32 v59, -v124, v122, v59
	ds_bpermute_b32 v122, v106, v123
	s_wait_dscnt 0x0
	v_fma_f32 v60, -v124, v122, v60
	ds_bpermute_b32 v122, v107, v123
	s_wait_dscnt 0x0
	v_fma_f32 v61, -v124, v122, v61
	ds_bpermute_b32 v122, v108, v123
	s_wait_dscnt 0x0
	v_fma_f32 v62, -v124, v122, v62
	ds_bpermute_b32 v122, v109, v123
	s_wait_dscnt 0x0
	v_fma_f32 v63, -v124, v122, v63
	ds_bpermute_b32 v122, v110, v123
	s_wait_dscnt 0x0
	v_fma_f32 v64, -v124, v122, v64
	ds_bpermute_b32 v122, v111, v123
	s_wait_dscnt 0x0
	v_fma_f32 v65, -v124, v122, v65
	ds_bpermute_b32 v122, v112, v123
	s_wait_dscnt 0x0
	v_fma_f32 v66, -v124, v122, v66
	ds_bpermute_b32 v122, v113, v123
	s_wait_dscnt 0x0
	v_fma_f32 v67, -v124, v122, v67
	ds_bpermute_b32 v122, v114, v123
	s_wait_dscnt 0x0
	v_fma_f32 v68, -v124, v122, v68
	ds_bpermute_b32 v122, v115, v123
	s_wait_dscnt 0x0
	v_fma_f32 v69, -v124, v122, v69
	ds_bpermute_b32 v122, v116, v123
	s_wait_dscnt 0x0
	v_fma_f32 v70, -v124, v122, v70
	ds_bpermute_b32 v122, v117, v123
	s_wait_dscnt 0x0
	v_fma_f32 v71, -v124, v122, v71
	ds_bpermute_b32 v122, v118, v123
	s_wait_dscnt 0x0
	v_fma_f32 v72, -v124, v122, v72
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e32 v122, 0, v43, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v122, v122, v59, s0
	ds_bpermute_b32 v123, v103, v122
	s_wait_dscnt 0x0
	v_add_f32_e32 v124, v122, v123
	ds_load_2addr_b32 v[122:123], v125 offset0:128 offset1:160
	s_wait_dscnt 0x0
	ds_bpermute_b32 v126, v120, v122
	s_wait_dscnt 0x0
	v_fma_f32 v41, -v124, v126, v41
	ds_bpermute_b32 v126, v104, v122
	s_wait_dscnt 0x0
	v_fma_f32 v42, -v124, v126, v42
	ds_bpermute_b32 v126, v105, v122
	s_wait_dscnt 0x0
	v_fma_f32 v43, -v124, v126, v43
	ds_bpermute_b32 v126, v106, v122
	s_wait_dscnt 0x0
	v_fma_f32 v44, -v124, v126, v44
	ds_bpermute_b32 v126, v107, v122
	s_wait_dscnt 0x0
	v_fma_f32 v45, -v124, v126, v45
	ds_bpermute_b32 v126, v108, v122
	s_wait_dscnt 0x0
	v_fma_f32 v46, -v124, v126, v46
	ds_bpermute_b32 v126, v109, v122
	s_wait_dscnt 0x0
	v_fma_f32 v47, -v124, v126, v47
	ds_bpermute_b32 v126, v110, v122
	s_wait_dscnt 0x0
	v_fma_f32 v48, -v124, v126, v48
	ds_bpermute_b32 v126, v111, v122
	s_wait_dscnt 0x0
	v_fma_f32 v49, -v124, v126, v49
	ds_bpermute_b32 v126, v112, v122
	s_wait_dscnt 0x0
	v_fma_f32 v50, -v124, v126, v50
	ds_bpermute_b32 v126, v113, v122
	s_wait_dscnt 0x0
	v_fma_f32 v51, -v124, v126, v51
	ds_bpermute_b32 v126, v114, v122
	s_wait_dscnt 0x0
	v_fma_f32 v52, -v124, v126, v52
	ds_bpermute_b32 v126, v115, v122
	s_wait_dscnt 0x0
	v_fma_f32 v53, -v124, v126, v53
	ds_bpermute_b32 v126, v116, v122
	s_wait_dscnt 0x0
	v_fma_f32 v54, -v124, v126, v54
	ds_bpermute_b32 v126, v117, v122
	ds_bpermute_b32 v122, v118, v122
	s_wait_dscnt 0x1
	v_fma_f32 v55, -v124, v126, v55
	s_wait_dscnt 0x0
	v_fma_f32 v56, -v124, v122, v56
	ds_bpermute_b32 v122, v120, v123
	s_wait_dscnt 0x0
	v_fma_f32 v57, -v124, v122, v57
	ds_bpermute_b32 v122, v104, v123
	s_wait_dscnt 0x0
	v_fma_f32 v58, -v124, v122, v58
	ds_bpermute_b32 v122, v105, v123
	s_wait_dscnt 0x0
	v_fma_f32 v59, -v124, v122, v59
	ds_bpermute_b32 v122, v106, v123
	s_wait_dscnt 0x0
	v_fma_f32 v60, -v124, v122, v60
	ds_bpermute_b32 v122, v107, v123
	s_wait_dscnt 0x0
	v_fma_f32 v61, -v124, v122, v61
	ds_bpermute_b32 v122, v108, v123
	s_wait_dscnt 0x0
	v_fma_f32 v62, -v124, v122, v62
	ds_bpermute_b32 v122, v109, v123
	s_wait_dscnt 0x0
	v_fma_f32 v63, -v124, v122, v63
	ds_bpermute_b32 v122, v110, v123
	s_wait_dscnt 0x0
	v_fma_f32 v64, -v124, v122, v64
	ds_bpermute_b32 v122, v111, v123
	s_wait_dscnt 0x0
	v_fma_f32 v65, -v124, v122, v65
	ds_bpermute_b32 v122, v112, v123
	s_wait_dscnt 0x0
	v_fma_f32 v66, -v124, v122, v66
	ds_bpermute_b32 v122, v113, v123
	s_wait_dscnt 0x0
	v_fma_f32 v67, -v124, v122, v67
	ds_bpermute_b32 v122, v114, v123
	s_wait_dscnt 0x0
	v_fma_f32 v68, -v124, v122, v68
	ds_bpermute_b32 v122, v115, v123
	s_wait_dscnt 0x0
	v_fma_f32 v69, -v124, v122, v69
	ds_bpermute_b32 v122, v116, v123
	s_wait_dscnt 0x0
	v_fma_f32 v70, -v124, v122, v70
	ds_bpermute_b32 v122, v117, v123
	s_wait_dscnt 0x0
	v_fma_f32 v71, -v124, v122, v71
	ds_bpermute_b32 v122, v118, v123
	s_wait_dscnt 0x0
	v_fma_f32 v72, -v124, v122, v72
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e32 v122, 0, v44, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v122, v122, v60, s0
	ds_bpermute_b32 v123, v103, v122
	s_wait_dscnt 0x0
	v_add_f32_e32 v124, v122, v123
	ds_load_2addr_b32 v[122:123], v125 offset0:192 offset1:224
	s_wait_dscnt 0x0
	ds_bpermute_b32 v125, v120, v122
	s_wait_dscnt 0x0
	v_fma_f32 v41, -v124, v125, v41
	ds_bpermute_b32 v125, v104, v122
	s_wait_dscnt 0x0
	v_fma_f32 v42, -v124, v125, v42
	ds_bpermute_b32 v125, v105, v122
	s_wait_dscnt 0x0
	v_fma_f32 v43, -v124, v125, v43
	ds_bpermute_b32 v125, v106, v122
	s_wait_dscnt 0x0
	v_fma_f32 v44, -v124, v125, v44
	ds_bpermute_b32 v125, v107, v122
	s_wait_dscnt 0x0
	v_fma_f32 v45, -v124, v125, v45
	ds_bpermute_b32 v125, v108, v122
	s_wait_dscnt 0x0
	v_fma_f32 v46, -v124, v125, v46
	ds_bpermute_b32 v125, v109, v122
	s_wait_dscnt 0x0
	v_fma_f32 v47, -v124, v125, v47
	ds_bpermute_b32 v125, v110, v122
	s_wait_dscnt 0x0
	v_fma_f32 v48, -v124, v125, v48
	ds_bpermute_b32 v125, v111, v122
	s_wait_dscnt 0x0
	v_fma_f32 v49, -v124, v125, v49
	ds_bpermute_b32 v125, v112, v122
	s_wait_dscnt 0x0
	v_fma_f32 v50, -v124, v125, v50
	ds_bpermute_b32 v125, v113, v122
	s_wait_dscnt 0x0
	v_fma_f32 v51, -v124, v125, v51
	ds_bpermute_b32 v125, v114, v122
	s_wait_dscnt 0x0
	v_fma_f32 v52, -v124, v125, v52
	ds_bpermute_b32 v125, v115, v122
	s_wait_dscnt 0x0
	v_fma_f32 v53, -v124, v125, v53
	ds_bpermute_b32 v125, v116, v122
	s_wait_dscnt 0x0
	v_fma_f32 v54, -v124, v125, v54
	ds_bpermute_b32 v125, v117, v122
	ds_bpermute_b32 v122, v118, v122
	s_wait_dscnt 0x1
	v_fma_f32 v55, -v124, v125, v55
	s_wait_dscnt 0x0
	v_fma_f32 v56, -v124, v122, v56
	ds_bpermute_b32 v122, v120, v123
	v_add_nc_u32_e32 v125, 0xc00, v121
	s_wait_dscnt 0x0
	v_fma_f32 v57, -v124, v122, v57
	ds_bpermute_b32 v122, v104, v123
	s_wait_dscnt 0x0
	v_fma_f32 v58, -v124, v122, v58
	ds_bpermute_b32 v122, v105, v123
	s_wait_dscnt 0x0
	v_fma_f32 v59, -v124, v122, v59
	ds_bpermute_b32 v122, v106, v123
	s_wait_dscnt 0x0
	v_fma_f32 v60, -v124, v122, v60
	ds_bpermute_b32 v122, v107, v123
	s_wait_dscnt 0x0
	v_fma_f32 v61, -v124, v122, v61
	ds_bpermute_b32 v122, v108, v123
	s_wait_dscnt 0x0
	v_fma_f32 v62, -v124, v122, v62
	ds_bpermute_b32 v122, v109, v123
	s_wait_dscnt 0x0
	v_fma_f32 v63, -v124, v122, v63
	ds_bpermute_b32 v122, v110, v123
	s_wait_dscnt 0x0
	v_fma_f32 v64, -v124, v122, v64
	ds_bpermute_b32 v122, v111, v123
	s_wait_dscnt 0x0
	v_fma_f32 v65, -v124, v122, v65
	ds_bpermute_b32 v122, v112, v123
	s_wait_dscnt 0x0
	v_fma_f32 v66, -v124, v122, v66
	ds_bpermute_b32 v122, v113, v123
	s_wait_dscnt 0x0
	v_fma_f32 v67, -v124, v122, v67
	ds_bpermute_b32 v122, v114, v123
	s_wait_dscnt 0x0
	v_fma_f32 v68, -v124, v122, v68
	ds_bpermute_b32 v122, v115, v123
	s_wait_dscnt 0x0
	v_fma_f32 v69, -v124, v122, v69
	ds_bpermute_b32 v122, v116, v123
	s_wait_dscnt 0x0
	v_fma_f32 v70, -v124, v122, v70
	ds_bpermute_b32 v122, v117, v123
	s_wait_dscnt 0x0
	v_fma_f32 v71, -v124, v122, v71
	ds_bpermute_b32 v122, v118, v123
	s_wait_dscnt 0x0
	v_fma_f32 v72, -v124, v122, v72
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e32 v122, 0, v45, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v122, v122, v61, s0
	ds_bpermute_b32 v123, v103, v122
	s_wait_dscnt 0x0
	v_add_f32_e32 v124, v122, v123
	ds_load_2addr_b32 v[122:123], v125 offset1:32
	s_wait_dscnt 0x0
	ds_bpermute_b32 v126, v120, v122
	s_wait_dscnt 0x0
	v_fma_f32 v41, -v124, v126, v41
	ds_bpermute_b32 v126, v104, v122
	s_wait_dscnt 0x0
	v_fma_f32 v42, -v124, v126, v42
	ds_bpermute_b32 v126, v105, v122
	s_wait_dscnt 0x0
	v_fma_f32 v43, -v124, v126, v43
	ds_bpermute_b32 v126, v106, v122
	s_wait_dscnt 0x0
	v_fma_f32 v44, -v124, v126, v44
	ds_bpermute_b32 v126, v107, v122
	s_wait_dscnt 0x0
	v_fma_f32 v45, -v124, v126, v45
	ds_bpermute_b32 v126, v108, v122
	s_wait_dscnt 0x0
	v_fma_f32 v46, -v124, v126, v46
	ds_bpermute_b32 v126, v109, v122
	s_wait_dscnt 0x0
	v_fma_f32 v47, -v124, v126, v47
	ds_bpermute_b32 v126, v110, v122
	s_wait_dscnt 0x0
	v_fma_f32 v48, -v124, v126, v48
	ds_bpermute_b32 v126, v111, v122
	s_wait_dscnt 0x0
	v_fma_f32 v49, -v124, v126, v49
	ds_bpermute_b32 v126, v112, v122
	s_wait_dscnt 0x0
	v_fma_f32 v50, -v124, v126, v50
	ds_bpermute_b32 v126, v113, v122
	s_wait_dscnt 0x0
	v_fma_f32 v51, -v124, v126, v51
	ds_bpermute_b32 v126, v114, v122
	s_wait_dscnt 0x0
	v_fma_f32 v52, -v124, v126, v52
	ds_bpermute_b32 v126, v115, v122
	s_wait_dscnt 0x0
	v_fma_f32 v53, -v124, v126, v53
	ds_bpermute_b32 v126, v116, v122
	s_wait_dscnt 0x0
	v_fma_f32 v54, -v124, v126, v54
	ds_bpermute_b32 v126, v117, v122
	ds_bpermute_b32 v122, v118, v122
	s_wait_dscnt 0x1
	v_fma_f32 v55, -v124, v126, v55
	s_wait_dscnt 0x0
	v_fma_f32 v56, -v124, v122, v56
	ds_bpermute_b32 v122, v120, v123
	s_wait_dscnt 0x0
	v_fma_f32 v57, -v124, v122, v57
	ds_bpermute_b32 v122, v104, v123
	s_wait_dscnt 0x0
	v_fma_f32 v58, -v124, v122, v58
	ds_bpermute_b32 v122, v105, v123
	s_wait_dscnt 0x0
	v_fma_f32 v59, -v124, v122, v59
	ds_bpermute_b32 v122, v106, v123
	s_wait_dscnt 0x0
	v_fma_f32 v60, -v124, v122, v60
	ds_bpermute_b32 v122, v107, v123
	s_wait_dscnt 0x0
	v_fma_f32 v61, -v124, v122, v61
	ds_bpermute_b32 v122, v108, v123
	s_wait_dscnt 0x0
	v_fma_f32 v62, -v124, v122, v62
	ds_bpermute_b32 v122, v109, v123
	s_wait_dscnt 0x0
	v_fma_f32 v63, -v124, v122, v63
	ds_bpermute_b32 v122, v110, v123
	s_wait_dscnt 0x0
	v_fma_f32 v64, -v124, v122, v64
	ds_bpermute_b32 v122, v111, v123
	s_wait_dscnt 0x0
	v_fma_f32 v65, -v124, v122, v65
	ds_bpermute_b32 v122, v112, v123
	s_wait_dscnt 0x0
	v_fma_f32 v66, -v124, v122, v66
	ds_bpermute_b32 v122, v113, v123
	s_wait_dscnt 0x0
	v_fma_f32 v67, -v124, v122, v67
	ds_bpermute_b32 v122, v114, v123
	s_wait_dscnt 0x0
	v_fma_f32 v68, -v124, v122, v68
	ds_bpermute_b32 v122, v115, v123
	s_wait_dscnt 0x0
	v_fma_f32 v69, -v124, v122, v69
	ds_bpermute_b32 v122, v116, v123
	s_wait_dscnt 0x0
	v_fma_f32 v70, -v124, v122, v70
	ds_bpermute_b32 v122, v117, v123
	s_wait_dscnt 0x0
	v_fma_f32 v71, -v124, v122, v71
	ds_bpermute_b32 v122, v118, v123
	s_wait_dscnt 0x0
	v_fma_f32 v72, -v124, v122, v72
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e32 v122, 0, v46, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v122, v122, v62, s0
	ds_bpermute_b32 v123, v103, v122
	s_wait_dscnt 0x0
	v_add_f32_e32 v124, v122, v123
	ds_load_2addr_b32 v[122:123], v125 offset0:64 offset1:96
	s_wait_dscnt 0x0
	ds_bpermute_b32 v126, v120, v122
	s_wait_dscnt 0x0
	v_fma_f32 v41, -v124, v126, v41
	ds_bpermute_b32 v126, v104, v122
	s_wait_dscnt 0x0
	v_fma_f32 v42, -v124, v126, v42
	ds_bpermute_b32 v126, v105, v122
	s_wait_dscnt 0x0
	v_fma_f32 v43, -v124, v126, v43
	ds_bpermute_b32 v126, v106, v122
	s_wait_dscnt 0x0
	v_fma_f32 v44, -v124, v126, v44
	ds_bpermute_b32 v126, v107, v122
	s_wait_dscnt 0x0
	v_fma_f32 v45, -v124, v126, v45
	ds_bpermute_b32 v126, v108, v122
	s_wait_dscnt 0x0
	v_fma_f32 v46, -v124, v126, v46
	ds_bpermute_b32 v126, v109, v122
	s_wait_dscnt 0x0
	v_fma_f32 v47, -v124, v126, v47
	ds_bpermute_b32 v126, v110, v122
	s_wait_dscnt 0x0
	v_fma_f32 v48, -v124, v126, v48
	ds_bpermute_b32 v126, v111, v122
	s_wait_dscnt 0x0
	v_fma_f32 v49, -v124, v126, v49
	ds_bpermute_b32 v126, v112, v122
	s_wait_dscnt 0x0
	v_fma_f32 v50, -v124, v126, v50
	ds_bpermute_b32 v126, v113, v122
	s_wait_dscnt 0x0
	v_fma_f32 v51, -v124, v126, v51
	ds_bpermute_b32 v126, v114, v122
	s_wait_dscnt 0x0
	v_fma_f32 v52, -v124, v126, v52
	ds_bpermute_b32 v126, v115, v122
	s_wait_dscnt 0x0
	v_fma_f32 v53, -v124, v126, v53
	ds_bpermute_b32 v126, v116, v122
	s_wait_dscnt 0x0
	v_fma_f32 v54, -v124, v126, v54
	ds_bpermute_b32 v126, v117, v122
	ds_bpermute_b32 v122, v118, v122
	s_wait_dscnt 0x1
	v_fma_f32 v55, -v124, v126, v55
	s_wait_dscnt 0x0
	v_fma_f32 v56, -v124, v122, v56
	ds_bpermute_b32 v122, v120, v123
	s_wait_dscnt 0x0
	v_fma_f32 v57, -v124, v122, v57
	ds_bpermute_b32 v122, v104, v123
	s_wait_dscnt 0x0
	v_fma_f32 v58, -v124, v122, v58
	ds_bpermute_b32 v122, v105, v123
	s_wait_dscnt 0x0
	v_fma_f32 v59, -v124, v122, v59
	ds_bpermute_b32 v122, v106, v123
	s_wait_dscnt 0x0
	v_fma_f32 v60, -v124, v122, v60
	ds_bpermute_b32 v122, v107, v123
	s_wait_dscnt 0x0
	v_fma_f32 v61, -v124, v122, v61
	ds_bpermute_b32 v122, v108, v123
	s_wait_dscnt 0x0
	v_fma_f32 v62, -v124, v122, v62
	ds_bpermute_b32 v122, v109, v123
	s_wait_dscnt 0x0
	v_fma_f32 v63, -v124, v122, v63
	ds_bpermute_b32 v122, v110, v123
	s_wait_dscnt 0x0
	v_fma_f32 v64, -v124, v122, v64
	ds_bpermute_b32 v122, v111, v123
	s_wait_dscnt 0x0
	v_fma_f32 v65, -v124, v122, v65
	ds_bpermute_b32 v122, v112, v123
	s_wait_dscnt 0x0
	v_fma_f32 v66, -v124, v122, v66
	ds_bpermute_b32 v122, v113, v123
	s_wait_dscnt 0x0
	v_fma_f32 v67, -v124, v122, v67
	ds_bpermute_b32 v122, v114, v123
	s_wait_dscnt 0x0
	v_fma_f32 v68, -v124, v122, v68
	ds_bpermute_b32 v122, v115, v123
	s_wait_dscnt 0x0
	v_fma_f32 v69, -v124, v122, v69
	ds_bpermute_b32 v122, v116, v123
	s_wait_dscnt 0x0
	v_fma_f32 v70, -v124, v122, v70
	ds_bpermute_b32 v122, v117, v123
	s_wait_dscnt 0x0
	v_fma_f32 v71, -v124, v122, v71
	ds_bpermute_b32 v122, v118, v123
	s_wait_dscnt 0x0
	v_fma_f32 v72, -v124, v122, v72
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e32 v122, 0, v47, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v122, v122, v63, s0
	ds_bpermute_b32 v123, v103, v122
	s_wait_dscnt 0x0
	v_add_f32_e32 v124, v122, v123
	ds_load_2addr_b32 v[122:123], v125 offset0:128 offset1:160
	s_wait_dscnt 0x0
	ds_bpermute_b32 v126, v120, v122
	s_wait_dscnt 0x0
	v_fma_f32 v41, -v124, v126, v41
	ds_bpermute_b32 v126, v104, v122
	s_wait_dscnt 0x0
	v_fma_f32 v42, -v124, v126, v42
	ds_bpermute_b32 v126, v105, v122
	s_wait_dscnt 0x0
	v_fma_f32 v43, -v124, v126, v43
	ds_bpermute_b32 v126, v106, v122
	s_wait_dscnt 0x0
	v_fma_f32 v44, -v124, v126, v44
	ds_bpermute_b32 v126, v107, v122
	s_wait_dscnt 0x0
	v_fma_f32 v45, -v124, v126, v45
	ds_bpermute_b32 v126, v108, v122
	s_wait_dscnt 0x0
	v_fma_f32 v46, -v124, v126, v46
	ds_bpermute_b32 v126, v109, v122
	s_wait_dscnt 0x0
	v_fma_f32 v47, -v124, v126, v47
	ds_bpermute_b32 v126, v110, v122
	s_wait_dscnt 0x0
	v_fma_f32 v48, -v124, v126, v48
	ds_bpermute_b32 v126, v111, v122
	s_wait_dscnt 0x0
	v_fma_f32 v49, -v124, v126, v49
	ds_bpermute_b32 v126, v112, v122
	s_wait_dscnt 0x0
	v_fma_f32 v50, -v124, v126, v50
	ds_bpermute_b32 v126, v113, v122
	s_wait_dscnt 0x0
	v_fma_f32 v51, -v124, v126, v51
	ds_bpermute_b32 v126, v114, v122
	s_wait_dscnt 0x0
	v_fma_f32 v52, -v124, v126, v52
	ds_bpermute_b32 v126, v115, v122
	s_wait_dscnt 0x0
	v_fma_f32 v53, -v124, v126, v53
	ds_bpermute_b32 v126, v116, v122
	s_wait_dscnt 0x0
	v_fma_f32 v54, -v124, v126, v54
	ds_bpermute_b32 v126, v117, v122
	ds_bpermute_b32 v122, v118, v122
	s_wait_dscnt 0x1
	v_fma_f32 v55, -v124, v126, v55
	s_wait_dscnt 0x0
	v_fma_f32 v56, -v124, v122, v56
	ds_bpermute_b32 v122, v120, v123
	s_wait_dscnt 0x0
	v_fma_f32 v57, -v124, v122, v57
	ds_bpermute_b32 v122, v104, v123
	s_wait_dscnt 0x0
	v_fma_f32 v58, -v124, v122, v58
	ds_bpermute_b32 v122, v105, v123
	s_wait_dscnt 0x0
	v_fma_f32 v59, -v124, v122, v59
	ds_bpermute_b32 v122, v106, v123
	s_wait_dscnt 0x0
	v_fma_f32 v60, -v124, v122, v60
	ds_bpermute_b32 v122, v107, v123
	s_wait_dscnt 0x0
	v_fma_f32 v61, -v124, v122, v61
	ds_bpermute_b32 v122, v108, v123
	s_wait_dscnt 0x0
	v_fma_f32 v62, -v124, v122, v62
	ds_bpermute_b32 v122, v109, v123
	s_wait_dscnt 0x0
	v_fma_f32 v63, -v124, v122, v63
	ds_bpermute_b32 v122, v110, v123
	s_wait_dscnt 0x0
	v_fma_f32 v64, -v124, v122, v64
	ds_bpermute_b32 v122, v111, v123
	s_wait_dscnt 0x0
	v_fma_f32 v65, -v124, v122, v65
	ds_bpermute_b32 v122, v112, v123
	s_wait_dscnt 0x0
	v_fma_f32 v66, -v124, v122, v66
	ds_bpermute_b32 v122, v113, v123
	s_wait_dscnt 0x0
	v_fma_f32 v67, -v124, v122, v67
	ds_bpermute_b32 v122, v114, v123
	s_wait_dscnt 0x0
	v_fma_f32 v68, -v124, v122, v68
	ds_bpermute_b32 v122, v115, v123
	s_wait_dscnt 0x0
	v_fma_f32 v69, -v124, v122, v69
	ds_bpermute_b32 v122, v116, v123
	s_wait_dscnt 0x0
	v_fma_f32 v70, -v124, v122, v70
	ds_bpermute_b32 v122, v117, v123
	s_wait_dscnt 0x0
	v_fma_f32 v71, -v124, v122, v71
	ds_bpermute_b32 v122, v118, v123
	s_wait_dscnt 0x0
	v_fma_f32 v72, -v124, v122, v72
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e32 v122, 0, v48, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v122, v122, v64, s0
	ds_bpermute_b32 v123, v103, v122
	s_wait_dscnt 0x0
	v_add_f32_e32 v124, v122, v123
	ds_load_2addr_b32 v[122:123], v125 offset0:192 offset1:224
	s_wait_dscnt 0x0
	ds_bpermute_b32 v125, v120, v122
	s_wait_dscnt 0x0
	v_fma_f32 v41, -v124, v125, v41
	ds_bpermute_b32 v125, v104, v122
	s_wait_dscnt 0x0
	v_fma_f32 v42, -v124, v125, v42
	ds_bpermute_b32 v125, v105, v122
	s_wait_dscnt 0x0
	v_fma_f32 v43, -v124, v125, v43
	ds_bpermute_b32 v125, v106, v122
	s_wait_dscnt 0x0
	v_fma_f32 v44, -v124, v125, v44
	ds_bpermute_b32 v125, v107, v122
	s_wait_dscnt 0x0
	v_fma_f32 v45, -v124, v125, v45
	ds_bpermute_b32 v125, v108, v122
	s_wait_dscnt 0x0
	v_fma_f32 v46, -v124, v125, v46
	ds_bpermute_b32 v125, v109, v122
	s_wait_dscnt 0x0
	v_fma_f32 v47, -v124, v125, v47
	ds_bpermute_b32 v125, v110, v122
	s_wait_dscnt 0x0
	v_fma_f32 v48, -v124, v125, v48
	ds_bpermute_b32 v125, v111, v122
	s_wait_dscnt 0x0
	v_fma_f32 v49, -v124, v125, v49
	ds_bpermute_b32 v125, v112, v122
	s_wait_dscnt 0x0
	v_fma_f32 v50, -v124, v125, v50
	ds_bpermute_b32 v125, v113, v122
	s_wait_dscnt 0x0
	v_fma_f32 v51, -v124, v125, v51
	ds_bpermute_b32 v125, v114, v122
	s_wait_dscnt 0x0
	v_fma_f32 v52, -v124, v125, v52
	ds_bpermute_b32 v125, v115, v122
	s_wait_dscnt 0x0
	v_fma_f32 v53, -v124, v125, v53
	ds_bpermute_b32 v125, v116, v122
	s_wait_dscnt 0x0
	v_fma_f32 v54, -v124, v125, v54
	ds_bpermute_b32 v125, v117, v122
	ds_bpermute_b32 v122, v118, v122
	s_wait_dscnt 0x1
	v_fma_f32 v55, -v124, v125, v55
	s_wait_dscnt 0x0
	v_fma_f32 v56, -v124, v122, v56
	ds_bpermute_b32 v122, v120, v123
	v_add_nc_u32_e32 v125, 0x1000, v121
	s_wait_dscnt 0x0
	v_fma_f32 v57, -v124, v122, v57
	ds_bpermute_b32 v122, v104, v123
	s_wait_dscnt 0x0
	v_fma_f32 v58, -v124, v122, v58
	ds_bpermute_b32 v122, v105, v123
	s_wait_dscnt 0x0
	v_fma_f32 v59, -v124, v122, v59
	ds_bpermute_b32 v122, v106, v123
	s_wait_dscnt 0x0
	v_fma_f32 v60, -v124, v122, v60
	ds_bpermute_b32 v122, v107, v123
	s_wait_dscnt 0x0
	v_fma_f32 v61, -v124, v122, v61
	ds_bpermute_b32 v122, v108, v123
	s_wait_dscnt 0x0
	v_fma_f32 v62, -v124, v122, v62
	ds_bpermute_b32 v122, v109, v123
	s_wait_dscnt 0x0
	v_fma_f32 v63, -v124, v122, v63
	ds_bpermute_b32 v122, v110, v123
	s_wait_dscnt 0x0
	v_fma_f32 v64, -v124, v122, v64
	ds_bpermute_b32 v122, v111, v123
	s_wait_dscnt 0x0
	v_fma_f32 v65, -v124, v122, v65
	ds_bpermute_b32 v122, v112, v123
	s_wait_dscnt 0x0
	v_fma_f32 v66, -v124, v122, v66
	ds_bpermute_b32 v122, v113, v123
	s_wait_dscnt 0x0
	v_fma_f32 v67, -v124, v122, v67
	ds_bpermute_b32 v122, v114, v123
	s_wait_dscnt 0x0
	v_fma_f32 v68, -v124, v122, v68
	ds_bpermute_b32 v122, v115, v123
	s_wait_dscnt 0x0
	v_fma_f32 v69, -v124, v122, v69
	ds_bpermute_b32 v122, v116, v123
	s_wait_dscnt 0x0
	v_fma_f32 v70, -v124, v122, v70
	ds_bpermute_b32 v122, v117, v123
	s_wait_dscnt 0x0
	v_fma_f32 v71, -v124, v122, v71
	ds_bpermute_b32 v122, v118, v123
	s_wait_dscnt 0x0
	v_fma_f32 v72, -v124, v122, v72
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v122, 0, v49, s1
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v122, v122, v65, s2
	ds_bpermute_b32 v123, v103, v122
	s_wait_dscnt 0x0
	v_add_f32_e32 v124, v122, v123
	ds_load_2addr_b32 v[122:123], v125 offset1:32
	s_wait_dscnt 0x0
	ds_bpermute_b32 v126, v120, v122
	s_wait_dscnt 0x0
	v_fma_f32 v41, -v124, v126, v41
	ds_bpermute_b32 v126, v104, v122
	s_wait_dscnt 0x0
	v_fma_f32 v42, -v124, v126, v42
	ds_bpermute_b32 v126, v105, v122
	s_wait_dscnt 0x0
	v_fma_f32 v43, -v124, v126, v43
	ds_bpermute_b32 v126, v106, v122
	s_wait_dscnt 0x0
	v_fma_f32 v44, -v124, v126, v44
	ds_bpermute_b32 v126, v107, v122
	s_wait_dscnt 0x0
	v_fma_f32 v45, -v124, v126, v45
	ds_bpermute_b32 v126, v108, v122
	s_wait_dscnt 0x0
	v_fma_f32 v46, -v124, v126, v46
	ds_bpermute_b32 v126, v109, v122
	s_wait_dscnt 0x0
	v_fma_f32 v47, -v124, v126, v47
	ds_bpermute_b32 v126, v110, v122
	s_wait_dscnt 0x0
	v_fma_f32 v48, -v124, v126, v48
	ds_bpermute_b32 v126, v111, v122
	s_wait_dscnt 0x0
	v_fma_f32 v49, -v124, v126, v49
	ds_bpermute_b32 v126, v112, v122
	s_wait_dscnt 0x0
	v_fma_f32 v50, -v124, v126, v50
	ds_bpermute_b32 v126, v113, v122
	s_wait_dscnt 0x0
	v_fma_f32 v51, -v124, v126, v51
	ds_bpermute_b32 v126, v114, v122
	s_wait_dscnt 0x0
	v_fma_f32 v52, -v124, v126, v52
	ds_bpermute_b32 v126, v115, v122
	s_wait_dscnt 0x0
	v_fma_f32 v53, -v124, v126, v53
	ds_bpermute_b32 v126, v116, v122
	s_wait_dscnt 0x0
	v_fma_f32 v54, -v124, v126, v54
	ds_bpermute_b32 v126, v117, v122
	ds_bpermute_b32 v122, v118, v122
	s_wait_dscnt 0x1
	v_fma_f32 v55, -v124, v126, v55
	s_wait_dscnt 0x0
	v_fma_f32 v56, -v124, v122, v56
	ds_bpermute_b32 v122, v120, v123
	s_wait_dscnt 0x0
	v_fma_f32 v57, -v124, v122, v57
	ds_bpermute_b32 v122, v104, v123
	s_wait_dscnt 0x0
	v_fma_f32 v58, -v124, v122, v58
	ds_bpermute_b32 v122, v105, v123
	s_wait_dscnt 0x0
	v_fma_f32 v59, -v124, v122, v59
	ds_bpermute_b32 v122, v106, v123
	s_wait_dscnt 0x0
	v_fma_f32 v60, -v124, v122, v60
	ds_bpermute_b32 v122, v107, v123
	s_wait_dscnt 0x0
	v_fma_f32 v61, -v124, v122, v61
	ds_bpermute_b32 v122, v108, v123
	s_wait_dscnt 0x0
	v_fma_f32 v62, -v124, v122, v62
	ds_bpermute_b32 v122, v109, v123
	s_wait_dscnt 0x0
	v_fma_f32 v63, -v124, v122, v63
	ds_bpermute_b32 v122, v110, v123
	s_wait_dscnt 0x0
	v_fma_f32 v64, -v124, v122, v64
	ds_bpermute_b32 v122, v111, v123
	s_wait_dscnt 0x0
	v_fma_f32 v65, -v124, v122, v65
	ds_bpermute_b32 v122, v112, v123
	s_wait_dscnt 0x0
	v_fma_f32 v66, -v124, v122, v66
	ds_bpermute_b32 v122, v113, v123
	s_wait_dscnt 0x0
	v_fma_f32 v67, -v124, v122, v67
	ds_bpermute_b32 v122, v114, v123
	s_wait_dscnt 0x0
	v_fma_f32 v68, -v124, v122, v68
	ds_bpermute_b32 v122, v115, v123
	s_wait_dscnt 0x0
	v_fma_f32 v69, -v124, v122, v69
	ds_bpermute_b32 v122, v116, v123
	s_wait_dscnt 0x0
	v_fma_f32 v70, -v124, v122, v70
	ds_bpermute_b32 v122, v117, v123
	s_wait_dscnt 0x0
	v_fma_f32 v71, -v124, v122, v71
	ds_bpermute_b32 v122, v118, v123
	s_wait_dscnt 0x0
	v_fma_f32 v72, -v124, v122, v72
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v122, 0, v50, s1
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v122, v122, v66, s2
	ds_bpermute_b32 v123, v103, v122
	s_wait_dscnt 0x0
	v_add_f32_e32 v124, v122, v123
	ds_load_2addr_b32 v[122:123], v125 offset0:64 offset1:96
	s_wait_dscnt 0x0
	ds_bpermute_b32 v126, v120, v122
	s_wait_dscnt 0x0
	v_fma_f32 v41, -v124, v126, v41
	ds_bpermute_b32 v126, v104, v122
	s_wait_dscnt 0x0
	v_fma_f32 v42, -v124, v126, v42
	ds_bpermute_b32 v126, v105, v122
	s_wait_dscnt 0x0
	v_fma_f32 v43, -v124, v126, v43
	ds_bpermute_b32 v126, v106, v122
	s_wait_dscnt 0x0
	v_fma_f32 v44, -v124, v126, v44
	ds_bpermute_b32 v126, v107, v122
	s_wait_dscnt 0x0
	v_fma_f32 v45, -v124, v126, v45
	ds_bpermute_b32 v126, v108, v122
	s_wait_dscnt 0x0
	v_fma_f32 v46, -v124, v126, v46
	ds_bpermute_b32 v126, v109, v122
	s_wait_dscnt 0x0
	v_fma_f32 v47, -v124, v126, v47
	ds_bpermute_b32 v126, v110, v122
	s_wait_dscnt 0x0
	v_fma_f32 v48, -v124, v126, v48
	ds_bpermute_b32 v126, v111, v122
	s_wait_dscnt 0x0
	v_fma_f32 v49, -v124, v126, v49
	ds_bpermute_b32 v126, v112, v122
	s_wait_dscnt 0x0
	v_fma_f32 v50, -v124, v126, v50
	ds_bpermute_b32 v126, v113, v122
	s_wait_dscnt 0x0
	v_fma_f32 v51, -v124, v126, v51
	ds_bpermute_b32 v126, v114, v122
	s_wait_dscnt 0x0
	v_fma_f32 v52, -v124, v126, v52
	ds_bpermute_b32 v126, v115, v122
	s_wait_dscnt 0x0
	v_fma_f32 v53, -v124, v126, v53
	ds_bpermute_b32 v126, v116, v122
	s_wait_dscnt 0x0
	v_fma_f32 v54, -v124, v126, v54
	ds_bpermute_b32 v126, v117, v122
	ds_bpermute_b32 v122, v118, v122
	s_wait_dscnt 0x1
	v_fma_f32 v55, -v124, v126, v55
	s_wait_dscnt 0x0
	v_fma_f32 v56, -v124, v122, v56
	ds_bpermute_b32 v122, v120, v123
	s_wait_dscnt 0x0
	v_fma_f32 v57, -v124, v122, v57
	ds_bpermute_b32 v122, v104, v123
	s_wait_dscnt 0x0
	v_fma_f32 v58, -v124, v122, v58
	ds_bpermute_b32 v122, v105, v123
	s_wait_dscnt 0x0
	v_fma_f32 v59, -v124, v122, v59
	ds_bpermute_b32 v122, v106, v123
	s_wait_dscnt 0x0
	v_fma_f32 v60, -v124, v122, v60
	ds_bpermute_b32 v122, v107, v123
	s_wait_dscnt 0x0
	v_fma_f32 v61, -v124, v122, v61
	ds_bpermute_b32 v122, v108, v123
	s_wait_dscnt 0x0
	v_fma_f32 v62, -v124, v122, v62
	ds_bpermute_b32 v122, v109, v123
	s_wait_dscnt 0x0
	v_fma_f32 v63, -v124, v122, v63
	ds_bpermute_b32 v122, v110, v123
	s_wait_dscnt 0x0
	v_fma_f32 v64, -v124, v122, v64
	ds_bpermute_b32 v122, v111, v123
	s_wait_dscnt 0x0
	v_fma_f32 v65, -v124, v122, v65
	ds_bpermute_b32 v122, v112, v123
	s_wait_dscnt 0x0
	v_fma_f32 v66, -v124, v122, v66
	ds_bpermute_b32 v122, v113, v123
	s_wait_dscnt 0x0
	v_fma_f32 v67, -v124, v122, v67
	ds_bpermute_b32 v122, v114, v123
	s_wait_dscnt 0x0
	v_fma_f32 v68, -v124, v122, v68
	ds_bpermute_b32 v122, v115, v123
	s_wait_dscnt 0x0
	v_fma_f32 v69, -v124, v122, v69
	ds_bpermute_b32 v122, v116, v123
	s_wait_dscnt 0x0
	v_fma_f32 v70, -v124, v122, v70
	ds_bpermute_b32 v122, v117, v123
	s_wait_dscnt 0x0
	v_fma_f32 v71, -v124, v122, v71
	ds_bpermute_b32 v122, v118, v123
	s_wait_dscnt 0x0
	v_fma_f32 v72, -v124, v122, v72
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v122, 0, v51, s1
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v122, v122, v67, s2
	ds_bpermute_b32 v123, v103, v122
	s_wait_dscnt 0x0
	v_add_f32_e32 v124, v122, v123
	ds_load_2addr_b32 v[122:123], v125 offset0:128 offset1:160
	s_wait_dscnt 0x0
	ds_bpermute_b32 v126, v120, v122
	s_wait_dscnt 0x0
	v_fma_f32 v41, -v124, v126, v41
	ds_bpermute_b32 v126, v104, v122
	s_wait_dscnt 0x0
	v_fma_f32 v42, -v124, v126, v42
	ds_bpermute_b32 v126, v105, v122
	s_wait_dscnt 0x0
	v_fma_f32 v43, -v124, v126, v43
	ds_bpermute_b32 v126, v106, v122
	s_wait_dscnt 0x0
	v_fma_f32 v44, -v124, v126, v44
	ds_bpermute_b32 v126, v107, v122
	s_wait_dscnt 0x0
	v_fma_f32 v45, -v124, v126, v45
	ds_bpermute_b32 v126, v108, v122
	s_wait_dscnt 0x0
	v_fma_f32 v46, -v124, v126, v46
	ds_bpermute_b32 v126, v109, v122
	s_wait_dscnt 0x0
	v_fma_f32 v47, -v124, v126, v47
	ds_bpermute_b32 v126, v110, v122
	s_wait_dscnt 0x0
	v_fma_f32 v48, -v124, v126, v48
	ds_bpermute_b32 v126, v111, v122
	s_wait_dscnt 0x0
	v_fma_f32 v49, -v124, v126, v49
	ds_bpermute_b32 v126, v112, v122
	s_wait_dscnt 0x0
	v_fma_f32 v50, -v124, v126, v50
	ds_bpermute_b32 v126, v113, v122
	s_wait_dscnt 0x0
	v_fma_f32 v51, -v124, v126, v51
	ds_bpermute_b32 v126, v114, v122
	s_wait_dscnt 0x0
	v_fma_f32 v52, -v124, v126, v52
	ds_bpermute_b32 v126, v115, v122
	s_wait_dscnt 0x0
	v_fma_f32 v53, -v124, v126, v53
	ds_bpermute_b32 v126, v116, v122
	s_wait_dscnt 0x0
	v_fma_f32 v54, -v124, v126, v54
	ds_bpermute_b32 v126, v117, v122
	ds_bpermute_b32 v122, v118, v122
	s_wait_dscnt 0x1
	v_fma_f32 v55, -v124, v126, v55
	s_wait_dscnt 0x0
	v_fma_f32 v56, -v124, v122, v56
	ds_bpermute_b32 v122, v120, v123
	s_wait_dscnt 0x0
	v_fma_f32 v57, -v124, v122, v57
	ds_bpermute_b32 v122, v104, v123
	s_wait_dscnt 0x0
	v_fma_f32 v58, -v124, v122, v58
	ds_bpermute_b32 v122, v105, v123
	s_wait_dscnt 0x0
	v_fma_f32 v59, -v124, v122, v59
	ds_bpermute_b32 v122, v106, v123
	s_wait_dscnt 0x0
	v_fma_f32 v60, -v124, v122, v60
	ds_bpermute_b32 v122, v107, v123
	s_wait_dscnt 0x0
	v_fma_f32 v61, -v124, v122, v61
	ds_bpermute_b32 v122, v108, v123
	s_wait_dscnt 0x0
	v_fma_f32 v62, -v124, v122, v62
	ds_bpermute_b32 v122, v109, v123
	s_wait_dscnt 0x0
	v_fma_f32 v63, -v124, v122, v63
	ds_bpermute_b32 v122, v110, v123
	s_wait_dscnt 0x0
	v_fma_f32 v64, -v124, v122, v64
	ds_bpermute_b32 v122, v111, v123
	s_wait_dscnt 0x0
	v_fma_f32 v65, -v124, v122, v65
	ds_bpermute_b32 v122, v112, v123
	s_wait_dscnt 0x0
	v_fma_f32 v66, -v124, v122, v66
	ds_bpermute_b32 v122, v113, v123
	s_wait_dscnt 0x0
	v_fma_f32 v67, -v124, v122, v67
	ds_bpermute_b32 v122, v114, v123
	s_wait_dscnt 0x0
	v_fma_f32 v68, -v124, v122, v68
	ds_bpermute_b32 v122, v115, v123
	s_wait_dscnt 0x0
	v_fma_f32 v69, -v124, v122, v69
	ds_bpermute_b32 v122, v116, v123
	s_wait_dscnt 0x0
	v_fma_f32 v70, -v124, v122, v70
	ds_bpermute_b32 v122, v117, v123
	s_wait_dscnt 0x0
	v_fma_f32 v71, -v124, v122, v71
	ds_bpermute_b32 v122, v118, v123
	s_wait_dscnt 0x0
	v_fma_f32 v72, -v124, v122, v72
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v122, 0, v52, s1
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v122, v122, v68, s2
	ds_bpermute_b32 v123, v103, v122
	s_wait_dscnt 0x0
	v_add_f32_e32 v124, v122, v123
	ds_load_2addr_b32 v[122:123], v125 offset0:192 offset1:224
	s_wait_dscnt 0x0
	ds_bpermute_b32 v125, v120, v122
	s_wait_dscnt 0x0
	v_fma_f32 v41, -v124, v125, v41
	ds_bpermute_b32 v125, v104, v122
	s_wait_dscnt 0x0
	v_fma_f32 v42, -v124, v125, v42
	ds_bpermute_b32 v125, v105, v122
	s_wait_dscnt 0x0
	v_fma_f32 v43, -v124, v125, v43
	ds_bpermute_b32 v125, v106, v122
	s_wait_dscnt 0x0
	v_fma_f32 v44, -v124, v125, v44
	ds_bpermute_b32 v125, v107, v122
	s_wait_dscnt 0x0
	v_fma_f32 v45, -v124, v125, v45
	ds_bpermute_b32 v125, v108, v122
	s_wait_dscnt 0x0
	v_fma_f32 v46, -v124, v125, v46
	ds_bpermute_b32 v125, v109, v122
	s_wait_dscnt 0x0
	v_fma_f32 v47, -v124, v125, v47
	ds_bpermute_b32 v125, v110, v122
	s_wait_dscnt 0x0
	v_fma_f32 v48, -v124, v125, v48
	ds_bpermute_b32 v125, v111, v122
	s_wait_dscnt 0x0
	v_fma_f32 v49, -v124, v125, v49
	ds_bpermute_b32 v125, v112, v122
	s_wait_dscnt 0x0
	v_fma_f32 v50, -v124, v125, v50
	ds_bpermute_b32 v125, v113, v122
	s_wait_dscnt 0x0
	v_fma_f32 v51, -v124, v125, v51
	ds_bpermute_b32 v125, v114, v122
	s_wait_dscnt 0x0
	v_fma_f32 v52, -v124, v125, v52
	ds_bpermute_b32 v125, v115, v122
	s_wait_dscnt 0x0
	v_fma_f32 v53, -v124, v125, v53
	ds_bpermute_b32 v125, v116, v122
	s_wait_dscnt 0x0
	v_fma_f32 v54, -v124, v125, v54
	ds_bpermute_b32 v125, v117, v122
	ds_bpermute_b32 v122, v118, v122
	s_wait_dscnt 0x1
	v_fma_f32 v55, -v124, v125, v55
	s_wait_dscnt 0x0
	v_fma_f32 v56, -v124, v122, v56
	ds_bpermute_b32 v122, v120, v123
	v_add_nc_u32_e32 v125, 0x1400, v121
	v_add_nc_u32_e32 v121, 0x1c00, v121
	s_wait_dscnt 0x0
	v_fma_f32 v57, -v124, v122, v57
	ds_bpermute_b32 v122, v104, v123
	s_wait_dscnt 0x0
	v_fma_f32 v58, -v124, v122, v58
	ds_bpermute_b32 v122, v105, v123
	s_wait_dscnt 0x0
	v_fma_f32 v59, -v124, v122, v59
	ds_bpermute_b32 v122, v106, v123
	s_wait_dscnt 0x0
	v_fma_f32 v60, -v124, v122, v60
	ds_bpermute_b32 v122, v107, v123
	s_wait_dscnt 0x0
	v_fma_f32 v61, -v124, v122, v61
	ds_bpermute_b32 v122, v108, v123
	s_wait_dscnt 0x0
	v_fma_f32 v62, -v124, v122, v62
	ds_bpermute_b32 v122, v109, v123
	s_wait_dscnt 0x0
	v_fma_f32 v63, -v124, v122, v63
	ds_bpermute_b32 v122, v110, v123
	s_wait_dscnt 0x0
	v_fma_f32 v64, -v124, v122, v64
	ds_bpermute_b32 v122, v111, v123
	s_wait_dscnt 0x0
	v_fma_f32 v65, -v124, v122, v65
	ds_bpermute_b32 v122, v112, v123
	s_wait_dscnt 0x0
	v_fma_f32 v66, -v124, v122, v66
	ds_bpermute_b32 v122, v113, v123
	s_wait_dscnt 0x0
	v_fma_f32 v67, -v124, v122, v67
	ds_bpermute_b32 v122, v114, v123
	s_wait_dscnt 0x0
	v_fma_f32 v68, -v124, v122, v68
	ds_bpermute_b32 v122, v115, v123
	s_wait_dscnt 0x0
	v_fma_f32 v69, -v124, v122, v69
	ds_bpermute_b32 v122, v116, v123
	s_wait_dscnt 0x0
	v_fma_f32 v70, -v124, v122, v70
	ds_bpermute_b32 v122, v117, v123
	s_wait_dscnt 0x0
	v_fma_f32 v71, -v124, v122, v71
	ds_bpermute_b32 v122, v118, v123
	s_wait_dscnt 0x0
	v_fma_f32 v72, -v124, v122, v72
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v122, 0, v53, s1
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v122, v122, v69, s2
	ds_bpermute_b32 v123, v103, v122
	s_wait_dscnt 0x0
	v_add_f32_e32 v124, v122, v123
	ds_load_2addr_b32 v[122:123], v125 offset1:32
	s_wait_dscnt 0x0
	ds_bpermute_b32 v126, v120, v122
	s_wait_dscnt 0x0
	v_fma_f32 v41, -v124, v126, v41
	ds_bpermute_b32 v126, v104, v122
	s_wait_dscnt 0x0
	v_fma_f32 v42, -v124, v126, v42
	ds_bpermute_b32 v126, v105, v122
	s_wait_dscnt 0x0
	v_fma_f32 v43, -v124, v126, v43
	ds_bpermute_b32 v126, v106, v122
	s_wait_dscnt 0x0
	v_fma_f32 v44, -v124, v126, v44
	ds_bpermute_b32 v126, v107, v122
	s_wait_dscnt 0x0
	v_fma_f32 v45, -v124, v126, v45
	ds_bpermute_b32 v126, v108, v122
	s_wait_dscnt 0x0
	v_fma_f32 v46, -v124, v126, v46
	ds_bpermute_b32 v126, v109, v122
	s_wait_dscnt 0x0
	v_fma_f32 v47, -v124, v126, v47
	ds_bpermute_b32 v126, v110, v122
	s_wait_dscnt 0x0
	v_fma_f32 v48, -v124, v126, v48
	ds_bpermute_b32 v126, v111, v122
	s_wait_dscnt 0x0
	v_fma_f32 v49, -v124, v126, v49
	ds_bpermute_b32 v126, v112, v122
	s_wait_dscnt 0x0
	v_fma_f32 v50, -v124, v126, v50
	ds_bpermute_b32 v126, v113, v122
	s_wait_dscnt 0x0
	v_fma_f32 v51, -v124, v126, v51
	ds_bpermute_b32 v126, v114, v122
	s_wait_dscnt 0x0
	v_fma_f32 v52, -v124, v126, v52
	ds_bpermute_b32 v126, v115, v122
	s_wait_dscnt 0x0
	v_fma_f32 v53, -v124, v126, v53
	ds_bpermute_b32 v126, v116, v122
	s_wait_dscnt 0x0
	v_fma_f32 v54, -v124, v126, v54
	ds_bpermute_b32 v126, v117, v122
	ds_bpermute_b32 v122, v118, v122
	s_wait_dscnt 0x1
	v_fma_f32 v55, -v124, v126, v55
	s_wait_dscnt 0x0
	v_fma_f32 v56, -v124, v122, v56
	ds_bpermute_b32 v122, v120, v123
	s_wait_dscnt 0x0
	v_fma_f32 v57, -v124, v122, v57
	ds_bpermute_b32 v122, v104, v123
	s_wait_dscnt 0x0
	v_fma_f32 v58, -v124, v122, v58
	ds_bpermute_b32 v122, v105, v123
	s_wait_dscnt 0x0
	v_fma_f32 v59, -v124, v122, v59
	ds_bpermute_b32 v122, v106, v123
	s_wait_dscnt 0x0
	v_fma_f32 v60, -v124, v122, v60
	ds_bpermute_b32 v122, v107, v123
	s_wait_dscnt 0x0
	v_fma_f32 v61, -v124, v122, v61
	ds_bpermute_b32 v122, v108, v123
	s_wait_dscnt 0x0
	v_fma_f32 v62, -v124, v122, v62
	ds_bpermute_b32 v122, v109, v123
	s_wait_dscnt 0x0
	v_fma_f32 v63, -v124, v122, v63
	ds_bpermute_b32 v122, v110, v123
	s_wait_dscnt 0x0
	v_fma_f32 v64, -v124, v122, v64
	ds_bpermute_b32 v122, v111, v123
	s_wait_dscnt 0x0
	v_fma_f32 v65, -v124, v122, v65
	ds_bpermute_b32 v122, v112, v123
	s_wait_dscnt 0x0
	v_fma_f32 v66, -v124, v122, v66
	ds_bpermute_b32 v122, v113, v123
	s_wait_dscnt 0x0
	v_fma_f32 v67, -v124, v122, v67
	ds_bpermute_b32 v122, v114, v123
	s_wait_dscnt 0x0
	v_fma_f32 v68, -v124, v122, v68
	ds_bpermute_b32 v122, v115, v123
	s_wait_dscnt 0x0
	v_fma_f32 v69, -v124, v122, v69
	ds_bpermute_b32 v122, v116, v123
	s_wait_dscnt 0x0
	v_fma_f32 v70, -v124, v122, v70
	ds_bpermute_b32 v122, v117, v123
	s_wait_dscnt 0x0
	v_fma_f32 v71, -v124, v122, v71
	ds_bpermute_b32 v122, v118, v123
	s_wait_dscnt 0x0
	v_fma_f32 v72, -v124, v122, v72
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v122, 0, v54, s1
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v122, v122, v70, s2
	ds_bpermute_b32 v123, v103, v122
	s_wait_dscnt 0x0
	v_add_f32_e32 v124, v122, v123
	ds_load_2addr_b32 v[122:123], v125 offset0:64 offset1:96
	s_wait_dscnt 0x0
	ds_bpermute_b32 v126, v120, v122
	s_wait_dscnt 0x0
	v_fma_f32 v41, -v124, v126, v41
	ds_bpermute_b32 v126, v104, v122
	s_wait_dscnt 0x0
	v_fma_f32 v42, -v124, v126, v42
	ds_bpermute_b32 v126, v105, v122
	s_wait_dscnt 0x0
	v_fma_f32 v43, -v124, v126, v43
	ds_bpermute_b32 v126, v106, v122
	s_wait_dscnt 0x0
	v_fma_f32 v44, -v124, v126, v44
	ds_bpermute_b32 v126, v107, v122
	s_wait_dscnt 0x0
	v_fma_f32 v45, -v124, v126, v45
	ds_bpermute_b32 v126, v108, v122
	s_wait_dscnt 0x0
	v_fma_f32 v46, -v124, v126, v46
	ds_bpermute_b32 v126, v109, v122
	s_wait_dscnt 0x0
	v_fma_f32 v47, -v124, v126, v47
	ds_bpermute_b32 v126, v110, v122
	s_wait_dscnt 0x0
	v_fma_f32 v48, -v124, v126, v48
	ds_bpermute_b32 v126, v111, v122
	s_wait_dscnt 0x0
	v_fma_f32 v49, -v124, v126, v49
	ds_bpermute_b32 v126, v112, v122
	s_wait_dscnt 0x0
	v_fma_f32 v50, -v124, v126, v50
	ds_bpermute_b32 v126, v113, v122
	s_wait_dscnt 0x0
	v_fma_f32 v51, -v124, v126, v51
	ds_bpermute_b32 v126, v114, v122
	s_wait_dscnt 0x0
	v_fma_f32 v52, -v124, v126, v52
	ds_bpermute_b32 v126, v115, v122
	s_wait_dscnt 0x0
	v_fma_f32 v53, -v124, v126, v53
	ds_bpermute_b32 v126, v116, v122
	s_wait_dscnt 0x0
	v_fma_f32 v54, -v124, v126, v54
	ds_bpermute_b32 v126, v117, v122
	ds_bpermute_b32 v122, v118, v122
	s_wait_dscnt 0x1
	v_fma_f32 v55, -v124, v126, v55
	s_wait_dscnt 0x0
	v_fma_f32 v56, -v124, v122, v56
	ds_bpermute_b32 v122, v120, v123
	s_wait_dscnt 0x0
	v_fma_f32 v57, -v124, v122, v57
	ds_bpermute_b32 v122, v104, v123
	s_wait_dscnt 0x0
	v_fma_f32 v58, -v124, v122, v58
	ds_bpermute_b32 v122, v105, v123
	s_wait_dscnt 0x0
	v_fma_f32 v59, -v124, v122, v59
	ds_bpermute_b32 v122, v106, v123
	s_wait_dscnt 0x0
	v_fma_f32 v60, -v124, v122, v60
	ds_bpermute_b32 v122, v107, v123
	s_wait_dscnt 0x0
	v_fma_f32 v61, -v124, v122, v61
	ds_bpermute_b32 v122, v108, v123
	s_wait_dscnt 0x0
	v_fma_f32 v62, -v124, v122, v62
	ds_bpermute_b32 v122, v109, v123
	s_wait_dscnt 0x0
	v_fma_f32 v63, -v124, v122, v63
	ds_bpermute_b32 v122, v110, v123
	s_wait_dscnt 0x0
	v_fma_f32 v64, -v124, v122, v64
	ds_bpermute_b32 v122, v111, v123
	s_wait_dscnt 0x0
	v_fma_f32 v65, -v124, v122, v65
	ds_bpermute_b32 v122, v112, v123
	s_wait_dscnt 0x0
	v_fma_f32 v66, -v124, v122, v66
	ds_bpermute_b32 v122, v113, v123
	s_wait_dscnt 0x0
	v_fma_f32 v67, -v124, v122, v67
	ds_bpermute_b32 v122, v114, v123
	s_wait_dscnt 0x0
	v_fma_f32 v68, -v124, v122, v68
	ds_bpermute_b32 v122, v115, v123
	s_wait_dscnt 0x0
	v_fma_f32 v69, -v124, v122, v69
	ds_bpermute_b32 v122, v116, v123
	s_wait_dscnt 0x0
	v_fma_f32 v70, -v124, v122, v70
	ds_bpermute_b32 v122, v117, v123
	s_wait_dscnt 0x0
	v_fma_f32 v71, -v124, v122, v71
	ds_bpermute_b32 v122, v118, v123
	s_wait_dscnt 0x0
	v_fma_f32 v72, -v124, v122, v72
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v122, 0, v55, s1
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v122, v122, v71, s2
	ds_bpermute_b32 v123, v103, v122
	s_wait_dscnt 0x0
	v_add_f32_e32 v124, v122, v123
	ds_load_2addr_b32 v[122:123], v125 offset0:128 offset1:160
	s_wait_dscnt 0x0
	ds_bpermute_b32 v126, v120, v122
	s_wait_dscnt 0x0
	v_fma_f32 v41, -v124, v126, v41
	ds_bpermute_b32 v126, v104, v122
	s_wait_dscnt 0x0
	v_fma_f32 v42, -v124, v126, v42
	ds_bpermute_b32 v126, v105, v122
	s_wait_dscnt 0x0
	v_fma_f32 v43, -v124, v126, v43
	ds_bpermute_b32 v126, v106, v122
	s_wait_dscnt 0x0
	v_fma_f32 v44, -v124, v126, v44
	ds_bpermute_b32 v126, v107, v122
	s_wait_dscnt 0x0
	v_fma_f32 v45, -v124, v126, v45
	ds_bpermute_b32 v126, v108, v122
	s_wait_dscnt 0x0
	v_fma_f32 v46, -v124, v126, v46
	ds_bpermute_b32 v126, v109, v122
	s_wait_dscnt 0x0
	v_fma_f32 v47, -v124, v126, v47
	ds_bpermute_b32 v126, v110, v122
	s_wait_dscnt 0x0
	v_fma_f32 v48, -v124, v126, v48
	ds_bpermute_b32 v126, v111, v122
	s_wait_dscnt 0x0
	v_fma_f32 v49, -v124, v126, v49
	ds_bpermute_b32 v126, v112, v122
	s_wait_dscnt 0x0
	v_fma_f32 v50, -v124, v126, v50
	ds_bpermute_b32 v126, v113, v122
	s_wait_dscnt 0x0
	v_fma_f32 v51, -v124, v126, v51
	ds_bpermute_b32 v126, v114, v122
	s_wait_dscnt 0x0
	v_fma_f32 v52, -v124, v126, v52
	ds_bpermute_b32 v126, v115, v122
	s_wait_dscnt 0x0
	v_fma_f32 v53, -v124, v126, v53
	ds_bpermute_b32 v126, v116, v122
	s_wait_dscnt 0x0
	v_fma_f32 v54, -v124, v126, v54
	ds_bpermute_b32 v126, v117, v122
	ds_bpermute_b32 v122, v118, v122
	s_wait_dscnt 0x1
	v_fma_f32 v55, -v124, v126, v55
	s_wait_dscnt 0x0
	v_fma_f32 v56, -v124, v122, v56
	ds_bpermute_b32 v122, v120, v123
	s_wait_dscnt 0x0
	v_fma_f32 v57, -v124, v122, v57
	ds_bpermute_b32 v122, v104, v123
	s_wait_dscnt 0x0
	v_fma_f32 v58, -v124, v122, v58
	ds_bpermute_b32 v122, v105, v123
	s_wait_dscnt 0x0
	v_fma_f32 v59, -v124, v122, v59
	ds_bpermute_b32 v122, v106, v123
	s_wait_dscnt 0x0
	v_fma_f32 v60, -v124, v122, v60
	ds_bpermute_b32 v122, v107, v123
	s_wait_dscnt 0x0
	v_fma_f32 v61, -v124, v122, v61
	ds_bpermute_b32 v122, v108, v123
	s_wait_dscnt 0x0
	v_fma_f32 v62, -v124, v122, v62
	ds_bpermute_b32 v122, v109, v123
	s_wait_dscnt 0x0
	v_fma_f32 v63, -v124, v122, v63
	ds_bpermute_b32 v122, v110, v123
	s_wait_dscnt 0x0
	v_fma_f32 v64, -v124, v122, v64
	ds_bpermute_b32 v122, v111, v123
	s_wait_dscnt 0x0
	v_fma_f32 v65, -v124, v122, v65
	ds_bpermute_b32 v122, v112, v123
	s_wait_dscnt 0x0
	v_fma_f32 v66, -v124, v122, v66
	ds_bpermute_b32 v122, v113, v123
	s_wait_dscnt 0x0
	v_fma_f32 v67, -v124, v122, v67
	ds_bpermute_b32 v122, v114, v123
	s_wait_dscnt 0x0
	v_fma_f32 v68, -v124, v122, v68
	ds_bpermute_b32 v122, v115, v123
	s_wait_dscnt 0x0
	v_fma_f32 v69, -v124, v122, v69
	ds_bpermute_b32 v122, v116, v123
	s_wait_dscnt 0x0
	v_fma_f32 v70, -v124, v122, v70
	ds_bpermute_b32 v122, v117, v123
	s_wait_dscnt 0x0
	v_fma_f32 v71, -v124, v122, v71
	ds_bpermute_b32 v122, v118, v123
	s_wait_dscnt 0x0
	v_fma_f32 v72, -v124, v122, v72
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v122, 0, v56, s1
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v122, v122, v72, s2
	ds_bpermute_b32 v123, v103, v122
	s_wait_dscnt 0x0
	v_add_f32_e32 v124, v122, v123
	ds_load_2addr_b32 v[122:123], v125 offset0:192 offset1:224
	s_wait_dscnt 0x0
	ds_bpermute_b32 v125, v120, v122
	s_wait_dscnt 0x0
	v_fma_f32 v41, -v124, v125, v41
	ds_bpermute_b32 v125, v104, v122
	s_wait_dscnt 0x0
	v_fma_f32 v42, -v124, v125, v42
	ds_bpermute_b32 v125, v105, v122
	s_wait_dscnt 0x0
	v_fma_f32 v43, -v124, v125, v43
	ds_bpermute_b32 v125, v106, v122
	s_wait_dscnt 0x0
	v_fma_f32 v44, -v124, v125, v44
	ds_bpermute_b32 v125, v107, v122
	s_wait_dscnt 0x0
	v_fma_f32 v45, -v124, v125, v45
	ds_bpermute_b32 v125, v108, v122
	s_wait_dscnt 0x0
	v_fma_f32 v46, -v124, v125, v46
	ds_bpermute_b32 v125, v109, v122
	s_wait_dscnt 0x0
	v_fma_f32 v47, -v124, v125, v47
	ds_bpermute_b32 v125, v110, v122
	s_wait_dscnt 0x0
	v_fma_f32 v48, -v124, v125, v48
	ds_bpermute_b32 v125, v111, v122
	s_wait_dscnt 0x0
	v_fma_f32 v49, -v124, v125, v49
	ds_bpermute_b32 v125, v112, v122
	s_wait_dscnt 0x0
	v_fma_f32 v50, -v124, v125, v50
	ds_bpermute_b32 v125, v113, v122
	s_wait_dscnt 0x0
	v_fma_f32 v51, -v124, v125, v51
	ds_bpermute_b32 v125, v114, v122
	s_wait_dscnt 0x0
	v_fma_f32 v52, -v124, v125, v52
	ds_bpermute_b32 v125, v115, v122
	s_wait_dscnt 0x0
	v_fma_f32 v53, -v124, v125, v53
	ds_bpermute_b32 v125, v116, v122
	s_wait_dscnt 0x0
	v_fma_f32 v54, -v124, v125, v54
	ds_bpermute_b32 v125, v117, v122
	ds_bpermute_b32 v122, v118, v122
	s_wait_dscnt 0x1
	v_fma_f32 v55, -v124, v125, v55
	s_wait_dscnt 0x0
	v_fma_f32 v56, -v124, v122, v56
	ds_bpermute_b32 v122, v120, v123
	s_wait_dscnt 0x0
	v_fma_f32 v57, -v124, v122, v57
	ds_bpermute_b32 v122, v104, v123
	s_wait_dscnt 0x0
	v_fma_f32 v58, -v124, v122, v58
	ds_bpermute_b32 v122, v105, v123
	s_wait_dscnt 0x0
	v_fma_f32 v59, -v124, v122, v59
	ds_bpermute_b32 v122, v106, v123
	s_wait_dscnt 0x0
	v_fma_f32 v60, -v124, v122, v60
	ds_bpermute_b32 v122, v107, v123
	s_wait_dscnt 0x0
	v_fma_f32 v61, -v124, v122, v61
	ds_bpermute_b32 v122, v108, v123
	s_wait_dscnt 0x0
	v_fma_f32 v62, -v124, v122, v62
	ds_bpermute_b32 v122, v109, v123
	s_wait_dscnt 0x0
	v_fma_f32 v63, -v124, v122, v63
	ds_bpermute_b32 v122, v110, v123
	s_wait_dscnt 0x0
	v_fma_f32 v64, -v124, v122, v64
	ds_bpermute_b32 v122, v111, v123
	s_wait_dscnt 0x0
	v_fma_f32 v65, -v124, v122, v65
	ds_bpermute_b32 v122, v112, v123
	s_wait_dscnt 0x0
	v_fma_f32 v66, -v124, v122, v66
	ds_bpermute_b32 v122, v113, v123
	s_wait_dscnt 0x0
	v_fma_f32 v67, -v124, v122, v67
	ds_bpermute_b32 v122, v114, v123
	s_wait_dscnt 0x0
	v_fma_f32 v68, -v124, v122, v68
	ds_bpermute_b32 v122, v115, v123
	s_wait_dscnt 0x0
	v_fma_f32 v69, -v124, v122, v69
	ds_bpermute_b32 v122, v116, v123
	s_wait_dscnt 0x0
	v_fma_f32 v70, -v124, v122, v70
	ds_bpermute_b32 v122, v117, v123
	s_wait_dscnt 0x0
	v_fma_f32 v71, -v124, v122, v71
	ds_bpermute_b32 v122, v118, v123
	s_wait_dscnt 0x0
	v_fma_f32 v72, -v124, v122, v72
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e32 v122, 0, v49, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v122, v122, v65, s0
	ds_bpermute_b32 v123, v103, v122
	s_wait_dscnt 0x0
	v_add_f32_e32 v124, v122, v123
	ds_load_2addr_b32 v[122:123], v138 offset1:32
	s_wait_dscnt 0x0
	ds_bpermute_b32 v125, v120, v122
	s_wait_dscnt 0x0
	v_fma_f32 v41, -v124, v125, v41
	ds_bpermute_b32 v125, v104, v122
	s_wait_dscnt 0x0
	v_fma_f32 v42, -v124, v125, v42
	ds_bpermute_b32 v125, v105, v122
	s_wait_dscnt 0x0
	v_fma_f32 v43, -v124, v125, v43
	ds_bpermute_b32 v125, v106, v122
	s_wait_dscnt 0x0
	v_fma_f32 v44, -v124, v125, v44
	ds_bpermute_b32 v125, v107, v122
	s_wait_dscnt 0x0
	v_fma_f32 v45, -v124, v125, v45
	ds_bpermute_b32 v125, v108, v122
	s_wait_dscnt 0x0
	v_fma_f32 v46, -v124, v125, v46
	ds_bpermute_b32 v125, v109, v122
	s_wait_dscnt 0x0
	v_fma_f32 v47, -v124, v125, v47
	ds_bpermute_b32 v125, v110, v122
	s_wait_dscnt 0x0
	v_fma_f32 v48, -v124, v125, v48
	ds_bpermute_b32 v125, v111, v122
	s_wait_dscnt 0x0
	v_fma_f32 v49, -v124, v125, v49
	ds_bpermute_b32 v125, v112, v122
	s_wait_dscnt 0x0
	v_fma_f32 v50, -v124, v125, v50
	ds_bpermute_b32 v125, v113, v122
	s_wait_dscnt 0x0
	v_fma_f32 v51, -v124, v125, v51
	ds_bpermute_b32 v125, v114, v122
	s_wait_dscnt 0x0
	v_fma_f32 v52, -v124, v125, v52
	ds_bpermute_b32 v125, v115, v122
	s_wait_dscnt 0x0
	v_fma_f32 v53, -v124, v125, v53
	ds_bpermute_b32 v125, v116, v122
	s_wait_dscnt 0x0
	v_fma_f32 v54, -v124, v125, v54
	ds_bpermute_b32 v125, v117, v122
	ds_bpermute_b32 v122, v118, v122
	s_wait_dscnt 0x1
	v_fma_f32 v55, -v124, v125, v55
	s_wait_dscnt 0x0
	v_fma_f32 v56, -v124, v122, v56
	ds_bpermute_b32 v122, v120, v123
	s_wait_dscnt 0x0
	v_fma_f32 v57, -v124, v122, v57
	ds_bpermute_b32 v122, v104, v123
	s_wait_dscnt 0x0
	v_fma_f32 v58, -v124, v122, v58
	ds_bpermute_b32 v122, v105, v123
	s_wait_dscnt 0x0
	v_fma_f32 v59, -v124, v122, v59
	ds_bpermute_b32 v122, v106, v123
	s_wait_dscnt 0x0
	v_fma_f32 v60, -v124, v122, v60
	ds_bpermute_b32 v122, v107, v123
	s_wait_dscnt 0x0
	v_fma_f32 v61, -v124, v122, v61
	ds_bpermute_b32 v122, v108, v123
	s_wait_dscnt 0x0
	v_fma_f32 v62, -v124, v122, v62
	ds_bpermute_b32 v122, v109, v123
	s_wait_dscnt 0x0
	v_fma_f32 v63, -v124, v122, v63
	ds_bpermute_b32 v122, v110, v123
	s_wait_dscnt 0x0
	v_fma_f32 v64, -v124, v122, v64
	ds_bpermute_b32 v122, v111, v123
	s_wait_dscnt 0x0
	v_fma_f32 v65, -v124, v122, v65
	ds_bpermute_b32 v122, v112, v123
	s_wait_dscnt 0x0
	v_fma_f32 v66, -v124, v122, v66
	ds_bpermute_b32 v122, v113, v123
	s_wait_dscnt 0x0
	v_fma_f32 v67, -v124, v122, v67
	ds_bpermute_b32 v122, v114, v123
	s_wait_dscnt 0x0
	v_fma_f32 v68, -v124, v122, v68
	ds_bpermute_b32 v122, v115, v123
	s_wait_dscnt 0x0
	v_fma_f32 v69, -v124, v122, v69
	ds_bpermute_b32 v122, v116, v123
	s_wait_dscnt 0x0
	v_fma_f32 v70, -v124, v122, v70
	ds_bpermute_b32 v122, v117, v123
	s_wait_dscnt 0x0
	v_fma_f32 v71, -v124, v122, v71
	ds_bpermute_b32 v122, v118, v123
	s_wait_dscnt 0x0
	v_fma_f32 v72, -v124, v122, v72
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e32 v122, 0, v50, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v122, v122, v66, s0
	ds_bpermute_b32 v123, v103, v122
	s_wait_dscnt 0x0
	v_add_f32_e32 v124, v122, v123
	ds_load_2addr_b32 v[122:123], v138 offset0:64 offset1:96
	s_wait_dscnt 0x0
	ds_bpermute_b32 v125, v120, v122
	s_wait_dscnt 0x0
	v_fma_f32 v41, -v124, v125, v41
	ds_bpermute_b32 v125, v104, v122
	s_wait_dscnt 0x0
	v_fma_f32 v42, -v124, v125, v42
	ds_bpermute_b32 v125, v105, v122
	s_wait_dscnt 0x0
	v_fma_f32 v43, -v124, v125, v43
	ds_bpermute_b32 v125, v106, v122
	s_wait_dscnt 0x0
	v_fma_f32 v44, -v124, v125, v44
	ds_bpermute_b32 v125, v107, v122
	s_wait_dscnt 0x0
	v_fma_f32 v45, -v124, v125, v45
	ds_bpermute_b32 v125, v108, v122
	s_wait_dscnt 0x0
	v_fma_f32 v46, -v124, v125, v46
	ds_bpermute_b32 v125, v109, v122
	s_wait_dscnt 0x0
	v_fma_f32 v47, -v124, v125, v47
	ds_bpermute_b32 v125, v110, v122
	s_wait_dscnt 0x0
	v_fma_f32 v48, -v124, v125, v48
	ds_bpermute_b32 v125, v111, v122
	s_wait_dscnt 0x0
	v_fma_f32 v49, -v124, v125, v49
	ds_bpermute_b32 v125, v112, v122
	s_wait_dscnt 0x0
	v_fma_f32 v50, -v124, v125, v50
	ds_bpermute_b32 v125, v113, v122
	s_wait_dscnt 0x0
	v_fma_f32 v51, -v124, v125, v51
	ds_bpermute_b32 v125, v114, v122
	s_wait_dscnt 0x0
	v_fma_f32 v52, -v124, v125, v52
	ds_bpermute_b32 v125, v115, v122
	s_wait_dscnt 0x0
	v_fma_f32 v53, -v124, v125, v53
	ds_bpermute_b32 v125, v116, v122
	s_wait_dscnt 0x0
	v_fma_f32 v54, -v124, v125, v54
	ds_bpermute_b32 v125, v117, v122
	ds_bpermute_b32 v122, v118, v122
	s_wait_dscnt 0x1
	v_fma_f32 v55, -v124, v125, v55
	s_wait_dscnt 0x0
	v_fma_f32 v56, -v124, v122, v56
	ds_bpermute_b32 v122, v120, v123
	s_wait_dscnt 0x0
	v_fma_f32 v57, -v124, v122, v57
	ds_bpermute_b32 v122, v104, v123
	s_wait_dscnt 0x0
	v_fma_f32 v58, -v124, v122, v58
	ds_bpermute_b32 v122, v105, v123
	s_wait_dscnt 0x0
	v_fma_f32 v59, -v124, v122, v59
	ds_bpermute_b32 v122, v106, v123
	s_wait_dscnt 0x0
	v_fma_f32 v60, -v124, v122, v60
	ds_bpermute_b32 v122, v107, v123
	s_wait_dscnt 0x0
	v_fma_f32 v61, -v124, v122, v61
	ds_bpermute_b32 v122, v108, v123
	s_wait_dscnt 0x0
	v_fma_f32 v62, -v124, v122, v62
	ds_bpermute_b32 v122, v109, v123
	s_wait_dscnt 0x0
	v_fma_f32 v63, -v124, v122, v63
	ds_bpermute_b32 v122, v110, v123
	s_wait_dscnt 0x0
	v_fma_f32 v64, -v124, v122, v64
	ds_bpermute_b32 v122, v111, v123
	s_wait_dscnt 0x0
	v_fma_f32 v65, -v124, v122, v65
	ds_bpermute_b32 v122, v112, v123
	s_wait_dscnt 0x0
	v_fma_f32 v66, -v124, v122, v66
	ds_bpermute_b32 v122, v113, v123
	s_wait_dscnt 0x0
	v_fma_f32 v67, -v124, v122, v67
	ds_bpermute_b32 v122, v114, v123
	s_wait_dscnt 0x0
	v_fma_f32 v68, -v124, v122, v68
	ds_bpermute_b32 v122, v115, v123
	s_wait_dscnt 0x0
	v_fma_f32 v69, -v124, v122, v69
	ds_bpermute_b32 v122, v116, v123
	s_wait_dscnt 0x0
	v_fma_f32 v70, -v124, v122, v70
	ds_bpermute_b32 v122, v117, v123
	s_wait_dscnt 0x0
	v_fma_f32 v71, -v124, v122, v71
	ds_bpermute_b32 v122, v118, v123
	s_wait_dscnt 0x0
	v_fma_f32 v72, -v124, v122, v72
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e32 v122, 0, v51, vcc_lo
	ds_load_2addr_b32 v[136:137], v138 offset0:128 offset1:160
	v_cndmask_b32_e64 v122, v122, v67, s0
	ds_bpermute_b32 v123, v103, v122
	s_wait_dscnt 0x0
	v_add_f32_e32 v139, v122, v123
	ds_bpermute_b32 v122, v120, v136
	s_wait_dscnt 0x0
	v_fma_f32 v41, -v139, v122, v41
	ds_bpermute_b32 v122, v104, v136
	s_wait_dscnt 0x0
	v_fma_f32 v42, -v139, v122, v42
	ds_bpermute_b32 v122, v105, v136
	s_wait_dscnt 0x0
	v_fma_f32 v43, -v139, v122, v43
	ds_bpermute_b32 v122, v106, v136
	s_wait_dscnt 0x0
	v_fma_f32 v44, -v139, v122, v44
	ds_bpermute_b32 v122, v107, v136
	s_wait_dscnt 0x0
	v_fma_f32 v45, -v139, v122, v45
	ds_bpermute_b32 v122, v108, v136
	s_wait_dscnt 0x0
	v_fma_f32 v46, -v139, v122, v46
	ds_bpermute_b32 v122, v109, v136
	s_wait_dscnt 0x0
	v_fma_f32 v47, -v139, v122, v47
	ds_bpermute_b32 v122, v110, v136
	s_wait_dscnt 0x0
	v_fma_f32 v48, -v139, v122, v48
	ds_bpermute_b32 v122, v111, v136
	s_wait_dscnt 0x0
	v_fma_f32 v49, -v139, v122, v49
	ds_bpermute_b32 v122, v112, v136
	s_wait_dscnt 0x0
	v_fma_f32 v50, -v139, v122, v50
	ds_bpermute_b32 v122, v113, v136
	s_wait_dscnt 0x0
	v_fma_f32 v51, -v139, v122, v51
	ds_bpermute_b32 v122, v114, v136
	s_wait_dscnt 0x0
	v_fma_f32 v52, -v139, v122, v52
	ds_bpermute_b32 v122, v115, v136
	s_wait_dscnt 0x0
	v_fma_f32 v53, -v139, v122, v53
	ds_bpermute_b32 v122, v116, v136
	s_wait_dscnt 0x0
	v_fma_f32 v54, -v139, v122, v54
	ds_bpermute_b32 v122, v117, v136
	s_wait_dscnt 0x0
	v_fma_f32 v55, -v139, v122, v55
	ds_bpermute_b32 v122, v118, v136
	s_wait_dscnt 0x0
	v_fma_f32 v56, -v139, v122, v56
	ds_bpermute_b32 v122, v120, v137
	s_wait_dscnt 0x0
	v_fma_f32 v122, -v139, v122, v57
	ds_bpermute_b32 v57, v104, v137
	s_wait_dscnt 0x0
	v_fma_f32 v123, -v139, v57, v58
	ds_bpermute_b32 v57, v105, v137
	s_wait_dscnt 0x0
	v_fma_f32 v124, -v139, v57, v59
	ds_bpermute_b32 v57, v106, v137
	s_wait_dscnt 0x0
	v_fma_f32 v125, -v139, v57, v60
	ds_bpermute_b32 v57, v107, v137
	s_wait_dscnt 0x0
	v_fma_f32 v126, -v139, v57, v61
	ds_bpermute_b32 v57, v108, v137
	s_wait_dscnt 0x0
	v_fma_f32 v127, -v139, v57, v62
	ds_bpermute_b32 v57, v109, v137
	s_wait_dscnt 0x0
	v_fma_f32 v128, -v139, v57, v63
	ds_bpermute_b32 v57, v110, v137
	s_wait_dscnt 0x0
	v_fma_f32 v129, -v139, v57, v64
	ds_bpermute_b32 v57, v111, v137
	s_wait_dscnt 0x0
	v_fma_f32 v130, -v139, v57, v65
	ds_bpermute_b32 v57, v112, v137
	s_wait_dscnt 0x0
	v_fma_f32 v131, -v139, v57, v66
	ds_bpermute_b32 v57, v113, v137
	s_wait_dscnt 0x0
	v_fma_f32 v132, -v139, v57, v67
	ds_bpermute_b32 v57, v114, v137
	s_wait_dscnt 0x0
	v_fma_f32 v133, -v139, v57, v68
	ds_bpermute_b32 v57, v115, v137
	s_wait_dscnt 0x0
	v_fma_f32 v134, -v139, v57, v69
	ds_bpermute_b32 v57, v116, v137
	s_wait_dscnt 0x0
	v_fma_f32 v135, -v139, v57, v70
	ds_bpermute_b32 v57, v117, v137
	s_wait_dscnt 0x0
	v_fma_f32 v136, -v139, v57, v71
	ds_bpermute_b32 v57, v118, v137
	s_wait_dscnt 0x0
	v_fma_f32 v137, -v139, v57, v72
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e32 v57, 0, v52, vcc_lo
	ds_load_2addr_b32 v[138:139], v138 offset0:192 offset1:224
	v_cndmask_b32_e64 v57, v57, v133, s0
	ds_bpermute_b32 v58, v103, v57
	s_wait_dscnt 0x0
	v_add_f32_e32 v140, v57, v58
	ds_bpermute_b32 v57, v120, v138
	s_wait_dscnt 0x0
	v_fma_f32 v65, -v140, v57, v41
	ds_bpermute_b32 v41, v104, v138
	s_wait_dscnt 0x0
	v_fma_f32 v66, -v140, v41, v42
	ds_bpermute_b32 v41, v105, v138
	ds_bpermute_b32 v42, v112, v139
	s_wait_dscnt 0x1
	v_fma_f32 v67, -v140, v41, v43
	ds_bpermute_b32 v41, v106, v138
	ds_bpermute_b32 v43, v113, v139
	s_wait_dscnt 0x2
	v_fma_f32 v42, -v140, v42, v131
	s_wait_dscnt 0x1
	v_fma_f32 v68, -v140, v41, v44
	ds_bpermute_b32 v41, v107, v138
	ds_bpermute_b32 v44, v114, v139
	s_wait_dscnt 0x2
	v_fma_f32 v43, -v140, v43, v132
	s_wait_dscnt 0x1
	v_fma_f32 v69, -v140, v41, v45
	ds_bpermute_b32 v41, v108, v138
	ds_bpermute_b32 v45, v115, v139
	s_wait_dscnt 0x2
	v_fma_f32 v44, -v140, v44, v133
	s_wait_dscnt 0x1
	v_fma_f32 v70, -v140, v41, v46
	ds_bpermute_b32 v41, v109, v138
	ds_bpermute_b32 v46, v116, v139
	s_wait_dscnt 0x2
	v_fma_f32 v45, -v140, v45, v134
	s_wait_dscnt 0x1
	v_fma_f32 v71, -v140, v41, v47
	ds_bpermute_b32 v41, v110, v138
	ds_bpermute_b32 v47, v117, v139
	s_wait_dscnt 0x2
	v_fma_f32 v46, -v140, v46, v135
	s_wait_dscnt 0x1
	v_fma_f32 v72, -v140, v41, v48
	ds_bpermute_b32 v41, v111, v138
	ds_bpermute_b32 v48, v118, v139
	s_wait_dscnt 0x2
	v_fma_f32 v47, -v140, v47, v136
	s_wait_dscnt 0x1
	v_fma_f32 v57, -v140, v41, v49
	ds_bpermute_b32 v41, v112, v138
	s_wait_dscnt 0x1
	v_fma_f32 v48, -v140, v48, v137
	s_wait_dscnt 0x0
	v_fma_f32 v58, -v140, v41, v50
	ds_bpermute_b32 v41, v113, v138
	s_wait_dscnt 0x0
	v_fma_f32 v59, -v140, v41, v51
	ds_bpermute_b32 v41, v114, v138
	s_wait_dscnt 0x0
	v_fma_f32 v60, -v140, v41, v52
	ds_bpermute_b32 v41, v115, v138
	s_wait_dscnt 0x0
	v_fma_f32 v61, -v140, v41, v53
	ds_bpermute_b32 v41, v116, v138
	s_wait_dscnt 0x0
	v_fma_f32 v62, -v140, v41, v54
	ds_bpermute_b32 v41, v117, v138
	s_wait_dscnt 0x0
	v_fma_f32 v63, -v140, v41, v55
	ds_bpermute_b32 v41, v118, v138
	s_wait_dscnt 0x0
	v_fma_f32 v64, -v140, v41, v56
	ds_bpermute_b32 v41, v120, v139
	s_wait_dscnt 0x0
	v_fma_f32 v49, -v140, v41, v122
	ds_bpermute_b32 v41, v104, v139
	s_wait_dscnt 0x0
	v_fma_f32 v50, -v140, v41, v123
	ds_bpermute_b32 v41, v105, v139
	s_wait_dscnt 0x0
	v_fma_f32 v51, -v140, v41, v124
	ds_bpermute_b32 v41, v106, v139
	s_wait_dscnt 0x0
	v_fma_f32 v52, -v140, v41, v125
	ds_bpermute_b32 v41, v107, v139
	s_wait_dscnt 0x0
	v_fma_f32 v53, -v140, v41, v126
	ds_bpermute_b32 v41, v108, v139
	s_wait_dscnt 0x0
	v_fma_f32 v54, -v140, v41, v127
	ds_bpermute_b32 v41, v109, v139
	s_wait_dscnt 0x0
	v_fma_f32 v55, -v140, v41, v128
	ds_bpermute_b32 v41, v110, v139
	s_wait_dscnt 0x0
	v_fma_f32 v56, -v140, v41, v129
	ds_bpermute_b32 v41, v111, v139
	s_wait_dscnt 0x0
	v_fma_f32 v41, -v140, v41, v130
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e32 v122, 0, v61, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v122, v122, v45, s0
	ds_bpermute_b32 v123, v103, v122
	s_wait_dscnt 0x0
	v_add_f32_e32 v122, v122, v123
	ds_load_2addr_b32 v[123:124], v121 offset1:32
	s_wait_dscnt 0x0
	ds_bpermute_b32 v125, v120, v123
	s_wait_dscnt 0x0
	v_fma_f32 v65, -v122, v125, v65
	ds_bpermute_b32 v125, v104, v123
	s_wait_dscnt 0x0
	v_fma_f32 v66, -v122, v125, v66
	ds_bpermute_b32 v125, v105, v123
	s_wait_dscnt 0x0
	v_fma_f32 v67, -v122, v125, v67
	ds_bpermute_b32 v125, v106, v123
	s_wait_dscnt 0x0
	v_fma_f32 v68, -v122, v125, v68
	ds_bpermute_b32 v125, v107, v123
	s_wait_dscnt 0x0
	v_fma_f32 v69, -v122, v125, v69
	ds_bpermute_b32 v125, v108, v123
	s_wait_dscnt 0x0
	v_fma_f32 v70, -v122, v125, v70
	ds_bpermute_b32 v125, v109, v123
	s_wait_dscnt 0x0
	v_fma_f32 v71, -v122, v125, v71
	ds_bpermute_b32 v125, v110, v123
	s_wait_dscnt 0x0
	v_fma_f32 v72, -v122, v125, v72
	ds_bpermute_b32 v125, v111, v123
	s_wait_dscnt 0x0
	v_fma_f32 v57, -v122, v125, v57
	ds_bpermute_b32 v125, v112, v123
	s_wait_dscnt 0x0
	v_fma_f32 v58, -v122, v125, v58
	ds_bpermute_b32 v125, v113, v123
	s_wait_dscnt 0x0
	v_fma_f32 v59, -v122, v125, v59
	ds_bpermute_b32 v125, v114, v123
	s_wait_dscnt 0x0
	v_fma_f32 v60, -v122, v125, v60
	ds_bpermute_b32 v125, v115, v123
	s_wait_dscnt 0x0
	v_fma_f32 v61, -v122, v125, v61
	ds_bpermute_b32 v125, v116, v123
	s_wait_dscnt 0x0
	v_fma_f32 v62, -v122, v125, v62
	ds_bpermute_b32 v125, v117, v123
	ds_bpermute_b32 v123, v118, v123
	s_wait_dscnt 0x1
	v_fma_f32 v63, -v122, v125, v63
	s_wait_dscnt 0x0
	v_fma_f32 v64, -v122, v123, v64
	ds_bpermute_b32 v123, v120, v124
	s_wait_dscnt 0x0
	v_fma_f32 v49, -v122, v123, v49
	ds_bpermute_b32 v123, v104, v124
	s_wait_dscnt 0x0
	v_fma_f32 v50, -v122, v123, v50
	ds_bpermute_b32 v123, v105, v124
	s_wait_dscnt 0x0
	v_fma_f32 v51, -v122, v123, v51
	ds_bpermute_b32 v123, v106, v124
	s_wait_dscnt 0x0
	v_fma_f32 v52, -v122, v123, v52
	ds_bpermute_b32 v123, v107, v124
	s_wait_dscnt 0x0
	v_fma_f32 v53, -v122, v123, v53
	ds_bpermute_b32 v123, v108, v124
	s_wait_dscnt 0x0
	v_fma_f32 v54, -v122, v123, v54
	ds_bpermute_b32 v123, v109, v124
	s_wait_dscnt 0x0
	v_fma_f32 v55, -v122, v123, v55
	ds_bpermute_b32 v123, v110, v124
	s_wait_dscnt 0x0
	v_fma_f32 v56, -v122, v123, v56
	ds_bpermute_b32 v123, v111, v124
	s_wait_dscnt 0x0
	v_fma_f32 v41, -v122, v123, v41
	ds_bpermute_b32 v123, v112, v124
	s_wait_dscnt 0x0
	v_fma_f32 v42, -v122, v123, v42
	ds_bpermute_b32 v123, v113, v124
	s_wait_dscnt 0x0
	v_fma_f32 v43, -v122, v123, v43
	ds_bpermute_b32 v123, v114, v124
	s_wait_dscnt 0x0
	v_fma_f32 v44, -v122, v123, v44
	ds_bpermute_b32 v123, v115, v124
	s_wait_dscnt 0x0
	v_fma_f32 v45, -v122, v123, v45
	ds_bpermute_b32 v123, v116, v124
	s_wait_dscnt 0x0
	v_fma_f32 v46, -v122, v123, v46
	ds_bpermute_b32 v123, v117, v124
	s_wait_dscnt 0x0
	v_fma_f32 v47, -v122, v123, v47
	ds_bpermute_b32 v123, v118, v124
	s_wait_dscnt 0x0
	v_fma_f32 v48, -v122, v123, v48
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e32 v122, 0, v62, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v122, v122, v46, s0
	ds_bpermute_b32 v123, v103, v122
	s_wait_dscnt 0x0
	v_add_f32_e32 v124, v122, v123
	ds_load_2addr_b32 v[122:123], v121 offset0:64 offset1:96
	s_wait_dscnt 0x0
	ds_bpermute_b32 v125, v120, v122
	s_wait_dscnt 0x0
	v_fma_f32 v65, -v124, v125, v65
	ds_bpermute_b32 v125, v104, v122
	s_wait_dscnt 0x0
	v_fma_f32 v66, -v124, v125, v66
	ds_bpermute_b32 v125, v105, v122
	s_wait_dscnt 0x0
	v_fma_f32 v67, -v124, v125, v67
	ds_bpermute_b32 v125, v106, v122
	s_wait_dscnt 0x0
	v_fma_f32 v68, -v124, v125, v68
	ds_bpermute_b32 v125, v107, v122
	s_wait_dscnt 0x0
	v_fma_f32 v69, -v124, v125, v69
	ds_bpermute_b32 v125, v108, v122
	s_wait_dscnt 0x0
	v_fma_f32 v70, -v124, v125, v70
	ds_bpermute_b32 v125, v109, v122
	s_wait_dscnt 0x0
	v_fma_f32 v71, -v124, v125, v71
	ds_bpermute_b32 v125, v110, v122
	s_wait_dscnt 0x0
	v_fma_f32 v72, -v124, v125, v72
	ds_bpermute_b32 v125, v111, v122
	s_wait_dscnt 0x0
	v_fma_f32 v57, -v124, v125, v57
	ds_bpermute_b32 v125, v112, v122
	s_wait_dscnt 0x0
	v_fma_f32 v58, -v124, v125, v58
	ds_bpermute_b32 v125, v113, v122
	s_wait_dscnt 0x0
	v_fma_f32 v59, -v124, v125, v59
	ds_bpermute_b32 v125, v114, v122
	s_wait_dscnt 0x0
	v_fma_f32 v60, -v124, v125, v60
	ds_bpermute_b32 v125, v115, v122
	s_wait_dscnt 0x0
	v_fma_f32 v61, -v124, v125, v61
	ds_bpermute_b32 v125, v116, v122
	s_wait_dscnt 0x0
	v_fma_f32 v62, -v124, v125, v62
	ds_bpermute_b32 v125, v117, v122
	ds_bpermute_b32 v122, v118, v122
	s_wait_dscnt 0x1
	v_fma_f32 v63, -v124, v125, v63
	s_wait_dscnt 0x0
	v_fma_f32 v64, -v124, v122, v64
	ds_bpermute_b32 v122, v120, v123
	s_wait_dscnt 0x0
	v_fma_f32 v49, -v124, v122, v49
	ds_bpermute_b32 v122, v104, v123
	s_wait_dscnt 0x0
	v_fma_f32 v50, -v124, v122, v50
	ds_bpermute_b32 v122, v105, v123
	s_wait_dscnt 0x0
	v_fma_f32 v51, -v124, v122, v51
	ds_bpermute_b32 v122, v106, v123
	s_wait_dscnt 0x0
	v_fma_f32 v52, -v124, v122, v52
	ds_bpermute_b32 v122, v107, v123
	s_wait_dscnt 0x0
	v_fma_f32 v53, -v124, v122, v53
	ds_bpermute_b32 v122, v108, v123
	s_wait_dscnt 0x0
	v_fma_f32 v54, -v124, v122, v54
	ds_bpermute_b32 v122, v109, v123
	s_wait_dscnt 0x0
	v_fma_f32 v55, -v124, v122, v55
	ds_bpermute_b32 v122, v110, v123
	s_wait_dscnt 0x0
	v_fma_f32 v56, -v124, v122, v56
	ds_bpermute_b32 v122, v111, v123
	s_wait_dscnt 0x0
	v_fma_f32 v41, -v124, v122, v41
	ds_bpermute_b32 v122, v112, v123
	s_wait_dscnt 0x0
	v_fma_f32 v42, -v124, v122, v42
	ds_bpermute_b32 v122, v113, v123
	s_wait_dscnt 0x0
	v_fma_f32 v43, -v124, v122, v43
	ds_bpermute_b32 v122, v114, v123
	s_wait_dscnt 0x0
	v_fma_f32 v44, -v124, v122, v44
	ds_bpermute_b32 v122, v115, v123
	s_wait_dscnt 0x0
	v_fma_f32 v45, -v124, v122, v45
	ds_bpermute_b32 v122, v116, v123
	s_wait_dscnt 0x0
	v_fma_f32 v46, -v124, v122, v46
	ds_bpermute_b32 v122, v117, v123
	s_wait_dscnt 0x0
	v_fma_f32 v47, -v124, v122, v47
	ds_bpermute_b32 v122, v118, v123
	s_wait_dscnt 0x0
	v_fma_f32 v48, -v124, v122, v48
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e32 v122, 0, v63, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v122, v122, v47, s0
	ds_bpermute_b32 v123, v103, v122
	s_wait_dscnt 0x0
	v_add_f32_e32 v124, v122, v123
	ds_load_2addr_b32 v[122:123], v121 offset0:128 offset1:160
	s_wait_dscnt 0x0
	ds_bpermute_b32 v125, v120, v122
	s_wait_dscnt 0x0
	v_fma_f32 v65, -v124, v125, v65
	ds_bpermute_b32 v125, v104, v122
	s_wait_dscnt 0x0
	v_fma_f32 v66, -v124, v125, v66
	ds_bpermute_b32 v125, v105, v122
	s_wait_dscnt 0x0
	v_fma_f32 v67, -v124, v125, v67
	ds_bpermute_b32 v125, v106, v122
	s_wait_dscnt 0x0
	v_fma_f32 v68, -v124, v125, v68
	ds_bpermute_b32 v125, v107, v122
	s_wait_dscnt 0x0
	v_fma_f32 v69, -v124, v125, v69
	ds_bpermute_b32 v125, v108, v122
	s_wait_dscnt 0x0
	v_fma_f32 v70, -v124, v125, v70
	ds_bpermute_b32 v125, v109, v122
	s_wait_dscnt 0x0
	v_fma_f32 v71, -v124, v125, v71
	ds_bpermute_b32 v125, v110, v122
	s_wait_dscnt 0x0
	v_fma_f32 v72, -v124, v125, v72
	ds_bpermute_b32 v125, v111, v122
	s_wait_dscnt 0x0
	v_fma_f32 v57, -v124, v125, v57
	ds_bpermute_b32 v125, v112, v122
	s_wait_dscnt 0x0
	v_fma_f32 v58, -v124, v125, v58
	ds_bpermute_b32 v125, v113, v122
	s_wait_dscnt 0x0
	v_fma_f32 v59, -v124, v125, v59
	ds_bpermute_b32 v125, v114, v122
	s_wait_dscnt 0x0
	v_fma_f32 v60, -v124, v125, v60
	ds_bpermute_b32 v125, v115, v122
	s_wait_dscnt 0x0
	v_fma_f32 v61, -v124, v125, v61
	ds_bpermute_b32 v125, v116, v122
	s_wait_dscnt 0x0
	v_fma_f32 v62, -v124, v125, v62
	ds_bpermute_b32 v125, v117, v122
	ds_bpermute_b32 v122, v118, v122
	s_wait_dscnt 0x1
	v_fma_f32 v63, -v124, v125, v63
	s_wait_dscnt 0x0
	v_fma_f32 v64, -v124, v122, v64
	ds_bpermute_b32 v122, v120, v123
	s_wait_dscnt 0x0
	v_fma_f32 v49, -v124, v122, v49
	ds_bpermute_b32 v122, v104, v123
	s_wait_dscnt 0x0
	v_fma_f32 v50, -v124, v122, v50
	ds_bpermute_b32 v122, v105, v123
	s_wait_dscnt 0x0
	v_fma_f32 v51, -v124, v122, v51
	ds_bpermute_b32 v122, v106, v123
	s_wait_dscnt 0x0
	v_fma_f32 v52, -v124, v122, v52
	ds_bpermute_b32 v122, v107, v123
	s_wait_dscnt 0x0
	v_fma_f32 v53, -v124, v122, v53
	ds_bpermute_b32 v122, v108, v123
	s_wait_dscnt 0x0
	v_fma_f32 v54, -v124, v122, v54
	ds_bpermute_b32 v122, v109, v123
	s_wait_dscnt 0x0
	v_fma_f32 v55, -v124, v122, v55
	ds_bpermute_b32 v122, v110, v123
	s_wait_dscnt 0x0
	v_fma_f32 v56, -v124, v122, v56
	ds_bpermute_b32 v122, v111, v123
	s_wait_dscnt 0x0
	v_fma_f32 v41, -v124, v122, v41
	ds_bpermute_b32 v122, v112, v123
	s_wait_dscnt 0x0
	v_fma_f32 v42, -v124, v122, v42
	ds_bpermute_b32 v122, v113, v123
	s_wait_dscnt 0x0
	v_fma_f32 v43, -v124, v122, v43
	ds_bpermute_b32 v122, v114, v123
	s_wait_dscnt 0x0
	v_fma_f32 v44, -v124, v122, v44
	ds_bpermute_b32 v122, v115, v123
	s_wait_dscnt 0x0
	v_fma_f32 v45, -v124, v122, v45
	ds_bpermute_b32 v122, v116, v123
	s_wait_dscnt 0x0
	v_fma_f32 v46, -v124, v122, v46
	ds_bpermute_b32 v122, v117, v123
	s_wait_dscnt 0x0
	v_fma_f32 v47, -v124, v122, v47
	ds_bpermute_b32 v122, v118, v123
	s_wait_dscnt 0x0
	v_fma_f32 v48, -v124, v122, v48
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e32 v122, 0, v64, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v122, v122, v48, s0
	ds_bpermute_b32 v123, v103, v122
	s_wait_dscnt 0x0
	v_add_f32_e32 v123, v122, v123
	ds_load_2addr_b32 v[121:122], v121 offset0:192 offset1:224
	s_wait_dscnt 0x0
	ds_bpermute_b32 v124, v120, v121
	ds_bpermute_b32 v120, v120, v122
	s_wait_dscnt 0x1
	v_fma_f32 v65, -v123, v124, v65
	s_wait_dscnt 0x0
	v_fma_f32 v49, -v123, v120, v49
	ds_bpermute_b32 v120, v104, v122
	ds_bpermute_b32 v124, v104, v121
	s_wait_dscnt 0x1
	v_fma_f32 v50, -v123, v120, v50
	ds_bpermute_b32 v120, v105, v122
	s_wait_dscnt 0x1
	v_fma_f32 v66, -v123, v124, v66
	ds_bpermute_b32 v124, v105, v121
	s_wait_dscnt 0x1
	v_fma_f32 v51, -v123, v120, v51
	ds_bpermute_b32 v120, v106, v122
	s_wait_dscnt 0x1
	v_fma_f32 v67, -v123, v124, v67
	ds_bpermute_b32 v124, v106, v121
	s_wait_dscnt 0x1
	v_fma_f32 v52, -v123, v120, v52
	ds_bpermute_b32 v120, v107, v122
	s_wait_dscnt 0x1
	v_fma_f32 v68, -v123, v124, v68
	ds_bpermute_b32 v124, v107, v121
	s_wait_dscnt 0x1
	v_fma_f32 v53, -v123, v120, v53
	ds_bpermute_b32 v120, v108, v122
	s_wait_dscnt 0x1
	v_fma_f32 v69, -v123, v124, v69
	ds_bpermute_b32 v124, v108, v121
	s_wait_dscnt 0x1
	v_fma_f32 v54, -v123, v120, v54
	ds_bpermute_b32 v120, v109, v122
	s_wait_dscnt 0x1
	v_fma_f32 v70, -v123, v124, v70
	ds_bpermute_b32 v124, v109, v121
	s_wait_dscnt 0x1
	v_fma_f32 v55, -v123, v120, v55
	ds_bpermute_b32 v120, v110, v122
	s_wait_dscnt 0x1
	v_fma_f32 v71, -v123, v124, v71
	ds_bpermute_b32 v124, v110, v121
	s_wait_dscnt 0x1
	v_fma_f32 v56, -v123, v120, v56
	ds_bpermute_b32 v120, v111, v122
	s_wait_dscnt 0x1
	v_fma_f32 v72, -v123, v124, v72
	ds_bpermute_b32 v124, v111, v121
	s_wait_dscnt 0x1
	v_fma_f32 v41, -v123, v120, v41
	ds_bpermute_b32 v120, v112, v122
	s_wait_dscnt 0x1
	v_fma_f32 v57, -v123, v124, v57
	ds_bpermute_b32 v124, v112, v121
	s_wait_dscnt 0x1
	v_fma_f32 v42, -v123, v120, v42
	ds_bpermute_b32 v120, v113, v122
	s_wait_dscnt 0x1
	v_fma_f32 v58, -v123, v124, v58
	ds_bpermute_b32 v124, v113, v121
	s_wait_dscnt 0x1
	v_fma_f32 v43, -v123, v120, v43
	ds_bpermute_b32 v120, v114, v122
	s_wait_dscnt 0x1
	v_fma_f32 v59, -v123, v124, v59
	ds_bpermute_b32 v124, v114, v121
	s_wait_dscnt 0x1
	v_fma_f32 v44, -v123, v120, v44
	ds_bpermute_b32 v120, v115, v122
	s_wait_dscnt 0x1
	v_fma_f32 v60, -v123, v124, v60
	ds_bpermute_b32 v124, v115, v121
	s_wait_dscnt 0x1
	v_fma_f32 v45, -v123, v120, v45
	ds_bpermute_b32 v120, v116, v122
	s_wait_dscnt 0x1
	v_fma_f32 v61, -v123, v124, v61
	ds_bpermute_b32 v124, v116, v121
	s_wait_dscnt 0x1
	v_fma_f32 v46, -v123, v120, v46
	ds_bpermute_b32 v120, v117, v122
	s_wait_dscnt 0x1
	v_fma_f32 v62, -v123, v124, v62
	ds_bpermute_b32 v124, v117, v121
	ds_bpermute_b32 v121, v118, v121
	s_wait_dscnt 0x2
	v_fma_f32 v47, -v123, v120, v47
	ds_bpermute_b32 v120, v118, v122
	s_wait_dscnt 0x2
	v_fma_f32 v63, -v123, v124, v63
	s_wait_dscnt 0x1
	v_fma_f32 v64, -v123, v121, v64
	s_wait_dscnt 0x0
	v_fma_f32 v48, -v123, v120, v48
	;;#ASMSTART
	;;#ASMEND
	s_cbranch_scc0 .LBB6_27
; %bb.28:                               ;   in Loop: Header=BB6_26 Depth=1
	v_wmma_f32_16x16x16_f16 v[120:127], v[77:80], v[81:84], v[33:40]
	v_cvt_f16_f32_e32 v72.h, v72
	v_cvt_f16_f32_e32 v72.l, v71
	v_cvt_f16_f32_e32 v71.h, v70
	s_delay_alu instid0(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[120:127], v[77:80], v[85:88], v[120:127]
	v_cvt_f16_f32_e32 v71.l, v69
	v_cvt_f16_f32_e32 v70.h, v68
	v_cvt_f16_f32_e32 v70.l, v67
	v_cvt_f16_f32_e32 v69.h, v66
	v_wmma_f32_16x16x16_f16 v[120:127], v[77:80], v[89:92], v[120:127]
	v_cvt_f16_f32_e32 v69.l, v65
	v_cvt_f16_f32_e32 v64.h, v64
	v_cvt_f16_f32_e32 v64.l, v63
	v_cvt_f16_f32_e32 v63.h, v62
	v_wmma_f32_16x16x16_f16 v[120:127], v[77:80], v[93:96], v[120:127]
	v_cvt_f16_f32_e32 v63.l, v61
	v_cvt_f16_f32_e32 v62.h, v60
	v_cvt_f16_f32_e32 v62.l, v59
	v_cvt_f16_f32_e32 v61.h, v58
	v_cvt_f16_f32_e32 v61.l, v57
	v_wmma_f32_16x16x16_f16 v[120:127], v[73:76], v[69:72], v[120:127]
	v_cvt_f16_f32_e32 v60.h, v56
	v_cvt_f16_f32_e32 v60.l, v55
	v_cvt_f16_f32_e32 v59.h, v54
	v_cvt_f16_f32_e32 v59.l, v53
	v_wmma_f32_16x16x16_f16 v[120:127], v[73:76], v[61:64], v[120:127]
	v_cvt_f16_f32_e32 v58.h, v52
	v_cvt_f16_f32_e32 v58.l, v51
	v_cvt_f16_f32_e32 v57.h, v50
	v_cvt_f16_f32_e32 v57.l, v49
	v_cvt_f16_f32_e32 v68.h, v48
	v_cvt_f16_f32_e32 v68.l, v47
	v_cvt_f16_f32_e32 v67.h, v46
	v_cvt_f16_f32_e32 v67.l, v45
	v_wmma_f32_16x16x16_f16 v[120:127], v[73:76], v[57:60], v[120:127]
	v_cvt_f16_f32_e32 v66.h, v44
	v_cvt_f16_f32_e32 v66.l, v43
	v_cvt_f16_f32_e32 v65.h, v42
	v_cvt_f16_f32_e32 v65.l, v41
	v_dual_add_f32 v48, v40, v40 :: v_dual_add_f32 v47, v39, v39
	v_dual_add_f32 v46, v38, v38 :: v_dual_add_f32 v45, v37, v37
	v_dual_add_f32 v44, v36, v36 :: v_dual_add_f32 v43, v35, v35
	v_dual_add_f32 v42, v34, v34 :: v_dual_add_f32 v41, v33, v33
	v_wmma_f32_16x16x16_f16 v[120:127], v[73:76], v[65:68], v[120:127]
	v_mul_f32_e32 v54, 0x40400000, v38
	v_mul_f32_e32 v56, 0x40400000, v40
	v_mul_f32_e32 v52, 0x40400000, v36
	v_wmma_f32_16x16x16_f16 v[41:48], v[77:80], v[81:84], v[41:48]
	v_add_f32_e32 v49, 0, v120
	v_dual_mul_f32 v53, 0x40400000, v37 :: v_dual_mul_f32 v38, 4.0, v38
	v_mul_f32_e32 v50, 0x40400000, v34
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[41:48], v[77:80], v[85:88], v[41:48]
	v_add_f32_e32 v49, v49, v121
	v_dual_mul_f32 v51, 0x40400000, v35 :: v_dual_mul_f32 v36, 4.0, v36
	v_dual_mul_f32 v37, 4.0, v37 :: v_dual_mul_f32 v26, 0x3f7d70a4, v26
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[41:48], v[77:80], v[89:92], v[41:48]
	v_dual_add_f32 v49, v49, v122 :: v_dual_mul_f32 v40, 4.0, v40
	v_dual_mul_f32 v35, 4.0, v35 :: v_dual_mul_f32 v14, 0x3f7d70a4, v14
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_wmma_f32_16x16x16_f16 v[41:48], v[77:80], v[93:96], v[41:48]
	v_dual_add_f32 v49, v49, v123 :: v_dual_mul_f32 v34, 4.0, v34
	v_mul_f32_e32 v30, 0x3f7d70a4, v30
	v_mul_f32_e32 v12, 0x3f7d70a4, v12
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[41:48], v[73:76], v[69:72], v[41:48]
	v_add_f32_e32 v49, v49, v124
	v_mul_f32_e32 v23, 0x3f7d70a4, v23
	v_mul_f32_e32 v21, 0x3f7d70a4, v21
	v_mul_f32_e32 v19, 0x3f7d70a4, v19
	v_wmma_f32_16x16x16_f16 v[41:48], v[73:76], v[61:64], v[41:48]
	v_add_f32_e32 v49, v49, v125
	v_mul_f32_e32 v55, 0x40400000, v39
	v_dual_mul_f32 v24, 0x3f7d70a4, v24 :: v_dual_mul_f32 v39, 4.0, v39
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[41:48], v[73:76], v[57:60], v[41:48]
	v_dual_add_f32 v120, v49, v126 :: v_dual_mul_f32 v49, 0x40400000, v33
	v_dual_mul_f32 v20, 0x3f7d70a4, v20 :: v_dual_mul_f32 v33, 4.0, v33
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_wmma_f32_16x16x16_f16 v[41:48], v[73:76], v[65:68], v[41:48]
	v_add_f32_e32 v120, v120, v127
	s_delay_alu instid0(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[49:56], v[77:80], v[81:84], v[49:56]
	v_mul_f32_e32 v22, 0x3f7d70a4, v22
	v_wmma_f32_16x16x16_f16 v[33:40], v[77:80], v[81:84], v[33:40]
	v_mul_f32_e32 v17, 0x3f7d70a4, v17
	v_add_f32_e32 v41, v120, v41
	v_wmma_f32_16x16x16_f16 v[49:56], v[77:80], v[85:88], v[49:56]
	v_mul_f32_e32 v18, 0x3f7d70a4, v18
	v_wmma_f32_16x16x16_f16 v[33:40], v[77:80], v[85:88], v[33:40]
	v_mul_f32_e32 v31, 0x3f7d70a4, v31
	v_add_f32_e32 v41, v41, v42
	v_wmma_f32_16x16x16_f16 v[49:56], v[77:80], v[89:92], v[49:56]
	v_mul_f32_e32 v32, 0x3f7d70a4, v32
	v_wmma_f32_16x16x16_f16 v[33:40], v[77:80], v[89:92], v[33:40]
	v_mul_f32_e32 v29, 0x3f7d70a4, v29
	v_add_f32_e32 v41, v41, v43
	v_wmma_f32_16x16x16_f16 v[49:56], v[77:80], v[93:96], v[49:56]
	v_mul_f32_e32 v28, 0x3f7d70a4, v28
	v_wmma_f32_16x16x16_f16 v[33:40], v[77:80], v[93:96], v[33:40]
	v_mul_f32_e32 v27, 0x3f7d70a4, v27
	v_add_f32_e32 v41, v41, v44
	v_wmma_f32_16x16x16_f16 v[49:56], v[73:76], v[69:72], v[49:56]
	v_mul_f32_e32 v16, 0x3f7d70a4, v16
	v_wmma_f32_16x16x16_f16 v[33:40], v[73:76], v[69:72], v[33:40]
	v_mul_f32_e32 v25, 0x3f7d70a4, v25
	v_add_f32_e32 v41, v41, v45
	v_wmma_f32_16x16x16_f16 v[49:56], v[73:76], v[61:64], v[49:56]
	v_mul_f32_e32 v10, 0x3f7d70a4, v10
	v_wmma_f32_16x16x16_f16 v[33:40], v[73:76], v[61:64], v[33:40]
	v_mul_f32_e32 v15, 0x3f7d70a4, v15
	v_add_f32_e32 v41, v41, v46
	v_wmma_f32_16x16x16_f16 v[49:56], v[73:76], v[57:60], v[49:56]
	v_mul_f32_e32 v8, 0x3f7d70a4, v8
	v_wmma_f32_16x16x16_f16 v[33:40], v[73:76], v[57:60], v[33:40]
	v_mul_f32_e32 v13, 0x3f7d70a4, v13
	v_add_f32_e32 v41, v41, v47
	v_wmma_f32_16x16x16_f16 v[49:56], v[73:76], v[65:68], v[49:56]
	v_mul_f32_e32 v4, 0x3f7d70a4, v4
	v_wmma_f32_16x16x16_f16 v[33:40], v[73:76], v[65:68], v[33:40]
	v_mul_f32_e32 v11, 0x3f7d70a4, v11
	v_dual_add_f32 v41, v41, v48 :: v_dual_mul_f32 v6, 0x3f7d70a4, v6
	v_mul_f32_e32 v9, 0x3f7d70a4, v9
	v_mul_f32_e32 v7, 0x3f7d70a4, v7
	v_mul_f32_e32 v5, 0x3f7d70a4, v5
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_dual_add_f32 v41, v41, v49 :: v_dual_mul_f32 v2, 0x3f7d70a4, v2
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
	s_add_co_i32 s6, s6, 1
	v_add_f32_e32 v41, v41, v54
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s6, s3
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v41, v41, v55
	v_add_f32_e32 v41, v41, v56
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v33, v41, v33
	v_add_f32_e32 v33, v33, v34
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
	v_cndmask_b32_e32 v33, 0x3a83126f, v119, vcc_lo
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
	.set .L_Z5probeILi4ELb0ELb1EEvPfPKfi.num_vgpr, 141
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
; codeLenInByte = 26296
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
	s_load_b32 s3, s[0:1], 0x10
	s_mov_b32 s0, 0x3c23d70a
	s_mov_b32 s1, 0x38d1b717
	s_mov_b32 s2, 0x399d4951
	s_wait_loadcnt 0xf
	v_fma_mixhi_f16 v61, v10, s0, s1
	s_mov_b32 s1, 0x3951b717
	s_wait_loadcnt_dscnt 0x0
	s_wait_alu depctr_sa_sdst(0)
	v_fma_mixhi_f16 v65, v10, s0, s1
	s_mov_b32 s1, 0x39d1b717
	s_barrier_signal -1
	s_wait_alu depctr_sa_sdst(0)
	v_fma_mixhi_f16 v66, v10, s0, s1
	s_mov_b32 s1, 0x3a1d4951
	v_fma_mixlo_f16 v66, v10, s0, s2
	s_mov_b32 s2, 0x3a03126e
	s_wait_alu depctr_sa_sdst(0)
	v_fma_mixhi_f16 v67, v10, s0, s1
	s_mov_b32 s1, 0x3a51b717
	v_fma_mixlo_f16 v67, v10, s0, s2
	s_mov_b32 s2, 0x3a378034
	s_wait_alu depctr_sa_sdst(0)
	v_fma_mixhi_f16 v68, v10, s0, s1
	s_mov_b32 s1, 0x3a6bedfa
	v_fma_mixlo_f16 v68, v10, s0, s2
	s_wait_alu depctr_sa_sdst(0)
	v_fma_mixhi_f16 v72, v10, s0, s1
	s_mov_b32 s1, 0x3a83126e
	v_fma_mixlo_f16 v61, v10, s0, 0
	v_mov_b16_e32 v65.l, v61.h
	v_mov_b16_e32 v62.l, v65.h
	v_mov_b16_e32 v62.h, v66.l
	v_mov_b16_e32 v63.l, v66.h
	v_mov_b16_e32 v63.h, v67.l
	v_mov_b16_e32 v64.l, v67.h
	v_mov_b16_e32 v64.h, v68.l
	v_mov_b16_e32 v72.l, v68.h
	s_wait_alu depctr_sa_sdst(0)
	v_fma_mixhi_f16 v60, v10, s0, s1
	v_mov_b16_e32 v60.l, v72.h
	s_mov_b32 s6, 0
	s_wait_kmcnt 0x0
	s_cmp_lt_i32 s3, 1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_cbranch_scc1 .LBB7_29
; %bb.25:
	v_mbcnt_lo_u32_b32 v2, -1, 0
	s_mov_b32 s0, 0x3a83126f
	v_dual_mov_b32 v70, v63 :: v_dual_and_b32 v1, 31, v0
	s_wait_alu depctr_sa_sdst(0)
	v_fma_mixlo_f16 v53, v3, s0, 0
	v_fma_mixhi_f16 v49, v4, s0, 0
	v_fma_mixhi_f16 v53, v5, s0, 0
	v_lshrrev_b32_e32 v3, 1, v0
	v_xor_b32_e32 v4, 16, v2
	v_bfi_b32 v5, v2, 0, 32
	v_fma_mixhi_f16 v54, v9, s0, 0
	v_dual_mov_b32 v58, v67 :: v_dual_lshlrev_b32 v9, 7, v0
	v_fma_mixlo_f16 v49, v10, s0, 0
	s_delay_alu instid0(VALU_DEP_4)
	v_cmp_lt_u32_e32 vcc_lo, v4, v5
	v_and_b32_e32 v73, 8, v3
	v_fma_mixlo_f16 v50, v6, s0, 0
	v_fma_mixlo_f16 v54, v7, s0, 0
	v_fma_mixhi_f16 v50, v8, s0, 0
	v_fma_mixhi_f16 v51, v14, s0, 0
	v_fma_mixhi_f16 v55, v15, s0, 0
	v_fma_mixlo_f16 v52, v16, s0, 0
	v_fma_mixlo_f16 v56, v12, s0, 0
	v_fma_mixhi_f16 v52, v13, s0, 0
	v_fma_mixhi_f16 v56, v11, s0, 0
	v_cndmask_b32_e32 v2, v2, v4, vcc_lo
	v_lshl_add_u32 v74, v1, 2, 0
	v_or_b32_e32 v1, 1, v73
	v_or_b32_e32 v3, 2, v73
	v_or_b32_e32 v4, 3, v73
	v_or_b32_e32 v5, 4, v73
	v_or_b32_e32 v6, 5, v73
	v_or_b32_e32 v7, 6, v73
	v_or_b32_e32 v8, 7, v73
	v_or_b32_e32 v10, 16, v73
	v_or_b32_e32 v11, 17, v73
	v_or_b32_e32 v12, 18, v73
	v_or_b32_e32 v13, 19, v73
	v_and_b32_e32 v75, 0x800, v9
	v_or_b32_e32 v9, 20, v73
	v_or_b32_e32 v14, 21, v73
	v_or_b32_e32 v15, 22, v73
	v_or_b32_e32 v16, 23, v73
	v_fma_mixlo_f16 v51, v17, s0, 0
	v_fma_mixlo_f16 v55, v18, s0, 0
	v_dual_mov_b32 v69, v62 :: v_dual_lshlrev_b32 v76, 2, v2
	v_lshlrev_b32_e32 v77, 2, v1
	v_dual_mov_b32 v71, v64 :: v_dual_lshlrev_b32 v78, 2, v3
	v_lshlrev_b32_e32 v79, 2, v4
	v_dual_mov_b32 v57, v66 :: v_dual_lshlrev_b32 v80, 2, v5
	v_lshlrev_b32_e32 v81, 2, v6
	v_dual_mov_b32 v59, v68 :: v_dual_lshlrev_b32 v82, 2, v7
	v_lshlrev_b32_e32 v83, 2, v8
	v_lshlrev_b32_e32 v84, 2, v10
	v_lshlrev_b32_e32 v85, 2, v11
	v_lshlrev_b32_e32 v86, 2, v12
	v_lshlrev_b32_e32 v87, 2, v13
	v_lshlrev_b32_e32 v88, 2, v9
	v_lshlrev_b32_e32 v89, 2, v14
	v_lshlrev_b32_e32 v90, 2, v15
	v_lshlrev_b32_e32 v91, 2, v16
	s_mov_b32 s7, 0x3f7d70a4
.LBB7_26:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB7_27 Depth 2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[49:52], v[53:56], 0
	s_mov_b32 s8, 0
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
	v_dual_add_f32 v48, 0x37fba882, v8 :: v_dual_add_f32 v47, 0x37fba882, v7
	v_dual_add_f32 v46, 0x37fba882, v6 :: v_dual_add_f32 v45, 0x37fba882, v5
	v_dual_add_f32 v44, 0x37fba882, v4 :: v_dual_add_f32 v43, 0x37fba882, v3
	v_dual_add_f32 v42, 0x37fba882, v2 :: v_dual_add_f32 v41, 0x37fba882, v1
	v_dual_add_f32 v99, 0x3827c5ac, v8 :: v_dual_add_f32 v98, 0x3827c5ac, v7
	v_dual_add_f32 v97, 0x3827c5ac, v6 :: v_dual_add_f32 v96, 0x3827c5ac, v5
	v_dual_add_f32 v95, 0x3827c5ac, v4 :: v_dual_add_f32 v94, 0x3827c5ac, v3
	v_dual_add_f32 v93, 0x3827c5ac, v2 :: v_dual_add_f32 v92, 0x3827c5ac, v1
	v_wmma_f32_16x16x16_f16 v[9:16], v[49:52], v[61:64], v[9:16]
	v_wmma_f32_16x16x16_f16 v[17:24], v[49:52], v[61:64], v[17:24]
	v_wmma_f32_16x16x16_f16 v[41:48], v[49:52], v[61:64], v[41:48]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[92:99], v[49:52], v[61:64], v[92:99]
	v_wmma_f32_16x16x16_f16 v[9:16], v[49:52], v[65:68], v[9:16]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[17:24], v[49:52], v[65:68], v[17:24]
	v_wmma_f32_16x16x16_f16 v[41:48], v[49:52], v[65:68], v[41:48]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[92:99], v[49:52], v[65:68], v[92:99]
	v_wmma_f32_16x16x16_f16 v[9:16], v[49:52], v[69:72], v[9:16]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[17:24], v[49:52], v[69:72], v[17:24]
	v_wmma_f32_16x16x16_f16 v[41:48], v[49:52], v[69:72], v[41:48]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[92:99], v[49:52], v[69:72], v[92:99]
	v_wmma_f32_16x16x16_f16 v[9:16], v[49:52], v[57:60], v[9:16]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[17:24], v[49:52], v[57:60], v[17:24]
	v_wmma_f32_16x16x16_f16 v[41:48], v[49:52], v[57:60], v[41:48]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[92:99], v[49:52], v[57:60], v[92:99]
	v_dual_mul_f32 v33, 0x3727c5ac, v9 :: v_dual_mul_f32 v34, 0x3727c5ac, v10
	v_dual_mul_f32 v35, 0x3727c5ac, v11 :: v_dual_mul_f32 v36, 0x3727c5ac, v12
	v_dual_mul_f32 v37, 0x3727c5ac, v13 :: v_dual_mul_f32 v38, 0x3727c5ac, v14
	v_dual_mul_f32 v39, 0x3727c5ac, v15 :: v_dual_mul_f32 v40, 0x3727c5ac, v16
	v_dual_mul_f32 v25, 0x3727c5ac, v17 :: v_dual_mul_f32 v26, 0x3727c5ac, v18
	v_dual_mul_f32 v27, 0x3727c5ac, v19 :: v_dual_mul_f32 v28, 0x3727c5ac, v20
	v_dual_mul_f32 v29, 0x3727c5ac, v21 :: v_dual_mul_f32 v30, 0x3727c5ac, v22
	v_dual_mul_f32 v31, 0x3727c5ac, v23 :: v_dual_mul_f32 v32, 0x3727c5ac, v24
	v_dual_mul_f32 v17, 0x3727c5ac, v41 :: v_dual_mul_f32 v18, 0x3727c5ac, v42
	v_dual_mul_f32 v19, 0x3727c5ac, v43 :: v_dual_mul_f32 v20, 0x3727c5ac, v44
	v_dual_mul_f32 v21, 0x3727c5ac, v45 :: v_dual_mul_f32 v22, 0x3727c5ac, v46
	v_dual_mul_f32 v23, 0x3727c5ac, v47 :: v_dual_mul_f32 v24, 0x3727c5ac, v48
	v_dual_mul_f32 v9, 0x3727c5ac, v92 :: v_dual_mul_f32 v10, 0x3727c5ac, v93
	v_dual_mul_f32 v11, 0x3727c5ac, v94 :: v_dual_mul_f32 v12, 0x3727c5ac, v95
	v_dual_mul_f32 v13, 0x3727c5ac, v96 :: v_dual_mul_f32 v14, 0x3727c5ac, v97
	v_dual_mul_f32 v15, 0x3727c5ac, v98 :: v_dual_mul_f32 v16, 0x3727c5ac, v99
.LBB7_27:                               ;   Parent Loop BB7_26 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_wait_alu depctr_sa_sdst(0)
	v_cmp_eq_u32_e64 s1, s8, v75
	v_or_b32_e32 v44, 0x2000, v75
	v_add_nc_u32_e32 v93, s8, v74
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v43, 0, v33, s1
	v_cmp_eq_u32_e64 s2, s8, v44
	ds_load_2addr_b32 v[41:42], v93 offset1:32
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v43, v43, v17, s2
	ds_bpermute_b32 v44, v76, v43
	v_lshlrev_b32_e32 v92, 2, v73
	s_wait_dscnt 0x1
	ds_bpermute_b32 v46, v77, v41
	ds_bpermute_b32 v47, v78, v41
	ds_bpermute_b32 v48, v79, v41
	ds_bpermute_b32 v94, v80, v41
	ds_bpermute_b32 v95, v81, v41
	ds_bpermute_b32 v96, v82, v41
	ds_bpermute_b32 v97, v83, v41
	ds_bpermute_b32 v98, v84, v41
	ds_bpermute_b32 v99, v85, v41
	ds_bpermute_b32 v100, v86, v41
	ds_bpermute_b32 v101, v87, v41
	ds_bpermute_b32 v102, v88, v41
	ds_bpermute_b32 v103, v89, v41
	ds_bpermute_b32 v104, v90, v41
	ds_bpermute_b32 v106, v77, v42
	ds_bpermute_b32 v107, v78, v42
	ds_bpermute_b32 v108, v79, v42
	ds_bpermute_b32 v109, v80, v42
	ds_bpermute_b32 v110, v81, v42
	ds_bpermute_b32 v111, v82, v42
	ds_bpermute_b32 v112, v83, v42
	ds_bpermute_b32 v113, v84, v42
	ds_bpermute_b32 v114, v85, v42
	ds_bpermute_b32 v115, v86, v42
	ds_bpermute_b32 v116, v87, v42
	ds_bpermute_b32 v117, v88, v42
	ds_bpermute_b32 v118, v89, v42
	ds_bpermute_b32 v119, v90, v42
	s_wait_dscnt 0x1c
	v_add_f32_e32 v43, v43, v44
	ds_bpermute_b32 v45, v92, v41
	ds_bpermute_b32 v41, v91, v41
	ds_bpermute_b32 v105, v92, v42
	ds_bpermute_b32 v42, v91, v42
	s_wait_dscnt 0x1f
	v_fma_f32 v34, -v43, v46, v34
	s_wait_dscnt 0x1e
	v_fma_f32 v35, -v43, v47, v35
	s_wait_dscnt 0x1d
	v_fma_f32 v36, -v43, v48, v36
	s_wait_dscnt 0x1c
	v_fma_f32 v37, -v43, v94, v37
	s_wait_dscnt 0x1b
	v_fma_f32 v38, -v43, v95, v38
	s_wait_dscnt 0x1a
	v_fma_f32 v39, -v43, v96, v39
	s_wait_dscnt 0x19
	v_fma_f32 v40, -v43, v97, v40
	s_wait_dscnt 0x18
	v_fma_f32 v25, -v43, v98, v25
	s_wait_dscnt 0x17
	v_fma_f32 v26, -v43, v99, v26
	s_wait_dscnt 0x16
	v_fma_f32 v27, -v43, v100, v27
	s_wait_dscnt 0x15
	v_fma_f32 v28, -v43, v101, v28
	s_wait_dscnt 0x14
	v_fma_f32 v29, -v43, v102, v29
	s_wait_dscnt 0x13
	v_fma_f32 v30, -v43, v103, v30
	s_wait_dscnt 0x12
	v_fma_f32 v31, -v43, v104, v31
	s_wait_dscnt 0x11
	v_fma_f32 v18, -v43, v106, v18
	s_wait_dscnt 0x10
	v_fma_f32 v19, -v43, v107, v19
	s_wait_dscnt 0x3
	v_fma_f32 v33, -v43, v45, v33
	s_wait_dscnt 0x2
	v_fma_f32 v32, -v43, v41, v32
	s_wait_dscnt 0x1
	v_fma_f32 v17, -v43, v105, v17
	v_fma_f32 v20, -v43, v108, v20
	v_fma_f32 v21, -v43, v109, v21
	v_fma_f32 v22, -v43, v110, v22
	v_fma_f32 v23, -v43, v111, v23
	v_fma_f32 v24, -v43, v112, v24
	v_fma_f32 v9, -v43, v113, v9
	v_fma_f32 v10, -v43, v114, v10
	v_fma_f32 v11, -v43, v115, v11
	v_fma_f32 v12, -v43, v116, v12
	v_fma_f32 v13, -v43, v117, v13
	v_fma_f32 v14, -v43, v118, v14
	v_fma_f32 v15, -v43, v119, v15
	s_wait_dscnt 0x0
	v_fma_f32 v16, -v43, v42, v16
	;;#ASMSTART
	;;#ASMEND
	ds_load_2addr_b32 v[41:42], v93 offset0:64 offset1:96
	v_cndmask_b32_e64 v43, 0, v34, s1
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v43, v43, v18, s2
	ds_bpermute_b32 v44, v76, v43
	s_wait_dscnt 0x1
	ds_bpermute_b32 v45, v92, v41
	ds_bpermute_b32 v46, v77, v41
	ds_bpermute_b32 v47, v78, v41
	ds_bpermute_b32 v48, v79, v41
	ds_bpermute_b32 v94, v80, v41
	ds_bpermute_b32 v95, v81, v41
	ds_bpermute_b32 v96, v82, v41
	ds_bpermute_b32 v97, v83, v41
	ds_bpermute_b32 v98, v84, v41
	ds_bpermute_b32 v99, v85, v41
	ds_bpermute_b32 v100, v86, v41
	ds_bpermute_b32 v101, v87, v41
	ds_bpermute_b32 v102, v88, v41
	ds_bpermute_b32 v103, v89, v41
	ds_bpermute_b32 v104, v90, v41
	ds_bpermute_b32 v41, v91, v41
	ds_bpermute_b32 v105, v92, v42
	ds_bpermute_b32 v106, v77, v42
	ds_bpermute_b32 v107, v78, v42
	ds_bpermute_b32 v108, v79, v42
	ds_bpermute_b32 v109, v80, v42
	ds_bpermute_b32 v110, v81, v42
	ds_bpermute_b32 v111, v82, v42
	ds_bpermute_b32 v112, v83, v42
	ds_bpermute_b32 v113, v84, v42
	ds_bpermute_b32 v114, v85, v42
	ds_bpermute_b32 v115, v86, v42
	ds_bpermute_b32 v116, v87, v42
	ds_bpermute_b32 v117, v88, v42
	ds_bpermute_b32 v118, v89, v42
	ds_bpermute_b32 v119, v90, v42
	ds_bpermute_b32 v120, v91, v42
	s_wait_dscnt 0x20
	v_add_f32_e32 v121, v43, v44
	s_wait_dscnt 0x1f
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v33, -v121, v45, v33
	s_wait_dscnt 0x1e
	v_fma_f32 v34, -v121, v46, v34
	s_wait_dscnt 0x1d
	v_fma_f32 v35, -v121, v47, v35
	s_wait_dscnt 0x1c
	v_fma_f32 v36, -v121, v48, v36
	s_wait_dscnt 0x1b
	v_fma_f32 v37, -v121, v94, v37
	s_wait_dscnt 0x1a
	v_fma_f32 v38, -v121, v95, v38
	s_wait_dscnt 0x19
	v_fma_f32 v39, -v121, v96, v39
	s_wait_dscnt 0x18
	v_fma_f32 v40, -v121, v97, v40
	s_wait_dscnt 0x17
	v_fma_f32 v25, -v121, v98, v25
	s_wait_dscnt 0x16
	v_fma_f32 v26, -v121, v99, v26
	s_wait_dscnt 0x15
	v_fma_f32 v27, -v121, v100, v27
	s_wait_dscnt 0x14
	v_fma_f32 v28, -v121, v101, v28
	s_wait_dscnt 0x13
	v_fma_f32 v29, -v121, v102, v29
	s_wait_dscnt 0x12
	v_fma_f32 v30, -v121, v103, v30
	s_wait_dscnt 0x11
	v_fma_f32 v31, -v121, v104, v31
	s_wait_dscnt 0x10
	v_fma_f32 v32, -v121, v41, v32
	s_wait_dscnt 0xf
	v_fma_f32 v41, -v121, v105, v17
	s_wait_dscnt 0xe
	v_fma_f32 v42, -v121, v106, v18
	s_wait_dscnt 0xd
	v_fma_f32 v43, -v121, v107, v19
	s_wait_dscnt 0xc
	v_fma_f32 v44, -v121, v108, v20
	s_wait_dscnt 0xb
	v_fma_f32 v45, -v121, v109, v21
	s_wait_dscnt 0xa
	v_fma_f32 v46, -v121, v110, v22
	s_wait_dscnt 0x9
	v_fma_f32 v47, -v121, v111, v23
	s_wait_dscnt 0x8
	v_fma_f32 v48, -v121, v112, v24
	s_wait_dscnt 0x7
	v_fma_f32 v94, -v121, v113, v9
	s_wait_dscnt 0x6
	v_fma_f32 v95, -v121, v114, v10
	s_wait_dscnt 0x5
	v_fma_f32 v96, -v121, v115, v11
	s_wait_dscnt 0x4
	v_fma_f32 v97, -v121, v116, v12
	s_wait_dscnt 0x3
	v_fma_f32 v98, -v121, v117, v13
	s_wait_dscnt 0x2
	v_fma_f32 v99, -v121, v118, v14
	s_wait_dscnt 0x1
	v_fma_f32 v100, -v121, v119, v15
	s_wait_dscnt 0x0
	v_fma_f32 v101, -v121, v120, v16
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v11, 0, v35, s1
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v11, v11, v43, s2
	ds_bpermute_b32 v12, v76, v11
	s_wait_dscnt 0x0
	v_add_f32_e32 v122, v11, v12
	ds_load_2addr_b32 v[9:10], v93 offset0:128 offset1:160
	s_wait_dscnt 0x0
	ds_bpermute_b32 v13, v92, v9
	ds_bpermute_b32 v14, v77, v9
	ds_bpermute_b32 v15, v78, v9
	ds_bpermute_b32 v16, v79, v9
	ds_bpermute_b32 v17, v80, v9
	ds_bpermute_b32 v18, v81, v9
	ds_bpermute_b32 v19, v82, v9
	ds_bpermute_b32 v20, v83, v9
	ds_bpermute_b32 v21, v84, v9
	ds_bpermute_b32 v22, v85, v9
	ds_bpermute_b32 v23, v86, v9
	ds_bpermute_b32 v24, v87, v9
	ds_bpermute_b32 v102, v88, v9
	ds_bpermute_b32 v103, v89, v9
	ds_bpermute_b32 v104, v90, v9
	ds_bpermute_b32 v105, v91, v9
	ds_bpermute_b32 v106, v92, v10
	ds_bpermute_b32 v107, v77, v10
	ds_bpermute_b32 v108, v78, v10
	ds_bpermute_b32 v109, v79, v10
	ds_bpermute_b32 v110, v80, v10
	ds_bpermute_b32 v111, v81, v10
	ds_bpermute_b32 v112, v82, v10
	ds_bpermute_b32 v113, v83, v10
	ds_bpermute_b32 v114, v84, v10
	ds_bpermute_b32 v115, v85, v10
	ds_bpermute_b32 v116, v86, v10
	ds_bpermute_b32 v117, v87, v10
	ds_bpermute_b32 v118, v88, v10
	ds_bpermute_b32 v119, v89, v10
	ds_bpermute_b32 v120, v90, v10
	ds_bpermute_b32 v121, v91, v10
	s_wait_dscnt 0x1f
	v_fma_f32 v9, -v122, v13, v33
	s_wait_dscnt 0x1e
	v_fma_f32 v10, -v122, v14, v34
	s_wait_dscnt 0x1d
	v_fma_f32 v11, -v122, v15, v35
	s_wait_dscnt 0x1c
	v_fma_f32 v12, -v122, v16, v36
	s_wait_dscnt 0x1b
	v_fma_f32 v13, -v122, v17, v37
	s_wait_dscnt 0x1a
	v_fma_f32 v14, -v122, v18, v38
	s_wait_dscnt 0x19
	v_fma_f32 v15, -v122, v19, v39
	s_wait_dscnt 0x18
	v_fma_f32 v16, -v122, v20, v40
	s_wait_dscnt 0x17
	v_fma_f32 v17, -v122, v21, v25
	s_wait_dscnt 0x16
	v_fma_f32 v18, -v122, v22, v26
	s_wait_dscnt 0x15
	v_fma_f32 v19, -v122, v23, v27
	s_wait_dscnt 0x14
	v_fma_f32 v20, -v122, v24, v28
	s_wait_dscnt 0x13
	v_fma_f32 v21, -v122, v102, v29
	s_wait_dscnt 0x12
	v_fma_f32 v22, -v122, v103, v30
	s_wait_dscnt 0x11
	v_fma_f32 v23, -v122, v104, v31
	s_wait_dscnt 0x10
	v_fma_f32 v24, -v122, v105, v32
	s_wait_dscnt 0xf
	v_fma_f32 v25, -v122, v106, v41
	s_wait_dscnt 0xe
	v_fma_f32 v26, -v122, v107, v42
	s_wait_dscnt 0xd
	v_fma_f32 v27, -v122, v108, v43
	s_wait_dscnt 0xc
	v_fma_f32 v28, -v122, v109, v44
	s_wait_dscnt 0xb
	v_fma_f32 v29, -v122, v110, v45
	s_wait_dscnt 0xa
	v_fma_f32 v30, -v122, v111, v46
	s_wait_dscnt 0x9
	v_fma_f32 v31, -v122, v112, v47
	s_wait_dscnt 0x8
	v_fma_f32 v32, -v122, v113, v48
	s_wait_dscnt 0x7
	v_fma_f32 v33, -v122, v114, v94
	s_wait_dscnt 0x6
	v_fma_f32 v34, -v122, v115, v95
	s_wait_dscnt 0x5
	v_fma_f32 v35, -v122, v116, v96
	s_wait_dscnt 0x4
	v_fma_f32 v36, -v122, v117, v97
	s_wait_dscnt 0x3
	v_fma_f32 v37, -v122, v118, v98
	s_wait_dscnt 0x2
	v_fma_f32 v38, -v122, v119, v99
	s_wait_dscnt 0x1
	v_fma_f32 v39, -v122, v120, v100
	s_wait_dscnt 0x0
	v_fma_f32 v40, -v122, v121, v101
	;;#ASMSTART
	;;#ASMEND
	ds_load_2addr_b32 v[41:42], v93 offset0:192 offset1:224
	v_cndmask_b32_e64 v43, 0, v12, s1
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v43, v43, v28, s2
	ds_bpermute_b32 v44, v76, v43
	s_wait_dscnt 0x1
	ds_bpermute_b32 v45, v92, v41
	ds_bpermute_b32 v46, v77, v41
	ds_bpermute_b32 v47, v78, v41
	ds_bpermute_b32 v48, v79, v41
	ds_bpermute_b32 v94, v80, v41
	ds_bpermute_b32 v95, v81, v41
	ds_bpermute_b32 v96, v82, v41
	ds_bpermute_b32 v97, v83, v41
	ds_bpermute_b32 v98, v84, v41
	ds_bpermute_b32 v99, v85, v41
	ds_bpermute_b32 v100, v86, v41
	ds_bpermute_b32 v101, v87, v41
	ds_bpermute_b32 v102, v88, v41
	ds_bpermute_b32 v103, v89, v41
	ds_bpermute_b32 v104, v90, v41
	ds_bpermute_b32 v41, v91, v41
	ds_bpermute_b32 v105, v92, v42
	ds_bpermute_b32 v106, v77, v42
	ds_bpermute_b32 v107, v78, v42
	ds_bpermute_b32 v108, v79, v42
	ds_bpermute_b32 v109, v80, v42
	ds_bpermute_b32 v110, v81, v42
	ds_bpermute_b32 v111, v82, v42
	ds_bpermute_b32 v112, v83, v42
	ds_bpermute_b32 v113, v84, v42
	ds_bpermute_b32 v114, v85, v42
	ds_bpermute_b32 v115, v86, v42
	ds_bpermute_b32 v116, v87, v42
	ds_bpermute_b32 v117, v88, v42
	ds_bpermute_b32 v118, v89, v42
	ds_bpermute_b32 v119, v90, v42
	ds_bpermute_b32 v42, v91, v42
	s_wait_dscnt 0x20
	v_dual_add_f32 v43, v43, v44 :: v_dual_add_nc_u32 v44, 0x400, v93
	s_wait_dscnt 0x1f
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v9, -v43, v45, v9
	s_wait_dscnt 0x1e
	v_fma_f32 v10, -v43, v46, v10
	s_wait_dscnt 0x1d
	v_fma_f32 v11, -v43, v47, v11
	s_wait_dscnt 0x1c
	v_fma_f32 v12, -v43, v48, v12
	s_wait_dscnt 0x1b
	v_fma_f32 v13, -v43, v94, v13
	s_wait_dscnt 0x1a
	v_fma_f32 v14, -v43, v95, v14
	s_wait_dscnt 0x19
	v_fma_f32 v15, -v43, v96, v15
	s_wait_dscnt 0x18
	v_fma_f32 v16, -v43, v97, v16
	s_wait_dscnt 0x17
	v_fma_f32 v17, -v43, v98, v17
	s_wait_dscnt 0x16
	v_fma_f32 v18, -v43, v99, v18
	s_wait_dscnt 0x15
	v_fma_f32 v19, -v43, v100, v19
	s_wait_dscnt 0x14
	v_fma_f32 v20, -v43, v101, v20
	s_wait_dscnt 0x13
	v_fma_f32 v21, -v43, v102, v21
	s_wait_dscnt 0x12
	v_fma_f32 v22, -v43, v103, v22
	s_wait_dscnt 0x11
	v_fma_f32 v23, -v43, v104, v23
	s_wait_dscnt 0x10
	v_fma_f32 v24, -v43, v41, v24
	s_wait_dscnt 0xf
	v_fma_f32 v25, -v43, v105, v25
	s_wait_dscnt 0xe
	v_fma_f32 v26, -v43, v106, v26
	s_wait_dscnt 0xd
	v_fma_f32 v27, -v43, v107, v27
	s_wait_dscnt 0xc
	v_fma_f32 v28, -v43, v108, v28
	s_wait_dscnt 0xb
	v_fma_f32 v29, -v43, v109, v29
	s_wait_dscnt 0xa
	v_fma_f32 v30, -v43, v110, v30
	s_wait_dscnt 0x9
	v_fma_f32 v31, -v43, v111, v31
	s_wait_dscnt 0x8
	v_fma_f32 v32, -v43, v112, v32
	s_wait_dscnt 0x7
	v_fma_f32 v33, -v43, v113, v33
	s_wait_dscnt 0x6
	v_fma_f32 v34, -v43, v114, v34
	s_wait_dscnt 0x5
	v_fma_f32 v35, -v43, v115, v35
	s_wait_dscnt 0x4
	v_fma_f32 v36, -v43, v116, v36
	s_wait_dscnt 0x3
	v_fma_f32 v37, -v43, v117, v37
	s_wait_dscnt 0x2
	v_fma_f32 v38, -v43, v118, v38
	s_wait_dscnt 0x1
	v_fma_f32 v39, -v43, v119, v39
	s_wait_dscnt 0x0
	v_fma_f32 v40, -v43, v42, v40
	;;#ASMSTART
	;;#ASMEND
	ds_load_2addr_b32 v[41:42], v44 offset1:32
	v_cndmask_b32_e64 v43, 0, v13, s1
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v43, v43, v29, s2
	ds_bpermute_b32 v45, v76, v43
	s_wait_dscnt 0x1
	ds_bpermute_b32 v46, v92, v41
	ds_bpermute_b32 v47, v77, v41
	ds_bpermute_b32 v48, v78, v41
	ds_bpermute_b32 v94, v79, v41
	ds_bpermute_b32 v95, v80, v41
	ds_bpermute_b32 v96, v81, v41
	ds_bpermute_b32 v97, v82, v41
	ds_bpermute_b32 v98, v83, v41
	ds_bpermute_b32 v99, v84, v41
	ds_bpermute_b32 v100, v85, v41
	ds_bpermute_b32 v101, v86, v41
	ds_bpermute_b32 v102, v87, v41
	ds_bpermute_b32 v103, v88, v41
	ds_bpermute_b32 v104, v89, v41
	ds_bpermute_b32 v105, v90, v41
	ds_bpermute_b32 v41, v91, v41
	ds_bpermute_b32 v106, v92, v42
	ds_bpermute_b32 v107, v77, v42
	ds_bpermute_b32 v108, v78, v42
	ds_bpermute_b32 v109, v79, v42
	ds_bpermute_b32 v110, v80, v42
	ds_bpermute_b32 v111, v81, v42
	ds_bpermute_b32 v112, v82, v42
	ds_bpermute_b32 v113, v83, v42
	ds_bpermute_b32 v114, v84, v42
	ds_bpermute_b32 v115, v85, v42
	ds_bpermute_b32 v116, v86, v42
	ds_bpermute_b32 v117, v87, v42
	ds_bpermute_b32 v118, v88, v42
	ds_bpermute_b32 v119, v89, v42
	ds_bpermute_b32 v120, v90, v42
	ds_bpermute_b32 v42, v91, v42
	s_wait_dscnt 0x20
	v_add_f32_e32 v43, v43, v45
	s_wait_dscnt 0x1f
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v9, -v43, v46, v9
	s_wait_dscnt 0x1e
	v_fma_f32 v10, -v43, v47, v10
	s_wait_dscnt 0x1d
	v_fma_f32 v11, -v43, v48, v11
	s_wait_dscnt 0x1c
	v_fma_f32 v12, -v43, v94, v12
	s_wait_dscnt 0x1b
	v_fma_f32 v13, -v43, v95, v13
	s_wait_dscnt 0x1a
	v_fma_f32 v14, -v43, v96, v14
	s_wait_dscnt 0x19
	v_fma_f32 v15, -v43, v97, v15
	s_wait_dscnt 0x18
	v_fma_f32 v16, -v43, v98, v16
	s_wait_dscnt 0x17
	v_fma_f32 v17, -v43, v99, v17
	s_wait_dscnt 0x16
	v_fma_f32 v18, -v43, v100, v18
	s_wait_dscnt 0x15
	v_fma_f32 v19, -v43, v101, v19
	s_wait_dscnt 0x14
	v_fma_f32 v20, -v43, v102, v20
	s_wait_dscnt 0x13
	v_fma_f32 v21, -v43, v103, v21
	s_wait_dscnt 0x12
	v_fma_f32 v22, -v43, v104, v22
	s_wait_dscnt 0x11
	v_fma_f32 v23, -v43, v105, v23
	s_wait_dscnt 0x10
	v_fma_f32 v24, -v43, v41, v24
	s_wait_dscnt 0xf
	v_fma_f32 v25, -v43, v106, v25
	s_wait_dscnt 0xe
	v_fma_f32 v26, -v43, v107, v26
	s_wait_dscnt 0xd
	v_fma_f32 v27, -v43, v108, v27
	s_wait_dscnt 0xc
	v_fma_f32 v28, -v43, v109, v28
	s_wait_dscnt 0xb
	v_fma_f32 v29, -v43, v110, v29
	s_wait_dscnt 0xa
	v_fma_f32 v30, -v43, v111, v30
	s_wait_dscnt 0x9
	v_fma_f32 v31, -v43, v112, v31
	s_wait_dscnt 0x8
	v_fma_f32 v32, -v43, v113, v32
	s_wait_dscnt 0x7
	v_fma_f32 v33, -v43, v114, v33
	s_wait_dscnt 0x6
	v_fma_f32 v34, -v43, v115, v34
	s_wait_dscnt 0x5
	v_fma_f32 v35, -v43, v116, v35
	s_wait_dscnt 0x4
	v_fma_f32 v36, -v43, v117, v36
	s_wait_dscnt 0x3
	v_fma_f32 v37, -v43, v118, v37
	s_wait_dscnt 0x2
	v_fma_f32 v38, -v43, v119, v38
	s_wait_dscnt 0x1
	v_fma_f32 v39, -v43, v120, v39
	s_wait_dscnt 0x0
	v_fma_f32 v40, -v43, v42, v40
	;;#ASMSTART
	;;#ASMEND
	ds_load_2addr_b32 v[41:42], v44 offset0:64 offset1:96
	v_cndmask_b32_e64 v43, 0, v14, s1
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v43, v43, v30, s2
	ds_bpermute_b32 v45, v76, v43
	s_wait_dscnt 0x1
	ds_bpermute_b32 v46, v92, v41
	ds_bpermute_b32 v47, v77, v41
	ds_bpermute_b32 v48, v78, v41
	ds_bpermute_b32 v94, v79, v41
	ds_bpermute_b32 v95, v80, v41
	ds_bpermute_b32 v96, v81, v41
	ds_bpermute_b32 v97, v82, v41
	ds_bpermute_b32 v98, v83, v41
	ds_bpermute_b32 v99, v84, v41
	ds_bpermute_b32 v100, v85, v41
	ds_bpermute_b32 v101, v86, v41
	ds_bpermute_b32 v102, v87, v41
	ds_bpermute_b32 v103, v88, v41
	ds_bpermute_b32 v104, v89, v41
	ds_bpermute_b32 v105, v90, v41
	ds_bpermute_b32 v41, v91, v41
	ds_bpermute_b32 v106, v92, v42
	ds_bpermute_b32 v107, v77, v42
	ds_bpermute_b32 v108, v78, v42
	ds_bpermute_b32 v109, v79, v42
	ds_bpermute_b32 v110, v80, v42
	ds_bpermute_b32 v111, v81, v42
	ds_bpermute_b32 v112, v82, v42
	ds_bpermute_b32 v113, v83, v42
	ds_bpermute_b32 v114, v84, v42
	ds_bpermute_b32 v115, v85, v42
	ds_bpermute_b32 v116, v86, v42
	ds_bpermute_b32 v117, v87, v42
	ds_bpermute_b32 v118, v88, v42
	ds_bpermute_b32 v119, v89, v42
	ds_bpermute_b32 v120, v90, v42
	ds_bpermute_b32 v42, v91, v42
	s_wait_dscnt 0x20
	v_add_f32_e32 v43, v43, v45
	s_wait_dscnt 0x1f
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v9, -v43, v46, v9
	s_wait_dscnt 0x1e
	v_fma_f32 v10, -v43, v47, v10
	s_wait_dscnt 0x1d
	v_fma_f32 v11, -v43, v48, v11
	s_wait_dscnt 0x1c
	v_fma_f32 v12, -v43, v94, v12
	s_wait_dscnt 0x1b
	v_fma_f32 v13, -v43, v95, v13
	s_wait_dscnt 0x1a
	v_fma_f32 v14, -v43, v96, v14
	s_wait_dscnt 0x19
	v_fma_f32 v15, -v43, v97, v15
	s_wait_dscnt 0x18
	v_fma_f32 v16, -v43, v98, v16
	s_wait_dscnt 0x17
	v_fma_f32 v17, -v43, v99, v17
	s_wait_dscnt 0x16
	v_fma_f32 v18, -v43, v100, v18
	s_wait_dscnt 0x15
	v_fma_f32 v19, -v43, v101, v19
	s_wait_dscnt 0x14
	v_fma_f32 v20, -v43, v102, v20
	s_wait_dscnt 0x13
	v_fma_f32 v21, -v43, v103, v21
	s_wait_dscnt 0x12
	v_fma_f32 v22, -v43, v104, v22
	s_wait_dscnt 0x11
	v_fma_f32 v23, -v43, v105, v23
	s_wait_dscnt 0x10
	v_fma_f32 v24, -v43, v41, v24
	s_wait_dscnt 0xf
	v_fma_f32 v25, -v43, v106, v25
	s_wait_dscnt 0xe
	v_fma_f32 v26, -v43, v107, v26
	s_wait_dscnt 0xd
	v_fma_f32 v27, -v43, v108, v27
	s_wait_dscnt 0xc
	v_fma_f32 v28, -v43, v109, v28
	s_wait_dscnt 0xb
	v_fma_f32 v29, -v43, v110, v29
	s_wait_dscnt 0xa
	v_fma_f32 v30, -v43, v111, v30
	s_wait_dscnt 0x9
	v_fma_f32 v31, -v43, v112, v31
	s_wait_dscnt 0x8
	v_fma_f32 v32, -v43, v113, v32
	s_wait_dscnt 0x7
	v_fma_f32 v33, -v43, v114, v33
	s_wait_dscnt 0x6
	v_fma_f32 v34, -v43, v115, v34
	s_wait_dscnt 0x5
	v_fma_f32 v35, -v43, v116, v35
	s_wait_dscnt 0x4
	v_fma_f32 v36, -v43, v117, v36
	s_wait_dscnt 0x3
	v_fma_f32 v37, -v43, v118, v37
	s_wait_dscnt 0x2
	v_fma_f32 v38, -v43, v119, v38
	s_wait_dscnt 0x1
	v_fma_f32 v39, -v43, v120, v39
	s_wait_dscnt 0x0
	v_fma_f32 v40, -v43, v42, v40
	;;#ASMSTART
	;;#ASMEND
	ds_load_2addr_b32 v[41:42], v44 offset0:128 offset1:160
	v_cndmask_b32_e64 v43, 0, v15, s1
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v43, v43, v31, s2
	ds_bpermute_b32 v45, v76, v43
	s_wait_dscnt 0x1
	ds_bpermute_b32 v46, v92, v41
	ds_bpermute_b32 v47, v77, v41
	ds_bpermute_b32 v48, v78, v41
	ds_bpermute_b32 v94, v79, v41
	ds_bpermute_b32 v95, v80, v41
	ds_bpermute_b32 v96, v81, v41
	ds_bpermute_b32 v97, v82, v41
	ds_bpermute_b32 v98, v83, v41
	ds_bpermute_b32 v99, v84, v41
	ds_bpermute_b32 v100, v85, v41
	ds_bpermute_b32 v101, v86, v41
	ds_bpermute_b32 v102, v87, v41
	ds_bpermute_b32 v103, v88, v41
	ds_bpermute_b32 v104, v89, v41
	ds_bpermute_b32 v105, v90, v41
	ds_bpermute_b32 v41, v91, v41
	ds_bpermute_b32 v106, v92, v42
	ds_bpermute_b32 v107, v77, v42
	ds_bpermute_b32 v108, v78, v42
	ds_bpermute_b32 v109, v79, v42
	ds_bpermute_b32 v110, v80, v42
	ds_bpermute_b32 v111, v81, v42
	ds_bpermute_b32 v112, v82, v42
	ds_bpermute_b32 v113, v83, v42
	ds_bpermute_b32 v114, v84, v42
	ds_bpermute_b32 v115, v85, v42
	ds_bpermute_b32 v116, v86, v42
	ds_bpermute_b32 v117, v87, v42
	ds_bpermute_b32 v118, v88, v42
	ds_bpermute_b32 v119, v89, v42
	ds_bpermute_b32 v120, v90, v42
	ds_bpermute_b32 v42, v91, v42
	s_wait_dscnt 0x20
	v_add_f32_e32 v43, v43, v45
	s_wait_dscnt 0x1f
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v9, -v43, v46, v9
	s_wait_dscnt 0x1e
	v_fma_f32 v10, -v43, v47, v10
	s_wait_dscnt 0x1d
	v_fma_f32 v11, -v43, v48, v11
	s_wait_dscnt 0x1c
	v_fma_f32 v12, -v43, v94, v12
	s_wait_dscnt 0x1b
	v_fma_f32 v13, -v43, v95, v13
	s_wait_dscnt 0x1a
	v_fma_f32 v14, -v43, v96, v14
	s_wait_dscnt 0x19
	v_fma_f32 v15, -v43, v97, v15
	s_wait_dscnt 0x18
	v_fma_f32 v16, -v43, v98, v16
	s_wait_dscnt 0x17
	v_fma_f32 v17, -v43, v99, v17
	s_wait_dscnt 0x16
	v_fma_f32 v18, -v43, v100, v18
	s_wait_dscnt 0x15
	v_fma_f32 v19, -v43, v101, v19
	s_wait_dscnt 0x14
	v_fma_f32 v20, -v43, v102, v20
	s_wait_dscnt 0x13
	v_fma_f32 v21, -v43, v103, v21
	s_wait_dscnt 0x12
	v_fma_f32 v22, -v43, v104, v22
	s_wait_dscnt 0x11
	v_fma_f32 v23, -v43, v105, v23
	s_wait_dscnt 0x10
	v_fma_f32 v24, -v43, v41, v24
	s_wait_dscnt 0xf
	v_fma_f32 v25, -v43, v106, v25
	s_wait_dscnt 0xe
	v_fma_f32 v26, -v43, v107, v26
	s_wait_dscnt 0xd
	v_fma_f32 v27, -v43, v108, v27
	s_wait_dscnt 0xc
	v_fma_f32 v28, -v43, v109, v28
	s_wait_dscnt 0xb
	v_fma_f32 v29, -v43, v110, v29
	s_wait_dscnt 0xa
	v_fma_f32 v30, -v43, v111, v30
	s_wait_dscnt 0x9
	v_fma_f32 v31, -v43, v112, v31
	s_wait_dscnt 0x8
	v_fma_f32 v32, -v43, v113, v32
	s_wait_dscnt 0x7
	v_fma_f32 v33, -v43, v114, v33
	s_wait_dscnt 0x6
	v_fma_f32 v34, -v43, v115, v34
	s_wait_dscnt 0x5
	v_fma_f32 v35, -v43, v116, v35
	s_wait_dscnt 0x4
	v_fma_f32 v36, -v43, v117, v36
	s_wait_dscnt 0x3
	v_fma_f32 v37, -v43, v118, v37
	s_wait_dscnt 0x2
	v_fma_f32 v38, -v43, v119, v38
	s_wait_dscnt 0x1
	v_fma_f32 v39, -v43, v120, v39
	s_wait_dscnt 0x0
	v_fma_f32 v40, -v43, v42, v40
	;;#ASMSTART
	;;#ASMEND
	ds_load_2addr_b32 v[41:42], v44 offset0:192 offset1:224
	v_cndmask_b32_e64 v43, 0, v16, s1
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v43, v43, v32, s2
	ds_bpermute_b32 v44, v76, v43
	s_wait_dscnt 0x1
	ds_bpermute_b32 v46, v77, v41
	ds_bpermute_b32 v45, v92, v41
	ds_bpermute_b32 v47, v78, v41
	ds_bpermute_b32 v48, v79, v41
	ds_bpermute_b32 v94, v80, v41
	ds_bpermute_b32 v95, v81, v41
	ds_bpermute_b32 v96, v82, v41
	ds_bpermute_b32 v97, v83, v41
	ds_bpermute_b32 v98, v84, v41
	ds_bpermute_b32 v99, v85, v41
	ds_bpermute_b32 v100, v86, v41
	ds_bpermute_b32 v101, v87, v41
	ds_bpermute_b32 v102, v88, v41
	ds_bpermute_b32 v103, v89, v41
	ds_bpermute_b32 v104, v90, v41
	ds_bpermute_b32 v41, v91, v41
	ds_bpermute_b32 v105, v92, v42
	ds_bpermute_b32 v106, v77, v42
	ds_bpermute_b32 v107, v78, v42
	ds_bpermute_b32 v108, v79, v42
	ds_bpermute_b32 v109, v80, v42
	ds_bpermute_b32 v110, v81, v42
	ds_bpermute_b32 v111, v82, v42
	ds_bpermute_b32 v112, v83, v42
	ds_bpermute_b32 v113, v84, v42
	ds_bpermute_b32 v114, v85, v42
	ds_bpermute_b32 v115, v86, v42
	ds_bpermute_b32 v116, v87, v42
	ds_bpermute_b32 v117, v88, v42
	ds_bpermute_b32 v118, v89, v42
	ds_bpermute_b32 v119, v90, v42
	ds_bpermute_b32 v42, v91, v42
	s_wait_dscnt 0x20
	v_dual_add_f32 v43, v43, v44 :: v_dual_add_nc_u32 v44, 0xfffff800, v75
	s_wait_dscnt 0x1f
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v10, -v43, v46, v10
	v_add_nc_u32_e32 v46, 0x800, v93
	s_wait_dscnt 0x1e
	v_fma_f32 v9, -v43, v45, v9
	s_wait_dscnt 0x1d
	v_fma_f32 v11, -v43, v47, v11
	s_wait_dscnt 0x1c
	v_fma_f32 v12, -v43, v48, v12
	s_wait_dscnt 0x1b
	v_fma_f32 v13, -v43, v94, v13
	s_wait_dscnt 0x1a
	v_fma_f32 v14, -v43, v95, v14
	s_wait_dscnt 0x19
	v_fma_f32 v15, -v43, v96, v15
	s_wait_dscnt 0x18
	v_fma_f32 v16, -v43, v97, v16
	s_wait_dscnt 0x17
	v_fma_f32 v17, -v43, v98, v17
	s_wait_dscnt 0x16
	v_fma_f32 v18, -v43, v99, v18
	s_wait_dscnt 0x15
	v_fma_f32 v19, -v43, v100, v19
	s_wait_dscnt 0x14
	v_fma_f32 v20, -v43, v101, v20
	s_wait_dscnt 0x13
	v_fma_f32 v21, -v43, v102, v21
	s_wait_dscnt 0x12
	v_fma_f32 v22, -v43, v103, v22
	s_wait_dscnt 0x11
	v_fma_f32 v23, -v43, v104, v23
	s_wait_dscnt 0x10
	v_fma_f32 v24, -v43, v41, v24
	s_wait_dscnt 0xf
	v_fma_f32 v25, -v43, v105, v25
	s_wait_dscnt 0xe
	v_fma_f32 v26, -v43, v106, v26
	s_wait_dscnt 0xd
	v_fma_f32 v27, -v43, v107, v27
	s_wait_dscnt 0xc
	v_fma_f32 v28, -v43, v108, v28
	s_wait_dscnt 0xb
	v_fma_f32 v29, -v43, v109, v29
	s_wait_dscnt 0xa
	v_fma_f32 v30, -v43, v110, v30
	s_wait_dscnt 0x9
	v_fma_f32 v31, -v43, v111, v31
	s_wait_dscnt 0x8
	v_fma_f32 v32, -v43, v112, v32
	s_wait_dscnt 0x7
	v_fma_f32 v33, -v43, v113, v33
	s_wait_dscnt 0x6
	v_fma_f32 v34, -v43, v114, v34
	s_wait_dscnt 0x5
	v_fma_f32 v35, -v43, v115, v35
	s_wait_dscnt 0x4
	v_fma_f32 v36, -v43, v116, v36
	s_wait_dscnt 0x3
	v_fma_f32 v37, -v43, v117, v37
	s_wait_dscnt 0x2
	v_fma_f32 v38, -v43, v118, v38
	s_wait_dscnt 0x1
	v_fma_f32 v39, -v43, v119, v39
	s_wait_dscnt 0x0
	v_fma_f32 v40, -v43, v42, v40
	;;#ASMSTART
	;;#ASMEND
	v_add_nc_u32_e32 v45, 0x1800, v75
	v_cmp_eq_u32_e32 vcc_lo, s8, v44
	ds_load_2addr_b32 v[41:42], v46 offset1:32
	v_cmp_eq_u32_e64 s0, s8, v45
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v43, 0, v9, vcc_lo
	s_addk_co_i32 s8, 0x2000
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s8, 0x4000
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v43, v43, v25, s0
	ds_bpermute_b32 v44, v76, v43
	s_wait_dscnt 0x1
	ds_bpermute_b32 v45, v92, v41
	ds_bpermute_b32 v47, v77, v41
	ds_bpermute_b32 v48, v78, v41
	ds_bpermute_b32 v94, v79, v41
	ds_bpermute_b32 v95, v80, v41
	ds_bpermute_b32 v96, v81, v41
	ds_bpermute_b32 v97, v82, v41
	ds_bpermute_b32 v98, v83, v41
	ds_bpermute_b32 v99, v84, v41
	ds_bpermute_b32 v100, v85, v41
	ds_bpermute_b32 v101, v86, v41
	ds_bpermute_b32 v102, v87, v41
	ds_bpermute_b32 v103, v88, v41
	ds_bpermute_b32 v104, v89, v41
	ds_bpermute_b32 v105, v90, v41
	ds_bpermute_b32 v41, v91, v41
	ds_bpermute_b32 v106, v92, v42
	ds_bpermute_b32 v107, v77, v42
	ds_bpermute_b32 v108, v78, v42
	ds_bpermute_b32 v109, v79, v42
	ds_bpermute_b32 v110, v80, v42
	ds_bpermute_b32 v111, v81, v42
	ds_bpermute_b32 v112, v82, v42
	ds_bpermute_b32 v113, v83, v42
	ds_bpermute_b32 v114, v84, v42
	ds_bpermute_b32 v115, v85, v42
	ds_bpermute_b32 v116, v86, v42
	ds_bpermute_b32 v117, v87, v42
	ds_bpermute_b32 v118, v88, v42
	ds_bpermute_b32 v119, v89, v42
	ds_bpermute_b32 v120, v90, v42
	ds_bpermute_b32 v42, v91, v42
	s_wait_dscnt 0x20
	v_add_f32_e32 v43, v43, v44
	s_wait_dscnt 0x1f
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v9, -v43, v45, v9
	s_wait_dscnt 0x1e
	v_fma_f32 v10, -v43, v47, v10
	s_wait_dscnt 0x1d
	v_fma_f32 v11, -v43, v48, v11
	s_wait_dscnt 0x1c
	v_fma_f32 v12, -v43, v94, v12
	s_wait_dscnt 0x1b
	v_fma_f32 v13, -v43, v95, v13
	s_wait_dscnt 0x1a
	v_fma_f32 v14, -v43, v96, v14
	s_wait_dscnt 0x19
	v_fma_f32 v15, -v43, v97, v15
	s_wait_dscnt 0x18
	v_fma_f32 v16, -v43, v98, v16
	s_wait_dscnt 0x17
	v_fma_f32 v17, -v43, v99, v17
	s_wait_dscnt 0x16
	v_fma_f32 v18, -v43, v100, v18
	s_wait_dscnt 0x15
	v_fma_f32 v19, -v43, v101, v19
	s_wait_dscnt 0x14
	v_fma_f32 v20, -v43, v102, v20
	s_wait_dscnt 0x13
	v_fma_f32 v21, -v43, v103, v21
	s_wait_dscnt 0x12
	v_fma_f32 v22, -v43, v104, v22
	s_wait_dscnt 0x11
	v_fma_f32 v23, -v43, v105, v23
	s_wait_dscnt 0x10
	v_fma_f32 v24, -v43, v41, v24
	s_wait_dscnt 0xf
	v_fma_f32 v25, -v43, v106, v25
	s_wait_dscnt 0xe
	v_fma_f32 v26, -v43, v107, v26
	s_wait_dscnt 0xd
	v_fma_f32 v27, -v43, v108, v27
	s_wait_dscnt 0xc
	v_fma_f32 v28, -v43, v109, v28
	s_wait_dscnt 0xb
	v_fma_f32 v29, -v43, v110, v29
	s_wait_dscnt 0xa
	v_fma_f32 v30, -v43, v111, v30
	s_wait_dscnt 0x9
	v_fma_f32 v31, -v43, v112, v31
	s_wait_dscnt 0x8
	v_fma_f32 v32, -v43, v113, v32
	s_wait_dscnt 0x7
	v_fma_f32 v33, -v43, v114, v33
	s_wait_dscnt 0x6
	v_fma_f32 v34, -v43, v115, v34
	s_wait_dscnt 0x5
	v_fma_f32 v35, -v43, v116, v35
	s_wait_dscnt 0x4
	v_fma_f32 v36, -v43, v117, v36
	s_wait_dscnt 0x3
	v_fma_f32 v37, -v43, v118, v37
	s_wait_dscnt 0x2
	v_fma_f32 v38, -v43, v119, v38
	s_wait_dscnt 0x1
	v_fma_f32 v39, -v43, v120, v39
	s_wait_dscnt 0x0
	v_fma_f32 v40, -v43, v42, v40
	;;#ASMSTART
	;;#ASMEND
	ds_load_2addr_b32 v[41:42], v46 offset0:64 offset1:96
	v_cndmask_b32_e32 v43, 0, v10, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v43, v43, v26, s0
	ds_bpermute_b32 v44, v76, v43
	s_wait_dscnt 0x1
	ds_bpermute_b32 v45, v92, v41
	ds_bpermute_b32 v47, v77, v41
	ds_bpermute_b32 v48, v78, v41
	ds_bpermute_b32 v94, v79, v41
	ds_bpermute_b32 v95, v80, v41
	ds_bpermute_b32 v96, v81, v41
	ds_bpermute_b32 v97, v82, v41
	ds_bpermute_b32 v98, v83, v41
	ds_bpermute_b32 v99, v84, v41
	ds_bpermute_b32 v100, v85, v41
	ds_bpermute_b32 v101, v86, v41
	ds_bpermute_b32 v102, v87, v41
	ds_bpermute_b32 v103, v88, v41
	ds_bpermute_b32 v104, v89, v41
	ds_bpermute_b32 v105, v90, v41
	ds_bpermute_b32 v41, v91, v41
	ds_bpermute_b32 v106, v92, v42
	ds_bpermute_b32 v107, v77, v42
	ds_bpermute_b32 v108, v78, v42
	ds_bpermute_b32 v109, v79, v42
	ds_bpermute_b32 v110, v80, v42
	ds_bpermute_b32 v111, v81, v42
	ds_bpermute_b32 v112, v82, v42
	ds_bpermute_b32 v113, v83, v42
	ds_bpermute_b32 v114, v84, v42
	ds_bpermute_b32 v115, v85, v42
	ds_bpermute_b32 v116, v86, v42
	ds_bpermute_b32 v117, v87, v42
	ds_bpermute_b32 v118, v88, v42
	ds_bpermute_b32 v119, v89, v42
	ds_bpermute_b32 v120, v90, v42
	ds_bpermute_b32 v42, v91, v42
	s_wait_dscnt 0x20
	v_add_f32_e32 v43, v43, v44
	s_wait_dscnt 0x1f
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v9, -v43, v45, v9
	s_wait_dscnt 0x1e
	v_fma_f32 v10, -v43, v47, v10
	s_wait_dscnt 0x1d
	v_fma_f32 v11, -v43, v48, v11
	s_wait_dscnt 0x1c
	v_fma_f32 v12, -v43, v94, v12
	s_wait_dscnt 0x1b
	v_fma_f32 v13, -v43, v95, v13
	s_wait_dscnt 0x1a
	v_fma_f32 v14, -v43, v96, v14
	s_wait_dscnt 0x19
	v_fma_f32 v15, -v43, v97, v15
	s_wait_dscnt 0x18
	v_fma_f32 v16, -v43, v98, v16
	s_wait_dscnt 0x17
	v_fma_f32 v17, -v43, v99, v17
	s_wait_dscnt 0x16
	v_fma_f32 v18, -v43, v100, v18
	s_wait_dscnt 0x15
	v_fma_f32 v19, -v43, v101, v19
	s_wait_dscnt 0x14
	v_fma_f32 v20, -v43, v102, v20
	s_wait_dscnt 0x13
	v_fma_f32 v21, -v43, v103, v21
	s_wait_dscnt 0x12
	v_fma_f32 v22, -v43, v104, v22
	s_wait_dscnt 0x11
	v_fma_f32 v23, -v43, v105, v23
	s_wait_dscnt 0x10
	v_fma_f32 v24, -v43, v41, v24
	s_wait_dscnt 0xf
	v_fma_f32 v25, -v43, v106, v25
	s_wait_dscnt 0xe
	v_fma_f32 v26, -v43, v107, v26
	s_wait_dscnt 0xd
	v_fma_f32 v27, -v43, v108, v27
	s_wait_dscnt 0xc
	v_fma_f32 v28, -v43, v109, v28
	s_wait_dscnt 0xb
	v_fma_f32 v29, -v43, v110, v29
	s_wait_dscnt 0xa
	v_fma_f32 v30, -v43, v111, v30
	s_wait_dscnt 0x9
	v_fma_f32 v31, -v43, v112, v31
	s_wait_dscnt 0x8
	v_fma_f32 v32, -v43, v113, v32
	s_wait_dscnt 0x7
	v_fma_f32 v33, -v43, v114, v33
	s_wait_dscnt 0x6
	v_fma_f32 v34, -v43, v115, v34
	s_wait_dscnt 0x5
	v_fma_f32 v35, -v43, v116, v35
	s_wait_dscnt 0x4
	v_fma_f32 v36, -v43, v117, v36
	s_wait_dscnt 0x3
	v_fma_f32 v37, -v43, v118, v37
	s_wait_dscnt 0x2
	v_fma_f32 v38, -v43, v119, v38
	s_wait_dscnt 0x1
	v_fma_f32 v39, -v43, v120, v39
	s_wait_dscnt 0x0
	v_fma_f32 v40, -v43, v42, v40
	;;#ASMSTART
	;;#ASMEND
	ds_load_2addr_b32 v[41:42], v46 offset0:128 offset1:160
	v_cndmask_b32_e32 v43, 0, v11, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v43, v43, v27, s0
	ds_bpermute_b32 v44, v76, v43
	s_wait_dscnt 0x1
	ds_bpermute_b32 v45, v92, v41
	ds_bpermute_b32 v47, v77, v41
	ds_bpermute_b32 v48, v78, v41
	ds_bpermute_b32 v94, v79, v41
	ds_bpermute_b32 v95, v80, v41
	ds_bpermute_b32 v96, v81, v41
	ds_bpermute_b32 v97, v82, v41
	ds_bpermute_b32 v98, v83, v41
	ds_bpermute_b32 v99, v84, v41
	ds_bpermute_b32 v100, v85, v41
	ds_bpermute_b32 v101, v86, v41
	ds_bpermute_b32 v102, v87, v41
	ds_bpermute_b32 v103, v88, v41
	ds_bpermute_b32 v104, v89, v41
	ds_bpermute_b32 v105, v90, v41
	ds_bpermute_b32 v41, v91, v41
	ds_bpermute_b32 v106, v92, v42
	ds_bpermute_b32 v107, v77, v42
	ds_bpermute_b32 v108, v78, v42
	ds_bpermute_b32 v109, v79, v42
	ds_bpermute_b32 v110, v80, v42
	ds_bpermute_b32 v111, v81, v42
	ds_bpermute_b32 v112, v82, v42
	ds_bpermute_b32 v113, v83, v42
	ds_bpermute_b32 v114, v84, v42
	ds_bpermute_b32 v115, v85, v42
	ds_bpermute_b32 v116, v86, v42
	ds_bpermute_b32 v117, v87, v42
	ds_bpermute_b32 v118, v88, v42
	ds_bpermute_b32 v119, v89, v42
	ds_bpermute_b32 v120, v90, v42
	ds_bpermute_b32 v42, v91, v42
	s_wait_dscnt 0x20
	v_add_f32_e32 v43, v43, v44
	s_wait_dscnt 0x1f
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v9, -v43, v45, v9
	s_wait_dscnt 0x1e
	v_fma_f32 v10, -v43, v47, v10
	s_wait_dscnt 0x1d
	v_fma_f32 v11, -v43, v48, v11
	s_wait_dscnt 0x1c
	v_fma_f32 v12, -v43, v94, v12
	s_wait_dscnt 0x1b
	v_fma_f32 v13, -v43, v95, v13
	s_wait_dscnt 0x1a
	v_fma_f32 v14, -v43, v96, v14
	s_wait_dscnt 0x19
	v_fma_f32 v15, -v43, v97, v15
	s_wait_dscnt 0x18
	v_fma_f32 v16, -v43, v98, v16
	s_wait_dscnt 0x17
	v_fma_f32 v17, -v43, v99, v17
	s_wait_dscnt 0x16
	v_fma_f32 v18, -v43, v100, v18
	s_wait_dscnt 0x15
	v_fma_f32 v19, -v43, v101, v19
	s_wait_dscnt 0x14
	v_fma_f32 v20, -v43, v102, v20
	s_wait_dscnt 0x13
	v_fma_f32 v21, -v43, v103, v21
	s_wait_dscnt 0x12
	v_fma_f32 v22, -v43, v104, v22
	s_wait_dscnt 0x11
	v_fma_f32 v23, -v43, v105, v23
	s_wait_dscnt 0x10
	v_fma_f32 v24, -v43, v41, v24
	s_wait_dscnt 0xf
	v_fma_f32 v25, -v43, v106, v25
	s_wait_dscnt 0xe
	v_fma_f32 v26, -v43, v107, v26
	s_wait_dscnt 0xd
	v_fma_f32 v27, -v43, v108, v27
	s_wait_dscnt 0xc
	v_fma_f32 v28, -v43, v109, v28
	s_wait_dscnt 0xb
	v_fma_f32 v29, -v43, v110, v29
	s_wait_dscnt 0xa
	v_fma_f32 v30, -v43, v111, v30
	s_wait_dscnt 0x9
	v_fma_f32 v31, -v43, v112, v31
	s_wait_dscnt 0x8
	v_fma_f32 v32, -v43, v113, v32
	s_wait_dscnt 0x7
	v_fma_f32 v33, -v43, v114, v33
	s_wait_dscnt 0x6
	v_fma_f32 v34, -v43, v115, v34
	s_wait_dscnt 0x5
	v_fma_f32 v35, -v43, v116, v35
	s_wait_dscnt 0x4
	v_fma_f32 v36, -v43, v117, v36
	s_wait_dscnt 0x3
	v_fma_f32 v37, -v43, v118, v37
	s_wait_dscnt 0x2
	v_fma_f32 v38, -v43, v119, v38
	s_wait_dscnt 0x1
	v_fma_f32 v39, -v43, v120, v39
	s_wait_dscnt 0x0
	v_fma_f32 v40, -v43, v42, v40
	;;#ASMSTART
	;;#ASMEND
	ds_load_2addr_b32 v[41:42], v46 offset0:192 offset1:224
	v_cndmask_b32_e32 v43, 0, v12, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v43, v43, v28, s0
	ds_bpermute_b32 v44, v76, v43
	s_wait_dscnt 0x1
	ds_bpermute_b32 v45, v92, v41
	ds_bpermute_b32 v46, v77, v41
	ds_bpermute_b32 v47, v78, v41
	ds_bpermute_b32 v48, v79, v41
	ds_bpermute_b32 v94, v80, v41
	ds_bpermute_b32 v95, v81, v41
	ds_bpermute_b32 v96, v82, v41
	ds_bpermute_b32 v97, v83, v41
	ds_bpermute_b32 v98, v84, v41
	ds_bpermute_b32 v99, v85, v41
	ds_bpermute_b32 v100, v86, v41
	ds_bpermute_b32 v101, v87, v41
	ds_bpermute_b32 v102, v88, v41
	ds_bpermute_b32 v103, v89, v41
	ds_bpermute_b32 v104, v90, v41
	ds_bpermute_b32 v41, v91, v41
	ds_bpermute_b32 v105, v92, v42
	ds_bpermute_b32 v106, v77, v42
	ds_bpermute_b32 v107, v78, v42
	ds_bpermute_b32 v108, v79, v42
	ds_bpermute_b32 v109, v80, v42
	ds_bpermute_b32 v110, v81, v42
	ds_bpermute_b32 v111, v82, v42
	ds_bpermute_b32 v112, v83, v42
	ds_bpermute_b32 v113, v84, v42
	ds_bpermute_b32 v114, v85, v42
	ds_bpermute_b32 v115, v86, v42
	ds_bpermute_b32 v116, v87, v42
	ds_bpermute_b32 v117, v88, v42
	ds_bpermute_b32 v118, v89, v42
	ds_bpermute_b32 v119, v90, v42
	ds_bpermute_b32 v42, v91, v42
	s_wait_dscnt 0x20
	v_dual_add_f32 v43, v43, v44 :: v_dual_add_nc_u32 v44, 0xc00, v93
	s_wait_dscnt 0x1f
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v9, -v43, v45, v9
	s_wait_dscnt 0x1e
	v_fma_f32 v10, -v43, v46, v10
	s_wait_dscnt 0x1d
	v_fma_f32 v11, -v43, v47, v11
	s_wait_dscnt 0x1c
	v_fma_f32 v12, -v43, v48, v12
	s_wait_dscnt 0x1b
	v_fma_f32 v13, -v43, v94, v13
	s_wait_dscnt 0x1a
	v_fma_f32 v14, -v43, v95, v14
	s_wait_dscnt 0x19
	v_fma_f32 v15, -v43, v96, v15
	s_wait_dscnt 0x18
	v_fma_f32 v16, -v43, v97, v16
	s_wait_dscnt 0x17
	v_fma_f32 v17, -v43, v98, v17
	s_wait_dscnt 0x16
	v_fma_f32 v18, -v43, v99, v18
	s_wait_dscnt 0x15
	v_fma_f32 v19, -v43, v100, v19
	s_wait_dscnt 0x14
	v_fma_f32 v20, -v43, v101, v20
	s_wait_dscnt 0x13
	v_fma_f32 v21, -v43, v102, v21
	s_wait_dscnt 0x12
	v_fma_f32 v22, -v43, v103, v22
	s_wait_dscnt 0x11
	v_fma_f32 v23, -v43, v104, v23
	s_wait_dscnt 0x10
	v_fma_f32 v24, -v43, v41, v24
	s_wait_dscnt 0xf
	v_fma_f32 v25, -v43, v105, v25
	s_wait_dscnt 0xe
	v_fma_f32 v26, -v43, v106, v26
	s_wait_dscnt 0xd
	v_fma_f32 v27, -v43, v107, v27
	s_wait_dscnt 0xc
	v_fma_f32 v28, -v43, v108, v28
	s_wait_dscnt 0xb
	v_fma_f32 v29, -v43, v109, v29
	s_wait_dscnt 0xa
	v_fma_f32 v30, -v43, v110, v30
	s_wait_dscnt 0x9
	v_fma_f32 v31, -v43, v111, v31
	s_wait_dscnt 0x8
	v_fma_f32 v32, -v43, v112, v32
	s_wait_dscnt 0x7
	v_fma_f32 v33, -v43, v113, v33
	s_wait_dscnt 0x6
	v_fma_f32 v34, -v43, v114, v34
	s_wait_dscnt 0x5
	v_fma_f32 v35, -v43, v115, v35
	s_wait_dscnt 0x4
	v_fma_f32 v36, -v43, v116, v36
	s_wait_dscnt 0x3
	v_fma_f32 v37, -v43, v117, v37
	s_wait_dscnt 0x2
	v_fma_f32 v38, -v43, v118, v38
	s_wait_dscnt 0x1
	v_fma_f32 v39, -v43, v119, v39
	s_wait_dscnt 0x0
	v_fma_f32 v40, -v43, v42, v40
	;;#ASMSTART
	;;#ASMEND
	ds_load_2addr_b32 v[41:42], v44 offset1:32
	v_cndmask_b32_e32 v43, 0, v13, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v43, v43, v29, s0
	ds_bpermute_b32 v45, v76, v43
	s_wait_dscnt 0x1
	ds_bpermute_b32 v46, v92, v41
	ds_bpermute_b32 v47, v77, v41
	ds_bpermute_b32 v48, v78, v41
	ds_bpermute_b32 v94, v79, v41
	ds_bpermute_b32 v95, v80, v41
	ds_bpermute_b32 v96, v81, v41
	ds_bpermute_b32 v97, v82, v41
	ds_bpermute_b32 v98, v83, v41
	ds_bpermute_b32 v99, v84, v41
	ds_bpermute_b32 v100, v85, v41
	ds_bpermute_b32 v101, v86, v41
	ds_bpermute_b32 v102, v87, v41
	ds_bpermute_b32 v103, v88, v41
	ds_bpermute_b32 v104, v89, v41
	ds_bpermute_b32 v105, v90, v41
	ds_bpermute_b32 v41, v91, v41
	ds_bpermute_b32 v106, v92, v42
	ds_bpermute_b32 v107, v77, v42
	ds_bpermute_b32 v108, v78, v42
	ds_bpermute_b32 v109, v79, v42
	ds_bpermute_b32 v110, v80, v42
	ds_bpermute_b32 v111, v81, v42
	ds_bpermute_b32 v112, v82, v42
	ds_bpermute_b32 v113, v83, v42
	ds_bpermute_b32 v114, v84, v42
	ds_bpermute_b32 v115, v85, v42
	ds_bpermute_b32 v116, v86, v42
	ds_bpermute_b32 v117, v87, v42
	ds_bpermute_b32 v118, v88, v42
	ds_bpermute_b32 v119, v89, v42
	ds_bpermute_b32 v120, v90, v42
	ds_bpermute_b32 v42, v91, v42
	s_wait_dscnt 0x20
	v_add_f32_e32 v43, v43, v45
	s_wait_dscnt 0x1f
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v9, -v43, v46, v9
	s_wait_dscnt 0x1e
	v_fma_f32 v10, -v43, v47, v10
	s_wait_dscnt 0x1d
	v_fma_f32 v11, -v43, v48, v11
	s_wait_dscnt 0x1c
	v_fma_f32 v12, -v43, v94, v12
	s_wait_dscnt 0x1b
	v_fma_f32 v13, -v43, v95, v13
	s_wait_dscnt 0x1a
	v_fma_f32 v14, -v43, v96, v14
	s_wait_dscnt 0x19
	v_fma_f32 v15, -v43, v97, v15
	s_wait_dscnt 0x18
	v_fma_f32 v16, -v43, v98, v16
	s_wait_dscnt 0x17
	v_fma_f32 v17, -v43, v99, v17
	s_wait_dscnt 0x16
	v_fma_f32 v18, -v43, v100, v18
	s_wait_dscnt 0x15
	v_fma_f32 v19, -v43, v101, v19
	s_wait_dscnt 0x14
	v_fma_f32 v20, -v43, v102, v20
	s_wait_dscnt 0x13
	v_fma_f32 v21, -v43, v103, v21
	s_wait_dscnt 0x12
	v_fma_f32 v22, -v43, v104, v22
	s_wait_dscnt 0x11
	v_fma_f32 v23, -v43, v105, v23
	s_wait_dscnt 0x10
	v_fma_f32 v24, -v43, v41, v24
	s_wait_dscnt 0xf
	v_fma_f32 v25, -v43, v106, v25
	s_wait_dscnt 0xe
	v_fma_f32 v26, -v43, v107, v26
	s_wait_dscnt 0xd
	v_fma_f32 v27, -v43, v108, v27
	s_wait_dscnt 0xc
	v_fma_f32 v28, -v43, v109, v28
	s_wait_dscnt 0xb
	v_fma_f32 v29, -v43, v110, v29
	s_wait_dscnt 0xa
	v_fma_f32 v30, -v43, v111, v30
	s_wait_dscnt 0x9
	v_fma_f32 v31, -v43, v112, v31
	s_wait_dscnt 0x8
	v_fma_f32 v32, -v43, v113, v32
	s_wait_dscnt 0x7
	v_fma_f32 v33, -v43, v114, v33
	s_wait_dscnt 0x6
	v_fma_f32 v34, -v43, v115, v34
	s_wait_dscnt 0x5
	v_fma_f32 v35, -v43, v116, v35
	s_wait_dscnt 0x4
	v_fma_f32 v36, -v43, v117, v36
	s_wait_dscnt 0x3
	v_fma_f32 v37, -v43, v118, v37
	s_wait_dscnt 0x2
	v_fma_f32 v38, -v43, v119, v38
	s_wait_dscnt 0x1
	v_fma_f32 v39, -v43, v120, v39
	s_wait_dscnt 0x0
	v_fma_f32 v40, -v43, v42, v40
	;;#ASMSTART
	;;#ASMEND
	ds_load_2addr_b32 v[41:42], v44 offset0:64 offset1:96
	v_cndmask_b32_e32 v43, 0, v14, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v43, v43, v30, s0
	ds_bpermute_b32 v45, v76, v43
	s_wait_dscnt 0x1
	ds_bpermute_b32 v46, v92, v41
	ds_bpermute_b32 v47, v77, v41
	ds_bpermute_b32 v48, v78, v41
	ds_bpermute_b32 v94, v79, v41
	ds_bpermute_b32 v95, v80, v41
	ds_bpermute_b32 v96, v81, v41
	ds_bpermute_b32 v97, v82, v41
	ds_bpermute_b32 v98, v83, v41
	ds_bpermute_b32 v99, v84, v41
	ds_bpermute_b32 v100, v85, v41
	ds_bpermute_b32 v101, v86, v41
	ds_bpermute_b32 v102, v87, v41
	ds_bpermute_b32 v103, v88, v41
	ds_bpermute_b32 v104, v89, v41
	ds_bpermute_b32 v105, v90, v41
	ds_bpermute_b32 v41, v91, v41
	ds_bpermute_b32 v106, v92, v42
	ds_bpermute_b32 v107, v77, v42
	ds_bpermute_b32 v108, v78, v42
	ds_bpermute_b32 v109, v79, v42
	ds_bpermute_b32 v110, v80, v42
	ds_bpermute_b32 v111, v81, v42
	ds_bpermute_b32 v112, v82, v42
	ds_bpermute_b32 v113, v83, v42
	ds_bpermute_b32 v114, v84, v42
	ds_bpermute_b32 v115, v85, v42
	ds_bpermute_b32 v116, v86, v42
	ds_bpermute_b32 v117, v87, v42
	ds_bpermute_b32 v118, v88, v42
	ds_bpermute_b32 v119, v89, v42
	ds_bpermute_b32 v120, v90, v42
	ds_bpermute_b32 v42, v91, v42
	s_wait_dscnt 0x20
	v_add_f32_e32 v43, v43, v45
	s_wait_dscnt 0x1f
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v9, -v43, v46, v9
	s_wait_dscnt 0x1e
	v_fma_f32 v10, -v43, v47, v10
	s_wait_dscnt 0x1d
	v_fma_f32 v11, -v43, v48, v11
	s_wait_dscnt 0x1c
	v_fma_f32 v12, -v43, v94, v12
	s_wait_dscnt 0x1b
	v_fma_f32 v13, -v43, v95, v13
	s_wait_dscnt 0x1a
	v_fma_f32 v14, -v43, v96, v14
	s_wait_dscnt 0x19
	v_fma_f32 v15, -v43, v97, v15
	s_wait_dscnt 0x18
	v_fma_f32 v16, -v43, v98, v16
	s_wait_dscnt 0x17
	v_fma_f32 v17, -v43, v99, v17
	s_wait_dscnt 0x16
	v_fma_f32 v18, -v43, v100, v18
	s_wait_dscnt 0x15
	v_fma_f32 v19, -v43, v101, v19
	s_wait_dscnt 0x14
	v_fma_f32 v20, -v43, v102, v20
	s_wait_dscnt 0x13
	v_fma_f32 v21, -v43, v103, v21
	s_wait_dscnt 0x12
	v_fma_f32 v22, -v43, v104, v22
	s_wait_dscnt 0x11
	v_fma_f32 v23, -v43, v105, v23
	s_wait_dscnt 0x10
	v_fma_f32 v24, -v43, v41, v24
	s_wait_dscnt 0xf
	v_fma_f32 v25, -v43, v106, v25
	s_wait_dscnt 0xe
	v_fma_f32 v26, -v43, v107, v26
	s_wait_dscnt 0xd
	v_fma_f32 v27, -v43, v108, v27
	s_wait_dscnt 0xc
	v_fma_f32 v28, -v43, v109, v28
	s_wait_dscnt 0xb
	v_fma_f32 v29, -v43, v110, v29
	s_wait_dscnt 0xa
	v_fma_f32 v30, -v43, v111, v30
	s_wait_dscnt 0x9
	v_fma_f32 v31, -v43, v112, v31
	s_wait_dscnt 0x8
	v_fma_f32 v32, -v43, v113, v32
	s_wait_dscnt 0x7
	v_fma_f32 v33, -v43, v114, v33
	s_wait_dscnt 0x6
	v_fma_f32 v34, -v43, v115, v34
	s_wait_dscnt 0x5
	v_fma_f32 v35, -v43, v116, v35
	s_wait_dscnt 0x4
	v_fma_f32 v36, -v43, v117, v36
	s_wait_dscnt 0x3
	v_fma_f32 v37, -v43, v118, v37
	s_wait_dscnt 0x2
	v_fma_f32 v38, -v43, v119, v38
	s_wait_dscnt 0x1
	v_fma_f32 v39, -v43, v120, v39
	s_wait_dscnt 0x0
	v_fma_f32 v40, -v43, v42, v40
	;;#ASMSTART
	;;#ASMEND
	ds_load_2addr_b32 v[41:42], v44 offset0:128 offset1:160
	v_cndmask_b32_e32 v43, 0, v15, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v43, v43, v31, s0
	ds_bpermute_b32 v45, v76, v43
	s_wait_dscnt 0x1
	ds_bpermute_b32 v46, v92, v41
	ds_bpermute_b32 v47, v77, v41
	ds_bpermute_b32 v48, v78, v41
	ds_bpermute_b32 v94, v79, v41
	ds_bpermute_b32 v95, v80, v41
	ds_bpermute_b32 v96, v81, v41
	ds_bpermute_b32 v97, v82, v41
	ds_bpermute_b32 v98, v83, v41
	ds_bpermute_b32 v99, v84, v41
	ds_bpermute_b32 v100, v85, v41
	ds_bpermute_b32 v101, v86, v41
	ds_bpermute_b32 v102, v87, v41
	ds_bpermute_b32 v103, v88, v41
	ds_bpermute_b32 v104, v89, v41
	ds_bpermute_b32 v105, v90, v41
	ds_bpermute_b32 v41, v91, v41
	ds_bpermute_b32 v106, v92, v42
	ds_bpermute_b32 v107, v77, v42
	ds_bpermute_b32 v108, v78, v42
	ds_bpermute_b32 v109, v79, v42
	ds_bpermute_b32 v110, v80, v42
	ds_bpermute_b32 v111, v81, v42
	ds_bpermute_b32 v112, v82, v42
	ds_bpermute_b32 v113, v83, v42
	ds_bpermute_b32 v114, v84, v42
	ds_bpermute_b32 v115, v85, v42
	ds_bpermute_b32 v116, v86, v42
	ds_bpermute_b32 v117, v87, v42
	ds_bpermute_b32 v118, v88, v42
	ds_bpermute_b32 v119, v89, v42
	ds_bpermute_b32 v120, v90, v42
	ds_bpermute_b32 v42, v91, v42
	s_wait_dscnt 0x20
	v_add_f32_e32 v43, v43, v45
	s_wait_dscnt 0x1f
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v9, -v43, v46, v9
	s_wait_dscnt 0x1e
	v_fma_f32 v10, -v43, v47, v10
	s_wait_dscnt 0x1d
	v_fma_f32 v11, -v43, v48, v11
	s_wait_dscnt 0x1c
	v_fma_f32 v12, -v43, v94, v12
	s_wait_dscnt 0x1b
	v_fma_f32 v13, -v43, v95, v13
	s_wait_dscnt 0x1a
	v_fma_f32 v14, -v43, v96, v14
	s_wait_dscnt 0x19
	v_fma_f32 v15, -v43, v97, v15
	s_wait_dscnt 0x18
	v_fma_f32 v16, -v43, v98, v16
	s_wait_dscnt 0x17
	v_fma_f32 v17, -v43, v99, v17
	s_wait_dscnt 0x16
	v_fma_f32 v18, -v43, v100, v18
	s_wait_dscnt 0x15
	v_fma_f32 v19, -v43, v101, v19
	s_wait_dscnt 0x14
	v_fma_f32 v20, -v43, v102, v20
	s_wait_dscnt 0x13
	v_fma_f32 v21, -v43, v103, v21
	s_wait_dscnt 0x12
	v_fma_f32 v22, -v43, v104, v22
	s_wait_dscnt 0x11
	v_fma_f32 v23, -v43, v105, v23
	s_wait_dscnt 0x10
	v_fma_f32 v24, -v43, v41, v24
	s_wait_dscnt 0xf
	v_fma_f32 v25, -v43, v106, v25
	s_wait_dscnt 0xe
	v_fma_f32 v26, -v43, v107, v26
	s_wait_dscnt 0xd
	v_fma_f32 v27, -v43, v108, v27
	s_wait_dscnt 0xc
	v_fma_f32 v28, -v43, v109, v28
	s_wait_dscnt 0xb
	v_fma_f32 v29, -v43, v110, v29
	s_wait_dscnt 0xa
	v_fma_f32 v30, -v43, v111, v30
	s_wait_dscnt 0x9
	v_fma_f32 v31, -v43, v112, v31
	s_wait_dscnt 0x8
	v_fma_f32 v32, -v43, v113, v32
	s_wait_dscnt 0x7
	v_fma_f32 v33, -v43, v114, v33
	s_wait_dscnt 0x6
	v_fma_f32 v34, -v43, v115, v34
	s_wait_dscnt 0x5
	v_fma_f32 v35, -v43, v116, v35
	s_wait_dscnt 0x4
	v_fma_f32 v36, -v43, v117, v36
	s_wait_dscnt 0x3
	v_fma_f32 v37, -v43, v118, v37
	s_wait_dscnt 0x2
	v_fma_f32 v38, -v43, v119, v38
	s_wait_dscnt 0x1
	v_fma_f32 v39, -v43, v120, v39
	s_wait_dscnt 0x0
	v_fma_f32 v40, -v43, v42, v40
	;;#ASMSTART
	;;#ASMEND
	ds_load_2addr_b32 v[41:42], v44 offset0:192 offset1:224
	v_cndmask_b32_e32 v43, 0, v16, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v43, v43, v32, s0
	ds_bpermute_b32 v44, v76, v43
	s_wait_dscnt 0x1
	ds_bpermute_b32 v45, v92, v41
	ds_bpermute_b32 v46, v77, v41
	ds_bpermute_b32 v47, v78, v41
	ds_bpermute_b32 v48, v79, v41
	ds_bpermute_b32 v94, v80, v41
	ds_bpermute_b32 v95, v81, v41
	ds_bpermute_b32 v96, v82, v41
	ds_bpermute_b32 v97, v83, v41
	ds_bpermute_b32 v98, v84, v41
	ds_bpermute_b32 v99, v85, v41
	ds_bpermute_b32 v100, v86, v41
	ds_bpermute_b32 v101, v87, v41
	ds_bpermute_b32 v102, v88, v41
	ds_bpermute_b32 v103, v89, v41
	ds_bpermute_b32 v104, v90, v41
	ds_bpermute_b32 v41, v91, v41
	ds_bpermute_b32 v105, v92, v42
	ds_bpermute_b32 v106, v77, v42
	ds_bpermute_b32 v107, v78, v42
	ds_bpermute_b32 v108, v79, v42
	ds_bpermute_b32 v109, v80, v42
	ds_bpermute_b32 v110, v81, v42
	ds_bpermute_b32 v111, v82, v42
	ds_bpermute_b32 v112, v83, v42
	ds_bpermute_b32 v113, v84, v42
	ds_bpermute_b32 v114, v85, v42
	ds_bpermute_b32 v115, v86, v42
	ds_bpermute_b32 v116, v87, v42
	ds_bpermute_b32 v117, v88, v42
	ds_bpermute_b32 v118, v89, v42
	ds_bpermute_b32 v119, v90, v42
	ds_bpermute_b32 v42, v91, v42
	s_wait_dscnt 0x20
	v_dual_add_f32 v43, v43, v44 :: v_dual_add_nc_u32 v44, 0x1000, v93
	s_wait_dscnt 0x1f
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v9, -v43, v45, v9
	s_wait_dscnt 0x1e
	v_fma_f32 v10, -v43, v46, v10
	s_wait_dscnt 0x1d
	v_fma_f32 v11, -v43, v47, v11
	s_wait_dscnt 0x1c
	v_fma_f32 v12, -v43, v48, v12
	s_wait_dscnt 0x1b
	v_fma_f32 v13, -v43, v94, v13
	s_wait_dscnt 0x1a
	v_fma_f32 v14, -v43, v95, v14
	s_wait_dscnt 0x19
	v_fma_f32 v15, -v43, v96, v15
	s_wait_dscnt 0x18
	v_fma_f32 v16, -v43, v97, v16
	s_wait_dscnt 0x17
	v_fma_f32 v17, -v43, v98, v17
	s_wait_dscnt 0x16
	v_fma_f32 v18, -v43, v99, v18
	s_wait_dscnt 0x15
	v_fma_f32 v19, -v43, v100, v19
	s_wait_dscnt 0x14
	v_fma_f32 v20, -v43, v101, v20
	s_wait_dscnt 0x13
	v_fma_f32 v21, -v43, v102, v21
	s_wait_dscnt 0x12
	v_fma_f32 v22, -v43, v103, v22
	s_wait_dscnt 0x11
	v_fma_f32 v23, -v43, v104, v23
	s_wait_dscnt 0x10
	v_fma_f32 v24, -v43, v41, v24
	s_wait_dscnt 0xf
	v_fma_f32 v25, -v43, v105, v25
	s_wait_dscnt 0xe
	v_fma_f32 v26, -v43, v106, v26
	s_wait_dscnt 0xd
	v_fma_f32 v27, -v43, v107, v27
	s_wait_dscnt 0xc
	v_fma_f32 v28, -v43, v108, v28
	s_wait_dscnt 0xb
	v_fma_f32 v29, -v43, v109, v29
	s_wait_dscnt 0xa
	v_fma_f32 v30, -v43, v110, v30
	s_wait_dscnt 0x9
	v_fma_f32 v31, -v43, v111, v31
	s_wait_dscnt 0x8
	v_fma_f32 v32, -v43, v112, v32
	s_wait_dscnt 0x7
	v_fma_f32 v33, -v43, v113, v33
	s_wait_dscnt 0x6
	v_fma_f32 v34, -v43, v114, v34
	s_wait_dscnt 0x5
	v_fma_f32 v35, -v43, v115, v35
	s_wait_dscnt 0x4
	v_fma_f32 v36, -v43, v116, v36
	s_wait_dscnt 0x3
	v_fma_f32 v37, -v43, v117, v37
	s_wait_dscnt 0x2
	v_fma_f32 v38, -v43, v118, v38
	s_wait_dscnt 0x1
	v_fma_f32 v39, -v43, v119, v39
	s_wait_dscnt 0x0
	v_fma_f32 v40, -v43, v42, v40
	;;#ASMSTART
	;;#ASMEND
	ds_load_2addr_b32 v[41:42], v44 offset1:32
	v_cndmask_b32_e64 v43, 0, v17, s1
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v43, v43, v33, s2
	ds_bpermute_b32 v45, v76, v43
	s_wait_dscnt 0x1
	ds_bpermute_b32 v46, v92, v41
	ds_bpermute_b32 v47, v77, v41
	ds_bpermute_b32 v48, v78, v41
	ds_bpermute_b32 v94, v79, v41
	ds_bpermute_b32 v95, v80, v41
	ds_bpermute_b32 v96, v81, v41
	ds_bpermute_b32 v97, v82, v41
	ds_bpermute_b32 v98, v83, v41
	ds_bpermute_b32 v99, v84, v41
	ds_bpermute_b32 v100, v85, v41
	ds_bpermute_b32 v101, v86, v41
	ds_bpermute_b32 v102, v87, v41
	ds_bpermute_b32 v103, v88, v41
	ds_bpermute_b32 v104, v89, v41
	ds_bpermute_b32 v105, v90, v41
	ds_bpermute_b32 v41, v91, v41
	ds_bpermute_b32 v106, v92, v42
	ds_bpermute_b32 v107, v77, v42
	ds_bpermute_b32 v108, v78, v42
	ds_bpermute_b32 v109, v79, v42
	ds_bpermute_b32 v110, v80, v42
	ds_bpermute_b32 v111, v81, v42
	ds_bpermute_b32 v112, v82, v42
	ds_bpermute_b32 v113, v83, v42
	ds_bpermute_b32 v114, v84, v42
	ds_bpermute_b32 v115, v85, v42
	ds_bpermute_b32 v116, v86, v42
	ds_bpermute_b32 v117, v87, v42
	ds_bpermute_b32 v118, v88, v42
	ds_bpermute_b32 v119, v89, v42
	ds_bpermute_b32 v120, v90, v42
	ds_bpermute_b32 v42, v91, v42
	s_wait_dscnt 0x20
	v_add_f32_e32 v43, v43, v45
	s_wait_dscnt 0x1f
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v9, -v43, v46, v9
	s_wait_dscnt 0x1e
	v_fma_f32 v10, -v43, v47, v10
	s_wait_dscnt 0x1d
	v_fma_f32 v11, -v43, v48, v11
	s_wait_dscnt 0x1c
	v_fma_f32 v12, -v43, v94, v12
	s_wait_dscnt 0x1b
	v_fma_f32 v13, -v43, v95, v13
	s_wait_dscnt 0x1a
	v_fma_f32 v14, -v43, v96, v14
	s_wait_dscnt 0x19
	v_fma_f32 v15, -v43, v97, v15
	s_wait_dscnt 0x18
	v_fma_f32 v16, -v43, v98, v16
	s_wait_dscnt 0x17
	v_fma_f32 v17, -v43, v99, v17
	s_wait_dscnt 0x16
	v_fma_f32 v18, -v43, v100, v18
	s_wait_dscnt 0x15
	v_fma_f32 v19, -v43, v101, v19
	s_wait_dscnt 0x14
	v_fma_f32 v20, -v43, v102, v20
	s_wait_dscnt 0x13
	v_fma_f32 v21, -v43, v103, v21
	s_wait_dscnt 0x12
	v_fma_f32 v22, -v43, v104, v22
	s_wait_dscnt 0x11
	v_fma_f32 v23, -v43, v105, v23
	s_wait_dscnt 0x10
	v_fma_f32 v24, -v43, v41, v24
	s_wait_dscnt 0xf
	v_fma_f32 v25, -v43, v106, v25
	s_wait_dscnt 0xe
	v_fma_f32 v26, -v43, v107, v26
	s_wait_dscnt 0xd
	v_fma_f32 v27, -v43, v108, v27
	s_wait_dscnt 0xc
	v_fma_f32 v28, -v43, v109, v28
	s_wait_dscnt 0xb
	v_fma_f32 v29, -v43, v110, v29
	s_wait_dscnt 0xa
	v_fma_f32 v30, -v43, v111, v30
	s_wait_dscnt 0x9
	v_fma_f32 v31, -v43, v112, v31
	s_wait_dscnt 0x8
	v_fma_f32 v32, -v43, v113, v32
	s_wait_dscnt 0x7
	v_fma_f32 v33, -v43, v114, v33
	s_wait_dscnt 0x6
	v_fma_f32 v34, -v43, v115, v34
	s_wait_dscnt 0x5
	v_fma_f32 v35, -v43, v116, v35
	s_wait_dscnt 0x4
	v_fma_f32 v36, -v43, v117, v36
	s_wait_dscnt 0x3
	v_fma_f32 v37, -v43, v118, v37
	s_wait_dscnt 0x2
	v_fma_f32 v38, -v43, v119, v38
	s_wait_dscnt 0x1
	v_fma_f32 v39, -v43, v120, v39
	s_wait_dscnt 0x0
	v_fma_f32 v40, -v43, v42, v40
	;;#ASMSTART
	;;#ASMEND
	ds_load_2addr_b32 v[41:42], v44 offset0:64 offset1:96
	v_cndmask_b32_e64 v43, 0, v18, s1
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v43, v43, v34, s2
	ds_bpermute_b32 v45, v76, v43
	s_wait_dscnt 0x1
	ds_bpermute_b32 v46, v92, v41
	ds_bpermute_b32 v47, v77, v41
	ds_bpermute_b32 v48, v78, v41
	ds_bpermute_b32 v94, v79, v41
	ds_bpermute_b32 v95, v80, v41
	ds_bpermute_b32 v96, v81, v41
	ds_bpermute_b32 v97, v82, v41
	ds_bpermute_b32 v98, v83, v41
	ds_bpermute_b32 v99, v84, v41
	ds_bpermute_b32 v100, v85, v41
	ds_bpermute_b32 v101, v86, v41
	ds_bpermute_b32 v102, v87, v41
	ds_bpermute_b32 v103, v88, v41
	ds_bpermute_b32 v104, v89, v41
	ds_bpermute_b32 v105, v90, v41
	ds_bpermute_b32 v41, v91, v41
	ds_bpermute_b32 v106, v92, v42
	ds_bpermute_b32 v107, v77, v42
	ds_bpermute_b32 v108, v78, v42
	ds_bpermute_b32 v109, v79, v42
	ds_bpermute_b32 v110, v80, v42
	ds_bpermute_b32 v111, v81, v42
	ds_bpermute_b32 v112, v82, v42
	ds_bpermute_b32 v113, v83, v42
	ds_bpermute_b32 v114, v84, v42
	ds_bpermute_b32 v115, v85, v42
	ds_bpermute_b32 v116, v86, v42
	ds_bpermute_b32 v117, v87, v42
	ds_bpermute_b32 v118, v88, v42
	ds_bpermute_b32 v119, v89, v42
	ds_bpermute_b32 v120, v90, v42
	ds_bpermute_b32 v42, v91, v42
	s_wait_dscnt 0x20
	v_add_f32_e32 v43, v43, v45
	s_wait_dscnt 0x1f
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v9, -v43, v46, v9
	s_wait_dscnt 0x1e
	v_fma_f32 v10, -v43, v47, v10
	s_wait_dscnt 0x1d
	v_fma_f32 v11, -v43, v48, v11
	s_wait_dscnt 0x1c
	v_fma_f32 v12, -v43, v94, v12
	s_wait_dscnt 0x1b
	v_fma_f32 v13, -v43, v95, v13
	s_wait_dscnt 0x1a
	v_fma_f32 v14, -v43, v96, v14
	s_wait_dscnt 0x19
	v_fma_f32 v15, -v43, v97, v15
	s_wait_dscnt 0x18
	v_fma_f32 v16, -v43, v98, v16
	s_wait_dscnt 0x17
	v_fma_f32 v17, -v43, v99, v17
	s_wait_dscnt 0x16
	v_fma_f32 v18, -v43, v100, v18
	s_wait_dscnt 0x15
	v_fma_f32 v19, -v43, v101, v19
	s_wait_dscnt 0x14
	v_fma_f32 v20, -v43, v102, v20
	s_wait_dscnt 0x13
	v_fma_f32 v21, -v43, v103, v21
	s_wait_dscnt 0x12
	v_fma_f32 v22, -v43, v104, v22
	s_wait_dscnt 0x11
	v_fma_f32 v23, -v43, v105, v23
	s_wait_dscnt 0x10
	v_fma_f32 v24, -v43, v41, v24
	s_wait_dscnt 0xf
	v_fma_f32 v25, -v43, v106, v25
	s_wait_dscnt 0xe
	v_fma_f32 v26, -v43, v107, v26
	s_wait_dscnt 0xd
	v_fma_f32 v27, -v43, v108, v27
	s_wait_dscnt 0xc
	v_fma_f32 v28, -v43, v109, v28
	s_wait_dscnt 0xb
	v_fma_f32 v29, -v43, v110, v29
	s_wait_dscnt 0xa
	v_fma_f32 v30, -v43, v111, v30
	s_wait_dscnt 0x9
	v_fma_f32 v31, -v43, v112, v31
	s_wait_dscnt 0x8
	v_fma_f32 v32, -v43, v113, v32
	s_wait_dscnt 0x7
	v_fma_f32 v33, -v43, v114, v33
	s_wait_dscnt 0x6
	v_fma_f32 v34, -v43, v115, v34
	s_wait_dscnt 0x5
	v_fma_f32 v35, -v43, v116, v35
	s_wait_dscnt 0x4
	v_fma_f32 v36, -v43, v117, v36
	s_wait_dscnt 0x3
	v_fma_f32 v37, -v43, v118, v37
	s_wait_dscnt 0x2
	v_fma_f32 v38, -v43, v119, v38
	s_wait_dscnt 0x1
	v_fma_f32 v39, -v43, v120, v39
	s_wait_dscnt 0x0
	v_fma_f32 v40, -v43, v42, v40
	;;#ASMSTART
	;;#ASMEND
	ds_load_2addr_b32 v[41:42], v44 offset0:128 offset1:160
	v_cndmask_b32_e64 v43, 0, v19, s1
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v43, v43, v35, s2
	ds_bpermute_b32 v45, v76, v43
	s_wait_dscnt 0x1
	ds_bpermute_b32 v46, v92, v41
	ds_bpermute_b32 v47, v77, v41
	ds_bpermute_b32 v48, v78, v41
	ds_bpermute_b32 v94, v79, v41
	ds_bpermute_b32 v95, v80, v41
	ds_bpermute_b32 v96, v81, v41
	ds_bpermute_b32 v97, v82, v41
	ds_bpermute_b32 v98, v83, v41
	ds_bpermute_b32 v99, v84, v41
	ds_bpermute_b32 v100, v85, v41
	ds_bpermute_b32 v101, v86, v41
	ds_bpermute_b32 v102, v87, v41
	ds_bpermute_b32 v103, v88, v41
	ds_bpermute_b32 v104, v89, v41
	ds_bpermute_b32 v105, v90, v41
	ds_bpermute_b32 v41, v91, v41
	ds_bpermute_b32 v106, v92, v42
	ds_bpermute_b32 v107, v77, v42
	ds_bpermute_b32 v108, v78, v42
	ds_bpermute_b32 v109, v79, v42
	ds_bpermute_b32 v110, v80, v42
	ds_bpermute_b32 v111, v81, v42
	ds_bpermute_b32 v112, v82, v42
	ds_bpermute_b32 v113, v83, v42
	ds_bpermute_b32 v114, v84, v42
	ds_bpermute_b32 v115, v85, v42
	ds_bpermute_b32 v116, v86, v42
	ds_bpermute_b32 v117, v87, v42
	ds_bpermute_b32 v118, v88, v42
	ds_bpermute_b32 v119, v89, v42
	ds_bpermute_b32 v120, v90, v42
	ds_bpermute_b32 v42, v91, v42
	s_wait_dscnt 0x20
	v_add_f32_e32 v43, v43, v45
	s_wait_dscnt 0x1f
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v9, -v43, v46, v9
	s_wait_dscnt 0x1e
	v_fma_f32 v10, -v43, v47, v10
	s_wait_dscnt 0x1d
	v_fma_f32 v11, -v43, v48, v11
	s_wait_dscnt 0x1c
	v_fma_f32 v12, -v43, v94, v12
	s_wait_dscnt 0x1b
	v_fma_f32 v13, -v43, v95, v13
	s_wait_dscnt 0x1a
	v_fma_f32 v14, -v43, v96, v14
	s_wait_dscnt 0x19
	v_fma_f32 v15, -v43, v97, v15
	s_wait_dscnt 0x18
	v_fma_f32 v16, -v43, v98, v16
	s_wait_dscnt 0x17
	v_fma_f32 v17, -v43, v99, v17
	s_wait_dscnt 0x16
	v_fma_f32 v18, -v43, v100, v18
	s_wait_dscnt 0x15
	v_fma_f32 v19, -v43, v101, v19
	s_wait_dscnt 0x14
	v_fma_f32 v20, -v43, v102, v20
	s_wait_dscnt 0x13
	v_fma_f32 v21, -v43, v103, v21
	s_wait_dscnt 0x12
	v_fma_f32 v22, -v43, v104, v22
	s_wait_dscnt 0x11
	v_fma_f32 v23, -v43, v105, v23
	s_wait_dscnt 0x10
	v_fma_f32 v24, -v43, v41, v24
	s_wait_dscnt 0xf
	v_fma_f32 v25, -v43, v106, v25
	s_wait_dscnt 0xe
	v_fma_f32 v26, -v43, v107, v26
	s_wait_dscnt 0xd
	v_fma_f32 v27, -v43, v108, v27
	s_wait_dscnt 0xc
	v_fma_f32 v28, -v43, v109, v28
	s_wait_dscnt 0xb
	v_fma_f32 v29, -v43, v110, v29
	s_wait_dscnt 0xa
	v_fma_f32 v30, -v43, v111, v30
	s_wait_dscnt 0x9
	v_fma_f32 v31, -v43, v112, v31
	s_wait_dscnt 0x8
	v_fma_f32 v32, -v43, v113, v32
	s_wait_dscnt 0x7
	v_fma_f32 v33, -v43, v114, v33
	s_wait_dscnt 0x6
	v_fma_f32 v34, -v43, v115, v34
	s_wait_dscnt 0x5
	v_fma_f32 v35, -v43, v116, v35
	s_wait_dscnt 0x4
	v_fma_f32 v36, -v43, v117, v36
	s_wait_dscnt 0x3
	v_fma_f32 v37, -v43, v118, v37
	s_wait_dscnt 0x2
	v_fma_f32 v38, -v43, v119, v38
	s_wait_dscnt 0x1
	v_fma_f32 v39, -v43, v120, v39
	s_wait_dscnt 0x0
	v_fma_f32 v40, -v43, v42, v40
	;;#ASMSTART
	;;#ASMEND
	ds_load_2addr_b32 v[41:42], v44 offset0:192 offset1:224
	v_cndmask_b32_e64 v43, 0, v20, s1
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v43, v43, v36, s2
	ds_bpermute_b32 v44, v76, v43
	s_wait_dscnt 0x1
	ds_bpermute_b32 v45, v92, v41
	ds_bpermute_b32 v46, v77, v41
	ds_bpermute_b32 v47, v78, v41
	ds_bpermute_b32 v48, v79, v41
	ds_bpermute_b32 v94, v80, v41
	ds_bpermute_b32 v95, v81, v41
	ds_bpermute_b32 v96, v82, v41
	ds_bpermute_b32 v97, v83, v41
	ds_bpermute_b32 v98, v84, v41
	ds_bpermute_b32 v99, v85, v41
	ds_bpermute_b32 v100, v86, v41
	ds_bpermute_b32 v101, v87, v41
	ds_bpermute_b32 v102, v88, v41
	ds_bpermute_b32 v103, v89, v41
	ds_bpermute_b32 v104, v90, v41
	ds_bpermute_b32 v41, v91, v41
	ds_bpermute_b32 v105, v92, v42
	ds_bpermute_b32 v106, v77, v42
	ds_bpermute_b32 v107, v78, v42
	ds_bpermute_b32 v108, v79, v42
	ds_bpermute_b32 v109, v80, v42
	ds_bpermute_b32 v110, v81, v42
	ds_bpermute_b32 v111, v82, v42
	ds_bpermute_b32 v112, v83, v42
	ds_bpermute_b32 v113, v84, v42
	ds_bpermute_b32 v114, v85, v42
	ds_bpermute_b32 v115, v86, v42
	ds_bpermute_b32 v116, v87, v42
	ds_bpermute_b32 v117, v88, v42
	ds_bpermute_b32 v118, v89, v42
	ds_bpermute_b32 v119, v90, v42
	ds_bpermute_b32 v42, v91, v42
	s_wait_dscnt 0x20
	v_dual_add_f32 v43, v43, v44 :: v_dual_add_nc_u32 v44, 0x1400, v93
	s_wait_dscnt 0x1f
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v9, -v43, v45, v9
	s_wait_dscnt 0x1e
	v_fma_f32 v10, -v43, v46, v10
	s_wait_dscnt 0x1d
	v_fma_f32 v11, -v43, v47, v11
	s_wait_dscnt 0x1c
	v_fma_f32 v12, -v43, v48, v12
	s_wait_dscnt 0x1b
	v_fma_f32 v13, -v43, v94, v13
	s_wait_dscnt 0x1a
	v_fma_f32 v14, -v43, v95, v14
	s_wait_dscnt 0x19
	v_fma_f32 v15, -v43, v96, v15
	s_wait_dscnt 0x18
	v_fma_f32 v16, -v43, v97, v16
	s_wait_dscnt 0x17
	v_fma_f32 v17, -v43, v98, v17
	s_wait_dscnt 0x16
	v_fma_f32 v18, -v43, v99, v18
	s_wait_dscnt 0x15
	v_fma_f32 v19, -v43, v100, v19
	s_wait_dscnt 0x14
	v_fma_f32 v20, -v43, v101, v20
	s_wait_dscnt 0x13
	v_fma_f32 v21, -v43, v102, v21
	s_wait_dscnt 0x12
	v_fma_f32 v22, -v43, v103, v22
	s_wait_dscnt 0x11
	v_fma_f32 v23, -v43, v104, v23
	s_wait_dscnt 0x10
	v_fma_f32 v24, -v43, v41, v24
	s_wait_dscnt 0xf
	v_fma_f32 v25, -v43, v105, v25
	s_wait_dscnt 0xe
	v_fma_f32 v26, -v43, v106, v26
	s_wait_dscnt 0xd
	v_fma_f32 v27, -v43, v107, v27
	s_wait_dscnt 0xc
	v_fma_f32 v28, -v43, v108, v28
	s_wait_dscnt 0xb
	v_fma_f32 v29, -v43, v109, v29
	s_wait_dscnt 0xa
	v_fma_f32 v30, -v43, v110, v30
	s_wait_dscnt 0x9
	v_fma_f32 v31, -v43, v111, v31
	s_wait_dscnt 0x8
	v_fma_f32 v32, -v43, v112, v32
	s_wait_dscnt 0x7
	v_fma_f32 v33, -v43, v113, v33
	s_wait_dscnt 0x6
	v_fma_f32 v34, -v43, v114, v34
	s_wait_dscnt 0x5
	v_fma_f32 v35, -v43, v115, v35
	s_wait_dscnt 0x4
	v_fma_f32 v36, -v43, v116, v36
	s_wait_dscnt 0x3
	v_fma_f32 v37, -v43, v117, v37
	s_wait_dscnt 0x2
	v_fma_f32 v38, -v43, v118, v38
	s_wait_dscnt 0x1
	v_fma_f32 v39, -v43, v119, v39
	s_wait_dscnt 0x0
	v_fma_f32 v40, -v43, v42, v40
	;;#ASMSTART
	;;#ASMEND
	ds_load_2addr_b32 v[41:42], v44 offset1:32
	v_cndmask_b32_e64 v43, 0, v21, s1
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v43, v43, v37, s2
	ds_bpermute_b32 v45, v76, v43
	s_wait_dscnt 0x1
	ds_bpermute_b32 v46, v92, v41
	ds_bpermute_b32 v47, v77, v41
	ds_bpermute_b32 v48, v78, v41
	ds_bpermute_b32 v94, v79, v41
	ds_bpermute_b32 v95, v80, v41
	ds_bpermute_b32 v96, v81, v41
	ds_bpermute_b32 v97, v82, v41
	ds_bpermute_b32 v98, v83, v41
	ds_bpermute_b32 v99, v84, v41
	ds_bpermute_b32 v100, v85, v41
	ds_bpermute_b32 v101, v86, v41
	ds_bpermute_b32 v102, v87, v41
	ds_bpermute_b32 v103, v88, v41
	ds_bpermute_b32 v104, v89, v41
	ds_bpermute_b32 v105, v90, v41
	ds_bpermute_b32 v41, v91, v41
	ds_bpermute_b32 v106, v92, v42
	ds_bpermute_b32 v107, v77, v42
	ds_bpermute_b32 v108, v78, v42
	ds_bpermute_b32 v109, v79, v42
	ds_bpermute_b32 v110, v80, v42
	ds_bpermute_b32 v111, v81, v42
	ds_bpermute_b32 v112, v82, v42
	ds_bpermute_b32 v113, v83, v42
	ds_bpermute_b32 v114, v84, v42
	ds_bpermute_b32 v115, v85, v42
	ds_bpermute_b32 v116, v86, v42
	ds_bpermute_b32 v117, v87, v42
	ds_bpermute_b32 v118, v88, v42
	ds_bpermute_b32 v119, v89, v42
	ds_bpermute_b32 v120, v90, v42
	ds_bpermute_b32 v42, v91, v42
	s_wait_dscnt 0x20
	v_add_f32_e32 v43, v43, v45
	s_wait_dscnt 0x1f
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v9, -v43, v46, v9
	s_wait_dscnt 0x1e
	v_fma_f32 v10, -v43, v47, v10
	s_wait_dscnt 0x1d
	v_fma_f32 v11, -v43, v48, v11
	s_wait_dscnt 0x1c
	v_fma_f32 v12, -v43, v94, v12
	s_wait_dscnt 0x1b
	v_fma_f32 v13, -v43, v95, v13
	s_wait_dscnt 0x1a
	v_fma_f32 v14, -v43, v96, v14
	s_wait_dscnt 0x19
	v_fma_f32 v15, -v43, v97, v15
	s_wait_dscnt 0x18
	v_fma_f32 v16, -v43, v98, v16
	s_wait_dscnt 0x17
	v_fma_f32 v17, -v43, v99, v17
	s_wait_dscnt 0x16
	v_fma_f32 v18, -v43, v100, v18
	s_wait_dscnt 0x15
	v_fma_f32 v19, -v43, v101, v19
	s_wait_dscnt 0x14
	v_fma_f32 v20, -v43, v102, v20
	s_wait_dscnt 0x13
	v_fma_f32 v21, -v43, v103, v21
	s_wait_dscnt 0x12
	v_fma_f32 v22, -v43, v104, v22
	s_wait_dscnt 0x11
	v_fma_f32 v23, -v43, v105, v23
	s_wait_dscnt 0x10
	v_fma_f32 v24, -v43, v41, v24
	s_wait_dscnt 0xf
	v_fma_f32 v25, -v43, v106, v25
	s_wait_dscnt 0xe
	v_fma_f32 v26, -v43, v107, v26
	s_wait_dscnt 0xd
	v_fma_f32 v27, -v43, v108, v27
	s_wait_dscnt 0xc
	v_fma_f32 v28, -v43, v109, v28
	s_wait_dscnt 0xb
	v_fma_f32 v29, -v43, v110, v29
	s_wait_dscnt 0xa
	v_fma_f32 v30, -v43, v111, v30
	s_wait_dscnt 0x9
	v_fma_f32 v31, -v43, v112, v31
	s_wait_dscnt 0x8
	v_fma_f32 v32, -v43, v113, v32
	s_wait_dscnt 0x7
	v_fma_f32 v33, -v43, v114, v33
	s_wait_dscnt 0x6
	v_fma_f32 v34, -v43, v115, v34
	s_wait_dscnt 0x5
	v_fma_f32 v35, -v43, v116, v35
	s_wait_dscnt 0x4
	v_fma_f32 v36, -v43, v117, v36
	s_wait_dscnt 0x3
	v_fma_f32 v37, -v43, v118, v37
	s_wait_dscnt 0x2
	v_fma_f32 v38, -v43, v119, v38
	s_wait_dscnt 0x1
	v_fma_f32 v39, -v43, v120, v39
	s_wait_dscnt 0x0
	v_fma_f32 v40, -v43, v42, v40
	;;#ASMSTART
	;;#ASMEND
	ds_load_2addr_b32 v[41:42], v44 offset0:64 offset1:96
	v_cndmask_b32_e64 v43, 0, v22, s1
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v43, v43, v38, s2
	ds_bpermute_b32 v45, v76, v43
	s_wait_dscnt 0x1
	ds_bpermute_b32 v46, v92, v41
	ds_bpermute_b32 v47, v77, v41
	ds_bpermute_b32 v48, v78, v41
	ds_bpermute_b32 v94, v79, v41
	ds_bpermute_b32 v95, v80, v41
	ds_bpermute_b32 v96, v81, v41
	ds_bpermute_b32 v97, v82, v41
	ds_bpermute_b32 v98, v83, v41
	ds_bpermute_b32 v99, v84, v41
	ds_bpermute_b32 v100, v85, v41
	ds_bpermute_b32 v101, v86, v41
	ds_bpermute_b32 v102, v87, v41
	ds_bpermute_b32 v103, v88, v41
	ds_bpermute_b32 v104, v89, v41
	ds_bpermute_b32 v105, v90, v41
	ds_bpermute_b32 v41, v91, v41
	ds_bpermute_b32 v106, v92, v42
	ds_bpermute_b32 v107, v77, v42
	ds_bpermute_b32 v108, v78, v42
	ds_bpermute_b32 v109, v79, v42
	ds_bpermute_b32 v110, v80, v42
	ds_bpermute_b32 v111, v81, v42
	ds_bpermute_b32 v112, v82, v42
	ds_bpermute_b32 v113, v83, v42
	ds_bpermute_b32 v114, v84, v42
	ds_bpermute_b32 v115, v85, v42
	ds_bpermute_b32 v116, v86, v42
	ds_bpermute_b32 v117, v87, v42
	ds_bpermute_b32 v118, v88, v42
	ds_bpermute_b32 v119, v89, v42
	ds_bpermute_b32 v120, v90, v42
	ds_bpermute_b32 v42, v91, v42
	s_wait_dscnt 0x20
	v_add_f32_e32 v43, v43, v45
	s_wait_dscnt 0x1f
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v9, -v43, v46, v9
	s_wait_dscnt 0x1e
	v_fma_f32 v10, -v43, v47, v10
	s_wait_dscnt 0x1d
	v_fma_f32 v11, -v43, v48, v11
	s_wait_dscnt 0x1c
	v_fma_f32 v12, -v43, v94, v12
	s_wait_dscnt 0x1b
	v_fma_f32 v13, -v43, v95, v13
	s_wait_dscnt 0x1a
	v_fma_f32 v14, -v43, v96, v14
	s_wait_dscnt 0x19
	v_fma_f32 v15, -v43, v97, v15
	s_wait_dscnt 0x18
	v_fma_f32 v16, -v43, v98, v16
	s_wait_dscnt 0x17
	v_fma_f32 v17, -v43, v99, v17
	s_wait_dscnt 0x16
	v_fma_f32 v18, -v43, v100, v18
	s_wait_dscnt 0x15
	v_fma_f32 v19, -v43, v101, v19
	s_wait_dscnt 0x14
	v_fma_f32 v20, -v43, v102, v20
	s_wait_dscnt 0x13
	v_fma_f32 v21, -v43, v103, v21
	s_wait_dscnt 0x12
	v_fma_f32 v22, -v43, v104, v22
	s_wait_dscnt 0x11
	v_fma_f32 v23, -v43, v105, v23
	s_wait_dscnt 0x10
	v_fma_f32 v24, -v43, v41, v24
	s_wait_dscnt 0xf
	v_fma_f32 v25, -v43, v106, v25
	s_wait_dscnt 0xe
	v_fma_f32 v26, -v43, v107, v26
	s_wait_dscnt 0xd
	v_fma_f32 v27, -v43, v108, v27
	s_wait_dscnt 0xc
	v_fma_f32 v28, -v43, v109, v28
	s_wait_dscnt 0xb
	v_fma_f32 v29, -v43, v110, v29
	s_wait_dscnt 0xa
	v_fma_f32 v30, -v43, v111, v30
	s_wait_dscnt 0x9
	v_fma_f32 v31, -v43, v112, v31
	s_wait_dscnt 0x8
	v_fma_f32 v32, -v43, v113, v32
	s_wait_dscnt 0x7
	v_fma_f32 v33, -v43, v114, v33
	s_wait_dscnt 0x6
	v_fma_f32 v34, -v43, v115, v34
	s_wait_dscnt 0x5
	v_fma_f32 v35, -v43, v116, v35
	s_wait_dscnt 0x4
	v_fma_f32 v36, -v43, v117, v36
	s_wait_dscnt 0x3
	v_fma_f32 v37, -v43, v118, v37
	s_wait_dscnt 0x2
	v_fma_f32 v38, -v43, v119, v38
	s_wait_dscnt 0x1
	v_fma_f32 v39, -v43, v120, v39
	s_wait_dscnt 0x0
	v_fma_f32 v40, -v43, v42, v40
	;;#ASMSTART
	;;#ASMEND
	ds_load_2addr_b32 v[41:42], v44 offset0:128 offset1:160
	v_cndmask_b32_e64 v43, 0, v23, s1
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v43, v43, v39, s2
	ds_bpermute_b32 v45, v76, v43
	s_wait_dscnt 0x1
	ds_bpermute_b32 v46, v92, v41
	ds_bpermute_b32 v47, v77, v41
	ds_bpermute_b32 v48, v78, v41
	ds_bpermute_b32 v94, v79, v41
	ds_bpermute_b32 v95, v80, v41
	ds_bpermute_b32 v96, v81, v41
	ds_bpermute_b32 v97, v82, v41
	ds_bpermute_b32 v98, v83, v41
	ds_bpermute_b32 v99, v84, v41
	ds_bpermute_b32 v100, v85, v41
	ds_bpermute_b32 v101, v86, v41
	ds_bpermute_b32 v102, v87, v41
	ds_bpermute_b32 v103, v88, v41
	ds_bpermute_b32 v104, v89, v41
	ds_bpermute_b32 v105, v90, v41
	ds_bpermute_b32 v41, v91, v41
	ds_bpermute_b32 v106, v92, v42
	ds_bpermute_b32 v107, v77, v42
	ds_bpermute_b32 v108, v78, v42
	ds_bpermute_b32 v109, v79, v42
	ds_bpermute_b32 v110, v80, v42
	ds_bpermute_b32 v111, v81, v42
	ds_bpermute_b32 v112, v82, v42
	ds_bpermute_b32 v113, v83, v42
	ds_bpermute_b32 v114, v84, v42
	ds_bpermute_b32 v115, v85, v42
	ds_bpermute_b32 v116, v86, v42
	ds_bpermute_b32 v117, v87, v42
	ds_bpermute_b32 v118, v88, v42
	ds_bpermute_b32 v119, v89, v42
	ds_bpermute_b32 v120, v90, v42
	ds_bpermute_b32 v42, v91, v42
	s_wait_dscnt 0x20
	v_add_f32_e32 v43, v43, v45
	s_wait_dscnt 0x1f
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v9, -v43, v46, v9
	s_wait_dscnt 0x1e
	v_fma_f32 v10, -v43, v47, v10
	s_wait_dscnt 0x1d
	v_fma_f32 v11, -v43, v48, v11
	s_wait_dscnt 0x1c
	v_fma_f32 v12, -v43, v94, v12
	s_wait_dscnt 0x1b
	v_fma_f32 v13, -v43, v95, v13
	s_wait_dscnt 0x1a
	v_fma_f32 v14, -v43, v96, v14
	s_wait_dscnt 0x19
	v_fma_f32 v15, -v43, v97, v15
	s_wait_dscnt 0x18
	v_fma_f32 v16, -v43, v98, v16
	s_wait_dscnt 0x17
	v_fma_f32 v17, -v43, v99, v17
	s_wait_dscnt 0x16
	v_fma_f32 v18, -v43, v100, v18
	s_wait_dscnt 0x15
	v_fma_f32 v19, -v43, v101, v19
	s_wait_dscnt 0x14
	v_fma_f32 v20, -v43, v102, v20
	s_wait_dscnt 0x13
	v_fma_f32 v21, -v43, v103, v21
	s_wait_dscnt 0x12
	v_fma_f32 v22, -v43, v104, v22
	s_wait_dscnt 0x11
	v_fma_f32 v23, -v43, v105, v23
	s_wait_dscnt 0x10
	v_fma_f32 v24, -v43, v41, v24
	s_wait_dscnt 0xf
	v_fma_f32 v25, -v43, v106, v25
	s_wait_dscnt 0xe
	v_fma_f32 v26, -v43, v107, v26
	s_wait_dscnt 0xd
	v_fma_f32 v27, -v43, v108, v27
	s_wait_dscnt 0xc
	v_fma_f32 v28, -v43, v109, v28
	s_wait_dscnt 0xb
	v_fma_f32 v29, -v43, v110, v29
	s_wait_dscnt 0xa
	v_fma_f32 v30, -v43, v111, v30
	s_wait_dscnt 0x9
	v_fma_f32 v31, -v43, v112, v31
	s_wait_dscnt 0x8
	v_fma_f32 v32, -v43, v113, v32
	s_wait_dscnt 0x7
	v_fma_f32 v33, -v43, v114, v33
	s_wait_dscnt 0x6
	v_fma_f32 v34, -v43, v115, v34
	s_wait_dscnt 0x5
	v_fma_f32 v35, -v43, v116, v35
	s_wait_dscnt 0x4
	v_fma_f32 v36, -v43, v117, v36
	s_wait_dscnt 0x3
	v_fma_f32 v37, -v43, v118, v37
	s_wait_dscnt 0x2
	v_fma_f32 v38, -v43, v119, v38
	s_wait_dscnt 0x1
	v_fma_f32 v39, -v43, v120, v39
	s_wait_dscnt 0x0
	v_fma_f32 v40, -v43, v42, v40
	;;#ASMSTART
	;;#ASMEND
	ds_load_2addr_b32 v[41:42], v44 offset0:192 offset1:224
	v_cndmask_b32_e64 v43, 0, v24, s1
	v_add_nc_u32_e32 v120, 0x1800, v93
	v_add_nc_u32_e32 v93, 0x1c00, v93
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v43, v43, v40, s2
	ds_bpermute_b32 v44, v76, v43
	s_wait_dscnt 0x1
	ds_bpermute_b32 v45, v92, v41
	ds_bpermute_b32 v46, v77, v41
	ds_bpermute_b32 v47, v78, v41
	ds_bpermute_b32 v48, v79, v41
	ds_bpermute_b32 v94, v80, v41
	ds_bpermute_b32 v95, v81, v41
	ds_bpermute_b32 v96, v82, v41
	ds_bpermute_b32 v97, v83, v41
	ds_bpermute_b32 v98, v84, v41
	ds_bpermute_b32 v99, v85, v41
	ds_bpermute_b32 v100, v86, v41
	ds_bpermute_b32 v101, v87, v41
	ds_bpermute_b32 v102, v88, v41
	ds_bpermute_b32 v103, v89, v41
	ds_bpermute_b32 v104, v90, v41
	ds_bpermute_b32 v41, v91, v41
	ds_bpermute_b32 v105, v92, v42
	ds_bpermute_b32 v106, v77, v42
	ds_bpermute_b32 v107, v78, v42
	ds_bpermute_b32 v108, v79, v42
	ds_bpermute_b32 v109, v80, v42
	ds_bpermute_b32 v110, v81, v42
	ds_bpermute_b32 v111, v82, v42
	ds_bpermute_b32 v112, v83, v42
	ds_bpermute_b32 v113, v84, v42
	ds_bpermute_b32 v114, v85, v42
	ds_bpermute_b32 v115, v86, v42
	ds_bpermute_b32 v116, v87, v42
	ds_bpermute_b32 v117, v88, v42
	ds_bpermute_b32 v118, v89, v42
	ds_bpermute_b32 v119, v90, v42
	ds_bpermute_b32 v42, v91, v42
	s_wait_dscnt 0x20
	v_add_f32_e32 v43, v43, v44
	s_wait_dscnt 0x1f
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v9, -v43, v45, v9
	s_wait_dscnt 0x1e
	v_fma_f32 v10, -v43, v46, v10
	s_wait_dscnt 0x1d
	v_fma_f32 v11, -v43, v47, v11
	s_wait_dscnt 0x1c
	v_fma_f32 v12, -v43, v48, v12
	s_wait_dscnt 0x1b
	v_fma_f32 v13, -v43, v94, v13
	s_wait_dscnt 0x1a
	v_fma_f32 v14, -v43, v95, v14
	s_wait_dscnt 0x19
	v_fma_f32 v15, -v43, v96, v15
	s_wait_dscnt 0x18
	v_fma_f32 v16, -v43, v97, v16
	s_wait_dscnt 0x17
	v_fma_f32 v17, -v43, v98, v17
	s_wait_dscnt 0x16
	v_fma_f32 v18, -v43, v99, v18
	s_wait_dscnt 0x15
	v_fma_f32 v19, -v43, v100, v19
	s_wait_dscnt 0x14
	v_fma_f32 v20, -v43, v101, v20
	s_wait_dscnt 0x13
	v_fma_f32 v21, -v43, v102, v21
	s_wait_dscnt 0x12
	v_fma_f32 v22, -v43, v103, v22
	s_wait_dscnt 0x11
	v_fma_f32 v23, -v43, v104, v23
	s_wait_dscnt 0x10
	v_fma_f32 v24, -v43, v41, v24
	s_wait_dscnt 0xf
	v_fma_f32 v25, -v43, v105, v25
	s_wait_dscnt 0xe
	v_fma_f32 v26, -v43, v106, v26
	s_wait_dscnt 0xd
	v_fma_f32 v27, -v43, v107, v27
	s_wait_dscnt 0xc
	v_fma_f32 v28, -v43, v108, v28
	s_wait_dscnt 0xb
	v_fma_f32 v29, -v43, v109, v29
	s_wait_dscnt 0xa
	v_fma_f32 v30, -v43, v110, v30
	s_wait_dscnt 0x9
	v_fma_f32 v31, -v43, v111, v31
	s_wait_dscnt 0x8
	v_fma_f32 v32, -v43, v112, v32
	s_wait_dscnt 0x7
	v_fma_f32 v33, -v43, v113, v33
	s_wait_dscnt 0x6
	v_fma_f32 v34, -v43, v114, v34
	s_wait_dscnt 0x5
	v_fma_f32 v35, -v43, v115, v35
	s_wait_dscnt 0x4
	v_fma_f32 v36, -v43, v116, v36
	s_wait_dscnt 0x3
	v_fma_f32 v37, -v43, v117, v37
	s_wait_dscnt 0x2
	v_fma_f32 v38, -v43, v118, v38
	s_wait_dscnt 0x1
	v_fma_f32 v39, -v43, v119, v39
	s_wait_dscnt 0x0
	v_fma_f32 v40, -v43, v42, v40
	;;#ASMSTART
	;;#ASMEND
	ds_load_2addr_b32 v[41:42], v120 offset1:32
	v_cndmask_b32_e32 v43, 0, v17, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v43, v43, v33, s0
	ds_bpermute_b32 v44, v76, v43
	s_wait_dscnt 0x1
	ds_bpermute_b32 v45, v92, v41
	ds_bpermute_b32 v46, v77, v41
	ds_bpermute_b32 v47, v78, v41
	ds_bpermute_b32 v48, v79, v41
	ds_bpermute_b32 v94, v80, v41
	ds_bpermute_b32 v95, v81, v41
	ds_bpermute_b32 v96, v82, v41
	ds_bpermute_b32 v97, v83, v41
	ds_bpermute_b32 v98, v84, v41
	ds_bpermute_b32 v99, v85, v41
	ds_bpermute_b32 v100, v86, v41
	ds_bpermute_b32 v101, v87, v41
	ds_bpermute_b32 v102, v88, v41
	ds_bpermute_b32 v103, v89, v41
	ds_bpermute_b32 v104, v90, v41
	ds_bpermute_b32 v41, v91, v41
	ds_bpermute_b32 v105, v92, v42
	ds_bpermute_b32 v106, v77, v42
	ds_bpermute_b32 v107, v78, v42
	ds_bpermute_b32 v108, v79, v42
	ds_bpermute_b32 v109, v80, v42
	ds_bpermute_b32 v110, v81, v42
	ds_bpermute_b32 v111, v82, v42
	ds_bpermute_b32 v112, v83, v42
	ds_bpermute_b32 v113, v84, v42
	ds_bpermute_b32 v114, v85, v42
	ds_bpermute_b32 v115, v86, v42
	ds_bpermute_b32 v116, v87, v42
	ds_bpermute_b32 v117, v88, v42
	ds_bpermute_b32 v118, v89, v42
	ds_bpermute_b32 v119, v90, v42
	ds_bpermute_b32 v42, v91, v42
	s_wait_dscnt 0x20
	v_add_f32_e32 v43, v43, v44
	s_wait_dscnt 0x1f
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v9, -v43, v45, v9
	s_wait_dscnt 0x1e
	v_fma_f32 v10, -v43, v46, v10
	s_wait_dscnt 0x1d
	v_fma_f32 v11, -v43, v47, v11
	s_wait_dscnt 0x1c
	v_fma_f32 v12, -v43, v48, v12
	s_wait_dscnt 0x1b
	v_fma_f32 v13, -v43, v94, v13
	s_wait_dscnt 0x1a
	v_fma_f32 v14, -v43, v95, v14
	s_wait_dscnt 0x19
	v_fma_f32 v15, -v43, v96, v15
	s_wait_dscnt 0x18
	v_fma_f32 v16, -v43, v97, v16
	s_wait_dscnt 0x17
	v_fma_f32 v17, -v43, v98, v17
	s_wait_dscnt 0x16
	v_fma_f32 v18, -v43, v99, v18
	s_wait_dscnt 0x15
	v_fma_f32 v19, -v43, v100, v19
	s_wait_dscnt 0x14
	v_fma_f32 v20, -v43, v101, v20
	s_wait_dscnt 0x13
	v_fma_f32 v21, -v43, v102, v21
	s_wait_dscnt 0x12
	v_fma_f32 v22, -v43, v103, v22
	s_wait_dscnt 0x11
	v_fma_f32 v23, -v43, v104, v23
	s_wait_dscnt 0x10
	v_fma_f32 v24, -v43, v41, v24
	s_wait_dscnt 0xf
	v_fma_f32 v25, -v43, v105, v25
	s_wait_dscnt 0xe
	v_fma_f32 v26, -v43, v106, v26
	s_wait_dscnt 0xd
	v_fma_f32 v27, -v43, v107, v27
	s_wait_dscnt 0xc
	v_fma_f32 v28, -v43, v108, v28
	s_wait_dscnt 0xb
	v_fma_f32 v29, -v43, v109, v29
	s_wait_dscnt 0xa
	v_fma_f32 v30, -v43, v110, v30
	s_wait_dscnt 0x9
	v_fma_f32 v31, -v43, v111, v31
	s_wait_dscnt 0x8
	v_fma_f32 v32, -v43, v112, v32
	s_wait_dscnt 0x7
	v_fma_f32 v33, -v43, v113, v33
	s_wait_dscnt 0x6
	v_fma_f32 v34, -v43, v114, v34
	s_wait_dscnt 0x5
	v_fma_f32 v35, -v43, v115, v35
	s_wait_dscnt 0x4
	v_fma_f32 v36, -v43, v116, v36
	s_wait_dscnt 0x3
	v_fma_f32 v37, -v43, v117, v37
	s_wait_dscnt 0x2
	v_fma_f32 v38, -v43, v118, v38
	s_wait_dscnt 0x1
	v_fma_f32 v39, -v43, v119, v39
	s_wait_dscnt 0x0
	v_fma_f32 v40, -v43, v42, v40
	;;#ASMSTART
	;;#ASMEND
	ds_load_2addr_b32 v[41:42], v120 offset0:64 offset1:96
	v_cndmask_b32_e32 v43, 0, v18, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v43, v43, v34, s0
	ds_bpermute_b32 v44, v76, v43
	s_wait_dscnt 0x1
	ds_bpermute_b32 v45, v92, v41
	ds_bpermute_b32 v46, v77, v41
	ds_bpermute_b32 v47, v78, v41
	ds_bpermute_b32 v48, v79, v41
	ds_bpermute_b32 v94, v80, v41
	ds_bpermute_b32 v95, v81, v41
	ds_bpermute_b32 v96, v82, v41
	ds_bpermute_b32 v97, v83, v41
	ds_bpermute_b32 v98, v84, v41
	ds_bpermute_b32 v99, v85, v41
	ds_bpermute_b32 v100, v86, v41
	ds_bpermute_b32 v101, v87, v41
	ds_bpermute_b32 v102, v88, v41
	ds_bpermute_b32 v103, v89, v41
	ds_bpermute_b32 v104, v90, v41
	ds_bpermute_b32 v41, v91, v41
	ds_bpermute_b32 v105, v92, v42
	ds_bpermute_b32 v106, v77, v42
	ds_bpermute_b32 v107, v78, v42
	ds_bpermute_b32 v108, v79, v42
	ds_bpermute_b32 v109, v80, v42
	ds_bpermute_b32 v110, v81, v42
	ds_bpermute_b32 v111, v82, v42
	ds_bpermute_b32 v112, v83, v42
	ds_bpermute_b32 v113, v84, v42
	ds_bpermute_b32 v114, v85, v42
	ds_bpermute_b32 v115, v86, v42
	ds_bpermute_b32 v116, v87, v42
	ds_bpermute_b32 v117, v88, v42
	ds_bpermute_b32 v118, v89, v42
	ds_bpermute_b32 v119, v90, v42
	ds_bpermute_b32 v121, v91, v42
	s_wait_dscnt 0x20
	v_add_f32_e32 v122, v43, v44
	s_wait_dscnt 0x1f
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v9, -v122, v45, v9
	s_wait_dscnt 0x1e
	v_fma_f32 v10, -v122, v46, v10
	s_wait_dscnt 0x1d
	v_fma_f32 v11, -v122, v47, v11
	s_wait_dscnt 0x1c
	v_fma_f32 v12, -v122, v48, v12
	s_wait_dscnt 0x1b
	v_fma_f32 v13, -v122, v94, v13
	s_wait_dscnt 0x1a
	v_fma_f32 v14, -v122, v95, v14
	s_wait_dscnt 0x19
	v_fma_f32 v15, -v122, v96, v15
	s_wait_dscnt 0x18
	v_fma_f32 v16, -v122, v97, v16
	s_wait_dscnt 0x17
	v_fma_f32 v17, -v122, v98, v17
	s_wait_dscnt 0x16
	v_fma_f32 v18, -v122, v99, v18
	s_wait_dscnt 0x15
	v_fma_f32 v19, -v122, v100, v19
	s_wait_dscnt 0x14
	v_fma_f32 v20, -v122, v101, v20
	s_wait_dscnt 0x13
	v_fma_f32 v21, -v122, v102, v21
	s_wait_dscnt 0x12
	v_fma_f32 v22, -v122, v103, v22
	s_wait_dscnt 0x11
	v_fma_f32 v23, -v122, v104, v23
	s_wait_dscnt 0x10
	v_fma_f32 v24, -v122, v41, v24
	s_wait_dscnt 0xf
	v_fma_f32 v25, -v122, v105, v25
	s_wait_dscnt 0xe
	v_fma_f32 v26, -v122, v106, v26
	s_wait_dscnt 0xd
	v_fma_f32 v27, -v122, v107, v27
	s_wait_dscnt 0xc
	v_fma_f32 v28, -v122, v108, v28
	s_wait_dscnt 0xb
	v_fma_f32 v29, -v122, v109, v29
	s_wait_dscnt 0xa
	v_fma_f32 v30, -v122, v110, v30
	s_wait_dscnt 0x9
	v_fma_f32 v31, -v122, v111, v31
	s_wait_dscnt 0x8
	v_fma_f32 v32, -v122, v112, v32
	s_wait_dscnt 0x7
	v_fma_f32 v41, -v122, v113, v33
	s_wait_dscnt 0x6
	v_fma_f32 v42, -v122, v114, v34
	s_wait_dscnt 0x5
	v_fma_f32 v43, -v122, v115, v35
	s_wait_dscnt 0x4
	v_fma_f32 v44, -v122, v116, v36
	s_wait_dscnt 0x3
	v_fma_f32 v45, -v122, v117, v37
	s_wait_dscnt 0x2
	v_fma_f32 v46, -v122, v118, v38
	s_wait_dscnt 0x1
	v_fma_f32 v47, -v122, v119, v39
	s_wait_dscnt 0x0
	v_fma_f32 v48, -v122, v121, v40
	;;#ASMSTART
	;;#ASMEND
	ds_load_2addr_b32 v[33:34], v120 offset0:128 offset1:160
	v_cndmask_b32_e32 v35, 0, v19, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v35, v35, v43, s0
	ds_bpermute_b32 v36, v76, v35
	s_wait_dscnt 0x1
	ds_bpermute_b32 v37, v92, v33
	ds_bpermute_b32 v38, v77, v33
	ds_bpermute_b32 v39, v78, v33
	ds_bpermute_b32 v40, v79, v33
	ds_bpermute_b32 v94, v80, v33
	ds_bpermute_b32 v95, v81, v33
	ds_bpermute_b32 v96, v82, v33
	ds_bpermute_b32 v97, v83, v33
	ds_bpermute_b32 v98, v84, v33
	ds_bpermute_b32 v99, v85, v33
	ds_bpermute_b32 v100, v86, v33
	ds_bpermute_b32 v101, v87, v33
	ds_bpermute_b32 v102, v88, v33
	ds_bpermute_b32 v103, v89, v33
	ds_bpermute_b32 v104, v90, v33
	ds_bpermute_b32 v33, v91, v33
	ds_bpermute_b32 v105, v92, v34
	ds_bpermute_b32 v106, v77, v34
	ds_bpermute_b32 v107, v78, v34
	ds_bpermute_b32 v108, v79, v34
	ds_bpermute_b32 v109, v80, v34
	ds_bpermute_b32 v110, v81, v34
	ds_bpermute_b32 v111, v82, v34
	ds_bpermute_b32 v112, v83, v34
	ds_bpermute_b32 v113, v84, v34
	ds_bpermute_b32 v114, v85, v34
	ds_bpermute_b32 v115, v86, v34
	ds_bpermute_b32 v116, v87, v34
	ds_bpermute_b32 v117, v88, v34
	ds_bpermute_b32 v118, v89, v34
	ds_bpermute_b32 v119, v90, v34
	ds_bpermute_b32 v121, v91, v34
	s_wait_dscnt 0x20
	v_add_f32_e32 v122, v35, v36
	s_wait_dscnt 0x1f
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v9, -v122, v37, v9
	s_wait_dscnt 0x1e
	v_fma_f32 v10, -v122, v38, v10
	s_wait_dscnt 0x1d
	v_fma_f32 v11, -v122, v39, v11
	s_wait_dscnt 0x1c
	v_fma_f32 v12, -v122, v40, v12
	s_wait_dscnt 0x1b
	v_fma_f32 v13, -v122, v94, v13
	s_wait_dscnt 0x1a
	v_fma_f32 v14, -v122, v95, v14
	s_wait_dscnt 0x19
	v_fma_f32 v15, -v122, v96, v15
	s_wait_dscnt 0x18
	v_fma_f32 v16, -v122, v97, v16
	s_wait_dscnt 0x17
	v_fma_f32 v17, -v122, v98, v17
	s_wait_dscnt 0x16
	v_fma_f32 v18, -v122, v99, v18
	s_wait_dscnt 0x15
	v_fma_f32 v19, -v122, v100, v19
	s_wait_dscnt 0x14
	v_fma_f32 v20, -v122, v101, v20
	s_wait_dscnt 0x13
	v_fma_f32 v21, -v122, v102, v21
	s_wait_dscnt 0x12
	v_fma_f32 v22, -v122, v103, v22
	s_wait_dscnt 0x11
	v_fma_f32 v23, -v122, v104, v23
	s_wait_dscnt 0x10
	v_fma_f32 v24, -v122, v33, v24
	s_wait_dscnt 0xf
	v_fma_f32 v33, -v122, v105, v25
	s_wait_dscnt 0xe
	v_fma_f32 v34, -v122, v106, v26
	s_wait_dscnt 0xd
	v_fma_f32 v35, -v122, v107, v27
	s_wait_dscnt 0xc
	v_fma_f32 v36, -v122, v108, v28
	s_wait_dscnt 0xb
	v_fma_f32 v37, -v122, v109, v29
	s_wait_dscnt 0xa
	v_fma_f32 v38, -v122, v110, v30
	s_wait_dscnt 0x9
	v_fma_f32 v39, -v122, v111, v31
	s_wait_dscnt 0x8
	v_fma_f32 v40, -v122, v112, v32
	s_wait_dscnt 0x7
	v_fma_f32 v41, -v122, v113, v41
	s_wait_dscnt 0x6
	v_fma_f32 v42, -v122, v114, v42
	s_wait_dscnt 0x5
	v_fma_f32 v43, -v122, v115, v43
	s_wait_dscnt 0x4
	v_fma_f32 v44, -v122, v116, v44
	s_wait_dscnt 0x3
	v_fma_f32 v45, -v122, v117, v45
	s_wait_dscnt 0x2
	v_fma_f32 v46, -v122, v118, v46
	s_wait_dscnt 0x1
	v_fma_f32 v47, -v122, v119, v47
	s_wait_dscnt 0x0
	v_fma_f32 v48, -v122, v121, v48
	;;#ASMSTART
	;;#ASMEND
	ds_load_2addr_b32 v[25:26], v120 offset0:192 offset1:224
	v_cndmask_b32_e32 v27, 0, v20, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v27, v27, v44, s0
	ds_bpermute_b32 v28, v76, v27
	s_wait_dscnt 0x1
	ds_bpermute_b32 v29, v92, v25
	ds_bpermute_b32 v30, v77, v25
	ds_bpermute_b32 v31, v78, v25
	ds_bpermute_b32 v32, v79, v25
	ds_bpermute_b32 v94, v80, v25
	ds_bpermute_b32 v95, v81, v25
	ds_bpermute_b32 v96, v82, v25
	ds_bpermute_b32 v97, v83, v25
	ds_bpermute_b32 v98, v84, v25
	ds_bpermute_b32 v99, v85, v25
	ds_bpermute_b32 v100, v86, v25
	ds_bpermute_b32 v101, v87, v25
	ds_bpermute_b32 v102, v88, v25
	ds_bpermute_b32 v103, v89, v25
	ds_bpermute_b32 v104, v90, v25
	ds_bpermute_b32 v105, v91, v25
	ds_bpermute_b32 v106, v92, v26
	ds_bpermute_b32 v107, v77, v26
	ds_bpermute_b32 v108, v78, v26
	ds_bpermute_b32 v109, v79, v26
	ds_bpermute_b32 v110, v80, v26
	ds_bpermute_b32 v111, v81, v26
	ds_bpermute_b32 v112, v82, v26
	ds_bpermute_b32 v113, v83, v26
	ds_bpermute_b32 v114, v84, v26
	ds_bpermute_b32 v115, v85, v26
	ds_bpermute_b32 v116, v86, v26
	ds_bpermute_b32 v117, v87, v26
	ds_bpermute_b32 v118, v88, v26
	ds_bpermute_b32 v119, v89, v26
	ds_bpermute_b32 v120, v90, v26
	ds_bpermute_b32 v121, v91, v26
	s_wait_dscnt 0x20
	v_add_f32_e32 v122, v27, v28
	s_wait_dscnt 0x1f
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v9, -v122, v29, v9
	s_wait_dscnt 0x1e
	v_fma_f32 v10, -v122, v30, v10
	s_wait_dscnt 0x1d
	v_fma_f32 v11, -v122, v31, v11
	s_wait_dscnt 0x1c
	v_fma_f32 v12, -v122, v32, v12
	s_wait_dscnt 0x1b
	v_fma_f32 v13, -v122, v94, v13
	s_wait_dscnt 0x1a
	v_fma_f32 v14, -v122, v95, v14
	s_wait_dscnt 0x19
	v_fma_f32 v15, -v122, v96, v15
	s_wait_dscnt 0x18
	v_fma_f32 v16, -v122, v97, v16
	s_wait_dscnt 0x17
	v_fma_f32 v25, -v122, v98, v17
	s_wait_dscnt 0x16
	v_fma_f32 v26, -v122, v99, v18
	s_wait_dscnt 0x15
	v_fma_f32 v27, -v122, v100, v19
	s_wait_dscnt 0x14
	v_fma_f32 v28, -v122, v101, v20
	s_wait_dscnt 0x13
	v_fma_f32 v29, -v122, v102, v21
	s_wait_dscnt 0x12
	v_fma_f32 v30, -v122, v103, v22
	s_wait_dscnt 0x11
	v_fma_f32 v31, -v122, v104, v23
	s_wait_dscnt 0x10
	v_fma_f32 v32, -v122, v105, v24
	s_wait_dscnt 0xf
	v_fma_f32 v33, -v122, v106, v33
	s_wait_dscnt 0xe
	v_fma_f32 v34, -v122, v107, v34
	s_wait_dscnt 0xd
	v_fma_f32 v35, -v122, v108, v35
	s_wait_dscnt 0xc
	v_fma_f32 v36, -v122, v109, v36
	s_wait_dscnt 0xb
	v_fma_f32 v37, -v122, v110, v37
	s_wait_dscnt 0xa
	v_fma_f32 v38, -v122, v111, v38
	s_wait_dscnt 0x9
	v_fma_f32 v39, -v122, v112, v39
	s_wait_dscnt 0x8
	v_fma_f32 v40, -v122, v113, v40
	s_wait_dscnt 0x7
	v_fma_f32 v17, -v122, v114, v41
	s_wait_dscnt 0x6
	v_fma_f32 v18, -v122, v115, v42
	s_wait_dscnt 0x5
	v_fma_f32 v19, -v122, v116, v43
	s_wait_dscnt 0x4
	v_fma_f32 v20, -v122, v117, v44
	s_wait_dscnt 0x3
	v_fma_f32 v21, -v122, v118, v45
	s_wait_dscnt 0x2
	v_fma_f32 v22, -v122, v119, v46
	s_wait_dscnt 0x1
	v_fma_f32 v23, -v122, v120, v47
	s_wait_dscnt 0x0
	v_fma_f32 v24, -v122, v121, v48
	;;#ASMSTART
	;;#ASMEND
	ds_load_2addr_b32 v[41:42], v93 offset1:32
	v_cndmask_b32_e32 v43, 0, v29, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v43, v43, v21, s0
	ds_bpermute_b32 v44, v76, v43
	s_wait_dscnt 0x1
	ds_bpermute_b32 v45, v92, v41
	ds_bpermute_b32 v46, v77, v41
	ds_bpermute_b32 v47, v78, v41
	ds_bpermute_b32 v48, v79, v41
	ds_bpermute_b32 v94, v80, v41
	ds_bpermute_b32 v95, v81, v41
	ds_bpermute_b32 v96, v82, v41
	ds_bpermute_b32 v97, v83, v41
	ds_bpermute_b32 v98, v84, v41
	ds_bpermute_b32 v99, v85, v41
	ds_bpermute_b32 v100, v86, v41
	ds_bpermute_b32 v101, v87, v41
	ds_bpermute_b32 v102, v88, v41
	ds_bpermute_b32 v103, v89, v41
	ds_bpermute_b32 v104, v90, v41
	ds_bpermute_b32 v41, v91, v41
	ds_bpermute_b32 v105, v92, v42
	ds_bpermute_b32 v106, v77, v42
	ds_bpermute_b32 v107, v78, v42
	ds_bpermute_b32 v108, v79, v42
	ds_bpermute_b32 v109, v80, v42
	ds_bpermute_b32 v110, v81, v42
	ds_bpermute_b32 v111, v82, v42
	ds_bpermute_b32 v112, v83, v42
	ds_bpermute_b32 v113, v84, v42
	ds_bpermute_b32 v114, v85, v42
	ds_bpermute_b32 v115, v86, v42
	ds_bpermute_b32 v116, v87, v42
	ds_bpermute_b32 v117, v88, v42
	ds_bpermute_b32 v118, v89, v42
	ds_bpermute_b32 v119, v90, v42
	ds_bpermute_b32 v42, v91, v42
	s_wait_dscnt 0x20
	v_add_f32_e32 v43, v43, v44
	s_wait_dscnt 0x1f
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v9, -v43, v45, v9
	s_wait_dscnt 0x1e
	v_fma_f32 v10, -v43, v46, v10
	s_wait_dscnt 0x1d
	v_fma_f32 v11, -v43, v47, v11
	s_wait_dscnt 0x1c
	v_fma_f32 v12, -v43, v48, v12
	s_wait_dscnt 0x1b
	v_fma_f32 v13, -v43, v94, v13
	s_wait_dscnt 0x1a
	v_fma_f32 v14, -v43, v95, v14
	s_wait_dscnt 0x19
	v_fma_f32 v15, -v43, v96, v15
	s_wait_dscnt 0x18
	v_fma_f32 v16, -v43, v97, v16
	s_wait_dscnt 0x17
	v_fma_f32 v25, -v43, v98, v25
	s_wait_dscnt 0x16
	v_fma_f32 v26, -v43, v99, v26
	s_wait_dscnt 0x15
	v_fma_f32 v27, -v43, v100, v27
	s_wait_dscnt 0x14
	v_fma_f32 v28, -v43, v101, v28
	s_wait_dscnt 0x13
	v_fma_f32 v29, -v43, v102, v29
	s_wait_dscnt 0x12
	v_fma_f32 v30, -v43, v103, v30
	s_wait_dscnt 0x11
	v_fma_f32 v31, -v43, v104, v31
	s_wait_dscnt 0x10
	v_fma_f32 v32, -v43, v41, v32
	s_wait_dscnt 0xf
	v_fma_f32 v33, -v43, v105, v33
	s_wait_dscnt 0xe
	v_fma_f32 v34, -v43, v106, v34
	s_wait_dscnt 0xd
	v_fma_f32 v35, -v43, v107, v35
	s_wait_dscnt 0xc
	v_fma_f32 v36, -v43, v108, v36
	s_wait_dscnt 0xb
	v_fma_f32 v37, -v43, v109, v37
	s_wait_dscnt 0xa
	v_fma_f32 v38, -v43, v110, v38
	s_wait_dscnt 0x9
	v_fma_f32 v39, -v43, v111, v39
	s_wait_dscnt 0x8
	v_fma_f32 v40, -v43, v112, v40
	s_wait_dscnt 0x7
	v_fma_f32 v17, -v43, v113, v17
	s_wait_dscnt 0x6
	v_fma_f32 v18, -v43, v114, v18
	s_wait_dscnt 0x5
	v_fma_f32 v19, -v43, v115, v19
	s_wait_dscnt 0x4
	v_fma_f32 v20, -v43, v116, v20
	s_wait_dscnt 0x3
	v_fma_f32 v21, -v43, v117, v21
	s_wait_dscnt 0x2
	v_fma_f32 v22, -v43, v118, v22
	s_wait_dscnt 0x1
	v_fma_f32 v23, -v43, v119, v23
	s_wait_dscnt 0x0
	v_fma_f32 v24, -v43, v42, v24
	;;#ASMSTART
	;;#ASMEND
	ds_load_2addr_b32 v[41:42], v93 offset0:64 offset1:96
	v_cndmask_b32_e32 v43, 0, v30, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v43, v43, v22, s0
	ds_bpermute_b32 v44, v76, v43
	s_wait_dscnt 0x1
	ds_bpermute_b32 v45, v92, v41
	ds_bpermute_b32 v46, v77, v41
	ds_bpermute_b32 v47, v78, v41
	ds_bpermute_b32 v48, v79, v41
	ds_bpermute_b32 v94, v80, v41
	ds_bpermute_b32 v95, v81, v41
	ds_bpermute_b32 v96, v82, v41
	ds_bpermute_b32 v97, v83, v41
	ds_bpermute_b32 v98, v84, v41
	ds_bpermute_b32 v99, v85, v41
	ds_bpermute_b32 v100, v86, v41
	ds_bpermute_b32 v101, v87, v41
	ds_bpermute_b32 v102, v88, v41
	ds_bpermute_b32 v103, v89, v41
	ds_bpermute_b32 v104, v90, v41
	ds_bpermute_b32 v41, v91, v41
	ds_bpermute_b32 v105, v92, v42
	ds_bpermute_b32 v106, v77, v42
	ds_bpermute_b32 v107, v78, v42
	ds_bpermute_b32 v108, v79, v42
	ds_bpermute_b32 v109, v80, v42
	ds_bpermute_b32 v110, v81, v42
	ds_bpermute_b32 v111, v82, v42
	ds_bpermute_b32 v112, v83, v42
	ds_bpermute_b32 v113, v84, v42
	ds_bpermute_b32 v114, v85, v42
	ds_bpermute_b32 v115, v86, v42
	ds_bpermute_b32 v116, v87, v42
	ds_bpermute_b32 v117, v88, v42
	ds_bpermute_b32 v118, v89, v42
	ds_bpermute_b32 v119, v90, v42
	ds_bpermute_b32 v42, v91, v42
	s_wait_dscnt 0x20
	v_add_f32_e32 v43, v43, v44
	s_wait_dscnt 0x1f
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v9, -v43, v45, v9
	s_wait_dscnt 0x1e
	v_fma_f32 v10, -v43, v46, v10
	s_wait_dscnt 0x1d
	v_fma_f32 v11, -v43, v47, v11
	s_wait_dscnt 0x1c
	v_fma_f32 v12, -v43, v48, v12
	s_wait_dscnt 0x1b
	v_fma_f32 v13, -v43, v94, v13
	s_wait_dscnt 0x1a
	v_fma_f32 v14, -v43, v95, v14
	s_wait_dscnt 0x19
	v_fma_f32 v15, -v43, v96, v15
	s_wait_dscnt 0x18
	v_fma_f32 v16, -v43, v97, v16
	s_wait_dscnt 0x17
	v_fma_f32 v25, -v43, v98, v25
	s_wait_dscnt 0x16
	v_fma_f32 v26, -v43, v99, v26
	s_wait_dscnt 0x15
	v_fma_f32 v27, -v43, v100, v27
	s_wait_dscnt 0x14
	v_fma_f32 v28, -v43, v101, v28
	s_wait_dscnt 0x13
	v_fma_f32 v29, -v43, v102, v29
	s_wait_dscnt 0x12
	v_fma_f32 v30, -v43, v103, v30
	s_wait_dscnt 0x11
	v_fma_f32 v31, -v43, v104, v31
	s_wait_dscnt 0x10
	v_fma_f32 v32, -v43, v41, v32
	s_wait_dscnt 0xf
	v_fma_f32 v33, -v43, v105, v33
	s_wait_dscnt 0xe
	v_fma_f32 v34, -v43, v106, v34
	s_wait_dscnt 0xd
	v_fma_f32 v35, -v43, v107, v35
	s_wait_dscnt 0xc
	v_fma_f32 v36, -v43, v108, v36
	s_wait_dscnt 0xb
	v_fma_f32 v37, -v43, v109, v37
	s_wait_dscnt 0xa
	v_fma_f32 v38, -v43, v110, v38
	s_wait_dscnt 0x9
	v_fma_f32 v39, -v43, v111, v39
	s_wait_dscnt 0x8
	v_fma_f32 v40, -v43, v112, v40
	s_wait_dscnt 0x7
	v_fma_f32 v17, -v43, v113, v17
	s_wait_dscnt 0x6
	v_fma_f32 v18, -v43, v114, v18
	s_wait_dscnt 0x5
	v_fma_f32 v19, -v43, v115, v19
	s_wait_dscnt 0x4
	v_fma_f32 v20, -v43, v116, v20
	s_wait_dscnt 0x3
	v_fma_f32 v21, -v43, v117, v21
	s_wait_dscnt 0x2
	v_fma_f32 v22, -v43, v118, v22
	s_wait_dscnt 0x1
	v_fma_f32 v23, -v43, v119, v23
	s_wait_dscnt 0x0
	v_fma_f32 v24, -v43, v42, v24
	;;#ASMSTART
	;;#ASMEND
	ds_load_2addr_b32 v[41:42], v93 offset0:128 offset1:160
	v_cndmask_b32_e32 v43, 0, v31, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v43, v43, v23, s0
	ds_bpermute_b32 v44, v76, v43
	s_wait_dscnt 0x1
	ds_bpermute_b32 v45, v92, v41
	ds_bpermute_b32 v46, v77, v41
	ds_bpermute_b32 v47, v78, v41
	ds_bpermute_b32 v48, v79, v41
	ds_bpermute_b32 v94, v80, v41
	ds_bpermute_b32 v95, v81, v41
	ds_bpermute_b32 v96, v82, v41
	ds_bpermute_b32 v97, v83, v41
	ds_bpermute_b32 v98, v84, v41
	ds_bpermute_b32 v99, v85, v41
	ds_bpermute_b32 v100, v86, v41
	ds_bpermute_b32 v101, v87, v41
	ds_bpermute_b32 v102, v88, v41
	ds_bpermute_b32 v103, v89, v41
	ds_bpermute_b32 v104, v90, v41
	ds_bpermute_b32 v41, v91, v41
	ds_bpermute_b32 v105, v92, v42
	ds_bpermute_b32 v106, v77, v42
	ds_bpermute_b32 v107, v78, v42
	ds_bpermute_b32 v108, v79, v42
	ds_bpermute_b32 v109, v80, v42
	ds_bpermute_b32 v110, v81, v42
	ds_bpermute_b32 v111, v82, v42
	ds_bpermute_b32 v112, v83, v42
	ds_bpermute_b32 v113, v84, v42
	ds_bpermute_b32 v114, v85, v42
	ds_bpermute_b32 v115, v86, v42
	ds_bpermute_b32 v116, v87, v42
	ds_bpermute_b32 v117, v88, v42
	ds_bpermute_b32 v118, v89, v42
	ds_bpermute_b32 v119, v90, v42
	ds_bpermute_b32 v120, v91, v42
	s_wait_dscnt 0x20
	v_add_f32_e32 v121, v43, v44
	s_wait_dscnt 0x1f
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v9, -v121, v45, v9
	s_wait_dscnt 0x1e
	v_fma_f32 v10, -v121, v46, v10
	s_wait_dscnt 0x1d
	v_fma_f32 v11, -v121, v47, v11
	s_wait_dscnt 0x1c
	v_fma_f32 v12, -v121, v48, v12
	s_wait_dscnt 0x1b
	v_fma_f32 v13, -v121, v94, v13
	s_wait_dscnt 0x1a
	v_fma_f32 v14, -v121, v95, v14
	s_wait_dscnt 0x19
	v_fma_f32 v15, -v121, v96, v15
	s_wait_dscnt 0x18
	v_fma_f32 v16, -v121, v97, v16
	s_wait_dscnt 0x17
	v_fma_f32 v25, -v121, v98, v25
	s_wait_dscnt 0x16
	v_fma_f32 v26, -v121, v99, v26
	s_wait_dscnt 0x15
	v_fma_f32 v27, -v121, v100, v27
	s_wait_dscnt 0x14
	v_fma_f32 v28, -v121, v101, v28
	s_wait_dscnt 0x13
	v_fma_f32 v29, -v121, v102, v29
	s_wait_dscnt 0x12
	v_fma_f32 v30, -v121, v103, v30
	s_wait_dscnt 0x11
	v_fma_f32 v31, -v121, v104, v31
	s_wait_dscnt 0x10
	v_fma_f32 v32, -v121, v41, v32
	s_wait_dscnt 0xf
	v_fma_f32 v41, -v121, v105, v33
	s_wait_dscnt 0xe
	v_fma_f32 v42, -v121, v106, v34
	s_wait_dscnt 0xd
	v_fma_f32 v43, -v121, v107, v35
	s_wait_dscnt 0xc
	v_fma_f32 v44, -v121, v108, v36
	s_wait_dscnt 0xb
	v_fma_f32 v45, -v121, v109, v37
	s_wait_dscnt 0xa
	v_fma_f32 v46, -v121, v110, v38
	s_wait_dscnt 0x9
	v_fma_f32 v47, -v121, v111, v39
	s_wait_dscnt 0x8
	v_fma_f32 v48, -v121, v112, v40
	s_wait_dscnt 0x7
	v_fma_f32 v94, -v121, v113, v17
	s_wait_dscnt 0x6
	v_fma_f32 v95, -v121, v114, v18
	s_wait_dscnt 0x5
	v_fma_f32 v96, -v121, v115, v19
	s_wait_dscnt 0x4
	v_fma_f32 v97, -v121, v116, v20
	s_wait_dscnt 0x3
	v_fma_f32 v98, -v121, v117, v21
	s_wait_dscnt 0x2
	v_fma_f32 v99, -v121, v118, v22
	s_wait_dscnt 0x1
	v_fma_f32 v100, -v121, v119, v23
	s_wait_dscnt 0x0
	v_fma_f32 v101, -v121, v120, v24
	;;#ASMSTART
	;;#ASMEND
	ds_load_2addr_b32 v[17:18], v93 offset0:192 offset1:224
	v_cndmask_b32_e32 v19, 0, v32, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v19, v19, v101, s0
	ds_bpermute_b32 v20, v76, v19
	s_wait_dscnt 0x1
	ds_bpermute_b32 v21, v92, v17
	ds_bpermute_b32 v22, v77, v17
	ds_bpermute_b32 v23, v78, v17
	ds_bpermute_b32 v24, v79, v17
	ds_bpermute_b32 v37, v80, v17
	ds_bpermute_b32 v38, v81, v17
	ds_bpermute_b32 v39, v82, v17
	ds_bpermute_b32 v40, v83, v17
	ds_bpermute_b32 v93, v84, v17
	ds_bpermute_b32 v102, v85, v17
	ds_bpermute_b32 v103, v86, v17
	ds_bpermute_b32 v104, v87, v17
	ds_bpermute_b32 v105, v88, v17
	ds_bpermute_b32 v106, v89, v17
	ds_bpermute_b32 v107, v90, v17
	ds_bpermute_b32 v17, v91, v17
	ds_bpermute_b32 v92, v92, v18
	ds_bpermute_b32 v108, v77, v18
	ds_bpermute_b32 v109, v78, v18
	ds_bpermute_b32 v110, v79, v18
	ds_bpermute_b32 v111, v80, v18
	ds_bpermute_b32 v112, v81, v18
	ds_bpermute_b32 v113, v82, v18
	ds_bpermute_b32 v114, v83, v18
	ds_bpermute_b32 v115, v84, v18
	ds_bpermute_b32 v116, v85, v18
	ds_bpermute_b32 v117, v86, v18
	ds_bpermute_b32 v118, v87, v18
	ds_bpermute_b32 v119, v88, v18
	ds_bpermute_b32 v120, v89, v18
	ds_bpermute_b32 v121, v90, v18
	ds_bpermute_b32 v122, v91, v18
	s_wait_dscnt 0x20
	v_add_f32_e32 v123, v19, v20
	s_wait_dscnt 0x1f
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_f32 v33, -v123, v21, v9
	s_wait_dscnt 0x1e
	v_fma_f32 v34, -v123, v22, v10
	s_wait_dscnt 0x1d
	v_fma_f32 v35, -v123, v23, v11
	s_wait_dscnt 0x1c
	v_fma_f32 v36, -v123, v24, v12
	s_wait_dscnt 0x1b
	v_fma_f32 v37, -v123, v37, v13
	s_wait_dscnt 0x1a
	v_fma_f32 v38, -v123, v38, v14
	s_wait_dscnt 0x19
	v_fma_f32 v39, -v123, v39, v15
	s_wait_dscnt 0x18
	v_fma_f32 v40, -v123, v40, v16
	s_wait_dscnt 0x17
	v_fma_f32 v25, -v123, v93, v25
	s_wait_dscnt 0x16
	v_fma_f32 v26, -v123, v102, v26
	s_wait_dscnt 0x15
	v_fma_f32 v27, -v123, v103, v27
	s_wait_dscnt 0x14
	v_fma_f32 v28, -v123, v104, v28
	s_wait_dscnt 0x13
	v_fma_f32 v29, -v123, v105, v29
	s_wait_dscnt 0x12
	v_fma_f32 v30, -v123, v106, v30
	s_wait_dscnt 0x11
	v_fma_f32 v31, -v123, v107, v31
	s_wait_dscnt 0x10
	v_fma_f32 v32, -v123, v17, v32
	s_wait_dscnt 0xf
	v_fma_f32 v17, -v123, v92, v41
	s_wait_dscnt 0xe
	v_fma_f32 v18, -v123, v108, v42
	s_wait_dscnt 0xd
	v_fma_f32 v19, -v123, v109, v43
	s_wait_dscnt 0xc
	v_fma_f32 v20, -v123, v110, v44
	s_wait_dscnt 0xb
	v_fma_f32 v21, -v123, v111, v45
	s_wait_dscnt 0xa
	v_fma_f32 v22, -v123, v112, v46
	s_wait_dscnt 0x9
	v_fma_f32 v23, -v123, v113, v47
	s_wait_dscnt 0x8
	v_fma_f32 v24, -v123, v114, v48
	s_wait_dscnt 0x7
	v_fma_f32 v9, -v123, v115, v94
	s_wait_dscnt 0x6
	v_fma_f32 v10, -v123, v116, v95
	s_wait_dscnt 0x5
	v_fma_f32 v11, -v123, v117, v96
	s_wait_dscnt 0x4
	v_fma_f32 v12, -v123, v118, v97
	s_wait_dscnt 0x3
	v_fma_f32 v13, -v123, v119, v98
	s_wait_dscnt 0x2
	v_fma_f32 v14, -v123, v120, v99
	s_wait_dscnt 0x1
	v_fma_f32 v15, -v123, v121, v100
	s_wait_dscnt 0x0
	v_fma_f32 v16, -v123, v122, v101
	;;#ASMSTART
	;;#ASMEND
	s_cbranch_scc0 .LBB7_27
; %bb.28:                               ;   in Loop: Header=BB7_26 Depth=1
	v_wmma_f32_16x16x16_f16 v[41:48], v[53:56], v[61:64], v[1:8]
	v_cvt_f16_f32_e32 v95.h, v40
	v_cvt_f16_f32_e32 v95.l, v39
	v_cvt_f16_f32_e32 v94.h, v38
	s_delay_alu instid0(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[41:48], v[53:56], v[65:68], v[41:48]
	v_cvt_f16_f32_e32 v94.l, v37
	v_cvt_f16_f32_e32 v93.h, v36
	v_cvt_f16_f32_e32 v93.l, v35
	v_cvt_f16_f32_e32 v92.h, v34
	v_wmma_f32_16x16x16_f16 v[41:48], v[53:56], v[69:72], v[41:48]
	v_cvt_f16_f32_e32 v92.l, v33
	v_cvt_f16_f32_e32 v99.h, v32
	v_cvt_f16_f32_e32 v99.l, v31
	v_cvt_f16_f32_e32 v98.h, v30
	v_wmma_f32_16x16x16_f16 v[41:48], v[53:56], v[57:60], v[41:48]
	v_cvt_f16_f32_e32 v98.l, v29
	v_cvt_f16_f32_e32 v97.h, v28
	v_cvt_f16_f32_e32 v97.l, v27
	v_cvt_f16_f32_e32 v96.h, v26
	v_cvt_f16_f32_e32 v96.l, v25
	v_wmma_f32_16x16x16_f16 v[41:48], v[49:52], v[92:95], v[41:48]
	v_cvt_f16_f32_e32 v103.h, v24
	v_cvt_f16_f32_e32 v103.l, v23
	v_cvt_f16_f32_e32 v102.h, v22
	v_cvt_f16_f32_e32 v102.l, v21
	v_wmma_f32_16x16x16_f16 v[41:48], v[49:52], v[96:99], v[41:48]
	v_cvt_f16_f32_e32 v101.h, v20
	v_cvt_f16_f32_e32 v101.l, v19
	v_cvt_f16_f32_e32 v100.h, v18
	v_cvt_f16_f32_e32 v100.l, v17
	v_cvt_f16_f32_e32 v107.h, v16
	v_cvt_f16_f32_e32 v107.l, v15
	v_cvt_f16_f32_e32 v106.h, v14
	v_cvt_f16_f32_e32 v106.l, v13
	v_wmma_f32_16x16x16_f16 v[41:48], v[49:52], v[100:103], v[41:48]
	v_cvt_f16_f32_e32 v105.h, v12
	v_cvt_f16_f32_e32 v105.l, v11
	v_cvt_f16_f32_e32 v104.h, v10
	v_cvt_f16_f32_e32 v104.l, v9
	v_dual_add_f32 v16, v8, v8 :: v_dual_add_f32 v15, v7, v7
	v_dual_add_f32 v14, v6, v6 :: v_dual_add_f32 v13, v5, v5
	v_dual_add_f32 v12, v4, v4 :: v_dual_add_f32 v11, v3, v3
	v_dual_add_f32 v10, v2, v2 :: v_dual_add_f32 v9, v1, v1
	v_wmma_f32_16x16x16_f16 v[41:48], v[49:52], v[104:107], v[41:48]
	v_mul_f32_e32 v24, 0x40400000, v8
	v_mul_f32_e32 v20, 0x40400000, v4
	v_mul_f32_e32 v22, 0x40400000, v6
	v_wmma_f32_16x16x16_f16 v[9:16], v[53:56], v[61:64], v[9:16]
	v_add_f32_e32 v17, 0, v41
	v_dual_mul_f32 v23, 0x40400000, v7 :: v_dual_mul_f32 v6, 4.0, v6
	v_mul_f32_e32 v18, 0x40400000, v2
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[9:16], v[53:56], v[65:68], v[9:16]
	v_add_f32_e32 v17, v17, v42
	v_dual_mul_f32 v21, 0x40400000, v5 :: v_dual_mul_f32 v2, 4.0, v2
	v_mul_f32_e32 v8, 4.0, v8
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[9:16], v[53:56], v[69:72], v[9:16]
	v_add_f32_e32 v17, v17, v43
	v_dual_mul_f32 v19, 0x40400000, v3 :: v_dual_mul_f32 v4, 4.0, v4
	v_mul_f32_e32 v7, 4.0, v7
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[9:16], v[53:56], v[57:60], v[9:16]
	v_add_f32_e32 v17, v17, v44
	v_mul_f32_e32 v5, 4.0, v5
	v_mul_f32_e32 v3, 4.0, v3
	v_fma_mix_f32 v32, v64, s7, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_wmma_f32_16x16x16_f16 v[9:16], v[49:52], v[92:95], v[9:16]
	v_add_f32_e32 v17, v17, v45
	v_fma_mix_f32 v31, v64, s7, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v30, v63, s7, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v29, v63, s7, neg(0) op_sel_hi:[1,0,0]
	v_wmma_f32_16x16x16_f16 v[9:16], v[49:52], v[96:99], v[9:16]
	v_add_f32_e32 v17, v17, v46
	v_fma_mix_f32 v28, v62, s7, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v27, v62, s7, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v26, v61, s7, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_wmma_f32_16x16x16_f16 v[9:16], v[49:52], v[100:103], v[9:16]
	v_add_f32_e32 v25, v17, v47
	v_mul_f32_e32 v17, 0x40400000, v1
	v_mul_f32_e32 v1, 4.0, v1
	v_fma_mix_f32 v40, v68, s7, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_wmma_f32_16x16x16_f16 v[9:16], v[49:52], v[104:107], v[9:16]
	v_add_f32_e32 v25, v25, v48
	v_wmma_f32_16x16x16_f16 v[17:24], v[53:56], v[61:64], v[17:24]
	v_wmma_f32_16x16x16_f16 v[1:8], v[53:56], v[61:64], v[1:8]
	v_fma_mix_f32 v39, v68, s7, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v38, v67, s7, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_add_f32_e32 v9, v25, v9
	v_wmma_f32_16x16x16_f16 v[17:24], v[53:56], v[65:68], v[17:24]
	v_wmma_f32_16x16x16_f16 v[1:8], v[53:56], v[65:68], v[1:8]
	v_fma_mix_f32 v25, v61, s7, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v37, v67, s7, neg(0) op_sel_hi:[1,0,0]
	v_add_f32_e32 v9, v9, v10
	v_wmma_f32_16x16x16_f16 v[17:24], v[53:56], v[69:72], v[17:24]
	v_wmma_f32_16x16x16_f16 v[1:8], v[53:56], v[69:72], v[1:8]
	v_fma_mix_f32 v36, v66, s7, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v35, v66, s7, neg(0) op_sel_hi:[1,0,0]
	v_add_f32_e32 v9, v9, v11
	v_wmma_f32_16x16x16_f16 v[17:24], v[53:56], v[57:60], v[17:24]
	v_wmma_f32_16x16x16_f16 v[1:8], v[53:56], v[57:60], v[1:8]
	v_fma_mix_f32 v34, v65, s7, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v33, v65, s7, neg(0) op_sel_hi:[1,0,0]
	v_add_f32_e32 v9, v9, v12
	v_wmma_f32_16x16x16_f16 v[17:24], v[49:52], v[92:95], v[17:24]
	v_wmma_f32_16x16x16_f16 v[1:8], v[49:52], v[92:95], v[1:8]
	v_fma_mix_f32 v48, v72, s7, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v47, v72, s7, neg(0) op_sel_hi:[1,0,0]
	v_add_f32_e32 v9, v9, v13
	v_wmma_f32_16x16x16_f16 v[17:24], v[49:52], v[96:99], v[17:24]
	v_wmma_f32_16x16x16_f16 v[1:8], v[49:52], v[96:99], v[1:8]
	v_fma_mix_f32 v46, v71, s7, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v45, v71, s7, neg(0) op_sel_hi:[1,0,0]
	v_add_f32_e32 v9, v9, v14
	v_wmma_f32_16x16x16_f16 v[17:24], v[49:52], v[100:103], v[17:24]
	v_wmma_f32_16x16x16_f16 v[1:8], v[49:52], v[100:103], v[1:8]
	v_fma_mix_f32 v44, v70, s7, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v43, v70, s7, neg(0) op_sel_hi:[1,0,0]
	v_add_f32_e32 v9, v9, v15
	v_wmma_f32_16x16x16_f16 v[17:24], v[49:52], v[104:107], v[17:24]
	v_wmma_f32_16x16x16_f16 v[1:8], v[49:52], v[104:107], v[1:8]
	v_fma_mix_f32 v42, v69, s7, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v41, v69, s7, neg(0) op_sel_hi:[1,0,0]
	v_add_f32_e32 v9, v9, v16
	v_fma_mix_f32 v16, v60, s7, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v15, v60, s7, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v14, v59, s7, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v13, v59, s7, neg(0) op_sel_hi:[1,0,0]
	v_add_f32_e32 v9, v9, v17
	v_fma_mix_f32 v12, v58, s7, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v11, v58, s7, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v10, v57, s7, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_wmma_f32_16x16x16_f16 v[25:32], v[49:52], v[92:95], v[25:32]
	v_add_f32_e32 v9, v9, v18
	v_wmma_f32_16x16x16_f16 v[33:40], v[49:52], v[92:95], v[33:40]
	v_wmma_f32_16x16x16_f16 v[41:48], v[49:52], v[92:95], v[41:48]
	v_mov_b32_e32 v18, 0x3b03126f
	v_wmma_f32_16x16x16_f16 v[25:32], v[49:52], v[96:99], v[25:32]
	v_add_f32_e32 v9, v9, v19
	v_wmma_f32_16x16x16_f16 v[33:40], v[49:52], v[96:99], v[33:40]
	v_wmma_f32_16x16x16_f16 v[41:48], v[49:52], v[96:99], v[41:48]
	s_add_co_i32 s6, s6, 1
	v_wmma_f32_16x16x16_f16 v[25:32], v[49:52], v[100:103], v[25:32]
	v_add_f32_e32 v17, v9, v20
	v_fma_mix_f32 v9, v57, s7, neg(0) op_sel_hi:[1,0,0]
	v_wmma_f32_16x16x16_f16 v[33:40], v[49:52], v[100:103], v[33:40]
	v_wmma_f32_16x16x16_f16 v[41:48], v[49:52], v[100:103], v[41:48]
	v_wmma_f32_16x16x16_f16 v[25:32], v[49:52], v[104:107], v[25:32]
	v_add_f32_e32 v17, v17, v21
	v_wmma_f32_16x16x16_f16 v[9:16], v[49:52], v[92:95], v[9:16]
	v_wmma_f32_16x16x16_f16 v[33:40], v[49:52], v[104:107], v[33:40]
	v_wmma_f32_16x16x16_f16 v[41:48], v[49:52], v[104:107], v[41:48]
	v_cvt_f16_f32_e32 v61.h, v26
	v_add_f32_e32 v17, v17, v22
	v_wmma_f32_16x16x16_f16 v[9:16], v[49:52], v[96:99], v[9:16]
	v_cvt_f16_f32_e32 v61.l, v25
	v_cvt_f16_f32_e32 v62.h, v28
	v_cvt_f16_f32_e32 v62.l, v27
	v_add_f32_e32 v17, v17, v23
	v_wmma_f32_16x16x16_f16 v[9:16], v[49:52], v[100:103], v[9:16]
	v_cvt_f16_f32_e32 v63.h, v30
	v_cvt_f16_f32_e32 v63.l, v29
	v_cvt_f16_f32_e32 v64.h, v32
	v_add_f32_e32 v17, v17, v24
	v_wmma_f32_16x16x16_f16 v[9:16], v[49:52], v[104:107], v[9:16]
	v_cvt_f16_f32_e32 v64.l, v31
	v_cvt_f16_f32_e32 v65.h, v34
	v_cvt_f16_f32_e32 v65.l, v33
	v_add_f32_e32 v1, v17, v1
	v_cvt_f16_f32_e32 v66.h, v36
	v_cvt_f16_f32_e32 v66.l, v35
	v_cvt_f16_f32_e32 v67.h, v38
	v_cvt_f16_f32_e32 v67.l, v37
	v_add_f32_e32 v1, v1, v2
	v_cvt_f16_f32_e32 v68.h, v40
	v_cvt_f16_f32_e32 v68.l, v39
	v_cvt_f16_f32_e32 v69.h, v42
	v_cvt_f16_f32_e32 v69.l, v41
	v_add_f32_e32 v1, v1, v3
	v_cvt_f16_f32_e32 v70.h, v44
	v_cvt_f16_f32_e32 v70.l, v43
	v_cvt_f16_f32_e32 v71.h, v46
	v_cvt_f16_f32_e32 v71.l, v45
	v_add_f32_e32 v1, v1, v4
	v_cvt_f16_f32_e32 v72.h, v48
	v_cvt_f16_f32_e32 v72.l, v47
	v_cvt_f16_f32_e32 v58.h, v12
	v_cvt_f16_f32_e32 v58.l, v11
	v_add_f32_e32 v1, v1, v5
	v_cvt_f16_f32_e32 v59.h, v14
	v_cvt_f16_f32_e32 v59.l, v13
	v_cvt_f16_f32_e32 v60.h, v16
	v_cvt_f16_f32_e32 v60.l, v15
	v_add_f32_e32 v1, v1, v6
	v_cvt_f16_f32_e32 v57.h, v10
	v_cvt_f16_f32_e32 v57.l, v9
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s6, s3
	v_add_f32_e32 v1, v1, v7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v8
	v_cmp_lt_f32_e32 vcc_lo, 0x60ad78ec, v1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v1, 0x3a83126f, v18, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v53.l, v1
	s_cbranch_scc0 .LBB7_26
	s_branch .LBB7_30
.LBB7_29:
	v_dual_mov_b32 v59, v68 :: v_dual_mov_b32 v58, v67
	v_dual_mov_b32 v57, v66 :: v_dual_mov_b32 v70, v63
	v_mov_b32_e32 v71, v64
	v_mov_b32_e32 v69, v62
.LBB7_30:
	v_fma_mix_f32 v1, v61, 1.0, 0 op_sel_hi:[1,1,0]
	v_lshl_or_b32 v0, ttmp9, 8, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
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
	v_fma_mix_f32 v1, v57, 1.0, v1 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v1, v57, 1.0, v1 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v1, v58, 1.0, v1 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v1, v58, 1.0, v1 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v1, v59, 1.0, v1 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_fma_mix_f32 v2, v59, 1.0, v1 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mov_b32_e32 v1, 0
	v_fma_mix_f32 v2, v60, 1.0, v2 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	v_fma_mix_f32 v2, v60, 1.0, v2 op_sel:[1,0,0] op_sel_hi:[1,1,0]
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
		.amdhsa_next_free_vgpr 124
		.amdhsa_next_free_sgpr 9
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
	.set .L_Z5probeILi4ELb1ELb1EEvPfPKfi.num_vgpr, 124
	.set .L_Z5probeILi4ELb1ELb1EEvPfPKfi.num_agpr, 0
	.set .L_Z5probeILi4ELb1ELb1EEvPfPKfi.numbered_sgpr, 9
	.set .L_Z5probeILi4ELb1ELb1EEvPfPKfi.num_named_barrier, 0
	.set .L_Z5probeILi4ELb1ELb1EEvPfPKfi.private_seg_size, 0
	.set .L_Z5probeILi4ELb1ELb1EEvPfPKfi.uses_vcc, 1
	.set .L_Z5probeILi4ELb1ELb1EEvPfPKfi.uses_flat_scratch, 0
	.set .L_Z5probeILi4ELb1ELb1EEvPfPKfi.has_dyn_sized_stack, 0
	.set .L_Z5probeILi4ELb1ELb1EEvPfPKfi.has_recursion, 0
	.set .L_Z5probeILi4ELb1ELb1EEvPfPKfi.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 26528
; TotalNumSgprs: 11
; NumVgprs: 124
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 15
; NumSGPRsForWavesPerEU: 11
; NumVGPRsForWavesPerEU: 124
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
	s_load_b32 s3, s[0:1], 0x10
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_mov_b32 s0, 0x3c23d70a
	v_fma_f32 v9, 0x3c23d70a, v19, 0
	v_fmaak_f32 v10, s0, v19, 0x38d1b717
	v_fmaak_f32 v11, s0, v19, 0x3951b717
	v_fmaak_f32 v12, s0, v19, 0x399d4951
	v_fmaak_f32 v13, s0, v19, 0x39d1b717
	v_fmaak_f32 v14, s0, v19, 0x3a03126e
	v_fmaak_f32 v15, s0, v19, 0x3a1d4951
	v_fmaak_f32 v16, s0, v19, 0x3a378034
	v_fmaak_f32 v8, s0, v19, 0x3a51b717
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_kmcnt 0x0
	s_cmp_lt_i32 s3, 1
	s_cbranch_scc1 .LBB8_29
; %bb.25:
	v_mbcnt_lo_u32_b32 v2, -1, 0
	s_mov_b32 s0, 0x3a83126f
	v_and_b32_e32 v1, 31, v0
	s_wait_alu depctr_sa_sdst(0)
	v_fma_mixhi_f16 v57, v4, s0, 0
	v_fma_mixhi_f16 v61, v5, s0, 0
	v_xor_b32_e32 v4, 16, v2
	v_bfi_b32 v5, v2, 0, 32
	v_fma_mixlo_f16 v61, v3, s0, 0
	v_lshrrev_b32_e32 v3, 1, v0
	v_fma_mixhi_f16 v62, v18, s0, 0
	v_dual_mov_b32 v95, 0x3b03126f :: v_dual_lshlrev_b32 v18, 7, v0
	v_cmp_lt_u32_e32 vcc_lo, v4, v5
	s_delay_alu instid0(VALU_DEP_4)
	v_and_b32_e32 v73, 8, v3
	v_fma_mixlo_f16 v57, v19, s0, 0
	v_fma_mixlo_f16 v58, v6, s0, 0
	v_fma_mixlo_f16 v62, v7, s0, 0
	v_cndmask_b32_e32 v2, v2, v4, vcc_lo
	v_fma_mixhi_f16 v58, v17, s0, 0
	v_fma_mixhi_f16 v59, v23, s0, 0
	v_fma_mixhi_f16 v63, v24, s0, 0
	v_fma_mixlo_f16 v60, v25, s0, 0
	v_fma_mixlo_f16 v64, v21, s0, 0
	v_fma_mixhi_f16 v60, v22, s0, 0
	v_fma_mixhi_f16 v64, v20, s0, 0
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
	v_and_b32_e32 v75, 0x800, v18
	v_or_b32_e32 v18, 20, v73
	v_or_b32_e32 v23, 21, v73
	v_or_b32_e32 v24, 22, v73
	v_or_b32_e32 v25, 23, v73
	v_dual_mov_b32 v2, v11 :: v_dual_lshlrev_b32 v79, 2, v2
	v_lshl_add_u32 v74, v1, 2, 0
	v_or_b32_e32 v1, 1, v73
	v_fma_mixlo_f16 v59, v26, s0, 0
	v_fma_mixlo_f16 v63, v27, s0, 0
	v_lshlrev_b32_e32 v81, 2, v3
	v_lshlrev_b32_e32 v82, 2, v4
	v_lshlrev_b32_e32 v80, 2, v1
	v_lshlrev_b32_e32 v83, 2, v5
	v_lshlrev_b32_e32 v84, 2, v6
	v_lshlrev_b32_e32 v85, 2, v7
	v_lshlrev_b32_e32 v86, 2, v17
	v_lshlrev_b32_e32 v87, 2, v19
	v_lshlrev_b32_e32 v88, 2, v20
	v_lshlrev_b32_e32 v89, 2, v21
	v_lshlrev_b32_e32 v90, 2, v22
	v_lshlrev_b32_e32 v91, 2, v18
	v_lshlrev_b32_e32 v92, 2, v23
	v_lshlrev_b32_e32 v93, 2, v24
	v_dual_mov_b32 v1, v10 :: v_dual_lshlrev_b32 v94, 2, v25
	v_dual_mov_b32 v3, v12 :: v_dual_mov_b32 v4, v13
	v_add_nc_u32_e32 v76, 0x1800, v75
	v_or_b32_e32 v77, 0x2000, v75
	v_dual_mov_b32 v5, v14 :: v_dual_add_nc_u32 v78, 0xfffff800, v75
	v_dual_mov_b32 v6, v15 :: v_dual_mov_b32 v7, v16
	s_mov_b32 s6, 0
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
	s_mov_b32 s7, 0
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
	v_dual_add_f32 v103, 0x37fba882, v24 :: v_dual_add_f32 v102, 0x37fba882, v23
	v_dual_add_f32 v101, 0x37fba882, v22 :: v_dual_add_f32 v100, 0x37fba882, v21
	v_dual_add_f32 v99, 0x37fba882, v20 :: v_dual_add_f32 v98, 0x37fba882, v19
	v_dual_add_f32 v97, 0x37fba882, v18 :: v_dual_add_f32 v96, 0x37fba882, v17
	v_dual_add_f32 v111, 0x3827c5ac, v24 :: v_dual_add_f32 v110, 0x3827c5ac, v23
	v_dual_add_f32 v109, 0x3827c5ac, v22 :: v_dual_add_f32 v108, 0x3827c5ac, v21
	v_dual_add_f32 v107, 0x3827c5ac, v20 :: v_dual_add_f32 v106, 0x3827c5ac, v19
	v_dual_add_f32 v105, 0x3827c5ac, v18 :: v_dual_add_f32 v104, 0x3827c5ac, v17
	v_wmma_f32_16x16x16_f16 v[25:32], v[57:60], v[65:68], v[25:32]
	v_wmma_f32_16x16x16_f16 v[33:40], v[57:60], v[65:68], v[33:40]
	v_wmma_f32_16x16x16_f16 v[96:103], v[57:60], v[65:68], v[96:103]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[104:111], v[57:60], v[65:68], v[104:111]
	v_wmma_f32_16x16x16_f16 v[25:32], v[57:60], v[69:72], v[25:32]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[33:40], v[57:60], v[69:72], v[33:40]
	v_wmma_f32_16x16x16_f16 v[96:103], v[57:60], v[69:72], v[96:103]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[104:111], v[57:60], v[69:72], v[104:111]
	v_dual_mul_f32 v49, 0x3727c5ac, v25 :: v_dual_mul_f32 v50, 0x3727c5ac, v26
	v_dual_mul_f32 v51, 0x3727c5ac, v27 :: v_dual_mul_f32 v52, 0x3727c5ac, v28
	v_dual_mul_f32 v53, 0x3727c5ac, v29 :: v_dual_mul_f32 v54, 0x3727c5ac, v30
	v_dual_mul_f32 v55, 0x3727c5ac, v31 :: v_dual_mul_f32 v56, 0x3727c5ac, v32
	v_dual_mul_f32 v41, 0x3727c5ac, v33 :: v_dual_mul_f32 v42, 0x3727c5ac, v34
	v_dual_mul_f32 v43, 0x3727c5ac, v35 :: v_dual_mul_f32 v44, 0x3727c5ac, v36
	v_dual_mul_f32 v45, 0x3727c5ac, v37 :: v_dual_mul_f32 v46, 0x3727c5ac, v38
	v_dual_mul_f32 v47, 0x3727c5ac, v39 :: v_dual_mul_f32 v48, 0x3727c5ac, v40
	v_dual_mul_f32 v33, 0x3727c5ac, v96 :: v_dual_mul_f32 v34, 0x3727c5ac, v97
	v_dual_mul_f32 v35, 0x3727c5ac, v98 :: v_dual_mul_f32 v36, 0x3727c5ac, v99
	v_dual_mul_f32 v37, 0x3727c5ac, v100 :: v_dual_mul_f32 v38, 0x3727c5ac, v101
	v_dual_mul_f32 v39, 0x3727c5ac, v102 :: v_dual_mul_f32 v40, 0x3727c5ac, v103
	v_dual_mul_f32 v25, 0x3727c5ac, v104 :: v_dual_mul_f32 v26, 0x3727c5ac, v105
	v_dual_mul_f32 v27, 0x3727c5ac, v106 :: v_dual_mul_f32 v28, 0x3727c5ac, v107
	v_dual_mul_f32 v29, 0x3727c5ac, v108 :: v_dual_mul_f32 v30, 0x3727c5ac, v109
	v_dual_mul_f32 v31, 0x3727c5ac, v110 :: v_dual_mul_f32 v32, 0x3727c5ac, v111
.LBB8_27:                               ;   Parent Loop BB8_26 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_wait_alu depctr_sa_sdst(0)
	v_cmp_eq_u32_e64 s1, s7, v75
	v_cmp_eq_u32_e64 s2, s7, v77
	v_cmp_eq_u32_e32 vcc_lo, s7, v78
	v_cmp_eq_u32_e64 s0, s7, v76
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v96, 0, v49, s1
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v96, v96, v33, s2
	ds_bpermute_b32 v97, v79, v96
	s_wait_dscnt 0x0
	v_dual_add_f32 v100, v96, v97 :: v_dual_add_nc_u32 v97, s7, v74
	v_lshlrev_b32_e32 v96, 2, v73
	s_addk_co_i32 s7, 0x2000
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s7, 0x4000
	ds_load_2addr_b32 v[98:99], v97 offset1:32
	s_wait_dscnt 0x0
	ds_bpermute_b32 v101, v96, v98
	s_wait_dscnt 0x0
	v_fma_f32 v49, -v100, v101, v49
	ds_bpermute_b32 v101, v80, v98
	s_wait_dscnt 0x0
	v_fma_f32 v50, -v100, v101, v50
	ds_bpermute_b32 v101, v81, v98
	s_wait_dscnt 0x0
	v_fma_f32 v51, -v100, v101, v51
	ds_bpermute_b32 v101, v82, v98
	s_wait_dscnt 0x0
	v_fma_f32 v52, -v100, v101, v52
	ds_bpermute_b32 v101, v83, v98
	s_wait_dscnt 0x0
	v_fma_f32 v53, -v100, v101, v53
	ds_bpermute_b32 v101, v84, v98
	s_wait_dscnt 0x0
	v_fma_f32 v54, -v100, v101, v54
	ds_bpermute_b32 v101, v85, v98
	s_wait_dscnt 0x0
	v_fma_f32 v55, -v100, v101, v55
	ds_bpermute_b32 v101, v86, v98
	s_wait_dscnt 0x0
	v_fma_f32 v56, -v100, v101, v56
	ds_bpermute_b32 v101, v87, v98
	s_wait_dscnt 0x0
	v_fma_f32 v41, -v100, v101, v41
	ds_bpermute_b32 v101, v88, v98
	s_wait_dscnt 0x0
	v_fma_f32 v42, -v100, v101, v42
	ds_bpermute_b32 v101, v89, v98
	s_wait_dscnt 0x0
	v_fma_f32 v43, -v100, v101, v43
	ds_bpermute_b32 v101, v90, v98
	s_wait_dscnt 0x0
	v_fma_f32 v44, -v100, v101, v44
	ds_bpermute_b32 v101, v91, v98
	s_wait_dscnt 0x0
	v_fma_f32 v45, -v100, v101, v45
	ds_bpermute_b32 v101, v92, v98
	s_wait_dscnt 0x0
	v_fma_f32 v46, -v100, v101, v46
	ds_bpermute_b32 v101, v93, v98
	ds_bpermute_b32 v98, v94, v98
	s_wait_dscnt 0x1
	v_fma_f32 v47, -v100, v101, v47
	s_wait_dscnt 0x0
	v_fma_f32 v48, -v100, v98, v48
	ds_bpermute_b32 v98, v96, v99
	s_wait_dscnt 0x0
	v_fma_f32 v33, -v100, v98, v33
	ds_bpermute_b32 v98, v80, v99
	s_wait_dscnt 0x0
	v_fma_f32 v34, -v100, v98, v34
	ds_bpermute_b32 v98, v81, v99
	s_wait_dscnt 0x0
	v_fma_f32 v35, -v100, v98, v35
	ds_bpermute_b32 v98, v82, v99
	s_wait_dscnt 0x0
	v_fma_f32 v36, -v100, v98, v36
	ds_bpermute_b32 v98, v83, v99
	s_wait_dscnt 0x0
	v_fma_f32 v37, -v100, v98, v37
	ds_bpermute_b32 v98, v84, v99
	s_wait_dscnt 0x0
	v_fma_f32 v38, -v100, v98, v38
	ds_bpermute_b32 v98, v85, v99
	s_wait_dscnt 0x0
	v_fma_f32 v39, -v100, v98, v39
	ds_bpermute_b32 v98, v86, v99
	s_wait_dscnt 0x0
	v_fma_f32 v40, -v100, v98, v40
	ds_bpermute_b32 v98, v87, v99
	s_wait_dscnt 0x0
	v_fma_f32 v25, -v100, v98, v25
	ds_bpermute_b32 v98, v88, v99
	s_wait_dscnt 0x0
	v_fma_f32 v26, -v100, v98, v26
	ds_bpermute_b32 v98, v89, v99
	s_wait_dscnt 0x0
	v_fma_f32 v27, -v100, v98, v27
	ds_bpermute_b32 v98, v90, v99
	s_wait_dscnt 0x0
	v_fma_f32 v28, -v100, v98, v28
	ds_bpermute_b32 v98, v91, v99
	s_wait_dscnt 0x0
	v_fma_f32 v29, -v100, v98, v29
	ds_bpermute_b32 v98, v92, v99
	s_wait_dscnt 0x0
	v_fma_f32 v30, -v100, v98, v30
	ds_bpermute_b32 v98, v93, v99
	s_wait_dscnt 0x0
	v_fma_f32 v31, -v100, v98, v31
	ds_bpermute_b32 v98, v94, v99
	s_wait_dscnt 0x0
	v_fma_f32 v32, -v100, v98, v32
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v98, 0, v50, s1
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v98, v98, v34, s2
	ds_bpermute_b32 v99, v79, v98
	s_wait_dscnt 0x0
	v_add_f32_e32 v100, v98, v99
	ds_load_2addr_b32 v[98:99], v97 offset0:64 offset1:96
	s_wait_dscnt 0x0
	ds_bpermute_b32 v101, v96, v98
	s_wait_dscnt 0x0
	v_fma_f32 v49, -v100, v101, v49
	ds_bpermute_b32 v101, v80, v98
	s_wait_dscnt 0x0
	v_fma_f32 v50, -v100, v101, v50
	ds_bpermute_b32 v101, v81, v98
	s_wait_dscnt 0x0
	v_fma_f32 v51, -v100, v101, v51
	ds_bpermute_b32 v101, v82, v98
	s_wait_dscnt 0x0
	v_fma_f32 v52, -v100, v101, v52
	ds_bpermute_b32 v101, v83, v98
	s_wait_dscnt 0x0
	v_fma_f32 v53, -v100, v101, v53
	ds_bpermute_b32 v101, v84, v98
	s_wait_dscnt 0x0
	v_fma_f32 v54, -v100, v101, v54
	ds_bpermute_b32 v101, v85, v98
	s_wait_dscnt 0x0
	v_fma_f32 v55, -v100, v101, v55
	ds_bpermute_b32 v101, v86, v98
	s_wait_dscnt 0x0
	v_fma_f32 v56, -v100, v101, v56
	ds_bpermute_b32 v101, v87, v98
	s_wait_dscnt 0x0
	v_fma_f32 v41, -v100, v101, v41
	ds_bpermute_b32 v101, v88, v98
	s_wait_dscnt 0x0
	v_fma_f32 v42, -v100, v101, v42
	ds_bpermute_b32 v101, v89, v98
	s_wait_dscnt 0x0
	v_fma_f32 v43, -v100, v101, v43
	ds_bpermute_b32 v101, v90, v98
	s_wait_dscnt 0x0
	v_fma_f32 v44, -v100, v101, v44
	ds_bpermute_b32 v101, v91, v98
	s_wait_dscnt 0x0
	v_fma_f32 v45, -v100, v101, v45
	ds_bpermute_b32 v101, v92, v98
	s_wait_dscnt 0x0
	v_fma_f32 v46, -v100, v101, v46
	ds_bpermute_b32 v101, v93, v98
	ds_bpermute_b32 v98, v94, v98
	s_wait_dscnt 0x1
	v_fma_f32 v47, -v100, v101, v47
	s_wait_dscnt 0x0
	v_fma_f32 v48, -v100, v98, v48
	ds_bpermute_b32 v98, v96, v99
	s_wait_dscnt 0x0
	v_fma_f32 v33, -v100, v98, v33
	ds_bpermute_b32 v98, v80, v99
	s_wait_dscnt 0x0
	v_fma_f32 v34, -v100, v98, v34
	ds_bpermute_b32 v98, v81, v99
	s_wait_dscnt 0x0
	v_fma_f32 v35, -v100, v98, v35
	ds_bpermute_b32 v98, v82, v99
	s_wait_dscnt 0x0
	v_fma_f32 v36, -v100, v98, v36
	ds_bpermute_b32 v98, v83, v99
	s_wait_dscnt 0x0
	v_fma_f32 v37, -v100, v98, v37
	ds_bpermute_b32 v98, v84, v99
	s_wait_dscnt 0x0
	v_fma_f32 v38, -v100, v98, v38
	ds_bpermute_b32 v98, v85, v99
	s_wait_dscnt 0x0
	v_fma_f32 v39, -v100, v98, v39
	ds_bpermute_b32 v98, v86, v99
	s_wait_dscnt 0x0
	v_fma_f32 v40, -v100, v98, v40
	ds_bpermute_b32 v98, v87, v99
	s_wait_dscnt 0x0
	v_fma_f32 v25, -v100, v98, v25
	ds_bpermute_b32 v98, v88, v99
	s_wait_dscnt 0x0
	v_fma_f32 v26, -v100, v98, v26
	ds_bpermute_b32 v98, v89, v99
	s_wait_dscnt 0x0
	v_fma_f32 v27, -v100, v98, v27
	ds_bpermute_b32 v98, v90, v99
	s_wait_dscnt 0x0
	v_fma_f32 v28, -v100, v98, v28
	ds_bpermute_b32 v98, v91, v99
	s_wait_dscnt 0x0
	v_fma_f32 v29, -v100, v98, v29
	ds_bpermute_b32 v98, v92, v99
	s_wait_dscnt 0x0
	v_fma_f32 v30, -v100, v98, v30
	ds_bpermute_b32 v98, v93, v99
	s_wait_dscnt 0x0
	v_fma_f32 v31, -v100, v98, v31
	ds_bpermute_b32 v98, v94, v99
	s_wait_dscnt 0x0
	v_fma_f32 v32, -v100, v98, v32
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v98, 0, v51, s1
	ds_load_2addr_b32 v[112:113], v97 offset0:128 offset1:160
	v_cndmask_b32_e64 v98, v98, v35, s2
	ds_bpermute_b32 v99, v79, v98
	s_wait_dscnt 0x0
	v_add_f32_e32 v114, v98, v99
	ds_bpermute_b32 v98, v96, v112
	s_wait_dscnt 0x0
	v_fma_f32 v49, -v114, v98, v49
	ds_bpermute_b32 v98, v80, v112
	s_wait_dscnt 0x0
	v_fma_f32 v50, -v114, v98, v50
	ds_bpermute_b32 v98, v81, v112
	s_wait_dscnt 0x0
	v_fma_f32 v51, -v114, v98, v51
	ds_bpermute_b32 v98, v82, v112
	s_wait_dscnt 0x0
	v_fma_f32 v52, -v114, v98, v52
	ds_bpermute_b32 v98, v83, v112
	s_wait_dscnt 0x0
	v_fma_f32 v53, -v114, v98, v53
	ds_bpermute_b32 v98, v84, v112
	s_wait_dscnt 0x0
	v_fma_f32 v54, -v114, v98, v54
	ds_bpermute_b32 v98, v85, v112
	s_wait_dscnt 0x0
	v_fma_f32 v55, -v114, v98, v55
	ds_bpermute_b32 v98, v86, v112
	s_wait_dscnt 0x0
	v_fma_f32 v56, -v114, v98, v56
	ds_bpermute_b32 v98, v87, v112
	s_wait_dscnt 0x0
	v_fma_f32 v41, -v114, v98, v41
	ds_bpermute_b32 v98, v88, v112
	s_wait_dscnt 0x0
	v_fma_f32 v42, -v114, v98, v42
	ds_bpermute_b32 v98, v89, v112
	s_wait_dscnt 0x0
	v_fma_f32 v43, -v114, v98, v43
	ds_bpermute_b32 v98, v90, v112
	s_wait_dscnt 0x0
	v_fma_f32 v44, -v114, v98, v44
	ds_bpermute_b32 v98, v91, v112
	s_wait_dscnt 0x0
	v_fma_f32 v45, -v114, v98, v45
	ds_bpermute_b32 v98, v92, v112
	s_wait_dscnt 0x0
	v_fma_f32 v46, -v114, v98, v46
	ds_bpermute_b32 v98, v93, v112
	s_wait_dscnt 0x0
	v_fma_f32 v47, -v114, v98, v47
	ds_bpermute_b32 v98, v94, v112
	s_wait_dscnt 0x0
	v_fma_f32 v48, -v114, v98, v48
	ds_bpermute_b32 v98, v96, v113
	s_wait_dscnt 0x0
	v_fma_f32 v98, -v114, v98, v33
	ds_bpermute_b32 v33, v80, v113
	s_wait_dscnt 0x0
	v_fma_f32 v99, -v114, v33, v34
	ds_bpermute_b32 v33, v81, v113
	s_wait_dscnt 0x0
	v_fma_f32 v100, -v114, v33, v35
	ds_bpermute_b32 v33, v82, v113
	s_wait_dscnt 0x0
	v_fma_f32 v101, -v114, v33, v36
	ds_bpermute_b32 v33, v83, v113
	s_wait_dscnt 0x0
	v_fma_f32 v102, -v114, v33, v37
	ds_bpermute_b32 v33, v84, v113
	s_wait_dscnt 0x0
	v_fma_f32 v103, -v114, v33, v38
	ds_bpermute_b32 v33, v85, v113
	s_wait_dscnt 0x0
	v_fma_f32 v104, -v114, v33, v39
	ds_bpermute_b32 v33, v86, v113
	s_wait_dscnt 0x0
	v_fma_f32 v105, -v114, v33, v40
	ds_bpermute_b32 v33, v87, v113
	s_wait_dscnt 0x0
	v_fma_f32 v106, -v114, v33, v25
	ds_bpermute_b32 v25, v88, v113
	s_wait_dscnt 0x0
	v_fma_f32 v107, -v114, v25, v26
	ds_bpermute_b32 v25, v89, v113
	s_wait_dscnt 0x0
	v_fma_f32 v108, -v114, v25, v27
	ds_bpermute_b32 v25, v90, v113
	s_wait_dscnt 0x0
	v_fma_f32 v109, -v114, v25, v28
	ds_bpermute_b32 v25, v91, v113
	s_wait_dscnt 0x0
	v_fma_f32 v110, -v114, v25, v29
	ds_bpermute_b32 v25, v92, v113
	s_wait_dscnt 0x0
	v_fma_f32 v111, -v114, v25, v30
	ds_bpermute_b32 v25, v93, v113
	s_wait_dscnt 0x0
	v_fma_f32 v112, -v114, v25, v31
	ds_bpermute_b32 v25, v94, v113
	s_wait_dscnt 0x0
	v_fma_f32 v113, -v114, v25, v32
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v25, 0, v52, s1
	ds_load_2addr_b32 v[114:115], v97 offset0:192 offset1:224
	v_cndmask_b32_e64 v25, v25, v101, s2
	ds_bpermute_b32 v26, v79, v25
	s_wait_dscnt 0x1
	ds_bpermute_b32 v27, v81, v114
	ds_bpermute_b32 v28, v82, v114
	ds_bpermute_b32 v29, v83, v114
	ds_bpermute_b32 v30, v84, v114
	ds_bpermute_b32 v31, v85, v114
	ds_bpermute_b32 v32, v86, v114
	ds_bpermute_b32 v33, v87, v114
	ds_bpermute_b32 v34, v88, v114
	ds_bpermute_b32 v35, v89, v114
	s_wait_dscnt 0x9
	v_add_f32_e32 v116, v25, v26
	ds_bpermute_b32 v25, v96, v114
	ds_bpermute_b32 v26, v80, v114
	ds_bpermute_b32 v36, v90, v114
	ds_bpermute_b32 v37, v91, v114
	ds_bpermute_b32 v38, v92, v114
	ds_bpermute_b32 v39, v93, v114
	ds_bpermute_b32 v40, v94, v114
	v_add_nc_u32_e32 v114, 0x1800, v97
	s_wait_dscnt 0xf
	v_fma_f32 v27, -v116, v27, v51
	s_wait_dscnt 0xe
	v_fma_f32 v28, -v116, v28, v52
	s_wait_dscnt 0xd
	v_fma_f32 v29, -v116, v29, v53
	s_wait_dscnt 0xc
	v_fma_f32 v30, -v116, v30, v54
	s_wait_dscnt 0xb
	v_fma_f32 v31, -v116, v31, v55
	s_wait_dscnt 0xa
	v_fma_f32 v32, -v116, v32, v56
	s_wait_dscnt 0x9
	v_fma_f32 v33, -v116, v33, v41
	s_wait_dscnt 0x8
	v_fma_f32 v34, -v116, v34, v42
	s_wait_dscnt 0x7
	v_fma_f32 v35, -v116, v35, v43
	ds_bpermute_b32 v41, v96, v115
	s_wait_dscnt 0x7
	v_fma_f32 v25, -v116, v25, v49
	s_wait_dscnt 0x6
	v_fma_f32 v26, -v116, v26, v50
	s_wait_dscnt 0x5
	v_fma_f32 v36, -v116, v36, v44
	s_wait_dscnt 0x4
	v_fma_f32 v37, -v116, v37, v45
	s_wait_dscnt 0x3
	v_fma_f32 v38, -v116, v38, v46
	s_wait_dscnt 0x2
	v_fma_f32 v39, -v116, v39, v47
	s_wait_dscnt 0x1
	v_fma_f32 v40, -v116, v40, v48
	ds_bpermute_b32 v42, v80, v115
	ds_bpermute_b32 v43, v81, v115
	ds_bpermute_b32 v44, v82, v115
	ds_bpermute_b32 v45, v83, v115
	ds_bpermute_b32 v46, v84, v115
	ds_bpermute_b32 v47, v85, v115
	ds_bpermute_b32 v48, v86, v115
	ds_bpermute_b32 v49, v87, v115
	ds_bpermute_b32 v50, v88, v115
	ds_bpermute_b32 v51, v89, v115
	ds_bpermute_b32 v52, v90, v115
	ds_bpermute_b32 v53, v91, v115
	ds_bpermute_b32 v54, v92, v115
	ds_bpermute_b32 v55, v93, v115
	ds_bpermute_b32 v56, v94, v115
	s_wait_dscnt 0xf
	v_fma_f32 v41, -v116, v41, v98
	s_wait_dscnt 0xe
	v_fma_f32 v42, -v116, v42, v99
	s_wait_dscnt 0xd
	v_fma_f32 v43, -v116, v43, v100
	s_wait_dscnt 0xc
	v_fma_f32 v44, -v116, v44, v101
	s_wait_dscnt 0xb
	v_fma_f32 v45, -v116, v45, v102
	s_wait_dscnt 0xa
	v_fma_f32 v46, -v116, v46, v103
	s_wait_dscnt 0x9
	v_fma_f32 v47, -v116, v47, v104
	s_wait_dscnt 0x8
	v_fma_f32 v48, -v116, v48, v105
	s_wait_dscnt 0x7
	v_fma_f32 v49, -v116, v49, v106
	s_wait_dscnt 0x6
	v_fma_f32 v50, -v116, v50, v107
	s_wait_dscnt 0x5
	v_fma_f32 v51, -v116, v51, v108
	s_wait_dscnt 0x4
	v_fma_f32 v52, -v116, v52, v109
	s_wait_dscnt 0x3
	v_fma_f32 v53, -v116, v53, v110
	s_wait_dscnt 0x2
	v_fma_f32 v54, -v116, v54, v111
	s_wait_dscnt 0x1
	v_fma_f32 v55, -v116, v55, v112
	s_wait_dscnt 0x0
	v_fma_f32 v56, -v116, v56, v113
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v98, 0, v29, s1
	v_add_nc_u32_e32 v101, 0x400, v97
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e64 v98, v98, v45, s2
	ds_bpermute_b32 v99, v79, v98
	s_wait_dscnt 0x0
	v_add_f32_e32 v100, v98, v99
	ds_load_2addr_b32 v[98:99], v101 offset1:32
	s_wait_dscnt 0x0
	ds_bpermute_b32 v102, v96, v98
	s_wait_dscnt 0x0
	v_fma_f32 v25, -v100, v102, v25
	ds_bpermute_b32 v102, v80, v98
	s_wait_dscnt 0x0
	v_fma_f32 v26, -v100, v102, v26
	ds_bpermute_b32 v102, v81, v98
	s_wait_dscnt 0x0
	v_fma_f32 v27, -v100, v102, v27
	ds_bpermute_b32 v102, v82, v98
	s_wait_dscnt 0x0
	v_fma_f32 v28, -v100, v102, v28
	ds_bpermute_b32 v102, v83, v98
	s_wait_dscnt 0x0
	v_fma_f32 v29, -v100, v102, v29
	ds_bpermute_b32 v102, v84, v98
	s_wait_dscnt 0x0
	v_fma_f32 v30, -v100, v102, v30
	ds_bpermute_b32 v102, v85, v98
	s_wait_dscnt 0x0
	v_fma_f32 v31, -v100, v102, v31
	ds_bpermute_b32 v102, v86, v98
	s_wait_dscnt 0x0
	v_fma_f32 v32, -v100, v102, v32
	ds_bpermute_b32 v102, v87, v98
	s_wait_dscnt 0x0
	v_fma_f32 v33, -v100, v102, v33
	ds_bpermute_b32 v102, v88, v98
	s_wait_dscnt 0x0
	v_fma_f32 v34, -v100, v102, v34
	ds_bpermute_b32 v102, v89, v98
	s_wait_dscnt 0x0
	v_fma_f32 v35, -v100, v102, v35
	ds_bpermute_b32 v102, v90, v98
	s_wait_dscnt 0x0
	v_fma_f32 v36, -v100, v102, v36
	ds_bpermute_b32 v102, v91, v98
	s_wait_dscnt 0x0
	v_fma_f32 v37, -v100, v102, v37
	ds_bpermute_b32 v102, v92, v98
	s_wait_dscnt 0x0
	v_fma_f32 v38, -v100, v102, v38
	ds_bpermute_b32 v102, v93, v98
	ds_bpermute_b32 v98, v94, v98
	s_wait_dscnt 0x1
	v_fma_f32 v39, -v100, v102, v39
	s_wait_dscnt 0x0
	v_fma_f32 v40, -v100, v98, v40
	ds_bpermute_b32 v98, v96, v99
	s_wait_dscnt 0x0
	v_fma_f32 v41, -v100, v98, v41
	ds_bpermute_b32 v98, v80, v99
	s_wait_dscnt 0x0
	v_fma_f32 v42, -v100, v98, v42
	ds_bpermute_b32 v98, v81, v99
	s_wait_dscnt 0x0
	v_fma_f32 v43, -v100, v98, v43
	ds_bpermute_b32 v98, v82, v99
	s_wait_dscnt 0x0
	v_fma_f32 v44, -v100, v98, v44
	ds_bpermute_b32 v98, v83, v99
	s_wait_dscnt 0x0
	v_fma_f32 v45, -v100, v98, v45
	ds_bpermute_b32 v98, v84, v99
	s_wait_dscnt 0x0
	v_fma_f32 v46, -v100, v98, v46
	ds_bpermute_b32 v98, v85, v99
	s_wait_dscnt 0x0
	v_fma_f32 v47, -v100, v98, v47
	ds_bpermute_b32 v98, v86, v99
	s_wait_dscnt 0x0
	v_fma_f32 v48, -v100, v98, v48
	ds_bpermute_b32 v98, v87, v99
	s_wait_dscnt 0x0
	v_fma_f32 v49, -v100, v98, v49
	ds_bpermute_b32 v98, v88, v99
	s_wait_dscnt 0x0
	v_fma_f32 v50, -v100, v98, v50
	ds_bpermute_b32 v98, v89, v99
	s_wait_dscnt 0x0
	v_fma_f32 v51, -v100, v98, v51
	ds_bpermute_b32 v98, v90, v99
	s_wait_dscnt 0x0
	v_fma_f32 v52, -v100, v98, v52
	ds_bpermute_b32 v98, v91, v99
	s_wait_dscnt 0x0
	v_fma_f32 v53, -v100, v98, v53
	ds_bpermute_b32 v98, v92, v99
	s_wait_dscnt 0x0
	v_fma_f32 v54, -v100, v98, v54
	ds_bpermute_b32 v98, v93, v99
	s_wait_dscnt 0x0
	v_fma_f32 v55, -v100, v98, v55
	ds_bpermute_b32 v98, v94, v99
	s_wait_dscnt 0x0
	v_fma_f32 v56, -v100, v98, v56
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v98, 0, v30, s1
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v98, v98, v46, s2
	ds_bpermute_b32 v99, v79, v98
	s_wait_dscnt 0x0
	v_add_f32_e32 v100, v98, v99
	ds_load_2addr_b32 v[98:99], v101 offset0:64 offset1:96
	s_wait_dscnt 0x0
	ds_bpermute_b32 v102, v96, v98
	s_wait_dscnt 0x0
	v_fma_f32 v25, -v100, v102, v25
	ds_bpermute_b32 v102, v80, v98
	s_wait_dscnt 0x0
	v_fma_f32 v26, -v100, v102, v26
	ds_bpermute_b32 v102, v81, v98
	s_wait_dscnt 0x0
	v_fma_f32 v27, -v100, v102, v27
	ds_bpermute_b32 v102, v82, v98
	s_wait_dscnt 0x0
	v_fma_f32 v28, -v100, v102, v28
	ds_bpermute_b32 v102, v83, v98
	s_wait_dscnt 0x0
	v_fma_f32 v29, -v100, v102, v29
	ds_bpermute_b32 v102, v84, v98
	s_wait_dscnt 0x0
	v_fma_f32 v30, -v100, v102, v30
	ds_bpermute_b32 v102, v85, v98
	s_wait_dscnt 0x0
	v_fma_f32 v31, -v100, v102, v31
	ds_bpermute_b32 v102, v86, v98
	s_wait_dscnt 0x0
	v_fma_f32 v32, -v100, v102, v32
	ds_bpermute_b32 v102, v87, v98
	s_wait_dscnt 0x0
	v_fma_f32 v33, -v100, v102, v33
	ds_bpermute_b32 v102, v88, v98
	s_wait_dscnt 0x0
	v_fma_f32 v34, -v100, v102, v34
	ds_bpermute_b32 v102, v89, v98
	s_wait_dscnt 0x0
	v_fma_f32 v35, -v100, v102, v35
	ds_bpermute_b32 v102, v90, v98
	s_wait_dscnt 0x0
	v_fma_f32 v36, -v100, v102, v36
	ds_bpermute_b32 v102, v91, v98
	s_wait_dscnt 0x0
	v_fma_f32 v37, -v100, v102, v37
	ds_bpermute_b32 v102, v92, v98
	s_wait_dscnt 0x0
	v_fma_f32 v38, -v100, v102, v38
	ds_bpermute_b32 v102, v93, v98
	ds_bpermute_b32 v98, v94, v98
	s_wait_dscnt 0x1
	v_fma_f32 v39, -v100, v102, v39
	s_wait_dscnt 0x0
	v_fma_f32 v40, -v100, v98, v40
	ds_bpermute_b32 v98, v96, v99
	s_wait_dscnt 0x0
	v_fma_f32 v41, -v100, v98, v41
	ds_bpermute_b32 v98, v80, v99
	s_wait_dscnt 0x0
	v_fma_f32 v42, -v100, v98, v42
	ds_bpermute_b32 v98, v81, v99
	s_wait_dscnt 0x0
	v_fma_f32 v43, -v100, v98, v43
	ds_bpermute_b32 v98, v82, v99
	s_wait_dscnt 0x0
	v_fma_f32 v44, -v100, v98, v44
	ds_bpermute_b32 v98, v83, v99
	s_wait_dscnt 0x0
	v_fma_f32 v45, -v100, v98, v45
	ds_bpermute_b32 v98, v84, v99
	s_wait_dscnt 0x0
	v_fma_f32 v46, -v100, v98, v46
	ds_bpermute_b32 v98, v85, v99
	s_wait_dscnt 0x0
	v_fma_f32 v47, -v100, v98, v47
	ds_bpermute_b32 v98, v86, v99
	s_wait_dscnt 0x0
	v_fma_f32 v48, -v100, v98, v48
	ds_bpermute_b32 v98, v87, v99
	s_wait_dscnt 0x0
	v_fma_f32 v49, -v100, v98, v49
	ds_bpermute_b32 v98, v88, v99
	s_wait_dscnt 0x0
	v_fma_f32 v50, -v100, v98, v50
	ds_bpermute_b32 v98, v89, v99
	s_wait_dscnt 0x0
	v_fma_f32 v51, -v100, v98, v51
	ds_bpermute_b32 v98, v90, v99
	s_wait_dscnt 0x0
	v_fma_f32 v52, -v100, v98, v52
	ds_bpermute_b32 v98, v91, v99
	s_wait_dscnt 0x0
	v_fma_f32 v53, -v100, v98, v53
	ds_bpermute_b32 v98, v92, v99
	s_wait_dscnt 0x0
	v_fma_f32 v54, -v100, v98, v54
	ds_bpermute_b32 v98, v93, v99
	s_wait_dscnt 0x0
	v_fma_f32 v55, -v100, v98, v55
	ds_bpermute_b32 v98, v94, v99
	s_wait_dscnt 0x0
	v_fma_f32 v56, -v100, v98, v56
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v98, 0, v31, s1
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v98, v98, v47, s2
	ds_bpermute_b32 v99, v79, v98
	s_wait_dscnt 0x0
	v_add_f32_e32 v100, v98, v99
	ds_load_2addr_b32 v[98:99], v101 offset0:128 offset1:160
	s_wait_dscnt 0x0
	ds_bpermute_b32 v102, v96, v98
	s_wait_dscnt 0x0
	v_fma_f32 v25, -v100, v102, v25
	ds_bpermute_b32 v102, v80, v98
	s_wait_dscnt 0x0
	v_fma_f32 v26, -v100, v102, v26
	ds_bpermute_b32 v102, v81, v98
	s_wait_dscnt 0x0
	v_fma_f32 v27, -v100, v102, v27
	ds_bpermute_b32 v102, v82, v98
	s_wait_dscnt 0x0
	v_fma_f32 v28, -v100, v102, v28
	ds_bpermute_b32 v102, v83, v98
	s_wait_dscnt 0x0
	v_fma_f32 v29, -v100, v102, v29
	ds_bpermute_b32 v102, v84, v98
	s_wait_dscnt 0x0
	v_fma_f32 v30, -v100, v102, v30
	ds_bpermute_b32 v102, v85, v98
	s_wait_dscnt 0x0
	v_fma_f32 v31, -v100, v102, v31
	ds_bpermute_b32 v102, v86, v98
	s_wait_dscnt 0x0
	v_fma_f32 v32, -v100, v102, v32
	ds_bpermute_b32 v102, v87, v98
	s_wait_dscnt 0x0
	v_fma_f32 v33, -v100, v102, v33
	ds_bpermute_b32 v102, v88, v98
	s_wait_dscnt 0x0
	v_fma_f32 v34, -v100, v102, v34
	ds_bpermute_b32 v102, v89, v98
	s_wait_dscnt 0x0
	v_fma_f32 v35, -v100, v102, v35
	ds_bpermute_b32 v102, v90, v98
	s_wait_dscnt 0x0
	v_fma_f32 v36, -v100, v102, v36
	ds_bpermute_b32 v102, v91, v98
	s_wait_dscnt 0x0
	v_fma_f32 v37, -v100, v102, v37
	ds_bpermute_b32 v102, v92, v98
	s_wait_dscnt 0x0
	v_fma_f32 v38, -v100, v102, v38
	ds_bpermute_b32 v102, v93, v98
	ds_bpermute_b32 v98, v94, v98
	s_wait_dscnt 0x1
	v_fma_f32 v39, -v100, v102, v39
	s_wait_dscnt 0x0
	v_fma_f32 v40, -v100, v98, v40
	ds_bpermute_b32 v98, v96, v99
	s_wait_dscnt 0x0
	v_fma_f32 v41, -v100, v98, v41
	ds_bpermute_b32 v98, v80, v99
	s_wait_dscnt 0x0
	v_fma_f32 v42, -v100, v98, v42
	ds_bpermute_b32 v98, v81, v99
	s_wait_dscnt 0x0
	v_fma_f32 v43, -v100, v98, v43
	ds_bpermute_b32 v98, v82, v99
	s_wait_dscnt 0x0
	v_fma_f32 v44, -v100, v98, v44
	ds_bpermute_b32 v98, v83, v99
	s_wait_dscnt 0x0
	v_fma_f32 v45, -v100, v98, v45
	ds_bpermute_b32 v98, v84, v99
	s_wait_dscnt 0x0
	v_fma_f32 v46, -v100, v98, v46
	ds_bpermute_b32 v98, v85, v99
	s_wait_dscnt 0x0
	v_fma_f32 v47, -v100, v98, v47
	ds_bpermute_b32 v98, v86, v99
	s_wait_dscnt 0x0
	v_fma_f32 v48, -v100, v98, v48
	ds_bpermute_b32 v98, v87, v99
	s_wait_dscnt 0x0
	v_fma_f32 v49, -v100, v98, v49
	ds_bpermute_b32 v98, v88, v99
	s_wait_dscnt 0x0
	v_fma_f32 v50, -v100, v98, v50
	ds_bpermute_b32 v98, v89, v99
	s_wait_dscnt 0x0
	v_fma_f32 v51, -v100, v98, v51
	ds_bpermute_b32 v98, v90, v99
	s_wait_dscnt 0x0
	v_fma_f32 v52, -v100, v98, v52
	ds_bpermute_b32 v98, v91, v99
	s_wait_dscnt 0x0
	v_fma_f32 v53, -v100, v98, v53
	ds_bpermute_b32 v98, v92, v99
	s_wait_dscnt 0x0
	v_fma_f32 v54, -v100, v98, v54
	ds_bpermute_b32 v98, v93, v99
	s_wait_dscnt 0x0
	v_fma_f32 v55, -v100, v98, v55
	ds_bpermute_b32 v98, v94, v99
	s_wait_dscnt 0x0
	v_fma_f32 v56, -v100, v98, v56
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v98, 0, v32, s1
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v98, v98, v48, s2
	ds_bpermute_b32 v99, v79, v98
	s_wait_dscnt 0x0
	v_add_f32_e32 v100, v98, v99
	ds_load_2addr_b32 v[98:99], v101 offset0:192 offset1:224
	s_wait_dscnt 0x0
	ds_bpermute_b32 v101, v96, v98
	s_wait_dscnt 0x0
	v_fma_f32 v25, -v100, v101, v25
	ds_bpermute_b32 v101, v80, v98
	s_wait_dscnt 0x0
	v_fma_f32 v26, -v100, v101, v26
	ds_bpermute_b32 v101, v81, v98
	s_wait_dscnt 0x0
	v_fma_f32 v27, -v100, v101, v27
	ds_bpermute_b32 v101, v82, v98
	s_wait_dscnt 0x0
	v_fma_f32 v28, -v100, v101, v28
	ds_bpermute_b32 v101, v83, v98
	s_wait_dscnt 0x0
	v_fma_f32 v29, -v100, v101, v29
	ds_bpermute_b32 v101, v84, v98
	s_wait_dscnt 0x0
	v_fma_f32 v30, -v100, v101, v30
	ds_bpermute_b32 v101, v85, v98
	s_wait_dscnt 0x0
	v_fma_f32 v31, -v100, v101, v31
	ds_bpermute_b32 v101, v86, v98
	s_wait_dscnt 0x0
	v_fma_f32 v32, -v100, v101, v32
	ds_bpermute_b32 v101, v87, v98
	s_wait_dscnt 0x0
	v_fma_f32 v33, -v100, v101, v33
	ds_bpermute_b32 v101, v88, v98
	s_wait_dscnt 0x0
	v_fma_f32 v34, -v100, v101, v34
	ds_bpermute_b32 v101, v89, v98
	s_wait_dscnt 0x0
	v_fma_f32 v35, -v100, v101, v35
	ds_bpermute_b32 v101, v90, v98
	s_wait_dscnt 0x0
	v_fma_f32 v36, -v100, v101, v36
	ds_bpermute_b32 v101, v91, v98
	s_wait_dscnt 0x0
	v_fma_f32 v37, -v100, v101, v37
	ds_bpermute_b32 v101, v92, v98
	s_wait_dscnt 0x0
	v_fma_f32 v38, -v100, v101, v38
	ds_bpermute_b32 v101, v93, v98
	ds_bpermute_b32 v98, v94, v98
	s_wait_dscnt 0x1
	v_fma_f32 v39, -v100, v101, v39
	s_wait_dscnt 0x0
	v_fma_f32 v40, -v100, v98, v40
	ds_bpermute_b32 v98, v96, v99
	v_add_nc_u32_e32 v101, 0x800, v97
	s_wait_dscnt 0x0
	v_fma_f32 v41, -v100, v98, v41
	ds_bpermute_b32 v98, v80, v99
	s_wait_dscnt 0x0
	v_fma_f32 v42, -v100, v98, v42
	ds_bpermute_b32 v98, v81, v99
	s_wait_dscnt 0x0
	v_fma_f32 v43, -v100, v98, v43
	ds_bpermute_b32 v98, v82, v99
	s_wait_dscnt 0x0
	v_fma_f32 v44, -v100, v98, v44
	ds_bpermute_b32 v98, v83, v99
	s_wait_dscnt 0x0
	v_fma_f32 v45, -v100, v98, v45
	ds_bpermute_b32 v98, v84, v99
	s_wait_dscnt 0x0
	v_fma_f32 v46, -v100, v98, v46
	ds_bpermute_b32 v98, v85, v99
	s_wait_dscnt 0x0
	v_fma_f32 v47, -v100, v98, v47
	ds_bpermute_b32 v98, v86, v99
	s_wait_dscnt 0x0
	v_fma_f32 v48, -v100, v98, v48
	ds_bpermute_b32 v98, v87, v99
	s_wait_dscnt 0x0
	v_fma_f32 v49, -v100, v98, v49
	ds_bpermute_b32 v98, v88, v99
	s_wait_dscnt 0x0
	v_fma_f32 v50, -v100, v98, v50
	ds_bpermute_b32 v98, v89, v99
	s_wait_dscnt 0x0
	v_fma_f32 v51, -v100, v98, v51
	ds_bpermute_b32 v98, v90, v99
	s_wait_dscnt 0x0
	v_fma_f32 v52, -v100, v98, v52
	ds_bpermute_b32 v98, v91, v99
	s_wait_dscnt 0x0
	v_fma_f32 v53, -v100, v98, v53
	ds_bpermute_b32 v98, v92, v99
	s_wait_dscnt 0x0
	v_fma_f32 v54, -v100, v98, v54
	ds_bpermute_b32 v98, v93, v99
	s_wait_dscnt 0x0
	v_fma_f32 v55, -v100, v98, v55
	ds_bpermute_b32 v98, v94, v99
	s_wait_dscnt 0x0
	v_fma_f32 v56, -v100, v98, v56
	;;#ASMSTART
	;;#ASMEND
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v98, 0, v25, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v98, v98, v41, s0
	ds_bpermute_b32 v99, v79, v98
	s_wait_dscnt 0x0
	v_add_f32_e32 v100, v98, v99
	ds_load_2addr_b32 v[98:99], v101 offset1:32
	s_wait_dscnt 0x0
	ds_bpermute_b32 v102, v96, v98
	s_wait_dscnt 0x0
	v_fma_f32 v25, -v100, v102, v25
	ds_bpermute_b32 v102, v80, v98
	s_wait_dscnt 0x0
	v_fma_f32 v26, -v100, v102, v26
	ds_bpermute_b32 v102, v81, v98
	s_wait_dscnt 0x0
	v_fma_f32 v27, -v100, v102, v27
	ds_bpermute_b32 v102, v82, v98
	s_wait_dscnt 0x0
	v_fma_f32 v28, -v100, v102, v28
	ds_bpermute_b32 v102, v83, v98
	s_wait_dscnt 0x0
	v_fma_f32 v29, -v100, v102, v29
	ds_bpermute_b32 v102, v84, v98
	s_wait_dscnt 0x0
	v_fma_f32 v30, -v100, v102, v30
	ds_bpermute_b32 v102, v85, v98
	s_wait_dscnt 0x0
	v_fma_f32 v31, -v100, v102, v31
	ds_bpermute_b32 v102, v86, v98
	s_wait_dscnt 0x0
	v_fma_f32 v32, -v100, v102, v32
	ds_bpermute_b32 v102, v87, v98
	s_wait_dscnt 0x0
	v_fma_f32 v33, -v100, v102, v33
	ds_bpermute_b32 v102, v88, v98
	s_wait_dscnt 0x0
	v_fma_f32 v34, -v100, v102, v34
	ds_bpermute_b32 v102, v89, v98
	s_wait_dscnt 0x0
	v_fma_f32 v35, -v100, v102, v35
	ds_bpermute_b32 v102, v90, v98
	s_wait_dscnt 0x0
	v_fma_f32 v36, -v100, v102, v36
	ds_bpermute_b32 v102, v91, v98
	s_wait_dscnt 0x0
	v_fma_f32 v37, -v100, v102, v37
	ds_bpermute_b32 v102, v92, v98
	s_wait_dscnt 0x0
	v_fma_f32 v38, -v100, v102, v38
	ds_bpermute_b32 v102, v93, v98
	ds_bpermute_b32 v98, v94, v98
	s_wait_dscnt 0x1
	v_fma_f32 v39, -v100, v102, v39
	s_wait_dscnt 0x0
	v_fma_f32 v40, -v100, v98, v40
	ds_bpermute_b32 v98, v96, v99
	s_wait_dscnt 0x0
	v_fma_f32 v41, -v100, v98, v41
	ds_bpermute_b32 v98, v80, v99
	s_wait_dscnt 0x0
	v_fma_f32 v42, -v100, v98, v42
	ds_bpermute_b32 v98, v81, v99
	s_wait_dscnt 0x0
	v_fma_f32 v43, -v100, v98, v43
	ds_bpermute_b32 v98, v82, v99
	s_wait_dscnt 0x0
	v_fma_f32 v44, -v100, v98, v44
	ds_bpermute_b32 v98, v83, v99
	s_wait_dscnt 0x0
	v_fma_f32 v45, -v100, v98, v45
	ds_bpermute_b32 v98, v84, v99
	s_wait_dscnt 0x0
	v_fma_f32 v46, -v100, v98, v46
	ds_bpermute_b32 v98, v85, v99
	s_wait_dscnt 0x0
	v_fma_f32 v47, -v100, v98, v47
	ds_bpermute_b32 v98, v86, v99
	s_wait_dscnt 0x0
	v_fma_f32 v48, -v100, v98, v48
	ds_bpermute_b32 v98, v87, v99
	s_wait_dscnt 0x0
	v_fma_f32 v49, -v100, v98, v49
	ds_bpermute_b32 v98, v88, v99
	s_wait_dscnt 0x0
	v_fma_f32 v50, -v100, v98, v50
	ds_bpermute_b32 v98, v89, v99
	s_wait_dscnt 0x0
	v_fma_f32 v51, -v100, v98, v51
	ds_bpermute_b32 v98, v90, v99
	s_wait_dscnt 0x0
	v_fma_f32 v52, -v100, v98, v52
	ds_bpermute_b32 v98, v91, v99
	s_wait_dscnt 0x0
	v_fma_f32 v53, -v100, v98, v53
	ds_bpermute_b32 v98, v92, v99
	s_wait_dscnt 0x0
	v_fma_f32 v54, -v100, v98, v54
	ds_bpermute_b32 v98, v93, v99
	s_wait_dscnt 0x0
	v_fma_f32 v55, -v100, v98, v55
	ds_bpermute_b32 v98, v94, v99
	s_wait_dscnt 0x0
	v_fma_f32 v56, -v100, v98, v56
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e32 v98, 0, v26, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v98, v98, v42, s0
	ds_bpermute_b32 v99, v79, v98
	s_wait_dscnt 0x0
	v_add_f32_e32 v100, v98, v99
	ds_load_2addr_b32 v[98:99], v101 offset0:64 offset1:96
	s_wait_dscnt 0x0
	ds_bpermute_b32 v102, v96, v98
	s_wait_dscnt 0x0
	v_fma_f32 v25, -v100, v102, v25
	ds_bpermute_b32 v102, v80, v98
	s_wait_dscnt 0x0
	v_fma_f32 v26, -v100, v102, v26
	ds_bpermute_b32 v102, v81, v98
	s_wait_dscnt 0x0
	v_fma_f32 v27, -v100, v102, v27
	ds_bpermute_b32 v102, v82, v98
	s_wait_dscnt 0x0
	v_fma_f32 v28, -v100, v102, v28
	ds_bpermute_b32 v102, v83, v98
	s_wait_dscnt 0x0
	v_fma_f32 v29, -v100, v102, v29
	ds_bpermute_b32 v102, v84, v98
	s_wait_dscnt 0x0
	v_fma_f32 v30, -v100, v102, v30
	ds_bpermute_b32 v102, v85, v98
	s_wait_dscnt 0x0
	v_fma_f32 v31, -v100, v102, v31
	ds_bpermute_b32 v102, v86, v98
	s_wait_dscnt 0x0
	v_fma_f32 v32, -v100, v102, v32
	ds_bpermute_b32 v102, v87, v98
	s_wait_dscnt 0x0
	v_fma_f32 v33, -v100, v102, v33
	ds_bpermute_b32 v102, v88, v98
	s_wait_dscnt 0x0
	v_fma_f32 v34, -v100, v102, v34
	ds_bpermute_b32 v102, v89, v98
	s_wait_dscnt 0x0
	v_fma_f32 v35, -v100, v102, v35
	ds_bpermute_b32 v102, v90, v98
	s_wait_dscnt 0x0
	v_fma_f32 v36, -v100, v102, v36
	ds_bpermute_b32 v102, v91, v98
	s_wait_dscnt 0x0
	v_fma_f32 v37, -v100, v102, v37
	ds_bpermute_b32 v102, v92, v98
	s_wait_dscnt 0x0
	v_fma_f32 v38, -v100, v102, v38
	ds_bpermute_b32 v102, v93, v98
	ds_bpermute_b32 v98, v94, v98
	s_wait_dscnt 0x1
	v_fma_f32 v39, -v100, v102, v39
	s_wait_dscnt 0x0
	v_fma_f32 v40, -v100, v98, v40
	ds_bpermute_b32 v98, v96, v99
	s_wait_dscnt 0x0
	v_fma_f32 v41, -v100, v98, v41
	ds_bpermute_b32 v98, v80, v99
	s_wait_dscnt 0x0
	v_fma_f32 v42, -v100, v98, v42
	ds_bpermute_b32 v98, v81, v99
	s_wait_dscnt 0x0
	v_fma_f32 v43, -v100, v98, v43
	ds_bpermute_b32 v98, v82, v99
	s_wait_dscnt 0x0
	v_fma_f32 v44, -v100, v98, v44
	ds_bpermute_b32 v98, v83, v99
	s_wait_dscnt 0x0
	v_fma_f32 v45, -v100, v98, v45
	ds_bpermute_b32 v98, v84, v99
	s_wait_dscnt 0x0
	v_fma_f32 v46, -v100, v98, v46
	ds_bpermute_b32 v98, v85, v99
	s_wait_dscnt 0x0
	v_fma_f32 v47, -v100, v98, v47
	ds_bpermute_b32 v98, v86, v99
	s_wait_dscnt 0x0
	v_fma_f32 v48, -v100, v98, v48
	ds_bpermute_b32 v98, v87, v99
	s_wait_dscnt 0x0
	v_fma_f32 v49, -v100, v98, v49
	ds_bpermute_b32 v98, v88, v99
	s_wait_dscnt 0x0
	v_fma_f32 v50, -v100, v98, v50
	ds_bpermute_b32 v98, v89, v99
	s_wait_dscnt 0x0
	v_fma_f32 v51, -v100, v98, v51
	ds_bpermute_b32 v98, v90, v99
	s_wait_dscnt 0x0
	v_fma_f32 v52, -v100, v98, v52
	ds_bpermute_b32 v98, v91, v99
	s_wait_dscnt 0x0
	v_fma_f32 v53, -v100, v98, v53
	ds_bpermute_b32 v98, v92, v99
	s_wait_dscnt 0x0
	v_fma_f32 v54, -v100, v98, v54
	ds_bpermute_b32 v98, v93, v99
	s_wait_dscnt 0x0
	v_fma_f32 v55, -v100, v98, v55
	ds_bpermute_b32 v98, v94, v99
	s_wait_dscnt 0x0
	v_fma_f32 v56, -v100, v98, v56
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e32 v98, 0, v27, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v98, v98, v43, s0
	ds_bpermute_b32 v99, v79, v98
	s_wait_dscnt 0x0
	v_add_f32_e32 v100, v98, v99
	ds_load_2addr_b32 v[98:99], v101 offset0:128 offset1:160
	s_wait_dscnt 0x0
	ds_bpermute_b32 v102, v96, v98
	s_wait_dscnt 0x0
	v_fma_f32 v25, -v100, v102, v25
	ds_bpermute_b32 v102, v80, v98
	s_wait_dscnt 0x0
	v_fma_f32 v26, -v100, v102, v26
	ds_bpermute_b32 v102, v81, v98
	s_wait_dscnt 0x0
	v_fma_f32 v27, -v100, v102, v27
	ds_bpermute_b32 v102, v82, v98
	s_wait_dscnt 0x0
	v_fma_f32 v28, -v100, v102, v28
	ds_bpermute_b32 v102, v83, v98
	s_wait_dscnt 0x0
	v_fma_f32 v29, -v100, v102, v29
	ds_bpermute_b32 v102, v84, v98
	s_wait_dscnt 0x0
	v_fma_f32 v30, -v100, v102, v30
	ds_bpermute_b32 v102, v85, v98
	s_wait_dscnt 0x0
	v_fma_f32 v31, -v100, v102, v31
	ds_bpermute_b32 v102, v86, v98
	s_wait_dscnt 0x0
	v_fma_f32 v32, -v100, v102, v32
	ds_bpermute_b32 v102, v87, v98
	s_wait_dscnt 0x0
	v_fma_f32 v33, -v100, v102, v33
	ds_bpermute_b32 v102, v88, v98
	s_wait_dscnt 0x0
	v_fma_f32 v34, -v100, v102, v34
	ds_bpermute_b32 v102, v89, v98
	s_wait_dscnt 0x0
	v_fma_f32 v35, -v100, v102, v35
	ds_bpermute_b32 v102, v90, v98
	s_wait_dscnt 0x0
	v_fma_f32 v36, -v100, v102, v36
	ds_bpermute_b32 v102, v91, v98
	s_wait_dscnt 0x0
	v_fma_f32 v37, -v100, v102, v37
	ds_bpermute_b32 v102, v92, v98
	s_wait_dscnt 0x0
	v_fma_f32 v38, -v100, v102, v38
	ds_bpermute_b32 v102, v93, v98
	ds_bpermute_b32 v98, v94, v98
	s_wait_dscnt 0x1
	v_fma_f32 v39, -v100, v102, v39
	s_wait_dscnt 0x0
	v_fma_f32 v40, -v100, v98, v40
	ds_bpermute_b32 v98, v96, v99
	s_wait_dscnt 0x0
	v_fma_f32 v41, -v100, v98, v41
	ds_bpermute_b32 v98, v80, v99
	s_wait_dscnt 0x0
	v_fma_f32 v42, -v100, v98, v42
	ds_bpermute_b32 v98, v81, v99
	s_wait_dscnt 0x0
	v_fma_f32 v43, -v100, v98, v43
	ds_bpermute_b32 v98, v82, v99
	s_wait_dscnt 0x0
	v_fma_f32 v44, -v100, v98, v44
	ds_bpermute_b32 v98, v83, v99
	s_wait_dscnt 0x0
	v_fma_f32 v45, -v100, v98, v45
	ds_bpermute_b32 v98, v84, v99
	s_wait_dscnt 0x0
	v_fma_f32 v46, -v100, v98, v46
	ds_bpermute_b32 v98, v85, v99
	s_wait_dscnt 0x0
	v_fma_f32 v47, -v100, v98, v47
	ds_bpermute_b32 v98, v86, v99
	s_wait_dscnt 0x0
	v_fma_f32 v48, -v100, v98, v48
	ds_bpermute_b32 v98, v87, v99
	s_wait_dscnt 0x0
	v_fma_f32 v49, -v100, v98, v49
	ds_bpermute_b32 v98, v88, v99
	s_wait_dscnt 0x0
	v_fma_f32 v50, -v100, v98, v50
	ds_bpermute_b32 v98, v89, v99
	s_wait_dscnt 0x0
	v_fma_f32 v51, -v100, v98, v51
	ds_bpermute_b32 v98, v90, v99
	s_wait_dscnt 0x0
	v_fma_f32 v52, -v100, v98, v52
	ds_bpermute_b32 v98, v91, v99
	s_wait_dscnt 0x0
	v_fma_f32 v53, -v100, v98, v53
	ds_bpermute_b32 v98, v92, v99
	s_wait_dscnt 0x0
	v_fma_f32 v54, -v100, v98, v54
	ds_bpermute_b32 v98, v93, v99
	s_wait_dscnt 0x0
	v_fma_f32 v55, -v100, v98, v55
	ds_bpermute_b32 v98, v94, v99
	s_wait_dscnt 0x0
	v_fma_f32 v56, -v100, v98, v56
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e32 v98, 0, v28, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v98, v98, v44, s0
	ds_bpermute_b32 v99, v79, v98
	s_wait_dscnt 0x0
	v_add_f32_e32 v100, v98, v99
	ds_load_2addr_b32 v[98:99], v101 offset0:192 offset1:224
	s_wait_dscnt 0x0
	ds_bpermute_b32 v101, v96, v98
	s_wait_dscnt 0x0
	v_fma_f32 v25, -v100, v101, v25
	ds_bpermute_b32 v101, v80, v98
	s_wait_dscnt 0x0
	v_fma_f32 v26, -v100, v101, v26
	ds_bpermute_b32 v101, v81, v98
	s_wait_dscnt 0x0
	v_fma_f32 v27, -v100, v101, v27
	ds_bpermute_b32 v101, v82, v98
	s_wait_dscnt 0x0
	v_fma_f32 v28, -v100, v101, v28
	ds_bpermute_b32 v101, v83, v98
	s_wait_dscnt 0x0
	v_fma_f32 v29, -v100, v101, v29
	ds_bpermute_b32 v101, v84, v98
	s_wait_dscnt 0x0
	v_fma_f32 v30, -v100, v101, v30
	ds_bpermute_b32 v101, v85, v98
	s_wait_dscnt 0x0
	v_fma_f32 v31, -v100, v101, v31
	ds_bpermute_b32 v101, v86, v98
	s_wait_dscnt 0x0
	v_fma_f32 v32, -v100, v101, v32
	ds_bpermute_b32 v101, v87, v98
	s_wait_dscnt 0x0
	v_fma_f32 v33, -v100, v101, v33
	ds_bpermute_b32 v101, v88, v98
	s_wait_dscnt 0x0
	v_fma_f32 v34, -v100, v101, v34
	ds_bpermute_b32 v101, v89, v98
	s_wait_dscnt 0x0
	v_fma_f32 v35, -v100, v101, v35
	ds_bpermute_b32 v101, v90, v98
	s_wait_dscnt 0x0
	v_fma_f32 v36, -v100, v101, v36
	ds_bpermute_b32 v101, v91, v98
	s_wait_dscnt 0x0
	v_fma_f32 v37, -v100, v101, v37
	ds_bpermute_b32 v101, v92, v98
	s_wait_dscnt 0x0
	v_fma_f32 v38, -v100, v101, v38
	ds_bpermute_b32 v101, v93, v98
	ds_bpermute_b32 v98, v94, v98
	s_wait_dscnt 0x1
	v_fma_f32 v39, -v100, v101, v39
	s_wait_dscnt 0x0
	v_fma_f32 v40, -v100, v98, v40
	ds_bpermute_b32 v98, v96, v99
	v_add_nc_u32_e32 v101, 0xc00, v97
	s_wait_dscnt 0x0
	v_fma_f32 v41, -v100, v98, v41
	ds_bpermute_b32 v98, v80, v99
	s_wait_dscnt 0x0
	v_fma_f32 v42, -v100, v98, v42
	ds_bpermute_b32 v98, v81, v99
	s_wait_dscnt 0x0
	v_fma_f32 v43, -v100, v98, v43
	ds_bpermute_b32 v98, v82, v99
	s_wait_dscnt 0x0
	v_fma_f32 v44, -v100, v98, v44
	ds_bpermute_b32 v98, v83, v99
	s_wait_dscnt 0x0
	v_fma_f32 v45, -v100, v98, v45
	ds_bpermute_b32 v98, v84, v99
	s_wait_dscnt 0x0
	v_fma_f32 v46, -v100, v98, v46
	ds_bpermute_b32 v98, v85, v99
	s_wait_dscnt 0x0
	v_fma_f32 v47, -v100, v98, v47
	ds_bpermute_b32 v98, v86, v99
	s_wait_dscnt 0x0
	v_fma_f32 v48, -v100, v98, v48
	ds_bpermute_b32 v98, v87, v99
	s_wait_dscnt 0x0
	v_fma_f32 v49, -v100, v98, v49
	ds_bpermute_b32 v98, v88, v99
	s_wait_dscnt 0x0
	v_fma_f32 v50, -v100, v98, v50
	ds_bpermute_b32 v98, v89, v99
	s_wait_dscnt 0x0
	v_fma_f32 v51, -v100, v98, v51
	ds_bpermute_b32 v98, v90, v99
	s_wait_dscnt 0x0
	v_fma_f32 v52, -v100, v98, v52
	ds_bpermute_b32 v98, v91, v99
	s_wait_dscnt 0x0
	v_fma_f32 v53, -v100, v98, v53
	ds_bpermute_b32 v98, v92, v99
	s_wait_dscnt 0x0
	v_fma_f32 v54, -v100, v98, v54
	ds_bpermute_b32 v98, v93, v99
	s_wait_dscnt 0x0
	v_fma_f32 v55, -v100, v98, v55
	ds_bpermute_b32 v98, v94, v99
	s_wait_dscnt 0x0
	v_fma_f32 v56, -v100, v98, v56
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e32 v98, 0, v29, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v98, v98, v45, s0
	ds_bpermute_b32 v99, v79, v98
	s_wait_dscnt 0x0
	v_add_f32_e32 v100, v98, v99
	ds_load_2addr_b32 v[98:99], v101 offset1:32
	s_wait_dscnt 0x0
	ds_bpermute_b32 v102, v96, v98
	s_wait_dscnt 0x0
	v_fma_f32 v25, -v100, v102, v25
	ds_bpermute_b32 v102, v80, v98
	s_wait_dscnt 0x0
	v_fma_f32 v26, -v100, v102, v26
	ds_bpermute_b32 v102, v81, v98
	s_wait_dscnt 0x0
	v_fma_f32 v27, -v100, v102, v27
	ds_bpermute_b32 v102, v82, v98
	s_wait_dscnt 0x0
	v_fma_f32 v28, -v100, v102, v28
	ds_bpermute_b32 v102, v83, v98
	s_wait_dscnt 0x0
	v_fma_f32 v29, -v100, v102, v29
	ds_bpermute_b32 v102, v84, v98
	s_wait_dscnt 0x0
	v_fma_f32 v30, -v100, v102, v30
	ds_bpermute_b32 v102, v85, v98
	s_wait_dscnt 0x0
	v_fma_f32 v31, -v100, v102, v31
	ds_bpermute_b32 v102, v86, v98
	s_wait_dscnt 0x0
	v_fma_f32 v32, -v100, v102, v32
	ds_bpermute_b32 v102, v87, v98
	s_wait_dscnt 0x0
	v_fma_f32 v33, -v100, v102, v33
	ds_bpermute_b32 v102, v88, v98
	s_wait_dscnt 0x0
	v_fma_f32 v34, -v100, v102, v34
	ds_bpermute_b32 v102, v89, v98
	s_wait_dscnt 0x0
	v_fma_f32 v35, -v100, v102, v35
	ds_bpermute_b32 v102, v90, v98
	s_wait_dscnt 0x0
	v_fma_f32 v36, -v100, v102, v36
	ds_bpermute_b32 v102, v91, v98
	s_wait_dscnt 0x0
	v_fma_f32 v37, -v100, v102, v37
	ds_bpermute_b32 v102, v92, v98
	s_wait_dscnt 0x0
	v_fma_f32 v38, -v100, v102, v38
	ds_bpermute_b32 v102, v93, v98
	ds_bpermute_b32 v98, v94, v98
	s_wait_dscnt 0x1
	v_fma_f32 v39, -v100, v102, v39
	s_wait_dscnt 0x0
	v_fma_f32 v40, -v100, v98, v40
	ds_bpermute_b32 v98, v96, v99
	s_wait_dscnt 0x0
	v_fma_f32 v41, -v100, v98, v41
	ds_bpermute_b32 v98, v80, v99
	s_wait_dscnt 0x0
	v_fma_f32 v42, -v100, v98, v42
	ds_bpermute_b32 v98, v81, v99
	s_wait_dscnt 0x0
	v_fma_f32 v43, -v100, v98, v43
	ds_bpermute_b32 v98, v82, v99
	s_wait_dscnt 0x0
	v_fma_f32 v44, -v100, v98, v44
	ds_bpermute_b32 v98, v83, v99
	s_wait_dscnt 0x0
	v_fma_f32 v45, -v100, v98, v45
	ds_bpermute_b32 v98, v84, v99
	s_wait_dscnt 0x0
	v_fma_f32 v46, -v100, v98, v46
	ds_bpermute_b32 v98, v85, v99
	s_wait_dscnt 0x0
	v_fma_f32 v47, -v100, v98, v47
	ds_bpermute_b32 v98, v86, v99
	s_wait_dscnt 0x0
	v_fma_f32 v48, -v100, v98, v48
	ds_bpermute_b32 v98, v87, v99
	s_wait_dscnt 0x0
	v_fma_f32 v49, -v100, v98, v49
	ds_bpermute_b32 v98, v88, v99
	s_wait_dscnt 0x0
	v_fma_f32 v50, -v100, v98, v50
	ds_bpermute_b32 v98, v89, v99
	s_wait_dscnt 0x0
	v_fma_f32 v51, -v100, v98, v51
	ds_bpermute_b32 v98, v90, v99
	s_wait_dscnt 0x0
	v_fma_f32 v52, -v100, v98, v52
	ds_bpermute_b32 v98, v91, v99
	s_wait_dscnt 0x0
	v_fma_f32 v53, -v100, v98, v53
	ds_bpermute_b32 v98, v92, v99
	s_wait_dscnt 0x0
	v_fma_f32 v54, -v100, v98, v54
	ds_bpermute_b32 v98, v93, v99
	s_wait_dscnt 0x0
	v_fma_f32 v55, -v100, v98, v55
	ds_bpermute_b32 v98, v94, v99
	s_wait_dscnt 0x0
	v_fma_f32 v56, -v100, v98, v56
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e32 v98, 0, v30, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v98, v98, v46, s0
	ds_bpermute_b32 v99, v79, v98
	s_wait_dscnt 0x0
	v_add_f32_e32 v100, v98, v99
	ds_load_2addr_b32 v[98:99], v101 offset0:64 offset1:96
	s_wait_dscnt 0x0
	ds_bpermute_b32 v102, v96, v98
	s_wait_dscnt 0x0
	v_fma_f32 v25, -v100, v102, v25
	ds_bpermute_b32 v102, v80, v98
	s_wait_dscnt 0x0
	v_fma_f32 v26, -v100, v102, v26
	ds_bpermute_b32 v102, v81, v98
	s_wait_dscnt 0x0
	v_fma_f32 v27, -v100, v102, v27
	ds_bpermute_b32 v102, v82, v98
	s_wait_dscnt 0x0
	v_fma_f32 v28, -v100, v102, v28
	ds_bpermute_b32 v102, v83, v98
	s_wait_dscnt 0x0
	v_fma_f32 v29, -v100, v102, v29
	ds_bpermute_b32 v102, v84, v98
	s_wait_dscnt 0x0
	v_fma_f32 v30, -v100, v102, v30
	ds_bpermute_b32 v102, v85, v98
	s_wait_dscnt 0x0
	v_fma_f32 v31, -v100, v102, v31
	ds_bpermute_b32 v102, v86, v98
	s_wait_dscnt 0x0
	v_fma_f32 v32, -v100, v102, v32
	ds_bpermute_b32 v102, v87, v98
	s_wait_dscnt 0x0
	v_fma_f32 v33, -v100, v102, v33
	ds_bpermute_b32 v102, v88, v98
	s_wait_dscnt 0x0
	v_fma_f32 v34, -v100, v102, v34
	ds_bpermute_b32 v102, v89, v98
	s_wait_dscnt 0x0
	v_fma_f32 v35, -v100, v102, v35
	ds_bpermute_b32 v102, v90, v98
	s_wait_dscnt 0x0
	v_fma_f32 v36, -v100, v102, v36
	ds_bpermute_b32 v102, v91, v98
	s_wait_dscnt 0x0
	v_fma_f32 v37, -v100, v102, v37
	ds_bpermute_b32 v102, v92, v98
	s_wait_dscnt 0x0
	v_fma_f32 v38, -v100, v102, v38
	ds_bpermute_b32 v102, v93, v98
	ds_bpermute_b32 v98, v94, v98
	s_wait_dscnt 0x1
	v_fma_f32 v39, -v100, v102, v39
	s_wait_dscnt 0x0
	v_fma_f32 v40, -v100, v98, v40
	ds_bpermute_b32 v98, v96, v99
	s_wait_dscnt 0x0
	v_fma_f32 v41, -v100, v98, v41
	ds_bpermute_b32 v98, v80, v99
	s_wait_dscnt 0x0
	v_fma_f32 v42, -v100, v98, v42
	ds_bpermute_b32 v98, v81, v99
	s_wait_dscnt 0x0
	v_fma_f32 v43, -v100, v98, v43
	ds_bpermute_b32 v98, v82, v99
	s_wait_dscnt 0x0
	v_fma_f32 v44, -v100, v98, v44
	ds_bpermute_b32 v98, v83, v99
	s_wait_dscnt 0x0
	v_fma_f32 v45, -v100, v98, v45
	ds_bpermute_b32 v98, v84, v99
	s_wait_dscnt 0x0
	v_fma_f32 v46, -v100, v98, v46
	ds_bpermute_b32 v98, v85, v99
	s_wait_dscnt 0x0
	v_fma_f32 v47, -v100, v98, v47
	ds_bpermute_b32 v98, v86, v99
	s_wait_dscnt 0x0
	v_fma_f32 v48, -v100, v98, v48
	ds_bpermute_b32 v98, v87, v99
	s_wait_dscnt 0x0
	v_fma_f32 v49, -v100, v98, v49
	ds_bpermute_b32 v98, v88, v99
	s_wait_dscnt 0x0
	v_fma_f32 v50, -v100, v98, v50
	ds_bpermute_b32 v98, v89, v99
	s_wait_dscnt 0x0
	v_fma_f32 v51, -v100, v98, v51
	ds_bpermute_b32 v98, v90, v99
	s_wait_dscnt 0x0
	v_fma_f32 v52, -v100, v98, v52
	ds_bpermute_b32 v98, v91, v99
	s_wait_dscnt 0x0
	v_fma_f32 v53, -v100, v98, v53
	ds_bpermute_b32 v98, v92, v99
	s_wait_dscnt 0x0
	v_fma_f32 v54, -v100, v98, v54
	ds_bpermute_b32 v98, v93, v99
	s_wait_dscnt 0x0
	v_fma_f32 v55, -v100, v98, v55
	ds_bpermute_b32 v98, v94, v99
	s_wait_dscnt 0x0
	v_fma_f32 v56, -v100, v98, v56
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e32 v98, 0, v31, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v98, v98, v47, s0
	ds_bpermute_b32 v99, v79, v98
	s_wait_dscnt 0x0
	v_add_f32_e32 v100, v98, v99
	ds_load_2addr_b32 v[98:99], v101 offset0:128 offset1:160
	s_wait_dscnt 0x0
	ds_bpermute_b32 v102, v96, v98
	s_wait_dscnt 0x0
	v_fma_f32 v25, -v100, v102, v25
	ds_bpermute_b32 v102, v80, v98
	s_wait_dscnt 0x0
	v_fma_f32 v26, -v100, v102, v26
	ds_bpermute_b32 v102, v81, v98
	s_wait_dscnt 0x0
	v_fma_f32 v27, -v100, v102, v27
	ds_bpermute_b32 v102, v82, v98
	s_wait_dscnt 0x0
	v_fma_f32 v28, -v100, v102, v28
	ds_bpermute_b32 v102, v83, v98
	s_wait_dscnt 0x0
	v_fma_f32 v29, -v100, v102, v29
	ds_bpermute_b32 v102, v84, v98
	s_wait_dscnt 0x0
	v_fma_f32 v30, -v100, v102, v30
	ds_bpermute_b32 v102, v85, v98
	s_wait_dscnt 0x0
	v_fma_f32 v31, -v100, v102, v31
	ds_bpermute_b32 v102, v86, v98
	s_wait_dscnt 0x0
	v_fma_f32 v32, -v100, v102, v32
	ds_bpermute_b32 v102, v87, v98
	s_wait_dscnt 0x0
	v_fma_f32 v33, -v100, v102, v33
	ds_bpermute_b32 v102, v88, v98
	s_wait_dscnt 0x0
	v_fma_f32 v34, -v100, v102, v34
	ds_bpermute_b32 v102, v89, v98
	s_wait_dscnt 0x0
	v_fma_f32 v35, -v100, v102, v35
	ds_bpermute_b32 v102, v90, v98
	s_wait_dscnt 0x0
	v_fma_f32 v36, -v100, v102, v36
	ds_bpermute_b32 v102, v91, v98
	s_wait_dscnt 0x0
	v_fma_f32 v37, -v100, v102, v37
	ds_bpermute_b32 v102, v92, v98
	s_wait_dscnt 0x0
	v_fma_f32 v38, -v100, v102, v38
	ds_bpermute_b32 v102, v93, v98
	ds_bpermute_b32 v98, v94, v98
	s_wait_dscnt 0x1
	v_fma_f32 v39, -v100, v102, v39
	s_wait_dscnt 0x0
	v_fma_f32 v40, -v100, v98, v40
	ds_bpermute_b32 v98, v96, v99
	s_wait_dscnt 0x0
	v_fma_f32 v41, -v100, v98, v41
	ds_bpermute_b32 v98, v80, v99
	s_wait_dscnt 0x0
	v_fma_f32 v42, -v100, v98, v42
	ds_bpermute_b32 v98, v81, v99
	s_wait_dscnt 0x0
	v_fma_f32 v43, -v100, v98, v43
	ds_bpermute_b32 v98, v82, v99
	s_wait_dscnt 0x0
	v_fma_f32 v44, -v100, v98, v44
	ds_bpermute_b32 v98, v83, v99
	s_wait_dscnt 0x0
	v_fma_f32 v45, -v100, v98, v45
	ds_bpermute_b32 v98, v84, v99
	s_wait_dscnt 0x0
	v_fma_f32 v46, -v100, v98, v46
	ds_bpermute_b32 v98, v85, v99
	s_wait_dscnt 0x0
	v_fma_f32 v47, -v100, v98, v47
	ds_bpermute_b32 v98, v86, v99
	s_wait_dscnt 0x0
	v_fma_f32 v48, -v100, v98, v48
	ds_bpermute_b32 v98, v87, v99
	s_wait_dscnt 0x0
	v_fma_f32 v49, -v100, v98, v49
	ds_bpermute_b32 v98, v88, v99
	s_wait_dscnt 0x0
	v_fma_f32 v50, -v100, v98, v50
	ds_bpermute_b32 v98, v89, v99
	s_wait_dscnt 0x0
	v_fma_f32 v51, -v100, v98, v51
	ds_bpermute_b32 v98, v90, v99
	s_wait_dscnt 0x0
	v_fma_f32 v52, -v100, v98, v52
	ds_bpermute_b32 v98, v91, v99
	s_wait_dscnt 0x0
	v_fma_f32 v53, -v100, v98, v53
	ds_bpermute_b32 v98, v92, v99
	s_wait_dscnt 0x0
	v_fma_f32 v54, -v100, v98, v54
	ds_bpermute_b32 v98, v93, v99
	s_wait_dscnt 0x0
	v_fma_f32 v55, -v100, v98, v55
	ds_bpermute_b32 v98, v94, v99
	s_wait_dscnt 0x0
	v_fma_f32 v56, -v100, v98, v56
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e32 v98, 0, v32, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v98, v98, v48, s0
	ds_bpermute_b32 v99, v79, v98
	s_wait_dscnt 0x0
	v_add_f32_e32 v100, v98, v99
	ds_load_2addr_b32 v[98:99], v101 offset0:192 offset1:224
	s_wait_dscnt 0x0
	ds_bpermute_b32 v101, v96, v98
	s_wait_dscnt 0x0
	v_fma_f32 v25, -v100, v101, v25
	ds_bpermute_b32 v101, v80, v98
	s_wait_dscnt 0x0
	v_fma_f32 v26, -v100, v101, v26
	ds_bpermute_b32 v101, v81, v98
	s_wait_dscnt 0x0
	v_fma_f32 v27, -v100, v101, v27
	ds_bpermute_b32 v101, v82, v98
	s_wait_dscnt 0x0
	v_fma_f32 v28, -v100, v101, v28
	ds_bpermute_b32 v101, v83, v98
	s_wait_dscnt 0x0
	v_fma_f32 v29, -v100, v101, v29
	ds_bpermute_b32 v101, v84, v98
	s_wait_dscnt 0x0
	v_fma_f32 v30, -v100, v101, v30
	ds_bpermute_b32 v101, v85, v98
	s_wait_dscnt 0x0
	v_fma_f32 v31, -v100, v101, v31
	ds_bpermute_b32 v101, v86, v98
	s_wait_dscnt 0x0
	v_fma_f32 v32, -v100, v101, v32
	ds_bpermute_b32 v101, v87, v98
	s_wait_dscnt 0x0
	v_fma_f32 v33, -v100, v101, v33
	ds_bpermute_b32 v101, v88, v98
	s_wait_dscnt 0x0
	v_fma_f32 v34, -v100, v101, v34
	ds_bpermute_b32 v101, v89, v98
	s_wait_dscnt 0x0
	v_fma_f32 v35, -v100, v101, v35
	ds_bpermute_b32 v101, v90, v98
	s_wait_dscnt 0x0
	v_fma_f32 v36, -v100, v101, v36
	ds_bpermute_b32 v101, v91, v98
	s_wait_dscnt 0x0
	v_fma_f32 v37, -v100, v101, v37
	ds_bpermute_b32 v101, v92, v98
	s_wait_dscnt 0x0
	v_fma_f32 v38, -v100, v101, v38
	ds_bpermute_b32 v101, v93, v98
	ds_bpermute_b32 v98, v94, v98
	s_wait_dscnt 0x1
	v_fma_f32 v39, -v100, v101, v39
	s_wait_dscnt 0x0
	v_fma_f32 v40, -v100, v98, v40
	ds_bpermute_b32 v98, v96, v99
	v_add_nc_u32_e32 v101, 0x1000, v97
	s_wait_dscnt 0x0
	v_fma_f32 v41, -v100, v98, v41
	ds_bpermute_b32 v98, v80, v99
	s_wait_dscnt 0x0
	v_fma_f32 v42, -v100, v98, v42
	ds_bpermute_b32 v98, v81, v99
	s_wait_dscnt 0x0
	v_fma_f32 v43, -v100, v98, v43
	ds_bpermute_b32 v98, v82, v99
	s_wait_dscnt 0x0
	v_fma_f32 v44, -v100, v98, v44
	ds_bpermute_b32 v98, v83, v99
	s_wait_dscnt 0x0
	v_fma_f32 v45, -v100, v98, v45
	ds_bpermute_b32 v98, v84, v99
	s_wait_dscnt 0x0
	v_fma_f32 v46, -v100, v98, v46
	ds_bpermute_b32 v98, v85, v99
	s_wait_dscnt 0x0
	v_fma_f32 v47, -v100, v98, v47
	ds_bpermute_b32 v98, v86, v99
	s_wait_dscnt 0x0
	v_fma_f32 v48, -v100, v98, v48
	ds_bpermute_b32 v98, v87, v99
	s_wait_dscnt 0x0
	v_fma_f32 v49, -v100, v98, v49
	ds_bpermute_b32 v98, v88, v99
	s_wait_dscnt 0x0
	v_fma_f32 v50, -v100, v98, v50
	ds_bpermute_b32 v98, v89, v99
	s_wait_dscnt 0x0
	v_fma_f32 v51, -v100, v98, v51
	ds_bpermute_b32 v98, v90, v99
	s_wait_dscnt 0x0
	v_fma_f32 v52, -v100, v98, v52
	ds_bpermute_b32 v98, v91, v99
	s_wait_dscnt 0x0
	v_fma_f32 v53, -v100, v98, v53
	ds_bpermute_b32 v98, v92, v99
	s_wait_dscnt 0x0
	v_fma_f32 v54, -v100, v98, v54
	ds_bpermute_b32 v98, v93, v99
	s_wait_dscnt 0x0
	v_fma_f32 v55, -v100, v98, v55
	ds_bpermute_b32 v98, v94, v99
	s_wait_dscnt 0x0
	v_fma_f32 v56, -v100, v98, v56
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v98, 0, v33, s1
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v98, v98, v49, s2
	ds_bpermute_b32 v99, v79, v98
	s_wait_dscnt 0x0
	v_add_f32_e32 v100, v98, v99
	ds_load_2addr_b32 v[98:99], v101 offset1:32
	s_wait_dscnt 0x0
	ds_bpermute_b32 v102, v96, v98
	s_wait_dscnt 0x0
	v_fma_f32 v25, -v100, v102, v25
	ds_bpermute_b32 v102, v80, v98
	s_wait_dscnt 0x0
	v_fma_f32 v26, -v100, v102, v26
	ds_bpermute_b32 v102, v81, v98
	s_wait_dscnt 0x0
	v_fma_f32 v27, -v100, v102, v27
	ds_bpermute_b32 v102, v82, v98
	s_wait_dscnt 0x0
	v_fma_f32 v28, -v100, v102, v28
	ds_bpermute_b32 v102, v83, v98
	s_wait_dscnt 0x0
	v_fma_f32 v29, -v100, v102, v29
	ds_bpermute_b32 v102, v84, v98
	s_wait_dscnt 0x0
	v_fma_f32 v30, -v100, v102, v30
	ds_bpermute_b32 v102, v85, v98
	s_wait_dscnt 0x0
	v_fma_f32 v31, -v100, v102, v31
	ds_bpermute_b32 v102, v86, v98
	s_wait_dscnt 0x0
	v_fma_f32 v32, -v100, v102, v32
	ds_bpermute_b32 v102, v87, v98
	s_wait_dscnt 0x0
	v_fma_f32 v33, -v100, v102, v33
	ds_bpermute_b32 v102, v88, v98
	s_wait_dscnt 0x0
	v_fma_f32 v34, -v100, v102, v34
	ds_bpermute_b32 v102, v89, v98
	s_wait_dscnt 0x0
	v_fma_f32 v35, -v100, v102, v35
	ds_bpermute_b32 v102, v90, v98
	s_wait_dscnt 0x0
	v_fma_f32 v36, -v100, v102, v36
	ds_bpermute_b32 v102, v91, v98
	s_wait_dscnt 0x0
	v_fma_f32 v37, -v100, v102, v37
	ds_bpermute_b32 v102, v92, v98
	s_wait_dscnt 0x0
	v_fma_f32 v38, -v100, v102, v38
	ds_bpermute_b32 v102, v93, v98
	ds_bpermute_b32 v98, v94, v98
	s_wait_dscnt 0x1
	v_fma_f32 v39, -v100, v102, v39
	s_wait_dscnt 0x0
	v_fma_f32 v40, -v100, v98, v40
	ds_bpermute_b32 v98, v96, v99
	s_wait_dscnt 0x0
	v_fma_f32 v41, -v100, v98, v41
	ds_bpermute_b32 v98, v80, v99
	s_wait_dscnt 0x0
	v_fma_f32 v42, -v100, v98, v42
	ds_bpermute_b32 v98, v81, v99
	s_wait_dscnt 0x0
	v_fma_f32 v43, -v100, v98, v43
	ds_bpermute_b32 v98, v82, v99
	s_wait_dscnt 0x0
	v_fma_f32 v44, -v100, v98, v44
	ds_bpermute_b32 v98, v83, v99
	s_wait_dscnt 0x0
	v_fma_f32 v45, -v100, v98, v45
	ds_bpermute_b32 v98, v84, v99
	s_wait_dscnt 0x0
	v_fma_f32 v46, -v100, v98, v46
	ds_bpermute_b32 v98, v85, v99
	s_wait_dscnt 0x0
	v_fma_f32 v47, -v100, v98, v47
	ds_bpermute_b32 v98, v86, v99
	s_wait_dscnt 0x0
	v_fma_f32 v48, -v100, v98, v48
	ds_bpermute_b32 v98, v87, v99
	s_wait_dscnt 0x0
	v_fma_f32 v49, -v100, v98, v49
	ds_bpermute_b32 v98, v88, v99
	s_wait_dscnt 0x0
	v_fma_f32 v50, -v100, v98, v50
	ds_bpermute_b32 v98, v89, v99
	s_wait_dscnt 0x0
	v_fma_f32 v51, -v100, v98, v51
	ds_bpermute_b32 v98, v90, v99
	s_wait_dscnt 0x0
	v_fma_f32 v52, -v100, v98, v52
	ds_bpermute_b32 v98, v91, v99
	s_wait_dscnt 0x0
	v_fma_f32 v53, -v100, v98, v53
	ds_bpermute_b32 v98, v92, v99
	s_wait_dscnt 0x0
	v_fma_f32 v54, -v100, v98, v54
	ds_bpermute_b32 v98, v93, v99
	s_wait_dscnt 0x0
	v_fma_f32 v55, -v100, v98, v55
	ds_bpermute_b32 v98, v94, v99
	s_wait_dscnt 0x0
	v_fma_f32 v56, -v100, v98, v56
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v98, 0, v34, s1
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v98, v98, v50, s2
	ds_bpermute_b32 v99, v79, v98
	s_wait_dscnt 0x0
	v_add_f32_e32 v100, v98, v99
	ds_load_2addr_b32 v[98:99], v101 offset0:64 offset1:96
	s_wait_dscnt 0x0
	ds_bpermute_b32 v102, v96, v98
	s_wait_dscnt 0x0
	v_fma_f32 v25, -v100, v102, v25
	ds_bpermute_b32 v102, v80, v98
	s_wait_dscnt 0x0
	v_fma_f32 v26, -v100, v102, v26
	ds_bpermute_b32 v102, v81, v98
	s_wait_dscnt 0x0
	v_fma_f32 v27, -v100, v102, v27
	ds_bpermute_b32 v102, v82, v98
	s_wait_dscnt 0x0
	v_fma_f32 v28, -v100, v102, v28
	ds_bpermute_b32 v102, v83, v98
	s_wait_dscnt 0x0
	v_fma_f32 v29, -v100, v102, v29
	ds_bpermute_b32 v102, v84, v98
	s_wait_dscnt 0x0
	v_fma_f32 v30, -v100, v102, v30
	ds_bpermute_b32 v102, v85, v98
	s_wait_dscnt 0x0
	v_fma_f32 v31, -v100, v102, v31
	ds_bpermute_b32 v102, v86, v98
	s_wait_dscnt 0x0
	v_fma_f32 v32, -v100, v102, v32
	ds_bpermute_b32 v102, v87, v98
	s_wait_dscnt 0x0
	v_fma_f32 v33, -v100, v102, v33
	ds_bpermute_b32 v102, v88, v98
	s_wait_dscnt 0x0
	v_fma_f32 v34, -v100, v102, v34
	ds_bpermute_b32 v102, v89, v98
	s_wait_dscnt 0x0
	v_fma_f32 v35, -v100, v102, v35
	ds_bpermute_b32 v102, v90, v98
	s_wait_dscnt 0x0
	v_fma_f32 v36, -v100, v102, v36
	ds_bpermute_b32 v102, v91, v98
	s_wait_dscnt 0x0
	v_fma_f32 v37, -v100, v102, v37
	ds_bpermute_b32 v102, v92, v98
	s_wait_dscnt 0x0
	v_fma_f32 v38, -v100, v102, v38
	ds_bpermute_b32 v102, v93, v98
	ds_bpermute_b32 v98, v94, v98
	s_wait_dscnt 0x1
	v_fma_f32 v39, -v100, v102, v39
	s_wait_dscnt 0x0
	v_fma_f32 v40, -v100, v98, v40
	ds_bpermute_b32 v98, v96, v99
	s_wait_dscnt 0x0
	v_fma_f32 v41, -v100, v98, v41
	ds_bpermute_b32 v98, v80, v99
	s_wait_dscnt 0x0
	v_fma_f32 v42, -v100, v98, v42
	ds_bpermute_b32 v98, v81, v99
	s_wait_dscnt 0x0
	v_fma_f32 v43, -v100, v98, v43
	ds_bpermute_b32 v98, v82, v99
	s_wait_dscnt 0x0
	v_fma_f32 v44, -v100, v98, v44
	ds_bpermute_b32 v98, v83, v99
	s_wait_dscnt 0x0
	v_fma_f32 v45, -v100, v98, v45
	ds_bpermute_b32 v98, v84, v99
	s_wait_dscnt 0x0
	v_fma_f32 v46, -v100, v98, v46
	ds_bpermute_b32 v98, v85, v99
	s_wait_dscnt 0x0
	v_fma_f32 v47, -v100, v98, v47
	ds_bpermute_b32 v98, v86, v99
	s_wait_dscnt 0x0
	v_fma_f32 v48, -v100, v98, v48
	ds_bpermute_b32 v98, v87, v99
	s_wait_dscnt 0x0
	v_fma_f32 v49, -v100, v98, v49
	ds_bpermute_b32 v98, v88, v99
	s_wait_dscnt 0x0
	v_fma_f32 v50, -v100, v98, v50
	ds_bpermute_b32 v98, v89, v99
	s_wait_dscnt 0x0
	v_fma_f32 v51, -v100, v98, v51
	ds_bpermute_b32 v98, v90, v99
	s_wait_dscnt 0x0
	v_fma_f32 v52, -v100, v98, v52
	ds_bpermute_b32 v98, v91, v99
	s_wait_dscnt 0x0
	v_fma_f32 v53, -v100, v98, v53
	ds_bpermute_b32 v98, v92, v99
	s_wait_dscnt 0x0
	v_fma_f32 v54, -v100, v98, v54
	ds_bpermute_b32 v98, v93, v99
	s_wait_dscnt 0x0
	v_fma_f32 v55, -v100, v98, v55
	ds_bpermute_b32 v98, v94, v99
	s_wait_dscnt 0x0
	v_fma_f32 v56, -v100, v98, v56
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v98, 0, v35, s1
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v98, v98, v51, s2
	ds_bpermute_b32 v99, v79, v98
	s_wait_dscnt 0x0
	v_add_f32_e32 v100, v98, v99
	ds_load_2addr_b32 v[98:99], v101 offset0:128 offset1:160
	s_wait_dscnt 0x0
	ds_bpermute_b32 v102, v96, v98
	s_wait_dscnt 0x0
	v_fma_f32 v25, -v100, v102, v25
	ds_bpermute_b32 v102, v80, v98
	s_wait_dscnt 0x0
	v_fma_f32 v26, -v100, v102, v26
	ds_bpermute_b32 v102, v81, v98
	s_wait_dscnt 0x0
	v_fma_f32 v27, -v100, v102, v27
	ds_bpermute_b32 v102, v82, v98
	s_wait_dscnt 0x0
	v_fma_f32 v28, -v100, v102, v28
	ds_bpermute_b32 v102, v83, v98
	s_wait_dscnt 0x0
	v_fma_f32 v29, -v100, v102, v29
	ds_bpermute_b32 v102, v84, v98
	s_wait_dscnt 0x0
	v_fma_f32 v30, -v100, v102, v30
	ds_bpermute_b32 v102, v85, v98
	s_wait_dscnt 0x0
	v_fma_f32 v31, -v100, v102, v31
	ds_bpermute_b32 v102, v86, v98
	s_wait_dscnt 0x0
	v_fma_f32 v32, -v100, v102, v32
	ds_bpermute_b32 v102, v87, v98
	s_wait_dscnt 0x0
	v_fma_f32 v33, -v100, v102, v33
	ds_bpermute_b32 v102, v88, v98
	s_wait_dscnt 0x0
	v_fma_f32 v34, -v100, v102, v34
	ds_bpermute_b32 v102, v89, v98
	s_wait_dscnt 0x0
	v_fma_f32 v35, -v100, v102, v35
	ds_bpermute_b32 v102, v90, v98
	s_wait_dscnt 0x0
	v_fma_f32 v36, -v100, v102, v36
	ds_bpermute_b32 v102, v91, v98
	s_wait_dscnt 0x0
	v_fma_f32 v37, -v100, v102, v37
	ds_bpermute_b32 v102, v92, v98
	s_wait_dscnt 0x0
	v_fma_f32 v38, -v100, v102, v38
	ds_bpermute_b32 v102, v93, v98
	ds_bpermute_b32 v98, v94, v98
	s_wait_dscnt 0x1
	v_fma_f32 v39, -v100, v102, v39
	s_wait_dscnt 0x0
	v_fma_f32 v40, -v100, v98, v40
	ds_bpermute_b32 v98, v96, v99
	s_wait_dscnt 0x0
	v_fma_f32 v41, -v100, v98, v41
	ds_bpermute_b32 v98, v80, v99
	s_wait_dscnt 0x0
	v_fma_f32 v42, -v100, v98, v42
	ds_bpermute_b32 v98, v81, v99
	s_wait_dscnt 0x0
	v_fma_f32 v43, -v100, v98, v43
	ds_bpermute_b32 v98, v82, v99
	s_wait_dscnt 0x0
	v_fma_f32 v44, -v100, v98, v44
	ds_bpermute_b32 v98, v83, v99
	s_wait_dscnt 0x0
	v_fma_f32 v45, -v100, v98, v45
	ds_bpermute_b32 v98, v84, v99
	s_wait_dscnt 0x0
	v_fma_f32 v46, -v100, v98, v46
	ds_bpermute_b32 v98, v85, v99
	s_wait_dscnt 0x0
	v_fma_f32 v47, -v100, v98, v47
	ds_bpermute_b32 v98, v86, v99
	s_wait_dscnt 0x0
	v_fma_f32 v48, -v100, v98, v48
	ds_bpermute_b32 v98, v87, v99
	s_wait_dscnt 0x0
	v_fma_f32 v49, -v100, v98, v49
	ds_bpermute_b32 v98, v88, v99
	s_wait_dscnt 0x0
	v_fma_f32 v50, -v100, v98, v50
	ds_bpermute_b32 v98, v89, v99
	s_wait_dscnt 0x0
	v_fma_f32 v51, -v100, v98, v51
	ds_bpermute_b32 v98, v90, v99
	s_wait_dscnt 0x0
	v_fma_f32 v52, -v100, v98, v52
	ds_bpermute_b32 v98, v91, v99
	s_wait_dscnt 0x0
	v_fma_f32 v53, -v100, v98, v53
	ds_bpermute_b32 v98, v92, v99
	s_wait_dscnt 0x0
	v_fma_f32 v54, -v100, v98, v54
	ds_bpermute_b32 v98, v93, v99
	s_wait_dscnt 0x0
	v_fma_f32 v55, -v100, v98, v55
	ds_bpermute_b32 v98, v94, v99
	s_wait_dscnt 0x0
	v_fma_f32 v56, -v100, v98, v56
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v98, 0, v36, s1
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v98, v98, v52, s2
	ds_bpermute_b32 v99, v79, v98
	s_wait_dscnt 0x0
	v_add_f32_e32 v100, v98, v99
	ds_load_2addr_b32 v[98:99], v101 offset0:192 offset1:224
	s_wait_dscnt 0x0
	ds_bpermute_b32 v101, v96, v98
	s_wait_dscnt 0x0
	v_fma_f32 v25, -v100, v101, v25
	ds_bpermute_b32 v101, v80, v98
	s_wait_dscnt 0x0
	v_fma_f32 v26, -v100, v101, v26
	ds_bpermute_b32 v101, v81, v98
	s_wait_dscnt 0x0
	v_fma_f32 v27, -v100, v101, v27
	ds_bpermute_b32 v101, v82, v98
	s_wait_dscnt 0x0
	v_fma_f32 v28, -v100, v101, v28
	ds_bpermute_b32 v101, v83, v98
	s_wait_dscnt 0x0
	v_fma_f32 v29, -v100, v101, v29
	ds_bpermute_b32 v101, v84, v98
	s_wait_dscnt 0x0
	v_fma_f32 v30, -v100, v101, v30
	ds_bpermute_b32 v101, v85, v98
	s_wait_dscnt 0x0
	v_fma_f32 v31, -v100, v101, v31
	ds_bpermute_b32 v101, v86, v98
	s_wait_dscnt 0x0
	v_fma_f32 v32, -v100, v101, v32
	ds_bpermute_b32 v101, v87, v98
	s_wait_dscnt 0x0
	v_fma_f32 v33, -v100, v101, v33
	ds_bpermute_b32 v101, v88, v98
	s_wait_dscnt 0x0
	v_fma_f32 v34, -v100, v101, v34
	ds_bpermute_b32 v101, v89, v98
	s_wait_dscnt 0x0
	v_fma_f32 v35, -v100, v101, v35
	ds_bpermute_b32 v101, v90, v98
	s_wait_dscnt 0x0
	v_fma_f32 v36, -v100, v101, v36
	ds_bpermute_b32 v101, v91, v98
	s_wait_dscnt 0x0
	v_fma_f32 v37, -v100, v101, v37
	ds_bpermute_b32 v101, v92, v98
	s_wait_dscnt 0x0
	v_fma_f32 v38, -v100, v101, v38
	ds_bpermute_b32 v101, v93, v98
	ds_bpermute_b32 v98, v94, v98
	s_wait_dscnt 0x1
	v_fma_f32 v39, -v100, v101, v39
	s_wait_dscnt 0x0
	v_fma_f32 v40, -v100, v98, v40
	ds_bpermute_b32 v98, v96, v99
	v_add_nc_u32_e32 v101, 0x1400, v97
	v_add_nc_u32_e32 v97, 0x1c00, v97
	s_wait_dscnt 0x0
	v_fma_f32 v41, -v100, v98, v41
	ds_bpermute_b32 v98, v80, v99
	s_wait_dscnt 0x0
	v_fma_f32 v42, -v100, v98, v42
	ds_bpermute_b32 v98, v81, v99
	s_wait_dscnt 0x0
	v_fma_f32 v43, -v100, v98, v43
	ds_bpermute_b32 v98, v82, v99
	s_wait_dscnt 0x0
	v_fma_f32 v44, -v100, v98, v44
	ds_bpermute_b32 v98, v83, v99
	s_wait_dscnt 0x0
	v_fma_f32 v45, -v100, v98, v45
	ds_bpermute_b32 v98, v84, v99
	s_wait_dscnt 0x0
	v_fma_f32 v46, -v100, v98, v46
	ds_bpermute_b32 v98, v85, v99
	s_wait_dscnt 0x0
	v_fma_f32 v47, -v100, v98, v47
	ds_bpermute_b32 v98, v86, v99
	s_wait_dscnt 0x0
	v_fma_f32 v48, -v100, v98, v48
	ds_bpermute_b32 v98, v87, v99
	s_wait_dscnt 0x0
	v_fma_f32 v49, -v100, v98, v49
	ds_bpermute_b32 v98, v88, v99
	s_wait_dscnt 0x0
	v_fma_f32 v50, -v100, v98, v50
	ds_bpermute_b32 v98, v89, v99
	s_wait_dscnt 0x0
	v_fma_f32 v51, -v100, v98, v51
	ds_bpermute_b32 v98, v90, v99
	s_wait_dscnt 0x0
	v_fma_f32 v52, -v100, v98, v52
	ds_bpermute_b32 v98, v91, v99
	s_wait_dscnt 0x0
	v_fma_f32 v53, -v100, v98, v53
	ds_bpermute_b32 v98, v92, v99
	s_wait_dscnt 0x0
	v_fma_f32 v54, -v100, v98, v54
	ds_bpermute_b32 v98, v93, v99
	s_wait_dscnt 0x0
	v_fma_f32 v55, -v100, v98, v55
	ds_bpermute_b32 v98, v94, v99
	s_wait_dscnt 0x0
	v_fma_f32 v56, -v100, v98, v56
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v98, 0, v37, s1
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v98, v98, v53, s2
	ds_bpermute_b32 v99, v79, v98
	s_wait_dscnt 0x0
	v_add_f32_e32 v100, v98, v99
	ds_load_2addr_b32 v[98:99], v101 offset1:32
	s_wait_dscnt 0x0
	ds_bpermute_b32 v102, v96, v98
	s_wait_dscnt 0x0
	v_fma_f32 v25, -v100, v102, v25
	ds_bpermute_b32 v102, v80, v98
	s_wait_dscnt 0x0
	v_fma_f32 v26, -v100, v102, v26
	ds_bpermute_b32 v102, v81, v98
	s_wait_dscnt 0x0
	v_fma_f32 v27, -v100, v102, v27
	ds_bpermute_b32 v102, v82, v98
	s_wait_dscnt 0x0
	v_fma_f32 v28, -v100, v102, v28
	ds_bpermute_b32 v102, v83, v98
	s_wait_dscnt 0x0
	v_fma_f32 v29, -v100, v102, v29
	ds_bpermute_b32 v102, v84, v98
	s_wait_dscnt 0x0
	v_fma_f32 v30, -v100, v102, v30
	ds_bpermute_b32 v102, v85, v98
	s_wait_dscnt 0x0
	v_fma_f32 v31, -v100, v102, v31
	ds_bpermute_b32 v102, v86, v98
	s_wait_dscnt 0x0
	v_fma_f32 v32, -v100, v102, v32
	ds_bpermute_b32 v102, v87, v98
	s_wait_dscnt 0x0
	v_fma_f32 v33, -v100, v102, v33
	ds_bpermute_b32 v102, v88, v98
	s_wait_dscnt 0x0
	v_fma_f32 v34, -v100, v102, v34
	ds_bpermute_b32 v102, v89, v98
	s_wait_dscnt 0x0
	v_fma_f32 v35, -v100, v102, v35
	ds_bpermute_b32 v102, v90, v98
	s_wait_dscnt 0x0
	v_fma_f32 v36, -v100, v102, v36
	ds_bpermute_b32 v102, v91, v98
	s_wait_dscnt 0x0
	v_fma_f32 v37, -v100, v102, v37
	ds_bpermute_b32 v102, v92, v98
	s_wait_dscnt 0x0
	v_fma_f32 v38, -v100, v102, v38
	ds_bpermute_b32 v102, v93, v98
	ds_bpermute_b32 v98, v94, v98
	s_wait_dscnt 0x1
	v_fma_f32 v39, -v100, v102, v39
	s_wait_dscnt 0x0
	v_fma_f32 v40, -v100, v98, v40
	ds_bpermute_b32 v98, v96, v99
	s_wait_dscnt 0x0
	v_fma_f32 v41, -v100, v98, v41
	ds_bpermute_b32 v98, v80, v99
	s_wait_dscnt 0x0
	v_fma_f32 v42, -v100, v98, v42
	ds_bpermute_b32 v98, v81, v99
	s_wait_dscnt 0x0
	v_fma_f32 v43, -v100, v98, v43
	ds_bpermute_b32 v98, v82, v99
	s_wait_dscnt 0x0
	v_fma_f32 v44, -v100, v98, v44
	ds_bpermute_b32 v98, v83, v99
	s_wait_dscnt 0x0
	v_fma_f32 v45, -v100, v98, v45
	ds_bpermute_b32 v98, v84, v99
	s_wait_dscnt 0x0
	v_fma_f32 v46, -v100, v98, v46
	ds_bpermute_b32 v98, v85, v99
	s_wait_dscnt 0x0
	v_fma_f32 v47, -v100, v98, v47
	ds_bpermute_b32 v98, v86, v99
	s_wait_dscnt 0x0
	v_fma_f32 v48, -v100, v98, v48
	ds_bpermute_b32 v98, v87, v99
	s_wait_dscnt 0x0
	v_fma_f32 v49, -v100, v98, v49
	ds_bpermute_b32 v98, v88, v99
	s_wait_dscnt 0x0
	v_fma_f32 v50, -v100, v98, v50
	ds_bpermute_b32 v98, v89, v99
	s_wait_dscnt 0x0
	v_fma_f32 v51, -v100, v98, v51
	ds_bpermute_b32 v98, v90, v99
	s_wait_dscnt 0x0
	v_fma_f32 v52, -v100, v98, v52
	ds_bpermute_b32 v98, v91, v99
	s_wait_dscnt 0x0
	v_fma_f32 v53, -v100, v98, v53
	ds_bpermute_b32 v98, v92, v99
	s_wait_dscnt 0x0
	v_fma_f32 v54, -v100, v98, v54
	ds_bpermute_b32 v98, v93, v99
	s_wait_dscnt 0x0
	v_fma_f32 v55, -v100, v98, v55
	ds_bpermute_b32 v98, v94, v99
	s_wait_dscnt 0x0
	v_fma_f32 v56, -v100, v98, v56
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v98, 0, v38, s1
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v98, v98, v54, s2
	ds_bpermute_b32 v99, v79, v98
	s_wait_dscnt 0x0
	v_add_f32_e32 v100, v98, v99
	ds_load_2addr_b32 v[98:99], v101 offset0:64 offset1:96
	s_wait_dscnt 0x0
	ds_bpermute_b32 v102, v96, v98
	s_wait_dscnt 0x0
	v_fma_f32 v25, -v100, v102, v25
	ds_bpermute_b32 v102, v80, v98
	s_wait_dscnt 0x0
	v_fma_f32 v26, -v100, v102, v26
	ds_bpermute_b32 v102, v81, v98
	s_wait_dscnt 0x0
	v_fma_f32 v27, -v100, v102, v27
	ds_bpermute_b32 v102, v82, v98
	s_wait_dscnt 0x0
	v_fma_f32 v28, -v100, v102, v28
	ds_bpermute_b32 v102, v83, v98
	s_wait_dscnt 0x0
	v_fma_f32 v29, -v100, v102, v29
	ds_bpermute_b32 v102, v84, v98
	s_wait_dscnt 0x0
	v_fma_f32 v30, -v100, v102, v30
	ds_bpermute_b32 v102, v85, v98
	s_wait_dscnt 0x0
	v_fma_f32 v31, -v100, v102, v31
	ds_bpermute_b32 v102, v86, v98
	s_wait_dscnt 0x0
	v_fma_f32 v32, -v100, v102, v32
	ds_bpermute_b32 v102, v87, v98
	s_wait_dscnt 0x0
	v_fma_f32 v33, -v100, v102, v33
	ds_bpermute_b32 v102, v88, v98
	s_wait_dscnt 0x0
	v_fma_f32 v34, -v100, v102, v34
	ds_bpermute_b32 v102, v89, v98
	s_wait_dscnt 0x0
	v_fma_f32 v35, -v100, v102, v35
	ds_bpermute_b32 v102, v90, v98
	s_wait_dscnt 0x0
	v_fma_f32 v36, -v100, v102, v36
	ds_bpermute_b32 v102, v91, v98
	s_wait_dscnt 0x0
	v_fma_f32 v37, -v100, v102, v37
	ds_bpermute_b32 v102, v92, v98
	s_wait_dscnt 0x0
	v_fma_f32 v38, -v100, v102, v38
	ds_bpermute_b32 v102, v93, v98
	ds_bpermute_b32 v98, v94, v98
	s_wait_dscnt 0x1
	v_fma_f32 v39, -v100, v102, v39
	s_wait_dscnt 0x0
	v_fma_f32 v40, -v100, v98, v40
	ds_bpermute_b32 v98, v96, v99
	s_wait_dscnt 0x0
	v_fma_f32 v41, -v100, v98, v41
	ds_bpermute_b32 v98, v80, v99
	s_wait_dscnt 0x0
	v_fma_f32 v42, -v100, v98, v42
	ds_bpermute_b32 v98, v81, v99
	s_wait_dscnt 0x0
	v_fma_f32 v43, -v100, v98, v43
	ds_bpermute_b32 v98, v82, v99
	s_wait_dscnt 0x0
	v_fma_f32 v44, -v100, v98, v44
	ds_bpermute_b32 v98, v83, v99
	s_wait_dscnt 0x0
	v_fma_f32 v45, -v100, v98, v45
	ds_bpermute_b32 v98, v84, v99
	s_wait_dscnt 0x0
	v_fma_f32 v46, -v100, v98, v46
	ds_bpermute_b32 v98, v85, v99
	s_wait_dscnt 0x0
	v_fma_f32 v47, -v100, v98, v47
	ds_bpermute_b32 v98, v86, v99
	s_wait_dscnt 0x0
	v_fma_f32 v48, -v100, v98, v48
	ds_bpermute_b32 v98, v87, v99
	s_wait_dscnt 0x0
	v_fma_f32 v49, -v100, v98, v49
	ds_bpermute_b32 v98, v88, v99
	s_wait_dscnt 0x0
	v_fma_f32 v50, -v100, v98, v50
	ds_bpermute_b32 v98, v89, v99
	s_wait_dscnt 0x0
	v_fma_f32 v51, -v100, v98, v51
	ds_bpermute_b32 v98, v90, v99
	s_wait_dscnt 0x0
	v_fma_f32 v52, -v100, v98, v52
	ds_bpermute_b32 v98, v91, v99
	s_wait_dscnt 0x0
	v_fma_f32 v53, -v100, v98, v53
	ds_bpermute_b32 v98, v92, v99
	s_wait_dscnt 0x0
	v_fma_f32 v54, -v100, v98, v54
	ds_bpermute_b32 v98, v93, v99
	s_wait_dscnt 0x0
	v_fma_f32 v55, -v100, v98, v55
	ds_bpermute_b32 v98, v94, v99
	s_wait_dscnt 0x0
	v_fma_f32 v56, -v100, v98, v56
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v98, 0, v39, s1
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v98, v98, v55, s2
	ds_bpermute_b32 v99, v79, v98
	s_wait_dscnt 0x0
	v_add_f32_e32 v100, v98, v99
	ds_load_2addr_b32 v[98:99], v101 offset0:128 offset1:160
	s_wait_dscnt 0x0
	ds_bpermute_b32 v102, v96, v98
	s_wait_dscnt 0x0
	v_fma_f32 v25, -v100, v102, v25
	ds_bpermute_b32 v102, v80, v98
	s_wait_dscnt 0x0
	v_fma_f32 v26, -v100, v102, v26
	ds_bpermute_b32 v102, v81, v98
	s_wait_dscnt 0x0
	v_fma_f32 v27, -v100, v102, v27
	ds_bpermute_b32 v102, v82, v98
	s_wait_dscnt 0x0
	v_fma_f32 v28, -v100, v102, v28
	ds_bpermute_b32 v102, v83, v98
	s_wait_dscnt 0x0
	v_fma_f32 v29, -v100, v102, v29
	ds_bpermute_b32 v102, v84, v98
	s_wait_dscnt 0x0
	v_fma_f32 v30, -v100, v102, v30
	ds_bpermute_b32 v102, v85, v98
	s_wait_dscnt 0x0
	v_fma_f32 v31, -v100, v102, v31
	ds_bpermute_b32 v102, v86, v98
	s_wait_dscnt 0x0
	v_fma_f32 v32, -v100, v102, v32
	ds_bpermute_b32 v102, v87, v98
	s_wait_dscnt 0x0
	v_fma_f32 v33, -v100, v102, v33
	ds_bpermute_b32 v102, v88, v98
	s_wait_dscnt 0x0
	v_fma_f32 v34, -v100, v102, v34
	ds_bpermute_b32 v102, v89, v98
	s_wait_dscnt 0x0
	v_fma_f32 v35, -v100, v102, v35
	ds_bpermute_b32 v102, v90, v98
	s_wait_dscnt 0x0
	v_fma_f32 v36, -v100, v102, v36
	ds_bpermute_b32 v102, v91, v98
	s_wait_dscnt 0x0
	v_fma_f32 v37, -v100, v102, v37
	ds_bpermute_b32 v102, v92, v98
	s_wait_dscnt 0x0
	v_fma_f32 v38, -v100, v102, v38
	ds_bpermute_b32 v102, v93, v98
	ds_bpermute_b32 v98, v94, v98
	s_wait_dscnt 0x1
	v_fma_f32 v39, -v100, v102, v39
	s_wait_dscnt 0x0
	v_fma_f32 v40, -v100, v98, v40
	ds_bpermute_b32 v98, v96, v99
	s_wait_dscnt 0x0
	v_fma_f32 v41, -v100, v98, v41
	ds_bpermute_b32 v98, v80, v99
	s_wait_dscnt 0x0
	v_fma_f32 v42, -v100, v98, v42
	ds_bpermute_b32 v98, v81, v99
	s_wait_dscnt 0x0
	v_fma_f32 v43, -v100, v98, v43
	ds_bpermute_b32 v98, v82, v99
	s_wait_dscnt 0x0
	v_fma_f32 v44, -v100, v98, v44
	ds_bpermute_b32 v98, v83, v99
	s_wait_dscnt 0x0
	v_fma_f32 v45, -v100, v98, v45
	ds_bpermute_b32 v98, v84, v99
	s_wait_dscnt 0x0
	v_fma_f32 v46, -v100, v98, v46
	ds_bpermute_b32 v98, v85, v99
	s_wait_dscnt 0x0
	v_fma_f32 v47, -v100, v98, v47
	ds_bpermute_b32 v98, v86, v99
	s_wait_dscnt 0x0
	v_fma_f32 v48, -v100, v98, v48
	ds_bpermute_b32 v98, v87, v99
	s_wait_dscnt 0x0
	v_fma_f32 v49, -v100, v98, v49
	ds_bpermute_b32 v98, v88, v99
	s_wait_dscnt 0x0
	v_fma_f32 v50, -v100, v98, v50
	ds_bpermute_b32 v98, v89, v99
	s_wait_dscnt 0x0
	v_fma_f32 v51, -v100, v98, v51
	ds_bpermute_b32 v98, v90, v99
	s_wait_dscnt 0x0
	v_fma_f32 v52, -v100, v98, v52
	ds_bpermute_b32 v98, v91, v99
	s_wait_dscnt 0x0
	v_fma_f32 v53, -v100, v98, v53
	ds_bpermute_b32 v98, v92, v99
	s_wait_dscnt 0x0
	v_fma_f32 v54, -v100, v98, v54
	ds_bpermute_b32 v98, v93, v99
	s_wait_dscnt 0x0
	v_fma_f32 v55, -v100, v98, v55
	ds_bpermute_b32 v98, v94, v99
	s_wait_dscnt 0x0
	v_fma_f32 v56, -v100, v98, v56
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v98, 0, v40, s1
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v98, v98, v56, s2
	ds_bpermute_b32 v99, v79, v98
	s_wait_dscnt 0x0
	v_add_f32_e32 v100, v98, v99
	ds_load_2addr_b32 v[98:99], v101 offset0:192 offset1:224
	s_wait_dscnt 0x0
	ds_bpermute_b32 v101, v96, v98
	s_wait_dscnt 0x0
	v_fma_f32 v25, -v100, v101, v25
	ds_bpermute_b32 v101, v80, v98
	s_wait_dscnt 0x0
	v_fma_f32 v26, -v100, v101, v26
	ds_bpermute_b32 v101, v81, v98
	s_wait_dscnt 0x0
	v_fma_f32 v27, -v100, v101, v27
	ds_bpermute_b32 v101, v82, v98
	s_wait_dscnt 0x0
	v_fma_f32 v28, -v100, v101, v28
	ds_bpermute_b32 v101, v83, v98
	s_wait_dscnt 0x0
	v_fma_f32 v29, -v100, v101, v29
	ds_bpermute_b32 v101, v84, v98
	s_wait_dscnt 0x0
	v_fma_f32 v30, -v100, v101, v30
	ds_bpermute_b32 v101, v85, v98
	s_wait_dscnt 0x0
	v_fma_f32 v31, -v100, v101, v31
	ds_bpermute_b32 v101, v86, v98
	s_wait_dscnt 0x0
	v_fma_f32 v32, -v100, v101, v32
	ds_bpermute_b32 v101, v87, v98
	s_wait_dscnt 0x0
	v_fma_f32 v33, -v100, v101, v33
	ds_bpermute_b32 v101, v88, v98
	s_wait_dscnt 0x0
	v_fma_f32 v34, -v100, v101, v34
	ds_bpermute_b32 v101, v89, v98
	s_wait_dscnt 0x0
	v_fma_f32 v35, -v100, v101, v35
	ds_bpermute_b32 v101, v90, v98
	s_wait_dscnt 0x0
	v_fma_f32 v36, -v100, v101, v36
	ds_bpermute_b32 v101, v91, v98
	s_wait_dscnt 0x0
	v_fma_f32 v37, -v100, v101, v37
	ds_bpermute_b32 v101, v92, v98
	s_wait_dscnt 0x0
	v_fma_f32 v38, -v100, v101, v38
	ds_bpermute_b32 v101, v93, v98
	ds_bpermute_b32 v98, v94, v98
	s_wait_dscnt 0x1
	v_fma_f32 v39, -v100, v101, v39
	s_wait_dscnt 0x0
	v_fma_f32 v40, -v100, v98, v40
	ds_bpermute_b32 v98, v96, v99
	s_wait_dscnt 0x0
	v_fma_f32 v41, -v100, v98, v41
	ds_bpermute_b32 v98, v80, v99
	s_wait_dscnt 0x0
	v_fma_f32 v42, -v100, v98, v42
	ds_bpermute_b32 v98, v81, v99
	s_wait_dscnt 0x0
	v_fma_f32 v43, -v100, v98, v43
	ds_bpermute_b32 v98, v82, v99
	s_wait_dscnt 0x0
	v_fma_f32 v44, -v100, v98, v44
	ds_bpermute_b32 v98, v83, v99
	s_wait_dscnt 0x0
	v_fma_f32 v45, -v100, v98, v45
	ds_bpermute_b32 v98, v84, v99
	s_wait_dscnt 0x0
	v_fma_f32 v46, -v100, v98, v46
	ds_bpermute_b32 v98, v85, v99
	s_wait_dscnt 0x0
	v_fma_f32 v47, -v100, v98, v47
	ds_bpermute_b32 v98, v86, v99
	s_wait_dscnt 0x0
	v_fma_f32 v48, -v100, v98, v48
	ds_bpermute_b32 v98, v87, v99
	s_wait_dscnt 0x0
	v_fma_f32 v49, -v100, v98, v49
	ds_bpermute_b32 v98, v88, v99
	s_wait_dscnt 0x0
	v_fma_f32 v50, -v100, v98, v50
	ds_bpermute_b32 v98, v89, v99
	s_wait_dscnt 0x0
	v_fma_f32 v51, -v100, v98, v51
	ds_bpermute_b32 v98, v90, v99
	s_wait_dscnt 0x0
	v_fma_f32 v52, -v100, v98, v52
	ds_bpermute_b32 v98, v91, v99
	s_wait_dscnt 0x0
	v_fma_f32 v53, -v100, v98, v53
	ds_bpermute_b32 v98, v92, v99
	s_wait_dscnt 0x0
	v_fma_f32 v54, -v100, v98, v54
	ds_bpermute_b32 v98, v93, v99
	s_wait_dscnt 0x0
	v_fma_f32 v55, -v100, v98, v55
	ds_bpermute_b32 v98, v94, v99
	s_wait_dscnt 0x0
	v_fma_f32 v56, -v100, v98, v56
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e32 v98, 0, v33, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v98, v98, v49, s0
	ds_bpermute_b32 v99, v79, v98
	s_wait_dscnt 0x0
	v_add_f32_e32 v100, v98, v99
	ds_load_2addr_b32 v[98:99], v114 offset1:32
	s_wait_dscnt 0x0
	ds_bpermute_b32 v101, v96, v98
	s_wait_dscnt 0x0
	v_fma_f32 v25, -v100, v101, v25
	ds_bpermute_b32 v101, v80, v98
	s_wait_dscnt 0x0
	v_fma_f32 v26, -v100, v101, v26
	ds_bpermute_b32 v101, v81, v98
	s_wait_dscnt 0x0
	v_fma_f32 v27, -v100, v101, v27
	ds_bpermute_b32 v101, v82, v98
	s_wait_dscnt 0x0
	v_fma_f32 v28, -v100, v101, v28
	ds_bpermute_b32 v101, v83, v98
	s_wait_dscnt 0x0
	v_fma_f32 v29, -v100, v101, v29
	ds_bpermute_b32 v101, v84, v98
	s_wait_dscnt 0x0
	v_fma_f32 v30, -v100, v101, v30
	ds_bpermute_b32 v101, v85, v98
	s_wait_dscnt 0x0
	v_fma_f32 v31, -v100, v101, v31
	ds_bpermute_b32 v101, v86, v98
	s_wait_dscnt 0x0
	v_fma_f32 v32, -v100, v101, v32
	ds_bpermute_b32 v101, v87, v98
	s_wait_dscnt 0x0
	v_fma_f32 v33, -v100, v101, v33
	ds_bpermute_b32 v101, v88, v98
	s_wait_dscnt 0x0
	v_fma_f32 v34, -v100, v101, v34
	ds_bpermute_b32 v101, v89, v98
	s_wait_dscnt 0x0
	v_fma_f32 v35, -v100, v101, v35
	ds_bpermute_b32 v101, v90, v98
	s_wait_dscnt 0x0
	v_fma_f32 v36, -v100, v101, v36
	ds_bpermute_b32 v101, v91, v98
	s_wait_dscnt 0x0
	v_fma_f32 v37, -v100, v101, v37
	ds_bpermute_b32 v101, v92, v98
	s_wait_dscnt 0x0
	v_fma_f32 v38, -v100, v101, v38
	ds_bpermute_b32 v101, v93, v98
	ds_bpermute_b32 v98, v94, v98
	s_wait_dscnt 0x1
	v_fma_f32 v39, -v100, v101, v39
	s_wait_dscnt 0x0
	v_fma_f32 v40, -v100, v98, v40
	ds_bpermute_b32 v98, v96, v99
	s_wait_dscnt 0x0
	v_fma_f32 v41, -v100, v98, v41
	ds_bpermute_b32 v98, v80, v99
	s_wait_dscnt 0x0
	v_fma_f32 v42, -v100, v98, v42
	ds_bpermute_b32 v98, v81, v99
	s_wait_dscnt 0x0
	v_fma_f32 v43, -v100, v98, v43
	ds_bpermute_b32 v98, v82, v99
	s_wait_dscnt 0x0
	v_fma_f32 v44, -v100, v98, v44
	ds_bpermute_b32 v98, v83, v99
	s_wait_dscnt 0x0
	v_fma_f32 v45, -v100, v98, v45
	ds_bpermute_b32 v98, v84, v99
	s_wait_dscnt 0x0
	v_fma_f32 v46, -v100, v98, v46
	ds_bpermute_b32 v98, v85, v99
	s_wait_dscnt 0x0
	v_fma_f32 v47, -v100, v98, v47
	ds_bpermute_b32 v98, v86, v99
	s_wait_dscnt 0x0
	v_fma_f32 v48, -v100, v98, v48
	ds_bpermute_b32 v98, v87, v99
	s_wait_dscnt 0x0
	v_fma_f32 v49, -v100, v98, v49
	ds_bpermute_b32 v98, v88, v99
	s_wait_dscnt 0x0
	v_fma_f32 v50, -v100, v98, v50
	ds_bpermute_b32 v98, v89, v99
	s_wait_dscnt 0x0
	v_fma_f32 v51, -v100, v98, v51
	ds_bpermute_b32 v98, v90, v99
	s_wait_dscnt 0x0
	v_fma_f32 v52, -v100, v98, v52
	ds_bpermute_b32 v98, v91, v99
	s_wait_dscnt 0x0
	v_fma_f32 v53, -v100, v98, v53
	ds_bpermute_b32 v98, v92, v99
	s_wait_dscnt 0x0
	v_fma_f32 v54, -v100, v98, v54
	ds_bpermute_b32 v98, v93, v99
	s_wait_dscnt 0x0
	v_fma_f32 v55, -v100, v98, v55
	ds_bpermute_b32 v98, v94, v99
	s_wait_dscnt 0x0
	v_fma_f32 v56, -v100, v98, v56
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e32 v98, 0, v34, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v98, v98, v50, s0
	ds_bpermute_b32 v99, v79, v98
	s_wait_dscnt 0x0
	v_add_f32_e32 v100, v98, v99
	ds_load_2addr_b32 v[98:99], v114 offset0:64 offset1:96
	s_wait_dscnt 0x0
	ds_bpermute_b32 v101, v96, v98
	s_wait_dscnt 0x0
	v_fma_f32 v25, -v100, v101, v25
	ds_bpermute_b32 v101, v80, v98
	s_wait_dscnt 0x0
	v_fma_f32 v26, -v100, v101, v26
	ds_bpermute_b32 v101, v81, v98
	s_wait_dscnt 0x0
	v_fma_f32 v27, -v100, v101, v27
	ds_bpermute_b32 v101, v82, v98
	s_wait_dscnt 0x0
	v_fma_f32 v28, -v100, v101, v28
	ds_bpermute_b32 v101, v83, v98
	s_wait_dscnt 0x0
	v_fma_f32 v29, -v100, v101, v29
	ds_bpermute_b32 v101, v84, v98
	s_wait_dscnt 0x0
	v_fma_f32 v30, -v100, v101, v30
	ds_bpermute_b32 v101, v85, v98
	s_wait_dscnt 0x0
	v_fma_f32 v31, -v100, v101, v31
	ds_bpermute_b32 v101, v86, v98
	s_wait_dscnt 0x0
	v_fma_f32 v32, -v100, v101, v32
	ds_bpermute_b32 v101, v87, v98
	s_wait_dscnt 0x0
	v_fma_f32 v33, -v100, v101, v33
	ds_bpermute_b32 v101, v88, v98
	s_wait_dscnt 0x0
	v_fma_f32 v34, -v100, v101, v34
	ds_bpermute_b32 v101, v89, v98
	s_wait_dscnt 0x0
	v_fma_f32 v35, -v100, v101, v35
	ds_bpermute_b32 v101, v90, v98
	s_wait_dscnt 0x0
	v_fma_f32 v36, -v100, v101, v36
	ds_bpermute_b32 v101, v91, v98
	s_wait_dscnt 0x0
	v_fma_f32 v37, -v100, v101, v37
	ds_bpermute_b32 v101, v92, v98
	s_wait_dscnt 0x0
	v_fma_f32 v38, -v100, v101, v38
	ds_bpermute_b32 v101, v93, v98
	ds_bpermute_b32 v98, v94, v98
	s_wait_dscnt 0x1
	v_fma_f32 v39, -v100, v101, v39
	s_wait_dscnt 0x0
	v_fma_f32 v40, -v100, v98, v40
	ds_bpermute_b32 v98, v96, v99
	s_wait_dscnt 0x0
	v_fma_f32 v41, -v100, v98, v41
	ds_bpermute_b32 v98, v80, v99
	s_wait_dscnt 0x0
	v_fma_f32 v42, -v100, v98, v42
	ds_bpermute_b32 v98, v81, v99
	s_wait_dscnt 0x0
	v_fma_f32 v43, -v100, v98, v43
	ds_bpermute_b32 v98, v82, v99
	s_wait_dscnt 0x0
	v_fma_f32 v44, -v100, v98, v44
	ds_bpermute_b32 v98, v83, v99
	s_wait_dscnt 0x0
	v_fma_f32 v45, -v100, v98, v45
	ds_bpermute_b32 v98, v84, v99
	s_wait_dscnt 0x0
	v_fma_f32 v46, -v100, v98, v46
	ds_bpermute_b32 v98, v85, v99
	s_wait_dscnt 0x0
	v_fma_f32 v47, -v100, v98, v47
	ds_bpermute_b32 v98, v86, v99
	s_wait_dscnt 0x0
	v_fma_f32 v48, -v100, v98, v48
	ds_bpermute_b32 v98, v87, v99
	s_wait_dscnt 0x0
	v_fma_f32 v49, -v100, v98, v49
	ds_bpermute_b32 v98, v88, v99
	s_wait_dscnt 0x0
	v_fma_f32 v50, -v100, v98, v50
	ds_bpermute_b32 v98, v89, v99
	s_wait_dscnt 0x0
	v_fma_f32 v51, -v100, v98, v51
	ds_bpermute_b32 v98, v90, v99
	s_wait_dscnt 0x0
	v_fma_f32 v52, -v100, v98, v52
	ds_bpermute_b32 v98, v91, v99
	s_wait_dscnt 0x0
	v_fma_f32 v53, -v100, v98, v53
	ds_bpermute_b32 v98, v92, v99
	s_wait_dscnt 0x0
	v_fma_f32 v54, -v100, v98, v54
	ds_bpermute_b32 v98, v93, v99
	s_wait_dscnt 0x0
	v_fma_f32 v55, -v100, v98, v55
	ds_bpermute_b32 v98, v94, v99
	s_wait_dscnt 0x0
	v_fma_f32 v56, -v100, v98, v56
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e32 v98, 0, v35, vcc_lo
	ds_load_2addr_b32 v[112:113], v114 offset0:128 offset1:160
	v_cndmask_b32_e64 v98, v98, v51, s0
	ds_bpermute_b32 v99, v79, v98
	s_wait_dscnt 0x0
	v_add_f32_e32 v115, v98, v99
	ds_bpermute_b32 v98, v96, v112
	s_wait_dscnt 0x0
	v_fma_f32 v25, -v115, v98, v25
	ds_bpermute_b32 v98, v80, v112
	s_wait_dscnt 0x0
	v_fma_f32 v26, -v115, v98, v26
	ds_bpermute_b32 v98, v81, v112
	s_wait_dscnt 0x0
	v_fma_f32 v27, -v115, v98, v27
	ds_bpermute_b32 v98, v82, v112
	s_wait_dscnt 0x0
	v_fma_f32 v28, -v115, v98, v28
	ds_bpermute_b32 v98, v83, v112
	s_wait_dscnt 0x0
	v_fma_f32 v29, -v115, v98, v29
	ds_bpermute_b32 v98, v84, v112
	s_wait_dscnt 0x0
	v_fma_f32 v30, -v115, v98, v30
	ds_bpermute_b32 v98, v85, v112
	s_wait_dscnt 0x0
	v_fma_f32 v31, -v115, v98, v31
	ds_bpermute_b32 v98, v86, v112
	s_wait_dscnt 0x0
	v_fma_f32 v32, -v115, v98, v32
	ds_bpermute_b32 v98, v87, v112
	s_wait_dscnt 0x0
	v_fma_f32 v33, -v115, v98, v33
	ds_bpermute_b32 v98, v88, v112
	s_wait_dscnt 0x0
	v_fma_f32 v34, -v115, v98, v34
	ds_bpermute_b32 v98, v89, v112
	s_wait_dscnt 0x0
	v_fma_f32 v35, -v115, v98, v35
	ds_bpermute_b32 v98, v90, v112
	s_wait_dscnt 0x0
	v_fma_f32 v36, -v115, v98, v36
	ds_bpermute_b32 v98, v91, v112
	s_wait_dscnt 0x0
	v_fma_f32 v37, -v115, v98, v37
	ds_bpermute_b32 v98, v92, v112
	s_wait_dscnt 0x0
	v_fma_f32 v38, -v115, v98, v38
	ds_bpermute_b32 v98, v93, v112
	s_wait_dscnt 0x0
	v_fma_f32 v39, -v115, v98, v39
	ds_bpermute_b32 v98, v94, v112
	s_wait_dscnt 0x0
	v_fma_f32 v40, -v115, v98, v40
	ds_bpermute_b32 v98, v96, v113
	s_wait_dscnt 0x0
	v_fma_f32 v98, -v115, v98, v41
	ds_bpermute_b32 v41, v80, v113
	s_wait_dscnt 0x0
	v_fma_f32 v99, -v115, v41, v42
	ds_bpermute_b32 v41, v81, v113
	s_wait_dscnt 0x0
	v_fma_f32 v100, -v115, v41, v43
	ds_bpermute_b32 v41, v82, v113
	s_wait_dscnt 0x0
	v_fma_f32 v101, -v115, v41, v44
	ds_bpermute_b32 v41, v83, v113
	s_wait_dscnt 0x0
	v_fma_f32 v102, -v115, v41, v45
	ds_bpermute_b32 v41, v84, v113
	s_wait_dscnt 0x0
	v_fma_f32 v103, -v115, v41, v46
	ds_bpermute_b32 v41, v85, v113
	s_wait_dscnt 0x0
	v_fma_f32 v104, -v115, v41, v47
	ds_bpermute_b32 v41, v86, v113
	s_wait_dscnt 0x0
	v_fma_f32 v105, -v115, v41, v48
	ds_bpermute_b32 v41, v87, v113
	s_wait_dscnt 0x0
	v_fma_f32 v106, -v115, v41, v49
	ds_bpermute_b32 v41, v88, v113
	s_wait_dscnt 0x0
	v_fma_f32 v107, -v115, v41, v50
	ds_bpermute_b32 v41, v89, v113
	s_wait_dscnt 0x0
	v_fma_f32 v108, -v115, v41, v51
	ds_bpermute_b32 v41, v90, v113
	s_wait_dscnt 0x0
	v_fma_f32 v109, -v115, v41, v52
	ds_bpermute_b32 v41, v91, v113
	s_wait_dscnt 0x0
	v_fma_f32 v110, -v115, v41, v53
	ds_bpermute_b32 v41, v92, v113
	s_wait_dscnt 0x0
	v_fma_f32 v111, -v115, v41, v54
	ds_bpermute_b32 v41, v93, v113
	s_wait_dscnt 0x0
	v_fma_f32 v112, -v115, v41, v55
	ds_bpermute_b32 v41, v94, v113
	s_wait_dscnt 0x0
	v_fma_f32 v113, -v115, v41, v56
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e32 v41, 0, v36, vcc_lo
	ds_load_2addr_b32 v[114:115], v114 offset0:192 offset1:224
	v_cndmask_b32_e64 v41, v41, v109, s0
	ds_bpermute_b32 v42, v79, v41
	s_wait_dscnt 0x0
	v_add_f32_e32 v116, v41, v42
	ds_bpermute_b32 v41, v96, v114
	s_wait_dscnt 0x0
	v_fma_f32 v49, -v116, v41, v25
	ds_bpermute_b32 v25, v80, v114
	s_wait_dscnt 0x0
	v_fma_f32 v50, -v116, v25, v26
	ds_bpermute_b32 v25, v81, v114
	ds_bpermute_b32 v26, v88, v115
	s_wait_dscnt 0x1
	v_fma_f32 v51, -v116, v25, v27
	ds_bpermute_b32 v25, v82, v114
	ds_bpermute_b32 v27, v89, v115
	s_wait_dscnt 0x2
	v_fma_f32 v26, -v116, v26, v107
	s_wait_dscnt 0x1
	v_fma_f32 v52, -v116, v25, v28
	ds_bpermute_b32 v25, v83, v114
	ds_bpermute_b32 v28, v90, v115
	s_wait_dscnt 0x2
	v_fma_f32 v27, -v116, v27, v108
	s_wait_dscnt 0x1
	v_fma_f32 v53, -v116, v25, v29
	ds_bpermute_b32 v25, v84, v114
	ds_bpermute_b32 v29, v91, v115
	s_wait_dscnt 0x2
	v_fma_f32 v28, -v116, v28, v109
	s_wait_dscnt 0x1
	v_fma_f32 v54, -v116, v25, v30
	ds_bpermute_b32 v25, v85, v114
	ds_bpermute_b32 v30, v92, v115
	s_wait_dscnt 0x2
	v_fma_f32 v29, -v116, v29, v110
	s_wait_dscnt 0x1
	v_fma_f32 v55, -v116, v25, v31
	ds_bpermute_b32 v25, v86, v114
	ds_bpermute_b32 v31, v93, v115
	s_wait_dscnt 0x2
	v_fma_f32 v30, -v116, v30, v111
	s_wait_dscnt 0x1
	v_fma_f32 v56, -v116, v25, v32
	ds_bpermute_b32 v25, v87, v114
	ds_bpermute_b32 v32, v94, v115
	s_wait_dscnt 0x2
	v_fma_f32 v31, -v116, v31, v112
	s_wait_dscnt 0x1
	v_fma_f32 v41, -v116, v25, v33
	ds_bpermute_b32 v25, v88, v114
	s_wait_dscnt 0x1
	v_fma_f32 v32, -v116, v32, v113
	s_wait_dscnt 0x0
	v_fma_f32 v42, -v116, v25, v34
	ds_bpermute_b32 v25, v89, v114
	s_wait_dscnt 0x0
	v_fma_f32 v43, -v116, v25, v35
	ds_bpermute_b32 v25, v90, v114
	s_wait_dscnt 0x0
	v_fma_f32 v44, -v116, v25, v36
	ds_bpermute_b32 v25, v91, v114
	s_wait_dscnt 0x0
	v_fma_f32 v45, -v116, v25, v37
	ds_bpermute_b32 v25, v92, v114
	s_wait_dscnt 0x0
	v_fma_f32 v46, -v116, v25, v38
	ds_bpermute_b32 v25, v93, v114
	s_wait_dscnt 0x0
	v_fma_f32 v47, -v116, v25, v39
	ds_bpermute_b32 v25, v94, v114
	s_wait_dscnt 0x0
	v_fma_f32 v48, -v116, v25, v40
	ds_bpermute_b32 v25, v96, v115
	s_wait_dscnt 0x0
	v_fma_f32 v33, -v116, v25, v98
	ds_bpermute_b32 v25, v80, v115
	s_wait_dscnt 0x0
	v_fma_f32 v34, -v116, v25, v99
	ds_bpermute_b32 v25, v81, v115
	s_wait_dscnt 0x0
	v_fma_f32 v35, -v116, v25, v100
	ds_bpermute_b32 v25, v82, v115
	s_wait_dscnt 0x0
	v_fma_f32 v36, -v116, v25, v101
	ds_bpermute_b32 v25, v83, v115
	s_wait_dscnt 0x0
	v_fma_f32 v37, -v116, v25, v102
	ds_bpermute_b32 v25, v84, v115
	s_wait_dscnt 0x0
	v_fma_f32 v38, -v116, v25, v103
	ds_bpermute_b32 v25, v85, v115
	s_wait_dscnt 0x0
	v_fma_f32 v39, -v116, v25, v104
	ds_bpermute_b32 v25, v86, v115
	s_wait_dscnt 0x0
	v_fma_f32 v40, -v116, v25, v105
	ds_bpermute_b32 v25, v87, v115
	s_wait_dscnt 0x0
	v_fma_f32 v25, -v116, v25, v106
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e32 v98, 0, v45, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v98, v98, v29, s0
	ds_bpermute_b32 v99, v79, v98
	s_wait_dscnt 0x0
	v_add_f32_e32 v98, v98, v99
	ds_load_2addr_b32 v[99:100], v97 offset1:32
	s_wait_dscnt 0x0
	ds_bpermute_b32 v101, v96, v99
	s_wait_dscnt 0x0
	v_fma_f32 v49, -v98, v101, v49
	ds_bpermute_b32 v101, v80, v99
	s_wait_dscnt 0x0
	v_fma_f32 v50, -v98, v101, v50
	ds_bpermute_b32 v101, v81, v99
	s_wait_dscnt 0x0
	v_fma_f32 v51, -v98, v101, v51
	ds_bpermute_b32 v101, v82, v99
	s_wait_dscnt 0x0
	v_fma_f32 v52, -v98, v101, v52
	ds_bpermute_b32 v101, v83, v99
	s_wait_dscnt 0x0
	v_fma_f32 v53, -v98, v101, v53
	ds_bpermute_b32 v101, v84, v99
	s_wait_dscnt 0x0
	v_fma_f32 v54, -v98, v101, v54
	ds_bpermute_b32 v101, v85, v99
	s_wait_dscnt 0x0
	v_fma_f32 v55, -v98, v101, v55
	ds_bpermute_b32 v101, v86, v99
	s_wait_dscnt 0x0
	v_fma_f32 v56, -v98, v101, v56
	ds_bpermute_b32 v101, v87, v99
	s_wait_dscnt 0x0
	v_fma_f32 v41, -v98, v101, v41
	ds_bpermute_b32 v101, v88, v99
	s_wait_dscnt 0x0
	v_fma_f32 v42, -v98, v101, v42
	ds_bpermute_b32 v101, v89, v99
	s_wait_dscnt 0x0
	v_fma_f32 v43, -v98, v101, v43
	ds_bpermute_b32 v101, v90, v99
	s_wait_dscnt 0x0
	v_fma_f32 v44, -v98, v101, v44
	ds_bpermute_b32 v101, v91, v99
	s_wait_dscnt 0x0
	v_fma_f32 v45, -v98, v101, v45
	ds_bpermute_b32 v101, v92, v99
	s_wait_dscnt 0x0
	v_fma_f32 v46, -v98, v101, v46
	ds_bpermute_b32 v101, v93, v99
	ds_bpermute_b32 v99, v94, v99
	s_wait_dscnt 0x1
	v_fma_f32 v47, -v98, v101, v47
	s_wait_dscnt 0x0
	v_fma_f32 v48, -v98, v99, v48
	ds_bpermute_b32 v99, v96, v100
	s_wait_dscnt 0x0
	v_fma_f32 v33, -v98, v99, v33
	ds_bpermute_b32 v99, v80, v100
	s_wait_dscnt 0x0
	v_fma_f32 v34, -v98, v99, v34
	ds_bpermute_b32 v99, v81, v100
	s_wait_dscnt 0x0
	v_fma_f32 v35, -v98, v99, v35
	ds_bpermute_b32 v99, v82, v100
	s_wait_dscnt 0x0
	v_fma_f32 v36, -v98, v99, v36
	ds_bpermute_b32 v99, v83, v100
	s_wait_dscnt 0x0
	v_fma_f32 v37, -v98, v99, v37
	ds_bpermute_b32 v99, v84, v100
	s_wait_dscnt 0x0
	v_fma_f32 v38, -v98, v99, v38
	ds_bpermute_b32 v99, v85, v100
	s_wait_dscnt 0x0
	v_fma_f32 v39, -v98, v99, v39
	ds_bpermute_b32 v99, v86, v100
	s_wait_dscnt 0x0
	v_fma_f32 v40, -v98, v99, v40
	ds_bpermute_b32 v99, v87, v100
	s_wait_dscnt 0x0
	v_fma_f32 v25, -v98, v99, v25
	ds_bpermute_b32 v99, v88, v100
	s_wait_dscnt 0x0
	v_fma_f32 v26, -v98, v99, v26
	ds_bpermute_b32 v99, v89, v100
	s_wait_dscnt 0x0
	v_fma_f32 v27, -v98, v99, v27
	ds_bpermute_b32 v99, v90, v100
	s_wait_dscnt 0x0
	v_fma_f32 v28, -v98, v99, v28
	ds_bpermute_b32 v99, v91, v100
	s_wait_dscnt 0x0
	v_fma_f32 v29, -v98, v99, v29
	ds_bpermute_b32 v99, v92, v100
	s_wait_dscnt 0x0
	v_fma_f32 v30, -v98, v99, v30
	ds_bpermute_b32 v99, v93, v100
	s_wait_dscnt 0x0
	v_fma_f32 v31, -v98, v99, v31
	ds_bpermute_b32 v99, v94, v100
	s_wait_dscnt 0x0
	v_fma_f32 v32, -v98, v99, v32
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e32 v98, 0, v46, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v98, v98, v30, s0
	ds_bpermute_b32 v99, v79, v98
	s_wait_dscnt 0x0
	v_add_f32_e32 v100, v98, v99
	ds_load_2addr_b32 v[98:99], v97 offset0:64 offset1:96
	s_wait_dscnt 0x0
	ds_bpermute_b32 v101, v96, v98
	s_wait_dscnt 0x0
	v_fma_f32 v49, -v100, v101, v49
	ds_bpermute_b32 v101, v80, v98
	s_wait_dscnt 0x0
	v_fma_f32 v50, -v100, v101, v50
	ds_bpermute_b32 v101, v81, v98
	s_wait_dscnt 0x0
	v_fma_f32 v51, -v100, v101, v51
	ds_bpermute_b32 v101, v82, v98
	s_wait_dscnt 0x0
	v_fma_f32 v52, -v100, v101, v52
	ds_bpermute_b32 v101, v83, v98
	s_wait_dscnt 0x0
	v_fma_f32 v53, -v100, v101, v53
	ds_bpermute_b32 v101, v84, v98
	s_wait_dscnt 0x0
	v_fma_f32 v54, -v100, v101, v54
	ds_bpermute_b32 v101, v85, v98
	s_wait_dscnt 0x0
	v_fma_f32 v55, -v100, v101, v55
	ds_bpermute_b32 v101, v86, v98
	s_wait_dscnt 0x0
	v_fma_f32 v56, -v100, v101, v56
	ds_bpermute_b32 v101, v87, v98
	s_wait_dscnt 0x0
	v_fma_f32 v41, -v100, v101, v41
	ds_bpermute_b32 v101, v88, v98
	s_wait_dscnt 0x0
	v_fma_f32 v42, -v100, v101, v42
	ds_bpermute_b32 v101, v89, v98
	s_wait_dscnt 0x0
	v_fma_f32 v43, -v100, v101, v43
	ds_bpermute_b32 v101, v90, v98
	s_wait_dscnt 0x0
	v_fma_f32 v44, -v100, v101, v44
	ds_bpermute_b32 v101, v91, v98
	s_wait_dscnt 0x0
	v_fma_f32 v45, -v100, v101, v45
	ds_bpermute_b32 v101, v92, v98
	s_wait_dscnt 0x0
	v_fma_f32 v46, -v100, v101, v46
	ds_bpermute_b32 v101, v93, v98
	ds_bpermute_b32 v98, v94, v98
	s_wait_dscnt 0x1
	v_fma_f32 v47, -v100, v101, v47
	s_wait_dscnt 0x0
	v_fma_f32 v48, -v100, v98, v48
	ds_bpermute_b32 v98, v96, v99
	s_wait_dscnt 0x0
	v_fma_f32 v33, -v100, v98, v33
	ds_bpermute_b32 v98, v80, v99
	s_wait_dscnt 0x0
	v_fma_f32 v34, -v100, v98, v34
	ds_bpermute_b32 v98, v81, v99
	s_wait_dscnt 0x0
	v_fma_f32 v35, -v100, v98, v35
	ds_bpermute_b32 v98, v82, v99
	s_wait_dscnt 0x0
	v_fma_f32 v36, -v100, v98, v36
	ds_bpermute_b32 v98, v83, v99
	s_wait_dscnt 0x0
	v_fma_f32 v37, -v100, v98, v37
	ds_bpermute_b32 v98, v84, v99
	s_wait_dscnt 0x0
	v_fma_f32 v38, -v100, v98, v38
	ds_bpermute_b32 v98, v85, v99
	s_wait_dscnt 0x0
	v_fma_f32 v39, -v100, v98, v39
	ds_bpermute_b32 v98, v86, v99
	s_wait_dscnt 0x0
	v_fma_f32 v40, -v100, v98, v40
	ds_bpermute_b32 v98, v87, v99
	s_wait_dscnt 0x0
	v_fma_f32 v25, -v100, v98, v25
	ds_bpermute_b32 v98, v88, v99
	s_wait_dscnt 0x0
	v_fma_f32 v26, -v100, v98, v26
	ds_bpermute_b32 v98, v89, v99
	s_wait_dscnt 0x0
	v_fma_f32 v27, -v100, v98, v27
	ds_bpermute_b32 v98, v90, v99
	s_wait_dscnt 0x0
	v_fma_f32 v28, -v100, v98, v28
	ds_bpermute_b32 v98, v91, v99
	s_wait_dscnt 0x0
	v_fma_f32 v29, -v100, v98, v29
	ds_bpermute_b32 v98, v92, v99
	s_wait_dscnt 0x0
	v_fma_f32 v30, -v100, v98, v30
	ds_bpermute_b32 v98, v93, v99
	s_wait_dscnt 0x0
	v_fma_f32 v31, -v100, v98, v31
	ds_bpermute_b32 v98, v94, v99
	s_wait_dscnt 0x0
	v_fma_f32 v32, -v100, v98, v32
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e32 v98, 0, v47, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v98, v98, v31, s0
	ds_bpermute_b32 v99, v79, v98
	s_wait_dscnt 0x0
	v_add_f32_e32 v100, v98, v99
	ds_load_2addr_b32 v[98:99], v97 offset0:128 offset1:160
	s_wait_dscnt 0x0
	ds_bpermute_b32 v101, v96, v98
	s_wait_dscnt 0x0
	v_fma_f32 v49, -v100, v101, v49
	ds_bpermute_b32 v101, v80, v98
	s_wait_dscnt 0x0
	v_fma_f32 v50, -v100, v101, v50
	ds_bpermute_b32 v101, v81, v98
	s_wait_dscnt 0x0
	v_fma_f32 v51, -v100, v101, v51
	ds_bpermute_b32 v101, v82, v98
	s_wait_dscnt 0x0
	v_fma_f32 v52, -v100, v101, v52
	ds_bpermute_b32 v101, v83, v98
	s_wait_dscnt 0x0
	v_fma_f32 v53, -v100, v101, v53
	ds_bpermute_b32 v101, v84, v98
	s_wait_dscnt 0x0
	v_fma_f32 v54, -v100, v101, v54
	ds_bpermute_b32 v101, v85, v98
	s_wait_dscnt 0x0
	v_fma_f32 v55, -v100, v101, v55
	ds_bpermute_b32 v101, v86, v98
	s_wait_dscnt 0x0
	v_fma_f32 v56, -v100, v101, v56
	ds_bpermute_b32 v101, v87, v98
	s_wait_dscnt 0x0
	v_fma_f32 v41, -v100, v101, v41
	ds_bpermute_b32 v101, v88, v98
	s_wait_dscnt 0x0
	v_fma_f32 v42, -v100, v101, v42
	ds_bpermute_b32 v101, v89, v98
	s_wait_dscnt 0x0
	v_fma_f32 v43, -v100, v101, v43
	ds_bpermute_b32 v101, v90, v98
	s_wait_dscnt 0x0
	v_fma_f32 v44, -v100, v101, v44
	ds_bpermute_b32 v101, v91, v98
	s_wait_dscnt 0x0
	v_fma_f32 v45, -v100, v101, v45
	ds_bpermute_b32 v101, v92, v98
	s_wait_dscnt 0x0
	v_fma_f32 v46, -v100, v101, v46
	ds_bpermute_b32 v101, v93, v98
	ds_bpermute_b32 v98, v94, v98
	s_wait_dscnt 0x1
	v_fma_f32 v47, -v100, v101, v47
	s_wait_dscnt 0x0
	v_fma_f32 v48, -v100, v98, v48
	ds_bpermute_b32 v98, v96, v99
	s_wait_dscnt 0x0
	v_fma_f32 v33, -v100, v98, v33
	ds_bpermute_b32 v98, v80, v99
	s_wait_dscnt 0x0
	v_fma_f32 v34, -v100, v98, v34
	ds_bpermute_b32 v98, v81, v99
	s_wait_dscnt 0x0
	v_fma_f32 v35, -v100, v98, v35
	ds_bpermute_b32 v98, v82, v99
	s_wait_dscnt 0x0
	v_fma_f32 v36, -v100, v98, v36
	ds_bpermute_b32 v98, v83, v99
	s_wait_dscnt 0x0
	v_fma_f32 v37, -v100, v98, v37
	ds_bpermute_b32 v98, v84, v99
	s_wait_dscnt 0x0
	v_fma_f32 v38, -v100, v98, v38
	ds_bpermute_b32 v98, v85, v99
	s_wait_dscnt 0x0
	v_fma_f32 v39, -v100, v98, v39
	ds_bpermute_b32 v98, v86, v99
	s_wait_dscnt 0x0
	v_fma_f32 v40, -v100, v98, v40
	ds_bpermute_b32 v98, v87, v99
	s_wait_dscnt 0x0
	v_fma_f32 v25, -v100, v98, v25
	ds_bpermute_b32 v98, v88, v99
	s_wait_dscnt 0x0
	v_fma_f32 v26, -v100, v98, v26
	ds_bpermute_b32 v98, v89, v99
	s_wait_dscnt 0x0
	v_fma_f32 v27, -v100, v98, v27
	ds_bpermute_b32 v98, v90, v99
	s_wait_dscnt 0x0
	v_fma_f32 v28, -v100, v98, v28
	ds_bpermute_b32 v98, v91, v99
	s_wait_dscnt 0x0
	v_fma_f32 v29, -v100, v98, v29
	ds_bpermute_b32 v98, v92, v99
	s_wait_dscnt 0x0
	v_fma_f32 v30, -v100, v98, v30
	ds_bpermute_b32 v98, v93, v99
	s_wait_dscnt 0x0
	v_fma_f32 v31, -v100, v98, v31
	ds_bpermute_b32 v98, v94, v99
	s_wait_dscnt 0x0
	v_fma_f32 v32, -v100, v98, v32
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e32 v98, 0, v48, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v98, v98, v32, s0
	ds_bpermute_b32 v99, v79, v98
	s_wait_dscnt 0x0
	v_add_f32_e32 v99, v98, v99
	ds_load_2addr_b32 v[97:98], v97 offset0:192 offset1:224
	s_wait_dscnt 0x0
	ds_bpermute_b32 v100, v96, v97
	ds_bpermute_b32 v96, v96, v98
	s_wait_dscnt 0x1
	v_fma_f32 v49, -v99, v100, v49
	s_wait_dscnt 0x0
	v_fma_f32 v33, -v99, v96, v33
	ds_bpermute_b32 v96, v80, v98
	ds_bpermute_b32 v100, v80, v97
	s_wait_dscnt 0x1
	v_fma_f32 v34, -v99, v96, v34
	ds_bpermute_b32 v96, v81, v98
	s_wait_dscnt 0x1
	v_fma_f32 v50, -v99, v100, v50
	ds_bpermute_b32 v100, v81, v97
	s_wait_dscnt 0x1
	v_fma_f32 v35, -v99, v96, v35
	ds_bpermute_b32 v96, v82, v98
	s_wait_dscnt 0x1
	v_fma_f32 v51, -v99, v100, v51
	ds_bpermute_b32 v100, v82, v97
	s_wait_dscnt 0x1
	v_fma_f32 v36, -v99, v96, v36
	ds_bpermute_b32 v96, v83, v98
	s_wait_dscnt 0x1
	v_fma_f32 v52, -v99, v100, v52
	ds_bpermute_b32 v100, v83, v97
	s_wait_dscnt 0x1
	v_fma_f32 v37, -v99, v96, v37
	ds_bpermute_b32 v96, v84, v98
	s_wait_dscnt 0x1
	v_fma_f32 v53, -v99, v100, v53
	ds_bpermute_b32 v100, v84, v97
	s_wait_dscnt 0x1
	v_fma_f32 v38, -v99, v96, v38
	ds_bpermute_b32 v96, v85, v98
	s_wait_dscnt 0x1
	v_fma_f32 v54, -v99, v100, v54
	ds_bpermute_b32 v100, v85, v97
	s_wait_dscnt 0x1
	v_fma_f32 v39, -v99, v96, v39
	ds_bpermute_b32 v96, v86, v98
	s_wait_dscnt 0x1
	v_fma_f32 v55, -v99, v100, v55
	ds_bpermute_b32 v100, v86, v97
	s_wait_dscnt 0x1
	v_fma_f32 v40, -v99, v96, v40
	ds_bpermute_b32 v96, v87, v98
	s_wait_dscnt 0x1
	v_fma_f32 v56, -v99, v100, v56
	ds_bpermute_b32 v100, v87, v97
	s_wait_dscnt 0x1
	v_fma_f32 v25, -v99, v96, v25
	ds_bpermute_b32 v96, v88, v98
	s_wait_dscnt 0x1
	v_fma_f32 v41, -v99, v100, v41
	ds_bpermute_b32 v100, v88, v97
	s_wait_dscnt 0x1
	v_fma_f32 v26, -v99, v96, v26
	ds_bpermute_b32 v96, v89, v98
	s_wait_dscnt 0x1
	v_fma_f32 v42, -v99, v100, v42
	ds_bpermute_b32 v100, v89, v97
	s_wait_dscnt 0x1
	v_fma_f32 v27, -v99, v96, v27
	ds_bpermute_b32 v96, v90, v98
	s_wait_dscnt 0x1
	v_fma_f32 v43, -v99, v100, v43
	ds_bpermute_b32 v100, v90, v97
	s_wait_dscnt 0x1
	v_fma_f32 v28, -v99, v96, v28
	ds_bpermute_b32 v96, v91, v98
	s_wait_dscnt 0x1
	v_fma_f32 v44, -v99, v100, v44
	ds_bpermute_b32 v100, v91, v97
	s_wait_dscnt 0x1
	v_fma_f32 v29, -v99, v96, v29
	ds_bpermute_b32 v96, v92, v98
	s_wait_dscnt 0x1
	v_fma_f32 v45, -v99, v100, v45
	ds_bpermute_b32 v100, v92, v97
	s_wait_dscnt 0x1
	v_fma_f32 v30, -v99, v96, v30
	ds_bpermute_b32 v96, v93, v98
	s_wait_dscnt 0x1
	v_fma_f32 v46, -v99, v100, v46
	ds_bpermute_b32 v100, v93, v97
	ds_bpermute_b32 v97, v94, v97
	s_wait_dscnt 0x2
	v_fma_f32 v31, -v99, v96, v31
	ds_bpermute_b32 v96, v94, v98
	s_wait_dscnt 0x2
	v_fma_f32 v47, -v99, v100, v47
	s_wait_dscnt 0x1
	v_fma_f32 v48, -v99, v97, v48
	s_wait_dscnt 0x0
	v_fma_f32 v32, -v99, v96, v32
	;;#ASMSTART
	;;#ASMEND
	s_cbranch_scc0 .LBB8_27
; %bb.28:                               ;   in Loop: Header=BB8_26 Depth=1
	v_wmma_f32_16x16x16_f16 v[96:103], v[61:64], v[65:68], v[17:24]
	v_cvt_f16_f32_e32 v56.h, v56
	v_cvt_f16_f32_e32 v56.l, v55
	v_cvt_f16_f32_e32 v55.h, v54
	s_delay_alu instid0(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[96:103], v[61:64], v[69:72], v[96:103]
	v_cvt_f16_f32_e32 v55.l, v53
	v_cvt_f16_f32_e32 v54.h, v52
	v_cvt_f16_f32_e32 v54.l, v51
	v_cvt_f16_f32_e32 v53.h, v50
	v_cvt_f16_f32_e32 v53.l, v49
	v_cvt_f16_f32_e32 v48.h, v48
	v_cvt_f16_f32_e32 v48.l, v47
	v_cvt_f16_f32_e32 v47.h, v46
	v_cvt_f16_f32_e32 v47.l, v45
	v_wmma_f32_16x16x16_f16 v[96:103], v[57:60], v[53:56], v[96:103]
	v_cvt_f16_f32_e32 v46.h, v44
	v_cvt_f16_f32_e32 v46.l, v43
	v_cvt_f16_f32_e32 v45.h, v42
	v_cvt_f16_f32_e32 v45.l, v41
	v_cvt_f16_f32_e32 v44.h, v40
	v_cvt_f16_f32_e32 v44.l, v39
	v_cvt_f16_f32_e32 v43.h, v38
	v_cvt_f16_f32_e32 v43.l, v37
	v_wmma_f32_16x16x16_f16 v[96:103], v[57:60], v[45:48], v[96:103]
	v_cvt_f16_f32_e32 v42.h, v36
	v_cvt_f16_f32_e32 v42.l, v35
	v_cvt_f16_f32_e32 v41.h, v34
	v_cvt_f16_f32_e32 v41.l, v33
	v_cvt_f16_f32_e32 v52.h, v32
	v_cvt_f16_f32_e32 v52.l, v31
	v_cvt_f16_f32_e32 v51.h, v30
	v_cvt_f16_f32_e32 v51.l, v29
	v_wmma_f32_16x16x16_f16 v[96:103], v[57:60], v[41:44], v[96:103]
	v_cvt_f16_f32_e32 v50.h, v28
	v_cvt_f16_f32_e32 v50.l, v27
	v_cvt_f16_f32_e32 v49.h, v26
	v_cvt_f16_f32_e32 v49.l, v25
	v_dual_add_f32 v32, v24, v24 :: v_dual_add_f32 v31, v23, v23
	v_dual_add_f32 v30, v22, v22 :: v_dual_add_f32 v29, v21, v21
	v_add_f32_e32 v28, v20, v20
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_wmma_f32_16x16x16_f16 v[96:103], v[57:60], v[49:52], v[96:103]
	v_dual_add_f32 v27, v19, v19 :: v_dual_add_f32 v26, v18, v18
	v_dual_add_f32 v25, v17, v17 :: v_dual_mul_f32 v40, 0x40400000, v24
	v_dual_add_f32 v33, 0, v96 :: v_dual_mul_f32 v38, 0x40400000, v22
	v_mul_f32_e32 v36, 0x40400000, v20
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_wmma_f32_16x16x16_f16 v[25:32], v[61:64], v[65:68], v[25:32]
	v_dual_mul_f32 v34, 0x40400000, v18 :: v_dual_add_f32 v33, v33, v97
	v_mul_f32_e32 v22, 4.0, v22
	v_mul_f32_e32 v24, 4.0, v24
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[25:32], v[61:64], v[69:72], v[25:32]
	v_mul_f32_e32 v39, 0x40400000, v23
	v_add_f32_e32 v33, v33, v98
	v_dual_mul_f32 v37, 0x40400000, v21 :: v_dual_mul_f32 v20, 4.0, v20
	v_wmma_f32_16x16x16_f16 v[25:32], v[57:60], v[53:56], v[25:32]
	v_mul_f32_e32 v35, 0x40400000, v19
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_f32_e32 v33, v33, v99
	v_dual_mul_f32 v23, 4.0, v23 :: v_dual_mul_f32 v18, 4.0, v18
	v_wmma_f32_16x16x16_f16 v[25:32], v[57:60], v[45:48], v[25:32]
	v_mul_f32_e32 v21, 4.0, v21
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_f32_e32 v33, v33, v100
	v_dual_mul_f32 v19, 4.0, v19 :: v_dual_mul_f32 v16, 0x3f7d70a4, v16
	v_wmma_f32_16x16x16_f16 v[25:32], v[57:60], v[41:44], v[25:32]
	v_mul_f32_e32 v15, 0x3f7d70a4, v15
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_f32_e32 v33, v33, v101
	v_dual_mul_f32 v13, 0x3f7d70a4, v13 :: v_dual_mul_f32 v12, 0x3f7d70a4, v12
	v_wmma_f32_16x16x16_f16 v[25:32], v[57:60], v[49:52], v[25:32]
	v_mul_f32_e32 v11, 0x3f7d70a4, v11
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_add_f32_e32 v33, v33, v102
	v_dual_mul_f32 v9, 0x3f7d70a4, v9 :: v_dual_mul_f32 v14, 0x3f7d70a4, v14
	v_mul_f32_e32 v7, 0x3f7d70a4, v7
	v_mul_f32_e32 v5, 0x3f7d70a4, v5
	v_add_f32_e32 v33, v33, v103
	v_dual_mul_f32 v3, 0x3f7d70a4, v3 :: v_dual_mul_f32 v10, 0x3f7d70a4, v10
	v_mul_f32_e32 v1, 0x3f7d70a4, v1
	s_add_co_i32 s6, s6, 1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_add_f32_e32 v25, v33, v25
	v_mul_f32_e32 v33, 0x40400000, v17
	v_dual_mul_f32 v8, 0x3f7d70a4, v8 :: v_dual_mul_f32 v17, 4.0, v17
	v_wmma_f32_16x16x16_f16 v[9:16], v[57:60], v[53:56], v[9:16]
	v_add_f32_e32 v25, v25, v26
	s_delay_alu instid0(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[33:40], v[61:64], v[65:68], v[33:40]
	v_mul_f32_e32 v4, 0x3f7d70a4, v4
	v_wmma_f32_16x16x16_f16 v[17:24], v[61:64], v[65:68], v[17:24]
	v_wmma_f32_16x16x16_f16 v[9:16], v[57:60], v[45:48], v[9:16]
	v_add_f32_e32 v25, v25, v27
	v_wmma_f32_16x16x16_f16 v[33:40], v[61:64], v[69:72], v[33:40]
	v_mul_f32_e32 v6, 0x3f7d70a4, v6
	v_wmma_f32_16x16x16_f16 v[17:24], v[61:64], v[69:72], v[17:24]
	v_wmma_f32_16x16x16_f16 v[9:16], v[57:60], v[41:44], v[9:16]
	v_add_f32_e32 v25, v25, v28
	v_wmma_f32_16x16x16_f16 v[33:40], v[57:60], v[53:56], v[33:40]
	v_mul_f32_e32 v2, 0x3f7d70a4, v2
	v_wmma_f32_16x16x16_f16 v[17:24], v[57:60], v[53:56], v[17:24]
	v_wmma_f32_16x16x16_f16 v[9:16], v[57:60], v[49:52], v[9:16]
	v_add_f32_e32 v25, v25, v29
	v_wmma_f32_16x16x16_f16 v[33:40], v[57:60], v[45:48], v[33:40]
	v_wmma_f32_16x16x16_f16 v[1:8], v[57:60], v[53:56], v[1:8]
	v_wmma_f32_16x16x16_f16 v[17:24], v[57:60], v[45:48], v[17:24]
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s6, s3
	v_add_f32_e32 v25, v25, v30
	v_wmma_f32_16x16x16_f16 v[33:40], v[57:60], v[41:44], v[33:40]
	v_wmma_f32_16x16x16_f16 v[1:8], v[57:60], v[45:48], v[1:8]
	v_wmma_f32_16x16x16_f16 v[17:24], v[57:60], v[41:44], v[17:24]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_add_f32_e32 v25, v25, v31
	v_wmma_f32_16x16x16_f16 v[33:40], v[57:60], v[49:52], v[33:40]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[1:8], v[57:60], v[41:44], v[1:8]
	v_wmma_f32_16x16x16_f16 v[17:24], v[57:60], v[49:52], v[17:24]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_f32_e32 v25, v25, v32
	v_wmma_f32_16x16x16_f16 v[1:8], v[57:60], v[49:52], v[1:8]
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v25, v25, v33
	v_add_f32_e32 v25, v25, v34
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v25, v25, v35
	v_add_f32_e32 v25, v25, v36
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v25, v25, v37
	v_add_f32_e32 v25, v25, v38
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v25, v25, v39
	v_add_f32_e32 v25, v25, v40
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v17, v25, v17
	v_add_f32_e32 v17, v17, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v17, v17, v19
	v_add_f32_e32 v17, v17, v20
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v17, v17, v21
	v_add_f32_e32 v17, v17, v22
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v17, v17, v23
	v_add_f32_e32 v17, v17, v24
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cmp_lt_f32_e32 vcc_lo, 0x60ad78ec, v17
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v17, 0x3a83126f, v95, vcc_lo
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
		.amdhsa_next_free_vgpr 117
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
	.set .L_Z5probeILi2ELb0ELb1EEvPfPKfi.num_vgpr, 117
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
; codeLenInByte = 25688
; TotalNumSgprs: 10
; NumVgprs: 117
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 14
; NumSGPRsForWavesPerEU: 10
; NumVGPRsForWavesPerEU: 117
; Occupancy: 12
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
	s_load_b32 s3, s[0:1], 0x10
	s_mov_b32 s0, 0x3c23d70a
	s_mov_b32 s1, 0x38d1b717
	s_mov_b32 s2, 0x399d4951
	s_wait_loadcnt 0xf
	v_fma_mixhi_f16 v45, v10, s0, s1
	s_mov_b32 s1, 0x3951b717
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_wait_alu depctr_sa_sdst(0)
	v_fma_mixlo_f16 v46, v10, s0, s1
	v_fma_mixhi_f16 v46, v10, s0, s2
	s_mov_b32 s1, 0x39d1b717
	s_mov_b32 s2, 0x3a03126e
	s_wait_alu depctr_sa_sdst(0)
	v_fma_mixlo_f16 v47, v10, s0, s1
	v_fma_mixhi_f16 v47, v10, s0, s2
	s_mov_b32 s1, 0x3a1d4951
	s_mov_b32 s2, 0x3a378034
	s_wait_alu depctr_sa_sdst(0)
	v_fma_mixlo_f16 v48, v10, s0, s1
	v_fma_mixhi_f16 v48, v10, s0, s2
	s_mov_b32 s1, 0x3a51b717
	v_fma_mixlo_f16 v45, v10, s0, 0
	v_mov_b16_e32 v41.l, v45.h
	v_mov_b16_e32 v41.h, v46.l
	v_mov_b16_e32 v42.l, v46.h
	v_mov_b16_e32 v42.h, v47.l
	v_mov_b16_e32 v43.l, v47.h
	v_mov_b16_e32 v43.h, v48.l
	s_wait_alu depctr_sa_sdst(0)
	v_fma_mixhi_f16 v44, v10, s0, s1
	v_mov_b16_e32 v44.l, v48.h
	s_mov_b32 s6, 0
	s_wait_kmcnt 0x0
	s_cmp_lt_i32 s3, 1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_cbranch_scc1 .LBB9_29
; %bb.25:
	v_mbcnt_lo_u32_b32 v2, -1, 0
	s_mov_b32 s0, 0x3a83126f
	v_and_b32_e32 v1, 31, v0
	s_wait_alu depctr_sa_sdst(0)
	v_fma_mixlo_f16 v53, v3, s0, 0
	v_fma_mixhi_f16 v49, v4, s0, 0
	v_fma_mixhi_f16 v53, v5, s0, 0
	v_lshrrev_b32_e32 v3, 1, v0
	v_xor_b32_e32 v4, 16, v2
	v_bfi_b32 v5, v2, 0, 32
	v_fma_mixhi_f16 v54, v9, s0, 0
	v_lshlrev_b32_e32 v9, 7, v0
	v_fma_mixlo_f16 v49, v10, s0, 0
	v_fma_mixlo_f16 v50, v6, s0, 0
	v_cmp_lt_u32_e32 vcc_lo, v4, v5
	v_and_b32_e32 v57, 8, v3
	v_fma_mixlo_f16 v54, v7, s0, 0
	v_fma_mixhi_f16 v50, v8, s0, 0
	v_fma_mixhi_f16 v51, v14, s0, 0
	v_fma_mixhi_f16 v55, v15, s0, 0
	v_fma_mixlo_f16 v52, v16, s0, 0
	v_fma_mixlo_f16 v56, v12, s0, 0
	v_fma_mixhi_f16 v52, v13, s0, 0
	v_fma_mixhi_f16 v56, v11, s0, 0
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
	v_fma_mixlo_f16 v51, v17, s0, 0
	v_fma_mixlo_f16 v55, v18, s0, 0
	v_add_nc_u32_e32 v60, 0x1800, v59
	v_or_b32_e32 v61, 0x2000, v59
	v_add_nc_u32_e32 v62, 0xfffff800, v59
	v_lshlrev_b32_e32 v63, 2, v2
	v_dual_mov_b32 v79, 0x3b03126f :: v_dual_lshlrev_b32 v64, 2, v1
	v_lshlrev_b32_e32 v65, 2, v3
	v_lshlrev_b32_e32 v66, 2, v4
	v_lshlrev_b32_e32 v67, 2, v5
	v_lshlrev_b32_e32 v68, 2, v6
	v_lshlrev_b32_e32 v69, 2, v7
	v_lshlrev_b32_e32 v70, 2, v8
	v_lshlrev_b32_e32 v71, 2, v10
	v_lshlrev_b32_e32 v72, 2, v11
	v_lshlrev_b32_e32 v73, 2, v12
	v_lshlrev_b32_e32 v74, 2, v13
	v_lshlrev_b32_e32 v75, 2, v9
	v_lshlrev_b32_e32 v76, 2, v14
	v_lshlrev_b32_e32 v77, 2, v15
	v_lshlrev_b32_e32 v78, 2, v16
	s_mov_b32 s7, 0x3f7d70a4
.LBB9_26:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB9_27 Depth 2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[49:52], v[53:56], 0
	s_mov_b32 s8, 0
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
	v_dual_add_f32 v87, 0x37fba882, v8 :: v_dual_add_f32 v86, 0x37fba882, v7
	v_dual_add_f32 v85, 0x37fba882, v6 :: v_dual_add_f32 v84, 0x37fba882, v5
	v_dual_add_f32 v83, 0x37fba882, v4 :: v_dual_add_f32 v82, 0x37fba882, v3
	v_dual_add_f32 v81, 0x37fba882, v2 :: v_dual_add_f32 v80, 0x37fba882, v1
	v_dual_add_f32 v95, 0x3827c5ac, v8 :: v_dual_add_f32 v94, 0x3827c5ac, v7
	v_dual_add_f32 v93, 0x3827c5ac, v6 :: v_dual_add_f32 v92, 0x3827c5ac, v5
	v_dual_add_f32 v91, 0x3827c5ac, v4 :: v_dual_add_f32 v90, 0x3827c5ac, v3
	v_dual_add_f32 v89, 0x3827c5ac, v2 :: v_dual_add_f32 v88, 0x3827c5ac, v1
	v_wmma_f32_16x16x16_f16 v[9:16], v[49:52], v[45:48], v[9:16]
	v_wmma_f32_16x16x16_f16 v[17:24], v[49:52], v[45:48], v[17:24]
	v_wmma_f32_16x16x16_f16 v[80:87], v[49:52], v[45:48], v[80:87]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[88:95], v[49:52], v[45:48], v[88:95]
	v_wmma_f32_16x16x16_f16 v[9:16], v[49:52], v[41:44], v[9:16]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[17:24], v[49:52], v[41:44], v[17:24]
	v_wmma_f32_16x16x16_f16 v[80:87], v[49:52], v[41:44], v[80:87]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[88:95], v[49:52], v[41:44], v[88:95]
	v_dual_mul_f32 v33, 0x3727c5ac, v9 :: v_dual_mul_f32 v34, 0x3727c5ac, v10
	v_dual_mul_f32 v35, 0x3727c5ac, v11 :: v_dual_mul_f32 v36, 0x3727c5ac, v12
	v_dual_mul_f32 v37, 0x3727c5ac, v13 :: v_dual_mul_f32 v38, 0x3727c5ac, v14
	v_dual_mul_f32 v39, 0x3727c5ac, v15 :: v_dual_mul_f32 v40, 0x3727c5ac, v16
	v_dual_mul_f32 v25, 0x3727c5ac, v17 :: v_dual_mul_f32 v26, 0x3727c5ac, v18
	v_dual_mul_f32 v27, 0x3727c5ac, v19 :: v_dual_mul_f32 v28, 0x3727c5ac, v20
	v_dual_mul_f32 v29, 0x3727c5ac, v21 :: v_dual_mul_f32 v30, 0x3727c5ac, v22
	v_dual_mul_f32 v31, 0x3727c5ac, v23 :: v_dual_mul_f32 v32, 0x3727c5ac, v24
	v_dual_mul_f32 v17, 0x3727c5ac, v80 :: v_dual_mul_f32 v18, 0x3727c5ac, v81
	v_dual_mul_f32 v19, 0x3727c5ac, v82 :: v_dual_mul_f32 v20, 0x3727c5ac, v83
	v_dual_mul_f32 v21, 0x3727c5ac, v84 :: v_dual_mul_f32 v22, 0x3727c5ac, v85
	v_dual_mul_f32 v23, 0x3727c5ac, v86 :: v_dual_mul_f32 v24, 0x3727c5ac, v87
	v_dual_mul_f32 v9, 0x3727c5ac, v88 :: v_dual_mul_f32 v10, 0x3727c5ac, v89
	v_dual_mul_f32 v11, 0x3727c5ac, v90 :: v_dual_mul_f32 v12, 0x3727c5ac, v91
	v_dual_mul_f32 v13, 0x3727c5ac, v92 :: v_dual_mul_f32 v14, 0x3727c5ac, v93
	v_dual_mul_f32 v15, 0x3727c5ac, v94 :: v_dual_mul_f32 v16, 0x3727c5ac, v95
.LBB9_27:                               ;   Parent Loop BB9_26 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_wait_alu depctr_sa_sdst(0)
	v_cmp_eq_u32_e64 s1, s8, v59
	v_cmp_eq_u32_e64 s2, s8, v61
	v_cmp_eq_u32_e32 vcc_lo, s8, v62
	v_cmp_eq_u32_e64 s0, s8, v60
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v80, 0, v33, s1
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v80, v80, v17, s2
	ds_bpermute_b32 v81, v63, v80
	s_wait_dscnt 0x0
	v_dual_add_f32 v84, v80, v81 :: v_dual_add_nc_u32 v81, s8, v58
	v_lshlrev_b32_e32 v80, 2, v57
	s_addk_co_i32 s8, 0x2000
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s8, 0x4000
	ds_load_2addr_b32 v[82:83], v81 offset1:32
	s_wait_dscnt 0x0
	ds_bpermute_b32 v85, v80, v82
	s_wait_dscnt 0x0
	v_fma_f32 v33, -v84, v85, v33
	ds_bpermute_b32 v85, v64, v82
	s_wait_dscnt 0x0
	v_fma_f32 v34, -v84, v85, v34
	ds_bpermute_b32 v85, v65, v82
	s_wait_dscnt 0x0
	v_fma_f32 v35, -v84, v85, v35
	ds_bpermute_b32 v85, v66, v82
	s_wait_dscnt 0x0
	v_fma_f32 v36, -v84, v85, v36
	ds_bpermute_b32 v85, v67, v82
	s_wait_dscnt 0x0
	v_fma_f32 v37, -v84, v85, v37
	ds_bpermute_b32 v85, v68, v82
	s_wait_dscnt 0x0
	v_fma_f32 v38, -v84, v85, v38
	ds_bpermute_b32 v85, v69, v82
	s_wait_dscnt 0x0
	v_fma_f32 v39, -v84, v85, v39
	ds_bpermute_b32 v85, v70, v82
	s_wait_dscnt 0x0
	v_fma_f32 v40, -v84, v85, v40
	ds_bpermute_b32 v85, v71, v82
	s_wait_dscnt 0x0
	v_fma_f32 v25, -v84, v85, v25
	ds_bpermute_b32 v85, v72, v82
	s_wait_dscnt 0x0
	v_fma_f32 v26, -v84, v85, v26
	ds_bpermute_b32 v85, v73, v82
	s_wait_dscnt 0x0
	v_fma_f32 v27, -v84, v85, v27
	ds_bpermute_b32 v85, v74, v82
	s_wait_dscnt 0x0
	v_fma_f32 v28, -v84, v85, v28
	ds_bpermute_b32 v85, v75, v82
	s_wait_dscnt 0x0
	v_fma_f32 v29, -v84, v85, v29
	ds_bpermute_b32 v85, v76, v82
	s_wait_dscnt 0x0
	v_fma_f32 v30, -v84, v85, v30
	ds_bpermute_b32 v85, v77, v82
	ds_bpermute_b32 v82, v78, v82
	s_wait_dscnt 0x1
	v_fma_f32 v31, -v84, v85, v31
	s_wait_dscnt 0x0
	v_fma_f32 v32, -v84, v82, v32
	ds_bpermute_b32 v82, v80, v83
	s_wait_dscnt 0x0
	v_fma_f32 v17, -v84, v82, v17
	ds_bpermute_b32 v82, v64, v83
	s_wait_dscnt 0x0
	v_fma_f32 v18, -v84, v82, v18
	ds_bpermute_b32 v82, v65, v83
	s_wait_dscnt 0x0
	v_fma_f32 v19, -v84, v82, v19
	ds_bpermute_b32 v82, v66, v83
	s_wait_dscnt 0x0
	v_fma_f32 v20, -v84, v82, v20
	ds_bpermute_b32 v82, v67, v83
	s_wait_dscnt 0x0
	v_fma_f32 v21, -v84, v82, v21
	ds_bpermute_b32 v82, v68, v83
	s_wait_dscnt 0x0
	v_fma_f32 v22, -v84, v82, v22
	ds_bpermute_b32 v82, v69, v83
	s_wait_dscnt 0x0
	v_fma_f32 v23, -v84, v82, v23
	ds_bpermute_b32 v82, v70, v83
	s_wait_dscnt 0x0
	v_fma_f32 v24, -v84, v82, v24
	ds_bpermute_b32 v82, v71, v83
	s_wait_dscnt 0x0
	v_fma_f32 v9, -v84, v82, v9
	ds_bpermute_b32 v82, v72, v83
	s_wait_dscnt 0x0
	v_fma_f32 v10, -v84, v82, v10
	ds_bpermute_b32 v82, v73, v83
	s_wait_dscnt 0x0
	v_fma_f32 v11, -v84, v82, v11
	ds_bpermute_b32 v82, v74, v83
	s_wait_dscnt 0x0
	v_fma_f32 v12, -v84, v82, v12
	ds_bpermute_b32 v82, v75, v83
	s_wait_dscnt 0x0
	v_fma_f32 v13, -v84, v82, v13
	ds_bpermute_b32 v82, v76, v83
	s_wait_dscnt 0x0
	v_fma_f32 v14, -v84, v82, v14
	ds_bpermute_b32 v82, v77, v83
	s_wait_dscnt 0x0
	v_fma_f32 v15, -v84, v82, v15
	ds_bpermute_b32 v82, v78, v83
	s_wait_dscnt 0x0
	v_fma_f32 v16, -v84, v82, v16
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v82, 0, v34, s1
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v82, v82, v18, s2
	ds_bpermute_b32 v83, v63, v82
	s_wait_dscnt 0x0
	v_add_f32_e32 v84, v82, v83
	ds_load_2addr_b32 v[82:83], v81 offset0:64 offset1:96
	s_wait_dscnt 0x0
	ds_bpermute_b32 v85, v80, v82
	s_wait_dscnt 0x0
	v_fma_f32 v33, -v84, v85, v33
	ds_bpermute_b32 v85, v64, v82
	s_wait_dscnt 0x0
	v_fma_f32 v34, -v84, v85, v34
	ds_bpermute_b32 v85, v65, v82
	s_wait_dscnt 0x0
	v_fma_f32 v35, -v84, v85, v35
	ds_bpermute_b32 v85, v66, v82
	s_wait_dscnt 0x0
	v_fma_f32 v36, -v84, v85, v36
	ds_bpermute_b32 v85, v67, v82
	s_wait_dscnt 0x0
	v_fma_f32 v37, -v84, v85, v37
	ds_bpermute_b32 v85, v68, v82
	s_wait_dscnt 0x0
	v_fma_f32 v38, -v84, v85, v38
	ds_bpermute_b32 v85, v69, v82
	s_wait_dscnt 0x0
	v_fma_f32 v39, -v84, v85, v39
	ds_bpermute_b32 v85, v70, v82
	s_wait_dscnt 0x0
	v_fma_f32 v40, -v84, v85, v40
	ds_bpermute_b32 v85, v71, v82
	s_wait_dscnt 0x0
	v_fma_f32 v25, -v84, v85, v25
	ds_bpermute_b32 v85, v72, v82
	s_wait_dscnt 0x0
	v_fma_f32 v26, -v84, v85, v26
	ds_bpermute_b32 v85, v73, v82
	s_wait_dscnt 0x0
	v_fma_f32 v27, -v84, v85, v27
	ds_bpermute_b32 v85, v74, v82
	s_wait_dscnt 0x0
	v_fma_f32 v28, -v84, v85, v28
	ds_bpermute_b32 v85, v75, v82
	s_wait_dscnt 0x0
	v_fma_f32 v29, -v84, v85, v29
	ds_bpermute_b32 v85, v76, v82
	s_wait_dscnt 0x0
	v_fma_f32 v30, -v84, v85, v30
	ds_bpermute_b32 v85, v77, v82
	ds_bpermute_b32 v82, v78, v82
	s_wait_dscnt 0x1
	v_fma_f32 v31, -v84, v85, v31
	s_wait_dscnt 0x0
	v_fma_f32 v32, -v84, v82, v32
	ds_bpermute_b32 v82, v80, v83
	s_wait_dscnt 0x0
	v_fma_f32 v17, -v84, v82, v17
	ds_bpermute_b32 v82, v64, v83
	s_wait_dscnt 0x0
	v_fma_f32 v18, -v84, v82, v18
	ds_bpermute_b32 v82, v65, v83
	s_wait_dscnt 0x0
	v_fma_f32 v19, -v84, v82, v19
	ds_bpermute_b32 v82, v66, v83
	s_wait_dscnt 0x0
	v_fma_f32 v20, -v84, v82, v20
	ds_bpermute_b32 v82, v67, v83
	s_wait_dscnt 0x0
	v_fma_f32 v21, -v84, v82, v21
	ds_bpermute_b32 v82, v68, v83
	s_wait_dscnt 0x0
	v_fma_f32 v22, -v84, v82, v22
	ds_bpermute_b32 v82, v69, v83
	s_wait_dscnt 0x0
	v_fma_f32 v23, -v84, v82, v23
	ds_bpermute_b32 v82, v70, v83
	s_wait_dscnt 0x0
	v_fma_f32 v24, -v84, v82, v24
	ds_bpermute_b32 v82, v71, v83
	s_wait_dscnt 0x0
	v_fma_f32 v9, -v84, v82, v9
	ds_bpermute_b32 v82, v72, v83
	s_wait_dscnt 0x0
	v_fma_f32 v10, -v84, v82, v10
	ds_bpermute_b32 v82, v73, v83
	s_wait_dscnt 0x0
	v_fma_f32 v11, -v84, v82, v11
	ds_bpermute_b32 v82, v74, v83
	s_wait_dscnt 0x0
	v_fma_f32 v12, -v84, v82, v12
	ds_bpermute_b32 v82, v75, v83
	s_wait_dscnt 0x0
	v_fma_f32 v13, -v84, v82, v13
	ds_bpermute_b32 v82, v76, v83
	s_wait_dscnt 0x0
	v_fma_f32 v14, -v84, v82, v14
	ds_bpermute_b32 v82, v77, v83
	s_wait_dscnt 0x0
	v_fma_f32 v15, -v84, v82, v15
	ds_bpermute_b32 v82, v78, v83
	s_wait_dscnt 0x0
	v_fma_f32 v16, -v84, v82, v16
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v82, 0, v35, s1
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v82, v82, v19, s2
	ds_bpermute_b32 v83, v63, v82
	s_wait_dscnt 0x0
	v_add_f32_e32 v84, v82, v83
	ds_load_2addr_b32 v[82:83], v81 offset0:128 offset1:160
	s_wait_dscnt 0x0
	ds_bpermute_b32 v85, v80, v82
	s_wait_dscnt 0x0
	v_fma_f32 v33, -v84, v85, v33
	ds_bpermute_b32 v85, v64, v82
	s_wait_dscnt 0x0
	v_fma_f32 v34, -v84, v85, v34
	ds_bpermute_b32 v85, v65, v82
	s_wait_dscnt 0x0
	v_fma_f32 v35, -v84, v85, v35
	ds_bpermute_b32 v85, v66, v82
	s_wait_dscnt 0x0
	v_fma_f32 v36, -v84, v85, v36
	ds_bpermute_b32 v85, v67, v82
	s_wait_dscnt 0x0
	v_fma_f32 v37, -v84, v85, v37
	ds_bpermute_b32 v85, v68, v82
	s_wait_dscnt 0x0
	v_fma_f32 v38, -v84, v85, v38
	ds_bpermute_b32 v85, v69, v82
	s_wait_dscnt 0x0
	v_fma_f32 v39, -v84, v85, v39
	ds_bpermute_b32 v85, v70, v82
	s_wait_dscnt 0x0
	v_fma_f32 v40, -v84, v85, v40
	ds_bpermute_b32 v85, v71, v82
	s_wait_dscnt 0x0
	v_fma_f32 v25, -v84, v85, v25
	ds_bpermute_b32 v85, v72, v82
	s_wait_dscnt 0x0
	v_fma_f32 v26, -v84, v85, v26
	ds_bpermute_b32 v85, v73, v82
	s_wait_dscnt 0x0
	v_fma_f32 v27, -v84, v85, v27
	ds_bpermute_b32 v85, v74, v82
	s_wait_dscnt 0x0
	v_fma_f32 v28, -v84, v85, v28
	ds_bpermute_b32 v85, v75, v82
	s_wait_dscnt 0x0
	v_fma_f32 v29, -v84, v85, v29
	ds_bpermute_b32 v85, v76, v82
	s_wait_dscnt 0x0
	v_fma_f32 v30, -v84, v85, v30
	ds_bpermute_b32 v85, v77, v82
	ds_bpermute_b32 v82, v78, v82
	s_wait_dscnt 0x1
	v_fma_f32 v31, -v84, v85, v31
	s_wait_dscnt 0x0
	v_fma_f32 v32, -v84, v82, v32
	ds_bpermute_b32 v82, v80, v83
	s_wait_dscnt 0x0
	v_fma_f32 v17, -v84, v82, v17
	ds_bpermute_b32 v82, v64, v83
	s_wait_dscnt 0x0
	v_fma_f32 v18, -v84, v82, v18
	ds_bpermute_b32 v82, v65, v83
	s_wait_dscnt 0x0
	v_fma_f32 v19, -v84, v82, v19
	ds_bpermute_b32 v82, v66, v83
	s_wait_dscnt 0x0
	v_fma_f32 v20, -v84, v82, v20
	ds_bpermute_b32 v82, v67, v83
	s_wait_dscnt 0x0
	v_fma_f32 v21, -v84, v82, v21
	ds_bpermute_b32 v82, v68, v83
	s_wait_dscnt 0x0
	v_fma_f32 v22, -v84, v82, v22
	ds_bpermute_b32 v82, v69, v83
	s_wait_dscnt 0x0
	v_fma_f32 v23, -v84, v82, v23
	ds_bpermute_b32 v82, v70, v83
	s_wait_dscnt 0x0
	v_fma_f32 v24, -v84, v82, v24
	ds_bpermute_b32 v82, v71, v83
	s_wait_dscnt 0x0
	v_fma_f32 v9, -v84, v82, v9
	ds_bpermute_b32 v82, v72, v83
	s_wait_dscnt 0x0
	v_fma_f32 v10, -v84, v82, v10
	ds_bpermute_b32 v82, v73, v83
	s_wait_dscnt 0x0
	v_fma_f32 v11, -v84, v82, v11
	ds_bpermute_b32 v82, v74, v83
	s_wait_dscnt 0x0
	v_fma_f32 v12, -v84, v82, v12
	ds_bpermute_b32 v82, v75, v83
	s_wait_dscnt 0x0
	v_fma_f32 v13, -v84, v82, v13
	ds_bpermute_b32 v82, v76, v83
	s_wait_dscnt 0x0
	v_fma_f32 v14, -v84, v82, v14
	ds_bpermute_b32 v82, v77, v83
	s_wait_dscnt 0x0
	v_fma_f32 v15, -v84, v82, v15
	ds_bpermute_b32 v82, v78, v83
	s_wait_dscnt 0x0
	v_fma_f32 v16, -v84, v82, v16
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v82, 0, v36, s1
	ds_load_2addr_b32 v[88:89], v81 offset0:192 offset1:224
	v_cndmask_b32_e64 v82, v82, v20, s2
	ds_bpermute_b32 v83, v63, v82
	s_wait_dscnt 0x0
	v_add_f32_e32 v90, v82, v83
	ds_bpermute_b32 v82, v80, v88
	s_wait_dscnt 0x0
	v_fma_f32 v33, -v90, v82, v33
	ds_bpermute_b32 v82, v64, v88
	s_wait_dscnt 0x0
	v_fma_f32 v34, -v90, v82, v34
	ds_bpermute_b32 v82, v65, v88
	s_wait_dscnt 0x0
	v_fma_f32 v35, -v90, v82, v35
	ds_bpermute_b32 v82, v66, v88
	s_wait_dscnt 0x0
	v_fma_f32 v36, -v90, v82, v36
	ds_bpermute_b32 v82, v67, v88
	s_wait_dscnt 0x0
	v_fma_f32 v37, -v90, v82, v37
	ds_bpermute_b32 v82, v68, v88
	s_wait_dscnt 0x0
	v_fma_f32 v38, -v90, v82, v38
	ds_bpermute_b32 v82, v69, v88
	s_wait_dscnt 0x0
	v_fma_f32 v39, -v90, v82, v39
	ds_bpermute_b32 v82, v70, v88
	s_wait_dscnt 0x0
	v_fma_f32 v40, -v90, v82, v40
	ds_bpermute_b32 v82, v71, v88
	s_wait_dscnt 0x0
	v_fma_f32 v25, -v90, v82, v25
	ds_bpermute_b32 v82, v72, v88
	s_wait_dscnt 0x0
	v_fma_f32 v26, -v90, v82, v26
	ds_bpermute_b32 v82, v73, v88
	s_wait_dscnt 0x0
	v_fma_f32 v27, -v90, v82, v27
	ds_bpermute_b32 v82, v74, v88
	s_wait_dscnt 0x0
	v_fma_f32 v28, -v90, v82, v28
	ds_bpermute_b32 v82, v75, v88
	s_wait_dscnt 0x0
	v_fma_f32 v29, -v90, v82, v29
	ds_bpermute_b32 v82, v76, v88
	s_wait_dscnt 0x0
	v_fma_f32 v30, -v90, v82, v30
	ds_bpermute_b32 v82, v77, v88
	s_wait_dscnt 0x0
	v_fma_f32 v31, -v90, v82, v31
	ds_bpermute_b32 v82, v78, v88
	s_wait_dscnt 0x0
	v_fma_f32 v32, -v90, v82, v32
	ds_bpermute_b32 v82, v80, v89
	s_wait_dscnt 0x0
	v_fma_f32 v17, -v90, v82, v17
	ds_bpermute_b32 v82, v64, v89
	s_wait_dscnt 0x0
	v_fma_f32 v18, -v90, v82, v18
	ds_bpermute_b32 v82, v65, v89
	s_wait_dscnt 0x0
	v_fma_f32 v19, -v90, v82, v19
	ds_bpermute_b32 v82, v66, v89
	s_wait_dscnt 0x0
	v_fma_f32 v20, -v90, v82, v20
	ds_bpermute_b32 v82, v67, v89
	s_wait_dscnt 0x0
	v_fma_f32 v21, -v90, v82, v21
	ds_bpermute_b32 v82, v68, v89
	s_wait_dscnt 0x0
	v_fma_f32 v22, -v90, v82, v22
	ds_bpermute_b32 v82, v69, v89
	s_wait_dscnt 0x0
	v_fma_f32 v23, -v90, v82, v23
	ds_bpermute_b32 v82, v70, v89
	s_wait_dscnt 0x0
	v_fma_f32 v24, -v90, v82, v24
	ds_bpermute_b32 v82, v71, v89
	s_wait_dscnt 0x0
	v_fma_f32 v82, -v90, v82, v9
	ds_bpermute_b32 v9, v72, v89
	s_wait_dscnt 0x0
	v_fma_f32 v83, -v90, v9, v10
	ds_bpermute_b32 v9, v73, v89
	s_wait_dscnt 0x0
	v_fma_f32 v84, -v90, v9, v11
	ds_bpermute_b32 v9, v74, v89
	s_wait_dscnt 0x0
	v_fma_f32 v85, -v90, v9, v12
	ds_bpermute_b32 v9, v75, v89
	s_wait_dscnt 0x0
	v_fma_f32 v86, -v90, v9, v13
	ds_bpermute_b32 v9, v76, v89
	s_wait_dscnt 0x0
	v_fma_f32 v87, -v90, v9, v14
	ds_bpermute_b32 v9, v77, v89
	s_wait_dscnt 0x0
	v_fma_f32 v88, -v90, v9, v15
	ds_bpermute_b32 v9, v78, v89
	s_wait_dscnt 0x0
	v_fma_f32 v89, -v90, v9, v16
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v9, 0, v37, s1
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v9, v9, v21, s2
	ds_bpermute_b32 v10, v63, v9
	s_wait_dscnt 0x0
	v_dual_add_f32 v92, v9, v10 :: v_dual_add_nc_u32 v9, 0x400, v81
	ds_load_2addr_b32 v[90:91], v9 offset1:32
	s_wait_dscnt 0x0
	ds_bpermute_b32 v10, v80, v90
	ds_bpermute_b32 v11, v64, v91
	ds_bpermute_b32 v12, v65, v91
	ds_bpermute_b32 v13, v66, v91
	ds_bpermute_b32 v14, v67, v91
	ds_bpermute_b32 v15, v68, v91
	ds_bpermute_b32 v16, v69, v91
	s_wait_dscnt 0x6
	v_fma_f32 v33, -v92, v10, v33
	ds_bpermute_b32 v10, v64, v90
	s_wait_dscnt 0x6
	v_fma_f32 v11, -v92, v11, v18
	ds_bpermute_b32 v18, v71, v91
	s_wait_dscnt 0x6
	v_fma_f32 v12, -v92, v12, v19
	s_wait_dscnt 0x5
	v_fma_f32 v13, -v92, v13, v20
	s_wait_dscnt 0x4
	v_fma_f32 v14, -v92, v14, v21
	s_wait_dscnt 0x3
	v_fma_f32 v15, -v92, v15, v22
	s_wait_dscnt 0x2
	v_fma_f32 v16, -v92, v16, v23
	s_wait_dscnt 0x1
	v_fma_f32 v34, -v92, v10, v34
	ds_bpermute_b32 v10, v65, v90
	s_wait_dscnt 0x1
	v_fma_f32 v82, -v92, v18, v82
	ds_bpermute_b32 v18, v72, v91
	s_wait_dscnt 0x1
	v_fma_f32 v35, -v92, v10, v35
	ds_bpermute_b32 v10, v66, v90
	s_wait_dscnt 0x1
	v_fma_f32 v83, -v92, v18, v83
	ds_bpermute_b32 v18, v73, v91
	s_wait_dscnt 0x1
	v_fma_f32 v36, -v92, v10, v36
	ds_bpermute_b32 v10, v67, v90
	s_wait_dscnt 0x1
	v_fma_f32 v84, -v92, v18, v84
	ds_bpermute_b32 v18, v74, v91
	s_wait_dscnt 0x1
	v_fma_f32 v37, -v92, v10, v37
	ds_bpermute_b32 v10, v68, v90
	s_wait_dscnt 0x1
	v_fma_f32 v85, -v92, v18, v85
	ds_bpermute_b32 v18, v75, v91
	s_wait_dscnt 0x1
	v_fma_f32 v38, -v92, v10, v38
	ds_bpermute_b32 v10, v69, v90
	s_wait_dscnt 0x1
	v_fma_f32 v86, -v92, v18, v86
	ds_bpermute_b32 v18, v76, v91
	s_wait_dscnt 0x1
	v_fma_f32 v39, -v92, v10, v39
	ds_bpermute_b32 v10, v70, v90
	s_wait_dscnt 0x1
	v_fma_f32 v87, -v92, v18, v87
	ds_bpermute_b32 v18, v77, v91
	s_wait_dscnt 0x1
	v_fma_f32 v40, -v92, v10, v40
	ds_bpermute_b32 v10, v71, v90
	s_wait_dscnt 0x1
	v_fma_f32 v88, -v92, v18, v88
	ds_bpermute_b32 v18, v78, v91
	s_wait_dscnt 0x1
	v_fma_f32 v25, -v92, v10, v25
	ds_bpermute_b32 v10, v72, v90
	s_wait_dscnt 0x1
	v_fma_f32 v89, -v92, v18, v89
	s_wait_dscnt 0x0
	v_fma_f32 v26, -v92, v10, v26
	ds_bpermute_b32 v10, v73, v90
	s_wait_dscnt 0x0
	v_fma_f32 v27, -v92, v10, v27
	ds_bpermute_b32 v10, v74, v90
	s_wait_dscnt 0x0
	v_fma_f32 v28, -v92, v10, v28
	ds_bpermute_b32 v10, v75, v90
	s_wait_dscnt 0x0
	v_fma_f32 v29, -v92, v10, v29
	ds_bpermute_b32 v10, v76, v90
	s_wait_dscnt 0x0
	v_fma_f32 v30, -v92, v10, v30
	ds_bpermute_b32 v10, v77, v90
	s_wait_dscnt 0x0
	v_fma_f32 v31, -v92, v10, v31
	ds_bpermute_b32 v10, v78, v90
	s_wait_dscnt 0x0
	v_fma_f32 v32, -v92, v10, v32
	ds_bpermute_b32 v10, v80, v91
	s_wait_dscnt 0x0
	v_fma_f32 v10, -v92, v10, v17
	ds_bpermute_b32 v17, v70, v91
	s_wait_dscnt 0x0
	v_fma_f32 v17, -v92, v17, v24
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v18, 0, v38, s1
	ds_load_2addr_b32 v[90:91], v9 offset0:64 offset1:96
	v_cndmask_b32_e64 v18, v18, v15, s2
	ds_bpermute_b32 v19, v63, v18
	s_wait_dscnt 0x1
	ds_bpermute_b32 v20, v73, v90
	ds_bpermute_b32 v21, v74, v90
	ds_bpermute_b32 v22, v75, v90
	ds_bpermute_b32 v23, v76, v90
	ds_bpermute_b32 v24, v77, v90
	s_wait_dscnt 0x5
	v_add_f32_e32 v92, v18, v19
	ds_bpermute_b32 v19, v72, v90
	ds_bpermute_b32 v18, v80, v90
	s_wait_dscnt 0x6
	v_fma_f32 v20, -v92, v20, v27
	s_wait_dscnt 0x5
	v_fma_f32 v21, -v92, v21, v28
	s_wait_dscnt 0x4
	v_fma_f32 v22, -v92, v22, v29
	s_wait_dscnt 0x3
	v_fma_f32 v23, -v92, v23, v30
	s_wait_dscnt 0x2
	v_fma_f32 v24, -v92, v24, v31
	s_wait_dscnt 0x1
	v_fma_f32 v19, -v92, v19, v26
	ds_bpermute_b32 v26, v80, v91
	s_wait_dscnt 0x1
	v_fma_f32 v33, -v92, v18, v33
	ds_bpermute_b32 v18, v64, v90
	s_wait_dscnt 0x1
	v_fma_f32 v10, -v92, v26, v10
	ds_bpermute_b32 v26, v64, v91
	s_wait_dscnt 0x1
	v_fma_f32 v34, -v92, v18, v34
	ds_bpermute_b32 v18, v65, v90
	s_wait_dscnt 0x1
	v_fma_f32 v11, -v92, v26, v11
	ds_bpermute_b32 v26, v65, v91
	s_wait_dscnt 0x1
	v_fma_f32 v35, -v92, v18, v35
	ds_bpermute_b32 v18, v66, v90
	s_wait_dscnt 0x1
	v_fma_f32 v12, -v92, v26, v12
	ds_bpermute_b32 v26, v66, v91
	s_wait_dscnt 0x1
	v_fma_f32 v36, -v92, v18, v36
	ds_bpermute_b32 v18, v67, v90
	s_wait_dscnt 0x1
	v_fma_f32 v13, -v92, v26, v13
	ds_bpermute_b32 v26, v67, v91
	s_wait_dscnt 0x1
	v_fma_f32 v37, -v92, v18, v37
	ds_bpermute_b32 v18, v68, v90
	s_wait_dscnt 0x1
	v_fma_f32 v14, -v92, v26, v14
	ds_bpermute_b32 v26, v68, v91
	s_wait_dscnt 0x1
	v_fma_f32 v38, -v92, v18, v38
	ds_bpermute_b32 v18, v69, v90
	s_wait_dscnt 0x1
	v_fma_f32 v15, -v92, v26, v15
	ds_bpermute_b32 v26, v69, v91
	s_wait_dscnt 0x1
	v_fma_f32 v39, -v92, v18, v39
	ds_bpermute_b32 v18, v70, v90
	s_wait_dscnt 0x1
	v_fma_f32 v16, -v92, v26, v16
	ds_bpermute_b32 v26, v70, v91
	s_wait_dscnt 0x1
	v_fma_f32 v40, -v92, v18, v40
	ds_bpermute_b32 v18, v71, v90
	s_wait_dscnt 0x1
	v_fma_f32 v17, -v92, v26, v17
	ds_bpermute_b32 v26, v71, v91
	s_wait_dscnt 0x1
	v_fma_f32 v18, -v92, v18, v25
	ds_bpermute_b32 v25, v78, v90
	s_wait_dscnt 0x1
	v_fma_f32 v82, -v92, v26, v82
	ds_bpermute_b32 v26, v72, v91
	s_wait_dscnt 0x1
	v_fma_f32 v25, -v92, v25, v32
	s_wait_dscnt 0x0
	v_fma_f32 v83, -v92, v26, v83
	ds_bpermute_b32 v26, v73, v91
	s_wait_dscnt 0x0
	v_fma_f32 v84, -v92, v26, v84
	ds_bpermute_b32 v26, v74, v91
	s_wait_dscnt 0x0
	v_fma_f32 v85, -v92, v26, v85
	ds_bpermute_b32 v26, v75, v91
	s_wait_dscnt 0x0
	v_fma_f32 v86, -v92, v26, v86
	ds_bpermute_b32 v26, v76, v91
	s_wait_dscnt 0x0
	v_fma_f32 v87, -v92, v26, v87
	ds_bpermute_b32 v26, v77, v91
	s_wait_dscnt 0x0
	v_fma_f32 v88, -v92, v26, v88
	ds_bpermute_b32 v26, v78, v91
	s_wait_dscnt 0x0
	v_fma_f32 v89, -v92, v26, v89
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v26, 0, v39, s1
	ds_load_2addr_b32 v[90:91], v9 offset0:128 offset1:160
	v_cndmask_b32_e64 v26, v26, v16, s2
	ds_bpermute_b32 v27, v63, v26
	s_wait_dscnt 0x1
	ds_bpermute_b32 v28, v65, v90
	ds_bpermute_b32 v29, v66, v90
	ds_bpermute_b32 v30, v67, v90
	ds_bpermute_b32 v31, v68, v90
	ds_bpermute_b32 v32, v69, v90
	s_wait_dscnt 0x5
	v_add_f32_e32 v92, v26, v27
	ds_bpermute_b32 v27, v64, v90
	ds_bpermute_b32 v26, v80, v90
	s_wait_dscnt 0x6
	v_fma_f32 v28, -v92, v28, v35
	s_wait_dscnt 0x5
	v_fma_f32 v29, -v92, v29, v36
	s_wait_dscnt 0x4
	v_fma_f32 v30, -v92, v30, v37
	s_wait_dscnt 0x3
	v_fma_f32 v31, -v92, v31, v38
	s_wait_dscnt 0x2
	v_fma_f32 v32, -v92, v32, v39
	s_wait_dscnt 0x1
	v_fma_f32 v27, -v92, v27, v34
	ds_bpermute_b32 v34, v71, v90
	s_wait_dscnt 0x1
	v_fma_f32 v26, -v92, v26, v33
	ds_bpermute_b32 v33, v70, v90
	s_wait_dscnt 0x1
	v_fma_f32 v18, -v92, v34, v18
	ds_bpermute_b32 v34, v72, v90
	s_wait_dscnt 0x1
	v_fma_f32 v33, -v92, v33, v40
	s_wait_dscnt 0x0
	v_fma_f32 v19, -v92, v34, v19
	ds_bpermute_b32 v34, v73, v90
	s_wait_dscnt 0x0
	v_fma_f32 v20, -v92, v34, v20
	ds_bpermute_b32 v34, v74, v90
	s_wait_dscnt 0x0
	v_fma_f32 v21, -v92, v34, v21
	ds_bpermute_b32 v34, v75, v90
	s_wait_dscnt 0x0
	v_fma_f32 v22, -v92, v34, v22
	ds_bpermute_b32 v34, v76, v90
	s_wait_dscnt 0x0
	v_fma_f32 v23, -v92, v34, v23
	ds_bpermute_b32 v34, v77, v90
	s_wait_dscnt 0x0
	v_fma_f32 v24, -v92, v34, v24
	ds_bpermute_b32 v34, v78, v90
	s_wait_dscnt 0x0
	v_fma_f32 v25, -v92, v34, v25
	ds_bpermute_b32 v34, v80, v91
	s_wait_dscnt 0x0
	v_fma_f32 v10, -v92, v34, v10
	ds_bpermute_b32 v34, v64, v91
	s_wait_dscnt 0x0
	v_fma_f32 v11, -v92, v34, v11
	ds_bpermute_b32 v34, v65, v91
	s_wait_dscnt 0x0
	v_fma_f32 v12, -v92, v34, v12
	ds_bpermute_b32 v34, v66, v91
	s_wait_dscnt 0x0
	v_fma_f32 v13, -v92, v34, v13
	ds_bpermute_b32 v34, v67, v91
	s_wait_dscnt 0x0
	v_fma_f32 v14, -v92, v34, v14
	ds_bpermute_b32 v34, v68, v91
	s_wait_dscnt 0x0
	v_fma_f32 v15, -v92, v34, v15
	ds_bpermute_b32 v34, v69, v91
	s_wait_dscnt 0x0
	v_fma_f32 v16, -v92, v34, v16
	ds_bpermute_b32 v34, v70, v91
	s_wait_dscnt 0x0
	v_fma_f32 v17, -v92, v34, v17
	ds_bpermute_b32 v34, v71, v91
	s_wait_dscnt 0x0
	v_fma_f32 v82, -v92, v34, v82
	ds_bpermute_b32 v34, v72, v91
	s_wait_dscnt 0x0
	v_fma_f32 v83, -v92, v34, v83
	ds_bpermute_b32 v34, v73, v91
	s_wait_dscnt 0x0
	v_fma_f32 v84, -v92, v34, v84
	ds_bpermute_b32 v34, v74, v91
	s_wait_dscnt 0x0
	v_fma_f32 v85, -v92, v34, v85
	ds_bpermute_b32 v34, v75, v91
	s_wait_dscnt 0x0
	v_fma_f32 v86, -v92, v34, v86
	ds_bpermute_b32 v34, v76, v91
	s_wait_dscnt 0x0
	v_fma_f32 v87, -v92, v34, v87
	ds_bpermute_b32 v34, v77, v91
	s_wait_dscnt 0x0
	v_fma_f32 v88, -v92, v34, v88
	ds_bpermute_b32 v34, v78, v91
	s_wait_dscnt 0x0
	v_fma_f32 v89, -v92, v34, v89
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v34, 0, v33, s1
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v34, v34, v17, s2
	ds_bpermute_b32 v35, v63, v34
	s_wait_dscnt 0x0
	v_add_f32_e32 v36, v34, v35
	ds_load_2addr_b32 v[34:35], v9 offset0:192 offset1:224
	s_wait_dscnt 0x0
	ds_bpermute_b32 v9, v80, v34
	s_wait_dscnt 0x0
	v_fma_f32 v26, -v36, v9, v26
	ds_bpermute_b32 v9, v64, v34
	s_wait_dscnt 0x0
	v_fma_f32 v27, -v36, v9, v27
	ds_bpermute_b32 v9, v65, v34
	s_wait_dscnt 0x0
	v_fma_f32 v28, -v36, v9, v28
	ds_bpermute_b32 v9, v66, v34
	s_wait_dscnt 0x0
	v_fma_f32 v29, -v36, v9, v29
	ds_bpermute_b32 v9, v67, v34
	s_wait_dscnt 0x0
	v_fma_f32 v30, -v36, v9, v30
	ds_bpermute_b32 v9, v68, v34
	s_wait_dscnt 0x0
	v_fma_f32 v31, -v36, v9, v31
	ds_bpermute_b32 v9, v69, v34
	s_wait_dscnt 0x0
	v_fma_f32 v32, -v36, v9, v32
	ds_bpermute_b32 v9, v70, v34
	s_wait_dscnt 0x0
	v_fma_f32 v33, -v36, v9, v33
	ds_bpermute_b32 v9, v71, v34
	s_wait_dscnt 0x0
	v_fma_f32 v18, -v36, v9, v18
	ds_bpermute_b32 v9, v72, v34
	s_wait_dscnt 0x0
	v_fma_f32 v19, -v36, v9, v19
	ds_bpermute_b32 v9, v73, v34
	s_wait_dscnt 0x0
	v_fma_f32 v20, -v36, v9, v20
	ds_bpermute_b32 v9, v74, v34
	s_wait_dscnt 0x0
	v_fma_f32 v21, -v36, v9, v21
	ds_bpermute_b32 v9, v75, v34
	s_wait_dscnt 0x0
	v_fma_f32 v22, -v36, v9, v22
	ds_bpermute_b32 v9, v76, v34
	s_wait_dscnt 0x0
	v_fma_f32 v23, -v36, v9, v23
	ds_bpermute_b32 v9, v77, v34
	s_wait_dscnt 0x0
	v_fma_f32 v24, -v36, v9, v24
	ds_bpermute_b32 v9, v78, v34
	s_wait_dscnt 0x0
	v_fma_f32 v25, -v36, v9, v25
	ds_bpermute_b32 v9, v80, v35
	s_wait_dscnt 0x0
	v_fma_f32 v10, -v36, v9, v10
	ds_bpermute_b32 v9, v64, v35
	s_wait_dscnt 0x0
	v_fma_f32 v11, -v36, v9, v11
	ds_bpermute_b32 v9, v65, v35
	s_wait_dscnt 0x0
	v_fma_f32 v12, -v36, v9, v12
	ds_bpermute_b32 v9, v66, v35
	s_wait_dscnt 0x0
	v_fma_f32 v13, -v36, v9, v13
	ds_bpermute_b32 v9, v67, v35
	s_wait_dscnt 0x0
	v_fma_f32 v14, -v36, v9, v14
	ds_bpermute_b32 v9, v68, v35
	s_wait_dscnt 0x0
	v_fma_f32 v15, -v36, v9, v15
	ds_bpermute_b32 v9, v69, v35
	s_wait_dscnt 0x0
	v_fma_f32 v16, -v36, v9, v16
	ds_bpermute_b32 v9, v70, v35
	s_wait_dscnt 0x0
	v_fma_f32 v17, -v36, v9, v17
	ds_bpermute_b32 v9, v71, v35
	s_wait_dscnt 0x0
	v_fma_f32 v82, -v36, v9, v82
	ds_bpermute_b32 v9, v72, v35
	s_wait_dscnt 0x0
	v_fma_f32 v83, -v36, v9, v83
	ds_bpermute_b32 v9, v73, v35
	s_wait_dscnt 0x0
	v_fma_f32 v84, -v36, v9, v84
	ds_bpermute_b32 v9, v74, v35
	s_wait_dscnt 0x0
	v_fma_f32 v85, -v36, v9, v85
	ds_bpermute_b32 v9, v75, v35
	s_wait_dscnt 0x0
	v_fma_f32 v86, -v36, v9, v86
	ds_bpermute_b32 v9, v76, v35
	s_wait_dscnt 0x0
	v_fma_f32 v87, -v36, v9, v87
	ds_bpermute_b32 v9, v77, v35
	s_wait_dscnt 0x0
	v_fma_f32 v88, -v36, v9, v88
	ds_bpermute_b32 v9, v78, v35
	s_wait_dscnt 0x0
	v_fma_f32 v89, -v36, v9, v89
	;;#ASMSTART
	;;#ASMEND
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v9, 0, v26, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v9, v9, v10, s0
	ds_bpermute_b32 v34, v63, v9
	s_wait_dscnt 0x0
	v_dual_add_f32 v36, v9, v34 :: v_dual_add_nc_u32 v9, 0x800, v81
	ds_load_2addr_b32 v[34:35], v9 offset1:32
	s_wait_dscnt 0x0
	ds_bpermute_b32 v37, v80, v34
	s_wait_dscnt 0x0
	v_fma_f32 v26, -v36, v37, v26
	ds_bpermute_b32 v37, v64, v34
	s_wait_dscnt 0x0
	v_fma_f32 v27, -v36, v37, v27
	ds_bpermute_b32 v37, v65, v34
	s_wait_dscnt 0x0
	v_fma_f32 v28, -v36, v37, v28
	ds_bpermute_b32 v37, v66, v34
	s_wait_dscnt 0x0
	v_fma_f32 v29, -v36, v37, v29
	ds_bpermute_b32 v37, v67, v34
	s_wait_dscnt 0x0
	v_fma_f32 v30, -v36, v37, v30
	ds_bpermute_b32 v37, v68, v34
	s_wait_dscnt 0x0
	v_fma_f32 v31, -v36, v37, v31
	ds_bpermute_b32 v37, v69, v34
	s_wait_dscnt 0x0
	v_fma_f32 v32, -v36, v37, v32
	ds_bpermute_b32 v37, v70, v34
	s_wait_dscnt 0x0
	v_fma_f32 v33, -v36, v37, v33
	ds_bpermute_b32 v37, v71, v34
	s_wait_dscnt 0x0
	v_fma_f32 v18, -v36, v37, v18
	ds_bpermute_b32 v37, v72, v34
	s_wait_dscnt 0x0
	v_fma_f32 v19, -v36, v37, v19
	ds_bpermute_b32 v37, v73, v34
	s_wait_dscnt 0x0
	v_fma_f32 v20, -v36, v37, v20
	ds_bpermute_b32 v37, v74, v34
	s_wait_dscnt 0x0
	v_fma_f32 v21, -v36, v37, v21
	ds_bpermute_b32 v37, v75, v34
	s_wait_dscnt 0x0
	v_fma_f32 v22, -v36, v37, v22
	ds_bpermute_b32 v37, v76, v34
	s_wait_dscnt 0x0
	v_fma_f32 v23, -v36, v37, v23
	ds_bpermute_b32 v37, v77, v34
	ds_bpermute_b32 v34, v78, v34
	s_wait_dscnt 0x1
	v_fma_f32 v24, -v36, v37, v24
	s_wait_dscnt 0x0
	v_fma_f32 v25, -v36, v34, v25
	ds_bpermute_b32 v34, v80, v35
	s_wait_dscnt 0x0
	v_fma_f32 v10, -v36, v34, v10
	ds_bpermute_b32 v34, v64, v35
	s_wait_dscnt 0x0
	v_fma_f32 v11, -v36, v34, v11
	ds_bpermute_b32 v34, v65, v35
	s_wait_dscnt 0x0
	v_fma_f32 v12, -v36, v34, v12
	ds_bpermute_b32 v34, v66, v35
	s_wait_dscnt 0x0
	v_fma_f32 v13, -v36, v34, v13
	ds_bpermute_b32 v34, v67, v35
	s_wait_dscnt 0x0
	v_fma_f32 v14, -v36, v34, v14
	ds_bpermute_b32 v34, v68, v35
	s_wait_dscnt 0x0
	v_fma_f32 v15, -v36, v34, v15
	ds_bpermute_b32 v34, v69, v35
	s_wait_dscnt 0x0
	v_fma_f32 v16, -v36, v34, v16
	ds_bpermute_b32 v34, v70, v35
	s_wait_dscnt 0x0
	v_fma_f32 v17, -v36, v34, v17
	ds_bpermute_b32 v34, v71, v35
	s_wait_dscnt 0x0
	v_fma_f32 v82, -v36, v34, v82
	ds_bpermute_b32 v34, v72, v35
	s_wait_dscnt 0x0
	v_fma_f32 v83, -v36, v34, v83
	ds_bpermute_b32 v34, v73, v35
	s_wait_dscnt 0x0
	v_fma_f32 v84, -v36, v34, v84
	ds_bpermute_b32 v34, v74, v35
	s_wait_dscnt 0x0
	v_fma_f32 v85, -v36, v34, v85
	ds_bpermute_b32 v34, v75, v35
	s_wait_dscnt 0x0
	v_fma_f32 v86, -v36, v34, v86
	ds_bpermute_b32 v34, v76, v35
	s_wait_dscnt 0x0
	v_fma_f32 v87, -v36, v34, v87
	ds_bpermute_b32 v34, v77, v35
	s_wait_dscnt 0x0
	v_fma_f32 v88, -v36, v34, v88
	ds_bpermute_b32 v34, v78, v35
	s_wait_dscnt 0x0
	v_fma_f32 v89, -v36, v34, v89
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e32 v34, 0, v27, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v34, v34, v11, s0
	ds_bpermute_b32 v35, v63, v34
	s_wait_dscnt 0x0
	v_add_f32_e32 v36, v34, v35
	ds_load_2addr_b32 v[34:35], v9 offset0:64 offset1:96
	s_wait_dscnt 0x0
	ds_bpermute_b32 v37, v80, v34
	s_wait_dscnt 0x0
	v_fma_f32 v26, -v36, v37, v26
	ds_bpermute_b32 v37, v64, v34
	s_wait_dscnt 0x0
	v_fma_f32 v27, -v36, v37, v27
	ds_bpermute_b32 v37, v65, v34
	s_wait_dscnt 0x0
	v_fma_f32 v28, -v36, v37, v28
	ds_bpermute_b32 v37, v66, v34
	s_wait_dscnt 0x0
	v_fma_f32 v29, -v36, v37, v29
	ds_bpermute_b32 v37, v67, v34
	s_wait_dscnt 0x0
	v_fma_f32 v30, -v36, v37, v30
	ds_bpermute_b32 v37, v68, v34
	s_wait_dscnt 0x0
	v_fma_f32 v31, -v36, v37, v31
	ds_bpermute_b32 v37, v69, v34
	s_wait_dscnt 0x0
	v_fma_f32 v32, -v36, v37, v32
	ds_bpermute_b32 v37, v70, v34
	s_wait_dscnt 0x0
	v_fma_f32 v33, -v36, v37, v33
	ds_bpermute_b32 v37, v71, v34
	s_wait_dscnt 0x0
	v_fma_f32 v18, -v36, v37, v18
	ds_bpermute_b32 v37, v72, v34
	s_wait_dscnt 0x0
	v_fma_f32 v19, -v36, v37, v19
	ds_bpermute_b32 v37, v73, v34
	s_wait_dscnt 0x0
	v_fma_f32 v20, -v36, v37, v20
	ds_bpermute_b32 v37, v74, v34
	s_wait_dscnt 0x0
	v_fma_f32 v21, -v36, v37, v21
	ds_bpermute_b32 v37, v75, v34
	s_wait_dscnt 0x0
	v_fma_f32 v22, -v36, v37, v22
	ds_bpermute_b32 v37, v76, v34
	s_wait_dscnt 0x0
	v_fma_f32 v23, -v36, v37, v23
	ds_bpermute_b32 v37, v77, v34
	ds_bpermute_b32 v34, v78, v34
	s_wait_dscnt 0x1
	v_fma_f32 v24, -v36, v37, v24
	s_wait_dscnt 0x0
	v_fma_f32 v25, -v36, v34, v25
	ds_bpermute_b32 v34, v80, v35
	s_wait_dscnt 0x0
	v_fma_f32 v10, -v36, v34, v10
	ds_bpermute_b32 v34, v64, v35
	s_wait_dscnt 0x0
	v_fma_f32 v11, -v36, v34, v11
	ds_bpermute_b32 v34, v65, v35
	s_wait_dscnt 0x0
	v_fma_f32 v12, -v36, v34, v12
	ds_bpermute_b32 v34, v66, v35
	s_wait_dscnt 0x0
	v_fma_f32 v13, -v36, v34, v13
	ds_bpermute_b32 v34, v67, v35
	s_wait_dscnt 0x0
	v_fma_f32 v14, -v36, v34, v14
	ds_bpermute_b32 v34, v68, v35
	s_wait_dscnt 0x0
	v_fma_f32 v15, -v36, v34, v15
	ds_bpermute_b32 v34, v69, v35
	s_wait_dscnt 0x0
	v_fma_f32 v16, -v36, v34, v16
	ds_bpermute_b32 v34, v70, v35
	s_wait_dscnt 0x0
	v_fma_f32 v17, -v36, v34, v17
	ds_bpermute_b32 v34, v71, v35
	s_wait_dscnt 0x0
	v_fma_f32 v82, -v36, v34, v82
	ds_bpermute_b32 v34, v72, v35
	s_wait_dscnt 0x0
	v_fma_f32 v83, -v36, v34, v83
	ds_bpermute_b32 v34, v73, v35
	s_wait_dscnt 0x0
	v_fma_f32 v84, -v36, v34, v84
	ds_bpermute_b32 v34, v74, v35
	s_wait_dscnt 0x0
	v_fma_f32 v85, -v36, v34, v85
	ds_bpermute_b32 v34, v75, v35
	s_wait_dscnt 0x0
	v_fma_f32 v86, -v36, v34, v86
	ds_bpermute_b32 v34, v76, v35
	s_wait_dscnt 0x0
	v_fma_f32 v87, -v36, v34, v87
	ds_bpermute_b32 v34, v77, v35
	s_wait_dscnt 0x0
	v_fma_f32 v88, -v36, v34, v88
	ds_bpermute_b32 v34, v78, v35
	s_wait_dscnt 0x0
	v_fma_f32 v89, -v36, v34, v89
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e32 v34, 0, v28, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v34, v34, v12, s0
	ds_bpermute_b32 v35, v63, v34
	s_wait_dscnt 0x0
	v_add_f32_e32 v36, v34, v35
	ds_load_2addr_b32 v[34:35], v9 offset0:128 offset1:160
	s_wait_dscnt 0x0
	ds_bpermute_b32 v37, v80, v34
	s_wait_dscnt 0x0
	v_fma_f32 v26, -v36, v37, v26
	ds_bpermute_b32 v37, v64, v34
	s_wait_dscnt 0x0
	v_fma_f32 v27, -v36, v37, v27
	ds_bpermute_b32 v37, v65, v34
	s_wait_dscnt 0x0
	v_fma_f32 v28, -v36, v37, v28
	ds_bpermute_b32 v37, v66, v34
	s_wait_dscnt 0x0
	v_fma_f32 v29, -v36, v37, v29
	ds_bpermute_b32 v37, v67, v34
	s_wait_dscnt 0x0
	v_fma_f32 v30, -v36, v37, v30
	ds_bpermute_b32 v37, v68, v34
	s_wait_dscnt 0x0
	v_fma_f32 v31, -v36, v37, v31
	ds_bpermute_b32 v37, v69, v34
	s_wait_dscnt 0x0
	v_fma_f32 v32, -v36, v37, v32
	ds_bpermute_b32 v37, v70, v34
	s_wait_dscnt 0x0
	v_fma_f32 v33, -v36, v37, v33
	ds_bpermute_b32 v37, v71, v34
	s_wait_dscnt 0x0
	v_fma_f32 v18, -v36, v37, v18
	ds_bpermute_b32 v37, v72, v34
	s_wait_dscnt 0x0
	v_fma_f32 v19, -v36, v37, v19
	ds_bpermute_b32 v37, v73, v34
	s_wait_dscnt 0x0
	v_fma_f32 v20, -v36, v37, v20
	ds_bpermute_b32 v37, v74, v34
	s_wait_dscnt 0x0
	v_fma_f32 v21, -v36, v37, v21
	ds_bpermute_b32 v37, v75, v34
	s_wait_dscnt 0x0
	v_fma_f32 v22, -v36, v37, v22
	ds_bpermute_b32 v37, v76, v34
	s_wait_dscnt 0x0
	v_fma_f32 v23, -v36, v37, v23
	ds_bpermute_b32 v37, v77, v34
	ds_bpermute_b32 v34, v78, v34
	s_wait_dscnt 0x1
	v_fma_f32 v24, -v36, v37, v24
	s_wait_dscnt 0x0
	v_fma_f32 v25, -v36, v34, v25
	ds_bpermute_b32 v34, v80, v35
	s_wait_dscnt 0x0
	v_fma_f32 v10, -v36, v34, v10
	ds_bpermute_b32 v34, v64, v35
	s_wait_dscnt 0x0
	v_fma_f32 v11, -v36, v34, v11
	ds_bpermute_b32 v34, v65, v35
	s_wait_dscnt 0x0
	v_fma_f32 v12, -v36, v34, v12
	ds_bpermute_b32 v34, v66, v35
	s_wait_dscnt 0x0
	v_fma_f32 v13, -v36, v34, v13
	ds_bpermute_b32 v34, v67, v35
	s_wait_dscnt 0x0
	v_fma_f32 v14, -v36, v34, v14
	ds_bpermute_b32 v34, v68, v35
	s_wait_dscnt 0x0
	v_fma_f32 v15, -v36, v34, v15
	ds_bpermute_b32 v34, v69, v35
	s_wait_dscnt 0x0
	v_fma_f32 v16, -v36, v34, v16
	ds_bpermute_b32 v34, v70, v35
	s_wait_dscnt 0x0
	v_fma_f32 v17, -v36, v34, v17
	ds_bpermute_b32 v34, v71, v35
	s_wait_dscnt 0x0
	v_fma_f32 v82, -v36, v34, v82
	ds_bpermute_b32 v34, v72, v35
	s_wait_dscnt 0x0
	v_fma_f32 v83, -v36, v34, v83
	ds_bpermute_b32 v34, v73, v35
	s_wait_dscnt 0x0
	v_fma_f32 v84, -v36, v34, v84
	ds_bpermute_b32 v34, v74, v35
	s_wait_dscnt 0x0
	v_fma_f32 v85, -v36, v34, v85
	ds_bpermute_b32 v34, v75, v35
	s_wait_dscnt 0x0
	v_fma_f32 v86, -v36, v34, v86
	ds_bpermute_b32 v34, v76, v35
	s_wait_dscnt 0x0
	v_fma_f32 v87, -v36, v34, v87
	ds_bpermute_b32 v34, v77, v35
	s_wait_dscnt 0x0
	v_fma_f32 v88, -v36, v34, v88
	ds_bpermute_b32 v34, v78, v35
	s_wait_dscnt 0x0
	v_fma_f32 v89, -v36, v34, v89
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e32 v34, 0, v29, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v34, v34, v13, s0
	ds_bpermute_b32 v35, v63, v34
	s_wait_dscnt 0x0
	v_add_f32_e32 v36, v34, v35
	ds_load_2addr_b32 v[34:35], v9 offset0:192 offset1:224
	s_wait_dscnt 0x0
	ds_bpermute_b32 v9, v80, v34
	s_wait_dscnt 0x0
	v_fma_f32 v26, -v36, v9, v26
	ds_bpermute_b32 v9, v64, v34
	s_wait_dscnt 0x0
	v_fma_f32 v27, -v36, v9, v27
	ds_bpermute_b32 v9, v65, v34
	s_wait_dscnt 0x0
	v_fma_f32 v28, -v36, v9, v28
	ds_bpermute_b32 v9, v66, v34
	s_wait_dscnt 0x0
	v_fma_f32 v29, -v36, v9, v29
	ds_bpermute_b32 v9, v67, v34
	s_wait_dscnt 0x0
	v_fma_f32 v30, -v36, v9, v30
	ds_bpermute_b32 v9, v68, v34
	s_wait_dscnt 0x0
	v_fma_f32 v31, -v36, v9, v31
	ds_bpermute_b32 v9, v69, v34
	s_wait_dscnt 0x0
	v_fma_f32 v32, -v36, v9, v32
	ds_bpermute_b32 v9, v70, v34
	s_wait_dscnt 0x0
	v_fma_f32 v33, -v36, v9, v33
	ds_bpermute_b32 v9, v71, v34
	s_wait_dscnt 0x0
	v_fma_f32 v18, -v36, v9, v18
	ds_bpermute_b32 v9, v72, v34
	s_wait_dscnt 0x0
	v_fma_f32 v19, -v36, v9, v19
	ds_bpermute_b32 v9, v73, v34
	s_wait_dscnt 0x0
	v_fma_f32 v20, -v36, v9, v20
	ds_bpermute_b32 v9, v74, v34
	s_wait_dscnt 0x0
	v_fma_f32 v21, -v36, v9, v21
	ds_bpermute_b32 v9, v75, v34
	s_wait_dscnt 0x0
	v_fma_f32 v22, -v36, v9, v22
	ds_bpermute_b32 v9, v76, v34
	s_wait_dscnt 0x0
	v_fma_f32 v23, -v36, v9, v23
	ds_bpermute_b32 v9, v77, v34
	s_wait_dscnt 0x0
	v_fma_f32 v24, -v36, v9, v24
	ds_bpermute_b32 v9, v78, v34
	s_wait_dscnt 0x0
	v_fma_f32 v25, -v36, v9, v25
	ds_bpermute_b32 v9, v80, v35
	s_wait_dscnt 0x0
	v_fma_f32 v10, -v36, v9, v10
	ds_bpermute_b32 v9, v64, v35
	s_wait_dscnt 0x0
	v_fma_f32 v11, -v36, v9, v11
	ds_bpermute_b32 v9, v65, v35
	s_wait_dscnt 0x0
	v_fma_f32 v12, -v36, v9, v12
	ds_bpermute_b32 v9, v66, v35
	s_wait_dscnt 0x0
	v_fma_f32 v13, -v36, v9, v13
	ds_bpermute_b32 v9, v67, v35
	s_wait_dscnt 0x0
	v_fma_f32 v14, -v36, v9, v14
	ds_bpermute_b32 v9, v68, v35
	s_wait_dscnt 0x0
	v_fma_f32 v15, -v36, v9, v15
	ds_bpermute_b32 v9, v69, v35
	s_wait_dscnt 0x0
	v_fma_f32 v16, -v36, v9, v16
	ds_bpermute_b32 v9, v70, v35
	s_wait_dscnt 0x0
	v_fma_f32 v17, -v36, v9, v17
	ds_bpermute_b32 v9, v71, v35
	s_wait_dscnt 0x0
	v_fma_f32 v82, -v36, v9, v82
	ds_bpermute_b32 v9, v72, v35
	s_wait_dscnt 0x0
	v_fma_f32 v83, -v36, v9, v83
	ds_bpermute_b32 v9, v73, v35
	s_wait_dscnt 0x0
	v_fma_f32 v84, -v36, v9, v84
	ds_bpermute_b32 v9, v74, v35
	s_wait_dscnt 0x0
	v_fma_f32 v85, -v36, v9, v85
	ds_bpermute_b32 v9, v75, v35
	s_wait_dscnt 0x0
	v_fma_f32 v86, -v36, v9, v86
	ds_bpermute_b32 v9, v76, v35
	s_wait_dscnt 0x0
	v_fma_f32 v87, -v36, v9, v87
	ds_bpermute_b32 v9, v77, v35
	s_wait_dscnt 0x0
	v_fma_f32 v88, -v36, v9, v88
	ds_bpermute_b32 v9, v78, v35
	s_wait_dscnt 0x0
	v_fma_f32 v89, -v36, v9, v89
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e32 v9, 0, v30, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v9, v9, v14, s0
	ds_bpermute_b32 v34, v63, v9
	s_wait_dscnt 0x0
	v_dual_add_f32 v36, v9, v34 :: v_dual_add_nc_u32 v9, 0xc00, v81
	ds_load_2addr_b32 v[34:35], v9 offset1:32
	s_wait_dscnt 0x0
	ds_bpermute_b32 v37, v80, v34
	s_wait_dscnt 0x0
	v_fma_f32 v26, -v36, v37, v26
	ds_bpermute_b32 v37, v64, v34
	s_wait_dscnt 0x0
	v_fma_f32 v27, -v36, v37, v27
	ds_bpermute_b32 v37, v65, v34
	s_wait_dscnt 0x0
	v_fma_f32 v28, -v36, v37, v28
	ds_bpermute_b32 v37, v66, v34
	s_wait_dscnt 0x0
	v_fma_f32 v29, -v36, v37, v29
	ds_bpermute_b32 v37, v67, v34
	s_wait_dscnt 0x0
	v_fma_f32 v30, -v36, v37, v30
	ds_bpermute_b32 v37, v68, v34
	s_wait_dscnt 0x0
	v_fma_f32 v31, -v36, v37, v31
	ds_bpermute_b32 v37, v69, v34
	s_wait_dscnt 0x0
	v_fma_f32 v32, -v36, v37, v32
	ds_bpermute_b32 v37, v70, v34
	s_wait_dscnt 0x0
	v_fma_f32 v33, -v36, v37, v33
	ds_bpermute_b32 v37, v71, v34
	s_wait_dscnt 0x0
	v_fma_f32 v18, -v36, v37, v18
	ds_bpermute_b32 v37, v72, v34
	s_wait_dscnt 0x0
	v_fma_f32 v19, -v36, v37, v19
	ds_bpermute_b32 v37, v73, v34
	s_wait_dscnt 0x0
	v_fma_f32 v20, -v36, v37, v20
	ds_bpermute_b32 v37, v74, v34
	s_wait_dscnt 0x0
	v_fma_f32 v21, -v36, v37, v21
	ds_bpermute_b32 v37, v75, v34
	s_wait_dscnt 0x0
	v_fma_f32 v22, -v36, v37, v22
	ds_bpermute_b32 v37, v76, v34
	s_wait_dscnt 0x0
	v_fma_f32 v23, -v36, v37, v23
	ds_bpermute_b32 v37, v77, v34
	ds_bpermute_b32 v34, v78, v34
	s_wait_dscnt 0x1
	v_fma_f32 v24, -v36, v37, v24
	s_wait_dscnt 0x0
	v_fma_f32 v25, -v36, v34, v25
	ds_bpermute_b32 v34, v80, v35
	s_wait_dscnt 0x0
	v_fma_f32 v10, -v36, v34, v10
	ds_bpermute_b32 v34, v64, v35
	s_wait_dscnt 0x0
	v_fma_f32 v11, -v36, v34, v11
	ds_bpermute_b32 v34, v65, v35
	s_wait_dscnt 0x0
	v_fma_f32 v12, -v36, v34, v12
	ds_bpermute_b32 v34, v66, v35
	s_wait_dscnt 0x0
	v_fma_f32 v13, -v36, v34, v13
	ds_bpermute_b32 v34, v67, v35
	s_wait_dscnt 0x0
	v_fma_f32 v14, -v36, v34, v14
	ds_bpermute_b32 v34, v68, v35
	s_wait_dscnt 0x0
	v_fma_f32 v15, -v36, v34, v15
	ds_bpermute_b32 v34, v69, v35
	s_wait_dscnt 0x0
	v_fma_f32 v16, -v36, v34, v16
	ds_bpermute_b32 v34, v70, v35
	s_wait_dscnt 0x0
	v_fma_f32 v17, -v36, v34, v17
	ds_bpermute_b32 v34, v71, v35
	s_wait_dscnt 0x0
	v_fma_f32 v82, -v36, v34, v82
	ds_bpermute_b32 v34, v72, v35
	s_wait_dscnt 0x0
	v_fma_f32 v83, -v36, v34, v83
	ds_bpermute_b32 v34, v73, v35
	s_wait_dscnt 0x0
	v_fma_f32 v84, -v36, v34, v84
	ds_bpermute_b32 v34, v74, v35
	s_wait_dscnt 0x0
	v_fma_f32 v85, -v36, v34, v85
	ds_bpermute_b32 v34, v75, v35
	s_wait_dscnt 0x0
	v_fma_f32 v86, -v36, v34, v86
	ds_bpermute_b32 v34, v76, v35
	s_wait_dscnt 0x0
	v_fma_f32 v87, -v36, v34, v87
	ds_bpermute_b32 v34, v77, v35
	s_wait_dscnt 0x0
	v_fma_f32 v88, -v36, v34, v88
	ds_bpermute_b32 v34, v78, v35
	s_wait_dscnt 0x0
	v_fma_f32 v89, -v36, v34, v89
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e32 v34, 0, v31, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v34, v34, v15, s0
	ds_bpermute_b32 v35, v63, v34
	s_wait_dscnt 0x0
	v_add_f32_e32 v36, v34, v35
	ds_load_2addr_b32 v[34:35], v9 offset0:64 offset1:96
	s_wait_dscnt 0x0
	ds_bpermute_b32 v37, v80, v34
	s_wait_dscnt 0x0
	v_fma_f32 v26, -v36, v37, v26
	ds_bpermute_b32 v37, v64, v34
	s_wait_dscnt 0x0
	v_fma_f32 v27, -v36, v37, v27
	ds_bpermute_b32 v37, v65, v34
	s_wait_dscnt 0x0
	v_fma_f32 v28, -v36, v37, v28
	ds_bpermute_b32 v37, v66, v34
	s_wait_dscnt 0x0
	v_fma_f32 v29, -v36, v37, v29
	ds_bpermute_b32 v37, v67, v34
	s_wait_dscnt 0x0
	v_fma_f32 v30, -v36, v37, v30
	ds_bpermute_b32 v37, v68, v34
	s_wait_dscnt 0x0
	v_fma_f32 v31, -v36, v37, v31
	ds_bpermute_b32 v37, v69, v34
	s_wait_dscnt 0x0
	v_fma_f32 v32, -v36, v37, v32
	ds_bpermute_b32 v37, v70, v34
	s_wait_dscnt 0x0
	v_fma_f32 v33, -v36, v37, v33
	ds_bpermute_b32 v37, v71, v34
	s_wait_dscnt 0x0
	v_fma_f32 v18, -v36, v37, v18
	ds_bpermute_b32 v37, v72, v34
	s_wait_dscnt 0x0
	v_fma_f32 v19, -v36, v37, v19
	ds_bpermute_b32 v37, v73, v34
	s_wait_dscnt 0x0
	v_fma_f32 v20, -v36, v37, v20
	ds_bpermute_b32 v37, v74, v34
	s_wait_dscnt 0x0
	v_fma_f32 v21, -v36, v37, v21
	ds_bpermute_b32 v37, v75, v34
	s_wait_dscnt 0x0
	v_fma_f32 v22, -v36, v37, v22
	ds_bpermute_b32 v37, v76, v34
	s_wait_dscnt 0x0
	v_fma_f32 v23, -v36, v37, v23
	ds_bpermute_b32 v37, v77, v34
	ds_bpermute_b32 v34, v78, v34
	s_wait_dscnt 0x1
	v_fma_f32 v24, -v36, v37, v24
	s_wait_dscnt 0x0
	v_fma_f32 v25, -v36, v34, v25
	ds_bpermute_b32 v34, v80, v35
	s_wait_dscnt 0x0
	v_fma_f32 v10, -v36, v34, v10
	ds_bpermute_b32 v34, v64, v35
	s_wait_dscnt 0x0
	v_fma_f32 v11, -v36, v34, v11
	ds_bpermute_b32 v34, v65, v35
	s_wait_dscnt 0x0
	v_fma_f32 v12, -v36, v34, v12
	ds_bpermute_b32 v34, v66, v35
	s_wait_dscnt 0x0
	v_fma_f32 v13, -v36, v34, v13
	ds_bpermute_b32 v34, v67, v35
	s_wait_dscnt 0x0
	v_fma_f32 v14, -v36, v34, v14
	ds_bpermute_b32 v34, v68, v35
	s_wait_dscnt 0x0
	v_fma_f32 v15, -v36, v34, v15
	ds_bpermute_b32 v34, v69, v35
	s_wait_dscnt 0x0
	v_fma_f32 v16, -v36, v34, v16
	ds_bpermute_b32 v34, v70, v35
	s_wait_dscnt 0x0
	v_fma_f32 v17, -v36, v34, v17
	ds_bpermute_b32 v34, v71, v35
	s_wait_dscnt 0x0
	v_fma_f32 v82, -v36, v34, v82
	ds_bpermute_b32 v34, v72, v35
	s_wait_dscnt 0x0
	v_fma_f32 v83, -v36, v34, v83
	ds_bpermute_b32 v34, v73, v35
	s_wait_dscnt 0x0
	v_fma_f32 v84, -v36, v34, v84
	ds_bpermute_b32 v34, v74, v35
	s_wait_dscnt 0x0
	v_fma_f32 v85, -v36, v34, v85
	ds_bpermute_b32 v34, v75, v35
	s_wait_dscnt 0x0
	v_fma_f32 v86, -v36, v34, v86
	ds_bpermute_b32 v34, v76, v35
	s_wait_dscnt 0x0
	v_fma_f32 v87, -v36, v34, v87
	ds_bpermute_b32 v34, v77, v35
	s_wait_dscnt 0x0
	v_fma_f32 v88, -v36, v34, v88
	ds_bpermute_b32 v34, v78, v35
	s_wait_dscnt 0x0
	v_fma_f32 v89, -v36, v34, v89
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e32 v34, 0, v32, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v34, v34, v16, s0
	ds_bpermute_b32 v35, v63, v34
	s_wait_dscnt 0x0
	v_add_f32_e32 v36, v34, v35
	ds_load_2addr_b32 v[34:35], v9 offset0:128 offset1:160
	s_wait_dscnt 0x0
	ds_bpermute_b32 v37, v80, v34
	s_wait_dscnt 0x0
	v_fma_f32 v26, -v36, v37, v26
	ds_bpermute_b32 v37, v64, v34
	s_wait_dscnt 0x0
	v_fma_f32 v27, -v36, v37, v27
	ds_bpermute_b32 v37, v65, v34
	s_wait_dscnt 0x0
	v_fma_f32 v28, -v36, v37, v28
	ds_bpermute_b32 v37, v66, v34
	s_wait_dscnt 0x0
	v_fma_f32 v29, -v36, v37, v29
	ds_bpermute_b32 v37, v67, v34
	s_wait_dscnt 0x0
	v_fma_f32 v30, -v36, v37, v30
	ds_bpermute_b32 v37, v68, v34
	s_wait_dscnt 0x0
	v_fma_f32 v31, -v36, v37, v31
	ds_bpermute_b32 v37, v69, v34
	s_wait_dscnt 0x0
	v_fma_f32 v32, -v36, v37, v32
	ds_bpermute_b32 v37, v70, v34
	s_wait_dscnt 0x0
	v_fma_f32 v33, -v36, v37, v33
	ds_bpermute_b32 v37, v71, v34
	s_wait_dscnt 0x0
	v_fma_f32 v18, -v36, v37, v18
	ds_bpermute_b32 v37, v72, v34
	s_wait_dscnt 0x0
	v_fma_f32 v19, -v36, v37, v19
	ds_bpermute_b32 v37, v73, v34
	s_wait_dscnt 0x0
	v_fma_f32 v20, -v36, v37, v20
	ds_bpermute_b32 v37, v74, v34
	s_wait_dscnt 0x0
	v_fma_f32 v21, -v36, v37, v21
	ds_bpermute_b32 v37, v75, v34
	s_wait_dscnt 0x0
	v_fma_f32 v22, -v36, v37, v22
	ds_bpermute_b32 v37, v76, v34
	s_wait_dscnt 0x0
	v_fma_f32 v23, -v36, v37, v23
	ds_bpermute_b32 v37, v77, v34
	ds_bpermute_b32 v34, v78, v34
	s_wait_dscnt 0x1
	v_fma_f32 v24, -v36, v37, v24
	s_wait_dscnt 0x0
	v_fma_f32 v25, -v36, v34, v25
	ds_bpermute_b32 v34, v80, v35
	s_wait_dscnt 0x0
	v_fma_f32 v10, -v36, v34, v10
	ds_bpermute_b32 v34, v64, v35
	s_wait_dscnt 0x0
	v_fma_f32 v11, -v36, v34, v11
	ds_bpermute_b32 v34, v65, v35
	s_wait_dscnt 0x0
	v_fma_f32 v12, -v36, v34, v12
	ds_bpermute_b32 v34, v66, v35
	s_wait_dscnt 0x0
	v_fma_f32 v13, -v36, v34, v13
	ds_bpermute_b32 v34, v67, v35
	s_wait_dscnt 0x0
	v_fma_f32 v14, -v36, v34, v14
	ds_bpermute_b32 v34, v68, v35
	s_wait_dscnt 0x0
	v_fma_f32 v15, -v36, v34, v15
	ds_bpermute_b32 v34, v69, v35
	s_wait_dscnt 0x0
	v_fma_f32 v16, -v36, v34, v16
	ds_bpermute_b32 v34, v70, v35
	s_wait_dscnt 0x0
	v_fma_f32 v17, -v36, v34, v17
	ds_bpermute_b32 v34, v71, v35
	s_wait_dscnt 0x0
	v_fma_f32 v82, -v36, v34, v82
	ds_bpermute_b32 v34, v72, v35
	s_wait_dscnt 0x0
	v_fma_f32 v83, -v36, v34, v83
	ds_bpermute_b32 v34, v73, v35
	s_wait_dscnt 0x0
	v_fma_f32 v84, -v36, v34, v84
	ds_bpermute_b32 v34, v74, v35
	s_wait_dscnt 0x0
	v_fma_f32 v85, -v36, v34, v85
	ds_bpermute_b32 v34, v75, v35
	s_wait_dscnt 0x0
	v_fma_f32 v86, -v36, v34, v86
	ds_bpermute_b32 v34, v76, v35
	s_wait_dscnt 0x0
	v_fma_f32 v87, -v36, v34, v87
	ds_bpermute_b32 v34, v77, v35
	s_wait_dscnt 0x0
	v_fma_f32 v88, -v36, v34, v88
	ds_bpermute_b32 v34, v78, v35
	s_wait_dscnt 0x0
	v_fma_f32 v89, -v36, v34, v89
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e32 v34, 0, v33, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v34, v34, v17, s0
	ds_bpermute_b32 v35, v63, v34
	s_wait_dscnt 0x0
	v_add_f32_e32 v36, v34, v35
	ds_load_2addr_b32 v[34:35], v9 offset0:192 offset1:224
	s_wait_dscnt 0x0
	ds_bpermute_b32 v9, v80, v34
	s_wait_dscnt 0x0
	v_fma_f32 v26, -v36, v9, v26
	ds_bpermute_b32 v9, v64, v34
	s_wait_dscnt 0x0
	v_fma_f32 v27, -v36, v9, v27
	ds_bpermute_b32 v9, v65, v34
	s_wait_dscnt 0x0
	v_fma_f32 v28, -v36, v9, v28
	ds_bpermute_b32 v9, v66, v34
	s_wait_dscnt 0x0
	v_fma_f32 v29, -v36, v9, v29
	ds_bpermute_b32 v9, v67, v34
	s_wait_dscnt 0x0
	v_fma_f32 v30, -v36, v9, v30
	ds_bpermute_b32 v9, v68, v34
	s_wait_dscnt 0x0
	v_fma_f32 v31, -v36, v9, v31
	ds_bpermute_b32 v9, v69, v34
	s_wait_dscnt 0x0
	v_fma_f32 v32, -v36, v9, v32
	ds_bpermute_b32 v9, v70, v34
	s_wait_dscnt 0x0
	v_fma_f32 v33, -v36, v9, v33
	ds_bpermute_b32 v9, v71, v34
	s_wait_dscnt 0x0
	v_fma_f32 v18, -v36, v9, v18
	ds_bpermute_b32 v9, v72, v34
	s_wait_dscnt 0x0
	v_fma_f32 v19, -v36, v9, v19
	ds_bpermute_b32 v9, v73, v34
	s_wait_dscnt 0x0
	v_fma_f32 v20, -v36, v9, v20
	ds_bpermute_b32 v9, v74, v34
	s_wait_dscnt 0x0
	v_fma_f32 v21, -v36, v9, v21
	ds_bpermute_b32 v9, v75, v34
	s_wait_dscnt 0x0
	v_fma_f32 v22, -v36, v9, v22
	ds_bpermute_b32 v9, v76, v34
	s_wait_dscnt 0x0
	v_fma_f32 v23, -v36, v9, v23
	ds_bpermute_b32 v9, v77, v34
	s_wait_dscnt 0x0
	v_fma_f32 v24, -v36, v9, v24
	ds_bpermute_b32 v9, v78, v34
	s_wait_dscnt 0x0
	v_fma_f32 v25, -v36, v9, v25
	ds_bpermute_b32 v9, v80, v35
	s_wait_dscnt 0x0
	v_fma_f32 v10, -v36, v9, v10
	ds_bpermute_b32 v9, v64, v35
	s_wait_dscnt 0x0
	v_fma_f32 v11, -v36, v9, v11
	ds_bpermute_b32 v9, v65, v35
	s_wait_dscnt 0x0
	v_fma_f32 v12, -v36, v9, v12
	ds_bpermute_b32 v9, v66, v35
	s_wait_dscnt 0x0
	v_fma_f32 v13, -v36, v9, v13
	ds_bpermute_b32 v9, v67, v35
	s_wait_dscnt 0x0
	v_fma_f32 v14, -v36, v9, v14
	ds_bpermute_b32 v9, v68, v35
	s_wait_dscnt 0x0
	v_fma_f32 v15, -v36, v9, v15
	ds_bpermute_b32 v9, v69, v35
	s_wait_dscnt 0x0
	v_fma_f32 v16, -v36, v9, v16
	ds_bpermute_b32 v9, v70, v35
	s_wait_dscnt 0x0
	v_fma_f32 v17, -v36, v9, v17
	ds_bpermute_b32 v9, v71, v35
	s_wait_dscnt 0x0
	v_fma_f32 v82, -v36, v9, v82
	ds_bpermute_b32 v9, v72, v35
	s_wait_dscnt 0x0
	v_fma_f32 v83, -v36, v9, v83
	ds_bpermute_b32 v9, v73, v35
	s_wait_dscnt 0x0
	v_fma_f32 v84, -v36, v9, v84
	ds_bpermute_b32 v9, v74, v35
	s_wait_dscnt 0x0
	v_fma_f32 v85, -v36, v9, v85
	ds_bpermute_b32 v9, v75, v35
	s_wait_dscnt 0x0
	v_fma_f32 v86, -v36, v9, v86
	ds_bpermute_b32 v9, v76, v35
	s_wait_dscnt 0x0
	v_fma_f32 v87, -v36, v9, v87
	ds_bpermute_b32 v9, v77, v35
	s_wait_dscnt 0x0
	v_fma_f32 v88, -v36, v9, v88
	ds_bpermute_b32 v9, v78, v35
	s_wait_dscnt 0x0
	v_fma_f32 v89, -v36, v9, v89
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v9, 0, v18, s1
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v9, v9, v82, s2
	ds_bpermute_b32 v34, v63, v9
	s_wait_dscnt 0x0
	v_dual_add_f32 v36, v9, v34 :: v_dual_add_nc_u32 v9, 0x1000, v81
	ds_load_2addr_b32 v[34:35], v9 offset1:32
	s_wait_dscnt 0x0
	ds_bpermute_b32 v37, v80, v34
	s_wait_dscnt 0x0
	v_fma_f32 v26, -v36, v37, v26
	ds_bpermute_b32 v37, v64, v34
	s_wait_dscnt 0x0
	v_fma_f32 v27, -v36, v37, v27
	ds_bpermute_b32 v37, v65, v34
	s_wait_dscnt 0x0
	v_fma_f32 v28, -v36, v37, v28
	ds_bpermute_b32 v37, v66, v34
	s_wait_dscnt 0x0
	v_fma_f32 v29, -v36, v37, v29
	ds_bpermute_b32 v37, v67, v34
	s_wait_dscnt 0x0
	v_fma_f32 v30, -v36, v37, v30
	ds_bpermute_b32 v37, v68, v34
	s_wait_dscnt 0x0
	v_fma_f32 v31, -v36, v37, v31
	ds_bpermute_b32 v37, v69, v34
	s_wait_dscnt 0x0
	v_fma_f32 v32, -v36, v37, v32
	ds_bpermute_b32 v37, v70, v34
	s_wait_dscnt 0x0
	v_fma_f32 v33, -v36, v37, v33
	ds_bpermute_b32 v37, v71, v34
	s_wait_dscnt 0x0
	v_fma_f32 v18, -v36, v37, v18
	ds_bpermute_b32 v37, v72, v34
	s_wait_dscnt 0x0
	v_fma_f32 v19, -v36, v37, v19
	ds_bpermute_b32 v37, v73, v34
	s_wait_dscnt 0x0
	v_fma_f32 v20, -v36, v37, v20
	ds_bpermute_b32 v37, v74, v34
	s_wait_dscnt 0x0
	v_fma_f32 v21, -v36, v37, v21
	ds_bpermute_b32 v37, v75, v34
	s_wait_dscnt 0x0
	v_fma_f32 v22, -v36, v37, v22
	ds_bpermute_b32 v37, v76, v34
	s_wait_dscnt 0x0
	v_fma_f32 v23, -v36, v37, v23
	ds_bpermute_b32 v37, v77, v34
	ds_bpermute_b32 v34, v78, v34
	s_wait_dscnt 0x1
	v_fma_f32 v24, -v36, v37, v24
	s_wait_dscnt 0x0
	v_fma_f32 v25, -v36, v34, v25
	ds_bpermute_b32 v34, v80, v35
	s_wait_dscnt 0x0
	v_fma_f32 v10, -v36, v34, v10
	ds_bpermute_b32 v34, v64, v35
	s_wait_dscnt 0x0
	v_fma_f32 v11, -v36, v34, v11
	ds_bpermute_b32 v34, v65, v35
	s_wait_dscnt 0x0
	v_fma_f32 v12, -v36, v34, v12
	ds_bpermute_b32 v34, v66, v35
	s_wait_dscnt 0x0
	v_fma_f32 v13, -v36, v34, v13
	ds_bpermute_b32 v34, v67, v35
	s_wait_dscnt 0x0
	v_fma_f32 v14, -v36, v34, v14
	ds_bpermute_b32 v34, v68, v35
	s_wait_dscnt 0x0
	v_fma_f32 v15, -v36, v34, v15
	ds_bpermute_b32 v34, v69, v35
	s_wait_dscnt 0x0
	v_fma_f32 v16, -v36, v34, v16
	ds_bpermute_b32 v34, v70, v35
	s_wait_dscnt 0x0
	v_fma_f32 v17, -v36, v34, v17
	ds_bpermute_b32 v34, v71, v35
	s_wait_dscnt 0x0
	v_fma_f32 v82, -v36, v34, v82
	ds_bpermute_b32 v34, v72, v35
	s_wait_dscnt 0x0
	v_fma_f32 v83, -v36, v34, v83
	ds_bpermute_b32 v34, v73, v35
	s_wait_dscnt 0x0
	v_fma_f32 v84, -v36, v34, v84
	ds_bpermute_b32 v34, v74, v35
	s_wait_dscnt 0x0
	v_fma_f32 v85, -v36, v34, v85
	ds_bpermute_b32 v34, v75, v35
	s_wait_dscnt 0x0
	v_fma_f32 v86, -v36, v34, v86
	ds_bpermute_b32 v34, v76, v35
	s_wait_dscnt 0x0
	v_fma_f32 v87, -v36, v34, v87
	ds_bpermute_b32 v34, v77, v35
	s_wait_dscnt 0x0
	v_fma_f32 v88, -v36, v34, v88
	ds_bpermute_b32 v34, v78, v35
	s_wait_dscnt 0x0
	v_fma_f32 v89, -v36, v34, v89
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v34, 0, v19, s1
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v34, v34, v83, s2
	ds_bpermute_b32 v35, v63, v34
	s_wait_dscnt 0x0
	v_add_f32_e32 v36, v34, v35
	ds_load_2addr_b32 v[34:35], v9 offset0:64 offset1:96
	s_wait_dscnt 0x0
	ds_bpermute_b32 v37, v80, v34
	s_wait_dscnt 0x0
	v_fma_f32 v26, -v36, v37, v26
	ds_bpermute_b32 v37, v64, v34
	s_wait_dscnt 0x0
	v_fma_f32 v27, -v36, v37, v27
	ds_bpermute_b32 v37, v65, v34
	s_wait_dscnt 0x0
	v_fma_f32 v28, -v36, v37, v28
	ds_bpermute_b32 v37, v66, v34
	s_wait_dscnt 0x0
	v_fma_f32 v29, -v36, v37, v29
	ds_bpermute_b32 v37, v67, v34
	s_wait_dscnt 0x0
	v_fma_f32 v30, -v36, v37, v30
	ds_bpermute_b32 v37, v68, v34
	s_wait_dscnt 0x0
	v_fma_f32 v31, -v36, v37, v31
	ds_bpermute_b32 v37, v69, v34
	s_wait_dscnt 0x0
	v_fma_f32 v32, -v36, v37, v32
	ds_bpermute_b32 v37, v70, v34
	s_wait_dscnt 0x0
	v_fma_f32 v33, -v36, v37, v33
	ds_bpermute_b32 v37, v71, v34
	s_wait_dscnt 0x0
	v_fma_f32 v18, -v36, v37, v18
	ds_bpermute_b32 v37, v72, v34
	s_wait_dscnt 0x0
	v_fma_f32 v19, -v36, v37, v19
	ds_bpermute_b32 v37, v73, v34
	s_wait_dscnt 0x0
	v_fma_f32 v20, -v36, v37, v20
	ds_bpermute_b32 v37, v74, v34
	s_wait_dscnt 0x0
	v_fma_f32 v21, -v36, v37, v21
	ds_bpermute_b32 v37, v75, v34
	s_wait_dscnt 0x0
	v_fma_f32 v22, -v36, v37, v22
	ds_bpermute_b32 v37, v76, v34
	s_wait_dscnt 0x0
	v_fma_f32 v23, -v36, v37, v23
	ds_bpermute_b32 v37, v77, v34
	ds_bpermute_b32 v34, v78, v34
	s_wait_dscnt 0x1
	v_fma_f32 v24, -v36, v37, v24
	s_wait_dscnt 0x0
	v_fma_f32 v25, -v36, v34, v25
	ds_bpermute_b32 v34, v80, v35
	s_wait_dscnt 0x0
	v_fma_f32 v10, -v36, v34, v10
	ds_bpermute_b32 v34, v64, v35
	s_wait_dscnt 0x0
	v_fma_f32 v11, -v36, v34, v11
	ds_bpermute_b32 v34, v65, v35
	s_wait_dscnt 0x0
	v_fma_f32 v12, -v36, v34, v12
	ds_bpermute_b32 v34, v66, v35
	s_wait_dscnt 0x0
	v_fma_f32 v13, -v36, v34, v13
	ds_bpermute_b32 v34, v67, v35
	s_wait_dscnt 0x0
	v_fma_f32 v14, -v36, v34, v14
	ds_bpermute_b32 v34, v68, v35
	s_wait_dscnt 0x0
	v_fma_f32 v15, -v36, v34, v15
	ds_bpermute_b32 v34, v69, v35
	s_wait_dscnt 0x0
	v_fma_f32 v16, -v36, v34, v16
	ds_bpermute_b32 v34, v70, v35
	s_wait_dscnt 0x0
	v_fma_f32 v17, -v36, v34, v17
	ds_bpermute_b32 v34, v71, v35
	s_wait_dscnt 0x0
	v_fma_f32 v82, -v36, v34, v82
	ds_bpermute_b32 v34, v72, v35
	s_wait_dscnt 0x0
	v_fma_f32 v83, -v36, v34, v83
	ds_bpermute_b32 v34, v73, v35
	s_wait_dscnt 0x0
	v_fma_f32 v84, -v36, v34, v84
	ds_bpermute_b32 v34, v74, v35
	s_wait_dscnt 0x0
	v_fma_f32 v85, -v36, v34, v85
	ds_bpermute_b32 v34, v75, v35
	s_wait_dscnt 0x0
	v_fma_f32 v86, -v36, v34, v86
	ds_bpermute_b32 v34, v76, v35
	s_wait_dscnt 0x0
	v_fma_f32 v87, -v36, v34, v87
	ds_bpermute_b32 v34, v77, v35
	s_wait_dscnt 0x0
	v_fma_f32 v88, -v36, v34, v88
	ds_bpermute_b32 v34, v78, v35
	s_wait_dscnt 0x0
	v_fma_f32 v89, -v36, v34, v89
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v34, 0, v20, s1
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v34, v34, v84, s2
	ds_bpermute_b32 v35, v63, v34
	s_wait_dscnt 0x0
	v_add_f32_e32 v36, v34, v35
	ds_load_2addr_b32 v[34:35], v9 offset0:128 offset1:160
	s_wait_dscnt 0x0
	ds_bpermute_b32 v37, v80, v34
	s_wait_dscnt 0x0
	v_fma_f32 v26, -v36, v37, v26
	ds_bpermute_b32 v37, v64, v34
	s_wait_dscnt 0x0
	v_fma_f32 v27, -v36, v37, v27
	ds_bpermute_b32 v37, v65, v34
	s_wait_dscnt 0x0
	v_fma_f32 v28, -v36, v37, v28
	ds_bpermute_b32 v37, v66, v34
	s_wait_dscnt 0x0
	v_fma_f32 v29, -v36, v37, v29
	ds_bpermute_b32 v37, v67, v34
	s_wait_dscnt 0x0
	v_fma_f32 v30, -v36, v37, v30
	ds_bpermute_b32 v37, v68, v34
	s_wait_dscnt 0x0
	v_fma_f32 v31, -v36, v37, v31
	ds_bpermute_b32 v37, v69, v34
	s_wait_dscnt 0x0
	v_fma_f32 v32, -v36, v37, v32
	ds_bpermute_b32 v37, v70, v34
	s_wait_dscnt 0x0
	v_fma_f32 v33, -v36, v37, v33
	ds_bpermute_b32 v37, v71, v34
	s_wait_dscnt 0x0
	v_fma_f32 v18, -v36, v37, v18
	ds_bpermute_b32 v37, v72, v34
	s_wait_dscnt 0x0
	v_fma_f32 v19, -v36, v37, v19
	ds_bpermute_b32 v37, v73, v34
	s_wait_dscnt 0x0
	v_fma_f32 v20, -v36, v37, v20
	ds_bpermute_b32 v37, v74, v34
	s_wait_dscnt 0x0
	v_fma_f32 v21, -v36, v37, v21
	ds_bpermute_b32 v37, v75, v34
	s_wait_dscnt 0x0
	v_fma_f32 v22, -v36, v37, v22
	ds_bpermute_b32 v37, v76, v34
	s_wait_dscnt 0x0
	v_fma_f32 v23, -v36, v37, v23
	ds_bpermute_b32 v37, v77, v34
	ds_bpermute_b32 v34, v78, v34
	s_wait_dscnt 0x1
	v_fma_f32 v24, -v36, v37, v24
	s_wait_dscnt 0x0
	v_fma_f32 v25, -v36, v34, v25
	ds_bpermute_b32 v34, v80, v35
	s_wait_dscnt 0x0
	v_fma_f32 v10, -v36, v34, v10
	ds_bpermute_b32 v34, v64, v35
	s_wait_dscnt 0x0
	v_fma_f32 v11, -v36, v34, v11
	ds_bpermute_b32 v34, v65, v35
	s_wait_dscnt 0x0
	v_fma_f32 v12, -v36, v34, v12
	ds_bpermute_b32 v34, v66, v35
	s_wait_dscnt 0x0
	v_fma_f32 v13, -v36, v34, v13
	ds_bpermute_b32 v34, v67, v35
	s_wait_dscnt 0x0
	v_fma_f32 v14, -v36, v34, v14
	ds_bpermute_b32 v34, v68, v35
	s_wait_dscnt 0x0
	v_fma_f32 v15, -v36, v34, v15
	ds_bpermute_b32 v34, v69, v35
	s_wait_dscnt 0x0
	v_fma_f32 v16, -v36, v34, v16
	ds_bpermute_b32 v34, v70, v35
	s_wait_dscnt 0x0
	v_fma_f32 v17, -v36, v34, v17
	ds_bpermute_b32 v34, v71, v35
	s_wait_dscnt 0x0
	v_fma_f32 v82, -v36, v34, v82
	ds_bpermute_b32 v34, v72, v35
	s_wait_dscnt 0x0
	v_fma_f32 v83, -v36, v34, v83
	ds_bpermute_b32 v34, v73, v35
	s_wait_dscnt 0x0
	v_fma_f32 v84, -v36, v34, v84
	ds_bpermute_b32 v34, v74, v35
	s_wait_dscnt 0x0
	v_fma_f32 v85, -v36, v34, v85
	ds_bpermute_b32 v34, v75, v35
	s_wait_dscnt 0x0
	v_fma_f32 v86, -v36, v34, v86
	ds_bpermute_b32 v34, v76, v35
	s_wait_dscnt 0x0
	v_fma_f32 v87, -v36, v34, v87
	ds_bpermute_b32 v34, v77, v35
	s_wait_dscnt 0x0
	v_fma_f32 v88, -v36, v34, v88
	ds_bpermute_b32 v34, v78, v35
	s_wait_dscnt 0x0
	v_fma_f32 v89, -v36, v34, v89
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v34, 0, v21, s1
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v34, v34, v85, s2
	ds_bpermute_b32 v35, v63, v34
	s_wait_dscnt 0x0
	v_add_f32_e32 v36, v34, v35
	ds_load_2addr_b32 v[34:35], v9 offset0:192 offset1:224
	s_wait_dscnt 0x0
	ds_bpermute_b32 v9, v80, v34
	s_wait_dscnt 0x0
	v_fma_f32 v26, -v36, v9, v26
	ds_bpermute_b32 v9, v64, v34
	s_wait_dscnt 0x0
	v_fma_f32 v27, -v36, v9, v27
	ds_bpermute_b32 v9, v65, v34
	s_wait_dscnt 0x0
	v_fma_f32 v28, -v36, v9, v28
	ds_bpermute_b32 v9, v66, v34
	s_wait_dscnt 0x0
	v_fma_f32 v29, -v36, v9, v29
	ds_bpermute_b32 v9, v67, v34
	s_wait_dscnt 0x0
	v_fma_f32 v30, -v36, v9, v30
	ds_bpermute_b32 v9, v68, v34
	s_wait_dscnt 0x0
	v_fma_f32 v31, -v36, v9, v31
	ds_bpermute_b32 v9, v69, v34
	s_wait_dscnt 0x0
	v_fma_f32 v32, -v36, v9, v32
	ds_bpermute_b32 v9, v70, v34
	s_wait_dscnt 0x0
	v_fma_f32 v33, -v36, v9, v33
	ds_bpermute_b32 v9, v71, v34
	s_wait_dscnt 0x0
	v_fma_f32 v18, -v36, v9, v18
	ds_bpermute_b32 v9, v72, v34
	s_wait_dscnt 0x0
	v_fma_f32 v19, -v36, v9, v19
	ds_bpermute_b32 v9, v73, v34
	s_wait_dscnt 0x0
	v_fma_f32 v20, -v36, v9, v20
	ds_bpermute_b32 v9, v74, v34
	s_wait_dscnt 0x0
	v_fma_f32 v21, -v36, v9, v21
	ds_bpermute_b32 v9, v75, v34
	s_wait_dscnt 0x0
	v_fma_f32 v22, -v36, v9, v22
	ds_bpermute_b32 v9, v76, v34
	s_wait_dscnt 0x0
	v_fma_f32 v23, -v36, v9, v23
	ds_bpermute_b32 v9, v77, v34
	s_wait_dscnt 0x0
	v_fma_f32 v24, -v36, v9, v24
	ds_bpermute_b32 v9, v78, v34
	s_wait_dscnt 0x0
	v_fma_f32 v25, -v36, v9, v25
	ds_bpermute_b32 v9, v80, v35
	s_wait_dscnt 0x0
	v_fma_f32 v10, -v36, v9, v10
	ds_bpermute_b32 v9, v64, v35
	s_wait_dscnt 0x0
	v_fma_f32 v11, -v36, v9, v11
	ds_bpermute_b32 v9, v65, v35
	s_wait_dscnt 0x0
	v_fma_f32 v12, -v36, v9, v12
	ds_bpermute_b32 v9, v66, v35
	s_wait_dscnt 0x0
	v_fma_f32 v13, -v36, v9, v13
	ds_bpermute_b32 v9, v67, v35
	s_wait_dscnt 0x0
	v_fma_f32 v14, -v36, v9, v14
	ds_bpermute_b32 v9, v68, v35
	s_wait_dscnt 0x0
	v_fma_f32 v15, -v36, v9, v15
	ds_bpermute_b32 v9, v69, v35
	s_wait_dscnt 0x0
	v_fma_f32 v16, -v36, v9, v16
	ds_bpermute_b32 v9, v70, v35
	s_wait_dscnt 0x0
	v_fma_f32 v17, -v36, v9, v17
	ds_bpermute_b32 v9, v71, v35
	s_wait_dscnt 0x0
	v_fma_f32 v82, -v36, v9, v82
	ds_bpermute_b32 v9, v72, v35
	s_wait_dscnt 0x0
	v_fma_f32 v83, -v36, v9, v83
	ds_bpermute_b32 v9, v73, v35
	s_wait_dscnt 0x0
	v_fma_f32 v84, -v36, v9, v84
	ds_bpermute_b32 v9, v74, v35
	s_wait_dscnt 0x0
	v_fma_f32 v85, -v36, v9, v85
	ds_bpermute_b32 v9, v75, v35
	s_wait_dscnt 0x0
	v_fma_f32 v86, -v36, v9, v86
	ds_bpermute_b32 v9, v76, v35
	s_wait_dscnt 0x0
	v_fma_f32 v87, -v36, v9, v87
	ds_bpermute_b32 v9, v77, v35
	s_wait_dscnt 0x0
	v_fma_f32 v88, -v36, v9, v88
	ds_bpermute_b32 v9, v78, v35
	s_wait_dscnt 0x0
	v_fma_f32 v89, -v36, v9, v89
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v9, 0, v22, s1
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v9, v9, v86, s2
	ds_bpermute_b32 v34, v63, v9
	s_wait_dscnt 0x0
	v_dual_add_f32 v36, v9, v34 :: v_dual_add_nc_u32 v9, 0x1400, v81
	ds_load_2addr_b32 v[34:35], v9 offset1:32
	s_wait_dscnt 0x0
	ds_bpermute_b32 v37, v80, v34
	s_wait_dscnt 0x0
	v_fma_f32 v26, -v36, v37, v26
	ds_bpermute_b32 v37, v64, v34
	s_wait_dscnt 0x0
	v_fma_f32 v27, -v36, v37, v27
	ds_bpermute_b32 v37, v65, v34
	s_wait_dscnt 0x0
	v_fma_f32 v28, -v36, v37, v28
	ds_bpermute_b32 v37, v66, v34
	s_wait_dscnt 0x0
	v_fma_f32 v29, -v36, v37, v29
	ds_bpermute_b32 v37, v67, v34
	s_wait_dscnt 0x0
	v_fma_f32 v30, -v36, v37, v30
	ds_bpermute_b32 v37, v68, v34
	s_wait_dscnt 0x0
	v_fma_f32 v31, -v36, v37, v31
	ds_bpermute_b32 v37, v69, v34
	s_wait_dscnt 0x0
	v_fma_f32 v32, -v36, v37, v32
	ds_bpermute_b32 v37, v70, v34
	s_wait_dscnt 0x0
	v_fma_f32 v33, -v36, v37, v33
	ds_bpermute_b32 v37, v71, v34
	s_wait_dscnt 0x0
	v_fma_f32 v18, -v36, v37, v18
	ds_bpermute_b32 v37, v72, v34
	s_wait_dscnt 0x0
	v_fma_f32 v19, -v36, v37, v19
	ds_bpermute_b32 v37, v73, v34
	s_wait_dscnt 0x0
	v_fma_f32 v20, -v36, v37, v20
	ds_bpermute_b32 v37, v74, v34
	s_wait_dscnt 0x0
	v_fma_f32 v21, -v36, v37, v21
	ds_bpermute_b32 v37, v75, v34
	s_wait_dscnt 0x0
	v_fma_f32 v22, -v36, v37, v22
	ds_bpermute_b32 v37, v76, v34
	s_wait_dscnt 0x0
	v_fma_f32 v23, -v36, v37, v23
	ds_bpermute_b32 v37, v77, v34
	ds_bpermute_b32 v34, v78, v34
	s_wait_dscnt 0x1
	v_fma_f32 v24, -v36, v37, v24
	s_wait_dscnt 0x0
	v_fma_f32 v25, -v36, v34, v25
	ds_bpermute_b32 v34, v80, v35
	s_wait_dscnt 0x0
	v_fma_f32 v10, -v36, v34, v10
	ds_bpermute_b32 v34, v64, v35
	s_wait_dscnt 0x0
	v_fma_f32 v11, -v36, v34, v11
	ds_bpermute_b32 v34, v65, v35
	s_wait_dscnt 0x0
	v_fma_f32 v12, -v36, v34, v12
	ds_bpermute_b32 v34, v66, v35
	s_wait_dscnt 0x0
	v_fma_f32 v13, -v36, v34, v13
	ds_bpermute_b32 v34, v67, v35
	s_wait_dscnt 0x0
	v_fma_f32 v14, -v36, v34, v14
	ds_bpermute_b32 v34, v68, v35
	s_wait_dscnt 0x0
	v_fma_f32 v15, -v36, v34, v15
	ds_bpermute_b32 v34, v69, v35
	s_wait_dscnt 0x0
	v_fma_f32 v16, -v36, v34, v16
	ds_bpermute_b32 v34, v70, v35
	s_wait_dscnt 0x0
	v_fma_f32 v17, -v36, v34, v17
	ds_bpermute_b32 v34, v71, v35
	s_wait_dscnt 0x0
	v_fma_f32 v82, -v36, v34, v82
	ds_bpermute_b32 v34, v72, v35
	s_wait_dscnt 0x0
	v_fma_f32 v83, -v36, v34, v83
	ds_bpermute_b32 v34, v73, v35
	s_wait_dscnt 0x0
	v_fma_f32 v84, -v36, v34, v84
	ds_bpermute_b32 v34, v74, v35
	s_wait_dscnt 0x0
	v_fma_f32 v85, -v36, v34, v85
	ds_bpermute_b32 v34, v75, v35
	s_wait_dscnt 0x0
	v_fma_f32 v86, -v36, v34, v86
	ds_bpermute_b32 v34, v76, v35
	s_wait_dscnt 0x0
	v_fma_f32 v87, -v36, v34, v87
	ds_bpermute_b32 v34, v77, v35
	s_wait_dscnt 0x0
	v_fma_f32 v88, -v36, v34, v88
	ds_bpermute_b32 v34, v78, v35
	s_wait_dscnt 0x0
	v_fma_f32 v89, -v36, v34, v89
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v34, 0, v23, s1
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v34, v34, v87, s2
	ds_bpermute_b32 v35, v63, v34
	s_wait_dscnt 0x0
	v_add_f32_e32 v36, v34, v35
	ds_load_2addr_b32 v[34:35], v9 offset0:64 offset1:96
	s_wait_dscnt 0x0
	ds_bpermute_b32 v37, v80, v34
	s_wait_dscnt 0x0
	v_fma_f32 v26, -v36, v37, v26
	ds_bpermute_b32 v37, v64, v34
	s_wait_dscnt 0x0
	v_fma_f32 v27, -v36, v37, v27
	ds_bpermute_b32 v37, v65, v34
	s_wait_dscnt 0x0
	v_fma_f32 v28, -v36, v37, v28
	ds_bpermute_b32 v37, v66, v34
	s_wait_dscnt 0x0
	v_fma_f32 v29, -v36, v37, v29
	ds_bpermute_b32 v37, v67, v34
	s_wait_dscnt 0x0
	v_fma_f32 v30, -v36, v37, v30
	ds_bpermute_b32 v37, v68, v34
	s_wait_dscnt 0x0
	v_fma_f32 v31, -v36, v37, v31
	ds_bpermute_b32 v37, v69, v34
	s_wait_dscnt 0x0
	v_fma_f32 v32, -v36, v37, v32
	ds_bpermute_b32 v37, v70, v34
	s_wait_dscnt 0x0
	v_fma_f32 v33, -v36, v37, v33
	ds_bpermute_b32 v37, v71, v34
	s_wait_dscnt 0x0
	v_fma_f32 v18, -v36, v37, v18
	ds_bpermute_b32 v37, v72, v34
	s_wait_dscnt 0x0
	v_fma_f32 v19, -v36, v37, v19
	ds_bpermute_b32 v37, v73, v34
	s_wait_dscnt 0x0
	v_fma_f32 v20, -v36, v37, v20
	ds_bpermute_b32 v37, v74, v34
	s_wait_dscnt 0x0
	v_fma_f32 v21, -v36, v37, v21
	ds_bpermute_b32 v37, v75, v34
	s_wait_dscnt 0x0
	v_fma_f32 v22, -v36, v37, v22
	ds_bpermute_b32 v37, v76, v34
	s_wait_dscnt 0x0
	v_fma_f32 v23, -v36, v37, v23
	ds_bpermute_b32 v37, v77, v34
	ds_bpermute_b32 v34, v78, v34
	s_wait_dscnt 0x1
	v_fma_f32 v24, -v36, v37, v24
	s_wait_dscnt 0x0
	v_fma_f32 v25, -v36, v34, v25
	ds_bpermute_b32 v34, v80, v35
	s_wait_dscnt 0x0
	v_fma_f32 v10, -v36, v34, v10
	ds_bpermute_b32 v34, v64, v35
	s_wait_dscnt 0x0
	v_fma_f32 v11, -v36, v34, v11
	ds_bpermute_b32 v34, v65, v35
	s_wait_dscnt 0x0
	v_fma_f32 v12, -v36, v34, v12
	ds_bpermute_b32 v34, v66, v35
	s_wait_dscnt 0x0
	v_fma_f32 v13, -v36, v34, v13
	ds_bpermute_b32 v34, v67, v35
	s_wait_dscnt 0x0
	v_fma_f32 v14, -v36, v34, v14
	ds_bpermute_b32 v34, v68, v35
	s_wait_dscnt 0x0
	v_fma_f32 v15, -v36, v34, v15
	ds_bpermute_b32 v34, v69, v35
	s_wait_dscnt 0x0
	v_fma_f32 v16, -v36, v34, v16
	ds_bpermute_b32 v34, v70, v35
	s_wait_dscnt 0x0
	v_fma_f32 v17, -v36, v34, v17
	ds_bpermute_b32 v34, v71, v35
	s_wait_dscnt 0x0
	v_fma_f32 v82, -v36, v34, v82
	ds_bpermute_b32 v34, v72, v35
	s_wait_dscnt 0x0
	v_fma_f32 v83, -v36, v34, v83
	ds_bpermute_b32 v34, v73, v35
	s_wait_dscnt 0x0
	v_fma_f32 v84, -v36, v34, v84
	ds_bpermute_b32 v34, v74, v35
	s_wait_dscnt 0x0
	v_fma_f32 v85, -v36, v34, v85
	ds_bpermute_b32 v34, v75, v35
	s_wait_dscnt 0x0
	v_fma_f32 v86, -v36, v34, v86
	ds_bpermute_b32 v34, v76, v35
	s_wait_dscnt 0x0
	v_fma_f32 v87, -v36, v34, v87
	ds_bpermute_b32 v34, v77, v35
	s_wait_dscnt 0x0
	v_fma_f32 v88, -v36, v34, v88
	ds_bpermute_b32 v34, v78, v35
	s_wait_dscnt 0x0
	v_fma_f32 v89, -v36, v34, v89
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v34, 0, v24, s1
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v34, v34, v88, s2
	ds_bpermute_b32 v35, v63, v34
	s_wait_dscnt 0x0
	v_add_f32_e32 v36, v34, v35
	ds_load_2addr_b32 v[34:35], v9 offset0:128 offset1:160
	s_wait_dscnt 0x0
	ds_bpermute_b32 v37, v80, v34
	s_wait_dscnt 0x0
	v_fma_f32 v26, -v36, v37, v26
	ds_bpermute_b32 v37, v64, v34
	s_wait_dscnt 0x0
	v_fma_f32 v27, -v36, v37, v27
	ds_bpermute_b32 v37, v65, v34
	s_wait_dscnt 0x0
	v_fma_f32 v28, -v36, v37, v28
	ds_bpermute_b32 v37, v66, v34
	s_wait_dscnt 0x0
	v_fma_f32 v29, -v36, v37, v29
	ds_bpermute_b32 v37, v67, v34
	s_wait_dscnt 0x0
	v_fma_f32 v30, -v36, v37, v30
	ds_bpermute_b32 v37, v68, v34
	s_wait_dscnt 0x0
	v_fma_f32 v31, -v36, v37, v31
	ds_bpermute_b32 v37, v69, v34
	s_wait_dscnt 0x0
	v_fma_f32 v32, -v36, v37, v32
	ds_bpermute_b32 v37, v70, v34
	s_wait_dscnt 0x0
	v_fma_f32 v33, -v36, v37, v33
	ds_bpermute_b32 v37, v71, v34
	s_wait_dscnt 0x0
	v_fma_f32 v18, -v36, v37, v18
	ds_bpermute_b32 v37, v72, v34
	s_wait_dscnt 0x0
	v_fma_f32 v19, -v36, v37, v19
	ds_bpermute_b32 v37, v73, v34
	s_wait_dscnt 0x0
	v_fma_f32 v20, -v36, v37, v20
	ds_bpermute_b32 v37, v74, v34
	s_wait_dscnt 0x0
	v_fma_f32 v21, -v36, v37, v21
	ds_bpermute_b32 v37, v75, v34
	s_wait_dscnt 0x0
	v_fma_f32 v22, -v36, v37, v22
	ds_bpermute_b32 v37, v76, v34
	s_wait_dscnt 0x0
	v_fma_f32 v23, -v36, v37, v23
	ds_bpermute_b32 v37, v77, v34
	ds_bpermute_b32 v34, v78, v34
	s_wait_dscnt 0x1
	v_fma_f32 v24, -v36, v37, v24
	s_wait_dscnt 0x0
	v_fma_f32 v25, -v36, v34, v25
	ds_bpermute_b32 v34, v80, v35
	s_wait_dscnt 0x0
	v_fma_f32 v10, -v36, v34, v10
	ds_bpermute_b32 v34, v64, v35
	s_wait_dscnt 0x0
	v_fma_f32 v11, -v36, v34, v11
	ds_bpermute_b32 v34, v65, v35
	s_wait_dscnt 0x0
	v_fma_f32 v12, -v36, v34, v12
	ds_bpermute_b32 v34, v66, v35
	s_wait_dscnt 0x0
	v_fma_f32 v13, -v36, v34, v13
	ds_bpermute_b32 v34, v67, v35
	s_wait_dscnt 0x0
	v_fma_f32 v14, -v36, v34, v14
	ds_bpermute_b32 v34, v68, v35
	s_wait_dscnt 0x0
	v_fma_f32 v15, -v36, v34, v15
	ds_bpermute_b32 v34, v69, v35
	s_wait_dscnt 0x0
	v_fma_f32 v16, -v36, v34, v16
	ds_bpermute_b32 v34, v70, v35
	s_wait_dscnt 0x0
	v_fma_f32 v17, -v36, v34, v17
	ds_bpermute_b32 v34, v71, v35
	s_wait_dscnt 0x0
	v_fma_f32 v82, -v36, v34, v82
	ds_bpermute_b32 v34, v72, v35
	s_wait_dscnt 0x0
	v_fma_f32 v83, -v36, v34, v83
	ds_bpermute_b32 v34, v73, v35
	s_wait_dscnt 0x0
	v_fma_f32 v84, -v36, v34, v84
	ds_bpermute_b32 v34, v74, v35
	s_wait_dscnt 0x0
	v_fma_f32 v85, -v36, v34, v85
	ds_bpermute_b32 v34, v75, v35
	s_wait_dscnt 0x0
	v_fma_f32 v86, -v36, v34, v86
	ds_bpermute_b32 v34, v76, v35
	s_wait_dscnt 0x0
	v_fma_f32 v87, -v36, v34, v87
	ds_bpermute_b32 v34, v77, v35
	s_wait_dscnt 0x0
	v_fma_f32 v88, -v36, v34, v88
	ds_bpermute_b32 v34, v78, v35
	s_wait_dscnt 0x0
	v_fma_f32 v89, -v36, v34, v89
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e64 v34, 0, v25, s1
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v34, v34, v89, s2
	ds_bpermute_b32 v35, v63, v34
	s_wait_dscnt 0x0
	v_add_f32_e32 v36, v34, v35
	ds_load_2addr_b32 v[34:35], v9 offset0:192 offset1:224
	s_wait_dscnt 0x0
	ds_bpermute_b32 v9, v80, v34
	s_wait_dscnt 0x0
	v_fma_f32 v26, -v36, v9, v26
	ds_bpermute_b32 v9, v64, v34
	s_wait_dscnt 0x0
	v_fma_f32 v27, -v36, v9, v27
	ds_bpermute_b32 v9, v65, v34
	s_wait_dscnt 0x0
	v_fma_f32 v28, -v36, v9, v28
	ds_bpermute_b32 v9, v66, v34
	s_wait_dscnt 0x0
	v_fma_f32 v29, -v36, v9, v29
	ds_bpermute_b32 v9, v67, v34
	s_wait_dscnt 0x0
	v_fma_f32 v30, -v36, v9, v30
	ds_bpermute_b32 v9, v68, v34
	s_wait_dscnt 0x0
	v_fma_f32 v31, -v36, v9, v31
	ds_bpermute_b32 v9, v69, v34
	s_wait_dscnt 0x0
	v_fma_f32 v32, -v36, v9, v32
	ds_bpermute_b32 v9, v70, v34
	s_wait_dscnt 0x0
	v_fma_f32 v33, -v36, v9, v33
	ds_bpermute_b32 v9, v71, v34
	s_wait_dscnt 0x0
	v_fma_f32 v18, -v36, v9, v18
	ds_bpermute_b32 v9, v72, v34
	s_wait_dscnt 0x0
	v_fma_f32 v19, -v36, v9, v19
	ds_bpermute_b32 v9, v73, v34
	s_wait_dscnt 0x0
	v_fma_f32 v20, -v36, v9, v20
	ds_bpermute_b32 v9, v74, v34
	s_wait_dscnt 0x0
	v_fma_f32 v21, -v36, v9, v21
	ds_bpermute_b32 v9, v75, v34
	s_wait_dscnt 0x0
	v_fma_f32 v22, -v36, v9, v22
	ds_bpermute_b32 v9, v76, v34
	s_wait_dscnt 0x0
	v_fma_f32 v23, -v36, v9, v23
	ds_bpermute_b32 v9, v77, v34
	s_wait_dscnt 0x0
	v_fma_f32 v24, -v36, v9, v24
	ds_bpermute_b32 v9, v78, v34
	s_wait_dscnt 0x0
	v_fma_f32 v25, -v36, v9, v25
	ds_bpermute_b32 v9, v80, v35
	s_wait_dscnt 0x0
	v_fma_f32 v10, -v36, v9, v10
	ds_bpermute_b32 v9, v64, v35
	s_wait_dscnt 0x0
	v_fma_f32 v11, -v36, v9, v11
	ds_bpermute_b32 v9, v65, v35
	s_wait_dscnt 0x0
	v_fma_f32 v12, -v36, v9, v12
	ds_bpermute_b32 v9, v66, v35
	s_wait_dscnt 0x0
	v_fma_f32 v13, -v36, v9, v13
	ds_bpermute_b32 v9, v67, v35
	s_wait_dscnt 0x0
	v_fma_f32 v14, -v36, v9, v14
	ds_bpermute_b32 v9, v68, v35
	s_wait_dscnt 0x0
	v_fma_f32 v15, -v36, v9, v15
	ds_bpermute_b32 v9, v69, v35
	s_wait_dscnt 0x0
	v_fma_f32 v16, -v36, v9, v16
	ds_bpermute_b32 v9, v70, v35
	s_wait_dscnt 0x0
	v_fma_f32 v17, -v36, v9, v17
	ds_bpermute_b32 v9, v71, v35
	s_wait_dscnt 0x0
	v_fma_f32 v82, -v36, v9, v82
	ds_bpermute_b32 v9, v72, v35
	s_wait_dscnt 0x0
	v_fma_f32 v83, -v36, v9, v83
	ds_bpermute_b32 v9, v73, v35
	s_wait_dscnt 0x0
	v_fma_f32 v84, -v36, v9, v84
	ds_bpermute_b32 v9, v74, v35
	s_wait_dscnt 0x0
	v_fma_f32 v85, -v36, v9, v85
	ds_bpermute_b32 v9, v75, v35
	s_wait_dscnt 0x0
	v_fma_f32 v86, -v36, v9, v86
	ds_bpermute_b32 v9, v76, v35
	s_wait_dscnt 0x0
	v_fma_f32 v87, -v36, v9, v87
	ds_bpermute_b32 v9, v77, v35
	s_wait_dscnt 0x0
	v_fma_f32 v88, -v36, v9, v88
	ds_bpermute_b32 v9, v78, v35
	s_wait_dscnt 0x0
	v_fma_f32 v89, -v36, v9, v89
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e32 v9, 0, v18, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v9, v9, v82, s0
	ds_bpermute_b32 v34, v63, v9
	s_wait_dscnt 0x0
	v_dual_add_f32 v36, v9, v34 :: v_dual_add_nc_u32 v9, 0x1800, v81
	ds_load_2addr_b32 v[34:35], v9 offset1:32
	s_wait_dscnt 0x0
	ds_bpermute_b32 v37, v80, v34
	s_wait_dscnt 0x0
	v_fma_f32 v26, -v36, v37, v26
	ds_bpermute_b32 v37, v64, v34
	s_wait_dscnt 0x0
	v_fma_f32 v27, -v36, v37, v27
	ds_bpermute_b32 v37, v65, v34
	s_wait_dscnt 0x0
	v_fma_f32 v28, -v36, v37, v28
	ds_bpermute_b32 v37, v66, v34
	s_wait_dscnt 0x0
	v_fma_f32 v29, -v36, v37, v29
	ds_bpermute_b32 v37, v67, v34
	s_wait_dscnt 0x0
	v_fma_f32 v30, -v36, v37, v30
	ds_bpermute_b32 v37, v68, v34
	s_wait_dscnt 0x0
	v_fma_f32 v31, -v36, v37, v31
	ds_bpermute_b32 v37, v69, v34
	s_wait_dscnt 0x0
	v_fma_f32 v32, -v36, v37, v32
	ds_bpermute_b32 v37, v70, v34
	s_wait_dscnt 0x0
	v_fma_f32 v33, -v36, v37, v33
	ds_bpermute_b32 v37, v71, v34
	s_wait_dscnt 0x0
	v_fma_f32 v18, -v36, v37, v18
	ds_bpermute_b32 v37, v72, v34
	s_wait_dscnt 0x0
	v_fma_f32 v19, -v36, v37, v19
	ds_bpermute_b32 v37, v73, v34
	s_wait_dscnt 0x0
	v_fma_f32 v20, -v36, v37, v20
	ds_bpermute_b32 v37, v74, v34
	s_wait_dscnt 0x0
	v_fma_f32 v21, -v36, v37, v21
	ds_bpermute_b32 v37, v75, v34
	s_wait_dscnt 0x0
	v_fma_f32 v22, -v36, v37, v22
	ds_bpermute_b32 v37, v76, v34
	s_wait_dscnt 0x0
	v_fma_f32 v23, -v36, v37, v23
	ds_bpermute_b32 v37, v77, v34
	ds_bpermute_b32 v34, v78, v34
	s_wait_dscnt 0x1
	v_fma_f32 v24, -v36, v37, v24
	s_wait_dscnt 0x0
	v_fma_f32 v25, -v36, v34, v25
	ds_bpermute_b32 v34, v80, v35
	s_wait_dscnt 0x0
	v_fma_f32 v10, -v36, v34, v10
	ds_bpermute_b32 v34, v64, v35
	s_wait_dscnt 0x0
	v_fma_f32 v11, -v36, v34, v11
	ds_bpermute_b32 v34, v65, v35
	s_wait_dscnt 0x0
	v_fma_f32 v12, -v36, v34, v12
	ds_bpermute_b32 v34, v66, v35
	s_wait_dscnt 0x0
	v_fma_f32 v13, -v36, v34, v13
	ds_bpermute_b32 v34, v67, v35
	s_wait_dscnt 0x0
	v_fma_f32 v14, -v36, v34, v14
	ds_bpermute_b32 v34, v68, v35
	s_wait_dscnt 0x0
	v_fma_f32 v15, -v36, v34, v15
	ds_bpermute_b32 v34, v69, v35
	s_wait_dscnt 0x0
	v_fma_f32 v16, -v36, v34, v16
	ds_bpermute_b32 v34, v70, v35
	s_wait_dscnt 0x0
	v_fma_f32 v17, -v36, v34, v17
	ds_bpermute_b32 v34, v71, v35
	s_wait_dscnt 0x0
	v_fma_f32 v82, -v36, v34, v82
	ds_bpermute_b32 v34, v72, v35
	s_wait_dscnt 0x0
	v_fma_f32 v83, -v36, v34, v83
	ds_bpermute_b32 v34, v73, v35
	s_wait_dscnt 0x0
	v_fma_f32 v84, -v36, v34, v84
	ds_bpermute_b32 v34, v74, v35
	s_wait_dscnt 0x0
	v_fma_f32 v85, -v36, v34, v85
	ds_bpermute_b32 v34, v75, v35
	s_wait_dscnt 0x0
	v_fma_f32 v86, -v36, v34, v86
	ds_bpermute_b32 v34, v76, v35
	s_wait_dscnt 0x0
	v_fma_f32 v87, -v36, v34, v87
	ds_bpermute_b32 v34, v77, v35
	s_wait_dscnt 0x0
	v_fma_f32 v88, -v36, v34, v88
	ds_bpermute_b32 v34, v78, v35
	s_wait_dscnt 0x0
	v_fma_f32 v89, -v36, v34, v89
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e32 v34, 0, v19, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v34, v34, v83, s0
	ds_bpermute_b32 v35, v63, v34
	s_wait_dscnt 0x0
	v_add_f32_e32 v36, v34, v35
	ds_load_2addr_b32 v[34:35], v9 offset0:64 offset1:96
	s_wait_dscnt 0x0
	ds_bpermute_b32 v37, v80, v34
	s_wait_dscnt 0x0
	v_fma_f32 v26, -v36, v37, v26
	ds_bpermute_b32 v37, v64, v34
	s_wait_dscnt 0x0
	v_fma_f32 v27, -v36, v37, v27
	ds_bpermute_b32 v37, v65, v34
	s_wait_dscnt 0x0
	v_fma_f32 v28, -v36, v37, v28
	ds_bpermute_b32 v37, v66, v34
	s_wait_dscnt 0x0
	v_fma_f32 v29, -v36, v37, v29
	ds_bpermute_b32 v37, v67, v34
	s_wait_dscnt 0x0
	v_fma_f32 v30, -v36, v37, v30
	ds_bpermute_b32 v37, v68, v34
	s_wait_dscnt 0x0
	v_fma_f32 v31, -v36, v37, v31
	ds_bpermute_b32 v37, v69, v34
	s_wait_dscnt 0x0
	v_fma_f32 v32, -v36, v37, v32
	ds_bpermute_b32 v37, v70, v34
	s_wait_dscnt 0x0
	v_fma_f32 v33, -v36, v37, v33
	ds_bpermute_b32 v37, v71, v34
	s_wait_dscnt 0x0
	v_fma_f32 v18, -v36, v37, v18
	ds_bpermute_b32 v37, v72, v34
	s_wait_dscnt 0x0
	v_fma_f32 v19, -v36, v37, v19
	ds_bpermute_b32 v37, v73, v34
	s_wait_dscnt 0x0
	v_fma_f32 v20, -v36, v37, v20
	ds_bpermute_b32 v37, v74, v34
	s_wait_dscnt 0x0
	v_fma_f32 v21, -v36, v37, v21
	ds_bpermute_b32 v37, v75, v34
	s_wait_dscnt 0x0
	v_fma_f32 v22, -v36, v37, v22
	ds_bpermute_b32 v37, v76, v34
	s_wait_dscnt 0x0
	v_fma_f32 v23, -v36, v37, v23
	ds_bpermute_b32 v37, v77, v34
	ds_bpermute_b32 v34, v78, v34
	s_wait_dscnt 0x1
	v_fma_f32 v24, -v36, v37, v24
	s_wait_dscnt 0x0
	v_fma_f32 v25, -v36, v34, v25
	ds_bpermute_b32 v34, v80, v35
	s_wait_dscnt 0x0
	v_fma_f32 v10, -v36, v34, v10
	ds_bpermute_b32 v34, v64, v35
	s_wait_dscnt 0x0
	v_fma_f32 v11, -v36, v34, v11
	ds_bpermute_b32 v34, v65, v35
	s_wait_dscnt 0x0
	v_fma_f32 v12, -v36, v34, v12
	ds_bpermute_b32 v34, v66, v35
	s_wait_dscnt 0x0
	v_fma_f32 v13, -v36, v34, v13
	ds_bpermute_b32 v34, v67, v35
	s_wait_dscnt 0x0
	v_fma_f32 v14, -v36, v34, v14
	ds_bpermute_b32 v34, v68, v35
	s_wait_dscnt 0x0
	v_fma_f32 v15, -v36, v34, v15
	ds_bpermute_b32 v34, v69, v35
	s_wait_dscnt 0x0
	v_fma_f32 v16, -v36, v34, v16
	ds_bpermute_b32 v34, v70, v35
	s_wait_dscnt 0x0
	v_fma_f32 v17, -v36, v34, v17
	ds_bpermute_b32 v34, v71, v35
	s_wait_dscnt 0x0
	v_fma_f32 v82, -v36, v34, v82
	ds_bpermute_b32 v34, v72, v35
	s_wait_dscnt 0x0
	v_fma_f32 v83, -v36, v34, v83
	ds_bpermute_b32 v34, v73, v35
	s_wait_dscnt 0x0
	v_fma_f32 v84, -v36, v34, v84
	ds_bpermute_b32 v34, v74, v35
	s_wait_dscnt 0x0
	v_fma_f32 v85, -v36, v34, v85
	ds_bpermute_b32 v34, v75, v35
	s_wait_dscnt 0x0
	v_fma_f32 v86, -v36, v34, v86
	ds_bpermute_b32 v34, v76, v35
	s_wait_dscnt 0x0
	v_fma_f32 v87, -v36, v34, v87
	ds_bpermute_b32 v34, v77, v35
	s_wait_dscnt 0x0
	v_fma_f32 v88, -v36, v34, v88
	ds_bpermute_b32 v34, v78, v35
	s_wait_dscnt 0x0
	v_fma_f32 v89, -v36, v34, v89
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e32 v34, 0, v20, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v34, v34, v84, s0
	ds_bpermute_b32 v35, v63, v34
	s_wait_dscnt 0x0
	v_add_f32_e32 v36, v34, v35
	ds_load_2addr_b32 v[34:35], v9 offset0:128 offset1:160
	s_wait_dscnt 0x0
	ds_bpermute_b32 v37, v80, v34
	s_wait_dscnt 0x0
	v_fma_f32 v26, -v36, v37, v26
	ds_bpermute_b32 v37, v64, v34
	s_wait_dscnt 0x0
	v_fma_f32 v27, -v36, v37, v27
	ds_bpermute_b32 v37, v65, v34
	s_wait_dscnt 0x0
	v_fma_f32 v28, -v36, v37, v28
	ds_bpermute_b32 v37, v66, v34
	s_wait_dscnt 0x0
	v_fma_f32 v29, -v36, v37, v29
	ds_bpermute_b32 v37, v67, v34
	s_wait_dscnt 0x0
	v_fma_f32 v30, -v36, v37, v30
	ds_bpermute_b32 v37, v68, v34
	s_wait_dscnt 0x0
	v_fma_f32 v31, -v36, v37, v31
	ds_bpermute_b32 v37, v69, v34
	s_wait_dscnt 0x0
	v_fma_f32 v32, -v36, v37, v32
	ds_bpermute_b32 v37, v70, v34
	s_wait_dscnt 0x0
	v_fma_f32 v33, -v36, v37, v33
	ds_bpermute_b32 v37, v71, v34
	s_wait_dscnt 0x0
	v_fma_f32 v18, -v36, v37, v18
	ds_bpermute_b32 v37, v72, v34
	s_wait_dscnt 0x0
	v_fma_f32 v19, -v36, v37, v19
	ds_bpermute_b32 v37, v73, v34
	s_wait_dscnt 0x0
	v_fma_f32 v20, -v36, v37, v20
	ds_bpermute_b32 v37, v74, v34
	s_wait_dscnt 0x0
	v_fma_f32 v21, -v36, v37, v21
	ds_bpermute_b32 v37, v75, v34
	s_wait_dscnt 0x0
	v_fma_f32 v22, -v36, v37, v22
	ds_bpermute_b32 v37, v76, v34
	s_wait_dscnt 0x0
	v_fma_f32 v23, -v36, v37, v23
	ds_bpermute_b32 v37, v77, v34
	ds_bpermute_b32 v34, v78, v34
	s_wait_dscnt 0x1
	v_fma_f32 v24, -v36, v37, v24
	s_wait_dscnt 0x0
	v_fma_f32 v25, -v36, v34, v25
	ds_bpermute_b32 v34, v80, v35
	s_wait_dscnt 0x0
	v_fma_f32 v10, -v36, v34, v10
	ds_bpermute_b32 v34, v64, v35
	s_wait_dscnt 0x0
	v_fma_f32 v11, -v36, v34, v11
	ds_bpermute_b32 v34, v65, v35
	s_wait_dscnt 0x0
	v_fma_f32 v12, -v36, v34, v12
	ds_bpermute_b32 v34, v66, v35
	s_wait_dscnt 0x0
	v_fma_f32 v13, -v36, v34, v13
	ds_bpermute_b32 v34, v67, v35
	s_wait_dscnt 0x0
	v_fma_f32 v14, -v36, v34, v14
	ds_bpermute_b32 v34, v68, v35
	s_wait_dscnt 0x0
	v_fma_f32 v15, -v36, v34, v15
	ds_bpermute_b32 v34, v69, v35
	s_wait_dscnt 0x0
	v_fma_f32 v16, -v36, v34, v16
	ds_bpermute_b32 v34, v70, v35
	s_wait_dscnt 0x0
	v_fma_f32 v17, -v36, v34, v17
	ds_bpermute_b32 v34, v71, v35
	s_wait_dscnt 0x0
	v_fma_f32 v82, -v36, v34, v82
	ds_bpermute_b32 v34, v72, v35
	s_wait_dscnt 0x0
	v_fma_f32 v83, -v36, v34, v83
	ds_bpermute_b32 v34, v73, v35
	s_wait_dscnt 0x0
	v_fma_f32 v84, -v36, v34, v84
	ds_bpermute_b32 v34, v74, v35
	s_wait_dscnt 0x0
	v_fma_f32 v85, -v36, v34, v85
	ds_bpermute_b32 v34, v75, v35
	s_wait_dscnt 0x0
	v_fma_f32 v86, -v36, v34, v86
	ds_bpermute_b32 v34, v76, v35
	s_wait_dscnt 0x0
	v_fma_f32 v87, -v36, v34, v87
	ds_bpermute_b32 v34, v77, v35
	s_wait_dscnt 0x0
	v_fma_f32 v88, -v36, v34, v88
	ds_bpermute_b32 v34, v78, v35
	s_wait_dscnt 0x0
	v_fma_f32 v89, -v36, v34, v89
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e32 v34, 0, v21, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v34, v34, v85, s0
	ds_bpermute_b32 v35, v63, v34
	s_wait_dscnt 0x0
	v_add_f32_e32 v36, v34, v35
	ds_load_2addr_b32 v[34:35], v9 offset0:192 offset1:224
	s_wait_dscnt 0x0
	ds_bpermute_b32 v9, v80, v34
	s_wait_dscnt 0x0
	v_fma_f32 v26, -v36, v9, v26
	ds_bpermute_b32 v9, v64, v34
	s_wait_dscnt 0x0
	v_fma_f32 v27, -v36, v9, v27
	ds_bpermute_b32 v9, v65, v34
	s_wait_dscnt 0x0
	v_fma_f32 v28, -v36, v9, v28
	ds_bpermute_b32 v9, v66, v34
	s_wait_dscnt 0x0
	v_fma_f32 v29, -v36, v9, v29
	ds_bpermute_b32 v9, v67, v34
	s_wait_dscnt 0x0
	v_fma_f32 v30, -v36, v9, v30
	ds_bpermute_b32 v9, v68, v34
	s_wait_dscnt 0x0
	v_fma_f32 v31, -v36, v9, v31
	ds_bpermute_b32 v9, v69, v34
	s_wait_dscnt 0x0
	v_fma_f32 v32, -v36, v9, v32
	ds_bpermute_b32 v9, v70, v34
	s_wait_dscnt 0x0
	v_fma_f32 v33, -v36, v9, v33
	ds_bpermute_b32 v9, v71, v34
	s_wait_dscnt 0x0
	v_fma_f32 v18, -v36, v9, v18
	ds_bpermute_b32 v9, v72, v34
	s_wait_dscnt 0x0
	v_fma_f32 v19, -v36, v9, v19
	ds_bpermute_b32 v9, v73, v34
	s_wait_dscnt 0x0
	v_fma_f32 v20, -v36, v9, v20
	ds_bpermute_b32 v9, v74, v34
	s_wait_dscnt 0x0
	v_fma_f32 v21, -v36, v9, v21
	ds_bpermute_b32 v9, v75, v34
	s_wait_dscnt 0x0
	v_fma_f32 v22, -v36, v9, v22
	ds_bpermute_b32 v9, v76, v34
	s_wait_dscnt 0x0
	v_fma_f32 v23, -v36, v9, v23
	ds_bpermute_b32 v9, v77, v34
	s_wait_dscnt 0x0
	v_fma_f32 v24, -v36, v9, v24
	ds_bpermute_b32 v9, v78, v34
	s_wait_dscnt 0x0
	v_fma_f32 v25, -v36, v9, v25
	ds_bpermute_b32 v9, v80, v35
	s_wait_dscnt 0x0
	v_fma_f32 v10, -v36, v9, v10
	ds_bpermute_b32 v9, v64, v35
	s_wait_dscnt 0x0
	v_fma_f32 v11, -v36, v9, v11
	ds_bpermute_b32 v9, v65, v35
	s_wait_dscnt 0x0
	v_fma_f32 v12, -v36, v9, v12
	ds_bpermute_b32 v9, v66, v35
	s_wait_dscnt 0x0
	v_fma_f32 v13, -v36, v9, v13
	ds_bpermute_b32 v9, v67, v35
	s_wait_dscnt 0x0
	v_fma_f32 v14, -v36, v9, v14
	ds_bpermute_b32 v9, v68, v35
	s_wait_dscnt 0x0
	v_fma_f32 v15, -v36, v9, v15
	ds_bpermute_b32 v9, v69, v35
	s_wait_dscnt 0x0
	v_fma_f32 v16, -v36, v9, v16
	ds_bpermute_b32 v9, v70, v35
	s_wait_dscnt 0x0
	v_fma_f32 v17, -v36, v9, v17
	ds_bpermute_b32 v9, v71, v35
	s_wait_dscnt 0x0
	v_fma_f32 v82, -v36, v9, v82
	ds_bpermute_b32 v9, v72, v35
	s_wait_dscnt 0x0
	v_fma_f32 v83, -v36, v9, v83
	ds_bpermute_b32 v9, v73, v35
	s_wait_dscnt 0x0
	v_fma_f32 v84, -v36, v9, v84
	ds_bpermute_b32 v9, v74, v35
	s_wait_dscnt 0x0
	v_fma_f32 v85, -v36, v9, v85
	ds_bpermute_b32 v9, v75, v35
	s_wait_dscnt 0x0
	v_fma_f32 v86, -v36, v9, v86
	ds_bpermute_b32 v9, v76, v35
	s_wait_dscnt 0x0
	v_fma_f32 v87, -v36, v9, v87
	ds_bpermute_b32 v9, v77, v35
	s_wait_dscnt 0x0
	v_fma_f32 v88, -v36, v9, v88
	ds_bpermute_b32 v9, v78, v35
	s_wait_dscnt 0x0
	v_fma_f32 v89, -v36, v9, v89
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e32 v9, 0, v22, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v9, v9, v86, s0
	ds_bpermute_b32 v34, v63, v9
	s_wait_dscnt 0x0
	v_dual_add_f32 v36, v9, v34 :: v_dual_add_nc_u32 v9, 0x1c00, v81
	ds_load_2addr_b32 v[34:35], v9 offset1:32
	s_wait_dscnt 0x0
	ds_bpermute_b32 v37, v80, v34
	s_wait_dscnt 0x0
	v_fma_f32 v26, -v36, v37, v26
	ds_bpermute_b32 v37, v64, v34
	s_wait_dscnt 0x0
	v_fma_f32 v27, -v36, v37, v27
	ds_bpermute_b32 v37, v65, v34
	s_wait_dscnt 0x0
	v_fma_f32 v28, -v36, v37, v28
	ds_bpermute_b32 v37, v66, v34
	s_wait_dscnt 0x0
	v_fma_f32 v29, -v36, v37, v29
	ds_bpermute_b32 v37, v67, v34
	s_wait_dscnt 0x0
	v_fma_f32 v30, -v36, v37, v30
	ds_bpermute_b32 v37, v68, v34
	s_wait_dscnt 0x0
	v_fma_f32 v31, -v36, v37, v31
	ds_bpermute_b32 v37, v69, v34
	s_wait_dscnt 0x0
	v_fma_f32 v32, -v36, v37, v32
	ds_bpermute_b32 v37, v70, v34
	s_wait_dscnt 0x0
	v_fma_f32 v33, -v36, v37, v33
	ds_bpermute_b32 v37, v71, v34
	s_wait_dscnt 0x0
	v_fma_f32 v18, -v36, v37, v18
	ds_bpermute_b32 v37, v72, v34
	s_wait_dscnt 0x0
	v_fma_f32 v19, -v36, v37, v19
	ds_bpermute_b32 v37, v73, v34
	s_wait_dscnt 0x0
	v_fma_f32 v20, -v36, v37, v20
	ds_bpermute_b32 v37, v74, v34
	s_wait_dscnt 0x0
	v_fma_f32 v21, -v36, v37, v21
	ds_bpermute_b32 v37, v75, v34
	s_wait_dscnt 0x0
	v_fma_f32 v22, -v36, v37, v22
	ds_bpermute_b32 v37, v76, v34
	s_wait_dscnt 0x0
	v_fma_f32 v23, -v36, v37, v23
	ds_bpermute_b32 v37, v77, v34
	ds_bpermute_b32 v34, v78, v34
	s_wait_dscnt 0x1
	v_fma_f32 v24, -v36, v37, v24
	s_wait_dscnt 0x0
	v_fma_f32 v25, -v36, v34, v25
	ds_bpermute_b32 v34, v80, v35
	s_wait_dscnt 0x0
	v_fma_f32 v10, -v36, v34, v10
	ds_bpermute_b32 v34, v64, v35
	s_wait_dscnt 0x0
	v_fma_f32 v11, -v36, v34, v11
	ds_bpermute_b32 v34, v65, v35
	s_wait_dscnt 0x0
	v_fma_f32 v12, -v36, v34, v12
	ds_bpermute_b32 v34, v66, v35
	s_wait_dscnt 0x0
	v_fma_f32 v13, -v36, v34, v13
	ds_bpermute_b32 v34, v67, v35
	s_wait_dscnt 0x0
	v_fma_f32 v14, -v36, v34, v14
	ds_bpermute_b32 v34, v68, v35
	s_wait_dscnt 0x0
	v_fma_f32 v15, -v36, v34, v15
	ds_bpermute_b32 v34, v69, v35
	s_wait_dscnt 0x0
	v_fma_f32 v16, -v36, v34, v16
	ds_bpermute_b32 v34, v70, v35
	s_wait_dscnt 0x0
	v_fma_f32 v17, -v36, v34, v17
	ds_bpermute_b32 v34, v71, v35
	s_wait_dscnt 0x0
	v_fma_f32 v81, -v36, v34, v82
	ds_bpermute_b32 v34, v72, v35
	s_wait_dscnt 0x0
	v_fma_f32 v82, -v36, v34, v83
	ds_bpermute_b32 v34, v73, v35
	s_wait_dscnt 0x0
	v_fma_f32 v83, -v36, v34, v84
	ds_bpermute_b32 v34, v74, v35
	s_wait_dscnt 0x0
	v_fma_f32 v84, -v36, v34, v85
	ds_bpermute_b32 v34, v75, v35
	s_wait_dscnt 0x0
	v_fma_f32 v85, -v36, v34, v86
	ds_bpermute_b32 v34, v76, v35
	s_wait_dscnt 0x0
	v_fma_f32 v86, -v36, v34, v87
	ds_bpermute_b32 v34, v77, v35
	s_wait_dscnt 0x0
	v_fma_f32 v87, -v36, v34, v88
	ds_bpermute_b32 v34, v78, v35
	s_wait_dscnt 0x0
	v_fma_f32 v88, -v36, v34, v89
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e32 v34, 0, v23, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v34, v34, v86, s0
	ds_bpermute_b32 v35, v63, v34
	s_wait_dscnt 0x0
	v_add_f32_e32 v36, v34, v35
	ds_load_2addr_b32 v[34:35], v9 offset0:64 offset1:96
	s_wait_dscnt 0x0
	ds_bpermute_b32 v37, v80, v34
	s_wait_dscnt 0x0
	v_fma_f32 v26, -v36, v37, v26
	ds_bpermute_b32 v37, v64, v34
	s_wait_dscnt 0x0
	v_fma_f32 v27, -v36, v37, v27
	ds_bpermute_b32 v37, v65, v34
	s_wait_dscnt 0x0
	v_fma_f32 v28, -v36, v37, v28
	ds_bpermute_b32 v37, v66, v34
	s_wait_dscnt 0x0
	v_fma_f32 v29, -v36, v37, v29
	ds_bpermute_b32 v37, v67, v34
	s_wait_dscnt 0x0
	v_fma_f32 v30, -v36, v37, v30
	ds_bpermute_b32 v37, v68, v34
	s_wait_dscnt 0x0
	v_fma_f32 v31, -v36, v37, v31
	ds_bpermute_b32 v37, v69, v34
	s_wait_dscnt 0x0
	v_fma_f32 v32, -v36, v37, v32
	ds_bpermute_b32 v37, v70, v34
	s_wait_dscnt 0x0
	v_fma_f32 v33, -v36, v37, v33
	ds_bpermute_b32 v37, v71, v34
	s_wait_dscnt 0x0
	v_fma_f32 v18, -v36, v37, v18
	ds_bpermute_b32 v37, v72, v34
	s_wait_dscnt 0x0
	v_fma_f32 v19, -v36, v37, v19
	ds_bpermute_b32 v37, v73, v34
	s_wait_dscnt 0x0
	v_fma_f32 v20, -v36, v37, v20
	ds_bpermute_b32 v37, v74, v34
	s_wait_dscnt 0x0
	v_fma_f32 v21, -v36, v37, v21
	ds_bpermute_b32 v37, v75, v34
	s_wait_dscnt 0x0
	v_fma_f32 v22, -v36, v37, v22
	ds_bpermute_b32 v37, v76, v34
	s_wait_dscnt 0x0
	v_fma_f32 v23, -v36, v37, v23
	ds_bpermute_b32 v37, v77, v34
	ds_bpermute_b32 v34, v78, v34
	s_wait_dscnt 0x1
	v_fma_f32 v24, -v36, v37, v24
	s_wait_dscnt 0x0
	v_fma_f32 v25, -v36, v34, v25
	ds_bpermute_b32 v34, v80, v35
	s_wait_dscnt 0x0
	v_fma_f32 v10, -v36, v34, v10
	ds_bpermute_b32 v34, v64, v35
	s_wait_dscnt 0x0
	v_fma_f32 v11, -v36, v34, v11
	ds_bpermute_b32 v34, v65, v35
	s_wait_dscnt 0x0
	v_fma_f32 v12, -v36, v34, v12
	ds_bpermute_b32 v34, v66, v35
	s_wait_dscnt 0x0
	v_fma_f32 v13, -v36, v34, v13
	ds_bpermute_b32 v34, v67, v35
	s_wait_dscnt 0x0
	v_fma_f32 v14, -v36, v34, v14
	ds_bpermute_b32 v34, v68, v35
	s_wait_dscnt 0x0
	v_fma_f32 v15, -v36, v34, v15
	ds_bpermute_b32 v34, v69, v35
	s_wait_dscnt 0x0
	v_fma_f32 v16, -v36, v34, v16
	ds_bpermute_b32 v34, v70, v35
	s_wait_dscnt 0x0
	v_fma_f32 v17, -v36, v34, v17
	ds_bpermute_b32 v34, v71, v35
	s_wait_dscnt 0x0
	v_fma_f32 v81, -v36, v34, v81
	ds_bpermute_b32 v34, v72, v35
	s_wait_dscnt 0x0
	v_fma_f32 v82, -v36, v34, v82
	ds_bpermute_b32 v34, v73, v35
	s_wait_dscnt 0x0
	v_fma_f32 v83, -v36, v34, v83
	ds_bpermute_b32 v34, v74, v35
	s_wait_dscnt 0x0
	v_fma_f32 v84, -v36, v34, v84
	ds_bpermute_b32 v34, v75, v35
	s_wait_dscnt 0x0
	v_fma_f32 v85, -v36, v34, v85
	ds_bpermute_b32 v34, v76, v35
	s_wait_dscnt 0x0
	v_fma_f32 v86, -v36, v34, v86
	ds_bpermute_b32 v34, v77, v35
	s_wait_dscnt 0x0
	v_fma_f32 v87, -v36, v34, v87
	ds_bpermute_b32 v34, v78, v35
	s_wait_dscnt 0x0
	v_fma_f32 v88, -v36, v34, v88
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e32 v34, 0, v24, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v34, v34, v87, s0
	ds_bpermute_b32 v35, v63, v34
	s_wait_dscnt 0x0
	v_add_f32_e32 v36, v34, v35
	ds_load_2addr_b32 v[34:35], v9 offset0:128 offset1:160
	s_wait_dscnt 0x0
	ds_bpermute_b32 v37, v80, v34
	s_wait_dscnt 0x0
	v_fma_f32 v89, -v36, v37, v26
	ds_bpermute_b32 v26, v64, v34
	s_wait_dscnt 0x0
	v_fma_f32 v90, -v36, v26, v27
	ds_bpermute_b32 v26, v65, v34
	s_wait_dscnt 0x0
	v_fma_f32 v91, -v36, v26, v28
	ds_bpermute_b32 v26, v66, v34
	s_wait_dscnt 0x0
	v_fma_f32 v92, -v36, v26, v29
	ds_bpermute_b32 v26, v67, v34
	s_wait_dscnt 0x0
	v_fma_f32 v93, -v36, v26, v30
	ds_bpermute_b32 v26, v68, v34
	s_wait_dscnt 0x0
	v_fma_f32 v94, -v36, v26, v31
	ds_bpermute_b32 v26, v69, v34
	s_wait_dscnt 0x0
	v_fma_f32 v95, -v36, v26, v32
	ds_bpermute_b32 v26, v70, v34
	s_wait_dscnt 0x0
	v_fma_f32 v96, -v36, v26, v33
	ds_bpermute_b32 v26, v71, v34
	s_wait_dscnt 0x0
	v_fma_f32 v97, -v36, v26, v18
	ds_bpermute_b32 v18, v72, v34
	s_wait_dscnt 0x0
	v_fma_f32 v98, -v36, v18, v19
	ds_bpermute_b32 v18, v73, v34
	s_wait_dscnt 0x0
	v_fma_f32 v99, -v36, v18, v20
	ds_bpermute_b32 v18, v74, v34
	s_wait_dscnt 0x0
	v_fma_f32 v100, -v36, v18, v21
	ds_bpermute_b32 v18, v75, v34
	s_wait_dscnt 0x0
	v_fma_f32 v101, -v36, v18, v22
	ds_bpermute_b32 v18, v76, v34
	s_wait_dscnt 0x0
	v_fma_f32 v102, -v36, v18, v23
	ds_bpermute_b32 v18, v77, v34
	s_wait_dscnt 0x0
	v_fma_f32 v103, -v36, v18, v24
	ds_bpermute_b32 v18, v78, v34
	s_wait_dscnt 0x0
	v_fma_f32 v104, -v36, v18, v25
	ds_bpermute_b32 v18, v80, v35
	s_wait_dscnt 0x0
	v_fma_f32 v105, -v36, v18, v10
	ds_bpermute_b32 v10, v64, v35
	s_wait_dscnt 0x0
	v_fma_f32 v106, -v36, v10, v11
	ds_bpermute_b32 v10, v65, v35
	s_wait_dscnt 0x0
	v_fma_f32 v107, -v36, v10, v12
	ds_bpermute_b32 v10, v66, v35
	s_wait_dscnt 0x0
	v_fma_f32 v108, -v36, v10, v13
	ds_bpermute_b32 v10, v67, v35
	s_wait_dscnt 0x0
	v_fma_f32 v109, -v36, v10, v14
	ds_bpermute_b32 v10, v68, v35
	s_wait_dscnt 0x0
	v_fma_f32 v110, -v36, v10, v15
	ds_bpermute_b32 v10, v69, v35
	s_wait_dscnt 0x0
	v_fma_f32 v111, -v36, v10, v16
	ds_bpermute_b32 v10, v70, v35
	s_wait_dscnt 0x0
	v_fma_f32 v112, -v36, v10, v17
	ds_bpermute_b32 v10, v71, v35
	s_wait_dscnt 0x0
	v_fma_f32 v81, -v36, v10, v81
	ds_bpermute_b32 v10, v72, v35
	s_wait_dscnt 0x0
	v_fma_f32 v82, -v36, v10, v82
	ds_bpermute_b32 v10, v73, v35
	s_wait_dscnt 0x0
	v_fma_f32 v83, -v36, v10, v83
	ds_bpermute_b32 v10, v74, v35
	s_wait_dscnt 0x0
	v_fma_f32 v84, -v36, v10, v84
	ds_bpermute_b32 v10, v75, v35
	s_wait_dscnt 0x0
	v_fma_f32 v85, -v36, v10, v85
	ds_bpermute_b32 v10, v76, v35
	s_wait_dscnt 0x0
	v_fma_f32 v86, -v36, v10, v86
	ds_bpermute_b32 v10, v77, v35
	s_wait_dscnt 0x0
	v_fma_f32 v87, -v36, v10, v87
	ds_bpermute_b32 v10, v78, v35
	s_wait_dscnt 0x0
	v_fma_f32 v88, -v36, v10, v88
	;;#ASMSTART
	;;#ASMEND
	ds_load_2addr_b32 v[15:16], v9 offset0:192 offset1:224
	v_cndmask_b32_e32 v10, 0, v104, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v10, v10, v88, s0
	ds_bpermute_b32 v11, v63, v10
	s_wait_dscnt 0x1
	ds_bpermute_b32 v9, v80, v15
	ds_bpermute_b32 v12, v74, v16
	ds_bpermute_b32 v13, v75, v16
	ds_bpermute_b32 v14, v76, v16
	s_wait_dscnt 0x4
	v_add_f32_e32 v113, v10, v11
	ds_bpermute_b32 v10, v72, v16
	ds_bpermute_b32 v11, v73, v16
	s_wait_dscnt 0x5
	v_fma_f32 v33, -v113, v9, v89
	ds_bpermute_b32 v9, v64, v15
	s_wait_dscnt 0x5
	v_fma_f32 v12, -v113, v12, v84
	s_wait_dscnt 0x4
	v_fma_f32 v13, -v113, v13, v85
	s_wait_dscnt 0x3
	v_fma_f32 v14, -v113, v14, v86
	s_wait_dscnt 0x2
	v_fma_f32 v10, -v113, v10, v82
	s_wait_dscnt 0x1
	v_fma_f32 v11, -v113, v11, v83
	s_wait_dscnt 0x0
	v_fma_f32 v34, -v113, v9, v90
	ds_bpermute_b32 v9, v65, v15
	s_wait_dscnt 0x0
	v_fma_f32 v35, -v113, v9, v91
	ds_bpermute_b32 v9, v66, v15
	s_wait_dscnt 0x0
	v_fma_f32 v36, -v113, v9, v92
	ds_bpermute_b32 v9, v67, v15
	s_wait_dscnt 0x0
	v_fma_f32 v37, -v113, v9, v93
	ds_bpermute_b32 v9, v68, v15
	s_wait_dscnt 0x0
	v_fma_f32 v38, -v113, v9, v94
	ds_bpermute_b32 v9, v69, v15
	s_wait_dscnt 0x0
	v_fma_f32 v39, -v113, v9, v95
	ds_bpermute_b32 v9, v70, v15
	s_wait_dscnt 0x0
	v_fma_f32 v40, -v113, v9, v96
	ds_bpermute_b32 v9, v71, v15
	s_wait_dscnt 0x0
	v_fma_f32 v25, -v113, v9, v97
	ds_bpermute_b32 v9, v72, v15
	s_wait_dscnt 0x0
	v_fma_f32 v26, -v113, v9, v98
	ds_bpermute_b32 v9, v73, v15
	s_wait_dscnt 0x0
	v_fma_f32 v27, -v113, v9, v99
	ds_bpermute_b32 v9, v74, v15
	s_wait_dscnt 0x0
	v_fma_f32 v28, -v113, v9, v100
	ds_bpermute_b32 v9, v75, v15
	s_wait_dscnt 0x0
	v_fma_f32 v29, -v113, v9, v101
	ds_bpermute_b32 v9, v76, v15
	s_wait_dscnt 0x0
	v_fma_f32 v30, -v113, v9, v102
	ds_bpermute_b32 v9, v77, v15
	s_wait_dscnt 0x0
	v_fma_f32 v31, -v113, v9, v103
	ds_bpermute_b32 v9, v78, v15
	ds_bpermute_b32 v15, v77, v16
	s_wait_dscnt 0x1
	v_fma_f32 v32, -v113, v9, v104
	ds_bpermute_b32 v9, v80, v16
	s_wait_dscnt 0x1
	v_fma_f32 v15, -v113, v15, v87
	s_wait_dscnt 0x0
	v_fma_f32 v17, -v113, v9, v105
	ds_bpermute_b32 v9, v64, v16
	s_wait_dscnt 0x0
	v_fma_f32 v18, -v113, v9, v106
	ds_bpermute_b32 v9, v65, v16
	s_wait_dscnt 0x0
	v_fma_f32 v19, -v113, v9, v107
	ds_bpermute_b32 v9, v66, v16
	s_wait_dscnt 0x0
	v_fma_f32 v20, -v113, v9, v108
	ds_bpermute_b32 v9, v67, v16
	s_wait_dscnt 0x0
	v_fma_f32 v21, -v113, v9, v109
	ds_bpermute_b32 v9, v68, v16
	s_wait_dscnt 0x0
	v_fma_f32 v22, -v113, v9, v110
	ds_bpermute_b32 v9, v69, v16
	s_wait_dscnt 0x0
	v_fma_f32 v23, -v113, v9, v111
	ds_bpermute_b32 v9, v70, v16
	s_wait_dscnt 0x0
	v_fma_f32 v24, -v113, v9, v112
	ds_bpermute_b32 v9, v71, v16
	ds_bpermute_b32 v16, v78, v16
	s_wait_dscnt 0x1
	v_fma_f32 v9, -v113, v9, v81
	s_wait_dscnt 0x0
	v_fma_f32 v16, -v113, v16, v88
	;;#ASMSTART
	;;#ASMEND
	s_cbranch_scc0 .LBB9_27
; %bb.28:                               ;   in Loop: Header=BB9_26 Depth=1
	v_wmma_f32_16x16x16_f16 v[80:87], v[53:56], v[45:48], v[1:8]
	v_cvt_f16_f32_e32 v40.h, v40
	v_cvt_f16_f32_e32 v40.l, v39
	v_cvt_f16_f32_e32 v39.h, v38
	s_delay_alu instid0(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[80:87], v[53:56], v[41:44], v[80:87]
	v_cvt_f16_f32_e32 v39.l, v37
	v_cvt_f16_f32_e32 v38.h, v36
	v_cvt_f16_f32_e32 v38.l, v35
	v_cvt_f16_f32_e32 v37.h, v34
	v_cvt_f16_f32_e32 v37.l, v33
	v_cvt_f16_f32_e32 v32.h, v32
	v_cvt_f16_f32_e32 v32.l, v31
	v_cvt_f16_f32_e32 v31.h, v30
	v_cvt_f16_f32_e32 v31.l, v29
	v_wmma_f32_16x16x16_f16 v[80:87], v[49:52], v[37:40], v[80:87]
	v_cvt_f16_f32_e32 v30.h, v28
	v_cvt_f16_f32_e32 v30.l, v27
	v_cvt_f16_f32_e32 v29.h, v26
	v_cvt_f16_f32_e32 v29.l, v25
	v_cvt_f16_f32_e32 v28.h, v24
	v_cvt_f16_f32_e32 v28.l, v23
	v_cvt_f16_f32_e32 v27.h, v22
	v_cvt_f16_f32_e32 v27.l, v21
	v_wmma_f32_16x16x16_f16 v[80:87], v[49:52], v[29:32], v[80:87]
	v_cvt_f16_f32_e32 v26.h, v20
	v_cvt_f16_f32_e32 v26.l, v19
	v_cvt_f16_f32_e32 v25.h, v18
	v_cvt_f16_f32_e32 v25.l, v17
	v_cvt_f16_f32_e32 v36.h, v16
	v_cvt_f16_f32_e32 v36.l, v15
	v_cvt_f16_f32_e32 v35.h, v14
	v_cvt_f16_f32_e32 v35.l, v13
	v_wmma_f32_16x16x16_f16 v[80:87], v[49:52], v[25:28], v[80:87]
	v_cvt_f16_f32_e32 v34.h, v12
	v_cvt_f16_f32_e32 v34.l, v11
	v_cvt_f16_f32_e32 v33.h, v10
	v_cvt_f16_f32_e32 v33.l, v9
	v_dual_add_f32 v16, v8, v8 :: v_dual_add_f32 v15, v7, v7
	v_dual_add_f32 v14, v6, v6 :: v_dual_add_f32 v13, v5, v5
	v_add_f32_e32 v12, v4, v4
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_wmma_f32_16x16x16_f16 v[80:87], v[49:52], v[33:36], v[80:87]
	v_dual_add_f32 v11, v3, v3 :: v_dual_add_f32 v10, v2, v2
	v_dual_add_f32 v9, v1, v1 :: v_dual_mul_f32 v24, 0x40400000, v8
	v_dual_add_f32 v17, 0, v80 :: v_dual_mul_f32 v22, 0x40400000, v6
	v_mul_f32_e32 v20, 0x40400000, v4
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_wmma_f32_16x16x16_f16 v[9:16], v[53:56], v[45:48], v[9:16]
	v_dual_mul_f32 v18, 0x40400000, v2 :: v_dual_add_f32 v17, v17, v81
	v_mul_f32_e32 v6, 4.0, v6
	v_mul_f32_e32 v8, 4.0, v8
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[9:16], v[53:56], v[41:44], v[9:16]
	v_mul_f32_e32 v23, 0x40400000, v7
	v_add_f32_e32 v17, v17, v82
	v_dual_mul_f32 v21, 0x40400000, v5 :: v_dual_mul_f32 v4, 4.0, v4
	v_wmma_f32_16x16x16_f16 v[9:16], v[49:52], v[37:40], v[9:16]
	v_mul_f32_e32 v19, 0x40400000, v3
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_f32_e32 v17, v17, v83
	v_dual_mul_f32 v7, 4.0, v7 :: v_dual_mul_f32 v2, 4.0, v2
	v_wmma_f32_16x16x16_f16 v[9:16], v[49:52], v[29:32], v[9:16]
	v_mul_f32_e32 v5, 4.0, v5
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v17, v17, v84
	v_mul_f32_e32 v3, 4.0, v3
	s_add_co_i32 s6, s6, 1
	v_wmma_f32_16x16x16_f16 v[9:16], v[49:52], v[25:28], v[9:16]
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s6, s3
	v_add_f32_e32 v17, v17, v85
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_wmma_f32_16x16x16_f16 v[9:16], v[49:52], v[33:36], v[9:16]
	v_add_f32_e32 v17, v17, v86
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v17, v17, v87
	v_add_f32_e32 v9, v17, v9
	v_mul_f32_e32 v17, 0x40400000, v1
	v_mul_f32_e32 v1, 4.0, v1
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_f32_e32 v9, v9, v10
	v_wmma_f32_16x16x16_f16 v[17:24], v[53:56], v[45:48], v[17:24]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[1:8], v[53:56], v[45:48], v[1:8]
	v_fma_mix_f32 v10, v45, s7, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_add_f32_e32 v9, v9, v11
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[17:24], v[53:56], v[41:44], v[17:24]
	v_wmma_f32_16x16x16_f16 v[1:8], v[53:56], v[41:44], v[1:8]
	v_fma_mix_f32 v11, v46, s7, neg(0) op_sel_hi:[1,0,0]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_add_f32_e32 v9, v9, v12
	v_wmma_f32_16x16x16_f16 v[17:24], v[49:52], v[37:40], v[17:24]
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[1:8], v[49:52], v[37:40], v[1:8]
	v_fma_mix_f32 v12, v46, s7, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_add_f32_e32 v9, v9, v13
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[17:24], v[49:52], v[29:32], v[17:24]
	v_wmma_f32_16x16x16_f16 v[1:8], v[49:52], v[29:32], v[1:8]
	v_fma_mix_f32 v13, v47, s7, neg(0) op_sel_hi:[1,0,0]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_add_f32_e32 v9, v9, v14
	v_wmma_f32_16x16x16_f16 v[17:24], v[49:52], v[25:28], v[17:24]
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[1:8], v[49:52], v[25:28], v[1:8]
	v_fma_mix_f32 v14, v47, s7, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_add_f32_e32 v9, v9, v15
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_wmma_f32_16x16x16_f16 v[17:24], v[49:52], v[33:36], v[17:24]
	v_wmma_f32_16x16x16_f16 v[1:8], v[49:52], v[33:36], v[1:8]
	v_fma_mix_f32 v15, v48, s7, neg(0) op_sel_hi:[1,0,0]
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v9, v9, v16
	v_fma_mix_f32 v16, v48, s7, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_add_f32_e32 v9, v9, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v9, v9, v18
	v_fma_mix_f32 v18, v41, s7, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_add_f32_e32 v9, v9, v19
	v_fma_mix_f32 v19, v42, s7, neg(0) op_sel_hi:[1,0,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v9, v9, v20
	v_fma_mix_f32 v20, v42, s7, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_add_f32_e32 v9, v9, v21
	v_fma_mix_f32 v21, v43, s7, neg(0) op_sel_hi:[1,0,0]
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v9, v9, v22
	v_fma_mix_f32 v22, v43, s7, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_add_f32_e32 v17, v9, v23
	v_fma_mix_f32 v9, v45, s7, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v23, v44, s7, neg(0) op_sel_hi:[1,0,0]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_f32_e32 v17, v17, v24
	v_fma_mix_f32 v24, v44, s7, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_wmma_f32_16x16x16_f16 v[9:16], v[49:52], v[37:40], v[9:16]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_f32_e32 v1, v17, v1
	v_fma_mix_f32 v17, v41, s7, neg(0) op_sel_hi:[1,0,0]
	v_wmma_f32_16x16x16_f16 v[9:16], v[49:52], v[29:32], v[9:16]
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_f32_e32 v1, v1, v2
	v_wmma_f32_16x16x16_f16 v[17:24], v[49:52], v[37:40], v[17:24]
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_wmma_f32_16x16x16_f16 v[9:16], v[49:52], v[25:28], v[9:16]
	v_add_f32_e32 v1, v1, v3
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
	v_cndmask_b32_e32 v1, 0x3a83126f, v79, vcc_lo
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
		.amdhsa_next_free_vgpr 114
		.amdhsa_next_free_sgpr 9
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
	.set .L_Z5probeILi2ELb1ELb1EEvPfPKfi.num_vgpr, 114
	.set .L_Z5probeILi2ELb1ELb1EEvPfPKfi.num_agpr, 0
	.set .L_Z5probeILi2ELb1ELb1EEvPfPKfi.numbered_sgpr, 9
	.set .L_Z5probeILi2ELb1ELb1EEvPfPKfi.num_named_barrier, 0
	.set .L_Z5probeILi2ELb1ELb1EEvPfPKfi.private_seg_size, 0
	.set .L_Z5probeILi2ELb1ELb1EEvPfPKfi.uses_vcc, 1
	.set .L_Z5probeILi2ELb1ELb1EEvPfPKfi.uses_flat_scratch, 0
	.set .L_Z5probeILi2ELb1ELb1EEvPfPKfi.has_dyn_sized_stack, 0
	.set .L_Z5probeILi2ELb1ELb1EEvPfPKfi.has_recursion, 0
	.set .L_Z5probeILi2ELb1ELb1EEvPfPKfi.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 25864
; TotalNumSgprs: 11
; NumVgprs: 114
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 14
; NumSGPRsForWavesPerEU: 11
; NumVGPRsForWavesPerEU: 114
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
	.type	__hip_cuid_371df25d07f12a49,@object ; @__hip_cuid_371df25d07f12a49
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_371df25d07f12a49
__hip_cuid_371df25d07f12a49:
	.byte	0                               ; 0x0
	.size	__hip_cuid_371df25d07f12a49, 1

	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git 8f497e0992fb7513f7f78a6f6b6f1056c375e961)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_371df25d07f12a49
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
    .vgpr_count:     189
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
    .sgpr_count:     11
    .sgpr_spill_count: 0
    .symbol:         _Z5probeILi8ELb1ELb1EEvPfPKfi.kd
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
    .name:           _Z5probeILi4ELb0ELb1EEvPfPKfi
    .private_segment_fixed_size: 0
    .sgpr_count:     10
    .sgpr_spill_count: 0
    .symbol:         _Z5probeILi4ELb0ELb1EEvPfPKfi.kd
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
    .name:           _Z5probeILi4ELb1ELb1EEvPfPKfi
    .private_segment_fixed_size: 0
    .sgpr_count:     11
    .sgpr_spill_count: 0
    .symbol:         _Z5probeILi4ELb1ELb1EEvPfPKfi.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     124
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
    .vgpr_count:     117
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
    .sgpr_count:     11
    .sgpr_spill_count: 0
    .symbol:         _Z5probeILi2ELb1ELb1EEvPfPKfi.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     114
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target:   amdgcn-amd-amdhsa--gfx1201
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
