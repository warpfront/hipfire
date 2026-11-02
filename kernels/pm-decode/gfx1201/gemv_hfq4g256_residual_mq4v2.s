.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
.amdhsa_code_object_version 6
.text
.protected gemv_mq4g256v2_residual
.globl gemv_mq4g256v2_residual
.p2align 8
.type gemv_mq4g256v2_residual,@function
gemv_mq4g256v2_residual:
	s_load_b64 s[10:11], s[0:1], 0x18
	s_lshl_b32 s12, ttmp9, 1
	s_wait_kmcnt 0x0
	s_cmp_ge_i32 s12, s10
	s_cbranch_scc1 .Lres_end
	s_load_b128 s[4:7], s[0:1], 0x0
	s_load_b64 s[8:9], s[0:1], 0x10
	s_wait_kmcnt 0x0
	s_add_co_i32 s13, s12, 1
	s_cmp_lt_i32 s13, s10
	s_cselect_b32 s18, 1, 0
	s_cselect_b32 s13, s13, s12
	s_lshr_b32 s14, s11, 8
	s_mul_i32 s19, s14, 0x88
	s_mul_i32 s16, s19, s12
	s_mul_i32 s17, s19, s13
	s_lshr_b32 s2, s14, 2
	s_mov_b32 s15, 0
	s_mov_b32 s20, s4
	s_and_b32 s21, s5, 0xffff
	s_mov_b32 s22, -1
	s_mov_b32 s23, 0x31004000
	v_lshlrev_b32_e32 v1, 2, v0
	v_mov_b32_e32 v24, 0
	v_mov_b32_e32 v25, 0
	v_mov_b32_e32 v26, 0
	v_mov_b32_e32 v27, 0
	v_mov_b32_e32 v28, 0
	v_mov_b32_e32 v29, 0
	v_mov_b32_e32 v30, 0
	v_mov_b32_e32 v31, 0
	s_cmp_eq_u32 s2, 0
	s_cbranch_scc1 .Lres_tail
	.Lres_quad:
	s_mul_i32 s24, s15, 0x88
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s25, s16, s24
	s_add_co_i32 s26, s17, s24
	s_lshl_b32 s27, s15, 10
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v3, s25
	v_add_nc_u32_e32 v2, s25, v1
	v_mov_b32_e32 v45, s26
	v_add_nc_u32_e32 v44, s26, v1
	v_lshlrev_b32_e32 v46, 5, v0
	v_add_nc_u32_e32 v46, s27, v46
	buffer_load_b32 v92, v2, s[20:23], null offen offset:8 scope:SCOPE_DEV
	buffer_load_b32 v94, v2, s[20:23], null offen offset:144 scope:SCOPE_DEV
	buffer_load_b32 v95, v44, s[20:23], null offen offset:144 scope:SCOPE_DEV
	buffer_load_b32 v96, v2, s[20:23], null offen offset:280 scope:SCOPE_DEV
	buffer_load_b32 v97, v44, s[20:23], null offen offset:280 scope:SCOPE_DEV
	buffer_load_b32 v99, v44, s[20:23], null offen offset:416 scope:SCOPE_DEV
	buffer_load_b64 v[76:77], v3, s[20:23], null offen scope:SCOPE_DEV
	buffer_load_b64 v[80:81], v3, s[20:23], null offen offset:136 scope:SCOPE_DEV
	buffer_load_b64 v[84:85], v3, s[20:23], null offen offset:272 scope:SCOPE_DEV
	buffer_load_b64 v[88:89], v3, s[20:23], null offen offset:408 scope:SCOPE_DEV
	buffer_load_b64 v[78:79], v45, s[20:23], null offen scope:SCOPE_DEV
	buffer_load_b64 v[82:83], v45, s[20:23], null offen offset:136 scope:SCOPE_DEV
	buffer_load_b64 v[86:87], v45, s[20:23], null offen offset:272 scope:SCOPE_DEV
	buffer_load_b64 v[90:91], v45, s[20:23], null offen offset:408 scope:SCOPE_DEV
	buffer_load_b32 v93, v44, s[20:23], null offen offset:8 scope:SCOPE_DEV
	buffer_load_b32 v98, v2, s[20:23], null offen offset:416 scope:SCOPE_DEV
	global_load_b128 v[8:11], v46, s[6:7]
	global_load_b128 v[49:52], v46, s[6:7] offset:1024
	global_load_b128 v[58:61], v46, s[6:7] offset:2048
	v_cmp_gt_u32_e32 vcc_lo, 16, v0
	s_wait_loadcnt 0x12
	v_bfe_u32 v16, v92, 0, 4
	v_cvt_f32_ubyte0_e32 v16, v16
	s_wait_loadcnt 0x11
	v_bfe_u32 v18, v94, 0, 4
	v_cvt_f32_ubyte0_e32 v18, v18
	s_wait_loadcnt 0xc
	v_cndmask_b32_e32 v76, v77, v76, vcc_lo
	s_wait_loadcnt 0xb
	v_cndmask_b32_e32 v80, v81, v80, vcc_lo
	s_wait_loadcnt 0xa
	v_cndmask_b32_e32 v84, v85, v84, vcc_lo
	s_wait_loadcnt 0x9
	v_cndmask_b32_e32 v88, v89, v88, vcc_lo
	s_wait_loadcnt 0x7
	v_cndmask_b32_e32 v82, v83, v82, vcc_lo
	s_wait_loadcnt 0x1
	v_cndmask_b32_e32 v78, v79, v78, vcc_lo
	v_cndmask_b32_e32 v86, v87, v86, vcc_lo
	v_cndmask_b32_e32 v90, v91, v90, vcc_lo
	v_bfe_u32 v17, v93, 0, 4
	v_cvt_f32_ubyte0_e32 v17, v17
	v_bfe_u32 v19, v95, 0, 4
	v_cvt_f32_ubyte0_e32 v19, v19
	v_bfe_u32 v20, v96, 0, 4
	v_cvt_f32_ubyte0_e32 v20, v20
	v_bfe_u32 v21, v97, 0, 4
	v_cvt_f32_ubyte0_e32 v21, v21
	v_bfe_u32 v22, v98, 0, 4
	v_cvt_f32_ubyte0_e32 v22, v22
	v_bfe_u32 v23, v99, 0, 4
	v_cvt_f32_ubyte0_e32 v23, v23
	s_wait_loadcnt 0x0
	global_load_b128 v[67:70], v46, s[6:7] offset:3072
	global_load_b128 v[12:15], v46, s[6:7] offset:16
	global_load_b128 v[53:56], v46, s[6:7] offset:1040
	global_load_b128 v[62:65], v46, s[6:7] offset:2064
	global_load_b128 v[71:74], v46, s[6:7] offset:3088
	v_fma_mix_f32 v16, v76, v16, v76 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v17, v78, v17, v78 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v18, v80, v18, v80 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v19, v82, v19, v82 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v20, v84, v20, v84 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v21, v86, v21, v86 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v22, v88, v22, v88 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v23, v90, v23, v90 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_bfe_u32 v32, v92, 4, 4
	v_cvt_f32_ubyte0_e32 v32, v32
	v_bfe_u32 v33, v93, 4, 4
	v_cvt_f32_ubyte0_e32 v33, v33
	v_bfe_u32 v34, v94, 4, 4
	v_cvt_f32_ubyte0_e32 v34, v34
	v_bfe_u32 v35, v95, 4, 4
	v_cvt_f32_ubyte0_e32 v35, v35
	v_bfe_u32 v36, v96, 4, 4
	v_cvt_f32_ubyte0_e32 v36, v36
	v_bfe_u32 v37, v97, 4, 4
	v_cvt_f32_ubyte0_e32 v37, v37
	v_bfe_u32 v38, v98, 4, 4
	v_cvt_f32_ubyte0_e32 v38, v38
	v_bfe_u32 v39, v99, 4, 4
	v_cvt_f32_ubyte0_e32 v39, v39
	v_fma_mix_f32 v32, v76, v32, v76 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v33, v78, v33, v78 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v34, v80, v34, v80 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v35, v82, v35, v82 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v36, v84, v36, v84 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v37, v86, v37, v86 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v38, v88, v38, v88 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v39, v90, v39, v90 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_dual_mul_f32 v100, v8, v16 :: v_dual_mul_f32 v101, v49, v18
	s_wait_loadcnt 0x4
	v_dual_mul_f32 v102, v58, v20 :: v_dual_mul_f32 v103, v67, v22
	v_dual_mul_f32 v104, v9, v33 :: v_dual_mul_f32 v105, v50, v35
	v_dual_mul_f32 v106, v59, v37 :: v_dual_mul_f32 v107, v68, v39
	v_dual_fmac_f32 v100, v9, v32 :: v_dual_fmac_f32 v101, v50, v34
	v_dual_fmac_f32 v102, v59, v36 :: v_dual_fmac_f32 v103, v68, v38
	v_dual_fmac_f32 v104, v8, v17 :: v_dual_fmac_f32 v105, v49, v19
	v_dual_fmac_f32 v106, v58, v21 :: v_dual_fmac_f32 v107, v67, v23
	v_bfe_u32 v16, v92, 8, 4
	v_bfe_u32 v17, v93, 8, 4
	v_bfe_u32 v18, v94, 8, 4
	v_bfe_u32 v19, v95, 8, 4
	v_bfe_u32 v20, v96, 8, 4
	v_bfe_u32 v21, v97, 8, 4
	v_bfe_u32 v22, v98, 8, 4
	v_bfe_u32 v23, v99, 8, 4
	v_cvt_f32_ubyte0_e32 v16, v16
	v_cvt_f32_ubyte0_e32 v17, v17
	v_cvt_f32_ubyte0_e32 v18, v18
	v_cvt_f32_ubyte0_e32 v19, v19
	v_cvt_f32_ubyte0_e32 v20, v20
	v_cvt_f32_ubyte0_e32 v21, v21
	v_cvt_f32_ubyte0_e32 v22, v22
	v_cvt_f32_ubyte0_e32 v23, v23
	v_fma_mix_f32 v16, v76, v16, v76 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v17, v78, v17, v78 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v18, v80, v18, v80 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v19, v82, v19, v82 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v20, v84, v20, v84 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v21, v86, v21, v86 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v22, v88, v22, v88 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v23, v90, v23, v90 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_dual_fmac_f32 v100, v10, v16 :: v_dual_fmac_f32 v101, v51, v18
	v_dual_fmac_f32 v102, v60, v20 :: v_dual_fmac_f32 v103, v69, v22
	v_dual_fmac_f32 v104, v10, v17 :: v_dual_fmac_f32 v105, v51, v19
	v_dual_fmac_f32 v106, v60, v21 :: v_dual_fmac_f32 v107, v69, v23
	v_bfe_u32 v16, v92, 12, 4
	v_bfe_u32 v17, v93, 12, 4
	v_bfe_u32 v18, v94, 12, 4
	v_bfe_u32 v19, v95, 12, 4
	v_bfe_u32 v20, v96, 12, 4
	v_bfe_u32 v21, v97, 12, 4
	v_bfe_u32 v22, v98, 12, 4
	v_bfe_u32 v23, v99, 12, 4
	v_cvt_f32_ubyte0_e32 v16, v16
	v_cvt_f32_ubyte0_e32 v17, v17
	v_cvt_f32_ubyte0_e32 v18, v18
	v_cvt_f32_ubyte0_e32 v19, v19
	v_cvt_f32_ubyte0_e32 v20, v20
	v_cvt_f32_ubyte0_e32 v21, v21
	v_cvt_f32_ubyte0_e32 v22, v22
	v_cvt_f32_ubyte0_e32 v23, v23
	v_fma_mix_f32 v16, v76, v16, v76 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v17, v78, v17, v78 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v18, v80, v18, v80 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v19, v82, v19, v82 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v20, v84, v20, v84 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v21, v86, v21, v86 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v22, v88, v22, v88 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v23, v90, v23, v90 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_dual_fmac_f32 v100, v11, v16 :: v_dual_fmac_f32 v101, v52, v18
	v_dual_fmac_f32 v102, v61, v20 :: v_dual_fmac_f32 v103, v70, v22
	v_dual_fmac_f32 v104, v11, v17 :: v_dual_fmac_f32 v105, v52, v19
	v_dual_fmac_f32 v106, v61, v21 :: v_dual_fmac_f32 v107, v70, v23
	s_wait_loadcnt 0x3
	v_bfe_u32 v16, v92, 16, 4
	v_bfe_u32 v17, v93, 16, 4
	s_wait_loadcnt 0x2
	v_bfe_u32 v18, v94, 16, 4
	v_bfe_u32 v19, v95, 16, 4
	s_wait_loadcnt 0x1
	v_bfe_u32 v20, v96, 16, 4
	v_bfe_u32 v21, v97, 16, 4
	s_wait_loadcnt 0x0
	v_bfe_u32 v22, v98, 16, 4
	v_bfe_u32 v23, v99, 16, 4
	v_cvt_f32_ubyte0_e32 v16, v16
	v_cvt_f32_ubyte0_e32 v17, v17
	v_cvt_f32_ubyte0_e32 v18, v18
	v_cvt_f32_ubyte0_e32 v19, v19
	v_cvt_f32_ubyte0_e32 v20, v20
	v_cvt_f32_ubyte0_e32 v21, v21
	v_cvt_f32_ubyte0_e32 v22, v22
	v_cvt_f32_ubyte0_e32 v23, v23
	v_fma_mix_f32 v16, v76, v16, v76 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v17, v78, v17, v78 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v18, v80, v18, v80 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v19, v82, v19, v82 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v20, v84, v20, v84 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v21, v86, v21, v86 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v22, v88, v22, v88 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v23, v90, v23, v90 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_dual_fmac_f32 v100, v12, v16 :: v_dual_fmac_f32 v101, v53, v18
	v_dual_fmac_f32 v102, v62, v20 :: v_dual_fmac_f32 v103, v71, v22
	v_dual_fmac_f32 v104, v12, v17 :: v_dual_fmac_f32 v105, v53, v19
	v_dual_fmac_f32 v106, v62, v21 :: v_dual_fmac_f32 v107, v71, v23
	v_bfe_u32 v16, v92, 20, 4
	v_bfe_u32 v17, v93, 20, 4
	v_bfe_u32 v18, v94, 20, 4
	v_bfe_u32 v19, v95, 20, 4
	v_bfe_u32 v20, v96, 20, 4
	v_bfe_u32 v21, v97, 20, 4
	v_bfe_u32 v22, v98, 20, 4
	v_bfe_u32 v23, v99, 20, 4
	v_cvt_f32_ubyte0_e32 v16, v16
	v_cvt_f32_ubyte0_e32 v17, v17
	v_cvt_f32_ubyte0_e32 v18, v18
	v_cvt_f32_ubyte0_e32 v19, v19
	v_cvt_f32_ubyte0_e32 v20, v20
	v_cvt_f32_ubyte0_e32 v21, v21
	v_cvt_f32_ubyte0_e32 v22, v22
	v_cvt_f32_ubyte0_e32 v23, v23
	v_fma_mix_f32 v16, v76, v16, v76 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v17, v78, v17, v78 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v18, v80, v18, v80 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v19, v82, v19, v82 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v20, v84, v20, v84 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v21, v86, v21, v86 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v22, v88, v22, v88 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v23, v90, v23, v90 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_dual_fmac_f32 v100, v13, v16 :: v_dual_fmac_f32 v101, v54, v18
	v_dual_fmac_f32 v102, v63, v20 :: v_dual_fmac_f32 v103, v72, v22
	v_dual_fmac_f32 v104, v13, v17 :: v_dual_fmac_f32 v105, v54, v19
	v_dual_fmac_f32 v106, v63, v21 :: v_dual_fmac_f32 v107, v72, v23
	v_bfe_u32 v16, v92, 24, 4
	v_bfe_u32 v17, v93, 24, 4
	v_bfe_u32 v18, v94, 24, 4
	v_bfe_u32 v19, v95, 24, 4
	v_bfe_u32 v20, v96, 24, 4
	v_bfe_u32 v21, v97, 24, 4
	v_bfe_u32 v22, v98, 24, 4
	v_bfe_u32 v23, v99, 24, 4
	v_cvt_f32_ubyte0_e32 v16, v16
	v_cvt_f32_ubyte0_e32 v17, v17
	v_cvt_f32_ubyte0_e32 v18, v18
	v_cvt_f32_ubyte0_e32 v19, v19
	v_cvt_f32_ubyte0_e32 v20, v20
	v_cvt_f32_ubyte0_e32 v21, v21
	v_cvt_f32_ubyte0_e32 v22, v22
	v_cvt_f32_ubyte0_e32 v23, v23
	v_fma_mix_f32 v16, v76, v16, v76 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v17, v78, v17, v78 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v18, v80, v18, v80 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v19, v82, v19, v82 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v20, v84, v20, v84 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v21, v86, v21, v86 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v22, v88, v22, v88 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v23, v90, v23, v90 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_dual_fmac_f32 v100, v14, v16 :: v_dual_fmac_f32 v101, v55, v18
	v_dual_fmac_f32 v102, v64, v20 :: v_dual_fmac_f32 v103, v73, v22
	v_dual_fmac_f32 v104, v14, v17 :: v_dual_fmac_f32 v105, v55, v19
	v_dual_fmac_f32 v106, v64, v21 :: v_dual_fmac_f32 v107, v73, v23
	v_bfe_u32 v16, v92, 28, 4
	v_bfe_u32 v17, v93, 28, 4
	v_bfe_u32 v18, v94, 28, 4
	v_bfe_u32 v19, v95, 28, 4
	v_bfe_u32 v20, v96, 28, 4
	v_bfe_u32 v21, v97, 28, 4
	v_bfe_u32 v22, v98, 28, 4
	v_bfe_u32 v23, v99, 28, 4
	v_cvt_f32_ubyte0_e32 v16, v16
	v_cvt_f32_ubyte0_e32 v17, v17
	v_cvt_f32_ubyte0_e32 v18, v18
	v_cvt_f32_ubyte0_e32 v19, v19
	v_cvt_f32_ubyte0_e32 v20, v20
	v_cvt_f32_ubyte0_e32 v21, v21
	v_cvt_f32_ubyte0_e32 v22, v22
	v_cvt_f32_ubyte0_e32 v23, v23
	v_fma_mix_f32 v16, v76, v16, v76 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v17, v78, v17, v78 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v18, v80, v18, v80 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v19, v82, v19, v82 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v20, v84, v20, v84 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v21, v86, v21, v86 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v22, v88, v22, v88 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v23, v90, v23, v90 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_dual_fmac_f32 v100, v15, v16 :: v_dual_fmac_f32 v101, v56, v18
	v_dual_fmac_f32 v102, v65, v20 :: v_dual_fmac_f32 v103, v74, v22
	v_dual_fmac_f32 v104, v15, v17 :: v_dual_fmac_f32 v105, v56, v19
	v_dual_fmac_f32 v106, v65, v21 :: v_dual_fmac_f32 v107, v74, v23
	v_dual_add_f32 v24, v100, v24 :: v_dual_add_f32 v25, v101, v25
	v_dual_add_f32 v26, v102, v26 :: v_dual_add_f32 v27, v103, v27
	v_dual_add_f32 v28, v28, v104 :: v_dual_add_f32 v29, v29, v105
	v_dual_add_f32 v30, v30, v106 :: v_dual_add_f32 v31, v31, v107
	s_add_co_i32 s15, s15, 4
	s_add_co_i32 s2, s2, -1
	s_cmp_lg_u32 s2, 0
	s_cbranch_scc1 .Lres_quad
	.Lres_tail:
	s_cmp_lt_u32 s15, s14
	s_cbranch_scc0 .Lres_fold
	s_mul_i32 s24, s15, 0x88
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s25, s16, s24
	s_add_co_i32 s26, s17, s24
	s_lshl_b32 s27, s15, 10
	v_lshlrev_b32_e32 v3, 5, v0
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v3, s27, v3
	global_load_b128 v[8:11], v3, s[6:7]
	global_load_b128 v[12:15], v3, s[6:7] offset:16
	s_wait_loadcnt 0x0
	v_mov_b32_e32 v3, s25
	buffer_load_b64 v[4:5], v3, s[20:23], null offen scope:SCOPE_DEV
	v_add_nc_u32_e32 v2, s25, v1
	buffer_load_b32 v16, v2, s[20:23], null offen offset:8 scope:SCOPE_DEV
	s_wait_loadcnt 0x0
	v_cmp_gt_u32_e32 vcc_lo, 16, v0
	v_cndmask_b32_e32 v34, v5, v4, vcc_lo
	v_bfe_u32 v35, v16, 0, 4
	v_cvt_f32_ubyte0_e32 v35, v35
	v_fma_mix_f32 v36, v34, v35, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_mul_f32_e32 v32, v8, v36
	v_bfe_u32 v35, v16, 4, 4
	v_cvt_f32_ubyte0_e32 v35, v35
	v_fma_mix_f32 v36, v34, v35, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fmac_f32_e32 v32, v9, v36
	v_bfe_u32 v35, v16, 8, 4
	v_cvt_f32_ubyte0_e32 v35, v35
	v_fma_mix_f32 v36, v34, v35, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fmac_f32_e32 v32, v10, v36
	v_bfe_u32 v35, v16, 12, 4
	v_cvt_f32_ubyte0_e32 v35, v35
	v_fma_mix_f32 v36, v34, v35, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fmac_f32_e32 v32, v11, v36
	v_bfe_u32 v35, v16, 16, 4
	v_cvt_f32_ubyte0_e32 v35, v35
	v_fma_mix_f32 v36, v34, v35, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fmac_f32_e32 v32, v12, v36
	v_bfe_u32 v35, v16, 20, 4
	v_cvt_f32_ubyte0_e32 v35, v35
	v_fma_mix_f32 v36, v34, v35, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fmac_f32_e32 v32, v13, v36
	v_bfe_u32 v35, v16, 24, 4
	v_cvt_f32_ubyte0_e32 v35, v35
	v_fma_mix_f32 v36, v34, v35, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fmac_f32_e32 v32, v14, v36
	v_bfe_u32 v35, v16, 28, 4
	v_cvt_f32_ubyte0_e32 v35, v35
	v_fma_mix_f32 v36, v34, v35, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fmac_f32_e32 v32, v15, v36
	v_add_f32_e32 v24, v32, v24
	v_mov_b32_e32 v3, s26
	buffer_load_b64 v[6:7], v3, s[20:23], null offen scope:SCOPE_DEV
	v_add_nc_u32_e32 v2, s26, v1
	buffer_load_b32 v17, v2, s[20:23], null offen offset:8 scope:SCOPE_DEV
	s_wait_loadcnt 0x0
	v_cmp_gt_u32_e32 vcc_lo, 16, v0
	v_cndmask_b32_e32 v34, v7, v6, vcc_lo
	v_bfe_u32 v35, v17, 4, 4
	v_cvt_f32_ubyte0_e32 v35, v35
	v_fma_mix_f32 v36, v34, v35, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_mul_f32_e32 v33, v9, v36
	v_bfe_u32 v35, v17, 0, 4
	v_cvt_f32_ubyte0_e32 v35, v35
	v_fma_mix_f32 v36, v34, v35, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fmac_f32_e32 v33, v8, v36
	v_bfe_u32 v35, v17, 8, 4
	v_cvt_f32_ubyte0_e32 v35, v35
	v_fma_mix_f32 v36, v34, v35, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fmac_f32_e32 v33, v10, v36
	v_bfe_u32 v35, v17, 12, 4
	v_cvt_f32_ubyte0_e32 v35, v35
	v_fma_mix_f32 v36, v34, v35, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fmac_f32_e32 v33, v11, v36
	v_bfe_u32 v35, v17, 16, 4
	v_cvt_f32_ubyte0_e32 v35, v35
	v_fma_mix_f32 v36, v34, v35, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fmac_f32_e32 v33, v12, v36
	v_bfe_u32 v35, v17, 20, 4
	v_cvt_f32_ubyte0_e32 v35, v35
	v_fma_mix_f32 v36, v34, v35, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fmac_f32_e32 v33, v13, v36
	v_bfe_u32 v35, v17, 24, 4
	v_cvt_f32_ubyte0_e32 v35, v35
	v_fma_mix_f32 v36, v34, v35, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fmac_f32_e32 v33, v14, v36
	v_bfe_u32 v35, v17, 28, 4
	v_cvt_f32_ubyte0_e32 v35, v35
	v_fma_mix_f32 v36, v34, v35, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fmac_f32_e32 v33, v15, v36
	v_add_f32_e32 v28, v28, v33
	s_add_co_i32 s15, s15, 1
	s_cmp_lt_u32 s15, s14
	s_cbranch_scc0 .Lres_fold
	s_mul_i32 s24, s15, 0x88
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s25, s16, s24
	s_add_co_i32 s26, s17, s24
	s_lshl_b32 s27, s15, 10
	v_lshlrev_b32_e32 v3, 5, v0
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v3, s27, v3
	global_load_b128 v[8:11], v3, s[6:7]
	global_load_b128 v[12:15], v3, s[6:7] offset:16
	s_wait_loadcnt 0x0
	v_mov_b32_e32 v3, s25
	buffer_load_b64 v[4:5], v3, s[20:23], null offen scope:SCOPE_DEV
	v_add_nc_u32_e32 v2, s25, v1
	buffer_load_b32 v16, v2, s[20:23], null offen offset:8 scope:SCOPE_DEV
	s_wait_loadcnt 0x0
	v_cmp_gt_u32_e32 vcc_lo, 16, v0
	v_cndmask_b32_e32 v34, v5, v4, vcc_lo
	v_bfe_u32 v35, v16, 0, 4
	v_cvt_f32_ubyte0_e32 v35, v35
	v_fma_mix_f32 v36, v34, v35, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_mul_f32_e32 v32, v8, v36
	v_bfe_u32 v35, v16, 4, 4
	v_cvt_f32_ubyte0_e32 v35, v35
	v_fma_mix_f32 v36, v34, v35, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fmac_f32_e32 v32, v9, v36
	v_bfe_u32 v35, v16, 8, 4
	v_cvt_f32_ubyte0_e32 v35, v35
	v_fma_mix_f32 v36, v34, v35, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fmac_f32_e32 v32, v10, v36
	v_bfe_u32 v35, v16, 12, 4
	v_cvt_f32_ubyte0_e32 v35, v35
	v_fma_mix_f32 v36, v34, v35, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fmac_f32_e32 v32, v11, v36
	v_bfe_u32 v35, v16, 16, 4
	v_cvt_f32_ubyte0_e32 v35, v35
	v_fma_mix_f32 v36, v34, v35, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fmac_f32_e32 v32, v12, v36
	v_bfe_u32 v35, v16, 20, 4
	v_cvt_f32_ubyte0_e32 v35, v35
	v_fma_mix_f32 v36, v34, v35, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fmac_f32_e32 v32, v13, v36
	v_bfe_u32 v35, v16, 24, 4
	v_cvt_f32_ubyte0_e32 v35, v35
	v_fma_mix_f32 v36, v34, v35, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fmac_f32_e32 v32, v14, v36
	v_bfe_u32 v35, v16, 28, 4
	v_cvt_f32_ubyte0_e32 v35, v35
	v_fma_mix_f32 v36, v34, v35, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fmac_f32_e32 v32, v15, v36
	v_add_f32_e32 v25, v32, v25
	v_mov_b32_e32 v3, s26
	buffer_load_b64 v[6:7], v3, s[20:23], null offen scope:SCOPE_DEV
	v_add_nc_u32_e32 v2, s26, v1
	buffer_load_b32 v17, v2, s[20:23], null offen offset:8 scope:SCOPE_DEV
	s_wait_loadcnt 0x0
	v_cmp_gt_u32_e32 vcc_lo, 16, v0
	v_cndmask_b32_e32 v34, v7, v6, vcc_lo
	v_bfe_u32 v35, v17, 4, 4
	v_cvt_f32_ubyte0_e32 v35, v35
	v_fma_mix_f32 v36, v34, v35, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_mul_f32_e32 v33, v9, v36
	v_bfe_u32 v35, v17, 0, 4
	v_cvt_f32_ubyte0_e32 v35, v35
	v_fma_mix_f32 v36, v34, v35, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fmac_f32_e32 v33, v8, v36
	v_bfe_u32 v35, v17, 8, 4
	v_cvt_f32_ubyte0_e32 v35, v35
	v_fma_mix_f32 v36, v34, v35, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fmac_f32_e32 v33, v10, v36
	v_bfe_u32 v35, v17, 12, 4
	v_cvt_f32_ubyte0_e32 v35, v35
	v_fma_mix_f32 v36, v34, v35, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fmac_f32_e32 v33, v11, v36
	v_bfe_u32 v35, v17, 16, 4
	v_cvt_f32_ubyte0_e32 v35, v35
	v_fma_mix_f32 v36, v34, v35, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fmac_f32_e32 v33, v12, v36
	v_bfe_u32 v35, v17, 20, 4
	v_cvt_f32_ubyte0_e32 v35, v35
	v_fma_mix_f32 v36, v34, v35, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fmac_f32_e32 v33, v13, v36
	v_bfe_u32 v35, v17, 24, 4
	v_cvt_f32_ubyte0_e32 v35, v35
	v_fma_mix_f32 v36, v34, v35, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fmac_f32_e32 v33, v14, v36
	v_bfe_u32 v35, v17, 28, 4
	v_cvt_f32_ubyte0_e32 v35, v35
	v_fma_mix_f32 v36, v34, v35, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fmac_f32_e32 v33, v15, v36
	v_add_f32_e32 v29, v29, v33
	s_add_co_i32 s15, s15, 1
	s_cmp_lt_u32 s15, s14
	s_cbranch_scc0 .Lres_fold
	s_mul_i32 s24, s15, 0x88
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s25, s16, s24
	s_add_co_i32 s26, s17, s24
	s_lshl_b32 s27, s15, 10
	v_lshlrev_b32_e32 v3, 5, v0
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v3, s27, v3
	global_load_b128 v[8:11], v3, s[6:7]
	global_load_b128 v[12:15], v3, s[6:7] offset:16
	s_wait_loadcnt 0x0
	v_mov_b32_e32 v3, s25
	buffer_load_b64 v[4:5], v3, s[20:23], null offen scope:SCOPE_DEV
	v_add_nc_u32_e32 v2, s25, v1
	buffer_load_b32 v16, v2, s[20:23], null offen offset:8 scope:SCOPE_DEV
	s_wait_loadcnt 0x0
	v_cmp_gt_u32_e32 vcc_lo, 16, v0
	v_cndmask_b32_e32 v34, v5, v4, vcc_lo
	v_bfe_u32 v35, v16, 0, 4
	v_cvt_f32_ubyte0_e32 v35, v35
	v_fma_mix_f32 v36, v34, v35, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_mul_f32_e32 v32, v8, v36
	v_bfe_u32 v35, v16, 4, 4
	v_cvt_f32_ubyte0_e32 v35, v35
	v_fma_mix_f32 v36, v34, v35, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fmac_f32_e32 v32, v9, v36
	v_bfe_u32 v35, v16, 8, 4
	v_cvt_f32_ubyte0_e32 v35, v35
	v_fma_mix_f32 v36, v34, v35, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fmac_f32_e32 v32, v10, v36
	v_bfe_u32 v35, v16, 12, 4
	v_cvt_f32_ubyte0_e32 v35, v35
	v_fma_mix_f32 v36, v34, v35, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fmac_f32_e32 v32, v11, v36
	v_bfe_u32 v35, v16, 16, 4
	v_cvt_f32_ubyte0_e32 v35, v35
	v_fma_mix_f32 v36, v34, v35, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fmac_f32_e32 v32, v12, v36
	v_bfe_u32 v35, v16, 20, 4
	v_cvt_f32_ubyte0_e32 v35, v35
	v_fma_mix_f32 v36, v34, v35, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fmac_f32_e32 v32, v13, v36
	v_bfe_u32 v35, v16, 24, 4
	v_cvt_f32_ubyte0_e32 v35, v35
	v_fma_mix_f32 v36, v34, v35, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fmac_f32_e32 v32, v14, v36
	v_bfe_u32 v35, v16, 28, 4
	v_cvt_f32_ubyte0_e32 v35, v35
	v_fma_mix_f32 v36, v34, v35, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fmac_f32_e32 v32, v15, v36
	v_add_f32_e32 v26, v32, v26
	v_mov_b32_e32 v3, s26
	buffer_load_b64 v[6:7], v3, s[20:23], null offen scope:SCOPE_DEV
	v_add_nc_u32_e32 v2, s26, v1
	buffer_load_b32 v17, v2, s[20:23], null offen offset:8 scope:SCOPE_DEV
	s_wait_loadcnt 0x0
	v_cmp_gt_u32_e32 vcc_lo, 16, v0
	v_cndmask_b32_e32 v34, v7, v6, vcc_lo
	v_bfe_u32 v35, v17, 4, 4
	v_cvt_f32_ubyte0_e32 v35, v35
	v_fma_mix_f32 v36, v34, v35, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_mul_f32_e32 v33, v9, v36
	v_bfe_u32 v35, v17, 0, 4
	v_cvt_f32_ubyte0_e32 v35, v35
	v_fma_mix_f32 v36, v34, v35, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fmac_f32_e32 v33, v8, v36
	v_bfe_u32 v35, v17, 8, 4
	v_cvt_f32_ubyte0_e32 v35, v35
	v_fma_mix_f32 v36, v34, v35, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fmac_f32_e32 v33, v10, v36
	v_bfe_u32 v35, v17, 12, 4
	v_cvt_f32_ubyte0_e32 v35, v35
	v_fma_mix_f32 v36, v34, v35, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fmac_f32_e32 v33, v11, v36
	v_bfe_u32 v35, v17, 16, 4
	v_cvt_f32_ubyte0_e32 v35, v35
	v_fma_mix_f32 v36, v34, v35, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fmac_f32_e32 v33, v12, v36
	v_bfe_u32 v35, v17, 20, 4
	v_cvt_f32_ubyte0_e32 v35, v35
	v_fma_mix_f32 v36, v34, v35, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fmac_f32_e32 v33, v13, v36
	v_bfe_u32 v35, v17, 24, 4
	v_cvt_f32_ubyte0_e32 v35, v35
	v_fma_mix_f32 v36, v34, v35, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fmac_f32_e32 v33, v14, v36
	v_bfe_u32 v35, v17, 28, 4
	v_cvt_f32_ubyte0_e32 v35, v35
	v_fma_mix_f32 v36, v34, v35, v34 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fmac_f32_e32 v33, v15, v36
	v_add_f32_e32 v30, v30, v33
	s_add_co_i32 s15, s15, 1
	.Lres_fold:
	v_add_f32_e32 v32, v24, v25
	v_add_f32_e32 v33, v26, v27
	v_add_f32_e32 v24, v32, v33
	ds_swizzle_b32 v39, v24 offset:swizzle(BITMASK_PERM,"1pppp")
	s_wait_dscnt 0x0
	v_add_f32_e32 v24, v24, v39
	v_cmp_gt_u32_e32 vcc_lo, 24, v0
	v_cndmask_b32_e64 v38, 0, 8, vcc_lo
	v_add_lshl_u32 v38, v38, v0, 2
	ds_bpermute_b32 v39, v38, v24
	s_wait_dscnt 0x0
	v_add_f32_e32 v24, v24, v39
	v_cmp_gt_u32_e32 vcc_lo, 28, v0
	v_cndmask_b32_e64 v38, 0, 4, vcc_lo
	v_add_lshl_u32 v38, v38, v0, 2
	ds_bpermute_b32 v39, v38, v24
	s_wait_dscnt 0x0
	v_add_f32_e32 v24, v24, v39
	v_cmp_gt_u32_e32 vcc_lo, 30, v0
	v_cndmask_b32_e64 v38, 0, 2, vcc_lo
	v_add_lshl_u32 v38, v38, v0, 2
	ds_bpermute_b32 v39, v38, v24
	s_wait_dscnt 0x0
	v_add_f32_e32 v24, v24, v39
	v_cmp_gt_u32_e32 vcc_lo, 31, v0
	v_cndmask_b32_e64 v38, 0, 1, vcc_lo
	v_add_lshl_u32 v38, v38, v0, 2
	ds_bpermute_b32 v39, v38, v24
	s_wait_dscnt 0x0
	v_add_f32_e32 v24, v24, v39
	v_add_f32_e32 v32, v28, v29
	v_add_f32_e32 v33, v30, v31
	v_add_f32_e32 v28, v32, v33
	ds_swizzle_b32 v39, v28 offset:swizzle(BITMASK_PERM,"1pppp")
	s_wait_dscnt 0x0
	v_add_f32_e32 v28, v28, v39
	v_cmp_gt_u32_e32 vcc_lo, 24, v0
	v_cndmask_b32_e64 v38, 0, 8, vcc_lo
	v_add_lshl_u32 v38, v38, v0, 2
	ds_bpermute_b32 v39, v38, v28
	s_wait_dscnt 0x0
	v_add_f32_e32 v28, v28, v39
	v_cmp_gt_u32_e32 vcc_lo, 28, v0
	v_cndmask_b32_e64 v38, 0, 4, vcc_lo
	v_add_lshl_u32 v38, v38, v0, 2
	ds_bpermute_b32 v39, v38, v28
	s_wait_dscnt 0x0
	v_add_f32_e32 v28, v28, v39
	v_cmp_gt_u32_e32 vcc_lo, 30, v0
	v_cndmask_b32_e64 v38, 0, 2, vcc_lo
	v_add_lshl_u32 v38, v38, v0, 2
	ds_bpermute_b32 v39, v38, v28
	s_wait_dscnt 0x0
	v_add_f32_e32 v28, v28, v39
	v_cmp_gt_u32_e32 vcc_lo, 31, v0
	v_cndmask_b32_e64 v38, 0, 1, vcc_lo
	v_add_lshl_u32 v38, v38, v0, 2
	ds_bpermute_b32 v39, v38, v28
	s_wait_dscnt 0x0
	v_add_f32_e32 v28, v28, v39
	v_cmpx_eq_u32_e32 0, v0
	s_cbranch_execz .Lres_end
	s_lshl_b32 s27, s12, 2
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v40, s27
	s_cmp_eq_u32 s18, 0
	s_cbranch_scc1 .Lres_single
	global_load_b64 v[41:42], v40, s[8:9]
	s_wait_loadcnt 0x0
	v_add_f32_e32 v41, v24, v41
	v_add_f32_e32 v42, v28, v42
	global_store_b64 v40, v[41:42], s[8:9]
	s_wait_storecnt 0x0
	s_branch .Lres_end
	.Lres_single:
	global_load_b32 v41, v40, s[8:9]
	s_wait_loadcnt 0x0
	v_add_f32_e32 v41, v41, v24
	global_store_b32 v40, v41, s[8:9]
	s_wait_storecnt 0x0
	.Lres_end:
	s_endpgm
.Lgemv_mq4g256v2_residual_end:
.size gemv_mq4g256v2_residual, .Lgemv_mq4g256v2_residual_end-gemv_mq4g256v2_residual
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel gemv_mq4g256v2_residual
	.amdhsa_group_segment_fixed_size 0
	.amdhsa_private_segment_fixed_size 0
	.amdhsa_kernarg_size 32
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
	.amdhsa_next_free_vgpr 108
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
	.amdhsa_inst_pref_size ((instprefsize(.Lgemv_mq4g256v2_residual_end-gemv_mq4g256v2_residual)<<4)&4080)>>4
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
.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .actual_access: read_only
        .address_space: global
        .name: A
        .offset: 0
        .size: 8
        .value_kind: global_buffer
      - .actual_access: read_only
        .address_space: global
        .name: x
        .offset: 8
        .size: 8
        .value_kind: global_buffer
      - .address_space: global
        .name: y
        .offset: 16
        .size: 8
        .value_kind: global_buffer
      - 
        .name: M
        .offset: 24
        .size: 4
        .value_kind: by_value
      - 
        .name: K
        .offset: 28
        .size: 4
        .value_kind: by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 32
    .max_flat_workgroup_size: 32
    .name: gemv_mq4g256v2_residual
    .private_segment_fixed_size: 0
    .sgpr_count: 30
    .sgpr_spill_count: 0
    .symbol: gemv_mq4g256v2_residual.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count: 108
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target: amdgcn-amd-amdhsa--gfx1201
amdhsa.version: [1, 2]
...
.end_amdgpu_metadata
.text
	.p2alignl 7, 3214868480
	.fill 96, 4, 3214868480
