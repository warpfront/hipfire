.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
.amdhsa_code_object_version 6
.text
.protected gemv_mq4g256v2_residual_xbatch
.globl gemv_mq4g256v2_residual_xbatch
.p2align 8
.type gemv_mq4g256v2_residual_xbatch,@function
gemv_mq4g256v2_residual_xbatch:
	s_load_b128 s[4:7], s[0:1], 0x0
	s_load_b128 s[8:11], s[0:1], 0x10
	s_load_b32 s3, s[0:1], 0x20
	s_wait_kmcnt 0x0
	s_lshr_b32 s14, s11, 8
	s_mul_i32 s19, s14, 0x88
	s_mov_b32 s20, s4
	s_and_b32 s21, s5, 0xffff
	s_mov_b32 s22, -1
	s_mov_b32 s23, 0x31004000
	s_lshl_b32 s44, s11, 2
	s_lshl_b32 s45, s10, 2
	s_mov_b32 s28, s6
	s_mov_b32 s29, s7
	s_mul_i32 s24, s44, 1
	s_add_co_u32 s30, s6, s24
	s_add_co_ci_u32 s31, s7, 0
	s_mul_i32 s24, s44, 2
	s_add_co_u32 s32, s6, s24
	s_add_co_ci_u32 s33, s7, 0
	s_mul_i32 s24, s44, 3
	s_add_co_u32 s34, s6, s24
	s_add_co_ci_u32 s35, s7, 0
	s_mul_i32 s24, s44, 4
	s_add_co_u32 s36, s6, s24
	s_add_co_ci_u32 s37, s7, 0
	s_mul_i32 s24, s44, 5
	s_add_co_u32 s38, s6, s24
	s_add_co_ci_u32 s39, s7, 0
	s_mul_i32 s24, s44, 6
	s_add_co_u32 s40, s6, s24
	s_add_co_ci_u32 s41, s7, 0
	s_mul_i32 s24, s44, 7
	s_add_co_u32 s42, s6, s24
	s_add_co_ci_u32 s43, s7, 0
	v_lshlrev_b32_e32 v1, 2, v0
	s_cmp_eq_u32 s3, 1
	s_cbranch_scc1 .Lrx_b1
	s_cmp_eq_u32 s3, 2
	s_cbranch_scc1 .Lrx_b2
	s_cmp_eq_u32 s3, 3
	s_cbranch_scc1 .Lrx_b3
	s_cmp_eq_u32 s3, 4
	s_cbranch_scc1 .Lrx_b4
	s_cmp_eq_u32 s3, 5
	s_cbranch_scc1 .Lrx_b5
	s_cmp_eq_u32 s3, 6
	s_cbranch_scc1 .Lrx_b6
	s_cmp_eq_u32 s3, 7
	s_cbranch_scc1 .Lrx_b7
	s_cmp_eq_u32 s3, 8
	s_cbranch_scc1 .Lrx_b8
	s_branch .Lrx_end
	.Lrx_b1:
	s_mul_i32 s12, ttmp9, 4
	s_cmp_ge_i32 s12, s10
	s_cbranch_scc1 .Lrx_end
	s_mov_b32 s13, s12
	s_mul_i32 s13, s13, s19
	v_mov_b32_e32 v8, s13
	v_add_nc_u32_e32 v4, s13, v1
	s_add_co_i32 s13, s12, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cselect_b32 s13, s13, s12
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s13, s13, s19
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v9, s13
	v_add_nc_u32_e32 v5, s13, v1
	s_add_co_i32 s13, s12, 2
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cselect_b32 s13, s13, s12
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s13, s13, s19
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v10, s13
	v_add_nc_u32_e32 v6, s13, v1
	s_add_co_i32 s13, s12, 3
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cselect_b32 s13, s13, s12
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s13, s13, s19
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v11, s13
	v_add_nc_u32_e32 v7, s13, v1
	v_mov_b32_e32 v64, 0
	v_mov_b32_e32 v65, 0
	v_mov_b32_e32 v66, 0
	v_mov_b32_e32 v67, 0
	v_mov_b32_e32 v68, 0
	v_mov_b32_e32 v69, 0
	v_mov_b32_e32 v70, 0
	v_mov_b32_e32 v71, 0
	v_mov_b32_e32 v72, 0
	v_mov_b32_e32 v73, 0
	v_mov_b32_e32 v74, 0
	v_mov_b32_e32 v75, 0
	s_add_co_i32 s2, s14, 3
	s_lshr_b32 s2, s2, 2
	s_cmp_eq_u32 s2, 0
	s_cbranch_scc1 .Lrx_b1_s0_done
	s_mov_b32 s15, 0
	v_lshlrev_b32_e32 v2, 5, v0
	buffer_load_b64 v[12:13], v8, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v20, v4, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[14:15], v9, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v21, v5, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[16:17], v10, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v22, v6, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[18:19], v11, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v23, v7, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	global_load_b128 v[77:80], v2, s[28:29]
	global_load_b128 v[81:84], v2, s[28:29] offset:16
	.Lrx_b1_s0:
	s_cmp_gt_u32 s2, 1
	s_cselect_b32 s17, 0x220, 0
	s_cselect_b32 s18, 0x1000, 0
	s_add_co_i32 s16, s15, s17
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v3, s18, v2
	v_cmp_gt_u32_e32 vcc_lo, 16, v0
	s_wait_loadcnt 0x9
	v_cndmask_b32_e32 v12, v13, v12, vcc_lo
	s_wait_loadcnt 0x8
	v_and_b32_e32 v30, 0xf0f0f0f, v20
	v_lshrrev_b32_e32 v31, 4, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v31, 0xf0f0f0f, v31
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v24, v30
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v26, v30
	v_cvt_f32_ubyte2_e32 v28, v30
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v25, v31
	v_cvt_f32_ubyte1_e32 v27, v31
	v_cvt_f32_ubyte2_e32 v29, v31
	v_cvt_f32_ubyte3_e32 v30, v30
	v_cvt_f32_ubyte3_e32 v31, v31
	v_fma_mix_f32 v24, v12, v24, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v25, v12, v25, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v26, v12, v26, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v27, v12, v27, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v28, v12, v28, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v29, v12, v29, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v30, v12, v30, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v31, v12, v31, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x7
	v_cndmask_b32_e32 v14, v15, v14, vcc_lo
	s_wait_loadcnt 0x6
	v_and_b32_e32 v40, 0xf0f0f0f, v21
	v_lshrrev_b32_e32 v41, 4, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v41, 0xf0f0f0f, v41
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v34, v40
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v36, v40
	v_cvt_f32_ubyte2_e32 v38, v40
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v35, v41
	v_cvt_f32_ubyte1_e32 v37, v41
	v_cvt_f32_ubyte2_e32 v39, v41
	v_cvt_f32_ubyte3_e32 v40, v40
	v_cvt_f32_ubyte3_e32 v41, v41
	v_fma_mix_f32 v34, v14, v34, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v35, v14, v35, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v36, v14, v36, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v37, v14, v37, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v38, v14, v38, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v39, v14, v39, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v40, v14, v40, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v41, v14, v41, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x5
	v_cndmask_b32_e32 v16, v17, v16, vcc_lo
	s_wait_loadcnt 0x4
	v_and_b32_e32 v50, 0xf0f0f0f, v22
	v_lshrrev_b32_e32 v51, 4, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v51, 0xf0f0f0f, v51
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v44, v50
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v46, v50
	v_cvt_f32_ubyte2_e32 v48, v50
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v45, v51
	v_cvt_f32_ubyte1_e32 v47, v51
	v_cvt_f32_ubyte2_e32 v49, v51
	v_cvt_f32_ubyte3_e32 v50, v50
	v_cvt_f32_ubyte3_e32 v51, v51
	v_fma_mix_f32 v44, v16, v44, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v16, v45, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v16, v46, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v16, v47, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v48, v16, v48, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v49, v16, v49, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v16, v50, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v16, v51, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x3
	v_cndmask_b32_e32 v18, v19, v18, vcc_lo
	s_wait_loadcnt 0x2
	v_and_b32_e32 v60, 0xf0f0f0f, v23
	v_lshrrev_b32_e32 v61, 4, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v61, 0xf0f0f0f, v61
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v54, v60
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v56, v60
	v_cvt_f32_ubyte2_e32 v58, v60
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v55, v61
	v_cvt_f32_ubyte1_e32 v57, v61
	v_cvt_f32_ubyte2_e32 v59, v61
	v_cvt_f32_ubyte3_e32 v60, v60
	v_cvt_f32_ubyte3_e32 v61, v61
	v_fma_mix_f32 v54, v18, v54, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v18, v55, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v18, v56, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v18, v57, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v58, v18, v58, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v59, v18, v59, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v18, v60, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v18, v61, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	buffer_load_b64 v[12:13], v8, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v20, v4, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[14:15], v9, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v21, v5, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[16:17], v10, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v22, v6, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[18:19], v11, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v23, v7, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	s_wait_loadcnt 0x9
	v_dual_mul_f32 v32, v24, v77 :: v_dual_mul_f32 v33, v35, v78
	v_dual_mul_f32 v42, v44, v77 :: v_dual_mul_f32 v43, v55, v78
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v25, v78 :: v_dual_fmac_f32 v33, v34, v77
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v45, v78 :: v_dual_fmac_f32 v43, v54, v77
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v26, v79 :: v_dual_fmac_f32 v33, v36, v79
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v46, v79 :: v_dual_fmac_f32 v43, v56, v79
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v27, v80 :: v_dual_fmac_f32 v33, v37, v80
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v47, v80 :: v_dual_fmac_f32 v43, v57, v80
	s_wait_loadcnt 0x8
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v28, v81 :: v_dual_fmac_f32 v33, v38, v81
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v48, v81 :: v_dual_fmac_f32 v43, v58, v81
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v29, v82 :: v_dual_fmac_f32 v33, v39, v82
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v49, v82 :: v_dual_fmac_f32 v43, v59, v82
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v30, v83 :: v_dual_fmac_f32 v33, v40, v83
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v50, v83 :: v_dual_fmac_f32 v43, v60, v83
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v31, v84 :: v_dual_fmac_f32 v33, v41, v84
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v51, v84 :: v_dual_fmac_f32 v43, v61, v84
	global_load_b128 v[77:80], v3, s[28:29]
	global_load_b128 v[81:84], v3, s[28:29] offset:16
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_add_f32 v64, v32, v64 :: v_dual_add_f32 v65, v65, v33
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_add_f32 v66, v42, v66 :: v_dual_add_f32 v67, v67, v43
	s_mov_b32 s15, s16
	v_mov_b32_e32 v2, v3
	s_add_co_i32 s2, s2, -1
	s_cmp_lg_u32 s2, 0
	s_cbranch_scc1 .Lrx_b1_s0
	s_wait_loadcnt 0x0
	.Lrx_b1_s0_done:
	s_add_co_i32 s2, s14, 2
	s_lshr_b32 s2, s2, 2
	s_cmp_eq_u32 s2, 0
	s_cbranch_scc1 .Lrx_b1_s1_done
	s_mov_b32 s15, 0x88
	v_lshlrev_b32_e32 v2, 5, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v2, 0x400, v2
	buffer_load_b64 v[12:13], v8, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v20, v4, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[14:15], v9, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v21, v5, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[16:17], v10, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v22, v6, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[18:19], v11, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v23, v7, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	global_load_b128 v[77:80], v2, s[28:29]
	global_load_b128 v[81:84], v2, s[28:29] offset:16
	.Lrx_b1_s1:
	s_cmp_gt_u32 s2, 1
	s_cselect_b32 s17, 0x220, 0
	s_cselect_b32 s18, 0x1000, 0
	s_add_co_i32 s16, s15, s17
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v3, s18, v2
	v_cmp_gt_u32_e32 vcc_lo, 16, v0
	s_wait_loadcnt 0x9
	v_cndmask_b32_e32 v12, v13, v12, vcc_lo
	s_wait_loadcnt 0x8
	v_and_b32_e32 v30, 0xf0f0f0f, v20
	v_lshrrev_b32_e32 v31, 4, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v31, 0xf0f0f0f, v31
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v24, v30
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v26, v30
	v_cvt_f32_ubyte2_e32 v28, v30
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v25, v31
	v_cvt_f32_ubyte1_e32 v27, v31
	v_cvt_f32_ubyte2_e32 v29, v31
	v_cvt_f32_ubyte3_e32 v30, v30
	v_cvt_f32_ubyte3_e32 v31, v31
	v_fma_mix_f32 v24, v12, v24, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v25, v12, v25, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v26, v12, v26, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v27, v12, v27, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v28, v12, v28, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v29, v12, v29, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v30, v12, v30, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v31, v12, v31, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x7
	v_cndmask_b32_e32 v14, v15, v14, vcc_lo
	s_wait_loadcnt 0x6
	v_and_b32_e32 v40, 0xf0f0f0f, v21
	v_lshrrev_b32_e32 v41, 4, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v41, 0xf0f0f0f, v41
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v34, v40
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v36, v40
	v_cvt_f32_ubyte2_e32 v38, v40
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v35, v41
	v_cvt_f32_ubyte1_e32 v37, v41
	v_cvt_f32_ubyte2_e32 v39, v41
	v_cvt_f32_ubyte3_e32 v40, v40
	v_cvt_f32_ubyte3_e32 v41, v41
	v_fma_mix_f32 v34, v14, v34, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v35, v14, v35, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v36, v14, v36, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v37, v14, v37, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v38, v14, v38, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v39, v14, v39, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v40, v14, v40, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v41, v14, v41, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x5
	v_cndmask_b32_e32 v16, v17, v16, vcc_lo
	s_wait_loadcnt 0x4
	v_and_b32_e32 v50, 0xf0f0f0f, v22
	v_lshrrev_b32_e32 v51, 4, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v51, 0xf0f0f0f, v51
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v44, v50
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v46, v50
	v_cvt_f32_ubyte2_e32 v48, v50
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v45, v51
	v_cvt_f32_ubyte1_e32 v47, v51
	v_cvt_f32_ubyte2_e32 v49, v51
	v_cvt_f32_ubyte3_e32 v50, v50
	v_cvt_f32_ubyte3_e32 v51, v51
	v_fma_mix_f32 v44, v16, v44, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v16, v45, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v16, v46, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v16, v47, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v48, v16, v48, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v49, v16, v49, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v16, v50, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v16, v51, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x3
	v_cndmask_b32_e32 v18, v19, v18, vcc_lo
	s_wait_loadcnt 0x2
	v_and_b32_e32 v60, 0xf0f0f0f, v23
	v_lshrrev_b32_e32 v61, 4, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v61, 0xf0f0f0f, v61
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v54, v60
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v56, v60
	v_cvt_f32_ubyte2_e32 v58, v60
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v55, v61
	v_cvt_f32_ubyte1_e32 v57, v61
	v_cvt_f32_ubyte2_e32 v59, v61
	v_cvt_f32_ubyte3_e32 v60, v60
	v_cvt_f32_ubyte3_e32 v61, v61
	v_fma_mix_f32 v54, v18, v54, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v18, v55, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v18, v56, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v18, v57, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v58, v18, v58, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v59, v18, v59, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v18, v60, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v18, v61, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	buffer_load_b64 v[12:13], v8, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v20, v4, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[14:15], v9, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v21, v5, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[16:17], v10, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v22, v6, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[18:19], v11, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v23, v7, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	s_wait_loadcnt 0x9
	v_dual_mul_f32 v32, v24, v77 :: v_dual_mul_f32 v33, v35, v78
	v_dual_mul_f32 v42, v44, v77 :: v_dual_mul_f32 v43, v55, v78
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v25, v78 :: v_dual_fmac_f32 v33, v34, v77
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v45, v78 :: v_dual_fmac_f32 v43, v54, v77
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v26, v79 :: v_dual_fmac_f32 v33, v36, v79
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v46, v79 :: v_dual_fmac_f32 v43, v56, v79
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v27, v80 :: v_dual_fmac_f32 v33, v37, v80
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v47, v80 :: v_dual_fmac_f32 v43, v57, v80
	s_wait_loadcnt 0x8
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v28, v81 :: v_dual_fmac_f32 v33, v38, v81
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v48, v81 :: v_dual_fmac_f32 v43, v58, v81
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v29, v82 :: v_dual_fmac_f32 v33, v39, v82
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v49, v82 :: v_dual_fmac_f32 v43, v59, v82
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v30, v83 :: v_dual_fmac_f32 v33, v40, v83
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v50, v83 :: v_dual_fmac_f32 v43, v60, v83
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v31, v84 :: v_dual_fmac_f32 v33, v41, v84
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v51, v84 :: v_dual_fmac_f32 v43, v61, v84
	global_load_b128 v[77:80], v3, s[28:29]
	global_load_b128 v[81:84], v3, s[28:29] offset:16
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_add_f32 v68, v32, v68 :: v_dual_add_f32 v69, v69, v33
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_add_f32 v70, v42, v70 :: v_dual_add_f32 v71, v71, v43
	s_mov_b32 s15, s16
	v_mov_b32_e32 v2, v3
	s_add_co_i32 s2, s2, -1
	s_cmp_lg_u32 s2, 0
	s_cbranch_scc1 .Lrx_b1_s1
	s_wait_loadcnt 0x0
	.Lrx_b1_s1_done:
	v_dual_add_f32 v64, v64, v68 :: v_dual_add_f32 v65, v65, v69
	v_dual_add_f32 v66, v66, v70 :: v_dual_add_f32 v67, v67, v71
	v_mov_b32_e32 v68, 0
	v_mov_b32_e32 v69, 0
	v_mov_b32_e32 v70, 0
	v_mov_b32_e32 v71, 0
	s_add_co_i32 s2, s14, 1
	s_lshr_b32 s2, s2, 2
	s_cmp_eq_u32 s2, 0
	s_cbranch_scc1 .Lrx_b1_s2_done
	s_mov_b32 s15, 0x110
	v_lshlrev_b32_e32 v2, 5, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v2, 0x800, v2
	buffer_load_b64 v[12:13], v8, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v20, v4, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[14:15], v9, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v21, v5, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[16:17], v10, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v22, v6, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[18:19], v11, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v23, v7, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	global_load_b128 v[77:80], v2, s[28:29]
	global_load_b128 v[81:84], v2, s[28:29] offset:16
	.Lrx_b1_s2:
	s_cmp_gt_u32 s2, 1
	s_cselect_b32 s17, 0x220, 0
	s_cselect_b32 s18, 0x1000, 0
	s_add_co_i32 s16, s15, s17
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v3, s18, v2
	v_cmp_gt_u32_e32 vcc_lo, 16, v0
	s_wait_loadcnt 0x9
	v_cndmask_b32_e32 v12, v13, v12, vcc_lo
	s_wait_loadcnt 0x8
	v_and_b32_e32 v30, 0xf0f0f0f, v20
	v_lshrrev_b32_e32 v31, 4, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v31, 0xf0f0f0f, v31
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v24, v30
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v26, v30
	v_cvt_f32_ubyte2_e32 v28, v30
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v25, v31
	v_cvt_f32_ubyte1_e32 v27, v31
	v_cvt_f32_ubyte2_e32 v29, v31
	v_cvt_f32_ubyte3_e32 v30, v30
	v_cvt_f32_ubyte3_e32 v31, v31
	v_fma_mix_f32 v24, v12, v24, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v25, v12, v25, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v26, v12, v26, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v27, v12, v27, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v28, v12, v28, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v29, v12, v29, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v30, v12, v30, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v31, v12, v31, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x7
	v_cndmask_b32_e32 v14, v15, v14, vcc_lo
	s_wait_loadcnt 0x6
	v_and_b32_e32 v40, 0xf0f0f0f, v21
	v_lshrrev_b32_e32 v41, 4, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v41, 0xf0f0f0f, v41
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v34, v40
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v36, v40
	v_cvt_f32_ubyte2_e32 v38, v40
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v35, v41
	v_cvt_f32_ubyte1_e32 v37, v41
	v_cvt_f32_ubyte2_e32 v39, v41
	v_cvt_f32_ubyte3_e32 v40, v40
	v_cvt_f32_ubyte3_e32 v41, v41
	v_fma_mix_f32 v34, v14, v34, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v35, v14, v35, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v36, v14, v36, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v37, v14, v37, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v38, v14, v38, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v39, v14, v39, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v40, v14, v40, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v41, v14, v41, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x5
	v_cndmask_b32_e32 v16, v17, v16, vcc_lo
	s_wait_loadcnt 0x4
	v_and_b32_e32 v50, 0xf0f0f0f, v22
	v_lshrrev_b32_e32 v51, 4, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v51, 0xf0f0f0f, v51
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v44, v50
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v46, v50
	v_cvt_f32_ubyte2_e32 v48, v50
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v45, v51
	v_cvt_f32_ubyte1_e32 v47, v51
	v_cvt_f32_ubyte2_e32 v49, v51
	v_cvt_f32_ubyte3_e32 v50, v50
	v_cvt_f32_ubyte3_e32 v51, v51
	v_fma_mix_f32 v44, v16, v44, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v16, v45, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v16, v46, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v16, v47, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v48, v16, v48, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v49, v16, v49, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v16, v50, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v16, v51, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x3
	v_cndmask_b32_e32 v18, v19, v18, vcc_lo
	s_wait_loadcnt 0x2
	v_and_b32_e32 v60, 0xf0f0f0f, v23
	v_lshrrev_b32_e32 v61, 4, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v61, 0xf0f0f0f, v61
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v54, v60
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v56, v60
	v_cvt_f32_ubyte2_e32 v58, v60
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v55, v61
	v_cvt_f32_ubyte1_e32 v57, v61
	v_cvt_f32_ubyte2_e32 v59, v61
	v_cvt_f32_ubyte3_e32 v60, v60
	v_cvt_f32_ubyte3_e32 v61, v61
	v_fma_mix_f32 v54, v18, v54, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v18, v55, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v18, v56, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v18, v57, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v58, v18, v58, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v59, v18, v59, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v18, v60, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v18, v61, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	buffer_load_b64 v[12:13], v8, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v20, v4, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[14:15], v9, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v21, v5, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[16:17], v10, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v22, v6, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[18:19], v11, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v23, v7, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	s_wait_loadcnt 0x9
	v_dual_mul_f32 v32, v24, v77 :: v_dual_mul_f32 v33, v35, v78
	v_dual_mul_f32 v42, v44, v77 :: v_dual_mul_f32 v43, v55, v78
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v25, v78 :: v_dual_fmac_f32 v33, v34, v77
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v45, v78 :: v_dual_fmac_f32 v43, v54, v77
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v26, v79 :: v_dual_fmac_f32 v33, v36, v79
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v46, v79 :: v_dual_fmac_f32 v43, v56, v79
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v27, v80 :: v_dual_fmac_f32 v33, v37, v80
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v47, v80 :: v_dual_fmac_f32 v43, v57, v80
	s_wait_loadcnt 0x8
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v28, v81 :: v_dual_fmac_f32 v33, v38, v81
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v48, v81 :: v_dual_fmac_f32 v43, v58, v81
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v29, v82 :: v_dual_fmac_f32 v33, v39, v82
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v49, v82 :: v_dual_fmac_f32 v43, v59, v82
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v30, v83 :: v_dual_fmac_f32 v33, v40, v83
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v50, v83 :: v_dual_fmac_f32 v43, v60, v83
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v31, v84 :: v_dual_fmac_f32 v33, v41, v84
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v51, v84 :: v_dual_fmac_f32 v43, v61, v84
	global_load_b128 v[77:80], v3, s[28:29]
	global_load_b128 v[81:84], v3, s[28:29] offset:16
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_add_f32 v68, v32, v68 :: v_dual_add_f32 v69, v69, v33
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_add_f32 v70, v42, v70 :: v_dual_add_f32 v71, v71, v43
	s_mov_b32 s15, s16
	v_mov_b32_e32 v2, v3
	s_add_co_i32 s2, s2, -1
	s_cmp_lg_u32 s2, 0
	s_cbranch_scc1 .Lrx_b1_s2
	s_wait_loadcnt 0x0
	.Lrx_b1_s2_done:
	s_add_co_i32 s2, s14, 0
	s_lshr_b32 s2, s2, 2
	s_cmp_eq_u32 s2, 0
	s_cbranch_scc1 .Lrx_b1_s3_done
	s_mov_b32 s15, 0x198
	v_lshlrev_b32_e32 v2, 5, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v2, 0xc00, v2
	buffer_load_b64 v[12:13], v8, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v20, v4, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[14:15], v9, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v21, v5, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[16:17], v10, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v22, v6, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[18:19], v11, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v23, v7, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	global_load_b128 v[77:80], v2, s[28:29]
	global_load_b128 v[81:84], v2, s[28:29] offset:16
	.Lrx_b1_s3:
	s_cmp_gt_u32 s2, 1
	s_cselect_b32 s17, 0x220, 0
	s_cselect_b32 s18, 0x1000, 0
	s_add_co_i32 s16, s15, s17
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v3, s18, v2
	v_cmp_gt_u32_e32 vcc_lo, 16, v0
	s_wait_loadcnt 0x9
	v_cndmask_b32_e32 v12, v13, v12, vcc_lo
	s_wait_loadcnt 0x8
	v_and_b32_e32 v30, 0xf0f0f0f, v20
	v_lshrrev_b32_e32 v31, 4, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v31, 0xf0f0f0f, v31
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v24, v30
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v26, v30
	v_cvt_f32_ubyte2_e32 v28, v30
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v25, v31
	v_cvt_f32_ubyte1_e32 v27, v31
	v_cvt_f32_ubyte2_e32 v29, v31
	v_cvt_f32_ubyte3_e32 v30, v30
	v_cvt_f32_ubyte3_e32 v31, v31
	v_fma_mix_f32 v24, v12, v24, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v25, v12, v25, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v26, v12, v26, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v27, v12, v27, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v28, v12, v28, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v29, v12, v29, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v30, v12, v30, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v31, v12, v31, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x7
	v_cndmask_b32_e32 v14, v15, v14, vcc_lo
	s_wait_loadcnt 0x6
	v_and_b32_e32 v40, 0xf0f0f0f, v21
	v_lshrrev_b32_e32 v41, 4, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v41, 0xf0f0f0f, v41
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v34, v40
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v36, v40
	v_cvt_f32_ubyte2_e32 v38, v40
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v35, v41
	v_cvt_f32_ubyte1_e32 v37, v41
	v_cvt_f32_ubyte2_e32 v39, v41
	v_cvt_f32_ubyte3_e32 v40, v40
	v_cvt_f32_ubyte3_e32 v41, v41
	v_fma_mix_f32 v34, v14, v34, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v35, v14, v35, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v36, v14, v36, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v37, v14, v37, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v38, v14, v38, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v39, v14, v39, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v40, v14, v40, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v41, v14, v41, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x5
	v_cndmask_b32_e32 v16, v17, v16, vcc_lo
	s_wait_loadcnt 0x4
	v_and_b32_e32 v50, 0xf0f0f0f, v22
	v_lshrrev_b32_e32 v51, 4, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v51, 0xf0f0f0f, v51
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v44, v50
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v46, v50
	v_cvt_f32_ubyte2_e32 v48, v50
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v45, v51
	v_cvt_f32_ubyte1_e32 v47, v51
	v_cvt_f32_ubyte2_e32 v49, v51
	v_cvt_f32_ubyte3_e32 v50, v50
	v_cvt_f32_ubyte3_e32 v51, v51
	v_fma_mix_f32 v44, v16, v44, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v16, v45, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v16, v46, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v16, v47, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v48, v16, v48, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v49, v16, v49, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v16, v50, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v16, v51, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x3
	v_cndmask_b32_e32 v18, v19, v18, vcc_lo
	s_wait_loadcnt 0x2
	v_and_b32_e32 v60, 0xf0f0f0f, v23
	v_lshrrev_b32_e32 v61, 4, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v61, 0xf0f0f0f, v61
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v54, v60
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v56, v60
	v_cvt_f32_ubyte2_e32 v58, v60
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v55, v61
	v_cvt_f32_ubyte1_e32 v57, v61
	v_cvt_f32_ubyte2_e32 v59, v61
	v_cvt_f32_ubyte3_e32 v60, v60
	v_cvt_f32_ubyte3_e32 v61, v61
	v_fma_mix_f32 v54, v18, v54, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v18, v55, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v18, v56, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v18, v57, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v58, v18, v58, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v59, v18, v59, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v18, v60, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v18, v61, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	buffer_load_b64 v[12:13], v8, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v20, v4, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[14:15], v9, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v21, v5, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[16:17], v10, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v22, v6, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[18:19], v11, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v23, v7, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	s_wait_loadcnt 0x9
	v_dual_mul_f32 v32, v24, v77 :: v_dual_mul_f32 v33, v35, v78
	v_dual_mul_f32 v42, v44, v77 :: v_dual_mul_f32 v43, v55, v78
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v25, v78 :: v_dual_fmac_f32 v33, v34, v77
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v45, v78 :: v_dual_fmac_f32 v43, v54, v77
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v26, v79 :: v_dual_fmac_f32 v33, v36, v79
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v46, v79 :: v_dual_fmac_f32 v43, v56, v79
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v27, v80 :: v_dual_fmac_f32 v33, v37, v80
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v47, v80 :: v_dual_fmac_f32 v43, v57, v80
	s_wait_loadcnt 0x8
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v28, v81 :: v_dual_fmac_f32 v33, v38, v81
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v48, v81 :: v_dual_fmac_f32 v43, v58, v81
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v29, v82 :: v_dual_fmac_f32 v33, v39, v82
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v49, v82 :: v_dual_fmac_f32 v43, v59, v82
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v30, v83 :: v_dual_fmac_f32 v33, v40, v83
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v50, v83 :: v_dual_fmac_f32 v43, v60, v83
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v31, v84 :: v_dual_fmac_f32 v33, v41, v84
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v51, v84 :: v_dual_fmac_f32 v43, v61, v84
	global_load_b128 v[77:80], v3, s[28:29]
	global_load_b128 v[81:84], v3, s[28:29] offset:16
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_add_f32 v72, v32, v72 :: v_dual_add_f32 v73, v73, v33
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_add_f32 v74, v42, v74 :: v_dual_add_f32 v75, v75, v43
	s_mov_b32 s15, s16
	v_mov_b32_e32 v2, v3
	s_add_co_i32 s2, s2, -1
	s_cmp_lg_u32 s2, 0
	s_cbranch_scc1 .Lrx_b1_s3
	s_wait_loadcnt 0x0
	.Lrx_b1_s3_done:
	v_dual_add_f32 v68, v68, v72 :: v_dual_add_f32 v69, v69, v73
	v_dual_add_f32 v70, v70, v74 :: v_dual_add_f32 v71, v71, v75
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_add_f32 v64, v64, v68 :: v_dual_add_f32 v65, v65, v69
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_add_f32 v66, v66, v70 :: v_dual_add_f32 v67, v67, v71
	ds_swizzle_b32 v24, v64 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v25, v65 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v26, v66 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v27, v67 offset:swizzle(BITMASK_PERM,"1pppp")
	s_wait_dscnt 0x3
	s_delay_alu instid0(VALU_DEP_2)
	v_add_f32_e32 v64, v64, v24
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_3)
	v_add_f32_e32 v65, v65, v25
	s_wait_dscnt 0x1
	s_delay_alu instid0(VALU_DEP_3)
	v_add_f32_e32 v66, v66, v26
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v67, v67, v27
	v_cmp_gt_u32_e32 vcc_lo, 24, v0
	v_cndmask_b32_e64 v3, 0, 8, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v3, v3, v0, 2
	ds_bpermute_b32 v24, v3, v64
	ds_bpermute_b32 v25, v3, v65
	ds_bpermute_b32 v26, v3, v66
	ds_bpermute_b32 v27, v3, v67
	s_wait_dscnt 0x3
	v_add_f32_e32 v64, v64, v24
	s_wait_dscnt 0x2
	v_add_f32_e32 v65, v65, v25
	s_wait_dscnt 0x1
	v_add_f32_e32 v66, v66, v26
	s_wait_dscnt 0x0
	v_add_f32_e32 v67, v67, v27
	v_cmp_gt_u32_e32 vcc_lo, 28, v0
	v_cndmask_b32_e64 v3, 0, 4, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v3, v3, v0, 2
	ds_bpermute_b32 v24, v3, v64
	ds_bpermute_b32 v25, v3, v65
	ds_bpermute_b32 v26, v3, v66
	ds_bpermute_b32 v27, v3, v67
	s_wait_dscnt 0x3
	v_add_f32_e32 v64, v64, v24
	s_wait_dscnt 0x2
	v_add_f32_e32 v65, v65, v25
	s_wait_dscnt 0x1
	v_add_f32_e32 v66, v66, v26
	s_wait_dscnt 0x0
	v_add_f32_e32 v67, v67, v27
	v_cmp_gt_u32_e32 vcc_lo, 30, v0
	v_cndmask_b32_e64 v3, 0, 2, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v3, v3, v0, 2
	ds_bpermute_b32 v24, v3, v64
	ds_bpermute_b32 v25, v3, v65
	ds_bpermute_b32 v26, v3, v66
	ds_bpermute_b32 v27, v3, v67
	s_wait_dscnt 0x3
	v_add_f32_e32 v64, v64, v24
	s_wait_dscnt 0x2
	v_add_f32_e32 v65, v65, v25
	s_wait_dscnt 0x1
	v_add_f32_e32 v66, v66, v26
	s_wait_dscnt 0x0
	v_add_f32_e32 v67, v67, v27
	v_cmp_gt_u32_e32 vcc_lo, 31, v0
	v_cndmask_b32_e64 v3, 0, 1, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v3, v3, v0, 2
	ds_bpermute_b32 v24, v3, v64
	ds_bpermute_b32 v25, v3, v65
	ds_bpermute_b32 v26, v3, v66
	ds_bpermute_b32 v27, v3, v67
	s_wait_dscnt 0x3
	v_add_f32_e32 v64, v64, v24
	s_wait_dscnt 0x2
	v_add_f32_e32 v65, v65, v25
	s_wait_dscnt 0x1
	v_add_f32_e32 v66, v66, v26
	s_wait_dscnt 0x0
	v_add_f32_e32 v67, v67, v27
	v_cmpx_eq_u32_e32 0, v0
	s_lshl_b32 s46, s12, 2
	v_mov_b32_e32 v32, s46
	s_add_co_i32 s13, s12, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cbranch_scc0 .Lrx_b1_p0_single
	global_load_b32 v28, v32, s[8:9]
	global_load_b32 v29, v32, s[8:9] offset:4
	s_wait_loadcnt 0x1
	v_add_f32_e32 v28, v64, v28
	global_store_b32 v32, v28, s[8:9]
	s_wait_loadcnt 0x0
	v_add_f32_e32 v29, v65, v29
	global_store_b32 v32, v29, s[8:9] offset:4
	s_wait_storecnt 0x0
	s_branch .Lrx_b1_p0_next
	.Lrx_b1_p0_single:
	global_load_b32 v28, v32, s[8:9]
	s_wait_loadcnt 0x0
	v_add_f32_e32 v28, v28, v64
	global_store_b32 v32, v28, s[8:9]
	s_wait_storecnt 0x0
	s_branch .Lrx_b1_stored
	.Lrx_b1_p0_next:
	s_add_co_i32 s13, s12, 2
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cbranch_scc0 .Lrx_b1_stored
	s_add_co_i32 s13, s12, 3
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cbranch_scc0 .Lrx_b1_p1_single
	global_load_b32 v30, v32, s[8:9] offset:8
	global_load_b32 v31, v32, s[8:9] offset:12
	s_wait_loadcnt 0x1
	v_add_f32_e32 v30, v66, v30
	global_store_b32 v32, v30, s[8:9] offset:8
	s_wait_loadcnt 0x0
	v_add_f32_e32 v31, v67, v31
	global_store_b32 v32, v31, s[8:9] offset:12
	s_wait_storecnt 0x0
	s_branch .Lrx_b1_stored
	.Lrx_b1_p1_single:
	global_load_b32 v30, v32, s[8:9] offset:8
	s_wait_loadcnt 0x0
	v_add_f32_e32 v30, v30, v66
	global_store_b32 v32, v30, s[8:9] offset:8
	s_wait_storecnt 0x0
	s_branch .Lrx_b1_stored
	.Lrx_b1_stored:
	s_branch .Lrx_end
	.Lrx_b2:
	s_mul_i32 s12, ttmp9, 4
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_i32 s12, s10
	s_cbranch_scc1 .Lrx_end
	s_mov_b32 s13, s12
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s13, s13, s19
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v8, s13
	v_add_nc_u32_e32 v4, s13, v1
	s_add_co_i32 s13, s12, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cselect_b32 s13, s13, s12
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s13, s13, s19
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v9, s13
	v_add_nc_u32_e32 v5, s13, v1
	s_add_co_i32 s13, s12, 2
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cselect_b32 s13, s13, s12
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s13, s13, s19
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v10, s13
	v_add_nc_u32_e32 v6, s13, v1
	s_add_co_i32 s13, s12, 3
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cselect_b32 s13, s13, s12
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s13, s13, s19
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v11, s13
	v_add_nc_u32_e32 v7, s13, v1
	v_mov_b32_e32 v64, 0
	v_mov_b32_e32 v65, 0
	v_mov_b32_e32 v66, 0
	v_mov_b32_e32 v67, 0
	v_mov_b32_e32 v68, 0
	v_mov_b32_e32 v69, 0
	v_mov_b32_e32 v70, 0
	v_mov_b32_e32 v71, 0
	v_mov_b32_e32 v72, 0
	v_mov_b32_e32 v73, 0
	v_mov_b32_e32 v74, 0
	v_mov_b32_e32 v75, 0
	v_mov_b32_e32 v76, 0
	v_mov_b32_e32 v77, 0
	v_mov_b32_e32 v78, 0
	v_mov_b32_e32 v79, 0
	v_mov_b32_e32 v80, 0
	v_mov_b32_e32 v81, 0
	v_mov_b32_e32 v82, 0
	v_mov_b32_e32 v83, 0
	v_mov_b32_e32 v84, 0
	v_mov_b32_e32 v85, 0
	v_mov_b32_e32 v86, 0
	v_mov_b32_e32 v87, 0
	s_add_co_i32 s2, s14, 3
	s_lshr_b32 s2, s2, 2
	s_cmp_eq_u32 s2, 0
	s_cbranch_scc1 .Lrx_b2_s0_done
	s_mov_b32 s15, 0
	v_lshlrev_b32_e32 v2, 5, v0
	buffer_load_b64 v[12:13], v8, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v20, v4, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[14:15], v9, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v21, v5, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[16:17], v10, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v22, v6, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[18:19], v11, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v23, v7, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	global_load_b128 v[89:92], v2, s[28:29]
	global_load_b128 v[93:96], v2, s[28:29] offset:16
	global_load_b128 v[97:100], v2, s[30:31]
	global_load_b128 v[101:104], v2, s[30:31] offset:16
	.Lrx_b2_s0:
	s_cmp_gt_u32 s2, 1
	s_cselect_b32 s17, 0x220, 0
	s_cselect_b32 s18, 0x1000, 0
	s_add_co_i32 s16, s15, s17
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v3, s18, v2
	v_cmp_gt_u32_e32 vcc_lo, 16, v0
	s_wait_loadcnt 0xb
	v_cndmask_b32_e32 v12, v13, v12, vcc_lo
	s_wait_loadcnt 0xa
	v_and_b32_e32 v30, 0xf0f0f0f, v20
	v_lshrrev_b32_e32 v31, 4, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v31, 0xf0f0f0f, v31
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v24, v30
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v26, v30
	v_cvt_f32_ubyte2_e32 v28, v30
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v25, v31
	v_cvt_f32_ubyte1_e32 v27, v31
	v_cvt_f32_ubyte2_e32 v29, v31
	v_cvt_f32_ubyte3_e32 v30, v30
	v_cvt_f32_ubyte3_e32 v31, v31
	v_fma_mix_f32 v24, v12, v24, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v25, v12, v25, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v26, v12, v26, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v27, v12, v27, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v28, v12, v28, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v29, v12, v29, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v30, v12, v30, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v31, v12, v31, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x9
	v_cndmask_b32_e32 v14, v15, v14, vcc_lo
	s_wait_loadcnt 0x8
	v_and_b32_e32 v40, 0xf0f0f0f, v21
	v_lshrrev_b32_e32 v41, 4, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v41, 0xf0f0f0f, v41
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v34, v40
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v36, v40
	v_cvt_f32_ubyte2_e32 v38, v40
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v35, v41
	v_cvt_f32_ubyte1_e32 v37, v41
	v_cvt_f32_ubyte2_e32 v39, v41
	v_cvt_f32_ubyte3_e32 v40, v40
	v_cvt_f32_ubyte3_e32 v41, v41
	v_fma_mix_f32 v34, v14, v34, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v35, v14, v35, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v36, v14, v36, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v37, v14, v37, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v38, v14, v38, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v39, v14, v39, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v40, v14, v40, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v41, v14, v41, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x7
	v_cndmask_b32_e32 v16, v17, v16, vcc_lo
	s_wait_loadcnt 0x6
	v_and_b32_e32 v50, 0xf0f0f0f, v22
	v_lshrrev_b32_e32 v51, 4, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v51, 0xf0f0f0f, v51
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v44, v50
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v46, v50
	v_cvt_f32_ubyte2_e32 v48, v50
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v45, v51
	v_cvt_f32_ubyte1_e32 v47, v51
	v_cvt_f32_ubyte2_e32 v49, v51
	v_cvt_f32_ubyte3_e32 v50, v50
	v_cvt_f32_ubyte3_e32 v51, v51
	v_fma_mix_f32 v44, v16, v44, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v16, v45, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v16, v46, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v16, v47, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v48, v16, v48, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v49, v16, v49, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v16, v50, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v16, v51, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x5
	v_cndmask_b32_e32 v18, v19, v18, vcc_lo
	s_wait_loadcnt 0x4
	v_and_b32_e32 v60, 0xf0f0f0f, v23
	v_lshrrev_b32_e32 v61, 4, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v61, 0xf0f0f0f, v61
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v54, v60
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v56, v60
	v_cvt_f32_ubyte2_e32 v58, v60
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v55, v61
	v_cvt_f32_ubyte1_e32 v57, v61
	v_cvt_f32_ubyte2_e32 v59, v61
	v_cvt_f32_ubyte3_e32 v60, v60
	v_cvt_f32_ubyte3_e32 v61, v61
	v_fma_mix_f32 v54, v18, v54, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v18, v55, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v18, v56, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v18, v57, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v58, v18, v58, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v59, v18, v59, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v18, v60, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v18, v61, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	buffer_load_b64 v[12:13], v8, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v20, v4, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[14:15], v9, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v21, v5, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[16:17], v10, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v22, v6, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[18:19], v11, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v23, v7, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	s_wait_loadcnt 0xb
	v_dual_mul_f32 v32, v24, v89 :: v_dual_mul_f32 v33, v35, v90
	v_dual_mul_f32 v42, v44, v89 :: v_dual_mul_f32 v43, v55, v90
	s_wait_loadcnt 0x9
	v_dual_mul_f32 v52, v24, v97 :: v_dual_mul_f32 v53, v35, v98
	v_dual_mul_f32 v62, v44, v97 :: v_dual_mul_f32 v63, v55, v98
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v25, v90 :: v_dual_fmac_f32 v33, v34, v89
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v45, v90 :: v_dual_fmac_f32 v43, v54, v89
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v25, v98 :: v_dual_fmac_f32 v53, v34, v97
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v45, v98 :: v_dual_fmac_f32 v63, v54, v97
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v26, v91 :: v_dual_fmac_f32 v33, v36, v91
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v46, v91 :: v_dual_fmac_f32 v43, v56, v91
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v26, v99 :: v_dual_fmac_f32 v53, v36, v99
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v46, v99 :: v_dual_fmac_f32 v63, v56, v99
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v27, v92 :: v_dual_fmac_f32 v33, v37, v92
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v47, v92 :: v_dual_fmac_f32 v43, v57, v92
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v27, v100 :: v_dual_fmac_f32 v53, v37, v100
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v47, v100 :: v_dual_fmac_f32 v63, v57, v100
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v28, v93 :: v_dual_fmac_f32 v33, v38, v93
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v48, v93 :: v_dual_fmac_f32 v43, v58, v93
	s_wait_loadcnt 0x8
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v28, v101 :: v_dual_fmac_f32 v53, v38, v101
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v48, v101 :: v_dual_fmac_f32 v63, v58, v101
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v29, v94 :: v_dual_fmac_f32 v33, v39, v94
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v49, v94 :: v_dual_fmac_f32 v43, v59, v94
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v29, v102 :: v_dual_fmac_f32 v53, v39, v102
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v49, v102 :: v_dual_fmac_f32 v63, v59, v102
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v30, v95 :: v_dual_fmac_f32 v33, v40, v95
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v50, v95 :: v_dual_fmac_f32 v43, v60, v95
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v30, v103 :: v_dual_fmac_f32 v53, v40, v103
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v50, v103 :: v_dual_fmac_f32 v63, v60, v103
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v31, v96 :: v_dual_fmac_f32 v33, v41, v96
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v51, v96 :: v_dual_fmac_f32 v43, v61, v96
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v31, v104 :: v_dual_fmac_f32 v53, v41, v104
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v51, v104 :: v_dual_fmac_f32 v63, v61, v104
	global_load_b128 v[89:92], v3, s[28:29]
	global_load_b128 v[93:96], v3, s[28:29] offset:16
	global_load_b128 v[97:100], v3, s[30:31]
	global_load_b128 v[101:104], v3, s[30:31] offset:16
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v64, v32, v64 :: v_dual_add_f32 v65, v65, v33
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v66, v42, v66 :: v_dual_add_f32 v67, v67, v43
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v76, v52, v76 :: v_dual_add_f32 v77, v77, v53
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v78, v62, v78 :: v_dual_add_f32 v79, v79, v63
	s_mov_b32 s15, s16
	v_mov_b32_e32 v2, v3
	s_add_co_i32 s2, s2, -1
	s_cmp_lg_u32 s2, 0
	s_cbranch_scc1 .Lrx_b2_s0
	s_wait_loadcnt 0x0
	.Lrx_b2_s0_done:
	s_add_co_i32 s2, s14, 2
	s_lshr_b32 s2, s2, 2
	s_cmp_eq_u32 s2, 0
	s_cbranch_scc1 .Lrx_b2_s1_done
	s_mov_b32 s15, 0x88
	v_lshlrev_b32_e32 v2, 5, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v2, 0x400, v2
	buffer_load_b64 v[12:13], v8, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v20, v4, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[14:15], v9, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v21, v5, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[16:17], v10, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v22, v6, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[18:19], v11, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v23, v7, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	global_load_b128 v[89:92], v2, s[28:29]
	global_load_b128 v[93:96], v2, s[28:29] offset:16
	global_load_b128 v[97:100], v2, s[30:31]
	global_load_b128 v[101:104], v2, s[30:31] offset:16
	.Lrx_b2_s1:
	s_cmp_gt_u32 s2, 1
	s_cselect_b32 s17, 0x220, 0
	s_cselect_b32 s18, 0x1000, 0
	s_add_co_i32 s16, s15, s17
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v3, s18, v2
	v_cmp_gt_u32_e32 vcc_lo, 16, v0
	s_wait_loadcnt 0xb
	v_cndmask_b32_e32 v12, v13, v12, vcc_lo
	s_wait_loadcnt 0xa
	v_and_b32_e32 v30, 0xf0f0f0f, v20
	v_lshrrev_b32_e32 v31, 4, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v31, 0xf0f0f0f, v31
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v24, v30
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v26, v30
	v_cvt_f32_ubyte2_e32 v28, v30
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v25, v31
	v_cvt_f32_ubyte1_e32 v27, v31
	v_cvt_f32_ubyte2_e32 v29, v31
	v_cvt_f32_ubyte3_e32 v30, v30
	v_cvt_f32_ubyte3_e32 v31, v31
	v_fma_mix_f32 v24, v12, v24, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v25, v12, v25, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v26, v12, v26, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v27, v12, v27, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v28, v12, v28, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v29, v12, v29, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v30, v12, v30, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v31, v12, v31, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x9
	v_cndmask_b32_e32 v14, v15, v14, vcc_lo
	s_wait_loadcnt 0x8
	v_and_b32_e32 v40, 0xf0f0f0f, v21
	v_lshrrev_b32_e32 v41, 4, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v41, 0xf0f0f0f, v41
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v34, v40
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v36, v40
	v_cvt_f32_ubyte2_e32 v38, v40
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v35, v41
	v_cvt_f32_ubyte1_e32 v37, v41
	v_cvt_f32_ubyte2_e32 v39, v41
	v_cvt_f32_ubyte3_e32 v40, v40
	v_cvt_f32_ubyte3_e32 v41, v41
	v_fma_mix_f32 v34, v14, v34, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v35, v14, v35, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v36, v14, v36, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v37, v14, v37, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v38, v14, v38, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v39, v14, v39, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v40, v14, v40, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v41, v14, v41, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x7
	v_cndmask_b32_e32 v16, v17, v16, vcc_lo
	s_wait_loadcnt 0x6
	v_and_b32_e32 v50, 0xf0f0f0f, v22
	v_lshrrev_b32_e32 v51, 4, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v51, 0xf0f0f0f, v51
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v44, v50
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v46, v50
	v_cvt_f32_ubyte2_e32 v48, v50
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v45, v51
	v_cvt_f32_ubyte1_e32 v47, v51
	v_cvt_f32_ubyte2_e32 v49, v51
	v_cvt_f32_ubyte3_e32 v50, v50
	v_cvt_f32_ubyte3_e32 v51, v51
	v_fma_mix_f32 v44, v16, v44, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v16, v45, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v16, v46, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v16, v47, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v48, v16, v48, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v49, v16, v49, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v16, v50, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v16, v51, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x5
	v_cndmask_b32_e32 v18, v19, v18, vcc_lo
	s_wait_loadcnt 0x4
	v_and_b32_e32 v60, 0xf0f0f0f, v23
	v_lshrrev_b32_e32 v61, 4, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v61, 0xf0f0f0f, v61
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v54, v60
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v56, v60
	v_cvt_f32_ubyte2_e32 v58, v60
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v55, v61
	v_cvt_f32_ubyte1_e32 v57, v61
	v_cvt_f32_ubyte2_e32 v59, v61
	v_cvt_f32_ubyte3_e32 v60, v60
	v_cvt_f32_ubyte3_e32 v61, v61
	v_fma_mix_f32 v54, v18, v54, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v18, v55, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v18, v56, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v18, v57, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v58, v18, v58, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v59, v18, v59, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v18, v60, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v18, v61, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	buffer_load_b64 v[12:13], v8, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v20, v4, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[14:15], v9, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v21, v5, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[16:17], v10, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v22, v6, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[18:19], v11, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v23, v7, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	s_wait_loadcnt 0xb
	v_dual_mul_f32 v32, v24, v89 :: v_dual_mul_f32 v33, v35, v90
	v_dual_mul_f32 v42, v44, v89 :: v_dual_mul_f32 v43, v55, v90
	s_wait_loadcnt 0x9
	v_dual_mul_f32 v52, v24, v97 :: v_dual_mul_f32 v53, v35, v98
	v_dual_mul_f32 v62, v44, v97 :: v_dual_mul_f32 v63, v55, v98
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v25, v90 :: v_dual_fmac_f32 v33, v34, v89
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v45, v90 :: v_dual_fmac_f32 v43, v54, v89
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v25, v98 :: v_dual_fmac_f32 v53, v34, v97
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v45, v98 :: v_dual_fmac_f32 v63, v54, v97
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v26, v91 :: v_dual_fmac_f32 v33, v36, v91
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v46, v91 :: v_dual_fmac_f32 v43, v56, v91
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v26, v99 :: v_dual_fmac_f32 v53, v36, v99
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v46, v99 :: v_dual_fmac_f32 v63, v56, v99
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v27, v92 :: v_dual_fmac_f32 v33, v37, v92
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v47, v92 :: v_dual_fmac_f32 v43, v57, v92
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v27, v100 :: v_dual_fmac_f32 v53, v37, v100
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v47, v100 :: v_dual_fmac_f32 v63, v57, v100
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v28, v93 :: v_dual_fmac_f32 v33, v38, v93
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v48, v93 :: v_dual_fmac_f32 v43, v58, v93
	s_wait_loadcnt 0x8
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v28, v101 :: v_dual_fmac_f32 v53, v38, v101
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v48, v101 :: v_dual_fmac_f32 v63, v58, v101
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v29, v94 :: v_dual_fmac_f32 v33, v39, v94
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v49, v94 :: v_dual_fmac_f32 v43, v59, v94
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v29, v102 :: v_dual_fmac_f32 v53, v39, v102
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v49, v102 :: v_dual_fmac_f32 v63, v59, v102
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v30, v95 :: v_dual_fmac_f32 v33, v40, v95
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v50, v95 :: v_dual_fmac_f32 v43, v60, v95
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v30, v103 :: v_dual_fmac_f32 v53, v40, v103
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v50, v103 :: v_dual_fmac_f32 v63, v60, v103
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v31, v96 :: v_dual_fmac_f32 v33, v41, v96
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v51, v96 :: v_dual_fmac_f32 v43, v61, v96
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v31, v104 :: v_dual_fmac_f32 v53, v41, v104
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v51, v104 :: v_dual_fmac_f32 v63, v61, v104
	global_load_b128 v[89:92], v3, s[28:29]
	global_load_b128 v[93:96], v3, s[28:29] offset:16
	global_load_b128 v[97:100], v3, s[30:31]
	global_load_b128 v[101:104], v3, s[30:31] offset:16
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v68, v32, v68 :: v_dual_add_f32 v69, v69, v33
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v70, v42, v70 :: v_dual_add_f32 v71, v71, v43
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v80, v52, v80 :: v_dual_add_f32 v81, v81, v53
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v82, v62, v82 :: v_dual_add_f32 v83, v83, v63
	s_mov_b32 s15, s16
	v_mov_b32_e32 v2, v3
	s_add_co_i32 s2, s2, -1
	s_cmp_lg_u32 s2, 0
	s_cbranch_scc1 .Lrx_b2_s1
	s_wait_loadcnt 0x0
	.Lrx_b2_s1_done:
	v_dual_add_f32 v64, v64, v68 :: v_dual_add_f32 v65, v65, v69
	v_dual_add_f32 v66, v66, v70 :: v_dual_add_f32 v67, v67, v71
	v_dual_add_f32 v76, v76, v80 :: v_dual_add_f32 v77, v77, v81
	v_dual_add_f32 v78, v78, v82 :: v_dual_add_f32 v79, v79, v83
	v_mov_b32_e32 v68, 0
	v_mov_b32_e32 v69, 0
	v_mov_b32_e32 v70, 0
	v_mov_b32_e32 v71, 0
	v_mov_b32_e32 v80, 0
	v_mov_b32_e32 v81, 0
	v_mov_b32_e32 v82, 0
	v_mov_b32_e32 v83, 0
	s_add_co_i32 s2, s14, 1
	s_lshr_b32 s2, s2, 2
	s_cmp_eq_u32 s2, 0
	s_cbranch_scc1 .Lrx_b2_s2_done
	s_mov_b32 s15, 0x110
	v_lshlrev_b32_e32 v2, 5, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v2, 0x800, v2
	buffer_load_b64 v[12:13], v8, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v20, v4, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[14:15], v9, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v21, v5, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[16:17], v10, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v22, v6, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[18:19], v11, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v23, v7, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	global_load_b128 v[89:92], v2, s[28:29]
	global_load_b128 v[93:96], v2, s[28:29] offset:16
	global_load_b128 v[97:100], v2, s[30:31]
	global_load_b128 v[101:104], v2, s[30:31] offset:16
	.Lrx_b2_s2:
	s_cmp_gt_u32 s2, 1
	s_cselect_b32 s17, 0x220, 0
	s_cselect_b32 s18, 0x1000, 0
	s_add_co_i32 s16, s15, s17
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v3, s18, v2
	v_cmp_gt_u32_e32 vcc_lo, 16, v0
	s_wait_loadcnt 0xb
	v_cndmask_b32_e32 v12, v13, v12, vcc_lo
	s_wait_loadcnt 0xa
	v_and_b32_e32 v30, 0xf0f0f0f, v20
	v_lshrrev_b32_e32 v31, 4, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v31, 0xf0f0f0f, v31
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v24, v30
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v26, v30
	v_cvt_f32_ubyte2_e32 v28, v30
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v25, v31
	v_cvt_f32_ubyte1_e32 v27, v31
	v_cvt_f32_ubyte2_e32 v29, v31
	v_cvt_f32_ubyte3_e32 v30, v30
	v_cvt_f32_ubyte3_e32 v31, v31
	v_fma_mix_f32 v24, v12, v24, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v25, v12, v25, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v26, v12, v26, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v27, v12, v27, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v28, v12, v28, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v29, v12, v29, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v30, v12, v30, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v31, v12, v31, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x9
	v_cndmask_b32_e32 v14, v15, v14, vcc_lo
	s_wait_loadcnt 0x8
	v_and_b32_e32 v40, 0xf0f0f0f, v21
	v_lshrrev_b32_e32 v41, 4, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v41, 0xf0f0f0f, v41
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v34, v40
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v36, v40
	v_cvt_f32_ubyte2_e32 v38, v40
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v35, v41
	v_cvt_f32_ubyte1_e32 v37, v41
	v_cvt_f32_ubyte2_e32 v39, v41
	v_cvt_f32_ubyte3_e32 v40, v40
	v_cvt_f32_ubyte3_e32 v41, v41
	v_fma_mix_f32 v34, v14, v34, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v35, v14, v35, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v36, v14, v36, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v37, v14, v37, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v38, v14, v38, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v39, v14, v39, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v40, v14, v40, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v41, v14, v41, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x7
	v_cndmask_b32_e32 v16, v17, v16, vcc_lo
	s_wait_loadcnt 0x6
	v_and_b32_e32 v50, 0xf0f0f0f, v22
	v_lshrrev_b32_e32 v51, 4, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v51, 0xf0f0f0f, v51
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v44, v50
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v46, v50
	v_cvt_f32_ubyte2_e32 v48, v50
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v45, v51
	v_cvt_f32_ubyte1_e32 v47, v51
	v_cvt_f32_ubyte2_e32 v49, v51
	v_cvt_f32_ubyte3_e32 v50, v50
	v_cvt_f32_ubyte3_e32 v51, v51
	v_fma_mix_f32 v44, v16, v44, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v16, v45, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v16, v46, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v16, v47, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v48, v16, v48, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v49, v16, v49, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v16, v50, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v16, v51, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x5
	v_cndmask_b32_e32 v18, v19, v18, vcc_lo
	s_wait_loadcnt 0x4
	v_and_b32_e32 v60, 0xf0f0f0f, v23
	v_lshrrev_b32_e32 v61, 4, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v61, 0xf0f0f0f, v61
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v54, v60
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v56, v60
	v_cvt_f32_ubyte2_e32 v58, v60
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v55, v61
	v_cvt_f32_ubyte1_e32 v57, v61
	v_cvt_f32_ubyte2_e32 v59, v61
	v_cvt_f32_ubyte3_e32 v60, v60
	v_cvt_f32_ubyte3_e32 v61, v61
	v_fma_mix_f32 v54, v18, v54, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v18, v55, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v18, v56, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v18, v57, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v58, v18, v58, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v59, v18, v59, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v18, v60, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v18, v61, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	buffer_load_b64 v[12:13], v8, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v20, v4, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[14:15], v9, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v21, v5, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[16:17], v10, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v22, v6, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[18:19], v11, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v23, v7, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	s_wait_loadcnt 0xb
	v_dual_mul_f32 v32, v24, v89 :: v_dual_mul_f32 v33, v35, v90
	v_dual_mul_f32 v42, v44, v89 :: v_dual_mul_f32 v43, v55, v90
	s_wait_loadcnt 0x9
	v_dual_mul_f32 v52, v24, v97 :: v_dual_mul_f32 v53, v35, v98
	v_dual_mul_f32 v62, v44, v97 :: v_dual_mul_f32 v63, v55, v98
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v25, v90 :: v_dual_fmac_f32 v33, v34, v89
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v45, v90 :: v_dual_fmac_f32 v43, v54, v89
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v25, v98 :: v_dual_fmac_f32 v53, v34, v97
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v45, v98 :: v_dual_fmac_f32 v63, v54, v97
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v26, v91 :: v_dual_fmac_f32 v33, v36, v91
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v46, v91 :: v_dual_fmac_f32 v43, v56, v91
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v26, v99 :: v_dual_fmac_f32 v53, v36, v99
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v46, v99 :: v_dual_fmac_f32 v63, v56, v99
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v27, v92 :: v_dual_fmac_f32 v33, v37, v92
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v47, v92 :: v_dual_fmac_f32 v43, v57, v92
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v27, v100 :: v_dual_fmac_f32 v53, v37, v100
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v47, v100 :: v_dual_fmac_f32 v63, v57, v100
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v28, v93 :: v_dual_fmac_f32 v33, v38, v93
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v48, v93 :: v_dual_fmac_f32 v43, v58, v93
	s_wait_loadcnt 0x8
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v28, v101 :: v_dual_fmac_f32 v53, v38, v101
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v48, v101 :: v_dual_fmac_f32 v63, v58, v101
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v29, v94 :: v_dual_fmac_f32 v33, v39, v94
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v49, v94 :: v_dual_fmac_f32 v43, v59, v94
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v29, v102 :: v_dual_fmac_f32 v53, v39, v102
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v49, v102 :: v_dual_fmac_f32 v63, v59, v102
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v30, v95 :: v_dual_fmac_f32 v33, v40, v95
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v50, v95 :: v_dual_fmac_f32 v43, v60, v95
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v30, v103 :: v_dual_fmac_f32 v53, v40, v103
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v50, v103 :: v_dual_fmac_f32 v63, v60, v103
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v31, v96 :: v_dual_fmac_f32 v33, v41, v96
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v51, v96 :: v_dual_fmac_f32 v43, v61, v96
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v31, v104 :: v_dual_fmac_f32 v53, v41, v104
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v51, v104 :: v_dual_fmac_f32 v63, v61, v104
	global_load_b128 v[89:92], v3, s[28:29]
	global_load_b128 v[93:96], v3, s[28:29] offset:16
	global_load_b128 v[97:100], v3, s[30:31]
	global_load_b128 v[101:104], v3, s[30:31] offset:16
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v68, v32, v68 :: v_dual_add_f32 v69, v69, v33
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v70, v42, v70 :: v_dual_add_f32 v71, v71, v43
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v80, v52, v80 :: v_dual_add_f32 v81, v81, v53
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v82, v62, v82 :: v_dual_add_f32 v83, v83, v63
	s_mov_b32 s15, s16
	v_mov_b32_e32 v2, v3
	s_add_co_i32 s2, s2, -1
	s_cmp_lg_u32 s2, 0
	s_cbranch_scc1 .Lrx_b2_s2
	s_wait_loadcnt 0x0
	.Lrx_b2_s2_done:
	s_add_co_i32 s2, s14, 0
	s_lshr_b32 s2, s2, 2
	s_cmp_eq_u32 s2, 0
	s_cbranch_scc1 .Lrx_b2_s3_done
	s_mov_b32 s15, 0x198
	v_lshlrev_b32_e32 v2, 5, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v2, 0xc00, v2
	buffer_load_b64 v[12:13], v8, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v20, v4, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[14:15], v9, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v21, v5, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[16:17], v10, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v22, v6, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[18:19], v11, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v23, v7, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	global_load_b128 v[89:92], v2, s[28:29]
	global_load_b128 v[93:96], v2, s[28:29] offset:16
	global_load_b128 v[97:100], v2, s[30:31]
	global_load_b128 v[101:104], v2, s[30:31] offset:16
	.Lrx_b2_s3:
	s_cmp_gt_u32 s2, 1
	s_cselect_b32 s17, 0x220, 0
	s_cselect_b32 s18, 0x1000, 0
	s_add_co_i32 s16, s15, s17
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v3, s18, v2
	v_cmp_gt_u32_e32 vcc_lo, 16, v0
	s_wait_loadcnt 0xb
	v_cndmask_b32_e32 v12, v13, v12, vcc_lo
	s_wait_loadcnt 0xa
	v_and_b32_e32 v30, 0xf0f0f0f, v20
	v_lshrrev_b32_e32 v31, 4, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v31, 0xf0f0f0f, v31
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v24, v30
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v26, v30
	v_cvt_f32_ubyte2_e32 v28, v30
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v25, v31
	v_cvt_f32_ubyte1_e32 v27, v31
	v_cvt_f32_ubyte2_e32 v29, v31
	v_cvt_f32_ubyte3_e32 v30, v30
	v_cvt_f32_ubyte3_e32 v31, v31
	v_fma_mix_f32 v24, v12, v24, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v25, v12, v25, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v26, v12, v26, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v27, v12, v27, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v28, v12, v28, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v29, v12, v29, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v30, v12, v30, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v31, v12, v31, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x9
	v_cndmask_b32_e32 v14, v15, v14, vcc_lo
	s_wait_loadcnt 0x8
	v_and_b32_e32 v40, 0xf0f0f0f, v21
	v_lshrrev_b32_e32 v41, 4, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v41, 0xf0f0f0f, v41
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v34, v40
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v36, v40
	v_cvt_f32_ubyte2_e32 v38, v40
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v35, v41
	v_cvt_f32_ubyte1_e32 v37, v41
	v_cvt_f32_ubyte2_e32 v39, v41
	v_cvt_f32_ubyte3_e32 v40, v40
	v_cvt_f32_ubyte3_e32 v41, v41
	v_fma_mix_f32 v34, v14, v34, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v35, v14, v35, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v36, v14, v36, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v37, v14, v37, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v38, v14, v38, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v39, v14, v39, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v40, v14, v40, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v41, v14, v41, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x7
	v_cndmask_b32_e32 v16, v17, v16, vcc_lo
	s_wait_loadcnt 0x6
	v_and_b32_e32 v50, 0xf0f0f0f, v22
	v_lshrrev_b32_e32 v51, 4, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v51, 0xf0f0f0f, v51
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v44, v50
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v46, v50
	v_cvt_f32_ubyte2_e32 v48, v50
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v45, v51
	v_cvt_f32_ubyte1_e32 v47, v51
	v_cvt_f32_ubyte2_e32 v49, v51
	v_cvt_f32_ubyte3_e32 v50, v50
	v_cvt_f32_ubyte3_e32 v51, v51
	v_fma_mix_f32 v44, v16, v44, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v16, v45, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v16, v46, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v16, v47, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v48, v16, v48, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v49, v16, v49, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v16, v50, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v16, v51, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x5
	v_cndmask_b32_e32 v18, v19, v18, vcc_lo
	s_wait_loadcnt 0x4
	v_and_b32_e32 v60, 0xf0f0f0f, v23
	v_lshrrev_b32_e32 v61, 4, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v61, 0xf0f0f0f, v61
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v54, v60
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v56, v60
	v_cvt_f32_ubyte2_e32 v58, v60
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v55, v61
	v_cvt_f32_ubyte1_e32 v57, v61
	v_cvt_f32_ubyte2_e32 v59, v61
	v_cvt_f32_ubyte3_e32 v60, v60
	v_cvt_f32_ubyte3_e32 v61, v61
	v_fma_mix_f32 v54, v18, v54, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v18, v55, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v18, v56, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v18, v57, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v58, v18, v58, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v59, v18, v59, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v18, v60, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v18, v61, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	buffer_load_b64 v[12:13], v8, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v20, v4, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[14:15], v9, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v21, v5, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[16:17], v10, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v22, v6, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[18:19], v11, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v23, v7, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	s_wait_loadcnt 0xb
	v_dual_mul_f32 v32, v24, v89 :: v_dual_mul_f32 v33, v35, v90
	v_dual_mul_f32 v42, v44, v89 :: v_dual_mul_f32 v43, v55, v90
	s_wait_loadcnt 0x9
	v_dual_mul_f32 v52, v24, v97 :: v_dual_mul_f32 v53, v35, v98
	v_dual_mul_f32 v62, v44, v97 :: v_dual_mul_f32 v63, v55, v98
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v25, v90 :: v_dual_fmac_f32 v33, v34, v89
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v45, v90 :: v_dual_fmac_f32 v43, v54, v89
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v25, v98 :: v_dual_fmac_f32 v53, v34, v97
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v45, v98 :: v_dual_fmac_f32 v63, v54, v97
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v26, v91 :: v_dual_fmac_f32 v33, v36, v91
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v46, v91 :: v_dual_fmac_f32 v43, v56, v91
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v26, v99 :: v_dual_fmac_f32 v53, v36, v99
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v46, v99 :: v_dual_fmac_f32 v63, v56, v99
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v27, v92 :: v_dual_fmac_f32 v33, v37, v92
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v47, v92 :: v_dual_fmac_f32 v43, v57, v92
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v27, v100 :: v_dual_fmac_f32 v53, v37, v100
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v47, v100 :: v_dual_fmac_f32 v63, v57, v100
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v28, v93 :: v_dual_fmac_f32 v33, v38, v93
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v48, v93 :: v_dual_fmac_f32 v43, v58, v93
	s_wait_loadcnt 0x8
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v28, v101 :: v_dual_fmac_f32 v53, v38, v101
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v48, v101 :: v_dual_fmac_f32 v63, v58, v101
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v29, v94 :: v_dual_fmac_f32 v33, v39, v94
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v49, v94 :: v_dual_fmac_f32 v43, v59, v94
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v29, v102 :: v_dual_fmac_f32 v53, v39, v102
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v49, v102 :: v_dual_fmac_f32 v63, v59, v102
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v30, v95 :: v_dual_fmac_f32 v33, v40, v95
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v50, v95 :: v_dual_fmac_f32 v43, v60, v95
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v30, v103 :: v_dual_fmac_f32 v53, v40, v103
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v50, v103 :: v_dual_fmac_f32 v63, v60, v103
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v31, v96 :: v_dual_fmac_f32 v33, v41, v96
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v51, v96 :: v_dual_fmac_f32 v43, v61, v96
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v31, v104 :: v_dual_fmac_f32 v53, v41, v104
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v51, v104 :: v_dual_fmac_f32 v63, v61, v104
	global_load_b128 v[89:92], v3, s[28:29]
	global_load_b128 v[93:96], v3, s[28:29] offset:16
	global_load_b128 v[97:100], v3, s[30:31]
	global_load_b128 v[101:104], v3, s[30:31] offset:16
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v72, v32, v72 :: v_dual_add_f32 v73, v73, v33
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v74, v42, v74 :: v_dual_add_f32 v75, v75, v43
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v84, v52, v84 :: v_dual_add_f32 v85, v85, v53
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v86, v62, v86 :: v_dual_add_f32 v87, v87, v63
	s_mov_b32 s15, s16
	v_mov_b32_e32 v2, v3
	s_add_co_i32 s2, s2, -1
	s_cmp_lg_u32 s2, 0
	s_cbranch_scc1 .Lrx_b2_s3
	s_wait_loadcnt 0x0
	.Lrx_b2_s3_done:
	v_dual_add_f32 v68, v68, v72 :: v_dual_add_f32 v69, v69, v73
	v_dual_add_f32 v70, v70, v74 :: v_dual_add_f32 v71, v71, v75
	v_dual_add_f32 v80, v80, v84 :: v_dual_add_f32 v81, v81, v85
	v_dual_add_f32 v82, v82, v86 :: v_dual_add_f32 v83, v83, v87
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v64, v64, v68 :: v_dual_add_f32 v65, v65, v69
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v66, v66, v70 :: v_dual_add_f32 v67, v67, v71
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v76, v76, v80 :: v_dual_add_f32 v77, v77, v81
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v78, v78, v82 :: v_dual_add_f32 v79, v79, v83
	ds_swizzle_b32 v24, v64 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v25, v65 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v26, v66 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v27, v67 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v28, v76 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v29, v77 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v30, v78 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v31, v79 offset:swizzle(BITMASK_PERM,"1pppp")
	s_wait_dscnt 0x7
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v64, v64, v24
	s_wait_dscnt 0x6
	v_add_f32_e32 v65, v65, v25
	s_wait_dscnt 0x5
	v_add_f32_e32 v66, v66, v26
	s_wait_dscnt 0x4
	v_add_f32_e32 v67, v67, v27
	s_wait_dscnt 0x3
	v_add_f32_e32 v76, v76, v28
	s_wait_dscnt 0x2
	v_add_f32_e32 v77, v77, v29
	s_wait_dscnt 0x1
	v_add_f32_e32 v78, v78, v30
	s_wait_dscnt 0x0
	v_add_f32_e32 v79, v79, v31
	v_cmp_gt_u32_e32 vcc_lo, 24, v0
	v_cndmask_b32_e64 v3, 0, 8, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v3, v3, v0, 2
	ds_bpermute_b32 v24, v3, v64
	ds_bpermute_b32 v25, v3, v65
	ds_bpermute_b32 v26, v3, v66
	ds_bpermute_b32 v27, v3, v67
	ds_bpermute_b32 v28, v3, v76
	ds_bpermute_b32 v29, v3, v77
	ds_bpermute_b32 v30, v3, v78
	ds_bpermute_b32 v31, v3, v79
	s_wait_dscnt 0x7
	v_add_f32_e32 v64, v64, v24
	s_wait_dscnt 0x6
	v_add_f32_e32 v65, v65, v25
	s_wait_dscnt 0x5
	v_add_f32_e32 v66, v66, v26
	s_wait_dscnt 0x4
	v_add_f32_e32 v67, v67, v27
	s_wait_dscnt 0x3
	v_add_f32_e32 v76, v76, v28
	s_wait_dscnt 0x2
	v_add_f32_e32 v77, v77, v29
	s_wait_dscnt 0x1
	v_add_f32_e32 v78, v78, v30
	s_wait_dscnt 0x0
	v_add_f32_e32 v79, v79, v31
	v_cmp_gt_u32_e32 vcc_lo, 28, v0
	v_cndmask_b32_e64 v3, 0, 4, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v3, v3, v0, 2
	ds_bpermute_b32 v24, v3, v64
	ds_bpermute_b32 v25, v3, v65
	ds_bpermute_b32 v26, v3, v66
	ds_bpermute_b32 v27, v3, v67
	ds_bpermute_b32 v28, v3, v76
	ds_bpermute_b32 v29, v3, v77
	ds_bpermute_b32 v30, v3, v78
	ds_bpermute_b32 v31, v3, v79
	s_wait_dscnt 0x7
	v_add_f32_e32 v64, v64, v24
	s_wait_dscnt 0x6
	v_add_f32_e32 v65, v65, v25
	s_wait_dscnt 0x5
	v_add_f32_e32 v66, v66, v26
	s_wait_dscnt 0x4
	v_add_f32_e32 v67, v67, v27
	s_wait_dscnt 0x3
	v_add_f32_e32 v76, v76, v28
	s_wait_dscnt 0x2
	v_add_f32_e32 v77, v77, v29
	s_wait_dscnt 0x1
	v_add_f32_e32 v78, v78, v30
	s_wait_dscnt 0x0
	v_add_f32_e32 v79, v79, v31
	v_cmp_gt_u32_e32 vcc_lo, 30, v0
	v_cndmask_b32_e64 v3, 0, 2, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v3, v3, v0, 2
	ds_bpermute_b32 v24, v3, v64
	ds_bpermute_b32 v25, v3, v65
	ds_bpermute_b32 v26, v3, v66
	ds_bpermute_b32 v27, v3, v67
	ds_bpermute_b32 v28, v3, v76
	ds_bpermute_b32 v29, v3, v77
	ds_bpermute_b32 v30, v3, v78
	ds_bpermute_b32 v31, v3, v79
	s_wait_dscnt 0x7
	v_add_f32_e32 v64, v64, v24
	s_wait_dscnt 0x6
	v_add_f32_e32 v65, v65, v25
	s_wait_dscnt 0x5
	v_add_f32_e32 v66, v66, v26
	s_wait_dscnt 0x4
	v_add_f32_e32 v67, v67, v27
	s_wait_dscnt 0x3
	v_add_f32_e32 v76, v76, v28
	s_wait_dscnt 0x2
	v_add_f32_e32 v77, v77, v29
	s_wait_dscnt 0x1
	v_add_f32_e32 v78, v78, v30
	s_wait_dscnt 0x0
	v_add_f32_e32 v79, v79, v31
	v_cmp_gt_u32_e32 vcc_lo, 31, v0
	v_cndmask_b32_e64 v3, 0, 1, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v3, v3, v0, 2
	ds_bpermute_b32 v24, v3, v64
	ds_bpermute_b32 v25, v3, v65
	ds_bpermute_b32 v26, v3, v66
	ds_bpermute_b32 v27, v3, v67
	ds_bpermute_b32 v28, v3, v76
	ds_bpermute_b32 v29, v3, v77
	ds_bpermute_b32 v30, v3, v78
	ds_bpermute_b32 v31, v3, v79
	s_wait_dscnt 0x7
	v_add_f32_e32 v64, v64, v24
	s_wait_dscnt 0x6
	v_add_f32_e32 v65, v65, v25
	s_wait_dscnt 0x5
	v_add_f32_e32 v66, v66, v26
	s_wait_dscnt 0x4
	v_add_f32_e32 v67, v67, v27
	s_wait_dscnt 0x3
	v_add_f32_e32 v76, v76, v28
	s_wait_dscnt 0x2
	v_add_f32_e32 v77, v77, v29
	s_wait_dscnt 0x1
	v_add_f32_e32 v78, v78, v30
	s_wait_dscnt 0x0
	v_add_f32_e32 v79, v79, v31
	v_cmpx_eq_u32_e32 0, v0
	s_lshl_b32 s46, s12, 2
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v40, s46
	s_mul_i32 s47, s45, 1
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s47, s47, s46
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v41, s47
	s_add_co_i32 s13, s12, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cbranch_scc0 .Lrx_b2_p0_single
	global_load_b32 v32, v40, s[8:9]
	global_load_b32 v33, v40, s[8:9] offset:4
	global_load_b32 v36, v41, s[8:9]
	global_load_b32 v37, v41, s[8:9] offset:4
	s_wait_loadcnt 0x3
	v_add_f32_e32 v32, v64, v32
	global_store_b32 v40, v32, s[8:9]
	s_wait_loadcnt 0x2
	v_add_f32_e32 v33, v65, v33
	global_store_b32 v40, v33, s[8:9] offset:4
	s_wait_loadcnt 0x1
	v_add_f32_e32 v36, v76, v36
	global_store_b32 v41, v36, s[8:9]
	s_wait_loadcnt 0x0
	v_add_f32_e32 v37, v77, v37
	global_store_b32 v41, v37, s[8:9] offset:4
	s_wait_storecnt 0x0
	s_branch .Lrx_b2_p0_next
	.Lrx_b2_p0_single:
	global_load_b32 v32, v40, s[8:9]
	global_load_b32 v36, v41, s[8:9]
	s_wait_loadcnt 0x1
	v_add_f32_e32 v32, v32, v64
	global_store_b32 v40, v32, s[8:9]
	s_wait_loadcnt 0x0
	v_add_f32_e32 v36, v36, v76
	global_store_b32 v41, v36, s[8:9]
	s_wait_storecnt 0x0
	s_branch .Lrx_b2_stored
	.Lrx_b2_p0_next:
	s_add_co_i32 s13, s12, 2
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cbranch_scc0 .Lrx_b2_stored
	s_add_co_i32 s13, s12, 3
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cbranch_scc0 .Lrx_b2_p1_single
	global_load_b32 v34, v40, s[8:9] offset:8
	global_load_b32 v35, v40, s[8:9] offset:12
	global_load_b32 v38, v41, s[8:9] offset:8
	global_load_b32 v39, v41, s[8:9] offset:12
	s_wait_loadcnt 0x3
	v_add_f32_e32 v34, v66, v34
	global_store_b32 v40, v34, s[8:9] offset:8
	s_wait_loadcnt 0x2
	v_add_f32_e32 v35, v67, v35
	global_store_b32 v40, v35, s[8:9] offset:12
	s_wait_loadcnt 0x1
	v_add_f32_e32 v38, v78, v38
	global_store_b32 v41, v38, s[8:9] offset:8
	s_wait_loadcnt 0x0
	v_add_f32_e32 v39, v79, v39
	global_store_b32 v41, v39, s[8:9] offset:12
	s_wait_storecnt 0x0
	s_branch .Lrx_b2_stored
	.Lrx_b2_p1_single:
	global_load_b32 v34, v40, s[8:9] offset:8
	global_load_b32 v38, v41, s[8:9] offset:8
	s_wait_loadcnt 0x1
	v_add_f32_e32 v34, v34, v66
	global_store_b32 v40, v34, s[8:9] offset:8
	s_wait_loadcnt 0x0
	v_add_f32_e32 v38, v38, v78
	global_store_b32 v41, v38, s[8:9] offset:8
	s_wait_storecnt 0x0
	s_branch .Lrx_b2_stored
	.Lrx_b2_stored:
	s_branch .Lrx_end
	.Lrx_b3:
	s_mul_i32 s12, ttmp9, 4
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_i32 s12, s10
	s_cbranch_scc1 .Lrx_end
	s_mov_b32 s13, s12
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s13, s13, s19
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v8, s13
	v_add_nc_u32_e32 v4, s13, v1
	s_add_co_i32 s13, s12, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cselect_b32 s13, s13, s12
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s13, s13, s19
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v9, s13
	v_add_nc_u32_e32 v5, s13, v1
	s_add_co_i32 s13, s12, 2
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cselect_b32 s13, s13, s12
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s13, s13, s19
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v10, s13
	v_add_nc_u32_e32 v6, s13, v1
	s_add_co_i32 s13, s12, 3
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cselect_b32 s13, s13, s12
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s13, s13, s19
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v11, s13
	v_add_nc_u32_e32 v7, s13, v1
	v_mov_b32_e32 v64, 0
	v_mov_b32_e32 v65, 0
	v_mov_b32_e32 v66, 0
	v_mov_b32_e32 v67, 0
	v_mov_b32_e32 v68, 0
	v_mov_b32_e32 v69, 0
	v_mov_b32_e32 v70, 0
	v_mov_b32_e32 v71, 0
	v_mov_b32_e32 v72, 0
	v_mov_b32_e32 v73, 0
	v_mov_b32_e32 v74, 0
	v_mov_b32_e32 v75, 0
	v_mov_b32_e32 v76, 0
	v_mov_b32_e32 v77, 0
	v_mov_b32_e32 v78, 0
	v_mov_b32_e32 v79, 0
	v_mov_b32_e32 v80, 0
	v_mov_b32_e32 v81, 0
	v_mov_b32_e32 v82, 0
	v_mov_b32_e32 v83, 0
	v_mov_b32_e32 v84, 0
	v_mov_b32_e32 v85, 0
	v_mov_b32_e32 v86, 0
	v_mov_b32_e32 v87, 0
	v_mov_b32_e32 v88, 0
	v_mov_b32_e32 v89, 0
	v_mov_b32_e32 v90, 0
	v_mov_b32_e32 v91, 0
	v_mov_b32_e32 v92, 0
	v_mov_b32_e32 v93, 0
	v_mov_b32_e32 v94, 0
	v_mov_b32_e32 v95, 0
	v_mov_b32_e32 v96, 0
	v_mov_b32_e32 v97, 0
	v_mov_b32_e32 v98, 0
	v_mov_b32_e32 v99, 0
	s_add_co_i32 s2, s14, 3
	s_lshr_b32 s2, s2, 2
	s_cmp_eq_u32 s2, 0
	s_cbranch_scc1 .Lrx_b3_s0_done
	s_mov_b32 s15, 0
	v_lshlrev_b32_e32 v2, 5, v0
	buffer_load_b64 v[12:13], v8, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v20, v4, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[14:15], v9, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v21, v5, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[16:17], v10, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v22, v6, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[18:19], v11, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v23, v7, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	global_load_b128 v[101:104], v2, s[28:29]
	global_load_b128 v[105:108], v2, s[28:29] offset:16
	global_load_b128 v[109:112], v2, s[30:31]
	global_load_b128 v[113:116], v2, s[30:31] offset:16
	global_load_b128 v[117:120], v2, s[32:33]
	global_load_b128 v[121:124], v2, s[32:33] offset:16
	.Lrx_b3_s0:
	s_cmp_gt_u32 s2, 1
	s_cselect_b32 s17, 0x220, 0
	s_cselect_b32 s18, 0x1000, 0
	s_add_co_i32 s16, s15, s17
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v3, s18, v2
	v_cmp_gt_u32_e32 vcc_lo, 16, v0
	s_wait_loadcnt 0xd
	v_cndmask_b32_e32 v12, v13, v12, vcc_lo
	s_wait_loadcnt 0xc
	v_and_b32_e32 v30, 0xf0f0f0f, v20
	v_lshrrev_b32_e32 v31, 4, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v31, 0xf0f0f0f, v31
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v24, v30
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v26, v30
	v_cvt_f32_ubyte2_e32 v28, v30
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v25, v31
	v_cvt_f32_ubyte1_e32 v27, v31
	v_cvt_f32_ubyte2_e32 v29, v31
	v_cvt_f32_ubyte3_e32 v30, v30
	v_cvt_f32_ubyte3_e32 v31, v31
	v_fma_mix_f32 v24, v12, v24, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v25, v12, v25, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v26, v12, v26, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v27, v12, v27, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v28, v12, v28, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v29, v12, v29, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v30, v12, v30, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v31, v12, v31, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xb
	v_cndmask_b32_e32 v14, v15, v14, vcc_lo
	s_wait_loadcnt 0xa
	v_and_b32_e32 v40, 0xf0f0f0f, v21
	v_lshrrev_b32_e32 v41, 4, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v41, 0xf0f0f0f, v41
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v34, v40
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v36, v40
	v_cvt_f32_ubyte2_e32 v38, v40
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v35, v41
	v_cvt_f32_ubyte1_e32 v37, v41
	v_cvt_f32_ubyte2_e32 v39, v41
	v_cvt_f32_ubyte3_e32 v40, v40
	v_cvt_f32_ubyte3_e32 v41, v41
	v_fma_mix_f32 v34, v14, v34, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v35, v14, v35, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v36, v14, v36, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v37, v14, v37, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v38, v14, v38, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v39, v14, v39, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v40, v14, v40, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v41, v14, v41, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x9
	v_cndmask_b32_e32 v16, v17, v16, vcc_lo
	s_wait_loadcnt 0x8
	v_and_b32_e32 v50, 0xf0f0f0f, v22
	v_lshrrev_b32_e32 v51, 4, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v51, 0xf0f0f0f, v51
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v44, v50
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v46, v50
	v_cvt_f32_ubyte2_e32 v48, v50
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v45, v51
	v_cvt_f32_ubyte1_e32 v47, v51
	v_cvt_f32_ubyte2_e32 v49, v51
	v_cvt_f32_ubyte3_e32 v50, v50
	v_cvt_f32_ubyte3_e32 v51, v51
	v_fma_mix_f32 v44, v16, v44, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v16, v45, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v16, v46, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v16, v47, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v48, v16, v48, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v49, v16, v49, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v16, v50, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v16, v51, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x7
	v_cndmask_b32_e32 v18, v19, v18, vcc_lo
	s_wait_loadcnt 0x6
	v_and_b32_e32 v60, 0xf0f0f0f, v23
	v_lshrrev_b32_e32 v61, 4, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v61, 0xf0f0f0f, v61
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v54, v60
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v56, v60
	v_cvt_f32_ubyte2_e32 v58, v60
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v55, v61
	v_cvt_f32_ubyte1_e32 v57, v61
	v_cvt_f32_ubyte2_e32 v59, v61
	v_cvt_f32_ubyte3_e32 v60, v60
	v_cvt_f32_ubyte3_e32 v61, v61
	v_fma_mix_f32 v54, v18, v54, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v18, v55, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v18, v56, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v18, v57, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v58, v18, v58, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v59, v18, v59, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v18, v60, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v18, v61, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	buffer_load_b64 v[12:13], v8, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v20, v4, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[14:15], v9, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v21, v5, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[16:17], v10, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v22, v6, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[18:19], v11, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v23, v7, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	s_wait_loadcnt 0xd
	v_dual_mul_f32 v32, v24, v101 :: v_dual_mul_f32 v33, v35, v102
	v_dual_mul_f32 v42, v44, v101 :: v_dual_mul_f32 v43, v55, v102
	s_wait_loadcnt 0xb
	v_dual_mul_f32 v52, v24, v109 :: v_dual_mul_f32 v53, v35, v110
	v_dual_mul_f32 v62, v44, v109 :: v_dual_mul_f32 v63, v55, v110
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v25, v102 :: v_dual_fmac_f32 v33, v34, v101
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v45, v102 :: v_dual_fmac_f32 v43, v54, v101
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v25, v110 :: v_dual_fmac_f32 v53, v34, v109
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v45, v110 :: v_dual_fmac_f32 v63, v54, v109
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v26, v103 :: v_dual_fmac_f32 v33, v36, v103
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v46, v103 :: v_dual_fmac_f32 v43, v56, v103
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v26, v111 :: v_dual_fmac_f32 v53, v36, v111
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v46, v111 :: v_dual_fmac_f32 v63, v56, v111
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v27, v104 :: v_dual_fmac_f32 v33, v37, v104
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v47, v104 :: v_dual_fmac_f32 v43, v57, v104
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v27, v112 :: v_dual_fmac_f32 v53, v37, v112
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v47, v112 :: v_dual_fmac_f32 v63, v57, v112
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v28, v105 :: v_dual_fmac_f32 v33, v38, v105
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v48, v105 :: v_dual_fmac_f32 v43, v58, v105
	s_wait_loadcnt 0xa
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v28, v113 :: v_dual_fmac_f32 v53, v38, v113
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v48, v113 :: v_dual_fmac_f32 v63, v58, v113
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v29, v106 :: v_dual_fmac_f32 v33, v39, v106
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v49, v106 :: v_dual_fmac_f32 v43, v59, v106
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v29, v114 :: v_dual_fmac_f32 v53, v39, v114
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v49, v114 :: v_dual_fmac_f32 v63, v59, v114
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v30, v107 :: v_dual_fmac_f32 v33, v40, v107
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v50, v107 :: v_dual_fmac_f32 v43, v60, v107
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v30, v115 :: v_dual_fmac_f32 v53, v40, v115
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v50, v115 :: v_dual_fmac_f32 v63, v60, v115
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v31, v108 :: v_dual_fmac_f32 v33, v41, v108
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v51, v108 :: v_dual_fmac_f32 v43, v61, v108
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v31, v116 :: v_dual_fmac_f32 v53, v41, v116
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v51, v116 :: v_dual_fmac_f32 v63, v61, v116
	global_load_b128 v[101:104], v3, s[28:29]
	global_load_b128 v[105:108], v3, s[28:29] offset:16
	global_load_b128 v[109:112], v3, s[30:31]
	global_load_b128 v[113:116], v3, s[30:31] offset:16
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v64, v32, v64 :: v_dual_add_f32 v65, v65, v33
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v66, v42, v66 :: v_dual_add_f32 v67, v67, v43
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v76, v52, v76 :: v_dual_add_f32 v77, v77, v53
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v78, v62, v78 :: v_dual_add_f32 v79, v79, v63
	s_wait_loadcnt 0xd
	v_dual_mul_f32 v32, v24, v117 :: v_dual_mul_f32 v33, v35, v118
	v_dual_mul_f32 v42, v44, v117 :: v_dual_mul_f32 v43, v55, v118
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v25, v118 :: v_dual_fmac_f32 v33, v34, v117
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v45, v118 :: v_dual_fmac_f32 v43, v54, v117
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v26, v119 :: v_dual_fmac_f32 v33, v36, v119
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v46, v119 :: v_dual_fmac_f32 v43, v56, v119
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v27, v120 :: v_dual_fmac_f32 v33, v37, v120
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v47, v120 :: v_dual_fmac_f32 v43, v57, v120
	s_wait_loadcnt 0xc
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v28, v121 :: v_dual_fmac_f32 v33, v38, v121
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v48, v121 :: v_dual_fmac_f32 v43, v58, v121
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v29, v122 :: v_dual_fmac_f32 v33, v39, v122
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v49, v122 :: v_dual_fmac_f32 v43, v59, v122
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v30, v123 :: v_dual_fmac_f32 v33, v40, v123
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v50, v123 :: v_dual_fmac_f32 v43, v60, v123
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v31, v124 :: v_dual_fmac_f32 v33, v41, v124
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v51, v124 :: v_dual_fmac_f32 v43, v61, v124
	global_load_b128 v[117:120], v3, s[32:33]
	global_load_b128 v[121:124], v3, s[32:33] offset:16
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_add_f32 v88, v32, v88 :: v_dual_add_f32 v89, v89, v33
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_add_f32 v90, v42, v90 :: v_dual_add_f32 v91, v91, v43
	s_mov_b32 s15, s16
	v_mov_b32_e32 v2, v3
	s_add_co_i32 s2, s2, -1
	s_cmp_lg_u32 s2, 0
	s_cbranch_scc1 .Lrx_b3_s0
	s_wait_loadcnt 0x0
	.Lrx_b3_s0_done:
	s_add_co_i32 s2, s14, 2
	s_lshr_b32 s2, s2, 2
	s_cmp_eq_u32 s2, 0
	s_cbranch_scc1 .Lrx_b3_s1_done
	s_mov_b32 s15, 0x88
	v_lshlrev_b32_e32 v2, 5, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v2, 0x400, v2
	buffer_load_b64 v[12:13], v8, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v20, v4, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[14:15], v9, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v21, v5, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[16:17], v10, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v22, v6, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[18:19], v11, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v23, v7, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	global_load_b128 v[101:104], v2, s[28:29]
	global_load_b128 v[105:108], v2, s[28:29] offset:16
	global_load_b128 v[109:112], v2, s[30:31]
	global_load_b128 v[113:116], v2, s[30:31] offset:16
	global_load_b128 v[117:120], v2, s[32:33]
	global_load_b128 v[121:124], v2, s[32:33] offset:16
	.Lrx_b3_s1:
	s_cmp_gt_u32 s2, 1
	s_cselect_b32 s17, 0x220, 0
	s_cselect_b32 s18, 0x1000, 0
	s_add_co_i32 s16, s15, s17
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v3, s18, v2
	v_cmp_gt_u32_e32 vcc_lo, 16, v0
	s_wait_loadcnt 0xd
	v_cndmask_b32_e32 v12, v13, v12, vcc_lo
	s_wait_loadcnt 0xc
	v_and_b32_e32 v30, 0xf0f0f0f, v20
	v_lshrrev_b32_e32 v31, 4, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v31, 0xf0f0f0f, v31
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v24, v30
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v26, v30
	v_cvt_f32_ubyte2_e32 v28, v30
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v25, v31
	v_cvt_f32_ubyte1_e32 v27, v31
	v_cvt_f32_ubyte2_e32 v29, v31
	v_cvt_f32_ubyte3_e32 v30, v30
	v_cvt_f32_ubyte3_e32 v31, v31
	v_fma_mix_f32 v24, v12, v24, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v25, v12, v25, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v26, v12, v26, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v27, v12, v27, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v28, v12, v28, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v29, v12, v29, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v30, v12, v30, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v31, v12, v31, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xb
	v_cndmask_b32_e32 v14, v15, v14, vcc_lo
	s_wait_loadcnt 0xa
	v_and_b32_e32 v40, 0xf0f0f0f, v21
	v_lshrrev_b32_e32 v41, 4, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v41, 0xf0f0f0f, v41
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v34, v40
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v36, v40
	v_cvt_f32_ubyte2_e32 v38, v40
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v35, v41
	v_cvt_f32_ubyte1_e32 v37, v41
	v_cvt_f32_ubyte2_e32 v39, v41
	v_cvt_f32_ubyte3_e32 v40, v40
	v_cvt_f32_ubyte3_e32 v41, v41
	v_fma_mix_f32 v34, v14, v34, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v35, v14, v35, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v36, v14, v36, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v37, v14, v37, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v38, v14, v38, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v39, v14, v39, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v40, v14, v40, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v41, v14, v41, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x9
	v_cndmask_b32_e32 v16, v17, v16, vcc_lo
	s_wait_loadcnt 0x8
	v_and_b32_e32 v50, 0xf0f0f0f, v22
	v_lshrrev_b32_e32 v51, 4, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v51, 0xf0f0f0f, v51
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v44, v50
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v46, v50
	v_cvt_f32_ubyte2_e32 v48, v50
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v45, v51
	v_cvt_f32_ubyte1_e32 v47, v51
	v_cvt_f32_ubyte2_e32 v49, v51
	v_cvt_f32_ubyte3_e32 v50, v50
	v_cvt_f32_ubyte3_e32 v51, v51
	v_fma_mix_f32 v44, v16, v44, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v16, v45, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v16, v46, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v16, v47, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v48, v16, v48, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v49, v16, v49, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v16, v50, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v16, v51, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x7
	v_cndmask_b32_e32 v18, v19, v18, vcc_lo
	s_wait_loadcnt 0x6
	v_and_b32_e32 v60, 0xf0f0f0f, v23
	v_lshrrev_b32_e32 v61, 4, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v61, 0xf0f0f0f, v61
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v54, v60
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v56, v60
	v_cvt_f32_ubyte2_e32 v58, v60
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v55, v61
	v_cvt_f32_ubyte1_e32 v57, v61
	v_cvt_f32_ubyte2_e32 v59, v61
	v_cvt_f32_ubyte3_e32 v60, v60
	v_cvt_f32_ubyte3_e32 v61, v61
	v_fma_mix_f32 v54, v18, v54, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v18, v55, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v18, v56, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v18, v57, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v58, v18, v58, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v59, v18, v59, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v18, v60, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v18, v61, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	buffer_load_b64 v[12:13], v8, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v20, v4, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[14:15], v9, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v21, v5, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[16:17], v10, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v22, v6, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[18:19], v11, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v23, v7, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	s_wait_loadcnt 0xd
	v_dual_mul_f32 v32, v24, v101 :: v_dual_mul_f32 v33, v35, v102
	v_dual_mul_f32 v42, v44, v101 :: v_dual_mul_f32 v43, v55, v102
	s_wait_loadcnt 0xb
	v_dual_mul_f32 v52, v24, v109 :: v_dual_mul_f32 v53, v35, v110
	v_dual_mul_f32 v62, v44, v109 :: v_dual_mul_f32 v63, v55, v110
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v25, v102 :: v_dual_fmac_f32 v33, v34, v101
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v45, v102 :: v_dual_fmac_f32 v43, v54, v101
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v25, v110 :: v_dual_fmac_f32 v53, v34, v109
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v45, v110 :: v_dual_fmac_f32 v63, v54, v109
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v26, v103 :: v_dual_fmac_f32 v33, v36, v103
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v46, v103 :: v_dual_fmac_f32 v43, v56, v103
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v26, v111 :: v_dual_fmac_f32 v53, v36, v111
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v46, v111 :: v_dual_fmac_f32 v63, v56, v111
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v27, v104 :: v_dual_fmac_f32 v33, v37, v104
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v47, v104 :: v_dual_fmac_f32 v43, v57, v104
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v27, v112 :: v_dual_fmac_f32 v53, v37, v112
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v47, v112 :: v_dual_fmac_f32 v63, v57, v112
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v28, v105 :: v_dual_fmac_f32 v33, v38, v105
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v48, v105 :: v_dual_fmac_f32 v43, v58, v105
	s_wait_loadcnt 0xa
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v28, v113 :: v_dual_fmac_f32 v53, v38, v113
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v48, v113 :: v_dual_fmac_f32 v63, v58, v113
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v29, v106 :: v_dual_fmac_f32 v33, v39, v106
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v49, v106 :: v_dual_fmac_f32 v43, v59, v106
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v29, v114 :: v_dual_fmac_f32 v53, v39, v114
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v49, v114 :: v_dual_fmac_f32 v63, v59, v114
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v30, v107 :: v_dual_fmac_f32 v33, v40, v107
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v50, v107 :: v_dual_fmac_f32 v43, v60, v107
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v30, v115 :: v_dual_fmac_f32 v53, v40, v115
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v50, v115 :: v_dual_fmac_f32 v63, v60, v115
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v31, v108 :: v_dual_fmac_f32 v33, v41, v108
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v51, v108 :: v_dual_fmac_f32 v43, v61, v108
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v31, v116 :: v_dual_fmac_f32 v53, v41, v116
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v51, v116 :: v_dual_fmac_f32 v63, v61, v116
	global_load_b128 v[101:104], v3, s[28:29]
	global_load_b128 v[105:108], v3, s[28:29] offset:16
	global_load_b128 v[109:112], v3, s[30:31]
	global_load_b128 v[113:116], v3, s[30:31] offset:16
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v68, v32, v68 :: v_dual_add_f32 v69, v69, v33
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v70, v42, v70 :: v_dual_add_f32 v71, v71, v43
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v80, v52, v80 :: v_dual_add_f32 v81, v81, v53
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v82, v62, v82 :: v_dual_add_f32 v83, v83, v63
	s_wait_loadcnt 0xd
	v_dual_mul_f32 v32, v24, v117 :: v_dual_mul_f32 v33, v35, v118
	v_dual_mul_f32 v42, v44, v117 :: v_dual_mul_f32 v43, v55, v118
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v25, v118 :: v_dual_fmac_f32 v33, v34, v117
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v45, v118 :: v_dual_fmac_f32 v43, v54, v117
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v26, v119 :: v_dual_fmac_f32 v33, v36, v119
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v46, v119 :: v_dual_fmac_f32 v43, v56, v119
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v27, v120 :: v_dual_fmac_f32 v33, v37, v120
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v47, v120 :: v_dual_fmac_f32 v43, v57, v120
	s_wait_loadcnt 0xc
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v28, v121 :: v_dual_fmac_f32 v33, v38, v121
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v48, v121 :: v_dual_fmac_f32 v43, v58, v121
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v29, v122 :: v_dual_fmac_f32 v33, v39, v122
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v49, v122 :: v_dual_fmac_f32 v43, v59, v122
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v30, v123 :: v_dual_fmac_f32 v33, v40, v123
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v50, v123 :: v_dual_fmac_f32 v43, v60, v123
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v31, v124 :: v_dual_fmac_f32 v33, v41, v124
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v51, v124 :: v_dual_fmac_f32 v43, v61, v124
	global_load_b128 v[117:120], v3, s[32:33]
	global_load_b128 v[121:124], v3, s[32:33] offset:16
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_add_f32 v92, v32, v92 :: v_dual_add_f32 v93, v93, v33
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_add_f32 v94, v42, v94 :: v_dual_add_f32 v95, v95, v43
	s_mov_b32 s15, s16
	v_mov_b32_e32 v2, v3
	s_add_co_i32 s2, s2, -1
	s_cmp_lg_u32 s2, 0
	s_cbranch_scc1 .Lrx_b3_s1
	s_wait_loadcnt 0x0
	.Lrx_b3_s1_done:
	v_dual_add_f32 v64, v64, v68 :: v_dual_add_f32 v65, v65, v69
	v_dual_add_f32 v66, v66, v70 :: v_dual_add_f32 v67, v67, v71
	v_dual_add_f32 v76, v76, v80 :: v_dual_add_f32 v77, v77, v81
	v_dual_add_f32 v78, v78, v82 :: v_dual_add_f32 v79, v79, v83
	v_dual_add_f32 v88, v88, v92 :: v_dual_add_f32 v89, v89, v93
	v_dual_add_f32 v90, v90, v94 :: v_dual_add_f32 v91, v91, v95
	v_mov_b32_e32 v68, 0
	v_mov_b32_e32 v69, 0
	v_mov_b32_e32 v70, 0
	v_mov_b32_e32 v71, 0
	v_mov_b32_e32 v80, 0
	v_mov_b32_e32 v81, 0
	v_mov_b32_e32 v82, 0
	v_mov_b32_e32 v83, 0
	v_mov_b32_e32 v92, 0
	v_mov_b32_e32 v93, 0
	v_mov_b32_e32 v94, 0
	v_mov_b32_e32 v95, 0
	s_add_co_i32 s2, s14, 1
	s_lshr_b32 s2, s2, 2
	s_cmp_eq_u32 s2, 0
	s_cbranch_scc1 .Lrx_b3_s2_done
	s_mov_b32 s15, 0x110
	v_lshlrev_b32_e32 v2, 5, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v2, 0x800, v2
	buffer_load_b64 v[12:13], v8, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v20, v4, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[14:15], v9, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v21, v5, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[16:17], v10, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v22, v6, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[18:19], v11, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v23, v7, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	global_load_b128 v[101:104], v2, s[28:29]
	global_load_b128 v[105:108], v2, s[28:29] offset:16
	global_load_b128 v[109:112], v2, s[30:31]
	global_load_b128 v[113:116], v2, s[30:31] offset:16
	global_load_b128 v[117:120], v2, s[32:33]
	global_load_b128 v[121:124], v2, s[32:33] offset:16
	.Lrx_b3_s2:
	s_cmp_gt_u32 s2, 1
	s_cselect_b32 s17, 0x220, 0
	s_cselect_b32 s18, 0x1000, 0
	s_add_co_i32 s16, s15, s17
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v3, s18, v2
	v_cmp_gt_u32_e32 vcc_lo, 16, v0
	s_wait_loadcnt 0xd
	v_cndmask_b32_e32 v12, v13, v12, vcc_lo
	s_wait_loadcnt 0xc
	v_and_b32_e32 v30, 0xf0f0f0f, v20
	v_lshrrev_b32_e32 v31, 4, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v31, 0xf0f0f0f, v31
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v24, v30
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v26, v30
	v_cvt_f32_ubyte2_e32 v28, v30
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v25, v31
	v_cvt_f32_ubyte1_e32 v27, v31
	v_cvt_f32_ubyte2_e32 v29, v31
	v_cvt_f32_ubyte3_e32 v30, v30
	v_cvt_f32_ubyte3_e32 v31, v31
	v_fma_mix_f32 v24, v12, v24, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v25, v12, v25, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v26, v12, v26, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v27, v12, v27, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v28, v12, v28, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v29, v12, v29, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v30, v12, v30, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v31, v12, v31, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xb
	v_cndmask_b32_e32 v14, v15, v14, vcc_lo
	s_wait_loadcnt 0xa
	v_and_b32_e32 v40, 0xf0f0f0f, v21
	v_lshrrev_b32_e32 v41, 4, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v41, 0xf0f0f0f, v41
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v34, v40
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v36, v40
	v_cvt_f32_ubyte2_e32 v38, v40
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v35, v41
	v_cvt_f32_ubyte1_e32 v37, v41
	v_cvt_f32_ubyte2_e32 v39, v41
	v_cvt_f32_ubyte3_e32 v40, v40
	v_cvt_f32_ubyte3_e32 v41, v41
	v_fma_mix_f32 v34, v14, v34, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v35, v14, v35, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v36, v14, v36, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v37, v14, v37, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v38, v14, v38, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v39, v14, v39, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v40, v14, v40, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v41, v14, v41, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x9
	v_cndmask_b32_e32 v16, v17, v16, vcc_lo
	s_wait_loadcnt 0x8
	v_and_b32_e32 v50, 0xf0f0f0f, v22
	v_lshrrev_b32_e32 v51, 4, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v51, 0xf0f0f0f, v51
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v44, v50
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v46, v50
	v_cvt_f32_ubyte2_e32 v48, v50
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v45, v51
	v_cvt_f32_ubyte1_e32 v47, v51
	v_cvt_f32_ubyte2_e32 v49, v51
	v_cvt_f32_ubyte3_e32 v50, v50
	v_cvt_f32_ubyte3_e32 v51, v51
	v_fma_mix_f32 v44, v16, v44, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v16, v45, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v16, v46, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v16, v47, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v48, v16, v48, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v49, v16, v49, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v16, v50, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v16, v51, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x7
	v_cndmask_b32_e32 v18, v19, v18, vcc_lo
	s_wait_loadcnt 0x6
	v_and_b32_e32 v60, 0xf0f0f0f, v23
	v_lshrrev_b32_e32 v61, 4, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v61, 0xf0f0f0f, v61
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v54, v60
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v56, v60
	v_cvt_f32_ubyte2_e32 v58, v60
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v55, v61
	v_cvt_f32_ubyte1_e32 v57, v61
	v_cvt_f32_ubyte2_e32 v59, v61
	v_cvt_f32_ubyte3_e32 v60, v60
	v_cvt_f32_ubyte3_e32 v61, v61
	v_fma_mix_f32 v54, v18, v54, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v18, v55, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v18, v56, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v18, v57, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v58, v18, v58, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v59, v18, v59, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v18, v60, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v18, v61, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	buffer_load_b64 v[12:13], v8, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v20, v4, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[14:15], v9, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v21, v5, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[16:17], v10, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v22, v6, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[18:19], v11, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v23, v7, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	s_wait_loadcnt 0xd
	v_dual_mul_f32 v32, v24, v101 :: v_dual_mul_f32 v33, v35, v102
	v_dual_mul_f32 v42, v44, v101 :: v_dual_mul_f32 v43, v55, v102
	s_wait_loadcnt 0xb
	v_dual_mul_f32 v52, v24, v109 :: v_dual_mul_f32 v53, v35, v110
	v_dual_mul_f32 v62, v44, v109 :: v_dual_mul_f32 v63, v55, v110
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v25, v102 :: v_dual_fmac_f32 v33, v34, v101
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v45, v102 :: v_dual_fmac_f32 v43, v54, v101
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v25, v110 :: v_dual_fmac_f32 v53, v34, v109
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v45, v110 :: v_dual_fmac_f32 v63, v54, v109
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v26, v103 :: v_dual_fmac_f32 v33, v36, v103
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v46, v103 :: v_dual_fmac_f32 v43, v56, v103
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v26, v111 :: v_dual_fmac_f32 v53, v36, v111
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v46, v111 :: v_dual_fmac_f32 v63, v56, v111
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v27, v104 :: v_dual_fmac_f32 v33, v37, v104
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v47, v104 :: v_dual_fmac_f32 v43, v57, v104
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v27, v112 :: v_dual_fmac_f32 v53, v37, v112
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v47, v112 :: v_dual_fmac_f32 v63, v57, v112
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v28, v105 :: v_dual_fmac_f32 v33, v38, v105
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v48, v105 :: v_dual_fmac_f32 v43, v58, v105
	s_wait_loadcnt 0xa
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v28, v113 :: v_dual_fmac_f32 v53, v38, v113
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v48, v113 :: v_dual_fmac_f32 v63, v58, v113
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v29, v106 :: v_dual_fmac_f32 v33, v39, v106
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v49, v106 :: v_dual_fmac_f32 v43, v59, v106
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v29, v114 :: v_dual_fmac_f32 v53, v39, v114
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v49, v114 :: v_dual_fmac_f32 v63, v59, v114
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v30, v107 :: v_dual_fmac_f32 v33, v40, v107
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v50, v107 :: v_dual_fmac_f32 v43, v60, v107
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v30, v115 :: v_dual_fmac_f32 v53, v40, v115
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v50, v115 :: v_dual_fmac_f32 v63, v60, v115
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v31, v108 :: v_dual_fmac_f32 v33, v41, v108
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v51, v108 :: v_dual_fmac_f32 v43, v61, v108
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v31, v116 :: v_dual_fmac_f32 v53, v41, v116
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v51, v116 :: v_dual_fmac_f32 v63, v61, v116
	global_load_b128 v[101:104], v3, s[28:29]
	global_load_b128 v[105:108], v3, s[28:29] offset:16
	global_load_b128 v[109:112], v3, s[30:31]
	global_load_b128 v[113:116], v3, s[30:31] offset:16
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v68, v32, v68 :: v_dual_add_f32 v69, v69, v33
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v70, v42, v70 :: v_dual_add_f32 v71, v71, v43
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v80, v52, v80 :: v_dual_add_f32 v81, v81, v53
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v82, v62, v82 :: v_dual_add_f32 v83, v83, v63
	s_wait_loadcnt 0xd
	v_dual_mul_f32 v32, v24, v117 :: v_dual_mul_f32 v33, v35, v118
	v_dual_mul_f32 v42, v44, v117 :: v_dual_mul_f32 v43, v55, v118
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v25, v118 :: v_dual_fmac_f32 v33, v34, v117
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v45, v118 :: v_dual_fmac_f32 v43, v54, v117
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v26, v119 :: v_dual_fmac_f32 v33, v36, v119
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v46, v119 :: v_dual_fmac_f32 v43, v56, v119
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v27, v120 :: v_dual_fmac_f32 v33, v37, v120
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v47, v120 :: v_dual_fmac_f32 v43, v57, v120
	s_wait_loadcnt 0xc
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v28, v121 :: v_dual_fmac_f32 v33, v38, v121
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v48, v121 :: v_dual_fmac_f32 v43, v58, v121
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v29, v122 :: v_dual_fmac_f32 v33, v39, v122
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v49, v122 :: v_dual_fmac_f32 v43, v59, v122
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v30, v123 :: v_dual_fmac_f32 v33, v40, v123
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v50, v123 :: v_dual_fmac_f32 v43, v60, v123
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v31, v124 :: v_dual_fmac_f32 v33, v41, v124
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v51, v124 :: v_dual_fmac_f32 v43, v61, v124
	global_load_b128 v[117:120], v3, s[32:33]
	global_load_b128 v[121:124], v3, s[32:33] offset:16
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_add_f32 v92, v32, v92 :: v_dual_add_f32 v93, v93, v33
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_add_f32 v94, v42, v94 :: v_dual_add_f32 v95, v95, v43
	s_mov_b32 s15, s16
	v_mov_b32_e32 v2, v3
	s_add_co_i32 s2, s2, -1
	s_cmp_lg_u32 s2, 0
	s_cbranch_scc1 .Lrx_b3_s2
	s_wait_loadcnt 0x0
	.Lrx_b3_s2_done:
	s_add_co_i32 s2, s14, 0
	s_lshr_b32 s2, s2, 2
	s_cmp_eq_u32 s2, 0
	s_cbranch_scc1 .Lrx_b3_s3_done
	s_mov_b32 s15, 0x198
	v_lshlrev_b32_e32 v2, 5, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v2, 0xc00, v2
	buffer_load_b64 v[12:13], v8, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v20, v4, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[14:15], v9, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v21, v5, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[16:17], v10, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v22, v6, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[18:19], v11, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v23, v7, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	global_load_b128 v[101:104], v2, s[28:29]
	global_load_b128 v[105:108], v2, s[28:29] offset:16
	global_load_b128 v[109:112], v2, s[30:31]
	global_load_b128 v[113:116], v2, s[30:31] offset:16
	global_load_b128 v[117:120], v2, s[32:33]
	global_load_b128 v[121:124], v2, s[32:33] offset:16
	.Lrx_b3_s3:
	s_cmp_gt_u32 s2, 1
	s_cselect_b32 s17, 0x220, 0
	s_cselect_b32 s18, 0x1000, 0
	s_add_co_i32 s16, s15, s17
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v3, s18, v2
	v_cmp_gt_u32_e32 vcc_lo, 16, v0
	s_wait_loadcnt 0xd
	v_cndmask_b32_e32 v12, v13, v12, vcc_lo
	s_wait_loadcnt 0xc
	v_and_b32_e32 v30, 0xf0f0f0f, v20
	v_lshrrev_b32_e32 v31, 4, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v31, 0xf0f0f0f, v31
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v24, v30
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v26, v30
	v_cvt_f32_ubyte2_e32 v28, v30
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v25, v31
	v_cvt_f32_ubyte1_e32 v27, v31
	v_cvt_f32_ubyte2_e32 v29, v31
	v_cvt_f32_ubyte3_e32 v30, v30
	v_cvt_f32_ubyte3_e32 v31, v31
	v_fma_mix_f32 v24, v12, v24, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v25, v12, v25, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v26, v12, v26, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v27, v12, v27, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v28, v12, v28, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v29, v12, v29, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v30, v12, v30, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v31, v12, v31, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xb
	v_cndmask_b32_e32 v14, v15, v14, vcc_lo
	s_wait_loadcnt 0xa
	v_and_b32_e32 v40, 0xf0f0f0f, v21
	v_lshrrev_b32_e32 v41, 4, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v41, 0xf0f0f0f, v41
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v34, v40
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v36, v40
	v_cvt_f32_ubyte2_e32 v38, v40
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v35, v41
	v_cvt_f32_ubyte1_e32 v37, v41
	v_cvt_f32_ubyte2_e32 v39, v41
	v_cvt_f32_ubyte3_e32 v40, v40
	v_cvt_f32_ubyte3_e32 v41, v41
	v_fma_mix_f32 v34, v14, v34, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v35, v14, v35, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v36, v14, v36, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v37, v14, v37, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v38, v14, v38, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v39, v14, v39, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v40, v14, v40, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v41, v14, v41, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x9
	v_cndmask_b32_e32 v16, v17, v16, vcc_lo
	s_wait_loadcnt 0x8
	v_and_b32_e32 v50, 0xf0f0f0f, v22
	v_lshrrev_b32_e32 v51, 4, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v51, 0xf0f0f0f, v51
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v44, v50
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v46, v50
	v_cvt_f32_ubyte2_e32 v48, v50
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v45, v51
	v_cvt_f32_ubyte1_e32 v47, v51
	v_cvt_f32_ubyte2_e32 v49, v51
	v_cvt_f32_ubyte3_e32 v50, v50
	v_cvt_f32_ubyte3_e32 v51, v51
	v_fma_mix_f32 v44, v16, v44, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v16, v45, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v16, v46, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v16, v47, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v48, v16, v48, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v49, v16, v49, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v16, v50, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v16, v51, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x7
	v_cndmask_b32_e32 v18, v19, v18, vcc_lo
	s_wait_loadcnt 0x6
	v_and_b32_e32 v60, 0xf0f0f0f, v23
	v_lshrrev_b32_e32 v61, 4, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v61, 0xf0f0f0f, v61
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v54, v60
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v56, v60
	v_cvt_f32_ubyte2_e32 v58, v60
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v55, v61
	v_cvt_f32_ubyte1_e32 v57, v61
	v_cvt_f32_ubyte2_e32 v59, v61
	v_cvt_f32_ubyte3_e32 v60, v60
	v_cvt_f32_ubyte3_e32 v61, v61
	v_fma_mix_f32 v54, v18, v54, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v18, v55, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v18, v56, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v18, v57, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v58, v18, v58, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v59, v18, v59, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v18, v60, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v18, v61, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	buffer_load_b64 v[12:13], v8, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v20, v4, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[14:15], v9, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v21, v5, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[16:17], v10, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v22, v6, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[18:19], v11, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v23, v7, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	s_wait_loadcnt 0xd
	v_dual_mul_f32 v32, v24, v101 :: v_dual_mul_f32 v33, v35, v102
	v_dual_mul_f32 v42, v44, v101 :: v_dual_mul_f32 v43, v55, v102
	s_wait_loadcnt 0xb
	v_dual_mul_f32 v52, v24, v109 :: v_dual_mul_f32 v53, v35, v110
	v_dual_mul_f32 v62, v44, v109 :: v_dual_mul_f32 v63, v55, v110
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v25, v102 :: v_dual_fmac_f32 v33, v34, v101
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v45, v102 :: v_dual_fmac_f32 v43, v54, v101
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v25, v110 :: v_dual_fmac_f32 v53, v34, v109
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v45, v110 :: v_dual_fmac_f32 v63, v54, v109
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v26, v103 :: v_dual_fmac_f32 v33, v36, v103
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v46, v103 :: v_dual_fmac_f32 v43, v56, v103
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v26, v111 :: v_dual_fmac_f32 v53, v36, v111
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v46, v111 :: v_dual_fmac_f32 v63, v56, v111
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v27, v104 :: v_dual_fmac_f32 v33, v37, v104
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v47, v104 :: v_dual_fmac_f32 v43, v57, v104
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v27, v112 :: v_dual_fmac_f32 v53, v37, v112
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v47, v112 :: v_dual_fmac_f32 v63, v57, v112
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v28, v105 :: v_dual_fmac_f32 v33, v38, v105
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v48, v105 :: v_dual_fmac_f32 v43, v58, v105
	s_wait_loadcnt 0xa
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v28, v113 :: v_dual_fmac_f32 v53, v38, v113
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v48, v113 :: v_dual_fmac_f32 v63, v58, v113
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v29, v106 :: v_dual_fmac_f32 v33, v39, v106
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v49, v106 :: v_dual_fmac_f32 v43, v59, v106
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v29, v114 :: v_dual_fmac_f32 v53, v39, v114
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v49, v114 :: v_dual_fmac_f32 v63, v59, v114
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v30, v107 :: v_dual_fmac_f32 v33, v40, v107
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v50, v107 :: v_dual_fmac_f32 v43, v60, v107
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v30, v115 :: v_dual_fmac_f32 v53, v40, v115
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v50, v115 :: v_dual_fmac_f32 v63, v60, v115
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v31, v108 :: v_dual_fmac_f32 v33, v41, v108
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v51, v108 :: v_dual_fmac_f32 v43, v61, v108
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v31, v116 :: v_dual_fmac_f32 v53, v41, v116
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v51, v116 :: v_dual_fmac_f32 v63, v61, v116
	global_load_b128 v[101:104], v3, s[28:29]
	global_load_b128 v[105:108], v3, s[28:29] offset:16
	global_load_b128 v[109:112], v3, s[30:31]
	global_load_b128 v[113:116], v3, s[30:31] offset:16
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v72, v32, v72 :: v_dual_add_f32 v73, v73, v33
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v74, v42, v74 :: v_dual_add_f32 v75, v75, v43
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v84, v52, v84 :: v_dual_add_f32 v85, v85, v53
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v86, v62, v86 :: v_dual_add_f32 v87, v87, v63
	s_wait_loadcnt 0xd
	v_dual_mul_f32 v32, v24, v117 :: v_dual_mul_f32 v33, v35, v118
	v_dual_mul_f32 v42, v44, v117 :: v_dual_mul_f32 v43, v55, v118
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v25, v118 :: v_dual_fmac_f32 v33, v34, v117
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v45, v118 :: v_dual_fmac_f32 v43, v54, v117
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v26, v119 :: v_dual_fmac_f32 v33, v36, v119
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v46, v119 :: v_dual_fmac_f32 v43, v56, v119
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v27, v120 :: v_dual_fmac_f32 v33, v37, v120
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v47, v120 :: v_dual_fmac_f32 v43, v57, v120
	s_wait_loadcnt 0xc
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v28, v121 :: v_dual_fmac_f32 v33, v38, v121
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v48, v121 :: v_dual_fmac_f32 v43, v58, v121
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v29, v122 :: v_dual_fmac_f32 v33, v39, v122
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v49, v122 :: v_dual_fmac_f32 v43, v59, v122
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v30, v123 :: v_dual_fmac_f32 v33, v40, v123
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v50, v123 :: v_dual_fmac_f32 v43, v60, v123
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v31, v124 :: v_dual_fmac_f32 v33, v41, v124
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v51, v124 :: v_dual_fmac_f32 v43, v61, v124
	global_load_b128 v[117:120], v3, s[32:33]
	global_load_b128 v[121:124], v3, s[32:33] offset:16
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_add_f32 v96, v32, v96 :: v_dual_add_f32 v97, v97, v33
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_add_f32 v98, v42, v98 :: v_dual_add_f32 v99, v99, v43
	s_mov_b32 s15, s16
	v_mov_b32_e32 v2, v3
	s_add_co_i32 s2, s2, -1
	s_cmp_lg_u32 s2, 0
	s_cbranch_scc1 .Lrx_b3_s3
	s_wait_loadcnt 0x0
	.Lrx_b3_s3_done:
	v_dual_add_f32 v68, v68, v72 :: v_dual_add_f32 v69, v69, v73
	v_dual_add_f32 v70, v70, v74 :: v_dual_add_f32 v71, v71, v75
	v_dual_add_f32 v80, v80, v84 :: v_dual_add_f32 v81, v81, v85
	v_dual_add_f32 v82, v82, v86 :: v_dual_add_f32 v83, v83, v87
	v_dual_add_f32 v92, v92, v96 :: v_dual_add_f32 v93, v93, v97
	v_dual_add_f32 v94, v94, v98 :: v_dual_add_f32 v95, v95, v99
	v_dual_add_f32 v64, v64, v68 :: v_dual_add_f32 v65, v65, v69
	v_dual_add_f32 v66, v66, v70 :: v_dual_add_f32 v67, v67, v71
	v_dual_add_f32 v76, v76, v80 :: v_dual_add_f32 v77, v77, v81
	v_dual_add_f32 v78, v78, v82 :: v_dual_add_f32 v79, v79, v83
	v_dual_add_f32 v88, v88, v92 :: v_dual_add_f32 v89, v89, v93
	v_dual_add_f32 v90, v90, v94 :: v_dual_add_f32 v91, v91, v95
	ds_swizzle_b32 v24, v64 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v25, v65 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v26, v66 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v27, v67 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v28, v76 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v29, v77 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v30, v78 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v31, v79 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v32, v88 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v33, v89 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v34, v90 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v35, v91 offset:swizzle(BITMASK_PERM,"1pppp")
	s_wait_dscnt 0xb
	v_add_f32_e32 v64, v64, v24
	s_wait_dscnt 0xa
	v_add_f32_e32 v65, v65, v25
	s_wait_dscnt 0x9
	v_add_f32_e32 v66, v66, v26
	s_wait_dscnt 0x8
	v_add_f32_e32 v67, v67, v27
	s_wait_dscnt 0x7
	v_add_f32_e32 v76, v76, v28
	s_wait_dscnt 0x6
	v_add_f32_e32 v77, v77, v29
	s_wait_dscnt 0x5
	v_add_f32_e32 v78, v78, v30
	s_wait_dscnt 0x4
	v_add_f32_e32 v79, v79, v31
	s_wait_dscnt 0x3
	v_add_f32_e32 v88, v88, v32
	s_wait_dscnt 0x2
	v_add_f32_e32 v89, v89, v33
	s_wait_dscnt 0x1
	v_add_f32_e32 v90, v90, v34
	s_wait_dscnt 0x0
	v_add_f32_e32 v91, v91, v35
	v_cmp_gt_u32_e32 vcc_lo, 24, v0
	v_cndmask_b32_e64 v3, 0, 8, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v3, v3, v0, 2
	ds_bpermute_b32 v24, v3, v64
	ds_bpermute_b32 v25, v3, v65
	ds_bpermute_b32 v26, v3, v66
	ds_bpermute_b32 v27, v3, v67
	ds_bpermute_b32 v28, v3, v76
	ds_bpermute_b32 v29, v3, v77
	ds_bpermute_b32 v30, v3, v78
	ds_bpermute_b32 v31, v3, v79
	ds_bpermute_b32 v32, v3, v88
	ds_bpermute_b32 v33, v3, v89
	ds_bpermute_b32 v34, v3, v90
	ds_bpermute_b32 v35, v3, v91
	s_wait_dscnt 0xb
	v_add_f32_e32 v64, v64, v24
	s_wait_dscnt 0xa
	v_add_f32_e32 v65, v65, v25
	s_wait_dscnt 0x9
	v_add_f32_e32 v66, v66, v26
	s_wait_dscnt 0x8
	v_add_f32_e32 v67, v67, v27
	s_wait_dscnt 0x7
	v_add_f32_e32 v76, v76, v28
	s_wait_dscnt 0x6
	v_add_f32_e32 v77, v77, v29
	s_wait_dscnt 0x5
	v_add_f32_e32 v78, v78, v30
	s_wait_dscnt 0x4
	v_add_f32_e32 v79, v79, v31
	s_wait_dscnt 0x3
	v_add_f32_e32 v88, v88, v32
	s_wait_dscnt 0x2
	v_add_f32_e32 v89, v89, v33
	s_wait_dscnt 0x1
	v_add_f32_e32 v90, v90, v34
	s_wait_dscnt 0x0
	v_add_f32_e32 v91, v91, v35
	v_cmp_gt_u32_e32 vcc_lo, 28, v0
	v_cndmask_b32_e64 v3, 0, 4, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v3, v3, v0, 2
	ds_bpermute_b32 v24, v3, v64
	ds_bpermute_b32 v25, v3, v65
	ds_bpermute_b32 v26, v3, v66
	ds_bpermute_b32 v27, v3, v67
	ds_bpermute_b32 v28, v3, v76
	ds_bpermute_b32 v29, v3, v77
	ds_bpermute_b32 v30, v3, v78
	ds_bpermute_b32 v31, v3, v79
	ds_bpermute_b32 v32, v3, v88
	ds_bpermute_b32 v33, v3, v89
	ds_bpermute_b32 v34, v3, v90
	ds_bpermute_b32 v35, v3, v91
	s_wait_dscnt 0xb
	v_add_f32_e32 v64, v64, v24
	s_wait_dscnt 0xa
	v_add_f32_e32 v65, v65, v25
	s_wait_dscnt 0x9
	v_add_f32_e32 v66, v66, v26
	s_wait_dscnt 0x8
	v_add_f32_e32 v67, v67, v27
	s_wait_dscnt 0x7
	v_add_f32_e32 v76, v76, v28
	s_wait_dscnt 0x6
	v_add_f32_e32 v77, v77, v29
	s_wait_dscnt 0x5
	v_add_f32_e32 v78, v78, v30
	s_wait_dscnt 0x4
	v_add_f32_e32 v79, v79, v31
	s_wait_dscnt 0x3
	v_add_f32_e32 v88, v88, v32
	s_wait_dscnt 0x2
	v_add_f32_e32 v89, v89, v33
	s_wait_dscnt 0x1
	v_add_f32_e32 v90, v90, v34
	s_wait_dscnt 0x0
	v_add_f32_e32 v91, v91, v35
	v_cmp_gt_u32_e32 vcc_lo, 30, v0
	v_cndmask_b32_e64 v3, 0, 2, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v3, v3, v0, 2
	ds_bpermute_b32 v24, v3, v64
	ds_bpermute_b32 v25, v3, v65
	ds_bpermute_b32 v26, v3, v66
	ds_bpermute_b32 v27, v3, v67
	ds_bpermute_b32 v28, v3, v76
	ds_bpermute_b32 v29, v3, v77
	ds_bpermute_b32 v30, v3, v78
	ds_bpermute_b32 v31, v3, v79
	ds_bpermute_b32 v32, v3, v88
	ds_bpermute_b32 v33, v3, v89
	ds_bpermute_b32 v34, v3, v90
	ds_bpermute_b32 v35, v3, v91
	s_wait_dscnt 0xb
	v_add_f32_e32 v64, v64, v24
	s_wait_dscnt 0xa
	v_add_f32_e32 v65, v65, v25
	s_wait_dscnt 0x9
	v_add_f32_e32 v66, v66, v26
	s_wait_dscnt 0x8
	v_add_f32_e32 v67, v67, v27
	s_wait_dscnt 0x7
	v_add_f32_e32 v76, v76, v28
	s_wait_dscnt 0x6
	v_add_f32_e32 v77, v77, v29
	s_wait_dscnt 0x5
	v_add_f32_e32 v78, v78, v30
	s_wait_dscnt 0x4
	v_add_f32_e32 v79, v79, v31
	s_wait_dscnt 0x3
	v_add_f32_e32 v88, v88, v32
	s_wait_dscnt 0x2
	v_add_f32_e32 v89, v89, v33
	s_wait_dscnt 0x1
	v_add_f32_e32 v90, v90, v34
	s_wait_dscnt 0x0
	v_add_f32_e32 v91, v91, v35
	v_cmp_gt_u32_e32 vcc_lo, 31, v0
	v_cndmask_b32_e64 v3, 0, 1, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v3, v3, v0, 2
	ds_bpermute_b32 v24, v3, v64
	ds_bpermute_b32 v25, v3, v65
	ds_bpermute_b32 v26, v3, v66
	ds_bpermute_b32 v27, v3, v67
	ds_bpermute_b32 v28, v3, v76
	ds_bpermute_b32 v29, v3, v77
	ds_bpermute_b32 v30, v3, v78
	ds_bpermute_b32 v31, v3, v79
	ds_bpermute_b32 v32, v3, v88
	ds_bpermute_b32 v33, v3, v89
	ds_bpermute_b32 v34, v3, v90
	ds_bpermute_b32 v35, v3, v91
	s_wait_dscnt 0xb
	v_add_f32_e32 v64, v64, v24
	s_wait_dscnt 0xa
	v_add_f32_e32 v65, v65, v25
	s_wait_dscnt 0x9
	v_add_f32_e32 v66, v66, v26
	s_wait_dscnt 0x8
	v_add_f32_e32 v67, v67, v27
	s_wait_dscnt 0x7
	v_add_f32_e32 v76, v76, v28
	s_wait_dscnt 0x6
	v_add_f32_e32 v77, v77, v29
	s_wait_dscnt 0x5
	v_add_f32_e32 v78, v78, v30
	s_wait_dscnt 0x4
	v_add_f32_e32 v79, v79, v31
	s_wait_dscnt 0x3
	v_add_f32_e32 v88, v88, v32
	s_wait_dscnt 0x2
	v_add_f32_e32 v89, v89, v33
	s_wait_dscnt 0x1
	v_add_f32_e32 v90, v90, v34
	s_wait_dscnt 0x0
	v_add_f32_e32 v91, v91, v35
	v_cmpx_eq_u32_e32 0, v0
	s_lshl_b32 s46, s12, 2
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v48, s46
	s_mul_i32 s47, s45, 1
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s47, s47, s46
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v49, s47
	s_mul_i32 s47, s45, 2
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s47, s47, s46
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v50, s47
	s_add_co_i32 s13, s12, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cbranch_scc0 .Lrx_b3_p0_single
	global_load_b32 v36, v48, s[8:9]
	global_load_b32 v37, v48, s[8:9] offset:4
	global_load_b32 v40, v49, s[8:9]
	global_load_b32 v41, v49, s[8:9] offset:4
	global_load_b32 v44, v50, s[8:9]
	global_load_b32 v45, v50, s[8:9] offset:4
	s_wait_loadcnt 0x5
	v_add_f32_e32 v36, v64, v36
	global_store_b32 v48, v36, s[8:9]
	s_wait_loadcnt 0x4
	v_add_f32_e32 v37, v65, v37
	global_store_b32 v48, v37, s[8:9] offset:4
	s_wait_loadcnt 0x3
	v_add_f32_e32 v40, v76, v40
	global_store_b32 v49, v40, s[8:9]
	s_wait_loadcnt 0x2
	v_add_f32_e32 v41, v77, v41
	global_store_b32 v49, v41, s[8:9] offset:4
	s_wait_loadcnt 0x1
	v_add_f32_e32 v44, v88, v44
	global_store_b32 v50, v44, s[8:9]
	s_wait_loadcnt 0x0
	v_add_f32_e32 v45, v89, v45
	global_store_b32 v50, v45, s[8:9] offset:4
	s_wait_storecnt 0x0
	s_branch .Lrx_b3_p0_next
	.Lrx_b3_p0_single:
	global_load_b32 v36, v48, s[8:9]
	global_load_b32 v40, v49, s[8:9]
	global_load_b32 v44, v50, s[8:9]
	s_wait_loadcnt 0x2
	v_add_f32_e32 v36, v36, v64
	global_store_b32 v48, v36, s[8:9]
	s_wait_loadcnt 0x1
	v_add_f32_e32 v40, v40, v76
	global_store_b32 v49, v40, s[8:9]
	s_wait_loadcnt 0x0
	v_add_f32_e32 v44, v44, v88
	global_store_b32 v50, v44, s[8:9]
	s_wait_storecnt 0x0
	s_branch .Lrx_b3_stored
	.Lrx_b3_p0_next:
	s_add_co_i32 s13, s12, 2
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cbranch_scc0 .Lrx_b3_stored
	s_add_co_i32 s13, s12, 3
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cbranch_scc0 .Lrx_b3_p1_single
	global_load_b32 v38, v48, s[8:9] offset:8
	global_load_b32 v39, v48, s[8:9] offset:12
	global_load_b32 v42, v49, s[8:9] offset:8
	global_load_b32 v43, v49, s[8:9] offset:12
	global_load_b32 v46, v50, s[8:9] offset:8
	global_load_b32 v47, v50, s[8:9] offset:12
	s_wait_loadcnt 0x5
	v_add_f32_e32 v38, v66, v38
	global_store_b32 v48, v38, s[8:9] offset:8
	s_wait_loadcnt 0x4
	v_add_f32_e32 v39, v67, v39
	global_store_b32 v48, v39, s[8:9] offset:12
	s_wait_loadcnt 0x3
	v_add_f32_e32 v42, v78, v42
	global_store_b32 v49, v42, s[8:9] offset:8
	s_wait_loadcnt 0x2
	v_add_f32_e32 v43, v79, v43
	global_store_b32 v49, v43, s[8:9] offset:12
	s_wait_loadcnt 0x1
	v_add_f32_e32 v46, v90, v46
	global_store_b32 v50, v46, s[8:9] offset:8
	s_wait_loadcnt 0x0
	v_add_f32_e32 v47, v91, v47
	global_store_b32 v50, v47, s[8:9] offset:12
	s_wait_storecnt 0x0
	s_branch .Lrx_b3_stored
	.Lrx_b3_p1_single:
	global_load_b32 v38, v48, s[8:9] offset:8
	global_load_b32 v42, v49, s[8:9] offset:8
	global_load_b32 v46, v50, s[8:9] offset:8
	s_wait_loadcnt 0x2
	v_add_f32_e32 v38, v38, v66
	global_store_b32 v48, v38, s[8:9] offset:8
	s_wait_loadcnt 0x1
	v_add_f32_e32 v42, v42, v78
	global_store_b32 v49, v42, s[8:9] offset:8
	s_wait_loadcnt 0x0
	v_add_f32_e32 v46, v46, v90
	global_store_b32 v50, v46, s[8:9] offset:8
	s_wait_storecnt 0x0
	s_branch .Lrx_b3_stored
	.Lrx_b3_stored:
	s_branch .Lrx_end
	.Lrx_b4:
	s_mul_i32 s12, ttmp9, 4
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_i32 s12, s10
	s_cbranch_scc1 .Lrx_end
	s_mov_b32 s13, s12
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s13, s13, s19
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v8, s13
	v_add_nc_u32_e32 v4, s13, v1
	s_add_co_i32 s13, s12, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cselect_b32 s13, s13, s12
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s13, s13, s19
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v9, s13
	v_add_nc_u32_e32 v5, s13, v1
	s_add_co_i32 s13, s12, 2
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cselect_b32 s13, s13, s12
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s13, s13, s19
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v10, s13
	v_add_nc_u32_e32 v6, s13, v1
	s_add_co_i32 s13, s12, 3
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cselect_b32 s13, s13, s12
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s13, s13, s19
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v11, s13
	v_add_nc_u32_e32 v7, s13, v1
	v_mov_b32_e32 v64, 0
	v_mov_b32_e32 v65, 0
	v_mov_b32_e32 v66, 0
	v_mov_b32_e32 v67, 0
	v_mov_b32_e32 v68, 0
	v_mov_b32_e32 v69, 0
	v_mov_b32_e32 v70, 0
	v_mov_b32_e32 v71, 0
	v_mov_b32_e32 v72, 0
	v_mov_b32_e32 v73, 0
	v_mov_b32_e32 v74, 0
	v_mov_b32_e32 v75, 0
	v_mov_b32_e32 v76, 0
	v_mov_b32_e32 v77, 0
	v_mov_b32_e32 v78, 0
	v_mov_b32_e32 v79, 0
	v_mov_b32_e32 v80, 0
	v_mov_b32_e32 v81, 0
	v_mov_b32_e32 v82, 0
	v_mov_b32_e32 v83, 0
	v_mov_b32_e32 v84, 0
	v_mov_b32_e32 v85, 0
	v_mov_b32_e32 v86, 0
	v_mov_b32_e32 v87, 0
	v_mov_b32_e32 v88, 0
	v_mov_b32_e32 v89, 0
	v_mov_b32_e32 v90, 0
	v_mov_b32_e32 v91, 0
	v_mov_b32_e32 v92, 0
	v_mov_b32_e32 v93, 0
	v_mov_b32_e32 v94, 0
	v_mov_b32_e32 v95, 0
	v_mov_b32_e32 v96, 0
	v_mov_b32_e32 v97, 0
	v_mov_b32_e32 v98, 0
	v_mov_b32_e32 v99, 0
	v_mov_b32_e32 v100, 0
	v_mov_b32_e32 v101, 0
	v_mov_b32_e32 v102, 0
	v_mov_b32_e32 v103, 0
	v_mov_b32_e32 v104, 0
	v_mov_b32_e32 v105, 0
	v_mov_b32_e32 v106, 0
	v_mov_b32_e32 v107, 0
	v_mov_b32_e32 v108, 0
	v_mov_b32_e32 v109, 0
	v_mov_b32_e32 v110, 0
	v_mov_b32_e32 v111, 0
	s_add_co_i32 s2, s14, 3
	s_lshr_b32 s2, s2, 2
	s_cmp_eq_u32 s2, 0
	s_cbranch_scc1 .Lrx_b4_s0_done
	s_mov_b32 s15, 0
	v_lshlrev_b32_e32 v2, 5, v0
	buffer_load_b64 v[12:13], v8, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v20, v4, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[14:15], v9, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v21, v5, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[16:17], v10, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v22, v6, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[18:19], v11, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v23, v7, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	global_load_b128 v[113:116], v2, s[28:29]
	global_load_b128 v[117:120], v2, s[28:29] offset:16
	global_load_b128 v[121:124], v2, s[30:31]
	global_load_b128 v[125:128], v2, s[30:31] offset:16
	global_load_b128 v[129:132], v2, s[32:33]
	global_load_b128 v[133:136], v2, s[32:33] offset:16
	global_load_b128 v[137:140], v2, s[34:35]
	global_load_b128 v[141:144], v2, s[34:35] offset:16
	.Lrx_b4_s0:
	s_cmp_gt_u32 s2, 1
	s_cselect_b32 s17, 0x220, 0
	s_cselect_b32 s18, 0x1000, 0
	s_add_co_i32 s16, s15, s17
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v3, s18, v2
	v_cmp_gt_u32_e32 vcc_lo, 16, v0
	s_wait_loadcnt 0xf
	v_cndmask_b32_e32 v12, v13, v12, vcc_lo
	s_wait_loadcnt 0xe
	v_and_b32_e32 v30, 0xf0f0f0f, v20
	v_lshrrev_b32_e32 v31, 4, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v31, 0xf0f0f0f, v31
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v24, v30
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v26, v30
	v_cvt_f32_ubyte2_e32 v28, v30
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v25, v31
	v_cvt_f32_ubyte1_e32 v27, v31
	v_cvt_f32_ubyte2_e32 v29, v31
	v_cvt_f32_ubyte3_e32 v30, v30
	v_cvt_f32_ubyte3_e32 v31, v31
	v_fma_mix_f32 v24, v12, v24, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v25, v12, v25, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v26, v12, v26, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v27, v12, v27, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v28, v12, v28, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v29, v12, v29, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v30, v12, v30, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v31, v12, v31, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xd
	v_cndmask_b32_e32 v14, v15, v14, vcc_lo
	s_wait_loadcnt 0xc
	v_and_b32_e32 v40, 0xf0f0f0f, v21
	v_lshrrev_b32_e32 v41, 4, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v41, 0xf0f0f0f, v41
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v34, v40
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v36, v40
	v_cvt_f32_ubyte2_e32 v38, v40
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v35, v41
	v_cvt_f32_ubyte1_e32 v37, v41
	v_cvt_f32_ubyte2_e32 v39, v41
	v_cvt_f32_ubyte3_e32 v40, v40
	v_cvt_f32_ubyte3_e32 v41, v41
	v_fma_mix_f32 v34, v14, v34, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v35, v14, v35, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v36, v14, v36, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v37, v14, v37, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v38, v14, v38, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v39, v14, v39, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v40, v14, v40, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v41, v14, v41, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xb
	v_cndmask_b32_e32 v16, v17, v16, vcc_lo
	s_wait_loadcnt 0xa
	v_and_b32_e32 v50, 0xf0f0f0f, v22
	v_lshrrev_b32_e32 v51, 4, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v51, 0xf0f0f0f, v51
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v44, v50
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v46, v50
	v_cvt_f32_ubyte2_e32 v48, v50
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v45, v51
	v_cvt_f32_ubyte1_e32 v47, v51
	v_cvt_f32_ubyte2_e32 v49, v51
	v_cvt_f32_ubyte3_e32 v50, v50
	v_cvt_f32_ubyte3_e32 v51, v51
	v_fma_mix_f32 v44, v16, v44, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v16, v45, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v16, v46, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v16, v47, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v48, v16, v48, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v49, v16, v49, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v16, v50, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v16, v51, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x9
	v_cndmask_b32_e32 v18, v19, v18, vcc_lo
	s_wait_loadcnt 0x8
	v_and_b32_e32 v60, 0xf0f0f0f, v23
	v_lshrrev_b32_e32 v61, 4, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v61, 0xf0f0f0f, v61
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v54, v60
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v56, v60
	v_cvt_f32_ubyte2_e32 v58, v60
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v55, v61
	v_cvt_f32_ubyte1_e32 v57, v61
	v_cvt_f32_ubyte2_e32 v59, v61
	v_cvt_f32_ubyte3_e32 v60, v60
	v_cvt_f32_ubyte3_e32 v61, v61
	v_fma_mix_f32 v54, v18, v54, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v18, v55, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v18, v56, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v18, v57, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v58, v18, v58, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v59, v18, v59, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v18, v60, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v18, v61, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	buffer_load_b64 v[12:13], v8, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v20, v4, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[14:15], v9, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v21, v5, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[16:17], v10, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v22, v6, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[18:19], v11, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v23, v7, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	s_wait_loadcnt 0xf
	v_dual_mul_f32 v32, v24, v113 :: v_dual_mul_f32 v33, v35, v114
	v_dual_mul_f32 v42, v44, v113 :: v_dual_mul_f32 v43, v55, v114
	s_wait_loadcnt 0xd
	v_dual_mul_f32 v52, v24, v121 :: v_dual_mul_f32 v53, v35, v122
	v_dual_mul_f32 v62, v44, v121 :: v_dual_mul_f32 v63, v55, v122
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v25, v114 :: v_dual_fmac_f32 v33, v34, v113
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v45, v114 :: v_dual_fmac_f32 v43, v54, v113
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v25, v122 :: v_dual_fmac_f32 v53, v34, v121
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v45, v122 :: v_dual_fmac_f32 v63, v54, v121
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v26, v115 :: v_dual_fmac_f32 v33, v36, v115
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v46, v115 :: v_dual_fmac_f32 v43, v56, v115
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v26, v123 :: v_dual_fmac_f32 v53, v36, v123
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v46, v123 :: v_dual_fmac_f32 v63, v56, v123
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v27, v116 :: v_dual_fmac_f32 v33, v37, v116
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v47, v116 :: v_dual_fmac_f32 v43, v57, v116
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v27, v124 :: v_dual_fmac_f32 v53, v37, v124
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v47, v124 :: v_dual_fmac_f32 v63, v57, v124
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v28, v117 :: v_dual_fmac_f32 v33, v38, v117
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v48, v117 :: v_dual_fmac_f32 v43, v58, v117
	s_wait_loadcnt 0xc
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v28, v125 :: v_dual_fmac_f32 v53, v38, v125
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v48, v125 :: v_dual_fmac_f32 v63, v58, v125
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v29, v118 :: v_dual_fmac_f32 v33, v39, v118
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v49, v118 :: v_dual_fmac_f32 v43, v59, v118
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v29, v126 :: v_dual_fmac_f32 v53, v39, v126
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v49, v126 :: v_dual_fmac_f32 v63, v59, v126
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v30, v119 :: v_dual_fmac_f32 v33, v40, v119
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v50, v119 :: v_dual_fmac_f32 v43, v60, v119
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v30, v127 :: v_dual_fmac_f32 v53, v40, v127
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v50, v127 :: v_dual_fmac_f32 v63, v60, v127
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v31, v120 :: v_dual_fmac_f32 v33, v41, v120
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v51, v120 :: v_dual_fmac_f32 v43, v61, v120
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v31, v128 :: v_dual_fmac_f32 v53, v41, v128
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v51, v128 :: v_dual_fmac_f32 v63, v61, v128
	global_load_b128 v[113:116], v3, s[28:29]
	global_load_b128 v[117:120], v3, s[28:29] offset:16
	global_load_b128 v[121:124], v3, s[30:31]
	global_load_b128 v[125:128], v3, s[30:31] offset:16
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v64, v32, v64 :: v_dual_add_f32 v65, v65, v33
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v66, v42, v66 :: v_dual_add_f32 v67, v67, v43
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v76, v52, v76 :: v_dual_add_f32 v77, v77, v53
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v78, v62, v78 :: v_dual_add_f32 v79, v79, v63
	s_wait_loadcnt 0xf
	v_dual_mul_f32 v32, v24, v129 :: v_dual_mul_f32 v33, v35, v130
	v_dual_mul_f32 v42, v44, v129 :: v_dual_mul_f32 v43, v55, v130
	s_wait_loadcnt 0xd
	v_dual_mul_f32 v52, v24, v137 :: v_dual_mul_f32 v53, v35, v138
	v_dual_mul_f32 v62, v44, v137 :: v_dual_mul_f32 v63, v55, v138
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v25, v130 :: v_dual_fmac_f32 v33, v34, v129
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v45, v130 :: v_dual_fmac_f32 v43, v54, v129
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v25, v138 :: v_dual_fmac_f32 v53, v34, v137
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v45, v138 :: v_dual_fmac_f32 v63, v54, v137
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v26, v131 :: v_dual_fmac_f32 v33, v36, v131
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v46, v131 :: v_dual_fmac_f32 v43, v56, v131
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v26, v139 :: v_dual_fmac_f32 v53, v36, v139
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v46, v139 :: v_dual_fmac_f32 v63, v56, v139
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v27, v132 :: v_dual_fmac_f32 v33, v37, v132
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v47, v132 :: v_dual_fmac_f32 v43, v57, v132
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v27, v140 :: v_dual_fmac_f32 v53, v37, v140
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v47, v140 :: v_dual_fmac_f32 v63, v57, v140
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v28, v133 :: v_dual_fmac_f32 v33, v38, v133
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v48, v133 :: v_dual_fmac_f32 v43, v58, v133
	s_wait_loadcnt 0xc
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v28, v141 :: v_dual_fmac_f32 v53, v38, v141
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v48, v141 :: v_dual_fmac_f32 v63, v58, v141
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v29, v134 :: v_dual_fmac_f32 v33, v39, v134
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v49, v134 :: v_dual_fmac_f32 v43, v59, v134
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v29, v142 :: v_dual_fmac_f32 v53, v39, v142
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v49, v142 :: v_dual_fmac_f32 v63, v59, v142
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v30, v135 :: v_dual_fmac_f32 v33, v40, v135
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v50, v135 :: v_dual_fmac_f32 v43, v60, v135
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v30, v143 :: v_dual_fmac_f32 v53, v40, v143
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v50, v143 :: v_dual_fmac_f32 v63, v60, v143
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v31, v136 :: v_dual_fmac_f32 v33, v41, v136
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v51, v136 :: v_dual_fmac_f32 v43, v61, v136
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v31, v144 :: v_dual_fmac_f32 v53, v41, v144
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v51, v144 :: v_dual_fmac_f32 v63, v61, v144
	global_load_b128 v[129:132], v3, s[32:33]
	global_load_b128 v[133:136], v3, s[32:33] offset:16
	global_load_b128 v[137:140], v3, s[34:35]
	global_load_b128 v[141:144], v3, s[34:35] offset:16
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v88, v32, v88 :: v_dual_add_f32 v89, v89, v33
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v90, v42, v90 :: v_dual_add_f32 v91, v91, v43
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v100, v52, v100 :: v_dual_add_f32 v101, v101, v53
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v102, v62, v102 :: v_dual_add_f32 v103, v103, v63
	s_mov_b32 s15, s16
	v_mov_b32_e32 v2, v3
	s_add_co_i32 s2, s2, -1
	s_cmp_lg_u32 s2, 0
	s_cbranch_scc1 .Lrx_b4_s0
	s_wait_loadcnt 0x0
	.Lrx_b4_s0_done:
	s_add_co_i32 s2, s14, 2
	s_lshr_b32 s2, s2, 2
	s_cmp_eq_u32 s2, 0
	s_cbranch_scc1 .Lrx_b4_s1_done
	s_mov_b32 s15, 0x88
	v_lshlrev_b32_e32 v2, 5, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v2, 0x400, v2
	buffer_load_b64 v[12:13], v8, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v20, v4, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[14:15], v9, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v21, v5, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[16:17], v10, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v22, v6, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[18:19], v11, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v23, v7, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	global_load_b128 v[113:116], v2, s[28:29]
	global_load_b128 v[117:120], v2, s[28:29] offset:16
	global_load_b128 v[121:124], v2, s[30:31]
	global_load_b128 v[125:128], v2, s[30:31] offset:16
	global_load_b128 v[129:132], v2, s[32:33]
	global_load_b128 v[133:136], v2, s[32:33] offset:16
	global_load_b128 v[137:140], v2, s[34:35]
	global_load_b128 v[141:144], v2, s[34:35] offset:16
	.Lrx_b4_s1:
	s_cmp_gt_u32 s2, 1
	s_cselect_b32 s17, 0x220, 0
	s_cselect_b32 s18, 0x1000, 0
	s_add_co_i32 s16, s15, s17
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v3, s18, v2
	v_cmp_gt_u32_e32 vcc_lo, 16, v0
	s_wait_loadcnt 0xf
	v_cndmask_b32_e32 v12, v13, v12, vcc_lo
	s_wait_loadcnt 0xe
	v_and_b32_e32 v30, 0xf0f0f0f, v20
	v_lshrrev_b32_e32 v31, 4, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v31, 0xf0f0f0f, v31
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v24, v30
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v26, v30
	v_cvt_f32_ubyte2_e32 v28, v30
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v25, v31
	v_cvt_f32_ubyte1_e32 v27, v31
	v_cvt_f32_ubyte2_e32 v29, v31
	v_cvt_f32_ubyte3_e32 v30, v30
	v_cvt_f32_ubyte3_e32 v31, v31
	v_fma_mix_f32 v24, v12, v24, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v25, v12, v25, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v26, v12, v26, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v27, v12, v27, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v28, v12, v28, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v29, v12, v29, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v30, v12, v30, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v31, v12, v31, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xd
	v_cndmask_b32_e32 v14, v15, v14, vcc_lo
	s_wait_loadcnt 0xc
	v_and_b32_e32 v40, 0xf0f0f0f, v21
	v_lshrrev_b32_e32 v41, 4, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v41, 0xf0f0f0f, v41
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v34, v40
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v36, v40
	v_cvt_f32_ubyte2_e32 v38, v40
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v35, v41
	v_cvt_f32_ubyte1_e32 v37, v41
	v_cvt_f32_ubyte2_e32 v39, v41
	v_cvt_f32_ubyte3_e32 v40, v40
	v_cvt_f32_ubyte3_e32 v41, v41
	v_fma_mix_f32 v34, v14, v34, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v35, v14, v35, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v36, v14, v36, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v37, v14, v37, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v38, v14, v38, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v39, v14, v39, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v40, v14, v40, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v41, v14, v41, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xb
	v_cndmask_b32_e32 v16, v17, v16, vcc_lo
	s_wait_loadcnt 0xa
	v_and_b32_e32 v50, 0xf0f0f0f, v22
	v_lshrrev_b32_e32 v51, 4, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v51, 0xf0f0f0f, v51
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v44, v50
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v46, v50
	v_cvt_f32_ubyte2_e32 v48, v50
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v45, v51
	v_cvt_f32_ubyte1_e32 v47, v51
	v_cvt_f32_ubyte2_e32 v49, v51
	v_cvt_f32_ubyte3_e32 v50, v50
	v_cvt_f32_ubyte3_e32 v51, v51
	v_fma_mix_f32 v44, v16, v44, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v16, v45, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v16, v46, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v16, v47, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v48, v16, v48, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v49, v16, v49, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v16, v50, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v16, v51, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x9
	v_cndmask_b32_e32 v18, v19, v18, vcc_lo
	s_wait_loadcnt 0x8
	v_and_b32_e32 v60, 0xf0f0f0f, v23
	v_lshrrev_b32_e32 v61, 4, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v61, 0xf0f0f0f, v61
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v54, v60
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v56, v60
	v_cvt_f32_ubyte2_e32 v58, v60
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v55, v61
	v_cvt_f32_ubyte1_e32 v57, v61
	v_cvt_f32_ubyte2_e32 v59, v61
	v_cvt_f32_ubyte3_e32 v60, v60
	v_cvt_f32_ubyte3_e32 v61, v61
	v_fma_mix_f32 v54, v18, v54, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v18, v55, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v18, v56, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v18, v57, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v58, v18, v58, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v59, v18, v59, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v18, v60, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v18, v61, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	buffer_load_b64 v[12:13], v8, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v20, v4, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[14:15], v9, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v21, v5, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[16:17], v10, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v22, v6, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[18:19], v11, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v23, v7, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	s_wait_loadcnt 0xf
	v_dual_mul_f32 v32, v24, v113 :: v_dual_mul_f32 v33, v35, v114
	v_dual_mul_f32 v42, v44, v113 :: v_dual_mul_f32 v43, v55, v114
	s_wait_loadcnt 0xd
	v_dual_mul_f32 v52, v24, v121 :: v_dual_mul_f32 v53, v35, v122
	v_dual_mul_f32 v62, v44, v121 :: v_dual_mul_f32 v63, v55, v122
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v25, v114 :: v_dual_fmac_f32 v33, v34, v113
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v45, v114 :: v_dual_fmac_f32 v43, v54, v113
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v25, v122 :: v_dual_fmac_f32 v53, v34, v121
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v45, v122 :: v_dual_fmac_f32 v63, v54, v121
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v26, v115 :: v_dual_fmac_f32 v33, v36, v115
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v46, v115 :: v_dual_fmac_f32 v43, v56, v115
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v26, v123 :: v_dual_fmac_f32 v53, v36, v123
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v46, v123 :: v_dual_fmac_f32 v63, v56, v123
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v27, v116 :: v_dual_fmac_f32 v33, v37, v116
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v47, v116 :: v_dual_fmac_f32 v43, v57, v116
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v27, v124 :: v_dual_fmac_f32 v53, v37, v124
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v47, v124 :: v_dual_fmac_f32 v63, v57, v124
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v28, v117 :: v_dual_fmac_f32 v33, v38, v117
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v48, v117 :: v_dual_fmac_f32 v43, v58, v117
	s_wait_loadcnt 0xc
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v28, v125 :: v_dual_fmac_f32 v53, v38, v125
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v48, v125 :: v_dual_fmac_f32 v63, v58, v125
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v29, v118 :: v_dual_fmac_f32 v33, v39, v118
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v49, v118 :: v_dual_fmac_f32 v43, v59, v118
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v29, v126 :: v_dual_fmac_f32 v53, v39, v126
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v49, v126 :: v_dual_fmac_f32 v63, v59, v126
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v30, v119 :: v_dual_fmac_f32 v33, v40, v119
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v50, v119 :: v_dual_fmac_f32 v43, v60, v119
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v30, v127 :: v_dual_fmac_f32 v53, v40, v127
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v50, v127 :: v_dual_fmac_f32 v63, v60, v127
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v31, v120 :: v_dual_fmac_f32 v33, v41, v120
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v51, v120 :: v_dual_fmac_f32 v43, v61, v120
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v31, v128 :: v_dual_fmac_f32 v53, v41, v128
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v51, v128 :: v_dual_fmac_f32 v63, v61, v128
	global_load_b128 v[113:116], v3, s[28:29]
	global_load_b128 v[117:120], v3, s[28:29] offset:16
	global_load_b128 v[121:124], v3, s[30:31]
	global_load_b128 v[125:128], v3, s[30:31] offset:16
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v68, v32, v68 :: v_dual_add_f32 v69, v69, v33
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v70, v42, v70 :: v_dual_add_f32 v71, v71, v43
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v80, v52, v80 :: v_dual_add_f32 v81, v81, v53
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v82, v62, v82 :: v_dual_add_f32 v83, v83, v63
	s_wait_loadcnt 0xf
	v_dual_mul_f32 v32, v24, v129 :: v_dual_mul_f32 v33, v35, v130
	v_dual_mul_f32 v42, v44, v129 :: v_dual_mul_f32 v43, v55, v130
	s_wait_loadcnt 0xd
	v_dual_mul_f32 v52, v24, v137 :: v_dual_mul_f32 v53, v35, v138
	v_dual_mul_f32 v62, v44, v137 :: v_dual_mul_f32 v63, v55, v138
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v25, v130 :: v_dual_fmac_f32 v33, v34, v129
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v45, v130 :: v_dual_fmac_f32 v43, v54, v129
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v25, v138 :: v_dual_fmac_f32 v53, v34, v137
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v45, v138 :: v_dual_fmac_f32 v63, v54, v137
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v26, v131 :: v_dual_fmac_f32 v33, v36, v131
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v46, v131 :: v_dual_fmac_f32 v43, v56, v131
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v26, v139 :: v_dual_fmac_f32 v53, v36, v139
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v46, v139 :: v_dual_fmac_f32 v63, v56, v139
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v27, v132 :: v_dual_fmac_f32 v33, v37, v132
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v47, v132 :: v_dual_fmac_f32 v43, v57, v132
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v27, v140 :: v_dual_fmac_f32 v53, v37, v140
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v47, v140 :: v_dual_fmac_f32 v63, v57, v140
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v28, v133 :: v_dual_fmac_f32 v33, v38, v133
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v48, v133 :: v_dual_fmac_f32 v43, v58, v133
	s_wait_loadcnt 0xc
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v28, v141 :: v_dual_fmac_f32 v53, v38, v141
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v48, v141 :: v_dual_fmac_f32 v63, v58, v141
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v29, v134 :: v_dual_fmac_f32 v33, v39, v134
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v49, v134 :: v_dual_fmac_f32 v43, v59, v134
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v29, v142 :: v_dual_fmac_f32 v53, v39, v142
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v49, v142 :: v_dual_fmac_f32 v63, v59, v142
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v30, v135 :: v_dual_fmac_f32 v33, v40, v135
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v50, v135 :: v_dual_fmac_f32 v43, v60, v135
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v30, v143 :: v_dual_fmac_f32 v53, v40, v143
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v50, v143 :: v_dual_fmac_f32 v63, v60, v143
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v31, v136 :: v_dual_fmac_f32 v33, v41, v136
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v51, v136 :: v_dual_fmac_f32 v43, v61, v136
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v31, v144 :: v_dual_fmac_f32 v53, v41, v144
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v51, v144 :: v_dual_fmac_f32 v63, v61, v144
	global_load_b128 v[129:132], v3, s[32:33]
	global_load_b128 v[133:136], v3, s[32:33] offset:16
	global_load_b128 v[137:140], v3, s[34:35]
	global_load_b128 v[141:144], v3, s[34:35] offset:16
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v92, v32, v92 :: v_dual_add_f32 v93, v93, v33
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v94, v42, v94 :: v_dual_add_f32 v95, v95, v43
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v104, v52, v104 :: v_dual_add_f32 v105, v105, v53
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v106, v62, v106 :: v_dual_add_f32 v107, v107, v63
	s_mov_b32 s15, s16
	v_mov_b32_e32 v2, v3
	s_add_co_i32 s2, s2, -1
	s_cmp_lg_u32 s2, 0
	s_cbranch_scc1 .Lrx_b4_s1
	s_wait_loadcnt 0x0
	.Lrx_b4_s1_done:
	v_dual_add_f32 v64, v64, v68 :: v_dual_add_f32 v65, v65, v69
	v_dual_add_f32 v66, v66, v70 :: v_dual_add_f32 v67, v67, v71
	v_dual_add_f32 v76, v76, v80 :: v_dual_add_f32 v77, v77, v81
	v_dual_add_f32 v78, v78, v82 :: v_dual_add_f32 v79, v79, v83
	v_dual_add_f32 v88, v88, v92 :: v_dual_add_f32 v89, v89, v93
	v_dual_add_f32 v90, v90, v94 :: v_dual_add_f32 v91, v91, v95
	v_dual_add_f32 v100, v100, v104 :: v_dual_add_f32 v101, v101, v105
	v_dual_add_f32 v102, v102, v106 :: v_dual_add_f32 v103, v103, v107
	v_mov_b32_e32 v68, 0
	v_mov_b32_e32 v69, 0
	v_mov_b32_e32 v70, 0
	v_mov_b32_e32 v71, 0
	v_mov_b32_e32 v80, 0
	v_mov_b32_e32 v81, 0
	v_mov_b32_e32 v82, 0
	v_mov_b32_e32 v83, 0
	v_mov_b32_e32 v92, 0
	v_mov_b32_e32 v93, 0
	v_mov_b32_e32 v94, 0
	v_mov_b32_e32 v95, 0
	v_mov_b32_e32 v104, 0
	v_mov_b32_e32 v105, 0
	v_mov_b32_e32 v106, 0
	v_mov_b32_e32 v107, 0
	s_add_co_i32 s2, s14, 1
	s_lshr_b32 s2, s2, 2
	s_cmp_eq_u32 s2, 0
	s_cbranch_scc1 .Lrx_b4_s2_done
	s_mov_b32 s15, 0x110
	v_lshlrev_b32_e32 v2, 5, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v2, 0x800, v2
	buffer_load_b64 v[12:13], v8, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v20, v4, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[14:15], v9, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v21, v5, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[16:17], v10, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v22, v6, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[18:19], v11, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v23, v7, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	global_load_b128 v[113:116], v2, s[28:29]
	global_load_b128 v[117:120], v2, s[28:29] offset:16
	global_load_b128 v[121:124], v2, s[30:31]
	global_load_b128 v[125:128], v2, s[30:31] offset:16
	global_load_b128 v[129:132], v2, s[32:33]
	global_load_b128 v[133:136], v2, s[32:33] offset:16
	global_load_b128 v[137:140], v2, s[34:35]
	global_load_b128 v[141:144], v2, s[34:35] offset:16
	.Lrx_b4_s2:
	s_cmp_gt_u32 s2, 1
	s_cselect_b32 s17, 0x220, 0
	s_cselect_b32 s18, 0x1000, 0
	s_add_co_i32 s16, s15, s17
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v3, s18, v2
	v_cmp_gt_u32_e32 vcc_lo, 16, v0
	s_wait_loadcnt 0xf
	v_cndmask_b32_e32 v12, v13, v12, vcc_lo
	s_wait_loadcnt 0xe
	v_and_b32_e32 v30, 0xf0f0f0f, v20
	v_lshrrev_b32_e32 v31, 4, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v31, 0xf0f0f0f, v31
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v24, v30
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v26, v30
	v_cvt_f32_ubyte2_e32 v28, v30
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v25, v31
	v_cvt_f32_ubyte1_e32 v27, v31
	v_cvt_f32_ubyte2_e32 v29, v31
	v_cvt_f32_ubyte3_e32 v30, v30
	v_cvt_f32_ubyte3_e32 v31, v31
	v_fma_mix_f32 v24, v12, v24, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v25, v12, v25, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v26, v12, v26, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v27, v12, v27, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v28, v12, v28, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v29, v12, v29, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v30, v12, v30, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v31, v12, v31, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xd
	v_cndmask_b32_e32 v14, v15, v14, vcc_lo
	s_wait_loadcnt 0xc
	v_and_b32_e32 v40, 0xf0f0f0f, v21
	v_lshrrev_b32_e32 v41, 4, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v41, 0xf0f0f0f, v41
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v34, v40
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v36, v40
	v_cvt_f32_ubyte2_e32 v38, v40
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v35, v41
	v_cvt_f32_ubyte1_e32 v37, v41
	v_cvt_f32_ubyte2_e32 v39, v41
	v_cvt_f32_ubyte3_e32 v40, v40
	v_cvt_f32_ubyte3_e32 v41, v41
	v_fma_mix_f32 v34, v14, v34, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v35, v14, v35, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v36, v14, v36, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v37, v14, v37, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v38, v14, v38, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v39, v14, v39, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v40, v14, v40, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v41, v14, v41, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xb
	v_cndmask_b32_e32 v16, v17, v16, vcc_lo
	s_wait_loadcnt 0xa
	v_and_b32_e32 v50, 0xf0f0f0f, v22
	v_lshrrev_b32_e32 v51, 4, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v51, 0xf0f0f0f, v51
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v44, v50
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v46, v50
	v_cvt_f32_ubyte2_e32 v48, v50
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v45, v51
	v_cvt_f32_ubyte1_e32 v47, v51
	v_cvt_f32_ubyte2_e32 v49, v51
	v_cvt_f32_ubyte3_e32 v50, v50
	v_cvt_f32_ubyte3_e32 v51, v51
	v_fma_mix_f32 v44, v16, v44, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v16, v45, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v16, v46, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v16, v47, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v48, v16, v48, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v49, v16, v49, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v16, v50, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v16, v51, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x9
	v_cndmask_b32_e32 v18, v19, v18, vcc_lo
	s_wait_loadcnt 0x8
	v_and_b32_e32 v60, 0xf0f0f0f, v23
	v_lshrrev_b32_e32 v61, 4, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v61, 0xf0f0f0f, v61
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v54, v60
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v56, v60
	v_cvt_f32_ubyte2_e32 v58, v60
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v55, v61
	v_cvt_f32_ubyte1_e32 v57, v61
	v_cvt_f32_ubyte2_e32 v59, v61
	v_cvt_f32_ubyte3_e32 v60, v60
	v_cvt_f32_ubyte3_e32 v61, v61
	v_fma_mix_f32 v54, v18, v54, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v18, v55, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v18, v56, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v18, v57, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v58, v18, v58, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v59, v18, v59, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v18, v60, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v18, v61, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	buffer_load_b64 v[12:13], v8, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v20, v4, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[14:15], v9, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v21, v5, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[16:17], v10, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v22, v6, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[18:19], v11, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v23, v7, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	s_wait_loadcnt 0xf
	v_dual_mul_f32 v32, v24, v113 :: v_dual_mul_f32 v33, v35, v114
	v_dual_mul_f32 v42, v44, v113 :: v_dual_mul_f32 v43, v55, v114
	s_wait_loadcnt 0xd
	v_dual_mul_f32 v52, v24, v121 :: v_dual_mul_f32 v53, v35, v122
	v_dual_mul_f32 v62, v44, v121 :: v_dual_mul_f32 v63, v55, v122
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v25, v114 :: v_dual_fmac_f32 v33, v34, v113
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v45, v114 :: v_dual_fmac_f32 v43, v54, v113
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v25, v122 :: v_dual_fmac_f32 v53, v34, v121
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v45, v122 :: v_dual_fmac_f32 v63, v54, v121
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v26, v115 :: v_dual_fmac_f32 v33, v36, v115
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v46, v115 :: v_dual_fmac_f32 v43, v56, v115
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v26, v123 :: v_dual_fmac_f32 v53, v36, v123
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v46, v123 :: v_dual_fmac_f32 v63, v56, v123
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v27, v116 :: v_dual_fmac_f32 v33, v37, v116
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v47, v116 :: v_dual_fmac_f32 v43, v57, v116
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v27, v124 :: v_dual_fmac_f32 v53, v37, v124
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v47, v124 :: v_dual_fmac_f32 v63, v57, v124
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v28, v117 :: v_dual_fmac_f32 v33, v38, v117
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v48, v117 :: v_dual_fmac_f32 v43, v58, v117
	s_wait_loadcnt 0xc
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v28, v125 :: v_dual_fmac_f32 v53, v38, v125
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v48, v125 :: v_dual_fmac_f32 v63, v58, v125
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v29, v118 :: v_dual_fmac_f32 v33, v39, v118
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v49, v118 :: v_dual_fmac_f32 v43, v59, v118
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v29, v126 :: v_dual_fmac_f32 v53, v39, v126
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v49, v126 :: v_dual_fmac_f32 v63, v59, v126
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v30, v119 :: v_dual_fmac_f32 v33, v40, v119
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v50, v119 :: v_dual_fmac_f32 v43, v60, v119
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v30, v127 :: v_dual_fmac_f32 v53, v40, v127
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v50, v127 :: v_dual_fmac_f32 v63, v60, v127
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v31, v120 :: v_dual_fmac_f32 v33, v41, v120
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v51, v120 :: v_dual_fmac_f32 v43, v61, v120
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v31, v128 :: v_dual_fmac_f32 v53, v41, v128
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v51, v128 :: v_dual_fmac_f32 v63, v61, v128
	global_load_b128 v[113:116], v3, s[28:29]
	global_load_b128 v[117:120], v3, s[28:29] offset:16
	global_load_b128 v[121:124], v3, s[30:31]
	global_load_b128 v[125:128], v3, s[30:31] offset:16
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v68, v32, v68 :: v_dual_add_f32 v69, v69, v33
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v70, v42, v70 :: v_dual_add_f32 v71, v71, v43
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v80, v52, v80 :: v_dual_add_f32 v81, v81, v53
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v82, v62, v82 :: v_dual_add_f32 v83, v83, v63
	s_wait_loadcnt 0xf
	v_dual_mul_f32 v32, v24, v129 :: v_dual_mul_f32 v33, v35, v130
	v_dual_mul_f32 v42, v44, v129 :: v_dual_mul_f32 v43, v55, v130
	s_wait_loadcnt 0xd
	v_dual_mul_f32 v52, v24, v137 :: v_dual_mul_f32 v53, v35, v138
	v_dual_mul_f32 v62, v44, v137 :: v_dual_mul_f32 v63, v55, v138
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v25, v130 :: v_dual_fmac_f32 v33, v34, v129
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v45, v130 :: v_dual_fmac_f32 v43, v54, v129
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v25, v138 :: v_dual_fmac_f32 v53, v34, v137
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v45, v138 :: v_dual_fmac_f32 v63, v54, v137
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v26, v131 :: v_dual_fmac_f32 v33, v36, v131
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v46, v131 :: v_dual_fmac_f32 v43, v56, v131
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v26, v139 :: v_dual_fmac_f32 v53, v36, v139
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v46, v139 :: v_dual_fmac_f32 v63, v56, v139
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v27, v132 :: v_dual_fmac_f32 v33, v37, v132
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v47, v132 :: v_dual_fmac_f32 v43, v57, v132
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v27, v140 :: v_dual_fmac_f32 v53, v37, v140
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v47, v140 :: v_dual_fmac_f32 v63, v57, v140
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v28, v133 :: v_dual_fmac_f32 v33, v38, v133
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v48, v133 :: v_dual_fmac_f32 v43, v58, v133
	s_wait_loadcnt 0xc
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v28, v141 :: v_dual_fmac_f32 v53, v38, v141
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v48, v141 :: v_dual_fmac_f32 v63, v58, v141
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v29, v134 :: v_dual_fmac_f32 v33, v39, v134
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v49, v134 :: v_dual_fmac_f32 v43, v59, v134
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v29, v142 :: v_dual_fmac_f32 v53, v39, v142
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v49, v142 :: v_dual_fmac_f32 v63, v59, v142
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v30, v135 :: v_dual_fmac_f32 v33, v40, v135
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v50, v135 :: v_dual_fmac_f32 v43, v60, v135
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v30, v143 :: v_dual_fmac_f32 v53, v40, v143
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v50, v143 :: v_dual_fmac_f32 v63, v60, v143
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v31, v136 :: v_dual_fmac_f32 v33, v41, v136
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v51, v136 :: v_dual_fmac_f32 v43, v61, v136
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v31, v144 :: v_dual_fmac_f32 v53, v41, v144
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v51, v144 :: v_dual_fmac_f32 v63, v61, v144
	global_load_b128 v[129:132], v3, s[32:33]
	global_load_b128 v[133:136], v3, s[32:33] offset:16
	global_load_b128 v[137:140], v3, s[34:35]
	global_load_b128 v[141:144], v3, s[34:35] offset:16
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v92, v32, v92 :: v_dual_add_f32 v93, v93, v33
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v94, v42, v94 :: v_dual_add_f32 v95, v95, v43
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v104, v52, v104 :: v_dual_add_f32 v105, v105, v53
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v106, v62, v106 :: v_dual_add_f32 v107, v107, v63
	s_mov_b32 s15, s16
	v_mov_b32_e32 v2, v3
	s_add_co_i32 s2, s2, -1
	s_cmp_lg_u32 s2, 0
	s_cbranch_scc1 .Lrx_b4_s2
	s_wait_loadcnt 0x0
	.Lrx_b4_s2_done:
	s_add_co_i32 s2, s14, 0
	s_lshr_b32 s2, s2, 2
	s_cmp_eq_u32 s2, 0
	s_cbranch_scc1 .Lrx_b4_s3_done
	s_mov_b32 s15, 0x198
	v_lshlrev_b32_e32 v2, 5, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v2, 0xc00, v2
	buffer_load_b64 v[12:13], v8, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v20, v4, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[14:15], v9, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v21, v5, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[16:17], v10, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v22, v6, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[18:19], v11, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v23, v7, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	global_load_b128 v[113:116], v2, s[28:29]
	global_load_b128 v[117:120], v2, s[28:29] offset:16
	global_load_b128 v[121:124], v2, s[30:31]
	global_load_b128 v[125:128], v2, s[30:31] offset:16
	global_load_b128 v[129:132], v2, s[32:33]
	global_load_b128 v[133:136], v2, s[32:33] offset:16
	global_load_b128 v[137:140], v2, s[34:35]
	global_load_b128 v[141:144], v2, s[34:35] offset:16
	.Lrx_b4_s3:
	s_cmp_gt_u32 s2, 1
	s_cselect_b32 s17, 0x220, 0
	s_cselect_b32 s18, 0x1000, 0
	s_add_co_i32 s16, s15, s17
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v3, s18, v2
	v_cmp_gt_u32_e32 vcc_lo, 16, v0
	s_wait_loadcnt 0xf
	v_cndmask_b32_e32 v12, v13, v12, vcc_lo
	s_wait_loadcnt 0xe
	v_and_b32_e32 v30, 0xf0f0f0f, v20
	v_lshrrev_b32_e32 v31, 4, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v31, 0xf0f0f0f, v31
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v24, v30
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v26, v30
	v_cvt_f32_ubyte2_e32 v28, v30
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v25, v31
	v_cvt_f32_ubyte1_e32 v27, v31
	v_cvt_f32_ubyte2_e32 v29, v31
	v_cvt_f32_ubyte3_e32 v30, v30
	v_cvt_f32_ubyte3_e32 v31, v31
	v_fma_mix_f32 v24, v12, v24, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v25, v12, v25, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v26, v12, v26, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v27, v12, v27, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v28, v12, v28, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v29, v12, v29, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v30, v12, v30, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v31, v12, v31, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xd
	v_cndmask_b32_e32 v14, v15, v14, vcc_lo
	s_wait_loadcnt 0xc
	v_and_b32_e32 v40, 0xf0f0f0f, v21
	v_lshrrev_b32_e32 v41, 4, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v41, 0xf0f0f0f, v41
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v34, v40
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v36, v40
	v_cvt_f32_ubyte2_e32 v38, v40
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v35, v41
	v_cvt_f32_ubyte1_e32 v37, v41
	v_cvt_f32_ubyte2_e32 v39, v41
	v_cvt_f32_ubyte3_e32 v40, v40
	v_cvt_f32_ubyte3_e32 v41, v41
	v_fma_mix_f32 v34, v14, v34, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v35, v14, v35, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v36, v14, v36, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v37, v14, v37, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v38, v14, v38, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v39, v14, v39, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v40, v14, v40, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v41, v14, v41, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xb
	v_cndmask_b32_e32 v16, v17, v16, vcc_lo
	s_wait_loadcnt 0xa
	v_and_b32_e32 v50, 0xf0f0f0f, v22
	v_lshrrev_b32_e32 v51, 4, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v51, 0xf0f0f0f, v51
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v44, v50
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v46, v50
	v_cvt_f32_ubyte2_e32 v48, v50
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v45, v51
	v_cvt_f32_ubyte1_e32 v47, v51
	v_cvt_f32_ubyte2_e32 v49, v51
	v_cvt_f32_ubyte3_e32 v50, v50
	v_cvt_f32_ubyte3_e32 v51, v51
	v_fma_mix_f32 v44, v16, v44, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v16, v45, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v16, v46, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v16, v47, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v48, v16, v48, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v49, v16, v49, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v16, v50, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v16, v51, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x9
	v_cndmask_b32_e32 v18, v19, v18, vcc_lo
	s_wait_loadcnt 0x8
	v_and_b32_e32 v60, 0xf0f0f0f, v23
	v_lshrrev_b32_e32 v61, 4, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v61, 0xf0f0f0f, v61
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v54, v60
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v56, v60
	v_cvt_f32_ubyte2_e32 v58, v60
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v55, v61
	v_cvt_f32_ubyte1_e32 v57, v61
	v_cvt_f32_ubyte2_e32 v59, v61
	v_cvt_f32_ubyte3_e32 v60, v60
	v_cvt_f32_ubyte3_e32 v61, v61
	v_fma_mix_f32 v54, v18, v54, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v18, v55, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v18, v56, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v18, v57, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v58, v18, v58, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v59, v18, v59, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v18, v60, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v18, v61, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	buffer_load_b64 v[12:13], v8, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v20, v4, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[14:15], v9, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v21, v5, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[16:17], v10, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v22, v6, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[18:19], v11, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v23, v7, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	s_wait_loadcnt 0xf
	v_dual_mul_f32 v32, v24, v113 :: v_dual_mul_f32 v33, v35, v114
	v_dual_mul_f32 v42, v44, v113 :: v_dual_mul_f32 v43, v55, v114
	s_wait_loadcnt 0xd
	v_dual_mul_f32 v52, v24, v121 :: v_dual_mul_f32 v53, v35, v122
	v_dual_mul_f32 v62, v44, v121 :: v_dual_mul_f32 v63, v55, v122
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v25, v114 :: v_dual_fmac_f32 v33, v34, v113
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v45, v114 :: v_dual_fmac_f32 v43, v54, v113
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v25, v122 :: v_dual_fmac_f32 v53, v34, v121
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v45, v122 :: v_dual_fmac_f32 v63, v54, v121
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v26, v115 :: v_dual_fmac_f32 v33, v36, v115
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v46, v115 :: v_dual_fmac_f32 v43, v56, v115
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v26, v123 :: v_dual_fmac_f32 v53, v36, v123
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v46, v123 :: v_dual_fmac_f32 v63, v56, v123
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v27, v116 :: v_dual_fmac_f32 v33, v37, v116
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v47, v116 :: v_dual_fmac_f32 v43, v57, v116
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v27, v124 :: v_dual_fmac_f32 v53, v37, v124
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v47, v124 :: v_dual_fmac_f32 v63, v57, v124
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v28, v117 :: v_dual_fmac_f32 v33, v38, v117
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v48, v117 :: v_dual_fmac_f32 v43, v58, v117
	s_wait_loadcnt 0xc
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v28, v125 :: v_dual_fmac_f32 v53, v38, v125
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v48, v125 :: v_dual_fmac_f32 v63, v58, v125
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v29, v118 :: v_dual_fmac_f32 v33, v39, v118
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v49, v118 :: v_dual_fmac_f32 v43, v59, v118
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v29, v126 :: v_dual_fmac_f32 v53, v39, v126
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v49, v126 :: v_dual_fmac_f32 v63, v59, v126
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v30, v119 :: v_dual_fmac_f32 v33, v40, v119
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v50, v119 :: v_dual_fmac_f32 v43, v60, v119
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v30, v127 :: v_dual_fmac_f32 v53, v40, v127
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v50, v127 :: v_dual_fmac_f32 v63, v60, v127
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v31, v120 :: v_dual_fmac_f32 v33, v41, v120
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v51, v120 :: v_dual_fmac_f32 v43, v61, v120
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v31, v128 :: v_dual_fmac_f32 v53, v41, v128
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v51, v128 :: v_dual_fmac_f32 v63, v61, v128
	global_load_b128 v[113:116], v3, s[28:29]
	global_load_b128 v[117:120], v3, s[28:29] offset:16
	global_load_b128 v[121:124], v3, s[30:31]
	global_load_b128 v[125:128], v3, s[30:31] offset:16
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v72, v32, v72 :: v_dual_add_f32 v73, v73, v33
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v74, v42, v74 :: v_dual_add_f32 v75, v75, v43
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v84, v52, v84 :: v_dual_add_f32 v85, v85, v53
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v86, v62, v86 :: v_dual_add_f32 v87, v87, v63
	s_wait_loadcnt 0xf
	v_dual_mul_f32 v32, v24, v129 :: v_dual_mul_f32 v33, v35, v130
	v_dual_mul_f32 v42, v44, v129 :: v_dual_mul_f32 v43, v55, v130
	s_wait_loadcnt 0xd
	v_dual_mul_f32 v52, v24, v137 :: v_dual_mul_f32 v53, v35, v138
	v_dual_mul_f32 v62, v44, v137 :: v_dual_mul_f32 v63, v55, v138
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v25, v130 :: v_dual_fmac_f32 v33, v34, v129
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v45, v130 :: v_dual_fmac_f32 v43, v54, v129
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v25, v138 :: v_dual_fmac_f32 v53, v34, v137
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v45, v138 :: v_dual_fmac_f32 v63, v54, v137
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v26, v131 :: v_dual_fmac_f32 v33, v36, v131
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v46, v131 :: v_dual_fmac_f32 v43, v56, v131
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v26, v139 :: v_dual_fmac_f32 v53, v36, v139
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v46, v139 :: v_dual_fmac_f32 v63, v56, v139
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v27, v132 :: v_dual_fmac_f32 v33, v37, v132
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v47, v132 :: v_dual_fmac_f32 v43, v57, v132
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v27, v140 :: v_dual_fmac_f32 v53, v37, v140
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v47, v140 :: v_dual_fmac_f32 v63, v57, v140
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v28, v133 :: v_dual_fmac_f32 v33, v38, v133
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v48, v133 :: v_dual_fmac_f32 v43, v58, v133
	s_wait_loadcnt 0xc
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v28, v141 :: v_dual_fmac_f32 v53, v38, v141
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v48, v141 :: v_dual_fmac_f32 v63, v58, v141
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v29, v134 :: v_dual_fmac_f32 v33, v39, v134
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v49, v134 :: v_dual_fmac_f32 v43, v59, v134
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v29, v142 :: v_dual_fmac_f32 v53, v39, v142
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v49, v142 :: v_dual_fmac_f32 v63, v59, v142
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v30, v135 :: v_dual_fmac_f32 v33, v40, v135
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v50, v135 :: v_dual_fmac_f32 v43, v60, v135
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v30, v143 :: v_dual_fmac_f32 v53, v40, v143
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v50, v143 :: v_dual_fmac_f32 v63, v60, v143
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v31, v136 :: v_dual_fmac_f32 v33, v41, v136
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v51, v136 :: v_dual_fmac_f32 v43, v61, v136
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v31, v144 :: v_dual_fmac_f32 v53, v41, v144
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v51, v144 :: v_dual_fmac_f32 v63, v61, v144
	global_load_b128 v[129:132], v3, s[32:33]
	global_load_b128 v[133:136], v3, s[32:33] offset:16
	global_load_b128 v[137:140], v3, s[34:35]
	global_load_b128 v[141:144], v3, s[34:35] offset:16
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v96, v32, v96 :: v_dual_add_f32 v97, v97, v33
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v98, v42, v98 :: v_dual_add_f32 v99, v99, v43
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v108, v52, v108 :: v_dual_add_f32 v109, v109, v53
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v110, v62, v110 :: v_dual_add_f32 v111, v111, v63
	s_mov_b32 s15, s16
	v_mov_b32_e32 v2, v3
	s_add_co_i32 s2, s2, -1
	s_cmp_lg_u32 s2, 0
	s_cbranch_scc1 .Lrx_b4_s3
	s_wait_loadcnt 0x0
	.Lrx_b4_s3_done:
	v_dual_add_f32 v68, v68, v72 :: v_dual_add_f32 v69, v69, v73
	v_dual_add_f32 v70, v70, v74 :: v_dual_add_f32 v71, v71, v75
	v_dual_add_f32 v80, v80, v84 :: v_dual_add_f32 v81, v81, v85
	v_dual_add_f32 v82, v82, v86 :: v_dual_add_f32 v83, v83, v87
	v_dual_add_f32 v92, v92, v96 :: v_dual_add_f32 v93, v93, v97
	v_dual_add_f32 v94, v94, v98 :: v_dual_add_f32 v95, v95, v99
	v_dual_add_f32 v104, v104, v108 :: v_dual_add_f32 v105, v105, v109
	v_dual_add_f32 v106, v106, v110 :: v_dual_add_f32 v107, v107, v111
	v_dual_add_f32 v64, v64, v68 :: v_dual_add_f32 v65, v65, v69
	v_dual_add_f32 v66, v66, v70 :: v_dual_add_f32 v67, v67, v71
	v_dual_add_f32 v76, v76, v80 :: v_dual_add_f32 v77, v77, v81
	v_dual_add_f32 v78, v78, v82 :: v_dual_add_f32 v79, v79, v83
	v_dual_add_f32 v88, v88, v92 :: v_dual_add_f32 v89, v89, v93
	v_dual_add_f32 v90, v90, v94 :: v_dual_add_f32 v91, v91, v95
	v_dual_add_f32 v100, v100, v104 :: v_dual_add_f32 v101, v101, v105
	v_dual_add_f32 v102, v102, v106 :: v_dual_add_f32 v103, v103, v107
	ds_swizzle_b32 v24, v64 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v25, v65 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v26, v66 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v27, v67 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v28, v76 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v29, v77 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v30, v78 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v31, v79 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v32, v88 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v33, v89 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v34, v90 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v35, v91 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v36, v100 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v37, v101 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v38, v102 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v39, v103 offset:swizzle(BITMASK_PERM,"1pppp")
	s_wait_dscnt 0xf
	v_add_f32_e32 v64, v64, v24
	s_wait_dscnt 0xe
	v_add_f32_e32 v65, v65, v25
	s_wait_dscnt 0xd
	v_add_f32_e32 v66, v66, v26
	s_wait_dscnt 0xc
	v_add_f32_e32 v67, v67, v27
	s_wait_dscnt 0xb
	v_add_f32_e32 v76, v76, v28
	s_wait_dscnt 0xa
	v_add_f32_e32 v77, v77, v29
	s_wait_dscnt 0x9
	v_add_f32_e32 v78, v78, v30
	s_wait_dscnt 0x8
	v_add_f32_e32 v79, v79, v31
	s_wait_dscnt 0x7
	v_add_f32_e32 v88, v88, v32
	s_wait_dscnt 0x6
	v_add_f32_e32 v89, v89, v33
	s_wait_dscnt 0x5
	v_add_f32_e32 v90, v90, v34
	s_wait_dscnt 0x4
	v_add_f32_e32 v91, v91, v35
	s_wait_dscnt 0x3
	v_add_f32_e32 v100, v100, v36
	s_wait_dscnt 0x2
	v_add_f32_e32 v101, v101, v37
	s_wait_dscnt 0x1
	v_add_f32_e32 v102, v102, v38
	s_wait_dscnt 0x0
	v_add_f32_e32 v103, v103, v39
	v_cmp_gt_u32_e32 vcc_lo, 24, v0
	v_cndmask_b32_e64 v3, 0, 8, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v3, v3, v0, 2
	ds_bpermute_b32 v24, v3, v64
	ds_bpermute_b32 v25, v3, v65
	ds_bpermute_b32 v26, v3, v66
	ds_bpermute_b32 v27, v3, v67
	ds_bpermute_b32 v28, v3, v76
	ds_bpermute_b32 v29, v3, v77
	ds_bpermute_b32 v30, v3, v78
	ds_bpermute_b32 v31, v3, v79
	ds_bpermute_b32 v32, v3, v88
	ds_bpermute_b32 v33, v3, v89
	ds_bpermute_b32 v34, v3, v90
	ds_bpermute_b32 v35, v3, v91
	ds_bpermute_b32 v36, v3, v100
	ds_bpermute_b32 v37, v3, v101
	ds_bpermute_b32 v38, v3, v102
	ds_bpermute_b32 v39, v3, v103
	s_wait_dscnt 0xf
	v_add_f32_e32 v64, v64, v24
	s_wait_dscnt 0xe
	v_add_f32_e32 v65, v65, v25
	s_wait_dscnt 0xd
	v_add_f32_e32 v66, v66, v26
	s_wait_dscnt 0xc
	v_add_f32_e32 v67, v67, v27
	s_wait_dscnt 0xb
	v_add_f32_e32 v76, v76, v28
	s_wait_dscnt 0xa
	v_add_f32_e32 v77, v77, v29
	s_wait_dscnt 0x9
	v_add_f32_e32 v78, v78, v30
	s_wait_dscnt 0x8
	v_add_f32_e32 v79, v79, v31
	s_wait_dscnt 0x7
	v_add_f32_e32 v88, v88, v32
	s_wait_dscnt 0x6
	v_add_f32_e32 v89, v89, v33
	s_wait_dscnt 0x5
	v_add_f32_e32 v90, v90, v34
	s_wait_dscnt 0x4
	v_add_f32_e32 v91, v91, v35
	s_wait_dscnt 0x3
	v_add_f32_e32 v100, v100, v36
	s_wait_dscnt 0x2
	v_add_f32_e32 v101, v101, v37
	s_wait_dscnt 0x1
	v_add_f32_e32 v102, v102, v38
	s_wait_dscnt 0x0
	v_add_f32_e32 v103, v103, v39
	v_cmp_gt_u32_e32 vcc_lo, 28, v0
	v_cndmask_b32_e64 v3, 0, 4, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v3, v3, v0, 2
	ds_bpermute_b32 v24, v3, v64
	ds_bpermute_b32 v25, v3, v65
	ds_bpermute_b32 v26, v3, v66
	ds_bpermute_b32 v27, v3, v67
	ds_bpermute_b32 v28, v3, v76
	ds_bpermute_b32 v29, v3, v77
	ds_bpermute_b32 v30, v3, v78
	ds_bpermute_b32 v31, v3, v79
	ds_bpermute_b32 v32, v3, v88
	ds_bpermute_b32 v33, v3, v89
	ds_bpermute_b32 v34, v3, v90
	ds_bpermute_b32 v35, v3, v91
	ds_bpermute_b32 v36, v3, v100
	ds_bpermute_b32 v37, v3, v101
	ds_bpermute_b32 v38, v3, v102
	ds_bpermute_b32 v39, v3, v103
	s_wait_dscnt 0xf
	v_add_f32_e32 v64, v64, v24
	s_wait_dscnt 0xe
	v_add_f32_e32 v65, v65, v25
	s_wait_dscnt 0xd
	v_add_f32_e32 v66, v66, v26
	s_wait_dscnt 0xc
	v_add_f32_e32 v67, v67, v27
	s_wait_dscnt 0xb
	v_add_f32_e32 v76, v76, v28
	s_wait_dscnt 0xa
	v_add_f32_e32 v77, v77, v29
	s_wait_dscnt 0x9
	v_add_f32_e32 v78, v78, v30
	s_wait_dscnt 0x8
	v_add_f32_e32 v79, v79, v31
	s_wait_dscnt 0x7
	v_add_f32_e32 v88, v88, v32
	s_wait_dscnt 0x6
	v_add_f32_e32 v89, v89, v33
	s_wait_dscnt 0x5
	v_add_f32_e32 v90, v90, v34
	s_wait_dscnt 0x4
	v_add_f32_e32 v91, v91, v35
	s_wait_dscnt 0x3
	v_add_f32_e32 v100, v100, v36
	s_wait_dscnt 0x2
	v_add_f32_e32 v101, v101, v37
	s_wait_dscnt 0x1
	v_add_f32_e32 v102, v102, v38
	s_wait_dscnt 0x0
	v_add_f32_e32 v103, v103, v39
	v_cmp_gt_u32_e32 vcc_lo, 30, v0
	v_cndmask_b32_e64 v3, 0, 2, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v3, v3, v0, 2
	ds_bpermute_b32 v24, v3, v64
	ds_bpermute_b32 v25, v3, v65
	ds_bpermute_b32 v26, v3, v66
	ds_bpermute_b32 v27, v3, v67
	ds_bpermute_b32 v28, v3, v76
	ds_bpermute_b32 v29, v3, v77
	ds_bpermute_b32 v30, v3, v78
	ds_bpermute_b32 v31, v3, v79
	ds_bpermute_b32 v32, v3, v88
	ds_bpermute_b32 v33, v3, v89
	ds_bpermute_b32 v34, v3, v90
	ds_bpermute_b32 v35, v3, v91
	ds_bpermute_b32 v36, v3, v100
	ds_bpermute_b32 v37, v3, v101
	ds_bpermute_b32 v38, v3, v102
	ds_bpermute_b32 v39, v3, v103
	s_wait_dscnt 0xf
	v_add_f32_e32 v64, v64, v24
	s_wait_dscnt 0xe
	v_add_f32_e32 v65, v65, v25
	s_wait_dscnt 0xd
	v_add_f32_e32 v66, v66, v26
	s_wait_dscnt 0xc
	v_add_f32_e32 v67, v67, v27
	s_wait_dscnt 0xb
	v_add_f32_e32 v76, v76, v28
	s_wait_dscnt 0xa
	v_add_f32_e32 v77, v77, v29
	s_wait_dscnt 0x9
	v_add_f32_e32 v78, v78, v30
	s_wait_dscnt 0x8
	v_add_f32_e32 v79, v79, v31
	s_wait_dscnt 0x7
	v_add_f32_e32 v88, v88, v32
	s_wait_dscnt 0x6
	v_add_f32_e32 v89, v89, v33
	s_wait_dscnt 0x5
	v_add_f32_e32 v90, v90, v34
	s_wait_dscnt 0x4
	v_add_f32_e32 v91, v91, v35
	s_wait_dscnt 0x3
	v_add_f32_e32 v100, v100, v36
	s_wait_dscnt 0x2
	v_add_f32_e32 v101, v101, v37
	s_wait_dscnt 0x1
	v_add_f32_e32 v102, v102, v38
	s_wait_dscnt 0x0
	v_add_f32_e32 v103, v103, v39
	v_cmp_gt_u32_e32 vcc_lo, 31, v0
	v_cndmask_b32_e64 v3, 0, 1, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v3, v3, v0, 2
	ds_bpermute_b32 v24, v3, v64
	ds_bpermute_b32 v25, v3, v65
	ds_bpermute_b32 v26, v3, v66
	ds_bpermute_b32 v27, v3, v67
	ds_bpermute_b32 v28, v3, v76
	ds_bpermute_b32 v29, v3, v77
	ds_bpermute_b32 v30, v3, v78
	ds_bpermute_b32 v31, v3, v79
	ds_bpermute_b32 v32, v3, v88
	ds_bpermute_b32 v33, v3, v89
	ds_bpermute_b32 v34, v3, v90
	ds_bpermute_b32 v35, v3, v91
	ds_bpermute_b32 v36, v3, v100
	ds_bpermute_b32 v37, v3, v101
	ds_bpermute_b32 v38, v3, v102
	ds_bpermute_b32 v39, v3, v103
	s_wait_dscnt 0xf
	v_add_f32_e32 v64, v64, v24
	s_wait_dscnt 0xe
	v_add_f32_e32 v65, v65, v25
	s_wait_dscnt 0xd
	v_add_f32_e32 v66, v66, v26
	s_wait_dscnt 0xc
	v_add_f32_e32 v67, v67, v27
	s_wait_dscnt 0xb
	v_add_f32_e32 v76, v76, v28
	s_wait_dscnt 0xa
	v_add_f32_e32 v77, v77, v29
	s_wait_dscnt 0x9
	v_add_f32_e32 v78, v78, v30
	s_wait_dscnt 0x8
	v_add_f32_e32 v79, v79, v31
	s_wait_dscnt 0x7
	v_add_f32_e32 v88, v88, v32
	s_wait_dscnt 0x6
	v_add_f32_e32 v89, v89, v33
	s_wait_dscnt 0x5
	v_add_f32_e32 v90, v90, v34
	s_wait_dscnt 0x4
	v_add_f32_e32 v91, v91, v35
	s_wait_dscnt 0x3
	v_add_f32_e32 v100, v100, v36
	s_wait_dscnt 0x2
	v_add_f32_e32 v101, v101, v37
	s_wait_dscnt 0x1
	v_add_f32_e32 v102, v102, v38
	s_wait_dscnt 0x0
	v_add_f32_e32 v103, v103, v39
	v_cmpx_eq_u32_e32 0, v0
	s_lshl_b32 s46, s12, 2
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v56, s46
	s_mul_i32 s47, s45, 1
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s47, s47, s46
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v57, s47
	s_mul_i32 s47, s45, 2
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s47, s47, s46
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v58, s47
	s_mul_i32 s47, s45, 3
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s47, s47, s46
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v59, s47
	s_add_co_i32 s13, s12, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cbranch_scc0 .Lrx_b4_p0_single
	global_load_b32 v40, v56, s[8:9]
	global_load_b32 v41, v56, s[8:9] offset:4
	global_load_b32 v44, v57, s[8:9]
	global_load_b32 v45, v57, s[8:9] offset:4
	global_load_b32 v48, v58, s[8:9]
	global_load_b32 v49, v58, s[8:9] offset:4
	global_load_b32 v52, v59, s[8:9]
	global_load_b32 v53, v59, s[8:9] offset:4
	s_wait_loadcnt 0x7
	v_add_f32_e32 v40, v64, v40
	global_store_b32 v56, v40, s[8:9]
	s_wait_loadcnt 0x6
	v_add_f32_e32 v41, v65, v41
	global_store_b32 v56, v41, s[8:9] offset:4
	s_wait_loadcnt 0x5
	v_add_f32_e32 v44, v76, v44
	global_store_b32 v57, v44, s[8:9]
	s_wait_loadcnt 0x4
	v_add_f32_e32 v45, v77, v45
	global_store_b32 v57, v45, s[8:9] offset:4
	s_wait_loadcnt 0x3
	v_add_f32_e32 v48, v88, v48
	global_store_b32 v58, v48, s[8:9]
	s_wait_loadcnt 0x2
	v_add_f32_e32 v49, v89, v49
	global_store_b32 v58, v49, s[8:9] offset:4
	s_wait_loadcnt 0x1
	v_add_f32_e32 v52, v100, v52
	global_store_b32 v59, v52, s[8:9]
	s_wait_loadcnt 0x0
	v_add_f32_e32 v53, v101, v53
	global_store_b32 v59, v53, s[8:9] offset:4
	s_wait_storecnt 0x0
	s_branch .Lrx_b4_p0_next
	.Lrx_b4_p0_single:
	global_load_b32 v40, v56, s[8:9]
	global_load_b32 v44, v57, s[8:9]
	global_load_b32 v48, v58, s[8:9]
	global_load_b32 v52, v59, s[8:9]
	s_wait_loadcnt 0x3
	v_add_f32_e32 v40, v40, v64
	global_store_b32 v56, v40, s[8:9]
	s_wait_loadcnt 0x2
	v_add_f32_e32 v44, v44, v76
	global_store_b32 v57, v44, s[8:9]
	s_wait_loadcnt 0x1
	v_add_f32_e32 v48, v48, v88
	global_store_b32 v58, v48, s[8:9]
	s_wait_loadcnt 0x0
	v_add_f32_e32 v52, v52, v100
	global_store_b32 v59, v52, s[8:9]
	s_wait_storecnt 0x0
	s_branch .Lrx_b4_stored
	.Lrx_b4_p0_next:
	s_add_co_i32 s13, s12, 2
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cbranch_scc0 .Lrx_b4_stored
	s_add_co_i32 s13, s12, 3
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cbranch_scc0 .Lrx_b4_p1_single
	global_load_b32 v42, v56, s[8:9] offset:8
	global_load_b32 v43, v56, s[8:9] offset:12
	global_load_b32 v46, v57, s[8:9] offset:8
	global_load_b32 v47, v57, s[8:9] offset:12
	global_load_b32 v50, v58, s[8:9] offset:8
	global_load_b32 v51, v58, s[8:9] offset:12
	global_load_b32 v54, v59, s[8:9] offset:8
	global_load_b32 v55, v59, s[8:9] offset:12
	s_wait_loadcnt 0x7
	v_add_f32_e32 v42, v66, v42
	global_store_b32 v56, v42, s[8:9] offset:8
	s_wait_loadcnt 0x6
	v_add_f32_e32 v43, v67, v43
	global_store_b32 v56, v43, s[8:9] offset:12
	s_wait_loadcnt 0x5
	v_add_f32_e32 v46, v78, v46
	global_store_b32 v57, v46, s[8:9] offset:8
	s_wait_loadcnt 0x4
	v_add_f32_e32 v47, v79, v47
	global_store_b32 v57, v47, s[8:9] offset:12
	s_wait_loadcnt 0x3
	v_add_f32_e32 v50, v90, v50
	global_store_b32 v58, v50, s[8:9] offset:8
	s_wait_loadcnt 0x2
	v_add_f32_e32 v51, v91, v51
	global_store_b32 v58, v51, s[8:9] offset:12
	s_wait_loadcnt 0x1
	v_add_f32_e32 v54, v102, v54
	global_store_b32 v59, v54, s[8:9] offset:8
	s_wait_loadcnt 0x0
	v_add_f32_e32 v55, v103, v55
	global_store_b32 v59, v55, s[8:9] offset:12
	s_wait_storecnt 0x0
	s_branch .Lrx_b4_stored
	.Lrx_b4_p1_single:
	global_load_b32 v42, v56, s[8:9] offset:8
	global_load_b32 v46, v57, s[8:9] offset:8
	global_load_b32 v50, v58, s[8:9] offset:8
	global_load_b32 v54, v59, s[8:9] offset:8
	s_wait_loadcnt 0x3
	v_add_f32_e32 v42, v42, v66
	global_store_b32 v56, v42, s[8:9] offset:8
	s_wait_loadcnt 0x2
	v_add_f32_e32 v46, v46, v78
	global_store_b32 v57, v46, s[8:9] offset:8
	s_wait_loadcnt 0x1
	v_add_f32_e32 v50, v50, v90
	global_store_b32 v58, v50, s[8:9] offset:8
	s_wait_loadcnt 0x0
	v_add_f32_e32 v54, v54, v102
	global_store_b32 v59, v54, s[8:9] offset:8
	s_wait_storecnt 0x0
	s_branch .Lrx_b4_stored
	.Lrx_b4_stored:
	s_branch .Lrx_end
	.Lrx_b5:
	s_mul_i32 s12, ttmp9, 4
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_i32 s12, s10
	s_cbranch_scc1 .Lrx_end
	s_mov_b32 s13, s12
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s13, s13, s19
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v8, s13
	v_add_nc_u32_e32 v4, s13, v1
	s_add_co_i32 s13, s12, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cselect_b32 s13, s13, s12
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s13, s13, s19
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v9, s13
	v_add_nc_u32_e32 v5, s13, v1
	s_add_co_i32 s13, s12, 2
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cselect_b32 s13, s13, s12
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s13, s13, s19
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v10, s13
	v_add_nc_u32_e32 v6, s13, v1
	s_add_co_i32 s13, s12, 3
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cselect_b32 s13, s13, s12
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s13, s13, s19
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v11, s13
	v_add_nc_u32_e32 v7, s13, v1
	v_mov_b32_e32 v64, 0
	v_mov_b32_e32 v65, 0
	v_mov_b32_e32 v66, 0
	v_mov_b32_e32 v67, 0
	v_mov_b32_e32 v68, 0
	v_mov_b32_e32 v69, 0
	v_mov_b32_e32 v70, 0
	v_mov_b32_e32 v71, 0
	v_mov_b32_e32 v72, 0
	v_mov_b32_e32 v73, 0
	v_mov_b32_e32 v74, 0
	v_mov_b32_e32 v75, 0
	v_mov_b32_e32 v76, 0
	v_mov_b32_e32 v77, 0
	v_mov_b32_e32 v78, 0
	v_mov_b32_e32 v79, 0
	v_mov_b32_e32 v80, 0
	v_mov_b32_e32 v81, 0
	v_mov_b32_e32 v82, 0
	v_mov_b32_e32 v83, 0
	v_mov_b32_e32 v84, 0
	v_mov_b32_e32 v85, 0
	v_mov_b32_e32 v86, 0
	v_mov_b32_e32 v87, 0
	v_mov_b32_e32 v88, 0
	v_mov_b32_e32 v89, 0
	v_mov_b32_e32 v90, 0
	v_mov_b32_e32 v91, 0
	v_mov_b32_e32 v92, 0
	v_mov_b32_e32 v93, 0
	v_mov_b32_e32 v94, 0
	v_mov_b32_e32 v95, 0
	v_mov_b32_e32 v96, 0
	v_mov_b32_e32 v97, 0
	v_mov_b32_e32 v98, 0
	v_mov_b32_e32 v99, 0
	v_mov_b32_e32 v100, 0
	v_mov_b32_e32 v101, 0
	v_mov_b32_e32 v102, 0
	v_mov_b32_e32 v103, 0
	v_mov_b32_e32 v104, 0
	v_mov_b32_e32 v105, 0
	v_mov_b32_e32 v106, 0
	v_mov_b32_e32 v107, 0
	v_mov_b32_e32 v108, 0
	v_mov_b32_e32 v109, 0
	v_mov_b32_e32 v110, 0
	v_mov_b32_e32 v111, 0
	v_mov_b32_e32 v112, 0
	v_mov_b32_e32 v113, 0
	v_mov_b32_e32 v114, 0
	v_mov_b32_e32 v115, 0
	v_mov_b32_e32 v116, 0
	v_mov_b32_e32 v117, 0
	v_mov_b32_e32 v118, 0
	v_mov_b32_e32 v119, 0
	v_mov_b32_e32 v120, 0
	v_mov_b32_e32 v121, 0
	v_mov_b32_e32 v122, 0
	v_mov_b32_e32 v123, 0
	s_add_co_i32 s2, s14, 3
	s_lshr_b32 s2, s2, 2
	s_cmp_eq_u32 s2, 0
	s_cbranch_scc1 .Lrx_b5_s0_done
	s_mov_b32 s15, 0
	v_lshlrev_b32_e32 v2, 5, v0
	buffer_load_b64 v[12:13], v8, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v20, v4, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[14:15], v9, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v21, v5, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[16:17], v10, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v22, v6, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[18:19], v11, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v23, v7, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	global_load_b128 v[125:128], v2, s[28:29]
	global_load_b128 v[129:132], v2, s[28:29] offset:16
	global_load_b128 v[133:136], v2, s[30:31]
	global_load_b128 v[137:140], v2, s[30:31] offset:16
	global_load_b128 v[141:144], v2, s[32:33]
	global_load_b128 v[145:148], v2, s[32:33] offset:16
	global_load_b128 v[149:152], v2, s[34:35]
	global_load_b128 v[153:156], v2, s[34:35] offset:16
	global_load_b128 v[157:160], v2, s[36:37]
	global_load_b128 v[161:164], v2, s[36:37] offset:16
	.Lrx_b5_s0:
	s_cmp_gt_u32 s2, 1
	s_cselect_b32 s17, 0x220, 0
	s_cselect_b32 s18, 0x1000, 0
	s_add_co_i32 s16, s15, s17
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v3, s18, v2
	v_cmp_gt_u32_e32 vcc_lo, 16, v0
	s_wait_loadcnt 0x11
	v_cndmask_b32_e32 v12, v13, v12, vcc_lo
	s_wait_loadcnt 0x10
	v_and_b32_e32 v30, 0xf0f0f0f, v20
	v_lshrrev_b32_e32 v31, 4, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v31, 0xf0f0f0f, v31
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v24, v30
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v26, v30
	v_cvt_f32_ubyte2_e32 v28, v30
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v25, v31
	v_cvt_f32_ubyte1_e32 v27, v31
	v_cvt_f32_ubyte2_e32 v29, v31
	v_cvt_f32_ubyte3_e32 v30, v30
	v_cvt_f32_ubyte3_e32 v31, v31
	v_fma_mix_f32 v24, v12, v24, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v25, v12, v25, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v26, v12, v26, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v27, v12, v27, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v28, v12, v28, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v29, v12, v29, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v30, v12, v30, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v31, v12, v31, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xf
	v_cndmask_b32_e32 v14, v15, v14, vcc_lo
	s_wait_loadcnt 0xe
	v_and_b32_e32 v40, 0xf0f0f0f, v21
	v_lshrrev_b32_e32 v41, 4, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v41, 0xf0f0f0f, v41
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v34, v40
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v36, v40
	v_cvt_f32_ubyte2_e32 v38, v40
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v35, v41
	v_cvt_f32_ubyte1_e32 v37, v41
	v_cvt_f32_ubyte2_e32 v39, v41
	v_cvt_f32_ubyte3_e32 v40, v40
	v_cvt_f32_ubyte3_e32 v41, v41
	v_fma_mix_f32 v34, v14, v34, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v35, v14, v35, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v36, v14, v36, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v37, v14, v37, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v38, v14, v38, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v39, v14, v39, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v40, v14, v40, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v41, v14, v41, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xd
	v_cndmask_b32_e32 v16, v17, v16, vcc_lo
	s_wait_loadcnt 0xc
	v_and_b32_e32 v50, 0xf0f0f0f, v22
	v_lshrrev_b32_e32 v51, 4, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v51, 0xf0f0f0f, v51
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v44, v50
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v46, v50
	v_cvt_f32_ubyte2_e32 v48, v50
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v45, v51
	v_cvt_f32_ubyte1_e32 v47, v51
	v_cvt_f32_ubyte2_e32 v49, v51
	v_cvt_f32_ubyte3_e32 v50, v50
	v_cvt_f32_ubyte3_e32 v51, v51
	v_fma_mix_f32 v44, v16, v44, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v16, v45, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v16, v46, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v16, v47, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v48, v16, v48, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v49, v16, v49, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v16, v50, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v16, v51, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xb
	v_cndmask_b32_e32 v18, v19, v18, vcc_lo
	s_wait_loadcnt 0xa
	v_and_b32_e32 v60, 0xf0f0f0f, v23
	v_lshrrev_b32_e32 v61, 4, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v61, 0xf0f0f0f, v61
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v54, v60
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v56, v60
	v_cvt_f32_ubyte2_e32 v58, v60
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v55, v61
	v_cvt_f32_ubyte1_e32 v57, v61
	v_cvt_f32_ubyte2_e32 v59, v61
	v_cvt_f32_ubyte3_e32 v60, v60
	v_cvt_f32_ubyte3_e32 v61, v61
	v_fma_mix_f32 v54, v18, v54, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v18, v55, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v18, v56, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v18, v57, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v58, v18, v58, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v59, v18, v59, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v18, v60, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v18, v61, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	buffer_load_b64 v[12:13], v8, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v20, v4, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[14:15], v9, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v21, v5, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[16:17], v10, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v22, v6, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[18:19], v11, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v23, v7, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	s_wait_loadcnt 0x11
	v_dual_mul_f32 v32, v24, v125 :: v_dual_mul_f32 v33, v35, v126
	v_dual_mul_f32 v42, v44, v125 :: v_dual_mul_f32 v43, v55, v126
	s_wait_loadcnt 0xf
	v_dual_mul_f32 v52, v24, v133 :: v_dual_mul_f32 v53, v35, v134
	v_dual_mul_f32 v62, v44, v133 :: v_dual_mul_f32 v63, v55, v134
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v25, v126 :: v_dual_fmac_f32 v33, v34, v125
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v45, v126 :: v_dual_fmac_f32 v43, v54, v125
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v25, v134 :: v_dual_fmac_f32 v53, v34, v133
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v45, v134 :: v_dual_fmac_f32 v63, v54, v133
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v26, v127 :: v_dual_fmac_f32 v33, v36, v127
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v46, v127 :: v_dual_fmac_f32 v43, v56, v127
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v26, v135 :: v_dual_fmac_f32 v53, v36, v135
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v46, v135 :: v_dual_fmac_f32 v63, v56, v135
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v27, v128 :: v_dual_fmac_f32 v33, v37, v128
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v47, v128 :: v_dual_fmac_f32 v43, v57, v128
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v27, v136 :: v_dual_fmac_f32 v53, v37, v136
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v47, v136 :: v_dual_fmac_f32 v63, v57, v136
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v28, v129 :: v_dual_fmac_f32 v33, v38, v129
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v48, v129 :: v_dual_fmac_f32 v43, v58, v129
	s_wait_loadcnt 0xe
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v28, v137 :: v_dual_fmac_f32 v53, v38, v137
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v48, v137 :: v_dual_fmac_f32 v63, v58, v137
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v29, v130 :: v_dual_fmac_f32 v33, v39, v130
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v49, v130 :: v_dual_fmac_f32 v43, v59, v130
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v29, v138 :: v_dual_fmac_f32 v53, v39, v138
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v49, v138 :: v_dual_fmac_f32 v63, v59, v138
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v30, v131 :: v_dual_fmac_f32 v33, v40, v131
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v50, v131 :: v_dual_fmac_f32 v43, v60, v131
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v30, v139 :: v_dual_fmac_f32 v53, v40, v139
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v50, v139 :: v_dual_fmac_f32 v63, v60, v139
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v31, v132 :: v_dual_fmac_f32 v33, v41, v132
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v51, v132 :: v_dual_fmac_f32 v43, v61, v132
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v31, v140 :: v_dual_fmac_f32 v53, v41, v140
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v51, v140 :: v_dual_fmac_f32 v63, v61, v140
	global_load_b128 v[125:128], v3, s[28:29]
	global_load_b128 v[129:132], v3, s[28:29] offset:16
	global_load_b128 v[133:136], v3, s[30:31]
	global_load_b128 v[137:140], v3, s[30:31] offset:16
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v64, v32, v64 :: v_dual_add_f32 v65, v65, v33
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v66, v42, v66 :: v_dual_add_f32 v67, v67, v43
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v76, v52, v76 :: v_dual_add_f32 v77, v77, v53
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v78, v62, v78 :: v_dual_add_f32 v79, v79, v63
	s_wait_loadcnt 0x11
	v_dual_mul_f32 v32, v24, v141 :: v_dual_mul_f32 v33, v35, v142
	v_dual_mul_f32 v42, v44, v141 :: v_dual_mul_f32 v43, v55, v142
	s_wait_loadcnt 0xf
	v_dual_mul_f32 v52, v24, v149 :: v_dual_mul_f32 v53, v35, v150
	v_dual_mul_f32 v62, v44, v149 :: v_dual_mul_f32 v63, v55, v150
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v25, v142 :: v_dual_fmac_f32 v33, v34, v141
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v45, v142 :: v_dual_fmac_f32 v43, v54, v141
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v25, v150 :: v_dual_fmac_f32 v53, v34, v149
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v45, v150 :: v_dual_fmac_f32 v63, v54, v149
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v26, v143 :: v_dual_fmac_f32 v33, v36, v143
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v46, v143 :: v_dual_fmac_f32 v43, v56, v143
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v26, v151 :: v_dual_fmac_f32 v53, v36, v151
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v46, v151 :: v_dual_fmac_f32 v63, v56, v151
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v27, v144 :: v_dual_fmac_f32 v33, v37, v144
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v47, v144 :: v_dual_fmac_f32 v43, v57, v144
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v27, v152 :: v_dual_fmac_f32 v53, v37, v152
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v47, v152 :: v_dual_fmac_f32 v63, v57, v152
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v28, v145 :: v_dual_fmac_f32 v33, v38, v145
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v48, v145 :: v_dual_fmac_f32 v43, v58, v145
	s_wait_loadcnt 0xe
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v28, v153 :: v_dual_fmac_f32 v53, v38, v153
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v48, v153 :: v_dual_fmac_f32 v63, v58, v153
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v29, v146 :: v_dual_fmac_f32 v33, v39, v146
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v49, v146 :: v_dual_fmac_f32 v43, v59, v146
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v29, v154 :: v_dual_fmac_f32 v53, v39, v154
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v49, v154 :: v_dual_fmac_f32 v63, v59, v154
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v30, v147 :: v_dual_fmac_f32 v33, v40, v147
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v50, v147 :: v_dual_fmac_f32 v43, v60, v147
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v30, v155 :: v_dual_fmac_f32 v53, v40, v155
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v50, v155 :: v_dual_fmac_f32 v63, v60, v155
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v31, v148 :: v_dual_fmac_f32 v33, v41, v148
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v51, v148 :: v_dual_fmac_f32 v43, v61, v148
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v31, v156 :: v_dual_fmac_f32 v53, v41, v156
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v51, v156 :: v_dual_fmac_f32 v63, v61, v156
	global_load_b128 v[141:144], v3, s[32:33]
	global_load_b128 v[145:148], v3, s[32:33] offset:16
	global_load_b128 v[149:152], v3, s[34:35]
	global_load_b128 v[153:156], v3, s[34:35] offset:16
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v88, v32, v88 :: v_dual_add_f32 v89, v89, v33
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v90, v42, v90 :: v_dual_add_f32 v91, v91, v43
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v100, v52, v100 :: v_dual_add_f32 v101, v101, v53
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v102, v62, v102 :: v_dual_add_f32 v103, v103, v63
	s_wait_loadcnt 0x11
	v_dual_mul_f32 v32, v24, v157 :: v_dual_mul_f32 v33, v35, v158
	v_dual_mul_f32 v42, v44, v157 :: v_dual_mul_f32 v43, v55, v158
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v25, v158 :: v_dual_fmac_f32 v33, v34, v157
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v45, v158 :: v_dual_fmac_f32 v43, v54, v157
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v26, v159 :: v_dual_fmac_f32 v33, v36, v159
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v46, v159 :: v_dual_fmac_f32 v43, v56, v159
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v27, v160 :: v_dual_fmac_f32 v33, v37, v160
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v47, v160 :: v_dual_fmac_f32 v43, v57, v160
	s_wait_loadcnt 0x10
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v28, v161 :: v_dual_fmac_f32 v33, v38, v161
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v48, v161 :: v_dual_fmac_f32 v43, v58, v161
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v29, v162 :: v_dual_fmac_f32 v33, v39, v162
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v49, v162 :: v_dual_fmac_f32 v43, v59, v162
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v30, v163 :: v_dual_fmac_f32 v33, v40, v163
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v50, v163 :: v_dual_fmac_f32 v43, v60, v163
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v31, v164 :: v_dual_fmac_f32 v33, v41, v164
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v51, v164 :: v_dual_fmac_f32 v43, v61, v164
	global_load_b128 v[157:160], v3, s[36:37]
	global_load_b128 v[161:164], v3, s[36:37] offset:16
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_add_f32 v112, v32, v112 :: v_dual_add_f32 v113, v113, v33
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_add_f32 v114, v42, v114 :: v_dual_add_f32 v115, v115, v43
	s_mov_b32 s15, s16
	v_mov_b32_e32 v2, v3
	s_add_co_i32 s2, s2, -1
	s_cmp_lg_u32 s2, 0
	s_cbranch_scc1 .Lrx_b5_s0
	s_wait_loadcnt 0x0
	.Lrx_b5_s0_done:
	s_add_co_i32 s2, s14, 2
	s_lshr_b32 s2, s2, 2
	s_cmp_eq_u32 s2, 0
	s_cbranch_scc1 .Lrx_b5_s1_done
	s_mov_b32 s15, 0x88
	v_lshlrev_b32_e32 v2, 5, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v2, 0x400, v2
	buffer_load_b64 v[12:13], v8, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v20, v4, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[14:15], v9, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v21, v5, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[16:17], v10, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v22, v6, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[18:19], v11, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v23, v7, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	global_load_b128 v[125:128], v2, s[28:29]
	global_load_b128 v[129:132], v2, s[28:29] offset:16
	global_load_b128 v[133:136], v2, s[30:31]
	global_load_b128 v[137:140], v2, s[30:31] offset:16
	global_load_b128 v[141:144], v2, s[32:33]
	global_load_b128 v[145:148], v2, s[32:33] offset:16
	global_load_b128 v[149:152], v2, s[34:35]
	global_load_b128 v[153:156], v2, s[34:35] offset:16
	global_load_b128 v[157:160], v2, s[36:37]
	global_load_b128 v[161:164], v2, s[36:37] offset:16
	.Lrx_b5_s1:
	s_cmp_gt_u32 s2, 1
	s_cselect_b32 s17, 0x220, 0
	s_cselect_b32 s18, 0x1000, 0
	s_add_co_i32 s16, s15, s17
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v3, s18, v2
	v_cmp_gt_u32_e32 vcc_lo, 16, v0
	s_wait_loadcnt 0x11
	v_cndmask_b32_e32 v12, v13, v12, vcc_lo
	s_wait_loadcnt 0x10
	v_and_b32_e32 v30, 0xf0f0f0f, v20
	v_lshrrev_b32_e32 v31, 4, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v31, 0xf0f0f0f, v31
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v24, v30
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v26, v30
	v_cvt_f32_ubyte2_e32 v28, v30
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v25, v31
	v_cvt_f32_ubyte1_e32 v27, v31
	v_cvt_f32_ubyte2_e32 v29, v31
	v_cvt_f32_ubyte3_e32 v30, v30
	v_cvt_f32_ubyte3_e32 v31, v31
	v_fma_mix_f32 v24, v12, v24, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v25, v12, v25, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v26, v12, v26, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v27, v12, v27, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v28, v12, v28, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v29, v12, v29, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v30, v12, v30, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v31, v12, v31, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xf
	v_cndmask_b32_e32 v14, v15, v14, vcc_lo
	s_wait_loadcnt 0xe
	v_and_b32_e32 v40, 0xf0f0f0f, v21
	v_lshrrev_b32_e32 v41, 4, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v41, 0xf0f0f0f, v41
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v34, v40
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v36, v40
	v_cvt_f32_ubyte2_e32 v38, v40
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v35, v41
	v_cvt_f32_ubyte1_e32 v37, v41
	v_cvt_f32_ubyte2_e32 v39, v41
	v_cvt_f32_ubyte3_e32 v40, v40
	v_cvt_f32_ubyte3_e32 v41, v41
	v_fma_mix_f32 v34, v14, v34, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v35, v14, v35, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v36, v14, v36, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v37, v14, v37, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v38, v14, v38, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v39, v14, v39, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v40, v14, v40, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v41, v14, v41, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xd
	v_cndmask_b32_e32 v16, v17, v16, vcc_lo
	s_wait_loadcnt 0xc
	v_and_b32_e32 v50, 0xf0f0f0f, v22
	v_lshrrev_b32_e32 v51, 4, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v51, 0xf0f0f0f, v51
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v44, v50
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v46, v50
	v_cvt_f32_ubyte2_e32 v48, v50
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v45, v51
	v_cvt_f32_ubyte1_e32 v47, v51
	v_cvt_f32_ubyte2_e32 v49, v51
	v_cvt_f32_ubyte3_e32 v50, v50
	v_cvt_f32_ubyte3_e32 v51, v51
	v_fma_mix_f32 v44, v16, v44, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v16, v45, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v16, v46, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v16, v47, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v48, v16, v48, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v49, v16, v49, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v16, v50, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v16, v51, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xb
	v_cndmask_b32_e32 v18, v19, v18, vcc_lo
	s_wait_loadcnt 0xa
	v_and_b32_e32 v60, 0xf0f0f0f, v23
	v_lshrrev_b32_e32 v61, 4, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v61, 0xf0f0f0f, v61
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v54, v60
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v56, v60
	v_cvt_f32_ubyte2_e32 v58, v60
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v55, v61
	v_cvt_f32_ubyte1_e32 v57, v61
	v_cvt_f32_ubyte2_e32 v59, v61
	v_cvt_f32_ubyte3_e32 v60, v60
	v_cvt_f32_ubyte3_e32 v61, v61
	v_fma_mix_f32 v54, v18, v54, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v18, v55, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v18, v56, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v18, v57, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v58, v18, v58, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v59, v18, v59, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v18, v60, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v18, v61, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	buffer_load_b64 v[12:13], v8, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v20, v4, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[14:15], v9, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v21, v5, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[16:17], v10, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v22, v6, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[18:19], v11, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v23, v7, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	s_wait_loadcnt 0x11
	v_dual_mul_f32 v32, v24, v125 :: v_dual_mul_f32 v33, v35, v126
	v_dual_mul_f32 v42, v44, v125 :: v_dual_mul_f32 v43, v55, v126
	s_wait_loadcnt 0xf
	v_dual_mul_f32 v52, v24, v133 :: v_dual_mul_f32 v53, v35, v134
	v_dual_mul_f32 v62, v44, v133 :: v_dual_mul_f32 v63, v55, v134
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v25, v126 :: v_dual_fmac_f32 v33, v34, v125
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v45, v126 :: v_dual_fmac_f32 v43, v54, v125
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v25, v134 :: v_dual_fmac_f32 v53, v34, v133
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v45, v134 :: v_dual_fmac_f32 v63, v54, v133
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v26, v127 :: v_dual_fmac_f32 v33, v36, v127
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v46, v127 :: v_dual_fmac_f32 v43, v56, v127
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v26, v135 :: v_dual_fmac_f32 v53, v36, v135
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v46, v135 :: v_dual_fmac_f32 v63, v56, v135
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v27, v128 :: v_dual_fmac_f32 v33, v37, v128
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v47, v128 :: v_dual_fmac_f32 v43, v57, v128
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v27, v136 :: v_dual_fmac_f32 v53, v37, v136
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v47, v136 :: v_dual_fmac_f32 v63, v57, v136
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v28, v129 :: v_dual_fmac_f32 v33, v38, v129
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v48, v129 :: v_dual_fmac_f32 v43, v58, v129
	s_wait_loadcnt 0xe
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v28, v137 :: v_dual_fmac_f32 v53, v38, v137
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v48, v137 :: v_dual_fmac_f32 v63, v58, v137
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v29, v130 :: v_dual_fmac_f32 v33, v39, v130
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v49, v130 :: v_dual_fmac_f32 v43, v59, v130
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v29, v138 :: v_dual_fmac_f32 v53, v39, v138
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v49, v138 :: v_dual_fmac_f32 v63, v59, v138
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v30, v131 :: v_dual_fmac_f32 v33, v40, v131
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v50, v131 :: v_dual_fmac_f32 v43, v60, v131
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v30, v139 :: v_dual_fmac_f32 v53, v40, v139
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v50, v139 :: v_dual_fmac_f32 v63, v60, v139
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v31, v132 :: v_dual_fmac_f32 v33, v41, v132
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v51, v132 :: v_dual_fmac_f32 v43, v61, v132
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v31, v140 :: v_dual_fmac_f32 v53, v41, v140
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v51, v140 :: v_dual_fmac_f32 v63, v61, v140
	global_load_b128 v[125:128], v3, s[28:29]
	global_load_b128 v[129:132], v3, s[28:29] offset:16
	global_load_b128 v[133:136], v3, s[30:31]
	global_load_b128 v[137:140], v3, s[30:31] offset:16
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v68, v32, v68 :: v_dual_add_f32 v69, v69, v33
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v70, v42, v70 :: v_dual_add_f32 v71, v71, v43
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v80, v52, v80 :: v_dual_add_f32 v81, v81, v53
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v82, v62, v82 :: v_dual_add_f32 v83, v83, v63
	s_wait_loadcnt 0x11
	v_dual_mul_f32 v32, v24, v141 :: v_dual_mul_f32 v33, v35, v142
	v_dual_mul_f32 v42, v44, v141 :: v_dual_mul_f32 v43, v55, v142
	s_wait_loadcnt 0xf
	v_dual_mul_f32 v52, v24, v149 :: v_dual_mul_f32 v53, v35, v150
	v_dual_mul_f32 v62, v44, v149 :: v_dual_mul_f32 v63, v55, v150
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v25, v142 :: v_dual_fmac_f32 v33, v34, v141
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v45, v142 :: v_dual_fmac_f32 v43, v54, v141
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v25, v150 :: v_dual_fmac_f32 v53, v34, v149
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v45, v150 :: v_dual_fmac_f32 v63, v54, v149
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v26, v143 :: v_dual_fmac_f32 v33, v36, v143
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v46, v143 :: v_dual_fmac_f32 v43, v56, v143
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v26, v151 :: v_dual_fmac_f32 v53, v36, v151
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v46, v151 :: v_dual_fmac_f32 v63, v56, v151
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v27, v144 :: v_dual_fmac_f32 v33, v37, v144
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v47, v144 :: v_dual_fmac_f32 v43, v57, v144
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v27, v152 :: v_dual_fmac_f32 v53, v37, v152
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v47, v152 :: v_dual_fmac_f32 v63, v57, v152
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v28, v145 :: v_dual_fmac_f32 v33, v38, v145
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v48, v145 :: v_dual_fmac_f32 v43, v58, v145
	s_wait_loadcnt 0xe
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v28, v153 :: v_dual_fmac_f32 v53, v38, v153
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v48, v153 :: v_dual_fmac_f32 v63, v58, v153
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v29, v146 :: v_dual_fmac_f32 v33, v39, v146
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v49, v146 :: v_dual_fmac_f32 v43, v59, v146
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v29, v154 :: v_dual_fmac_f32 v53, v39, v154
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v49, v154 :: v_dual_fmac_f32 v63, v59, v154
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v30, v147 :: v_dual_fmac_f32 v33, v40, v147
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v50, v147 :: v_dual_fmac_f32 v43, v60, v147
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v30, v155 :: v_dual_fmac_f32 v53, v40, v155
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v50, v155 :: v_dual_fmac_f32 v63, v60, v155
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v31, v148 :: v_dual_fmac_f32 v33, v41, v148
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v51, v148 :: v_dual_fmac_f32 v43, v61, v148
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v31, v156 :: v_dual_fmac_f32 v53, v41, v156
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v51, v156 :: v_dual_fmac_f32 v63, v61, v156
	global_load_b128 v[141:144], v3, s[32:33]
	global_load_b128 v[145:148], v3, s[32:33] offset:16
	global_load_b128 v[149:152], v3, s[34:35]
	global_load_b128 v[153:156], v3, s[34:35] offset:16
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v92, v32, v92 :: v_dual_add_f32 v93, v93, v33
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v94, v42, v94 :: v_dual_add_f32 v95, v95, v43
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v104, v52, v104 :: v_dual_add_f32 v105, v105, v53
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v106, v62, v106 :: v_dual_add_f32 v107, v107, v63
	s_wait_loadcnt 0x11
	v_dual_mul_f32 v32, v24, v157 :: v_dual_mul_f32 v33, v35, v158
	v_dual_mul_f32 v42, v44, v157 :: v_dual_mul_f32 v43, v55, v158
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v25, v158 :: v_dual_fmac_f32 v33, v34, v157
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v45, v158 :: v_dual_fmac_f32 v43, v54, v157
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v26, v159 :: v_dual_fmac_f32 v33, v36, v159
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v46, v159 :: v_dual_fmac_f32 v43, v56, v159
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v27, v160 :: v_dual_fmac_f32 v33, v37, v160
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v47, v160 :: v_dual_fmac_f32 v43, v57, v160
	s_wait_loadcnt 0x10
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v28, v161 :: v_dual_fmac_f32 v33, v38, v161
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v48, v161 :: v_dual_fmac_f32 v43, v58, v161
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v29, v162 :: v_dual_fmac_f32 v33, v39, v162
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v49, v162 :: v_dual_fmac_f32 v43, v59, v162
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v30, v163 :: v_dual_fmac_f32 v33, v40, v163
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v50, v163 :: v_dual_fmac_f32 v43, v60, v163
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v31, v164 :: v_dual_fmac_f32 v33, v41, v164
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v51, v164 :: v_dual_fmac_f32 v43, v61, v164
	global_load_b128 v[157:160], v3, s[36:37]
	global_load_b128 v[161:164], v3, s[36:37] offset:16
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_add_f32 v116, v32, v116 :: v_dual_add_f32 v117, v117, v33
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_add_f32 v118, v42, v118 :: v_dual_add_f32 v119, v119, v43
	s_mov_b32 s15, s16
	v_mov_b32_e32 v2, v3
	s_add_co_i32 s2, s2, -1
	s_cmp_lg_u32 s2, 0
	s_cbranch_scc1 .Lrx_b5_s1
	s_wait_loadcnt 0x0
	.Lrx_b5_s1_done:
	v_dual_add_f32 v64, v64, v68 :: v_dual_add_f32 v65, v65, v69
	v_dual_add_f32 v66, v66, v70 :: v_dual_add_f32 v67, v67, v71
	v_dual_add_f32 v76, v76, v80 :: v_dual_add_f32 v77, v77, v81
	v_dual_add_f32 v78, v78, v82 :: v_dual_add_f32 v79, v79, v83
	v_dual_add_f32 v88, v88, v92 :: v_dual_add_f32 v89, v89, v93
	v_dual_add_f32 v90, v90, v94 :: v_dual_add_f32 v91, v91, v95
	v_dual_add_f32 v100, v100, v104 :: v_dual_add_f32 v101, v101, v105
	v_dual_add_f32 v102, v102, v106 :: v_dual_add_f32 v103, v103, v107
	v_dual_add_f32 v112, v112, v116 :: v_dual_add_f32 v113, v113, v117
	v_dual_add_f32 v114, v114, v118 :: v_dual_add_f32 v115, v115, v119
	v_mov_b32_e32 v68, 0
	v_mov_b32_e32 v69, 0
	v_mov_b32_e32 v70, 0
	v_mov_b32_e32 v71, 0
	v_mov_b32_e32 v80, 0
	v_mov_b32_e32 v81, 0
	v_mov_b32_e32 v82, 0
	v_mov_b32_e32 v83, 0
	v_mov_b32_e32 v92, 0
	v_mov_b32_e32 v93, 0
	v_mov_b32_e32 v94, 0
	v_mov_b32_e32 v95, 0
	v_mov_b32_e32 v104, 0
	v_mov_b32_e32 v105, 0
	v_mov_b32_e32 v106, 0
	v_mov_b32_e32 v107, 0
	v_mov_b32_e32 v116, 0
	v_mov_b32_e32 v117, 0
	v_mov_b32_e32 v118, 0
	v_mov_b32_e32 v119, 0
	s_add_co_i32 s2, s14, 1
	s_lshr_b32 s2, s2, 2
	s_cmp_eq_u32 s2, 0
	s_cbranch_scc1 .Lrx_b5_s2_done
	s_mov_b32 s15, 0x110
	v_lshlrev_b32_e32 v2, 5, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v2, 0x800, v2
	buffer_load_b64 v[12:13], v8, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v20, v4, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[14:15], v9, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v21, v5, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[16:17], v10, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v22, v6, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[18:19], v11, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v23, v7, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	global_load_b128 v[125:128], v2, s[28:29]
	global_load_b128 v[129:132], v2, s[28:29] offset:16
	global_load_b128 v[133:136], v2, s[30:31]
	global_load_b128 v[137:140], v2, s[30:31] offset:16
	global_load_b128 v[141:144], v2, s[32:33]
	global_load_b128 v[145:148], v2, s[32:33] offset:16
	global_load_b128 v[149:152], v2, s[34:35]
	global_load_b128 v[153:156], v2, s[34:35] offset:16
	global_load_b128 v[157:160], v2, s[36:37]
	global_load_b128 v[161:164], v2, s[36:37] offset:16
	.Lrx_b5_s2:
	s_cmp_gt_u32 s2, 1
	s_cselect_b32 s17, 0x220, 0
	s_cselect_b32 s18, 0x1000, 0
	s_add_co_i32 s16, s15, s17
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v3, s18, v2
	v_cmp_gt_u32_e32 vcc_lo, 16, v0
	s_wait_loadcnt 0x11
	v_cndmask_b32_e32 v12, v13, v12, vcc_lo
	s_wait_loadcnt 0x10
	v_and_b32_e32 v30, 0xf0f0f0f, v20
	v_lshrrev_b32_e32 v31, 4, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v31, 0xf0f0f0f, v31
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v24, v30
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v26, v30
	v_cvt_f32_ubyte2_e32 v28, v30
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v25, v31
	v_cvt_f32_ubyte1_e32 v27, v31
	v_cvt_f32_ubyte2_e32 v29, v31
	v_cvt_f32_ubyte3_e32 v30, v30
	v_cvt_f32_ubyte3_e32 v31, v31
	v_fma_mix_f32 v24, v12, v24, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v25, v12, v25, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v26, v12, v26, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v27, v12, v27, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v28, v12, v28, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v29, v12, v29, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v30, v12, v30, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v31, v12, v31, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xf
	v_cndmask_b32_e32 v14, v15, v14, vcc_lo
	s_wait_loadcnt 0xe
	v_and_b32_e32 v40, 0xf0f0f0f, v21
	v_lshrrev_b32_e32 v41, 4, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v41, 0xf0f0f0f, v41
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v34, v40
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v36, v40
	v_cvt_f32_ubyte2_e32 v38, v40
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v35, v41
	v_cvt_f32_ubyte1_e32 v37, v41
	v_cvt_f32_ubyte2_e32 v39, v41
	v_cvt_f32_ubyte3_e32 v40, v40
	v_cvt_f32_ubyte3_e32 v41, v41
	v_fma_mix_f32 v34, v14, v34, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v35, v14, v35, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v36, v14, v36, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v37, v14, v37, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v38, v14, v38, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v39, v14, v39, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v40, v14, v40, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v41, v14, v41, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xd
	v_cndmask_b32_e32 v16, v17, v16, vcc_lo
	s_wait_loadcnt 0xc
	v_and_b32_e32 v50, 0xf0f0f0f, v22
	v_lshrrev_b32_e32 v51, 4, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v51, 0xf0f0f0f, v51
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v44, v50
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v46, v50
	v_cvt_f32_ubyte2_e32 v48, v50
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v45, v51
	v_cvt_f32_ubyte1_e32 v47, v51
	v_cvt_f32_ubyte2_e32 v49, v51
	v_cvt_f32_ubyte3_e32 v50, v50
	v_cvt_f32_ubyte3_e32 v51, v51
	v_fma_mix_f32 v44, v16, v44, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v16, v45, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v16, v46, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v16, v47, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v48, v16, v48, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v49, v16, v49, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v16, v50, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v16, v51, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xb
	v_cndmask_b32_e32 v18, v19, v18, vcc_lo
	s_wait_loadcnt 0xa
	v_and_b32_e32 v60, 0xf0f0f0f, v23
	v_lshrrev_b32_e32 v61, 4, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v61, 0xf0f0f0f, v61
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v54, v60
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v56, v60
	v_cvt_f32_ubyte2_e32 v58, v60
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v55, v61
	v_cvt_f32_ubyte1_e32 v57, v61
	v_cvt_f32_ubyte2_e32 v59, v61
	v_cvt_f32_ubyte3_e32 v60, v60
	v_cvt_f32_ubyte3_e32 v61, v61
	v_fma_mix_f32 v54, v18, v54, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v18, v55, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v18, v56, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v18, v57, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v58, v18, v58, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v59, v18, v59, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v18, v60, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v18, v61, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	buffer_load_b64 v[12:13], v8, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v20, v4, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[14:15], v9, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v21, v5, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[16:17], v10, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v22, v6, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[18:19], v11, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v23, v7, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	s_wait_loadcnt 0x11
	v_dual_mul_f32 v32, v24, v125 :: v_dual_mul_f32 v33, v35, v126
	v_dual_mul_f32 v42, v44, v125 :: v_dual_mul_f32 v43, v55, v126
	s_wait_loadcnt 0xf
	v_dual_mul_f32 v52, v24, v133 :: v_dual_mul_f32 v53, v35, v134
	v_dual_mul_f32 v62, v44, v133 :: v_dual_mul_f32 v63, v55, v134
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v25, v126 :: v_dual_fmac_f32 v33, v34, v125
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v45, v126 :: v_dual_fmac_f32 v43, v54, v125
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v25, v134 :: v_dual_fmac_f32 v53, v34, v133
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v45, v134 :: v_dual_fmac_f32 v63, v54, v133
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v26, v127 :: v_dual_fmac_f32 v33, v36, v127
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v46, v127 :: v_dual_fmac_f32 v43, v56, v127
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v26, v135 :: v_dual_fmac_f32 v53, v36, v135
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v46, v135 :: v_dual_fmac_f32 v63, v56, v135
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v27, v128 :: v_dual_fmac_f32 v33, v37, v128
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v47, v128 :: v_dual_fmac_f32 v43, v57, v128
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v27, v136 :: v_dual_fmac_f32 v53, v37, v136
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v47, v136 :: v_dual_fmac_f32 v63, v57, v136
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v28, v129 :: v_dual_fmac_f32 v33, v38, v129
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v48, v129 :: v_dual_fmac_f32 v43, v58, v129
	s_wait_loadcnt 0xe
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v28, v137 :: v_dual_fmac_f32 v53, v38, v137
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v48, v137 :: v_dual_fmac_f32 v63, v58, v137
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v29, v130 :: v_dual_fmac_f32 v33, v39, v130
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v49, v130 :: v_dual_fmac_f32 v43, v59, v130
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v29, v138 :: v_dual_fmac_f32 v53, v39, v138
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v49, v138 :: v_dual_fmac_f32 v63, v59, v138
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v30, v131 :: v_dual_fmac_f32 v33, v40, v131
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v50, v131 :: v_dual_fmac_f32 v43, v60, v131
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v30, v139 :: v_dual_fmac_f32 v53, v40, v139
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v50, v139 :: v_dual_fmac_f32 v63, v60, v139
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v31, v132 :: v_dual_fmac_f32 v33, v41, v132
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v51, v132 :: v_dual_fmac_f32 v43, v61, v132
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v31, v140 :: v_dual_fmac_f32 v53, v41, v140
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v51, v140 :: v_dual_fmac_f32 v63, v61, v140
	global_load_b128 v[125:128], v3, s[28:29]
	global_load_b128 v[129:132], v3, s[28:29] offset:16
	global_load_b128 v[133:136], v3, s[30:31]
	global_load_b128 v[137:140], v3, s[30:31] offset:16
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v68, v32, v68 :: v_dual_add_f32 v69, v69, v33
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v70, v42, v70 :: v_dual_add_f32 v71, v71, v43
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v80, v52, v80 :: v_dual_add_f32 v81, v81, v53
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v82, v62, v82 :: v_dual_add_f32 v83, v83, v63
	s_wait_loadcnt 0x11
	v_dual_mul_f32 v32, v24, v141 :: v_dual_mul_f32 v33, v35, v142
	v_dual_mul_f32 v42, v44, v141 :: v_dual_mul_f32 v43, v55, v142
	s_wait_loadcnt 0xf
	v_dual_mul_f32 v52, v24, v149 :: v_dual_mul_f32 v53, v35, v150
	v_dual_mul_f32 v62, v44, v149 :: v_dual_mul_f32 v63, v55, v150
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v25, v142 :: v_dual_fmac_f32 v33, v34, v141
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v45, v142 :: v_dual_fmac_f32 v43, v54, v141
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v25, v150 :: v_dual_fmac_f32 v53, v34, v149
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v45, v150 :: v_dual_fmac_f32 v63, v54, v149
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v26, v143 :: v_dual_fmac_f32 v33, v36, v143
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v46, v143 :: v_dual_fmac_f32 v43, v56, v143
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v26, v151 :: v_dual_fmac_f32 v53, v36, v151
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v46, v151 :: v_dual_fmac_f32 v63, v56, v151
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v27, v144 :: v_dual_fmac_f32 v33, v37, v144
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v47, v144 :: v_dual_fmac_f32 v43, v57, v144
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v27, v152 :: v_dual_fmac_f32 v53, v37, v152
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v47, v152 :: v_dual_fmac_f32 v63, v57, v152
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v28, v145 :: v_dual_fmac_f32 v33, v38, v145
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v48, v145 :: v_dual_fmac_f32 v43, v58, v145
	s_wait_loadcnt 0xe
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v28, v153 :: v_dual_fmac_f32 v53, v38, v153
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v48, v153 :: v_dual_fmac_f32 v63, v58, v153
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v29, v146 :: v_dual_fmac_f32 v33, v39, v146
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v49, v146 :: v_dual_fmac_f32 v43, v59, v146
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v29, v154 :: v_dual_fmac_f32 v53, v39, v154
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v49, v154 :: v_dual_fmac_f32 v63, v59, v154
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v30, v147 :: v_dual_fmac_f32 v33, v40, v147
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v50, v147 :: v_dual_fmac_f32 v43, v60, v147
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v30, v155 :: v_dual_fmac_f32 v53, v40, v155
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v50, v155 :: v_dual_fmac_f32 v63, v60, v155
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v31, v148 :: v_dual_fmac_f32 v33, v41, v148
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v51, v148 :: v_dual_fmac_f32 v43, v61, v148
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v31, v156 :: v_dual_fmac_f32 v53, v41, v156
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v51, v156 :: v_dual_fmac_f32 v63, v61, v156
	global_load_b128 v[141:144], v3, s[32:33]
	global_load_b128 v[145:148], v3, s[32:33] offset:16
	global_load_b128 v[149:152], v3, s[34:35]
	global_load_b128 v[153:156], v3, s[34:35] offset:16
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v92, v32, v92 :: v_dual_add_f32 v93, v93, v33
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v94, v42, v94 :: v_dual_add_f32 v95, v95, v43
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v104, v52, v104 :: v_dual_add_f32 v105, v105, v53
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v106, v62, v106 :: v_dual_add_f32 v107, v107, v63
	s_wait_loadcnt 0x11
	v_dual_mul_f32 v32, v24, v157 :: v_dual_mul_f32 v33, v35, v158
	v_dual_mul_f32 v42, v44, v157 :: v_dual_mul_f32 v43, v55, v158
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v25, v158 :: v_dual_fmac_f32 v33, v34, v157
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v45, v158 :: v_dual_fmac_f32 v43, v54, v157
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v26, v159 :: v_dual_fmac_f32 v33, v36, v159
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v46, v159 :: v_dual_fmac_f32 v43, v56, v159
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v27, v160 :: v_dual_fmac_f32 v33, v37, v160
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v47, v160 :: v_dual_fmac_f32 v43, v57, v160
	s_wait_loadcnt 0x10
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v28, v161 :: v_dual_fmac_f32 v33, v38, v161
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v48, v161 :: v_dual_fmac_f32 v43, v58, v161
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v29, v162 :: v_dual_fmac_f32 v33, v39, v162
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v49, v162 :: v_dual_fmac_f32 v43, v59, v162
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v30, v163 :: v_dual_fmac_f32 v33, v40, v163
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v50, v163 :: v_dual_fmac_f32 v43, v60, v163
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v31, v164 :: v_dual_fmac_f32 v33, v41, v164
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v51, v164 :: v_dual_fmac_f32 v43, v61, v164
	global_load_b128 v[157:160], v3, s[36:37]
	global_load_b128 v[161:164], v3, s[36:37] offset:16
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_add_f32 v116, v32, v116 :: v_dual_add_f32 v117, v117, v33
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_add_f32 v118, v42, v118 :: v_dual_add_f32 v119, v119, v43
	s_mov_b32 s15, s16
	v_mov_b32_e32 v2, v3
	s_add_co_i32 s2, s2, -1
	s_cmp_lg_u32 s2, 0
	s_cbranch_scc1 .Lrx_b5_s2
	s_wait_loadcnt 0x0
	.Lrx_b5_s2_done:
	s_add_co_i32 s2, s14, 0
	s_lshr_b32 s2, s2, 2
	s_cmp_eq_u32 s2, 0
	s_cbranch_scc1 .Lrx_b5_s3_done
	s_mov_b32 s15, 0x198
	v_lshlrev_b32_e32 v2, 5, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v2, 0xc00, v2
	buffer_load_b64 v[12:13], v8, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v20, v4, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[14:15], v9, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v21, v5, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[16:17], v10, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v22, v6, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[18:19], v11, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v23, v7, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	global_load_b128 v[125:128], v2, s[28:29]
	global_load_b128 v[129:132], v2, s[28:29] offset:16
	global_load_b128 v[133:136], v2, s[30:31]
	global_load_b128 v[137:140], v2, s[30:31] offset:16
	global_load_b128 v[141:144], v2, s[32:33]
	global_load_b128 v[145:148], v2, s[32:33] offset:16
	global_load_b128 v[149:152], v2, s[34:35]
	global_load_b128 v[153:156], v2, s[34:35] offset:16
	global_load_b128 v[157:160], v2, s[36:37]
	global_load_b128 v[161:164], v2, s[36:37] offset:16
	.Lrx_b5_s3:
	s_cmp_gt_u32 s2, 1
	s_cselect_b32 s17, 0x220, 0
	s_cselect_b32 s18, 0x1000, 0
	s_add_co_i32 s16, s15, s17
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v3, s18, v2
	v_cmp_gt_u32_e32 vcc_lo, 16, v0
	s_wait_loadcnt 0x11
	v_cndmask_b32_e32 v12, v13, v12, vcc_lo
	s_wait_loadcnt 0x10
	v_and_b32_e32 v30, 0xf0f0f0f, v20
	v_lshrrev_b32_e32 v31, 4, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v31, 0xf0f0f0f, v31
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v24, v30
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v26, v30
	v_cvt_f32_ubyte2_e32 v28, v30
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v25, v31
	v_cvt_f32_ubyte1_e32 v27, v31
	v_cvt_f32_ubyte2_e32 v29, v31
	v_cvt_f32_ubyte3_e32 v30, v30
	v_cvt_f32_ubyte3_e32 v31, v31
	v_fma_mix_f32 v24, v12, v24, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v25, v12, v25, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v26, v12, v26, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v27, v12, v27, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v28, v12, v28, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v29, v12, v29, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v30, v12, v30, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v31, v12, v31, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xf
	v_cndmask_b32_e32 v14, v15, v14, vcc_lo
	s_wait_loadcnt 0xe
	v_and_b32_e32 v40, 0xf0f0f0f, v21
	v_lshrrev_b32_e32 v41, 4, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v41, 0xf0f0f0f, v41
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v34, v40
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v36, v40
	v_cvt_f32_ubyte2_e32 v38, v40
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v35, v41
	v_cvt_f32_ubyte1_e32 v37, v41
	v_cvt_f32_ubyte2_e32 v39, v41
	v_cvt_f32_ubyte3_e32 v40, v40
	v_cvt_f32_ubyte3_e32 v41, v41
	v_fma_mix_f32 v34, v14, v34, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v35, v14, v35, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v36, v14, v36, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v37, v14, v37, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v38, v14, v38, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v39, v14, v39, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v40, v14, v40, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v41, v14, v41, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xd
	v_cndmask_b32_e32 v16, v17, v16, vcc_lo
	s_wait_loadcnt 0xc
	v_and_b32_e32 v50, 0xf0f0f0f, v22
	v_lshrrev_b32_e32 v51, 4, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v51, 0xf0f0f0f, v51
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v44, v50
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v46, v50
	v_cvt_f32_ubyte2_e32 v48, v50
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v45, v51
	v_cvt_f32_ubyte1_e32 v47, v51
	v_cvt_f32_ubyte2_e32 v49, v51
	v_cvt_f32_ubyte3_e32 v50, v50
	v_cvt_f32_ubyte3_e32 v51, v51
	v_fma_mix_f32 v44, v16, v44, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v16, v45, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v16, v46, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v16, v47, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v48, v16, v48, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v49, v16, v49, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v16, v50, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v16, v51, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xb
	v_cndmask_b32_e32 v18, v19, v18, vcc_lo
	s_wait_loadcnt 0xa
	v_and_b32_e32 v60, 0xf0f0f0f, v23
	v_lshrrev_b32_e32 v61, 4, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v61, 0xf0f0f0f, v61
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v54, v60
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v56, v60
	v_cvt_f32_ubyte2_e32 v58, v60
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v55, v61
	v_cvt_f32_ubyte1_e32 v57, v61
	v_cvt_f32_ubyte2_e32 v59, v61
	v_cvt_f32_ubyte3_e32 v60, v60
	v_cvt_f32_ubyte3_e32 v61, v61
	v_fma_mix_f32 v54, v18, v54, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v18, v55, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v18, v56, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v18, v57, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v58, v18, v58, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v59, v18, v59, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v18, v60, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v18, v61, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	buffer_load_b64 v[12:13], v8, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v20, v4, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[14:15], v9, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v21, v5, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[16:17], v10, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v22, v6, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[18:19], v11, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v23, v7, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	s_wait_loadcnt 0x11
	v_dual_mul_f32 v32, v24, v125 :: v_dual_mul_f32 v33, v35, v126
	v_dual_mul_f32 v42, v44, v125 :: v_dual_mul_f32 v43, v55, v126
	s_wait_loadcnt 0xf
	v_dual_mul_f32 v52, v24, v133 :: v_dual_mul_f32 v53, v35, v134
	v_dual_mul_f32 v62, v44, v133 :: v_dual_mul_f32 v63, v55, v134
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v25, v126 :: v_dual_fmac_f32 v33, v34, v125
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v45, v126 :: v_dual_fmac_f32 v43, v54, v125
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v25, v134 :: v_dual_fmac_f32 v53, v34, v133
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v45, v134 :: v_dual_fmac_f32 v63, v54, v133
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v26, v127 :: v_dual_fmac_f32 v33, v36, v127
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v46, v127 :: v_dual_fmac_f32 v43, v56, v127
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v26, v135 :: v_dual_fmac_f32 v53, v36, v135
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v46, v135 :: v_dual_fmac_f32 v63, v56, v135
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v27, v128 :: v_dual_fmac_f32 v33, v37, v128
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v47, v128 :: v_dual_fmac_f32 v43, v57, v128
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v27, v136 :: v_dual_fmac_f32 v53, v37, v136
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v47, v136 :: v_dual_fmac_f32 v63, v57, v136
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v28, v129 :: v_dual_fmac_f32 v33, v38, v129
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v48, v129 :: v_dual_fmac_f32 v43, v58, v129
	s_wait_loadcnt 0xe
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v28, v137 :: v_dual_fmac_f32 v53, v38, v137
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v48, v137 :: v_dual_fmac_f32 v63, v58, v137
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v29, v130 :: v_dual_fmac_f32 v33, v39, v130
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v49, v130 :: v_dual_fmac_f32 v43, v59, v130
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v29, v138 :: v_dual_fmac_f32 v53, v39, v138
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v49, v138 :: v_dual_fmac_f32 v63, v59, v138
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v30, v131 :: v_dual_fmac_f32 v33, v40, v131
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v50, v131 :: v_dual_fmac_f32 v43, v60, v131
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v30, v139 :: v_dual_fmac_f32 v53, v40, v139
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v50, v139 :: v_dual_fmac_f32 v63, v60, v139
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v31, v132 :: v_dual_fmac_f32 v33, v41, v132
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v51, v132 :: v_dual_fmac_f32 v43, v61, v132
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v31, v140 :: v_dual_fmac_f32 v53, v41, v140
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v51, v140 :: v_dual_fmac_f32 v63, v61, v140
	global_load_b128 v[125:128], v3, s[28:29]
	global_load_b128 v[129:132], v3, s[28:29] offset:16
	global_load_b128 v[133:136], v3, s[30:31]
	global_load_b128 v[137:140], v3, s[30:31] offset:16
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v72, v32, v72 :: v_dual_add_f32 v73, v73, v33
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v74, v42, v74 :: v_dual_add_f32 v75, v75, v43
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v84, v52, v84 :: v_dual_add_f32 v85, v85, v53
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v86, v62, v86 :: v_dual_add_f32 v87, v87, v63
	s_wait_loadcnt 0x11
	v_dual_mul_f32 v32, v24, v141 :: v_dual_mul_f32 v33, v35, v142
	v_dual_mul_f32 v42, v44, v141 :: v_dual_mul_f32 v43, v55, v142
	s_wait_loadcnt 0xf
	v_dual_mul_f32 v52, v24, v149 :: v_dual_mul_f32 v53, v35, v150
	v_dual_mul_f32 v62, v44, v149 :: v_dual_mul_f32 v63, v55, v150
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v25, v142 :: v_dual_fmac_f32 v33, v34, v141
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v45, v142 :: v_dual_fmac_f32 v43, v54, v141
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v25, v150 :: v_dual_fmac_f32 v53, v34, v149
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v45, v150 :: v_dual_fmac_f32 v63, v54, v149
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v26, v143 :: v_dual_fmac_f32 v33, v36, v143
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v46, v143 :: v_dual_fmac_f32 v43, v56, v143
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v26, v151 :: v_dual_fmac_f32 v53, v36, v151
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v46, v151 :: v_dual_fmac_f32 v63, v56, v151
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v27, v144 :: v_dual_fmac_f32 v33, v37, v144
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v47, v144 :: v_dual_fmac_f32 v43, v57, v144
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v27, v152 :: v_dual_fmac_f32 v53, v37, v152
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v47, v152 :: v_dual_fmac_f32 v63, v57, v152
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v28, v145 :: v_dual_fmac_f32 v33, v38, v145
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v48, v145 :: v_dual_fmac_f32 v43, v58, v145
	s_wait_loadcnt 0xe
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v28, v153 :: v_dual_fmac_f32 v53, v38, v153
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v48, v153 :: v_dual_fmac_f32 v63, v58, v153
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v29, v146 :: v_dual_fmac_f32 v33, v39, v146
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v49, v146 :: v_dual_fmac_f32 v43, v59, v146
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v29, v154 :: v_dual_fmac_f32 v53, v39, v154
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v49, v154 :: v_dual_fmac_f32 v63, v59, v154
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v30, v147 :: v_dual_fmac_f32 v33, v40, v147
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v50, v147 :: v_dual_fmac_f32 v43, v60, v147
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v30, v155 :: v_dual_fmac_f32 v53, v40, v155
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v50, v155 :: v_dual_fmac_f32 v63, v60, v155
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v31, v148 :: v_dual_fmac_f32 v33, v41, v148
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v51, v148 :: v_dual_fmac_f32 v43, v61, v148
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v31, v156 :: v_dual_fmac_f32 v53, v41, v156
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v51, v156 :: v_dual_fmac_f32 v63, v61, v156
	global_load_b128 v[141:144], v3, s[32:33]
	global_load_b128 v[145:148], v3, s[32:33] offset:16
	global_load_b128 v[149:152], v3, s[34:35]
	global_load_b128 v[153:156], v3, s[34:35] offset:16
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v96, v32, v96 :: v_dual_add_f32 v97, v97, v33
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v98, v42, v98 :: v_dual_add_f32 v99, v99, v43
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v108, v52, v108 :: v_dual_add_f32 v109, v109, v53
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v110, v62, v110 :: v_dual_add_f32 v111, v111, v63
	s_wait_loadcnt 0x11
	v_dual_mul_f32 v32, v24, v157 :: v_dual_mul_f32 v33, v35, v158
	v_dual_mul_f32 v42, v44, v157 :: v_dual_mul_f32 v43, v55, v158
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v25, v158 :: v_dual_fmac_f32 v33, v34, v157
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v45, v158 :: v_dual_fmac_f32 v43, v54, v157
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v26, v159 :: v_dual_fmac_f32 v33, v36, v159
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v46, v159 :: v_dual_fmac_f32 v43, v56, v159
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v27, v160 :: v_dual_fmac_f32 v33, v37, v160
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v47, v160 :: v_dual_fmac_f32 v43, v57, v160
	s_wait_loadcnt 0x10
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v28, v161 :: v_dual_fmac_f32 v33, v38, v161
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v48, v161 :: v_dual_fmac_f32 v43, v58, v161
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v29, v162 :: v_dual_fmac_f32 v33, v39, v162
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v49, v162 :: v_dual_fmac_f32 v43, v59, v162
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v30, v163 :: v_dual_fmac_f32 v33, v40, v163
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v50, v163 :: v_dual_fmac_f32 v43, v60, v163
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v31, v164 :: v_dual_fmac_f32 v33, v41, v164
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v51, v164 :: v_dual_fmac_f32 v43, v61, v164
	global_load_b128 v[157:160], v3, s[36:37]
	global_load_b128 v[161:164], v3, s[36:37] offset:16
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_add_f32 v120, v32, v120 :: v_dual_add_f32 v121, v121, v33
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_add_f32 v122, v42, v122 :: v_dual_add_f32 v123, v123, v43
	s_mov_b32 s15, s16
	v_mov_b32_e32 v2, v3
	s_add_co_i32 s2, s2, -1
	s_cmp_lg_u32 s2, 0
	s_cbranch_scc1 .Lrx_b5_s3
	s_wait_loadcnt 0x0
	.Lrx_b5_s3_done:
	v_dual_add_f32 v68, v68, v72 :: v_dual_add_f32 v69, v69, v73
	v_dual_add_f32 v70, v70, v74 :: v_dual_add_f32 v71, v71, v75
	v_dual_add_f32 v80, v80, v84 :: v_dual_add_f32 v81, v81, v85
	v_dual_add_f32 v82, v82, v86 :: v_dual_add_f32 v83, v83, v87
	v_dual_add_f32 v92, v92, v96 :: v_dual_add_f32 v93, v93, v97
	v_dual_add_f32 v94, v94, v98 :: v_dual_add_f32 v95, v95, v99
	v_dual_add_f32 v104, v104, v108 :: v_dual_add_f32 v105, v105, v109
	v_dual_add_f32 v106, v106, v110 :: v_dual_add_f32 v107, v107, v111
	v_dual_add_f32 v116, v116, v120 :: v_dual_add_f32 v117, v117, v121
	v_dual_add_f32 v118, v118, v122 :: v_dual_add_f32 v119, v119, v123
	v_dual_add_f32 v64, v64, v68 :: v_dual_add_f32 v65, v65, v69
	v_dual_add_f32 v66, v66, v70 :: v_dual_add_f32 v67, v67, v71
	v_dual_add_f32 v76, v76, v80 :: v_dual_add_f32 v77, v77, v81
	v_dual_add_f32 v78, v78, v82 :: v_dual_add_f32 v79, v79, v83
	v_dual_add_f32 v88, v88, v92 :: v_dual_add_f32 v89, v89, v93
	v_dual_add_f32 v90, v90, v94 :: v_dual_add_f32 v91, v91, v95
	v_dual_add_f32 v100, v100, v104 :: v_dual_add_f32 v101, v101, v105
	v_dual_add_f32 v102, v102, v106 :: v_dual_add_f32 v103, v103, v107
	v_dual_add_f32 v112, v112, v116 :: v_dual_add_f32 v113, v113, v117
	v_dual_add_f32 v114, v114, v118 :: v_dual_add_f32 v115, v115, v119
	ds_swizzle_b32 v24, v64 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v25, v65 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v26, v66 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v27, v67 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v28, v76 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v29, v77 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v30, v78 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v31, v79 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v32, v88 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v33, v89 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v34, v90 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v35, v91 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v36, v100 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v37, v101 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v38, v102 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v39, v103 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v40, v112 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v41, v113 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v42, v114 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v43, v115 offset:swizzle(BITMASK_PERM,"1pppp")
	s_wait_dscnt 0x13
	v_add_f32_e32 v64, v64, v24
	s_wait_dscnt 0x12
	v_add_f32_e32 v65, v65, v25
	s_wait_dscnt 0x11
	v_add_f32_e32 v66, v66, v26
	s_wait_dscnt 0x10
	v_add_f32_e32 v67, v67, v27
	s_wait_dscnt 0xf
	v_add_f32_e32 v76, v76, v28
	s_wait_dscnt 0xe
	v_add_f32_e32 v77, v77, v29
	s_wait_dscnt 0xd
	v_add_f32_e32 v78, v78, v30
	s_wait_dscnt 0xc
	v_add_f32_e32 v79, v79, v31
	s_wait_dscnt 0xb
	v_add_f32_e32 v88, v88, v32
	s_wait_dscnt 0xa
	v_add_f32_e32 v89, v89, v33
	s_wait_dscnt 0x9
	v_add_f32_e32 v90, v90, v34
	s_wait_dscnt 0x8
	v_add_f32_e32 v91, v91, v35
	s_wait_dscnt 0x7
	v_add_f32_e32 v100, v100, v36
	s_wait_dscnt 0x6
	v_add_f32_e32 v101, v101, v37
	s_wait_dscnt 0x5
	v_add_f32_e32 v102, v102, v38
	s_wait_dscnt 0x4
	v_add_f32_e32 v103, v103, v39
	s_wait_dscnt 0x3
	v_add_f32_e32 v112, v112, v40
	s_wait_dscnt 0x2
	v_add_f32_e32 v113, v113, v41
	s_wait_dscnt 0x1
	v_add_f32_e32 v114, v114, v42
	s_wait_dscnt 0x0
	v_add_f32_e32 v115, v115, v43
	v_cmp_gt_u32_e32 vcc_lo, 24, v0
	v_cndmask_b32_e64 v3, 0, 8, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v3, v3, v0, 2
	ds_bpermute_b32 v24, v3, v64
	ds_bpermute_b32 v25, v3, v65
	ds_bpermute_b32 v26, v3, v66
	ds_bpermute_b32 v27, v3, v67
	ds_bpermute_b32 v28, v3, v76
	ds_bpermute_b32 v29, v3, v77
	ds_bpermute_b32 v30, v3, v78
	ds_bpermute_b32 v31, v3, v79
	ds_bpermute_b32 v32, v3, v88
	ds_bpermute_b32 v33, v3, v89
	ds_bpermute_b32 v34, v3, v90
	ds_bpermute_b32 v35, v3, v91
	ds_bpermute_b32 v36, v3, v100
	ds_bpermute_b32 v37, v3, v101
	ds_bpermute_b32 v38, v3, v102
	ds_bpermute_b32 v39, v3, v103
	ds_bpermute_b32 v40, v3, v112
	ds_bpermute_b32 v41, v3, v113
	ds_bpermute_b32 v42, v3, v114
	ds_bpermute_b32 v43, v3, v115
	s_wait_dscnt 0x13
	v_add_f32_e32 v64, v64, v24
	s_wait_dscnt 0x12
	v_add_f32_e32 v65, v65, v25
	s_wait_dscnt 0x11
	v_add_f32_e32 v66, v66, v26
	s_wait_dscnt 0x10
	v_add_f32_e32 v67, v67, v27
	s_wait_dscnt 0xf
	v_add_f32_e32 v76, v76, v28
	s_wait_dscnt 0xe
	v_add_f32_e32 v77, v77, v29
	s_wait_dscnt 0xd
	v_add_f32_e32 v78, v78, v30
	s_wait_dscnt 0xc
	v_add_f32_e32 v79, v79, v31
	s_wait_dscnt 0xb
	v_add_f32_e32 v88, v88, v32
	s_wait_dscnt 0xa
	v_add_f32_e32 v89, v89, v33
	s_wait_dscnt 0x9
	v_add_f32_e32 v90, v90, v34
	s_wait_dscnt 0x8
	v_add_f32_e32 v91, v91, v35
	s_wait_dscnt 0x7
	v_add_f32_e32 v100, v100, v36
	s_wait_dscnt 0x6
	v_add_f32_e32 v101, v101, v37
	s_wait_dscnt 0x5
	v_add_f32_e32 v102, v102, v38
	s_wait_dscnt 0x4
	v_add_f32_e32 v103, v103, v39
	s_wait_dscnt 0x3
	v_add_f32_e32 v112, v112, v40
	s_wait_dscnt 0x2
	v_add_f32_e32 v113, v113, v41
	s_wait_dscnt 0x1
	v_add_f32_e32 v114, v114, v42
	s_wait_dscnt 0x0
	v_add_f32_e32 v115, v115, v43
	v_cmp_gt_u32_e32 vcc_lo, 28, v0
	v_cndmask_b32_e64 v3, 0, 4, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v3, v3, v0, 2
	ds_bpermute_b32 v24, v3, v64
	ds_bpermute_b32 v25, v3, v65
	ds_bpermute_b32 v26, v3, v66
	ds_bpermute_b32 v27, v3, v67
	ds_bpermute_b32 v28, v3, v76
	ds_bpermute_b32 v29, v3, v77
	ds_bpermute_b32 v30, v3, v78
	ds_bpermute_b32 v31, v3, v79
	ds_bpermute_b32 v32, v3, v88
	ds_bpermute_b32 v33, v3, v89
	ds_bpermute_b32 v34, v3, v90
	ds_bpermute_b32 v35, v3, v91
	ds_bpermute_b32 v36, v3, v100
	ds_bpermute_b32 v37, v3, v101
	ds_bpermute_b32 v38, v3, v102
	ds_bpermute_b32 v39, v3, v103
	ds_bpermute_b32 v40, v3, v112
	ds_bpermute_b32 v41, v3, v113
	ds_bpermute_b32 v42, v3, v114
	ds_bpermute_b32 v43, v3, v115
	s_wait_dscnt 0x13
	v_add_f32_e32 v64, v64, v24
	s_wait_dscnt 0x12
	v_add_f32_e32 v65, v65, v25
	s_wait_dscnt 0x11
	v_add_f32_e32 v66, v66, v26
	s_wait_dscnt 0x10
	v_add_f32_e32 v67, v67, v27
	s_wait_dscnt 0xf
	v_add_f32_e32 v76, v76, v28
	s_wait_dscnt 0xe
	v_add_f32_e32 v77, v77, v29
	s_wait_dscnt 0xd
	v_add_f32_e32 v78, v78, v30
	s_wait_dscnt 0xc
	v_add_f32_e32 v79, v79, v31
	s_wait_dscnt 0xb
	v_add_f32_e32 v88, v88, v32
	s_wait_dscnt 0xa
	v_add_f32_e32 v89, v89, v33
	s_wait_dscnt 0x9
	v_add_f32_e32 v90, v90, v34
	s_wait_dscnt 0x8
	v_add_f32_e32 v91, v91, v35
	s_wait_dscnt 0x7
	v_add_f32_e32 v100, v100, v36
	s_wait_dscnt 0x6
	v_add_f32_e32 v101, v101, v37
	s_wait_dscnt 0x5
	v_add_f32_e32 v102, v102, v38
	s_wait_dscnt 0x4
	v_add_f32_e32 v103, v103, v39
	s_wait_dscnt 0x3
	v_add_f32_e32 v112, v112, v40
	s_wait_dscnt 0x2
	v_add_f32_e32 v113, v113, v41
	s_wait_dscnt 0x1
	v_add_f32_e32 v114, v114, v42
	s_wait_dscnt 0x0
	v_add_f32_e32 v115, v115, v43
	v_cmp_gt_u32_e32 vcc_lo, 30, v0
	v_cndmask_b32_e64 v3, 0, 2, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v3, v3, v0, 2
	ds_bpermute_b32 v24, v3, v64
	ds_bpermute_b32 v25, v3, v65
	ds_bpermute_b32 v26, v3, v66
	ds_bpermute_b32 v27, v3, v67
	ds_bpermute_b32 v28, v3, v76
	ds_bpermute_b32 v29, v3, v77
	ds_bpermute_b32 v30, v3, v78
	ds_bpermute_b32 v31, v3, v79
	ds_bpermute_b32 v32, v3, v88
	ds_bpermute_b32 v33, v3, v89
	ds_bpermute_b32 v34, v3, v90
	ds_bpermute_b32 v35, v3, v91
	ds_bpermute_b32 v36, v3, v100
	ds_bpermute_b32 v37, v3, v101
	ds_bpermute_b32 v38, v3, v102
	ds_bpermute_b32 v39, v3, v103
	ds_bpermute_b32 v40, v3, v112
	ds_bpermute_b32 v41, v3, v113
	ds_bpermute_b32 v42, v3, v114
	ds_bpermute_b32 v43, v3, v115
	s_wait_dscnt 0x13
	v_add_f32_e32 v64, v64, v24
	s_wait_dscnt 0x12
	v_add_f32_e32 v65, v65, v25
	s_wait_dscnt 0x11
	v_add_f32_e32 v66, v66, v26
	s_wait_dscnt 0x10
	v_add_f32_e32 v67, v67, v27
	s_wait_dscnt 0xf
	v_add_f32_e32 v76, v76, v28
	s_wait_dscnt 0xe
	v_add_f32_e32 v77, v77, v29
	s_wait_dscnt 0xd
	v_add_f32_e32 v78, v78, v30
	s_wait_dscnt 0xc
	v_add_f32_e32 v79, v79, v31
	s_wait_dscnt 0xb
	v_add_f32_e32 v88, v88, v32
	s_wait_dscnt 0xa
	v_add_f32_e32 v89, v89, v33
	s_wait_dscnt 0x9
	v_add_f32_e32 v90, v90, v34
	s_wait_dscnt 0x8
	v_add_f32_e32 v91, v91, v35
	s_wait_dscnt 0x7
	v_add_f32_e32 v100, v100, v36
	s_wait_dscnt 0x6
	v_add_f32_e32 v101, v101, v37
	s_wait_dscnt 0x5
	v_add_f32_e32 v102, v102, v38
	s_wait_dscnt 0x4
	v_add_f32_e32 v103, v103, v39
	s_wait_dscnt 0x3
	v_add_f32_e32 v112, v112, v40
	s_wait_dscnt 0x2
	v_add_f32_e32 v113, v113, v41
	s_wait_dscnt 0x1
	v_add_f32_e32 v114, v114, v42
	s_wait_dscnt 0x0
	v_add_f32_e32 v115, v115, v43
	v_cmp_gt_u32_e32 vcc_lo, 31, v0
	v_cndmask_b32_e64 v3, 0, 1, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v3, v3, v0, 2
	ds_bpermute_b32 v24, v3, v64
	ds_bpermute_b32 v25, v3, v65
	ds_bpermute_b32 v26, v3, v66
	ds_bpermute_b32 v27, v3, v67
	ds_bpermute_b32 v28, v3, v76
	ds_bpermute_b32 v29, v3, v77
	ds_bpermute_b32 v30, v3, v78
	ds_bpermute_b32 v31, v3, v79
	ds_bpermute_b32 v32, v3, v88
	ds_bpermute_b32 v33, v3, v89
	ds_bpermute_b32 v34, v3, v90
	ds_bpermute_b32 v35, v3, v91
	ds_bpermute_b32 v36, v3, v100
	ds_bpermute_b32 v37, v3, v101
	ds_bpermute_b32 v38, v3, v102
	ds_bpermute_b32 v39, v3, v103
	ds_bpermute_b32 v40, v3, v112
	ds_bpermute_b32 v41, v3, v113
	ds_bpermute_b32 v42, v3, v114
	ds_bpermute_b32 v43, v3, v115
	s_wait_dscnt 0x13
	v_add_f32_e32 v64, v64, v24
	s_wait_dscnt 0x12
	v_add_f32_e32 v65, v65, v25
	s_wait_dscnt 0x11
	v_add_f32_e32 v66, v66, v26
	s_wait_dscnt 0x10
	v_add_f32_e32 v67, v67, v27
	s_wait_dscnt 0xf
	v_add_f32_e32 v76, v76, v28
	s_wait_dscnt 0xe
	v_add_f32_e32 v77, v77, v29
	s_wait_dscnt 0xd
	v_add_f32_e32 v78, v78, v30
	s_wait_dscnt 0xc
	v_add_f32_e32 v79, v79, v31
	s_wait_dscnt 0xb
	v_add_f32_e32 v88, v88, v32
	s_wait_dscnt 0xa
	v_add_f32_e32 v89, v89, v33
	s_wait_dscnt 0x9
	v_add_f32_e32 v90, v90, v34
	s_wait_dscnt 0x8
	v_add_f32_e32 v91, v91, v35
	s_wait_dscnt 0x7
	v_add_f32_e32 v100, v100, v36
	s_wait_dscnt 0x6
	v_add_f32_e32 v101, v101, v37
	s_wait_dscnt 0x5
	v_add_f32_e32 v102, v102, v38
	s_wait_dscnt 0x4
	v_add_f32_e32 v103, v103, v39
	s_wait_dscnt 0x3
	v_add_f32_e32 v112, v112, v40
	s_wait_dscnt 0x2
	v_add_f32_e32 v113, v113, v41
	s_wait_dscnt 0x1
	v_add_f32_e32 v114, v114, v42
	s_wait_dscnt 0x0
	v_add_f32_e32 v115, v115, v43
	v_cmpx_eq_u32_e32 0, v0
	s_lshl_b32 s46, s12, 2
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v125, s46
	s_mul_i32 s47, s45, 1
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s47, s47, s46
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v126, s47
	s_mul_i32 s47, s45, 2
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s47, s47, s46
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v127, s47
	s_mul_i32 s47, s45, 3
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s47, s47, s46
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v128, s47
	s_mul_i32 s47, s45, 4
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s47, s47, s46
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v129, s47
	s_add_co_i32 s13, s12, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cbranch_scc0 .Lrx_b5_p0_single
	global_load_b32 v44, v125, s[8:9]
	global_load_b32 v45, v125, s[8:9] offset:4
	global_load_b32 v48, v126, s[8:9]
	global_load_b32 v49, v126, s[8:9] offset:4
	global_load_b32 v52, v127, s[8:9]
	global_load_b32 v53, v127, s[8:9] offset:4
	global_load_b32 v56, v128, s[8:9]
	global_load_b32 v57, v128, s[8:9] offset:4
	global_load_b32 v60, v129, s[8:9]
	global_load_b32 v61, v129, s[8:9] offset:4
	s_wait_loadcnt 0x9
	v_add_f32_e32 v44, v64, v44
	global_store_b32 v125, v44, s[8:9]
	s_wait_loadcnt 0x8
	v_add_f32_e32 v45, v65, v45
	global_store_b32 v125, v45, s[8:9] offset:4
	s_wait_loadcnt 0x7
	v_add_f32_e32 v48, v76, v48
	global_store_b32 v126, v48, s[8:9]
	s_wait_loadcnt 0x6
	v_add_f32_e32 v49, v77, v49
	global_store_b32 v126, v49, s[8:9] offset:4
	s_wait_loadcnt 0x5
	v_add_f32_e32 v52, v88, v52
	global_store_b32 v127, v52, s[8:9]
	s_wait_loadcnt 0x4
	v_add_f32_e32 v53, v89, v53
	global_store_b32 v127, v53, s[8:9] offset:4
	s_wait_loadcnt 0x3
	v_add_f32_e32 v56, v100, v56
	global_store_b32 v128, v56, s[8:9]
	s_wait_loadcnt 0x2
	v_add_f32_e32 v57, v101, v57
	global_store_b32 v128, v57, s[8:9] offset:4
	s_wait_loadcnt 0x1
	v_add_f32_e32 v60, v112, v60
	global_store_b32 v129, v60, s[8:9]
	s_wait_loadcnt 0x0
	v_add_f32_e32 v61, v113, v61
	global_store_b32 v129, v61, s[8:9] offset:4
	s_wait_storecnt 0x0
	s_branch .Lrx_b5_p0_next
	.Lrx_b5_p0_single:
	global_load_b32 v44, v125, s[8:9]
	global_load_b32 v48, v126, s[8:9]
	global_load_b32 v52, v127, s[8:9]
	global_load_b32 v56, v128, s[8:9]
	global_load_b32 v60, v129, s[8:9]
	s_wait_loadcnt 0x4
	v_add_f32_e32 v44, v44, v64
	global_store_b32 v125, v44, s[8:9]
	s_wait_loadcnt 0x3
	v_add_f32_e32 v48, v48, v76
	global_store_b32 v126, v48, s[8:9]
	s_wait_loadcnt 0x2
	v_add_f32_e32 v52, v52, v88
	global_store_b32 v127, v52, s[8:9]
	s_wait_loadcnt 0x1
	v_add_f32_e32 v56, v56, v100
	global_store_b32 v128, v56, s[8:9]
	s_wait_loadcnt 0x0
	v_add_f32_e32 v60, v60, v112
	global_store_b32 v129, v60, s[8:9]
	s_wait_storecnt 0x0
	s_branch .Lrx_b5_stored
	.Lrx_b5_p0_next:
	s_add_co_i32 s13, s12, 2
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cbranch_scc0 .Lrx_b5_stored
	s_add_co_i32 s13, s12, 3
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cbranch_scc0 .Lrx_b5_p1_single
	global_load_b32 v46, v125, s[8:9] offset:8
	global_load_b32 v47, v125, s[8:9] offset:12
	global_load_b32 v50, v126, s[8:9] offset:8
	global_load_b32 v51, v126, s[8:9] offset:12
	global_load_b32 v54, v127, s[8:9] offset:8
	global_load_b32 v55, v127, s[8:9] offset:12
	global_load_b32 v58, v128, s[8:9] offset:8
	global_load_b32 v59, v128, s[8:9] offset:12
	global_load_b32 v62, v129, s[8:9] offset:8
	global_load_b32 v63, v129, s[8:9] offset:12
	s_wait_loadcnt 0x9
	v_add_f32_e32 v46, v66, v46
	global_store_b32 v125, v46, s[8:9] offset:8
	s_wait_loadcnt 0x8
	v_add_f32_e32 v47, v67, v47
	global_store_b32 v125, v47, s[8:9] offset:12
	s_wait_loadcnt 0x7
	v_add_f32_e32 v50, v78, v50
	global_store_b32 v126, v50, s[8:9] offset:8
	s_wait_loadcnt 0x6
	v_add_f32_e32 v51, v79, v51
	global_store_b32 v126, v51, s[8:9] offset:12
	s_wait_loadcnt 0x5
	v_add_f32_e32 v54, v90, v54
	global_store_b32 v127, v54, s[8:9] offset:8
	s_wait_loadcnt 0x4
	v_add_f32_e32 v55, v91, v55
	global_store_b32 v127, v55, s[8:9] offset:12
	s_wait_loadcnt 0x3
	v_add_f32_e32 v58, v102, v58
	global_store_b32 v128, v58, s[8:9] offset:8
	s_wait_loadcnt 0x2
	v_add_f32_e32 v59, v103, v59
	global_store_b32 v128, v59, s[8:9] offset:12
	s_wait_loadcnt 0x1
	v_add_f32_e32 v62, v114, v62
	global_store_b32 v129, v62, s[8:9] offset:8
	s_wait_loadcnt 0x0
	v_add_f32_e32 v63, v115, v63
	global_store_b32 v129, v63, s[8:9] offset:12
	s_wait_storecnt 0x0
	s_branch .Lrx_b5_stored
	.Lrx_b5_p1_single:
	global_load_b32 v46, v125, s[8:9] offset:8
	global_load_b32 v50, v126, s[8:9] offset:8
	global_load_b32 v54, v127, s[8:9] offset:8
	global_load_b32 v58, v128, s[8:9] offset:8
	global_load_b32 v62, v129, s[8:9] offset:8
	s_wait_loadcnt 0x4
	v_add_f32_e32 v46, v46, v66
	global_store_b32 v125, v46, s[8:9] offset:8
	s_wait_loadcnt 0x3
	v_add_f32_e32 v50, v50, v78
	global_store_b32 v126, v50, s[8:9] offset:8
	s_wait_loadcnt 0x2
	v_add_f32_e32 v54, v54, v90
	global_store_b32 v127, v54, s[8:9] offset:8
	s_wait_loadcnt 0x1
	v_add_f32_e32 v58, v58, v102
	global_store_b32 v128, v58, s[8:9] offset:8
	s_wait_loadcnt 0x0
	v_add_f32_e32 v62, v62, v114
	global_store_b32 v129, v62, s[8:9] offset:8
	s_wait_storecnt 0x0
	s_branch .Lrx_b5_stored
	.Lrx_b5_stored:
	s_branch .Lrx_end
	.Lrx_b6:
	s_mul_i32 s12, ttmp9, 4
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_i32 s12, s10
	s_cbranch_scc1 .Lrx_end
	s_mov_b32 s13, s12
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s13, s13, s19
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v8, s13
	v_add_nc_u32_e32 v4, s13, v1
	s_add_co_i32 s13, s12, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cselect_b32 s13, s13, s12
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s13, s13, s19
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v9, s13
	v_add_nc_u32_e32 v5, s13, v1
	s_add_co_i32 s13, s12, 2
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cselect_b32 s13, s13, s12
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s13, s13, s19
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v10, s13
	v_add_nc_u32_e32 v6, s13, v1
	s_add_co_i32 s13, s12, 3
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cselect_b32 s13, s13, s12
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s13, s13, s19
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v11, s13
	v_add_nc_u32_e32 v7, s13, v1
	v_mov_b32_e32 v64, 0
	v_mov_b32_e32 v65, 0
	v_mov_b32_e32 v66, 0
	v_mov_b32_e32 v67, 0
	v_mov_b32_e32 v68, 0
	v_mov_b32_e32 v69, 0
	v_mov_b32_e32 v70, 0
	v_mov_b32_e32 v71, 0
	v_mov_b32_e32 v72, 0
	v_mov_b32_e32 v73, 0
	v_mov_b32_e32 v74, 0
	v_mov_b32_e32 v75, 0
	v_mov_b32_e32 v76, 0
	v_mov_b32_e32 v77, 0
	v_mov_b32_e32 v78, 0
	v_mov_b32_e32 v79, 0
	v_mov_b32_e32 v80, 0
	v_mov_b32_e32 v81, 0
	v_mov_b32_e32 v82, 0
	v_mov_b32_e32 v83, 0
	v_mov_b32_e32 v84, 0
	v_mov_b32_e32 v85, 0
	v_mov_b32_e32 v86, 0
	v_mov_b32_e32 v87, 0
	v_mov_b32_e32 v88, 0
	v_mov_b32_e32 v89, 0
	v_mov_b32_e32 v90, 0
	v_mov_b32_e32 v91, 0
	v_mov_b32_e32 v92, 0
	v_mov_b32_e32 v93, 0
	v_mov_b32_e32 v94, 0
	v_mov_b32_e32 v95, 0
	v_mov_b32_e32 v96, 0
	v_mov_b32_e32 v97, 0
	v_mov_b32_e32 v98, 0
	v_mov_b32_e32 v99, 0
	v_mov_b32_e32 v100, 0
	v_mov_b32_e32 v101, 0
	v_mov_b32_e32 v102, 0
	v_mov_b32_e32 v103, 0
	v_mov_b32_e32 v104, 0
	v_mov_b32_e32 v105, 0
	v_mov_b32_e32 v106, 0
	v_mov_b32_e32 v107, 0
	v_mov_b32_e32 v108, 0
	v_mov_b32_e32 v109, 0
	v_mov_b32_e32 v110, 0
	v_mov_b32_e32 v111, 0
	v_mov_b32_e32 v112, 0
	v_mov_b32_e32 v113, 0
	v_mov_b32_e32 v114, 0
	v_mov_b32_e32 v115, 0
	v_mov_b32_e32 v116, 0
	v_mov_b32_e32 v117, 0
	v_mov_b32_e32 v118, 0
	v_mov_b32_e32 v119, 0
	v_mov_b32_e32 v120, 0
	v_mov_b32_e32 v121, 0
	v_mov_b32_e32 v122, 0
	v_mov_b32_e32 v123, 0
	v_mov_b32_e32 v124, 0
	v_mov_b32_e32 v125, 0
	v_mov_b32_e32 v126, 0
	v_mov_b32_e32 v127, 0
	v_mov_b32_e32 v128, 0
	v_mov_b32_e32 v129, 0
	v_mov_b32_e32 v130, 0
	v_mov_b32_e32 v131, 0
	v_mov_b32_e32 v132, 0
	v_mov_b32_e32 v133, 0
	v_mov_b32_e32 v134, 0
	v_mov_b32_e32 v135, 0
	s_add_co_i32 s2, s14, 3
	s_lshr_b32 s2, s2, 2
	s_cmp_eq_u32 s2, 0
	s_cbranch_scc1 .Lrx_b6_s0_done
	s_mov_b32 s15, 0
	v_lshlrev_b32_e32 v2, 5, v0
	buffer_load_b64 v[12:13], v8, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v20, v4, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[14:15], v9, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v21, v5, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[16:17], v10, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v22, v6, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[18:19], v11, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v23, v7, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	global_load_b128 v[137:140], v2, s[28:29]
	global_load_b128 v[141:144], v2, s[28:29] offset:16
	global_load_b128 v[145:148], v2, s[30:31]
	global_load_b128 v[149:152], v2, s[30:31] offset:16
	global_load_b128 v[153:156], v2, s[32:33]
	global_load_b128 v[157:160], v2, s[32:33] offset:16
	global_load_b128 v[161:164], v2, s[34:35]
	global_load_b128 v[165:168], v2, s[34:35] offset:16
	global_load_b128 v[169:172], v2, s[36:37]
	global_load_b128 v[173:176], v2, s[36:37] offset:16
	global_load_b128 v[177:180], v2, s[38:39]
	global_load_b128 v[181:184], v2, s[38:39] offset:16
	.Lrx_b6_s0:
	s_cmp_gt_u32 s2, 1
	s_cselect_b32 s17, 0x220, 0
	s_cselect_b32 s18, 0x1000, 0
	s_add_co_i32 s16, s15, s17
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v3, s18, v2
	v_cmp_gt_u32_e32 vcc_lo, 16, v0
	s_wait_loadcnt 0x13
	v_cndmask_b32_e32 v12, v13, v12, vcc_lo
	s_wait_loadcnt 0x12
	v_and_b32_e32 v30, 0xf0f0f0f, v20
	v_lshrrev_b32_e32 v31, 4, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v31, 0xf0f0f0f, v31
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v24, v30
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v26, v30
	v_cvt_f32_ubyte2_e32 v28, v30
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v25, v31
	v_cvt_f32_ubyte1_e32 v27, v31
	v_cvt_f32_ubyte2_e32 v29, v31
	v_cvt_f32_ubyte3_e32 v30, v30
	v_cvt_f32_ubyte3_e32 v31, v31
	v_fma_mix_f32 v24, v12, v24, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v25, v12, v25, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v26, v12, v26, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v27, v12, v27, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v28, v12, v28, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v29, v12, v29, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v30, v12, v30, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v31, v12, v31, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x11
	v_cndmask_b32_e32 v14, v15, v14, vcc_lo
	s_wait_loadcnt 0x10
	v_and_b32_e32 v40, 0xf0f0f0f, v21
	v_lshrrev_b32_e32 v41, 4, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v41, 0xf0f0f0f, v41
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v34, v40
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v36, v40
	v_cvt_f32_ubyte2_e32 v38, v40
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v35, v41
	v_cvt_f32_ubyte1_e32 v37, v41
	v_cvt_f32_ubyte2_e32 v39, v41
	v_cvt_f32_ubyte3_e32 v40, v40
	v_cvt_f32_ubyte3_e32 v41, v41
	v_fma_mix_f32 v34, v14, v34, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v35, v14, v35, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v36, v14, v36, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v37, v14, v37, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v38, v14, v38, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v39, v14, v39, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v40, v14, v40, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v41, v14, v41, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xf
	v_cndmask_b32_e32 v16, v17, v16, vcc_lo
	s_wait_loadcnt 0xe
	v_and_b32_e32 v50, 0xf0f0f0f, v22
	v_lshrrev_b32_e32 v51, 4, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v51, 0xf0f0f0f, v51
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v44, v50
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v46, v50
	v_cvt_f32_ubyte2_e32 v48, v50
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v45, v51
	v_cvt_f32_ubyte1_e32 v47, v51
	v_cvt_f32_ubyte2_e32 v49, v51
	v_cvt_f32_ubyte3_e32 v50, v50
	v_cvt_f32_ubyte3_e32 v51, v51
	v_fma_mix_f32 v44, v16, v44, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v16, v45, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v16, v46, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v16, v47, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v48, v16, v48, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v49, v16, v49, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v16, v50, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v16, v51, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xd
	v_cndmask_b32_e32 v18, v19, v18, vcc_lo
	s_wait_loadcnt 0xc
	v_and_b32_e32 v60, 0xf0f0f0f, v23
	v_lshrrev_b32_e32 v61, 4, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v61, 0xf0f0f0f, v61
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v54, v60
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v56, v60
	v_cvt_f32_ubyte2_e32 v58, v60
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v55, v61
	v_cvt_f32_ubyte1_e32 v57, v61
	v_cvt_f32_ubyte2_e32 v59, v61
	v_cvt_f32_ubyte3_e32 v60, v60
	v_cvt_f32_ubyte3_e32 v61, v61
	v_fma_mix_f32 v54, v18, v54, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v18, v55, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v18, v56, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v18, v57, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v58, v18, v58, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v59, v18, v59, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v18, v60, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v18, v61, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	buffer_load_b64 v[12:13], v8, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v20, v4, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[14:15], v9, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v21, v5, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[16:17], v10, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v22, v6, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[18:19], v11, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v23, v7, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	s_wait_loadcnt 0x13
	v_dual_mul_f32 v32, v24, v137 :: v_dual_mul_f32 v33, v35, v138
	v_dual_mul_f32 v42, v44, v137 :: v_dual_mul_f32 v43, v55, v138
	s_wait_loadcnt 0x11
	v_dual_mul_f32 v52, v24, v145 :: v_dual_mul_f32 v53, v35, v146
	v_dual_mul_f32 v62, v44, v145 :: v_dual_mul_f32 v63, v55, v146
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v25, v138 :: v_dual_fmac_f32 v33, v34, v137
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v45, v138 :: v_dual_fmac_f32 v43, v54, v137
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v25, v146 :: v_dual_fmac_f32 v53, v34, v145
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v45, v146 :: v_dual_fmac_f32 v63, v54, v145
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v26, v139 :: v_dual_fmac_f32 v33, v36, v139
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v46, v139 :: v_dual_fmac_f32 v43, v56, v139
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v26, v147 :: v_dual_fmac_f32 v53, v36, v147
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v46, v147 :: v_dual_fmac_f32 v63, v56, v147
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v27, v140 :: v_dual_fmac_f32 v33, v37, v140
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v47, v140 :: v_dual_fmac_f32 v43, v57, v140
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v27, v148 :: v_dual_fmac_f32 v53, v37, v148
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v47, v148 :: v_dual_fmac_f32 v63, v57, v148
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v28, v141 :: v_dual_fmac_f32 v33, v38, v141
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v48, v141 :: v_dual_fmac_f32 v43, v58, v141
	s_wait_loadcnt 0x10
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v28, v149 :: v_dual_fmac_f32 v53, v38, v149
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v48, v149 :: v_dual_fmac_f32 v63, v58, v149
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v29, v142 :: v_dual_fmac_f32 v33, v39, v142
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v49, v142 :: v_dual_fmac_f32 v43, v59, v142
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v29, v150 :: v_dual_fmac_f32 v53, v39, v150
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v49, v150 :: v_dual_fmac_f32 v63, v59, v150
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v30, v143 :: v_dual_fmac_f32 v33, v40, v143
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v50, v143 :: v_dual_fmac_f32 v43, v60, v143
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v30, v151 :: v_dual_fmac_f32 v53, v40, v151
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v50, v151 :: v_dual_fmac_f32 v63, v60, v151
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v31, v144 :: v_dual_fmac_f32 v33, v41, v144
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v51, v144 :: v_dual_fmac_f32 v43, v61, v144
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v31, v152 :: v_dual_fmac_f32 v53, v41, v152
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v51, v152 :: v_dual_fmac_f32 v63, v61, v152
	global_load_b128 v[137:140], v3, s[28:29]
	global_load_b128 v[141:144], v3, s[28:29] offset:16
	global_load_b128 v[145:148], v3, s[30:31]
	global_load_b128 v[149:152], v3, s[30:31] offset:16
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v64, v32, v64 :: v_dual_add_f32 v65, v65, v33
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v66, v42, v66 :: v_dual_add_f32 v67, v67, v43
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v76, v52, v76 :: v_dual_add_f32 v77, v77, v53
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v78, v62, v78 :: v_dual_add_f32 v79, v79, v63
	s_wait_loadcnt 0x13
	v_dual_mul_f32 v32, v24, v153 :: v_dual_mul_f32 v33, v35, v154
	v_dual_mul_f32 v42, v44, v153 :: v_dual_mul_f32 v43, v55, v154
	s_wait_loadcnt 0x11
	v_dual_mul_f32 v52, v24, v161 :: v_dual_mul_f32 v53, v35, v162
	v_dual_mul_f32 v62, v44, v161 :: v_dual_mul_f32 v63, v55, v162
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v25, v154 :: v_dual_fmac_f32 v33, v34, v153
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v45, v154 :: v_dual_fmac_f32 v43, v54, v153
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v25, v162 :: v_dual_fmac_f32 v53, v34, v161
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v45, v162 :: v_dual_fmac_f32 v63, v54, v161
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v26, v155 :: v_dual_fmac_f32 v33, v36, v155
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v46, v155 :: v_dual_fmac_f32 v43, v56, v155
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v26, v163 :: v_dual_fmac_f32 v53, v36, v163
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v46, v163 :: v_dual_fmac_f32 v63, v56, v163
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v27, v156 :: v_dual_fmac_f32 v33, v37, v156
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v47, v156 :: v_dual_fmac_f32 v43, v57, v156
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v27, v164 :: v_dual_fmac_f32 v53, v37, v164
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v47, v164 :: v_dual_fmac_f32 v63, v57, v164
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v28, v157 :: v_dual_fmac_f32 v33, v38, v157
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v48, v157 :: v_dual_fmac_f32 v43, v58, v157
	s_wait_loadcnt 0x10
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v28, v165 :: v_dual_fmac_f32 v53, v38, v165
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v48, v165 :: v_dual_fmac_f32 v63, v58, v165
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v29, v158 :: v_dual_fmac_f32 v33, v39, v158
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v49, v158 :: v_dual_fmac_f32 v43, v59, v158
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v29, v166 :: v_dual_fmac_f32 v53, v39, v166
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v49, v166 :: v_dual_fmac_f32 v63, v59, v166
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v30, v159 :: v_dual_fmac_f32 v33, v40, v159
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v50, v159 :: v_dual_fmac_f32 v43, v60, v159
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v30, v167 :: v_dual_fmac_f32 v53, v40, v167
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v50, v167 :: v_dual_fmac_f32 v63, v60, v167
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v31, v160 :: v_dual_fmac_f32 v33, v41, v160
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v51, v160 :: v_dual_fmac_f32 v43, v61, v160
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v31, v168 :: v_dual_fmac_f32 v53, v41, v168
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v51, v168 :: v_dual_fmac_f32 v63, v61, v168
	global_load_b128 v[153:156], v3, s[32:33]
	global_load_b128 v[157:160], v3, s[32:33] offset:16
	global_load_b128 v[161:164], v3, s[34:35]
	global_load_b128 v[165:168], v3, s[34:35] offset:16
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v88, v32, v88 :: v_dual_add_f32 v89, v89, v33
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v90, v42, v90 :: v_dual_add_f32 v91, v91, v43
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v100, v52, v100 :: v_dual_add_f32 v101, v101, v53
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v102, v62, v102 :: v_dual_add_f32 v103, v103, v63
	s_wait_loadcnt 0x13
	v_dual_mul_f32 v32, v24, v169 :: v_dual_mul_f32 v33, v35, v170
	v_dual_mul_f32 v42, v44, v169 :: v_dual_mul_f32 v43, v55, v170
	s_wait_loadcnt 0x11
	v_dual_mul_f32 v52, v24, v177 :: v_dual_mul_f32 v53, v35, v178
	v_dual_mul_f32 v62, v44, v177 :: v_dual_mul_f32 v63, v55, v178
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v25, v170 :: v_dual_fmac_f32 v33, v34, v169
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v45, v170 :: v_dual_fmac_f32 v43, v54, v169
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v25, v178 :: v_dual_fmac_f32 v53, v34, v177
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v45, v178 :: v_dual_fmac_f32 v63, v54, v177
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v26, v171 :: v_dual_fmac_f32 v33, v36, v171
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v46, v171 :: v_dual_fmac_f32 v43, v56, v171
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v26, v179 :: v_dual_fmac_f32 v53, v36, v179
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v46, v179 :: v_dual_fmac_f32 v63, v56, v179
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v27, v172 :: v_dual_fmac_f32 v33, v37, v172
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v47, v172 :: v_dual_fmac_f32 v43, v57, v172
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v27, v180 :: v_dual_fmac_f32 v53, v37, v180
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v47, v180 :: v_dual_fmac_f32 v63, v57, v180
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v28, v173 :: v_dual_fmac_f32 v33, v38, v173
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v48, v173 :: v_dual_fmac_f32 v43, v58, v173
	s_wait_loadcnt 0x10
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v28, v181 :: v_dual_fmac_f32 v53, v38, v181
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v48, v181 :: v_dual_fmac_f32 v63, v58, v181
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v29, v174 :: v_dual_fmac_f32 v33, v39, v174
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v49, v174 :: v_dual_fmac_f32 v43, v59, v174
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v29, v182 :: v_dual_fmac_f32 v53, v39, v182
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v49, v182 :: v_dual_fmac_f32 v63, v59, v182
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v30, v175 :: v_dual_fmac_f32 v33, v40, v175
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v50, v175 :: v_dual_fmac_f32 v43, v60, v175
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v30, v183 :: v_dual_fmac_f32 v53, v40, v183
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v50, v183 :: v_dual_fmac_f32 v63, v60, v183
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v31, v176 :: v_dual_fmac_f32 v33, v41, v176
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v51, v176 :: v_dual_fmac_f32 v43, v61, v176
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v31, v184 :: v_dual_fmac_f32 v53, v41, v184
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v51, v184 :: v_dual_fmac_f32 v63, v61, v184
	global_load_b128 v[169:172], v3, s[36:37]
	global_load_b128 v[173:176], v3, s[36:37] offset:16
	global_load_b128 v[177:180], v3, s[38:39]
	global_load_b128 v[181:184], v3, s[38:39] offset:16
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v112, v32, v112 :: v_dual_add_f32 v113, v113, v33
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v114, v42, v114 :: v_dual_add_f32 v115, v115, v43
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v124, v52, v124 :: v_dual_add_f32 v125, v125, v53
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v126, v62, v126 :: v_dual_add_f32 v127, v127, v63
	s_mov_b32 s15, s16
	v_mov_b32_e32 v2, v3
	s_add_co_i32 s2, s2, -1
	s_cmp_lg_u32 s2, 0
	s_cbranch_scc1 .Lrx_b6_s0
	s_wait_loadcnt 0x0
	.Lrx_b6_s0_done:
	s_add_co_i32 s2, s14, 2
	s_lshr_b32 s2, s2, 2
	s_cmp_eq_u32 s2, 0
	s_cbranch_scc1 .Lrx_b6_s1_done
	s_mov_b32 s15, 0x88
	v_lshlrev_b32_e32 v2, 5, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v2, 0x400, v2
	buffer_load_b64 v[12:13], v8, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v20, v4, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[14:15], v9, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v21, v5, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[16:17], v10, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v22, v6, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[18:19], v11, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v23, v7, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	global_load_b128 v[137:140], v2, s[28:29]
	global_load_b128 v[141:144], v2, s[28:29] offset:16
	global_load_b128 v[145:148], v2, s[30:31]
	global_load_b128 v[149:152], v2, s[30:31] offset:16
	global_load_b128 v[153:156], v2, s[32:33]
	global_load_b128 v[157:160], v2, s[32:33] offset:16
	global_load_b128 v[161:164], v2, s[34:35]
	global_load_b128 v[165:168], v2, s[34:35] offset:16
	global_load_b128 v[169:172], v2, s[36:37]
	global_load_b128 v[173:176], v2, s[36:37] offset:16
	global_load_b128 v[177:180], v2, s[38:39]
	global_load_b128 v[181:184], v2, s[38:39] offset:16
	.Lrx_b6_s1:
	s_cmp_gt_u32 s2, 1
	s_cselect_b32 s17, 0x220, 0
	s_cselect_b32 s18, 0x1000, 0
	s_add_co_i32 s16, s15, s17
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v3, s18, v2
	v_cmp_gt_u32_e32 vcc_lo, 16, v0
	s_wait_loadcnt 0x13
	v_cndmask_b32_e32 v12, v13, v12, vcc_lo
	s_wait_loadcnt 0x12
	v_and_b32_e32 v30, 0xf0f0f0f, v20
	v_lshrrev_b32_e32 v31, 4, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v31, 0xf0f0f0f, v31
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v24, v30
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v26, v30
	v_cvt_f32_ubyte2_e32 v28, v30
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v25, v31
	v_cvt_f32_ubyte1_e32 v27, v31
	v_cvt_f32_ubyte2_e32 v29, v31
	v_cvt_f32_ubyte3_e32 v30, v30
	v_cvt_f32_ubyte3_e32 v31, v31
	v_fma_mix_f32 v24, v12, v24, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v25, v12, v25, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v26, v12, v26, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v27, v12, v27, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v28, v12, v28, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v29, v12, v29, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v30, v12, v30, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v31, v12, v31, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x11
	v_cndmask_b32_e32 v14, v15, v14, vcc_lo
	s_wait_loadcnt 0x10
	v_and_b32_e32 v40, 0xf0f0f0f, v21
	v_lshrrev_b32_e32 v41, 4, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v41, 0xf0f0f0f, v41
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v34, v40
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v36, v40
	v_cvt_f32_ubyte2_e32 v38, v40
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v35, v41
	v_cvt_f32_ubyte1_e32 v37, v41
	v_cvt_f32_ubyte2_e32 v39, v41
	v_cvt_f32_ubyte3_e32 v40, v40
	v_cvt_f32_ubyte3_e32 v41, v41
	v_fma_mix_f32 v34, v14, v34, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v35, v14, v35, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v36, v14, v36, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v37, v14, v37, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v38, v14, v38, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v39, v14, v39, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v40, v14, v40, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v41, v14, v41, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xf
	v_cndmask_b32_e32 v16, v17, v16, vcc_lo
	s_wait_loadcnt 0xe
	v_and_b32_e32 v50, 0xf0f0f0f, v22
	v_lshrrev_b32_e32 v51, 4, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v51, 0xf0f0f0f, v51
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v44, v50
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v46, v50
	v_cvt_f32_ubyte2_e32 v48, v50
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v45, v51
	v_cvt_f32_ubyte1_e32 v47, v51
	v_cvt_f32_ubyte2_e32 v49, v51
	v_cvt_f32_ubyte3_e32 v50, v50
	v_cvt_f32_ubyte3_e32 v51, v51
	v_fma_mix_f32 v44, v16, v44, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v16, v45, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v16, v46, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v16, v47, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v48, v16, v48, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v49, v16, v49, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v16, v50, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v16, v51, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xd
	v_cndmask_b32_e32 v18, v19, v18, vcc_lo
	s_wait_loadcnt 0xc
	v_and_b32_e32 v60, 0xf0f0f0f, v23
	v_lshrrev_b32_e32 v61, 4, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v61, 0xf0f0f0f, v61
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v54, v60
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v56, v60
	v_cvt_f32_ubyte2_e32 v58, v60
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v55, v61
	v_cvt_f32_ubyte1_e32 v57, v61
	v_cvt_f32_ubyte2_e32 v59, v61
	v_cvt_f32_ubyte3_e32 v60, v60
	v_cvt_f32_ubyte3_e32 v61, v61
	v_fma_mix_f32 v54, v18, v54, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v18, v55, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v18, v56, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v18, v57, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v58, v18, v58, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v59, v18, v59, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v18, v60, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v18, v61, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	buffer_load_b64 v[12:13], v8, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v20, v4, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[14:15], v9, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v21, v5, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[16:17], v10, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v22, v6, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[18:19], v11, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v23, v7, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	s_wait_loadcnt 0x13
	v_dual_mul_f32 v32, v24, v137 :: v_dual_mul_f32 v33, v35, v138
	v_dual_mul_f32 v42, v44, v137 :: v_dual_mul_f32 v43, v55, v138
	s_wait_loadcnt 0x11
	v_dual_mul_f32 v52, v24, v145 :: v_dual_mul_f32 v53, v35, v146
	v_dual_mul_f32 v62, v44, v145 :: v_dual_mul_f32 v63, v55, v146
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v25, v138 :: v_dual_fmac_f32 v33, v34, v137
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v45, v138 :: v_dual_fmac_f32 v43, v54, v137
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v25, v146 :: v_dual_fmac_f32 v53, v34, v145
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v45, v146 :: v_dual_fmac_f32 v63, v54, v145
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v26, v139 :: v_dual_fmac_f32 v33, v36, v139
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v46, v139 :: v_dual_fmac_f32 v43, v56, v139
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v26, v147 :: v_dual_fmac_f32 v53, v36, v147
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v46, v147 :: v_dual_fmac_f32 v63, v56, v147
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v27, v140 :: v_dual_fmac_f32 v33, v37, v140
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v47, v140 :: v_dual_fmac_f32 v43, v57, v140
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v27, v148 :: v_dual_fmac_f32 v53, v37, v148
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v47, v148 :: v_dual_fmac_f32 v63, v57, v148
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v28, v141 :: v_dual_fmac_f32 v33, v38, v141
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v48, v141 :: v_dual_fmac_f32 v43, v58, v141
	s_wait_loadcnt 0x10
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v28, v149 :: v_dual_fmac_f32 v53, v38, v149
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v48, v149 :: v_dual_fmac_f32 v63, v58, v149
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v29, v142 :: v_dual_fmac_f32 v33, v39, v142
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v49, v142 :: v_dual_fmac_f32 v43, v59, v142
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v29, v150 :: v_dual_fmac_f32 v53, v39, v150
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v49, v150 :: v_dual_fmac_f32 v63, v59, v150
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v30, v143 :: v_dual_fmac_f32 v33, v40, v143
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v50, v143 :: v_dual_fmac_f32 v43, v60, v143
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v30, v151 :: v_dual_fmac_f32 v53, v40, v151
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v50, v151 :: v_dual_fmac_f32 v63, v60, v151
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v31, v144 :: v_dual_fmac_f32 v33, v41, v144
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v51, v144 :: v_dual_fmac_f32 v43, v61, v144
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v31, v152 :: v_dual_fmac_f32 v53, v41, v152
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v51, v152 :: v_dual_fmac_f32 v63, v61, v152
	global_load_b128 v[137:140], v3, s[28:29]
	global_load_b128 v[141:144], v3, s[28:29] offset:16
	global_load_b128 v[145:148], v3, s[30:31]
	global_load_b128 v[149:152], v3, s[30:31] offset:16
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v68, v32, v68 :: v_dual_add_f32 v69, v69, v33
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v70, v42, v70 :: v_dual_add_f32 v71, v71, v43
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v80, v52, v80 :: v_dual_add_f32 v81, v81, v53
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v82, v62, v82 :: v_dual_add_f32 v83, v83, v63
	s_wait_loadcnt 0x13
	v_dual_mul_f32 v32, v24, v153 :: v_dual_mul_f32 v33, v35, v154
	v_dual_mul_f32 v42, v44, v153 :: v_dual_mul_f32 v43, v55, v154
	s_wait_loadcnt 0x11
	v_dual_mul_f32 v52, v24, v161 :: v_dual_mul_f32 v53, v35, v162
	v_dual_mul_f32 v62, v44, v161 :: v_dual_mul_f32 v63, v55, v162
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v25, v154 :: v_dual_fmac_f32 v33, v34, v153
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v45, v154 :: v_dual_fmac_f32 v43, v54, v153
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v25, v162 :: v_dual_fmac_f32 v53, v34, v161
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v45, v162 :: v_dual_fmac_f32 v63, v54, v161
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v26, v155 :: v_dual_fmac_f32 v33, v36, v155
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v46, v155 :: v_dual_fmac_f32 v43, v56, v155
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v26, v163 :: v_dual_fmac_f32 v53, v36, v163
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v46, v163 :: v_dual_fmac_f32 v63, v56, v163
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v27, v156 :: v_dual_fmac_f32 v33, v37, v156
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v47, v156 :: v_dual_fmac_f32 v43, v57, v156
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v27, v164 :: v_dual_fmac_f32 v53, v37, v164
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v47, v164 :: v_dual_fmac_f32 v63, v57, v164
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v28, v157 :: v_dual_fmac_f32 v33, v38, v157
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v48, v157 :: v_dual_fmac_f32 v43, v58, v157
	s_wait_loadcnt 0x10
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v28, v165 :: v_dual_fmac_f32 v53, v38, v165
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v48, v165 :: v_dual_fmac_f32 v63, v58, v165
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v29, v158 :: v_dual_fmac_f32 v33, v39, v158
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v49, v158 :: v_dual_fmac_f32 v43, v59, v158
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v29, v166 :: v_dual_fmac_f32 v53, v39, v166
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v49, v166 :: v_dual_fmac_f32 v63, v59, v166
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v30, v159 :: v_dual_fmac_f32 v33, v40, v159
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v50, v159 :: v_dual_fmac_f32 v43, v60, v159
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v30, v167 :: v_dual_fmac_f32 v53, v40, v167
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v50, v167 :: v_dual_fmac_f32 v63, v60, v167
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v31, v160 :: v_dual_fmac_f32 v33, v41, v160
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v51, v160 :: v_dual_fmac_f32 v43, v61, v160
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v31, v168 :: v_dual_fmac_f32 v53, v41, v168
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v51, v168 :: v_dual_fmac_f32 v63, v61, v168
	global_load_b128 v[153:156], v3, s[32:33]
	global_load_b128 v[157:160], v3, s[32:33] offset:16
	global_load_b128 v[161:164], v3, s[34:35]
	global_load_b128 v[165:168], v3, s[34:35] offset:16
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v92, v32, v92 :: v_dual_add_f32 v93, v93, v33
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v94, v42, v94 :: v_dual_add_f32 v95, v95, v43
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v104, v52, v104 :: v_dual_add_f32 v105, v105, v53
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v106, v62, v106 :: v_dual_add_f32 v107, v107, v63
	s_wait_loadcnt 0x13
	v_dual_mul_f32 v32, v24, v169 :: v_dual_mul_f32 v33, v35, v170
	v_dual_mul_f32 v42, v44, v169 :: v_dual_mul_f32 v43, v55, v170
	s_wait_loadcnt 0x11
	v_dual_mul_f32 v52, v24, v177 :: v_dual_mul_f32 v53, v35, v178
	v_dual_mul_f32 v62, v44, v177 :: v_dual_mul_f32 v63, v55, v178
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v25, v170 :: v_dual_fmac_f32 v33, v34, v169
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v45, v170 :: v_dual_fmac_f32 v43, v54, v169
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v25, v178 :: v_dual_fmac_f32 v53, v34, v177
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v45, v178 :: v_dual_fmac_f32 v63, v54, v177
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v26, v171 :: v_dual_fmac_f32 v33, v36, v171
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v46, v171 :: v_dual_fmac_f32 v43, v56, v171
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v26, v179 :: v_dual_fmac_f32 v53, v36, v179
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v46, v179 :: v_dual_fmac_f32 v63, v56, v179
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v27, v172 :: v_dual_fmac_f32 v33, v37, v172
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v47, v172 :: v_dual_fmac_f32 v43, v57, v172
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v27, v180 :: v_dual_fmac_f32 v53, v37, v180
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v47, v180 :: v_dual_fmac_f32 v63, v57, v180
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v28, v173 :: v_dual_fmac_f32 v33, v38, v173
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v48, v173 :: v_dual_fmac_f32 v43, v58, v173
	s_wait_loadcnt 0x10
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v28, v181 :: v_dual_fmac_f32 v53, v38, v181
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v48, v181 :: v_dual_fmac_f32 v63, v58, v181
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v29, v174 :: v_dual_fmac_f32 v33, v39, v174
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v49, v174 :: v_dual_fmac_f32 v43, v59, v174
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v29, v182 :: v_dual_fmac_f32 v53, v39, v182
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v49, v182 :: v_dual_fmac_f32 v63, v59, v182
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v30, v175 :: v_dual_fmac_f32 v33, v40, v175
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v50, v175 :: v_dual_fmac_f32 v43, v60, v175
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v30, v183 :: v_dual_fmac_f32 v53, v40, v183
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v50, v183 :: v_dual_fmac_f32 v63, v60, v183
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v31, v176 :: v_dual_fmac_f32 v33, v41, v176
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v51, v176 :: v_dual_fmac_f32 v43, v61, v176
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v31, v184 :: v_dual_fmac_f32 v53, v41, v184
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v51, v184 :: v_dual_fmac_f32 v63, v61, v184
	global_load_b128 v[169:172], v3, s[36:37]
	global_load_b128 v[173:176], v3, s[36:37] offset:16
	global_load_b128 v[177:180], v3, s[38:39]
	global_load_b128 v[181:184], v3, s[38:39] offset:16
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v116, v32, v116 :: v_dual_add_f32 v117, v117, v33
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v118, v42, v118 :: v_dual_add_f32 v119, v119, v43
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v128, v52, v128 :: v_dual_add_f32 v129, v129, v53
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v130, v62, v130 :: v_dual_add_f32 v131, v131, v63
	s_mov_b32 s15, s16
	v_mov_b32_e32 v2, v3
	s_add_co_i32 s2, s2, -1
	s_cmp_lg_u32 s2, 0
	s_cbranch_scc1 .Lrx_b6_s1
	s_wait_loadcnt 0x0
	.Lrx_b6_s1_done:
	v_dual_add_f32 v64, v64, v68 :: v_dual_add_f32 v65, v65, v69
	v_dual_add_f32 v66, v66, v70 :: v_dual_add_f32 v67, v67, v71
	v_dual_add_f32 v76, v76, v80 :: v_dual_add_f32 v77, v77, v81
	v_dual_add_f32 v78, v78, v82 :: v_dual_add_f32 v79, v79, v83
	v_dual_add_f32 v88, v88, v92 :: v_dual_add_f32 v89, v89, v93
	v_dual_add_f32 v90, v90, v94 :: v_dual_add_f32 v91, v91, v95
	v_dual_add_f32 v100, v100, v104 :: v_dual_add_f32 v101, v101, v105
	v_dual_add_f32 v102, v102, v106 :: v_dual_add_f32 v103, v103, v107
	v_dual_add_f32 v112, v112, v116 :: v_dual_add_f32 v113, v113, v117
	v_dual_add_f32 v114, v114, v118 :: v_dual_add_f32 v115, v115, v119
	v_dual_add_f32 v124, v124, v128 :: v_dual_add_f32 v125, v125, v129
	v_dual_add_f32 v126, v126, v130 :: v_dual_add_f32 v127, v127, v131
	v_mov_b32_e32 v68, 0
	v_mov_b32_e32 v69, 0
	v_mov_b32_e32 v70, 0
	v_mov_b32_e32 v71, 0
	v_mov_b32_e32 v80, 0
	v_mov_b32_e32 v81, 0
	v_mov_b32_e32 v82, 0
	v_mov_b32_e32 v83, 0
	v_mov_b32_e32 v92, 0
	v_mov_b32_e32 v93, 0
	v_mov_b32_e32 v94, 0
	v_mov_b32_e32 v95, 0
	v_mov_b32_e32 v104, 0
	v_mov_b32_e32 v105, 0
	v_mov_b32_e32 v106, 0
	v_mov_b32_e32 v107, 0
	v_mov_b32_e32 v116, 0
	v_mov_b32_e32 v117, 0
	v_mov_b32_e32 v118, 0
	v_mov_b32_e32 v119, 0
	v_mov_b32_e32 v128, 0
	v_mov_b32_e32 v129, 0
	v_mov_b32_e32 v130, 0
	v_mov_b32_e32 v131, 0
	s_add_co_i32 s2, s14, 1
	s_lshr_b32 s2, s2, 2
	s_cmp_eq_u32 s2, 0
	s_cbranch_scc1 .Lrx_b6_s2_done
	s_mov_b32 s15, 0x110
	v_lshlrev_b32_e32 v2, 5, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v2, 0x800, v2
	buffer_load_b64 v[12:13], v8, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v20, v4, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[14:15], v9, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v21, v5, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[16:17], v10, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v22, v6, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[18:19], v11, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v23, v7, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	global_load_b128 v[137:140], v2, s[28:29]
	global_load_b128 v[141:144], v2, s[28:29] offset:16
	global_load_b128 v[145:148], v2, s[30:31]
	global_load_b128 v[149:152], v2, s[30:31] offset:16
	global_load_b128 v[153:156], v2, s[32:33]
	global_load_b128 v[157:160], v2, s[32:33] offset:16
	global_load_b128 v[161:164], v2, s[34:35]
	global_load_b128 v[165:168], v2, s[34:35] offset:16
	global_load_b128 v[169:172], v2, s[36:37]
	global_load_b128 v[173:176], v2, s[36:37] offset:16
	global_load_b128 v[177:180], v2, s[38:39]
	global_load_b128 v[181:184], v2, s[38:39] offset:16
	.Lrx_b6_s2:
	s_cmp_gt_u32 s2, 1
	s_cselect_b32 s17, 0x220, 0
	s_cselect_b32 s18, 0x1000, 0
	s_add_co_i32 s16, s15, s17
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v3, s18, v2
	v_cmp_gt_u32_e32 vcc_lo, 16, v0
	s_wait_loadcnt 0x13
	v_cndmask_b32_e32 v12, v13, v12, vcc_lo
	s_wait_loadcnt 0x12
	v_and_b32_e32 v30, 0xf0f0f0f, v20
	v_lshrrev_b32_e32 v31, 4, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v31, 0xf0f0f0f, v31
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v24, v30
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v26, v30
	v_cvt_f32_ubyte2_e32 v28, v30
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v25, v31
	v_cvt_f32_ubyte1_e32 v27, v31
	v_cvt_f32_ubyte2_e32 v29, v31
	v_cvt_f32_ubyte3_e32 v30, v30
	v_cvt_f32_ubyte3_e32 v31, v31
	v_fma_mix_f32 v24, v12, v24, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v25, v12, v25, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v26, v12, v26, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v27, v12, v27, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v28, v12, v28, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v29, v12, v29, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v30, v12, v30, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v31, v12, v31, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x11
	v_cndmask_b32_e32 v14, v15, v14, vcc_lo
	s_wait_loadcnt 0x10
	v_and_b32_e32 v40, 0xf0f0f0f, v21
	v_lshrrev_b32_e32 v41, 4, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v41, 0xf0f0f0f, v41
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v34, v40
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v36, v40
	v_cvt_f32_ubyte2_e32 v38, v40
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v35, v41
	v_cvt_f32_ubyte1_e32 v37, v41
	v_cvt_f32_ubyte2_e32 v39, v41
	v_cvt_f32_ubyte3_e32 v40, v40
	v_cvt_f32_ubyte3_e32 v41, v41
	v_fma_mix_f32 v34, v14, v34, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v35, v14, v35, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v36, v14, v36, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v37, v14, v37, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v38, v14, v38, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v39, v14, v39, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v40, v14, v40, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v41, v14, v41, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xf
	v_cndmask_b32_e32 v16, v17, v16, vcc_lo
	s_wait_loadcnt 0xe
	v_and_b32_e32 v50, 0xf0f0f0f, v22
	v_lshrrev_b32_e32 v51, 4, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v51, 0xf0f0f0f, v51
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v44, v50
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v46, v50
	v_cvt_f32_ubyte2_e32 v48, v50
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v45, v51
	v_cvt_f32_ubyte1_e32 v47, v51
	v_cvt_f32_ubyte2_e32 v49, v51
	v_cvt_f32_ubyte3_e32 v50, v50
	v_cvt_f32_ubyte3_e32 v51, v51
	v_fma_mix_f32 v44, v16, v44, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v16, v45, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v16, v46, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v16, v47, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v48, v16, v48, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v49, v16, v49, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v16, v50, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v16, v51, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xd
	v_cndmask_b32_e32 v18, v19, v18, vcc_lo
	s_wait_loadcnt 0xc
	v_and_b32_e32 v60, 0xf0f0f0f, v23
	v_lshrrev_b32_e32 v61, 4, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v61, 0xf0f0f0f, v61
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v54, v60
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v56, v60
	v_cvt_f32_ubyte2_e32 v58, v60
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v55, v61
	v_cvt_f32_ubyte1_e32 v57, v61
	v_cvt_f32_ubyte2_e32 v59, v61
	v_cvt_f32_ubyte3_e32 v60, v60
	v_cvt_f32_ubyte3_e32 v61, v61
	v_fma_mix_f32 v54, v18, v54, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v18, v55, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v18, v56, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v18, v57, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v58, v18, v58, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v59, v18, v59, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v18, v60, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v18, v61, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	buffer_load_b64 v[12:13], v8, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v20, v4, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[14:15], v9, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v21, v5, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[16:17], v10, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v22, v6, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[18:19], v11, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v23, v7, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	s_wait_loadcnt 0x13
	v_dual_mul_f32 v32, v24, v137 :: v_dual_mul_f32 v33, v35, v138
	v_dual_mul_f32 v42, v44, v137 :: v_dual_mul_f32 v43, v55, v138
	s_wait_loadcnt 0x11
	v_dual_mul_f32 v52, v24, v145 :: v_dual_mul_f32 v53, v35, v146
	v_dual_mul_f32 v62, v44, v145 :: v_dual_mul_f32 v63, v55, v146
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v25, v138 :: v_dual_fmac_f32 v33, v34, v137
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v45, v138 :: v_dual_fmac_f32 v43, v54, v137
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v25, v146 :: v_dual_fmac_f32 v53, v34, v145
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v45, v146 :: v_dual_fmac_f32 v63, v54, v145
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v26, v139 :: v_dual_fmac_f32 v33, v36, v139
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v46, v139 :: v_dual_fmac_f32 v43, v56, v139
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v26, v147 :: v_dual_fmac_f32 v53, v36, v147
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v46, v147 :: v_dual_fmac_f32 v63, v56, v147
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v27, v140 :: v_dual_fmac_f32 v33, v37, v140
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v47, v140 :: v_dual_fmac_f32 v43, v57, v140
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v27, v148 :: v_dual_fmac_f32 v53, v37, v148
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v47, v148 :: v_dual_fmac_f32 v63, v57, v148
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v28, v141 :: v_dual_fmac_f32 v33, v38, v141
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v48, v141 :: v_dual_fmac_f32 v43, v58, v141
	s_wait_loadcnt 0x10
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v28, v149 :: v_dual_fmac_f32 v53, v38, v149
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v48, v149 :: v_dual_fmac_f32 v63, v58, v149
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v29, v142 :: v_dual_fmac_f32 v33, v39, v142
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v49, v142 :: v_dual_fmac_f32 v43, v59, v142
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v29, v150 :: v_dual_fmac_f32 v53, v39, v150
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v49, v150 :: v_dual_fmac_f32 v63, v59, v150
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v30, v143 :: v_dual_fmac_f32 v33, v40, v143
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v50, v143 :: v_dual_fmac_f32 v43, v60, v143
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v30, v151 :: v_dual_fmac_f32 v53, v40, v151
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v50, v151 :: v_dual_fmac_f32 v63, v60, v151
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v31, v144 :: v_dual_fmac_f32 v33, v41, v144
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v51, v144 :: v_dual_fmac_f32 v43, v61, v144
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v31, v152 :: v_dual_fmac_f32 v53, v41, v152
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v51, v152 :: v_dual_fmac_f32 v63, v61, v152
	global_load_b128 v[137:140], v3, s[28:29]
	global_load_b128 v[141:144], v3, s[28:29] offset:16
	global_load_b128 v[145:148], v3, s[30:31]
	global_load_b128 v[149:152], v3, s[30:31] offset:16
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v68, v32, v68 :: v_dual_add_f32 v69, v69, v33
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v70, v42, v70 :: v_dual_add_f32 v71, v71, v43
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v80, v52, v80 :: v_dual_add_f32 v81, v81, v53
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v82, v62, v82 :: v_dual_add_f32 v83, v83, v63
	s_wait_loadcnt 0x13
	v_dual_mul_f32 v32, v24, v153 :: v_dual_mul_f32 v33, v35, v154
	v_dual_mul_f32 v42, v44, v153 :: v_dual_mul_f32 v43, v55, v154
	s_wait_loadcnt 0x11
	v_dual_mul_f32 v52, v24, v161 :: v_dual_mul_f32 v53, v35, v162
	v_dual_mul_f32 v62, v44, v161 :: v_dual_mul_f32 v63, v55, v162
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v25, v154 :: v_dual_fmac_f32 v33, v34, v153
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v45, v154 :: v_dual_fmac_f32 v43, v54, v153
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v25, v162 :: v_dual_fmac_f32 v53, v34, v161
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v45, v162 :: v_dual_fmac_f32 v63, v54, v161
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v26, v155 :: v_dual_fmac_f32 v33, v36, v155
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v46, v155 :: v_dual_fmac_f32 v43, v56, v155
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v26, v163 :: v_dual_fmac_f32 v53, v36, v163
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v46, v163 :: v_dual_fmac_f32 v63, v56, v163
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v27, v156 :: v_dual_fmac_f32 v33, v37, v156
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v47, v156 :: v_dual_fmac_f32 v43, v57, v156
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v27, v164 :: v_dual_fmac_f32 v53, v37, v164
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v47, v164 :: v_dual_fmac_f32 v63, v57, v164
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v28, v157 :: v_dual_fmac_f32 v33, v38, v157
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v48, v157 :: v_dual_fmac_f32 v43, v58, v157
	s_wait_loadcnt 0x10
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v28, v165 :: v_dual_fmac_f32 v53, v38, v165
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v48, v165 :: v_dual_fmac_f32 v63, v58, v165
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v29, v158 :: v_dual_fmac_f32 v33, v39, v158
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v49, v158 :: v_dual_fmac_f32 v43, v59, v158
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v29, v166 :: v_dual_fmac_f32 v53, v39, v166
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v49, v166 :: v_dual_fmac_f32 v63, v59, v166
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v30, v159 :: v_dual_fmac_f32 v33, v40, v159
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v50, v159 :: v_dual_fmac_f32 v43, v60, v159
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v30, v167 :: v_dual_fmac_f32 v53, v40, v167
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v50, v167 :: v_dual_fmac_f32 v63, v60, v167
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v31, v160 :: v_dual_fmac_f32 v33, v41, v160
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v51, v160 :: v_dual_fmac_f32 v43, v61, v160
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v31, v168 :: v_dual_fmac_f32 v53, v41, v168
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v51, v168 :: v_dual_fmac_f32 v63, v61, v168
	global_load_b128 v[153:156], v3, s[32:33]
	global_load_b128 v[157:160], v3, s[32:33] offset:16
	global_load_b128 v[161:164], v3, s[34:35]
	global_load_b128 v[165:168], v3, s[34:35] offset:16
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v92, v32, v92 :: v_dual_add_f32 v93, v93, v33
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v94, v42, v94 :: v_dual_add_f32 v95, v95, v43
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v104, v52, v104 :: v_dual_add_f32 v105, v105, v53
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v106, v62, v106 :: v_dual_add_f32 v107, v107, v63
	s_wait_loadcnt 0x13
	v_dual_mul_f32 v32, v24, v169 :: v_dual_mul_f32 v33, v35, v170
	v_dual_mul_f32 v42, v44, v169 :: v_dual_mul_f32 v43, v55, v170
	s_wait_loadcnt 0x11
	v_dual_mul_f32 v52, v24, v177 :: v_dual_mul_f32 v53, v35, v178
	v_dual_mul_f32 v62, v44, v177 :: v_dual_mul_f32 v63, v55, v178
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v25, v170 :: v_dual_fmac_f32 v33, v34, v169
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v45, v170 :: v_dual_fmac_f32 v43, v54, v169
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v25, v178 :: v_dual_fmac_f32 v53, v34, v177
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v45, v178 :: v_dual_fmac_f32 v63, v54, v177
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v26, v171 :: v_dual_fmac_f32 v33, v36, v171
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v46, v171 :: v_dual_fmac_f32 v43, v56, v171
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v26, v179 :: v_dual_fmac_f32 v53, v36, v179
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v46, v179 :: v_dual_fmac_f32 v63, v56, v179
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v27, v172 :: v_dual_fmac_f32 v33, v37, v172
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v47, v172 :: v_dual_fmac_f32 v43, v57, v172
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v27, v180 :: v_dual_fmac_f32 v53, v37, v180
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v47, v180 :: v_dual_fmac_f32 v63, v57, v180
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v28, v173 :: v_dual_fmac_f32 v33, v38, v173
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v48, v173 :: v_dual_fmac_f32 v43, v58, v173
	s_wait_loadcnt 0x10
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v28, v181 :: v_dual_fmac_f32 v53, v38, v181
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v48, v181 :: v_dual_fmac_f32 v63, v58, v181
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v29, v174 :: v_dual_fmac_f32 v33, v39, v174
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v49, v174 :: v_dual_fmac_f32 v43, v59, v174
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v29, v182 :: v_dual_fmac_f32 v53, v39, v182
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v49, v182 :: v_dual_fmac_f32 v63, v59, v182
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v30, v175 :: v_dual_fmac_f32 v33, v40, v175
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v50, v175 :: v_dual_fmac_f32 v43, v60, v175
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v30, v183 :: v_dual_fmac_f32 v53, v40, v183
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v50, v183 :: v_dual_fmac_f32 v63, v60, v183
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v31, v176 :: v_dual_fmac_f32 v33, v41, v176
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v51, v176 :: v_dual_fmac_f32 v43, v61, v176
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v31, v184 :: v_dual_fmac_f32 v53, v41, v184
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v51, v184 :: v_dual_fmac_f32 v63, v61, v184
	global_load_b128 v[169:172], v3, s[36:37]
	global_load_b128 v[173:176], v3, s[36:37] offset:16
	global_load_b128 v[177:180], v3, s[38:39]
	global_load_b128 v[181:184], v3, s[38:39] offset:16
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v116, v32, v116 :: v_dual_add_f32 v117, v117, v33
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v118, v42, v118 :: v_dual_add_f32 v119, v119, v43
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v128, v52, v128 :: v_dual_add_f32 v129, v129, v53
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v130, v62, v130 :: v_dual_add_f32 v131, v131, v63
	s_mov_b32 s15, s16
	v_mov_b32_e32 v2, v3
	s_add_co_i32 s2, s2, -1
	s_cmp_lg_u32 s2, 0
	s_cbranch_scc1 .Lrx_b6_s2
	s_wait_loadcnt 0x0
	.Lrx_b6_s2_done:
	s_add_co_i32 s2, s14, 0
	s_lshr_b32 s2, s2, 2
	s_cmp_eq_u32 s2, 0
	s_cbranch_scc1 .Lrx_b6_s3_done
	s_mov_b32 s15, 0x198
	v_lshlrev_b32_e32 v2, 5, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v2, 0xc00, v2
	buffer_load_b64 v[12:13], v8, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v20, v4, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[14:15], v9, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v21, v5, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[16:17], v10, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v22, v6, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[18:19], v11, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v23, v7, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	global_load_b128 v[137:140], v2, s[28:29]
	global_load_b128 v[141:144], v2, s[28:29] offset:16
	global_load_b128 v[145:148], v2, s[30:31]
	global_load_b128 v[149:152], v2, s[30:31] offset:16
	global_load_b128 v[153:156], v2, s[32:33]
	global_load_b128 v[157:160], v2, s[32:33] offset:16
	global_load_b128 v[161:164], v2, s[34:35]
	global_load_b128 v[165:168], v2, s[34:35] offset:16
	global_load_b128 v[169:172], v2, s[36:37]
	global_load_b128 v[173:176], v2, s[36:37] offset:16
	global_load_b128 v[177:180], v2, s[38:39]
	global_load_b128 v[181:184], v2, s[38:39] offset:16
	.Lrx_b6_s3:
	s_cmp_gt_u32 s2, 1
	s_cselect_b32 s17, 0x220, 0
	s_cselect_b32 s18, 0x1000, 0
	s_add_co_i32 s16, s15, s17
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v3, s18, v2
	v_cmp_gt_u32_e32 vcc_lo, 16, v0
	s_wait_loadcnt 0x13
	v_cndmask_b32_e32 v12, v13, v12, vcc_lo
	s_wait_loadcnt 0x12
	v_and_b32_e32 v30, 0xf0f0f0f, v20
	v_lshrrev_b32_e32 v31, 4, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v31, 0xf0f0f0f, v31
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v24, v30
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v26, v30
	v_cvt_f32_ubyte2_e32 v28, v30
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v25, v31
	v_cvt_f32_ubyte1_e32 v27, v31
	v_cvt_f32_ubyte2_e32 v29, v31
	v_cvt_f32_ubyte3_e32 v30, v30
	v_cvt_f32_ubyte3_e32 v31, v31
	v_fma_mix_f32 v24, v12, v24, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v25, v12, v25, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v26, v12, v26, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v27, v12, v27, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v28, v12, v28, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v29, v12, v29, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v30, v12, v30, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v31, v12, v31, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x11
	v_cndmask_b32_e32 v14, v15, v14, vcc_lo
	s_wait_loadcnt 0x10
	v_and_b32_e32 v40, 0xf0f0f0f, v21
	v_lshrrev_b32_e32 v41, 4, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v41, 0xf0f0f0f, v41
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v34, v40
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v36, v40
	v_cvt_f32_ubyte2_e32 v38, v40
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v35, v41
	v_cvt_f32_ubyte1_e32 v37, v41
	v_cvt_f32_ubyte2_e32 v39, v41
	v_cvt_f32_ubyte3_e32 v40, v40
	v_cvt_f32_ubyte3_e32 v41, v41
	v_fma_mix_f32 v34, v14, v34, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v35, v14, v35, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v36, v14, v36, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v37, v14, v37, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v38, v14, v38, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v39, v14, v39, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v40, v14, v40, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v41, v14, v41, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xf
	v_cndmask_b32_e32 v16, v17, v16, vcc_lo
	s_wait_loadcnt 0xe
	v_and_b32_e32 v50, 0xf0f0f0f, v22
	v_lshrrev_b32_e32 v51, 4, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v51, 0xf0f0f0f, v51
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v44, v50
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v46, v50
	v_cvt_f32_ubyte2_e32 v48, v50
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v45, v51
	v_cvt_f32_ubyte1_e32 v47, v51
	v_cvt_f32_ubyte2_e32 v49, v51
	v_cvt_f32_ubyte3_e32 v50, v50
	v_cvt_f32_ubyte3_e32 v51, v51
	v_fma_mix_f32 v44, v16, v44, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v16, v45, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v16, v46, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v16, v47, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v48, v16, v48, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v49, v16, v49, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v16, v50, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v16, v51, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xd
	v_cndmask_b32_e32 v18, v19, v18, vcc_lo
	s_wait_loadcnt 0xc
	v_and_b32_e32 v60, 0xf0f0f0f, v23
	v_lshrrev_b32_e32 v61, 4, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v61, 0xf0f0f0f, v61
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v54, v60
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v56, v60
	v_cvt_f32_ubyte2_e32 v58, v60
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v55, v61
	v_cvt_f32_ubyte1_e32 v57, v61
	v_cvt_f32_ubyte2_e32 v59, v61
	v_cvt_f32_ubyte3_e32 v60, v60
	v_cvt_f32_ubyte3_e32 v61, v61
	v_fma_mix_f32 v54, v18, v54, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v18, v55, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v18, v56, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v18, v57, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v58, v18, v58, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v59, v18, v59, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v18, v60, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v18, v61, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	buffer_load_b64 v[12:13], v8, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v20, v4, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[14:15], v9, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v21, v5, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[16:17], v10, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v22, v6, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[18:19], v11, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v23, v7, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	s_wait_loadcnt 0x13
	v_dual_mul_f32 v32, v24, v137 :: v_dual_mul_f32 v33, v35, v138
	v_dual_mul_f32 v42, v44, v137 :: v_dual_mul_f32 v43, v55, v138
	s_wait_loadcnt 0x11
	v_dual_mul_f32 v52, v24, v145 :: v_dual_mul_f32 v53, v35, v146
	v_dual_mul_f32 v62, v44, v145 :: v_dual_mul_f32 v63, v55, v146
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v25, v138 :: v_dual_fmac_f32 v33, v34, v137
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v45, v138 :: v_dual_fmac_f32 v43, v54, v137
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v25, v146 :: v_dual_fmac_f32 v53, v34, v145
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v45, v146 :: v_dual_fmac_f32 v63, v54, v145
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v26, v139 :: v_dual_fmac_f32 v33, v36, v139
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v46, v139 :: v_dual_fmac_f32 v43, v56, v139
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v26, v147 :: v_dual_fmac_f32 v53, v36, v147
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v46, v147 :: v_dual_fmac_f32 v63, v56, v147
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v27, v140 :: v_dual_fmac_f32 v33, v37, v140
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v47, v140 :: v_dual_fmac_f32 v43, v57, v140
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v27, v148 :: v_dual_fmac_f32 v53, v37, v148
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v47, v148 :: v_dual_fmac_f32 v63, v57, v148
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v28, v141 :: v_dual_fmac_f32 v33, v38, v141
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v48, v141 :: v_dual_fmac_f32 v43, v58, v141
	s_wait_loadcnt 0x10
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v28, v149 :: v_dual_fmac_f32 v53, v38, v149
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v48, v149 :: v_dual_fmac_f32 v63, v58, v149
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v29, v142 :: v_dual_fmac_f32 v33, v39, v142
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v49, v142 :: v_dual_fmac_f32 v43, v59, v142
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v29, v150 :: v_dual_fmac_f32 v53, v39, v150
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v49, v150 :: v_dual_fmac_f32 v63, v59, v150
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v30, v143 :: v_dual_fmac_f32 v33, v40, v143
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v50, v143 :: v_dual_fmac_f32 v43, v60, v143
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v30, v151 :: v_dual_fmac_f32 v53, v40, v151
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v50, v151 :: v_dual_fmac_f32 v63, v60, v151
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v31, v144 :: v_dual_fmac_f32 v33, v41, v144
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v51, v144 :: v_dual_fmac_f32 v43, v61, v144
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v31, v152 :: v_dual_fmac_f32 v53, v41, v152
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v51, v152 :: v_dual_fmac_f32 v63, v61, v152
	global_load_b128 v[137:140], v3, s[28:29]
	global_load_b128 v[141:144], v3, s[28:29] offset:16
	global_load_b128 v[145:148], v3, s[30:31]
	global_load_b128 v[149:152], v3, s[30:31] offset:16
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v72, v32, v72 :: v_dual_add_f32 v73, v73, v33
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v74, v42, v74 :: v_dual_add_f32 v75, v75, v43
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v84, v52, v84 :: v_dual_add_f32 v85, v85, v53
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v86, v62, v86 :: v_dual_add_f32 v87, v87, v63
	s_wait_loadcnt 0x13
	v_dual_mul_f32 v32, v24, v153 :: v_dual_mul_f32 v33, v35, v154
	v_dual_mul_f32 v42, v44, v153 :: v_dual_mul_f32 v43, v55, v154
	s_wait_loadcnt 0x11
	v_dual_mul_f32 v52, v24, v161 :: v_dual_mul_f32 v53, v35, v162
	v_dual_mul_f32 v62, v44, v161 :: v_dual_mul_f32 v63, v55, v162
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v25, v154 :: v_dual_fmac_f32 v33, v34, v153
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v45, v154 :: v_dual_fmac_f32 v43, v54, v153
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v25, v162 :: v_dual_fmac_f32 v53, v34, v161
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v45, v162 :: v_dual_fmac_f32 v63, v54, v161
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v26, v155 :: v_dual_fmac_f32 v33, v36, v155
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v46, v155 :: v_dual_fmac_f32 v43, v56, v155
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v26, v163 :: v_dual_fmac_f32 v53, v36, v163
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v46, v163 :: v_dual_fmac_f32 v63, v56, v163
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v27, v156 :: v_dual_fmac_f32 v33, v37, v156
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v47, v156 :: v_dual_fmac_f32 v43, v57, v156
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v27, v164 :: v_dual_fmac_f32 v53, v37, v164
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v47, v164 :: v_dual_fmac_f32 v63, v57, v164
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v28, v157 :: v_dual_fmac_f32 v33, v38, v157
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v48, v157 :: v_dual_fmac_f32 v43, v58, v157
	s_wait_loadcnt 0x10
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v28, v165 :: v_dual_fmac_f32 v53, v38, v165
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v48, v165 :: v_dual_fmac_f32 v63, v58, v165
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v29, v158 :: v_dual_fmac_f32 v33, v39, v158
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v49, v158 :: v_dual_fmac_f32 v43, v59, v158
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v29, v166 :: v_dual_fmac_f32 v53, v39, v166
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v49, v166 :: v_dual_fmac_f32 v63, v59, v166
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v30, v159 :: v_dual_fmac_f32 v33, v40, v159
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v50, v159 :: v_dual_fmac_f32 v43, v60, v159
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v30, v167 :: v_dual_fmac_f32 v53, v40, v167
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v50, v167 :: v_dual_fmac_f32 v63, v60, v167
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v31, v160 :: v_dual_fmac_f32 v33, v41, v160
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v51, v160 :: v_dual_fmac_f32 v43, v61, v160
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v31, v168 :: v_dual_fmac_f32 v53, v41, v168
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v51, v168 :: v_dual_fmac_f32 v63, v61, v168
	global_load_b128 v[153:156], v3, s[32:33]
	global_load_b128 v[157:160], v3, s[32:33] offset:16
	global_load_b128 v[161:164], v3, s[34:35]
	global_load_b128 v[165:168], v3, s[34:35] offset:16
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v96, v32, v96 :: v_dual_add_f32 v97, v97, v33
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v98, v42, v98 :: v_dual_add_f32 v99, v99, v43
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v108, v52, v108 :: v_dual_add_f32 v109, v109, v53
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v110, v62, v110 :: v_dual_add_f32 v111, v111, v63
	s_wait_loadcnt 0x13
	v_dual_mul_f32 v32, v24, v169 :: v_dual_mul_f32 v33, v35, v170
	v_dual_mul_f32 v42, v44, v169 :: v_dual_mul_f32 v43, v55, v170
	s_wait_loadcnt 0x11
	v_dual_mul_f32 v52, v24, v177 :: v_dual_mul_f32 v53, v35, v178
	v_dual_mul_f32 v62, v44, v177 :: v_dual_mul_f32 v63, v55, v178
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v25, v170 :: v_dual_fmac_f32 v33, v34, v169
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v45, v170 :: v_dual_fmac_f32 v43, v54, v169
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v25, v178 :: v_dual_fmac_f32 v53, v34, v177
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v45, v178 :: v_dual_fmac_f32 v63, v54, v177
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v26, v171 :: v_dual_fmac_f32 v33, v36, v171
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v46, v171 :: v_dual_fmac_f32 v43, v56, v171
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v26, v179 :: v_dual_fmac_f32 v53, v36, v179
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v46, v179 :: v_dual_fmac_f32 v63, v56, v179
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v27, v172 :: v_dual_fmac_f32 v33, v37, v172
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v47, v172 :: v_dual_fmac_f32 v43, v57, v172
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v27, v180 :: v_dual_fmac_f32 v53, v37, v180
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v47, v180 :: v_dual_fmac_f32 v63, v57, v180
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v28, v173 :: v_dual_fmac_f32 v33, v38, v173
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v48, v173 :: v_dual_fmac_f32 v43, v58, v173
	s_wait_loadcnt 0x10
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v28, v181 :: v_dual_fmac_f32 v53, v38, v181
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v48, v181 :: v_dual_fmac_f32 v63, v58, v181
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v29, v174 :: v_dual_fmac_f32 v33, v39, v174
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v49, v174 :: v_dual_fmac_f32 v43, v59, v174
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v29, v182 :: v_dual_fmac_f32 v53, v39, v182
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v49, v182 :: v_dual_fmac_f32 v63, v59, v182
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v30, v175 :: v_dual_fmac_f32 v33, v40, v175
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v50, v175 :: v_dual_fmac_f32 v43, v60, v175
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v30, v183 :: v_dual_fmac_f32 v53, v40, v183
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v50, v183 :: v_dual_fmac_f32 v63, v60, v183
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v31, v176 :: v_dual_fmac_f32 v33, v41, v176
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v51, v176 :: v_dual_fmac_f32 v43, v61, v176
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v31, v184 :: v_dual_fmac_f32 v53, v41, v184
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v51, v184 :: v_dual_fmac_f32 v63, v61, v184
	global_load_b128 v[169:172], v3, s[36:37]
	global_load_b128 v[173:176], v3, s[36:37] offset:16
	global_load_b128 v[177:180], v3, s[38:39]
	global_load_b128 v[181:184], v3, s[38:39] offset:16
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v120, v32, v120 :: v_dual_add_f32 v121, v121, v33
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v122, v42, v122 :: v_dual_add_f32 v123, v123, v43
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v132, v52, v132 :: v_dual_add_f32 v133, v133, v53
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v134, v62, v134 :: v_dual_add_f32 v135, v135, v63
	s_mov_b32 s15, s16
	v_mov_b32_e32 v2, v3
	s_add_co_i32 s2, s2, -1
	s_cmp_lg_u32 s2, 0
	s_cbranch_scc1 .Lrx_b6_s3
	s_wait_loadcnt 0x0
	.Lrx_b6_s3_done:
	v_dual_add_f32 v68, v68, v72 :: v_dual_add_f32 v69, v69, v73
	v_dual_add_f32 v70, v70, v74 :: v_dual_add_f32 v71, v71, v75
	v_dual_add_f32 v80, v80, v84 :: v_dual_add_f32 v81, v81, v85
	v_dual_add_f32 v82, v82, v86 :: v_dual_add_f32 v83, v83, v87
	v_dual_add_f32 v92, v92, v96 :: v_dual_add_f32 v93, v93, v97
	v_dual_add_f32 v94, v94, v98 :: v_dual_add_f32 v95, v95, v99
	v_dual_add_f32 v104, v104, v108 :: v_dual_add_f32 v105, v105, v109
	v_dual_add_f32 v106, v106, v110 :: v_dual_add_f32 v107, v107, v111
	v_dual_add_f32 v116, v116, v120 :: v_dual_add_f32 v117, v117, v121
	v_dual_add_f32 v118, v118, v122 :: v_dual_add_f32 v119, v119, v123
	v_dual_add_f32 v128, v128, v132 :: v_dual_add_f32 v129, v129, v133
	v_dual_add_f32 v130, v130, v134 :: v_dual_add_f32 v131, v131, v135
	v_dual_add_f32 v64, v64, v68 :: v_dual_add_f32 v65, v65, v69
	v_dual_add_f32 v66, v66, v70 :: v_dual_add_f32 v67, v67, v71
	v_dual_add_f32 v76, v76, v80 :: v_dual_add_f32 v77, v77, v81
	v_dual_add_f32 v78, v78, v82 :: v_dual_add_f32 v79, v79, v83
	v_dual_add_f32 v88, v88, v92 :: v_dual_add_f32 v89, v89, v93
	v_dual_add_f32 v90, v90, v94 :: v_dual_add_f32 v91, v91, v95
	v_dual_add_f32 v100, v100, v104 :: v_dual_add_f32 v101, v101, v105
	v_dual_add_f32 v102, v102, v106 :: v_dual_add_f32 v103, v103, v107
	v_dual_add_f32 v112, v112, v116 :: v_dual_add_f32 v113, v113, v117
	v_dual_add_f32 v114, v114, v118 :: v_dual_add_f32 v115, v115, v119
	v_dual_add_f32 v124, v124, v128 :: v_dual_add_f32 v125, v125, v129
	v_dual_add_f32 v126, v126, v130 :: v_dual_add_f32 v127, v127, v131
	ds_swizzle_b32 v24, v64 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v25, v65 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v26, v66 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v27, v67 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v28, v76 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v29, v77 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v30, v78 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v31, v79 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v32, v88 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v33, v89 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v34, v90 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v35, v91 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v36, v100 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v37, v101 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v38, v102 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v39, v103 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v40, v112 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v41, v113 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v42, v114 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v43, v115 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v44, v124 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v45, v125 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v46, v126 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v47, v127 offset:swizzle(BITMASK_PERM,"1pppp")
	s_wait_dscnt 0x17
	v_add_f32_e32 v64, v64, v24
	s_wait_dscnt 0x16
	v_add_f32_e32 v65, v65, v25
	s_wait_dscnt 0x15
	v_add_f32_e32 v66, v66, v26
	s_wait_dscnt 0x14
	v_add_f32_e32 v67, v67, v27
	s_wait_dscnt 0x13
	v_add_f32_e32 v76, v76, v28
	s_wait_dscnt 0x12
	v_add_f32_e32 v77, v77, v29
	s_wait_dscnt 0x11
	v_add_f32_e32 v78, v78, v30
	s_wait_dscnt 0x10
	v_add_f32_e32 v79, v79, v31
	s_wait_dscnt 0xf
	v_add_f32_e32 v88, v88, v32
	s_wait_dscnt 0xe
	v_add_f32_e32 v89, v89, v33
	s_wait_dscnt 0xd
	v_add_f32_e32 v90, v90, v34
	s_wait_dscnt 0xc
	v_add_f32_e32 v91, v91, v35
	s_wait_dscnt 0xb
	v_add_f32_e32 v100, v100, v36
	s_wait_dscnt 0xa
	v_add_f32_e32 v101, v101, v37
	s_wait_dscnt 0x9
	v_add_f32_e32 v102, v102, v38
	s_wait_dscnt 0x8
	v_add_f32_e32 v103, v103, v39
	s_wait_dscnt 0x7
	v_add_f32_e32 v112, v112, v40
	s_wait_dscnt 0x6
	v_add_f32_e32 v113, v113, v41
	s_wait_dscnt 0x5
	v_add_f32_e32 v114, v114, v42
	s_wait_dscnt 0x4
	v_add_f32_e32 v115, v115, v43
	s_wait_dscnt 0x3
	v_add_f32_e32 v124, v124, v44
	s_wait_dscnt 0x2
	v_add_f32_e32 v125, v125, v45
	s_wait_dscnt 0x1
	v_add_f32_e32 v126, v126, v46
	s_wait_dscnt 0x0
	v_add_f32_e32 v127, v127, v47
	v_cmp_gt_u32_e32 vcc_lo, 24, v0
	v_cndmask_b32_e64 v3, 0, 8, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v3, v3, v0, 2
	ds_bpermute_b32 v24, v3, v64
	ds_bpermute_b32 v25, v3, v65
	ds_bpermute_b32 v26, v3, v66
	ds_bpermute_b32 v27, v3, v67
	ds_bpermute_b32 v28, v3, v76
	ds_bpermute_b32 v29, v3, v77
	ds_bpermute_b32 v30, v3, v78
	ds_bpermute_b32 v31, v3, v79
	ds_bpermute_b32 v32, v3, v88
	ds_bpermute_b32 v33, v3, v89
	ds_bpermute_b32 v34, v3, v90
	ds_bpermute_b32 v35, v3, v91
	ds_bpermute_b32 v36, v3, v100
	ds_bpermute_b32 v37, v3, v101
	ds_bpermute_b32 v38, v3, v102
	ds_bpermute_b32 v39, v3, v103
	ds_bpermute_b32 v40, v3, v112
	ds_bpermute_b32 v41, v3, v113
	ds_bpermute_b32 v42, v3, v114
	ds_bpermute_b32 v43, v3, v115
	ds_bpermute_b32 v44, v3, v124
	ds_bpermute_b32 v45, v3, v125
	ds_bpermute_b32 v46, v3, v126
	ds_bpermute_b32 v47, v3, v127
	s_wait_dscnt 0x17
	v_add_f32_e32 v64, v64, v24
	s_wait_dscnt 0x16
	v_add_f32_e32 v65, v65, v25
	s_wait_dscnt 0x15
	v_add_f32_e32 v66, v66, v26
	s_wait_dscnt 0x14
	v_add_f32_e32 v67, v67, v27
	s_wait_dscnt 0x13
	v_add_f32_e32 v76, v76, v28
	s_wait_dscnt 0x12
	v_add_f32_e32 v77, v77, v29
	s_wait_dscnt 0x11
	v_add_f32_e32 v78, v78, v30
	s_wait_dscnt 0x10
	v_add_f32_e32 v79, v79, v31
	s_wait_dscnt 0xf
	v_add_f32_e32 v88, v88, v32
	s_wait_dscnt 0xe
	v_add_f32_e32 v89, v89, v33
	s_wait_dscnt 0xd
	v_add_f32_e32 v90, v90, v34
	s_wait_dscnt 0xc
	v_add_f32_e32 v91, v91, v35
	s_wait_dscnt 0xb
	v_add_f32_e32 v100, v100, v36
	s_wait_dscnt 0xa
	v_add_f32_e32 v101, v101, v37
	s_wait_dscnt 0x9
	v_add_f32_e32 v102, v102, v38
	s_wait_dscnt 0x8
	v_add_f32_e32 v103, v103, v39
	s_wait_dscnt 0x7
	v_add_f32_e32 v112, v112, v40
	s_wait_dscnt 0x6
	v_add_f32_e32 v113, v113, v41
	s_wait_dscnt 0x5
	v_add_f32_e32 v114, v114, v42
	s_wait_dscnt 0x4
	v_add_f32_e32 v115, v115, v43
	s_wait_dscnt 0x3
	v_add_f32_e32 v124, v124, v44
	s_wait_dscnt 0x2
	v_add_f32_e32 v125, v125, v45
	s_wait_dscnt 0x1
	v_add_f32_e32 v126, v126, v46
	s_wait_dscnt 0x0
	v_add_f32_e32 v127, v127, v47
	v_cmp_gt_u32_e32 vcc_lo, 28, v0
	v_cndmask_b32_e64 v3, 0, 4, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v3, v3, v0, 2
	ds_bpermute_b32 v24, v3, v64
	ds_bpermute_b32 v25, v3, v65
	ds_bpermute_b32 v26, v3, v66
	ds_bpermute_b32 v27, v3, v67
	ds_bpermute_b32 v28, v3, v76
	ds_bpermute_b32 v29, v3, v77
	ds_bpermute_b32 v30, v3, v78
	ds_bpermute_b32 v31, v3, v79
	ds_bpermute_b32 v32, v3, v88
	ds_bpermute_b32 v33, v3, v89
	ds_bpermute_b32 v34, v3, v90
	ds_bpermute_b32 v35, v3, v91
	ds_bpermute_b32 v36, v3, v100
	ds_bpermute_b32 v37, v3, v101
	ds_bpermute_b32 v38, v3, v102
	ds_bpermute_b32 v39, v3, v103
	ds_bpermute_b32 v40, v3, v112
	ds_bpermute_b32 v41, v3, v113
	ds_bpermute_b32 v42, v3, v114
	ds_bpermute_b32 v43, v3, v115
	ds_bpermute_b32 v44, v3, v124
	ds_bpermute_b32 v45, v3, v125
	ds_bpermute_b32 v46, v3, v126
	ds_bpermute_b32 v47, v3, v127
	s_wait_dscnt 0x17
	v_add_f32_e32 v64, v64, v24
	s_wait_dscnt 0x16
	v_add_f32_e32 v65, v65, v25
	s_wait_dscnt 0x15
	v_add_f32_e32 v66, v66, v26
	s_wait_dscnt 0x14
	v_add_f32_e32 v67, v67, v27
	s_wait_dscnt 0x13
	v_add_f32_e32 v76, v76, v28
	s_wait_dscnt 0x12
	v_add_f32_e32 v77, v77, v29
	s_wait_dscnt 0x11
	v_add_f32_e32 v78, v78, v30
	s_wait_dscnt 0x10
	v_add_f32_e32 v79, v79, v31
	s_wait_dscnt 0xf
	v_add_f32_e32 v88, v88, v32
	s_wait_dscnt 0xe
	v_add_f32_e32 v89, v89, v33
	s_wait_dscnt 0xd
	v_add_f32_e32 v90, v90, v34
	s_wait_dscnt 0xc
	v_add_f32_e32 v91, v91, v35
	s_wait_dscnt 0xb
	v_add_f32_e32 v100, v100, v36
	s_wait_dscnt 0xa
	v_add_f32_e32 v101, v101, v37
	s_wait_dscnt 0x9
	v_add_f32_e32 v102, v102, v38
	s_wait_dscnt 0x8
	v_add_f32_e32 v103, v103, v39
	s_wait_dscnt 0x7
	v_add_f32_e32 v112, v112, v40
	s_wait_dscnt 0x6
	v_add_f32_e32 v113, v113, v41
	s_wait_dscnt 0x5
	v_add_f32_e32 v114, v114, v42
	s_wait_dscnt 0x4
	v_add_f32_e32 v115, v115, v43
	s_wait_dscnt 0x3
	v_add_f32_e32 v124, v124, v44
	s_wait_dscnt 0x2
	v_add_f32_e32 v125, v125, v45
	s_wait_dscnt 0x1
	v_add_f32_e32 v126, v126, v46
	s_wait_dscnt 0x0
	v_add_f32_e32 v127, v127, v47
	v_cmp_gt_u32_e32 vcc_lo, 30, v0
	v_cndmask_b32_e64 v3, 0, 2, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v3, v3, v0, 2
	ds_bpermute_b32 v24, v3, v64
	ds_bpermute_b32 v25, v3, v65
	ds_bpermute_b32 v26, v3, v66
	ds_bpermute_b32 v27, v3, v67
	ds_bpermute_b32 v28, v3, v76
	ds_bpermute_b32 v29, v3, v77
	ds_bpermute_b32 v30, v3, v78
	ds_bpermute_b32 v31, v3, v79
	ds_bpermute_b32 v32, v3, v88
	ds_bpermute_b32 v33, v3, v89
	ds_bpermute_b32 v34, v3, v90
	ds_bpermute_b32 v35, v3, v91
	ds_bpermute_b32 v36, v3, v100
	ds_bpermute_b32 v37, v3, v101
	ds_bpermute_b32 v38, v3, v102
	ds_bpermute_b32 v39, v3, v103
	ds_bpermute_b32 v40, v3, v112
	ds_bpermute_b32 v41, v3, v113
	ds_bpermute_b32 v42, v3, v114
	ds_bpermute_b32 v43, v3, v115
	ds_bpermute_b32 v44, v3, v124
	ds_bpermute_b32 v45, v3, v125
	ds_bpermute_b32 v46, v3, v126
	ds_bpermute_b32 v47, v3, v127
	s_wait_dscnt 0x17
	v_add_f32_e32 v64, v64, v24
	s_wait_dscnt 0x16
	v_add_f32_e32 v65, v65, v25
	s_wait_dscnt 0x15
	v_add_f32_e32 v66, v66, v26
	s_wait_dscnt 0x14
	v_add_f32_e32 v67, v67, v27
	s_wait_dscnt 0x13
	v_add_f32_e32 v76, v76, v28
	s_wait_dscnt 0x12
	v_add_f32_e32 v77, v77, v29
	s_wait_dscnt 0x11
	v_add_f32_e32 v78, v78, v30
	s_wait_dscnt 0x10
	v_add_f32_e32 v79, v79, v31
	s_wait_dscnt 0xf
	v_add_f32_e32 v88, v88, v32
	s_wait_dscnt 0xe
	v_add_f32_e32 v89, v89, v33
	s_wait_dscnt 0xd
	v_add_f32_e32 v90, v90, v34
	s_wait_dscnt 0xc
	v_add_f32_e32 v91, v91, v35
	s_wait_dscnt 0xb
	v_add_f32_e32 v100, v100, v36
	s_wait_dscnt 0xa
	v_add_f32_e32 v101, v101, v37
	s_wait_dscnt 0x9
	v_add_f32_e32 v102, v102, v38
	s_wait_dscnt 0x8
	v_add_f32_e32 v103, v103, v39
	s_wait_dscnt 0x7
	v_add_f32_e32 v112, v112, v40
	s_wait_dscnt 0x6
	v_add_f32_e32 v113, v113, v41
	s_wait_dscnt 0x5
	v_add_f32_e32 v114, v114, v42
	s_wait_dscnt 0x4
	v_add_f32_e32 v115, v115, v43
	s_wait_dscnt 0x3
	v_add_f32_e32 v124, v124, v44
	s_wait_dscnt 0x2
	v_add_f32_e32 v125, v125, v45
	s_wait_dscnt 0x1
	v_add_f32_e32 v126, v126, v46
	s_wait_dscnt 0x0
	v_add_f32_e32 v127, v127, v47
	v_cmp_gt_u32_e32 vcc_lo, 31, v0
	v_cndmask_b32_e64 v3, 0, 1, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v3, v3, v0, 2
	ds_bpermute_b32 v24, v3, v64
	ds_bpermute_b32 v25, v3, v65
	ds_bpermute_b32 v26, v3, v66
	ds_bpermute_b32 v27, v3, v67
	ds_bpermute_b32 v28, v3, v76
	ds_bpermute_b32 v29, v3, v77
	ds_bpermute_b32 v30, v3, v78
	ds_bpermute_b32 v31, v3, v79
	ds_bpermute_b32 v32, v3, v88
	ds_bpermute_b32 v33, v3, v89
	ds_bpermute_b32 v34, v3, v90
	ds_bpermute_b32 v35, v3, v91
	ds_bpermute_b32 v36, v3, v100
	ds_bpermute_b32 v37, v3, v101
	ds_bpermute_b32 v38, v3, v102
	ds_bpermute_b32 v39, v3, v103
	ds_bpermute_b32 v40, v3, v112
	ds_bpermute_b32 v41, v3, v113
	ds_bpermute_b32 v42, v3, v114
	ds_bpermute_b32 v43, v3, v115
	ds_bpermute_b32 v44, v3, v124
	ds_bpermute_b32 v45, v3, v125
	ds_bpermute_b32 v46, v3, v126
	ds_bpermute_b32 v47, v3, v127
	s_wait_dscnt 0x17
	v_add_f32_e32 v64, v64, v24
	s_wait_dscnt 0x16
	v_add_f32_e32 v65, v65, v25
	s_wait_dscnt 0x15
	v_add_f32_e32 v66, v66, v26
	s_wait_dscnt 0x14
	v_add_f32_e32 v67, v67, v27
	s_wait_dscnt 0x13
	v_add_f32_e32 v76, v76, v28
	s_wait_dscnt 0x12
	v_add_f32_e32 v77, v77, v29
	s_wait_dscnt 0x11
	v_add_f32_e32 v78, v78, v30
	s_wait_dscnt 0x10
	v_add_f32_e32 v79, v79, v31
	s_wait_dscnt 0xf
	v_add_f32_e32 v88, v88, v32
	s_wait_dscnt 0xe
	v_add_f32_e32 v89, v89, v33
	s_wait_dscnt 0xd
	v_add_f32_e32 v90, v90, v34
	s_wait_dscnt 0xc
	v_add_f32_e32 v91, v91, v35
	s_wait_dscnt 0xb
	v_add_f32_e32 v100, v100, v36
	s_wait_dscnt 0xa
	v_add_f32_e32 v101, v101, v37
	s_wait_dscnt 0x9
	v_add_f32_e32 v102, v102, v38
	s_wait_dscnt 0x8
	v_add_f32_e32 v103, v103, v39
	s_wait_dscnt 0x7
	v_add_f32_e32 v112, v112, v40
	s_wait_dscnt 0x6
	v_add_f32_e32 v113, v113, v41
	s_wait_dscnt 0x5
	v_add_f32_e32 v114, v114, v42
	s_wait_dscnt 0x4
	v_add_f32_e32 v115, v115, v43
	s_wait_dscnt 0x3
	v_add_f32_e32 v124, v124, v44
	s_wait_dscnt 0x2
	v_add_f32_e32 v125, v125, v45
	s_wait_dscnt 0x1
	v_add_f32_e32 v126, v126, v46
	s_wait_dscnt 0x0
	v_add_f32_e32 v127, v127, v47
	v_cmpx_eq_u32_e32 0, v0
	s_lshl_b32 s46, s12, 2
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v145, s46
	s_mul_i32 s47, s45, 1
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s47, s47, s46
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v146, s47
	s_mul_i32 s47, s45, 2
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s47, s47, s46
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v147, s47
	s_mul_i32 s47, s45, 3
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s47, s47, s46
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v148, s47
	s_mul_i32 s47, s45, 4
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s47, s47, s46
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v149, s47
	s_mul_i32 s47, s45, 5
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s47, s47, s46
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v150, s47
	s_add_co_i32 s13, s12, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cbranch_scc0 .Lrx_b6_p0_single
	global_load_b32 v48, v145, s[8:9]
	global_load_b32 v49, v145, s[8:9] offset:4
	global_load_b32 v52, v146, s[8:9]
	global_load_b32 v53, v146, s[8:9] offset:4
	global_load_b32 v56, v147, s[8:9]
	global_load_b32 v57, v147, s[8:9] offset:4
	global_load_b32 v60, v148, s[8:9]
	global_load_b32 v61, v148, s[8:9] offset:4
	global_load_b32 v137, v149, s[8:9]
	global_load_b32 v138, v149, s[8:9] offset:4
	global_load_b32 v141, v150, s[8:9]
	global_load_b32 v142, v150, s[8:9] offset:4
	s_wait_loadcnt 0xb
	v_add_f32_e32 v48, v64, v48
	global_store_b32 v145, v48, s[8:9]
	s_wait_loadcnt 0xa
	v_add_f32_e32 v49, v65, v49
	global_store_b32 v145, v49, s[8:9] offset:4
	s_wait_loadcnt 0x9
	v_add_f32_e32 v52, v76, v52
	global_store_b32 v146, v52, s[8:9]
	s_wait_loadcnt 0x8
	v_add_f32_e32 v53, v77, v53
	global_store_b32 v146, v53, s[8:9] offset:4
	s_wait_loadcnt 0x7
	v_add_f32_e32 v56, v88, v56
	global_store_b32 v147, v56, s[8:9]
	s_wait_loadcnt 0x6
	v_add_f32_e32 v57, v89, v57
	global_store_b32 v147, v57, s[8:9] offset:4
	s_wait_loadcnt 0x5
	v_add_f32_e32 v60, v100, v60
	global_store_b32 v148, v60, s[8:9]
	s_wait_loadcnt 0x4
	v_add_f32_e32 v61, v101, v61
	global_store_b32 v148, v61, s[8:9] offset:4
	s_wait_loadcnt 0x3
	v_add_f32_e32 v137, v112, v137
	global_store_b32 v149, v137, s[8:9]
	s_wait_loadcnt 0x2
	v_add_f32_e32 v138, v113, v138
	global_store_b32 v149, v138, s[8:9] offset:4
	s_wait_loadcnt 0x1
	v_add_f32_e32 v141, v124, v141
	global_store_b32 v150, v141, s[8:9]
	s_wait_loadcnt 0x0
	v_add_f32_e32 v142, v125, v142
	global_store_b32 v150, v142, s[8:9] offset:4
	s_wait_storecnt 0x0
	s_branch .Lrx_b6_p0_next
	.Lrx_b6_p0_single:
	global_load_b32 v48, v145, s[8:9]
	global_load_b32 v52, v146, s[8:9]
	global_load_b32 v56, v147, s[8:9]
	global_load_b32 v60, v148, s[8:9]
	global_load_b32 v137, v149, s[8:9]
	global_load_b32 v141, v150, s[8:9]
	s_wait_loadcnt 0x5
	v_add_f32_e32 v48, v48, v64
	global_store_b32 v145, v48, s[8:9]
	s_wait_loadcnt 0x4
	v_add_f32_e32 v52, v52, v76
	global_store_b32 v146, v52, s[8:9]
	s_wait_loadcnt 0x3
	v_add_f32_e32 v56, v56, v88
	global_store_b32 v147, v56, s[8:9]
	s_wait_loadcnt 0x2
	v_add_f32_e32 v60, v60, v100
	global_store_b32 v148, v60, s[8:9]
	s_wait_loadcnt 0x1
	v_add_f32_e32 v137, v137, v112
	global_store_b32 v149, v137, s[8:9]
	s_wait_loadcnt 0x0
	v_add_f32_e32 v141, v141, v124
	global_store_b32 v150, v141, s[8:9]
	s_wait_storecnt 0x0
	s_branch .Lrx_b6_stored
	.Lrx_b6_p0_next:
	s_add_co_i32 s13, s12, 2
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cbranch_scc0 .Lrx_b6_stored
	s_add_co_i32 s13, s12, 3
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cbranch_scc0 .Lrx_b6_p1_single
	global_load_b32 v50, v145, s[8:9] offset:8
	global_load_b32 v51, v145, s[8:9] offset:12
	global_load_b32 v54, v146, s[8:9] offset:8
	global_load_b32 v55, v146, s[8:9] offset:12
	global_load_b32 v58, v147, s[8:9] offset:8
	global_load_b32 v59, v147, s[8:9] offset:12
	global_load_b32 v62, v148, s[8:9] offset:8
	global_load_b32 v63, v148, s[8:9] offset:12
	global_load_b32 v139, v149, s[8:9] offset:8
	global_load_b32 v140, v149, s[8:9] offset:12
	global_load_b32 v143, v150, s[8:9] offset:8
	global_load_b32 v144, v150, s[8:9] offset:12
	s_wait_loadcnt 0xb
	v_add_f32_e32 v50, v66, v50
	global_store_b32 v145, v50, s[8:9] offset:8
	s_wait_loadcnt 0xa
	v_add_f32_e32 v51, v67, v51
	global_store_b32 v145, v51, s[8:9] offset:12
	s_wait_loadcnt 0x9
	v_add_f32_e32 v54, v78, v54
	global_store_b32 v146, v54, s[8:9] offset:8
	s_wait_loadcnt 0x8
	v_add_f32_e32 v55, v79, v55
	global_store_b32 v146, v55, s[8:9] offset:12
	s_wait_loadcnt 0x7
	v_add_f32_e32 v58, v90, v58
	global_store_b32 v147, v58, s[8:9] offset:8
	s_wait_loadcnt 0x6
	v_add_f32_e32 v59, v91, v59
	global_store_b32 v147, v59, s[8:9] offset:12
	s_wait_loadcnt 0x5
	v_add_f32_e32 v62, v102, v62
	global_store_b32 v148, v62, s[8:9] offset:8
	s_wait_loadcnt 0x4
	v_add_f32_e32 v63, v103, v63
	global_store_b32 v148, v63, s[8:9] offset:12
	s_wait_loadcnt 0x3
	v_add_f32_e32 v139, v114, v139
	global_store_b32 v149, v139, s[8:9] offset:8
	s_wait_loadcnt 0x2
	v_add_f32_e32 v140, v115, v140
	global_store_b32 v149, v140, s[8:9] offset:12
	s_wait_loadcnt 0x1
	v_add_f32_e32 v143, v126, v143
	global_store_b32 v150, v143, s[8:9] offset:8
	s_wait_loadcnt 0x0
	v_add_f32_e32 v144, v127, v144
	global_store_b32 v150, v144, s[8:9] offset:12
	s_wait_storecnt 0x0
	s_branch .Lrx_b6_stored
	.Lrx_b6_p1_single:
	global_load_b32 v50, v145, s[8:9] offset:8
	global_load_b32 v54, v146, s[8:9] offset:8
	global_load_b32 v58, v147, s[8:9] offset:8
	global_load_b32 v62, v148, s[8:9] offset:8
	global_load_b32 v139, v149, s[8:9] offset:8
	global_load_b32 v143, v150, s[8:9] offset:8
	s_wait_loadcnt 0x5
	v_add_f32_e32 v50, v50, v66
	global_store_b32 v145, v50, s[8:9] offset:8
	s_wait_loadcnt 0x4
	v_add_f32_e32 v54, v54, v78
	global_store_b32 v146, v54, s[8:9] offset:8
	s_wait_loadcnt 0x3
	v_add_f32_e32 v58, v58, v90
	global_store_b32 v147, v58, s[8:9] offset:8
	s_wait_loadcnt 0x2
	v_add_f32_e32 v62, v62, v102
	global_store_b32 v148, v62, s[8:9] offset:8
	s_wait_loadcnt 0x1
	v_add_f32_e32 v139, v139, v114
	global_store_b32 v149, v139, s[8:9] offset:8
	s_wait_loadcnt 0x0
	v_add_f32_e32 v143, v143, v126
	global_store_b32 v150, v143, s[8:9] offset:8
	s_wait_storecnt 0x0
	s_branch .Lrx_b6_stored
	.Lrx_b6_stored:
	s_branch .Lrx_end
	.Lrx_b7:
	s_mul_i32 s12, ttmp9, 4
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_i32 s12, s10
	s_cbranch_scc1 .Lrx_end
	s_mov_b32 s13, s12
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s13, s13, s19
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v8, s13
	v_add_nc_u32_e32 v4, s13, v1
	s_add_co_i32 s13, s12, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cselect_b32 s13, s13, s12
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s13, s13, s19
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v9, s13
	v_add_nc_u32_e32 v5, s13, v1
	s_add_co_i32 s13, s12, 2
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cselect_b32 s13, s13, s12
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s13, s13, s19
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v10, s13
	v_add_nc_u32_e32 v6, s13, v1
	s_add_co_i32 s13, s12, 3
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cselect_b32 s13, s13, s12
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s13, s13, s19
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v11, s13
	v_add_nc_u32_e32 v7, s13, v1
	v_mov_b32_e32 v64, 0
	v_mov_b32_e32 v65, 0
	v_mov_b32_e32 v66, 0
	v_mov_b32_e32 v67, 0
	v_mov_b32_e32 v68, 0
	v_mov_b32_e32 v69, 0
	v_mov_b32_e32 v70, 0
	v_mov_b32_e32 v71, 0
	v_mov_b32_e32 v72, 0
	v_mov_b32_e32 v73, 0
	v_mov_b32_e32 v74, 0
	v_mov_b32_e32 v75, 0
	v_mov_b32_e32 v76, 0
	v_mov_b32_e32 v77, 0
	v_mov_b32_e32 v78, 0
	v_mov_b32_e32 v79, 0
	v_mov_b32_e32 v80, 0
	v_mov_b32_e32 v81, 0
	v_mov_b32_e32 v82, 0
	v_mov_b32_e32 v83, 0
	v_mov_b32_e32 v84, 0
	v_mov_b32_e32 v85, 0
	v_mov_b32_e32 v86, 0
	v_mov_b32_e32 v87, 0
	v_mov_b32_e32 v88, 0
	v_mov_b32_e32 v89, 0
	v_mov_b32_e32 v90, 0
	v_mov_b32_e32 v91, 0
	v_mov_b32_e32 v92, 0
	v_mov_b32_e32 v93, 0
	v_mov_b32_e32 v94, 0
	v_mov_b32_e32 v95, 0
	v_mov_b32_e32 v96, 0
	v_mov_b32_e32 v97, 0
	v_mov_b32_e32 v98, 0
	v_mov_b32_e32 v99, 0
	v_mov_b32_e32 v100, 0
	v_mov_b32_e32 v101, 0
	v_mov_b32_e32 v102, 0
	v_mov_b32_e32 v103, 0
	v_mov_b32_e32 v104, 0
	v_mov_b32_e32 v105, 0
	v_mov_b32_e32 v106, 0
	v_mov_b32_e32 v107, 0
	v_mov_b32_e32 v108, 0
	v_mov_b32_e32 v109, 0
	v_mov_b32_e32 v110, 0
	v_mov_b32_e32 v111, 0
	v_mov_b32_e32 v112, 0
	v_mov_b32_e32 v113, 0
	v_mov_b32_e32 v114, 0
	v_mov_b32_e32 v115, 0
	v_mov_b32_e32 v116, 0
	v_mov_b32_e32 v117, 0
	v_mov_b32_e32 v118, 0
	v_mov_b32_e32 v119, 0
	v_mov_b32_e32 v120, 0
	v_mov_b32_e32 v121, 0
	v_mov_b32_e32 v122, 0
	v_mov_b32_e32 v123, 0
	v_mov_b32_e32 v124, 0
	v_mov_b32_e32 v125, 0
	v_mov_b32_e32 v126, 0
	v_mov_b32_e32 v127, 0
	v_mov_b32_e32 v128, 0
	v_mov_b32_e32 v129, 0
	v_mov_b32_e32 v130, 0
	v_mov_b32_e32 v131, 0
	v_mov_b32_e32 v132, 0
	v_mov_b32_e32 v133, 0
	v_mov_b32_e32 v134, 0
	v_mov_b32_e32 v135, 0
	v_mov_b32_e32 v136, 0
	v_mov_b32_e32 v137, 0
	v_mov_b32_e32 v138, 0
	v_mov_b32_e32 v139, 0
	v_mov_b32_e32 v140, 0
	v_mov_b32_e32 v141, 0
	v_mov_b32_e32 v142, 0
	v_mov_b32_e32 v143, 0
	v_mov_b32_e32 v144, 0
	v_mov_b32_e32 v145, 0
	v_mov_b32_e32 v146, 0
	v_mov_b32_e32 v147, 0
	s_add_co_i32 s2, s14, 3
	s_lshr_b32 s2, s2, 2
	s_cmp_eq_u32 s2, 0
	s_cbranch_scc1 .Lrx_b7_s0_done
	s_mov_b32 s15, 0
	v_lshlrev_b32_e32 v2, 5, v0
	buffer_load_b64 v[12:13], v8, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v20, v4, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[14:15], v9, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v21, v5, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[16:17], v10, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v22, v6, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[18:19], v11, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v23, v7, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	global_load_b128 v[149:152], v2, s[28:29]
	global_load_b128 v[153:156], v2, s[28:29] offset:16
	global_load_b128 v[157:160], v2, s[30:31]
	global_load_b128 v[161:164], v2, s[30:31] offset:16
	global_load_b128 v[165:168], v2, s[32:33]
	global_load_b128 v[169:172], v2, s[32:33] offset:16
	global_load_b128 v[173:176], v2, s[34:35]
	global_load_b128 v[177:180], v2, s[34:35] offset:16
	global_load_b128 v[181:184], v2, s[36:37]
	global_load_b128 v[185:188], v2, s[36:37] offset:16
	global_load_b128 v[189:192], v2, s[38:39]
	global_load_b128 v[193:196], v2, s[38:39] offset:16
	global_load_b128 v[197:200], v2, s[40:41]
	global_load_b128 v[201:204], v2, s[40:41] offset:16
	.Lrx_b7_s0:
	s_cmp_gt_u32 s2, 1
	s_cselect_b32 s17, 0x220, 0
	s_cselect_b32 s18, 0x1000, 0
	s_add_co_i32 s16, s15, s17
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v3, s18, v2
	v_cmp_gt_u32_e32 vcc_lo, 16, v0
	s_wait_loadcnt 0x15
	v_cndmask_b32_e32 v12, v13, v12, vcc_lo
	s_wait_loadcnt 0x14
	v_and_b32_e32 v30, 0xf0f0f0f, v20
	v_lshrrev_b32_e32 v31, 4, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v31, 0xf0f0f0f, v31
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v24, v30
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v26, v30
	v_cvt_f32_ubyte2_e32 v28, v30
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v25, v31
	v_cvt_f32_ubyte1_e32 v27, v31
	v_cvt_f32_ubyte2_e32 v29, v31
	v_cvt_f32_ubyte3_e32 v30, v30
	v_cvt_f32_ubyte3_e32 v31, v31
	v_fma_mix_f32 v24, v12, v24, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v25, v12, v25, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v26, v12, v26, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v27, v12, v27, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v28, v12, v28, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v29, v12, v29, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v30, v12, v30, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v31, v12, v31, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x13
	v_cndmask_b32_e32 v14, v15, v14, vcc_lo
	s_wait_loadcnt 0x12
	v_and_b32_e32 v40, 0xf0f0f0f, v21
	v_lshrrev_b32_e32 v41, 4, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v41, 0xf0f0f0f, v41
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v34, v40
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v36, v40
	v_cvt_f32_ubyte2_e32 v38, v40
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v35, v41
	v_cvt_f32_ubyte1_e32 v37, v41
	v_cvt_f32_ubyte2_e32 v39, v41
	v_cvt_f32_ubyte3_e32 v40, v40
	v_cvt_f32_ubyte3_e32 v41, v41
	v_fma_mix_f32 v34, v14, v34, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v35, v14, v35, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v36, v14, v36, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v37, v14, v37, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v38, v14, v38, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v39, v14, v39, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v40, v14, v40, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v41, v14, v41, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x11
	v_cndmask_b32_e32 v16, v17, v16, vcc_lo
	s_wait_loadcnt 0x10
	v_and_b32_e32 v50, 0xf0f0f0f, v22
	v_lshrrev_b32_e32 v51, 4, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v51, 0xf0f0f0f, v51
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v44, v50
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v46, v50
	v_cvt_f32_ubyte2_e32 v48, v50
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v45, v51
	v_cvt_f32_ubyte1_e32 v47, v51
	v_cvt_f32_ubyte2_e32 v49, v51
	v_cvt_f32_ubyte3_e32 v50, v50
	v_cvt_f32_ubyte3_e32 v51, v51
	v_fma_mix_f32 v44, v16, v44, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v16, v45, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v16, v46, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v16, v47, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v48, v16, v48, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v49, v16, v49, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v16, v50, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v16, v51, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xf
	v_cndmask_b32_e32 v18, v19, v18, vcc_lo
	s_wait_loadcnt 0xe
	v_and_b32_e32 v60, 0xf0f0f0f, v23
	v_lshrrev_b32_e32 v61, 4, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v61, 0xf0f0f0f, v61
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v54, v60
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v56, v60
	v_cvt_f32_ubyte2_e32 v58, v60
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v55, v61
	v_cvt_f32_ubyte1_e32 v57, v61
	v_cvt_f32_ubyte2_e32 v59, v61
	v_cvt_f32_ubyte3_e32 v60, v60
	v_cvt_f32_ubyte3_e32 v61, v61
	v_fma_mix_f32 v54, v18, v54, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v18, v55, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v18, v56, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v18, v57, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v58, v18, v58, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v59, v18, v59, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v18, v60, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v18, v61, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	buffer_load_b64 v[12:13], v8, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v20, v4, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[14:15], v9, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v21, v5, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[16:17], v10, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v22, v6, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[18:19], v11, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v23, v7, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v32, v24, v149 :: v_dual_mul_f32 v33, v35, v150
	v_dual_mul_f32 v42, v44, v149 :: v_dual_mul_f32 v43, v55, v150
	s_wait_loadcnt 0x13
	v_dual_mul_f32 v52, v24, v157 :: v_dual_mul_f32 v53, v35, v158
	v_dual_mul_f32 v62, v44, v157 :: v_dual_mul_f32 v63, v55, v158
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v25, v150 :: v_dual_fmac_f32 v33, v34, v149
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v45, v150 :: v_dual_fmac_f32 v43, v54, v149
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v25, v158 :: v_dual_fmac_f32 v53, v34, v157
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v45, v158 :: v_dual_fmac_f32 v63, v54, v157
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v26, v151 :: v_dual_fmac_f32 v33, v36, v151
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v46, v151 :: v_dual_fmac_f32 v43, v56, v151
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v26, v159 :: v_dual_fmac_f32 v53, v36, v159
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v46, v159 :: v_dual_fmac_f32 v63, v56, v159
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v27, v152 :: v_dual_fmac_f32 v33, v37, v152
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v47, v152 :: v_dual_fmac_f32 v43, v57, v152
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v27, v160 :: v_dual_fmac_f32 v53, v37, v160
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v47, v160 :: v_dual_fmac_f32 v63, v57, v160
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v28, v153 :: v_dual_fmac_f32 v33, v38, v153
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v48, v153 :: v_dual_fmac_f32 v43, v58, v153
	s_wait_loadcnt 0x12
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v28, v161 :: v_dual_fmac_f32 v53, v38, v161
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v48, v161 :: v_dual_fmac_f32 v63, v58, v161
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v29, v154 :: v_dual_fmac_f32 v33, v39, v154
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v49, v154 :: v_dual_fmac_f32 v43, v59, v154
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v29, v162 :: v_dual_fmac_f32 v53, v39, v162
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v49, v162 :: v_dual_fmac_f32 v63, v59, v162
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v30, v155 :: v_dual_fmac_f32 v33, v40, v155
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v50, v155 :: v_dual_fmac_f32 v43, v60, v155
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v30, v163 :: v_dual_fmac_f32 v53, v40, v163
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v50, v163 :: v_dual_fmac_f32 v63, v60, v163
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v31, v156 :: v_dual_fmac_f32 v33, v41, v156
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v51, v156 :: v_dual_fmac_f32 v43, v61, v156
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v31, v164 :: v_dual_fmac_f32 v53, v41, v164
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v51, v164 :: v_dual_fmac_f32 v63, v61, v164
	global_load_b128 v[149:152], v3, s[28:29]
	global_load_b128 v[153:156], v3, s[28:29] offset:16
	global_load_b128 v[157:160], v3, s[30:31]
	global_load_b128 v[161:164], v3, s[30:31] offset:16
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v64, v32, v64 :: v_dual_add_f32 v65, v65, v33
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v66, v42, v66 :: v_dual_add_f32 v67, v67, v43
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v76, v52, v76 :: v_dual_add_f32 v77, v77, v53
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v78, v62, v78 :: v_dual_add_f32 v79, v79, v63
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v32, v24, v165 :: v_dual_mul_f32 v33, v35, v166
	v_dual_mul_f32 v42, v44, v165 :: v_dual_mul_f32 v43, v55, v166
	s_wait_loadcnt 0x13
	v_dual_mul_f32 v52, v24, v173 :: v_dual_mul_f32 v53, v35, v174
	v_dual_mul_f32 v62, v44, v173 :: v_dual_mul_f32 v63, v55, v174
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v25, v166 :: v_dual_fmac_f32 v33, v34, v165
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v45, v166 :: v_dual_fmac_f32 v43, v54, v165
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v25, v174 :: v_dual_fmac_f32 v53, v34, v173
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v45, v174 :: v_dual_fmac_f32 v63, v54, v173
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v26, v167 :: v_dual_fmac_f32 v33, v36, v167
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v46, v167 :: v_dual_fmac_f32 v43, v56, v167
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v26, v175 :: v_dual_fmac_f32 v53, v36, v175
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v46, v175 :: v_dual_fmac_f32 v63, v56, v175
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v27, v168 :: v_dual_fmac_f32 v33, v37, v168
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v47, v168 :: v_dual_fmac_f32 v43, v57, v168
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v27, v176 :: v_dual_fmac_f32 v53, v37, v176
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v47, v176 :: v_dual_fmac_f32 v63, v57, v176
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v28, v169 :: v_dual_fmac_f32 v33, v38, v169
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v48, v169 :: v_dual_fmac_f32 v43, v58, v169
	s_wait_loadcnt 0x12
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v28, v177 :: v_dual_fmac_f32 v53, v38, v177
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v48, v177 :: v_dual_fmac_f32 v63, v58, v177
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v29, v170 :: v_dual_fmac_f32 v33, v39, v170
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v49, v170 :: v_dual_fmac_f32 v43, v59, v170
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v29, v178 :: v_dual_fmac_f32 v53, v39, v178
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v49, v178 :: v_dual_fmac_f32 v63, v59, v178
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v30, v171 :: v_dual_fmac_f32 v33, v40, v171
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v50, v171 :: v_dual_fmac_f32 v43, v60, v171
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v30, v179 :: v_dual_fmac_f32 v53, v40, v179
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v50, v179 :: v_dual_fmac_f32 v63, v60, v179
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v31, v172 :: v_dual_fmac_f32 v33, v41, v172
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v51, v172 :: v_dual_fmac_f32 v43, v61, v172
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v31, v180 :: v_dual_fmac_f32 v53, v41, v180
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v51, v180 :: v_dual_fmac_f32 v63, v61, v180
	global_load_b128 v[165:168], v3, s[32:33]
	global_load_b128 v[169:172], v3, s[32:33] offset:16
	global_load_b128 v[173:176], v3, s[34:35]
	global_load_b128 v[177:180], v3, s[34:35] offset:16
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v88, v32, v88 :: v_dual_add_f32 v89, v89, v33
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v90, v42, v90 :: v_dual_add_f32 v91, v91, v43
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v100, v52, v100 :: v_dual_add_f32 v101, v101, v53
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v102, v62, v102 :: v_dual_add_f32 v103, v103, v63
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v32, v24, v181 :: v_dual_mul_f32 v33, v35, v182
	v_dual_mul_f32 v42, v44, v181 :: v_dual_mul_f32 v43, v55, v182
	s_wait_loadcnt 0x13
	v_dual_mul_f32 v52, v24, v189 :: v_dual_mul_f32 v53, v35, v190
	v_dual_mul_f32 v62, v44, v189 :: v_dual_mul_f32 v63, v55, v190
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v25, v182 :: v_dual_fmac_f32 v33, v34, v181
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v45, v182 :: v_dual_fmac_f32 v43, v54, v181
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v25, v190 :: v_dual_fmac_f32 v53, v34, v189
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v45, v190 :: v_dual_fmac_f32 v63, v54, v189
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v26, v183 :: v_dual_fmac_f32 v33, v36, v183
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v46, v183 :: v_dual_fmac_f32 v43, v56, v183
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v26, v191 :: v_dual_fmac_f32 v53, v36, v191
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v46, v191 :: v_dual_fmac_f32 v63, v56, v191
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v27, v184 :: v_dual_fmac_f32 v33, v37, v184
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v47, v184 :: v_dual_fmac_f32 v43, v57, v184
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v27, v192 :: v_dual_fmac_f32 v53, v37, v192
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v47, v192 :: v_dual_fmac_f32 v63, v57, v192
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v28, v185 :: v_dual_fmac_f32 v33, v38, v185
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v48, v185 :: v_dual_fmac_f32 v43, v58, v185
	s_wait_loadcnt 0x12
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v28, v193 :: v_dual_fmac_f32 v53, v38, v193
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v48, v193 :: v_dual_fmac_f32 v63, v58, v193
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v29, v186 :: v_dual_fmac_f32 v33, v39, v186
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v49, v186 :: v_dual_fmac_f32 v43, v59, v186
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v29, v194 :: v_dual_fmac_f32 v53, v39, v194
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v49, v194 :: v_dual_fmac_f32 v63, v59, v194
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v30, v187 :: v_dual_fmac_f32 v33, v40, v187
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v50, v187 :: v_dual_fmac_f32 v43, v60, v187
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v30, v195 :: v_dual_fmac_f32 v53, v40, v195
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v50, v195 :: v_dual_fmac_f32 v63, v60, v195
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v31, v188 :: v_dual_fmac_f32 v33, v41, v188
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v51, v188 :: v_dual_fmac_f32 v43, v61, v188
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v31, v196 :: v_dual_fmac_f32 v53, v41, v196
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v51, v196 :: v_dual_fmac_f32 v63, v61, v196
	global_load_b128 v[181:184], v3, s[36:37]
	global_load_b128 v[185:188], v3, s[36:37] offset:16
	global_load_b128 v[189:192], v3, s[38:39]
	global_load_b128 v[193:196], v3, s[38:39] offset:16
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v112, v32, v112 :: v_dual_add_f32 v113, v113, v33
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v114, v42, v114 :: v_dual_add_f32 v115, v115, v43
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v124, v52, v124 :: v_dual_add_f32 v125, v125, v53
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v126, v62, v126 :: v_dual_add_f32 v127, v127, v63
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v32, v24, v197 :: v_dual_mul_f32 v33, v35, v198
	v_dual_mul_f32 v42, v44, v197 :: v_dual_mul_f32 v43, v55, v198
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v25, v198 :: v_dual_fmac_f32 v33, v34, v197
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v45, v198 :: v_dual_fmac_f32 v43, v54, v197
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v26, v199 :: v_dual_fmac_f32 v33, v36, v199
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v46, v199 :: v_dual_fmac_f32 v43, v56, v199
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v27, v200 :: v_dual_fmac_f32 v33, v37, v200
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v47, v200 :: v_dual_fmac_f32 v43, v57, v200
	s_wait_loadcnt 0x14
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v28, v201 :: v_dual_fmac_f32 v33, v38, v201
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v48, v201 :: v_dual_fmac_f32 v43, v58, v201
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v29, v202 :: v_dual_fmac_f32 v33, v39, v202
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v49, v202 :: v_dual_fmac_f32 v43, v59, v202
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v30, v203 :: v_dual_fmac_f32 v33, v40, v203
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v50, v203 :: v_dual_fmac_f32 v43, v60, v203
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v31, v204 :: v_dual_fmac_f32 v33, v41, v204
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v51, v204 :: v_dual_fmac_f32 v43, v61, v204
	global_load_b128 v[197:200], v3, s[40:41]
	global_load_b128 v[201:204], v3, s[40:41] offset:16
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_add_f32 v136, v32, v136 :: v_dual_add_f32 v137, v137, v33
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_add_f32 v138, v42, v138 :: v_dual_add_f32 v139, v139, v43
	s_mov_b32 s15, s16
	v_mov_b32_e32 v2, v3
	s_add_co_i32 s2, s2, -1
	s_cmp_lg_u32 s2, 0
	s_cbranch_scc1 .Lrx_b7_s0
	s_wait_loadcnt 0x0
	.Lrx_b7_s0_done:
	s_add_co_i32 s2, s14, 2
	s_lshr_b32 s2, s2, 2
	s_cmp_eq_u32 s2, 0
	s_cbranch_scc1 .Lrx_b7_s1_done
	s_mov_b32 s15, 0x88
	v_lshlrev_b32_e32 v2, 5, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v2, 0x400, v2
	buffer_load_b64 v[12:13], v8, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v20, v4, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[14:15], v9, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v21, v5, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[16:17], v10, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v22, v6, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[18:19], v11, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v23, v7, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	global_load_b128 v[149:152], v2, s[28:29]
	global_load_b128 v[153:156], v2, s[28:29] offset:16
	global_load_b128 v[157:160], v2, s[30:31]
	global_load_b128 v[161:164], v2, s[30:31] offset:16
	global_load_b128 v[165:168], v2, s[32:33]
	global_load_b128 v[169:172], v2, s[32:33] offset:16
	global_load_b128 v[173:176], v2, s[34:35]
	global_load_b128 v[177:180], v2, s[34:35] offset:16
	global_load_b128 v[181:184], v2, s[36:37]
	global_load_b128 v[185:188], v2, s[36:37] offset:16
	global_load_b128 v[189:192], v2, s[38:39]
	global_load_b128 v[193:196], v2, s[38:39] offset:16
	global_load_b128 v[197:200], v2, s[40:41]
	global_load_b128 v[201:204], v2, s[40:41] offset:16
	.Lrx_b7_s1:
	s_cmp_gt_u32 s2, 1
	s_cselect_b32 s17, 0x220, 0
	s_cselect_b32 s18, 0x1000, 0
	s_add_co_i32 s16, s15, s17
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v3, s18, v2
	v_cmp_gt_u32_e32 vcc_lo, 16, v0
	s_wait_loadcnt 0x15
	v_cndmask_b32_e32 v12, v13, v12, vcc_lo
	s_wait_loadcnt 0x14
	v_and_b32_e32 v30, 0xf0f0f0f, v20
	v_lshrrev_b32_e32 v31, 4, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v31, 0xf0f0f0f, v31
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v24, v30
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v26, v30
	v_cvt_f32_ubyte2_e32 v28, v30
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v25, v31
	v_cvt_f32_ubyte1_e32 v27, v31
	v_cvt_f32_ubyte2_e32 v29, v31
	v_cvt_f32_ubyte3_e32 v30, v30
	v_cvt_f32_ubyte3_e32 v31, v31
	v_fma_mix_f32 v24, v12, v24, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v25, v12, v25, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v26, v12, v26, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v27, v12, v27, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v28, v12, v28, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v29, v12, v29, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v30, v12, v30, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v31, v12, v31, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x13
	v_cndmask_b32_e32 v14, v15, v14, vcc_lo
	s_wait_loadcnt 0x12
	v_and_b32_e32 v40, 0xf0f0f0f, v21
	v_lshrrev_b32_e32 v41, 4, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v41, 0xf0f0f0f, v41
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v34, v40
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v36, v40
	v_cvt_f32_ubyte2_e32 v38, v40
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v35, v41
	v_cvt_f32_ubyte1_e32 v37, v41
	v_cvt_f32_ubyte2_e32 v39, v41
	v_cvt_f32_ubyte3_e32 v40, v40
	v_cvt_f32_ubyte3_e32 v41, v41
	v_fma_mix_f32 v34, v14, v34, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v35, v14, v35, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v36, v14, v36, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v37, v14, v37, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v38, v14, v38, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v39, v14, v39, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v40, v14, v40, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v41, v14, v41, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x11
	v_cndmask_b32_e32 v16, v17, v16, vcc_lo
	s_wait_loadcnt 0x10
	v_and_b32_e32 v50, 0xf0f0f0f, v22
	v_lshrrev_b32_e32 v51, 4, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v51, 0xf0f0f0f, v51
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v44, v50
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v46, v50
	v_cvt_f32_ubyte2_e32 v48, v50
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v45, v51
	v_cvt_f32_ubyte1_e32 v47, v51
	v_cvt_f32_ubyte2_e32 v49, v51
	v_cvt_f32_ubyte3_e32 v50, v50
	v_cvt_f32_ubyte3_e32 v51, v51
	v_fma_mix_f32 v44, v16, v44, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v16, v45, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v16, v46, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v16, v47, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v48, v16, v48, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v49, v16, v49, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v16, v50, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v16, v51, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xf
	v_cndmask_b32_e32 v18, v19, v18, vcc_lo
	s_wait_loadcnt 0xe
	v_and_b32_e32 v60, 0xf0f0f0f, v23
	v_lshrrev_b32_e32 v61, 4, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v61, 0xf0f0f0f, v61
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v54, v60
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v56, v60
	v_cvt_f32_ubyte2_e32 v58, v60
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v55, v61
	v_cvt_f32_ubyte1_e32 v57, v61
	v_cvt_f32_ubyte2_e32 v59, v61
	v_cvt_f32_ubyte3_e32 v60, v60
	v_cvt_f32_ubyte3_e32 v61, v61
	v_fma_mix_f32 v54, v18, v54, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v18, v55, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v18, v56, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v18, v57, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v58, v18, v58, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v59, v18, v59, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v18, v60, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v18, v61, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	buffer_load_b64 v[12:13], v8, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v20, v4, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[14:15], v9, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v21, v5, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[16:17], v10, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v22, v6, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[18:19], v11, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v23, v7, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v32, v24, v149 :: v_dual_mul_f32 v33, v35, v150
	v_dual_mul_f32 v42, v44, v149 :: v_dual_mul_f32 v43, v55, v150
	s_wait_loadcnt 0x13
	v_dual_mul_f32 v52, v24, v157 :: v_dual_mul_f32 v53, v35, v158
	v_dual_mul_f32 v62, v44, v157 :: v_dual_mul_f32 v63, v55, v158
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v25, v150 :: v_dual_fmac_f32 v33, v34, v149
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v45, v150 :: v_dual_fmac_f32 v43, v54, v149
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v25, v158 :: v_dual_fmac_f32 v53, v34, v157
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v45, v158 :: v_dual_fmac_f32 v63, v54, v157
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v26, v151 :: v_dual_fmac_f32 v33, v36, v151
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v46, v151 :: v_dual_fmac_f32 v43, v56, v151
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v26, v159 :: v_dual_fmac_f32 v53, v36, v159
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v46, v159 :: v_dual_fmac_f32 v63, v56, v159
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v27, v152 :: v_dual_fmac_f32 v33, v37, v152
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v47, v152 :: v_dual_fmac_f32 v43, v57, v152
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v27, v160 :: v_dual_fmac_f32 v53, v37, v160
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v47, v160 :: v_dual_fmac_f32 v63, v57, v160
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v28, v153 :: v_dual_fmac_f32 v33, v38, v153
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v48, v153 :: v_dual_fmac_f32 v43, v58, v153
	s_wait_loadcnt 0x12
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v28, v161 :: v_dual_fmac_f32 v53, v38, v161
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v48, v161 :: v_dual_fmac_f32 v63, v58, v161
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v29, v154 :: v_dual_fmac_f32 v33, v39, v154
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v49, v154 :: v_dual_fmac_f32 v43, v59, v154
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v29, v162 :: v_dual_fmac_f32 v53, v39, v162
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v49, v162 :: v_dual_fmac_f32 v63, v59, v162
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v30, v155 :: v_dual_fmac_f32 v33, v40, v155
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v50, v155 :: v_dual_fmac_f32 v43, v60, v155
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v30, v163 :: v_dual_fmac_f32 v53, v40, v163
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v50, v163 :: v_dual_fmac_f32 v63, v60, v163
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v31, v156 :: v_dual_fmac_f32 v33, v41, v156
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v51, v156 :: v_dual_fmac_f32 v43, v61, v156
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v31, v164 :: v_dual_fmac_f32 v53, v41, v164
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v51, v164 :: v_dual_fmac_f32 v63, v61, v164
	global_load_b128 v[149:152], v3, s[28:29]
	global_load_b128 v[153:156], v3, s[28:29] offset:16
	global_load_b128 v[157:160], v3, s[30:31]
	global_load_b128 v[161:164], v3, s[30:31] offset:16
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v68, v32, v68 :: v_dual_add_f32 v69, v69, v33
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v70, v42, v70 :: v_dual_add_f32 v71, v71, v43
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v80, v52, v80 :: v_dual_add_f32 v81, v81, v53
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v82, v62, v82 :: v_dual_add_f32 v83, v83, v63
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v32, v24, v165 :: v_dual_mul_f32 v33, v35, v166
	v_dual_mul_f32 v42, v44, v165 :: v_dual_mul_f32 v43, v55, v166
	s_wait_loadcnt 0x13
	v_dual_mul_f32 v52, v24, v173 :: v_dual_mul_f32 v53, v35, v174
	v_dual_mul_f32 v62, v44, v173 :: v_dual_mul_f32 v63, v55, v174
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v25, v166 :: v_dual_fmac_f32 v33, v34, v165
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v45, v166 :: v_dual_fmac_f32 v43, v54, v165
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v25, v174 :: v_dual_fmac_f32 v53, v34, v173
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v45, v174 :: v_dual_fmac_f32 v63, v54, v173
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v26, v167 :: v_dual_fmac_f32 v33, v36, v167
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v46, v167 :: v_dual_fmac_f32 v43, v56, v167
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v26, v175 :: v_dual_fmac_f32 v53, v36, v175
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v46, v175 :: v_dual_fmac_f32 v63, v56, v175
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v27, v168 :: v_dual_fmac_f32 v33, v37, v168
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v47, v168 :: v_dual_fmac_f32 v43, v57, v168
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v27, v176 :: v_dual_fmac_f32 v53, v37, v176
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v47, v176 :: v_dual_fmac_f32 v63, v57, v176
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v28, v169 :: v_dual_fmac_f32 v33, v38, v169
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v48, v169 :: v_dual_fmac_f32 v43, v58, v169
	s_wait_loadcnt 0x12
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v28, v177 :: v_dual_fmac_f32 v53, v38, v177
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v48, v177 :: v_dual_fmac_f32 v63, v58, v177
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v29, v170 :: v_dual_fmac_f32 v33, v39, v170
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v49, v170 :: v_dual_fmac_f32 v43, v59, v170
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v29, v178 :: v_dual_fmac_f32 v53, v39, v178
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v49, v178 :: v_dual_fmac_f32 v63, v59, v178
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v30, v171 :: v_dual_fmac_f32 v33, v40, v171
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v50, v171 :: v_dual_fmac_f32 v43, v60, v171
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v30, v179 :: v_dual_fmac_f32 v53, v40, v179
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v50, v179 :: v_dual_fmac_f32 v63, v60, v179
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v31, v172 :: v_dual_fmac_f32 v33, v41, v172
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v51, v172 :: v_dual_fmac_f32 v43, v61, v172
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v31, v180 :: v_dual_fmac_f32 v53, v41, v180
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v51, v180 :: v_dual_fmac_f32 v63, v61, v180
	global_load_b128 v[165:168], v3, s[32:33]
	global_load_b128 v[169:172], v3, s[32:33] offset:16
	global_load_b128 v[173:176], v3, s[34:35]
	global_load_b128 v[177:180], v3, s[34:35] offset:16
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v92, v32, v92 :: v_dual_add_f32 v93, v93, v33
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v94, v42, v94 :: v_dual_add_f32 v95, v95, v43
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v104, v52, v104 :: v_dual_add_f32 v105, v105, v53
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v106, v62, v106 :: v_dual_add_f32 v107, v107, v63
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v32, v24, v181 :: v_dual_mul_f32 v33, v35, v182
	v_dual_mul_f32 v42, v44, v181 :: v_dual_mul_f32 v43, v55, v182
	s_wait_loadcnt 0x13
	v_dual_mul_f32 v52, v24, v189 :: v_dual_mul_f32 v53, v35, v190
	v_dual_mul_f32 v62, v44, v189 :: v_dual_mul_f32 v63, v55, v190
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v25, v182 :: v_dual_fmac_f32 v33, v34, v181
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v45, v182 :: v_dual_fmac_f32 v43, v54, v181
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v25, v190 :: v_dual_fmac_f32 v53, v34, v189
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v45, v190 :: v_dual_fmac_f32 v63, v54, v189
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v26, v183 :: v_dual_fmac_f32 v33, v36, v183
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v46, v183 :: v_dual_fmac_f32 v43, v56, v183
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v26, v191 :: v_dual_fmac_f32 v53, v36, v191
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v46, v191 :: v_dual_fmac_f32 v63, v56, v191
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v27, v184 :: v_dual_fmac_f32 v33, v37, v184
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v47, v184 :: v_dual_fmac_f32 v43, v57, v184
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v27, v192 :: v_dual_fmac_f32 v53, v37, v192
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v47, v192 :: v_dual_fmac_f32 v63, v57, v192
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v28, v185 :: v_dual_fmac_f32 v33, v38, v185
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v48, v185 :: v_dual_fmac_f32 v43, v58, v185
	s_wait_loadcnt 0x12
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v28, v193 :: v_dual_fmac_f32 v53, v38, v193
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v48, v193 :: v_dual_fmac_f32 v63, v58, v193
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v29, v186 :: v_dual_fmac_f32 v33, v39, v186
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v49, v186 :: v_dual_fmac_f32 v43, v59, v186
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v29, v194 :: v_dual_fmac_f32 v53, v39, v194
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v49, v194 :: v_dual_fmac_f32 v63, v59, v194
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v30, v187 :: v_dual_fmac_f32 v33, v40, v187
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v50, v187 :: v_dual_fmac_f32 v43, v60, v187
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v30, v195 :: v_dual_fmac_f32 v53, v40, v195
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v50, v195 :: v_dual_fmac_f32 v63, v60, v195
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v31, v188 :: v_dual_fmac_f32 v33, v41, v188
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v51, v188 :: v_dual_fmac_f32 v43, v61, v188
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v31, v196 :: v_dual_fmac_f32 v53, v41, v196
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v51, v196 :: v_dual_fmac_f32 v63, v61, v196
	global_load_b128 v[181:184], v3, s[36:37]
	global_load_b128 v[185:188], v3, s[36:37] offset:16
	global_load_b128 v[189:192], v3, s[38:39]
	global_load_b128 v[193:196], v3, s[38:39] offset:16
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v116, v32, v116 :: v_dual_add_f32 v117, v117, v33
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v118, v42, v118 :: v_dual_add_f32 v119, v119, v43
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v128, v52, v128 :: v_dual_add_f32 v129, v129, v53
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v130, v62, v130 :: v_dual_add_f32 v131, v131, v63
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v32, v24, v197 :: v_dual_mul_f32 v33, v35, v198
	v_dual_mul_f32 v42, v44, v197 :: v_dual_mul_f32 v43, v55, v198
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v25, v198 :: v_dual_fmac_f32 v33, v34, v197
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v45, v198 :: v_dual_fmac_f32 v43, v54, v197
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v26, v199 :: v_dual_fmac_f32 v33, v36, v199
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v46, v199 :: v_dual_fmac_f32 v43, v56, v199
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v27, v200 :: v_dual_fmac_f32 v33, v37, v200
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v47, v200 :: v_dual_fmac_f32 v43, v57, v200
	s_wait_loadcnt 0x14
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v28, v201 :: v_dual_fmac_f32 v33, v38, v201
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v48, v201 :: v_dual_fmac_f32 v43, v58, v201
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v29, v202 :: v_dual_fmac_f32 v33, v39, v202
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v49, v202 :: v_dual_fmac_f32 v43, v59, v202
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v30, v203 :: v_dual_fmac_f32 v33, v40, v203
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v50, v203 :: v_dual_fmac_f32 v43, v60, v203
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v31, v204 :: v_dual_fmac_f32 v33, v41, v204
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v51, v204 :: v_dual_fmac_f32 v43, v61, v204
	global_load_b128 v[197:200], v3, s[40:41]
	global_load_b128 v[201:204], v3, s[40:41] offset:16
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_add_f32 v140, v32, v140 :: v_dual_add_f32 v141, v141, v33
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_add_f32 v142, v42, v142 :: v_dual_add_f32 v143, v143, v43
	s_mov_b32 s15, s16
	v_mov_b32_e32 v2, v3
	s_add_co_i32 s2, s2, -1
	s_cmp_lg_u32 s2, 0
	s_cbranch_scc1 .Lrx_b7_s1
	s_wait_loadcnt 0x0
	.Lrx_b7_s1_done:
	v_dual_add_f32 v64, v64, v68 :: v_dual_add_f32 v65, v65, v69
	v_dual_add_f32 v66, v66, v70 :: v_dual_add_f32 v67, v67, v71
	v_dual_add_f32 v76, v76, v80 :: v_dual_add_f32 v77, v77, v81
	v_dual_add_f32 v78, v78, v82 :: v_dual_add_f32 v79, v79, v83
	v_dual_add_f32 v88, v88, v92 :: v_dual_add_f32 v89, v89, v93
	v_dual_add_f32 v90, v90, v94 :: v_dual_add_f32 v91, v91, v95
	v_dual_add_f32 v100, v100, v104 :: v_dual_add_f32 v101, v101, v105
	v_dual_add_f32 v102, v102, v106 :: v_dual_add_f32 v103, v103, v107
	v_dual_add_f32 v112, v112, v116 :: v_dual_add_f32 v113, v113, v117
	v_dual_add_f32 v114, v114, v118 :: v_dual_add_f32 v115, v115, v119
	v_dual_add_f32 v124, v124, v128 :: v_dual_add_f32 v125, v125, v129
	v_dual_add_f32 v126, v126, v130 :: v_dual_add_f32 v127, v127, v131
	v_dual_add_f32 v136, v136, v140 :: v_dual_add_f32 v137, v137, v141
	v_dual_add_f32 v138, v138, v142 :: v_dual_add_f32 v139, v139, v143
	v_mov_b32_e32 v68, 0
	v_mov_b32_e32 v69, 0
	v_mov_b32_e32 v70, 0
	v_mov_b32_e32 v71, 0
	v_mov_b32_e32 v80, 0
	v_mov_b32_e32 v81, 0
	v_mov_b32_e32 v82, 0
	v_mov_b32_e32 v83, 0
	v_mov_b32_e32 v92, 0
	v_mov_b32_e32 v93, 0
	v_mov_b32_e32 v94, 0
	v_mov_b32_e32 v95, 0
	v_mov_b32_e32 v104, 0
	v_mov_b32_e32 v105, 0
	v_mov_b32_e32 v106, 0
	v_mov_b32_e32 v107, 0
	v_mov_b32_e32 v116, 0
	v_mov_b32_e32 v117, 0
	v_mov_b32_e32 v118, 0
	v_mov_b32_e32 v119, 0
	v_mov_b32_e32 v128, 0
	v_mov_b32_e32 v129, 0
	v_mov_b32_e32 v130, 0
	v_mov_b32_e32 v131, 0
	v_mov_b32_e32 v140, 0
	v_mov_b32_e32 v141, 0
	v_mov_b32_e32 v142, 0
	v_mov_b32_e32 v143, 0
	s_add_co_i32 s2, s14, 1
	s_lshr_b32 s2, s2, 2
	s_cmp_eq_u32 s2, 0
	s_cbranch_scc1 .Lrx_b7_s2_done
	s_mov_b32 s15, 0x110
	v_lshlrev_b32_e32 v2, 5, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v2, 0x800, v2
	buffer_load_b64 v[12:13], v8, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v20, v4, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[14:15], v9, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v21, v5, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[16:17], v10, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v22, v6, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[18:19], v11, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v23, v7, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	global_load_b128 v[149:152], v2, s[28:29]
	global_load_b128 v[153:156], v2, s[28:29] offset:16
	global_load_b128 v[157:160], v2, s[30:31]
	global_load_b128 v[161:164], v2, s[30:31] offset:16
	global_load_b128 v[165:168], v2, s[32:33]
	global_load_b128 v[169:172], v2, s[32:33] offset:16
	global_load_b128 v[173:176], v2, s[34:35]
	global_load_b128 v[177:180], v2, s[34:35] offset:16
	global_load_b128 v[181:184], v2, s[36:37]
	global_load_b128 v[185:188], v2, s[36:37] offset:16
	global_load_b128 v[189:192], v2, s[38:39]
	global_load_b128 v[193:196], v2, s[38:39] offset:16
	global_load_b128 v[197:200], v2, s[40:41]
	global_load_b128 v[201:204], v2, s[40:41] offset:16
	.Lrx_b7_s2:
	s_cmp_gt_u32 s2, 1
	s_cselect_b32 s17, 0x220, 0
	s_cselect_b32 s18, 0x1000, 0
	s_add_co_i32 s16, s15, s17
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v3, s18, v2
	v_cmp_gt_u32_e32 vcc_lo, 16, v0
	s_wait_loadcnt 0x15
	v_cndmask_b32_e32 v12, v13, v12, vcc_lo
	s_wait_loadcnt 0x14
	v_and_b32_e32 v30, 0xf0f0f0f, v20
	v_lshrrev_b32_e32 v31, 4, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v31, 0xf0f0f0f, v31
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v24, v30
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v26, v30
	v_cvt_f32_ubyte2_e32 v28, v30
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v25, v31
	v_cvt_f32_ubyte1_e32 v27, v31
	v_cvt_f32_ubyte2_e32 v29, v31
	v_cvt_f32_ubyte3_e32 v30, v30
	v_cvt_f32_ubyte3_e32 v31, v31
	v_fma_mix_f32 v24, v12, v24, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v25, v12, v25, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v26, v12, v26, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v27, v12, v27, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v28, v12, v28, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v29, v12, v29, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v30, v12, v30, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v31, v12, v31, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x13
	v_cndmask_b32_e32 v14, v15, v14, vcc_lo
	s_wait_loadcnt 0x12
	v_and_b32_e32 v40, 0xf0f0f0f, v21
	v_lshrrev_b32_e32 v41, 4, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v41, 0xf0f0f0f, v41
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v34, v40
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v36, v40
	v_cvt_f32_ubyte2_e32 v38, v40
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v35, v41
	v_cvt_f32_ubyte1_e32 v37, v41
	v_cvt_f32_ubyte2_e32 v39, v41
	v_cvt_f32_ubyte3_e32 v40, v40
	v_cvt_f32_ubyte3_e32 v41, v41
	v_fma_mix_f32 v34, v14, v34, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v35, v14, v35, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v36, v14, v36, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v37, v14, v37, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v38, v14, v38, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v39, v14, v39, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v40, v14, v40, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v41, v14, v41, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x11
	v_cndmask_b32_e32 v16, v17, v16, vcc_lo
	s_wait_loadcnt 0x10
	v_and_b32_e32 v50, 0xf0f0f0f, v22
	v_lshrrev_b32_e32 v51, 4, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v51, 0xf0f0f0f, v51
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v44, v50
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v46, v50
	v_cvt_f32_ubyte2_e32 v48, v50
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v45, v51
	v_cvt_f32_ubyte1_e32 v47, v51
	v_cvt_f32_ubyte2_e32 v49, v51
	v_cvt_f32_ubyte3_e32 v50, v50
	v_cvt_f32_ubyte3_e32 v51, v51
	v_fma_mix_f32 v44, v16, v44, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v16, v45, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v16, v46, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v16, v47, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v48, v16, v48, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v49, v16, v49, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v16, v50, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v16, v51, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xf
	v_cndmask_b32_e32 v18, v19, v18, vcc_lo
	s_wait_loadcnt 0xe
	v_and_b32_e32 v60, 0xf0f0f0f, v23
	v_lshrrev_b32_e32 v61, 4, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v61, 0xf0f0f0f, v61
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v54, v60
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v56, v60
	v_cvt_f32_ubyte2_e32 v58, v60
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v55, v61
	v_cvt_f32_ubyte1_e32 v57, v61
	v_cvt_f32_ubyte2_e32 v59, v61
	v_cvt_f32_ubyte3_e32 v60, v60
	v_cvt_f32_ubyte3_e32 v61, v61
	v_fma_mix_f32 v54, v18, v54, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v18, v55, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v18, v56, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v18, v57, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v58, v18, v58, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v59, v18, v59, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v18, v60, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v18, v61, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	buffer_load_b64 v[12:13], v8, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v20, v4, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[14:15], v9, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v21, v5, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[16:17], v10, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v22, v6, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[18:19], v11, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v23, v7, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v32, v24, v149 :: v_dual_mul_f32 v33, v35, v150
	v_dual_mul_f32 v42, v44, v149 :: v_dual_mul_f32 v43, v55, v150
	s_wait_loadcnt 0x13
	v_dual_mul_f32 v52, v24, v157 :: v_dual_mul_f32 v53, v35, v158
	v_dual_mul_f32 v62, v44, v157 :: v_dual_mul_f32 v63, v55, v158
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v25, v150 :: v_dual_fmac_f32 v33, v34, v149
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v45, v150 :: v_dual_fmac_f32 v43, v54, v149
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v25, v158 :: v_dual_fmac_f32 v53, v34, v157
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v45, v158 :: v_dual_fmac_f32 v63, v54, v157
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v26, v151 :: v_dual_fmac_f32 v33, v36, v151
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v46, v151 :: v_dual_fmac_f32 v43, v56, v151
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v26, v159 :: v_dual_fmac_f32 v53, v36, v159
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v46, v159 :: v_dual_fmac_f32 v63, v56, v159
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v27, v152 :: v_dual_fmac_f32 v33, v37, v152
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v47, v152 :: v_dual_fmac_f32 v43, v57, v152
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v27, v160 :: v_dual_fmac_f32 v53, v37, v160
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v47, v160 :: v_dual_fmac_f32 v63, v57, v160
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v28, v153 :: v_dual_fmac_f32 v33, v38, v153
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v48, v153 :: v_dual_fmac_f32 v43, v58, v153
	s_wait_loadcnt 0x12
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v28, v161 :: v_dual_fmac_f32 v53, v38, v161
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v48, v161 :: v_dual_fmac_f32 v63, v58, v161
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v29, v154 :: v_dual_fmac_f32 v33, v39, v154
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v49, v154 :: v_dual_fmac_f32 v43, v59, v154
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v29, v162 :: v_dual_fmac_f32 v53, v39, v162
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v49, v162 :: v_dual_fmac_f32 v63, v59, v162
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v30, v155 :: v_dual_fmac_f32 v33, v40, v155
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v50, v155 :: v_dual_fmac_f32 v43, v60, v155
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v30, v163 :: v_dual_fmac_f32 v53, v40, v163
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v50, v163 :: v_dual_fmac_f32 v63, v60, v163
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v31, v156 :: v_dual_fmac_f32 v33, v41, v156
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v51, v156 :: v_dual_fmac_f32 v43, v61, v156
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v31, v164 :: v_dual_fmac_f32 v53, v41, v164
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v51, v164 :: v_dual_fmac_f32 v63, v61, v164
	global_load_b128 v[149:152], v3, s[28:29]
	global_load_b128 v[153:156], v3, s[28:29] offset:16
	global_load_b128 v[157:160], v3, s[30:31]
	global_load_b128 v[161:164], v3, s[30:31] offset:16
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v68, v32, v68 :: v_dual_add_f32 v69, v69, v33
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v70, v42, v70 :: v_dual_add_f32 v71, v71, v43
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v80, v52, v80 :: v_dual_add_f32 v81, v81, v53
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v82, v62, v82 :: v_dual_add_f32 v83, v83, v63
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v32, v24, v165 :: v_dual_mul_f32 v33, v35, v166
	v_dual_mul_f32 v42, v44, v165 :: v_dual_mul_f32 v43, v55, v166
	s_wait_loadcnt 0x13
	v_dual_mul_f32 v52, v24, v173 :: v_dual_mul_f32 v53, v35, v174
	v_dual_mul_f32 v62, v44, v173 :: v_dual_mul_f32 v63, v55, v174
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v25, v166 :: v_dual_fmac_f32 v33, v34, v165
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v45, v166 :: v_dual_fmac_f32 v43, v54, v165
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v25, v174 :: v_dual_fmac_f32 v53, v34, v173
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v45, v174 :: v_dual_fmac_f32 v63, v54, v173
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v26, v167 :: v_dual_fmac_f32 v33, v36, v167
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v46, v167 :: v_dual_fmac_f32 v43, v56, v167
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v26, v175 :: v_dual_fmac_f32 v53, v36, v175
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v46, v175 :: v_dual_fmac_f32 v63, v56, v175
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v27, v168 :: v_dual_fmac_f32 v33, v37, v168
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v47, v168 :: v_dual_fmac_f32 v43, v57, v168
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v27, v176 :: v_dual_fmac_f32 v53, v37, v176
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v47, v176 :: v_dual_fmac_f32 v63, v57, v176
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v28, v169 :: v_dual_fmac_f32 v33, v38, v169
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v48, v169 :: v_dual_fmac_f32 v43, v58, v169
	s_wait_loadcnt 0x12
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v28, v177 :: v_dual_fmac_f32 v53, v38, v177
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v48, v177 :: v_dual_fmac_f32 v63, v58, v177
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v29, v170 :: v_dual_fmac_f32 v33, v39, v170
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v49, v170 :: v_dual_fmac_f32 v43, v59, v170
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v29, v178 :: v_dual_fmac_f32 v53, v39, v178
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v49, v178 :: v_dual_fmac_f32 v63, v59, v178
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v30, v171 :: v_dual_fmac_f32 v33, v40, v171
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v50, v171 :: v_dual_fmac_f32 v43, v60, v171
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v30, v179 :: v_dual_fmac_f32 v53, v40, v179
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v50, v179 :: v_dual_fmac_f32 v63, v60, v179
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v31, v172 :: v_dual_fmac_f32 v33, v41, v172
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v51, v172 :: v_dual_fmac_f32 v43, v61, v172
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v31, v180 :: v_dual_fmac_f32 v53, v41, v180
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v51, v180 :: v_dual_fmac_f32 v63, v61, v180
	global_load_b128 v[165:168], v3, s[32:33]
	global_load_b128 v[169:172], v3, s[32:33] offset:16
	global_load_b128 v[173:176], v3, s[34:35]
	global_load_b128 v[177:180], v3, s[34:35] offset:16
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v92, v32, v92 :: v_dual_add_f32 v93, v93, v33
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v94, v42, v94 :: v_dual_add_f32 v95, v95, v43
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v104, v52, v104 :: v_dual_add_f32 v105, v105, v53
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v106, v62, v106 :: v_dual_add_f32 v107, v107, v63
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v32, v24, v181 :: v_dual_mul_f32 v33, v35, v182
	v_dual_mul_f32 v42, v44, v181 :: v_dual_mul_f32 v43, v55, v182
	s_wait_loadcnt 0x13
	v_dual_mul_f32 v52, v24, v189 :: v_dual_mul_f32 v53, v35, v190
	v_dual_mul_f32 v62, v44, v189 :: v_dual_mul_f32 v63, v55, v190
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v25, v182 :: v_dual_fmac_f32 v33, v34, v181
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v45, v182 :: v_dual_fmac_f32 v43, v54, v181
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v25, v190 :: v_dual_fmac_f32 v53, v34, v189
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v45, v190 :: v_dual_fmac_f32 v63, v54, v189
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v26, v183 :: v_dual_fmac_f32 v33, v36, v183
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v46, v183 :: v_dual_fmac_f32 v43, v56, v183
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v26, v191 :: v_dual_fmac_f32 v53, v36, v191
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v46, v191 :: v_dual_fmac_f32 v63, v56, v191
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v27, v184 :: v_dual_fmac_f32 v33, v37, v184
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v47, v184 :: v_dual_fmac_f32 v43, v57, v184
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v27, v192 :: v_dual_fmac_f32 v53, v37, v192
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v47, v192 :: v_dual_fmac_f32 v63, v57, v192
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v28, v185 :: v_dual_fmac_f32 v33, v38, v185
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v48, v185 :: v_dual_fmac_f32 v43, v58, v185
	s_wait_loadcnt 0x12
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v28, v193 :: v_dual_fmac_f32 v53, v38, v193
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v48, v193 :: v_dual_fmac_f32 v63, v58, v193
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v29, v186 :: v_dual_fmac_f32 v33, v39, v186
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v49, v186 :: v_dual_fmac_f32 v43, v59, v186
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v29, v194 :: v_dual_fmac_f32 v53, v39, v194
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v49, v194 :: v_dual_fmac_f32 v63, v59, v194
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v30, v187 :: v_dual_fmac_f32 v33, v40, v187
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v50, v187 :: v_dual_fmac_f32 v43, v60, v187
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v30, v195 :: v_dual_fmac_f32 v53, v40, v195
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v50, v195 :: v_dual_fmac_f32 v63, v60, v195
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v31, v188 :: v_dual_fmac_f32 v33, v41, v188
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v51, v188 :: v_dual_fmac_f32 v43, v61, v188
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v31, v196 :: v_dual_fmac_f32 v53, v41, v196
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v51, v196 :: v_dual_fmac_f32 v63, v61, v196
	global_load_b128 v[181:184], v3, s[36:37]
	global_load_b128 v[185:188], v3, s[36:37] offset:16
	global_load_b128 v[189:192], v3, s[38:39]
	global_load_b128 v[193:196], v3, s[38:39] offset:16
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v116, v32, v116 :: v_dual_add_f32 v117, v117, v33
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v118, v42, v118 :: v_dual_add_f32 v119, v119, v43
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v128, v52, v128 :: v_dual_add_f32 v129, v129, v53
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v130, v62, v130 :: v_dual_add_f32 v131, v131, v63
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v32, v24, v197 :: v_dual_mul_f32 v33, v35, v198
	v_dual_mul_f32 v42, v44, v197 :: v_dual_mul_f32 v43, v55, v198
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v25, v198 :: v_dual_fmac_f32 v33, v34, v197
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v45, v198 :: v_dual_fmac_f32 v43, v54, v197
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v26, v199 :: v_dual_fmac_f32 v33, v36, v199
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v46, v199 :: v_dual_fmac_f32 v43, v56, v199
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v27, v200 :: v_dual_fmac_f32 v33, v37, v200
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v47, v200 :: v_dual_fmac_f32 v43, v57, v200
	s_wait_loadcnt 0x14
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v28, v201 :: v_dual_fmac_f32 v33, v38, v201
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v48, v201 :: v_dual_fmac_f32 v43, v58, v201
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v29, v202 :: v_dual_fmac_f32 v33, v39, v202
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v49, v202 :: v_dual_fmac_f32 v43, v59, v202
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v30, v203 :: v_dual_fmac_f32 v33, v40, v203
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v50, v203 :: v_dual_fmac_f32 v43, v60, v203
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v31, v204 :: v_dual_fmac_f32 v33, v41, v204
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v51, v204 :: v_dual_fmac_f32 v43, v61, v204
	global_load_b128 v[197:200], v3, s[40:41]
	global_load_b128 v[201:204], v3, s[40:41] offset:16
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_add_f32 v140, v32, v140 :: v_dual_add_f32 v141, v141, v33
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_add_f32 v142, v42, v142 :: v_dual_add_f32 v143, v143, v43
	s_mov_b32 s15, s16
	v_mov_b32_e32 v2, v3
	s_add_co_i32 s2, s2, -1
	s_cmp_lg_u32 s2, 0
	s_cbranch_scc1 .Lrx_b7_s2
	s_wait_loadcnt 0x0
	.Lrx_b7_s2_done:
	s_add_co_i32 s2, s14, 0
	s_lshr_b32 s2, s2, 2
	s_cmp_eq_u32 s2, 0
	s_cbranch_scc1 .Lrx_b7_s3_done
	s_mov_b32 s15, 0x198
	v_lshlrev_b32_e32 v2, 5, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v2, 0xc00, v2
	buffer_load_b64 v[12:13], v8, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v20, v4, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[14:15], v9, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v21, v5, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[16:17], v10, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v22, v6, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[18:19], v11, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v23, v7, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	global_load_b128 v[149:152], v2, s[28:29]
	global_load_b128 v[153:156], v2, s[28:29] offset:16
	global_load_b128 v[157:160], v2, s[30:31]
	global_load_b128 v[161:164], v2, s[30:31] offset:16
	global_load_b128 v[165:168], v2, s[32:33]
	global_load_b128 v[169:172], v2, s[32:33] offset:16
	global_load_b128 v[173:176], v2, s[34:35]
	global_load_b128 v[177:180], v2, s[34:35] offset:16
	global_load_b128 v[181:184], v2, s[36:37]
	global_load_b128 v[185:188], v2, s[36:37] offset:16
	global_load_b128 v[189:192], v2, s[38:39]
	global_load_b128 v[193:196], v2, s[38:39] offset:16
	global_load_b128 v[197:200], v2, s[40:41]
	global_load_b128 v[201:204], v2, s[40:41] offset:16
	.Lrx_b7_s3:
	s_cmp_gt_u32 s2, 1
	s_cselect_b32 s17, 0x220, 0
	s_cselect_b32 s18, 0x1000, 0
	s_add_co_i32 s16, s15, s17
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v3, s18, v2
	v_cmp_gt_u32_e32 vcc_lo, 16, v0
	s_wait_loadcnt 0x15
	v_cndmask_b32_e32 v12, v13, v12, vcc_lo
	s_wait_loadcnt 0x14
	v_and_b32_e32 v30, 0xf0f0f0f, v20
	v_lshrrev_b32_e32 v31, 4, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v31, 0xf0f0f0f, v31
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v24, v30
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v26, v30
	v_cvt_f32_ubyte2_e32 v28, v30
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v25, v31
	v_cvt_f32_ubyte1_e32 v27, v31
	v_cvt_f32_ubyte2_e32 v29, v31
	v_cvt_f32_ubyte3_e32 v30, v30
	v_cvt_f32_ubyte3_e32 v31, v31
	v_fma_mix_f32 v24, v12, v24, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v25, v12, v25, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v26, v12, v26, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v27, v12, v27, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v28, v12, v28, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v29, v12, v29, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v30, v12, v30, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v31, v12, v31, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x13
	v_cndmask_b32_e32 v14, v15, v14, vcc_lo
	s_wait_loadcnt 0x12
	v_and_b32_e32 v40, 0xf0f0f0f, v21
	v_lshrrev_b32_e32 v41, 4, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v41, 0xf0f0f0f, v41
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v34, v40
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v36, v40
	v_cvt_f32_ubyte2_e32 v38, v40
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v35, v41
	v_cvt_f32_ubyte1_e32 v37, v41
	v_cvt_f32_ubyte2_e32 v39, v41
	v_cvt_f32_ubyte3_e32 v40, v40
	v_cvt_f32_ubyte3_e32 v41, v41
	v_fma_mix_f32 v34, v14, v34, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v35, v14, v35, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v36, v14, v36, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v37, v14, v37, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v38, v14, v38, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v39, v14, v39, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v40, v14, v40, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v41, v14, v41, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x11
	v_cndmask_b32_e32 v16, v17, v16, vcc_lo
	s_wait_loadcnt 0x10
	v_and_b32_e32 v50, 0xf0f0f0f, v22
	v_lshrrev_b32_e32 v51, 4, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v51, 0xf0f0f0f, v51
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v44, v50
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v46, v50
	v_cvt_f32_ubyte2_e32 v48, v50
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v45, v51
	v_cvt_f32_ubyte1_e32 v47, v51
	v_cvt_f32_ubyte2_e32 v49, v51
	v_cvt_f32_ubyte3_e32 v50, v50
	v_cvt_f32_ubyte3_e32 v51, v51
	v_fma_mix_f32 v44, v16, v44, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v16, v45, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v16, v46, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v16, v47, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v48, v16, v48, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v49, v16, v49, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v16, v50, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v16, v51, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xf
	v_cndmask_b32_e32 v18, v19, v18, vcc_lo
	s_wait_loadcnt 0xe
	v_and_b32_e32 v60, 0xf0f0f0f, v23
	v_lshrrev_b32_e32 v61, 4, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v61, 0xf0f0f0f, v61
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v54, v60
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v56, v60
	v_cvt_f32_ubyte2_e32 v58, v60
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v55, v61
	v_cvt_f32_ubyte1_e32 v57, v61
	v_cvt_f32_ubyte2_e32 v59, v61
	v_cvt_f32_ubyte3_e32 v60, v60
	v_cvt_f32_ubyte3_e32 v61, v61
	v_fma_mix_f32 v54, v18, v54, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v18, v55, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v18, v56, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v18, v57, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v58, v18, v58, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v59, v18, v59, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v18, v60, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v18, v61, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	buffer_load_b64 v[12:13], v8, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v20, v4, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[14:15], v9, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v21, v5, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[16:17], v10, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v22, v6, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[18:19], v11, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v23, v7, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v32, v24, v149 :: v_dual_mul_f32 v33, v35, v150
	v_dual_mul_f32 v42, v44, v149 :: v_dual_mul_f32 v43, v55, v150
	s_wait_loadcnt 0x13
	v_dual_mul_f32 v52, v24, v157 :: v_dual_mul_f32 v53, v35, v158
	v_dual_mul_f32 v62, v44, v157 :: v_dual_mul_f32 v63, v55, v158
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v25, v150 :: v_dual_fmac_f32 v33, v34, v149
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v45, v150 :: v_dual_fmac_f32 v43, v54, v149
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v25, v158 :: v_dual_fmac_f32 v53, v34, v157
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v45, v158 :: v_dual_fmac_f32 v63, v54, v157
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v26, v151 :: v_dual_fmac_f32 v33, v36, v151
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v46, v151 :: v_dual_fmac_f32 v43, v56, v151
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v26, v159 :: v_dual_fmac_f32 v53, v36, v159
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v46, v159 :: v_dual_fmac_f32 v63, v56, v159
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v27, v152 :: v_dual_fmac_f32 v33, v37, v152
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v47, v152 :: v_dual_fmac_f32 v43, v57, v152
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v27, v160 :: v_dual_fmac_f32 v53, v37, v160
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v47, v160 :: v_dual_fmac_f32 v63, v57, v160
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v28, v153 :: v_dual_fmac_f32 v33, v38, v153
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v48, v153 :: v_dual_fmac_f32 v43, v58, v153
	s_wait_loadcnt 0x12
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v28, v161 :: v_dual_fmac_f32 v53, v38, v161
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v48, v161 :: v_dual_fmac_f32 v63, v58, v161
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v29, v154 :: v_dual_fmac_f32 v33, v39, v154
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v49, v154 :: v_dual_fmac_f32 v43, v59, v154
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v29, v162 :: v_dual_fmac_f32 v53, v39, v162
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v49, v162 :: v_dual_fmac_f32 v63, v59, v162
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v30, v155 :: v_dual_fmac_f32 v33, v40, v155
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v50, v155 :: v_dual_fmac_f32 v43, v60, v155
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v30, v163 :: v_dual_fmac_f32 v53, v40, v163
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v50, v163 :: v_dual_fmac_f32 v63, v60, v163
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v31, v156 :: v_dual_fmac_f32 v33, v41, v156
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v51, v156 :: v_dual_fmac_f32 v43, v61, v156
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v31, v164 :: v_dual_fmac_f32 v53, v41, v164
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v51, v164 :: v_dual_fmac_f32 v63, v61, v164
	global_load_b128 v[149:152], v3, s[28:29]
	global_load_b128 v[153:156], v3, s[28:29] offset:16
	global_load_b128 v[157:160], v3, s[30:31]
	global_load_b128 v[161:164], v3, s[30:31] offset:16
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v72, v32, v72 :: v_dual_add_f32 v73, v73, v33
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v74, v42, v74 :: v_dual_add_f32 v75, v75, v43
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v84, v52, v84 :: v_dual_add_f32 v85, v85, v53
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v86, v62, v86 :: v_dual_add_f32 v87, v87, v63
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v32, v24, v165 :: v_dual_mul_f32 v33, v35, v166
	v_dual_mul_f32 v42, v44, v165 :: v_dual_mul_f32 v43, v55, v166
	s_wait_loadcnt 0x13
	v_dual_mul_f32 v52, v24, v173 :: v_dual_mul_f32 v53, v35, v174
	v_dual_mul_f32 v62, v44, v173 :: v_dual_mul_f32 v63, v55, v174
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v25, v166 :: v_dual_fmac_f32 v33, v34, v165
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v45, v166 :: v_dual_fmac_f32 v43, v54, v165
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v25, v174 :: v_dual_fmac_f32 v53, v34, v173
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v45, v174 :: v_dual_fmac_f32 v63, v54, v173
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v26, v167 :: v_dual_fmac_f32 v33, v36, v167
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v46, v167 :: v_dual_fmac_f32 v43, v56, v167
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v26, v175 :: v_dual_fmac_f32 v53, v36, v175
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v46, v175 :: v_dual_fmac_f32 v63, v56, v175
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v27, v168 :: v_dual_fmac_f32 v33, v37, v168
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v47, v168 :: v_dual_fmac_f32 v43, v57, v168
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v27, v176 :: v_dual_fmac_f32 v53, v37, v176
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v47, v176 :: v_dual_fmac_f32 v63, v57, v176
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v28, v169 :: v_dual_fmac_f32 v33, v38, v169
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v48, v169 :: v_dual_fmac_f32 v43, v58, v169
	s_wait_loadcnt 0x12
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v28, v177 :: v_dual_fmac_f32 v53, v38, v177
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v48, v177 :: v_dual_fmac_f32 v63, v58, v177
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v29, v170 :: v_dual_fmac_f32 v33, v39, v170
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v49, v170 :: v_dual_fmac_f32 v43, v59, v170
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v29, v178 :: v_dual_fmac_f32 v53, v39, v178
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v49, v178 :: v_dual_fmac_f32 v63, v59, v178
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v30, v171 :: v_dual_fmac_f32 v33, v40, v171
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v50, v171 :: v_dual_fmac_f32 v43, v60, v171
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v30, v179 :: v_dual_fmac_f32 v53, v40, v179
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v50, v179 :: v_dual_fmac_f32 v63, v60, v179
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v31, v172 :: v_dual_fmac_f32 v33, v41, v172
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v51, v172 :: v_dual_fmac_f32 v43, v61, v172
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v31, v180 :: v_dual_fmac_f32 v53, v41, v180
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v51, v180 :: v_dual_fmac_f32 v63, v61, v180
	global_load_b128 v[165:168], v3, s[32:33]
	global_load_b128 v[169:172], v3, s[32:33] offset:16
	global_load_b128 v[173:176], v3, s[34:35]
	global_load_b128 v[177:180], v3, s[34:35] offset:16
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v96, v32, v96 :: v_dual_add_f32 v97, v97, v33
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v98, v42, v98 :: v_dual_add_f32 v99, v99, v43
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v108, v52, v108 :: v_dual_add_f32 v109, v109, v53
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v110, v62, v110 :: v_dual_add_f32 v111, v111, v63
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v32, v24, v181 :: v_dual_mul_f32 v33, v35, v182
	v_dual_mul_f32 v42, v44, v181 :: v_dual_mul_f32 v43, v55, v182
	s_wait_loadcnt 0x13
	v_dual_mul_f32 v52, v24, v189 :: v_dual_mul_f32 v53, v35, v190
	v_dual_mul_f32 v62, v44, v189 :: v_dual_mul_f32 v63, v55, v190
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v25, v182 :: v_dual_fmac_f32 v33, v34, v181
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v45, v182 :: v_dual_fmac_f32 v43, v54, v181
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v25, v190 :: v_dual_fmac_f32 v53, v34, v189
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v45, v190 :: v_dual_fmac_f32 v63, v54, v189
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v26, v183 :: v_dual_fmac_f32 v33, v36, v183
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v46, v183 :: v_dual_fmac_f32 v43, v56, v183
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v26, v191 :: v_dual_fmac_f32 v53, v36, v191
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v46, v191 :: v_dual_fmac_f32 v63, v56, v191
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v27, v184 :: v_dual_fmac_f32 v33, v37, v184
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v47, v184 :: v_dual_fmac_f32 v43, v57, v184
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v27, v192 :: v_dual_fmac_f32 v53, v37, v192
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v47, v192 :: v_dual_fmac_f32 v63, v57, v192
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v28, v185 :: v_dual_fmac_f32 v33, v38, v185
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v48, v185 :: v_dual_fmac_f32 v43, v58, v185
	s_wait_loadcnt 0x12
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v28, v193 :: v_dual_fmac_f32 v53, v38, v193
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v48, v193 :: v_dual_fmac_f32 v63, v58, v193
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v29, v186 :: v_dual_fmac_f32 v33, v39, v186
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v49, v186 :: v_dual_fmac_f32 v43, v59, v186
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v29, v194 :: v_dual_fmac_f32 v53, v39, v194
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v49, v194 :: v_dual_fmac_f32 v63, v59, v194
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v30, v187 :: v_dual_fmac_f32 v33, v40, v187
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v50, v187 :: v_dual_fmac_f32 v43, v60, v187
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v30, v195 :: v_dual_fmac_f32 v53, v40, v195
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v50, v195 :: v_dual_fmac_f32 v63, v60, v195
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v31, v188 :: v_dual_fmac_f32 v33, v41, v188
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v51, v188 :: v_dual_fmac_f32 v43, v61, v188
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v31, v196 :: v_dual_fmac_f32 v53, v41, v196
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v51, v196 :: v_dual_fmac_f32 v63, v61, v196
	global_load_b128 v[181:184], v3, s[36:37]
	global_load_b128 v[185:188], v3, s[36:37] offset:16
	global_load_b128 v[189:192], v3, s[38:39]
	global_load_b128 v[193:196], v3, s[38:39] offset:16
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v120, v32, v120 :: v_dual_add_f32 v121, v121, v33
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v122, v42, v122 :: v_dual_add_f32 v123, v123, v43
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v132, v52, v132 :: v_dual_add_f32 v133, v133, v53
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v134, v62, v134 :: v_dual_add_f32 v135, v135, v63
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v32, v24, v197 :: v_dual_mul_f32 v33, v35, v198
	v_dual_mul_f32 v42, v44, v197 :: v_dual_mul_f32 v43, v55, v198
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v25, v198 :: v_dual_fmac_f32 v33, v34, v197
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v45, v198 :: v_dual_fmac_f32 v43, v54, v197
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v26, v199 :: v_dual_fmac_f32 v33, v36, v199
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v46, v199 :: v_dual_fmac_f32 v43, v56, v199
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v27, v200 :: v_dual_fmac_f32 v33, v37, v200
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v47, v200 :: v_dual_fmac_f32 v43, v57, v200
	s_wait_loadcnt 0x14
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v28, v201 :: v_dual_fmac_f32 v33, v38, v201
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v48, v201 :: v_dual_fmac_f32 v43, v58, v201
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v29, v202 :: v_dual_fmac_f32 v33, v39, v202
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v49, v202 :: v_dual_fmac_f32 v43, v59, v202
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v30, v203 :: v_dual_fmac_f32 v33, v40, v203
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v50, v203 :: v_dual_fmac_f32 v43, v60, v203
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v32, v31, v204 :: v_dual_fmac_f32 v33, v41, v204
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_fmac_f32 v42, v51, v204 :: v_dual_fmac_f32 v43, v61, v204
	global_load_b128 v[197:200], v3, s[40:41]
	global_load_b128 v[201:204], v3, s[40:41] offset:16
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_add_f32 v144, v32, v144 :: v_dual_add_f32 v145, v145, v33
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_add_f32 v146, v42, v146 :: v_dual_add_f32 v147, v147, v43
	s_mov_b32 s15, s16
	v_mov_b32_e32 v2, v3
	s_add_co_i32 s2, s2, -1
	s_cmp_lg_u32 s2, 0
	s_cbranch_scc1 .Lrx_b7_s3
	s_wait_loadcnt 0x0
	.Lrx_b7_s3_done:
	v_dual_add_f32 v68, v68, v72 :: v_dual_add_f32 v69, v69, v73
	v_dual_add_f32 v70, v70, v74 :: v_dual_add_f32 v71, v71, v75
	v_dual_add_f32 v80, v80, v84 :: v_dual_add_f32 v81, v81, v85
	v_dual_add_f32 v82, v82, v86 :: v_dual_add_f32 v83, v83, v87
	v_dual_add_f32 v92, v92, v96 :: v_dual_add_f32 v93, v93, v97
	v_dual_add_f32 v94, v94, v98 :: v_dual_add_f32 v95, v95, v99
	v_dual_add_f32 v104, v104, v108 :: v_dual_add_f32 v105, v105, v109
	v_dual_add_f32 v106, v106, v110 :: v_dual_add_f32 v107, v107, v111
	v_dual_add_f32 v116, v116, v120 :: v_dual_add_f32 v117, v117, v121
	v_dual_add_f32 v118, v118, v122 :: v_dual_add_f32 v119, v119, v123
	v_dual_add_f32 v128, v128, v132 :: v_dual_add_f32 v129, v129, v133
	v_dual_add_f32 v130, v130, v134 :: v_dual_add_f32 v131, v131, v135
	v_dual_add_f32 v140, v140, v144 :: v_dual_add_f32 v141, v141, v145
	v_dual_add_f32 v142, v142, v146 :: v_dual_add_f32 v143, v143, v147
	v_dual_add_f32 v64, v64, v68 :: v_dual_add_f32 v65, v65, v69
	v_dual_add_f32 v66, v66, v70 :: v_dual_add_f32 v67, v67, v71
	v_dual_add_f32 v76, v76, v80 :: v_dual_add_f32 v77, v77, v81
	v_dual_add_f32 v78, v78, v82 :: v_dual_add_f32 v79, v79, v83
	v_dual_add_f32 v88, v88, v92 :: v_dual_add_f32 v89, v89, v93
	v_dual_add_f32 v90, v90, v94 :: v_dual_add_f32 v91, v91, v95
	v_dual_add_f32 v100, v100, v104 :: v_dual_add_f32 v101, v101, v105
	v_dual_add_f32 v102, v102, v106 :: v_dual_add_f32 v103, v103, v107
	v_dual_add_f32 v112, v112, v116 :: v_dual_add_f32 v113, v113, v117
	v_dual_add_f32 v114, v114, v118 :: v_dual_add_f32 v115, v115, v119
	v_dual_add_f32 v124, v124, v128 :: v_dual_add_f32 v125, v125, v129
	v_dual_add_f32 v126, v126, v130 :: v_dual_add_f32 v127, v127, v131
	v_dual_add_f32 v136, v136, v140 :: v_dual_add_f32 v137, v137, v141
	v_dual_add_f32 v138, v138, v142 :: v_dual_add_f32 v139, v139, v143
	ds_swizzle_b32 v24, v64 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v25, v65 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v26, v66 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v27, v67 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v28, v76 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v29, v77 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v30, v78 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v31, v79 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v32, v88 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v33, v89 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v34, v90 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v35, v91 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v36, v100 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v37, v101 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v38, v102 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v39, v103 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v40, v112 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v41, v113 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v42, v114 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v43, v115 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v44, v124 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v45, v125 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v46, v126 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v47, v127 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v48, v136 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v49, v137 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v50, v138 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v51, v139 offset:swizzle(BITMASK_PERM,"1pppp")
	s_wait_dscnt 0x1b
	v_add_f32_e32 v64, v64, v24
	s_wait_dscnt 0x1a
	v_add_f32_e32 v65, v65, v25
	s_wait_dscnt 0x19
	v_add_f32_e32 v66, v66, v26
	s_wait_dscnt 0x18
	v_add_f32_e32 v67, v67, v27
	s_wait_dscnt 0x17
	v_add_f32_e32 v76, v76, v28
	s_wait_dscnt 0x16
	v_add_f32_e32 v77, v77, v29
	s_wait_dscnt 0x15
	v_add_f32_e32 v78, v78, v30
	s_wait_dscnt 0x14
	v_add_f32_e32 v79, v79, v31
	s_wait_dscnt 0x13
	v_add_f32_e32 v88, v88, v32
	s_wait_dscnt 0x12
	v_add_f32_e32 v89, v89, v33
	s_wait_dscnt 0x11
	v_add_f32_e32 v90, v90, v34
	s_wait_dscnt 0x10
	v_add_f32_e32 v91, v91, v35
	s_wait_dscnt 0xf
	v_add_f32_e32 v100, v100, v36
	s_wait_dscnt 0xe
	v_add_f32_e32 v101, v101, v37
	s_wait_dscnt 0xd
	v_add_f32_e32 v102, v102, v38
	s_wait_dscnt 0xc
	v_add_f32_e32 v103, v103, v39
	s_wait_dscnt 0xb
	v_add_f32_e32 v112, v112, v40
	s_wait_dscnt 0xa
	v_add_f32_e32 v113, v113, v41
	s_wait_dscnt 0x9
	v_add_f32_e32 v114, v114, v42
	s_wait_dscnt 0x8
	v_add_f32_e32 v115, v115, v43
	s_wait_dscnt 0x7
	v_add_f32_e32 v124, v124, v44
	s_wait_dscnt 0x6
	v_add_f32_e32 v125, v125, v45
	s_wait_dscnt 0x5
	v_add_f32_e32 v126, v126, v46
	s_wait_dscnt 0x4
	v_add_f32_e32 v127, v127, v47
	s_wait_dscnt 0x3
	v_add_f32_e32 v136, v136, v48
	s_wait_dscnt 0x2
	v_add_f32_e32 v137, v137, v49
	s_wait_dscnt 0x1
	v_add_f32_e32 v138, v138, v50
	s_wait_dscnt 0x0
	v_add_f32_e32 v139, v139, v51
	v_cmp_gt_u32_e32 vcc_lo, 24, v0
	v_cndmask_b32_e64 v3, 0, 8, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v3, v3, v0, 2
	ds_bpermute_b32 v24, v3, v64
	ds_bpermute_b32 v25, v3, v65
	ds_bpermute_b32 v26, v3, v66
	ds_bpermute_b32 v27, v3, v67
	ds_bpermute_b32 v28, v3, v76
	ds_bpermute_b32 v29, v3, v77
	ds_bpermute_b32 v30, v3, v78
	ds_bpermute_b32 v31, v3, v79
	ds_bpermute_b32 v32, v3, v88
	ds_bpermute_b32 v33, v3, v89
	ds_bpermute_b32 v34, v3, v90
	ds_bpermute_b32 v35, v3, v91
	ds_bpermute_b32 v36, v3, v100
	ds_bpermute_b32 v37, v3, v101
	ds_bpermute_b32 v38, v3, v102
	ds_bpermute_b32 v39, v3, v103
	ds_bpermute_b32 v40, v3, v112
	ds_bpermute_b32 v41, v3, v113
	ds_bpermute_b32 v42, v3, v114
	ds_bpermute_b32 v43, v3, v115
	ds_bpermute_b32 v44, v3, v124
	ds_bpermute_b32 v45, v3, v125
	ds_bpermute_b32 v46, v3, v126
	ds_bpermute_b32 v47, v3, v127
	ds_bpermute_b32 v48, v3, v136
	ds_bpermute_b32 v49, v3, v137
	ds_bpermute_b32 v50, v3, v138
	ds_bpermute_b32 v51, v3, v139
	s_wait_dscnt 0x1b
	v_add_f32_e32 v64, v64, v24
	s_wait_dscnt 0x1a
	v_add_f32_e32 v65, v65, v25
	s_wait_dscnt 0x19
	v_add_f32_e32 v66, v66, v26
	s_wait_dscnt 0x18
	v_add_f32_e32 v67, v67, v27
	s_wait_dscnt 0x17
	v_add_f32_e32 v76, v76, v28
	s_wait_dscnt 0x16
	v_add_f32_e32 v77, v77, v29
	s_wait_dscnt 0x15
	v_add_f32_e32 v78, v78, v30
	s_wait_dscnt 0x14
	v_add_f32_e32 v79, v79, v31
	s_wait_dscnt 0x13
	v_add_f32_e32 v88, v88, v32
	s_wait_dscnt 0x12
	v_add_f32_e32 v89, v89, v33
	s_wait_dscnt 0x11
	v_add_f32_e32 v90, v90, v34
	s_wait_dscnt 0x10
	v_add_f32_e32 v91, v91, v35
	s_wait_dscnt 0xf
	v_add_f32_e32 v100, v100, v36
	s_wait_dscnt 0xe
	v_add_f32_e32 v101, v101, v37
	s_wait_dscnt 0xd
	v_add_f32_e32 v102, v102, v38
	s_wait_dscnt 0xc
	v_add_f32_e32 v103, v103, v39
	s_wait_dscnt 0xb
	v_add_f32_e32 v112, v112, v40
	s_wait_dscnt 0xa
	v_add_f32_e32 v113, v113, v41
	s_wait_dscnt 0x9
	v_add_f32_e32 v114, v114, v42
	s_wait_dscnt 0x8
	v_add_f32_e32 v115, v115, v43
	s_wait_dscnt 0x7
	v_add_f32_e32 v124, v124, v44
	s_wait_dscnt 0x6
	v_add_f32_e32 v125, v125, v45
	s_wait_dscnt 0x5
	v_add_f32_e32 v126, v126, v46
	s_wait_dscnt 0x4
	v_add_f32_e32 v127, v127, v47
	s_wait_dscnt 0x3
	v_add_f32_e32 v136, v136, v48
	s_wait_dscnt 0x2
	v_add_f32_e32 v137, v137, v49
	s_wait_dscnt 0x1
	v_add_f32_e32 v138, v138, v50
	s_wait_dscnt 0x0
	v_add_f32_e32 v139, v139, v51
	v_cmp_gt_u32_e32 vcc_lo, 28, v0
	v_cndmask_b32_e64 v3, 0, 4, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v3, v3, v0, 2
	ds_bpermute_b32 v24, v3, v64
	ds_bpermute_b32 v25, v3, v65
	ds_bpermute_b32 v26, v3, v66
	ds_bpermute_b32 v27, v3, v67
	ds_bpermute_b32 v28, v3, v76
	ds_bpermute_b32 v29, v3, v77
	ds_bpermute_b32 v30, v3, v78
	ds_bpermute_b32 v31, v3, v79
	ds_bpermute_b32 v32, v3, v88
	ds_bpermute_b32 v33, v3, v89
	ds_bpermute_b32 v34, v3, v90
	ds_bpermute_b32 v35, v3, v91
	ds_bpermute_b32 v36, v3, v100
	ds_bpermute_b32 v37, v3, v101
	ds_bpermute_b32 v38, v3, v102
	ds_bpermute_b32 v39, v3, v103
	ds_bpermute_b32 v40, v3, v112
	ds_bpermute_b32 v41, v3, v113
	ds_bpermute_b32 v42, v3, v114
	ds_bpermute_b32 v43, v3, v115
	ds_bpermute_b32 v44, v3, v124
	ds_bpermute_b32 v45, v3, v125
	ds_bpermute_b32 v46, v3, v126
	ds_bpermute_b32 v47, v3, v127
	ds_bpermute_b32 v48, v3, v136
	ds_bpermute_b32 v49, v3, v137
	ds_bpermute_b32 v50, v3, v138
	ds_bpermute_b32 v51, v3, v139
	s_wait_dscnt 0x1b
	v_add_f32_e32 v64, v64, v24
	s_wait_dscnt 0x1a
	v_add_f32_e32 v65, v65, v25
	s_wait_dscnt 0x19
	v_add_f32_e32 v66, v66, v26
	s_wait_dscnt 0x18
	v_add_f32_e32 v67, v67, v27
	s_wait_dscnt 0x17
	v_add_f32_e32 v76, v76, v28
	s_wait_dscnt 0x16
	v_add_f32_e32 v77, v77, v29
	s_wait_dscnt 0x15
	v_add_f32_e32 v78, v78, v30
	s_wait_dscnt 0x14
	v_add_f32_e32 v79, v79, v31
	s_wait_dscnt 0x13
	v_add_f32_e32 v88, v88, v32
	s_wait_dscnt 0x12
	v_add_f32_e32 v89, v89, v33
	s_wait_dscnt 0x11
	v_add_f32_e32 v90, v90, v34
	s_wait_dscnt 0x10
	v_add_f32_e32 v91, v91, v35
	s_wait_dscnt 0xf
	v_add_f32_e32 v100, v100, v36
	s_wait_dscnt 0xe
	v_add_f32_e32 v101, v101, v37
	s_wait_dscnt 0xd
	v_add_f32_e32 v102, v102, v38
	s_wait_dscnt 0xc
	v_add_f32_e32 v103, v103, v39
	s_wait_dscnt 0xb
	v_add_f32_e32 v112, v112, v40
	s_wait_dscnt 0xa
	v_add_f32_e32 v113, v113, v41
	s_wait_dscnt 0x9
	v_add_f32_e32 v114, v114, v42
	s_wait_dscnt 0x8
	v_add_f32_e32 v115, v115, v43
	s_wait_dscnt 0x7
	v_add_f32_e32 v124, v124, v44
	s_wait_dscnt 0x6
	v_add_f32_e32 v125, v125, v45
	s_wait_dscnt 0x5
	v_add_f32_e32 v126, v126, v46
	s_wait_dscnt 0x4
	v_add_f32_e32 v127, v127, v47
	s_wait_dscnt 0x3
	v_add_f32_e32 v136, v136, v48
	s_wait_dscnt 0x2
	v_add_f32_e32 v137, v137, v49
	s_wait_dscnt 0x1
	v_add_f32_e32 v138, v138, v50
	s_wait_dscnt 0x0
	v_add_f32_e32 v139, v139, v51
	v_cmp_gt_u32_e32 vcc_lo, 30, v0
	v_cndmask_b32_e64 v3, 0, 2, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v3, v3, v0, 2
	ds_bpermute_b32 v24, v3, v64
	ds_bpermute_b32 v25, v3, v65
	ds_bpermute_b32 v26, v3, v66
	ds_bpermute_b32 v27, v3, v67
	ds_bpermute_b32 v28, v3, v76
	ds_bpermute_b32 v29, v3, v77
	ds_bpermute_b32 v30, v3, v78
	ds_bpermute_b32 v31, v3, v79
	ds_bpermute_b32 v32, v3, v88
	ds_bpermute_b32 v33, v3, v89
	ds_bpermute_b32 v34, v3, v90
	ds_bpermute_b32 v35, v3, v91
	ds_bpermute_b32 v36, v3, v100
	ds_bpermute_b32 v37, v3, v101
	ds_bpermute_b32 v38, v3, v102
	ds_bpermute_b32 v39, v3, v103
	ds_bpermute_b32 v40, v3, v112
	ds_bpermute_b32 v41, v3, v113
	ds_bpermute_b32 v42, v3, v114
	ds_bpermute_b32 v43, v3, v115
	ds_bpermute_b32 v44, v3, v124
	ds_bpermute_b32 v45, v3, v125
	ds_bpermute_b32 v46, v3, v126
	ds_bpermute_b32 v47, v3, v127
	ds_bpermute_b32 v48, v3, v136
	ds_bpermute_b32 v49, v3, v137
	ds_bpermute_b32 v50, v3, v138
	ds_bpermute_b32 v51, v3, v139
	s_wait_dscnt 0x1b
	v_add_f32_e32 v64, v64, v24
	s_wait_dscnt 0x1a
	v_add_f32_e32 v65, v65, v25
	s_wait_dscnt 0x19
	v_add_f32_e32 v66, v66, v26
	s_wait_dscnt 0x18
	v_add_f32_e32 v67, v67, v27
	s_wait_dscnt 0x17
	v_add_f32_e32 v76, v76, v28
	s_wait_dscnt 0x16
	v_add_f32_e32 v77, v77, v29
	s_wait_dscnt 0x15
	v_add_f32_e32 v78, v78, v30
	s_wait_dscnt 0x14
	v_add_f32_e32 v79, v79, v31
	s_wait_dscnt 0x13
	v_add_f32_e32 v88, v88, v32
	s_wait_dscnt 0x12
	v_add_f32_e32 v89, v89, v33
	s_wait_dscnt 0x11
	v_add_f32_e32 v90, v90, v34
	s_wait_dscnt 0x10
	v_add_f32_e32 v91, v91, v35
	s_wait_dscnt 0xf
	v_add_f32_e32 v100, v100, v36
	s_wait_dscnt 0xe
	v_add_f32_e32 v101, v101, v37
	s_wait_dscnt 0xd
	v_add_f32_e32 v102, v102, v38
	s_wait_dscnt 0xc
	v_add_f32_e32 v103, v103, v39
	s_wait_dscnt 0xb
	v_add_f32_e32 v112, v112, v40
	s_wait_dscnt 0xa
	v_add_f32_e32 v113, v113, v41
	s_wait_dscnt 0x9
	v_add_f32_e32 v114, v114, v42
	s_wait_dscnt 0x8
	v_add_f32_e32 v115, v115, v43
	s_wait_dscnt 0x7
	v_add_f32_e32 v124, v124, v44
	s_wait_dscnt 0x6
	v_add_f32_e32 v125, v125, v45
	s_wait_dscnt 0x5
	v_add_f32_e32 v126, v126, v46
	s_wait_dscnt 0x4
	v_add_f32_e32 v127, v127, v47
	s_wait_dscnt 0x3
	v_add_f32_e32 v136, v136, v48
	s_wait_dscnt 0x2
	v_add_f32_e32 v137, v137, v49
	s_wait_dscnt 0x1
	v_add_f32_e32 v138, v138, v50
	s_wait_dscnt 0x0
	v_add_f32_e32 v139, v139, v51
	v_cmp_gt_u32_e32 vcc_lo, 31, v0
	v_cndmask_b32_e64 v3, 0, 1, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v3, v3, v0, 2
	ds_bpermute_b32 v24, v3, v64
	ds_bpermute_b32 v25, v3, v65
	ds_bpermute_b32 v26, v3, v66
	ds_bpermute_b32 v27, v3, v67
	ds_bpermute_b32 v28, v3, v76
	ds_bpermute_b32 v29, v3, v77
	ds_bpermute_b32 v30, v3, v78
	ds_bpermute_b32 v31, v3, v79
	ds_bpermute_b32 v32, v3, v88
	ds_bpermute_b32 v33, v3, v89
	ds_bpermute_b32 v34, v3, v90
	ds_bpermute_b32 v35, v3, v91
	ds_bpermute_b32 v36, v3, v100
	ds_bpermute_b32 v37, v3, v101
	ds_bpermute_b32 v38, v3, v102
	ds_bpermute_b32 v39, v3, v103
	ds_bpermute_b32 v40, v3, v112
	ds_bpermute_b32 v41, v3, v113
	ds_bpermute_b32 v42, v3, v114
	ds_bpermute_b32 v43, v3, v115
	ds_bpermute_b32 v44, v3, v124
	ds_bpermute_b32 v45, v3, v125
	ds_bpermute_b32 v46, v3, v126
	ds_bpermute_b32 v47, v3, v127
	ds_bpermute_b32 v48, v3, v136
	ds_bpermute_b32 v49, v3, v137
	ds_bpermute_b32 v50, v3, v138
	ds_bpermute_b32 v51, v3, v139
	s_wait_dscnt 0x1b
	v_add_f32_e32 v64, v64, v24
	s_wait_dscnt 0x1a
	v_add_f32_e32 v65, v65, v25
	s_wait_dscnt 0x19
	v_add_f32_e32 v66, v66, v26
	s_wait_dscnt 0x18
	v_add_f32_e32 v67, v67, v27
	s_wait_dscnt 0x17
	v_add_f32_e32 v76, v76, v28
	s_wait_dscnt 0x16
	v_add_f32_e32 v77, v77, v29
	s_wait_dscnt 0x15
	v_add_f32_e32 v78, v78, v30
	s_wait_dscnt 0x14
	v_add_f32_e32 v79, v79, v31
	s_wait_dscnt 0x13
	v_add_f32_e32 v88, v88, v32
	s_wait_dscnt 0x12
	v_add_f32_e32 v89, v89, v33
	s_wait_dscnt 0x11
	v_add_f32_e32 v90, v90, v34
	s_wait_dscnt 0x10
	v_add_f32_e32 v91, v91, v35
	s_wait_dscnt 0xf
	v_add_f32_e32 v100, v100, v36
	s_wait_dscnt 0xe
	v_add_f32_e32 v101, v101, v37
	s_wait_dscnt 0xd
	v_add_f32_e32 v102, v102, v38
	s_wait_dscnt 0xc
	v_add_f32_e32 v103, v103, v39
	s_wait_dscnt 0xb
	v_add_f32_e32 v112, v112, v40
	s_wait_dscnt 0xa
	v_add_f32_e32 v113, v113, v41
	s_wait_dscnt 0x9
	v_add_f32_e32 v114, v114, v42
	s_wait_dscnt 0x8
	v_add_f32_e32 v115, v115, v43
	s_wait_dscnt 0x7
	v_add_f32_e32 v124, v124, v44
	s_wait_dscnt 0x6
	v_add_f32_e32 v125, v125, v45
	s_wait_dscnt 0x5
	v_add_f32_e32 v126, v126, v46
	s_wait_dscnt 0x4
	v_add_f32_e32 v127, v127, v47
	s_wait_dscnt 0x3
	v_add_f32_e32 v136, v136, v48
	s_wait_dscnt 0x2
	v_add_f32_e32 v137, v137, v49
	s_wait_dscnt 0x1
	v_add_f32_e32 v138, v138, v50
	s_wait_dscnt 0x0
	v_add_f32_e32 v139, v139, v51
	v_cmpx_eq_u32_e32 0, v0
	s_lshl_b32 s46, s12, 2
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v165, s46
	s_mul_i32 s47, s45, 1
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s47, s47, s46
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v166, s47
	s_mul_i32 s47, s45, 2
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s47, s47, s46
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v167, s47
	s_mul_i32 s47, s45, 3
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s47, s47, s46
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v168, s47
	s_mul_i32 s47, s45, 4
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s47, s47, s46
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v169, s47
	s_mul_i32 s47, s45, 5
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s47, s47, s46
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v170, s47
	s_mul_i32 s47, s45, 6
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s47, s47, s46
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v171, s47
	s_add_co_i32 s13, s12, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cbranch_scc0 .Lrx_b7_p0_single
	global_load_b32 v52, v165, s[8:9]
	global_load_b32 v53, v165, s[8:9] offset:4
	global_load_b32 v56, v166, s[8:9]
	global_load_b32 v57, v166, s[8:9] offset:4
	global_load_b32 v60, v167, s[8:9]
	global_load_b32 v61, v167, s[8:9] offset:4
	global_load_b32 v149, v168, s[8:9]
	global_load_b32 v150, v168, s[8:9] offset:4
	global_load_b32 v153, v169, s[8:9]
	global_load_b32 v154, v169, s[8:9] offset:4
	global_load_b32 v157, v170, s[8:9]
	global_load_b32 v158, v170, s[8:9] offset:4
	global_load_b32 v161, v171, s[8:9]
	global_load_b32 v162, v171, s[8:9] offset:4
	s_wait_loadcnt 0xd
	v_add_f32_e32 v52, v64, v52
	global_store_b32 v165, v52, s[8:9]
	s_wait_loadcnt 0xc
	v_add_f32_e32 v53, v65, v53
	global_store_b32 v165, v53, s[8:9] offset:4
	s_wait_loadcnt 0xb
	v_add_f32_e32 v56, v76, v56
	global_store_b32 v166, v56, s[8:9]
	s_wait_loadcnt 0xa
	v_add_f32_e32 v57, v77, v57
	global_store_b32 v166, v57, s[8:9] offset:4
	s_wait_loadcnt 0x9
	v_add_f32_e32 v60, v88, v60
	global_store_b32 v167, v60, s[8:9]
	s_wait_loadcnt 0x8
	v_add_f32_e32 v61, v89, v61
	global_store_b32 v167, v61, s[8:9] offset:4
	s_wait_loadcnt 0x7
	v_add_f32_e32 v149, v100, v149
	global_store_b32 v168, v149, s[8:9]
	s_wait_loadcnt 0x6
	v_add_f32_e32 v150, v101, v150
	global_store_b32 v168, v150, s[8:9] offset:4
	s_wait_loadcnt 0x5
	v_add_f32_e32 v153, v112, v153
	global_store_b32 v169, v153, s[8:9]
	s_wait_loadcnt 0x4
	v_add_f32_e32 v154, v113, v154
	global_store_b32 v169, v154, s[8:9] offset:4
	s_wait_loadcnt 0x3
	v_add_f32_e32 v157, v124, v157
	global_store_b32 v170, v157, s[8:9]
	s_wait_loadcnt 0x2
	v_add_f32_e32 v158, v125, v158
	global_store_b32 v170, v158, s[8:9] offset:4
	s_wait_loadcnt 0x1
	v_add_f32_e32 v161, v136, v161
	global_store_b32 v171, v161, s[8:9]
	s_wait_loadcnt 0x0
	v_add_f32_e32 v162, v137, v162
	global_store_b32 v171, v162, s[8:9] offset:4
	s_wait_storecnt 0x0
	s_branch .Lrx_b7_p0_next
	.Lrx_b7_p0_single:
	global_load_b32 v52, v165, s[8:9]
	global_load_b32 v56, v166, s[8:9]
	global_load_b32 v60, v167, s[8:9]
	global_load_b32 v149, v168, s[8:9]
	global_load_b32 v153, v169, s[8:9]
	global_load_b32 v157, v170, s[8:9]
	global_load_b32 v161, v171, s[8:9]
	s_wait_loadcnt 0x6
	v_add_f32_e32 v52, v52, v64
	global_store_b32 v165, v52, s[8:9]
	s_wait_loadcnt 0x5
	v_add_f32_e32 v56, v56, v76
	global_store_b32 v166, v56, s[8:9]
	s_wait_loadcnt 0x4
	v_add_f32_e32 v60, v60, v88
	global_store_b32 v167, v60, s[8:9]
	s_wait_loadcnt 0x3
	v_add_f32_e32 v149, v149, v100
	global_store_b32 v168, v149, s[8:9]
	s_wait_loadcnt 0x2
	v_add_f32_e32 v153, v153, v112
	global_store_b32 v169, v153, s[8:9]
	s_wait_loadcnt 0x1
	v_add_f32_e32 v157, v157, v124
	global_store_b32 v170, v157, s[8:9]
	s_wait_loadcnt 0x0
	v_add_f32_e32 v161, v161, v136
	global_store_b32 v171, v161, s[8:9]
	s_wait_storecnt 0x0
	s_branch .Lrx_b7_stored
	.Lrx_b7_p0_next:
	s_add_co_i32 s13, s12, 2
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cbranch_scc0 .Lrx_b7_stored
	s_add_co_i32 s13, s12, 3
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cbranch_scc0 .Lrx_b7_p1_single
	global_load_b32 v54, v165, s[8:9] offset:8
	global_load_b32 v55, v165, s[8:9] offset:12
	global_load_b32 v58, v166, s[8:9] offset:8
	global_load_b32 v59, v166, s[8:9] offset:12
	global_load_b32 v62, v167, s[8:9] offset:8
	global_load_b32 v63, v167, s[8:9] offset:12
	global_load_b32 v151, v168, s[8:9] offset:8
	global_load_b32 v152, v168, s[8:9] offset:12
	global_load_b32 v155, v169, s[8:9] offset:8
	global_load_b32 v156, v169, s[8:9] offset:12
	global_load_b32 v159, v170, s[8:9] offset:8
	global_load_b32 v160, v170, s[8:9] offset:12
	global_load_b32 v163, v171, s[8:9] offset:8
	global_load_b32 v164, v171, s[8:9] offset:12
	s_wait_loadcnt 0xd
	v_add_f32_e32 v54, v66, v54
	global_store_b32 v165, v54, s[8:9] offset:8
	s_wait_loadcnt 0xc
	v_add_f32_e32 v55, v67, v55
	global_store_b32 v165, v55, s[8:9] offset:12
	s_wait_loadcnt 0xb
	v_add_f32_e32 v58, v78, v58
	global_store_b32 v166, v58, s[8:9] offset:8
	s_wait_loadcnt 0xa
	v_add_f32_e32 v59, v79, v59
	global_store_b32 v166, v59, s[8:9] offset:12
	s_wait_loadcnt 0x9
	v_add_f32_e32 v62, v90, v62
	global_store_b32 v167, v62, s[8:9] offset:8
	s_wait_loadcnt 0x8
	v_add_f32_e32 v63, v91, v63
	global_store_b32 v167, v63, s[8:9] offset:12
	s_wait_loadcnt 0x7
	v_add_f32_e32 v151, v102, v151
	global_store_b32 v168, v151, s[8:9] offset:8
	s_wait_loadcnt 0x6
	v_add_f32_e32 v152, v103, v152
	global_store_b32 v168, v152, s[8:9] offset:12
	s_wait_loadcnt 0x5
	v_add_f32_e32 v155, v114, v155
	global_store_b32 v169, v155, s[8:9] offset:8
	s_wait_loadcnt 0x4
	v_add_f32_e32 v156, v115, v156
	global_store_b32 v169, v156, s[8:9] offset:12
	s_wait_loadcnt 0x3
	v_add_f32_e32 v159, v126, v159
	global_store_b32 v170, v159, s[8:9] offset:8
	s_wait_loadcnt 0x2
	v_add_f32_e32 v160, v127, v160
	global_store_b32 v170, v160, s[8:9] offset:12
	s_wait_loadcnt 0x1
	v_add_f32_e32 v163, v138, v163
	global_store_b32 v171, v163, s[8:9] offset:8
	s_wait_loadcnt 0x0
	v_add_f32_e32 v164, v139, v164
	global_store_b32 v171, v164, s[8:9] offset:12
	s_wait_storecnt 0x0
	s_branch .Lrx_b7_stored
	.Lrx_b7_p1_single:
	global_load_b32 v54, v165, s[8:9] offset:8
	global_load_b32 v58, v166, s[8:9] offset:8
	global_load_b32 v62, v167, s[8:9] offset:8
	global_load_b32 v151, v168, s[8:9] offset:8
	global_load_b32 v155, v169, s[8:9] offset:8
	global_load_b32 v159, v170, s[8:9] offset:8
	global_load_b32 v163, v171, s[8:9] offset:8
	s_wait_loadcnt 0x6
	v_add_f32_e32 v54, v54, v66
	global_store_b32 v165, v54, s[8:9] offset:8
	s_wait_loadcnt 0x5
	v_add_f32_e32 v58, v58, v78
	global_store_b32 v166, v58, s[8:9] offset:8
	s_wait_loadcnt 0x4
	v_add_f32_e32 v62, v62, v90
	global_store_b32 v167, v62, s[8:9] offset:8
	s_wait_loadcnt 0x3
	v_add_f32_e32 v151, v151, v102
	global_store_b32 v168, v151, s[8:9] offset:8
	s_wait_loadcnt 0x2
	v_add_f32_e32 v155, v155, v114
	global_store_b32 v169, v155, s[8:9] offset:8
	s_wait_loadcnt 0x1
	v_add_f32_e32 v159, v159, v126
	global_store_b32 v170, v159, s[8:9] offset:8
	s_wait_loadcnt 0x0
	v_add_f32_e32 v163, v163, v138
	global_store_b32 v171, v163, s[8:9] offset:8
	s_wait_storecnt 0x0
	s_branch .Lrx_b7_stored
	.Lrx_b7_stored:
	s_branch .Lrx_end
	.Lrx_b8:
	s_mul_i32 s12, ttmp9, 4
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_ge_i32 s12, s10
	s_cbranch_scc1 .Lrx_end
	s_mov_b32 s13, s12
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s13, s13, s19
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v8, s13
	v_add_nc_u32_e32 v4, s13, v1
	s_add_co_i32 s13, s12, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cselect_b32 s13, s13, s12
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s13, s13, s19
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v9, s13
	v_add_nc_u32_e32 v5, s13, v1
	s_add_co_i32 s13, s12, 2
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cselect_b32 s13, s13, s12
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s13, s13, s19
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v10, s13
	v_add_nc_u32_e32 v6, s13, v1
	s_add_co_i32 s13, s12, 3
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cselect_b32 s13, s13, s12
	s_wait_alu depctr_sa_sdst(0)
	s_mul_i32 s13, s13, s19
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v11, s13
	v_add_nc_u32_e32 v7, s13, v1
	v_mov_b32_e32 v64, 0
	v_mov_b32_e32 v65, 0
	v_mov_b32_e32 v66, 0
	v_mov_b32_e32 v67, 0
	v_mov_b32_e32 v68, 0
	v_mov_b32_e32 v69, 0
	v_mov_b32_e32 v70, 0
	v_mov_b32_e32 v71, 0
	v_mov_b32_e32 v72, 0
	v_mov_b32_e32 v73, 0
	v_mov_b32_e32 v74, 0
	v_mov_b32_e32 v75, 0
	v_mov_b32_e32 v76, 0
	v_mov_b32_e32 v77, 0
	v_mov_b32_e32 v78, 0
	v_mov_b32_e32 v79, 0
	v_mov_b32_e32 v80, 0
	v_mov_b32_e32 v81, 0
	v_mov_b32_e32 v82, 0
	v_mov_b32_e32 v83, 0
	v_mov_b32_e32 v84, 0
	v_mov_b32_e32 v85, 0
	v_mov_b32_e32 v86, 0
	v_mov_b32_e32 v87, 0
	v_mov_b32_e32 v88, 0
	v_mov_b32_e32 v89, 0
	v_mov_b32_e32 v90, 0
	v_mov_b32_e32 v91, 0
	v_mov_b32_e32 v92, 0
	v_mov_b32_e32 v93, 0
	v_mov_b32_e32 v94, 0
	v_mov_b32_e32 v95, 0
	v_mov_b32_e32 v96, 0
	v_mov_b32_e32 v97, 0
	v_mov_b32_e32 v98, 0
	v_mov_b32_e32 v99, 0
	v_mov_b32_e32 v100, 0
	v_mov_b32_e32 v101, 0
	v_mov_b32_e32 v102, 0
	v_mov_b32_e32 v103, 0
	v_mov_b32_e32 v104, 0
	v_mov_b32_e32 v105, 0
	v_mov_b32_e32 v106, 0
	v_mov_b32_e32 v107, 0
	v_mov_b32_e32 v108, 0
	v_mov_b32_e32 v109, 0
	v_mov_b32_e32 v110, 0
	v_mov_b32_e32 v111, 0
	v_mov_b32_e32 v112, 0
	v_mov_b32_e32 v113, 0
	v_mov_b32_e32 v114, 0
	v_mov_b32_e32 v115, 0
	v_mov_b32_e32 v116, 0
	v_mov_b32_e32 v117, 0
	v_mov_b32_e32 v118, 0
	v_mov_b32_e32 v119, 0
	v_mov_b32_e32 v120, 0
	v_mov_b32_e32 v121, 0
	v_mov_b32_e32 v122, 0
	v_mov_b32_e32 v123, 0
	v_mov_b32_e32 v124, 0
	v_mov_b32_e32 v125, 0
	v_mov_b32_e32 v126, 0
	v_mov_b32_e32 v127, 0
	v_mov_b32_e32 v128, 0
	v_mov_b32_e32 v129, 0
	v_mov_b32_e32 v130, 0
	v_mov_b32_e32 v131, 0
	v_mov_b32_e32 v132, 0
	v_mov_b32_e32 v133, 0
	v_mov_b32_e32 v134, 0
	v_mov_b32_e32 v135, 0
	v_mov_b32_e32 v136, 0
	v_mov_b32_e32 v137, 0
	v_mov_b32_e32 v138, 0
	v_mov_b32_e32 v139, 0
	v_mov_b32_e32 v140, 0
	v_mov_b32_e32 v141, 0
	v_mov_b32_e32 v142, 0
	v_mov_b32_e32 v143, 0
	v_mov_b32_e32 v144, 0
	v_mov_b32_e32 v145, 0
	v_mov_b32_e32 v146, 0
	v_mov_b32_e32 v147, 0
	v_mov_b32_e32 v148, 0
	v_mov_b32_e32 v149, 0
	v_mov_b32_e32 v150, 0
	v_mov_b32_e32 v151, 0
	v_mov_b32_e32 v152, 0
	v_mov_b32_e32 v153, 0
	v_mov_b32_e32 v154, 0
	v_mov_b32_e32 v155, 0
	v_mov_b32_e32 v156, 0
	v_mov_b32_e32 v157, 0
	v_mov_b32_e32 v158, 0
	v_mov_b32_e32 v159, 0
	s_add_co_i32 s2, s14, 3
	s_lshr_b32 s2, s2, 2
	s_cmp_eq_u32 s2, 0
	s_cbranch_scc1 .Lrx_b8_s0_done
	s_mov_b32 s15, 0
	v_lshlrev_b32_e32 v2, 5, v0
	buffer_load_b64 v[12:13], v8, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v20, v4, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[14:15], v9, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v21, v5, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[16:17], v10, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v22, v6, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[18:19], v11, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v23, v7, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	global_load_b128 v[161:164], v2, s[28:29]
	global_load_b128 v[165:168], v2, s[28:29] offset:16
	global_load_b128 v[169:172], v2, s[30:31]
	global_load_b128 v[173:176], v2, s[30:31] offset:16
	global_load_b128 v[177:180], v2, s[32:33]
	global_load_b128 v[181:184], v2, s[32:33] offset:16
	global_load_b128 v[185:188], v2, s[34:35]
	global_load_b128 v[189:192], v2, s[34:35] offset:16
	global_load_b128 v[193:196], v2, s[36:37]
	global_load_b128 v[197:200], v2, s[36:37] offset:16
	global_load_b128 v[201:204], v2, s[38:39]
	global_load_b128 v[205:208], v2, s[38:39] offset:16
	global_load_b128 v[209:212], v2, s[40:41]
	global_load_b128 v[213:216], v2, s[40:41] offset:16
	global_load_b128 v[217:220], v2, s[42:43]
	global_load_b128 v[221:224], v2, s[42:43] offset:16
	.Lrx_b8_s0:
	s_cmp_gt_u32 s2, 1
	s_cselect_b32 s17, 0x220, 0
	s_cselect_b32 s18, 0x1000, 0
	s_add_co_i32 s16, s15, s17
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v3, s18, v2
	v_cmp_gt_u32_e32 vcc_lo, 16, v0
	s_wait_loadcnt 0x17
	v_cndmask_b32_e32 v12, v13, v12, vcc_lo
	s_wait_loadcnt 0x16
	v_and_b32_e32 v30, 0xf0f0f0f, v20
	v_lshrrev_b32_e32 v31, 4, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v31, 0xf0f0f0f, v31
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v24, v30
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v26, v30
	v_cvt_f32_ubyte2_e32 v28, v30
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v25, v31
	v_cvt_f32_ubyte1_e32 v27, v31
	v_cvt_f32_ubyte2_e32 v29, v31
	v_cvt_f32_ubyte3_e32 v30, v30
	v_cvt_f32_ubyte3_e32 v31, v31
	v_fma_mix_f32 v24, v12, v24, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v25, v12, v25, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v26, v12, v26, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v27, v12, v27, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v28, v12, v28, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v29, v12, v29, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v30, v12, v30, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v31, v12, v31, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x15
	v_cndmask_b32_e32 v14, v15, v14, vcc_lo
	s_wait_loadcnt 0x14
	v_and_b32_e32 v40, 0xf0f0f0f, v21
	v_lshrrev_b32_e32 v41, 4, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v41, 0xf0f0f0f, v41
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v34, v40
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v36, v40
	v_cvt_f32_ubyte2_e32 v38, v40
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v35, v41
	v_cvt_f32_ubyte1_e32 v37, v41
	v_cvt_f32_ubyte2_e32 v39, v41
	v_cvt_f32_ubyte3_e32 v40, v40
	v_cvt_f32_ubyte3_e32 v41, v41
	v_fma_mix_f32 v34, v14, v34, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v35, v14, v35, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v36, v14, v36, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v37, v14, v37, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v38, v14, v38, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v39, v14, v39, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v40, v14, v40, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v41, v14, v41, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x13
	v_cndmask_b32_e32 v16, v17, v16, vcc_lo
	s_wait_loadcnt 0x12
	v_and_b32_e32 v50, 0xf0f0f0f, v22
	v_lshrrev_b32_e32 v51, 4, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v51, 0xf0f0f0f, v51
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v44, v50
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v46, v50
	v_cvt_f32_ubyte2_e32 v48, v50
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v45, v51
	v_cvt_f32_ubyte1_e32 v47, v51
	v_cvt_f32_ubyte2_e32 v49, v51
	v_cvt_f32_ubyte3_e32 v50, v50
	v_cvt_f32_ubyte3_e32 v51, v51
	v_fma_mix_f32 v44, v16, v44, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v16, v45, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v16, v46, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v16, v47, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v48, v16, v48, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v49, v16, v49, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v16, v50, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v16, v51, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x11
	v_cndmask_b32_e32 v18, v19, v18, vcc_lo
	s_wait_loadcnt 0x10
	v_and_b32_e32 v60, 0xf0f0f0f, v23
	v_lshrrev_b32_e32 v61, 4, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v61, 0xf0f0f0f, v61
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v54, v60
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v56, v60
	v_cvt_f32_ubyte2_e32 v58, v60
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v55, v61
	v_cvt_f32_ubyte1_e32 v57, v61
	v_cvt_f32_ubyte2_e32 v59, v61
	v_cvt_f32_ubyte3_e32 v60, v60
	v_cvt_f32_ubyte3_e32 v61, v61
	v_fma_mix_f32 v54, v18, v54, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v18, v55, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v18, v56, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v18, v57, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v58, v18, v58, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v59, v18, v59, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v18, v60, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v18, v61, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	buffer_load_b64 v[12:13], v8, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v20, v4, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[14:15], v9, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v21, v5, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[16:17], v10, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v22, v6, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[18:19], v11, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v23, v7, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	s_wait_loadcnt 0x17
	v_dual_mul_f32 v32, v24, v161 :: v_dual_mul_f32 v33, v35, v162
	v_dual_mul_f32 v42, v44, v161 :: v_dual_mul_f32 v43, v55, v162
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v52, v24, v169 :: v_dual_mul_f32 v53, v35, v170
	v_dual_mul_f32 v62, v44, v169 :: v_dual_mul_f32 v63, v55, v170
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v25, v162 :: v_dual_fmac_f32 v33, v34, v161
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v45, v162 :: v_dual_fmac_f32 v43, v54, v161
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v25, v170 :: v_dual_fmac_f32 v53, v34, v169
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v45, v170 :: v_dual_fmac_f32 v63, v54, v169
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v26, v163 :: v_dual_fmac_f32 v33, v36, v163
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v46, v163 :: v_dual_fmac_f32 v43, v56, v163
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v26, v171 :: v_dual_fmac_f32 v53, v36, v171
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v46, v171 :: v_dual_fmac_f32 v63, v56, v171
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v27, v164 :: v_dual_fmac_f32 v33, v37, v164
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v47, v164 :: v_dual_fmac_f32 v43, v57, v164
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v27, v172 :: v_dual_fmac_f32 v53, v37, v172
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v47, v172 :: v_dual_fmac_f32 v63, v57, v172
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v28, v165 :: v_dual_fmac_f32 v33, v38, v165
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v48, v165 :: v_dual_fmac_f32 v43, v58, v165
	s_wait_loadcnt 0x14
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v28, v173 :: v_dual_fmac_f32 v53, v38, v173
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v48, v173 :: v_dual_fmac_f32 v63, v58, v173
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v29, v166 :: v_dual_fmac_f32 v33, v39, v166
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v49, v166 :: v_dual_fmac_f32 v43, v59, v166
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v29, v174 :: v_dual_fmac_f32 v53, v39, v174
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v49, v174 :: v_dual_fmac_f32 v63, v59, v174
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v30, v167 :: v_dual_fmac_f32 v33, v40, v167
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v50, v167 :: v_dual_fmac_f32 v43, v60, v167
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v30, v175 :: v_dual_fmac_f32 v53, v40, v175
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v50, v175 :: v_dual_fmac_f32 v63, v60, v175
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v31, v168 :: v_dual_fmac_f32 v33, v41, v168
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v51, v168 :: v_dual_fmac_f32 v43, v61, v168
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v31, v176 :: v_dual_fmac_f32 v53, v41, v176
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v51, v176 :: v_dual_fmac_f32 v63, v61, v176
	global_load_b128 v[161:164], v3, s[28:29]
	global_load_b128 v[165:168], v3, s[28:29] offset:16
	global_load_b128 v[169:172], v3, s[30:31]
	global_load_b128 v[173:176], v3, s[30:31] offset:16
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v64, v32, v64 :: v_dual_add_f32 v65, v65, v33
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v66, v42, v66 :: v_dual_add_f32 v67, v67, v43
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v76, v52, v76 :: v_dual_add_f32 v77, v77, v53
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v78, v62, v78 :: v_dual_add_f32 v79, v79, v63
	s_wait_loadcnt 0x17
	v_dual_mul_f32 v32, v24, v177 :: v_dual_mul_f32 v33, v35, v178
	v_dual_mul_f32 v42, v44, v177 :: v_dual_mul_f32 v43, v55, v178
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v52, v24, v185 :: v_dual_mul_f32 v53, v35, v186
	v_dual_mul_f32 v62, v44, v185 :: v_dual_mul_f32 v63, v55, v186
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v25, v178 :: v_dual_fmac_f32 v33, v34, v177
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v45, v178 :: v_dual_fmac_f32 v43, v54, v177
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v25, v186 :: v_dual_fmac_f32 v53, v34, v185
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v45, v186 :: v_dual_fmac_f32 v63, v54, v185
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v26, v179 :: v_dual_fmac_f32 v33, v36, v179
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v46, v179 :: v_dual_fmac_f32 v43, v56, v179
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v26, v187 :: v_dual_fmac_f32 v53, v36, v187
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v46, v187 :: v_dual_fmac_f32 v63, v56, v187
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v27, v180 :: v_dual_fmac_f32 v33, v37, v180
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v47, v180 :: v_dual_fmac_f32 v43, v57, v180
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v27, v188 :: v_dual_fmac_f32 v53, v37, v188
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v47, v188 :: v_dual_fmac_f32 v63, v57, v188
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v28, v181 :: v_dual_fmac_f32 v33, v38, v181
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v48, v181 :: v_dual_fmac_f32 v43, v58, v181
	s_wait_loadcnt 0x14
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v28, v189 :: v_dual_fmac_f32 v53, v38, v189
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v48, v189 :: v_dual_fmac_f32 v63, v58, v189
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v29, v182 :: v_dual_fmac_f32 v33, v39, v182
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v49, v182 :: v_dual_fmac_f32 v43, v59, v182
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v29, v190 :: v_dual_fmac_f32 v53, v39, v190
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v49, v190 :: v_dual_fmac_f32 v63, v59, v190
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v30, v183 :: v_dual_fmac_f32 v33, v40, v183
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v50, v183 :: v_dual_fmac_f32 v43, v60, v183
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v30, v191 :: v_dual_fmac_f32 v53, v40, v191
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v50, v191 :: v_dual_fmac_f32 v63, v60, v191
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v31, v184 :: v_dual_fmac_f32 v33, v41, v184
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v51, v184 :: v_dual_fmac_f32 v43, v61, v184
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v31, v192 :: v_dual_fmac_f32 v53, v41, v192
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v51, v192 :: v_dual_fmac_f32 v63, v61, v192
	global_load_b128 v[177:180], v3, s[32:33]
	global_load_b128 v[181:184], v3, s[32:33] offset:16
	global_load_b128 v[185:188], v3, s[34:35]
	global_load_b128 v[189:192], v3, s[34:35] offset:16
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v88, v32, v88 :: v_dual_add_f32 v89, v89, v33
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v90, v42, v90 :: v_dual_add_f32 v91, v91, v43
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v100, v52, v100 :: v_dual_add_f32 v101, v101, v53
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v102, v62, v102 :: v_dual_add_f32 v103, v103, v63
	s_wait_loadcnt 0x17
	v_dual_mul_f32 v32, v24, v193 :: v_dual_mul_f32 v33, v35, v194
	v_dual_mul_f32 v42, v44, v193 :: v_dual_mul_f32 v43, v55, v194
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v52, v24, v201 :: v_dual_mul_f32 v53, v35, v202
	v_dual_mul_f32 v62, v44, v201 :: v_dual_mul_f32 v63, v55, v202
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v25, v194 :: v_dual_fmac_f32 v33, v34, v193
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v45, v194 :: v_dual_fmac_f32 v43, v54, v193
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v25, v202 :: v_dual_fmac_f32 v53, v34, v201
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v45, v202 :: v_dual_fmac_f32 v63, v54, v201
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v26, v195 :: v_dual_fmac_f32 v33, v36, v195
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v46, v195 :: v_dual_fmac_f32 v43, v56, v195
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v26, v203 :: v_dual_fmac_f32 v53, v36, v203
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v46, v203 :: v_dual_fmac_f32 v63, v56, v203
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v27, v196 :: v_dual_fmac_f32 v33, v37, v196
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v47, v196 :: v_dual_fmac_f32 v43, v57, v196
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v27, v204 :: v_dual_fmac_f32 v53, v37, v204
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v47, v204 :: v_dual_fmac_f32 v63, v57, v204
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v28, v197 :: v_dual_fmac_f32 v33, v38, v197
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v48, v197 :: v_dual_fmac_f32 v43, v58, v197
	s_wait_loadcnt 0x14
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v28, v205 :: v_dual_fmac_f32 v53, v38, v205
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v48, v205 :: v_dual_fmac_f32 v63, v58, v205
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v29, v198 :: v_dual_fmac_f32 v33, v39, v198
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v49, v198 :: v_dual_fmac_f32 v43, v59, v198
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v29, v206 :: v_dual_fmac_f32 v53, v39, v206
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v49, v206 :: v_dual_fmac_f32 v63, v59, v206
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v30, v199 :: v_dual_fmac_f32 v33, v40, v199
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v50, v199 :: v_dual_fmac_f32 v43, v60, v199
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v30, v207 :: v_dual_fmac_f32 v53, v40, v207
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v50, v207 :: v_dual_fmac_f32 v63, v60, v207
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v31, v200 :: v_dual_fmac_f32 v33, v41, v200
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v51, v200 :: v_dual_fmac_f32 v43, v61, v200
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v31, v208 :: v_dual_fmac_f32 v53, v41, v208
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v51, v208 :: v_dual_fmac_f32 v63, v61, v208
	global_load_b128 v[193:196], v3, s[36:37]
	global_load_b128 v[197:200], v3, s[36:37] offset:16
	global_load_b128 v[201:204], v3, s[38:39]
	global_load_b128 v[205:208], v3, s[38:39] offset:16
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v112, v32, v112 :: v_dual_add_f32 v113, v113, v33
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v114, v42, v114 :: v_dual_add_f32 v115, v115, v43
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v124, v52, v124 :: v_dual_add_f32 v125, v125, v53
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v126, v62, v126 :: v_dual_add_f32 v127, v127, v63
	s_wait_loadcnt 0x17
	v_dual_mul_f32 v32, v24, v209 :: v_dual_mul_f32 v33, v35, v210
	v_dual_mul_f32 v42, v44, v209 :: v_dual_mul_f32 v43, v55, v210
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v52, v24, v217 :: v_dual_mul_f32 v53, v35, v218
	v_dual_mul_f32 v62, v44, v217 :: v_dual_mul_f32 v63, v55, v218
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v25, v210 :: v_dual_fmac_f32 v33, v34, v209
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v45, v210 :: v_dual_fmac_f32 v43, v54, v209
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v25, v218 :: v_dual_fmac_f32 v53, v34, v217
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v45, v218 :: v_dual_fmac_f32 v63, v54, v217
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v26, v211 :: v_dual_fmac_f32 v33, v36, v211
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v46, v211 :: v_dual_fmac_f32 v43, v56, v211
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v26, v219 :: v_dual_fmac_f32 v53, v36, v219
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v46, v219 :: v_dual_fmac_f32 v63, v56, v219
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v27, v212 :: v_dual_fmac_f32 v33, v37, v212
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v47, v212 :: v_dual_fmac_f32 v43, v57, v212
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v27, v220 :: v_dual_fmac_f32 v53, v37, v220
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v47, v220 :: v_dual_fmac_f32 v63, v57, v220
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v28, v213 :: v_dual_fmac_f32 v33, v38, v213
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v48, v213 :: v_dual_fmac_f32 v43, v58, v213
	s_wait_loadcnt 0x14
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v28, v221 :: v_dual_fmac_f32 v53, v38, v221
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v48, v221 :: v_dual_fmac_f32 v63, v58, v221
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v29, v214 :: v_dual_fmac_f32 v33, v39, v214
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v49, v214 :: v_dual_fmac_f32 v43, v59, v214
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v29, v222 :: v_dual_fmac_f32 v53, v39, v222
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v49, v222 :: v_dual_fmac_f32 v63, v59, v222
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v30, v215 :: v_dual_fmac_f32 v33, v40, v215
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v50, v215 :: v_dual_fmac_f32 v43, v60, v215
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v30, v223 :: v_dual_fmac_f32 v53, v40, v223
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v50, v223 :: v_dual_fmac_f32 v63, v60, v223
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v31, v216 :: v_dual_fmac_f32 v33, v41, v216
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v51, v216 :: v_dual_fmac_f32 v43, v61, v216
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v31, v224 :: v_dual_fmac_f32 v53, v41, v224
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v51, v224 :: v_dual_fmac_f32 v63, v61, v224
	global_load_b128 v[209:212], v3, s[40:41]
	global_load_b128 v[213:216], v3, s[40:41] offset:16
	global_load_b128 v[217:220], v3, s[42:43]
	global_load_b128 v[221:224], v3, s[42:43] offset:16
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v136, v32, v136 :: v_dual_add_f32 v137, v137, v33
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v138, v42, v138 :: v_dual_add_f32 v139, v139, v43
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v148, v52, v148 :: v_dual_add_f32 v149, v149, v53
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v150, v62, v150 :: v_dual_add_f32 v151, v151, v63
	s_mov_b32 s15, s16
	v_mov_b32_e32 v2, v3
	s_add_co_i32 s2, s2, -1
	s_cmp_lg_u32 s2, 0
	s_cbranch_scc1 .Lrx_b8_s0
	s_wait_loadcnt 0x0
	.Lrx_b8_s0_done:
	s_add_co_i32 s2, s14, 2
	s_lshr_b32 s2, s2, 2
	s_cmp_eq_u32 s2, 0
	s_cbranch_scc1 .Lrx_b8_s1_done
	s_mov_b32 s15, 0x88
	v_lshlrev_b32_e32 v2, 5, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v2, 0x400, v2
	buffer_load_b64 v[12:13], v8, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v20, v4, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[14:15], v9, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v21, v5, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[16:17], v10, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v22, v6, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[18:19], v11, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v23, v7, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	global_load_b128 v[161:164], v2, s[28:29]
	global_load_b128 v[165:168], v2, s[28:29] offset:16
	global_load_b128 v[169:172], v2, s[30:31]
	global_load_b128 v[173:176], v2, s[30:31] offset:16
	global_load_b128 v[177:180], v2, s[32:33]
	global_load_b128 v[181:184], v2, s[32:33] offset:16
	global_load_b128 v[185:188], v2, s[34:35]
	global_load_b128 v[189:192], v2, s[34:35] offset:16
	global_load_b128 v[193:196], v2, s[36:37]
	global_load_b128 v[197:200], v2, s[36:37] offset:16
	global_load_b128 v[201:204], v2, s[38:39]
	global_load_b128 v[205:208], v2, s[38:39] offset:16
	global_load_b128 v[209:212], v2, s[40:41]
	global_load_b128 v[213:216], v2, s[40:41] offset:16
	global_load_b128 v[217:220], v2, s[42:43]
	global_load_b128 v[221:224], v2, s[42:43] offset:16
	.Lrx_b8_s1:
	s_cmp_gt_u32 s2, 1
	s_cselect_b32 s17, 0x220, 0
	s_cselect_b32 s18, 0x1000, 0
	s_add_co_i32 s16, s15, s17
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v3, s18, v2
	v_cmp_gt_u32_e32 vcc_lo, 16, v0
	s_wait_loadcnt 0x17
	v_cndmask_b32_e32 v12, v13, v12, vcc_lo
	s_wait_loadcnt 0x16
	v_and_b32_e32 v30, 0xf0f0f0f, v20
	v_lshrrev_b32_e32 v31, 4, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v31, 0xf0f0f0f, v31
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v24, v30
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v26, v30
	v_cvt_f32_ubyte2_e32 v28, v30
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v25, v31
	v_cvt_f32_ubyte1_e32 v27, v31
	v_cvt_f32_ubyte2_e32 v29, v31
	v_cvt_f32_ubyte3_e32 v30, v30
	v_cvt_f32_ubyte3_e32 v31, v31
	v_fma_mix_f32 v24, v12, v24, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v25, v12, v25, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v26, v12, v26, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v27, v12, v27, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v28, v12, v28, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v29, v12, v29, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v30, v12, v30, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v31, v12, v31, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x15
	v_cndmask_b32_e32 v14, v15, v14, vcc_lo
	s_wait_loadcnt 0x14
	v_and_b32_e32 v40, 0xf0f0f0f, v21
	v_lshrrev_b32_e32 v41, 4, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v41, 0xf0f0f0f, v41
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v34, v40
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v36, v40
	v_cvt_f32_ubyte2_e32 v38, v40
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v35, v41
	v_cvt_f32_ubyte1_e32 v37, v41
	v_cvt_f32_ubyte2_e32 v39, v41
	v_cvt_f32_ubyte3_e32 v40, v40
	v_cvt_f32_ubyte3_e32 v41, v41
	v_fma_mix_f32 v34, v14, v34, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v35, v14, v35, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v36, v14, v36, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v37, v14, v37, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v38, v14, v38, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v39, v14, v39, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v40, v14, v40, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v41, v14, v41, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x13
	v_cndmask_b32_e32 v16, v17, v16, vcc_lo
	s_wait_loadcnt 0x12
	v_and_b32_e32 v50, 0xf0f0f0f, v22
	v_lshrrev_b32_e32 v51, 4, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v51, 0xf0f0f0f, v51
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v44, v50
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v46, v50
	v_cvt_f32_ubyte2_e32 v48, v50
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v45, v51
	v_cvt_f32_ubyte1_e32 v47, v51
	v_cvt_f32_ubyte2_e32 v49, v51
	v_cvt_f32_ubyte3_e32 v50, v50
	v_cvt_f32_ubyte3_e32 v51, v51
	v_fma_mix_f32 v44, v16, v44, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v16, v45, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v16, v46, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v16, v47, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v48, v16, v48, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v49, v16, v49, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v16, v50, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v16, v51, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x11
	v_cndmask_b32_e32 v18, v19, v18, vcc_lo
	s_wait_loadcnt 0x10
	v_and_b32_e32 v60, 0xf0f0f0f, v23
	v_lshrrev_b32_e32 v61, 4, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v61, 0xf0f0f0f, v61
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v54, v60
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v56, v60
	v_cvt_f32_ubyte2_e32 v58, v60
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v55, v61
	v_cvt_f32_ubyte1_e32 v57, v61
	v_cvt_f32_ubyte2_e32 v59, v61
	v_cvt_f32_ubyte3_e32 v60, v60
	v_cvt_f32_ubyte3_e32 v61, v61
	v_fma_mix_f32 v54, v18, v54, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v18, v55, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v18, v56, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v18, v57, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v58, v18, v58, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v59, v18, v59, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v18, v60, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v18, v61, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	buffer_load_b64 v[12:13], v8, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v20, v4, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[14:15], v9, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v21, v5, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[16:17], v10, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v22, v6, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[18:19], v11, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v23, v7, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	s_wait_loadcnt 0x17
	v_dual_mul_f32 v32, v24, v161 :: v_dual_mul_f32 v33, v35, v162
	v_dual_mul_f32 v42, v44, v161 :: v_dual_mul_f32 v43, v55, v162
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v52, v24, v169 :: v_dual_mul_f32 v53, v35, v170
	v_dual_mul_f32 v62, v44, v169 :: v_dual_mul_f32 v63, v55, v170
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v25, v162 :: v_dual_fmac_f32 v33, v34, v161
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v45, v162 :: v_dual_fmac_f32 v43, v54, v161
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v25, v170 :: v_dual_fmac_f32 v53, v34, v169
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v45, v170 :: v_dual_fmac_f32 v63, v54, v169
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v26, v163 :: v_dual_fmac_f32 v33, v36, v163
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v46, v163 :: v_dual_fmac_f32 v43, v56, v163
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v26, v171 :: v_dual_fmac_f32 v53, v36, v171
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v46, v171 :: v_dual_fmac_f32 v63, v56, v171
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v27, v164 :: v_dual_fmac_f32 v33, v37, v164
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v47, v164 :: v_dual_fmac_f32 v43, v57, v164
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v27, v172 :: v_dual_fmac_f32 v53, v37, v172
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v47, v172 :: v_dual_fmac_f32 v63, v57, v172
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v28, v165 :: v_dual_fmac_f32 v33, v38, v165
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v48, v165 :: v_dual_fmac_f32 v43, v58, v165
	s_wait_loadcnt 0x14
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v28, v173 :: v_dual_fmac_f32 v53, v38, v173
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v48, v173 :: v_dual_fmac_f32 v63, v58, v173
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v29, v166 :: v_dual_fmac_f32 v33, v39, v166
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v49, v166 :: v_dual_fmac_f32 v43, v59, v166
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v29, v174 :: v_dual_fmac_f32 v53, v39, v174
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v49, v174 :: v_dual_fmac_f32 v63, v59, v174
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v30, v167 :: v_dual_fmac_f32 v33, v40, v167
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v50, v167 :: v_dual_fmac_f32 v43, v60, v167
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v30, v175 :: v_dual_fmac_f32 v53, v40, v175
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v50, v175 :: v_dual_fmac_f32 v63, v60, v175
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v31, v168 :: v_dual_fmac_f32 v33, v41, v168
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v51, v168 :: v_dual_fmac_f32 v43, v61, v168
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v31, v176 :: v_dual_fmac_f32 v53, v41, v176
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v51, v176 :: v_dual_fmac_f32 v63, v61, v176
	global_load_b128 v[161:164], v3, s[28:29]
	global_load_b128 v[165:168], v3, s[28:29] offset:16
	global_load_b128 v[169:172], v3, s[30:31]
	global_load_b128 v[173:176], v3, s[30:31] offset:16
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v68, v32, v68 :: v_dual_add_f32 v69, v69, v33
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v70, v42, v70 :: v_dual_add_f32 v71, v71, v43
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v80, v52, v80 :: v_dual_add_f32 v81, v81, v53
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v82, v62, v82 :: v_dual_add_f32 v83, v83, v63
	s_wait_loadcnt 0x17
	v_dual_mul_f32 v32, v24, v177 :: v_dual_mul_f32 v33, v35, v178
	v_dual_mul_f32 v42, v44, v177 :: v_dual_mul_f32 v43, v55, v178
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v52, v24, v185 :: v_dual_mul_f32 v53, v35, v186
	v_dual_mul_f32 v62, v44, v185 :: v_dual_mul_f32 v63, v55, v186
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v25, v178 :: v_dual_fmac_f32 v33, v34, v177
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v45, v178 :: v_dual_fmac_f32 v43, v54, v177
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v25, v186 :: v_dual_fmac_f32 v53, v34, v185
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v45, v186 :: v_dual_fmac_f32 v63, v54, v185
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v26, v179 :: v_dual_fmac_f32 v33, v36, v179
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v46, v179 :: v_dual_fmac_f32 v43, v56, v179
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v26, v187 :: v_dual_fmac_f32 v53, v36, v187
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v46, v187 :: v_dual_fmac_f32 v63, v56, v187
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v27, v180 :: v_dual_fmac_f32 v33, v37, v180
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v47, v180 :: v_dual_fmac_f32 v43, v57, v180
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v27, v188 :: v_dual_fmac_f32 v53, v37, v188
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v47, v188 :: v_dual_fmac_f32 v63, v57, v188
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v28, v181 :: v_dual_fmac_f32 v33, v38, v181
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v48, v181 :: v_dual_fmac_f32 v43, v58, v181
	s_wait_loadcnt 0x14
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v28, v189 :: v_dual_fmac_f32 v53, v38, v189
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v48, v189 :: v_dual_fmac_f32 v63, v58, v189
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v29, v182 :: v_dual_fmac_f32 v33, v39, v182
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v49, v182 :: v_dual_fmac_f32 v43, v59, v182
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v29, v190 :: v_dual_fmac_f32 v53, v39, v190
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v49, v190 :: v_dual_fmac_f32 v63, v59, v190
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v30, v183 :: v_dual_fmac_f32 v33, v40, v183
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v50, v183 :: v_dual_fmac_f32 v43, v60, v183
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v30, v191 :: v_dual_fmac_f32 v53, v40, v191
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v50, v191 :: v_dual_fmac_f32 v63, v60, v191
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v31, v184 :: v_dual_fmac_f32 v33, v41, v184
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v51, v184 :: v_dual_fmac_f32 v43, v61, v184
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v31, v192 :: v_dual_fmac_f32 v53, v41, v192
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v51, v192 :: v_dual_fmac_f32 v63, v61, v192
	global_load_b128 v[177:180], v3, s[32:33]
	global_load_b128 v[181:184], v3, s[32:33] offset:16
	global_load_b128 v[185:188], v3, s[34:35]
	global_load_b128 v[189:192], v3, s[34:35] offset:16
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v92, v32, v92 :: v_dual_add_f32 v93, v93, v33
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v94, v42, v94 :: v_dual_add_f32 v95, v95, v43
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v104, v52, v104 :: v_dual_add_f32 v105, v105, v53
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v106, v62, v106 :: v_dual_add_f32 v107, v107, v63
	s_wait_loadcnt 0x17
	v_dual_mul_f32 v32, v24, v193 :: v_dual_mul_f32 v33, v35, v194
	v_dual_mul_f32 v42, v44, v193 :: v_dual_mul_f32 v43, v55, v194
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v52, v24, v201 :: v_dual_mul_f32 v53, v35, v202
	v_dual_mul_f32 v62, v44, v201 :: v_dual_mul_f32 v63, v55, v202
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v25, v194 :: v_dual_fmac_f32 v33, v34, v193
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v45, v194 :: v_dual_fmac_f32 v43, v54, v193
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v25, v202 :: v_dual_fmac_f32 v53, v34, v201
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v45, v202 :: v_dual_fmac_f32 v63, v54, v201
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v26, v195 :: v_dual_fmac_f32 v33, v36, v195
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v46, v195 :: v_dual_fmac_f32 v43, v56, v195
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v26, v203 :: v_dual_fmac_f32 v53, v36, v203
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v46, v203 :: v_dual_fmac_f32 v63, v56, v203
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v27, v196 :: v_dual_fmac_f32 v33, v37, v196
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v47, v196 :: v_dual_fmac_f32 v43, v57, v196
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v27, v204 :: v_dual_fmac_f32 v53, v37, v204
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v47, v204 :: v_dual_fmac_f32 v63, v57, v204
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v28, v197 :: v_dual_fmac_f32 v33, v38, v197
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v48, v197 :: v_dual_fmac_f32 v43, v58, v197
	s_wait_loadcnt 0x14
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v28, v205 :: v_dual_fmac_f32 v53, v38, v205
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v48, v205 :: v_dual_fmac_f32 v63, v58, v205
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v29, v198 :: v_dual_fmac_f32 v33, v39, v198
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v49, v198 :: v_dual_fmac_f32 v43, v59, v198
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v29, v206 :: v_dual_fmac_f32 v53, v39, v206
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v49, v206 :: v_dual_fmac_f32 v63, v59, v206
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v30, v199 :: v_dual_fmac_f32 v33, v40, v199
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v50, v199 :: v_dual_fmac_f32 v43, v60, v199
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v30, v207 :: v_dual_fmac_f32 v53, v40, v207
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v50, v207 :: v_dual_fmac_f32 v63, v60, v207
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v31, v200 :: v_dual_fmac_f32 v33, v41, v200
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v51, v200 :: v_dual_fmac_f32 v43, v61, v200
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v31, v208 :: v_dual_fmac_f32 v53, v41, v208
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v51, v208 :: v_dual_fmac_f32 v63, v61, v208
	global_load_b128 v[193:196], v3, s[36:37]
	global_load_b128 v[197:200], v3, s[36:37] offset:16
	global_load_b128 v[201:204], v3, s[38:39]
	global_load_b128 v[205:208], v3, s[38:39] offset:16
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v116, v32, v116 :: v_dual_add_f32 v117, v117, v33
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v118, v42, v118 :: v_dual_add_f32 v119, v119, v43
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v128, v52, v128 :: v_dual_add_f32 v129, v129, v53
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v130, v62, v130 :: v_dual_add_f32 v131, v131, v63
	s_wait_loadcnt 0x17
	v_dual_mul_f32 v32, v24, v209 :: v_dual_mul_f32 v33, v35, v210
	v_dual_mul_f32 v42, v44, v209 :: v_dual_mul_f32 v43, v55, v210
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v52, v24, v217 :: v_dual_mul_f32 v53, v35, v218
	v_dual_mul_f32 v62, v44, v217 :: v_dual_mul_f32 v63, v55, v218
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v25, v210 :: v_dual_fmac_f32 v33, v34, v209
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v45, v210 :: v_dual_fmac_f32 v43, v54, v209
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v25, v218 :: v_dual_fmac_f32 v53, v34, v217
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v45, v218 :: v_dual_fmac_f32 v63, v54, v217
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v26, v211 :: v_dual_fmac_f32 v33, v36, v211
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v46, v211 :: v_dual_fmac_f32 v43, v56, v211
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v26, v219 :: v_dual_fmac_f32 v53, v36, v219
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v46, v219 :: v_dual_fmac_f32 v63, v56, v219
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v27, v212 :: v_dual_fmac_f32 v33, v37, v212
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v47, v212 :: v_dual_fmac_f32 v43, v57, v212
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v27, v220 :: v_dual_fmac_f32 v53, v37, v220
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v47, v220 :: v_dual_fmac_f32 v63, v57, v220
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v28, v213 :: v_dual_fmac_f32 v33, v38, v213
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v48, v213 :: v_dual_fmac_f32 v43, v58, v213
	s_wait_loadcnt 0x14
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v28, v221 :: v_dual_fmac_f32 v53, v38, v221
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v48, v221 :: v_dual_fmac_f32 v63, v58, v221
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v29, v214 :: v_dual_fmac_f32 v33, v39, v214
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v49, v214 :: v_dual_fmac_f32 v43, v59, v214
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v29, v222 :: v_dual_fmac_f32 v53, v39, v222
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v49, v222 :: v_dual_fmac_f32 v63, v59, v222
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v30, v215 :: v_dual_fmac_f32 v33, v40, v215
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v50, v215 :: v_dual_fmac_f32 v43, v60, v215
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v30, v223 :: v_dual_fmac_f32 v53, v40, v223
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v50, v223 :: v_dual_fmac_f32 v63, v60, v223
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v31, v216 :: v_dual_fmac_f32 v33, v41, v216
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v51, v216 :: v_dual_fmac_f32 v43, v61, v216
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v31, v224 :: v_dual_fmac_f32 v53, v41, v224
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v51, v224 :: v_dual_fmac_f32 v63, v61, v224
	global_load_b128 v[209:212], v3, s[40:41]
	global_load_b128 v[213:216], v3, s[40:41] offset:16
	global_load_b128 v[217:220], v3, s[42:43]
	global_load_b128 v[221:224], v3, s[42:43] offset:16
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v140, v32, v140 :: v_dual_add_f32 v141, v141, v33
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v142, v42, v142 :: v_dual_add_f32 v143, v143, v43
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v152, v52, v152 :: v_dual_add_f32 v153, v153, v53
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v154, v62, v154 :: v_dual_add_f32 v155, v155, v63
	s_mov_b32 s15, s16
	v_mov_b32_e32 v2, v3
	s_add_co_i32 s2, s2, -1
	s_cmp_lg_u32 s2, 0
	s_cbranch_scc1 .Lrx_b8_s1
	s_wait_loadcnt 0x0
	.Lrx_b8_s1_done:
	v_dual_add_f32 v64, v64, v68 :: v_dual_add_f32 v65, v65, v69
	v_dual_add_f32 v66, v66, v70 :: v_dual_add_f32 v67, v67, v71
	v_dual_add_f32 v76, v76, v80 :: v_dual_add_f32 v77, v77, v81
	v_dual_add_f32 v78, v78, v82 :: v_dual_add_f32 v79, v79, v83
	v_dual_add_f32 v88, v88, v92 :: v_dual_add_f32 v89, v89, v93
	v_dual_add_f32 v90, v90, v94 :: v_dual_add_f32 v91, v91, v95
	v_dual_add_f32 v100, v100, v104 :: v_dual_add_f32 v101, v101, v105
	v_dual_add_f32 v102, v102, v106 :: v_dual_add_f32 v103, v103, v107
	v_dual_add_f32 v112, v112, v116 :: v_dual_add_f32 v113, v113, v117
	v_dual_add_f32 v114, v114, v118 :: v_dual_add_f32 v115, v115, v119
	v_dual_add_f32 v124, v124, v128 :: v_dual_add_f32 v125, v125, v129
	v_dual_add_f32 v126, v126, v130 :: v_dual_add_f32 v127, v127, v131
	v_dual_add_f32 v136, v136, v140 :: v_dual_add_f32 v137, v137, v141
	v_dual_add_f32 v138, v138, v142 :: v_dual_add_f32 v139, v139, v143
	v_dual_add_f32 v148, v148, v152 :: v_dual_add_f32 v149, v149, v153
	v_dual_add_f32 v150, v150, v154 :: v_dual_add_f32 v151, v151, v155
	v_mov_b32_e32 v68, 0
	v_mov_b32_e32 v69, 0
	v_mov_b32_e32 v70, 0
	v_mov_b32_e32 v71, 0
	v_mov_b32_e32 v80, 0
	v_mov_b32_e32 v81, 0
	v_mov_b32_e32 v82, 0
	v_mov_b32_e32 v83, 0
	v_mov_b32_e32 v92, 0
	v_mov_b32_e32 v93, 0
	v_mov_b32_e32 v94, 0
	v_mov_b32_e32 v95, 0
	v_mov_b32_e32 v104, 0
	v_mov_b32_e32 v105, 0
	v_mov_b32_e32 v106, 0
	v_mov_b32_e32 v107, 0
	v_mov_b32_e32 v116, 0
	v_mov_b32_e32 v117, 0
	v_mov_b32_e32 v118, 0
	v_mov_b32_e32 v119, 0
	v_mov_b32_e32 v128, 0
	v_mov_b32_e32 v129, 0
	v_mov_b32_e32 v130, 0
	v_mov_b32_e32 v131, 0
	v_mov_b32_e32 v140, 0
	v_mov_b32_e32 v141, 0
	v_mov_b32_e32 v142, 0
	v_mov_b32_e32 v143, 0
	v_mov_b32_e32 v152, 0
	v_mov_b32_e32 v153, 0
	v_mov_b32_e32 v154, 0
	v_mov_b32_e32 v155, 0
	s_add_co_i32 s2, s14, 1
	s_lshr_b32 s2, s2, 2
	s_cmp_eq_u32 s2, 0
	s_cbranch_scc1 .Lrx_b8_s2_done
	s_mov_b32 s15, 0x110
	v_lshlrev_b32_e32 v2, 5, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v2, 0x800, v2
	buffer_load_b64 v[12:13], v8, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v20, v4, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[14:15], v9, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v21, v5, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[16:17], v10, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v22, v6, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[18:19], v11, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v23, v7, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	global_load_b128 v[161:164], v2, s[28:29]
	global_load_b128 v[165:168], v2, s[28:29] offset:16
	global_load_b128 v[169:172], v2, s[30:31]
	global_load_b128 v[173:176], v2, s[30:31] offset:16
	global_load_b128 v[177:180], v2, s[32:33]
	global_load_b128 v[181:184], v2, s[32:33] offset:16
	global_load_b128 v[185:188], v2, s[34:35]
	global_load_b128 v[189:192], v2, s[34:35] offset:16
	global_load_b128 v[193:196], v2, s[36:37]
	global_load_b128 v[197:200], v2, s[36:37] offset:16
	global_load_b128 v[201:204], v2, s[38:39]
	global_load_b128 v[205:208], v2, s[38:39] offset:16
	global_load_b128 v[209:212], v2, s[40:41]
	global_load_b128 v[213:216], v2, s[40:41] offset:16
	global_load_b128 v[217:220], v2, s[42:43]
	global_load_b128 v[221:224], v2, s[42:43] offset:16
	.Lrx_b8_s2:
	s_cmp_gt_u32 s2, 1
	s_cselect_b32 s17, 0x220, 0
	s_cselect_b32 s18, 0x1000, 0
	s_add_co_i32 s16, s15, s17
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v3, s18, v2
	v_cmp_gt_u32_e32 vcc_lo, 16, v0
	s_wait_loadcnt 0x17
	v_cndmask_b32_e32 v12, v13, v12, vcc_lo
	s_wait_loadcnt 0x16
	v_and_b32_e32 v30, 0xf0f0f0f, v20
	v_lshrrev_b32_e32 v31, 4, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v31, 0xf0f0f0f, v31
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v24, v30
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v26, v30
	v_cvt_f32_ubyte2_e32 v28, v30
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v25, v31
	v_cvt_f32_ubyte1_e32 v27, v31
	v_cvt_f32_ubyte2_e32 v29, v31
	v_cvt_f32_ubyte3_e32 v30, v30
	v_cvt_f32_ubyte3_e32 v31, v31
	v_fma_mix_f32 v24, v12, v24, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v25, v12, v25, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v26, v12, v26, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v27, v12, v27, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v28, v12, v28, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v29, v12, v29, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v30, v12, v30, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v31, v12, v31, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x15
	v_cndmask_b32_e32 v14, v15, v14, vcc_lo
	s_wait_loadcnt 0x14
	v_and_b32_e32 v40, 0xf0f0f0f, v21
	v_lshrrev_b32_e32 v41, 4, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v41, 0xf0f0f0f, v41
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v34, v40
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v36, v40
	v_cvt_f32_ubyte2_e32 v38, v40
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v35, v41
	v_cvt_f32_ubyte1_e32 v37, v41
	v_cvt_f32_ubyte2_e32 v39, v41
	v_cvt_f32_ubyte3_e32 v40, v40
	v_cvt_f32_ubyte3_e32 v41, v41
	v_fma_mix_f32 v34, v14, v34, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v35, v14, v35, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v36, v14, v36, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v37, v14, v37, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v38, v14, v38, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v39, v14, v39, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v40, v14, v40, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v41, v14, v41, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x13
	v_cndmask_b32_e32 v16, v17, v16, vcc_lo
	s_wait_loadcnt 0x12
	v_and_b32_e32 v50, 0xf0f0f0f, v22
	v_lshrrev_b32_e32 v51, 4, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v51, 0xf0f0f0f, v51
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v44, v50
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v46, v50
	v_cvt_f32_ubyte2_e32 v48, v50
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v45, v51
	v_cvt_f32_ubyte1_e32 v47, v51
	v_cvt_f32_ubyte2_e32 v49, v51
	v_cvt_f32_ubyte3_e32 v50, v50
	v_cvt_f32_ubyte3_e32 v51, v51
	v_fma_mix_f32 v44, v16, v44, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v16, v45, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v16, v46, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v16, v47, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v48, v16, v48, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v49, v16, v49, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v16, v50, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v16, v51, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x11
	v_cndmask_b32_e32 v18, v19, v18, vcc_lo
	s_wait_loadcnt 0x10
	v_and_b32_e32 v60, 0xf0f0f0f, v23
	v_lshrrev_b32_e32 v61, 4, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v61, 0xf0f0f0f, v61
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v54, v60
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v56, v60
	v_cvt_f32_ubyte2_e32 v58, v60
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v55, v61
	v_cvt_f32_ubyte1_e32 v57, v61
	v_cvt_f32_ubyte2_e32 v59, v61
	v_cvt_f32_ubyte3_e32 v60, v60
	v_cvt_f32_ubyte3_e32 v61, v61
	v_fma_mix_f32 v54, v18, v54, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v18, v55, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v18, v56, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v18, v57, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v58, v18, v58, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v59, v18, v59, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v18, v60, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v18, v61, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	buffer_load_b64 v[12:13], v8, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v20, v4, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[14:15], v9, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v21, v5, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[16:17], v10, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v22, v6, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[18:19], v11, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v23, v7, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	s_wait_loadcnt 0x17
	v_dual_mul_f32 v32, v24, v161 :: v_dual_mul_f32 v33, v35, v162
	v_dual_mul_f32 v42, v44, v161 :: v_dual_mul_f32 v43, v55, v162
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v52, v24, v169 :: v_dual_mul_f32 v53, v35, v170
	v_dual_mul_f32 v62, v44, v169 :: v_dual_mul_f32 v63, v55, v170
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v25, v162 :: v_dual_fmac_f32 v33, v34, v161
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v45, v162 :: v_dual_fmac_f32 v43, v54, v161
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v25, v170 :: v_dual_fmac_f32 v53, v34, v169
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v45, v170 :: v_dual_fmac_f32 v63, v54, v169
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v26, v163 :: v_dual_fmac_f32 v33, v36, v163
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v46, v163 :: v_dual_fmac_f32 v43, v56, v163
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v26, v171 :: v_dual_fmac_f32 v53, v36, v171
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v46, v171 :: v_dual_fmac_f32 v63, v56, v171
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v27, v164 :: v_dual_fmac_f32 v33, v37, v164
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v47, v164 :: v_dual_fmac_f32 v43, v57, v164
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v27, v172 :: v_dual_fmac_f32 v53, v37, v172
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v47, v172 :: v_dual_fmac_f32 v63, v57, v172
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v28, v165 :: v_dual_fmac_f32 v33, v38, v165
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v48, v165 :: v_dual_fmac_f32 v43, v58, v165
	s_wait_loadcnt 0x14
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v28, v173 :: v_dual_fmac_f32 v53, v38, v173
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v48, v173 :: v_dual_fmac_f32 v63, v58, v173
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v29, v166 :: v_dual_fmac_f32 v33, v39, v166
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v49, v166 :: v_dual_fmac_f32 v43, v59, v166
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v29, v174 :: v_dual_fmac_f32 v53, v39, v174
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v49, v174 :: v_dual_fmac_f32 v63, v59, v174
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v30, v167 :: v_dual_fmac_f32 v33, v40, v167
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v50, v167 :: v_dual_fmac_f32 v43, v60, v167
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v30, v175 :: v_dual_fmac_f32 v53, v40, v175
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v50, v175 :: v_dual_fmac_f32 v63, v60, v175
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v31, v168 :: v_dual_fmac_f32 v33, v41, v168
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v51, v168 :: v_dual_fmac_f32 v43, v61, v168
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v31, v176 :: v_dual_fmac_f32 v53, v41, v176
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v51, v176 :: v_dual_fmac_f32 v63, v61, v176
	global_load_b128 v[161:164], v3, s[28:29]
	global_load_b128 v[165:168], v3, s[28:29] offset:16
	global_load_b128 v[169:172], v3, s[30:31]
	global_load_b128 v[173:176], v3, s[30:31] offset:16
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v68, v32, v68 :: v_dual_add_f32 v69, v69, v33
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v70, v42, v70 :: v_dual_add_f32 v71, v71, v43
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v80, v52, v80 :: v_dual_add_f32 v81, v81, v53
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v82, v62, v82 :: v_dual_add_f32 v83, v83, v63
	s_wait_loadcnt 0x17
	v_dual_mul_f32 v32, v24, v177 :: v_dual_mul_f32 v33, v35, v178
	v_dual_mul_f32 v42, v44, v177 :: v_dual_mul_f32 v43, v55, v178
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v52, v24, v185 :: v_dual_mul_f32 v53, v35, v186
	v_dual_mul_f32 v62, v44, v185 :: v_dual_mul_f32 v63, v55, v186
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v25, v178 :: v_dual_fmac_f32 v33, v34, v177
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v45, v178 :: v_dual_fmac_f32 v43, v54, v177
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v25, v186 :: v_dual_fmac_f32 v53, v34, v185
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v45, v186 :: v_dual_fmac_f32 v63, v54, v185
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v26, v179 :: v_dual_fmac_f32 v33, v36, v179
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v46, v179 :: v_dual_fmac_f32 v43, v56, v179
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v26, v187 :: v_dual_fmac_f32 v53, v36, v187
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v46, v187 :: v_dual_fmac_f32 v63, v56, v187
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v27, v180 :: v_dual_fmac_f32 v33, v37, v180
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v47, v180 :: v_dual_fmac_f32 v43, v57, v180
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v27, v188 :: v_dual_fmac_f32 v53, v37, v188
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v47, v188 :: v_dual_fmac_f32 v63, v57, v188
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v28, v181 :: v_dual_fmac_f32 v33, v38, v181
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v48, v181 :: v_dual_fmac_f32 v43, v58, v181
	s_wait_loadcnt 0x14
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v28, v189 :: v_dual_fmac_f32 v53, v38, v189
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v48, v189 :: v_dual_fmac_f32 v63, v58, v189
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v29, v182 :: v_dual_fmac_f32 v33, v39, v182
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v49, v182 :: v_dual_fmac_f32 v43, v59, v182
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v29, v190 :: v_dual_fmac_f32 v53, v39, v190
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v49, v190 :: v_dual_fmac_f32 v63, v59, v190
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v30, v183 :: v_dual_fmac_f32 v33, v40, v183
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v50, v183 :: v_dual_fmac_f32 v43, v60, v183
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v30, v191 :: v_dual_fmac_f32 v53, v40, v191
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v50, v191 :: v_dual_fmac_f32 v63, v60, v191
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v31, v184 :: v_dual_fmac_f32 v33, v41, v184
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v51, v184 :: v_dual_fmac_f32 v43, v61, v184
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v31, v192 :: v_dual_fmac_f32 v53, v41, v192
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v51, v192 :: v_dual_fmac_f32 v63, v61, v192
	global_load_b128 v[177:180], v3, s[32:33]
	global_load_b128 v[181:184], v3, s[32:33] offset:16
	global_load_b128 v[185:188], v3, s[34:35]
	global_load_b128 v[189:192], v3, s[34:35] offset:16
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v92, v32, v92 :: v_dual_add_f32 v93, v93, v33
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v94, v42, v94 :: v_dual_add_f32 v95, v95, v43
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v104, v52, v104 :: v_dual_add_f32 v105, v105, v53
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v106, v62, v106 :: v_dual_add_f32 v107, v107, v63
	s_wait_loadcnt 0x17
	v_dual_mul_f32 v32, v24, v193 :: v_dual_mul_f32 v33, v35, v194
	v_dual_mul_f32 v42, v44, v193 :: v_dual_mul_f32 v43, v55, v194
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v52, v24, v201 :: v_dual_mul_f32 v53, v35, v202
	v_dual_mul_f32 v62, v44, v201 :: v_dual_mul_f32 v63, v55, v202
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v25, v194 :: v_dual_fmac_f32 v33, v34, v193
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v45, v194 :: v_dual_fmac_f32 v43, v54, v193
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v25, v202 :: v_dual_fmac_f32 v53, v34, v201
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v45, v202 :: v_dual_fmac_f32 v63, v54, v201
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v26, v195 :: v_dual_fmac_f32 v33, v36, v195
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v46, v195 :: v_dual_fmac_f32 v43, v56, v195
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v26, v203 :: v_dual_fmac_f32 v53, v36, v203
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v46, v203 :: v_dual_fmac_f32 v63, v56, v203
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v27, v196 :: v_dual_fmac_f32 v33, v37, v196
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v47, v196 :: v_dual_fmac_f32 v43, v57, v196
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v27, v204 :: v_dual_fmac_f32 v53, v37, v204
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v47, v204 :: v_dual_fmac_f32 v63, v57, v204
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v28, v197 :: v_dual_fmac_f32 v33, v38, v197
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v48, v197 :: v_dual_fmac_f32 v43, v58, v197
	s_wait_loadcnt 0x14
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v28, v205 :: v_dual_fmac_f32 v53, v38, v205
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v48, v205 :: v_dual_fmac_f32 v63, v58, v205
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v29, v198 :: v_dual_fmac_f32 v33, v39, v198
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v49, v198 :: v_dual_fmac_f32 v43, v59, v198
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v29, v206 :: v_dual_fmac_f32 v53, v39, v206
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v49, v206 :: v_dual_fmac_f32 v63, v59, v206
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v30, v199 :: v_dual_fmac_f32 v33, v40, v199
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v50, v199 :: v_dual_fmac_f32 v43, v60, v199
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v30, v207 :: v_dual_fmac_f32 v53, v40, v207
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v50, v207 :: v_dual_fmac_f32 v63, v60, v207
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v31, v200 :: v_dual_fmac_f32 v33, v41, v200
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v51, v200 :: v_dual_fmac_f32 v43, v61, v200
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v31, v208 :: v_dual_fmac_f32 v53, v41, v208
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v51, v208 :: v_dual_fmac_f32 v63, v61, v208
	global_load_b128 v[193:196], v3, s[36:37]
	global_load_b128 v[197:200], v3, s[36:37] offset:16
	global_load_b128 v[201:204], v3, s[38:39]
	global_load_b128 v[205:208], v3, s[38:39] offset:16
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v116, v32, v116 :: v_dual_add_f32 v117, v117, v33
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v118, v42, v118 :: v_dual_add_f32 v119, v119, v43
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v128, v52, v128 :: v_dual_add_f32 v129, v129, v53
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v130, v62, v130 :: v_dual_add_f32 v131, v131, v63
	s_wait_loadcnt 0x17
	v_dual_mul_f32 v32, v24, v209 :: v_dual_mul_f32 v33, v35, v210
	v_dual_mul_f32 v42, v44, v209 :: v_dual_mul_f32 v43, v55, v210
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v52, v24, v217 :: v_dual_mul_f32 v53, v35, v218
	v_dual_mul_f32 v62, v44, v217 :: v_dual_mul_f32 v63, v55, v218
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v25, v210 :: v_dual_fmac_f32 v33, v34, v209
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v45, v210 :: v_dual_fmac_f32 v43, v54, v209
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v25, v218 :: v_dual_fmac_f32 v53, v34, v217
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v45, v218 :: v_dual_fmac_f32 v63, v54, v217
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v26, v211 :: v_dual_fmac_f32 v33, v36, v211
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v46, v211 :: v_dual_fmac_f32 v43, v56, v211
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v26, v219 :: v_dual_fmac_f32 v53, v36, v219
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v46, v219 :: v_dual_fmac_f32 v63, v56, v219
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v27, v212 :: v_dual_fmac_f32 v33, v37, v212
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v47, v212 :: v_dual_fmac_f32 v43, v57, v212
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v27, v220 :: v_dual_fmac_f32 v53, v37, v220
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v47, v220 :: v_dual_fmac_f32 v63, v57, v220
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v28, v213 :: v_dual_fmac_f32 v33, v38, v213
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v48, v213 :: v_dual_fmac_f32 v43, v58, v213
	s_wait_loadcnt 0x14
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v28, v221 :: v_dual_fmac_f32 v53, v38, v221
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v48, v221 :: v_dual_fmac_f32 v63, v58, v221
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v29, v214 :: v_dual_fmac_f32 v33, v39, v214
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v49, v214 :: v_dual_fmac_f32 v43, v59, v214
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v29, v222 :: v_dual_fmac_f32 v53, v39, v222
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v49, v222 :: v_dual_fmac_f32 v63, v59, v222
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v30, v215 :: v_dual_fmac_f32 v33, v40, v215
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v50, v215 :: v_dual_fmac_f32 v43, v60, v215
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v30, v223 :: v_dual_fmac_f32 v53, v40, v223
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v50, v223 :: v_dual_fmac_f32 v63, v60, v223
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v31, v216 :: v_dual_fmac_f32 v33, v41, v216
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v51, v216 :: v_dual_fmac_f32 v43, v61, v216
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v31, v224 :: v_dual_fmac_f32 v53, v41, v224
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v51, v224 :: v_dual_fmac_f32 v63, v61, v224
	global_load_b128 v[209:212], v3, s[40:41]
	global_load_b128 v[213:216], v3, s[40:41] offset:16
	global_load_b128 v[217:220], v3, s[42:43]
	global_load_b128 v[221:224], v3, s[42:43] offset:16
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v140, v32, v140 :: v_dual_add_f32 v141, v141, v33
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v142, v42, v142 :: v_dual_add_f32 v143, v143, v43
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v152, v52, v152 :: v_dual_add_f32 v153, v153, v53
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v154, v62, v154 :: v_dual_add_f32 v155, v155, v63
	s_mov_b32 s15, s16
	v_mov_b32_e32 v2, v3
	s_add_co_i32 s2, s2, -1
	s_cmp_lg_u32 s2, 0
	s_cbranch_scc1 .Lrx_b8_s2
	s_wait_loadcnt 0x0
	.Lrx_b8_s2_done:
	s_add_co_i32 s2, s14, 0
	s_lshr_b32 s2, s2, 2
	s_cmp_eq_u32 s2, 0
	s_cbranch_scc1 .Lrx_b8_s3_done
	s_mov_b32 s15, 0x198
	v_lshlrev_b32_e32 v2, 5, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v2, 0xc00, v2
	buffer_load_b64 v[12:13], v8, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v20, v4, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[14:15], v9, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v21, v5, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[16:17], v10, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v22, v6, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[18:19], v11, s[20:23], s15 offen scope:SCOPE_DEV
	buffer_load_b32 v23, v7, s[20:23], s15 offen offset:8 scope:SCOPE_DEV
	global_load_b128 v[161:164], v2, s[28:29]
	global_load_b128 v[165:168], v2, s[28:29] offset:16
	global_load_b128 v[169:172], v2, s[30:31]
	global_load_b128 v[173:176], v2, s[30:31] offset:16
	global_load_b128 v[177:180], v2, s[32:33]
	global_load_b128 v[181:184], v2, s[32:33] offset:16
	global_load_b128 v[185:188], v2, s[34:35]
	global_load_b128 v[189:192], v2, s[34:35] offset:16
	global_load_b128 v[193:196], v2, s[36:37]
	global_load_b128 v[197:200], v2, s[36:37] offset:16
	global_load_b128 v[201:204], v2, s[38:39]
	global_load_b128 v[205:208], v2, s[38:39] offset:16
	global_load_b128 v[209:212], v2, s[40:41]
	global_load_b128 v[213:216], v2, s[40:41] offset:16
	global_load_b128 v[217:220], v2, s[42:43]
	global_load_b128 v[221:224], v2, s[42:43] offset:16
	.Lrx_b8_s3:
	s_cmp_gt_u32 s2, 1
	s_cselect_b32 s17, 0x220, 0
	s_cselect_b32 s18, 0x1000, 0
	s_add_co_i32 s16, s15, s17
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v3, s18, v2
	v_cmp_gt_u32_e32 vcc_lo, 16, v0
	s_wait_loadcnt 0x17
	v_cndmask_b32_e32 v12, v13, v12, vcc_lo
	s_wait_loadcnt 0x16
	v_and_b32_e32 v30, 0xf0f0f0f, v20
	v_lshrrev_b32_e32 v31, 4, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v31, 0xf0f0f0f, v31
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v24, v30
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v26, v30
	v_cvt_f32_ubyte2_e32 v28, v30
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v25, v31
	v_cvt_f32_ubyte1_e32 v27, v31
	v_cvt_f32_ubyte2_e32 v29, v31
	v_cvt_f32_ubyte3_e32 v30, v30
	v_cvt_f32_ubyte3_e32 v31, v31
	v_fma_mix_f32 v24, v12, v24, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v25, v12, v25, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v26, v12, v26, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v27, v12, v27, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v28, v12, v28, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v29, v12, v29, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v30, v12, v30, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v31, v12, v31, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x15
	v_cndmask_b32_e32 v14, v15, v14, vcc_lo
	s_wait_loadcnt 0x14
	v_and_b32_e32 v40, 0xf0f0f0f, v21
	v_lshrrev_b32_e32 v41, 4, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v41, 0xf0f0f0f, v41
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v34, v40
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v36, v40
	v_cvt_f32_ubyte2_e32 v38, v40
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v35, v41
	v_cvt_f32_ubyte1_e32 v37, v41
	v_cvt_f32_ubyte2_e32 v39, v41
	v_cvt_f32_ubyte3_e32 v40, v40
	v_cvt_f32_ubyte3_e32 v41, v41
	v_fma_mix_f32 v34, v14, v34, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v35, v14, v35, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v36, v14, v36, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v37, v14, v37, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v38, v14, v38, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v39, v14, v39, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v40, v14, v40, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v41, v14, v41, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x13
	v_cndmask_b32_e32 v16, v17, v16, vcc_lo
	s_wait_loadcnt 0x12
	v_and_b32_e32 v50, 0xf0f0f0f, v22
	v_lshrrev_b32_e32 v51, 4, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v51, 0xf0f0f0f, v51
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v44, v50
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v46, v50
	v_cvt_f32_ubyte2_e32 v48, v50
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v45, v51
	v_cvt_f32_ubyte1_e32 v47, v51
	v_cvt_f32_ubyte2_e32 v49, v51
	v_cvt_f32_ubyte3_e32 v50, v50
	v_cvt_f32_ubyte3_e32 v51, v51
	v_fma_mix_f32 v44, v16, v44, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v16, v45, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v16, v46, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v16, v47, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v48, v16, v48, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v49, v16, v49, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v16, v50, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v16, v51, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x11
	v_cndmask_b32_e32 v18, v19, v18, vcc_lo
	s_wait_loadcnt 0x10
	v_and_b32_e32 v60, 0xf0f0f0f, v23
	v_lshrrev_b32_e32 v61, 4, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v61, 0xf0f0f0f, v61
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f32_ubyte0_e32 v54, v60
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte1_e32 v56, v60
	v_cvt_f32_ubyte2_e32 v58, v60
	s_delay_alu instid0(VALU_DEP_4)
	v_cvt_f32_ubyte0_e32 v55, v61
	v_cvt_f32_ubyte1_e32 v57, v61
	v_cvt_f32_ubyte2_e32 v59, v61
	v_cvt_f32_ubyte3_e32 v60, v60
	v_cvt_f32_ubyte3_e32 v61, v61
	v_fma_mix_f32 v54, v18, v54, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v18, v55, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v18, v56, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v18, v57, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v58, v18, v58, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v59, v18, v59, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v18, v60, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v18, v61, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	buffer_load_b64 v[12:13], v8, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v20, v4, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[14:15], v9, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v21, v5, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[16:17], v10, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v22, v6, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	buffer_load_b64 v[18:19], v11, s[20:23], s16 offen scope:SCOPE_DEV
	buffer_load_b32 v23, v7, s[20:23], s16 offen offset:8 scope:SCOPE_DEV
	s_wait_loadcnt 0x17
	v_dual_mul_f32 v32, v24, v161 :: v_dual_mul_f32 v33, v35, v162
	v_dual_mul_f32 v42, v44, v161 :: v_dual_mul_f32 v43, v55, v162
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v52, v24, v169 :: v_dual_mul_f32 v53, v35, v170
	v_dual_mul_f32 v62, v44, v169 :: v_dual_mul_f32 v63, v55, v170
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v25, v162 :: v_dual_fmac_f32 v33, v34, v161
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v45, v162 :: v_dual_fmac_f32 v43, v54, v161
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v25, v170 :: v_dual_fmac_f32 v53, v34, v169
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v45, v170 :: v_dual_fmac_f32 v63, v54, v169
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v26, v163 :: v_dual_fmac_f32 v33, v36, v163
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v46, v163 :: v_dual_fmac_f32 v43, v56, v163
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v26, v171 :: v_dual_fmac_f32 v53, v36, v171
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v46, v171 :: v_dual_fmac_f32 v63, v56, v171
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v27, v164 :: v_dual_fmac_f32 v33, v37, v164
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v47, v164 :: v_dual_fmac_f32 v43, v57, v164
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v27, v172 :: v_dual_fmac_f32 v53, v37, v172
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v47, v172 :: v_dual_fmac_f32 v63, v57, v172
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v28, v165 :: v_dual_fmac_f32 v33, v38, v165
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v48, v165 :: v_dual_fmac_f32 v43, v58, v165
	s_wait_loadcnt 0x14
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v28, v173 :: v_dual_fmac_f32 v53, v38, v173
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v48, v173 :: v_dual_fmac_f32 v63, v58, v173
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v29, v166 :: v_dual_fmac_f32 v33, v39, v166
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v49, v166 :: v_dual_fmac_f32 v43, v59, v166
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v29, v174 :: v_dual_fmac_f32 v53, v39, v174
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v49, v174 :: v_dual_fmac_f32 v63, v59, v174
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v30, v167 :: v_dual_fmac_f32 v33, v40, v167
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v50, v167 :: v_dual_fmac_f32 v43, v60, v167
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v30, v175 :: v_dual_fmac_f32 v53, v40, v175
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v50, v175 :: v_dual_fmac_f32 v63, v60, v175
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v31, v168 :: v_dual_fmac_f32 v33, v41, v168
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v51, v168 :: v_dual_fmac_f32 v43, v61, v168
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v31, v176 :: v_dual_fmac_f32 v53, v41, v176
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v51, v176 :: v_dual_fmac_f32 v63, v61, v176
	global_load_b128 v[161:164], v3, s[28:29]
	global_load_b128 v[165:168], v3, s[28:29] offset:16
	global_load_b128 v[169:172], v3, s[30:31]
	global_load_b128 v[173:176], v3, s[30:31] offset:16
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v72, v32, v72 :: v_dual_add_f32 v73, v73, v33
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v74, v42, v74 :: v_dual_add_f32 v75, v75, v43
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v84, v52, v84 :: v_dual_add_f32 v85, v85, v53
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v86, v62, v86 :: v_dual_add_f32 v87, v87, v63
	s_wait_loadcnt 0x17
	v_dual_mul_f32 v32, v24, v177 :: v_dual_mul_f32 v33, v35, v178
	v_dual_mul_f32 v42, v44, v177 :: v_dual_mul_f32 v43, v55, v178
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v52, v24, v185 :: v_dual_mul_f32 v53, v35, v186
	v_dual_mul_f32 v62, v44, v185 :: v_dual_mul_f32 v63, v55, v186
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v25, v178 :: v_dual_fmac_f32 v33, v34, v177
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v45, v178 :: v_dual_fmac_f32 v43, v54, v177
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v25, v186 :: v_dual_fmac_f32 v53, v34, v185
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v45, v186 :: v_dual_fmac_f32 v63, v54, v185
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v26, v179 :: v_dual_fmac_f32 v33, v36, v179
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v46, v179 :: v_dual_fmac_f32 v43, v56, v179
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v26, v187 :: v_dual_fmac_f32 v53, v36, v187
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v46, v187 :: v_dual_fmac_f32 v63, v56, v187
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v27, v180 :: v_dual_fmac_f32 v33, v37, v180
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v47, v180 :: v_dual_fmac_f32 v43, v57, v180
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v27, v188 :: v_dual_fmac_f32 v53, v37, v188
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v47, v188 :: v_dual_fmac_f32 v63, v57, v188
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v28, v181 :: v_dual_fmac_f32 v33, v38, v181
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v48, v181 :: v_dual_fmac_f32 v43, v58, v181
	s_wait_loadcnt 0x14
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v28, v189 :: v_dual_fmac_f32 v53, v38, v189
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v48, v189 :: v_dual_fmac_f32 v63, v58, v189
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v29, v182 :: v_dual_fmac_f32 v33, v39, v182
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v49, v182 :: v_dual_fmac_f32 v43, v59, v182
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v29, v190 :: v_dual_fmac_f32 v53, v39, v190
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v49, v190 :: v_dual_fmac_f32 v63, v59, v190
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v30, v183 :: v_dual_fmac_f32 v33, v40, v183
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v50, v183 :: v_dual_fmac_f32 v43, v60, v183
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v30, v191 :: v_dual_fmac_f32 v53, v40, v191
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v50, v191 :: v_dual_fmac_f32 v63, v60, v191
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v31, v184 :: v_dual_fmac_f32 v33, v41, v184
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v51, v184 :: v_dual_fmac_f32 v43, v61, v184
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v31, v192 :: v_dual_fmac_f32 v53, v41, v192
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v51, v192 :: v_dual_fmac_f32 v63, v61, v192
	global_load_b128 v[177:180], v3, s[32:33]
	global_load_b128 v[181:184], v3, s[32:33] offset:16
	global_load_b128 v[185:188], v3, s[34:35]
	global_load_b128 v[189:192], v3, s[34:35] offset:16
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v96, v32, v96 :: v_dual_add_f32 v97, v97, v33
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v98, v42, v98 :: v_dual_add_f32 v99, v99, v43
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v108, v52, v108 :: v_dual_add_f32 v109, v109, v53
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v110, v62, v110 :: v_dual_add_f32 v111, v111, v63
	s_wait_loadcnt 0x17
	v_dual_mul_f32 v32, v24, v193 :: v_dual_mul_f32 v33, v35, v194
	v_dual_mul_f32 v42, v44, v193 :: v_dual_mul_f32 v43, v55, v194
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v52, v24, v201 :: v_dual_mul_f32 v53, v35, v202
	v_dual_mul_f32 v62, v44, v201 :: v_dual_mul_f32 v63, v55, v202
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v25, v194 :: v_dual_fmac_f32 v33, v34, v193
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v45, v194 :: v_dual_fmac_f32 v43, v54, v193
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v25, v202 :: v_dual_fmac_f32 v53, v34, v201
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v45, v202 :: v_dual_fmac_f32 v63, v54, v201
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v26, v195 :: v_dual_fmac_f32 v33, v36, v195
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v46, v195 :: v_dual_fmac_f32 v43, v56, v195
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v26, v203 :: v_dual_fmac_f32 v53, v36, v203
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v46, v203 :: v_dual_fmac_f32 v63, v56, v203
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v27, v196 :: v_dual_fmac_f32 v33, v37, v196
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v47, v196 :: v_dual_fmac_f32 v43, v57, v196
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v27, v204 :: v_dual_fmac_f32 v53, v37, v204
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v47, v204 :: v_dual_fmac_f32 v63, v57, v204
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v28, v197 :: v_dual_fmac_f32 v33, v38, v197
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v48, v197 :: v_dual_fmac_f32 v43, v58, v197
	s_wait_loadcnt 0x14
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v28, v205 :: v_dual_fmac_f32 v53, v38, v205
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v48, v205 :: v_dual_fmac_f32 v63, v58, v205
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v29, v198 :: v_dual_fmac_f32 v33, v39, v198
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v49, v198 :: v_dual_fmac_f32 v43, v59, v198
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v29, v206 :: v_dual_fmac_f32 v53, v39, v206
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v49, v206 :: v_dual_fmac_f32 v63, v59, v206
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v30, v199 :: v_dual_fmac_f32 v33, v40, v199
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v50, v199 :: v_dual_fmac_f32 v43, v60, v199
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v30, v207 :: v_dual_fmac_f32 v53, v40, v207
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v50, v207 :: v_dual_fmac_f32 v63, v60, v207
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v31, v200 :: v_dual_fmac_f32 v33, v41, v200
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v51, v200 :: v_dual_fmac_f32 v43, v61, v200
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v31, v208 :: v_dual_fmac_f32 v53, v41, v208
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v51, v208 :: v_dual_fmac_f32 v63, v61, v208
	global_load_b128 v[193:196], v3, s[36:37]
	global_load_b128 v[197:200], v3, s[36:37] offset:16
	global_load_b128 v[201:204], v3, s[38:39]
	global_load_b128 v[205:208], v3, s[38:39] offset:16
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v120, v32, v120 :: v_dual_add_f32 v121, v121, v33
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v122, v42, v122 :: v_dual_add_f32 v123, v123, v43
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v132, v52, v132 :: v_dual_add_f32 v133, v133, v53
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v134, v62, v134 :: v_dual_add_f32 v135, v135, v63
	s_wait_loadcnt 0x17
	v_dual_mul_f32 v32, v24, v209 :: v_dual_mul_f32 v33, v35, v210
	v_dual_mul_f32 v42, v44, v209 :: v_dual_mul_f32 v43, v55, v210
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v52, v24, v217 :: v_dual_mul_f32 v53, v35, v218
	v_dual_mul_f32 v62, v44, v217 :: v_dual_mul_f32 v63, v55, v218
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v25, v210 :: v_dual_fmac_f32 v33, v34, v209
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v45, v210 :: v_dual_fmac_f32 v43, v54, v209
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v25, v218 :: v_dual_fmac_f32 v53, v34, v217
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v45, v218 :: v_dual_fmac_f32 v63, v54, v217
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v26, v211 :: v_dual_fmac_f32 v33, v36, v211
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v46, v211 :: v_dual_fmac_f32 v43, v56, v211
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v26, v219 :: v_dual_fmac_f32 v53, v36, v219
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v46, v219 :: v_dual_fmac_f32 v63, v56, v219
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v27, v212 :: v_dual_fmac_f32 v33, v37, v212
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v47, v212 :: v_dual_fmac_f32 v43, v57, v212
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v27, v220 :: v_dual_fmac_f32 v53, v37, v220
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v47, v220 :: v_dual_fmac_f32 v63, v57, v220
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v28, v213 :: v_dual_fmac_f32 v33, v38, v213
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v48, v213 :: v_dual_fmac_f32 v43, v58, v213
	s_wait_loadcnt 0x14
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v28, v221 :: v_dual_fmac_f32 v53, v38, v221
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v48, v221 :: v_dual_fmac_f32 v63, v58, v221
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v29, v214 :: v_dual_fmac_f32 v33, v39, v214
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v49, v214 :: v_dual_fmac_f32 v43, v59, v214
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v29, v222 :: v_dual_fmac_f32 v53, v39, v222
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v49, v222 :: v_dual_fmac_f32 v63, v59, v222
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v30, v215 :: v_dual_fmac_f32 v33, v40, v215
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v50, v215 :: v_dual_fmac_f32 v43, v60, v215
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v30, v223 :: v_dual_fmac_f32 v53, v40, v223
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v50, v223 :: v_dual_fmac_f32 v63, v60, v223
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v32, v31, v216 :: v_dual_fmac_f32 v33, v41, v216
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v42, v51, v216 :: v_dual_fmac_f32 v43, v61, v216
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v52, v31, v224 :: v_dual_fmac_f32 v53, v41, v224
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v62, v51, v224 :: v_dual_fmac_f32 v63, v61, v224
	global_load_b128 v[209:212], v3, s[40:41]
	global_load_b128 v[213:216], v3, s[40:41] offset:16
	global_load_b128 v[217:220], v3, s[42:43]
	global_load_b128 v[221:224], v3, s[42:43] offset:16
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v144, v32, v144 :: v_dual_add_f32 v145, v145, v33
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v146, v42, v146 :: v_dual_add_f32 v147, v147, v43
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v156, v52, v156 :: v_dual_add_f32 v157, v157, v53
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_add_f32 v158, v62, v158 :: v_dual_add_f32 v159, v159, v63
	s_mov_b32 s15, s16
	v_mov_b32_e32 v2, v3
	s_add_co_i32 s2, s2, -1
	s_cmp_lg_u32 s2, 0
	s_cbranch_scc1 .Lrx_b8_s3
	s_wait_loadcnt 0x0
	.Lrx_b8_s3_done:
	v_dual_add_f32 v68, v68, v72 :: v_dual_add_f32 v69, v69, v73
	v_dual_add_f32 v70, v70, v74 :: v_dual_add_f32 v71, v71, v75
	v_dual_add_f32 v80, v80, v84 :: v_dual_add_f32 v81, v81, v85
	v_dual_add_f32 v82, v82, v86 :: v_dual_add_f32 v83, v83, v87
	v_dual_add_f32 v92, v92, v96 :: v_dual_add_f32 v93, v93, v97
	v_dual_add_f32 v94, v94, v98 :: v_dual_add_f32 v95, v95, v99
	v_dual_add_f32 v104, v104, v108 :: v_dual_add_f32 v105, v105, v109
	v_dual_add_f32 v106, v106, v110 :: v_dual_add_f32 v107, v107, v111
	v_dual_add_f32 v116, v116, v120 :: v_dual_add_f32 v117, v117, v121
	v_dual_add_f32 v118, v118, v122 :: v_dual_add_f32 v119, v119, v123
	v_dual_add_f32 v128, v128, v132 :: v_dual_add_f32 v129, v129, v133
	v_dual_add_f32 v130, v130, v134 :: v_dual_add_f32 v131, v131, v135
	v_dual_add_f32 v140, v140, v144 :: v_dual_add_f32 v141, v141, v145
	v_dual_add_f32 v142, v142, v146 :: v_dual_add_f32 v143, v143, v147
	v_dual_add_f32 v152, v152, v156 :: v_dual_add_f32 v153, v153, v157
	v_dual_add_f32 v154, v154, v158 :: v_dual_add_f32 v155, v155, v159
	v_dual_add_f32 v64, v64, v68 :: v_dual_add_f32 v65, v65, v69
	v_dual_add_f32 v66, v66, v70 :: v_dual_add_f32 v67, v67, v71
	v_dual_add_f32 v76, v76, v80 :: v_dual_add_f32 v77, v77, v81
	v_dual_add_f32 v78, v78, v82 :: v_dual_add_f32 v79, v79, v83
	v_dual_add_f32 v88, v88, v92 :: v_dual_add_f32 v89, v89, v93
	v_dual_add_f32 v90, v90, v94 :: v_dual_add_f32 v91, v91, v95
	v_dual_add_f32 v100, v100, v104 :: v_dual_add_f32 v101, v101, v105
	v_dual_add_f32 v102, v102, v106 :: v_dual_add_f32 v103, v103, v107
	v_dual_add_f32 v112, v112, v116 :: v_dual_add_f32 v113, v113, v117
	v_dual_add_f32 v114, v114, v118 :: v_dual_add_f32 v115, v115, v119
	v_dual_add_f32 v124, v124, v128 :: v_dual_add_f32 v125, v125, v129
	v_dual_add_f32 v126, v126, v130 :: v_dual_add_f32 v127, v127, v131
	v_dual_add_f32 v136, v136, v140 :: v_dual_add_f32 v137, v137, v141
	v_dual_add_f32 v138, v138, v142 :: v_dual_add_f32 v139, v139, v143
	v_dual_add_f32 v148, v148, v152 :: v_dual_add_f32 v149, v149, v153
	v_dual_add_f32 v150, v150, v154 :: v_dual_add_f32 v151, v151, v155
	ds_swizzle_b32 v24, v64 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v25, v65 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v26, v66 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v27, v67 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v28, v76 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v29, v77 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v30, v78 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v31, v79 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v32, v88 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v33, v89 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v34, v90 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v35, v91 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v36, v100 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v37, v101 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v38, v102 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v39, v103 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v40, v112 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v41, v113 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v42, v114 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v43, v115 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v44, v124 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v45, v125 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v46, v126 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v47, v127 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v48, v136 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v49, v137 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v50, v138 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v51, v139 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v52, v148 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v53, v149 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v54, v150 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v55, v151 offset:swizzle(BITMASK_PERM,"1pppp")
	s_wait_dscnt 0x1f
	v_add_f32_e32 v64, v64, v24
	s_wait_dscnt 0x1e
	v_add_f32_e32 v65, v65, v25
	s_wait_dscnt 0x1d
	v_add_f32_e32 v66, v66, v26
	s_wait_dscnt 0x1c
	v_add_f32_e32 v67, v67, v27
	s_wait_dscnt 0x1b
	v_add_f32_e32 v76, v76, v28
	s_wait_dscnt 0x1a
	v_add_f32_e32 v77, v77, v29
	s_wait_dscnt 0x19
	v_add_f32_e32 v78, v78, v30
	s_wait_dscnt 0x18
	v_add_f32_e32 v79, v79, v31
	s_wait_dscnt 0x17
	v_add_f32_e32 v88, v88, v32
	s_wait_dscnt 0x16
	v_add_f32_e32 v89, v89, v33
	s_wait_dscnt 0x15
	v_add_f32_e32 v90, v90, v34
	s_wait_dscnt 0x14
	v_add_f32_e32 v91, v91, v35
	s_wait_dscnt 0x13
	v_add_f32_e32 v100, v100, v36
	s_wait_dscnt 0x12
	v_add_f32_e32 v101, v101, v37
	s_wait_dscnt 0x11
	v_add_f32_e32 v102, v102, v38
	s_wait_dscnt 0x10
	v_add_f32_e32 v103, v103, v39
	s_wait_dscnt 0xf
	v_add_f32_e32 v112, v112, v40
	s_wait_dscnt 0xe
	v_add_f32_e32 v113, v113, v41
	s_wait_dscnt 0xd
	v_add_f32_e32 v114, v114, v42
	s_wait_dscnt 0xc
	v_add_f32_e32 v115, v115, v43
	s_wait_dscnt 0xb
	v_add_f32_e32 v124, v124, v44
	s_wait_dscnt 0xa
	v_add_f32_e32 v125, v125, v45
	s_wait_dscnt 0x9
	v_add_f32_e32 v126, v126, v46
	s_wait_dscnt 0x8
	v_add_f32_e32 v127, v127, v47
	s_wait_dscnt 0x7
	v_add_f32_e32 v136, v136, v48
	s_wait_dscnt 0x6
	v_add_f32_e32 v137, v137, v49
	s_wait_dscnt 0x5
	v_add_f32_e32 v138, v138, v50
	s_wait_dscnt 0x4
	v_add_f32_e32 v139, v139, v51
	s_wait_dscnt 0x3
	v_add_f32_e32 v148, v148, v52
	s_wait_dscnt 0x2
	v_add_f32_e32 v149, v149, v53
	s_wait_dscnt 0x1
	v_add_f32_e32 v150, v150, v54
	s_wait_dscnt 0x0
	v_add_f32_e32 v151, v151, v55
	v_cmp_gt_u32_e32 vcc_lo, 24, v0
	v_cndmask_b32_e64 v3, 0, 8, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v3, v3, v0, 2
	ds_bpermute_b32 v24, v3, v64
	ds_bpermute_b32 v25, v3, v65
	ds_bpermute_b32 v26, v3, v66
	ds_bpermute_b32 v27, v3, v67
	ds_bpermute_b32 v28, v3, v76
	ds_bpermute_b32 v29, v3, v77
	ds_bpermute_b32 v30, v3, v78
	ds_bpermute_b32 v31, v3, v79
	ds_bpermute_b32 v32, v3, v88
	ds_bpermute_b32 v33, v3, v89
	ds_bpermute_b32 v34, v3, v90
	ds_bpermute_b32 v35, v3, v91
	ds_bpermute_b32 v36, v3, v100
	ds_bpermute_b32 v37, v3, v101
	ds_bpermute_b32 v38, v3, v102
	ds_bpermute_b32 v39, v3, v103
	ds_bpermute_b32 v40, v3, v112
	ds_bpermute_b32 v41, v3, v113
	ds_bpermute_b32 v42, v3, v114
	ds_bpermute_b32 v43, v3, v115
	ds_bpermute_b32 v44, v3, v124
	ds_bpermute_b32 v45, v3, v125
	ds_bpermute_b32 v46, v3, v126
	ds_bpermute_b32 v47, v3, v127
	ds_bpermute_b32 v48, v3, v136
	ds_bpermute_b32 v49, v3, v137
	ds_bpermute_b32 v50, v3, v138
	ds_bpermute_b32 v51, v3, v139
	ds_bpermute_b32 v52, v3, v148
	ds_bpermute_b32 v53, v3, v149
	ds_bpermute_b32 v54, v3, v150
	ds_bpermute_b32 v55, v3, v151
	s_wait_dscnt 0x1f
	v_add_f32_e32 v64, v64, v24
	s_wait_dscnt 0x1e
	v_add_f32_e32 v65, v65, v25
	s_wait_dscnt 0x1d
	v_add_f32_e32 v66, v66, v26
	s_wait_dscnt 0x1c
	v_add_f32_e32 v67, v67, v27
	s_wait_dscnt 0x1b
	v_add_f32_e32 v76, v76, v28
	s_wait_dscnt 0x1a
	v_add_f32_e32 v77, v77, v29
	s_wait_dscnt 0x19
	v_add_f32_e32 v78, v78, v30
	s_wait_dscnt 0x18
	v_add_f32_e32 v79, v79, v31
	s_wait_dscnt 0x17
	v_add_f32_e32 v88, v88, v32
	s_wait_dscnt 0x16
	v_add_f32_e32 v89, v89, v33
	s_wait_dscnt 0x15
	v_add_f32_e32 v90, v90, v34
	s_wait_dscnt 0x14
	v_add_f32_e32 v91, v91, v35
	s_wait_dscnt 0x13
	v_add_f32_e32 v100, v100, v36
	s_wait_dscnt 0x12
	v_add_f32_e32 v101, v101, v37
	s_wait_dscnt 0x11
	v_add_f32_e32 v102, v102, v38
	s_wait_dscnt 0x10
	v_add_f32_e32 v103, v103, v39
	s_wait_dscnt 0xf
	v_add_f32_e32 v112, v112, v40
	s_wait_dscnt 0xe
	v_add_f32_e32 v113, v113, v41
	s_wait_dscnt 0xd
	v_add_f32_e32 v114, v114, v42
	s_wait_dscnt 0xc
	v_add_f32_e32 v115, v115, v43
	s_wait_dscnt 0xb
	v_add_f32_e32 v124, v124, v44
	s_wait_dscnt 0xa
	v_add_f32_e32 v125, v125, v45
	s_wait_dscnt 0x9
	v_add_f32_e32 v126, v126, v46
	s_wait_dscnt 0x8
	v_add_f32_e32 v127, v127, v47
	s_wait_dscnt 0x7
	v_add_f32_e32 v136, v136, v48
	s_wait_dscnt 0x6
	v_add_f32_e32 v137, v137, v49
	s_wait_dscnt 0x5
	v_add_f32_e32 v138, v138, v50
	s_wait_dscnt 0x4
	v_add_f32_e32 v139, v139, v51
	s_wait_dscnt 0x3
	v_add_f32_e32 v148, v148, v52
	s_wait_dscnt 0x2
	v_add_f32_e32 v149, v149, v53
	s_wait_dscnt 0x1
	v_add_f32_e32 v150, v150, v54
	s_wait_dscnt 0x0
	v_add_f32_e32 v151, v151, v55
	v_cmp_gt_u32_e32 vcc_lo, 28, v0
	v_cndmask_b32_e64 v3, 0, 4, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v3, v3, v0, 2
	ds_bpermute_b32 v24, v3, v64
	ds_bpermute_b32 v25, v3, v65
	ds_bpermute_b32 v26, v3, v66
	ds_bpermute_b32 v27, v3, v67
	ds_bpermute_b32 v28, v3, v76
	ds_bpermute_b32 v29, v3, v77
	ds_bpermute_b32 v30, v3, v78
	ds_bpermute_b32 v31, v3, v79
	ds_bpermute_b32 v32, v3, v88
	ds_bpermute_b32 v33, v3, v89
	ds_bpermute_b32 v34, v3, v90
	ds_bpermute_b32 v35, v3, v91
	ds_bpermute_b32 v36, v3, v100
	ds_bpermute_b32 v37, v3, v101
	ds_bpermute_b32 v38, v3, v102
	ds_bpermute_b32 v39, v3, v103
	ds_bpermute_b32 v40, v3, v112
	ds_bpermute_b32 v41, v3, v113
	ds_bpermute_b32 v42, v3, v114
	ds_bpermute_b32 v43, v3, v115
	ds_bpermute_b32 v44, v3, v124
	ds_bpermute_b32 v45, v3, v125
	ds_bpermute_b32 v46, v3, v126
	ds_bpermute_b32 v47, v3, v127
	ds_bpermute_b32 v48, v3, v136
	ds_bpermute_b32 v49, v3, v137
	ds_bpermute_b32 v50, v3, v138
	ds_bpermute_b32 v51, v3, v139
	ds_bpermute_b32 v52, v3, v148
	ds_bpermute_b32 v53, v3, v149
	ds_bpermute_b32 v54, v3, v150
	ds_bpermute_b32 v55, v3, v151
	s_wait_dscnt 0x1f
	v_add_f32_e32 v64, v64, v24
	s_wait_dscnt 0x1e
	v_add_f32_e32 v65, v65, v25
	s_wait_dscnt 0x1d
	v_add_f32_e32 v66, v66, v26
	s_wait_dscnt 0x1c
	v_add_f32_e32 v67, v67, v27
	s_wait_dscnt 0x1b
	v_add_f32_e32 v76, v76, v28
	s_wait_dscnt 0x1a
	v_add_f32_e32 v77, v77, v29
	s_wait_dscnt 0x19
	v_add_f32_e32 v78, v78, v30
	s_wait_dscnt 0x18
	v_add_f32_e32 v79, v79, v31
	s_wait_dscnt 0x17
	v_add_f32_e32 v88, v88, v32
	s_wait_dscnt 0x16
	v_add_f32_e32 v89, v89, v33
	s_wait_dscnt 0x15
	v_add_f32_e32 v90, v90, v34
	s_wait_dscnt 0x14
	v_add_f32_e32 v91, v91, v35
	s_wait_dscnt 0x13
	v_add_f32_e32 v100, v100, v36
	s_wait_dscnt 0x12
	v_add_f32_e32 v101, v101, v37
	s_wait_dscnt 0x11
	v_add_f32_e32 v102, v102, v38
	s_wait_dscnt 0x10
	v_add_f32_e32 v103, v103, v39
	s_wait_dscnt 0xf
	v_add_f32_e32 v112, v112, v40
	s_wait_dscnt 0xe
	v_add_f32_e32 v113, v113, v41
	s_wait_dscnt 0xd
	v_add_f32_e32 v114, v114, v42
	s_wait_dscnt 0xc
	v_add_f32_e32 v115, v115, v43
	s_wait_dscnt 0xb
	v_add_f32_e32 v124, v124, v44
	s_wait_dscnt 0xa
	v_add_f32_e32 v125, v125, v45
	s_wait_dscnt 0x9
	v_add_f32_e32 v126, v126, v46
	s_wait_dscnt 0x8
	v_add_f32_e32 v127, v127, v47
	s_wait_dscnt 0x7
	v_add_f32_e32 v136, v136, v48
	s_wait_dscnt 0x6
	v_add_f32_e32 v137, v137, v49
	s_wait_dscnt 0x5
	v_add_f32_e32 v138, v138, v50
	s_wait_dscnt 0x4
	v_add_f32_e32 v139, v139, v51
	s_wait_dscnt 0x3
	v_add_f32_e32 v148, v148, v52
	s_wait_dscnt 0x2
	v_add_f32_e32 v149, v149, v53
	s_wait_dscnt 0x1
	v_add_f32_e32 v150, v150, v54
	s_wait_dscnt 0x0
	v_add_f32_e32 v151, v151, v55
	v_cmp_gt_u32_e32 vcc_lo, 30, v0
	v_cndmask_b32_e64 v3, 0, 2, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v3, v3, v0, 2
	ds_bpermute_b32 v24, v3, v64
	ds_bpermute_b32 v25, v3, v65
	ds_bpermute_b32 v26, v3, v66
	ds_bpermute_b32 v27, v3, v67
	ds_bpermute_b32 v28, v3, v76
	ds_bpermute_b32 v29, v3, v77
	ds_bpermute_b32 v30, v3, v78
	ds_bpermute_b32 v31, v3, v79
	ds_bpermute_b32 v32, v3, v88
	ds_bpermute_b32 v33, v3, v89
	ds_bpermute_b32 v34, v3, v90
	ds_bpermute_b32 v35, v3, v91
	ds_bpermute_b32 v36, v3, v100
	ds_bpermute_b32 v37, v3, v101
	ds_bpermute_b32 v38, v3, v102
	ds_bpermute_b32 v39, v3, v103
	ds_bpermute_b32 v40, v3, v112
	ds_bpermute_b32 v41, v3, v113
	ds_bpermute_b32 v42, v3, v114
	ds_bpermute_b32 v43, v3, v115
	ds_bpermute_b32 v44, v3, v124
	ds_bpermute_b32 v45, v3, v125
	ds_bpermute_b32 v46, v3, v126
	ds_bpermute_b32 v47, v3, v127
	ds_bpermute_b32 v48, v3, v136
	ds_bpermute_b32 v49, v3, v137
	ds_bpermute_b32 v50, v3, v138
	ds_bpermute_b32 v51, v3, v139
	ds_bpermute_b32 v52, v3, v148
	ds_bpermute_b32 v53, v3, v149
	ds_bpermute_b32 v54, v3, v150
	ds_bpermute_b32 v55, v3, v151
	s_wait_dscnt 0x1f
	v_add_f32_e32 v64, v64, v24
	s_wait_dscnt 0x1e
	v_add_f32_e32 v65, v65, v25
	s_wait_dscnt 0x1d
	v_add_f32_e32 v66, v66, v26
	s_wait_dscnt 0x1c
	v_add_f32_e32 v67, v67, v27
	s_wait_dscnt 0x1b
	v_add_f32_e32 v76, v76, v28
	s_wait_dscnt 0x1a
	v_add_f32_e32 v77, v77, v29
	s_wait_dscnt 0x19
	v_add_f32_e32 v78, v78, v30
	s_wait_dscnt 0x18
	v_add_f32_e32 v79, v79, v31
	s_wait_dscnt 0x17
	v_add_f32_e32 v88, v88, v32
	s_wait_dscnt 0x16
	v_add_f32_e32 v89, v89, v33
	s_wait_dscnt 0x15
	v_add_f32_e32 v90, v90, v34
	s_wait_dscnt 0x14
	v_add_f32_e32 v91, v91, v35
	s_wait_dscnt 0x13
	v_add_f32_e32 v100, v100, v36
	s_wait_dscnt 0x12
	v_add_f32_e32 v101, v101, v37
	s_wait_dscnt 0x11
	v_add_f32_e32 v102, v102, v38
	s_wait_dscnt 0x10
	v_add_f32_e32 v103, v103, v39
	s_wait_dscnt 0xf
	v_add_f32_e32 v112, v112, v40
	s_wait_dscnt 0xe
	v_add_f32_e32 v113, v113, v41
	s_wait_dscnt 0xd
	v_add_f32_e32 v114, v114, v42
	s_wait_dscnt 0xc
	v_add_f32_e32 v115, v115, v43
	s_wait_dscnt 0xb
	v_add_f32_e32 v124, v124, v44
	s_wait_dscnt 0xa
	v_add_f32_e32 v125, v125, v45
	s_wait_dscnt 0x9
	v_add_f32_e32 v126, v126, v46
	s_wait_dscnt 0x8
	v_add_f32_e32 v127, v127, v47
	s_wait_dscnt 0x7
	v_add_f32_e32 v136, v136, v48
	s_wait_dscnt 0x6
	v_add_f32_e32 v137, v137, v49
	s_wait_dscnt 0x5
	v_add_f32_e32 v138, v138, v50
	s_wait_dscnt 0x4
	v_add_f32_e32 v139, v139, v51
	s_wait_dscnt 0x3
	v_add_f32_e32 v148, v148, v52
	s_wait_dscnt 0x2
	v_add_f32_e32 v149, v149, v53
	s_wait_dscnt 0x1
	v_add_f32_e32 v150, v150, v54
	s_wait_dscnt 0x0
	v_add_f32_e32 v151, v151, v55
	v_cmp_gt_u32_e32 vcc_lo, 31, v0
	v_cndmask_b32_e64 v3, 0, 1, vcc_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v3, v3, v0, 2
	ds_bpermute_b32 v24, v3, v64
	ds_bpermute_b32 v25, v3, v65
	ds_bpermute_b32 v26, v3, v66
	ds_bpermute_b32 v27, v3, v67
	ds_bpermute_b32 v28, v3, v76
	ds_bpermute_b32 v29, v3, v77
	ds_bpermute_b32 v30, v3, v78
	ds_bpermute_b32 v31, v3, v79
	ds_bpermute_b32 v32, v3, v88
	ds_bpermute_b32 v33, v3, v89
	ds_bpermute_b32 v34, v3, v90
	ds_bpermute_b32 v35, v3, v91
	ds_bpermute_b32 v36, v3, v100
	ds_bpermute_b32 v37, v3, v101
	ds_bpermute_b32 v38, v3, v102
	ds_bpermute_b32 v39, v3, v103
	ds_bpermute_b32 v40, v3, v112
	ds_bpermute_b32 v41, v3, v113
	ds_bpermute_b32 v42, v3, v114
	ds_bpermute_b32 v43, v3, v115
	ds_bpermute_b32 v44, v3, v124
	ds_bpermute_b32 v45, v3, v125
	ds_bpermute_b32 v46, v3, v126
	ds_bpermute_b32 v47, v3, v127
	ds_bpermute_b32 v48, v3, v136
	ds_bpermute_b32 v49, v3, v137
	ds_bpermute_b32 v50, v3, v138
	ds_bpermute_b32 v51, v3, v139
	ds_bpermute_b32 v52, v3, v148
	ds_bpermute_b32 v53, v3, v149
	ds_bpermute_b32 v54, v3, v150
	ds_bpermute_b32 v55, v3, v151
	s_wait_dscnt 0x1f
	v_add_f32_e32 v64, v64, v24
	s_wait_dscnt 0x1e
	v_add_f32_e32 v65, v65, v25
	s_wait_dscnt 0x1d
	v_add_f32_e32 v66, v66, v26
	s_wait_dscnt 0x1c
	v_add_f32_e32 v67, v67, v27
	s_wait_dscnt 0x1b
	v_add_f32_e32 v76, v76, v28
	s_wait_dscnt 0x1a
	v_add_f32_e32 v77, v77, v29
	s_wait_dscnt 0x19
	v_add_f32_e32 v78, v78, v30
	s_wait_dscnt 0x18
	v_add_f32_e32 v79, v79, v31
	s_wait_dscnt 0x17
	v_add_f32_e32 v88, v88, v32
	s_wait_dscnt 0x16
	v_add_f32_e32 v89, v89, v33
	s_wait_dscnt 0x15
	v_add_f32_e32 v90, v90, v34
	s_wait_dscnt 0x14
	v_add_f32_e32 v91, v91, v35
	s_wait_dscnt 0x13
	v_add_f32_e32 v100, v100, v36
	s_wait_dscnt 0x12
	v_add_f32_e32 v101, v101, v37
	s_wait_dscnt 0x11
	v_add_f32_e32 v102, v102, v38
	s_wait_dscnt 0x10
	v_add_f32_e32 v103, v103, v39
	s_wait_dscnt 0xf
	v_add_f32_e32 v112, v112, v40
	s_wait_dscnt 0xe
	v_add_f32_e32 v113, v113, v41
	s_wait_dscnt 0xd
	v_add_f32_e32 v114, v114, v42
	s_wait_dscnt 0xc
	v_add_f32_e32 v115, v115, v43
	s_wait_dscnt 0xb
	v_add_f32_e32 v124, v124, v44
	s_wait_dscnt 0xa
	v_add_f32_e32 v125, v125, v45
	s_wait_dscnt 0x9
	v_add_f32_e32 v126, v126, v46
	s_wait_dscnt 0x8
	v_add_f32_e32 v127, v127, v47
	s_wait_dscnt 0x7
	v_add_f32_e32 v136, v136, v48
	s_wait_dscnt 0x6
	v_add_f32_e32 v137, v137, v49
	s_wait_dscnt 0x5
	v_add_f32_e32 v138, v138, v50
	s_wait_dscnt 0x4
	v_add_f32_e32 v139, v139, v51
	s_wait_dscnt 0x3
	v_add_f32_e32 v148, v148, v52
	s_wait_dscnt 0x2
	v_add_f32_e32 v149, v149, v53
	s_wait_dscnt 0x1
	v_add_f32_e32 v150, v150, v54
	s_wait_dscnt 0x0
	v_add_f32_e32 v151, v151, v55
	v_cmpx_eq_u32_e32 0, v0
	s_lshl_b32 s46, s12, 2
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v185, s46
	s_mul_i32 s47, s45, 1
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s47, s47, s46
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v186, s47
	s_mul_i32 s47, s45, 2
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s47, s47, s46
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v187, s47
	s_mul_i32 s47, s45, 3
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s47, s47, s46
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v188, s47
	s_mul_i32 s47, s45, 4
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s47, s47, s46
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v189, s47
	s_mul_i32 s47, s45, 5
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s47, s47, s46
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v190, s47
	s_mul_i32 s47, s45, 6
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s47, s47, s46
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v191, s47
	s_mul_i32 s47, s45, 7
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s47, s47, s46
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v192, s47
	s_add_co_i32 s13, s12, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cbranch_scc0 .Lrx_b8_p0_single
	global_load_b32 v56, v185, s[8:9]
	global_load_b32 v57, v185, s[8:9] offset:4
	global_load_b32 v60, v186, s[8:9]
	global_load_b32 v61, v186, s[8:9] offset:4
	global_load_b32 v161, v187, s[8:9]
	global_load_b32 v162, v187, s[8:9] offset:4
	global_load_b32 v165, v188, s[8:9]
	global_load_b32 v166, v188, s[8:9] offset:4
	global_load_b32 v169, v189, s[8:9]
	global_load_b32 v170, v189, s[8:9] offset:4
	global_load_b32 v173, v190, s[8:9]
	global_load_b32 v174, v190, s[8:9] offset:4
	global_load_b32 v177, v191, s[8:9]
	global_load_b32 v178, v191, s[8:9] offset:4
	global_load_b32 v181, v192, s[8:9]
	global_load_b32 v182, v192, s[8:9] offset:4
	s_wait_loadcnt 0xf
	v_add_f32_e32 v56, v64, v56
	global_store_b32 v185, v56, s[8:9]
	s_wait_loadcnt 0xe
	v_add_f32_e32 v57, v65, v57
	global_store_b32 v185, v57, s[8:9] offset:4
	s_wait_loadcnt 0xd
	v_add_f32_e32 v60, v76, v60
	global_store_b32 v186, v60, s[8:9]
	s_wait_loadcnt 0xc
	v_add_f32_e32 v61, v77, v61
	global_store_b32 v186, v61, s[8:9] offset:4
	s_wait_loadcnt 0xb
	v_add_f32_e32 v161, v88, v161
	global_store_b32 v187, v161, s[8:9]
	s_wait_loadcnt 0xa
	v_add_f32_e32 v162, v89, v162
	global_store_b32 v187, v162, s[8:9] offset:4
	s_wait_loadcnt 0x9
	v_add_f32_e32 v165, v100, v165
	global_store_b32 v188, v165, s[8:9]
	s_wait_loadcnt 0x8
	v_add_f32_e32 v166, v101, v166
	global_store_b32 v188, v166, s[8:9] offset:4
	s_wait_loadcnt 0x7
	v_add_f32_e32 v169, v112, v169
	global_store_b32 v189, v169, s[8:9]
	s_wait_loadcnt 0x6
	v_add_f32_e32 v170, v113, v170
	global_store_b32 v189, v170, s[8:9] offset:4
	s_wait_loadcnt 0x5
	v_add_f32_e32 v173, v124, v173
	global_store_b32 v190, v173, s[8:9]
	s_wait_loadcnt 0x4
	v_add_f32_e32 v174, v125, v174
	global_store_b32 v190, v174, s[8:9] offset:4
	s_wait_loadcnt 0x3
	v_add_f32_e32 v177, v136, v177
	global_store_b32 v191, v177, s[8:9]
	s_wait_loadcnt 0x2
	v_add_f32_e32 v178, v137, v178
	global_store_b32 v191, v178, s[8:9] offset:4
	s_wait_loadcnt 0x1
	v_add_f32_e32 v181, v148, v181
	global_store_b32 v192, v181, s[8:9]
	s_wait_loadcnt 0x0
	v_add_f32_e32 v182, v149, v182
	global_store_b32 v192, v182, s[8:9] offset:4
	s_wait_storecnt 0x0
	s_branch .Lrx_b8_p0_next
	.Lrx_b8_p0_single:
	global_load_b32 v56, v185, s[8:9]
	global_load_b32 v60, v186, s[8:9]
	global_load_b32 v161, v187, s[8:9]
	global_load_b32 v165, v188, s[8:9]
	global_load_b32 v169, v189, s[8:9]
	global_load_b32 v173, v190, s[8:9]
	global_load_b32 v177, v191, s[8:9]
	global_load_b32 v181, v192, s[8:9]
	s_wait_loadcnt 0x7
	v_add_f32_e32 v56, v56, v64
	global_store_b32 v185, v56, s[8:9]
	s_wait_loadcnt 0x6
	v_add_f32_e32 v60, v60, v76
	global_store_b32 v186, v60, s[8:9]
	s_wait_loadcnt 0x5
	v_add_f32_e32 v161, v161, v88
	global_store_b32 v187, v161, s[8:9]
	s_wait_loadcnt 0x4
	v_add_f32_e32 v165, v165, v100
	global_store_b32 v188, v165, s[8:9]
	s_wait_loadcnt 0x3
	v_add_f32_e32 v169, v169, v112
	global_store_b32 v189, v169, s[8:9]
	s_wait_loadcnt 0x2
	v_add_f32_e32 v173, v173, v124
	global_store_b32 v190, v173, s[8:9]
	s_wait_loadcnt 0x1
	v_add_f32_e32 v177, v177, v136
	global_store_b32 v191, v177, s[8:9]
	s_wait_loadcnt 0x0
	v_add_f32_e32 v181, v181, v148
	global_store_b32 v192, v181, s[8:9]
	s_wait_storecnt 0x0
	s_branch .Lrx_b8_stored
	.Lrx_b8_p0_next:
	s_add_co_i32 s13, s12, 2
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cbranch_scc0 .Lrx_b8_stored
	s_add_co_i32 s13, s12, 3
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s13, s10
	s_cbranch_scc0 .Lrx_b8_p1_single
	global_load_b32 v58, v185, s[8:9] offset:8
	global_load_b32 v59, v185, s[8:9] offset:12
	global_load_b32 v62, v186, s[8:9] offset:8
	global_load_b32 v63, v186, s[8:9] offset:12
	global_load_b32 v163, v187, s[8:9] offset:8
	global_load_b32 v164, v187, s[8:9] offset:12
	global_load_b32 v167, v188, s[8:9] offset:8
	global_load_b32 v168, v188, s[8:9] offset:12
	global_load_b32 v171, v189, s[8:9] offset:8
	global_load_b32 v172, v189, s[8:9] offset:12
	global_load_b32 v175, v190, s[8:9] offset:8
	global_load_b32 v176, v190, s[8:9] offset:12
	global_load_b32 v179, v191, s[8:9] offset:8
	global_load_b32 v180, v191, s[8:9] offset:12
	global_load_b32 v183, v192, s[8:9] offset:8
	global_load_b32 v184, v192, s[8:9] offset:12
	s_wait_loadcnt 0xf
	v_add_f32_e32 v58, v66, v58
	global_store_b32 v185, v58, s[8:9] offset:8
	s_wait_loadcnt 0xe
	v_add_f32_e32 v59, v67, v59
	global_store_b32 v185, v59, s[8:9] offset:12
	s_wait_loadcnt 0xd
	v_add_f32_e32 v62, v78, v62
	global_store_b32 v186, v62, s[8:9] offset:8
	s_wait_loadcnt 0xc
	v_add_f32_e32 v63, v79, v63
	global_store_b32 v186, v63, s[8:9] offset:12
	s_wait_loadcnt 0xb
	v_add_f32_e32 v163, v90, v163
	global_store_b32 v187, v163, s[8:9] offset:8
	s_wait_loadcnt 0xa
	v_add_f32_e32 v164, v91, v164
	global_store_b32 v187, v164, s[8:9] offset:12
	s_wait_loadcnt 0x9
	v_add_f32_e32 v167, v102, v167
	global_store_b32 v188, v167, s[8:9] offset:8
	s_wait_loadcnt 0x8
	v_add_f32_e32 v168, v103, v168
	global_store_b32 v188, v168, s[8:9] offset:12
	s_wait_loadcnt 0x7
	v_add_f32_e32 v171, v114, v171
	global_store_b32 v189, v171, s[8:9] offset:8
	s_wait_loadcnt 0x6
	v_add_f32_e32 v172, v115, v172
	global_store_b32 v189, v172, s[8:9] offset:12
	s_wait_loadcnt 0x5
	v_add_f32_e32 v175, v126, v175
	global_store_b32 v190, v175, s[8:9] offset:8
	s_wait_loadcnt 0x4
	v_add_f32_e32 v176, v127, v176
	global_store_b32 v190, v176, s[8:9] offset:12
	s_wait_loadcnt 0x3
	v_add_f32_e32 v179, v138, v179
	global_store_b32 v191, v179, s[8:9] offset:8
	s_wait_loadcnt 0x2
	v_add_f32_e32 v180, v139, v180
	global_store_b32 v191, v180, s[8:9] offset:12
	s_wait_loadcnt 0x1
	v_add_f32_e32 v183, v150, v183
	global_store_b32 v192, v183, s[8:9] offset:8
	s_wait_loadcnt 0x0
	v_add_f32_e32 v184, v151, v184
	global_store_b32 v192, v184, s[8:9] offset:12
	s_wait_storecnt 0x0
	s_branch .Lrx_b8_stored
	.Lrx_b8_p1_single:
	global_load_b32 v58, v185, s[8:9] offset:8
	global_load_b32 v62, v186, s[8:9] offset:8
	global_load_b32 v163, v187, s[8:9] offset:8
	global_load_b32 v167, v188, s[8:9] offset:8
	global_load_b32 v171, v189, s[8:9] offset:8
	global_load_b32 v175, v190, s[8:9] offset:8
	global_load_b32 v179, v191, s[8:9] offset:8
	global_load_b32 v183, v192, s[8:9] offset:8
	s_wait_loadcnt 0x7
	v_add_f32_e32 v58, v58, v66
	global_store_b32 v185, v58, s[8:9] offset:8
	s_wait_loadcnt 0x6
	v_add_f32_e32 v62, v62, v78
	global_store_b32 v186, v62, s[8:9] offset:8
	s_wait_loadcnt 0x5
	v_add_f32_e32 v163, v163, v90
	global_store_b32 v187, v163, s[8:9] offset:8
	s_wait_loadcnt 0x4
	v_add_f32_e32 v167, v167, v102
	global_store_b32 v188, v167, s[8:9] offset:8
	s_wait_loadcnt 0x3
	v_add_f32_e32 v171, v171, v114
	global_store_b32 v189, v171, s[8:9] offset:8
	s_wait_loadcnt 0x2
	v_add_f32_e32 v175, v175, v126
	global_store_b32 v190, v175, s[8:9] offset:8
	s_wait_loadcnt 0x1
	v_add_f32_e32 v179, v179, v138
	global_store_b32 v191, v179, s[8:9] offset:8
	s_wait_loadcnt 0x0
	v_add_f32_e32 v183, v183, v150
	global_store_b32 v192, v183, s[8:9] offset:8
	s_wait_storecnt 0x0
	s_branch .Lrx_b8_stored
	.Lrx_b8_stored:
	s_branch .Lrx_end
	.Lrx_end:
	s_endpgm
.Lgemv_mq4g256v2_residual_xbatch_end:
.size gemv_mq4g256v2_residual_xbatch, .Lgemv_mq4g256v2_residual_xbatch_end-gemv_mq4g256v2_residual_xbatch
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel gemv_mq4g256v2_residual_xbatch
	.amdhsa_group_segment_fixed_size 0
	.amdhsa_private_segment_fixed_size 0
	.amdhsa_kernarg_size 36
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
	.amdhsa_next_free_vgpr 225
	.amdhsa_next_free_sgpr 48
	.amdhsa_reserve_vcc 1
	.amdhsa_float_round_mode_32 0
	.amdhsa_float_round_mode_16_64 0
	.amdhsa_float_denorm_mode_32 3
	.amdhsa_float_denorm_mode_16_64 3
	.amdhsa_fp16_overflow 0
	.amdhsa_workgroup_processor_mode 1
	.amdhsa_memory_ordered 1
	.amdhsa_forward_progress 1
	.amdhsa_inst_pref_size ((instprefsize(.Lgemv_mq4g256v2_residual_xbatch_end-gemv_mq4g256v2_residual_xbatch)<<4)&4080)>>4
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
      - 
        .name: B
        .offset: 32
        .size: 4
        .value_kind: by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 36
    .max_flat_workgroup_size: 32
    .name: gemv_mq4g256v2_residual_xbatch
    .private_segment_fixed_size: 0
    .sgpr_count: 50
    .sgpr_spill_count: 0
    .symbol: gemv_mq4g256v2_residual_xbatch.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count: 225
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
