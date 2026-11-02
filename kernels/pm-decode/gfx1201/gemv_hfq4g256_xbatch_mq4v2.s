.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
.amdhsa_code_object_version 6
.text
.protected gemv_mq4g256v2_xbatch_pm
.globl gemv_mq4g256v2_xbatch_pm
.p2align 8
.type gemv_mq4g256v2_xbatch_pm,@function
gemv_mq4g256v2_xbatch_pm:
	s_load_b64 s[10:11], s[0:1], 0x18
	s_lshl_b32 s12, ttmp9, 2
	s_wait_kmcnt 0x0
	s_cmp_ge_i32 s12, s10
	s_cbranch_scc1 .Lpx_end
	s_load_b128 s[4:7], s[0:1], 0x0
	s_load_b64 s[8:9], s[0:1], 0x10
	s_load_b32 s3, s[0:1], 0x20
	s_sub_co_i32 s13, s10, s12
	s_min_i32 s13, s13, 4
	s_add_co_i32 s40, s10, -1
	s_lshr_b32 s14, s11, 8
	s_mul_i32 s15, s14, 0x88
	s_add_co_i32 s41, s12, 0
	s_min_i32 s41, s41, s40
	s_mul_i32 s16, s41, s15
	s_add_co_i32 s41, s12, 1
	s_min_i32 s41, s41, s40
	s_mul_i32 s17, s41, s15
	s_add_co_i32 s41, s12, 2
	s_min_i32 s41, s41, s40
	s_mul_i32 s18, s41, s15
	s_add_co_i32 s41, s12, 3
	s_min_i32 s41, s41, s40
	s_mul_i32 s19, s41, s15
	s_lshl_b32 s38, s11, 2
	s_lshl_b32 s39, s10, 2
	s_lshl_b32 s42, s12, 2
	s_mul_i32 s28, s38, 0
	s_mul_i32 s29, s38, 1
	s_mul_i32 s30, s38, 2
	s_mul_i32 s31, s38, 3
	s_mul_i32 s32, s38, 4
	s_mul_i32 s33, s38, 5
	s_mul_i32 s34, s38, 6
	s_mul_i32 s35, s38, 7
	v_lshrrev_b32_e32 v1, 4, v0
	v_lshlrev_b32_e32 v1, 2, v1
	v_lshlrev_b32_e32 v2, 2, v0
	v_lshlrev_b32_e32 v3, 5, v0
	s_wait_kmcnt 0x0
	s_mov_b32 s20, s4
	s_and_b32 s21, s5, 0xffff
	s_mov_b32 s22, -1
	s_mov_b32 s23, 0x31004000
	s_mov_b32 s24, s6
	s_and_b32 s25, s7, 0xffff
	s_mov_b32 s26, -1
	s_mov_b32 s27, 0x31004000
	s_cmp_eq_u32 s3, 1
	s_cbranch_scc1 .Lpx_b1
	s_cmp_eq_u32 s3, 2
	s_cbranch_scc1 .Lpx_b2
	s_cmp_eq_u32 s3, 3
	s_cbranch_scc1 .Lpx_b3
	s_cmp_eq_u32 s3, 4
	s_cbranch_scc1 .Lpx_b4
	s_cmp_eq_u32 s3, 5
	s_cbranch_scc1 .Lpx_b5
	s_cmp_eq_u32 s3, 6
	s_cbranch_scc1 .Lpx_b6
	s_cmp_eq_u32 s3, 7
	s_cbranch_scc1 .Lpx_b7
	s_cmp_eq_u32 s3, 8
	s_cbranch_scc1 .Lpx_b8
	s_branch .Lpx_end
	.Lpx_b1:
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
	s_lshr_b32 s36, s14, 2
	s_and_b32 s37, s14, 3
	s_cmp_eq_u32 s36, 0
	s_cbranch_scc1 .Lpx_b1_tail
	.Lpx_b1_quad:
	buffer_load_b32 v8, v1, s[20:23], s16 offen
	buffer_load_b32 v9, v1, s[20:23], s17 offen
	buffer_load_b32 v10, v1, s[20:23], s18 offen
	buffer_load_b32 v11, v1, s[20:23], s19 offen
	buffer_load_b32 v24, v2, s[20:23], s16 offen offset:8
	buffer_load_b32 v25, v2, s[20:23], s17 offen offset:8
	buffer_load_b32 v26, v2, s[20:23], s18 offen offset:8
	buffer_load_b32 v27, v2, s[20:23], s19 offen offset:8
	buffer_load_b128 v[80:83], v3, s[24:27], s28 offen
	buffer_load_b128 v[84:87], v3, s[24:27], s28 offen offset:16
	buffer_load_b32 v12, v1, s[20:23], s16 offen offset:136
	buffer_load_b32 v13, v1, s[20:23], s17 offen offset:136
	buffer_load_b32 v14, v1, s[20:23], s18 offen offset:136
	buffer_load_b32 v15, v1, s[20:23], s19 offen offset:136
	buffer_load_b32 v28, v2, s[20:23], s16 offen offset:144
	buffer_load_b32 v29, v2, s[20:23], s17 offen offset:144
	buffer_load_b32 v30, v2, s[20:23], s18 offen offset:144
	buffer_load_b32 v31, v2, s[20:23], s19 offen offset:144
	buffer_load_b32 v16, v1, s[20:23], s16 offen offset:272
	buffer_load_b32 v17, v1, s[20:23], s17 offen offset:272
	buffer_load_b32 v18, v1, s[20:23], s18 offen offset:272
	buffer_load_b32 v19, v1, s[20:23], s19 offen offset:272
	buffer_load_b32 v32, v2, s[20:23], s16 offen offset:280
	buffer_load_b32 v33, v2, s[20:23], s17 offen offset:280
	buffer_load_b32 v34, v2, s[20:23], s18 offen offset:280
	buffer_load_b32 v35, v2, s[20:23], s19 offen offset:280
	buffer_load_b32 v20, v1, s[20:23], s16 offen offset:408
	buffer_load_b32 v21, v1, s[20:23], s17 offen offset:408
	buffer_load_b32 v22, v1, s[20:23], s18 offen offset:408
	buffer_load_b32 v23, v1, s[20:23], s19 offen offset:408
	buffer_load_b32 v36, v2, s[20:23], s16 offen offset:416
	buffer_load_b32 v37, v2, s[20:23], s17 offen offset:416
	buffer_load_b32 v38, v2, s[20:23], s18 offen offset:416
	buffer_load_b32 v39, v2, s[20:23], s19 offen offset:416
	s_wait_loadcnt 0x1d
	v_and_b32_e32 v46, 0xf0f0f0f, v24
	s_wait_loadcnt 0x1c
	v_and_b32_e32 v56, 0xf0f0f0f, v25
	s_wait_loadcnt 0x1b
	v_and_b32_e32 v66, 0xf0f0f0f, v26
	s_wait_loadcnt 0x1a
	v_and_b32_e32 v76, 0xf0f0f0f, v27
	v_lshrrev_b32_e32 v47, 4, v24
	v_lshrrev_b32_e32 v57, 4, v25
	v_lshrrev_b32_e32 v67, 4, v26
	v_lshrrev_b32_e32 v77, 4, v27
	v_and_b32_e32 v47, 0xf0f0f0f, v47
	v_and_b32_e32 v57, 0xf0f0f0f, v57
	v_and_b32_e32 v67, 0xf0f0f0f, v67
	v_and_b32_e32 v77, 0xf0f0f0f, v77
	v_cvt_f32_ubyte0_e32 v40, v46
	v_cvt_f32_ubyte0_e32 v50, v56
	v_cvt_f32_ubyte0_e32 v60, v66
	v_cvt_f32_ubyte0_e32 v70, v76
	v_cvt_f32_ubyte1_e32 v42, v46
	v_cvt_f32_ubyte1_e32 v52, v56
	v_cvt_f32_ubyte1_e32 v62, v66
	v_cvt_f32_ubyte1_e32 v72, v76
	v_cvt_f32_ubyte2_e32 v44, v46
	v_cvt_f32_ubyte2_e32 v54, v56
	v_cvt_f32_ubyte2_e32 v64, v66
	v_cvt_f32_ubyte2_e32 v74, v76
	v_cvt_f32_ubyte3_e32 v46, v46
	v_cvt_f32_ubyte3_e32 v56, v56
	v_cvt_f32_ubyte3_e32 v66, v66
	v_cvt_f32_ubyte3_e32 v76, v76
	v_cvt_f32_ubyte0_e32 v41, v47
	v_cvt_f32_ubyte0_e32 v51, v57
	v_cvt_f32_ubyte0_e32 v61, v67
	v_cvt_f32_ubyte0_e32 v71, v77
	v_cvt_f32_ubyte1_e32 v43, v47
	v_cvt_f32_ubyte1_e32 v53, v57
	v_cvt_f32_ubyte1_e32 v63, v67
	v_cvt_f32_ubyte1_e32 v73, v77
	v_cvt_f32_ubyte2_e32 v45, v47
	v_cvt_f32_ubyte2_e32 v55, v57
	v_cvt_f32_ubyte2_e32 v65, v67
	v_cvt_f32_ubyte2_e32 v75, v77
	v_cvt_f32_ubyte3_e32 v47, v47
	v_cvt_f32_ubyte3_e32 v57, v57
	v_cvt_f32_ubyte3_e32 v67, v67
	v_cvt_f32_ubyte3_e32 v77, v77
	v_fma_mix_f32 v40, v8, v40, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v9, v50, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v10, v60, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v70, v11, v70, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v41, v8, v41, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v9, v51, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v10, v61, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v71, v11, v71, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v42, v8, v42, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v52, v9, v52, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v62, v10, v62, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v72, v11, v72, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v43, v8, v43, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v53, v9, v53, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v63, v10, v63, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v73, v11, v73, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v44, v8, v44, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v54, v9, v54, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v64, v10, v64, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v74, v11, v74, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v8, v45, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v9, v55, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v65, v10, v65, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v75, v11, v75, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v8, v46, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v9, v56, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v66, v10, v66, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v76, v11, v76, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v8, v47, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v9, v57, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v67, v10, v67, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v77, v11, v77, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	buffer_load_b128 v[88:91], v3, s[24:27], s28 offen offset:1024
	buffer_load_b128 v[92:95], v3, s[24:27], s28 offen offset:1040
	buffer_load_b128 v[96:99], v3, s[24:27], s28 offen offset:2048
	buffer_load_b128 v[100:103], v3, s[24:27], s28 offen offset:2064
	buffer_load_b128 v[104:107], v3, s[24:27], s28 offen offset:3072
	buffer_load_b128 v[108:111], v3, s[24:27], s28 offen offset:3088
	s_wait_loadcnt 0x1f
	v_dual_mul_f32 v48, v41, v81 :: v_dual_mul_f32 v49, v51, v81
	v_dual_mul_f32 v58, v61, v81 :: v_dual_mul_f32 v59, v71, v81
	v_dual_fmac_f32 v48, v40, v80 :: v_dual_fmac_f32 v49, v50, v80
	v_dual_fmac_f32 v58, v60, v80 :: v_dual_fmac_f32 v59, v70, v80
	v_dual_fmac_f32 v48, v42, v82 :: v_dual_fmac_f32 v49, v52, v82
	v_dual_fmac_f32 v58, v62, v82 :: v_dual_fmac_f32 v59, v72, v82
	v_dual_fmac_f32 v48, v43, v83 :: v_dual_fmac_f32 v49, v53, v83
	v_dual_fmac_f32 v58, v63, v83 :: v_dual_fmac_f32 v59, v73, v83
	s_wait_loadcnt 0x1e
	v_dual_fmac_f32 v48, v44, v84 :: v_dual_fmac_f32 v49, v54, v84
	v_dual_fmac_f32 v58, v64, v84 :: v_dual_fmac_f32 v59, v74, v84
	v_dual_fmac_f32 v48, v45, v85 :: v_dual_fmac_f32 v49, v55, v85
	v_dual_fmac_f32 v58, v65, v85 :: v_dual_fmac_f32 v59, v75, v85
	v_dual_fmac_f32 v48, v46, v86 :: v_dual_fmac_f32 v49, v56, v86
	v_dual_fmac_f32 v58, v66, v86 :: v_dual_fmac_f32 v59, v76, v86
	v_dual_fmac_f32 v48, v47, v87 :: v_dual_fmac_f32 v49, v57, v87
	v_dual_fmac_f32 v58, v67, v87 :: v_dual_fmac_f32 v59, v77, v87
	v_dual_add_f32 v112, v112, v48 :: v_dual_add_f32 v113, v113, v49
	v_dual_add_f32 v114, v114, v58 :: v_dual_add_f32 v115, v115, v59
	s_wait_loadcnt 0x19
	v_and_b32_e32 v46, 0xf0f0f0f, v28
	s_wait_loadcnt 0x18
	v_and_b32_e32 v56, 0xf0f0f0f, v29
	s_wait_loadcnt 0x17
	v_and_b32_e32 v66, 0xf0f0f0f, v30
	s_wait_loadcnt 0x16
	v_and_b32_e32 v76, 0xf0f0f0f, v31
	v_lshrrev_b32_e32 v47, 4, v28
	v_lshrrev_b32_e32 v57, 4, v29
	v_lshrrev_b32_e32 v67, 4, v30
	v_lshrrev_b32_e32 v77, 4, v31
	v_and_b32_e32 v47, 0xf0f0f0f, v47
	v_and_b32_e32 v57, 0xf0f0f0f, v57
	v_and_b32_e32 v67, 0xf0f0f0f, v67
	v_and_b32_e32 v77, 0xf0f0f0f, v77
	v_cvt_f32_ubyte0_e32 v40, v46
	v_cvt_f32_ubyte0_e32 v50, v56
	v_cvt_f32_ubyte0_e32 v60, v66
	v_cvt_f32_ubyte0_e32 v70, v76
	v_cvt_f32_ubyte1_e32 v42, v46
	v_cvt_f32_ubyte1_e32 v52, v56
	v_cvt_f32_ubyte1_e32 v62, v66
	v_cvt_f32_ubyte1_e32 v72, v76
	v_cvt_f32_ubyte2_e32 v44, v46
	v_cvt_f32_ubyte2_e32 v54, v56
	v_cvt_f32_ubyte2_e32 v64, v66
	v_cvt_f32_ubyte2_e32 v74, v76
	v_cvt_f32_ubyte3_e32 v46, v46
	v_cvt_f32_ubyte3_e32 v56, v56
	v_cvt_f32_ubyte3_e32 v66, v66
	v_cvt_f32_ubyte3_e32 v76, v76
	v_cvt_f32_ubyte0_e32 v41, v47
	v_cvt_f32_ubyte0_e32 v51, v57
	v_cvt_f32_ubyte0_e32 v61, v67
	v_cvt_f32_ubyte0_e32 v71, v77
	v_cvt_f32_ubyte1_e32 v43, v47
	v_cvt_f32_ubyte1_e32 v53, v57
	v_cvt_f32_ubyte1_e32 v63, v67
	v_cvt_f32_ubyte1_e32 v73, v77
	v_cvt_f32_ubyte2_e32 v45, v47
	v_cvt_f32_ubyte2_e32 v55, v57
	v_cvt_f32_ubyte2_e32 v65, v67
	v_cvt_f32_ubyte2_e32 v75, v77
	v_cvt_f32_ubyte3_e32 v47, v47
	v_cvt_f32_ubyte3_e32 v57, v57
	v_cvt_f32_ubyte3_e32 v67, v67
	v_cvt_f32_ubyte3_e32 v77, v77
	v_fma_mix_f32 v40, v12, v40, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v13, v50, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v14, v60, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v70, v15, v70, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v41, v12, v41, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v13, v51, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v14, v61, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v71, v15, v71, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v42, v12, v42, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v52, v13, v52, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v62, v14, v62, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v72, v15, v72, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v43, v12, v43, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v53, v13, v53, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v63, v14, v63, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v73, v15, v73, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v44, v12, v44, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v54, v13, v54, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v64, v14, v64, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v74, v15, v74, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v12, v45, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v13, v55, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v65, v14, v65, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v75, v15, v75, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v12, v46, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v13, v56, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v66, v14, v66, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v76, v15, v76, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v12, v47, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v13, v57, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v67, v14, v67, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v77, v15, v77, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x5
	v_dual_mul_f32 v48, v41, v89 :: v_dual_mul_f32 v49, v51, v89
	v_dual_mul_f32 v58, v61, v89 :: v_dual_mul_f32 v59, v71, v89
	v_dual_fmac_f32 v48, v40, v88 :: v_dual_fmac_f32 v49, v50, v88
	v_dual_fmac_f32 v58, v60, v88 :: v_dual_fmac_f32 v59, v70, v88
	v_dual_fmac_f32 v48, v42, v90 :: v_dual_fmac_f32 v49, v52, v90
	v_dual_fmac_f32 v58, v62, v90 :: v_dual_fmac_f32 v59, v72, v90
	v_dual_fmac_f32 v48, v43, v91 :: v_dual_fmac_f32 v49, v53, v91
	v_dual_fmac_f32 v58, v63, v91 :: v_dual_fmac_f32 v59, v73, v91
	s_wait_loadcnt 0x4
	v_dual_fmac_f32 v48, v44, v92 :: v_dual_fmac_f32 v49, v54, v92
	v_dual_fmac_f32 v58, v64, v92 :: v_dual_fmac_f32 v59, v74, v92
	v_dual_fmac_f32 v48, v45, v93 :: v_dual_fmac_f32 v49, v55, v93
	v_dual_fmac_f32 v58, v65, v93 :: v_dual_fmac_f32 v59, v75, v93
	v_dual_fmac_f32 v48, v46, v94 :: v_dual_fmac_f32 v49, v56, v94
	v_dual_fmac_f32 v58, v66, v94 :: v_dual_fmac_f32 v59, v76, v94
	v_dual_fmac_f32 v48, v47, v95 :: v_dual_fmac_f32 v49, v57, v95
	v_dual_fmac_f32 v58, v67, v95 :: v_dual_fmac_f32 v59, v77, v95
	v_dual_add_f32 v116, v116, v48 :: v_dual_add_f32 v117, v117, v49
	v_dual_add_f32 v118, v118, v58 :: v_dual_add_f32 v119, v119, v59
	v_and_b32_e32 v46, 0xf0f0f0f, v32
	v_and_b32_e32 v56, 0xf0f0f0f, v33
	v_and_b32_e32 v66, 0xf0f0f0f, v34
	v_and_b32_e32 v76, 0xf0f0f0f, v35
	v_lshrrev_b32_e32 v47, 4, v32
	v_lshrrev_b32_e32 v57, 4, v33
	v_lshrrev_b32_e32 v67, 4, v34
	v_lshrrev_b32_e32 v77, 4, v35
	v_and_b32_e32 v47, 0xf0f0f0f, v47
	v_and_b32_e32 v57, 0xf0f0f0f, v57
	v_and_b32_e32 v67, 0xf0f0f0f, v67
	v_and_b32_e32 v77, 0xf0f0f0f, v77
	v_cvt_f32_ubyte0_e32 v40, v46
	v_cvt_f32_ubyte0_e32 v50, v56
	v_cvt_f32_ubyte0_e32 v60, v66
	v_cvt_f32_ubyte0_e32 v70, v76
	v_cvt_f32_ubyte1_e32 v42, v46
	v_cvt_f32_ubyte1_e32 v52, v56
	v_cvt_f32_ubyte1_e32 v62, v66
	v_cvt_f32_ubyte1_e32 v72, v76
	v_cvt_f32_ubyte2_e32 v44, v46
	v_cvt_f32_ubyte2_e32 v54, v56
	v_cvt_f32_ubyte2_e32 v64, v66
	v_cvt_f32_ubyte2_e32 v74, v76
	v_cvt_f32_ubyte3_e32 v46, v46
	v_cvt_f32_ubyte3_e32 v56, v56
	v_cvt_f32_ubyte3_e32 v66, v66
	v_cvt_f32_ubyte3_e32 v76, v76
	v_cvt_f32_ubyte0_e32 v41, v47
	v_cvt_f32_ubyte0_e32 v51, v57
	v_cvt_f32_ubyte0_e32 v61, v67
	v_cvt_f32_ubyte0_e32 v71, v77
	v_cvt_f32_ubyte1_e32 v43, v47
	v_cvt_f32_ubyte1_e32 v53, v57
	v_cvt_f32_ubyte1_e32 v63, v67
	v_cvt_f32_ubyte1_e32 v73, v77
	v_cvt_f32_ubyte2_e32 v45, v47
	v_cvt_f32_ubyte2_e32 v55, v57
	v_cvt_f32_ubyte2_e32 v65, v67
	v_cvt_f32_ubyte2_e32 v75, v77
	v_cvt_f32_ubyte3_e32 v47, v47
	v_cvt_f32_ubyte3_e32 v57, v57
	v_cvt_f32_ubyte3_e32 v67, v67
	v_cvt_f32_ubyte3_e32 v77, v77
	v_fma_mix_f32 v40, v16, v40, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v17, v50, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v18, v60, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v70, v19, v70, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v41, v16, v41, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v17, v51, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v18, v61, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v71, v19, v71, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v42, v16, v42, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v52, v17, v52, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v62, v18, v62, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v72, v19, v72, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v43, v16, v43, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v53, v17, v53, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v63, v18, v63, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v73, v19, v73, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v44, v16, v44, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v54, v17, v54, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v64, v18, v64, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v74, v19, v74, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v16, v45, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v17, v55, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v65, v18, v65, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v75, v19, v75, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v16, v46, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v17, v56, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v66, v18, v66, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v76, v19, v76, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v16, v47, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v17, v57, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v67, v18, v67, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v77, v19, v77, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x3
	v_dual_mul_f32 v48, v41, v97 :: v_dual_mul_f32 v49, v51, v97
	v_dual_mul_f32 v58, v61, v97 :: v_dual_mul_f32 v59, v71, v97
	v_dual_fmac_f32 v48, v40, v96 :: v_dual_fmac_f32 v49, v50, v96
	v_dual_fmac_f32 v58, v60, v96 :: v_dual_fmac_f32 v59, v70, v96
	v_dual_fmac_f32 v48, v42, v98 :: v_dual_fmac_f32 v49, v52, v98
	v_dual_fmac_f32 v58, v62, v98 :: v_dual_fmac_f32 v59, v72, v98
	v_dual_fmac_f32 v48, v43, v99 :: v_dual_fmac_f32 v49, v53, v99
	v_dual_fmac_f32 v58, v63, v99 :: v_dual_fmac_f32 v59, v73, v99
	s_wait_loadcnt 0x2
	v_dual_fmac_f32 v48, v44, v100 :: v_dual_fmac_f32 v49, v54, v100
	v_dual_fmac_f32 v58, v64, v100 :: v_dual_fmac_f32 v59, v74, v100
	v_dual_fmac_f32 v48, v45, v101 :: v_dual_fmac_f32 v49, v55, v101
	v_dual_fmac_f32 v58, v65, v101 :: v_dual_fmac_f32 v59, v75, v101
	v_dual_fmac_f32 v48, v46, v102 :: v_dual_fmac_f32 v49, v56, v102
	v_dual_fmac_f32 v58, v66, v102 :: v_dual_fmac_f32 v59, v76, v102
	v_dual_fmac_f32 v48, v47, v103 :: v_dual_fmac_f32 v49, v57, v103
	v_dual_fmac_f32 v58, v67, v103 :: v_dual_fmac_f32 v59, v77, v103
	v_dual_add_f32 v120, v120, v48 :: v_dual_add_f32 v121, v121, v49
	v_dual_add_f32 v122, v122, v58 :: v_dual_add_f32 v123, v123, v59
	v_and_b32_e32 v46, 0xf0f0f0f, v36
	v_and_b32_e32 v56, 0xf0f0f0f, v37
	v_and_b32_e32 v66, 0xf0f0f0f, v38
	v_and_b32_e32 v76, 0xf0f0f0f, v39
	v_lshrrev_b32_e32 v47, 4, v36
	v_lshrrev_b32_e32 v57, 4, v37
	v_lshrrev_b32_e32 v67, 4, v38
	v_lshrrev_b32_e32 v77, 4, v39
	v_and_b32_e32 v47, 0xf0f0f0f, v47
	v_and_b32_e32 v57, 0xf0f0f0f, v57
	v_and_b32_e32 v67, 0xf0f0f0f, v67
	v_and_b32_e32 v77, 0xf0f0f0f, v77
	v_cvt_f32_ubyte0_e32 v40, v46
	v_cvt_f32_ubyte0_e32 v50, v56
	v_cvt_f32_ubyte0_e32 v60, v66
	v_cvt_f32_ubyte0_e32 v70, v76
	v_cvt_f32_ubyte1_e32 v42, v46
	v_cvt_f32_ubyte1_e32 v52, v56
	v_cvt_f32_ubyte1_e32 v62, v66
	v_cvt_f32_ubyte1_e32 v72, v76
	v_cvt_f32_ubyte2_e32 v44, v46
	v_cvt_f32_ubyte2_e32 v54, v56
	v_cvt_f32_ubyte2_e32 v64, v66
	v_cvt_f32_ubyte2_e32 v74, v76
	v_cvt_f32_ubyte3_e32 v46, v46
	v_cvt_f32_ubyte3_e32 v56, v56
	v_cvt_f32_ubyte3_e32 v66, v66
	v_cvt_f32_ubyte3_e32 v76, v76
	v_cvt_f32_ubyte0_e32 v41, v47
	v_cvt_f32_ubyte0_e32 v51, v57
	v_cvt_f32_ubyte0_e32 v61, v67
	v_cvt_f32_ubyte0_e32 v71, v77
	v_cvt_f32_ubyte1_e32 v43, v47
	v_cvt_f32_ubyte1_e32 v53, v57
	v_cvt_f32_ubyte1_e32 v63, v67
	v_cvt_f32_ubyte1_e32 v73, v77
	v_cvt_f32_ubyte2_e32 v45, v47
	v_cvt_f32_ubyte2_e32 v55, v57
	v_cvt_f32_ubyte2_e32 v65, v67
	v_cvt_f32_ubyte2_e32 v75, v77
	v_cvt_f32_ubyte3_e32 v47, v47
	v_cvt_f32_ubyte3_e32 v57, v57
	v_cvt_f32_ubyte3_e32 v67, v67
	v_cvt_f32_ubyte3_e32 v77, v77
	v_fma_mix_f32 v40, v20, v40, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v21, v50, v21 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v22, v60, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v70, v23, v70, v23 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v41, v20, v41, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v21, v51, v21 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v22, v61, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v71, v23, v71, v23 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v42, v20, v42, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v52, v21, v52, v21 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v62, v22, v62, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v72, v23, v72, v23 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v43, v20, v43, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v53, v21, v53, v21 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v63, v22, v63, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v73, v23, v73, v23 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v44, v20, v44, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v54, v21, v54, v21 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v64, v22, v64, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v74, v23, v74, v23 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v20, v45, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v21, v55, v21 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v65, v22, v65, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v75, v23, v75, v23 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v20, v46, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v21, v56, v21 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v66, v22, v66, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v76, v23, v76, v23 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v20, v47, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v21, v57, v21 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v67, v22, v67, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v77, v23, v77, v23 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x1
	v_dual_mul_f32 v48, v41, v105 :: v_dual_mul_f32 v49, v51, v105
	v_dual_mul_f32 v58, v61, v105 :: v_dual_mul_f32 v59, v71, v105
	v_dual_fmac_f32 v48, v40, v104 :: v_dual_fmac_f32 v49, v50, v104
	v_dual_fmac_f32 v58, v60, v104 :: v_dual_fmac_f32 v59, v70, v104
	v_dual_fmac_f32 v48, v42, v106 :: v_dual_fmac_f32 v49, v52, v106
	v_dual_fmac_f32 v58, v62, v106 :: v_dual_fmac_f32 v59, v72, v106
	v_dual_fmac_f32 v48, v43, v107 :: v_dual_fmac_f32 v49, v53, v107
	v_dual_fmac_f32 v58, v63, v107 :: v_dual_fmac_f32 v59, v73, v107
	s_wait_loadcnt 0x0
	v_dual_fmac_f32 v48, v44, v108 :: v_dual_fmac_f32 v49, v54, v108
	v_dual_fmac_f32 v58, v64, v108 :: v_dual_fmac_f32 v59, v74, v108
	v_dual_fmac_f32 v48, v45, v109 :: v_dual_fmac_f32 v49, v55, v109
	v_dual_fmac_f32 v58, v65, v109 :: v_dual_fmac_f32 v59, v75, v109
	v_dual_fmac_f32 v48, v46, v110 :: v_dual_fmac_f32 v49, v56, v110
	v_dual_fmac_f32 v58, v66, v110 :: v_dual_fmac_f32 v59, v76, v110
	v_dual_fmac_f32 v48, v47, v111 :: v_dual_fmac_f32 v49, v57, v111
	v_dual_fmac_f32 v58, v67, v111 :: v_dual_fmac_f32 v59, v77, v111
	v_dual_add_f32 v124, v124, v48 :: v_dual_add_f32 v125, v125, v49
	v_dual_add_f32 v126, v126, v58 :: v_dual_add_f32 v127, v127, v59
	s_add_co_i32 s16, s16, 0x220
	s_add_co_i32 s17, s17, 0x220
	s_add_co_i32 s18, s18, 0x220
	s_add_co_i32 s19, s19, 0x220
	s_add_co_i32 s28, s28, 0x1000
	s_add_co_i32 s36, s36, -1
	s_cmp_lg_u32 s36, 0
	s_cbranch_scc1 .Lpx_b1_quad
	.Lpx_b1_tail:
	s_cmp_gt_u32 s37, 0
	s_cbranch_scc0 .Lpx_b1_fold
	buffer_load_b32 v8, v1, s[20:23], s16 offen
	buffer_load_b32 v9, v1, s[20:23], s17 offen
	buffer_load_b32 v10, v1, s[20:23], s18 offen
	buffer_load_b32 v11, v1, s[20:23], s19 offen
	buffer_load_b32 v24, v2, s[20:23], s16 offen offset:8
	buffer_load_b32 v25, v2, s[20:23], s17 offen offset:8
	buffer_load_b32 v26, v2, s[20:23], s18 offen offset:8
	buffer_load_b32 v27, v2, s[20:23], s19 offen offset:8
	buffer_load_b128 v[80:83], v3, s[24:27], s28 offen
	buffer_load_b128 v[84:87], v3, s[24:27], s28 offen offset:16
	s_wait_loadcnt 0x5
	v_and_b32_e32 v46, 0xf0f0f0f, v24
	s_wait_loadcnt 0x4
	v_and_b32_e32 v56, 0xf0f0f0f, v25
	s_wait_loadcnt 0x3
	v_and_b32_e32 v66, 0xf0f0f0f, v26
	s_wait_loadcnt 0x2
	v_and_b32_e32 v76, 0xf0f0f0f, v27
	v_lshrrev_b32_e32 v47, 4, v24
	v_lshrrev_b32_e32 v57, 4, v25
	v_lshrrev_b32_e32 v67, 4, v26
	v_lshrrev_b32_e32 v77, 4, v27
	v_and_b32_e32 v47, 0xf0f0f0f, v47
	v_and_b32_e32 v57, 0xf0f0f0f, v57
	v_and_b32_e32 v67, 0xf0f0f0f, v67
	v_and_b32_e32 v77, 0xf0f0f0f, v77
	v_cvt_f32_ubyte0_e32 v40, v46
	v_cvt_f32_ubyte0_e32 v50, v56
	v_cvt_f32_ubyte0_e32 v60, v66
	v_cvt_f32_ubyte0_e32 v70, v76
	v_cvt_f32_ubyte1_e32 v42, v46
	v_cvt_f32_ubyte1_e32 v52, v56
	v_cvt_f32_ubyte1_e32 v62, v66
	v_cvt_f32_ubyte1_e32 v72, v76
	v_cvt_f32_ubyte2_e32 v44, v46
	v_cvt_f32_ubyte2_e32 v54, v56
	v_cvt_f32_ubyte2_e32 v64, v66
	v_cvt_f32_ubyte2_e32 v74, v76
	v_cvt_f32_ubyte3_e32 v46, v46
	v_cvt_f32_ubyte3_e32 v56, v56
	v_cvt_f32_ubyte3_e32 v66, v66
	v_cvt_f32_ubyte3_e32 v76, v76
	v_cvt_f32_ubyte0_e32 v41, v47
	v_cvt_f32_ubyte0_e32 v51, v57
	v_cvt_f32_ubyte0_e32 v61, v67
	v_cvt_f32_ubyte0_e32 v71, v77
	v_cvt_f32_ubyte1_e32 v43, v47
	v_cvt_f32_ubyte1_e32 v53, v57
	v_cvt_f32_ubyte1_e32 v63, v67
	v_cvt_f32_ubyte1_e32 v73, v77
	v_cvt_f32_ubyte2_e32 v45, v47
	v_cvt_f32_ubyte2_e32 v55, v57
	v_cvt_f32_ubyte2_e32 v65, v67
	v_cvt_f32_ubyte2_e32 v75, v77
	v_cvt_f32_ubyte3_e32 v47, v47
	v_cvt_f32_ubyte3_e32 v57, v57
	v_cvt_f32_ubyte3_e32 v67, v67
	v_cvt_f32_ubyte3_e32 v77, v77
	v_fma_mix_f32 v40, v8, v40, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v9, v50, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v10, v60, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v70, v11, v70, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v41, v8, v41, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v9, v51, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v10, v61, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v71, v11, v71, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v42, v8, v42, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v52, v9, v52, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v62, v10, v62, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v72, v11, v72, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v43, v8, v43, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v53, v9, v53, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v63, v10, v63, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v73, v11, v73, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v44, v8, v44, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v54, v9, v54, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v64, v10, v64, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v74, v11, v74, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v8, v45, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v9, v55, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v65, v10, v65, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v75, v11, v75, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v8, v46, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v9, v56, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v66, v10, v66, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v76, v11, v76, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v8, v47, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v9, v57, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v67, v10, v67, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v77, v11, v77, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x1
	v_dual_mul_f32 v48, v41, v81 :: v_dual_mul_f32 v49, v51, v81
	v_dual_mul_f32 v58, v61, v81 :: v_dual_mul_f32 v59, v71, v81
	v_dual_fmac_f32 v48, v40, v80 :: v_dual_fmac_f32 v49, v50, v80
	v_dual_fmac_f32 v58, v60, v80 :: v_dual_fmac_f32 v59, v70, v80
	v_dual_fmac_f32 v48, v42, v82 :: v_dual_fmac_f32 v49, v52, v82
	v_dual_fmac_f32 v58, v62, v82 :: v_dual_fmac_f32 v59, v72, v82
	v_dual_fmac_f32 v48, v43, v83 :: v_dual_fmac_f32 v49, v53, v83
	v_dual_fmac_f32 v58, v63, v83 :: v_dual_fmac_f32 v59, v73, v83
	s_wait_loadcnt 0x0
	v_dual_fmac_f32 v48, v44, v84 :: v_dual_fmac_f32 v49, v54, v84
	v_dual_fmac_f32 v58, v64, v84 :: v_dual_fmac_f32 v59, v74, v84
	v_dual_fmac_f32 v48, v45, v85 :: v_dual_fmac_f32 v49, v55, v85
	v_dual_fmac_f32 v58, v65, v85 :: v_dual_fmac_f32 v59, v75, v85
	v_dual_fmac_f32 v48, v46, v86 :: v_dual_fmac_f32 v49, v56, v86
	v_dual_fmac_f32 v58, v66, v86 :: v_dual_fmac_f32 v59, v76, v86
	v_dual_fmac_f32 v48, v47, v87 :: v_dual_fmac_f32 v49, v57, v87
	v_dual_fmac_f32 v58, v67, v87 :: v_dual_fmac_f32 v59, v77, v87
	v_dual_add_f32 v112, v112, v48 :: v_dual_add_f32 v113, v113, v49
	v_dual_add_f32 v114, v114, v58 :: v_dual_add_f32 v115, v115, v59
	s_cmp_gt_u32 s37, 1
	s_cbranch_scc0 .Lpx_b1_fold
	buffer_load_b32 v12, v1, s[20:23], s16 offen offset:136
	buffer_load_b32 v13, v1, s[20:23], s17 offen offset:136
	buffer_load_b32 v14, v1, s[20:23], s18 offen offset:136
	buffer_load_b32 v15, v1, s[20:23], s19 offen offset:136
	buffer_load_b32 v28, v2, s[20:23], s16 offen offset:144
	buffer_load_b32 v29, v2, s[20:23], s17 offen offset:144
	buffer_load_b32 v30, v2, s[20:23], s18 offen offset:144
	buffer_load_b32 v31, v2, s[20:23], s19 offen offset:144
	buffer_load_b128 v[80:83], v3, s[24:27], s28 offen offset:1024
	buffer_load_b128 v[84:87], v3, s[24:27], s28 offen offset:1040
	s_wait_loadcnt 0x5
	v_and_b32_e32 v46, 0xf0f0f0f, v28
	s_wait_loadcnt 0x4
	v_and_b32_e32 v56, 0xf0f0f0f, v29
	s_wait_loadcnt 0x3
	v_and_b32_e32 v66, 0xf0f0f0f, v30
	s_wait_loadcnt 0x2
	v_and_b32_e32 v76, 0xf0f0f0f, v31
	v_lshrrev_b32_e32 v47, 4, v28
	v_lshrrev_b32_e32 v57, 4, v29
	v_lshrrev_b32_e32 v67, 4, v30
	v_lshrrev_b32_e32 v77, 4, v31
	v_and_b32_e32 v47, 0xf0f0f0f, v47
	v_and_b32_e32 v57, 0xf0f0f0f, v57
	v_and_b32_e32 v67, 0xf0f0f0f, v67
	v_and_b32_e32 v77, 0xf0f0f0f, v77
	v_cvt_f32_ubyte0_e32 v40, v46
	v_cvt_f32_ubyte0_e32 v50, v56
	v_cvt_f32_ubyte0_e32 v60, v66
	v_cvt_f32_ubyte0_e32 v70, v76
	v_cvt_f32_ubyte1_e32 v42, v46
	v_cvt_f32_ubyte1_e32 v52, v56
	v_cvt_f32_ubyte1_e32 v62, v66
	v_cvt_f32_ubyte1_e32 v72, v76
	v_cvt_f32_ubyte2_e32 v44, v46
	v_cvt_f32_ubyte2_e32 v54, v56
	v_cvt_f32_ubyte2_e32 v64, v66
	v_cvt_f32_ubyte2_e32 v74, v76
	v_cvt_f32_ubyte3_e32 v46, v46
	v_cvt_f32_ubyte3_e32 v56, v56
	v_cvt_f32_ubyte3_e32 v66, v66
	v_cvt_f32_ubyte3_e32 v76, v76
	v_cvt_f32_ubyte0_e32 v41, v47
	v_cvt_f32_ubyte0_e32 v51, v57
	v_cvt_f32_ubyte0_e32 v61, v67
	v_cvt_f32_ubyte0_e32 v71, v77
	v_cvt_f32_ubyte1_e32 v43, v47
	v_cvt_f32_ubyte1_e32 v53, v57
	v_cvt_f32_ubyte1_e32 v63, v67
	v_cvt_f32_ubyte1_e32 v73, v77
	v_cvt_f32_ubyte2_e32 v45, v47
	v_cvt_f32_ubyte2_e32 v55, v57
	v_cvt_f32_ubyte2_e32 v65, v67
	v_cvt_f32_ubyte2_e32 v75, v77
	v_cvt_f32_ubyte3_e32 v47, v47
	v_cvt_f32_ubyte3_e32 v57, v57
	v_cvt_f32_ubyte3_e32 v67, v67
	v_cvt_f32_ubyte3_e32 v77, v77
	v_fma_mix_f32 v40, v12, v40, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v13, v50, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v14, v60, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v70, v15, v70, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v41, v12, v41, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v13, v51, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v14, v61, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v71, v15, v71, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v42, v12, v42, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v52, v13, v52, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v62, v14, v62, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v72, v15, v72, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v43, v12, v43, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v53, v13, v53, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v63, v14, v63, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v73, v15, v73, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v44, v12, v44, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v54, v13, v54, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v64, v14, v64, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v74, v15, v74, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v12, v45, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v13, v55, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v65, v14, v65, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v75, v15, v75, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v12, v46, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v13, v56, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v66, v14, v66, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v76, v15, v76, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v12, v47, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v13, v57, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v67, v14, v67, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v77, v15, v77, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x1
	v_dual_mul_f32 v48, v41, v81 :: v_dual_mul_f32 v49, v51, v81
	v_dual_mul_f32 v58, v61, v81 :: v_dual_mul_f32 v59, v71, v81
	v_dual_fmac_f32 v48, v40, v80 :: v_dual_fmac_f32 v49, v50, v80
	v_dual_fmac_f32 v58, v60, v80 :: v_dual_fmac_f32 v59, v70, v80
	v_dual_fmac_f32 v48, v42, v82 :: v_dual_fmac_f32 v49, v52, v82
	v_dual_fmac_f32 v58, v62, v82 :: v_dual_fmac_f32 v59, v72, v82
	v_dual_fmac_f32 v48, v43, v83 :: v_dual_fmac_f32 v49, v53, v83
	v_dual_fmac_f32 v58, v63, v83 :: v_dual_fmac_f32 v59, v73, v83
	s_wait_loadcnt 0x0
	v_dual_fmac_f32 v48, v44, v84 :: v_dual_fmac_f32 v49, v54, v84
	v_dual_fmac_f32 v58, v64, v84 :: v_dual_fmac_f32 v59, v74, v84
	v_dual_fmac_f32 v48, v45, v85 :: v_dual_fmac_f32 v49, v55, v85
	v_dual_fmac_f32 v58, v65, v85 :: v_dual_fmac_f32 v59, v75, v85
	v_dual_fmac_f32 v48, v46, v86 :: v_dual_fmac_f32 v49, v56, v86
	v_dual_fmac_f32 v58, v66, v86 :: v_dual_fmac_f32 v59, v76, v86
	v_dual_fmac_f32 v48, v47, v87 :: v_dual_fmac_f32 v49, v57, v87
	v_dual_fmac_f32 v58, v67, v87 :: v_dual_fmac_f32 v59, v77, v87
	v_dual_add_f32 v116, v116, v48 :: v_dual_add_f32 v117, v117, v49
	v_dual_add_f32 v118, v118, v58 :: v_dual_add_f32 v119, v119, v59
	s_cmp_gt_u32 s37, 2
	s_cbranch_scc0 .Lpx_b1_fold
	buffer_load_b32 v16, v1, s[20:23], s16 offen offset:272
	buffer_load_b32 v17, v1, s[20:23], s17 offen offset:272
	buffer_load_b32 v18, v1, s[20:23], s18 offen offset:272
	buffer_load_b32 v19, v1, s[20:23], s19 offen offset:272
	buffer_load_b32 v32, v2, s[20:23], s16 offen offset:280
	buffer_load_b32 v33, v2, s[20:23], s17 offen offset:280
	buffer_load_b32 v34, v2, s[20:23], s18 offen offset:280
	buffer_load_b32 v35, v2, s[20:23], s19 offen offset:280
	buffer_load_b128 v[80:83], v3, s[24:27], s28 offen offset:2048
	buffer_load_b128 v[84:87], v3, s[24:27], s28 offen offset:2064
	s_wait_loadcnt 0x5
	v_and_b32_e32 v46, 0xf0f0f0f, v32
	s_wait_loadcnt 0x4
	v_and_b32_e32 v56, 0xf0f0f0f, v33
	s_wait_loadcnt 0x3
	v_and_b32_e32 v66, 0xf0f0f0f, v34
	s_wait_loadcnt 0x2
	v_and_b32_e32 v76, 0xf0f0f0f, v35
	v_lshrrev_b32_e32 v47, 4, v32
	v_lshrrev_b32_e32 v57, 4, v33
	v_lshrrev_b32_e32 v67, 4, v34
	v_lshrrev_b32_e32 v77, 4, v35
	v_and_b32_e32 v47, 0xf0f0f0f, v47
	v_and_b32_e32 v57, 0xf0f0f0f, v57
	v_and_b32_e32 v67, 0xf0f0f0f, v67
	v_and_b32_e32 v77, 0xf0f0f0f, v77
	v_cvt_f32_ubyte0_e32 v40, v46
	v_cvt_f32_ubyte0_e32 v50, v56
	v_cvt_f32_ubyte0_e32 v60, v66
	v_cvt_f32_ubyte0_e32 v70, v76
	v_cvt_f32_ubyte1_e32 v42, v46
	v_cvt_f32_ubyte1_e32 v52, v56
	v_cvt_f32_ubyte1_e32 v62, v66
	v_cvt_f32_ubyte1_e32 v72, v76
	v_cvt_f32_ubyte2_e32 v44, v46
	v_cvt_f32_ubyte2_e32 v54, v56
	v_cvt_f32_ubyte2_e32 v64, v66
	v_cvt_f32_ubyte2_e32 v74, v76
	v_cvt_f32_ubyte3_e32 v46, v46
	v_cvt_f32_ubyte3_e32 v56, v56
	v_cvt_f32_ubyte3_e32 v66, v66
	v_cvt_f32_ubyte3_e32 v76, v76
	v_cvt_f32_ubyte0_e32 v41, v47
	v_cvt_f32_ubyte0_e32 v51, v57
	v_cvt_f32_ubyte0_e32 v61, v67
	v_cvt_f32_ubyte0_e32 v71, v77
	v_cvt_f32_ubyte1_e32 v43, v47
	v_cvt_f32_ubyte1_e32 v53, v57
	v_cvt_f32_ubyte1_e32 v63, v67
	v_cvt_f32_ubyte1_e32 v73, v77
	v_cvt_f32_ubyte2_e32 v45, v47
	v_cvt_f32_ubyte2_e32 v55, v57
	v_cvt_f32_ubyte2_e32 v65, v67
	v_cvt_f32_ubyte2_e32 v75, v77
	v_cvt_f32_ubyte3_e32 v47, v47
	v_cvt_f32_ubyte3_e32 v57, v57
	v_cvt_f32_ubyte3_e32 v67, v67
	v_cvt_f32_ubyte3_e32 v77, v77
	v_fma_mix_f32 v40, v16, v40, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v17, v50, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v18, v60, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v70, v19, v70, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v41, v16, v41, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v17, v51, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v18, v61, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v71, v19, v71, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v42, v16, v42, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v52, v17, v52, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v62, v18, v62, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v72, v19, v72, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v43, v16, v43, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v53, v17, v53, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v63, v18, v63, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v73, v19, v73, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v44, v16, v44, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v54, v17, v54, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v64, v18, v64, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v74, v19, v74, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v16, v45, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v17, v55, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v65, v18, v65, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v75, v19, v75, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v16, v46, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v17, v56, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v66, v18, v66, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v76, v19, v76, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v16, v47, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v17, v57, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v67, v18, v67, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v77, v19, v77, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x1
	v_dual_mul_f32 v48, v41, v81 :: v_dual_mul_f32 v49, v51, v81
	v_dual_mul_f32 v58, v61, v81 :: v_dual_mul_f32 v59, v71, v81
	v_dual_fmac_f32 v48, v40, v80 :: v_dual_fmac_f32 v49, v50, v80
	v_dual_fmac_f32 v58, v60, v80 :: v_dual_fmac_f32 v59, v70, v80
	v_dual_fmac_f32 v48, v42, v82 :: v_dual_fmac_f32 v49, v52, v82
	v_dual_fmac_f32 v58, v62, v82 :: v_dual_fmac_f32 v59, v72, v82
	v_dual_fmac_f32 v48, v43, v83 :: v_dual_fmac_f32 v49, v53, v83
	v_dual_fmac_f32 v58, v63, v83 :: v_dual_fmac_f32 v59, v73, v83
	s_wait_loadcnt 0x0
	v_dual_fmac_f32 v48, v44, v84 :: v_dual_fmac_f32 v49, v54, v84
	v_dual_fmac_f32 v58, v64, v84 :: v_dual_fmac_f32 v59, v74, v84
	v_dual_fmac_f32 v48, v45, v85 :: v_dual_fmac_f32 v49, v55, v85
	v_dual_fmac_f32 v58, v65, v85 :: v_dual_fmac_f32 v59, v75, v85
	v_dual_fmac_f32 v48, v46, v86 :: v_dual_fmac_f32 v49, v56, v86
	v_dual_fmac_f32 v58, v66, v86 :: v_dual_fmac_f32 v59, v76, v86
	v_dual_fmac_f32 v48, v47, v87 :: v_dual_fmac_f32 v49, v57, v87
	v_dual_fmac_f32 v58, v67, v87 :: v_dual_fmac_f32 v59, v77, v87
	v_dual_add_f32 v120, v120, v48 :: v_dual_add_f32 v121, v121, v49
	v_dual_add_f32 v122, v122, v58 :: v_dual_add_f32 v123, v123, v59
	.Lpx_b1_fold:
	v_dual_add_f32 v112, v112, v116 :: v_dual_add_f32 v113, v113, v117
	v_dual_add_f32 v114, v114, v118 :: v_dual_add_f32 v115, v115, v119
	v_dual_add_f32 v120, v120, v124 :: v_dual_add_f32 v121, v121, v125
	v_dual_add_f32 v122, v122, v126 :: v_dual_add_f32 v123, v123, v127
	v_dual_add_f32 v112, v112, v120 :: v_dual_add_f32 v113, v113, v121
	v_dual_add_f32 v114, v114, v122 :: v_dual_add_f32 v115, v115, v123
	ds_swizzle_b32 v8, v112 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v9, v113 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v10, v114 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v11, v115 offset:swizzle(BITMASK_PERM,"1pppp")
	s_wait_dscnt 0x3
	v_add_f32_e32 v112, v112, v8
	s_wait_dscnt 0x2
	v_add_f32_e32 v113, v113, v9
	s_wait_dscnt 0x1
	v_add_f32_e32 v114, v114, v10
	s_wait_dscnt 0x0
	v_add_f32_e32 v115, v115, v11
	v_cmp_gt_u32_e32 vcc_lo, 24, v0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v4, 0, 8, vcc_lo
	v_add_lshl_u32 v4, v4, v0, 2
	ds_bpermute_b32 v8, v4, v112
	ds_bpermute_b32 v9, v4, v113
	ds_bpermute_b32 v10, v4, v114
	ds_bpermute_b32 v11, v4, v115
	s_wait_dscnt 0x3
	v_add_f32_e32 v112, v112, v8
	s_wait_dscnt 0x2
	v_add_f32_e32 v113, v113, v9
	s_wait_dscnt 0x1
	v_add_f32_e32 v114, v114, v10
	s_wait_dscnt 0x0
	v_add_f32_e32 v115, v115, v11
	v_cmp_gt_u32_e32 vcc_lo, 28, v0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v4, 0, 4, vcc_lo
	v_add_lshl_u32 v4, v4, v0, 2
	ds_bpermute_b32 v8, v4, v112
	ds_bpermute_b32 v9, v4, v113
	ds_bpermute_b32 v10, v4, v114
	ds_bpermute_b32 v11, v4, v115
	s_wait_dscnt 0x3
	v_add_f32_e32 v112, v112, v8
	s_wait_dscnt 0x2
	v_add_f32_e32 v113, v113, v9
	s_wait_dscnt 0x1
	v_add_f32_e32 v114, v114, v10
	s_wait_dscnt 0x0
	v_add_f32_e32 v115, v115, v11
	v_cmp_gt_u32_e32 vcc_lo, 30, v0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v4, 0, 2, vcc_lo
	v_add_lshl_u32 v4, v4, v0, 2
	ds_bpermute_b32 v8, v4, v112
	ds_bpermute_b32 v9, v4, v113
	ds_bpermute_b32 v10, v4, v114
	ds_bpermute_b32 v11, v4, v115
	s_wait_dscnt 0x3
	v_add_f32_e32 v112, v112, v8
	s_wait_dscnt 0x2
	v_add_f32_e32 v113, v113, v9
	s_wait_dscnt 0x1
	v_add_f32_e32 v114, v114, v10
	s_wait_dscnt 0x0
	v_add_f32_e32 v115, v115, v11
	v_cmp_gt_u32_e32 vcc_lo, 31, v0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v4, 0, 1, vcc_lo
	v_add_lshl_u32 v4, v4, v0, 2
	ds_bpermute_b32 v8, v4, v112
	ds_bpermute_b32 v9, v4, v113
	ds_bpermute_b32 v10, v4, v114
	ds_bpermute_b32 v11, v4, v115
	s_wait_dscnt 0x3
	v_add_f32_e32 v112, v112, v8
	s_wait_dscnt 0x2
	v_add_f32_e32 v113, v113, v9
	s_wait_dscnt 0x1
	v_add_f32_e32 v114, v114, v10
	s_wait_dscnt 0x0
	v_add_f32_e32 v115, v115, v11
	v_cmpx_eq_u32_e32 0, v0
	v_mov_b32_e32 v40, s42
	global_store_b32 v40, v112, s[8:9]
	s_cmp_le_i32 s13, 1
	s_cbranch_scc1 .Lpx_b1_stored
	global_store_b32 v40, v113, s[8:9] offset:4
	s_cmp_le_i32 s13, 2
	s_cbranch_scc1 .Lpx_b1_stored
	global_store_b32 v40, v114, s[8:9] offset:8
	s_cmp_le_i32 s13, 3
	s_cbranch_scc1 .Lpx_b1_stored
	global_store_b32 v40, v115, s[8:9] offset:12
	.Lpx_b1_stored:
	s_wait_storecnt 0x0
	s_branch .Lpx_end
	.Lpx_b2:
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
	v_mov_b32_e32 v160, 0
	v_mov_b32_e32 v161, 0
	v_mov_b32_e32 v162, 0
	v_mov_b32_e32 v163, 0
	v_mov_b32_e32 v164, 0
	v_mov_b32_e32 v165, 0
	v_mov_b32_e32 v166, 0
	v_mov_b32_e32 v167, 0
	v_mov_b32_e32 v168, 0
	v_mov_b32_e32 v169, 0
	v_mov_b32_e32 v170, 0
	v_mov_b32_e32 v171, 0
	v_mov_b32_e32 v172, 0
	v_mov_b32_e32 v173, 0
	v_mov_b32_e32 v174, 0
	v_mov_b32_e32 v175, 0
	s_lshr_b32 s36, s14, 2
	s_and_b32 s37, s14, 3
	s_cmp_eq_u32 s36, 0
	s_cbranch_scc1 .Lpx_b2_tail
	.Lpx_b2_quad:
	buffer_load_b32 v8, v1, s[20:23], s16 offen
	buffer_load_b32 v9, v1, s[20:23], s17 offen
	buffer_load_b32 v10, v1, s[20:23], s18 offen
	buffer_load_b32 v11, v1, s[20:23], s19 offen
	buffer_load_b32 v24, v2, s[20:23], s16 offen offset:8
	buffer_load_b32 v25, v2, s[20:23], s17 offen offset:8
	buffer_load_b32 v26, v2, s[20:23], s18 offen offset:8
	buffer_load_b32 v27, v2, s[20:23], s19 offen offset:8
	buffer_load_b128 v[80:83], v3, s[24:27], s28 offen
	buffer_load_b128 v[84:87], v3, s[24:27], s28 offen offset:16
	buffer_load_b128 v[88:91], v3, s[24:27], s29 offen
	buffer_load_b128 v[92:95], v3, s[24:27], s29 offen offset:16
	buffer_load_b32 v12, v1, s[20:23], s16 offen offset:136
	buffer_load_b32 v13, v1, s[20:23], s17 offen offset:136
	buffer_load_b32 v14, v1, s[20:23], s18 offen offset:136
	buffer_load_b32 v15, v1, s[20:23], s19 offen offset:136
	buffer_load_b32 v28, v2, s[20:23], s16 offen offset:144
	buffer_load_b32 v29, v2, s[20:23], s17 offen offset:144
	buffer_load_b32 v30, v2, s[20:23], s18 offen offset:144
	buffer_load_b32 v31, v2, s[20:23], s19 offen offset:144
	buffer_load_b32 v16, v1, s[20:23], s16 offen offset:272
	buffer_load_b32 v17, v1, s[20:23], s17 offen offset:272
	buffer_load_b32 v18, v1, s[20:23], s18 offen offset:272
	buffer_load_b32 v19, v1, s[20:23], s19 offen offset:272
	buffer_load_b32 v32, v2, s[20:23], s16 offen offset:280
	buffer_load_b32 v33, v2, s[20:23], s17 offen offset:280
	buffer_load_b32 v34, v2, s[20:23], s18 offen offset:280
	buffer_load_b32 v35, v2, s[20:23], s19 offen offset:280
	buffer_load_b32 v20, v1, s[20:23], s16 offen offset:408
	buffer_load_b32 v21, v1, s[20:23], s17 offen offset:408
	buffer_load_b32 v22, v1, s[20:23], s18 offen offset:408
	buffer_load_b32 v23, v1, s[20:23], s19 offen offset:408
	buffer_load_b32 v36, v2, s[20:23], s16 offen offset:416
	buffer_load_b32 v37, v2, s[20:23], s17 offen offset:416
	buffer_load_b32 v38, v2, s[20:23], s18 offen offset:416
	buffer_load_b32 v39, v2, s[20:23], s19 offen offset:416
	s_wait_loadcnt 0x1f
	v_and_b32_e32 v46, 0xf0f0f0f, v24
	s_wait_loadcnt 0x1e
	v_and_b32_e32 v56, 0xf0f0f0f, v25
	s_wait_loadcnt 0x1d
	v_and_b32_e32 v66, 0xf0f0f0f, v26
	s_wait_loadcnt 0x1c
	v_and_b32_e32 v76, 0xf0f0f0f, v27
	v_lshrrev_b32_e32 v47, 4, v24
	v_lshrrev_b32_e32 v57, 4, v25
	v_lshrrev_b32_e32 v67, 4, v26
	v_lshrrev_b32_e32 v77, 4, v27
	v_and_b32_e32 v47, 0xf0f0f0f, v47
	v_and_b32_e32 v57, 0xf0f0f0f, v57
	v_and_b32_e32 v67, 0xf0f0f0f, v67
	v_and_b32_e32 v77, 0xf0f0f0f, v77
	v_cvt_f32_ubyte0_e32 v40, v46
	v_cvt_f32_ubyte0_e32 v50, v56
	v_cvt_f32_ubyte0_e32 v60, v66
	v_cvt_f32_ubyte0_e32 v70, v76
	v_cvt_f32_ubyte1_e32 v42, v46
	v_cvt_f32_ubyte1_e32 v52, v56
	v_cvt_f32_ubyte1_e32 v62, v66
	v_cvt_f32_ubyte1_e32 v72, v76
	v_cvt_f32_ubyte2_e32 v44, v46
	v_cvt_f32_ubyte2_e32 v54, v56
	v_cvt_f32_ubyte2_e32 v64, v66
	v_cvt_f32_ubyte2_e32 v74, v76
	v_cvt_f32_ubyte3_e32 v46, v46
	v_cvt_f32_ubyte3_e32 v56, v56
	v_cvt_f32_ubyte3_e32 v66, v66
	v_cvt_f32_ubyte3_e32 v76, v76
	v_cvt_f32_ubyte0_e32 v41, v47
	v_cvt_f32_ubyte0_e32 v51, v57
	v_cvt_f32_ubyte0_e32 v61, v67
	v_cvt_f32_ubyte0_e32 v71, v77
	v_cvt_f32_ubyte1_e32 v43, v47
	v_cvt_f32_ubyte1_e32 v53, v57
	v_cvt_f32_ubyte1_e32 v63, v67
	v_cvt_f32_ubyte1_e32 v73, v77
	v_cvt_f32_ubyte2_e32 v45, v47
	v_cvt_f32_ubyte2_e32 v55, v57
	v_cvt_f32_ubyte2_e32 v65, v67
	v_cvt_f32_ubyte2_e32 v75, v77
	v_cvt_f32_ubyte3_e32 v47, v47
	v_cvt_f32_ubyte3_e32 v57, v57
	v_cvt_f32_ubyte3_e32 v67, v67
	v_cvt_f32_ubyte3_e32 v77, v77
	v_fma_mix_f32 v40, v8, v40, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v9, v50, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v10, v60, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v70, v11, v70, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v41, v8, v41, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v9, v51, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v10, v61, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v71, v11, v71, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v42, v8, v42, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v52, v9, v52, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v62, v10, v62, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v72, v11, v72, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v43, v8, v43, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v53, v9, v53, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v63, v10, v63, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v73, v11, v73, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v44, v8, v44, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v54, v9, v54, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v64, v10, v64, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v74, v11, v74, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v8, v45, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v9, v55, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v65, v10, v65, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v75, v11, v75, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v8, v46, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v9, v56, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v66, v10, v66, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v76, v11, v76, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v8, v47, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v9, v57, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v67, v10, v67, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v77, v11, v77, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	buffer_load_b128 v[96:99], v3, s[24:27], s28 offen offset:1024
	buffer_load_b128 v[100:103], v3, s[24:27], s28 offen offset:1040
	buffer_load_b128 v[104:107], v3, s[24:27], s29 offen offset:1024
	buffer_load_b128 v[108:111], v3, s[24:27], s29 offen offset:1040
	buffer_load_b128 v[112:115], v3, s[24:27], s28 offen offset:2048
	buffer_load_b128 v[116:119], v3, s[24:27], s28 offen offset:2064
	buffer_load_b128 v[120:123], v3, s[24:27], s29 offen offset:2048
	buffer_load_b128 v[124:127], v3, s[24:27], s29 offen offset:2064
	buffer_load_b128 v[128:131], v3, s[24:27], s28 offen offset:3072
	buffer_load_b128 v[132:135], v3, s[24:27], s28 offen offset:3088
	buffer_load_b128 v[136:139], v3, s[24:27], s29 offen offset:3072
	buffer_load_b128 v[140:143], v3, s[24:27], s29 offen offset:3088
	s_wait_loadcnt 0x27
	v_dual_mul_f32 v48, v41, v81 :: v_dual_mul_f32 v49, v51, v81
	v_dual_mul_f32 v58, v61, v81 :: v_dual_mul_f32 v59, v71, v81
	s_wait_loadcnt 0x25
	v_dual_mul_f32 v68, v41, v89 :: v_dual_mul_f32 v69, v51, v89
	v_dual_mul_f32 v78, v61, v89 :: v_dual_mul_f32 v79, v71, v89
	v_dual_fmac_f32 v48, v40, v80 :: v_dual_fmac_f32 v49, v50, v80
	v_dual_fmac_f32 v58, v60, v80 :: v_dual_fmac_f32 v59, v70, v80
	v_dual_fmac_f32 v68, v40, v88 :: v_dual_fmac_f32 v69, v50, v88
	v_dual_fmac_f32 v78, v60, v88 :: v_dual_fmac_f32 v79, v70, v88
	v_dual_fmac_f32 v48, v42, v82 :: v_dual_fmac_f32 v49, v52, v82
	v_dual_fmac_f32 v58, v62, v82 :: v_dual_fmac_f32 v59, v72, v82
	v_dual_fmac_f32 v68, v42, v90 :: v_dual_fmac_f32 v69, v52, v90
	v_dual_fmac_f32 v78, v62, v90 :: v_dual_fmac_f32 v79, v72, v90
	v_dual_fmac_f32 v48, v43, v83 :: v_dual_fmac_f32 v49, v53, v83
	v_dual_fmac_f32 v58, v63, v83 :: v_dual_fmac_f32 v59, v73, v83
	v_dual_fmac_f32 v68, v43, v91 :: v_dual_fmac_f32 v69, v53, v91
	v_dual_fmac_f32 v78, v63, v91 :: v_dual_fmac_f32 v79, v73, v91
	v_dual_fmac_f32 v48, v44, v84 :: v_dual_fmac_f32 v49, v54, v84
	v_dual_fmac_f32 v58, v64, v84 :: v_dual_fmac_f32 v59, v74, v84
	s_wait_loadcnt 0x24
	v_dual_fmac_f32 v68, v44, v92 :: v_dual_fmac_f32 v69, v54, v92
	v_dual_fmac_f32 v78, v64, v92 :: v_dual_fmac_f32 v79, v74, v92
	v_dual_fmac_f32 v48, v45, v85 :: v_dual_fmac_f32 v49, v55, v85
	v_dual_fmac_f32 v58, v65, v85 :: v_dual_fmac_f32 v59, v75, v85
	v_dual_fmac_f32 v68, v45, v93 :: v_dual_fmac_f32 v69, v55, v93
	v_dual_fmac_f32 v78, v65, v93 :: v_dual_fmac_f32 v79, v75, v93
	v_dual_fmac_f32 v48, v46, v86 :: v_dual_fmac_f32 v49, v56, v86
	v_dual_fmac_f32 v58, v66, v86 :: v_dual_fmac_f32 v59, v76, v86
	v_dual_fmac_f32 v68, v46, v94 :: v_dual_fmac_f32 v69, v56, v94
	v_dual_fmac_f32 v78, v66, v94 :: v_dual_fmac_f32 v79, v76, v94
	v_dual_fmac_f32 v48, v47, v87 :: v_dual_fmac_f32 v49, v57, v87
	v_dual_fmac_f32 v58, v67, v87 :: v_dual_fmac_f32 v59, v77, v87
	v_dual_fmac_f32 v68, v47, v95 :: v_dual_fmac_f32 v69, v57, v95
	v_dual_fmac_f32 v78, v67, v95 :: v_dual_fmac_f32 v79, v77, v95
	v_dual_add_f32 v144, v144, v48 :: v_dual_add_f32 v145, v145, v49
	v_dual_add_f32 v146, v146, v58 :: v_dual_add_f32 v147, v147, v59
	v_dual_add_f32 v160, v160, v68 :: v_dual_add_f32 v161, v161, v69
	v_dual_add_f32 v162, v162, v78 :: v_dual_add_f32 v163, v163, v79
	s_wait_loadcnt 0x1f
	v_and_b32_e32 v46, 0xf0f0f0f, v28
	s_wait_loadcnt 0x1e
	v_and_b32_e32 v56, 0xf0f0f0f, v29
	s_wait_loadcnt 0x1d
	v_and_b32_e32 v66, 0xf0f0f0f, v30
	s_wait_loadcnt 0x1c
	v_and_b32_e32 v76, 0xf0f0f0f, v31
	v_lshrrev_b32_e32 v47, 4, v28
	v_lshrrev_b32_e32 v57, 4, v29
	v_lshrrev_b32_e32 v67, 4, v30
	v_lshrrev_b32_e32 v77, 4, v31
	v_and_b32_e32 v47, 0xf0f0f0f, v47
	v_and_b32_e32 v57, 0xf0f0f0f, v57
	v_and_b32_e32 v67, 0xf0f0f0f, v67
	v_and_b32_e32 v77, 0xf0f0f0f, v77
	v_cvt_f32_ubyte0_e32 v40, v46
	v_cvt_f32_ubyte0_e32 v50, v56
	v_cvt_f32_ubyte0_e32 v60, v66
	v_cvt_f32_ubyte0_e32 v70, v76
	v_cvt_f32_ubyte1_e32 v42, v46
	v_cvt_f32_ubyte1_e32 v52, v56
	v_cvt_f32_ubyte1_e32 v62, v66
	v_cvt_f32_ubyte1_e32 v72, v76
	v_cvt_f32_ubyte2_e32 v44, v46
	v_cvt_f32_ubyte2_e32 v54, v56
	v_cvt_f32_ubyte2_e32 v64, v66
	v_cvt_f32_ubyte2_e32 v74, v76
	v_cvt_f32_ubyte3_e32 v46, v46
	v_cvt_f32_ubyte3_e32 v56, v56
	v_cvt_f32_ubyte3_e32 v66, v66
	v_cvt_f32_ubyte3_e32 v76, v76
	v_cvt_f32_ubyte0_e32 v41, v47
	v_cvt_f32_ubyte0_e32 v51, v57
	v_cvt_f32_ubyte0_e32 v61, v67
	v_cvt_f32_ubyte0_e32 v71, v77
	v_cvt_f32_ubyte1_e32 v43, v47
	v_cvt_f32_ubyte1_e32 v53, v57
	v_cvt_f32_ubyte1_e32 v63, v67
	v_cvt_f32_ubyte1_e32 v73, v77
	v_cvt_f32_ubyte2_e32 v45, v47
	v_cvt_f32_ubyte2_e32 v55, v57
	v_cvt_f32_ubyte2_e32 v65, v67
	v_cvt_f32_ubyte2_e32 v75, v77
	v_cvt_f32_ubyte3_e32 v47, v47
	v_cvt_f32_ubyte3_e32 v57, v57
	v_cvt_f32_ubyte3_e32 v67, v67
	v_cvt_f32_ubyte3_e32 v77, v77
	v_fma_mix_f32 v40, v12, v40, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v13, v50, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v14, v60, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v70, v15, v70, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v41, v12, v41, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v13, v51, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v14, v61, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v71, v15, v71, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v42, v12, v42, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v52, v13, v52, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v62, v14, v62, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v72, v15, v72, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v43, v12, v43, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v53, v13, v53, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v63, v14, v63, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v73, v15, v73, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v44, v12, v44, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v54, v13, v54, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v64, v14, v64, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v74, v15, v74, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v12, v45, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v13, v55, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v65, v14, v65, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v75, v15, v75, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v12, v46, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v13, v56, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v66, v14, v66, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v76, v15, v76, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v12, v47, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v13, v57, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v67, v14, v67, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v77, v15, v77, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xb
	v_dual_mul_f32 v48, v41, v97 :: v_dual_mul_f32 v49, v51, v97
	v_dual_mul_f32 v58, v61, v97 :: v_dual_mul_f32 v59, v71, v97
	s_wait_loadcnt 0x9
	v_dual_mul_f32 v68, v41, v105 :: v_dual_mul_f32 v69, v51, v105
	v_dual_mul_f32 v78, v61, v105 :: v_dual_mul_f32 v79, v71, v105
	v_dual_fmac_f32 v48, v40, v96 :: v_dual_fmac_f32 v49, v50, v96
	v_dual_fmac_f32 v58, v60, v96 :: v_dual_fmac_f32 v59, v70, v96
	v_dual_fmac_f32 v68, v40, v104 :: v_dual_fmac_f32 v69, v50, v104
	v_dual_fmac_f32 v78, v60, v104 :: v_dual_fmac_f32 v79, v70, v104
	v_dual_fmac_f32 v48, v42, v98 :: v_dual_fmac_f32 v49, v52, v98
	v_dual_fmac_f32 v58, v62, v98 :: v_dual_fmac_f32 v59, v72, v98
	v_dual_fmac_f32 v68, v42, v106 :: v_dual_fmac_f32 v69, v52, v106
	v_dual_fmac_f32 v78, v62, v106 :: v_dual_fmac_f32 v79, v72, v106
	v_dual_fmac_f32 v48, v43, v99 :: v_dual_fmac_f32 v49, v53, v99
	v_dual_fmac_f32 v58, v63, v99 :: v_dual_fmac_f32 v59, v73, v99
	v_dual_fmac_f32 v68, v43, v107 :: v_dual_fmac_f32 v69, v53, v107
	v_dual_fmac_f32 v78, v63, v107 :: v_dual_fmac_f32 v79, v73, v107
	v_dual_fmac_f32 v48, v44, v100 :: v_dual_fmac_f32 v49, v54, v100
	v_dual_fmac_f32 v58, v64, v100 :: v_dual_fmac_f32 v59, v74, v100
	s_wait_loadcnt 0x8
	v_dual_fmac_f32 v68, v44, v108 :: v_dual_fmac_f32 v69, v54, v108
	v_dual_fmac_f32 v78, v64, v108 :: v_dual_fmac_f32 v79, v74, v108
	v_dual_fmac_f32 v48, v45, v101 :: v_dual_fmac_f32 v49, v55, v101
	v_dual_fmac_f32 v58, v65, v101 :: v_dual_fmac_f32 v59, v75, v101
	v_dual_fmac_f32 v68, v45, v109 :: v_dual_fmac_f32 v69, v55, v109
	v_dual_fmac_f32 v78, v65, v109 :: v_dual_fmac_f32 v79, v75, v109
	v_dual_fmac_f32 v48, v46, v102 :: v_dual_fmac_f32 v49, v56, v102
	v_dual_fmac_f32 v58, v66, v102 :: v_dual_fmac_f32 v59, v76, v102
	v_dual_fmac_f32 v68, v46, v110 :: v_dual_fmac_f32 v69, v56, v110
	v_dual_fmac_f32 v78, v66, v110 :: v_dual_fmac_f32 v79, v76, v110
	v_dual_fmac_f32 v48, v47, v103 :: v_dual_fmac_f32 v49, v57, v103
	v_dual_fmac_f32 v58, v67, v103 :: v_dual_fmac_f32 v59, v77, v103
	v_dual_fmac_f32 v68, v47, v111 :: v_dual_fmac_f32 v69, v57, v111
	v_dual_fmac_f32 v78, v67, v111 :: v_dual_fmac_f32 v79, v77, v111
	v_dual_add_f32 v148, v148, v48 :: v_dual_add_f32 v149, v149, v49
	v_dual_add_f32 v150, v150, v58 :: v_dual_add_f32 v151, v151, v59
	v_dual_add_f32 v164, v164, v68 :: v_dual_add_f32 v165, v165, v69
	v_dual_add_f32 v166, v166, v78 :: v_dual_add_f32 v167, v167, v79
	v_and_b32_e32 v46, 0xf0f0f0f, v32
	v_and_b32_e32 v56, 0xf0f0f0f, v33
	v_and_b32_e32 v66, 0xf0f0f0f, v34
	v_and_b32_e32 v76, 0xf0f0f0f, v35
	v_lshrrev_b32_e32 v47, 4, v32
	v_lshrrev_b32_e32 v57, 4, v33
	v_lshrrev_b32_e32 v67, 4, v34
	v_lshrrev_b32_e32 v77, 4, v35
	v_and_b32_e32 v47, 0xf0f0f0f, v47
	v_and_b32_e32 v57, 0xf0f0f0f, v57
	v_and_b32_e32 v67, 0xf0f0f0f, v67
	v_and_b32_e32 v77, 0xf0f0f0f, v77
	v_cvt_f32_ubyte0_e32 v40, v46
	v_cvt_f32_ubyte0_e32 v50, v56
	v_cvt_f32_ubyte0_e32 v60, v66
	v_cvt_f32_ubyte0_e32 v70, v76
	v_cvt_f32_ubyte1_e32 v42, v46
	v_cvt_f32_ubyte1_e32 v52, v56
	v_cvt_f32_ubyte1_e32 v62, v66
	v_cvt_f32_ubyte1_e32 v72, v76
	v_cvt_f32_ubyte2_e32 v44, v46
	v_cvt_f32_ubyte2_e32 v54, v56
	v_cvt_f32_ubyte2_e32 v64, v66
	v_cvt_f32_ubyte2_e32 v74, v76
	v_cvt_f32_ubyte3_e32 v46, v46
	v_cvt_f32_ubyte3_e32 v56, v56
	v_cvt_f32_ubyte3_e32 v66, v66
	v_cvt_f32_ubyte3_e32 v76, v76
	v_cvt_f32_ubyte0_e32 v41, v47
	v_cvt_f32_ubyte0_e32 v51, v57
	v_cvt_f32_ubyte0_e32 v61, v67
	v_cvt_f32_ubyte0_e32 v71, v77
	v_cvt_f32_ubyte1_e32 v43, v47
	v_cvt_f32_ubyte1_e32 v53, v57
	v_cvt_f32_ubyte1_e32 v63, v67
	v_cvt_f32_ubyte1_e32 v73, v77
	v_cvt_f32_ubyte2_e32 v45, v47
	v_cvt_f32_ubyte2_e32 v55, v57
	v_cvt_f32_ubyte2_e32 v65, v67
	v_cvt_f32_ubyte2_e32 v75, v77
	v_cvt_f32_ubyte3_e32 v47, v47
	v_cvt_f32_ubyte3_e32 v57, v57
	v_cvt_f32_ubyte3_e32 v67, v67
	v_cvt_f32_ubyte3_e32 v77, v77
	v_fma_mix_f32 v40, v16, v40, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v17, v50, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v18, v60, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v70, v19, v70, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v41, v16, v41, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v17, v51, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v18, v61, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v71, v19, v71, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v42, v16, v42, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v52, v17, v52, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v62, v18, v62, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v72, v19, v72, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v43, v16, v43, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v53, v17, v53, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v63, v18, v63, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v73, v19, v73, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v44, v16, v44, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v54, v17, v54, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v64, v18, v64, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v74, v19, v74, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v16, v45, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v17, v55, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v65, v18, v65, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v75, v19, v75, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v16, v46, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v17, v56, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v66, v18, v66, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v76, v19, v76, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v16, v47, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v17, v57, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v67, v18, v67, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v77, v19, v77, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x7
	v_dual_mul_f32 v48, v41, v113 :: v_dual_mul_f32 v49, v51, v113
	v_dual_mul_f32 v58, v61, v113 :: v_dual_mul_f32 v59, v71, v113
	s_wait_loadcnt 0x5
	v_dual_mul_f32 v68, v41, v121 :: v_dual_mul_f32 v69, v51, v121
	v_dual_mul_f32 v78, v61, v121 :: v_dual_mul_f32 v79, v71, v121
	v_dual_fmac_f32 v48, v40, v112 :: v_dual_fmac_f32 v49, v50, v112
	v_dual_fmac_f32 v58, v60, v112 :: v_dual_fmac_f32 v59, v70, v112
	v_dual_fmac_f32 v68, v40, v120 :: v_dual_fmac_f32 v69, v50, v120
	v_dual_fmac_f32 v78, v60, v120 :: v_dual_fmac_f32 v79, v70, v120
	v_dual_fmac_f32 v48, v42, v114 :: v_dual_fmac_f32 v49, v52, v114
	v_dual_fmac_f32 v58, v62, v114 :: v_dual_fmac_f32 v59, v72, v114
	v_dual_fmac_f32 v68, v42, v122 :: v_dual_fmac_f32 v69, v52, v122
	v_dual_fmac_f32 v78, v62, v122 :: v_dual_fmac_f32 v79, v72, v122
	v_dual_fmac_f32 v48, v43, v115 :: v_dual_fmac_f32 v49, v53, v115
	v_dual_fmac_f32 v58, v63, v115 :: v_dual_fmac_f32 v59, v73, v115
	v_dual_fmac_f32 v68, v43, v123 :: v_dual_fmac_f32 v69, v53, v123
	v_dual_fmac_f32 v78, v63, v123 :: v_dual_fmac_f32 v79, v73, v123
	v_dual_fmac_f32 v48, v44, v116 :: v_dual_fmac_f32 v49, v54, v116
	v_dual_fmac_f32 v58, v64, v116 :: v_dual_fmac_f32 v59, v74, v116
	s_wait_loadcnt 0x4
	v_dual_fmac_f32 v68, v44, v124 :: v_dual_fmac_f32 v69, v54, v124
	v_dual_fmac_f32 v78, v64, v124 :: v_dual_fmac_f32 v79, v74, v124
	v_dual_fmac_f32 v48, v45, v117 :: v_dual_fmac_f32 v49, v55, v117
	v_dual_fmac_f32 v58, v65, v117 :: v_dual_fmac_f32 v59, v75, v117
	v_dual_fmac_f32 v68, v45, v125 :: v_dual_fmac_f32 v69, v55, v125
	v_dual_fmac_f32 v78, v65, v125 :: v_dual_fmac_f32 v79, v75, v125
	v_dual_fmac_f32 v48, v46, v118 :: v_dual_fmac_f32 v49, v56, v118
	v_dual_fmac_f32 v58, v66, v118 :: v_dual_fmac_f32 v59, v76, v118
	v_dual_fmac_f32 v68, v46, v126 :: v_dual_fmac_f32 v69, v56, v126
	v_dual_fmac_f32 v78, v66, v126 :: v_dual_fmac_f32 v79, v76, v126
	v_dual_fmac_f32 v48, v47, v119 :: v_dual_fmac_f32 v49, v57, v119
	v_dual_fmac_f32 v58, v67, v119 :: v_dual_fmac_f32 v59, v77, v119
	v_dual_fmac_f32 v68, v47, v127 :: v_dual_fmac_f32 v69, v57, v127
	v_dual_fmac_f32 v78, v67, v127 :: v_dual_fmac_f32 v79, v77, v127
	v_dual_add_f32 v152, v152, v48 :: v_dual_add_f32 v153, v153, v49
	v_dual_add_f32 v154, v154, v58 :: v_dual_add_f32 v155, v155, v59
	v_dual_add_f32 v168, v168, v68 :: v_dual_add_f32 v169, v169, v69
	v_dual_add_f32 v170, v170, v78 :: v_dual_add_f32 v171, v171, v79
	v_and_b32_e32 v46, 0xf0f0f0f, v36
	v_and_b32_e32 v56, 0xf0f0f0f, v37
	v_and_b32_e32 v66, 0xf0f0f0f, v38
	v_and_b32_e32 v76, 0xf0f0f0f, v39
	v_lshrrev_b32_e32 v47, 4, v36
	v_lshrrev_b32_e32 v57, 4, v37
	v_lshrrev_b32_e32 v67, 4, v38
	v_lshrrev_b32_e32 v77, 4, v39
	v_and_b32_e32 v47, 0xf0f0f0f, v47
	v_and_b32_e32 v57, 0xf0f0f0f, v57
	v_and_b32_e32 v67, 0xf0f0f0f, v67
	v_and_b32_e32 v77, 0xf0f0f0f, v77
	v_cvt_f32_ubyte0_e32 v40, v46
	v_cvt_f32_ubyte0_e32 v50, v56
	v_cvt_f32_ubyte0_e32 v60, v66
	v_cvt_f32_ubyte0_e32 v70, v76
	v_cvt_f32_ubyte1_e32 v42, v46
	v_cvt_f32_ubyte1_e32 v52, v56
	v_cvt_f32_ubyte1_e32 v62, v66
	v_cvt_f32_ubyte1_e32 v72, v76
	v_cvt_f32_ubyte2_e32 v44, v46
	v_cvt_f32_ubyte2_e32 v54, v56
	v_cvt_f32_ubyte2_e32 v64, v66
	v_cvt_f32_ubyte2_e32 v74, v76
	v_cvt_f32_ubyte3_e32 v46, v46
	v_cvt_f32_ubyte3_e32 v56, v56
	v_cvt_f32_ubyte3_e32 v66, v66
	v_cvt_f32_ubyte3_e32 v76, v76
	v_cvt_f32_ubyte0_e32 v41, v47
	v_cvt_f32_ubyte0_e32 v51, v57
	v_cvt_f32_ubyte0_e32 v61, v67
	v_cvt_f32_ubyte0_e32 v71, v77
	v_cvt_f32_ubyte1_e32 v43, v47
	v_cvt_f32_ubyte1_e32 v53, v57
	v_cvt_f32_ubyte1_e32 v63, v67
	v_cvt_f32_ubyte1_e32 v73, v77
	v_cvt_f32_ubyte2_e32 v45, v47
	v_cvt_f32_ubyte2_e32 v55, v57
	v_cvt_f32_ubyte2_e32 v65, v67
	v_cvt_f32_ubyte2_e32 v75, v77
	v_cvt_f32_ubyte3_e32 v47, v47
	v_cvt_f32_ubyte3_e32 v57, v57
	v_cvt_f32_ubyte3_e32 v67, v67
	v_cvt_f32_ubyte3_e32 v77, v77
	v_fma_mix_f32 v40, v20, v40, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v21, v50, v21 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v22, v60, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v70, v23, v70, v23 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v41, v20, v41, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v21, v51, v21 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v22, v61, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v71, v23, v71, v23 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v42, v20, v42, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v52, v21, v52, v21 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v62, v22, v62, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v72, v23, v72, v23 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v43, v20, v43, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v53, v21, v53, v21 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v63, v22, v63, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v73, v23, v73, v23 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v44, v20, v44, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v54, v21, v54, v21 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v64, v22, v64, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v74, v23, v74, v23 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v20, v45, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v21, v55, v21 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v65, v22, v65, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v75, v23, v75, v23 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v20, v46, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v21, v56, v21 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v66, v22, v66, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v76, v23, v76, v23 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v20, v47, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v21, v57, v21 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v67, v22, v67, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v77, v23, v77, v23 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x3
	v_dual_mul_f32 v48, v41, v129 :: v_dual_mul_f32 v49, v51, v129
	v_dual_mul_f32 v58, v61, v129 :: v_dual_mul_f32 v59, v71, v129
	s_wait_loadcnt 0x1
	v_dual_mul_f32 v68, v41, v137 :: v_dual_mul_f32 v69, v51, v137
	v_dual_mul_f32 v78, v61, v137 :: v_dual_mul_f32 v79, v71, v137
	v_dual_fmac_f32 v48, v40, v128 :: v_dual_fmac_f32 v49, v50, v128
	v_dual_fmac_f32 v58, v60, v128 :: v_dual_fmac_f32 v59, v70, v128
	v_dual_fmac_f32 v68, v40, v136 :: v_dual_fmac_f32 v69, v50, v136
	v_dual_fmac_f32 v78, v60, v136 :: v_dual_fmac_f32 v79, v70, v136
	v_dual_fmac_f32 v48, v42, v130 :: v_dual_fmac_f32 v49, v52, v130
	v_dual_fmac_f32 v58, v62, v130 :: v_dual_fmac_f32 v59, v72, v130
	v_dual_fmac_f32 v68, v42, v138 :: v_dual_fmac_f32 v69, v52, v138
	v_dual_fmac_f32 v78, v62, v138 :: v_dual_fmac_f32 v79, v72, v138
	v_dual_fmac_f32 v48, v43, v131 :: v_dual_fmac_f32 v49, v53, v131
	v_dual_fmac_f32 v58, v63, v131 :: v_dual_fmac_f32 v59, v73, v131
	v_dual_fmac_f32 v68, v43, v139 :: v_dual_fmac_f32 v69, v53, v139
	v_dual_fmac_f32 v78, v63, v139 :: v_dual_fmac_f32 v79, v73, v139
	v_dual_fmac_f32 v48, v44, v132 :: v_dual_fmac_f32 v49, v54, v132
	v_dual_fmac_f32 v58, v64, v132 :: v_dual_fmac_f32 v59, v74, v132
	s_wait_loadcnt 0x0
	v_dual_fmac_f32 v68, v44, v140 :: v_dual_fmac_f32 v69, v54, v140
	v_dual_fmac_f32 v78, v64, v140 :: v_dual_fmac_f32 v79, v74, v140
	v_dual_fmac_f32 v48, v45, v133 :: v_dual_fmac_f32 v49, v55, v133
	v_dual_fmac_f32 v58, v65, v133 :: v_dual_fmac_f32 v59, v75, v133
	v_dual_fmac_f32 v68, v45, v141 :: v_dual_fmac_f32 v69, v55, v141
	v_dual_fmac_f32 v78, v65, v141 :: v_dual_fmac_f32 v79, v75, v141
	v_dual_fmac_f32 v48, v46, v134 :: v_dual_fmac_f32 v49, v56, v134
	v_dual_fmac_f32 v58, v66, v134 :: v_dual_fmac_f32 v59, v76, v134
	v_dual_fmac_f32 v68, v46, v142 :: v_dual_fmac_f32 v69, v56, v142
	v_dual_fmac_f32 v78, v66, v142 :: v_dual_fmac_f32 v79, v76, v142
	v_dual_fmac_f32 v48, v47, v135 :: v_dual_fmac_f32 v49, v57, v135
	v_dual_fmac_f32 v58, v67, v135 :: v_dual_fmac_f32 v59, v77, v135
	v_dual_fmac_f32 v68, v47, v143 :: v_dual_fmac_f32 v69, v57, v143
	v_dual_fmac_f32 v78, v67, v143 :: v_dual_fmac_f32 v79, v77, v143
	v_dual_add_f32 v156, v156, v48 :: v_dual_add_f32 v157, v157, v49
	v_dual_add_f32 v158, v158, v58 :: v_dual_add_f32 v159, v159, v59
	v_dual_add_f32 v172, v172, v68 :: v_dual_add_f32 v173, v173, v69
	v_dual_add_f32 v174, v174, v78 :: v_dual_add_f32 v175, v175, v79
	s_add_co_i32 s16, s16, 0x220
	s_add_co_i32 s17, s17, 0x220
	s_add_co_i32 s18, s18, 0x220
	s_add_co_i32 s19, s19, 0x220
	s_add_co_i32 s28, s28, 0x1000
	s_add_co_i32 s29, s29, 0x1000
	s_add_co_i32 s36, s36, -1
	s_cmp_lg_u32 s36, 0
	s_cbranch_scc1 .Lpx_b2_quad
	.Lpx_b2_tail:
	s_cmp_gt_u32 s37, 0
	s_cbranch_scc0 .Lpx_b2_fold
	buffer_load_b32 v8, v1, s[20:23], s16 offen
	buffer_load_b32 v9, v1, s[20:23], s17 offen
	buffer_load_b32 v10, v1, s[20:23], s18 offen
	buffer_load_b32 v11, v1, s[20:23], s19 offen
	buffer_load_b32 v24, v2, s[20:23], s16 offen offset:8
	buffer_load_b32 v25, v2, s[20:23], s17 offen offset:8
	buffer_load_b32 v26, v2, s[20:23], s18 offen offset:8
	buffer_load_b32 v27, v2, s[20:23], s19 offen offset:8
	buffer_load_b128 v[80:83], v3, s[24:27], s28 offen
	buffer_load_b128 v[84:87], v3, s[24:27], s28 offen offset:16
	buffer_load_b128 v[88:91], v3, s[24:27], s29 offen
	buffer_load_b128 v[92:95], v3, s[24:27], s29 offen offset:16
	s_wait_loadcnt 0x7
	v_and_b32_e32 v46, 0xf0f0f0f, v24
	s_wait_loadcnt 0x6
	v_and_b32_e32 v56, 0xf0f0f0f, v25
	s_wait_loadcnt 0x5
	v_and_b32_e32 v66, 0xf0f0f0f, v26
	s_wait_loadcnt 0x4
	v_and_b32_e32 v76, 0xf0f0f0f, v27
	v_lshrrev_b32_e32 v47, 4, v24
	v_lshrrev_b32_e32 v57, 4, v25
	v_lshrrev_b32_e32 v67, 4, v26
	v_lshrrev_b32_e32 v77, 4, v27
	v_and_b32_e32 v47, 0xf0f0f0f, v47
	v_and_b32_e32 v57, 0xf0f0f0f, v57
	v_and_b32_e32 v67, 0xf0f0f0f, v67
	v_and_b32_e32 v77, 0xf0f0f0f, v77
	v_cvt_f32_ubyte0_e32 v40, v46
	v_cvt_f32_ubyte0_e32 v50, v56
	v_cvt_f32_ubyte0_e32 v60, v66
	v_cvt_f32_ubyte0_e32 v70, v76
	v_cvt_f32_ubyte1_e32 v42, v46
	v_cvt_f32_ubyte1_e32 v52, v56
	v_cvt_f32_ubyte1_e32 v62, v66
	v_cvt_f32_ubyte1_e32 v72, v76
	v_cvt_f32_ubyte2_e32 v44, v46
	v_cvt_f32_ubyte2_e32 v54, v56
	v_cvt_f32_ubyte2_e32 v64, v66
	v_cvt_f32_ubyte2_e32 v74, v76
	v_cvt_f32_ubyte3_e32 v46, v46
	v_cvt_f32_ubyte3_e32 v56, v56
	v_cvt_f32_ubyte3_e32 v66, v66
	v_cvt_f32_ubyte3_e32 v76, v76
	v_cvt_f32_ubyte0_e32 v41, v47
	v_cvt_f32_ubyte0_e32 v51, v57
	v_cvt_f32_ubyte0_e32 v61, v67
	v_cvt_f32_ubyte0_e32 v71, v77
	v_cvt_f32_ubyte1_e32 v43, v47
	v_cvt_f32_ubyte1_e32 v53, v57
	v_cvt_f32_ubyte1_e32 v63, v67
	v_cvt_f32_ubyte1_e32 v73, v77
	v_cvt_f32_ubyte2_e32 v45, v47
	v_cvt_f32_ubyte2_e32 v55, v57
	v_cvt_f32_ubyte2_e32 v65, v67
	v_cvt_f32_ubyte2_e32 v75, v77
	v_cvt_f32_ubyte3_e32 v47, v47
	v_cvt_f32_ubyte3_e32 v57, v57
	v_cvt_f32_ubyte3_e32 v67, v67
	v_cvt_f32_ubyte3_e32 v77, v77
	v_fma_mix_f32 v40, v8, v40, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v9, v50, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v10, v60, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v70, v11, v70, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v41, v8, v41, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v9, v51, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v10, v61, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v71, v11, v71, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v42, v8, v42, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v52, v9, v52, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v62, v10, v62, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v72, v11, v72, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v43, v8, v43, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v53, v9, v53, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v63, v10, v63, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v73, v11, v73, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v44, v8, v44, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v54, v9, v54, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v64, v10, v64, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v74, v11, v74, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v8, v45, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v9, v55, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v65, v10, v65, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v75, v11, v75, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v8, v46, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v9, v56, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v66, v10, v66, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v76, v11, v76, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v8, v47, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v9, v57, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v67, v10, v67, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v77, v11, v77, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x3
	v_dual_mul_f32 v48, v41, v81 :: v_dual_mul_f32 v49, v51, v81
	v_dual_mul_f32 v58, v61, v81 :: v_dual_mul_f32 v59, v71, v81
	s_wait_loadcnt 0x1
	v_dual_mul_f32 v68, v41, v89 :: v_dual_mul_f32 v69, v51, v89
	v_dual_mul_f32 v78, v61, v89 :: v_dual_mul_f32 v79, v71, v89
	v_dual_fmac_f32 v48, v40, v80 :: v_dual_fmac_f32 v49, v50, v80
	v_dual_fmac_f32 v58, v60, v80 :: v_dual_fmac_f32 v59, v70, v80
	v_dual_fmac_f32 v68, v40, v88 :: v_dual_fmac_f32 v69, v50, v88
	v_dual_fmac_f32 v78, v60, v88 :: v_dual_fmac_f32 v79, v70, v88
	v_dual_fmac_f32 v48, v42, v82 :: v_dual_fmac_f32 v49, v52, v82
	v_dual_fmac_f32 v58, v62, v82 :: v_dual_fmac_f32 v59, v72, v82
	v_dual_fmac_f32 v68, v42, v90 :: v_dual_fmac_f32 v69, v52, v90
	v_dual_fmac_f32 v78, v62, v90 :: v_dual_fmac_f32 v79, v72, v90
	v_dual_fmac_f32 v48, v43, v83 :: v_dual_fmac_f32 v49, v53, v83
	v_dual_fmac_f32 v58, v63, v83 :: v_dual_fmac_f32 v59, v73, v83
	v_dual_fmac_f32 v68, v43, v91 :: v_dual_fmac_f32 v69, v53, v91
	v_dual_fmac_f32 v78, v63, v91 :: v_dual_fmac_f32 v79, v73, v91
	v_dual_fmac_f32 v48, v44, v84 :: v_dual_fmac_f32 v49, v54, v84
	v_dual_fmac_f32 v58, v64, v84 :: v_dual_fmac_f32 v59, v74, v84
	s_wait_loadcnt 0x0
	v_dual_fmac_f32 v68, v44, v92 :: v_dual_fmac_f32 v69, v54, v92
	v_dual_fmac_f32 v78, v64, v92 :: v_dual_fmac_f32 v79, v74, v92
	v_dual_fmac_f32 v48, v45, v85 :: v_dual_fmac_f32 v49, v55, v85
	v_dual_fmac_f32 v58, v65, v85 :: v_dual_fmac_f32 v59, v75, v85
	v_dual_fmac_f32 v68, v45, v93 :: v_dual_fmac_f32 v69, v55, v93
	v_dual_fmac_f32 v78, v65, v93 :: v_dual_fmac_f32 v79, v75, v93
	v_dual_fmac_f32 v48, v46, v86 :: v_dual_fmac_f32 v49, v56, v86
	v_dual_fmac_f32 v58, v66, v86 :: v_dual_fmac_f32 v59, v76, v86
	v_dual_fmac_f32 v68, v46, v94 :: v_dual_fmac_f32 v69, v56, v94
	v_dual_fmac_f32 v78, v66, v94 :: v_dual_fmac_f32 v79, v76, v94
	v_dual_fmac_f32 v48, v47, v87 :: v_dual_fmac_f32 v49, v57, v87
	v_dual_fmac_f32 v58, v67, v87 :: v_dual_fmac_f32 v59, v77, v87
	v_dual_fmac_f32 v68, v47, v95 :: v_dual_fmac_f32 v69, v57, v95
	v_dual_fmac_f32 v78, v67, v95 :: v_dual_fmac_f32 v79, v77, v95
	v_dual_add_f32 v144, v144, v48 :: v_dual_add_f32 v145, v145, v49
	v_dual_add_f32 v146, v146, v58 :: v_dual_add_f32 v147, v147, v59
	v_dual_add_f32 v160, v160, v68 :: v_dual_add_f32 v161, v161, v69
	v_dual_add_f32 v162, v162, v78 :: v_dual_add_f32 v163, v163, v79
	s_cmp_gt_u32 s37, 1
	s_cbranch_scc0 .Lpx_b2_fold
	buffer_load_b32 v12, v1, s[20:23], s16 offen offset:136
	buffer_load_b32 v13, v1, s[20:23], s17 offen offset:136
	buffer_load_b32 v14, v1, s[20:23], s18 offen offset:136
	buffer_load_b32 v15, v1, s[20:23], s19 offen offset:136
	buffer_load_b32 v28, v2, s[20:23], s16 offen offset:144
	buffer_load_b32 v29, v2, s[20:23], s17 offen offset:144
	buffer_load_b32 v30, v2, s[20:23], s18 offen offset:144
	buffer_load_b32 v31, v2, s[20:23], s19 offen offset:144
	buffer_load_b128 v[80:83], v3, s[24:27], s28 offen offset:1024
	buffer_load_b128 v[84:87], v3, s[24:27], s28 offen offset:1040
	buffer_load_b128 v[88:91], v3, s[24:27], s29 offen offset:1024
	buffer_load_b128 v[92:95], v3, s[24:27], s29 offen offset:1040
	s_wait_loadcnt 0x7
	v_and_b32_e32 v46, 0xf0f0f0f, v28
	s_wait_loadcnt 0x6
	v_and_b32_e32 v56, 0xf0f0f0f, v29
	s_wait_loadcnt 0x5
	v_and_b32_e32 v66, 0xf0f0f0f, v30
	s_wait_loadcnt 0x4
	v_and_b32_e32 v76, 0xf0f0f0f, v31
	v_lshrrev_b32_e32 v47, 4, v28
	v_lshrrev_b32_e32 v57, 4, v29
	v_lshrrev_b32_e32 v67, 4, v30
	v_lshrrev_b32_e32 v77, 4, v31
	v_and_b32_e32 v47, 0xf0f0f0f, v47
	v_and_b32_e32 v57, 0xf0f0f0f, v57
	v_and_b32_e32 v67, 0xf0f0f0f, v67
	v_and_b32_e32 v77, 0xf0f0f0f, v77
	v_cvt_f32_ubyte0_e32 v40, v46
	v_cvt_f32_ubyte0_e32 v50, v56
	v_cvt_f32_ubyte0_e32 v60, v66
	v_cvt_f32_ubyte0_e32 v70, v76
	v_cvt_f32_ubyte1_e32 v42, v46
	v_cvt_f32_ubyte1_e32 v52, v56
	v_cvt_f32_ubyte1_e32 v62, v66
	v_cvt_f32_ubyte1_e32 v72, v76
	v_cvt_f32_ubyte2_e32 v44, v46
	v_cvt_f32_ubyte2_e32 v54, v56
	v_cvt_f32_ubyte2_e32 v64, v66
	v_cvt_f32_ubyte2_e32 v74, v76
	v_cvt_f32_ubyte3_e32 v46, v46
	v_cvt_f32_ubyte3_e32 v56, v56
	v_cvt_f32_ubyte3_e32 v66, v66
	v_cvt_f32_ubyte3_e32 v76, v76
	v_cvt_f32_ubyte0_e32 v41, v47
	v_cvt_f32_ubyte0_e32 v51, v57
	v_cvt_f32_ubyte0_e32 v61, v67
	v_cvt_f32_ubyte0_e32 v71, v77
	v_cvt_f32_ubyte1_e32 v43, v47
	v_cvt_f32_ubyte1_e32 v53, v57
	v_cvt_f32_ubyte1_e32 v63, v67
	v_cvt_f32_ubyte1_e32 v73, v77
	v_cvt_f32_ubyte2_e32 v45, v47
	v_cvt_f32_ubyte2_e32 v55, v57
	v_cvt_f32_ubyte2_e32 v65, v67
	v_cvt_f32_ubyte2_e32 v75, v77
	v_cvt_f32_ubyte3_e32 v47, v47
	v_cvt_f32_ubyte3_e32 v57, v57
	v_cvt_f32_ubyte3_e32 v67, v67
	v_cvt_f32_ubyte3_e32 v77, v77
	v_fma_mix_f32 v40, v12, v40, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v13, v50, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v14, v60, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v70, v15, v70, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v41, v12, v41, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v13, v51, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v14, v61, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v71, v15, v71, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v42, v12, v42, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v52, v13, v52, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v62, v14, v62, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v72, v15, v72, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v43, v12, v43, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v53, v13, v53, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v63, v14, v63, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v73, v15, v73, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v44, v12, v44, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v54, v13, v54, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v64, v14, v64, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v74, v15, v74, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v12, v45, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v13, v55, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v65, v14, v65, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v75, v15, v75, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v12, v46, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v13, v56, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v66, v14, v66, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v76, v15, v76, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v12, v47, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v13, v57, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v67, v14, v67, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v77, v15, v77, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x3
	v_dual_mul_f32 v48, v41, v81 :: v_dual_mul_f32 v49, v51, v81
	v_dual_mul_f32 v58, v61, v81 :: v_dual_mul_f32 v59, v71, v81
	s_wait_loadcnt 0x1
	v_dual_mul_f32 v68, v41, v89 :: v_dual_mul_f32 v69, v51, v89
	v_dual_mul_f32 v78, v61, v89 :: v_dual_mul_f32 v79, v71, v89
	v_dual_fmac_f32 v48, v40, v80 :: v_dual_fmac_f32 v49, v50, v80
	v_dual_fmac_f32 v58, v60, v80 :: v_dual_fmac_f32 v59, v70, v80
	v_dual_fmac_f32 v68, v40, v88 :: v_dual_fmac_f32 v69, v50, v88
	v_dual_fmac_f32 v78, v60, v88 :: v_dual_fmac_f32 v79, v70, v88
	v_dual_fmac_f32 v48, v42, v82 :: v_dual_fmac_f32 v49, v52, v82
	v_dual_fmac_f32 v58, v62, v82 :: v_dual_fmac_f32 v59, v72, v82
	v_dual_fmac_f32 v68, v42, v90 :: v_dual_fmac_f32 v69, v52, v90
	v_dual_fmac_f32 v78, v62, v90 :: v_dual_fmac_f32 v79, v72, v90
	v_dual_fmac_f32 v48, v43, v83 :: v_dual_fmac_f32 v49, v53, v83
	v_dual_fmac_f32 v58, v63, v83 :: v_dual_fmac_f32 v59, v73, v83
	v_dual_fmac_f32 v68, v43, v91 :: v_dual_fmac_f32 v69, v53, v91
	v_dual_fmac_f32 v78, v63, v91 :: v_dual_fmac_f32 v79, v73, v91
	v_dual_fmac_f32 v48, v44, v84 :: v_dual_fmac_f32 v49, v54, v84
	v_dual_fmac_f32 v58, v64, v84 :: v_dual_fmac_f32 v59, v74, v84
	s_wait_loadcnt 0x0
	v_dual_fmac_f32 v68, v44, v92 :: v_dual_fmac_f32 v69, v54, v92
	v_dual_fmac_f32 v78, v64, v92 :: v_dual_fmac_f32 v79, v74, v92
	v_dual_fmac_f32 v48, v45, v85 :: v_dual_fmac_f32 v49, v55, v85
	v_dual_fmac_f32 v58, v65, v85 :: v_dual_fmac_f32 v59, v75, v85
	v_dual_fmac_f32 v68, v45, v93 :: v_dual_fmac_f32 v69, v55, v93
	v_dual_fmac_f32 v78, v65, v93 :: v_dual_fmac_f32 v79, v75, v93
	v_dual_fmac_f32 v48, v46, v86 :: v_dual_fmac_f32 v49, v56, v86
	v_dual_fmac_f32 v58, v66, v86 :: v_dual_fmac_f32 v59, v76, v86
	v_dual_fmac_f32 v68, v46, v94 :: v_dual_fmac_f32 v69, v56, v94
	v_dual_fmac_f32 v78, v66, v94 :: v_dual_fmac_f32 v79, v76, v94
	v_dual_fmac_f32 v48, v47, v87 :: v_dual_fmac_f32 v49, v57, v87
	v_dual_fmac_f32 v58, v67, v87 :: v_dual_fmac_f32 v59, v77, v87
	v_dual_fmac_f32 v68, v47, v95 :: v_dual_fmac_f32 v69, v57, v95
	v_dual_fmac_f32 v78, v67, v95 :: v_dual_fmac_f32 v79, v77, v95
	v_dual_add_f32 v148, v148, v48 :: v_dual_add_f32 v149, v149, v49
	v_dual_add_f32 v150, v150, v58 :: v_dual_add_f32 v151, v151, v59
	v_dual_add_f32 v164, v164, v68 :: v_dual_add_f32 v165, v165, v69
	v_dual_add_f32 v166, v166, v78 :: v_dual_add_f32 v167, v167, v79
	s_cmp_gt_u32 s37, 2
	s_cbranch_scc0 .Lpx_b2_fold
	buffer_load_b32 v16, v1, s[20:23], s16 offen offset:272
	buffer_load_b32 v17, v1, s[20:23], s17 offen offset:272
	buffer_load_b32 v18, v1, s[20:23], s18 offen offset:272
	buffer_load_b32 v19, v1, s[20:23], s19 offen offset:272
	buffer_load_b32 v32, v2, s[20:23], s16 offen offset:280
	buffer_load_b32 v33, v2, s[20:23], s17 offen offset:280
	buffer_load_b32 v34, v2, s[20:23], s18 offen offset:280
	buffer_load_b32 v35, v2, s[20:23], s19 offen offset:280
	buffer_load_b128 v[80:83], v3, s[24:27], s28 offen offset:2048
	buffer_load_b128 v[84:87], v3, s[24:27], s28 offen offset:2064
	buffer_load_b128 v[88:91], v3, s[24:27], s29 offen offset:2048
	buffer_load_b128 v[92:95], v3, s[24:27], s29 offen offset:2064
	s_wait_loadcnt 0x7
	v_and_b32_e32 v46, 0xf0f0f0f, v32
	s_wait_loadcnt 0x6
	v_and_b32_e32 v56, 0xf0f0f0f, v33
	s_wait_loadcnt 0x5
	v_and_b32_e32 v66, 0xf0f0f0f, v34
	s_wait_loadcnt 0x4
	v_and_b32_e32 v76, 0xf0f0f0f, v35
	v_lshrrev_b32_e32 v47, 4, v32
	v_lshrrev_b32_e32 v57, 4, v33
	v_lshrrev_b32_e32 v67, 4, v34
	v_lshrrev_b32_e32 v77, 4, v35
	v_and_b32_e32 v47, 0xf0f0f0f, v47
	v_and_b32_e32 v57, 0xf0f0f0f, v57
	v_and_b32_e32 v67, 0xf0f0f0f, v67
	v_and_b32_e32 v77, 0xf0f0f0f, v77
	v_cvt_f32_ubyte0_e32 v40, v46
	v_cvt_f32_ubyte0_e32 v50, v56
	v_cvt_f32_ubyte0_e32 v60, v66
	v_cvt_f32_ubyte0_e32 v70, v76
	v_cvt_f32_ubyte1_e32 v42, v46
	v_cvt_f32_ubyte1_e32 v52, v56
	v_cvt_f32_ubyte1_e32 v62, v66
	v_cvt_f32_ubyte1_e32 v72, v76
	v_cvt_f32_ubyte2_e32 v44, v46
	v_cvt_f32_ubyte2_e32 v54, v56
	v_cvt_f32_ubyte2_e32 v64, v66
	v_cvt_f32_ubyte2_e32 v74, v76
	v_cvt_f32_ubyte3_e32 v46, v46
	v_cvt_f32_ubyte3_e32 v56, v56
	v_cvt_f32_ubyte3_e32 v66, v66
	v_cvt_f32_ubyte3_e32 v76, v76
	v_cvt_f32_ubyte0_e32 v41, v47
	v_cvt_f32_ubyte0_e32 v51, v57
	v_cvt_f32_ubyte0_e32 v61, v67
	v_cvt_f32_ubyte0_e32 v71, v77
	v_cvt_f32_ubyte1_e32 v43, v47
	v_cvt_f32_ubyte1_e32 v53, v57
	v_cvt_f32_ubyte1_e32 v63, v67
	v_cvt_f32_ubyte1_e32 v73, v77
	v_cvt_f32_ubyte2_e32 v45, v47
	v_cvt_f32_ubyte2_e32 v55, v57
	v_cvt_f32_ubyte2_e32 v65, v67
	v_cvt_f32_ubyte2_e32 v75, v77
	v_cvt_f32_ubyte3_e32 v47, v47
	v_cvt_f32_ubyte3_e32 v57, v57
	v_cvt_f32_ubyte3_e32 v67, v67
	v_cvt_f32_ubyte3_e32 v77, v77
	v_fma_mix_f32 v40, v16, v40, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v17, v50, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v18, v60, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v70, v19, v70, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v41, v16, v41, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v17, v51, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v18, v61, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v71, v19, v71, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v42, v16, v42, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v52, v17, v52, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v62, v18, v62, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v72, v19, v72, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v43, v16, v43, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v53, v17, v53, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v63, v18, v63, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v73, v19, v73, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v44, v16, v44, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v54, v17, v54, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v64, v18, v64, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v74, v19, v74, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v16, v45, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v17, v55, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v65, v18, v65, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v75, v19, v75, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v16, v46, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v17, v56, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v66, v18, v66, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v76, v19, v76, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v16, v47, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v17, v57, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v67, v18, v67, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v77, v19, v77, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x3
	v_dual_mul_f32 v48, v41, v81 :: v_dual_mul_f32 v49, v51, v81
	v_dual_mul_f32 v58, v61, v81 :: v_dual_mul_f32 v59, v71, v81
	s_wait_loadcnt 0x1
	v_dual_mul_f32 v68, v41, v89 :: v_dual_mul_f32 v69, v51, v89
	v_dual_mul_f32 v78, v61, v89 :: v_dual_mul_f32 v79, v71, v89
	v_dual_fmac_f32 v48, v40, v80 :: v_dual_fmac_f32 v49, v50, v80
	v_dual_fmac_f32 v58, v60, v80 :: v_dual_fmac_f32 v59, v70, v80
	v_dual_fmac_f32 v68, v40, v88 :: v_dual_fmac_f32 v69, v50, v88
	v_dual_fmac_f32 v78, v60, v88 :: v_dual_fmac_f32 v79, v70, v88
	v_dual_fmac_f32 v48, v42, v82 :: v_dual_fmac_f32 v49, v52, v82
	v_dual_fmac_f32 v58, v62, v82 :: v_dual_fmac_f32 v59, v72, v82
	v_dual_fmac_f32 v68, v42, v90 :: v_dual_fmac_f32 v69, v52, v90
	v_dual_fmac_f32 v78, v62, v90 :: v_dual_fmac_f32 v79, v72, v90
	v_dual_fmac_f32 v48, v43, v83 :: v_dual_fmac_f32 v49, v53, v83
	v_dual_fmac_f32 v58, v63, v83 :: v_dual_fmac_f32 v59, v73, v83
	v_dual_fmac_f32 v68, v43, v91 :: v_dual_fmac_f32 v69, v53, v91
	v_dual_fmac_f32 v78, v63, v91 :: v_dual_fmac_f32 v79, v73, v91
	v_dual_fmac_f32 v48, v44, v84 :: v_dual_fmac_f32 v49, v54, v84
	v_dual_fmac_f32 v58, v64, v84 :: v_dual_fmac_f32 v59, v74, v84
	s_wait_loadcnt 0x0
	v_dual_fmac_f32 v68, v44, v92 :: v_dual_fmac_f32 v69, v54, v92
	v_dual_fmac_f32 v78, v64, v92 :: v_dual_fmac_f32 v79, v74, v92
	v_dual_fmac_f32 v48, v45, v85 :: v_dual_fmac_f32 v49, v55, v85
	v_dual_fmac_f32 v58, v65, v85 :: v_dual_fmac_f32 v59, v75, v85
	v_dual_fmac_f32 v68, v45, v93 :: v_dual_fmac_f32 v69, v55, v93
	v_dual_fmac_f32 v78, v65, v93 :: v_dual_fmac_f32 v79, v75, v93
	v_dual_fmac_f32 v48, v46, v86 :: v_dual_fmac_f32 v49, v56, v86
	v_dual_fmac_f32 v58, v66, v86 :: v_dual_fmac_f32 v59, v76, v86
	v_dual_fmac_f32 v68, v46, v94 :: v_dual_fmac_f32 v69, v56, v94
	v_dual_fmac_f32 v78, v66, v94 :: v_dual_fmac_f32 v79, v76, v94
	v_dual_fmac_f32 v48, v47, v87 :: v_dual_fmac_f32 v49, v57, v87
	v_dual_fmac_f32 v58, v67, v87 :: v_dual_fmac_f32 v59, v77, v87
	v_dual_fmac_f32 v68, v47, v95 :: v_dual_fmac_f32 v69, v57, v95
	v_dual_fmac_f32 v78, v67, v95 :: v_dual_fmac_f32 v79, v77, v95
	v_dual_add_f32 v152, v152, v48 :: v_dual_add_f32 v153, v153, v49
	v_dual_add_f32 v154, v154, v58 :: v_dual_add_f32 v155, v155, v59
	v_dual_add_f32 v168, v168, v68 :: v_dual_add_f32 v169, v169, v69
	v_dual_add_f32 v170, v170, v78 :: v_dual_add_f32 v171, v171, v79
	.Lpx_b2_fold:
	v_dual_add_f32 v144, v144, v148 :: v_dual_add_f32 v145, v145, v149
	v_dual_add_f32 v146, v146, v150 :: v_dual_add_f32 v147, v147, v151
	v_dual_add_f32 v152, v152, v156 :: v_dual_add_f32 v153, v153, v157
	v_dual_add_f32 v154, v154, v158 :: v_dual_add_f32 v155, v155, v159
	v_dual_add_f32 v144, v144, v152 :: v_dual_add_f32 v145, v145, v153
	v_dual_add_f32 v146, v146, v154 :: v_dual_add_f32 v147, v147, v155
	v_dual_add_f32 v160, v160, v164 :: v_dual_add_f32 v161, v161, v165
	v_dual_add_f32 v162, v162, v166 :: v_dual_add_f32 v163, v163, v167
	v_dual_add_f32 v168, v168, v172 :: v_dual_add_f32 v169, v169, v173
	v_dual_add_f32 v170, v170, v174 :: v_dual_add_f32 v171, v171, v175
	v_dual_add_f32 v160, v160, v168 :: v_dual_add_f32 v161, v161, v169
	v_dual_add_f32 v162, v162, v170 :: v_dual_add_f32 v163, v163, v171
	ds_swizzle_b32 v8, v144 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v9, v145 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v10, v146 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v11, v147 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v12, v160 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v13, v161 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v14, v162 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v15, v163 offset:swizzle(BITMASK_PERM,"1pppp")
	s_wait_dscnt 0x7
	v_add_f32_e32 v144, v144, v8
	s_wait_dscnt 0x6
	v_add_f32_e32 v145, v145, v9
	s_wait_dscnt 0x5
	v_add_f32_e32 v146, v146, v10
	s_wait_dscnt 0x4
	v_add_f32_e32 v147, v147, v11
	s_wait_dscnt 0x3
	v_add_f32_e32 v160, v160, v12
	s_wait_dscnt 0x2
	v_add_f32_e32 v161, v161, v13
	s_wait_dscnt 0x1
	v_add_f32_e32 v162, v162, v14
	s_wait_dscnt 0x0
	v_add_f32_e32 v163, v163, v15
	v_cmp_gt_u32_e32 vcc_lo, 24, v0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v4, 0, 8, vcc_lo
	v_add_lshl_u32 v4, v4, v0, 2
	ds_bpermute_b32 v8, v4, v144
	ds_bpermute_b32 v9, v4, v145
	ds_bpermute_b32 v10, v4, v146
	ds_bpermute_b32 v11, v4, v147
	ds_bpermute_b32 v12, v4, v160
	ds_bpermute_b32 v13, v4, v161
	ds_bpermute_b32 v14, v4, v162
	ds_bpermute_b32 v15, v4, v163
	s_wait_dscnt 0x7
	v_add_f32_e32 v144, v144, v8
	s_wait_dscnt 0x6
	v_add_f32_e32 v145, v145, v9
	s_wait_dscnt 0x5
	v_add_f32_e32 v146, v146, v10
	s_wait_dscnt 0x4
	v_add_f32_e32 v147, v147, v11
	s_wait_dscnt 0x3
	v_add_f32_e32 v160, v160, v12
	s_wait_dscnt 0x2
	v_add_f32_e32 v161, v161, v13
	s_wait_dscnt 0x1
	v_add_f32_e32 v162, v162, v14
	s_wait_dscnt 0x0
	v_add_f32_e32 v163, v163, v15
	v_cmp_gt_u32_e32 vcc_lo, 28, v0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v4, 0, 4, vcc_lo
	v_add_lshl_u32 v4, v4, v0, 2
	ds_bpermute_b32 v8, v4, v144
	ds_bpermute_b32 v9, v4, v145
	ds_bpermute_b32 v10, v4, v146
	ds_bpermute_b32 v11, v4, v147
	ds_bpermute_b32 v12, v4, v160
	ds_bpermute_b32 v13, v4, v161
	ds_bpermute_b32 v14, v4, v162
	ds_bpermute_b32 v15, v4, v163
	s_wait_dscnt 0x7
	v_add_f32_e32 v144, v144, v8
	s_wait_dscnt 0x6
	v_add_f32_e32 v145, v145, v9
	s_wait_dscnt 0x5
	v_add_f32_e32 v146, v146, v10
	s_wait_dscnt 0x4
	v_add_f32_e32 v147, v147, v11
	s_wait_dscnt 0x3
	v_add_f32_e32 v160, v160, v12
	s_wait_dscnt 0x2
	v_add_f32_e32 v161, v161, v13
	s_wait_dscnt 0x1
	v_add_f32_e32 v162, v162, v14
	s_wait_dscnt 0x0
	v_add_f32_e32 v163, v163, v15
	v_cmp_gt_u32_e32 vcc_lo, 30, v0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v4, 0, 2, vcc_lo
	v_add_lshl_u32 v4, v4, v0, 2
	ds_bpermute_b32 v8, v4, v144
	ds_bpermute_b32 v9, v4, v145
	ds_bpermute_b32 v10, v4, v146
	ds_bpermute_b32 v11, v4, v147
	ds_bpermute_b32 v12, v4, v160
	ds_bpermute_b32 v13, v4, v161
	ds_bpermute_b32 v14, v4, v162
	ds_bpermute_b32 v15, v4, v163
	s_wait_dscnt 0x7
	v_add_f32_e32 v144, v144, v8
	s_wait_dscnt 0x6
	v_add_f32_e32 v145, v145, v9
	s_wait_dscnt 0x5
	v_add_f32_e32 v146, v146, v10
	s_wait_dscnt 0x4
	v_add_f32_e32 v147, v147, v11
	s_wait_dscnt 0x3
	v_add_f32_e32 v160, v160, v12
	s_wait_dscnt 0x2
	v_add_f32_e32 v161, v161, v13
	s_wait_dscnt 0x1
	v_add_f32_e32 v162, v162, v14
	s_wait_dscnt 0x0
	v_add_f32_e32 v163, v163, v15
	v_cmp_gt_u32_e32 vcc_lo, 31, v0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v4, 0, 1, vcc_lo
	v_add_lshl_u32 v4, v4, v0, 2
	ds_bpermute_b32 v8, v4, v144
	ds_bpermute_b32 v9, v4, v145
	ds_bpermute_b32 v10, v4, v146
	ds_bpermute_b32 v11, v4, v147
	ds_bpermute_b32 v12, v4, v160
	ds_bpermute_b32 v13, v4, v161
	ds_bpermute_b32 v14, v4, v162
	ds_bpermute_b32 v15, v4, v163
	s_wait_dscnt 0x7
	v_add_f32_e32 v144, v144, v8
	s_wait_dscnt 0x6
	v_add_f32_e32 v145, v145, v9
	s_wait_dscnt 0x5
	v_add_f32_e32 v146, v146, v10
	s_wait_dscnt 0x4
	v_add_f32_e32 v147, v147, v11
	s_wait_dscnt 0x3
	v_add_f32_e32 v160, v160, v12
	s_wait_dscnt 0x2
	v_add_f32_e32 v161, v161, v13
	s_wait_dscnt 0x1
	v_add_f32_e32 v162, v162, v14
	s_wait_dscnt 0x0
	v_add_f32_e32 v163, v163, v15
	v_cmpx_eq_u32_e32 0, v0
	v_mov_b32_e32 v40, s42
	s_mul_i32 s40, s39, 1
	s_add_co_i32 s40, s40, s42
	v_mov_b32_e32 v41, s40
	global_store_b32 v40, v144, s[8:9]
	global_store_b32 v41, v160, s[8:9]
	s_cmp_le_i32 s13, 1
	s_cbranch_scc1 .Lpx_b2_stored
	global_store_b32 v40, v145, s[8:9] offset:4
	global_store_b32 v41, v161, s[8:9] offset:4
	s_cmp_le_i32 s13, 2
	s_cbranch_scc1 .Lpx_b2_stored
	global_store_b32 v40, v146, s[8:9] offset:8
	global_store_b32 v41, v162, s[8:9] offset:8
	s_cmp_le_i32 s13, 3
	s_cbranch_scc1 .Lpx_b2_stored
	global_store_b32 v40, v147, s[8:9] offset:12
	global_store_b32 v41, v163, s[8:9] offset:12
	.Lpx_b2_stored:
	s_wait_storecnt 0x0
	s_branch .Lpx_end
	.Lpx_b3:
	v_mov_b32_e32 v176, 0
	v_mov_b32_e32 v177, 0
	v_mov_b32_e32 v178, 0
	v_mov_b32_e32 v179, 0
	v_mov_b32_e32 v180, 0
	v_mov_b32_e32 v181, 0
	v_mov_b32_e32 v182, 0
	v_mov_b32_e32 v183, 0
	v_mov_b32_e32 v184, 0
	v_mov_b32_e32 v185, 0
	v_mov_b32_e32 v186, 0
	v_mov_b32_e32 v187, 0
	v_mov_b32_e32 v188, 0
	v_mov_b32_e32 v189, 0
	v_mov_b32_e32 v190, 0
	v_mov_b32_e32 v191, 0
	v_mov_b32_e32 v192, 0
	v_mov_b32_e32 v193, 0
	v_mov_b32_e32 v194, 0
	v_mov_b32_e32 v195, 0
	v_mov_b32_e32 v196, 0
	v_mov_b32_e32 v197, 0
	v_mov_b32_e32 v198, 0
	v_mov_b32_e32 v199, 0
	v_mov_b32_e32 v200, 0
	v_mov_b32_e32 v201, 0
	v_mov_b32_e32 v202, 0
	v_mov_b32_e32 v203, 0
	v_mov_b32_e32 v204, 0
	v_mov_b32_e32 v205, 0
	v_mov_b32_e32 v206, 0
	v_mov_b32_e32 v207, 0
	v_mov_b32_e32 v208, 0
	v_mov_b32_e32 v209, 0
	v_mov_b32_e32 v210, 0
	v_mov_b32_e32 v211, 0
	v_mov_b32_e32 v212, 0
	v_mov_b32_e32 v213, 0
	v_mov_b32_e32 v214, 0
	v_mov_b32_e32 v215, 0
	v_mov_b32_e32 v216, 0
	v_mov_b32_e32 v217, 0
	v_mov_b32_e32 v218, 0
	v_mov_b32_e32 v219, 0
	v_mov_b32_e32 v220, 0
	v_mov_b32_e32 v221, 0
	v_mov_b32_e32 v222, 0
	v_mov_b32_e32 v223, 0
	s_lshr_b32 s36, s14, 2
	s_and_b32 s37, s14, 3
	s_cmp_eq_u32 s36, 0
	s_cbranch_scc1 .Lpx_b3_tail
	.Lpx_b3_quad:
	buffer_load_b32 v8, v1, s[20:23], s16 offen
	buffer_load_b32 v9, v1, s[20:23], s17 offen
	buffer_load_b32 v10, v1, s[20:23], s18 offen
	buffer_load_b32 v11, v1, s[20:23], s19 offen
	buffer_load_b32 v24, v2, s[20:23], s16 offen offset:8
	buffer_load_b32 v25, v2, s[20:23], s17 offen offset:8
	buffer_load_b32 v26, v2, s[20:23], s18 offen offset:8
	buffer_load_b32 v27, v2, s[20:23], s19 offen offset:8
	buffer_load_b128 v[80:83], v3, s[24:27], s28 offen
	buffer_load_b128 v[84:87], v3, s[24:27], s28 offen offset:16
	buffer_load_b128 v[88:91], v3, s[24:27], s29 offen
	buffer_load_b128 v[92:95], v3, s[24:27], s29 offen offset:16
	buffer_load_b128 v[96:99], v3, s[24:27], s30 offen
	buffer_load_b128 v[100:103], v3, s[24:27], s30 offen offset:16
	buffer_load_b32 v12, v1, s[20:23], s16 offen offset:136
	buffer_load_b32 v13, v1, s[20:23], s17 offen offset:136
	buffer_load_b32 v14, v1, s[20:23], s18 offen offset:136
	buffer_load_b32 v15, v1, s[20:23], s19 offen offset:136
	buffer_load_b32 v28, v2, s[20:23], s16 offen offset:144
	buffer_load_b32 v29, v2, s[20:23], s17 offen offset:144
	buffer_load_b32 v30, v2, s[20:23], s18 offen offset:144
	buffer_load_b32 v31, v2, s[20:23], s19 offen offset:144
	buffer_load_b32 v16, v1, s[20:23], s16 offen offset:272
	buffer_load_b32 v17, v1, s[20:23], s17 offen offset:272
	buffer_load_b32 v18, v1, s[20:23], s18 offen offset:272
	buffer_load_b32 v19, v1, s[20:23], s19 offen offset:272
	buffer_load_b32 v32, v2, s[20:23], s16 offen offset:280
	buffer_load_b32 v33, v2, s[20:23], s17 offen offset:280
	buffer_load_b32 v34, v2, s[20:23], s18 offen offset:280
	buffer_load_b32 v35, v2, s[20:23], s19 offen offset:280
	buffer_load_b32 v20, v1, s[20:23], s16 offen offset:408
	buffer_load_b32 v21, v1, s[20:23], s17 offen offset:408
	buffer_load_b32 v22, v1, s[20:23], s18 offen offset:408
	buffer_load_b32 v23, v1, s[20:23], s19 offen offset:408
	buffer_load_b32 v36, v2, s[20:23], s16 offen offset:416
	buffer_load_b32 v37, v2, s[20:23], s17 offen offset:416
	buffer_load_b32 v38, v2, s[20:23], s18 offen offset:416
	buffer_load_b32 v39, v2, s[20:23], s19 offen offset:416
	s_wait_loadcnt 0x21
	v_and_b32_e32 v46, 0xf0f0f0f, v24
	s_wait_loadcnt 0x20
	v_and_b32_e32 v56, 0xf0f0f0f, v25
	s_wait_loadcnt 0x1f
	v_and_b32_e32 v66, 0xf0f0f0f, v26
	s_wait_loadcnt 0x1e
	v_and_b32_e32 v76, 0xf0f0f0f, v27
	v_lshrrev_b32_e32 v47, 4, v24
	v_lshrrev_b32_e32 v57, 4, v25
	v_lshrrev_b32_e32 v67, 4, v26
	v_lshrrev_b32_e32 v77, 4, v27
	v_and_b32_e32 v47, 0xf0f0f0f, v47
	v_and_b32_e32 v57, 0xf0f0f0f, v57
	v_and_b32_e32 v67, 0xf0f0f0f, v67
	v_and_b32_e32 v77, 0xf0f0f0f, v77
	v_cvt_f32_ubyte0_e32 v40, v46
	v_cvt_f32_ubyte0_e32 v50, v56
	v_cvt_f32_ubyte0_e32 v60, v66
	v_cvt_f32_ubyte0_e32 v70, v76
	v_cvt_f32_ubyte1_e32 v42, v46
	v_cvt_f32_ubyte1_e32 v52, v56
	v_cvt_f32_ubyte1_e32 v62, v66
	v_cvt_f32_ubyte1_e32 v72, v76
	v_cvt_f32_ubyte2_e32 v44, v46
	v_cvt_f32_ubyte2_e32 v54, v56
	v_cvt_f32_ubyte2_e32 v64, v66
	v_cvt_f32_ubyte2_e32 v74, v76
	v_cvt_f32_ubyte3_e32 v46, v46
	v_cvt_f32_ubyte3_e32 v56, v56
	v_cvt_f32_ubyte3_e32 v66, v66
	v_cvt_f32_ubyte3_e32 v76, v76
	v_cvt_f32_ubyte0_e32 v41, v47
	v_cvt_f32_ubyte0_e32 v51, v57
	v_cvt_f32_ubyte0_e32 v61, v67
	v_cvt_f32_ubyte0_e32 v71, v77
	v_cvt_f32_ubyte1_e32 v43, v47
	v_cvt_f32_ubyte1_e32 v53, v57
	v_cvt_f32_ubyte1_e32 v63, v67
	v_cvt_f32_ubyte1_e32 v73, v77
	v_cvt_f32_ubyte2_e32 v45, v47
	v_cvt_f32_ubyte2_e32 v55, v57
	v_cvt_f32_ubyte2_e32 v65, v67
	v_cvt_f32_ubyte2_e32 v75, v77
	v_cvt_f32_ubyte3_e32 v47, v47
	v_cvt_f32_ubyte3_e32 v57, v57
	v_cvt_f32_ubyte3_e32 v67, v67
	v_cvt_f32_ubyte3_e32 v77, v77
	v_fma_mix_f32 v40, v8, v40, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v9, v50, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v10, v60, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v70, v11, v70, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v41, v8, v41, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v9, v51, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v10, v61, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v71, v11, v71, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v42, v8, v42, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v52, v9, v52, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v62, v10, v62, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v72, v11, v72, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v43, v8, v43, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v53, v9, v53, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v63, v10, v63, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v73, v11, v73, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v44, v8, v44, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v54, v9, v54, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v64, v10, v64, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v74, v11, v74, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v8, v45, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v9, v55, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v65, v10, v65, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v75, v11, v75, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v8, v46, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v9, v56, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v66, v10, v66, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v76, v11, v76, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v8, v47, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v9, v57, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v67, v10, v67, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v77, v11, v77, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	buffer_load_b128 v[104:107], v3, s[24:27], s28 offen offset:1024
	buffer_load_b128 v[108:111], v3, s[24:27], s28 offen offset:1040
	buffer_load_b128 v[112:115], v3, s[24:27], s29 offen offset:1024
	buffer_load_b128 v[116:119], v3, s[24:27], s29 offen offset:1040
	buffer_load_b128 v[120:123], v3, s[24:27], s30 offen offset:1024
	buffer_load_b128 v[124:127], v3, s[24:27], s30 offen offset:1040
	buffer_load_b128 v[128:131], v3, s[24:27], s28 offen offset:2048
	buffer_load_b128 v[132:135], v3, s[24:27], s28 offen offset:2064
	buffer_load_b128 v[136:139], v3, s[24:27], s29 offen offset:2048
	buffer_load_b128 v[140:143], v3, s[24:27], s29 offen offset:2064
	buffer_load_b128 v[144:147], v3, s[24:27], s30 offen offset:2048
	buffer_load_b128 v[148:151], v3, s[24:27], s30 offen offset:2064
	buffer_load_b128 v[152:155], v3, s[24:27], s28 offen offset:3072
	buffer_load_b128 v[156:159], v3, s[24:27], s28 offen offset:3088
	buffer_load_b128 v[160:163], v3, s[24:27], s29 offen offset:3072
	buffer_load_b128 v[164:167], v3, s[24:27], s29 offen offset:3088
	buffer_load_b128 v[168:171], v3, s[24:27], s30 offen offset:3072
	buffer_load_b128 v[172:175], v3, s[24:27], s30 offen offset:3088
	s_wait_loadcnt 0x2f
	v_dual_mul_f32 v48, v41, v81 :: v_dual_mul_f32 v49, v51, v81
	v_dual_mul_f32 v58, v61, v81 :: v_dual_mul_f32 v59, v71, v81
	s_wait_loadcnt 0x2d
	v_dual_mul_f32 v68, v41, v89 :: v_dual_mul_f32 v69, v51, v89
	v_dual_mul_f32 v78, v61, v89 :: v_dual_mul_f32 v79, v71, v89
	v_dual_fmac_f32 v48, v40, v80 :: v_dual_fmac_f32 v49, v50, v80
	v_dual_fmac_f32 v58, v60, v80 :: v_dual_fmac_f32 v59, v70, v80
	v_dual_fmac_f32 v68, v40, v88 :: v_dual_fmac_f32 v69, v50, v88
	v_dual_fmac_f32 v78, v60, v88 :: v_dual_fmac_f32 v79, v70, v88
	v_dual_fmac_f32 v48, v42, v82 :: v_dual_fmac_f32 v49, v52, v82
	v_dual_fmac_f32 v58, v62, v82 :: v_dual_fmac_f32 v59, v72, v82
	v_dual_fmac_f32 v68, v42, v90 :: v_dual_fmac_f32 v69, v52, v90
	v_dual_fmac_f32 v78, v62, v90 :: v_dual_fmac_f32 v79, v72, v90
	v_dual_fmac_f32 v48, v43, v83 :: v_dual_fmac_f32 v49, v53, v83
	v_dual_fmac_f32 v58, v63, v83 :: v_dual_fmac_f32 v59, v73, v83
	v_dual_fmac_f32 v68, v43, v91 :: v_dual_fmac_f32 v69, v53, v91
	v_dual_fmac_f32 v78, v63, v91 :: v_dual_fmac_f32 v79, v73, v91
	v_dual_fmac_f32 v48, v44, v84 :: v_dual_fmac_f32 v49, v54, v84
	v_dual_fmac_f32 v58, v64, v84 :: v_dual_fmac_f32 v59, v74, v84
	s_wait_loadcnt 0x2c
	v_dual_fmac_f32 v68, v44, v92 :: v_dual_fmac_f32 v69, v54, v92
	v_dual_fmac_f32 v78, v64, v92 :: v_dual_fmac_f32 v79, v74, v92
	v_dual_fmac_f32 v48, v45, v85 :: v_dual_fmac_f32 v49, v55, v85
	v_dual_fmac_f32 v58, v65, v85 :: v_dual_fmac_f32 v59, v75, v85
	v_dual_fmac_f32 v68, v45, v93 :: v_dual_fmac_f32 v69, v55, v93
	v_dual_fmac_f32 v78, v65, v93 :: v_dual_fmac_f32 v79, v75, v93
	v_dual_fmac_f32 v48, v46, v86 :: v_dual_fmac_f32 v49, v56, v86
	v_dual_fmac_f32 v58, v66, v86 :: v_dual_fmac_f32 v59, v76, v86
	v_dual_fmac_f32 v68, v46, v94 :: v_dual_fmac_f32 v69, v56, v94
	v_dual_fmac_f32 v78, v66, v94 :: v_dual_fmac_f32 v79, v76, v94
	v_dual_fmac_f32 v48, v47, v87 :: v_dual_fmac_f32 v49, v57, v87
	v_dual_fmac_f32 v58, v67, v87 :: v_dual_fmac_f32 v59, v77, v87
	v_dual_fmac_f32 v68, v47, v95 :: v_dual_fmac_f32 v69, v57, v95
	v_dual_fmac_f32 v78, v67, v95 :: v_dual_fmac_f32 v79, v77, v95
	v_dual_add_f32 v176, v176, v48 :: v_dual_add_f32 v177, v177, v49
	v_dual_add_f32 v178, v178, v58 :: v_dual_add_f32 v179, v179, v59
	v_dual_add_f32 v192, v192, v68 :: v_dual_add_f32 v193, v193, v69
	v_dual_add_f32 v194, v194, v78 :: v_dual_add_f32 v195, v195, v79
	s_wait_loadcnt 0x2b
	v_dual_mul_f32 v48, v41, v97 :: v_dual_mul_f32 v49, v51, v97
	v_dual_mul_f32 v58, v61, v97 :: v_dual_mul_f32 v59, v71, v97
	v_dual_fmac_f32 v48, v40, v96 :: v_dual_fmac_f32 v49, v50, v96
	v_dual_fmac_f32 v58, v60, v96 :: v_dual_fmac_f32 v59, v70, v96
	v_dual_fmac_f32 v48, v42, v98 :: v_dual_fmac_f32 v49, v52, v98
	v_dual_fmac_f32 v58, v62, v98 :: v_dual_fmac_f32 v59, v72, v98
	v_dual_fmac_f32 v48, v43, v99 :: v_dual_fmac_f32 v49, v53, v99
	v_dual_fmac_f32 v58, v63, v99 :: v_dual_fmac_f32 v59, v73, v99
	s_wait_loadcnt 0x2a
	v_dual_fmac_f32 v48, v44, v100 :: v_dual_fmac_f32 v49, v54, v100
	v_dual_fmac_f32 v58, v64, v100 :: v_dual_fmac_f32 v59, v74, v100
	v_dual_fmac_f32 v48, v45, v101 :: v_dual_fmac_f32 v49, v55, v101
	v_dual_fmac_f32 v58, v65, v101 :: v_dual_fmac_f32 v59, v75, v101
	v_dual_fmac_f32 v48, v46, v102 :: v_dual_fmac_f32 v49, v56, v102
	v_dual_fmac_f32 v58, v66, v102 :: v_dual_fmac_f32 v59, v76, v102
	v_dual_fmac_f32 v48, v47, v103 :: v_dual_fmac_f32 v49, v57, v103
	v_dual_fmac_f32 v58, v67, v103 :: v_dual_fmac_f32 v59, v77, v103
	v_dual_add_f32 v208, v208, v48 :: v_dual_add_f32 v209, v209, v49
	v_dual_add_f32 v210, v210, v58 :: v_dual_add_f32 v211, v211, v59
	s_wait_loadcnt 0x25
	v_and_b32_e32 v46, 0xf0f0f0f, v28
	s_wait_loadcnt 0x24
	v_and_b32_e32 v56, 0xf0f0f0f, v29
	s_wait_loadcnt 0x23
	v_and_b32_e32 v66, 0xf0f0f0f, v30
	s_wait_loadcnt 0x22
	v_and_b32_e32 v76, 0xf0f0f0f, v31
	v_lshrrev_b32_e32 v47, 4, v28
	v_lshrrev_b32_e32 v57, 4, v29
	v_lshrrev_b32_e32 v67, 4, v30
	v_lshrrev_b32_e32 v77, 4, v31
	v_and_b32_e32 v47, 0xf0f0f0f, v47
	v_and_b32_e32 v57, 0xf0f0f0f, v57
	v_and_b32_e32 v67, 0xf0f0f0f, v67
	v_and_b32_e32 v77, 0xf0f0f0f, v77
	v_cvt_f32_ubyte0_e32 v40, v46
	v_cvt_f32_ubyte0_e32 v50, v56
	v_cvt_f32_ubyte0_e32 v60, v66
	v_cvt_f32_ubyte0_e32 v70, v76
	v_cvt_f32_ubyte1_e32 v42, v46
	v_cvt_f32_ubyte1_e32 v52, v56
	v_cvt_f32_ubyte1_e32 v62, v66
	v_cvt_f32_ubyte1_e32 v72, v76
	v_cvt_f32_ubyte2_e32 v44, v46
	v_cvt_f32_ubyte2_e32 v54, v56
	v_cvt_f32_ubyte2_e32 v64, v66
	v_cvt_f32_ubyte2_e32 v74, v76
	v_cvt_f32_ubyte3_e32 v46, v46
	v_cvt_f32_ubyte3_e32 v56, v56
	v_cvt_f32_ubyte3_e32 v66, v66
	v_cvt_f32_ubyte3_e32 v76, v76
	v_cvt_f32_ubyte0_e32 v41, v47
	v_cvt_f32_ubyte0_e32 v51, v57
	v_cvt_f32_ubyte0_e32 v61, v67
	v_cvt_f32_ubyte0_e32 v71, v77
	v_cvt_f32_ubyte1_e32 v43, v47
	v_cvt_f32_ubyte1_e32 v53, v57
	v_cvt_f32_ubyte1_e32 v63, v67
	v_cvt_f32_ubyte1_e32 v73, v77
	v_cvt_f32_ubyte2_e32 v45, v47
	v_cvt_f32_ubyte2_e32 v55, v57
	v_cvt_f32_ubyte2_e32 v65, v67
	v_cvt_f32_ubyte2_e32 v75, v77
	v_cvt_f32_ubyte3_e32 v47, v47
	v_cvt_f32_ubyte3_e32 v57, v57
	v_cvt_f32_ubyte3_e32 v67, v67
	v_cvt_f32_ubyte3_e32 v77, v77
	v_fma_mix_f32 v40, v12, v40, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v13, v50, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v14, v60, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v70, v15, v70, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v41, v12, v41, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v13, v51, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v14, v61, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v71, v15, v71, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v42, v12, v42, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v52, v13, v52, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v62, v14, v62, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v72, v15, v72, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v43, v12, v43, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v53, v13, v53, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v63, v14, v63, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v73, v15, v73, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v44, v12, v44, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v54, v13, v54, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v64, v14, v64, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v74, v15, v74, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v12, v45, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v13, v55, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v65, v14, v65, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v75, v15, v75, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v12, v46, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v13, v56, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v66, v14, v66, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v76, v15, v76, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v12, v47, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v13, v57, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v67, v14, v67, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v77, v15, v77, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x11
	v_dual_mul_f32 v48, v41, v105 :: v_dual_mul_f32 v49, v51, v105
	v_dual_mul_f32 v58, v61, v105 :: v_dual_mul_f32 v59, v71, v105
	s_wait_loadcnt 0xf
	v_dual_mul_f32 v68, v41, v113 :: v_dual_mul_f32 v69, v51, v113
	v_dual_mul_f32 v78, v61, v113 :: v_dual_mul_f32 v79, v71, v113
	v_dual_fmac_f32 v48, v40, v104 :: v_dual_fmac_f32 v49, v50, v104
	v_dual_fmac_f32 v58, v60, v104 :: v_dual_fmac_f32 v59, v70, v104
	v_dual_fmac_f32 v68, v40, v112 :: v_dual_fmac_f32 v69, v50, v112
	v_dual_fmac_f32 v78, v60, v112 :: v_dual_fmac_f32 v79, v70, v112
	v_dual_fmac_f32 v48, v42, v106 :: v_dual_fmac_f32 v49, v52, v106
	v_dual_fmac_f32 v58, v62, v106 :: v_dual_fmac_f32 v59, v72, v106
	v_dual_fmac_f32 v68, v42, v114 :: v_dual_fmac_f32 v69, v52, v114
	v_dual_fmac_f32 v78, v62, v114 :: v_dual_fmac_f32 v79, v72, v114
	v_dual_fmac_f32 v48, v43, v107 :: v_dual_fmac_f32 v49, v53, v107
	v_dual_fmac_f32 v58, v63, v107 :: v_dual_fmac_f32 v59, v73, v107
	v_dual_fmac_f32 v68, v43, v115 :: v_dual_fmac_f32 v69, v53, v115
	v_dual_fmac_f32 v78, v63, v115 :: v_dual_fmac_f32 v79, v73, v115
	v_dual_fmac_f32 v48, v44, v108 :: v_dual_fmac_f32 v49, v54, v108
	v_dual_fmac_f32 v58, v64, v108 :: v_dual_fmac_f32 v59, v74, v108
	s_wait_loadcnt 0xe
	v_dual_fmac_f32 v68, v44, v116 :: v_dual_fmac_f32 v69, v54, v116
	v_dual_fmac_f32 v78, v64, v116 :: v_dual_fmac_f32 v79, v74, v116
	v_dual_fmac_f32 v48, v45, v109 :: v_dual_fmac_f32 v49, v55, v109
	v_dual_fmac_f32 v58, v65, v109 :: v_dual_fmac_f32 v59, v75, v109
	v_dual_fmac_f32 v68, v45, v117 :: v_dual_fmac_f32 v69, v55, v117
	v_dual_fmac_f32 v78, v65, v117 :: v_dual_fmac_f32 v79, v75, v117
	v_dual_fmac_f32 v48, v46, v110 :: v_dual_fmac_f32 v49, v56, v110
	v_dual_fmac_f32 v58, v66, v110 :: v_dual_fmac_f32 v59, v76, v110
	v_dual_fmac_f32 v68, v46, v118 :: v_dual_fmac_f32 v69, v56, v118
	v_dual_fmac_f32 v78, v66, v118 :: v_dual_fmac_f32 v79, v76, v118
	v_dual_fmac_f32 v48, v47, v111 :: v_dual_fmac_f32 v49, v57, v111
	v_dual_fmac_f32 v58, v67, v111 :: v_dual_fmac_f32 v59, v77, v111
	v_dual_fmac_f32 v68, v47, v119 :: v_dual_fmac_f32 v69, v57, v119
	v_dual_fmac_f32 v78, v67, v119 :: v_dual_fmac_f32 v79, v77, v119
	v_dual_add_f32 v180, v180, v48 :: v_dual_add_f32 v181, v181, v49
	v_dual_add_f32 v182, v182, v58 :: v_dual_add_f32 v183, v183, v59
	v_dual_add_f32 v196, v196, v68 :: v_dual_add_f32 v197, v197, v69
	v_dual_add_f32 v198, v198, v78 :: v_dual_add_f32 v199, v199, v79
	s_wait_loadcnt 0xd
	v_dual_mul_f32 v48, v41, v121 :: v_dual_mul_f32 v49, v51, v121
	v_dual_mul_f32 v58, v61, v121 :: v_dual_mul_f32 v59, v71, v121
	v_dual_fmac_f32 v48, v40, v120 :: v_dual_fmac_f32 v49, v50, v120
	v_dual_fmac_f32 v58, v60, v120 :: v_dual_fmac_f32 v59, v70, v120
	v_dual_fmac_f32 v48, v42, v122 :: v_dual_fmac_f32 v49, v52, v122
	v_dual_fmac_f32 v58, v62, v122 :: v_dual_fmac_f32 v59, v72, v122
	v_dual_fmac_f32 v48, v43, v123 :: v_dual_fmac_f32 v49, v53, v123
	v_dual_fmac_f32 v58, v63, v123 :: v_dual_fmac_f32 v59, v73, v123
	s_wait_loadcnt 0xc
	v_dual_fmac_f32 v48, v44, v124 :: v_dual_fmac_f32 v49, v54, v124
	v_dual_fmac_f32 v58, v64, v124 :: v_dual_fmac_f32 v59, v74, v124
	v_dual_fmac_f32 v48, v45, v125 :: v_dual_fmac_f32 v49, v55, v125
	v_dual_fmac_f32 v58, v65, v125 :: v_dual_fmac_f32 v59, v75, v125
	v_dual_fmac_f32 v48, v46, v126 :: v_dual_fmac_f32 v49, v56, v126
	v_dual_fmac_f32 v58, v66, v126 :: v_dual_fmac_f32 v59, v76, v126
	v_dual_fmac_f32 v48, v47, v127 :: v_dual_fmac_f32 v49, v57, v127
	v_dual_fmac_f32 v58, v67, v127 :: v_dual_fmac_f32 v59, v77, v127
	v_dual_add_f32 v212, v212, v48 :: v_dual_add_f32 v213, v213, v49
	v_dual_add_f32 v214, v214, v58 :: v_dual_add_f32 v215, v215, v59
	v_and_b32_e32 v46, 0xf0f0f0f, v32
	v_and_b32_e32 v56, 0xf0f0f0f, v33
	v_and_b32_e32 v66, 0xf0f0f0f, v34
	v_and_b32_e32 v76, 0xf0f0f0f, v35
	v_lshrrev_b32_e32 v47, 4, v32
	v_lshrrev_b32_e32 v57, 4, v33
	v_lshrrev_b32_e32 v67, 4, v34
	v_lshrrev_b32_e32 v77, 4, v35
	v_and_b32_e32 v47, 0xf0f0f0f, v47
	v_and_b32_e32 v57, 0xf0f0f0f, v57
	v_and_b32_e32 v67, 0xf0f0f0f, v67
	v_and_b32_e32 v77, 0xf0f0f0f, v77
	v_cvt_f32_ubyte0_e32 v40, v46
	v_cvt_f32_ubyte0_e32 v50, v56
	v_cvt_f32_ubyte0_e32 v60, v66
	v_cvt_f32_ubyte0_e32 v70, v76
	v_cvt_f32_ubyte1_e32 v42, v46
	v_cvt_f32_ubyte1_e32 v52, v56
	v_cvt_f32_ubyte1_e32 v62, v66
	v_cvt_f32_ubyte1_e32 v72, v76
	v_cvt_f32_ubyte2_e32 v44, v46
	v_cvt_f32_ubyte2_e32 v54, v56
	v_cvt_f32_ubyte2_e32 v64, v66
	v_cvt_f32_ubyte2_e32 v74, v76
	v_cvt_f32_ubyte3_e32 v46, v46
	v_cvt_f32_ubyte3_e32 v56, v56
	v_cvt_f32_ubyte3_e32 v66, v66
	v_cvt_f32_ubyte3_e32 v76, v76
	v_cvt_f32_ubyte0_e32 v41, v47
	v_cvt_f32_ubyte0_e32 v51, v57
	v_cvt_f32_ubyte0_e32 v61, v67
	v_cvt_f32_ubyte0_e32 v71, v77
	v_cvt_f32_ubyte1_e32 v43, v47
	v_cvt_f32_ubyte1_e32 v53, v57
	v_cvt_f32_ubyte1_e32 v63, v67
	v_cvt_f32_ubyte1_e32 v73, v77
	v_cvt_f32_ubyte2_e32 v45, v47
	v_cvt_f32_ubyte2_e32 v55, v57
	v_cvt_f32_ubyte2_e32 v65, v67
	v_cvt_f32_ubyte2_e32 v75, v77
	v_cvt_f32_ubyte3_e32 v47, v47
	v_cvt_f32_ubyte3_e32 v57, v57
	v_cvt_f32_ubyte3_e32 v67, v67
	v_cvt_f32_ubyte3_e32 v77, v77
	v_fma_mix_f32 v40, v16, v40, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v17, v50, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v18, v60, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v70, v19, v70, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v41, v16, v41, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v17, v51, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v18, v61, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v71, v19, v71, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v42, v16, v42, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v52, v17, v52, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v62, v18, v62, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v72, v19, v72, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v43, v16, v43, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v53, v17, v53, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v63, v18, v63, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v73, v19, v73, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v44, v16, v44, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v54, v17, v54, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v64, v18, v64, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v74, v19, v74, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v16, v45, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v17, v55, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v65, v18, v65, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v75, v19, v75, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v16, v46, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v17, v56, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v66, v18, v66, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v76, v19, v76, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v16, v47, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v17, v57, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v67, v18, v67, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v77, v19, v77, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xb
	v_dual_mul_f32 v48, v41, v129 :: v_dual_mul_f32 v49, v51, v129
	v_dual_mul_f32 v58, v61, v129 :: v_dual_mul_f32 v59, v71, v129
	s_wait_loadcnt 0x9
	v_dual_mul_f32 v68, v41, v137 :: v_dual_mul_f32 v69, v51, v137
	v_dual_mul_f32 v78, v61, v137 :: v_dual_mul_f32 v79, v71, v137
	v_dual_fmac_f32 v48, v40, v128 :: v_dual_fmac_f32 v49, v50, v128
	v_dual_fmac_f32 v58, v60, v128 :: v_dual_fmac_f32 v59, v70, v128
	v_dual_fmac_f32 v68, v40, v136 :: v_dual_fmac_f32 v69, v50, v136
	v_dual_fmac_f32 v78, v60, v136 :: v_dual_fmac_f32 v79, v70, v136
	v_dual_fmac_f32 v48, v42, v130 :: v_dual_fmac_f32 v49, v52, v130
	v_dual_fmac_f32 v58, v62, v130 :: v_dual_fmac_f32 v59, v72, v130
	v_dual_fmac_f32 v68, v42, v138 :: v_dual_fmac_f32 v69, v52, v138
	v_dual_fmac_f32 v78, v62, v138 :: v_dual_fmac_f32 v79, v72, v138
	v_dual_fmac_f32 v48, v43, v131 :: v_dual_fmac_f32 v49, v53, v131
	v_dual_fmac_f32 v58, v63, v131 :: v_dual_fmac_f32 v59, v73, v131
	v_dual_fmac_f32 v68, v43, v139 :: v_dual_fmac_f32 v69, v53, v139
	v_dual_fmac_f32 v78, v63, v139 :: v_dual_fmac_f32 v79, v73, v139
	v_dual_fmac_f32 v48, v44, v132 :: v_dual_fmac_f32 v49, v54, v132
	v_dual_fmac_f32 v58, v64, v132 :: v_dual_fmac_f32 v59, v74, v132
	s_wait_loadcnt 0x8
	v_dual_fmac_f32 v68, v44, v140 :: v_dual_fmac_f32 v69, v54, v140
	v_dual_fmac_f32 v78, v64, v140 :: v_dual_fmac_f32 v79, v74, v140
	v_dual_fmac_f32 v48, v45, v133 :: v_dual_fmac_f32 v49, v55, v133
	v_dual_fmac_f32 v58, v65, v133 :: v_dual_fmac_f32 v59, v75, v133
	v_dual_fmac_f32 v68, v45, v141 :: v_dual_fmac_f32 v69, v55, v141
	v_dual_fmac_f32 v78, v65, v141 :: v_dual_fmac_f32 v79, v75, v141
	v_dual_fmac_f32 v48, v46, v134 :: v_dual_fmac_f32 v49, v56, v134
	v_dual_fmac_f32 v58, v66, v134 :: v_dual_fmac_f32 v59, v76, v134
	v_dual_fmac_f32 v68, v46, v142 :: v_dual_fmac_f32 v69, v56, v142
	v_dual_fmac_f32 v78, v66, v142 :: v_dual_fmac_f32 v79, v76, v142
	v_dual_fmac_f32 v48, v47, v135 :: v_dual_fmac_f32 v49, v57, v135
	v_dual_fmac_f32 v58, v67, v135 :: v_dual_fmac_f32 v59, v77, v135
	v_dual_fmac_f32 v68, v47, v143 :: v_dual_fmac_f32 v69, v57, v143
	v_dual_fmac_f32 v78, v67, v143 :: v_dual_fmac_f32 v79, v77, v143
	v_dual_add_f32 v184, v184, v48 :: v_dual_add_f32 v185, v185, v49
	v_dual_add_f32 v186, v186, v58 :: v_dual_add_f32 v187, v187, v59
	v_dual_add_f32 v200, v200, v68 :: v_dual_add_f32 v201, v201, v69
	v_dual_add_f32 v202, v202, v78 :: v_dual_add_f32 v203, v203, v79
	s_wait_loadcnt 0x7
	v_dual_mul_f32 v48, v41, v145 :: v_dual_mul_f32 v49, v51, v145
	v_dual_mul_f32 v58, v61, v145 :: v_dual_mul_f32 v59, v71, v145
	v_dual_fmac_f32 v48, v40, v144 :: v_dual_fmac_f32 v49, v50, v144
	v_dual_fmac_f32 v58, v60, v144 :: v_dual_fmac_f32 v59, v70, v144
	v_dual_fmac_f32 v48, v42, v146 :: v_dual_fmac_f32 v49, v52, v146
	v_dual_fmac_f32 v58, v62, v146 :: v_dual_fmac_f32 v59, v72, v146
	v_dual_fmac_f32 v48, v43, v147 :: v_dual_fmac_f32 v49, v53, v147
	v_dual_fmac_f32 v58, v63, v147 :: v_dual_fmac_f32 v59, v73, v147
	s_wait_loadcnt 0x6
	v_dual_fmac_f32 v48, v44, v148 :: v_dual_fmac_f32 v49, v54, v148
	v_dual_fmac_f32 v58, v64, v148 :: v_dual_fmac_f32 v59, v74, v148
	v_dual_fmac_f32 v48, v45, v149 :: v_dual_fmac_f32 v49, v55, v149
	v_dual_fmac_f32 v58, v65, v149 :: v_dual_fmac_f32 v59, v75, v149
	v_dual_fmac_f32 v48, v46, v150 :: v_dual_fmac_f32 v49, v56, v150
	v_dual_fmac_f32 v58, v66, v150 :: v_dual_fmac_f32 v59, v76, v150
	v_dual_fmac_f32 v48, v47, v151 :: v_dual_fmac_f32 v49, v57, v151
	v_dual_fmac_f32 v58, v67, v151 :: v_dual_fmac_f32 v59, v77, v151
	v_dual_add_f32 v216, v216, v48 :: v_dual_add_f32 v217, v217, v49
	v_dual_add_f32 v218, v218, v58 :: v_dual_add_f32 v219, v219, v59
	v_and_b32_e32 v46, 0xf0f0f0f, v36
	v_and_b32_e32 v56, 0xf0f0f0f, v37
	v_and_b32_e32 v66, 0xf0f0f0f, v38
	v_and_b32_e32 v76, 0xf0f0f0f, v39
	v_lshrrev_b32_e32 v47, 4, v36
	v_lshrrev_b32_e32 v57, 4, v37
	v_lshrrev_b32_e32 v67, 4, v38
	v_lshrrev_b32_e32 v77, 4, v39
	v_and_b32_e32 v47, 0xf0f0f0f, v47
	v_and_b32_e32 v57, 0xf0f0f0f, v57
	v_and_b32_e32 v67, 0xf0f0f0f, v67
	v_and_b32_e32 v77, 0xf0f0f0f, v77
	v_cvt_f32_ubyte0_e32 v40, v46
	v_cvt_f32_ubyte0_e32 v50, v56
	v_cvt_f32_ubyte0_e32 v60, v66
	v_cvt_f32_ubyte0_e32 v70, v76
	v_cvt_f32_ubyte1_e32 v42, v46
	v_cvt_f32_ubyte1_e32 v52, v56
	v_cvt_f32_ubyte1_e32 v62, v66
	v_cvt_f32_ubyte1_e32 v72, v76
	v_cvt_f32_ubyte2_e32 v44, v46
	v_cvt_f32_ubyte2_e32 v54, v56
	v_cvt_f32_ubyte2_e32 v64, v66
	v_cvt_f32_ubyte2_e32 v74, v76
	v_cvt_f32_ubyte3_e32 v46, v46
	v_cvt_f32_ubyte3_e32 v56, v56
	v_cvt_f32_ubyte3_e32 v66, v66
	v_cvt_f32_ubyte3_e32 v76, v76
	v_cvt_f32_ubyte0_e32 v41, v47
	v_cvt_f32_ubyte0_e32 v51, v57
	v_cvt_f32_ubyte0_e32 v61, v67
	v_cvt_f32_ubyte0_e32 v71, v77
	v_cvt_f32_ubyte1_e32 v43, v47
	v_cvt_f32_ubyte1_e32 v53, v57
	v_cvt_f32_ubyte1_e32 v63, v67
	v_cvt_f32_ubyte1_e32 v73, v77
	v_cvt_f32_ubyte2_e32 v45, v47
	v_cvt_f32_ubyte2_e32 v55, v57
	v_cvt_f32_ubyte2_e32 v65, v67
	v_cvt_f32_ubyte2_e32 v75, v77
	v_cvt_f32_ubyte3_e32 v47, v47
	v_cvt_f32_ubyte3_e32 v57, v57
	v_cvt_f32_ubyte3_e32 v67, v67
	v_cvt_f32_ubyte3_e32 v77, v77
	v_fma_mix_f32 v40, v20, v40, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v21, v50, v21 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v22, v60, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v70, v23, v70, v23 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v41, v20, v41, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v21, v51, v21 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v22, v61, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v71, v23, v71, v23 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v42, v20, v42, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v52, v21, v52, v21 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v62, v22, v62, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v72, v23, v72, v23 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v43, v20, v43, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v53, v21, v53, v21 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v63, v22, v63, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v73, v23, v73, v23 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v44, v20, v44, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v54, v21, v54, v21 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v64, v22, v64, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v74, v23, v74, v23 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v20, v45, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v21, v55, v21 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v65, v22, v65, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v75, v23, v75, v23 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v20, v46, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v21, v56, v21 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v66, v22, v66, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v76, v23, v76, v23 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v20, v47, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v21, v57, v21 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v67, v22, v67, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v77, v23, v77, v23 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x5
	v_dual_mul_f32 v48, v41, v153 :: v_dual_mul_f32 v49, v51, v153
	v_dual_mul_f32 v58, v61, v153 :: v_dual_mul_f32 v59, v71, v153
	s_wait_loadcnt 0x3
	v_dual_mul_f32 v68, v41, v161 :: v_dual_mul_f32 v69, v51, v161
	v_dual_mul_f32 v78, v61, v161 :: v_dual_mul_f32 v79, v71, v161
	v_dual_fmac_f32 v48, v40, v152 :: v_dual_fmac_f32 v49, v50, v152
	v_dual_fmac_f32 v58, v60, v152 :: v_dual_fmac_f32 v59, v70, v152
	v_dual_fmac_f32 v68, v40, v160 :: v_dual_fmac_f32 v69, v50, v160
	v_dual_fmac_f32 v78, v60, v160 :: v_dual_fmac_f32 v79, v70, v160
	v_dual_fmac_f32 v48, v42, v154 :: v_dual_fmac_f32 v49, v52, v154
	v_dual_fmac_f32 v58, v62, v154 :: v_dual_fmac_f32 v59, v72, v154
	v_dual_fmac_f32 v68, v42, v162 :: v_dual_fmac_f32 v69, v52, v162
	v_dual_fmac_f32 v78, v62, v162 :: v_dual_fmac_f32 v79, v72, v162
	v_dual_fmac_f32 v48, v43, v155 :: v_dual_fmac_f32 v49, v53, v155
	v_dual_fmac_f32 v58, v63, v155 :: v_dual_fmac_f32 v59, v73, v155
	v_dual_fmac_f32 v68, v43, v163 :: v_dual_fmac_f32 v69, v53, v163
	v_dual_fmac_f32 v78, v63, v163 :: v_dual_fmac_f32 v79, v73, v163
	v_dual_fmac_f32 v48, v44, v156 :: v_dual_fmac_f32 v49, v54, v156
	v_dual_fmac_f32 v58, v64, v156 :: v_dual_fmac_f32 v59, v74, v156
	s_wait_loadcnt 0x2
	v_dual_fmac_f32 v68, v44, v164 :: v_dual_fmac_f32 v69, v54, v164
	v_dual_fmac_f32 v78, v64, v164 :: v_dual_fmac_f32 v79, v74, v164
	v_dual_fmac_f32 v48, v45, v157 :: v_dual_fmac_f32 v49, v55, v157
	v_dual_fmac_f32 v58, v65, v157 :: v_dual_fmac_f32 v59, v75, v157
	v_dual_fmac_f32 v68, v45, v165 :: v_dual_fmac_f32 v69, v55, v165
	v_dual_fmac_f32 v78, v65, v165 :: v_dual_fmac_f32 v79, v75, v165
	v_dual_fmac_f32 v48, v46, v158 :: v_dual_fmac_f32 v49, v56, v158
	v_dual_fmac_f32 v58, v66, v158 :: v_dual_fmac_f32 v59, v76, v158
	v_dual_fmac_f32 v68, v46, v166 :: v_dual_fmac_f32 v69, v56, v166
	v_dual_fmac_f32 v78, v66, v166 :: v_dual_fmac_f32 v79, v76, v166
	v_dual_fmac_f32 v48, v47, v159 :: v_dual_fmac_f32 v49, v57, v159
	v_dual_fmac_f32 v58, v67, v159 :: v_dual_fmac_f32 v59, v77, v159
	v_dual_fmac_f32 v68, v47, v167 :: v_dual_fmac_f32 v69, v57, v167
	v_dual_fmac_f32 v78, v67, v167 :: v_dual_fmac_f32 v79, v77, v167
	v_dual_add_f32 v188, v188, v48 :: v_dual_add_f32 v189, v189, v49
	v_dual_add_f32 v190, v190, v58 :: v_dual_add_f32 v191, v191, v59
	v_dual_add_f32 v204, v204, v68 :: v_dual_add_f32 v205, v205, v69
	v_dual_add_f32 v206, v206, v78 :: v_dual_add_f32 v207, v207, v79
	s_wait_loadcnt 0x1
	v_dual_mul_f32 v48, v41, v169 :: v_dual_mul_f32 v49, v51, v169
	v_dual_mul_f32 v58, v61, v169 :: v_dual_mul_f32 v59, v71, v169
	v_dual_fmac_f32 v48, v40, v168 :: v_dual_fmac_f32 v49, v50, v168
	v_dual_fmac_f32 v58, v60, v168 :: v_dual_fmac_f32 v59, v70, v168
	v_dual_fmac_f32 v48, v42, v170 :: v_dual_fmac_f32 v49, v52, v170
	v_dual_fmac_f32 v58, v62, v170 :: v_dual_fmac_f32 v59, v72, v170
	v_dual_fmac_f32 v48, v43, v171 :: v_dual_fmac_f32 v49, v53, v171
	v_dual_fmac_f32 v58, v63, v171 :: v_dual_fmac_f32 v59, v73, v171
	s_wait_loadcnt 0x0
	v_dual_fmac_f32 v48, v44, v172 :: v_dual_fmac_f32 v49, v54, v172
	v_dual_fmac_f32 v58, v64, v172 :: v_dual_fmac_f32 v59, v74, v172
	v_dual_fmac_f32 v48, v45, v173 :: v_dual_fmac_f32 v49, v55, v173
	v_dual_fmac_f32 v58, v65, v173 :: v_dual_fmac_f32 v59, v75, v173
	v_dual_fmac_f32 v48, v46, v174 :: v_dual_fmac_f32 v49, v56, v174
	v_dual_fmac_f32 v58, v66, v174 :: v_dual_fmac_f32 v59, v76, v174
	v_dual_fmac_f32 v48, v47, v175 :: v_dual_fmac_f32 v49, v57, v175
	v_dual_fmac_f32 v58, v67, v175 :: v_dual_fmac_f32 v59, v77, v175
	v_dual_add_f32 v220, v220, v48 :: v_dual_add_f32 v221, v221, v49
	v_dual_add_f32 v222, v222, v58 :: v_dual_add_f32 v223, v223, v59
	s_add_co_i32 s16, s16, 0x220
	s_add_co_i32 s17, s17, 0x220
	s_add_co_i32 s18, s18, 0x220
	s_add_co_i32 s19, s19, 0x220
	s_add_co_i32 s28, s28, 0x1000
	s_add_co_i32 s29, s29, 0x1000
	s_add_co_i32 s30, s30, 0x1000
	s_add_co_i32 s36, s36, -1
	s_cmp_lg_u32 s36, 0
	s_cbranch_scc1 .Lpx_b3_quad
	.Lpx_b3_tail:
	s_cmp_gt_u32 s37, 0
	s_cbranch_scc0 .Lpx_b3_fold
	buffer_load_b32 v8, v1, s[20:23], s16 offen
	buffer_load_b32 v9, v1, s[20:23], s17 offen
	buffer_load_b32 v10, v1, s[20:23], s18 offen
	buffer_load_b32 v11, v1, s[20:23], s19 offen
	buffer_load_b32 v24, v2, s[20:23], s16 offen offset:8
	buffer_load_b32 v25, v2, s[20:23], s17 offen offset:8
	buffer_load_b32 v26, v2, s[20:23], s18 offen offset:8
	buffer_load_b32 v27, v2, s[20:23], s19 offen offset:8
	buffer_load_b128 v[80:83], v3, s[24:27], s28 offen
	buffer_load_b128 v[84:87], v3, s[24:27], s28 offen offset:16
	buffer_load_b128 v[88:91], v3, s[24:27], s29 offen
	buffer_load_b128 v[92:95], v3, s[24:27], s29 offen offset:16
	buffer_load_b128 v[96:99], v3, s[24:27], s30 offen
	buffer_load_b128 v[100:103], v3, s[24:27], s30 offen offset:16
	s_wait_loadcnt 0x9
	v_and_b32_e32 v46, 0xf0f0f0f, v24
	s_wait_loadcnt 0x8
	v_and_b32_e32 v56, 0xf0f0f0f, v25
	s_wait_loadcnt 0x7
	v_and_b32_e32 v66, 0xf0f0f0f, v26
	s_wait_loadcnt 0x6
	v_and_b32_e32 v76, 0xf0f0f0f, v27
	v_lshrrev_b32_e32 v47, 4, v24
	v_lshrrev_b32_e32 v57, 4, v25
	v_lshrrev_b32_e32 v67, 4, v26
	v_lshrrev_b32_e32 v77, 4, v27
	v_and_b32_e32 v47, 0xf0f0f0f, v47
	v_and_b32_e32 v57, 0xf0f0f0f, v57
	v_and_b32_e32 v67, 0xf0f0f0f, v67
	v_and_b32_e32 v77, 0xf0f0f0f, v77
	v_cvt_f32_ubyte0_e32 v40, v46
	v_cvt_f32_ubyte0_e32 v50, v56
	v_cvt_f32_ubyte0_e32 v60, v66
	v_cvt_f32_ubyte0_e32 v70, v76
	v_cvt_f32_ubyte1_e32 v42, v46
	v_cvt_f32_ubyte1_e32 v52, v56
	v_cvt_f32_ubyte1_e32 v62, v66
	v_cvt_f32_ubyte1_e32 v72, v76
	v_cvt_f32_ubyte2_e32 v44, v46
	v_cvt_f32_ubyte2_e32 v54, v56
	v_cvt_f32_ubyte2_e32 v64, v66
	v_cvt_f32_ubyte2_e32 v74, v76
	v_cvt_f32_ubyte3_e32 v46, v46
	v_cvt_f32_ubyte3_e32 v56, v56
	v_cvt_f32_ubyte3_e32 v66, v66
	v_cvt_f32_ubyte3_e32 v76, v76
	v_cvt_f32_ubyte0_e32 v41, v47
	v_cvt_f32_ubyte0_e32 v51, v57
	v_cvt_f32_ubyte0_e32 v61, v67
	v_cvt_f32_ubyte0_e32 v71, v77
	v_cvt_f32_ubyte1_e32 v43, v47
	v_cvt_f32_ubyte1_e32 v53, v57
	v_cvt_f32_ubyte1_e32 v63, v67
	v_cvt_f32_ubyte1_e32 v73, v77
	v_cvt_f32_ubyte2_e32 v45, v47
	v_cvt_f32_ubyte2_e32 v55, v57
	v_cvt_f32_ubyte2_e32 v65, v67
	v_cvt_f32_ubyte2_e32 v75, v77
	v_cvt_f32_ubyte3_e32 v47, v47
	v_cvt_f32_ubyte3_e32 v57, v57
	v_cvt_f32_ubyte3_e32 v67, v67
	v_cvt_f32_ubyte3_e32 v77, v77
	v_fma_mix_f32 v40, v8, v40, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v9, v50, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v10, v60, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v70, v11, v70, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v41, v8, v41, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v9, v51, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v10, v61, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v71, v11, v71, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v42, v8, v42, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v52, v9, v52, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v62, v10, v62, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v72, v11, v72, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v43, v8, v43, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v53, v9, v53, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v63, v10, v63, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v73, v11, v73, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v44, v8, v44, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v54, v9, v54, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v64, v10, v64, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v74, v11, v74, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v8, v45, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v9, v55, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v65, v10, v65, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v75, v11, v75, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v8, v46, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v9, v56, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v66, v10, v66, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v76, v11, v76, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v8, v47, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v9, v57, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v67, v10, v67, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v77, v11, v77, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x5
	v_dual_mul_f32 v48, v41, v81 :: v_dual_mul_f32 v49, v51, v81
	v_dual_mul_f32 v58, v61, v81 :: v_dual_mul_f32 v59, v71, v81
	s_wait_loadcnt 0x3
	v_dual_mul_f32 v68, v41, v89 :: v_dual_mul_f32 v69, v51, v89
	v_dual_mul_f32 v78, v61, v89 :: v_dual_mul_f32 v79, v71, v89
	v_dual_fmac_f32 v48, v40, v80 :: v_dual_fmac_f32 v49, v50, v80
	v_dual_fmac_f32 v58, v60, v80 :: v_dual_fmac_f32 v59, v70, v80
	v_dual_fmac_f32 v68, v40, v88 :: v_dual_fmac_f32 v69, v50, v88
	v_dual_fmac_f32 v78, v60, v88 :: v_dual_fmac_f32 v79, v70, v88
	v_dual_fmac_f32 v48, v42, v82 :: v_dual_fmac_f32 v49, v52, v82
	v_dual_fmac_f32 v58, v62, v82 :: v_dual_fmac_f32 v59, v72, v82
	v_dual_fmac_f32 v68, v42, v90 :: v_dual_fmac_f32 v69, v52, v90
	v_dual_fmac_f32 v78, v62, v90 :: v_dual_fmac_f32 v79, v72, v90
	v_dual_fmac_f32 v48, v43, v83 :: v_dual_fmac_f32 v49, v53, v83
	v_dual_fmac_f32 v58, v63, v83 :: v_dual_fmac_f32 v59, v73, v83
	v_dual_fmac_f32 v68, v43, v91 :: v_dual_fmac_f32 v69, v53, v91
	v_dual_fmac_f32 v78, v63, v91 :: v_dual_fmac_f32 v79, v73, v91
	v_dual_fmac_f32 v48, v44, v84 :: v_dual_fmac_f32 v49, v54, v84
	v_dual_fmac_f32 v58, v64, v84 :: v_dual_fmac_f32 v59, v74, v84
	s_wait_loadcnt 0x2
	v_dual_fmac_f32 v68, v44, v92 :: v_dual_fmac_f32 v69, v54, v92
	v_dual_fmac_f32 v78, v64, v92 :: v_dual_fmac_f32 v79, v74, v92
	v_dual_fmac_f32 v48, v45, v85 :: v_dual_fmac_f32 v49, v55, v85
	v_dual_fmac_f32 v58, v65, v85 :: v_dual_fmac_f32 v59, v75, v85
	v_dual_fmac_f32 v68, v45, v93 :: v_dual_fmac_f32 v69, v55, v93
	v_dual_fmac_f32 v78, v65, v93 :: v_dual_fmac_f32 v79, v75, v93
	v_dual_fmac_f32 v48, v46, v86 :: v_dual_fmac_f32 v49, v56, v86
	v_dual_fmac_f32 v58, v66, v86 :: v_dual_fmac_f32 v59, v76, v86
	v_dual_fmac_f32 v68, v46, v94 :: v_dual_fmac_f32 v69, v56, v94
	v_dual_fmac_f32 v78, v66, v94 :: v_dual_fmac_f32 v79, v76, v94
	v_dual_fmac_f32 v48, v47, v87 :: v_dual_fmac_f32 v49, v57, v87
	v_dual_fmac_f32 v58, v67, v87 :: v_dual_fmac_f32 v59, v77, v87
	v_dual_fmac_f32 v68, v47, v95 :: v_dual_fmac_f32 v69, v57, v95
	v_dual_fmac_f32 v78, v67, v95 :: v_dual_fmac_f32 v79, v77, v95
	v_dual_add_f32 v176, v176, v48 :: v_dual_add_f32 v177, v177, v49
	v_dual_add_f32 v178, v178, v58 :: v_dual_add_f32 v179, v179, v59
	v_dual_add_f32 v192, v192, v68 :: v_dual_add_f32 v193, v193, v69
	v_dual_add_f32 v194, v194, v78 :: v_dual_add_f32 v195, v195, v79
	s_wait_loadcnt 0x1
	v_dual_mul_f32 v48, v41, v97 :: v_dual_mul_f32 v49, v51, v97
	v_dual_mul_f32 v58, v61, v97 :: v_dual_mul_f32 v59, v71, v97
	v_dual_fmac_f32 v48, v40, v96 :: v_dual_fmac_f32 v49, v50, v96
	v_dual_fmac_f32 v58, v60, v96 :: v_dual_fmac_f32 v59, v70, v96
	v_dual_fmac_f32 v48, v42, v98 :: v_dual_fmac_f32 v49, v52, v98
	v_dual_fmac_f32 v58, v62, v98 :: v_dual_fmac_f32 v59, v72, v98
	v_dual_fmac_f32 v48, v43, v99 :: v_dual_fmac_f32 v49, v53, v99
	v_dual_fmac_f32 v58, v63, v99 :: v_dual_fmac_f32 v59, v73, v99
	s_wait_loadcnt 0x0
	v_dual_fmac_f32 v48, v44, v100 :: v_dual_fmac_f32 v49, v54, v100
	v_dual_fmac_f32 v58, v64, v100 :: v_dual_fmac_f32 v59, v74, v100
	v_dual_fmac_f32 v48, v45, v101 :: v_dual_fmac_f32 v49, v55, v101
	v_dual_fmac_f32 v58, v65, v101 :: v_dual_fmac_f32 v59, v75, v101
	v_dual_fmac_f32 v48, v46, v102 :: v_dual_fmac_f32 v49, v56, v102
	v_dual_fmac_f32 v58, v66, v102 :: v_dual_fmac_f32 v59, v76, v102
	v_dual_fmac_f32 v48, v47, v103 :: v_dual_fmac_f32 v49, v57, v103
	v_dual_fmac_f32 v58, v67, v103 :: v_dual_fmac_f32 v59, v77, v103
	v_dual_add_f32 v208, v208, v48 :: v_dual_add_f32 v209, v209, v49
	v_dual_add_f32 v210, v210, v58 :: v_dual_add_f32 v211, v211, v59
	s_cmp_gt_u32 s37, 1
	s_cbranch_scc0 .Lpx_b3_fold
	buffer_load_b32 v12, v1, s[20:23], s16 offen offset:136
	buffer_load_b32 v13, v1, s[20:23], s17 offen offset:136
	buffer_load_b32 v14, v1, s[20:23], s18 offen offset:136
	buffer_load_b32 v15, v1, s[20:23], s19 offen offset:136
	buffer_load_b32 v28, v2, s[20:23], s16 offen offset:144
	buffer_load_b32 v29, v2, s[20:23], s17 offen offset:144
	buffer_load_b32 v30, v2, s[20:23], s18 offen offset:144
	buffer_load_b32 v31, v2, s[20:23], s19 offen offset:144
	buffer_load_b128 v[80:83], v3, s[24:27], s28 offen offset:1024
	buffer_load_b128 v[84:87], v3, s[24:27], s28 offen offset:1040
	buffer_load_b128 v[88:91], v3, s[24:27], s29 offen offset:1024
	buffer_load_b128 v[92:95], v3, s[24:27], s29 offen offset:1040
	buffer_load_b128 v[96:99], v3, s[24:27], s30 offen offset:1024
	buffer_load_b128 v[100:103], v3, s[24:27], s30 offen offset:1040
	s_wait_loadcnt 0x9
	v_and_b32_e32 v46, 0xf0f0f0f, v28
	s_wait_loadcnt 0x8
	v_and_b32_e32 v56, 0xf0f0f0f, v29
	s_wait_loadcnt 0x7
	v_and_b32_e32 v66, 0xf0f0f0f, v30
	s_wait_loadcnt 0x6
	v_and_b32_e32 v76, 0xf0f0f0f, v31
	v_lshrrev_b32_e32 v47, 4, v28
	v_lshrrev_b32_e32 v57, 4, v29
	v_lshrrev_b32_e32 v67, 4, v30
	v_lshrrev_b32_e32 v77, 4, v31
	v_and_b32_e32 v47, 0xf0f0f0f, v47
	v_and_b32_e32 v57, 0xf0f0f0f, v57
	v_and_b32_e32 v67, 0xf0f0f0f, v67
	v_and_b32_e32 v77, 0xf0f0f0f, v77
	v_cvt_f32_ubyte0_e32 v40, v46
	v_cvt_f32_ubyte0_e32 v50, v56
	v_cvt_f32_ubyte0_e32 v60, v66
	v_cvt_f32_ubyte0_e32 v70, v76
	v_cvt_f32_ubyte1_e32 v42, v46
	v_cvt_f32_ubyte1_e32 v52, v56
	v_cvt_f32_ubyte1_e32 v62, v66
	v_cvt_f32_ubyte1_e32 v72, v76
	v_cvt_f32_ubyte2_e32 v44, v46
	v_cvt_f32_ubyte2_e32 v54, v56
	v_cvt_f32_ubyte2_e32 v64, v66
	v_cvt_f32_ubyte2_e32 v74, v76
	v_cvt_f32_ubyte3_e32 v46, v46
	v_cvt_f32_ubyte3_e32 v56, v56
	v_cvt_f32_ubyte3_e32 v66, v66
	v_cvt_f32_ubyte3_e32 v76, v76
	v_cvt_f32_ubyte0_e32 v41, v47
	v_cvt_f32_ubyte0_e32 v51, v57
	v_cvt_f32_ubyte0_e32 v61, v67
	v_cvt_f32_ubyte0_e32 v71, v77
	v_cvt_f32_ubyte1_e32 v43, v47
	v_cvt_f32_ubyte1_e32 v53, v57
	v_cvt_f32_ubyte1_e32 v63, v67
	v_cvt_f32_ubyte1_e32 v73, v77
	v_cvt_f32_ubyte2_e32 v45, v47
	v_cvt_f32_ubyte2_e32 v55, v57
	v_cvt_f32_ubyte2_e32 v65, v67
	v_cvt_f32_ubyte2_e32 v75, v77
	v_cvt_f32_ubyte3_e32 v47, v47
	v_cvt_f32_ubyte3_e32 v57, v57
	v_cvt_f32_ubyte3_e32 v67, v67
	v_cvt_f32_ubyte3_e32 v77, v77
	v_fma_mix_f32 v40, v12, v40, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v13, v50, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v14, v60, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v70, v15, v70, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v41, v12, v41, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v13, v51, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v14, v61, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v71, v15, v71, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v42, v12, v42, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v52, v13, v52, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v62, v14, v62, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v72, v15, v72, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v43, v12, v43, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v53, v13, v53, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v63, v14, v63, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v73, v15, v73, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v44, v12, v44, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v54, v13, v54, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v64, v14, v64, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v74, v15, v74, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v12, v45, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v13, v55, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v65, v14, v65, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v75, v15, v75, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v12, v46, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v13, v56, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v66, v14, v66, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v76, v15, v76, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v12, v47, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v13, v57, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v67, v14, v67, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v77, v15, v77, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x5
	v_dual_mul_f32 v48, v41, v81 :: v_dual_mul_f32 v49, v51, v81
	v_dual_mul_f32 v58, v61, v81 :: v_dual_mul_f32 v59, v71, v81
	s_wait_loadcnt 0x3
	v_dual_mul_f32 v68, v41, v89 :: v_dual_mul_f32 v69, v51, v89
	v_dual_mul_f32 v78, v61, v89 :: v_dual_mul_f32 v79, v71, v89
	v_dual_fmac_f32 v48, v40, v80 :: v_dual_fmac_f32 v49, v50, v80
	v_dual_fmac_f32 v58, v60, v80 :: v_dual_fmac_f32 v59, v70, v80
	v_dual_fmac_f32 v68, v40, v88 :: v_dual_fmac_f32 v69, v50, v88
	v_dual_fmac_f32 v78, v60, v88 :: v_dual_fmac_f32 v79, v70, v88
	v_dual_fmac_f32 v48, v42, v82 :: v_dual_fmac_f32 v49, v52, v82
	v_dual_fmac_f32 v58, v62, v82 :: v_dual_fmac_f32 v59, v72, v82
	v_dual_fmac_f32 v68, v42, v90 :: v_dual_fmac_f32 v69, v52, v90
	v_dual_fmac_f32 v78, v62, v90 :: v_dual_fmac_f32 v79, v72, v90
	v_dual_fmac_f32 v48, v43, v83 :: v_dual_fmac_f32 v49, v53, v83
	v_dual_fmac_f32 v58, v63, v83 :: v_dual_fmac_f32 v59, v73, v83
	v_dual_fmac_f32 v68, v43, v91 :: v_dual_fmac_f32 v69, v53, v91
	v_dual_fmac_f32 v78, v63, v91 :: v_dual_fmac_f32 v79, v73, v91
	v_dual_fmac_f32 v48, v44, v84 :: v_dual_fmac_f32 v49, v54, v84
	v_dual_fmac_f32 v58, v64, v84 :: v_dual_fmac_f32 v59, v74, v84
	s_wait_loadcnt 0x2
	v_dual_fmac_f32 v68, v44, v92 :: v_dual_fmac_f32 v69, v54, v92
	v_dual_fmac_f32 v78, v64, v92 :: v_dual_fmac_f32 v79, v74, v92
	v_dual_fmac_f32 v48, v45, v85 :: v_dual_fmac_f32 v49, v55, v85
	v_dual_fmac_f32 v58, v65, v85 :: v_dual_fmac_f32 v59, v75, v85
	v_dual_fmac_f32 v68, v45, v93 :: v_dual_fmac_f32 v69, v55, v93
	v_dual_fmac_f32 v78, v65, v93 :: v_dual_fmac_f32 v79, v75, v93
	v_dual_fmac_f32 v48, v46, v86 :: v_dual_fmac_f32 v49, v56, v86
	v_dual_fmac_f32 v58, v66, v86 :: v_dual_fmac_f32 v59, v76, v86
	v_dual_fmac_f32 v68, v46, v94 :: v_dual_fmac_f32 v69, v56, v94
	v_dual_fmac_f32 v78, v66, v94 :: v_dual_fmac_f32 v79, v76, v94
	v_dual_fmac_f32 v48, v47, v87 :: v_dual_fmac_f32 v49, v57, v87
	v_dual_fmac_f32 v58, v67, v87 :: v_dual_fmac_f32 v59, v77, v87
	v_dual_fmac_f32 v68, v47, v95 :: v_dual_fmac_f32 v69, v57, v95
	v_dual_fmac_f32 v78, v67, v95 :: v_dual_fmac_f32 v79, v77, v95
	v_dual_add_f32 v180, v180, v48 :: v_dual_add_f32 v181, v181, v49
	v_dual_add_f32 v182, v182, v58 :: v_dual_add_f32 v183, v183, v59
	v_dual_add_f32 v196, v196, v68 :: v_dual_add_f32 v197, v197, v69
	v_dual_add_f32 v198, v198, v78 :: v_dual_add_f32 v199, v199, v79
	s_wait_loadcnt 0x1
	v_dual_mul_f32 v48, v41, v97 :: v_dual_mul_f32 v49, v51, v97
	v_dual_mul_f32 v58, v61, v97 :: v_dual_mul_f32 v59, v71, v97
	v_dual_fmac_f32 v48, v40, v96 :: v_dual_fmac_f32 v49, v50, v96
	v_dual_fmac_f32 v58, v60, v96 :: v_dual_fmac_f32 v59, v70, v96
	v_dual_fmac_f32 v48, v42, v98 :: v_dual_fmac_f32 v49, v52, v98
	v_dual_fmac_f32 v58, v62, v98 :: v_dual_fmac_f32 v59, v72, v98
	v_dual_fmac_f32 v48, v43, v99 :: v_dual_fmac_f32 v49, v53, v99
	v_dual_fmac_f32 v58, v63, v99 :: v_dual_fmac_f32 v59, v73, v99
	s_wait_loadcnt 0x0
	v_dual_fmac_f32 v48, v44, v100 :: v_dual_fmac_f32 v49, v54, v100
	v_dual_fmac_f32 v58, v64, v100 :: v_dual_fmac_f32 v59, v74, v100
	v_dual_fmac_f32 v48, v45, v101 :: v_dual_fmac_f32 v49, v55, v101
	v_dual_fmac_f32 v58, v65, v101 :: v_dual_fmac_f32 v59, v75, v101
	v_dual_fmac_f32 v48, v46, v102 :: v_dual_fmac_f32 v49, v56, v102
	v_dual_fmac_f32 v58, v66, v102 :: v_dual_fmac_f32 v59, v76, v102
	v_dual_fmac_f32 v48, v47, v103 :: v_dual_fmac_f32 v49, v57, v103
	v_dual_fmac_f32 v58, v67, v103 :: v_dual_fmac_f32 v59, v77, v103
	v_dual_add_f32 v212, v212, v48 :: v_dual_add_f32 v213, v213, v49
	v_dual_add_f32 v214, v214, v58 :: v_dual_add_f32 v215, v215, v59
	s_cmp_gt_u32 s37, 2
	s_cbranch_scc0 .Lpx_b3_fold
	buffer_load_b32 v16, v1, s[20:23], s16 offen offset:272
	buffer_load_b32 v17, v1, s[20:23], s17 offen offset:272
	buffer_load_b32 v18, v1, s[20:23], s18 offen offset:272
	buffer_load_b32 v19, v1, s[20:23], s19 offen offset:272
	buffer_load_b32 v32, v2, s[20:23], s16 offen offset:280
	buffer_load_b32 v33, v2, s[20:23], s17 offen offset:280
	buffer_load_b32 v34, v2, s[20:23], s18 offen offset:280
	buffer_load_b32 v35, v2, s[20:23], s19 offen offset:280
	buffer_load_b128 v[80:83], v3, s[24:27], s28 offen offset:2048
	buffer_load_b128 v[84:87], v3, s[24:27], s28 offen offset:2064
	buffer_load_b128 v[88:91], v3, s[24:27], s29 offen offset:2048
	buffer_load_b128 v[92:95], v3, s[24:27], s29 offen offset:2064
	buffer_load_b128 v[96:99], v3, s[24:27], s30 offen offset:2048
	buffer_load_b128 v[100:103], v3, s[24:27], s30 offen offset:2064
	s_wait_loadcnt 0x9
	v_and_b32_e32 v46, 0xf0f0f0f, v32
	s_wait_loadcnt 0x8
	v_and_b32_e32 v56, 0xf0f0f0f, v33
	s_wait_loadcnt 0x7
	v_and_b32_e32 v66, 0xf0f0f0f, v34
	s_wait_loadcnt 0x6
	v_and_b32_e32 v76, 0xf0f0f0f, v35
	v_lshrrev_b32_e32 v47, 4, v32
	v_lshrrev_b32_e32 v57, 4, v33
	v_lshrrev_b32_e32 v67, 4, v34
	v_lshrrev_b32_e32 v77, 4, v35
	v_and_b32_e32 v47, 0xf0f0f0f, v47
	v_and_b32_e32 v57, 0xf0f0f0f, v57
	v_and_b32_e32 v67, 0xf0f0f0f, v67
	v_and_b32_e32 v77, 0xf0f0f0f, v77
	v_cvt_f32_ubyte0_e32 v40, v46
	v_cvt_f32_ubyte0_e32 v50, v56
	v_cvt_f32_ubyte0_e32 v60, v66
	v_cvt_f32_ubyte0_e32 v70, v76
	v_cvt_f32_ubyte1_e32 v42, v46
	v_cvt_f32_ubyte1_e32 v52, v56
	v_cvt_f32_ubyte1_e32 v62, v66
	v_cvt_f32_ubyte1_e32 v72, v76
	v_cvt_f32_ubyte2_e32 v44, v46
	v_cvt_f32_ubyte2_e32 v54, v56
	v_cvt_f32_ubyte2_e32 v64, v66
	v_cvt_f32_ubyte2_e32 v74, v76
	v_cvt_f32_ubyte3_e32 v46, v46
	v_cvt_f32_ubyte3_e32 v56, v56
	v_cvt_f32_ubyte3_e32 v66, v66
	v_cvt_f32_ubyte3_e32 v76, v76
	v_cvt_f32_ubyte0_e32 v41, v47
	v_cvt_f32_ubyte0_e32 v51, v57
	v_cvt_f32_ubyte0_e32 v61, v67
	v_cvt_f32_ubyte0_e32 v71, v77
	v_cvt_f32_ubyte1_e32 v43, v47
	v_cvt_f32_ubyte1_e32 v53, v57
	v_cvt_f32_ubyte1_e32 v63, v67
	v_cvt_f32_ubyte1_e32 v73, v77
	v_cvt_f32_ubyte2_e32 v45, v47
	v_cvt_f32_ubyte2_e32 v55, v57
	v_cvt_f32_ubyte2_e32 v65, v67
	v_cvt_f32_ubyte2_e32 v75, v77
	v_cvt_f32_ubyte3_e32 v47, v47
	v_cvt_f32_ubyte3_e32 v57, v57
	v_cvt_f32_ubyte3_e32 v67, v67
	v_cvt_f32_ubyte3_e32 v77, v77
	v_fma_mix_f32 v40, v16, v40, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v17, v50, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v18, v60, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v70, v19, v70, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v41, v16, v41, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v17, v51, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v18, v61, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v71, v19, v71, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v42, v16, v42, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v52, v17, v52, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v62, v18, v62, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v72, v19, v72, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v43, v16, v43, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v53, v17, v53, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v63, v18, v63, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v73, v19, v73, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v44, v16, v44, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v54, v17, v54, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v64, v18, v64, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v74, v19, v74, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v16, v45, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v17, v55, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v65, v18, v65, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v75, v19, v75, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v16, v46, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v17, v56, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v66, v18, v66, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v76, v19, v76, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v16, v47, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v17, v57, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v67, v18, v67, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v77, v19, v77, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x5
	v_dual_mul_f32 v48, v41, v81 :: v_dual_mul_f32 v49, v51, v81
	v_dual_mul_f32 v58, v61, v81 :: v_dual_mul_f32 v59, v71, v81
	s_wait_loadcnt 0x3
	v_dual_mul_f32 v68, v41, v89 :: v_dual_mul_f32 v69, v51, v89
	v_dual_mul_f32 v78, v61, v89 :: v_dual_mul_f32 v79, v71, v89
	v_dual_fmac_f32 v48, v40, v80 :: v_dual_fmac_f32 v49, v50, v80
	v_dual_fmac_f32 v58, v60, v80 :: v_dual_fmac_f32 v59, v70, v80
	v_dual_fmac_f32 v68, v40, v88 :: v_dual_fmac_f32 v69, v50, v88
	v_dual_fmac_f32 v78, v60, v88 :: v_dual_fmac_f32 v79, v70, v88
	v_dual_fmac_f32 v48, v42, v82 :: v_dual_fmac_f32 v49, v52, v82
	v_dual_fmac_f32 v58, v62, v82 :: v_dual_fmac_f32 v59, v72, v82
	v_dual_fmac_f32 v68, v42, v90 :: v_dual_fmac_f32 v69, v52, v90
	v_dual_fmac_f32 v78, v62, v90 :: v_dual_fmac_f32 v79, v72, v90
	v_dual_fmac_f32 v48, v43, v83 :: v_dual_fmac_f32 v49, v53, v83
	v_dual_fmac_f32 v58, v63, v83 :: v_dual_fmac_f32 v59, v73, v83
	v_dual_fmac_f32 v68, v43, v91 :: v_dual_fmac_f32 v69, v53, v91
	v_dual_fmac_f32 v78, v63, v91 :: v_dual_fmac_f32 v79, v73, v91
	v_dual_fmac_f32 v48, v44, v84 :: v_dual_fmac_f32 v49, v54, v84
	v_dual_fmac_f32 v58, v64, v84 :: v_dual_fmac_f32 v59, v74, v84
	s_wait_loadcnt 0x2
	v_dual_fmac_f32 v68, v44, v92 :: v_dual_fmac_f32 v69, v54, v92
	v_dual_fmac_f32 v78, v64, v92 :: v_dual_fmac_f32 v79, v74, v92
	v_dual_fmac_f32 v48, v45, v85 :: v_dual_fmac_f32 v49, v55, v85
	v_dual_fmac_f32 v58, v65, v85 :: v_dual_fmac_f32 v59, v75, v85
	v_dual_fmac_f32 v68, v45, v93 :: v_dual_fmac_f32 v69, v55, v93
	v_dual_fmac_f32 v78, v65, v93 :: v_dual_fmac_f32 v79, v75, v93
	v_dual_fmac_f32 v48, v46, v86 :: v_dual_fmac_f32 v49, v56, v86
	v_dual_fmac_f32 v58, v66, v86 :: v_dual_fmac_f32 v59, v76, v86
	v_dual_fmac_f32 v68, v46, v94 :: v_dual_fmac_f32 v69, v56, v94
	v_dual_fmac_f32 v78, v66, v94 :: v_dual_fmac_f32 v79, v76, v94
	v_dual_fmac_f32 v48, v47, v87 :: v_dual_fmac_f32 v49, v57, v87
	v_dual_fmac_f32 v58, v67, v87 :: v_dual_fmac_f32 v59, v77, v87
	v_dual_fmac_f32 v68, v47, v95 :: v_dual_fmac_f32 v69, v57, v95
	v_dual_fmac_f32 v78, v67, v95 :: v_dual_fmac_f32 v79, v77, v95
	v_dual_add_f32 v184, v184, v48 :: v_dual_add_f32 v185, v185, v49
	v_dual_add_f32 v186, v186, v58 :: v_dual_add_f32 v187, v187, v59
	v_dual_add_f32 v200, v200, v68 :: v_dual_add_f32 v201, v201, v69
	v_dual_add_f32 v202, v202, v78 :: v_dual_add_f32 v203, v203, v79
	s_wait_loadcnt 0x1
	v_dual_mul_f32 v48, v41, v97 :: v_dual_mul_f32 v49, v51, v97
	v_dual_mul_f32 v58, v61, v97 :: v_dual_mul_f32 v59, v71, v97
	v_dual_fmac_f32 v48, v40, v96 :: v_dual_fmac_f32 v49, v50, v96
	v_dual_fmac_f32 v58, v60, v96 :: v_dual_fmac_f32 v59, v70, v96
	v_dual_fmac_f32 v48, v42, v98 :: v_dual_fmac_f32 v49, v52, v98
	v_dual_fmac_f32 v58, v62, v98 :: v_dual_fmac_f32 v59, v72, v98
	v_dual_fmac_f32 v48, v43, v99 :: v_dual_fmac_f32 v49, v53, v99
	v_dual_fmac_f32 v58, v63, v99 :: v_dual_fmac_f32 v59, v73, v99
	s_wait_loadcnt 0x0
	v_dual_fmac_f32 v48, v44, v100 :: v_dual_fmac_f32 v49, v54, v100
	v_dual_fmac_f32 v58, v64, v100 :: v_dual_fmac_f32 v59, v74, v100
	v_dual_fmac_f32 v48, v45, v101 :: v_dual_fmac_f32 v49, v55, v101
	v_dual_fmac_f32 v58, v65, v101 :: v_dual_fmac_f32 v59, v75, v101
	v_dual_fmac_f32 v48, v46, v102 :: v_dual_fmac_f32 v49, v56, v102
	v_dual_fmac_f32 v58, v66, v102 :: v_dual_fmac_f32 v59, v76, v102
	v_dual_fmac_f32 v48, v47, v103 :: v_dual_fmac_f32 v49, v57, v103
	v_dual_fmac_f32 v58, v67, v103 :: v_dual_fmac_f32 v59, v77, v103
	v_dual_add_f32 v216, v216, v48 :: v_dual_add_f32 v217, v217, v49
	v_dual_add_f32 v218, v218, v58 :: v_dual_add_f32 v219, v219, v59
	.Lpx_b3_fold:
	v_dual_add_f32 v176, v176, v180 :: v_dual_add_f32 v177, v177, v181
	v_dual_add_f32 v178, v178, v182 :: v_dual_add_f32 v179, v179, v183
	v_dual_add_f32 v184, v184, v188 :: v_dual_add_f32 v185, v185, v189
	v_dual_add_f32 v186, v186, v190 :: v_dual_add_f32 v187, v187, v191
	v_dual_add_f32 v176, v176, v184 :: v_dual_add_f32 v177, v177, v185
	v_dual_add_f32 v178, v178, v186 :: v_dual_add_f32 v179, v179, v187
	v_dual_add_f32 v192, v192, v196 :: v_dual_add_f32 v193, v193, v197
	v_dual_add_f32 v194, v194, v198 :: v_dual_add_f32 v195, v195, v199
	v_dual_add_f32 v200, v200, v204 :: v_dual_add_f32 v201, v201, v205
	v_dual_add_f32 v202, v202, v206 :: v_dual_add_f32 v203, v203, v207
	v_dual_add_f32 v192, v192, v200 :: v_dual_add_f32 v193, v193, v201
	v_dual_add_f32 v194, v194, v202 :: v_dual_add_f32 v195, v195, v203
	v_dual_add_f32 v208, v208, v212 :: v_dual_add_f32 v209, v209, v213
	v_dual_add_f32 v210, v210, v214 :: v_dual_add_f32 v211, v211, v215
	v_dual_add_f32 v216, v216, v220 :: v_dual_add_f32 v217, v217, v221
	v_dual_add_f32 v218, v218, v222 :: v_dual_add_f32 v219, v219, v223
	v_dual_add_f32 v208, v208, v216 :: v_dual_add_f32 v209, v209, v217
	v_dual_add_f32 v210, v210, v218 :: v_dual_add_f32 v211, v211, v219
	ds_swizzle_b32 v8, v176 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v9, v177 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v10, v178 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v11, v179 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v12, v192 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v13, v193 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v14, v194 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v15, v195 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v16, v208 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v17, v209 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v18, v210 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v19, v211 offset:swizzle(BITMASK_PERM,"1pppp")
	s_wait_dscnt 0xb
	v_add_f32_e32 v176, v176, v8
	s_wait_dscnt 0xa
	v_add_f32_e32 v177, v177, v9
	s_wait_dscnt 0x9
	v_add_f32_e32 v178, v178, v10
	s_wait_dscnt 0x8
	v_add_f32_e32 v179, v179, v11
	s_wait_dscnt 0x7
	v_add_f32_e32 v192, v192, v12
	s_wait_dscnt 0x6
	v_add_f32_e32 v193, v193, v13
	s_wait_dscnt 0x5
	v_add_f32_e32 v194, v194, v14
	s_wait_dscnt 0x4
	v_add_f32_e32 v195, v195, v15
	s_wait_dscnt 0x3
	v_add_f32_e32 v208, v208, v16
	s_wait_dscnt 0x2
	v_add_f32_e32 v209, v209, v17
	s_wait_dscnt 0x1
	v_add_f32_e32 v210, v210, v18
	s_wait_dscnt 0x0
	v_add_f32_e32 v211, v211, v19
	v_cmp_gt_u32_e32 vcc_lo, 24, v0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v4, 0, 8, vcc_lo
	v_add_lshl_u32 v4, v4, v0, 2
	ds_bpermute_b32 v8, v4, v176
	ds_bpermute_b32 v9, v4, v177
	ds_bpermute_b32 v10, v4, v178
	ds_bpermute_b32 v11, v4, v179
	ds_bpermute_b32 v12, v4, v192
	ds_bpermute_b32 v13, v4, v193
	ds_bpermute_b32 v14, v4, v194
	ds_bpermute_b32 v15, v4, v195
	ds_bpermute_b32 v16, v4, v208
	ds_bpermute_b32 v17, v4, v209
	ds_bpermute_b32 v18, v4, v210
	ds_bpermute_b32 v19, v4, v211
	s_wait_dscnt 0xb
	v_add_f32_e32 v176, v176, v8
	s_wait_dscnt 0xa
	v_add_f32_e32 v177, v177, v9
	s_wait_dscnt 0x9
	v_add_f32_e32 v178, v178, v10
	s_wait_dscnt 0x8
	v_add_f32_e32 v179, v179, v11
	s_wait_dscnt 0x7
	v_add_f32_e32 v192, v192, v12
	s_wait_dscnt 0x6
	v_add_f32_e32 v193, v193, v13
	s_wait_dscnt 0x5
	v_add_f32_e32 v194, v194, v14
	s_wait_dscnt 0x4
	v_add_f32_e32 v195, v195, v15
	s_wait_dscnt 0x3
	v_add_f32_e32 v208, v208, v16
	s_wait_dscnt 0x2
	v_add_f32_e32 v209, v209, v17
	s_wait_dscnt 0x1
	v_add_f32_e32 v210, v210, v18
	s_wait_dscnt 0x0
	v_add_f32_e32 v211, v211, v19
	v_cmp_gt_u32_e32 vcc_lo, 28, v0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v4, 0, 4, vcc_lo
	v_add_lshl_u32 v4, v4, v0, 2
	ds_bpermute_b32 v8, v4, v176
	ds_bpermute_b32 v9, v4, v177
	ds_bpermute_b32 v10, v4, v178
	ds_bpermute_b32 v11, v4, v179
	ds_bpermute_b32 v12, v4, v192
	ds_bpermute_b32 v13, v4, v193
	ds_bpermute_b32 v14, v4, v194
	ds_bpermute_b32 v15, v4, v195
	ds_bpermute_b32 v16, v4, v208
	ds_bpermute_b32 v17, v4, v209
	ds_bpermute_b32 v18, v4, v210
	ds_bpermute_b32 v19, v4, v211
	s_wait_dscnt 0xb
	v_add_f32_e32 v176, v176, v8
	s_wait_dscnt 0xa
	v_add_f32_e32 v177, v177, v9
	s_wait_dscnt 0x9
	v_add_f32_e32 v178, v178, v10
	s_wait_dscnt 0x8
	v_add_f32_e32 v179, v179, v11
	s_wait_dscnt 0x7
	v_add_f32_e32 v192, v192, v12
	s_wait_dscnt 0x6
	v_add_f32_e32 v193, v193, v13
	s_wait_dscnt 0x5
	v_add_f32_e32 v194, v194, v14
	s_wait_dscnt 0x4
	v_add_f32_e32 v195, v195, v15
	s_wait_dscnt 0x3
	v_add_f32_e32 v208, v208, v16
	s_wait_dscnt 0x2
	v_add_f32_e32 v209, v209, v17
	s_wait_dscnt 0x1
	v_add_f32_e32 v210, v210, v18
	s_wait_dscnt 0x0
	v_add_f32_e32 v211, v211, v19
	v_cmp_gt_u32_e32 vcc_lo, 30, v0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v4, 0, 2, vcc_lo
	v_add_lshl_u32 v4, v4, v0, 2
	ds_bpermute_b32 v8, v4, v176
	ds_bpermute_b32 v9, v4, v177
	ds_bpermute_b32 v10, v4, v178
	ds_bpermute_b32 v11, v4, v179
	ds_bpermute_b32 v12, v4, v192
	ds_bpermute_b32 v13, v4, v193
	ds_bpermute_b32 v14, v4, v194
	ds_bpermute_b32 v15, v4, v195
	ds_bpermute_b32 v16, v4, v208
	ds_bpermute_b32 v17, v4, v209
	ds_bpermute_b32 v18, v4, v210
	ds_bpermute_b32 v19, v4, v211
	s_wait_dscnt 0xb
	v_add_f32_e32 v176, v176, v8
	s_wait_dscnt 0xa
	v_add_f32_e32 v177, v177, v9
	s_wait_dscnt 0x9
	v_add_f32_e32 v178, v178, v10
	s_wait_dscnt 0x8
	v_add_f32_e32 v179, v179, v11
	s_wait_dscnt 0x7
	v_add_f32_e32 v192, v192, v12
	s_wait_dscnt 0x6
	v_add_f32_e32 v193, v193, v13
	s_wait_dscnt 0x5
	v_add_f32_e32 v194, v194, v14
	s_wait_dscnt 0x4
	v_add_f32_e32 v195, v195, v15
	s_wait_dscnt 0x3
	v_add_f32_e32 v208, v208, v16
	s_wait_dscnt 0x2
	v_add_f32_e32 v209, v209, v17
	s_wait_dscnt 0x1
	v_add_f32_e32 v210, v210, v18
	s_wait_dscnt 0x0
	v_add_f32_e32 v211, v211, v19
	v_cmp_gt_u32_e32 vcc_lo, 31, v0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v4, 0, 1, vcc_lo
	v_add_lshl_u32 v4, v4, v0, 2
	ds_bpermute_b32 v8, v4, v176
	ds_bpermute_b32 v9, v4, v177
	ds_bpermute_b32 v10, v4, v178
	ds_bpermute_b32 v11, v4, v179
	ds_bpermute_b32 v12, v4, v192
	ds_bpermute_b32 v13, v4, v193
	ds_bpermute_b32 v14, v4, v194
	ds_bpermute_b32 v15, v4, v195
	ds_bpermute_b32 v16, v4, v208
	ds_bpermute_b32 v17, v4, v209
	ds_bpermute_b32 v18, v4, v210
	ds_bpermute_b32 v19, v4, v211
	s_wait_dscnt 0xb
	v_add_f32_e32 v176, v176, v8
	s_wait_dscnt 0xa
	v_add_f32_e32 v177, v177, v9
	s_wait_dscnt 0x9
	v_add_f32_e32 v178, v178, v10
	s_wait_dscnt 0x8
	v_add_f32_e32 v179, v179, v11
	s_wait_dscnt 0x7
	v_add_f32_e32 v192, v192, v12
	s_wait_dscnt 0x6
	v_add_f32_e32 v193, v193, v13
	s_wait_dscnt 0x5
	v_add_f32_e32 v194, v194, v14
	s_wait_dscnt 0x4
	v_add_f32_e32 v195, v195, v15
	s_wait_dscnt 0x3
	v_add_f32_e32 v208, v208, v16
	s_wait_dscnt 0x2
	v_add_f32_e32 v209, v209, v17
	s_wait_dscnt 0x1
	v_add_f32_e32 v210, v210, v18
	s_wait_dscnt 0x0
	v_add_f32_e32 v211, v211, v19
	v_cmpx_eq_u32_e32 0, v0
	v_mov_b32_e32 v40, s42
	s_mul_i32 s40, s39, 1
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s40, s40, s42
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v41, s40
	s_mul_i32 s40, s39, 2
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s40, s40, s42
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v42, s40
	global_store_b32 v40, v176, s[8:9]
	global_store_b32 v41, v192, s[8:9]
	global_store_b32 v42, v208, s[8:9]
	s_cmp_le_i32 s13, 1
	s_cbranch_scc1 .Lpx_b3_stored
	global_store_b32 v40, v177, s[8:9] offset:4
	global_store_b32 v41, v193, s[8:9] offset:4
	global_store_b32 v42, v209, s[8:9] offset:4
	s_cmp_le_i32 s13, 2
	s_cbranch_scc1 .Lpx_b3_stored
	global_store_b32 v40, v178, s[8:9] offset:8
	global_store_b32 v41, v194, s[8:9] offset:8
	global_store_b32 v42, v210, s[8:9] offset:8
	s_cmp_le_i32 s13, 3
	s_cbranch_scc1 .Lpx_b3_stored
	global_store_b32 v40, v179, s[8:9] offset:12
	global_store_b32 v41, v195, s[8:9] offset:12
	global_store_b32 v42, v211, s[8:9] offset:12
	.Lpx_b3_stored:
	s_wait_storecnt 0x0
	s_branch .Lpx_end
	.Lpx_b4:
	v_mov_b32_e32 v192, 0
	v_mov_b32_e32 v193, 0
	v_mov_b32_e32 v194, 0
	v_mov_b32_e32 v195, 0
	v_mov_b32_e32 v196, 0
	v_mov_b32_e32 v197, 0
	v_mov_b32_e32 v198, 0
	v_mov_b32_e32 v199, 0
	v_mov_b32_e32 v200, 0
	v_mov_b32_e32 v201, 0
	v_mov_b32_e32 v202, 0
	v_mov_b32_e32 v203, 0
	v_mov_b32_e32 v204, 0
	v_mov_b32_e32 v205, 0
	v_mov_b32_e32 v206, 0
	v_mov_b32_e32 v207, 0
	v_mov_b32_e32 v208, 0
	v_mov_b32_e32 v209, 0
	v_mov_b32_e32 v210, 0
	v_mov_b32_e32 v211, 0
	v_mov_b32_e32 v212, 0
	v_mov_b32_e32 v213, 0
	v_mov_b32_e32 v214, 0
	v_mov_b32_e32 v215, 0
	v_mov_b32_e32 v216, 0
	v_mov_b32_e32 v217, 0
	v_mov_b32_e32 v218, 0
	v_mov_b32_e32 v219, 0
	v_mov_b32_e32 v220, 0
	v_mov_b32_e32 v221, 0
	v_mov_b32_e32 v222, 0
	v_mov_b32_e32 v223, 0
	v_mov_b32_e32 v224, 0
	v_mov_b32_e32 v225, 0
	v_mov_b32_e32 v226, 0
	v_mov_b32_e32 v227, 0
	v_mov_b32_e32 v228, 0
	v_mov_b32_e32 v229, 0
	v_mov_b32_e32 v230, 0
	v_mov_b32_e32 v231, 0
	v_mov_b32_e32 v232, 0
	v_mov_b32_e32 v233, 0
	v_mov_b32_e32 v234, 0
	v_mov_b32_e32 v235, 0
	v_mov_b32_e32 v236, 0
	v_mov_b32_e32 v237, 0
	v_mov_b32_e32 v238, 0
	v_mov_b32_e32 v239, 0
	v_mov_b32_e32 v240, 0
	v_mov_b32_e32 v241, 0
	v_mov_b32_e32 v242, 0
	v_mov_b32_e32 v243, 0
	v_mov_b32_e32 v244, 0
	v_mov_b32_e32 v245, 0
	v_mov_b32_e32 v246, 0
	v_mov_b32_e32 v247, 0
	v_mov_b32_e32 v248, 0
	v_mov_b32_e32 v249, 0
	v_mov_b32_e32 v250, 0
	v_mov_b32_e32 v251, 0
	v_mov_b32_e32 v252, 0
	v_mov_b32_e32 v253, 0
	v_mov_b32_e32 v254, 0
	v_mov_b32_e32 v255, 0
	s_lshr_b32 s36, s14, 2
	s_and_b32 s37, s14, 3
	s_cmp_eq_u32 s36, 0
	s_cbranch_scc1 .Lpx_b4_tail
	.Lpx_b4_quad:
	buffer_load_b32 v8, v1, s[20:23], s16 offen
	buffer_load_b32 v9, v1, s[20:23], s17 offen
	buffer_load_b32 v10, v1, s[20:23], s18 offen
	buffer_load_b32 v11, v1, s[20:23], s19 offen
	buffer_load_b32 v24, v2, s[20:23], s16 offen offset:8
	buffer_load_b32 v25, v2, s[20:23], s17 offen offset:8
	buffer_load_b32 v26, v2, s[20:23], s18 offen offset:8
	buffer_load_b32 v27, v2, s[20:23], s19 offen offset:8
	buffer_load_b128 v[80:83], v3, s[24:27], s28 offen
	buffer_load_b128 v[84:87], v3, s[24:27], s28 offen offset:16
	buffer_load_b128 v[88:91], v3, s[24:27], s29 offen
	buffer_load_b128 v[92:95], v3, s[24:27], s29 offen offset:16
	buffer_load_b128 v[96:99], v3, s[24:27], s30 offen
	buffer_load_b128 v[100:103], v3, s[24:27], s30 offen offset:16
	buffer_load_b128 v[104:107], v3, s[24:27], s31 offen
	buffer_load_b128 v[108:111], v3, s[24:27], s31 offen offset:16
	buffer_load_b32 v12, v1, s[20:23], s16 offen offset:136
	buffer_load_b32 v13, v1, s[20:23], s17 offen offset:136
	buffer_load_b32 v14, v1, s[20:23], s18 offen offset:136
	buffer_load_b32 v15, v1, s[20:23], s19 offen offset:136
	buffer_load_b32 v28, v2, s[20:23], s16 offen offset:144
	buffer_load_b32 v29, v2, s[20:23], s17 offen offset:144
	buffer_load_b32 v30, v2, s[20:23], s18 offen offset:144
	buffer_load_b32 v31, v2, s[20:23], s19 offen offset:144
	buffer_load_b32 v16, v1, s[20:23], s16 offen offset:272
	buffer_load_b32 v17, v1, s[20:23], s17 offen offset:272
	buffer_load_b32 v18, v1, s[20:23], s18 offen offset:272
	buffer_load_b32 v19, v1, s[20:23], s19 offen offset:272
	buffer_load_b32 v32, v2, s[20:23], s16 offen offset:280
	buffer_load_b32 v33, v2, s[20:23], s17 offen offset:280
	buffer_load_b32 v34, v2, s[20:23], s18 offen offset:280
	buffer_load_b32 v35, v2, s[20:23], s19 offen offset:280
	buffer_load_b32 v20, v1, s[20:23], s16 offen offset:408
	buffer_load_b32 v21, v1, s[20:23], s17 offen offset:408
	buffer_load_b32 v22, v1, s[20:23], s18 offen offset:408
	buffer_load_b32 v23, v1, s[20:23], s19 offen offset:408
	buffer_load_b32 v36, v2, s[20:23], s16 offen offset:416
	buffer_load_b32 v37, v2, s[20:23], s17 offen offset:416
	buffer_load_b32 v38, v2, s[20:23], s18 offen offset:416
	buffer_load_b32 v39, v2, s[20:23], s19 offen offset:416
	s_wait_loadcnt 0x23
	v_and_b32_e32 v46, 0xf0f0f0f, v24
	s_wait_loadcnt 0x22
	v_and_b32_e32 v56, 0xf0f0f0f, v25
	s_wait_loadcnt 0x21
	v_and_b32_e32 v66, 0xf0f0f0f, v26
	s_wait_loadcnt 0x20
	v_and_b32_e32 v76, 0xf0f0f0f, v27
	v_lshrrev_b32_e32 v47, 4, v24
	v_lshrrev_b32_e32 v57, 4, v25
	v_lshrrev_b32_e32 v67, 4, v26
	v_lshrrev_b32_e32 v77, 4, v27
	v_and_b32_e32 v47, 0xf0f0f0f, v47
	v_and_b32_e32 v57, 0xf0f0f0f, v57
	v_and_b32_e32 v67, 0xf0f0f0f, v67
	v_and_b32_e32 v77, 0xf0f0f0f, v77
	v_cvt_f32_ubyte0_e32 v40, v46
	v_cvt_f32_ubyte0_e32 v50, v56
	v_cvt_f32_ubyte0_e32 v60, v66
	v_cvt_f32_ubyte0_e32 v70, v76
	v_cvt_f32_ubyte1_e32 v42, v46
	v_cvt_f32_ubyte1_e32 v52, v56
	v_cvt_f32_ubyte1_e32 v62, v66
	v_cvt_f32_ubyte1_e32 v72, v76
	v_cvt_f32_ubyte2_e32 v44, v46
	v_cvt_f32_ubyte2_e32 v54, v56
	v_cvt_f32_ubyte2_e32 v64, v66
	v_cvt_f32_ubyte2_e32 v74, v76
	v_cvt_f32_ubyte3_e32 v46, v46
	v_cvt_f32_ubyte3_e32 v56, v56
	v_cvt_f32_ubyte3_e32 v66, v66
	v_cvt_f32_ubyte3_e32 v76, v76
	v_cvt_f32_ubyte0_e32 v41, v47
	v_cvt_f32_ubyte0_e32 v51, v57
	v_cvt_f32_ubyte0_e32 v61, v67
	v_cvt_f32_ubyte0_e32 v71, v77
	v_cvt_f32_ubyte1_e32 v43, v47
	v_cvt_f32_ubyte1_e32 v53, v57
	v_cvt_f32_ubyte1_e32 v63, v67
	v_cvt_f32_ubyte1_e32 v73, v77
	v_cvt_f32_ubyte2_e32 v45, v47
	v_cvt_f32_ubyte2_e32 v55, v57
	v_cvt_f32_ubyte2_e32 v65, v67
	v_cvt_f32_ubyte2_e32 v75, v77
	v_cvt_f32_ubyte3_e32 v47, v47
	v_cvt_f32_ubyte3_e32 v57, v57
	v_cvt_f32_ubyte3_e32 v67, v67
	v_cvt_f32_ubyte3_e32 v77, v77
	v_fma_mix_f32 v40, v8, v40, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v9, v50, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v10, v60, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v70, v11, v70, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v41, v8, v41, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v9, v51, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v10, v61, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v71, v11, v71, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v42, v8, v42, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v52, v9, v52, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v62, v10, v62, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v72, v11, v72, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v43, v8, v43, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v53, v9, v53, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v63, v10, v63, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v73, v11, v73, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v44, v8, v44, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v54, v9, v54, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v64, v10, v64, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v74, v11, v74, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v8, v45, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v9, v55, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v65, v10, v65, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v75, v11, v75, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v8, v46, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v9, v56, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v66, v10, v66, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v76, v11, v76, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v8, v47, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v9, v57, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v67, v10, v67, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v77, v11, v77, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	buffer_load_b128 v[112:115], v3, s[24:27], s28 offen offset:1024
	buffer_load_b128 v[116:119], v3, s[24:27], s28 offen offset:1040
	buffer_load_b128 v[120:123], v3, s[24:27], s29 offen offset:1024
	buffer_load_b128 v[124:127], v3, s[24:27], s29 offen offset:1040
	buffer_load_b128 v[128:131], v3, s[24:27], s30 offen offset:1024
	buffer_load_b128 v[132:135], v3, s[24:27], s30 offen offset:1040
	buffer_load_b128 v[136:139], v3, s[24:27], s31 offen offset:1024
	buffer_load_b128 v[140:143], v3, s[24:27], s31 offen offset:1040
	buffer_load_b128 v[144:147], v3, s[24:27], s28 offen offset:2048
	buffer_load_b128 v[148:151], v3, s[24:27], s28 offen offset:2064
	buffer_load_b128 v[152:155], v3, s[24:27], s29 offen offset:2048
	buffer_load_b128 v[156:159], v3, s[24:27], s29 offen offset:2064
	buffer_load_b128 v[160:163], v3, s[24:27], s30 offen offset:2048
	buffer_load_b128 v[164:167], v3, s[24:27], s30 offen offset:2064
	buffer_load_b128 v[168:171], v3, s[24:27], s31 offen offset:2048
	buffer_load_b128 v[172:175], v3, s[24:27], s31 offen offset:2064
	buffer_load_b128 v[176:179], v3, s[24:27], s28 offen offset:3072
	buffer_load_b128 v[180:183], v3, s[24:27], s28 offen offset:3088
	buffer_load_b128 v[184:187], v3, s[24:27], s29 offen offset:3072
	buffer_load_b128 v[188:191], v3, s[24:27], s29 offen offset:3088
	s_wait_loadcnt 0x33
	v_dual_mul_f32 v48, v41, v81 :: v_dual_mul_f32 v49, v51, v81
	v_dual_mul_f32 v58, v61, v81 :: v_dual_mul_f32 v59, v71, v81
	s_wait_loadcnt 0x31
	v_dual_mul_f32 v68, v41, v89 :: v_dual_mul_f32 v69, v51, v89
	v_dual_mul_f32 v78, v61, v89 :: v_dual_mul_f32 v79, v71, v89
	v_dual_fmac_f32 v48, v40, v80 :: v_dual_fmac_f32 v49, v50, v80
	v_dual_fmac_f32 v58, v60, v80 :: v_dual_fmac_f32 v59, v70, v80
	v_dual_fmac_f32 v68, v40, v88 :: v_dual_fmac_f32 v69, v50, v88
	v_dual_fmac_f32 v78, v60, v88 :: v_dual_fmac_f32 v79, v70, v88
	v_dual_fmac_f32 v48, v42, v82 :: v_dual_fmac_f32 v49, v52, v82
	v_dual_fmac_f32 v58, v62, v82 :: v_dual_fmac_f32 v59, v72, v82
	v_dual_fmac_f32 v68, v42, v90 :: v_dual_fmac_f32 v69, v52, v90
	v_dual_fmac_f32 v78, v62, v90 :: v_dual_fmac_f32 v79, v72, v90
	v_dual_fmac_f32 v48, v43, v83 :: v_dual_fmac_f32 v49, v53, v83
	v_dual_fmac_f32 v58, v63, v83 :: v_dual_fmac_f32 v59, v73, v83
	v_dual_fmac_f32 v68, v43, v91 :: v_dual_fmac_f32 v69, v53, v91
	v_dual_fmac_f32 v78, v63, v91 :: v_dual_fmac_f32 v79, v73, v91
	v_dual_fmac_f32 v48, v44, v84 :: v_dual_fmac_f32 v49, v54, v84
	v_dual_fmac_f32 v58, v64, v84 :: v_dual_fmac_f32 v59, v74, v84
	s_wait_loadcnt 0x30
	v_dual_fmac_f32 v68, v44, v92 :: v_dual_fmac_f32 v69, v54, v92
	v_dual_fmac_f32 v78, v64, v92 :: v_dual_fmac_f32 v79, v74, v92
	v_dual_fmac_f32 v48, v45, v85 :: v_dual_fmac_f32 v49, v55, v85
	v_dual_fmac_f32 v58, v65, v85 :: v_dual_fmac_f32 v59, v75, v85
	v_dual_fmac_f32 v68, v45, v93 :: v_dual_fmac_f32 v69, v55, v93
	v_dual_fmac_f32 v78, v65, v93 :: v_dual_fmac_f32 v79, v75, v93
	v_dual_fmac_f32 v48, v46, v86 :: v_dual_fmac_f32 v49, v56, v86
	v_dual_fmac_f32 v58, v66, v86 :: v_dual_fmac_f32 v59, v76, v86
	v_dual_fmac_f32 v68, v46, v94 :: v_dual_fmac_f32 v69, v56, v94
	v_dual_fmac_f32 v78, v66, v94 :: v_dual_fmac_f32 v79, v76, v94
	v_dual_fmac_f32 v48, v47, v87 :: v_dual_fmac_f32 v49, v57, v87
	v_dual_fmac_f32 v58, v67, v87 :: v_dual_fmac_f32 v59, v77, v87
	v_dual_fmac_f32 v68, v47, v95 :: v_dual_fmac_f32 v69, v57, v95
	v_dual_fmac_f32 v78, v67, v95 :: v_dual_fmac_f32 v79, v77, v95
	v_dual_add_f32 v192, v192, v48 :: v_dual_add_f32 v193, v193, v49
	v_dual_add_f32 v194, v194, v58 :: v_dual_add_f32 v195, v195, v59
	v_dual_add_f32 v208, v208, v68 :: v_dual_add_f32 v209, v209, v69
	v_dual_add_f32 v210, v210, v78 :: v_dual_add_f32 v211, v211, v79
	buffer_load_b128 v[80:83], v3, s[24:27], s30 offen offset:3072
	buffer_load_b128 v[84:87], v3, s[24:27], s30 offen offset:3088
	buffer_load_b128 v[88:91], v3, s[24:27], s31 offen offset:3072
	buffer_load_b128 v[92:95], v3, s[24:27], s31 offen offset:3088
	s_wait_loadcnt 0x33
	v_dual_mul_f32 v48, v41, v97 :: v_dual_mul_f32 v49, v51, v97
	v_dual_mul_f32 v58, v61, v97 :: v_dual_mul_f32 v59, v71, v97
	s_wait_loadcnt 0x31
	v_dual_mul_f32 v68, v41, v105 :: v_dual_mul_f32 v69, v51, v105
	v_dual_mul_f32 v78, v61, v105 :: v_dual_mul_f32 v79, v71, v105
	v_dual_fmac_f32 v48, v40, v96 :: v_dual_fmac_f32 v49, v50, v96
	v_dual_fmac_f32 v58, v60, v96 :: v_dual_fmac_f32 v59, v70, v96
	v_dual_fmac_f32 v68, v40, v104 :: v_dual_fmac_f32 v69, v50, v104
	v_dual_fmac_f32 v78, v60, v104 :: v_dual_fmac_f32 v79, v70, v104
	v_dual_fmac_f32 v48, v42, v98 :: v_dual_fmac_f32 v49, v52, v98
	v_dual_fmac_f32 v58, v62, v98 :: v_dual_fmac_f32 v59, v72, v98
	v_dual_fmac_f32 v68, v42, v106 :: v_dual_fmac_f32 v69, v52, v106
	v_dual_fmac_f32 v78, v62, v106 :: v_dual_fmac_f32 v79, v72, v106
	v_dual_fmac_f32 v48, v43, v99 :: v_dual_fmac_f32 v49, v53, v99
	v_dual_fmac_f32 v58, v63, v99 :: v_dual_fmac_f32 v59, v73, v99
	v_dual_fmac_f32 v68, v43, v107 :: v_dual_fmac_f32 v69, v53, v107
	v_dual_fmac_f32 v78, v63, v107 :: v_dual_fmac_f32 v79, v73, v107
	v_dual_fmac_f32 v48, v44, v100 :: v_dual_fmac_f32 v49, v54, v100
	v_dual_fmac_f32 v58, v64, v100 :: v_dual_fmac_f32 v59, v74, v100
	s_wait_loadcnt 0x30
	v_dual_fmac_f32 v68, v44, v108 :: v_dual_fmac_f32 v69, v54, v108
	v_dual_fmac_f32 v78, v64, v108 :: v_dual_fmac_f32 v79, v74, v108
	v_dual_fmac_f32 v48, v45, v101 :: v_dual_fmac_f32 v49, v55, v101
	v_dual_fmac_f32 v58, v65, v101 :: v_dual_fmac_f32 v59, v75, v101
	v_dual_fmac_f32 v68, v45, v109 :: v_dual_fmac_f32 v69, v55, v109
	v_dual_fmac_f32 v78, v65, v109 :: v_dual_fmac_f32 v79, v75, v109
	v_dual_fmac_f32 v48, v46, v102 :: v_dual_fmac_f32 v49, v56, v102
	v_dual_fmac_f32 v58, v66, v102 :: v_dual_fmac_f32 v59, v76, v102
	v_dual_fmac_f32 v68, v46, v110 :: v_dual_fmac_f32 v69, v56, v110
	v_dual_fmac_f32 v78, v66, v110 :: v_dual_fmac_f32 v79, v76, v110
	v_dual_fmac_f32 v48, v47, v103 :: v_dual_fmac_f32 v49, v57, v103
	v_dual_fmac_f32 v58, v67, v103 :: v_dual_fmac_f32 v59, v77, v103
	v_dual_fmac_f32 v68, v47, v111 :: v_dual_fmac_f32 v69, v57, v111
	v_dual_fmac_f32 v78, v67, v111 :: v_dual_fmac_f32 v79, v77, v111
	v_dual_add_f32 v224, v224, v48 :: v_dual_add_f32 v225, v225, v49
	v_dual_add_f32 v226, v226, v58 :: v_dual_add_f32 v227, v227, v59
	v_dual_add_f32 v240, v240, v68 :: v_dual_add_f32 v241, v241, v69
	v_dual_add_f32 v242, v242, v78 :: v_dual_add_f32 v243, v243, v79
	s_wait_loadcnt 0x2b
	v_and_b32_e32 v46, 0xf0f0f0f, v28
	s_wait_loadcnt 0x2a
	v_and_b32_e32 v56, 0xf0f0f0f, v29
	s_wait_loadcnt 0x29
	v_and_b32_e32 v66, 0xf0f0f0f, v30
	s_wait_loadcnt 0x28
	v_and_b32_e32 v76, 0xf0f0f0f, v31
	v_lshrrev_b32_e32 v47, 4, v28
	v_lshrrev_b32_e32 v57, 4, v29
	v_lshrrev_b32_e32 v67, 4, v30
	v_lshrrev_b32_e32 v77, 4, v31
	v_and_b32_e32 v47, 0xf0f0f0f, v47
	v_and_b32_e32 v57, 0xf0f0f0f, v57
	v_and_b32_e32 v67, 0xf0f0f0f, v67
	v_and_b32_e32 v77, 0xf0f0f0f, v77
	v_cvt_f32_ubyte0_e32 v40, v46
	v_cvt_f32_ubyte0_e32 v50, v56
	v_cvt_f32_ubyte0_e32 v60, v66
	v_cvt_f32_ubyte0_e32 v70, v76
	v_cvt_f32_ubyte1_e32 v42, v46
	v_cvt_f32_ubyte1_e32 v52, v56
	v_cvt_f32_ubyte1_e32 v62, v66
	v_cvt_f32_ubyte1_e32 v72, v76
	v_cvt_f32_ubyte2_e32 v44, v46
	v_cvt_f32_ubyte2_e32 v54, v56
	v_cvt_f32_ubyte2_e32 v64, v66
	v_cvt_f32_ubyte2_e32 v74, v76
	v_cvt_f32_ubyte3_e32 v46, v46
	v_cvt_f32_ubyte3_e32 v56, v56
	v_cvt_f32_ubyte3_e32 v66, v66
	v_cvt_f32_ubyte3_e32 v76, v76
	v_cvt_f32_ubyte0_e32 v41, v47
	v_cvt_f32_ubyte0_e32 v51, v57
	v_cvt_f32_ubyte0_e32 v61, v67
	v_cvt_f32_ubyte0_e32 v71, v77
	v_cvt_f32_ubyte1_e32 v43, v47
	v_cvt_f32_ubyte1_e32 v53, v57
	v_cvt_f32_ubyte1_e32 v63, v67
	v_cvt_f32_ubyte1_e32 v73, v77
	v_cvt_f32_ubyte2_e32 v45, v47
	v_cvt_f32_ubyte2_e32 v55, v57
	v_cvt_f32_ubyte2_e32 v65, v67
	v_cvt_f32_ubyte2_e32 v75, v77
	v_cvt_f32_ubyte3_e32 v47, v47
	v_cvt_f32_ubyte3_e32 v57, v57
	v_cvt_f32_ubyte3_e32 v67, v67
	v_cvt_f32_ubyte3_e32 v77, v77
	v_fma_mix_f32 v40, v12, v40, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v13, v50, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v14, v60, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v70, v15, v70, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v41, v12, v41, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v13, v51, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v14, v61, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v71, v15, v71, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v42, v12, v42, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v52, v13, v52, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v62, v14, v62, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v72, v15, v72, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v43, v12, v43, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v53, v13, v53, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v63, v14, v63, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v73, v15, v73, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v44, v12, v44, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v54, v13, v54, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v64, v14, v64, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v74, v15, v74, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v12, v45, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v13, v55, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v65, v14, v65, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v75, v15, v75, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v12, v46, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v13, v56, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v66, v14, v66, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v76, v15, v76, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v12, v47, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v13, v57, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v67, v14, v67, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v77, v15, v77, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x17
	v_dual_mul_f32 v48, v41, v113 :: v_dual_mul_f32 v49, v51, v113
	v_dual_mul_f32 v58, v61, v113 :: v_dual_mul_f32 v59, v71, v113
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v68, v41, v121 :: v_dual_mul_f32 v69, v51, v121
	v_dual_mul_f32 v78, v61, v121 :: v_dual_mul_f32 v79, v71, v121
	v_dual_fmac_f32 v48, v40, v112 :: v_dual_fmac_f32 v49, v50, v112
	v_dual_fmac_f32 v58, v60, v112 :: v_dual_fmac_f32 v59, v70, v112
	v_dual_fmac_f32 v68, v40, v120 :: v_dual_fmac_f32 v69, v50, v120
	v_dual_fmac_f32 v78, v60, v120 :: v_dual_fmac_f32 v79, v70, v120
	v_dual_fmac_f32 v48, v42, v114 :: v_dual_fmac_f32 v49, v52, v114
	v_dual_fmac_f32 v58, v62, v114 :: v_dual_fmac_f32 v59, v72, v114
	v_dual_fmac_f32 v68, v42, v122 :: v_dual_fmac_f32 v69, v52, v122
	v_dual_fmac_f32 v78, v62, v122 :: v_dual_fmac_f32 v79, v72, v122
	v_dual_fmac_f32 v48, v43, v115 :: v_dual_fmac_f32 v49, v53, v115
	v_dual_fmac_f32 v58, v63, v115 :: v_dual_fmac_f32 v59, v73, v115
	v_dual_fmac_f32 v68, v43, v123 :: v_dual_fmac_f32 v69, v53, v123
	v_dual_fmac_f32 v78, v63, v123 :: v_dual_fmac_f32 v79, v73, v123
	v_dual_fmac_f32 v48, v44, v116 :: v_dual_fmac_f32 v49, v54, v116
	v_dual_fmac_f32 v58, v64, v116 :: v_dual_fmac_f32 v59, v74, v116
	s_wait_loadcnt 0x14
	v_dual_fmac_f32 v68, v44, v124 :: v_dual_fmac_f32 v69, v54, v124
	v_dual_fmac_f32 v78, v64, v124 :: v_dual_fmac_f32 v79, v74, v124
	v_dual_fmac_f32 v48, v45, v117 :: v_dual_fmac_f32 v49, v55, v117
	v_dual_fmac_f32 v58, v65, v117 :: v_dual_fmac_f32 v59, v75, v117
	v_dual_fmac_f32 v68, v45, v125 :: v_dual_fmac_f32 v69, v55, v125
	v_dual_fmac_f32 v78, v65, v125 :: v_dual_fmac_f32 v79, v75, v125
	v_dual_fmac_f32 v48, v46, v118 :: v_dual_fmac_f32 v49, v56, v118
	v_dual_fmac_f32 v58, v66, v118 :: v_dual_fmac_f32 v59, v76, v118
	v_dual_fmac_f32 v68, v46, v126 :: v_dual_fmac_f32 v69, v56, v126
	v_dual_fmac_f32 v78, v66, v126 :: v_dual_fmac_f32 v79, v76, v126
	v_dual_fmac_f32 v48, v47, v119 :: v_dual_fmac_f32 v49, v57, v119
	v_dual_fmac_f32 v58, v67, v119 :: v_dual_fmac_f32 v59, v77, v119
	v_dual_fmac_f32 v68, v47, v127 :: v_dual_fmac_f32 v69, v57, v127
	v_dual_fmac_f32 v78, v67, v127 :: v_dual_fmac_f32 v79, v77, v127
	v_dual_add_f32 v196, v196, v48 :: v_dual_add_f32 v197, v197, v49
	v_dual_add_f32 v198, v198, v58 :: v_dual_add_f32 v199, v199, v59
	v_dual_add_f32 v212, v212, v68 :: v_dual_add_f32 v213, v213, v69
	v_dual_add_f32 v214, v214, v78 :: v_dual_add_f32 v215, v215, v79
	s_wait_loadcnt 0x13
	v_dual_mul_f32 v48, v41, v129 :: v_dual_mul_f32 v49, v51, v129
	v_dual_mul_f32 v58, v61, v129 :: v_dual_mul_f32 v59, v71, v129
	s_wait_loadcnt 0x11
	v_dual_mul_f32 v68, v41, v137 :: v_dual_mul_f32 v69, v51, v137
	v_dual_mul_f32 v78, v61, v137 :: v_dual_mul_f32 v79, v71, v137
	v_dual_fmac_f32 v48, v40, v128 :: v_dual_fmac_f32 v49, v50, v128
	v_dual_fmac_f32 v58, v60, v128 :: v_dual_fmac_f32 v59, v70, v128
	v_dual_fmac_f32 v68, v40, v136 :: v_dual_fmac_f32 v69, v50, v136
	v_dual_fmac_f32 v78, v60, v136 :: v_dual_fmac_f32 v79, v70, v136
	v_dual_fmac_f32 v48, v42, v130 :: v_dual_fmac_f32 v49, v52, v130
	v_dual_fmac_f32 v58, v62, v130 :: v_dual_fmac_f32 v59, v72, v130
	v_dual_fmac_f32 v68, v42, v138 :: v_dual_fmac_f32 v69, v52, v138
	v_dual_fmac_f32 v78, v62, v138 :: v_dual_fmac_f32 v79, v72, v138
	v_dual_fmac_f32 v48, v43, v131 :: v_dual_fmac_f32 v49, v53, v131
	v_dual_fmac_f32 v58, v63, v131 :: v_dual_fmac_f32 v59, v73, v131
	v_dual_fmac_f32 v68, v43, v139 :: v_dual_fmac_f32 v69, v53, v139
	v_dual_fmac_f32 v78, v63, v139 :: v_dual_fmac_f32 v79, v73, v139
	v_dual_fmac_f32 v48, v44, v132 :: v_dual_fmac_f32 v49, v54, v132
	v_dual_fmac_f32 v58, v64, v132 :: v_dual_fmac_f32 v59, v74, v132
	s_wait_loadcnt 0x10
	v_dual_fmac_f32 v68, v44, v140 :: v_dual_fmac_f32 v69, v54, v140
	v_dual_fmac_f32 v78, v64, v140 :: v_dual_fmac_f32 v79, v74, v140
	v_dual_fmac_f32 v48, v45, v133 :: v_dual_fmac_f32 v49, v55, v133
	v_dual_fmac_f32 v58, v65, v133 :: v_dual_fmac_f32 v59, v75, v133
	v_dual_fmac_f32 v68, v45, v141 :: v_dual_fmac_f32 v69, v55, v141
	v_dual_fmac_f32 v78, v65, v141 :: v_dual_fmac_f32 v79, v75, v141
	v_dual_fmac_f32 v48, v46, v134 :: v_dual_fmac_f32 v49, v56, v134
	v_dual_fmac_f32 v58, v66, v134 :: v_dual_fmac_f32 v59, v76, v134
	v_dual_fmac_f32 v68, v46, v142 :: v_dual_fmac_f32 v69, v56, v142
	v_dual_fmac_f32 v78, v66, v142 :: v_dual_fmac_f32 v79, v76, v142
	v_dual_fmac_f32 v48, v47, v135 :: v_dual_fmac_f32 v49, v57, v135
	v_dual_fmac_f32 v58, v67, v135 :: v_dual_fmac_f32 v59, v77, v135
	v_dual_fmac_f32 v68, v47, v143 :: v_dual_fmac_f32 v69, v57, v143
	v_dual_fmac_f32 v78, v67, v143 :: v_dual_fmac_f32 v79, v77, v143
	v_dual_add_f32 v228, v228, v48 :: v_dual_add_f32 v229, v229, v49
	v_dual_add_f32 v230, v230, v58 :: v_dual_add_f32 v231, v231, v59
	v_dual_add_f32 v244, v244, v68 :: v_dual_add_f32 v245, v245, v69
	v_dual_add_f32 v246, v246, v78 :: v_dual_add_f32 v247, v247, v79
	v_and_b32_e32 v46, 0xf0f0f0f, v32
	v_and_b32_e32 v56, 0xf0f0f0f, v33
	v_and_b32_e32 v66, 0xf0f0f0f, v34
	v_and_b32_e32 v76, 0xf0f0f0f, v35
	v_lshrrev_b32_e32 v47, 4, v32
	v_lshrrev_b32_e32 v57, 4, v33
	v_lshrrev_b32_e32 v67, 4, v34
	v_lshrrev_b32_e32 v77, 4, v35
	v_and_b32_e32 v47, 0xf0f0f0f, v47
	v_and_b32_e32 v57, 0xf0f0f0f, v57
	v_and_b32_e32 v67, 0xf0f0f0f, v67
	v_and_b32_e32 v77, 0xf0f0f0f, v77
	v_cvt_f32_ubyte0_e32 v40, v46
	v_cvt_f32_ubyte0_e32 v50, v56
	v_cvt_f32_ubyte0_e32 v60, v66
	v_cvt_f32_ubyte0_e32 v70, v76
	v_cvt_f32_ubyte1_e32 v42, v46
	v_cvt_f32_ubyte1_e32 v52, v56
	v_cvt_f32_ubyte1_e32 v62, v66
	v_cvt_f32_ubyte1_e32 v72, v76
	v_cvt_f32_ubyte2_e32 v44, v46
	v_cvt_f32_ubyte2_e32 v54, v56
	v_cvt_f32_ubyte2_e32 v64, v66
	v_cvt_f32_ubyte2_e32 v74, v76
	v_cvt_f32_ubyte3_e32 v46, v46
	v_cvt_f32_ubyte3_e32 v56, v56
	v_cvt_f32_ubyte3_e32 v66, v66
	v_cvt_f32_ubyte3_e32 v76, v76
	v_cvt_f32_ubyte0_e32 v41, v47
	v_cvt_f32_ubyte0_e32 v51, v57
	v_cvt_f32_ubyte0_e32 v61, v67
	v_cvt_f32_ubyte0_e32 v71, v77
	v_cvt_f32_ubyte1_e32 v43, v47
	v_cvt_f32_ubyte1_e32 v53, v57
	v_cvt_f32_ubyte1_e32 v63, v67
	v_cvt_f32_ubyte1_e32 v73, v77
	v_cvt_f32_ubyte2_e32 v45, v47
	v_cvt_f32_ubyte2_e32 v55, v57
	v_cvt_f32_ubyte2_e32 v65, v67
	v_cvt_f32_ubyte2_e32 v75, v77
	v_cvt_f32_ubyte3_e32 v47, v47
	v_cvt_f32_ubyte3_e32 v57, v57
	v_cvt_f32_ubyte3_e32 v67, v67
	v_cvt_f32_ubyte3_e32 v77, v77
	v_fma_mix_f32 v40, v16, v40, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v17, v50, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v18, v60, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v70, v19, v70, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v41, v16, v41, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v17, v51, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v18, v61, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v71, v19, v71, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v42, v16, v42, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v52, v17, v52, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v62, v18, v62, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v72, v19, v72, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v43, v16, v43, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v53, v17, v53, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v63, v18, v63, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v73, v19, v73, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v44, v16, v44, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v54, v17, v54, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v64, v18, v64, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v74, v19, v74, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v16, v45, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v17, v55, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v65, v18, v65, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v75, v19, v75, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v16, v46, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v17, v56, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v66, v18, v66, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v76, v19, v76, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v16, v47, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v17, v57, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v67, v18, v67, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v77, v19, v77, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xf
	v_dual_mul_f32 v48, v41, v145 :: v_dual_mul_f32 v49, v51, v145
	v_dual_mul_f32 v58, v61, v145 :: v_dual_mul_f32 v59, v71, v145
	s_wait_loadcnt 0xd
	v_dual_mul_f32 v68, v41, v153 :: v_dual_mul_f32 v69, v51, v153
	v_dual_mul_f32 v78, v61, v153 :: v_dual_mul_f32 v79, v71, v153
	v_dual_fmac_f32 v48, v40, v144 :: v_dual_fmac_f32 v49, v50, v144
	v_dual_fmac_f32 v58, v60, v144 :: v_dual_fmac_f32 v59, v70, v144
	v_dual_fmac_f32 v68, v40, v152 :: v_dual_fmac_f32 v69, v50, v152
	v_dual_fmac_f32 v78, v60, v152 :: v_dual_fmac_f32 v79, v70, v152
	v_dual_fmac_f32 v48, v42, v146 :: v_dual_fmac_f32 v49, v52, v146
	v_dual_fmac_f32 v58, v62, v146 :: v_dual_fmac_f32 v59, v72, v146
	v_dual_fmac_f32 v68, v42, v154 :: v_dual_fmac_f32 v69, v52, v154
	v_dual_fmac_f32 v78, v62, v154 :: v_dual_fmac_f32 v79, v72, v154
	v_dual_fmac_f32 v48, v43, v147 :: v_dual_fmac_f32 v49, v53, v147
	v_dual_fmac_f32 v58, v63, v147 :: v_dual_fmac_f32 v59, v73, v147
	v_dual_fmac_f32 v68, v43, v155 :: v_dual_fmac_f32 v69, v53, v155
	v_dual_fmac_f32 v78, v63, v155 :: v_dual_fmac_f32 v79, v73, v155
	v_dual_fmac_f32 v48, v44, v148 :: v_dual_fmac_f32 v49, v54, v148
	v_dual_fmac_f32 v58, v64, v148 :: v_dual_fmac_f32 v59, v74, v148
	s_wait_loadcnt 0xc
	v_dual_fmac_f32 v68, v44, v156 :: v_dual_fmac_f32 v69, v54, v156
	v_dual_fmac_f32 v78, v64, v156 :: v_dual_fmac_f32 v79, v74, v156
	v_dual_fmac_f32 v48, v45, v149 :: v_dual_fmac_f32 v49, v55, v149
	v_dual_fmac_f32 v58, v65, v149 :: v_dual_fmac_f32 v59, v75, v149
	v_dual_fmac_f32 v68, v45, v157 :: v_dual_fmac_f32 v69, v55, v157
	v_dual_fmac_f32 v78, v65, v157 :: v_dual_fmac_f32 v79, v75, v157
	v_dual_fmac_f32 v48, v46, v150 :: v_dual_fmac_f32 v49, v56, v150
	v_dual_fmac_f32 v58, v66, v150 :: v_dual_fmac_f32 v59, v76, v150
	v_dual_fmac_f32 v68, v46, v158 :: v_dual_fmac_f32 v69, v56, v158
	v_dual_fmac_f32 v78, v66, v158 :: v_dual_fmac_f32 v79, v76, v158
	v_dual_fmac_f32 v48, v47, v151 :: v_dual_fmac_f32 v49, v57, v151
	v_dual_fmac_f32 v58, v67, v151 :: v_dual_fmac_f32 v59, v77, v151
	v_dual_fmac_f32 v68, v47, v159 :: v_dual_fmac_f32 v69, v57, v159
	v_dual_fmac_f32 v78, v67, v159 :: v_dual_fmac_f32 v79, v77, v159
	v_dual_add_f32 v200, v200, v48 :: v_dual_add_f32 v201, v201, v49
	v_dual_add_f32 v202, v202, v58 :: v_dual_add_f32 v203, v203, v59
	v_dual_add_f32 v216, v216, v68 :: v_dual_add_f32 v217, v217, v69
	v_dual_add_f32 v218, v218, v78 :: v_dual_add_f32 v219, v219, v79
	s_wait_loadcnt 0xb
	v_dual_mul_f32 v48, v41, v161 :: v_dual_mul_f32 v49, v51, v161
	v_dual_mul_f32 v58, v61, v161 :: v_dual_mul_f32 v59, v71, v161
	s_wait_loadcnt 0x9
	v_dual_mul_f32 v68, v41, v169 :: v_dual_mul_f32 v69, v51, v169
	v_dual_mul_f32 v78, v61, v169 :: v_dual_mul_f32 v79, v71, v169
	v_dual_fmac_f32 v48, v40, v160 :: v_dual_fmac_f32 v49, v50, v160
	v_dual_fmac_f32 v58, v60, v160 :: v_dual_fmac_f32 v59, v70, v160
	v_dual_fmac_f32 v68, v40, v168 :: v_dual_fmac_f32 v69, v50, v168
	v_dual_fmac_f32 v78, v60, v168 :: v_dual_fmac_f32 v79, v70, v168
	v_dual_fmac_f32 v48, v42, v162 :: v_dual_fmac_f32 v49, v52, v162
	v_dual_fmac_f32 v58, v62, v162 :: v_dual_fmac_f32 v59, v72, v162
	v_dual_fmac_f32 v68, v42, v170 :: v_dual_fmac_f32 v69, v52, v170
	v_dual_fmac_f32 v78, v62, v170 :: v_dual_fmac_f32 v79, v72, v170
	v_dual_fmac_f32 v48, v43, v163 :: v_dual_fmac_f32 v49, v53, v163
	v_dual_fmac_f32 v58, v63, v163 :: v_dual_fmac_f32 v59, v73, v163
	v_dual_fmac_f32 v68, v43, v171 :: v_dual_fmac_f32 v69, v53, v171
	v_dual_fmac_f32 v78, v63, v171 :: v_dual_fmac_f32 v79, v73, v171
	v_dual_fmac_f32 v48, v44, v164 :: v_dual_fmac_f32 v49, v54, v164
	v_dual_fmac_f32 v58, v64, v164 :: v_dual_fmac_f32 v59, v74, v164
	s_wait_loadcnt 0x8
	v_dual_fmac_f32 v68, v44, v172 :: v_dual_fmac_f32 v69, v54, v172
	v_dual_fmac_f32 v78, v64, v172 :: v_dual_fmac_f32 v79, v74, v172
	v_dual_fmac_f32 v48, v45, v165 :: v_dual_fmac_f32 v49, v55, v165
	v_dual_fmac_f32 v58, v65, v165 :: v_dual_fmac_f32 v59, v75, v165
	v_dual_fmac_f32 v68, v45, v173 :: v_dual_fmac_f32 v69, v55, v173
	v_dual_fmac_f32 v78, v65, v173 :: v_dual_fmac_f32 v79, v75, v173
	v_dual_fmac_f32 v48, v46, v166 :: v_dual_fmac_f32 v49, v56, v166
	v_dual_fmac_f32 v58, v66, v166 :: v_dual_fmac_f32 v59, v76, v166
	v_dual_fmac_f32 v68, v46, v174 :: v_dual_fmac_f32 v69, v56, v174
	v_dual_fmac_f32 v78, v66, v174 :: v_dual_fmac_f32 v79, v76, v174
	v_dual_fmac_f32 v48, v47, v167 :: v_dual_fmac_f32 v49, v57, v167
	v_dual_fmac_f32 v58, v67, v167 :: v_dual_fmac_f32 v59, v77, v167
	v_dual_fmac_f32 v68, v47, v175 :: v_dual_fmac_f32 v69, v57, v175
	v_dual_fmac_f32 v78, v67, v175 :: v_dual_fmac_f32 v79, v77, v175
	v_dual_add_f32 v232, v232, v48 :: v_dual_add_f32 v233, v233, v49
	v_dual_add_f32 v234, v234, v58 :: v_dual_add_f32 v235, v235, v59
	v_dual_add_f32 v248, v248, v68 :: v_dual_add_f32 v249, v249, v69
	v_dual_add_f32 v250, v250, v78 :: v_dual_add_f32 v251, v251, v79
	v_and_b32_e32 v46, 0xf0f0f0f, v36
	v_and_b32_e32 v56, 0xf0f0f0f, v37
	v_and_b32_e32 v66, 0xf0f0f0f, v38
	v_and_b32_e32 v76, 0xf0f0f0f, v39
	v_lshrrev_b32_e32 v47, 4, v36
	v_lshrrev_b32_e32 v57, 4, v37
	v_lshrrev_b32_e32 v67, 4, v38
	v_lshrrev_b32_e32 v77, 4, v39
	v_and_b32_e32 v47, 0xf0f0f0f, v47
	v_and_b32_e32 v57, 0xf0f0f0f, v57
	v_and_b32_e32 v67, 0xf0f0f0f, v67
	v_and_b32_e32 v77, 0xf0f0f0f, v77
	v_cvt_f32_ubyte0_e32 v40, v46
	v_cvt_f32_ubyte0_e32 v50, v56
	v_cvt_f32_ubyte0_e32 v60, v66
	v_cvt_f32_ubyte0_e32 v70, v76
	v_cvt_f32_ubyte1_e32 v42, v46
	v_cvt_f32_ubyte1_e32 v52, v56
	v_cvt_f32_ubyte1_e32 v62, v66
	v_cvt_f32_ubyte1_e32 v72, v76
	v_cvt_f32_ubyte2_e32 v44, v46
	v_cvt_f32_ubyte2_e32 v54, v56
	v_cvt_f32_ubyte2_e32 v64, v66
	v_cvt_f32_ubyte2_e32 v74, v76
	v_cvt_f32_ubyte3_e32 v46, v46
	v_cvt_f32_ubyte3_e32 v56, v56
	v_cvt_f32_ubyte3_e32 v66, v66
	v_cvt_f32_ubyte3_e32 v76, v76
	v_cvt_f32_ubyte0_e32 v41, v47
	v_cvt_f32_ubyte0_e32 v51, v57
	v_cvt_f32_ubyte0_e32 v61, v67
	v_cvt_f32_ubyte0_e32 v71, v77
	v_cvt_f32_ubyte1_e32 v43, v47
	v_cvt_f32_ubyte1_e32 v53, v57
	v_cvt_f32_ubyte1_e32 v63, v67
	v_cvt_f32_ubyte1_e32 v73, v77
	v_cvt_f32_ubyte2_e32 v45, v47
	v_cvt_f32_ubyte2_e32 v55, v57
	v_cvt_f32_ubyte2_e32 v65, v67
	v_cvt_f32_ubyte2_e32 v75, v77
	v_cvt_f32_ubyte3_e32 v47, v47
	v_cvt_f32_ubyte3_e32 v57, v57
	v_cvt_f32_ubyte3_e32 v67, v67
	v_cvt_f32_ubyte3_e32 v77, v77
	v_fma_mix_f32 v40, v20, v40, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v21, v50, v21 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v22, v60, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v70, v23, v70, v23 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v41, v20, v41, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v21, v51, v21 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v22, v61, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v71, v23, v71, v23 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v42, v20, v42, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v52, v21, v52, v21 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v62, v22, v62, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v72, v23, v72, v23 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v43, v20, v43, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v53, v21, v53, v21 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v63, v22, v63, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v73, v23, v73, v23 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v44, v20, v44, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v54, v21, v54, v21 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v64, v22, v64, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v74, v23, v74, v23 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v20, v45, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v21, v55, v21 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v65, v22, v65, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v75, v23, v75, v23 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v20, v46, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v21, v56, v21 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v66, v22, v66, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v76, v23, v76, v23 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v20, v47, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v21, v57, v21 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v67, v22, v67, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v77, v23, v77, v23 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x7
	v_dual_mul_f32 v48, v41, v177 :: v_dual_mul_f32 v49, v51, v177
	v_dual_mul_f32 v58, v61, v177 :: v_dual_mul_f32 v59, v71, v177
	s_wait_loadcnt 0x5
	v_dual_mul_f32 v68, v41, v185 :: v_dual_mul_f32 v69, v51, v185
	v_dual_mul_f32 v78, v61, v185 :: v_dual_mul_f32 v79, v71, v185
	v_dual_fmac_f32 v48, v40, v176 :: v_dual_fmac_f32 v49, v50, v176
	v_dual_fmac_f32 v58, v60, v176 :: v_dual_fmac_f32 v59, v70, v176
	v_dual_fmac_f32 v68, v40, v184 :: v_dual_fmac_f32 v69, v50, v184
	v_dual_fmac_f32 v78, v60, v184 :: v_dual_fmac_f32 v79, v70, v184
	v_dual_fmac_f32 v48, v42, v178 :: v_dual_fmac_f32 v49, v52, v178
	v_dual_fmac_f32 v58, v62, v178 :: v_dual_fmac_f32 v59, v72, v178
	v_dual_fmac_f32 v68, v42, v186 :: v_dual_fmac_f32 v69, v52, v186
	v_dual_fmac_f32 v78, v62, v186 :: v_dual_fmac_f32 v79, v72, v186
	v_dual_fmac_f32 v48, v43, v179 :: v_dual_fmac_f32 v49, v53, v179
	v_dual_fmac_f32 v58, v63, v179 :: v_dual_fmac_f32 v59, v73, v179
	v_dual_fmac_f32 v68, v43, v187 :: v_dual_fmac_f32 v69, v53, v187
	v_dual_fmac_f32 v78, v63, v187 :: v_dual_fmac_f32 v79, v73, v187
	v_dual_fmac_f32 v48, v44, v180 :: v_dual_fmac_f32 v49, v54, v180
	v_dual_fmac_f32 v58, v64, v180 :: v_dual_fmac_f32 v59, v74, v180
	s_wait_loadcnt 0x4
	v_dual_fmac_f32 v68, v44, v188 :: v_dual_fmac_f32 v69, v54, v188
	v_dual_fmac_f32 v78, v64, v188 :: v_dual_fmac_f32 v79, v74, v188
	v_dual_fmac_f32 v48, v45, v181 :: v_dual_fmac_f32 v49, v55, v181
	v_dual_fmac_f32 v58, v65, v181 :: v_dual_fmac_f32 v59, v75, v181
	v_dual_fmac_f32 v68, v45, v189 :: v_dual_fmac_f32 v69, v55, v189
	v_dual_fmac_f32 v78, v65, v189 :: v_dual_fmac_f32 v79, v75, v189
	v_dual_fmac_f32 v48, v46, v182 :: v_dual_fmac_f32 v49, v56, v182
	v_dual_fmac_f32 v58, v66, v182 :: v_dual_fmac_f32 v59, v76, v182
	v_dual_fmac_f32 v68, v46, v190 :: v_dual_fmac_f32 v69, v56, v190
	v_dual_fmac_f32 v78, v66, v190 :: v_dual_fmac_f32 v79, v76, v190
	v_dual_fmac_f32 v48, v47, v183 :: v_dual_fmac_f32 v49, v57, v183
	v_dual_fmac_f32 v58, v67, v183 :: v_dual_fmac_f32 v59, v77, v183
	v_dual_fmac_f32 v68, v47, v191 :: v_dual_fmac_f32 v69, v57, v191
	v_dual_fmac_f32 v78, v67, v191 :: v_dual_fmac_f32 v79, v77, v191
	v_dual_add_f32 v204, v204, v48 :: v_dual_add_f32 v205, v205, v49
	v_dual_add_f32 v206, v206, v58 :: v_dual_add_f32 v207, v207, v59
	v_dual_add_f32 v220, v220, v68 :: v_dual_add_f32 v221, v221, v69
	v_dual_add_f32 v222, v222, v78 :: v_dual_add_f32 v223, v223, v79
	s_wait_loadcnt 0x3
	v_dual_mul_f32 v48, v41, v81 :: v_dual_mul_f32 v49, v51, v81
	v_dual_mul_f32 v58, v61, v81 :: v_dual_mul_f32 v59, v71, v81
	s_wait_loadcnt 0x1
	v_dual_mul_f32 v68, v41, v89 :: v_dual_mul_f32 v69, v51, v89
	v_dual_mul_f32 v78, v61, v89 :: v_dual_mul_f32 v79, v71, v89
	v_dual_fmac_f32 v48, v40, v80 :: v_dual_fmac_f32 v49, v50, v80
	v_dual_fmac_f32 v58, v60, v80 :: v_dual_fmac_f32 v59, v70, v80
	v_dual_fmac_f32 v68, v40, v88 :: v_dual_fmac_f32 v69, v50, v88
	v_dual_fmac_f32 v78, v60, v88 :: v_dual_fmac_f32 v79, v70, v88
	v_dual_fmac_f32 v48, v42, v82 :: v_dual_fmac_f32 v49, v52, v82
	v_dual_fmac_f32 v58, v62, v82 :: v_dual_fmac_f32 v59, v72, v82
	v_dual_fmac_f32 v68, v42, v90 :: v_dual_fmac_f32 v69, v52, v90
	v_dual_fmac_f32 v78, v62, v90 :: v_dual_fmac_f32 v79, v72, v90
	v_dual_fmac_f32 v48, v43, v83 :: v_dual_fmac_f32 v49, v53, v83
	v_dual_fmac_f32 v58, v63, v83 :: v_dual_fmac_f32 v59, v73, v83
	v_dual_fmac_f32 v68, v43, v91 :: v_dual_fmac_f32 v69, v53, v91
	v_dual_fmac_f32 v78, v63, v91 :: v_dual_fmac_f32 v79, v73, v91
	v_dual_fmac_f32 v48, v44, v84 :: v_dual_fmac_f32 v49, v54, v84
	v_dual_fmac_f32 v58, v64, v84 :: v_dual_fmac_f32 v59, v74, v84
	s_wait_loadcnt 0x0
	v_dual_fmac_f32 v68, v44, v92 :: v_dual_fmac_f32 v69, v54, v92
	v_dual_fmac_f32 v78, v64, v92 :: v_dual_fmac_f32 v79, v74, v92
	v_dual_fmac_f32 v48, v45, v85 :: v_dual_fmac_f32 v49, v55, v85
	v_dual_fmac_f32 v58, v65, v85 :: v_dual_fmac_f32 v59, v75, v85
	v_dual_fmac_f32 v68, v45, v93 :: v_dual_fmac_f32 v69, v55, v93
	v_dual_fmac_f32 v78, v65, v93 :: v_dual_fmac_f32 v79, v75, v93
	v_dual_fmac_f32 v48, v46, v86 :: v_dual_fmac_f32 v49, v56, v86
	v_dual_fmac_f32 v58, v66, v86 :: v_dual_fmac_f32 v59, v76, v86
	v_dual_fmac_f32 v68, v46, v94 :: v_dual_fmac_f32 v69, v56, v94
	v_dual_fmac_f32 v78, v66, v94 :: v_dual_fmac_f32 v79, v76, v94
	v_dual_fmac_f32 v48, v47, v87 :: v_dual_fmac_f32 v49, v57, v87
	v_dual_fmac_f32 v58, v67, v87 :: v_dual_fmac_f32 v59, v77, v87
	v_dual_fmac_f32 v68, v47, v95 :: v_dual_fmac_f32 v69, v57, v95
	v_dual_fmac_f32 v78, v67, v95 :: v_dual_fmac_f32 v79, v77, v95
	v_dual_add_f32 v236, v236, v48 :: v_dual_add_f32 v237, v237, v49
	v_dual_add_f32 v238, v238, v58 :: v_dual_add_f32 v239, v239, v59
	v_dual_add_f32 v252, v252, v68 :: v_dual_add_f32 v253, v253, v69
	v_dual_add_f32 v254, v254, v78 :: v_dual_add_f32 v255, v255, v79
	s_add_co_i32 s16, s16, 0x220
	s_add_co_i32 s17, s17, 0x220
	s_add_co_i32 s18, s18, 0x220
	s_add_co_i32 s19, s19, 0x220
	s_add_co_i32 s28, s28, 0x1000
	s_add_co_i32 s29, s29, 0x1000
	s_add_co_i32 s30, s30, 0x1000
	s_add_co_i32 s31, s31, 0x1000
	s_add_co_i32 s36, s36, -1
	s_cmp_lg_u32 s36, 0
	s_cbranch_scc1 .Lpx_b4_quad
	.Lpx_b4_tail:
	s_cmp_gt_u32 s37, 0
	s_cbranch_scc0 .Lpx_b4_fold
	buffer_load_b32 v8, v1, s[20:23], s16 offen
	buffer_load_b32 v9, v1, s[20:23], s17 offen
	buffer_load_b32 v10, v1, s[20:23], s18 offen
	buffer_load_b32 v11, v1, s[20:23], s19 offen
	buffer_load_b32 v24, v2, s[20:23], s16 offen offset:8
	buffer_load_b32 v25, v2, s[20:23], s17 offen offset:8
	buffer_load_b32 v26, v2, s[20:23], s18 offen offset:8
	buffer_load_b32 v27, v2, s[20:23], s19 offen offset:8
	buffer_load_b128 v[80:83], v3, s[24:27], s28 offen
	buffer_load_b128 v[84:87], v3, s[24:27], s28 offen offset:16
	buffer_load_b128 v[88:91], v3, s[24:27], s29 offen
	buffer_load_b128 v[92:95], v3, s[24:27], s29 offen offset:16
	buffer_load_b128 v[96:99], v3, s[24:27], s30 offen
	buffer_load_b128 v[100:103], v3, s[24:27], s30 offen offset:16
	buffer_load_b128 v[104:107], v3, s[24:27], s31 offen
	buffer_load_b128 v[108:111], v3, s[24:27], s31 offen offset:16
	s_wait_loadcnt 0xb
	v_and_b32_e32 v46, 0xf0f0f0f, v24
	s_wait_loadcnt 0xa
	v_and_b32_e32 v56, 0xf0f0f0f, v25
	s_wait_loadcnt 0x9
	v_and_b32_e32 v66, 0xf0f0f0f, v26
	s_wait_loadcnt 0x8
	v_and_b32_e32 v76, 0xf0f0f0f, v27
	v_lshrrev_b32_e32 v47, 4, v24
	v_lshrrev_b32_e32 v57, 4, v25
	v_lshrrev_b32_e32 v67, 4, v26
	v_lshrrev_b32_e32 v77, 4, v27
	v_and_b32_e32 v47, 0xf0f0f0f, v47
	v_and_b32_e32 v57, 0xf0f0f0f, v57
	v_and_b32_e32 v67, 0xf0f0f0f, v67
	v_and_b32_e32 v77, 0xf0f0f0f, v77
	v_cvt_f32_ubyte0_e32 v40, v46
	v_cvt_f32_ubyte0_e32 v50, v56
	v_cvt_f32_ubyte0_e32 v60, v66
	v_cvt_f32_ubyte0_e32 v70, v76
	v_cvt_f32_ubyte1_e32 v42, v46
	v_cvt_f32_ubyte1_e32 v52, v56
	v_cvt_f32_ubyte1_e32 v62, v66
	v_cvt_f32_ubyte1_e32 v72, v76
	v_cvt_f32_ubyte2_e32 v44, v46
	v_cvt_f32_ubyte2_e32 v54, v56
	v_cvt_f32_ubyte2_e32 v64, v66
	v_cvt_f32_ubyte2_e32 v74, v76
	v_cvt_f32_ubyte3_e32 v46, v46
	v_cvt_f32_ubyte3_e32 v56, v56
	v_cvt_f32_ubyte3_e32 v66, v66
	v_cvt_f32_ubyte3_e32 v76, v76
	v_cvt_f32_ubyte0_e32 v41, v47
	v_cvt_f32_ubyte0_e32 v51, v57
	v_cvt_f32_ubyte0_e32 v61, v67
	v_cvt_f32_ubyte0_e32 v71, v77
	v_cvt_f32_ubyte1_e32 v43, v47
	v_cvt_f32_ubyte1_e32 v53, v57
	v_cvt_f32_ubyte1_e32 v63, v67
	v_cvt_f32_ubyte1_e32 v73, v77
	v_cvt_f32_ubyte2_e32 v45, v47
	v_cvt_f32_ubyte2_e32 v55, v57
	v_cvt_f32_ubyte2_e32 v65, v67
	v_cvt_f32_ubyte2_e32 v75, v77
	v_cvt_f32_ubyte3_e32 v47, v47
	v_cvt_f32_ubyte3_e32 v57, v57
	v_cvt_f32_ubyte3_e32 v67, v67
	v_cvt_f32_ubyte3_e32 v77, v77
	v_fma_mix_f32 v40, v8, v40, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v9, v50, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v10, v60, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v70, v11, v70, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v41, v8, v41, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v9, v51, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v10, v61, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v71, v11, v71, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v42, v8, v42, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v52, v9, v52, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v62, v10, v62, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v72, v11, v72, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v43, v8, v43, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v53, v9, v53, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v63, v10, v63, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v73, v11, v73, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v44, v8, v44, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v54, v9, v54, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v64, v10, v64, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v74, v11, v74, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v8, v45, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v9, v55, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v65, v10, v65, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v75, v11, v75, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v8, v46, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v9, v56, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v66, v10, v66, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v76, v11, v76, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v8, v47, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v9, v57, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v67, v10, v67, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v77, v11, v77, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x7
	v_dual_mul_f32 v48, v41, v81 :: v_dual_mul_f32 v49, v51, v81
	v_dual_mul_f32 v58, v61, v81 :: v_dual_mul_f32 v59, v71, v81
	s_wait_loadcnt 0x5
	v_dual_mul_f32 v68, v41, v89 :: v_dual_mul_f32 v69, v51, v89
	v_dual_mul_f32 v78, v61, v89 :: v_dual_mul_f32 v79, v71, v89
	v_dual_fmac_f32 v48, v40, v80 :: v_dual_fmac_f32 v49, v50, v80
	v_dual_fmac_f32 v58, v60, v80 :: v_dual_fmac_f32 v59, v70, v80
	v_dual_fmac_f32 v68, v40, v88 :: v_dual_fmac_f32 v69, v50, v88
	v_dual_fmac_f32 v78, v60, v88 :: v_dual_fmac_f32 v79, v70, v88
	v_dual_fmac_f32 v48, v42, v82 :: v_dual_fmac_f32 v49, v52, v82
	v_dual_fmac_f32 v58, v62, v82 :: v_dual_fmac_f32 v59, v72, v82
	v_dual_fmac_f32 v68, v42, v90 :: v_dual_fmac_f32 v69, v52, v90
	v_dual_fmac_f32 v78, v62, v90 :: v_dual_fmac_f32 v79, v72, v90
	v_dual_fmac_f32 v48, v43, v83 :: v_dual_fmac_f32 v49, v53, v83
	v_dual_fmac_f32 v58, v63, v83 :: v_dual_fmac_f32 v59, v73, v83
	v_dual_fmac_f32 v68, v43, v91 :: v_dual_fmac_f32 v69, v53, v91
	v_dual_fmac_f32 v78, v63, v91 :: v_dual_fmac_f32 v79, v73, v91
	v_dual_fmac_f32 v48, v44, v84 :: v_dual_fmac_f32 v49, v54, v84
	v_dual_fmac_f32 v58, v64, v84 :: v_dual_fmac_f32 v59, v74, v84
	s_wait_loadcnt 0x4
	v_dual_fmac_f32 v68, v44, v92 :: v_dual_fmac_f32 v69, v54, v92
	v_dual_fmac_f32 v78, v64, v92 :: v_dual_fmac_f32 v79, v74, v92
	v_dual_fmac_f32 v48, v45, v85 :: v_dual_fmac_f32 v49, v55, v85
	v_dual_fmac_f32 v58, v65, v85 :: v_dual_fmac_f32 v59, v75, v85
	v_dual_fmac_f32 v68, v45, v93 :: v_dual_fmac_f32 v69, v55, v93
	v_dual_fmac_f32 v78, v65, v93 :: v_dual_fmac_f32 v79, v75, v93
	v_dual_fmac_f32 v48, v46, v86 :: v_dual_fmac_f32 v49, v56, v86
	v_dual_fmac_f32 v58, v66, v86 :: v_dual_fmac_f32 v59, v76, v86
	v_dual_fmac_f32 v68, v46, v94 :: v_dual_fmac_f32 v69, v56, v94
	v_dual_fmac_f32 v78, v66, v94 :: v_dual_fmac_f32 v79, v76, v94
	v_dual_fmac_f32 v48, v47, v87 :: v_dual_fmac_f32 v49, v57, v87
	v_dual_fmac_f32 v58, v67, v87 :: v_dual_fmac_f32 v59, v77, v87
	v_dual_fmac_f32 v68, v47, v95 :: v_dual_fmac_f32 v69, v57, v95
	v_dual_fmac_f32 v78, v67, v95 :: v_dual_fmac_f32 v79, v77, v95
	v_dual_add_f32 v192, v192, v48 :: v_dual_add_f32 v193, v193, v49
	v_dual_add_f32 v194, v194, v58 :: v_dual_add_f32 v195, v195, v59
	v_dual_add_f32 v208, v208, v68 :: v_dual_add_f32 v209, v209, v69
	v_dual_add_f32 v210, v210, v78 :: v_dual_add_f32 v211, v211, v79
	s_wait_loadcnt 0x3
	v_dual_mul_f32 v48, v41, v97 :: v_dual_mul_f32 v49, v51, v97
	v_dual_mul_f32 v58, v61, v97 :: v_dual_mul_f32 v59, v71, v97
	s_wait_loadcnt 0x1
	v_dual_mul_f32 v68, v41, v105 :: v_dual_mul_f32 v69, v51, v105
	v_dual_mul_f32 v78, v61, v105 :: v_dual_mul_f32 v79, v71, v105
	v_dual_fmac_f32 v48, v40, v96 :: v_dual_fmac_f32 v49, v50, v96
	v_dual_fmac_f32 v58, v60, v96 :: v_dual_fmac_f32 v59, v70, v96
	v_dual_fmac_f32 v68, v40, v104 :: v_dual_fmac_f32 v69, v50, v104
	v_dual_fmac_f32 v78, v60, v104 :: v_dual_fmac_f32 v79, v70, v104
	v_dual_fmac_f32 v48, v42, v98 :: v_dual_fmac_f32 v49, v52, v98
	v_dual_fmac_f32 v58, v62, v98 :: v_dual_fmac_f32 v59, v72, v98
	v_dual_fmac_f32 v68, v42, v106 :: v_dual_fmac_f32 v69, v52, v106
	v_dual_fmac_f32 v78, v62, v106 :: v_dual_fmac_f32 v79, v72, v106
	v_dual_fmac_f32 v48, v43, v99 :: v_dual_fmac_f32 v49, v53, v99
	v_dual_fmac_f32 v58, v63, v99 :: v_dual_fmac_f32 v59, v73, v99
	v_dual_fmac_f32 v68, v43, v107 :: v_dual_fmac_f32 v69, v53, v107
	v_dual_fmac_f32 v78, v63, v107 :: v_dual_fmac_f32 v79, v73, v107
	v_dual_fmac_f32 v48, v44, v100 :: v_dual_fmac_f32 v49, v54, v100
	v_dual_fmac_f32 v58, v64, v100 :: v_dual_fmac_f32 v59, v74, v100
	s_wait_loadcnt 0x0
	v_dual_fmac_f32 v68, v44, v108 :: v_dual_fmac_f32 v69, v54, v108
	v_dual_fmac_f32 v78, v64, v108 :: v_dual_fmac_f32 v79, v74, v108
	v_dual_fmac_f32 v48, v45, v101 :: v_dual_fmac_f32 v49, v55, v101
	v_dual_fmac_f32 v58, v65, v101 :: v_dual_fmac_f32 v59, v75, v101
	v_dual_fmac_f32 v68, v45, v109 :: v_dual_fmac_f32 v69, v55, v109
	v_dual_fmac_f32 v78, v65, v109 :: v_dual_fmac_f32 v79, v75, v109
	v_dual_fmac_f32 v48, v46, v102 :: v_dual_fmac_f32 v49, v56, v102
	v_dual_fmac_f32 v58, v66, v102 :: v_dual_fmac_f32 v59, v76, v102
	v_dual_fmac_f32 v68, v46, v110 :: v_dual_fmac_f32 v69, v56, v110
	v_dual_fmac_f32 v78, v66, v110 :: v_dual_fmac_f32 v79, v76, v110
	v_dual_fmac_f32 v48, v47, v103 :: v_dual_fmac_f32 v49, v57, v103
	v_dual_fmac_f32 v58, v67, v103 :: v_dual_fmac_f32 v59, v77, v103
	v_dual_fmac_f32 v68, v47, v111 :: v_dual_fmac_f32 v69, v57, v111
	v_dual_fmac_f32 v78, v67, v111 :: v_dual_fmac_f32 v79, v77, v111
	v_dual_add_f32 v224, v224, v48 :: v_dual_add_f32 v225, v225, v49
	v_dual_add_f32 v226, v226, v58 :: v_dual_add_f32 v227, v227, v59
	v_dual_add_f32 v240, v240, v68 :: v_dual_add_f32 v241, v241, v69
	v_dual_add_f32 v242, v242, v78 :: v_dual_add_f32 v243, v243, v79
	s_cmp_gt_u32 s37, 1
	s_cbranch_scc0 .Lpx_b4_fold
	buffer_load_b32 v12, v1, s[20:23], s16 offen offset:136
	buffer_load_b32 v13, v1, s[20:23], s17 offen offset:136
	buffer_load_b32 v14, v1, s[20:23], s18 offen offset:136
	buffer_load_b32 v15, v1, s[20:23], s19 offen offset:136
	buffer_load_b32 v28, v2, s[20:23], s16 offen offset:144
	buffer_load_b32 v29, v2, s[20:23], s17 offen offset:144
	buffer_load_b32 v30, v2, s[20:23], s18 offen offset:144
	buffer_load_b32 v31, v2, s[20:23], s19 offen offset:144
	buffer_load_b128 v[80:83], v3, s[24:27], s28 offen offset:1024
	buffer_load_b128 v[84:87], v3, s[24:27], s28 offen offset:1040
	buffer_load_b128 v[88:91], v3, s[24:27], s29 offen offset:1024
	buffer_load_b128 v[92:95], v3, s[24:27], s29 offen offset:1040
	buffer_load_b128 v[96:99], v3, s[24:27], s30 offen offset:1024
	buffer_load_b128 v[100:103], v3, s[24:27], s30 offen offset:1040
	buffer_load_b128 v[104:107], v3, s[24:27], s31 offen offset:1024
	buffer_load_b128 v[108:111], v3, s[24:27], s31 offen offset:1040
	s_wait_loadcnt 0xb
	v_and_b32_e32 v46, 0xf0f0f0f, v28
	s_wait_loadcnt 0xa
	v_and_b32_e32 v56, 0xf0f0f0f, v29
	s_wait_loadcnt 0x9
	v_and_b32_e32 v66, 0xf0f0f0f, v30
	s_wait_loadcnt 0x8
	v_and_b32_e32 v76, 0xf0f0f0f, v31
	v_lshrrev_b32_e32 v47, 4, v28
	v_lshrrev_b32_e32 v57, 4, v29
	v_lshrrev_b32_e32 v67, 4, v30
	v_lshrrev_b32_e32 v77, 4, v31
	v_and_b32_e32 v47, 0xf0f0f0f, v47
	v_and_b32_e32 v57, 0xf0f0f0f, v57
	v_and_b32_e32 v67, 0xf0f0f0f, v67
	v_and_b32_e32 v77, 0xf0f0f0f, v77
	v_cvt_f32_ubyte0_e32 v40, v46
	v_cvt_f32_ubyte0_e32 v50, v56
	v_cvt_f32_ubyte0_e32 v60, v66
	v_cvt_f32_ubyte0_e32 v70, v76
	v_cvt_f32_ubyte1_e32 v42, v46
	v_cvt_f32_ubyte1_e32 v52, v56
	v_cvt_f32_ubyte1_e32 v62, v66
	v_cvt_f32_ubyte1_e32 v72, v76
	v_cvt_f32_ubyte2_e32 v44, v46
	v_cvt_f32_ubyte2_e32 v54, v56
	v_cvt_f32_ubyte2_e32 v64, v66
	v_cvt_f32_ubyte2_e32 v74, v76
	v_cvt_f32_ubyte3_e32 v46, v46
	v_cvt_f32_ubyte3_e32 v56, v56
	v_cvt_f32_ubyte3_e32 v66, v66
	v_cvt_f32_ubyte3_e32 v76, v76
	v_cvt_f32_ubyte0_e32 v41, v47
	v_cvt_f32_ubyte0_e32 v51, v57
	v_cvt_f32_ubyte0_e32 v61, v67
	v_cvt_f32_ubyte0_e32 v71, v77
	v_cvt_f32_ubyte1_e32 v43, v47
	v_cvt_f32_ubyte1_e32 v53, v57
	v_cvt_f32_ubyte1_e32 v63, v67
	v_cvt_f32_ubyte1_e32 v73, v77
	v_cvt_f32_ubyte2_e32 v45, v47
	v_cvt_f32_ubyte2_e32 v55, v57
	v_cvt_f32_ubyte2_e32 v65, v67
	v_cvt_f32_ubyte2_e32 v75, v77
	v_cvt_f32_ubyte3_e32 v47, v47
	v_cvt_f32_ubyte3_e32 v57, v57
	v_cvt_f32_ubyte3_e32 v67, v67
	v_cvt_f32_ubyte3_e32 v77, v77
	v_fma_mix_f32 v40, v12, v40, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v13, v50, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v14, v60, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v70, v15, v70, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v41, v12, v41, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v13, v51, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v14, v61, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v71, v15, v71, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v42, v12, v42, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v52, v13, v52, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v62, v14, v62, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v72, v15, v72, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v43, v12, v43, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v53, v13, v53, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v63, v14, v63, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v73, v15, v73, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v44, v12, v44, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v54, v13, v54, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v64, v14, v64, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v74, v15, v74, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v12, v45, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v13, v55, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v65, v14, v65, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v75, v15, v75, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v12, v46, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v13, v56, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v66, v14, v66, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v76, v15, v76, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v12, v47, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v13, v57, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v67, v14, v67, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v77, v15, v77, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x7
	v_dual_mul_f32 v48, v41, v81 :: v_dual_mul_f32 v49, v51, v81
	v_dual_mul_f32 v58, v61, v81 :: v_dual_mul_f32 v59, v71, v81
	s_wait_loadcnt 0x5
	v_dual_mul_f32 v68, v41, v89 :: v_dual_mul_f32 v69, v51, v89
	v_dual_mul_f32 v78, v61, v89 :: v_dual_mul_f32 v79, v71, v89
	v_dual_fmac_f32 v48, v40, v80 :: v_dual_fmac_f32 v49, v50, v80
	v_dual_fmac_f32 v58, v60, v80 :: v_dual_fmac_f32 v59, v70, v80
	v_dual_fmac_f32 v68, v40, v88 :: v_dual_fmac_f32 v69, v50, v88
	v_dual_fmac_f32 v78, v60, v88 :: v_dual_fmac_f32 v79, v70, v88
	v_dual_fmac_f32 v48, v42, v82 :: v_dual_fmac_f32 v49, v52, v82
	v_dual_fmac_f32 v58, v62, v82 :: v_dual_fmac_f32 v59, v72, v82
	v_dual_fmac_f32 v68, v42, v90 :: v_dual_fmac_f32 v69, v52, v90
	v_dual_fmac_f32 v78, v62, v90 :: v_dual_fmac_f32 v79, v72, v90
	v_dual_fmac_f32 v48, v43, v83 :: v_dual_fmac_f32 v49, v53, v83
	v_dual_fmac_f32 v58, v63, v83 :: v_dual_fmac_f32 v59, v73, v83
	v_dual_fmac_f32 v68, v43, v91 :: v_dual_fmac_f32 v69, v53, v91
	v_dual_fmac_f32 v78, v63, v91 :: v_dual_fmac_f32 v79, v73, v91
	v_dual_fmac_f32 v48, v44, v84 :: v_dual_fmac_f32 v49, v54, v84
	v_dual_fmac_f32 v58, v64, v84 :: v_dual_fmac_f32 v59, v74, v84
	s_wait_loadcnt 0x4
	v_dual_fmac_f32 v68, v44, v92 :: v_dual_fmac_f32 v69, v54, v92
	v_dual_fmac_f32 v78, v64, v92 :: v_dual_fmac_f32 v79, v74, v92
	v_dual_fmac_f32 v48, v45, v85 :: v_dual_fmac_f32 v49, v55, v85
	v_dual_fmac_f32 v58, v65, v85 :: v_dual_fmac_f32 v59, v75, v85
	v_dual_fmac_f32 v68, v45, v93 :: v_dual_fmac_f32 v69, v55, v93
	v_dual_fmac_f32 v78, v65, v93 :: v_dual_fmac_f32 v79, v75, v93
	v_dual_fmac_f32 v48, v46, v86 :: v_dual_fmac_f32 v49, v56, v86
	v_dual_fmac_f32 v58, v66, v86 :: v_dual_fmac_f32 v59, v76, v86
	v_dual_fmac_f32 v68, v46, v94 :: v_dual_fmac_f32 v69, v56, v94
	v_dual_fmac_f32 v78, v66, v94 :: v_dual_fmac_f32 v79, v76, v94
	v_dual_fmac_f32 v48, v47, v87 :: v_dual_fmac_f32 v49, v57, v87
	v_dual_fmac_f32 v58, v67, v87 :: v_dual_fmac_f32 v59, v77, v87
	v_dual_fmac_f32 v68, v47, v95 :: v_dual_fmac_f32 v69, v57, v95
	v_dual_fmac_f32 v78, v67, v95 :: v_dual_fmac_f32 v79, v77, v95
	v_dual_add_f32 v196, v196, v48 :: v_dual_add_f32 v197, v197, v49
	v_dual_add_f32 v198, v198, v58 :: v_dual_add_f32 v199, v199, v59
	v_dual_add_f32 v212, v212, v68 :: v_dual_add_f32 v213, v213, v69
	v_dual_add_f32 v214, v214, v78 :: v_dual_add_f32 v215, v215, v79
	s_wait_loadcnt 0x3
	v_dual_mul_f32 v48, v41, v97 :: v_dual_mul_f32 v49, v51, v97
	v_dual_mul_f32 v58, v61, v97 :: v_dual_mul_f32 v59, v71, v97
	s_wait_loadcnt 0x1
	v_dual_mul_f32 v68, v41, v105 :: v_dual_mul_f32 v69, v51, v105
	v_dual_mul_f32 v78, v61, v105 :: v_dual_mul_f32 v79, v71, v105
	v_dual_fmac_f32 v48, v40, v96 :: v_dual_fmac_f32 v49, v50, v96
	v_dual_fmac_f32 v58, v60, v96 :: v_dual_fmac_f32 v59, v70, v96
	v_dual_fmac_f32 v68, v40, v104 :: v_dual_fmac_f32 v69, v50, v104
	v_dual_fmac_f32 v78, v60, v104 :: v_dual_fmac_f32 v79, v70, v104
	v_dual_fmac_f32 v48, v42, v98 :: v_dual_fmac_f32 v49, v52, v98
	v_dual_fmac_f32 v58, v62, v98 :: v_dual_fmac_f32 v59, v72, v98
	v_dual_fmac_f32 v68, v42, v106 :: v_dual_fmac_f32 v69, v52, v106
	v_dual_fmac_f32 v78, v62, v106 :: v_dual_fmac_f32 v79, v72, v106
	v_dual_fmac_f32 v48, v43, v99 :: v_dual_fmac_f32 v49, v53, v99
	v_dual_fmac_f32 v58, v63, v99 :: v_dual_fmac_f32 v59, v73, v99
	v_dual_fmac_f32 v68, v43, v107 :: v_dual_fmac_f32 v69, v53, v107
	v_dual_fmac_f32 v78, v63, v107 :: v_dual_fmac_f32 v79, v73, v107
	v_dual_fmac_f32 v48, v44, v100 :: v_dual_fmac_f32 v49, v54, v100
	v_dual_fmac_f32 v58, v64, v100 :: v_dual_fmac_f32 v59, v74, v100
	s_wait_loadcnt 0x0
	v_dual_fmac_f32 v68, v44, v108 :: v_dual_fmac_f32 v69, v54, v108
	v_dual_fmac_f32 v78, v64, v108 :: v_dual_fmac_f32 v79, v74, v108
	v_dual_fmac_f32 v48, v45, v101 :: v_dual_fmac_f32 v49, v55, v101
	v_dual_fmac_f32 v58, v65, v101 :: v_dual_fmac_f32 v59, v75, v101
	v_dual_fmac_f32 v68, v45, v109 :: v_dual_fmac_f32 v69, v55, v109
	v_dual_fmac_f32 v78, v65, v109 :: v_dual_fmac_f32 v79, v75, v109
	v_dual_fmac_f32 v48, v46, v102 :: v_dual_fmac_f32 v49, v56, v102
	v_dual_fmac_f32 v58, v66, v102 :: v_dual_fmac_f32 v59, v76, v102
	v_dual_fmac_f32 v68, v46, v110 :: v_dual_fmac_f32 v69, v56, v110
	v_dual_fmac_f32 v78, v66, v110 :: v_dual_fmac_f32 v79, v76, v110
	v_dual_fmac_f32 v48, v47, v103 :: v_dual_fmac_f32 v49, v57, v103
	v_dual_fmac_f32 v58, v67, v103 :: v_dual_fmac_f32 v59, v77, v103
	v_dual_fmac_f32 v68, v47, v111 :: v_dual_fmac_f32 v69, v57, v111
	v_dual_fmac_f32 v78, v67, v111 :: v_dual_fmac_f32 v79, v77, v111
	v_dual_add_f32 v228, v228, v48 :: v_dual_add_f32 v229, v229, v49
	v_dual_add_f32 v230, v230, v58 :: v_dual_add_f32 v231, v231, v59
	v_dual_add_f32 v244, v244, v68 :: v_dual_add_f32 v245, v245, v69
	v_dual_add_f32 v246, v246, v78 :: v_dual_add_f32 v247, v247, v79
	s_cmp_gt_u32 s37, 2
	s_cbranch_scc0 .Lpx_b4_fold
	buffer_load_b32 v16, v1, s[20:23], s16 offen offset:272
	buffer_load_b32 v17, v1, s[20:23], s17 offen offset:272
	buffer_load_b32 v18, v1, s[20:23], s18 offen offset:272
	buffer_load_b32 v19, v1, s[20:23], s19 offen offset:272
	buffer_load_b32 v32, v2, s[20:23], s16 offen offset:280
	buffer_load_b32 v33, v2, s[20:23], s17 offen offset:280
	buffer_load_b32 v34, v2, s[20:23], s18 offen offset:280
	buffer_load_b32 v35, v2, s[20:23], s19 offen offset:280
	buffer_load_b128 v[80:83], v3, s[24:27], s28 offen offset:2048
	buffer_load_b128 v[84:87], v3, s[24:27], s28 offen offset:2064
	buffer_load_b128 v[88:91], v3, s[24:27], s29 offen offset:2048
	buffer_load_b128 v[92:95], v3, s[24:27], s29 offen offset:2064
	buffer_load_b128 v[96:99], v3, s[24:27], s30 offen offset:2048
	buffer_load_b128 v[100:103], v3, s[24:27], s30 offen offset:2064
	buffer_load_b128 v[104:107], v3, s[24:27], s31 offen offset:2048
	buffer_load_b128 v[108:111], v3, s[24:27], s31 offen offset:2064
	s_wait_loadcnt 0xb
	v_and_b32_e32 v46, 0xf0f0f0f, v32
	s_wait_loadcnt 0xa
	v_and_b32_e32 v56, 0xf0f0f0f, v33
	s_wait_loadcnt 0x9
	v_and_b32_e32 v66, 0xf0f0f0f, v34
	s_wait_loadcnt 0x8
	v_and_b32_e32 v76, 0xf0f0f0f, v35
	v_lshrrev_b32_e32 v47, 4, v32
	v_lshrrev_b32_e32 v57, 4, v33
	v_lshrrev_b32_e32 v67, 4, v34
	v_lshrrev_b32_e32 v77, 4, v35
	v_and_b32_e32 v47, 0xf0f0f0f, v47
	v_and_b32_e32 v57, 0xf0f0f0f, v57
	v_and_b32_e32 v67, 0xf0f0f0f, v67
	v_and_b32_e32 v77, 0xf0f0f0f, v77
	v_cvt_f32_ubyte0_e32 v40, v46
	v_cvt_f32_ubyte0_e32 v50, v56
	v_cvt_f32_ubyte0_e32 v60, v66
	v_cvt_f32_ubyte0_e32 v70, v76
	v_cvt_f32_ubyte1_e32 v42, v46
	v_cvt_f32_ubyte1_e32 v52, v56
	v_cvt_f32_ubyte1_e32 v62, v66
	v_cvt_f32_ubyte1_e32 v72, v76
	v_cvt_f32_ubyte2_e32 v44, v46
	v_cvt_f32_ubyte2_e32 v54, v56
	v_cvt_f32_ubyte2_e32 v64, v66
	v_cvt_f32_ubyte2_e32 v74, v76
	v_cvt_f32_ubyte3_e32 v46, v46
	v_cvt_f32_ubyte3_e32 v56, v56
	v_cvt_f32_ubyte3_e32 v66, v66
	v_cvt_f32_ubyte3_e32 v76, v76
	v_cvt_f32_ubyte0_e32 v41, v47
	v_cvt_f32_ubyte0_e32 v51, v57
	v_cvt_f32_ubyte0_e32 v61, v67
	v_cvt_f32_ubyte0_e32 v71, v77
	v_cvt_f32_ubyte1_e32 v43, v47
	v_cvt_f32_ubyte1_e32 v53, v57
	v_cvt_f32_ubyte1_e32 v63, v67
	v_cvt_f32_ubyte1_e32 v73, v77
	v_cvt_f32_ubyte2_e32 v45, v47
	v_cvt_f32_ubyte2_e32 v55, v57
	v_cvt_f32_ubyte2_e32 v65, v67
	v_cvt_f32_ubyte2_e32 v75, v77
	v_cvt_f32_ubyte3_e32 v47, v47
	v_cvt_f32_ubyte3_e32 v57, v57
	v_cvt_f32_ubyte3_e32 v67, v67
	v_cvt_f32_ubyte3_e32 v77, v77
	v_fma_mix_f32 v40, v16, v40, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v17, v50, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v18, v60, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v70, v19, v70, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v41, v16, v41, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v17, v51, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v18, v61, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v71, v19, v71, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v42, v16, v42, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v52, v17, v52, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v62, v18, v62, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v72, v19, v72, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v43, v16, v43, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v53, v17, v53, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v63, v18, v63, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v73, v19, v73, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v44, v16, v44, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v54, v17, v54, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v64, v18, v64, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v74, v19, v74, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v16, v45, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v17, v55, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v65, v18, v65, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v75, v19, v75, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v16, v46, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v17, v56, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v66, v18, v66, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v76, v19, v76, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v16, v47, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v17, v57, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v67, v18, v67, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v77, v19, v77, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x7
	v_dual_mul_f32 v48, v41, v81 :: v_dual_mul_f32 v49, v51, v81
	v_dual_mul_f32 v58, v61, v81 :: v_dual_mul_f32 v59, v71, v81
	s_wait_loadcnt 0x5
	v_dual_mul_f32 v68, v41, v89 :: v_dual_mul_f32 v69, v51, v89
	v_dual_mul_f32 v78, v61, v89 :: v_dual_mul_f32 v79, v71, v89
	v_dual_fmac_f32 v48, v40, v80 :: v_dual_fmac_f32 v49, v50, v80
	v_dual_fmac_f32 v58, v60, v80 :: v_dual_fmac_f32 v59, v70, v80
	v_dual_fmac_f32 v68, v40, v88 :: v_dual_fmac_f32 v69, v50, v88
	v_dual_fmac_f32 v78, v60, v88 :: v_dual_fmac_f32 v79, v70, v88
	v_dual_fmac_f32 v48, v42, v82 :: v_dual_fmac_f32 v49, v52, v82
	v_dual_fmac_f32 v58, v62, v82 :: v_dual_fmac_f32 v59, v72, v82
	v_dual_fmac_f32 v68, v42, v90 :: v_dual_fmac_f32 v69, v52, v90
	v_dual_fmac_f32 v78, v62, v90 :: v_dual_fmac_f32 v79, v72, v90
	v_dual_fmac_f32 v48, v43, v83 :: v_dual_fmac_f32 v49, v53, v83
	v_dual_fmac_f32 v58, v63, v83 :: v_dual_fmac_f32 v59, v73, v83
	v_dual_fmac_f32 v68, v43, v91 :: v_dual_fmac_f32 v69, v53, v91
	v_dual_fmac_f32 v78, v63, v91 :: v_dual_fmac_f32 v79, v73, v91
	v_dual_fmac_f32 v48, v44, v84 :: v_dual_fmac_f32 v49, v54, v84
	v_dual_fmac_f32 v58, v64, v84 :: v_dual_fmac_f32 v59, v74, v84
	s_wait_loadcnt 0x4
	v_dual_fmac_f32 v68, v44, v92 :: v_dual_fmac_f32 v69, v54, v92
	v_dual_fmac_f32 v78, v64, v92 :: v_dual_fmac_f32 v79, v74, v92
	v_dual_fmac_f32 v48, v45, v85 :: v_dual_fmac_f32 v49, v55, v85
	v_dual_fmac_f32 v58, v65, v85 :: v_dual_fmac_f32 v59, v75, v85
	v_dual_fmac_f32 v68, v45, v93 :: v_dual_fmac_f32 v69, v55, v93
	v_dual_fmac_f32 v78, v65, v93 :: v_dual_fmac_f32 v79, v75, v93
	v_dual_fmac_f32 v48, v46, v86 :: v_dual_fmac_f32 v49, v56, v86
	v_dual_fmac_f32 v58, v66, v86 :: v_dual_fmac_f32 v59, v76, v86
	v_dual_fmac_f32 v68, v46, v94 :: v_dual_fmac_f32 v69, v56, v94
	v_dual_fmac_f32 v78, v66, v94 :: v_dual_fmac_f32 v79, v76, v94
	v_dual_fmac_f32 v48, v47, v87 :: v_dual_fmac_f32 v49, v57, v87
	v_dual_fmac_f32 v58, v67, v87 :: v_dual_fmac_f32 v59, v77, v87
	v_dual_fmac_f32 v68, v47, v95 :: v_dual_fmac_f32 v69, v57, v95
	v_dual_fmac_f32 v78, v67, v95 :: v_dual_fmac_f32 v79, v77, v95
	v_dual_add_f32 v200, v200, v48 :: v_dual_add_f32 v201, v201, v49
	v_dual_add_f32 v202, v202, v58 :: v_dual_add_f32 v203, v203, v59
	v_dual_add_f32 v216, v216, v68 :: v_dual_add_f32 v217, v217, v69
	v_dual_add_f32 v218, v218, v78 :: v_dual_add_f32 v219, v219, v79
	s_wait_loadcnt 0x3
	v_dual_mul_f32 v48, v41, v97 :: v_dual_mul_f32 v49, v51, v97
	v_dual_mul_f32 v58, v61, v97 :: v_dual_mul_f32 v59, v71, v97
	s_wait_loadcnt 0x1
	v_dual_mul_f32 v68, v41, v105 :: v_dual_mul_f32 v69, v51, v105
	v_dual_mul_f32 v78, v61, v105 :: v_dual_mul_f32 v79, v71, v105
	v_dual_fmac_f32 v48, v40, v96 :: v_dual_fmac_f32 v49, v50, v96
	v_dual_fmac_f32 v58, v60, v96 :: v_dual_fmac_f32 v59, v70, v96
	v_dual_fmac_f32 v68, v40, v104 :: v_dual_fmac_f32 v69, v50, v104
	v_dual_fmac_f32 v78, v60, v104 :: v_dual_fmac_f32 v79, v70, v104
	v_dual_fmac_f32 v48, v42, v98 :: v_dual_fmac_f32 v49, v52, v98
	v_dual_fmac_f32 v58, v62, v98 :: v_dual_fmac_f32 v59, v72, v98
	v_dual_fmac_f32 v68, v42, v106 :: v_dual_fmac_f32 v69, v52, v106
	v_dual_fmac_f32 v78, v62, v106 :: v_dual_fmac_f32 v79, v72, v106
	v_dual_fmac_f32 v48, v43, v99 :: v_dual_fmac_f32 v49, v53, v99
	v_dual_fmac_f32 v58, v63, v99 :: v_dual_fmac_f32 v59, v73, v99
	v_dual_fmac_f32 v68, v43, v107 :: v_dual_fmac_f32 v69, v53, v107
	v_dual_fmac_f32 v78, v63, v107 :: v_dual_fmac_f32 v79, v73, v107
	v_dual_fmac_f32 v48, v44, v100 :: v_dual_fmac_f32 v49, v54, v100
	v_dual_fmac_f32 v58, v64, v100 :: v_dual_fmac_f32 v59, v74, v100
	s_wait_loadcnt 0x0
	v_dual_fmac_f32 v68, v44, v108 :: v_dual_fmac_f32 v69, v54, v108
	v_dual_fmac_f32 v78, v64, v108 :: v_dual_fmac_f32 v79, v74, v108
	v_dual_fmac_f32 v48, v45, v101 :: v_dual_fmac_f32 v49, v55, v101
	v_dual_fmac_f32 v58, v65, v101 :: v_dual_fmac_f32 v59, v75, v101
	v_dual_fmac_f32 v68, v45, v109 :: v_dual_fmac_f32 v69, v55, v109
	v_dual_fmac_f32 v78, v65, v109 :: v_dual_fmac_f32 v79, v75, v109
	v_dual_fmac_f32 v48, v46, v102 :: v_dual_fmac_f32 v49, v56, v102
	v_dual_fmac_f32 v58, v66, v102 :: v_dual_fmac_f32 v59, v76, v102
	v_dual_fmac_f32 v68, v46, v110 :: v_dual_fmac_f32 v69, v56, v110
	v_dual_fmac_f32 v78, v66, v110 :: v_dual_fmac_f32 v79, v76, v110
	v_dual_fmac_f32 v48, v47, v103 :: v_dual_fmac_f32 v49, v57, v103
	v_dual_fmac_f32 v58, v67, v103 :: v_dual_fmac_f32 v59, v77, v103
	v_dual_fmac_f32 v68, v47, v111 :: v_dual_fmac_f32 v69, v57, v111
	v_dual_fmac_f32 v78, v67, v111 :: v_dual_fmac_f32 v79, v77, v111
	v_dual_add_f32 v232, v232, v48 :: v_dual_add_f32 v233, v233, v49
	v_dual_add_f32 v234, v234, v58 :: v_dual_add_f32 v235, v235, v59
	v_dual_add_f32 v248, v248, v68 :: v_dual_add_f32 v249, v249, v69
	v_dual_add_f32 v250, v250, v78 :: v_dual_add_f32 v251, v251, v79
	.Lpx_b4_fold:
	v_dual_add_f32 v192, v192, v196 :: v_dual_add_f32 v193, v193, v197
	v_dual_add_f32 v194, v194, v198 :: v_dual_add_f32 v195, v195, v199
	v_dual_add_f32 v200, v200, v204 :: v_dual_add_f32 v201, v201, v205
	v_dual_add_f32 v202, v202, v206 :: v_dual_add_f32 v203, v203, v207
	v_dual_add_f32 v192, v192, v200 :: v_dual_add_f32 v193, v193, v201
	v_dual_add_f32 v194, v194, v202 :: v_dual_add_f32 v195, v195, v203
	v_dual_add_f32 v208, v208, v212 :: v_dual_add_f32 v209, v209, v213
	v_dual_add_f32 v210, v210, v214 :: v_dual_add_f32 v211, v211, v215
	v_dual_add_f32 v216, v216, v220 :: v_dual_add_f32 v217, v217, v221
	v_dual_add_f32 v218, v218, v222 :: v_dual_add_f32 v219, v219, v223
	v_dual_add_f32 v208, v208, v216 :: v_dual_add_f32 v209, v209, v217
	v_dual_add_f32 v210, v210, v218 :: v_dual_add_f32 v211, v211, v219
	v_dual_add_f32 v224, v224, v228 :: v_dual_add_f32 v225, v225, v229
	v_dual_add_f32 v226, v226, v230 :: v_dual_add_f32 v227, v227, v231
	v_dual_add_f32 v232, v232, v236 :: v_dual_add_f32 v233, v233, v237
	v_dual_add_f32 v234, v234, v238 :: v_dual_add_f32 v235, v235, v239
	v_dual_add_f32 v224, v224, v232 :: v_dual_add_f32 v225, v225, v233
	v_dual_add_f32 v226, v226, v234 :: v_dual_add_f32 v227, v227, v235
	v_dual_add_f32 v240, v240, v244 :: v_dual_add_f32 v241, v241, v245
	v_dual_add_f32 v242, v242, v246 :: v_dual_add_f32 v243, v243, v247
	v_dual_add_f32 v248, v248, v252 :: v_dual_add_f32 v249, v249, v253
	v_dual_add_f32 v250, v250, v254 :: v_dual_add_f32 v251, v251, v255
	v_dual_add_f32 v240, v240, v248 :: v_dual_add_f32 v241, v241, v249
	v_dual_add_f32 v242, v242, v250 :: v_dual_add_f32 v243, v243, v251
	ds_swizzle_b32 v8, v192 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v9, v193 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v10, v194 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v11, v195 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v12, v208 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v13, v209 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v14, v210 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v15, v211 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v16, v224 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v17, v225 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v18, v226 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v19, v227 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v20, v240 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v21, v241 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v22, v242 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v23, v243 offset:swizzle(BITMASK_PERM,"1pppp")
	s_wait_dscnt 0xf
	v_add_f32_e32 v192, v192, v8
	s_wait_dscnt 0xe
	v_add_f32_e32 v193, v193, v9
	s_wait_dscnt 0xd
	v_add_f32_e32 v194, v194, v10
	s_wait_dscnt 0xc
	v_add_f32_e32 v195, v195, v11
	s_wait_dscnt 0xb
	v_add_f32_e32 v208, v208, v12
	s_wait_dscnt 0xa
	v_add_f32_e32 v209, v209, v13
	s_wait_dscnt 0x9
	v_add_f32_e32 v210, v210, v14
	s_wait_dscnt 0x8
	v_add_f32_e32 v211, v211, v15
	s_wait_dscnt 0x7
	v_add_f32_e32 v224, v224, v16
	s_wait_dscnt 0x6
	v_add_f32_e32 v225, v225, v17
	s_wait_dscnt 0x5
	v_add_f32_e32 v226, v226, v18
	s_wait_dscnt 0x4
	v_add_f32_e32 v227, v227, v19
	s_wait_dscnt 0x3
	v_add_f32_e32 v240, v240, v20
	s_wait_dscnt 0x2
	v_add_f32_e32 v241, v241, v21
	s_wait_dscnt 0x1
	v_add_f32_e32 v242, v242, v22
	s_wait_dscnt 0x0
	v_add_f32_e32 v243, v243, v23
	v_cmp_gt_u32_e32 vcc_lo, 24, v0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v4, 0, 8, vcc_lo
	v_add_lshl_u32 v4, v4, v0, 2
	ds_bpermute_b32 v8, v4, v192
	ds_bpermute_b32 v9, v4, v193
	ds_bpermute_b32 v10, v4, v194
	ds_bpermute_b32 v11, v4, v195
	ds_bpermute_b32 v12, v4, v208
	ds_bpermute_b32 v13, v4, v209
	ds_bpermute_b32 v14, v4, v210
	ds_bpermute_b32 v15, v4, v211
	ds_bpermute_b32 v16, v4, v224
	ds_bpermute_b32 v17, v4, v225
	ds_bpermute_b32 v18, v4, v226
	ds_bpermute_b32 v19, v4, v227
	ds_bpermute_b32 v20, v4, v240
	ds_bpermute_b32 v21, v4, v241
	ds_bpermute_b32 v22, v4, v242
	ds_bpermute_b32 v23, v4, v243
	s_wait_dscnt 0xf
	v_add_f32_e32 v192, v192, v8
	s_wait_dscnt 0xe
	v_add_f32_e32 v193, v193, v9
	s_wait_dscnt 0xd
	v_add_f32_e32 v194, v194, v10
	s_wait_dscnt 0xc
	v_add_f32_e32 v195, v195, v11
	s_wait_dscnt 0xb
	v_add_f32_e32 v208, v208, v12
	s_wait_dscnt 0xa
	v_add_f32_e32 v209, v209, v13
	s_wait_dscnt 0x9
	v_add_f32_e32 v210, v210, v14
	s_wait_dscnt 0x8
	v_add_f32_e32 v211, v211, v15
	s_wait_dscnt 0x7
	v_add_f32_e32 v224, v224, v16
	s_wait_dscnt 0x6
	v_add_f32_e32 v225, v225, v17
	s_wait_dscnt 0x5
	v_add_f32_e32 v226, v226, v18
	s_wait_dscnt 0x4
	v_add_f32_e32 v227, v227, v19
	s_wait_dscnt 0x3
	v_add_f32_e32 v240, v240, v20
	s_wait_dscnt 0x2
	v_add_f32_e32 v241, v241, v21
	s_wait_dscnt 0x1
	v_add_f32_e32 v242, v242, v22
	s_wait_dscnt 0x0
	v_add_f32_e32 v243, v243, v23
	v_cmp_gt_u32_e32 vcc_lo, 28, v0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v4, 0, 4, vcc_lo
	v_add_lshl_u32 v4, v4, v0, 2
	ds_bpermute_b32 v8, v4, v192
	ds_bpermute_b32 v9, v4, v193
	ds_bpermute_b32 v10, v4, v194
	ds_bpermute_b32 v11, v4, v195
	ds_bpermute_b32 v12, v4, v208
	ds_bpermute_b32 v13, v4, v209
	ds_bpermute_b32 v14, v4, v210
	ds_bpermute_b32 v15, v4, v211
	ds_bpermute_b32 v16, v4, v224
	ds_bpermute_b32 v17, v4, v225
	ds_bpermute_b32 v18, v4, v226
	ds_bpermute_b32 v19, v4, v227
	ds_bpermute_b32 v20, v4, v240
	ds_bpermute_b32 v21, v4, v241
	ds_bpermute_b32 v22, v4, v242
	ds_bpermute_b32 v23, v4, v243
	s_wait_dscnt 0xf
	v_add_f32_e32 v192, v192, v8
	s_wait_dscnt 0xe
	v_add_f32_e32 v193, v193, v9
	s_wait_dscnt 0xd
	v_add_f32_e32 v194, v194, v10
	s_wait_dscnt 0xc
	v_add_f32_e32 v195, v195, v11
	s_wait_dscnt 0xb
	v_add_f32_e32 v208, v208, v12
	s_wait_dscnt 0xa
	v_add_f32_e32 v209, v209, v13
	s_wait_dscnt 0x9
	v_add_f32_e32 v210, v210, v14
	s_wait_dscnt 0x8
	v_add_f32_e32 v211, v211, v15
	s_wait_dscnt 0x7
	v_add_f32_e32 v224, v224, v16
	s_wait_dscnt 0x6
	v_add_f32_e32 v225, v225, v17
	s_wait_dscnt 0x5
	v_add_f32_e32 v226, v226, v18
	s_wait_dscnt 0x4
	v_add_f32_e32 v227, v227, v19
	s_wait_dscnt 0x3
	v_add_f32_e32 v240, v240, v20
	s_wait_dscnt 0x2
	v_add_f32_e32 v241, v241, v21
	s_wait_dscnt 0x1
	v_add_f32_e32 v242, v242, v22
	s_wait_dscnt 0x0
	v_add_f32_e32 v243, v243, v23
	v_cmp_gt_u32_e32 vcc_lo, 30, v0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v4, 0, 2, vcc_lo
	v_add_lshl_u32 v4, v4, v0, 2
	ds_bpermute_b32 v8, v4, v192
	ds_bpermute_b32 v9, v4, v193
	ds_bpermute_b32 v10, v4, v194
	ds_bpermute_b32 v11, v4, v195
	ds_bpermute_b32 v12, v4, v208
	ds_bpermute_b32 v13, v4, v209
	ds_bpermute_b32 v14, v4, v210
	ds_bpermute_b32 v15, v4, v211
	ds_bpermute_b32 v16, v4, v224
	ds_bpermute_b32 v17, v4, v225
	ds_bpermute_b32 v18, v4, v226
	ds_bpermute_b32 v19, v4, v227
	ds_bpermute_b32 v20, v4, v240
	ds_bpermute_b32 v21, v4, v241
	ds_bpermute_b32 v22, v4, v242
	ds_bpermute_b32 v23, v4, v243
	s_wait_dscnt 0xf
	v_add_f32_e32 v192, v192, v8
	s_wait_dscnt 0xe
	v_add_f32_e32 v193, v193, v9
	s_wait_dscnt 0xd
	v_add_f32_e32 v194, v194, v10
	s_wait_dscnt 0xc
	v_add_f32_e32 v195, v195, v11
	s_wait_dscnt 0xb
	v_add_f32_e32 v208, v208, v12
	s_wait_dscnt 0xa
	v_add_f32_e32 v209, v209, v13
	s_wait_dscnt 0x9
	v_add_f32_e32 v210, v210, v14
	s_wait_dscnt 0x8
	v_add_f32_e32 v211, v211, v15
	s_wait_dscnt 0x7
	v_add_f32_e32 v224, v224, v16
	s_wait_dscnt 0x6
	v_add_f32_e32 v225, v225, v17
	s_wait_dscnt 0x5
	v_add_f32_e32 v226, v226, v18
	s_wait_dscnt 0x4
	v_add_f32_e32 v227, v227, v19
	s_wait_dscnt 0x3
	v_add_f32_e32 v240, v240, v20
	s_wait_dscnt 0x2
	v_add_f32_e32 v241, v241, v21
	s_wait_dscnt 0x1
	v_add_f32_e32 v242, v242, v22
	s_wait_dscnt 0x0
	v_add_f32_e32 v243, v243, v23
	v_cmp_gt_u32_e32 vcc_lo, 31, v0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v4, 0, 1, vcc_lo
	v_add_lshl_u32 v4, v4, v0, 2
	ds_bpermute_b32 v8, v4, v192
	ds_bpermute_b32 v9, v4, v193
	ds_bpermute_b32 v10, v4, v194
	ds_bpermute_b32 v11, v4, v195
	ds_bpermute_b32 v12, v4, v208
	ds_bpermute_b32 v13, v4, v209
	ds_bpermute_b32 v14, v4, v210
	ds_bpermute_b32 v15, v4, v211
	ds_bpermute_b32 v16, v4, v224
	ds_bpermute_b32 v17, v4, v225
	ds_bpermute_b32 v18, v4, v226
	ds_bpermute_b32 v19, v4, v227
	ds_bpermute_b32 v20, v4, v240
	ds_bpermute_b32 v21, v4, v241
	ds_bpermute_b32 v22, v4, v242
	ds_bpermute_b32 v23, v4, v243
	s_wait_dscnt 0xf
	v_add_f32_e32 v192, v192, v8
	s_wait_dscnt 0xe
	v_add_f32_e32 v193, v193, v9
	s_wait_dscnt 0xd
	v_add_f32_e32 v194, v194, v10
	s_wait_dscnt 0xc
	v_add_f32_e32 v195, v195, v11
	s_wait_dscnt 0xb
	v_add_f32_e32 v208, v208, v12
	s_wait_dscnt 0xa
	v_add_f32_e32 v209, v209, v13
	s_wait_dscnt 0x9
	v_add_f32_e32 v210, v210, v14
	s_wait_dscnt 0x8
	v_add_f32_e32 v211, v211, v15
	s_wait_dscnt 0x7
	v_add_f32_e32 v224, v224, v16
	s_wait_dscnt 0x6
	v_add_f32_e32 v225, v225, v17
	s_wait_dscnt 0x5
	v_add_f32_e32 v226, v226, v18
	s_wait_dscnt 0x4
	v_add_f32_e32 v227, v227, v19
	s_wait_dscnt 0x3
	v_add_f32_e32 v240, v240, v20
	s_wait_dscnt 0x2
	v_add_f32_e32 v241, v241, v21
	s_wait_dscnt 0x1
	v_add_f32_e32 v242, v242, v22
	s_wait_dscnt 0x0
	v_add_f32_e32 v243, v243, v23
	v_cmpx_eq_u32_e32 0, v0
	v_mov_b32_e32 v40, s42
	s_mul_i32 s40, s39, 1
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s40, s40, s42
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v41, s40
	s_mul_i32 s40, s39, 2
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s40, s40, s42
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v42, s40
	s_mul_i32 s40, s39, 3
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s40, s40, s42
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v43, s40
	global_store_b32 v40, v192, s[8:9]
	global_store_b32 v41, v208, s[8:9]
	global_store_b32 v42, v224, s[8:9]
	global_store_b32 v43, v240, s[8:9]
	s_cmp_le_i32 s13, 1
	s_cbranch_scc1 .Lpx_b4_stored
	global_store_b32 v40, v193, s[8:9] offset:4
	global_store_b32 v41, v209, s[8:9] offset:4
	global_store_b32 v42, v225, s[8:9] offset:4
	global_store_b32 v43, v241, s[8:9] offset:4
	s_cmp_le_i32 s13, 2
	s_cbranch_scc1 .Lpx_b4_stored
	global_store_b32 v40, v194, s[8:9] offset:8
	global_store_b32 v41, v210, s[8:9] offset:8
	global_store_b32 v42, v226, s[8:9] offset:8
	global_store_b32 v43, v242, s[8:9] offset:8
	s_cmp_le_i32 s13, 3
	s_cbranch_scc1 .Lpx_b4_stored
	global_store_b32 v40, v195, s[8:9] offset:12
	global_store_b32 v41, v211, s[8:9] offset:12
	global_store_b32 v42, v227, s[8:9] offset:12
	global_store_b32 v43, v243, s[8:9] offset:12
	.Lpx_b4_stored:
	s_wait_storecnt 0x0
	s_branch .Lpx_end
	.Lpx_b5:
	v_mov_b32_e32 v176, 0
	v_mov_b32_e32 v177, 0
	v_mov_b32_e32 v178, 0
	v_mov_b32_e32 v179, 0
	v_mov_b32_e32 v180, 0
	v_mov_b32_e32 v181, 0
	v_mov_b32_e32 v182, 0
	v_mov_b32_e32 v183, 0
	v_mov_b32_e32 v184, 0
	v_mov_b32_e32 v185, 0
	v_mov_b32_e32 v186, 0
	v_mov_b32_e32 v187, 0
	v_mov_b32_e32 v188, 0
	v_mov_b32_e32 v189, 0
	v_mov_b32_e32 v190, 0
	v_mov_b32_e32 v191, 0
	v_mov_b32_e32 v192, 0
	v_mov_b32_e32 v193, 0
	v_mov_b32_e32 v194, 0
	v_mov_b32_e32 v195, 0
	v_mov_b32_e32 v196, 0
	v_mov_b32_e32 v197, 0
	v_mov_b32_e32 v198, 0
	v_mov_b32_e32 v199, 0
	v_mov_b32_e32 v200, 0
	v_mov_b32_e32 v201, 0
	v_mov_b32_e32 v202, 0
	v_mov_b32_e32 v203, 0
	v_mov_b32_e32 v204, 0
	v_mov_b32_e32 v205, 0
	v_mov_b32_e32 v206, 0
	v_mov_b32_e32 v207, 0
	v_mov_b32_e32 v208, 0
	v_mov_b32_e32 v209, 0
	v_mov_b32_e32 v210, 0
	v_mov_b32_e32 v211, 0
	v_mov_b32_e32 v212, 0
	v_mov_b32_e32 v213, 0
	v_mov_b32_e32 v214, 0
	v_mov_b32_e32 v215, 0
	v_mov_b32_e32 v216, 0
	v_mov_b32_e32 v217, 0
	v_mov_b32_e32 v218, 0
	v_mov_b32_e32 v219, 0
	v_mov_b32_e32 v220, 0
	v_mov_b32_e32 v221, 0
	v_mov_b32_e32 v222, 0
	v_mov_b32_e32 v223, 0
	v_mov_b32_e32 v224, 0
	v_mov_b32_e32 v225, 0
	v_mov_b32_e32 v226, 0
	v_mov_b32_e32 v227, 0
	v_mov_b32_e32 v228, 0
	v_mov_b32_e32 v229, 0
	v_mov_b32_e32 v230, 0
	v_mov_b32_e32 v231, 0
	v_mov_b32_e32 v232, 0
	v_mov_b32_e32 v233, 0
	v_mov_b32_e32 v234, 0
	v_mov_b32_e32 v235, 0
	v_mov_b32_e32 v236, 0
	v_mov_b32_e32 v237, 0
	v_mov_b32_e32 v238, 0
	v_mov_b32_e32 v239, 0
	v_mov_b32_e32 v240, 0
	v_mov_b32_e32 v241, 0
	v_mov_b32_e32 v242, 0
	v_mov_b32_e32 v243, 0
	v_mov_b32_e32 v244, 0
	v_mov_b32_e32 v245, 0
	v_mov_b32_e32 v246, 0
	v_mov_b32_e32 v247, 0
	v_mov_b32_e32 v248, 0
	v_mov_b32_e32 v249, 0
	v_mov_b32_e32 v250, 0
	v_mov_b32_e32 v251, 0
	v_mov_b32_e32 v252, 0
	v_mov_b32_e32 v253, 0
	v_mov_b32_e32 v254, 0
	v_mov_b32_e32 v255, 0
	s_lshr_b32 s36, s14, 2
	s_and_b32 s37, s14, 3
	s_cmp_eq_u32 s36, 0
	s_cbranch_scc1 .Lpx_b5_tail
	.Lpx_b5_quad:
	buffer_load_b32 v8, v1, s[20:23], s16 offen
	buffer_load_b32 v9, v1, s[20:23], s17 offen
	buffer_load_b32 v10, v1, s[20:23], s18 offen
	buffer_load_b32 v11, v1, s[20:23], s19 offen
	buffer_load_b32 v24, v2, s[20:23], s16 offen offset:8
	buffer_load_b32 v25, v2, s[20:23], s17 offen offset:8
	buffer_load_b32 v26, v2, s[20:23], s18 offen offset:8
	buffer_load_b32 v27, v2, s[20:23], s19 offen offset:8
	buffer_load_b128 v[80:83], v3, s[24:27], s28 offen
	buffer_load_b128 v[84:87], v3, s[24:27], s28 offen offset:16
	buffer_load_b128 v[88:91], v3, s[24:27], s29 offen
	buffer_load_b128 v[92:95], v3, s[24:27], s29 offen offset:16
	buffer_load_b128 v[96:99], v3, s[24:27], s30 offen
	buffer_load_b128 v[100:103], v3, s[24:27], s30 offen offset:16
	buffer_load_b128 v[104:107], v3, s[24:27], s31 offen
	buffer_load_b128 v[108:111], v3, s[24:27], s31 offen offset:16
	buffer_load_b128 v[112:115], v3, s[24:27], s32 offen
	buffer_load_b128 v[116:119], v3, s[24:27], s32 offen offset:16
	buffer_load_b32 v12, v1, s[20:23], s16 offen offset:136
	buffer_load_b32 v13, v1, s[20:23], s17 offen offset:136
	buffer_load_b32 v14, v1, s[20:23], s18 offen offset:136
	buffer_load_b32 v15, v1, s[20:23], s19 offen offset:136
	buffer_load_b32 v28, v2, s[20:23], s16 offen offset:144
	buffer_load_b32 v29, v2, s[20:23], s17 offen offset:144
	buffer_load_b32 v30, v2, s[20:23], s18 offen offset:144
	buffer_load_b32 v31, v2, s[20:23], s19 offen offset:144
	buffer_load_b32 v16, v1, s[20:23], s16 offen offset:272
	buffer_load_b32 v17, v1, s[20:23], s17 offen offset:272
	buffer_load_b32 v18, v1, s[20:23], s18 offen offset:272
	buffer_load_b32 v19, v1, s[20:23], s19 offen offset:272
	buffer_load_b32 v32, v2, s[20:23], s16 offen offset:280
	buffer_load_b32 v33, v2, s[20:23], s17 offen offset:280
	buffer_load_b32 v34, v2, s[20:23], s18 offen offset:280
	buffer_load_b32 v35, v2, s[20:23], s19 offen offset:280
	buffer_load_b32 v20, v1, s[20:23], s16 offen offset:408
	buffer_load_b32 v21, v1, s[20:23], s17 offen offset:408
	buffer_load_b32 v22, v1, s[20:23], s18 offen offset:408
	buffer_load_b32 v23, v1, s[20:23], s19 offen offset:408
	buffer_load_b32 v36, v2, s[20:23], s16 offen offset:416
	buffer_load_b32 v37, v2, s[20:23], s17 offen offset:416
	buffer_load_b32 v38, v2, s[20:23], s18 offen offset:416
	buffer_load_b32 v39, v2, s[20:23], s19 offen offset:416
	s_wait_loadcnt 0x25
	v_and_b32_e32 v46, 0xf0f0f0f, v24
	s_wait_loadcnt 0x24
	v_and_b32_e32 v56, 0xf0f0f0f, v25
	s_wait_loadcnt 0x23
	v_and_b32_e32 v66, 0xf0f0f0f, v26
	s_wait_loadcnt 0x22
	v_and_b32_e32 v76, 0xf0f0f0f, v27
	v_lshrrev_b32_e32 v47, 4, v24
	v_lshrrev_b32_e32 v57, 4, v25
	v_lshrrev_b32_e32 v67, 4, v26
	v_lshrrev_b32_e32 v77, 4, v27
	v_and_b32_e32 v47, 0xf0f0f0f, v47
	v_and_b32_e32 v57, 0xf0f0f0f, v57
	v_and_b32_e32 v67, 0xf0f0f0f, v67
	v_and_b32_e32 v77, 0xf0f0f0f, v77
	v_cvt_f32_ubyte0_e32 v40, v46
	v_cvt_f32_ubyte0_e32 v50, v56
	v_cvt_f32_ubyte0_e32 v60, v66
	v_cvt_f32_ubyte0_e32 v70, v76
	v_cvt_f32_ubyte1_e32 v42, v46
	v_cvt_f32_ubyte1_e32 v52, v56
	v_cvt_f32_ubyte1_e32 v62, v66
	v_cvt_f32_ubyte1_e32 v72, v76
	v_cvt_f32_ubyte2_e32 v44, v46
	v_cvt_f32_ubyte2_e32 v54, v56
	v_cvt_f32_ubyte2_e32 v64, v66
	v_cvt_f32_ubyte2_e32 v74, v76
	v_cvt_f32_ubyte3_e32 v46, v46
	v_cvt_f32_ubyte3_e32 v56, v56
	v_cvt_f32_ubyte3_e32 v66, v66
	v_cvt_f32_ubyte3_e32 v76, v76
	v_cvt_f32_ubyte0_e32 v41, v47
	v_cvt_f32_ubyte0_e32 v51, v57
	v_cvt_f32_ubyte0_e32 v61, v67
	v_cvt_f32_ubyte0_e32 v71, v77
	v_cvt_f32_ubyte1_e32 v43, v47
	v_cvt_f32_ubyte1_e32 v53, v57
	v_cvt_f32_ubyte1_e32 v63, v67
	v_cvt_f32_ubyte1_e32 v73, v77
	v_cvt_f32_ubyte2_e32 v45, v47
	v_cvt_f32_ubyte2_e32 v55, v57
	v_cvt_f32_ubyte2_e32 v65, v67
	v_cvt_f32_ubyte2_e32 v75, v77
	v_cvt_f32_ubyte3_e32 v47, v47
	v_cvt_f32_ubyte3_e32 v57, v57
	v_cvt_f32_ubyte3_e32 v67, v67
	v_cvt_f32_ubyte3_e32 v77, v77
	v_fma_mix_f32 v40, v8, v40, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v9, v50, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v10, v60, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v70, v11, v70, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v41, v8, v41, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v9, v51, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v10, v61, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v71, v11, v71, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v42, v8, v42, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v52, v9, v52, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v62, v10, v62, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v72, v11, v72, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v43, v8, v43, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v53, v9, v53, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v63, v10, v63, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v73, v11, v73, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v44, v8, v44, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v54, v9, v54, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v64, v10, v64, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v74, v11, v74, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v8, v45, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v9, v55, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v65, v10, v65, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v75, v11, v75, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v8, v46, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v9, v56, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v66, v10, v66, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v76, v11, v76, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v8, v47, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v9, v57, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v67, v10, v67, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v77, v11, v77, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	buffer_load_b128 v[120:123], v3, s[24:27], s28 offen offset:1024
	buffer_load_b128 v[124:127], v3, s[24:27], s28 offen offset:1040
	buffer_load_b128 v[128:131], v3, s[24:27], s29 offen offset:1024
	buffer_load_b128 v[132:135], v3, s[24:27], s29 offen offset:1040
	buffer_load_b128 v[136:139], v3, s[24:27], s30 offen offset:1024
	buffer_load_b128 v[140:143], v3, s[24:27], s30 offen offset:1040
	buffer_load_b128 v[144:147], v3, s[24:27], s31 offen offset:1024
	buffer_load_b128 v[148:151], v3, s[24:27], s31 offen offset:1040
	buffer_load_b128 v[152:155], v3, s[24:27], s32 offen offset:1024
	buffer_load_b128 v[156:159], v3, s[24:27], s32 offen offset:1040
	buffer_load_b128 v[160:163], v3, s[24:27], s28 offen offset:2048
	buffer_load_b128 v[164:167], v3, s[24:27], s28 offen offset:2064
	buffer_load_b128 v[168:171], v3, s[24:27], s29 offen offset:2048
	buffer_load_b128 v[172:175], v3, s[24:27], s29 offen offset:2064
	s_wait_loadcnt 0x2f
	v_dual_mul_f32 v48, v41, v81 :: v_dual_mul_f32 v49, v51, v81
	v_dual_mul_f32 v58, v61, v81 :: v_dual_mul_f32 v59, v71, v81
	s_wait_loadcnt 0x2d
	v_dual_mul_f32 v68, v41, v89 :: v_dual_mul_f32 v69, v51, v89
	v_dual_mul_f32 v78, v61, v89 :: v_dual_mul_f32 v79, v71, v89
	v_dual_fmac_f32 v48, v40, v80 :: v_dual_fmac_f32 v49, v50, v80
	v_dual_fmac_f32 v58, v60, v80 :: v_dual_fmac_f32 v59, v70, v80
	v_dual_fmac_f32 v68, v40, v88 :: v_dual_fmac_f32 v69, v50, v88
	v_dual_fmac_f32 v78, v60, v88 :: v_dual_fmac_f32 v79, v70, v88
	v_dual_fmac_f32 v48, v42, v82 :: v_dual_fmac_f32 v49, v52, v82
	v_dual_fmac_f32 v58, v62, v82 :: v_dual_fmac_f32 v59, v72, v82
	v_dual_fmac_f32 v68, v42, v90 :: v_dual_fmac_f32 v69, v52, v90
	v_dual_fmac_f32 v78, v62, v90 :: v_dual_fmac_f32 v79, v72, v90
	v_dual_fmac_f32 v48, v43, v83 :: v_dual_fmac_f32 v49, v53, v83
	v_dual_fmac_f32 v58, v63, v83 :: v_dual_fmac_f32 v59, v73, v83
	v_dual_fmac_f32 v68, v43, v91 :: v_dual_fmac_f32 v69, v53, v91
	v_dual_fmac_f32 v78, v63, v91 :: v_dual_fmac_f32 v79, v73, v91
	v_dual_fmac_f32 v48, v44, v84 :: v_dual_fmac_f32 v49, v54, v84
	v_dual_fmac_f32 v58, v64, v84 :: v_dual_fmac_f32 v59, v74, v84
	s_wait_loadcnt 0x2c
	v_dual_fmac_f32 v68, v44, v92 :: v_dual_fmac_f32 v69, v54, v92
	v_dual_fmac_f32 v78, v64, v92 :: v_dual_fmac_f32 v79, v74, v92
	v_dual_fmac_f32 v48, v45, v85 :: v_dual_fmac_f32 v49, v55, v85
	v_dual_fmac_f32 v58, v65, v85 :: v_dual_fmac_f32 v59, v75, v85
	v_dual_fmac_f32 v68, v45, v93 :: v_dual_fmac_f32 v69, v55, v93
	v_dual_fmac_f32 v78, v65, v93 :: v_dual_fmac_f32 v79, v75, v93
	v_dual_fmac_f32 v48, v46, v86 :: v_dual_fmac_f32 v49, v56, v86
	v_dual_fmac_f32 v58, v66, v86 :: v_dual_fmac_f32 v59, v76, v86
	v_dual_fmac_f32 v68, v46, v94 :: v_dual_fmac_f32 v69, v56, v94
	v_dual_fmac_f32 v78, v66, v94 :: v_dual_fmac_f32 v79, v76, v94
	v_dual_fmac_f32 v48, v47, v87 :: v_dual_fmac_f32 v49, v57, v87
	v_dual_fmac_f32 v58, v67, v87 :: v_dual_fmac_f32 v59, v77, v87
	v_dual_fmac_f32 v68, v47, v95 :: v_dual_fmac_f32 v69, v57, v95
	v_dual_fmac_f32 v78, v67, v95 :: v_dual_fmac_f32 v79, v77, v95
	v_dual_add_f32 v176, v176, v48 :: v_dual_add_f32 v177, v177, v49
	v_dual_add_f32 v178, v178, v58 :: v_dual_add_f32 v179, v179, v59
	v_dual_add_f32 v192, v192, v68 :: v_dual_add_f32 v193, v193, v69
	v_dual_add_f32 v194, v194, v78 :: v_dual_add_f32 v195, v195, v79
	buffer_load_b128 v[80:83], v3, s[24:27], s30 offen offset:2048
	buffer_load_b128 v[84:87], v3, s[24:27], s30 offen offset:2064
	buffer_load_b128 v[88:91], v3, s[24:27], s31 offen offset:2048
	buffer_load_b128 v[92:95], v3, s[24:27], s31 offen offset:2064
	s_wait_loadcnt 0x2f
	v_dual_mul_f32 v48, v41, v97 :: v_dual_mul_f32 v49, v51, v97
	v_dual_mul_f32 v58, v61, v97 :: v_dual_mul_f32 v59, v71, v97
	s_wait_loadcnt 0x2d
	v_dual_mul_f32 v68, v41, v105 :: v_dual_mul_f32 v69, v51, v105
	v_dual_mul_f32 v78, v61, v105 :: v_dual_mul_f32 v79, v71, v105
	v_dual_fmac_f32 v48, v40, v96 :: v_dual_fmac_f32 v49, v50, v96
	v_dual_fmac_f32 v58, v60, v96 :: v_dual_fmac_f32 v59, v70, v96
	v_dual_fmac_f32 v68, v40, v104 :: v_dual_fmac_f32 v69, v50, v104
	v_dual_fmac_f32 v78, v60, v104 :: v_dual_fmac_f32 v79, v70, v104
	v_dual_fmac_f32 v48, v42, v98 :: v_dual_fmac_f32 v49, v52, v98
	v_dual_fmac_f32 v58, v62, v98 :: v_dual_fmac_f32 v59, v72, v98
	v_dual_fmac_f32 v68, v42, v106 :: v_dual_fmac_f32 v69, v52, v106
	v_dual_fmac_f32 v78, v62, v106 :: v_dual_fmac_f32 v79, v72, v106
	v_dual_fmac_f32 v48, v43, v99 :: v_dual_fmac_f32 v49, v53, v99
	v_dual_fmac_f32 v58, v63, v99 :: v_dual_fmac_f32 v59, v73, v99
	v_dual_fmac_f32 v68, v43, v107 :: v_dual_fmac_f32 v69, v53, v107
	v_dual_fmac_f32 v78, v63, v107 :: v_dual_fmac_f32 v79, v73, v107
	v_dual_fmac_f32 v48, v44, v100 :: v_dual_fmac_f32 v49, v54, v100
	v_dual_fmac_f32 v58, v64, v100 :: v_dual_fmac_f32 v59, v74, v100
	s_wait_loadcnt 0x2c
	v_dual_fmac_f32 v68, v44, v108 :: v_dual_fmac_f32 v69, v54, v108
	v_dual_fmac_f32 v78, v64, v108 :: v_dual_fmac_f32 v79, v74, v108
	v_dual_fmac_f32 v48, v45, v101 :: v_dual_fmac_f32 v49, v55, v101
	v_dual_fmac_f32 v58, v65, v101 :: v_dual_fmac_f32 v59, v75, v101
	v_dual_fmac_f32 v68, v45, v109 :: v_dual_fmac_f32 v69, v55, v109
	v_dual_fmac_f32 v78, v65, v109 :: v_dual_fmac_f32 v79, v75, v109
	v_dual_fmac_f32 v48, v46, v102 :: v_dual_fmac_f32 v49, v56, v102
	v_dual_fmac_f32 v58, v66, v102 :: v_dual_fmac_f32 v59, v76, v102
	v_dual_fmac_f32 v68, v46, v110 :: v_dual_fmac_f32 v69, v56, v110
	v_dual_fmac_f32 v78, v66, v110 :: v_dual_fmac_f32 v79, v76, v110
	v_dual_fmac_f32 v48, v47, v103 :: v_dual_fmac_f32 v49, v57, v103
	v_dual_fmac_f32 v58, v67, v103 :: v_dual_fmac_f32 v59, v77, v103
	v_dual_fmac_f32 v68, v47, v111 :: v_dual_fmac_f32 v69, v57, v111
	v_dual_fmac_f32 v78, v67, v111 :: v_dual_fmac_f32 v79, v77, v111
	v_dual_add_f32 v208, v208, v48 :: v_dual_add_f32 v209, v209, v49
	v_dual_add_f32 v210, v210, v58 :: v_dual_add_f32 v211, v211, v59
	v_dual_add_f32 v224, v224, v68 :: v_dual_add_f32 v225, v225, v69
	v_dual_add_f32 v226, v226, v78 :: v_dual_add_f32 v227, v227, v79
	buffer_load_b128 v[96:99], v3, s[24:27], s32 offen offset:2048
	buffer_load_b128 v[100:103], v3, s[24:27], s32 offen offset:2064
	buffer_load_b128 v[104:107], v3, s[24:27], s28 offen offset:3072
	buffer_load_b128 v[108:111], v3, s[24:27], s28 offen offset:3088
	s_wait_loadcnt 0x2f
	v_dual_mul_f32 v48, v41, v113 :: v_dual_mul_f32 v49, v51, v113
	v_dual_mul_f32 v58, v61, v113 :: v_dual_mul_f32 v59, v71, v113
	v_dual_fmac_f32 v48, v40, v112 :: v_dual_fmac_f32 v49, v50, v112
	v_dual_fmac_f32 v58, v60, v112 :: v_dual_fmac_f32 v59, v70, v112
	v_dual_fmac_f32 v48, v42, v114 :: v_dual_fmac_f32 v49, v52, v114
	v_dual_fmac_f32 v58, v62, v114 :: v_dual_fmac_f32 v59, v72, v114
	v_dual_fmac_f32 v48, v43, v115 :: v_dual_fmac_f32 v49, v53, v115
	v_dual_fmac_f32 v58, v63, v115 :: v_dual_fmac_f32 v59, v73, v115
	s_wait_loadcnt 0x2e
	v_dual_fmac_f32 v48, v44, v116 :: v_dual_fmac_f32 v49, v54, v116
	v_dual_fmac_f32 v58, v64, v116 :: v_dual_fmac_f32 v59, v74, v116
	v_dual_fmac_f32 v48, v45, v117 :: v_dual_fmac_f32 v49, v55, v117
	v_dual_fmac_f32 v58, v65, v117 :: v_dual_fmac_f32 v59, v75, v117
	v_dual_fmac_f32 v48, v46, v118 :: v_dual_fmac_f32 v49, v56, v118
	v_dual_fmac_f32 v58, v66, v118 :: v_dual_fmac_f32 v59, v76, v118
	v_dual_fmac_f32 v48, v47, v119 :: v_dual_fmac_f32 v49, v57, v119
	v_dual_fmac_f32 v58, v67, v119 :: v_dual_fmac_f32 v59, v77, v119
	v_dual_add_f32 v240, v240, v48 :: v_dual_add_f32 v241, v241, v49
	v_dual_add_f32 v242, v242, v58 :: v_dual_add_f32 v243, v243, v59
	s_wait_loadcnt 0x29
	v_and_b32_e32 v46, 0xf0f0f0f, v28
	s_wait_loadcnt 0x28
	v_and_b32_e32 v56, 0xf0f0f0f, v29
	s_wait_loadcnt 0x27
	v_and_b32_e32 v66, 0xf0f0f0f, v30
	s_wait_loadcnt 0x26
	v_and_b32_e32 v76, 0xf0f0f0f, v31
	v_lshrrev_b32_e32 v47, 4, v28
	v_lshrrev_b32_e32 v57, 4, v29
	v_lshrrev_b32_e32 v67, 4, v30
	v_lshrrev_b32_e32 v77, 4, v31
	v_and_b32_e32 v47, 0xf0f0f0f, v47
	v_and_b32_e32 v57, 0xf0f0f0f, v57
	v_and_b32_e32 v67, 0xf0f0f0f, v67
	v_and_b32_e32 v77, 0xf0f0f0f, v77
	v_cvt_f32_ubyte0_e32 v40, v46
	v_cvt_f32_ubyte0_e32 v50, v56
	v_cvt_f32_ubyte0_e32 v60, v66
	v_cvt_f32_ubyte0_e32 v70, v76
	v_cvt_f32_ubyte1_e32 v42, v46
	v_cvt_f32_ubyte1_e32 v52, v56
	v_cvt_f32_ubyte1_e32 v62, v66
	v_cvt_f32_ubyte1_e32 v72, v76
	v_cvt_f32_ubyte2_e32 v44, v46
	v_cvt_f32_ubyte2_e32 v54, v56
	v_cvt_f32_ubyte2_e32 v64, v66
	v_cvt_f32_ubyte2_e32 v74, v76
	v_cvt_f32_ubyte3_e32 v46, v46
	v_cvt_f32_ubyte3_e32 v56, v56
	v_cvt_f32_ubyte3_e32 v66, v66
	v_cvt_f32_ubyte3_e32 v76, v76
	v_cvt_f32_ubyte0_e32 v41, v47
	v_cvt_f32_ubyte0_e32 v51, v57
	v_cvt_f32_ubyte0_e32 v61, v67
	v_cvt_f32_ubyte0_e32 v71, v77
	v_cvt_f32_ubyte1_e32 v43, v47
	v_cvt_f32_ubyte1_e32 v53, v57
	v_cvt_f32_ubyte1_e32 v63, v67
	v_cvt_f32_ubyte1_e32 v73, v77
	v_cvt_f32_ubyte2_e32 v45, v47
	v_cvt_f32_ubyte2_e32 v55, v57
	v_cvt_f32_ubyte2_e32 v65, v67
	v_cvt_f32_ubyte2_e32 v75, v77
	v_cvt_f32_ubyte3_e32 v47, v47
	v_cvt_f32_ubyte3_e32 v57, v57
	v_cvt_f32_ubyte3_e32 v67, v67
	v_cvt_f32_ubyte3_e32 v77, v77
	v_fma_mix_f32 v40, v12, v40, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v13, v50, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v14, v60, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v70, v15, v70, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v41, v12, v41, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v13, v51, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v14, v61, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v71, v15, v71, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v42, v12, v42, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v52, v13, v52, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v62, v14, v62, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v72, v15, v72, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v43, v12, v43, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v53, v13, v53, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v63, v14, v63, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v73, v15, v73, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v44, v12, v44, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v54, v13, v54, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v64, v14, v64, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v74, v15, v74, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v12, v45, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v13, v55, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v65, v14, v65, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v75, v15, v75, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v12, v46, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v13, v56, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v66, v14, v66, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v76, v15, v76, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v12, v47, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v13, v57, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v67, v14, v67, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v77, v15, v77, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	buffer_load_b128 v[112:115], v3, s[24:27], s29 offen offset:3072
	buffer_load_b128 v[116:119], v3, s[24:27], s29 offen offset:3088
	s_wait_loadcnt 0x17
	v_dual_mul_f32 v48, v41, v121 :: v_dual_mul_f32 v49, v51, v121
	v_dual_mul_f32 v58, v61, v121 :: v_dual_mul_f32 v59, v71, v121
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v68, v41, v129 :: v_dual_mul_f32 v69, v51, v129
	v_dual_mul_f32 v78, v61, v129 :: v_dual_mul_f32 v79, v71, v129
	v_dual_fmac_f32 v48, v40, v120 :: v_dual_fmac_f32 v49, v50, v120
	v_dual_fmac_f32 v58, v60, v120 :: v_dual_fmac_f32 v59, v70, v120
	v_dual_fmac_f32 v68, v40, v128 :: v_dual_fmac_f32 v69, v50, v128
	v_dual_fmac_f32 v78, v60, v128 :: v_dual_fmac_f32 v79, v70, v128
	v_dual_fmac_f32 v48, v42, v122 :: v_dual_fmac_f32 v49, v52, v122
	v_dual_fmac_f32 v58, v62, v122 :: v_dual_fmac_f32 v59, v72, v122
	v_dual_fmac_f32 v68, v42, v130 :: v_dual_fmac_f32 v69, v52, v130
	v_dual_fmac_f32 v78, v62, v130 :: v_dual_fmac_f32 v79, v72, v130
	v_dual_fmac_f32 v48, v43, v123 :: v_dual_fmac_f32 v49, v53, v123
	v_dual_fmac_f32 v58, v63, v123 :: v_dual_fmac_f32 v59, v73, v123
	v_dual_fmac_f32 v68, v43, v131 :: v_dual_fmac_f32 v69, v53, v131
	v_dual_fmac_f32 v78, v63, v131 :: v_dual_fmac_f32 v79, v73, v131
	v_dual_fmac_f32 v48, v44, v124 :: v_dual_fmac_f32 v49, v54, v124
	v_dual_fmac_f32 v58, v64, v124 :: v_dual_fmac_f32 v59, v74, v124
	s_wait_loadcnt 0x14
	v_dual_fmac_f32 v68, v44, v132 :: v_dual_fmac_f32 v69, v54, v132
	v_dual_fmac_f32 v78, v64, v132 :: v_dual_fmac_f32 v79, v74, v132
	v_dual_fmac_f32 v48, v45, v125 :: v_dual_fmac_f32 v49, v55, v125
	v_dual_fmac_f32 v58, v65, v125 :: v_dual_fmac_f32 v59, v75, v125
	v_dual_fmac_f32 v68, v45, v133 :: v_dual_fmac_f32 v69, v55, v133
	v_dual_fmac_f32 v78, v65, v133 :: v_dual_fmac_f32 v79, v75, v133
	v_dual_fmac_f32 v48, v46, v126 :: v_dual_fmac_f32 v49, v56, v126
	v_dual_fmac_f32 v58, v66, v126 :: v_dual_fmac_f32 v59, v76, v126
	v_dual_fmac_f32 v68, v46, v134 :: v_dual_fmac_f32 v69, v56, v134
	v_dual_fmac_f32 v78, v66, v134 :: v_dual_fmac_f32 v79, v76, v134
	v_dual_fmac_f32 v48, v47, v127 :: v_dual_fmac_f32 v49, v57, v127
	v_dual_fmac_f32 v58, v67, v127 :: v_dual_fmac_f32 v59, v77, v127
	v_dual_fmac_f32 v68, v47, v135 :: v_dual_fmac_f32 v69, v57, v135
	v_dual_fmac_f32 v78, v67, v135 :: v_dual_fmac_f32 v79, v77, v135
	v_dual_add_f32 v180, v180, v48 :: v_dual_add_f32 v181, v181, v49
	v_dual_add_f32 v182, v182, v58 :: v_dual_add_f32 v183, v183, v59
	v_dual_add_f32 v196, v196, v68 :: v_dual_add_f32 v197, v197, v69
	v_dual_add_f32 v198, v198, v78 :: v_dual_add_f32 v199, v199, v79
	buffer_load_b128 v[120:123], v3, s[24:27], s30 offen offset:3072
	buffer_load_b128 v[124:127], v3, s[24:27], s30 offen offset:3088
	buffer_load_b128 v[128:131], v3, s[24:27], s31 offen offset:3072
	buffer_load_b128 v[132:135], v3, s[24:27], s31 offen offset:3088
	s_wait_loadcnt 0x17
	v_dual_mul_f32 v48, v41, v137 :: v_dual_mul_f32 v49, v51, v137
	v_dual_mul_f32 v58, v61, v137 :: v_dual_mul_f32 v59, v71, v137
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v68, v41, v145 :: v_dual_mul_f32 v69, v51, v145
	v_dual_mul_f32 v78, v61, v145 :: v_dual_mul_f32 v79, v71, v145
	v_dual_fmac_f32 v48, v40, v136 :: v_dual_fmac_f32 v49, v50, v136
	v_dual_fmac_f32 v58, v60, v136 :: v_dual_fmac_f32 v59, v70, v136
	v_dual_fmac_f32 v68, v40, v144 :: v_dual_fmac_f32 v69, v50, v144
	v_dual_fmac_f32 v78, v60, v144 :: v_dual_fmac_f32 v79, v70, v144
	v_dual_fmac_f32 v48, v42, v138 :: v_dual_fmac_f32 v49, v52, v138
	v_dual_fmac_f32 v58, v62, v138 :: v_dual_fmac_f32 v59, v72, v138
	v_dual_fmac_f32 v68, v42, v146 :: v_dual_fmac_f32 v69, v52, v146
	v_dual_fmac_f32 v78, v62, v146 :: v_dual_fmac_f32 v79, v72, v146
	v_dual_fmac_f32 v48, v43, v139 :: v_dual_fmac_f32 v49, v53, v139
	v_dual_fmac_f32 v58, v63, v139 :: v_dual_fmac_f32 v59, v73, v139
	v_dual_fmac_f32 v68, v43, v147 :: v_dual_fmac_f32 v69, v53, v147
	v_dual_fmac_f32 v78, v63, v147 :: v_dual_fmac_f32 v79, v73, v147
	v_dual_fmac_f32 v48, v44, v140 :: v_dual_fmac_f32 v49, v54, v140
	v_dual_fmac_f32 v58, v64, v140 :: v_dual_fmac_f32 v59, v74, v140
	s_wait_loadcnt 0x14
	v_dual_fmac_f32 v68, v44, v148 :: v_dual_fmac_f32 v69, v54, v148
	v_dual_fmac_f32 v78, v64, v148 :: v_dual_fmac_f32 v79, v74, v148
	v_dual_fmac_f32 v48, v45, v141 :: v_dual_fmac_f32 v49, v55, v141
	v_dual_fmac_f32 v58, v65, v141 :: v_dual_fmac_f32 v59, v75, v141
	v_dual_fmac_f32 v68, v45, v149 :: v_dual_fmac_f32 v69, v55, v149
	v_dual_fmac_f32 v78, v65, v149 :: v_dual_fmac_f32 v79, v75, v149
	v_dual_fmac_f32 v48, v46, v142 :: v_dual_fmac_f32 v49, v56, v142
	v_dual_fmac_f32 v58, v66, v142 :: v_dual_fmac_f32 v59, v76, v142
	v_dual_fmac_f32 v68, v46, v150 :: v_dual_fmac_f32 v69, v56, v150
	v_dual_fmac_f32 v78, v66, v150 :: v_dual_fmac_f32 v79, v76, v150
	v_dual_fmac_f32 v48, v47, v143 :: v_dual_fmac_f32 v49, v57, v143
	v_dual_fmac_f32 v58, v67, v143 :: v_dual_fmac_f32 v59, v77, v143
	v_dual_fmac_f32 v68, v47, v151 :: v_dual_fmac_f32 v69, v57, v151
	v_dual_fmac_f32 v78, v67, v151 :: v_dual_fmac_f32 v79, v77, v151
	v_dual_add_f32 v212, v212, v48 :: v_dual_add_f32 v213, v213, v49
	v_dual_add_f32 v214, v214, v58 :: v_dual_add_f32 v215, v215, v59
	v_dual_add_f32 v228, v228, v68 :: v_dual_add_f32 v229, v229, v69
	v_dual_add_f32 v230, v230, v78 :: v_dual_add_f32 v231, v231, v79
	buffer_load_b128 v[136:139], v3, s[24:27], s32 offen offset:3072
	buffer_load_b128 v[140:143], v3, s[24:27], s32 offen offset:3088
	s_wait_loadcnt 0x15
	v_dual_mul_f32 v48, v41, v153 :: v_dual_mul_f32 v49, v51, v153
	v_dual_mul_f32 v58, v61, v153 :: v_dual_mul_f32 v59, v71, v153
	v_dual_fmac_f32 v48, v40, v152 :: v_dual_fmac_f32 v49, v50, v152
	v_dual_fmac_f32 v58, v60, v152 :: v_dual_fmac_f32 v59, v70, v152
	v_dual_fmac_f32 v48, v42, v154 :: v_dual_fmac_f32 v49, v52, v154
	v_dual_fmac_f32 v58, v62, v154 :: v_dual_fmac_f32 v59, v72, v154
	v_dual_fmac_f32 v48, v43, v155 :: v_dual_fmac_f32 v49, v53, v155
	v_dual_fmac_f32 v58, v63, v155 :: v_dual_fmac_f32 v59, v73, v155
	s_wait_loadcnt 0x14
	v_dual_fmac_f32 v48, v44, v156 :: v_dual_fmac_f32 v49, v54, v156
	v_dual_fmac_f32 v58, v64, v156 :: v_dual_fmac_f32 v59, v74, v156
	v_dual_fmac_f32 v48, v45, v157 :: v_dual_fmac_f32 v49, v55, v157
	v_dual_fmac_f32 v58, v65, v157 :: v_dual_fmac_f32 v59, v75, v157
	v_dual_fmac_f32 v48, v46, v158 :: v_dual_fmac_f32 v49, v56, v158
	v_dual_fmac_f32 v58, v66, v158 :: v_dual_fmac_f32 v59, v76, v158
	v_dual_fmac_f32 v48, v47, v159 :: v_dual_fmac_f32 v49, v57, v159
	v_dual_fmac_f32 v58, v67, v159 :: v_dual_fmac_f32 v59, v77, v159
	v_dual_add_f32 v244, v244, v48 :: v_dual_add_f32 v245, v245, v49
	v_dual_add_f32 v246, v246, v58 :: v_dual_add_f32 v247, v247, v59
	v_and_b32_e32 v46, 0xf0f0f0f, v32
	v_and_b32_e32 v56, 0xf0f0f0f, v33
	v_and_b32_e32 v66, 0xf0f0f0f, v34
	v_and_b32_e32 v76, 0xf0f0f0f, v35
	v_lshrrev_b32_e32 v47, 4, v32
	v_lshrrev_b32_e32 v57, 4, v33
	v_lshrrev_b32_e32 v67, 4, v34
	v_lshrrev_b32_e32 v77, 4, v35
	v_and_b32_e32 v47, 0xf0f0f0f, v47
	v_and_b32_e32 v57, 0xf0f0f0f, v57
	v_and_b32_e32 v67, 0xf0f0f0f, v67
	v_and_b32_e32 v77, 0xf0f0f0f, v77
	v_cvt_f32_ubyte0_e32 v40, v46
	v_cvt_f32_ubyte0_e32 v50, v56
	v_cvt_f32_ubyte0_e32 v60, v66
	v_cvt_f32_ubyte0_e32 v70, v76
	v_cvt_f32_ubyte1_e32 v42, v46
	v_cvt_f32_ubyte1_e32 v52, v56
	v_cvt_f32_ubyte1_e32 v62, v66
	v_cvt_f32_ubyte1_e32 v72, v76
	v_cvt_f32_ubyte2_e32 v44, v46
	v_cvt_f32_ubyte2_e32 v54, v56
	v_cvt_f32_ubyte2_e32 v64, v66
	v_cvt_f32_ubyte2_e32 v74, v76
	v_cvt_f32_ubyte3_e32 v46, v46
	v_cvt_f32_ubyte3_e32 v56, v56
	v_cvt_f32_ubyte3_e32 v66, v66
	v_cvt_f32_ubyte3_e32 v76, v76
	v_cvt_f32_ubyte0_e32 v41, v47
	v_cvt_f32_ubyte0_e32 v51, v57
	v_cvt_f32_ubyte0_e32 v61, v67
	v_cvt_f32_ubyte0_e32 v71, v77
	v_cvt_f32_ubyte1_e32 v43, v47
	v_cvt_f32_ubyte1_e32 v53, v57
	v_cvt_f32_ubyte1_e32 v63, v67
	v_cvt_f32_ubyte1_e32 v73, v77
	v_cvt_f32_ubyte2_e32 v45, v47
	v_cvt_f32_ubyte2_e32 v55, v57
	v_cvt_f32_ubyte2_e32 v65, v67
	v_cvt_f32_ubyte2_e32 v75, v77
	v_cvt_f32_ubyte3_e32 v47, v47
	v_cvt_f32_ubyte3_e32 v57, v57
	v_cvt_f32_ubyte3_e32 v67, v67
	v_cvt_f32_ubyte3_e32 v77, v77
	v_fma_mix_f32 v40, v16, v40, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v17, v50, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v18, v60, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v70, v19, v70, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v41, v16, v41, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v17, v51, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v18, v61, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v71, v19, v71, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v42, v16, v42, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v52, v17, v52, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v62, v18, v62, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v72, v19, v72, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v43, v16, v43, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v53, v17, v53, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v63, v18, v63, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v73, v19, v73, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v44, v16, v44, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v54, v17, v54, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v64, v18, v64, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v74, v19, v74, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v16, v45, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v17, v55, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v65, v18, v65, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v75, v19, v75, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v16, v46, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v17, v56, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v66, v18, v66, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v76, v19, v76, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v16, v47, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v17, v57, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v67, v18, v67, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v77, v19, v77, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x13
	v_dual_mul_f32 v48, v41, v161 :: v_dual_mul_f32 v49, v51, v161
	v_dual_mul_f32 v58, v61, v161 :: v_dual_mul_f32 v59, v71, v161
	s_wait_loadcnt 0x11
	v_dual_mul_f32 v68, v41, v169 :: v_dual_mul_f32 v69, v51, v169
	v_dual_mul_f32 v78, v61, v169 :: v_dual_mul_f32 v79, v71, v169
	v_dual_fmac_f32 v48, v40, v160 :: v_dual_fmac_f32 v49, v50, v160
	v_dual_fmac_f32 v58, v60, v160 :: v_dual_fmac_f32 v59, v70, v160
	v_dual_fmac_f32 v68, v40, v168 :: v_dual_fmac_f32 v69, v50, v168
	v_dual_fmac_f32 v78, v60, v168 :: v_dual_fmac_f32 v79, v70, v168
	v_dual_fmac_f32 v48, v42, v162 :: v_dual_fmac_f32 v49, v52, v162
	v_dual_fmac_f32 v58, v62, v162 :: v_dual_fmac_f32 v59, v72, v162
	v_dual_fmac_f32 v68, v42, v170 :: v_dual_fmac_f32 v69, v52, v170
	v_dual_fmac_f32 v78, v62, v170 :: v_dual_fmac_f32 v79, v72, v170
	v_dual_fmac_f32 v48, v43, v163 :: v_dual_fmac_f32 v49, v53, v163
	v_dual_fmac_f32 v58, v63, v163 :: v_dual_fmac_f32 v59, v73, v163
	v_dual_fmac_f32 v68, v43, v171 :: v_dual_fmac_f32 v69, v53, v171
	v_dual_fmac_f32 v78, v63, v171 :: v_dual_fmac_f32 v79, v73, v171
	v_dual_fmac_f32 v48, v44, v164 :: v_dual_fmac_f32 v49, v54, v164
	v_dual_fmac_f32 v58, v64, v164 :: v_dual_fmac_f32 v59, v74, v164
	s_wait_loadcnt 0x10
	v_dual_fmac_f32 v68, v44, v172 :: v_dual_fmac_f32 v69, v54, v172
	v_dual_fmac_f32 v78, v64, v172 :: v_dual_fmac_f32 v79, v74, v172
	v_dual_fmac_f32 v48, v45, v165 :: v_dual_fmac_f32 v49, v55, v165
	v_dual_fmac_f32 v58, v65, v165 :: v_dual_fmac_f32 v59, v75, v165
	v_dual_fmac_f32 v68, v45, v173 :: v_dual_fmac_f32 v69, v55, v173
	v_dual_fmac_f32 v78, v65, v173 :: v_dual_fmac_f32 v79, v75, v173
	v_dual_fmac_f32 v48, v46, v166 :: v_dual_fmac_f32 v49, v56, v166
	v_dual_fmac_f32 v58, v66, v166 :: v_dual_fmac_f32 v59, v76, v166
	v_dual_fmac_f32 v68, v46, v174 :: v_dual_fmac_f32 v69, v56, v174
	v_dual_fmac_f32 v78, v66, v174 :: v_dual_fmac_f32 v79, v76, v174
	v_dual_fmac_f32 v48, v47, v167 :: v_dual_fmac_f32 v49, v57, v167
	v_dual_fmac_f32 v58, v67, v167 :: v_dual_fmac_f32 v59, v77, v167
	v_dual_fmac_f32 v68, v47, v175 :: v_dual_fmac_f32 v69, v57, v175
	v_dual_fmac_f32 v78, v67, v175 :: v_dual_fmac_f32 v79, v77, v175
	v_dual_add_f32 v184, v184, v48 :: v_dual_add_f32 v185, v185, v49
	v_dual_add_f32 v186, v186, v58 :: v_dual_add_f32 v187, v187, v59
	v_dual_add_f32 v200, v200, v68 :: v_dual_add_f32 v201, v201, v69
	v_dual_add_f32 v202, v202, v78 :: v_dual_add_f32 v203, v203, v79
	s_wait_loadcnt 0xf
	v_dual_mul_f32 v48, v41, v81 :: v_dual_mul_f32 v49, v51, v81
	v_dual_mul_f32 v58, v61, v81 :: v_dual_mul_f32 v59, v71, v81
	s_wait_loadcnt 0xd
	v_dual_mul_f32 v68, v41, v89 :: v_dual_mul_f32 v69, v51, v89
	v_dual_mul_f32 v78, v61, v89 :: v_dual_mul_f32 v79, v71, v89
	v_dual_fmac_f32 v48, v40, v80 :: v_dual_fmac_f32 v49, v50, v80
	v_dual_fmac_f32 v58, v60, v80 :: v_dual_fmac_f32 v59, v70, v80
	v_dual_fmac_f32 v68, v40, v88 :: v_dual_fmac_f32 v69, v50, v88
	v_dual_fmac_f32 v78, v60, v88 :: v_dual_fmac_f32 v79, v70, v88
	v_dual_fmac_f32 v48, v42, v82 :: v_dual_fmac_f32 v49, v52, v82
	v_dual_fmac_f32 v58, v62, v82 :: v_dual_fmac_f32 v59, v72, v82
	v_dual_fmac_f32 v68, v42, v90 :: v_dual_fmac_f32 v69, v52, v90
	v_dual_fmac_f32 v78, v62, v90 :: v_dual_fmac_f32 v79, v72, v90
	v_dual_fmac_f32 v48, v43, v83 :: v_dual_fmac_f32 v49, v53, v83
	v_dual_fmac_f32 v58, v63, v83 :: v_dual_fmac_f32 v59, v73, v83
	v_dual_fmac_f32 v68, v43, v91 :: v_dual_fmac_f32 v69, v53, v91
	v_dual_fmac_f32 v78, v63, v91 :: v_dual_fmac_f32 v79, v73, v91
	v_dual_fmac_f32 v48, v44, v84 :: v_dual_fmac_f32 v49, v54, v84
	v_dual_fmac_f32 v58, v64, v84 :: v_dual_fmac_f32 v59, v74, v84
	s_wait_loadcnt 0xc
	v_dual_fmac_f32 v68, v44, v92 :: v_dual_fmac_f32 v69, v54, v92
	v_dual_fmac_f32 v78, v64, v92 :: v_dual_fmac_f32 v79, v74, v92
	v_dual_fmac_f32 v48, v45, v85 :: v_dual_fmac_f32 v49, v55, v85
	v_dual_fmac_f32 v58, v65, v85 :: v_dual_fmac_f32 v59, v75, v85
	v_dual_fmac_f32 v68, v45, v93 :: v_dual_fmac_f32 v69, v55, v93
	v_dual_fmac_f32 v78, v65, v93 :: v_dual_fmac_f32 v79, v75, v93
	v_dual_fmac_f32 v48, v46, v86 :: v_dual_fmac_f32 v49, v56, v86
	v_dual_fmac_f32 v58, v66, v86 :: v_dual_fmac_f32 v59, v76, v86
	v_dual_fmac_f32 v68, v46, v94 :: v_dual_fmac_f32 v69, v56, v94
	v_dual_fmac_f32 v78, v66, v94 :: v_dual_fmac_f32 v79, v76, v94
	v_dual_fmac_f32 v48, v47, v87 :: v_dual_fmac_f32 v49, v57, v87
	v_dual_fmac_f32 v58, v67, v87 :: v_dual_fmac_f32 v59, v77, v87
	v_dual_fmac_f32 v68, v47, v95 :: v_dual_fmac_f32 v69, v57, v95
	v_dual_fmac_f32 v78, v67, v95 :: v_dual_fmac_f32 v79, v77, v95
	v_dual_add_f32 v216, v216, v48 :: v_dual_add_f32 v217, v217, v49
	v_dual_add_f32 v218, v218, v58 :: v_dual_add_f32 v219, v219, v59
	v_dual_add_f32 v232, v232, v68 :: v_dual_add_f32 v233, v233, v69
	v_dual_add_f32 v234, v234, v78 :: v_dual_add_f32 v235, v235, v79
	s_wait_loadcnt 0xb
	v_dual_mul_f32 v48, v41, v97 :: v_dual_mul_f32 v49, v51, v97
	v_dual_mul_f32 v58, v61, v97 :: v_dual_mul_f32 v59, v71, v97
	v_dual_fmac_f32 v48, v40, v96 :: v_dual_fmac_f32 v49, v50, v96
	v_dual_fmac_f32 v58, v60, v96 :: v_dual_fmac_f32 v59, v70, v96
	v_dual_fmac_f32 v48, v42, v98 :: v_dual_fmac_f32 v49, v52, v98
	v_dual_fmac_f32 v58, v62, v98 :: v_dual_fmac_f32 v59, v72, v98
	v_dual_fmac_f32 v48, v43, v99 :: v_dual_fmac_f32 v49, v53, v99
	v_dual_fmac_f32 v58, v63, v99 :: v_dual_fmac_f32 v59, v73, v99
	s_wait_loadcnt 0xa
	v_dual_fmac_f32 v48, v44, v100 :: v_dual_fmac_f32 v49, v54, v100
	v_dual_fmac_f32 v58, v64, v100 :: v_dual_fmac_f32 v59, v74, v100
	v_dual_fmac_f32 v48, v45, v101 :: v_dual_fmac_f32 v49, v55, v101
	v_dual_fmac_f32 v58, v65, v101 :: v_dual_fmac_f32 v59, v75, v101
	v_dual_fmac_f32 v48, v46, v102 :: v_dual_fmac_f32 v49, v56, v102
	v_dual_fmac_f32 v58, v66, v102 :: v_dual_fmac_f32 v59, v76, v102
	v_dual_fmac_f32 v48, v47, v103 :: v_dual_fmac_f32 v49, v57, v103
	v_dual_fmac_f32 v58, v67, v103 :: v_dual_fmac_f32 v59, v77, v103
	v_dual_add_f32 v248, v248, v48 :: v_dual_add_f32 v249, v249, v49
	v_dual_add_f32 v250, v250, v58 :: v_dual_add_f32 v251, v251, v59
	v_and_b32_e32 v46, 0xf0f0f0f, v36
	v_and_b32_e32 v56, 0xf0f0f0f, v37
	v_and_b32_e32 v66, 0xf0f0f0f, v38
	v_and_b32_e32 v76, 0xf0f0f0f, v39
	v_lshrrev_b32_e32 v47, 4, v36
	v_lshrrev_b32_e32 v57, 4, v37
	v_lshrrev_b32_e32 v67, 4, v38
	v_lshrrev_b32_e32 v77, 4, v39
	v_and_b32_e32 v47, 0xf0f0f0f, v47
	v_and_b32_e32 v57, 0xf0f0f0f, v57
	v_and_b32_e32 v67, 0xf0f0f0f, v67
	v_and_b32_e32 v77, 0xf0f0f0f, v77
	v_cvt_f32_ubyte0_e32 v40, v46
	v_cvt_f32_ubyte0_e32 v50, v56
	v_cvt_f32_ubyte0_e32 v60, v66
	v_cvt_f32_ubyte0_e32 v70, v76
	v_cvt_f32_ubyte1_e32 v42, v46
	v_cvt_f32_ubyte1_e32 v52, v56
	v_cvt_f32_ubyte1_e32 v62, v66
	v_cvt_f32_ubyte1_e32 v72, v76
	v_cvt_f32_ubyte2_e32 v44, v46
	v_cvt_f32_ubyte2_e32 v54, v56
	v_cvt_f32_ubyte2_e32 v64, v66
	v_cvt_f32_ubyte2_e32 v74, v76
	v_cvt_f32_ubyte3_e32 v46, v46
	v_cvt_f32_ubyte3_e32 v56, v56
	v_cvt_f32_ubyte3_e32 v66, v66
	v_cvt_f32_ubyte3_e32 v76, v76
	v_cvt_f32_ubyte0_e32 v41, v47
	v_cvt_f32_ubyte0_e32 v51, v57
	v_cvt_f32_ubyte0_e32 v61, v67
	v_cvt_f32_ubyte0_e32 v71, v77
	v_cvt_f32_ubyte1_e32 v43, v47
	v_cvt_f32_ubyte1_e32 v53, v57
	v_cvt_f32_ubyte1_e32 v63, v67
	v_cvt_f32_ubyte1_e32 v73, v77
	v_cvt_f32_ubyte2_e32 v45, v47
	v_cvt_f32_ubyte2_e32 v55, v57
	v_cvt_f32_ubyte2_e32 v65, v67
	v_cvt_f32_ubyte2_e32 v75, v77
	v_cvt_f32_ubyte3_e32 v47, v47
	v_cvt_f32_ubyte3_e32 v57, v57
	v_cvt_f32_ubyte3_e32 v67, v67
	v_cvt_f32_ubyte3_e32 v77, v77
	v_fma_mix_f32 v40, v20, v40, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v21, v50, v21 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v22, v60, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v70, v23, v70, v23 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v41, v20, v41, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v21, v51, v21 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v22, v61, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v71, v23, v71, v23 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v42, v20, v42, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v52, v21, v52, v21 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v62, v22, v62, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v72, v23, v72, v23 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v43, v20, v43, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v53, v21, v53, v21 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v63, v22, v63, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v73, v23, v73, v23 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v44, v20, v44, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v54, v21, v54, v21 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v64, v22, v64, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v74, v23, v74, v23 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v20, v45, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v21, v55, v21 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v65, v22, v65, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v75, v23, v75, v23 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v20, v46, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v21, v56, v21 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v66, v22, v66, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v76, v23, v76, v23 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v20, v47, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v21, v57, v21 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v67, v22, v67, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v77, v23, v77, v23 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x9
	v_dual_mul_f32 v48, v41, v105 :: v_dual_mul_f32 v49, v51, v105
	v_dual_mul_f32 v58, v61, v105 :: v_dual_mul_f32 v59, v71, v105
	s_wait_loadcnt 0x7
	v_dual_mul_f32 v68, v41, v113 :: v_dual_mul_f32 v69, v51, v113
	v_dual_mul_f32 v78, v61, v113 :: v_dual_mul_f32 v79, v71, v113
	v_dual_fmac_f32 v48, v40, v104 :: v_dual_fmac_f32 v49, v50, v104
	v_dual_fmac_f32 v58, v60, v104 :: v_dual_fmac_f32 v59, v70, v104
	v_dual_fmac_f32 v68, v40, v112 :: v_dual_fmac_f32 v69, v50, v112
	v_dual_fmac_f32 v78, v60, v112 :: v_dual_fmac_f32 v79, v70, v112
	v_dual_fmac_f32 v48, v42, v106 :: v_dual_fmac_f32 v49, v52, v106
	v_dual_fmac_f32 v58, v62, v106 :: v_dual_fmac_f32 v59, v72, v106
	v_dual_fmac_f32 v68, v42, v114 :: v_dual_fmac_f32 v69, v52, v114
	v_dual_fmac_f32 v78, v62, v114 :: v_dual_fmac_f32 v79, v72, v114
	v_dual_fmac_f32 v48, v43, v107 :: v_dual_fmac_f32 v49, v53, v107
	v_dual_fmac_f32 v58, v63, v107 :: v_dual_fmac_f32 v59, v73, v107
	v_dual_fmac_f32 v68, v43, v115 :: v_dual_fmac_f32 v69, v53, v115
	v_dual_fmac_f32 v78, v63, v115 :: v_dual_fmac_f32 v79, v73, v115
	v_dual_fmac_f32 v48, v44, v108 :: v_dual_fmac_f32 v49, v54, v108
	v_dual_fmac_f32 v58, v64, v108 :: v_dual_fmac_f32 v59, v74, v108
	s_wait_loadcnt 0x6
	v_dual_fmac_f32 v68, v44, v116 :: v_dual_fmac_f32 v69, v54, v116
	v_dual_fmac_f32 v78, v64, v116 :: v_dual_fmac_f32 v79, v74, v116
	v_dual_fmac_f32 v48, v45, v109 :: v_dual_fmac_f32 v49, v55, v109
	v_dual_fmac_f32 v58, v65, v109 :: v_dual_fmac_f32 v59, v75, v109
	v_dual_fmac_f32 v68, v45, v117 :: v_dual_fmac_f32 v69, v55, v117
	v_dual_fmac_f32 v78, v65, v117 :: v_dual_fmac_f32 v79, v75, v117
	v_dual_fmac_f32 v48, v46, v110 :: v_dual_fmac_f32 v49, v56, v110
	v_dual_fmac_f32 v58, v66, v110 :: v_dual_fmac_f32 v59, v76, v110
	v_dual_fmac_f32 v68, v46, v118 :: v_dual_fmac_f32 v69, v56, v118
	v_dual_fmac_f32 v78, v66, v118 :: v_dual_fmac_f32 v79, v76, v118
	v_dual_fmac_f32 v48, v47, v111 :: v_dual_fmac_f32 v49, v57, v111
	v_dual_fmac_f32 v58, v67, v111 :: v_dual_fmac_f32 v59, v77, v111
	v_dual_fmac_f32 v68, v47, v119 :: v_dual_fmac_f32 v69, v57, v119
	v_dual_fmac_f32 v78, v67, v119 :: v_dual_fmac_f32 v79, v77, v119
	v_dual_add_f32 v188, v188, v48 :: v_dual_add_f32 v189, v189, v49
	v_dual_add_f32 v190, v190, v58 :: v_dual_add_f32 v191, v191, v59
	v_dual_add_f32 v204, v204, v68 :: v_dual_add_f32 v205, v205, v69
	v_dual_add_f32 v206, v206, v78 :: v_dual_add_f32 v207, v207, v79
	s_wait_loadcnt 0x5
	v_dual_mul_f32 v48, v41, v121 :: v_dual_mul_f32 v49, v51, v121
	v_dual_mul_f32 v58, v61, v121 :: v_dual_mul_f32 v59, v71, v121
	s_wait_loadcnt 0x3
	v_dual_mul_f32 v68, v41, v129 :: v_dual_mul_f32 v69, v51, v129
	v_dual_mul_f32 v78, v61, v129 :: v_dual_mul_f32 v79, v71, v129
	v_dual_fmac_f32 v48, v40, v120 :: v_dual_fmac_f32 v49, v50, v120
	v_dual_fmac_f32 v58, v60, v120 :: v_dual_fmac_f32 v59, v70, v120
	v_dual_fmac_f32 v68, v40, v128 :: v_dual_fmac_f32 v69, v50, v128
	v_dual_fmac_f32 v78, v60, v128 :: v_dual_fmac_f32 v79, v70, v128
	v_dual_fmac_f32 v48, v42, v122 :: v_dual_fmac_f32 v49, v52, v122
	v_dual_fmac_f32 v58, v62, v122 :: v_dual_fmac_f32 v59, v72, v122
	v_dual_fmac_f32 v68, v42, v130 :: v_dual_fmac_f32 v69, v52, v130
	v_dual_fmac_f32 v78, v62, v130 :: v_dual_fmac_f32 v79, v72, v130
	v_dual_fmac_f32 v48, v43, v123 :: v_dual_fmac_f32 v49, v53, v123
	v_dual_fmac_f32 v58, v63, v123 :: v_dual_fmac_f32 v59, v73, v123
	v_dual_fmac_f32 v68, v43, v131 :: v_dual_fmac_f32 v69, v53, v131
	v_dual_fmac_f32 v78, v63, v131 :: v_dual_fmac_f32 v79, v73, v131
	v_dual_fmac_f32 v48, v44, v124 :: v_dual_fmac_f32 v49, v54, v124
	v_dual_fmac_f32 v58, v64, v124 :: v_dual_fmac_f32 v59, v74, v124
	s_wait_loadcnt 0x2
	v_dual_fmac_f32 v68, v44, v132 :: v_dual_fmac_f32 v69, v54, v132
	v_dual_fmac_f32 v78, v64, v132 :: v_dual_fmac_f32 v79, v74, v132
	v_dual_fmac_f32 v48, v45, v125 :: v_dual_fmac_f32 v49, v55, v125
	v_dual_fmac_f32 v58, v65, v125 :: v_dual_fmac_f32 v59, v75, v125
	v_dual_fmac_f32 v68, v45, v133 :: v_dual_fmac_f32 v69, v55, v133
	v_dual_fmac_f32 v78, v65, v133 :: v_dual_fmac_f32 v79, v75, v133
	v_dual_fmac_f32 v48, v46, v126 :: v_dual_fmac_f32 v49, v56, v126
	v_dual_fmac_f32 v58, v66, v126 :: v_dual_fmac_f32 v59, v76, v126
	v_dual_fmac_f32 v68, v46, v134 :: v_dual_fmac_f32 v69, v56, v134
	v_dual_fmac_f32 v78, v66, v134 :: v_dual_fmac_f32 v79, v76, v134
	v_dual_fmac_f32 v48, v47, v127 :: v_dual_fmac_f32 v49, v57, v127
	v_dual_fmac_f32 v58, v67, v127 :: v_dual_fmac_f32 v59, v77, v127
	v_dual_fmac_f32 v68, v47, v135 :: v_dual_fmac_f32 v69, v57, v135
	v_dual_fmac_f32 v78, v67, v135 :: v_dual_fmac_f32 v79, v77, v135
	v_dual_add_f32 v220, v220, v48 :: v_dual_add_f32 v221, v221, v49
	v_dual_add_f32 v222, v222, v58 :: v_dual_add_f32 v223, v223, v59
	v_dual_add_f32 v236, v236, v68 :: v_dual_add_f32 v237, v237, v69
	v_dual_add_f32 v238, v238, v78 :: v_dual_add_f32 v239, v239, v79
	s_wait_loadcnt 0x1
	v_dual_mul_f32 v48, v41, v137 :: v_dual_mul_f32 v49, v51, v137
	v_dual_mul_f32 v58, v61, v137 :: v_dual_mul_f32 v59, v71, v137
	v_dual_fmac_f32 v48, v40, v136 :: v_dual_fmac_f32 v49, v50, v136
	v_dual_fmac_f32 v58, v60, v136 :: v_dual_fmac_f32 v59, v70, v136
	v_dual_fmac_f32 v48, v42, v138 :: v_dual_fmac_f32 v49, v52, v138
	v_dual_fmac_f32 v58, v62, v138 :: v_dual_fmac_f32 v59, v72, v138
	v_dual_fmac_f32 v48, v43, v139 :: v_dual_fmac_f32 v49, v53, v139
	v_dual_fmac_f32 v58, v63, v139 :: v_dual_fmac_f32 v59, v73, v139
	s_wait_loadcnt 0x0
	v_dual_fmac_f32 v48, v44, v140 :: v_dual_fmac_f32 v49, v54, v140
	v_dual_fmac_f32 v58, v64, v140 :: v_dual_fmac_f32 v59, v74, v140
	v_dual_fmac_f32 v48, v45, v141 :: v_dual_fmac_f32 v49, v55, v141
	v_dual_fmac_f32 v58, v65, v141 :: v_dual_fmac_f32 v59, v75, v141
	v_dual_fmac_f32 v48, v46, v142 :: v_dual_fmac_f32 v49, v56, v142
	v_dual_fmac_f32 v58, v66, v142 :: v_dual_fmac_f32 v59, v76, v142
	v_dual_fmac_f32 v48, v47, v143 :: v_dual_fmac_f32 v49, v57, v143
	v_dual_fmac_f32 v58, v67, v143 :: v_dual_fmac_f32 v59, v77, v143
	v_dual_add_f32 v252, v252, v48 :: v_dual_add_f32 v253, v253, v49
	v_dual_add_f32 v254, v254, v58 :: v_dual_add_f32 v255, v255, v59
	s_add_co_i32 s16, s16, 0x220
	s_add_co_i32 s17, s17, 0x220
	s_add_co_i32 s18, s18, 0x220
	s_add_co_i32 s19, s19, 0x220
	s_add_co_i32 s28, s28, 0x1000
	s_add_co_i32 s29, s29, 0x1000
	s_add_co_i32 s30, s30, 0x1000
	s_add_co_i32 s31, s31, 0x1000
	s_add_co_i32 s32, s32, 0x1000
	s_add_co_i32 s36, s36, -1
	s_cmp_lg_u32 s36, 0
	s_cbranch_scc1 .Lpx_b5_quad
	.Lpx_b5_tail:
	s_cmp_gt_u32 s37, 0
	s_cbranch_scc0 .Lpx_b5_fold
	buffer_load_b32 v8, v1, s[20:23], s16 offen
	buffer_load_b32 v9, v1, s[20:23], s17 offen
	buffer_load_b32 v10, v1, s[20:23], s18 offen
	buffer_load_b32 v11, v1, s[20:23], s19 offen
	buffer_load_b32 v24, v2, s[20:23], s16 offen offset:8
	buffer_load_b32 v25, v2, s[20:23], s17 offen offset:8
	buffer_load_b32 v26, v2, s[20:23], s18 offen offset:8
	buffer_load_b32 v27, v2, s[20:23], s19 offen offset:8
	buffer_load_b128 v[80:83], v3, s[24:27], s28 offen
	buffer_load_b128 v[84:87], v3, s[24:27], s28 offen offset:16
	buffer_load_b128 v[88:91], v3, s[24:27], s29 offen
	buffer_load_b128 v[92:95], v3, s[24:27], s29 offen offset:16
	buffer_load_b128 v[96:99], v3, s[24:27], s30 offen
	buffer_load_b128 v[100:103], v3, s[24:27], s30 offen offset:16
	buffer_load_b128 v[104:107], v3, s[24:27], s31 offen
	buffer_load_b128 v[108:111], v3, s[24:27], s31 offen offset:16
	buffer_load_b128 v[112:115], v3, s[24:27], s32 offen
	buffer_load_b128 v[116:119], v3, s[24:27], s32 offen offset:16
	s_wait_loadcnt 0xd
	v_and_b32_e32 v46, 0xf0f0f0f, v24
	s_wait_loadcnt 0xc
	v_and_b32_e32 v56, 0xf0f0f0f, v25
	s_wait_loadcnt 0xb
	v_and_b32_e32 v66, 0xf0f0f0f, v26
	s_wait_loadcnt 0xa
	v_and_b32_e32 v76, 0xf0f0f0f, v27
	v_lshrrev_b32_e32 v47, 4, v24
	v_lshrrev_b32_e32 v57, 4, v25
	v_lshrrev_b32_e32 v67, 4, v26
	v_lshrrev_b32_e32 v77, 4, v27
	v_and_b32_e32 v47, 0xf0f0f0f, v47
	v_and_b32_e32 v57, 0xf0f0f0f, v57
	v_and_b32_e32 v67, 0xf0f0f0f, v67
	v_and_b32_e32 v77, 0xf0f0f0f, v77
	v_cvt_f32_ubyte0_e32 v40, v46
	v_cvt_f32_ubyte0_e32 v50, v56
	v_cvt_f32_ubyte0_e32 v60, v66
	v_cvt_f32_ubyte0_e32 v70, v76
	v_cvt_f32_ubyte1_e32 v42, v46
	v_cvt_f32_ubyte1_e32 v52, v56
	v_cvt_f32_ubyte1_e32 v62, v66
	v_cvt_f32_ubyte1_e32 v72, v76
	v_cvt_f32_ubyte2_e32 v44, v46
	v_cvt_f32_ubyte2_e32 v54, v56
	v_cvt_f32_ubyte2_e32 v64, v66
	v_cvt_f32_ubyte2_e32 v74, v76
	v_cvt_f32_ubyte3_e32 v46, v46
	v_cvt_f32_ubyte3_e32 v56, v56
	v_cvt_f32_ubyte3_e32 v66, v66
	v_cvt_f32_ubyte3_e32 v76, v76
	v_cvt_f32_ubyte0_e32 v41, v47
	v_cvt_f32_ubyte0_e32 v51, v57
	v_cvt_f32_ubyte0_e32 v61, v67
	v_cvt_f32_ubyte0_e32 v71, v77
	v_cvt_f32_ubyte1_e32 v43, v47
	v_cvt_f32_ubyte1_e32 v53, v57
	v_cvt_f32_ubyte1_e32 v63, v67
	v_cvt_f32_ubyte1_e32 v73, v77
	v_cvt_f32_ubyte2_e32 v45, v47
	v_cvt_f32_ubyte2_e32 v55, v57
	v_cvt_f32_ubyte2_e32 v65, v67
	v_cvt_f32_ubyte2_e32 v75, v77
	v_cvt_f32_ubyte3_e32 v47, v47
	v_cvt_f32_ubyte3_e32 v57, v57
	v_cvt_f32_ubyte3_e32 v67, v67
	v_cvt_f32_ubyte3_e32 v77, v77
	v_fma_mix_f32 v40, v8, v40, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v9, v50, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v10, v60, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v70, v11, v70, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v41, v8, v41, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v9, v51, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v10, v61, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v71, v11, v71, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v42, v8, v42, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v52, v9, v52, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v62, v10, v62, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v72, v11, v72, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v43, v8, v43, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v53, v9, v53, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v63, v10, v63, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v73, v11, v73, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v44, v8, v44, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v54, v9, v54, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v64, v10, v64, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v74, v11, v74, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v8, v45, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v9, v55, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v65, v10, v65, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v75, v11, v75, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v8, v46, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v9, v56, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v66, v10, v66, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v76, v11, v76, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v8, v47, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v9, v57, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v67, v10, v67, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v77, v11, v77, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x9
	v_dual_mul_f32 v48, v41, v81 :: v_dual_mul_f32 v49, v51, v81
	v_dual_mul_f32 v58, v61, v81 :: v_dual_mul_f32 v59, v71, v81
	s_wait_loadcnt 0x7
	v_dual_mul_f32 v68, v41, v89 :: v_dual_mul_f32 v69, v51, v89
	v_dual_mul_f32 v78, v61, v89 :: v_dual_mul_f32 v79, v71, v89
	v_dual_fmac_f32 v48, v40, v80 :: v_dual_fmac_f32 v49, v50, v80
	v_dual_fmac_f32 v58, v60, v80 :: v_dual_fmac_f32 v59, v70, v80
	v_dual_fmac_f32 v68, v40, v88 :: v_dual_fmac_f32 v69, v50, v88
	v_dual_fmac_f32 v78, v60, v88 :: v_dual_fmac_f32 v79, v70, v88
	v_dual_fmac_f32 v48, v42, v82 :: v_dual_fmac_f32 v49, v52, v82
	v_dual_fmac_f32 v58, v62, v82 :: v_dual_fmac_f32 v59, v72, v82
	v_dual_fmac_f32 v68, v42, v90 :: v_dual_fmac_f32 v69, v52, v90
	v_dual_fmac_f32 v78, v62, v90 :: v_dual_fmac_f32 v79, v72, v90
	v_dual_fmac_f32 v48, v43, v83 :: v_dual_fmac_f32 v49, v53, v83
	v_dual_fmac_f32 v58, v63, v83 :: v_dual_fmac_f32 v59, v73, v83
	v_dual_fmac_f32 v68, v43, v91 :: v_dual_fmac_f32 v69, v53, v91
	v_dual_fmac_f32 v78, v63, v91 :: v_dual_fmac_f32 v79, v73, v91
	v_dual_fmac_f32 v48, v44, v84 :: v_dual_fmac_f32 v49, v54, v84
	v_dual_fmac_f32 v58, v64, v84 :: v_dual_fmac_f32 v59, v74, v84
	s_wait_loadcnt 0x6
	v_dual_fmac_f32 v68, v44, v92 :: v_dual_fmac_f32 v69, v54, v92
	v_dual_fmac_f32 v78, v64, v92 :: v_dual_fmac_f32 v79, v74, v92
	v_dual_fmac_f32 v48, v45, v85 :: v_dual_fmac_f32 v49, v55, v85
	v_dual_fmac_f32 v58, v65, v85 :: v_dual_fmac_f32 v59, v75, v85
	v_dual_fmac_f32 v68, v45, v93 :: v_dual_fmac_f32 v69, v55, v93
	v_dual_fmac_f32 v78, v65, v93 :: v_dual_fmac_f32 v79, v75, v93
	v_dual_fmac_f32 v48, v46, v86 :: v_dual_fmac_f32 v49, v56, v86
	v_dual_fmac_f32 v58, v66, v86 :: v_dual_fmac_f32 v59, v76, v86
	v_dual_fmac_f32 v68, v46, v94 :: v_dual_fmac_f32 v69, v56, v94
	v_dual_fmac_f32 v78, v66, v94 :: v_dual_fmac_f32 v79, v76, v94
	v_dual_fmac_f32 v48, v47, v87 :: v_dual_fmac_f32 v49, v57, v87
	v_dual_fmac_f32 v58, v67, v87 :: v_dual_fmac_f32 v59, v77, v87
	v_dual_fmac_f32 v68, v47, v95 :: v_dual_fmac_f32 v69, v57, v95
	v_dual_fmac_f32 v78, v67, v95 :: v_dual_fmac_f32 v79, v77, v95
	v_dual_add_f32 v176, v176, v48 :: v_dual_add_f32 v177, v177, v49
	v_dual_add_f32 v178, v178, v58 :: v_dual_add_f32 v179, v179, v59
	v_dual_add_f32 v192, v192, v68 :: v_dual_add_f32 v193, v193, v69
	v_dual_add_f32 v194, v194, v78 :: v_dual_add_f32 v195, v195, v79
	s_wait_loadcnt 0x5
	v_dual_mul_f32 v48, v41, v97 :: v_dual_mul_f32 v49, v51, v97
	v_dual_mul_f32 v58, v61, v97 :: v_dual_mul_f32 v59, v71, v97
	s_wait_loadcnt 0x3
	v_dual_mul_f32 v68, v41, v105 :: v_dual_mul_f32 v69, v51, v105
	v_dual_mul_f32 v78, v61, v105 :: v_dual_mul_f32 v79, v71, v105
	v_dual_fmac_f32 v48, v40, v96 :: v_dual_fmac_f32 v49, v50, v96
	v_dual_fmac_f32 v58, v60, v96 :: v_dual_fmac_f32 v59, v70, v96
	v_dual_fmac_f32 v68, v40, v104 :: v_dual_fmac_f32 v69, v50, v104
	v_dual_fmac_f32 v78, v60, v104 :: v_dual_fmac_f32 v79, v70, v104
	v_dual_fmac_f32 v48, v42, v98 :: v_dual_fmac_f32 v49, v52, v98
	v_dual_fmac_f32 v58, v62, v98 :: v_dual_fmac_f32 v59, v72, v98
	v_dual_fmac_f32 v68, v42, v106 :: v_dual_fmac_f32 v69, v52, v106
	v_dual_fmac_f32 v78, v62, v106 :: v_dual_fmac_f32 v79, v72, v106
	v_dual_fmac_f32 v48, v43, v99 :: v_dual_fmac_f32 v49, v53, v99
	v_dual_fmac_f32 v58, v63, v99 :: v_dual_fmac_f32 v59, v73, v99
	v_dual_fmac_f32 v68, v43, v107 :: v_dual_fmac_f32 v69, v53, v107
	v_dual_fmac_f32 v78, v63, v107 :: v_dual_fmac_f32 v79, v73, v107
	v_dual_fmac_f32 v48, v44, v100 :: v_dual_fmac_f32 v49, v54, v100
	v_dual_fmac_f32 v58, v64, v100 :: v_dual_fmac_f32 v59, v74, v100
	s_wait_loadcnt 0x2
	v_dual_fmac_f32 v68, v44, v108 :: v_dual_fmac_f32 v69, v54, v108
	v_dual_fmac_f32 v78, v64, v108 :: v_dual_fmac_f32 v79, v74, v108
	v_dual_fmac_f32 v48, v45, v101 :: v_dual_fmac_f32 v49, v55, v101
	v_dual_fmac_f32 v58, v65, v101 :: v_dual_fmac_f32 v59, v75, v101
	v_dual_fmac_f32 v68, v45, v109 :: v_dual_fmac_f32 v69, v55, v109
	v_dual_fmac_f32 v78, v65, v109 :: v_dual_fmac_f32 v79, v75, v109
	v_dual_fmac_f32 v48, v46, v102 :: v_dual_fmac_f32 v49, v56, v102
	v_dual_fmac_f32 v58, v66, v102 :: v_dual_fmac_f32 v59, v76, v102
	v_dual_fmac_f32 v68, v46, v110 :: v_dual_fmac_f32 v69, v56, v110
	v_dual_fmac_f32 v78, v66, v110 :: v_dual_fmac_f32 v79, v76, v110
	v_dual_fmac_f32 v48, v47, v103 :: v_dual_fmac_f32 v49, v57, v103
	v_dual_fmac_f32 v58, v67, v103 :: v_dual_fmac_f32 v59, v77, v103
	v_dual_fmac_f32 v68, v47, v111 :: v_dual_fmac_f32 v69, v57, v111
	v_dual_fmac_f32 v78, v67, v111 :: v_dual_fmac_f32 v79, v77, v111
	v_dual_add_f32 v208, v208, v48 :: v_dual_add_f32 v209, v209, v49
	v_dual_add_f32 v210, v210, v58 :: v_dual_add_f32 v211, v211, v59
	v_dual_add_f32 v224, v224, v68 :: v_dual_add_f32 v225, v225, v69
	v_dual_add_f32 v226, v226, v78 :: v_dual_add_f32 v227, v227, v79
	s_wait_loadcnt 0x1
	v_dual_mul_f32 v48, v41, v113 :: v_dual_mul_f32 v49, v51, v113
	v_dual_mul_f32 v58, v61, v113 :: v_dual_mul_f32 v59, v71, v113
	v_dual_fmac_f32 v48, v40, v112 :: v_dual_fmac_f32 v49, v50, v112
	v_dual_fmac_f32 v58, v60, v112 :: v_dual_fmac_f32 v59, v70, v112
	v_dual_fmac_f32 v48, v42, v114 :: v_dual_fmac_f32 v49, v52, v114
	v_dual_fmac_f32 v58, v62, v114 :: v_dual_fmac_f32 v59, v72, v114
	v_dual_fmac_f32 v48, v43, v115 :: v_dual_fmac_f32 v49, v53, v115
	v_dual_fmac_f32 v58, v63, v115 :: v_dual_fmac_f32 v59, v73, v115
	s_wait_loadcnt 0x0
	v_dual_fmac_f32 v48, v44, v116 :: v_dual_fmac_f32 v49, v54, v116
	v_dual_fmac_f32 v58, v64, v116 :: v_dual_fmac_f32 v59, v74, v116
	v_dual_fmac_f32 v48, v45, v117 :: v_dual_fmac_f32 v49, v55, v117
	v_dual_fmac_f32 v58, v65, v117 :: v_dual_fmac_f32 v59, v75, v117
	v_dual_fmac_f32 v48, v46, v118 :: v_dual_fmac_f32 v49, v56, v118
	v_dual_fmac_f32 v58, v66, v118 :: v_dual_fmac_f32 v59, v76, v118
	v_dual_fmac_f32 v48, v47, v119 :: v_dual_fmac_f32 v49, v57, v119
	v_dual_fmac_f32 v58, v67, v119 :: v_dual_fmac_f32 v59, v77, v119
	v_dual_add_f32 v240, v240, v48 :: v_dual_add_f32 v241, v241, v49
	v_dual_add_f32 v242, v242, v58 :: v_dual_add_f32 v243, v243, v59
	s_cmp_gt_u32 s37, 1
	s_cbranch_scc0 .Lpx_b5_fold
	buffer_load_b32 v12, v1, s[20:23], s16 offen offset:136
	buffer_load_b32 v13, v1, s[20:23], s17 offen offset:136
	buffer_load_b32 v14, v1, s[20:23], s18 offen offset:136
	buffer_load_b32 v15, v1, s[20:23], s19 offen offset:136
	buffer_load_b32 v28, v2, s[20:23], s16 offen offset:144
	buffer_load_b32 v29, v2, s[20:23], s17 offen offset:144
	buffer_load_b32 v30, v2, s[20:23], s18 offen offset:144
	buffer_load_b32 v31, v2, s[20:23], s19 offen offset:144
	buffer_load_b128 v[80:83], v3, s[24:27], s28 offen offset:1024
	buffer_load_b128 v[84:87], v3, s[24:27], s28 offen offset:1040
	buffer_load_b128 v[88:91], v3, s[24:27], s29 offen offset:1024
	buffer_load_b128 v[92:95], v3, s[24:27], s29 offen offset:1040
	buffer_load_b128 v[96:99], v3, s[24:27], s30 offen offset:1024
	buffer_load_b128 v[100:103], v3, s[24:27], s30 offen offset:1040
	buffer_load_b128 v[104:107], v3, s[24:27], s31 offen offset:1024
	buffer_load_b128 v[108:111], v3, s[24:27], s31 offen offset:1040
	buffer_load_b128 v[112:115], v3, s[24:27], s32 offen offset:1024
	buffer_load_b128 v[116:119], v3, s[24:27], s32 offen offset:1040
	s_wait_loadcnt 0xd
	v_and_b32_e32 v46, 0xf0f0f0f, v28
	s_wait_loadcnt 0xc
	v_and_b32_e32 v56, 0xf0f0f0f, v29
	s_wait_loadcnt 0xb
	v_and_b32_e32 v66, 0xf0f0f0f, v30
	s_wait_loadcnt 0xa
	v_and_b32_e32 v76, 0xf0f0f0f, v31
	v_lshrrev_b32_e32 v47, 4, v28
	v_lshrrev_b32_e32 v57, 4, v29
	v_lshrrev_b32_e32 v67, 4, v30
	v_lshrrev_b32_e32 v77, 4, v31
	v_and_b32_e32 v47, 0xf0f0f0f, v47
	v_and_b32_e32 v57, 0xf0f0f0f, v57
	v_and_b32_e32 v67, 0xf0f0f0f, v67
	v_and_b32_e32 v77, 0xf0f0f0f, v77
	v_cvt_f32_ubyte0_e32 v40, v46
	v_cvt_f32_ubyte0_e32 v50, v56
	v_cvt_f32_ubyte0_e32 v60, v66
	v_cvt_f32_ubyte0_e32 v70, v76
	v_cvt_f32_ubyte1_e32 v42, v46
	v_cvt_f32_ubyte1_e32 v52, v56
	v_cvt_f32_ubyte1_e32 v62, v66
	v_cvt_f32_ubyte1_e32 v72, v76
	v_cvt_f32_ubyte2_e32 v44, v46
	v_cvt_f32_ubyte2_e32 v54, v56
	v_cvt_f32_ubyte2_e32 v64, v66
	v_cvt_f32_ubyte2_e32 v74, v76
	v_cvt_f32_ubyte3_e32 v46, v46
	v_cvt_f32_ubyte3_e32 v56, v56
	v_cvt_f32_ubyte3_e32 v66, v66
	v_cvt_f32_ubyte3_e32 v76, v76
	v_cvt_f32_ubyte0_e32 v41, v47
	v_cvt_f32_ubyte0_e32 v51, v57
	v_cvt_f32_ubyte0_e32 v61, v67
	v_cvt_f32_ubyte0_e32 v71, v77
	v_cvt_f32_ubyte1_e32 v43, v47
	v_cvt_f32_ubyte1_e32 v53, v57
	v_cvt_f32_ubyte1_e32 v63, v67
	v_cvt_f32_ubyte1_e32 v73, v77
	v_cvt_f32_ubyte2_e32 v45, v47
	v_cvt_f32_ubyte2_e32 v55, v57
	v_cvt_f32_ubyte2_e32 v65, v67
	v_cvt_f32_ubyte2_e32 v75, v77
	v_cvt_f32_ubyte3_e32 v47, v47
	v_cvt_f32_ubyte3_e32 v57, v57
	v_cvt_f32_ubyte3_e32 v67, v67
	v_cvt_f32_ubyte3_e32 v77, v77
	v_fma_mix_f32 v40, v12, v40, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v13, v50, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v14, v60, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v70, v15, v70, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v41, v12, v41, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v13, v51, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v14, v61, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v71, v15, v71, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v42, v12, v42, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v52, v13, v52, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v62, v14, v62, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v72, v15, v72, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v43, v12, v43, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v53, v13, v53, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v63, v14, v63, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v73, v15, v73, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v44, v12, v44, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v54, v13, v54, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v64, v14, v64, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v74, v15, v74, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v12, v45, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v13, v55, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v65, v14, v65, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v75, v15, v75, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v12, v46, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v13, v56, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v66, v14, v66, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v76, v15, v76, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v12, v47, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v13, v57, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v67, v14, v67, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v77, v15, v77, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x9
	v_dual_mul_f32 v48, v41, v81 :: v_dual_mul_f32 v49, v51, v81
	v_dual_mul_f32 v58, v61, v81 :: v_dual_mul_f32 v59, v71, v81
	s_wait_loadcnt 0x7
	v_dual_mul_f32 v68, v41, v89 :: v_dual_mul_f32 v69, v51, v89
	v_dual_mul_f32 v78, v61, v89 :: v_dual_mul_f32 v79, v71, v89
	v_dual_fmac_f32 v48, v40, v80 :: v_dual_fmac_f32 v49, v50, v80
	v_dual_fmac_f32 v58, v60, v80 :: v_dual_fmac_f32 v59, v70, v80
	v_dual_fmac_f32 v68, v40, v88 :: v_dual_fmac_f32 v69, v50, v88
	v_dual_fmac_f32 v78, v60, v88 :: v_dual_fmac_f32 v79, v70, v88
	v_dual_fmac_f32 v48, v42, v82 :: v_dual_fmac_f32 v49, v52, v82
	v_dual_fmac_f32 v58, v62, v82 :: v_dual_fmac_f32 v59, v72, v82
	v_dual_fmac_f32 v68, v42, v90 :: v_dual_fmac_f32 v69, v52, v90
	v_dual_fmac_f32 v78, v62, v90 :: v_dual_fmac_f32 v79, v72, v90
	v_dual_fmac_f32 v48, v43, v83 :: v_dual_fmac_f32 v49, v53, v83
	v_dual_fmac_f32 v58, v63, v83 :: v_dual_fmac_f32 v59, v73, v83
	v_dual_fmac_f32 v68, v43, v91 :: v_dual_fmac_f32 v69, v53, v91
	v_dual_fmac_f32 v78, v63, v91 :: v_dual_fmac_f32 v79, v73, v91
	v_dual_fmac_f32 v48, v44, v84 :: v_dual_fmac_f32 v49, v54, v84
	v_dual_fmac_f32 v58, v64, v84 :: v_dual_fmac_f32 v59, v74, v84
	s_wait_loadcnt 0x6
	v_dual_fmac_f32 v68, v44, v92 :: v_dual_fmac_f32 v69, v54, v92
	v_dual_fmac_f32 v78, v64, v92 :: v_dual_fmac_f32 v79, v74, v92
	v_dual_fmac_f32 v48, v45, v85 :: v_dual_fmac_f32 v49, v55, v85
	v_dual_fmac_f32 v58, v65, v85 :: v_dual_fmac_f32 v59, v75, v85
	v_dual_fmac_f32 v68, v45, v93 :: v_dual_fmac_f32 v69, v55, v93
	v_dual_fmac_f32 v78, v65, v93 :: v_dual_fmac_f32 v79, v75, v93
	v_dual_fmac_f32 v48, v46, v86 :: v_dual_fmac_f32 v49, v56, v86
	v_dual_fmac_f32 v58, v66, v86 :: v_dual_fmac_f32 v59, v76, v86
	v_dual_fmac_f32 v68, v46, v94 :: v_dual_fmac_f32 v69, v56, v94
	v_dual_fmac_f32 v78, v66, v94 :: v_dual_fmac_f32 v79, v76, v94
	v_dual_fmac_f32 v48, v47, v87 :: v_dual_fmac_f32 v49, v57, v87
	v_dual_fmac_f32 v58, v67, v87 :: v_dual_fmac_f32 v59, v77, v87
	v_dual_fmac_f32 v68, v47, v95 :: v_dual_fmac_f32 v69, v57, v95
	v_dual_fmac_f32 v78, v67, v95 :: v_dual_fmac_f32 v79, v77, v95
	v_dual_add_f32 v180, v180, v48 :: v_dual_add_f32 v181, v181, v49
	v_dual_add_f32 v182, v182, v58 :: v_dual_add_f32 v183, v183, v59
	v_dual_add_f32 v196, v196, v68 :: v_dual_add_f32 v197, v197, v69
	v_dual_add_f32 v198, v198, v78 :: v_dual_add_f32 v199, v199, v79
	s_wait_loadcnt 0x5
	v_dual_mul_f32 v48, v41, v97 :: v_dual_mul_f32 v49, v51, v97
	v_dual_mul_f32 v58, v61, v97 :: v_dual_mul_f32 v59, v71, v97
	s_wait_loadcnt 0x3
	v_dual_mul_f32 v68, v41, v105 :: v_dual_mul_f32 v69, v51, v105
	v_dual_mul_f32 v78, v61, v105 :: v_dual_mul_f32 v79, v71, v105
	v_dual_fmac_f32 v48, v40, v96 :: v_dual_fmac_f32 v49, v50, v96
	v_dual_fmac_f32 v58, v60, v96 :: v_dual_fmac_f32 v59, v70, v96
	v_dual_fmac_f32 v68, v40, v104 :: v_dual_fmac_f32 v69, v50, v104
	v_dual_fmac_f32 v78, v60, v104 :: v_dual_fmac_f32 v79, v70, v104
	v_dual_fmac_f32 v48, v42, v98 :: v_dual_fmac_f32 v49, v52, v98
	v_dual_fmac_f32 v58, v62, v98 :: v_dual_fmac_f32 v59, v72, v98
	v_dual_fmac_f32 v68, v42, v106 :: v_dual_fmac_f32 v69, v52, v106
	v_dual_fmac_f32 v78, v62, v106 :: v_dual_fmac_f32 v79, v72, v106
	v_dual_fmac_f32 v48, v43, v99 :: v_dual_fmac_f32 v49, v53, v99
	v_dual_fmac_f32 v58, v63, v99 :: v_dual_fmac_f32 v59, v73, v99
	v_dual_fmac_f32 v68, v43, v107 :: v_dual_fmac_f32 v69, v53, v107
	v_dual_fmac_f32 v78, v63, v107 :: v_dual_fmac_f32 v79, v73, v107
	v_dual_fmac_f32 v48, v44, v100 :: v_dual_fmac_f32 v49, v54, v100
	v_dual_fmac_f32 v58, v64, v100 :: v_dual_fmac_f32 v59, v74, v100
	s_wait_loadcnt 0x2
	v_dual_fmac_f32 v68, v44, v108 :: v_dual_fmac_f32 v69, v54, v108
	v_dual_fmac_f32 v78, v64, v108 :: v_dual_fmac_f32 v79, v74, v108
	v_dual_fmac_f32 v48, v45, v101 :: v_dual_fmac_f32 v49, v55, v101
	v_dual_fmac_f32 v58, v65, v101 :: v_dual_fmac_f32 v59, v75, v101
	v_dual_fmac_f32 v68, v45, v109 :: v_dual_fmac_f32 v69, v55, v109
	v_dual_fmac_f32 v78, v65, v109 :: v_dual_fmac_f32 v79, v75, v109
	v_dual_fmac_f32 v48, v46, v102 :: v_dual_fmac_f32 v49, v56, v102
	v_dual_fmac_f32 v58, v66, v102 :: v_dual_fmac_f32 v59, v76, v102
	v_dual_fmac_f32 v68, v46, v110 :: v_dual_fmac_f32 v69, v56, v110
	v_dual_fmac_f32 v78, v66, v110 :: v_dual_fmac_f32 v79, v76, v110
	v_dual_fmac_f32 v48, v47, v103 :: v_dual_fmac_f32 v49, v57, v103
	v_dual_fmac_f32 v58, v67, v103 :: v_dual_fmac_f32 v59, v77, v103
	v_dual_fmac_f32 v68, v47, v111 :: v_dual_fmac_f32 v69, v57, v111
	v_dual_fmac_f32 v78, v67, v111 :: v_dual_fmac_f32 v79, v77, v111
	v_dual_add_f32 v212, v212, v48 :: v_dual_add_f32 v213, v213, v49
	v_dual_add_f32 v214, v214, v58 :: v_dual_add_f32 v215, v215, v59
	v_dual_add_f32 v228, v228, v68 :: v_dual_add_f32 v229, v229, v69
	v_dual_add_f32 v230, v230, v78 :: v_dual_add_f32 v231, v231, v79
	s_wait_loadcnt 0x1
	v_dual_mul_f32 v48, v41, v113 :: v_dual_mul_f32 v49, v51, v113
	v_dual_mul_f32 v58, v61, v113 :: v_dual_mul_f32 v59, v71, v113
	v_dual_fmac_f32 v48, v40, v112 :: v_dual_fmac_f32 v49, v50, v112
	v_dual_fmac_f32 v58, v60, v112 :: v_dual_fmac_f32 v59, v70, v112
	v_dual_fmac_f32 v48, v42, v114 :: v_dual_fmac_f32 v49, v52, v114
	v_dual_fmac_f32 v58, v62, v114 :: v_dual_fmac_f32 v59, v72, v114
	v_dual_fmac_f32 v48, v43, v115 :: v_dual_fmac_f32 v49, v53, v115
	v_dual_fmac_f32 v58, v63, v115 :: v_dual_fmac_f32 v59, v73, v115
	s_wait_loadcnt 0x0
	v_dual_fmac_f32 v48, v44, v116 :: v_dual_fmac_f32 v49, v54, v116
	v_dual_fmac_f32 v58, v64, v116 :: v_dual_fmac_f32 v59, v74, v116
	v_dual_fmac_f32 v48, v45, v117 :: v_dual_fmac_f32 v49, v55, v117
	v_dual_fmac_f32 v58, v65, v117 :: v_dual_fmac_f32 v59, v75, v117
	v_dual_fmac_f32 v48, v46, v118 :: v_dual_fmac_f32 v49, v56, v118
	v_dual_fmac_f32 v58, v66, v118 :: v_dual_fmac_f32 v59, v76, v118
	v_dual_fmac_f32 v48, v47, v119 :: v_dual_fmac_f32 v49, v57, v119
	v_dual_fmac_f32 v58, v67, v119 :: v_dual_fmac_f32 v59, v77, v119
	v_dual_add_f32 v244, v244, v48 :: v_dual_add_f32 v245, v245, v49
	v_dual_add_f32 v246, v246, v58 :: v_dual_add_f32 v247, v247, v59
	s_cmp_gt_u32 s37, 2
	s_cbranch_scc0 .Lpx_b5_fold
	buffer_load_b32 v16, v1, s[20:23], s16 offen offset:272
	buffer_load_b32 v17, v1, s[20:23], s17 offen offset:272
	buffer_load_b32 v18, v1, s[20:23], s18 offen offset:272
	buffer_load_b32 v19, v1, s[20:23], s19 offen offset:272
	buffer_load_b32 v32, v2, s[20:23], s16 offen offset:280
	buffer_load_b32 v33, v2, s[20:23], s17 offen offset:280
	buffer_load_b32 v34, v2, s[20:23], s18 offen offset:280
	buffer_load_b32 v35, v2, s[20:23], s19 offen offset:280
	buffer_load_b128 v[80:83], v3, s[24:27], s28 offen offset:2048
	buffer_load_b128 v[84:87], v3, s[24:27], s28 offen offset:2064
	buffer_load_b128 v[88:91], v3, s[24:27], s29 offen offset:2048
	buffer_load_b128 v[92:95], v3, s[24:27], s29 offen offset:2064
	buffer_load_b128 v[96:99], v3, s[24:27], s30 offen offset:2048
	buffer_load_b128 v[100:103], v3, s[24:27], s30 offen offset:2064
	buffer_load_b128 v[104:107], v3, s[24:27], s31 offen offset:2048
	buffer_load_b128 v[108:111], v3, s[24:27], s31 offen offset:2064
	buffer_load_b128 v[112:115], v3, s[24:27], s32 offen offset:2048
	buffer_load_b128 v[116:119], v3, s[24:27], s32 offen offset:2064
	s_wait_loadcnt 0xd
	v_and_b32_e32 v46, 0xf0f0f0f, v32
	s_wait_loadcnt 0xc
	v_and_b32_e32 v56, 0xf0f0f0f, v33
	s_wait_loadcnt 0xb
	v_and_b32_e32 v66, 0xf0f0f0f, v34
	s_wait_loadcnt 0xa
	v_and_b32_e32 v76, 0xf0f0f0f, v35
	v_lshrrev_b32_e32 v47, 4, v32
	v_lshrrev_b32_e32 v57, 4, v33
	v_lshrrev_b32_e32 v67, 4, v34
	v_lshrrev_b32_e32 v77, 4, v35
	v_and_b32_e32 v47, 0xf0f0f0f, v47
	v_and_b32_e32 v57, 0xf0f0f0f, v57
	v_and_b32_e32 v67, 0xf0f0f0f, v67
	v_and_b32_e32 v77, 0xf0f0f0f, v77
	v_cvt_f32_ubyte0_e32 v40, v46
	v_cvt_f32_ubyte0_e32 v50, v56
	v_cvt_f32_ubyte0_e32 v60, v66
	v_cvt_f32_ubyte0_e32 v70, v76
	v_cvt_f32_ubyte1_e32 v42, v46
	v_cvt_f32_ubyte1_e32 v52, v56
	v_cvt_f32_ubyte1_e32 v62, v66
	v_cvt_f32_ubyte1_e32 v72, v76
	v_cvt_f32_ubyte2_e32 v44, v46
	v_cvt_f32_ubyte2_e32 v54, v56
	v_cvt_f32_ubyte2_e32 v64, v66
	v_cvt_f32_ubyte2_e32 v74, v76
	v_cvt_f32_ubyte3_e32 v46, v46
	v_cvt_f32_ubyte3_e32 v56, v56
	v_cvt_f32_ubyte3_e32 v66, v66
	v_cvt_f32_ubyte3_e32 v76, v76
	v_cvt_f32_ubyte0_e32 v41, v47
	v_cvt_f32_ubyte0_e32 v51, v57
	v_cvt_f32_ubyte0_e32 v61, v67
	v_cvt_f32_ubyte0_e32 v71, v77
	v_cvt_f32_ubyte1_e32 v43, v47
	v_cvt_f32_ubyte1_e32 v53, v57
	v_cvt_f32_ubyte1_e32 v63, v67
	v_cvt_f32_ubyte1_e32 v73, v77
	v_cvt_f32_ubyte2_e32 v45, v47
	v_cvt_f32_ubyte2_e32 v55, v57
	v_cvt_f32_ubyte2_e32 v65, v67
	v_cvt_f32_ubyte2_e32 v75, v77
	v_cvt_f32_ubyte3_e32 v47, v47
	v_cvt_f32_ubyte3_e32 v57, v57
	v_cvt_f32_ubyte3_e32 v67, v67
	v_cvt_f32_ubyte3_e32 v77, v77
	v_fma_mix_f32 v40, v16, v40, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v17, v50, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v18, v60, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v70, v19, v70, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v41, v16, v41, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v17, v51, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v18, v61, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v71, v19, v71, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v42, v16, v42, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v52, v17, v52, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v62, v18, v62, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v72, v19, v72, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v43, v16, v43, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v53, v17, v53, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v63, v18, v63, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v73, v19, v73, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v44, v16, v44, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v54, v17, v54, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v64, v18, v64, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v74, v19, v74, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v16, v45, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v17, v55, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v65, v18, v65, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v75, v19, v75, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v16, v46, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v17, v56, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v66, v18, v66, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v76, v19, v76, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v16, v47, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v17, v57, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v67, v18, v67, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v77, v19, v77, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x9
	v_dual_mul_f32 v48, v41, v81 :: v_dual_mul_f32 v49, v51, v81
	v_dual_mul_f32 v58, v61, v81 :: v_dual_mul_f32 v59, v71, v81
	s_wait_loadcnt 0x7
	v_dual_mul_f32 v68, v41, v89 :: v_dual_mul_f32 v69, v51, v89
	v_dual_mul_f32 v78, v61, v89 :: v_dual_mul_f32 v79, v71, v89
	v_dual_fmac_f32 v48, v40, v80 :: v_dual_fmac_f32 v49, v50, v80
	v_dual_fmac_f32 v58, v60, v80 :: v_dual_fmac_f32 v59, v70, v80
	v_dual_fmac_f32 v68, v40, v88 :: v_dual_fmac_f32 v69, v50, v88
	v_dual_fmac_f32 v78, v60, v88 :: v_dual_fmac_f32 v79, v70, v88
	v_dual_fmac_f32 v48, v42, v82 :: v_dual_fmac_f32 v49, v52, v82
	v_dual_fmac_f32 v58, v62, v82 :: v_dual_fmac_f32 v59, v72, v82
	v_dual_fmac_f32 v68, v42, v90 :: v_dual_fmac_f32 v69, v52, v90
	v_dual_fmac_f32 v78, v62, v90 :: v_dual_fmac_f32 v79, v72, v90
	v_dual_fmac_f32 v48, v43, v83 :: v_dual_fmac_f32 v49, v53, v83
	v_dual_fmac_f32 v58, v63, v83 :: v_dual_fmac_f32 v59, v73, v83
	v_dual_fmac_f32 v68, v43, v91 :: v_dual_fmac_f32 v69, v53, v91
	v_dual_fmac_f32 v78, v63, v91 :: v_dual_fmac_f32 v79, v73, v91
	v_dual_fmac_f32 v48, v44, v84 :: v_dual_fmac_f32 v49, v54, v84
	v_dual_fmac_f32 v58, v64, v84 :: v_dual_fmac_f32 v59, v74, v84
	s_wait_loadcnt 0x6
	v_dual_fmac_f32 v68, v44, v92 :: v_dual_fmac_f32 v69, v54, v92
	v_dual_fmac_f32 v78, v64, v92 :: v_dual_fmac_f32 v79, v74, v92
	v_dual_fmac_f32 v48, v45, v85 :: v_dual_fmac_f32 v49, v55, v85
	v_dual_fmac_f32 v58, v65, v85 :: v_dual_fmac_f32 v59, v75, v85
	v_dual_fmac_f32 v68, v45, v93 :: v_dual_fmac_f32 v69, v55, v93
	v_dual_fmac_f32 v78, v65, v93 :: v_dual_fmac_f32 v79, v75, v93
	v_dual_fmac_f32 v48, v46, v86 :: v_dual_fmac_f32 v49, v56, v86
	v_dual_fmac_f32 v58, v66, v86 :: v_dual_fmac_f32 v59, v76, v86
	v_dual_fmac_f32 v68, v46, v94 :: v_dual_fmac_f32 v69, v56, v94
	v_dual_fmac_f32 v78, v66, v94 :: v_dual_fmac_f32 v79, v76, v94
	v_dual_fmac_f32 v48, v47, v87 :: v_dual_fmac_f32 v49, v57, v87
	v_dual_fmac_f32 v58, v67, v87 :: v_dual_fmac_f32 v59, v77, v87
	v_dual_fmac_f32 v68, v47, v95 :: v_dual_fmac_f32 v69, v57, v95
	v_dual_fmac_f32 v78, v67, v95 :: v_dual_fmac_f32 v79, v77, v95
	v_dual_add_f32 v184, v184, v48 :: v_dual_add_f32 v185, v185, v49
	v_dual_add_f32 v186, v186, v58 :: v_dual_add_f32 v187, v187, v59
	v_dual_add_f32 v200, v200, v68 :: v_dual_add_f32 v201, v201, v69
	v_dual_add_f32 v202, v202, v78 :: v_dual_add_f32 v203, v203, v79
	s_wait_loadcnt 0x5
	v_dual_mul_f32 v48, v41, v97 :: v_dual_mul_f32 v49, v51, v97
	v_dual_mul_f32 v58, v61, v97 :: v_dual_mul_f32 v59, v71, v97
	s_wait_loadcnt 0x3
	v_dual_mul_f32 v68, v41, v105 :: v_dual_mul_f32 v69, v51, v105
	v_dual_mul_f32 v78, v61, v105 :: v_dual_mul_f32 v79, v71, v105
	v_dual_fmac_f32 v48, v40, v96 :: v_dual_fmac_f32 v49, v50, v96
	v_dual_fmac_f32 v58, v60, v96 :: v_dual_fmac_f32 v59, v70, v96
	v_dual_fmac_f32 v68, v40, v104 :: v_dual_fmac_f32 v69, v50, v104
	v_dual_fmac_f32 v78, v60, v104 :: v_dual_fmac_f32 v79, v70, v104
	v_dual_fmac_f32 v48, v42, v98 :: v_dual_fmac_f32 v49, v52, v98
	v_dual_fmac_f32 v58, v62, v98 :: v_dual_fmac_f32 v59, v72, v98
	v_dual_fmac_f32 v68, v42, v106 :: v_dual_fmac_f32 v69, v52, v106
	v_dual_fmac_f32 v78, v62, v106 :: v_dual_fmac_f32 v79, v72, v106
	v_dual_fmac_f32 v48, v43, v99 :: v_dual_fmac_f32 v49, v53, v99
	v_dual_fmac_f32 v58, v63, v99 :: v_dual_fmac_f32 v59, v73, v99
	v_dual_fmac_f32 v68, v43, v107 :: v_dual_fmac_f32 v69, v53, v107
	v_dual_fmac_f32 v78, v63, v107 :: v_dual_fmac_f32 v79, v73, v107
	v_dual_fmac_f32 v48, v44, v100 :: v_dual_fmac_f32 v49, v54, v100
	v_dual_fmac_f32 v58, v64, v100 :: v_dual_fmac_f32 v59, v74, v100
	s_wait_loadcnt 0x2
	v_dual_fmac_f32 v68, v44, v108 :: v_dual_fmac_f32 v69, v54, v108
	v_dual_fmac_f32 v78, v64, v108 :: v_dual_fmac_f32 v79, v74, v108
	v_dual_fmac_f32 v48, v45, v101 :: v_dual_fmac_f32 v49, v55, v101
	v_dual_fmac_f32 v58, v65, v101 :: v_dual_fmac_f32 v59, v75, v101
	v_dual_fmac_f32 v68, v45, v109 :: v_dual_fmac_f32 v69, v55, v109
	v_dual_fmac_f32 v78, v65, v109 :: v_dual_fmac_f32 v79, v75, v109
	v_dual_fmac_f32 v48, v46, v102 :: v_dual_fmac_f32 v49, v56, v102
	v_dual_fmac_f32 v58, v66, v102 :: v_dual_fmac_f32 v59, v76, v102
	v_dual_fmac_f32 v68, v46, v110 :: v_dual_fmac_f32 v69, v56, v110
	v_dual_fmac_f32 v78, v66, v110 :: v_dual_fmac_f32 v79, v76, v110
	v_dual_fmac_f32 v48, v47, v103 :: v_dual_fmac_f32 v49, v57, v103
	v_dual_fmac_f32 v58, v67, v103 :: v_dual_fmac_f32 v59, v77, v103
	v_dual_fmac_f32 v68, v47, v111 :: v_dual_fmac_f32 v69, v57, v111
	v_dual_fmac_f32 v78, v67, v111 :: v_dual_fmac_f32 v79, v77, v111
	v_dual_add_f32 v216, v216, v48 :: v_dual_add_f32 v217, v217, v49
	v_dual_add_f32 v218, v218, v58 :: v_dual_add_f32 v219, v219, v59
	v_dual_add_f32 v232, v232, v68 :: v_dual_add_f32 v233, v233, v69
	v_dual_add_f32 v234, v234, v78 :: v_dual_add_f32 v235, v235, v79
	s_wait_loadcnt 0x1
	v_dual_mul_f32 v48, v41, v113 :: v_dual_mul_f32 v49, v51, v113
	v_dual_mul_f32 v58, v61, v113 :: v_dual_mul_f32 v59, v71, v113
	v_dual_fmac_f32 v48, v40, v112 :: v_dual_fmac_f32 v49, v50, v112
	v_dual_fmac_f32 v58, v60, v112 :: v_dual_fmac_f32 v59, v70, v112
	v_dual_fmac_f32 v48, v42, v114 :: v_dual_fmac_f32 v49, v52, v114
	v_dual_fmac_f32 v58, v62, v114 :: v_dual_fmac_f32 v59, v72, v114
	v_dual_fmac_f32 v48, v43, v115 :: v_dual_fmac_f32 v49, v53, v115
	v_dual_fmac_f32 v58, v63, v115 :: v_dual_fmac_f32 v59, v73, v115
	s_wait_loadcnt 0x0
	v_dual_fmac_f32 v48, v44, v116 :: v_dual_fmac_f32 v49, v54, v116
	v_dual_fmac_f32 v58, v64, v116 :: v_dual_fmac_f32 v59, v74, v116
	v_dual_fmac_f32 v48, v45, v117 :: v_dual_fmac_f32 v49, v55, v117
	v_dual_fmac_f32 v58, v65, v117 :: v_dual_fmac_f32 v59, v75, v117
	v_dual_fmac_f32 v48, v46, v118 :: v_dual_fmac_f32 v49, v56, v118
	v_dual_fmac_f32 v58, v66, v118 :: v_dual_fmac_f32 v59, v76, v118
	v_dual_fmac_f32 v48, v47, v119 :: v_dual_fmac_f32 v49, v57, v119
	v_dual_fmac_f32 v58, v67, v119 :: v_dual_fmac_f32 v59, v77, v119
	v_dual_add_f32 v248, v248, v48 :: v_dual_add_f32 v249, v249, v49
	v_dual_add_f32 v250, v250, v58 :: v_dual_add_f32 v251, v251, v59
	.Lpx_b5_fold:
	v_dual_add_f32 v176, v176, v180 :: v_dual_add_f32 v177, v177, v181
	v_dual_add_f32 v178, v178, v182 :: v_dual_add_f32 v179, v179, v183
	v_dual_add_f32 v184, v184, v188 :: v_dual_add_f32 v185, v185, v189
	v_dual_add_f32 v186, v186, v190 :: v_dual_add_f32 v187, v187, v191
	v_dual_add_f32 v176, v176, v184 :: v_dual_add_f32 v177, v177, v185
	v_dual_add_f32 v178, v178, v186 :: v_dual_add_f32 v179, v179, v187
	v_dual_add_f32 v192, v192, v196 :: v_dual_add_f32 v193, v193, v197
	v_dual_add_f32 v194, v194, v198 :: v_dual_add_f32 v195, v195, v199
	v_dual_add_f32 v200, v200, v204 :: v_dual_add_f32 v201, v201, v205
	v_dual_add_f32 v202, v202, v206 :: v_dual_add_f32 v203, v203, v207
	v_dual_add_f32 v192, v192, v200 :: v_dual_add_f32 v193, v193, v201
	v_dual_add_f32 v194, v194, v202 :: v_dual_add_f32 v195, v195, v203
	v_dual_add_f32 v208, v208, v212 :: v_dual_add_f32 v209, v209, v213
	v_dual_add_f32 v210, v210, v214 :: v_dual_add_f32 v211, v211, v215
	v_dual_add_f32 v216, v216, v220 :: v_dual_add_f32 v217, v217, v221
	v_dual_add_f32 v218, v218, v222 :: v_dual_add_f32 v219, v219, v223
	v_dual_add_f32 v208, v208, v216 :: v_dual_add_f32 v209, v209, v217
	v_dual_add_f32 v210, v210, v218 :: v_dual_add_f32 v211, v211, v219
	v_dual_add_f32 v224, v224, v228 :: v_dual_add_f32 v225, v225, v229
	v_dual_add_f32 v226, v226, v230 :: v_dual_add_f32 v227, v227, v231
	v_dual_add_f32 v232, v232, v236 :: v_dual_add_f32 v233, v233, v237
	v_dual_add_f32 v234, v234, v238 :: v_dual_add_f32 v235, v235, v239
	v_dual_add_f32 v224, v224, v232 :: v_dual_add_f32 v225, v225, v233
	v_dual_add_f32 v226, v226, v234 :: v_dual_add_f32 v227, v227, v235
	v_dual_add_f32 v240, v240, v244 :: v_dual_add_f32 v241, v241, v245
	v_dual_add_f32 v242, v242, v246 :: v_dual_add_f32 v243, v243, v247
	v_dual_add_f32 v248, v248, v252 :: v_dual_add_f32 v249, v249, v253
	v_dual_add_f32 v250, v250, v254 :: v_dual_add_f32 v251, v251, v255
	v_dual_add_f32 v240, v240, v248 :: v_dual_add_f32 v241, v241, v249
	v_dual_add_f32 v242, v242, v250 :: v_dual_add_f32 v243, v243, v251
	ds_swizzle_b32 v8, v176 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v9, v177 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v10, v178 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v11, v179 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v12, v192 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v13, v193 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v14, v194 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v15, v195 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v16, v208 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v17, v209 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v18, v210 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v19, v211 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v20, v224 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v21, v225 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v22, v226 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v23, v227 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v24, v240 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v25, v241 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v26, v242 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v27, v243 offset:swizzle(BITMASK_PERM,"1pppp")
	s_wait_dscnt 0x13
	v_add_f32_e32 v176, v176, v8
	s_wait_dscnt 0x12
	v_add_f32_e32 v177, v177, v9
	s_wait_dscnt 0x11
	v_add_f32_e32 v178, v178, v10
	s_wait_dscnt 0x10
	v_add_f32_e32 v179, v179, v11
	s_wait_dscnt 0xf
	v_add_f32_e32 v192, v192, v12
	s_wait_dscnt 0xe
	v_add_f32_e32 v193, v193, v13
	s_wait_dscnt 0xd
	v_add_f32_e32 v194, v194, v14
	s_wait_dscnt 0xc
	v_add_f32_e32 v195, v195, v15
	s_wait_dscnt 0xb
	v_add_f32_e32 v208, v208, v16
	s_wait_dscnt 0xa
	v_add_f32_e32 v209, v209, v17
	s_wait_dscnt 0x9
	v_add_f32_e32 v210, v210, v18
	s_wait_dscnt 0x8
	v_add_f32_e32 v211, v211, v19
	s_wait_dscnt 0x7
	v_add_f32_e32 v224, v224, v20
	s_wait_dscnt 0x6
	v_add_f32_e32 v225, v225, v21
	s_wait_dscnt 0x5
	v_add_f32_e32 v226, v226, v22
	s_wait_dscnt 0x4
	v_add_f32_e32 v227, v227, v23
	s_wait_dscnt 0x3
	v_add_f32_e32 v240, v240, v24
	s_wait_dscnt 0x2
	v_add_f32_e32 v241, v241, v25
	s_wait_dscnt 0x1
	v_add_f32_e32 v242, v242, v26
	s_wait_dscnt 0x0
	v_add_f32_e32 v243, v243, v27
	v_cmp_gt_u32_e32 vcc_lo, 24, v0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v4, 0, 8, vcc_lo
	v_add_lshl_u32 v4, v4, v0, 2
	ds_bpermute_b32 v8, v4, v176
	ds_bpermute_b32 v9, v4, v177
	ds_bpermute_b32 v10, v4, v178
	ds_bpermute_b32 v11, v4, v179
	ds_bpermute_b32 v12, v4, v192
	ds_bpermute_b32 v13, v4, v193
	ds_bpermute_b32 v14, v4, v194
	ds_bpermute_b32 v15, v4, v195
	ds_bpermute_b32 v16, v4, v208
	ds_bpermute_b32 v17, v4, v209
	ds_bpermute_b32 v18, v4, v210
	ds_bpermute_b32 v19, v4, v211
	ds_bpermute_b32 v20, v4, v224
	ds_bpermute_b32 v21, v4, v225
	ds_bpermute_b32 v22, v4, v226
	ds_bpermute_b32 v23, v4, v227
	ds_bpermute_b32 v24, v4, v240
	ds_bpermute_b32 v25, v4, v241
	ds_bpermute_b32 v26, v4, v242
	ds_bpermute_b32 v27, v4, v243
	s_wait_dscnt 0x13
	v_add_f32_e32 v176, v176, v8
	s_wait_dscnt 0x12
	v_add_f32_e32 v177, v177, v9
	s_wait_dscnt 0x11
	v_add_f32_e32 v178, v178, v10
	s_wait_dscnt 0x10
	v_add_f32_e32 v179, v179, v11
	s_wait_dscnt 0xf
	v_add_f32_e32 v192, v192, v12
	s_wait_dscnt 0xe
	v_add_f32_e32 v193, v193, v13
	s_wait_dscnt 0xd
	v_add_f32_e32 v194, v194, v14
	s_wait_dscnt 0xc
	v_add_f32_e32 v195, v195, v15
	s_wait_dscnt 0xb
	v_add_f32_e32 v208, v208, v16
	s_wait_dscnt 0xa
	v_add_f32_e32 v209, v209, v17
	s_wait_dscnt 0x9
	v_add_f32_e32 v210, v210, v18
	s_wait_dscnt 0x8
	v_add_f32_e32 v211, v211, v19
	s_wait_dscnt 0x7
	v_add_f32_e32 v224, v224, v20
	s_wait_dscnt 0x6
	v_add_f32_e32 v225, v225, v21
	s_wait_dscnt 0x5
	v_add_f32_e32 v226, v226, v22
	s_wait_dscnt 0x4
	v_add_f32_e32 v227, v227, v23
	s_wait_dscnt 0x3
	v_add_f32_e32 v240, v240, v24
	s_wait_dscnt 0x2
	v_add_f32_e32 v241, v241, v25
	s_wait_dscnt 0x1
	v_add_f32_e32 v242, v242, v26
	s_wait_dscnt 0x0
	v_add_f32_e32 v243, v243, v27
	v_cmp_gt_u32_e32 vcc_lo, 28, v0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v4, 0, 4, vcc_lo
	v_add_lshl_u32 v4, v4, v0, 2
	ds_bpermute_b32 v8, v4, v176
	ds_bpermute_b32 v9, v4, v177
	ds_bpermute_b32 v10, v4, v178
	ds_bpermute_b32 v11, v4, v179
	ds_bpermute_b32 v12, v4, v192
	ds_bpermute_b32 v13, v4, v193
	ds_bpermute_b32 v14, v4, v194
	ds_bpermute_b32 v15, v4, v195
	ds_bpermute_b32 v16, v4, v208
	ds_bpermute_b32 v17, v4, v209
	ds_bpermute_b32 v18, v4, v210
	ds_bpermute_b32 v19, v4, v211
	ds_bpermute_b32 v20, v4, v224
	ds_bpermute_b32 v21, v4, v225
	ds_bpermute_b32 v22, v4, v226
	ds_bpermute_b32 v23, v4, v227
	ds_bpermute_b32 v24, v4, v240
	ds_bpermute_b32 v25, v4, v241
	ds_bpermute_b32 v26, v4, v242
	ds_bpermute_b32 v27, v4, v243
	s_wait_dscnt 0x13
	v_add_f32_e32 v176, v176, v8
	s_wait_dscnt 0x12
	v_add_f32_e32 v177, v177, v9
	s_wait_dscnt 0x11
	v_add_f32_e32 v178, v178, v10
	s_wait_dscnt 0x10
	v_add_f32_e32 v179, v179, v11
	s_wait_dscnt 0xf
	v_add_f32_e32 v192, v192, v12
	s_wait_dscnt 0xe
	v_add_f32_e32 v193, v193, v13
	s_wait_dscnt 0xd
	v_add_f32_e32 v194, v194, v14
	s_wait_dscnt 0xc
	v_add_f32_e32 v195, v195, v15
	s_wait_dscnt 0xb
	v_add_f32_e32 v208, v208, v16
	s_wait_dscnt 0xa
	v_add_f32_e32 v209, v209, v17
	s_wait_dscnt 0x9
	v_add_f32_e32 v210, v210, v18
	s_wait_dscnt 0x8
	v_add_f32_e32 v211, v211, v19
	s_wait_dscnt 0x7
	v_add_f32_e32 v224, v224, v20
	s_wait_dscnt 0x6
	v_add_f32_e32 v225, v225, v21
	s_wait_dscnt 0x5
	v_add_f32_e32 v226, v226, v22
	s_wait_dscnt 0x4
	v_add_f32_e32 v227, v227, v23
	s_wait_dscnt 0x3
	v_add_f32_e32 v240, v240, v24
	s_wait_dscnt 0x2
	v_add_f32_e32 v241, v241, v25
	s_wait_dscnt 0x1
	v_add_f32_e32 v242, v242, v26
	s_wait_dscnt 0x0
	v_add_f32_e32 v243, v243, v27
	v_cmp_gt_u32_e32 vcc_lo, 30, v0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v4, 0, 2, vcc_lo
	v_add_lshl_u32 v4, v4, v0, 2
	ds_bpermute_b32 v8, v4, v176
	ds_bpermute_b32 v9, v4, v177
	ds_bpermute_b32 v10, v4, v178
	ds_bpermute_b32 v11, v4, v179
	ds_bpermute_b32 v12, v4, v192
	ds_bpermute_b32 v13, v4, v193
	ds_bpermute_b32 v14, v4, v194
	ds_bpermute_b32 v15, v4, v195
	ds_bpermute_b32 v16, v4, v208
	ds_bpermute_b32 v17, v4, v209
	ds_bpermute_b32 v18, v4, v210
	ds_bpermute_b32 v19, v4, v211
	ds_bpermute_b32 v20, v4, v224
	ds_bpermute_b32 v21, v4, v225
	ds_bpermute_b32 v22, v4, v226
	ds_bpermute_b32 v23, v4, v227
	ds_bpermute_b32 v24, v4, v240
	ds_bpermute_b32 v25, v4, v241
	ds_bpermute_b32 v26, v4, v242
	ds_bpermute_b32 v27, v4, v243
	s_wait_dscnt 0x13
	v_add_f32_e32 v176, v176, v8
	s_wait_dscnt 0x12
	v_add_f32_e32 v177, v177, v9
	s_wait_dscnt 0x11
	v_add_f32_e32 v178, v178, v10
	s_wait_dscnt 0x10
	v_add_f32_e32 v179, v179, v11
	s_wait_dscnt 0xf
	v_add_f32_e32 v192, v192, v12
	s_wait_dscnt 0xe
	v_add_f32_e32 v193, v193, v13
	s_wait_dscnt 0xd
	v_add_f32_e32 v194, v194, v14
	s_wait_dscnt 0xc
	v_add_f32_e32 v195, v195, v15
	s_wait_dscnt 0xb
	v_add_f32_e32 v208, v208, v16
	s_wait_dscnt 0xa
	v_add_f32_e32 v209, v209, v17
	s_wait_dscnt 0x9
	v_add_f32_e32 v210, v210, v18
	s_wait_dscnt 0x8
	v_add_f32_e32 v211, v211, v19
	s_wait_dscnt 0x7
	v_add_f32_e32 v224, v224, v20
	s_wait_dscnt 0x6
	v_add_f32_e32 v225, v225, v21
	s_wait_dscnt 0x5
	v_add_f32_e32 v226, v226, v22
	s_wait_dscnt 0x4
	v_add_f32_e32 v227, v227, v23
	s_wait_dscnt 0x3
	v_add_f32_e32 v240, v240, v24
	s_wait_dscnt 0x2
	v_add_f32_e32 v241, v241, v25
	s_wait_dscnt 0x1
	v_add_f32_e32 v242, v242, v26
	s_wait_dscnt 0x0
	v_add_f32_e32 v243, v243, v27
	v_cmp_gt_u32_e32 vcc_lo, 31, v0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v4, 0, 1, vcc_lo
	v_add_lshl_u32 v4, v4, v0, 2
	ds_bpermute_b32 v8, v4, v176
	ds_bpermute_b32 v9, v4, v177
	ds_bpermute_b32 v10, v4, v178
	ds_bpermute_b32 v11, v4, v179
	ds_bpermute_b32 v12, v4, v192
	ds_bpermute_b32 v13, v4, v193
	ds_bpermute_b32 v14, v4, v194
	ds_bpermute_b32 v15, v4, v195
	ds_bpermute_b32 v16, v4, v208
	ds_bpermute_b32 v17, v4, v209
	ds_bpermute_b32 v18, v4, v210
	ds_bpermute_b32 v19, v4, v211
	ds_bpermute_b32 v20, v4, v224
	ds_bpermute_b32 v21, v4, v225
	ds_bpermute_b32 v22, v4, v226
	ds_bpermute_b32 v23, v4, v227
	ds_bpermute_b32 v24, v4, v240
	ds_bpermute_b32 v25, v4, v241
	ds_bpermute_b32 v26, v4, v242
	ds_bpermute_b32 v27, v4, v243
	s_wait_dscnt 0x13
	v_add_f32_e32 v176, v176, v8
	s_wait_dscnt 0x12
	v_add_f32_e32 v177, v177, v9
	s_wait_dscnt 0x11
	v_add_f32_e32 v178, v178, v10
	s_wait_dscnt 0x10
	v_add_f32_e32 v179, v179, v11
	s_wait_dscnt 0xf
	v_add_f32_e32 v192, v192, v12
	s_wait_dscnt 0xe
	v_add_f32_e32 v193, v193, v13
	s_wait_dscnt 0xd
	v_add_f32_e32 v194, v194, v14
	s_wait_dscnt 0xc
	v_add_f32_e32 v195, v195, v15
	s_wait_dscnt 0xb
	v_add_f32_e32 v208, v208, v16
	s_wait_dscnt 0xa
	v_add_f32_e32 v209, v209, v17
	s_wait_dscnt 0x9
	v_add_f32_e32 v210, v210, v18
	s_wait_dscnt 0x8
	v_add_f32_e32 v211, v211, v19
	s_wait_dscnt 0x7
	v_add_f32_e32 v224, v224, v20
	s_wait_dscnt 0x6
	v_add_f32_e32 v225, v225, v21
	s_wait_dscnt 0x5
	v_add_f32_e32 v226, v226, v22
	s_wait_dscnt 0x4
	v_add_f32_e32 v227, v227, v23
	s_wait_dscnt 0x3
	v_add_f32_e32 v240, v240, v24
	s_wait_dscnt 0x2
	v_add_f32_e32 v241, v241, v25
	s_wait_dscnt 0x1
	v_add_f32_e32 v242, v242, v26
	s_wait_dscnt 0x0
	v_add_f32_e32 v243, v243, v27
	v_cmpx_eq_u32_e32 0, v0
	v_mov_b32_e32 v40, s42
	s_mul_i32 s40, s39, 1
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s40, s40, s42
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v41, s40
	s_mul_i32 s40, s39, 2
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s40, s40, s42
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v42, s40
	s_mul_i32 s40, s39, 3
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s40, s40, s42
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v43, s40
	s_mul_i32 s40, s39, 4
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s40, s40, s42
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v44, s40
	global_store_b32 v40, v176, s[8:9]
	global_store_b32 v41, v192, s[8:9]
	global_store_b32 v42, v208, s[8:9]
	global_store_b32 v43, v224, s[8:9]
	global_store_b32 v44, v240, s[8:9]
	s_cmp_le_i32 s13, 1
	s_cbranch_scc1 .Lpx_b5_stored
	global_store_b32 v40, v177, s[8:9] offset:4
	global_store_b32 v41, v193, s[8:9] offset:4
	global_store_b32 v42, v209, s[8:9] offset:4
	global_store_b32 v43, v225, s[8:9] offset:4
	global_store_b32 v44, v241, s[8:9] offset:4
	s_cmp_le_i32 s13, 2
	s_cbranch_scc1 .Lpx_b5_stored
	global_store_b32 v40, v178, s[8:9] offset:8
	global_store_b32 v41, v194, s[8:9] offset:8
	global_store_b32 v42, v210, s[8:9] offset:8
	global_store_b32 v43, v226, s[8:9] offset:8
	global_store_b32 v44, v242, s[8:9] offset:8
	s_cmp_le_i32 s13, 3
	s_cbranch_scc1 .Lpx_b5_stored
	global_store_b32 v40, v179, s[8:9] offset:12
	global_store_b32 v41, v195, s[8:9] offset:12
	global_store_b32 v42, v211, s[8:9] offset:12
	global_store_b32 v43, v227, s[8:9] offset:12
	global_store_b32 v44, v243, s[8:9] offset:12
	.Lpx_b5_stored:
	s_wait_storecnt 0x0
	s_branch .Lpx_end
	.Lpx_b6:
	v_mov_b32_e32 v160, 0
	v_mov_b32_e32 v161, 0
	v_mov_b32_e32 v162, 0
	v_mov_b32_e32 v163, 0
	v_mov_b32_e32 v164, 0
	v_mov_b32_e32 v165, 0
	v_mov_b32_e32 v166, 0
	v_mov_b32_e32 v167, 0
	v_mov_b32_e32 v168, 0
	v_mov_b32_e32 v169, 0
	v_mov_b32_e32 v170, 0
	v_mov_b32_e32 v171, 0
	v_mov_b32_e32 v172, 0
	v_mov_b32_e32 v173, 0
	v_mov_b32_e32 v174, 0
	v_mov_b32_e32 v175, 0
	v_mov_b32_e32 v176, 0
	v_mov_b32_e32 v177, 0
	v_mov_b32_e32 v178, 0
	v_mov_b32_e32 v179, 0
	v_mov_b32_e32 v180, 0
	v_mov_b32_e32 v181, 0
	v_mov_b32_e32 v182, 0
	v_mov_b32_e32 v183, 0
	v_mov_b32_e32 v184, 0
	v_mov_b32_e32 v185, 0
	v_mov_b32_e32 v186, 0
	v_mov_b32_e32 v187, 0
	v_mov_b32_e32 v188, 0
	v_mov_b32_e32 v189, 0
	v_mov_b32_e32 v190, 0
	v_mov_b32_e32 v191, 0
	v_mov_b32_e32 v192, 0
	v_mov_b32_e32 v193, 0
	v_mov_b32_e32 v194, 0
	v_mov_b32_e32 v195, 0
	v_mov_b32_e32 v196, 0
	v_mov_b32_e32 v197, 0
	v_mov_b32_e32 v198, 0
	v_mov_b32_e32 v199, 0
	v_mov_b32_e32 v200, 0
	v_mov_b32_e32 v201, 0
	v_mov_b32_e32 v202, 0
	v_mov_b32_e32 v203, 0
	v_mov_b32_e32 v204, 0
	v_mov_b32_e32 v205, 0
	v_mov_b32_e32 v206, 0
	v_mov_b32_e32 v207, 0
	v_mov_b32_e32 v208, 0
	v_mov_b32_e32 v209, 0
	v_mov_b32_e32 v210, 0
	v_mov_b32_e32 v211, 0
	v_mov_b32_e32 v212, 0
	v_mov_b32_e32 v213, 0
	v_mov_b32_e32 v214, 0
	v_mov_b32_e32 v215, 0
	v_mov_b32_e32 v216, 0
	v_mov_b32_e32 v217, 0
	v_mov_b32_e32 v218, 0
	v_mov_b32_e32 v219, 0
	v_mov_b32_e32 v220, 0
	v_mov_b32_e32 v221, 0
	v_mov_b32_e32 v222, 0
	v_mov_b32_e32 v223, 0
	v_mov_b32_e32 v224, 0
	v_mov_b32_e32 v225, 0
	v_mov_b32_e32 v226, 0
	v_mov_b32_e32 v227, 0
	v_mov_b32_e32 v228, 0
	v_mov_b32_e32 v229, 0
	v_mov_b32_e32 v230, 0
	v_mov_b32_e32 v231, 0
	v_mov_b32_e32 v232, 0
	v_mov_b32_e32 v233, 0
	v_mov_b32_e32 v234, 0
	v_mov_b32_e32 v235, 0
	v_mov_b32_e32 v236, 0
	v_mov_b32_e32 v237, 0
	v_mov_b32_e32 v238, 0
	v_mov_b32_e32 v239, 0
	v_mov_b32_e32 v240, 0
	v_mov_b32_e32 v241, 0
	v_mov_b32_e32 v242, 0
	v_mov_b32_e32 v243, 0
	v_mov_b32_e32 v244, 0
	v_mov_b32_e32 v245, 0
	v_mov_b32_e32 v246, 0
	v_mov_b32_e32 v247, 0
	v_mov_b32_e32 v248, 0
	v_mov_b32_e32 v249, 0
	v_mov_b32_e32 v250, 0
	v_mov_b32_e32 v251, 0
	v_mov_b32_e32 v252, 0
	v_mov_b32_e32 v253, 0
	v_mov_b32_e32 v254, 0
	v_mov_b32_e32 v255, 0
	s_lshr_b32 s36, s14, 2
	s_and_b32 s37, s14, 3
	s_cmp_eq_u32 s36, 0
	s_cbranch_scc1 .Lpx_b6_tail
	.Lpx_b6_quad:
	buffer_load_b32 v8, v1, s[20:23], s16 offen
	buffer_load_b32 v9, v1, s[20:23], s17 offen
	buffer_load_b32 v10, v1, s[20:23], s18 offen
	buffer_load_b32 v11, v1, s[20:23], s19 offen
	buffer_load_b32 v24, v2, s[20:23], s16 offen offset:8
	buffer_load_b32 v25, v2, s[20:23], s17 offen offset:8
	buffer_load_b32 v26, v2, s[20:23], s18 offen offset:8
	buffer_load_b32 v27, v2, s[20:23], s19 offen offset:8
	buffer_load_b128 v[80:83], v3, s[24:27], s28 offen
	buffer_load_b128 v[84:87], v3, s[24:27], s28 offen offset:16
	buffer_load_b128 v[88:91], v3, s[24:27], s29 offen
	buffer_load_b128 v[92:95], v3, s[24:27], s29 offen offset:16
	buffer_load_b128 v[96:99], v3, s[24:27], s30 offen
	buffer_load_b128 v[100:103], v3, s[24:27], s30 offen offset:16
	buffer_load_b128 v[104:107], v3, s[24:27], s31 offen
	buffer_load_b128 v[108:111], v3, s[24:27], s31 offen offset:16
	buffer_load_b128 v[112:115], v3, s[24:27], s32 offen
	buffer_load_b128 v[116:119], v3, s[24:27], s32 offen offset:16
	buffer_load_b128 v[120:123], v3, s[24:27], s33 offen
	buffer_load_b128 v[124:127], v3, s[24:27], s33 offen offset:16
	buffer_load_b32 v12, v1, s[20:23], s16 offen offset:136
	buffer_load_b32 v13, v1, s[20:23], s17 offen offset:136
	buffer_load_b32 v14, v1, s[20:23], s18 offen offset:136
	buffer_load_b32 v15, v1, s[20:23], s19 offen offset:136
	buffer_load_b32 v28, v2, s[20:23], s16 offen offset:144
	buffer_load_b32 v29, v2, s[20:23], s17 offen offset:144
	buffer_load_b32 v30, v2, s[20:23], s18 offen offset:144
	buffer_load_b32 v31, v2, s[20:23], s19 offen offset:144
	buffer_load_b32 v16, v1, s[20:23], s16 offen offset:272
	buffer_load_b32 v17, v1, s[20:23], s17 offen offset:272
	buffer_load_b32 v18, v1, s[20:23], s18 offen offset:272
	buffer_load_b32 v19, v1, s[20:23], s19 offen offset:272
	buffer_load_b32 v32, v2, s[20:23], s16 offen offset:280
	buffer_load_b32 v33, v2, s[20:23], s17 offen offset:280
	buffer_load_b32 v34, v2, s[20:23], s18 offen offset:280
	buffer_load_b32 v35, v2, s[20:23], s19 offen offset:280
	buffer_load_b32 v20, v1, s[20:23], s16 offen offset:408
	buffer_load_b32 v21, v1, s[20:23], s17 offen offset:408
	buffer_load_b32 v22, v1, s[20:23], s18 offen offset:408
	buffer_load_b32 v23, v1, s[20:23], s19 offen offset:408
	buffer_load_b32 v36, v2, s[20:23], s16 offen offset:416
	buffer_load_b32 v37, v2, s[20:23], s17 offen offset:416
	buffer_load_b32 v38, v2, s[20:23], s18 offen offset:416
	buffer_load_b32 v39, v2, s[20:23], s19 offen offset:416
	s_wait_loadcnt 0x27
	v_and_b32_e32 v46, 0xf0f0f0f, v24
	s_wait_loadcnt 0x26
	v_and_b32_e32 v56, 0xf0f0f0f, v25
	s_wait_loadcnt 0x25
	v_and_b32_e32 v66, 0xf0f0f0f, v26
	s_wait_loadcnt 0x24
	v_and_b32_e32 v76, 0xf0f0f0f, v27
	v_lshrrev_b32_e32 v47, 4, v24
	v_lshrrev_b32_e32 v57, 4, v25
	v_lshrrev_b32_e32 v67, 4, v26
	v_lshrrev_b32_e32 v77, 4, v27
	v_and_b32_e32 v47, 0xf0f0f0f, v47
	v_and_b32_e32 v57, 0xf0f0f0f, v57
	v_and_b32_e32 v67, 0xf0f0f0f, v67
	v_and_b32_e32 v77, 0xf0f0f0f, v77
	v_cvt_f32_ubyte0_e32 v40, v46
	v_cvt_f32_ubyte0_e32 v50, v56
	v_cvt_f32_ubyte0_e32 v60, v66
	v_cvt_f32_ubyte0_e32 v70, v76
	v_cvt_f32_ubyte1_e32 v42, v46
	v_cvt_f32_ubyte1_e32 v52, v56
	v_cvt_f32_ubyte1_e32 v62, v66
	v_cvt_f32_ubyte1_e32 v72, v76
	v_cvt_f32_ubyte2_e32 v44, v46
	v_cvt_f32_ubyte2_e32 v54, v56
	v_cvt_f32_ubyte2_e32 v64, v66
	v_cvt_f32_ubyte2_e32 v74, v76
	v_cvt_f32_ubyte3_e32 v46, v46
	v_cvt_f32_ubyte3_e32 v56, v56
	v_cvt_f32_ubyte3_e32 v66, v66
	v_cvt_f32_ubyte3_e32 v76, v76
	v_cvt_f32_ubyte0_e32 v41, v47
	v_cvt_f32_ubyte0_e32 v51, v57
	v_cvt_f32_ubyte0_e32 v61, v67
	v_cvt_f32_ubyte0_e32 v71, v77
	v_cvt_f32_ubyte1_e32 v43, v47
	v_cvt_f32_ubyte1_e32 v53, v57
	v_cvt_f32_ubyte1_e32 v63, v67
	v_cvt_f32_ubyte1_e32 v73, v77
	v_cvt_f32_ubyte2_e32 v45, v47
	v_cvt_f32_ubyte2_e32 v55, v57
	v_cvt_f32_ubyte2_e32 v65, v67
	v_cvt_f32_ubyte2_e32 v75, v77
	v_cvt_f32_ubyte3_e32 v47, v47
	v_cvt_f32_ubyte3_e32 v57, v57
	v_cvt_f32_ubyte3_e32 v67, v67
	v_cvt_f32_ubyte3_e32 v77, v77
	v_fma_mix_f32 v40, v8, v40, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v9, v50, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v10, v60, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v70, v11, v70, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v41, v8, v41, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v9, v51, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v10, v61, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v71, v11, v71, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v42, v8, v42, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v52, v9, v52, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v62, v10, v62, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v72, v11, v72, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v43, v8, v43, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v53, v9, v53, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v63, v10, v63, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v73, v11, v73, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v44, v8, v44, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v54, v9, v54, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v64, v10, v64, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v74, v11, v74, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v8, v45, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v9, v55, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v65, v10, v65, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v75, v11, v75, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v8, v46, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v9, v56, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v66, v10, v66, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v76, v11, v76, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v8, v47, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v9, v57, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v67, v10, v67, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v77, v11, v77, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	buffer_load_b128 v[128:131], v3, s[24:27], s28 offen offset:1024
	buffer_load_b128 v[132:135], v3, s[24:27], s28 offen offset:1040
	buffer_load_b128 v[136:139], v3, s[24:27], s29 offen offset:1024
	buffer_load_b128 v[140:143], v3, s[24:27], s29 offen offset:1040
	buffer_load_b128 v[144:147], v3, s[24:27], s30 offen offset:1024
	buffer_load_b128 v[148:151], v3, s[24:27], s30 offen offset:1040
	buffer_load_b128 v[152:155], v3, s[24:27], s31 offen offset:1024
	buffer_load_b128 v[156:159], v3, s[24:27], s31 offen offset:1040
	s_wait_loadcnt 0x2b
	v_dual_mul_f32 v48, v41, v81 :: v_dual_mul_f32 v49, v51, v81
	v_dual_mul_f32 v58, v61, v81 :: v_dual_mul_f32 v59, v71, v81
	s_wait_loadcnt 0x29
	v_dual_mul_f32 v68, v41, v89 :: v_dual_mul_f32 v69, v51, v89
	v_dual_mul_f32 v78, v61, v89 :: v_dual_mul_f32 v79, v71, v89
	v_dual_fmac_f32 v48, v40, v80 :: v_dual_fmac_f32 v49, v50, v80
	v_dual_fmac_f32 v58, v60, v80 :: v_dual_fmac_f32 v59, v70, v80
	v_dual_fmac_f32 v68, v40, v88 :: v_dual_fmac_f32 v69, v50, v88
	v_dual_fmac_f32 v78, v60, v88 :: v_dual_fmac_f32 v79, v70, v88
	v_dual_fmac_f32 v48, v42, v82 :: v_dual_fmac_f32 v49, v52, v82
	v_dual_fmac_f32 v58, v62, v82 :: v_dual_fmac_f32 v59, v72, v82
	v_dual_fmac_f32 v68, v42, v90 :: v_dual_fmac_f32 v69, v52, v90
	v_dual_fmac_f32 v78, v62, v90 :: v_dual_fmac_f32 v79, v72, v90
	v_dual_fmac_f32 v48, v43, v83 :: v_dual_fmac_f32 v49, v53, v83
	v_dual_fmac_f32 v58, v63, v83 :: v_dual_fmac_f32 v59, v73, v83
	v_dual_fmac_f32 v68, v43, v91 :: v_dual_fmac_f32 v69, v53, v91
	v_dual_fmac_f32 v78, v63, v91 :: v_dual_fmac_f32 v79, v73, v91
	v_dual_fmac_f32 v48, v44, v84 :: v_dual_fmac_f32 v49, v54, v84
	v_dual_fmac_f32 v58, v64, v84 :: v_dual_fmac_f32 v59, v74, v84
	s_wait_loadcnt 0x28
	v_dual_fmac_f32 v68, v44, v92 :: v_dual_fmac_f32 v69, v54, v92
	v_dual_fmac_f32 v78, v64, v92 :: v_dual_fmac_f32 v79, v74, v92
	v_dual_fmac_f32 v48, v45, v85 :: v_dual_fmac_f32 v49, v55, v85
	v_dual_fmac_f32 v58, v65, v85 :: v_dual_fmac_f32 v59, v75, v85
	v_dual_fmac_f32 v68, v45, v93 :: v_dual_fmac_f32 v69, v55, v93
	v_dual_fmac_f32 v78, v65, v93 :: v_dual_fmac_f32 v79, v75, v93
	v_dual_fmac_f32 v48, v46, v86 :: v_dual_fmac_f32 v49, v56, v86
	v_dual_fmac_f32 v58, v66, v86 :: v_dual_fmac_f32 v59, v76, v86
	v_dual_fmac_f32 v68, v46, v94 :: v_dual_fmac_f32 v69, v56, v94
	v_dual_fmac_f32 v78, v66, v94 :: v_dual_fmac_f32 v79, v76, v94
	v_dual_fmac_f32 v48, v47, v87 :: v_dual_fmac_f32 v49, v57, v87
	v_dual_fmac_f32 v58, v67, v87 :: v_dual_fmac_f32 v59, v77, v87
	v_dual_fmac_f32 v68, v47, v95 :: v_dual_fmac_f32 v69, v57, v95
	v_dual_fmac_f32 v78, v67, v95 :: v_dual_fmac_f32 v79, v77, v95
	v_dual_add_f32 v160, v160, v48 :: v_dual_add_f32 v161, v161, v49
	v_dual_add_f32 v162, v162, v58 :: v_dual_add_f32 v163, v163, v59
	v_dual_add_f32 v176, v176, v68 :: v_dual_add_f32 v177, v177, v69
	v_dual_add_f32 v178, v178, v78 :: v_dual_add_f32 v179, v179, v79
	buffer_load_b128 v[80:83], v3, s[24:27], s32 offen offset:1024
	buffer_load_b128 v[84:87], v3, s[24:27], s32 offen offset:1040
	buffer_load_b128 v[88:91], v3, s[24:27], s33 offen offset:1024
	buffer_load_b128 v[92:95], v3, s[24:27], s33 offen offset:1040
	s_wait_loadcnt 0x2b
	v_dual_mul_f32 v48, v41, v97 :: v_dual_mul_f32 v49, v51, v97
	v_dual_mul_f32 v58, v61, v97 :: v_dual_mul_f32 v59, v71, v97
	s_wait_loadcnt 0x29
	v_dual_mul_f32 v68, v41, v105 :: v_dual_mul_f32 v69, v51, v105
	v_dual_mul_f32 v78, v61, v105 :: v_dual_mul_f32 v79, v71, v105
	v_dual_fmac_f32 v48, v40, v96 :: v_dual_fmac_f32 v49, v50, v96
	v_dual_fmac_f32 v58, v60, v96 :: v_dual_fmac_f32 v59, v70, v96
	v_dual_fmac_f32 v68, v40, v104 :: v_dual_fmac_f32 v69, v50, v104
	v_dual_fmac_f32 v78, v60, v104 :: v_dual_fmac_f32 v79, v70, v104
	v_dual_fmac_f32 v48, v42, v98 :: v_dual_fmac_f32 v49, v52, v98
	v_dual_fmac_f32 v58, v62, v98 :: v_dual_fmac_f32 v59, v72, v98
	v_dual_fmac_f32 v68, v42, v106 :: v_dual_fmac_f32 v69, v52, v106
	v_dual_fmac_f32 v78, v62, v106 :: v_dual_fmac_f32 v79, v72, v106
	v_dual_fmac_f32 v48, v43, v99 :: v_dual_fmac_f32 v49, v53, v99
	v_dual_fmac_f32 v58, v63, v99 :: v_dual_fmac_f32 v59, v73, v99
	v_dual_fmac_f32 v68, v43, v107 :: v_dual_fmac_f32 v69, v53, v107
	v_dual_fmac_f32 v78, v63, v107 :: v_dual_fmac_f32 v79, v73, v107
	v_dual_fmac_f32 v48, v44, v100 :: v_dual_fmac_f32 v49, v54, v100
	v_dual_fmac_f32 v58, v64, v100 :: v_dual_fmac_f32 v59, v74, v100
	s_wait_loadcnt 0x28
	v_dual_fmac_f32 v68, v44, v108 :: v_dual_fmac_f32 v69, v54, v108
	v_dual_fmac_f32 v78, v64, v108 :: v_dual_fmac_f32 v79, v74, v108
	v_dual_fmac_f32 v48, v45, v101 :: v_dual_fmac_f32 v49, v55, v101
	v_dual_fmac_f32 v58, v65, v101 :: v_dual_fmac_f32 v59, v75, v101
	v_dual_fmac_f32 v68, v45, v109 :: v_dual_fmac_f32 v69, v55, v109
	v_dual_fmac_f32 v78, v65, v109 :: v_dual_fmac_f32 v79, v75, v109
	v_dual_fmac_f32 v48, v46, v102 :: v_dual_fmac_f32 v49, v56, v102
	v_dual_fmac_f32 v58, v66, v102 :: v_dual_fmac_f32 v59, v76, v102
	v_dual_fmac_f32 v68, v46, v110 :: v_dual_fmac_f32 v69, v56, v110
	v_dual_fmac_f32 v78, v66, v110 :: v_dual_fmac_f32 v79, v76, v110
	v_dual_fmac_f32 v48, v47, v103 :: v_dual_fmac_f32 v49, v57, v103
	v_dual_fmac_f32 v58, v67, v103 :: v_dual_fmac_f32 v59, v77, v103
	v_dual_fmac_f32 v68, v47, v111 :: v_dual_fmac_f32 v69, v57, v111
	v_dual_fmac_f32 v78, v67, v111 :: v_dual_fmac_f32 v79, v77, v111
	v_dual_add_f32 v192, v192, v48 :: v_dual_add_f32 v193, v193, v49
	v_dual_add_f32 v194, v194, v58 :: v_dual_add_f32 v195, v195, v59
	v_dual_add_f32 v208, v208, v68 :: v_dual_add_f32 v209, v209, v69
	v_dual_add_f32 v210, v210, v78 :: v_dual_add_f32 v211, v211, v79
	buffer_load_b128 v[96:99], v3, s[24:27], s28 offen offset:2048
	buffer_load_b128 v[100:103], v3, s[24:27], s28 offen offset:2064
	buffer_load_b128 v[104:107], v3, s[24:27], s29 offen offset:2048
	buffer_load_b128 v[108:111], v3, s[24:27], s29 offen offset:2064
	s_wait_loadcnt 0x2b
	v_dual_mul_f32 v48, v41, v113 :: v_dual_mul_f32 v49, v51, v113
	v_dual_mul_f32 v58, v61, v113 :: v_dual_mul_f32 v59, v71, v113
	s_wait_loadcnt 0x29
	v_dual_mul_f32 v68, v41, v121 :: v_dual_mul_f32 v69, v51, v121
	v_dual_mul_f32 v78, v61, v121 :: v_dual_mul_f32 v79, v71, v121
	v_dual_fmac_f32 v48, v40, v112 :: v_dual_fmac_f32 v49, v50, v112
	v_dual_fmac_f32 v58, v60, v112 :: v_dual_fmac_f32 v59, v70, v112
	v_dual_fmac_f32 v68, v40, v120 :: v_dual_fmac_f32 v69, v50, v120
	v_dual_fmac_f32 v78, v60, v120 :: v_dual_fmac_f32 v79, v70, v120
	v_dual_fmac_f32 v48, v42, v114 :: v_dual_fmac_f32 v49, v52, v114
	v_dual_fmac_f32 v58, v62, v114 :: v_dual_fmac_f32 v59, v72, v114
	v_dual_fmac_f32 v68, v42, v122 :: v_dual_fmac_f32 v69, v52, v122
	v_dual_fmac_f32 v78, v62, v122 :: v_dual_fmac_f32 v79, v72, v122
	v_dual_fmac_f32 v48, v43, v115 :: v_dual_fmac_f32 v49, v53, v115
	v_dual_fmac_f32 v58, v63, v115 :: v_dual_fmac_f32 v59, v73, v115
	v_dual_fmac_f32 v68, v43, v123 :: v_dual_fmac_f32 v69, v53, v123
	v_dual_fmac_f32 v78, v63, v123 :: v_dual_fmac_f32 v79, v73, v123
	v_dual_fmac_f32 v48, v44, v116 :: v_dual_fmac_f32 v49, v54, v116
	v_dual_fmac_f32 v58, v64, v116 :: v_dual_fmac_f32 v59, v74, v116
	s_wait_loadcnt 0x28
	v_dual_fmac_f32 v68, v44, v124 :: v_dual_fmac_f32 v69, v54, v124
	v_dual_fmac_f32 v78, v64, v124 :: v_dual_fmac_f32 v79, v74, v124
	v_dual_fmac_f32 v48, v45, v117 :: v_dual_fmac_f32 v49, v55, v117
	v_dual_fmac_f32 v58, v65, v117 :: v_dual_fmac_f32 v59, v75, v117
	v_dual_fmac_f32 v68, v45, v125 :: v_dual_fmac_f32 v69, v55, v125
	v_dual_fmac_f32 v78, v65, v125 :: v_dual_fmac_f32 v79, v75, v125
	v_dual_fmac_f32 v48, v46, v118 :: v_dual_fmac_f32 v49, v56, v118
	v_dual_fmac_f32 v58, v66, v118 :: v_dual_fmac_f32 v59, v76, v118
	v_dual_fmac_f32 v68, v46, v126 :: v_dual_fmac_f32 v69, v56, v126
	v_dual_fmac_f32 v78, v66, v126 :: v_dual_fmac_f32 v79, v76, v126
	v_dual_fmac_f32 v48, v47, v119 :: v_dual_fmac_f32 v49, v57, v119
	v_dual_fmac_f32 v58, v67, v119 :: v_dual_fmac_f32 v59, v77, v119
	v_dual_fmac_f32 v68, v47, v127 :: v_dual_fmac_f32 v69, v57, v127
	v_dual_fmac_f32 v78, v67, v127 :: v_dual_fmac_f32 v79, v77, v127
	v_dual_add_f32 v224, v224, v48 :: v_dual_add_f32 v225, v225, v49
	v_dual_add_f32 v226, v226, v58 :: v_dual_add_f32 v227, v227, v59
	v_dual_add_f32 v240, v240, v68 :: v_dual_add_f32 v241, v241, v69
	v_dual_add_f32 v242, v242, v78 :: v_dual_add_f32 v243, v243, v79
	s_wait_loadcnt 0x23
	v_and_b32_e32 v46, 0xf0f0f0f, v28
	s_wait_loadcnt 0x22
	v_and_b32_e32 v56, 0xf0f0f0f, v29
	s_wait_loadcnt 0x21
	v_and_b32_e32 v66, 0xf0f0f0f, v30
	s_wait_loadcnt 0x20
	v_and_b32_e32 v76, 0xf0f0f0f, v31
	v_lshrrev_b32_e32 v47, 4, v28
	v_lshrrev_b32_e32 v57, 4, v29
	v_lshrrev_b32_e32 v67, 4, v30
	v_lshrrev_b32_e32 v77, 4, v31
	v_and_b32_e32 v47, 0xf0f0f0f, v47
	v_and_b32_e32 v57, 0xf0f0f0f, v57
	v_and_b32_e32 v67, 0xf0f0f0f, v67
	v_and_b32_e32 v77, 0xf0f0f0f, v77
	v_cvt_f32_ubyte0_e32 v40, v46
	v_cvt_f32_ubyte0_e32 v50, v56
	v_cvt_f32_ubyte0_e32 v60, v66
	v_cvt_f32_ubyte0_e32 v70, v76
	v_cvt_f32_ubyte1_e32 v42, v46
	v_cvt_f32_ubyte1_e32 v52, v56
	v_cvt_f32_ubyte1_e32 v62, v66
	v_cvt_f32_ubyte1_e32 v72, v76
	v_cvt_f32_ubyte2_e32 v44, v46
	v_cvt_f32_ubyte2_e32 v54, v56
	v_cvt_f32_ubyte2_e32 v64, v66
	v_cvt_f32_ubyte2_e32 v74, v76
	v_cvt_f32_ubyte3_e32 v46, v46
	v_cvt_f32_ubyte3_e32 v56, v56
	v_cvt_f32_ubyte3_e32 v66, v66
	v_cvt_f32_ubyte3_e32 v76, v76
	v_cvt_f32_ubyte0_e32 v41, v47
	v_cvt_f32_ubyte0_e32 v51, v57
	v_cvt_f32_ubyte0_e32 v61, v67
	v_cvt_f32_ubyte0_e32 v71, v77
	v_cvt_f32_ubyte1_e32 v43, v47
	v_cvt_f32_ubyte1_e32 v53, v57
	v_cvt_f32_ubyte1_e32 v63, v67
	v_cvt_f32_ubyte1_e32 v73, v77
	v_cvt_f32_ubyte2_e32 v45, v47
	v_cvt_f32_ubyte2_e32 v55, v57
	v_cvt_f32_ubyte2_e32 v65, v67
	v_cvt_f32_ubyte2_e32 v75, v77
	v_cvt_f32_ubyte3_e32 v47, v47
	v_cvt_f32_ubyte3_e32 v57, v57
	v_cvt_f32_ubyte3_e32 v67, v67
	v_cvt_f32_ubyte3_e32 v77, v77
	v_fma_mix_f32 v40, v12, v40, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v13, v50, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v14, v60, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v70, v15, v70, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v41, v12, v41, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v13, v51, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v14, v61, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v71, v15, v71, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v42, v12, v42, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v52, v13, v52, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v62, v14, v62, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v72, v15, v72, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v43, v12, v43, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v53, v13, v53, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v63, v14, v63, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v73, v15, v73, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v44, v12, v44, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v54, v13, v54, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v64, v14, v64, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v74, v15, v74, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v12, v45, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v13, v55, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v65, v14, v65, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v75, v15, v75, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v12, v46, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v13, v56, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v66, v14, v66, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v76, v15, v76, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v12, v47, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v13, v57, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v67, v14, v67, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v77, v15, v77, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	buffer_load_b128 v[112:115], v3, s[24:27], s30 offen offset:2048
	buffer_load_b128 v[116:119], v3, s[24:27], s30 offen offset:2064
	buffer_load_b128 v[120:123], v3, s[24:27], s31 offen offset:2048
	buffer_load_b128 v[124:127], v3, s[24:27], s31 offen offset:2064
	s_wait_loadcnt 0x13
	v_dual_mul_f32 v48, v41, v129 :: v_dual_mul_f32 v49, v51, v129
	v_dual_mul_f32 v58, v61, v129 :: v_dual_mul_f32 v59, v71, v129
	s_wait_loadcnt 0x11
	v_dual_mul_f32 v68, v41, v137 :: v_dual_mul_f32 v69, v51, v137
	v_dual_mul_f32 v78, v61, v137 :: v_dual_mul_f32 v79, v71, v137
	v_dual_fmac_f32 v48, v40, v128 :: v_dual_fmac_f32 v49, v50, v128
	v_dual_fmac_f32 v58, v60, v128 :: v_dual_fmac_f32 v59, v70, v128
	v_dual_fmac_f32 v68, v40, v136 :: v_dual_fmac_f32 v69, v50, v136
	v_dual_fmac_f32 v78, v60, v136 :: v_dual_fmac_f32 v79, v70, v136
	v_dual_fmac_f32 v48, v42, v130 :: v_dual_fmac_f32 v49, v52, v130
	v_dual_fmac_f32 v58, v62, v130 :: v_dual_fmac_f32 v59, v72, v130
	v_dual_fmac_f32 v68, v42, v138 :: v_dual_fmac_f32 v69, v52, v138
	v_dual_fmac_f32 v78, v62, v138 :: v_dual_fmac_f32 v79, v72, v138
	v_dual_fmac_f32 v48, v43, v131 :: v_dual_fmac_f32 v49, v53, v131
	v_dual_fmac_f32 v58, v63, v131 :: v_dual_fmac_f32 v59, v73, v131
	v_dual_fmac_f32 v68, v43, v139 :: v_dual_fmac_f32 v69, v53, v139
	v_dual_fmac_f32 v78, v63, v139 :: v_dual_fmac_f32 v79, v73, v139
	v_dual_fmac_f32 v48, v44, v132 :: v_dual_fmac_f32 v49, v54, v132
	v_dual_fmac_f32 v58, v64, v132 :: v_dual_fmac_f32 v59, v74, v132
	s_wait_loadcnt 0x10
	v_dual_fmac_f32 v68, v44, v140 :: v_dual_fmac_f32 v69, v54, v140
	v_dual_fmac_f32 v78, v64, v140 :: v_dual_fmac_f32 v79, v74, v140
	v_dual_fmac_f32 v48, v45, v133 :: v_dual_fmac_f32 v49, v55, v133
	v_dual_fmac_f32 v58, v65, v133 :: v_dual_fmac_f32 v59, v75, v133
	v_dual_fmac_f32 v68, v45, v141 :: v_dual_fmac_f32 v69, v55, v141
	v_dual_fmac_f32 v78, v65, v141 :: v_dual_fmac_f32 v79, v75, v141
	v_dual_fmac_f32 v48, v46, v134 :: v_dual_fmac_f32 v49, v56, v134
	v_dual_fmac_f32 v58, v66, v134 :: v_dual_fmac_f32 v59, v76, v134
	v_dual_fmac_f32 v68, v46, v142 :: v_dual_fmac_f32 v69, v56, v142
	v_dual_fmac_f32 v78, v66, v142 :: v_dual_fmac_f32 v79, v76, v142
	v_dual_fmac_f32 v48, v47, v135 :: v_dual_fmac_f32 v49, v57, v135
	v_dual_fmac_f32 v58, v67, v135 :: v_dual_fmac_f32 v59, v77, v135
	v_dual_fmac_f32 v68, v47, v143 :: v_dual_fmac_f32 v69, v57, v143
	v_dual_fmac_f32 v78, v67, v143 :: v_dual_fmac_f32 v79, v77, v143
	v_dual_add_f32 v164, v164, v48 :: v_dual_add_f32 v165, v165, v49
	v_dual_add_f32 v166, v166, v58 :: v_dual_add_f32 v167, v167, v59
	v_dual_add_f32 v180, v180, v68 :: v_dual_add_f32 v181, v181, v69
	v_dual_add_f32 v182, v182, v78 :: v_dual_add_f32 v183, v183, v79
	buffer_load_b128 v[128:131], v3, s[24:27], s32 offen offset:2048
	buffer_load_b128 v[132:135], v3, s[24:27], s32 offen offset:2064
	buffer_load_b128 v[136:139], v3, s[24:27], s33 offen offset:2048
	buffer_load_b128 v[140:143], v3, s[24:27], s33 offen offset:2064
	s_wait_loadcnt 0x13
	v_dual_mul_f32 v48, v41, v145 :: v_dual_mul_f32 v49, v51, v145
	v_dual_mul_f32 v58, v61, v145 :: v_dual_mul_f32 v59, v71, v145
	s_wait_loadcnt 0x11
	v_dual_mul_f32 v68, v41, v153 :: v_dual_mul_f32 v69, v51, v153
	v_dual_mul_f32 v78, v61, v153 :: v_dual_mul_f32 v79, v71, v153
	v_dual_fmac_f32 v48, v40, v144 :: v_dual_fmac_f32 v49, v50, v144
	v_dual_fmac_f32 v58, v60, v144 :: v_dual_fmac_f32 v59, v70, v144
	v_dual_fmac_f32 v68, v40, v152 :: v_dual_fmac_f32 v69, v50, v152
	v_dual_fmac_f32 v78, v60, v152 :: v_dual_fmac_f32 v79, v70, v152
	v_dual_fmac_f32 v48, v42, v146 :: v_dual_fmac_f32 v49, v52, v146
	v_dual_fmac_f32 v58, v62, v146 :: v_dual_fmac_f32 v59, v72, v146
	v_dual_fmac_f32 v68, v42, v154 :: v_dual_fmac_f32 v69, v52, v154
	v_dual_fmac_f32 v78, v62, v154 :: v_dual_fmac_f32 v79, v72, v154
	v_dual_fmac_f32 v48, v43, v147 :: v_dual_fmac_f32 v49, v53, v147
	v_dual_fmac_f32 v58, v63, v147 :: v_dual_fmac_f32 v59, v73, v147
	v_dual_fmac_f32 v68, v43, v155 :: v_dual_fmac_f32 v69, v53, v155
	v_dual_fmac_f32 v78, v63, v155 :: v_dual_fmac_f32 v79, v73, v155
	v_dual_fmac_f32 v48, v44, v148 :: v_dual_fmac_f32 v49, v54, v148
	v_dual_fmac_f32 v58, v64, v148 :: v_dual_fmac_f32 v59, v74, v148
	s_wait_loadcnt 0x10
	v_dual_fmac_f32 v68, v44, v156 :: v_dual_fmac_f32 v69, v54, v156
	v_dual_fmac_f32 v78, v64, v156 :: v_dual_fmac_f32 v79, v74, v156
	v_dual_fmac_f32 v48, v45, v149 :: v_dual_fmac_f32 v49, v55, v149
	v_dual_fmac_f32 v58, v65, v149 :: v_dual_fmac_f32 v59, v75, v149
	v_dual_fmac_f32 v68, v45, v157 :: v_dual_fmac_f32 v69, v55, v157
	v_dual_fmac_f32 v78, v65, v157 :: v_dual_fmac_f32 v79, v75, v157
	v_dual_fmac_f32 v48, v46, v150 :: v_dual_fmac_f32 v49, v56, v150
	v_dual_fmac_f32 v58, v66, v150 :: v_dual_fmac_f32 v59, v76, v150
	v_dual_fmac_f32 v68, v46, v158 :: v_dual_fmac_f32 v69, v56, v158
	v_dual_fmac_f32 v78, v66, v158 :: v_dual_fmac_f32 v79, v76, v158
	v_dual_fmac_f32 v48, v47, v151 :: v_dual_fmac_f32 v49, v57, v151
	v_dual_fmac_f32 v58, v67, v151 :: v_dual_fmac_f32 v59, v77, v151
	v_dual_fmac_f32 v68, v47, v159 :: v_dual_fmac_f32 v69, v57, v159
	v_dual_fmac_f32 v78, v67, v159 :: v_dual_fmac_f32 v79, v77, v159
	v_dual_add_f32 v196, v196, v48 :: v_dual_add_f32 v197, v197, v49
	v_dual_add_f32 v198, v198, v58 :: v_dual_add_f32 v199, v199, v59
	v_dual_add_f32 v212, v212, v68 :: v_dual_add_f32 v213, v213, v69
	v_dual_add_f32 v214, v214, v78 :: v_dual_add_f32 v215, v215, v79
	buffer_load_b128 v[144:147], v3, s[24:27], s28 offen offset:3072
	buffer_load_b128 v[148:151], v3, s[24:27], s28 offen offset:3088
	buffer_load_b128 v[152:155], v3, s[24:27], s29 offen offset:3072
	buffer_load_b128 v[156:159], v3, s[24:27], s29 offen offset:3088
	s_wait_loadcnt 0x13
	v_dual_mul_f32 v48, v41, v81 :: v_dual_mul_f32 v49, v51, v81
	v_dual_mul_f32 v58, v61, v81 :: v_dual_mul_f32 v59, v71, v81
	s_wait_loadcnt 0x11
	v_dual_mul_f32 v68, v41, v89 :: v_dual_mul_f32 v69, v51, v89
	v_dual_mul_f32 v78, v61, v89 :: v_dual_mul_f32 v79, v71, v89
	v_dual_fmac_f32 v48, v40, v80 :: v_dual_fmac_f32 v49, v50, v80
	v_dual_fmac_f32 v58, v60, v80 :: v_dual_fmac_f32 v59, v70, v80
	v_dual_fmac_f32 v68, v40, v88 :: v_dual_fmac_f32 v69, v50, v88
	v_dual_fmac_f32 v78, v60, v88 :: v_dual_fmac_f32 v79, v70, v88
	v_dual_fmac_f32 v48, v42, v82 :: v_dual_fmac_f32 v49, v52, v82
	v_dual_fmac_f32 v58, v62, v82 :: v_dual_fmac_f32 v59, v72, v82
	v_dual_fmac_f32 v68, v42, v90 :: v_dual_fmac_f32 v69, v52, v90
	v_dual_fmac_f32 v78, v62, v90 :: v_dual_fmac_f32 v79, v72, v90
	v_dual_fmac_f32 v48, v43, v83 :: v_dual_fmac_f32 v49, v53, v83
	v_dual_fmac_f32 v58, v63, v83 :: v_dual_fmac_f32 v59, v73, v83
	v_dual_fmac_f32 v68, v43, v91 :: v_dual_fmac_f32 v69, v53, v91
	v_dual_fmac_f32 v78, v63, v91 :: v_dual_fmac_f32 v79, v73, v91
	v_dual_fmac_f32 v48, v44, v84 :: v_dual_fmac_f32 v49, v54, v84
	v_dual_fmac_f32 v58, v64, v84 :: v_dual_fmac_f32 v59, v74, v84
	s_wait_loadcnt 0x10
	v_dual_fmac_f32 v68, v44, v92 :: v_dual_fmac_f32 v69, v54, v92
	v_dual_fmac_f32 v78, v64, v92 :: v_dual_fmac_f32 v79, v74, v92
	v_dual_fmac_f32 v48, v45, v85 :: v_dual_fmac_f32 v49, v55, v85
	v_dual_fmac_f32 v58, v65, v85 :: v_dual_fmac_f32 v59, v75, v85
	v_dual_fmac_f32 v68, v45, v93 :: v_dual_fmac_f32 v69, v55, v93
	v_dual_fmac_f32 v78, v65, v93 :: v_dual_fmac_f32 v79, v75, v93
	v_dual_fmac_f32 v48, v46, v86 :: v_dual_fmac_f32 v49, v56, v86
	v_dual_fmac_f32 v58, v66, v86 :: v_dual_fmac_f32 v59, v76, v86
	v_dual_fmac_f32 v68, v46, v94 :: v_dual_fmac_f32 v69, v56, v94
	v_dual_fmac_f32 v78, v66, v94 :: v_dual_fmac_f32 v79, v76, v94
	v_dual_fmac_f32 v48, v47, v87 :: v_dual_fmac_f32 v49, v57, v87
	v_dual_fmac_f32 v58, v67, v87 :: v_dual_fmac_f32 v59, v77, v87
	v_dual_fmac_f32 v68, v47, v95 :: v_dual_fmac_f32 v69, v57, v95
	v_dual_fmac_f32 v78, v67, v95 :: v_dual_fmac_f32 v79, v77, v95
	v_dual_add_f32 v228, v228, v48 :: v_dual_add_f32 v229, v229, v49
	v_dual_add_f32 v230, v230, v58 :: v_dual_add_f32 v231, v231, v59
	v_dual_add_f32 v244, v244, v68 :: v_dual_add_f32 v245, v245, v69
	v_dual_add_f32 v246, v246, v78 :: v_dual_add_f32 v247, v247, v79
	v_and_b32_e32 v46, 0xf0f0f0f, v32
	v_and_b32_e32 v56, 0xf0f0f0f, v33
	v_and_b32_e32 v66, 0xf0f0f0f, v34
	v_and_b32_e32 v76, 0xf0f0f0f, v35
	v_lshrrev_b32_e32 v47, 4, v32
	v_lshrrev_b32_e32 v57, 4, v33
	v_lshrrev_b32_e32 v67, 4, v34
	v_lshrrev_b32_e32 v77, 4, v35
	v_and_b32_e32 v47, 0xf0f0f0f, v47
	v_and_b32_e32 v57, 0xf0f0f0f, v57
	v_and_b32_e32 v67, 0xf0f0f0f, v67
	v_and_b32_e32 v77, 0xf0f0f0f, v77
	v_cvt_f32_ubyte0_e32 v40, v46
	v_cvt_f32_ubyte0_e32 v50, v56
	v_cvt_f32_ubyte0_e32 v60, v66
	v_cvt_f32_ubyte0_e32 v70, v76
	v_cvt_f32_ubyte1_e32 v42, v46
	v_cvt_f32_ubyte1_e32 v52, v56
	v_cvt_f32_ubyte1_e32 v62, v66
	v_cvt_f32_ubyte1_e32 v72, v76
	v_cvt_f32_ubyte2_e32 v44, v46
	v_cvt_f32_ubyte2_e32 v54, v56
	v_cvt_f32_ubyte2_e32 v64, v66
	v_cvt_f32_ubyte2_e32 v74, v76
	v_cvt_f32_ubyte3_e32 v46, v46
	v_cvt_f32_ubyte3_e32 v56, v56
	v_cvt_f32_ubyte3_e32 v66, v66
	v_cvt_f32_ubyte3_e32 v76, v76
	v_cvt_f32_ubyte0_e32 v41, v47
	v_cvt_f32_ubyte0_e32 v51, v57
	v_cvt_f32_ubyte0_e32 v61, v67
	v_cvt_f32_ubyte0_e32 v71, v77
	v_cvt_f32_ubyte1_e32 v43, v47
	v_cvt_f32_ubyte1_e32 v53, v57
	v_cvt_f32_ubyte1_e32 v63, v67
	v_cvt_f32_ubyte1_e32 v73, v77
	v_cvt_f32_ubyte2_e32 v45, v47
	v_cvt_f32_ubyte2_e32 v55, v57
	v_cvt_f32_ubyte2_e32 v65, v67
	v_cvt_f32_ubyte2_e32 v75, v77
	v_cvt_f32_ubyte3_e32 v47, v47
	v_cvt_f32_ubyte3_e32 v57, v57
	v_cvt_f32_ubyte3_e32 v67, v67
	v_cvt_f32_ubyte3_e32 v77, v77
	v_fma_mix_f32 v40, v16, v40, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v17, v50, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v18, v60, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v70, v19, v70, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v41, v16, v41, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v17, v51, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v18, v61, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v71, v19, v71, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v42, v16, v42, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v52, v17, v52, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v62, v18, v62, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v72, v19, v72, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v43, v16, v43, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v53, v17, v53, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v63, v18, v63, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v73, v19, v73, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v44, v16, v44, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v54, v17, v54, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v64, v18, v64, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v74, v19, v74, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v16, v45, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v17, v55, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v65, v18, v65, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v75, v19, v75, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v16, v46, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v17, v56, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v66, v18, v66, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v76, v19, v76, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v16, v47, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v17, v57, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v67, v18, v67, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v77, v19, v77, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	buffer_load_b128 v[80:83], v3, s[24:27], s30 offen offset:3072
	buffer_load_b128 v[84:87], v3, s[24:27], s30 offen offset:3088
	buffer_load_b128 v[88:91], v3, s[24:27], s31 offen offset:3072
	buffer_load_b128 v[92:95], v3, s[24:27], s31 offen offset:3088
	s_wait_loadcnt 0x13
	v_dual_mul_f32 v48, v41, v97 :: v_dual_mul_f32 v49, v51, v97
	v_dual_mul_f32 v58, v61, v97 :: v_dual_mul_f32 v59, v71, v97
	s_wait_loadcnt 0x11
	v_dual_mul_f32 v68, v41, v105 :: v_dual_mul_f32 v69, v51, v105
	v_dual_mul_f32 v78, v61, v105 :: v_dual_mul_f32 v79, v71, v105
	v_dual_fmac_f32 v48, v40, v96 :: v_dual_fmac_f32 v49, v50, v96
	v_dual_fmac_f32 v58, v60, v96 :: v_dual_fmac_f32 v59, v70, v96
	v_dual_fmac_f32 v68, v40, v104 :: v_dual_fmac_f32 v69, v50, v104
	v_dual_fmac_f32 v78, v60, v104 :: v_dual_fmac_f32 v79, v70, v104
	v_dual_fmac_f32 v48, v42, v98 :: v_dual_fmac_f32 v49, v52, v98
	v_dual_fmac_f32 v58, v62, v98 :: v_dual_fmac_f32 v59, v72, v98
	v_dual_fmac_f32 v68, v42, v106 :: v_dual_fmac_f32 v69, v52, v106
	v_dual_fmac_f32 v78, v62, v106 :: v_dual_fmac_f32 v79, v72, v106
	v_dual_fmac_f32 v48, v43, v99 :: v_dual_fmac_f32 v49, v53, v99
	v_dual_fmac_f32 v58, v63, v99 :: v_dual_fmac_f32 v59, v73, v99
	v_dual_fmac_f32 v68, v43, v107 :: v_dual_fmac_f32 v69, v53, v107
	v_dual_fmac_f32 v78, v63, v107 :: v_dual_fmac_f32 v79, v73, v107
	v_dual_fmac_f32 v48, v44, v100 :: v_dual_fmac_f32 v49, v54, v100
	v_dual_fmac_f32 v58, v64, v100 :: v_dual_fmac_f32 v59, v74, v100
	s_wait_loadcnt 0x10
	v_dual_fmac_f32 v68, v44, v108 :: v_dual_fmac_f32 v69, v54, v108
	v_dual_fmac_f32 v78, v64, v108 :: v_dual_fmac_f32 v79, v74, v108
	v_dual_fmac_f32 v48, v45, v101 :: v_dual_fmac_f32 v49, v55, v101
	v_dual_fmac_f32 v58, v65, v101 :: v_dual_fmac_f32 v59, v75, v101
	v_dual_fmac_f32 v68, v45, v109 :: v_dual_fmac_f32 v69, v55, v109
	v_dual_fmac_f32 v78, v65, v109 :: v_dual_fmac_f32 v79, v75, v109
	v_dual_fmac_f32 v48, v46, v102 :: v_dual_fmac_f32 v49, v56, v102
	v_dual_fmac_f32 v58, v66, v102 :: v_dual_fmac_f32 v59, v76, v102
	v_dual_fmac_f32 v68, v46, v110 :: v_dual_fmac_f32 v69, v56, v110
	v_dual_fmac_f32 v78, v66, v110 :: v_dual_fmac_f32 v79, v76, v110
	v_dual_fmac_f32 v48, v47, v103 :: v_dual_fmac_f32 v49, v57, v103
	v_dual_fmac_f32 v58, v67, v103 :: v_dual_fmac_f32 v59, v77, v103
	v_dual_fmac_f32 v68, v47, v111 :: v_dual_fmac_f32 v69, v57, v111
	v_dual_fmac_f32 v78, v67, v111 :: v_dual_fmac_f32 v79, v77, v111
	v_dual_add_f32 v168, v168, v48 :: v_dual_add_f32 v169, v169, v49
	v_dual_add_f32 v170, v170, v58 :: v_dual_add_f32 v171, v171, v59
	v_dual_add_f32 v184, v184, v68 :: v_dual_add_f32 v185, v185, v69
	v_dual_add_f32 v186, v186, v78 :: v_dual_add_f32 v187, v187, v79
	buffer_load_b128 v[96:99], v3, s[24:27], s32 offen offset:3072
	buffer_load_b128 v[100:103], v3, s[24:27], s32 offen offset:3088
	buffer_load_b128 v[104:107], v3, s[24:27], s33 offen offset:3072
	buffer_load_b128 v[108:111], v3, s[24:27], s33 offen offset:3088
	s_wait_loadcnt 0x13
	v_dual_mul_f32 v48, v41, v113 :: v_dual_mul_f32 v49, v51, v113
	v_dual_mul_f32 v58, v61, v113 :: v_dual_mul_f32 v59, v71, v113
	s_wait_loadcnt 0x11
	v_dual_mul_f32 v68, v41, v121 :: v_dual_mul_f32 v69, v51, v121
	v_dual_mul_f32 v78, v61, v121 :: v_dual_mul_f32 v79, v71, v121
	v_dual_fmac_f32 v48, v40, v112 :: v_dual_fmac_f32 v49, v50, v112
	v_dual_fmac_f32 v58, v60, v112 :: v_dual_fmac_f32 v59, v70, v112
	v_dual_fmac_f32 v68, v40, v120 :: v_dual_fmac_f32 v69, v50, v120
	v_dual_fmac_f32 v78, v60, v120 :: v_dual_fmac_f32 v79, v70, v120
	v_dual_fmac_f32 v48, v42, v114 :: v_dual_fmac_f32 v49, v52, v114
	v_dual_fmac_f32 v58, v62, v114 :: v_dual_fmac_f32 v59, v72, v114
	v_dual_fmac_f32 v68, v42, v122 :: v_dual_fmac_f32 v69, v52, v122
	v_dual_fmac_f32 v78, v62, v122 :: v_dual_fmac_f32 v79, v72, v122
	v_dual_fmac_f32 v48, v43, v115 :: v_dual_fmac_f32 v49, v53, v115
	v_dual_fmac_f32 v58, v63, v115 :: v_dual_fmac_f32 v59, v73, v115
	v_dual_fmac_f32 v68, v43, v123 :: v_dual_fmac_f32 v69, v53, v123
	v_dual_fmac_f32 v78, v63, v123 :: v_dual_fmac_f32 v79, v73, v123
	v_dual_fmac_f32 v48, v44, v116 :: v_dual_fmac_f32 v49, v54, v116
	v_dual_fmac_f32 v58, v64, v116 :: v_dual_fmac_f32 v59, v74, v116
	s_wait_loadcnt 0x10
	v_dual_fmac_f32 v68, v44, v124 :: v_dual_fmac_f32 v69, v54, v124
	v_dual_fmac_f32 v78, v64, v124 :: v_dual_fmac_f32 v79, v74, v124
	v_dual_fmac_f32 v48, v45, v117 :: v_dual_fmac_f32 v49, v55, v117
	v_dual_fmac_f32 v58, v65, v117 :: v_dual_fmac_f32 v59, v75, v117
	v_dual_fmac_f32 v68, v45, v125 :: v_dual_fmac_f32 v69, v55, v125
	v_dual_fmac_f32 v78, v65, v125 :: v_dual_fmac_f32 v79, v75, v125
	v_dual_fmac_f32 v48, v46, v118 :: v_dual_fmac_f32 v49, v56, v118
	v_dual_fmac_f32 v58, v66, v118 :: v_dual_fmac_f32 v59, v76, v118
	v_dual_fmac_f32 v68, v46, v126 :: v_dual_fmac_f32 v69, v56, v126
	v_dual_fmac_f32 v78, v66, v126 :: v_dual_fmac_f32 v79, v76, v126
	v_dual_fmac_f32 v48, v47, v119 :: v_dual_fmac_f32 v49, v57, v119
	v_dual_fmac_f32 v58, v67, v119 :: v_dual_fmac_f32 v59, v77, v119
	v_dual_fmac_f32 v68, v47, v127 :: v_dual_fmac_f32 v69, v57, v127
	v_dual_fmac_f32 v78, v67, v127 :: v_dual_fmac_f32 v79, v77, v127
	v_dual_add_f32 v200, v200, v48 :: v_dual_add_f32 v201, v201, v49
	v_dual_add_f32 v202, v202, v58 :: v_dual_add_f32 v203, v203, v59
	v_dual_add_f32 v216, v216, v68 :: v_dual_add_f32 v217, v217, v69
	v_dual_add_f32 v218, v218, v78 :: v_dual_add_f32 v219, v219, v79
	s_wait_loadcnt 0xf
	v_dual_mul_f32 v48, v41, v129 :: v_dual_mul_f32 v49, v51, v129
	v_dual_mul_f32 v58, v61, v129 :: v_dual_mul_f32 v59, v71, v129
	s_wait_loadcnt 0xd
	v_dual_mul_f32 v68, v41, v137 :: v_dual_mul_f32 v69, v51, v137
	v_dual_mul_f32 v78, v61, v137 :: v_dual_mul_f32 v79, v71, v137
	v_dual_fmac_f32 v48, v40, v128 :: v_dual_fmac_f32 v49, v50, v128
	v_dual_fmac_f32 v58, v60, v128 :: v_dual_fmac_f32 v59, v70, v128
	v_dual_fmac_f32 v68, v40, v136 :: v_dual_fmac_f32 v69, v50, v136
	v_dual_fmac_f32 v78, v60, v136 :: v_dual_fmac_f32 v79, v70, v136
	v_dual_fmac_f32 v48, v42, v130 :: v_dual_fmac_f32 v49, v52, v130
	v_dual_fmac_f32 v58, v62, v130 :: v_dual_fmac_f32 v59, v72, v130
	v_dual_fmac_f32 v68, v42, v138 :: v_dual_fmac_f32 v69, v52, v138
	v_dual_fmac_f32 v78, v62, v138 :: v_dual_fmac_f32 v79, v72, v138
	v_dual_fmac_f32 v48, v43, v131 :: v_dual_fmac_f32 v49, v53, v131
	v_dual_fmac_f32 v58, v63, v131 :: v_dual_fmac_f32 v59, v73, v131
	v_dual_fmac_f32 v68, v43, v139 :: v_dual_fmac_f32 v69, v53, v139
	v_dual_fmac_f32 v78, v63, v139 :: v_dual_fmac_f32 v79, v73, v139
	v_dual_fmac_f32 v48, v44, v132 :: v_dual_fmac_f32 v49, v54, v132
	v_dual_fmac_f32 v58, v64, v132 :: v_dual_fmac_f32 v59, v74, v132
	s_wait_loadcnt 0xc
	v_dual_fmac_f32 v68, v44, v140 :: v_dual_fmac_f32 v69, v54, v140
	v_dual_fmac_f32 v78, v64, v140 :: v_dual_fmac_f32 v79, v74, v140
	v_dual_fmac_f32 v48, v45, v133 :: v_dual_fmac_f32 v49, v55, v133
	v_dual_fmac_f32 v58, v65, v133 :: v_dual_fmac_f32 v59, v75, v133
	v_dual_fmac_f32 v68, v45, v141 :: v_dual_fmac_f32 v69, v55, v141
	v_dual_fmac_f32 v78, v65, v141 :: v_dual_fmac_f32 v79, v75, v141
	v_dual_fmac_f32 v48, v46, v134 :: v_dual_fmac_f32 v49, v56, v134
	v_dual_fmac_f32 v58, v66, v134 :: v_dual_fmac_f32 v59, v76, v134
	v_dual_fmac_f32 v68, v46, v142 :: v_dual_fmac_f32 v69, v56, v142
	v_dual_fmac_f32 v78, v66, v142 :: v_dual_fmac_f32 v79, v76, v142
	v_dual_fmac_f32 v48, v47, v135 :: v_dual_fmac_f32 v49, v57, v135
	v_dual_fmac_f32 v58, v67, v135 :: v_dual_fmac_f32 v59, v77, v135
	v_dual_fmac_f32 v68, v47, v143 :: v_dual_fmac_f32 v69, v57, v143
	v_dual_fmac_f32 v78, v67, v143 :: v_dual_fmac_f32 v79, v77, v143
	v_dual_add_f32 v232, v232, v48 :: v_dual_add_f32 v233, v233, v49
	v_dual_add_f32 v234, v234, v58 :: v_dual_add_f32 v235, v235, v59
	v_dual_add_f32 v248, v248, v68 :: v_dual_add_f32 v249, v249, v69
	v_dual_add_f32 v250, v250, v78 :: v_dual_add_f32 v251, v251, v79
	v_and_b32_e32 v46, 0xf0f0f0f, v36
	v_and_b32_e32 v56, 0xf0f0f0f, v37
	v_and_b32_e32 v66, 0xf0f0f0f, v38
	v_and_b32_e32 v76, 0xf0f0f0f, v39
	v_lshrrev_b32_e32 v47, 4, v36
	v_lshrrev_b32_e32 v57, 4, v37
	v_lshrrev_b32_e32 v67, 4, v38
	v_lshrrev_b32_e32 v77, 4, v39
	v_and_b32_e32 v47, 0xf0f0f0f, v47
	v_and_b32_e32 v57, 0xf0f0f0f, v57
	v_and_b32_e32 v67, 0xf0f0f0f, v67
	v_and_b32_e32 v77, 0xf0f0f0f, v77
	v_cvt_f32_ubyte0_e32 v40, v46
	v_cvt_f32_ubyte0_e32 v50, v56
	v_cvt_f32_ubyte0_e32 v60, v66
	v_cvt_f32_ubyte0_e32 v70, v76
	v_cvt_f32_ubyte1_e32 v42, v46
	v_cvt_f32_ubyte1_e32 v52, v56
	v_cvt_f32_ubyte1_e32 v62, v66
	v_cvt_f32_ubyte1_e32 v72, v76
	v_cvt_f32_ubyte2_e32 v44, v46
	v_cvt_f32_ubyte2_e32 v54, v56
	v_cvt_f32_ubyte2_e32 v64, v66
	v_cvt_f32_ubyte2_e32 v74, v76
	v_cvt_f32_ubyte3_e32 v46, v46
	v_cvt_f32_ubyte3_e32 v56, v56
	v_cvt_f32_ubyte3_e32 v66, v66
	v_cvt_f32_ubyte3_e32 v76, v76
	v_cvt_f32_ubyte0_e32 v41, v47
	v_cvt_f32_ubyte0_e32 v51, v57
	v_cvt_f32_ubyte0_e32 v61, v67
	v_cvt_f32_ubyte0_e32 v71, v77
	v_cvt_f32_ubyte1_e32 v43, v47
	v_cvt_f32_ubyte1_e32 v53, v57
	v_cvt_f32_ubyte1_e32 v63, v67
	v_cvt_f32_ubyte1_e32 v73, v77
	v_cvt_f32_ubyte2_e32 v45, v47
	v_cvt_f32_ubyte2_e32 v55, v57
	v_cvt_f32_ubyte2_e32 v65, v67
	v_cvt_f32_ubyte2_e32 v75, v77
	v_cvt_f32_ubyte3_e32 v47, v47
	v_cvt_f32_ubyte3_e32 v57, v57
	v_cvt_f32_ubyte3_e32 v67, v67
	v_cvt_f32_ubyte3_e32 v77, v77
	v_fma_mix_f32 v40, v20, v40, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v21, v50, v21 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v22, v60, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v70, v23, v70, v23 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v41, v20, v41, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v21, v51, v21 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v22, v61, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v71, v23, v71, v23 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v42, v20, v42, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v52, v21, v52, v21 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v62, v22, v62, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v72, v23, v72, v23 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v43, v20, v43, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v53, v21, v53, v21 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v63, v22, v63, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v73, v23, v73, v23 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v44, v20, v44, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v54, v21, v54, v21 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v64, v22, v64, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v74, v23, v74, v23 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v20, v45, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v21, v55, v21 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v65, v22, v65, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v75, v23, v75, v23 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v20, v46, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v21, v56, v21 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v66, v22, v66, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v76, v23, v76, v23 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v20, v47, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v21, v57, v21 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v67, v22, v67, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v77, v23, v77, v23 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xb
	v_dual_mul_f32 v48, v41, v145 :: v_dual_mul_f32 v49, v51, v145
	v_dual_mul_f32 v58, v61, v145 :: v_dual_mul_f32 v59, v71, v145
	s_wait_loadcnt 0x9
	v_dual_mul_f32 v68, v41, v153 :: v_dual_mul_f32 v69, v51, v153
	v_dual_mul_f32 v78, v61, v153 :: v_dual_mul_f32 v79, v71, v153
	v_dual_fmac_f32 v48, v40, v144 :: v_dual_fmac_f32 v49, v50, v144
	v_dual_fmac_f32 v58, v60, v144 :: v_dual_fmac_f32 v59, v70, v144
	v_dual_fmac_f32 v68, v40, v152 :: v_dual_fmac_f32 v69, v50, v152
	v_dual_fmac_f32 v78, v60, v152 :: v_dual_fmac_f32 v79, v70, v152
	v_dual_fmac_f32 v48, v42, v146 :: v_dual_fmac_f32 v49, v52, v146
	v_dual_fmac_f32 v58, v62, v146 :: v_dual_fmac_f32 v59, v72, v146
	v_dual_fmac_f32 v68, v42, v154 :: v_dual_fmac_f32 v69, v52, v154
	v_dual_fmac_f32 v78, v62, v154 :: v_dual_fmac_f32 v79, v72, v154
	v_dual_fmac_f32 v48, v43, v147 :: v_dual_fmac_f32 v49, v53, v147
	v_dual_fmac_f32 v58, v63, v147 :: v_dual_fmac_f32 v59, v73, v147
	v_dual_fmac_f32 v68, v43, v155 :: v_dual_fmac_f32 v69, v53, v155
	v_dual_fmac_f32 v78, v63, v155 :: v_dual_fmac_f32 v79, v73, v155
	v_dual_fmac_f32 v48, v44, v148 :: v_dual_fmac_f32 v49, v54, v148
	v_dual_fmac_f32 v58, v64, v148 :: v_dual_fmac_f32 v59, v74, v148
	s_wait_loadcnt 0x8
	v_dual_fmac_f32 v68, v44, v156 :: v_dual_fmac_f32 v69, v54, v156
	v_dual_fmac_f32 v78, v64, v156 :: v_dual_fmac_f32 v79, v74, v156
	v_dual_fmac_f32 v48, v45, v149 :: v_dual_fmac_f32 v49, v55, v149
	v_dual_fmac_f32 v58, v65, v149 :: v_dual_fmac_f32 v59, v75, v149
	v_dual_fmac_f32 v68, v45, v157 :: v_dual_fmac_f32 v69, v55, v157
	v_dual_fmac_f32 v78, v65, v157 :: v_dual_fmac_f32 v79, v75, v157
	v_dual_fmac_f32 v48, v46, v150 :: v_dual_fmac_f32 v49, v56, v150
	v_dual_fmac_f32 v58, v66, v150 :: v_dual_fmac_f32 v59, v76, v150
	v_dual_fmac_f32 v68, v46, v158 :: v_dual_fmac_f32 v69, v56, v158
	v_dual_fmac_f32 v78, v66, v158 :: v_dual_fmac_f32 v79, v76, v158
	v_dual_fmac_f32 v48, v47, v151 :: v_dual_fmac_f32 v49, v57, v151
	v_dual_fmac_f32 v58, v67, v151 :: v_dual_fmac_f32 v59, v77, v151
	v_dual_fmac_f32 v68, v47, v159 :: v_dual_fmac_f32 v69, v57, v159
	v_dual_fmac_f32 v78, v67, v159 :: v_dual_fmac_f32 v79, v77, v159
	v_dual_add_f32 v172, v172, v48 :: v_dual_add_f32 v173, v173, v49
	v_dual_add_f32 v174, v174, v58 :: v_dual_add_f32 v175, v175, v59
	v_dual_add_f32 v188, v188, v68 :: v_dual_add_f32 v189, v189, v69
	v_dual_add_f32 v190, v190, v78 :: v_dual_add_f32 v191, v191, v79
	s_wait_loadcnt 0x7
	v_dual_mul_f32 v48, v41, v81 :: v_dual_mul_f32 v49, v51, v81
	v_dual_mul_f32 v58, v61, v81 :: v_dual_mul_f32 v59, v71, v81
	s_wait_loadcnt 0x5
	v_dual_mul_f32 v68, v41, v89 :: v_dual_mul_f32 v69, v51, v89
	v_dual_mul_f32 v78, v61, v89 :: v_dual_mul_f32 v79, v71, v89
	v_dual_fmac_f32 v48, v40, v80 :: v_dual_fmac_f32 v49, v50, v80
	v_dual_fmac_f32 v58, v60, v80 :: v_dual_fmac_f32 v59, v70, v80
	v_dual_fmac_f32 v68, v40, v88 :: v_dual_fmac_f32 v69, v50, v88
	v_dual_fmac_f32 v78, v60, v88 :: v_dual_fmac_f32 v79, v70, v88
	v_dual_fmac_f32 v48, v42, v82 :: v_dual_fmac_f32 v49, v52, v82
	v_dual_fmac_f32 v58, v62, v82 :: v_dual_fmac_f32 v59, v72, v82
	v_dual_fmac_f32 v68, v42, v90 :: v_dual_fmac_f32 v69, v52, v90
	v_dual_fmac_f32 v78, v62, v90 :: v_dual_fmac_f32 v79, v72, v90
	v_dual_fmac_f32 v48, v43, v83 :: v_dual_fmac_f32 v49, v53, v83
	v_dual_fmac_f32 v58, v63, v83 :: v_dual_fmac_f32 v59, v73, v83
	v_dual_fmac_f32 v68, v43, v91 :: v_dual_fmac_f32 v69, v53, v91
	v_dual_fmac_f32 v78, v63, v91 :: v_dual_fmac_f32 v79, v73, v91
	v_dual_fmac_f32 v48, v44, v84 :: v_dual_fmac_f32 v49, v54, v84
	v_dual_fmac_f32 v58, v64, v84 :: v_dual_fmac_f32 v59, v74, v84
	s_wait_loadcnt 0x4
	v_dual_fmac_f32 v68, v44, v92 :: v_dual_fmac_f32 v69, v54, v92
	v_dual_fmac_f32 v78, v64, v92 :: v_dual_fmac_f32 v79, v74, v92
	v_dual_fmac_f32 v48, v45, v85 :: v_dual_fmac_f32 v49, v55, v85
	v_dual_fmac_f32 v58, v65, v85 :: v_dual_fmac_f32 v59, v75, v85
	v_dual_fmac_f32 v68, v45, v93 :: v_dual_fmac_f32 v69, v55, v93
	v_dual_fmac_f32 v78, v65, v93 :: v_dual_fmac_f32 v79, v75, v93
	v_dual_fmac_f32 v48, v46, v86 :: v_dual_fmac_f32 v49, v56, v86
	v_dual_fmac_f32 v58, v66, v86 :: v_dual_fmac_f32 v59, v76, v86
	v_dual_fmac_f32 v68, v46, v94 :: v_dual_fmac_f32 v69, v56, v94
	v_dual_fmac_f32 v78, v66, v94 :: v_dual_fmac_f32 v79, v76, v94
	v_dual_fmac_f32 v48, v47, v87 :: v_dual_fmac_f32 v49, v57, v87
	v_dual_fmac_f32 v58, v67, v87 :: v_dual_fmac_f32 v59, v77, v87
	v_dual_fmac_f32 v68, v47, v95 :: v_dual_fmac_f32 v69, v57, v95
	v_dual_fmac_f32 v78, v67, v95 :: v_dual_fmac_f32 v79, v77, v95
	v_dual_add_f32 v204, v204, v48 :: v_dual_add_f32 v205, v205, v49
	v_dual_add_f32 v206, v206, v58 :: v_dual_add_f32 v207, v207, v59
	v_dual_add_f32 v220, v220, v68 :: v_dual_add_f32 v221, v221, v69
	v_dual_add_f32 v222, v222, v78 :: v_dual_add_f32 v223, v223, v79
	s_wait_loadcnt 0x3
	v_dual_mul_f32 v48, v41, v97 :: v_dual_mul_f32 v49, v51, v97
	v_dual_mul_f32 v58, v61, v97 :: v_dual_mul_f32 v59, v71, v97
	s_wait_loadcnt 0x1
	v_dual_mul_f32 v68, v41, v105 :: v_dual_mul_f32 v69, v51, v105
	v_dual_mul_f32 v78, v61, v105 :: v_dual_mul_f32 v79, v71, v105
	v_dual_fmac_f32 v48, v40, v96 :: v_dual_fmac_f32 v49, v50, v96
	v_dual_fmac_f32 v58, v60, v96 :: v_dual_fmac_f32 v59, v70, v96
	v_dual_fmac_f32 v68, v40, v104 :: v_dual_fmac_f32 v69, v50, v104
	v_dual_fmac_f32 v78, v60, v104 :: v_dual_fmac_f32 v79, v70, v104
	v_dual_fmac_f32 v48, v42, v98 :: v_dual_fmac_f32 v49, v52, v98
	v_dual_fmac_f32 v58, v62, v98 :: v_dual_fmac_f32 v59, v72, v98
	v_dual_fmac_f32 v68, v42, v106 :: v_dual_fmac_f32 v69, v52, v106
	v_dual_fmac_f32 v78, v62, v106 :: v_dual_fmac_f32 v79, v72, v106
	v_dual_fmac_f32 v48, v43, v99 :: v_dual_fmac_f32 v49, v53, v99
	v_dual_fmac_f32 v58, v63, v99 :: v_dual_fmac_f32 v59, v73, v99
	v_dual_fmac_f32 v68, v43, v107 :: v_dual_fmac_f32 v69, v53, v107
	v_dual_fmac_f32 v78, v63, v107 :: v_dual_fmac_f32 v79, v73, v107
	v_dual_fmac_f32 v48, v44, v100 :: v_dual_fmac_f32 v49, v54, v100
	v_dual_fmac_f32 v58, v64, v100 :: v_dual_fmac_f32 v59, v74, v100
	s_wait_loadcnt 0x0
	v_dual_fmac_f32 v68, v44, v108 :: v_dual_fmac_f32 v69, v54, v108
	v_dual_fmac_f32 v78, v64, v108 :: v_dual_fmac_f32 v79, v74, v108
	v_dual_fmac_f32 v48, v45, v101 :: v_dual_fmac_f32 v49, v55, v101
	v_dual_fmac_f32 v58, v65, v101 :: v_dual_fmac_f32 v59, v75, v101
	v_dual_fmac_f32 v68, v45, v109 :: v_dual_fmac_f32 v69, v55, v109
	v_dual_fmac_f32 v78, v65, v109 :: v_dual_fmac_f32 v79, v75, v109
	v_dual_fmac_f32 v48, v46, v102 :: v_dual_fmac_f32 v49, v56, v102
	v_dual_fmac_f32 v58, v66, v102 :: v_dual_fmac_f32 v59, v76, v102
	v_dual_fmac_f32 v68, v46, v110 :: v_dual_fmac_f32 v69, v56, v110
	v_dual_fmac_f32 v78, v66, v110 :: v_dual_fmac_f32 v79, v76, v110
	v_dual_fmac_f32 v48, v47, v103 :: v_dual_fmac_f32 v49, v57, v103
	v_dual_fmac_f32 v58, v67, v103 :: v_dual_fmac_f32 v59, v77, v103
	v_dual_fmac_f32 v68, v47, v111 :: v_dual_fmac_f32 v69, v57, v111
	v_dual_fmac_f32 v78, v67, v111 :: v_dual_fmac_f32 v79, v77, v111
	v_dual_add_f32 v236, v236, v48 :: v_dual_add_f32 v237, v237, v49
	v_dual_add_f32 v238, v238, v58 :: v_dual_add_f32 v239, v239, v59
	v_dual_add_f32 v252, v252, v68 :: v_dual_add_f32 v253, v253, v69
	v_dual_add_f32 v254, v254, v78 :: v_dual_add_f32 v255, v255, v79
	s_add_co_i32 s16, s16, 0x220
	s_add_co_i32 s17, s17, 0x220
	s_add_co_i32 s18, s18, 0x220
	s_add_co_i32 s19, s19, 0x220
	s_add_co_i32 s28, s28, 0x1000
	s_add_co_i32 s29, s29, 0x1000
	s_add_co_i32 s30, s30, 0x1000
	s_add_co_i32 s31, s31, 0x1000
	s_add_co_i32 s32, s32, 0x1000
	s_add_co_i32 s33, s33, 0x1000
	s_add_co_i32 s36, s36, -1
	s_cmp_lg_u32 s36, 0
	s_cbranch_scc1 .Lpx_b6_quad
	.Lpx_b6_tail:
	s_cmp_gt_u32 s37, 0
	s_cbranch_scc0 .Lpx_b6_fold
	buffer_load_b32 v8, v1, s[20:23], s16 offen
	buffer_load_b32 v9, v1, s[20:23], s17 offen
	buffer_load_b32 v10, v1, s[20:23], s18 offen
	buffer_load_b32 v11, v1, s[20:23], s19 offen
	buffer_load_b32 v24, v2, s[20:23], s16 offen offset:8
	buffer_load_b32 v25, v2, s[20:23], s17 offen offset:8
	buffer_load_b32 v26, v2, s[20:23], s18 offen offset:8
	buffer_load_b32 v27, v2, s[20:23], s19 offen offset:8
	buffer_load_b128 v[80:83], v3, s[24:27], s28 offen
	buffer_load_b128 v[84:87], v3, s[24:27], s28 offen offset:16
	buffer_load_b128 v[88:91], v3, s[24:27], s29 offen
	buffer_load_b128 v[92:95], v3, s[24:27], s29 offen offset:16
	buffer_load_b128 v[96:99], v3, s[24:27], s30 offen
	buffer_load_b128 v[100:103], v3, s[24:27], s30 offen offset:16
	buffer_load_b128 v[104:107], v3, s[24:27], s31 offen
	buffer_load_b128 v[108:111], v3, s[24:27], s31 offen offset:16
	buffer_load_b128 v[112:115], v3, s[24:27], s32 offen
	buffer_load_b128 v[116:119], v3, s[24:27], s32 offen offset:16
	buffer_load_b128 v[120:123], v3, s[24:27], s33 offen
	buffer_load_b128 v[124:127], v3, s[24:27], s33 offen offset:16
	s_wait_loadcnt 0xf
	v_and_b32_e32 v46, 0xf0f0f0f, v24
	s_wait_loadcnt 0xe
	v_and_b32_e32 v56, 0xf0f0f0f, v25
	s_wait_loadcnt 0xd
	v_and_b32_e32 v66, 0xf0f0f0f, v26
	s_wait_loadcnt 0xc
	v_and_b32_e32 v76, 0xf0f0f0f, v27
	v_lshrrev_b32_e32 v47, 4, v24
	v_lshrrev_b32_e32 v57, 4, v25
	v_lshrrev_b32_e32 v67, 4, v26
	v_lshrrev_b32_e32 v77, 4, v27
	v_and_b32_e32 v47, 0xf0f0f0f, v47
	v_and_b32_e32 v57, 0xf0f0f0f, v57
	v_and_b32_e32 v67, 0xf0f0f0f, v67
	v_and_b32_e32 v77, 0xf0f0f0f, v77
	v_cvt_f32_ubyte0_e32 v40, v46
	v_cvt_f32_ubyte0_e32 v50, v56
	v_cvt_f32_ubyte0_e32 v60, v66
	v_cvt_f32_ubyte0_e32 v70, v76
	v_cvt_f32_ubyte1_e32 v42, v46
	v_cvt_f32_ubyte1_e32 v52, v56
	v_cvt_f32_ubyte1_e32 v62, v66
	v_cvt_f32_ubyte1_e32 v72, v76
	v_cvt_f32_ubyte2_e32 v44, v46
	v_cvt_f32_ubyte2_e32 v54, v56
	v_cvt_f32_ubyte2_e32 v64, v66
	v_cvt_f32_ubyte2_e32 v74, v76
	v_cvt_f32_ubyte3_e32 v46, v46
	v_cvt_f32_ubyte3_e32 v56, v56
	v_cvt_f32_ubyte3_e32 v66, v66
	v_cvt_f32_ubyte3_e32 v76, v76
	v_cvt_f32_ubyte0_e32 v41, v47
	v_cvt_f32_ubyte0_e32 v51, v57
	v_cvt_f32_ubyte0_e32 v61, v67
	v_cvt_f32_ubyte0_e32 v71, v77
	v_cvt_f32_ubyte1_e32 v43, v47
	v_cvt_f32_ubyte1_e32 v53, v57
	v_cvt_f32_ubyte1_e32 v63, v67
	v_cvt_f32_ubyte1_e32 v73, v77
	v_cvt_f32_ubyte2_e32 v45, v47
	v_cvt_f32_ubyte2_e32 v55, v57
	v_cvt_f32_ubyte2_e32 v65, v67
	v_cvt_f32_ubyte2_e32 v75, v77
	v_cvt_f32_ubyte3_e32 v47, v47
	v_cvt_f32_ubyte3_e32 v57, v57
	v_cvt_f32_ubyte3_e32 v67, v67
	v_cvt_f32_ubyte3_e32 v77, v77
	v_fma_mix_f32 v40, v8, v40, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v9, v50, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v10, v60, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v70, v11, v70, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v41, v8, v41, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v9, v51, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v10, v61, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v71, v11, v71, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v42, v8, v42, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v52, v9, v52, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v62, v10, v62, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v72, v11, v72, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v43, v8, v43, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v53, v9, v53, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v63, v10, v63, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v73, v11, v73, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v44, v8, v44, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v54, v9, v54, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v64, v10, v64, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v74, v11, v74, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v8, v45, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v9, v55, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v65, v10, v65, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v75, v11, v75, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v8, v46, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v9, v56, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v66, v10, v66, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v76, v11, v76, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v8, v47, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v9, v57, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v67, v10, v67, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v77, v11, v77, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xb
	v_dual_mul_f32 v48, v41, v81 :: v_dual_mul_f32 v49, v51, v81
	v_dual_mul_f32 v58, v61, v81 :: v_dual_mul_f32 v59, v71, v81
	s_wait_loadcnt 0x9
	v_dual_mul_f32 v68, v41, v89 :: v_dual_mul_f32 v69, v51, v89
	v_dual_mul_f32 v78, v61, v89 :: v_dual_mul_f32 v79, v71, v89
	v_dual_fmac_f32 v48, v40, v80 :: v_dual_fmac_f32 v49, v50, v80
	v_dual_fmac_f32 v58, v60, v80 :: v_dual_fmac_f32 v59, v70, v80
	v_dual_fmac_f32 v68, v40, v88 :: v_dual_fmac_f32 v69, v50, v88
	v_dual_fmac_f32 v78, v60, v88 :: v_dual_fmac_f32 v79, v70, v88
	v_dual_fmac_f32 v48, v42, v82 :: v_dual_fmac_f32 v49, v52, v82
	v_dual_fmac_f32 v58, v62, v82 :: v_dual_fmac_f32 v59, v72, v82
	v_dual_fmac_f32 v68, v42, v90 :: v_dual_fmac_f32 v69, v52, v90
	v_dual_fmac_f32 v78, v62, v90 :: v_dual_fmac_f32 v79, v72, v90
	v_dual_fmac_f32 v48, v43, v83 :: v_dual_fmac_f32 v49, v53, v83
	v_dual_fmac_f32 v58, v63, v83 :: v_dual_fmac_f32 v59, v73, v83
	v_dual_fmac_f32 v68, v43, v91 :: v_dual_fmac_f32 v69, v53, v91
	v_dual_fmac_f32 v78, v63, v91 :: v_dual_fmac_f32 v79, v73, v91
	v_dual_fmac_f32 v48, v44, v84 :: v_dual_fmac_f32 v49, v54, v84
	v_dual_fmac_f32 v58, v64, v84 :: v_dual_fmac_f32 v59, v74, v84
	s_wait_loadcnt 0x8
	v_dual_fmac_f32 v68, v44, v92 :: v_dual_fmac_f32 v69, v54, v92
	v_dual_fmac_f32 v78, v64, v92 :: v_dual_fmac_f32 v79, v74, v92
	v_dual_fmac_f32 v48, v45, v85 :: v_dual_fmac_f32 v49, v55, v85
	v_dual_fmac_f32 v58, v65, v85 :: v_dual_fmac_f32 v59, v75, v85
	v_dual_fmac_f32 v68, v45, v93 :: v_dual_fmac_f32 v69, v55, v93
	v_dual_fmac_f32 v78, v65, v93 :: v_dual_fmac_f32 v79, v75, v93
	v_dual_fmac_f32 v48, v46, v86 :: v_dual_fmac_f32 v49, v56, v86
	v_dual_fmac_f32 v58, v66, v86 :: v_dual_fmac_f32 v59, v76, v86
	v_dual_fmac_f32 v68, v46, v94 :: v_dual_fmac_f32 v69, v56, v94
	v_dual_fmac_f32 v78, v66, v94 :: v_dual_fmac_f32 v79, v76, v94
	v_dual_fmac_f32 v48, v47, v87 :: v_dual_fmac_f32 v49, v57, v87
	v_dual_fmac_f32 v58, v67, v87 :: v_dual_fmac_f32 v59, v77, v87
	v_dual_fmac_f32 v68, v47, v95 :: v_dual_fmac_f32 v69, v57, v95
	v_dual_fmac_f32 v78, v67, v95 :: v_dual_fmac_f32 v79, v77, v95
	v_dual_add_f32 v160, v160, v48 :: v_dual_add_f32 v161, v161, v49
	v_dual_add_f32 v162, v162, v58 :: v_dual_add_f32 v163, v163, v59
	v_dual_add_f32 v176, v176, v68 :: v_dual_add_f32 v177, v177, v69
	v_dual_add_f32 v178, v178, v78 :: v_dual_add_f32 v179, v179, v79
	s_wait_loadcnt 0x7
	v_dual_mul_f32 v48, v41, v97 :: v_dual_mul_f32 v49, v51, v97
	v_dual_mul_f32 v58, v61, v97 :: v_dual_mul_f32 v59, v71, v97
	s_wait_loadcnt 0x5
	v_dual_mul_f32 v68, v41, v105 :: v_dual_mul_f32 v69, v51, v105
	v_dual_mul_f32 v78, v61, v105 :: v_dual_mul_f32 v79, v71, v105
	v_dual_fmac_f32 v48, v40, v96 :: v_dual_fmac_f32 v49, v50, v96
	v_dual_fmac_f32 v58, v60, v96 :: v_dual_fmac_f32 v59, v70, v96
	v_dual_fmac_f32 v68, v40, v104 :: v_dual_fmac_f32 v69, v50, v104
	v_dual_fmac_f32 v78, v60, v104 :: v_dual_fmac_f32 v79, v70, v104
	v_dual_fmac_f32 v48, v42, v98 :: v_dual_fmac_f32 v49, v52, v98
	v_dual_fmac_f32 v58, v62, v98 :: v_dual_fmac_f32 v59, v72, v98
	v_dual_fmac_f32 v68, v42, v106 :: v_dual_fmac_f32 v69, v52, v106
	v_dual_fmac_f32 v78, v62, v106 :: v_dual_fmac_f32 v79, v72, v106
	v_dual_fmac_f32 v48, v43, v99 :: v_dual_fmac_f32 v49, v53, v99
	v_dual_fmac_f32 v58, v63, v99 :: v_dual_fmac_f32 v59, v73, v99
	v_dual_fmac_f32 v68, v43, v107 :: v_dual_fmac_f32 v69, v53, v107
	v_dual_fmac_f32 v78, v63, v107 :: v_dual_fmac_f32 v79, v73, v107
	v_dual_fmac_f32 v48, v44, v100 :: v_dual_fmac_f32 v49, v54, v100
	v_dual_fmac_f32 v58, v64, v100 :: v_dual_fmac_f32 v59, v74, v100
	s_wait_loadcnt 0x4
	v_dual_fmac_f32 v68, v44, v108 :: v_dual_fmac_f32 v69, v54, v108
	v_dual_fmac_f32 v78, v64, v108 :: v_dual_fmac_f32 v79, v74, v108
	v_dual_fmac_f32 v48, v45, v101 :: v_dual_fmac_f32 v49, v55, v101
	v_dual_fmac_f32 v58, v65, v101 :: v_dual_fmac_f32 v59, v75, v101
	v_dual_fmac_f32 v68, v45, v109 :: v_dual_fmac_f32 v69, v55, v109
	v_dual_fmac_f32 v78, v65, v109 :: v_dual_fmac_f32 v79, v75, v109
	v_dual_fmac_f32 v48, v46, v102 :: v_dual_fmac_f32 v49, v56, v102
	v_dual_fmac_f32 v58, v66, v102 :: v_dual_fmac_f32 v59, v76, v102
	v_dual_fmac_f32 v68, v46, v110 :: v_dual_fmac_f32 v69, v56, v110
	v_dual_fmac_f32 v78, v66, v110 :: v_dual_fmac_f32 v79, v76, v110
	v_dual_fmac_f32 v48, v47, v103 :: v_dual_fmac_f32 v49, v57, v103
	v_dual_fmac_f32 v58, v67, v103 :: v_dual_fmac_f32 v59, v77, v103
	v_dual_fmac_f32 v68, v47, v111 :: v_dual_fmac_f32 v69, v57, v111
	v_dual_fmac_f32 v78, v67, v111 :: v_dual_fmac_f32 v79, v77, v111
	v_dual_add_f32 v192, v192, v48 :: v_dual_add_f32 v193, v193, v49
	v_dual_add_f32 v194, v194, v58 :: v_dual_add_f32 v195, v195, v59
	v_dual_add_f32 v208, v208, v68 :: v_dual_add_f32 v209, v209, v69
	v_dual_add_f32 v210, v210, v78 :: v_dual_add_f32 v211, v211, v79
	s_wait_loadcnt 0x3
	v_dual_mul_f32 v48, v41, v113 :: v_dual_mul_f32 v49, v51, v113
	v_dual_mul_f32 v58, v61, v113 :: v_dual_mul_f32 v59, v71, v113
	s_wait_loadcnt 0x1
	v_dual_mul_f32 v68, v41, v121 :: v_dual_mul_f32 v69, v51, v121
	v_dual_mul_f32 v78, v61, v121 :: v_dual_mul_f32 v79, v71, v121
	v_dual_fmac_f32 v48, v40, v112 :: v_dual_fmac_f32 v49, v50, v112
	v_dual_fmac_f32 v58, v60, v112 :: v_dual_fmac_f32 v59, v70, v112
	v_dual_fmac_f32 v68, v40, v120 :: v_dual_fmac_f32 v69, v50, v120
	v_dual_fmac_f32 v78, v60, v120 :: v_dual_fmac_f32 v79, v70, v120
	v_dual_fmac_f32 v48, v42, v114 :: v_dual_fmac_f32 v49, v52, v114
	v_dual_fmac_f32 v58, v62, v114 :: v_dual_fmac_f32 v59, v72, v114
	v_dual_fmac_f32 v68, v42, v122 :: v_dual_fmac_f32 v69, v52, v122
	v_dual_fmac_f32 v78, v62, v122 :: v_dual_fmac_f32 v79, v72, v122
	v_dual_fmac_f32 v48, v43, v115 :: v_dual_fmac_f32 v49, v53, v115
	v_dual_fmac_f32 v58, v63, v115 :: v_dual_fmac_f32 v59, v73, v115
	v_dual_fmac_f32 v68, v43, v123 :: v_dual_fmac_f32 v69, v53, v123
	v_dual_fmac_f32 v78, v63, v123 :: v_dual_fmac_f32 v79, v73, v123
	v_dual_fmac_f32 v48, v44, v116 :: v_dual_fmac_f32 v49, v54, v116
	v_dual_fmac_f32 v58, v64, v116 :: v_dual_fmac_f32 v59, v74, v116
	s_wait_loadcnt 0x0
	v_dual_fmac_f32 v68, v44, v124 :: v_dual_fmac_f32 v69, v54, v124
	v_dual_fmac_f32 v78, v64, v124 :: v_dual_fmac_f32 v79, v74, v124
	v_dual_fmac_f32 v48, v45, v117 :: v_dual_fmac_f32 v49, v55, v117
	v_dual_fmac_f32 v58, v65, v117 :: v_dual_fmac_f32 v59, v75, v117
	v_dual_fmac_f32 v68, v45, v125 :: v_dual_fmac_f32 v69, v55, v125
	v_dual_fmac_f32 v78, v65, v125 :: v_dual_fmac_f32 v79, v75, v125
	v_dual_fmac_f32 v48, v46, v118 :: v_dual_fmac_f32 v49, v56, v118
	v_dual_fmac_f32 v58, v66, v118 :: v_dual_fmac_f32 v59, v76, v118
	v_dual_fmac_f32 v68, v46, v126 :: v_dual_fmac_f32 v69, v56, v126
	v_dual_fmac_f32 v78, v66, v126 :: v_dual_fmac_f32 v79, v76, v126
	v_dual_fmac_f32 v48, v47, v119 :: v_dual_fmac_f32 v49, v57, v119
	v_dual_fmac_f32 v58, v67, v119 :: v_dual_fmac_f32 v59, v77, v119
	v_dual_fmac_f32 v68, v47, v127 :: v_dual_fmac_f32 v69, v57, v127
	v_dual_fmac_f32 v78, v67, v127 :: v_dual_fmac_f32 v79, v77, v127
	v_dual_add_f32 v224, v224, v48 :: v_dual_add_f32 v225, v225, v49
	v_dual_add_f32 v226, v226, v58 :: v_dual_add_f32 v227, v227, v59
	v_dual_add_f32 v240, v240, v68 :: v_dual_add_f32 v241, v241, v69
	v_dual_add_f32 v242, v242, v78 :: v_dual_add_f32 v243, v243, v79
	s_cmp_gt_u32 s37, 1
	s_cbranch_scc0 .Lpx_b6_fold
	buffer_load_b32 v12, v1, s[20:23], s16 offen offset:136
	buffer_load_b32 v13, v1, s[20:23], s17 offen offset:136
	buffer_load_b32 v14, v1, s[20:23], s18 offen offset:136
	buffer_load_b32 v15, v1, s[20:23], s19 offen offset:136
	buffer_load_b32 v28, v2, s[20:23], s16 offen offset:144
	buffer_load_b32 v29, v2, s[20:23], s17 offen offset:144
	buffer_load_b32 v30, v2, s[20:23], s18 offen offset:144
	buffer_load_b32 v31, v2, s[20:23], s19 offen offset:144
	buffer_load_b128 v[80:83], v3, s[24:27], s28 offen offset:1024
	buffer_load_b128 v[84:87], v3, s[24:27], s28 offen offset:1040
	buffer_load_b128 v[88:91], v3, s[24:27], s29 offen offset:1024
	buffer_load_b128 v[92:95], v3, s[24:27], s29 offen offset:1040
	buffer_load_b128 v[96:99], v3, s[24:27], s30 offen offset:1024
	buffer_load_b128 v[100:103], v3, s[24:27], s30 offen offset:1040
	buffer_load_b128 v[104:107], v3, s[24:27], s31 offen offset:1024
	buffer_load_b128 v[108:111], v3, s[24:27], s31 offen offset:1040
	buffer_load_b128 v[112:115], v3, s[24:27], s32 offen offset:1024
	buffer_load_b128 v[116:119], v3, s[24:27], s32 offen offset:1040
	buffer_load_b128 v[120:123], v3, s[24:27], s33 offen offset:1024
	buffer_load_b128 v[124:127], v3, s[24:27], s33 offen offset:1040
	s_wait_loadcnt 0xf
	v_and_b32_e32 v46, 0xf0f0f0f, v28
	s_wait_loadcnt 0xe
	v_and_b32_e32 v56, 0xf0f0f0f, v29
	s_wait_loadcnt 0xd
	v_and_b32_e32 v66, 0xf0f0f0f, v30
	s_wait_loadcnt 0xc
	v_and_b32_e32 v76, 0xf0f0f0f, v31
	v_lshrrev_b32_e32 v47, 4, v28
	v_lshrrev_b32_e32 v57, 4, v29
	v_lshrrev_b32_e32 v67, 4, v30
	v_lshrrev_b32_e32 v77, 4, v31
	v_and_b32_e32 v47, 0xf0f0f0f, v47
	v_and_b32_e32 v57, 0xf0f0f0f, v57
	v_and_b32_e32 v67, 0xf0f0f0f, v67
	v_and_b32_e32 v77, 0xf0f0f0f, v77
	v_cvt_f32_ubyte0_e32 v40, v46
	v_cvt_f32_ubyte0_e32 v50, v56
	v_cvt_f32_ubyte0_e32 v60, v66
	v_cvt_f32_ubyte0_e32 v70, v76
	v_cvt_f32_ubyte1_e32 v42, v46
	v_cvt_f32_ubyte1_e32 v52, v56
	v_cvt_f32_ubyte1_e32 v62, v66
	v_cvt_f32_ubyte1_e32 v72, v76
	v_cvt_f32_ubyte2_e32 v44, v46
	v_cvt_f32_ubyte2_e32 v54, v56
	v_cvt_f32_ubyte2_e32 v64, v66
	v_cvt_f32_ubyte2_e32 v74, v76
	v_cvt_f32_ubyte3_e32 v46, v46
	v_cvt_f32_ubyte3_e32 v56, v56
	v_cvt_f32_ubyte3_e32 v66, v66
	v_cvt_f32_ubyte3_e32 v76, v76
	v_cvt_f32_ubyte0_e32 v41, v47
	v_cvt_f32_ubyte0_e32 v51, v57
	v_cvt_f32_ubyte0_e32 v61, v67
	v_cvt_f32_ubyte0_e32 v71, v77
	v_cvt_f32_ubyte1_e32 v43, v47
	v_cvt_f32_ubyte1_e32 v53, v57
	v_cvt_f32_ubyte1_e32 v63, v67
	v_cvt_f32_ubyte1_e32 v73, v77
	v_cvt_f32_ubyte2_e32 v45, v47
	v_cvt_f32_ubyte2_e32 v55, v57
	v_cvt_f32_ubyte2_e32 v65, v67
	v_cvt_f32_ubyte2_e32 v75, v77
	v_cvt_f32_ubyte3_e32 v47, v47
	v_cvt_f32_ubyte3_e32 v57, v57
	v_cvt_f32_ubyte3_e32 v67, v67
	v_cvt_f32_ubyte3_e32 v77, v77
	v_fma_mix_f32 v40, v12, v40, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v13, v50, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v14, v60, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v70, v15, v70, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v41, v12, v41, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v13, v51, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v14, v61, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v71, v15, v71, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v42, v12, v42, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v52, v13, v52, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v62, v14, v62, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v72, v15, v72, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v43, v12, v43, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v53, v13, v53, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v63, v14, v63, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v73, v15, v73, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v44, v12, v44, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v54, v13, v54, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v64, v14, v64, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v74, v15, v74, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v12, v45, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v13, v55, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v65, v14, v65, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v75, v15, v75, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v12, v46, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v13, v56, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v66, v14, v66, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v76, v15, v76, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v12, v47, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v13, v57, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v67, v14, v67, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v77, v15, v77, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xb
	v_dual_mul_f32 v48, v41, v81 :: v_dual_mul_f32 v49, v51, v81
	v_dual_mul_f32 v58, v61, v81 :: v_dual_mul_f32 v59, v71, v81
	s_wait_loadcnt 0x9
	v_dual_mul_f32 v68, v41, v89 :: v_dual_mul_f32 v69, v51, v89
	v_dual_mul_f32 v78, v61, v89 :: v_dual_mul_f32 v79, v71, v89
	v_dual_fmac_f32 v48, v40, v80 :: v_dual_fmac_f32 v49, v50, v80
	v_dual_fmac_f32 v58, v60, v80 :: v_dual_fmac_f32 v59, v70, v80
	v_dual_fmac_f32 v68, v40, v88 :: v_dual_fmac_f32 v69, v50, v88
	v_dual_fmac_f32 v78, v60, v88 :: v_dual_fmac_f32 v79, v70, v88
	v_dual_fmac_f32 v48, v42, v82 :: v_dual_fmac_f32 v49, v52, v82
	v_dual_fmac_f32 v58, v62, v82 :: v_dual_fmac_f32 v59, v72, v82
	v_dual_fmac_f32 v68, v42, v90 :: v_dual_fmac_f32 v69, v52, v90
	v_dual_fmac_f32 v78, v62, v90 :: v_dual_fmac_f32 v79, v72, v90
	v_dual_fmac_f32 v48, v43, v83 :: v_dual_fmac_f32 v49, v53, v83
	v_dual_fmac_f32 v58, v63, v83 :: v_dual_fmac_f32 v59, v73, v83
	v_dual_fmac_f32 v68, v43, v91 :: v_dual_fmac_f32 v69, v53, v91
	v_dual_fmac_f32 v78, v63, v91 :: v_dual_fmac_f32 v79, v73, v91
	v_dual_fmac_f32 v48, v44, v84 :: v_dual_fmac_f32 v49, v54, v84
	v_dual_fmac_f32 v58, v64, v84 :: v_dual_fmac_f32 v59, v74, v84
	s_wait_loadcnt 0x8
	v_dual_fmac_f32 v68, v44, v92 :: v_dual_fmac_f32 v69, v54, v92
	v_dual_fmac_f32 v78, v64, v92 :: v_dual_fmac_f32 v79, v74, v92
	v_dual_fmac_f32 v48, v45, v85 :: v_dual_fmac_f32 v49, v55, v85
	v_dual_fmac_f32 v58, v65, v85 :: v_dual_fmac_f32 v59, v75, v85
	v_dual_fmac_f32 v68, v45, v93 :: v_dual_fmac_f32 v69, v55, v93
	v_dual_fmac_f32 v78, v65, v93 :: v_dual_fmac_f32 v79, v75, v93
	v_dual_fmac_f32 v48, v46, v86 :: v_dual_fmac_f32 v49, v56, v86
	v_dual_fmac_f32 v58, v66, v86 :: v_dual_fmac_f32 v59, v76, v86
	v_dual_fmac_f32 v68, v46, v94 :: v_dual_fmac_f32 v69, v56, v94
	v_dual_fmac_f32 v78, v66, v94 :: v_dual_fmac_f32 v79, v76, v94
	v_dual_fmac_f32 v48, v47, v87 :: v_dual_fmac_f32 v49, v57, v87
	v_dual_fmac_f32 v58, v67, v87 :: v_dual_fmac_f32 v59, v77, v87
	v_dual_fmac_f32 v68, v47, v95 :: v_dual_fmac_f32 v69, v57, v95
	v_dual_fmac_f32 v78, v67, v95 :: v_dual_fmac_f32 v79, v77, v95
	v_dual_add_f32 v164, v164, v48 :: v_dual_add_f32 v165, v165, v49
	v_dual_add_f32 v166, v166, v58 :: v_dual_add_f32 v167, v167, v59
	v_dual_add_f32 v180, v180, v68 :: v_dual_add_f32 v181, v181, v69
	v_dual_add_f32 v182, v182, v78 :: v_dual_add_f32 v183, v183, v79
	s_wait_loadcnt 0x7
	v_dual_mul_f32 v48, v41, v97 :: v_dual_mul_f32 v49, v51, v97
	v_dual_mul_f32 v58, v61, v97 :: v_dual_mul_f32 v59, v71, v97
	s_wait_loadcnt 0x5
	v_dual_mul_f32 v68, v41, v105 :: v_dual_mul_f32 v69, v51, v105
	v_dual_mul_f32 v78, v61, v105 :: v_dual_mul_f32 v79, v71, v105
	v_dual_fmac_f32 v48, v40, v96 :: v_dual_fmac_f32 v49, v50, v96
	v_dual_fmac_f32 v58, v60, v96 :: v_dual_fmac_f32 v59, v70, v96
	v_dual_fmac_f32 v68, v40, v104 :: v_dual_fmac_f32 v69, v50, v104
	v_dual_fmac_f32 v78, v60, v104 :: v_dual_fmac_f32 v79, v70, v104
	v_dual_fmac_f32 v48, v42, v98 :: v_dual_fmac_f32 v49, v52, v98
	v_dual_fmac_f32 v58, v62, v98 :: v_dual_fmac_f32 v59, v72, v98
	v_dual_fmac_f32 v68, v42, v106 :: v_dual_fmac_f32 v69, v52, v106
	v_dual_fmac_f32 v78, v62, v106 :: v_dual_fmac_f32 v79, v72, v106
	v_dual_fmac_f32 v48, v43, v99 :: v_dual_fmac_f32 v49, v53, v99
	v_dual_fmac_f32 v58, v63, v99 :: v_dual_fmac_f32 v59, v73, v99
	v_dual_fmac_f32 v68, v43, v107 :: v_dual_fmac_f32 v69, v53, v107
	v_dual_fmac_f32 v78, v63, v107 :: v_dual_fmac_f32 v79, v73, v107
	v_dual_fmac_f32 v48, v44, v100 :: v_dual_fmac_f32 v49, v54, v100
	v_dual_fmac_f32 v58, v64, v100 :: v_dual_fmac_f32 v59, v74, v100
	s_wait_loadcnt 0x4
	v_dual_fmac_f32 v68, v44, v108 :: v_dual_fmac_f32 v69, v54, v108
	v_dual_fmac_f32 v78, v64, v108 :: v_dual_fmac_f32 v79, v74, v108
	v_dual_fmac_f32 v48, v45, v101 :: v_dual_fmac_f32 v49, v55, v101
	v_dual_fmac_f32 v58, v65, v101 :: v_dual_fmac_f32 v59, v75, v101
	v_dual_fmac_f32 v68, v45, v109 :: v_dual_fmac_f32 v69, v55, v109
	v_dual_fmac_f32 v78, v65, v109 :: v_dual_fmac_f32 v79, v75, v109
	v_dual_fmac_f32 v48, v46, v102 :: v_dual_fmac_f32 v49, v56, v102
	v_dual_fmac_f32 v58, v66, v102 :: v_dual_fmac_f32 v59, v76, v102
	v_dual_fmac_f32 v68, v46, v110 :: v_dual_fmac_f32 v69, v56, v110
	v_dual_fmac_f32 v78, v66, v110 :: v_dual_fmac_f32 v79, v76, v110
	v_dual_fmac_f32 v48, v47, v103 :: v_dual_fmac_f32 v49, v57, v103
	v_dual_fmac_f32 v58, v67, v103 :: v_dual_fmac_f32 v59, v77, v103
	v_dual_fmac_f32 v68, v47, v111 :: v_dual_fmac_f32 v69, v57, v111
	v_dual_fmac_f32 v78, v67, v111 :: v_dual_fmac_f32 v79, v77, v111
	v_dual_add_f32 v196, v196, v48 :: v_dual_add_f32 v197, v197, v49
	v_dual_add_f32 v198, v198, v58 :: v_dual_add_f32 v199, v199, v59
	v_dual_add_f32 v212, v212, v68 :: v_dual_add_f32 v213, v213, v69
	v_dual_add_f32 v214, v214, v78 :: v_dual_add_f32 v215, v215, v79
	s_wait_loadcnt 0x3
	v_dual_mul_f32 v48, v41, v113 :: v_dual_mul_f32 v49, v51, v113
	v_dual_mul_f32 v58, v61, v113 :: v_dual_mul_f32 v59, v71, v113
	s_wait_loadcnt 0x1
	v_dual_mul_f32 v68, v41, v121 :: v_dual_mul_f32 v69, v51, v121
	v_dual_mul_f32 v78, v61, v121 :: v_dual_mul_f32 v79, v71, v121
	v_dual_fmac_f32 v48, v40, v112 :: v_dual_fmac_f32 v49, v50, v112
	v_dual_fmac_f32 v58, v60, v112 :: v_dual_fmac_f32 v59, v70, v112
	v_dual_fmac_f32 v68, v40, v120 :: v_dual_fmac_f32 v69, v50, v120
	v_dual_fmac_f32 v78, v60, v120 :: v_dual_fmac_f32 v79, v70, v120
	v_dual_fmac_f32 v48, v42, v114 :: v_dual_fmac_f32 v49, v52, v114
	v_dual_fmac_f32 v58, v62, v114 :: v_dual_fmac_f32 v59, v72, v114
	v_dual_fmac_f32 v68, v42, v122 :: v_dual_fmac_f32 v69, v52, v122
	v_dual_fmac_f32 v78, v62, v122 :: v_dual_fmac_f32 v79, v72, v122
	v_dual_fmac_f32 v48, v43, v115 :: v_dual_fmac_f32 v49, v53, v115
	v_dual_fmac_f32 v58, v63, v115 :: v_dual_fmac_f32 v59, v73, v115
	v_dual_fmac_f32 v68, v43, v123 :: v_dual_fmac_f32 v69, v53, v123
	v_dual_fmac_f32 v78, v63, v123 :: v_dual_fmac_f32 v79, v73, v123
	v_dual_fmac_f32 v48, v44, v116 :: v_dual_fmac_f32 v49, v54, v116
	v_dual_fmac_f32 v58, v64, v116 :: v_dual_fmac_f32 v59, v74, v116
	s_wait_loadcnt 0x0
	v_dual_fmac_f32 v68, v44, v124 :: v_dual_fmac_f32 v69, v54, v124
	v_dual_fmac_f32 v78, v64, v124 :: v_dual_fmac_f32 v79, v74, v124
	v_dual_fmac_f32 v48, v45, v117 :: v_dual_fmac_f32 v49, v55, v117
	v_dual_fmac_f32 v58, v65, v117 :: v_dual_fmac_f32 v59, v75, v117
	v_dual_fmac_f32 v68, v45, v125 :: v_dual_fmac_f32 v69, v55, v125
	v_dual_fmac_f32 v78, v65, v125 :: v_dual_fmac_f32 v79, v75, v125
	v_dual_fmac_f32 v48, v46, v118 :: v_dual_fmac_f32 v49, v56, v118
	v_dual_fmac_f32 v58, v66, v118 :: v_dual_fmac_f32 v59, v76, v118
	v_dual_fmac_f32 v68, v46, v126 :: v_dual_fmac_f32 v69, v56, v126
	v_dual_fmac_f32 v78, v66, v126 :: v_dual_fmac_f32 v79, v76, v126
	v_dual_fmac_f32 v48, v47, v119 :: v_dual_fmac_f32 v49, v57, v119
	v_dual_fmac_f32 v58, v67, v119 :: v_dual_fmac_f32 v59, v77, v119
	v_dual_fmac_f32 v68, v47, v127 :: v_dual_fmac_f32 v69, v57, v127
	v_dual_fmac_f32 v78, v67, v127 :: v_dual_fmac_f32 v79, v77, v127
	v_dual_add_f32 v228, v228, v48 :: v_dual_add_f32 v229, v229, v49
	v_dual_add_f32 v230, v230, v58 :: v_dual_add_f32 v231, v231, v59
	v_dual_add_f32 v244, v244, v68 :: v_dual_add_f32 v245, v245, v69
	v_dual_add_f32 v246, v246, v78 :: v_dual_add_f32 v247, v247, v79
	s_cmp_gt_u32 s37, 2
	s_cbranch_scc0 .Lpx_b6_fold
	buffer_load_b32 v16, v1, s[20:23], s16 offen offset:272
	buffer_load_b32 v17, v1, s[20:23], s17 offen offset:272
	buffer_load_b32 v18, v1, s[20:23], s18 offen offset:272
	buffer_load_b32 v19, v1, s[20:23], s19 offen offset:272
	buffer_load_b32 v32, v2, s[20:23], s16 offen offset:280
	buffer_load_b32 v33, v2, s[20:23], s17 offen offset:280
	buffer_load_b32 v34, v2, s[20:23], s18 offen offset:280
	buffer_load_b32 v35, v2, s[20:23], s19 offen offset:280
	buffer_load_b128 v[80:83], v3, s[24:27], s28 offen offset:2048
	buffer_load_b128 v[84:87], v3, s[24:27], s28 offen offset:2064
	buffer_load_b128 v[88:91], v3, s[24:27], s29 offen offset:2048
	buffer_load_b128 v[92:95], v3, s[24:27], s29 offen offset:2064
	buffer_load_b128 v[96:99], v3, s[24:27], s30 offen offset:2048
	buffer_load_b128 v[100:103], v3, s[24:27], s30 offen offset:2064
	buffer_load_b128 v[104:107], v3, s[24:27], s31 offen offset:2048
	buffer_load_b128 v[108:111], v3, s[24:27], s31 offen offset:2064
	buffer_load_b128 v[112:115], v3, s[24:27], s32 offen offset:2048
	buffer_load_b128 v[116:119], v3, s[24:27], s32 offen offset:2064
	buffer_load_b128 v[120:123], v3, s[24:27], s33 offen offset:2048
	buffer_load_b128 v[124:127], v3, s[24:27], s33 offen offset:2064
	s_wait_loadcnt 0xf
	v_and_b32_e32 v46, 0xf0f0f0f, v32
	s_wait_loadcnt 0xe
	v_and_b32_e32 v56, 0xf0f0f0f, v33
	s_wait_loadcnt 0xd
	v_and_b32_e32 v66, 0xf0f0f0f, v34
	s_wait_loadcnt 0xc
	v_and_b32_e32 v76, 0xf0f0f0f, v35
	v_lshrrev_b32_e32 v47, 4, v32
	v_lshrrev_b32_e32 v57, 4, v33
	v_lshrrev_b32_e32 v67, 4, v34
	v_lshrrev_b32_e32 v77, 4, v35
	v_and_b32_e32 v47, 0xf0f0f0f, v47
	v_and_b32_e32 v57, 0xf0f0f0f, v57
	v_and_b32_e32 v67, 0xf0f0f0f, v67
	v_and_b32_e32 v77, 0xf0f0f0f, v77
	v_cvt_f32_ubyte0_e32 v40, v46
	v_cvt_f32_ubyte0_e32 v50, v56
	v_cvt_f32_ubyte0_e32 v60, v66
	v_cvt_f32_ubyte0_e32 v70, v76
	v_cvt_f32_ubyte1_e32 v42, v46
	v_cvt_f32_ubyte1_e32 v52, v56
	v_cvt_f32_ubyte1_e32 v62, v66
	v_cvt_f32_ubyte1_e32 v72, v76
	v_cvt_f32_ubyte2_e32 v44, v46
	v_cvt_f32_ubyte2_e32 v54, v56
	v_cvt_f32_ubyte2_e32 v64, v66
	v_cvt_f32_ubyte2_e32 v74, v76
	v_cvt_f32_ubyte3_e32 v46, v46
	v_cvt_f32_ubyte3_e32 v56, v56
	v_cvt_f32_ubyte3_e32 v66, v66
	v_cvt_f32_ubyte3_e32 v76, v76
	v_cvt_f32_ubyte0_e32 v41, v47
	v_cvt_f32_ubyte0_e32 v51, v57
	v_cvt_f32_ubyte0_e32 v61, v67
	v_cvt_f32_ubyte0_e32 v71, v77
	v_cvt_f32_ubyte1_e32 v43, v47
	v_cvt_f32_ubyte1_e32 v53, v57
	v_cvt_f32_ubyte1_e32 v63, v67
	v_cvt_f32_ubyte1_e32 v73, v77
	v_cvt_f32_ubyte2_e32 v45, v47
	v_cvt_f32_ubyte2_e32 v55, v57
	v_cvt_f32_ubyte2_e32 v65, v67
	v_cvt_f32_ubyte2_e32 v75, v77
	v_cvt_f32_ubyte3_e32 v47, v47
	v_cvt_f32_ubyte3_e32 v57, v57
	v_cvt_f32_ubyte3_e32 v67, v67
	v_cvt_f32_ubyte3_e32 v77, v77
	v_fma_mix_f32 v40, v16, v40, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v17, v50, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v18, v60, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v70, v19, v70, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v41, v16, v41, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v17, v51, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v18, v61, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v71, v19, v71, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v42, v16, v42, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v52, v17, v52, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v62, v18, v62, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v72, v19, v72, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v43, v16, v43, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v53, v17, v53, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v63, v18, v63, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v73, v19, v73, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v44, v16, v44, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v54, v17, v54, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v64, v18, v64, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v74, v19, v74, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v16, v45, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v17, v55, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v65, v18, v65, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v75, v19, v75, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v16, v46, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v17, v56, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v66, v18, v66, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v76, v19, v76, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v16, v47, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v17, v57, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v67, v18, v67, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v77, v19, v77, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xb
	v_dual_mul_f32 v48, v41, v81 :: v_dual_mul_f32 v49, v51, v81
	v_dual_mul_f32 v58, v61, v81 :: v_dual_mul_f32 v59, v71, v81
	s_wait_loadcnt 0x9
	v_dual_mul_f32 v68, v41, v89 :: v_dual_mul_f32 v69, v51, v89
	v_dual_mul_f32 v78, v61, v89 :: v_dual_mul_f32 v79, v71, v89
	v_dual_fmac_f32 v48, v40, v80 :: v_dual_fmac_f32 v49, v50, v80
	v_dual_fmac_f32 v58, v60, v80 :: v_dual_fmac_f32 v59, v70, v80
	v_dual_fmac_f32 v68, v40, v88 :: v_dual_fmac_f32 v69, v50, v88
	v_dual_fmac_f32 v78, v60, v88 :: v_dual_fmac_f32 v79, v70, v88
	v_dual_fmac_f32 v48, v42, v82 :: v_dual_fmac_f32 v49, v52, v82
	v_dual_fmac_f32 v58, v62, v82 :: v_dual_fmac_f32 v59, v72, v82
	v_dual_fmac_f32 v68, v42, v90 :: v_dual_fmac_f32 v69, v52, v90
	v_dual_fmac_f32 v78, v62, v90 :: v_dual_fmac_f32 v79, v72, v90
	v_dual_fmac_f32 v48, v43, v83 :: v_dual_fmac_f32 v49, v53, v83
	v_dual_fmac_f32 v58, v63, v83 :: v_dual_fmac_f32 v59, v73, v83
	v_dual_fmac_f32 v68, v43, v91 :: v_dual_fmac_f32 v69, v53, v91
	v_dual_fmac_f32 v78, v63, v91 :: v_dual_fmac_f32 v79, v73, v91
	v_dual_fmac_f32 v48, v44, v84 :: v_dual_fmac_f32 v49, v54, v84
	v_dual_fmac_f32 v58, v64, v84 :: v_dual_fmac_f32 v59, v74, v84
	s_wait_loadcnt 0x8
	v_dual_fmac_f32 v68, v44, v92 :: v_dual_fmac_f32 v69, v54, v92
	v_dual_fmac_f32 v78, v64, v92 :: v_dual_fmac_f32 v79, v74, v92
	v_dual_fmac_f32 v48, v45, v85 :: v_dual_fmac_f32 v49, v55, v85
	v_dual_fmac_f32 v58, v65, v85 :: v_dual_fmac_f32 v59, v75, v85
	v_dual_fmac_f32 v68, v45, v93 :: v_dual_fmac_f32 v69, v55, v93
	v_dual_fmac_f32 v78, v65, v93 :: v_dual_fmac_f32 v79, v75, v93
	v_dual_fmac_f32 v48, v46, v86 :: v_dual_fmac_f32 v49, v56, v86
	v_dual_fmac_f32 v58, v66, v86 :: v_dual_fmac_f32 v59, v76, v86
	v_dual_fmac_f32 v68, v46, v94 :: v_dual_fmac_f32 v69, v56, v94
	v_dual_fmac_f32 v78, v66, v94 :: v_dual_fmac_f32 v79, v76, v94
	v_dual_fmac_f32 v48, v47, v87 :: v_dual_fmac_f32 v49, v57, v87
	v_dual_fmac_f32 v58, v67, v87 :: v_dual_fmac_f32 v59, v77, v87
	v_dual_fmac_f32 v68, v47, v95 :: v_dual_fmac_f32 v69, v57, v95
	v_dual_fmac_f32 v78, v67, v95 :: v_dual_fmac_f32 v79, v77, v95
	v_dual_add_f32 v168, v168, v48 :: v_dual_add_f32 v169, v169, v49
	v_dual_add_f32 v170, v170, v58 :: v_dual_add_f32 v171, v171, v59
	v_dual_add_f32 v184, v184, v68 :: v_dual_add_f32 v185, v185, v69
	v_dual_add_f32 v186, v186, v78 :: v_dual_add_f32 v187, v187, v79
	s_wait_loadcnt 0x7
	v_dual_mul_f32 v48, v41, v97 :: v_dual_mul_f32 v49, v51, v97
	v_dual_mul_f32 v58, v61, v97 :: v_dual_mul_f32 v59, v71, v97
	s_wait_loadcnt 0x5
	v_dual_mul_f32 v68, v41, v105 :: v_dual_mul_f32 v69, v51, v105
	v_dual_mul_f32 v78, v61, v105 :: v_dual_mul_f32 v79, v71, v105
	v_dual_fmac_f32 v48, v40, v96 :: v_dual_fmac_f32 v49, v50, v96
	v_dual_fmac_f32 v58, v60, v96 :: v_dual_fmac_f32 v59, v70, v96
	v_dual_fmac_f32 v68, v40, v104 :: v_dual_fmac_f32 v69, v50, v104
	v_dual_fmac_f32 v78, v60, v104 :: v_dual_fmac_f32 v79, v70, v104
	v_dual_fmac_f32 v48, v42, v98 :: v_dual_fmac_f32 v49, v52, v98
	v_dual_fmac_f32 v58, v62, v98 :: v_dual_fmac_f32 v59, v72, v98
	v_dual_fmac_f32 v68, v42, v106 :: v_dual_fmac_f32 v69, v52, v106
	v_dual_fmac_f32 v78, v62, v106 :: v_dual_fmac_f32 v79, v72, v106
	v_dual_fmac_f32 v48, v43, v99 :: v_dual_fmac_f32 v49, v53, v99
	v_dual_fmac_f32 v58, v63, v99 :: v_dual_fmac_f32 v59, v73, v99
	v_dual_fmac_f32 v68, v43, v107 :: v_dual_fmac_f32 v69, v53, v107
	v_dual_fmac_f32 v78, v63, v107 :: v_dual_fmac_f32 v79, v73, v107
	v_dual_fmac_f32 v48, v44, v100 :: v_dual_fmac_f32 v49, v54, v100
	v_dual_fmac_f32 v58, v64, v100 :: v_dual_fmac_f32 v59, v74, v100
	s_wait_loadcnt 0x4
	v_dual_fmac_f32 v68, v44, v108 :: v_dual_fmac_f32 v69, v54, v108
	v_dual_fmac_f32 v78, v64, v108 :: v_dual_fmac_f32 v79, v74, v108
	v_dual_fmac_f32 v48, v45, v101 :: v_dual_fmac_f32 v49, v55, v101
	v_dual_fmac_f32 v58, v65, v101 :: v_dual_fmac_f32 v59, v75, v101
	v_dual_fmac_f32 v68, v45, v109 :: v_dual_fmac_f32 v69, v55, v109
	v_dual_fmac_f32 v78, v65, v109 :: v_dual_fmac_f32 v79, v75, v109
	v_dual_fmac_f32 v48, v46, v102 :: v_dual_fmac_f32 v49, v56, v102
	v_dual_fmac_f32 v58, v66, v102 :: v_dual_fmac_f32 v59, v76, v102
	v_dual_fmac_f32 v68, v46, v110 :: v_dual_fmac_f32 v69, v56, v110
	v_dual_fmac_f32 v78, v66, v110 :: v_dual_fmac_f32 v79, v76, v110
	v_dual_fmac_f32 v48, v47, v103 :: v_dual_fmac_f32 v49, v57, v103
	v_dual_fmac_f32 v58, v67, v103 :: v_dual_fmac_f32 v59, v77, v103
	v_dual_fmac_f32 v68, v47, v111 :: v_dual_fmac_f32 v69, v57, v111
	v_dual_fmac_f32 v78, v67, v111 :: v_dual_fmac_f32 v79, v77, v111
	v_dual_add_f32 v200, v200, v48 :: v_dual_add_f32 v201, v201, v49
	v_dual_add_f32 v202, v202, v58 :: v_dual_add_f32 v203, v203, v59
	v_dual_add_f32 v216, v216, v68 :: v_dual_add_f32 v217, v217, v69
	v_dual_add_f32 v218, v218, v78 :: v_dual_add_f32 v219, v219, v79
	s_wait_loadcnt 0x3
	v_dual_mul_f32 v48, v41, v113 :: v_dual_mul_f32 v49, v51, v113
	v_dual_mul_f32 v58, v61, v113 :: v_dual_mul_f32 v59, v71, v113
	s_wait_loadcnt 0x1
	v_dual_mul_f32 v68, v41, v121 :: v_dual_mul_f32 v69, v51, v121
	v_dual_mul_f32 v78, v61, v121 :: v_dual_mul_f32 v79, v71, v121
	v_dual_fmac_f32 v48, v40, v112 :: v_dual_fmac_f32 v49, v50, v112
	v_dual_fmac_f32 v58, v60, v112 :: v_dual_fmac_f32 v59, v70, v112
	v_dual_fmac_f32 v68, v40, v120 :: v_dual_fmac_f32 v69, v50, v120
	v_dual_fmac_f32 v78, v60, v120 :: v_dual_fmac_f32 v79, v70, v120
	v_dual_fmac_f32 v48, v42, v114 :: v_dual_fmac_f32 v49, v52, v114
	v_dual_fmac_f32 v58, v62, v114 :: v_dual_fmac_f32 v59, v72, v114
	v_dual_fmac_f32 v68, v42, v122 :: v_dual_fmac_f32 v69, v52, v122
	v_dual_fmac_f32 v78, v62, v122 :: v_dual_fmac_f32 v79, v72, v122
	v_dual_fmac_f32 v48, v43, v115 :: v_dual_fmac_f32 v49, v53, v115
	v_dual_fmac_f32 v58, v63, v115 :: v_dual_fmac_f32 v59, v73, v115
	v_dual_fmac_f32 v68, v43, v123 :: v_dual_fmac_f32 v69, v53, v123
	v_dual_fmac_f32 v78, v63, v123 :: v_dual_fmac_f32 v79, v73, v123
	v_dual_fmac_f32 v48, v44, v116 :: v_dual_fmac_f32 v49, v54, v116
	v_dual_fmac_f32 v58, v64, v116 :: v_dual_fmac_f32 v59, v74, v116
	s_wait_loadcnt 0x0
	v_dual_fmac_f32 v68, v44, v124 :: v_dual_fmac_f32 v69, v54, v124
	v_dual_fmac_f32 v78, v64, v124 :: v_dual_fmac_f32 v79, v74, v124
	v_dual_fmac_f32 v48, v45, v117 :: v_dual_fmac_f32 v49, v55, v117
	v_dual_fmac_f32 v58, v65, v117 :: v_dual_fmac_f32 v59, v75, v117
	v_dual_fmac_f32 v68, v45, v125 :: v_dual_fmac_f32 v69, v55, v125
	v_dual_fmac_f32 v78, v65, v125 :: v_dual_fmac_f32 v79, v75, v125
	v_dual_fmac_f32 v48, v46, v118 :: v_dual_fmac_f32 v49, v56, v118
	v_dual_fmac_f32 v58, v66, v118 :: v_dual_fmac_f32 v59, v76, v118
	v_dual_fmac_f32 v68, v46, v126 :: v_dual_fmac_f32 v69, v56, v126
	v_dual_fmac_f32 v78, v66, v126 :: v_dual_fmac_f32 v79, v76, v126
	v_dual_fmac_f32 v48, v47, v119 :: v_dual_fmac_f32 v49, v57, v119
	v_dual_fmac_f32 v58, v67, v119 :: v_dual_fmac_f32 v59, v77, v119
	v_dual_fmac_f32 v68, v47, v127 :: v_dual_fmac_f32 v69, v57, v127
	v_dual_fmac_f32 v78, v67, v127 :: v_dual_fmac_f32 v79, v77, v127
	v_dual_add_f32 v232, v232, v48 :: v_dual_add_f32 v233, v233, v49
	v_dual_add_f32 v234, v234, v58 :: v_dual_add_f32 v235, v235, v59
	v_dual_add_f32 v248, v248, v68 :: v_dual_add_f32 v249, v249, v69
	v_dual_add_f32 v250, v250, v78 :: v_dual_add_f32 v251, v251, v79
	.Lpx_b6_fold:
	v_dual_add_f32 v160, v160, v164 :: v_dual_add_f32 v161, v161, v165
	v_dual_add_f32 v162, v162, v166 :: v_dual_add_f32 v163, v163, v167
	v_dual_add_f32 v168, v168, v172 :: v_dual_add_f32 v169, v169, v173
	v_dual_add_f32 v170, v170, v174 :: v_dual_add_f32 v171, v171, v175
	v_dual_add_f32 v160, v160, v168 :: v_dual_add_f32 v161, v161, v169
	v_dual_add_f32 v162, v162, v170 :: v_dual_add_f32 v163, v163, v171
	v_dual_add_f32 v176, v176, v180 :: v_dual_add_f32 v177, v177, v181
	v_dual_add_f32 v178, v178, v182 :: v_dual_add_f32 v179, v179, v183
	v_dual_add_f32 v184, v184, v188 :: v_dual_add_f32 v185, v185, v189
	v_dual_add_f32 v186, v186, v190 :: v_dual_add_f32 v187, v187, v191
	v_dual_add_f32 v176, v176, v184 :: v_dual_add_f32 v177, v177, v185
	v_dual_add_f32 v178, v178, v186 :: v_dual_add_f32 v179, v179, v187
	v_dual_add_f32 v192, v192, v196 :: v_dual_add_f32 v193, v193, v197
	v_dual_add_f32 v194, v194, v198 :: v_dual_add_f32 v195, v195, v199
	v_dual_add_f32 v200, v200, v204 :: v_dual_add_f32 v201, v201, v205
	v_dual_add_f32 v202, v202, v206 :: v_dual_add_f32 v203, v203, v207
	v_dual_add_f32 v192, v192, v200 :: v_dual_add_f32 v193, v193, v201
	v_dual_add_f32 v194, v194, v202 :: v_dual_add_f32 v195, v195, v203
	v_dual_add_f32 v208, v208, v212 :: v_dual_add_f32 v209, v209, v213
	v_dual_add_f32 v210, v210, v214 :: v_dual_add_f32 v211, v211, v215
	v_dual_add_f32 v216, v216, v220 :: v_dual_add_f32 v217, v217, v221
	v_dual_add_f32 v218, v218, v222 :: v_dual_add_f32 v219, v219, v223
	v_dual_add_f32 v208, v208, v216 :: v_dual_add_f32 v209, v209, v217
	v_dual_add_f32 v210, v210, v218 :: v_dual_add_f32 v211, v211, v219
	v_dual_add_f32 v224, v224, v228 :: v_dual_add_f32 v225, v225, v229
	v_dual_add_f32 v226, v226, v230 :: v_dual_add_f32 v227, v227, v231
	v_dual_add_f32 v232, v232, v236 :: v_dual_add_f32 v233, v233, v237
	v_dual_add_f32 v234, v234, v238 :: v_dual_add_f32 v235, v235, v239
	v_dual_add_f32 v224, v224, v232 :: v_dual_add_f32 v225, v225, v233
	v_dual_add_f32 v226, v226, v234 :: v_dual_add_f32 v227, v227, v235
	v_dual_add_f32 v240, v240, v244 :: v_dual_add_f32 v241, v241, v245
	v_dual_add_f32 v242, v242, v246 :: v_dual_add_f32 v243, v243, v247
	v_dual_add_f32 v248, v248, v252 :: v_dual_add_f32 v249, v249, v253
	v_dual_add_f32 v250, v250, v254 :: v_dual_add_f32 v251, v251, v255
	v_dual_add_f32 v240, v240, v248 :: v_dual_add_f32 v241, v241, v249
	v_dual_add_f32 v242, v242, v250 :: v_dual_add_f32 v243, v243, v251
	ds_swizzle_b32 v8, v160 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v9, v161 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v10, v162 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v11, v163 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v12, v176 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v13, v177 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v14, v178 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v15, v179 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v16, v192 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v17, v193 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v18, v194 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v19, v195 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v20, v208 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v21, v209 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v22, v210 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v23, v211 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v24, v224 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v25, v225 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v26, v226 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v27, v227 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v28, v240 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v29, v241 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v30, v242 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v31, v243 offset:swizzle(BITMASK_PERM,"1pppp")
	s_wait_dscnt 0x17
	v_add_f32_e32 v160, v160, v8
	s_wait_dscnt 0x16
	v_add_f32_e32 v161, v161, v9
	s_wait_dscnt 0x15
	v_add_f32_e32 v162, v162, v10
	s_wait_dscnt 0x14
	v_add_f32_e32 v163, v163, v11
	s_wait_dscnt 0x13
	v_add_f32_e32 v176, v176, v12
	s_wait_dscnt 0x12
	v_add_f32_e32 v177, v177, v13
	s_wait_dscnt 0x11
	v_add_f32_e32 v178, v178, v14
	s_wait_dscnt 0x10
	v_add_f32_e32 v179, v179, v15
	s_wait_dscnt 0xf
	v_add_f32_e32 v192, v192, v16
	s_wait_dscnt 0xe
	v_add_f32_e32 v193, v193, v17
	s_wait_dscnt 0xd
	v_add_f32_e32 v194, v194, v18
	s_wait_dscnt 0xc
	v_add_f32_e32 v195, v195, v19
	s_wait_dscnt 0xb
	v_add_f32_e32 v208, v208, v20
	s_wait_dscnt 0xa
	v_add_f32_e32 v209, v209, v21
	s_wait_dscnt 0x9
	v_add_f32_e32 v210, v210, v22
	s_wait_dscnt 0x8
	v_add_f32_e32 v211, v211, v23
	s_wait_dscnt 0x7
	v_add_f32_e32 v224, v224, v24
	s_wait_dscnt 0x6
	v_add_f32_e32 v225, v225, v25
	s_wait_dscnt 0x5
	v_add_f32_e32 v226, v226, v26
	s_wait_dscnt 0x4
	v_add_f32_e32 v227, v227, v27
	s_wait_dscnt 0x3
	v_add_f32_e32 v240, v240, v28
	s_wait_dscnt 0x2
	v_add_f32_e32 v241, v241, v29
	s_wait_dscnt 0x1
	v_add_f32_e32 v242, v242, v30
	s_wait_dscnt 0x0
	v_add_f32_e32 v243, v243, v31
	v_cmp_gt_u32_e32 vcc_lo, 24, v0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v4, 0, 8, vcc_lo
	v_add_lshl_u32 v4, v4, v0, 2
	ds_bpermute_b32 v8, v4, v160
	ds_bpermute_b32 v9, v4, v161
	ds_bpermute_b32 v10, v4, v162
	ds_bpermute_b32 v11, v4, v163
	ds_bpermute_b32 v12, v4, v176
	ds_bpermute_b32 v13, v4, v177
	ds_bpermute_b32 v14, v4, v178
	ds_bpermute_b32 v15, v4, v179
	ds_bpermute_b32 v16, v4, v192
	ds_bpermute_b32 v17, v4, v193
	ds_bpermute_b32 v18, v4, v194
	ds_bpermute_b32 v19, v4, v195
	ds_bpermute_b32 v20, v4, v208
	ds_bpermute_b32 v21, v4, v209
	ds_bpermute_b32 v22, v4, v210
	ds_bpermute_b32 v23, v4, v211
	ds_bpermute_b32 v24, v4, v224
	ds_bpermute_b32 v25, v4, v225
	ds_bpermute_b32 v26, v4, v226
	ds_bpermute_b32 v27, v4, v227
	ds_bpermute_b32 v28, v4, v240
	ds_bpermute_b32 v29, v4, v241
	ds_bpermute_b32 v30, v4, v242
	ds_bpermute_b32 v31, v4, v243
	s_wait_dscnt 0x17
	v_add_f32_e32 v160, v160, v8
	s_wait_dscnt 0x16
	v_add_f32_e32 v161, v161, v9
	s_wait_dscnt 0x15
	v_add_f32_e32 v162, v162, v10
	s_wait_dscnt 0x14
	v_add_f32_e32 v163, v163, v11
	s_wait_dscnt 0x13
	v_add_f32_e32 v176, v176, v12
	s_wait_dscnt 0x12
	v_add_f32_e32 v177, v177, v13
	s_wait_dscnt 0x11
	v_add_f32_e32 v178, v178, v14
	s_wait_dscnt 0x10
	v_add_f32_e32 v179, v179, v15
	s_wait_dscnt 0xf
	v_add_f32_e32 v192, v192, v16
	s_wait_dscnt 0xe
	v_add_f32_e32 v193, v193, v17
	s_wait_dscnt 0xd
	v_add_f32_e32 v194, v194, v18
	s_wait_dscnt 0xc
	v_add_f32_e32 v195, v195, v19
	s_wait_dscnt 0xb
	v_add_f32_e32 v208, v208, v20
	s_wait_dscnt 0xa
	v_add_f32_e32 v209, v209, v21
	s_wait_dscnt 0x9
	v_add_f32_e32 v210, v210, v22
	s_wait_dscnt 0x8
	v_add_f32_e32 v211, v211, v23
	s_wait_dscnt 0x7
	v_add_f32_e32 v224, v224, v24
	s_wait_dscnt 0x6
	v_add_f32_e32 v225, v225, v25
	s_wait_dscnt 0x5
	v_add_f32_e32 v226, v226, v26
	s_wait_dscnt 0x4
	v_add_f32_e32 v227, v227, v27
	s_wait_dscnt 0x3
	v_add_f32_e32 v240, v240, v28
	s_wait_dscnt 0x2
	v_add_f32_e32 v241, v241, v29
	s_wait_dscnt 0x1
	v_add_f32_e32 v242, v242, v30
	s_wait_dscnt 0x0
	v_add_f32_e32 v243, v243, v31
	v_cmp_gt_u32_e32 vcc_lo, 28, v0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v4, 0, 4, vcc_lo
	v_add_lshl_u32 v4, v4, v0, 2
	ds_bpermute_b32 v8, v4, v160
	ds_bpermute_b32 v9, v4, v161
	ds_bpermute_b32 v10, v4, v162
	ds_bpermute_b32 v11, v4, v163
	ds_bpermute_b32 v12, v4, v176
	ds_bpermute_b32 v13, v4, v177
	ds_bpermute_b32 v14, v4, v178
	ds_bpermute_b32 v15, v4, v179
	ds_bpermute_b32 v16, v4, v192
	ds_bpermute_b32 v17, v4, v193
	ds_bpermute_b32 v18, v4, v194
	ds_bpermute_b32 v19, v4, v195
	ds_bpermute_b32 v20, v4, v208
	ds_bpermute_b32 v21, v4, v209
	ds_bpermute_b32 v22, v4, v210
	ds_bpermute_b32 v23, v4, v211
	ds_bpermute_b32 v24, v4, v224
	ds_bpermute_b32 v25, v4, v225
	ds_bpermute_b32 v26, v4, v226
	ds_bpermute_b32 v27, v4, v227
	ds_bpermute_b32 v28, v4, v240
	ds_bpermute_b32 v29, v4, v241
	ds_bpermute_b32 v30, v4, v242
	ds_bpermute_b32 v31, v4, v243
	s_wait_dscnt 0x17
	v_add_f32_e32 v160, v160, v8
	s_wait_dscnt 0x16
	v_add_f32_e32 v161, v161, v9
	s_wait_dscnt 0x15
	v_add_f32_e32 v162, v162, v10
	s_wait_dscnt 0x14
	v_add_f32_e32 v163, v163, v11
	s_wait_dscnt 0x13
	v_add_f32_e32 v176, v176, v12
	s_wait_dscnt 0x12
	v_add_f32_e32 v177, v177, v13
	s_wait_dscnt 0x11
	v_add_f32_e32 v178, v178, v14
	s_wait_dscnt 0x10
	v_add_f32_e32 v179, v179, v15
	s_wait_dscnt 0xf
	v_add_f32_e32 v192, v192, v16
	s_wait_dscnt 0xe
	v_add_f32_e32 v193, v193, v17
	s_wait_dscnt 0xd
	v_add_f32_e32 v194, v194, v18
	s_wait_dscnt 0xc
	v_add_f32_e32 v195, v195, v19
	s_wait_dscnt 0xb
	v_add_f32_e32 v208, v208, v20
	s_wait_dscnt 0xa
	v_add_f32_e32 v209, v209, v21
	s_wait_dscnt 0x9
	v_add_f32_e32 v210, v210, v22
	s_wait_dscnt 0x8
	v_add_f32_e32 v211, v211, v23
	s_wait_dscnt 0x7
	v_add_f32_e32 v224, v224, v24
	s_wait_dscnt 0x6
	v_add_f32_e32 v225, v225, v25
	s_wait_dscnt 0x5
	v_add_f32_e32 v226, v226, v26
	s_wait_dscnt 0x4
	v_add_f32_e32 v227, v227, v27
	s_wait_dscnt 0x3
	v_add_f32_e32 v240, v240, v28
	s_wait_dscnt 0x2
	v_add_f32_e32 v241, v241, v29
	s_wait_dscnt 0x1
	v_add_f32_e32 v242, v242, v30
	s_wait_dscnt 0x0
	v_add_f32_e32 v243, v243, v31
	v_cmp_gt_u32_e32 vcc_lo, 30, v0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v4, 0, 2, vcc_lo
	v_add_lshl_u32 v4, v4, v0, 2
	ds_bpermute_b32 v8, v4, v160
	ds_bpermute_b32 v9, v4, v161
	ds_bpermute_b32 v10, v4, v162
	ds_bpermute_b32 v11, v4, v163
	ds_bpermute_b32 v12, v4, v176
	ds_bpermute_b32 v13, v4, v177
	ds_bpermute_b32 v14, v4, v178
	ds_bpermute_b32 v15, v4, v179
	ds_bpermute_b32 v16, v4, v192
	ds_bpermute_b32 v17, v4, v193
	ds_bpermute_b32 v18, v4, v194
	ds_bpermute_b32 v19, v4, v195
	ds_bpermute_b32 v20, v4, v208
	ds_bpermute_b32 v21, v4, v209
	ds_bpermute_b32 v22, v4, v210
	ds_bpermute_b32 v23, v4, v211
	ds_bpermute_b32 v24, v4, v224
	ds_bpermute_b32 v25, v4, v225
	ds_bpermute_b32 v26, v4, v226
	ds_bpermute_b32 v27, v4, v227
	ds_bpermute_b32 v28, v4, v240
	ds_bpermute_b32 v29, v4, v241
	ds_bpermute_b32 v30, v4, v242
	ds_bpermute_b32 v31, v4, v243
	s_wait_dscnt 0x17
	v_add_f32_e32 v160, v160, v8
	s_wait_dscnt 0x16
	v_add_f32_e32 v161, v161, v9
	s_wait_dscnt 0x15
	v_add_f32_e32 v162, v162, v10
	s_wait_dscnt 0x14
	v_add_f32_e32 v163, v163, v11
	s_wait_dscnt 0x13
	v_add_f32_e32 v176, v176, v12
	s_wait_dscnt 0x12
	v_add_f32_e32 v177, v177, v13
	s_wait_dscnt 0x11
	v_add_f32_e32 v178, v178, v14
	s_wait_dscnt 0x10
	v_add_f32_e32 v179, v179, v15
	s_wait_dscnt 0xf
	v_add_f32_e32 v192, v192, v16
	s_wait_dscnt 0xe
	v_add_f32_e32 v193, v193, v17
	s_wait_dscnt 0xd
	v_add_f32_e32 v194, v194, v18
	s_wait_dscnt 0xc
	v_add_f32_e32 v195, v195, v19
	s_wait_dscnt 0xb
	v_add_f32_e32 v208, v208, v20
	s_wait_dscnt 0xa
	v_add_f32_e32 v209, v209, v21
	s_wait_dscnt 0x9
	v_add_f32_e32 v210, v210, v22
	s_wait_dscnt 0x8
	v_add_f32_e32 v211, v211, v23
	s_wait_dscnt 0x7
	v_add_f32_e32 v224, v224, v24
	s_wait_dscnt 0x6
	v_add_f32_e32 v225, v225, v25
	s_wait_dscnt 0x5
	v_add_f32_e32 v226, v226, v26
	s_wait_dscnt 0x4
	v_add_f32_e32 v227, v227, v27
	s_wait_dscnt 0x3
	v_add_f32_e32 v240, v240, v28
	s_wait_dscnt 0x2
	v_add_f32_e32 v241, v241, v29
	s_wait_dscnt 0x1
	v_add_f32_e32 v242, v242, v30
	s_wait_dscnt 0x0
	v_add_f32_e32 v243, v243, v31
	v_cmp_gt_u32_e32 vcc_lo, 31, v0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v4, 0, 1, vcc_lo
	v_add_lshl_u32 v4, v4, v0, 2
	ds_bpermute_b32 v8, v4, v160
	ds_bpermute_b32 v9, v4, v161
	ds_bpermute_b32 v10, v4, v162
	ds_bpermute_b32 v11, v4, v163
	ds_bpermute_b32 v12, v4, v176
	ds_bpermute_b32 v13, v4, v177
	ds_bpermute_b32 v14, v4, v178
	ds_bpermute_b32 v15, v4, v179
	ds_bpermute_b32 v16, v4, v192
	ds_bpermute_b32 v17, v4, v193
	ds_bpermute_b32 v18, v4, v194
	ds_bpermute_b32 v19, v4, v195
	ds_bpermute_b32 v20, v4, v208
	ds_bpermute_b32 v21, v4, v209
	ds_bpermute_b32 v22, v4, v210
	ds_bpermute_b32 v23, v4, v211
	ds_bpermute_b32 v24, v4, v224
	ds_bpermute_b32 v25, v4, v225
	ds_bpermute_b32 v26, v4, v226
	ds_bpermute_b32 v27, v4, v227
	ds_bpermute_b32 v28, v4, v240
	ds_bpermute_b32 v29, v4, v241
	ds_bpermute_b32 v30, v4, v242
	ds_bpermute_b32 v31, v4, v243
	s_wait_dscnt 0x17
	v_add_f32_e32 v160, v160, v8
	s_wait_dscnt 0x16
	v_add_f32_e32 v161, v161, v9
	s_wait_dscnt 0x15
	v_add_f32_e32 v162, v162, v10
	s_wait_dscnt 0x14
	v_add_f32_e32 v163, v163, v11
	s_wait_dscnt 0x13
	v_add_f32_e32 v176, v176, v12
	s_wait_dscnt 0x12
	v_add_f32_e32 v177, v177, v13
	s_wait_dscnt 0x11
	v_add_f32_e32 v178, v178, v14
	s_wait_dscnt 0x10
	v_add_f32_e32 v179, v179, v15
	s_wait_dscnt 0xf
	v_add_f32_e32 v192, v192, v16
	s_wait_dscnt 0xe
	v_add_f32_e32 v193, v193, v17
	s_wait_dscnt 0xd
	v_add_f32_e32 v194, v194, v18
	s_wait_dscnt 0xc
	v_add_f32_e32 v195, v195, v19
	s_wait_dscnt 0xb
	v_add_f32_e32 v208, v208, v20
	s_wait_dscnt 0xa
	v_add_f32_e32 v209, v209, v21
	s_wait_dscnt 0x9
	v_add_f32_e32 v210, v210, v22
	s_wait_dscnt 0x8
	v_add_f32_e32 v211, v211, v23
	s_wait_dscnt 0x7
	v_add_f32_e32 v224, v224, v24
	s_wait_dscnt 0x6
	v_add_f32_e32 v225, v225, v25
	s_wait_dscnt 0x5
	v_add_f32_e32 v226, v226, v26
	s_wait_dscnt 0x4
	v_add_f32_e32 v227, v227, v27
	s_wait_dscnt 0x3
	v_add_f32_e32 v240, v240, v28
	s_wait_dscnt 0x2
	v_add_f32_e32 v241, v241, v29
	s_wait_dscnt 0x1
	v_add_f32_e32 v242, v242, v30
	s_wait_dscnt 0x0
	v_add_f32_e32 v243, v243, v31
	v_cmpx_eq_u32_e32 0, v0
	v_mov_b32_e32 v40, s42
	s_mul_i32 s40, s39, 1
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s40, s40, s42
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v41, s40
	s_mul_i32 s40, s39, 2
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s40, s40, s42
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v42, s40
	s_mul_i32 s40, s39, 3
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s40, s40, s42
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v43, s40
	s_mul_i32 s40, s39, 4
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s40, s40, s42
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v44, s40
	s_mul_i32 s40, s39, 5
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s40, s40, s42
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v45, s40
	global_store_b32 v40, v160, s[8:9]
	global_store_b32 v41, v176, s[8:9]
	global_store_b32 v42, v192, s[8:9]
	global_store_b32 v43, v208, s[8:9]
	global_store_b32 v44, v224, s[8:9]
	global_store_b32 v45, v240, s[8:9]
	s_cmp_le_i32 s13, 1
	s_cbranch_scc1 .Lpx_b6_stored
	global_store_b32 v40, v161, s[8:9] offset:4
	global_store_b32 v41, v177, s[8:9] offset:4
	global_store_b32 v42, v193, s[8:9] offset:4
	global_store_b32 v43, v209, s[8:9] offset:4
	global_store_b32 v44, v225, s[8:9] offset:4
	global_store_b32 v45, v241, s[8:9] offset:4
	s_cmp_le_i32 s13, 2
	s_cbranch_scc1 .Lpx_b6_stored
	global_store_b32 v40, v162, s[8:9] offset:8
	global_store_b32 v41, v178, s[8:9] offset:8
	global_store_b32 v42, v194, s[8:9] offset:8
	global_store_b32 v43, v210, s[8:9] offset:8
	global_store_b32 v44, v226, s[8:9] offset:8
	global_store_b32 v45, v242, s[8:9] offset:8
	s_cmp_le_i32 s13, 3
	s_cbranch_scc1 .Lpx_b6_stored
	global_store_b32 v40, v163, s[8:9] offset:12
	global_store_b32 v41, v179, s[8:9] offset:12
	global_store_b32 v42, v195, s[8:9] offset:12
	global_store_b32 v43, v211, s[8:9] offset:12
	global_store_b32 v44, v227, s[8:9] offset:12
	global_store_b32 v45, v243, s[8:9] offset:12
	.Lpx_b6_stored:
	s_wait_storecnt 0x0
	s_branch .Lpx_end
	.Lpx_b7:
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
	v_mov_b32_e32 v160, 0
	v_mov_b32_e32 v161, 0
	v_mov_b32_e32 v162, 0
	v_mov_b32_e32 v163, 0
	v_mov_b32_e32 v164, 0
	v_mov_b32_e32 v165, 0
	v_mov_b32_e32 v166, 0
	v_mov_b32_e32 v167, 0
	v_mov_b32_e32 v168, 0
	v_mov_b32_e32 v169, 0
	v_mov_b32_e32 v170, 0
	v_mov_b32_e32 v171, 0
	v_mov_b32_e32 v172, 0
	v_mov_b32_e32 v173, 0
	v_mov_b32_e32 v174, 0
	v_mov_b32_e32 v175, 0
	v_mov_b32_e32 v176, 0
	v_mov_b32_e32 v177, 0
	v_mov_b32_e32 v178, 0
	v_mov_b32_e32 v179, 0
	v_mov_b32_e32 v180, 0
	v_mov_b32_e32 v181, 0
	v_mov_b32_e32 v182, 0
	v_mov_b32_e32 v183, 0
	v_mov_b32_e32 v184, 0
	v_mov_b32_e32 v185, 0
	v_mov_b32_e32 v186, 0
	v_mov_b32_e32 v187, 0
	v_mov_b32_e32 v188, 0
	v_mov_b32_e32 v189, 0
	v_mov_b32_e32 v190, 0
	v_mov_b32_e32 v191, 0
	v_mov_b32_e32 v192, 0
	v_mov_b32_e32 v193, 0
	v_mov_b32_e32 v194, 0
	v_mov_b32_e32 v195, 0
	v_mov_b32_e32 v196, 0
	v_mov_b32_e32 v197, 0
	v_mov_b32_e32 v198, 0
	v_mov_b32_e32 v199, 0
	v_mov_b32_e32 v200, 0
	v_mov_b32_e32 v201, 0
	v_mov_b32_e32 v202, 0
	v_mov_b32_e32 v203, 0
	v_mov_b32_e32 v204, 0
	v_mov_b32_e32 v205, 0
	v_mov_b32_e32 v206, 0
	v_mov_b32_e32 v207, 0
	v_mov_b32_e32 v208, 0
	v_mov_b32_e32 v209, 0
	v_mov_b32_e32 v210, 0
	v_mov_b32_e32 v211, 0
	v_mov_b32_e32 v212, 0
	v_mov_b32_e32 v213, 0
	v_mov_b32_e32 v214, 0
	v_mov_b32_e32 v215, 0
	v_mov_b32_e32 v216, 0
	v_mov_b32_e32 v217, 0
	v_mov_b32_e32 v218, 0
	v_mov_b32_e32 v219, 0
	v_mov_b32_e32 v220, 0
	v_mov_b32_e32 v221, 0
	v_mov_b32_e32 v222, 0
	v_mov_b32_e32 v223, 0
	v_mov_b32_e32 v224, 0
	v_mov_b32_e32 v225, 0
	v_mov_b32_e32 v226, 0
	v_mov_b32_e32 v227, 0
	v_mov_b32_e32 v228, 0
	v_mov_b32_e32 v229, 0
	v_mov_b32_e32 v230, 0
	v_mov_b32_e32 v231, 0
	v_mov_b32_e32 v232, 0
	v_mov_b32_e32 v233, 0
	v_mov_b32_e32 v234, 0
	v_mov_b32_e32 v235, 0
	v_mov_b32_e32 v236, 0
	v_mov_b32_e32 v237, 0
	v_mov_b32_e32 v238, 0
	v_mov_b32_e32 v239, 0
	v_mov_b32_e32 v240, 0
	v_mov_b32_e32 v241, 0
	v_mov_b32_e32 v242, 0
	v_mov_b32_e32 v243, 0
	v_mov_b32_e32 v244, 0
	v_mov_b32_e32 v245, 0
	v_mov_b32_e32 v246, 0
	v_mov_b32_e32 v247, 0
	v_mov_b32_e32 v248, 0
	v_mov_b32_e32 v249, 0
	v_mov_b32_e32 v250, 0
	v_mov_b32_e32 v251, 0
	v_mov_b32_e32 v252, 0
	v_mov_b32_e32 v253, 0
	v_mov_b32_e32 v254, 0
	v_mov_b32_e32 v255, 0
	s_lshr_b32 s36, s14, 2
	s_and_b32 s37, s14, 3
	s_cmp_eq_u32 s36, 0
	s_cbranch_scc1 .Lpx_b7_tail
	.Lpx_b7_quad:
	buffer_load_b32 v8, v1, s[20:23], s16 offen
	buffer_load_b32 v9, v1, s[20:23], s17 offen
	buffer_load_b32 v10, v1, s[20:23], s18 offen
	buffer_load_b32 v11, v1, s[20:23], s19 offen
	buffer_load_b32 v24, v2, s[20:23], s16 offen offset:8
	buffer_load_b32 v25, v2, s[20:23], s17 offen offset:8
	buffer_load_b32 v26, v2, s[20:23], s18 offen offset:8
	buffer_load_b32 v27, v2, s[20:23], s19 offen offset:8
	buffer_load_b128 v[80:83], v3, s[24:27], s28 offen
	buffer_load_b128 v[84:87], v3, s[24:27], s28 offen offset:16
	buffer_load_b128 v[88:91], v3, s[24:27], s29 offen
	buffer_load_b128 v[92:95], v3, s[24:27], s29 offen offset:16
	buffer_load_b128 v[96:99], v3, s[24:27], s30 offen
	buffer_load_b128 v[100:103], v3, s[24:27], s30 offen offset:16
	buffer_load_b128 v[104:107], v3, s[24:27], s31 offen
	buffer_load_b128 v[108:111], v3, s[24:27], s31 offen offset:16
	buffer_load_b128 v[112:115], v3, s[24:27], s32 offen
	buffer_load_b128 v[116:119], v3, s[24:27], s32 offen offset:16
	buffer_load_b128 v[120:123], v3, s[24:27], s33 offen
	buffer_load_b128 v[124:127], v3, s[24:27], s33 offen offset:16
	buffer_load_b128 v[128:131], v3, s[24:27], s34 offen
	buffer_load_b128 v[132:135], v3, s[24:27], s34 offen offset:16
	buffer_load_b32 v12, v1, s[20:23], s16 offen offset:136
	buffer_load_b32 v13, v1, s[20:23], s17 offen offset:136
	buffer_load_b32 v14, v1, s[20:23], s18 offen offset:136
	buffer_load_b32 v15, v1, s[20:23], s19 offen offset:136
	buffer_load_b32 v28, v2, s[20:23], s16 offen offset:144
	buffer_load_b32 v29, v2, s[20:23], s17 offen offset:144
	buffer_load_b32 v30, v2, s[20:23], s18 offen offset:144
	buffer_load_b32 v31, v2, s[20:23], s19 offen offset:144
	buffer_load_b32 v16, v1, s[20:23], s16 offen offset:272
	buffer_load_b32 v17, v1, s[20:23], s17 offen offset:272
	buffer_load_b32 v18, v1, s[20:23], s18 offen offset:272
	buffer_load_b32 v19, v1, s[20:23], s19 offen offset:272
	buffer_load_b32 v32, v2, s[20:23], s16 offen offset:280
	buffer_load_b32 v33, v2, s[20:23], s17 offen offset:280
	buffer_load_b32 v34, v2, s[20:23], s18 offen offset:280
	buffer_load_b32 v35, v2, s[20:23], s19 offen offset:280
	buffer_load_b32 v20, v1, s[20:23], s16 offen offset:408
	buffer_load_b32 v21, v1, s[20:23], s17 offen offset:408
	buffer_load_b32 v22, v1, s[20:23], s18 offen offset:408
	buffer_load_b32 v23, v1, s[20:23], s19 offen offset:408
	buffer_load_b32 v36, v2, s[20:23], s16 offen offset:416
	buffer_load_b32 v37, v2, s[20:23], s17 offen offset:416
	buffer_load_b32 v38, v2, s[20:23], s18 offen offset:416
	buffer_load_b32 v39, v2, s[20:23], s19 offen offset:416
	s_wait_loadcnt 0x29
	v_and_b32_e32 v46, 0xf0f0f0f, v24
	s_wait_loadcnt 0x28
	v_and_b32_e32 v56, 0xf0f0f0f, v25
	s_wait_loadcnt 0x27
	v_and_b32_e32 v66, 0xf0f0f0f, v26
	s_wait_loadcnt 0x26
	v_and_b32_e32 v76, 0xf0f0f0f, v27
	v_lshrrev_b32_e32 v47, 4, v24
	v_lshrrev_b32_e32 v57, 4, v25
	v_lshrrev_b32_e32 v67, 4, v26
	v_lshrrev_b32_e32 v77, 4, v27
	v_and_b32_e32 v47, 0xf0f0f0f, v47
	v_and_b32_e32 v57, 0xf0f0f0f, v57
	v_and_b32_e32 v67, 0xf0f0f0f, v67
	v_and_b32_e32 v77, 0xf0f0f0f, v77
	v_cvt_f32_ubyte0_e32 v40, v46
	v_cvt_f32_ubyte0_e32 v50, v56
	v_cvt_f32_ubyte0_e32 v60, v66
	v_cvt_f32_ubyte0_e32 v70, v76
	v_cvt_f32_ubyte1_e32 v42, v46
	v_cvt_f32_ubyte1_e32 v52, v56
	v_cvt_f32_ubyte1_e32 v62, v66
	v_cvt_f32_ubyte1_e32 v72, v76
	v_cvt_f32_ubyte2_e32 v44, v46
	v_cvt_f32_ubyte2_e32 v54, v56
	v_cvt_f32_ubyte2_e32 v64, v66
	v_cvt_f32_ubyte2_e32 v74, v76
	v_cvt_f32_ubyte3_e32 v46, v46
	v_cvt_f32_ubyte3_e32 v56, v56
	v_cvt_f32_ubyte3_e32 v66, v66
	v_cvt_f32_ubyte3_e32 v76, v76
	v_cvt_f32_ubyte0_e32 v41, v47
	v_cvt_f32_ubyte0_e32 v51, v57
	v_cvt_f32_ubyte0_e32 v61, v67
	v_cvt_f32_ubyte0_e32 v71, v77
	v_cvt_f32_ubyte1_e32 v43, v47
	v_cvt_f32_ubyte1_e32 v53, v57
	v_cvt_f32_ubyte1_e32 v63, v67
	v_cvt_f32_ubyte1_e32 v73, v77
	v_cvt_f32_ubyte2_e32 v45, v47
	v_cvt_f32_ubyte2_e32 v55, v57
	v_cvt_f32_ubyte2_e32 v65, v67
	v_cvt_f32_ubyte2_e32 v75, v77
	v_cvt_f32_ubyte3_e32 v47, v47
	v_cvt_f32_ubyte3_e32 v57, v57
	v_cvt_f32_ubyte3_e32 v67, v67
	v_cvt_f32_ubyte3_e32 v77, v77
	v_fma_mix_f32 v40, v8, v40, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v9, v50, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v10, v60, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v70, v11, v70, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v41, v8, v41, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v9, v51, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v10, v61, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v71, v11, v71, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v42, v8, v42, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v52, v9, v52, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v62, v10, v62, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v72, v11, v72, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v43, v8, v43, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v53, v9, v53, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v63, v10, v63, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v73, v11, v73, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v44, v8, v44, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v54, v9, v54, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v64, v10, v64, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v74, v11, v74, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v8, v45, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v9, v55, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v65, v10, v65, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v75, v11, v75, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v8, v46, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v9, v56, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v66, v10, v66, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v76, v11, v76, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v8, v47, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v9, v57, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v67, v10, v67, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v77, v11, v77, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	buffer_load_b128 v[136:139], v3, s[24:27], s28 offen offset:1024
	buffer_load_b128 v[140:143], v3, s[24:27], s28 offen offset:1040
	s_wait_loadcnt 0x27
	v_dual_mul_f32 v48, v41, v81 :: v_dual_mul_f32 v49, v51, v81
	v_dual_mul_f32 v58, v61, v81 :: v_dual_mul_f32 v59, v71, v81
	s_wait_loadcnt 0x25
	v_dual_mul_f32 v68, v41, v89 :: v_dual_mul_f32 v69, v51, v89
	v_dual_mul_f32 v78, v61, v89 :: v_dual_mul_f32 v79, v71, v89
	v_dual_fmac_f32 v48, v40, v80 :: v_dual_fmac_f32 v49, v50, v80
	v_dual_fmac_f32 v58, v60, v80 :: v_dual_fmac_f32 v59, v70, v80
	v_dual_fmac_f32 v68, v40, v88 :: v_dual_fmac_f32 v69, v50, v88
	v_dual_fmac_f32 v78, v60, v88 :: v_dual_fmac_f32 v79, v70, v88
	v_dual_fmac_f32 v48, v42, v82 :: v_dual_fmac_f32 v49, v52, v82
	v_dual_fmac_f32 v58, v62, v82 :: v_dual_fmac_f32 v59, v72, v82
	v_dual_fmac_f32 v68, v42, v90 :: v_dual_fmac_f32 v69, v52, v90
	v_dual_fmac_f32 v78, v62, v90 :: v_dual_fmac_f32 v79, v72, v90
	v_dual_fmac_f32 v48, v43, v83 :: v_dual_fmac_f32 v49, v53, v83
	v_dual_fmac_f32 v58, v63, v83 :: v_dual_fmac_f32 v59, v73, v83
	v_dual_fmac_f32 v68, v43, v91 :: v_dual_fmac_f32 v69, v53, v91
	v_dual_fmac_f32 v78, v63, v91 :: v_dual_fmac_f32 v79, v73, v91
	v_dual_fmac_f32 v48, v44, v84 :: v_dual_fmac_f32 v49, v54, v84
	v_dual_fmac_f32 v58, v64, v84 :: v_dual_fmac_f32 v59, v74, v84
	s_wait_loadcnt 0x24
	v_dual_fmac_f32 v68, v44, v92 :: v_dual_fmac_f32 v69, v54, v92
	v_dual_fmac_f32 v78, v64, v92 :: v_dual_fmac_f32 v79, v74, v92
	v_dual_fmac_f32 v48, v45, v85 :: v_dual_fmac_f32 v49, v55, v85
	v_dual_fmac_f32 v58, v65, v85 :: v_dual_fmac_f32 v59, v75, v85
	v_dual_fmac_f32 v68, v45, v93 :: v_dual_fmac_f32 v69, v55, v93
	v_dual_fmac_f32 v78, v65, v93 :: v_dual_fmac_f32 v79, v75, v93
	v_dual_fmac_f32 v48, v46, v86 :: v_dual_fmac_f32 v49, v56, v86
	v_dual_fmac_f32 v58, v66, v86 :: v_dual_fmac_f32 v59, v76, v86
	v_dual_fmac_f32 v68, v46, v94 :: v_dual_fmac_f32 v69, v56, v94
	v_dual_fmac_f32 v78, v66, v94 :: v_dual_fmac_f32 v79, v76, v94
	v_dual_fmac_f32 v48, v47, v87 :: v_dual_fmac_f32 v49, v57, v87
	v_dual_fmac_f32 v58, v67, v87 :: v_dual_fmac_f32 v59, v77, v87
	v_dual_fmac_f32 v68, v47, v95 :: v_dual_fmac_f32 v69, v57, v95
	v_dual_fmac_f32 v78, v67, v95 :: v_dual_fmac_f32 v79, v77, v95
	v_dual_add_f32 v144, v144, v48 :: v_dual_add_f32 v145, v145, v49
	v_dual_add_f32 v146, v146, v58 :: v_dual_add_f32 v147, v147, v59
	v_dual_add_f32 v160, v160, v68 :: v_dual_add_f32 v161, v161, v69
	v_dual_add_f32 v162, v162, v78 :: v_dual_add_f32 v163, v163, v79
	buffer_load_b128 v[80:83], v3, s[24:27], s29 offen offset:1024
	buffer_load_b128 v[84:87], v3, s[24:27], s29 offen offset:1040
	buffer_load_b128 v[88:91], v3, s[24:27], s30 offen offset:1024
	buffer_load_b128 v[92:95], v3, s[24:27], s30 offen offset:1040
	s_wait_loadcnt 0x27
	v_dual_mul_f32 v48, v41, v97 :: v_dual_mul_f32 v49, v51, v97
	v_dual_mul_f32 v58, v61, v97 :: v_dual_mul_f32 v59, v71, v97
	s_wait_loadcnt 0x25
	v_dual_mul_f32 v68, v41, v105 :: v_dual_mul_f32 v69, v51, v105
	v_dual_mul_f32 v78, v61, v105 :: v_dual_mul_f32 v79, v71, v105
	v_dual_fmac_f32 v48, v40, v96 :: v_dual_fmac_f32 v49, v50, v96
	v_dual_fmac_f32 v58, v60, v96 :: v_dual_fmac_f32 v59, v70, v96
	v_dual_fmac_f32 v68, v40, v104 :: v_dual_fmac_f32 v69, v50, v104
	v_dual_fmac_f32 v78, v60, v104 :: v_dual_fmac_f32 v79, v70, v104
	v_dual_fmac_f32 v48, v42, v98 :: v_dual_fmac_f32 v49, v52, v98
	v_dual_fmac_f32 v58, v62, v98 :: v_dual_fmac_f32 v59, v72, v98
	v_dual_fmac_f32 v68, v42, v106 :: v_dual_fmac_f32 v69, v52, v106
	v_dual_fmac_f32 v78, v62, v106 :: v_dual_fmac_f32 v79, v72, v106
	v_dual_fmac_f32 v48, v43, v99 :: v_dual_fmac_f32 v49, v53, v99
	v_dual_fmac_f32 v58, v63, v99 :: v_dual_fmac_f32 v59, v73, v99
	v_dual_fmac_f32 v68, v43, v107 :: v_dual_fmac_f32 v69, v53, v107
	v_dual_fmac_f32 v78, v63, v107 :: v_dual_fmac_f32 v79, v73, v107
	v_dual_fmac_f32 v48, v44, v100 :: v_dual_fmac_f32 v49, v54, v100
	v_dual_fmac_f32 v58, v64, v100 :: v_dual_fmac_f32 v59, v74, v100
	s_wait_loadcnt 0x24
	v_dual_fmac_f32 v68, v44, v108 :: v_dual_fmac_f32 v69, v54, v108
	v_dual_fmac_f32 v78, v64, v108 :: v_dual_fmac_f32 v79, v74, v108
	v_dual_fmac_f32 v48, v45, v101 :: v_dual_fmac_f32 v49, v55, v101
	v_dual_fmac_f32 v58, v65, v101 :: v_dual_fmac_f32 v59, v75, v101
	v_dual_fmac_f32 v68, v45, v109 :: v_dual_fmac_f32 v69, v55, v109
	v_dual_fmac_f32 v78, v65, v109 :: v_dual_fmac_f32 v79, v75, v109
	v_dual_fmac_f32 v48, v46, v102 :: v_dual_fmac_f32 v49, v56, v102
	v_dual_fmac_f32 v58, v66, v102 :: v_dual_fmac_f32 v59, v76, v102
	v_dual_fmac_f32 v68, v46, v110 :: v_dual_fmac_f32 v69, v56, v110
	v_dual_fmac_f32 v78, v66, v110 :: v_dual_fmac_f32 v79, v76, v110
	v_dual_fmac_f32 v48, v47, v103 :: v_dual_fmac_f32 v49, v57, v103
	v_dual_fmac_f32 v58, v67, v103 :: v_dual_fmac_f32 v59, v77, v103
	v_dual_fmac_f32 v68, v47, v111 :: v_dual_fmac_f32 v69, v57, v111
	v_dual_fmac_f32 v78, v67, v111 :: v_dual_fmac_f32 v79, v77, v111
	v_dual_add_f32 v176, v176, v48 :: v_dual_add_f32 v177, v177, v49
	v_dual_add_f32 v178, v178, v58 :: v_dual_add_f32 v179, v179, v59
	v_dual_add_f32 v192, v192, v68 :: v_dual_add_f32 v193, v193, v69
	v_dual_add_f32 v194, v194, v78 :: v_dual_add_f32 v195, v195, v79
	buffer_load_b128 v[96:99], v3, s[24:27], s31 offen offset:1024
	buffer_load_b128 v[100:103], v3, s[24:27], s31 offen offset:1040
	buffer_load_b128 v[104:107], v3, s[24:27], s32 offen offset:1024
	buffer_load_b128 v[108:111], v3, s[24:27], s32 offen offset:1040
	s_wait_loadcnt 0x27
	v_dual_mul_f32 v48, v41, v113 :: v_dual_mul_f32 v49, v51, v113
	v_dual_mul_f32 v58, v61, v113 :: v_dual_mul_f32 v59, v71, v113
	s_wait_loadcnt 0x25
	v_dual_mul_f32 v68, v41, v121 :: v_dual_mul_f32 v69, v51, v121
	v_dual_mul_f32 v78, v61, v121 :: v_dual_mul_f32 v79, v71, v121
	v_dual_fmac_f32 v48, v40, v112 :: v_dual_fmac_f32 v49, v50, v112
	v_dual_fmac_f32 v58, v60, v112 :: v_dual_fmac_f32 v59, v70, v112
	v_dual_fmac_f32 v68, v40, v120 :: v_dual_fmac_f32 v69, v50, v120
	v_dual_fmac_f32 v78, v60, v120 :: v_dual_fmac_f32 v79, v70, v120
	v_dual_fmac_f32 v48, v42, v114 :: v_dual_fmac_f32 v49, v52, v114
	v_dual_fmac_f32 v58, v62, v114 :: v_dual_fmac_f32 v59, v72, v114
	v_dual_fmac_f32 v68, v42, v122 :: v_dual_fmac_f32 v69, v52, v122
	v_dual_fmac_f32 v78, v62, v122 :: v_dual_fmac_f32 v79, v72, v122
	v_dual_fmac_f32 v48, v43, v115 :: v_dual_fmac_f32 v49, v53, v115
	v_dual_fmac_f32 v58, v63, v115 :: v_dual_fmac_f32 v59, v73, v115
	v_dual_fmac_f32 v68, v43, v123 :: v_dual_fmac_f32 v69, v53, v123
	v_dual_fmac_f32 v78, v63, v123 :: v_dual_fmac_f32 v79, v73, v123
	v_dual_fmac_f32 v48, v44, v116 :: v_dual_fmac_f32 v49, v54, v116
	v_dual_fmac_f32 v58, v64, v116 :: v_dual_fmac_f32 v59, v74, v116
	s_wait_loadcnt 0x24
	v_dual_fmac_f32 v68, v44, v124 :: v_dual_fmac_f32 v69, v54, v124
	v_dual_fmac_f32 v78, v64, v124 :: v_dual_fmac_f32 v79, v74, v124
	v_dual_fmac_f32 v48, v45, v117 :: v_dual_fmac_f32 v49, v55, v117
	v_dual_fmac_f32 v58, v65, v117 :: v_dual_fmac_f32 v59, v75, v117
	v_dual_fmac_f32 v68, v45, v125 :: v_dual_fmac_f32 v69, v55, v125
	v_dual_fmac_f32 v78, v65, v125 :: v_dual_fmac_f32 v79, v75, v125
	v_dual_fmac_f32 v48, v46, v118 :: v_dual_fmac_f32 v49, v56, v118
	v_dual_fmac_f32 v58, v66, v118 :: v_dual_fmac_f32 v59, v76, v118
	v_dual_fmac_f32 v68, v46, v126 :: v_dual_fmac_f32 v69, v56, v126
	v_dual_fmac_f32 v78, v66, v126 :: v_dual_fmac_f32 v79, v76, v126
	v_dual_fmac_f32 v48, v47, v119 :: v_dual_fmac_f32 v49, v57, v119
	v_dual_fmac_f32 v58, v67, v119 :: v_dual_fmac_f32 v59, v77, v119
	v_dual_fmac_f32 v68, v47, v127 :: v_dual_fmac_f32 v69, v57, v127
	v_dual_fmac_f32 v78, v67, v127 :: v_dual_fmac_f32 v79, v77, v127
	v_dual_add_f32 v208, v208, v48 :: v_dual_add_f32 v209, v209, v49
	v_dual_add_f32 v210, v210, v58 :: v_dual_add_f32 v211, v211, v59
	v_dual_add_f32 v224, v224, v68 :: v_dual_add_f32 v225, v225, v69
	v_dual_add_f32 v226, v226, v78 :: v_dual_add_f32 v227, v227, v79
	buffer_load_b128 v[112:115], v3, s[24:27], s33 offen offset:1024
	buffer_load_b128 v[116:119], v3, s[24:27], s33 offen offset:1040
	buffer_load_b128 v[120:123], v3, s[24:27], s34 offen offset:1024
	buffer_load_b128 v[124:127], v3, s[24:27], s34 offen offset:1040
	s_wait_loadcnt 0x27
	v_dual_mul_f32 v48, v41, v129 :: v_dual_mul_f32 v49, v51, v129
	v_dual_mul_f32 v58, v61, v129 :: v_dual_mul_f32 v59, v71, v129
	v_dual_fmac_f32 v48, v40, v128 :: v_dual_fmac_f32 v49, v50, v128
	v_dual_fmac_f32 v58, v60, v128 :: v_dual_fmac_f32 v59, v70, v128
	v_dual_fmac_f32 v48, v42, v130 :: v_dual_fmac_f32 v49, v52, v130
	v_dual_fmac_f32 v58, v62, v130 :: v_dual_fmac_f32 v59, v72, v130
	v_dual_fmac_f32 v48, v43, v131 :: v_dual_fmac_f32 v49, v53, v131
	v_dual_fmac_f32 v58, v63, v131 :: v_dual_fmac_f32 v59, v73, v131
	s_wait_loadcnt 0x26
	v_dual_fmac_f32 v48, v44, v132 :: v_dual_fmac_f32 v49, v54, v132
	v_dual_fmac_f32 v58, v64, v132 :: v_dual_fmac_f32 v59, v74, v132
	v_dual_fmac_f32 v48, v45, v133 :: v_dual_fmac_f32 v49, v55, v133
	v_dual_fmac_f32 v58, v65, v133 :: v_dual_fmac_f32 v59, v75, v133
	v_dual_fmac_f32 v48, v46, v134 :: v_dual_fmac_f32 v49, v56, v134
	v_dual_fmac_f32 v58, v66, v134 :: v_dual_fmac_f32 v59, v76, v134
	v_dual_fmac_f32 v48, v47, v135 :: v_dual_fmac_f32 v49, v57, v135
	v_dual_fmac_f32 v58, v67, v135 :: v_dual_fmac_f32 v59, v77, v135
	v_dual_add_f32 v240, v240, v48 :: v_dual_add_f32 v241, v241, v49
	v_dual_add_f32 v242, v242, v58 :: v_dual_add_f32 v243, v243, v59
	s_wait_loadcnt 0x21
	v_and_b32_e32 v46, 0xf0f0f0f, v28
	s_wait_loadcnt 0x20
	v_and_b32_e32 v56, 0xf0f0f0f, v29
	s_wait_loadcnt 0x1f
	v_and_b32_e32 v66, 0xf0f0f0f, v30
	s_wait_loadcnt 0x1e
	v_and_b32_e32 v76, 0xf0f0f0f, v31
	v_lshrrev_b32_e32 v47, 4, v28
	v_lshrrev_b32_e32 v57, 4, v29
	v_lshrrev_b32_e32 v67, 4, v30
	v_lshrrev_b32_e32 v77, 4, v31
	v_and_b32_e32 v47, 0xf0f0f0f, v47
	v_and_b32_e32 v57, 0xf0f0f0f, v57
	v_and_b32_e32 v67, 0xf0f0f0f, v67
	v_and_b32_e32 v77, 0xf0f0f0f, v77
	v_cvt_f32_ubyte0_e32 v40, v46
	v_cvt_f32_ubyte0_e32 v50, v56
	v_cvt_f32_ubyte0_e32 v60, v66
	v_cvt_f32_ubyte0_e32 v70, v76
	v_cvt_f32_ubyte1_e32 v42, v46
	v_cvt_f32_ubyte1_e32 v52, v56
	v_cvt_f32_ubyte1_e32 v62, v66
	v_cvt_f32_ubyte1_e32 v72, v76
	v_cvt_f32_ubyte2_e32 v44, v46
	v_cvt_f32_ubyte2_e32 v54, v56
	v_cvt_f32_ubyte2_e32 v64, v66
	v_cvt_f32_ubyte2_e32 v74, v76
	v_cvt_f32_ubyte3_e32 v46, v46
	v_cvt_f32_ubyte3_e32 v56, v56
	v_cvt_f32_ubyte3_e32 v66, v66
	v_cvt_f32_ubyte3_e32 v76, v76
	v_cvt_f32_ubyte0_e32 v41, v47
	v_cvt_f32_ubyte0_e32 v51, v57
	v_cvt_f32_ubyte0_e32 v61, v67
	v_cvt_f32_ubyte0_e32 v71, v77
	v_cvt_f32_ubyte1_e32 v43, v47
	v_cvt_f32_ubyte1_e32 v53, v57
	v_cvt_f32_ubyte1_e32 v63, v67
	v_cvt_f32_ubyte1_e32 v73, v77
	v_cvt_f32_ubyte2_e32 v45, v47
	v_cvt_f32_ubyte2_e32 v55, v57
	v_cvt_f32_ubyte2_e32 v65, v67
	v_cvt_f32_ubyte2_e32 v75, v77
	v_cvt_f32_ubyte3_e32 v47, v47
	v_cvt_f32_ubyte3_e32 v57, v57
	v_cvt_f32_ubyte3_e32 v67, v67
	v_cvt_f32_ubyte3_e32 v77, v77
	v_fma_mix_f32 v40, v12, v40, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v13, v50, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v14, v60, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v70, v15, v70, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v41, v12, v41, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v13, v51, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v14, v61, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v71, v15, v71, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v42, v12, v42, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v52, v13, v52, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v62, v14, v62, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v72, v15, v72, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v43, v12, v43, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v53, v13, v53, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v63, v14, v63, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v73, v15, v73, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v44, v12, v44, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v54, v13, v54, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v64, v14, v64, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v74, v15, v74, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v12, v45, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v13, v55, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v65, v14, v65, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v75, v15, v75, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v12, v46, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v13, v56, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v66, v14, v66, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v76, v15, v76, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v12, v47, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v13, v57, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v67, v14, v67, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v77, v15, v77, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	buffer_load_b128 v[128:131], v3, s[24:27], s28 offen offset:2048
	buffer_load_b128 v[132:135], v3, s[24:27], s28 offen offset:2064
	s_wait_loadcnt 0xf
	v_dual_mul_f32 v48, v41, v137 :: v_dual_mul_f32 v49, v51, v137
	v_dual_mul_f32 v58, v61, v137 :: v_dual_mul_f32 v59, v71, v137
	s_wait_loadcnt 0xd
	v_dual_mul_f32 v68, v41, v81 :: v_dual_mul_f32 v69, v51, v81
	v_dual_mul_f32 v78, v61, v81 :: v_dual_mul_f32 v79, v71, v81
	v_dual_fmac_f32 v48, v40, v136 :: v_dual_fmac_f32 v49, v50, v136
	v_dual_fmac_f32 v58, v60, v136 :: v_dual_fmac_f32 v59, v70, v136
	v_dual_fmac_f32 v68, v40, v80 :: v_dual_fmac_f32 v69, v50, v80
	v_dual_fmac_f32 v78, v60, v80 :: v_dual_fmac_f32 v79, v70, v80
	v_dual_fmac_f32 v48, v42, v138 :: v_dual_fmac_f32 v49, v52, v138
	v_dual_fmac_f32 v58, v62, v138 :: v_dual_fmac_f32 v59, v72, v138
	v_dual_fmac_f32 v68, v42, v82 :: v_dual_fmac_f32 v69, v52, v82
	v_dual_fmac_f32 v78, v62, v82 :: v_dual_fmac_f32 v79, v72, v82
	v_dual_fmac_f32 v48, v43, v139 :: v_dual_fmac_f32 v49, v53, v139
	v_dual_fmac_f32 v58, v63, v139 :: v_dual_fmac_f32 v59, v73, v139
	v_dual_fmac_f32 v68, v43, v83 :: v_dual_fmac_f32 v69, v53, v83
	v_dual_fmac_f32 v78, v63, v83 :: v_dual_fmac_f32 v79, v73, v83
	v_dual_fmac_f32 v48, v44, v140 :: v_dual_fmac_f32 v49, v54, v140
	v_dual_fmac_f32 v58, v64, v140 :: v_dual_fmac_f32 v59, v74, v140
	s_wait_loadcnt 0xc
	v_dual_fmac_f32 v68, v44, v84 :: v_dual_fmac_f32 v69, v54, v84
	v_dual_fmac_f32 v78, v64, v84 :: v_dual_fmac_f32 v79, v74, v84
	v_dual_fmac_f32 v48, v45, v141 :: v_dual_fmac_f32 v49, v55, v141
	v_dual_fmac_f32 v58, v65, v141 :: v_dual_fmac_f32 v59, v75, v141
	v_dual_fmac_f32 v68, v45, v85 :: v_dual_fmac_f32 v69, v55, v85
	v_dual_fmac_f32 v78, v65, v85 :: v_dual_fmac_f32 v79, v75, v85
	v_dual_fmac_f32 v48, v46, v142 :: v_dual_fmac_f32 v49, v56, v142
	v_dual_fmac_f32 v58, v66, v142 :: v_dual_fmac_f32 v59, v76, v142
	v_dual_fmac_f32 v68, v46, v86 :: v_dual_fmac_f32 v69, v56, v86
	v_dual_fmac_f32 v78, v66, v86 :: v_dual_fmac_f32 v79, v76, v86
	v_dual_fmac_f32 v48, v47, v143 :: v_dual_fmac_f32 v49, v57, v143
	v_dual_fmac_f32 v58, v67, v143 :: v_dual_fmac_f32 v59, v77, v143
	v_dual_fmac_f32 v68, v47, v87 :: v_dual_fmac_f32 v69, v57, v87
	v_dual_fmac_f32 v78, v67, v87 :: v_dual_fmac_f32 v79, v77, v87
	v_dual_add_f32 v148, v148, v48 :: v_dual_add_f32 v149, v149, v49
	v_dual_add_f32 v150, v150, v58 :: v_dual_add_f32 v151, v151, v59
	v_dual_add_f32 v164, v164, v68 :: v_dual_add_f32 v165, v165, v69
	v_dual_add_f32 v166, v166, v78 :: v_dual_add_f32 v167, v167, v79
	buffer_load_b128 v[136:139], v3, s[24:27], s29 offen offset:2048
	buffer_load_b128 v[140:143], v3, s[24:27], s29 offen offset:2064
	buffer_load_b128 v[80:83], v3, s[24:27], s30 offen offset:2048
	buffer_load_b128 v[84:87], v3, s[24:27], s30 offen offset:2064
	s_wait_loadcnt 0xf
	v_dual_mul_f32 v48, v41, v89 :: v_dual_mul_f32 v49, v51, v89
	v_dual_mul_f32 v58, v61, v89 :: v_dual_mul_f32 v59, v71, v89
	s_wait_loadcnt 0xd
	v_dual_mul_f32 v68, v41, v97 :: v_dual_mul_f32 v69, v51, v97
	v_dual_mul_f32 v78, v61, v97 :: v_dual_mul_f32 v79, v71, v97
	v_dual_fmac_f32 v48, v40, v88 :: v_dual_fmac_f32 v49, v50, v88
	v_dual_fmac_f32 v58, v60, v88 :: v_dual_fmac_f32 v59, v70, v88
	v_dual_fmac_f32 v68, v40, v96 :: v_dual_fmac_f32 v69, v50, v96
	v_dual_fmac_f32 v78, v60, v96 :: v_dual_fmac_f32 v79, v70, v96
	v_dual_fmac_f32 v48, v42, v90 :: v_dual_fmac_f32 v49, v52, v90
	v_dual_fmac_f32 v58, v62, v90 :: v_dual_fmac_f32 v59, v72, v90
	v_dual_fmac_f32 v68, v42, v98 :: v_dual_fmac_f32 v69, v52, v98
	v_dual_fmac_f32 v78, v62, v98 :: v_dual_fmac_f32 v79, v72, v98
	v_dual_fmac_f32 v48, v43, v91 :: v_dual_fmac_f32 v49, v53, v91
	v_dual_fmac_f32 v58, v63, v91 :: v_dual_fmac_f32 v59, v73, v91
	v_dual_fmac_f32 v68, v43, v99 :: v_dual_fmac_f32 v69, v53, v99
	v_dual_fmac_f32 v78, v63, v99 :: v_dual_fmac_f32 v79, v73, v99
	v_dual_fmac_f32 v48, v44, v92 :: v_dual_fmac_f32 v49, v54, v92
	v_dual_fmac_f32 v58, v64, v92 :: v_dual_fmac_f32 v59, v74, v92
	s_wait_loadcnt 0xc
	v_dual_fmac_f32 v68, v44, v100 :: v_dual_fmac_f32 v69, v54, v100
	v_dual_fmac_f32 v78, v64, v100 :: v_dual_fmac_f32 v79, v74, v100
	v_dual_fmac_f32 v48, v45, v93 :: v_dual_fmac_f32 v49, v55, v93
	v_dual_fmac_f32 v58, v65, v93 :: v_dual_fmac_f32 v59, v75, v93
	v_dual_fmac_f32 v68, v45, v101 :: v_dual_fmac_f32 v69, v55, v101
	v_dual_fmac_f32 v78, v65, v101 :: v_dual_fmac_f32 v79, v75, v101
	v_dual_fmac_f32 v48, v46, v94 :: v_dual_fmac_f32 v49, v56, v94
	v_dual_fmac_f32 v58, v66, v94 :: v_dual_fmac_f32 v59, v76, v94
	v_dual_fmac_f32 v68, v46, v102 :: v_dual_fmac_f32 v69, v56, v102
	v_dual_fmac_f32 v78, v66, v102 :: v_dual_fmac_f32 v79, v76, v102
	v_dual_fmac_f32 v48, v47, v95 :: v_dual_fmac_f32 v49, v57, v95
	v_dual_fmac_f32 v58, v67, v95 :: v_dual_fmac_f32 v59, v77, v95
	v_dual_fmac_f32 v68, v47, v103 :: v_dual_fmac_f32 v69, v57, v103
	v_dual_fmac_f32 v78, v67, v103 :: v_dual_fmac_f32 v79, v77, v103
	v_dual_add_f32 v180, v180, v48 :: v_dual_add_f32 v181, v181, v49
	v_dual_add_f32 v182, v182, v58 :: v_dual_add_f32 v183, v183, v59
	v_dual_add_f32 v196, v196, v68 :: v_dual_add_f32 v197, v197, v69
	v_dual_add_f32 v198, v198, v78 :: v_dual_add_f32 v199, v199, v79
	buffer_load_b128 v[88:91], v3, s[24:27], s31 offen offset:2048
	buffer_load_b128 v[92:95], v3, s[24:27], s31 offen offset:2064
	buffer_load_b128 v[96:99], v3, s[24:27], s32 offen offset:2048
	buffer_load_b128 v[100:103], v3, s[24:27], s32 offen offset:2064
	s_wait_loadcnt 0xf
	v_dual_mul_f32 v48, v41, v105 :: v_dual_mul_f32 v49, v51, v105
	v_dual_mul_f32 v58, v61, v105 :: v_dual_mul_f32 v59, v71, v105
	s_wait_loadcnt 0xd
	v_dual_mul_f32 v68, v41, v113 :: v_dual_mul_f32 v69, v51, v113
	v_dual_mul_f32 v78, v61, v113 :: v_dual_mul_f32 v79, v71, v113
	v_dual_fmac_f32 v48, v40, v104 :: v_dual_fmac_f32 v49, v50, v104
	v_dual_fmac_f32 v58, v60, v104 :: v_dual_fmac_f32 v59, v70, v104
	v_dual_fmac_f32 v68, v40, v112 :: v_dual_fmac_f32 v69, v50, v112
	v_dual_fmac_f32 v78, v60, v112 :: v_dual_fmac_f32 v79, v70, v112
	v_dual_fmac_f32 v48, v42, v106 :: v_dual_fmac_f32 v49, v52, v106
	v_dual_fmac_f32 v58, v62, v106 :: v_dual_fmac_f32 v59, v72, v106
	v_dual_fmac_f32 v68, v42, v114 :: v_dual_fmac_f32 v69, v52, v114
	v_dual_fmac_f32 v78, v62, v114 :: v_dual_fmac_f32 v79, v72, v114
	v_dual_fmac_f32 v48, v43, v107 :: v_dual_fmac_f32 v49, v53, v107
	v_dual_fmac_f32 v58, v63, v107 :: v_dual_fmac_f32 v59, v73, v107
	v_dual_fmac_f32 v68, v43, v115 :: v_dual_fmac_f32 v69, v53, v115
	v_dual_fmac_f32 v78, v63, v115 :: v_dual_fmac_f32 v79, v73, v115
	v_dual_fmac_f32 v48, v44, v108 :: v_dual_fmac_f32 v49, v54, v108
	v_dual_fmac_f32 v58, v64, v108 :: v_dual_fmac_f32 v59, v74, v108
	s_wait_loadcnt 0xc
	v_dual_fmac_f32 v68, v44, v116 :: v_dual_fmac_f32 v69, v54, v116
	v_dual_fmac_f32 v78, v64, v116 :: v_dual_fmac_f32 v79, v74, v116
	v_dual_fmac_f32 v48, v45, v109 :: v_dual_fmac_f32 v49, v55, v109
	v_dual_fmac_f32 v58, v65, v109 :: v_dual_fmac_f32 v59, v75, v109
	v_dual_fmac_f32 v68, v45, v117 :: v_dual_fmac_f32 v69, v55, v117
	v_dual_fmac_f32 v78, v65, v117 :: v_dual_fmac_f32 v79, v75, v117
	v_dual_fmac_f32 v48, v46, v110 :: v_dual_fmac_f32 v49, v56, v110
	v_dual_fmac_f32 v58, v66, v110 :: v_dual_fmac_f32 v59, v76, v110
	v_dual_fmac_f32 v68, v46, v118 :: v_dual_fmac_f32 v69, v56, v118
	v_dual_fmac_f32 v78, v66, v118 :: v_dual_fmac_f32 v79, v76, v118
	v_dual_fmac_f32 v48, v47, v111 :: v_dual_fmac_f32 v49, v57, v111
	v_dual_fmac_f32 v58, v67, v111 :: v_dual_fmac_f32 v59, v77, v111
	v_dual_fmac_f32 v68, v47, v119 :: v_dual_fmac_f32 v69, v57, v119
	v_dual_fmac_f32 v78, v67, v119 :: v_dual_fmac_f32 v79, v77, v119
	v_dual_add_f32 v212, v212, v48 :: v_dual_add_f32 v213, v213, v49
	v_dual_add_f32 v214, v214, v58 :: v_dual_add_f32 v215, v215, v59
	v_dual_add_f32 v228, v228, v68 :: v_dual_add_f32 v229, v229, v69
	v_dual_add_f32 v230, v230, v78 :: v_dual_add_f32 v231, v231, v79
	buffer_load_b128 v[104:107], v3, s[24:27], s33 offen offset:2048
	buffer_load_b128 v[108:111], v3, s[24:27], s33 offen offset:2064
	buffer_load_b128 v[112:115], v3, s[24:27], s34 offen offset:2048
	buffer_load_b128 v[116:119], v3, s[24:27], s34 offen offset:2064
	s_wait_loadcnt 0xf
	v_dual_mul_f32 v48, v41, v121 :: v_dual_mul_f32 v49, v51, v121
	v_dual_mul_f32 v58, v61, v121 :: v_dual_mul_f32 v59, v71, v121
	v_dual_fmac_f32 v48, v40, v120 :: v_dual_fmac_f32 v49, v50, v120
	v_dual_fmac_f32 v58, v60, v120 :: v_dual_fmac_f32 v59, v70, v120
	v_dual_fmac_f32 v48, v42, v122 :: v_dual_fmac_f32 v49, v52, v122
	v_dual_fmac_f32 v58, v62, v122 :: v_dual_fmac_f32 v59, v72, v122
	v_dual_fmac_f32 v48, v43, v123 :: v_dual_fmac_f32 v49, v53, v123
	v_dual_fmac_f32 v58, v63, v123 :: v_dual_fmac_f32 v59, v73, v123
	s_wait_loadcnt 0xe
	v_dual_fmac_f32 v48, v44, v124 :: v_dual_fmac_f32 v49, v54, v124
	v_dual_fmac_f32 v58, v64, v124 :: v_dual_fmac_f32 v59, v74, v124
	v_dual_fmac_f32 v48, v45, v125 :: v_dual_fmac_f32 v49, v55, v125
	v_dual_fmac_f32 v58, v65, v125 :: v_dual_fmac_f32 v59, v75, v125
	v_dual_fmac_f32 v48, v46, v126 :: v_dual_fmac_f32 v49, v56, v126
	v_dual_fmac_f32 v58, v66, v126 :: v_dual_fmac_f32 v59, v76, v126
	v_dual_fmac_f32 v48, v47, v127 :: v_dual_fmac_f32 v49, v57, v127
	v_dual_fmac_f32 v58, v67, v127 :: v_dual_fmac_f32 v59, v77, v127
	v_dual_add_f32 v244, v244, v48 :: v_dual_add_f32 v245, v245, v49
	v_dual_add_f32 v246, v246, v58 :: v_dual_add_f32 v247, v247, v59
	v_and_b32_e32 v46, 0xf0f0f0f, v32
	v_and_b32_e32 v56, 0xf0f0f0f, v33
	v_and_b32_e32 v66, 0xf0f0f0f, v34
	v_and_b32_e32 v76, 0xf0f0f0f, v35
	v_lshrrev_b32_e32 v47, 4, v32
	v_lshrrev_b32_e32 v57, 4, v33
	v_lshrrev_b32_e32 v67, 4, v34
	v_lshrrev_b32_e32 v77, 4, v35
	v_and_b32_e32 v47, 0xf0f0f0f, v47
	v_and_b32_e32 v57, 0xf0f0f0f, v57
	v_and_b32_e32 v67, 0xf0f0f0f, v67
	v_and_b32_e32 v77, 0xf0f0f0f, v77
	v_cvt_f32_ubyte0_e32 v40, v46
	v_cvt_f32_ubyte0_e32 v50, v56
	v_cvt_f32_ubyte0_e32 v60, v66
	v_cvt_f32_ubyte0_e32 v70, v76
	v_cvt_f32_ubyte1_e32 v42, v46
	v_cvt_f32_ubyte1_e32 v52, v56
	v_cvt_f32_ubyte1_e32 v62, v66
	v_cvt_f32_ubyte1_e32 v72, v76
	v_cvt_f32_ubyte2_e32 v44, v46
	v_cvt_f32_ubyte2_e32 v54, v56
	v_cvt_f32_ubyte2_e32 v64, v66
	v_cvt_f32_ubyte2_e32 v74, v76
	v_cvt_f32_ubyte3_e32 v46, v46
	v_cvt_f32_ubyte3_e32 v56, v56
	v_cvt_f32_ubyte3_e32 v66, v66
	v_cvt_f32_ubyte3_e32 v76, v76
	v_cvt_f32_ubyte0_e32 v41, v47
	v_cvt_f32_ubyte0_e32 v51, v57
	v_cvt_f32_ubyte0_e32 v61, v67
	v_cvt_f32_ubyte0_e32 v71, v77
	v_cvt_f32_ubyte1_e32 v43, v47
	v_cvt_f32_ubyte1_e32 v53, v57
	v_cvt_f32_ubyte1_e32 v63, v67
	v_cvt_f32_ubyte1_e32 v73, v77
	v_cvt_f32_ubyte2_e32 v45, v47
	v_cvt_f32_ubyte2_e32 v55, v57
	v_cvt_f32_ubyte2_e32 v65, v67
	v_cvt_f32_ubyte2_e32 v75, v77
	v_cvt_f32_ubyte3_e32 v47, v47
	v_cvt_f32_ubyte3_e32 v57, v57
	v_cvt_f32_ubyte3_e32 v67, v67
	v_cvt_f32_ubyte3_e32 v77, v77
	v_fma_mix_f32 v40, v16, v40, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v17, v50, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v18, v60, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v70, v19, v70, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v41, v16, v41, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v17, v51, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v18, v61, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v71, v19, v71, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v42, v16, v42, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v52, v17, v52, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v62, v18, v62, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v72, v19, v72, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v43, v16, v43, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v53, v17, v53, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v63, v18, v63, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v73, v19, v73, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v44, v16, v44, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v54, v17, v54, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v64, v18, v64, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v74, v19, v74, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v16, v45, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v17, v55, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v65, v18, v65, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v75, v19, v75, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v16, v46, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v17, v56, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v66, v18, v66, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v76, v19, v76, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v16, v47, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v17, v57, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v67, v18, v67, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v77, v19, v77, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	buffer_load_b128 v[120:123], v3, s[24:27], s28 offen offset:3072
	buffer_load_b128 v[124:127], v3, s[24:27], s28 offen offset:3088
	s_wait_loadcnt 0xf
	v_dual_mul_f32 v48, v41, v129 :: v_dual_mul_f32 v49, v51, v129
	v_dual_mul_f32 v58, v61, v129 :: v_dual_mul_f32 v59, v71, v129
	s_wait_loadcnt 0xd
	v_dual_mul_f32 v68, v41, v137 :: v_dual_mul_f32 v69, v51, v137
	v_dual_mul_f32 v78, v61, v137 :: v_dual_mul_f32 v79, v71, v137
	v_dual_fmac_f32 v48, v40, v128 :: v_dual_fmac_f32 v49, v50, v128
	v_dual_fmac_f32 v58, v60, v128 :: v_dual_fmac_f32 v59, v70, v128
	v_dual_fmac_f32 v68, v40, v136 :: v_dual_fmac_f32 v69, v50, v136
	v_dual_fmac_f32 v78, v60, v136 :: v_dual_fmac_f32 v79, v70, v136
	v_dual_fmac_f32 v48, v42, v130 :: v_dual_fmac_f32 v49, v52, v130
	v_dual_fmac_f32 v58, v62, v130 :: v_dual_fmac_f32 v59, v72, v130
	v_dual_fmac_f32 v68, v42, v138 :: v_dual_fmac_f32 v69, v52, v138
	v_dual_fmac_f32 v78, v62, v138 :: v_dual_fmac_f32 v79, v72, v138
	v_dual_fmac_f32 v48, v43, v131 :: v_dual_fmac_f32 v49, v53, v131
	v_dual_fmac_f32 v58, v63, v131 :: v_dual_fmac_f32 v59, v73, v131
	v_dual_fmac_f32 v68, v43, v139 :: v_dual_fmac_f32 v69, v53, v139
	v_dual_fmac_f32 v78, v63, v139 :: v_dual_fmac_f32 v79, v73, v139
	v_dual_fmac_f32 v48, v44, v132 :: v_dual_fmac_f32 v49, v54, v132
	v_dual_fmac_f32 v58, v64, v132 :: v_dual_fmac_f32 v59, v74, v132
	s_wait_loadcnt 0xc
	v_dual_fmac_f32 v68, v44, v140 :: v_dual_fmac_f32 v69, v54, v140
	v_dual_fmac_f32 v78, v64, v140 :: v_dual_fmac_f32 v79, v74, v140
	v_dual_fmac_f32 v48, v45, v133 :: v_dual_fmac_f32 v49, v55, v133
	v_dual_fmac_f32 v58, v65, v133 :: v_dual_fmac_f32 v59, v75, v133
	v_dual_fmac_f32 v68, v45, v141 :: v_dual_fmac_f32 v69, v55, v141
	v_dual_fmac_f32 v78, v65, v141 :: v_dual_fmac_f32 v79, v75, v141
	v_dual_fmac_f32 v48, v46, v134 :: v_dual_fmac_f32 v49, v56, v134
	v_dual_fmac_f32 v58, v66, v134 :: v_dual_fmac_f32 v59, v76, v134
	v_dual_fmac_f32 v68, v46, v142 :: v_dual_fmac_f32 v69, v56, v142
	v_dual_fmac_f32 v78, v66, v142 :: v_dual_fmac_f32 v79, v76, v142
	v_dual_fmac_f32 v48, v47, v135 :: v_dual_fmac_f32 v49, v57, v135
	v_dual_fmac_f32 v58, v67, v135 :: v_dual_fmac_f32 v59, v77, v135
	v_dual_fmac_f32 v68, v47, v143 :: v_dual_fmac_f32 v69, v57, v143
	v_dual_fmac_f32 v78, v67, v143 :: v_dual_fmac_f32 v79, v77, v143
	v_dual_add_f32 v152, v152, v48 :: v_dual_add_f32 v153, v153, v49
	v_dual_add_f32 v154, v154, v58 :: v_dual_add_f32 v155, v155, v59
	v_dual_add_f32 v168, v168, v68 :: v_dual_add_f32 v169, v169, v69
	v_dual_add_f32 v170, v170, v78 :: v_dual_add_f32 v171, v171, v79
	buffer_load_b128 v[128:131], v3, s[24:27], s29 offen offset:3072
	buffer_load_b128 v[132:135], v3, s[24:27], s29 offen offset:3088
	buffer_load_b128 v[136:139], v3, s[24:27], s30 offen offset:3072
	buffer_load_b128 v[140:143], v3, s[24:27], s30 offen offset:3088
	s_wait_loadcnt 0xf
	v_dual_mul_f32 v48, v41, v81 :: v_dual_mul_f32 v49, v51, v81
	v_dual_mul_f32 v58, v61, v81 :: v_dual_mul_f32 v59, v71, v81
	s_wait_loadcnt 0xd
	v_dual_mul_f32 v68, v41, v89 :: v_dual_mul_f32 v69, v51, v89
	v_dual_mul_f32 v78, v61, v89 :: v_dual_mul_f32 v79, v71, v89
	v_dual_fmac_f32 v48, v40, v80 :: v_dual_fmac_f32 v49, v50, v80
	v_dual_fmac_f32 v58, v60, v80 :: v_dual_fmac_f32 v59, v70, v80
	v_dual_fmac_f32 v68, v40, v88 :: v_dual_fmac_f32 v69, v50, v88
	v_dual_fmac_f32 v78, v60, v88 :: v_dual_fmac_f32 v79, v70, v88
	v_dual_fmac_f32 v48, v42, v82 :: v_dual_fmac_f32 v49, v52, v82
	v_dual_fmac_f32 v58, v62, v82 :: v_dual_fmac_f32 v59, v72, v82
	v_dual_fmac_f32 v68, v42, v90 :: v_dual_fmac_f32 v69, v52, v90
	v_dual_fmac_f32 v78, v62, v90 :: v_dual_fmac_f32 v79, v72, v90
	v_dual_fmac_f32 v48, v43, v83 :: v_dual_fmac_f32 v49, v53, v83
	v_dual_fmac_f32 v58, v63, v83 :: v_dual_fmac_f32 v59, v73, v83
	v_dual_fmac_f32 v68, v43, v91 :: v_dual_fmac_f32 v69, v53, v91
	v_dual_fmac_f32 v78, v63, v91 :: v_dual_fmac_f32 v79, v73, v91
	v_dual_fmac_f32 v48, v44, v84 :: v_dual_fmac_f32 v49, v54, v84
	v_dual_fmac_f32 v58, v64, v84 :: v_dual_fmac_f32 v59, v74, v84
	s_wait_loadcnt 0xc
	v_dual_fmac_f32 v68, v44, v92 :: v_dual_fmac_f32 v69, v54, v92
	v_dual_fmac_f32 v78, v64, v92 :: v_dual_fmac_f32 v79, v74, v92
	v_dual_fmac_f32 v48, v45, v85 :: v_dual_fmac_f32 v49, v55, v85
	v_dual_fmac_f32 v58, v65, v85 :: v_dual_fmac_f32 v59, v75, v85
	v_dual_fmac_f32 v68, v45, v93 :: v_dual_fmac_f32 v69, v55, v93
	v_dual_fmac_f32 v78, v65, v93 :: v_dual_fmac_f32 v79, v75, v93
	v_dual_fmac_f32 v48, v46, v86 :: v_dual_fmac_f32 v49, v56, v86
	v_dual_fmac_f32 v58, v66, v86 :: v_dual_fmac_f32 v59, v76, v86
	v_dual_fmac_f32 v68, v46, v94 :: v_dual_fmac_f32 v69, v56, v94
	v_dual_fmac_f32 v78, v66, v94 :: v_dual_fmac_f32 v79, v76, v94
	v_dual_fmac_f32 v48, v47, v87 :: v_dual_fmac_f32 v49, v57, v87
	v_dual_fmac_f32 v58, v67, v87 :: v_dual_fmac_f32 v59, v77, v87
	v_dual_fmac_f32 v68, v47, v95 :: v_dual_fmac_f32 v69, v57, v95
	v_dual_fmac_f32 v78, v67, v95 :: v_dual_fmac_f32 v79, v77, v95
	v_dual_add_f32 v184, v184, v48 :: v_dual_add_f32 v185, v185, v49
	v_dual_add_f32 v186, v186, v58 :: v_dual_add_f32 v187, v187, v59
	v_dual_add_f32 v200, v200, v68 :: v_dual_add_f32 v201, v201, v69
	v_dual_add_f32 v202, v202, v78 :: v_dual_add_f32 v203, v203, v79
	buffer_load_b128 v[80:83], v3, s[24:27], s31 offen offset:3072
	buffer_load_b128 v[84:87], v3, s[24:27], s31 offen offset:3088
	buffer_load_b128 v[88:91], v3, s[24:27], s32 offen offset:3072
	buffer_load_b128 v[92:95], v3, s[24:27], s32 offen offset:3088
	s_wait_loadcnt 0xf
	v_dual_mul_f32 v48, v41, v97 :: v_dual_mul_f32 v49, v51, v97
	v_dual_mul_f32 v58, v61, v97 :: v_dual_mul_f32 v59, v71, v97
	s_wait_loadcnt 0xd
	v_dual_mul_f32 v68, v41, v105 :: v_dual_mul_f32 v69, v51, v105
	v_dual_mul_f32 v78, v61, v105 :: v_dual_mul_f32 v79, v71, v105
	v_dual_fmac_f32 v48, v40, v96 :: v_dual_fmac_f32 v49, v50, v96
	v_dual_fmac_f32 v58, v60, v96 :: v_dual_fmac_f32 v59, v70, v96
	v_dual_fmac_f32 v68, v40, v104 :: v_dual_fmac_f32 v69, v50, v104
	v_dual_fmac_f32 v78, v60, v104 :: v_dual_fmac_f32 v79, v70, v104
	v_dual_fmac_f32 v48, v42, v98 :: v_dual_fmac_f32 v49, v52, v98
	v_dual_fmac_f32 v58, v62, v98 :: v_dual_fmac_f32 v59, v72, v98
	v_dual_fmac_f32 v68, v42, v106 :: v_dual_fmac_f32 v69, v52, v106
	v_dual_fmac_f32 v78, v62, v106 :: v_dual_fmac_f32 v79, v72, v106
	v_dual_fmac_f32 v48, v43, v99 :: v_dual_fmac_f32 v49, v53, v99
	v_dual_fmac_f32 v58, v63, v99 :: v_dual_fmac_f32 v59, v73, v99
	v_dual_fmac_f32 v68, v43, v107 :: v_dual_fmac_f32 v69, v53, v107
	v_dual_fmac_f32 v78, v63, v107 :: v_dual_fmac_f32 v79, v73, v107
	v_dual_fmac_f32 v48, v44, v100 :: v_dual_fmac_f32 v49, v54, v100
	v_dual_fmac_f32 v58, v64, v100 :: v_dual_fmac_f32 v59, v74, v100
	s_wait_loadcnt 0xc
	v_dual_fmac_f32 v68, v44, v108 :: v_dual_fmac_f32 v69, v54, v108
	v_dual_fmac_f32 v78, v64, v108 :: v_dual_fmac_f32 v79, v74, v108
	v_dual_fmac_f32 v48, v45, v101 :: v_dual_fmac_f32 v49, v55, v101
	v_dual_fmac_f32 v58, v65, v101 :: v_dual_fmac_f32 v59, v75, v101
	v_dual_fmac_f32 v68, v45, v109 :: v_dual_fmac_f32 v69, v55, v109
	v_dual_fmac_f32 v78, v65, v109 :: v_dual_fmac_f32 v79, v75, v109
	v_dual_fmac_f32 v48, v46, v102 :: v_dual_fmac_f32 v49, v56, v102
	v_dual_fmac_f32 v58, v66, v102 :: v_dual_fmac_f32 v59, v76, v102
	v_dual_fmac_f32 v68, v46, v110 :: v_dual_fmac_f32 v69, v56, v110
	v_dual_fmac_f32 v78, v66, v110 :: v_dual_fmac_f32 v79, v76, v110
	v_dual_fmac_f32 v48, v47, v103 :: v_dual_fmac_f32 v49, v57, v103
	v_dual_fmac_f32 v58, v67, v103 :: v_dual_fmac_f32 v59, v77, v103
	v_dual_fmac_f32 v68, v47, v111 :: v_dual_fmac_f32 v69, v57, v111
	v_dual_fmac_f32 v78, v67, v111 :: v_dual_fmac_f32 v79, v77, v111
	v_dual_add_f32 v216, v216, v48 :: v_dual_add_f32 v217, v217, v49
	v_dual_add_f32 v218, v218, v58 :: v_dual_add_f32 v219, v219, v59
	v_dual_add_f32 v232, v232, v68 :: v_dual_add_f32 v233, v233, v69
	v_dual_add_f32 v234, v234, v78 :: v_dual_add_f32 v235, v235, v79
	buffer_load_b128 v[96:99], v3, s[24:27], s33 offen offset:3072
	buffer_load_b128 v[100:103], v3, s[24:27], s33 offen offset:3088
	buffer_load_b128 v[104:107], v3, s[24:27], s34 offen offset:3072
	buffer_load_b128 v[108:111], v3, s[24:27], s34 offen offset:3088
	s_wait_loadcnt 0xf
	v_dual_mul_f32 v48, v41, v113 :: v_dual_mul_f32 v49, v51, v113
	v_dual_mul_f32 v58, v61, v113 :: v_dual_mul_f32 v59, v71, v113
	v_dual_fmac_f32 v48, v40, v112 :: v_dual_fmac_f32 v49, v50, v112
	v_dual_fmac_f32 v58, v60, v112 :: v_dual_fmac_f32 v59, v70, v112
	v_dual_fmac_f32 v48, v42, v114 :: v_dual_fmac_f32 v49, v52, v114
	v_dual_fmac_f32 v58, v62, v114 :: v_dual_fmac_f32 v59, v72, v114
	v_dual_fmac_f32 v48, v43, v115 :: v_dual_fmac_f32 v49, v53, v115
	v_dual_fmac_f32 v58, v63, v115 :: v_dual_fmac_f32 v59, v73, v115
	s_wait_loadcnt 0xe
	v_dual_fmac_f32 v48, v44, v116 :: v_dual_fmac_f32 v49, v54, v116
	v_dual_fmac_f32 v58, v64, v116 :: v_dual_fmac_f32 v59, v74, v116
	v_dual_fmac_f32 v48, v45, v117 :: v_dual_fmac_f32 v49, v55, v117
	v_dual_fmac_f32 v58, v65, v117 :: v_dual_fmac_f32 v59, v75, v117
	v_dual_fmac_f32 v48, v46, v118 :: v_dual_fmac_f32 v49, v56, v118
	v_dual_fmac_f32 v58, v66, v118 :: v_dual_fmac_f32 v59, v76, v118
	v_dual_fmac_f32 v48, v47, v119 :: v_dual_fmac_f32 v49, v57, v119
	v_dual_fmac_f32 v58, v67, v119 :: v_dual_fmac_f32 v59, v77, v119
	v_dual_add_f32 v248, v248, v48 :: v_dual_add_f32 v249, v249, v49
	v_dual_add_f32 v250, v250, v58 :: v_dual_add_f32 v251, v251, v59
	v_and_b32_e32 v46, 0xf0f0f0f, v36
	v_and_b32_e32 v56, 0xf0f0f0f, v37
	v_and_b32_e32 v66, 0xf0f0f0f, v38
	v_and_b32_e32 v76, 0xf0f0f0f, v39
	v_lshrrev_b32_e32 v47, 4, v36
	v_lshrrev_b32_e32 v57, 4, v37
	v_lshrrev_b32_e32 v67, 4, v38
	v_lshrrev_b32_e32 v77, 4, v39
	v_and_b32_e32 v47, 0xf0f0f0f, v47
	v_and_b32_e32 v57, 0xf0f0f0f, v57
	v_and_b32_e32 v67, 0xf0f0f0f, v67
	v_and_b32_e32 v77, 0xf0f0f0f, v77
	v_cvt_f32_ubyte0_e32 v40, v46
	v_cvt_f32_ubyte0_e32 v50, v56
	v_cvt_f32_ubyte0_e32 v60, v66
	v_cvt_f32_ubyte0_e32 v70, v76
	v_cvt_f32_ubyte1_e32 v42, v46
	v_cvt_f32_ubyte1_e32 v52, v56
	v_cvt_f32_ubyte1_e32 v62, v66
	v_cvt_f32_ubyte1_e32 v72, v76
	v_cvt_f32_ubyte2_e32 v44, v46
	v_cvt_f32_ubyte2_e32 v54, v56
	v_cvt_f32_ubyte2_e32 v64, v66
	v_cvt_f32_ubyte2_e32 v74, v76
	v_cvt_f32_ubyte3_e32 v46, v46
	v_cvt_f32_ubyte3_e32 v56, v56
	v_cvt_f32_ubyte3_e32 v66, v66
	v_cvt_f32_ubyte3_e32 v76, v76
	v_cvt_f32_ubyte0_e32 v41, v47
	v_cvt_f32_ubyte0_e32 v51, v57
	v_cvt_f32_ubyte0_e32 v61, v67
	v_cvt_f32_ubyte0_e32 v71, v77
	v_cvt_f32_ubyte1_e32 v43, v47
	v_cvt_f32_ubyte1_e32 v53, v57
	v_cvt_f32_ubyte1_e32 v63, v67
	v_cvt_f32_ubyte1_e32 v73, v77
	v_cvt_f32_ubyte2_e32 v45, v47
	v_cvt_f32_ubyte2_e32 v55, v57
	v_cvt_f32_ubyte2_e32 v65, v67
	v_cvt_f32_ubyte2_e32 v75, v77
	v_cvt_f32_ubyte3_e32 v47, v47
	v_cvt_f32_ubyte3_e32 v57, v57
	v_cvt_f32_ubyte3_e32 v67, v67
	v_cvt_f32_ubyte3_e32 v77, v77
	v_fma_mix_f32 v40, v20, v40, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v21, v50, v21 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v22, v60, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v70, v23, v70, v23 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v41, v20, v41, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v21, v51, v21 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v22, v61, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v71, v23, v71, v23 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v42, v20, v42, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v52, v21, v52, v21 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v62, v22, v62, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v72, v23, v72, v23 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v43, v20, v43, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v53, v21, v53, v21 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v63, v22, v63, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v73, v23, v73, v23 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v44, v20, v44, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v54, v21, v54, v21 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v64, v22, v64, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v74, v23, v74, v23 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v20, v45, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v21, v55, v21 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v65, v22, v65, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v75, v23, v75, v23 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v20, v46, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v21, v56, v21 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v66, v22, v66, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v76, v23, v76, v23 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v20, v47, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v21, v57, v21 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v67, v22, v67, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v77, v23, v77, v23 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xd
	v_dual_mul_f32 v48, v41, v121 :: v_dual_mul_f32 v49, v51, v121
	v_dual_mul_f32 v58, v61, v121 :: v_dual_mul_f32 v59, v71, v121
	s_wait_loadcnt 0xb
	v_dual_mul_f32 v68, v41, v129 :: v_dual_mul_f32 v69, v51, v129
	v_dual_mul_f32 v78, v61, v129 :: v_dual_mul_f32 v79, v71, v129
	v_dual_fmac_f32 v48, v40, v120 :: v_dual_fmac_f32 v49, v50, v120
	v_dual_fmac_f32 v58, v60, v120 :: v_dual_fmac_f32 v59, v70, v120
	v_dual_fmac_f32 v68, v40, v128 :: v_dual_fmac_f32 v69, v50, v128
	v_dual_fmac_f32 v78, v60, v128 :: v_dual_fmac_f32 v79, v70, v128
	v_dual_fmac_f32 v48, v42, v122 :: v_dual_fmac_f32 v49, v52, v122
	v_dual_fmac_f32 v58, v62, v122 :: v_dual_fmac_f32 v59, v72, v122
	v_dual_fmac_f32 v68, v42, v130 :: v_dual_fmac_f32 v69, v52, v130
	v_dual_fmac_f32 v78, v62, v130 :: v_dual_fmac_f32 v79, v72, v130
	v_dual_fmac_f32 v48, v43, v123 :: v_dual_fmac_f32 v49, v53, v123
	v_dual_fmac_f32 v58, v63, v123 :: v_dual_fmac_f32 v59, v73, v123
	v_dual_fmac_f32 v68, v43, v131 :: v_dual_fmac_f32 v69, v53, v131
	v_dual_fmac_f32 v78, v63, v131 :: v_dual_fmac_f32 v79, v73, v131
	v_dual_fmac_f32 v48, v44, v124 :: v_dual_fmac_f32 v49, v54, v124
	v_dual_fmac_f32 v58, v64, v124 :: v_dual_fmac_f32 v59, v74, v124
	s_wait_loadcnt 0xa
	v_dual_fmac_f32 v68, v44, v132 :: v_dual_fmac_f32 v69, v54, v132
	v_dual_fmac_f32 v78, v64, v132 :: v_dual_fmac_f32 v79, v74, v132
	v_dual_fmac_f32 v48, v45, v125 :: v_dual_fmac_f32 v49, v55, v125
	v_dual_fmac_f32 v58, v65, v125 :: v_dual_fmac_f32 v59, v75, v125
	v_dual_fmac_f32 v68, v45, v133 :: v_dual_fmac_f32 v69, v55, v133
	v_dual_fmac_f32 v78, v65, v133 :: v_dual_fmac_f32 v79, v75, v133
	v_dual_fmac_f32 v48, v46, v126 :: v_dual_fmac_f32 v49, v56, v126
	v_dual_fmac_f32 v58, v66, v126 :: v_dual_fmac_f32 v59, v76, v126
	v_dual_fmac_f32 v68, v46, v134 :: v_dual_fmac_f32 v69, v56, v134
	v_dual_fmac_f32 v78, v66, v134 :: v_dual_fmac_f32 v79, v76, v134
	v_dual_fmac_f32 v48, v47, v127 :: v_dual_fmac_f32 v49, v57, v127
	v_dual_fmac_f32 v58, v67, v127 :: v_dual_fmac_f32 v59, v77, v127
	v_dual_fmac_f32 v68, v47, v135 :: v_dual_fmac_f32 v69, v57, v135
	v_dual_fmac_f32 v78, v67, v135 :: v_dual_fmac_f32 v79, v77, v135
	v_dual_add_f32 v156, v156, v48 :: v_dual_add_f32 v157, v157, v49
	v_dual_add_f32 v158, v158, v58 :: v_dual_add_f32 v159, v159, v59
	v_dual_add_f32 v172, v172, v68 :: v_dual_add_f32 v173, v173, v69
	v_dual_add_f32 v174, v174, v78 :: v_dual_add_f32 v175, v175, v79
	s_wait_loadcnt 0x9
	v_dual_mul_f32 v48, v41, v137 :: v_dual_mul_f32 v49, v51, v137
	v_dual_mul_f32 v58, v61, v137 :: v_dual_mul_f32 v59, v71, v137
	s_wait_loadcnt 0x7
	v_dual_mul_f32 v68, v41, v81 :: v_dual_mul_f32 v69, v51, v81
	v_dual_mul_f32 v78, v61, v81 :: v_dual_mul_f32 v79, v71, v81
	v_dual_fmac_f32 v48, v40, v136 :: v_dual_fmac_f32 v49, v50, v136
	v_dual_fmac_f32 v58, v60, v136 :: v_dual_fmac_f32 v59, v70, v136
	v_dual_fmac_f32 v68, v40, v80 :: v_dual_fmac_f32 v69, v50, v80
	v_dual_fmac_f32 v78, v60, v80 :: v_dual_fmac_f32 v79, v70, v80
	v_dual_fmac_f32 v48, v42, v138 :: v_dual_fmac_f32 v49, v52, v138
	v_dual_fmac_f32 v58, v62, v138 :: v_dual_fmac_f32 v59, v72, v138
	v_dual_fmac_f32 v68, v42, v82 :: v_dual_fmac_f32 v69, v52, v82
	v_dual_fmac_f32 v78, v62, v82 :: v_dual_fmac_f32 v79, v72, v82
	v_dual_fmac_f32 v48, v43, v139 :: v_dual_fmac_f32 v49, v53, v139
	v_dual_fmac_f32 v58, v63, v139 :: v_dual_fmac_f32 v59, v73, v139
	v_dual_fmac_f32 v68, v43, v83 :: v_dual_fmac_f32 v69, v53, v83
	v_dual_fmac_f32 v78, v63, v83 :: v_dual_fmac_f32 v79, v73, v83
	v_dual_fmac_f32 v48, v44, v140 :: v_dual_fmac_f32 v49, v54, v140
	v_dual_fmac_f32 v58, v64, v140 :: v_dual_fmac_f32 v59, v74, v140
	s_wait_loadcnt 0x6
	v_dual_fmac_f32 v68, v44, v84 :: v_dual_fmac_f32 v69, v54, v84
	v_dual_fmac_f32 v78, v64, v84 :: v_dual_fmac_f32 v79, v74, v84
	v_dual_fmac_f32 v48, v45, v141 :: v_dual_fmac_f32 v49, v55, v141
	v_dual_fmac_f32 v58, v65, v141 :: v_dual_fmac_f32 v59, v75, v141
	v_dual_fmac_f32 v68, v45, v85 :: v_dual_fmac_f32 v69, v55, v85
	v_dual_fmac_f32 v78, v65, v85 :: v_dual_fmac_f32 v79, v75, v85
	v_dual_fmac_f32 v48, v46, v142 :: v_dual_fmac_f32 v49, v56, v142
	v_dual_fmac_f32 v58, v66, v142 :: v_dual_fmac_f32 v59, v76, v142
	v_dual_fmac_f32 v68, v46, v86 :: v_dual_fmac_f32 v69, v56, v86
	v_dual_fmac_f32 v78, v66, v86 :: v_dual_fmac_f32 v79, v76, v86
	v_dual_fmac_f32 v48, v47, v143 :: v_dual_fmac_f32 v49, v57, v143
	v_dual_fmac_f32 v58, v67, v143 :: v_dual_fmac_f32 v59, v77, v143
	v_dual_fmac_f32 v68, v47, v87 :: v_dual_fmac_f32 v69, v57, v87
	v_dual_fmac_f32 v78, v67, v87 :: v_dual_fmac_f32 v79, v77, v87
	v_dual_add_f32 v188, v188, v48 :: v_dual_add_f32 v189, v189, v49
	v_dual_add_f32 v190, v190, v58 :: v_dual_add_f32 v191, v191, v59
	v_dual_add_f32 v204, v204, v68 :: v_dual_add_f32 v205, v205, v69
	v_dual_add_f32 v206, v206, v78 :: v_dual_add_f32 v207, v207, v79
	s_wait_loadcnt 0x5
	v_dual_mul_f32 v48, v41, v89 :: v_dual_mul_f32 v49, v51, v89
	v_dual_mul_f32 v58, v61, v89 :: v_dual_mul_f32 v59, v71, v89
	s_wait_loadcnt 0x3
	v_dual_mul_f32 v68, v41, v97 :: v_dual_mul_f32 v69, v51, v97
	v_dual_mul_f32 v78, v61, v97 :: v_dual_mul_f32 v79, v71, v97
	v_dual_fmac_f32 v48, v40, v88 :: v_dual_fmac_f32 v49, v50, v88
	v_dual_fmac_f32 v58, v60, v88 :: v_dual_fmac_f32 v59, v70, v88
	v_dual_fmac_f32 v68, v40, v96 :: v_dual_fmac_f32 v69, v50, v96
	v_dual_fmac_f32 v78, v60, v96 :: v_dual_fmac_f32 v79, v70, v96
	v_dual_fmac_f32 v48, v42, v90 :: v_dual_fmac_f32 v49, v52, v90
	v_dual_fmac_f32 v58, v62, v90 :: v_dual_fmac_f32 v59, v72, v90
	v_dual_fmac_f32 v68, v42, v98 :: v_dual_fmac_f32 v69, v52, v98
	v_dual_fmac_f32 v78, v62, v98 :: v_dual_fmac_f32 v79, v72, v98
	v_dual_fmac_f32 v48, v43, v91 :: v_dual_fmac_f32 v49, v53, v91
	v_dual_fmac_f32 v58, v63, v91 :: v_dual_fmac_f32 v59, v73, v91
	v_dual_fmac_f32 v68, v43, v99 :: v_dual_fmac_f32 v69, v53, v99
	v_dual_fmac_f32 v78, v63, v99 :: v_dual_fmac_f32 v79, v73, v99
	v_dual_fmac_f32 v48, v44, v92 :: v_dual_fmac_f32 v49, v54, v92
	v_dual_fmac_f32 v58, v64, v92 :: v_dual_fmac_f32 v59, v74, v92
	s_wait_loadcnt 0x2
	v_dual_fmac_f32 v68, v44, v100 :: v_dual_fmac_f32 v69, v54, v100
	v_dual_fmac_f32 v78, v64, v100 :: v_dual_fmac_f32 v79, v74, v100
	v_dual_fmac_f32 v48, v45, v93 :: v_dual_fmac_f32 v49, v55, v93
	v_dual_fmac_f32 v58, v65, v93 :: v_dual_fmac_f32 v59, v75, v93
	v_dual_fmac_f32 v68, v45, v101 :: v_dual_fmac_f32 v69, v55, v101
	v_dual_fmac_f32 v78, v65, v101 :: v_dual_fmac_f32 v79, v75, v101
	v_dual_fmac_f32 v48, v46, v94 :: v_dual_fmac_f32 v49, v56, v94
	v_dual_fmac_f32 v58, v66, v94 :: v_dual_fmac_f32 v59, v76, v94
	v_dual_fmac_f32 v68, v46, v102 :: v_dual_fmac_f32 v69, v56, v102
	v_dual_fmac_f32 v78, v66, v102 :: v_dual_fmac_f32 v79, v76, v102
	v_dual_fmac_f32 v48, v47, v95 :: v_dual_fmac_f32 v49, v57, v95
	v_dual_fmac_f32 v58, v67, v95 :: v_dual_fmac_f32 v59, v77, v95
	v_dual_fmac_f32 v68, v47, v103 :: v_dual_fmac_f32 v69, v57, v103
	v_dual_fmac_f32 v78, v67, v103 :: v_dual_fmac_f32 v79, v77, v103
	v_dual_add_f32 v220, v220, v48 :: v_dual_add_f32 v221, v221, v49
	v_dual_add_f32 v222, v222, v58 :: v_dual_add_f32 v223, v223, v59
	v_dual_add_f32 v236, v236, v68 :: v_dual_add_f32 v237, v237, v69
	v_dual_add_f32 v238, v238, v78 :: v_dual_add_f32 v239, v239, v79
	s_wait_loadcnt 0x1
	v_dual_mul_f32 v48, v41, v105 :: v_dual_mul_f32 v49, v51, v105
	v_dual_mul_f32 v58, v61, v105 :: v_dual_mul_f32 v59, v71, v105
	v_dual_fmac_f32 v48, v40, v104 :: v_dual_fmac_f32 v49, v50, v104
	v_dual_fmac_f32 v58, v60, v104 :: v_dual_fmac_f32 v59, v70, v104
	v_dual_fmac_f32 v48, v42, v106 :: v_dual_fmac_f32 v49, v52, v106
	v_dual_fmac_f32 v58, v62, v106 :: v_dual_fmac_f32 v59, v72, v106
	v_dual_fmac_f32 v48, v43, v107 :: v_dual_fmac_f32 v49, v53, v107
	v_dual_fmac_f32 v58, v63, v107 :: v_dual_fmac_f32 v59, v73, v107
	s_wait_loadcnt 0x0
	v_dual_fmac_f32 v48, v44, v108 :: v_dual_fmac_f32 v49, v54, v108
	v_dual_fmac_f32 v58, v64, v108 :: v_dual_fmac_f32 v59, v74, v108
	v_dual_fmac_f32 v48, v45, v109 :: v_dual_fmac_f32 v49, v55, v109
	v_dual_fmac_f32 v58, v65, v109 :: v_dual_fmac_f32 v59, v75, v109
	v_dual_fmac_f32 v48, v46, v110 :: v_dual_fmac_f32 v49, v56, v110
	v_dual_fmac_f32 v58, v66, v110 :: v_dual_fmac_f32 v59, v76, v110
	v_dual_fmac_f32 v48, v47, v111 :: v_dual_fmac_f32 v49, v57, v111
	v_dual_fmac_f32 v58, v67, v111 :: v_dual_fmac_f32 v59, v77, v111
	v_dual_add_f32 v252, v252, v48 :: v_dual_add_f32 v253, v253, v49
	v_dual_add_f32 v254, v254, v58 :: v_dual_add_f32 v255, v255, v59
	s_add_co_i32 s16, s16, 0x220
	s_add_co_i32 s17, s17, 0x220
	s_add_co_i32 s18, s18, 0x220
	s_add_co_i32 s19, s19, 0x220
	s_add_co_i32 s28, s28, 0x1000
	s_add_co_i32 s29, s29, 0x1000
	s_add_co_i32 s30, s30, 0x1000
	s_add_co_i32 s31, s31, 0x1000
	s_add_co_i32 s32, s32, 0x1000
	s_add_co_i32 s33, s33, 0x1000
	s_add_co_i32 s34, s34, 0x1000
	s_add_co_i32 s36, s36, -1
	s_cmp_lg_u32 s36, 0
	s_cbranch_scc1 .Lpx_b7_quad
	.Lpx_b7_tail:
	s_cmp_gt_u32 s37, 0
	s_cbranch_scc0 .Lpx_b7_fold
	buffer_load_b32 v8, v1, s[20:23], s16 offen
	buffer_load_b32 v9, v1, s[20:23], s17 offen
	buffer_load_b32 v10, v1, s[20:23], s18 offen
	buffer_load_b32 v11, v1, s[20:23], s19 offen
	buffer_load_b32 v24, v2, s[20:23], s16 offen offset:8
	buffer_load_b32 v25, v2, s[20:23], s17 offen offset:8
	buffer_load_b32 v26, v2, s[20:23], s18 offen offset:8
	buffer_load_b32 v27, v2, s[20:23], s19 offen offset:8
	buffer_load_b128 v[80:83], v3, s[24:27], s28 offen
	buffer_load_b128 v[84:87], v3, s[24:27], s28 offen offset:16
	buffer_load_b128 v[88:91], v3, s[24:27], s29 offen
	buffer_load_b128 v[92:95], v3, s[24:27], s29 offen offset:16
	buffer_load_b128 v[96:99], v3, s[24:27], s30 offen
	buffer_load_b128 v[100:103], v3, s[24:27], s30 offen offset:16
	buffer_load_b128 v[104:107], v3, s[24:27], s31 offen
	buffer_load_b128 v[108:111], v3, s[24:27], s31 offen offset:16
	buffer_load_b128 v[112:115], v3, s[24:27], s32 offen
	buffer_load_b128 v[116:119], v3, s[24:27], s32 offen offset:16
	buffer_load_b128 v[120:123], v3, s[24:27], s33 offen
	buffer_load_b128 v[124:127], v3, s[24:27], s33 offen offset:16
	buffer_load_b128 v[128:131], v3, s[24:27], s34 offen
	buffer_load_b128 v[132:135], v3, s[24:27], s34 offen offset:16
	s_wait_loadcnt 0x11
	v_and_b32_e32 v46, 0xf0f0f0f, v24
	s_wait_loadcnt 0x10
	v_and_b32_e32 v56, 0xf0f0f0f, v25
	s_wait_loadcnt 0xf
	v_and_b32_e32 v66, 0xf0f0f0f, v26
	s_wait_loadcnt 0xe
	v_and_b32_e32 v76, 0xf0f0f0f, v27
	v_lshrrev_b32_e32 v47, 4, v24
	v_lshrrev_b32_e32 v57, 4, v25
	v_lshrrev_b32_e32 v67, 4, v26
	v_lshrrev_b32_e32 v77, 4, v27
	v_and_b32_e32 v47, 0xf0f0f0f, v47
	v_and_b32_e32 v57, 0xf0f0f0f, v57
	v_and_b32_e32 v67, 0xf0f0f0f, v67
	v_and_b32_e32 v77, 0xf0f0f0f, v77
	v_cvt_f32_ubyte0_e32 v40, v46
	v_cvt_f32_ubyte0_e32 v50, v56
	v_cvt_f32_ubyte0_e32 v60, v66
	v_cvt_f32_ubyte0_e32 v70, v76
	v_cvt_f32_ubyte1_e32 v42, v46
	v_cvt_f32_ubyte1_e32 v52, v56
	v_cvt_f32_ubyte1_e32 v62, v66
	v_cvt_f32_ubyte1_e32 v72, v76
	v_cvt_f32_ubyte2_e32 v44, v46
	v_cvt_f32_ubyte2_e32 v54, v56
	v_cvt_f32_ubyte2_e32 v64, v66
	v_cvt_f32_ubyte2_e32 v74, v76
	v_cvt_f32_ubyte3_e32 v46, v46
	v_cvt_f32_ubyte3_e32 v56, v56
	v_cvt_f32_ubyte3_e32 v66, v66
	v_cvt_f32_ubyte3_e32 v76, v76
	v_cvt_f32_ubyte0_e32 v41, v47
	v_cvt_f32_ubyte0_e32 v51, v57
	v_cvt_f32_ubyte0_e32 v61, v67
	v_cvt_f32_ubyte0_e32 v71, v77
	v_cvt_f32_ubyte1_e32 v43, v47
	v_cvt_f32_ubyte1_e32 v53, v57
	v_cvt_f32_ubyte1_e32 v63, v67
	v_cvt_f32_ubyte1_e32 v73, v77
	v_cvt_f32_ubyte2_e32 v45, v47
	v_cvt_f32_ubyte2_e32 v55, v57
	v_cvt_f32_ubyte2_e32 v65, v67
	v_cvt_f32_ubyte2_e32 v75, v77
	v_cvt_f32_ubyte3_e32 v47, v47
	v_cvt_f32_ubyte3_e32 v57, v57
	v_cvt_f32_ubyte3_e32 v67, v67
	v_cvt_f32_ubyte3_e32 v77, v77
	v_fma_mix_f32 v40, v8, v40, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v9, v50, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v10, v60, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v70, v11, v70, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v41, v8, v41, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v9, v51, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v10, v61, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v71, v11, v71, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v42, v8, v42, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v52, v9, v52, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v62, v10, v62, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v72, v11, v72, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v43, v8, v43, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v53, v9, v53, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v63, v10, v63, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v73, v11, v73, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v44, v8, v44, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v54, v9, v54, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v64, v10, v64, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v74, v11, v74, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v8, v45, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v9, v55, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v65, v10, v65, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v75, v11, v75, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v8, v46, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v9, v56, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v66, v10, v66, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v76, v11, v76, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v8, v47, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v9, v57, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v67, v10, v67, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v77, v11, v77, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xd
	v_dual_mul_f32 v48, v41, v81 :: v_dual_mul_f32 v49, v51, v81
	v_dual_mul_f32 v58, v61, v81 :: v_dual_mul_f32 v59, v71, v81
	s_wait_loadcnt 0xb
	v_dual_mul_f32 v68, v41, v89 :: v_dual_mul_f32 v69, v51, v89
	v_dual_mul_f32 v78, v61, v89 :: v_dual_mul_f32 v79, v71, v89
	v_dual_fmac_f32 v48, v40, v80 :: v_dual_fmac_f32 v49, v50, v80
	v_dual_fmac_f32 v58, v60, v80 :: v_dual_fmac_f32 v59, v70, v80
	v_dual_fmac_f32 v68, v40, v88 :: v_dual_fmac_f32 v69, v50, v88
	v_dual_fmac_f32 v78, v60, v88 :: v_dual_fmac_f32 v79, v70, v88
	v_dual_fmac_f32 v48, v42, v82 :: v_dual_fmac_f32 v49, v52, v82
	v_dual_fmac_f32 v58, v62, v82 :: v_dual_fmac_f32 v59, v72, v82
	v_dual_fmac_f32 v68, v42, v90 :: v_dual_fmac_f32 v69, v52, v90
	v_dual_fmac_f32 v78, v62, v90 :: v_dual_fmac_f32 v79, v72, v90
	v_dual_fmac_f32 v48, v43, v83 :: v_dual_fmac_f32 v49, v53, v83
	v_dual_fmac_f32 v58, v63, v83 :: v_dual_fmac_f32 v59, v73, v83
	v_dual_fmac_f32 v68, v43, v91 :: v_dual_fmac_f32 v69, v53, v91
	v_dual_fmac_f32 v78, v63, v91 :: v_dual_fmac_f32 v79, v73, v91
	v_dual_fmac_f32 v48, v44, v84 :: v_dual_fmac_f32 v49, v54, v84
	v_dual_fmac_f32 v58, v64, v84 :: v_dual_fmac_f32 v59, v74, v84
	s_wait_loadcnt 0xa
	v_dual_fmac_f32 v68, v44, v92 :: v_dual_fmac_f32 v69, v54, v92
	v_dual_fmac_f32 v78, v64, v92 :: v_dual_fmac_f32 v79, v74, v92
	v_dual_fmac_f32 v48, v45, v85 :: v_dual_fmac_f32 v49, v55, v85
	v_dual_fmac_f32 v58, v65, v85 :: v_dual_fmac_f32 v59, v75, v85
	v_dual_fmac_f32 v68, v45, v93 :: v_dual_fmac_f32 v69, v55, v93
	v_dual_fmac_f32 v78, v65, v93 :: v_dual_fmac_f32 v79, v75, v93
	v_dual_fmac_f32 v48, v46, v86 :: v_dual_fmac_f32 v49, v56, v86
	v_dual_fmac_f32 v58, v66, v86 :: v_dual_fmac_f32 v59, v76, v86
	v_dual_fmac_f32 v68, v46, v94 :: v_dual_fmac_f32 v69, v56, v94
	v_dual_fmac_f32 v78, v66, v94 :: v_dual_fmac_f32 v79, v76, v94
	v_dual_fmac_f32 v48, v47, v87 :: v_dual_fmac_f32 v49, v57, v87
	v_dual_fmac_f32 v58, v67, v87 :: v_dual_fmac_f32 v59, v77, v87
	v_dual_fmac_f32 v68, v47, v95 :: v_dual_fmac_f32 v69, v57, v95
	v_dual_fmac_f32 v78, v67, v95 :: v_dual_fmac_f32 v79, v77, v95
	v_dual_add_f32 v144, v144, v48 :: v_dual_add_f32 v145, v145, v49
	v_dual_add_f32 v146, v146, v58 :: v_dual_add_f32 v147, v147, v59
	v_dual_add_f32 v160, v160, v68 :: v_dual_add_f32 v161, v161, v69
	v_dual_add_f32 v162, v162, v78 :: v_dual_add_f32 v163, v163, v79
	s_wait_loadcnt 0x9
	v_dual_mul_f32 v48, v41, v97 :: v_dual_mul_f32 v49, v51, v97
	v_dual_mul_f32 v58, v61, v97 :: v_dual_mul_f32 v59, v71, v97
	s_wait_loadcnt 0x7
	v_dual_mul_f32 v68, v41, v105 :: v_dual_mul_f32 v69, v51, v105
	v_dual_mul_f32 v78, v61, v105 :: v_dual_mul_f32 v79, v71, v105
	v_dual_fmac_f32 v48, v40, v96 :: v_dual_fmac_f32 v49, v50, v96
	v_dual_fmac_f32 v58, v60, v96 :: v_dual_fmac_f32 v59, v70, v96
	v_dual_fmac_f32 v68, v40, v104 :: v_dual_fmac_f32 v69, v50, v104
	v_dual_fmac_f32 v78, v60, v104 :: v_dual_fmac_f32 v79, v70, v104
	v_dual_fmac_f32 v48, v42, v98 :: v_dual_fmac_f32 v49, v52, v98
	v_dual_fmac_f32 v58, v62, v98 :: v_dual_fmac_f32 v59, v72, v98
	v_dual_fmac_f32 v68, v42, v106 :: v_dual_fmac_f32 v69, v52, v106
	v_dual_fmac_f32 v78, v62, v106 :: v_dual_fmac_f32 v79, v72, v106
	v_dual_fmac_f32 v48, v43, v99 :: v_dual_fmac_f32 v49, v53, v99
	v_dual_fmac_f32 v58, v63, v99 :: v_dual_fmac_f32 v59, v73, v99
	v_dual_fmac_f32 v68, v43, v107 :: v_dual_fmac_f32 v69, v53, v107
	v_dual_fmac_f32 v78, v63, v107 :: v_dual_fmac_f32 v79, v73, v107
	v_dual_fmac_f32 v48, v44, v100 :: v_dual_fmac_f32 v49, v54, v100
	v_dual_fmac_f32 v58, v64, v100 :: v_dual_fmac_f32 v59, v74, v100
	s_wait_loadcnt 0x6
	v_dual_fmac_f32 v68, v44, v108 :: v_dual_fmac_f32 v69, v54, v108
	v_dual_fmac_f32 v78, v64, v108 :: v_dual_fmac_f32 v79, v74, v108
	v_dual_fmac_f32 v48, v45, v101 :: v_dual_fmac_f32 v49, v55, v101
	v_dual_fmac_f32 v58, v65, v101 :: v_dual_fmac_f32 v59, v75, v101
	v_dual_fmac_f32 v68, v45, v109 :: v_dual_fmac_f32 v69, v55, v109
	v_dual_fmac_f32 v78, v65, v109 :: v_dual_fmac_f32 v79, v75, v109
	v_dual_fmac_f32 v48, v46, v102 :: v_dual_fmac_f32 v49, v56, v102
	v_dual_fmac_f32 v58, v66, v102 :: v_dual_fmac_f32 v59, v76, v102
	v_dual_fmac_f32 v68, v46, v110 :: v_dual_fmac_f32 v69, v56, v110
	v_dual_fmac_f32 v78, v66, v110 :: v_dual_fmac_f32 v79, v76, v110
	v_dual_fmac_f32 v48, v47, v103 :: v_dual_fmac_f32 v49, v57, v103
	v_dual_fmac_f32 v58, v67, v103 :: v_dual_fmac_f32 v59, v77, v103
	v_dual_fmac_f32 v68, v47, v111 :: v_dual_fmac_f32 v69, v57, v111
	v_dual_fmac_f32 v78, v67, v111 :: v_dual_fmac_f32 v79, v77, v111
	v_dual_add_f32 v176, v176, v48 :: v_dual_add_f32 v177, v177, v49
	v_dual_add_f32 v178, v178, v58 :: v_dual_add_f32 v179, v179, v59
	v_dual_add_f32 v192, v192, v68 :: v_dual_add_f32 v193, v193, v69
	v_dual_add_f32 v194, v194, v78 :: v_dual_add_f32 v195, v195, v79
	s_wait_loadcnt 0x5
	v_dual_mul_f32 v48, v41, v113 :: v_dual_mul_f32 v49, v51, v113
	v_dual_mul_f32 v58, v61, v113 :: v_dual_mul_f32 v59, v71, v113
	s_wait_loadcnt 0x3
	v_dual_mul_f32 v68, v41, v121 :: v_dual_mul_f32 v69, v51, v121
	v_dual_mul_f32 v78, v61, v121 :: v_dual_mul_f32 v79, v71, v121
	v_dual_fmac_f32 v48, v40, v112 :: v_dual_fmac_f32 v49, v50, v112
	v_dual_fmac_f32 v58, v60, v112 :: v_dual_fmac_f32 v59, v70, v112
	v_dual_fmac_f32 v68, v40, v120 :: v_dual_fmac_f32 v69, v50, v120
	v_dual_fmac_f32 v78, v60, v120 :: v_dual_fmac_f32 v79, v70, v120
	v_dual_fmac_f32 v48, v42, v114 :: v_dual_fmac_f32 v49, v52, v114
	v_dual_fmac_f32 v58, v62, v114 :: v_dual_fmac_f32 v59, v72, v114
	v_dual_fmac_f32 v68, v42, v122 :: v_dual_fmac_f32 v69, v52, v122
	v_dual_fmac_f32 v78, v62, v122 :: v_dual_fmac_f32 v79, v72, v122
	v_dual_fmac_f32 v48, v43, v115 :: v_dual_fmac_f32 v49, v53, v115
	v_dual_fmac_f32 v58, v63, v115 :: v_dual_fmac_f32 v59, v73, v115
	v_dual_fmac_f32 v68, v43, v123 :: v_dual_fmac_f32 v69, v53, v123
	v_dual_fmac_f32 v78, v63, v123 :: v_dual_fmac_f32 v79, v73, v123
	v_dual_fmac_f32 v48, v44, v116 :: v_dual_fmac_f32 v49, v54, v116
	v_dual_fmac_f32 v58, v64, v116 :: v_dual_fmac_f32 v59, v74, v116
	s_wait_loadcnt 0x2
	v_dual_fmac_f32 v68, v44, v124 :: v_dual_fmac_f32 v69, v54, v124
	v_dual_fmac_f32 v78, v64, v124 :: v_dual_fmac_f32 v79, v74, v124
	v_dual_fmac_f32 v48, v45, v117 :: v_dual_fmac_f32 v49, v55, v117
	v_dual_fmac_f32 v58, v65, v117 :: v_dual_fmac_f32 v59, v75, v117
	v_dual_fmac_f32 v68, v45, v125 :: v_dual_fmac_f32 v69, v55, v125
	v_dual_fmac_f32 v78, v65, v125 :: v_dual_fmac_f32 v79, v75, v125
	v_dual_fmac_f32 v48, v46, v118 :: v_dual_fmac_f32 v49, v56, v118
	v_dual_fmac_f32 v58, v66, v118 :: v_dual_fmac_f32 v59, v76, v118
	v_dual_fmac_f32 v68, v46, v126 :: v_dual_fmac_f32 v69, v56, v126
	v_dual_fmac_f32 v78, v66, v126 :: v_dual_fmac_f32 v79, v76, v126
	v_dual_fmac_f32 v48, v47, v119 :: v_dual_fmac_f32 v49, v57, v119
	v_dual_fmac_f32 v58, v67, v119 :: v_dual_fmac_f32 v59, v77, v119
	v_dual_fmac_f32 v68, v47, v127 :: v_dual_fmac_f32 v69, v57, v127
	v_dual_fmac_f32 v78, v67, v127 :: v_dual_fmac_f32 v79, v77, v127
	v_dual_add_f32 v208, v208, v48 :: v_dual_add_f32 v209, v209, v49
	v_dual_add_f32 v210, v210, v58 :: v_dual_add_f32 v211, v211, v59
	v_dual_add_f32 v224, v224, v68 :: v_dual_add_f32 v225, v225, v69
	v_dual_add_f32 v226, v226, v78 :: v_dual_add_f32 v227, v227, v79
	s_wait_loadcnt 0x1
	v_dual_mul_f32 v48, v41, v129 :: v_dual_mul_f32 v49, v51, v129
	v_dual_mul_f32 v58, v61, v129 :: v_dual_mul_f32 v59, v71, v129
	v_dual_fmac_f32 v48, v40, v128 :: v_dual_fmac_f32 v49, v50, v128
	v_dual_fmac_f32 v58, v60, v128 :: v_dual_fmac_f32 v59, v70, v128
	v_dual_fmac_f32 v48, v42, v130 :: v_dual_fmac_f32 v49, v52, v130
	v_dual_fmac_f32 v58, v62, v130 :: v_dual_fmac_f32 v59, v72, v130
	v_dual_fmac_f32 v48, v43, v131 :: v_dual_fmac_f32 v49, v53, v131
	v_dual_fmac_f32 v58, v63, v131 :: v_dual_fmac_f32 v59, v73, v131
	s_wait_loadcnt 0x0
	v_dual_fmac_f32 v48, v44, v132 :: v_dual_fmac_f32 v49, v54, v132
	v_dual_fmac_f32 v58, v64, v132 :: v_dual_fmac_f32 v59, v74, v132
	v_dual_fmac_f32 v48, v45, v133 :: v_dual_fmac_f32 v49, v55, v133
	v_dual_fmac_f32 v58, v65, v133 :: v_dual_fmac_f32 v59, v75, v133
	v_dual_fmac_f32 v48, v46, v134 :: v_dual_fmac_f32 v49, v56, v134
	v_dual_fmac_f32 v58, v66, v134 :: v_dual_fmac_f32 v59, v76, v134
	v_dual_fmac_f32 v48, v47, v135 :: v_dual_fmac_f32 v49, v57, v135
	v_dual_fmac_f32 v58, v67, v135 :: v_dual_fmac_f32 v59, v77, v135
	v_dual_add_f32 v240, v240, v48 :: v_dual_add_f32 v241, v241, v49
	v_dual_add_f32 v242, v242, v58 :: v_dual_add_f32 v243, v243, v59
	s_cmp_gt_u32 s37, 1
	s_cbranch_scc0 .Lpx_b7_fold
	buffer_load_b32 v12, v1, s[20:23], s16 offen offset:136
	buffer_load_b32 v13, v1, s[20:23], s17 offen offset:136
	buffer_load_b32 v14, v1, s[20:23], s18 offen offset:136
	buffer_load_b32 v15, v1, s[20:23], s19 offen offset:136
	buffer_load_b32 v28, v2, s[20:23], s16 offen offset:144
	buffer_load_b32 v29, v2, s[20:23], s17 offen offset:144
	buffer_load_b32 v30, v2, s[20:23], s18 offen offset:144
	buffer_load_b32 v31, v2, s[20:23], s19 offen offset:144
	buffer_load_b128 v[80:83], v3, s[24:27], s28 offen offset:1024
	buffer_load_b128 v[84:87], v3, s[24:27], s28 offen offset:1040
	buffer_load_b128 v[88:91], v3, s[24:27], s29 offen offset:1024
	buffer_load_b128 v[92:95], v3, s[24:27], s29 offen offset:1040
	buffer_load_b128 v[96:99], v3, s[24:27], s30 offen offset:1024
	buffer_load_b128 v[100:103], v3, s[24:27], s30 offen offset:1040
	buffer_load_b128 v[104:107], v3, s[24:27], s31 offen offset:1024
	buffer_load_b128 v[108:111], v3, s[24:27], s31 offen offset:1040
	buffer_load_b128 v[112:115], v3, s[24:27], s32 offen offset:1024
	buffer_load_b128 v[116:119], v3, s[24:27], s32 offen offset:1040
	buffer_load_b128 v[120:123], v3, s[24:27], s33 offen offset:1024
	buffer_load_b128 v[124:127], v3, s[24:27], s33 offen offset:1040
	buffer_load_b128 v[128:131], v3, s[24:27], s34 offen offset:1024
	buffer_load_b128 v[132:135], v3, s[24:27], s34 offen offset:1040
	s_wait_loadcnt 0x11
	v_and_b32_e32 v46, 0xf0f0f0f, v28
	s_wait_loadcnt 0x10
	v_and_b32_e32 v56, 0xf0f0f0f, v29
	s_wait_loadcnt 0xf
	v_and_b32_e32 v66, 0xf0f0f0f, v30
	s_wait_loadcnt 0xe
	v_and_b32_e32 v76, 0xf0f0f0f, v31
	v_lshrrev_b32_e32 v47, 4, v28
	v_lshrrev_b32_e32 v57, 4, v29
	v_lshrrev_b32_e32 v67, 4, v30
	v_lshrrev_b32_e32 v77, 4, v31
	v_and_b32_e32 v47, 0xf0f0f0f, v47
	v_and_b32_e32 v57, 0xf0f0f0f, v57
	v_and_b32_e32 v67, 0xf0f0f0f, v67
	v_and_b32_e32 v77, 0xf0f0f0f, v77
	v_cvt_f32_ubyte0_e32 v40, v46
	v_cvt_f32_ubyte0_e32 v50, v56
	v_cvt_f32_ubyte0_e32 v60, v66
	v_cvt_f32_ubyte0_e32 v70, v76
	v_cvt_f32_ubyte1_e32 v42, v46
	v_cvt_f32_ubyte1_e32 v52, v56
	v_cvt_f32_ubyte1_e32 v62, v66
	v_cvt_f32_ubyte1_e32 v72, v76
	v_cvt_f32_ubyte2_e32 v44, v46
	v_cvt_f32_ubyte2_e32 v54, v56
	v_cvt_f32_ubyte2_e32 v64, v66
	v_cvt_f32_ubyte2_e32 v74, v76
	v_cvt_f32_ubyte3_e32 v46, v46
	v_cvt_f32_ubyte3_e32 v56, v56
	v_cvt_f32_ubyte3_e32 v66, v66
	v_cvt_f32_ubyte3_e32 v76, v76
	v_cvt_f32_ubyte0_e32 v41, v47
	v_cvt_f32_ubyte0_e32 v51, v57
	v_cvt_f32_ubyte0_e32 v61, v67
	v_cvt_f32_ubyte0_e32 v71, v77
	v_cvt_f32_ubyte1_e32 v43, v47
	v_cvt_f32_ubyte1_e32 v53, v57
	v_cvt_f32_ubyte1_e32 v63, v67
	v_cvt_f32_ubyte1_e32 v73, v77
	v_cvt_f32_ubyte2_e32 v45, v47
	v_cvt_f32_ubyte2_e32 v55, v57
	v_cvt_f32_ubyte2_e32 v65, v67
	v_cvt_f32_ubyte2_e32 v75, v77
	v_cvt_f32_ubyte3_e32 v47, v47
	v_cvt_f32_ubyte3_e32 v57, v57
	v_cvt_f32_ubyte3_e32 v67, v67
	v_cvt_f32_ubyte3_e32 v77, v77
	v_fma_mix_f32 v40, v12, v40, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v13, v50, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v14, v60, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v70, v15, v70, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v41, v12, v41, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v13, v51, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v14, v61, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v71, v15, v71, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v42, v12, v42, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v52, v13, v52, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v62, v14, v62, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v72, v15, v72, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v43, v12, v43, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v53, v13, v53, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v63, v14, v63, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v73, v15, v73, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v44, v12, v44, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v54, v13, v54, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v64, v14, v64, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v74, v15, v74, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v12, v45, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v13, v55, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v65, v14, v65, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v75, v15, v75, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v12, v46, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v13, v56, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v66, v14, v66, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v76, v15, v76, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v12, v47, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v13, v57, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v67, v14, v67, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v77, v15, v77, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xd
	v_dual_mul_f32 v48, v41, v81 :: v_dual_mul_f32 v49, v51, v81
	v_dual_mul_f32 v58, v61, v81 :: v_dual_mul_f32 v59, v71, v81
	s_wait_loadcnt 0xb
	v_dual_mul_f32 v68, v41, v89 :: v_dual_mul_f32 v69, v51, v89
	v_dual_mul_f32 v78, v61, v89 :: v_dual_mul_f32 v79, v71, v89
	v_dual_fmac_f32 v48, v40, v80 :: v_dual_fmac_f32 v49, v50, v80
	v_dual_fmac_f32 v58, v60, v80 :: v_dual_fmac_f32 v59, v70, v80
	v_dual_fmac_f32 v68, v40, v88 :: v_dual_fmac_f32 v69, v50, v88
	v_dual_fmac_f32 v78, v60, v88 :: v_dual_fmac_f32 v79, v70, v88
	v_dual_fmac_f32 v48, v42, v82 :: v_dual_fmac_f32 v49, v52, v82
	v_dual_fmac_f32 v58, v62, v82 :: v_dual_fmac_f32 v59, v72, v82
	v_dual_fmac_f32 v68, v42, v90 :: v_dual_fmac_f32 v69, v52, v90
	v_dual_fmac_f32 v78, v62, v90 :: v_dual_fmac_f32 v79, v72, v90
	v_dual_fmac_f32 v48, v43, v83 :: v_dual_fmac_f32 v49, v53, v83
	v_dual_fmac_f32 v58, v63, v83 :: v_dual_fmac_f32 v59, v73, v83
	v_dual_fmac_f32 v68, v43, v91 :: v_dual_fmac_f32 v69, v53, v91
	v_dual_fmac_f32 v78, v63, v91 :: v_dual_fmac_f32 v79, v73, v91
	v_dual_fmac_f32 v48, v44, v84 :: v_dual_fmac_f32 v49, v54, v84
	v_dual_fmac_f32 v58, v64, v84 :: v_dual_fmac_f32 v59, v74, v84
	s_wait_loadcnt 0xa
	v_dual_fmac_f32 v68, v44, v92 :: v_dual_fmac_f32 v69, v54, v92
	v_dual_fmac_f32 v78, v64, v92 :: v_dual_fmac_f32 v79, v74, v92
	v_dual_fmac_f32 v48, v45, v85 :: v_dual_fmac_f32 v49, v55, v85
	v_dual_fmac_f32 v58, v65, v85 :: v_dual_fmac_f32 v59, v75, v85
	v_dual_fmac_f32 v68, v45, v93 :: v_dual_fmac_f32 v69, v55, v93
	v_dual_fmac_f32 v78, v65, v93 :: v_dual_fmac_f32 v79, v75, v93
	v_dual_fmac_f32 v48, v46, v86 :: v_dual_fmac_f32 v49, v56, v86
	v_dual_fmac_f32 v58, v66, v86 :: v_dual_fmac_f32 v59, v76, v86
	v_dual_fmac_f32 v68, v46, v94 :: v_dual_fmac_f32 v69, v56, v94
	v_dual_fmac_f32 v78, v66, v94 :: v_dual_fmac_f32 v79, v76, v94
	v_dual_fmac_f32 v48, v47, v87 :: v_dual_fmac_f32 v49, v57, v87
	v_dual_fmac_f32 v58, v67, v87 :: v_dual_fmac_f32 v59, v77, v87
	v_dual_fmac_f32 v68, v47, v95 :: v_dual_fmac_f32 v69, v57, v95
	v_dual_fmac_f32 v78, v67, v95 :: v_dual_fmac_f32 v79, v77, v95
	v_dual_add_f32 v148, v148, v48 :: v_dual_add_f32 v149, v149, v49
	v_dual_add_f32 v150, v150, v58 :: v_dual_add_f32 v151, v151, v59
	v_dual_add_f32 v164, v164, v68 :: v_dual_add_f32 v165, v165, v69
	v_dual_add_f32 v166, v166, v78 :: v_dual_add_f32 v167, v167, v79
	s_wait_loadcnt 0x9
	v_dual_mul_f32 v48, v41, v97 :: v_dual_mul_f32 v49, v51, v97
	v_dual_mul_f32 v58, v61, v97 :: v_dual_mul_f32 v59, v71, v97
	s_wait_loadcnt 0x7
	v_dual_mul_f32 v68, v41, v105 :: v_dual_mul_f32 v69, v51, v105
	v_dual_mul_f32 v78, v61, v105 :: v_dual_mul_f32 v79, v71, v105
	v_dual_fmac_f32 v48, v40, v96 :: v_dual_fmac_f32 v49, v50, v96
	v_dual_fmac_f32 v58, v60, v96 :: v_dual_fmac_f32 v59, v70, v96
	v_dual_fmac_f32 v68, v40, v104 :: v_dual_fmac_f32 v69, v50, v104
	v_dual_fmac_f32 v78, v60, v104 :: v_dual_fmac_f32 v79, v70, v104
	v_dual_fmac_f32 v48, v42, v98 :: v_dual_fmac_f32 v49, v52, v98
	v_dual_fmac_f32 v58, v62, v98 :: v_dual_fmac_f32 v59, v72, v98
	v_dual_fmac_f32 v68, v42, v106 :: v_dual_fmac_f32 v69, v52, v106
	v_dual_fmac_f32 v78, v62, v106 :: v_dual_fmac_f32 v79, v72, v106
	v_dual_fmac_f32 v48, v43, v99 :: v_dual_fmac_f32 v49, v53, v99
	v_dual_fmac_f32 v58, v63, v99 :: v_dual_fmac_f32 v59, v73, v99
	v_dual_fmac_f32 v68, v43, v107 :: v_dual_fmac_f32 v69, v53, v107
	v_dual_fmac_f32 v78, v63, v107 :: v_dual_fmac_f32 v79, v73, v107
	v_dual_fmac_f32 v48, v44, v100 :: v_dual_fmac_f32 v49, v54, v100
	v_dual_fmac_f32 v58, v64, v100 :: v_dual_fmac_f32 v59, v74, v100
	s_wait_loadcnt 0x6
	v_dual_fmac_f32 v68, v44, v108 :: v_dual_fmac_f32 v69, v54, v108
	v_dual_fmac_f32 v78, v64, v108 :: v_dual_fmac_f32 v79, v74, v108
	v_dual_fmac_f32 v48, v45, v101 :: v_dual_fmac_f32 v49, v55, v101
	v_dual_fmac_f32 v58, v65, v101 :: v_dual_fmac_f32 v59, v75, v101
	v_dual_fmac_f32 v68, v45, v109 :: v_dual_fmac_f32 v69, v55, v109
	v_dual_fmac_f32 v78, v65, v109 :: v_dual_fmac_f32 v79, v75, v109
	v_dual_fmac_f32 v48, v46, v102 :: v_dual_fmac_f32 v49, v56, v102
	v_dual_fmac_f32 v58, v66, v102 :: v_dual_fmac_f32 v59, v76, v102
	v_dual_fmac_f32 v68, v46, v110 :: v_dual_fmac_f32 v69, v56, v110
	v_dual_fmac_f32 v78, v66, v110 :: v_dual_fmac_f32 v79, v76, v110
	v_dual_fmac_f32 v48, v47, v103 :: v_dual_fmac_f32 v49, v57, v103
	v_dual_fmac_f32 v58, v67, v103 :: v_dual_fmac_f32 v59, v77, v103
	v_dual_fmac_f32 v68, v47, v111 :: v_dual_fmac_f32 v69, v57, v111
	v_dual_fmac_f32 v78, v67, v111 :: v_dual_fmac_f32 v79, v77, v111
	v_dual_add_f32 v180, v180, v48 :: v_dual_add_f32 v181, v181, v49
	v_dual_add_f32 v182, v182, v58 :: v_dual_add_f32 v183, v183, v59
	v_dual_add_f32 v196, v196, v68 :: v_dual_add_f32 v197, v197, v69
	v_dual_add_f32 v198, v198, v78 :: v_dual_add_f32 v199, v199, v79
	s_wait_loadcnt 0x5
	v_dual_mul_f32 v48, v41, v113 :: v_dual_mul_f32 v49, v51, v113
	v_dual_mul_f32 v58, v61, v113 :: v_dual_mul_f32 v59, v71, v113
	s_wait_loadcnt 0x3
	v_dual_mul_f32 v68, v41, v121 :: v_dual_mul_f32 v69, v51, v121
	v_dual_mul_f32 v78, v61, v121 :: v_dual_mul_f32 v79, v71, v121
	v_dual_fmac_f32 v48, v40, v112 :: v_dual_fmac_f32 v49, v50, v112
	v_dual_fmac_f32 v58, v60, v112 :: v_dual_fmac_f32 v59, v70, v112
	v_dual_fmac_f32 v68, v40, v120 :: v_dual_fmac_f32 v69, v50, v120
	v_dual_fmac_f32 v78, v60, v120 :: v_dual_fmac_f32 v79, v70, v120
	v_dual_fmac_f32 v48, v42, v114 :: v_dual_fmac_f32 v49, v52, v114
	v_dual_fmac_f32 v58, v62, v114 :: v_dual_fmac_f32 v59, v72, v114
	v_dual_fmac_f32 v68, v42, v122 :: v_dual_fmac_f32 v69, v52, v122
	v_dual_fmac_f32 v78, v62, v122 :: v_dual_fmac_f32 v79, v72, v122
	v_dual_fmac_f32 v48, v43, v115 :: v_dual_fmac_f32 v49, v53, v115
	v_dual_fmac_f32 v58, v63, v115 :: v_dual_fmac_f32 v59, v73, v115
	v_dual_fmac_f32 v68, v43, v123 :: v_dual_fmac_f32 v69, v53, v123
	v_dual_fmac_f32 v78, v63, v123 :: v_dual_fmac_f32 v79, v73, v123
	v_dual_fmac_f32 v48, v44, v116 :: v_dual_fmac_f32 v49, v54, v116
	v_dual_fmac_f32 v58, v64, v116 :: v_dual_fmac_f32 v59, v74, v116
	s_wait_loadcnt 0x2
	v_dual_fmac_f32 v68, v44, v124 :: v_dual_fmac_f32 v69, v54, v124
	v_dual_fmac_f32 v78, v64, v124 :: v_dual_fmac_f32 v79, v74, v124
	v_dual_fmac_f32 v48, v45, v117 :: v_dual_fmac_f32 v49, v55, v117
	v_dual_fmac_f32 v58, v65, v117 :: v_dual_fmac_f32 v59, v75, v117
	v_dual_fmac_f32 v68, v45, v125 :: v_dual_fmac_f32 v69, v55, v125
	v_dual_fmac_f32 v78, v65, v125 :: v_dual_fmac_f32 v79, v75, v125
	v_dual_fmac_f32 v48, v46, v118 :: v_dual_fmac_f32 v49, v56, v118
	v_dual_fmac_f32 v58, v66, v118 :: v_dual_fmac_f32 v59, v76, v118
	v_dual_fmac_f32 v68, v46, v126 :: v_dual_fmac_f32 v69, v56, v126
	v_dual_fmac_f32 v78, v66, v126 :: v_dual_fmac_f32 v79, v76, v126
	v_dual_fmac_f32 v48, v47, v119 :: v_dual_fmac_f32 v49, v57, v119
	v_dual_fmac_f32 v58, v67, v119 :: v_dual_fmac_f32 v59, v77, v119
	v_dual_fmac_f32 v68, v47, v127 :: v_dual_fmac_f32 v69, v57, v127
	v_dual_fmac_f32 v78, v67, v127 :: v_dual_fmac_f32 v79, v77, v127
	v_dual_add_f32 v212, v212, v48 :: v_dual_add_f32 v213, v213, v49
	v_dual_add_f32 v214, v214, v58 :: v_dual_add_f32 v215, v215, v59
	v_dual_add_f32 v228, v228, v68 :: v_dual_add_f32 v229, v229, v69
	v_dual_add_f32 v230, v230, v78 :: v_dual_add_f32 v231, v231, v79
	s_wait_loadcnt 0x1
	v_dual_mul_f32 v48, v41, v129 :: v_dual_mul_f32 v49, v51, v129
	v_dual_mul_f32 v58, v61, v129 :: v_dual_mul_f32 v59, v71, v129
	v_dual_fmac_f32 v48, v40, v128 :: v_dual_fmac_f32 v49, v50, v128
	v_dual_fmac_f32 v58, v60, v128 :: v_dual_fmac_f32 v59, v70, v128
	v_dual_fmac_f32 v48, v42, v130 :: v_dual_fmac_f32 v49, v52, v130
	v_dual_fmac_f32 v58, v62, v130 :: v_dual_fmac_f32 v59, v72, v130
	v_dual_fmac_f32 v48, v43, v131 :: v_dual_fmac_f32 v49, v53, v131
	v_dual_fmac_f32 v58, v63, v131 :: v_dual_fmac_f32 v59, v73, v131
	s_wait_loadcnt 0x0
	v_dual_fmac_f32 v48, v44, v132 :: v_dual_fmac_f32 v49, v54, v132
	v_dual_fmac_f32 v58, v64, v132 :: v_dual_fmac_f32 v59, v74, v132
	v_dual_fmac_f32 v48, v45, v133 :: v_dual_fmac_f32 v49, v55, v133
	v_dual_fmac_f32 v58, v65, v133 :: v_dual_fmac_f32 v59, v75, v133
	v_dual_fmac_f32 v48, v46, v134 :: v_dual_fmac_f32 v49, v56, v134
	v_dual_fmac_f32 v58, v66, v134 :: v_dual_fmac_f32 v59, v76, v134
	v_dual_fmac_f32 v48, v47, v135 :: v_dual_fmac_f32 v49, v57, v135
	v_dual_fmac_f32 v58, v67, v135 :: v_dual_fmac_f32 v59, v77, v135
	v_dual_add_f32 v244, v244, v48 :: v_dual_add_f32 v245, v245, v49
	v_dual_add_f32 v246, v246, v58 :: v_dual_add_f32 v247, v247, v59
	s_cmp_gt_u32 s37, 2
	s_cbranch_scc0 .Lpx_b7_fold
	buffer_load_b32 v16, v1, s[20:23], s16 offen offset:272
	buffer_load_b32 v17, v1, s[20:23], s17 offen offset:272
	buffer_load_b32 v18, v1, s[20:23], s18 offen offset:272
	buffer_load_b32 v19, v1, s[20:23], s19 offen offset:272
	buffer_load_b32 v32, v2, s[20:23], s16 offen offset:280
	buffer_load_b32 v33, v2, s[20:23], s17 offen offset:280
	buffer_load_b32 v34, v2, s[20:23], s18 offen offset:280
	buffer_load_b32 v35, v2, s[20:23], s19 offen offset:280
	buffer_load_b128 v[80:83], v3, s[24:27], s28 offen offset:2048
	buffer_load_b128 v[84:87], v3, s[24:27], s28 offen offset:2064
	buffer_load_b128 v[88:91], v3, s[24:27], s29 offen offset:2048
	buffer_load_b128 v[92:95], v3, s[24:27], s29 offen offset:2064
	buffer_load_b128 v[96:99], v3, s[24:27], s30 offen offset:2048
	buffer_load_b128 v[100:103], v3, s[24:27], s30 offen offset:2064
	buffer_load_b128 v[104:107], v3, s[24:27], s31 offen offset:2048
	buffer_load_b128 v[108:111], v3, s[24:27], s31 offen offset:2064
	buffer_load_b128 v[112:115], v3, s[24:27], s32 offen offset:2048
	buffer_load_b128 v[116:119], v3, s[24:27], s32 offen offset:2064
	buffer_load_b128 v[120:123], v3, s[24:27], s33 offen offset:2048
	buffer_load_b128 v[124:127], v3, s[24:27], s33 offen offset:2064
	buffer_load_b128 v[128:131], v3, s[24:27], s34 offen offset:2048
	buffer_load_b128 v[132:135], v3, s[24:27], s34 offen offset:2064
	s_wait_loadcnt 0x11
	v_and_b32_e32 v46, 0xf0f0f0f, v32
	s_wait_loadcnt 0x10
	v_and_b32_e32 v56, 0xf0f0f0f, v33
	s_wait_loadcnt 0xf
	v_and_b32_e32 v66, 0xf0f0f0f, v34
	s_wait_loadcnt 0xe
	v_and_b32_e32 v76, 0xf0f0f0f, v35
	v_lshrrev_b32_e32 v47, 4, v32
	v_lshrrev_b32_e32 v57, 4, v33
	v_lshrrev_b32_e32 v67, 4, v34
	v_lshrrev_b32_e32 v77, 4, v35
	v_and_b32_e32 v47, 0xf0f0f0f, v47
	v_and_b32_e32 v57, 0xf0f0f0f, v57
	v_and_b32_e32 v67, 0xf0f0f0f, v67
	v_and_b32_e32 v77, 0xf0f0f0f, v77
	v_cvt_f32_ubyte0_e32 v40, v46
	v_cvt_f32_ubyte0_e32 v50, v56
	v_cvt_f32_ubyte0_e32 v60, v66
	v_cvt_f32_ubyte0_e32 v70, v76
	v_cvt_f32_ubyte1_e32 v42, v46
	v_cvt_f32_ubyte1_e32 v52, v56
	v_cvt_f32_ubyte1_e32 v62, v66
	v_cvt_f32_ubyte1_e32 v72, v76
	v_cvt_f32_ubyte2_e32 v44, v46
	v_cvt_f32_ubyte2_e32 v54, v56
	v_cvt_f32_ubyte2_e32 v64, v66
	v_cvt_f32_ubyte2_e32 v74, v76
	v_cvt_f32_ubyte3_e32 v46, v46
	v_cvt_f32_ubyte3_e32 v56, v56
	v_cvt_f32_ubyte3_e32 v66, v66
	v_cvt_f32_ubyte3_e32 v76, v76
	v_cvt_f32_ubyte0_e32 v41, v47
	v_cvt_f32_ubyte0_e32 v51, v57
	v_cvt_f32_ubyte0_e32 v61, v67
	v_cvt_f32_ubyte0_e32 v71, v77
	v_cvt_f32_ubyte1_e32 v43, v47
	v_cvt_f32_ubyte1_e32 v53, v57
	v_cvt_f32_ubyte1_e32 v63, v67
	v_cvt_f32_ubyte1_e32 v73, v77
	v_cvt_f32_ubyte2_e32 v45, v47
	v_cvt_f32_ubyte2_e32 v55, v57
	v_cvt_f32_ubyte2_e32 v65, v67
	v_cvt_f32_ubyte2_e32 v75, v77
	v_cvt_f32_ubyte3_e32 v47, v47
	v_cvt_f32_ubyte3_e32 v57, v57
	v_cvt_f32_ubyte3_e32 v67, v67
	v_cvt_f32_ubyte3_e32 v77, v77
	v_fma_mix_f32 v40, v16, v40, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v17, v50, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v18, v60, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v70, v19, v70, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v41, v16, v41, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v17, v51, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v18, v61, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v71, v19, v71, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v42, v16, v42, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v52, v17, v52, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v62, v18, v62, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v72, v19, v72, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v43, v16, v43, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v53, v17, v53, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v63, v18, v63, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v73, v19, v73, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v44, v16, v44, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v54, v17, v54, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v64, v18, v64, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v74, v19, v74, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v16, v45, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v17, v55, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v65, v18, v65, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v75, v19, v75, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v16, v46, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v17, v56, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v66, v18, v66, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v76, v19, v76, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v16, v47, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v17, v57, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v67, v18, v67, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v77, v19, v77, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xd
	v_dual_mul_f32 v48, v41, v81 :: v_dual_mul_f32 v49, v51, v81
	v_dual_mul_f32 v58, v61, v81 :: v_dual_mul_f32 v59, v71, v81
	s_wait_loadcnt 0xb
	v_dual_mul_f32 v68, v41, v89 :: v_dual_mul_f32 v69, v51, v89
	v_dual_mul_f32 v78, v61, v89 :: v_dual_mul_f32 v79, v71, v89
	v_dual_fmac_f32 v48, v40, v80 :: v_dual_fmac_f32 v49, v50, v80
	v_dual_fmac_f32 v58, v60, v80 :: v_dual_fmac_f32 v59, v70, v80
	v_dual_fmac_f32 v68, v40, v88 :: v_dual_fmac_f32 v69, v50, v88
	v_dual_fmac_f32 v78, v60, v88 :: v_dual_fmac_f32 v79, v70, v88
	v_dual_fmac_f32 v48, v42, v82 :: v_dual_fmac_f32 v49, v52, v82
	v_dual_fmac_f32 v58, v62, v82 :: v_dual_fmac_f32 v59, v72, v82
	v_dual_fmac_f32 v68, v42, v90 :: v_dual_fmac_f32 v69, v52, v90
	v_dual_fmac_f32 v78, v62, v90 :: v_dual_fmac_f32 v79, v72, v90
	v_dual_fmac_f32 v48, v43, v83 :: v_dual_fmac_f32 v49, v53, v83
	v_dual_fmac_f32 v58, v63, v83 :: v_dual_fmac_f32 v59, v73, v83
	v_dual_fmac_f32 v68, v43, v91 :: v_dual_fmac_f32 v69, v53, v91
	v_dual_fmac_f32 v78, v63, v91 :: v_dual_fmac_f32 v79, v73, v91
	v_dual_fmac_f32 v48, v44, v84 :: v_dual_fmac_f32 v49, v54, v84
	v_dual_fmac_f32 v58, v64, v84 :: v_dual_fmac_f32 v59, v74, v84
	s_wait_loadcnt 0xa
	v_dual_fmac_f32 v68, v44, v92 :: v_dual_fmac_f32 v69, v54, v92
	v_dual_fmac_f32 v78, v64, v92 :: v_dual_fmac_f32 v79, v74, v92
	v_dual_fmac_f32 v48, v45, v85 :: v_dual_fmac_f32 v49, v55, v85
	v_dual_fmac_f32 v58, v65, v85 :: v_dual_fmac_f32 v59, v75, v85
	v_dual_fmac_f32 v68, v45, v93 :: v_dual_fmac_f32 v69, v55, v93
	v_dual_fmac_f32 v78, v65, v93 :: v_dual_fmac_f32 v79, v75, v93
	v_dual_fmac_f32 v48, v46, v86 :: v_dual_fmac_f32 v49, v56, v86
	v_dual_fmac_f32 v58, v66, v86 :: v_dual_fmac_f32 v59, v76, v86
	v_dual_fmac_f32 v68, v46, v94 :: v_dual_fmac_f32 v69, v56, v94
	v_dual_fmac_f32 v78, v66, v94 :: v_dual_fmac_f32 v79, v76, v94
	v_dual_fmac_f32 v48, v47, v87 :: v_dual_fmac_f32 v49, v57, v87
	v_dual_fmac_f32 v58, v67, v87 :: v_dual_fmac_f32 v59, v77, v87
	v_dual_fmac_f32 v68, v47, v95 :: v_dual_fmac_f32 v69, v57, v95
	v_dual_fmac_f32 v78, v67, v95 :: v_dual_fmac_f32 v79, v77, v95
	v_dual_add_f32 v152, v152, v48 :: v_dual_add_f32 v153, v153, v49
	v_dual_add_f32 v154, v154, v58 :: v_dual_add_f32 v155, v155, v59
	v_dual_add_f32 v168, v168, v68 :: v_dual_add_f32 v169, v169, v69
	v_dual_add_f32 v170, v170, v78 :: v_dual_add_f32 v171, v171, v79
	s_wait_loadcnt 0x9
	v_dual_mul_f32 v48, v41, v97 :: v_dual_mul_f32 v49, v51, v97
	v_dual_mul_f32 v58, v61, v97 :: v_dual_mul_f32 v59, v71, v97
	s_wait_loadcnt 0x7
	v_dual_mul_f32 v68, v41, v105 :: v_dual_mul_f32 v69, v51, v105
	v_dual_mul_f32 v78, v61, v105 :: v_dual_mul_f32 v79, v71, v105
	v_dual_fmac_f32 v48, v40, v96 :: v_dual_fmac_f32 v49, v50, v96
	v_dual_fmac_f32 v58, v60, v96 :: v_dual_fmac_f32 v59, v70, v96
	v_dual_fmac_f32 v68, v40, v104 :: v_dual_fmac_f32 v69, v50, v104
	v_dual_fmac_f32 v78, v60, v104 :: v_dual_fmac_f32 v79, v70, v104
	v_dual_fmac_f32 v48, v42, v98 :: v_dual_fmac_f32 v49, v52, v98
	v_dual_fmac_f32 v58, v62, v98 :: v_dual_fmac_f32 v59, v72, v98
	v_dual_fmac_f32 v68, v42, v106 :: v_dual_fmac_f32 v69, v52, v106
	v_dual_fmac_f32 v78, v62, v106 :: v_dual_fmac_f32 v79, v72, v106
	v_dual_fmac_f32 v48, v43, v99 :: v_dual_fmac_f32 v49, v53, v99
	v_dual_fmac_f32 v58, v63, v99 :: v_dual_fmac_f32 v59, v73, v99
	v_dual_fmac_f32 v68, v43, v107 :: v_dual_fmac_f32 v69, v53, v107
	v_dual_fmac_f32 v78, v63, v107 :: v_dual_fmac_f32 v79, v73, v107
	v_dual_fmac_f32 v48, v44, v100 :: v_dual_fmac_f32 v49, v54, v100
	v_dual_fmac_f32 v58, v64, v100 :: v_dual_fmac_f32 v59, v74, v100
	s_wait_loadcnt 0x6
	v_dual_fmac_f32 v68, v44, v108 :: v_dual_fmac_f32 v69, v54, v108
	v_dual_fmac_f32 v78, v64, v108 :: v_dual_fmac_f32 v79, v74, v108
	v_dual_fmac_f32 v48, v45, v101 :: v_dual_fmac_f32 v49, v55, v101
	v_dual_fmac_f32 v58, v65, v101 :: v_dual_fmac_f32 v59, v75, v101
	v_dual_fmac_f32 v68, v45, v109 :: v_dual_fmac_f32 v69, v55, v109
	v_dual_fmac_f32 v78, v65, v109 :: v_dual_fmac_f32 v79, v75, v109
	v_dual_fmac_f32 v48, v46, v102 :: v_dual_fmac_f32 v49, v56, v102
	v_dual_fmac_f32 v58, v66, v102 :: v_dual_fmac_f32 v59, v76, v102
	v_dual_fmac_f32 v68, v46, v110 :: v_dual_fmac_f32 v69, v56, v110
	v_dual_fmac_f32 v78, v66, v110 :: v_dual_fmac_f32 v79, v76, v110
	v_dual_fmac_f32 v48, v47, v103 :: v_dual_fmac_f32 v49, v57, v103
	v_dual_fmac_f32 v58, v67, v103 :: v_dual_fmac_f32 v59, v77, v103
	v_dual_fmac_f32 v68, v47, v111 :: v_dual_fmac_f32 v69, v57, v111
	v_dual_fmac_f32 v78, v67, v111 :: v_dual_fmac_f32 v79, v77, v111
	v_dual_add_f32 v184, v184, v48 :: v_dual_add_f32 v185, v185, v49
	v_dual_add_f32 v186, v186, v58 :: v_dual_add_f32 v187, v187, v59
	v_dual_add_f32 v200, v200, v68 :: v_dual_add_f32 v201, v201, v69
	v_dual_add_f32 v202, v202, v78 :: v_dual_add_f32 v203, v203, v79
	s_wait_loadcnt 0x5
	v_dual_mul_f32 v48, v41, v113 :: v_dual_mul_f32 v49, v51, v113
	v_dual_mul_f32 v58, v61, v113 :: v_dual_mul_f32 v59, v71, v113
	s_wait_loadcnt 0x3
	v_dual_mul_f32 v68, v41, v121 :: v_dual_mul_f32 v69, v51, v121
	v_dual_mul_f32 v78, v61, v121 :: v_dual_mul_f32 v79, v71, v121
	v_dual_fmac_f32 v48, v40, v112 :: v_dual_fmac_f32 v49, v50, v112
	v_dual_fmac_f32 v58, v60, v112 :: v_dual_fmac_f32 v59, v70, v112
	v_dual_fmac_f32 v68, v40, v120 :: v_dual_fmac_f32 v69, v50, v120
	v_dual_fmac_f32 v78, v60, v120 :: v_dual_fmac_f32 v79, v70, v120
	v_dual_fmac_f32 v48, v42, v114 :: v_dual_fmac_f32 v49, v52, v114
	v_dual_fmac_f32 v58, v62, v114 :: v_dual_fmac_f32 v59, v72, v114
	v_dual_fmac_f32 v68, v42, v122 :: v_dual_fmac_f32 v69, v52, v122
	v_dual_fmac_f32 v78, v62, v122 :: v_dual_fmac_f32 v79, v72, v122
	v_dual_fmac_f32 v48, v43, v115 :: v_dual_fmac_f32 v49, v53, v115
	v_dual_fmac_f32 v58, v63, v115 :: v_dual_fmac_f32 v59, v73, v115
	v_dual_fmac_f32 v68, v43, v123 :: v_dual_fmac_f32 v69, v53, v123
	v_dual_fmac_f32 v78, v63, v123 :: v_dual_fmac_f32 v79, v73, v123
	v_dual_fmac_f32 v48, v44, v116 :: v_dual_fmac_f32 v49, v54, v116
	v_dual_fmac_f32 v58, v64, v116 :: v_dual_fmac_f32 v59, v74, v116
	s_wait_loadcnt 0x2
	v_dual_fmac_f32 v68, v44, v124 :: v_dual_fmac_f32 v69, v54, v124
	v_dual_fmac_f32 v78, v64, v124 :: v_dual_fmac_f32 v79, v74, v124
	v_dual_fmac_f32 v48, v45, v117 :: v_dual_fmac_f32 v49, v55, v117
	v_dual_fmac_f32 v58, v65, v117 :: v_dual_fmac_f32 v59, v75, v117
	v_dual_fmac_f32 v68, v45, v125 :: v_dual_fmac_f32 v69, v55, v125
	v_dual_fmac_f32 v78, v65, v125 :: v_dual_fmac_f32 v79, v75, v125
	v_dual_fmac_f32 v48, v46, v118 :: v_dual_fmac_f32 v49, v56, v118
	v_dual_fmac_f32 v58, v66, v118 :: v_dual_fmac_f32 v59, v76, v118
	v_dual_fmac_f32 v68, v46, v126 :: v_dual_fmac_f32 v69, v56, v126
	v_dual_fmac_f32 v78, v66, v126 :: v_dual_fmac_f32 v79, v76, v126
	v_dual_fmac_f32 v48, v47, v119 :: v_dual_fmac_f32 v49, v57, v119
	v_dual_fmac_f32 v58, v67, v119 :: v_dual_fmac_f32 v59, v77, v119
	v_dual_fmac_f32 v68, v47, v127 :: v_dual_fmac_f32 v69, v57, v127
	v_dual_fmac_f32 v78, v67, v127 :: v_dual_fmac_f32 v79, v77, v127
	v_dual_add_f32 v216, v216, v48 :: v_dual_add_f32 v217, v217, v49
	v_dual_add_f32 v218, v218, v58 :: v_dual_add_f32 v219, v219, v59
	v_dual_add_f32 v232, v232, v68 :: v_dual_add_f32 v233, v233, v69
	v_dual_add_f32 v234, v234, v78 :: v_dual_add_f32 v235, v235, v79
	s_wait_loadcnt 0x1
	v_dual_mul_f32 v48, v41, v129 :: v_dual_mul_f32 v49, v51, v129
	v_dual_mul_f32 v58, v61, v129 :: v_dual_mul_f32 v59, v71, v129
	v_dual_fmac_f32 v48, v40, v128 :: v_dual_fmac_f32 v49, v50, v128
	v_dual_fmac_f32 v58, v60, v128 :: v_dual_fmac_f32 v59, v70, v128
	v_dual_fmac_f32 v48, v42, v130 :: v_dual_fmac_f32 v49, v52, v130
	v_dual_fmac_f32 v58, v62, v130 :: v_dual_fmac_f32 v59, v72, v130
	v_dual_fmac_f32 v48, v43, v131 :: v_dual_fmac_f32 v49, v53, v131
	v_dual_fmac_f32 v58, v63, v131 :: v_dual_fmac_f32 v59, v73, v131
	s_wait_loadcnt 0x0
	v_dual_fmac_f32 v48, v44, v132 :: v_dual_fmac_f32 v49, v54, v132
	v_dual_fmac_f32 v58, v64, v132 :: v_dual_fmac_f32 v59, v74, v132
	v_dual_fmac_f32 v48, v45, v133 :: v_dual_fmac_f32 v49, v55, v133
	v_dual_fmac_f32 v58, v65, v133 :: v_dual_fmac_f32 v59, v75, v133
	v_dual_fmac_f32 v48, v46, v134 :: v_dual_fmac_f32 v49, v56, v134
	v_dual_fmac_f32 v58, v66, v134 :: v_dual_fmac_f32 v59, v76, v134
	v_dual_fmac_f32 v48, v47, v135 :: v_dual_fmac_f32 v49, v57, v135
	v_dual_fmac_f32 v58, v67, v135 :: v_dual_fmac_f32 v59, v77, v135
	v_dual_add_f32 v248, v248, v48 :: v_dual_add_f32 v249, v249, v49
	v_dual_add_f32 v250, v250, v58 :: v_dual_add_f32 v251, v251, v59
	.Lpx_b7_fold:
	v_dual_add_f32 v144, v144, v148 :: v_dual_add_f32 v145, v145, v149
	v_dual_add_f32 v146, v146, v150 :: v_dual_add_f32 v147, v147, v151
	v_dual_add_f32 v152, v152, v156 :: v_dual_add_f32 v153, v153, v157
	v_dual_add_f32 v154, v154, v158 :: v_dual_add_f32 v155, v155, v159
	v_dual_add_f32 v144, v144, v152 :: v_dual_add_f32 v145, v145, v153
	v_dual_add_f32 v146, v146, v154 :: v_dual_add_f32 v147, v147, v155
	v_dual_add_f32 v160, v160, v164 :: v_dual_add_f32 v161, v161, v165
	v_dual_add_f32 v162, v162, v166 :: v_dual_add_f32 v163, v163, v167
	v_dual_add_f32 v168, v168, v172 :: v_dual_add_f32 v169, v169, v173
	v_dual_add_f32 v170, v170, v174 :: v_dual_add_f32 v171, v171, v175
	v_dual_add_f32 v160, v160, v168 :: v_dual_add_f32 v161, v161, v169
	v_dual_add_f32 v162, v162, v170 :: v_dual_add_f32 v163, v163, v171
	v_dual_add_f32 v176, v176, v180 :: v_dual_add_f32 v177, v177, v181
	v_dual_add_f32 v178, v178, v182 :: v_dual_add_f32 v179, v179, v183
	v_dual_add_f32 v184, v184, v188 :: v_dual_add_f32 v185, v185, v189
	v_dual_add_f32 v186, v186, v190 :: v_dual_add_f32 v187, v187, v191
	v_dual_add_f32 v176, v176, v184 :: v_dual_add_f32 v177, v177, v185
	v_dual_add_f32 v178, v178, v186 :: v_dual_add_f32 v179, v179, v187
	v_dual_add_f32 v192, v192, v196 :: v_dual_add_f32 v193, v193, v197
	v_dual_add_f32 v194, v194, v198 :: v_dual_add_f32 v195, v195, v199
	v_dual_add_f32 v200, v200, v204 :: v_dual_add_f32 v201, v201, v205
	v_dual_add_f32 v202, v202, v206 :: v_dual_add_f32 v203, v203, v207
	v_dual_add_f32 v192, v192, v200 :: v_dual_add_f32 v193, v193, v201
	v_dual_add_f32 v194, v194, v202 :: v_dual_add_f32 v195, v195, v203
	v_dual_add_f32 v208, v208, v212 :: v_dual_add_f32 v209, v209, v213
	v_dual_add_f32 v210, v210, v214 :: v_dual_add_f32 v211, v211, v215
	v_dual_add_f32 v216, v216, v220 :: v_dual_add_f32 v217, v217, v221
	v_dual_add_f32 v218, v218, v222 :: v_dual_add_f32 v219, v219, v223
	v_dual_add_f32 v208, v208, v216 :: v_dual_add_f32 v209, v209, v217
	v_dual_add_f32 v210, v210, v218 :: v_dual_add_f32 v211, v211, v219
	v_dual_add_f32 v224, v224, v228 :: v_dual_add_f32 v225, v225, v229
	v_dual_add_f32 v226, v226, v230 :: v_dual_add_f32 v227, v227, v231
	v_dual_add_f32 v232, v232, v236 :: v_dual_add_f32 v233, v233, v237
	v_dual_add_f32 v234, v234, v238 :: v_dual_add_f32 v235, v235, v239
	v_dual_add_f32 v224, v224, v232 :: v_dual_add_f32 v225, v225, v233
	v_dual_add_f32 v226, v226, v234 :: v_dual_add_f32 v227, v227, v235
	v_dual_add_f32 v240, v240, v244 :: v_dual_add_f32 v241, v241, v245
	v_dual_add_f32 v242, v242, v246 :: v_dual_add_f32 v243, v243, v247
	v_dual_add_f32 v248, v248, v252 :: v_dual_add_f32 v249, v249, v253
	v_dual_add_f32 v250, v250, v254 :: v_dual_add_f32 v251, v251, v255
	v_dual_add_f32 v240, v240, v248 :: v_dual_add_f32 v241, v241, v249
	v_dual_add_f32 v242, v242, v250 :: v_dual_add_f32 v243, v243, v251
	ds_swizzle_b32 v8, v144 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v9, v145 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v10, v146 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v11, v147 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v12, v160 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v13, v161 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v14, v162 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v15, v163 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v16, v176 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v17, v177 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v18, v178 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v19, v179 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v20, v192 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v21, v193 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v22, v194 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v23, v195 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v24, v208 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v25, v209 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v26, v210 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v27, v211 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v28, v224 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v29, v225 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v30, v226 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v31, v227 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v32, v240 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v33, v241 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v34, v242 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v35, v243 offset:swizzle(BITMASK_PERM,"1pppp")
	s_wait_dscnt 0x1b
	v_add_f32_e32 v144, v144, v8
	s_wait_dscnt 0x1a
	v_add_f32_e32 v145, v145, v9
	s_wait_dscnt 0x19
	v_add_f32_e32 v146, v146, v10
	s_wait_dscnt 0x18
	v_add_f32_e32 v147, v147, v11
	s_wait_dscnt 0x17
	v_add_f32_e32 v160, v160, v12
	s_wait_dscnt 0x16
	v_add_f32_e32 v161, v161, v13
	s_wait_dscnt 0x15
	v_add_f32_e32 v162, v162, v14
	s_wait_dscnt 0x14
	v_add_f32_e32 v163, v163, v15
	s_wait_dscnt 0x13
	v_add_f32_e32 v176, v176, v16
	s_wait_dscnt 0x12
	v_add_f32_e32 v177, v177, v17
	s_wait_dscnt 0x11
	v_add_f32_e32 v178, v178, v18
	s_wait_dscnt 0x10
	v_add_f32_e32 v179, v179, v19
	s_wait_dscnt 0xf
	v_add_f32_e32 v192, v192, v20
	s_wait_dscnt 0xe
	v_add_f32_e32 v193, v193, v21
	s_wait_dscnt 0xd
	v_add_f32_e32 v194, v194, v22
	s_wait_dscnt 0xc
	v_add_f32_e32 v195, v195, v23
	s_wait_dscnt 0xb
	v_add_f32_e32 v208, v208, v24
	s_wait_dscnt 0xa
	v_add_f32_e32 v209, v209, v25
	s_wait_dscnt 0x9
	v_add_f32_e32 v210, v210, v26
	s_wait_dscnt 0x8
	v_add_f32_e32 v211, v211, v27
	s_wait_dscnt 0x7
	v_add_f32_e32 v224, v224, v28
	s_wait_dscnt 0x6
	v_add_f32_e32 v225, v225, v29
	s_wait_dscnt 0x5
	v_add_f32_e32 v226, v226, v30
	s_wait_dscnt 0x4
	v_add_f32_e32 v227, v227, v31
	s_wait_dscnt 0x3
	v_add_f32_e32 v240, v240, v32
	s_wait_dscnt 0x2
	v_add_f32_e32 v241, v241, v33
	s_wait_dscnt 0x1
	v_add_f32_e32 v242, v242, v34
	s_wait_dscnt 0x0
	v_add_f32_e32 v243, v243, v35
	v_cmp_gt_u32_e32 vcc_lo, 24, v0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v4, 0, 8, vcc_lo
	v_add_lshl_u32 v4, v4, v0, 2
	ds_bpermute_b32 v8, v4, v144
	ds_bpermute_b32 v9, v4, v145
	ds_bpermute_b32 v10, v4, v146
	ds_bpermute_b32 v11, v4, v147
	ds_bpermute_b32 v12, v4, v160
	ds_bpermute_b32 v13, v4, v161
	ds_bpermute_b32 v14, v4, v162
	ds_bpermute_b32 v15, v4, v163
	ds_bpermute_b32 v16, v4, v176
	ds_bpermute_b32 v17, v4, v177
	ds_bpermute_b32 v18, v4, v178
	ds_bpermute_b32 v19, v4, v179
	ds_bpermute_b32 v20, v4, v192
	ds_bpermute_b32 v21, v4, v193
	ds_bpermute_b32 v22, v4, v194
	ds_bpermute_b32 v23, v4, v195
	ds_bpermute_b32 v24, v4, v208
	ds_bpermute_b32 v25, v4, v209
	ds_bpermute_b32 v26, v4, v210
	ds_bpermute_b32 v27, v4, v211
	ds_bpermute_b32 v28, v4, v224
	ds_bpermute_b32 v29, v4, v225
	ds_bpermute_b32 v30, v4, v226
	ds_bpermute_b32 v31, v4, v227
	ds_bpermute_b32 v32, v4, v240
	ds_bpermute_b32 v33, v4, v241
	ds_bpermute_b32 v34, v4, v242
	ds_bpermute_b32 v35, v4, v243
	s_wait_dscnt 0x1b
	v_add_f32_e32 v144, v144, v8
	s_wait_dscnt 0x1a
	v_add_f32_e32 v145, v145, v9
	s_wait_dscnt 0x19
	v_add_f32_e32 v146, v146, v10
	s_wait_dscnt 0x18
	v_add_f32_e32 v147, v147, v11
	s_wait_dscnt 0x17
	v_add_f32_e32 v160, v160, v12
	s_wait_dscnt 0x16
	v_add_f32_e32 v161, v161, v13
	s_wait_dscnt 0x15
	v_add_f32_e32 v162, v162, v14
	s_wait_dscnt 0x14
	v_add_f32_e32 v163, v163, v15
	s_wait_dscnt 0x13
	v_add_f32_e32 v176, v176, v16
	s_wait_dscnt 0x12
	v_add_f32_e32 v177, v177, v17
	s_wait_dscnt 0x11
	v_add_f32_e32 v178, v178, v18
	s_wait_dscnt 0x10
	v_add_f32_e32 v179, v179, v19
	s_wait_dscnt 0xf
	v_add_f32_e32 v192, v192, v20
	s_wait_dscnt 0xe
	v_add_f32_e32 v193, v193, v21
	s_wait_dscnt 0xd
	v_add_f32_e32 v194, v194, v22
	s_wait_dscnt 0xc
	v_add_f32_e32 v195, v195, v23
	s_wait_dscnt 0xb
	v_add_f32_e32 v208, v208, v24
	s_wait_dscnt 0xa
	v_add_f32_e32 v209, v209, v25
	s_wait_dscnt 0x9
	v_add_f32_e32 v210, v210, v26
	s_wait_dscnt 0x8
	v_add_f32_e32 v211, v211, v27
	s_wait_dscnt 0x7
	v_add_f32_e32 v224, v224, v28
	s_wait_dscnt 0x6
	v_add_f32_e32 v225, v225, v29
	s_wait_dscnt 0x5
	v_add_f32_e32 v226, v226, v30
	s_wait_dscnt 0x4
	v_add_f32_e32 v227, v227, v31
	s_wait_dscnt 0x3
	v_add_f32_e32 v240, v240, v32
	s_wait_dscnt 0x2
	v_add_f32_e32 v241, v241, v33
	s_wait_dscnt 0x1
	v_add_f32_e32 v242, v242, v34
	s_wait_dscnt 0x0
	v_add_f32_e32 v243, v243, v35
	v_cmp_gt_u32_e32 vcc_lo, 28, v0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v4, 0, 4, vcc_lo
	v_add_lshl_u32 v4, v4, v0, 2
	ds_bpermute_b32 v8, v4, v144
	ds_bpermute_b32 v9, v4, v145
	ds_bpermute_b32 v10, v4, v146
	ds_bpermute_b32 v11, v4, v147
	ds_bpermute_b32 v12, v4, v160
	ds_bpermute_b32 v13, v4, v161
	ds_bpermute_b32 v14, v4, v162
	ds_bpermute_b32 v15, v4, v163
	ds_bpermute_b32 v16, v4, v176
	ds_bpermute_b32 v17, v4, v177
	ds_bpermute_b32 v18, v4, v178
	ds_bpermute_b32 v19, v4, v179
	ds_bpermute_b32 v20, v4, v192
	ds_bpermute_b32 v21, v4, v193
	ds_bpermute_b32 v22, v4, v194
	ds_bpermute_b32 v23, v4, v195
	ds_bpermute_b32 v24, v4, v208
	ds_bpermute_b32 v25, v4, v209
	ds_bpermute_b32 v26, v4, v210
	ds_bpermute_b32 v27, v4, v211
	ds_bpermute_b32 v28, v4, v224
	ds_bpermute_b32 v29, v4, v225
	ds_bpermute_b32 v30, v4, v226
	ds_bpermute_b32 v31, v4, v227
	ds_bpermute_b32 v32, v4, v240
	ds_bpermute_b32 v33, v4, v241
	ds_bpermute_b32 v34, v4, v242
	ds_bpermute_b32 v35, v4, v243
	s_wait_dscnt 0x1b
	v_add_f32_e32 v144, v144, v8
	s_wait_dscnt 0x1a
	v_add_f32_e32 v145, v145, v9
	s_wait_dscnt 0x19
	v_add_f32_e32 v146, v146, v10
	s_wait_dscnt 0x18
	v_add_f32_e32 v147, v147, v11
	s_wait_dscnt 0x17
	v_add_f32_e32 v160, v160, v12
	s_wait_dscnt 0x16
	v_add_f32_e32 v161, v161, v13
	s_wait_dscnt 0x15
	v_add_f32_e32 v162, v162, v14
	s_wait_dscnt 0x14
	v_add_f32_e32 v163, v163, v15
	s_wait_dscnt 0x13
	v_add_f32_e32 v176, v176, v16
	s_wait_dscnt 0x12
	v_add_f32_e32 v177, v177, v17
	s_wait_dscnt 0x11
	v_add_f32_e32 v178, v178, v18
	s_wait_dscnt 0x10
	v_add_f32_e32 v179, v179, v19
	s_wait_dscnt 0xf
	v_add_f32_e32 v192, v192, v20
	s_wait_dscnt 0xe
	v_add_f32_e32 v193, v193, v21
	s_wait_dscnt 0xd
	v_add_f32_e32 v194, v194, v22
	s_wait_dscnt 0xc
	v_add_f32_e32 v195, v195, v23
	s_wait_dscnt 0xb
	v_add_f32_e32 v208, v208, v24
	s_wait_dscnt 0xa
	v_add_f32_e32 v209, v209, v25
	s_wait_dscnt 0x9
	v_add_f32_e32 v210, v210, v26
	s_wait_dscnt 0x8
	v_add_f32_e32 v211, v211, v27
	s_wait_dscnt 0x7
	v_add_f32_e32 v224, v224, v28
	s_wait_dscnt 0x6
	v_add_f32_e32 v225, v225, v29
	s_wait_dscnt 0x5
	v_add_f32_e32 v226, v226, v30
	s_wait_dscnt 0x4
	v_add_f32_e32 v227, v227, v31
	s_wait_dscnt 0x3
	v_add_f32_e32 v240, v240, v32
	s_wait_dscnt 0x2
	v_add_f32_e32 v241, v241, v33
	s_wait_dscnt 0x1
	v_add_f32_e32 v242, v242, v34
	s_wait_dscnt 0x0
	v_add_f32_e32 v243, v243, v35
	v_cmp_gt_u32_e32 vcc_lo, 30, v0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v4, 0, 2, vcc_lo
	v_add_lshl_u32 v4, v4, v0, 2
	ds_bpermute_b32 v8, v4, v144
	ds_bpermute_b32 v9, v4, v145
	ds_bpermute_b32 v10, v4, v146
	ds_bpermute_b32 v11, v4, v147
	ds_bpermute_b32 v12, v4, v160
	ds_bpermute_b32 v13, v4, v161
	ds_bpermute_b32 v14, v4, v162
	ds_bpermute_b32 v15, v4, v163
	ds_bpermute_b32 v16, v4, v176
	ds_bpermute_b32 v17, v4, v177
	ds_bpermute_b32 v18, v4, v178
	ds_bpermute_b32 v19, v4, v179
	ds_bpermute_b32 v20, v4, v192
	ds_bpermute_b32 v21, v4, v193
	ds_bpermute_b32 v22, v4, v194
	ds_bpermute_b32 v23, v4, v195
	ds_bpermute_b32 v24, v4, v208
	ds_bpermute_b32 v25, v4, v209
	ds_bpermute_b32 v26, v4, v210
	ds_bpermute_b32 v27, v4, v211
	ds_bpermute_b32 v28, v4, v224
	ds_bpermute_b32 v29, v4, v225
	ds_bpermute_b32 v30, v4, v226
	ds_bpermute_b32 v31, v4, v227
	ds_bpermute_b32 v32, v4, v240
	ds_bpermute_b32 v33, v4, v241
	ds_bpermute_b32 v34, v4, v242
	ds_bpermute_b32 v35, v4, v243
	s_wait_dscnt 0x1b
	v_add_f32_e32 v144, v144, v8
	s_wait_dscnt 0x1a
	v_add_f32_e32 v145, v145, v9
	s_wait_dscnt 0x19
	v_add_f32_e32 v146, v146, v10
	s_wait_dscnt 0x18
	v_add_f32_e32 v147, v147, v11
	s_wait_dscnt 0x17
	v_add_f32_e32 v160, v160, v12
	s_wait_dscnt 0x16
	v_add_f32_e32 v161, v161, v13
	s_wait_dscnt 0x15
	v_add_f32_e32 v162, v162, v14
	s_wait_dscnt 0x14
	v_add_f32_e32 v163, v163, v15
	s_wait_dscnt 0x13
	v_add_f32_e32 v176, v176, v16
	s_wait_dscnt 0x12
	v_add_f32_e32 v177, v177, v17
	s_wait_dscnt 0x11
	v_add_f32_e32 v178, v178, v18
	s_wait_dscnt 0x10
	v_add_f32_e32 v179, v179, v19
	s_wait_dscnt 0xf
	v_add_f32_e32 v192, v192, v20
	s_wait_dscnt 0xe
	v_add_f32_e32 v193, v193, v21
	s_wait_dscnt 0xd
	v_add_f32_e32 v194, v194, v22
	s_wait_dscnt 0xc
	v_add_f32_e32 v195, v195, v23
	s_wait_dscnt 0xb
	v_add_f32_e32 v208, v208, v24
	s_wait_dscnt 0xa
	v_add_f32_e32 v209, v209, v25
	s_wait_dscnt 0x9
	v_add_f32_e32 v210, v210, v26
	s_wait_dscnt 0x8
	v_add_f32_e32 v211, v211, v27
	s_wait_dscnt 0x7
	v_add_f32_e32 v224, v224, v28
	s_wait_dscnt 0x6
	v_add_f32_e32 v225, v225, v29
	s_wait_dscnt 0x5
	v_add_f32_e32 v226, v226, v30
	s_wait_dscnt 0x4
	v_add_f32_e32 v227, v227, v31
	s_wait_dscnt 0x3
	v_add_f32_e32 v240, v240, v32
	s_wait_dscnt 0x2
	v_add_f32_e32 v241, v241, v33
	s_wait_dscnt 0x1
	v_add_f32_e32 v242, v242, v34
	s_wait_dscnt 0x0
	v_add_f32_e32 v243, v243, v35
	v_cmp_gt_u32_e32 vcc_lo, 31, v0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v4, 0, 1, vcc_lo
	v_add_lshl_u32 v4, v4, v0, 2
	ds_bpermute_b32 v8, v4, v144
	ds_bpermute_b32 v9, v4, v145
	ds_bpermute_b32 v10, v4, v146
	ds_bpermute_b32 v11, v4, v147
	ds_bpermute_b32 v12, v4, v160
	ds_bpermute_b32 v13, v4, v161
	ds_bpermute_b32 v14, v4, v162
	ds_bpermute_b32 v15, v4, v163
	ds_bpermute_b32 v16, v4, v176
	ds_bpermute_b32 v17, v4, v177
	ds_bpermute_b32 v18, v4, v178
	ds_bpermute_b32 v19, v4, v179
	ds_bpermute_b32 v20, v4, v192
	ds_bpermute_b32 v21, v4, v193
	ds_bpermute_b32 v22, v4, v194
	ds_bpermute_b32 v23, v4, v195
	ds_bpermute_b32 v24, v4, v208
	ds_bpermute_b32 v25, v4, v209
	ds_bpermute_b32 v26, v4, v210
	ds_bpermute_b32 v27, v4, v211
	ds_bpermute_b32 v28, v4, v224
	ds_bpermute_b32 v29, v4, v225
	ds_bpermute_b32 v30, v4, v226
	ds_bpermute_b32 v31, v4, v227
	ds_bpermute_b32 v32, v4, v240
	ds_bpermute_b32 v33, v4, v241
	ds_bpermute_b32 v34, v4, v242
	ds_bpermute_b32 v35, v4, v243
	s_wait_dscnt 0x1b
	v_add_f32_e32 v144, v144, v8
	s_wait_dscnt 0x1a
	v_add_f32_e32 v145, v145, v9
	s_wait_dscnt 0x19
	v_add_f32_e32 v146, v146, v10
	s_wait_dscnt 0x18
	v_add_f32_e32 v147, v147, v11
	s_wait_dscnt 0x17
	v_add_f32_e32 v160, v160, v12
	s_wait_dscnt 0x16
	v_add_f32_e32 v161, v161, v13
	s_wait_dscnt 0x15
	v_add_f32_e32 v162, v162, v14
	s_wait_dscnt 0x14
	v_add_f32_e32 v163, v163, v15
	s_wait_dscnt 0x13
	v_add_f32_e32 v176, v176, v16
	s_wait_dscnt 0x12
	v_add_f32_e32 v177, v177, v17
	s_wait_dscnt 0x11
	v_add_f32_e32 v178, v178, v18
	s_wait_dscnt 0x10
	v_add_f32_e32 v179, v179, v19
	s_wait_dscnt 0xf
	v_add_f32_e32 v192, v192, v20
	s_wait_dscnt 0xe
	v_add_f32_e32 v193, v193, v21
	s_wait_dscnt 0xd
	v_add_f32_e32 v194, v194, v22
	s_wait_dscnt 0xc
	v_add_f32_e32 v195, v195, v23
	s_wait_dscnt 0xb
	v_add_f32_e32 v208, v208, v24
	s_wait_dscnt 0xa
	v_add_f32_e32 v209, v209, v25
	s_wait_dscnt 0x9
	v_add_f32_e32 v210, v210, v26
	s_wait_dscnt 0x8
	v_add_f32_e32 v211, v211, v27
	s_wait_dscnt 0x7
	v_add_f32_e32 v224, v224, v28
	s_wait_dscnt 0x6
	v_add_f32_e32 v225, v225, v29
	s_wait_dscnt 0x5
	v_add_f32_e32 v226, v226, v30
	s_wait_dscnt 0x4
	v_add_f32_e32 v227, v227, v31
	s_wait_dscnt 0x3
	v_add_f32_e32 v240, v240, v32
	s_wait_dscnt 0x2
	v_add_f32_e32 v241, v241, v33
	s_wait_dscnt 0x1
	v_add_f32_e32 v242, v242, v34
	s_wait_dscnt 0x0
	v_add_f32_e32 v243, v243, v35
	v_cmpx_eq_u32_e32 0, v0
	v_mov_b32_e32 v40, s42
	s_mul_i32 s40, s39, 1
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s40, s40, s42
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v41, s40
	s_mul_i32 s40, s39, 2
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s40, s40, s42
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v42, s40
	s_mul_i32 s40, s39, 3
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s40, s40, s42
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v43, s40
	s_mul_i32 s40, s39, 4
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s40, s40, s42
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v44, s40
	s_mul_i32 s40, s39, 5
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s40, s40, s42
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v45, s40
	s_mul_i32 s40, s39, 6
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s40, s40, s42
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v46, s40
	global_store_b32 v40, v144, s[8:9]
	global_store_b32 v41, v160, s[8:9]
	global_store_b32 v42, v176, s[8:9]
	global_store_b32 v43, v192, s[8:9]
	global_store_b32 v44, v208, s[8:9]
	global_store_b32 v45, v224, s[8:9]
	global_store_b32 v46, v240, s[8:9]
	s_cmp_le_i32 s13, 1
	s_cbranch_scc1 .Lpx_b7_stored
	global_store_b32 v40, v145, s[8:9] offset:4
	global_store_b32 v41, v161, s[8:9] offset:4
	global_store_b32 v42, v177, s[8:9] offset:4
	global_store_b32 v43, v193, s[8:9] offset:4
	global_store_b32 v44, v209, s[8:9] offset:4
	global_store_b32 v45, v225, s[8:9] offset:4
	global_store_b32 v46, v241, s[8:9] offset:4
	s_cmp_le_i32 s13, 2
	s_cbranch_scc1 .Lpx_b7_stored
	global_store_b32 v40, v146, s[8:9] offset:8
	global_store_b32 v41, v162, s[8:9] offset:8
	global_store_b32 v42, v178, s[8:9] offset:8
	global_store_b32 v43, v194, s[8:9] offset:8
	global_store_b32 v44, v210, s[8:9] offset:8
	global_store_b32 v45, v226, s[8:9] offset:8
	global_store_b32 v46, v242, s[8:9] offset:8
	s_cmp_le_i32 s13, 3
	s_cbranch_scc1 .Lpx_b7_stored
	global_store_b32 v40, v147, s[8:9] offset:12
	global_store_b32 v41, v163, s[8:9] offset:12
	global_store_b32 v42, v179, s[8:9] offset:12
	global_store_b32 v43, v195, s[8:9] offset:12
	global_store_b32 v44, v211, s[8:9] offset:12
	global_store_b32 v45, v227, s[8:9] offset:12
	global_store_b32 v46, v243, s[8:9] offset:12
	.Lpx_b7_stored:
	s_wait_storecnt 0x0
	s_branch .Lpx_end
	.Lpx_b8:
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
	v_mov_b32_e32 v160, 0
	v_mov_b32_e32 v161, 0
	v_mov_b32_e32 v162, 0
	v_mov_b32_e32 v163, 0
	v_mov_b32_e32 v164, 0
	v_mov_b32_e32 v165, 0
	v_mov_b32_e32 v166, 0
	v_mov_b32_e32 v167, 0
	v_mov_b32_e32 v168, 0
	v_mov_b32_e32 v169, 0
	v_mov_b32_e32 v170, 0
	v_mov_b32_e32 v171, 0
	v_mov_b32_e32 v172, 0
	v_mov_b32_e32 v173, 0
	v_mov_b32_e32 v174, 0
	v_mov_b32_e32 v175, 0
	v_mov_b32_e32 v176, 0
	v_mov_b32_e32 v177, 0
	v_mov_b32_e32 v178, 0
	v_mov_b32_e32 v179, 0
	v_mov_b32_e32 v180, 0
	v_mov_b32_e32 v181, 0
	v_mov_b32_e32 v182, 0
	v_mov_b32_e32 v183, 0
	v_mov_b32_e32 v184, 0
	v_mov_b32_e32 v185, 0
	v_mov_b32_e32 v186, 0
	v_mov_b32_e32 v187, 0
	v_mov_b32_e32 v188, 0
	v_mov_b32_e32 v189, 0
	v_mov_b32_e32 v190, 0
	v_mov_b32_e32 v191, 0
	v_mov_b32_e32 v192, 0
	v_mov_b32_e32 v193, 0
	v_mov_b32_e32 v194, 0
	v_mov_b32_e32 v195, 0
	v_mov_b32_e32 v196, 0
	v_mov_b32_e32 v197, 0
	v_mov_b32_e32 v198, 0
	v_mov_b32_e32 v199, 0
	v_mov_b32_e32 v200, 0
	v_mov_b32_e32 v201, 0
	v_mov_b32_e32 v202, 0
	v_mov_b32_e32 v203, 0
	v_mov_b32_e32 v204, 0
	v_mov_b32_e32 v205, 0
	v_mov_b32_e32 v206, 0
	v_mov_b32_e32 v207, 0
	v_mov_b32_e32 v208, 0
	v_mov_b32_e32 v209, 0
	v_mov_b32_e32 v210, 0
	v_mov_b32_e32 v211, 0
	v_mov_b32_e32 v212, 0
	v_mov_b32_e32 v213, 0
	v_mov_b32_e32 v214, 0
	v_mov_b32_e32 v215, 0
	v_mov_b32_e32 v216, 0
	v_mov_b32_e32 v217, 0
	v_mov_b32_e32 v218, 0
	v_mov_b32_e32 v219, 0
	v_mov_b32_e32 v220, 0
	v_mov_b32_e32 v221, 0
	v_mov_b32_e32 v222, 0
	v_mov_b32_e32 v223, 0
	v_mov_b32_e32 v224, 0
	v_mov_b32_e32 v225, 0
	v_mov_b32_e32 v226, 0
	v_mov_b32_e32 v227, 0
	v_mov_b32_e32 v228, 0
	v_mov_b32_e32 v229, 0
	v_mov_b32_e32 v230, 0
	v_mov_b32_e32 v231, 0
	v_mov_b32_e32 v232, 0
	v_mov_b32_e32 v233, 0
	v_mov_b32_e32 v234, 0
	v_mov_b32_e32 v235, 0
	v_mov_b32_e32 v236, 0
	v_mov_b32_e32 v237, 0
	v_mov_b32_e32 v238, 0
	v_mov_b32_e32 v239, 0
	v_mov_b32_e32 v240, 0
	v_mov_b32_e32 v241, 0
	v_mov_b32_e32 v242, 0
	v_mov_b32_e32 v243, 0
	v_mov_b32_e32 v244, 0
	v_mov_b32_e32 v245, 0
	v_mov_b32_e32 v246, 0
	v_mov_b32_e32 v247, 0
	v_mov_b32_e32 v248, 0
	v_mov_b32_e32 v249, 0
	v_mov_b32_e32 v250, 0
	v_mov_b32_e32 v251, 0
	v_mov_b32_e32 v252, 0
	v_mov_b32_e32 v253, 0
	v_mov_b32_e32 v254, 0
	v_mov_b32_e32 v255, 0
	s_lshr_b32 s36, s14, 2
	s_and_b32 s37, s14, 3
	s_cmp_eq_u32 s36, 0
	s_cbranch_scc1 .Lpx_b8_tail
	.Lpx_b8_quad:
	buffer_load_b32 v8, v1, s[20:23], s16 offen
	buffer_load_b32 v9, v1, s[20:23], s17 offen
	buffer_load_b32 v10, v1, s[20:23], s18 offen
	buffer_load_b32 v11, v1, s[20:23], s19 offen
	buffer_load_b32 v24, v2, s[20:23], s16 offen offset:8
	buffer_load_b32 v25, v2, s[20:23], s17 offen offset:8
	buffer_load_b32 v26, v2, s[20:23], s18 offen offset:8
	buffer_load_b32 v27, v2, s[20:23], s19 offen offset:8
	buffer_load_b128 v[80:83], v3, s[24:27], s28 offen
	buffer_load_b128 v[84:87], v3, s[24:27], s28 offen offset:16
	buffer_load_b128 v[88:91], v3, s[24:27], s29 offen
	buffer_load_b128 v[92:95], v3, s[24:27], s29 offen offset:16
	buffer_load_b128 v[96:99], v3, s[24:27], s30 offen
	buffer_load_b128 v[100:103], v3, s[24:27], s30 offen offset:16
	buffer_load_b128 v[104:107], v3, s[24:27], s31 offen
	buffer_load_b128 v[108:111], v3, s[24:27], s31 offen offset:16
	buffer_load_b128 v[112:115], v3, s[24:27], s32 offen
	buffer_load_b128 v[116:119], v3, s[24:27], s32 offen offset:16
	buffer_load_b128 v[120:123], v3, s[24:27], s33 offen
	buffer_load_b128 v[124:127], v3, s[24:27], s33 offen offset:16
	buffer_load_b32 v12, v1, s[20:23], s16 offen offset:136
	buffer_load_b32 v13, v1, s[20:23], s17 offen offset:136
	buffer_load_b32 v14, v1, s[20:23], s18 offen offset:136
	buffer_load_b32 v15, v1, s[20:23], s19 offen offset:136
	buffer_load_b32 v28, v2, s[20:23], s16 offen offset:144
	buffer_load_b32 v29, v2, s[20:23], s17 offen offset:144
	buffer_load_b32 v30, v2, s[20:23], s18 offen offset:144
	buffer_load_b32 v31, v2, s[20:23], s19 offen offset:144
	buffer_load_b32 v16, v1, s[20:23], s16 offen offset:272
	buffer_load_b32 v17, v1, s[20:23], s17 offen offset:272
	buffer_load_b32 v18, v1, s[20:23], s18 offen offset:272
	buffer_load_b32 v19, v1, s[20:23], s19 offen offset:272
	buffer_load_b32 v32, v2, s[20:23], s16 offen offset:280
	buffer_load_b32 v33, v2, s[20:23], s17 offen offset:280
	buffer_load_b32 v34, v2, s[20:23], s18 offen offset:280
	buffer_load_b32 v35, v2, s[20:23], s19 offen offset:280
	buffer_load_b32 v20, v1, s[20:23], s16 offen offset:408
	buffer_load_b32 v21, v1, s[20:23], s17 offen offset:408
	buffer_load_b32 v22, v1, s[20:23], s18 offen offset:408
	buffer_load_b32 v23, v1, s[20:23], s19 offen offset:408
	buffer_load_b32 v36, v2, s[20:23], s16 offen offset:416
	buffer_load_b32 v37, v2, s[20:23], s17 offen offset:416
	buffer_load_b32 v38, v2, s[20:23], s18 offen offset:416
	buffer_load_b32 v39, v2, s[20:23], s19 offen offset:416
	s_wait_loadcnt 0x27
	v_and_b32_e32 v46, 0xf0f0f0f, v24
	s_wait_loadcnt 0x26
	v_and_b32_e32 v56, 0xf0f0f0f, v25
	s_wait_loadcnt 0x25
	v_and_b32_e32 v66, 0xf0f0f0f, v26
	s_wait_loadcnt 0x24
	v_and_b32_e32 v76, 0xf0f0f0f, v27
	v_lshrrev_b32_e32 v47, 4, v24
	v_lshrrev_b32_e32 v57, 4, v25
	v_lshrrev_b32_e32 v67, 4, v26
	v_lshrrev_b32_e32 v77, 4, v27
	v_and_b32_e32 v47, 0xf0f0f0f, v47
	v_and_b32_e32 v57, 0xf0f0f0f, v57
	v_and_b32_e32 v67, 0xf0f0f0f, v67
	v_and_b32_e32 v77, 0xf0f0f0f, v77
	v_cvt_f32_ubyte0_e32 v40, v46
	v_cvt_f32_ubyte0_e32 v50, v56
	v_cvt_f32_ubyte0_e32 v60, v66
	v_cvt_f32_ubyte0_e32 v70, v76
	v_cvt_f32_ubyte1_e32 v42, v46
	v_cvt_f32_ubyte1_e32 v52, v56
	v_cvt_f32_ubyte1_e32 v62, v66
	v_cvt_f32_ubyte1_e32 v72, v76
	v_cvt_f32_ubyte2_e32 v44, v46
	v_cvt_f32_ubyte2_e32 v54, v56
	v_cvt_f32_ubyte2_e32 v64, v66
	v_cvt_f32_ubyte2_e32 v74, v76
	v_cvt_f32_ubyte3_e32 v46, v46
	v_cvt_f32_ubyte3_e32 v56, v56
	v_cvt_f32_ubyte3_e32 v66, v66
	v_cvt_f32_ubyte3_e32 v76, v76
	v_cvt_f32_ubyte0_e32 v41, v47
	v_cvt_f32_ubyte0_e32 v51, v57
	v_cvt_f32_ubyte0_e32 v61, v67
	v_cvt_f32_ubyte0_e32 v71, v77
	v_cvt_f32_ubyte1_e32 v43, v47
	v_cvt_f32_ubyte1_e32 v53, v57
	v_cvt_f32_ubyte1_e32 v63, v67
	v_cvt_f32_ubyte1_e32 v73, v77
	v_cvt_f32_ubyte2_e32 v45, v47
	v_cvt_f32_ubyte2_e32 v55, v57
	v_cvt_f32_ubyte2_e32 v65, v67
	v_cvt_f32_ubyte2_e32 v75, v77
	v_cvt_f32_ubyte3_e32 v47, v47
	v_cvt_f32_ubyte3_e32 v57, v57
	v_cvt_f32_ubyte3_e32 v67, v67
	v_cvt_f32_ubyte3_e32 v77, v77
	v_fma_mix_f32 v40, v8, v40, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v9, v50, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v10, v60, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v70, v11, v70, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v41, v8, v41, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v9, v51, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v10, v61, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v71, v11, v71, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v42, v8, v42, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v52, v9, v52, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v62, v10, v62, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v72, v11, v72, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v43, v8, v43, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v53, v9, v53, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v63, v10, v63, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v73, v11, v73, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v44, v8, v44, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v54, v9, v54, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v64, v10, v64, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v74, v11, v74, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v8, v45, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v9, v55, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v65, v10, v65, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v75, v11, v75, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v8, v46, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v9, v56, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v66, v10, v66, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v76, v11, v76, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v8, v47, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v9, v57, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v67, v10, v67, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v77, v11, v77, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x23
	v_dual_mul_f32 v48, v41, v81 :: v_dual_mul_f32 v49, v51, v81
	v_dual_mul_f32 v58, v61, v81 :: v_dual_mul_f32 v59, v71, v81
	s_wait_loadcnt 0x21
	v_dual_mul_f32 v68, v41, v89 :: v_dual_mul_f32 v69, v51, v89
	v_dual_mul_f32 v78, v61, v89 :: v_dual_mul_f32 v79, v71, v89
	v_dual_fmac_f32 v48, v40, v80 :: v_dual_fmac_f32 v49, v50, v80
	v_dual_fmac_f32 v58, v60, v80 :: v_dual_fmac_f32 v59, v70, v80
	v_dual_fmac_f32 v68, v40, v88 :: v_dual_fmac_f32 v69, v50, v88
	v_dual_fmac_f32 v78, v60, v88 :: v_dual_fmac_f32 v79, v70, v88
	v_dual_fmac_f32 v48, v42, v82 :: v_dual_fmac_f32 v49, v52, v82
	v_dual_fmac_f32 v58, v62, v82 :: v_dual_fmac_f32 v59, v72, v82
	v_dual_fmac_f32 v68, v42, v90 :: v_dual_fmac_f32 v69, v52, v90
	v_dual_fmac_f32 v78, v62, v90 :: v_dual_fmac_f32 v79, v72, v90
	v_dual_fmac_f32 v48, v43, v83 :: v_dual_fmac_f32 v49, v53, v83
	v_dual_fmac_f32 v58, v63, v83 :: v_dual_fmac_f32 v59, v73, v83
	v_dual_fmac_f32 v68, v43, v91 :: v_dual_fmac_f32 v69, v53, v91
	v_dual_fmac_f32 v78, v63, v91 :: v_dual_fmac_f32 v79, v73, v91
	v_dual_fmac_f32 v48, v44, v84 :: v_dual_fmac_f32 v49, v54, v84
	v_dual_fmac_f32 v58, v64, v84 :: v_dual_fmac_f32 v59, v74, v84
	s_wait_loadcnt 0x20
	v_dual_fmac_f32 v68, v44, v92 :: v_dual_fmac_f32 v69, v54, v92
	v_dual_fmac_f32 v78, v64, v92 :: v_dual_fmac_f32 v79, v74, v92
	v_dual_fmac_f32 v48, v45, v85 :: v_dual_fmac_f32 v49, v55, v85
	v_dual_fmac_f32 v58, v65, v85 :: v_dual_fmac_f32 v59, v75, v85
	v_dual_fmac_f32 v68, v45, v93 :: v_dual_fmac_f32 v69, v55, v93
	v_dual_fmac_f32 v78, v65, v93 :: v_dual_fmac_f32 v79, v75, v93
	v_dual_fmac_f32 v48, v46, v86 :: v_dual_fmac_f32 v49, v56, v86
	v_dual_fmac_f32 v58, v66, v86 :: v_dual_fmac_f32 v59, v76, v86
	v_dual_fmac_f32 v68, v46, v94 :: v_dual_fmac_f32 v69, v56, v94
	v_dual_fmac_f32 v78, v66, v94 :: v_dual_fmac_f32 v79, v76, v94
	v_dual_fmac_f32 v48, v47, v87 :: v_dual_fmac_f32 v49, v57, v87
	v_dual_fmac_f32 v58, v67, v87 :: v_dual_fmac_f32 v59, v77, v87
	v_dual_fmac_f32 v68, v47, v95 :: v_dual_fmac_f32 v69, v57, v95
	v_dual_fmac_f32 v78, v67, v95 :: v_dual_fmac_f32 v79, v77, v95
	v_dual_add_f32 v128, v128, v48 :: v_dual_add_f32 v129, v129, v49
	v_dual_add_f32 v130, v130, v58 :: v_dual_add_f32 v131, v131, v59
	v_dual_add_f32 v144, v144, v68 :: v_dual_add_f32 v145, v145, v69
	v_dual_add_f32 v146, v146, v78 :: v_dual_add_f32 v147, v147, v79
	buffer_load_b128 v[80:83], v3, s[24:27], s34 offen
	buffer_load_b128 v[84:87], v3, s[24:27], s34 offen offset:16
	buffer_load_b128 v[88:91], v3, s[24:27], s35 offen
	buffer_load_b128 v[92:95], v3, s[24:27], s35 offen offset:16
	s_wait_loadcnt 0x23
	v_dual_mul_f32 v48, v41, v97 :: v_dual_mul_f32 v49, v51, v97
	v_dual_mul_f32 v58, v61, v97 :: v_dual_mul_f32 v59, v71, v97
	s_wait_loadcnt 0x21
	v_dual_mul_f32 v68, v41, v105 :: v_dual_mul_f32 v69, v51, v105
	v_dual_mul_f32 v78, v61, v105 :: v_dual_mul_f32 v79, v71, v105
	v_dual_fmac_f32 v48, v40, v96 :: v_dual_fmac_f32 v49, v50, v96
	v_dual_fmac_f32 v58, v60, v96 :: v_dual_fmac_f32 v59, v70, v96
	v_dual_fmac_f32 v68, v40, v104 :: v_dual_fmac_f32 v69, v50, v104
	v_dual_fmac_f32 v78, v60, v104 :: v_dual_fmac_f32 v79, v70, v104
	v_dual_fmac_f32 v48, v42, v98 :: v_dual_fmac_f32 v49, v52, v98
	v_dual_fmac_f32 v58, v62, v98 :: v_dual_fmac_f32 v59, v72, v98
	v_dual_fmac_f32 v68, v42, v106 :: v_dual_fmac_f32 v69, v52, v106
	v_dual_fmac_f32 v78, v62, v106 :: v_dual_fmac_f32 v79, v72, v106
	v_dual_fmac_f32 v48, v43, v99 :: v_dual_fmac_f32 v49, v53, v99
	v_dual_fmac_f32 v58, v63, v99 :: v_dual_fmac_f32 v59, v73, v99
	v_dual_fmac_f32 v68, v43, v107 :: v_dual_fmac_f32 v69, v53, v107
	v_dual_fmac_f32 v78, v63, v107 :: v_dual_fmac_f32 v79, v73, v107
	v_dual_fmac_f32 v48, v44, v100 :: v_dual_fmac_f32 v49, v54, v100
	v_dual_fmac_f32 v58, v64, v100 :: v_dual_fmac_f32 v59, v74, v100
	s_wait_loadcnt 0x20
	v_dual_fmac_f32 v68, v44, v108 :: v_dual_fmac_f32 v69, v54, v108
	v_dual_fmac_f32 v78, v64, v108 :: v_dual_fmac_f32 v79, v74, v108
	v_dual_fmac_f32 v48, v45, v101 :: v_dual_fmac_f32 v49, v55, v101
	v_dual_fmac_f32 v58, v65, v101 :: v_dual_fmac_f32 v59, v75, v101
	v_dual_fmac_f32 v68, v45, v109 :: v_dual_fmac_f32 v69, v55, v109
	v_dual_fmac_f32 v78, v65, v109 :: v_dual_fmac_f32 v79, v75, v109
	v_dual_fmac_f32 v48, v46, v102 :: v_dual_fmac_f32 v49, v56, v102
	v_dual_fmac_f32 v58, v66, v102 :: v_dual_fmac_f32 v59, v76, v102
	v_dual_fmac_f32 v68, v46, v110 :: v_dual_fmac_f32 v69, v56, v110
	v_dual_fmac_f32 v78, v66, v110 :: v_dual_fmac_f32 v79, v76, v110
	v_dual_fmac_f32 v48, v47, v103 :: v_dual_fmac_f32 v49, v57, v103
	v_dual_fmac_f32 v58, v67, v103 :: v_dual_fmac_f32 v59, v77, v103
	v_dual_fmac_f32 v68, v47, v111 :: v_dual_fmac_f32 v69, v57, v111
	v_dual_fmac_f32 v78, v67, v111 :: v_dual_fmac_f32 v79, v77, v111
	v_dual_add_f32 v160, v160, v48 :: v_dual_add_f32 v161, v161, v49
	v_dual_add_f32 v162, v162, v58 :: v_dual_add_f32 v163, v163, v59
	v_dual_add_f32 v176, v176, v68 :: v_dual_add_f32 v177, v177, v69
	v_dual_add_f32 v178, v178, v78 :: v_dual_add_f32 v179, v179, v79
	buffer_load_b128 v[96:99], v3, s[24:27], s28 offen offset:1024
	buffer_load_b128 v[100:103], v3, s[24:27], s28 offen offset:1040
	buffer_load_b128 v[104:107], v3, s[24:27], s29 offen offset:1024
	buffer_load_b128 v[108:111], v3, s[24:27], s29 offen offset:1040
	s_wait_loadcnt 0x23
	v_dual_mul_f32 v48, v41, v113 :: v_dual_mul_f32 v49, v51, v113
	v_dual_mul_f32 v58, v61, v113 :: v_dual_mul_f32 v59, v71, v113
	s_wait_loadcnt 0x21
	v_dual_mul_f32 v68, v41, v121 :: v_dual_mul_f32 v69, v51, v121
	v_dual_mul_f32 v78, v61, v121 :: v_dual_mul_f32 v79, v71, v121
	v_dual_fmac_f32 v48, v40, v112 :: v_dual_fmac_f32 v49, v50, v112
	v_dual_fmac_f32 v58, v60, v112 :: v_dual_fmac_f32 v59, v70, v112
	v_dual_fmac_f32 v68, v40, v120 :: v_dual_fmac_f32 v69, v50, v120
	v_dual_fmac_f32 v78, v60, v120 :: v_dual_fmac_f32 v79, v70, v120
	v_dual_fmac_f32 v48, v42, v114 :: v_dual_fmac_f32 v49, v52, v114
	v_dual_fmac_f32 v58, v62, v114 :: v_dual_fmac_f32 v59, v72, v114
	v_dual_fmac_f32 v68, v42, v122 :: v_dual_fmac_f32 v69, v52, v122
	v_dual_fmac_f32 v78, v62, v122 :: v_dual_fmac_f32 v79, v72, v122
	v_dual_fmac_f32 v48, v43, v115 :: v_dual_fmac_f32 v49, v53, v115
	v_dual_fmac_f32 v58, v63, v115 :: v_dual_fmac_f32 v59, v73, v115
	v_dual_fmac_f32 v68, v43, v123 :: v_dual_fmac_f32 v69, v53, v123
	v_dual_fmac_f32 v78, v63, v123 :: v_dual_fmac_f32 v79, v73, v123
	v_dual_fmac_f32 v48, v44, v116 :: v_dual_fmac_f32 v49, v54, v116
	v_dual_fmac_f32 v58, v64, v116 :: v_dual_fmac_f32 v59, v74, v116
	s_wait_loadcnt 0x20
	v_dual_fmac_f32 v68, v44, v124 :: v_dual_fmac_f32 v69, v54, v124
	v_dual_fmac_f32 v78, v64, v124 :: v_dual_fmac_f32 v79, v74, v124
	v_dual_fmac_f32 v48, v45, v117 :: v_dual_fmac_f32 v49, v55, v117
	v_dual_fmac_f32 v58, v65, v117 :: v_dual_fmac_f32 v59, v75, v117
	v_dual_fmac_f32 v68, v45, v125 :: v_dual_fmac_f32 v69, v55, v125
	v_dual_fmac_f32 v78, v65, v125 :: v_dual_fmac_f32 v79, v75, v125
	v_dual_fmac_f32 v48, v46, v118 :: v_dual_fmac_f32 v49, v56, v118
	v_dual_fmac_f32 v58, v66, v118 :: v_dual_fmac_f32 v59, v76, v118
	v_dual_fmac_f32 v68, v46, v126 :: v_dual_fmac_f32 v69, v56, v126
	v_dual_fmac_f32 v78, v66, v126 :: v_dual_fmac_f32 v79, v76, v126
	v_dual_fmac_f32 v48, v47, v119 :: v_dual_fmac_f32 v49, v57, v119
	v_dual_fmac_f32 v58, v67, v119 :: v_dual_fmac_f32 v59, v77, v119
	v_dual_fmac_f32 v68, v47, v127 :: v_dual_fmac_f32 v69, v57, v127
	v_dual_fmac_f32 v78, v67, v127 :: v_dual_fmac_f32 v79, v77, v127
	v_dual_add_f32 v192, v192, v48 :: v_dual_add_f32 v193, v193, v49
	v_dual_add_f32 v194, v194, v58 :: v_dual_add_f32 v195, v195, v59
	v_dual_add_f32 v208, v208, v68 :: v_dual_add_f32 v209, v209, v69
	v_dual_add_f32 v210, v210, v78 :: v_dual_add_f32 v211, v211, v79
	buffer_load_b128 v[112:115], v3, s[24:27], s30 offen offset:1024
	buffer_load_b128 v[116:119], v3, s[24:27], s30 offen offset:1040
	buffer_load_b128 v[120:123], v3, s[24:27], s31 offen offset:1024
	buffer_load_b128 v[124:127], v3, s[24:27], s31 offen offset:1040
	s_wait_loadcnt 0xb
	v_dual_mul_f32 v48, v41, v81 :: v_dual_mul_f32 v49, v51, v81
	v_dual_mul_f32 v58, v61, v81 :: v_dual_mul_f32 v59, v71, v81
	s_wait_loadcnt 0x9
	v_dual_mul_f32 v68, v41, v89 :: v_dual_mul_f32 v69, v51, v89
	v_dual_mul_f32 v78, v61, v89 :: v_dual_mul_f32 v79, v71, v89
	v_dual_fmac_f32 v48, v40, v80 :: v_dual_fmac_f32 v49, v50, v80
	v_dual_fmac_f32 v58, v60, v80 :: v_dual_fmac_f32 v59, v70, v80
	v_dual_fmac_f32 v68, v40, v88 :: v_dual_fmac_f32 v69, v50, v88
	v_dual_fmac_f32 v78, v60, v88 :: v_dual_fmac_f32 v79, v70, v88
	v_dual_fmac_f32 v48, v42, v82 :: v_dual_fmac_f32 v49, v52, v82
	v_dual_fmac_f32 v58, v62, v82 :: v_dual_fmac_f32 v59, v72, v82
	v_dual_fmac_f32 v68, v42, v90 :: v_dual_fmac_f32 v69, v52, v90
	v_dual_fmac_f32 v78, v62, v90 :: v_dual_fmac_f32 v79, v72, v90
	v_dual_fmac_f32 v48, v43, v83 :: v_dual_fmac_f32 v49, v53, v83
	v_dual_fmac_f32 v58, v63, v83 :: v_dual_fmac_f32 v59, v73, v83
	v_dual_fmac_f32 v68, v43, v91 :: v_dual_fmac_f32 v69, v53, v91
	v_dual_fmac_f32 v78, v63, v91 :: v_dual_fmac_f32 v79, v73, v91
	v_dual_fmac_f32 v48, v44, v84 :: v_dual_fmac_f32 v49, v54, v84
	v_dual_fmac_f32 v58, v64, v84 :: v_dual_fmac_f32 v59, v74, v84
	s_wait_loadcnt 0x8
	v_dual_fmac_f32 v68, v44, v92 :: v_dual_fmac_f32 v69, v54, v92
	v_dual_fmac_f32 v78, v64, v92 :: v_dual_fmac_f32 v79, v74, v92
	v_dual_fmac_f32 v48, v45, v85 :: v_dual_fmac_f32 v49, v55, v85
	v_dual_fmac_f32 v58, v65, v85 :: v_dual_fmac_f32 v59, v75, v85
	v_dual_fmac_f32 v68, v45, v93 :: v_dual_fmac_f32 v69, v55, v93
	v_dual_fmac_f32 v78, v65, v93 :: v_dual_fmac_f32 v79, v75, v93
	v_dual_fmac_f32 v48, v46, v86 :: v_dual_fmac_f32 v49, v56, v86
	v_dual_fmac_f32 v58, v66, v86 :: v_dual_fmac_f32 v59, v76, v86
	v_dual_fmac_f32 v68, v46, v94 :: v_dual_fmac_f32 v69, v56, v94
	v_dual_fmac_f32 v78, v66, v94 :: v_dual_fmac_f32 v79, v76, v94
	v_dual_fmac_f32 v48, v47, v87 :: v_dual_fmac_f32 v49, v57, v87
	v_dual_fmac_f32 v58, v67, v87 :: v_dual_fmac_f32 v59, v77, v87
	v_dual_fmac_f32 v68, v47, v95 :: v_dual_fmac_f32 v69, v57, v95
	v_dual_fmac_f32 v78, v67, v95 :: v_dual_fmac_f32 v79, v77, v95
	v_dual_add_f32 v224, v224, v48 :: v_dual_add_f32 v225, v225, v49
	v_dual_add_f32 v226, v226, v58 :: v_dual_add_f32 v227, v227, v59
	v_dual_add_f32 v240, v240, v68 :: v_dual_add_f32 v241, v241, v69
	v_dual_add_f32 v242, v242, v78 :: v_dual_add_f32 v243, v243, v79
	v_and_b32_e32 v46, 0xf0f0f0f, v28
	v_and_b32_e32 v56, 0xf0f0f0f, v29
	v_and_b32_e32 v66, 0xf0f0f0f, v30
	v_and_b32_e32 v76, 0xf0f0f0f, v31
	v_lshrrev_b32_e32 v47, 4, v28
	v_lshrrev_b32_e32 v57, 4, v29
	v_lshrrev_b32_e32 v67, 4, v30
	v_lshrrev_b32_e32 v77, 4, v31
	v_and_b32_e32 v47, 0xf0f0f0f, v47
	v_and_b32_e32 v57, 0xf0f0f0f, v57
	v_and_b32_e32 v67, 0xf0f0f0f, v67
	v_and_b32_e32 v77, 0xf0f0f0f, v77
	v_cvt_f32_ubyte0_e32 v40, v46
	v_cvt_f32_ubyte0_e32 v50, v56
	v_cvt_f32_ubyte0_e32 v60, v66
	v_cvt_f32_ubyte0_e32 v70, v76
	v_cvt_f32_ubyte1_e32 v42, v46
	v_cvt_f32_ubyte1_e32 v52, v56
	v_cvt_f32_ubyte1_e32 v62, v66
	v_cvt_f32_ubyte1_e32 v72, v76
	v_cvt_f32_ubyte2_e32 v44, v46
	v_cvt_f32_ubyte2_e32 v54, v56
	v_cvt_f32_ubyte2_e32 v64, v66
	v_cvt_f32_ubyte2_e32 v74, v76
	v_cvt_f32_ubyte3_e32 v46, v46
	v_cvt_f32_ubyte3_e32 v56, v56
	v_cvt_f32_ubyte3_e32 v66, v66
	v_cvt_f32_ubyte3_e32 v76, v76
	v_cvt_f32_ubyte0_e32 v41, v47
	v_cvt_f32_ubyte0_e32 v51, v57
	v_cvt_f32_ubyte0_e32 v61, v67
	v_cvt_f32_ubyte0_e32 v71, v77
	v_cvt_f32_ubyte1_e32 v43, v47
	v_cvt_f32_ubyte1_e32 v53, v57
	v_cvt_f32_ubyte1_e32 v63, v67
	v_cvt_f32_ubyte1_e32 v73, v77
	v_cvt_f32_ubyte2_e32 v45, v47
	v_cvt_f32_ubyte2_e32 v55, v57
	v_cvt_f32_ubyte2_e32 v65, v67
	v_cvt_f32_ubyte2_e32 v75, v77
	v_cvt_f32_ubyte3_e32 v47, v47
	v_cvt_f32_ubyte3_e32 v57, v57
	v_cvt_f32_ubyte3_e32 v67, v67
	v_cvt_f32_ubyte3_e32 v77, v77
	v_fma_mix_f32 v40, v12, v40, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v13, v50, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v14, v60, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v70, v15, v70, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v41, v12, v41, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v13, v51, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v14, v61, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v71, v15, v71, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v42, v12, v42, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v52, v13, v52, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v62, v14, v62, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v72, v15, v72, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v43, v12, v43, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v53, v13, v53, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v63, v14, v63, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v73, v15, v73, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v44, v12, v44, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v54, v13, v54, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v64, v14, v64, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v74, v15, v74, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v12, v45, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v13, v55, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v65, v14, v65, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v75, v15, v75, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v12, v46, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v13, v56, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v66, v14, v66, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v76, v15, v76, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v12, v47, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v13, v57, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v67, v14, v67, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v77, v15, v77, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	buffer_load_b128 v[80:83], v3, s[24:27], s32 offen offset:1024
	buffer_load_b128 v[84:87], v3, s[24:27], s32 offen offset:1040
	buffer_load_b128 v[88:91], v3, s[24:27], s33 offen offset:1024
	buffer_load_b128 v[92:95], v3, s[24:27], s33 offen offset:1040
	s_wait_loadcnt 0xb
	v_dual_mul_f32 v48, v41, v97 :: v_dual_mul_f32 v49, v51, v97
	v_dual_mul_f32 v58, v61, v97 :: v_dual_mul_f32 v59, v71, v97
	s_wait_loadcnt 0x9
	v_dual_mul_f32 v68, v41, v105 :: v_dual_mul_f32 v69, v51, v105
	v_dual_mul_f32 v78, v61, v105 :: v_dual_mul_f32 v79, v71, v105
	v_dual_fmac_f32 v48, v40, v96 :: v_dual_fmac_f32 v49, v50, v96
	v_dual_fmac_f32 v58, v60, v96 :: v_dual_fmac_f32 v59, v70, v96
	v_dual_fmac_f32 v68, v40, v104 :: v_dual_fmac_f32 v69, v50, v104
	v_dual_fmac_f32 v78, v60, v104 :: v_dual_fmac_f32 v79, v70, v104
	v_dual_fmac_f32 v48, v42, v98 :: v_dual_fmac_f32 v49, v52, v98
	v_dual_fmac_f32 v58, v62, v98 :: v_dual_fmac_f32 v59, v72, v98
	v_dual_fmac_f32 v68, v42, v106 :: v_dual_fmac_f32 v69, v52, v106
	v_dual_fmac_f32 v78, v62, v106 :: v_dual_fmac_f32 v79, v72, v106
	v_dual_fmac_f32 v48, v43, v99 :: v_dual_fmac_f32 v49, v53, v99
	v_dual_fmac_f32 v58, v63, v99 :: v_dual_fmac_f32 v59, v73, v99
	v_dual_fmac_f32 v68, v43, v107 :: v_dual_fmac_f32 v69, v53, v107
	v_dual_fmac_f32 v78, v63, v107 :: v_dual_fmac_f32 v79, v73, v107
	v_dual_fmac_f32 v48, v44, v100 :: v_dual_fmac_f32 v49, v54, v100
	v_dual_fmac_f32 v58, v64, v100 :: v_dual_fmac_f32 v59, v74, v100
	s_wait_loadcnt 0x8
	v_dual_fmac_f32 v68, v44, v108 :: v_dual_fmac_f32 v69, v54, v108
	v_dual_fmac_f32 v78, v64, v108 :: v_dual_fmac_f32 v79, v74, v108
	v_dual_fmac_f32 v48, v45, v101 :: v_dual_fmac_f32 v49, v55, v101
	v_dual_fmac_f32 v58, v65, v101 :: v_dual_fmac_f32 v59, v75, v101
	v_dual_fmac_f32 v68, v45, v109 :: v_dual_fmac_f32 v69, v55, v109
	v_dual_fmac_f32 v78, v65, v109 :: v_dual_fmac_f32 v79, v75, v109
	v_dual_fmac_f32 v48, v46, v102 :: v_dual_fmac_f32 v49, v56, v102
	v_dual_fmac_f32 v58, v66, v102 :: v_dual_fmac_f32 v59, v76, v102
	v_dual_fmac_f32 v68, v46, v110 :: v_dual_fmac_f32 v69, v56, v110
	v_dual_fmac_f32 v78, v66, v110 :: v_dual_fmac_f32 v79, v76, v110
	v_dual_fmac_f32 v48, v47, v103 :: v_dual_fmac_f32 v49, v57, v103
	v_dual_fmac_f32 v58, v67, v103 :: v_dual_fmac_f32 v59, v77, v103
	v_dual_fmac_f32 v68, v47, v111 :: v_dual_fmac_f32 v69, v57, v111
	v_dual_fmac_f32 v78, v67, v111 :: v_dual_fmac_f32 v79, v77, v111
	v_dual_add_f32 v132, v132, v48 :: v_dual_add_f32 v133, v133, v49
	v_dual_add_f32 v134, v134, v58 :: v_dual_add_f32 v135, v135, v59
	v_dual_add_f32 v148, v148, v68 :: v_dual_add_f32 v149, v149, v69
	v_dual_add_f32 v150, v150, v78 :: v_dual_add_f32 v151, v151, v79
	buffer_load_b128 v[96:99], v3, s[24:27], s34 offen offset:1024
	buffer_load_b128 v[100:103], v3, s[24:27], s34 offen offset:1040
	buffer_load_b128 v[104:107], v3, s[24:27], s35 offen offset:1024
	buffer_load_b128 v[108:111], v3, s[24:27], s35 offen offset:1040
	s_wait_loadcnt 0xb
	v_dual_mul_f32 v48, v41, v113 :: v_dual_mul_f32 v49, v51, v113
	v_dual_mul_f32 v58, v61, v113 :: v_dual_mul_f32 v59, v71, v113
	s_wait_loadcnt 0x9
	v_dual_mul_f32 v68, v41, v121 :: v_dual_mul_f32 v69, v51, v121
	v_dual_mul_f32 v78, v61, v121 :: v_dual_mul_f32 v79, v71, v121
	v_dual_fmac_f32 v48, v40, v112 :: v_dual_fmac_f32 v49, v50, v112
	v_dual_fmac_f32 v58, v60, v112 :: v_dual_fmac_f32 v59, v70, v112
	v_dual_fmac_f32 v68, v40, v120 :: v_dual_fmac_f32 v69, v50, v120
	v_dual_fmac_f32 v78, v60, v120 :: v_dual_fmac_f32 v79, v70, v120
	v_dual_fmac_f32 v48, v42, v114 :: v_dual_fmac_f32 v49, v52, v114
	v_dual_fmac_f32 v58, v62, v114 :: v_dual_fmac_f32 v59, v72, v114
	v_dual_fmac_f32 v68, v42, v122 :: v_dual_fmac_f32 v69, v52, v122
	v_dual_fmac_f32 v78, v62, v122 :: v_dual_fmac_f32 v79, v72, v122
	v_dual_fmac_f32 v48, v43, v115 :: v_dual_fmac_f32 v49, v53, v115
	v_dual_fmac_f32 v58, v63, v115 :: v_dual_fmac_f32 v59, v73, v115
	v_dual_fmac_f32 v68, v43, v123 :: v_dual_fmac_f32 v69, v53, v123
	v_dual_fmac_f32 v78, v63, v123 :: v_dual_fmac_f32 v79, v73, v123
	v_dual_fmac_f32 v48, v44, v116 :: v_dual_fmac_f32 v49, v54, v116
	v_dual_fmac_f32 v58, v64, v116 :: v_dual_fmac_f32 v59, v74, v116
	s_wait_loadcnt 0x8
	v_dual_fmac_f32 v68, v44, v124 :: v_dual_fmac_f32 v69, v54, v124
	v_dual_fmac_f32 v78, v64, v124 :: v_dual_fmac_f32 v79, v74, v124
	v_dual_fmac_f32 v48, v45, v117 :: v_dual_fmac_f32 v49, v55, v117
	v_dual_fmac_f32 v58, v65, v117 :: v_dual_fmac_f32 v59, v75, v117
	v_dual_fmac_f32 v68, v45, v125 :: v_dual_fmac_f32 v69, v55, v125
	v_dual_fmac_f32 v78, v65, v125 :: v_dual_fmac_f32 v79, v75, v125
	v_dual_fmac_f32 v48, v46, v118 :: v_dual_fmac_f32 v49, v56, v118
	v_dual_fmac_f32 v58, v66, v118 :: v_dual_fmac_f32 v59, v76, v118
	v_dual_fmac_f32 v68, v46, v126 :: v_dual_fmac_f32 v69, v56, v126
	v_dual_fmac_f32 v78, v66, v126 :: v_dual_fmac_f32 v79, v76, v126
	v_dual_fmac_f32 v48, v47, v119 :: v_dual_fmac_f32 v49, v57, v119
	v_dual_fmac_f32 v58, v67, v119 :: v_dual_fmac_f32 v59, v77, v119
	v_dual_fmac_f32 v68, v47, v127 :: v_dual_fmac_f32 v69, v57, v127
	v_dual_fmac_f32 v78, v67, v127 :: v_dual_fmac_f32 v79, v77, v127
	v_dual_add_f32 v164, v164, v48 :: v_dual_add_f32 v165, v165, v49
	v_dual_add_f32 v166, v166, v58 :: v_dual_add_f32 v167, v167, v59
	v_dual_add_f32 v180, v180, v68 :: v_dual_add_f32 v181, v181, v69
	v_dual_add_f32 v182, v182, v78 :: v_dual_add_f32 v183, v183, v79
	buffer_load_b128 v[112:115], v3, s[24:27], s28 offen offset:2048
	buffer_load_b128 v[116:119], v3, s[24:27], s28 offen offset:2064
	buffer_load_b128 v[120:123], v3, s[24:27], s29 offen offset:2048
	buffer_load_b128 v[124:127], v3, s[24:27], s29 offen offset:2064
	s_wait_loadcnt 0xb
	v_dual_mul_f32 v48, v41, v81 :: v_dual_mul_f32 v49, v51, v81
	v_dual_mul_f32 v58, v61, v81 :: v_dual_mul_f32 v59, v71, v81
	s_wait_loadcnt 0x9
	v_dual_mul_f32 v68, v41, v89 :: v_dual_mul_f32 v69, v51, v89
	v_dual_mul_f32 v78, v61, v89 :: v_dual_mul_f32 v79, v71, v89
	v_dual_fmac_f32 v48, v40, v80 :: v_dual_fmac_f32 v49, v50, v80
	v_dual_fmac_f32 v58, v60, v80 :: v_dual_fmac_f32 v59, v70, v80
	v_dual_fmac_f32 v68, v40, v88 :: v_dual_fmac_f32 v69, v50, v88
	v_dual_fmac_f32 v78, v60, v88 :: v_dual_fmac_f32 v79, v70, v88
	v_dual_fmac_f32 v48, v42, v82 :: v_dual_fmac_f32 v49, v52, v82
	v_dual_fmac_f32 v58, v62, v82 :: v_dual_fmac_f32 v59, v72, v82
	v_dual_fmac_f32 v68, v42, v90 :: v_dual_fmac_f32 v69, v52, v90
	v_dual_fmac_f32 v78, v62, v90 :: v_dual_fmac_f32 v79, v72, v90
	v_dual_fmac_f32 v48, v43, v83 :: v_dual_fmac_f32 v49, v53, v83
	v_dual_fmac_f32 v58, v63, v83 :: v_dual_fmac_f32 v59, v73, v83
	v_dual_fmac_f32 v68, v43, v91 :: v_dual_fmac_f32 v69, v53, v91
	v_dual_fmac_f32 v78, v63, v91 :: v_dual_fmac_f32 v79, v73, v91
	v_dual_fmac_f32 v48, v44, v84 :: v_dual_fmac_f32 v49, v54, v84
	v_dual_fmac_f32 v58, v64, v84 :: v_dual_fmac_f32 v59, v74, v84
	s_wait_loadcnt 0x8
	v_dual_fmac_f32 v68, v44, v92 :: v_dual_fmac_f32 v69, v54, v92
	v_dual_fmac_f32 v78, v64, v92 :: v_dual_fmac_f32 v79, v74, v92
	v_dual_fmac_f32 v48, v45, v85 :: v_dual_fmac_f32 v49, v55, v85
	v_dual_fmac_f32 v58, v65, v85 :: v_dual_fmac_f32 v59, v75, v85
	v_dual_fmac_f32 v68, v45, v93 :: v_dual_fmac_f32 v69, v55, v93
	v_dual_fmac_f32 v78, v65, v93 :: v_dual_fmac_f32 v79, v75, v93
	v_dual_fmac_f32 v48, v46, v86 :: v_dual_fmac_f32 v49, v56, v86
	v_dual_fmac_f32 v58, v66, v86 :: v_dual_fmac_f32 v59, v76, v86
	v_dual_fmac_f32 v68, v46, v94 :: v_dual_fmac_f32 v69, v56, v94
	v_dual_fmac_f32 v78, v66, v94 :: v_dual_fmac_f32 v79, v76, v94
	v_dual_fmac_f32 v48, v47, v87 :: v_dual_fmac_f32 v49, v57, v87
	v_dual_fmac_f32 v58, v67, v87 :: v_dual_fmac_f32 v59, v77, v87
	v_dual_fmac_f32 v68, v47, v95 :: v_dual_fmac_f32 v69, v57, v95
	v_dual_fmac_f32 v78, v67, v95 :: v_dual_fmac_f32 v79, v77, v95
	v_dual_add_f32 v196, v196, v48 :: v_dual_add_f32 v197, v197, v49
	v_dual_add_f32 v198, v198, v58 :: v_dual_add_f32 v199, v199, v59
	v_dual_add_f32 v212, v212, v68 :: v_dual_add_f32 v213, v213, v69
	v_dual_add_f32 v214, v214, v78 :: v_dual_add_f32 v215, v215, v79
	buffer_load_b128 v[80:83], v3, s[24:27], s30 offen offset:2048
	buffer_load_b128 v[84:87], v3, s[24:27], s30 offen offset:2064
	buffer_load_b128 v[88:91], v3, s[24:27], s31 offen offset:2048
	buffer_load_b128 v[92:95], v3, s[24:27], s31 offen offset:2064
	s_wait_loadcnt 0xb
	v_dual_mul_f32 v48, v41, v97 :: v_dual_mul_f32 v49, v51, v97
	v_dual_mul_f32 v58, v61, v97 :: v_dual_mul_f32 v59, v71, v97
	s_wait_loadcnt 0x9
	v_dual_mul_f32 v68, v41, v105 :: v_dual_mul_f32 v69, v51, v105
	v_dual_mul_f32 v78, v61, v105 :: v_dual_mul_f32 v79, v71, v105
	v_dual_fmac_f32 v48, v40, v96 :: v_dual_fmac_f32 v49, v50, v96
	v_dual_fmac_f32 v58, v60, v96 :: v_dual_fmac_f32 v59, v70, v96
	v_dual_fmac_f32 v68, v40, v104 :: v_dual_fmac_f32 v69, v50, v104
	v_dual_fmac_f32 v78, v60, v104 :: v_dual_fmac_f32 v79, v70, v104
	v_dual_fmac_f32 v48, v42, v98 :: v_dual_fmac_f32 v49, v52, v98
	v_dual_fmac_f32 v58, v62, v98 :: v_dual_fmac_f32 v59, v72, v98
	v_dual_fmac_f32 v68, v42, v106 :: v_dual_fmac_f32 v69, v52, v106
	v_dual_fmac_f32 v78, v62, v106 :: v_dual_fmac_f32 v79, v72, v106
	v_dual_fmac_f32 v48, v43, v99 :: v_dual_fmac_f32 v49, v53, v99
	v_dual_fmac_f32 v58, v63, v99 :: v_dual_fmac_f32 v59, v73, v99
	v_dual_fmac_f32 v68, v43, v107 :: v_dual_fmac_f32 v69, v53, v107
	v_dual_fmac_f32 v78, v63, v107 :: v_dual_fmac_f32 v79, v73, v107
	v_dual_fmac_f32 v48, v44, v100 :: v_dual_fmac_f32 v49, v54, v100
	v_dual_fmac_f32 v58, v64, v100 :: v_dual_fmac_f32 v59, v74, v100
	s_wait_loadcnt 0x8
	v_dual_fmac_f32 v68, v44, v108 :: v_dual_fmac_f32 v69, v54, v108
	v_dual_fmac_f32 v78, v64, v108 :: v_dual_fmac_f32 v79, v74, v108
	v_dual_fmac_f32 v48, v45, v101 :: v_dual_fmac_f32 v49, v55, v101
	v_dual_fmac_f32 v58, v65, v101 :: v_dual_fmac_f32 v59, v75, v101
	v_dual_fmac_f32 v68, v45, v109 :: v_dual_fmac_f32 v69, v55, v109
	v_dual_fmac_f32 v78, v65, v109 :: v_dual_fmac_f32 v79, v75, v109
	v_dual_fmac_f32 v48, v46, v102 :: v_dual_fmac_f32 v49, v56, v102
	v_dual_fmac_f32 v58, v66, v102 :: v_dual_fmac_f32 v59, v76, v102
	v_dual_fmac_f32 v68, v46, v110 :: v_dual_fmac_f32 v69, v56, v110
	v_dual_fmac_f32 v78, v66, v110 :: v_dual_fmac_f32 v79, v76, v110
	v_dual_fmac_f32 v48, v47, v103 :: v_dual_fmac_f32 v49, v57, v103
	v_dual_fmac_f32 v58, v67, v103 :: v_dual_fmac_f32 v59, v77, v103
	v_dual_fmac_f32 v68, v47, v111 :: v_dual_fmac_f32 v69, v57, v111
	v_dual_fmac_f32 v78, v67, v111 :: v_dual_fmac_f32 v79, v77, v111
	v_dual_add_f32 v228, v228, v48 :: v_dual_add_f32 v229, v229, v49
	v_dual_add_f32 v230, v230, v58 :: v_dual_add_f32 v231, v231, v59
	v_dual_add_f32 v244, v244, v68 :: v_dual_add_f32 v245, v245, v69
	v_dual_add_f32 v246, v246, v78 :: v_dual_add_f32 v247, v247, v79
	v_and_b32_e32 v46, 0xf0f0f0f, v32
	v_and_b32_e32 v56, 0xf0f0f0f, v33
	v_and_b32_e32 v66, 0xf0f0f0f, v34
	v_and_b32_e32 v76, 0xf0f0f0f, v35
	v_lshrrev_b32_e32 v47, 4, v32
	v_lshrrev_b32_e32 v57, 4, v33
	v_lshrrev_b32_e32 v67, 4, v34
	v_lshrrev_b32_e32 v77, 4, v35
	v_and_b32_e32 v47, 0xf0f0f0f, v47
	v_and_b32_e32 v57, 0xf0f0f0f, v57
	v_and_b32_e32 v67, 0xf0f0f0f, v67
	v_and_b32_e32 v77, 0xf0f0f0f, v77
	v_cvt_f32_ubyte0_e32 v40, v46
	v_cvt_f32_ubyte0_e32 v50, v56
	v_cvt_f32_ubyte0_e32 v60, v66
	v_cvt_f32_ubyte0_e32 v70, v76
	v_cvt_f32_ubyte1_e32 v42, v46
	v_cvt_f32_ubyte1_e32 v52, v56
	v_cvt_f32_ubyte1_e32 v62, v66
	v_cvt_f32_ubyte1_e32 v72, v76
	v_cvt_f32_ubyte2_e32 v44, v46
	v_cvt_f32_ubyte2_e32 v54, v56
	v_cvt_f32_ubyte2_e32 v64, v66
	v_cvt_f32_ubyte2_e32 v74, v76
	v_cvt_f32_ubyte3_e32 v46, v46
	v_cvt_f32_ubyte3_e32 v56, v56
	v_cvt_f32_ubyte3_e32 v66, v66
	v_cvt_f32_ubyte3_e32 v76, v76
	v_cvt_f32_ubyte0_e32 v41, v47
	v_cvt_f32_ubyte0_e32 v51, v57
	v_cvt_f32_ubyte0_e32 v61, v67
	v_cvt_f32_ubyte0_e32 v71, v77
	v_cvt_f32_ubyte1_e32 v43, v47
	v_cvt_f32_ubyte1_e32 v53, v57
	v_cvt_f32_ubyte1_e32 v63, v67
	v_cvt_f32_ubyte1_e32 v73, v77
	v_cvt_f32_ubyte2_e32 v45, v47
	v_cvt_f32_ubyte2_e32 v55, v57
	v_cvt_f32_ubyte2_e32 v65, v67
	v_cvt_f32_ubyte2_e32 v75, v77
	v_cvt_f32_ubyte3_e32 v47, v47
	v_cvt_f32_ubyte3_e32 v57, v57
	v_cvt_f32_ubyte3_e32 v67, v67
	v_cvt_f32_ubyte3_e32 v77, v77
	v_fma_mix_f32 v40, v16, v40, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v17, v50, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v18, v60, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v70, v19, v70, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v41, v16, v41, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v17, v51, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v18, v61, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v71, v19, v71, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v42, v16, v42, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v52, v17, v52, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v62, v18, v62, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v72, v19, v72, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v43, v16, v43, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v53, v17, v53, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v63, v18, v63, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v73, v19, v73, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v44, v16, v44, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v54, v17, v54, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v64, v18, v64, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v74, v19, v74, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v16, v45, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v17, v55, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v65, v18, v65, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v75, v19, v75, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v16, v46, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v17, v56, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v66, v18, v66, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v76, v19, v76, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v16, v47, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v17, v57, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v67, v18, v67, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v77, v19, v77, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	buffer_load_b128 v[96:99], v3, s[24:27], s32 offen offset:2048
	buffer_load_b128 v[100:103], v3, s[24:27], s32 offen offset:2064
	buffer_load_b128 v[104:107], v3, s[24:27], s33 offen offset:2048
	buffer_load_b128 v[108:111], v3, s[24:27], s33 offen offset:2064
	s_wait_loadcnt 0xb
	v_dual_mul_f32 v48, v41, v113 :: v_dual_mul_f32 v49, v51, v113
	v_dual_mul_f32 v58, v61, v113 :: v_dual_mul_f32 v59, v71, v113
	s_wait_loadcnt 0x9
	v_dual_mul_f32 v68, v41, v121 :: v_dual_mul_f32 v69, v51, v121
	v_dual_mul_f32 v78, v61, v121 :: v_dual_mul_f32 v79, v71, v121
	v_dual_fmac_f32 v48, v40, v112 :: v_dual_fmac_f32 v49, v50, v112
	v_dual_fmac_f32 v58, v60, v112 :: v_dual_fmac_f32 v59, v70, v112
	v_dual_fmac_f32 v68, v40, v120 :: v_dual_fmac_f32 v69, v50, v120
	v_dual_fmac_f32 v78, v60, v120 :: v_dual_fmac_f32 v79, v70, v120
	v_dual_fmac_f32 v48, v42, v114 :: v_dual_fmac_f32 v49, v52, v114
	v_dual_fmac_f32 v58, v62, v114 :: v_dual_fmac_f32 v59, v72, v114
	v_dual_fmac_f32 v68, v42, v122 :: v_dual_fmac_f32 v69, v52, v122
	v_dual_fmac_f32 v78, v62, v122 :: v_dual_fmac_f32 v79, v72, v122
	v_dual_fmac_f32 v48, v43, v115 :: v_dual_fmac_f32 v49, v53, v115
	v_dual_fmac_f32 v58, v63, v115 :: v_dual_fmac_f32 v59, v73, v115
	v_dual_fmac_f32 v68, v43, v123 :: v_dual_fmac_f32 v69, v53, v123
	v_dual_fmac_f32 v78, v63, v123 :: v_dual_fmac_f32 v79, v73, v123
	v_dual_fmac_f32 v48, v44, v116 :: v_dual_fmac_f32 v49, v54, v116
	v_dual_fmac_f32 v58, v64, v116 :: v_dual_fmac_f32 v59, v74, v116
	s_wait_loadcnt 0x8
	v_dual_fmac_f32 v68, v44, v124 :: v_dual_fmac_f32 v69, v54, v124
	v_dual_fmac_f32 v78, v64, v124 :: v_dual_fmac_f32 v79, v74, v124
	v_dual_fmac_f32 v48, v45, v117 :: v_dual_fmac_f32 v49, v55, v117
	v_dual_fmac_f32 v58, v65, v117 :: v_dual_fmac_f32 v59, v75, v117
	v_dual_fmac_f32 v68, v45, v125 :: v_dual_fmac_f32 v69, v55, v125
	v_dual_fmac_f32 v78, v65, v125 :: v_dual_fmac_f32 v79, v75, v125
	v_dual_fmac_f32 v48, v46, v118 :: v_dual_fmac_f32 v49, v56, v118
	v_dual_fmac_f32 v58, v66, v118 :: v_dual_fmac_f32 v59, v76, v118
	v_dual_fmac_f32 v68, v46, v126 :: v_dual_fmac_f32 v69, v56, v126
	v_dual_fmac_f32 v78, v66, v126 :: v_dual_fmac_f32 v79, v76, v126
	v_dual_fmac_f32 v48, v47, v119 :: v_dual_fmac_f32 v49, v57, v119
	v_dual_fmac_f32 v58, v67, v119 :: v_dual_fmac_f32 v59, v77, v119
	v_dual_fmac_f32 v68, v47, v127 :: v_dual_fmac_f32 v69, v57, v127
	v_dual_fmac_f32 v78, v67, v127 :: v_dual_fmac_f32 v79, v77, v127
	v_dual_add_f32 v136, v136, v48 :: v_dual_add_f32 v137, v137, v49
	v_dual_add_f32 v138, v138, v58 :: v_dual_add_f32 v139, v139, v59
	v_dual_add_f32 v152, v152, v68 :: v_dual_add_f32 v153, v153, v69
	v_dual_add_f32 v154, v154, v78 :: v_dual_add_f32 v155, v155, v79
	buffer_load_b128 v[112:115], v3, s[24:27], s34 offen offset:2048
	buffer_load_b128 v[116:119], v3, s[24:27], s34 offen offset:2064
	buffer_load_b128 v[120:123], v3, s[24:27], s35 offen offset:2048
	buffer_load_b128 v[124:127], v3, s[24:27], s35 offen offset:2064
	s_wait_loadcnt 0xb
	v_dual_mul_f32 v48, v41, v81 :: v_dual_mul_f32 v49, v51, v81
	v_dual_mul_f32 v58, v61, v81 :: v_dual_mul_f32 v59, v71, v81
	s_wait_loadcnt 0x9
	v_dual_mul_f32 v68, v41, v89 :: v_dual_mul_f32 v69, v51, v89
	v_dual_mul_f32 v78, v61, v89 :: v_dual_mul_f32 v79, v71, v89
	v_dual_fmac_f32 v48, v40, v80 :: v_dual_fmac_f32 v49, v50, v80
	v_dual_fmac_f32 v58, v60, v80 :: v_dual_fmac_f32 v59, v70, v80
	v_dual_fmac_f32 v68, v40, v88 :: v_dual_fmac_f32 v69, v50, v88
	v_dual_fmac_f32 v78, v60, v88 :: v_dual_fmac_f32 v79, v70, v88
	v_dual_fmac_f32 v48, v42, v82 :: v_dual_fmac_f32 v49, v52, v82
	v_dual_fmac_f32 v58, v62, v82 :: v_dual_fmac_f32 v59, v72, v82
	v_dual_fmac_f32 v68, v42, v90 :: v_dual_fmac_f32 v69, v52, v90
	v_dual_fmac_f32 v78, v62, v90 :: v_dual_fmac_f32 v79, v72, v90
	v_dual_fmac_f32 v48, v43, v83 :: v_dual_fmac_f32 v49, v53, v83
	v_dual_fmac_f32 v58, v63, v83 :: v_dual_fmac_f32 v59, v73, v83
	v_dual_fmac_f32 v68, v43, v91 :: v_dual_fmac_f32 v69, v53, v91
	v_dual_fmac_f32 v78, v63, v91 :: v_dual_fmac_f32 v79, v73, v91
	v_dual_fmac_f32 v48, v44, v84 :: v_dual_fmac_f32 v49, v54, v84
	v_dual_fmac_f32 v58, v64, v84 :: v_dual_fmac_f32 v59, v74, v84
	s_wait_loadcnt 0x8
	v_dual_fmac_f32 v68, v44, v92 :: v_dual_fmac_f32 v69, v54, v92
	v_dual_fmac_f32 v78, v64, v92 :: v_dual_fmac_f32 v79, v74, v92
	v_dual_fmac_f32 v48, v45, v85 :: v_dual_fmac_f32 v49, v55, v85
	v_dual_fmac_f32 v58, v65, v85 :: v_dual_fmac_f32 v59, v75, v85
	v_dual_fmac_f32 v68, v45, v93 :: v_dual_fmac_f32 v69, v55, v93
	v_dual_fmac_f32 v78, v65, v93 :: v_dual_fmac_f32 v79, v75, v93
	v_dual_fmac_f32 v48, v46, v86 :: v_dual_fmac_f32 v49, v56, v86
	v_dual_fmac_f32 v58, v66, v86 :: v_dual_fmac_f32 v59, v76, v86
	v_dual_fmac_f32 v68, v46, v94 :: v_dual_fmac_f32 v69, v56, v94
	v_dual_fmac_f32 v78, v66, v94 :: v_dual_fmac_f32 v79, v76, v94
	v_dual_fmac_f32 v48, v47, v87 :: v_dual_fmac_f32 v49, v57, v87
	v_dual_fmac_f32 v58, v67, v87 :: v_dual_fmac_f32 v59, v77, v87
	v_dual_fmac_f32 v68, v47, v95 :: v_dual_fmac_f32 v69, v57, v95
	v_dual_fmac_f32 v78, v67, v95 :: v_dual_fmac_f32 v79, v77, v95
	v_dual_add_f32 v168, v168, v48 :: v_dual_add_f32 v169, v169, v49
	v_dual_add_f32 v170, v170, v58 :: v_dual_add_f32 v171, v171, v59
	v_dual_add_f32 v184, v184, v68 :: v_dual_add_f32 v185, v185, v69
	v_dual_add_f32 v186, v186, v78 :: v_dual_add_f32 v187, v187, v79
	buffer_load_b128 v[80:83], v3, s[24:27], s28 offen offset:3072
	buffer_load_b128 v[84:87], v3, s[24:27], s28 offen offset:3088
	buffer_load_b128 v[88:91], v3, s[24:27], s29 offen offset:3072
	buffer_load_b128 v[92:95], v3, s[24:27], s29 offen offset:3088
	s_wait_loadcnt 0xb
	v_dual_mul_f32 v48, v41, v97 :: v_dual_mul_f32 v49, v51, v97
	v_dual_mul_f32 v58, v61, v97 :: v_dual_mul_f32 v59, v71, v97
	s_wait_loadcnt 0x9
	v_dual_mul_f32 v68, v41, v105 :: v_dual_mul_f32 v69, v51, v105
	v_dual_mul_f32 v78, v61, v105 :: v_dual_mul_f32 v79, v71, v105
	v_dual_fmac_f32 v48, v40, v96 :: v_dual_fmac_f32 v49, v50, v96
	v_dual_fmac_f32 v58, v60, v96 :: v_dual_fmac_f32 v59, v70, v96
	v_dual_fmac_f32 v68, v40, v104 :: v_dual_fmac_f32 v69, v50, v104
	v_dual_fmac_f32 v78, v60, v104 :: v_dual_fmac_f32 v79, v70, v104
	v_dual_fmac_f32 v48, v42, v98 :: v_dual_fmac_f32 v49, v52, v98
	v_dual_fmac_f32 v58, v62, v98 :: v_dual_fmac_f32 v59, v72, v98
	v_dual_fmac_f32 v68, v42, v106 :: v_dual_fmac_f32 v69, v52, v106
	v_dual_fmac_f32 v78, v62, v106 :: v_dual_fmac_f32 v79, v72, v106
	v_dual_fmac_f32 v48, v43, v99 :: v_dual_fmac_f32 v49, v53, v99
	v_dual_fmac_f32 v58, v63, v99 :: v_dual_fmac_f32 v59, v73, v99
	v_dual_fmac_f32 v68, v43, v107 :: v_dual_fmac_f32 v69, v53, v107
	v_dual_fmac_f32 v78, v63, v107 :: v_dual_fmac_f32 v79, v73, v107
	v_dual_fmac_f32 v48, v44, v100 :: v_dual_fmac_f32 v49, v54, v100
	v_dual_fmac_f32 v58, v64, v100 :: v_dual_fmac_f32 v59, v74, v100
	s_wait_loadcnt 0x8
	v_dual_fmac_f32 v68, v44, v108 :: v_dual_fmac_f32 v69, v54, v108
	v_dual_fmac_f32 v78, v64, v108 :: v_dual_fmac_f32 v79, v74, v108
	v_dual_fmac_f32 v48, v45, v101 :: v_dual_fmac_f32 v49, v55, v101
	v_dual_fmac_f32 v58, v65, v101 :: v_dual_fmac_f32 v59, v75, v101
	v_dual_fmac_f32 v68, v45, v109 :: v_dual_fmac_f32 v69, v55, v109
	v_dual_fmac_f32 v78, v65, v109 :: v_dual_fmac_f32 v79, v75, v109
	v_dual_fmac_f32 v48, v46, v102 :: v_dual_fmac_f32 v49, v56, v102
	v_dual_fmac_f32 v58, v66, v102 :: v_dual_fmac_f32 v59, v76, v102
	v_dual_fmac_f32 v68, v46, v110 :: v_dual_fmac_f32 v69, v56, v110
	v_dual_fmac_f32 v78, v66, v110 :: v_dual_fmac_f32 v79, v76, v110
	v_dual_fmac_f32 v48, v47, v103 :: v_dual_fmac_f32 v49, v57, v103
	v_dual_fmac_f32 v58, v67, v103 :: v_dual_fmac_f32 v59, v77, v103
	v_dual_fmac_f32 v68, v47, v111 :: v_dual_fmac_f32 v69, v57, v111
	v_dual_fmac_f32 v78, v67, v111 :: v_dual_fmac_f32 v79, v77, v111
	v_dual_add_f32 v200, v200, v48 :: v_dual_add_f32 v201, v201, v49
	v_dual_add_f32 v202, v202, v58 :: v_dual_add_f32 v203, v203, v59
	v_dual_add_f32 v216, v216, v68 :: v_dual_add_f32 v217, v217, v69
	v_dual_add_f32 v218, v218, v78 :: v_dual_add_f32 v219, v219, v79
	buffer_load_b128 v[96:99], v3, s[24:27], s30 offen offset:3072
	buffer_load_b128 v[100:103], v3, s[24:27], s30 offen offset:3088
	buffer_load_b128 v[104:107], v3, s[24:27], s31 offen offset:3072
	buffer_load_b128 v[108:111], v3, s[24:27], s31 offen offset:3088
	s_wait_loadcnt 0xb
	v_dual_mul_f32 v48, v41, v113 :: v_dual_mul_f32 v49, v51, v113
	v_dual_mul_f32 v58, v61, v113 :: v_dual_mul_f32 v59, v71, v113
	s_wait_loadcnt 0x9
	v_dual_mul_f32 v68, v41, v121 :: v_dual_mul_f32 v69, v51, v121
	v_dual_mul_f32 v78, v61, v121 :: v_dual_mul_f32 v79, v71, v121
	v_dual_fmac_f32 v48, v40, v112 :: v_dual_fmac_f32 v49, v50, v112
	v_dual_fmac_f32 v58, v60, v112 :: v_dual_fmac_f32 v59, v70, v112
	v_dual_fmac_f32 v68, v40, v120 :: v_dual_fmac_f32 v69, v50, v120
	v_dual_fmac_f32 v78, v60, v120 :: v_dual_fmac_f32 v79, v70, v120
	v_dual_fmac_f32 v48, v42, v114 :: v_dual_fmac_f32 v49, v52, v114
	v_dual_fmac_f32 v58, v62, v114 :: v_dual_fmac_f32 v59, v72, v114
	v_dual_fmac_f32 v68, v42, v122 :: v_dual_fmac_f32 v69, v52, v122
	v_dual_fmac_f32 v78, v62, v122 :: v_dual_fmac_f32 v79, v72, v122
	v_dual_fmac_f32 v48, v43, v115 :: v_dual_fmac_f32 v49, v53, v115
	v_dual_fmac_f32 v58, v63, v115 :: v_dual_fmac_f32 v59, v73, v115
	v_dual_fmac_f32 v68, v43, v123 :: v_dual_fmac_f32 v69, v53, v123
	v_dual_fmac_f32 v78, v63, v123 :: v_dual_fmac_f32 v79, v73, v123
	v_dual_fmac_f32 v48, v44, v116 :: v_dual_fmac_f32 v49, v54, v116
	v_dual_fmac_f32 v58, v64, v116 :: v_dual_fmac_f32 v59, v74, v116
	s_wait_loadcnt 0x8
	v_dual_fmac_f32 v68, v44, v124 :: v_dual_fmac_f32 v69, v54, v124
	v_dual_fmac_f32 v78, v64, v124 :: v_dual_fmac_f32 v79, v74, v124
	v_dual_fmac_f32 v48, v45, v117 :: v_dual_fmac_f32 v49, v55, v117
	v_dual_fmac_f32 v58, v65, v117 :: v_dual_fmac_f32 v59, v75, v117
	v_dual_fmac_f32 v68, v45, v125 :: v_dual_fmac_f32 v69, v55, v125
	v_dual_fmac_f32 v78, v65, v125 :: v_dual_fmac_f32 v79, v75, v125
	v_dual_fmac_f32 v48, v46, v118 :: v_dual_fmac_f32 v49, v56, v118
	v_dual_fmac_f32 v58, v66, v118 :: v_dual_fmac_f32 v59, v76, v118
	v_dual_fmac_f32 v68, v46, v126 :: v_dual_fmac_f32 v69, v56, v126
	v_dual_fmac_f32 v78, v66, v126 :: v_dual_fmac_f32 v79, v76, v126
	v_dual_fmac_f32 v48, v47, v119 :: v_dual_fmac_f32 v49, v57, v119
	v_dual_fmac_f32 v58, v67, v119 :: v_dual_fmac_f32 v59, v77, v119
	v_dual_fmac_f32 v68, v47, v127 :: v_dual_fmac_f32 v69, v57, v127
	v_dual_fmac_f32 v78, v67, v127 :: v_dual_fmac_f32 v79, v77, v127
	v_dual_add_f32 v232, v232, v48 :: v_dual_add_f32 v233, v233, v49
	v_dual_add_f32 v234, v234, v58 :: v_dual_add_f32 v235, v235, v59
	v_dual_add_f32 v248, v248, v68 :: v_dual_add_f32 v249, v249, v69
	v_dual_add_f32 v250, v250, v78 :: v_dual_add_f32 v251, v251, v79
	v_and_b32_e32 v46, 0xf0f0f0f, v36
	v_and_b32_e32 v56, 0xf0f0f0f, v37
	v_and_b32_e32 v66, 0xf0f0f0f, v38
	v_and_b32_e32 v76, 0xf0f0f0f, v39
	v_lshrrev_b32_e32 v47, 4, v36
	v_lshrrev_b32_e32 v57, 4, v37
	v_lshrrev_b32_e32 v67, 4, v38
	v_lshrrev_b32_e32 v77, 4, v39
	v_and_b32_e32 v47, 0xf0f0f0f, v47
	v_and_b32_e32 v57, 0xf0f0f0f, v57
	v_and_b32_e32 v67, 0xf0f0f0f, v67
	v_and_b32_e32 v77, 0xf0f0f0f, v77
	v_cvt_f32_ubyte0_e32 v40, v46
	v_cvt_f32_ubyte0_e32 v50, v56
	v_cvt_f32_ubyte0_e32 v60, v66
	v_cvt_f32_ubyte0_e32 v70, v76
	v_cvt_f32_ubyte1_e32 v42, v46
	v_cvt_f32_ubyte1_e32 v52, v56
	v_cvt_f32_ubyte1_e32 v62, v66
	v_cvt_f32_ubyte1_e32 v72, v76
	v_cvt_f32_ubyte2_e32 v44, v46
	v_cvt_f32_ubyte2_e32 v54, v56
	v_cvt_f32_ubyte2_e32 v64, v66
	v_cvt_f32_ubyte2_e32 v74, v76
	v_cvt_f32_ubyte3_e32 v46, v46
	v_cvt_f32_ubyte3_e32 v56, v56
	v_cvt_f32_ubyte3_e32 v66, v66
	v_cvt_f32_ubyte3_e32 v76, v76
	v_cvt_f32_ubyte0_e32 v41, v47
	v_cvt_f32_ubyte0_e32 v51, v57
	v_cvt_f32_ubyte0_e32 v61, v67
	v_cvt_f32_ubyte0_e32 v71, v77
	v_cvt_f32_ubyte1_e32 v43, v47
	v_cvt_f32_ubyte1_e32 v53, v57
	v_cvt_f32_ubyte1_e32 v63, v67
	v_cvt_f32_ubyte1_e32 v73, v77
	v_cvt_f32_ubyte2_e32 v45, v47
	v_cvt_f32_ubyte2_e32 v55, v57
	v_cvt_f32_ubyte2_e32 v65, v67
	v_cvt_f32_ubyte2_e32 v75, v77
	v_cvt_f32_ubyte3_e32 v47, v47
	v_cvt_f32_ubyte3_e32 v57, v57
	v_cvt_f32_ubyte3_e32 v67, v67
	v_cvt_f32_ubyte3_e32 v77, v77
	v_fma_mix_f32 v40, v20, v40, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v21, v50, v21 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v22, v60, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v70, v23, v70, v23 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v41, v20, v41, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v21, v51, v21 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v22, v61, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v71, v23, v71, v23 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v42, v20, v42, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v52, v21, v52, v21 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v62, v22, v62, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v72, v23, v72, v23 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v43, v20, v43, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v53, v21, v53, v21 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v63, v22, v63, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v73, v23, v73, v23 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v44, v20, v44, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v54, v21, v54, v21 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v64, v22, v64, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v74, v23, v74, v23 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v20, v45, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v21, v55, v21 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v65, v22, v65, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v75, v23, v75, v23 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v20, v46, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v21, v56, v21 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v66, v22, v66, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v76, v23, v76, v23 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v20, v47, v20 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v21, v57, v21 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v67, v22, v67, v22 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v77, v23, v77, v23 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	buffer_load_b128 v[112:115], v3, s[24:27], s32 offen offset:3072
	buffer_load_b128 v[116:119], v3, s[24:27], s32 offen offset:3088
	buffer_load_b128 v[120:123], v3, s[24:27], s33 offen offset:3072
	buffer_load_b128 v[124:127], v3, s[24:27], s33 offen offset:3088
	s_wait_loadcnt 0xb
	v_dual_mul_f32 v48, v41, v81 :: v_dual_mul_f32 v49, v51, v81
	v_dual_mul_f32 v58, v61, v81 :: v_dual_mul_f32 v59, v71, v81
	s_wait_loadcnt 0x9
	v_dual_mul_f32 v68, v41, v89 :: v_dual_mul_f32 v69, v51, v89
	v_dual_mul_f32 v78, v61, v89 :: v_dual_mul_f32 v79, v71, v89
	v_dual_fmac_f32 v48, v40, v80 :: v_dual_fmac_f32 v49, v50, v80
	v_dual_fmac_f32 v58, v60, v80 :: v_dual_fmac_f32 v59, v70, v80
	v_dual_fmac_f32 v68, v40, v88 :: v_dual_fmac_f32 v69, v50, v88
	v_dual_fmac_f32 v78, v60, v88 :: v_dual_fmac_f32 v79, v70, v88
	v_dual_fmac_f32 v48, v42, v82 :: v_dual_fmac_f32 v49, v52, v82
	v_dual_fmac_f32 v58, v62, v82 :: v_dual_fmac_f32 v59, v72, v82
	v_dual_fmac_f32 v68, v42, v90 :: v_dual_fmac_f32 v69, v52, v90
	v_dual_fmac_f32 v78, v62, v90 :: v_dual_fmac_f32 v79, v72, v90
	v_dual_fmac_f32 v48, v43, v83 :: v_dual_fmac_f32 v49, v53, v83
	v_dual_fmac_f32 v58, v63, v83 :: v_dual_fmac_f32 v59, v73, v83
	v_dual_fmac_f32 v68, v43, v91 :: v_dual_fmac_f32 v69, v53, v91
	v_dual_fmac_f32 v78, v63, v91 :: v_dual_fmac_f32 v79, v73, v91
	v_dual_fmac_f32 v48, v44, v84 :: v_dual_fmac_f32 v49, v54, v84
	v_dual_fmac_f32 v58, v64, v84 :: v_dual_fmac_f32 v59, v74, v84
	s_wait_loadcnt 0x8
	v_dual_fmac_f32 v68, v44, v92 :: v_dual_fmac_f32 v69, v54, v92
	v_dual_fmac_f32 v78, v64, v92 :: v_dual_fmac_f32 v79, v74, v92
	v_dual_fmac_f32 v48, v45, v85 :: v_dual_fmac_f32 v49, v55, v85
	v_dual_fmac_f32 v58, v65, v85 :: v_dual_fmac_f32 v59, v75, v85
	v_dual_fmac_f32 v68, v45, v93 :: v_dual_fmac_f32 v69, v55, v93
	v_dual_fmac_f32 v78, v65, v93 :: v_dual_fmac_f32 v79, v75, v93
	v_dual_fmac_f32 v48, v46, v86 :: v_dual_fmac_f32 v49, v56, v86
	v_dual_fmac_f32 v58, v66, v86 :: v_dual_fmac_f32 v59, v76, v86
	v_dual_fmac_f32 v68, v46, v94 :: v_dual_fmac_f32 v69, v56, v94
	v_dual_fmac_f32 v78, v66, v94 :: v_dual_fmac_f32 v79, v76, v94
	v_dual_fmac_f32 v48, v47, v87 :: v_dual_fmac_f32 v49, v57, v87
	v_dual_fmac_f32 v58, v67, v87 :: v_dual_fmac_f32 v59, v77, v87
	v_dual_fmac_f32 v68, v47, v95 :: v_dual_fmac_f32 v69, v57, v95
	v_dual_fmac_f32 v78, v67, v95 :: v_dual_fmac_f32 v79, v77, v95
	v_dual_add_f32 v140, v140, v48 :: v_dual_add_f32 v141, v141, v49
	v_dual_add_f32 v142, v142, v58 :: v_dual_add_f32 v143, v143, v59
	v_dual_add_f32 v156, v156, v68 :: v_dual_add_f32 v157, v157, v69
	v_dual_add_f32 v158, v158, v78 :: v_dual_add_f32 v159, v159, v79
	buffer_load_b128 v[80:83], v3, s[24:27], s34 offen offset:3072
	buffer_load_b128 v[84:87], v3, s[24:27], s34 offen offset:3088
	buffer_load_b128 v[88:91], v3, s[24:27], s35 offen offset:3072
	buffer_load_b128 v[92:95], v3, s[24:27], s35 offen offset:3088
	s_wait_loadcnt 0xb
	v_dual_mul_f32 v48, v41, v97 :: v_dual_mul_f32 v49, v51, v97
	v_dual_mul_f32 v58, v61, v97 :: v_dual_mul_f32 v59, v71, v97
	s_wait_loadcnt 0x9
	v_dual_mul_f32 v68, v41, v105 :: v_dual_mul_f32 v69, v51, v105
	v_dual_mul_f32 v78, v61, v105 :: v_dual_mul_f32 v79, v71, v105
	v_dual_fmac_f32 v48, v40, v96 :: v_dual_fmac_f32 v49, v50, v96
	v_dual_fmac_f32 v58, v60, v96 :: v_dual_fmac_f32 v59, v70, v96
	v_dual_fmac_f32 v68, v40, v104 :: v_dual_fmac_f32 v69, v50, v104
	v_dual_fmac_f32 v78, v60, v104 :: v_dual_fmac_f32 v79, v70, v104
	v_dual_fmac_f32 v48, v42, v98 :: v_dual_fmac_f32 v49, v52, v98
	v_dual_fmac_f32 v58, v62, v98 :: v_dual_fmac_f32 v59, v72, v98
	v_dual_fmac_f32 v68, v42, v106 :: v_dual_fmac_f32 v69, v52, v106
	v_dual_fmac_f32 v78, v62, v106 :: v_dual_fmac_f32 v79, v72, v106
	v_dual_fmac_f32 v48, v43, v99 :: v_dual_fmac_f32 v49, v53, v99
	v_dual_fmac_f32 v58, v63, v99 :: v_dual_fmac_f32 v59, v73, v99
	v_dual_fmac_f32 v68, v43, v107 :: v_dual_fmac_f32 v69, v53, v107
	v_dual_fmac_f32 v78, v63, v107 :: v_dual_fmac_f32 v79, v73, v107
	v_dual_fmac_f32 v48, v44, v100 :: v_dual_fmac_f32 v49, v54, v100
	v_dual_fmac_f32 v58, v64, v100 :: v_dual_fmac_f32 v59, v74, v100
	s_wait_loadcnt 0x8
	v_dual_fmac_f32 v68, v44, v108 :: v_dual_fmac_f32 v69, v54, v108
	v_dual_fmac_f32 v78, v64, v108 :: v_dual_fmac_f32 v79, v74, v108
	v_dual_fmac_f32 v48, v45, v101 :: v_dual_fmac_f32 v49, v55, v101
	v_dual_fmac_f32 v58, v65, v101 :: v_dual_fmac_f32 v59, v75, v101
	v_dual_fmac_f32 v68, v45, v109 :: v_dual_fmac_f32 v69, v55, v109
	v_dual_fmac_f32 v78, v65, v109 :: v_dual_fmac_f32 v79, v75, v109
	v_dual_fmac_f32 v48, v46, v102 :: v_dual_fmac_f32 v49, v56, v102
	v_dual_fmac_f32 v58, v66, v102 :: v_dual_fmac_f32 v59, v76, v102
	v_dual_fmac_f32 v68, v46, v110 :: v_dual_fmac_f32 v69, v56, v110
	v_dual_fmac_f32 v78, v66, v110 :: v_dual_fmac_f32 v79, v76, v110
	v_dual_fmac_f32 v48, v47, v103 :: v_dual_fmac_f32 v49, v57, v103
	v_dual_fmac_f32 v58, v67, v103 :: v_dual_fmac_f32 v59, v77, v103
	v_dual_fmac_f32 v68, v47, v111 :: v_dual_fmac_f32 v69, v57, v111
	v_dual_fmac_f32 v78, v67, v111 :: v_dual_fmac_f32 v79, v77, v111
	v_dual_add_f32 v172, v172, v48 :: v_dual_add_f32 v173, v173, v49
	v_dual_add_f32 v174, v174, v58 :: v_dual_add_f32 v175, v175, v59
	v_dual_add_f32 v188, v188, v68 :: v_dual_add_f32 v189, v189, v69
	v_dual_add_f32 v190, v190, v78 :: v_dual_add_f32 v191, v191, v79
	s_wait_loadcnt 0x7
	v_dual_mul_f32 v48, v41, v113 :: v_dual_mul_f32 v49, v51, v113
	v_dual_mul_f32 v58, v61, v113 :: v_dual_mul_f32 v59, v71, v113
	s_wait_loadcnt 0x5
	v_dual_mul_f32 v68, v41, v121 :: v_dual_mul_f32 v69, v51, v121
	v_dual_mul_f32 v78, v61, v121 :: v_dual_mul_f32 v79, v71, v121
	v_dual_fmac_f32 v48, v40, v112 :: v_dual_fmac_f32 v49, v50, v112
	v_dual_fmac_f32 v58, v60, v112 :: v_dual_fmac_f32 v59, v70, v112
	v_dual_fmac_f32 v68, v40, v120 :: v_dual_fmac_f32 v69, v50, v120
	v_dual_fmac_f32 v78, v60, v120 :: v_dual_fmac_f32 v79, v70, v120
	v_dual_fmac_f32 v48, v42, v114 :: v_dual_fmac_f32 v49, v52, v114
	v_dual_fmac_f32 v58, v62, v114 :: v_dual_fmac_f32 v59, v72, v114
	v_dual_fmac_f32 v68, v42, v122 :: v_dual_fmac_f32 v69, v52, v122
	v_dual_fmac_f32 v78, v62, v122 :: v_dual_fmac_f32 v79, v72, v122
	v_dual_fmac_f32 v48, v43, v115 :: v_dual_fmac_f32 v49, v53, v115
	v_dual_fmac_f32 v58, v63, v115 :: v_dual_fmac_f32 v59, v73, v115
	v_dual_fmac_f32 v68, v43, v123 :: v_dual_fmac_f32 v69, v53, v123
	v_dual_fmac_f32 v78, v63, v123 :: v_dual_fmac_f32 v79, v73, v123
	v_dual_fmac_f32 v48, v44, v116 :: v_dual_fmac_f32 v49, v54, v116
	v_dual_fmac_f32 v58, v64, v116 :: v_dual_fmac_f32 v59, v74, v116
	s_wait_loadcnt 0x4
	v_dual_fmac_f32 v68, v44, v124 :: v_dual_fmac_f32 v69, v54, v124
	v_dual_fmac_f32 v78, v64, v124 :: v_dual_fmac_f32 v79, v74, v124
	v_dual_fmac_f32 v48, v45, v117 :: v_dual_fmac_f32 v49, v55, v117
	v_dual_fmac_f32 v58, v65, v117 :: v_dual_fmac_f32 v59, v75, v117
	v_dual_fmac_f32 v68, v45, v125 :: v_dual_fmac_f32 v69, v55, v125
	v_dual_fmac_f32 v78, v65, v125 :: v_dual_fmac_f32 v79, v75, v125
	v_dual_fmac_f32 v48, v46, v118 :: v_dual_fmac_f32 v49, v56, v118
	v_dual_fmac_f32 v58, v66, v118 :: v_dual_fmac_f32 v59, v76, v118
	v_dual_fmac_f32 v68, v46, v126 :: v_dual_fmac_f32 v69, v56, v126
	v_dual_fmac_f32 v78, v66, v126 :: v_dual_fmac_f32 v79, v76, v126
	v_dual_fmac_f32 v48, v47, v119 :: v_dual_fmac_f32 v49, v57, v119
	v_dual_fmac_f32 v58, v67, v119 :: v_dual_fmac_f32 v59, v77, v119
	v_dual_fmac_f32 v68, v47, v127 :: v_dual_fmac_f32 v69, v57, v127
	v_dual_fmac_f32 v78, v67, v127 :: v_dual_fmac_f32 v79, v77, v127
	v_dual_add_f32 v204, v204, v48 :: v_dual_add_f32 v205, v205, v49
	v_dual_add_f32 v206, v206, v58 :: v_dual_add_f32 v207, v207, v59
	v_dual_add_f32 v220, v220, v68 :: v_dual_add_f32 v221, v221, v69
	v_dual_add_f32 v222, v222, v78 :: v_dual_add_f32 v223, v223, v79
	s_wait_loadcnt 0x3
	v_dual_mul_f32 v48, v41, v81 :: v_dual_mul_f32 v49, v51, v81
	v_dual_mul_f32 v58, v61, v81 :: v_dual_mul_f32 v59, v71, v81
	s_wait_loadcnt 0x1
	v_dual_mul_f32 v68, v41, v89 :: v_dual_mul_f32 v69, v51, v89
	v_dual_mul_f32 v78, v61, v89 :: v_dual_mul_f32 v79, v71, v89
	v_dual_fmac_f32 v48, v40, v80 :: v_dual_fmac_f32 v49, v50, v80
	v_dual_fmac_f32 v58, v60, v80 :: v_dual_fmac_f32 v59, v70, v80
	v_dual_fmac_f32 v68, v40, v88 :: v_dual_fmac_f32 v69, v50, v88
	v_dual_fmac_f32 v78, v60, v88 :: v_dual_fmac_f32 v79, v70, v88
	v_dual_fmac_f32 v48, v42, v82 :: v_dual_fmac_f32 v49, v52, v82
	v_dual_fmac_f32 v58, v62, v82 :: v_dual_fmac_f32 v59, v72, v82
	v_dual_fmac_f32 v68, v42, v90 :: v_dual_fmac_f32 v69, v52, v90
	v_dual_fmac_f32 v78, v62, v90 :: v_dual_fmac_f32 v79, v72, v90
	v_dual_fmac_f32 v48, v43, v83 :: v_dual_fmac_f32 v49, v53, v83
	v_dual_fmac_f32 v58, v63, v83 :: v_dual_fmac_f32 v59, v73, v83
	v_dual_fmac_f32 v68, v43, v91 :: v_dual_fmac_f32 v69, v53, v91
	v_dual_fmac_f32 v78, v63, v91 :: v_dual_fmac_f32 v79, v73, v91
	v_dual_fmac_f32 v48, v44, v84 :: v_dual_fmac_f32 v49, v54, v84
	v_dual_fmac_f32 v58, v64, v84 :: v_dual_fmac_f32 v59, v74, v84
	s_wait_loadcnt 0x0
	v_dual_fmac_f32 v68, v44, v92 :: v_dual_fmac_f32 v69, v54, v92
	v_dual_fmac_f32 v78, v64, v92 :: v_dual_fmac_f32 v79, v74, v92
	v_dual_fmac_f32 v48, v45, v85 :: v_dual_fmac_f32 v49, v55, v85
	v_dual_fmac_f32 v58, v65, v85 :: v_dual_fmac_f32 v59, v75, v85
	v_dual_fmac_f32 v68, v45, v93 :: v_dual_fmac_f32 v69, v55, v93
	v_dual_fmac_f32 v78, v65, v93 :: v_dual_fmac_f32 v79, v75, v93
	v_dual_fmac_f32 v48, v46, v86 :: v_dual_fmac_f32 v49, v56, v86
	v_dual_fmac_f32 v58, v66, v86 :: v_dual_fmac_f32 v59, v76, v86
	v_dual_fmac_f32 v68, v46, v94 :: v_dual_fmac_f32 v69, v56, v94
	v_dual_fmac_f32 v78, v66, v94 :: v_dual_fmac_f32 v79, v76, v94
	v_dual_fmac_f32 v48, v47, v87 :: v_dual_fmac_f32 v49, v57, v87
	v_dual_fmac_f32 v58, v67, v87 :: v_dual_fmac_f32 v59, v77, v87
	v_dual_fmac_f32 v68, v47, v95 :: v_dual_fmac_f32 v69, v57, v95
	v_dual_fmac_f32 v78, v67, v95 :: v_dual_fmac_f32 v79, v77, v95
	v_dual_add_f32 v236, v236, v48 :: v_dual_add_f32 v237, v237, v49
	v_dual_add_f32 v238, v238, v58 :: v_dual_add_f32 v239, v239, v59
	v_dual_add_f32 v252, v252, v68 :: v_dual_add_f32 v253, v253, v69
	v_dual_add_f32 v254, v254, v78 :: v_dual_add_f32 v255, v255, v79
	s_add_co_i32 s16, s16, 0x220
	s_add_co_i32 s17, s17, 0x220
	s_add_co_i32 s18, s18, 0x220
	s_add_co_i32 s19, s19, 0x220
	s_add_co_i32 s28, s28, 0x1000
	s_add_co_i32 s29, s29, 0x1000
	s_add_co_i32 s30, s30, 0x1000
	s_add_co_i32 s31, s31, 0x1000
	s_add_co_i32 s32, s32, 0x1000
	s_add_co_i32 s33, s33, 0x1000
	s_add_co_i32 s34, s34, 0x1000
	s_add_co_i32 s35, s35, 0x1000
	s_add_co_i32 s36, s36, -1
	s_cmp_lg_u32 s36, 0
	s_cbranch_scc1 .Lpx_b8_quad
	.Lpx_b8_tail:
	s_cmp_gt_u32 s37, 0
	s_cbranch_scc0 .Lpx_b8_fold
	buffer_load_b32 v8, v1, s[20:23], s16 offen
	buffer_load_b32 v9, v1, s[20:23], s17 offen
	buffer_load_b32 v10, v1, s[20:23], s18 offen
	buffer_load_b32 v11, v1, s[20:23], s19 offen
	buffer_load_b32 v24, v2, s[20:23], s16 offen offset:8
	buffer_load_b32 v25, v2, s[20:23], s17 offen offset:8
	buffer_load_b32 v26, v2, s[20:23], s18 offen offset:8
	buffer_load_b32 v27, v2, s[20:23], s19 offen offset:8
	buffer_load_b128 v[80:83], v3, s[24:27], s28 offen
	buffer_load_b128 v[84:87], v3, s[24:27], s28 offen offset:16
	buffer_load_b128 v[88:91], v3, s[24:27], s29 offen
	buffer_load_b128 v[92:95], v3, s[24:27], s29 offen offset:16
	buffer_load_b128 v[96:99], v3, s[24:27], s30 offen
	buffer_load_b128 v[100:103], v3, s[24:27], s30 offen offset:16
	buffer_load_b128 v[104:107], v3, s[24:27], s31 offen
	buffer_load_b128 v[108:111], v3, s[24:27], s31 offen offset:16
	buffer_load_b128 v[112:115], v3, s[24:27], s32 offen
	buffer_load_b128 v[116:119], v3, s[24:27], s32 offen offset:16
	buffer_load_b128 v[120:123], v3, s[24:27], s33 offen
	buffer_load_b128 v[124:127], v3, s[24:27], s33 offen offset:16
	s_wait_loadcnt 0xf
	v_and_b32_e32 v46, 0xf0f0f0f, v24
	s_wait_loadcnt 0xe
	v_and_b32_e32 v56, 0xf0f0f0f, v25
	s_wait_loadcnt 0xd
	v_and_b32_e32 v66, 0xf0f0f0f, v26
	s_wait_loadcnt 0xc
	v_and_b32_e32 v76, 0xf0f0f0f, v27
	v_lshrrev_b32_e32 v47, 4, v24
	v_lshrrev_b32_e32 v57, 4, v25
	v_lshrrev_b32_e32 v67, 4, v26
	v_lshrrev_b32_e32 v77, 4, v27
	v_and_b32_e32 v47, 0xf0f0f0f, v47
	v_and_b32_e32 v57, 0xf0f0f0f, v57
	v_and_b32_e32 v67, 0xf0f0f0f, v67
	v_and_b32_e32 v77, 0xf0f0f0f, v77
	v_cvt_f32_ubyte0_e32 v40, v46
	v_cvt_f32_ubyte0_e32 v50, v56
	v_cvt_f32_ubyte0_e32 v60, v66
	v_cvt_f32_ubyte0_e32 v70, v76
	v_cvt_f32_ubyte1_e32 v42, v46
	v_cvt_f32_ubyte1_e32 v52, v56
	v_cvt_f32_ubyte1_e32 v62, v66
	v_cvt_f32_ubyte1_e32 v72, v76
	v_cvt_f32_ubyte2_e32 v44, v46
	v_cvt_f32_ubyte2_e32 v54, v56
	v_cvt_f32_ubyte2_e32 v64, v66
	v_cvt_f32_ubyte2_e32 v74, v76
	v_cvt_f32_ubyte3_e32 v46, v46
	v_cvt_f32_ubyte3_e32 v56, v56
	v_cvt_f32_ubyte3_e32 v66, v66
	v_cvt_f32_ubyte3_e32 v76, v76
	v_cvt_f32_ubyte0_e32 v41, v47
	v_cvt_f32_ubyte0_e32 v51, v57
	v_cvt_f32_ubyte0_e32 v61, v67
	v_cvt_f32_ubyte0_e32 v71, v77
	v_cvt_f32_ubyte1_e32 v43, v47
	v_cvt_f32_ubyte1_e32 v53, v57
	v_cvt_f32_ubyte1_e32 v63, v67
	v_cvt_f32_ubyte1_e32 v73, v77
	v_cvt_f32_ubyte2_e32 v45, v47
	v_cvt_f32_ubyte2_e32 v55, v57
	v_cvt_f32_ubyte2_e32 v65, v67
	v_cvt_f32_ubyte2_e32 v75, v77
	v_cvt_f32_ubyte3_e32 v47, v47
	v_cvt_f32_ubyte3_e32 v57, v57
	v_cvt_f32_ubyte3_e32 v67, v67
	v_cvt_f32_ubyte3_e32 v77, v77
	v_fma_mix_f32 v40, v8, v40, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v9, v50, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v10, v60, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v70, v11, v70, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v41, v8, v41, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v9, v51, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v10, v61, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v71, v11, v71, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v42, v8, v42, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v52, v9, v52, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v62, v10, v62, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v72, v11, v72, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v43, v8, v43, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v53, v9, v53, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v63, v10, v63, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v73, v11, v73, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v44, v8, v44, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v54, v9, v54, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v64, v10, v64, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v74, v11, v74, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v8, v45, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v9, v55, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v65, v10, v65, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v75, v11, v75, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v8, v46, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v9, v56, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v66, v10, v66, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v76, v11, v76, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v8, v47, v8 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v9, v57, v9 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v67, v10, v67, v10 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v77, v11, v77, v11 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xb
	v_dual_mul_f32 v48, v41, v81 :: v_dual_mul_f32 v49, v51, v81
	v_dual_mul_f32 v58, v61, v81 :: v_dual_mul_f32 v59, v71, v81
	s_wait_loadcnt 0x9
	v_dual_mul_f32 v68, v41, v89 :: v_dual_mul_f32 v69, v51, v89
	v_dual_mul_f32 v78, v61, v89 :: v_dual_mul_f32 v79, v71, v89
	v_dual_fmac_f32 v48, v40, v80 :: v_dual_fmac_f32 v49, v50, v80
	v_dual_fmac_f32 v58, v60, v80 :: v_dual_fmac_f32 v59, v70, v80
	v_dual_fmac_f32 v68, v40, v88 :: v_dual_fmac_f32 v69, v50, v88
	v_dual_fmac_f32 v78, v60, v88 :: v_dual_fmac_f32 v79, v70, v88
	v_dual_fmac_f32 v48, v42, v82 :: v_dual_fmac_f32 v49, v52, v82
	v_dual_fmac_f32 v58, v62, v82 :: v_dual_fmac_f32 v59, v72, v82
	v_dual_fmac_f32 v68, v42, v90 :: v_dual_fmac_f32 v69, v52, v90
	v_dual_fmac_f32 v78, v62, v90 :: v_dual_fmac_f32 v79, v72, v90
	v_dual_fmac_f32 v48, v43, v83 :: v_dual_fmac_f32 v49, v53, v83
	v_dual_fmac_f32 v58, v63, v83 :: v_dual_fmac_f32 v59, v73, v83
	v_dual_fmac_f32 v68, v43, v91 :: v_dual_fmac_f32 v69, v53, v91
	v_dual_fmac_f32 v78, v63, v91 :: v_dual_fmac_f32 v79, v73, v91
	v_dual_fmac_f32 v48, v44, v84 :: v_dual_fmac_f32 v49, v54, v84
	v_dual_fmac_f32 v58, v64, v84 :: v_dual_fmac_f32 v59, v74, v84
	s_wait_loadcnt 0x8
	v_dual_fmac_f32 v68, v44, v92 :: v_dual_fmac_f32 v69, v54, v92
	v_dual_fmac_f32 v78, v64, v92 :: v_dual_fmac_f32 v79, v74, v92
	v_dual_fmac_f32 v48, v45, v85 :: v_dual_fmac_f32 v49, v55, v85
	v_dual_fmac_f32 v58, v65, v85 :: v_dual_fmac_f32 v59, v75, v85
	v_dual_fmac_f32 v68, v45, v93 :: v_dual_fmac_f32 v69, v55, v93
	v_dual_fmac_f32 v78, v65, v93 :: v_dual_fmac_f32 v79, v75, v93
	v_dual_fmac_f32 v48, v46, v86 :: v_dual_fmac_f32 v49, v56, v86
	v_dual_fmac_f32 v58, v66, v86 :: v_dual_fmac_f32 v59, v76, v86
	v_dual_fmac_f32 v68, v46, v94 :: v_dual_fmac_f32 v69, v56, v94
	v_dual_fmac_f32 v78, v66, v94 :: v_dual_fmac_f32 v79, v76, v94
	v_dual_fmac_f32 v48, v47, v87 :: v_dual_fmac_f32 v49, v57, v87
	v_dual_fmac_f32 v58, v67, v87 :: v_dual_fmac_f32 v59, v77, v87
	v_dual_fmac_f32 v68, v47, v95 :: v_dual_fmac_f32 v69, v57, v95
	v_dual_fmac_f32 v78, v67, v95 :: v_dual_fmac_f32 v79, v77, v95
	v_dual_add_f32 v128, v128, v48 :: v_dual_add_f32 v129, v129, v49
	v_dual_add_f32 v130, v130, v58 :: v_dual_add_f32 v131, v131, v59
	v_dual_add_f32 v144, v144, v68 :: v_dual_add_f32 v145, v145, v69
	v_dual_add_f32 v146, v146, v78 :: v_dual_add_f32 v147, v147, v79
	buffer_load_b128 v[80:83], v3, s[24:27], s34 offen
	buffer_load_b128 v[84:87], v3, s[24:27], s34 offen offset:16
	buffer_load_b128 v[88:91], v3, s[24:27], s35 offen
	buffer_load_b128 v[92:95], v3, s[24:27], s35 offen offset:16
	s_wait_loadcnt 0xb
	v_dual_mul_f32 v48, v41, v97 :: v_dual_mul_f32 v49, v51, v97
	v_dual_mul_f32 v58, v61, v97 :: v_dual_mul_f32 v59, v71, v97
	s_wait_loadcnt 0x9
	v_dual_mul_f32 v68, v41, v105 :: v_dual_mul_f32 v69, v51, v105
	v_dual_mul_f32 v78, v61, v105 :: v_dual_mul_f32 v79, v71, v105
	v_dual_fmac_f32 v48, v40, v96 :: v_dual_fmac_f32 v49, v50, v96
	v_dual_fmac_f32 v58, v60, v96 :: v_dual_fmac_f32 v59, v70, v96
	v_dual_fmac_f32 v68, v40, v104 :: v_dual_fmac_f32 v69, v50, v104
	v_dual_fmac_f32 v78, v60, v104 :: v_dual_fmac_f32 v79, v70, v104
	v_dual_fmac_f32 v48, v42, v98 :: v_dual_fmac_f32 v49, v52, v98
	v_dual_fmac_f32 v58, v62, v98 :: v_dual_fmac_f32 v59, v72, v98
	v_dual_fmac_f32 v68, v42, v106 :: v_dual_fmac_f32 v69, v52, v106
	v_dual_fmac_f32 v78, v62, v106 :: v_dual_fmac_f32 v79, v72, v106
	v_dual_fmac_f32 v48, v43, v99 :: v_dual_fmac_f32 v49, v53, v99
	v_dual_fmac_f32 v58, v63, v99 :: v_dual_fmac_f32 v59, v73, v99
	v_dual_fmac_f32 v68, v43, v107 :: v_dual_fmac_f32 v69, v53, v107
	v_dual_fmac_f32 v78, v63, v107 :: v_dual_fmac_f32 v79, v73, v107
	v_dual_fmac_f32 v48, v44, v100 :: v_dual_fmac_f32 v49, v54, v100
	v_dual_fmac_f32 v58, v64, v100 :: v_dual_fmac_f32 v59, v74, v100
	s_wait_loadcnt 0x8
	v_dual_fmac_f32 v68, v44, v108 :: v_dual_fmac_f32 v69, v54, v108
	v_dual_fmac_f32 v78, v64, v108 :: v_dual_fmac_f32 v79, v74, v108
	v_dual_fmac_f32 v48, v45, v101 :: v_dual_fmac_f32 v49, v55, v101
	v_dual_fmac_f32 v58, v65, v101 :: v_dual_fmac_f32 v59, v75, v101
	v_dual_fmac_f32 v68, v45, v109 :: v_dual_fmac_f32 v69, v55, v109
	v_dual_fmac_f32 v78, v65, v109 :: v_dual_fmac_f32 v79, v75, v109
	v_dual_fmac_f32 v48, v46, v102 :: v_dual_fmac_f32 v49, v56, v102
	v_dual_fmac_f32 v58, v66, v102 :: v_dual_fmac_f32 v59, v76, v102
	v_dual_fmac_f32 v68, v46, v110 :: v_dual_fmac_f32 v69, v56, v110
	v_dual_fmac_f32 v78, v66, v110 :: v_dual_fmac_f32 v79, v76, v110
	v_dual_fmac_f32 v48, v47, v103 :: v_dual_fmac_f32 v49, v57, v103
	v_dual_fmac_f32 v58, v67, v103 :: v_dual_fmac_f32 v59, v77, v103
	v_dual_fmac_f32 v68, v47, v111 :: v_dual_fmac_f32 v69, v57, v111
	v_dual_fmac_f32 v78, v67, v111 :: v_dual_fmac_f32 v79, v77, v111
	v_dual_add_f32 v160, v160, v48 :: v_dual_add_f32 v161, v161, v49
	v_dual_add_f32 v162, v162, v58 :: v_dual_add_f32 v163, v163, v59
	v_dual_add_f32 v176, v176, v68 :: v_dual_add_f32 v177, v177, v69
	v_dual_add_f32 v178, v178, v78 :: v_dual_add_f32 v179, v179, v79
	s_wait_loadcnt 0x7
	v_dual_mul_f32 v48, v41, v113 :: v_dual_mul_f32 v49, v51, v113
	v_dual_mul_f32 v58, v61, v113 :: v_dual_mul_f32 v59, v71, v113
	s_wait_loadcnt 0x5
	v_dual_mul_f32 v68, v41, v121 :: v_dual_mul_f32 v69, v51, v121
	v_dual_mul_f32 v78, v61, v121 :: v_dual_mul_f32 v79, v71, v121
	v_dual_fmac_f32 v48, v40, v112 :: v_dual_fmac_f32 v49, v50, v112
	v_dual_fmac_f32 v58, v60, v112 :: v_dual_fmac_f32 v59, v70, v112
	v_dual_fmac_f32 v68, v40, v120 :: v_dual_fmac_f32 v69, v50, v120
	v_dual_fmac_f32 v78, v60, v120 :: v_dual_fmac_f32 v79, v70, v120
	v_dual_fmac_f32 v48, v42, v114 :: v_dual_fmac_f32 v49, v52, v114
	v_dual_fmac_f32 v58, v62, v114 :: v_dual_fmac_f32 v59, v72, v114
	v_dual_fmac_f32 v68, v42, v122 :: v_dual_fmac_f32 v69, v52, v122
	v_dual_fmac_f32 v78, v62, v122 :: v_dual_fmac_f32 v79, v72, v122
	v_dual_fmac_f32 v48, v43, v115 :: v_dual_fmac_f32 v49, v53, v115
	v_dual_fmac_f32 v58, v63, v115 :: v_dual_fmac_f32 v59, v73, v115
	v_dual_fmac_f32 v68, v43, v123 :: v_dual_fmac_f32 v69, v53, v123
	v_dual_fmac_f32 v78, v63, v123 :: v_dual_fmac_f32 v79, v73, v123
	v_dual_fmac_f32 v48, v44, v116 :: v_dual_fmac_f32 v49, v54, v116
	v_dual_fmac_f32 v58, v64, v116 :: v_dual_fmac_f32 v59, v74, v116
	s_wait_loadcnt 0x4
	v_dual_fmac_f32 v68, v44, v124 :: v_dual_fmac_f32 v69, v54, v124
	v_dual_fmac_f32 v78, v64, v124 :: v_dual_fmac_f32 v79, v74, v124
	v_dual_fmac_f32 v48, v45, v117 :: v_dual_fmac_f32 v49, v55, v117
	v_dual_fmac_f32 v58, v65, v117 :: v_dual_fmac_f32 v59, v75, v117
	v_dual_fmac_f32 v68, v45, v125 :: v_dual_fmac_f32 v69, v55, v125
	v_dual_fmac_f32 v78, v65, v125 :: v_dual_fmac_f32 v79, v75, v125
	v_dual_fmac_f32 v48, v46, v118 :: v_dual_fmac_f32 v49, v56, v118
	v_dual_fmac_f32 v58, v66, v118 :: v_dual_fmac_f32 v59, v76, v118
	v_dual_fmac_f32 v68, v46, v126 :: v_dual_fmac_f32 v69, v56, v126
	v_dual_fmac_f32 v78, v66, v126 :: v_dual_fmac_f32 v79, v76, v126
	v_dual_fmac_f32 v48, v47, v119 :: v_dual_fmac_f32 v49, v57, v119
	v_dual_fmac_f32 v58, v67, v119 :: v_dual_fmac_f32 v59, v77, v119
	v_dual_fmac_f32 v68, v47, v127 :: v_dual_fmac_f32 v69, v57, v127
	v_dual_fmac_f32 v78, v67, v127 :: v_dual_fmac_f32 v79, v77, v127
	v_dual_add_f32 v192, v192, v48 :: v_dual_add_f32 v193, v193, v49
	v_dual_add_f32 v194, v194, v58 :: v_dual_add_f32 v195, v195, v59
	v_dual_add_f32 v208, v208, v68 :: v_dual_add_f32 v209, v209, v69
	v_dual_add_f32 v210, v210, v78 :: v_dual_add_f32 v211, v211, v79
	s_wait_loadcnt 0x3
	v_dual_mul_f32 v48, v41, v81 :: v_dual_mul_f32 v49, v51, v81
	v_dual_mul_f32 v58, v61, v81 :: v_dual_mul_f32 v59, v71, v81
	s_wait_loadcnt 0x1
	v_dual_mul_f32 v68, v41, v89 :: v_dual_mul_f32 v69, v51, v89
	v_dual_mul_f32 v78, v61, v89 :: v_dual_mul_f32 v79, v71, v89
	v_dual_fmac_f32 v48, v40, v80 :: v_dual_fmac_f32 v49, v50, v80
	v_dual_fmac_f32 v58, v60, v80 :: v_dual_fmac_f32 v59, v70, v80
	v_dual_fmac_f32 v68, v40, v88 :: v_dual_fmac_f32 v69, v50, v88
	v_dual_fmac_f32 v78, v60, v88 :: v_dual_fmac_f32 v79, v70, v88
	v_dual_fmac_f32 v48, v42, v82 :: v_dual_fmac_f32 v49, v52, v82
	v_dual_fmac_f32 v58, v62, v82 :: v_dual_fmac_f32 v59, v72, v82
	v_dual_fmac_f32 v68, v42, v90 :: v_dual_fmac_f32 v69, v52, v90
	v_dual_fmac_f32 v78, v62, v90 :: v_dual_fmac_f32 v79, v72, v90
	v_dual_fmac_f32 v48, v43, v83 :: v_dual_fmac_f32 v49, v53, v83
	v_dual_fmac_f32 v58, v63, v83 :: v_dual_fmac_f32 v59, v73, v83
	v_dual_fmac_f32 v68, v43, v91 :: v_dual_fmac_f32 v69, v53, v91
	v_dual_fmac_f32 v78, v63, v91 :: v_dual_fmac_f32 v79, v73, v91
	v_dual_fmac_f32 v48, v44, v84 :: v_dual_fmac_f32 v49, v54, v84
	v_dual_fmac_f32 v58, v64, v84 :: v_dual_fmac_f32 v59, v74, v84
	s_wait_loadcnt 0x0
	v_dual_fmac_f32 v68, v44, v92 :: v_dual_fmac_f32 v69, v54, v92
	v_dual_fmac_f32 v78, v64, v92 :: v_dual_fmac_f32 v79, v74, v92
	v_dual_fmac_f32 v48, v45, v85 :: v_dual_fmac_f32 v49, v55, v85
	v_dual_fmac_f32 v58, v65, v85 :: v_dual_fmac_f32 v59, v75, v85
	v_dual_fmac_f32 v68, v45, v93 :: v_dual_fmac_f32 v69, v55, v93
	v_dual_fmac_f32 v78, v65, v93 :: v_dual_fmac_f32 v79, v75, v93
	v_dual_fmac_f32 v48, v46, v86 :: v_dual_fmac_f32 v49, v56, v86
	v_dual_fmac_f32 v58, v66, v86 :: v_dual_fmac_f32 v59, v76, v86
	v_dual_fmac_f32 v68, v46, v94 :: v_dual_fmac_f32 v69, v56, v94
	v_dual_fmac_f32 v78, v66, v94 :: v_dual_fmac_f32 v79, v76, v94
	v_dual_fmac_f32 v48, v47, v87 :: v_dual_fmac_f32 v49, v57, v87
	v_dual_fmac_f32 v58, v67, v87 :: v_dual_fmac_f32 v59, v77, v87
	v_dual_fmac_f32 v68, v47, v95 :: v_dual_fmac_f32 v69, v57, v95
	v_dual_fmac_f32 v78, v67, v95 :: v_dual_fmac_f32 v79, v77, v95
	v_dual_add_f32 v224, v224, v48 :: v_dual_add_f32 v225, v225, v49
	v_dual_add_f32 v226, v226, v58 :: v_dual_add_f32 v227, v227, v59
	v_dual_add_f32 v240, v240, v68 :: v_dual_add_f32 v241, v241, v69
	v_dual_add_f32 v242, v242, v78 :: v_dual_add_f32 v243, v243, v79
	s_cmp_gt_u32 s37, 1
	s_cbranch_scc0 .Lpx_b8_fold
	buffer_load_b32 v12, v1, s[20:23], s16 offen offset:136
	buffer_load_b32 v13, v1, s[20:23], s17 offen offset:136
	buffer_load_b32 v14, v1, s[20:23], s18 offen offset:136
	buffer_load_b32 v15, v1, s[20:23], s19 offen offset:136
	buffer_load_b32 v28, v2, s[20:23], s16 offen offset:144
	buffer_load_b32 v29, v2, s[20:23], s17 offen offset:144
	buffer_load_b32 v30, v2, s[20:23], s18 offen offset:144
	buffer_load_b32 v31, v2, s[20:23], s19 offen offset:144
	buffer_load_b128 v[80:83], v3, s[24:27], s28 offen offset:1024
	buffer_load_b128 v[84:87], v3, s[24:27], s28 offen offset:1040
	buffer_load_b128 v[88:91], v3, s[24:27], s29 offen offset:1024
	buffer_load_b128 v[92:95], v3, s[24:27], s29 offen offset:1040
	buffer_load_b128 v[96:99], v3, s[24:27], s30 offen offset:1024
	buffer_load_b128 v[100:103], v3, s[24:27], s30 offen offset:1040
	buffer_load_b128 v[104:107], v3, s[24:27], s31 offen offset:1024
	buffer_load_b128 v[108:111], v3, s[24:27], s31 offen offset:1040
	buffer_load_b128 v[112:115], v3, s[24:27], s32 offen offset:1024
	buffer_load_b128 v[116:119], v3, s[24:27], s32 offen offset:1040
	buffer_load_b128 v[120:123], v3, s[24:27], s33 offen offset:1024
	buffer_load_b128 v[124:127], v3, s[24:27], s33 offen offset:1040
	s_wait_loadcnt 0xf
	v_and_b32_e32 v46, 0xf0f0f0f, v28
	s_wait_loadcnt 0xe
	v_and_b32_e32 v56, 0xf0f0f0f, v29
	s_wait_loadcnt 0xd
	v_and_b32_e32 v66, 0xf0f0f0f, v30
	s_wait_loadcnt 0xc
	v_and_b32_e32 v76, 0xf0f0f0f, v31
	v_lshrrev_b32_e32 v47, 4, v28
	v_lshrrev_b32_e32 v57, 4, v29
	v_lshrrev_b32_e32 v67, 4, v30
	v_lshrrev_b32_e32 v77, 4, v31
	v_and_b32_e32 v47, 0xf0f0f0f, v47
	v_and_b32_e32 v57, 0xf0f0f0f, v57
	v_and_b32_e32 v67, 0xf0f0f0f, v67
	v_and_b32_e32 v77, 0xf0f0f0f, v77
	v_cvt_f32_ubyte0_e32 v40, v46
	v_cvt_f32_ubyte0_e32 v50, v56
	v_cvt_f32_ubyte0_e32 v60, v66
	v_cvt_f32_ubyte0_e32 v70, v76
	v_cvt_f32_ubyte1_e32 v42, v46
	v_cvt_f32_ubyte1_e32 v52, v56
	v_cvt_f32_ubyte1_e32 v62, v66
	v_cvt_f32_ubyte1_e32 v72, v76
	v_cvt_f32_ubyte2_e32 v44, v46
	v_cvt_f32_ubyte2_e32 v54, v56
	v_cvt_f32_ubyte2_e32 v64, v66
	v_cvt_f32_ubyte2_e32 v74, v76
	v_cvt_f32_ubyte3_e32 v46, v46
	v_cvt_f32_ubyte3_e32 v56, v56
	v_cvt_f32_ubyte3_e32 v66, v66
	v_cvt_f32_ubyte3_e32 v76, v76
	v_cvt_f32_ubyte0_e32 v41, v47
	v_cvt_f32_ubyte0_e32 v51, v57
	v_cvt_f32_ubyte0_e32 v61, v67
	v_cvt_f32_ubyte0_e32 v71, v77
	v_cvt_f32_ubyte1_e32 v43, v47
	v_cvt_f32_ubyte1_e32 v53, v57
	v_cvt_f32_ubyte1_e32 v63, v67
	v_cvt_f32_ubyte1_e32 v73, v77
	v_cvt_f32_ubyte2_e32 v45, v47
	v_cvt_f32_ubyte2_e32 v55, v57
	v_cvt_f32_ubyte2_e32 v65, v67
	v_cvt_f32_ubyte2_e32 v75, v77
	v_cvt_f32_ubyte3_e32 v47, v47
	v_cvt_f32_ubyte3_e32 v57, v57
	v_cvt_f32_ubyte3_e32 v67, v67
	v_cvt_f32_ubyte3_e32 v77, v77
	v_fma_mix_f32 v40, v12, v40, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v13, v50, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v14, v60, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v70, v15, v70, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v41, v12, v41, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v13, v51, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v14, v61, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v71, v15, v71, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v42, v12, v42, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v52, v13, v52, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v62, v14, v62, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v72, v15, v72, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v43, v12, v43, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v53, v13, v53, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v63, v14, v63, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v73, v15, v73, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v44, v12, v44, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v54, v13, v54, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v64, v14, v64, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v74, v15, v74, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v12, v45, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v13, v55, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v65, v14, v65, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v75, v15, v75, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v12, v46, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v13, v56, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v66, v14, v66, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v76, v15, v76, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v12, v47, v12 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v13, v57, v13 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v67, v14, v67, v14 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v77, v15, v77, v15 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xb
	v_dual_mul_f32 v48, v41, v81 :: v_dual_mul_f32 v49, v51, v81
	v_dual_mul_f32 v58, v61, v81 :: v_dual_mul_f32 v59, v71, v81
	s_wait_loadcnt 0x9
	v_dual_mul_f32 v68, v41, v89 :: v_dual_mul_f32 v69, v51, v89
	v_dual_mul_f32 v78, v61, v89 :: v_dual_mul_f32 v79, v71, v89
	v_dual_fmac_f32 v48, v40, v80 :: v_dual_fmac_f32 v49, v50, v80
	v_dual_fmac_f32 v58, v60, v80 :: v_dual_fmac_f32 v59, v70, v80
	v_dual_fmac_f32 v68, v40, v88 :: v_dual_fmac_f32 v69, v50, v88
	v_dual_fmac_f32 v78, v60, v88 :: v_dual_fmac_f32 v79, v70, v88
	v_dual_fmac_f32 v48, v42, v82 :: v_dual_fmac_f32 v49, v52, v82
	v_dual_fmac_f32 v58, v62, v82 :: v_dual_fmac_f32 v59, v72, v82
	v_dual_fmac_f32 v68, v42, v90 :: v_dual_fmac_f32 v69, v52, v90
	v_dual_fmac_f32 v78, v62, v90 :: v_dual_fmac_f32 v79, v72, v90
	v_dual_fmac_f32 v48, v43, v83 :: v_dual_fmac_f32 v49, v53, v83
	v_dual_fmac_f32 v58, v63, v83 :: v_dual_fmac_f32 v59, v73, v83
	v_dual_fmac_f32 v68, v43, v91 :: v_dual_fmac_f32 v69, v53, v91
	v_dual_fmac_f32 v78, v63, v91 :: v_dual_fmac_f32 v79, v73, v91
	v_dual_fmac_f32 v48, v44, v84 :: v_dual_fmac_f32 v49, v54, v84
	v_dual_fmac_f32 v58, v64, v84 :: v_dual_fmac_f32 v59, v74, v84
	s_wait_loadcnt 0x8
	v_dual_fmac_f32 v68, v44, v92 :: v_dual_fmac_f32 v69, v54, v92
	v_dual_fmac_f32 v78, v64, v92 :: v_dual_fmac_f32 v79, v74, v92
	v_dual_fmac_f32 v48, v45, v85 :: v_dual_fmac_f32 v49, v55, v85
	v_dual_fmac_f32 v58, v65, v85 :: v_dual_fmac_f32 v59, v75, v85
	v_dual_fmac_f32 v68, v45, v93 :: v_dual_fmac_f32 v69, v55, v93
	v_dual_fmac_f32 v78, v65, v93 :: v_dual_fmac_f32 v79, v75, v93
	v_dual_fmac_f32 v48, v46, v86 :: v_dual_fmac_f32 v49, v56, v86
	v_dual_fmac_f32 v58, v66, v86 :: v_dual_fmac_f32 v59, v76, v86
	v_dual_fmac_f32 v68, v46, v94 :: v_dual_fmac_f32 v69, v56, v94
	v_dual_fmac_f32 v78, v66, v94 :: v_dual_fmac_f32 v79, v76, v94
	v_dual_fmac_f32 v48, v47, v87 :: v_dual_fmac_f32 v49, v57, v87
	v_dual_fmac_f32 v58, v67, v87 :: v_dual_fmac_f32 v59, v77, v87
	v_dual_fmac_f32 v68, v47, v95 :: v_dual_fmac_f32 v69, v57, v95
	v_dual_fmac_f32 v78, v67, v95 :: v_dual_fmac_f32 v79, v77, v95
	v_dual_add_f32 v132, v132, v48 :: v_dual_add_f32 v133, v133, v49
	v_dual_add_f32 v134, v134, v58 :: v_dual_add_f32 v135, v135, v59
	v_dual_add_f32 v148, v148, v68 :: v_dual_add_f32 v149, v149, v69
	v_dual_add_f32 v150, v150, v78 :: v_dual_add_f32 v151, v151, v79
	buffer_load_b128 v[80:83], v3, s[24:27], s34 offen offset:1024
	buffer_load_b128 v[84:87], v3, s[24:27], s34 offen offset:1040
	buffer_load_b128 v[88:91], v3, s[24:27], s35 offen offset:1024
	buffer_load_b128 v[92:95], v3, s[24:27], s35 offen offset:1040
	s_wait_loadcnt 0xb
	v_dual_mul_f32 v48, v41, v97 :: v_dual_mul_f32 v49, v51, v97
	v_dual_mul_f32 v58, v61, v97 :: v_dual_mul_f32 v59, v71, v97
	s_wait_loadcnt 0x9
	v_dual_mul_f32 v68, v41, v105 :: v_dual_mul_f32 v69, v51, v105
	v_dual_mul_f32 v78, v61, v105 :: v_dual_mul_f32 v79, v71, v105
	v_dual_fmac_f32 v48, v40, v96 :: v_dual_fmac_f32 v49, v50, v96
	v_dual_fmac_f32 v58, v60, v96 :: v_dual_fmac_f32 v59, v70, v96
	v_dual_fmac_f32 v68, v40, v104 :: v_dual_fmac_f32 v69, v50, v104
	v_dual_fmac_f32 v78, v60, v104 :: v_dual_fmac_f32 v79, v70, v104
	v_dual_fmac_f32 v48, v42, v98 :: v_dual_fmac_f32 v49, v52, v98
	v_dual_fmac_f32 v58, v62, v98 :: v_dual_fmac_f32 v59, v72, v98
	v_dual_fmac_f32 v68, v42, v106 :: v_dual_fmac_f32 v69, v52, v106
	v_dual_fmac_f32 v78, v62, v106 :: v_dual_fmac_f32 v79, v72, v106
	v_dual_fmac_f32 v48, v43, v99 :: v_dual_fmac_f32 v49, v53, v99
	v_dual_fmac_f32 v58, v63, v99 :: v_dual_fmac_f32 v59, v73, v99
	v_dual_fmac_f32 v68, v43, v107 :: v_dual_fmac_f32 v69, v53, v107
	v_dual_fmac_f32 v78, v63, v107 :: v_dual_fmac_f32 v79, v73, v107
	v_dual_fmac_f32 v48, v44, v100 :: v_dual_fmac_f32 v49, v54, v100
	v_dual_fmac_f32 v58, v64, v100 :: v_dual_fmac_f32 v59, v74, v100
	s_wait_loadcnt 0x8
	v_dual_fmac_f32 v68, v44, v108 :: v_dual_fmac_f32 v69, v54, v108
	v_dual_fmac_f32 v78, v64, v108 :: v_dual_fmac_f32 v79, v74, v108
	v_dual_fmac_f32 v48, v45, v101 :: v_dual_fmac_f32 v49, v55, v101
	v_dual_fmac_f32 v58, v65, v101 :: v_dual_fmac_f32 v59, v75, v101
	v_dual_fmac_f32 v68, v45, v109 :: v_dual_fmac_f32 v69, v55, v109
	v_dual_fmac_f32 v78, v65, v109 :: v_dual_fmac_f32 v79, v75, v109
	v_dual_fmac_f32 v48, v46, v102 :: v_dual_fmac_f32 v49, v56, v102
	v_dual_fmac_f32 v58, v66, v102 :: v_dual_fmac_f32 v59, v76, v102
	v_dual_fmac_f32 v68, v46, v110 :: v_dual_fmac_f32 v69, v56, v110
	v_dual_fmac_f32 v78, v66, v110 :: v_dual_fmac_f32 v79, v76, v110
	v_dual_fmac_f32 v48, v47, v103 :: v_dual_fmac_f32 v49, v57, v103
	v_dual_fmac_f32 v58, v67, v103 :: v_dual_fmac_f32 v59, v77, v103
	v_dual_fmac_f32 v68, v47, v111 :: v_dual_fmac_f32 v69, v57, v111
	v_dual_fmac_f32 v78, v67, v111 :: v_dual_fmac_f32 v79, v77, v111
	v_dual_add_f32 v164, v164, v48 :: v_dual_add_f32 v165, v165, v49
	v_dual_add_f32 v166, v166, v58 :: v_dual_add_f32 v167, v167, v59
	v_dual_add_f32 v180, v180, v68 :: v_dual_add_f32 v181, v181, v69
	v_dual_add_f32 v182, v182, v78 :: v_dual_add_f32 v183, v183, v79
	s_wait_loadcnt 0x7
	v_dual_mul_f32 v48, v41, v113 :: v_dual_mul_f32 v49, v51, v113
	v_dual_mul_f32 v58, v61, v113 :: v_dual_mul_f32 v59, v71, v113
	s_wait_loadcnt 0x5
	v_dual_mul_f32 v68, v41, v121 :: v_dual_mul_f32 v69, v51, v121
	v_dual_mul_f32 v78, v61, v121 :: v_dual_mul_f32 v79, v71, v121
	v_dual_fmac_f32 v48, v40, v112 :: v_dual_fmac_f32 v49, v50, v112
	v_dual_fmac_f32 v58, v60, v112 :: v_dual_fmac_f32 v59, v70, v112
	v_dual_fmac_f32 v68, v40, v120 :: v_dual_fmac_f32 v69, v50, v120
	v_dual_fmac_f32 v78, v60, v120 :: v_dual_fmac_f32 v79, v70, v120
	v_dual_fmac_f32 v48, v42, v114 :: v_dual_fmac_f32 v49, v52, v114
	v_dual_fmac_f32 v58, v62, v114 :: v_dual_fmac_f32 v59, v72, v114
	v_dual_fmac_f32 v68, v42, v122 :: v_dual_fmac_f32 v69, v52, v122
	v_dual_fmac_f32 v78, v62, v122 :: v_dual_fmac_f32 v79, v72, v122
	v_dual_fmac_f32 v48, v43, v115 :: v_dual_fmac_f32 v49, v53, v115
	v_dual_fmac_f32 v58, v63, v115 :: v_dual_fmac_f32 v59, v73, v115
	v_dual_fmac_f32 v68, v43, v123 :: v_dual_fmac_f32 v69, v53, v123
	v_dual_fmac_f32 v78, v63, v123 :: v_dual_fmac_f32 v79, v73, v123
	v_dual_fmac_f32 v48, v44, v116 :: v_dual_fmac_f32 v49, v54, v116
	v_dual_fmac_f32 v58, v64, v116 :: v_dual_fmac_f32 v59, v74, v116
	s_wait_loadcnt 0x4
	v_dual_fmac_f32 v68, v44, v124 :: v_dual_fmac_f32 v69, v54, v124
	v_dual_fmac_f32 v78, v64, v124 :: v_dual_fmac_f32 v79, v74, v124
	v_dual_fmac_f32 v48, v45, v117 :: v_dual_fmac_f32 v49, v55, v117
	v_dual_fmac_f32 v58, v65, v117 :: v_dual_fmac_f32 v59, v75, v117
	v_dual_fmac_f32 v68, v45, v125 :: v_dual_fmac_f32 v69, v55, v125
	v_dual_fmac_f32 v78, v65, v125 :: v_dual_fmac_f32 v79, v75, v125
	v_dual_fmac_f32 v48, v46, v118 :: v_dual_fmac_f32 v49, v56, v118
	v_dual_fmac_f32 v58, v66, v118 :: v_dual_fmac_f32 v59, v76, v118
	v_dual_fmac_f32 v68, v46, v126 :: v_dual_fmac_f32 v69, v56, v126
	v_dual_fmac_f32 v78, v66, v126 :: v_dual_fmac_f32 v79, v76, v126
	v_dual_fmac_f32 v48, v47, v119 :: v_dual_fmac_f32 v49, v57, v119
	v_dual_fmac_f32 v58, v67, v119 :: v_dual_fmac_f32 v59, v77, v119
	v_dual_fmac_f32 v68, v47, v127 :: v_dual_fmac_f32 v69, v57, v127
	v_dual_fmac_f32 v78, v67, v127 :: v_dual_fmac_f32 v79, v77, v127
	v_dual_add_f32 v196, v196, v48 :: v_dual_add_f32 v197, v197, v49
	v_dual_add_f32 v198, v198, v58 :: v_dual_add_f32 v199, v199, v59
	v_dual_add_f32 v212, v212, v68 :: v_dual_add_f32 v213, v213, v69
	v_dual_add_f32 v214, v214, v78 :: v_dual_add_f32 v215, v215, v79
	s_wait_loadcnt 0x3
	v_dual_mul_f32 v48, v41, v81 :: v_dual_mul_f32 v49, v51, v81
	v_dual_mul_f32 v58, v61, v81 :: v_dual_mul_f32 v59, v71, v81
	s_wait_loadcnt 0x1
	v_dual_mul_f32 v68, v41, v89 :: v_dual_mul_f32 v69, v51, v89
	v_dual_mul_f32 v78, v61, v89 :: v_dual_mul_f32 v79, v71, v89
	v_dual_fmac_f32 v48, v40, v80 :: v_dual_fmac_f32 v49, v50, v80
	v_dual_fmac_f32 v58, v60, v80 :: v_dual_fmac_f32 v59, v70, v80
	v_dual_fmac_f32 v68, v40, v88 :: v_dual_fmac_f32 v69, v50, v88
	v_dual_fmac_f32 v78, v60, v88 :: v_dual_fmac_f32 v79, v70, v88
	v_dual_fmac_f32 v48, v42, v82 :: v_dual_fmac_f32 v49, v52, v82
	v_dual_fmac_f32 v58, v62, v82 :: v_dual_fmac_f32 v59, v72, v82
	v_dual_fmac_f32 v68, v42, v90 :: v_dual_fmac_f32 v69, v52, v90
	v_dual_fmac_f32 v78, v62, v90 :: v_dual_fmac_f32 v79, v72, v90
	v_dual_fmac_f32 v48, v43, v83 :: v_dual_fmac_f32 v49, v53, v83
	v_dual_fmac_f32 v58, v63, v83 :: v_dual_fmac_f32 v59, v73, v83
	v_dual_fmac_f32 v68, v43, v91 :: v_dual_fmac_f32 v69, v53, v91
	v_dual_fmac_f32 v78, v63, v91 :: v_dual_fmac_f32 v79, v73, v91
	v_dual_fmac_f32 v48, v44, v84 :: v_dual_fmac_f32 v49, v54, v84
	v_dual_fmac_f32 v58, v64, v84 :: v_dual_fmac_f32 v59, v74, v84
	s_wait_loadcnt 0x0
	v_dual_fmac_f32 v68, v44, v92 :: v_dual_fmac_f32 v69, v54, v92
	v_dual_fmac_f32 v78, v64, v92 :: v_dual_fmac_f32 v79, v74, v92
	v_dual_fmac_f32 v48, v45, v85 :: v_dual_fmac_f32 v49, v55, v85
	v_dual_fmac_f32 v58, v65, v85 :: v_dual_fmac_f32 v59, v75, v85
	v_dual_fmac_f32 v68, v45, v93 :: v_dual_fmac_f32 v69, v55, v93
	v_dual_fmac_f32 v78, v65, v93 :: v_dual_fmac_f32 v79, v75, v93
	v_dual_fmac_f32 v48, v46, v86 :: v_dual_fmac_f32 v49, v56, v86
	v_dual_fmac_f32 v58, v66, v86 :: v_dual_fmac_f32 v59, v76, v86
	v_dual_fmac_f32 v68, v46, v94 :: v_dual_fmac_f32 v69, v56, v94
	v_dual_fmac_f32 v78, v66, v94 :: v_dual_fmac_f32 v79, v76, v94
	v_dual_fmac_f32 v48, v47, v87 :: v_dual_fmac_f32 v49, v57, v87
	v_dual_fmac_f32 v58, v67, v87 :: v_dual_fmac_f32 v59, v77, v87
	v_dual_fmac_f32 v68, v47, v95 :: v_dual_fmac_f32 v69, v57, v95
	v_dual_fmac_f32 v78, v67, v95 :: v_dual_fmac_f32 v79, v77, v95
	v_dual_add_f32 v228, v228, v48 :: v_dual_add_f32 v229, v229, v49
	v_dual_add_f32 v230, v230, v58 :: v_dual_add_f32 v231, v231, v59
	v_dual_add_f32 v244, v244, v68 :: v_dual_add_f32 v245, v245, v69
	v_dual_add_f32 v246, v246, v78 :: v_dual_add_f32 v247, v247, v79
	s_cmp_gt_u32 s37, 2
	s_cbranch_scc0 .Lpx_b8_fold
	buffer_load_b32 v16, v1, s[20:23], s16 offen offset:272
	buffer_load_b32 v17, v1, s[20:23], s17 offen offset:272
	buffer_load_b32 v18, v1, s[20:23], s18 offen offset:272
	buffer_load_b32 v19, v1, s[20:23], s19 offen offset:272
	buffer_load_b32 v32, v2, s[20:23], s16 offen offset:280
	buffer_load_b32 v33, v2, s[20:23], s17 offen offset:280
	buffer_load_b32 v34, v2, s[20:23], s18 offen offset:280
	buffer_load_b32 v35, v2, s[20:23], s19 offen offset:280
	buffer_load_b128 v[80:83], v3, s[24:27], s28 offen offset:2048
	buffer_load_b128 v[84:87], v3, s[24:27], s28 offen offset:2064
	buffer_load_b128 v[88:91], v3, s[24:27], s29 offen offset:2048
	buffer_load_b128 v[92:95], v3, s[24:27], s29 offen offset:2064
	buffer_load_b128 v[96:99], v3, s[24:27], s30 offen offset:2048
	buffer_load_b128 v[100:103], v3, s[24:27], s30 offen offset:2064
	buffer_load_b128 v[104:107], v3, s[24:27], s31 offen offset:2048
	buffer_load_b128 v[108:111], v3, s[24:27], s31 offen offset:2064
	buffer_load_b128 v[112:115], v3, s[24:27], s32 offen offset:2048
	buffer_load_b128 v[116:119], v3, s[24:27], s32 offen offset:2064
	buffer_load_b128 v[120:123], v3, s[24:27], s33 offen offset:2048
	buffer_load_b128 v[124:127], v3, s[24:27], s33 offen offset:2064
	s_wait_loadcnt 0xf
	v_and_b32_e32 v46, 0xf0f0f0f, v32
	s_wait_loadcnt 0xe
	v_and_b32_e32 v56, 0xf0f0f0f, v33
	s_wait_loadcnt 0xd
	v_and_b32_e32 v66, 0xf0f0f0f, v34
	s_wait_loadcnt 0xc
	v_and_b32_e32 v76, 0xf0f0f0f, v35
	v_lshrrev_b32_e32 v47, 4, v32
	v_lshrrev_b32_e32 v57, 4, v33
	v_lshrrev_b32_e32 v67, 4, v34
	v_lshrrev_b32_e32 v77, 4, v35
	v_and_b32_e32 v47, 0xf0f0f0f, v47
	v_and_b32_e32 v57, 0xf0f0f0f, v57
	v_and_b32_e32 v67, 0xf0f0f0f, v67
	v_and_b32_e32 v77, 0xf0f0f0f, v77
	v_cvt_f32_ubyte0_e32 v40, v46
	v_cvt_f32_ubyte0_e32 v50, v56
	v_cvt_f32_ubyte0_e32 v60, v66
	v_cvt_f32_ubyte0_e32 v70, v76
	v_cvt_f32_ubyte1_e32 v42, v46
	v_cvt_f32_ubyte1_e32 v52, v56
	v_cvt_f32_ubyte1_e32 v62, v66
	v_cvt_f32_ubyte1_e32 v72, v76
	v_cvt_f32_ubyte2_e32 v44, v46
	v_cvt_f32_ubyte2_e32 v54, v56
	v_cvt_f32_ubyte2_e32 v64, v66
	v_cvt_f32_ubyte2_e32 v74, v76
	v_cvt_f32_ubyte3_e32 v46, v46
	v_cvt_f32_ubyte3_e32 v56, v56
	v_cvt_f32_ubyte3_e32 v66, v66
	v_cvt_f32_ubyte3_e32 v76, v76
	v_cvt_f32_ubyte0_e32 v41, v47
	v_cvt_f32_ubyte0_e32 v51, v57
	v_cvt_f32_ubyte0_e32 v61, v67
	v_cvt_f32_ubyte0_e32 v71, v77
	v_cvt_f32_ubyte1_e32 v43, v47
	v_cvt_f32_ubyte1_e32 v53, v57
	v_cvt_f32_ubyte1_e32 v63, v67
	v_cvt_f32_ubyte1_e32 v73, v77
	v_cvt_f32_ubyte2_e32 v45, v47
	v_cvt_f32_ubyte2_e32 v55, v57
	v_cvt_f32_ubyte2_e32 v65, v67
	v_cvt_f32_ubyte2_e32 v75, v77
	v_cvt_f32_ubyte3_e32 v47, v47
	v_cvt_f32_ubyte3_e32 v57, v57
	v_cvt_f32_ubyte3_e32 v67, v67
	v_cvt_f32_ubyte3_e32 v77, v77
	v_fma_mix_f32 v40, v16, v40, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v50, v17, v50, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v60, v18, v60, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v70, v19, v70, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v41, v16, v41, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v51, v17, v51, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v61, v18, v61, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v71, v19, v71, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v42, v16, v42, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v52, v17, v52, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v62, v18, v62, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v72, v19, v72, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v43, v16, v43, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v53, v17, v53, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v63, v18, v63, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v73, v19, v73, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v44, v16, v44, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v54, v17, v54, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v64, v18, v64, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v74, v19, v74, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v45, v16, v45, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v55, v17, v55, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v65, v18, v65, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v75, v19, v75, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v46, v16, v46, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v56, v17, v56, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v66, v18, v66, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v76, v19, v76, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v47, v16, v47, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v57, v17, v57, v17 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v67, v18, v67, v18 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fma_mix_f32 v77, v19, v77, v19 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0xb
	v_dual_mul_f32 v48, v41, v81 :: v_dual_mul_f32 v49, v51, v81
	v_dual_mul_f32 v58, v61, v81 :: v_dual_mul_f32 v59, v71, v81
	s_wait_loadcnt 0x9
	v_dual_mul_f32 v68, v41, v89 :: v_dual_mul_f32 v69, v51, v89
	v_dual_mul_f32 v78, v61, v89 :: v_dual_mul_f32 v79, v71, v89
	v_dual_fmac_f32 v48, v40, v80 :: v_dual_fmac_f32 v49, v50, v80
	v_dual_fmac_f32 v58, v60, v80 :: v_dual_fmac_f32 v59, v70, v80
	v_dual_fmac_f32 v68, v40, v88 :: v_dual_fmac_f32 v69, v50, v88
	v_dual_fmac_f32 v78, v60, v88 :: v_dual_fmac_f32 v79, v70, v88
	v_dual_fmac_f32 v48, v42, v82 :: v_dual_fmac_f32 v49, v52, v82
	v_dual_fmac_f32 v58, v62, v82 :: v_dual_fmac_f32 v59, v72, v82
	v_dual_fmac_f32 v68, v42, v90 :: v_dual_fmac_f32 v69, v52, v90
	v_dual_fmac_f32 v78, v62, v90 :: v_dual_fmac_f32 v79, v72, v90
	v_dual_fmac_f32 v48, v43, v83 :: v_dual_fmac_f32 v49, v53, v83
	v_dual_fmac_f32 v58, v63, v83 :: v_dual_fmac_f32 v59, v73, v83
	v_dual_fmac_f32 v68, v43, v91 :: v_dual_fmac_f32 v69, v53, v91
	v_dual_fmac_f32 v78, v63, v91 :: v_dual_fmac_f32 v79, v73, v91
	v_dual_fmac_f32 v48, v44, v84 :: v_dual_fmac_f32 v49, v54, v84
	v_dual_fmac_f32 v58, v64, v84 :: v_dual_fmac_f32 v59, v74, v84
	s_wait_loadcnt 0x8
	v_dual_fmac_f32 v68, v44, v92 :: v_dual_fmac_f32 v69, v54, v92
	v_dual_fmac_f32 v78, v64, v92 :: v_dual_fmac_f32 v79, v74, v92
	v_dual_fmac_f32 v48, v45, v85 :: v_dual_fmac_f32 v49, v55, v85
	v_dual_fmac_f32 v58, v65, v85 :: v_dual_fmac_f32 v59, v75, v85
	v_dual_fmac_f32 v68, v45, v93 :: v_dual_fmac_f32 v69, v55, v93
	v_dual_fmac_f32 v78, v65, v93 :: v_dual_fmac_f32 v79, v75, v93
	v_dual_fmac_f32 v48, v46, v86 :: v_dual_fmac_f32 v49, v56, v86
	v_dual_fmac_f32 v58, v66, v86 :: v_dual_fmac_f32 v59, v76, v86
	v_dual_fmac_f32 v68, v46, v94 :: v_dual_fmac_f32 v69, v56, v94
	v_dual_fmac_f32 v78, v66, v94 :: v_dual_fmac_f32 v79, v76, v94
	v_dual_fmac_f32 v48, v47, v87 :: v_dual_fmac_f32 v49, v57, v87
	v_dual_fmac_f32 v58, v67, v87 :: v_dual_fmac_f32 v59, v77, v87
	v_dual_fmac_f32 v68, v47, v95 :: v_dual_fmac_f32 v69, v57, v95
	v_dual_fmac_f32 v78, v67, v95 :: v_dual_fmac_f32 v79, v77, v95
	v_dual_add_f32 v136, v136, v48 :: v_dual_add_f32 v137, v137, v49
	v_dual_add_f32 v138, v138, v58 :: v_dual_add_f32 v139, v139, v59
	v_dual_add_f32 v152, v152, v68 :: v_dual_add_f32 v153, v153, v69
	v_dual_add_f32 v154, v154, v78 :: v_dual_add_f32 v155, v155, v79
	buffer_load_b128 v[80:83], v3, s[24:27], s34 offen offset:2048
	buffer_load_b128 v[84:87], v3, s[24:27], s34 offen offset:2064
	buffer_load_b128 v[88:91], v3, s[24:27], s35 offen offset:2048
	buffer_load_b128 v[92:95], v3, s[24:27], s35 offen offset:2064
	s_wait_loadcnt 0xb
	v_dual_mul_f32 v48, v41, v97 :: v_dual_mul_f32 v49, v51, v97
	v_dual_mul_f32 v58, v61, v97 :: v_dual_mul_f32 v59, v71, v97
	s_wait_loadcnt 0x9
	v_dual_mul_f32 v68, v41, v105 :: v_dual_mul_f32 v69, v51, v105
	v_dual_mul_f32 v78, v61, v105 :: v_dual_mul_f32 v79, v71, v105
	v_dual_fmac_f32 v48, v40, v96 :: v_dual_fmac_f32 v49, v50, v96
	v_dual_fmac_f32 v58, v60, v96 :: v_dual_fmac_f32 v59, v70, v96
	v_dual_fmac_f32 v68, v40, v104 :: v_dual_fmac_f32 v69, v50, v104
	v_dual_fmac_f32 v78, v60, v104 :: v_dual_fmac_f32 v79, v70, v104
	v_dual_fmac_f32 v48, v42, v98 :: v_dual_fmac_f32 v49, v52, v98
	v_dual_fmac_f32 v58, v62, v98 :: v_dual_fmac_f32 v59, v72, v98
	v_dual_fmac_f32 v68, v42, v106 :: v_dual_fmac_f32 v69, v52, v106
	v_dual_fmac_f32 v78, v62, v106 :: v_dual_fmac_f32 v79, v72, v106
	v_dual_fmac_f32 v48, v43, v99 :: v_dual_fmac_f32 v49, v53, v99
	v_dual_fmac_f32 v58, v63, v99 :: v_dual_fmac_f32 v59, v73, v99
	v_dual_fmac_f32 v68, v43, v107 :: v_dual_fmac_f32 v69, v53, v107
	v_dual_fmac_f32 v78, v63, v107 :: v_dual_fmac_f32 v79, v73, v107
	v_dual_fmac_f32 v48, v44, v100 :: v_dual_fmac_f32 v49, v54, v100
	v_dual_fmac_f32 v58, v64, v100 :: v_dual_fmac_f32 v59, v74, v100
	s_wait_loadcnt 0x8
	v_dual_fmac_f32 v68, v44, v108 :: v_dual_fmac_f32 v69, v54, v108
	v_dual_fmac_f32 v78, v64, v108 :: v_dual_fmac_f32 v79, v74, v108
	v_dual_fmac_f32 v48, v45, v101 :: v_dual_fmac_f32 v49, v55, v101
	v_dual_fmac_f32 v58, v65, v101 :: v_dual_fmac_f32 v59, v75, v101
	v_dual_fmac_f32 v68, v45, v109 :: v_dual_fmac_f32 v69, v55, v109
	v_dual_fmac_f32 v78, v65, v109 :: v_dual_fmac_f32 v79, v75, v109
	v_dual_fmac_f32 v48, v46, v102 :: v_dual_fmac_f32 v49, v56, v102
	v_dual_fmac_f32 v58, v66, v102 :: v_dual_fmac_f32 v59, v76, v102
	v_dual_fmac_f32 v68, v46, v110 :: v_dual_fmac_f32 v69, v56, v110
	v_dual_fmac_f32 v78, v66, v110 :: v_dual_fmac_f32 v79, v76, v110
	v_dual_fmac_f32 v48, v47, v103 :: v_dual_fmac_f32 v49, v57, v103
	v_dual_fmac_f32 v58, v67, v103 :: v_dual_fmac_f32 v59, v77, v103
	v_dual_fmac_f32 v68, v47, v111 :: v_dual_fmac_f32 v69, v57, v111
	v_dual_fmac_f32 v78, v67, v111 :: v_dual_fmac_f32 v79, v77, v111
	v_dual_add_f32 v168, v168, v48 :: v_dual_add_f32 v169, v169, v49
	v_dual_add_f32 v170, v170, v58 :: v_dual_add_f32 v171, v171, v59
	v_dual_add_f32 v184, v184, v68 :: v_dual_add_f32 v185, v185, v69
	v_dual_add_f32 v186, v186, v78 :: v_dual_add_f32 v187, v187, v79
	s_wait_loadcnt 0x7
	v_dual_mul_f32 v48, v41, v113 :: v_dual_mul_f32 v49, v51, v113
	v_dual_mul_f32 v58, v61, v113 :: v_dual_mul_f32 v59, v71, v113
	s_wait_loadcnt 0x5
	v_dual_mul_f32 v68, v41, v121 :: v_dual_mul_f32 v69, v51, v121
	v_dual_mul_f32 v78, v61, v121 :: v_dual_mul_f32 v79, v71, v121
	v_dual_fmac_f32 v48, v40, v112 :: v_dual_fmac_f32 v49, v50, v112
	v_dual_fmac_f32 v58, v60, v112 :: v_dual_fmac_f32 v59, v70, v112
	v_dual_fmac_f32 v68, v40, v120 :: v_dual_fmac_f32 v69, v50, v120
	v_dual_fmac_f32 v78, v60, v120 :: v_dual_fmac_f32 v79, v70, v120
	v_dual_fmac_f32 v48, v42, v114 :: v_dual_fmac_f32 v49, v52, v114
	v_dual_fmac_f32 v58, v62, v114 :: v_dual_fmac_f32 v59, v72, v114
	v_dual_fmac_f32 v68, v42, v122 :: v_dual_fmac_f32 v69, v52, v122
	v_dual_fmac_f32 v78, v62, v122 :: v_dual_fmac_f32 v79, v72, v122
	v_dual_fmac_f32 v48, v43, v115 :: v_dual_fmac_f32 v49, v53, v115
	v_dual_fmac_f32 v58, v63, v115 :: v_dual_fmac_f32 v59, v73, v115
	v_dual_fmac_f32 v68, v43, v123 :: v_dual_fmac_f32 v69, v53, v123
	v_dual_fmac_f32 v78, v63, v123 :: v_dual_fmac_f32 v79, v73, v123
	v_dual_fmac_f32 v48, v44, v116 :: v_dual_fmac_f32 v49, v54, v116
	v_dual_fmac_f32 v58, v64, v116 :: v_dual_fmac_f32 v59, v74, v116
	s_wait_loadcnt 0x4
	v_dual_fmac_f32 v68, v44, v124 :: v_dual_fmac_f32 v69, v54, v124
	v_dual_fmac_f32 v78, v64, v124 :: v_dual_fmac_f32 v79, v74, v124
	v_dual_fmac_f32 v48, v45, v117 :: v_dual_fmac_f32 v49, v55, v117
	v_dual_fmac_f32 v58, v65, v117 :: v_dual_fmac_f32 v59, v75, v117
	v_dual_fmac_f32 v68, v45, v125 :: v_dual_fmac_f32 v69, v55, v125
	v_dual_fmac_f32 v78, v65, v125 :: v_dual_fmac_f32 v79, v75, v125
	v_dual_fmac_f32 v48, v46, v118 :: v_dual_fmac_f32 v49, v56, v118
	v_dual_fmac_f32 v58, v66, v118 :: v_dual_fmac_f32 v59, v76, v118
	v_dual_fmac_f32 v68, v46, v126 :: v_dual_fmac_f32 v69, v56, v126
	v_dual_fmac_f32 v78, v66, v126 :: v_dual_fmac_f32 v79, v76, v126
	v_dual_fmac_f32 v48, v47, v119 :: v_dual_fmac_f32 v49, v57, v119
	v_dual_fmac_f32 v58, v67, v119 :: v_dual_fmac_f32 v59, v77, v119
	v_dual_fmac_f32 v68, v47, v127 :: v_dual_fmac_f32 v69, v57, v127
	v_dual_fmac_f32 v78, v67, v127 :: v_dual_fmac_f32 v79, v77, v127
	v_dual_add_f32 v200, v200, v48 :: v_dual_add_f32 v201, v201, v49
	v_dual_add_f32 v202, v202, v58 :: v_dual_add_f32 v203, v203, v59
	v_dual_add_f32 v216, v216, v68 :: v_dual_add_f32 v217, v217, v69
	v_dual_add_f32 v218, v218, v78 :: v_dual_add_f32 v219, v219, v79
	s_wait_loadcnt 0x3
	v_dual_mul_f32 v48, v41, v81 :: v_dual_mul_f32 v49, v51, v81
	v_dual_mul_f32 v58, v61, v81 :: v_dual_mul_f32 v59, v71, v81
	s_wait_loadcnt 0x1
	v_dual_mul_f32 v68, v41, v89 :: v_dual_mul_f32 v69, v51, v89
	v_dual_mul_f32 v78, v61, v89 :: v_dual_mul_f32 v79, v71, v89
	v_dual_fmac_f32 v48, v40, v80 :: v_dual_fmac_f32 v49, v50, v80
	v_dual_fmac_f32 v58, v60, v80 :: v_dual_fmac_f32 v59, v70, v80
	v_dual_fmac_f32 v68, v40, v88 :: v_dual_fmac_f32 v69, v50, v88
	v_dual_fmac_f32 v78, v60, v88 :: v_dual_fmac_f32 v79, v70, v88
	v_dual_fmac_f32 v48, v42, v82 :: v_dual_fmac_f32 v49, v52, v82
	v_dual_fmac_f32 v58, v62, v82 :: v_dual_fmac_f32 v59, v72, v82
	v_dual_fmac_f32 v68, v42, v90 :: v_dual_fmac_f32 v69, v52, v90
	v_dual_fmac_f32 v78, v62, v90 :: v_dual_fmac_f32 v79, v72, v90
	v_dual_fmac_f32 v48, v43, v83 :: v_dual_fmac_f32 v49, v53, v83
	v_dual_fmac_f32 v58, v63, v83 :: v_dual_fmac_f32 v59, v73, v83
	v_dual_fmac_f32 v68, v43, v91 :: v_dual_fmac_f32 v69, v53, v91
	v_dual_fmac_f32 v78, v63, v91 :: v_dual_fmac_f32 v79, v73, v91
	v_dual_fmac_f32 v48, v44, v84 :: v_dual_fmac_f32 v49, v54, v84
	v_dual_fmac_f32 v58, v64, v84 :: v_dual_fmac_f32 v59, v74, v84
	s_wait_loadcnt 0x0
	v_dual_fmac_f32 v68, v44, v92 :: v_dual_fmac_f32 v69, v54, v92
	v_dual_fmac_f32 v78, v64, v92 :: v_dual_fmac_f32 v79, v74, v92
	v_dual_fmac_f32 v48, v45, v85 :: v_dual_fmac_f32 v49, v55, v85
	v_dual_fmac_f32 v58, v65, v85 :: v_dual_fmac_f32 v59, v75, v85
	v_dual_fmac_f32 v68, v45, v93 :: v_dual_fmac_f32 v69, v55, v93
	v_dual_fmac_f32 v78, v65, v93 :: v_dual_fmac_f32 v79, v75, v93
	v_dual_fmac_f32 v48, v46, v86 :: v_dual_fmac_f32 v49, v56, v86
	v_dual_fmac_f32 v58, v66, v86 :: v_dual_fmac_f32 v59, v76, v86
	v_dual_fmac_f32 v68, v46, v94 :: v_dual_fmac_f32 v69, v56, v94
	v_dual_fmac_f32 v78, v66, v94 :: v_dual_fmac_f32 v79, v76, v94
	v_dual_fmac_f32 v48, v47, v87 :: v_dual_fmac_f32 v49, v57, v87
	v_dual_fmac_f32 v58, v67, v87 :: v_dual_fmac_f32 v59, v77, v87
	v_dual_fmac_f32 v68, v47, v95 :: v_dual_fmac_f32 v69, v57, v95
	v_dual_fmac_f32 v78, v67, v95 :: v_dual_fmac_f32 v79, v77, v95
	v_dual_add_f32 v232, v232, v48 :: v_dual_add_f32 v233, v233, v49
	v_dual_add_f32 v234, v234, v58 :: v_dual_add_f32 v235, v235, v59
	v_dual_add_f32 v248, v248, v68 :: v_dual_add_f32 v249, v249, v69
	v_dual_add_f32 v250, v250, v78 :: v_dual_add_f32 v251, v251, v79
	.Lpx_b8_fold:
	v_dual_add_f32 v128, v128, v132 :: v_dual_add_f32 v129, v129, v133
	v_dual_add_f32 v130, v130, v134 :: v_dual_add_f32 v131, v131, v135
	v_dual_add_f32 v136, v136, v140 :: v_dual_add_f32 v137, v137, v141
	v_dual_add_f32 v138, v138, v142 :: v_dual_add_f32 v139, v139, v143
	v_dual_add_f32 v128, v128, v136 :: v_dual_add_f32 v129, v129, v137
	v_dual_add_f32 v130, v130, v138 :: v_dual_add_f32 v131, v131, v139
	v_dual_add_f32 v144, v144, v148 :: v_dual_add_f32 v145, v145, v149
	v_dual_add_f32 v146, v146, v150 :: v_dual_add_f32 v147, v147, v151
	v_dual_add_f32 v152, v152, v156 :: v_dual_add_f32 v153, v153, v157
	v_dual_add_f32 v154, v154, v158 :: v_dual_add_f32 v155, v155, v159
	v_dual_add_f32 v144, v144, v152 :: v_dual_add_f32 v145, v145, v153
	v_dual_add_f32 v146, v146, v154 :: v_dual_add_f32 v147, v147, v155
	v_dual_add_f32 v160, v160, v164 :: v_dual_add_f32 v161, v161, v165
	v_dual_add_f32 v162, v162, v166 :: v_dual_add_f32 v163, v163, v167
	v_dual_add_f32 v168, v168, v172 :: v_dual_add_f32 v169, v169, v173
	v_dual_add_f32 v170, v170, v174 :: v_dual_add_f32 v171, v171, v175
	v_dual_add_f32 v160, v160, v168 :: v_dual_add_f32 v161, v161, v169
	v_dual_add_f32 v162, v162, v170 :: v_dual_add_f32 v163, v163, v171
	v_dual_add_f32 v176, v176, v180 :: v_dual_add_f32 v177, v177, v181
	v_dual_add_f32 v178, v178, v182 :: v_dual_add_f32 v179, v179, v183
	v_dual_add_f32 v184, v184, v188 :: v_dual_add_f32 v185, v185, v189
	v_dual_add_f32 v186, v186, v190 :: v_dual_add_f32 v187, v187, v191
	v_dual_add_f32 v176, v176, v184 :: v_dual_add_f32 v177, v177, v185
	v_dual_add_f32 v178, v178, v186 :: v_dual_add_f32 v179, v179, v187
	v_dual_add_f32 v192, v192, v196 :: v_dual_add_f32 v193, v193, v197
	v_dual_add_f32 v194, v194, v198 :: v_dual_add_f32 v195, v195, v199
	v_dual_add_f32 v200, v200, v204 :: v_dual_add_f32 v201, v201, v205
	v_dual_add_f32 v202, v202, v206 :: v_dual_add_f32 v203, v203, v207
	v_dual_add_f32 v192, v192, v200 :: v_dual_add_f32 v193, v193, v201
	v_dual_add_f32 v194, v194, v202 :: v_dual_add_f32 v195, v195, v203
	v_dual_add_f32 v208, v208, v212 :: v_dual_add_f32 v209, v209, v213
	v_dual_add_f32 v210, v210, v214 :: v_dual_add_f32 v211, v211, v215
	v_dual_add_f32 v216, v216, v220 :: v_dual_add_f32 v217, v217, v221
	v_dual_add_f32 v218, v218, v222 :: v_dual_add_f32 v219, v219, v223
	v_dual_add_f32 v208, v208, v216 :: v_dual_add_f32 v209, v209, v217
	v_dual_add_f32 v210, v210, v218 :: v_dual_add_f32 v211, v211, v219
	v_dual_add_f32 v224, v224, v228 :: v_dual_add_f32 v225, v225, v229
	v_dual_add_f32 v226, v226, v230 :: v_dual_add_f32 v227, v227, v231
	v_dual_add_f32 v232, v232, v236 :: v_dual_add_f32 v233, v233, v237
	v_dual_add_f32 v234, v234, v238 :: v_dual_add_f32 v235, v235, v239
	v_dual_add_f32 v224, v224, v232 :: v_dual_add_f32 v225, v225, v233
	v_dual_add_f32 v226, v226, v234 :: v_dual_add_f32 v227, v227, v235
	v_dual_add_f32 v240, v240, v244 :: v_dual_add_f32 v241, v241, v245
	v_dual_add_f32 v242, v242, v246 :: v_dual_add_f32 v243, v243, v247
	v_dual_add_f32 v248, v248, v252 :: v_dual_add_f32 v249, v249, v253
	v_dual_add_f32 v250, v250, v254 :: v_dual_add_f32 v251, v251, v255
	v_dual_add_f32 v240, v240, v248 :: v_dual_add_f32 v241, v241, v249
	v_dual_add_f32 v242, v242, v250 :: v_dual_add_f32 v243, v243, v251
	ds_swizzle_b32 v8, v128 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v9, v129 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v10, v130 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v11, v131 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v12, v144 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v13, v145 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v14, v146 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v15, v147 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v16, v160 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v17, v161 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v18, v162 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v19, v163 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v20, v176 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v21, v177 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v22, v178 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v23, v179 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v24, v192 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v25, v193 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v26, v194 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v27, v195 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v28, v208 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v29, v209 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v30, v210 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v31, v211 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v32, v224 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v33, v225 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v34, v226 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v35, v227 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v36, v240 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v37, v241 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v38, v242 offset:swizzle(BITMASK_PERM,"1pppp")
	ds_swizzle_b32 v39, v243 offset:swizzle(BITMASK_PERM,"1pppp")
	s_wait_dscnt 0x1f
	v_add_f32_e32 v128, v128, v8
	s_wait_dscnt 0x1e
	v_add_f32_e32 v129, v129, v9
	s_wait_dscnt 0x1d
	v_add_f32_e32 v130, v130, v10
	s_wait_dscnt 0x1c
	v_add_f32_e32 v131, v131, v11
	s_wait_dscnt 0x1b
	v_add_f32_e32 v144, v144, v12
	s_wait_dscnt 0x1a
	v_add_f32_e32 v145, v145, v13
	s_wait_dscnt 0x19
	v_add_f32_e32 v146, v146, v14
	s_wait_dscnt 0x18
	v_add_f32_e32 v147, v147, v15
	s_wait_dscnt 0x17
	v_add_f32_e32 v160, v160, v16
	s_wait_dscnt 0x16
	v_add_f32_e32 v161, v161, v17
	s_wait_dscnt 0x15
	v_add_f32_e32 v162, v162, v18
	s_wait_dscnt 0x14
	v_add_f32_e32 v163, v163, v19
	s_wait_dscnt 0x13
	v_add_f32_e32 v176, v176, v20
	s_wait_dscnt 0x12
	v_add_f32_e32 v177, v177, v21
	s_wait_dscnt 0x11
	v_add_f32_e32 v178, v178, v22
	s_wait_dscnt 0x10
	v_add_f32_e32 v179, v179, v23
	s_wait_dscnt 0xf
	v_add_f32_e32 v192, v192, v24
	s_wait_dscnt 0xe
	v_add_f32_e32 v193, v193, v25
	s_wait_dscnt 0xd
	v_add_f32_e32 v194, v194, v26
	s_wait_dscnt 0xc
	v_add_f32_e32 v195, v195, v27
	s_wait_dscnt 0xb
	v_add_f32_e32 v208, v208, v28
	s_wait_dscnt 0xa
	v_add_f32_e32 v209, v209, v29
	s_wait_dscnt 0x9
	v_add_f32_e32 v210, v210, v30
	s_wait_dscnt 0x8
	v_add_f32_e32 v211, v211, v31
	s_wait_dscnt 0x7
	v_add_f32_e32 v224, v224, v32
	s_wait_dscnt 0x6
	v_add_f32_e32 v225, v225, v33
	s_wait_dscnt 0x5
	v_add_f32_e32 v226, v226, v34
	s_wait_dscnt 0x4
	v_add_f32_e32 v227, v227, v35
	s_wait_dscnt 0x3
	v_add_f32_e32 v240, v240, v36
	s_wait_dscnt 0x2
	v_add_f32_e32 v241, v241, v37
	s_wait_dscnt 0x1
	v_add_f32_e32 v242, v242, v38
	s_wait_dscnt 0x0
	v_add_f32_e32 v243, v243, v39
	v_cmp_gt_u32_e32 vcc_lo, 24, v0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v4, 0, 8, vcc_lo
	v_add_lshl_u32 v4, v4, v0, 2
	ds_bpermute_b32 v8, v4, v128
	ds_bpermute_b32 v9, v4, v129
	ds_bpermute_b32 v10, v4, v130
	ds_bpermute_b32 v11, v4, v131
	ds_bpermute_b32 v12, v4, v144
	ds_bpermute_b32 v13, v4, v145
	ds_bpermute_b32 v14, v4, v146
	ds_bpermute_b32 v15, v4, v147
	ds_bpermute_b32 v16, v4, v160
	ds_bpermute_b32 v17, v4, v161
	ds_bpermute_b32 v18, v4, v162
	ds_bpermute_b32 v19, v4, v163
	ds_bpermute_b32 v20, v4, v176
	ds_bpermute_b32 v21, v4, v177
	ds_bpermute_b32 v22, v4, v178
	ds_bpermute_b32 v23, v4, v179
	ds_bpermute_b32 v24, v4, v192
	ds_bpermute_b32 v25, v4, v193
	ds_bpermute_b32 v26, v4, v194
	ds_bpermute_b32 v27, v4, v195
	ds_bpermute_b32 v28, v4, v208
	ds_bpermute_b32 v29, v4, v209
	ds_bpermute_b32 v30, v4, v210
	ds_bpermute_b32 v31, v4, v211
	ds_bpermute_b32 v32, v4, v224
	ds_bpermute_b32 v33, v4, v225
	ds_bpermute_b32 v34, v4, v226
	ds_bpermute_b32 v35, v4, v227
	ds_bpermute_b32 v36, v4, v240
	ds_bpermute_b32 v37, v4, v241
	ds_bpermute_b32 v38, v4, v242
	ds_bpermute_b32 v39, v4, v243
	s_wait_dscnt 0x1f
	v_add_f32_e32 v128, v128, v8
	s_wait_dscnt 0x1e
	v_add_f32_e32 v129, v129, v9
	s_wait_dscnt 0x1d
	v_add_f32_e32 v130, v130, v10
	s_wait_dscnt 0x1c
	v_add_f32_e32 v131, v131, v11
	s_wait_dscnt 0x1b
	v_add_f32_e32 v144, v144, v12
	s_wait_dscnt 0x1a
	v_add_f32_e32 v145, v145, v13
	s_wait_dscnt 0x19
	v_add_f32_e32 v146, v146, v14
	s_wait_dscnt 0x18
	v_add_f32_e32 v147, v147, v15
	s_wait_dscnt 0x17
	v_add_f32_e32 v160, v160, v16
	s_wait_dscnt 0x16
	v_add_f32_e32 v161, v161, v17
	s_wait_dscnt 0x15
	v_add_f32_e32 v162, v162, v18
	s_wait_dscnt 0x14
	v_add_f32_e32 v163, v163, v19
	s_wait_dscnt 0x13
	v_add_f32_e32 v176, v176, v20
	s_wait_dscnt 0x12
	v_add_f32_e32 v177, v177, v21
	s_wait_dscnt 0x11
	v_add_f32_e32 v178, v178, v22
	s_wait_dscnt 0x10
	v_add_f32_e32 v179, v179, v23
	s_wait_dscnt 0xf
	v_add_f32_e32 v192, v192, v24
	s_wait_dscnt 0xe
	v_add_f32_e32 v193, v193, v25
	s_wait_dscnt 0xd
	v_add_f32_e32 v194, v194, v26
	s_wait_dscnt 0xc
	v_add_f32_e32 v195, v195, v27
	s_wait_dscnt 0xb
	v_add_f32_e32 v208, v208, v28
	s_wait_dscnt 0xa
	v_add_f32_e32 v209, v209, v29
	s_wait_dscnt 0x9
	v_add_f32_e32 v210, v210, v30
	s_wait_dscnt 0x8
	v_add_f32_e32 v211, v211, v31
	s_wait_dscnt 0x7
	v_add_f32_e32 v224, v224, v32
	s_wait_dscnt 0x6
	v_add_f32_e32 v225, v225, v33
	s_wait_dscnt 0x5
	v_add_f32_e32 v226, v226, v34
	s_wait_dscnt 0x4
	v_add_f32_e32 v227, v227, v35
	s_wait_dscnt 0x3
	v_add_f32_e32 v240, v240, v36
	s_wait_dscnt 0x2
	v_add_f32_e32 v241, v241, v37
	s_wait_dscnt 0x1
	v_add_f32_e32 v242, v242, v38
	s_wait_dscnt 0x0
	v_add_f32_e32 v243, v243, v39
	v_cmp_gt_u32_e32 vcc_lo, 28, v0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v4, 0, 4, vcc_lo
	v_add_lshl_u32 v4, v4, v0, 2
	ds_bpermute_b32 v8, v4, v128
	ds_bpermute_b32 v9, v4, v129
	ds_bpermute_b32 v10, v4, v130
	ds_bpermute_b32 v11, v4, v131
	ds_bpermute_b32 v12, v4, v144
	ds_bpermute_b32 v13, v4, v145
	ds_bpermute_b32 v14, v4, v146
	ds_bpermute_b32 v15, v4, v147
	ds_bpermute_b32 v16, v4, v160
	ds_bpermute_b32 v17, v4, v161
	ds_bpermute_b32 v18, v4, v162
	ds_bpermute_b32 v19, v4, v163
	ds_bpermute_b32 v20, v4, v176
	ds_bpermute_b32 v21, v4, v177
	ds_bpermute_b32 v22, v4, v178
	ds_bpermute_b32 v23, v4, v179
	ds_bpermute_b32 v24, v4, v192
	ds_bpermute_b32 v25, v4, v193
	ds_bpermute_b32 v26, v4, v194
	ds_bpermute_b32 v27, v4, v195
	ds_bpermute_b32 v28, v4, v208
	ds_bpermute_b32 v29, v4, v209
	ds_bpermute_b32 v30, v4, v210
	ds_bpermute_b32 v31, v4, v211
	ds_bpermute_b32 v32, v4, v224
	ds_bpermute_b32 v33, v4, v225
	ds_bpermute_b32 v34, v4, v226
	ds_bpermute_b32 v35, v4, v227
	ds_bpermute_b32 v36, v4, v240
	ds_bpermute_b32 v37, v4, v241
	ds_bpermute_b32 v38, v4, v242
	ds_bpermute_b32 v39, v4, v243
	s_wait_dscnt 0x1f
	v_add_f32_e32 v128, v128, v8
	s_wait_dscnt 0x1e
	v_add_f32_e32 v129, v129, v9
	s_wait_dscnt 0x1d
	v_add_f32_e32 v130, v130, v10
	s_wait_dscnt 0x1c
	v_add_f32_e32 v131, v131, v11
	s_wait_dscnt 0x1b
	v_add_f32_e32 v144, v144, v12
	s_wait_dscnt 0x1a
	v_add_f32_e32 v145, v145, v13
	s_wait_dscnt 0x19
	v_add_f32_e32 v146, v146, v14
	s_wait_dscnt 0x18
	v_add_f32_e32 v147, v147, v15
	s_wait_dscnt 0x17
	v_add_f32_e32 v160, v160, v16
	s_wait_dscnt 0x16
	v_add_f32_e32 v161, v161, v17
	s_wait_dscnt 0x15
	v_add_f32_e32 v162, v162, v18
	s_wait_dscnt 0x14
	v_add_f32_e32 v163, v163, v19
	s_wait_dscnt 0x13
	v_add_f32_e32 v176, v176, v20
	s_wait_dscnt 0x12
	v_add_f32_e32 v177, v177, v21
	s_wait_dscnt 0x11
	v_add_f32_e32 v178, v178, v22
	s_wait_dscnt 0x10
	v_add_f32_e32 v179, v179, v23
	s_wait_dscnt 0xf
	v_add_f32_e32 v192, v192, v24
	s_wait_dscnt 0xe
	v_add_f32_e32 v193, v193, v25
	s_wait_dscnt 0xd
	v_add_f32_e32 v194, v194, v26
	s_wait_dscnt 0xc
	v_add_f32_e32 v195, v195, v27
	s_wait_dscnt 0xb
	v_add_f32_e32 v208, v208, v28
	s_wait_dscnt 0xa
	v_add_f32_e32 v209, v209, v29
	s_wait_dscnt 0x9
	v_add_f32_e32 v210, v210, v30
	s_wait_dscnt 0x8
	v_add_f32_e32 v211, v211, v31
	s_wait_dscnt 0x7
	v_add_f32_e32 v224, v224, v32
	s_wait_dscnt 0x6
	v_add_f32_e32 v225, v225, v33
	s_wait_dscnt 0x5
	v_add_f32_e32 v226, v226, v34
	s_wait_dscnt 0x4
	v_add_f32_e32 v227, v227, v35
	s_wait_dscnt 0x3
	v_add_f32_e32 v240, v240, v36
	s_wait_dscnt 0x2
	v_add_f32_e32 v241, v241, v37
	s_wait_dscnt 0x1
	v_add_f32_e32 v242, v242, v38
	s_wait_dscnt 0x0
	v_add_f32_e32 v243, v243, v39
	v_cmp_gt_u32_e32 vcc_lo, 30, v0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v4, 0, 2, vcc_lo
	v_add_lshl_u32 v4, v4, v0, 2
	ds_bpermute_b32 v8, v4, v128
	ds_bpermute_b32 v9, v4, v129
	ds_bpermute_b32 v10, v4, v130
	ds_bpermute_b32 v11, v4, v131
	ds_bpermute_b32 v12, v4, v144
	ds_bpermute_b32 v13, v4, v145
	ds_bpermute_b32 v14, v4, v146
	ds_bpermute_b32 v15, v4, v147
	ds_bpermute_b32 v16, v4, v160
	ds_bpermute_b32 v17, v4, v161
	ds_bpermute_b32 v18, v4, v162
	ds_bpermute_b32 v19, v4, v163
	ds_bpermute_b32 v20, v4, v176
	ds_bpermute_b32 v21, v4, v177
	ds_bpermute_b32 v22, v4, v178
	ds_bpermute_b32 v23, v4, v179
	ds_bpermute_b32 v24, v4, v192
	ds_bpermute_b32 v25, v4, v193
	ds_bpermute_b32 v26, v4, v194
	ds_bpermute_b32 v27, v4, v195
	ds_bpermute_b32 v28, v4, v208
	ds_bpermute_b32 v29, v4, v209
	ds_bpermute_b32 v30, v4, v210
	ds_bpermute_b32 v31, v4, v211
	ds_bpermute_b32 v32, v4, v224
	ds_bpermute_b32 v33, v4, v225
	ds_bpermute_b32 v34, v4, v226
	ds_bpermute_b32 v35, v4, v227
	ds_bpermute_b32 v36, v4, v240
	ds_bpermute_b32 v37, v4, v241
	ds_bpermute_b32 v38, v4, v242
	ds_bpermute_b32 v39, v4, v243
	s_wait_dscnt 0x1f
	v_add_f32_e32 v128, v128, v8
	s_wait_dscnt 0x1e
	v_add_f32_e32 v129, v129, v9
	s_wait_dscnt 0x1d
	v_add_f32_e32 v130, v130, v10
	s_wait_dscnt 0x1c
	v_add_f32_e32 v131, v131, v11
	s_wait_dscnt 0x1b
	v_add_f32_e32 v144, v144, v12
	s_wait_dscnt 0x1a
	v_add_f32_e32 v145, v145, v13
	s_wait_dscnt 0x19
	v_add_f32_e32 v146, v146, v14
	s_wait_dscnt 0x18
	v_add_f32_e32 v147, v147, v15
	s_wait_dscnt 0x17
	v_add_f32_e32 v160, v160, v16
	s_wait_dscnt 0x16
	v_add_f32_e32 v161, v161, v17
	s_wait_dscnt 0x15
	v_add_f32_e32 v162, v162, v18
	s_wait_dscnt 0x14
	v_add_f32_e32 v163, v163, v19
	s_wait_dscnt 0x13
	v_add_f32_e32 v176, v176, v20
	s_wait_dscnt 0x12
	v_add_f32_e32 v177, v177, v21
	s_wait_dscnt 0x11
	v_add_f32_e32 v178, v178, v22
	s_wait_dscnt 0x10
	v_add_f32_e32 v179, v179, v23
	s_wait_dscnt 0xf
	v_add_f32_e32 v192, v192, v24
	s_wait_dscnt 0xe
	v_add_f32_e32 v193, v193, v25
	s_wait_dscnt 0xd
	v_add_f32_e32 v194, v194, v26
	s_wait_dscnt 0xc
	v_add_f32_e32 v195, v195, v27
	s_wait_dscnt 0xb
	v_add_f32_e32 v208, v208, v28
	s_wait_dscnt 0xa
	v_add_f32_e32 v209, v209, v29
	s_wait_dscnt 0x9
	v_add_f32_e32 v210, v210, v30
	s_wait_dscnt 0x8
	v_add_f32_e32 v211, v211, v31
	s_wait_dscnt 0x7
	v_add_f32_e32 v224, v224, v32
	s_wait_dscnt 0x6
	v_add_f32_e32 v225, v225, v33
	s_wait_dscnt 0x5
	v_add_f32_e32 v226, v226, v34
	s_wait_dscnt 0x4
	v_add_f32_e32 v227, v227, v35
	s_wait_dscnt 0x3
	v_add_f32_e32 v240, v240, v36
	s_wait_dscnt 0x2
	v_add_f32_e32 v241, v241, v37
	s_wait_dscnt 0x1
	v_add_f32_e32 v242, v242, v38
	s_wait_dscnt 0x0
	v_add_f32_e32 v243, v243, v39
	v_cmp_gt_u32_e32 vcc_lo, 31, v0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v4, 0, 1, vcc_lo
	v_add_lshl_u32 v4, v4, v0, 2
	ds_bpermute_b32 v8, v4, v128
	ds_bpermute_b32 v9, v4, v129
	ds_bpermute_b32 v10, v4, v130
	ds_bpermute_b32 v11, v4, v131
	ds_bpermute_b32 v12, v4, v144
	ds_bpermute_b32 v13, v4, v145
	ds_bpermute_b32 v14, v4, v146
	ds_bpermute_b32 v15, v4, v147
	ds_bpermute_b32 v16, v4, v160
	ds_bpermute_b32 v17, v4, v161
	ds_bpermute_b32 v18, v4, v162
	ds_bpermute_b32 v19, v4, v163
	ds_bpermute_b32 v20, v4, v176
	ds_bpermute_b32 v21, v4, v177
	ds_bpermute_b32 v22, v4, v178
	ds_bpermute_b32 v23, v4, v179
	ds_bpermute_b32 v24, v4, v192
	ds_bpermute_b32 v25, v4, v193
	ds_bpermute_b32 v26, v4, v194
	ds_bpermute_b32 v27, v4, v195
	ds_bpermute_b32 v28, v4, v208
	ds_bpermute_b32 v29, v4, v209
	ds_bpermute_b32 v30, v4, v210
	ds_bpermute_b32 v31, v4, v211
	ds_bpermute_b32 v32, v4, v224
	ds_bpermute_b32 v33, v4, v225
	ds_bpermute_b32 v34, v4, v226
	ds_bpermute_b32 v35, v4, v227
	ds_bpermute_b32 v36, v4, v240
	ds_bpermute_b32 v37, v4, v241
	ds_bpermute_b32 v38, v4, v242
	ds_bpermute_b32 v39, v4, v243
	s_wait_dscnt 0x1f
	v_add_f32_e32 v128, v128, v8
	s_wait_dscnt 0x1e
	v_add_f32_e32 v129, v129, v9
	s_wait_dscnt 0x1d
	v_add_f32_e32 v130, v130, v10
	s_wait_dscnt 0x1c
	v_add_f32_e32 v131, v131, v11
	s_wait_dscnt 0x1b
	v_add_f32_e32 v144, v144, v12
	s_wait_dscnt 0x1a
	v_add_f32_e32 v145, v145, v13
	s_wait_dscnt 0x19
	v_add_f32_e32 v146, v146, v14
	s_wait_dscnt 0x18
	v_add_f32_e32 v147, v147, v15
	s_wait_dscnt 0x17
	v_add_f32_e32 v160, v160, v16
	s_wait_dscnt 0x16
	v_add_f32_e32 v161, v161, v17
	s_wait_dscnt 0x15
	v_add_f32_e32 v162, v162, v18
	s_wait_dscnt 0x14
	v_add_f32_e32 v163, v163, v19
	s_wait_dscnt 0x13
	v_add_f32_e32 v176, v176, v20
	s_wait_dscnt 0x12
	v_add_f32_e32 v177, v177, v21
	s_wait_dscnt 0x11
	v_add_f32_e32 v178, v178, v22
	s_wait_dscnt 0x10
	v_add_f32_e32 v179, v179, v23
	s_wait_dscnt 0xf
	v_add_f32_e32 v192, v192, v24
	s_wait_dscnt 0xe
	v_add_f32_e32 v193, v193, v25
	s_wait_dscnt 0xd
	v_add_f32_e32 v194, v194, v26
	s_wait_dscnt 0xc
	v_add_f32_e32 v195, v195, v27
	s_wait_dscnt 0xb
	v_add_f32_e32 v208, v208, v28
	s_wait_dscnt 0xa
	v_add_f32_e32 v209, v209, v29
	s_wait_dscnt 0x9
	v_add_f32_e32 v210, v210, v30
	s_wait_dscnt 0x8
	v_add_f32_e32 v211, v211, v31
	s_wait_dscnt 0x7
	v_add_f32_e32 v224, v224, v32
	s_wait_dscnt 0x6
	v_add_f32_e32 v225, v225, v33
	s_wait_dscnt 0x5
	v_add_f32_e32 v226, v226, v34
	s_wait_dscnt 0x4
	v_add_f32_e32 v227, v227, v35
	s_wait_dscnt 0x3
	v_add_f32_e32 v240, v240, v36
	s_wait_dscnt 0x2
	v_add_f32_e32 v241, v241, v37
	s_wait_dscnt 0x1
	v_add_f32_e32 v242, v242, v38
	s_wait_dscnt 0x0
	v_add_f32_e32 v243, v243, v39
	v_cmpx_eq_u32_e32 0, v0
	v_mov_b32_e32 v40, s42
	s_mul_i32 s40, s39, 1
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s40, s40, s42
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v41, s40
	s_mul_i32 s40, s39, 2
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s40, s40, s42
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v42, s40
	s_mul_i32 s40, s39, 3
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s40, s40, s42
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v43, s40
	s_mul_i32 s40, s39, 4
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s40, s40, s42
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v44, s40
	s_mul_i32 s40, s39, 5
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s40, s40, s42
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v45, s40
	s_mul_i32 s40, s39, 6
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s40, s40, s42
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v46, s40
	s_mul_i32 s40, s39, 7
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s40, s40, s42
	s_wait_alu depctr_sa_sdst(0)
	v_mov_b32_e32 v47, s40
	global_store_b32 v40, v128, s[8:9]
	global_store_b32 v41, v144, s[8:9]
	global_store_b32 v42, v160, s[8:9]
	global_store_b32 v43, v176, s[8:9]
	global_store_b32 v44, v192, s[8:9]
	global_store_b32 v45, v208, s[8:9]
	global_store_b32 v46, v224, s[8:9]
	global_store_b32 v47, v240, s[8:9]
	s_cmp_le_i32 s13, 1
	s_cbranch_scc1 .Lpx_b8_stored
	global_store_b32 v40, v129, s[8:9] offset:4
	global_store_b32 v41, v145, s[8:9] offset:4
	global_store_b32 v42, v161, s[8:9] offset:4
	global_store_b32 v43, v177, s[8:9] offset:4
	global_store_b32 v44, v193, s[8:9] offset:4
	global_store_b32 v45, v209, s[8:9] offset:4
	global_store_b32 v46, v225, s[8:9] offset:4
	global_store_b32 v47, v241, s[8:9] offset:4
	s_cmp_le_i32 s13, 2
	s_cbranch_scc1 .Lpx_b8_stored
	global_store_b32 v40, v130, s[8:9] offset:8
	global_store_b32 v41, v146, s[8:9] offset:8
	global_store_b32 v42, v162, s[8:9] offset:8
	global_store_b32 v43, v178, s[8:9] offset:8
	global_store_b32 v44, v194, s[8:9] offset:8
	global_store_b32 v45, v210, s[8:9] offset:8
	global_store_b32 v46, v226, s[8:9] offset:8
	global_store_b32 v47, v242, s[8:9] offset:8
	s_cmp_le_i32 s13, 3
	s_cbranch_scc1 .Lpx_b8_stored
	global_store_b32 v40, v131, s[8:9] offset:12
	global_store_b32 v41, v147, s[8:9] offset:12
	global_store_b32 v42, v163, s[8:9] offset:12
	global_store_b32 v43, v179, s[8:9] offset:12
	global_store_b32 v44, v195, s[8:9] offset:12
	global_store_b32 v45, v211, s[8:9] offset:12
	global_store_b32 v46, v227, s[8:9] offset:12
	global_store_b32 v47, v243, s[8:9] offset:12
	.Lpx_b8_stored:
	s_wait_storecnt 0x0
	s_branch .Lpx_end
	.Lpx_end:
	s_endpgm
.Lgemv_mq4g256v2_xbatch_pm_end:
.size gemv_mq4g256v2_xbatch_pm, .Lgemv_mq4g256v2_xbatch_pm_end-gemv_mq4g256v2_xbatch_pm
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel gemv_mq4g256v2_xbatch_pm
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
	.amdhsa_next_free_vgpr 256
	.amdhsa_next_free_sgpr 43
	.amdhsa_reserve_vcc 1
	.amdhsa_float_round_mode_32 0
	.amdhsa_float_round_mode_16_64 0
	.amdhsa_float_denorm_mode_32 3
	.amdhsa_float_denorm_mode_16_64 3
	.amdhsa_fp16_overflow 0
	.amdhsa_workgroup_processor_mode 1
	.amdhsa_memory_ordered 1
	.amdhsa_forward_progress 1
	.amdhsa_inst_pref_size ((instprefsize(.Lgemv_mq4g256v2_xbatch_pm_end-gemv_mq4g256v2_xbatch_pm)<<4)&4080)>>4
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
      - .actual_access: write_only
        .address_space: global
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
    .name: gemv_mq4g256v2_xbatch_pm
    .private_segment_fixed_size: 0
    .sgpr_count: 45
    .sgpr_spill_count: 0
    .symbol: gemv_mq4g256v2_xbatch_pm.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count: 256
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
