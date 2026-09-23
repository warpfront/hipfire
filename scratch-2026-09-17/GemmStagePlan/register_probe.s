	.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
	.amdhsa_code_object_version 6
	.text
	.protected	pure                    ; -- Begin function pure
	.globl	pure
	.p2align	8
	.type	pure,@function
pure:                                   ; @pure
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_load_b32 s14, s[0:1], 0x8
	s_mov_b32 s2, 0x38383838
	v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v66, 0x30303030
	s_mov_b32 s3, s2
	s_mov_b32 s4, 0x38383839
	s_mov_b32 s5, 0x38383938
	v_dual_mov_b32 v78, s3 :: v_dual_mov_b32 v77, s2
	s_mov_b32 s3, 0x38383e38
	v_dual_mov_b32 v68, s5 :: v_dual_mov_b32 v67, s4
	s_mov_b32 s2, 0x3838383e
	s_mov_b32 s5, 0x38383f38
	s_mov_b32 s6, 0x3838383a
	s_mov_b32 s7, 0x38383a38
	s_mov_b32 s8, 0x3838383b
	s_mov_b32 s9, 0x38383b38
	s_mov_b32 s10, 0x3838383c
	s_mov_b32 s11, 0x38383c38
	s_wait_kmcnt 0x0
	s_add_co_i32 s14, s14, 0x30303030
	s_mov_b32 s12, 0x3838383d
	s_mov_b32 s13, 0x38383d38
	s_wait_alu depctr_sa_sdst(0)
	v_dual_mov_b32 v65, s14 :: v_dual_mov_b32 v80, s3
	s_mov_b32 s4, 0x3838383f
	v_dual_mov_b32 v79, s2 :: v_dual_mov_b32 v82, s5
	v_dual_mov_b32 v70, s7 :: v_dual_mov_b32 v69, s6
	v_dual_mov_b32 v72, s9 :: v_dual_mov_b32 v71, s8
	v_dual_mov_b32 v74, s11 :: v_dual_mov_b32 v73, s10
	v_dual_mov_b32 v76, s13 :: v_dual_mov_b32 v75, s12
	s_wait_alu depctr_sa_sdst(0)
	v_dual_mov_b32 v81, s4 :: v_dual_mov_b32 v2, v1
	v_dual_mov_b32 v3, v1 :: v_dual_mov_b32 v4, v1
	v_dual_mov_b32 v5, v1 :: v_dual_mov_b32 v6, v1
	v_dual_mov_b32 v7, v1 :: v_dual_mov_b32 v8, v1
	v_dual_mov_b32 v57, v1 :: v_dual_mov_b32 v58, v1
	v_dual_mov_b32 v59, v1 :: v_dual_mov_b32 v60, v1
	v_dual_mov_b32 v61, v1 :: v_dual_mov_b32 v62, v1
	v_dual_mov_b32 v63, v1 :: v_dual_mov_b32 v64, v1
	v_dual_mov_b32 v49, v1 :: v_dual_mov_b32 v50, v1
	v_dual_mov_b32 v51, v1 :: v_dual_mov_b32 v52, v1
	v_dual_mov_b32 v53, v1 :: v_dual_mov_b32 v54, v1
	v_dual_mov_b32 v55, v1 :: v_dual_mov_b32 v56, v1
	v_dual_mov_b32 v41, v1 :: v_dual_mov_b32 v42, v1
	v_dual_mov_b32 v43, v1 :: v_dual_mov_b32 v44, v1
	v_dual_mov_b32 v45, v1 :: v_dual_mov_b32 v46, v1
	v_dual_mov_b32 v47, v1 :: v_dual_mov_b32 v48, v1
	v_dual_mov_b32 v33, v1 :: v_dual_mov_b32 v34, v1
	v_dual_mov_b32 v35, v1 :: v_dual_mov_b32 v36, v1
	v_dual_mov_b32 v37, v1 :: v_dual_mov_b32 v38, v1
	v_dual_mov_b32 v39, v1 :: v_dual_mov_b32 v40, v1
	v_dual_mov_b32 v25, v1 :: v_dual_mov_b32 v26, v1
	v_dual_mov_b32 v27, v1 :: v_dual_mov_b32 v28, v1
	v_dual_mov_b32 v29, v1 :: v_dual_mov_b32 v30, v1
	v_dual_mov_b32 v31, v1 :: v_dual_mov_b32 v32, v1
	v_dual_mov_b32 v17, v1 :: v_dual_mov_b32 v18, v1
	v_dual_mov_b32 v19, v1 :: v_dual_mov_b32 v20, v1
	v_dual_mov_b32 v21, v1 :: v_dual_mov_b32 v22, v1
	v_dual_mov_b32 v23, v1 :: v_dual_mov_b32 v24, v1
	v_dual_mov_b32 v9, v1 :: v_dual_mov_b32 v10, v1
	v_dual_mov_b32 v11, v1 :: v_dual_mov_b32 v12, v1
	v_dual_mov_b32 v13, v1 :: v_dual_mov_b32 v14, v1
	v_dual_mov_b32 v15, v1 :: v_dual_mov_b32 v16, v1
	s_movk_i32 s2, 0x400
.LBB0_1:                                ; =>This Inner Loop Header: Depth=1
	;;#ASMSTART
	;;#ASMEND
	v_wmma_f32_16x16x16_fp8_fp8 v[1:8], v[65:66], v[77:78], v[1:8]
	v_wmma_f32_16x16x16_fp8_fp8 v[57:64], v[65:66], v[67:68], v[57:64]
	v_wmma_f32_16x16x16_fp8_fp8 v[49:56], v[65:66], v[69:70], v[49:56]
	v_wmma_f32_16x16x16_fp8_fp8 v[41:48], v[65:66], v[71:72], v[41:48]
	v_wmma_f32_16x16x16_fp8_fp8 v[33:40], v[65:66], v[73:74], v[33:40]
	v_wmma_f32_16x16x16_fp8_fp8 v[25:32], v[65:66], v[75:76], v[25:32]
	v_wmma_f32_16x16x16_fp8_fp8 v[17:24], v[65:66], v[79:80], v[17:24]
	v_wmma_f32_16x16x16_fp8_fp8 v[9:16], v[65:66], v[81:82], v[9:16]
	; sched_barrier mask(0x00000000)
	v_wmma_f32_16x16x16_fp8_fp8 v[1:8], v[65:66], v[77:78], v[1:8]
	v_wmma_f32_16x16x16_fp8_fp8 v[57:64], v[65:66], v[67:68], v[57:64]
	v_wmma_f32_16x16x16_fp8_fp8 v[49:56], v[65:66], v[69:70], v[49:56]
	v_wmma_f32_16x16x16_fp8_fp8 v[41:48], v[65:66], v[71:72], v[41:48]
	v_wmma_f32_16x16x16_fp8_fp8 v[33:40], v[65:66], v[73:74], v[33:40]
	v_wmma_f32_16x16x16_fp8_fp8 v[25:32], v[65:66], v[75:76], v[25:32]
	v_wmma_f32_16x16x16_fp8_fp8 v[17:24], v[65:66], v[79:80], v[17:24]
	v_wmma_f32_16x16x16_fp8_fp8 v[9:16], v[65:66], v[81:82], v[9:16]
	; sched_barrier mask(0x00000000)
	v_wmma_f32_16x16x16_fp8_fp8 v[1:8], v[65:66], v[77:78], v[1:8]
	v_wmma_f32_16x16x16_fp8_fp8 v[57:64], v[65:66], v[67:68], v[57:64]
	v_wmma_f32_16x16x16_fp8_fp8 v[49:56], v[65:66], v[69:70], v[49:56]
	v_wmma_f32_16x16x16_fp8_fp8 v[41:48], v[65:66], v[71:72], v[41:48]
	v_wmma_f32_16x16x16_fp8_fp8 v[33:40], v[65:66], v[73:74], v[33:40]
	v_wmma_f32_16x16x16_fp8_fp8 v[25:32], v[65:66], v[75:76], v[25:32]
	v_wmma_f32_16x16x16_fp8_fp8 v[17:24], v[65:66], v[79:80], v[17:24]
	v_wmma_f32_16x16x16_fp8_fp8 v[9:16], v[65:66], v[81:82], v[9:16]
	; sched_barrier mask(0x00000000)
	v_wmma_f32_16x16x16_fp8_fp8 v[1:8], v[65:66], v[77:78], v[1:8]
	v_wmma_f32_16x16x16_fp8_fp8 v[57:64], v[65:66], v[67:68], v[57:64]
	v_wmma_f32_16x16x16_fp8_fp8 v[49:56], v[65:66], v[69:70], v[49:56]
	v_wmma_f32_16x16x16_fp8_fp8 v[41:48], v[65:66], v[71:72], v[41:48]
	v_wmma_f32_16x16x16_fp8_fp8 v[33:40], v[65:66], v[73:74], v[33:40]
	v_wmma_f32_16x16x16_fp8_fp8 v[25:32], v[65:66], v[75:76], v[25:32]
	v_wmma_f32_16x16x16_fp8_fp8 v[17:24], v[65:66], v[79:80], v[17:24]
	v_wmma_f32_16x16x16_fp8_fp8 v[9:16], v[65:66], v[81:82], v[9:16]
	; sched_barrier mask(0x00000000)
	v_wmma_f32_16x16x16_fp8_fp8 v[1:8], v[65:66], v[77:78], v[1:8]
	v_wmma_f32_16x16x16_fp8_fp8 v[57:64], v[65:66], v[67:68], v[57:64]
	v_wmma_f32_16x16x16_fp8_fp8 v[49:56], v[65:66], v[69:70], v[49:56]
	v_wmma_f32_16x16x16_fp8_fp8 v[41:48], v[65:66], v[71:72], v[41:48]
	v_wmma_f32_16x16x16_fp8_fp8 v[33:40], v[65:66], v[73:74], v[33:40]
	v_wmma_f32_16x16x16_fp8_fp8 v[25:32], v[65:66], v[75:76], v[25:32]
	v_wmma_f32_16x16x16_fp8_fp8 v[17:24], v[65:66], v[79:80], v[17:24]
	v_wmma_f32_16x16x16_fp8_fp8 v[9:16], v[65:66], v[81:82], v[9:16]
	; sched_barrier mask(0x00000000)
	v_wmma_f32_16x16x16_fp8_fp8 v[1:8], v[65:66], v[77:78], v[1:8]
	v_wmma_f32_16x16x16_fp8_fp8 v[57:64], v[65:66], v[67:68], v[57:64]
	v_wmma_f32_16x16x16_fp8_fp8 v[49:56], v[65:66], v[69:70], v[49:56]
	v_wmma_f32_16x16x16_fp8_fp8 v[41:48], v[65:66], v[71:72], v[41:48]
	v_wmma_f32_16x16x16_fp8_fp8 v[33:40], v[65:66], v[73:74], v[33:40]
	v_wmma_f32_16x16x16_fp8_fp8 v[25:32], v[65:66], v[75:76], v[25:32]
	v_wmma_f32_16x16x16_fp8_fp8 v[17:24], v[65:66], v[79:80], v[17:24]
	v_wmma_f32_16x16x16_fp8_fp8 v[9:16], v[65:66], v[81:82], v[9:16]
	; sched_barrier mask(0x00000000)
	v_wmma_f32_16x16x16_fp8_fp8 v[1:8], v[65:66], v[77:78], v[1:8]
	v_wmma_f32_16x16x16_fp8_fp8 v[57:64], v[65:66], v[67:68], v[57:64]
	v_wmma_f32_16x16x16_fp8_fp8 v[49:56], v[65:66], v[69:70], v[49:56]
	v_wmma_f32_16x16x16_fp8_fp8 v[41:48], v[65:66], v[71:72], v[41:48]
	v_wmma_f32_16x16x16_fp8_fp8 v[33:40], v[65:66], v[73:74], v[33:40]
	v_wmma_f32_16x16x16_fp8_fp8 v[25:32], v[65:66], v[75:76], v[25:32]
	v_wmma_f32_16x16x16_fp8_fp8 v[17:24], v[65:66], v[79:80], v[17:24]
	v_wmma_f32_16x16x16_fp8_fp8 v[9:16], v[65:66], v[81:82], v[9:16]
	; sched_barrier mask(0x00000000)
	v_wmma_f32_16x16x16_fp8_fp8 v[1:8], v[65:66], v[77:78], v[1:8]
	v_wmma_f32_16x16x16_fp8_fp8 v[57:64], v[65:66], v[67:68], v[57:64]
	v_wmma_f32_16x16x16_fp8_fp8 v[49:56], v[65:66], v[69:70], v[49:56]
	v_wmma_f32_16x16x16_fp8_fp8 v[41:48], v[65:66], v[71:72], v[41:48]
	v_wmma_f32_16x16x16_fp8_fp8 v[33:40], v[65:66], v[73:74], v[33:40]
	v_wmma_f32_16x16x16_fp8_fp8 v[25:32], v[65:66], v[75:76], v[25:32]
	v_wmma_f32_16x16x16_fp8_fp8 v[17:24], v[65:66], v[79:80], v[17:24]
	v_wmma_f32_16x16x16_fp8_fp8 v[9:16], v[65:66], v[81:82], v[9:16]
	; sched_barrier mask(0x00000000)
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s2, s2, -1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s2, 0
	s_cbranch_scc1 .LBB0_1
; %bb.2:
	v_add_f32_e32 v1, 0, v1
	s_load_b64 s[0:1], s[0:1], 0x0
	v_lshl_or_b32 v0, ttmp9, 8, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v2, v1
	v_add_f32_e32 v1, v3, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v4, v1
	v_add_f32_e32 v1, v5, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v6, v1
	v_add_f32_e32 v1, v7, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v8, v1
	v_add_f32_e32 v1, v57, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v58, v1
	v_add_f32_e32 v1, v59, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v60, v1
	v_add_f32_e32 v1, v61, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v62, v1
	v_add_f32_e32 v1, v63, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v64, v1
	v_add_f32_e32 v1, v49, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v50, v1
	v_add_f32_e32 v1, v51, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v52, v1
	v_add_f32_e32 v1, v53, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v54, v1
	v_add_f32_e32 v1, v55, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v56, v1
	v_add_f32_e32 v1, v41, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v42, v1
	v_add_f32_e32 v1, v43, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v44, v1
	v_add_f32_e32 v1, v45, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v46, v1
	v_add_f32_e32 v1, v47, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v48, v1
	v_add_f32_e32 v1, v33, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v34, v1
	v_add_f32_e32 v1, v35, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v36, v1
	v_add_f32_e32 v1, v37, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v38, v1
	v_add_f32_e32 v1, v39, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v40, v1
	v_add_f32_e32 v1, v25, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v26, v1
	v_add_f32_e32 v1, v27, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v28, v1
	v_add_f32_e32 v1, v29, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v30, v1
	v_add_f32_e32 v1, v31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v32, v1
	v_add_f32_e32 v1, v17, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v18, v1
	v_add_f32_e32 v1, v19, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v20, v1
	v_add_f32_e32 v1, v21, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v22, v1
	v_add_f32_e32 v1, v23, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v24, v1
	v_add_f32_e32 v1, v9, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v10, v1
	v_add_f32_e32 v1, v11, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v12, v1
	v_add_f32_e32 v1, v13, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_add_f32 v2, v14, v1 :: v_dual_mov_b32 v1, 0
	v_add_f32_e32 v2, v15, v2
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	v_add_f32_e32 v2, v16, v2
	s_wait_kmcnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v0, vcc_lo, s0, v0
	v_add_co_ci_u32_e64 v1, null, s1, v1, vcc_lo
	global_store_b32 v[0:1], v2, off
	s_endpgm
.Lfunc_end0:
	.size	pure, .Lfunc_end0-pure
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel pure
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 12
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
		.amdhsa_next_free_vgpr 83
		.amdhsa_next_free_sgpr 15
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end0-pure)<<4)&4080)>>4
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
	.set .Lpure.num_vgpr, 83
	.set .Lpure.num_agpr, 0
	.set .Lpure.numbered_sgpr, 15
	.set .Lpure.num_named_barrier, 0
	.set .Lpure.private_seg_size, 0
	.set .Lpure.uses_vcc, 1
	.set .Lpure.uses_flat_scratch, 0
	.set .Lpure.has_dyn_sized_stack, 0
	.set .Lpure.has_recursion, 0
	.set .Lpure.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 1468
; TotalNumSgprs: 17
; NumVgprs: 83
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 10
; NumSGPRsForWavesPerEU: 17
; NumVGPRsForWavesPerEU: 83
; Occupancy: 16
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.protected	fold                    ; -- Begin function fold
	.globl	fold
	.p2align	8
	.type	fold,@function
fold:                                   ; @fold
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_load_b32 s14, s[0:1], 0x8
	s_mov_b32 s2, 0x38383838
	s_mov_b32 s8, 0x3838383b
	s_mov_b32 s9, 0x38383b38
	s_mov_b32 s3, s2
	v_add_nc_u32_e32 v1, 2, v0
	s_mov_b32 s4, 0x38383839
	s_mov_b32 s5, 0x38383938
	s_mov_b32 s6, 0x3838383a
	s_mov_b32 s7, 0x38383a38
	v_dual_mov_b32 v7, s8 :: v_dual_mov_b32 v10, s3
	s_mov_b32 s10, 0x3838383c
	s_mov_b32 s11, 0x38383c38
	s_mov_b32 s13, 0x38383d38
	v_dual_mov_b32 v9, s2 :: v_dual_mov_b32 v12, s11
	s_mov_b32 s12, 0x3838383d
	v_dual_mov_b32 v3, s4 :: v_dual_mov_b32 v4, s5
	s_wait_kmcnt 0x0
	s_cvt_f32_i32 s3, s14
	s_add_co_i32 s14, s14, 0x30303030
	v_dual_mov_b32 v5, s6 :: v_dual_mov_b32 v6, s7
	s_mov_b32 s5, 0x38383e38
	s_mov_b32 s7, 0x38383f38
	v_mov_b32_e32 v8, s9
	v_cvt_f32_u32_e32 v20, v1
	v_dual_mov_b32 v1, s14 :: v_dual_mov_b32 v14, s13
	s_mov_b32 s4, 0x3838383e
	s_mov_b32 s6, 0x3838383f
	s_wait_alu depctr_sa_sdst(0)
	v_dual_mov_b32 v11, s10 :: v_dual_mov_b32 v16, s5
	v_dual_mov_b32 v13, s12 :: v_dual_mov_b32 v18, s7
	v_dual_mov_b32 v19, 0 :: v_dual_mov_b32 v2, 0x30303030
	v_dual_mov_b32 v15, s4 :: v_dual_mov_b32 v26, 0
	v_dual_mov_b32 v17, s6 :: v_dual_mov_b32 v28, 0
	v_dual_mov_b32 v25, 0 :: v_dual_mov_b32 v30, 0
	v_dual_mov_b32 v27, 0 :: v_dual_mov_b32 v32, 0
	v_dual_mov_b32 v29, 0 :: v_dual_mov_b32 v34, 0
	v_dual_mov_b32 v31, 0 :: v_dual_mov_b32 v36, 0
	v_dual_mov_b32 v33, 0 :: v_dual_mov_b32 v38, 0
	v_dual_mov_b32 v35, 0 :: v_dual_mov_b32 v40, 0
	v_dual_mov_b32 v37, 0 :: v_dual_mov_b32 v42, 0
	v_dual_mov_b32 v39, 0 :: v_dual_mov_b32 v44, 0
	v_dual_mov_b32 v41, 0 :: v_dual_mov_b32 v46, 0
	v_dual_mov_b32 v43, 0 :: v_dual_mov_b32 v48, 0
	v_dual_mov_b32 v45, 0 :: v_dual_mov_b32 v50, 0
	v_dual_mov_b32 v47, 0 :: v_dual_mov_b32 v52, 0
	v_dual_mov_b32 v49, 0 :: v_dual_mov_b32 v54, 0
	v_dual_mov_b32 v51, 0 :: v_dual_mov_b32 v56, 0
	v_dual_mov_b32 v53, 0 :: v_dual_mov_b32 v58, 0
	v_dual_mov_b32 v55, 0 :: v_dual_mov_b32 v60, 0
	v_dual_mov_b32 v57, 0 :: v_dual_mov_b32 v62, 0
	v_dual_mov_b32 v59, 0 :: v_dual_mov_b32 v64, 0
	v_dual_mov_b32 v61, 0 :: v_dual_mov_b32 v66, 0
	v_dual_mov_b32 v63, 0 :: v_dual_mov_b32 v68, 0
	v_dual_mov_b32 v65, 0 :: v_dual_mov_b32 v70, 0
	v_dual_mov_b32 v67, 0 :: v_dual_mov_b32 v72, 0
	v_dual_mov_b32 v69, 0 :: v_dual_mov_b32 v74, 0
	v_dual_mov_b32 v71, 0 :: v_dual_mov_b32 v76, 0
	v_dual_mov_b32 v73, 0 :: v_dual_mov_b32 v78, 0
	v_dual_mov_b32 v75, 0 :: v_dual_mov_b32 v80, 0
	v_dual_mov_b32 v77, 0 :: v_dual_mov_b32 v82, 0
	v_dual_mov_b32 v79, 0 :: v_dual_mov_b32 v24, 0
	v_dual_mov_b32 v81, 0 :: v_dual_mov_b32 v22, 0
	v_mov_b32_e32 v83, 0
	v_mov_b32_e32 v23, 0
	v_mov_b32_e32 v21, 0
	s_mul_f32 s2, s3, 0x3c23d70a
	s_mul_f32 s3, s3, 0x3d000000
	s_movk_i32 s4, 0x400
.LBB1_1:                                ; =>This Inner Loop Header: Depth=1
	;;#ASMSTART
	;;#ASMEND
	v_wmma_f32_16x16x16_fp8_fp8 v[84:91], v[1:2], v[9:10], 0
	v_wmma_f32_16x16x16_fp8_fp8 v[92:99], v[1:2], v[3:4], 0
	v_wmma_f32_16x16x16_fp8_fp8 v[100:107], v[1:2], v[5:6], 0
	v_wmma_f32_16x16x16_fp8_fp8 v[108:115], v[1:2], v[7:8], 0
	v_wmma_f32_16x16x16_fp8_fp8 v[116:123], v[1:2], v[11:12], 0
	v_wmma_f32_16x16x16_fp8_fp8 v[124:131], v[1:2], v[13:14], 0
	v_wmma_f32_16x16x16_fp8_fp8 v[132:139], v[1:2], v[15:16], 0
	v_wmma_f32_16x16x16_fp8_fp8 v[140:147], v[1:2], v[17:18], 0
	; sched_barrier mask(0x00000000)
	v_wmma_f32_16x16x16_fp8_fp8 v[84:91], v[1:2], v[9:10], v[84:91]
	v_wmma_f32_16x16x16_fp8_fp8 v[92:99], v[1:2], v[3:4], v[92:99]
	v_wmma_f32_16x16x16_fp8_fp8 v[100:107], v[1:2], v[5:6], v[100:107]
	v_wmma_f32_16x16x16_fp8_fp8 v[108:115], v[1:2], v[7:8], v[108:115]
	v_wmma_f32_16x16x16_fp8_fp8 v[116:123], v[1:2], v[11:12], v[116:123]
	v_wmma_f32_16x16x16_fp8_fp8 v[124:131], v[1:2], v[13:14], v[124:131]
	v_wmma_f32_16x16x16_fp8_fp8 v[132:139], v[1:2], v[15:16], v[132:139]
	v_wmma_f32_16x16x16_fp8_fp8 v[140:147], v[1:2], v[17:18], v[140:147]
	; sched_barrier mask(0x00000000)
	v_wmma_f32_16x16x16_fp8_fp8 v[84:91], v[1:2], v[9:10], v[84:91]
	v_wmma_f32_16x16x16_fp8_fp8 v[92:99], v[1:2], v[3:4], v[92:99]
	v_wmma_f32_16x16x16_fp8_fp8 v[100:107], v[1:2], v[5:6], v[100:107]
	v_wmma_f32_16x16x16_fp8_fp8 v[108:115], v[1:2], v[7:8], v[108:115]
	v_wmma_f32_16x16x16_fp8_fp8 v[116:123], v[1:2], v[11:12], v[116:123]
	v_wmma_f32_16x16x16_fp8_fp8 v[124:131], v[1:2], v[13:14], v[124:131]
	v_wmma_f32_16x16x16_fp8_fp8 v[132:139], v[1:2], v[15:16], v[132:139]
	v_wmma_f32_16x16x16_fp8_fp8 v[140:147], v[1:2], v[17:18], v[140:147]
	; sched_barrier mask(0x00000000)
	v_wmma_f32_16x16x16_fp8_fp8 v[84:91], v[1:2], v[9:10], v[84:91]
	v_wmma_f32_16x16x16_fp8_fp8 v[92:99], v[1:2], v[3:4], v[92:99]
	v_wmma_f32_16x16x16_fp8_fp8 v[100:107], v[1:2], v[5:6], v[100:107]
	v_wmma_f32_16x16x16_fp8_fp8 v[108:115], v[1:2], v[7:8], v[108:115]
	v_wmma_f32_16x16x16_fp8_fp8 v[116:123], v[1:2], v[11:12], v[116:123]
	v_wmma_f32_16x16x16_fp8_fp8 v[124:131], v[1:2], v[13:14], v[124:131]
	v_wmma_f32_16x16x16_fp8_fp8 v[132:139], v[1:2], v[15:16], v[132:139]
	v_wmma_f32_16x16x16_fp8_fp8 v[140:147], v[1:2], v[17:18], v[140:147]
	; sched_barrier mask(0x00000000)
	v_wmma_f32_16x16x16_fp8_fp8 v[84:91], v[1:2], v[9:10], v[84:91]
	v_wmma_f32_16x16x16_fp8_fp8 v[92:99], v[1:2], v[3:4], v[92:99]
	v_wmma_f32_16x16x16_fp8_fp8 v[100:107], v[1:2], v[5:6], v[100:107]
	v_wmma_f32_16x16x16_fp8_fp8 v[108:115], v[1:2], v[7:8], v[108:115]
	v_wmma_f32_16x16x16_fp8_fp8 v[116:123], v[1:2], v[11:12], v[116:123]
	v_wmma_f32_16x16x16_fp8_fp8 v[124:131], v[1:2], v[13:14], v[124:131]
	v_wmma_f32_16x16x16_fp8_fp8 v[132:139], v[1:2], v[15:16], v[132:139]
	v_wmma_f32_16x16x16_fp8_fp8 v[140:147], v[1:2], v[17:18], v[140:147]
	; sched_barrier mask(0x00000000)
	v_wmma_f32_16x16x16_fp8_fp8 v[84:91], v[1:2], v[9:10], v[84:91]
	v_wmma_f32_16x16x16_fp8_fp8 v[92:99], v[1:2], v[3:4], v[92:99]
	v_wmma_f32_16x16x16_fp8_fp8 v[100:107], v[1:2], v[5:6], v[100:107]
	v_wmma_f32_16x16x16_fp8_fp8 v[108:115], v[1:2], v[7:8], v[108:115]
	v_wmma_f32_16x16x16_fp8_fp8 v[116:123], v[1:2], v[11:12], v[116:123]
	v_wmma_f32_16x16x16_fp8_fp8 v[124:131], v[1:2], v[13:14], v[124:131]
	v_wmma_f32_16x16x16_fp8_fp8 v[132:139], v[1:2], v[15:16], v[132:139]
	v_wmma_f32_16x16x16_fp8_fp8 v[140:147], v[1:2], v[17:18], v[140:147]
	; sched_barrier mask(0x00000000)
	v_wmma_f32_16x16x16_fp8_fp8 v[84:91], v[1:2], v[9:10], v[84:91]
	v_wmma_f32_16x16x16_fp8_fp8 v[92:99], v[1:2], v[3:4], v[92:99]
	v_wmma_f32_16x16x16_fp8_fp8 v[100:107], v[1:2], v[5:6], v[100:107]
	v_wmma_f32_16x16x16_fp8_fp8 v[108:115], v[1:2], v[7:8], v[108:115]
	v_wmma_f32_16x16x16_fp8_fp8 v[116:123], v[1:2], v[11:12], v[116:123]
	v_wmma_f32_16x16x16_fp8_fp8 v[124:131], v[1:2], v[13:14], v[124:131]
	v_wmma_f32_16x16x16_fp8_fp8 v[132:139], v[1:2], v[15:16], v[132:139]
	v_wmma_f32_16x16x16_fp8_fp8 v[140:147], v[1:2], v[17:18], v[140:147]
	; sched_barrier mask(0x00000000)
	v_wmma_f32_16x16x16_fp8_fp8 v[84:91], v[1:2], v[9:10], v[84:91]
	v_wmma_f32_16x16x16_fp8_fp8 v[92:99], v[1:2], v[3:4], v[92:99]
	v_wmma_f32_16x16x16_fp8_fp8 v[100:107], v[1:2], v[5:6], v[100:107]
	v_wmma_f32_16x16x16_fp8_fp8 v[108:115], v[1:2], v[7:8], v[108:115]
	v_wmma_f32_16x16x16_fp8_fp8 v[116:123], v[1:2], v[11:12], v[116:123]
	v_wmma_f32_16x16x16_fp8_fp8 v[124:131], v[1:2], v[13:14], v[124:131]
	v_wmma_f32_16x16x16_fp8_fp8 v[132:139], v[1:2], v[15:16], v[132:139]
	v_wmma_f32_16x16x16_fp8_fp8 v[140:147], v[1:2], v[17:18], v[140:147]
	; sched_barrier mask(0x00000000)
	s_wait_alu depctr_sa_sdst(0)
	v_dual_fmac_f32 v25, s2, v85 :: v_dual_fmac_f32 v28, s2, v88
	v_dual_fmac_f32 v57, s2, v117 :: v_dual_fmac_f32 v60, s2, v120
	v_fmac_f32_e32 v63, s2, v123
	v_dual_fmac_f32 v69, s2, v129 :: v_dual_fmac_f32 v72, s2, v132
	v_dual_fmac_f32 v73, s2, v133 :: v_dual_fmac_f32 v76, s2, v136
	v_dual_fmac_f32 v79, s2, v139 :: v_dual_fmac_f32 v80, s2, v140
	v_dual_fmac_f32 v19, s2, v84 :: v_dual_fmac_f32 v26, s2, v86
	v_dual_fmac_f32 v27, s2, v87 :: v_dual_fmac_f32 v30, s2, v90
	v_dual_fmac_f32 v29, s2, v89 :: v_dual_fmac_f32 v32, s2, v92
	v_dual_fmac_f32 v31, s2, v91 :: v_dual_fmac_f32 v34, s2, v94
	v_dual_fmac_f32 v33, s2, v93 :: v_dual_fmac_f32 v36, s2, v96
	v_dual_fmac_f32 v35, s2, v95 :: v_dual_fmac_f32 v38, s2, v98
	v_dual_fmac_f32 v37, s2, v97 :: v_dual_fmac_f32 v40, s2, v100
	v_dual_fmac_f32 v39, s2, v99 :: v_dual_fmac_f32 v42, s2, v102
	v_dual_fmac_f32 v41, s2, v101 :: v_dual_fmac_f32 v44, s2, v104
	v_dual_fmac_f32 v43, s2, v103 :: v_dual_fmac_f32 v46, s2, v106
	v_dual_fmac_f32 v45, s2, v105 :: v_dual_fmac_f32 v48, s2, v108
	v_dual_fmac_f32 v47, s2, v107 :: v_dual_fmac_f32 v50, s2, v110
	v_dual_fmac_f32 v49, s2, v109 :: v_dual_fmac_f32 v52, s2, v112
	v_dual_fmac_f32 v51, s2, v111 :: v_dual_fmac_f32 v54, s2, v114
	v_dual_fmac_f32 v53, s2, v113 :: v_dual_fmac_f32 v56, s2, v116
	v_dual_fmac_f32 v55, s2, v115 :: v_dual_fmac_f32 v58, s2, v118
	v_dual_fmac_f32 v59, s2, v119 :: v_dual_fmac_f32 v62, s2, v122
	v_dual_fmac_f32 v61, s2, v121 :: v_dual_fmac_f32 v26, s3, v20
	v_dual_fmac_f32 v28, s3, v20 :: v_dual_fmac_f32 v57, s3, v20
	v_fmac_f32_e32 v66, s2, v126
	v_dual_fmac_f32 v64, s2, v124 :: v_dual_fmac_f32 v65, s2, v125
	v_dual_fmac_f32 v67, s2, v127 :: v_dual_fmac_f32 v68, s2, v128
	v_dual_fmac_f32 v63, s3, v20 :: v_dual_fmac_f32 v70, s2, v130
	v_dual_fmac_f32 v71, s2, v131 :: v_dual_fmac_f32 v72, s3, v20
	v_dual_fmac_f32 v69, s3, v20 :: v_dual_fmac_f32 v74, s2, v134
	v_dual_fmac_f32 v75, s2, v135 :: v_dual_fmac_f32 v76, s3, v20
	v_dual_fmac_f32 v77, s2, v137 :: v_dual_fmac_f32 v78, s2, v138
	v_dual_fmac_f32 v73, s3, v20 :: v_dual_fmac_f32 v82, s2, v142
	v_fmac_f32_e32 v81, s2, v141
	v_dual_fmac_f32 v79, s3, v20 :: v_dual_fmac_f32 v22, s2, v146
	v_dual_fmac_f32 v83, s2, v143 :: v_dual_fmac_f32 v24, s2, v144
	v_fmac_f32_e32 v23, s2, v145
	v_fmac_f32_e32 v21, s2, v147
	v_dual_fmac_f32 v19, s3, v20 :: v_dual_fmac_f32 v30, s3, v20
	v_dual_fmac_f32 v25, s3, v20 :: v_dual_fmac_f32 v32, s3, v20
	v_dual_fmac_f32 v27, s3, v20 :: v_dual_fmac_f32 v34, s3, v20
	v_dual_fmac_f32 v29, s3, v20 :: v_dual_fmac_f32 v36, s3, v20
	v_dual_fmac_f32 v31, s3, v20 :: v_dual_fmac_f32 v38, s3, v20
	v_dual_fmac_f32 v33, s3, v20 :: v_dual_fmac_f32 v40, s3, v20
	v_dual_fmac_f32 v35, s3, v20 :: v_dual_fmac_f32 v42, s3, v20
	v_dual_fmac_f32 v37, s3, v20 :: v_dual_fmac_f32 v44, s3, v20
	v_dual_fmac_f32 v39, s3, v20 :: v_dual_fmac_f32 v46, s3, v20
	v_dual_fmac_f32 v41, s3, v20 :: v_dual_fmac_f32 v48, s3, v20
	v_dual_fmac_f32 v43, s3, v20 :: v_dual_fmac_f32 v50, s3, v20
	v_dual_fmac_f32 v45, s3, v20 :: v_dual_fmac_f32 v52, s3, v20
	v_dual_fmac_f32 v47, s3, v20 :: v_dual_fmac_f32 v54, s3, v20
	v_dual_fmac_f32 v49, s3, v20 :: v_dual_fmac_f32 v56, s3, v20
	v_dual_fmac_f32 v51, s3, v20 :: v_dual_fmac_f32 v58, s3, v20
	v_dual_fmac_f32 v53, s3, v20 :: v_dual_fmac_f32 v60, s3, v20
	v_dual_fmac_f32 v55, s3, v20 :: v_dual_fmac_f32 v62, s3, v20
	v_dual_fmac_f32 v59, s3, v20 :: v_dual_fmac_f32 v64, s3, v20
	v_dual_fmac_f32 v61, s3, v20 :: v_dual_fmac_f32 v66, s3, v20
	v_dual_fmac_f32 v65, s3, v20 :: v_dual_fmac_f32 v68, s3, v20
	v_dual_fmac_f32 v67, s3, v20 :: v_dual_fmac_f32 v70, s3, v20
	v_dual_fmac_f32 v71, s3, v20 :: v_dual_fmac_f32 v74, s3, v20
	v_dual_fmac_f32 v75, s3, v20 :: v_dual_fmac_f32 v78, s3, v20
	v_dual_fmac_f32 v77, s3, v20 :: v_dual_fmac_f32 v80, s3, v20
	v_dual_fmac_f32 v82, s3, v20 :: v_dual_fmac_f32 v81, s3, v20
	v_dual_fmac_f32 v24, s3, v20 :: v_dual_fmac_f32 v83, s3, v20
	v_dual_fmac_f32 v22, s3, v20 :: v_dual_fmac_f32 v23, s3, v20
	v_fmac_f32_e32 v21, s3, v20
	s_add_co_i32 s4, s4, -1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s4, 0
	s_cbranch_scc1 .LBB1_1
; %bb.2:
	v_add_f32_e32 v1, 0, v19
	s_load_b64 s[0:1], s[0:1], 0x0
	v_lshl_or_b32 v0, ttmp9, 8, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v25, v1
	v_add_f32_e32 v1, v26, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v27, v1
	v_add_f32_e32 v1, v28, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v29, v1
	v_add_f32_e32 v1, v30, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v31, v1
	v_add_f32_e32 v1, v32, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v33, v1
	v_add_f32_e32 v1, v34, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v35, v1
	v_add_f32_e32 v1, v36, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v37, v1
	v_add_f32_e32 v1, v38, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v39, v1
	v_add_f32_e32 v1, v40, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v41, v1
	v_add_f32_e32 v1, v42, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v43, v1
	v_add_f32_e32 v1, v44, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v45, v1
	v_add_f32_e32 v1, v46, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v47, v1
	v_add_f32_e32 v1, v48, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v49, v1
	v_add_f32_e32 v1, v50, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v51, v1
	v_add_f32_e32 v1, v52, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v53, v1
	v_add_f32_e32 v1, v54, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v55, v1
	v_add_f32_e32 v1, v56, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v57, v1
	v_add_f32_e32 v1, v58, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v59, v1
	v_add_f32_e32 v1, v60, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v61, v1
	v_add_f32_e32 v1, v62, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v63, v1
	v_add_f32_e32 v1, v64, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v65, v1
	v_add_f32_e32 v1, v66, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v67, v1
	v_add_f32_e32 v1, v68, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v69, v1
	v_add_f32_e32 v1, v70, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v71, v1
	v_add_f32_e32 v1, v72, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v73, v1
	v_add_f32_e32 v1, v74, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v75, v1
	v_add_f32_e32 v1, v76, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v77, v1
	v_add_f32_e32 v1, v78, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v79, v1
	v_add_f32_e32 v1, v80, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v81, v1
	v_add_f32_e32 v1, v82, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v83, v1
	v_add_f32_e32 v1, v24, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_add_f32 v2, v23, v1 :: v_dual_mov_b32 v1, 0
	v_add_f32_e32 v2, v22, v2
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[0:1], 2, v[0:1]
	v_add_f32_e32 v2, v21, v2
	s_wait_kmcnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v0, vcc_lo, s0, v0
	v_add_co_ci_u32_e64 v1, null, s1, v1, vcc_lo
	global_store_b32 v[0:1], v2, off
	s_nop 0
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
.Lfunc_end1:
	.size	fold, .Lfunc_end1-fold
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel fold
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 12
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
		.amdhsa_next_free_vgpr 148
		.amdhsa_next_free_sgpr 15
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end1-fold)<<4)&4080)>>4
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
	.set .Lfold.num_vgpr, 148
	.set .Lfold.num_agpr, 0
	.set .Lfold.numbered_sgpr, 15
	.set .Lfold.num_named_barrier, 0
	.set .Lfold.private_seg_size, 0
	.set .Lfold.uses_vcc, 1
	.set .Lfold.uses_flat_scratch, 0
	.set .Lfold.has_dyn_sized_stack, 0
	.set .Lfold.has_recursion, 0
	.set .Lfold.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 2012
; TotalNumSgprs: 17
; NumVgprs: 148
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 18
; NumSGPRsForWavesPerEU: 17
; NumVGPRsForWavesPerEU: 148
; Occupancy: 9
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
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
	.type	__hip_cuid_425b31906c69eca4,@object ; @__hip_cuid_425b31906c69eca4
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_425b31906c69eca4
__hip_cuid_425b31906c69eca4:
	.byte	0                               ; 0x0
	.size	__hip_cuid_425b31906c69eca4, 1

	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git 8f497e0992fb7513f7f78a6f6b6f1056c375e961)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_425b31906c69eca4
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .offset:         8
        .size:           4
        .value_kind:     by_value
    .gfx1250_revision: B0
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 12
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           pure
    .private_segment_fixed_size: 0
    .sgpr_count:     17
    .sgpr_spill_count: 0
    .symbol:         pure.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     83
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .offset:         8
        .size:           4
        .value_kind:     by_value
    .gfx1250_revision: B0
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 12
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           fold
    .private_segment_fixed_size: 0
    .sgpr_count:     17
    .sgpr_spill_count: 0
    .symbol:         fold.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     148
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target:   amdgcn-amd-amdhsa--gfx1201
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
