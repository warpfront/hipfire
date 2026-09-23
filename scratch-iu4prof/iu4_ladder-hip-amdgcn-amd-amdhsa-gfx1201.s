	.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
	.amdhsa_code_object_version 6
	.section	.text._Z10iu4_ladderILi0EEvPKDv2_jS2_PKjPfPyiii,"axG",@progbits,_Z10iu4_ladderILi0EEvPKDv2_jS2_PKjPfPyiii,comdat
	.protected	_Z10iu4_ladderILi0EEvPKDv2_jS2_PKjPfPyiii ; -- Begin function _Z10iu4_ladderILi0EEvPKDv2_jS2_PKjPfPyiii
	.globl	_Z10iu4_ladderILi0EEvPKDv2_jS2_PKjPfPyiii
	.p2align	8
	.type	_Z10iu4_ladderILi0EEvPKDv2_jS2_PKjPfPyiii,@function
_Z10iu4_ladderILi0EEvPKDv2_jS2_PKjPfPyiii: ; @_Z10iu4_ladderILi0EEvPKDv2_jS2_PKjPfPyiii
	.cfi_startproc
; %bb.0:                                ; %.preheader34
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_load_b64 s[6:7], s[0:1], 0x28
	s_wait_kmcnt 0x0
	s_cmp_lt_i32 s6, 1
	s_cbranch_scc1 .LBB0_3
; %bb.1:                                ; %.lr.ph
	v_mov_b32_e32 v41, 0
	v_or_b32_e32 v65, 0x11111111, v0
	v_or_b32_e32 v66, 0x22222222, v0
	v_or_b32_e32 v67, 0x33333333, v0
	v_or_b32_e32 v68, 0x44444444, v0
	v_dual_mov_b32 v42, v41 :: v_dual_mov_b32 v43, v41
	v_dual_mov_b32 v44, v41 :: v_dual_mov_b32 v45, v41
	v_dual_mov_b32 v46, v41 :: v_dual_mov_b32 v47, v41
	v_dual_mov_b32 v48, v41 :: v_dual_mov_b32 v57, v41
	v_dual_mov_b32 v58, v41 :: v_dual_mov_b32 v59, v41
	v_dual_mov_b32 v60, v41 :: v_dual_mov_b32 v61, v41
	v_dual_mov_b32 v62, v41 :: v_dual_mov_b32 v63, v41
	v_dual_mov_b32 v64, v41 :: v_dual_mov_b32 v49, v41
	v_dual_mov_b32 v50, v41 :: v_dual_mov_b32 v51, v41
	v_dual_mov_b32 v52, v41 :: v_dual_mov_b32 v53, v41
	v_dual_mov_b32 v54, v41 :: v_dual_mov_b32 v55, v41
	v_dual_mov_b32 v56, v41 :: v_dual_mov_b32 v33, v41
	v_dual_mov_b32 v34, v41 :: v_dual_mov_b32 v35, v41
	v_dual_mov_b32 v36, v41 :: v_dual_mov_b32 v37, v41
	v_dual_mov_b32 v38, v41 :: v_dual_mov_b32 v39, v41
	v_dual_mov_b32 v40, v41 :: v_dual_mov_b32 v25, v41
	v_dual_mov_b32 v26, v41 :: v_dual_mov_b32 v27, v41
	v_dual_mov_b32 v28, v41 :: v_dual_mov_b32 v29, v41
	v_dual_mov_b32 v30, v41 :: v_dual_mov_b32 v31, v41
	v_dual_mov_b32 v32, v41 :: v_dual_mov_b32 v17, v41
	v_dual_mov_b32 v18, v41 :: v_dual_mov_b32 v19, v41
	v_dual_mov_b32 v20, v41 :: v_dual_mov_b32 v21, v41
	v_dual_mov_b32 v22, v41 :: v_dual_mov_b32 v23, v41
	v_dual_mov_b32 v24, v41 :: v_dual_mov_b32 v9, v41
	v_dual_mov_b32 v10, v41 :: v_dual_mov_b32 v11, v41
	v_dual_mov_b32 v12, v41 :: v_dual_mov_b32 v13, v41
	v_dual_mov_b32 v14, v41 :: v_dual_mov_b32 v15, v41
	v_dual_mov_b32 v16, v41 :: v_dual_mov_b32 v1, v41
	v_dual_mov_b32 v2, v41 :: v_dual_mov_b32 v3, v41
	v_dual_mov_b32 v4, v41 :: v_dual_mov_b32 v5, v41
	v_dual_mov_b32 v6, v41 :: v_dual_mov_b32 v7, v41
	v_mov_b32_e32 v8, v41
	s_lshl_b32 s2, s6, 1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_max_i32 s2, s2, 1
.LBB0_2:                                ; =>This Inner Loop Header: Depth=1
	v_wmma_i32_16x16x32_iu4 v[41:48], v[65:66], v[67:68], v[41:48] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[57:64], v[65:66], v[67:68], v[57:64] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[49:56], v[65:66], v[67:68], v[49:56] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[33:40], v[65:66], v[67:68], v[33:40] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[25:32], v[65:66], v[67:68], v[25:32] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:24], v[65:66], v[67:68], v[17:24] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:16], v[65:66], v[67:68], v[9:16] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:8], v[65:66], v[67:68], v[1:8] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[41:48], v[65:66], v[67:68], v[41:48] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[57:64], v[65:66], v[67:68], v[57:64] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[49:56], v[65:66], v[67:68], v[49:56] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[33:40], v[65:66], v[67:68], v[33:40] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[25:32], v[65:66], v[67:68], v[25:32] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:24], v[65:66], v[67:68], v[17:24] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:16], v[65:66], v[67:68], v[9:16] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:8], v[65:66], v[67:68], v[1:8] neg_lo:[1,1,0]
	s_add_co_i32 s2, s2, -1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_eq_u32 s2, 0
	s_cbranch_scc0 .LBB0_2
	s_branch .LBB0_4
.LBB0_3:
	v_mov_b32_e32 v8, 0
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_mov_b32 v7, v8 :: v_dual_mov_b32 v6, v8
	v_dual_mov_b32 v5, v8 :: v_dual_mov_b32 v4, v8
	v_dual_mov_b32 v3, v8 :: v_dual_mov_b32 v2, v8
	v_dual_mov_b32 v1, v8 :: v_dual_mov_b32 v16, v8
	v_dual_mov_b32 v15, v8 :: v_dual_mov_b32 v14, v8
	v_dual_mov_b32 v13, v8 :: v_dual_mov_b32 v12, v8
	v_dual_mov_b32 v11, v8 :: v_dual_mov_b32 v10, v8
	v_dual_mov_b32 v9, v8 :: v_dual_mov_b32 v24, v8
	v_dual_mov_b32 v23, v8 :: v_dual_mov_b32 v22, v8
	v_dual_mov_b32 v21, v8 :: v_dual_mov_b32 v20, v8
	v_dual_mov_b32 v19, v8 :: v_dual_mov_b32 v18, v8
	v_dual_mov_b32 v17, v8 :: v_dual_mov_b32 v32, v8
	v_dual_mov_b32 v31, v8 :: v_dual_mov_b32 v30, v8
	v_dual_mov_b32 v29, v8 :: v_dual_mov_b32 v28, v8
	v_dual_mov_b32 v27, v8 :: v_dual_mov_b32 v26, v8
	v_dual_mov_b32 v25, v8 :: v_dual_mov_b32 v40, v8
	v_dual_mov_b32 v39, v8 :: v_dual_mov_b32 v38, v8
	v_dual_mov_b32 v37, v8 :: v_dual_mov_b32 v36, v8
	v_dual_mov_b32 v35, v8 :: v_dual_mov_b32 v34, v8
	v_dual_mov_b32 v33, v8 :: v_dual_mov_b32 v56, v8
	v_dual_mov_b32 v55, v8 :: v_dual_mov_b32 v54, v8
	v_dual_mov_b32 v53, v8 :: v_dual_mov_b32 v52, v8
	v_dual_mov_b32 v51, v8 :: v_dual_mov_b32 v50, v8
	v_dual_mov_b32 v49, v8 :: v_dual_mov_b32 v64, v8
	v_dual_mov_b32 v63, v8 :: v_dual_mov_b32 v62, v8
	v_dual_mov_b32 v61, v8 :: v_dual_mov_b32 v60, v8
	v_dual_mov_b32 v59, v8 :: v_dual_mov_b32 v58, v8
	v_dual_mov_b32 v57, v8 :: v_dual_mov_b32 v48, v8
	v_dual_mov_b32 v47, v8 :: v_dual_mov_b32 v46, v8
	v_dual_mov_b32 v45, v8 :: v_dual_mov_b32 v44, v8
	v_dual_mov_b32 v43, v8 :: v_dual_mov_b32 v42, v8
	v_mov_b32_e32 v41, v8
.LBB0_4:                                ; %Flow412
	v_and_b32_e32 v65, 7, v0
	s_load_b64 s[8:9], s[0:1], 0x20
	v_lshlrev_b32_e32 v0, 3, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 1, v65
	v_cndmask_b32_e32 v41, v41, v42, vcc_lo
	v_cndmask_b32_e32 v42, v57, v58, vcc_lo
	v_cmp_eq_u32_e64 s0, 2, v65
	v_cmp_eq_u32_e64 s1, 3, v65
	v_cndmask_b32_e32 v9, v9, v10, vcc_lo
	v_cndmask_b32_e32 v33, v33, v34, vcc_lo
	v_cndmask_b32_e32 v25, v25, v26, vcc_lo
	v_cndmask_b32_e64 v10, v42, v59, s0
	v_cndmask_b32_e64 v41, v41, v43, s0
	v_cndmask_b32_e32 v17, v17, v18, vcc_lo
	v_cndmask_b32_e32 v1, v1, v2, vcc_lo
	v_cmp_eq_u32_e64 s2, 5, v65
	v_cndmask_b32_e64 v10, v10, v60, s1
	v_cndmask_b32_e32 v43, v49, v50, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, 4, v65
	v_cndmask_b32_e64 v2, v41, v44, s1
	v_cmp_eq_u32_e64 s3, 6, v65
	v_cmp_eq_u32_e64 s4, 7, v65
	v_cndmask_b32_e64 v25, v25, v27, s0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v10, v10, v61, vcc_lo
	v_cndmask_b32_e64 v18, v43, v51, s0
	v_cndmask_b32_e32 v2, v2, v45, vcc_lo
	v_cndmask_b32_e64 v9, v9, v11, s0
	v_cndmask_b32_e64 v17, v17, v19, s0
	v_cndmask_b32_e64 v10, v10, v62, s2
	v_cndmask_b32_e64 v18, v18, v52, s1
	v_cndmask_b32_e64 v2, v2, v46, s2
	v_cndmask_b32_e64 v19, v25, v28, s1
	v_cndmask_b32_e64 v17, v17, v20, s1
	v_cndmask_b32_e64 v10, v10, v63, s3
	v_cndmask_b32_e32 v18, v18, v53, vcc_lo
	v_cndmask_b32_e64 v26, v33, v35, s0
	v_cndmask_b32_e64 v2, v2, v47, s3
	v_cndmask_b32_e32 v17, v17, v21, vcc_lo
	v_cndmask_b32_e64 v10, v10, v64, s4
	v_cndmask_b32_e64 v18, v18, v54, s2
	v_cndmask_b32_e64 v11, v26, v36, s1
	v_cndmask_b32_e64 v2, v2, v48, s4
	v_cndmask_b32_e64 v9, v9, v12, s1
	v_cndmask_b32_e64 v1, v1, v3, s0
	v_cndmask_b32_e64 v18, v18, v55, s3
	v_cndmask_b32_e32 v11, v11, v37, vcc_lo
	v_add_co_u32 v2, s5, v2, v10
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v10, null, 0, 0, s5
	v_cndmask_b32_e64 v18, v18, v56, s4
	v_cndmask_b32_e64 v11, v11, v38, s2
	v_cndmask_b32_e64 v12, v17, v22, s2
	v_cndmask_b32_e32 v9, v9, v13, vcc_lo
	v_cndmask_b32_e64 v1, v1, v4, s1
	v_add_co_u32 v2, s5, v2, v18
	v_cndmask_b32_e32 v18, v19, v29, vcc_lo
	v_cndmask_b32_e64 v3, v11, v39, s3
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v10, null, 0, v10, s5
	v_cndmask_b32_e64 v9, v9, v14, s2
	v_cndmask_b32_e64 v11, v18, v30, s2
	v_cndmask_b32_e64 v3, v3, v40, s4
	v_cndmask_b32_e32 v1, v1, v5, vcc_lo
	s_mul_i32 s0, s7, ttmp7
	v_cndmask_b32_e64 v9, v9, v15, s3
	v_cndmask_b32_e64 v4, v11, v31, s3
	v_cndmask_b32_e64 v11, v12, v23, s3
	v_add_co_u32 v2, vcc_lo, v2, v3
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, 0, v10, vcc_lo
	v_cndmask_b32_e64 v4, v4, v32, s4
	v_cndmask_b32_e64 v5, v11, v24, s4
	v_cndmask_b32_e64 v1, v1, v6, s2
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s0, s0, ttmp9
	v_add_co_u32 v2, vcc_lo, v2, v4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, 0, v3, vcc_lo
	v_cndmask_b32_e64 v4, v9, v16, s4
	v_cndmask_b32_e64 v1, v1, v7, s3
	v_add_co_u32 v2, vcc_lo, v2, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, 0, v3, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v1, v1, v8, s4
	v_add_co_u32 v2, vcc_lo, v2, v4
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_ci_u32_e64 v3, null, 0, v3, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_ashr_i32 s1, s0, 31
	v_add_co_u32 v1, vcc_lo, v2, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, 0, v3, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b64 s[0:1], s[0:1], 11
	s_wait_kmcnt 0x0
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[0:1], s[8:9], s[0:1]
	global_store_b64 v0, v[1:2], s[0:1]
	s_endpgm
.Lfunc_end0:
	.size	_Z10iu4_ladderILi0EEvPKDv2_jS2_PKjPfPyiii, .Lfunc_end0-_Z10iu4_ladderILi0EEvPKDv2_jS2_PKjPfPyiii
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z10iu4_ladderILi0EEvPKDv2_jS2_PKjPfPyiii
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 52
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
		.amdhsa_next_free_vgpr 69
		.amdhsa_next_free_sgpr 10
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end0-_Z10iu4_ladderILi0EEvPKDv2_jS2_PKjPfPyiii)<<4)&4080)>>4
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._Z10iu4_ladderILi0EEvPKDv2_jS2_PKjPfPyiii,"axG",@progbits,_Z10iu4_ladderILi0EEvPKDv2_jS2_PKjPfPyiii,comdat
                                        ; -- End function
	.set .L_Z10iu4_ladderILi0EEvPKDv2_jS2_PKjPfPyiii.num_vgpr, 69
	.set .L_Z10iu4_ladderILi0EEvPKDv2_jS2_PKjPfPyiii.num_agpr, 0
	.set .L_Z10iu4_ladderILi0EEvPKDv2_jS2_PKjPfPyiii.numbered_sgpr, 10
	.set .L_Z10iu4_ladderILi0EEvPKDv2_jS2_PKjPfPyiii.num_named_barrier, 0
	.set .L_Z10iu4_ladderILi0EEvPKDv2_jS2_PKjPfPyiii.private_seg_size, 0
	.set .L_Z10iu4_ladderILi0EEvPKDv2_jS2_PKjPfPyiii.uses_vcc, 1
	.set .L_Z10iu4_ladderILi0EEvPKDv2_jS2_PKjPfPyiii.uses_flat_scratch, 0
	.set .L_Z10iu4_ladderILi0EEvPKDv2_jS2_PKjPfPyiii.has_dyn_sized_stack, 0
	.set .L_Z10iu4_ladderILi0EEvPKDv2_jS2_PKjPfPyiii.has_recursion, 0
	.set .L_Z10iu4_ladderILi0EEvPKDv2_jS2_PKjPfPyiii.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 1388
; TotalNumSgprs: 12
; NumVgprs: 69
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 8
; NumSGPRsForWavesPerEU: 12
; NumVGPRsForWavesPerEU: 69
; Occupancy: 16
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.section	.text._Z10iu4_ladderILi1EEvPKDv2_jS2_PKjPfPyiii,"axG",@progbits,_Z10iu4_ladderILi1EEvPKDv2_jS2_PKjPfPyiii,comdat
	.protected	_Z10iu4_ladderILi1EEvPKDv2_jS2_PKjPfPyiii ; -- Begin function _Z10iu4_ladderILi1EEvPKDv2_jS2_PKjPfPyiii
	.globl	_Z10iu4_ladderILi1EEvPKDv2_jS2_PKjPfPyiii
	.p2align	8
	.type	_Z10iu4_ladderILi1EEvPKDv2_jS2_PKjPfPyiii,@function
_Z10iu4_ladderILi1EEvPKDv2_jS2_PKjPfPyiii: ; @_Z10iu4_ladderILi1EEvPKDv2_jS2_PKjPfPyiii
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	v_or_b32_e32 v2, 0x11111111, v0
	v_lshlrev_b32_e32 v1, 4, v0
	v_or_b32_e32 v3, 0x22222222, v0
	s_load_b64 s[6:7], s[0:1], 0x28
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_mov_b32 v4, v2 :: v_dual_lshlrev_b32 v75, 3, v0
	v_dual_mov_b32 v5, v3 :: v_dual_add_nc_u32 v6, 0, v1
	ds_store_b128 v6, v[2:5]
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_wait_kmcnt 0x0
	s_cmp_lt_i32 s6, 1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_cbranch_scc1 .LBB1_3
; %bb.1:                                ; %.lr.ph
	v_dual_mov_b32 v41, 0 :: v_dual_and_b32 v2, 0xf8, v75
	v_and_b32_e32 v3, 0xe00, v1
	v_and_b32_e32 v1, 0xc00, v1
	v_or_b32_e32 v69, 0x33333333, v0
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_dual_mov_b32 v43, v41 :: v_dual_add_nc_u32 v2, 0, v2
	v_mov_b32_e32 v42, v41
	v_or_b32_e32 v4, 0x200, v3
	v_or_b32_e32 v3, 0x300, v3
	v_dual_mov_b32 v44, v41 :: v_dual_add_nc_u32 v1, v2, v1
	v_mov_b32_e32 v46, v41
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_mov_b32 v45, v41 :: v_dual_add_nc_u32 v4, v2, v4
	v_dual_mov_b32 v47, v41 :: v_dual_add_nc_u32 v2, v2, v3
	ds_load_2addr_b64 v[65:68], v1 offset1:32
	ds_load_b64 v[71:72], v4
	ds_load_b64 v[73:74], v2
	v_or_b32_e32 v70, 0x44444444, v0
	v_dual_mov_b32 v48, v41 :: v_dual_mov_b32 v57, v41
	v_dual_mov_b32 v58, v41 :: v_dual_mov_b32 v59, v41
	v_dual_mov_b32 v60, v41 :: v_dual_mov_b32 v61, v41
	v_dual_mov_b32 v62, v41 :: v_dual_mov_b32 v63, v41
	v_dual_mov_b32 v64, v41 :: v_dual_mov_b32 v49, v41
	v_dual_mov_b32 v50, v41 :: v_dual_mov_b32 v51, v41
	v_dual_mov_b32 v52, v41 :: v_dual_mov_b32 v53, v41
	v_dual_mov_b32 v54, v41 :: v_dual_mov_b32 v55, v41
	v_dual_mov_b32 v56, v41 :: v_dual_mov_b32 v33, v41
	v_dual_mov_b32 v34, v41 :: v_dual_mov_b32 v35, v41
	v_dual_mov_b32 v36, v41 :: v_dual_mov_b32 v37, v41
	v_dual_mov_b32 v38, v41 :: v_dual_mov_b32 v39, v41
	v_dual_mov_b32 v40, v41 :: v_dual_mov_b32 v25, v41
	v_dual_mov_b32 v26, v41 :: v_dual_mov_b32 v27, v41
	v_dual_mov_b32 v28, v41 :: v_dual_mov_b32 v29, v41
	v_dual_mov_b32 v30, v41 :: v_dual_mov_b32 v31, v41
	v_dual_mov_b32 v32, v41 :: v_dual_mov_b32 v17, v41
	v_dual_mov_b32 v18, v41 :: v_dual_mov_b32 v19, v41
	v_dual_mov_b32 v20, v41 :: v_dual_mov_b32 v21, v41
	v_dual_mov_b32 v22, v41 :: v_dual_mov_b32 v23, v41
	v_dual_mov_b32 v24, v41 :: v_dual_mov_b32 v9, v41
	v_dual_mov_b32 v10, v41 :: v_dual_mov_b32 v11, v41
	v_dual_mov_b32 v12, v41 :: v_dual_mov_b32 v13, v41
	v_dual_mov_b32 v14, v41 :: v_dual_mov_b32 v15, v41
	v_dual_mov_b32 v16, v41 :: v_dual_mov_b32 v1, v41
	v_dual_mov_b32 v2, v41 :: v_dual_mov_b32 v3, v41
	v_dual_mov_b32 v4, v41 :: v_dual_mov_b32 v5, v41
	v_dual_mov_b32 v6, v41 :: v_dual_mov_b32 v7, v41
	v_mov_b32_e32 v8, v41
	s_lshl_b32 s2, s6, 1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_max_i32 s2, s2, 1
	s_wait_dscnt 0x0
.LBB1_2:                                ; =>This Inner Loop Header: Depth=1
	v_wmma_i32_16x16x32_iu4 v[41:48], v[65:66], v[69:70], v[41:48] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[57:64], v[65:66], v[69:70], v[57:64] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[49:56], v[65:66], v[69:70], v[49:56] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[33:40], v[65:66], v[69:70], v[33:40] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[25:32], v[71:72], v[69:70], v[25:32] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:24], v[71:72], v[69:70], v[17:24] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:16], v[71:72], v[69:70], v[9:16] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:8], v[71:72], v[69:70], v[1:8] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[41:48], v[67:68], v[69:70], v[41:48] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[57:64], v[67:68], v[69:70], v[57:64] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[49:56], v[67:68], v[69:70], v[49:56] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[33:40], v[67:68], v[69:70], v[33:40] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[25:32], v[73:74], v[69:70], v[25:32] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:24], v[73:74], v[69:70], v[17:24] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:16], v[73:74], v[69:70], v[9:16] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:8], v[73:74], v[69:70], v[1:8] neg_lo:[1,1,0]
	s_add_co_i32 s2, s2, -1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_eq_u32 s2, 0
	s_cbranch_scc0 .LBB1_2
	s_branch .LBB1_4
.LBB1_3:
	v_mov_b32_e32 v8, 0
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_mov_b32 v7, v8 :: v_dual_mov_b32 v6, v8
	v_dual_mov_b32 v5, v8 :: v_dual_mov_b32 v4, v8
	v_dual_mov_b32 v3, v8 :: v_dual_mov_b32 v2, v8
	v_dual_mov_b32 v1, v8 :: v_dual_mov_b32 v16, v8
	v_dual_mov_b32 v15, v8 :: v_dual_mov_b32 v14, v8
	v_dual_mov_b32 v13, v8 :: v_dual_mov_b32 v12, v8
	v_dual_mov_b32 v11, v8 :: v_dual_mov_b32 v10, v8
	v_dual_mov_b32 v9, v8 :: v_dual_mov_b32 v24, v8
	v_dual_mov_b32 v23, v8 :: v_dual_mov_b32 v22, v8
	v_dual_mov_b32 v21, v8 :: v_dual_mov_b32 v20, v8
	v_dual_mov_b32 v19, v8 :: v_dual_mov_b32 v18, v8
	v_dual_mov_b32 v17, v8 :: v_dual_mov_b32 v32, v8
	v_dual_mov_b32 v31, v8 :: v_dual_mov_b32 v30, v8
	v_dual_mov_b32 v29, v8 :: v_dual_mov_b32 v28, v8
	v_dual_mov_b32 v27, v8 :: v_dual_mov_b32 v26, v8
	v_dual_mov_b32 v25, v8 :: v_dual_mov_b32 v40, v8
	v_dual_mov_b32 v39, v8 :: v_dual_mov_b32 v38, v8
	v_dual_mov_b32 v37, v8 :: v_dual_mov_b32 v36, v8
	v_dual_mov_b32 v35, v8 :: v_dual_mov_b32 v34, v8
	v_dual_mov_b32 v33, v8 :: v_dual_mov_b32 v56, v8
	v_dual_mov_b32 v55, v8 :: v_dual_mov_b32 v54, v8
	v_dual_mov_b32 v53, v8 :: v_dual_mov_b32 v52, v8
	v_dual_mov_b32 v51, v8 :: v_dual_mov_b32 v50, v8
	v_dual_mov_b32 v49, v8 :: v_dual_mov_b32 v64, v8
	v_dual_mov_b32 v63, v8 :: v_dual_mov_b32 v62, v8
	v_dual_mov_b32 v61, v8 :: v_dual_mov_b32 v60, v8
	v_dual_mov_b32 v59, v8 :: v_dual_mov_b32 v58, v8
	v_dual_mov_b32 v57, v8 :: v_dual_mov_b32 v48, v8
	v_dual_mov_b32 v47, v8 :: v_dual_mov_b32 v46, v8
	v_dual_mov_b32 v45, v8 :: v_dual_mov_b32 v44, v8
	v_dual_mov_b32 v43, v8 :: v_dual_mov_b32 v42, v8
	v_mov_b32_e32 v41, v8
.LBB1_4:                                ; %Flow419
	v_and_b32_e32 v0, 7, v0
	s_load_b64 s[8:9], s[0:1], 0x20
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cmp_eq_u32_e32 vcc_lo, 1, v0
	v_cndmask_b32_e32 v41, v41, v42, vcc_lo
	v_cmp_eq_u32_e64 s0, 2, v0
	v_cndmask_b32_e32 v42, v57, v58, vcc_lo
	v_cmp_eq_u32_e64 s1, 3, v0
	v_cndmask_b32_e32 v9, v9, v10, vcc_lo
	v_cndmask_b32_e32 v25, v25, v26, vcc_lo
	v_cndmask_b32_e64 v41, v41, v43, s0
	v_cndmask_b32_e64 v10, v42, v59, s0
	v_cndmask_b32_e32 v17, v17, v18, vcc_lo
	v_cndmask_b32_e32 v1, v1, v2, vcc_lo
	v_cmp_eq_u32_e64 s2, 5, v0
	v_cndmask_b32_e64 v2, v41, v44, s1
	v_cndmask_b32_e32 v43, v49, v50, vcc_lo
	v_cndmask_b32_e64 v10, v10, v60, s1
	v_cndmask_b32_e32 v33, v33, v34, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, 4, v0
	v_cmp_eq_u32_e64 s3, 6, v0
	v_cmp_eq_u32_e64 s4, 7, v0
	v_cndmask_b32_e64 v1, v1, v3, s0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v2, v2, v45, vcc_lo
	v_cndmask_b32_e64 v18, v43, v51, s0
	v_cndmask_b32_e32 v10, v10, v61, vcc_lo
	v_cndmask_b32_e64 v26, v33, v35, s0
	v_cndmask_b32_e64 v1, v1, v4, s1
	v_cndmask_b32_e64 v2, v2, v46, s2
	v_cndmask_b32_e64 v18, v18, v52, s1
	v_cndmask_b32_e64 v10, v10, v62, s2
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cndmask_b32_e32 v1, v1, v5, vcc_lo
	v_cndmask_b32_e64 v2, v2, v47, s3
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cndmask_b32_e32 v18, v18, v53, vcc_lo
	v_cndmask_b32_e64 v0, v10, v63, s3
	v_cndmask_b32_e64 v25, v25, v27, s0
	v_cndmask_b32_e64 v10, v26, v36, s1
	v_cndmask_b32_e64 v2, v2, v48, s4
	v_cndmask_b32_e64 v18, v18, v54, s2
	v_cndmask_b32_e64 v0, v0, v64, s4
	v_cndmask_b32_e64 v1, v1, v6, s2
	v_cndmask_b32_e32 v10, v10, v37, vcc_lo
	v_cndmask_b32_e64 v17, v17, v19, s0
	v_cndmask_b32_e64 v18, v18, v55, s3
	v_cndmask_b32_e64 v19, v25, v28, s1
	v_add_co_u32 v0, s5, v2, v0
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cndmask_b32_e64 v17, v17, v20, s1
	v_cndmask_b32_e64 v18, v18, v56, s4
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, 0, 0, s5
	v_cndmask_b32_e64 v10, v10, v38, s2
	v_cndmask_b32_e32 v17, v17, v21, vcc_lo
	v_add_co_u32 v0, s5, v0, v18
	v_cndmask_b32_e32 v18, v19, v29, vcc_lo
	v_cndmask_b32_e64 v9, v9, v11, s0
	v_cndmask_b32_e64 v3, v10, v39, s3
	v_cndmask_b32_e64 v11, v17, v22, s2
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, 0, v2, s5
	v_cndmask_b32_e64 v10, v18, v30, s2
	v_cndmask_b32_e64 v9, v9, v12, s1
	v_cndmask_b32_e64 v3, v3, v40, s4
	v_cndmask_b32_e64 v1, v1, v7, s3
	s_mul_i32 s0, s7, ttmp7
	v_cndmask_b32_e64 v4, v10, v31, s3
	v_cndmask_b32_e32 v9, v9, v13, vcc_lo
	v_cndmask_b32_e64 v10, v11, v23, s3
	v_add_co_u32 v0, vcc_lo, v0, v3
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cndmask_b32_e64 v3, v4, v32, s4
	v_cndmask_b32_e64 v5, v9, v14, s2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, 0, v2, vcc_lo
	v_cndmask_b32_e64 v4, v10, v24, s4
	v_add_co_u32 v0, vcc_lo, v0, v3
	v_cndmask_b32_e64 v3, v5, v15, s3
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, 0, v2, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_co_u32 v0, vcc_lo, v0, v4
	v_cndmask_b32_e64 v3, v3, v16, s4
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_ci_u32_e64 v2, null, 0, v2, vcc_lo
	v_cndmask_b32_e64 v1, v1, v8, s4
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s0, s0, ttmp9
	v_add_co_u32 v0, vcc_lo, v0, v3
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, 0, v2, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_ashr_i32 s1, s0, 31
	v_add_co_u32 v0, vcc_lo, v0, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, 0, v2, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b64 s[0:1], s[0:1], 11
	s_wait_kmcnt 0x0
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[0:1], s[8:9], s[0:1]
	global_store_b64 v75, v[0:1], s[0:1]
	s_endpgm
.Lfunc_end1:
	.size	_Z10iu4_ladderILi1EEvPKDv2_jS2_PKjPfPyiii, .Lfunc_end1-_Z10iu4_ladderILi1EEvPKDv2_jS2_PKjPfPyiii
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z10iu4_ladderILi1EEvPKDv2_jS2_PKjPfPyiii
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 52
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
		.amdhsa_next_free_vgpr 76
		.amdhsa_next_free_sgpr 10
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end1-_Z10iu4_ladderILi1EEvPKDv2_jS2_PKjPfPyiii)<<4)&4080)>>4
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._Z10iu4_ladderILi1EEvPKDv2_jS2_PKjPfPyiii,"axG",@progbits,_Z10iu4_ladderILi1EEvPKDv2_jS2_PKjPfPyiii,comdat
                                        ; -- End function
	.set .L_Z10iu4_ladderILi1EEvPKDv2_jS2_PKjPfPyiii.num_vgpr, 76
	.set .L_Z10iu4_ladderILi1EEvPKDv2_jS2_PKjPfPyiii.num_agpr, 0
	.set .L_Z10iu4_ladderILi1EEvPKDv2_jS2_PKjPfPyiii.numbered_sgpr, 10
	.set .L_Z10iu4_ladderILi1EEvPKDv2_jS2_PKjPfPyiii.num_named_barrier, 0
	.set .L_Z10iu4_ladderILi1EEvPKDv2_jS2_PKjPfPyiii.private_seg_size, 0
	.set .L_Z10iu4_ladderILi1EEvPKDv2_jS2_PKjPfPyiii.uses_vcc, 1
	.set .L_Z10iu4_ladderILi1EEvPKDv2_jS2_PKjPfPyiii.uses_flat_scratch, 0
	.set .L_Z10iu4_ladderILi1EEvPKDv2_jS2_PKjPfPyiii.has_dyn_sized_stack, 0
	.set .L_Z10iu4_ladderILi1EEvPKDv2_jS2_PKjPfPyiii.has_recursion, 0
	.set .L_Z10iu4_ladderILi1EEvPKDv2_jS2_PKjPfPyiii.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 1548
; TotalNumSgprs: 12
; NumVgprs: 76
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 9
; NumSGPRsForWavesPerEU: 12
; NumVGPRsForWavesPerEU: 76
; Occupancy: 16
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.section	.text._Z10iu4_ladderILi2EEvPKDv2_jS2_PKjPfPyiii,"axG",@progbits,_Z10iu4_ladderILi2EEvPKDv2_jS2_PKjPfPyiii,comdat
	.protected	_Z10iu4_ladderILi2EEvPKDv2_jS2_PKjPfPyiii ; -- Begin function _Z10iu4_ladderILi2EEvPKDv2_jS2_PKjPfPyiii
	.globl	_Z10iu4_ladderILi2EEvPKDv2_jS2_PKjPfPyiii
	.p2align	8
	.type	_Z10iu4_ladderILi2EEvPKDv2_jS2_PKjPfPyiii,@function
_Z10iu4_ladderILi2EEvPKDv2_jS2_PKjPfPyiii: ; @_Z10iu4_ladderILi2EEvPKDv2_jS2_PKjPfPyiii
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_clause 0x1
	s_load_b64 s[6:7], s[0:1], 0x28
	s_load_b64 s[8:9], s[0:1], 0x0
	s_mov_b32 s4, ttmp9
	s_ashr_i32 s5, ttmp9, 31
	v_lshlrev_b32_e32 v66, 4, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	v_add_nc_u32_e32 v74, 0, v66
	s_wait_kmcnt 0x0
	s_lshl_b32 s2, s6, 1
	s_ashr_i32 s3, s2, 31
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mul_u64 s[4:5], s[2:3], s[4:5]
	s_mov_b32 s3, 0
	s_lshl_b64 s[4:5], s[4:5], 12
	s_cmp_gt_i32 s6, 0
	s_add_nc_u64 s[4:5], s[8:9], s[4:5]
	global_load_b128 v[1:4], v66, s[4:5]
	s_wait_loadcnt 0x0
	ds_store_b128 v74, v[1:4]
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_cbranch_scc1 .LBB2_2
; %bb.1:                                ; %..preheader_crit_edge
	v_mov_b32_e32 v1, 0
	s_branch .LBB2_3
.LBB2_2:
	s_mov_b32 s3, -1
.LBB2_3:                                ; %Flow468
	v_mov_b32_e32 v9, 0
	s_and_not1_b32 vcc_lo, exec_lo, s3
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_mov_b32 v8, v9 :: v_dual_mov_b32 v7, v9
	v_dual_mov_b32 v6, v9 :: v_dual_mov_b32 v5, v9
	v_dual_mov_b32 v4, v9 :: v_dual_mov_b32 v3, v9
	v_dual_mov_b32 v2, v9 :: v_dual_mov_b32 v17, v9
	v_dual_mov_b32 v16, v9 :: v_dual_mov_b32 v15, v9
	v_dual_mov_b32 v14, v9 :: v_dual_mov_b32 v13, v9
	v_dual_mov_b32 v12, v9 :: v_dual_mov_b32 v11, v9
	v_dual_mov_b32 v10, v9 :: v_dual_mov_b32 v25, v9
	v_dual_mov_b32 v24, v9 :: v_dual_mov_b32 v23, v9
	v_dual_mov_b32 v22, v9 :: v_dual_mov_b32 v21, v9
	v_dual_mov_b32 v20, v9 :: v_dual_mov_b32 v19, v9
	v_dual_mov_b32 v18, v9 :: v_dual_mov_b32 v33, v9
	v_dual_mov_b32 v32, v9 :: v_dual_mov_b32 v31, v9
	v_dual_mov_b32 v30, v9 :: v_dual_mov_b32 v29, v9
	v_dual_mov_b32 v28, v9 :: v_dual_mov_b32 v27, v9
	v_dual_mov_b32 v26, v9 :: v_dual_mov_b32 v41, v9
	v_dual_mov_b32 v40, v9 :: v_dual_mov_b32 v39, v9
	v_dual_mov_b32 v38, v9 :: v_dual_mov_b32 v37, v9
	v_dual_mov_b32 v36, v9 :: v_dual_mov_b32 v35, v9
	v_dual_mov_b32 v34, v9 :: v_dual_mov_b32 v49, v9
	v_dual_mov_b32 v48, v9 :: v_dual_mov_b32 v47, v9
	v_dual_mov_b32 v46, v9 :: v_dual_mov_b32 v45, v9
	v_dual_mov_b32 v44, v9 :: v_dual_mov_b32 v43, v9
	v_dual_mov_b32 v42, v9 :: v_dual_mov_b32 v57, v9
	v_dual_mov_b32 v56, v9 :: v_dual_mov_b32 v55, v9
	v_dual_mov_b32 v54, v9 :: v_dual_mov_b32 v53, v9
	v_dual_mov_b32 v52, v9 :: v_dual_mov_b32 v51, v9
	v_dual_mov_b32 v50, v9 :: v_dual_mov_b32 v65, v9
	v_dual_mov_b32 v64, v9 :: v_dual_mov_b32 v63, v9
	v_dual_mov_b32 v62, v9 :: v_dual_mov_b32 v61, v9
	v_dual_mov_b32 v60, v9 :: v_dual_mov_b32 v59, v9
	v_mov_b32_e32 v58, v9
	s_cbranch_vccnz .LBB2_14
; %bb.4:                                ; %.lr.ph
	v_lshlrev_b32_e32 v1, 3, v0
	v_and_b32_e32 v2, 0xe00, v66
	v_add_co_u32 v6, s3, s4, v66
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_add_co_ci_u32_e64 v7, null, s5, 0, s3
	v_and_b32_e32 v1, 0xf8, v1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_add_co_u32 v72, vcc_lo, 0x2008, v6
	v_or_b32_e32 v70, 0x33333333, v0
	v_or_b32_e32 v71, 0x44444444, v0
	v_add_nc_u32_e32 v5, 0, v1
	v_mov_b32_e32 v1, 0
	v_and_b32_e32 v3, 0xc00, v66
	v_or_b32_e32 v4, 0x200, v2
	v_or_b32_e32 v2, 0x300, v2
	v_add_co_ci_u32_e64 v73, null, 0, v7, vcc_lo
	v_dual_mov_b32 v58, v1 :: v_dual_mov_b32 v59, v1
	v_mov_b32_e32 v62, v1
	v_mov_b32_e32 v60, v1
	v_add_nc_u32_e32 v75, v5, v3
	v_add_nc_u32_e32 v76, v5, v4
	v_add_nc_u32_e32 v77, v5, v2
	v_dual_mov_b32 v61, v1 :: v_dual_mov_b32 v64, v1
	v_dual_mov_b32 v63, v1 :: v_dual_mov_b32 v50, v1
	v_dual_mov_b32 v65, v1 :: v_dual_mov_b32 v52, v1
	v_dual_mov_b32 v51, v1 :: v_dual_mov_b32 v54, v1
	v_dual_mov_b32 v53, v1 :: v_dual_mov_b32 v56, v1
	v_dual_mov_b32 v55, v1 :: v_dual_mov_b32 v42, v1
	v_dual_mov_b32 v57, v1 :: v_dual_mov_b32 v44, v1
	v_dual_mov_b32 v43, v1 :: v_dual_mov_b32 v46, v1
	v_dual_mov_b32 v45, v1 :: v_dual_mov_b32 v48, v1
	v_dual_mov_b32 v47, v1 :: v_dual_mov_b32 v34, v1
	v_dual_mov_b32 v49, v1 :: v_dual_mov_b32 v36, v1
	v_dual_mov_b32 v35, v1 :: v_dual_mov_b32 v38, v1
	v_dual_mov_b32 v37, v1 :: v_dual_mov_b32 v40, v1
	v_dual_mov_b32 v39, v1 :: v_dual_mov_b32 v26, v1
	v_dual_mov_b32 v41, v1 :: v_dual_mov_b32 v28, v1
	v_dual_mov_b32 v27, v1 :: v_dual_mov_b32 v30, v1
	v_dual_mov_b32 v29, v1 :: v_dual_mov_b32 v32, v1
	v_dual_mov_b32 v31, v1 :: v_dual_mov_b32 v18, v1
	v_dual_mov_b32 v33, v1 :: v_dual_mov_b32 v20, v1
	v_dual_mov_b32 v19, v1 :: v_dual_mov_b32 v22, v1
	v_dual_mov_b32 v21, v1 :: v_dual_mov_b32 v24, v1
	v_dual_mov_b32 v23, v1 :: v_dual_mov_b32 v10, v1
	v_dual_mov_b32 v25, v1 :: v_dual_mov_b32 v12, v1
	v_dual_mov_b32 v11, v1 :: v_dual_mov_b32 v14, v1
	v_dual_mov_b32 v13, v1 :: v_dual_mov_b32 v16, v1
	v_dual_mov_b32 v15, v1 :: v_dual_mov_b32 v2, v1
	v_dual_mov_b32 v17, v1 :: v_dual_mov_b32 v4, v1
	v_dual_mov_b32 v3, v1 :: v_dual_mov_b32 v6, v1
	v_dual_mov_b32 v5, v1 :: v_dual_mov_b32 v8, v1
	v_mov_b32_e32 v7, v1
	v_mov_b32_e32 v9, v1
	s_mov_b32 s3, 1
	s_sub_co_i32 s4, 0, s2
	s_branch .LBB2_6
.LBB2_5:                                ;   in Loop: Header=BB2_6 Depth=1
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_add_co_u32 v72, vcc_lo, 0x2000, v72
	s_add_co_i32 s3, s3, 2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v73, null, 0, v73, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s5, s4, s3
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s5, 1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_cbranch_scc1 .LBB2_14
.LBB2_6:                                ; =>This Inner Loop Header: Depth=1
	v_dual_mov_b32 v66, 0 :: v_dual_mov_b32 v67, 0
	v_dual_mov_b32 v68, 0 :: v_dual_mov_b32 v69, 0
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s3, s2
	s_cselect_b32 s5, -1, 0
	s_cmp_ge_i32 s3, s2
	s_cbranch_scc1 .LBB2_8
; %bb.7:                                ;   in Loop: Header=BB2_6 Depth=1
	global_load_b128 v[66:69], v[72:73], off offset:-4104
.LBB2_8:                                ;   in Loop: Header=BB2_6 Depth=1
	ds_load_2addr_b64 v[78:81], v75 offset1:32
	ds_load_b64 v[82:83], v76
	ds_load_b64 v[84:85], v77
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s5
	s_wait_dscnt 0x2
	v_wmma_i32_16x16x32_iu4 v[58:65], v[78:79], v[70:71], v[58:65] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[50:57], v[78:79], v[70:71], v[50:57] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[42:49], v[78:79], v[70:71], v[42:49] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[34:41], v[78:79], v[70:71], v[34:41] neg_lo:[1,1,0]
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x32_iu4 v[26:33], v[82:83], v[70:71], v[26:33] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[18:25], v[82:83], v[70:71], v[18:25] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[10:17], v[82:83], v[70:71], v[10:17] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[2:9], v[82:83], v[70:71], v[2:9] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[58:65], v[80:81], v[70:71], v[58:65] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[50:57], v[80:81], v[70:71], v[50:57] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[42:49], v[80:81], v[70:71], v[42:49] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[34:41], v[80:81], v[70:71], v[34:41] neg_lo:[1,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[26:33], v[84:85], v[70:71], v[26:33] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[18:25], v[84:85], v[70:71], v[18:25] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[10:17], v[84:85], v[70:71], v[10:17] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[2:9], v[84:85], v[70:71], v[2:9] neg_lo:[1,1,0]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_10
; %bb.9:                                ;   in Loop: Header=BB2_6 Depth=1
	s_wait_loadcnt 0x0
	ds_store_b128 v74, v[66:69] offset:4096
.LBB2_10:                               ;   in Loop: Header=BB2_6 Depth=1
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_add_co_i32 s6, s3, 1
	v_dual_mov_b32 v66, 0 :: v_dual_mov_b32 v67, 0
	v_dual_mov_b32 v68, 0 :: v_dual_mov_b32 v69, 0
	s_cmp_lt_i32 s6, s2
	s_cselect_b32 s5, -1, 0
	s_cmp_ge_i32 s6, s2
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_cbranch_scc1 .LBB2_12
; %bb.11:                               ;   in Loop: Header=BB2_6 Depth=1
	global_load_b128 v[66:69], v[72:73], off offset:-8
.LBB2_12:                               ;   in Loop: Header=BB2_6 Depth=1
	v_add_nc_u32_e32 v78, 0x1000, v75
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s5
	ds_load_b64 v[82:83], v76 offset:4096
	ds_load_2addr_b64 v[78:81], v78 offset1:32
	ds_load_b64 v[84:85], v77 offset:4096
	s_wait_dscnt 0x2
	v_wmma_i32_16x16x32_iu4 v[26:33], v[82:83], v[70:71], v[26:33] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[18:25], v[82:83], v[70:71], v[18:25] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[10:17], v[82:83], v[70:71], v[10:17] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[2:9], v[82:83], v[70:71], v[2:9] neg_lo:[1,1,0]
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x32_iu4 v[58:65], v[78:79], v[70:71], v[58:65] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[50:57], v[78:79], v[70:71], v[50:57] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[42:49], v[78:79], v[70:71], v[42:49] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[34:41], v[78:79], v[70:71], v[34:41] neg_lo:[1,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[26:33], v[84:85], v[70:71], v[26:33] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[18:25], v[84:85], v[70:71], v[18:25] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[10:17], v[84:85], v[70:71], v[10:17] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[58:65], v[80:81], v[70:71], v[58:65] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[50:57], v[80:81], v[70:71], v[50:57] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[42:49], v[80:81], v[70:71], v[42:49] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[34:41], v[80:81], v[70:71], v[34:41] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[2:9], v[84:85], v[70:71], v[2:9] neg_lo:[1,1,0]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB2_5
; %bb.13:                               ;   in Loop: Header=BB2_6 Depth=1
	s_wait_loadcnt 0x0
	ds_store_b128 v74, v[66:69]
	s_branch .LBB2_5
.LBB2_14:                               ; %Flow469
	v_and_b32_e32 v66, 7, v0
	s_load_b64 s[8:9], s[0:1], 0x20
	v_lshlrev_b64_e32 v[0:1], 3, v[0:1]
	s_delay_alu instid0(VALU_DEP_2)
	v_cmp_eq_u32_e32 vcc_lo, 1, v66
	v_cmp_eq_u32_e64 s0, 2, v66
	v_cmp_eq_u32_e64 s1, 3, v66
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v42, v42, v43, vcc_lo
	v_cndmask_b32_e32 v18, v18, v19, vcc_lo
	v_cndmask_b32_e32 v10, v10, v11, vcc_lo
	v_cndmask_b32_e32 v34, v34, v35, vcc_lo
	v_cndmask_b32_e32 v26, v26, v27, vcc_lo
	v_cndmask_b32_e64 v19, v42, v44, s0
	v_cndmask_b32_e32 v2, v2, v3, vcc_lo
	v_cndmask_b32_e64 v18, v18, v20, s0
	v_cndmask_b32_e64 v27, v34, v36, s0
	v_cndmask_b32_e64 v26, v26, v28, s0
	v_cndmask_b32_e64 v19, v19, v45, s1
	v_cndmask_b32_e32 v50, v50, v51, vcc_lo
	v_cndmask_b32_e64 v18, v18, v21, s1
	v_cndmask_b32_e64 v20, v27, v37, s1
	v_cndmask_b32_e64 v26, v26, v29, s1
	v_cndmask_b32_e64 v10, v10, v12, s0
	v_cndmask_b32_e64 v11, v50, v52, s0
	v_cndmask_b32_e64 v2, v2, v4, s0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v10, v10, v13, s1
	v_cndmask_b32_e64 v11, v11, v53, s1
	v_cndmask_b32_e32 v58, v58, v59, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, 4, v66
	v_cndmask_b32_e64 v2, v2, v5, s1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v11, v11, v54, vcc_lo
	v_cndmask_b32_e64 v58, v58, v60, s0
	v_cndmask_b32_e32 v19, v19, v46, vcc_lo
	v_cndmask_b32_e32 v20, v20, v38, vcc_lo
	v_cndmask_b32_e32 v18, v18, v22, vcc_lo
	v_cndmask_b32_e32 v10, v10, v14, vcc_lo
	v_cndmask_b32_e64 v3, v58, v61, s1
	v_cndmask_b32_e32 v2, v2, v6, vcc_lo
	s_mul_i32 s0, s7, ttmp7
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s0, s0, ttmp9
	v_cndmask_b32_e32 v3, v3, v62, vcc_lo
	v_cmp_eq_u32_e64 s2, 5, v66
	v_cmp_eq_u32_e64 s3, 6, v66
	v_cmp_eq_u32_e64 s4, 7, v66
	s_wait_alu depctr_sa_sdst(0)
	s_ashr_i32 s1, s0, 31
	v_cndmask_b32_e64 v3, v3, v63, s2
	v_cndmask_b32_e64 v11, v11, v55, s2
	v_cndmask_b32_e64 v19, v19, v47, s2
	v_cndmask_b32_e64 v20, v20, v39, s2
	v_cndmask_b32_e64 v18, v18, v23, s2
	v_cndmask_b32_e64 v3, v3, v64, s3
	v_cndmask_b32_e64 v11, v11, v56, s3
	v_cndmask_b32_e64 v19, v19, v48, s3
	v_cndmask_b32_e64 v4, v20, v40, s3
	v_cndmask_b32_e64 v10, v10, v15, s2
	v_cndmask_b32_e64 v3, v3, v65, s4
	v_cndmask_b32_e64 v11, v11, v57, s4
	v_cndmask_b32_e64 v19, v19, v49, s4
	v_cndmask_b32_e64 v4, v4, v41, s4
	v_cndmask_b32_e64 v2, v2, v7, s2
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b64 s[0:1], s[0:1], 11
	v_add_co_u32 v3, s5, v3, v11
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v11, null, 0, 0, s5
	v_cndmask_b32_e64 v2, v2, v8, s3
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_add_co_u32 v3, s5, v3, v19
	v_cndmask_b32_e32 v19, v26, v30, vcc_lo
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v11, null, 0, v11, s5
	v_add_co_u32 v3, vcc_lo, v3, v4
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v12, v19, v31, s2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v4, null, 0, v11, vcc_lo
	v_cndmask_b32_e64 v2, v2, v9, s4
	s_wait_kmcnt 0x0
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[0:1], s[8:9], s[0:1]
	v_cndmask_b32_e64 v5, v12, v32, s3
	v_cndmask_b32_e64 v12, v18, v24, s3
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v5, v5, v33, s4
	v_cndmask_b32_e64 v6, v12, v25, s4
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_add_co_u32 v3, vcc_lo, v3, v5
	v_cndmask_b32_e64 v5, v10, v16, s3
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v4, null, 0, v4, vcc_lo
	v_add_co_u32 v3, vcc_lo, v3, v6
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v5, v5, v17, s4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v4, null, 0, v4, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v3, vcc_lo, v3, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v4, null, 0, v4, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v2, vcc_lo, v3, v2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, 0, v4, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v0, vcc_lo, s0, v0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, s1, v1, vcc_lo
	global_store_b64 v[0:1], v[2:3], off
	s_endpgm
.Lfunc_end2:
	.size	_Z10iu4_ladderILi2EEvPKDv2_jS2_PKjPfPyiii, .Lfunc_end2-_Z10iu4_ladderILi2EEvPKDv2_jS2_PKjPfPyiii
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z10iu4_ladderILi2EEvPKDv2_jS2_PKjPfPyiii
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 52
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
		.amdhsa_next_free_vgpr 86
		.amdhsa_next_free_sgpr 10
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end2-_Z10iu4_ladderILi2EEvPKDv2_jS2_PKjPfPyiii)<<4)&4080)>>4
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._Z10iu4_ladderILi2EEvPKDv2_jS2_PKjPfPyiii,"axG",@progbits,_Z10iu4_ladderILi2EEvPKDv2_jS2_PKjPfPyiii,comdat
                                        ; -- End function
	.set .L_Z10iu4_ladderILi2EEvPKDv2_jS2_PKjPfPyiii.num_vgpr, 86
	.set .L_Z10iu4_ladderILi2EEvPKDv2_jS2_PKjPfPyiii.num_agpr, 0
	.set .L_Z10iu4_ladderILi2EEvPKDv2_jS2_PKjPfPyiii.numbered_sgpr, 10
	.set .L_Z10iu4_ladderILi2EEvPKDv2_jS2_PKjPfPyiii.num_named_barrier, 0
	.set .L_Z10iu4_ladderILi2EEvPKDv2_jS2_PKjPfPyiii.private_seg_size, 0
	.set .L_Z10iu4_ladderILi2EEvPKDv2_jS2_PKjPfPyiii.uses_vcc, 1
	.set .L_Z10iu4_ladderILi2EEvPKDv2_jS2_PKjPfPyiii.uses_flat_scratch, 0
	.set .L_Z10iu4_ladderILi2EEvPKDv2_jS2_PKjPfPyiii.has_dyn_sized_stack, 0
	.set .L_Z10iu4_ladderILi2EEvPKDv2_jS2_PKjPfPyiii.has_recursion, 0
	.set .L_Z10iu4_ladderILi2EEvPKDv2_jS2_PKjPfPyiii.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 2100
; TotalNumSgprs: 12
; NumVgprs: 86
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 10
; NumSGPRsForWavesPerEU: 12
; NumVGPRsForWavesPerEU: 86
; Occupancy: 16
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.section	.text._Z10iu4_ladderILi3EEvPKDv2_jS2_PKjPfPyiii,"axG",@progbits,_Z10iu4_ladderILi3EEvPKDv2_jS2_PKjPfPyiii,comdat
	.protected	_Z10iu4_ladderILi3EEvPKDv2_jS2_PKjPfPyiii ; -- Begin function _Z10iu4_ladderILi3EEvPKDv2_jS2_PKjPfPyiii
	.globl	_Z10iu4_ladderILi3EEvPKDv2_jS2_PKjPfPyiii
	.p2align	8
	.type	_Z10iu4_ladderILi3EEvPKDv2_jS2_PKjPfPyiii,@function
_Z10iu4_ladderILi3EEvPKDv2_jS2_PKjPfPyiii: ; @_Z10iu4_ladderILi3EEvPKDv2_jS2_PKjPfPyiii
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_clause 0x1
	s_load_b64 s[8:9], s[0:1], 0x28
	s_load_b128 s[4:7], s[0:1], 0x0
	s_mov_b32 s10, ttmp9
	s_ashr_i32 s11, ttmp9, 31
	s_mov_b32 s12, ttmp7
	s_ashr_i32 s13, ttmp7, 31
	v_dual_mov_b32 v8, 0 :: v_dual_lshlrev_b32 v65, 4, v0
	v_lshlrev_b32_e32 v73, 3, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_mov_b32 v43, v8 :: v_dual_add_nc_u32 v74, 0, v65
	v_dual_mov_b32 v7, v8 :: v_dual_mov_b32 v4, v8
	v_dual_mov_b32 v6, v8 :: v_dual_mov_b32 v5, v8
	v_dual_mov_b32 v2, v8 :: v_dual_mov_b32 v3, v8
	v_mov_b32_e32 v16, v8
	s_wait_kmcnt 0x0
	s_lshl_b32 s2, s8, 1
	v_dual_mov_b32 v1, v8 :: v_dual_mov_b32 v14, v8
	s_ashr_i32 s3, s2, 31
	v_dual_mov_b32 v15, v8 :: v_dual_mov_b32 v12, v8
	s_mul_u64 s[10:11], s[2:3], s[10:11]
	s_mul_u64 s[12:13], s[2:3], s[12:13]
	s_lshl_b64 s[14:15], s[10:11], 12
	s_lshl_b64 s[16:17], s[12:13], 12
	s_add_nc_u64 s[14:15], s[4:5], s[14:15]
	s_add_nc_u64 s[16:17], s[6:7], s[16:17]
	s_clause 0x1
	global_load_b128 v[49:52], v65, s[14:15]
	global_load_b128 v[53:56], v65, s[16:17]
	v_dual_mov_b32 v13, v8 :: v_dual_mov_b32 v10, v8
	v_dual_mov_b32 v11, v8 :: v_dual_mov_b32 v24, v8
	v_dual_mov_b32 v9, v8 :: v_dual_mov_b32 v22, v8
	v_dual_mov_b32 v23, v8 :: v_dual_mov_b32 v20, v8
	v_dual_mov_b32 v21, v8 :: v_dual_mov_b32 v18, v8
	v_dual_mov_b32 v19, v8 :: v_dual_mov_b32 v32, v8
	v_dual_mov_b32 v17, v8 :: v_dual_mov_b32 v30, v8
	v_dual_mov_b32 v31, v8 :: v_dual_mov_b32 v28, v8
	v_dual_mov_b32 v29, v8 :: v_dual_mov_b32 v26, v8
	v_dual_mov_b32 v27, v8 :: v_dual_mov_b32 v40, v8
	v_dual_mov_b32 v25, v8 :: v_dual_mov_b32 v38, v8
	v_dual_mov_b32 v39, v8 :: v_dual_mov_b32 v36, v8
	v_dual_mov_b32 v37, v8 :: v_dual_mov_b32 v34, v8
	v_dual_mov_b32 v35, v8 :: v_dual_mov_b32 v48, v8
	v_dual_mov_b32 v33, v8 :: v_dual_mov_b32 v46, v8
	v_dual_mov_b32 v47, v8 :: v_dual_mov_b32 v44, v8
	v_dual_mov_b32 v45, v8 :: v_dual_mov_b32 v42, v8
	v_dual_mov_b32 v41, v8 :: v_dual_mov_b32 v64, v8
	v_dual_mov_b32 v63, v8 :: v_dual_mov_b32 v62, v8
	v_dual_mov_b32 v61, v8 :: v_dual_mov_b32 v60, v8
	v_dual_mov_b32 v59, v8 :: v_dual_mov_b32 v58, v8
	v_mov_b32_e32 v57, v8
	s_mov_b32 s15, 0
	s_cmp_lt_i32 s8, 1
	s_wait_loadcnt 0x1
	ds_store_b128 v74, v[49:52]
	s_wait_loadcnt 0x0
	ds_store_b128 v74, v[53:56] offset:8192
	v_dual_mov_b32 v56, v8 :: v_dual_mov_b32 v55, v8
	v_dual_mov_b32 v54, v8 :: v_dual_mov_b32 v53, v8
	v_dual_mov_b32 v52, v8 :: v_dual_mov_b32 v51, v8
	v_dual_mov_b32 v50, v8 :: v_dual_mov_b32 v49, v8
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_cbranch_scc1 .LBB3_7
; %bb.1:                                ; %.lr.ph
	v_dual_mov_b32 v49, 0 :: v_dual_lshlrev_b32 v2, 6, v0
	v_and_b32_e32 v1, 0xf8, v73
	v_lshlrev_b32_e32 v4, 1, v0
	v_and_b32_e32 v3, 0xe00, v65
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_mov_b32 v53, v49 :: v_dual_and_b32 v76, 0xc00, v65
	v_dual_mov_b32 v50, v49 :: v_dual_add_nc_u32 v75, 0, v1
	v_dual_mov_b32 v54, v49 :: v_dual_and_b32 v1, 0x800, v2
	v_dual_mov_b32 v56, v49 :: v_dual_mov_b32 v51, v49
	v_dual_mov_b32 v55, v49 :: v_dual_lshlrev_b32 v2, 3, v4
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_mov_b32 v58, v49 :: v_dual_add_nc_u32 v79, v75, v1
	v_mov_b32_e32 v52, v49
	v_add_co_u32 v80, s3, s4, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v81, null, s5, 0, s3
	v_add_co_u32 v82, s3, s6, v2
	v_or_b32_e32 v77, 0x200, v3
	v_or_b32_e32 v78, 0x300, v3
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v83, null, s7, 0, s3
	v_dual_mov_b32 v57, v49 :: v_dual_mov_b32 v60, v49
	v_dual_mov_b32 v59, v49 :: v_dual_mov_b32 v62, v49
	v_dual_mov_b32 v61, v49 :: v_dual_mov_b32 v64, v49
	v_dual_mov_b32 v63, v49 :: v_dual_mov_b32 v42, v49
	v_dual_mov_b32 v41, v49 :: v_dual_mov_b32 v44, v49
	v_dual_mov_b32 v43, v49 :: v_dual_mov_b32 v46, v49
	v_dual_mov_b32 v45, v49 :: v_dual_mov_b32 v48, v49
	v_dual_mov_b32 v47, v49 :: v_dual_mov_b32 v34, v49
	v_dual_mov_b32 v33, v49 :: v_dual_mov_b32 v36, v49
	v_dual_mov_b32 v35, v49 :: v_dual_mov_b32 v38, v49
	v_dual_mov_b32 v37, v49 :: v_dual_mov_b32 v40, v49
	v_dual_mov_b32 v39, v49 :: v_dual_mov_b32 v26, v49
	v_dual_mov_b32 v25, v49 :: v_dual_mov_b32 v28, v49
	v_dual_mov_b32 v27, v49 :: v_dual_mov_b32 v30, v49
	v_dual_mov_b32 v29, v49 :: v_dual_mov_b32 v32, v49
	v_dual_mov_b32 v31, v49 :: v_dual_mov_b32 v18, v49
	v_dual_mov_b32 v17, v49 :: v_dual_mov_b32 v20, v49
	v_dual_mov_b32 v19, v49 :: v_dual_mov_b32 v22, v49
	v_dual_mov_b32 v21, v49 :: v_dual_mov_b32 v24, v49
	v_dual_mov_b32 v23, v49 :: v_dual_mov_b32 v10, v49
	v_dual_mov_b32 v9, v49 :: v_dual_mov_b32 v12, v49
	v_dual_mov_b32 v11, v49 :: v_dual_mov_b32 v14, v49
	v_dual_mov_b32 v13, v49 :: v_dual_mov_b32 v16, v49
	v_dual_mov_b32 v15, v49 :: v_dual_mov_b32 v2, v49
	v_dual_mov_b32 v1, v49 :: v_dual_mov_b32 v4, v49
	v_dual_mov_b32 v3, v49 :: v_dual_mov_b32 v6, v49
	v_dual_mov_b32 v5, v49 :: v_dual_mov_b32 v8, v49
	v_mov_b32_e32 v7, v49
	s_movk_i32 s3, 0x2000
	s_mov_b32 s4, 0
	s_mov_b32 s14, s15
	s_branch .LBB3_3
.LBB3_2:                                ;   in Loop: Header=BB3_3 Depth=1
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_xor_b32 s4, s4, 1
	s_mov_b32 s14, s5
	s_cmp_eq_u32 s5, s2
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_cbranch_scc1 .LBB3_7
.LBB3_3:                                ; =>This Inner Loop Header: Depth=1
	s_add_co_i32 s5, s14, 1
	v_dual_mov_b32 v65, 0 :: v_dual_mov_b32 v66, 0
	v_dual_mov_b32 v67, 0 :: v_dual_mov_b32 v68, 0
	v_dual_mov_b32 v69, 0 :: v_dual_mov_b32 v70, 0
	v_dual_mov_b32 v71, 0 :: v_dual_mov_b32 v72, 0
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s5, s2
	s_cselect_b32 s6, -1, 0
	s_cmp_ge_i32 s5, s2
	s_cbranch_scc1 .LBB3_5
; %bb.4:                                ;   in Loop: Header=BB3_3 Depth=1
	s_add_nc_u64 s[16:17], s[10:11], s[14:15]
	s_add_nc_u64 s[18:19], s[12:13], s[14:15]
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b64 s[16:17], s[16:17], 12
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v65, vcc_lo, v80, s16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v66, null, s17, v81, vcc_lo
	s_lshl_b64 s[16:17], s[18:19], 12
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v69, vcc_lo, v82, s16
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v70, null, s17, v83, vcc_lo
	global_load_b128 v[65:68], v[65:66], off offset:4096
	global_load_b128 v[69:72], v[69:70], off offset:4096
.LBB3_5:                                ;   in Loop: Header=BB3_3 Depth=1
	s_cmp_eq_u32 s4, 0
	s_cselect_b32 s7, 0, 0x1000
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v106, s7, v75
	s_cselect_b32 s7, s3, 0x3000
	s_and_not1_b32 vcc_lo, exec_lo, s6
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v100, s7, v79
	v_add_nc_u32_e32 v88, v106, v77
	v_add_nc_u32_e32 v96, v106, v76
	v_add_nc_u32_e32 v106, v106, v78
	ds_load_2addr_b64 v[84:87], v100 offset1:32
	ds_load_b64 v[104:105], v88
	ds_load_2addr_b64 v[88:91], v100 offset0:64 offset1:96
	ds_load_2addr_b64 v[92:95], v100 offset0:128 offset1:160
	ds_load_2addr_b64 v[96:99], v96 offset1:32
	ds_load_2addr_b64 v[100:103], v100 offset0:192 offset1:224
	ds_load_b64 v[106:107], v106
	s_wait_dscnt 0x5
	v_wmma_i32_16x16x32_iu4 v[25:32], v[104:105], v[84:85], v[25:32] neg_lo:[1,1,0]
	s_wait_dscnt 0x4
	v_wmma_i32_16x16x32_iu4 v[17:24], v[104:105], v[88:89], v[17:24] neg_lo:[1,1,0]
	s_wait_dscnt 0x3
	v_wmma_i32_16x16x32_iu4 v[9:16], v[104:105], v[92:93], v[9:16] neg_lo:[1,1,0]
	s_wait_dscnt 0x2
	v_wmma_i32_16x16x32_iu4 v[49:56], v[96:97], v[84:85], v[49:56] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[57:64], v[96:97], v[88:89], v[57:64] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[41:48], v[96:97], v[92:93], v[41:48] neg_lo:[1,1,0]
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x32_iu4 v[33:40], v[96:97], v[100:101], v[33:40] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:8], v[104:105], v[100:101], v[1:8] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[49:56], v[98:99], v[86:87], v[49:56] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[57:64], v[98:99], v[90:91], v[57:64] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[41:48], v[98:99], v[94:95], v[41:48] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[33:40], v[98:99], v[102:103], v[33:40] neg_lo:[1,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[25:32], v[106:107], v[86:87], v[25:32] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:24], v[106:107], v[90:91], v[17:24] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:16], v[106:107], v[94:95], v[9:16] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:8], v[106:107], v[102:103], v[1:8] neg_lo:[1,1,0]
	s_cbranch_vccnz .LBB3_2
; %bb.6:                                ;   in Loop: Header=BB3_3 Depth=1
	s_cmp_eq_u32 s4, 1
	s_cselect_b32 s6, 0, 0x1000
	s_cselect_b32 s7, s3, 0x3000
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v84, s6, v74
	v_add_nc_u32_e32 v85, s7, v74
	s_wait_loadcnt 0x1
	ds_store_b128 v84, v[65:68]
	s_wait_loadcnt 0x0
	ds_store_b128 v85, v[69:72]
	s_branch .LBB3_2
.LBB3_7:                                ; %Flow482
	v_and_b32_e32 v0, 7, v0
	s_load_b64 s[6:7], s[0:1], 0x20
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_eq_u32_e32 vcc_lo, 1, v0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v49, v49, v50, vcc_lo
	v_cmp_eq_u32_e64 s0, 2, v0
	v_cndmask_b32_e32 v50, v57, v58, vcc_lo
	v_cmp_eq_u32_e64 s1, 3, v0
	v_cndmask_b32_e32 v9, v9, v10, vcc_lo
	v_cndmask_b32_e32 v25, v25, v26, vcc_lo
	v_cndmask_b32_e64 v49, v49, v51, s0
	v_cndmask_b32_e64 v10, v50, v59, s0
	v_cndmask_b32_e32 v17, v17, v18, vcc_lo
	v_cndmask_b32_e32 v1, v1, v2, vcc_lo
	v_cmp_eq_u32_e64 s2, 5, v0
	v_cndmask_b32_e64 v2, v49, v52, s1
	v_cndmask_b32_e32 v41, v41, v42, vcc_lo
	v_cndmask_b32_e64 v10, v10, v60, s1
	v_cndmask_b32_e32 v33, v33, v34, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, 4, v0
	v_cmp_eq_u32_e64 s3, 6, v0
	v_cmp_eq_u32_e64 s4, 7, v0
	v_cndmask_b32_e64 v1, v1, v3, s0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v2, v2, v53, vcc_lo
	v_cndmask_b32_e64 v18, v41, v43, s0
	v_cndmask_b32_e32 v10, v10, v61, vcc_lo
	v_cndmask_b32_e64 v26, v33, v35, s0
	v_cndmask_b32_e64 v1, v1, v4, s1
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v2, v2, v54, s2
	v_cndmask_b32_e64 v18, v18, v44, s1
	v_cndmask_b32_e64 v10, v10, v62, s2
	v_cndmask_b32_e32 v1, v1, v5, vcc_lo
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cndmask_b32_e64 v2, v2, v55, s3
	v_cndmask_b32_e32 v18, v18, v45, vcc_lo
	s_delay_alu instid0(VALU_DEP_4)
	v_cndmask_b32_e64 v0, v10, v63, s3
	v_cndmask_b32_e64 v25, v25, v27, s0
	v_cndmask_b32_e64 v10, v26, v36, s1
	v_cndmask_b32_e64 v2, v2, v56, s4
	v_cndmask_b32_e64 v18, v18, v46, s2
	v_cndmask_b32_e64 v0, v0, v64, s4
	v_cndmask_b32_e64 v1, v1, v6, s2
	v_cndmask_b32_e32 v10, v10, v37, vcc_lo
	v_cndmask_b32_e64 v17, v17, v19, s0
	v_cndmask_b32_e64 v18, v18, v47, s3
	v_cndmask_b32_e64 v19, v25, v28, s1
	v_add_co_u32 v0, s5, v2, v0
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cndmask_b32_e64 v17, v17, v20, s1
	v_cndmask_b32_e64 v18, v18, v48, s4
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, 0, 0, s5
	v_cndmask_b32_e64 v10, v10, v38, s2
	v_cndmask_b32_e32 v17, v17, v21, vcc_lo
	v_add_co_u32 v0, s5, v0, v18
	v_cndmask_b32_e32 v18, v19, v29, vcc_lo
	v_cndmask_b32_e64 v9, v9, v11, s0
	v_cndmask_b32_e64 v3, v10, v39, s3
	v_cndmask_b32_e64 v11, v17, v22, s2
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v2, null, 0, v2, s5
	v_cndmask_b32_e64 v10, v18, v30, s2
	v_cndmask_b32_e64 v9, v9, v12, s1
	v_cndmask_b32_e64 v3, v3, v40, s4
	v_cndmask_b32_e64 v1, v1, v7, s3
	s_mul_i32 s0, s9, ttmp7
	v_cndmask_b32_e64 v4, v10, v31, s3
	v_cndmask_b32_e32 v9, v9, v13, vcc_lo
	v_cndmask_b32_e64 v10, v11, v23, s3
	v_add_co_u32 v0, vcc_lo, v0, v3
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cndmask_b32_e64 v3, v4, v32, s4
	v_cndmask_b32_e64 v5, v9, v14, s2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, 0, v2, vcc_lo
	v_cndmask_b32_e64 v4, v10, v24, s4
	v_add_co_u32 v0, vcc_lo, v0, v3
	v_cndmask_b32_e64 v3, v5, v15, s3
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, 0, v2, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_co_u32 v0, vcc_lo, v0, v4
	v_cndmask_b32_e64 v3, v3, v16, s4
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_ci_u32_e64 v2, null, 0, v2, vcc_lo
	v_cndmask_b32_e64 v1, v1, v8, s4
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s0, s0, ttmp9
	v_add_co_u32 v0, vcc_lo, v0, v3
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, 0, v2, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_ashr_i32 s1, s0, 31
	v_add_co_u32 v0, vcc_lo, v0, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, 0, v2, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b64 s[0:1], s[0:1], 11
	s_wait_kmcnt 0x0
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[0:1], s[6:7], s[0:1]
	global_store_b64 v73, v[0:1], s[0:1]
	s_nop 0
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
.Lfunc_end3:
	.size	_Z10iu4_ladderILi3EEvPKDv2_jS2_PKjPfPyiii, .Lfunc_end3-_Z10iu4_ladderILi3EEvPKDv2_jS2_PKjPfPyiii
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z10iu4_ladderILi3EEvPKDv2_jS2_PKjPfPyiii
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 52
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
		.amdhsa_next_free_vgpr 108
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
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end3-_Z10iu4_ladderILi3EEvPKDv2_jS2_PKjPfPyiii)<<4)&4080)>>4
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._Z10iu4_ladderILi3EEvPKDv2_jS2_PKjPfPyiii,"axG",@progbits,_Z10iu4_ladderILi3EEvPKDv2_jS2_PKjPfPyiii,comdat
                                        ; -- End function
	.set .L_Z10iu4_ladderILi3EEvPKDv2_jS2_PKjPfPyiii.num_vgpr, 108
	.set .L_Z10iu4_ladderILi3EEvPKDv2_jS2_PKjPfPyiii.num_agpr, 0
	.set .L_Z10iu4_ladderILi3EEvPKDv2_jS2_PKjPfPyiii.numbered_sgpr, 20
	.set .L_Z10iu4_ladderILi3EEvPKDv2_jS2_PKjPfPyiii.num_named_barrier, 0
	.set .L_Z10iu4_ladderILi3EEvPKDv2_jS2_PKjPfPyiii.private_seg_size, 0
	.set .L_Z10iu4_ladderILi3EEvPKDv2_jS2_PKjPfPyiii.uses_vcc, 1
	.set .L_Z10iu4_ladderILi3EEvPKDv2_jS2_PKjPfPyiii.uses_flat_scratch, 0
	.set .L_Z10iu4_ladderILi3EEvPKDv2_jS2_PKjPfPyiii.has_dyn_sized_stack, 0
	.set .L_Z10iu4_ladderILi3EEvPKDv2_jS2_PKjPfPyiii.has_recursion, 0
	.set .L_Z10iu4_ladderILi3EEvPKDv2_jS2_PKjPfPyiii.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 2020
; TotalNumSgprs: 22
; NumVgprs: 108
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 13
; NumSGPRsForWavesPerEU: 22
; NumVGPRsForWavesPerEU: 108
; Occupancy: 12
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.section	.text._Z10iu4_ladderILi4EEvPKDv2_jS2_PKjPfPyiii,"axG",@progbits,_Z10iu4_ladderILi4EEvPKDv2_jS2_PKjPfPyiii,comdat
	.protected	_Z10iu4_ladderILi4EEvPKDv2_jS2_PKjPfPyiii ; -- Begin function _Z10iu4_ladderILi4EEvPKDv2_jS2_PKjPfPyiii
	.globl	_Z10iu4_ladderILi4EEvPKDv2_jS2_PKjPfPyiii
	.p2align	8
	.type	_Z10iu4_ladderILi4EEvPKDv2_jS2_PKjPfPyiii,@function
_Z10iu4_ladderILi4EEvPKDv2_jS2_PKjPfPyiii: ; @_Z10iu4_ladderILi4EEvPKDv2_jS2_PKjPfPyiii
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_clause 0x2
	s_load_b64 s[8:9], s[0:1], 0x28
	s_load_b64 s[20:21], s[0:1], 0x10
	s_load_b128 s[4:7], s[0:1], 0x0
	s_mov_b32 s16, ttmp7
	s_ashr_i32 s17, ttmp7, 31
	s_mov_b32 s12, ttmp9
	s_ashr_i32 s13, ttmp9, 31
	v_dual_mov_b32 v8, 0 :: v_dual_lshlrev_b32 v65, 4, v0
	v_lshlrev_b32_e32 v66, 2, v0
	v_lshlrev_b32_e32 v73, 3, v0
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_dual_mov_b32 v119, v8 :: v_dual_add_nc_u32 v138, 0, v65
	v_mov_b32_e32 v7, v8
	v_dual_mov_b32 v120, v8 :: v_dual_add_nc_u32 v139, 0, v66
	v_dual_mov_b32 v6, v8 :: v_dual_mov_b32 v5, v8
	v_mov_b32_e32 v4, v8
	s_wait_kmcnt 0x0
	s_ashr_i32 s19, s9, 31
	s_mov_b32 s18, s9
	s_lshl_b32 s2, s8, 1
	s_ashr_i32 s15, s8, 31
	s_mov_b32 s14, s8
	s_add_nc_u64 s[18:19], s[18:19], s[16:17]
	s_ashr_i32 s3, s2, 31
	s_mul_u64 s[10:11], s[14:15], s[12:13]
	s_mul_u64 s[14:15], s[18:19], s[14:15]
	s_mul_u64 s[12:13], s[2:3], s[12:13]
	s_mul_u64 s[16:17], s[2:3], s[16:17]
	s_lshl_b64 s[22:23], s[10:11], 10
	s_lshl_b64 s[18:19], s[14:15], 10
	s_lshl_b64 s[24:25], s[12:13], 12
	s_lshl_b64 s[26:27], s[16:17], 12
	s_add_nc_u64 s[22:23], s[20:21], s[22:23]
	s_add_nc_u64 s[18:19], s[20:21], s[18:19]
	s_add_nc_u64 s[24:25], s[4:5], s[24:25]
	s_add_nc_u64 s[26:27], s[6:7], s[26:27]
	s_clause 0x1
	global_load_b32 v71, v66, s[22:23]
	global_load_b32 v72, v66, s[18:19]
	s_clause 0x1
	global_load_b128 v[67:70], v65, s[24:25]
	global_load_b128 v[122:125], v65, s[26:27]
	v_dual_mov_b32 v3, v8 :: v_dual_mov_b32 v2, v8
	v_dual_mov_b32 v1, v8 :: v_dual_mov_b32 v16, v8
	v_dual_mov_b32 v15, v8 :: v_dual_mov_b32 v14, v8
	v_dual_mov_b32 v13, v8 :: v_dual_mov_b32 v12, v8
	v_dual_mov_b32 v11, v8 :: v_dual_mov_b32 v10, v8
	v_dual_mov_b32 v9, v8 :: v_dual_mov_b32 v24, v8
	v_dual_mov_b32 v23, v8 :: v_dual_mov_b32 v22, v8
	v_dual_mov_b32 v21, v8 :: v_dual_mov_b32 v20, v8
	v_dual_mov_b32 v19, v8 :: v_dual_mov_b32 v18, v8
	v_dual_mov_b32 v17, v8 :: v_dual_mov_b32 v32, v8
	v_dual_mov_b32 v31, v8 :: v_dual_mov_b32 v30, v8
	v_dual_mov_b32 v29, v8 :: v_dual_mov_b32 v28, v8
	v_dual_mov_b32 v27, v8 :: v_dual_mov_b32 v26, v8
	v_dual_mov_b32 v25, v8 :: v_dual_mov_b32 v40, v8
	v_dual_mov_b32 v39, v8 :: v_dual_mov_b32 v38, v8
	v_dual_mov_b32 v37, v8 :: v_dual_mov_b32 v36, v8
	v_dual_mov_b32 v35, v8 :: v_dual_mov_b32 v34, v8
	v_dual_mov_b32 v33, v8 :: v_dual_mov_b32 v48, v8
	v_dual_mov_b32 v47, v8 :: v_dual_mov_b32 v46, v8
	v_dual_mov_b32 v45, v8 :: v_dual_mov_b32 v44, v8
	v_dual_mov_b32 v43, v8 :: v_dual_mov_b32 v42, v8
	v_dual_mov_b32 v41, v8 :: v_dual_mov_b32 v56, v8
	v_dual_mov_b32 v55, v8 :: v_dual_mov_b32 v54, v8
	v_dual_mov_b32 v53, v8 :: v_dual_mov_b32 v52, v8
	v_dual_mov_b32 v51, v8 :: v_dual_mov_b32 v50, v8
	v_dual_mov_b32 v49, v8 :: v_dual_mov_b32 v64, v8
	v_dual_mov_b32 v63, v8 :: v_dual_mov_b32 v62, v8
	v_dual_mov_b32 v61, v8 :: v_dual_mov_b32 v60, v8
	v_dual_mov_b32 v59, v8 :: v_dual_mov_b32 v58, v8
	v_dual_mov_b32 v57, v8 :: v_dual_mov_b32 v92, v8
	v_dual_mov_b32 v89, v8 :: v_dual_mov_b32 v78, v8
	v_dual_mov_b32 v93, v8 :: v_dual_mov_b32 v82, v8
	v_dual_mov_b32 v95, v8 :: v_dual_mov_b32 v74, v8
	v_dual_mov_b32 v75, v8 :: v_dual_mov_b32 v76, v8
	v_dual_mov_b32 v81, v8 :: v_dual_mov_b32 v90, v8
	v_dual_mov_b32 v79, v8 :: v_dual_mov_b32 v80, v8
	v_dual_mov_b32 v83, v8 :: v_dual_mov_b32 v84, v8
	v_dual_mov_b32 v85, v8 :: v_dual_mov_b32 v86, v8
	v_dual_mov_b32 v87, v8 :: v_dual_mov_b32 v88, v8
	v_dual_mov_b32 v91, v8 :: v_dual_mov_b32 v94, v8
	v_dual_mov_b32 v77, v8 :: v_dual_mov_b32 v98, v8
	v_dual_mov_b32 v97, v8 :: v_dual_mov_b32 v96, v8
	v_dual_mov_b32 v99, v8 :: v_dual_mov_b32 v100, v8
	v_dual_mov_b32 v102, v8 :: v_dual_mov_b32 v109, v8
	v_dual_mov_b32 v104, v8 :: v_dual_mov_b32 v101, v8
	v_dual_mov_b32 v106, v8 :: v_dual_mov_b32 v103, v8
	v_dual_mov_b32 v108, v8 :: v_dual_mov_b32 v105, v8
	v_dual_mov_b32 v107, v8 :: v_dual_mov_b32 v110, v8
	v_dual_mov_b32 v112, v8 :: v_dual_mov_b32 v115, v8
	v_dual_mov_b32 v114, v8 :: v_dual_mov_b32 v111, v8
	v_dual_mov_b32 v113, v8 :: v_dual_mov_b32 v116, v8
	v_dual_mov_b32 v118, v8 :: v_dual_mov_b32 v121, v8
	v_dual_mov_b32 v126, v8 :: v_dual_mov_b32 v131, v8
	v_dual_mov_b32 v130, v8 :: v_dual_mov_b32 v117, v8
	v_dual_mov_b32 v127, v8 :: v_dual_mov_b32 v136, v8
	v_dual_mov_b32 v135, v8 :: v_dual_mov_b32 v128, v8
	v_mov_b32_e32 v129, v8
	v_dual_mov_b32 v133, v8 :: v_dual_mov_b32 v134, v8
	v_mov_b32_e32 v137, v8
	s_mov_b32 s19, 0
	s_cmp_lt_i32 s8, 1
	s_wait_loadcnt 0x1
	ds_store_b128 v138, v[67:70]
	s_wait_loadcnt 0x0
	ds_store_b128 v138, v[122:125] offset:8192
	ds_store_2addr_stride64_b32 v139, v71, v72 offset0:64 offset1:68
	v_dual_mov_b32 v123, v8 :: v_dual_mov_b32 v132, v8
	v_mov_b32_e32 v122, v8
	v_dual_mov_b32 v124, v8 :: v_dual_mov_b32 v125, v8
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_cbranch_scc1 .LBB4_15
; %bb.1:                                ; %.lr.ph
	v_dual_mov_b32 v137, 0 :: v_dual_and_b32 v2, 0xf8, v73
	v_dual_mov_b32 v134, 0 :: v_dual_lshlrev_b32 v1, 1, v0
	v_dual_mov_b32 v128, 0 :: v_dual_lshlrev_b32 v3, 6, v0
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_dual_mov_b32 v129, 0 :: v_dual_add_nc_u32 v142, 0, v2
	v_dual_mov_b32 v125, 0 :: v_dual_and_b32 v2, 15, v0
	v_dual_mov_b32 v133, 0 :: v_dual_and_b32 v4, 0xe00, v65
	v_add_co_u32 v140, s3, s20, v66
	v_and_or_b32 v2, v1, 64, v2
	v_dual_mov_b32 v136, 0 :: v_dual_lshlrev_b32 v1, 3, v1
	v_dual_mov_b32 v124, 0 :: v_dual_and_b32 v3, 0x800, v3
	v_or_b32_e32 v144, 0x200, v4
	v_or_b32_e32 v145, 0x300, v4
	v_or_b32_e32 v4, 0xb8, v66
	v_add_co_ci_u32_e64 v141, null, s21, 0, s3
	v_add_co_u32 v150, s3, s4, v1
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v151, null, s5, 0, s3
	v_add_co_u32 v152, s3, s6, v1
	v_dual_mov_b32 v122, 0 :: v_dual_and_b32 v143, 0xc00, v65
	v_dual_mov_b32 v135, 0 :: v_dual_add_nc_u32 v146, v142, v3
	v_dual_mov_b32 v132, 0 :: v_dual_and_b32 v147, 0x340, v66
	v_dual_mov_b32 v127, 0 :: v_dual_lshlrev_b32 v148, 3, v2
	v_dual_mov_b32 v120, 0 :: v_dual_and_b32 v149, 0x3f8, v4
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v153, null, s7, 0, s3
	v_dual_mov_b32 v123, 0 :: v_dual_mov_b32 v130, 0
	v_dual_mov_b32 v119, 0 :: v_dual_mov_b32 v126, 0
	v_dual_mov_b32 v117, 0 :: v_dual_mov_b32 v118, 0
	v_dual_mov_b32 v131, 0 :: v_dual_mov_b32 v116, 0
	v_dual_mov_b32 v121, 0 :: v_dual_mov_b32 v114, 0
	v_dual_mov_b32 v113, 0 :: v_dual_mov_b32 v112, 0
	v_dual_mov_b32 v111, 0 :: v_dual_mov_b32 v110, 0
	v_dual_mov_b32 v115, 0 :: v_dual_mov_b32 v108, 0
	v_dual_mov_b32 v107, 0 :: v_dual_mov_b32 v106, 0
	v_dual_mov_b32 v105, 0 :: v_dual_mov_b32 v104, 0
	v_dual_mov_b32 v103, 0 :: v_dual_mov_b32 v102, 0
	v_dual_mov_b32 v101, 0 :: v_dual_mov_b32 v100, 0
	v_dual_mov_b32 v109, 0 :: v_dual_mov_b32 v96, 0
	v_dual_mov_b32 v99, 0 :: v_dual_mov_b32 v98, 0
	v_dual_mov_b32 v97, 0 :: v_dual_mov_b32 v94, 0
	v_dual_mov_b32 v88, 0 :: v_dual_mov_b32 v77, 0
	v_dual_mov_b32 v86, 0 :: v_dual_mov_b32 v91, 0
	v_dual_mov_b32 v84, 0 :: v_dual_mov_b32 v87, 0
	v_dual_mov_b32 v80, 0 :: v_dual_mov_b32 v85, 0
	v_dual_mov_b32 v90, 0 :: v_dual_mov_b32 v83, 0
	v_dual_mov_b32 v79, 0 :: v_dual_mov_b32 v76, 0
	v_dual_mov_b32 v74, 0 :: v_dual_mov_b32 v81, 0
	v_dual_mov_b32 v82, 0 :: v_dual_mov_b32 v75, 0
	v_dual_mov_b32 v78, 0 :: v_dual_mov_b32 v57, 0
	v_dual_mov_b32 v58, v134 :: v_dual_mov_b32 v59, v134
	v_dual_mov_b32 v60, v134 :: v_dual_mov_b32 v61, v134
	v_dual_mov_b32 v62, v134 :: v_dual_mov_b32 v63, v134
	v_dual_mov_b32 v64, v134 :: v_dual_mov_b32 v49, 0
	v_dual_mov_b32 v50, v134 :: v_dual_mov_b32 v51, v134
	v_dual_mov_b32 v52, v134 :: v_dual_mov_b32 v53, v134
	v_dual_mov_b32 v54, v134 :: v_dual_mov_b32 v55, v134
	v_dual_mov_b32 v56, v134 :: v_dual_mov_b32 v41, 0
	v_dual_mov_b32 v42, v134 :: v_dual_mov_b32 v43, v134
	v_dual_mov_b32 v44, v134 :: v_dual_mov_b32 v45, v134
	v_dual_mov_b32 v46, v134 :: v_dual_mov_b32 v47, v134
	v_dual_mov_b32 v48, v134 :: v_dual_mov_b32 v33, 0
	v_dual_mov_b32 v34, v134 :: v_dual_mov_b32 v35, v134
	v_dual_mov_b32 v36, v134 :: v_dual_mov_b32 v37, v134
	v_dual_mov_b32 v38, v134 :: v_dual_mov_b32 v39, v134
	v_dual_mov_b32 v40, v134 :: v_dual_mov_b32 v25, 0
	v_dual_mov_b32 v26, v134 :: v_dual_mov_b32 v27, v134
	v_dual_mov_b32 v28, v134 :: v_dual_mov_b32 v29, v134
	v_dual_mov_b32 v30, v134 :: v_dual_mov_b32 v31, v134
	v_dual_mov_b32 v32, v134 :: v_dual_mov_b32 v17, 0
	v_dual_mov_b32 v18, v134 :: v_dual_mov_b32 v19, v134
	v_dual_mov_b32 v20, v134 :: v_dual_mov_b32 v21, v134
	v_dual_mov_b32 v22, v134 :: v_dual_mov_b32 v23, v134
	v_dual_mov_b32 v24, v134 :: v_dual_mov_b32 v9, 0
	v_dual_mov_b32 v10, v134 :: v_dual_mov_b32 v11, v134
	v_dual_mov_b32 v12, v134 :: v_dual_mov_b32 v13, v134
	v_dual_mov_b32 v14, v134 :: v_dual_mov_b32 v15, v134
	v_dual_mov_b32 v16, v134 :: v_dual_mov_b32 v1, 0
	v_dual_mov_b32 v2, v134 :: v_dual_mov_b32 v3, v134
	v_dual_mov_b32 v4, v134 :: v_dual_mov_b32 v5, v134
	v_dual_mov_b32 v6, v134 :: v_dual_mov_b32 v7, v134
	v_dual_mov_b32 v8, v134 :: v_dual_mov_b32 v95, 0
	v_dual_mov_b32 v93, 0 :: v_dual_mov_b32 v92, 0
	v_mov_b32_e32 v89, 0
	s_movk_i32 s3, 0x2000
	s_movk_i32 s6, 0x4000
	s_mov_b32 s7, 0
	s_mov_b32 s18, s19
	s_branch .LBB4_3
.LBB4_2:                                ;   in Loop: Header=BB4_3 Depth=1
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_xor_b32 s7, s7, 1
	s_mov_b32 s18, s8
	s_cmp_eq_u32 s8, s2
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_cbranch_scc1 .LBB4_15
.LBB4_3:                                ; =>This Inner Loop Header: Depth=1
	s_add_co_i32 s8, s18, 1
	s_mov_b32 s4, -1
	s_cmp_lt_i32 s8, s2
                                        ; implicit-def: $sgpr21
	s_cselect_b32 s20, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s20
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB4_5
; %bb.4:                                ; %._crit_edge
                                        ;   in Loop: Header=BB4_3 Depth=1
	s_bitcmp1_b32 s18, 0
	s_mov_b32 s4, 0
	s_cselect_b32 s21, -1, 0
.LBB4_5:                                ; %Flow1313
                                        ;   in Loop: Header=BB4_3 Depth=1
	v_dual_mov_b32 v154, 0 :: v_dual_mov_b32 v155, 0
	v_dual_mov_b32 v65, 0 :: v_dual_mov_b32 v66, 0
	v_dual_mov_b32 v67, 0 :: v_dual_mov_b32 v68, 0
	v_dual_mov_b32 v69, 0 :: v_dual_mov_b32 v70, 0
	v_dual_mov_b32 v71, 0 :: v_dual_mov_b32 v72, 0
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s4
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB4_10
; %bb.6:                                ;   in Loop: Header=BB4_3 Depth=1
	s_bitcmp1_b32 s18, 0
	s_cselect_b32 s21, -1, 0
	s_add_co_i32 s4, s18, 2
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s4, s2
	s_cselect_b32 s4, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s4, s21, s4
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s4
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB4_8
; %bb.7:                                ;   in Loop: Header=BB4_3 Depth=1
	s_lshr_b32 s4, s18, 1
	s_mov_b32 s5, s19
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s4, s4, 1
	s_mov_b32 s21, -1
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[22:23], s[10:11], s[4:5]
	s_add_nc_u64 s[4:5], s[14:15], s[4:5]
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b64 s[22:23], s[22:23], 10
	s_lshl_b64 s[4:5], s[4:5], 10
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v65, vcc_lo, v140, s22
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v66, null, s23, v141, vcc_lo
	v_add_co_u32 v67, vcc_lo, v140, s4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v68, null, s5, v141, vcc_lo
	s_clause 0x1
	global_load_b32 v155, v[65:66], off
	global_load_b32 v154, v[67:68], off
	s_branch .LBB4_9
.LBB4_8:                                ;   in Loop: Header=BB4_3 Depth=1
	v_dual_mov_b32 v155, 0 :: v_dual_mov_b32 v154, 0
.LBB4_9:                                ; %Flow1312
                                        ;   in Loop: Header=BB4_3 Depth=1
	s_add_nc_u64 s[4:5], s[12:13], s[18:19]
	s_add_nc_u64 s[22:23], s[16:17], s[18:19]
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b64 s[4:5], s[4:5], 12
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v65, vcc_lo, v150, s4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v66, null, s5, v151, vcc_lo
	s_lshl_b64 s[4:5], s[22:23], 12
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v69, vcc_lo, v152, s4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v70, null, s5, v153, vcc_lo
	global_load_b128 v[65:68], v[65:66], off offset:4096
	global_load_b128 v[69:72], v[69:70], off offset:4096
.LBB4_10:                               ;   in Loop: Header=BB4_3 Depth=1
	s_cmp_eq_u32 s7, 0
	s_cselect_b32 s4, 0, 0x1000
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v178, s4, v142
	s_cselect_b32 s4, s3, 0x3000
	s_and_not1_b32 vcc_lo, exec_lo, s21
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v172, s4, v146
	v_add_nc_u32_e32 v160, v178, v143
	v_add_nc_u32_e32 v176, v178, v144
	v_add_nc_u32_e32 v178, v178, v145
	ds_load_2addr_b64 v[156:159], v172 offset1:32
	ds_load_2addr_b64 v[160:163], v160 offset1:32
	ds_load_2addr_b64 v[164:167], v172 offset0:64 offset1:96
	ds_load_2addr_b64 v[168:171], v172 offset0:128 offset1:160
	ds_load_2addr_b64 v[172:175], v172 offset0:192 offset1:224
	ds_load_b64 v[176:177], v176
	ds_load_b64 v[178:179], v178
	s_wait_dscnt 0x5
	v_wmma_i32_16x16x32_iu4 v[57:64], v[160:161], v[156:157], v[57:64] neg_lo:[1,1,0]
	s_wait_dscnt 0x4
	v_wmma_i32_16x16x32_iu4 v[49:56], v[160:161], v[164:165], v[49:56] neg_lo:[1,1,0]
	s_wait_dscnt 0x3
	v_wmma_i32_16x16x32_iu4 v[41:48], v[160:161], v[168:169], v[41:48] neg_lo:[1,1,0]
	s_wait_dscnt 0x2
	v_wmma_i32_16x16x32_iu4 v[33:40], v[160:161], v[172:173], v[33:40] neg_lo:[1,1,0]
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x32_iu4 v[25:32], v[176:177], v[156:157], v[25:32] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:24], v[176:177], v[164:165], v[17:24] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:16], v[176:177], v[168:169], v[9:16] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:8], v[176:177], v[172:173], v[1:8] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[57:64], v[162:163], v[158:159], v[57:64] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[49:56], v[162:163], v[166:167], v[49:56] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[41:48], v[162:163], v[170:171], v[41:48] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[33:40], v[162:163], v[174:175], v[33:40] neg_lo:[1,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[25:32], v[178:179], v[158:159], v[25:32] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:24], v[178:179], v[166:167], v[17:24] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:16], v[178:179], v[170:171], v[9:16] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:8], v[178:179], v[174:175], v[1:8] neg_lo:[1,1,0]
	s_cbranch_vccnz .LBB4_12
; %bb.11:                               ; %.preheader214
                                        ;   in Loop: Header=BB4_3 Depth=1
	s_bitcmp0_b32 s18, 1
	s_mov_b64 s[4:5], src_shared_base
	s_cselect_b32 s22, s6, 0x4800
	v_cvt_f32_i32_e32 v57, v57
	s_wait_alu depctr_sa_sdst(0)
	v_dual_mov_b32 v157, s5 :: v_dual_add_nc_u32 v156, s22, v147
	v_cvt_f32_i32_e32 v58, v58
	v_cvt_f32_i32_e32 v60, v60
	v_cvt_f32_i32_e32 v51, v51
	v_mov_b32_e32 v158, s5
	flat_load_b32 v159, v[156:157] scope:SCOPE_SYS
	s_wait_loadcnt 0x0
	v_add_nc_u32_e32 v157, 8, v156
	v_add_nc_u32_e32 v167, s22, v148
	v_cvt_f32_i32_e32 v50, v50
	v_cvt_f32_i32_e32 v59, v59
	v_cvt_f32_i32_e32 v49, v49
	flat_load_b32 v160, v[157:158] scope:SCOPE_SYS
	s_wait_loadcnt 0x0
	v_add_nc_u32_e32 v157, 16, v156
	v_cvt_f32_i32_e32 v41, v41
	v_cvt_f32_i32_e32 v52, v52
	v_cvt_f32_i32_e32 v43, v43
	v_cvt_f32_i32_e32 v33, v33
	flat_load_b32 v161, v[157:158] scope:SCOPE_SYS
	s_wait_loadcnt 0x0
	v_add_nc_u32_e32 v157, 24, v156
	v_cvt_f32_i32_e32 v42, v42
	v_cvt_f32_i32_e32 v35, v35
	v_cvt_f32_i32_e32 v44, v44
	v_cvt_f32_i32_e32 v36, v36
	flat_load_b32 v162, v[157:158] scope:SCOPE_SYS
	s_wait_loadcnt 0x0
	v_add_nc_u32_e32 v157, 32, v156
	v_cvt_f32_i32_e32 v25, v25
	v_cvt_f32_i32_e32 v34, v34
	v_cvt_f32_i32_e32 v27, v27
	v_cvt_f32_i32_e32 v28, v28
	flat_load_b32 v163, v[157:158] scope:SCOPE_SYS
	s_wait_loadcnt 0x0
	v_add_nc_u32_e32 v157, 40, v156
	v_cvt_f32_i32_e32 v18, v18
	v_cvt_f32_i32_e32 v19, v19
	v_cvt_f32_i32_e32 v9, v9
	v_cvt_f32_i32_e32 v20, v20
	flat_load_b32 v164, v[157:158] scope:SCOPE_SYS
	s_wait_loadcnt 0x0
	v_add_nc_u32_e32 v157, 48, v156
	v_cvt_f32_i32_e32 v2, v2
	v_cvt_f32_i32_e32 v26, v26
	v_cvt_f32_i32_e32 v17, v17
	v_cvt_f32_i32_e32 v11, v11
	flat_load_b32 v165, v[157:158] scope:SCOPE_SYS
	s_wait_loadcnt 0x0
	v_add_nc_u32_e32 v157, 56, v156
	v_cvt_f32_i32_e32 v10, v10
	v_cvt_f32_i32_e32 v12, v12
	v_cvt_f32_i32_e32 v3, v3
	v_cvt_f32_i32_e32 v1, v1
	flat_load_b32 v166, v[157:158] scope:SCOPE_SYS
	s_wait_loadcnt 0x0
	v_add_nc_u32_e32 v157, 0x400, v167
	v_cvt_f32_i32_e32 v4, v4
	flat_load_b32 v168, v[157:158] scope:SCOPE_SYS
	s_wait_loadcnt_dscnt 0x0
	v_dual_mul_f32 v170, v168, v160 :: v_dual_mul_f32 v169, v168, v159
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v137, v170, v58
	v_mul_f32_e32 v58, v168, v162
	v_fmac_f32_e32 v129, v58, v60
	v_mul_f32_e32 v58, v168, v164
	v_cvt_f32_i32_e32 v60, v62
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v134, v169, v57 :: v_dual_fmac_f32 v125, v58, v60
	v_mul_f32_e32 v58, v168, v166
	v_cvt_f32_i32_e32 v60, v64
	v_dual_mul_f32 v57, v168, v161 :: v_dual_fmac_f32 v122, v58, v60
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v133, v57, v59
	v_mul_f32_e32 v57, v168, v163
	v_cvt_f32_i32_e32 v59, v61
	v_mov_b32_e32 v58, s5
	v_dual_fmac_f32 v128, v57, v59 :: v_dual_mul_f32 v57, v168, v165
	v_cvt_f32_i32_e32 v59, v63
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v124, v57, v59
	v_add_nc_u32_e32 v57, 0x480, v167
	flat_load_b32 v59, v[57:58] scope:SCOPE_SYS
	s_wait_loadcnt_dscnt 0x0
	v_dual_mul_f32 v60, v59, v159 :: v_dual_mul_f32 v61, v59, v160
	v_dual_fmac_f32 v135, v60, v49 :: v_dual_fmac_f32 v136, v61, v50
	v_dual_mul_f32 v49, v59, v161 :: v_dual_mul_f32 v50, v59, v162
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v132, v49, v51 :: v_dual_fmac_f32 v127, v50, v52
	v_dual_mul_f32 v49, v59, v163 :: v_dual_mul_f32 v50, v59, v164
	v_cvt_f32_i32_e32 v51, v53
	v_cvt_f32_i32_e32 v52, v54
	v_dual_fmac_f32 v123, v49, v51 :: v_dual_fmac_f32 v120, v50, v52
	v_dual_mul_f32 v49, v59, v165 :: v_dual_mul_f32 v50, v59, v166
	v_cvt_f32_i32_e32 v51, v55
	v_cvt_f32_i32_e32 v52, v56
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v119, v49, v51
	v_fmac_f32_e32 v117, v50, v52
	v_dual_mov_b32 v50, s5 :: v_dual_add_nc_u32 v49, 0x500, v167
	flat_load_b32 v51, v[49:50] scope:SCOPE_SYS
	s_wait_loadcnt_dscnt 0x0
	v_dual_mul_f32 v52, v51, v159 :: v_dual_mul_f32 v53, v51, v160
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v130, v52, v41 :: v_dual_fmac_f32 v131, v53, v42
	v_dual_mul_f32 v41, v51, v161 :: v_dual_mul_f32 v42, v51, v162
	v_dual_fmac_f32 v126, v41, v43 :: v_dual_fmac_f32 v121, v42, v44
	v_dual_mul_f32 v41, v51, v163 :: v_dual_mul_f32 v42, v51, v164
	v_cvt_f32_i32_e32 v43, v45
	v_cvt_f32_i32_e32 v44, v46
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v118, v41, v43
	v_dual_fmac_f32 v116, v42, v44 :: v_dual_mul_f32 v41, v51, v165
	v_mul_f32_e32 v42, v51, v166
	v_cvt_f32_i32_e32 v43, v47
	v_cvt_f32_i32_e32 v44, v48
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v113, v41, v43
	v_fmac_f32_e32 v111, v42, v44
	v_dual_mov_b32 v42, s5 :: v_dual_add_nc_u32 v41, 0x580, v167
	flat_load_b32 v43, v[41:42] scope:SCOPE_SYS
	s_wait_loadcnt_dscnt 0x0
	v_dual_mul_f32 v44, v43, v159 :: v_dual_mul_f32 v45, v43, v160
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v114, v44, v33 :: v_dual_fmac_f32 v115, v45, v34
	v_dual_mul_f32 v33, v43, v161 :: v_dual_mul_f32 v34, v43, v162
	v_fmac_f32_e32 v112, v33, v35
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_2)
	v_dual_fmac_f32 v110, v34, v36 :: v_dual_mul_f32 v33, v43, v163
	v_mul_f32_e32 v34, v43, v164
	v_cvt_f32_i32_e32 v35, v37
	v_cvt_f32_i32_e32 v36, v38
	v_fmac_f32_e32 v107, v33, v35
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v105, v34, v36
	v_dual_mul_f32 v33, v43, v165 :: v_dual_mul_f32 v34, v43, v166
	v_cvt_f32_i32_e32 v35, v39
	v_cvt_f32_i32_e32 v36, v40
	v_fmac_f32_e32 v103, v33, v35
	s_delay_alu instid0(VALU_DEP_2)
	v_fmac_f32_e32 v101, v34, v36
	v_dual_mov_b32 v34, s5 :: v_dual_add_nc_u32 v33, 0x80, v156
	flat_load_b32 v35, v[33:34] scope:SCOPE_SYS
	s_wait_loadcnt 0x0
	v_add_nc_u32_e32 v33, 0x88, v156
	flat_load_b32 v36, v[33:34] scope:SCOPE_SYS
	s_wait_loadcnt 0x0
	v_add_nc_u32_e32 v33, 0x90, v156
	flat_load_b32 v37, v[33:34] scope:SCOPE_SYS
	s_wait_loadcnt 0x0
	v_add_nc_u32_e32 v33, 0x98, v156
	flat_load_b32 v38, v[33:34] scope:SCOPE_SYS
	s_wait_loadcnt 0x0
	v_add_nc_u32_e32 v33, 0xa0, v156
	flat_load_b32 v39, v[33:34] scope:SCOPE_SYS
	s_wait_loadcnt 0x0
	v_add_nc_u32_e32 v33, 0xa8, v156
	flat_load_b32 v40, v[33:34] scope:SCOPE_SYS
	s_wait_loadcnt 0x0
	v_add_nc_u32_e32 v33, 0xb0, v156
	flat_load_b32 v43, v[33:34] scope:SCOPE_SYS
	s_wait_loadcnt 0x0
	v_add_nc_u32_e32 v33, s22, v149
	flat_load_b32 v33, v[33:34] scope:SCOPE_SYS
	s_wait_loadcnt 0x0
	flat_load_b32 v34, v[157:158] scope:SCOPE_SYS
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v44, v34, v35
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v108, v44, v25
	v_mul_f32_e32 v25, v34, v37
	v_fmac_f32_e32 v106, v25, v27
	v_mul_f32_e32 v25, v34, v39
	v_cvt_f32_i32_e32 v27, v29
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v102, v25, v27
	v_mul_f32_e32 v25, v34, v43
	v_cvt_f32_i32_e32 v27, v31
	v_fmac_f32_e32 v99, v25, v27
	flat_load_b32 v25, v[57:58] scope:SCOPE_SYS
	s_wait_loadcnt 0x0
	v_mul_f32_e32 v45, v34, v36
	v_mov_b32_e32 v57, 0
	s_delay_alu instid0(VALU_DEP_1)
	v_mov_b32_e32 v59, v57
	v_dual_mov_b32 v61, v57 :: v_dual_mov_b32 v52, v57
	v_dual_mov_b32 v63, v57 :: v_dual_mov_b32 v54, v57
	v_mov_b32_e32 v51, v57
	v_dual_mov_b32 v53, v57 :: v_dual_mov_b32 v44, v57
	v_dual_mov_b32 v55, v57 :: v_dual_mov_b32 v46, v57
	v_mov_b32_e32 v47, v57
	v_mov_b32_e32 v29, v57
	v_dual_mov_b32 v31, v57 :: v_dual_mov_b32 v56, v57
	v_mov_b32_e32 v48, v57
	v_mov_b32_e32 v58, v57
	v_mov_b32_e32 v60, v57
	v_mov_b32_e32 v62, v57
	v_mov_b32_e32 v64, v57
	s_wait_dscnt 0x0
	v_mul_f32_e32 v27, v25, v36
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v98, v27, v18
	v_mul_f32_e32 v18, v25, v38
	v_dual_mov_b32 v27, v57 :: v_dual_fmac_f32 v88, v18, v20
	v_mul_f32_e32 v18, v25, v40
	v_cvt_f32_i32_e32 v20, v22
	v_mov_b32_e32 v22, v57
	s_delay_alu instid0(VALU_DEP_2)
	v_fmac_f32_e32 v84, v18, v20
	v_mul_f32_e32 v18, v25, v33
	v_fmac_f32_e32 v109, v45, v26
	v_mul_f32_e32 v26, v34, v38
	v_cvt_f32_i32_e32 v20, v24
	v_dual_mov_b32 v45, v57 :: v_dual_mov_b32 v24, v57
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v104, v26, v28
	v_mul_f32_e32 v26, v34, v40
	v_cvt_f32_i32_e32 v28, v30
	v_dual_fmac_f32 v77, v18, v20 :: v_dual_mov_b32 v30, v57
	v_mov_b32_e32 v20, v57
	v_fmac_f32_e32 v100, v26, v28
	v_mul_f32_e32 v26, v34, v33
	v_cvt_f32_i32_e32 v28, v32
	v_mov_b32_e32 v34, v57
	v_mov_b32_e32 v32, v57
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v96, v26, v28
	v_mul_f32_e32 v26, v25, v35
	v_dual_mov_b32 v28, v57 :: v_dual_fmac_f32 v97, v26, v17
	v_mul_f32_e32 v17, v25, v37
	v_mov_b32_e32 v26, v57
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v94, v17, v19
	v_mul_f32_e32 v17, v25, v39
	v_cvt_f32_i32_e32 v19, v21
	v_mov_b32_e32 v21, v57
	v_fmac_f32_e32 v86, v17, v19
	v_mul_f32_e32 v17, v25, v43
	v_cvt_f32_i32_e32 v19, v23
	v_mov_b32_e32 v25, v57
	v_mov_b32_e32 v23, v57
	s_delay_alu instid0(VALU_DEP_3)
	v_fmac_f32_e32 v80, v17, v19
	flat_load_b32 v17, v[49:50] scope:SCOPE_SYS
	s_wait_loadcnt 0x0
	v_dual_mov_b32 v49, v57 :: v_dual_mov_b32 v50, v57
	s_wait_dscnt 0x0
	v_dual_mul_f32 v18, v17, v35 :: v_dual_mul_f32 v19, v17, v36
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v90, v18, v9
	v_mul_f32_e32 v9, v17, v37
	v_fmac_f32_e32 v91, v19, v10
	v_mul_f32_e32 v10, v17, v38
	v_dual_mov_b32 v18, v57 :: v_dual_mov_b32 v19, v57
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v87, v9, v11
	v_mul_f32_e32 v9, v17, v39
	v_cvt_f32_i32_e32 v11, v13
	v_mov_b32_e32 v13, v57
	v_fmac_f32_e32 v83, v9, v11
	v_mul_f32_e32 v9, v17, v43
	v_cvt_f32_i32_e32 v11, v15
	v_mov_b32_e32 v15, v57
	s_delay_alu instid0(VALU_DEP_2)
	v_fmac_f32_e32 v76, v9, v11
	flat_load_b32 v9, v[41:42] scope:SCOPE_SYS
	s_wait_loadcnt 0x0
	v_fmac_f32_e32 v85, v10, v12
	v_mul_f32_e32 v10, v17, v40
	v_cvt_f32_i32_e32 v12, v14
	v_dual_mov_b32 v41, v57 :: v_dual_mov_b32 v42, v57
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_2)
	v_dual_mov_b32 v14, v57 :: v_dual_fmac_f32 v79, v10, v12
	v_mul_f32_e32 v10, v17, v33
	v_cvt_f32_i32_e32 v12, v16
	v_dual_mov_b32 v17, v57 :: v_dual_mov_b32 v16, v57
	v_fmac_f32_e32 v74, v10, v12
	v_mov_b32_e32 v12, v57
	s_wait_dscnt 0x0
	v_dual_mul_f32 v10, v9, v35 :: v_dual_mul_f32 v11, v9, v36
	v_dual_mov_b32 v35, v57 :: v_dual_mov_b32 v36, v57
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v81, v10, v1 :: v_dual_fmac_f32 v82, v11, v2
	v_dual_mul_f32 v1, v9, v37 :: v_dual_mul_f32 v2, v9, v38
	v_dual_mov_b32 v37, v57 :: v_dual_mov_b32 v38, v57
	v_mov_b32_e32 v10, v57
	v_dual_fmac_f32 v78, v1, v3 :: v_dual_fmac_f32 v75, v2, v4
	v_dual_mul_f32 v1, v9, v39 :: v_dual_mul_f32 v2, v9, v40
	v_cvt_f32_i32_e32 v3, v5
	v_cvt_f32_i32_e32 v4, v6
	v_dual_mov_b32 v39, v57 :: v_dual_mov_b32 v40, v57
	v_mov_b32_e32 v11, v57
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v95, v1, v3
	v_fmac_f32_e32 v93, v2, v4
	v_dual_mul_f32 v1, v9, v43 :: v_dual_mul_f32 v2, v9, v33
	v_cvt_f32_i32_e32 v3, v7
	v_cvt_f32_i32_e32 v4, v8
	v_mov_b32_e32 v43, v57
	v_mov_b32_e32 v33, v57
	v_mov_b32_e32 v9, v57
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v92, v1, v3 :: v_dual_fmac_f32 v89, v2, v4
	v_dual_mov_b32 v1, v57 :: v_dual_mov_b32 v8, v57
	v_dual_mov_b32 v2, v57 :: v_dual_mov_b32 v3, v57
	v_dual_mov_b32 v4, v57 :: v_dual_mov_b32 v5, v57
	v_dual_mov_b32 v6, v57 :: v_dual_mov_b32 v7, v57
.LBB4_12:                               ; %.loopexit
                                        ;   in Loop: Header=BB4_3 Depth=1
	s_and_not1_b32 vcc_lo, exec_lo, s20
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB4_2
; %bb.13:                               ;   in Loop: Header=BB4_3 Depth=1
	s_cmp_eq_u32 s7, 1
	s_cselect_b32 s4, 0, 0x1000
	s_cselect_b32 s5, s3, 0x3000
	s_add_co_i32 s20, s18, 2
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v156, s4, v138
	s_cmp_lt_i32 s20, s2
	v_add_nc_u32_e32 v157, s5, v138
	s_cselect_b32 s20, -1, 0
	s_wait_loadcnt 0x1
	ds_store_b128 v156, v[65:68]
	s_wait_loadcnt 0x0
	ds_store_b128 v157, v[69:72]
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s20, s21, s20
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s20
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB4_2
; %bb.14:                               ;   in Loop: Header=BB4_3 Depth=1
	s_bitcmp0_b32 s18, 1
	s_cselect_b32 s4, 0x4800, s6
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v65, s4, v139
	ds_store_2addr_stride64_b32 v65, v155, v154 offset1:4
	s_branch .LBB4_2
.LBB4_15:                               ; %Flow1315
	v_and_b32_e32 v0, 7, v0
	s_load_b64 s[10:11], s[0:1], 0x20
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cmp_eq_u32_e64 s2, 1, v0
	v_cmp_eq_u32_e32 vcc_lo, 2, v0
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v57, v57, v58, s2
	v_cndmask_b32_e64 v58, v134, v137, s2
	v_cndmask_b32_e64 v49, v49, v50, s2
	v_cndmask_b32_e64 v50, v135, v136, s2
	v_cndmask_b32_e64 v41, v41, v42, s2
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v57, v57, v59, vcc_lo
	v_cmp_eq_u32_e64 s0, 3, v0
	v_dual_cndmask_b32 v58, v58, v133 :: v_dual_cndmask_b32 v49, v49, v51
	v_cmp_eq_u32_e64 s1, 4, v0
	v_dual_cndmask_b32 v50, v50, v132 :: v_dual_cndmask_b32 v41, v41, v43
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cndmask_b32_e64 v57, v57, v60, s0
	v_cndmask_b32_e64 v43, v58, v129, s0
	v_cndmask_b32_e64 v49, v49, v52, s0
	v_cmp_eq_u32_e64 s3, 5, v0
	v_cndmask_b32_e64 v50, v50, v127, s0
	v_cndmask_b32_e64 v51, v57, v61, s1
	v_cndmask_b32_e64 v43, v43, v128, s1
	v_cndmask_b32_e64 v49, v49, v53, s1
	v_cmp_eq_u32_e64 s4, 6, v0
	v_cndmask_b32_e64 v42, v130, v131, s2
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v51, v51, v62, s3
	v_cndmask_b32_e64 v43, v43, v125, s3
	v_cndmask_b32_e64 v41, v41, v44, s0
	v_cndmask_b32_e64 v49, v49, v54, s3
	v_cmp_eq_u32_e64 s5, 7, v0
	v_cndmask_b32_e64 v44, v51, v63, s4
	v_cndmask_b32_e64 v0, v43, v124, s4
	v_cndmask_b32_e64 v43, v50, v123, s1
	v_cndmask_b32_e64 v33, v33, v34, s2
	v_cndmask_b32_e32 v42, v42, v126, vcc_lo
	v_cndmask_b32_e64 v49, v49, v55, s4
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v44, v44, v64, s5
	v_cndmask_b32_e64 v0, v0, v122, s5
	v_cndmask_b32_e64 v41, v41, v45, s1
	v_cndmask_b32_e64 v43, v43, v120, s3
	v_cndmask_b32_e32 v33, v33, v35, vcc_lo
	v_cndmask_b32_e64 v42, v42, v121, s0
	v_cndmask_b32_e64 v17, v17, v18, s2
	v_cndmask_b32_e64 v45, v49, v56, s5
	v_add_co_u32 v0, s6, v44, v0
	v_cndmask_b32_e64 v43, v43, v119, s4
	v_cndmask_b32_e64 v41, v41, v46, s3
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v44, null, 0, 0, s6
	v_cndmask_b32_e64 v42, v42, v118, s1
	v_cndmask_b32_e32 v17, v17, v19, vcc_lo
	v_add_co_u32 v0, s6, v0, v45
	v_cndmask_b32_e64 v43, v43, v117, s5
	v_cndmask_b32_e64 v41, v41, v47, s4
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v34, null, 0, v44, s6
	v_cndmask_b32_e64 v42, v42, v116, s3
	v_cndmask_b32_e64 v35, v114, v115, s2
	v_cndmask_b32_e64 v17, v17, v20, s0
	v_add_co_u32 v0, s6, v0, v43
	v_cndmask_b32_e64 v41, v41, v48, s5
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v34, null, 0, v34, s6
	v_cndmask_b32_e64 v42, v42, v113, s4
	v_cndmask_b32_e64 v33, v33, v36, s0
	v_cndmask_b32_e64 v25, v25, v26, s2
	v_cndmask_b32_e64 v17, v17, v21, s1
	v_cndmask_b32_e64 v21, v97, v98, s2
	v_cndmask_b32_e32 v35, v35, v112, vcc_lo
	v_add_co_u32 v0, s6, v0, v41
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v26, null, 0, v34, s6
	v_cndmask_b32_e64 v34, v42, v111, s5
	v_cndmask_b32_e64 v33, v33, v37, s1
	v_cndmask_b32_e64 v35, v35, v110, s0
	v_cndmask_b32_e64 v1, v1, v2, s2
	v_cndmask_b32_e64 v2, v81, v82, s2
	v_cndmask_b32_e32 v25, v25, v27, vcc_lo
	v_cndmask_b32_e64 v27, v108, v109, s2
	v_add_co_u32 v0, s6, v0, v34
	v_cndmask_b32_e64 v33, v33, v38, s3
	v_cndmask_b32_e64 v34, v35, v107, s1
	v_cndmask_b32_e64 v9, v9, v10, s2
	v_cndmask_b32_e64 v10, v90, v91, s2
	v_cndmask_b32_e32 v27, v27, v106, vcc_lo
	v_cndmask_b32_e32 v2, v2, v78, vcc_lo
	v_cndmask_b32_e64 v25, v25, v28, s0
	v_cndmask_b32_e64 v18, v33, v39, s4
	v_cndmask_b32_e64 v28, v34, v105, s3
	v_cndmask_b32_e32 v10, v10, v87, vcc_lo
	v_cndmask_b32_e64 v27, v27, v104, s0
	v_cndmask_b32_e64 v25, v25, v29, s1
	v_cndmask_b32_e64 v18, v18, v40, s5
	v_cndmask_b32_e64 v19, v28, v103, s4
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v26, null, 0, v26, s6
	v_cndmask_b32_e64 v25, v25, v30, s3
	v_cndmask_b32_e64 v27, v27, v102, s1
	v_add_co_u32 v0, s6, v0, v18
	v_cndmask_b32_e64 v19, v19, v101, s5
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cndmask_b32_e64 v20, v25, v31, s4
	v_cndmask_b32_e64 v25, v27, v100, s3
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v18, null, 0, v26, s6
	v_add_co_u32 v0, s6, v0, v19
	v_cndmask_b32_e64 v19, v20, v32, s5
	v_cndmask_b32_e64 v20, v25, v99, s4
	v_cndmask_b32_e64 v17, v17, v22, s3
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v18, null, 0, v18, s6
	v_add_co_u32 v0, s6, v0, v19
	v_cndmask_b32_e64 v19, v20, v96, s5
	v_cndmask_b32_e64 v17, v17, v23, s4
	v_cndmask_b32_e32 v20, v21, v94, vcc_lo
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v18, null, 0, v18, s6
	v_add_co_u32 v0, s6, v0, v19
	v_cndmask_b32_e64 v17, v17, v24, s5
	v_cndmask_b32_e64 v19, v20, v88, s0
	v_cndmask_b32_e32 v9, v9, v11, vcc_lo
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v18, null, 0, v18, s6
	v_add_co_u32 v0, s6, v0, v17
	v_cndmask_b32_e64 v17, v19, v86, s1
	v_cndmask_b32_e64 v9, v9, v12, s0
	v_cndmask_b32_e64 v10, v10, v85, s0
	v_cndmask_b32_e32 v1, v1, v3, vcc_lo
	v_cndmask_b32_e64 v2, v2, v75, s0
	v_cndmask_b32_e64 v12, v17, v84, s3
	v_cndmask_b32_e64 v9, v9, v13, s1
	v_cndmask_b32_e64 v10, v10, v83, s1
	v_cndmask_b32_e64 v1, v1, v4, s0
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v11, null, 0, v18, s6
	v_cndmask_b32_e64 v3, v12, v80, s4
	v_cndmask_b32_e64 v9, v9, v14, s3
	v_cndmask_b32_e64 v10, v10, v79, s3
	v_cndmask_b32_e64 v1, v1, v5, s1
	v_cndmask_b32_e64 v2, v2, v95, s1
	v_cndmask_b32_e64 v3, v3, v77, s5
	v_cndmask_b32_e64 v4, v9, v15, s4
	v_cndmask_b32_e64 v9, v10, v76, s4
	v_cndmask_b32_e64 v1, v1, v6, s3
	v_cndmask_b32_e64 v2, v2, v93, s3
	v_add_co_u32 v0, vcc_lo, v0, v3
	v_cndmask_b32_e64 v4, v4, v16, s5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, 0, v11, vcc_lo
	v_cndmask_b32_e64 v5, v9, v74, s5
	v_cndmask_b32_e64 v1, v1, v7, s4
	v_add_co_u32 v0, vcc_lo, v0, v4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, 0, v3, vcc_lo
	v_cndmask_b32_e64 v2, v2, v92, s4
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_u32 v0, vcc_lo, v0, v5
	v_cndmask_b32_e64 v1, v1, v8, s5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, 0, v3, vcc_lo
	v_cndmask_b32_e64 v2, v2, v89, s5
	s_mul_i32 s0, s9, ttmp7
	v_add_co_u32 v0, vcc_lo, v0, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, 0, v3, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s0, s0, ttmp9
	v_add_co_u32 v0, vcc_lo, v0, v2
	s_wait_alu depctr_sa_sdst(0)
	s_ashr_i32 s1, s0, 31
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, 0, v1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b64 s[0:1], s[0:1], 11
	s_wait_kmcnt 0x0
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[0:1], s[10:11], s[0:1]
	global_store_b64 v73, v[0:1], s[0:1]
	s_nop 0
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
.Lfunc_end4:
	.size	_Z10iu4_ladderILi4EEvPKDv2_jS2_PKjPfPyiii, .Lfunc_end4-_Z10iu4_ladderILi4EEvPKDv2_jS2_PKjPfPyiii
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z10iu4_ladderILi4EEvPKDv2_jS2_PKjPfPyiii
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 52
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
		.amdhsa_next_free_vgpr 180
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
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end4-_Z10iu4_ladderILi4EEvPKDv2_jS2_PKjPfPyiii)<<4)&4080)>>4
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._Z10iu4_ladderILi4EEvPKDv2_jS2_PKjPfPyiii,"axG",@progbits,_Z10iu4_ladderILi4EEvPKDv2_jS2_PKjPfPyiii,comdat
                                        ; -- End function
	.set .L_Z10iu4_ladderILi4EEvPKDv2_jS2_PKjPfPyiii.num_vgpr, 180
	.set .L_Z10iu4_ladderILi4EEvPKDv2_jS2_PKjPfPyiii.num_agpr, 0
	.set .L_Z10iu4_ladderILi4EEvPKDv2_jS2_PKjPfPyiii.numbered_sgpr, 28
	.set .L_Z10iu4_ladderILi4EEvPKDv2_jS2_PKjPfPyiii.num_named_barrier, 0
	.set .L_Z10iu4_ladderILi4EEvPKDv2_jS2_PKjPfPyiii.private_seg_size, 0
	.set .L_Z10iu4_ladderILi4EEvPKDv2_jS2_PKjPfPyiii.uses_vcc, 1
	.set .L_Z10iu4_ladderILi4EEvPKDv2_jS2_PKjPfPyiii.uses_flat_scratch, 0
	.set .L_Z10iu4_ladderILi4EEvPKDv2_jS2_PKjPfPyiii.has_dyn_sized_stack, 0
	.set .L_Z10iu4_ladderILi4EEvPKDv2_jS2_PKjPfPyiii.has_recursion, 0
	.set .L_Z10iu4_ladderILi4EEvPKDv2_jS2_PKjPfPyiii.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 5292
; TotalNumSgprs: 30
; NumVgprs: 180
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 22
; NumSGPRsForWavesPerEU: 30
; NumVGPRsForWavesPerEU: 180
; Occupancy: 8
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.section	.text._Z10iu4_ladderILi5EEvPKDv2_jS2_PKjPfPyiii,"axG",@progbits,_Z10iu4_ladderILi5EEvPKDv2_jS2_PKjPfPyiii,comdat
	.protected	_Z10iu4_ladderILi5EEvPKDv2_jS2_PKjPfPyiii ; -- Begin function _Z10iu4_ladderILi5EEvPKDv2_jS2_PKjPfPyiii
	.globl	_Z10iu4_ladderILi5EEvPKDv2_jS2_PKjPfPyiii
	.p2align	8
	.type	_Z10iu4_ladderILi5EEvPKDv2_jS2_PKjPfPyiii,@function
_Z10iu4_ladderILi5EEvPKDv2_jS2_PKjPfPyiii: ; @_Z10iu4_ladderILi5EEvPKDv2_jS2_PKjPfPyiii
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_clause 0x1
	s_load_b128 s[8:11], s[0:1], 0x20
	s_load_b256 s[0:7], s[0:1], 0x0
	s_mov_b32 s20, ttmp7
	s_ashr_i32 s21, ttmp7, 31
	s_mov_b32 s16, ttmp9
	s_ashr_i32 s17, ttmp9, 31
	v_lshlrev_b32_e32 v67, 2, v0
	v_lshlrev_b32_e32 v9, 4, v0
	v_and_b32_e32 v139, 15, v0
	v_lshrrev_b32_e32 v140, 1, v0
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_add_nc_u32_e32 v142, 0, v67
	v_add_nc_u32_e32 v141, 0, v9
	s_wait_kmcnt 0x0
	s_ashr_i32 s23, s11, 31
	s_mov_b32 s22, s11
	s_lshl_b32 s12, s10, 1
	s_ashr_i32 s19, s10, 31
	s_mov_b32 s18, s10
	s_add_nc_u64 s[22:23], s[22:23], s[20:21]
	s_ashr_i32 s13, s12, 31
	s_mul_u64 s[14:15], s[18:19], s[16:17]
	s_mul_u64 s[18:19], s[22:23], s[18:19]
	s_mul_u64 s[16:17], s[12:13], s[16:17]
	s_mul_u64 s[20:21], s[12:13], s[20:21]
	s_lshl_b64 s[24:25], s[14:15], 10
	s_lshl_b64 s[22:23], s[18:19], 10
	s_lshl_b64 s[26:27], s[16:17], 12
	s_lshl_b64 s[28:29], s[20:21], 12
	s_add_nc_u64 s[24:25], s[4:5], s[24:25]
	s_add_nc_u64 s[22:23], s[4:5], s[22:23]
	s_add_nc_u64 s[26:27], s[0:1], s[26:27]
	s_add_nc_u64 s[28:29], s[2:3], s[28:29]
	s_clause 0x1
	global_load_b32 v10, v67, s[24:25]
	global_load_b32 v11, v67, s[22:23]
	s_clause 0x1
	global_load_b128 v[1:4], v9, s[26:27]
	global_load_b128 v[5:8], v9, s[28:29]
	s_mov_b32 s13, 0
	s_cmp_gt_i32 s10, 0
	s_wait_loadcnt 0x1
	ds_store_b128 v141, v[1:4]
	s_wait_loadcnt 0x0
	ds_store_b128 v141, v[5:8] offset:8192
	ds_store_2addr_stride64_b32 v142, v10, v11 offset0:64 offset1:68
	v_mov_b32_e32 v8, 0
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_cbranch_scc1 .LBB5_2
; %bb.1:                                ; %.._crit_edge_crit_edge
	v_and_b32_e32 v65, 15, v0
	v_lshrrev_b32_e32 v66, 1, v0
	s_branch .LBB5_3
.LBB5_2:
	s_mov_b32 s13, -1
                                        ; implicit-def: $vgpr65
                                        ; implicit-def: $vgpr66
.LBB5_3:                                ; %Flow1435
	v_lshrrev_b32_e32 v138, 5, v0
	v_dual_mov_b32 v6, 0 :: v_dual_lshlrev_b32 v73, 3, v0
	v_dual_mov_b32 v7, 0 :: v_dual_mov_b32 v4, 0
	v_dual_mov_b32 v5, 0 :: v_dual_mov_b32 v2, 0
	v_dual_mov_b32 v3, 0 :: v_dual_mov_b32 v16, 0
	v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v14, 0
	v_dual_mov_b32 v15, 0 :: v_dual_mov_b32 v12, 0
	v_dual_mov_b32 v13, 0 :: v_dual_mov_b32 v10, 0
	v_dual_mov_b32 v11, 0 :: v_dual_mov_b32 v24, 0
	v_dual_mov_b32 v9, 0 :: v_dual_mov_b32 v22, 0
	v_dual_mov_b32 v23, 0 :: v_dual_mov_b32 v20, 0
	v_dual_mov_b32 v21, 0 :: v_dual_mov_b32 v18, 0
	v_dual_mov_b32 v19, 0 :: v_dual_mov_b32 v32, 0
	v_dual_mov_b32 v17, 0 :: v_dual_mov_b32 v30, 0
	v_dual_mov_b32 v31, 0 :: v_dual_mov_b32 v28, 0
	v_dual_mov_b32 v29, 0 :: v_dual_mov_b32 v26, 0
	v_dual_mov_b32 v27, 0 :: v_dual_mov_b32 v40, 0
	v_dual_mov_b32 v25, 0 :: v_dual_mov_b32 v38, 0
	v_dual_mov_b32 v39, 0 :: v_dual_mov_b32 v36, 0
	v_dual_mov_b32 v37, 0 :: v_dual_mov_b32 v34, 0
	v_dual_mov_b32 v35, 0 :: v_dual_mov_b32 v48, 0
	v_dual_mov_b32 v33, 0 :: v_dual_mov_b32 v46, 0
	v_dual_mov_b32 v47, 0 :: v_dual_mov_b32 v44, 0
	v_dual_mov_b32 v45, 0 :: v_dual_mov_b32 v42, 0
	v_dual_mov_b32 v43, 0 :: v_dual_mov_b32 v56, 0
	v_dual_mov_b32 v41, 0 :: v_dual_mov_b32 v54, 0
	v_dual_mov_b32 v55, 0 :: v_dual_mov_b32 v52, 0
	v_dual_mov_b32 v53, 0 :: v_dual_mov_b32 v50, 0
	v_dual_mov_b32 v51, 0 :: v_dual_mov_b32 v64, 0
	v_dual_mov_b32 v49, 0 :: v_dual_mov_b32 v62, 0
	v_dual_mov_b32 v63, 0 :: v_dual_mov_b32 v60, 0
	v_dual_mov_b32 v61, 0 :: v_dual_mov_b32 v58, 0
	v_dual_mov_b32 v59, 0 :: v_dual_mov_b32 v80, 0
	v_dual_mov_b32 v57, 0 :: v_dual_mov_b32 v112, 0
	v_dual_mov_b32 v103, 0 :: v_dual_mov_b32 v136, 0
	v_dual_mov_b32 v123, 0 :: v_dual_mov_b32 v120, 0
	v_dual_mov_b32 v135, 0 :: v_dual_mov_b32 v74, 0
	v_dual_mov_b32 v137, 0 :: v_dual_mov_b32 v82, 0
	v_dual_mov_b32 v89, 0 :: v_dual_mov_b32 v96, 0
	v_dual_mov_b32 v104, 0 :: v_dual_mov_b32 v111, 0
	v_dual_mov_b32 v122, 0 :: v_dual_mov_b32 v121, 0
	v_dual_mov_b32 v75, 0 :: v_dual_mov_b32 v90, 0
	v_dual_mov_b32 v83, 0 :: v_dual_mov_b32 v124, 0
	v_dual_mov_b32 v97, 0 :: v_dual_mov_b32 v76, 0
	v_dual_mov_b32 v105, 0 :: v_dual_mov_b32 v84, 0
	v_dual_mov_b32 v113, 0 :: v_dual_mov_b32 v98, 0
	v_dual_mov_b32 v125, 0 :: v_dual_mov_b32 v106, 0
	v_dual_mov_b32 v91, 0 :: v_dual_mov_b32 v114, 0
	v_dual_mov_b32 v127, 0 :: v_dual_mov_b32 v126, 0
	v_dual_mov_b32 v77, 0 :: v_dual_mov_b32 v92, 0
	v_dual_mov_b32 v85, 0 :: v_dual_mov_b32 v128, 0
	v_dual_mov_b32 v99, 0 :: v_dual_mov_b32 v78, 0
	v_dual_mov_b32 v107, 0 :: v_dual_mov_b32 v86, 0
	v_dual_mov_b32 v115, 0 :: v_dual_mov_b32 v100, 0
	v_dual_mov_b32 v129, 0 :: v_dual_mov_b32 v108, 0
	v_dual_mov_b32 v93, 0 :: v_dual_mov_b32 v116, 0
	v_dual_mov_b32 v131, 0 :: v_dual_mov_b32 v130, 0
	v_dual_mov_b32 v79, 0 :: v_dual_mov_b32 v94, 0
	v_dual_mov_b32 v87, 0 :: v_dual_mov_b32 v118, 0
	v_dual_mov_b32 v101, 0 :: v_dual_mov_b32 v132, 0
	v_dual_mov_b32 v109, 0 :: v_dual_mov_b32 v88, 0
	v_dual_mov_b32 v133, 0 :: v_dual_mov_b32 v102, 0
	v_dual_mov_b32 v81, 0 :: v_dual_mov_b32 v110, 0
	v_dual_mov_b32 v95, 0 :: v_dual_mov_b32 v134, 0
	v_mov_b32_e32 v119, 0
	v_mov_b32_e32 v117, 0
	s_and_not1_b32 vcc_lo, exec_lo, s13
	s_cbranch_vccnz .LBB5_19
; %bb.4:                                ; %.lr.ph
	v_dual_mov_b32 v117, 0 :: v_dual_and_b32 v2, 0xf8, v73
	v_dual_mov_b32 v134, 0 :: v_dual_lshlrev_b32 v1, 1, v0
	v_dual_mov_b32 v110, 0 :: v_dual_lshlrev_b32 v3, 11, v138
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_dual_mov_b32 v102, 0 :: v_dual_add_nc_u32 v145, 0, v2
	v_dual_mov_b32 v95, 0 :: v_dual_lshlrev_b32 v2, 6, v138
	v_dual_mov_b32 v94, 0 :: v_dual_lshlrev_b32 v1, 3, v1
	v_dual_mov_b32 v119, 0 :: v_dual_lshlrev_b32 v4, 9, v138
	v_dual_mov_b32 v88, 0 :: v_dual_and_b32 v3, 0x800, v3
	v_dual_mov_b32 v132, 0 :: v_dual_lshlrev_b32 v5, 3, v140
	v_and_or_b32 v2, v2, 64, v139
	v_add_co_u32 v153, s0, s0, v1
	v_add_co_u32 v143, s4, s4, v67
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v154, null, s1, 0, s0
	v_add_co_u32 v155, s0, s2, v1
	v_add_co_ci_u32_e64 v144, null, s5, 0, s4
	v_dual_mov_b32 v81, 0 :: v_dual_and_b32 v146, 0xc00, v4
	v_or_b32_e32 v147, 0x200, v4
	v_or_b32_e32 v148, 0x300, v4
	v_dual_mov_b32 v118, 0 :: v_dual_add_nc_u32 v149, v145, v3
	v_dual_mov_b32 v133, 0 :: v_dual_and_b32 v150, 0x340, v5
	v_dual_mov_b32 v130, 0 :: v_dual_lshlrev_b32 v151, 3, v2
	v_or_b32_e32 v152, 0xb8, v5
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v156, null, s3, 0, s0
	v_dual_mov_b32 v109, 0 :: v_dual_mov_b32 v116, 0
	v_dual_mov_b32 v101, 0 :: v_dual_mov_b32 v108, 0
	v_dual_mov_b32 v87, 0 :: v_dual_mov_b32 v100, 0
	v_dual_mov_b32 v79, 0 :: v_dual_mov_b32 v86, 0
	v_dual_mov_b32 v131, 0 :: v_dual_mov_b32 v78, 0
	v_dual_mov_b32 v93, 0 :: v_dual_mov_b32 v128, 0
	v_dual_mov_b32 v129, 0 :: v_dual_mov_b32 v92, 0
	v_dual_mov_b32 v115, 0 :: v_dual_mov_b32 v126, 0
	v_dual_mov_b32 v107, 0 :: v_dual_mov_b32 v114, 0
	v_dual_mov_b32 v99, 0 :: v_dual_mov_b32 v106, 0
	v_dual_mov_b32 v85, 0 :: v_dual_mov_b32 v98, 0
	v_dual_mov_b32 v77, 0 :: v_dual_mov_b32 v84, 0
	v_dual_mov_b32 v127, 0 :: v_dual_mov_b32 v76, 0
	v_dual_mov_b32 v91, 0 :: v_dual_mov_b32 v124, 0
	v_dual_mov_b32 v125, 0 :: v_dual_mov_b32 v90, 0
	v_dual_mov_b32 v113, 0 :: v_dual_mov_b32 v122, 0
	v_dual_mov_b32 v105, 0 :: v_dual_mov_b32 v104, 0
	v_dual_mov_b32 v97, 0 :: v_dual_mov_b32 v96, 0
	v_dual_mov_b32 v83, 0 :: v_dual_mov_b32 v82, 0
	v_dual_mov_b32 v75, 0 :: v_dual_mov_b32 v74, 0
	v_dual_mov_b32 v121, 0 :: v_dual_mov_b32 v120, 0
	v_dual_mov_b32 v111, 0 :: v_dual_mov_b32 v58, v117
	v_dual_mov_b32 v89, 0 :: v_dual_mov_b32 v60, v117
	v_dual_mov_b32 v57, 0 :: v_dual_mov_b32 v62, v117
	v_dual_mov_b32 v59, v117 :: v_dual_mov_b32 v64, v117
	v_dual_mov_b32 v61, v117 :: v_dual_mov_b32 v50, v117
	v_dual_mov_b32 v63, v117 :: v_dual_mov_b32 v52, v117
	v_dual_mov_b32 v49, 0 :: v_dual_mov_b32 v54, v117
	v_dual_mov_b32 v51, v117 :: v_dual_mov_b32 v56, v117
	v_dual_mov_b32 v53, v117 :: v_dual_mov_b32 v42, v117
	v_dual_mov_b32 v55, v117 :: v_dual_mov_b32 v44, v117
	v_dual_mov_b32 v41, 0 :: v_dual_mov_b32 v46, v117
	v_dual_mov_b32 v43, v117 :: v_dual_mov_b32 v48, v117
	v_dual_mov_b32 v45, v117 :: v_dual_mov_b32 v34, v117
	v_dual_mov_b32 v47, v117 :: v_dual_mov_b32 v36, v117
	v_dual_mov_b32 v33, 0 :: v_dual_mov_b32 v38, v117
	v_dual_mov_b32 v35, v117 :: v_dual_mov_b32 v40, v117
	v_dual_mov_b32 v37, v117 :: v_dual_mov_b32 v26, v117
	v_dual_mov_b32 v39, v117 :: v_dual_mov_b32 v28, v117
	v_dual_mov_b32 v25, 0 :: v_dual_mov_b32 v30, v117
	v_dual_mov_b32 v27, v117 :: v_dual_mov_b32 v32, v117
	v_dual_mov_b32 v29, v117 :: v_dual_mov_b32 v18, v117
	v_dual_mov_b32 v31, v117 :: v_dual_mov_b32 v20, v117
	v_dual_mov_b32 v17, 0 :: v_dual_mov_b32 v22, v117
	v_dual_mov_b32 v19, v117 :: v_dual_mov_b32 v24, v117
	v_dual_mov_b32 v21, v117 :: v_dual_mov_b32 v10, v117
	v_dual_mov_b32 v23, v117 :: v_dual_mov_b32 v12, v117
	v_dual_mov_b32 v9, 0 :: v_dual_mov_b32 v14, v117
	v_dual_mov_b32 v11, v117 :: v_dual_mov_b32 v16, v117
	v_dual_mov_b32 v13, v117 :: v_dual_mov_b32 v2, v117
	v_dual_mov_b32 v15, v117 :: v_dual_mov_b32 v4, v117
	v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v6, v117
	v_dual_mov_b32 v3, v117 :: v_dual_mov_b32 v8, v117
	v_dual_mov_b32 v5, v117 :: v_dual_mov_b32 v136, 0
	v_dual_mov_b32 v7, v117 :: v_dual_mov_b32 v112, 0
	v_dual_mov_b32 v137, 0 :: v_dual_mov_b32 v80, 0
	v_mov_b32_e32 v135, 0
	v_mov_b32_e32 v123, 0
	v_mov_b32_e32 v103, 0
	s_mov_b32 s5, 0
	s_movk_i32 s2, 0x2000
	s_movk_i32 s3, 0x4000
	s_mov_b32 s10, 0
	s_wait_alu depctr_sa_sdst(0)
	s_mov_b32 s4, s5
	s_branch .LBB5_6
.LBB5_5:                                ;   in Loop: Header=BB5_6 Depth=1
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_xor_b32 s10, s10, 1
	s_mov_b32 s4, s13
	s_cmp_eq_u32 s13, s12
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_cbranch_scc1 .LBB5_18
.LBB5_6:                                ; =>This Inner Loop Header: Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s13, s4, 1
	s_mov_b32 s0, -1
	s_cmp_lt_i32 s13, s12
                                        ; implicit-def: $sgpr23
	s_cselect_b32 s22, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 vcc_lo, exec_lo, s22
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB5_8
; %bb.7:                                ; %._crit_edge391
                                        ;   in Loop: Header=BB5_6 Depth=1
	s_bitcmp1_b32 s4, 0
	s_mov_b32 s0, 0
	s_cselect_b32 s23, -1, 0
.LBB5_8:                                ; %Flow1432
                                        ;   in Loop: Header=BB5_6 Depth=1
	v_dual_mov_b32 v157, 0 :: v_dual_mov_b32 v158, 0
	v_dual_mov_b32 v65, 0 :: v_dual_mov_b32 v66, 0
	v_dual_mov_b32 v67, 0 :: v_dual_mov_b32 v68, 0
	v_dual_mov_b32 v69, 0 :: v_dual_mov_b32 v70, 0
	v_dual_mov_b32 v71, 0 :: v_dual_mov_b32 v72, 0
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s0
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB5_13
; %bb.9:                                ;   in Loop: Header=BB5_6 Depth=1
	s_bitcmp1_b32 s4, 0
	s_cselect_b32 s23, -1, 0
	s_add_co_i32 s0, s4, 2
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s0, s12
	s_cselect_b32 s0, -1, 0
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 s0, s23, s0
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s0
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB5_11
; %bb.10:                               ;   in Loop: Header=BB5_6 Depth=1
	s_lshr_b32 s0, s4, 1
	s_mov_b32 s1, s5
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s0, s0, 1
	s_mov_b32 s23, -1
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[24:25], s[14:15], s[0:1]
	s_add_nc_u64 s[0:1], s[18:19], s[0:1]
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b64 s[24:25], s[24:25], 10
	s_lshl_b64 s[0:1], s[0:1], 10
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v65, vcc_lo, v143, s24
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v66, null, s25, v144, vcc_lo
	v_add_co_u32 v67, vcc_lo, v143, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v68, null, s1, v144, vcc_lo
	s_clause 0x1
	global_load_b32 v158, v[65:66], off
	global_load_b32 v157, v[67:68], off
	s_branch .LBB5_12
.LBB5_11:                               ;   in Loop: Header=BB5_6 Depth=1
	v_dual_mov_b32 v158, 0 :: v_dual_mov_b32 v157, 0
.LBB5_12:                               ; %Flow1431
                                        ;   in Loop: Header=BB5_6 Depth=1
	s_add_nc_u64 s[0:1], s[16:17], s[4:5]
	s_add_nc_u64 s[24:25], s[20:21], s[4:5]
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b64 s[0:1], s[0:1], 12
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v65, vcc_lo, v153, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v66, null, s1, v154, vcc_lo
	s_lshl_b64 s[0:1], s[24:25], 12
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v69, vcc_lo, v155, s0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v70, null, s1, v156, vcc_lo
	global_load_b128 v[65:68], v[65:66], off offset:4096
	global_load_b128 v[69:72], v[69:70], off offset:4096
.LBB5_13:                               ;   in Loop: Header=BB5_6 Depth=1
	s_cmp_eq_u32 s10, 0
	s_cselect_b32 s0, 0, 0x1000
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v181, s0, v145
	s_cselect_b32 s0, s2, 0x3000
	s_and_not1_b32 vcc_lo, exec_lo, s23
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v175, s0, v149
	v_add_nc_u32_e32 v163, v181, v146
	v_add_nc_u32_e32 v179, v181, v147
	v_add_nc_u32_e32 v181, v181, v148
	ds_load_2addr_b64 v[159:162], v175 offset1:32
	ds_load_2addr_b64 v[163:166], v163 offset1:32
	ds_load_2addr_b64 v[167:170], v175 offset0:64 offset1:96
	ds_load_2addr_b64 v[171:174], v175 offset0:128 offset1:160
	ds_load_2addr_b64 v[175:178], v175 offset0:192 offset1:224
	ds_load_b64 v[179:180], v179
	ds_load_b64 v[181:182], v181
	s_wait_dscnt 0x5
	v_wmma_i32_16x16x32_iu4 v[57:64], v[163:164], v[159:160], v[57:64] neg_lo:[1,1,0]
	s_wait_dscnt 0x4
	v_wmma_i32_16x16x32_iu4 v[49:56], v[163:164], v[167:168], v[49:56] neg_lo:[1,1,0]
	s_wait_dscnt 0x3
	v_wmma_i32_16x16x32_iu4 v[41:48], v[163:164], v[171:172], v[41:48] neg_lo:[1,1,0]
	s_wait_dscnt 0x2
	v_wmma_i32_16x16x32_iu4 v[33:40], v[163:164], v[175:176], v[33:40] neg_lo:[1,1,0]
	s_wait_dscnt 0x1
	v_wmma_i32_16x16x32_iu4 v[25:32], v[179:180], v[159:160], v[25:32] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:24], v[179:180], v[167:168], v[17:24] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:16], v[179:180], v[171:172], v[9:16] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:8], v[179:180], v[175:176], v[1:8] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[57:64], v[165:166], v[161:162], v[57:64] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[49:56], v[165:166], v[169:170], v[49:56] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[41:48], v[165:166], v[173:174], v[41:48] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[33:40], v[165:166], v[177:178], v[33:40] neg_lo:[1,1,0]
	s_wait_dscnt 0x0
	v_wmma_i32_16x16x32_iu4 v[25:32], v[181:182], v[161:162], v[25:32] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:24], v[181:182], v[169:170], v[17:24] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:16], v[181:182], v[173:174], v[9:16] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[1:8], v[181:182], v[177:178], v[1:8] neg_lo:[1,1,0]
	s_cbranch_vccnz .LBB5_15
; %bb.14:                               ; %.preheader266
                                        ;   in Loop: Header=BB5_6 Depth=1
	s_bitcmp0_b32 s4, 1
	s_mov_b64 s[0:1], src_shared_base
	s_cselect_b32 s24, s3, 0x4800
	v_cvt_f32_i32_e32 v57, v57
	s_wait_alu depctr_sa_sdst(0)
	v_dual_mov_b32 v160, s1 :: v_dual_add_nc_u32 v159, s24, v150
	v_cvt_f32_i32_e32 v50, v50
	v_add_nc_u32_e32 v170, s24, v151
	v_cvt_f32_i32_e32 v59, v59
	v_mov_b32_e32 v161, s1
	flat_load_b32 v162, v[159:160] scope:SCOPE_SYS
	s_wait_loadcnt 0x0
	v_add_nc_u32_e32 v160, 8, v159
	v_cvt_f32_i32_e32 v58, v58
	v_cvt_f32_i32_e32 v60, v60
	v_cvt_f32_i32_e32 v52, v52
	v_cvt_f32_i32_e32 v42, v42
	flat_load_b32 v163, v[160:161] scope:SCOPE_SYS
	s_wait_loadcnt 0x0
	v_add_nc_u32_e32 v160, 16, v159
	v_cvt_f32_i32_e32 v49, v49
	v_cvt_f32_i32_e32 v51, v51
	v_cvt_f32_i32_e32 v41, v41
	v_cvt_f32_i32_e32 v43, v43
	flat_load_b32 v164, v[160:161] scope:SCOPE_SYS
	s_wait_loadcnt 0x0
	v_add_nc_u32_e32 v160, 24, v159
	v_cvt_f32_i32_e32 v44, v44
	v_cvt_f32_i32_e32 v34, v34
	v_cvt_f32_i32_e32 v35, v35
	v_cvt_f32_i32_e32 v36, v36
	flat_load_b32 v165, v[160:161] scope:SCOPE_SYS
	s_wait_loadcnt 0x0
	v_add_nc_u32_e32 v160, 32, v159
	v_cvt_f32_i32_e32 v33, v33
	v_cvt_f32_i32_e32 v25, v25
	v_cvt_f32_i32_e32 v26, v26
	v_cvt_f32_i32_e32 v27, v27
	flat_load_b32 v166, v[160:161] scope:SCOPE_SYS
	s_wait_loadcnt 0x0
	v_add_nc_u32_e32 v160, 40, v159
	v_cvt_f32_i32_e32 v28, v28
	v_cvt_f32_i32_e32 v18, v18
	v_cvt_f32_i32_e32 v17, v17
	v_cvt_f32_i32_e32 v20, v20
	flat_load_b32 v167, v[160:161] scope:SCOPE_SYS
	s_wait_loadcnt 0x0
	v_add_nc_u32_e32 v160, 48, v159
	v_cvt_f32_i32_e32 v19, v19
	v_cvt_f32_i32_e32 v10, v10
	v_cvt_f32_i32_e32 v12, v12
	v_cvt_f32_i32_e32 v9, v9
	flat_load_b32 v168, v[160:161] scope:SCOPE_SYS
	s_wait_loadcnt 0x0
	v_add_nc_u32_e32 v160, 56, v159
	v_cvt_f32_i32_e32 v11, v11
	v_cvt_f32_i32_e32 v2, v2
	v_cvt_f32_i32_e32 v1, v1
	v_cvt_f32_i32_e32 v3, v3
	flat_load_b32 v169, v[160:161] scope:SCOPE_SYS
	s_wait_loadcnt 0x0
	v_add_nc_u32_e32 v160, 0x400, v170
	v_cvt_f32_i32_e32 v4, v4
	flat_load_b32 v171, v[160:161] scope:SCOPE_SYS
	s_wait_loadcnt_dscnt 0x0
	v_dual_mul_f32 v172, v171, v162 :: v_dual_mul_f32 v173, v171, v163
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v117, v172, v57
	v_dual_mul_f32 v57, v171, v164 :: v_dual_fmac_f32 v134, v173, v58
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_dual_mul_f32 v58, v171, v165 :: v_dual_fmac_f32 v119, v57, v59
	v_mul_f32_e32 v57, v171, v166
	v_cvt_f32_i32_e32 v59, v61
	v_dual_fmac_f32 v102, v57, v59 :: v_dual_mul_f32 v57, v171, v168
	v_cvt_f32_i32_e32 v59, v63
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v88, v57, v59 :: v_dual_add_nc_u32 v57, 0x480, v170
	v_fmac_f32_e32 v110, v58, v60
	v_mul_f32_e32 v58, v171, v167
	v_cvt_f32_i32_e32 v60, v62
	v_dual_fmac_f32 v95, v58, v60 :: v_dual_mul_f32 v58, v171, v169
	v_cvt_f32_i32_e32 v60, v64
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v81, v58, v60 :: v_dual_mov_b32 v58, s1
	flat_load_b32 v59, v[57:58] scope:SCOPE_SYS
	s_wait_loadcnt_dscnt 0x0
	v_mul_f32_e32 v61, v59, v163
	v_dual_fmac_f32 v133, v61, v50 :: v_dual_mul_f32 v50, v59, v165
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v109, v50, v52 :: v_dual_mul_f32 v50, v59, v167
	v_cvt_f32_i32_e32 v52, v54
	v_fmac_f32_e32 v94, v50, v52
	v_mul_f32_e32 v50, v59, v169
	v_cvt_f32_i32_e32 v52, v56
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_mul_f32 v60, v59, v162 :: v_dual_fmac_f32 v79, v50, v52
	v_dual_fmac_f32 v132, v60, v49 :: v_dual_mul_f32 v49, v59, v164
	v_mov_b32_e32 v50, s1
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v118, v49, v51 :: v_dual_mul_f32 v49, v59, v166
	v_cvt_f32_i32_e32 v51, v53
	v_fmac_f32_e32 v101, v49, v51
	v_mul_f32_e32 v49, v59, v168
	v_cvt_f32_i32_e32 v51, v55
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v87, v49, v51
	v_add_nc_u32_e32 v49, 0x500, v170
	flat_load_b32 v51, v[49:50] scope:SCOPE_SYS
	s_wait_loadcnt_dscnt 0x0
	v_dual_mul_f32 v52, v51, v162 :: v_dual_mul_f32 v53, v51, v163
	v_dual_fmac_f32 v130, v52, v41 :: v_dual_fmac_f32 v131, v53, v42
	v_dual_mul_f32 v41, v51, v164 :: v_dual_mul_f32 v42, v51, v165
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v116, v41, v43
	v_dual_fmac_f32 v108, v42, v44 :: v_dual_mul_f32 v41, v51, v166
	v_mul_f32_e32 v42, v51, v167
	v_cvt_f32_i32_e32 v43, v45
	v_cvt_f32_i32_e32 v44, v46
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_2)
	v_dual_fmac_f32 v100, v41, v43 :: v_dual_fmac_f32 v93, v42, v44
	v_dual_mul_f32 v41, v51, v168 :: v_dual_mul_f32 v42, v51, v169
	v_cvt_f32_i32_e32 v43, v47
	v_cvt_f32_i32_e32 v44, v48
	v_fmac_f32_e32 v86, v41, v43
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v78, v42, v44 :: v_dual_add_nc_u32 v41, 0x580, v170
	v_mov_b32_e32 v42, s1
	flat_load_b32 v43, v[41:42] scope:SCOPE_SYS
	s_wait_loadcnt_dscnt 0x0
	v_dual_mul_f32 v44, v43, v162 :: v_dual_mul_f32 v45, v43, v163
	v_dual_fmac_f32 v128, v44, v33 :: v_dual_fmac_f32 v129, v45, v34
	v_dual_mul_f32 v33, v43, v164 :: v_dual_mul_f32 v34, v43, v165
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v115, v33, v35
	v_fmac_f32_e32 v107, v34, v36
	v_dual_mul_f32 v33, v43, v166 :: v_dual_mul_f32 v34, v43, v167
	v_cvt_f32_i32_e32 v35, v37
	v_cvt_f32_i32_e32 v36, v38
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_2)
	v_dual_fmac_f32 v99, v33, v35 :: v_dual_fmac_f32 v92, v34, v36
	v_dual_mul_f32 v33, v43, v168 :: v_dual_mul_f32 v34, v43, v169
	v_cvt_f32_i32_e32 v35, v39
	v_cvt_f32_i32_e32 v36, v40
	v_fmac_f32_e32 v85, v33, v35
	s_delay_alu instid0(VALU_DEP_2)
	v_fmac_f32_e32 v77, v34, v36
	v_dual_mov_b32 v34, s1 :: v_dual_add_nc_u32 v33, 0x80, v159
	flat_load_b32 v35, v[33:34] scope:SCOPE_SYS
	s_wait_loadcnt 0x0
	v_add_nc_u32_e32 v33, 0x88, v159
	flat_load_b32 v36, v[33:34] scope:SCOPE_SYS
	s_wait_loadcnt 0x0
	v_add_nc_u32_e32 v33, 0x90, v159
	flat_load_b32 v37, v[33:34] scope:SCOPE_SYS
	s_wait_loadcnt 0x0
	v_add_nc_u32_e32 v33, 0x98, v159
	flat_load_b32 v38, v[33:34] scope:SCOPE_SYS
	s_wait_loadcnt 0x0
	v_add_nc_u32_e32 v33, 0xa0, v159
	flat_load_b32 v39, v[33:34] scope:SCOPE_SYS
	s_wait_loadcnt 0x0
	v_add_nc_u32_e32 v33, 0xa8, v159
	flat_load_b32 v40, v[33:34] scope:SCOPE_SYS
	s_wait_loadcnt 0x0
	v_add_nc_u32_e32 v33, 0xb0, v159
	flat_load_b32 v43, v[33:34] scope:SCOPE_SYS
	s_wait_loadcnt 0x0
	v_add_nc_u32_e32 v33, s24, v152
	flat_load_b32 v33, v[33:34] scope:SCOPE_SYS
	s_wait_loadcnt 0x0
	flat_load_b32 v34, v[160:161] scope:SCOPE_SYS
	s_wait_loadcnt_dscnt 0x0
	v_dual_mul_f32 v44, v34, v35 :: v_dual_mul_f32 v45, v34, v36
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v126, v44, v25 :: v_dual_fmac_f32 v127, v45, v26
	v_dual_mul_f32 v25, v34, v37 :: v_dual_mul_f32 v26, v34, v38
	v_fmac_f32_e32 v106, v26, v28
	v_mul_f32_e32 v26, v34, v40
	v_cvt_f32_i32_e32 v28, v30
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v114, v25, v27
	v_mul_f32_e32 v25, v34, v39
	v_cvt_f32_i32_e32 v27, v29
	v_fmac_f32_e32 v91, v26, v28
	v_mul_f32_e32 v26, v34, v33
	v_cvt_f32_i32_e32 v28, v32
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v98, v25, v27
	v_mul_f32_e32 v25, v34, v43
	v_cvt_f32_i32_e32 v27, v31
	v_fmac_f32_e32 v84, v25, v27
	flat_load_b32 v25, v[57:58] scope:SCOPE_SYS
	s_wait_loadcnt 0x0
	v_dual_fmac_f32 v76, v26, v28 :: v_dual_mov_b32 v57, 0
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_mov_b32 v58, v57 :: v_dual_mov_b32 v59, v57
	v_dual_mov_b32 v60, v57 :: v_dual_mov_b32 v61, v57
	v_dual_mov_b32 v62, v57 :: v_dual_mov_b32 v63, v57
	v_dual_mov_b32 v64, v57 :: v_dual_mov_b32 v51, v57
	v_dual_mov_b32 v52, v57 :: v_dual_mov_b32 v53, v57
	v_dual_mov_b32 v54, v57 :: v_dual_mov_b32 v55, v57
	v_mov_b32_e32 v56, v57
	v_dual_mov_b32 v44, v57 :: v_dual_mov_b32 v45, v57
	v_dual_mov_b32 v46, v57 :: v_dual_mov_b32 v47, v57
	v_mov_b32_e32 v48, v57
	v_mov_b32_e32 v34, v57
	v_dual_mov_b32 v28, v57 :: v_dual_mov_b32 v29, v57
	v_dual_mov_b32 v30, v57 :: v_dual_mov_b32 v31, v57
	v_mov_b32_e32 v32, v57
	s_wait_dscnt 0x0
	v_mul_f32_e32 v27, v25, v36
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v125, v27, v18
	v_mul_f32_e32 v18, v25, v38
	v_mul_f32_e32 v26, v25, v35
	v_mov_b32_e32 v27, v57
	v_fmac_f32_e32 v105, v18, v20
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v124, v26, v17
	v_dual_mul_f32 v17, v25, v37 :: v_dual_mul_f32 v18, v25, v40
	v_cvt_f32_i32_e32 v20, v22
	v_mov_b32_e32 v26, v57
	v_mov_b32_e32 v22, v57
	v_fmac_f32_e32 v113, v17, v19
	v_mul_f32_e32 v17, v25, v39
	v_cvt_f32_i32_e32 v19, v21
	v_mov_b32_e32 v21, v57
	s_delay_alu instid0(VALU_DEP_2)
	v_fmac_f32_e32 v97, v17, v19
	v_mul_f32_e32 v17, v25, v43
	v_cvt_f32_i32_e32 v19, v23
	v_fmac_f32_e32 v90, v18, v20
	v_cvt_f32_i32_e32 v20, v24
	v_dual_mov_b32 v23, v57 :: v_dual_mov_b32 v24, v57
	s_delay_alu instid0(VALU_DEP_4)
	v_fmac_f32_e32 v83, v17, v19
	flat_load_b32 v17, v[49:50] scope:SCOPE_SYS
	s_wait_loadcnt 0x0
	v_dual_mov_b32 v49, v57 :: v_dual_mov_b32 v50, v57
	s_wait_dscnt 0x0
	v_mul_f32_e32 v19, v17, v36
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v122, v19, v10
	v_mul_f32_e32 v10, v17, v38
	v_dual_mov_b32 v19, v57 :: v_dual_fmac_f32 v104, v10, v12
	v_mul_f32_e32 v10, v17, v40
	v_cvt_f32_i32_e32 v12, v14
	v_mul_f32_e32 v18, v25, v33
	v_dual_mov_b32 v25, v57 :: v_dual_mov_b32 v14, v57
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v89, v10, v12
	v_dual_fmac_f32 v75, v18, v20 :: v_dual_mul_f32 v18, v17, v35
	v_mul_f32_e32 v10, v17, v33
	v_cvt_f32_i32_e32 v12, v16
	v_mov_b32_e32 v20, v57
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mov_b32 v16, v57 :: v_dual_fmac_f32 v121, v18, v9
	v_dual_mul_f32 v9, v17, v37 :: v_dual_fmac_f32 v74, v10, v12
	v_mov_b32_e32 v18, v57
	v_mov_b32_e32 v12, v57
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v111, v9, v11
	v_mul_f32_e32 v9, v17, v39
	v_cvt_f32_i32_e32 v11, v13
	v_mov_b32_e32 v13, v57
	v_fmac_f32_e32 v96, v9, v11
	v_mul_f32_e32 v9, v17, v43
	v_cvt_f32_i32_e32 v11, v15
	v_mov_b32_e32 v17, v57
	v_mov_b32_e32 v15, v57
	s_delay_alu instid0(VALU_DEP_3)
	v_fmac_f32_e32 v82, v9, v11
	flat_load_b32 v9, v[41:42] scope:SCOPE_SYS
	s_wait_loadcnt 0x0
	v_dual_mov_b32 v41, v57 :: v_dual_mov_b32 v42, v57
	s_wait_dscnt 0x0
	v_dual_mul_f32 v10, v9, v35 :: v_dual_mul_f32 v11, v9, v36
	v_dual_mov_b32 v35, v57 :: v_dual_mov_b32 v36, v57
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v120, v10, v1 :: v_dual_fmac_f32 v137, v11, v2
	v_dual_mul_f32 v1, v9, v37 :: v_dual_mul_f32 v2, v9, v38
	v_dual_mov_b32 v37, v57 :: v_dual_mov_b32 v38, v57
	v_dual_mov_b32 v10, v57 :: v_dual_mov_b32 v11, v57
	v_dual_fmac_f32 v136, v1, v3 :: v_dual_fmac_f32 v135, v2, v4
	v_dual_mul_f32 v1, v9, v39 :: v_dual_mul_f32 v2, v9, v40
	v_cvt_f32_i32_e32 v3, v5
	v_cvt_f32_i32_e32 v4, v6
	v_dual_mov_b32 v39, v57 :: v_dual_mov_b32 v40, v57
	v_mov_b32_e32 v5, v57
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v123, v1, v3
	v_dual_mul_f32 v1, v9, v43 :: v_dual_fmac_f32 v112, v2, v4
	v_mul_f32_e32 v2, v9, v33
	v_cvt_f32_i32_e32 v3, v7
	v_cvt_f32_i32_e32 v4, v8
	v_mov_b32_e32 v43, v57
	v_mov_b32_e32 v33, v57
	v_mov_b32_e32 v9, v57
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_fmac_f32 v103, v1, v3 :: v_dual_fmac_f32 v80, v2, v4
	v_dual_mov_b32 v1, v57 :: v_dual_mov_b32 v2, v57
	v_dual_mov_b32 v3, v57 :: v_dual_mov_b32 v4, v57
	v_dual_mov_b32 v6, v57 :: v_dual_mov_b32 v7, v57
	v_mov_b32_e32 v8, v57
.LBB5_15:                               ; %.loopexit
                                        ;   in Loop: Header=BB5_6 Depth=1
	s_and_not1_b32 vcc_lo, exec_lo, s22
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB5_5
; %bb.16:                               ;   in Loop: Header=BB5_6 Depth=1
	s_cmp_eq_u32 s10, 1
	s_cselect_b32 s0, 0, 0x1000
	s_cselect_b32 s1, s2, 0x3000
	s_add_co_i32 s22, s4, 2
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v159, s0, v141
	s_cmp_lt_i32 s22, s12
	v_add_nc_u32_e32 v160, s1, v141
	s_cselect_b32 s22, -1, 0
	s_wait_loadcnt 0x1
	ds_store_b128 v159, v[65:68]
	s_wait_loadcnt 0x0
	ds_store_b128 v160, v[69:72]
	s_and_b32 s22, s23, s22
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s22
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB5_5
; %bb.17:                               ;   in Loop: Header=BB5_6 Depth=1
	s_bitcmp0_b32 s4, 1
	s_cselect_b32 s0, 0x4800, s3
	s_wait_alu depctr_sa_sdst(0)
	v_add_nc_u32_e32 v65, s0, v142
	ds_store_2addr_stride64_b32 v65, v158, v157 offset1:4
	s_branch .LBB5_5
.LBB5_18:                               ; %Flow1433
	v_dual_mov_b32 v65, v139 :: v_dual_mov_b32 v66, v140
.LBB5_19:                               ; %Flow1436
	s_mov_b64 s[0:1], src_shared_base
	v_mul_u32_u24_e32 v67, 0x500, v138
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_dual_mov_b32 v139, s1 :: v_dual_lshlrev_b32 v68, 2, v65
	v_dual_mov_b32 v141, s1 :: v_dual_and_b32 v66, 8, v66
	v_mov_b32_e32 v143, s1
	v_add3_u32 v67, 0, v67, v68
	v_mov_b32_e32 v68, s1
	v_lshrrev_b32_e32 v150, 4, v0
	s_mul_i32 s0, s11, ttmp7
	v_mov_b32_e32 v159, s1
	v_mad_u32_u24 v67, 0x50, v66, v67
	v_mov_b32_e32 v70, s1
	v_dual_mov_b32 v72, s1 :: v_dual_mov_b32 v145, s1
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_mov_b32 v147, s1 :: v_dual_add_nc_u32 v140, 0x140, v67
	v_add_nc_u32_e32 v69, 0x50, v67
	v_add_nc_u32_e32 v138, 0xf0, v67
	v_add_nc_u32_e32 v71, 0xa0, v67
	v_dual_mov_b32 v149, s1 :: v_dual_add_nc_u32 v142, 0x190, v67
	v_dual_mov_b32 v151, s1 :: v_dual_add_nc_u32 v144, 0x1e0, v67
	v_dual_mov_b32 v155, s1 :: v_dual_add_nc_u32 v146, 0x230, v67
	s_wait_loadcnt 0x0
	flat_store_b32 v[67:68], v117 scope:SCOPE_SYS
	s_wait_storecnt 0x0
	flat_store_b32 v[69:70], v134 scope:SCOPE_SYS
	s_wait_storecnt 0x0
	flat_store_b32 v[71:72], v119 scope:SCOPE_SYS
	s_wait_storecnt 0x0
	flat_store_b32 v[138:139], v110 scope:SCOPE_SYS
	s_wait_storecnt 0x0
	flat_store_b32 v[140:141], v102 scope:SCOPE_SYS
	s_wait_storecnt 0x0
	flat_store_b32 v[142:143], v95 scope:SCOPE_SYS
	s_wait_storecnt 0x0
	flat_store_b32 v[144:145], v88 scope:SCOPE_SYS
	s_wait_storecnt 0x0
	flat_store_b32 v[146:147], v81 scope:SCOPE_SYS
	s_wait_storecnt_dscnt 0x0
	s_barrier_signal -1
	v_lshl_add_u32 v66, v150, 2, 0
	s_wait_alu depctr_sa_sdst(0)
	s_add_co_i32 s2, s0, ttmp9
	v_lshlrev_b32_e32 v168, 9, v150
	s_wait_alu depctr_sa_sdst(0)
	s_ashr_i32 s3, s2, 31
	v_mov_b32_e32 v157, s1
	v_mad_u32_u24 v148, 0x50, v65, v66
	v_mov_b32_e32 v66, 0
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b64 s[4:5], s[2:3], 16
	v_mov_b32_e32 v161, s1
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[4:5], s[6:7], s[4:5]
	v_and_b32_e32 v0, 7, v0
	v_lshlrev_b64_e32 v[65:66], 2, v[65:66]
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v150, s0, s4, v168
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v153, null, s5, 0, s0
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	flat_load_b32 v154, v[148:149] scope:SCOPE_SYS
	s_wait_loadcnt 0x0
	v_add_co_u32 v152, vcc_lo, v150, v65
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v153, null, v153, v66, vcc_lo
	v_dual_mov_b32 v163, s1 :: v_dual_add_nc_u32 v150, 0x500, v148
	v_mov_b32_e32 v165, s1
	s_wait_dscnt 0x0
	global_store_b32 v[152:153], v154, off
	flat_load_b32 v156, v[150:151] scope:SCOPE_SYS
	s_wait_loadcnt 0x0
	v_add_nc_u32_e32 v154, 0xa00, v148
	s_wait_dscnt 0x0
	global_store_b32 v[152:153], v156, off offset:32768
	flat_load_b32 v158, v[154:155] scope:SCOPE_SYS
	s_wait_loadcnt 0x0
	v_add_nc_u32_e32 v156, 0xf00, v148
	s_wait_dscnt 0x0
	global_store_b32 v[152:153], v158, off offset:128
	flat_load_b32 v160, v[156:157] scope:SCOPE_SYS
	s_wait_loadcnt 0x0
	v_add_nc_u32_e32 v158, 0x1400, v148
	s_wait_dscnt 0x0
	global_store_b32 v[152:153], v160, off offset:32896
	flat_load_b32 v162, v[158:159] scope:SCOPE_SYS
	s_wait_loadcnt 0x0
	v_add_nc_u32_e32 v160, 0x1900, v148
	s_wait_dscnt 0x0
	global_store_b32 v[152:153], v162, off offset:256
	flat_load_b32 v164, v[160:161] scope:SCOPE_SYS
	s_wait_loadcnt 0x0
	v_add_nc_u32_e32 v162, 0x1e00, v148
	s_wait_dscnt 0x0
	global_store_b32 v[152:153], v164, off offset:33024
	flat_load_b32 v166, v[162:163] scope:SCOPE_SYS
	s_wait_loadcnt 0x0
	v_add_nc_u32_e32 v164, 0x2300, v148
	s_wait_dscnt 0x0
	global_store_b32 v[152:153], v166, off offset:384
	flat_load_b32 v166, v[164:165] scope:SCOPE_SYS
	s_wait_loadcnt_dscnt 0x0
	global_store_b32 v[152:153], v166, off offset:33152
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_loadcnt 0x0
	flat_store_b32 v[67:68], v132 scope:SCOPE_SYS
	s_wait_storecnt 0x0
	flat_store_b32 v[69:70], v133 scope:SCOPE_SYS
	s_wait_storecnt 0x0
	flat_store_b32 v[71:72], v118 scope:SCOPE_SYS
	s_wait_storecnt 0x0
	flat_store_b32 v[138:139], v109 scope:SCOPE_SYS
	s_wait_storecnt 0x0
	flat_store_b32 v[140:141], v101 scope:SCOPE_SYS
	s_wait_storecnt 0x0
	flat_store_b32 v[142:143], v94 scope:SCOPE_SYS
	s_wait_storecnt 0x0
	flat_store_b32 v[144:145], v87 scope:SCOPE_SYS
	s_wait_storecnt 0x0
	flat_store_b32 v[146:147], v79 scope:SCOPE_SYS
	s_wait_storecnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	flat_load_b32 v166, v[148:149] scope:SCOPE_SYS
	s_wait_loadcnt_dscnt 0x0
	global_store_b32 v[152:153], v166, off offset:8192
	flat_load_b32 v166, v[150:151] scope:SCOPE_SYS
	s_wait_loadcnt_dscnt 0x0
	global_store_b32 v[152:153], v166, off offset:40960
	flat_load_b32 v166, v[154:155] scope:SCOPE_SYS
	s_wait_loadcnt_dscnt 0x0
	global_store_b32 v[152:153], v166, off offset:8320
	flat_load_b32 v166, v[156:157] scope:SCOPE_SYS
	s_wait_loadcnt_dscnt 0x0
	global_store_b32 v[152:153], v166, off offset:41088
	flat_load_b32 v166, v[158:159] scope:SCOPE_SYS
	s_wait_loadcnt_dscnt 0x0
	global_store_b32 v[152:153], v166, off offset:8448
	flat_load_b32 v166, v[160:161] scope:SCOPE_SYS
	s_wait_loadcnt_dscnt 0x0
	global_store_b32 v[152:153], v166, off offset:41216
	flat_load_b32 v166, v[162:163] scope:SCOPE_SYS
	s_wait_loadcnt_dscnt 0x0
	global_store_b32 v[152:153], v166, off offset:8576
	flat_load_b32 v166, v[164:165] scope:SCOPE_SYS
	s_wait_loadcnt_dscnt 0x0
	global_store_b32 v[152:153], v166, off offset:41344
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_loadcnt 0x0
	flat_store_b32 v[67:68], v130 scope:SCOPE_SYS
	s_wait_storecnt 0x0
	flat_store_b32 v[69:70], v131 scope:SCOPE_SYS
	s_wait_storecnt 0x0
	flat_store_b32 v[71:72], v116 scope:SCOPE_SYS
	s_wait_storecnt 0x0
	flat_store_b32 v[138:139], v108 scope:SCOPE_SYS
	s_wait_storecnt 0x0
	flat_store_b32 v[140:141], v100 scope:SCOPE_SYS
	s_wait_storecnt 0x0
	flat_store_b32 v[142:143], v93 scope:SCOPE_SYS
	s_wait_storecnt 0x0
	flat_store_b32 v[144:145], v86 scope:SCOPE_SYS
	s_wait_storecnt 0x0
	flat_store_b32 v[146:147], v78 scope:SCOPE_SYS
	s_wait_storecnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	flat_load_b32 v166, v[148:149] scope:SCOPE_SYS
	s_wait_loadcnt_dscnt 0x0
	global_store_b32 v[152:153], v166, off offset:16384
	flat_load_b32 v166, v[150:151] scope:SCOPE_SYS
	s_wait_loadcnt_dscnt 0x0
	global_store_b32 v[152:153], v166, off offset:49152
	flat_load_b32 v166, v[154:155] scope:SCOPE_SYS
	s_wait_loadcnt_dscnt 0x0
	global_store_b32 v[152:153], v166, off offset:16512
	flat_load_b32 v166, v[156:157] scope:SCOPE_SYS
	s_wait_loadcnt_dscnt 0x0
	global_store_b32 v[152:153], v166, off offset:49280
	flat_load_b32 v166, v[158:159] scope:SCOPE_SYS
	s_wait_loadcnt_dscnt 0x0
	global_store_b32 v[152:153], v166, off offset:16640
	flat_load_b32 v166, v[160:161] scope:SCOPE_SYS
	s_wait_loadcnt_dscnt 0x0
	global_store_b32 v[152:153], v166, off offset:49408
	flat_load_b32 v166, v[162:163] scope:SCOPE_SYS
	s_wait_loadcnt_dscnt 0x0
	global_store_b32 v[152:153], v166, off offset:16768
	flat_load_b32 v166, v[164:165] scope:SCOPE_SYS
	s_wait_loadcnt_dscnt 0x0
	global_store_b32 v[152:153], v166, off offset:49536
	v_or_b32_e32 v166, 0x6000, v168
	v_or_b32_e32 v168, 0xe000, v168
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_add_co_u32 v166, s0, s4, v166
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v167, null, s5, 0, s0
	v_add_co_u32 v168, s0, s4, v168
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v166, vcc_lo, v166, v65
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v167, null, v167, v66, vcc_lo
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v170, null, s5, 0, s0
	v_add_co_u32 v65, vcc_lo, v168, v65
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v66, null, v170, v66, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, 1, v0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v57, v57, v58, vcc_lo
	v_cndmask_b32_e32 v49, v49, v50, vcc_lo
	v_cndmask_b32_e32 v41, v41, v42, vcc_lo
	v_dual_cndmask_b32 v33, v33, v34 :: v_dual_cndmask_b32 v50, v132, v133
	v_cndmask_b32_e32 v17, v17, v18, vcc_lo
	v_cndmask_b32_e32 v25, v25, v26, vcc_lo
	v_cmp_eq_u32_e64 s0, 2, v0
	v_dual_cndmask_b32 v42, v130, v131 :: v_dual_cndmask_b32 v9, v9, v10
	v_cndmask_b32_e32 v34, v128, v129, vcc_lo
	v_dual_cndmask_b32 v26, v126, v127 :: v_dual_cndmask_b32 v1, v1, v2
	v_cndmask_b32_e32 v18, v124, v125, vcc_lo
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v10, v57, v59, s0
	v_cndmask_b32_e32 v57, v121, v122, vcc_lo
	v_cndmask_b32_e64 v49, v49, v51, s0
	v_cndmask_b32_e32 v2, v120, v137, vcc_lo
	v_cndmask_b32_e64 v50, v50, v118, s0
	v_cndmask_b32_e32 v58, v117, v134, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, 3, v0
	v_cndmask_b32_e64 v25, v25, v27, s0
	v_cmp_eq_u32_e64 s1, 4, v0
	v_cndmask_b32_e64 v41, v41, v43, s0
	v_cndmask_b32_e64 v42, v42, v116, s0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v27, v50, v109, vcc_lo
	v_cndmask_b32_e64 v58, v58, v119, s0
	v_cndmask_b32_e64 v33, v33, v35, s0
	v_cndmask_b32_e64 v34, v34, v115, s0
	v_cndmask_b32_e64 v26, v26, v114, s0
	v_cndmask_b32_e64 v17, v17, v19, s0
	v_cndmask_b32_e64 v9, v9, v11, s0
	v_cndmask_b32_e64 v11, v57, v111, s0
	v_cndmask_b32_e64 v1, v1, v3, s0
	v_cndmask_b32_e32 v3, v58, v110, vcc_lo
	v_cndmask_b32_e64 v18, v18, v113, s0
	v_cndmask_b32_e32 v19, v49, v52, vcc_lo
	v_cndmask_b32_e64 v2, v2, v136, s0
	v_dual_cndmask_b32 v35, v41, v44 :: v_dual_cndmask_b32 v34, v34, v107
	v_cndmask_b32_e32 v41, v42, v108, vcc_lo
	v_dual_cndmask_b32 v33, v33, v36 :: v_dual_cndmask_b32 v26, v26, v106
	v_dual_cndmask_b32 v25, v25, v28 :: v_dual_cndmask_b32 v18, v18, v105
	v_dual_cndmask_b32 v17, v17, v20 :: v_dual_cndmask_b32 v2, v2, v135
	v_cndmask_b32_e32 v11, v11, v104, vcc_lo
	v_cndmask_b32_e32 v1, v1, v4, vcc_lo
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v3, v3, v102, s1
	v_cndmask_b32_e32 v9, v9, v12, vcc_lo
	v_cndmask_b32_e64 v12, v27, v101, s1
	v_cndmask_b32_e32 v10, v10, v60, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, 5, v0
	v_cndmask_b32_e64 v1, v1, v5, s1
	v_cndmask_b32_e64 v27, v33, v37, s1
	v_cndmask_b32_e64 v28, v34, v99, s1
	v_cndmask_b32_e64 v25, v25, v29, s1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v5, v12, v94, vcc_lo
	v_cndmask_b32_e64 v4, v10, v61, s1
	v_cndmask_b32_e64 v10, v19, v53, s1
	v_cndmask_b32_e64 v19, v35, v45, s1
	v_cndmask_b32_e64 v26, v26, v98, s1
	v_cndmask_b32_e64 v17, v17, v21, s1
	v_cndmask_b32_e32 v4, v4, v62, vcc_lo
	v_cndmask_b32_e64 v20, v41, v100, s1
	v_cndmask_b32_e64 v18, v18, v97, s1
	v_cndmask_b32_e64 v9, v9, v13, s1
	v_cndmask_b32_e64 v2, v2, v123, s1
	v_cmp_eq_u32_e64 s0, 6, v0
	v_cndmask_b32_e32 v12, v20, v93, vcc_lo
	v_cndmask_b32_e64 v11, v11, v96, s1
	v_cmp_eq_u32_e64 s1, 7, v0
	v_cndmask_b32_e32 v0, v3, v95, vcc_lo
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v4, v4, v63, s0
	v_cndmask_b32_e32 v3, v10, v54, vcc_lo
	v_cndmask_b32_e32 v10, v19, v46, vcc_lo
	v_dual_cndmask_b32 v13, v27, v38 :: v_dual_cndmask_b32 v2, v2, v112
	v_cndmask_b32_e64 v0, v0, v88, s0
	v_dual_cndmask_b32 v19, v28, v92 :: v_dual_cndmask_b32 v20, v25, v30
	v_cndmask_b32_e32 v21, v26, v91, vcc_lo
	v_cndmask_b32_e32 v17, v17, v22, vcc_lo
	v_dual_cndmask_b32 v18, v18, v90 :: v_dual_cndmask_b32 v11, v11, v89
	v_cndmask_b32_e32 v9, v9, v14, vcc_lo
	v_cndmask_b32_e32 v1, v1, v6, vcc_lo
	v_cndmask_b32_e64 v3, v3, v55, s0
	v_cndmask_b32_e64 v4, v4, v64, s1
	v_cndmask_b32_e64 v0, v0, v81, s1
	v_cndmask_b32_e64 v5, v5, v87, s0
	v_cndmask_b32_e64 v6, v10, v47, s0
	v_cndmask_b32_e64 v10, v12, v86, s0
	v_cndmask_b32_e64 v12, v13, v39, s0
	v_cndmask_b32_e64 v13, v19, v85, s0
	v_cndmask_b32_e64 v14, v20, v31, s0
	v_cndmask_b32_e64 v19, v21, v84, s0
	v_cndmask_b32_e64 v17, v17, v23, s0
	v_cndmask_b32_e64 v18, v18, v83, s0
	v_cndmask_b32_e64 v9, v9, v15, s0
	v_cndmask_b32_e64 v11, v11, v82, s0
	v_cndmask_b32_e64 v1, v1, v7, s0
	v_cndmask_b32_e64 v2, v2, v103, s0
	v_cndmask_b32_e64 v3, v3, v56, s1
	v_add_co_u32 v0, s0, v4, v0
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v4, null, 0, 0, s0
	v_cndmask_b32_e64 v5, v5, v79, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v0, vcc_lo, v0, v3
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, 0, v4, vcc_lo
	v_cndmask_b32_e64 v6, v6, v48, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v0, vcc_lo, v0, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, 0, v3, vcc_lo
	v_cndmask_b32_e64 v4, v10, v78, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v0, vcc_lo, v0, v6
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, 0, v3, vcc_lo
	v_cndmask_b32_e64 v5, v12, v40, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v0, vcc_lo, v0, v4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, 0, v3, vcc_lo
	v_cndmask_b32_e64 v6, v13, v77, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v0, vcc_lo, v0, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, 0, v3, vcc_lo
	v_cndmask_b32_e64 v4, v14, v32, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v0, vcc_lo, v0, v6
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, 0, v3, vcc_lo
	v_cndmask_b32_e64 v5, v19, v76, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v0, vcc_lo, v0, v4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, 0, v3, vcc_lo
	v_cndmask_b32_e64 v6, v17, v24, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v0, vcc_lo, v0, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, 0, v3, vcc_lo
	v_cndmask_b32_e64 v4, v18, v75, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v0, vcc_lo, v0, v6
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, 0, v3, vcc_lo
	v_cndmask_b32_e64 v5, v9, v16, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v0, vcc_lo, v0, v4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, 0, v3, vcc_lo
	v_cndmask_b32_e64 v6, v11, v74, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v0, vcc_lo, v0, v5
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, 0, v3, vcc_lo
	v_cndmask_b32_e64 v1, v1, v8, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v0, vcc_lo, v0, v6
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v3, null, 0, v3, vcc_lo
	v_cndmask_b32_e64 v2, v2, v80, s1
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v0, vcc_lo, v0, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, 0, v3, vcc_lo
	s_lshl_b64 s[0:1], s[2:3], 11
	v_add_co_u32 v0, vcc_lo, v0, v2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, 0, v1, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[0:1], s[8:9], s[0:1]
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_loadcnt 0x0
	flat_store_b32 v[67:68], v128 scope:SCOPE_SYS
	s_wait_storecnt 0x0
	flat_store_b32 v[69:70], v129 scope:SCOPE_SYS
	s_wait_storecnt 0x0
	flat_store_b32 v[71:72], v115 scope:SCOPE_SYS
	s_wait_storecnt 0x0
	flat_store_b32 v[138:139], v107 scope:SCOPE_SYS
	s_wait_storecnt 0x0
	flat_store_b32 v[140:141], v99 scope:SCOPE_SYS
	s_wait_storecnt 0x0
	flat_store_b32 v[142:143], v92 scope:SCOPE_SYS
	s_wait_storecnt 0x0
	flat_store_b32 v[144:145], v85 scope:SCOPE_SYS
	s_wait_storecnt 0x0
	flat_store_b32 v[146:147], v77 scope:SCOPE_SYS
	s_wait_storecnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	flat_load_b32 v169, v[148:149] scope:SCOPE_SYS
	s_wait_loadcnt_dscnt 0x0
	global_store_b32 v[166:167], v169, off
	flat_load_b32 v169, v[150:151] scope:SCOPE_SYS
	s_wait_loadcnt_dscnt 0x0
	global_store_b32 v[65:66], v169, off
	flat_load_b32 v168, v[154:155] scope:SCOPE_SYS
	s_wait_loadcnt_dscnt 0x0
	global_store_b32 v[166:167], v168, off offset:128
	flat_load_b32 v168, v[156:157] scope:SCOPE_SYS
	s_wait_loadcnt_dscnt 0x0
	global_store_b32 v[65:66], v168, off offset:128
	flat_load_b32 v168, v[158:159] scope:SCOPE_SYS
	s_wait_loadcnt_dscnt 0x0
	global_store_b32 v[166:167], v168, off offset:256
	flat_load_b32 v168, v[160:161] scope:SCOPE_SYS
	s_wait_loadcnt_dscnt 0x0
	global_store_b32 v[65:66], v168, off offset:256
	flat_load_b32 v168, v[162:163] scope:SCOPE_SYS
	s_wait_loadcnt_dscnt 0x0
	global_store_b32 v[166:167], v168, off offset:384
	flat_load_b32 v168, v[164:165] scope:SCOPE_SYS
	s_wait_loadcnt_dscnt 0x0
	global_store_b32 v[65:66], v168, off offset:384
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_loadcnt 0x0
	flat_store_b32 v[67:68], v126 scope:SCOPE_SYS
	s_wait_storecnt 0x0
	flat_store_b32 v[69:70], v127 scope:SCOPE_SYS
	s_wait_storecnt 0x0
	flat_store_b32 v[71:72], v114 scope:SCOPE_SYS
	s_wait_storecnt 0x0
	flat_store_b32 v[138:139], v106 scope:SCOPE_SYS
	s_wait_storecnt 0x0
	flat_store_b32 v[140:141], v98 scope:SCOPE_SYS
	s_wait_storecnt 0x0
	flat_store_b32 v[142:143], v91 scope:SCOPE_SYS
	s_wait_storecnt 0x0
	flat_store_b32 v[144:145], v84 scope:SCOPE_SYS
	s_wait_storecnt 0x0
	flat_store_b32 v[146:147], v76 scope:SCOPE_SYS
	s_wait_storecnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	flat_load_b32 v168, v[148:149] scope:SCOPE_SYS
	s_wait_loadcnt_dscnt 0x0
	global_store_b32 v[152:153], v168, off offset:64
	flat_load_b32 v168, v[150:151] scope:SCOPE_SYS
	s_wait_loadcnt_dscnt 0x0
	global_store_b32 v[152:153], v168, off offset:32832
	flat_load_b32 v168, v[154:155] scope:SCOPE_SYS
	s_wait_loadcnt_dscnt 0x0
	global_store_b32 v[152:153], v168, off offset:192
	flat_load_b32 v168, v[156:157] scope:SCOPE_SYS
	s_wait_loadcnt_dscnt 0x0
	global_store_b32 v[152:153], v168, off offset:32960
	flat_load_b32 v168, v[158:159] scope:SCOPE_SYS
	s_wait_loadcnt_dscnt 0x0
	global_store_b32 v[152:153], v168, off offset:320
	flat_load_b32 v168, v[160:161] scope:SCOPE_SYS
	s_wait_loadcnt_dscnt 0x0
	global_store_b32 v[152:153], v168, off offset:33088
	flat_load_b32 v168, v[162:163] scope:SCOPE_SYS
	s_wait_loadcnt_dscnt 0x0
	global_store_b32 v[152:153], v168, off offset:448
	flat_load_b32 v168, v[164:165] scope:SCOPE_SYS
	s_wait_loadcnt_dscnt 0x0
	global_store_b32 v[152:153], v168, off offset:33216
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_loadcnt 0x0
	flat_store_b32 v[67:68], v124 scope:SCOPE_SYS
	s_wait_storecnt 0x0
	flat_store_b32 v[69:70], v125 scope:SCOPE_SYS
	s_wait_storecnt 0x0
	flat_store_b32 v[71:72], v113 scope:SCOPE_SYS
	s_wait_storecnt 0x0
	flat_store_b32 v[138:139], v105 scope:SCOPE_SYS
	s_wait_storecnt 0x0
	flat_store_b32 v[140:141], v97 scope:SCOPE_SYS
	s_wait_storecnt 0x0
	flat_store_b32 v[142:143], v90 scope:SCOPE_SYS
	s_wait_storecnt 0x0
	flat_store_b32 v[144:145], v83 scope:SCOPE_SYS
	s_wait_storecnt 0x0
	flat_store_b32 v[146:147], v75 scope:SCOPE_SYS
	s_wait_storecnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	flat_load_b32 v168, v[148:149] scope:SCOPE_SYS
	s_wait_loadcnt_dscnt 0x0
	global_store_b32 v[152:153], v168, off offset:8256
	flat_load_b32 v168, v[150:151] scope:SCOPE_SYS
	s_wait_loadcnt_dscnt 0x0
	global_store_b32 v[152:153], v168, off offset:41024
	flat_load_b32 v168, v[154:155] scope:SCOPE_SYS
	s_wait_loadcnt_dscnt 0x0
	global_store_b32 v[152:153], v168, off offset:8384
	flat_load_b32 v168, v[156:157] scope:SCOPE_SYS
	s_wait_loadcnt_dscnt 0x0
	global_store_b32 v[152:153], v168, off offset:41152
	flat_load_b32 v168, v[158:159] scope:SCOPE_SYS
	s_wait_loadcnt_dscnt 0x0
	global_store_b32 v[152:153], v168, off offset:8512
	flat_load_b32 v168, v[160:161] scope:SCOPE_SYS
	s_wait_loadcnt_dscnt 0x0
	global_store_b32 v[152:153], v168, off offset:41280
	flat_load_b32 v168, v[162:163] scope:SCOPE_SYS
	s_wait_loadcnt_dscnt 0x0
	global_store_b32 v[152:153], v168, off offset:8640
	flat_load_b32 v168, v[164:165] scope:SCOPE_SYS
	s_wait_loadcnt_dscnt 0x0
	global_store_b32 v[152:153], v168, off offset:41408
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_loadcnt 0x0
	flat_store_b32 v[67:68], v121 scope:SCOPE_SYS
	s_wait_storecnt 0x0
	flat_store_b32 v[69:70], v122 scope:SCOPE_SYS
	s_wait_storecnt 0x0
	flat_store_b32 v[71:72], v111 scope:SCOPE_SYS
	s_wait_storecnt 0x0
	flat_store_b32 v[138:139], v104 scope:SCOPE_SYS
	s_wait_storecnt 0x0
	flat_store_b32 v[140:141], v96 scope:SCOPE_SYS
	s_wait_storecnt 0x0
	flat_store_b32 v[142:143], v89 scope:SCOPE_SYS
	s_wait_storecnt 0x0
	flat_store_b32 v[144:145], v82 scope:SCOPE_SYS
	s_wait_storecnt 0x0
	flat_store_b32 v[146:147], v74 scope:SCOPE_SYS
	s_wait_storecnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	flat_load_b32 v168, v[148:149] scope:SCOPE_SYS
	s_wait_loadcnt_dscnt 0x0
	global_store_b32 v[152:153], v168, off offset:16448
	flat_load_b32 v168, v[150:151] scope:SCOPE_SYS
	s_wait_loadcnt_dscnt 0x0
	global_store_b32 v[152:153], v168, off offset:49216
	flat_load_b32 v168, v[154:155] scope:SCOPE_SYS
	s_wait_loadcnt_dscnt 0x0
	global_store_b32 v[152:153], v168, off offset:16576
	flat_load_b32 v168, v[156:157] scope:SCOPE_SYS
	s_wait_loadcnt_dscnt 0x0
	global_store_b32 v[152:153], v168, off offset:49344
	flat_load_b32 v168, v[158:159] scope:SCOPE_SYS
	s_wait_loadcnt_dscnt 0x0
	global_store_b32 v[152:153], v168, off offset:16704
	flat_load_b32 v168, v[160:161] scope:SCOPE_SYS
	s_wait_loadcnt_dscnt 0x0
	global_store_b32 v[152:153], v168, off offset:49472
	flat_load_b32 v168, v[162:163] scope:SCOPE_SYS
	s_wait_loadcnt_dscnt 0x0
	global_store_b32 v[152:153], v168, off offset:16832
	flat_load_b32 v168, v[164:165] scope:SCOPE_SYS
	s_wait_loadcnt_dscnt 0x0
	global_store_b32 v[152:153], v168, off offset:49600
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_loadcnt 0x0
	flat_store_b32 v[67:68], v120 scope:SCOPE_SYS
	s_wait_storecnt 0x0
	flat_store_b32 v[69:70], v137 scope:SCOPE_SYS
	s_wait_storecnt 0x0
	flat_store_b32 v[71:72], v136 scope:SCOPE_SYS
	s_wait_storecnt 0x0
	flat_store_b32 v[138:139], v135 scope:SCOPE_SYS
	s_wait_storecnt 0x0
	flat_store_b32 v[140:141], v123 scope:SCOPE_SYS
	s_wait_storecnt 0x0
	flat_store_b32 v[142:143], v112 scope:SCOPE_SYS
	s_wait_storecnt 0x0
	flat_store_b32 v[144:145], v103 scope:SCOPE_SYS
	s_wait_storecnt 0x0
	flat_store_b32 v[146:147], v80 scope:SCOPE_SYS
	s_wait_storecnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	flat_load_b32 v67, v[148:149] scope:SCOPE_SYS
	s_wait_loadcnt_dscnt 0x0
	global_store_b32 v[166:167], v67, off offset:64
	flat_load_b32 v67, v[150:151] scope:SCOPE_SYS
	s_wait_loadcnt_dscnt 0x0
	global_store_b32 v[65:66], v67, off offset:64
	flat_load_b32 v67, v[154:155] scope:SCOPE_SYS
	s_wait_loadcnt_dscnt 0x0
	global_store_b32 v[166:167], v67, off offset:192
	flat_load_b32 v67, v[156:157] scope:SCOPE_SYS
	s_wait_loadcnt_dscnt 0x0
	global_store_b32 v[65:66], v67, off offset:192
	flat_load_b32 v67, v[158:159] scope:SCOPE_SYS
	s_wait_loadcnt_dscnt 0x0
	global_store_b32 v[166:167], v67, off offset:320
	flat_load_b32 v67, v[160:161] scope:SCOPE_SYS
	s_wait_loadcnt_dscnt 0x0
	global_store_b32 v[65:66], v67, off offset:320
	flat_load_b32 v67, v[162:163] scope:SCOPE_SYS
	s_wait_loadcnt_dscnt 0x0
	global_store_b32 v[166:167], v67, off offset:448
	flat_load_b32 v67, v[164:165] scope:SCOPE_SYS
	s_wait_loadcnt_dscnt 0x0
	global_store_b32 v[65:66], v67, off offset:448
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	global_store_b64 v73, v[0:1], s[0:1]
	s_nop 0
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
.Lfunc_end5:
	.size	_Z10iu4_ladderILi5EEvPKDv2_jS2_PKjPfPyiii, .Lfunc_end5-_Z10iu4_ladderILi5EEvPKDv2_jS2_PKjPfPyiii
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z10iu4_ladderILi5EEvPKDv2_jS2_PKjPfPyiii
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 52
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
		.amdhsa_next_free_vgpr 183
		.amdhsa_next_free_sgpr 30
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end5-_Z10iu4_ladderILi5EEvPKDv2_jS2_PKjPfPyiii)<<4)&4080)>>4
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._Z10iu4_ladderILi5EEvPKDv2_jS2_PKjPfPyiii,"axG",@progbits,_Z10iu4_ladderILi5EEvPKDv2_jS2_PKjPfPyiii,comdat
                                        ; -- End function
	.set .L_Z10iu4_ladderILi5EEvPKDv2_jS2_PKjPfPyiii.num_vgpr, 183
	.set .L_Z10iu4_ladderILi5EEvPKDv2_jS2_PKjPfPyiii.num_agpr, 0
	.set .L_Z10iu4_ladderILi5EEvPKDv2_jS2_PKjPfPyiii.numbered_sgpr, 30
	.set .L_Z10iu4_ladderILi5EEvPKDv2_jS2_PKjPfPyiii.num_named_barrier, 0
	.set .L_Z10iu4_ladderILi5EEvPKDv2_jS2_PKjPfPyiii.private_seg_size, 0
	.set .L_Z10iu4_ladderILi5EEvPKDv2_jS2_PKjPfPyiii.uses_vcc, 1
	.set .L_Z10iu4_ladderILi5EEvPKDv2_jS2_PKjPfPyiii.uses_flat_scratch, 0
	.set .L_Z10iu4_ladderILi5EEvPKDv2_jS2_PKjPfPyiii.has_dyn_sized_stack, 0
	.set .L_Z10iu4_ladderILi5EEvPKDv2_jS2_PKjPfPyiii.has_recursion, 0
	.set .L_Z10iu4_ladderILi5EEvPKDv2_jS2_PKjPfPyiii.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 8884
; TotalNumSgprs: 32
; NumVgprs: 183
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 22
; NumSGPRsForWavesPerEU: 32
; NumVGPRsForWavesPerEU: 183
; Occupancy: 8
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.set amdgpu.max_num_named_barrier, 0
	.section	.AMDGPU.csdata,"",@progbits
	.type	__hip_cuid_9a7978dd31cac309,@object ; @__hip_cuid_9a7978dd31cac309
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_9a7978dd31cac309
__hip_cuid_9a7978dd31cac309:
	.byte	0                               ; 0x0
	.size	__hip_cuid_9a7978dd31cac309, 1

	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git 8f497e0992fb7513f7f78a6f6b6f1056c375e961)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym lds
	.addrsig_sym __hip_cuid_9a7978dd31cac309
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
      - .actual_access:  write_only
        .address_space:  global
        .offset:         32
        .size:           8
        .value_kind:     global_buffer
      - .offset:         40
        .size:           4
        .value_kind:     by_value
      - .offset:         44
        .size:           4
        .value_kind:     by_value
      - .offset:         48
        .size:           4
        .value_kind:     by_value
    .gfx1250_revision: B0
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 52
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           _Z10iu4_ladderILi0EEvPKDv2_jS2_PKjPfPyiii
    .private_segment_fixed_size: 0
    .sgpr_count:     12
    .sgpr_spill_count: 0
    .symbol:         _Z10iu4_ladderILi0EEvPKDv2_jS2_PKjPfPyiii.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     69
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
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
      - .actual_access:  write_only
        .address_space:  global
        .offset:         32
        .size:           8
        .value_kind:     global_buffer
      - .offset:         40
        .size:           4
        .value_kind:     by_value
      - .offset:         44
        .size:           4
        .value_kind:     by_value
      - .offset:         48
        .size:           4
        .value_kind:     by_value
    .gfx1250_revision: B0
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 52
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           _Z10iu4_ladderILi1EEvPKDv2_jS2_PKjPfPyiii
    .private_segment_fixed_size: 0
    .sgpr_count:     12
    .sgpr_spill_count: 0
    .symbol:         _Z10iu4_ladderILi1EEvPKDv2_jS2_PKjPfPyiii.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     76
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
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
      - .actual_access:  write_only
        .address_space:  global
        .offset:         32
        .size:           8
        .value_kind:     global_buffer
      - .offset:         40
        .size:           4
        .value_kind:     by_value
      - .offset:         44
        .size:           4
        .value_kind:     by_value
      - .offset:         48
        .size:           4
        .value_kind:     by_value
    .gfx1250_revision: B0
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 52
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           _Z10iu4_ladderILi2EEvPKDv2_jS2_PKjPfPyiii
    .private_segment_fixed_size: 0
    .sgpr_count:     12
    .sgpr_spill_count: 0
    .symbol:         _Z10iu4_ladderILi2EEvPKDv2_jS2_PKjPfPyiii.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     86
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
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
      - .actual_access:  write_only
        .address_space:  global
        .offset:         32
        .size:           8
        .value_kind:     global_buffer
      - .offset:         40
        .size:           4
        .value_kind:     by_value
      - .offset:         44
        .size:           4
        .value_kind:     by_value
      - .offset:         48
        .size:           4
        .value_kind:     by_value
    .gfx1250_revision: B0
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 52
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           _Z10iu4_ladderILi3EEvPKDv2_jS2_PKjPfPyiii
    .private_segment_fixed_size: 0
    .sgpr_count:     22
    .sgpr_spill_count: 0
    .symbol:         _Z10iu4_ladderILi3EEvPKDv2_jS2_PKjPfPyiii.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     108
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
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
      - .actual_access:  write_only
        .address_space:  global
        .offset:         32
        .size:           8
        .value_kind:     global_buffer
      - .offset:         40
        .size:           4
        .value_kind:     by_value
      - .offset:         44
        .size:           4
        .value_kind:     by_value
      - .offset:         48
        .size:           4
        .value_kind:     by_value
    .gfx1250_revision: B0
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 52
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           _Z10iu4_ladderILi4EEvPKDv2_jS2_PKjPfPyiii
    .private_segment_fixed_size: 0
    .sgpr_count:     30
    .sgpr_spill_count: 0
    .symbol:         _Z10iu4_ladderILi4EEvPKDv2_jS2_PKjPfPyiii.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     180
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
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
      - .actual_access:  write_only
        .address_space:  global
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  write_only
        .address_space:  global
        .offset:         32
        .size:           8
        .value_kind:     global_buffer
      - .offset:         40
        .size:           4
        .value_kind:     by_value
      - .offset:         44
        .size:           4
        .value_kind:     by_value
      - .offset:         48
        .size:           4
        .value_kind:     by_value
    .gfx1250_revision: B0
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 52
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           _Z10iu4_ladderILi5EEvPKDv2_jS2_PKjPfPyiii
    .private_segment_fixed_size: 0
    .sgpr_count:     32
    .sgpr_spill_count: 0
    .symbol:         _Z10iu4_ladderILi5EEvPKDv2_jS2_PKjPfPyiii.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     183
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target:   amdgcn-amd-amdhsa--gfx1201
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
