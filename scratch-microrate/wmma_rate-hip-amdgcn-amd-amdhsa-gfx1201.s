	.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
	.amdhsa_code_object_version 6
	.text
	.protected	peak_f16                ; -- Begin function peak_f16
	.globl	peak_f16
	.p2align	8
	.type	peak_f16,@function
peak_f16:                               ; @peak_f16
	.cfi_startproc
; %bb.0:                                ; %.preheader21
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_load_b32 s2, s[0:1], 0x8
	s_wait_kmcnt 0x0
	s_cmp_lt_i32 s2, 1
	s_cbranch_scc1 .LBB0_3
; %bb.1:                                ; %.preheader20.preheader
	s_mov_b32 s11, 0x3f803f00
	s_mov_b32 s10, 0x3e803e00
	s_mov_b32 s9, 0x3d803d00
	s_mov_b32 s8, 0x3c803c00
	v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v68, s11
	s_mov_b32 s7, 0x43804300
	s_mov_b32 s6, 0x42804200
	s_mov_b32 s5, 0x41804100
	s_mov_b32 s4, 0x40804000
	v_dual_mov_b32 v65, s8 :: v_dual_mov_b32 v72, s7
	v_dual_mov_b32 v67, s10 :: v_dual_mov_b32 v66, s9
	v_dual_mov_b32 v71, s6 :: v_dual_mov_b32 v70, s5
	v_dual_mov_b32 v69, s4 :: v_dual_mov_b32 v2, v1
	v_dual_mov_b32 v3, v1 :: v_dual_mov_b32 v4, v1
	v_dual_mov_b32 v5, v1 :: v_dual_mov_b32 v6, v1
	v_dual_mov_b32 v7, v1 :: v_dual_mov_b32 v8, v1
	v_dual_mov_b32 v9, v1 :: v_dual_mov_b32 v10, v1
	v_dual_mov_b32 v11, v1 :: v_dual_mov_b32 v12, v1
	v_dual_mov_b32 v13, v1 :: v_dual_mov_b32 v14, v1
	v_dual_mov_b32 v15, v1 :: v_dual_mov_b32 v16, v1
	v_dual_mov_b32 v17, v1 :: v_dual_mov_b32 v18, v1
	v_dual_mov_b32 v19, v1 :: v_dual_mov_b32 v20, v1
	v_dual_mov_b32 v21, v1 :: v_dual_mov_b32 v22, v1
	v_dual_mov_b32 v23, v1 :: v_dual_mov_b32 v24, v1
	v_dual_mov_b32 v25, v1 :: v_dual_mov_b32 v26, v1
	v_dual_mov_b32 v27, v1 :: v_dual_mov_b32 v28, v1
	v_dual_mov_b32 v29, v1 :: v_dual_mov_b32 v30, v1
	v_dual_mov_b32 v31, v1 :: v_dual_mov_b32 v32, v1
	v_dual_mov_b32 v33, v1 :: v_dual_mov_b32 v34, v1
	v_dual_mov_b32 v35, v1 :: v_dual_mov_b32 v36, v1
	v_dual_mov_b32 v37, v1 :: v_dual_mov_b32 v38, v1
	v_dual_mov_b32 v39, v1 :: v_dual_mov_b32 v40, v1
	v_dual_mov_b32 v41, v1 :: v_dual_mov_b32 v42, v1
	v_dual_mov_b32 v43, v1 :: v_dual_mov_b32 v44, v1
	v_dual_mov_b32 v45, v1 :: v_dual_mov_b32 v46, v1
	v_dual_mov_b32 v47, v1 :: v_dual_mov_b32 v48, v1
	v_dual_mov_b32 v49, v1 :: v_dual_mov_b32 v50, v1
	v_dual_mov_b32 v51, v1 :: v_dual_mov_b32 v52, v1
	v_dual_mov_b32 v53, v1 :: v_dual_mov_b32 v54, v1
	v_dual_mov_b32 v55, v1 :: v_dual_mov_b32 v56, v1
	v_dual_mov_b32 v57, v1 :: v_dual_mov_b32 v58, v1
	v_dual_mov_b32 v59, v1 :: v_dual_mov_b32 v60, v1
	v_dual_mov_b32 v61, v1 :: v_dual_mov_b32 v62, v1
	v_dual_mov_b32 v63, v1 :: v_dual_mov_b32 v64, v1
.LBB0_2:                                ; %.preheader20
                                        ; =>This Inner Loop Header: Depth=1
	v_wmma_f32_16x16x16_f16 v[1:8], v[65:68], v[69:72], v[1:8]
	v_wmma_f32_16x16x16_f16 v[9:16], v[65:68], v[69:72], v[9:16]
	v_wmma_f32_16x16x16_f16 v[17:24], v[65:68], v[69:72], v[17:24]
	v_wmma_f32_16x16x16_f16 v[25:32], v[65:68], v[69:72], v[25:32]
	v_wmma_f32_16x16x16_f16 v[33:40], v[65:68], v[69:72], v[33:40]
	v_wmma_f32_16x16x16_f16 v[41:48], v[65:68], v[69:72], v[41:48]
	v_wmma_f32_16x16x16_f16 v[49:56], v[65:68], v[69:72], v[49:56]
	v_wmma_f32_16x16x16_f16 v[57:64], v[65:68], v[69:72], v[57:64]
	s_add_co_i32 s2, s2, -1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_lg_u32 s2, 0
	s_cbranch_scc1 .LBB0_2
	s_branch .LBB0_4
.LBB0_3:
	v_mov_b32_e32 v57, 0
	v_mov_b32_e32 v49, 0
	v_mov_b32_e32 v41, 0
	v_mov_b32_e32 v33, 0
	v_mov_b32_e32 v25, 0
	v_mov_b32_e32 v17, 0
	v_mov_b32_e32 v9, 0
	v_mov_b32_e32 v1, 0
.LBB0_4:                                ; %.preheader
	v_and_b32_e32 v0, 31, v0
	s_mov_b32 s2, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_eq_u32_e32 0, v0
	s_cbranch_execz .LBB0_9
; %bb.5:
	v_add_f32_e32 v0, 0, v1
	s_mov_b32 s4, exec_lo
	s_mov_b64 s[2:3], 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v0, v0, v9
	v_add_f32_e32 v0, v0, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v0, v0, v25
	v_add_f32_e32 v0, v0, v33
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v0, v0, v41
	v_add_f32_e32 v0, v0, v49
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v0, v0, v57
	v_trunc_f32_e32 v0, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v1, 0x2f800000, v0
	v_floor_f32_e32 v1, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_fmamk_f32 v0, v1, 0xcf800000, v0
	v_cvt_u32_f32_e32 v1, v1
	v_cvt_u32_f32_e32 v0, v0
.LBB0_6:                                ; %ComputeLoop
                                        ; =>This Inner Loop Header: Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_ctz_i32_b32 s5, s4
	s_wait_alu depctr_sa_sdst(0)
	v_readlane_b32 s7, v1, s5
	v_readlane_b32 s6, v0, s5
	s_lshl_b32 s5, 1, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 s4, s4, s5
	s_add_nc_u64 s[2:3], s[2:3], s[6:7]
	s_cbranch_scc1 .LBB0_6
; %bb.7:                                ; %ComputeEnd
	v_mbcnt_lo_u32_b32 v0, exec_lo, 0
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_eq_u32_e32 0, v0
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s4, exec_lo, s4
	s_cbranch_execz .LBB0_9
; %bb.8:
	s_load_b64 s[0:1], s[0:1], 0x0
	v_mov_b32_e32 v0, s2
	v_dual_mov_b32 v2, 0 :: v_dual_mov_b32 v1, s3
	s_wait_kmcnt 0x0
	global_atomic_add_u64 v2, v[0:1], s[0:1] scope:SCOPE_DEV
.LBB0_9:                                ; %Flow363
	s_endpgm
.Lfunc_end0:
	.size	peak_f16, .Lfunc_end0-peak_f16
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel peak_f16
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
		.amdhsa_next_free_vgpr 73
		.amdhsa_next_free_sgpr 12
		.amdhsa_reserve_vcc 0
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end0-peak_f16)<<4)&4080)>>4
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
	.set .Lpeak_f16.num_vgpr, 73
	.set .Lpeak_f16.num_agpr, 0
	.set .Lpeak_f16.numbered_sgpr, 12
	.set .Lpeak_f16.num_named_barrier, 0
	.set .Lpeak_f16.private_seg_size, 0
	.set .Lpeak_f16.uses_vcc, 0
	.set .Lpeak_f16.uses_flat_scratch, 0
	.set .Lpeak_f16.has_dyn_sized_stack, 0
	.set .Lpeak_f16.has_recursion, 0
	.set .Lpeak_f16.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 724
; TotalNumSgprs: 12
; NumVgprs: 73
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 9
; NumSGPRsForWavesPerEU: 12
; NumVGPRsForWavesPerEU: 73
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
	.protected	peak_iu8                ; -- Begin function peak_iu8
	.globl	peak_iu8
	.p2align	8
	.type	peak_iu8,@function
peak_iu8:                               ; @peak_iu8
	.cfi_startproc
; %bb.0:                                ; %.preheader27
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_load_b32 s2, s[0:1], 0x8
	v_and_b32_e32 v69, 7, v0
	s_wait_kmcnt 0x0
	s_cmp_lt_i32 s2, 1
	s_cbranch_scc1 .LBB1_3
; %bb.1:                                ; %.preheader26.preheader
	v_mov_b32_e32 v41, 0
	v_or_b32_e32 v65, 0x38383838, v69
	v_or_b32_e32 v66, 0x40404040, v69
	v_or_b32_e32 v67, 0x42424242, v69
	v_or_b32_e32 v68, 0x44444444, v69
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
.LBB1_2:                                ; %.preheader26
                                        ; =>This Inner Loop Header: Depth=1
	v_wmma_i32_16x16x16_iu8 v[41:48], v[65:66], v[67:68], v[41:48] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu8 v[57:64], v[65:66], v[67:68], v[57:64] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu8 v[49:56], v[65:66], v[67:68], v[49:56] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu8 v[33:40], v[65:66], v[67:68], v[33:40] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu8 v[25:32], v[65:66], v[67:68], v[25:32] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu8 v[17:24], v[65:66], v[67:68], v[17:24] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu8 v[9:16], v[65:66], v[67:68], v[9:16] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu8 v[1:8], v[65:66], v[67:68], v[1:8] neg_lo:[1,1,0]
	s_add_co_i32 s2, s2, -1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_lg_u32 s2, 0
	s_cbranch_scc1 .LBB1_2
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
.LBB1_4:                                ; %.preheader
	v_and_b32_e32 v0, 31, v0
	s_mov_b32 s2, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_eq_u32_e32 0, v0
	s_cbranch_execz .LBB1_9
; %bb.5:
	v_cmp_eq_u32_e32 vcc_lo, 1, v69
	v_cmp_eq_u32_e64 s2, 2, v69
	v_cmp_eq_u32_e64 s3, 3, v69
	v_cmp_eq_u32_e64 s4, 4, v69
	v_cmp_eq_u32_e64 s5, 5, v69
	v_cndmask_b32_e32 v0, v41, v42, vcc_lo
	v_cndmask_b32_e32 v41, v57, v58, vcc_lo
	v_cndmask_b32_e32 v42, v49, v50, vcc_lo
	v_cndmask_b32_e32 v33, v33, v34, vcc_lo
	v_cndmask_b32_e32 v25, v25, v26, vcc_lo
	v_cndmask_b32_e64 v0, v0, v43, s2
	v_cndmask_b32_e64 v41, v41, v59, s2
	v_cndmask_b32_e64 v34, v42, v51, s2
	v_cndmask_b32_e32 v17, v17, v18, vcc_lo
	v_cmp_eq_u32_e64 s6, 6, v69
	v_cndmask_b32_e64 v0, v0, v44, s3
	v_cndmask_b32_e64 v41, v41, v60, s3
	v_cndmask_b32_e64 v26, v34, v52, s3
	v_cndmask_b32_e32 v9, v9, v10, vcc_lo
	v_cmp_eq_u32_e64 s7, 7, v69
	v_cndmask_b32_e64 v0, v0, v45, s4
	v_cndmask_b32_e64 v34, v41, v61, s4
	v_cndmask_b32_e64 v18, v26, v53, s4
	v_cndmask_b32_e64 v25, v25, v27, s2
	v_cndmask_b32_e64 v17, v17, v19, s2
	v_cndmask_b32_e64 v0, v0, v46, s5
	v_cndmask_b32_e64 v26, v34, v62, s5
	v_cndmask_b32_e64 v10, v18, v54, s5
	v_cndmask_b32_e64 v9, v9, v11, s2
	v_cndmask_b32_e64 v17, v17, v20, s3
	v_cndmask_b32_e64 v0, v0, v47, s6
	v_cndmask_b32_e64 v18, v26, v63, s6
	v_cndmask_b32_e64 v10, v10, v55, s6
	v_cndmask_b32_e64 v26, v33, v35, s2
	v_cndmask_b32_e32 v1, v1, v2, vcc_lo
	v_cndmask_b32_e64 v0, v0, v48, s7
	v_cndmask_b32_e64 v18, v18, v64, s7
	v_cndmask_b32_e64 v10, v10, v56, s7
	v_cndmask_b32_e64 v19, v26, v36, s3
	v_cndmask_b32_e64 v17, v17, v21, s4
	v_cndmask_b32_e64 v9, v9, v12, s3
	v_add_co_u32 v0, s8, v0, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_ci_u32_e64 v18, null, 0, 0, s8
	v_cndmask_b32_e64 v19, v19, v37, s4
	v_add_co_u32 v0, s8, v0, v10
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_ci_u32_e64 v10, null, 0, v18, s8
	v_cndmask_b32_e64 v18, v25, v28, s3
	v_cndmask_b32_e64 v2, v19, v38, s5
	v_cndmask_b32_e64 v1, v1, v3, s2
	v_cndmask_b32_e64 v9, v9, v13, s4
	s_mov_b32 s9, exec_lo
	v_cndmask_b32_e64 v11, v18, v29, s4
	v_cndmask_b32_e64 v2, v2, v39, s6
	v_cndmask_b32_e64 v1, v1, v4, s3
	v_cndmask_b32_e64 v9, v9, v14, s5
	s_mov_b64 s[2:3], 0
	v_cndmask_b32_e64 v3, v11, v30, s5
	v_cndmask_b32_e64 v11, v17, v22, s5
	v_cndmask_b32_e64 v2, v2, v40, s7
	v_cndmask_b32_e64 v1, v1, v5, s4
	v_cndmask_b32_e64 v5, v9, v15, s6
	v_cndmask_b32_e64 v3, v3, v31, s6
	v_cndmask_b32_e64 v4, v11, v23, s6
	v_add_co_u32 v0, vcc_lo, v0, v2
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, 0, v10, vcc_lo
	v_cndmask_b32_e64 v3, v3, v32, s7
	v_cndmask_b32_e64 v4, v4, v24, s7
	v_cndmask_b32_e64 v1, v1, v6, s5
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_u32 v0, vcc_lo, v0, v3
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, 0, v2, vcc_lo
	v_cndmask_b32_e64 v3, v5, v16, s7
	v_cndmask_b32_e64 v1, v1, v7, s6
	v_add_co_u32 v0, vcc_lo, v0, v4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, 0, v2, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v1, v1, v8, s7
	v_add_co_u32 v0, vcc_lo, v0, v3
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v2, null, 0, v2, vcc_lo
	v_add_co_u32 v0, vcc_lo, v0, v1
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v1, null, 0, v2, vcc_lo
.LBB1_6:                                ; %ComputeLoop
                                        ; =>This Inner Loop Header: Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_ctz_i32_b32 s6, s9
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_readlane_b32 s5, v1, s6
	v_readlane_b32 s4, v0, s6
	s_lshl_b32 s6, 1, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 s9, s9, s6
	s_add_nc_u64 s[2:3], s[2:3], s[4:5]
	s_cbranch_scc1 .LBB1_6
; %bb.7:                                ; %ComputeEnd
	v_mbcnt_lo_u32_b32 v0, exec_lo, 0
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_eq_u32_e32 0, v0
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s4, exec_lo, s4
	s_cbranch_execz .LBB1_9
; %bb.8:
	s_load_b64 s[0:1], s[0:1], 0x0
	v_mov_b32_e32 v0, s2
	v_dual_mov_b32 v2, 0 :: v_dual_mov_b32 v1, s3
	s_wait_kmcnt 0x0
	global_atomic_add_u64 v2, v[0:1], s[0:1] scope:SCOPE_DEV
.LBB1_9:                                ; %Flow369
	s_endpgm
.Lfunc_end1:
	.size	peak_iu8, .Lfunc_end1-peak_iu8
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel peak_iu8
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
		.amdhsa_next_free_vgpr 70
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
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end1-peak_iu8)<<4)&4080)>>4
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
	.set .Lpeak_iu8.num_vgpr, 70
	.set .Lpeak_iu8.num_agpr, 0
	.set .Lpeak_iu8.numbered_sgpr, 10
	.set .Lpeak_iu8.num_named_barrier, 0
	.set .Lpeak_iu8.private_seg_size, 0
	.set .Lpeak_iu8.uses_vcc, 1
	.set .Lpeak_iu8.uses_flat_scratch, 0
	.set .Lpeak_iu8.has_dyn_sized_stack, 0
	.set .Lpeak_iu8.has_recursion, 0
	.set .Lpeak_iu8.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 1436
; TotalNumSgprs: 12
; NumVgprs: 70
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 8
; NumSGPRsForWavesPerEU: 12
; NumVGPRsForWavesPerEU: 70
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
	.protected	peak_fp8                ; -- Begin function peak_fp8
	.globl	peak_fp8
	.p2align	8
	.type	peak_fp8,@function
peak_fp8:                               ; @peak_fp8
	.cfi_startproc
; %bb.0:                                ; %.preheader27
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_load_b32 s2, s[0:1], 0x8
	v_and_b32_e32 v69, 7, v0
	s_wait_kmcnt 0x0
	s_cmp_lt_i32 s2, 1
	s_cbranch_scc1 .LBB2_3
; %bb.1:                                ; %.preheader26.preheader
	v_mov_b32_e32 v57, 0
	v_or_b32_e32 v65, 0x38383838, v69
	v_or_b32_e32 v66, 0x40404040, v69
	v_or_b32_e32 v67, 0x42424242, v69
	v_or_b32_e32 v68, 0x44444444, v69
	v_dual_mov_b32 v58, v57 :: v_dual_mov_b32 v59, v57
	v_dual_mov_b32 v60, v57 :: v_dual_mov_b32 v61, v57
	v_dual_mov_b32 v62, v57 :: v_dual_mov_b32 v63, v57
	v_dual_mov_b32 v64, v57 :: v_dual_mov_b32 v49, v57
	v_dual_mov_b32 v50, v57 :: v_dual_mov_b32 v51, v57
	v_dual_mov_b32 v52, v57 :: v_dual_mov_b32 v53, v57
	v_dual_mov_b32 v54, v57 :: v_dual_mov_b32 v55, v57
	v_dual_mov_b32 v56, v57 :: v_dual_mov_b32 v41, v57
	v_dual_mov_b32 v42, v57 :: v_dual_mov_b32 v43, v57
	v_dual_mov_b32 v44, v57 :: v_dual_mov_b32 v45, v57
	v_dual_mov_b32 v46, v57 :: v_dual_mov_b32 v47, v57
	v_dual_mov_b32 v48, v57 :: v_dual_mov_b32 v33, v57
	v_dual_mov_b32 v34, v57 :: v_dual_mov_b32 v35, v57
	v_dual_mov_b32 v36, v57 :: v_dual_mov_b32 v37, v57
	v_dual_mov_b32 v38, v57 :: v_dual_mov_b32 v39, v57
	v_dual_mov_b32 v40, v57 :: v_dual_mov_b32 v25, v57
	v_dual_mov_b32 v26, v57 :: v_dual_mov_b32 v27, v57
	v_dual_mov_b32 v28, v57 :: v_dual_mov_b32 v29, v57
	v_dual_mov_b32 v30, v57 :: v_dual_mov_b32 v31, v57
	v_dual_mov_b32 v32, v57 :: v_dual_mov_b32 v17, v57
	v_dual_mov_b32 v18, v57 :: v_dual_mov_b32 v19, v57
	v_dual_mov_b32 v20, v57 :: v_dual_mov_b32 v21, v57
	v_dual_mov_b32 v22, v57 :: v_dual_mov_b32 v23, v57
	v_dual_mov_b32 v24, v57 :: v_dual_mov_b32 v9, v57
	v_dual_mov_b32 v10, v57 :: v_dual_mov_b32 v11, v57
	v_dual_mov_b32 v12, v57 :: v_dual_mov_b32 v13, v57
	v_dual_mov_b32 v14, v57 :: v_dual_mov_b32 v15, v57
	v_dual_mov_b32 v16, v57 :: v_dual_mov_b32 v1, v57
	v_dual_mov_b32 v2, v57 :: v_dual_mov_b32 v3, v57
	v_dual_mov_b32 v4, v57 :: v_dual_mov_b32 v5, v57
	v_dual_mov_b32 v6, v57 :: v_dual_mov_b32 v7, v57
	v_mov_b32_e32 v8, v57
.LBB2_2:                                ; %.preheader26
                                        ; =>This Inner Loop Header: Depth=1
	v_wmma_f32_16x16x16_fp8_fp8 v[57:64], v[65:66], v[67:68], v[57:64]
	v_wmma_f32_16x16x16_fp8_fp8 v[49:56], v[65:66], v[67:68], v[49:56]
	v_wmma_f32_16x16x16_fp8_fp8 v[41:48], v[65:66], v[67:68], v[41:48]
	v_wmma_f32_16x16x16_fp8_fp8 v[33:40], v[65:66], v[67:68], v[33:40]
	v_wmma_f32_16x16x16_fp8_fp8 v[25:32], v[65:66], v[67:68], v[25:32]
	v_wmma_f32_16x16x16_fp8_fp8 v[17:24], v[65:66], v[67:68], v[17:24]
	v_wmma_f32_16x16x16_fp8_fp8 v[9:16], v[65:66], v[67:68], v[9:16]
	v_wmma_f32_16x16x16_fp8_fp8 v[1:8], v[65:66], v[67:68], v[1:8]
	s_add_co_i32 s2, s2, -1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_lg_u32 s2, 0
	s_cbranch_scc1 .LBB2_2
	s_branch .LBB2_4
.LBB2_3:
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
	v_mov_b32_e32 v57, v8
.LBB2_4:                                ; %.preheader
	v_and_b32_e32 v0, 31, v0
	s_mov_b32 s2, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_eq_u32_e32 0, v0
	s_cbranch_execz .LBB2_9
; %bb.5:
	v_cmp_eq_u32_e32 vcc_lo, 1, v69
	v_cmp_eq_u32_e64 s2, 2, v69
	v_cmp_eq_u32_e64 s3, 3, v69
	v_cmp_eq_u32_e64 s4, 4, v69
	v_cmp_eq_u32_e64 s5, 5, v69
	v_cndmask_b32_e32 v0, v57, v58, vcc_lo
	v_cndmask_b32_e32 v1, v1, v2, vcc_lo
	v_cmp_eq_u32_e64 s6, 6, v69
	v_cmp_eq_u32_e64 s7, 7, v69
	v_cndmask_b32_e32 v33, v33, v34, vcc_lo
	v_cndmask_b32_e64 v0, v0, v59, s2
	v_cndmask_b32_e64 v1, v1, v3, s2
	v_cndmask_b32_e32 v25, v25, v26, vcc_lo
	v_cndmask_b32_e32 v9, v9, v10, vcc_lo
	s_mov_b32 s8, exec_lo
	v_cndmask_b32_e64 v0, v0, v60, s3
	v_cndmask_b32_e64 v1, v1, v4, s3
	v_cndmask_b32_e64 v25, v25, v27, s2
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v0, v0, v61, s4
	v_cndmask_b32_e64 v1, v1, v5, s4
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v25, v25, v28, s3
	v_cndmask_b32_e64 v0, v0, v62, s5
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v1, v1, v6, s5
	v_cndmask_b32_e64 v0, v0, v63, s6
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v1, v1, v7, s6
	v_cndmask_b32_e64 v0, v0, v64, s7
	v_cndmask_b32_e32 v49, v49, v50, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v1, v1, v8, s7
	v_dual_cndmask_b32 v41, v41, v42 :: v_dual_add_f32 v0, 0, v0
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v34, v49, v51, s2
	v_cndmask_b32_e32 v17, v17, v18, vcc_lo
	v_cndmask_b32_e64 v26, v41, v43, s2
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v34, v34, v52, s3
	v_cndmask_b32_e64 v18, v26, v44, s3
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v26, v34, v53, s4
	v_cndmask_b32_e64 v10, v18, v45, s4
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v18, v26, v54, s5
	v_cndmask_b32_e64 v26, v33, v35, s2
	v_cndmask_b32_e64 v10, v10, v46, s5
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v18, v18, v55, s6
	v_cndmask_b32_e64 v26, v26, v36, s3
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v10, v10, v47, s6
	v_cndmask_b32_e64 v18, v18, v56, s7
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v10, v10, v48, s7
	v_add_f32_e32 v0, v0, v18
	v_cndmask_b32_e64 v17, v17, v19, s2
	v_cndmask_b32_e64 v19, v26, v37, s4
	v_cndmask_b32_e64 v18, v25, v29, s4
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_f32_e32 v0, v0, v10
	v_cndmask_b32_e64 v9, v9, v11, s2
	v_cndmask_b32_e64 v11, v19, v38, s5
	v_cndmask_b32_e64 v17, v17, v20, s3
	v_cndmask_b32_e64 v10, v18, v30, s5
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cndmask_b32_e64 v9, v9, v12, s3
	v_cndmask_b32_e64 v2, v11, v39, s6
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cndmask_b32_e64 v11, v17, v21, s4
	v_cndmask_b32_e64 v3, v10, v31, s6
	s_mov_b64 s[2:3], 0
	v_cndmask_b32_e64 v9, v9, v13, s4
	v_cndmask_b32_e64 v2, v2, v40, s7
	v_cndmask_b32_e64 v10, v11, v22, s5
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v4, v9, v14, s5
	v_add_f32_e32 v0, v0, v2
	v_cndmask_b32_e64 v2, v3, v32, s7
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v3, v10, v23, s6
	v_add_f32_e32 v0, v0, v2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v2, v3, v24, s7
	v_cndmask_b32_e64 v3, v4, v15, s6
	v_add_f32_e32 v0, v0, v2
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v2, v3, v16, s7
	v_add_f32_e32 v0, v0, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v0, v0, v1
	v_trunc_f32_e32 v0, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v1, 0x2f800000, v0
	v_floor_f32_e32 v1, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_fmamk_f32 v0, v1, 0xcf800000, v0
	v_cvt_u32_f32_e32 v1, v1
	v_cvt_u32_f32_e32 v0, v0
.LBB2_6:                                ; %ComputeLoop
                                        ; =>This Inner Loop Header: Depth=1
	s_ctz_i32_b32 s6, s8
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_readlane_b32 s5, v1, s6
	v_readlane_b32 s4, v0, s6
	s_lshl_b32 s6, 1, s6
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 s8, s8, s6
	s_add_nc_u64 s[2:3], s[2:3], s[4:5]
	s_cbranch_scc1 .LBB2_6
; %bb.7:                                ; %ComputeEnd
	v_mbcnt_lo_u32_b32 v0, exec_lo, 0
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_eq_u32_e32 0, v0
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s4, exec_lo, s4
	s_cbranch_execz .LBB2_9
; %bb.8:
	s_load_b64 s[0:1], s[0:1], 0x0
	v_mov_b32_e32 v0, s2
	v_dual_mov_b32 v2, 0 :: v_dual_mov_b32 v1, s3
	s_wait_kmcnt 0x0
	global_atomic_add_u64 v2, v[0:1], s[0:1] scope:SCOPE_DEV
.LBB2_9:                                ; %Flow369
	s_endpgm
.Lfunc_end2:
	.size	peak_fp8, .Lfunc_end2-peak_fp8
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel peak_fp8
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
		.amdhsa_next_free_vgpr 70
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
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end2-peak_fp8)<<4)&4080)>>4
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
	.set .Lpeak_fp8.num_vgpr, 70
	.set .Lpeak_fp8.num_agpr, 0
	.set .Lpeak_fp8.numbered_sgpr, 9
	.set .Lpeak_fp8.num_named_barrier, 0
	.set .Lpeak_fp8.private_seg_size, 0
	.set .Lpeak_fp8.uses_vcc, 1
	.set .Lpeak_fp8.uses_flat_scratch, 0
	.set .Lpeak_fp8.has_dyn_sized_stack, 0
	.set .Lpeak_fp8.has_recursion, 0
	.set .Lpeak_fp8.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 1424
; TotalNumSgprs: 11
; NumVgprs: 70
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 8
; NumSGPRsForWavesPerEU: 11
; NumVGPRsForWavesPerEU: 70
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
	.protected	peak_iu4                ; -- Begin function peak_iu4
	.globl	peak_iu4
	.p2align	8
	.type	peak_iu4,@function
peak_iu4:                               ; @peak_iu4
	.cfi_startproc
; %bb.0:                                ; %.preheader23
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_load_b32 s2, s[0:1], 0x8
	s_wait_kmcnt 0x0
	s_cmp_lt_i32 s2, 1
	s_cbranch_scc1 .LBB3_3
; %bb.1:                                ; %.preheader22.preheader
	v_mov_b32_e32 v1, 0
	v_or_b32_e32 v65, 0x11111111, v0
	v_or_b32_e32 v66, 0x22222222, v0
	v_or_b32_e32 v67, 0x33333333, v0
	v_or_b32_e32 v68, 0x44444444, v0
	v_dual_mov_b32 v2, v1 :: v_dual_mov_b32 v3, v1
	v_dual_mov_b32 v4, v1 :: v_dual_mov_b32 v5, v1
	v_dual_mov_b32 v6, v1 :: v_dual_mov_b32 v7, v1
	v_dual_mov_b32 v8, v1 :: v_dual_mov_b32 v9, v1
	v_dual_mov_b32 v10, v1 :: v_dual_mov_b32 v11, v1
	v_dual_mov_b32 v12, v1 :: v_dual_mov_b32 v13, v1
	v_dual_mov_b32 v14, v1 :: v_dual_mov_b32 v15, v1
	v_dual_mov_b32 v16, v1 :: v_dual_mov_b32 v17, v1
	v_dual_mov_b32 v18, v1 :: v_dual_mov_b32 v19, v1
	v_dual_mov_b32 v20, v1 :: v_dual_mov_b32 v21, v1
	v_dual_mov_b32 v22, v1 :: v_dual_mov_b32 v23, v1
	v_dual_mov_b32 v24, v1 :: v_dual_mov_b32 v25, v1
	v_dual_mov_b32 v26, v1 :: v_dual_mov_b32 v27, v1
	v_dual_mov_b32 v28, v1 :: v_dual_mov_b32 v29, v1
	v_dual_mov_b32 v30, v1 :: v_dual_mov_b32 v31, v1
	v_dual_mov_b32 v32, v1 :: v_dual_mov_b32 v33, v1
	v_dual_mov_b32 v34, v1 :: v_dual_mov_b32 v35, v1
	v_dual_mov_b32 v36, v1 :: v_dual_mov_b32 v37, v1
	v_dual_mov_b32 v38, v1 :: v_dual_mov_b32 v39, v1
	v_dual_mov_b32 v40, v1 :: v_dual_mov_b32 v41, v1
	v_dual_mov_b32 v42, v1 :: v_dual_mov_b32 v43, v1
	v_dual_mov_b32 v44, v1 :: v_dual_mov_b32 v45, v1
	v_dual_mov_b32 v46, v1 :: v_dual_mov_b32 v47, v1
	v_dual_mov_b32 v48, v1 :: v_dual_mov_b32 v49, v1
	v_dual_mov_b32 v50, v1 :: v_dual_mov_b32 v51, v1
	v_dual_mov_b32 v52, v1 :: v_dual_mov_b32 v53, v1
	v_dual_mov_b32 v54, v1 :: v_dual_mov_b32 v55, v1
	v_dual_mov_b32 v56, v1 :: v_dual_mov_b32 v57, v1
	v_dual_mov_b32 v58, v1 :: v_dual_mov_b32 v59, v1
	v_dual_mov_b32 v60, v1 :: v_dual_mov_b32 v61, v1
	v_dual_mov_b32 v62, v1 :: v_dual_mov_b32 v63, v1
	v_mov_b32_e32 v64, v1
.LBB3_2:                                ; %.preheader22
                                        ; =>This Inner Loop Header: Depth=1
	v_wmma_i32_16x16x32_iu4 v[1:8], v[65:66], v[67:68], v[1:8] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[9:16], v[65:66], v[67:68], v[9:16] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[17:24], v[65:66], v[67:68], v[17:24] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[25:32], v[65:66], v[67:68], v[25:32] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[33:40], v[65:66], v[67:68], v[33:40] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[41:48], v[65:66], v[67:68], v[41:48] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[49:56], v[65:66], v[67:68], v[49:56] neg_lo:[1,1,0]
	v_wmma_i32_16x16x32_iu4 v[57:64], v[65:66], v[67:68], v[57:64] neg_lo:[1,1,0]
	s_add_co_i32 s2, s2, -1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_lg_u32 s2, 0
	s_cbranch_scc1 .LBB3_2
	s_branch .LBB3_4
.LBB3_3:
	v_mov_b32_e32 v57, 0
	v_mov_b32_e32 v49, 0
	v_mov_b32_e32 v41, 0
	v_mov_b32_e32 v33, 0
	v_mov_b32_e32 v25, 0
	v_mov_b32_e32 v17, 0
	v_mov_b32_e32 v9, 0
	v_mov_b32_e32 v1, 0
.LBB3_4:                                ; %.preheader
	v_and_b32_e32 v0, 31, v0
	s_mov_b32 s2, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_eq_u32_e32 0, v0
	s_cbranch_execz .LBB3_9
; %bb.5:
	v_add_co_u32 v0, s2, v1, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_add_co_ci_u32_e64 v1, null, 0, 0, s2
	s_mov_b32 s4, exec_lo
	v_add_co_u32 v0, vcc_lo, v0, v17
	v_add_co_ci_u32_e64 v1, null, 0, v1, vcc_lo
	s_mov_b64 s[2:3], 0
	v_add_co_u32 v0, vcc_lo, v0, v25
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, 0, v1, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v0, vcc_lo, v0, v33
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, 0, v1, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v0, vcc_lo, v0, v41
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, 0, v1, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v0, vcc_lo, v0, v49
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, 0, v1, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v0, vcc_lo, v0, v57
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v1, null, 0, v1, vcc_lo
.LBB3_6:                                ; %ComputeLoop
                                        ; =>This Inner Loop Header: Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_ctz_i32_b32 s5, s4
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_readlane_b32 s7, v1, s5
	v_readlane_b32 s6, v0, s5
	s_lshl_b32 s5, 1, s5
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 s4, s4, s5
	s_add_nc_u64 s[2:3], s[2:3], s[6:7]
	s_cbranch_scc1 .LBB3_6
; %bb.7:                                ; %ComputeEnd
	v_mbcnt_lo_u32_b32 v0, exec_lo, 0
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_eq_u32_e32 0, v0
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s4, exec_lo, s4
	s_cbranch_execz .LBB3_9
; %bb.8:
	s_load_b64 s[0:1], s[0:1], 0x0
	v_mov_b32_e32 v0, s2
	v_dual_mov_b32 v2, 0 :: v_dual_mov_b32 v1, s3
	s_wait_kmcnt 0x0
	global_atomic_add_u64 v2, v[0:1], s[0:1] scope:SCOPE_DEV
.LBB3_9:                                ; %Flow365
	s_endpgm
.Lfunc_end3:
	.size	peak_iu4, .Lfunc_end3-peak_iu4
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel peak_iu4
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
		.amdhsa_next_free_vgpr 69
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
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end3-peak_iu4)<<4)&4080)>>4
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
	.set .Lpeak_iu4.num_vgpr, 69
	.set .Lpeak_iu4.num_agpr, 0
	.set .Lpeak_iu4.numbered_sgpr, 8
	.set .Lpeak_iu4.num_named_barrier, 0
	.set .Lpeak_iu4.private_seg_size, 0
	.set .Lpeak_iu4.uses_vcc, 1
	.set .Lpeak_iu4.uses_flat_scratch, 0
	.set .Lpeak_iu4.has_dyn_sized_stack, 0
	.set .Lpeak_iu4.has_recursion, 0
	.set .Lpeak_iu4.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 728
; TotalNumSgprs: 10
; NumVgprs: 69
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 8
; NumSGPRsForWavesPerEU: 10
; NumVGPRsForWavesPerEU: 69
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
	.p2alignl 7, 3214868480
	.fill 96, 4, 3214868480
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.set amdgpu.max_num_named_barrier, 0
	.text
	.type	__hip_cuid_dec4f8244316d2d0,@object ; @__hip_cuid_dec4f8244316d2d0
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_dec4f8244316d2d0
__hip_cuid_dec4f8244316d2d0:
	.byte	0                               ; 0x0
	.size	__hip_cuid_dec4f8244316d2d0, 1

	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git 8f497e0992fb7513f7f78a6f6b6f1056c375e961)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_dec4f8244316d2d0
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
    .name:           peak_f16
    .private_segment_fixed_size: 0
    .sgpr_count:     12
    .sgpr_spill_count: 0
    .symbol:         peak_f16.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     73
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
    .name:           peak_iu8
    .private_segment_fixed_size: 0
    .sgpr_count:     12
    .sgpr_spill_count: 0
    .symbol:         peak_iu8.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     70
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
    .name:           peak_fp8
    .private_segment_fixed_size: 0
    .sgpr_count:     11
    .sgpr_spill_count: 0
    .symbol:         peak_fp8.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     70
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
    .name:           peak_iu4
    .private_segment_fixed_size: 0
    .sgpr_count:     10
    .sgpr_spill_count: 0
    .symbol:         peak_iu4.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     69
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target:   amdgcn-amd-amdhsa--gfx1201
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
