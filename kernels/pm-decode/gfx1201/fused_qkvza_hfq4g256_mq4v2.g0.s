.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
.amdhsa_code_object_version 6
.text
.protected pm_decode_qkvza_g0
.globl pm_decode_qkvza_g0
.p2align 8
.type pm_decode_qkvza_g0,@function
pm_decode_qkvza_g0:
	s_load_b64 s[4:5], s[0:1], 0x0
	s_load_b64 s[8:9], s[0:1], 0x8
	s_load_b64 s[10:11], s[0:1], 0x10
	s_mov_b32 s6, -1
	s_mov_b32 s7, 0x31004000
	v_lshlrev_b32_e32 v1, 2, v0
	v_mov_b32_e32 v2, 0
	v_lshlrev_b32_e32 v3, 5, v0
	v_mov_b32_e32 v4, 0
	s_wait_kmcnt 0x0
	buffer_load_b64 v[16:17], v2, s[4:7], null offen scope:SCOPE_DEV offset:0
	buffer_load_b32 v18, v1, s[4:7], null offen scope:SCOPE_DEV offset:8
	global_load_b128 v[8:11], v3, s[8:9] offset:0
	global_load_b128 v[12:15], v3, s[8:9] offset:16
	v_cmp_gt_u32_e32 vcc_lo, 16, v0
	s_wait_loadcnt 0x3
	v_cndmask_b32_e64 v16, v17, v16, vcc_lo
	s_wait_loadcnt 0x2
	v_bfe_u32 v19, v18, 4, 4
	v_cvt_f32_ubyte0_e32 v19, v19
	v_fma_mix_f32 v20, v16, v19, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v21, v20, v9
	v_bfe_u32 v19, v18, 0, 4
	v_cvt_f32_ubyte0_e32 v19, v19
	v_fma_mix_f32 v20, v16, v19, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fmac_f32_e32 v21, v20, v8
	v_bfe_u32 v19, v18, 8, 4
	v_cvt_f32_ubyte0_e32 v19, v19
	v_fma_mix_f32 v20, v16, v19, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fmac_f32_e32 v21, v20, v10
	v_bfe_u32 v19, v18, 12, 4
	v_cvt_f32_ubyte0_e32 v19, v19
	v_fma_mix_f32 v20, v16, v19, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fmac_f32_e32 v21, v20, v11
	v_bfe_u32 v19, v18, 16, 4
	v_cvt_f32_ubyte0_e32 v19, v19
	v_fma_mix_f32 v20, v16, v19, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x0
	v_fmac_f32_e32 v21, v20, v12
	v_bfe_u32 v19, v18, 20, 4
	v_cvt_f32_ubyte0_e32 v19, v19
	v_fma_mix_f32 v20, v16, v19, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fmac_f32_e32 v21, v20, v13
	v_bfe_u32 v19, v18, 24, 4
	v_cvt_f32_ubyte0_e32 v19, v19
	v_fma_mix_f32 v20, v16, v19, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fmac_f32_e32 v21, v20, v14
	v_bfe_u32 v19, v18, 28, 4
	v_cvt_f32_ubyte0_e32 v19, v19
	v_fma_mix_f32 v20, v16, v19, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fmac_f32_e32 v21, v20, v15
	v_add_f32_e32 v4, v4, v21
	v_mov_b32_e32 v5, 0
	buffer_load_b64 v[16:17], v2, s[4:7], null offen scope:SCOPE_DEV offset:136
	buffer_load_b32 v18, v1, s[4:7], null offen scope:SCOPE_DEV offset:144
	global_load_b128 v[8:11], v3, s[8:9] offset:1024
	global_load_b128 v[12:15], v3, s[8:9] offset:1040
	v_cmp_gt_u32_e32 vcc_lo, 16, v0
	s_wait_loadcnt 0x3
	v_cndmask_b32_e64 v16, v17, v16, vcc_lo
	s_wait_loadcnt 0x2
	v_bfe_u32 v19, v18, 4, 4
	v_cvt_f32_ubyte0_e32 v19, v19
	v_fma_mix_f32 v20, v16, v19, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v21, v20, v9
	v_bfe_u32 v19, v18, 0, 4
	v_cvt_f32_ubyte0_e32 v19, v19
	v_fma_mix_f32 v20, v16, v19, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fmac_f32_e32 v21, v20, v8
	v_bfe_u32 v19, v18, 8, 4
	v_cvt_f32_ubyte0_e32 v19, v19
	v_fma_mix_f32 v20, v16, v19, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fmac_f32_e32 v21, v20, v10
	v_bfe_u32 v19, v18, 12, 4
	v_cvt_f32_ubyte0_e32 v19, v19
	v_fma_mix_f32 v20, v16, v19, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fmac_f32_e32 v21, v20, v11
	v_bfe_u32 v19, v18, 16, 4
	v_cvt_f32_ubyte0_e32 v19, v19
	v_fma_mix_f32 v20, v16, v19, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x0
	v_fmac_f32_e32 v21, v20, v12
	v_bfe_u32 v19, v18, 20, 4
	v_cvt_f32_ubyte0_e32 v19, v19
	v_fma_mix_f32 v20, v16, v19, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fmac_f32_e32 v21, v20, v13
	v_bfe_u32 v19, v18, 24, 4
	v_cvt_f32_ubyte0_e32 v19, v19
	v_fma_mix_f32 v20, v16, v19, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fmac_f32_e32 v21, v20, v14
	v_bfe_u32 v19, v18, 28, 4
	v_cvt_f32_ubyte0_e32 v19, v19
	v_fma_mix_f32 v20, v16, v19, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fmac_f32_e32 v21, v20, v15
	v_add_f32_e32 v5, v5, v21
	v_mov_b32_e32 v6, 0
	buffer_load_b64 v[16:17], v2, s[4:7], null offen scope:SCOPE_DEV offset:272
	buffer_load_b32 v18, v1, s[4:7], null offen scope:SCOPE_DEV offset:280
	global_load_b128 v[8:11], v3, s[8:9] offset:2048
	global_load_b128 v[12:15], v3, s[8:9] offset:2064
	v_cmp_gt_u32_e32 vcc_lo, 16, v0
	s_wait_loadcnt 0x3
	v_cndmask_b32_e64 v16, v17, v16, vcc_lo
	s_wait_loadcnt 0x2
	v_bfe_u32 v19, v18, 4, 4
	v_cvt_f32_ubyte0_e32 v19, v19
	v_fma_mix_f32 v20, v16, v19, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v21, v20, v9
	v_bfe_u32 v19, v18, 0, 4
	v_cvt_f32_ubyte0_e32 v19, v19
	v_fma_mix_f32 v20, v16, v19, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fmac_f32_e32 v21, v20, v8
	v_bfe_u32 v19, v18, 8, 4
	v_cvt_f32_ubyte0_e32 v19, v19
	v_fma_mix_f32 v20, v16, v19, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fmac_f32_e32 v21, v20, v10
	v_bfe_u32 v19, v18, 12, 4
	v_cvt_f32_ubyte0_e32 v19, v19
	v_fma_mix_f32 v20, v16, v19, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fmac_f32_e32 v21, v20, v11
	v_bfe_u32 v19, v18, 16, 4
	v_cvt_f32_ubyte0_e32 v19, v19
	v_fma_mix_f32 v20, v16, v19, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x0
	v_fmac_f32_e32 v21, v20, v12
	v_bfe_u32 v19, v18, 20, 4
	v_cvt_f32_ubyte0_e32 v19, v19
	v_fma_mix_f32 v20, v16, v19, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fmac_f32_e32 v21, v20, v13
	v_bfe_u32 v19, v18, 24, 4
	v_cvt_f32_ubyte0_e32 v19, v19
	v_fma_mix_f32 v20, v16, v19, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fmac_f32_e32 v21, v20, v14
	v_bfe_u32 v19, v18, 28, 4
	v_cvt_f32_ubyte0_e32 v19, v19
	v_fma_mix_f32 v20, v16, v19, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fmac_f32_e32 v21, v20, v15
	v_add_f32_e32 v6, v6, v21
	v_mov_b32_e32 v7, 0
	buffer_load_b64 v[16:17], v2, s[4:7], null offen scope:SCOPE_DEV offset:408
	buffer_load_b32 v18, v1, s[4:7], null offen scope:SCOPE_DEV offset:416
	global_load_b128 v[8:11], v3, s[8:9] offset:3072
	global_load_b128 v[12:15], v3, s[8:9] offset:3088
	v_cmp_gt_u32_e32 vcc_lo, 16, v0
	s_wait_loadcnt 0x3
	v_cndmask_b32_e64 v16, v17, v16, vcc_lo
	s_wait_loadcnt 0x2
	v_bfe_u32 v19, v18, 4, 4
	v_cvt_f32_ubyte0_e32 v19, v19
	v_fma_mix_f32 v20, v16, v19, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v21, v20, v9
	v_bfe_u32 v19, v18, 0, 4
	v_cvt_f32_ubyte0_e32 v19, v19
	v_fma_mix_f32 v20, v16, v19, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fmac_f32_e32 v21, v20, v8
	v_bfe_u32 v19, v18, 8, 4
	v_cvt_f32_ubyte0_e32 v19, v19
	v_fma_mix_f32 v20, v16, v19, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fmac_f32_e32 v21, v20, v10
	v_bfe_u32 v19, v18, 12, 4
	v_cvt_f32_ubyte0_e32 v19, v19
	v_fma_mix_f32 v20, v16, v19, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fmac_f32_e32 v21, v20, v11
	v_bfe_u32 v19, v18, 16, 4
	v_cvt_f32_ubyte0_e32 v19, v19
	v_fma_mix_f32 v20, v16, v19, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	s_wait_loadcnt 0x0
	v_fmac_f32_e32 v21, v20, v12
	v_bfe_u32 v19, v18, 20, 4
	v_cvt_f32_ubyte0_e32 v19, v19
	v_fma_mix_f32 v20, v16, v19, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fmac_f32_e32 v21, v20, v13
	v_bfe_u32 v19, v18, 24, 4
	v_cvt_f32_ubyte0_e32 v19, v19
	v_fma_mix_f32 v20, v16, v19, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fmac_f32_e32 v21, v20, v14
	v_bfe_u32 v19, v18, 28, 4
	v_cvt_f32_ubyte0_e32 v19, v19
	v_fma_mix_f32 v20, v16, v19, v16 op_sel:[0,0,1] op_sel_hi:[1,0,1]
	v_fmac_f32_e32 v21, v20, v15
	v_add_f32_e32 v7, v7, v21
	v_add_f32_e32 v4, v4, v5
	v_add_f32_e32 v22, v6, v7
	v_add_f32_e32 v4, v4, v22
	ds_swizzle_b32 v21, v4 offset:swizzle(BITMASK_PERM,"1pppp")
	s_wait_dscnt 0x0
	v_add_f32_e32 v4, v4, v21
	v_cmp_gt_u32_e32 vcc_lo, 24, v0
	v_cndmask_b32_e64 v22, 0, 8, vcc_lo
	v_add_lshl_u32 v22, v22, v0, 2
	ds_bpermute_b32 v21, v22, v4
	s_wait_dscnt 0x0
	v_add_f32_e32 v4, v4, v21
	v_cmp_gt_u32_e32 vcc_lo, 28, v0
	v_cndmask_b32_e64 v22, 0, 4, vcc_lo
	v_add_lshl_u32 v22, v22, v0, 2
	ds_bpermute_b32 v21, v22, v4
	s_wait_dscnt 0x0
	v_add_f32_e32 v4, v4, v21
	v_cmp_gt_u32_e32 vcc_lo, 30, v0
	v_cndmask_b32_e64 v22, 0, 2, vcc_lo
	v_add_lshl_u32 v22, v22, v0, 2
	ds_bpermute_b32 v21, v22, v4
	s_wait_dscnt 0x0
	v_add_f32_e32 v4, v4, v21
	v_cmp_gt_u32_e32 vcc_lo, 31, v0
	v_cndmask_b32_e64 v22, 0, 1, vcc_lo
	v_add_lshl_u32 v22, v22, v0, 2
	ds_bpermute_b32 v21, v22, v4
	s_wait_dscnt 0x0
	v_add_f32_e32 v4, v4, v21
	global_store_b32 v1, v4, s[10:11]
	s_endpgm
.Lpm_decode_qkvza_g0_end:
.size pm_decode_qkvza_g0, .Lpm_decode_qkvza_g0_end-pm_decode_qkvza_g0
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel pm_decode_qkvza_g0
	.amdhsa_group_segment_fixed_size 0
	.amdhsa_private_segment_fixed_size 0
	.amdhsa_kernarg_size 24
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
	.amdhsa_next_free_vgpr 23
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
	.amdhsa_inst_pref_size ((instprefsize(.Lpm_decode_qkvza_g0_end-pm_decode_qkvza_g0)<<4)&4080)>>4
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
      - .address_space: global
        .name: weights
        .offset: 0
        .size: 8
        .value_kind: global_buffer
      - .address_space: global
        .name: activation
        .offset: 8
        .size: 8
        .value_kind: global_buffer
      - .address_space: global
        .name: lane_output
        .offset: 16
        .size: 8
        .value_kind: global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 24
    .max_flat_workgroup_size: 32
    .name: pm_decode_qkvza_g0
    .private_segment_fixed_size: 0
    .sgpr_count: 14
    .sgpr_spill_count: 0
    .symbol: pm_decode_qkvza_g0.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count: 23
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target: amdgcn-amd-amdhsa--gfx1201
amdhsa.version: [1, 2]
...
.end_amdgpu_metadata
