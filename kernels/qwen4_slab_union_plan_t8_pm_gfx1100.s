.amdgcn_target "amdgcn-amd-amdhsa--gfx1100"
.amdhsa_code_object_version 6
.text
.protected qwen4_slab_union_plan_t8_pm_gfx1100
.globl qwen4_slab_union_plan_t8_pm_gfx1100
.p2align 8
.type qwen4_slab_union_plan_t8_pm_gfx1100,@function
qwen4_slab_union_plan_t8_pm_gfx1100:
	s_load_b256 s[32:39], s[0:1], null
	s_load_b32 s16, s[0:1], 0x20
	s_mov_b32 s17, s15
	s_waitcnt lgkmcnt(0)
	s_cmp_lt_i32 s38, 1
	s_cbranch_scc1 .Lunion_end
	s_cmp_gt_i32 s38, 64
	s_cbranch_scc1 .Lunion_end
	s_cmp_lt_i32 s39, -1
	s_cbranch_scc1 .Lunion_end
	s_cmp_gt_i32 s39, 1
	s_cbranch_scc1 .Lunion_end
	s_cmp_lt_i32 s16, 0
	s_cbranch_scc1 .Lunion_end
	s_lshl_b32 s20, s17, 3
	s_cmp_ge_u32 s17, 8
	s_cbranch_scc1 .Lunion_end
	s_cmp_ge_u32 s20, s38
	s_cbranch_scc1 .Lunion_end
	s_mul_i32 s18, s17, 0x50
	s_mul_i32 s19, s38, 10
	s_mul_i32 s20, s17, 0x788
	s_add_u32 s36, s36, s20
	s_addc_u32 s37, s37, 0
	v_add_nc_u32_e32 v3, s18, v0
	v_mov_b32_e32 v1, -1
	v_mov_b32_e32 v2, -1
	v_cmp_gt_u32_e64 s22, 0x50, v0
	v_cmp_gt_u32_e64 s24, s19, v3
	s_and_b32 s22, s22, s24
	s_and_saveexec_b32 s26, s22
	v_lshlrev_b32_e32 v4, 2, v3
	global_load_b32 v1, v4, s[32:33]
	s_mov_b32 exec_lo, s26
	s_waitcnt vmcnt(0)
	v_cmp_gt_u32_e64 s22, s16, v1
	s_and_saveexec_b32 s26, s22
	s_cmp_ge_i32 s39, 0
	s_cbranch_scc1 .Lunion_uniform
	global_load_u8 v5, v1, s[34:35]
	s_waitcnt vmcnt(0)
	v_cmp_eq_u32_e64 s24, 1, v5
	v_cndmask_b32_e64 v2, 1, 0, s24
	s_branch .Lunion_bucket_done
	.Lunion_uniform:
	v_mov_b32_e32 v2, s39
	.Lunion_bucket_done:
	s_mov_b32 exec_lo, s26
	v_cmp_gt_u32_e64 s22, 0x50, v0
	v_lshlrev_b32_e32 v22, 2, v0
	s_and_saveexec_b32 s26, s22
	ds_store_b32 v22, v1 offset:0
	ds_store_b32 v22, v2 offset:320
	s_mov_b32 exec_lo, s26
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mov_b32_e32 v6, 0
	v_mov_b32_e32 v7, 0
	v_mov_b32_e32 v8, 0x50
	s_mov_b32 s21, 0
	.Lunion_matches:
	v_lshlrev_b32_e64 v4, 2, s21
	ds_load_b32 v10, v4 offset:0
	s_waitcnt lgkmcnt(0)
	v_cmp_eq_u32_e64 s22, v1, v10
	v_cmp_gt_u32_e64 s24, s16, v1
	s_and_b32 s22, s22, s24
	v_cndmask_b32_e64 v21, 0, 1, s22
	v_add_nc_u32_e32 v7, v7, v21
	v_min_u32_e64 v20, s21, v8
	v_cndmask_b32_e64 v8, v8, v20, s22
	v_cmp_lt_u32_e64 s24, s21, v0
	s_and_b32 s22, s22, s24
	v_cndmask_b32_e64 v20, 0, 1, s22
	v_add_nc_u32_e32 v6, v6, v20
	s_add_i32 s21, s21, 1
	s_cmp_lt_u32 s21, 0x50
	s_cbranch_scc1 .Lunion_matches
	v_cmp_eq_u32_e64 s22, v0, v8
	v_cndmask_b32_e64 v9, 0, 1, s22
	v_cmp_gt_u32_e64 s22, 0x50, v0
	s_and_saveexec_b32 s26, s22
	ds_store_b32 v22, v9 offset:640
	ds_store_b32 v22, v7 offset:960
	s_mov_b32 exec_lo, s26
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mov_b32_e32 v14, 0
	v_mov_b32_e32 v15, 0
	v_mov_b32_e32 v16, 0
	v_mov_b32_e32 v17, 0
	v_mov_b32_e32 v18, 0
	v_mov_b32_e32 v19, 0
	s_mov_b32 s21, 0
	.Lunion_heads:
	v_lshlrev_b32_e64 v4, 2, s21
	ds_load_b32 v11, v4 offset:320
	ds_load_b32 v12, v4 offset:640
	ds_load_b32 v13, v4 offset:960
	s_waitcnt lgkmcnt(0)
	v_cmp_eq_u32_e64 s22, 0, v11
	v_cmp_eq_u32_e64 s24, 1, v12
	s_and_b32 s22, s22, s24
	v_cndmask_b32_e64 v20, 0, 1, s22
	v_add_nc_u32_e32 v16, v16, v20
	v_cndmask_b32_e64 v20, 0, v13, s22
	v_add_nc_u32_e32 v18, v18, v20
	v_cmp_eq_u32_e64 s22, 1, v11
	v_cmp_eq_u32_e64 s24, 1, v12
	s_and_b32 s22, s22, s24
	v_cndmask_b32_e64 v20, 0, 1, s22
	v_add_nc_u32_e32 v17, v17, v20
	v_cndmask_b32_e64 v20, 0, v13, s22
	v_add_nc_u32_e32 v19, v19, v20
	v_cmp_eq_u32_e64 s22, v2, v11
	v_cmp_eq_u32_e64 s24, 1, v12
	s_and_b32 s22, s22, s24
	v_cmp_lt_u32_e64 s24, s21, v8
	s_and_b32 s22, s22, s24
	v_cndmask_b32_e64 v20, 0, 1, s22
	v_add_nc_u32_e32 v14, v14, v20
	v_cndmask_b32_e64 v20, 0, v13, s22
	v_add_nc_u32_e32 v15, v15, v20
	s_add_i32 s21, s21, 1
	s_cmp_lt_u32 s21, 0x50
	s_cbranch_scc1 .Lunion_heads
	v_cmp_eq_u32_e64 s22, 0, v2
	s_and_saveexec_b32 s26, s22
	v_add_nc_u32_e32 v20, v15, v6
	v_lshlrev_b32_e32 v4, 2, v20
	global_store_b32 v4, v3, s[36:37] offset:1288
	v_cmp_eq_u32_e64 s24, 1, v9
	s_and_saveexec_b32 s20, s24
	s_waitcnt_vscnt null, 0x0
	v_lshlrev_b32_e32 v4, 2, v14
	global_store_b32 v4, v1, s[36:37] offset:0
	s_waitcnt_vscnt null, 0x0
	v_lshlrev_b32_e32 v4, 2, v14
	global_store_b32 v4, v15, s[36:37] offset:640
	s_mov_b32 exec_lo, s20
	s_mov_b32 exec_lo, s26
	v_cmp_ge_u32_e64 s22, v0, v16
	v_cmp_gt_u32_e64 s24, 0x50, v0
	s_and_b32 s22, s22, s24
	v_mov_b32_e32 v5, 0
	s_and_saveexec_b32 s26, s22
	s_waitcnt_vscnt null, 0x0
	v_lshlrev_b32_e32 v4, 2, v0
	global_store_b32 v4, v5, s[36:37] offset:0
	s_waitcnt_vscnt null, 0x0
	v_lshlrev_b32_e32 v4, 2, v0
	global_store_b32 v4, v18, s[36:37] offset:640
	s_mov_b32 exec_lo, s26
	v_cmp_ge_u32_e64 s22, v0, v18
	v_cmp_gt_u32_e64 s24, 0x50, v0
	s_and_b32 s22, s22, s24
	s_and_saveexec_b32 s26, s22
	s_waitcnt_vscnt null, 0x0
	v_lshlrev_b32_e32 v4, 2, v0
	global_store_b32 v4, v5, s[36:37] offset:1288
	s_mov_b32 exec_lo, s26
	v_cmp_eq_u32_e64 s22, 0, v0
	s_and_saveexec_b32 s26, s22
	s_waitcnt_vscnt null, 0x0
	v_lshlrev_b32_e32 v4, 2, v0
	global_store_b32 v4, v18, s[36:37] offset:960
	s_mov_b32 exec_lo, s26
	v_cmp_eq_u32_e64 s22, 1, v2
	s_and_saveexec_b32 s26, s22
	v_add_nc_u32_e32 v20, v15, v6
	s_waitcnt_vscnt null, 0x0
	v_lshlrev_b32_e32 v4, 2, v20
	global_store_b32 v4, v3, s[36:37] offset:1608
	v_cmp_eq_u32_e64 s24, 1, v9
	s_and_saveexec_b32 s20, s24
	s_waitcnt_vscnt null, 0x0
	v_lshlrev_b32_e32 v4, 2, v14
	global_store_b32 v4, v1, s[36:37] offset:320
	s_waitcnt_vscnt null, 0x0
	v_lshlrev_b32_e32 v4, 2, v14
	global_store_b32 v4, v15, s[36:37] offset:964
	s_mov_b32 exec_lo, s20
	s_mov_b32 exec_lo, s26
	v_cmp_ge_u32_e64 s22, v0, v17
	v_cmp_gt_u32_e64 s24, 0x50, v0
	s_and_b32 s22, s22, s24
	v_mov_b32_e32 v5, 0
	s_and_saveexec_b32 s26, s22
	s_waitcnt_vscnt null, 0x0
	v_lshlrev_b32_e32 v4, 2, v0
	global_store_b32 v4, v5, s[36:37] offset:320
	s_waitcnt_vscnt null, 0x0
	v_lshlrev_b32_e32 v4, 2, v0
	global_store_b32 v4, v19, s[36:37] offset:964
	s_mov_b32 exec_lo, s26
	v_cmp_ge_u32_e64 s22, v0, v19
	v_cmp_gt_u32_e64 s24, 0x50, v0
	s_and_b32 s22, s22, s24
	s_and_saveexec_b32 s26, s22
	s_waitcnt_vscnt null, 0x0
	v_lshlrev_b32_e32 v4, 2, v0
	global_store_b32 v4, v5, s[36:37] offset:1608
	s_mov_b32 exec_lo, s26
	v_cmp_eq_u32_e64 s22, 0, v0
	s_and_saveexec_b32 s26, s22
	s_waitcnt_vscnt null, 0x0
	v_lshlrev_b32_e32 v4, 2, v0
	global_store_b32 v4, v19, s[36:37] offset:1284
	s_mov_b32 exec_lo, s26
	s_waitcnt_vscnt null, 0x0
	.Lunion_end:
	s_endpgm
.Lqwen4_slab_union_plan_t8_pm_gfx1100_end:
.size qwen4_slab_union_plan_t8_pm_gfx1100, .Lqwen4_slab_union_plan_t8_pm_gfx1100_end-qwen4_slab_union_plan_t8_pm_gfx1100
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel qwen4_slab_union_plan_t8_pm_gfx1100
	.amdhsa_group_segment_fixed_size 1280
	.amdhsa_private_segment_fixed_size 0
	.amdhsa_kernarg_size 40
	.amdhsa_user_sgpr_count 15
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
	.amdhsa_next_free_vgpr 24
	.amdhsa_next_free_sgpr 40
	.amdhsa_reserve_vcc 0
	.amdhsa_float_round_mode_32 0
	.amdhsa_float_round_mode_16_64 0
	.amdhsa_float_denorm_mode_32 3
	.amdhsa_float_denorm_mode_16_64 3
	.amdhsa_fp16_overflow 0
	.amdhsa_workgroup_processor_mode 1
	.amdhsa_memory_ordered 1
	.amdhsa_forward_progress 1
	.amdhsa_inst_pref_size ((instprefsize(.Lqwen4_slab_union_plan_t8_pm_gfx1100_end-qwen4_slab_union_plan_t8_pm_gfx1100)<<4)&1008)>>4
	.amdhsa_dx10_clamp 1
	.amdhsa_ieee_mode 1
	.amdhsa_shared_vgpr_count 0
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
        .name: topk
        .offset: 0
        .size: 8
        .value_kind: global_buffer
      - .actual_access: read_only
        .address_space: global
        .name: codec
        .offset: 8
        .size: 8
        .value_kind: global_buffer
      - .actual_access: write_only
        .address_space: global
        .name: out
        .offset: 16
        .size: 8
        .value_kind: global_buffer
      - 
        .name: rows
        .offset: 24
        .size: 4
        .value_kind: by_value
      - 
        .name: uniform_bucket
        .offset: 28
        .size: 4
        .value_kind: by_value
      - 
        .name: n_experts
        .offset: 32
        .size: 4
        .value_kind: by_value
    .group_segment_fixed_size: 1280
    .kernarg_segment_align: 8
    .kernarg_segment_size: 40
    .max_flat_workgroup_size: 128
    .name: qwen4_slab_union_plan_t8_pm_gfx1100
    .private_segment_fixed_size: 0
    .sgpr_count: 40
    .sgpr_spill_count: 0
    .symbol: qwen4_slab_union_plan_t8_pm_gfx1100.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count: 24
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target: amdgcn-amd-amdhsa--gfx1100
amdhsa.version: [1, 2]
...
.end_amdgpu_metadata
.text
	.p2alignl 7, 3214868480
	.fill 96, 4, 3214868480
