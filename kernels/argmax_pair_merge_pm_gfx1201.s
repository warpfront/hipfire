.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
.amdhsa_code_object_version 6
.text
.protected argmax_pair_merge_pm_gfx1201
.globl argmax_pair_merge_pm_gfx1201
.p2align 8
.type argmax_pair_merge_pm_gfx1201,@function
argmax_pair_merge_pm_gfx1201:
	s_load_b256 s[8:15], s[0:1], 0x0
	s_load_b128 s[16:19], s[0:1], 0x20
	s_load_b32 s20, s[0:1], 0x30
	s_mov_b32 s21, ttmp9
	v_lshl_add_u32 v1, s21, 8, v0
	s_wait_kmcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e64 s32, s18, v1
	s_and_b32 exec_lo, exec_lo, s32
	s_delay_alu instid0(VALU_DEP_2)
	v_mad_co_u64_u32 v[2:3], null, v1, 4, s[10:11]
	s_delay_alu instid0(VALU_DEP_3)
	v_mad_co_u64_u32 v[6:7], null, v1, 4, s[14:15]
	global_load_b32 v4, v[2:3], off
	global_load_b32 v8, v[6:7], off
	s_wait_loadcnt 0x1
	v_mov_b32_e32 v10, v4
	v_ashrrev_i32_e32 v11, 31, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_mad_co_i64_i32 v[10:11], null, v1, s19, v[10:11]
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b64_e64 v[10:11], 2, v[10:11]
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_u32 v10, s36, s8, v10
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v11, null, s9, v11, s36
	s_wait_loadcnt 0x0
	v_mov_b32_e32 v12, v8
	v_ashrrev_i32_e32 v13, 31, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_mad_co_i64_i32 v[12:13], null, v1, s20, v[12:13]
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b64_e64 v[12:13], 2, v[12:13]
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_u32 v12, s36, s12, v12
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v13, null, s13, v13, s36
	global_load_b32 v14, v[10:11], off
	global_load_b32 v15, v[12:13], off
	s_wait_loadcnt 0x1
	v_cmp_class_f32_e64 s24, v14, 0x1f8
	s_wait_loadcnt 0x0
	v_cmp_class_f32_e64 s26, v15, 0x1f8
	v_cmp_ge_f32_e64 s28, v14, v15
	s_not_b32 s34, s26
	s_or_b32 s30, s34, s28
	s_and_b32 s30, s24, s30
	v_add_nc_u32_e32 v16, s19, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v16, 0, v16, s26
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e64 v17, v16, v4, s30
	v_mad_co_u64_u32 v[18:19], null, v1, 4, s[16:17]
	global_store_b32 v[18:19], v17, off
	s_wait_storecnt 0x0
	.Lpair_end:
	s_endpgm
.Largmax_pair_merge_pm_gfx1201_end:
.size argmax_pair_merge_pm_gfx1201, .Largmax_pair_merge_pm_gfx1201_end-argmax_pair_merge_pm_gfx1201
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel argmax_pair_merge_pm_gfx1201
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
	.amdhsa_inst_pref_size ((instprefsize(.Largmax_pair_merge_pm_gfx1201_end-argmax_pair_merge_pm_gfx1201)<<4)&4080)>>4
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
        .name: local_logits
        .offset: 0
        .size: 8
        .value_kind: global_buffer
      - .actual_access: read_only
        .address_space: global
        .name: local_top1
        .offset: 8
        .size: 8
        .value_kind: global_buffer
      - .actual_access: read_only
        .address_space: global
        .name: peer_logits
        .offset: 16
        .size: 8
        .value_kind: global_buffer
      - .actual_access: read_only
        .address_space: global
        .name: peer_top1
        .offset: 24
        .size: 8
        .value_kind: global_buffer
      - .actual_access: write_only
        .address_space: global
        .name: out_top1
        .offset: 32
        .size: 8
        .value_kind: global_buffer
      - 
        .name: rows
        .offset: 40
        .size: 4
        .value_kind: by_value
      - 
        .name: w0
        .offset: 44
        .size: 4
        .value_kind: by_value
      - 
        .name: w1
        .offset: 48
        .size: 4
        .value_kind: by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 52
    .max_flat_workgroup_size: 256
    .name: argmax_pair_merge_pm_gfx1201
    .private_segment_fixed_size: 0
    .sgpr_count: 40
    .sgpr_spill_count: 0
    .symbol: argmax_pair_merge_pm_gfx1201.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count: 24
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
