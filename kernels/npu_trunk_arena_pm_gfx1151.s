.amdgcn_target "amdgcn-amd-amdhsa--gfx1151"
.amdhsa_code_object_version 6
.text
.protected npu_arena_copy_bf16_f32_pm
.globl npu_arena_copy_bf16_f32_pm
.p2align 8
.type npu_arena_copy_bf16_f32_pm,@function
npu_arena_copy_bf16_f32_pm:
	s_load_b64 s[4:5], s[0:1], null
	s_mov_b32 s6, -1
	s_mov_b32 s7, 0x31004000
	s_load_b64 s[8:9], s[0:1], 0x8
	s_mov_b32 s10, -1
	s_mov_b32 s11, 0x31004000
	s_load_b32 s12, s[0:1], 0x10
	s_load_b32 s13, s[0:1], 0x14
	s_load_b32 s14, s[0:1], 0x18
	s_load_b32 s15, s[0:1], 0x1c
	s_load_b32 s16, s[0:1], 0x20
	s_load_b32 s17, s[0:1], 0x24
	s_lshl_b32 s18, s2, 5
	v_add_nc_u32_e32 v1, s18, v0
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_u32_e64 s19, s16, v1
	s_mov_b32 exec_lo, s19
	s_mul_i32 s18, s3, s12
	s_add_u32 s18, s18, s13
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u32_e32 v2, s18, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v2, 1, v2
	buffer_load_u16 v3, v2, s[4:7], 0 offen
	s_add_u32 s18, s3, s17
	s_mul_i32 s18, s18, s14
	s_add_u32 s18, s18, s15
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v4, s18, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v4, 2, v4
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v3, 16, v3
	buffer_store_b32 v3, v4, s[8:11], 0 offen
	s_waitcnt_vscnt null, 0x0
	s_endpgm
.Lnpu_arena_copy_bf16_f32_pm_end:
.size npu_arena_copy_bf16_f32_pm, .Lnpu_arena_copy_bf16_f32_pm_end-npu_arena_copy_bf16_f32_pm
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel npu_arena_copy_bf16_f32_pm
	.amdhsa_group_segment_fixed_size 0
	.amdhsa_private_segment_fixed_size 0
	.amdhsa_kernarg_size 40
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
	.amdhsa_next_free_vgpr 16
	.amdhsa_next_free_sgpr 24
	.amdhsa_reserve_vcc 0
	.amdhsa_float_round_mode_32 0
	.amdhsa_float_round_mode_16_64 0
	.amdhsa_float_denorm_mode_32 3
	.amdhsa_float_denorm_mode_16_64 3
	.amdhsa_fp16_overflow 0
	.amdhsa_workgroup_processor_mode 1
	.amdhsa_memory_ordered 1
	.amdhsa_forward_progress 1
	.amdhsa_inst_pref_size ((instprefsize(.Lnpu_arena_copy_bf16_f32_pm_end-npu_arena_copy_bf16_f32_pm)<<4)&1008)>>4
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
.text
.protected npu_arena_copy_bf16_bf16_pm
.globl npu_arena_copy_bf16_bf16_pm
.p2align 8
.type npu_arena_copy_bf16_bf16_pm,@function
npu_arena_copy_bf16_bf16_pm:
	s_load_b64 s[4:5], s[0:1], null
	s_mov_b32 s6, -1
	s_mov_b32 s7, 0x31004000
	s_load_b64 s[8:9], s[0:1], 0x8
	s_mov_b32 s10, -1
	s_mov_b32 s11, 0x31004000
	s_load_b32 s12, s[0:1], 0x10
	s_load_b32 s13, s[0:1], 0x14
	s_load_b32 s14, s[0:1], 0x18
	s_load_b32 s15, s[0:1], 0x1c
	s_load_b32 s16, s[0:1], 0x20
	s_load_b32 s17, s[0:1], 0x24
	s_lshl_b32 s18, s2, 5
	v_add_nc_u32_e32 v1, s18, v0
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_u32_e64 s19, s16, v1
	s_mov_b32 exec_lo, s19
	s_mul_i32 s18, s3, s12
	s_add_u32 s18, s18, s13
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u32_e32 v2, s18, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v2, 1, v2
	buffer_load_u16 v3, v2, s[4:7], 0 offen
	s_add_u32 s18, s3, s17
	s_mul_i32 s18, s18, s14
	s_add_u32 s18, s18, s15
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v4, s18, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v4, 1, v4
	s_waitcnt vmcnt(0)
	buffer_store_b16 v3, v4, s[8:11], 0 offen
	s_waitcnt_vscnt null, 0x0
	s_endpgm
.Lnpu_arena_copy_bf16_bf16_pm_end:
.size npu_arena_copy_bf16_bf16_pm, .Lnpu_arena_copy_bf16_bf16_pm_end-npu_arena_copy_bf16_bf16_pm
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel npu_arena_copy_bf16_bf16_pm
	.amdhsa_group_segment_fixed_size 0
	.amdhsa_private_segment_fixed_size 0
	.amdhsa_kernarg_size 40
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
	.amdhsa_next_free_vgpr 16
	.amdhsa_next_free_sgpr 24
	.amdhsa_reserve_vcc 0
	.amdhsa_float_round_mode_32 0
	.amdhsa_float_round_mode_16_64 0
	.amdhsa_float_denorm_mode_32 3
	.amdhsa_float_denorm_mode_16_64 3
	.amdhsa_fp16_overflow 0
	.amdhsa_workgroup_processor_mode 1
	.amdhsa_memory_ordered 1
	.amdhsa_forward_progress 1
	.amdhsa_inst_pref_size ((instprefsize(.Lnpu_arena_copy_bf16_bf16_pm_end-npu_arena_copy_bf16_bf16_pm)<<4)&1008)>>4
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
.text
.protected npu_arena_copy_f32_f32_pm
.globl npu_arena_copy_f32_f32_pm
.p2align 8
.type npu_arena_copy_f32_f32_pm,@function
npu_arena_copy_f32_f32_pm:
	s_load_b64 s[4:5], s[0:1], null
	s_mov_b32 s6, -1
	s_mov_b32 s7, 0x31004000
	s_load_b64 s[8:9], s[0:1], 0x8
	s_mov_b32 s10, -1
	s_mov_b32 s11, 0x31004000
	s_load_b32 s12, s[0:1], 0x10
	s_load_b32 s13, s[0:1], 0x14
	s_load_b32 s14, s[0:1], 0x18
	s_load_b32 s15, s[0:1], 0x1c
	s_load_b32 s16, s[0:1], 0x20
	s_load_b32 s17, s[0:1], 0x24
	s_lshl_b32 s18, s2, 5
	v_add_nc_u32_e32 v1, s18, v0
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_u32_e64 s19, s16, v1
	s_mov_b32 exec_lo, s19
	s_mul_i32 s18, s3, s12
	s_add_u32 s18, s18, s13
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u32_e32 v2, s18, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v2, 2, v2
	buffer_load_b32 v3, v2, s[4:7], 0 offen
	s_add_u32 s18, s3, s17
	s_mul_i32 s18, s18, s14
	s_add_u32 s18, s18, s15
	s_delay_alu instid0(VALU_DEP_4)
	v_add_nc_u32_e32 v4, s18, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v4, 2, v4
	s_waitcnt vmcnt(0)
	buffer_store_b32 v3, v4, s[8:11], 0 offen
	s_waitcnt_vscnt null, 0x0
	s_endpgm
.Lnpu_arena_copy_f32_f32_pm_end:
.size npu_arena_copy_f32_f32_pm, .Lnpu_arena_copy_f32_f32_pm_end-npu_arena_copy_f32_f32_pm
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel npu_arena_copy_f32_f32_pm
	.amdhsa_group_segment_fixed_size 0
	.amdhsa_private_segment_fixed_size 0
	.amdhsa_kernarg_size 40
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
	.amdhsa_next_free_vgpr 16
	.amdhsa_next_free_sgpr 24
	.amdhsa_reserve_vcc 0
	.amdhsa_float_round_mode_32 0
	.amdhsa_float_round_mode_16_64 0
	.amdhsa_float_denorm_mode_32 3
	.amdhsa_float_denorm_mode_16_64 3
	.amdhsa_fp16_overflow 0
	.amdhsa_workgroup_processor_mode 1
	.amdhsa_memory_ordered 1
	.amdhsa_forward_progress 1
	.amdhsa_inst_pref_size ((instprefsize(.Lnpu_arena_copy_f32_f32_pm_end-npu_arena_copy_f32_f32_pm)<<4)&1008)>>4
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
      - .address_space: global
        .name: src
        .offset: 0
        .size: 8
        .value_kind: global_buffer
      - .address_space: global
        .name: dst
        .offset: 8
        .size: 8
        .value_kind: global_buffer
      - 
        .name: src_pitch
        .offset: 16
        .size: 4
        .value_kind: by_value
      - 
        .name: src_col0
        .offset: 20
        .size: 4
        .value_kind: by_value
      - 
        .name: dst_pitch
        .offset: 24
        .size: 4
        .value_kind: by_value
      - 
        .name: dst_col0
        .offset: 28
        .size: 4
        .value_kind: by_value
      - 
        .name: cols
        .offset: 32
        .size: 4
        .value_kind: by_value
      - 
        .name: position_start
        .offset: 36
        .size: 4
        .value_kind: by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 40
    .max_flat_workgroup_size: 32
    .name: npu_arena_copy_bf16_f32_pm
    .private_segment_fixed_size: 0
    .sgpr_count: 24
    .sgpr_spill_count: 0
    .symbol: npu_arena_copy_bf16_f32_pm.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count: 16
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
  - .args:
      - .address_space: global
        .name: src
        .offset: 0
        .size: 8
        .value_kind: global_buffer
      - .address_space: global
        .name: dst
        .offset: 8
        .size: 8
        .value_kind: global_buffer
      - 
        .name: src_pitch
        .offset: 16
        .size: 4
        .value_kind: by_value
      - 
        .name: src_col0
        .offset: 20
        .size: 4
        .value_kind: by_value
      - 
        .name: dst_pitch
        .offset: 24
        .size: 4
        .value_kind: by_value
      - 
        .name: dst_col0
        .offset: 28
        .size: 4
        .value_kind: by_value
      - 
        .name: cols
        .offset: 32
        .size: 4
        .value_kind: by_value
      - 
        .name: position_start
        .offset: 36
        .size: 4
        .value_kind: by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 40
    .max_flat_workgroup_size: 32
    .name: npu_arena_copy_bf16_bf16_pm
    .private_segment_fixed_size: 0
    .sgpr_count: 24
    .sgpr_spill_count: 0
    .symbol: npu_arena_copy_bf16_bf16_pm.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count: 16
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
  - .args:
      - .address_space: global
        .name: src
        .offset: 0
        .size: 8
        .value_kind: global_buffer
      - .address_space: global
        .name: dst
        .offset: 8
        .size: 8
        .value_kind: global_buffer
      - 
        .name: src_pitch
        .offset: 16
        .size: 4
        .value_kind: by_value
      - 
        .name: src_col0
        .offset: 20
        .size: 4
        .value_kind: by_value
      - 
        .name: dst_pitch
        .offset: 24
        .size: 4
        .value_kind: by_value
      - 
        .name: dst_col0
        .offset: 28
        .size: 4
        .value_kind: by_value
      - 
        .name: cols
        .offset: 32
        .size: 4
        .value_kind: by_value
      - 
        .name: position_start
        .offset: 36
        .size: 4
        .value_kind: by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 40
    .max_flat_workgroup_size: 32
    .name: npu_arena_copy_f32_f32_pm
    .private_segment_fixed_size: 0
    .sgpr_count: 24
    .sgpr_spill_count: 0
    .symbol: npu_arena_copy_f32_f32_pm.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count: 16
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target: amdgcn-amd-amdhsa--gfx1151
amdhsa.version: [1, 2]
...
.end_amdgpu_metadata
.text
	.p2alignl 7, 3214868480
	.fill 96, 4, 3214868480
