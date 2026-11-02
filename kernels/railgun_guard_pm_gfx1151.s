.amdgcn_target "amdgcn-amd-amdhsa--gfx1151"
.amdhsa_code_object_version 6
.text
.protected railgun_guard_add_pm
.globl railgun_guard_add_pm
.p2align 8
.type railgun_guard_add_pm,@function
railgun_guard_add_pm:
	s_load_b64 s[4:5], s[0:1], null
	s_waitcnt lgkmcnt(0)
	s_load_b32 s6, s[4:5], null
	s_waitcnt lgkmcnt(0)
	s_cmp_eq_u32 s6, 0
	s_cbranch_scc0 .Lrailgun_guard_add_pm_go
	s_endpgm
	.Lrailgun_guard_add_pm_go:
	s_load_b64 s[8:9], s[0:1], 0x8
	s_mov_b32 s10, -1
	s_mov_b32 s11, 0x31004000
	s_load_b64 s[12:13], s[0:1], 0x10
	s_mov_b32 s14, -1
	s_mov_b32 s15, 0x31004000
	s_load_b32 s16, s[0:1], 0x18
	s_lshl_b32 s17, s2, 5
	v_add_nc_u32_e32 v1, s17, v0
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_u32_e64 s18, s16, v1
	s_mov_b32 exec_lo, s18
	s_delay_alu instid0(VALU_DEP_2)
	v_lshlrev_b32_e32 v1, 2, v1
	buffer_load_b32 v2, v1, s[8:11], 0 offen
	buffer_load_b32 v3, v1, s[12:15], 0 offen
	s_waitcnt vmcnt(0)
	v_add_nc_u32_e32 v2, v2, v3
	buffer_store_b32 v2, v1, s[12:15], 0 offen
	s_waitcnt_vscnt null, 0x0
	s_endpgm
.Lrailgun_guard_add_pm_end:
.size railgun_guard_add_pm, .Lrailgun_guard_add_pm_end-railgun_guard_add_pm
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel railgun_guard_add_pm
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
	.amdhsa_next_free_vgpr 8
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
	.amdhsa_inst_pref_size ((instprefsize(.Lrailgun_guard_add_pm_end-railgun_guard_add_pm)<<4)&1008)>>4
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
.protected railgun_guard_step_pm
.globl railgun_guard_step_pm
.p2align 8
.type railgun_guard_step_pm,@function
railgun_guard_step_pm:
	s_load_b64 s[4:5], s[0:1], null
	s_waitcnt lgkmcnt(0)
	s_load_b32 s6, s[4:5], null
	s_waitcnt lgkmcnt(0)
	s_cmp_eq_u32 s6, 0
	s_cbranch_scc0 .Lrailgun_guard_step_pm_go
	s_endpgm
	.Lrailgun_guard_step_pm_go:
	s_load_b64 s[8:9], s[0:1], 0x8
	s_mov_b32 s10, -1
	s_mov_b32 s11, 0x31004000
	s_load_b32 s16, s[0:1], 0x10
	s_mov_b32 s12, s4
	s_mov_b32 s13, s5
	s_mov_b32 s14, -1
	s_mov_b32 s15, 0x31004000
	s_mov_b32 exec_lo, 1
	v_mov_b32_e32 v1, 0
	s_waitcnt lgkmcnt(0)
	buffer_load_b32 v2, v1, s[8:11], 0 offen
	s_waitcnt vmcnt(0)
	v_cmp_gt_u32_e64 s18, s16, v2
	v_cndmask_b32_e64 v3, 0, 1, s18
	buffer_store_b32 v3, v1, s[12:15], 0 offen
	s_waitcnt_vscnt null, 0x0
	s_endpgm
.Lrailgun_guard_step_pm_end:
.size railgun_guard_step_pm, .Lrailgun_guard_step_pm_end-railgun_guard_step_pm
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel railgun_guard_step_pm
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
	.amdhsa_next_free_vgpr 8
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
	.amdhsa_inst_pref_size ((instprefsize(.Lrailgun_guard_step_pm_end-railgun_guard_step_pm)<<4)&1008)>>4
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
.protected railgun_flag_set_pm
.globl railgun_flag_set_pm
.p2align 8
.type railgun_flag_set_pm,@function
railgun_flag_set_pm:
	s_load_b64 s[4:5], s[0:1], null
	s_load_b64 s[8:9], s[0:1], 0x8
	s_mov_b32 s10, -1
	s_mov_b32 s11, 0x31004000
	s_load_b32 s16, s[0:1], 0x10
	s_waitcnt lgkmcnt(0)
	s_mov_b32 s12, s4
	s_mov_b32 s13, s5
	s_mov_b32 s14, -1
	s_mov_b32 s15, 0x31004000
	s_mov_b32 exec_lo, 1
	v_mov_b32_e32 v1, 0
	buffer_load_b32 v2, v1, s[8:11], 0 offen
	s_waitcnt vmcnt(0)
	v_cmp_gt_u32_e64 s18, s16, v2
	v_cndmask_b32_e64 v3, 0, 1, s18
	buffer_store_b32 v3, v1, s[12:15], 0 offen
	s_waitcnt_vscnt null, 0x0
	s_endpgm
.Lrailgun_flag_set_pm_end:
.size railgun_flag_set_pm, .Lrailgun_flag_set_pm_end-railgun_flag_set_pm
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel railgun_flag_set_pm
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
	.amdhsa_next_free_vgpr 8
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
	.amdhsa_inst_pref_size ((instprefsize(.Lrailgun_flag_set_pm_end-railgun_flag_set_pm)<<4)&1008)>>4
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
        .name: flag
        .offset: 0
        .size: 8
        .value_kind: global_buffer
      - .address_space: global
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
        .name: n
        .offset: 24
        .size: 4
        .value_kind: by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 32
    .max_flat_workgroup_size: 32
    .name: railgun_guard_add_pm
    .private_segment_fixed_size: 0
    .sgpr_count: 24
    .sgpr_spill_count: 0
    .symbol: railgun_guard_add_pm.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count: 8
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
  - .args:
      - .address_space: global
        .name: flag
        .offset: 0
        .size: 8
        .value_kind: global_buffer
      - .address_space: global
        .name: y
        .offset: 8
        .size: 8
        .value_kind: global_buffer
      - 
        .name: limit
        .offset: 16
        .size: 4
        .value_kind: by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 24
    .max_flat_workgroup_size: 32
    .name: railgun_guard_step_pm
    .private_segment_fixed_size: 0
    .sgpr_count: 24
    .sgpr_spill_count: 0
    .symbol: railgun_guard_step_pm.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count: 8
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
  - .args:
      - .address_space: global
        .name: flag
        .offset: 0
        .size: 8
        .value_kind: global_buffer
      - .address_space: global
        .name: y
        .offset: 8
        .size: 8
        .value_kind: global_buffer
      - 
        .name: limit
        .offset: 16
        .size: 4
        .value_kind: by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 24
    .max_flat_workgroup_size: 32
    .name: railgun_flag_set_pm
    .private_segment_fixed_size: 0
    .sgpr_count: 24
    .sgpr_spill_count: 0
    .symbol: railgun_flag_set_pm.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count: 8
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
