    global_store_b128 v5, v[8:11], s[18:19]
    global_store_b128 v5, v[12:15], s[18:19] offset:16
    s_endpgm
.Lief_silu2_end:
.size ief_silu2, .Lief_silu2_end-ief_silu2
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel ief_silu2
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
    .amdhsa_system_sgpr_workgroup_id_z 1
    .amdhsa_system_sgpr_workgroup_info 0
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 72
    .amdhsa_next_free_sgpr 96
    .amdhsa_reserve_vcc 1
    .amdhsa_float_round_mode_32 0
    .amdhsa_float_round_mode_16_64 0
    .amdhsa_float_denorm_mode_32 3
    .amdhsa_float_denorm_mode_16_64 3
    .amdhsa_fp16_overflow 0
    .amdhsa_workgroup_processor_mode 1
    .amdhsa_memory_ordered 1
    .amdhsa_forward_progress 1
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
    .amdhsa_inst_pref_size ((instprefsize(.Lief_silu2_end-ief_silu2)<<4)&1008)>>4
.end_amdhsa_kernel
.text
.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .address_space: global
        .name: c
        .offset: 0
        .size: 8
        .value_kind: global_buffer
      - .address_space: global
        .name: y
        .offset: 8
        .size: 8
        .value_kind: global_buffer
      - .name: ldy
        .offset: 16
        .size: 4
        .value_kind: by_value
      - .name: col0
        .offset: 20
        .size: 4
        .value_kind: by_value
      - .name: waves
        .offset: 24
        .size: 4
        .value_kind: by_value
      - .name: nw
        .offset: 28
        .size: 4
        .value_kind: by_value
      - .name: half
        .offset: 32
        .size: 4
        .value_kind: by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 40
    .max_flat_workgroup_size: 256
    .name: ief_silu2
    .private_segment_fixed_size: 0
    .sgpr_count: 98
    .sgpr_spill_count: 0
    .symbol: ief_silu2.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count: 72
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target: amdgcn-amd-amdhsa--gfx1151
amdhsa.version: [1, 2]
...
.end_amdgpu_metadata
