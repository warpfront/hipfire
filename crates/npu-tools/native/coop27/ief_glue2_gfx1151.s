; PM-native gfx1151 co-op glue v2 (coop27): one coalesced pass over Xq, LDS transpose, wide contiguous chunk stores.
; Assembled by native/build.sh (peacemaker, wt-land-041m). Wave32.
;
; ief_pack2 kernarg 48 B: xq u64 @0, a u64 @8, dcomp u64 @16, T u32 @24, E u32 @28, NW u32 @32, aseg u32 @36, flags u32 @40.
;           Block 128 (LDS 2048 B), grid [E, MW, 8]: workgroup (ep, mw, c) builds the 2048 B code part of the A half chunk of
;           injecting column c (docs/ief15-n0.md §2.1): lane l = 4*(8*mbl + i) + q loads the 16 code bytes kb 4q..4q+3 of
;           token tok(mbl, i) with two 8-byte loads, scatters the four words into LDS at the chunk layout
;           ((mbl*16 + kb)*8 + i)*4, then lane l stores LDS bytes 16l..16l+15 to all NW copies (2048 contiguous bytes per copy,
;           flags bit 0: plain stores, else glc slc dlc). Every lane also stores d(t, ep) to dcomp [E][T] f32 (the four lanes
;           of a token store the same word), so the sidecar pass never rereads Xq. The tails (D, b) are written by ief_tail.
; ief_db2   kernarg 32 B: dcomp u64 @0, dc u64 @8, bc u64 @16, T u32 @24, E u32 @28. ief_db over the compact d (stride 4 B).
; ief_tail  kernarg 48 B: a u64 @0, dc u64 @8, bc u64 @16, T u32 @24, E u32 @28, NW u32 @32, aseg u32 @36, flags u32 @40.
;           Block 32, grid [MW, 8]: lane l = (mbl, i) of column c writes D(ep, t) at byte 2048 + 2l and b(t) at 2112 + 2l of
;           every chunk (ep, nw) of (c, mw).
.amdgcn_target "amdgcn-amd-amdhsa--gfx1151"
.amdhsa_code_object_version 6
.text
.protected ief_pack2
.globl ief_pack2
.p2align 8
.type ief_pack2,@function
ief_pack2:
    s_load_b256 s[16:23], s[0:1], null
    s_load_b128 s[24:27], s[0:1], 0x20
    s_lshr_b32 s5, s4, 2
    s_lshl_b32 s5, s5, 1
    s_and_b32 s6, s4, 1
    s_add_u32 s5, s5, s6
    s_lshr_b32 s6, s4, 1
    s_and_b32 s6, s6, 1
    s_lshl_b32 s7, s3, 8
    s_lshl_b32 s5, s5, 6
    s_add_u32 s7, s7, s5
    s_lshl_b32 s6, s6, 3
    s_add_u32 s7, s7, s6
    s_waitcnt lgkmcnt(0)
    s_mul_i32 s8, s3, s24
    s_mul_i32 s8, s8, s23
    s_add_u32 s8, s8, s2
    s_mulk_i32 s8, 0x880
    s_mul_i32 s9, s4, s25
    s_add_u32 s8, s8, s9
    s_mul_i32 s9, s23, 0x880
    s_add_u32 s10, s18, s8
    s_addc_u32 s11, s19, 0
    s_mul_i32 s12, s2, s22
    v_lshrrev_b32_e32 v1, 2, v0
    v_and_b32_e32 v2, 3, v0
    v_lshrrev_b32_e32 v3, 3, v1
    v_and_b32_e32 v4, 7, v1
    v_lshl_add_u32 v5, v3, 4, v4
    v_add3_u32 v5, v5, s7, s12
    v_mul_u32_u24_e32 v6, 0x48, v5
    v_lshl_add_u32 v7, v2, 4, v6
    global_load_b64 v[10:11], v7, s[16:17] offset:8
    global_load_b64 v[12:13], v7, s[16:17] offset:16
    global_load_b32 v14, v6, s[16:17]
    v_lshlrev_b32_e32 v15, 2, v5
    v_lshlrev_b32_e32 v8, 4, v3
    v_lshl_add_u32 v8, v2, 2, v8
    v_lshl_add_u32 v8, v8, 3, v4
    v_lshlrev_b32_e32 v8, 2, v8
    v_lshlrev_b32_e32 v9, 4, v0
    s_waitcnt vmcnt(0)
    global_store_b32 v15, v14, s[20:21]
    ds_store_b32 v8, v10
    ds_store_b32 v8, v11 offset:32
    ds_store_b32 v8, v12 offset:64
    ds_store_b32 v8, v13 offset:96
    s_waitcnt lgkmcnt(0)
    s_barrier
    ds_load_b128 v[16:19], v9
    s_mov_b32 s28, 0
    s_waitcnt lgkmcnt(0)
    s_bitcmp1_b32 s26, 0
    s_cbranch_scc1 .Lpack2_wb
.Lpack2_nw:
    global_store_b128 v9, v[16:19], s[10:11] glc slc dlc
    s_add_u32 s10, s10, s9
    s_addc_u32 s11, s11, 0
    s_add_u32 s28, s28, 1
    s_cmp_lt_u32 s28, s24
    s_cbranch_scc1 .Lpack2_nw
    s_waitcnt_vscnt null, 0x0
    s_endpgm
.Lpack2_wb:
    global_store_b128 v9, v[16:19], s[10:11]
    s_add_u32 s10, s10, s9
    s_addc_u32 s11, s11, 0
    s_add_u32 s28, s28, 1
    s_cmp_lt_u32 s28, s24
    s_cbranch_scc1 .Lpack2_wb
    s_waitcnt_vscnt null, 0x0
    s_endpgm
.Lief_pack2_end:
.size ief_pack2, .Lief_pack2_end-ief_pack2
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel ief_pack2
    .amdhsa_group_segment_fixed_size 2048
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 48
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
    .amdhsa_next_free_vgpr 20
    .amdhsa_next_free_sgpr 29
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
    .amdhsa_inst_pref_size ((instprefsize(.Lief_pack2_end-ief_pack2)<<4)&1008)>>4
.end_amdhsa_kernel
.text
.protected ief_db2
.globl ief_db2
.p2align 8
.type ief_db2,@function
ief_db2:
    s_load_b256 s[4:11], s[0:1], null
    s_lshl_b32 s12, s2, 8
    v_add_nc_u32_e32 v1, s12, v0
    s_waitcnt lgkmcnt(0)
    s_lshl_b32 s13, s10, 2
    s_lshl_b32 s15, s13, 1
    v_lshlrev_b32_e32 v2, 2, v1
    v_mov_b32_e32 v3, 0
    v_mov_b32_e32 v4, v2
    s_mov_b32 s14, 0
.Ldb2_max:
    v_add_nc_u32_e32 v5, s13, v4
    global_load_b32 v6, v4, s[4:5]
    global_load_b32 v7, v5, s[4:5]
    v_add_nc_u32_e32 v4, s15, v4
    v_add_nc_u32_e32 v5, s15, v5
    global_load_b32 v8, v4, s[4:5]
    global_load_b32 v9, v5, s[4:5]
    v_add_nc_u32_e32 v4, s15, v4
    s_waitcnt vmcnt(0)
    v_and_b32_e32 v6, 0x7fffffff, v6
    v_and_b32_e32 v7, 0x7fffffff, v7
    v_and_b32_e32 v8, 0x7fffffff, v8
    v_and_b32_e32 v9, 0x7fffffff, v9
    v_max_i32_e32 v3, v3, v6
    v_max_i32_e32 v3, v3, v7
    v_max_i32_e32 v3, v3, v8
    v_max_i32_e32 v3, v3, v9
    s_add_u32 s14, s14, 4
    s_cmp_lt_u32 s14, s11
    s_cbranch_scc1 .Ldb2_max
    v_lshrrev_b32_e32 v10, 23, v3
    v_and_b32_e32 v11, 0x7fffff, v3
    v_or_b32_e32 v12, 0x800000, v11
    v_add_nc_u32_e32 v13, 0xffffff81, v10
    v_clz_i32_u32_e32 v14, v11
    v_add_nc_u32_e32 v15, -8, v14
    v_lshlrev_b32_e32 v15, v15, v11
    v_sub_nc_u32_e32 v16, 0xffffff8a, v14
    v_cmp_eq_u32_e32 vcc_lo, 0, v10
    v_cndmask_b32_e32 v12, v12, v15, vcc_lo
    v_cndmask_b32_e32 v13, v13, v16, vcc_lo
    v_cmp_le_u32_e32 vcc_lo, 0xffff00, v12
    v_cndmask_b32_e64 v17, 0, 1, vcc_lo
    v_add3_u32 v13, v13, v17, -14
    v_cmp_eq_u32_e32 vcc_lo, 0, v3
    v_cndmask_b32_e64 v13, v13, 0, vcc_lo
    v_sub_nc_u32_e32 v18, 0, v13
    v_lshlrev_b32_e32 v19, 1, v1
    s_lshl_b32 s16, s10, 1
    v_mov_b32_e32 v4, v2
    s_mov_b32 s14, 0
.Ldb2_mant:
    v_add_nc_u32_e32 v5, s13, v4
    global_load_b32 v6, v4, s[4:5]
    global_load_b32 v7, v5, s[4:5]
    v_add_nc_u32_e32 v4, s15, v4
    s_waitcnt vmcnt(0)
    v_ldexp_f32 v6, v6, v18
    v_ldexp_f32 v7, v7, v18
    v_rndne_f32_e32 v6, v6
    v_rndne_f32_e32 v7, v7
    v_cvt_u32_f32_e32 v6, v6
    v_cvt_u32_f32_e32 v7, v7
    v_add_nc_u32_e32 v20, s16, v19
    global_store_b16 v19, v6, s[6:7]
    global_store_b16 v20, v7, s[6:7]
    v_add_nc_u32_e32 v19, s16, v20
    s_add_u32 s14, s14, 2
    s_cmp_lt_u32 s14, s11
    s_cbranch_scc1 .Ldb2_mant
    v_lshlrev_b32_e32 v21, 1, v1
    global_store_b16 v21, v13, s[8:9]
    s_endpgm
.Lief_db2_end:
.size ief_db2, .Lief_db2_end-ief_db2
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel ief_db2
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
    .amdhsa_next_free_vgpr 22
    .amdhsa_next_free_sgpr 17
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
    .amdhsa_inst_pref_size ((instprefsize(.Lief_db2_end-ief_db2)<<4)&1008)>>4
.end_amdhsa_kernel
.text
.protected ief_tail
.globl ief_tail
.p2align 8
.type ief_tail,@function
ief_tail:
    s_load_b256 s[16:23], s[0:1], null
    s_load_b128 s[24:27], s[0:1], 0x20
    s_lshr_b32 s5, s3, 2
    s_lshl_b32 s5, s5, 1
    s_and_b32 s6, s3, 1
    s_add_u32 s5, s5, s6
    s_lshr_b32 s6, s3, 1
    s_and_b32 s6, s6, 1
    s_lshl_b32 s7, s2, 8
    s_lshl_b32 s5, s5, 6
    s_add_u32 s7, s7, s5
    s_lshl_b32 s6, s6, 3
    s_add_u32 s7, s7, s6
    s_waitcnt lgkmcnt(0)
    s_mul_i32 s8, s2, s24
    s_mul_i32 s8, s8, s23
    s_mulk_i32 s8, 0x880
    s_mul_i32 s9, s3, s25
    s_add_u32 s8, s8, s9
    s_mul_i32 s9, s23, 0x880
    s_add_u32 s10, s16, s8
    s_addc_u32 s11, s17, 0
    s_lshl_b32 s12, s22, 1
    v_lshrrev_b32_e32 v1, 3, v0
    v_and_b32_e32 v2, 7, v0
    v_lshl_add_u32 v3, v1, 4, v2
    v_add_nc_u32_e32 v3, s7, v3
    v_lshlrev_b32_e32 v4, 1, v3
    global_load_u16 v11, v4, s[20:21]
    v_lshlrev_b32_e32 v5, 1, v0
    v_mov_b32_e32 v6, v4
    s_mov_b32 s14, 0
    s_waitcnt vmcnt(0)
.Ltail_ep:
    global_load_u16 v10, v6, s[18:19]
    v_add_nc_u32_e32 v6, s12, v6
    s_mov_b32 s28, s10
    s_mov_b32 s29, s11
    s_mov_b32 s15, 0
    s_waitcnt vmcnt(0)
    s_bitcmp1_b32 s26, 0
    s_cbranch_scc1 .Ltail_wb
.Ltail_nw:
    global_store_b16 v5, v10, s[28:29] offset:2048 glc slc dlc
    global_store_b16 v5, v11, s[28:29] offset:2112 glc slc dlc
    s_add_u32 s28, s28, s9
    s_addc_u32 s29, s29, 0
    s_add_u32 s15, s15, 1
    s_cmp_lt_u32 s15, s24
    s_cbranch_scc1 .Ltail_nw
    s_branch .Ltail_next
.Ltail_wb:
    global_store_b16 v5, v10, s[28:29] offset:2048
    global_store_b16 v5, v11, s[28:29] offset:2112
    s_add_u32 s28, s28, s9
    s_addc_u32 s29, s29, 0
    s_add_u32 s15, s15, 1
    s_cmp_lt_u32 s15, s24
    s_cbranch_scc1 .Ltail_wb
.Ltail_next:
    s_add_u32 s10, s10, 0x880
    s_addc_u32 s11, s11, 0
    s_add_u32 s14, s14, 1
    s_cmp_lt_u32 s14, s23
    s_cbranch_scc1 .Ltail_ep
    s_waitcnt_vscnt null, 0x0
    s_endpgm
.Lief_tail_end:
.size ief_tail, .Lief_tail_end-ief_tail
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel ief_tail
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 48
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
    .amdhsa_next_free_vgpr 12
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
    .amdhsa_inst_pref_size ((instprefsize(.Lief_tail_end-ief_tail)<<4)&1008)>>4
.end_amdhsa_kernel
.text
.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .address_space: global
        .name: xq
        .offset: 0
        .size: 8
        .value_kind: global_buffer
      - .address_space: global
        .name: a
        .offset: 8
        .size: 8
        .value_kind: global_buffer
      - .address_space: global
        .name: dcomp
        .offset: 16
        .size: 8
        .value_kind: global_buffer
      - .name: tokens
        .offset: 24
        .size: 4
        .value_kind: by_value
      - .name: epochs
        .offset: 28
        .size: 4
        .value_kind: by_value
      - .name: nw
        .offset: 32
        .size: 4
        .value_kind: by_value
      - .name: aseg
        .offset: 36
        .size: 4
        .value_kind: by_value
      - .name: flags
        .offset: 40
        .size: 4
        .value_kind: by_value
    .group_segment_fixed_size: 2048
    .kernarg_segment_align: 8
    .kernarg_segment_size: 48
    .max_flat_workgroup_size: 128
    .name: ief_pack2
    .private_segment_fixed_size: 0
    .sgpr_count: 31
    .sgpr_spill_count: 0
    .symbol: ief_pack2.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count: 20
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
  - .args:
      - .address_space: global
        .name: dcomp
        .offset: 0
        .size: 8
        .value_kind: global_buffer
      - .address_space: global
        .name: dc
        .offset: 8
        .size: 8
        .value_kind: global_buffer
      - .address_space: global
        .name: bc
        .offset: 16
        .size: 8
        .value_kind: global_buffer
      - .name: tokens
        .offset: 24
        .size: 4
        .value_kind: by_value
      - .name: epochs
        .offset: 28
        .size: 4
        .value_kind: by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 32
    .max_flat_workgroup_size: 256
    .name: ief_db2
    .private_segment_fixed_size: 0
    .sgpr_count: 19
    .sgpr_spill_count: 0
    .symbol: ief_db2.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count: 22
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
  - .args:
      - .address_space: global
        .name: a
        .offset: 0
        .size: 8
        .value_kind: global_buffer
      - .address_space: global
        .name: dc
        .offset: 8
        .size: 8
        .value_kind: global_buffer
      - .address_space: global
        .name: bc
        .offset: 16
        .size: 8
        .value_kind: global_buffer
      - .name: tokens
        .offset: 24
        .size: 4
        .value_kind: by_value
      - .name: epochs
        .offset: 28
        .size: 4
        .value_kind: by_value
      - .name: nw
        .offset: 32
        .size: 4
        .value_kind: by_value
      - .name: aseg
        .offset: 36
        .size: 4
        .value_kind: by_value
      - .name: flags
        .offset: 40
        .size: 4
        .value_kind: by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 48
    .max_flat_workgroup_size: 32
    .name: ief_tail
    .private_segment_fixed_size: 0
    .sgpr_count: 32
    .sgpr_spill_count: 0
    .symbol: ief_tail.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count: 12
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target: amdgcn-amd-amdhsa--gfx1151
amdhsa.version: [1, 2]
...
.end_amdgpu_metadata
