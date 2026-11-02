; PM-native gfx1151 kernels for the dense-27B down co-op prototype (coop27). No vendor assembler.
; Assembled by peacemaker (wt-land-041m peacemaker_isa::native): crates/npu-tools/native/coop27/build.sh.
; Wave32. Host pages shared with the NPU (userptr BO + hipHostRegister) are stored glc slc dlc and loaded glc dlc.
;
; ief_db    kernarg 32 B: xq u64 @0, dc u64 @8, bc u64 @16, T u32 @24, E u32 @28. Block 256, grid T/256 (E % 4 == 0).
;           IEF15 activation sidecars of fold-contract §3 / peacemaker_npu iu4_ief15::encode_act_scales, one lane per token t:
;           max = max_e |d(t,e)| (bit compare), b = smallest exponent with RNE(max * 2^-b) <= 32767 (exponent bits, no
;           log2), D(e,t) = RNE(d(t,e) * 2^-b) (v_ldexp + v_rndne: exact, the product is < 2^15 or rounds to 0).
;           dc: u16 [E][T]; bc: i16 [T]. All-zero tokens: b = 0, D = 0.
; ief_pack  kernarg 48 B: xq u64 @0, a u64 @8, dc u64 @16, bc u64 @24, T u32 @32, E u32 @36, NW u32 @40, aseg u32 @44.
;           Block 96, grid [E, MW, 8]: workgroup (ep, mw, c) writes the 2176 B A half chunk of injecting column c, M-wave
;           mw, epoch ep (docs/ief15-n0.md §2.1, Ief15Layout::pack_in) to all NW waves (mw, nw):
;           a + c*aseg + ((mw*NW + nw)*E + ep)*2176. Lanes 0..63: code block (mbl = l>>4, kb = l&15) = the 4 code bytes
;           kb*4.. of the 8 tokens tok(mbl, i); lanes 64..95: D / b of lane l-64 = (mbl, i). tok(mbl, i) =
;           mw*256 + blk*64 + (2*mbl + hh)*8 + i, blk = 2*(c/4) + c%2, hh = (c/2)%2.
; ief_addc  kernarg 32 B: c u64 @0, x u64 @8, ldx u32 @16, col0 u32 @20, waves u32 @24, NW u32 @28.
;           Block 256, grid [2*NW, 2*MW, 8]: workgroup (nw*2 + h, mw*2 + s, col) reads the 8192 B C tile of core
;           (col, rho = 2s + h) in wave w = mw*NW + nw (Geometry::pair_c_offset) and applies the ADD epilogue
;           x[t*ldx + col0 + r] = RN(x + Y) for its 64 tokens x 32 features; lane l = (mb, nb, i) owns 8 features.
.amdgcn_target "amdgcn-amd-amdhsa--gfx1151"
.amdhsa_code_object_version 6
.text
.protected ief_db
.globl ief_db
.p2align 8
.type ief_db,@function
ief_db:
    s_load_b256 s[4:11], s[0:1], null
    s_lshl_b32 s12, s2, 8
    v_add_nc_u32_e32 v1, s12, v0
    s_waitcnt lgkmcnt(0)
    s_mul_i32 s13, s10, 0x48
    s_lshl_b32 s15, s13, 1
    v_mul_u32_u24_e32 v2, 0x48, v1
    v_mov_b32_e32 v3, 0
    v_mov_b32_e32 v4, v2
    s_mov_b32 s14, 0
.Ldb_max:
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
    s_cbranch_scc1 .Ldb_max
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
.Ldb_mant:
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
    s_cbranch_scc1 .Ldb_mant
    v_lshlrev_b32_e32 v21, 1, v1
    global_store_b16 v21, v13, s[8:9]
    s_endpgm
.Lief_db_end:
.size ief_db, .Lief_db_end-ief_db
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel ief_db
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
    .amdhsa_inst_pref_size ((instprefsize(.Lief_db_end-ief_db)<<4)&1008)>>4
.end_amdhsa_kernel
.text
.protected ief_pack
.globl ief_pack
.p2align 8
.type ief_pack,@function
ief_pack:
    s_load_b256 s[16:23], s[0:1], null
    s_load_b128 s[24:27], s[0:1], 0x20
    s_load_b32 s29, s[0:1], 0x30
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
    s_mul_i32 s8, s3, s26
    s_mul_i32 s8, s8, s25
    s_add_u32 s8, s8, s2
    s_mulk_i32 s8, 0x880
    s_mul_i32 s9, s4, s27
    s_add_u32 s8, s8, s9
    s_mulk_i32 s25, 0x880
    s_mov_b32 s9, s25
    s_add_u32 s10, s18, s8
    s_addc_u32 s11, s19, 0
    s_mul_i32 s12, s2, s24
    v_lshrrev_b32_e32 v1, 5, v0
    v_readfirstlane_b32 s13, v1
    s_mov_b32 s28, 0
    s_cmp_eq_u32 s13, 2
    s_cbranch_scc1 .Lpack_tail
    v_lshrrev_b32_e32 v2, 4, v0
    v_and_b32_e32 v3, 15, v0
    v_lshlrev_b32_e32 v4, 4, v2
    v_add3_u32 v5, v4, s7, s12
    v_mul_u32_u24_e32 v5, 0x48, v5
    v_lshl_add_u32 v5, v3, 2, v5
    v_add_nc_u32_e32 v5, 8, v5
    global_load_b32 v10, v5, s[16:17]
    global_load_b32 v11, v5, s[16:17] offset:72
    global_load_b32 v12, v5, s[16:17] offset:144
    global_load_b32 v13, v5, s[16:17] offset:216
    global_load_b32 v14, v5, s[16:17] offset:288
    global_load_b32 v15, v5, s[16:17] offset:360
    global_load_b32 v16, v5, s[16:17] offset:432
    global_load_b32 v17, v5, s[16:17] offset:504
    v_lshlrev_b32_e32 v6, 5, v0
    s_waitcnt vmcnt(0)
    s_bitcmp1_b32 s29, 0
    s_cbranch_scc1 .Lpack_code_wb
.Lpack_code_nw:
    global_store_b128 v6, v[10:13], s[10:11] glc slc dlc
    global_store_b128 v6, v[14:17], s[10:11] offset:16 glc slc dlc
    s_add_u32 s10, s10, s9
    s_addc_u32 s11, s11, 0
    s_add_u32 s28, s28, 1
    s_cmp_lt_u32 s28, s26
    s_cbranch_scc1 .Lpack_code_nw
    s_waitcnt_vscnt null, 0x0
    s_endpgm
.Lpack_code_wb:
    global_store_b128 v6, v[10:13], s[10:11]
    global_store_b128 v6, v[14:17], s[10:11] offset:16
    s_add_u32 s10, s10, s9
    s_addc_u32 s11, s11, 0
    s_add_u32 s28, s28, 1
    s_cmp_lt_u32 s28, s26
    s_cbranch_scc1 .Lpack_code_wb
    s_waitcnt_vscnt null, 0x0
    s_endpgm
.Lpack_tail:
    v_subrev_nc_u32_e32 v2, 64, v0
    v_lshrrev_b32_e32 v3, 3, v2
    v_and_b32_e32 v4, 7, v2
    v_lshl_add_u32 v5, v3, 4, v4
    v_add_nc_u32_e32 v5, s7, v5
    v_add_nc_u32_e32 v7, s12, v5
    v_lshlrev_b32_e32 v7, 1, v7
    v_lshlrev_b32_e32 v8, 1, v5
    global_load_u16 v10, v7, s[20:21]
    global_load_u16 v11, v8, s[22:23]
    v_lshlrev_b32_e32 v6, 1, v2
    s_waitcnt vmcnt(0)
    s_bitcmp1_b32 s29, 0
    s_cbranch_scc1 .Lpack_tail_wb
.Lpack_tail_nw:
    global_store_b16 v6, v10, s[10:11] offset:2048 glc slc dlc
    global_store_b16 v6, v11, s[10:11] offset:2112 glc slc dlc
    s_add_u32 s10, s10, s9
    s_addc_u32 s11, s11, 0
    s_add_u32 s28, s28, 1
    s_cmp_lt_u32 s28, s26
    s_cbranch_scc1 .Lpack_tail_nw
    s_waitcnt_vscnt null, 0x0
    s_endpgm
.Lpack_tail_wb:
    global_store_b16 v6, v10, s[10:11] offset:2048
    global_store_b16 v6, v11, s[10:11] offset:2112
    s_add_u32 s10, s10, s9
    s_addc_u32 s11, s11, 0
    s_add_u32 s28, s28, 1
    s_cmp_lt_u32 s28, s26
    s_cbranch_scc1 .Lpack_tail_wb
    s_waitcnt_vscnt null, 0x0
    s_endpgm
.Lief_pack_end:
.size ief_pack, .Lief_pack_end-ief_pack
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel ief_pack
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 56
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
    .amdhsa_next_free_vgpr 18
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
    .amdhsa_inst_pref_size ((instprefsize(.Lief_pack_end-ief_pack)<<4)&1008)>>4
.end_amdhsa_kernel
.text
.protected ief_addc
.globl ief_addc
.p2align 8
.type ief_addc,@function
ief_addc:
    s_load_b256 s[16:23], s[0:1], null
    s_lshr_b32 s5, s2, 1
    s_and_b32 s6, s2, 1
    s_lshr_b32 s7, s3, 1
    s_and_b32 s8, s3, 1
    s_waitcnt lgkmcnt(0)
    s_mul_i32 s9, s7, s23
    s_add_u32 s9, s9, s5
    s_lshl_b32 s10, s4, 2
    s_lshl_b32 s11, s8, 1
    s_add_u32 s10, s10, s11
    s_mul_i32 s10, s10, s22
    s_lshl_b32 s11, s9, 1
    s_add_u32 s11, s11, s6
    s_add_u32 s10, s10, s11
    s_lshl_b32 s10, s10, 13
    s_add_u32 s12, s16, s10
    s_addc_u32 s13, s17, 0
    s_and_b32 s11, s4, 1
    s_lshl_b32 s14, s8, 1
    s_add_u32 s14, s14, s11
    s_lshl_b32 s14, s14, 6
    s_lshl_b32 s15, s7, 8
    s_add_u32 s14, s14, s15
    s_lshl_b32 s15, s5, 8
    s_lshr_b32 s11, s4, 1
    s_lshl_b32 s11, s11, 6
    s_add_u32 s15, s15, s11
    s_lshl_b32 s11, s6, 5
    s_add_u32 s15, s15, s11
    s_add_u32 s15, s15, s21
    v_and_b32_e32 v1, 7, v0
    v_bfe_u32 v2, v0, 3, 2
    v_lshrrev_b32_e32 v3, 5, v0
    v_lshl_add_u32 v4, v3, 3, v1
    v_add_nc_u32_e32 v4, s14, v4
    v_mul_lo_u32 v5, v4, s20
    v_lshl_add_u32 v6, v2, 3, s15
    v_add_nc_u32_e32 v5, v5, v6
    v_lshlrev_b32_e32 v5, 2, v5
    v_lshlrev_b32_e32 v7, 5, v0
    global_load_b128 v[8:11], v7, s[12:13] glc dlc
    global_load_b128 v[12:15], v7, s[12:13] offset:16 glc dlc
    global_load_b128 v[16:19], v5, s[18:19]
    global_load_b128 v[20:23], v5, s[18:19] offset:16
    s_waitcnt vmcnt(0)
    v_add_f32_e32 v16, v16, v8
    v_add_f32_e32 v17, v17, v9
    v_add_f32_e32 v18, v18, v10
    v_add_f32_e32 v19, v19, v11
    v_add_f32_e32 v20, v20, v12
    v_add_f32_e32 v21, v21, v13
    v_add_f32_e32 v22, v22, v14
    v_add_f32_e32 v23, v23, v15
    global_store_b128 v5, v[16:19], s[18:19]
    global_store_b128 v5, v[20:23], s[18:19] offset:16
    s_endpgm
.Lief_addc_end:
.size ief_addc, .Lief_addc_end-ief_addc
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel ief_addc
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
    .amdhsa_system_sgpr_workgroup_id_y 1
    .amdhsa_system_sgpr_workgroup_id_z 1
    .amdhsa_system_sgpr_workgroup_info 0
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 24
    .amdhsa_next_free_sgpr 24
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
    .amdhsa_inst_pref_size ((instprefsize(.Lief_addc_end-ief_addc)<<4)&1008)>>4
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
    .name: ief_db
    .private_segment_fixed_size: 0
    .sgpr_count: 19
    .sgpr_spill_count: 0
    .symbol: ief_db.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count: 22
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
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
        .name: dc
        .offset: 16
        .size: 8
        .value_kind: global_buffer
      - .address_space: global
        .name: bc
        .offset: 24
        .size: 8
        .value_kind: global_buffer
      - .name: tokens
        .offset: 32
        .size: 4
        .value_kind: by_value
      - .name: epochs
        .offset: 36
        .size: 4
        .value_kind: by_value
      - .name: nw
        .offset: 40
        .size: 4
        .value_kind: by_value
      - .name: aseg
        .offset: 44
        .size: 4
        .value_kind: by_value
      - .name: flags
        .offset: 48
        .size: 4
        .value_kind: by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 56
    .max_flat_workgroup_size: 96
    .name: ief_pack
    .private_segment_fixed_size: 0
    .sgpr_count: 32
    .sgpr_spill_count: 0
    .symbol: ief_pack.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count: 18
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
  - .args:
      - .address_space: global
        .name: c
        .offset: 0
        .size: 8
        .value_kind: global_buffer
      - .address_space: global
        .name: x
        .offset: 8
        .size: 8
        .value_kind: global_buffer
      - .name: ldx
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
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 32
    .max_flat_workgroup_size: 256
    .name: ief_addc
    .private_segment_fixed_size: 0
    .sgpr_count: 26
    .sgpr_spill_count: 0
    .symbol: ief_addc.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count: 24
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target: amdgcn-amd-amdhsa--gfx1151
amdhsa.version: [1, 2]
...
.end_amdgpu_metadata
