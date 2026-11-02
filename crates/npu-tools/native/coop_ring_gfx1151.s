; PM-native gfx1151 kernels for the GPU side of the NPU persistent ring (docs/npu/railgun-npu.md §3.1, §4). No vendor assembler.
; The ring, A arena and C arena live in HIP VMM host memory allocated UNCACHED (MTYPE UC: no GL2 caching) and shared with
; the NPU through dma-buf. Every load of NPU-written memory uses glc dlc (bypass GL0/GL1), every store glc slc dlc, and a
; store is complete (s_waitcnt_vscnt 0) before anything that depends on its visibility.
; Wave32. Realtime = s_sendmsg_rtn_b64 MSG_RTN_GET_REALTIME (the constant-rate GPU REFCLK counter).
;
; coop_copy        kernarg 32 B: dst u64 @0, src u64 @8, count u64 @16 (16-byte units), stride u64 @24 (total threads).
;                  Grid-stride uint4 copy, block 256: loads glc dlc, stores glc slc dlc. Producer (A) and consumer (C).
; coop_publish     kernarg 40 B: seq_addr u64 @0, done_addr u64 @8, stats u64 @16, seq u32 @24, free_ref u32 @28,
;                  max_iters u32 @32, flags u32 @36 (bit 0: wait done == free_ref before publishing). One lane (block 32,
;                  exec = 1). [free wait] t_start, store seq, vscnt 0, t_pub, poll done == seq (realtime stamped before
;                  every poll load), t_obs. Record (64 B) at stats: t_start u64, t_pub u64, t_obs u64, poll_iters u32,
;                  status u32 (1 ok, 2 done timeout, 3 slot never freed: NOT published), free_iters u32, last_done u32,
;                  seq u32, 0, t_hit_issue u64 (before the load that saw seq), t_miss_issue u64 (before the last load that
;                  did not; t_pub when the first load hit). The done line landed in (t_miss_issue, t_obs].
.amdgcn_target "amdgcn-amd-amdhsa--gfx1151"
.amdhsa_code_object_version 6
.text
.protected coop_copy
.globl coop_copy
.p2align 8
.type coop_copy,@function
coop_copy:
    s_load_b256 s[4:11], s[0:1], null
    s_lshl_b32 s3, s2, 8
    v_add_nc_u32_e32 v0, s3, v0
    v_mov_b32_e32 v1, v0
    v_mov_b32_e32 v2, 0
    s_waitcnt lgkmcnt(0)
.Lcoop_copy_loop:
    v_cmp_gt_u32_e64 s15, s9, v2
    v_cmp_eq_u32_e64 s16, s9, v2
    v_cmp_gt_u32_e64 s17, s8, v1
    s_and_b32 s16, s16, s17
    s_or_b32 s15, s15, s16
    s_and_b32 exec_lo, exec_lo, s15
    s_cbranch_execz .Lcoop_copy_done
    v_lshlrev_b64_e64 v[3:4], 4, v[1:2]
    v_add_co_u32 v5, vcc_lo, s6, v3
    v_add_co_ci_u32_e64 v6, vcc_lo, s7, v4, vcc_lo
    v_add_co_u32 v3, vcc_lo, s4, v3
    v_add_co_ci_u32_e64 v4, vcc_lo, s5, v4, vcc_lo
    global_load_b128 v[7:10], v[5:6], off glc dlc
    s_waitcnt vmcnt(0)
    global_store_b128 v[3:4], v[7:10], off glc slc dlc
    v_add_co_u32 v1, vcc_lo, s10, v1
    v_add_co_ci_u32_e64 v2, vcc_lo, s11, v2, vcc_lo
    s_branch .Lcoop_copy_loop
.Lcoop_copy_done:
    s_waitcnt_vscnt null, 0x0
    s_endpgm
.Lcoop_copy_end:
.size coop_copy, .Lcoop_copy_end-coop_copy
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel coop_copy
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
    .amdhsa_next_free_vgpr 11
    .amdhsa_next_free_sgpr 18
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
    .amdhsa_inst_pref_size ((instprefsize(.Lcoop_copy_end-coop_copy)<<4)&1008)>>4
.end_amdhsa_kernel
.text
.protected coop_publish
.globl coop_publish
.p2align 8
.type coop_publish,@function
coop_publish:
    s_load_b256 s[4:11], s[0:1], null
    s_load_b64 s[12:13], s[0:1], 0x20
    s_mov_b32 exec_lo, 1
    s_mov_b32 s20, 0
    s_mov_b32 s21, 0
    s_mov_b32 s22, 1
    s_mov_b32 s14, 0
    s_mov_b32 s24, 0
    s_mov_b32 s25, 0
    s_mov_b32 s26, 0
    s_mov_b32 s27, 0
    s_mov_b32 s28, 0
    s_mov_b32 s29, 0
    s_mov_b64 s[30:31], 0
    s_mov_b64 s[32:33], 0
    s_waitcnt lgkmcnt(0)
    v_mov_b32_e32 v1, s6
    v_mov_b32_e32 v2, s7
    v_mov_b32_e32 v3, s4
    v_mov_b32_e32 v4, s5
    s_and_b32 s15, s13, 1
    s_cmp_eq_u32 s15, 0
    s_cbranch_scc1 .Lcoop_publish_pub
.Lcoop_publish_free:
    global_load_b32 v5, v[1:2], off glc dlc
    s_waitcnt vmcnt(0)
    v_readfirstlane_b32 s14, v5
    s_add_u32 s20, s20, 1
    s_cmp_eq_u32 s14, s11
    s_cbranch_scc1 .Lcoop_publish_pub
    s_cmp_lt_u32 s20, s12
    s_cbranch_scc1 .Lcoop_publish_free
    s_mov_b32 s22, 3
    s_branch .Lcoop_publish_record
.Lcoop_publish_pub:
    s_sendmsg_rtn_b64 s[24:25], sendmsg(MSG_RTN_GET_REALTIME)
    s_waitcnt lgkmcnt(0)
    v_mov_b32_e32 v5, s10
    global_store_b32 v[3:4], v5, off glc slc dlc
    s_waitcnt_vscnt null, 0x0
    s_sendmsg_rtn_b64 s[26:27], sendmsg(MSG_RTN_GET_REALTIME)
    s_waitcnt lgkmcnt(0)
    s_mov_b64 s[30:31], s[26:27]
.Lcoop_publish_poll:
    s_mov_b64 s[32:33], s[30:31]
    s_sendmsg_rtn_b64 s[30:31], sendmsg(MSG_RTN_GET_REALTIME)
    s_waitcnt lgkmcnt(0)
    global_load_b32 v5, v[1:2], off glc dlc
    s_waitcnt vmcnt(0)
    v_readfirstlane_b32 s14, v5
    s_add_u32 s21, s21, 1
    s_cmp_eq_u32 s14, s10
    s_cbranch_scc1 .Lcoop_publish_seen
    s_cmp_lt_u32 s21, s12
    s_cbranch_scc1 .Lcoop_publish_poll
    s_mov_b32 s22, 2
.Lcoop_publish_seen:
    s_sendmsg_rtn_b64 s[28:29], sendmsg(MSG_RTN_GET_REALTIME)
    s_waitcnt lgkmcnt(0)
.Lcoop_publish_record:
    v_mov_b32_e32 v6, s8
    v_mov_b32_e32 v7, s9
    v_mov_b32_e32 v8, s24
    v_mov_b32_e32 v9, s25
    v_mov_b32_e32 v10, s26
    v_mov_b32_e32 v11, s27
    global_store_b128 v[6:7], v[8:11], off glc slc dlc
    v_mov_b32_e32 v8, s28
    v_mov_b32_e32 v9, s29
    v_mov_b32_e32 v10, s21
    v_mov_b32_e32 v11, s22
    global_store_b128 v[6:7], v[8:11], off offset:16 glc slc dlc
    v_mov_b32_e32 v8, s20
    v_mov_b32_e32 v9, s14
    v_mov_b32_e32 v10, s10
    v_mov_b32_e32 v11, 0
    global_store_b128 v[6:7], v[8:11], off offset:32 glc slc dlc
    v_mov_b32_e32 v8, s30
    v_mov_b32_e32 v9, s31
    v_mov_b32_e32 v10, s32
    v_mov_b32_e32 v11, s33
    global_store_b128 v[6:7], v[8:11], off offset:48 glc slc dlc
    s_waitcnt_vscnt null, 0x0
    s_endpgm
.Lcoop_publish_end:
.size coop_publish, .Lcoop_publish_end-coop_publish
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel coop_publish
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
    .amdhsa_system_sgpr_workgroup_id_y 0
    .amdhsa_system_sgpr_workgroup_id_z 0
    .amdhsa_system_sgpr_workgroup_info 0
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 12
    .amdhsa_next_free_sgpr 34
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
    .amdhsa_inst_pref_size ((instprefsize(.Lcoop_publish_end-coop_publish)<<4)&1008)>>4
.end_amdhsa_kernel
.text
.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .address_space: global
        .name: dst
        .offset: 0
        .size: 8
        .value_kind: global_buffer
      - .address_space: global
        .name: src
        .offset: 8
        .size: 8
        .value_kind: global_buffer
      - .name: count
        .offset: 16
        .size: 8
        .value_kind: by_value
      - .name: stride
        .offset: 24
        .size: 8
        .value_kind: by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 32
    .max_flat_workgroup_size: 256
    .name: coop_copy
    .private_segment_fixed_size: 0
    .sgpr_count: 20
    .sgpr_spill_count: 0
    .symbol: coop_copy.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count: 11
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
  - .args:
      - .address_space: global
        .name: seq_addr
        .offset: 0
        .size: 8
        .value_kind: global_buffer
      - .address_space: global
        .name: done_addr
        .offset: 8
        .size: 8
        .value_kind: global_buffer
      - .address_space: global
        .name: stats
        .offset: 16
        .size: 8
        .value_kind: global_buffer
      - .name: seq
        .offset: 24
        .size: 4
        .value_kind: by_value
      - .name: free_ref
        .offset: 28
        .size: 4
        .value_kind: by_value
      - .name: max_iters
        .offset: 32
        .size: 4
        .value_kind: by_value
      - .name: flags
        .offset: 36
        .size: 4
        .value_kind: by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 40
    .max_flat_workgroup_size: 32
    .name: coop_publish
    .private_segment_fixed_size: 0
    .sgpr_count: 36
    .sgpr_spill_count: 0
    .symbol: coop_publish.kd
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
