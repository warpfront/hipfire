; fold_sched.inc: NPU-column epilogue work scheduler shared by the fold twins and the wide tails (native/build.sh splices
; it with @P@ = kernel prefix, ;@LOADS@ / ;@BODY@ = the loads and the 8 group bodies of one pass, silu or addc).
; Entry state: s[0:1] kernarg pointer, s2 / s3 workgroup id x / y, s21 wave index in the workgroup; every other register
; dead. Kernarg extension (160 B kernarg): s[4:11] c0..c3, s[12:13] flags, s[14:15] ctr slot, s[16:17] h/x base,
; s18 round, s19 J, s20 ldy, s21 waves, s22 NW, s23 half, s24 lgx, s25 lgy, s26 upc, s27 drain workgroups,
; s28 workgroups, s29 mode (bit 0 static, bits 16.. grid x), s30 chw, s31 units per claim (multiple of 8).
; Control: s44 j, s45 drain (0/1) or static (2), s46 polls, s47 wave index, s29 end of the current claim.
; Scratch s32..s43, v1..v16, pass loads v80..v159 + v208..v255; the SiLU DAG owns v32..v39, v64..v71, v160..v207,
; s48..s95, vcc. ctr slot (device memory, zero before the launch): u32 claim[j] at j*128 (own L2 line each),
; status at 512 (2 = flag poll timed out), ready mirror rdy[j] at 576 + 4j (= round once a wave saw flags[j]).
; Dynamic mode: one scan (lanes 0..3 load claim[j] and rdy[j] from L2 in parallel) picks the first command j >= the
; current one that is not exhausted; if its mirror says ready the wave claims, else only wave 0 of a workgroup (or a
; drain wave) reads the host flag (uncached), mirroring it when set. Non-drain waves leave when nothing is ready; the
; last `drain` workgroups by linear id (dispatched last) poll until every command is ready and exhausted. Claims are
; pre-checked by a plain L2 read before the atomic. A pass is 8 wave-units = one whole ief_silu / ief_addc workgroup
; (same index math, lane l of the wave does that workgroup's threads l, l+32, .., l+224), so claims never straddle a
; command.
; Static mode (wide tails): wave q = s2*8 + s47 owns units 8q..8q+7 of the J*upc concatenation.
.L@P@_help:
    s_mov_b32 exec_lo, -1
    s_mov_b32 s47, s21
    s_load_b256 s[4:11], s[0:1], 0x30
    s_load_b256 s[12:19], s[0:1], 0x50
    s_load_b256 s[20:27], s[0:1], 0x70
    s_load_b128 s[28:31], s[0:1], 0x90
    v_mbcnt_lo_u32_b32 v1, -1, 0
    v_mov_b32_e32 v2, 0
    s_mov_b32 s44, 0
    s_mov_b32 s45, 0
    s_mov_b32 s46, 0
    s_waitcnt lgkmcnt(0)
    s_bitcmp1_b32 s29, 0
    s_cbranch_scc1 .L@P@_static
    s_lshr_b32 s32, s29, 16
    s_mul_i32 s32, s32, s3
    s_add_u32 s32, s32, s2
    s_sub_u32 s33, s28, s27
    s_cmp_ge_u32 s32, s33
    s_cselect_b32 s45, 1, 0
.L@P@_next:
    s_mov_b32 exec_lo, 15
    v_lshlrev_b32_e32 v20, 7, v1
    v_lshlrev_b32_e32 v22, 2, v1
    global_load_b32 v21, v20, s[14:15] glc dlc
    global_load_b32 v23, v22, s[14:15] offset:576 glc dlc
    s_waitcnt vmcnt(0)
    v_cmp_gt_u32_e32 vcc_lo, s26, v21
    s_and_b32 s32, vcc_lo, exec_lo
    s_mov_b32 exec_lo, -1
    s_lshl_b32 s33, 1, s19
    s_add_i32 s33, s33, -1
    s_and_b32 s32, s32, s33
    s_lshl_b32 s33, 1, s44
    s_add_i32 s33, s33, -1
    s_and_not1_b32 s32, s32, s33
    s_cmp_eq_u32 s32, 0
    s_cbranch_scc1 .L@P@_done
    s_ctz_i32_b32 s44, s32
    s_nop 3
    v_readlane_b32 s34, v23, s44
    s_cmp_eq_u32 s34, s18
    s_cbranch_scc1 .L@P@_claim
    s_cmp_eq_u32 s45, 0
    s_cbranch_scc0 .L@P@_pollj
    s_cmp_eq_u32 s47, 0
    s_cbranch_scc0 .L@P@_done
.L@P@_pollj:
    s_lshl_b32 s32, s44, 2
    s_add_u32 s34, s12, s32
    s_addc_u32 s35, s13, 0
.L@P@_poll:
    global_load_b32 v4, v2, s[34:35] glc dlc
    s_waitcnt vmcnt(0)
    v_readfirstlane_b32 s32, v4
    s_cmp_eq_u32 s32, s18
    s_cbranch_scc1 .L@P@_seen
    s_cmp_eq_u32 s45, 0
    s_cbranch_scc1 .L@P@_done
    s_add_u32 s46, s46, 1
    s_cmp_lt_u32 s46, 0x200000
    s_cbranch_scc1 .L@P@_poll
    s_mov_b32 exec_lo, 1
    v_mov_b32_e32 v3, 2
    global_store_b32 v2, v3, s[14:15] offset:512
    s_mov_b32 exec_lo, -1
    s_branch .L@P@_done
.L@P@_seen:
    s_lshl_b32 s32, s44, 2
    s_mov_b32 exec_lo, 1
    v_mov_b32_e32 v3, s18
    v_mov_b32_e32 v4, s32
    global_store_b32 v4, v3, s[14:15] offset:576
    s_mov_b32 exec_lo, -1
.L@P@_claim:
    s_lshl_b32 s32, s44, 7
    s_add_u32 s36, s14, s32
    s_addc_u32 s37, s15, 0
    global_load_b32 v4, v2, s[36:37] glc dlc
    s_waitcnt vmcnt(0)
    v_readfirstlane_b32 s33, v4
    s_cmp_ge_u32 s33, s26
    s_cbranch_scc1 .L@P@_exhausted
    s_mov_b32 exec_lo, 1
    v_mov_b32_e32 v3, s31
    global_atomic_add_u32 v4, v2, v3, s[36:37] glc
    s_waitcnt vmcnt(0)
    s_mov_b32 exec_lo, -1
    v_readfirstlane_b32 s33, v4
    s_cmp_ge_u32 s33, s26
    s_cbranch_scc1 .L@P@_exhausted
    s_add_u32 s29, s33, s31
    s_branch .L@P@_work
.L@P@_exhausted:
    s_add_u32 s44, s44, 1
    s_branch .L@P@_next
.L@P@_static:
    s_lshl_b32 s33, s2, 3
    s_add_u32 s33, s33, s47
    s_lshl_b32 s33, s33, 3
.L@P@_div:
    s_cmp_lt_u32 s33, s26
    s_cbranch_scc1 .L@P@_sgo
    s_sub_u32 s33, s33, s26
    s_add_u32 s44, s44, 1
    s_branch .L@P@_div
.L@P@_sgo:
    s_cmp_ge_u32 s44, s19
    s_cbranch_scc1 .L@P@_done
    s_add_u32 s29, s33, 8
    s_mov_b32 s45, 2
.L@P@_work:
    s_cmp_lt_u32 s44, 2
    s_cselect_b64 s[40:41], s[4:5], s[8:9]
    s_cmp_lt_u32 s44, 2
    s_cselect_b64 s[42:43], s[6:7], s[10:11]
    s_bitcmp1_b32 s44, 0
    s_cselect_b64 s[38:39], s[42:43], s[40:41]
    s_lshr_b32 s34, s33, 3
    s_lshl_b32 s35, 1, s24
    s_add_i32 s35, s35, -1
    s_and_b32 s36, s34, s35
    s_lshr_b32 s34, s34, s24
    s_lshl_b32 s35, 1, s25
    s_add_i32 s35, s35, -1
    s_and_b32 s37, s34, s35
    s_lshr_b32 s34, s34, s25
    s_lshr_b32 s40, s36, 1
    s_and_b32 s41, s36, 1
    s_lshr_b32 s42, s37, 1
    s_and_b32 s43, s37, 1
    s_mul_i32 s35, s42, s22
    s_add_u32 s35, s35, s40
    s_lshl_b32 s36, s34, 2
    s_lshl_b32 s37, s43, 1
    s_add_u32 s36, s36, s37
    s_mul_i32 s36, s36, s21
    s_lshl_b32 s37, s35, 1
    s_add_u32 s37, s37, s41
    s_add_u32 s36, s36, s37
    s_lshl_b32 s36, s36, 13
    s_add_u32 s38, s38, s36
    s_addc_u32 s39, s39, 0
    s_and_b32 s37, s34, 1
    s_lshl_b32 s36, s43, 1
    s_add_u32 s36, s36, s37
    s_lshl_b32 s36, s36, 6
    s_lshl_b32 s37, s42, 8
    s_add_u32 s36, s36, s37
    s_lshl_b32 s37, s40, 8
    s_lshr_b32 s35, s34, 1
    s_lshl_b32 s35, s35, 6
    s_add_u32 s37, s37, s35
    s_lshl_b32 s35, s41, 5
    s_add_u32 s37, s37, s35
    s_mul_i32 s35, s44, s30
    s_add_u32 s37, s37, s35
    s_lshl_b32 s35, s23, 14
    s_add_u32 s40, s38, s35
    s_addc_u32 s41, s39, 0
    v_and_b32_e32 v3, 7, v1
    v_bfe_u32 v4, v1, 3, 2
    v_add_nc_u32_e32 v5, s36, v3
    v_mul_lo_u32 v6, v5, s20
    v_lshl_add_u32 v7, v4, 3, s37
    v_add_nc_u32_e32 v6, v6, v7
    v_lshlrev_b32_e32 v6, 2, v6
    s_lshl_b32 s35, s20, 5
    v_add_nc_u32_e32 v8, s35, v6
    v_add_nc_u32_e32 v9, s35, v8
    v_add_nc_u32_e32 v10, s35, v9
    v_add_nc_u32_e32 v12, s35, v10
    v_add_nc_u32_e32 v13, s35, v12
    v_add_nc_u32_e32 v14, s35, v13
    v_add_nc_u32_e32 v15, s35, v14
    v_lshlrev_b32_e32 v11, 5, v1
    v_add_nc_u32_e32 v16, 0x1000, v11
;@LOADS@
;@BODY@
    s_add_u32 s33, s33, 8
    s_cmp_lt_u32 s33, s29
    s_cbranch_scc1 .L@P@_work
    s_cmp_eq_u32 s45, 2
    s_cbranch_scc1 .L@P@_done
    s_branch .L@P@_claim
.L@P@_done:
