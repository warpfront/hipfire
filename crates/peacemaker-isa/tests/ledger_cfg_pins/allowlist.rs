// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The exact allowlists of `ledger_cfg_pins.rs`: one entry per target. An entry names the
//! scope, the exact target, the substring of the exact verifier or lift error it covers, and
//! why. `why` says what is established: "MODEL LIMITATION" (the verifier's model, not the
//! code, produces the report), "UNPROVEN" (a reported path nobody has shown feasible or
//! infeasible; not claimed a race), or "UNSUPPORTED INPUT" / a lift refusal (no verification
//! happened). No entry claims a race. Entries are generated from the full output of the
//! exhaustive run; new targets need new entries with their own reason.

/// One allowlist entry: it covers an outcome of `scope` ("pins" or "pools") for exactly
/// `target` whose error contains `needle`.
pub struct Allow {
    pub scope: &'static str,
    pub target: &'static str,
    pub needle: &'static str,
    pub why: &'static str,
}

/// Reports of a missing wait on a CFG path, reviewed. None is a claimed race.
pub const FINDINGS: &[Allow] = &[
    Allow {
        scope: "pins",
        target: "gated_delta_vector_pm_gfx1201.hxaco#causal_conv_depthwise_silu_rows_f32_pm_gfx1201",
        needle: "`v_add_co_u32 v22, vcc_lo, v22, s38` (after label `.LBB76`): needs Store wait 0 (War { reg: \"v[22:23]\" })",
        why: "UNPROVEN, not a claimed race: the ledger keeps a pending VMEM store's source registers (data and address VGPRs) locked until its store-counter wait and reports the first write of one on a CFG path that has no such wait; whether the hardware has read the store sources by then (address captured at issue, vm_vsrc) is not established here and the path was not checked for feasibility",
    },
    Allow {
        scope: "pins",
        target: "gated_delta_vector_pm_gfx1201.hxaco#gated_delta_net_vector_serial_f32_pm_gfx1201",
        needle: "`v_add_co_u32 v25, s3, v25, s14` (after label `.LBB13`): needs Store wait 0 (War { reg: \"v[25:26]\" })",
        why: "UNPROVEN, not a claimed race: the ledger keeps a pending VMEM store's source registers (data and address VGPRs) locked until its store-counter wait and reports the first write of one on a CFG path that has no such wait; whether the hardware has read the store sources by then (address captured at issue, vm_vsrc) is not established here and the path was not checked for feasibility",
    },
    Allow {
        scope: "pins",
        target: "ggml_convert_shim_gfx1201.co#ggml_dequant_iq4_xs_f32_gfx1201",
        needle: "`v_and_b32_e32 v8, 0x7fffffff, v7` (after label `.LBB5`): needs Store wait 1 (War { reg: \"v8\" })",
        why: "UNPROVEN, not a claimed race: the ledger keeps a pending VMEM store's source registers (data and address VGPRs) locked until its store-counter wait and reports the first write of one on a CFG path that has no such wait; whether the hardware has read the store sources by then (address captured at issue, vm_vsrc) is not established here and the path was not checked for feasibility",
    },
    Allow {
        scope: "pins",
        target: "ggml_convert_shim_gfx1201.co#ggml_dequant_q4_k_f16_gfx1201",
        needle: "`global_load_d16_hi_u8 v6, v[3:4], off offset:4` (after label `.LBB5`): needs Load wait 0 (Use { reg: \"v6\" })",
        why: "MODEL LIMITATION, not a claimed race: a d16_hi load whose destination VGPR is also listed as a use (the other half is preserved), so the ledger reads the pending load into the same register as a RAW on its old value; it cannot tell a d16 lo/hi pair from an overwrite, and whether the hardware returns the pair in order is not established by this verifier",
    },
    Allow {
        scope: "pins",
        target: "ggml_convert_shim_gfx1201.co#ggml_dequant_q4_k_f32_gfx1201",
        needle: "`global_load_d16_hi_u8 v5, v[1:2], off offset:4` (after label `.LBB6`): needs Load wait 0 (Use { reg: \"v5\" })",
        why: "MODEL LIMITATION, not a claimed race: a d16_hi load whose destination VGPR is also listed as a use (the other half is preserved), so the ledger reads the pending load into the same register as a RAW on its old value; it cannot tell a d16 lo/hi pair from an overwrite, and whether the hardware returns the pair in order is not established by this verifier",
    },
    Allow {
        scope: "pins",
        target: "ggml_convert_shim_gfx1201.co#ggml_dequant_q5_k_f16_gfx1201",
        needle: "`global_load_d16_hi_u8 v7, v[3:4], off offset:4` (after label `.LBB5`): needs Load wait 0 (Use { reg: \"v7\" })",
        why: "MODEL LIMITATION, not a claimed race: a d16_hi load whose destination VGPR is also listed as a use (the other half is preserved), so the ledger reads the pending load into the same register as a RAW on its old value; it cannot tell a d16 lo/hi pair from an overwrite, and whether the hardware returns the pair in order is not established by this verifier",
    },
    Allow {
        scope: "pins",
        target: "ggml_convert_shim_gfx1201.co#ggml_dequant_q5_k_f32_gfx1201",
        needle: "`global_load_d16_hi_u8 v5, v[3:4], off offset:4` (after label `.LBB6`): needs Load wait 0 (Use { reg: \"v5\" })",
        why: "MODEL LIMITATION, not a claimed race: a d16_hi load whose destination VGPR is also listed as a use (the other half is preserved), so the ledger reads the pending load into the same register as a RAW on its old value; it cannot tell a d16 lo/hi pair from an overwrite, and whether the hardware returns the pair in order is not established by this verifier",
    },
    Allow {
        scope: "pins",
        target: "ggml_convert_shim_gfx1201.co#ggml_dequant_q6_k_f16_gfx1201",
        needle: "`global_load_d16_hi_u8 v0, v[7:8], off offset:32` (after label `.LBB4`): needs Store wait 0 (War { reg: \"v0\" })",
        why: "UNPROVEN, not a claimed race: the ledger keeps a pending VMEM store's source registers (data and address VGPRs) locked until its store-counter wait and reports the first write of one on a CFG path that has no such wait; whether the hardware has read the store sources by then (address captured at issue, vm_vsrc) is not established here and the path was not checked for feasibility",
    },
    Allow {
        scope: "pins",
        target: "ggml_convert_shim_gfx1201.co#ggml_dequant_q6_k_f32_gfx1201",
        needle: "`global_load_d16_hi_u8 v1, v[2:3], off` (after label `.LBB5`): needs Load wait 0 (Use { reg: \"v1\" })",
        why: "MODEL LIMITATION, not a claimed race: a d16_hi load whose destination VGPR is also listed as a use (the other half is preserved), so the ledger reads the pending load into the same register as a RAW on its old value; it cannot tell a d16 lo/hi pair from an overwrite, and whether the hardware returns the pair in order is not established by this verifier",
    },
    Allow {
        scope: "pins",
        target: "ggml_convert_shim_gfx1201.co#ggml_dequant_q8_0_f32_gfx1201",
        needle: "`v_bfe_u32 v2, v5, 16, 1` (after label `.LBB4`): needs Store wait 0 (War { reg: \"v2\" })",
        why: "UNPROVEN, not a claimed race: the ledger keeps a pending VMEM store's source registers (data and address VGPRs) locked until its store-counter wait and reports the first write of one on a CFG path that has no such wait; whether the hardware has read the store sources by then (address captured at issue, vm_vsrc) is not established here and the path was not checked for feasibility",
    },
    Allow {
        scope: "pins",
        target: "ggml_convert_shim_gfx1201.co#ggml_get_row1_q4_k_f32_gfx1201",
        needle: "`global_load_d16_hi_u8 v4, v[1:2], off offset:4` (after label `.LBB2`): needs Load wait 0 (Use { reg: \"v4\" })",
        why: "MODEL LIMITATION, not a claimed race: a d16_hi load whose destination VGPR is also listed as a use (the other half is preserved), so the ledger reads the pending load into the same register as a RAW on its old value; it cannot tell a d16 lo/hi pair from an overwrite, and whether the hardware returns the pair in order is not established by this verifier",
    },
    Allow {
        scope: "pins",
        target: "ggml_convert_shim_gfx1201.co#ggml_get_row1_q5_k_f32_gfx1201",
        needle: "`global_load_d16_hi_u8 v5, v[1:2], off offset:4` (after label `.LBB2`): needs Load wait 0 (Use { reg: \"v5\" })",
        why: "MODEL LIMITATION, not a claimed race: a d16_hi load whose destination VGPR is also listed as a use (the other half is preserved), so the ledger reads the pending load into the same register as a RAW on its old value; it cannot tell a d16 lo/hi pair from an overwrite, and whether the hardware returns the pair in order is not established by this verifier",
    },
    Allow {
        scope: "pins",
        target: "ggml_convert_shim_gfx1201.co#ggml_get_row1_q6_k_f32_gfx1201",
        needle: "`v_sub_co_u32 v2, s3, v0, v1` (after label `.LBB1`): needs Km wait 0 (Waw { reg: \"s[4:6]\" })",
        why: "MODEL LIMITATION, not a claimed race: the IR models the VOP3 carry-out SGPR as a pair (its printer collapses a 2-register sdst to `sN`) although a wave32 instruction writes one SGPR, so the reported WAW with the pending SMEM load of s[4:6] exists only through the pair's second register",
    },
    Allow {
        scope: "pins",
        target: "ggml_convert_shim_gfx1201.co#ggml_get_row_q4_k_f32_gfx1201",
        needle: "`global_load_d16_hi_u8 v4, v[1:2], off offset:4` (after label `.LBB2`): needs Load wait 0 (Use { reg: \"v4\" })",
        why: "MODEL LIMITATION, not a claimed race: a d16_hi load whose destination VGPR is also listed as a use (the other half is preserved), so the ledger reads the pending load into the same register as a RAW on its old value; it cannot tell a d16 lo/hi pair from an overwrite, and whether the hardware returns the pair in order is not established by this verifier",
    },
    Allow {
        scope: "pins",
        target: "ggml_convert_shim_gfx1201.co#ggml_get_row_q5_k_f32_gfx1201",
        needle: "`global_load_d16_hi_u8 v5, v[1:2], off offset:4` (after label `.LBB2`): needs Load wait 0 (Use { reg: \"v5\" })",
        why: "MODEL LIMITATION, not a claimed race: a d16_hi load whose destination VGPR is also listed as a use (the other half is preserved), so the ledger reads the pending load into the same register as a RAW on its old value; it cannot tell a d16 lo/hi pair from an overwrite, and whether the hardware returns the pair in order is not established by this verifier",
    },
    Allow {
        scope: "pins",
        target: "ggml_convert_shim_gfx1201.co#ggml_get_row_q6_k_f32_gfx1201",
        needle: "`global_load_d16_hi_u8 v0, v[4:5], off` (after label `.LBB1`): needs Load wait 1 (Use { reg: \"v0\" })",
        why: "MODEL LIMITATION, not a claimed race: a d16_hi load whose destination VGPR is also listed as a use (the other half is preserved), so the ledger reads the pending load into the same register as a RAW on its old value; it cannot tell a d16 lo/hi pair from an overwrite, and whether the hardware returns the pair in order is not established by this verifier",
    },
    Allow {
        scope: "pins",
        target: "ggml_mmvq_fused_shim_gfx1201.co#ggml_mmvq_q4_k_res_gfx1201",
        needle: "`global_load_d16_hi_b16 v10, v[16:17], off offset:12` (after label `.LBB2`): needs Load wait 0 (Use { reg: \"v10\" })",
        why: "MODEL LIMITATION, not a claimed race: a d16_hi load whose destination VGPR is also listed as a use (the other half is preserved), so the ledger reads the pending load into the same register as a RAW on its old value; it cannot tell a d16 lo/hi pair from an overwrite, and whether the hardware returns the pair in order is not established by this verifier",
    },
    Allow {
        scope: "pins",
        target: "ggml_mmvq_fused_shim_gfx1201.co#ggml_mmvq_q5_k_gfx1201",
        needle: "`global_load_d16_hi_b16 v12, v[18:19], off offset:12` (after label `.LBB2`): needs Load wait 0 (Use { reg: \"v12\" })",
        why: "MODEL LIMITATION, not a claimed race: a d16_hi load whose destination VGPR is also listed as a use (the other half is preserved), so the ledger reads the pending load into the same register as a RAW on its old value; it cannot tell a d16 lo/hi pair from an overwrite, and whether the hardware returns the pair in order is not established by this verifier",
    },
    Allow {
        scope: "pins",
        target: "ggml_mmvq_fused_shim_gfx1201.co#ggml_mmvq_q5_k_res_gfx1201",
        needle: "`global_load_d16_hi_b16 v12, v[18:19], off offset:12` (after label `.LBB2`): needs Load wait 0 (Use { reg: \"v12\" })",
        why: "MODEL LIMITATION, not a claimed race: a d16_hi load whose destination VGPR is also listed as a use (the other half is preserved), so the ledger reads the pending load into the same register as a RAW on its old value; it cannot tell a d16 lo/hi pair from an overwrite, and whether the hardware returns the pair in order is not established by this verifier",
    },
    Allow {
        scope: "pins",
        target: "ggml_mmvq_shim_gfx1201.co#ggml_mmvq_q4_k_gfx1201",
        needle: "`global_load_d16_hi_b16 v10, v[16:17], off offset:12` (after label `.LBB2`): needs Load wait 0 (Use { reg: \"v10\" })",
        why: "MODEL LIMITATION, not a claimed race: a d16_hi load whose destination VGPR is also listed as a use (the other half is preserved), so the ledger reads the pending load into the same register as a RAW on its old value; it cannot tell a d16 lo/hi pair from an overwrite, and whether the hardware returns the pair in order is not established by this verifier",
    },
    Allow {
        scope: "pins",
        target: "gated_delta_chunk_arena_pm_gfx1151.hxaco#gated_delta_chunk_gate_wmma_arena_pm_gfx1151",
        needle: "`v_add_nc_u32_e32 v75, -2, v203` (after label `.LBB24`): needs Lgkm wait 1 (War { reg: \"v[74:77]\" })",
        why: "MODEL LIMITATION, not a claimed race: the ledger keeps a pending DS store's source VGPRs locked until its LGKMcnt wait and reports the first write of one on a CFG path with no such wait. The existing text replay documents that hipcc code reuses DS store sources under a hardware interlock (`ledger_replay::replay_hazards`: \"hardware-interlocked DS-source reuse the strict replay already flags\"), so this stricter-than-hardware lock is the ledger's model; that this exact reuse is interlocked, and that the path is feasible, is not established here",
    },
    Allow {
        scope: "pins",
        target: "gated_delta_chunk_pm_gfx1100.hxaco#gated_delta_chunk_gate_wmma_pm_gfx1100",
        needle: "`v_add_nc_u32_e32 v75, -2, v196` (after label `.LBB24`): needs Lgkm wait 1 (War { reg: \"v[74:77]\" })",
        why: "MODEL LIMITATION, not a claimed race: the ledger keeps a pending DS store's source VGPRs locked until its LGKMcnt wait and reports the first write of one on a CFG path with no such wait. The existing text replay documents that hipcc code reuses DS store sources under a hardware interlock (`ledger_replay::replay_hazards`: \"hardware-interlocked DS-source reuse the strict replay already flags\"), so this stricter-than-hardware lock is the ledger's model; that this exact reuse is interlocked, and that the path is feasible, is not established here",
    },
    Allow {
        scope: "pins",
        target: "gated_delta_chunk_pm_gfx1151.hxaco#gated_delta_chunk_gate_wmma_pm_gfx1151",
        needle: "`v_add_nc_u32_e32 v75, -2, v203` (after label `.LBB24`): needs Lgkm wait 1 (War { reg: \"v[74:77]\" })",
        why: "MODEL LIMITATION, not a claimed race: the ledger keeps a pending DS store's source VGPRs locked until its LGKMcnt wait and reports the first write of one on a CFG path with no such wait. The existing text replay documents that hipcc code reuses DS store sources under a hardware interlock (`ledger_replay::replay_hazards`: \"hardware-interlocked DS-source reuse the strict replay already flags\"), so this stricter-than-hardware lock is the ledger's model; that this exact reuse is interlocked, and that the path is feasible, is not established here",
    },
    Allow {
        scope: "pins",
        target: "gated_delta_chunk_q8_arena_pm_gfx1151.hxaco#gated_delta_chunk_gate_q8_wmma_arena_pm_gfx1151",
        needle: "`v_and_b32_e32 v74, 16, v135` (after label `.LBB29`): needs Lgkm wait 1 (War { reg: \"v[74:77]\" })",
        why: "MODEL LIMITATION, not a claimed race: the ledger keeps a pending DS store's source VGPRs locked until its LGKMcnt wait and reports the first write of one on a CFG path with no such wait. The existing text replay documents that hipcc code reuses DS store sources under a hardware interlock (`ledger_replay::replay_hazards`: \"hardware-interlocked DS-source reuse the strict replay already flags\"), so this stricter-than-hardware lock is the ledger's model; that this exact reuse is interlocked, and that the path is feasible, is not established here",
    },
    Allow {
        scope: "pins",
        target: "gated_delta_chunk_q8_pm_gfx1151.hxaco#gated_delta_chunk_gate_q8_wmma_pm_gfx1151",
        needle: "`v_and_b32_e32 v74, 16, v135` (after label `.LBB29`): needs Lgkm wait 1 (War { reg: \"v[74:77]\" })",
        why: "MODEL LIMITATION, not a claimed race: the ledger keeps a pending DS store's source VGPRs locked until its LGKMcnt wait and reports the first write of one on a CFG path with no such wait. The existing text replay documents that hipcc code reuses DS store sources under a hardware interlock (`ledger_replay::replay_hazards`: \"hardware-interlocked DS-source reuse the strict replay already flags\"), so this stricter-than-hardware lock is the ledger's model; that this exact reuse is interlocked, and that the path is feasible, is not established here",
    },
    Allow {
        scope: "pins",
        target: "gated_delta_vector_pm_gfx1100.hxaco#gated_delta_net_vector_serial_f32_pm_gfx1100",
        needle: "`v_add_co_u32 v25, s3, v25, s26` (after label `.LBB12`): needs Vs wait 0 (War { reg: \"v[25:26]\" })",
        why: "UNPROVEN, not a claimed race: the ledger keeps a pending VMEM store's source registers (data and address VGPRs) locked until its VScnt wait and reports the first write of one on a CFG path that has no such wait; whether the hardware has read the store sources by then is not established here and the path was not checked for feasibility",
    },
    Allow {
        scope: "pins",
        target: "gated_delta_vector_pm_gfx1100.hxaco#causal_conv_depthwise_silu_rows_f32_pm_gfx1100",
        needle: "`v_add_co_u32 v22, vcc_lo, v22, s38` (after label `.LBB76`): needs Vs wait 0 (War { reg: \"v[22:23]\" })",
        why: "UNPROVEN, not a claimed race: the ledger keeps a pending VMEM store's source registers (data and address VGPRs) locked until its VScnt wait and reports the first write of one on a CFG path that has no such wait; whether the hardware has read the store sources by then is not established here and the path was not checked for feasibility",
    },
    Allow {
        scope: "pins",
        target: "npu_trunk_pack_pm_gfx1151.hxaco#npu_trunk_pack_a_bf16_pm",
        needle: "`v_ashrrev_i32_e32 v42, 31, v41` (after label `.LBB24`): needs Lgkm wait 0 (War { reg: \"v42\" })",
        why: "MODEL LIMITATION, not a claimed race: the ledger keeps a pending DS store's source VGPRs locked until its LGKMcnt wait and reports the first write of one on a CFG path with no such wait. The existing text replay documents that hipcc code reuses DS store sources under a hardware interlock (`ledger_replay::replay_hazards`: \"hardware-interlocked DS-source reuse the strict replay already flags\"), so this stricter-than-hardware lock is the ledger's model; that this exact reuse is interlocked, and that the path is feasible, is not established here",
    },
    Allow {
        scope: "pins",
        target: "npu_trunk_pack_pm_gfx1151.hxaco#npu_trunk_pack_a_bf16_k5120_pm",
        needle: "`v_ashrrev_i32_e32 v82, 31, v81` (after label `.LBB44`): needs Lgkm wait 0 (War { reg: \"v82\" })",
        why: "MODEL LIMITATION, not a claimed race: the ledger keeps a pending DS store's source VGPRs locked until its LGKMcnt wait and reports the first write of one on a CFG path with no such wait. The existing text replay documents that hipcc code reuses DS store sources under a hardware interlock (`ledger_replay::replay_hazards`: \"hardware-interlocked DS-source reuse the strict replay already flags\"), so this stricter-than-hardware lock is the ledger's model; that this exact reuse is interlocked, and that the path is feasible, is not established here",
    },
    Allow {
        scope: "pins",
        target: "gemm_mq4g256v2_moe_grouped_k2_pm_gfx1100.hxaco#gemm_mq4g256v2_moe_grouped_wmma_k2_pm_gfx1100",
        needle: "`global_load_b128 v[9:12], v[19:20], off` (after label `.LBB26`): needs Vm wait 0 (Waw { reg: \"v[9:12]\" })",
        why: "MODEL LIMITATION, not a claimed race: this younger global_load_b128 overwrites v[9:12], the destination of an older global_load_b128 v[9:12], v[9:10] (prologue, before the loop) that nothing read in between; two same-class in-order global loads return older first, and `s_waitcnt vmcnt(0) lgkmcnt(1)` follows before any consumer. The pending queue also holds two younger flat loads (flat_load_b128 v[13:16], flat_load_b32 v23), which the ledger models as unordered (in_order false, Vm and Lgkm), so it credits no nonzero vmcnt and asks for Vm wait 0 on the whole queue; the ledger has no representation of an ordered-load destination overwrite. That the path is feasible (loop count above one) and the hardware order of this exact pair are not established here",
    },
    Allow {
        scope: "pins",
        target: "lane_leviathan_hipcc_gfx1200.co#lane_leviathan_verify",
        needle: "`v_dual_mov_b32 v0, 0x7404 :: v_dual_mov_b32 v1, s3` (after label `.LBB7`): needs Ds wait 0 (War { reg: \"v0\" })",
        why: "UNPROVEN, not a claimed race: an original LLVM M7OPEN finding (the committed lane_leviathan_verify image is unchanged compiler output; no modified code is claimed to race). The ledger keeps a pending DS store's source VGPRs locked until its DS-counter wait and reports the first write of one (here the dual mov of v0) on a CFG path from the entry through the not-taken branches pc 30, 42, 67, 72 and 80 that has no such wait; whether the hardware has read the DS store sources by then (store-source timing) is not established here and the path was not checked for feasibility. Nothing here is discharged. The gfx1200 image is the gfx1201 image with only the ELF machine and target name changed, so this is the same instruction stream",
    },
    Allow {
        scope: "pins",
        target: "lane_leviathan_hipcc_gfx1201.co#lane_leviathan_verify",
        needle: "`v_dual_mov_b32 v0, 0x7404 :: v_dual_mov_b32 v1, s3` (after label `.LBB7`): needs Ds wait 0 (War { reg: \"v0\" })",
        why: "UNPROVEN, not a claimed race: an original LLVM M7OPEN finding (the committed lane_leviathan_verify image is unchanged compiler output; no modified code is claimed to race). The ledger keeps a pending DS store's source VGPRs locked until its DS-counter wait and reports the first write of one (here the dual mov of v0) on a CFG path from the entry through the not-taken branches pc 30, 42, 67, 72 and 80 that has no such wait; whether the hardware has read the DS store sources by then (store-source timing) is not established here and the path was not checked for feasibility. Nothing here is discharged",
    },
    Allow {
        scope: "pins",
        target: "lane_leviathan_pm_gfx1200.hxaco#lane_leviathan_verify",
        needle: "`v_dual_mov_b32 v0, 0x7404 :: v_dual_mov_b32 v1, s3` (after label `.LBB7`): needs Ds wait 0 (War { reg: \"v0\" })",
        why: "UNPROVEN, not a claimed race: an original LLVM M7OPEN finding (the committed lane_leviathan_verify image is the same instruction stream as the hipcc image; no modified code is claimed to race). The ledger keeps a pending DS store's source VGPRs locked until its DS-counter wait and reports the first write of one (here the dual mov of v0) on a CFG path from the entry through the not-taken branches pc 30, 42, 67, 72 and 80 that has no such wait; whether the hardware has read the DS store sources by then (store-source timing) is not established here and the path was not checked for feasibility. Nothing here is discharged. The gfx1200 image is the gfx1201 image with only the ELF machine and target name changed",
    },
    Allow {
        scope: "pins",
        target: "lane_leviathan_pm_gfx1201.hxaco#lane_leviathan_verify",
        needle: "`v_dual_mov_b32 v0, 0x7404 :: v_dual_mov_b32 v1, s3` (after label `.LBB7`): needs Ds wait 0 (War { reg: \"v0\" })",
        why: "UNPROVEN, not a claimed race: an original LLVM M7OPEN finding (the committed lane_leviathan_verify image is the same instruction stream as the hipcc image; no modified code is claimed to race). The ledger keeps a pending DS store's source VGPRs locked until its DS-counter wait and reports the first write of one (here the dual mov of v0) on a CFG path from the entry through the not-taken branches pc 30, 42, 67, 72 and 80 that has no such wait; whether the hardware has read the DS store sources by then (store-source timing) is not established here and the path was not checked for feasibility. Nothing here is discharged",
    },
];

/// Inputs with no verification: a verifier refusal or a refusal upstream of it (lift region, pool).
pub const UNSUPPORTED: &[Allow] = &[
    Allow {
        scope: "pools",
        target: "region:ggml_convert_shim_gfx1201.co#ggml_dequant_q8_0_f16_gfx1201",
        needle: "not convertible: `v_pk_mul_f16 v2, v4, v5` does not re-encode to its words [cc104002, 12020b04] (got Ok([cc104",
        why: "lift_region refuses the kernel (the printed instruction does not re-encode to its original words), so no pool region exists; not a verifier finding",
    },
    Allow {
        scope: "pools",
        target: "region:ggml_mmvq_fused_shim_gfx1201.co#ggml_mmvq_iq4_xs_res_gfx1201",
        needle: "not convertible: enables more than one workitem-id axis (packed v0)",
        why: "lift_region refuses the kernel (more than one workitem-id axis is enabled, packed v0), so no pool region exists; not a verifier finding",
    },
    Allow {
        scope: "pools",
        target: "region:ggml_mmvq_fused_shim_gfx1201.co#ggml_mmvq_q4_k_res_gfx1201",
        needle: "not convertible: enables more than one workitem-id axis (packed v0)",
        why: "lift_region refuses the kernel (more than one workitem-id axis is enabled, packed v0), so no pool region exists; not a verifier finding",
    },
    Allow {
        scope: "pools",
        target: "region:ggml_mmvq_fused_shim_gfx1201.co#ggml_mmvq_q5_k_gfx1201",
        needle: "not convertible: enables more than one workitem-id axis (packed v0)",
        why: "lift_region refuses the kernel (more than one workitem-id axis is enabled, packed v0), so no pool region exists; not a verifier finding",
    },
    Allow {
        scope: "pools",
        target: "region:ggml_mmvq_fused_shim_gfx1201.co#ggml_mmvq_q5_k_res_gfx1201",
        needle: "not convertible: enables more than one workitem-id axis (packed v0)",
        why: "lift_region refuses the kernel (more than one workitem-id axis is enabled, packed v0), so no pool region exists; not a verifier finding",
    },
    Allow {
        scope: "pools",
        target: "region:ggml_mmvq_fused_shim_gfx1201.co#ggml_mmvq_q6_k_res_gfx1201",
        needle: "not convertible: enables more than one workitem-id axis (packed v0)",
        why: "lift_region refuses the kernel (more than one workitem-id axis is enabled, packed v0), so no pool region exists; not a verifier finding",
    },
    Allow {
        scope: "pools",
        target: "region:ggml_mmvq_fused_shim_gfx1201.co#ggml_mmvq_q8_0_res_gfx1201",
        needle: "not convertible: enables more than one workitem-id axis (packed v0)",
        why: "lift_region refuses the kernel (more than one workitem-id axis is enabled, packed v0), so no pool region exists; not a verifier finding",
    },
    Allow {
        scope: "pools",
        target: "region:ggml_mmvq_shim_gfx1201.co#ggml_mmvq_iq4_xs_gfx1201",
        needle: "not convertible: enables more than one workitem-id axis (packed v0)",
        why: "lift_region refuses the kernel (more than one workitem-id axis is enabled, packed v0), so no pool region exists; not a verifier finding",
    },
    Allow {
        scope: "pools",
        target: "region:ggml_mmvq_shim_gfx1201.co#ggml_mmvq_q4_k_gfx1201",
        needle: "not convertible: enables more than one workitem-id axis (packed v0)",
        why: "lift_region refuses the kernel (more than one workitem-id axis is enabled, packed v0), so no pool region exists; not a verifier finding",
    },
    Allow {
        scope: "pools",
        target: "region:ggml_mmvq_shim_gfx1201.co#ggml_mmvq_q6_k_gfx1201",
        needle: "not convertible: enables more than one workitem-id axis (packed v0)",
        why: "lift_region refuses the kernel (more than one workitem-id axis is enabled, packed v0), so no pool region exists; not a verifier finding",
    },
    Allow {
        scope: "pools",
        target: "region:ggml_mmvq_shim_gfx1201.co#ggml_mmvq_q8_0_gfx1201",
        needle: "not convertible: enables more than one workitem-id axis (packed v0)",
        why: "lift_region refuses the kernel (more than one workitem-id axis is enabled, packed v0), so no pool region exists; not a verifier finding",
    },
    Allow {
        scope: "pools",
        target: "region:qwen4_ep_owner_assign_pinned_pm_gfx1201.hxaco#qwen4_ep_owner_assign_pinned_pm_gfx1201",
        needle: "not convertible: input SGPRs s102/s103 collide with the body's s0..s104",
        why: "lift_region refuses the kernel (the input SGPRs s102/s103 collide with the body's registers), so no pool region exists; not a verifier finding",
    },
    Allow {
        scope: "pools",
        target: "region:qwen4_mq6_x4_pm_gfx1201.hxaco#qwen4_mq6_x4_pm_gfx1201_w16",
        needle: "not convertible: `v_pk_fma_f16 v184, v208, v212, v208` does not re-encode to its words [cc0e60b8, 1743a9d0] (g",
        why: "lift_region refuses the kernel (the printed instruction does not re-encode to its original words), so no pool region exists; not a verifier finding",
    },
    Allow {
        scope: "pools",
        target: "region:qwen4_mq6_x4_pm_gfx1201.hxaco#qwen4_mq6_x4_pm_gfx1201_w4",
        needle: "not convertible: `v_pk_fma_f16 v184, v208, v212, v208` does not re-encode to its words [cc0e60b8, 1743a9d0] (g",
        why: "lift_region refuses the kernel (the printed instruction does not re-encode to its original words), so no pool region exists; not a verifier finding",
    },
    Allow {
        scope: "pools",
        target: "region:qwen4_mq6_x4_pm_gfx1201.hxaco#qwen4_mq6_x4_pm_gfx1201_w8",
        needle: "not convertible: `v_pk_fma_f16 v184, v208, v212, v208` does not re-encode to its words [cc0e60b8, 1743a9d0] (g",
        why: "lift_region refuses the kernel (the printed instruction does not re-encode to its original words), so no pool region exists; not a verifier finding",
    },
    Allow {
        scope: "pools",
        target: "region:e8p_i4_moe_gemv_pm_gfx1201.co#e8p_i4_moe_gemv_r256_gate_up_t8",
        needle: "not convertible: barrier participation: the paths from the per-wave branch of block 4 (EXEC Full; Scc1 on a per-wave SCC) to its post-dominator (the exit) pass different numbers of barriers {(0, 0), (0, 2)}; a 8-wave region would pair a barrier with another wave's dispatcher barrier",
        why: "lift_region refuses the kernel (barrier participation is not provable under its per-wave branches), so no pool region exists; not a verifier finding",
    },
    Allow {
        scope: "pools",
        target: "region:e8p_i4_moe_gemv_pm_gfx1201.co#e8p_i4_moe_gemv_r128_down_t8",
        needle: "not convertible: barrier participation: the paths from the per-wave branch of block 4 (EXEC Full; Scc1 on a per-wave SCC) to its post-dominator (the exit) pass different numbers of barriers {(0, 0), (0, 2)}; a 8-wave region would pair a barrier with another wave's dispatcher barrier",
        why: "lift_region refuses the kernel (barrier participation is not provable under its per-wave branches), so no pool region exists; not a verifier finding",
    },
    Allow {
        scope: "pools",
        target: "region:e8p_i4_moe_gemv_pm_gfx1201.hxaco#e8p_i4_moe_gemv_r256_gate_up_t8",
        needle: "not convertible: barrier participation: the paths from the per-wave branch of block 4 (EXEC Full; Scc1 on a per-wave SCC) to its post-dominator (the exit) pass different numbers of barriers {(0, 0), (0, 2)}; a 8-wave region would pair a barrier with another wave's dispatcher barrier",
        why: "lift_region refuses the kernel (barrier participation is not provable under its per-wave branches), so no pool region exists; not a verifier finding",
    },
    Allow {
        scope: "pools",
        target: "region:e8p_i4_moe_gemv_pm_gfx1201.hxaco#e8p_i4_moe_gemv_r128_down_t8",
        needle: "not convertible: barrier participation: the paths from the per-wave branch of block 4 (EXEC Full; Scc1 on a per-wave SCC) to its post-dominator (the exit) pass different numbers of barriers {(0, 0), (0, 2)}; a 8-wave region would pair a barrier with another wave's dispatcher barrier",
        why: "lift_region refuses the kernel (barrier participation is not provable under its per-wave branches), so no pool region exists; not a verifier finding",
    },
    Allow {
        scope: "pools",
        target: "region:fa_prefill_fp8_pm_gfx1201.hxaco#attention_fp8_e4m3_fa2_gqa_f16_pm_gfx1201",
        needle: "not convertible: barrier participation: block 8 repeats a barrier on a cycle under the per-wave branch of block 5 (EXEC Full; Scc1 on a per-wave SCC)",
        why: "lift_region refuses the kernel (barrier participation is not provable under its per-wave branches), so no pool region exists; not a verifier finding",
    },
    Allow {
        scope: "pools",
        target: "region:fa_prefill_pm_gfx1201.hxaco#attention_q8_0_fa2_gqa_pm_gfx1201",
        needle: "not convertible: barrier participation: block 8 repeats a barrier on a cycle under the per-wave branch of block 5 (EXEC Full; Scc1 on a per-wave SCC)",
        why: "lift_region refuses the kernel (barrier participation is not provable under its per-wave branches), so no pool region exists; not a verifier finding",
    },
    Allow {
        scope: "pools",
        target: "region:fa_qresident_a3b_pm_gfx1201.hxaco#attention_fp8_e4m3_fa2_gqa_qresident_v2_q8_a3b_pm_gfx1201",
        needle: "not convertible: barrier participation: block 61 repeats a barrier on a cycle under the per-wave branch of block 8 (EXEC Full; Scc1 on a per-wave SCC)",
        why: "lift_region refuses the kernel (barrier participation is not provable under its per-wave branches), so no pool region exists; not a verifier finding",
    },
    Allow {
        scope: "pools",
        target: "region:fa_qresident_a3b_pm_gfx1201.hxaco#attention_fp8_e4m3_fa2_gqa_qresident_v2_q8_range_a3b_pm_gfx1201",
        needle: "not convertible: barrier participation: block 61 repeats a barrier on a cycle under the per-wave branch of block 8 (EXEC Full; Scc1 on a per-wave SCC)",
        why: "lift_region refuses the kernel (barrier participation is not provable under its per-wave branches), so no pool region exists; not a verifier finding",
    },
    Allow {
        scope: "pools",
        target: "region:gemm_mq4g256v2_residual_mmq_iu4_gfx12_b1.hxaco#gemm_mq4g256v2_residual_mmq_iu4_qkvzagdn_b1",
        needle: "not convertible: input SGPRs s102/s103 collide with the body's s0..s103",
        why: "lift_region refuses the kernel (the input SGPRs s102/s103 collide with the body's registers), so no pool region exists; not a verifier finding",
    },
    Allow {
        scope: "pools",
        target: "region:gemm_mq4g256v2_residual_mmq_iu4_gfx12_b1s.hxaco#gemm_mq4g256v2_residual_mmq_iu4_qkvzagdn_b1s",
        needle: "not convertible: input SGPRs s102/s103 collide with the body's s0..s103",
        why: "lift_region refuses the kernel (the input SGPRs s102/s103 collide with the body's registers), so no pool region exists; not a verifier finding",
    },
    Allow {
        scope: "pools",
        target: "region:gemm_mq4g256v2_wmma_fp8_gfx12_b1.hxaco#gemm_mq4g256v2_fp8_set_row_b1",
        needle: "not convertible: input SGPRs s102/s103 collide with the body's s0..s104",
        why: "lift_region refuses the kernel (the input SGPRs s102/s103 collide with the body's registers), so no pool region exists; not a verifier finding",
    },
    Allow {
        scope: "pools",
        target: "region:gemm_mq4g256v2_wmma_fp8_gfx12_b1.hxaco#gemm_mq4g256v2_fp8_add_row_b1",
        needle: "not convertible: input SGPRs s102/s103 collide with the body's s0..s104",
        why: "lift_region refuses the kernel (the input SGPRs s102/s103 collide with the body's registers), so no pool region exists; not a verifier finding",
    },
    Allow {
        scope: "pools",
        target: "region:gemm_mq4g256v2_wmma_fp8_gfx12_b1.hxaco#gemm_mq4g256v2_fp8_silu_row_b1",
        needle: "not convertible: input SGPRs s102/s103 collide with the body's s0..s104",
        why: "lift_region refuses the kernel (the input SGPRs s102/s103 collide with the body's registers), so no pool region exists; not a verifier finding",
    },
    Allow {
        scope: "pools",
        target: "region:gemm_mq4g256v2_wmma_fp8_gfx12_b1.hxaco#gemm_mq4g256v2_fp8_qkv_row_b1",
        needle: "not convertible: input SGPRs s102/s103 collide with the body's s0..s104",
        why: "lift_region refuses the kernel (the input SGPRs s102/s103 collide with the body's registers), so no pool region exists; not a verifier finding",
    },
    Allow {
        scope: "pools",
        target: "region:gemm_mq4g256v2_wmma_fp8_gfx12_b1.hxaco#gemm_mq4g256v2_fp8_qkvza_row_b1",
        needle: "not convertible: input SGPRs s102/s103 collide with the body's s0..s104",
        why: "lift_region refuses the kernel (the input SGPRs s102/s103 collide with the body's registers), so no pool region exists; not a verifier finding",
    },
    Allow {
        scope: "pools",
        target: "region:gemm_mq4g256v2_wmma_fp8_gfx12_b1.hxaco#gemm_mq4g256v2_fp8_set_k128_b1",
        needle: "not convertible: input SGPRs s102/s103 collide with the body's s0..s104",
        why: "lift_region refuses the kernel (the input SGPRs s102/s103 collide with the body's registers), so no pool region exists; not a verifier finding",
    },
    Allow {
        scope: "pools",
        target: "region:gemm_mq4g256v2_wmma_fp8_gfx12_b1.hxaco#gemm_mq4g256v2_fp8_add_k128_b1",
        needle: "not convertible: input SGPRs s102/s103 collide with the body's s0..s104",
        why: "lift_region refuses the kernel (the input SGPRs s102/s103 collide with the body's registers), so no pool region exists; not a verifier finding",
    },
    Allow {
        scope: "pools",
        target: "region:gemm_mq4g256v2_wmma_fp8_gfx12_b1.hxaco#gemm_mq4g256v2_fp8_silu_k128_b1",
        needle: "not convertible: input SGPRs s102/s103 collide with the body's s0..s104",
        why: "lift_region refuses the kernel (the input SGPRs s102/s103 collide with the body's registers), so no pool region exists; not a verifier finding",
    },
    Allow {
        scope: "pools",
        target: "region:gemm_mq4g256v2_wmma_fp8_gfx12_b1.hxaco#gemm_mq4g256v2_fp8_qkv_k128_b1",
        needle: "not convertible: input SGPRs s102/s103 collide with the body's s0..s104",
        why: "lift_region refuses the kernel (the input SGPRs s102/s103 collide with the body's registers), so no pool region exists; not a verifier finding",
    },
    Allow {
        scope: "pools",
        target: "region:gemm_mq4g256v2_wmma_fp8_gfx12_b1.hxaco#gemm_mq4g256v2_fp8_qkvza_k128_b1",
        needle: "not convertible: input SGPRs s102/s103 collide with the body's s0..s104",
        why: "lift_region refuses the kernel (the input SGPRs s102/s103 collide with the body's registers), so no pool region exists; not a verifier finding",
    },
    Allow {
        scope: "pools",
        target: "region:gemm_mq4g256v2_wmma_fp8_gfx12_b1.hxaco#gemm_mq4g256v2_fp8_silu_row_b1_bf16",
        needle: "not convertible: input SGPRs s102/s103 collide with the body's s0..s104",
        why: "lift_region refuses the kernel (the input SGPRs s102/s103 collide with the body's registers), so no pool region exists; not a verifier finding",
    },
    Allow {
        scope: "pools",
        target: "region:gemm_mq4g256v2_wmma_fp8_gfx12_b1.hxaco#gemm_mq4g256v2_fp8_qkvzagdn_row_b1",
        needle: "not convertible: input SGPRs s102/s103 collide with the body's s0..s104",
        why: "lift_region refuses the kernel (the input SGPRs s102/s103 collide with the body's registers), so no pool region exists; not a verifier finding",
    },
    Allow {
        scope: "pools",
        target: "region:mq_i4_moe_gemv_pm_gfx1201.co#mq_i4_moe_gemv_mq3i4_r256_gate_up_t8",
        needle: "not convertible: barrier participation: block 25 repeats a barrier on a cycle under the per-wave branch of block 15 (EXEC Full; Vccnz on a per-wave VCC)",
        why: "lift_region refuses the kernel (barrier participation is not provable under its per-wave branches), so no pool region exists; not a verifier finding",
    },
    Allow {
        scope: "pools",
        target: "region:mq_i4_moe_gemv_pm_gfx1201.co#mq_i4_moe_gemv_mq3i4_r128_down_t8",
        needle: "not convertible: barrier participation: block 25 repeats a barrier on a cycle under the per-wave branch of block 15 (EXEC Full; Vccnz on a per-wave VCC)",
        why: "lift_region refuses the kernel (barrier participation is not provable under its per-wave branches), so no pool region exists; not a verifier finding",
    },
    Allow {
        scope: "pools",
        target: "region:mq_i4_moe_gemv_pm_gfx1201.co#mq_i4_moe_gemv_mq4sym_r256_gate_up_t8",
        needle: "not convertible: barrier participation: block 25 repeats a barrier on a cycle under the per-wave branch of block 15 (EXEC Full; Vccnz on a per-wave VCC)",
        why: "lift_region refuses the kernel (barrier participation is not provable under its per-wave branches), so no pool region exists; not a verifier finding",
    },
    Allow {
        scope: "pools",
        target: "region:mq_i4_moe_gemv_pm_gfx1201.co#mq_i4_moe_gemv_mq4sym_r128_down_t8",
        needle: "not convertible: barrier participation: block 25 repeats a barrier on a cycle under the per-wave branch of block 15 (EXEC Full; Vccnz on a per-wave VCC)",
        why: "lift_region refuses the kernel (barrier participation is not provable under its per-wave branches), so no pool region exists; not a verifier finding",
    },
    Allow {
        scope: "pools",
        target: "region:mq_i4_moe_gemv_pm_gfx1201.hxaco#mq_i4_moe_gemv_mq3i4_r256_gate_up_t8",
        needle: "not convertible: barrier participation: block 25 repeats a barrier on a cycle under the per-wave branch of block 15 (EXEC Full; Vccnz on a per-wave VCC)",
        why: "lift_region refuses the kernel (barrier participation is not provable under its per-wave branches), so no pool region exists; not a verifier finding",
    },
    Allow {
        scope: "pools",
        target: "region:mq_i4_moe_gemv_pm_gfx1201.hxaco#mq_i4_moe_gemv_mq3i4_r128_down_t8",
        needle: "not convertible: barrier participation: block 25 repeats a barrier on a cycle under the per-wave branch of block 15 (EXEC Full; Vccnz on a per-wave VCC)",
        why: "lift_region refuses the kernel (barrier participation is not provable under its per-wave branches), so no pool region exists; not a verifier finding",
    },
    Allow {
        scope: "pools",
        target: "region:mq_i4_moe_gemv_pm_gfx1201.hxaco#mq_i4_moe_gemv_mq4sym_r256_gate_up_t8",
        needle: "not convertible: barrier participation: block 25 repeats a barrier on a cycle under the per-wave branch of block 15 (EXEC Full; Vccnz on a per-wave VCC)",
        why: "lift_region refuses the kernel (barrier participation is not provable under its per-wave branches), so no pool region exists; not a verifier finding",
    },
    Allow {
        scope: "pools",
        target: "region:mq_i4_moe_gemv_pm_gfx1201.hxaco#mq_i4_moe_gemv_mq4sym_r128_down_t8",
        needle: "not convertible: barrier participation: block 25 repeats a barrier on a cycle under the per-wave branch of block 15 (EXEC Full; Vccnz on a per-wave VCC)",
        why: "lift_region refuses the kernel (barrier participation is not provable under its per-wave branches), so no pool region exists; not a verifier finding",
    },
    Allow {
        scope: "pools",
        target: "region:qwen4_moe_gu_c8_pm_gfx1201.co#gemm_mq4g256v2p_moe_grouped_top10_g32_r8_nodz_pm_persist",
        needle: "not convertible: input SGPRs s102/s103 collide with the body's s0..s104",
        why: "lift_region refuses the kernel (the input SGPRs s102/s103 collide with the body's registers), so no pool region exists; not a verifier finding",
    },
    Allow {
        scope: "pools",
        target: "region:qwen4_moe_gu_c8_pm_gfx1201.co#gemm_mq4g256v2p_moe_grouped_top10_g32_r4_nodz_pm_persist",
        needle: "not convertible: input SGPRs s102/s103 collide with the body's s0..s104",
        why: "lift_region refuses the kernel (the input SGPRs s102/s103 collide with the body's registers), so no pool region exists; not a verifier finding",
    },
    Allow {
        scope: "pools",
        target: "region:qwen4_moe_gu_c8_pm_gfx1201.hxaco#gemm_mq4g256v2p_moe_grouped_top10_g32_r8_nodz_pm_persist",
        needle: "not convertible: input SGPRs s102/s103 collide with the body's s0..s104",
        why: "lift_region refuses the kernel (the input SGPRs s102/s103 collide with the body's registers), so no pool region exists; not a verifier finding",
    },
    Allow {
        scope: "pools",
        target: "region:qwen4_moe_gu_c8_pm_gfx1201.hxaco#gemm_mq4g256v2p_moe_grouped_top10_g32_r4_nodz_pm_persist",
        needle: "not convertible: input SGPRs s102/s103 collide with the body's s0..s104",
        why: "lift_region refuses the kernel (the input SGPRs s102/s103 collide with the body's registers), so no pool region exists; not a verifier finding",
    },
    Allow {
        scope: "pools",
        target: "region:qwen4_moe_iu4_sym_p_pm_gfx1201.hxaco#qwen4_moe_gate_up_silu_iu4_sym_p_pm_gfx1201_nt4",
        needle: "not convertible: barrier participation: the paths from the per-wave branch of block 5 (EXEC Full; Scc1 on a per-wave SCC) to its post-dominator block 38 pass different numbers of barriers {(0, 0), (2, 2)}; a 4-wave region would pair a barrier with another wave's dispatcher barrier",
        why: "lift_region refuses the kernel (barrier participation is not provable under its per-wave branches), so no pool region exists; not a verifier finding",
    },
    Allow {
        scope: "pools",
        target: "region:qwen4_moe_iu4_sym_p_pm_gfx1201.hxaco#qwen4_moe_gate_up_silu_iu4_sym_p_pm_gfx1201_nt8",
        needle: "not convertible: barrier participation: block 7 repeats a barrier on a cycle under the per-wave branch of block 5 (EXEC Full; Scc1 on a per-wave SCC)",
        why: "lift_region refuses the kernel (barrier participation is not provable under its per-wave branches), so no pool region exists; not a verifier finding",
    },
    Allow {
        scope: "pools",
        target: "region:qwen4_moe_iu4_sym_pm_gfx1201.hxaco#qwen4_moe_gate_up_silu_iu4_sym_pm_gfx1201_nt4",
        needle: "not convertible: barrier participation: the paths from the per-wave branch of block 5 (EXEC Full; Scc1 on a per-wave SCC) to its post-dominator block 38 pass different numbers of barriers {(0, 0), (2, 2)}; a 4-wave region would pair a barrier with another wave's dispatcher barrier",
        why: "lift_region refuses the kernel (barrier participation is not provable under its per-wave branches), so no pool region exists; not a verifier finding",
    },
    Allow {
        scope: "pools",
        target: "region:qwen4_moe_iu4_sym_pm_gfx1201.hxaco#qwen4_moe_gate_up_silu_iu4_sym_pm_gfx1201_nt8",
        needle: "not convertible: barrier participation: block 7 repeats a barrier on a cycle under the per-wave branch of block 5 (EXEC Full; Scc1 on a per-wave SCC)",
        why: "lift_region refuses the kernel (barrier participation is not provable under its per-wave branches), so no pool region exists; not a verifier finding",
    },
    Allow {
        scope: "pools",
        target: "pool:b32:multi",
        needle: "pool refused: no VGPR left to park dispatcher SGPRs",
        why: "ld::pool refuses the multi-template pool (no VGPR is left to park the dispatcher's SGPRs for the hosted bodies combined), so no multi pool exists for this blockDim; every hosted body is still verified in its single-template pool; not a verifier finding",
    },
    Allow {
        scope: "pools",
        target: "region:lane_leviathan_hipcc_gfx1201.co#lane_leviathan_verify",
        needle: "not convertible: uses a private (scratch) segment",
        why: "UNSUPPORTED INPUT: lift_region refuses the kernel (it uses a private (scratch) segment), so no pool region exists and no pool body was verified; not a verifier finding",
    },
    Allow {
        scope: "pools",
        target: "region:lane_leviathan_pm_gfx1201.hxaco#lane_leviathan_verify",
        needle: "not convertible: uses a private (scratch) segment",
        why: "UNSUPPORTED INPUT: lift_region refuses the kernel (it uses a private (scratch) segment), so no pool region exists and no pool body was verified; not a verifier finding",
    },
];

/// How the ledger's report on a hosted body is classified; the same two classes as the
/// `pins` findings it is mapped to (`why` starts "MODEL LIMITATION" or "UNPROVEN").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Category {
    /// The verifier's model, not the code, produces the report.
    ModelLimit,
    /// A path the ledger reports; feasibility and hardware behavior not established.
    /// NOT a proven race.
    Candidate,
}

/// One finding of a hosted body inside a pool, exact: the template id the dispatcher hosts
/// it under, the pc within the body (the index into the lifted region's lines, i.e. the pc
/// of the same line in the standalone kernel as the `pins` scan prints it), the category,
/// and the instruction and need the verifier names. There is no wildcard and no dispatcher
/// entry: a dispatcher instruction is outside every body span and is fatal.
pub struct PoolBody {
    pub tid: u32,
    /// The `pins` target of the hosted kernel (`file#symbol`).
    pub target: &'static str,
    pub body_pc: usize,
    pub category: Category,
    pub insn: &'static str,
    pub need: &'static str,
}

/// Every hosted body whose strict verification reports, one entry per template (the walk
/// reports the first). Each is the pool-hosted copy of an existing `pins` finding (same
/// target, instruction, need and category; the label differs: `.LBBn` vs `.Lt<tid>b_n`),
/// at the same body pc. Template ids are those the single-template pools of the scan
/// `artifact://98021` (observed on the prior `pmprereq/uniform` base, before the rebase onto
/// the current beta with its new hc8 modules) reported: the gfx1201 ggml shim bodies moved
/// from 33-48 to 55-70, templates 27 and 28 are unchanged. X1 then added the committed
/// `argmax_pair_merge_pm_gfx1201.co` and `.hxaco` (one template each, sorted before every
/// hosted entry below), so every id below is the earlier one plus 2 (27-28 -> 29-30, 55-62 ->
/// 57-64, 65-70 -> 67-72). The next run on the rebased tree
/// may enumerate the templates differently; a changed id then fails as an unlisted body
/// finding plus a stale entry here and is fixed by editing the `tid` only (body pc, insn and
/// need are the same). Witnesses: `local://ledger-join-finding.md`. No entry is a proven
/// race, and there is no dispatcher entry.
pub const POOL_BODY: &[PoolBody] = &[
    // pins row 2, W2
    PoolBody {
        tid: 29,
        target: "gated_delta_vector_pm_gfx1201.hxaco#gated_delta_net_vector_serial_f32_pm_gfx1201",
        body_pc: 367,
        category: Category::Candidate,
        insn: "`v_add_co_u32 v25, s3, v25, s14`",
        need: "needs Store wait 0 (War { reg: \"v[25:26]\" })",
    },
    // pins row 1, W1; artifact://97452 pm_pool_28 pc 673
    PoolBody {
        tid: 30,
        target:
            "gated_delta_vector_pm_gfx1201.hxaco#causal_conv_depthwise_silu_rows_f32_pm_gfx1201",
        body_pc: 545,
        category: Category::Candidate,
        insn: "`v_add_co_u32 v22, vcc_lo, v22, s38`",
        need: "needs Store wait 0 (War { reg: \"v[22:23]\" })",
    },
    // pins row 4
    PoolBody {
        tid: 57,
        target: "ggml_convert_shim_gfx1201.co#ggml_dequant_q4_k_f16_gfx1201",
        body_pc: 143,
        category: Category::ModelLimit,
        insn: "`global_load_d16_hi_u8 v6, v[3:4], off offset:4`",
        need: "needs Load wait 0 (Use { reg: \"v6\" })",
    },
    // pins row 6; artifact://97452 pm_pool_34 pc 271
    PoolBody {
        tid: 58,
        target: "ggml_convert_shim_gfx1201.co#ggml_dequant_q5_k_f16_gfx1201",
        body_pc: 143,
        category: Category::ModelLimit,
        insn: "`global_load_d16_hi_u8 v7, v[3:4], off offset:4`",
        need: "needs Load wait 0 (Use { reg: \"v7\" })",
    },
    // pins row 8, W4; artifact://97452 pm_pool_35 pc 323
    PoolBody {
        tid: 59,
        target: "ggml_convert_shim_gfx1201.co#ggml_dequant_q6_k_f16_gfx1201",
        body_pc: 195,
        category: Category::Candidate,
        insn: "`global_load_d16_hi_u8 v0, v[7:8], off offset:32`",
        need: "needs Store wait 0 (War { reg: \"v0\" })",
    },
    // pins row 3, W3
    PoolBody {
        tid: 60,
        target: "ggml_convert_shim_gfx1201.co#ggml_dequant_iq4_xs_f32_gfx1201",
        body_pc: 252,
        category: Category::Candidate,
        insn: "`v_and_b32_e32 v8, 0x7fffffff, v7`",
        need: "needs Store wait 1 (War { reg: \"v8\" })",
    },
    // pins row 5
    PoolBody {
        tid: 61,
        target: "ggml_convert_shim_gfx1201.co#ggml_dequant_q4_k_f32_gfx1201",
        body_pc: 152,
        category: Category::ModelLimit,
        insn: "`global_load_d16_hi_u8 v5, v[1:2], off offset:4`",
        need: "needs Load wait 0 (Use { reg: \"v5\" })",
    },
    // pins row 7; artifact://97452 pm_pool_38 pc 280
    PoolBody {
        tid: 62,
        target: "ggml_convert_shim_gfx1201.co#ggml_dequant_q5_k_f32_gfx1201",
        body_pc: 152,
        category: Category::ModelLimit,
        insn: "`global_load_d16_hi_u8 v5, v[3:4], off offset:4`",
        need: "needs Load wait 0 (Use { reg: \"v5\" })",
    },
    // pins row 9; artifact://97452 pm_pool_39 pc 289
    PoolBody {
        tid: 63,
        target: "ggml_convert_shim_gfx1201.co#ggml_dequant_q6_k_f32_gfx1201",
        body_pc: 161,
        category: Category::ModelLimit,
        insn: "`global_load_d16_hi_u8 v1, v[2:3], off`",
        need: "needs Load wait 0 (Use { reg: \"v1\" })",
    },
    // pins row 10, W5; artifact://97452 pm_pool_40 pc 295
    PoolBody {
        tid: 64,
        target: "ggml_convert_shim_gfx1201.co#ggml_dequant_q8_0_f32_gfx1201",
        body_pc: 167,
        category: Category::Candidate,
        insn: "`v_bfe_u32 v2, v5, 16, 1`",
        need: "needs Store wait 0 (War { reg: \"v2\" })",
    },
    // pins row 14
    PoolBody {
        tid: 67,
        target: "ggml_convert_shim_gfx1201.co#ggml_get_row_q4_k_f32_gfx1201",
        body_pc: 45,
        category: Category::ModelLimit,
        insn: "`global_load_d16_hi_u8 v4, v[1:2], off offset:4`",
        need: "needs Load wait 0 (Use { reg: \"v4\" })",
    },
    // pins row 11
    PoolBody {
        tid: 68,
        target: "ggml_convert_shim_gfx1201.co#ggml_get_row1_q4_k_f32_gfx1201",
        body_pc: 38,
        category: Category::ModelLimit,
        insn: "`global_load_d16_hi_u8 v4, v[1:2], off offset:4`",
        need: "needs Load wait 0 (Use { reg: \"v4\" })",
    },
    // pins row 15; artifact://97452 pm_pool_45 pc 173
    PoolBody {
        tid: 69,
        target: "ggml_convert_shim_gfx1201.co#ggml_get_row_q5_k_f32_gfx1201",
        body_pc: 45,
        category: Category::ModelLimit,
        insn: "`global_load_d16_hi_u8 v5, v[1:2], off offset:4`",
        need: "needs Load wait 0 (Use { reg: \"v5\" })",
    },
    // pins row 12; artifact://97452 pm_pool_46 pc 166
    PoolBody {
        tid: 70,
        target: "ggml_convert_shim_gfx1201.co#ggml_get_row1_q5_k_f32_gfx1201",
        body_pc: 38,
        category: Category::ModelLimit,
        insn: "`global_load_d16_hi_u8 v5, v[1:2], off offset:4`",
        need: "needs Load wait 0 (Use { reg: \"v5\" })",
    },
    // pins row 16; artifact://97452 pm_pool_47 pc 194
    PoolBody {
        tid: 71,
        target: "ggml_convert_shim_gfx1201.co#ggml_get_row_q6_k_f32_gfx1201",
        body_pc: 66,
        category: Category::ModelLimit,
        insn: "`global_load_d16_hi_u8 v0, v[4:5], off`",
        need: "needs Load wait 1 (Use { reg: \"v0\" })",
    },
    // pins row 13; artifact://97452 pm_pool_48 pc 146
    PoolBody {
        tid: 72,
        target: "ggml_convert_shim_gfx1201.co#ggml_get_row1_q6_k_f32_gfx1201",
        body_pc: 18,
        category: Category::ModelLimit,
        insn: "`v_sub_co_u32 v2, s3, v0, v1`",
        need: "needs Km wait 0 (Waw { reg: \"s[4:6]\" })",
    },
];
