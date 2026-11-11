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

/// Shared reasons of the `topk8_rows_mq6g256v2_{hipcc,pm}_gfx1201` rows (one ELF per image, the
/// hipcc anchor and the PM twin carry the same instruction at the same pc in both images).
const TOPK8_ROWS_ST_WHY: &str = "UNPROVEN, not a claimed race: an original LLVM compiler obligation, OPEN (the committed topk8_rows_mq6g256v2 hipcc image is unchanged compiler output; the PM twin carries the same instruction at the same pc; no modified code is claimed to race). The ledger keeps a pending VMEM store's source registers locked until its store-counter wait and reports the first write of one on a CFG path of not-taken/taken branches that has no such wait; whether that path is feasible and whether the hardware has read the store sources by then is not established here and the path was not checked for feasibility. Text identity and hardware outputs do not discharge this strict WAR path in the ledger; nothing here is discharged";
const TOPK8_ROWS_KM_WHY: &str = "MODEL LIMITATION, not a claimed race: the IR models the VOP3 scalar-destination SGPR (here a `v_cmp_*_e64` result) as a pair although a wave32 instruction writes one SGPR, so the reported WAW with the pending Km-counter write of s4 exists only through the pair's second register (the committed topk8_rows_mq6g256v2 hipcc image is unchanged compiler output; the PM twin carries the same instruction at the same pc)";

/// Shared reason of the 32 `moe_shared_down_combine` rows below (16 hipcc originals, 16 PM
/// twins; same instruction and need in both images).
const MOE_SHARED_DOWN_COMBINE_WHY: &str = "UNPROVEN, not a claimed race: an original LLVM compiler obligation, OPEN (the committed moe_shared_down_combine hipcc image is unchanged compiler output; the PM twin carries the same instruction at the same pc; no modified code is claimed to race). The ledger keeps a pending DS store's source VGPRs locked until its DS-counter wait and reports the first write of one on a CFG path of not-taken/taken branches from the entry that has no such wait (pending DS source lock reuse without a wait); whether that path is feasible and whether the hardware has read the DS store sources by then (store-source timing) is not established here. Text identity and hardware outputs (source twin, direct GPU run) do not discharge the ledger; nothing here is discharged";

/// Shared reasons of the 134 `qwen4_lane_wmma_multi` rows below (67 kernels of the whole
/// `qwen4_lane_wmma.gfx1201.hip` ELF, once from the hipcc anchor `.co`, once from the sealed PM
/// twin `.hxaco`; same instruction and need in both images). Original LLVM obligations, OPEN.
const LWMMA_MULTI_DS_WHY: &str = "UNPROVEN, not a claimed race: an original LLVM compiler obligation, OPEN (the committed qwen4_lane_wmma_multi hipcc image is unchanged compiler output; the sealed PM twin is the same ELF; no modified code is claimed to race). The ledger keeps a pending DS store's source VGPRs locked until its DS-counter wait and reports the first write of one on a CFG path of not-taken/taken branches from the entry that has no such wait (pending DS source lock reuse without a wait); whether that path is feasible and whether the hardware has read the DS store sources by then (store-source timing) is not established here. Checked full-ELF byte identity between the hipcc anchor and the sealed twin, and the oracle outputs of the HIP kernels, do not discharge this strict WAR path in the ledger. Nothing here is discharged";
const LWMMA_MULTI_ST_WHY: &str = "UNPROVEN, not a claimed race: an original LLVM compiler obligation, OPEN (the committed qwen4_lane_wmma_multi hipcc image is unchanged compiler output; the sealed PM twin is the same ELF; no modified code is claimed to race). The ledger keeps a pending VMEM store's source registers (data and address VGPRs) locked until its store-counter wait and reports the first write of one on a CFG path that has no such wait; whether the hardware has read the store sources by then (address captured at issue, vm_vsrc) is not established here and the path was not checked for feasibility. Checked full-ELF byte identity does not discharge this strict WAR path in the ledger. Nothing here is discharged";
const LWMMA_MULTI_KM_WHY: &str = "MODEL LIMITATION, not a claimed race: the IR models the VOP3 scalar-destination SGPR (here a `v_cmp_*_e64` result) as a pair although a wave32 instruction writes one SGPR, so the reported WAW with the pending SMEM load of s[2:3] exists only through the pair's second register (the committed qwen4_lane_wmma_multi hipcc image is unchanged compiler output; the sealed PM twin is the same ELF)";

/// Shared reasons of the X8 twin rows (`moe_route_scatter_wide_{hipcc,pm}_gfx1201` and
/// `qwen4_moe_ep_a8_rot_{hipcc,pm}_gfx1201`: whole-ELF seals, hipcc image and twin byte-identical, every
/// row identical in both images).
/// ST1 rot activation-first re-seal: "unchanged compiler output" means the current hipcc
/// anchor is sealed without instruction rewriting, not that it equals the pre-ST1 image.
/// Full device ELF sha256: fa83561ed67bb97b17d873d3f3b939a66b58b32f275f067883edb00f6d3c28d2.
const X8_DS_WHY: &str = "UNPROVEN, not a claimed race: an original LLVM compiler obligation, OPEN (the committed moe_route_scatter_wide hipcc image is unchanged compiler output; the sealed PM twin is the same ELF; no modified code is claimed to race). The ledger keeps a pending DS store's source VGPRs locked until its DS-counter wait and reports the first write of one on a CFG path that has no such wait; whether that path is feasible and whether the hardware has read the DS store sources by then (store-source timing) is not established here. Checked full-ELF byte identity between the hipcc anchor and the sealed twin does not discharge this strict WAR path in the ledger; nothing here is discharged";
const X8_ST_WHY: &str = "UNPROVEN, not a claimed race: an original LLVM compiler obligation, OPEN (the committed moe_route_scatter_wide hipcc image is unchanged compiler output; the sealed PM twin is the same ELF; no modified code is claimed to race). The ledger keeps a pending VMEM store's source registers (data and address VGPRs) locked until its store-counter wait and reports the first write of one on a CFG path that has no such wait; whether the hardware has read the store sources by then (address captured at issue, vm_vsrc) is not established here and the path was not checked for feasibility. Checked full-ELF byte identity does not discharge this strict WAR path in the ledger; nothing here is discharged";
const X8_LOAD_WAR_WHY: &str = "UNPROVEN, not a claimed race: an original LLVM compiler obligation, OPEN (the committed qwen4_moe_ep_a8_rot hipcc image is unchanged compiler output; the sealed PM twin is the same ELF; no modified code is claimed to race). The ledger keeps the registers of a pending VMEM load locked until its load-counter wait and reports the first write of one (here a wide load whose address pair overlaps its own destination) on a CFG path that has no such wait; whether that path is feasible and whether the hardware has read the load address by then is not established here. Checked full-ELF byte identity between the hipcc anchor and the sealed twin does not discharge this strict WAR path in the ledger; nothing here is discharged";
const X8_LOAD_USE_WHY: &str = "MODEL LIMITATION, not a claimed race: a d16 load (or the `.l` half read of a register a d16 load fills) whose destination VGPR is also listed as a use (the other half is preserved), so the ledger reads the pending load into the same register as a RAW on its old value; it cannot tell a d16 lo/hi pair from an overwrite, and whether the hardware returns the pair in order is not established by this verifier (the committed qwen4_moe_ep_a8_rot hipcc image is unchanged compiler output; the sealed PM twin is the same ELF)";
const X8_KM_WHY: &str = "MODEL LIMITATION, not a claimed race: the IR models the VOP3 scalar-destination SGPR (here a `v_cmp_*_e64` result) as a pair although a wave32 instruction writes one SGPR, so the reported WAW with the pending SMEM load of s[2:3] exists only through the pair's second register (the committed qwen4_moe_ep_a8_rot hipcc image is unchanged compiler output; the sealed PM twin is the same ELF)";

/// Shared reasons of the 20 `qwen4_lwmma_tile` rows below (10 selected kernels of the whole
/// `qwen4_lwmma_tile.gfx1201.hip` ELF, once from the hipcc anchor `.co`, once from the sealed PM twin
/// `.hxaco`; same instruction and need in both images). Original LLVM obligations, OPEN.
const LWMMA_TILE_ST_WHY: &str = "UNPROVEN, not a claimed race: an original LLVM compiler obligation, OPEN (the committed qwen4_lwmma_tile hipcc image is unchanged compiler output; the sealed PM twin is the same ELF; no modified code is claimed to race). The ledger keeps a pending VMEM store's source registers (data and address VGPRs) locked until its store-counter wait and reports the first write of one on a CFG path that has no such wait; whether that path is feasible and whether the hardware has read the store sources by then (address captured at issue, vm_vsrc) is not established here and the path was not checked for feasibility. Checked full-ELF byte identity does not discharge this strict WAR path in the ledger. Nothing here is discharged";
const LWMMA_TILE_KM_WHY: &str = "MODEL LIMITATION, not a claimed race: the IR models the VOP3 scalar-destination SGPR (here a `v_cmp_*_e64` result) as a pair although a wave32 instruction writes one SGPR, so the reported WAW with the pending SMEM load of s[2:3] exists only through the pair's second register (the committed qwen4_lwmma_tile hipcc image is unchanged compiler output; the sealed PM twin is the same ELF)";

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
        target: "ggml_mmvq_shim_pm_gfx1201.hxaco#ggml_mmvq_q4_k_gfx1201",
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
    Allow {
        scope: "pins",
        target: "gemm_mq6sym_i8_hipcc_gfx1201.co#gemm_mq6sym_i8_set_m32n16_gfx1201",
        needle: "`v_and_b32_e32 v8, 0x3f00, v86` (after label `.LBB5`): needs Ds wait 1 (War { reg: \"v[8:9]\" })",
        why: "UNPROVEN, not a claimed race: an original LLVM DS-store-source lifetime finding (the committed gemm_mq6sym_i8 hipcc image is unchanged compiler output; no modified code is claimed to race). The ledger's report: the ledger keeps a pending DS store's source VGPRs locked until its DS-counter wait and reports the first write of one on a CFG path from the entry that has no such wait; whether the hardware has read the DS store sources by then (store-source timing) is not established here and the path was not checked for feasibility. Checked full-ELF byte identity between the hipcc anchor and the sealed twin does not discharge this strict WAR path in the ledger, and hardware source-read timing and path feasibility remain unproven. Nothing here is discharged",
    },
    Allow {
        scope: "pins",
        target: "gemm_mq6sym_i8_hipcc_gfx1201.co#gemm_mq6sym_i8_set_bf16_m32n16_gfx1201",
        needle: "`v_and_b32_e32 v8, 0x3f00, v86` (after label `.LBB5`): needs Ds wait 1 (War { reg: \"v[8:9]\" })",
        why: "UNPROVEN, not a claimed race: an original LLVM DS-store-source lifetime finding (the committed gemm_mq6sym_i8 hipcc image is unchanged compiler output; no modified code is claimed to race). The ledger's report: the ledger keeps a pending DS store's source VGPRs locked until its DS-counter wait and reports the first write of one on a CFG path from the entry that has no such wait; whether the hardware has read the DS store sources by then (store-source timing) is not established here and the path was not checked for feasibility. Checked full-ELF byte identity between the hipcc anchor and the sealed twin does not discharge this strict WAR path in the ledger, and hardware source-read timing and path feasibility remain unproven. Nothing here is discharged",
    },
    Allow {
        scope: "pins",
        target: "gemm_mq6sym_i8_hipcc_gfx1201.co#gemm_mq6sym_i8_set_m32n32_gfx1201",
        needle: "`v_lshrrev_b32_e32 v26, 4, v44` (after label `.LBB5`): needs Ds wait 1 (War { reg: \"v[26:27]\" })",
        why: "UNPROVEN, not a claimed race: an original LLVM DS-store-source lifetime finding (the committed gemm_mq6sym_i8 hipcc image is unchanged compiler output; no modified code is claimed to race). The ledger's report: the ledger keeps a pending DS store's source VGPRs locked until its DS-counter wait and reports the first write of one on a CFG path from the entry that has no such wait; whether the hardware has read the DS store sources by then (store-source timing) is not established here and the path was not checked for feasibility. Checked full-ELF byte identity between the hipcc anchor and the sealed twin does not discharge this strict WAR path in the ledger, and hardware source-read timing and path feasibility remain unproven. Nothing here is discharged",
    },
    Allow {
        scope: "pins",
        target: "gemm_mq6sym_i8_hipcc_gfx1201.co#gemm_mq6sym_i8_set_bf16_m32n32_gfx1201",
        needle: "`v_lshrrev_b32_e32 v26, 4, v44` (after label `.LBB5`): needs Ds wait 1 (War { reg: \"v[26:27]\" })",
        why: "UNPROVEN, not a claimed race: an original LLVM DS-store-source lifetime finding (the committed gemm_mq6sym_i8 hipcc image is unchanged compiler output; no modified code is claimed to race). The ledger's report: the ledger keeps a pending DS store's source VGPRs locked until its DS-counter wait and reports the first write of one on a CFG path from the entry that has no such wait; whether the hardware has read the DS store sources by then (store-source timing) is not established here and the path was not checked for feasibility. Checked full-ELF byte identity between the hipcc anchor and the sealed twin does not discharge this strict WAR path in the ledger, and hardware source-read timing and path feasibility remain unproven. Nothing here is discharged",
    },
    Allow {
        scope: "pins",
        target: "gemm_mq6sym_i8_hipcc_gfx1201.co#gemm_mq6sym_i8_set_m64n16_gfx1201",
        needle: "`v_and_or_b32 v10, v28, 63, v15` (after label `.LBB5`): needs Ds wait 1 (War { reg: \"v[10:11]\" })",
        why: "UNPROVEN, not a claimed race: an original LLVM DS-store-source lifetime finding (the committed gemm_mq6sym_i8 hipcc image is unchanged compiler output; no modified code is claimed to race). The ledger's report: the ledger keeps a pending DS store's source VGPRs locked until its DS-counter wait and reports the first write of one on a CFG path from the entry that has no such wait; whether the hardware has read the DS store sources by then (store-source timing) is not established here and the path was not checked for feasibility. Checked full-ELF byte identity between the hipcc anchor and the sealed twin does not discharge this strict WAR path in the ledger, and hardware source-read timing and path feasibility remain unproven. Nothing here is discharged",
    },
    Allow {
        scope: "pins",
        target: "gemm_mq6sym_i8_hipcc_gfx1201.co#gemm_mq6sym_i8_set_bf16_m64n16_gfx1201",
        needle: "`v_and_or_b32 v10, v28, 63, v15` (after label `.LBB5`): needs Ds wait 1 (War { reg: \"v[10:11]\" })",
        why: "UNPROVEN, not a claimed race: an original LLVM DS-store-source lifetime finding (the committed gemm_mq6sym_i8 hipcc image is unchanged compiler output; no modified code is claimed to race). The ledger's report: the ledger keeps a pending DS store's source VGPRs locked until its DS-counter wait and reports the first write of one on a CFG path from the entry that has no such wait; whether the hardware has read the DS store sources by then (store-source timing) is not established here and the path was not checked for feasibility. Checked full-ELF byte identity between the hipcc anchor and the sealed twin does not discharge this strict WAR path in the ledger, and hardware source-read timing and path feasibility remain unproven. Nothing here is discharged",
    },
    Allow {
        scope: "pins",
        target: "gemm_mq6sym_i8_hipcc_gfx1201.co#gemm_mq6sym_i8_set_m64n32_gfx1201",
        needle: "`v_lshlrev_b32_e32 v20, 12, v30` (after label `.LBB5`): needs Ds wait 0 (War { reg: \"v[20:21]\" })",
        why: "UNPROVEN, not a claimed race: an original LLVM DS-store-source lifetime finding (the committed gemm_mq6sym_i8 hipcc image is unchanged compiler output; no modified code is claimed to race). The ledger's report: the ledger keeps a pending DS store's source VGPRs locked until its DS-counter wait and reports the first write of one on a CFG path from the entry that has no such wait; whether the hardware has read the DS store sources by then (store-source timing) is not established here and the path was not checked for feasibility. Checked full-ELF byte identity between the hipcc anchor and the sealed twin does not discharge this strict WAR path in the ledger, and hardware source-read timing and path feasibility remain unproven. Nothing here is discharged",
    },
    Allow {
        scope: "pins",
        target: "gemm_mq6sym_i8_hipcc_gfx1201.co#gemm_mq6sym_i8_set_bf16_m64n32_gfx1201",
        needle: "`v_lshlrev_b32_e32 v20, 12, v30` (after label `.LBB5`): needs Ds wait 0 (War { reg: \"v[20:21]\" })",
        why: "UNPROVEN, not a claimed race: an original LLVM DS-store-source lifetime finding (the committed gemm_mq6sym_i8 hipcc image is unchanged compiler output; no modified code is claimed to race). The ledger's report: the ledger keeps a pending DS store's source VGPRs locked until its DS-counter wait and reports the first write of one on a CFG path from the entry that has no such wait; whether the hardware has read the DS store sources by then (store-source timing) is not established here and the path was not checked for feasibility. Checked full-ELF byte identity between the hipcc anchor and the sealed twin does not discharge this strict WAR path in the ledger, and hardware source-read timing and path feasibility remain unproven. Nothing here is discharged",
    },
    Allow {
        scope: "pins",
        target: "gemm_mq6sym_i8_hipcc_gfx1201.co#gemm_mq6sym_i8_set_m64n48_gfx1201",
        needle: "`v_alignbit_b32 v30, v61, v60, 24` (after label `.LBB5`): needs Ds wait 1 (War { reg: \"v[30:31]\" })",
        why: "UNPROVEN, not a claimed race: an original LLVM DS-store-source lifetime finding (the committed gemm_mq6sym_i8 hipcc image is unchanged compiler output; no modified code is claimed to race). The ledger's report: the ledger keeps a pending DS store's source VGPRs locked until its DS-counter wait and reports the first write of one on a CFG path from the entry that has no such wait; whether the hardware has read the DS store sources by then (store-source timing) is not established here and the path was not checked for feasibility. Checked full-ELF byte identity between the hipcc anchor and the sealed twin does not discharge this strict WAR path in the ledger, and hardware source-read timing and path feasibility remain unproven. Nothing here is discharged",
    },
    Allow {
        scope: "pins",
        target: "gemm_mq6sym_i8_hipcc_gfx1201.co#gemm_mq6sym_i8_set_bf16_m64n48_gfx1201",
        needle: "`v_alignbit_b32 v30, v61, v60, 24` (after label `.LBB5`): needs Ds wait 1 (War { reg: \"v[30:31]\" })",
        why: "UNPROVEN, not a claimed race: an original LLVM DS-store-source lifetime finding (the committed gemm_mq6sym_i8 hipcc image is unchanged compiler output; no modified code is claimed to race). The ledger's report: the ledger keeps a pending DS store's source VGPRs locked until its DS-counter wait and reports the first write of one on a CFG path from the entry that has no such wait; whether the hardware has read the DS store sources by then (store-source timing) is not established here and the path was not checked for feasibility. Checked full-ELF byte identity between the hipcc anchor and the sealed twin does not discharge this strict WAR path in the ledger, and hardware source-read timing and path feasibility remain unproven. Nothing here is discharged",
    },
    Allow {
        scope: "pins",
        target: "gemm_mq6sym_i8_hipcc_gfx1201.co#gemm_mq6sym_i8_set_m64n16y_gfx1201",
        needle: "`v_and_or_b32 v10, v28, 63, v15` (after label `.LBB5`): needs Ds wait 1 (War { reg: \"v[10:11]\" })",
        why: "UNPROVEN, not a claimed race: an original LLVM DS-store-source lifetime finding (the committed gemm_mq6sym_i8 hipcc image is unchanged compiler output; no modified code is claimed to race). The ledger's report: the ledger keeps a pending DS store's source VGPRs locked until its DS-counter wait and reports the first write of one on a CFG path from the entry that has no such wait; whether the hardware has read the DS store sources by then (store-source timing) is not established here and the path was not checked for feasibility. Checked full-ELF byte identity between the hipcc anchor and the sealed twin does not discharge this strict WAR path in the ledger, and hardware source-read timing and path feasibility remain unproven. Nothing here is discharged",
    },
    Allow {
        scope: "pins",
        target: "gemm_mq6sym_i8_hipcc_gfx1201.co#gemm_mq6sym_i8_set_bf16_m64n16y_gfx1201",
        needle: "`v_and_or_b32 v10, v28, 63, v15` (after label `.LBB5`): needs Ds wait 1 (War { reg: \"v[10:11]\" })",
        why: "UNPROVEN, not a claimed race: an original LLVM DS-store-source lifetime finding (the committed gemm_mq6sym_i8 hipcc image is unchanged compiler output; no modified code is claimed to race). The ledger's report: the ledger keeps a pending DS store's source VGPRs locked until its DS-counter wait and reports the first write of one on a CFG path from the entry that has no such wait; whether the hardware has read the DS store sources by then (store-source timing) is not established here and the path was not checked for feasibility. Checked full-ELF byte identity between the hipcc anchor and the sealed twin does not discharge this strict WAR path in the ledger, and hardware source-read timing and path feasibility remain unproven. Nothing here is discharged",
    },
    Allow {
        scope: "pins",
        target: "gemm_mq6sym_i8_hipcc_gfx1201.co#gemm_mq6sym_i8_set_m64n16_x4_gfx1201",
        needle: "`v_and_or_b32 v10, v28, 63, v15` (after label `.LBB14`): needs Ds wait 1 (War { reg: \"v[10:11]\" })",
        why: "UNPROVEN, not a claimed race: an original LLVM DS-store-source lifetime finding (the committed gemm_mq6sym_i8 hipcc image is unchanged compiler output; no modified code is claimed to race). The ledger's report: the ledger keeps a pending DS store's source VGPRs locked until its DS-counter wait and reports the first write of one on a CFG path from the entry that has no such wait; whether the hardware has read the DS store sources by then (store-source timing) is not established here and the path was not checked for feasibility. Checked full-ELF byte identity between the hipcc anchor and the sealed twin does not discharge this strict WAR path in the ledger, and hardware source-read timing and path feasibility remain unproven. Nothing here is discharged",
    },
    Allow {
        scope: "pins",
        target: "gemm_mq6sym_i8_hipcc_gfx1201.co#gemm_mq6sym_i8_set_m32n32_x4_gfx1201",
        needle: "`v_lshrrev_b32_e32 v26, 4, v44` (after label `.LBB14`): needs Ds wait 1 (War { reg: \"v[26:27]\" })",
        why: "UNPROVEN, not a claimed race: an original LLVM DS-store-source lifetime finding (the committed gemm_mq6sym_i8 hipcc image is unchanged compiler output; no modified code is claimed to race). The ledger's report: the ledger keeps a pending DS store's source VGPRs locked until its DS-counter wait and reports the first write of one on a CFG path from the entry that has no such wait; whether the hardware has read the DS store sources by then (store-source timing) is not established here and the path was not checked for feasibility. Checked full-ELF byte identity between the hipcc anchor and the sealed twin does not discharge this strict WAR path in the ledger, and hardware source-read timing and path feasibility remain unproven. Nothing here is discharged",
    },
    Allow {
        scope: "pins",
        target: "gemm_mq6sym_i8_hipcc_gfx1201.co#gemm_mq6sym_i8_set_m64n48_x4_gfx1201",
        needle: "`v_alignbit_b32 v30, v61, v60, 24` (after label `.LBB14`): needs Ds wait 1 (War { reg: \"v[30:31]\" })",
        why: "UNPROVEN, not a claimed race: an original LLVM DS-store-source lifetime finding (the committed gemm_mq6sym_i8 hipcc image is unchanged compiler output; no modified code is claimed to race). The ledger's report: the ledger keeps a pending DS store's source VGPRs locked until its DS-counter wait and reports the first write of one on a CFG path from the entry that has no such wait; whether the hardware has read the DS store sources by then (store-source timing) is not established here and the path was not checked for feasibility. Checked full-ELF byte identity between the hipcc anchor and the sealed twin does not discharge this strict WAR path in the ledger, and hardware source-read timing and path feasibility remain unproven. Nothing here is discharged",
    },
    Allow {
        scope: "pins",
        target: "gemm_mq6sym_i8_hipcc_gfx1201.co#gemm_mq6sym_i8_set_m64n16y_x4_gfx1201",
        needle: "`v_and_or_b32 v10, v28, 63, v15` (after label `.LBB14`): needs Ds wait 1 (War { reg: \"v[10:11]\" })",
        why: "UNPROVEN, not a claimed race: an original LLVM DS-store-source lifetime finding (the committed gemm_mq6sym_i8 hipcc image is unchanged compiler output; no modified code is claimed to race). The ledger's report: the ledger keeps a pending DS store's source VGPRs locked until its DS-counter wait and reports the first write of one on a CFG path from the entry that has no such wait; whether the hardware has read the DS store sources by then (store-source timing) is not established here and the path was not checked for feasibility. Checked full-ELF byte identity between the hipcc anchor and the sealed twin does not discharge this strict WAR path in the ledger, and hardware source-read timing and path feasibility remain unproven. Nothing here is discharged",
    },
    Allow {
        scope: "pins",
        target: "gemm_mq6sym_i8_hipcc_gfx1201.co#gemm_mq6sym_i8_set_gfx1201",
        needle: "`v_lshrrev_b32_e32 v3, 2, v0` (after label `.LBB5`): needs Ds wait 0 (War { reg: \"v3\" })",
        why: "UNPROVEN, not a claimed race: an original LLVM DS-store-source lifetime finding (the committed gemm_mq6sym_i8 hipcc image is unchanged compiler output; no modified code is claimed to race). The ledger's report: the ledger keeps a pending DS store's source VGPRs locked until its DS-counter wait and reports the first write of one on a CFG path from the entry that has no such wait; whether the hardware has read the DS store sources by then (store-source timing) is not established here and the path was not checked for feasibility. Checked full-ELF byte identity between the hipcc anchor and the sealed twin does not discharge this strict WAR path in the ledger, and hardware source-read timing and path feasibility remain unproven. Nothing here is discharged",
    },
    Allow {
        scope: "pins",
        target: "gemm_mq6sym_i8_hipcc_gfx1201.co#gemm_mq6sym_i8_set_bf16_gfx1201",
        needle: "`v_lshrrev_b32_e32 v3, 2, v0` (after label `.LBB5`): needs Ds wait 0 (War { reg: \"v3\" })",
        why: "UNPROVEN, not a claimed race: an original LLVM DS-store-source lifetime finding (the committed gemm_mq6sym_i8 hipcc image is unchanged compiler output; no modified code is claimed to race). The ledger's report: the ledger keeps a pending DS store's source VGPRs locked until its DS-counter wait and reports the first write of one on a CFG path from the entry that has no such wait; whether the hardware has read the DS store sources by then (store-source timing) is not established here and the path was not checked for feasibility. Checked full-ELF byte identity between the hipcc anchor and the sealed twin does not discharge this strict WAR path in the ledger, and hardware source-read timing and path feasibility remain unproven. Nothing here is discharged",
    },
    Allow {
        scope: "pins",
        target: "gemm_mq6sym_i8_pm_gfx1201.hxaco#gemm_mq6sym_i8_set_m32n16_gfx1201",
        needle: "`v_and_b32_e32 v8, 0x3f00, v86` (after label `.LBB5`): needs Ds wait 1 (War { reg: \"v[8:9]\" })",
        why: "UNPROVEN, not a claimed race: an original LLVM DS-store-source lifetime finding (the sealed twin is the same ELF as the hipcc image; no modified code is claimed to race). The ledger's report: the ledger keeps a pending DS store's source VGPRs locked until its DS-counter wait and reports the first write of one on a CFG path from the entry that has no such wait; whether the hardware has read the DS store sources by then (store-source timing) is not established here and the path was not checked for feasibility. Checked full-ELF byte identity between the hipcc anchor and the sealed twin does not discharge this strict WAR path in the ledger, and hardware source-read timing and path feasibility remain unproven. Nothing here is discharged",
    },
    Allow {
        scope: "pins",
        target: "gemm_mq6sym_i8_pm_gfx1201.hxaco#gemm_mq6sym_i8_set_bf16_m32n16_gfx1201",
        needle: "`v_and_b32_e32 v8, 0x3f00, v86` (after label `.LBB5`): needs Ds wait 1 (War { reg: \"v[8:9]\" })",
        why: "UNPROVEN, not a claimed race: an original LLVM DS-store-source lifetime finding (the sealed twin is the same ELF as the hipcc image; no modified code is claimed to race). The ledger's report: the ledger keeps a pending DS store's source VGPRs locked until its DS-counter wait and reports the first write of one on a CFG path from the entry that has no such wait; whether the hardware has read the DS store sources by then (store-source timing) is not established here and the path was not checked for feasibility. Checked full-ELF byte identity between the hipcc anchor and the sealed twin does not discharge this strict WAR path in the ledger, and hardware source-read timing and path feasibility remain unproven. Nothing here is discharged",
    },
    Allow {
        scope: "pins",
        target: "gemm_mq6sym_i8_pm_gfx1201.hxaco#gemm_mq6sym_i8_set_m32n32_gfx1201",
        needle: "`v_lshrrev_b32_e32 v26, 4, v44` (after label `.LBB5`): needs Ds wait 1 (War { reg: \"v[26:27]\" })",
        why: "UNPROVEN, not a claimed race: an original LLVM DS-store-source lifetime finding (the sealed twin is the same ELF as the hipcc image; no modified code is claimed to race). The ledger's report: the ledger keeps a pending DS store's source VGPRs locked until its DS-counter wait and reports the first write of one on a CFG path from the entry that has no such wait; whether the hardware has read the DS store sources by then (store-source timing) is not established here and the path was not checked for feasibility. Checked full-ELF byte identity between the hipcc anchor and the sealed twin does not discharge this strict WAR path in the ledger, and hardware source-read timing and path feasibility remain unproven. Nothing here is discharged",
    },
    Allow {
        scope: "pins",
        target: "gemm_mq6sym_i8_pm_gfx1201.hxaco#gemm_mq6sym_i8_set_bf16_m32n32_gfx1201",
        needle: "`v_lshrrev_b32_e32 v26, 4, v44` (after label `.LBB5`): needs Ds wait 1 (War { reg: \"v[26:27]\" })",
        why: "UNPROVEN, not a claimed race: an original LLVM DS-store-source lifetime finding (the sealed twin is the same ELF as the hipcc image; no modified code is claimed to race). The ledger's report: the ledger keeps a pending DS store's source VGPRs locked until its DS-counter wait and reports the first write of one on a CFG path from the entry that has no such wait; whether the hardware has read the DS store sources by then (store-source timing) is not established here and the path was not checked for feasibility. Checked full-ELF byte identity between the hipcc anchor and the sealed twin does not discharge this strict WAR path in the ledger, and hardware source-read timing and path feasibility remain unproven. Nothing here is discharged",
    },
    Allow {
        scope: "pins",
        target: "gemm_mq6sym_i8_pm_gfx1201.hxaco#gemm_mq6sym_i8_set_m64n16_gfx1201",
        needle: "`v_and_or_b32 v10, v28, 63, v15` (after label `.LBB5`): needs Ds wait 1 (War { reg: \"v[10:11]\" })",
        why: "UNPROVEN, not a claimed race: an original LLVM DS-store-source lifetime finding (the sealed twin is the same ELF as the hipcc image; no modified code is claimed to race). The ledger's report: the ledger keeps a pending DS store's source VGPRs locked until its DS-counter wait and reports the first write of one on a CFG path from the entry that has no such wait; whether the hardware has read the DS store sources by then (store-source timing) is not established here and the path was not checked for feasibility. Checked full-ELF byte identity between the hipcc anchor and the sealed twin does not discharge this strict WAR path in the ledger, and hardware source-read timing and path feasibility remain unproven. Nothing here is discharged",
    },
    Allow {
        scope: "pins",
        target: "gemm_mq6sym_i8_pm_gfx1201.hxaco#gemm_mq6sym_i8_set_bf16_m64n16_gfx1201",
        needle: "`v_and_or_b32 v10, v28, 63, v15` (after label `.LBB5`): needs Ds wait 1 (War { reg: \"v[10:11]\" })",
        why: "UNPROVEN, not a claimed race: an original LLVM DS-store-source lifetime finding (the sealed twin is the same ELF as the hipcc image; no modified code is claimed to race). The ledger's report: the ledger keeps a pending DS store's source VGPRs locked until its DS-counter wait and reports the first write of one on a CFG path from the entry that has no such wait; whether the hardware has read the DS store sources by then (store-source timing) is not established here and the path was not checked for feasibility. Checked full-ELF byte identity between the hipcc anchor and the sealed twin does not discharge this strict WAR path in the ledger, and hardware source-read timing and path feasibility remain unproven. Nothing here is discharged",
    },
    Allow {
        scope: "pins",
        target: "gemm_mq6sym_i8_pm_gfx1201.hxaco#gemm_mq6sym_i8_set_m64n32_gfx1201",
        needle: "`v_lshlrev_b32_e32 v20, 12, v30` (after label `.LBB5`): needs Ds wait 0 (War { reg: \"v[20:21]\" })",
        why: "UNPROVEN, not a claimed race: an original LLVM DS-store-source lifetime finding (the sealed twin is the same ELF as the hipcc image; no modified code is claimed to race). The ledger's report: the ledger keeps a pending DS store's source VGPRs locked until its DS-counter wait and reports the first write of one on a CFG path from the entry that has no such wait; whether the hardware has read the DS store sources by then (store-source timing) is not established here and the path was not checked for feasibility. Checked full-ELF byte identity between the hipcc anchor and the sealed twin does not discharge this strict WAR path in the ledger, and hardware source-read timing and path feasibility remain unproven. Nothing here is discharged",
    },
    Allow {
        scope: "pins",
        target: "gemm_mq6sym_i8_pm_gfx1201.hxaco#gemm_mq6sym_i8_set_bf16_m64n32_gfx1201",
        needle: "`v_lshlrev_b32_e32 v20, 12, v30` (after label `.LBB5`): needs Ds wait 0 (War { reg: \"v[20:21]\" })",
        why: "UNPROVEN, not a claimed race: an original LLVM DS-store-source lifetime finding (the sealed twin is the same ELF as the hipcc image; no modified code is claimed to race). The ledger's report: the ledger keeps a pending DS store's source VGPRs locked until its DS-counter wait and reports the first write of one on a CFG path from the entry that has no such wait; whether the hardware has read the DS store sources by then (store-source timing) is not established here and the path was not checked for feasibility. Checked full-ELF byte identity between the hipcc anchor and the sealed twin does not discharge this strict WAR path in the ledger, and hardware source-read timing and path feasibility remain unproven. Nothing here is discharged",
    },
    Allow {
        scope: "pins",
        target: "gemm_mq6sym_i8_pm_gfx1201.hxaco#gemm_mq6sym_i8_set_m64n48_gfx1201",
        needle: "`v_alignbit_b32 v30, v61, v60, 24` (after label `.LBB5`): needs Ds wait 1 (War { reg: \"v[30:31]\" })",
        why: "UNPROVEN, not a claimed race: an original LLVM DS-store-source lifetime finding (the sealed twin is the same ELF as the hipcc image; no modified code is claimed to race). The ledger's report: the ledger keeps a pending DS store's source VGPRs locked until its DS-counter wait and reports the first write of one on a CFG path from the entry that has no such wait; whether the hardware has read the DS store sources by then (store-source timing) is not established here and the path was not checked for feasibility. Checked full-ELF byte identity between the hipcc anchor and the sealed twin does not discharge this strict WAR path in the ledger, and hardware source-read timing and path feasibility remain unproven. Nothing here is discharged",
    },
    Allow {
        scope: "pins",
        target: "gemm_mq6sym_i8_pm_gfx1201.hxaco#gemm_mq6sym_i8_set_bf16_m64n48_gfx1201",
        needle: "`v_alignbit_b32 v30, v61, v60, 24` (after label `.LBB5`): needs Ds wait 1 (War { reg: \"v[30:31]\" })",
        why: "UNPROVEN, not a claimed race: an original LLVM DS-store-source lifetime finding (the sealed twin is the same ELF as the hipcc image; no modified code is claimed to race). The ledger's report: the ledger keeps a pending DS store's source VGPRs locked until its DS-counter wait and reports the first write of one on a CFG path from the entry that has no such wait; whether the hardware has read the DS store sources by then (store-source timing) is not established here and the path was not checked for feasibility. Checked full-ELF byte identity between the hipcc anchor and the sealed twin does not discharge this strict WAR path in the ledger, and hardware source-read timing and path feasibility remain unproven. Nothing here is discharged",
    },
    Allow {
        scope: "pins",
        target: "gemm_mq6sym_i8_pm_gfx1201.hxaco#gemm_mq6sym_i8_set_m64n16y_gfx1201",
        needle: "`v_and_or_b32 v10, v28, 63, v15` (after label `.LBB5`): needs Ds wait 1 (War { reg: \"v[10:11]\" })",
        why: "UNPROVEN, not a claimed race: an original LLVM DS-store-source lifetime finding (the sealed twin is the same ELF as the hipcc image; no modified code is claimed to race). The ledger's report: the ledger keeps a pending DS store's source VGPRs locked until its DS-counter wait and reports the first write of one on a CFG path from the entry that has no such wait; whether the hardware has read the DS store sources by then (store-source timing) is not established here and the path was not checked for feasibility. Checked full-ELF byte identity between the hipcc anchor and the sealed twin does not discharge this strict WAR path in the ledger, and hardware source-read timing and path feasibility remain unproven. Nothing here is discharged",
    },
    Allow {
        scope: "pins",
        target: "gemm_mq6sym_i8_pm_gfx1201.hxaco#gemm_mq6sym_i8_set_bf16_m64n16y_gfx1201",
        needle: "`v_and_or_b32 v10, v28, 63, v15` (after label `.LBB5`): needs Ds wait 1 (War { reg: \"v[10:11]\" })",
        why: "UNPROVEN, not a claimed race: an original LLVM DS-store-source lifetime finding (the sealed twin is the same ELF as the hipcc image; no modified code is claimed to race). The ledger's report: the ledger keeps a pending DS store's source VGPRs locked until its DS-counter wait and reports the first write of one on a CFG path from the entry that has no such wait; whether the hardware has read the DS store sources by then (store-source timing) is not established here and the path was not checked for feasibility. Checked full-ELF byte identity between the hipcc anchor and the sealed twin does not discharge this strict WAR path in the ledger, and hardware source-read timing and path feasibility remain unproven. Nothing here is discharged",
    },
    Allow {
        scope: "pins",
        target: "gemm_mq6sym_i8_pm_gfx1201.hxaco#gemm_mq6sym_i8_set_m64n16_x4_gfx1201",
        needle: "`v_and_or_b32 v10, v28, 63, v15` (after label `.LBB14`): needs Ds wait 1 (War { reg: \"v[10:11]\" })",
        why: "UNPROVEN, not a claimed race: an original LLVM DS-store-source lifetime finding (the sealed twin is the same ELF as the hipcc image; no modified code is claimed to race). The ledger's report: the ledger keeps a pending DS store's source VGPRs locked until its DS-counter wait and reports the first write of one on a CFG path from the entry that has no such wait; whether the hardware has read the DS store sources by then (store-source timing) is not established here and the path was not checked for feasibility. Checked full-ELF byte identity between the hipcc anchor and the sealed twin does not discharge this strict WAR path in the ledger, and hardware source-read timing and path feasibility remain unproven. Nothing here is discharged",
    },
    Allow {
        scope: "pins",
        target: "gemm_mq6sym_i8_pm_gfx1201.hxaco#gemm_mq6sym_i8_set_m32n32_x4_gfx1201",
        needle: "`v_lshrrev_b32_e32 v26, 4, v44` (after label `.LBB14`): needs Ds wait 1 (War { reg: \"v[26:27]\" })",
        why: "UNPROVEN, not a claimed race: an original LLVM DS-store-source lifetime finding (the sealed twin is the same ELF as the hipcc image; no modified code is claimed to race). The ledger's report: the ledger keeps a pending DS store's source VGPRs locked until its DS-counter wait and reports the first write of one on a CFG path from the entry that has no such wait; whether the hardware has read the DS store sources by then (store-source timing) is not established here and the path was not checked for feasibility. Checked full-ELF byte identity between the hipcc anchor and the sealed twin does not discharge this strict WAR path in the ledger, and hardware source-read timing and path feasibility remain unproven. Nothing here is discharged",
    },
    Allow {
        scope: "pins",
        target: "gemm_mq6sym_i8_pm_gfx1201.hxaco#gemm_mq6sym_i8_set_m64n48_x4_gfx1201",
        needle: "`v_alignbit_b32 v30, v61, v60, 24` (after label `.LBB14`): needs Ds wait 1 (War { reg: \"v[30:31]\" })",
        why: "UNPROVEN, not a claimed race: an original LLVM DS-store-source lifetime finding (the sealed twin is the same ELF as the hipcc image; no modified code is claimed to race). The ledger's report: the ledger keeps a pending DS store's source VGPRs locked until its DS-counter wait and reports the first write of one on a CFG path from the entry that has no such wait; whether the hardware has read the DS store sources by then (store-source timing) is not established here and the path was not checked for feasibility. Checked full-ELF byte identity between the hipcc anchor and the sealed twin does not discharge this strict WAR path in the ledger, and hardware source-read timing and path feasibility remain unproven. Nothing here is discharged",
    },
    Allow {
        scope: "pins",
        target: "gemm_mq6sym_i8_pm_gfx1201.hxaco#gemm_mq6sym_i8_set_m64n16y_x4_gfx1201",
        needle: "`v_and_or_b32 v10, v28, 63, v15` (after label `.LBB14`): needs Ds wait 1 (War { reg: \"v[10:11]\" })",
        why: "UNPROVEN, not a claimed race: an original LLVM DS-store-source lifetime finding (the sealed twin is the same ELF as the hipcc image; no modified code is claimed to race). The ledger's report: the ledger keeps a pending DS store's source VGPRs locked until its DS-counter wait and reports the first write of one on a CFG path from the entry that has no such wait; whether the hardware has read the DS store sources by then (store-source timing) is not established here and the path was not checked for feasibility. Checked full-ELF byte identity between the hipcc anchor and the sealed twin does not discharge this strict WAR path in the ledger, and hardware source-read timing and path feasibility remain unproven. Nothing here is discharged",
    },
    Allow {
        scope: "pins",
        target: "gemm_mq6sym_i8_pm_gfx1201.hxaco#gemm_mq6sym_i8_set_gfx1201",
        needle: "`v_lshrrev_b32_e32 v3, 2, v0` (after label `.LBB5`): needs Ds wait 0 (War { reg: \"v3\" })",
        why: "UNPROVEN, not a claimed race: an original LLVM DS-store-source lifetime finding (the sealed twin is the same ELF as the hipcc image; no modified code is claimed to race). The ledger's report: the ledger keeps a pending DS store's source VGPRs locked until its DS-counter wait and reports the first write of one on a CFG path from the entry that has no such wait; whether the hardware has read the DS store sources by then (store-source timing) is not established here and the path was not checked for feasibility. Checked full-ELF byte identity between the hipcc anchor and the sealed twin does not discharge this strict WAR path in the ledger, and hardware source-read timing and path feasibility remain unproven. Nothing here is discharged",
    },
    Allow {
        scope: "pins",
        target: "gemm_mq6sym_i8_pm_gfx1201.hxaco#gemm_mq6sym_i8_set_bf16_gfx1201",
        needle: "`v_lshrrev_b32_e32 v3, 2, v0` (after label `.LBB5`): needs Ds wait 0 (War { reg: \"v3\" })",
        why: "UNPROVEN, not a claimed race: an original LLVM DS-store-source lifetime finding (the sealed twin is the same ELF as the hipcc image; no modified code is claimed to race). The ledger's report: the ledger keeps a pending DS store's source VGPRs locked until its DS-counter wait and reports the first write of one on a CFG path from the entry that has no such wait; whether the hardware has read the DS store sources by then (store-source timing) is not established here and the path was not checked for feasibility. Checked full-ELF byte identity between the hipcc anchor and the sealed twin does not discharge this strict WAR path in the ledger, and hardware source-read timing and path feasibility remain unproven. Nothing here is discharged",
    },
    Allow {
        scope: "pins",
        target: "moe_shared_down_combine_hipcc_gfx1201.co#moe_shared_down_combine_rows_r1",
        needle: "`v_bfe_u32 v20, v18, 16, 1` (after label `.LBB1`): needs Ds wait 0 (War { reg: \"v20\" })",
        why: MOE_SHARED_DOWN_COMBINE_WHY,
    },
    Allow {
        scope: "pins",
        target: "moe_shared_down_combine_hipcc_gfx1201.co#moe_shared_down_combine_rows_r1_zinit",
        needle: "`v_bfe_u32 v19, v17, 16, 1` (after label `.LBB1`): needs Ds wait 0 (War { reg: \"v19\" })",
        why: MOE_SHARED_DOWN_COMBINE_WHY,
    },
    Allow {
        scope: "pins",
        target: "moe_shared_down_combine_hipcc_gfx1201.co#moe_shared_down_combine_rows_r2",
        needle: "`v_dual_add_f32 v5, v10, v12 :: v_dual_add_f32 v6, v8, v9` (after label `.LBB22`): needs Ds wait 0 (War { reg: \"v5\" })",
        why: MOE_SHARED_DOWN_COMBINE_WHY,
    },
    Allow {
        scope: "pins",
        target: "moe_shared_down_combine_hipcc_gfx1201.co#moe_shared_down_combine_rows_r2_zinit",
        needle: "`v_dual_add_f32 v5, v10, v12 :: v_dual_add_f32 v6, v8, v9` (after label `.LBB22`): needs Ds wait 0 (War { reg: \"v5\" })",
        why: MOE_SHARED_DOWN_COMBINE_WHY,
    },
    Allow {
        scope: "pins",
        target: "moe_shared_down_combine_hipcc_gfx1201.co#moe_shared_down_combine_rows_r3",
        needle: "`v_add_f32_e32 v5, v29, v32` (after label `.LBB24`): needs Ds wait 0 (War { reg: \"v5\" })",
        why: MOE_SHARED_DOWN_COMBINE_WHY,
    },
    Allow {
        scope: "pins",
        target: "moe_shared_down_combine_hipcc_gfx1201.co#moe_shared_down_combine_rows_r3_zinit",
        needle: "`v_add_f32_e32 v5, v29, v32` (after label `.LBB24`): needs Ds wait 0 (War { reg: \"v5\" })",
        why: MOE_SHARED_DOWN_COMBINE_WHY,
    },
    Allow {
        scope: "pins",
        target: "moe_shared_down_combine_hipcc_gfx1201.co#moe_shared_down_combine_rows_r4",
        needle: "`v_add_f32_e32 v5, v45, v46` (after label `.LBB22`): needs Ds wait 0 (War { reg: \"v5\" })",
        why: MOE_SHARED_DOWN_COMBINE_WHY,
    },
    Allow {
        scope: "pins",
        target: "moe_shared_down_combine_hipcc_gfx1201.co#moe_shared_down_combine_rows_r4_zinit",
        needle: "`v_add_f32_e32 v5, v45, v46` (after label `.LBB22`): needs Ds wait 0 (War { reg: \"v5\" })",
        why: MOE_SHARED_DOWN_COMBINE_WHY,
    },
    Allow {
        scope: "pins",
        target: "moe_shared_down_combine_hipcc_gfx1201.co#moe_shared_down_combine_rows_r5",
        needle: "`v_add_f32_e32 v5, v43, v45` (after label `.LBB22`): needs Ds wait 0 (War { reg: \"v5\" })",
        why: MOE_SHARED_DOWN_COMBINE_WHY,
    },
    Allow {
        scope: "pins",
        target: "moe_shared_down_combine_hipcc_gfx1201.co#moe_shared_down_combine_rows_r5_zinit",
        needle: "`v_add_f32_e32 v5, v43, v45` (after label `.LBB22`): needs Ds wait 0 (War { reg: \"v5\" })",
        why: MOE_SHARED_DOWN_COMBINE_WHY,
    },
    Allow {
        scope: "pins",
        target: "moe_shared_down_combine_hipcc_gfx1201.co#moe_shared_down_combine_rows_r6",
        needle: "`v_add_f32_e32 v5, v57, v59` (after label `.LBB22`): needs Ds wait 0 (War { reg: \"v5\" })",
        why: MOE_SHARED_DOWN_COMBINE_WHY,
    },
    Allow {
        scope: "pins",
        target: "moe_shared_down_combine_hipcc_gfx1201.co#moe_shared_down_combine_rows_r6_zinit",
        needle: "`v_add_f32_e32 v5, v57, v59` (after label `.LBB22`): needs Ds wait 0 (War { reg: \"v5\" })",
        why: MOE_SHARED_DOWN_COMBINE_WHY,
    },
    Allow {
        scope: "pins",
        target: "moe_shared_down_combine_hipcc_gfx1201.co#moe_shared_down_combine_rows_r7",
        needle: "`v_add_f32_e32 v5, v56, v58` (after label `.LBB22`): needs Ds wait 0 (War { reg: \"v5\" })",
        why: MOE_SHARED_DOWN_COMBINE_WHY,
    },
    Allow {
        scope: "pins",
        target: "moe_shared_down_combine_hipcc_gfx1201.co#moe_shared_down_combine_rows_r7_zinit",
        needle: "`v_add_f32_e32 v5, v56, v58` (after label `.LBB22`): needs Ds wait 0 (War { reg: \"v5\" })",
        why: MOE_SHARED_DOWN_COMBINE_WHY,
    },
    Allow {
        scope: "pins",
        target: "moe_shared_down_combine_hipcc_gfx1201.co#moe_shared_down_combine_rows_r8",
        needle: "`v_add_f32_e32 v5, v49, v52` (after label `.LBB22`): needs Ds wait 0 (War { reg: \"v5\" })",
        why: MOE_SHARED_DOWN_COMBINE_WHY,
    },
    Allow {
        scope: "pins",
        target: "moe_shared_down_combine_hipcc_gfx1201.co#moe_shared_down_combine_rows_r8_zinit",
        needle: "`v_add_f32_e32 v5, v49, v52` (after label `.LBB22`): needs Ds wait 0 (War { reg: \"v5\" })",
        why: MOE_SHARED_DOWN_COMBINE_WHY,
    },
    Allow {
        scope: "pins",
        target: "moe_shared_down_combine_pm_gfx1201.hxaco#moe_shared_down_combine_rows_r1",
        needle: "`v_bfe_u32 v20, v18, 16, 1` (after label `.LBB1`): needs Ds wait 0 (War { reg: \"v20\" })",
        why: MOE_SHARED_DOWN_COMBINE_WHY,
    },
    Allow {
        scope: "pins",
        target: "moe_shared_down_combine_pm_gfx1201.hxaco#moe_shared_down_combine_rows_r1_zinit",
        needle: "`v_bfe_u32 v19, v17, 16, 1` (after label `.LBB1`): needs Ds wait 0 (War { reg: \"v19\" })",
        why: MOE_SHARED_DOWN_COMBINE_WHY,
    },
    Allow {
        scope: "pins",
        target: "moe_shared_down_combine_pm_gfx1201.hxaco#moe_shared_down_combine_rows_r2",
        needle: "`v_dual_add_f32 v5, v10, v12 :: v_dual_add_f32 v6, v8, v9` (after label `.LBB22`): needs Ds wait 0 (War { reg: \"v5\" })",
        why: MOE_SHARED_DOWN_COMBINE_WHY,
    },
    Allow {
        scope: "pins",
        target: "moe_shared_down_combine_pm_gfx1201.hxaco#moe_shared_down_combine_rows_r2_zinit",
        needle: "`v_dual_add_f32 v5, v10, v12 :: v_dual_add_f32 v6, v8, v9` (after label `.LBB22`): needs Ds wait 0 (War { reg: \"v5\" })",
        why: MOE_SHARED_DOWN_COMBINE_WHY,
    },
    Allow {
        scope: "pins",
        target: "moe_shared_down_combine_pm_gfx1201.hxaco#moe_shared_down_combine_rows_r3",
        needle: "`v_add_f32_e32 v5, v29, v32` (after label `.LBB24`): needs Ds wait 0 (War { reg: \"v5\" })",
        why: MOE_SHARED_DOWN_COMBINE_WHY,
    },
    Allow {
        scope: "pins",
        target: "moe_shared_down_combine_pm_gfx1201.hxaco#moe_shared_down_combine_rows_r3_zinit",
        needle: "`v_add_f32_e32 v5, v29, v32` (after label `.LBB24`): needs Ds wait 0 (War { reg: \"v5\" })",
        why: MOE_SHARED_DOWN_COMBINE_WHY,
    },
    Allow {
        scope: "pins",
        target: "moe_shared_down_combine_pm_gfx1201.hxaco#moe_shared_down_combine_rows_r4",
        needle: "`v_add_f32_e32 v5, v45, v46` (after label `.LBB22`): needs Ds wait 0 (War { reg: \"v5\" })",
        why: MOE_SHARED_DOWN_COMBINE_WHY,
    },
    Allow {
        scope: "pins",
        target: "moe_shared_down_combine_pm_gfx1201.hxaco#moe_shared_down_combine_rows_r4_zinit",
        needle: "`v_add_f32_e32 v5, v45, v46` (after label `.LBB22`): needs Ds wait 0 (War { reg: \"v5\" })",
        why: MOE_SHARED_DOWN_COMBINE_WHY,
    },
    Allow {
        scope: "pins",
        target: "moe_shared_down_combine_pm_gfx1201.hxaco#moe_shared_down_combine_rows_r5",
        needle: "`v_add_f32_e32 v5, v43, v45` (after label `.LBB22`): needs Ds wait 0 (War { reg: \"v5\" })",
        why: MOE_SHARED_DOWN_COMBINE_WHY,
    },
    Allow {
        scope: "pins",
        target: "moe_shared_down_combine_pm_gfx1201.hxaco#moe_shared_down_combine_rows_r5_zinit",
        needle: "`v_add_f32_e32 v5, v43, v45` (after label `.LBB22`): needs Ds wait 0 (War { reg: \"v5\" })",
        why: MOE_SHARED_DOWN_COMBINE_WHY,
    },
    Allow {
        scope: "pins",
        target: "moe_shared_down_combine_pm_gfx1201.hxaco#moe_shared_down_combine_rows_r6",
        needle: "`v_add_f32_e32 v5, v57, v59` (after label `.LBB22`): needs Ds wait 0 (War { reg: \"v5\" })",
        why: MOE_SHARED_DOWN_COMBINE_WHY,
    },
    Allow {
        scope: "pins",
        target: "moe_shared_down_combine_pm_gfx1201.hxaco#moe_shared_down_combine_rows_r6_zinit",
        needle: "`v_add_f32_e32 v5, v57, v59` (after label `.LBB22`): needs Ds wait 0 (War { reg: \"v5\" })",
        why: MOE_SHARED_DOWN_COMBINE_WHY,
    },
    Allow {
        scope: "pins",
        target: "moe_shared_down_combine_pm_gfx1201.hxaco#moe_shared_down_combine_rows_r7",
        needle: "`v_add_f32_e32 v5, v56, v58` (after label `.LBB22`): needs Ds wait 0 (War { reg: \"v5\" })",
        why: MOE_SHARED_DOWN_COMBINE_WHY,
    },
    Allow {
        scope: "pins",
        target: "moe_shared_down_combine_pm_gfx1201.hxaco#moe_shared_down_combine_rows_r7_zinit",
        needle: "`v_add_f32_e32 v5, v56, v58` (after label `.LBB22`): needs Ds wait 0 (War { reg: \"v5\" })",
        why: MOE_SHARED_DOWN_COMBINE_WHY,
    },
    Allow {
        scope: "pins",
        target: "moe_shared_down_combine_pm_gfx1201.hxaco#moe_shared_down_combine_rows_r8",
        needle: "`v_add_f32_e32 v5, v49, v52` (after label `.LBB22`): needs Ds wait 0 (War { reg: \"v5\" })",
        why: MOE_SHARED_DOWN_COMBINE_WHY,
    },
    Allow {
        scope: "pins",
        target: "moe_shared_down_combine_pm_gfx1201.hxaco#moe_shared_down_combine_rows_r8_zinit",
        needle: "`v_add_f32_e32 v5, v49, v52` (after label `.LBB22`): needs Ds wait 0 (War { reg: \"v5\" })",
        why: MOE_SHARED_DOWN_COMBINE_WHY,
    },
    Allow {
        scope: "pins",
        target: "topk8_rows_mq6g256v2_hipcc_gfx1201.co#topk8_partial_f32_rows",
        needle: "`v_cndmask_b32_e64 v1, 0x7fffffff, v6, s2` (after label `.LBB19`): needs Store wait 1 (War { reg: \"v1\" })",
        why: TOPK8_ROWS_ST_WHY,
    },
    Allow {
        scope: "pins",
        target: "topk8_rows_mq6g256v2_hipcc_gfx1201.co#topk8_rescore_mq6g256v2_k2560_rows",
        needle: "`v_cmp_eq_u32_e64 s3, 0, v2` (after label `.LBB12`): needs Km wait 0 (Waw { reg: \"s4\" })",
        why: TOPK8_ROWS_KM_WHY,
    },
    Allow {
        scope: "pins",
        target: "topk8_rows_mq6g256v2_pm_gfx1201.hxaco#topk8_partial_f32_rows",
        needle: "`v_cndmask_b32_e64 v1, 0x7fffffff, v6, s2` (after label `.LBB19`): needs Store wait 1 (War { reg: \"v1\" })",
        why: TOPK8_ROWS_ST_WHY,
    },
    Allow {
        scope: "pins",
        target: "topk8_rows_mq6g256v2_pm_gfx1201.hxaco#topk8_rescore_mq6g256v2_k2560_rows",
        needle: "`v_cmp_eq_u32_e64 s3, 0, v2` (after label `.LBB12`): needs Km wait 0 (Waw { reg: \"s4\" })",
        why: TOPK8_ROWS_KM_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_gemm_bf16_t1_m1_n1",
        needle: "`v_lshlrev_b32_e32 v5, 2, v15` (after label `.LBB13`): needs Store wait 0 (War { reg: \"v[5:6]\" })",
        why: LWMMA_MULTI_ST_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_gemm_bf16_t1_m1_n2",
        needle: "`v_lshrrev_b32_e32 v3, 4, v11` (after label `.LBB13`): needs Store wait 0 (War { reg: \"v[3:4]\" })",
        why: LWMMA_MULTI_ST_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_gemm_bf16_t1_m2_n1",
        needle: "`v_bfe_u32 v5, v12, 5, 3` (after label `.LBB13`): needs Store wait 0 (War { reg: \"v[5:6]\" })",
        why: LWMMA_MULTI_ST_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_gemm_bf16_t1_m2_n2",
        needle: "`v_add_nc_u32_e32 v9, 0x400, v36` (after label `.LBB8`): needs Ds wait 1 (War { reg: \"v9\" })",
        why: LWMMA_MULTI_DS_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_gemm_bf16_t1_m3_n1",
        needle: "`v_add_nc_u32_e32 v17, 0x400, v25` (after label `.LBB8`): needs Ds wait 0 (War { reg: \"v17\" })",
        why: LWMMA_MULTI_DS_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_gemm_bf16_t1_m3_n2",
        needle: "`v_add_nc_u32_e32 v25, 0x400, v52` (after label `.LBB8`): needs Ds wait 1 (War { reg: \"v25\" })",
        why: LWMMA_MULTI_DS_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_gemm_bf16_t1_m4_n1",
        needle: "`v_add_nc_u32_e32 v25, 0x400, v33` (after label `.LBB8`): needs Ds wait 1 (War { reg: \"v25\" })",
        why: LWMMA_MULTI_DS_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_gemm_bf16_t1_m4_n2",
        needle: "`v_add_nc_u32_e32 v33, 0x400, v68` (after label `.LBB8`): needs Ds wait 2 (War { reg: \"v33\" })",
        why: LWMMA_MULTI_DS_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_gemm_bf16_t2_m1_n1",
        needle: "`v_lshlrev_b32_e32 v5, 2, v15` (after label `.LBB13`): needs Store wait 0 (War { reg: \"v[5:6]\" })",
        why: LWMMA_MULTI_ST_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_gemm_bf16_t2_m1_n2",
        needle: "`v_lshrrev_b32_e32 v3, 4, v11` (after label `.LBB13`): needs Store wait 0 (War { reg: \"v[3:4]\" })",
        why: LWMMA_MULTI_ST_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_gemm_bf16_t2_m2_n1",
        needle: "`v_bfe_u32 v5, v12, 5, 3` (after label `.LBB13`): needs Store wait 0 (War { reg: \"v[5:6]\" })",
        why: LWMMA_MULTI_ST_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_gemm_bf16_t2_m2_n2",
        needle: "`v_add_nc_u32_e32 v9, 0x400, v33` (after label `.LBB8`): needs Ds wait 1 (War { reg: \"v9\" })",
        why: LWMMA_MULTI_DS_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_gemm_bf16_t2_m3_n1",
        needle: "`v_add_nc_u32_e32 v17, 0x400, v25` (after label `.LBB8`): needs Ds wait 0 (War { reg: \"v17\" })",
        why: LWMMA_MULTI_DS_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_gemm_bf16_t2_m3_n2",
        needle: "`v_add_nc_u32_e32 v25, 0x400, v49` (after label `.LBB8`): needs Ds wait 1 (War { reg: \"v25\" })",
        why: LWMMA_MULTI_DS_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_gemm_bf16_t2_m4_n1",
        needle: "`v_add_nc_u32_e32 v25, 0x400, v33` (after label `.LBB8`): needs Ds wait 1 (War { reg: \"v25\" })",
        why: LWMMA_MULTI_DS_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_gemm_bf16_t2_m4_n2",
        needle: "`v_add_nc_u32_e32 v33, 0x400, v65` (after label `.LBB8`): needs Ds wait 2 (War { reg: \"v33\" })",
        why: LWMMA_MULTI_DS_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_gemm_bf16_t3_m1_n1",
        needle: "`v_lshlrev_b32_e32 v5, 2, v12` (after label `.LBB13`): needs Store wait 0 (War { reg: \"v[5:6]\" })",
        why: LWMMA_MULTI_ST_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_gemm_bf16_t3_m1_n2",
        needle: "`v_lshrrev_b32_e32 v3, 4, v11` (after label `.LBB13`): needs Store wait 0 (War { reg: \"v[3:4]\" })",
        why: LWMMA_MULTI_ST_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_gemm_bf16_t3_m2_n1",
        needle: "`v_bfe_u32 v5, v12, 5, 3` (after label `.LBB13`): needs Store wait 0 (War { reg: \"v[5:6]\" })",
        why: LWMMA_MULTI_ST_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_gemm_bf16_t3_m2_n2",
        needle: "`v_add_nc_u32_e32 v9, 0x400, v33` (after label `.LBB8`): needs Ds wait 1 (War { reg: \"v9\" })",
        why: LWMMA_MULTI_DS_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_gemm_bf16_t3_m3_n1",
        needle: "`v_add_nc_u32_e32 v17, 0x400, v25` (after label `.LBB8`): needs Ds wait 0 (War { reg: \"v17\" })",
        why: LWMMA_MULTI_DS_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_gemm_bf16_t3_m3_n2",
        needle: "`v_add_nc_u32_e32 v25, 0x400, v49` (after label `.LBB8`): needs Ds wait 1 (War { reg: \"v25\" })",
        why: LWMMA_MULTI_DS_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_gemm_bf16_t3_m4_n1",
        needle: "`v_add_nc_u32_e32 v25, 0x400, v33` (after label `.LBB8`): needs Ds wait 1 (War { reg: \"v25\" })",
        why: LWMMA_MULTI_DS_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_gemm_bf16_t3_m4_n2",
        needle: "`v_add_nc_u32_e32 v33, 0x400, v65` (after label `.LBB8`): needs Ds wait 2 (War { reg: \"v33\" })",
        why: LWMMA_MULTI_DS_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_gemm_w8_t1_m1_n1",
        needle: "`v_lshlrev_b32_e32 v6, 2, v15` (after label `.LBB13`): needs Store wait 0 (War { reg: \"v[6:7]\" })",
        why: LWMMA_MULTI_ST_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_gemm_w8_t1_m1_n2",
        needle: "`v_lshrrev_b32_e32 v3, 4, v11` (after label `.LBB13`): needs Store wait 0 (War { reg: \"v[3:4]\" })",
        why: LWMMA_MULTI_ST_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_gemm_w8_t1_m2_n1",
        needle: "`v_bfe_u32 v5, v13, 5, 3` (after label `.LBB13`): needs Store wait 0 (War { reg: \"v[5:6]\" })",
        why: LWMMA_MULTI_ST_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_gemm_w8_t1_m2_n2",
        needle: "`v_add_nc_u32_e32 v20, 0xc00, v20` (after label `.LBB8`): needs Ds wait 0 (War { reg: \"v20\" })",
        why: LWMMA_MULTI_DS_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_gemm_w8_t1_m3_n1",
        needle: "`v_add_nc_u32_e32 v10, 0x800, v10` (after label `.LBB7`): needs Ds wait 0 (War { reg: \"v10\" })",
        why: LWMMA_MULTI_DS_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_gemm_w8_t1_m3_n2",
        needle: "`v_add_nc_u32_e32 v5, 0x1000, v4` (after label `.LBB8`): needs Ds wait 5 (War { reg: \"v5\" })",
        why: LWMMA_MULTI_DS_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_gemm_w8_t1_m4_n1",
        needle: "`v_cmp_gt_i32_e64 s1, s16, v9` (after label `.LBB8`): needs Km wait 0 (Waw { reg: \"s[2:3]\" })",
        why: LWMMA_MULTI_KM_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_gemm_w8_t1_m4_n2",
        needle: "`v_add_nc_u32_e32 v2, 0x1400, v1` (after label `.LBB8`): needs Ds wait 8 (War { reg: \"v2\" })",
        why: LWMMA_MULTI_DS_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_gemm_w8_t2_m1_n1",
        needle: "`v_lshlrev_b32_e32 v5, 2, v13` (after label `.LBB13`): needs Store wait 0 (War { reg: \"v[5:6]\" })",
        why: LWMMA_MULTI_ST_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_gemm_w8_t2_m1_n2",
        needle: "`v_lshrrev_b32_e32 v3, 4, v11` (after label `.LBB13`): needs Store wait 0 (War { reg: \"v[3:4]\" })",
        why: LWMMA_MULTI_ST_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_gemm_w8_t2_m2_n1",
        needle: "`v_bfe_u32 v5, v12, 5, 3` (after label `.LBB13`): needs Store wait 0 (War { reg: \"v[5:6]\" })",
        why: LWMMA_MULTI_ST_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_gemm_w8_t2_m2_n2",
        needle: "`v_add_nc_u32_e32 v1, 0xc00, v1` (after label `.LBB8`): needs Ds wait 0 (War { reg: \"v1\" })",
        why: LWMMA_MULTI_DS_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_gemm_w8_t2_m3_n1",
        needle: "`v_add_nc_u32_e32 v1, 0x800, v1` (after label `.LBB7`): needs Ds wait 0 (War { reg: \"v1\" })",
        why: LWMMA_MULTI_DS_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_gemm_w8_t2_m3_n2",
        needle: "`v_add_nc_u32_e32 v2, 0x1000, v1` (after label `.LBB8`): needs Ds wait 5 (War { reg: \"v2\" })",
        why: LWMMA_MULTI_DS_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_gemm_w8_t2_m4_n1",
        needle: "`v_cmp_gt_i32_e64 s1, s16, v69` (after label `.LBB8`): needs Km wait 0 (Waw { reg: \"s[2:3]\" })",
        why: LWMMA_MULTI_KM_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_gemm_w8_t2_m4_n2",
        needle: "`v_add_nc_u32_e32 v2, 0x1400, v1` (after label `.LBB8`): needs Ds wait 8 (War { reg: \"v2\" })",
        why: LWMMA_MULTI_DS_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_gemm_w8_t3_m1_n1",
        needle: "`v_lshlrev_b32_e32 v5, 2, v12` (after label `.LBB13`): needs Store wait 0 (War { reg: \"v[5:6]\" })",
        why: LWMMA_MULTI_ST_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_gemm_w8_t3_m1_n2",
        needle: "`v_lshrrev_b32_e32 v3, 4, v11` (after label `.LBB13`): needs Store wait 0 (War { reg: \"v[3:4]\" })",
        why: LWMMA_MULTI_ST_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_gemm_w8_t3_m2_n1",
        needle: "`v_bfe_u32 v5, v12, 5, 3` (after label `.LBB13`): needs Store wait 0 (War { reg: \"v[5:6]\" })",
        why: LWMMA_MULTI_ST_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_gemm_w8_t3_m2_n2",
        needle: "`v_add_nc_u32_e32 v1, 0xc00, v1` (after label `.LBB8`): needs Ds wait 0 (War { reg: \"v1\" })",
        why: LWMMA_MULTI_DS_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_gemm_w8_t3_m3_n1",
        needle: "`v_add_nc_u32_e32 v1, 0x800, v1` (after label `.LBB7`): needs Ds wait 0 (War { reg: \"v1\" })",
        why: LWMMA_MULTI_DS_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_gemm_w8_t3_m3_n2",
        needle: "`v_add_nc_u32_e32 v2, 0x1000, v1` (after label `.LBB8`): needs Ds wait 5 (War { reg: \"v2\" })",
        why: LWMMA_MULTI_DS_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_gemm_w8_t3_m4_n1",
        needle: "`v_add_nc_u32_e32 v1, 0xc00, v1` (after label `.LBB8`): needs Ds wait 0 (War { reg: \"v1\" })",
        why: LWMMA_MULTI_DS_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_gemm_w8_t3_m4_n2",
        needle: "`v_add_nc_u32_e32 v2, 0x1400, v1` (after label `.LBB8`): needs Ds wait 8 (War { reg: \"v2\" })",
        why: LWMMA_MULTI_DS_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_gemm_bf16_t2_m1_n1_multi",
        needle: "`v_add_nc_u32_e32 v5, 0x80, v4` (after label `.LBB27`): needs Store wait 0 (War { reg: \"v[5:6]\" })",
        why: LWMMA_MULTI_ST_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_gemm_bf16_t2_m2_n1_multi",
        needle: "`v_add_nc_u32_e32 v5, 0x80, v4` (after label `.LBB28`): needs Store wait 0 (War { reg: \"v[5:6]\" })",
        why: LWMMA_MULTI_ST_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_gemm_bf16_t2_m3_n1_multi",
        needle: "`v_add_nc_u32_e32 v17, 0x400, v25` (after label `.LBB26`): needs Ds wait 0 (War { reg: \"v17\" })",
        why: LWMMA_MULTI_DS_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_gemm_bf16_t2_m4_n1_multi",
        needle: "`v_add_nc_u32_e32 v25, 0x400, v33` (after label `.LBB26`): needs Ds wait 0 (War { reg: \"v25\" })",
        why: LWMMA_MULTI_DS_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_read_up_bf16_t1_g1",
        needle: "`v_add_nc_u32_e32 v1, 0x400, v36` (after label `.LBB10`): needs Ds wait 1 (War { reg: \"v1\" })",
        why: LWMMA_MULTI_DS_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_read_up_bf16_t1_g2",
        needle: "`v_add_nc_u32_e32 v1, 0x400, v35` (after label `.LBB6`): needs Ds wait 1 (War { reg: \"v1\" })",
        why: LWMMA_MULTI_DS_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_read_up_bf16_t1_g4",
        needle: "`v_add_nc_u32_e32 v1, 0x400, v37` (after label `.LBB6`): needs Ds wait 1 (War { reg: \"v1\" })",
        why: LWMMA_MULTI_DS_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_read_up_bf16_t2_g1",
        needle: "`v_add_nc_u32_e32 v1, 0x400, v36` (after label `.LBB10`): needs Ds wait 1 (War { reg: \"v1\" })",
        why: LWMMA_MULTI_DS_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_read_up_bf16_t2_g2",
        needle: "`v_add_nc_u32_e32 v1, 0x400, v35` (after label `.LBB6`): needs Ds wait 1 (War { reg: \"v1\" })",
        why: LWMMA_MULTI_DS_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_read_up_bf16_t2_g4",
        needle: "`v_add_nc_u32_e32 v1, 0x400, v37` (after label `.LBB6`): needs Ds wait 1 (War { reg: \"v1\" })",
        why: LWMMA_MULTI_DS_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_read_up_bf16_t3_g1",
        needle: "`v_add_nc_u32_e32 v1, 0x400, v35` (after label `.LBB10`): needs Ds wait 1 (War { reg: \"v1\" })",
        why: LWMMA_MULTI_DS_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_read_up_bf16_t3_g2",
        needle: "`v_add_nc_u32_e32 v1, 0x400, v35` (after label `.LBB6`): needs Ds wait 1 (War { reg: \"v1\" })",
        why: LWMMA_MULTI_DS_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_read_up_bf16_t3_g4",
        needle: "`v_add_nc_u32_e32 v1, 0x400, v35` (after label `.LBB6`): needs Ds wait 1 (War { reg: \"v1\" })",
        why: LWMMA_MULTI_DS_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_read_up_w8_t1_g2",
        needle: "`v_add_nc_u32_e32 v3, 0xc00, v3` (after label `.LBB6`): needs Ds wait 0 (War { reg: \"v3\" })",
        why: LWMMA_MULTI_DS_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_read_up_w8_t1_g4",
        needle: "`v_add_nc_u32_e32 v3, 0xc00, v3` (after label `.LBB6`): needs Ds wait 0 (War { reg: \"v3\" })",
        why: LWMMA_MULTI_DS_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_read_up_w8_t2_g2",
        needle: "`v_add_nc_u32_e32 v3, 0xc00, v3` (after label `.LBB6`): needs Ds wait 0 (War { reg: \"v3\" })",
        why: LWMMA_MULTI_DS_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_read_up_w8_t2_g4",
        needle: "`v_add_nc_u32_e32 v3, 0xc00, v3` (after label `.LBB6`): needs Ds wait 0 (War { reg: \"v3\" })",
        why: LWMMA_MULTI_DS_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_read_up_w8_t3_g2",
        needle: "`v_add_nc_u32_e32 v3, 0xc00, v3` (after label `.LBB6`): needs Ds wait 0 (War { reg: \"v3\" })",
        why: LWMMA_MULTI_DS_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_read_up_w8_t3_g4",
        needle: "`v_add_nc_u32_e32 v3, 0xc00, v3` (after label `.LBB6`): needs Ds wait 0 (War { reg: \"v3\" })",
        why: LWMMA_MULTI_DS_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_gemm_bf16_t1_m1_n1",
        needle: "`v_lshlrev_b32_e32 v5, 2, v15` (after label `.LBB13`): needs Store wait 0 (War { reg: \"v[5:6]\" })",
        why: LWMMA_MULTI_ST_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_gemm_bf16_t1_m1_n2",
        needle: "`v_lshrrev_b32_e32 v3, 4, v11` (after label `.LBB13`): needs Store wait 0 (War { reg: \"v[3:4]\" })",
        why: LWMMA_MULTI_ST_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_gemm_bf16_t1_m2_n1",
        needle: "`v_bfe_u32 v5, v12, 5, 3` (after label `.LBB13`): needs Store wait 0 (War { reg: \"v[5:6]\" })",
        why: LWMMA_MULTI_ST_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_gemm_bf16_t1_m2_n2",
        needle: "`v_add_nc_u32_e32 v9, 0x400, v36` (after label `.LBB8`): needs Ds wait 1 (War { reg: \"v9\" })",
        why: LWMMA_MULTI_DS_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_gemm_bf16_t1_m3_n1",
        needle: "`v_add_nc_u32_e32 v17, 0x400, v25` (after label `.LBB8`): needs Ds wait 0 (War { reg: \"v17\" })",
        why: LWMMA_MULTI_DS_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_gemm_bf16_t1_m3_n2",
        needle: "`v_add_nc_u32_e32 v25, 0x400, v52` (after label `.LBB8`): needs Ds wait 1 (War { reg: \"v25\" })",
        why: LWMMA_MULTI_DS_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_gemm_bf16_t1_m4_n1",
        needle: "`v_add_nc_u32_e32 v25, 0x400, v33` (after label `.LBB8`): needs Ds wait 1 (War { reg: \"v25\" })",
        why: LWMMA_MULTI_DS_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_gemm_bf16_t1_m4_n2",
        needle: "`v_add_nc_u32_e32 v33, 0x400, v68` (after label `.LBB8`): needs Ds wait 2 (War { reg: \"v33\" })",
        why: LWMMA_MULTI_DS_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_gemm_bf16_t2_m1_n1",
        needle: "`v_lshlrev_b32_e32 v5, 2, v15` (after label `.LBB13`): needs Store wait 0 (War { reg: \"v[5:6]\" })",
        why: LWMMA_MULTI_ST_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_gemm_bf16_t2_m1_n2",
        needle: "`v_lshrrev_b32_e32 v3, 4, v11` (after label `.LBB13`): needs Store wait 0 (War { reg: \"v[3:4]\" })",
        why: LWMMA_MULTI_ST_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_gemm_bf16_t2_m2_n1",
        needle: "`v_bfe_u32 v5, v12, 5, 3` (after label `.LBB13`): needs Store wait 0 (War { reg: \"v[5:6]\" })",
        why: LWMMA_MULTI_ST_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_gemm_bf16_t2_m2_n2",
        needle: "`v_add_nc_u32_e32 v9, 0x400, v33` (after label `.LBB8`): needs Ds wait 1 (War { reg: \"v9\" })",
        why: LWMMA_MULTI_DS_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_gemm_bf16_t2_m3_n1",
        needle: "`v_add_nc_u32_e32 v17, 0x400, v25` (after label `.LBB8`): needs Ds wait 0 (War { reg: \"v17\" })",
        why: LWMMA_MULTI_DS_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_gemm_bf16_t2_m3_n2",
        needle: "`v_add_nc_u32_e32 v25, 0x400, v49` (after label `.LBB8`): needs Ds wait 1 (War { reg: \"v25\" })",
        why: LWMMA_MULTI_DS_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_gemm_bf16_t2_m4_n1",
        needle: "`v_add_nc_u32_e32 v25, 0x400, v33` (after label `.LBB8`): needs Ds wait 1 (War { reg: \"v25\" })",
        why: LWMMA_MULTI_DS_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_gemm_bf16_t2_m4_n2",
        needle: "`v_add_nc_u32_e32 v33, 0x400, v65` (after label `.LBB8`): needs Ds wait 2 (War { reg: \"v33\" })",
        why: LWMMA_MULTI_DS_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_gemm_bf16_t3_m1_n1",
        needle: "`v_lshlrev_b32_e32 v5, 2, v12` (after label `.LBB13`): needs Store wait 0 (War { reg: \"v[5:6]\" })",
        why: LWMMA_MULTI_ST_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_gemm_bf16_t3_m1_n2",
        needle: "`v_lshrrev_b32_e32 v3, 4, v11` (after label `.LBB13`): needs Store wait 0 (War { reg: \"v[3:4]\" })",
        why: LWMMA_MULTI_ST_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_gemm_bf16_t3_m2_n1",
        needle: "`v_bfe_u32 v5, v12, 5, 3` (after label `.LBB13`): needs Store wait 0 (War { reg: \"v[5:6]\" })",
        why: LWMMA_MULTI_ST_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_gemm_bf16_t3_m2_n2",
        needle: "`v_add_nc_u32_e32 v9, 0x400, v33` (after label `.LBB8`): needs Ds wait 1 (War { reg: \"v9\" })",
        why: LWMMA_MULTI_DS_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_gemm_bf16_t3_m3_n1",
        needle: "`v_add_nc_u32_e32 v17, 0x400, v25` (after label `.LBB8`): needs Ds wait 0 (War { reg: \"v17\" })",
        why: LWMMA_MULTI_DS_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_gemm_bf16_t3_m3_n2",
        needle: "`v_add_nc_u32_e32 v25, 0x400, v49` (after label `.LBB8`): needs Ds wait 1 (War { reg: \"v25\" })",
        why: LWMMA_MULTI_DS_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_gemm_bf16_t3_m4_n1",
        needle: "`v_add_nc_u32_e32 v25, 0x400, v33` (after label `.LBB8`): needs Ds wait 1 (War { reg: \"v25\" })",
        why: LWMMA_MULTI_DS_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_gemm_bf16_t3_m4_n2",
        needle: "`v_add_nc_u32_e32 v33, 0x400, v65` (after label `.LBB8`): needs Ds wait 2 (War { reg: \"v33\" })",
        why: LWMMA_MULTI_DS_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_gemm_w8_t1_m1_n1",
        needle: "`v_lshlrev_b32_e32 v6, 2, v15` (after label `.LBB13`): needs Store wait 0 (War { reg: \"v[6:7]\" })",
        why: LWMMA_MULTI_ST_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_gemm_w8_t1_m1_n2",
        needle: "`v_lshrrev_b32_e32 v3, 4, v11` (after label `.LBB13`): needs Store wait 0 (War { reg: \"v[3:4]\" })",
        why: LWMMA_MULTI_ST_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_gemm_w8_t1_m2_n1",
        needle: "`v_bfe_u32 v5, v13, 5, 3` (after label `.LBB13`): needs Store wait 0 (War { reg: \"v[5:6]\" })",
        why: LWMMA_MULTI_ST_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_gemm_w8_t1_m2_n2",
        needle: "`v_add_nc_u32_e32 v20, 0xc00, v20` (after label `.LBB8`): needs Ds wait 0 (War { reg: \"v20\" })",
        why: LWMMA_MULTI_DS_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_gemm_w8_t1_m3_n1",
        needle: "`v_add_nc_u32_e32 v10, 0x800, v10` (after label `.LBB7`): needs Ds wait 0 (War { reg: \"v10\" })",
        why: LWMMA_MULTI_DS_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_gemm_w8_t1_m3_n2",
        needle: "`v_add_nc_u32_e32 v5, 0x1000, v4` (after label `.LBB8`): needs Ds wait 5 (War { reg: \"v5\" })",
        why: LWMMA_MULTI_DS_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_gemm_w8_t1_m4_n1",
        needle: "`v_cmp_gt_i32_e64 s1, s16, v9` (after label `.LBB8`): needs Km wait 0 (Waw { reg: \"s[2:3]\" })",
        why: LWMMA_MULTI_KM_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_gemm_w8_t1_m4_n2",
        needle: "`v_add_nc_u32_e32 v2, 0x1400, v1` (after label `.LBB8`): needs Ds wait 8 (War { reg: \"v2\" })",
        why: LWMMA_MULTI_DS_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_gemm_w8_t2_m1_n1",
        needle: "`v_lshlrev_b32_e32 v5, 2, v13` (after label `.LBB13`): needs Store wait 0 (War { reg: \"v[5:6]\" })",
        why: LWMMA_MULTI_ST_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_gemm_w8_t2_m1_n2",
        needle: "`v_lshrrev_b32_e32 v3, 4, v11` (after label `.LBB13`): needs Store wait 0 (War { reg: \"v[3:4]\" })",
        why: LWMMA_MULTI_ST_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_gemm_w8_t2_m2_n1",
        needle: "`v_bfe_u32 v5, v12, 5, 3` (after label `.LBB13`): needs Store wait 0 (War { reg: \"v[5:6]\" })",
        why: LWMMA_MULTI_ST_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_gemm_w8_t2_m2_n2",
        needle: "`v_add_nc_u32_e32 v1, 0xc00, v1` (after label `.LBB8`): needs Ds wait 0 (War { reg: \"v1\" })",
        why: LWMMA_MULTI_DS_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_gemm_w8_t2_m3_n1",
        needle: "`v_add_nc_u32_e32 v1, 0x800, v1` (after label `.LBB7`): needs Ds wait 0 (War { reg: \"v1\" })",
        why: LWMMA_MULTI_DS_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_gemm_w8_t2_m3_n2",
        needle: "`v_add_nc_u32_e32 v2, 0x1000, v1` (after label `.LBB8`): needs Ds wait 5 (War { reg: \"v2\" })",
        why: LWMMA_MULTI_DS_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_gemm_w8_t2_m4_n1",
        needle: "`v_cmp_gt_i32_e64 s1, s16, v69` (after label `.LBB8`): needs Km wait 0 (Waw { reg: \"s[2:3]\" })",
        why: LWMMA_MULTI_KM_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_gemm_w8_t2_m4_n2",
        needle: "`v_add_nc_u32_e32 v2, 0x1400, v1` (after label `.LBB8`): needs Ds wait 8 (War { reg: \"v2\" })",
        why: LWMMA_MULTI_DS_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_gemm_w8_t3_m1_n1",
        needle: "`v_lshlrev_b32_e32 v5, 2, v12` (after label `.LBB13`): needs Store wait 0 (War { reg: \"v[5:6]\" })",
        why: LWMMA_MULTI_ST_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_gemm_w8_t3_m1_n2",
        needle: "`v_lshrrev_b32_e32 v3, 4, v11` (after label `.LBB13`): needs Store wait 0 (War { reg: \"v[3:4]\" })",
        why: LWMMA_MULTI_ST_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_gemm_w8_t3_m2_n1",
        needle: "`v_bfe_u32 v5, v12, 5, 3` (after label `.LBB13`): needs Store wait 0 (War { reg: \"v[5:6]\" })",
        why: LWMMA_MULTI_ST_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_gemm_w8_t3_m2_n2",
        needle: "`v_add_nc_u32_e32 v1, 0xc00, v1` (after label `.LBB8`): needs Ds wait 0 (War { reg: \"v1\" })",
        why: LWMMA_MULTI_DS_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_gemm_w8_t3_m3_n1",
        needle: "`v_add_nc_u32_e32 v1, 0x800, v1` (after label `.LBB7`): needs Ds wait 0 (War { reg: \"v1\" })",
        why: LWMMA_MULTI_DS_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_gemm_w8_t3_m3_n2",
        needle: "`v_add_nc_u32_e32 v2, 0x1000, v1` (after label `.LBB8`): needs Ds wait 5 (War { reg: \"v2\" })",
        why: LWMMA_MULTI_DS_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_gemm_w8_t3_m4_n1",
        needle: "`v_add_nc_u32_e32 v1, 0xc00, v1` (after label `.LBB8`): needs Ds wait 0 (War { reg: \"v1\" })",
        why: LWMMA_MULTI_DS_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_gemm_w8_t3_m4_n2",
        needle: "`v_add_nc_u32_e32 v2, 0x1400, v1` (after label `.LBB8`): needs Ds wait 8 (War { reg: \"v2\" })",
        why: LWMMA_MULTI_DS_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_gemm_bf16_t2_m1_n1_multi",
        needle: "`v_add_nc_u32_e32 v5, 0x80, v4` (after label `.LBB27`): needs Store wait 0 (War { reg: \"v[5:6]\" })",
        why: LWMMA_MULTI_ST_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_gemm_bf16_t2_m2_n1_multi",
        needle: "`v_add_nc_u32_e32 v5, 0x80, v4` (after label `.LBB28`): needs Store wait 0 (War { reg: \"v[5:6]\" })",
        why: LWMMA_MULTI_ST_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_gemm_bf16_t2_m3_n1_multi",
        needle: "`v_add_nc_u32_e32 v17, 0x400, v25` (after label `.LBB26`): needs Ds wait 0 (War { reg: \"v17\" })",
        why: LWMMA_MULTI_DS_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_gemm_bf16_t2_m4_n1_multi",
        needle: "`v_add_nc_u32_e32 v25, 0x400, v33` (after label `.LBB26`): needs Ds wait 0 (War { reg: \"v25\" })",
        why: LWMMA_MULTI_DS_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_read_up_bf16_t1_g1",
        needle: "`v_add_nc_u32_e32 v1, 0x400, v36` (after label `.LBB10`): needs Ds wait 1 (War { reg: \"v1\" })",
        why: LWMMA_MULTI_DS_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_read_up_bf16_t1_g2",
        needle: "`v_add_nc_u32_e32 v1, 0x400, v35` (after label `.LBB6`): needs Ds wait 1 (War { reg: \"v1\" })",
        why: LWMMA_MULTI_DS_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_read_up_bf16_t1_g4",
        needle: "`v_add_nc_u32_e32 v1, 0x400, v37` (after label `.LBB6`): needs Ds wait 1 (War { reg: \"v1\" })",
        why: LWMMA_MULTI_DS_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_read_up_bf16_t2_g1",
        needle: "`v_add_nc_u32_e32 v1, 0x400, v36` (after label `.LBB10`): needs Ds wait 1 (War { reg: \"v1\" })",
        why: LWMMA_MULTI_DS_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_read_up_bf16_t2_g2",
        needle: "`v_add_nc_u32_e32 v1, 0x400, v35` (after label `.LBB6`): needs Ds wait 1 (War { reg: \"v1\" })",
        why: LWMMA_MULTI_DS_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_read_up_bf16_t2_g4",
        needle: "`v_add_nc_u32_e32 v1, 0x400, v37` (after label `.LBB6`): needs Ds wait 1 (War { reg: \"v1\" })",
        why: LWMMA_MULTI_DS_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_read_up_bf16_t3_g1",
        needle: "`v_add_nc_u32_e32 v1, 0x400, v35` (after label `.LBB10`): needs Ds wait 1 (War { reg: \"v1\" })",
        why: LWMMA_MULTI_DS_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_read_up_bf16_t3_g2",
        needle: "`v_add_nc_u32_e32 v1, 0x400, v35` (after label `.LBB6`): needs Ds wait 1 (War { reg: \"v1\" })",
        why: LWMMA_MULTI_DS_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_read_up_bf16_t3_g4",
        needle: "`v_add_nc_u32_e32 v1, 0x400, v35` (after label `.LBB6`): needs Ds wait 1 (War { reg: \"v1\" })",
        why: LWMMA_MULTI_DS_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_read_up_w8_t1_g2",
        needle: "`v_add_nc_u32_e32 v3, 0xc00, v3` (after label `.LBB6`): needs Ds wait 0 (War { reg: \"v3\" })",
        why: LWMMA_MULTI_DS_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_read_up_w8_t1_g4",
        needle: "`v_add_nc_u32_e32 v3, 0xc00, v3` (after label `.LBB6`): needs Ds wait 0 (War { reg: \"v3\" })",
        why: LWMMA_MULTI_DS_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_read_up_w8_t2_g2",
        needle: "`v_add_nc_u32_e32 v3, 0xc00, v3` (after label `.LBB6`): needs Ds wait 0 (War { reg: \"v3\" })",
        why: LWMMA_MULTI_DS_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_read_up_w8_t2_g4",
        needle: "`v_add_nc_u32_e32 v3, 0xc00, v3` (after label `.LBB6`): needs Ds wait 0 (War { reg: \"v3\" })",
        why: LWMMA_MULTI_DS_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_read_up_w8_t3_g2",
        needle: "`v_add_nc_u32_e32 v3, 0xc00, v3` (after label `.LBB6`): needs Ds wait 0 (War { reg: \"v3\" })",
        why: LWMMA_MULTI_DS_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_read_up_w8_t3_g4",
        needle: "`v_add_nc_u32_e32 v3, 0xc00, v3` (after label `.LBB6`): needs Ds wait 0 (War { reg: \"v3\" })",
        why: LWMMA_MULTI_DS_WHY,
    },
    Allow {
        scope: "pins",
        target: "moe_route_scatter_wide_hipcc_gfx1201.co#moe_router_softmax_top10_f32_fast",
        needle: "`v_mov_b32_e32 v2, v1` (after label `.LBB6`): needs Store wait 0 (War { reg: \"v[2:5]\" })",
        why: X8_ST_WHY,
    },
    Allow {
        scope: "pins",
        target: "moe_route_scatter_wide_hipcc_gfx1201.co#moe_scatter_fused_k8",
        needle: "`v_add_nc_u32_e32 v1, s4, v1` (after label `.LBB2`): needs Ds wait 0 (War { reg: \"v1\" })",
        why: X8_DS_WHY,
    },
    Allow {
        scope: "pins",
        target: "moe_route_scatter_wide_hipcc_gfx1201.co#moe_scatter_fused_top10_pmask",
        needle: "`v_add_nc_u32_e32 v1, s4, v1` (after label `.LBB2`): needs Ds wait 0 (War { reg: \"v1\" })",
        why: X8_DS_WHY,
    },
    Allow {
        scope: "pins",
        target: "moe_route_scatter_wide_hipcc_gfx1201.co#moe_route_scatter_top10_wide_f32",
        needle: "`v_mov_b32_e32 v3, v2` (after label `.LBB5`): needs Store wait 0 (War { reg: \"v[3:6]\" })",
        why: X8_ST_WHY,
    },
    Allow {
        scope: "pins",
        target: "moe_route_scatter_wide_pm_gfx1201.hxaco#moe_router_softmax_top10_f32_fast",
        needle: "`v_mov_b32_e32 v2, v1` (after label `.LBB6`): needs Store wait 0 (War { reg: \"v[2:5]\" })",
        why: X8_ST_WHY,
    },
    Allow {
        scope: "pins",
        target: "moe_route_scatter_wide_pm_gfx1201.hxaco#moe_scatter_fused_k8",
        needle: "`v_add_nc_u32_e32 v1, s4, v1` (after label `.LBB2`): needs Ds wait 0 (War { reg: \"v1\" })",
        why: X8_DS_WHY,
    },
    Allow {
        scope: "pins",
        target: "moe_route_scatter_wide_pm_gfx1201.hxaco#moe_scatter_fused_top10_pmask",
        needle: "`v_add_nc_u32_e32 v1, s4, v1` (after label `.LBB2`): needs Ds wait 0 (War { reg: \"v1\" })",
        why: X8_DS_WHY,
    },
    Allow {
        scope: "pins",
        target: "moe_route_scatter_wide_pm_gfx1201.hxaco#moe_route_scatter_top10_wide_f32",
        needle: "`v_mov_b32_e32 v3, v2` (after label `.LBB5`): needs Store wait 0 (War { reg: \"v[3:6]\" })",
        why: X8_ST_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_moe_ep_a8_rot_hipcc_gfx1201.co#qwen4_moe_ep_a8_gate_up_nt1",
        needle: "`flat_load_b128 v[0:3], v[6:7], off offset:8` (after label `.LBB4`): needs Load wait 0 (War { reg: \"v[0:1]\" })",
        why: X8_LOAD_WAR_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_moe_ep_a8_rot_hipcc_gfx1201.co#qwen4_moe_ep_a8_gate_up_nt2",
        needle: "`flat_load_d16_hi_b16 v32, v[72:73], off` (after label `.LBB14`): needs Load wait 0 (Use { reg: \"v32\" })",
        why: X8_LOAD_USE_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_moe_ep_a8_rot_hipcc_gfx1201.co#qwen4_moe_ep_a8_gate_up_planar_nt1",
        needle: "`flat_load_d16_b16 v32, v[6:7], off` (after label `.LBB4`): needs Load wait 0 (Use { reg: \"v32\" })",
        why: X8_LOAD_USE_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_moe_ep_a8_rot_hipcc_gfx1201.co#qwen4_moe_ep_a8_gate_up_planar_nt2",
        needle: "`flat_load_d16_b16 v32, v[72:73], off` (after label `.LBB14`): needs Load wait 0 (Use { reg: \"v32\" })",
        why: X8_LOAD_USE_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_moe_ep_a8_rot_hipcc_gfx1201.co#qwen4_moe_ep_a8_down_nt1",
        needle: "`flat_load_b128 v[12:15], v[12:13], off offset:20` (after label `.LBB4`): needs Load wait 0 (War { reg: \"v[12:13]\" })",
        why: X8_LOAD_WAR_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_moe_ep_a8_rot_hipcc_gfx1201.co#qwen4_moe_ep_a8_down_nt2",
        needle: "`v_cvt_f32_f16_e32 v63, v24.l` (after label `.LBB20`): needs Load wait 0 (Use { reg: \"v24\" })",
        why: X8_LOAD_USE_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_moe_ep_a8_rot_hipcc_gfx1201.co#qwen4_moe_ep_a8_down_planar_nt1",
        needle: "`flat_load_b128 v[8:11], v[1:2], off` (after label `.LBB4`): needs Load wait 0 (War { reg: \"v[8:9]\" })",
        why: X8_LOAD_WAR_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_moe_ep_a8_rot_hipcc_gfx1201.co#qwen4_moe_ep_a8_down_planar_nt2",
        needle: "`v_cvt_f32_f16_e32 v65, v24.l` (after label `.LBB20`): needs Load wait 0 (Use { reg: \"v24\" })",
        why: X8_LOAD_USE_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_moe_ep_a8_rot_hipcc_gfx1201.co#qwen4_moe_ep_a8_gate_up_rot_planar_nt1",
        needle: "`flat_load_d16_hi_b16 v36, v[21:22], off` (after label `.LBB7`): needs Load wait 0 (Use { reg: \"v36\" })",
        why: X8_LOAD_USE_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_moe_ep_a8_rot_hipcc_gfx1201.co#qwen4_moe_ep_a8_gate_up_rot_planar_nt2",
        needle: "`v_cmp_le_u32_e64 s1, s5, v2` (after label `.LBB12`): needs Km wait 0 (Waw { reg: \"s[2:3]\" })",
        why: X8_KM_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_moe_ep_a8_rot_pm_gfx1201.hxaco#qwen4_moe_ep_a8_gate_up_nt1",
        needle: "`flat_load_b128 v[0:3], v[6:7], off offset:8` (after label `.LBB4`): needs Load wait 0 (War { reg: \"v[0:1]\" })",
        why: X8_LOAD_WAR_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_moe_ep_a8_rot_pm_gfx1201.hxaco#qwen4_moe_ep_a8_gate_up_nt2",
        needle: "`flat_load_d16_hi_b16 v32, v[72:73], off` (after label `.LBB14`): needs Load wait 0 (Use { reg: \"v32\" })",
        why: X8_LOAD_USE_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_moe_ep_a8_rot_pm_gfx1201.hxaco#qwen4_moe_ep_a8_gate_up_planar_nt1",
        needle: "`flat_load_d16_b16 v32, v[6:7], off` (after label `.LBB4`): needs Load wait 0 (Use { reg: \"v32\" })",
        why: X8_LOAD_USE_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_moe_ep_a8_rot_pm_gfx1201.hxaco#qwen4_moe_ep_a8_gate_up_planar_nt2",
        needle: "`flat_load_d16_b16 v32, v[72:73], off` (after label `.LBB14`): needs Load wait 0 (Use { reg: \"v32\" })",
        why: X8_LOAD_USE_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_moe_ep_a8_rot_pm_gfx1201.hxaco#qwen4_moe_ep_a8_down_nt1",
        needle: "`flat_load_b128 v[12:15], v[12:13], off offset:20` (after label `.LBB4`): needs Load wait 0 (War { reg: \"v[12:13]\" })",
        why: X8_LOAD_WAR_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_moe_ep_a8_rot_pm_gfx1201.hxaco#qwen4_moe_ep_a8_down_nt2",
        needle: "`v_cvt_f32_f16_e32 v63, v24.l` (after label `.LBB20`): needs Load wait 0 (Use { reg: \"v24\" })",
        why: X8_LOAD_USE_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_moe_ep_a8_rot_pm_gfx1201.hxaco#qwen4_moe_ep_a8_down_planar_nt1",
        needle: "`flat_load_b128 v[8:11], v[1:2], off` (after label `.LBB4`): needs Load wait 0 (War { reg: \"v[8:9]\" })",
        why: X8_LOAD_WAR_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_moe_ep_a8_rot_pm_gfx1201.hxaco#qwen4_moe_ep_a8_down_planar_nt2",
        needle: "`v_cvt_f32_f16_e32 v65, v24.l` (after label `.LBB20`): needs Load wait 0 (Use { reg: \"v24\" })",
        why: X8_LOAD_USE_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_moe_ep_a8_rot_pm_gfx1201.hxaco#qwen4_moe_ep_a8_gate_up_rot_planar_nt1",
        needle: "`flat_load_d16_hi_b16 v36, v[21:22], off` (after label `.LBB7`): needs Load wait 0 (Use { reg: \"v36\" })",
        why: X8_LOAD_USE_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_moe_ep_a8_rot_pm_gfx1201.hxaco#qwen4_moe_ep_a8_gate_up_rot_planar_nt2",
        needle: "`v_cmp_le_u32_e64 s1, s5, v2` (after label `.LBB12`): needs Km wait 0 (Waw { reg: \"s[2:3]\" })",
        why: X8_KM_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lwmma_tile_hipcc_gfx1201.co#lwmma_tile_bf16_t1_b32x32w2x1g4",
        needle: "`v_or_b32_e32 v4, 1, v3` (after label `.LBB25`): needs Store wait 0 (War { reg: \"v[4:5]\" })",
        why: LWMMA_TILE_ST_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lwmma_tile_hipcc_gfx1201.co#lwmma_tile_bf16_t1_b32x64w2x2",
        needle: "`v_or_b32_e32 v4, 1, v3` (after label `.LBB48`): needs Store wait 0 (War { reg: \"v[4:5]\" })",
        why: LWMMA_TILE_ST_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lwmma_tile_hipcc_gfx1201.co#lwmma_tile_bf16_t1_b64x64w4x1",
        needle: "`v_or_b32_e32 v4, 1, v3` (after label `.LBB72`): needs Store wait 0 (War { reg: \"v[4:5]\" })",
        why: LWMMA_TILE_ST_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lwmma_tile_hipcc_gfx1201.co#lwmma_tile_bf16_t2_b16x32w1x1g4",
        needle: "`v_cmp_ne_u32_e64 s1, 0, v45` (after label `.LBB1`): needs Km wait 0 (Waw { reg: \"s[2:3]\" })",
        why: LWMMA_TILE_KM_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lwmma_tile_hipcc_gfx1201.co#lwmma_tile_bf16_t2_b32x64w2x2",
        needle: "`v_or_b32_e32 v4, 1, v3` (after label `.LBB48`): needs Store wait 0 (War { reg: \"v[4:5]\" })",
        why: LWMMA_TILE_ST_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lwmma_tile_hipcc_gfx1201.co#lwmma_tile_bf16_t2_b64x64w4x1",
        needle: "`v_or_b32_e32 v4, 1, v3` (after label `.LBB72`): needs Store wait 0 (War { reg: \"v[4:5]\" })",
        why: LWMMA_TILE_ST_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lwmma_tile_hipcc_gfx1201.co#lwmma_tile_bf16_t2_b64x64w4x2",
        needle: "`v_or_b32_e32 v4, 1, v3` (after label `.LBB48`): needs Store wait 0 (War { reg: \"v[4:5]\" })",
        why: LWMMA_TILE_ST_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lwmma_tile_hipcc_gfx1201.co#lwmma_tile_bf16_t2_b32x128w2x2",
        needle: "`v_or_b32_e32 v4, 1, v3` (after label `.LBB72`): needs Store wait 0 (War { reg: \"v[4:5]\" })",
        why: LWMMA_TILE_ST_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lwmma_tile_hipcc_gfx1201.co#lwmma_tile_bf16_t2_b64x128w4x2",
        needle: "`v_or_b32_e32 v4, 1, v3` (after label `.LBB72`): needs Store wait 0 (War { reg: \"v[4:5]\" })",
        why: LWMMA_TILE_ST_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lwmma_tile_hipcc_gfx1201.co#lwmma_tile_bf16_t2_b64x128w2x4",
        needle: "`v_or_b32_e32 v4, 1, v3` (after label `.LBB63`): needs Store wait 0 (War { reg: \"v[4:5]\" })",
        why: LWMMA_TILE_ST_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lwmma_tile_pm_gfx1201.hxaco#lwmma_tile_bf16_t1_b32x32w2x1g4",
        needle: "`v_or_b32_e32 v4, 1, v3` (after label `.LBB25`): needs Store wait 0 (War { reg: \"v[4:5]\" })",
        why: LWMMA_TILE_ST_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lwmma_tile_pm_gfx1201.hxaco#lwmma_tile_bf16_t1_b32x64w2x2",
        needle: "`v_or_b32_e32 v4, 1, v3` (after label `.LBB48`): needs Store wait 0 (War { reg: \"v[4:5]\" })",
        why: LWMMA_TILE_ST_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lwmma_tile_pm_gfx1201.hxaco#lwmma_tile_bf16_t1_b64x64w4x1",
        needle: "`v_or_b32_e32 v4, 1, v3` (after label `.LBB72`): needs Store wait 0 (War { reg: \"v[4:5]\" })",
        why: LWMMA_TILE_ST_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lwmma_tile_pm_gfx1201.hxaco#lwmma_tile_bf16_t2_b16x32w1x1g4",
        needle: "`v_cmp_ne_u32_e64 s1, 0, v45` (after label `.LBB1`): needs Km wait 0 (Waw { reg: \"s[2:3]\" })",
        why: LWMMA_TILE_KM_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lwmma_tile_pm_gfx1201.hxaco#lwmma_tile_bf16_t2_b32x64w2x2",
        needle: "`v_or_b32_e32 v4, 1, v3` (after label `.LBB48`): needs Store wait 0 (War { reg: \"v[4:5]\" })",
        why: LWMMA_TILE_ST_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lwmma_tile_pm_gfx1201.hxaco#lwmma_tile_bf16_t2_b64x64w4x1",
        needle: "`v_or_b32_e32 v4, 1, v3` (after label `.LBB72`): needs Store wait 0 (War { reg: \"v[4:5]\" })",
        why: LWMMA_TILE_ST_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lwmma_tile_pm_gfx1201.hxaco#lwmma_tile_bf16_t2_b64x64w4x2",
        needle: "`v_or_b32_e32 v4, 1, v3` (after label `.LBB48`): needs Store wait 0 (War { reg: \"v[4:5]\" })",
        why: LWMMA_TILE_ST_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lwmma_tile_pm_gfx1201.hxaco#lwmma_tile_bf16_t2_b32x128w2x2",
        needle: "`v_or_b32_e32 v4, 1, v3` (after label `.LBB72`): needs Store wait 0 (War { reg: \"v[4:5]\" })",
        why: LWMMA_TILE_ST_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lwmma_tile_pm_gfx1201.hxaco#lwmma_tile_bf16_t2_b64x128w4x2",
        needle: "`v_or_b32_e32 v4, 1, v3` (after label `.LBB72`): needs Store wait 0 (War { reg: \"v[4:5]\" })",
        why: LWMMA_TILE_ST_WHY,
    },
    Allow {
        scope: "pins",
        target: "qwen4_lwmma_tile_pm_gfx1201.hxaco#lwmma_tile_bf16_t2_b64x128w2x4",
        needle: "`v_or_b32_e32 v4, 1, v3` (after label `.LBB63`): needs Store wait 0 (War { reg: \"v[4:5]\" })",
        why: LWMMA_TILE_ST_WHY,
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
        target: "region:ggml_mmvq_shim_pm_gfx1201.hxaco#ggml_mmvq_iq4_xs_gfx1201",
        needle: "not convertible: enables more than one workitem-id axis (packed v0)",
        why: "lift_region refuses the kernel (more than one workitem-id axis is enabled, packed v0), so no pool region exists; not a verifier finding",
    },
    Allow {
        scope: "pools",
        target: "region:ggml_mmvq_shim_pm_gfx1201.hxaco#ggml_mmvq_q4_k_gfx1201",
        needle: "not convertible: enables more than one workitem-id axis (packed v0)",
        why: "lift_region refuses the kernel (more than one workitem-id axis is enabled, packed v0), so no pool region exists; not a verifier finding",
    },
    Allow {
        scope: "pools",
        target: "region:ggml_mmvq_shim_pm_gfx1201.hxaco#ggml_mmvq_q6_k_gfx1201",
        needle: "not convertible: enables more than one workitem-id axis (packed v0)",
        why: "lift_region refuses the kernel (more than one workitem-id axis is enabled, packed v0), so no pool region exists; not a verifier finding",
    },
    Allow {
        scope: "pools",
        target: "region:ggml_mmvq_shim_pm_gfx1201.hxaco#ggml_mmvq_q8_0_gfx1201",
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
    Allow {
        scope: "pins",
        target: "moe_combine_ordered_hipcc_gfx1200.co#moe_down_combine_grouped_top10_bf16in_ordered",
        needle: "wait verification: more than 512 distinct in-flight memory states reach label `.LBB28` (path-sensitive state explosion)",
        why: "UNSUPPORTED INPUT: OPEN: verifier state cap, not proven; the verifier's path-sensitive wait analysis exceeds its 512 in-flight-memory-state cap at `.LBB28` (LLVM-compiled anchor), so the strict wait proof is NOT established for this image; the original LLVM bytes are retained byte-for-byte via Route-A; this is not a claimed race and not a successful full certification",
    },
    Allow {
        scope: "pins",
        target: "moe_combine_ordered_hipcc_gfx1200.co#moe_down_combine_grouped_top10_bf16in_ordered_zinit",
        needle: "wait verification: more than 512 distinct in-flight memory states reach label `.LBB28` (path-sensitive state explosion)",
        why: "UNSUPPORTED INPUT: OPEN: verifier state cap, not proven; the verifier's path-sensitive wait analysis exceeds its 512 in-flight-memory-state cap at `.LBB28` (LLVM-compiled anchor), so the strict wait proof is NOT established for this image; the original LLVM bytes are retained byte-for-byte via Route-A; this is not a claimed race and not a successful full certification",
    },
    Allow {
        scope: "pins",
        target: "moe_combine_ordered_hipcc_gfx1201.co#moe_down_combine_grouped_top10_bf16in_ordered",
        needle: "wait verification: more than 512 distinct in-flight memory states reach label `.LBB28` (path-sensitive state explosion)",
        why: "UNSUPPORTED INPUT: OPEN: verifier state cap, not proven; the verifier's path-sensitive wait analysis exceeds its 512 in-flight-memory-state cap at `.LBB28` (LLVM-compiled anchor), so the strict wait proof is NOT established for this image; the original LLVM bytes are retained byte-for-byte via Route-A; this is not a claimed race and not a successful full certification",
    },
    Allow {
        scope: "pins",
        target: "moe_combine_ordered_hipcc_gfx1201.co#moe_down_combine_grouped_top10_bf16in_ordered_zinit",
        needle: "wait verification: more than 512 distinct in-flight memory states reach label `.LBB28` (path-sensitive state explosion)",
        why: "UNSUPPORTED INPUT: OPEN: verifier state cap, not proven; the verifier's path-sensitive wait analysis exceeds its 512 in-flight-memory-state cap at `.LBB28` (LLVM-compiled anchor), so the strict wait proof is NOT established for this image; the original LLVM bytes are retained byte-for-byte via Route-A; this is not a claimed race and not a successful full certification",
    },
    Allow {
        scope: "pins",
        target: "moe_combine_ordered_pm_gfx1200.hxaco#moe_down_combine_grouped_top10_bf16in_ordered",
        needle: "wait verification: more than 512 distinct in-flight memory states reach label `.LBB28` (path-sensitive state explosion)",
        why: "UNSUPPORTED INPUT: OPEN: verifier state cap, not proven; the verifier's path-sensitive wait analysis exceeds its 512 in-flight-memory-state cap at `.LBB28` (Route-A twin), so the strict wait proof is NOT established for this image; the original LLVM bytes are retained byte-for-byte via Route-A; this is not a claimed race and not a successful full certification",
    },
    Allow {
        scope: "pins",
        target: "moe_combine_ordered_pm_gfx1200.hxaco#moe_down_combine_grouped_top10_bf16in_ordered_zinit",
        needle: "wait verification: more than 512 distinct in-flight memory states reach label `.LBB28` (path-sensitive state explosion)",
        why: "UNSUPPORTED INPUT: OPEN: verifier state cap, not proven; the verifier's path-sensitive wait analysis exceeds its 512 in-flight-memory-state cap at `.LBB28` (Route-A twin), so the strict wait proof is NOT established for this image; the original LLVM bytes are retained byte-for-byte via Route-A; this is not a claimed race and not a successful full certification",
    },
    Allow {
        scope: "pins",
        target: "moe_combine_ordered_pm_gfx1201.hxaco#moe_down_combine_grouped_top10_bf16in_ordered",
        needle: "wait verification: more than 512 distinct in-flight memory states reach label `.LBB28` (path-sensitive state explosion)",
        why: "UNSUPPORTED INPUT: OPEN: verifier state cap, not proven; the verifier's path-sensitive wait analysis exceeds its 512 in-flight-memory-state cap at `.LBB28` (Route-A twin), so the strict wait proof is NOT established for this image; the original LLVM bytes are retained byte-for-byte via Route-A; this is not a claimed race and not a successful full certification",
    },
    Allow {
        scope: "pins",
        target: "moe_combine_ordered_pm_gfx1201.hxaco#moe_down_combine_grouped_top10_bf16in_ordered_zinit",
        needle: "wait verification: more than 512 distinct in-flight memory states reach label `.LBB28` (path-sensitive state explosion)",
        why: "UNSUPPORTED INPUT: OPEN: verifier state cap, not proven; the verifier's path-sensitive wait analysis exceeds its 512 in-flight-memory-state cap at `.LBB28` (Route-A twin), so the strict wait proof is NOT established for this image; the original LLVM bytes are retained byte-for-byte via Route-A; this is not a claimed race and not a successful full certification",
    },
    Allow {
        scope: "pools",
        target: "region:moe_route_scatter_wide_hipcc_gfx1201.co#moe_route_scatter_top10_wide_f32",
        needle: "not convertible: barrier participation: the paths from the per-wave branch of block 60 (EXEC Full; Vccnz on a per-wave VCC) to its post-dominator (the exit) pass different numbers of barriers",
        why: "lift_region refuses the kernel (the last-arriving-workgroup scatter is behind a per-wave branch whose paths pass different numbers of barriers, so a multi-wave dispatcher region would mis-pair them), so no pool region exists; not a verifier finding",
    },
    Allow {
        scope: "pools",
        target: "region:moe_route_scatter_wide_pm_gfx1201.hxaco#moe_route_scatter_top10_wide_f32",
        needle: "not convertible: barrier participation: the paths from the per-wave branch of block 60 (EXEC Full; Vccnz on a per-wave VCC) to its post-dominator (the exit) pass different numbers of barriers",
        why: "lift_region refuses the kernel (the last-arriving-workgroup scatter is behind a per-wave branch whose paths pass different numbers of barriers, so a multi-wave dispatcher region would mis-pair them), so no pool region exists; not a verifier finding",
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

/// One finding of a hosted body inside a pool, exact: its module/symbol identity, the pc
/// within the body (the lifted region's line index), category, instruction and need.
/// Runtime template ids are deliberately not part of this key. There is no wildcard and
/// no dispatcher entry: a dispatcher instruction is outside every body span and is fatal.
pub struct PoolBody {
    /// The `pins` target of the hosted kernel (`file#symbol`).
    pub target: &'static str,
    pub body_pc: usize,
    pub category: Category,
    pub insn: &'static str,
    pub need: &'static str,
}

/// Every hosted body whose strict verification reports, one entry per module/symbol.
/// Each is the pool-hosted copy of an existing `pins` finding with the same target,
/// instruction, need and category, at the same body pc. The emitted label differs
/// (`.LBBn` versus `.Lt<tid>b_n`), so the scan resolves runtime ids to targets before
/// matching. Inserting an image never changes an existing row.
///
/// `legacy_pool_body.json` freezes all pre-migration positional rows. The migration
/// regression checks exact equivalence, including rejection of altered findings.
/// No entry is a proven race, and there is no dispatcher entry.
pub const POOL_BODY: &[PoolBody] = &[
    // pins row 2, W2
    PoolBody {
        target: "gated_delta_vector_pm_gfx1201.hxaco#gated_delta_net_vector_serial_f32_pm_gfx1201",
        body_pc: 367,
        category: Category::Candidate,
        insn: "`v_add_co_u32 v25, s3, v25, s14`",
        need: "needs Store wait 0 (War { reg: \"v[25:26]\" })",
    },
    // pins row 1, W1; artifact://97452 pm_pool_28 pc 673
    PoolBody {
        target:
            "gated_delta_vector_pm_gfx1201.hxaco#causal_conv_depthwise_silu_rows_f32_pm_gfx1201",
        body_pc: 545,
        category: Category::Candidate,
        insn: "`v_add_co_u32 v22, vcc_lo, v22, s38`",
        need: "needs Store wait 0 (War { reg: \"v[22:23]\" })",
    },
    // F4 hosted copy of the gemm_mq6sym_i8 hipcc_gfx1201.co pins finding
    PoolBody {
        target: "gemm_mq6sym_i8_hipcc_gfx1201.co#gemm_mq6sym_i8_set_m32n16_gfx1201",
        body_pc: 357,
        category: Category::Candidate,
        insn: "`v_and_b32_e32 v8, 0x3f00, v86`",
        need: "needs Ds wait 1 (War { reg: \"v[8:9]\" })",
    },
    // F4 hosted copy of the gemm_mq6sym_i8 hipcc_gfx1201.co pins finding
    PoolBody {
        target: "gemm_mq6sym_i8_hipcc_gfx1201.co#gemm_mq6sym_i8_set_bf16_m32n16_gfx1201",
        body_pc: 358,
        category: Category::Candidate,
        insn: "`v_and_b32_e32 v8, 0x3f00, v86`",
        need: "needs Ds wait 1 (War { reg: \"v[8:9]\" })",
    },
    // F4 hosted copy of the gemm_mq6sym_i8 hipcc_gfx1201.co pins finding
    PoolBody {
        target: "gemm_mq6sym_i8_hipcc_gfx1201.co#gemm_mq6sym_i8_set_m32n32_gfx1201",
        body_pc: 403,
        category: Category::Candidate,
        insn: "`v_lshrrev_b32_e32 v26, 4, v44`",
        need: "needs Ds wait 1 (War { reg: \"v[26:27]\" })",
    },
    // F4 hosted copy of the gemm_mq6sym_i8 hipcc_gfx1201.co pins finding
    PoolBody {
        target: "gemm_mq6sym_i8_hipcc_gfx1201.co#gemm_mq6sym_i8_set_bf16_m32n32_gfx1201",
        body_pc: 402,
        category: Category::Candidate,
        insn: "`v_lshrrev_b32_e32 v26, 4, v44`",
        need: "needs Ds wait 1 (War { reg: \"v[26:27]\" })",
    },
    // F4 hosted copy of the gemm_mq6sym_i8 hipcc_gfx1201.co pins finding
    PoolBody {
        target: "gemm_mq6sym_i8_hipcc_gfx1201.co#gemm_mq6sym_i8_set_m64n16_gfx1201",
        body_pc: 346,
        category: Category::Candidate,
        insn: "`v_and_or_b32 v10, v28, 63, v15`",
        need: "needs Ds wait 1 (War { reg: \"v[10:11]\" })",
    },
    // F4 hosted copy of the gemm_mq6sym_i8 hipcc_gfx1201.co pins finding
    PoolBody {
        target: "gemm_mq6sym_i8_hipcc_gfx1201.co#gemm_mq6sym_i8_set_bf16_m64n16_gfx1201",
        body_pc: 346,
        category: Category::Candidate,
        insn: "`v_and_or_b32 v10, v28, 63, v15`",
        need: "needs Ds wait 1 (War { reg: \"v[10:11]\" })",
    },
    // F4 hosted copy of the gemm_mq6sym_i8 hipcc_gfx1201.co pins finding
    PoolBody {
        target: "gemm_mq6sym_i8_hipcc_gfx1201.co#gemm_mq6sym_i8_set_m64n32_gfx1201",
        body_pc: 381,
        category: Category::Candidate,
        insn: "`v_lshlrev_b32_e32 v20, 12, v30`",
        need: "needs Ds wait 0 (War { reg: \"v[20:21]\" })",
    },
    // F4 hosted copy of the gemm_mq6sym_i8 hipcc_gfx1201.co pins finding
    PoolBody {
        target: "gemm_mq6sym_i8_hipcc_gfx1201.co#gemm_mq6sym_i8_set_bf16_m64n32_gfx1201",
        body_pc: 383,
        category: Category::Candidate,
        insn: "`v_lshlrev_b32_e32 v20, 12, v30`",
        need: "needs Ds wait 0 (War { reg: \"v[20:21]\" })",
    },
    // F4 hosted copy of the gemm_mq6sym_i8 hipcc_gfx1201.co pins finding
    PoolBody {
        target: "gemm_mq6sym_i8_hipcc_gfx1201.co#gemm_mq6sym_i8_set_m64n48_gfx1201",
        body_pc: 433,
        category: Category::Candidate,
        insn: "`v_alignbit_b32 v30, v61, v60, 24`",
        need: "needs Ds wait 1 (War { reg: \"v[30:31]\" })",
    },
    // F4 hosted copy of the gemm_mq6sym_i8 hipcc_gfx1201.co pins finding
    PoolBody {
        target: "gemm_mq6sym_i8_hipcc_gfx1201.co#gemm_mq6sym_i8_set_bf16_m64n48_gfx1201",
        body_pc: 435,
        category: Category::Candidate,
        insn: "`v_alignbit_b32 v30, v61, v60, 24`",
        need: "needs Ds wait 1 (War { reg: \"v[30:31]\" })",
    },
    // F4 hosted copy of the gemm_mq6sym_i8 hipcc_gfx1201.co pins finding
    PoolBody {
        target: "gemm_mq6sym_i8_hipcc_gfx1201.co#gemm_mq6sym_i8_set_m64n16y_gfx1201",
        body_pc: 358,
        category: Category::Candidate,
        insn: "`v_and_or_b32 v10, v28, 63, v15`",
        need: "needs Ds wait 1 (War { reg: \"v[10:11]\" })",
    },
    // F4 hosted copy of the gemm_mq6sym_i8 hipcc_gfx1201.co pins finding
    PoolBody {
        target: "gemm_mq6sym_i8_hipcc_gfx1201.co#gemm_mq6sym_i8_set_bf16_m64n16y_gfx1201",
        body_pc: 358,
        category: Category::Candidate,
        insn: "`v_and_or_b32 v10, v28, 63, v15`",
        need: "needs Ds wait 1 (War { reg: \"v[10:11]\" })",
    },
    // F4 hosted copy of the gemm_mq6sym_i8 hipcc_gfx1201.co pins finding
    PoolBody {
        target: "gemm_mq6sym_i8_hipcc_gfx1201.co#gemm_mq6sym_i8_set_m64n16_x4_gfx1201",
        body_pc: 402,
        category: Category::Candidate,
        insn: "`v_and_or_b32 v10, v28, 63, v15`",
        need: "needs Ds wait 1 (War { reg: \"v[10:11]\" })",
    },
    // F4 hosted copy of the gemm_mq6sym_i8 hipcc_gfx1201.co pins finding
    PoolBody {
        target: "gemm_mq6sym_i8_hipcc_gfx1201.co#gemm_mq6sym_i8_set_m32n32_x4_gfx1201",
        body_pc: 459,
        category: Category::Candidate,
        insn: "`v_lshrrev_b32_e32 v26, 4, v44`",
        need: "needs Ds wait 1 (War { reg: \"v[26:27]\" })",
    },
    // F4 hosted copy of the gemm_mq6sym_i8 hipcc_gfx1201.co pins finding
    PoolBody {
        target: "gemm_mq6sym_i8_hipcc_gfx1201.co#gemm_mq6sym_i8_set_m64n48_x4_gfx1201",
        body_pc: 480,
        category: Category::Candidate,
        insn: "`v_alignbit_b32 v30, v61, v60, 24`",
        need: "needs Ds wait 1 (War { reg: \"v[30:31]\" })",
    },
    // F4 hosted copy of the gemm_mq6sym_i8 hipcc_gfx1201.co pins finding
    PoolBody {
        target: "gemm_mq6sym_i8_hipcc_gfx1201.co#gemm_mq6sym_i8_set_m64n16y_x4_gfx1201",
        body_pc: 416,
        category: Category::Candidate,
        insn: "`v_and_or_b32 v10, v28, 63, v15`",
        need: "needs Ds wait 1 (War { reg: \"v[10:11]\" })",
    },
    // F4 hosted copy of the gemm_mq6sym_i8 hipcc_gfx1201.co pins finding
    PoolBody {
        target: "gemm_mq6sym_i8_hipcc_gfx1201.co#gemm_mq6sym_i8_set_gfx1201",
        body_pc: 163,
        category: Category::Candidate,
        insn: "`v_lshrrev_b32_e32 v3, 2, v0`",
        need: "needs Ds wait 0 (War { reg: \"v3\" })",
    },
    // F4 hosted copy of the gemm_mq6sym_i8 hipcc_gfx1201.co pins finding
    PoolBody {
        target: "gemm_mq6sym_i8_hipcc_gfx1201.co#gemm_mq6sym_i8_set_bf16_gfx1201",
        body_pc: 162,
        category: Category::Candidate,
        insn: "`v_lshrrev_b32_e32 v3, 2, v0`",
        need: "needs Ds wait 0 (War { reg: \"v3\" })",
    },
    // F4 hosted copy of the gemm_mq6sym_i8 pm_gfx1201.hxaco pins finding
    PoolBody {
        target: "gemm_mq6sym_i8_pm_gfx1201.hxaco#gemm_mq6sym_i8_set_m32n16_gfx1201",
        body_pc: 357,
        category: Category::Candidate,
        insn: "`v_and_b32_e32 v8, 0x3f00, v86`",
        need: "needs Ds wait 1 (War { reg: \"v[8:9]\" })",
    },
    // F4 hosted copy of the gemm_mq6sym_i8 pm_gfx1201.hxaco pins finding
    PoolBody {
        target: "gemm_mq6sym_i8_pm_gfx1201.hxaco#gemm_mq6sym_i8_set_bf16_m32n16_gfx1201",
        body_pc: 358,
        category: Category::Candidate,
        insn: "`v_and_b32_e32 v8, 0x3f00, v86`",
        need: "needs Ds wait 1 (War { reg: \"v[8:9]\" })",
    },
    // F4 hosted copy of the gemm_mq6sym_i8 pm_gfx1201.hxaco pins finding
    PoolBody {
        target: "gemm_mq6sym_i8_pm_gfx1201.hxaco#gemm_mq6sym_i8_set_m32n32_gfx1201",
        body_pc: 403,
        category: Category::Candidate,
        insn: "`v_lshrrev_b32_e32 v26, 4, v44`",
        need: "needs Ds wait 1 (War { reg: \"v[26:27]\" })",
    },
    // F4 hosted copy of the gemm_mq6sym_i8 pm_gfx1201.hxaco pins finding
    PoolBody {
        target: "gemm_mq6sym_i8_pm_gfx1201.hxaco#gemm_mq6sym_i8_set_bf16_m32n32_gfx1201",
        body_pc: 402,
        category: Category::Candidate,
        insn: "`v_lshrrev_b32_e32 v26, 4, v44`",
        need: "needs Ds wait 1 (War { reg: \"v[26:27]\" })",
    },
    // F4 hosted copy of the gemm_mq6sym_i8 pm_gfx1201.hxaco pins finding
    PoolBody {
        target: "gemm_mq6sym_i8_pm_gfx1201.hxaco#gemm_mq6sym_i8_set_m64n16_gfx1201",
        body_pc: 346,
        category: Category::Candidate,
        insn: "`v_and_or_b32 v10, v28, 63, v15`",
        need: "needs Ds wait 1 (War { reg: \"v[10:11]\" })",
    },
    // F4 hosted copy of the gemm_mq6sym_i8 pm_gfx1201.hxaco pins finding
    PoolBody {
        target: "gemm_mq6sym_i8_pm_gfx1201.hxaco#gemm_mq6sym_i8_set_bf16_m64n16_gfx1201",
        body_pc: 346,
        category: Category::Candidate,
        insn: "`v_and_or_b32 v10, v28, 63, v15`",
        need: "needs Ds wait 1 (War { reg: \"v[10:11]\" })",
    },
    // F4 hosted copy of the gemm_mq6sym_i8 pm_gfx1201.hxaco pins finding
    PoolBody {
        target: "gemm_mq6sym_i8_pm_gfx1201.hxaco#gemm_mq6sym_i8_set_m64n32_gfx1201",
        body_pc: 381,
        category: Category::Candidate,
        insn: "`v_lshlrev_b32_e32 v20, 12, v30`",
        need: "needs Ds wait 0 (War { reg: \"v[20:21]\" })",
    },
    // F4 hosted copy of the gemm_mq6sym_i8 pm_gfx1201.hxaco pins finding
    PoolBody {
        target: "gemm_mq6sym_i8_pm_gfx1201.hxaco#gemm_mq6sym_i8_set_bf16_m64n32_gfx1201",
        body_pc: 383,
        category: Category::Candidate,
        insn: "`v_lshlrev_b32_e32 v20, 12, v30`",
        need: "needs Ds wait 0 (War { reg: \"v[20:21]\" })",
    },
    // F4 hosted copy of the gemm_mq6sym_i8 pm_gfx1201.hxaco pins finding
    PoolBody {
        target: "gemm_mq6sym_i8_pm_gfx1201.hxaco#gemm_mq6sym_i8_set_m64n48_gfx1201",
        body_pc: 433,
        category: Category::Candidate,
        insn: "`v_alignbit_b32 v30, v61, v60, 24`",
        need: "needs Ds wait 1 (War { reg: \"v[30:31]\" })",
    },
    // F4 hosted copy of the gemm_mq6sym_i8 pm_gfx1201.hxaco pins finding
    PoolBody {
        target: "gemm_mq6sym_i8_pm_gfx1201.hxaco#gemm_mq6sym_i8_set_bf16_m64n48_gfx1201",
        body_pc: 435,
        category: Category::Candidate,
        insn: "`v_alignbit_b32 v30, v61, v60, 24`",
        need: "needs Ds wait 1 (War { reg: \"v[30:31]\" })",
    },
    // F4 hosted copy of the gemm_mq6sym_i8 pm_gfx1201.hxaco pins finding
    PoolBody {
        target: "gemm_mq6sym_i8_pm_gfx1201.hxaco#gemm_mq6sym_i8_set_m64n16y_gfx1201",
        body_pc: 358,
        category: Category::Candidate,
        insn: "`v_and_or_b32 v10, v28, 63, v15`",
        need: "needs Ds wait 1 (War { reg: \"v[10:11]\" })",
    },
    // F4 hosted copy of the gemm_mq6sym_i8 pm_gfx1201.hxaco pins finding
    PoolBody {
        target: "gemm_mq6sym_i8_pm_gfx1201.hxaco#gemm_mq6sym_i8_set_bf16_m64n16y_gfx1201",
        body_pc: 358,
        category: Category::Candidate,
        insn: "`v_and_or_b32 v10, v28, 63, v15`",
        need: "needs Ds wait 1 (War { reg: \"v[10:11]\" })",
    },
    // F4 hosted copy of the gemm_mq6sym_i8 pm_gfx1201.hxaco pins finding
    PoolBody {
        target: "gemm_mq6sym_i8_pm_gfx1201.hxaco#gemm_mq6sym_i8_set_m64n16_x4_gfx1201",
        body_pc: 402,
        category: Category::Candidate,
        insn: "`v_and_or_b32 v10, v28, 63, v15`",
        need: "needs Ds wait 1 (War { reg: \"v[10:11]\" })",
    },
    // F4 hosted copy of the gemm_mq6sym_i8 pm_gfx1201.hxaco pins finding
    PoolBody {
        target: "gemm_mq6sym_i8_pm_gfx1201.hxaco#gemm_mq6sym_i8_set_m32n32_x4_gfx1201",
        body_pc: 459,
        category: Category::Candidate,
        insn: "`v_lshrrev_b32_e32 v26, 4, v44`",
        need: "needs Ds wait 1 (War { reg: \"v[26:27]\" })",
    },
    // F4 hosted copy of the gemm_mq6sym_i8 pm_gfx1201.hxaco pins finding
    PoolBody {
        target: "gemm_mq6sym_i8_pm_gfx1201.hxaco#gemm_mq6sym_i8_set_m64n48_x4_gfx1201",
        body_pc: 480,
        category: Category::Candidate,
        insn: "`v_alignbit_b32 v30, v61, v60, 24`",
        need: "needs Ds wait 1 (War { reg: \"v[30:31]\" })",
    },
    // F4 hosted copy of the gemm_mq6sym_i8 pm_gfx1201.hxaco pins finding
    PoolBody {
        target: "gemm_mq6sym_i8_pm_gfx1201.hxaco#gemm_mq6sym_i8_set_m64n16y_x4_gfx1201",
        body_pc: 416,
        category: Category::Candidate,
        insn: "`v_and_or_b32 v10, v28, 63, v15`",
        need: "needs Ds wait 1 (War { reg: \"v[10:11]\" })",
    },
    // F4 hosted copy of the gemm_mq6sym_i8 pm_gfx1201.hxaco pins finding
    PoolBody {
        target: "gemm_mq6sym_i8_pm_gfx1201.hxaco#gemm_mq6sym_i8_set_gfx1201",
        body_pc: 163,
        category: Category::Candidate,
        insn: "`v_lshrrev_b32_e32 v3, 2, v0`",
        need: "needs Ds wait 0 (War { reg: \"v3\" })",
    },
    // F4 hosted copy of the gemm_mq6sym_i8 pm_gfx1201.hxaco pins finding
    PoolBody {
        target: "gemm_mq6sym_i8_pm_gfx1201.hxaco#gemm_mq6sym_i8_set_bf16_gfx1201",
        body_pc: 162,
        category: Category::Candidate,
        insn: "`v_lshrrev_b32_e32 v3, 2, v0`",
        need: "needs Ds wait 0 (War { reg: \"v3\" })",
    },
    // pins row 4
    PoolBody {
        target: "ggml_convert_shim_gfx1201.co#ggml_dequant_q4_k_f16_gfx1201",
        body_pc: 143,
        category: Category::ModelLimit,
        insn: "`global_load_d16_hi_u8 v6, v[3:4], off offset:4`",
        need: "needs Load wait 0 (Use { reg: \"v6\" })",
    },
    // pins row 6; artifact://97452 pm_pool_34 pc 271
    PoolBody {
        target: "ggml_convert_shim_gfx1201.co#ggml_dequant_q5_k_f16_gfx1201",
        body_pc: 143,
        category: Category::ModelLimit,
        insn: "`global_load_d16_hi_u8 v7, v[3:4], off offset:4`",
        need: "needs Load wait 0 (Use { reg: \"v7\" })",
    },
    // pins row 8, W4; artifact://97452 pm_pool_35 pc 323
    PoolBody {
        target: "ggml_convert_shim_gfx1201.co#ggml_dequant_q6_k_f16_gfx1201",
        body_pc: 195,
        category: Category::Candidate,
        insn: "`global_load_d16_hi_u8 v0, v[7:8], off offset:32`",
        need: "needs Store wait 0 (War { reg: \"v0\" })",
    },
    // pins row 3, W3
    PoolBody {
        target: "ggml_convert_shim_gfx1201.co#ggml_dequant_iq4_xs_f32_gfx1201",
        body_pc: 252,
        category: Category::Candidate,
        insn: "`v_and_b32_e32 v8, 0x7fffffff, v7`",
        need: "needs Store wait 1 (War { reg: \"v8\" })",
    },
    // pins row 5
    PoolBody {
        target: "ggml_convert_shim_gfx1201.co#ggml_dequant_q4_k_f32_gfx1201",
        body_pc: 152,
        category: Category::ModelLimit,
        insn: "`global_load_d16_hi_u8 v5, v[1:2], off offset:4`",
        need: "needs Load wait 0 (Use { reg: \"v5\" })",
    },
    // pins row 7; artifact://97452 pm_pool_38 pc 280
    PoolBody {
        target: "ggml_convert_shim_gfx1201.co#ggml_dequant_q5_k_f32_gfx1201",
        body_pc: 152,
        category: Category::ModelLimit,
        insn: "`global_load_d16_hi_u8 v5, v[3:4], off offset:4`",
        need: "needs Load wait 0 (Use { reg: \"v5\" })",
    },
    // pins row 9; artifact://97452 pm_pool_39 pc 289
    PoolBody {
        target: "ggml_convert_shim_gfx1201.co#ggml_dequant_q6_k_f32_gfx1201",
        body_pc: 161,
        category: Category::ModelLimit,
        insn: "`global_load_d16_hi_u8 v1, v[2:3], off`",
        need: "needs Load wait 0 (Use { reg: \"v1\" })",
    },
    // pins row 10, W5; artifact://97452 pm_pool_40 pc 295
    PoolBody {
        target: "ggml_convert_shim_gfx1201.co#ggml_dequant_q8_0_f32_gfx1201",
        body_pc: 167,
        category: Category::Candidate,
        insn: "`v_bfe_u32 v2, v5, 16, 1`",
        need: "needs Store wait 0 (War { reg: \"v2\" })",
    },
    // pins row 14
    PoolBody {
        target: "ggml_convert_shim_gfx1201.co#ggml_get_row_q4_k_f32_gfx1201",
        body_pc: 45,
        category: Category::ModelLimit,
        insn: "`global_load_d16_hi_u8 v4, v[1:2], off offset:4`",
        need: "needs Load wait 0 (Use { reg: \"v4\" })",
    },
    // pins row 11
    PoolBody {
        target: "ggml_convert_shim_gfx1201.co#ggml_get_row1_q4_k_f32_gfx1201",
        body_pc: 38,
        category: Category::ModelLimit,
        insn: "`global_load_d16_hi_u8 v4, v[1:2], off offset:4`",
        need: "needs Load wait 0 (Use { reg: \"v4\" })",
    },
    // pins row 15; artifact://97452 pm_pool_45 pc 173
    PoolBody {
        target: "ggml_convert_shim_gfx1201.co#ggml_get_row_q5_k_f32_gfx1201",
        body_pc: 45,
        category: Category::ModelLimit,
        insn: "`global_load_d16_hi_u8 v5, v[1:2], off offset:4`",
        need: "needs Load wait 0 (Use { reg: \"v5\" })",
    },
    // pins row 12; artifact://97452 pm_pool_46 pc 166
    PoolBody {
        target: "ggml_convert_shim_gfx1201.co#ggml_get_row1_q5_k_f32_gfx1201",
        body_pc: 38,
        category: Category::ModelLimit,
        insn: "`global_load_d16_hi_u8 v5, v[1:2], off offset:4`",
        need: "needs Load wait 0 (Use { reg: \"v5\" })",
    },
    // pins row 16; artifact://97452 pm_pool_47 pc 194
    PoolBody {
        target: "ggml_convert_shim_gfx1201.co#ggml_get_row_q6_k_f32_gfx1201",
        body_pc: 66,
        category: Category::ModelLimit,
        insn: "`global_load_d16_hi_u8 v0, v[4:5], off`",
        need: "needs Load wait 1 (Use { reg: \"v0\" })",
    },
    // pins row 13; artifact://97452 pm_pool_48 pc 146
    PoolBody {
        target: "ggml_convert_shim_gfx1201.co#ggml_get_row1_q6_k_f32_gfx1201",
        body_pc: 18,
        category: Category::ModelLimit,
        insn: "`v_sub_co_u32 v2, s3, v0, v1`",
        need: "needs Km wait 0 (Waw { reg: \"s[4:6]\" })",
    },
    // moe_shared_down_combine hipcc originals (tid 265-280)
    PoolBody {
        target: "moe_shared_down_combine_hipcc_gfx1201.co#moe_shared_down_combine_rows_r1",
        body_pc: 259,
        category: Category::Candidate,
        insn: "`v_bfe_u32 v20, v18, 16, 1`",
        need: "needs Ds wait 0 (War { reg: \"v20\" })",
    },
    PoolBody {
        target: "moe_shared_down_combine_hipcc_gfx1201.co#moe_shared_down_combine_rows_r1_zinit",
        body_pc: 254,
        category: Category::Candidate,
        insn: "`v_bfe_u32 v19, v17, 16, 1`",
        need: "needs Ds wait 0 (War { reg: \"v19\" })",
    },
    PoolBody {
        target: "moe_shared_down_combine_hipcc_gfx1201.co#moe_shared_down_combine_rows_r2",
        body_pc: 1304,
        category: Category::Candidate,
        insn: "`v_dual_add_f32 v5, v10, v12 :: v_dual_add_f32 v6, v8, v9`",
        need: "needs Ds wait 0 (War { reg: \"v5\" })",
    },
    PoolBody {
        target: "moe_shared_down_combine_hipcc_gfx1201.co#moe_shared_down_combine_rows_r2_zinit",
        body_pc: 1275,
        category: Category::Candidate,
        insn: "`v_dual_add_f32 v5, v10, v12 :: v_dual_add_f32 v6, v8, v9`",
        need: "needs Ds wait 0 (War { reg: \"v5\" })",
    },
    PoolBody {
        target: "moe_shared_down_combine_hipcc_gfx1201.co#moe_shared_down_combine_rows_r3",
        body_pc: 1402,
        category: Category::Candidate,
        insn: "`v_add_f32_e32 v5, v29, v32`",
        need: "needs Ds wait 0 (War { reg: \"v5\" })",
    },
    PoolBody {
        target: "moe_shared_down_combine_hipcc_gfx1201.co#moe_shared_down_combine_rows_r3_zinit",
        body_pc: 1373,
        category: Category::Candidate,
        insn: "`v_add_f32_e32 v5, v29, v32`",
        need: "needs Ds wait 0 (War { reg: \"v5\" })",
    },
    PoolBody {
        target: "moe_shared_down_combine_hipcc_gfx1201.co#moe_shared_down_combine_rows_r4",
        body_pc: 1509,
        category: Category::Candidate,
        insn: "`v_add_f32_e32 v5, v45, v46`",
        need: "needs Ds wait 0 (War { reg: \"v5\" })",
    },
    PoolBody {
        target: "moe_shared_down_combine_hipcc_gfx1201.co#moe_shared_down_combine_rows_r4_zinit",
        body_pc: 1480,
        category: Category::Candidate,
        insn: "`v_add_f32_e32 v5, v45, v46`",
        need: "needs Ds wait 0 (War { reg: \"v5\" })",
    },
    PoolBody {
        target: "moe_shared_down_combine_hipcc_gfx1201.co#moe_shared_down_combine_rows_r5",
        body_pc: 1618,
        category: Category::Candidate,
        insn: "`v_add_f32_e32 v5, v43, v45`",
        need: "needs Ds wait 0 (War { reg: \"v5\" })",
    },
    PoolBody {
        target: "moe_shared_down_combine_hipcc_gfx1201.co#moe_shared_down_combine_rows_r5_zinit",
        body_pc: 1589,
        category: Category::Candidate,
        insn: "`v_add_f32_e32 v5, v43, v45`",
        need: "needs Ds wait 0 (War { reg: \"v5\" })",
    },
    PoolBody {
        target: "moe_shared_down_combine_hipcc_gfx1201.co#moe_shared_down_combine_rows_r6",
        body_pc: 1727,
        category: Category::Candidate,
        insn: "`v_add_f32_e32 v5, v57, v59`",
        need: "needs Ds wait 0 (War { reg: \"v5\" })",
    },
    PoolBody {
        target: "moe_shared_down_combine_hipcc_gfx1201.co#moe_shared_down_combine_rows_r6_zinit",
        body_pc: 1698,
        category: Category::Candidate,
        insn: "`v_add_f32_e32 v5, v57, v59`",
        need: "needs Ds wait 0 (War { reg: \"v5\" })",
    },
    PoolBody {
        target: "moe_shared_down_combine_hipcc_gfx1201.co#moe_shared_down_combine_rows_r7",
        body_pc: 1824,
        category: Category::Candidate,
        insn: "`v_add_f32_e32 v5, v56, v58`",
        need: "needs Ds wait 0 (War { reg: \"v5\" })",
    },
    PoolBody {
        target: "moe_shared_down_combine_hipcc_gfx1201.co#moe_shared_down_combine_rows_r7_zinit",
        body_pc: 1795,
        category: Category::Candidate,
        insn: "`v_add_f32_e32 v5, v56, v58`",
        need: "needs Ds wait 0 (War { reg: \"v5\" })",
    },
    PoolBody {
        target: "moe_shared_down_combine_hipcc_gfx1201.co#moe_shared_down_combine_rows_r8",
        body_pc: 1928,
        category: Category::Candidate,
        insn: "`v_add_f32_e32 v5, v49, v52`",
        need: "needs Ds wait 0 (War { reg: \"v5\" })",
    },
    PoolBody {
        target: "moe_shared_down_combine_hipcc_gfx1201.co#moe_shared_down_combine_rows_r8_zinit",
        body_pc: 1899,
        category: Category::Candidate,
        insn: "`v_add_f32_e32 v5, v49, v52`",
        need: "needs Ds wait 0 (War { reg: \"v5\" })",
    },
    // moe_shared_down_combine PM twins (tid 281-296)
    PoolBody {
        target: "moe_shared_down_combine_pm_gfx1201.hxaco#moe_shared_down_combine_rows_r1",
        body_pc: 259,
        category: Category::Candidate,
        insn: "`v_bfe_u32 v20, v18, 16, 1`",
        need: "needs Ds wait 0 (War { reg: \"v20\" })",
    },
    PoolBody {
        target: "moe_shared_down_combine_pm_gfx1201.hxaco#moe_shared_down_combine_rows_r1_zinit",
        body_pc: 254,
        category: Category::Candidate,
        insn: "`v_bfe_u32 v19, v17, 16, 1`",
        need: "needs Ds wait 0 (War { reg: \"v19\" })",
    },
    PoolBody {
        target: "moe_shared_down_combine_pm_gfx1201.hxaco#moe_shared_down_combine_rows_r2",
        body_pc: 1304,
        category: Category::Candidate,
        insn: "`v_dual_add_f32 v5, v10, v12 :: v_dual_add_f32 v6, v8, v9`",
        need: "needs Ds wait 0 (War { reg: \"v5\" })",
    },
    PoolBody {
        target: "moe_shared_down_combine_pm_gfx1201.hxaco#moe_shared_down_combine_rows_r2_zinit",
        body_pc: 1275,
        category: Category::Candidate,
        insn: "`v_dual_add_f32 v5, v10, v12 :: v_dual_add_f32 v6, v8, v9`",
        need: "needs Ds wait 0 (War { reg: \"v5\" })",
    },
    PoolBody {
        target: "moe_shared_down_combine_pm_gfx1201.hxaco#moe_shared_down_combine_rows_r3",
        body_pc: 1402,
        category: Category::Candidate,
        insn: "`v_add_f32_e32 v5, v29, v32`",
        need: "needs Ds wait 0 (War { reg: \"v5\" })",
    },
    PoolBody {
        target: "moe_shared_down_combine_pm_gfx1201.hxaco#moe_shared_down_combine_rows_r3_zinit",
        body_pc: 1373,
        category: Category::Candidate,
        insn: "`v_add_f32_e32 v5, v29, v32`",
        need: "needs Ds wait 0 (War { reg: \"v5\" })",
    },
    PoolBody {
        target: "moe_shared_down_combine_pm_gfx1201.hxaco#moe_shared_down_combine_rows_r4",
        body_pc: 1509,
        category: Category::Candidate,
        insn: "`v_add_f32_e32 v5, v45, v46`",
        need: "needs Ds wait 0 (War { reg: \"v5\" })",
    },
    PoolBody {
        target: "moe_shared_down_combine_pm_gfx1201.hxaco#moe_shared_down_combine_rows_r4_zinit",
        body_pc: 1480,
        category: Category::Candidate,
        insn: "`v_add_f32_e32 v5, v45, v46`",
        need: "needs Ds wait 0 (War { reg: \"v5\" })",
    },
    PoolBody {
        target: "moe_shared_down_combine_pm_gfx1201.hxaco#moe_shared_down_combine_rows_r5",
        body_pc: 1618,
        category: Category::Candidate,
        insn: "`v_add_f32_e32 v5, v43, v45`",
        need: "needs Ds wait 0 (War { reg: \"v5\" })",
    },
    PoolBody {
        target: "moe_shared_down_combine_pm_gfx1201.hxaco#moe_shared_down_combine_rows_r5_zinit",
        body_pc: 1589,
        category: Category::Candidate,
        insn: "`v_add_f32_e32 v5, v43, v45`",
        need: "needs Ds wait 0 (War { reg: \"v5\" })",
    },
    PoolBody {
        target: "moe_shared_down_combine_pm_gfx1201.hxaco#moe_shared_down_combine_rows_r6",
        body_pc: 1727,
        category: Category::Candidate,
        insn: "`v_add_f32_e32 v5, v57, v59`",
        need: "needs Ds wait 0 (War { reg: \"v5\" })",
    },
    PoolBody {
        target: "moe_shared_down_combine_pm_gfx1201.hxaco#moe_shared_down_combine_rows_r6_zinit",
        body_pc: 1698,
        category: Category::Candidate,
        insn: "`v_add_f32_e32 v5, v57, v59`",
        need: "needs Ds wait 0 (War { reg: \"v5\" })",
    },
    PoolBody {
        target: "moe_shared_down_combine_pm_gfx1201.hxaco#moe_shared_down_combine_rows_r7",
        body_pc: 1824,
        category: Category::Candidate,
        insn: "`v_add_f32_e32 v5, v56, v58`",
        need: "needs Ds wait 0 (War { reg: \"v5\" })",
    },
    PoolBody {
        target: "moe_shared_down_combine_pm_gfx1201.hxaco#moe_shared_down_combine_rows_r7_zinit",
        body_pc: 1795,
        category: Category::Candidate,
        insn: "`v_add_f32_e32 v5, v56, v58`",
        need: "needs Ds wait 0 (War { reg: \"v5\" })",
    },
    PoolBody {
        target: "moe_shared_down_combine_pm_gfx1201.hxaco#moe_shared_down_combine_rows_r8",
        body_pc: 1928,
        category: Category::Candidate,
        insn: "`v_add_f32_e32 v5, v49, v52`",
        need: "needs Ds wait 0 (War { reg: \"v5\" })",
    },
    PoolBody {
        target: "moe_shared_down_combine_pm_gfx1201.hxaco#moe_shared_down_combine_rows_r8_zinit",
        body_pc: 1899,
        category: Category::Candidate,
        insn: "`v_add_f32_e32 v5, v49, v52`",
        need: "needs Ds wait 0 (War { reg: \"v5\" })",
    },
    // topk8_rows_mq6g256v2 hipcc originals (tid 557-558) and PM twins (tid 561-562); b256 pools
    PoolBody {
        target: "topk8_rows_mq6g256v2_hipcc_gfx1201.co#topk8_partial_f32_rows",
        body_pc: 274,
        category: Category::Candidate,
        insn: "`v_cndmask_b32_e64 v1, 0x7fffffff, v6, s2`",
        need: "needs Store wait 1 (War { reg: \"v1\" })",
    },
    PoolBody {
        target: "topk8_rows_mq6g256v2_hipcc_gfx1201.co#topk8_rescore_mq6g256v2_k2560_rows",
        body_pc: 75,
        category: Category::ModelLimit,
        insn: "`v_cmp_eq_u32_e64 s3, 0, v2`",
        need: "needs Km wait 0 (Waw { reg: \"s4\" })",
    },
    PoolBody {
        target: "topk8_rows_mq6g256v2_pm_gfx1201.hxaco#topk8_partial_f32_rows",
        body_pc: 274,
        category: Category::Candidate,
        insn: "`v_cndmask_b32_e64 v1, 0x7fffffff, v6, s2`",
        need: "needs Store wait 1 (War { reg: \"v1\" })",
    },
    PoolBody {
        target: "topk8_rows_mq6g256v2_pm_gfx1201.hxaco#topk8_rescore_mq6g256v2_k2560_rows",
        body_pc: 75,
        category: Category::ModelLimit,
        insn: "`v_cmp_eq_u32_e64 s3, 0, v2`",
        need: "needs Km wait 0 (Waw { reg: \"s4\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_gemm_bf16_t1_m1_n1",
        body_pc: 440,
        category: Category::Candidate,
        insn: "`v_lshlrev_b32_e32 v5, 2, v15`",
        need: "needs Store wait 0 (War { reg: \"v[5:6]\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_gemm_bf16_t1_m1_n2",
        body_pc: 530,
        category: Category::Candidate,
        insn: "`v_lshrrev_b32_e32 v3, 4, v11`",
        need: "needs Store wait 0 (War { reg: \"v[3:4]\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_gemm_bf16_t1_m2_n1",
        body_pc: 677,
        category: Category::Candidate,
        insn: "`v_bfe_u32 v5, v12, 5, 3`",
        need: "needs Store wait 0 (War { reg: \"v[5:6]\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_gemm_bf16_t1_m2_n2",
        body_pc: 667,
        category: Category::Candidate,
        insn: "`v_add_nc_u32_e32 v9, 0x400, v36`",
        need: "needs Ds wait 1 (War { reg: \"v9\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_gemm_bf16_t1_m3_n1",
        body_pc: 818,
        category: Category::Candidate,
        insn: "`v_add_nc_u32_e32 v17, 0x400, v25`",
        need: "needs Ds wait 0 (War { reg: \"v17\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_gemm_bf16_t1_m3_n2",
        body_pc: 970,
        category: Category::Candidate,
        insn: "`v_add_nc_u32_e32 v25, 0x400, v52`",
        need: "needs Ds wait 1 (War { reg: \"v25\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_gemm_bf16_t1_m4_n1",
        body_pc: 996,
        category: Category::Candidate,
        insn: "`v_add_nc_u32_e32 v25, 0x400, v33`",
        need: "needs Ds wait 1 (War { reg: \"v25\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_gemm_bf16_t1_m4_n2",
        body_pc: 1217,
        category: Category::Candidate,
        insn: "`v_add_nc_u32_e32 v33, 0x400, v68`",
        need: "needs Ds wait 2 (War { reg: \"v33\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_gemm_bf16_t2_m1_n1",
        body_pc: 683,
        category: Category::Candidate,
        insn: "`v_lshlrev_b32_e32 v5, 2, v15`",
        need: "needs Store wait 0 (War { reg: \"v[5:6]\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_gemm_bf16_t2_m1_n2",
        body_pc: 871,
        category: Category::Candidate,
        insn: "`v_lshrrev_b32_e32 v3, 4, v11`",
        need: "needs Store wait 0 (War { reg: \"v[3:4]\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_gemm_bf16_t2_m2_n1",
        body_pc: 1353,
        category: Category::Candidate,
        insn: "`v_bfe_u32 v5, v12, 5, 3`",
        need: "needs Store wait 0 (War { reg: \"v[5:6]\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_gemm_bf16_t2_m2_n2",
        body_pc: 1387,
        category: Category::Candidate,
        insn: "`v_add_nc_u32_e32 v9, 0x400, v33`",
        need: "needs Ds wait 1 (War { reg: \"v9\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_gemm_bf16_t2_m3_n1",
        body_pc: 1835,
        category: Category::Candidate,
        insn: "`v_add_nc_u32_e32 v17, 0x400, v25`",
        need: "needs Ds wait 0 (War { reg: \"v17\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_gemm_bf16_t2_m3_n2",
        body_pc: 1835,
        category: Category::Candidate,
        insn: "`v_add_nc_u32_e32 v25, 0x400, v49`",
        need: "needs Ds wait 1 (War { reg: \"v25\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_gemm_bf16_t2_m4_n1",
        body_pc: 2269,
        category: Category::Candidate,
        insn: "`v_add_nc_u32_e32 v25, 0x400, v33`",
        need: "needs Ds wait 1 (War { reg: \"v25\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_gemm_bf16_t2_m4_n2",
        body_pc: 2412,
        category: Category::Candidate,
        insn: "`v_add_nc_u32_e32 v33, 0x400, v65`",
        need: "needs Ds wait 2 (War { reg: \"v33\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_gemm_bf16_t3_m1_n1",
        body_pc: 1108,
        category: Category::Candidate,
        insn: "`v_lshlrev_b32_e32 v5, 2, v12`",
        need: "needs Store wait 0 (War { reg: \"v[5:6]\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_gemm_bf16_t3_m1_n2",
        body_pc: 1187,
        category: Category::Candidate,
        insn: "`v_lshrrev_b32_e32 v3, 4, v11`",
        need: "needs Store wait 0 (War { reg: \"v[3:4]\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_gemm_bf16_t3_m2_n1",
        body_pc: 1967,
        category: Category::Candidate,
        insn: "`v_bfe_u32 v5, v12, 5, 3`",
        need: "needs Store wait 0 (War { reg: \"v[5:6]\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_gemm_bf16_t3_m2_n2",
        body_pc: 2020,
        category: Category::Candidate,
        insn: "`v_add_nc_u32_e32 v9, 0x400, v33`",
        need: "needs Ds wait 1 (War { reg: \"v9\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_gemm_bf16_t3_m3_n1",
        body_pc: 2805,
        category: Category::Candidate,
        insn: "`v_add_nc_u32_e32 v17, 0x400, v25`",
        need: "needs Ds wait 0 (War { reg: \"v17\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_gemm_bf16_t3_m3_n2",
        body_pc: 2801,
        category: Category::Candidate,
        insn: "`v_add_nc_u32_e32 v25, 0x400, v49`",
        need: "needs Ds wait 1 (War { reg: \"v25\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_gemm_bf16_t3_m4_n1",
        body_pc: 3687,
        category: Category::Candidate,
        insn: "`v_add_nc_u32_e32 v25, 0x400, v33`",
        need: "needs Ds wait 1 (War { reg: \"v25\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_gemm_bf16_t3_m4_n2",
        body_pc: 3661,
        category: Category::Candidate,
        insn: "`v_add_nc_u32_e32 v33, 0x400, v65`",
        need: "needs Ds wait 2 (War { reg: \"v33\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_gemm_w8_t1_m1_n1",
        body_pc: 647,
        category: Category::Candidate,
        insn: "`v_lshlrev_b32_e32 v6, 2, v15`",
        need: "needs Store wait 0 (War { reg: \"v[6:7]\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_gemm_w8_t1_m1_n2",
        body_pc: 972,
        category: Category::Candidate,
        insn: "`v_lshrrev_b32_e32 v3, 4, v11`",
        need: "needs Store wait 0 (War { reg: \"v[3:4]\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_gemm_w8_t1_m2_n1",
        body_pc: 948,
        category: Category::Candidate,
        insn: "`v_bfe_u32 v5, v13, 5, 3`",
        need: "needs Store wait 0 (War { reg: \"v[5:6]\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_gemm_w8_t1_m2_n2",
        body_pc: 1190,
        category: Category::Candidate,
        insn: "`v_add_nc_u32_e32 v20, 0xc00, v20`",
        need: "needs Ds wait 0 (War { reg: \"v20\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_gemm_w8_t1_m3_n1",
        body_pc: 1118,
        category: Category::Candidate,
        insn: "`v_add_nc_u32_e32 v10, 0x800, v10`",
        need: "needs Ds wait 0 (War { reg: \"v10\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_gemm_w8_t1_m3_n2",
        body_pc: 1521,
        category: Category::Candidate,
        insn: "`v_add_nc_u32_e32 v5, 0x1000, v4`",
        need: "needs Ds wait 5 (War { reg: \"v5\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_gemm_w8_t1_m4_n1",
        body_pc: 1343,
        category: Category::ModelLimit,
        insn: "`v_cmp_gt_i32_e64 s1, s16, v9`",
        need: "needs Km wait 0 (Waw { reg: \"s[2:3]\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_gemm_w8_t1_m4_n2",
        body_pc: 1833,
        category: Category::Candidate,
        insn: "`v_add_nc_u32_e32 v2, 0x1400, v1`",
        need: "needs Ds wait 8 (War { reg: \"v2\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_gemm_w8_t2_m1_n1",
        body_pc: 1030,
        category: Category::Candidate,
        insn: "`v_lshlrev_b32_e32 v5, 2, v13`",
        need: "needs Store wait 0 (War { reg: \"v[5:6]\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_gemm_w8_t2_m1_n2",
        body_pc: 1307,
        category: Category::Candidate,
        insn: "`v_lshrrev_b32_e32 v3, 4, v11`",
        need: "needs Store wait 0 (War { reg: \"v[3:4]\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_gemm_w8_t2_m2_n1",
        body_pc: 1734,
        category: Category::Candidate,
        insn: "`v_bfe_u32 v5, v12, 5, 3`",
        need: "needs Store wait 0 (War { reg: \"v[5:6]\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_gemm_w8_t2_m2_n2",
        body_pc: 1876,
        category: Category::Candidate,
        insn: "`v_add_nc_u32_e32 v1, 0xc00, v1`",
        need: "needs Ds wait 0 (War { reg: \"v1\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_gemm_w8_t2_m3_n1",
        body_pc: 2124,
        category: Category::Candidate,
        insn: "`v_add_nc_u32_e32 v1, 0x800, v1`",
        need: "needs Ds wait 0 (War { reg: \"v1\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_gemm_w8_t2_m3_n2",
        body_pc: 2427,
        category: Category::Candidate,
        insn: "`v_add_nc_u32_e32 v2, 0x1000, v1`",
        need: "needs Ds wait 5 (War { reg: \"v2\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_gemm_w8_t2_m4_n1",
        body_pc: 2552,
        category: Category::ModelLimit,
        insn: "`v_cmp_gt_i32_e64 s1, s16, v69`",
        need: "needs Km wait 0 (Waw { reg: \"s[2:3]\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_gemm_w8_t2_m4_n2",
        body_pc: 3023,
        category: Category::Candidate,
        insn: "`v_add_nc_u32_e32 v2, 0x1400, v1`",
        need: "needs Ds wait 8 (War { reg: \"v2\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_gemm_w8_t3_m1_n1",
        body_pc: 1341,
        category: Category::Candidate,
        insn: "`v_lshlrev_b32_e32 v5, 2, v12`",
        need: "needs Store wait 0 (War { reg: \"v[5:6]\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_gemm_w8_t3_m1_n2",
        body_pc: 1624,
        category: Category::Candidate,
        insn: "`v_lshrrev_b32_e32 v3, 4, v11`",
        need: "needs Store wait 0 (War { reg: \"v[3:4]\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_gemm_w8_t3_m2_n1",
        body_pc: 2421,
        category: Category::Candidate,
        insn: "`v_bfe_u32 v5, v12, 5, 3`",
        need: "needs Store wait 0 (War { reg: \"v[5:6]\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_gemm_w8_t3_m2_n2",
        body_pc: 2453,
        category: Category::Candidate,
        insn: "`v_add_nc_u32_e32 v1, 0xc00, v1`",
        need: "needs Ds wait 0 (War { reg: \"v1\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_gemm_w8_t3_m3_n1",
        body_pc: 3070,
        category: Category::Candidate,
        insn: "`v_add_nc_u32_e32 v1, 0x800, v1`",
        need: "needs Ds wait 0 (War { reg: \"v1\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_gemm_w8_t3_m3_n2",
        body_pc: 3380,
        category: Category::Candidate,
        insn: "`v_add_nc_u32_e32 v2, 0x1000, v1`",
        need: "needs Ds wait 5 (War { reg: \"v2\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_gemm_w8_t3_m4_n1",
        body_pc: 4005,
        category: Category::Candidate,
        insn: "`v_add_nc_u32_e32 v1, 0xc00, v1`",
        need: "needs Ds wait 0 (War { reg: \"v1\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_gemm_w8_t3_m4_n2",
        body_pc: 4305,
        category: Category::Candidate,
        insn: "`v_add_nc_u32_e32 v2, 0x1400, v1`",
        need: "needs Ds wait 8 (War { reg: \"v2\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_gemm_bf16_t2_m1_n1_multi",
        body_pc: 818,
        category: Category::Candidate,
        insn: "`v_add_nc_u32_e32 v5, 0x80, v4`",
        need: "needs Store wait 0 (War { reg: \"v[5:6]\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_gemm_bf16_t2_m2_n1_multi",
        body_pc: 1494,
        category: Category::Candidate,
        insn: "`v_add_nc_u32_e32 v5, 0x80, v4`",
        need: "needs Store wait 0 (War { reg: \"v[5:6]\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_gemm_bf16_t2_m3_n1_multi",
        body_pc: 1993,
        category: Category::Candidate,
        insn: "`v_add_nc_u32_e32 v17, 0x400, v25`",
        need: "needs Ds wait 0 (War { reg: \"v17\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_gemm_bf16_t2_m4_n1_multi",
        body_pc: 2423,
        category: Category::Candidate,
        insn: "`v_add_nc_u32_e32 v25, 0x400, v33`",
        need: "needs Ds wait 0 (War { reg: \"v25\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_read_up_bf16_t1_g1",
        body_pc: 293,
        category: Category::Candidate,
        insn: "`v_add_nc_u32_e32 v1, 0x400, v36`",
        need: "needs Ds wait 1 (War { reg: \"v1\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_read_up_bf16_t1_g2",
        body_pc: 293,
        category: Category::Candidate,
        insn: "`v_add_nc_u32_e32 v1, 0x400, v35`",
        need: "needs Ds wait 1 (War { reg: \"v1\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_read_up_bf16_t1_g4",
        body_pc: 295,
        category: Category::Candidate,
        insn: "`v_add_nc_u32_e32 v1, 0x400, v37`",
        need: "needs Ds wait 1 (War { reg: \"v1\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_read_up_bf16_t2_g1",
        body_pc: 378,
        category: Category::Candidate,
        insn: "`v_add_nc_u32_e32 v1, 0x400, v36`",
        need: "needs Ds wait 1 (War { reg: \"v1\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_read_up_bf16_t2_g2",
        body_pc: 378,
        category: Category::Candidate,
        insn: "`v_add_nc_u32_e32 v1, 0x400, v35`",
        need: "needs Ds wait 1 (War { reg: \"v1\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_read_up_bf16_t2_g4",
        body_pc: 380,
        category: Category::Candidate,
        insn: "`v_add_nc_u32_e32 v1, 0x400, v37`",
        need: "needs Ds wait 1 (War { reg: \"v1\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_read_up_bf16_t3_g1",
        body_pc: 469,
        category: Category::Candidate,
        insn: "`v_add_nc_u32_e32 v1, 0x400, v35`",
        need: "needs Ds wait 1 (War { reg: \"v1\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_read_up_bf16_t3_g2",
        body_pc: 469,
        category: Category::Candidate,
        insn: "`v_add_nc_u32_e32 v1, 0x400, v35`",
        need: "needs Ds wait 1 (War { reg: \"v1\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_read_up_bf16_t3_g4",
        body_pc: 480,
        category: Category::Candidate,
        insn: "`v_add_nc_u32_e32 v1, 0x400, v35`",
        need: "needs Ds wait 1 (War { reg: \"v1\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_read_up_w8_t1_g2",
        body_pc: 583,
        category: Category::Candidate,
        insn: "`v_add_nc_u32_e32 v3, 0xc00, v3`",
        need: "needs Ds wait 0 (War { reg: \"v3\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_read_up_w8_t1_g4",
        body_pc: 590,
        category: Category::Candidate,
        insn: "`v_add_nc_u32_e32 v3, 0xc00, v3`",
        need: "needs Ds wait 0 (War { reg: \"v3\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_read_up_w8_t2_g2",
        body_pc: 663,
        category: Category::Candidate,
        insn: "`v_add_nc_u32_e32 v3, 0xc00, v3`",
        need: "needs Ds wait 0 (War { reg: \"v3\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_read_up_w8_t2_g4",
        body_pc: 670,
        category: Category::Candidate,
        insn: "`v_add_nc_u32_e32 v3, 0xc00, v3`",
        need: "needs Ds wait 0 (War { reg: \"v3\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_read_up_w8_t3_g2",
        body_pc: 751,
        category: Category::Candidate,
        insn: "`v_add_nc_u32_e32 v3, 0xc00, v3`",
        need: "needs Ds wait 0 (War { reg: \"v3\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_hipcc_gfx1201.co#lwmma_read_up_w8_t3_g4",
        body_pc: 758,
        category: Category::Candidate,
        insn: "`v_add_nc_u32_e32 v3, 0xc00, v3`",
        need: "needs Ds wait 0 (War { reg: \"v3\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_gemm_bf16_t1_m1_n1",
        body_pc: 440,
        category: Category::Candidate,
        insn: "`v_lshlrev_b32_e32 v5, 2, v15`",
        need: "needs Store wait 0 (War { reg: \"v[5:6]\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_gemm_bf16_t1_m1_n2",
        body_pc: 530,
        category: Category::Candidate,
        insn: "`v_lshrrev_b32_e32 v3, 4, v11`",
        need: "needs Store wait 0 (War { reg: \"v[3:4]\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_gemm_bf16_t1_m2_n1",
        body_pc: 677,
        category: Category::Candidate,
        insn: "`v_bfe_u32 v5, v12, 5, 3`",
        need: "needs Store wait 0 (War { reg: \"v[5:6]\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_gemm_bf16_t1_m2_n2",
        body_pc: 667,
        category: Category::Candidate,
        insn: "`v_add_nc_u32_e32 v9, 0x400, v36`",
        need: "needs Ds wait 1 (War { reg: \"v9\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_gemm_bf16_t1_m3_n1",
        body_pc: 818,
        category: Category::Candidate,
        insn: "`v_add_nc_u32_e32 v17, 0x400, v25`",
        need: "needs Ds wait 0 (War { reg: \"v17\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_gemm_bf16_t1_m3_n2",
        body_pc: 970,
        category: Category::Candidate,
        insn: "`v_add_nc_u32_e32 v25, 0x400, v52`",
        need: "needs Ds wait 1 (War { reg: \"v25\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_gemm_bf16_t1_m4_n1",
        body_pc: 996,
        category: Category::Candidate,
        insn: "`v_add_nc_u32_e32 v25, 0x400, v33`",
        need: "needs Ds wait 1 (War { reg: \"v25\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_gemm_bf16_t1_m4_n2",
        body_pc: 1217,
        category: Category::Candidate,
        insn: "`v_add_nc_u32_e32 v33, 0x400, v68`",
        need: "needs Ds wait 2 (War { reg: \"v33\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_gemm_bf16_t2_m1_n1",
        body_pc: 683,
        category: Category::Candidate,
        insn: "`v_lshlrev_b32_e32 v5, 2, v15`",
        need: "needs Store wait 0 (War { reg: \"v[5:6]\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_gemm_bf16_t2_m1_n2",
        body_pc: 871,
        category: Category::Candidate,
        insn: "`v_lshrrev_b32_e32 v3, 4, v11`",
        need: "needs Store wait 0 (War { reg: \"v[3:4]\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_gemm_bf16_t2_m2_n1",
        body_pc: 1353,
        category: Category::Candidate,
        insn: "`v_bfe_u32 v5, v12, 5, 3`",
        need: "needs Store wait 0 (War { reg: \"v[5:6]\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_gemm_bf16_t2_m2_n2",
        body_pc: 1387,
        category: Category::Candidate,
        insn: "`v_add_nc_u32_e32 v9, 0x400, v33`",
        need: "needs Ds wait 1 (War { reg: \"v9\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_gemm_bf16_t2_m3_n1",
        body_pc: 1835,
        category: Category::Candidate,
        insn: "`v_add_nc_u32_e32 v17, 0x400, v25`",
        need: "needs Ds wait 0 (War { reg: \"v17\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_gemm_bf16_t2_m3_n2",
        body_pc: 1835,
        category: Category::Candidate,
        insn: "`v_add_nc_u32_e32 v25, 0x400, v49`",
        need: "needs Ds wait 1 (War { reg: \"v25\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_gemm_bf16_t2_m4_n1",
        body_pc: 2269,
        category: Category::Candidate,
        insn: "`v_add_nc_u32_e32 v25, 0x400, v33`",
        need: "needs Ds wait 1 (War { reg: \"v25\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_gemm_bf16_t2_m4_n2",
        body_pc: 2412,
        category: Category::Candidate,
        insn: "`v_add_nc_u32_e32 v33, 0x400, v65`",
        need: "needs Ds wait 2 (War { reg: \"v33\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_gemm_bf16_t3_m1_n1",
        body_pc: 1108,
        category: Category::Candidate,
        insn: "`v_lshlrev_b32_e32 v5, 2, v12`",
        need: "needs Store wait 0 (War { reg: \"v[5:6]\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_gemm_bf16_t3_m1_n2",
        body_pc: 1187,
        category: Category::Candidate,
        insn: "`v_lshrrev_b32_e32 v3, 4, v11`",
        need: "needs Store wait 0 (War { reg: \"v[3:4]\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_gemm_bf16_t3_m2_n1",
        body_pc: 1967,
        category: Category::Candidate,
        insn: "`v_bfe_u32 v5, v12, 5, 3`",
        need: "needs Store wait 0 (War { reg: \"v[5:6]\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_gemm_bf16_t3_m2_n2",
        body_pc: 2020,
        category: Category::Candidate,
        insn: "`v_add_nc_u32_e32 v9, 0x400, v33`",
        need: "needs Ds wait 1 (War { reg: \"v9\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_gemm_bf16_t3_m3_n1",
        body_pc: 2805,
        category: Category::Candidate,
        insn: "`v_add_nc_u32_e32 v17, 0x400, v25`",
        need: "needs Ds wait 0 (War { reg: \"v17\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_gemm_bf16_t3_m3_n2",
        body_pc: 2801,
        category: Category::Candidate,
        insn: "`v_add_nc_u32_e32 v25, 0x400, v49`",
        need: "needs Ds wait 1 (War { reg: \"v25\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_gemm_bf16_t3_m4_n1",
        body_pc: 3687,
        category: Category::Candidate,
        insn: "`v_add_nc_u32_e32 v25, 0x400, v33`",
        need: "needs Ds wait 1 (War { reg: \"v25\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_gemm_bf16_t3_m4_n2",
        body_pc: 3661,
        category: Category::Candidate,
        insn: "`v_add_nc_u32_e32 v33, 0x400, v65`",
        need: "needs Ds wait 2 (War { reg: \"v33\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_gemm_w8_t1_m1_n1",
        body_pc: 647,
        category: Category::Candidate,
        insn: "`v_lshlrev_b32_e32 v6, 2, v15`",
        need: "needs Store wait 0 (War { reg: \"v[6:7]\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_gemm_w8_t1_m1_n2",
        body_pc: 972,
        category: Category::Candidate,
        insn: "`v_lshrrev_b32_e32 v3, 4, v11`",
        need: "needs Store wait 0 (War { reg: \"v[3:4]\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_gemm_w8_t1_m2_n1",
        body_pc: 948,
        category: Category::Candidate,
        insn: "`v_bfe_u32 v5, v13, 5, 3`",
        need: "needs Store wait 0 (War { reg: \"v[5:6]\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_gemm_w8_t1_m2_n2",
        body_pc: 1190,
        category: Category::Candidate,
        insn: "`v_add_nc_u32_e32 v20, 0xc00, v20`",
        need: "needs Ds wait 0 (War { reg: \"v20\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_gemm_w8_t1_m3_n1",
        body_pc: 1118,
        category: Category::Candidate,
        insn: "`v_add_nc_u32_e32 v10, 0x800, v10`",
        need: "needs Ds wait 0 (War { reg: \"v10\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_gemm_w8_t1_m3_n2",
        body_pc: 1521,
        category: Category::Candidate,
        insn: "`v_add_nc_u32_e32 v5, 0x1000, v4`",
        need: "needs Ds wait 5 (War { reg: \"v5\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_gemm_w8_t1_m4_n1",
        body_pc: 1343,
        category: Category::ModelLimit,
        insn: "`v_cmp_gt_i32_e64 s1, s16, v9`",
        need: "needs Km wait 0 (Waw { reg: \"s[2:3]\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_gemm_w8_t1_m4_n2",
        body_pc: 1833,
        category: Category::Candidate,
        insn: "`v_add_nc_u32_e32 v2, 0x1400, v1`",
        need: "needs Ds wait 8 (War { reg: \"v2\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_gemm_w8_t2_m1_n1",
        body_pc: 1030,
        category: Category::Candidate,
        insn: "`v_lshlrev_b32_e32 v5, 2, v13`",
        need: "needs Store wait 0 (War { reg: \"v[5:6]\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_gemm_w8_t2_m1_n2",
        body_pc: 1307,
        category: Category::Candidate,
        insn: "`v_lshrrev_b32_e32 v3, 4, v11`",
        need: "needs Store wait 0 (War { reg: \"v[3:4]\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_gemm_w8_t2_m2_n1",
        body_pc: 1734,
        category: Category::Candidate,
        insn: "`v_bfe_u32 v5, v12, 5, 3`",
        need: "needs Store wait 0 (War { reg: \"v[5:6]\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_gemm_w8_t2_m2_n2",
        body_pc: 1876,
        category: Category::Candidate,
        insn: "`v_add_nc_u32_e32 v1, 0xc00, v1`",
        need: "needs Ds wait 0 (War { reg: \"v1\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_gemm_w8_t2_m3_n1",
        body_pc: 2124,
        category: Category::Candidate,
        insn: "`v_add_nc_u32_e32 v1, 0x800, v1`",
        need: "needs Ds wait 0 (War { reg: \"v1\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_gemm_w8_t2_m3_n2",
        body_pc: 2427,
        category: Category::Candidate,
        insn: "`v_add_nc_u32_e32 v2, 0x1000, v1`",
        need: "needs Ds wait 5 (War { reg: \"v2\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_gemm_w8_t2_m4_n1",
        body_pc: 2552,
        category: Category::ModelLimit,
        insn: "`v_cmp_gt_i32_e64 s1, s16, v69`",
        need: "needs Km wait 0 (Waw { reg: \"s[2:3]\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_gemm_w8_t2_m4_n2",
        body_pc: 3023,
        category: Category::Candidate,
        insn: "`v_add_nc_u32_e32 v2, 0x1400, v1`",
        need: "needs Ds wait 8 (War { reg: \"v2\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_gemm_w8_t3_m1_n1",
        body_pc: 1341,
        category: Category::Candidate,
        insn: "`v_lshlrev_b32_e32 v5, 2, v12`",
        need: "needs Store wait 0 (War { reg: \"v[5:6]\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_gemm_w8_t3_m1_n2",
        body_pc: 1624,
        category: Category::Candidate,
        insn: "`v_lshrrev_b32_e32 v3, 4, v11`",
        need: "needs Store wait 0 (War { reg: \"v[3:4]\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_gemm_w8_t3_m2_n1",
        body_pc: 2421,
        category: Category::Candidate,
        insn: "`v_bfe_u32 v5, v12, 5, 3`",
        need: "needs Store wait 0 (War { reg: \"v[5:6]\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_gemm_w8_t3_m2_n2",
        body_pc: 2453,
        category: Category::Candidate,
        insn: "`v_add_nc_u32_e32 v1, 0xc00, v1`",
        need: "needs Ds wait 0 (War { reg: \"v1\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_gemm_w8_t3_m3_n1",
        body_pc: 3070,
        category: Category::Candidate,
        insn: "`v_add_nc_u32_e32 v1, 0x800, v1`",
        need: "needs Ds wait 0 (War { reg: \"v1\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_gemm_w8_t3_m3_n2",
        body_pc: 3380,
        category: Category::Candidate,
        insn: "`v_add_nc_u32_e32 v2, 0x1000, v1`",
        need: "needs Ds wait 5 (War { reg: \"v2\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_gemm_w8_t3_m4_n1",
        body_pc: 4005,
        category: Category::Candidate,
        insn: "`v_add_nc_u32_e32 v1, 0xc00, v1`",
        need: "needs Ds wait 0 (War { reg: \"v1\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_gemm_w8_t3_m4_n2",
        body_pc: 4305,
        category: Category::Candidate,
        insn: "`v_add_nc_u32_e32 v2, 0x1400, v1`",
        need: "needs Ds wait 8 (War { reg: \"v2\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_gemm_bf16_t2_m1_n1_multi",
        body_pc: 818,
        category: Category::Candidate,
        insn: "`v_add_nc_u32_e32 v5, 0x80, v4`",
        need: "needs Store wait 0 (War { reg: \"v[5:6]\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_gemm_bf16_t2_m2_n1_multi",
        body_pc: 1494,
        category: Category::Candidate,
        insn: "`v_add_nc_u32_e32 v5, 0x80, v4`",
        need: "needs Store wait 0 (War { reg: \"v[5:6]\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_gemm_bf16_t2_m3_n1_multi",
        body_pc: 1993,
        category: Category::Candidate,
        insn: "`v_add_nc_u32_e32 v17, 0x400, v25`",
        need: "needs Ds wait 0 (War { reg: \"v17\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_gemm_bf16_t2_m4_n1_multi",
        body_pc: 2423,
        category: Category::Candidate,
        insn: "`v_add_nc_u32_e32 v25, 0x400, v33`",
        need: "needs Ds wait 0 (War { reg: \"v25\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_read_up_bf16_t1_g1",
        body_pc: 293,
        category: Category::Candidate,
        insn: "`v_add_nc_u32_e32 v1, 0x400, v36`",
        need: "needs Ds wait 1 (War { reg: \"v1\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_read_up_bf16_t1_g2",
        body_pc: 293,
        category: Category::Candidate,
        insn: "`v_add_nc_u32_e32 v1, 0x400, v35`",
        need: "needs Ds wait 1 (War { reg: \"v1\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_read_up_bf16_t1_g4",
        body_pc: 295,
        category: Category::Candidate,
        insn: "`v_add_nc_u32_e32 v1, 0x400, v37`",
        need: "needs Ds wait 1 (War { reg: \"v1\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_read_up_bf16_t2_g1",
        body_pc: 378,
        category: Category::Candidate,
        insn: "`v_add_nc_u32_e32 v1, 0x400, v36`",
        need: "needs Ds wait 1 (War { reg: \"v1\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_read_up_bf16_t2_g2",
        body_pc: 378,
        category: Category::Candidate,
        insn: "`v_add_nc_u32_e32 v1, 0x400, v35`",
        need: "needs Ds wait 1 (War { reg: \"v1\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_read_up_bf16_t2_g4",
        body_pc: 380,
        category: Category::Candidate,
        insn: "`v_add_nc_u32_e32 v1, 0x400, v37`",
        need: "needs Ds wait 1 (War { reg: \"v1\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_read_up_bf16_t3_g1",
        body_pc: 469,
        category: Category::Candidate,
        insn: "`v_add_nc_u32_e32 v1, 0x400, v35`",
        need: "needs Ds wait 1 (War { reg: \"v1\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_read_up_bf16_t3_g2",
        body_pc: 469,
        category: Category::Candidate,
        insn: "`v_add_nc_u32_e32 v1, 0x400, v35`",
        need: "needs Ds wait 1 (War { reg: \"v1\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_read_up_bf16_t3_g4",
        body_pc: 480,
        category: Category::Candidate,
        insn: "`v_add_nc_u32_e32 v1, 0x400, v35`",
        need: "needs Ds wait 1 (War { reg: \"v1\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_read_up_w8_t1_g2",
        body_pc: 583,
        category: Category::Candidate,
        insn: "`v_add_nc_u32_e32 v3, 0xc00, v3`",
        need: "needs Ds wait 0 (War { reg: \"v3\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_read_up_w8_t1_g4",
        body_pc: 590,
        category: Category::Candidate,
        insn: "`v_add_nc_u32_e32 v3, 0xc00, v3`",
        need: "needs Ds wait 0 (War { reg: \"v3\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_read_up_w8_t2_g2",
        body_pc: 663,
        category: Category::Candidate,
        insn: "`v_add_nc_u32_e32 v3, 0xc00, v3`",
        need: "needs Ds wait 0 (War { reg: \"v3\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_read_up_w8_t2_g4",
        body_pc: 670,
        category: Category::Candidate,
        insn: "`v_add_nc_u32_e32 v3, 0xc00, v3`",
        need: "needs Ds wait 0 (War { reg: \"v3\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_read_up_w8_t3_g2",
        body_pc: 751,
        category: Category::Candidate,
        insn: "`v_add_nc_u32_e32 v3, 0xc00, v3`",
        need: "needs Ds wait 0 (War { reg: \"v3\" })",
    },
    // S1 hosted copy of the qwen4_lane_wmma_multi pins finding
    PoolBody {
        target: "qwen4_lane_wmma_multi_pm_gfx1201.hxaco#lwmma_read_up_w8_t3_g4",
        body_pc: 758,
        category: Category::Candidate,
        insn: "`v_add_nc_u32_e32 v3, 0xc00, v3`",
        need: "needs Ds wait 0 (War { reg: \"v3\" })",
    },
    // X8 twins: moe_route_scatter_wide and qwen4_moe_ep_a8_rot (pins rows above; tids 259-264, 513-532)
    PoolBody {
        target: "moe_route_scatter_wide_hipcc_gfx1201.co#moe_router_softmax_top10_f32_fast",
        body_pc: 255,
        category: Category::Candidate,
        insn: "`v_mov_b32_e32 v2, v1`",
        need: "needs Store wait 0 (War { reg: \"v[2:5]\" })",
    },
    PoolBody {
        target: "moe_route_scatter_wide_hipcc_gfx1201.co#moe_scatter_fused_k8",
        body_pc: 19,
        category: Category::Candidate,
        insn: "`v_add_nc_u32_e32 v1, s4, v1`",
        need: "needs Ds wait 0 (War { reg: \"v1\" })",
    },
    PoolBody {
        target: "moe_route_scatter_wide_hipcc_gfx1201.co#moe_scatter_fused_top10_pmask",
        body_pc: 19,
        category: Category::Candidate,
        insn: "`v_add_nc_u32_e32 v1, s4, v1`",
        need: "needs Ds wait 0 (War { reg: \"v1\" })",
    },
    PoolBody {
        target: "moe_route_scatter_wide_pm_gfx1201.hxaco#moe_router_softmax_top10_f32_fast",
        body_pc: 255,
        category: Category::Candidate,
        insn: "`v_mov_b32_e32 v2, v1`",
        need: "needs Store wait 0 (War { reg: \"v[2:5]\" })",
    },
    PoolBody {
        target: "moe_route_scatter_wide_pm_gfx1201.hxaco#moe_scatter_fused_k8",
        body_pc: 19,
        category: Category::Candidate,
        insn: "`v_add_nc_u32_e32 v1, s4, v1`",
        need: "needs Ds wait 0 (War { reg: \"v1\" })",
    },
    PoolBody {
        target: "moe_route_scatter_wide_pm_gfx1201.hxaco#moe_scatter_fused_top10_pmask",
        body_pc: 19,
        category: Category::Candidate,
        insn: "`v_add_nc_u32_e32 v1, s4, v1`",
        need: "needs Ds wait 0 (War { reg: \"v1\" })",
    },
    PoolBody {
        target: "qwen4_moe_ep_a8_rot_hipcc_gfx1201.co#qwen4_moe_ep_a8_gate_up_nt1",
        body_pc: 126,
        category: Category::Candidate,
        insn: "`flat_load_b128 v[0:3], v[6:7], off offset:8`",
        need: "needs Load wait 0 (War { reg: \"v[0:1]\" })",
    },
    PoolBody {
        target: "qwen4_moe_ep_a8_rot_hipcc_gfx1201.co#qwen4_moe_ep_a8_gate_up_nt2",
        body_pc: 234,
        category: Category::ModelLimit,
        insn: "`flat_load_d16_hi_b16 v32, v[72:73], off`",
        need: "needs Load wait 0 (Use { reg: \"v32\" })",
    },
    PoolBody {
        target: "qwen4_moe_ep_a8_rot_hipcc_gfx1201.co#qwen4_moe_ep_a8_gate_up_planar_nt1",
        body_pc: 130,
        category: Category::ModelLimit,
        insn: "`flat_load_d16_b16 v32, v[6:7], off`",
        need: "needs Load wait 0 (Use { reg: \"v32\" })",
    },
    PoolBody {
        target: "qwen4_moe_ep_a8_rot_hipcc_gfx1201.co#qwen4_moe_ep_a8_gate_up_planar_nt2",
        body_pc: 242,
        category: Category::ModelLimit,
        insn: "`flat_load_d16_b16 v32, v[72:73], off`",
        need: "needs Load wait 0 (Use { reg: \"v32\" })",
    },
    PoolBody {
        target: "qwen4_moe_ep_a8_rot_hipcc_gfx1201.co#qwen4_moe_ep_a8_down_nt1",
        body_pc: 115,
        category: Category::Candidate,
        insn: "`flat_load_b128 v[12:15], v[12:13], off offset:20`",
        need: "needs Load wait 0 (War { reg: \"v[12:13]\" })",
    },
    PoolBody {
        target: "qwen4_moe_ep_a8_rot_hipcc_gfx1201.co#qwen4_moe_ep_a8_down_nt2",
        body_pc: 282,
        category: Category::ModelLimit,
        insn: "`v_cvt_f32_f16_e32 v63, v24.l`",
        need: "needs Load wait 0 (Use { reg: \"v24\" })",
    },
    PoolBody {
        target: "qwen4_moe_ep_a8_rot_hipcc_gfx1201.co#qwen4_moe_ep_a8_down_planar_nt1",
        body_pc: 116,
        category: Category::Candidate,
        insn: "`flat_load_b128 v[8:11], v[1:2], off`",
        need: "needs Load wait 0 (War { reg: \"v[8:9]\" })",
    },
    PoolBody {
        target: "qwen4_moe_ep_a8_rot_hipcc_gfx1201.co#qwen4_moe_ep_a8_down_planar_nt2",
        body_pc: 284,
        category: Category::ModelLimit,
        insn: "`v_cvt_f32_f16_e32 v65, v24.l`",
        need: "needs Load wait 0 (Use { reg: \"v24\" })",
    },
    PoolBody {
        target: "qwen4_moe_ep_a8_rot_hipcc_gfx1201.co#qwen4_moe_ep_a8_gate_up_rot_planar_nt1",
        body_pc: 168,
        category: Category::ModelLimit,
        insn: "`flat_load_d16_hi_b16 v36, v[21:22], off`",
        need: "needs Load wait 0 (Use { reg: \"v36\" })",
    },
    PoolBody {
        target: "qwen4_moe_ep_a8_rot_hipcc_gfx1201.co#qwen4_moe_ep_a8_gate_up_rot_planar_nt2",
        body_pc: 180,
        category: Category::ModelLimit,
        insn: "`v_cmp_le_u32_e64 s1, s5, v2`",
        need: "needs Km wait 0 (Waw { reg: \"s[2:3]\" })",
    },
    PoolBody {
        target: "qwen4_moe_ep_a8_rot_pm_gfx1201.hxaco#qwen4_moe_ep_a8_gate_up_nt1",
        body_pc: 126,
        category: Category::Candidate,
        insn: "`flat_load_b128 v[0:3], v[6:7], off offset:8`",
        need: "needs Load wait 0 (War { reg: \"v[0:1]\" })",
    },
    PoolBody {
        target: "qwen4_moe_ep_a8_rot_pm_gfx1201.hxaco#qwen4_moe_ep_a8_gate_up_nt2",
        body_pc: 234,
        category: Category::ModelLimit,
        insn: "`flat_load_d16_hi_b16 v32, v[72:73], off`",
        need: "needs Load wait 0 (Use { reg: \"v32\" })",
    },
    PoolBody {
        target: "qwen4_moe_ep_a8_rot_pm_gfx1201.hxaco#qwen4_moe_ep_a8_gate_up_planar_nt1",
        body_pc: 130,
        category: Category::ModelLimit,
        insn: "`flat_load_d16_b16 v32, v[6:7], off`",
        need: "needs Load wait 0 (Use { reg: \"v32\" })",
    },
    PoolBody {
        target: "qwen4_moe_ep_a8_rot_pm_gfx1201.hxaco#qwen4_moe_ep_a8_gate_up_planar_nt2",
        body_pc: 242,
        category: Category::ModelLimit,
        insn: "`flat_load_d16_b16 v32, v[72:73], off`",
        need: "needs Load wait 0 (Use { reg: \"v32\" })",
    },
    PoolBody {
        target: "qwen4_moe_ep_a8_rot_pm_gfx1201.hxaco#qwen4_moe_ep_a8_down_nt1",
        body_pc: 115,
        category: Category::Candidate,
        insn: "`flat_load_b128 v[12:15], v[12:13], off offset:20`",
        need: "needs Load wait 0 (War { reg: \"v[12:13]\" })",
    },
    PoolBody {
        target: "qwen4_moe_ep_a8_rot_pm_gfx1201.hxaco#qwen4_moe_ep_a8_down_nt2",
        body_pc: 282,
        category: Category::ModelLimit,
        insn: "`v_cvt_f32_f16_e32 v63, v24.l`",
        need: "needs Load wait 0 (Use { reg: \"v24\" })",
    },
    PoolBody {
        target: "qwen4_moe_ep_a8_rot_pm_gfx1201.hxaco#qwen4_moe_ep_a8_down_planar_nt1",
        body_pc: 116,
        category: Category::Candidate,
        insn: "`flat_load_b128 v[8:11], v[1:2], off`",
        need: "needs Load wait 0 (War { reg: \"v[8:9]\" })",
    },
    PoolBody {
        target: "qwen4_moe_ep_a8_rot_pm_gfx1201.hxaco#qwen4_moe_ep_a8_down_planar_nt2",
        body_pc: 284,
        category: Category::ModelLimit,
        insn: "`v_cvt_f32_f16_e32 v65, v24.l`",
        need: "needs Load wait 0 (Use { reg: \"v24\" })",
    },
    PoolBody {
        target: "qwen4_moe_ep_a8_rot_pm_gfx1201.hxaco#qwen4_moe_ep_a8_gate_up_rot_planar_nt1",
        body_pc: 168,
        category: Category::ModelLimit,
        insn: "`flat_load_d16_hi_b16 v36, v[21:22], off`",
        need: "needs Load wait 0 (Use { reg: \"v36\" })",
    },
    PoolBody {
        target: "qwen4_moe_ep_a8_rot_pm_gfx1201.hxaco#qwen4_moe_ep_a8_gate_up_rot_planar_nt2",
        body_pc: 180,
        category: Category::ModelLimit,
        insn: "`v_cmp_le_u32_e64 s1, s5, v2`",
        need: "needs Km wait 0 (Waw { reg: \"s[2:3]\" })",
    },
    // P3 hosted copy of the qwen4_lwmma_tile pins finding
    PoolBody {
        target: "qwen4_lwmma_tile_hipcc_gfx1201.co#lwmma_tile_bf16_t1_b32x32w2x1g4",
        body_pc: 742,
        category: Category::Candidate,
        insn: "`v_or_b32_e32 v4, 1, v3`",
        need: "needs Store wait 0 (War { reg: \"v[4:5]\" })",
    },
    // P3 hosted copy of the qwen4_lwmma_tile pins finding
    PoolBody {
        target: "qwen4_lwmma_tile_hipcc_gfx1201.co#lwmma_tile_bf16_t1_b32x64w2x2",
        body_pc: 712,
        category: Category::Candidate,
        insn: "`v_or_b32_e32 v4, 1, v3`",
        need: "needs Store wait 0 (War { reg: \"v[4:5]\" })",
    },
    // P3 hosted copy of the qwen4_lwmma_tile pins finding
    PoolBody {
        target: "qwen4_lwmma_tile_hipcc_gfx1201.co#lwmma_tile_bf16_t1_b64x64w4x1",
        body_pc: 1034,
        category: Category::Candidate,
        insn: "`v_or_b32_e32 v4, 1, v3`",
        need: "needs Store wait 0 (War { reg: \"v[4:5]\" })",
    },
    // P3 hosted copy of the qwen4_lwmma_tile pins finding
    PoolBody {
        target: "qwen4_lwmma_tile_hipcc_gfx1201.co#lwmma_tile_bf16_t2_b16x32w1x1g4",
        body_pc: 320,
        category: Category::ModelLimit,
        insn: "`v_cmp_ne_u32_e64 s1, 0, v45`",
        need: "needs Km wait 0 (Waw { reg: \"s[2:3]\" })",
    },
    // P3 hosted copy of the qwen4_lwmma_tile pins finding
    PoolBody {
        target: "qwen4_lwmma_tile_hipcc_gfx1201.co#lwmma_tile_bf16_t2_b32x64w2x2",
        body_pc: 799,
        category: Category::Candidate,
        insn: "`v_or_b32_e32 v4, 1, v3`",
        need: "needs Store wait 0 (War { reg: \"v[4:5]\" })",
    },
    // P3 hosted copy of the qwen4_lwmma_tile pins finding
    PoolBody {
        target: "qwen4_lwmma_tile_hipcc_gfx1201.co#lwmma_tile_bf16_t2_b64x64w4x1",
        body_pc: 1202,
        category: Category::Candidate,
        insn: "`v_or_b32_e32 v4, 1, v3`",
        need: "needs Store wait 0 (War { reg: \"v[4:5]\" })",
    },
    // P3 hosted copy of the qwen4_lwmma_tile pins finding
    PoolBody {
        target: "qwen4_lwmma_tile_hipcc_gfx1201.co#lwmma_tile_bf16_t2_b64x64w4x2",
        body_pc: 768,
        category: Category::Candidate,
        insn: "`v_or_b32_e32 v4, 1, v3`",
        need: "needs Store wait 0 (War { reg: \"v[4:5]\" })",
    },
    // P3 hosted copy of the qwen4_lwmma_tile pins finding
    PoolBody {
        target: "qwen4_lwmma_tile_hipcc_gfx1201.co#lwmma_tile_bf16_t2_b32x128w2x2",
        body_pc: 1101,
        category: Category::Candidate,
        insn: "`v_or_b32_e32 v4, 1, v3`",
        need: "needs Store wait 0 (War { reg: \"v[4:5]\" })",
    },
    // P3 hosted copy of the qwen4_lwmma_tile pins finding
    PoolBody {
        target: "qwen4_lwmma_tile_hipcc_gfx1201.co#lwmma_tile_bf16_t2_b64x128w4x2",
        body_pc: 1062,
        category: Category::Candidate,
        insn: "`v_or_b32_e32 v4, 1, v3`",
        need: "needs Store wait 0 (War { reg: \"v[4:5]\" })",
    },
    // P3 hosted copy of the qwen4_lwmma_tile pins finding
    PoolBody {
        target: "qwen4_lwmma_tile_hipcc_gfx1201.co#lwmma_tile_bf16_t2_b64x128w2x4",
        body_pc: 1000,
        category: Category::Candidate,
        insn: "`v_or_b32_e32 v4, 1, v3`",
        need: "needs Store wait 0 (War { reg: \"v[4:5]\" })",
    },
    // P3 hosted copy of the qwen4_lwmma_tile pins finding
    PoolBody {
        target: "qwen4_lwmma_tile_pm_gfx1201.hxaco#lwmma_tile_bf16_t1_b32x32w2x1g4",
        body_pc: 742,
        category: Category::Candidate,
        insn: "`v_or_b32_e32 v4, 1, v3`",
        need: "needs Store wait 0 (War { reg: \"v[4:5]\" })",
    },
    // P3 hosted copy of the qwen4_lwmma_tile pins finding
    PoolBody {
        target: "qwen4_lwmma_tile_pm_gfx1201.hxaco#lwmma_tile_bf16_t1_b32x64w2x2",
        body_pc: 712,
        category: Category::Candidate,
        insn: "`v_or_b32_e32 v4, 1, v3`",
        need: "needs Store wait 0 (War { reg: \"v[4:5]\" })",
    },
    // P3 hosted copy of the qwen4_lwmma_tile pins finding
    PoolBody {
        target: "qwen4_lwmma_tile_pm_gfx1201.hxaco#lwmma_tile_bf16_t1_b64x64w4x1",
        body_pc: 1034,
        category: Category::Candidate,
        insn: "`v_or_b32_e32 v4, 1, v3`",
        need: "needs Store wait 0 (War { reg: \"v[4:5]\" })",
    },
    // P3 hosted copy of the qwen4_lwmma_tile pins finding
    PoolBody {
        target: "qwen4_lwmma_tile_pm_gfx1201.hxaco#lwmma_tile_bf16_t2_b16x32w1x1g4",
        body_pc: 320,
        category: Category::ModelLimit,
        insn: "`v_cmp_ne_u32_e64 s1, 0, v45`",
        need: "needs Km wait 0 (Waw { reg: \"s[2:3]\" })",
    },
    // P3 hosted copy of the qwen4_lwmma_tile pins finding
    PoolBody {
        target: "qwen4_lwmma_tile_pm_gfx1201.hxaco#lwmma_tile_bf16_t2_b32x64w2x2",
        body_pc: 799,
        category: Category::Candidate,
        insn: "`v_or_b32_e32 v4, 1, v3`",
        need: "needs Store wait 0 (War { reg: \"v[4:5]\" })",
    },
    // P3 hosted copy of the qwen4_lwmma_tile pins finding
    PoolBody {
        target: "qwen4_lwmma_tile_pm_gfx1201.hxaco#lwmma_tile_bf16_t2_b64x64w4x1",
        body_pc: 1202,
        category: Category::Candidate,
        insn: "`v_or_b32_e32 v4, 1, v3`",
        need: "needs Store wait 0 (War { reg: \"v[4:5]\" })",
    },
    // P3 hosted copy of the qwen4_lwmma_tile pins finding
    PoolBody {
        target: "qwen4_lwmma_tile_pm_gfx1201.hxaco#lwmma_tile_bf16_t2_b64x64w4x2",
        body_pc: 768,
        category: Category::Candidate,
        insn: "`v_or_b32_e32 v4, 1, v3`",
        need: "needs Store wait 0 (War { reg: \"v[4:5]\" })",
    },
    // P3 hosted copy of the qwen4_lwmma_tile pins finding
    PoolBody {
        target: "qwen4_lwmma_tile_pm_gfx1201.hxaco#lwmma_tile_bf16_t2_b32x128w2x2",
        body_pc: 1101,
        category: Category::Candidate,
        insn: "`v_or_b32_e32 v4, 1, v3`",
        need: "needs Store wait 0 (War { reg: \"v[4:5]\" })",
    },
    // P3 hosted copy of the qwen4_lwmma_tile pins finding
    PoolBody {
        target: "qwen4_lwmma_tile_pm_gfx1201.hxaco#lwmma_tile_bf16_t2_b64x128w4x2",
        body_pc: 1062,
        category: Category::Candidate,
        insn: "`v_or_b32_e32 v4, 1, v3`",
        need: "needs Store wait 0 (War { reg: \"v[4:5]\" })",
    },
    // P3 hosted copy of the qwen4_lwmma_tile pins finding
    PoolBody {
        target: "qwen4_lwmma_tile_pm_gfx1201.hxaco#lwmma_tile_bf16_t2_b64x128w2x4",
        body_pc: 1000,
        category: Category::Candidate,
        insn: "`v_or_b32_e32 v4, 1, v3`",
        need: "needs Store wait 0 (War { reg: \"v[4:5]\" })",
    },
];

/// A hosted-body verifier STATE-CAP refusal inside a pool: `ledger_cfg` stops with
/// `wait verification: more than 512 distinct in-flight memory states reach label
/// `.Lt<TID>b_<n>`` (path-sensitive state explosion). OPEN: verifier state cap, not proven.
/// This is neither a verified body nor a hazard finding, and full strict wait certification
/// is NOT claimed for these bodies; they are the pool-hosted copies of the standalone
/// `UNSUPPORTED` `.LBB28` rows of the same template target.
///
/// Exact, no wildcard: the pool (`pool_target`, so a finding moving to another pool, single
/// or multi, is a changed outcome), the hosted template (`template_target`, `file#symbol`;
/// the current template id is mapped from it at run time, never hardcoded), the canonical
/// body label (`.LBB<n>`, the standalone spelling of the pool's `.Lt<TID>b_<n>`) and the
/// whole error text with that template's current id. The named label must lie inside that
/// template's emitted body span (never the dispatcher, entry or exit).
pub struct PoolStateCap {
    pub pool_target: &'static str,
    pub template_target: &'static str,
    pub body_label: &'static str,
    pub error: &'static str,
    pub why: &'static str,
}

const CAP_WHY: &str = "OPEN: verifier state cap, not proven: the 512 in-flight-memory-state cap is reached at `.LBB28` of the hosted body, so the strict wait proof is NOT established for it; not a claimed race and not a successful full certification";

/// `{tid}` and `{n}` are replaced by the mapped template id and the label's index (28).
pub const STATE_CAP_ERROR: &str = "wait verification: more than 512 distinct in-flight memory states reach label `.Lt{tid}b_{n}` (path-sensitive state explosion); wait the counters before the branches that reach it";

/// The five hosted state-cap refusals observed in `ld::pool` strict verification at blockDim
/// 64: four single-template pools and the multi pool, whose reported template is the last
/// hosted one.
pub const POOL_STATE_CAP: &[PoolStateCap] = &[
    PoolStateCap {
        pool_target: "pool:b64:moe_combine_ordered_hipcc_gfx1201.co#moe_down_combine_grouped_top10_bf16in_ordered",
        template_target: "moe_combine_ordered_hipcc_gfx1201.co#moe_down_combine_grouped_top10_bf16in_ordered",
        body_label: ".LBB28",
        error: STATE_CAP_ERROR,
        why: CAP_WHY,
    },
    PoolStateCap {
        pool_target: "pool:b64:moe_combine_ordered_hipcc_gfx1201.co#moe_down_combine_grouped_top10_bf16in_ordered_zinit",
        template_target: "moe_combine_ordered_hipcc_gfx1201.co#moe_down_combine_grouped_top10_bf16in_ordered_zinit",
        body_label: ".LBB28",
        error: STATE_CAP_ERROR,
        why: CAP_WHY,
    },
    PoolStateCap {
        pool_target: "pool:b64:moe_combine_ordered_pm_gfx1201.hxaco#moe_down_combine_grouped_top10_bf16in_ordered",
        template_target: "moe_combine_ordered_pm_gfx1201.hxaco#moe_down_combine_grouped_top10_bf16in_ordered",
        body_label: ".LBB28",
        error: STATE_CAP_ERROR,
        why: CAP_WHY,
    },
    PoolStateCap {
        pool_target: "pool:b64:moe_combine_ordered_pm_gfx1201.hxaco#moe_down_combine_grouped_top10_bf16in_ordered_zinit",
        template_target: "moe_combine_ordered_pm_gfx1201.hxaco#moe_down_combine_grouped_top10_bf16in_ordered_zinit",
        body_label: ".LBB28",
        error: STATE_CAP_ERROR,
        why: CAP_WHY,
    },
    PoolStateCap {
        pool_target: "pool:b64:multi",
        template_target: "moe_combine_ordered_pm_gfx1201.hxaco#moe_down_combine_grouped_top10_bf16in_ordered_zinit",
        body_label: ".LBB28",
        error: STATE_CAP_ERROR,
        why: CAP_WHY,
    },
];
