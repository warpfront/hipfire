// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! npu-gemm: run `peacemaker_npu::kernels::gemm_i8::design` (single core) or, with `--array`/`--variant`,
//! `peacemaker_npu::kernels::gemm_array::design_variant_core` (whole 8x4 array or a bisect variant, selectable core
//! program) on the NPU via railgun::npu.
//!
//! usage: npu-gemm M N K [--iters I] [--col C0] [--timeout-ms T] [--dump-status]
//!        npu-gemm M N K --array [--core Fast|Serial|LockOnly|FastSlowCtl|ClockProbe] [--probe-iters N] [--iters I] [--reuse-ctx] [--loop SECONDS [--gap-us G]] [--timeout-ms T] [--dump-status]
//!        npu-gemm M N K --variant V1|V2|V3|V4|V5|V6|V6b [--core Fast|Serial|LockOnly|FastSlowCtl|ClockProbe] [--probe-iters N] [--iters I] [--timeout-ms T] [--dump-status]
//!        npu-gemm M N K --variant V8|V9|V10 [--epi i32|int8] [--shift S] [--core Fast|FastSlowCtl] [--ctl-probe knob,...] [--a-from-hip] [--iters I] [--reuse-ctx] [--loop SECONDS [--gap-us G]] [--timeout-ms T] [--dump-status]
//!        ... [--memory native|vmm-host|vmm-host-uncached] [--burst 0..3] [--axcache 0..15] [--axqos 0..15]
//!        npu-gemm M 1280 K --variant G80 [--epi i32|int8|bf16] [--rowmajor-pitch P --rowmajor-col0 C0] [--shift S] [--core Fast|FastSlowCtl] [--iters I] [--reuse-ctx] [--timeout-ms T].
//!        npu-gemm M 512 K --variant G80 --epi bf16 --n512 [--rowmajor-pitch P --rowmajor-col0 C0] [--iters I] [--reuse-ctx] [--loop SECONDS]   (out-projection design, K <= 6336).
//!        npu-gemm M N K --variant V10 --b-prefix [same V8|V9|V10 options]             (B prefix mode; V10 only)
//!        npu-gemm TOKENS FEATURES K --fold-contract ief15 [--iters N] [--loop SECS] [--reuse-ctx] [--lean] [--switch-with T2,F2,K2] [--timeout-ms T] [--dump-status]   (IEF15 N0 probe; K a multiple of 128 with 1..=136 epochs)
//!        npu-gemm --ief15-probe [--col C] [--timeout-ms T] [--dump PATH] [--dump-status] [--expected-only]   (IEF15 N0 single-core semantics probe: expected bytes from the simulator running the exact PDI+TXN, one fresh-context submit, per-case MATCH/MISMATCH table; `--expected-only` never touches the hardware)
//! Single-core (native default, `npu-gemm 64 64 64`): each iteration uses a fresh hardware context (the design
//! runs its job list once per PDI load). Output is compared with an exact CPU int8×int8→int32 reference;
//! timing is submit→completion wall time and is reported with the useful-op count 2·M·N·K. `--core` is an
//! array-only option: giving it without `--array`/`--variant` is rejected (exit 2), never silently ignored.
//! Array: the design pads M,N to 512 internally; useful ops are the real 2·M·N·K. Default is one fresh
//! context per iteration (every iteration verified); `--reuse-ctx` creates one context + buffers and resubmits
//! `--iters` times (every submit verified); `--loop SECONDS` implies reuse, resubmits until the wall clock
//! elapses (at least one submit), then runs one more submit on a freshly 0xA5-poisoned output; the first and that
//! final result are verified exactly (intermediate submits are not poisoned or checked), and it prints submits,
//! useful ops, mean TOPS (useful ops / summed submit→completion time) and host bytes moved.
//! `--gap-us G` (u64 microseconds, default 0, only with `--loop`, else exit 2 before any hardware access) sleeps G us after
//! each completed submit of the loop (including the final one; never after a failed submit), outside the timed busy region
//! (so wall includes the sleep, busy does not); the summary prints `gap_us=G` and `npu_dram_GBps` = (host bytes read + written) / wall / 1e9.
//! `--variant` implies array mode (and may be combined with `--array`); `--array` alone is V6. V1..V5 are
//! static-CDO bisect designs and only support a fresh context per iteration (`--reuse-ctx`/`--loop` are
//! rejected); V6 and V6b keep the TXN restarts and also accept reuse.
//! `--core` (array only, default Fast) selects the core program: `Fast` is the deployed kernel, `Serial` the
//! serialised-core bisect program, `LockOnly` a lock/DMA-only program that never computes: it acquires and
//! releases every A/B chunk then C-full and writes the first 64 C words of each physical tile for each wave as
//! 0xC0DE0000 + wave*0x100 + word index (rest of C unspecified). With LockOnly, success is that pattern in the
//! first 64 little-endian u32 words of EVERY physical packed-C range (`output_tile_ranges`, not unpacked GEMM
//! coordinates) — the GEMM reference is not used. LockOnly inputs are zero buffers (no random matrices, CPU
//! reference or packing) and it reports useful ops 0, hence 0 TOPS: it does no GEMM. `FastSlowCtl` is Fast's exact
//! compute with serialised control code (7-NOP lock/branch gaps, no delay-slot work). `ClockProbe` is LockOnly plus,
//! once per tile before the C-full release, a delay loop of exactly `--probe-iters N` x `clock_probe_cycles_per_iter()`
//! issue cycles (N required to be a u32, default 0, only with `--core ClockProbe`, else exit 2); each submit line then
//! ends with `probe_iters=N probe_cycles=N*cycles_per_iter*tiles` so the submit-time slope vs probe_cycles gives the AIE clock.
//! Every GEMM wait uses `--timeout-ms` (default 3000). When a wait does not complete (or a submit fails), BEFORE
//! the hardware context is dropped the AIE column status is queried, written raw (full API buffer) to
//! `aie-status-<tag>-<iter>.bin` in the working directory and decoded (see `decode_status`), and the C-buffer
//! ranges that changed from the 0xA5 poison are printed per output tile (array modes). `--dump-status` does
//! the same after EVERY wait, completed or not (every iteration/submit of fresh, reused and `--loop` paths, and
//! the single-core path), so a `--loop` run writes one .bin per submit; a failing wait is still dumped only once.
//! `<tag>` is `single`, `array`, or the variant name; a non-default `--core` is appended (`V2-LockOnly`) so runs
//! sharing a directory do not collide.
//! `--variant V8` is the pair A-sharing whole-array design (`gemm_array::design_v8`, dynamic TXN, context reusable,
//! `--reuse-ctx`/`--loop` work like V6). `--epi int8` (default) stores int8 C tiles produced by the on-core SRS
//! epilogue `clamp(floor(c / 2^S), -128, 127)` with `--shift S` (default 12, at most 31); `--epi i32` stores the
//! exact int32 C (`--shift` is then rejected). `--core` for V8 is `Fast` (default) or `FastSlowCtl` (the slow
//! control discipline `Control::Slow`); other cores, and `--epi`/`--shift` without `--variant V8`/`V9`, exit 2. Every
//! submit is compared with the CPU reference that applies the same epilogue (`ArrayDesign::reference`).
//! `--variant V9` is the resident-B nw-outer design (`gemm_array::design_v9`, same pair core as V8, dynamic TXN,
//! context reusable); it takes the same `--shift`/`--core`/`--ctl-probe`/`--reuse-ctx`/`--loop` controls as V8 but is
//! Int8-epilogue only: `--epi i32` exits 2.
//! `--variant G80` is the G80 design (`gemm_g80::design_g80`, N exactly 1280, K a multiple of 64 up to 2560;
//! same `--core`/`--ctl-probe`/`--a-from-hip`/`--reuse-ctx`/`--loop` controls as V8). It takes `--epi int8` (default,
//! `--shift S`) or `--epi i32` (exact int32 C) or `--epi bf16` (integer exact-product, single RNE).
//! I32/BF16 reject `--shift`. BF16 uses deterministic grid scales and compares every checked output
//! against `w8a8_epi::bf16_direct`. Row-major additionally checks zero padded rows and untouched poison outside the panel window.
//! `--variant V10` is the mw-outer both-operands-resident design (`gemm_array::design_v10`, same pair core and
//! compact C as V9, dynamic TXN, context reusable): the whole B segment (`NW*kc*4096` words per row) stays resident in
//! the memtile and is replayed for every M-wave while A is streamed per M-wave (wave order `w = mw*NW + nw`), so each
//! of A and B is read from DDR once per submit (the design line's `host bytes/submit read` is that model byte count,
//! e.g. 4096 2560 640 reads 4.0625 MiB). Same controls as V9 (`--shift`/`--core Fast|FastSlowCtl`/`--ctl-probe`/
//! `--a-from-hip`/`--reuse-ctx`/`--loop`), Int8 epilogue only (`--epi i32` exits 2). The design asserts
//! `kc*(2+NW) <= 112`, `MW <= 16` and `NW <= 8`. The ESTIMATE line is a model, not a hardware measurement.
//! `--variant V10 --b-prefix` (bare flag, V10 only; any other variant exits 2) builds V10 through
//! `gemm_array::design_v10_prefix(m, n, k, epi, ctl)`; the tag gains `-bprefix`. Retained prefix: B is filled by BD4/5 and
//! guarded30/31 with groups <= 64, B_FULL is released per chunk (+1/-1), and above 63 chunks a B_EMPTY63 credit window
//! applies; after the guarded first pass, lockfree whole-BD36 serves the remaining MW-1 passes (only when MW > 1) and is
//! rewritten on every submit. Works with `--shift`/`--core Fast|FastSlowCtl`/`--ctl-probe`/`--a-from-hip`/`--reuse-ctx`/
//! `--loop`; without it the old `design_v10` dispatch is unchanged. With the flag the ESTIMATE line is the baseline V10
//! whole-B startup model only (a note says so); it does not model the prefix descriptor overlap, so only measured
//! hardware times say anything about that design.
//! `--a-from-hip` (V8, V9 or V10, anything else exits 2; shared helper `railgun::npu::hip_runtime`): the packed A operand (arg0) is put
//! in HIP VMM host-backed (system/GTT) physical memory (`hipMemCreate`, `hipMemAllocationTypePinned`,
//! `hipMemLocationTypeHost`; `hipHostMalloc` is not dma-buf exportable and device mallocs are VRAM), written by the
//! GPU (`hipMemcpyHtoD`, then `hipDeviceSynchronize`), exported as a dma-buf (`hipMemGetHandleForAddressRange`) and
//! imported with `Device::import_dmabuf` as the NPU's arg0 instead of a normal SHMEM BO copy; B and C are the
//! normal SHMEM BOs and the CPU reference is unchanged. `libamdhip64.so` is dlopen'ed at run time, nothing is
//! linked. The NPU device is opened first; no HIP call happens before that succeeds. The HIP allocation, fd and
//! library stay alive until the contexts and BOs are dropped; fresh-context iterations re-import the same fd every
//! iteration, `--reuse-ctx`/`--loop` import once. The host never writes the imported A mapping (it is only
//! cache-flushed). Each stage prints `a-from-hip: ...` on success; a failure prints the stage plus the HIP status
//! and `hipGetErrorString` (or the numeric OS errno for open/import failures), ownership is released by normal
//! drops and the run ends FAIL (exit 1, no panic). The default `--core` stays `Fast`; pass `--core FastSlowCtl`
//! explicitly for that program.
//! `--iu4-epochs` (with `--variant V8 --epi i32`, core Fast or FastSlowCtl): M = tokens, N = weight rows, K a multiple
//! of 256. Deterministic native hipfire IU4 operands (symmetric MQ4G256V2 weights x `block_i4_128` activations,
//! `peacemaker_npu::kernels::iu4`) run on `iu4::EpochGemm` (V8 i32, K = 128 per wave, one M-wave block per K128 epoch), and every
//! verified submit is compared with the CPU reference of the GPU's per-epoch int32 partials `C_e` (all `[e][t][r]`).
//! Useful ops are the real 2·M·N·K; `epochs * ceil(M/512) * ceil(N/512) <= 256` waves per submit.
//! `--fold-contract ief15` (`peacemaker_npu::kernels::iu4_ief15::Ief15Gemm`, fold-contract §3 / §8 M0; AxQOS 0 and the vendor BD
//! word 5 defaults, so `--burst/--axcache/--axqos/--memory/--a-from-hip/--variant/--array/--core/--epi/--shift/--ctl-probe/
//! --iu4-epochs` exit 2): TOKENS x FEATURES x K, deterministic native QT44 / A4 operands (`iu4::random_operands`, extremes
//! on) with the production sidecar encoders (`encode_weight_scales`, `encode_act_scales`), packed with `pack_in` and checked
//! byte for byte against the CPU oracle `reference` (LE f32 `[token][feature]`; the check prints the mismatch count and the
//! first mismatch). Default: a fresh hardware context per iteration, every iteration verified. `--reuse-ctx` (implied by
//! `--loop` and `--lean`) resubmits on one context: only the first and a final submit on a freshly 0xA5-poisoned output are
//! verified (the final one is the last `--iters` submit, or the one after the `--loop SECS` deadline). `--lean` makes every
//! submit after the first full one use the lean TXN (`ArrayDesign::lean_insts`, valid only directly after a completed submit
//! of the same design). `--switch-with T2,F2,K2` builds a second IEF15 design (its own operands, its own hardware context,
//! both contexts alive) and measures three modes of `--iters` rounds each: repeat A, repeat B and alternate A,B, printing
//! mean us/submit (busy: submit -> completion) per mode and the alternate-vs-repeat difference (the context switch cost);
//! the first submit and a final poisoned submit of each design are verified. `--switch-with` rejects `--loop` and `--lean`
//! (a lean TXN is only valid directly after a completed submit of the same design). Useful ops are the real 2*T*F*K; busy
//! TOPS is useful ops over summed submit -> completion time, wall TOPS over the wall clock including host verification.
use npu_tools::util::lcg;
use peacemaker_npu::kernels::gemm_array::{
    design_v10, design_v10_prefix, design_v8, design_v9, design_variant_probe, Variant,
};
use peacemaker_npu::kernels::gemm_core::{
    clock_probe_cycles_per_iter, Control, CoreVariant, Epilogue, Probe,
};
use peacemaker_npu::kernels::gemm_i8::{cpu_reference, design, ArgKind};
use peacemaker_npu::kernels::iu4::{self, EpochGemm};
use peacemaker_npu::kernels::iu4_ief15::{self, Ief15Gemm};
use railgun::npu::hip_runtime::{HipDmabufA, HipMemoryKind};
use railgun::npu::{Bo, Device, HwCtx, ERT_STATE_COMPLETED};
use std::ops::Range;

/// `--name value` parser; `None` when the flag is absent.
fn opt_str<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter().position(|a| a == name).map(|i| {
        args.get(i + 1)
            .unwrap_or_else(|| panic!("{name} needs a value"))
            .as_str()
    })
}

/// Wait timeout for every GEMM completion wait (`--timeout-ms`, default 3000).
fn timeout_ms(args: &[String]) -> u64 {
    let t: u64 = opt_str(args, "--timeout-ms")
        .map(|s| s.parse().expect("--timeout-ms N"))
        .unwrap_or(3000);
    assert!(t >= 1, "--timeout-ms must be >= 1");
    t
}

/// Idle gap after every completed submit of a `--loop` run (`--gap-us G` microseconds, default 0). Exit 2 when given
/// without `--loop` or when G is not a u64; called from `main` before any hardware access.
fn gap_us(args: &[String]) -> u64 {
    let Some(s) = opt_str(args, "--gap-us") else {
        return 0;
    };
    if opt_str(args, "--loop").is_none() {
        eprintln!("--gap-us is only valid with --loop SECONDS");
        std::process::exit(2);
    }
    s.parse().unwrap_or_else(|_| {
        eprintln!("--gap-us must be a u64, got {s:?}");
        std::process::exit(2)
    })
}

/// Bytes of firmware column status per column (MSG_OP_QUERY_COL_STATUS, aie_status_version 1.1, NPU5).
const COL_STATUS_BYTES: usize = 504;

/// Core status register bit names (aie-rt xaie2pgbl_params.h CORE_MODULE_CORE_STATUS); bit index = position.
const CORE_STATUS_BITS: [&str; 22] = [
    "Enable",
    "Reset",
    "MemStall_S",
    "MemStall_W",
    "MemStall_N",
    "MemStall_E",
    "LockStall_S",
    "LockStall_W",
    "LockStall_N",
    "LockStall_E",
    "StreamStall_SS0",
    "b11",
    "StreamStall_MS0",
    "b13",
    "CascadeStall_SCD",
    "CascadeStall_MCD",
    "DebugHalt",
    "ECC_ErrorStall",
    "ECC_ScrubStall",
    "ErrorHalt",
    "CoreDone",
    "ProcBusStall",
];

/// DMA channel status bit names (aie-rt xaie2pgbl_params.h MEMORY_MODULE_DMA_S2MM_STATUS_0) as (bit, name).
const DMA_STATUS_BITS: [(u32, &str); 12] = [
    (2, "LockAcqStall"),
    (3, "LockRelStall"),
    (4, "StreamStarved"),
    (5, "TCT/CountFull"),
    (8, "ErrLockUnavail"),
    (9, "ErrDMUnavail"),
    (10, "ErrBDUnavail"),
    (11, "ErrBDInvalid"),
    (12, "ErrFoTLength"),
    (13, "ErrFoTBDs"),
    (18, "TaskQOverflow"),
    (19, "Running"),
];

fn core_status_str(v: u32) -> String {
    let mut names: Vec<String> = CORE_STATUS_BITS
        .iter()
        .enumerate()
        .filter(|(i, _)| v >> i & 1 == 1)
        .map(|(_, n)| n.to_string())
        .collect();
    let unlabeled = v & !((1u32 << CORE_STATUS_BITS.len()) - 1);
    if unlabeled != 0 {
        names.push(format!("unlabeled={unlabeled:#x}"));
    }
    if names.is_empty() {
        "0".to_string()
    } else {
        names.join("|")
    }
}

/// `0` or `STATE bd=B q=Q FLAG,FLAG[ other=0x..]`; STATE is bits 1:0, q bits 22:20, bd bits 29:24.
fn dma_str(v: u32) -> String {
    if v == 0 {
        return "0".to_string();
    }
    let state = ["IDLE", "STARTING", "RUNNING", "?"][(v & 3) as usize];
    let flags: Vec<&str> = DMA_STATUS_BITS
        .iter()
        .filter(|(b, _)| v >> b & 1 == 1)
        .map(|&(_, n)| n)
        .collect();
    let known = DMA_STATUS_BITS
        .iter()
        .fold(0x3u32 | 7 << 20 | 0x3f << 24, |m, &(b, _)| m | 1 << b);
    let other = v & !known;
    let mut s = format!("{state} bd={} q={}", v >> 24 & 0x3f, v >> 20 & 7);
    if !flags.is_empty() {
        s.push(' ');
        s.push_str(&flags.join(","));
    }
    if other != 0 {
        s.push_str(&format!(" other={other:#x}"));
    }
    s
}

fn le_words<const N: usize>(blk: &[u8], o: usize) -> [u32; N] {
    std::array::from_fn(|i| u32::from_le_bytes(blk[o + 4 * i..o + 4 * i + 4].try_into().unwrap()))
}

fn hex_words(w: &[u32]) -> String {
    w.iter()
        .map(|x| format!("{x:08x}"))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Lock counter shown as the 6-bit value; a nonzero upper part is shown too (raw byte).
fn lock_val(x: u8) -> String {
    if x & !0x3f == 0 {
        x.to_string()
    } else {
        format!("{}(raw {x:#04x})", x & 0x3f)
    }
}

/// Decode a DRM_AMDXDNA_QUERY_AIE_STATUS buffer (firmware MSG_OP_QUERY_COL_STATUS, aie_status_version 1.1,
/// col_size 504). Authoritative decoder: npu_tools::aie_status. One 504 B block per SET bit of the `cols_filled`
/// bitmap, densely packed in ascending column order (XRT core/common/info_aie2.cpp). Per block:
///   0..320   4 core rows (physical rows 2..5) x 80 B: dma u32[4] (s2mm0, mm2s0, s2mm1, mm2s1), core_events u32[4],
///            mem_events u32[4], core_status, pc, sp, lr, u8 lock[16]
///   320..456 memtile (row 1) 136 B: dma u32[12] (6 channels x {s2mm, mm2s}), events u32[6], u8 lock[64]
///   456..504 shim (row 0) 48 B: dma u32[4] (s2mm0, mm2s0, s2mm1, mm2s1), events u32[4], u8 lock[16]
/// Event words are printed as raw hex (no event naming). Blocks that do not fit in `buf` are reported as
/// truncated, never guessed; bytes past the last decoded block are reported as not interpreted.
fn decode_status(buf: &[u8], bitmap: u32) -> Vec<String> {
    let cols: Vec<u32> = (0..32u32).filter(|&c| bitmap >> c & 1 == 1).collect();
    let mut out = Vec::new();
    let mut decoded = 0usize;
    for (i, &col) in cols.iter().enumerate() {
        let Some(blk) = buf.get(i * COL_STATUS_BYTES..(i + 1) * COL_STATUS_BYTES) else {
            let have = buf.len().saturating_sub(i * COL_STATUS_BYTES);
            out.push(format!(
                "TRUNCATED: column {col} (packed block {i}) needs bytes {}..{} but the buffer holds {} B ({have} B of this block); columns {:?} NOT decoded",
                i * COL_STATUS_BYTES,
                (i + 1) * COL_STATUS_BYTES,
                buf.len(),
                &cols[i..]
            ));
            break;
        };
        decoded += 1;
        out.push(format!(
            "column {col} (packed block {i}, bytes {}..{}):",
            i * COL_STATUS_BYTES,
            (i + 1) * COL_STATUS_BYTES
        ));
        let mut held_reset = Vec::new();
        for r in 0..4usize {
            let o = r * 80;
            let d: [u32; 4] = le_words(blk, o);
            let ce: [u32; 4] = le_words(blk, o + 16);
            let me: [u32; 4] = le_words(blk, o + 32);
            let [st, pc, sp, lr]: [u32; 4] = le_words(blk, o + 48);
            let locks = &blk[o + 64..o + 80];
            if st == 2
                && d.iter().all(|&x| x == 0)
                && ce.iter().all(|&x| x == 0)
                && me.iter().all(|&x| x == 0)
                && locks.iter().all(|&x| x == 0)
            {
                held_reset.push(format!("({col},{})", r + 2));
                continue;
            }
            out.push(format!(
                "core ({col},{}) status={st:#x} [{}] pc={pc:#x} sp={sp:#x} lr={lr:#x}",
                r + 2,
                core_status_str(st)
            ));
            out.push(format!(
                "   s2mm0 {} | mm2s0 {} | s2mm1 {} | mm2s1 {}",
                dma_str(d[0]),
                dma_str(d[1]),
                dma_str(d[2]),
                dma_str(d[3])
            ));
            out.push(format!(
                "   locks {}  core_ev {}  mem_ev {}",
                locks
                    .iter()
                    .map(|&x| lock_val(x))
                    .collect::<Vec<_>>()
                    .join(" "),
                hex_words(&ce),
                hex_words(&me)
            ));
        }
        if !held_reset.is_empty() {
            out.push(format!(
                "cores {} status=0x2 [Reset], DMA/events/locks all zero (not expanded)",
                held_reset.join(" ")
            ));
        }
        let d: [u32; 12] = le_words(blk, 320);
        let ev: [u32; 6] = le_words(blk, 368);
        let locks = &blk[392..456];
        if d.iter().any(|&x| x != 0) || ev.iter().any(|&x| x != 0) || locks.iter().any(|&x| x != 0)
        {
            out.push(format!("mem  ({col},1)"));
            for ch in 0..6 {
                if d[2 * ch] != 0 || d[2 * ch + 1] != 0 {
                    out.push(format!(
                        "   ch{ch}: s2mm {} | mm2s {}",
                        dma_str(d[2 * ch]),
                        dma_str(d[2 * ch + 1])
                    ));
                }
            }
            out.push(format!("   events {}", hex_words(&ev)));
            out.push(format!(
                "   locks {}",
                locks
                    .iter()
                    .enumerate()
                    .filter(|(_, &x)| x != 0)
                    .map(|(j, &x)| format!("{j}:{}", lock_val(x)))
                    .collect::<Vec<_>>()
                    .join(" ")
            ));
        } else {
            out.push(format!("mem  ({col},1) all zero"));
        }
        let d: [u32; 4] = le_words(blk, 456);
        let ev: [u32; 4] = le_words(blk, 472);
        let locks = &blk[488..504];
        if d.iter().any(|&x| x != 0) || ev.iter().any(|&x| x != 0) || locks.iter().any(|&x| x != 0)
        {
            out.push(format!(
                "shim ({col},0) s2mm0 {} | mm2s0 {} | s2mm1 {} | mm2s1 {}  events {}  locks {}",
                dma_str(d[0]),
                dma_str(d[1]),
                dma_str(d[2]),
                dma_str(d[3]),
                hex_words(&ev),
                locks
                    .iter()
                    .map(|&x| lock_val(x))
                    .collect::<Vec<_>>()
                    .join(" ")
            ));
        } else {
            out.push(format!("shim ({col},0) all zero"));
        }
    }
    let used = decoded * COL_STATUS_BYTES;
    if decoded == cols.len() && buf.len() > used {
        out.push(format!(
            "{} trailing B beyond the {} decoded block(s) not interpreted (kept in the .bin)",
            buf.len() - used,
            decoded
        ));
    }
    out
}

/// Query the firmware column status, write the ENTIRE API buffer raw to `aie-status-<tag>-<iter>.bin` and print the
/// decode (`decode_status`). The buffer capacity is the firmware metadata `cols * col_size`; the driver's
/// `cols_filled` is a column BITMAP and the true response size is not exposed, so nothing is truncated. A failed
/// query, an empty bitmap or invalid metadata are reported as INVALID (the attempted buffer is still kept raw
/// where one exists), and a `col_size` other than 504 is saved raw but not decoded. Never panics; any failure only
/// prints. Call BEFORE the hardware context is dropped. `why` is a short context label for the log line.
fn dump_aie_status(dev: &Device, tag: &str, iter: usize, why: &str) {
    let (meta_cols, col_size) = (dev.meta.cols as usize, dev.meta.col_size as usize);
    let capacity = meta_cols * col_size;
    let head = format!("aie-status[{tag} iter {iter}, {why}]");
    let path = format!("aie-status-{tag}-{iter}.bin");
    let save = |bytes: &[u8]| match std::fs::write(&path, bytes) {
        Ok(()) => format!("{} B written raw to {path}", bytes.len()),
        Err(e) => format!("{} B; writing {path} failed: {e}", bytes.len()),
    };
    if capacity == 0 {
        println!("{head}: INVALID firmware metadata (cols {meta_cols} x col_size {col_size} = 0 B capacity); status query not attempted (no buffer to give the driver); empty attempted buffer kept raw ({}), NOT decoded", save(&[]));
        return;
    }
    let mut buf = vec![0u8; capacity];
    let bitmap = match dev.aie_status(&mut buf) {
        Ok(b) => b,
        Err(e) => {
            println!("{head}: query failed: {e}; INVALID/uninitialized capture, attempted API buffer kept raw ({}), NOT decoded", save(&buf));
            return;
        }
    };
    let cols: Vec<u32> = (0..32u32).filter(|&c| (bitmap >> c) & 1 == 1).collect();
    if cols.is_empty() {
        println!("{head}: empty column bitmap {bitmap:#x}; INVALID/uninitialized capture, attempted API buffer kept raw ({}), NOT decoded", save(&buf));
        return;
    }
    println!(
        "{head}: column bitmap {bitmap:#x} (set bits {cols:?}); capacity {capacity} B = meta cols {meta_cols} x col_size {col_size}; response size not exposed; {}",
        save(&buf)
    );
    if col_size != COL_STATUS_BYTES {
        println!("  NOT decoded: meta col_size {col_size} != {COL_STATUS_BYTES}, the only layout defined here; raw buffer only");
        return;
    }
    for line in decode_status(&buf, bitmap) {
        println!("  {line}");
    }
}

/// (changed byte count, first changed offset) versus the all-0xA5 poison.
fn changed_vs_poison(buf: &[u8]) -> (usize, Option<usize>) {
    let mut n = 0;
    let mut first = None;
    for (i, &b) in buf.iter().enumerate() {
        if b != 0xA5 {
            n += 1;
            first.get_or_insert(i);
        }
    }
    (n, first)
}

/// Print whether the C buffer changed from the 0xA5 poison, overall and per output tile (col,row).
fn report_output(out: &[u8], tiles: &[((u32, u32), Vec<Range<usize>>)]) {
    let (n, first) = changed_vs_poison(out);
    println!(
        "C buffer: {n}/{} bytes differ from 0xA5 poison{}",
        out.len(),
        first
            .map(|f| format!(" (first at byte {f})"))
            .unwrap_or_default()
    );
    let mut delivered = Vec::new();
    for &((col, row), ref ranges) in tiles {
        let (mut changed, mut total, mut bad_range) = (0usize, 0usize, false);
        let mut first_changed = None;
        for r in ranges {
            match out.get(r.clone()) {
                Some(s) => {
                    let (c, f) = changed_vs_poison(s);
                    changed += c;
                    total += s.len();
                    if first_changed.is_none() {
                        first_changed = f.map(|f| r.start + f);
                    }
                }
                None => bad_range = true,
            }
        }
        if changed > 0 {
            delivered.push((col, row));
        }
        println!(
            "  C tile (col {col}, row {row}): {changed}/{total} bytes changed over {} range(s){}{}",
            ranges.len(),
            first_changed
                .map(|f| format!(", first changed byte {f}"))
                .unwrap_or_default(),
            if bad_range {
                " [range outside buffer]"
            } else {
                ""
            }
        );
    }
    println!(
        "  tiles with any delivered data: {}/{} {:?}",
        delivered.len(),
        tiles.len(),
        delivered
    );
}

/// Words of C the LockOnly program writes per physical tile and wave.
const LOCK_WORDS: usize = 64;

/// The LockOnly pattern word: 0xC0DE0000 + zero-based wave * 0x100 + word index.
fn lock_word(wave: usize, i: usize) -> u32 {
    0xC0DE_0000 + wave as u32 * 0x100 + i as u32
}

/// Per physical tile result of the LockOnly pattern check over every wave range.
struct TileLock {
    tile: (u32, u32),
    bad: usize,
    total: usize,
    /// (wave, word index, got (None = outside buffer/range), want) of the first mismatching word.
    first_bad: Option<(usize, usize, Option<u32>, u32)>,
}

/// Check the first [`LOCK_WORDS`] little-endian u32 words of EVERY packed-C range of EVERY tile against
/// [`lock_word`] (range index = wave).
fn lock_pattern<'a>(
    out: &'a [u8],
    tiles: &'a [((u32, u32), Vec<Range<usize>>)],
) -> impl Iterator<Item = TileLock> + 'a {
    tiles.iter().map(move |&(tile, ref ranges)| {
        let mut t = TileLock {
            tile,
            bad: 0,
            total: 0,
            first_bad: None,
        };
        for (wave, r) in ranges.iter().enumerate() {
            for i in 0..LOCK_WORDS {
                let want = lock_word(wave, i);
                let got = (4 * i + 4 <= r.len())
                    .then(|| out.get(r.start + 4 * i..r.start + 4 * i + 4))
                    .flatten()
                    .map(|b| u32::from_le_bytes(b.try_into().unwrap()));
                t.total += 1;
                if got != Some(want) {
                    t.bad += 1;
                    t.first_bad.get_or_insert((wave, i, got, want));
                }
            }
        }
        t
    })
}

/// Print the LockOnly pattern result per physical tile (col,row).
fn report_lock_pattern(out: &[u8], tiles: &[((u32, u32), Vec<Range<usize>>)]) {
    let res: Vec<TileLock> = lock_pattern(out, tiles).collect();
    for t in &res {
        let first = t.first_bad.map(|(w, i, got, want)| {
            format!(
                ", first mismatch wave {w} word {i}: got {} want {want:#010x}",
                got.map_or("<outside buffer>".to_string(), |g| format!("{g:#010x}"))
            )
        });
        println!(
            "  lock pattern tile (col {}, row {}): {}/{} words match{}",
            t.tile.0,
            t.tile.1,
            t.total - t.bad,
            t.total,
            first.unwrap_or_default()
        );
    }
    let ok: Vec<(u32, u32)> = res.iter().filter(|t| t.bad == 0).map(|t| t.tile).collect();
    println!(
        "  tiles with the full lock pattern: {}/{} {:?}",
        ok.len(),
        res.len(),
        ok
    );
}

fn run_array(args: &[String], m: usize, n: usize, k: usize) -> bool {
    let loop_secs: Option<f64> =
        opt_str(args, "--loop").map(|s| s.parse().expect("--loop SECONDS"));
    if let Some(s) = loop_secs {
        assert!(
            s.is_finite() && s > 0.0,
            "--loop SECONDS must be finite and > 0"
        );
    }
    let gap_us = gap_us(args);
    let iters: usize = opt_str(args, "--iters")
        .map(|s| s.parse().expect("--iters I"))
        .unwrap_or(3);
    assert!(iters >= 1, "--iters must be >= 1");
    let reuse = loop_secs.is_some() || args.iter().any(|a| a == "--reuse-ctx");
    let timeout_ms = timeout_ms(args);
    let variant: Option<Variant> = opt_str(args, "--variant").map(|s| {
        s.parse::<Variant>().unwrap_or_else(|_| {
            eprintln!("--variant must be one of V1 V2 V3 V4 V5 V6 V6b V8 V9 V10 G80, got {s:?}");
            std::process::exit(2)
        })
    });
    let core: CoreVariant = match opt_str(args, "--core")
        .unwrap_or("Fast")
        .to_ascii_lowercase()
        .as_str()
    {
        "fast" => CoreVariant::Fast,
        "serial" => CoreVariant::Serial,
        "lockonly" => CoreVariant::LockOnly,
        "fastslowctl" => CoreVariant::FastSlowCtl,
        "clockprobe" => CoreVariant::ClockProbe,
        other => {
            eprintln!(
                "--core must be one of Fast Serial LockOnly FastSlowCtl ClockProbe, got {other:?}"
            );
            std::process::exit(2)
        }
    };
    let v8 = variant == Some(Variant::V8);
    let v9 = variant == Some(Variant::V9);
    let v10 = variant == Some(Variant::V10);
    let g80 = variant == Some(Variant::G80);
    let pair = v8 || v9 || v10 || g80;
    let epi: Option<Epilogue> = if pair {
        let shift: Option<u8> = opt_str(args, "--shift").map(|s| match s.parse::<u8>() {
            Ok(v) if v <= 31 => v,
            _ => {
                eprintln!("--shift must be an integer in 0..=31, got {s:?}");
                std::process::exit(2)
            }
        });
        match opt_str(args, "--epi")
            .unwrap_or("int8")
            .to_ascii_lowercase()
            .as_str()
        {
            "int8" => Some(Epilogue::Int8 {
                shift: shift.unwrap_or(12),
            }),
            "i32" if v9 || v10 => {
                eprintln!("--variant {} supports --epi int8 only (the int32 epilogue is not implemented for V9/V10)", variant.as_ref().map_or("V9", |v| v.name()));
                std::process::exit(2)
            }
            "i32" if shift.is_none() => Some(Epilogue::I32),
            "bf16" if g80 && shift.is_none() => Some(Epilogue::Bf16),
            "bf16" => {
                eprintln!("--epi bf16 needs --variant G80 and no --shift");
                std::process::exit(2)
            }
            "i32" => {
                eprintln!("--shift needs --epi int8");
                std::process::exit(2)
            }
            other => {
                eprintln!("--epi must be i32, int8 or (G80 only) bf16, got {other:?}");
                std::process::exit(2)
            }
        }
    } else if opt_str(args, "--epi").is_some() || opt_str(args, "--shift").is_some() {
        eprintln!("--epi/--shift are only valid with --variant V8, V9, V10 or G80");
        std::process::exit(2)
    } else {
        None
    };
    // `--n512` (G80 BF16 only): the N=512 / K<=6336 resident-B out-projection design (`gemm_g80_n512`, 512-row waves).
    let n512 = args.iter().any(|a| a == "--n512");
    if n512 && !(g80 && epi == Some(Epilogue::Bf16)) {
        eprintln!("--n512 needs --variant G80 --epi bf16");
        std::process::exit(2)
    }
    let (wave_m, wave_n) = if n512 { (512, 512) } else { (256, 1280) };
    let rowmajor = opt_str(args, "--rowmajor-pitch").map(|p| {
        let parse = |name: &str, value: &str| value.parse::<usize>().unwrap_or_else(|_| {
            eprintln!("{name} needs an unsigned integer"); std::process::exit(2)
        });
        let pitch = parse("--rowmajor-pitch", p);
        let col0 = parse("--rowmajor-col0", opt_str(args, "--rowmajor-col0").unwrap_or("0"));
        if epi != Some(Epilogue::Bf16) || pitch % 2 != 0 || col0 % 2 != 0 ||
            pitch / 2 > 1 << 20 || col0.checked_add(wave_n).is_none_or(|end| end > pitch) ||
            m.div_ceil(wave_m).checked_mul(wave_m).and_then(|r| r.checked_mul(pitch)).and_then(|n| n.checked_mul(2)).is_none() {
            eprintln!("row-major needs G80 BF16, even pitch/col0, col0+{wave_n} <= pitch <= {}, and a representable arena", 2 << 20);
            std::process::exit(2)
        }
        peacemaker_npu::kernels::gemm_g80::Bf16Layout::RowMajor { pitch_elems: pitch, col0_elems: col0 }
    });
    if rowmajor.is_none() && opt_str(args, "--rowmajor-col0").is_some() {
        eprintln!("--rowmajor-col0 requires --rowmajor-pitch");
        std::process::exit(2)
    }
    let a_from_hip = args.iter().any(|a| a == "--a-from-hip");
    let memory = opt_str(args, "--memory")
        .map(|s| {
            if !pair || a_from_hip {
                eprintln!("--memory needs V8/V9/V10 and cannot be combined with --a-from-hip");
                std::process::exit(2)
            }
            if s == "native" {
                None
            } else {
                Some(HipMemoryKind::parse(s).unwrap_or_else(|e| {
                    eprintln!("{e}");
                    std::process::exit(2)
                }))
            }
        })
        .flatten();
    if a_from_hip && !pair {
        eprintln!("--a-from-hip is only valid with --variant V8, V9, V10 or G80");
        std::process::exit(2)
    }
    // `--b-prefix` (V10 only, bare flag) selects `design_v10_prefix`. It is an explicit opt-in: without the flag the
    // old `design_v10` is used. It is rejected on any other variant (never silently ignored).
    let b_prefix = args.iter().any(|a| a == "--b-prefix");
    if b_prefix && !v10 {
        eprintln!(
            "--b-prefix is only valid with --variant V10, got {}",
            variant
                .as_ref()
                .map_or("no --variant (--array is V6)", |v| v.name())
        );
        std::process::exit(2)
    }
    if pair && !matches!(core, CoreVariant::Fast | CoreVariant::FastSlowCtl) {
        eprintln!(
            "--variant {} supports --core Fast or FastSlowCtl only",
            variant.as_ref().map_or("V8", |v| v.name())
        );
        std::process::exit(2)
    }
    if g80 {
        let (max_k, wave_n) = if n512 {
            (peacemaker_npu::kernels::gemm_g80_n512::MAX_KC * 64, 512)
        } else {
            (
                peacemaker_npu::kernels::gemm_core_g80::MAX_KC * 64,
                Variant::G80.shape().1,
            )
        };
        if n != wave_n || k % 64 != 0 || k == 0 || k > max_k {
            eprintln!("--variant G80{} needs N = {wave_n} and K a multiple of 64 in 64..={max_k}, got N={n} K={k}", if n512 { " --n512" } else { "" });
            std::process::exit(2)
        }
    }
    // `--ctl-probe knob,knob,...` (V8/V9/V10/G80 only): hardware-bisect control knobs / timing-only probes (`gemm_core::Probe`).
    let ctl_probe: Option<Probe> = opt_str(args, "--ctl-probe").map(|s| {
        if !pair || core != CoreVariant::Fast {
            eprintln!("--ctl-probe needs --variant V8, V9, V10 or G80 and --core Fast");
            std::process::exit(2)
        }
        Probe::parse(s).unwrap_or_else(|e| {
            eprintln!("--ctl-probe: {e}");
            std::process::exit(2)
        })
    });
    let timing_only = ctl_probe.is_some_and(Probe::timing_only);
    if epi == Some(Epilogue::Bf16) && ctl_probe.is_some_and(|p| p.timing_only() || p.layout != 0) {
        eprintln!("--epi bf16 requires ordinary GEMM compute and layout 0");
        std::process::exit(2)
    }
    let probe_iters: u32 = match opt_str(args, "--probe-iters") {
        None => 0,
        Some(_) if core != CoreVariant::ClockProbe => {
            eprintln!("--probe-iters is only valid with --core ClockProbe");
            std::process::exit(2)
        }
        Some(s) => s.parse().unwrap_or_else(|_| {
            eprintln!("--probe-iters must be a u32, got {s:?}");
            std::process::exit(2)
        }),
    };
    let core_name = match core {
        CoreVariant::Fast => "Fast",
        CoreVariant::Serial => "Serial",
        CoreVariant::LockOnly => "LockOnly",
        CoreVariant::FastSlowCtl => "FastSlowCtl",
        CoreVariant::ClockProbe => "ClockProbe",
    };
    let lock_only = matches!(core, CoreVariant::LockOnly | CoreVariant::ClockProbe);
    let dump = args.iter().any(|a| a == "--dump-status");
    let base_tag: &str = variant.as_ref().map_or("array", |v| v.name());
    let tag_owned = match ctl_probe {
        Some(p) => format!(
            "{base_tag}-probe{}",
            if p.timing_only() { "-timing" } else { "" }
        ),
        None if matches!(core, CoreVariant::Fast) => base_tag.to_string(),
        None => format!("{base_tag}-{core_name}"),
    };
    let tag_owned = if a_from_hip {
        format!("{tag_owned}-hipA")
    } else {
        tag_owned
    };
    let tag_owned = if b_prefix {
        format!("{tag_owned}-bprefix")
    } else {
        tag_owned
    };
    let tag: &str = &tag_owned;
    if reuse
        && !matches!(
            base_tag,
            "array" | "V6" | "V6b" | "V8" | "V9" | "V10" | "G80"
        )
    {
        eprintln!("--variant {base_tag} is a static-CDO design: it needs a fresh hardware context per iteration; --reuse-ctx/--loop are not supported");
        std::process::exit(2);
    }

    // `--array` alone is the deployed V6; with the default Fast core this is exactly `gemm_array::design`.
    let control = match ctl_probe {
        Some(p) => Control::Probe(p),
        None if core == CoreVariant::FastSlowCtl => Control::Slow,
        None => Control::Fast,
    };
    // `--iu4-epochs` (V8, `--epi i32`): M = tokens, N = weight rows (features), K; native hipfire IU4 operands
    // (symmetric MQ4G256V2 x block_i4_128) and the per-K128-epoch int32 partials `C_e` as the checked output.
    let iu4 = args.iter().any(|a| a == "--iu4-epochs");
    if iu4 && !(v8 && epi == Some(Epilogue::I32) && !lock_only) {
        eprintln!("--iu4-epochs needs --variant V8 --epi i32 with --core Fast or FastSlowCtl");
        std::process::exit(2)
    }
    let mut iu4_io: Option<(Vec<i32>, [Vec<u8>; 2], u64)> = None;
    let mut d = match epi {
        Some(_) if iu4 => {
            let g = EpochGemm::new(m, n, k, control);
            let (wb, xb) = iu4::random_operands(m, n, k, 0x5eed, true);
            let (w, x) = (iu4::Weights::new(n, k, &wb), iu4::Acts::new(m, k, &xb));
            let t0 = std::time::Instant::now();
            let partials = iu4::partials(&w, &x);
            println!("iu4-epochs: tokens {m} features {n} K {k}: {} epochs, {} int32 partials [e][t][r] (CPU reference {:.2} s); \
                native weights {} B, activations {} B", g.epochs(), partials.len(), t0.elapsed().as_secs_f64(), wb.len(), xb.len());
            iu4_io = Some((g.padded(&partials), g.pack_in(&w, &x), g.useful_ops()));
            g.design
        }
        Some(epi) if g80 && n512 => peacemaker_npu::kernels::gemm_g80_n512::design_g80_n512_layout(
            m,
            n,
            k,
            epi,
            control,
            rowmajor.unwrap_or_default(),
        ),
        Some(epi) if g80 => peacemaker_npu::kernels::gemm_g80::design_g80_bf16_layout(
            m,
            n,
            k,
            epi,
            control,
            rowmajor.unwrap_or_default(),
        ),
        Some(epi) if v10 && b_prefix => design_v10_prefix(m, n, k, epi, control),
        Some(epi) if v10 => design_v10(m, n, k, epi, control),
        Some(epi) if v9 => design_v9(m, n, k, epi, control),
        Some(epi) => design_v8(m, n, k, epi, control),
        None => design_variant_probe(m, n, k, variant.unwrap_or(Variant::V6), core, probe_iters),
    };
    let defaults = peacemaker_npu::dma::ShimAxi::default();
    let field = |name: &str, default: u32, max: u32| {
        opt_str(args, name).map_or(default, |s| match s.parse::<u32>() {
            Ok(v) if v <= max => v,
            _ => {
                eprintln!("{name} must be 0..={max}");
                std::process::exit(2)
            }
        })
    };
    let axi = peacemaker_npu::dma::ShimAxi {
        burst: field("--burst", defaults.burst, 3),
        cache: field("--axcache", defaults.cache, 15),
        qos: field("--axqos", defaults.qos, 15),
    };
    d.set_shim_axi(axi).unwrap_or_else(|e| {
        eprintln!("shim AXI: {e}");
        std::process::exit(2)
    });
    println!(
        "shim AXI: {axi:?}, memory={}",
        memory.map_or("native", |k| k.describe())
    );
    let tag_owned = if iu4 {
        format!("{tag}-iu4")
    } else {
        tag.to_string()
    };
    let tag: &str = &tag_owned;
    // ClockProbe: cycles the delay loops add per submit (every active core runs `waves` tiles of `probe_iters`
    // iterations; cores run in parallel, so this is per core): the slope of time over this is the AIE clock period.
    let probe_note = if core == CoreVariant::ClockProbe {
        format!(
            " probe_iters={probe_iters} probe_cycles={}",
            probe_iters as u64 * clock_probe_cycles_per_iter() * d.waves() as u64
        )
    } else {
        String::new()
    };
    // IU4 epochs: native IU4 operands/reference prepared above. Timing-only and LockOnly probes need buffers but no
    // unused CPU GEMM reference (LockOnly: zero inputs, useful ops 0, success is the lock pattern). Fast/Serial:
    // random int8 inputs, exact CPU reference.
    let (want, inputs, useful) = if let Some(io) = iu4_io {
        io
    } else if lock_only || timing_only {
        (
            Vec::new(),
            [vec![0u8; d.args[0].bytes], vec![0u8; d.args[1].bytes]],
            if lock_only { 0 } else { d.useful_ops() },
        )
    } else {
        let mut seed = 0x5eed_u64;
        let a: Vec<i8> = (0..m * k).map(|_| lcg(&mut seed)).collect();
        let b: Vec<i8> = (0..k * n).map(|_| lcg(&mut seed)).collect();
        if epi == Some(Epilogue::Bf16) {
            use peacemaker_npu::kernels::{
                gemm_g80::{pack_a_bf16, pack_b_bf16},
                w8a8_epi::bf16_direct,
            };
            let mut scale = || {
                let mantissa = 16384 + (u16::from(lcg(&mut seed) as u8) << 6);
                let exponent = -20 + i16::from(lcg(&mut seed) as u8 % 7);
                (mantissa, exponent)
            };
            let ds: Vec<_> = (0..m).map(|_| scale()).collect();
            let ss: Vec<_> = (0..n).map(|_| scale()).collect();
            let want = peacemaker_npu::kernels::gemm_i8::cpu_reference(&a, &b, m, n, k)
                .into_iter()
                .enumerate()
                .map(|(at, i)| {
                    let (sm, se) = ss[at % n];
                    let (dm, de) = ds[at / n];
                    i32::from(bf16_direct(i, sm, se, dm, de))
                })
                .collect();
            (
                want,
                [pack_a_bf16(&d, &a, &ds), pack_b_bf16(&d, &b, &ss)],
                d.useful_ops(),
            )
        } else {
            (d.reference(&a, &b), d.pack_in(&a, &b), d.useful_ops())
        }
    };
    let want_c0 = if timing_only {
        "n/a (timing-only)".to_string()
    } else if lock_only {
        format!("{:#010x} (lock pattern)", lock_word(0, 0))
    } else {
        want[0].to_string()
    };
    let (host_rd, host_wr) = (d.host_bytes_read(), d.host_bytes_written());
    println!(
        "array design M={m} N={n} K={k}: waves={} pdi {} B, insts {} B, args {:?}, useful ops {useful}, host bytes/submit read {host_rd} write {host_wr}",
        d.waves(),
        d.pdi.len(),
        d.insts.len(),
        d.args
    );
    if let Some(epi) = d.epilogue() {
        let e = d.estimate(1_800_000_000);
        println!("{} epilogue {epi:?}, control {:?}; ESTIMATE @1.80 GHz: {} cycles = {:.1} us, {:.2} useful TOPS (peak {:.1})",
            variant.as_ref().map_or("V8", |v| v.name()),
            d.control().unwrap(), e.estimated_cycles, e.estimated_seconds * 1e6, e.useful_tops, e.peak_ops_per_second as f64 / 1e12);
        if b_prefix {
            println!("note: the ESTIMATE above is the baseline V10 whole-B startup model; it does not model --b-prefix descriptor overlap or timing");
        }
    }
    println!("variant {tag}: core {core_name}, wait timeout {timeout_ms} ms");
    assert!(d.args.len() <= 5, "firmware patches at most 5 args");
    let out_idx = d
        .args
        .iter()
        .position(|s| s.kind == ArgKind::Out)
        .expect("output arg");
    assert!(
        d.args[0].kind == ArgKind::In && d.args[0].bytes == inputs[0].len(),
        "arg0 must be the packed A input"
    );

    // Declared before `dev` on purpose: locals drop in reverse order, so the HIP allocation and dma-buf fd outlive
    // the hardware contexts, the imported BO and the device. It is only initialised after the device opened, so a
    // missing/failed NPU device never reaches any HIP call.
    let hip_args: Vec<Option<HipDmabufA>>;
    let mut dev = match Device::open() {
        Ok(dev) => dev,
        Err(e) => {
            println!("device open failed: {e}");
            return false;
        }
    };
    // Only the PDI and instruction stream live in the DEV heap (dev_bo); args are SHMEM BOs outside it.
    if let Err(e) = dev.map_heap(64 << 20) {
        println!("device heap map failed: {e}");
        return false;
    }
    hip_args = if memory.is_none() && !a_from_hip {
        Vec::new()
    } else {
        let mut allocations = Vec::with_capacity(d.args.len());
        for (i, spec) in d.args.iter().enumerate() {
            let kind = memory.or_else(|| (a_from_hip && i == 0).then_some(HipMemoryKind::VmmHost));
            let allocation = if let Some(kind) = kind {
                let poison;
                let bytes = match spec.kind {
                    ArgKind::In => &inputs[i],
                    ArgKind::Out => {
                        poison = vec![0xA5; spec.bytes];
                        &poison
                    }
                };
                match HipDmabufA::create_kind(bytes, kind) {
                    Ok(h) => {
                        println!("memory arg{i}: {}: {}", kind.describe(), h.describe());
                        Some(h)
                    }
                    Err(e) => {
                        println!("{e}");
                        return false;
                    }
                }
            } else {
                None
            };
            allocations.push(allocation);
        }
        allocations
    };
    let ntiles = dev.meta.cols as u32 * dev.meta.core.0 as u32;

    // Builds the argument BOs (inputs filled; Out poisoned).
    let make_bos = |dev: &Device| -> Result<Vec<Bo>, String> {
        let mut next_in = inputs.iter();
        let mut bos = Vec::with_capacity(d.args.len());
        for (i, spec) in d.args.iter().enumerate() {
            if let Some(h) = hip_args.get(i).and_then(Option::as_ref) {
                if spec.kind == ArgKind::In {
                    next_in.next();
                }
                let bo = dev.import_dmabuf(h.fd(), h.bytes()).map_err(|e| {
                    format!(
                        "memory arg{i} import_dmabuf: fd {} size {} failed: {e}",
                        h.fd(),
                        h.bytes()
                    )
                })?;
                bo.flush();
                println!("memory arg{i}: import succeeded: fd {} gem handle {} dev_addr {:#x} host {:p} len {}", h.fd(), bo.handle, bo.dev_addr(), bo.host, bo.len);
                bos.push(bo);
                continue;
            }
            let mut bo = dev
                .shmem_bo(spec.bytes)
                .map_err(|e| format!("arg {i} shmem bo ({} B) failed: {e}", spec.bytes))?;
            match spec.kind {
                ArgKind::In => bo
                    .as_mut_slice()
                    .copy_from_slice(next_in.next().expect("input buffer")),
                ArgKind::Out => bo.as_mut_slice().fill(0xA5),
            }
            bo.flush();
            bos.push(bo);
        }
        Ok(bos)
    };
    let tile_ranges = d.output_tile_ranges();
    let lock_words: usize = tile_ranges.iter().map(|(_, r)| r.len() * LOCK_WORDS).sum();
    // Exact compare of the Out BO against the CPU reference (LockOnly: against the lock pattern over every packed
    // tile range); returns (mismatches, c[0] as text).
    let check = |bos: &[Bo]| -> (usize, String) {
        bos[out_idx].flush();
        if timing_only {
            let (changed, _) = changed_vs_poison(bos[out_idx].as_slice());
            return (0, format!("n/a ({changed} C bytes differ from poison)"));
        }
        if lock_only {
            let out = bos[out_idx].as_slice();
            let bad = lock_pattern(out, &tile_ranges).map(|t| t.bad).sum();
            let c0 = tile_ranges
                .first()
                .and_then(|(_, r)| r.first())
                .and_then(|r| out.get(r.start..r.start + 4))
                .map_or("n/a".to_string(), |b| {
                    format!("{:#010x}", u32::from_le_bytes(b.try_into().unwrap()))
                });
            return (bad, c0);
        }
        let got = d.unpack_out(bos[out_idx].as_slice());
        if got.len() != want.len() {
            return (usize::MAX, got.first().map_or(0, |v| *v).to_string());
        }
        let outside_bad = if let Some(peacemaker_npu::kernels::gemm_g80::Bf16Layout::RowMajor {
            pitch_elems: pitch,
            col0_elems: col0,
        }) = rowmajor
        {
            bos[out_idx]
                .as_slice()
                .chunks_exact(2)
                .enumerate()
                .filter(|(at, bytes)| {
                    let col = at % pitch;
                    (col < col0 || col >= col0 + n) && bytes.iter().any(|&b| b != 0xA5)
                })
                .count()
        } else {
            0
        };
        let padding_bad = if let Some(peacemaker_npu::kernels::gemm_g80::Bf16Layout::RowMajor {
            pitch_elems: pitch,
            col0_elems: col0,
        }) = rowmajor
        {
            (m..d.waves() * d.wave_m())
                .flat_map(|row| (0..n).map(move |col| 2 * (row * pitch + col0 + col)))
                .filter(|&off| bos[out_idx].as_slice()[off..off + 2] != [0, 0])
                .count()
        } else {
            0
        };
        (
            got.iter().zip(&want).filter(|(g, w)| g != w).count() + outside_bad + padding_bad,
            got[0].to_string(),
        )
    };
    let fmt_bad = |bad: usize| {
        if timing_only {
            "timing-only (C not a GEMM, not verified)".to_string()
        } else if lock_only {
            format!("{bad}/{lock_words} lock-pattern words")
        } else if bad == usize::MAX {
            "unpacked length mismatch".to_string()
        } else {
            format!("{bad}/{}", want.len())
        }
    };
    // Which output tiles received anything (vs the 0xA5 poison); printed for any failed submit.
    let report_c = |bos: &[Bo]| {
        report_output(bos[out_idx].as_slice(), &tile_ranges);
        if lock_only {
            report_lock_pattern(bos[out_idx].as_slice(), &tile_ranges);
        }
    };

    let print_summary = |submits: u64, busy: f64, wall: f64| {
        let total_ops = useful as f64 * submits as f64;
        let total_bytes = (host_rd as u128 + host_wr as u128) * submits as u128;
        println!(
            "summary: submits={submits} useful_ops={} wall={wall:.3} s busy={busy:.3} s mean={:.4} TOPS (busy) {:.4} TOPS (wall) host_bytes read={} written={} gap_us={gap_us} npu_dram_GBps={:.3}",
            useful as u128 * submits as u128,
            total_ops / busy / 1e12,
            total_ops / wall / 1e12,
            host_rd as u128 * submits as u128,
            host_wr as u128 * submits as u128,
            total_bytes as f64 / wall / 1e9
        );
        println!(
            "bandwidth: read_GBps={:.3} write_GBps={:.3} exactness={}",
            host_rd as f64 * submits as f64 / busy / 1e9,
            host_wr as f64 * submits as f64 / busy / 1e9,
            if timing_only {
                "timing-only"
            } else {
                "see-first-final-checks"
            }
        );
    };
    // Setup failures (railgun `Err(String)`, which carries the errno) print the stage and end the run FAIL instead
    // of panicking; locals (contexts, BOs, the HIP allocation) are released by their normal drops.
    macro_rules! step {
        ($e:expr, $what:expr, $fail:expr) => {
            match $e {
                Ok(v) => v,
                Err(e) => {
                    println!("{}: {e}", $what);
                    $fail
                }
            }
        };
    }
    let mut ok_all = true;
    if !reuse {
        let start = std::time::Instant::now();
        let mut submits = 0u64;
        let mut busy = 0.0f64;
        for it in 0..iters {
            let ctx = step!(
                HwCtx::create(&dev, ntiles, 2048),
                format!("iter {it}: create_hwctx failed"),
                {
                    ok_all = false;
                    break;
                }
            );
            let pdi = step!(dev.dev_bo(&d.pdi), format!("iter {it}: pdi bo failed"), {
                ok_all = false;
                break;
            });
            step!(
                ctx.config_cu(&pdi),
                format!("iter {it}: config_cu failed"),
                {
                    ok_all = false;
                    break;
                }
            );
            let insts = step!(
                dev.dev_bo(&d.insts),
                format!("iter {it}: insts bo failed"),
                {
                    ok_all = false;
                    break;
                }
            );
            let bos = match make_bos(&dev) {
                Ok(b) => b,
                Err(e) => {
                    println!("iter {it}: {e}");
                    ok_all = false;
                    break;
                }
            };
            let mut cmd = step!(dev.cmd_bo(), format!("iter {it}: cmd bo failed"), {
                ok_all = false;
                break;
            });
            let refs: Vec<&Bo> = bos.iter().collect();
            let t0 = std::time::Instant::now();
            let seq = match ctx.submit(&mut cmd, &insts, &refs) {
                Ok(s) => s,
                Err(e) => {
                    println!("iter {it}: submit error: {e}");
                    dump_aie_status(&dev, tag, it, "submit error");
                    ok_all = false;
                    break;
                }
            };
            let state = ctx.wait(&cmd, seq, timeout_ms);
            let dt = t0.elapsed().as_secs_f64();
            submits += 1;
            busy += dt;
            // Query the column status BEFORE the hwctx is destroyed (destroy resets the columns).
            if dump || !matches!(state, Ok(ERT_STATE_COMPLETED)) {
                dump_aie_status(&dev, tag, it, &format!("wait {state:?}"));
            }
            drop(ctx);
            let (bad, c0) = check(&bos);
            let ok = matches!(state, Ok(ERT_STATE_COMPLETED)) && bad == 0;
            println!(
                "iter {it}: state={state:?} time={:.1} us useful={:.4} TOPS mismatches={} c[0]={} want {}{probe_note}",
                dt * 1e6,
                useful as f64 / dt / 1e12,
                fmt_bad(bad),
                c0,
                want_c0
            );
            if !ok {
                report_c(&bos);
            }
            ok_all &= ok;
            if !ok {
                break;
            }
        }
        print_summary(submits, busy, start.elapsed().as_secs_f64());
        return ok_all;
    }

    // Reused context: one hwctx, one PDI/insts/arg set, many submits.
    let ctx = step!(
        HwCtx::create(&dev, ntiles, 2048),
        "create_hwctx failed",
        return false
    );
    let pdi = step!(dev.dev_bo(&d.pdi), "pdi bo failed", return false);
    step!(ctx.config_cu(&pdi), "config_cu failed", return false);
    let insts = step!(dev.dev_bo(&d.insts), "insts bo failed", return false);
    let bos = match make_bos(&dev) {
        Ok(b) => b,
        Err(e) => {
            println!("{e}");
            return false;
        }
    };
    let mut cmd = step!(dev.cmd_bo(), "cmd bo failed", return false);
    let refs: Vec<&Bo> = bos.iter().collect();

    let verify_submit = |idx: u64, state: &Result<u32, String>, dt: f64| -> bool {
        let (bad, c0) = check(&bos);
        let ok = matches!(state, Ok(ERT_STATE_COMPLETED)) && bad == 0;
        println!(
            "submit {idx}: state={state:?} time={:.1} us useful={:.4} TOPS mismatches={} c[0]={} want {}{probe_note}",
            dt * 1e6,
            useful as f64 / dt / 1e12,
            fmt_bad(bad),
            c0,
            want_c0
        );
        if !ok {
            report_c(&bos);
        }
        ok
    };
    let start = std::time::Instant::now();
    let mut submits = 0u64;
    let mut busy = 0.0f64;
    let mut first_ok: Option<bool> = None;
    let mut last_ok = true;
    let mut fail = false;
    let loop_mode = loop_secs.is_some();
    // Loop mode: after the deadline one more submit runs on a freshly poisoned output and is verified as the last one,
    // so intermediate submits need no 0xA5 poison + cache flush of the whole C buffer (host overhead, not NPU work).
    let mut final_pass = false;
    loop {
        // Poison the output so a result left over from a previous submit cannot pass.
        if submits > 0 && (!loop_mode || final_pass) {
            let out = &bos[out_idx];
            unsafe { std::ptr::write_bytes(out.host, 0xA5, out.len) };
            out.flush();
        }
        let t0 = std::time::Instant::now();
        let seq = match ctx.submit(&mut cmd, &insts, &refs) {
            Ok(s) => s,
            Err(e) => {
                println!("submit {submits}: submit error: {e}");
                dump_aie_status(&dev, tag, submits as usize, "submit error");
                fail = true;
                break;
            }
        };
        let state = ctx.wait(&cmd, seq, timeout_ms);
        let dt = t0.elapsed().as_secs_f64();
        submits += 1;
        busy += dt;
        let completed = matches!(state, Ok(ERT_STATE_COMPLETED));
        if dump || !completed {
            dump_aie_status(&dev, tag, submits as usize - 1, &format!("wait {state:?}"));
        }
        // Fixed-iteration mode verifies every submit; loop mode verifies the first, the final poisoned one, and any
        // non-completed one.
        let verified = if !loop_mode || submits == 1 || final_pass || !completed {
            Some(verify_submit(submits - 1, &state, dt))
        } else {
            None
        };
        // Deadline is checked after any verification so a slow first check cannot let a further submit slip in.
        let done = match loop_secs {
            Some(_) if final_pass => true,
            Some(s) => {
                final_pass = start.elapsed().as_secs_f64() >= s;
                false
            }
            None => submits as usize >= iters,
        };
        if let Some(ok) = verified {
            if loop_mode {
                if submits == 1 {
                    first_ok = Some(ok);
                }
                last_ok = ok;
            }
            if !ok {
                fail = true;
            }
        }
        if gap_us > 0 && completed && !fail {
            // Idle gap outside the timed busy region (t0..wait); it counts in wall only.
            std::thread::sleep(std::time::Duration::from_micros(gap_us));
        }
        if fail || done {
            break;
        }
    }
    print_summary(submits, busy, start.elapsed().as_secs_f64());
    if loop_secs.is_some() {
        let name = |v: Option<bool>| {
            v.map_or("n/a", |v| {
                if !v {
                    "MISMATCH"
                } else if timing_only {
                    "completed-unverified"
                } else {
                    "exact"
                }
            })
        };
        println!(
            "verify: first={} last={}",
            name(first_ok),
            name(Some(last_ok))
        );
    }
    drop(ctx);
    !fail
}

/// One IEF15 problem: design, packed operands and the CPU oracle bytes.
struct Ief15Job {
    name: String,
    g: Ief15Gemm,
    inputs: [Vec<u8>; 2],
    want: Vec<u8>,
    tokens: usize,
    features: usize,
}

fn ief15_job(
    name: &str,
    tokens: usize,
    features: usize,
    k: usize,
    seed: u64,
) -> Result<Ief15Job, String> {
    let g = Ief15Gemm::new(tokens, features, k, Control::Fast)?;
    let (wb, xb) = iu4::random_operands(tokens, features, k, seed, true);
    let (w, x) = (
        iu4::Weights::new(features, k, &wb),
        iu4::Acts::new(tokens, k, &xb),
    );
    let t0 = std::time::Instant::now();
    let (s, a) = iu4_ief15::encode_weight_scales(&w);
    let (d, b) = iu4_ief15::encode_act_scales(&x);
    let inputs = g.pack_in(&w, &x, &s, &a, &d, &b)?;
    let t1 = std::time::Instant::now();
    let want = g.reference(&w, &x, &s, &a, &d, &b);
    println!("ief15 {name}: tokens {tokens} features {features} K {k}: {} epochs, {} waves, args {:?}, pdi {} B, insts {} B; \
        sidecars+pack {:.2} s, CPU reference {:.2} s; useful ops {}",
        g.epochs(), g.design().waves(), g.design().args.iter().map(|a| a.bytes).collect::<Vec<_>>(), g.design().pdi.len(),
        g.design().insts.len(), (t1 - t0).as_secs_f64(), t1.elapsed().as_secs_f64(), g.useful_ops());
    Ok(Ief15Job {
        name: name.to_string(),
        g,
        inputs,
        want,
        tokens,
        features,
    })
}

/// One hardware context with its buffers for one IEF15 design.
struct Ief15Side<'d> {
    ctx: HwCtx<'d>,
    _pdi: Bo,
    insts: Bo,
    lean: Option<Bo>,
    bos: Vec<Bo>,
    cmd: Bo,
}

fn ief15_side<'d>(
    dev: &'d Device,
    job: &Ief15Job,
    ntiles: u32,
    lean: bool,
) -> Result<Ief15Side<'d>, String> {
    let d = job.g.design();
    let ctx = HwCtx::create(dev, ntiles, 2048).map_err(|e| format!("create_hwctx failed: {e}"))?;
    let pdi = dev
        .dev_bo(&d.pdi)
        .map_err(|e| format!("pdi bo failed: {e}"))?;
    ctx.config_cu(&pdi)
        .map_err(|e| format!("config_cu failed: {e}"))?;
    let insts = dev
        .dev_bo(&d.insts)
        .map_err(|e| format!("insts bo failed: {e}"))?;
    let lean = if lean {
        let l = d.lean_insts().map_err(|e| format!("lean_insts: {e}"))?;
        Some(
            dev.dev_bo(&l)
                .map_err(|e| format!("lean insts bo failed: {e}"))?,
        )
    } else {
        None
    };
    let mut bos = Vec::with_capacity(d.args.len());
    for (i, spec) in d.args.iter().enumerate() {
        let mut bo = dev
            .shmem_bo(spec.bytes)
            .map_err(|e| format!("arg {i} shmem bo ({} B) failed: {e}", spec.bytes))?;
        match spec.kind {
            ArgKind::In => bo.as_mut_slice().copy_from_slice(&job.inputs[i]),
            ArgKind::Out => bo.as_mut_slice().fill(0xA5),
        }
        bo.flush();
        bos.push(bo);
    }
    let cmd = dev.cmd_bo().map_err(|e| format!("cmd bo failed: {e}"))?;
    Ok(Ief15Side {
        ctx,
        _pdi: pdi,
        insts,
        lean,
        bos,
        cmd,
    })
}

/// One submit (full TXN, or the lean one) and its wait: `(state, seconds submit -> completion)`.
fn ief15_submit(side: &mut Ief15Side, lean: bool, timeout_ms: u64) -> (Result<u32, String>, f64) {
    let refs: Vec<&Bo> = side.bos.iter().collect();
    let insts = if lean {
        side.lean.as_ref().expect("lean TXN built")
    } else {
        &side.insts
    };
    let t0 = std::time::Instant::now();
    let seq = match side.ctx.submit(&mut side.cmd, insts, &refs) {
        Ok(s) => s,
        Err(e) => {
            return (
                Err(format!("submit error: {e}")),
                t0.elapsed().as_secs_f64(),
            )
        }
    };
    let state = side.ctx.wait(&side.cmd, seq, timeout_ms);
    (state, t0.elapsed().as_secs_f64())
}

/// Refill the C buffer with the 0xA5 poison so a stale result cannot pass.
fn ief15_poison(side: &Ief15Side) {
    let out = &side.bos[2];
    unsafe { std::ptr::write_bytes(out.host, 0xA5, out.len) };
    out.flush();
}

/// Exact compare of the C buffer against the oracle: `(mismatching f32 words, description of the first one)`.
fn ief15_check(side: &Ief15Side, job: &Ief15Job) -> (usize, String) {
    side.bos[2].flush();
    let got = job.g.unpack_out(side.bos[2].as_slice());
    let word = |v: &[u8], i: usize| u32::from_le_bytes(v[i * 4..i * 4 + 4].try_into().unwrap());
    let n = got.len() / 4;
    let bad: Vec<usize> = (0..n)
        .filter(|&i| word(&got, i) != word(&job.want, i))
        .collect();
    let first = bad.first().map_or("none".to_string(), |&i| {
        format!(
            "token {} feature {}: got {:#010x} want {:#010x}",
            i / job.features,
            i % job.features,
            word(&got, i),
            word(&job.want, i)
        )
    });
    (bad.len(), first)
}

/// Check + print one verified submit; `completed && mismatches == 0`.
fn ief15_verify(
    dev: &Device,
    label: &str,
    side: &Ief15Side,
    job: &Ief15Job,
    state: &Result<u32, String>,
    dt: f64,
    dump: bool,
) -> bool {
    let completed = matches!(state, Ok(ERT_STATE_COMPLETED));
    if dump || !completed {
        dump_aie_status(dev, &job.name, 0, &format!("{label} wait {state:?}"));
    }
    let (bad, first) = ief15_check(side, job);
    let ok = completed && bad == 0;
    println!("{label}: state={state:?} time={:.1} us useful={:.4} TOPS mismatches={bad}/{} first mismatch: {first}",
        dt * 1e6, job.g.useful_ops() as f64 / dt / 1e12, job.tokens * job.features);
    ok
}

/// `--fold-contract ief15`; see the module header. Exit-2 flag errors are reported before any hardware access.
fn run_ief15(args: &[String], tokens: usize, features: usize, k: usize) -> bool {
    fn bad_flag(msg: String) -> ! {
        eprintln!("{msg}");
        std::process::exit(2)
    }
    if opt_str(args, "--fold-contract") != Some("ief15") {
        bad_flag("--fold-contract must be ief15".into());
    }
    for f in [
        "--array",
        "--variant",
        "--core",
        "--epi",
        "--shift",
        "--ctl-probe",
        "--a-from-hip",
        "--memory",
        "--burst",
        "--axcache",
        "--axqos",
        "--b-prefix",
        "--probe-iters",
        "--iu4-epochs",
        "--gap-us",
        "--col",
    ] {
        if args.iter().any(|a| a == f) {
            bad_flag(format!("{f} is not valid with --fold-contract ief15 (AxQOS 0 / vendor BD word 5 defaults, no AXI or variant knobs)"));
        }
    }
    let loop_secs: Option<f64> = opt_str(args, "--loop").map(|s| {
        s.parse()
            .unwrap_or_else(|_| bad_flag("--loop SECONDS".into()))
    });
    if loop_secs.is_some_and(|s| !(s.is_finite() && s > 0.0)) {
        bad_flag("--loop SECONDS must be finite and > 0".into());
    }
    let iters: usize = opt_str(args, "--iters").map_or(3, |s| {
        s.parse().unwrap_or_else(|_| bad_flag("--iters N".into()))
    });
    if iters == 0 {
        bad_flag("--iters must be >= 1".into());
    }
    let lean = args.iter().any(|a| a == "--lean");
    let reuse = loop_secs.is_some() || lean || args.iter().any(|a| a == "--reuse-ctx");
    let dump = args.iter().any(|a| a == "--dump-status");
    let timeout = timeout_ms(args);
    let switch: Option<(usize, usize, usize)> = opt_str(args, "--switch-with").map(|s| {
        let v: Vec<usize> = s
            .split(',')
            .map(|p| {
                p.trim()
                    .parse()
                    .unwrap_or_else(|_| bad_flag(format!("--switch-with T2,F2,K2, got {s:?}")))
            })
            .collect();
        if v.len() != 3 {
            bad_flag(format!("--switch-with T2,F2,K2, got {s:?}"));
        }
        (v[0], v[1], v[2])
    });
    if switch.is_some() && (loop_secs.is_some() || lean) {
        bad_flag("--switch-with cannot be combined with --loop or --lean (a lean TXN is valid only directly after a completed submit of the same design)".into());
    }
    let jobs: Vec<Ief15Job> = {
        let mut v = vec![ief15_job("A", tokens, features, k, 0x5eed)];
        if let Some((t2, f2, k2)) = switch {
            v.push(ief15_job("B", t2, f2, k2, 0x5eee));
        }
        match v.into_iter().collect::<Result<Vec<_>, _>>() {
            Ok(v) => v,
            Err(e) => {
                println!("ief15 design: {e}");
                return false;
            }
        }
    };
    let mut dev = match Device::open() {
        Ok(dev) => dev,
        Err(e) => {
            println!("device open failed: {e}");
            return false;
        }
    };
    if let Err(e) = dev.map_heap(64 << 20) {
        println!("device heap map failed: {e}");
        return false;
    }
    let ntiles = dev.meta.cols as u32 * dev.meta.core.0 as u32;
    let job = &jobs[0];
    println!("variant IEF15: core Fast, wait timeout {timeout} ms");

    if let Some((t2, f2, k2)) = switch {
        let other = &jobs[1];
        let mut sides = Vec::new();
        for j in &jobs {
            match ief15_side(&dev, j, ntiles, false) {
                Ok(s) => sides.push(s),
                Err(e) => {
                    println!("{}: {e}", j.name);
                    return false;
                }
            }
        }
        let mut ok_all = true;
        // First submit of each design, verified.
        for (i, j) in jobs.iter().enumerate() {
            let (state, dt) = ief15_submit(&mut sides[i], false, timeout);
            ok_all &= ief15_verify(
                &dev,
                &format!("first {}", j.name),
                &sides[i],
                j,
                &state,
                dt,
                dump,
            );
            if !ok_all {
                return false;
            }
        }
        let modes: [(&str, Vec<usize>); 3] = [
            ("repeat-A", vec![0; iters]),
            ("repeat-B", vec![1; iters]),
            ("alternate", (0..iters).flat_map(|_| [0, 1]).collect()),
        ];
        let mut means = Vec::new();
        for (name, seq) in modes {
            let (mut busy, mut count) = ([0.0f64; 2], [0usize; 2]);
            let start = std::time::Instant::now();
            for (pos, &i) in seq.iter().enumerate() {
                // The last submit of each design in the mode runs on a poisoned output and is verified.
                let last = seq[pos + 1..].iter().all(|&o| o != i);
                if last {
                    ief15_poison(&sides[i]);
                }
                let (state, dt) = ief15_submit(&mut sides[i], false, timeout);
                busy[i] += dt;
                count[i] += 1;
                let completed = matches!(state, Ok(ERT_STATE_COMPLETED));
                if last || !completed || dump {
                    ok_all &= ief15_verify(
                        &dev,
                        &format!("{name} final {}", jobs[i].name),
                        &sides[i],
                        &jobs[i],
                        &state,
                        dt,
                        dump,
                    );
                }
                if !completed || !ok_all {
                    break;
                }
            }
            if !ok_all {
                break;
            }
            let wall = start.elapsed().as_secs_f64();
            let per = |i: usize| {
                if count[i] > 0 {
                    busy[i] / count[i] as f64 * 1e6
                } else {
                    f64::NAN
                }
            };
            let total = (busy[0] + busy[1]) / (count[0] + count[1]) as f64 * 1e6;
            println!("mode {name}: submits A={} B={} mean {total:.1} us/submit busy (A {:.1} us, B {:.1} us), wall {:.3} s",
                count[0], count[1], per(0), per(1), wall);
            means.push((name, total, per(0), per(1)));
        }
        if ok_all && means.len() == 3 {
            let (ra, rb, alt) = (means[0].2, means[1].3, &means[2]);
            println!("switch cost: A->B alternation adds {:+.1} us on A (alt {:.1} vs repeat {ra:.1}) and {:+.1} us on B (alt {:.1} vs repeat {rb:.1}); \
                mean per submit {:.1} us alternating vs {:.1} us repeating; designs A {tokens}x{features}x{k}, B {t2}x{f2}x{k2}",
                alt.2 - ra, alt.2, alt.3 - rb, alt.3, alt.1, (ra + rb) / 2.0);
            let _ = other;
        }
        return ok_all;
    }

    let start = std::time::Instant::now();
    let (mut submits, mut busy) = (0u64, 0.0f64);
    let mut ok_all = true;
    if !reuse {
        for it in 0..iters {
            let mut side = match ief15_side(&dev, job, ntiles, false) {
                Ok(s) => s,
                Err(e) => {
                    println!("iter {it}: {e}");
                    ok_all = false;
                    break;
                }
            };
            let (state, dt) = ief15_submit(&mut side, false, timeout);
            submits += 1;
            busy += dt;
            let ok = ief15_verify(&dev, &format!("iter {it}"), &side, job, &state, dt, dump);
            drop(side);
            ok_all &= ok;
            if !ok {
                break;
            }
        }
    } else {
        let mut side = match ief15_side(&dev, job, ntiles, lean) {
            Ok(s) => s,
            Err(e) => {
                println!("{e}");
                return false;
            }
        };
        let mut final_pass = false;
        loop {
            // Submit 0 is always full; with --lean every later submit is the lean TXN.
            let use_lean = lean && submits > 0;
            let is_final = match loop_secs {
                Some(_) => final_pass,
                None => submits as usize + 1 == iters,
            };
            if is_final && submits > 0 {
                ief15_poison(&side);
            }
            let (state, dt) = ief15_submit(&mut side, use_lean, timeout);
            submits += 1;
            busy += dt;
            let completed = matches!(state, Ok(ERT_STATE_COMPLETED));
            if submits == 1 || is_final || !completed || dump {
                let label = format!(
                    "submit {} ({}{})",
                    submits - 1,
                    if use_lean { "lean" } else { "full" },
                    if is_final { ", final" } else { "" }
                );
                ok_all &= ief15_verify(&dev, &label, &side, job, &state, dt, dump);
            }
            let done = match loop_secs {
                Some(_) if final_pass => true,
                Some(s) => {
                    final_pass = start.elapsed().as_secs_f64() >= s;
                    false
                }
                None => submits as usize >= iters,
            };
            if !ok_all || done {
                break;
            }
        }
        drop(side);
    }
    let wall = start.elapsed().as_secs_f64();
    let total_ops = job.g.useful_ops() as f64 * submits as f64;
    let d = job.g.design();
    println!("summary: submits={submits} useful_ops={} wall={wall:.3} s busy={busy:.3} s mean={:.4} TOPS (busy) {:.4} TOPS (wall) \
        host_bytes read={} written={} mean {:.1} us/submit (busy)",
        job.g.useful_ops() as u128 * submits as u128, total_ops / busy / 1e12, total_ops / wall / 1e12,
        d.host_bytes_read() as u128 * submits as u128, d.host_bytes_written() as u128 * submits as u128, busy / submits as f64 * 1e6);
    ok_all
}

/// `--ief15-probe [--col C] [--timeout-ms T] [--dump PATH] [--expected-only]`: the IEF15 N0 single-core semantics probe
/// (`peacemaker_npu::kernels::iu4_ief15_probe`, design doc `docs/ief15-n0.md` section 7 step 1). The expected bytes are the output of
/// the exact probe PDI + TXN run in `peacemaker_npu::sim::config::Config`; the same PDI + TXN is then submitted once on a fresh
/// hardware context (shim BD word 5 and AxQoS are the vendor defaults) and compared byte for byte. Prints one
/// MATCH/MISMATCH line per case (case id, encoding, operands, output byte range; a mismatch also the first differing
/// lane, expected vs got) and returns false on any mismatch or a non-completed wait. `--dump PATH` writes the expected
/// bytes to `PATH.expected` and the hardware bytes to `PATH.got`; `--expected-only` stops after the simulator run (no
/// hardware access): it prints the expected blob's size and FNV-1a/64 and returns true.
fn run_ief15_probe(args: &[String]) -> bool {
    use peacemaker_npu::kernels::iu4_ief15_probe::{self as probe, SemanticsProbe};
    fn bad_flag(msg: String) -> ! {
        eprintln!("{msg}");
        std::process::exit(2)
    }
    for f in [
        "--array",
        "--variant",
        "--core",
        "--epi",
        "--shift",
        "--ctl-probe",
        "--a-from-hip",
        "--memory",
        "--burst",
        "--axcache",
        "--axqos",
        "--b-prefix",
        "--probe-iters",
        "--iu4-epochs",
        "--gap-us",
        "--fold-contract",
        "--loop",
        "--reuse-ctx",
        "--iters",
    ] {
        if args.iter().any(|a| a == f) {
            bad_flag(format!(
                "{f} is not valid with --ief15-probe (one fresh-context submit, vendor BD word 5)"
            ));
        }
    }
    let col: u32 = opt_str(args, "--col").map_or(0, |s| {
        s.parse()
            .unwrap_or_else(|_| bad_flag("--col C (0..7)".into()))
    });
    if col >= 8 {
        bad_flag("--col C (0..7)".into());
    }
    let dump = opt_str(args, "--dump").map(str::to_owned);
    let d = SemanticsProbe::design(col);
    let input = SemanticsProbe::input();
    let cases = SemanticsProbe::cases();
    println!(
        "ief15 probe: col {col}, {} cases, program {} B, in {} B, out {} B, pdi {} B, insts {} B",
        cases.len(),
        d.program_bytes,
        d.args[0].bytes,
        d.args[1].bytes,
        d.pdi.len(),
        d.insts.len()
    );
    // Expected bytes: the exact PDI + TXN in the simulator.
    let mut sim_args = vec![input.clone(), vec![0xA5u8; d.args[1].bytes]];
    let mut sim =
        peacemaker_npu::sim::config::Config::from_pdi(&d.pdi).expect("simulator: probe PDI");
    sim.tick_limit = 50_000_000;
    if let Err(e) = sim.submit(&d.insts, &mut sim_args) {
        println!("simulator run of the probe failed: {e:?}");
        return false;
    }
    let expected = sim_args.remove(1);
    let fnv = expected.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
    });
    println!(
        "expected (simulator): {} B, fnv1a64 {fnv:016x}",
        expected.len()
    );
    if let Some(p) = &dump {
        std::fs::write(format!("{p}.expected"), &expected).expect("write --dump PATH.expected");
    }
    if args.iter().any(|a| a == "--expected-only") {
        println!("--expected-only: simulator run only, no hardware access, nothing compared");
        return true;
    }

    let timeout_ms = timeout_ms(args);
    let mut dev = Device::open().expect("open");
    dev.map_heap(64 << 20).expect("heap");
    let ntiles = dev.meta.cols as u32 * dev.meta.core.0 as u32;
    let ctx = HwCtx::create(&dev, ntiles, 2048).expect("hwctx");
    let pdi = dev.dev_bo(&d.pdi).expect("pdi bo");
    ctx.config_cu(&pdi).expect("config_cu");
    let insts = dev.dev_bo(&d.insts).expect("insts bo");
    let mut bos = Vec::new();
    for (spec, data) in d.args.iter().zip([Some(&input), None]) {
        let mut bo = dev.shmem_bo(spec.bytes).expect("arg bo");
        match data {
            Some(v) => bo.as_mut_slice().copy_from_slice(v),
            None => bo.as_mut_slice().fill(0xA5),
        }
        bo.flush();
        bos.push(bo);
    }
    let mut cmd = dev.cmd_bo().expect("cmd bo");
    let refs: Vec<&Bo> = bos.iter().collect();
    let t0 = std::time::Instant::now();
    let seq = match ctx.submit(&mut cmd, &insts, &refs) {
        Ok(s) => s,
        Err(e) => {
            println!("submit error: {e}");
            dump_aie_status(&dev, "ief15-probe", 0, "submit error");
            return false;
        }
    };
    let state = ctx.wait(&cmd, seq, timeout_ms);
    let dt = t0.elapsed().as_secs_f64();
    let completed = matches!(state, Ok(ERT_STATE_COMPLETED));
    if !completed || args.iter().any(|a| a == "--dump-status") {
        dump_aie_status(&dev, "ief15-probe", 0, &format!("wait {state:?}"));
    }
    drop(ctx);
    bos[1].flush();
    let got = bos[1].as_slice().to_vec();
    if let Some(p) = &dump {
        std::fs::write(format!("{p}.got"), &got).expect("write --dump PATH.got");
    }
    println!("state={state:?} time={:.1} us", dt * 1e6);
    let bad = SemanticsProbe::compare(&expected, &got);
    let word = |case: &probe::Case, buf: &[u8], lane: usize| {
        format!(
            "{:#0w$x}",
            case.lane_value(buf, lane),
            w = case.lane_bytes() * 2 + 2
        )
    };
    for c in cases {
        let r = c.out_off as usize..c.out_off as usize + c.out_len as usize;
        match bad.iter().find(|m| m.case.id == c.id) {
            None => println!("MATCH    case {:3} {} {} out[{}..{}]", c.id, c.instruction(), c.operands(), r.start, r.end),
            Some(m) => println!("MISMATCH case {:3} {} {} out[{}..{}] bad bytes {}/{} first lane {} (byte {}): expected {} got {}",
                c.id, c.instruction(), c.operands(), r.start, r.end, m.bad_bytes, c.out_len, m.lane(), m.byte,
                word(c, &expected, m.lane()), word(c, &got, m.lane())),
        }
    }
    println!(
        "ief15 probe: {} of {} cases match{}",
        cases.len() - bad.len(),
        cases.len(),
        if completed {
            ""
        } else {
            " (WAIT DID NOT COMPLETE)"
        }
    );
    if let Some(m) = bad.first() {
        println!(
            "FIRST MISMATCH: case {} {} {} lane {} expected {} got {}",
            m.case.id,
            m.case.instruction(),
            m.case.operands(),
            m.lane(),
            word(m.case, &expected, m.lane()),
            word(m.case, &got, m.lane())
        );
    }
    completed && bad.is_empty()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--ief15-probe") {
        let ok_all = run_ief15_probe(&args);
        println!("RESULT: {}", if ok_all { "PASS" } else { "FAIL" });
        std::process::exit(if ok_all { 0 } else { 1 });
    }
    let num = |i: usize| -> usize {
        args.get(i).and_then(|s| s.parse().ok()).expect("usage: npu-gemm M N K [--iters I] [--col C] | --array [--core Fast|Serial|LockOnly|FastSlowCtl|ClockProbe] [--probe-iters N] [--reuse-ctx] [--loop SECONDS] [--dump-status]")
    };
    let (m, n, k) = (num(1), num(2), num(3));
    gap_us(&args);
    if args.iter().any(|a| a == "--fold-contract") {
        let ok_all = run_ief15(&args, m, n, k);
        println!("RESULT: {}", if ok_all { "PASS" } else { "FAIL" });
        std::process::exit(if ok_all { 0 } else { 1 });
    }
    let array_mode = args.iter().any(|a| a == "--array") || opt_str(&args, "--variant").is_some();
    if !array_mode && args.iter().any(|a| a == "--core") {
        eprintln!("--core selects an array core program and needs --array or --variant; the single-core gemm_i8 path has no core variants");
        std::process::exit(2);
    }
    if !array_mode && (args.iter().any(|a| a == "--epi") || args.iter().any(|a| a == "--shift")) {
        eprintln!("--epi/--shift are only valid with --variant V8, V9, V10 or G80");
        std::process::exit(2);
    }
    if !array_mode && args.iter().any(|a| a == "--a-from-hip") {
        eprintln!("--a-from-hip is only valid with --variant V8, V9, V10 or G80");
        std::process::exit(2);
    }
    if !array_mode && args.iter().any(|a| a == "--b-prefix") {
        eprintln!("--b-prefix is only valid with --variant V10 (array mode); the single-core gemm_i8 path has no B prefix mode");
        std::process::exit(2);
    }
    if !array_mode && args.iter().any(|a| a == "--probe-iters") {
        eprintln!("--probe-iters is only valid with --core ClockProbe (array mode)");
        std::process::exit(2);
    }
    if !array_mode
        && args
            .iter()
            .any(|a| matches!(a.as_str(), "--memory" | "--burst" | "--axcache" | "--axqos"))
    {
        eprintln!("--memory/--burst/--axcache/--axqos require --array or --variant");
        std::process::exit(2);
    }
    if array_mode {
        let ok_all = run_array(&args, m, n, k);
        println!("RESULT: {}", if ok_all { "PASS" } else { "FAIL" });
        std::process::exit(if ok_all { 0 } else { 1 });
    }
    let opt = |name: &str, d: usize| {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .map(|s| s.parse().unwrap())
            .unwrap_or(d)
    };
    let iters = opt("--iters", 3);
    let col = opt("--col", 0) as u32;

    let d = design(&[col], m, n, k);
    let mut seed = 0x5eed_u64;
    let a: Vec<i8> = (0..m * k).map(|_| lcg(&mut seed)).collect();
    let b: Vec<i8> = (0..k * n).map(|_| lcg(&mut seed)).collect();
    let want = cpu_reference(&a, &b, m, n, k);
    let inputs = d.pack_in(&a, &b);
    println!(
        "design M={m} N={n} K={k} col={col}: pdi {} B, insts {} B, args {:?}",
        d.pdi.len(),
        d.insts.len(),
        d.args
    );
    let dump = args.iter().any(|a| a == "--dump-status");
    let timeout_ms = timeout_ms(&args);

    let mut dev = Device::open().expect("open");
    dev.map_heap(64 << 20).expect("heap");
    let ntiles = dev.meta.cols as u32 * dev.meta.core.0 as u32;
    let ops = 2.0 * (m * n * k) as f64;
    let mut ok_all = true;
    for it in 0..iters {
        let ctx = HwCtx::create(&dev, ntiles, 2048).expect("hwctx");
        let pdi = dev.dev_bo(&d.pdi).expect("pdi bo");
        ctx.config_cu(&pdi).expect("config_cu");
        let insts = dev.dev_bo(&d.insts).expect("insts bo");
        let mut bos = Vec::new();
        let mut next_in = inputs.iter();
        for spec in &d.args {
            let mut bo = dev.shmem_bo(spec.bytes).expect("arg bo");
            match spec.kind {
                ArgKind::In => bo
                    .as_mut_slice()
                    .copy_from_slice(next_in.next().expect("input buffer")),
                ArgKind::Out => bo.as_mut_slice().fill(0xA5),
            }
            bo.flush();
            bos.push(bo);
        }
        let mut cmd = dev.cmd_bo().expect("cmd bo");
        let refs: Vec<&railgun::npu::Bo> = bos.iter().collect();
        let t0 = std::time::Instant::now();
        let seq = match ctx.submit(&mut cmd, &insts, &refs) {
            Ok(s) => s,
            Err(e) => {
                println!("iter {it}: submit error: {e}");
                dump_aie_status(&dev, "single", it, "submit error");
                ok_all = false;
                break;
            }
        };
        let state = ctx.wait(&cmd, seq, timeout_ms);
        let dt = t0.elapsed().as_secs_f64();
        // Query the column status BEFORE the hwctx is destroyed (destroy resets the columns).
        if dump || !matches!(state, Ok(ERT_STATE_COMPLETED)) {
            dump_aie_status(&dev, "single", it, &format!("wait {state:?}"));
        }
        drop(ctx);
        let out_idx = d.args.iter().position(|s| s.kind == ArgKind::Out).unwrap();
        bos[out_idx].flush();
        let got = d.unpack_out(bos[out_idx].as_slice());
        let bad = got.iter().zip(&want).filter(|(g, w)| g != w).count();
        let ok = matches!(state, Ok(ERT_STATE_COMPLETED)) && bad == 0;
        println!(
            "iter {it}: state={state:?} time={:.1} us useful={:.3} GOPS mismatches={bad}/{} c[0]={} want {}",
            dt * 1e6,
            ops / dt / 1e9,
            got.len(),
            got[0],
            want[0]
        );
        if !ok {
            let out = bos[out_idx].as_slice();
            report_output(out, &[((col, 2), vec![0..out.len()])]);
        }
        ok_all &= ok;
        if !ok {
            break;
        }
    }
    println!("RESULT: {}", if ok_all { "PASS" } else { "FAIL" });
    std::process::exit(if ok_all { 0 } else { 1 });
}
