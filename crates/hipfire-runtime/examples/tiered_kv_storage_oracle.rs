// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! Model-free storage/attention oracle, NOT a generation or quality gate.
//! Synthetic encoded rows stand in for post-RoPE Q8 KV. A CPU f64 softmax
//! reference independently reads those ORIGINAL absolute rows; it never reads
//! the device's packed slab. Resident pressure is explicitly synthetic too.
//! Use the external VRAM guard in addition to the managed-allocation cap.

use hip_bridge::{HipError, HipResult, HipRuntime};
use hipfire_config::emu_budget::{self, BudgetLimits};
use hipfire_runtime::tiered_kv::{ArchiveConfig, ArchiveLayout, FixedKvWindow, PinnedKvArchive};
use rdna_compute::{DType, Gpu, GpuTensor};
use serde_json::json;
use std::io::Write;

const HEADS: usize = 24;
const KV_HEADS: usize = 4;
const HD: usize = 256;
const ROW: usize = KV_HEADS * (HD / 32) * 34;
const BLOCK: usize = 128;
const LAYERS: usize = 16;
const GIB: usize = 1 << 30;

fn err(message: &str) -> HipError {
    HipError::new(0, message)
}

// Stateless absolute-row fixture, scale=1/128 (exact fp16). This independent
// expectation is also used by byte comparisons and by the CPU attention.
fn code(row: usize, side: usize, d: usize) -> i8 {
    let mut v = (row as u32).wrapping_mul(0x9e3779b9)
        ^ (side as u32).wrapping_mul(0x85ebca6b)
        ^ (d as u32).wrapping_mul(0xc2b2ae35);
    v ^= v >> 16;
    v = v.wrapping_mul(0x7feb352d);
    v ^= v >> 15;
    ((v % 255) as i16 - 127) as i8
}

fn encoded(start: usize, rows: usize, side: usize) -> Vec<u8> {
    let mut bytes = vec![0; rows * ROW];
    for t in 0..rows {
        for block in 0..KV_HEADS * HD / 32 {
            let at = t * ROW + block * 34;
            bytes[at..at + 2].copy_from_slice(&0x2000u16.to_le_bytes()); // fp16 1/128
            for d in 0..32 {
                bytes[at + 2 + d] = code(start + t, side, block * 32 + d) as u8;
            }
        }
    }
    bytes
}

fn cpu_attention(absolute: &[usize], layer: usize, q: &[f32]) -> Vec<f32> {
    let mut out = vec![0.; q.len()];
    for h in 0..HEADS {
        let kh = h / (HEADS / KV_HEADS);
        let scores: Vec<_> = absolute
            .iter()
            .map(|&t| {
                (0..HD)
                    .map(|d| q[h * HD + d] as f64 * code(t, 2 * layer, kh * HD + d) as f64 / 128.)
                    .sum::<f64>()
                    / (HD as f64).sqrt()
            })
            .collect();
        let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let weights: Vec<_> = scores.iter().map(|s| (s - max).exp()).collect();
        let sum: f64 = weights.iter().sum();
        for d in 0..HD {
            out[h * HD + d] = (absolute
                .iter()
                .zip(&weights)
                .map(|(&t, &w)| w * code(t, 2 * layer + 1, kh * HD + d) as f64 / 128.)
                .sum::<f64>()
                / sum) as f32;
        }
    }
    out
}

fn read_window(gpu: &Gpu, slab: &GpuTensor, bytes: usize, offset: usize) -> HipResult<Vec<u8>> {
    let mut out = vec![0; bytes];
    gpu.hip.memcpy_dtoh_at(&mut out, &slab.buf, offset)?;
    Ok(out)
}

fn verify(
    gpu: &mut Gpu,
    window: &FixedKvWindow,
    starts: &[usize],
    layout: ArchiveLayout,
    replacement_last: bool,
) -> HipResult<serde_json::Value> {
    let absolute: Vec<_> = starts.iter().flat_map(|s| *s..*s + BLOCK).collect();
    // Change the replaced suffix's fixture payload without changing its logical
    // positions; the reference must detect stale bytes after a divergent turn.
    let fixture_starts: Vec<_> = starts
        .iter()
        .enumerate()
        .map(|(slot, &start)| {
            start
                + if replacement_last && slot + 1 == starts.len() {
                    BLOCK
                } else {
                    0
                }
        })
        .collect();
    let fixture_rows: Vec<_> = fixture_starts.iter().flat_map(|s| *s..*s + BLOCK).collect();
    let sides: Vec<_> = window
        .sides()
        .iter()
        .map(|buffer| GpuTensor {
            // SAFETY: immutable borrowed views are used only within this function;
            // `window` owns the allocations until all GPU work below is drained.
            buf: unsafe { buffer.alias() },
            shape: vec![layout.device_side_bytes() / 4],
            dtype: DType::F32,
        })
        .collect();
    let visible = absolute.len();
    for (side, slab) in sides.iter().enumerate() {
        let got = read_window(gpu, slab, visible * ROW, 0)?;
        for (slot, &start) in fixture_starts.iter().enumerate() {
            if got[slot * BLOCK * ROW..(slot + 1) * BLOCK * ROW] != encoded(start, BLOCK, side) {
                return Err(err("archive -> packed slab is not byte-exact"));
            }
        }
        // A write into the generation reserve is an out-of-budget/history bug.
        let offset = layout.selected_capacity_tokens() * ROW;
        let tail = read_window(gpu, slab, layout.device_side_bytes() - offset, offset)?;
        if tail.iter().any(|&b| b != 0xa5) {
            return Err(err("generation reserve canary overwritten"));
        }
    }
    let q: Vec<f32> = (0..HEADS * HD)
        .map(|d| (d as f32 * 0.17).sin() * 0.25)
        .collect();
    let q_t = gpu.upload_f32(&q, &[q.len()])?;
    let out = gpu.alloc_tensor_exact(&[q.len()], DType::F32)?;
    let pos = gpu.upload_raw(&((visible - 1) as i32).to_le_bytes(), &[4])?;
    let max_seq = layout.device_side_bytes() / ROW;
    let tile = rdna_compute::attention::q8_flash_tile_size(&gpu.arch, HEADS, KV_HEADS, HD, max_seq);
    let partials = gpu.zeros(&[HEADS * max_seq.div_ceil(tile) * (HD + 2)], DType::F32)?;
    let mut max_abs = 0f32;
    for layer in [0, LAYERS - 1] {
        gpu.hip.memset(&out.buf, 0xff, q.len() * 4)?; // NaN poison detects unwritten lanes.
        gpu.attention_flash_q8_0(
            &q_t,
            &sides[2 * layer],
            &sides[2 * layer + 1],
            &out,
            &pos.buf,
            visible,
            HEADS,
            KV_HEADS,
            HD,
            max_seq,
            &partials,
        )?;
        gpu.hip.device_synchronize()?;
        let actual = gpu.download_f32(&out)?;
        let expected = cpu_attention(&fixture_rows, layer, &q);
        for (a, b) in actual.iter().zip(expected) {
            if !a.is_finite() || !b.is_finite() {
                return Err(err("non-finite attention result"));
            }
            max_abs = max_abs.max((a - b).abs());
        }
    }
    for t in [q_t, out, pos, partials] {
        gpu.free_tensor(t)?;
    }
    if max_abs > 0.0001 {
        return Err(err(&format!("CPU f64 attention mismatch: {max_abs}")));
    }
    Ok(
        json!({"visible_tokens": visible, "first_absolute": absolute[0],
        "last_absolute": absolute[visible-1], "bytes_exact": true,
        "generation_reserve_unchanged": true, "cpu_f64_attention_max_abs": max_abs}),
    )
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut history = 32768usize;
    let mut window = 8192usize;
    let mut resident_bytes = 0usize;
    let mut output = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let value = args.next().ok_or("missing argument value")?;
        match arg.as_str() {
            "--history-tokens" => history = value.parse()?,
            "--window-tokens" => window = value.parse()?,
            "--resident-reserve-bytes" => resident_bytes = value.parse()?,
            "--out" => output = Some(value),
            _ => return Err(format!("unknown argument {arg}").into()),
        }
    }
    if !history.is_multiple_of(BLOCK)
        || !window.is_multiple_of(BLOCK)
        || window < 3 * BLOCK
        || window > history.saturating_sub(BLOCK)
        || history > 262144
    {
        return Err("require block-aligned history <=262144, 384<=window<history".into());
    }
    let output = output.ok_or("--out is required (new file only)")?;
    let mut report_file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)?;
    let outcome = run_storage(history, window, resident_bytes);
    let report = match &outcome {
        Ok(report) => report.clone(),
        Err(error) => json!({"schema_version":1,"pass":false,"error":error.to_string()}),
    };
    serde_json::to_writer_pretty(&mut report_file, &report)?;
    writeln!(report_file)?;
    println!("{}", serde_json::to_string(&report)?);
    outcome.map(|_| ())
}

fn run_storage(
    history: usize,
    window: usize,
    resident_bytes: usize,
) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    if hipfire_config::developer_var("HIPFIRE_EMU").is_ok() {
        return Err(
            "this oracle installs its own 24GiB-minus-margin ledger; unset HIPFIRE_EMU".into(),
        );
    }
    // Install BEFORE Gpu creates any managed allocation. This directly uses the
    // existing ledger API; it neither spoofs arch nor adds a production knob.
    hipfire_config::devices::apply_entry_point_device_visibility()?;
    let raw = HipRuntime::load()?;
    raw.set_device(0)?;
    let bdf = raw.device_pci_bus_id(0)?;
    let cap = 24 * GIB - (512 << 20);
    let budget = emu_budget::install(
        &bdf,
        BudgetLimits {
            device_bytes: cap as u64,
            gtt_bytes: Some(12 * GIB as u64),
            shared_host_bytes: None,
            uma: false,
        },
    )?;
    let layout = ArchiveLayout::new(ArchiveConfig {
        block_tokens: BLOCK,
        row_bytes_per_side: ROW,
        full_attention_layers: LAYERS,
        max_history_tokens: history,
        window_blocks: window / BLOCK,
        generation_tokens: 8192,
        host_limit_bytes: 10 * GIB,
        device_limit_bytes: 2 * GIB,
    })?;
    let mut gpu = Gpu::init()?;
    if gpu.hip.device_pci_bus_id(gpu.device_id)? != bdf {
        return Err("ledger and initialized GPU have different PCI identities".into());
    }
    let raw_before = gpu.hip.get_vram_info()?;
    let mut peak_managed = 0u64;
    let mut snapshots = vec![];
    let mut sample = |phase: &str| {
        let snap = emu_budget::snapshot(budget);
        peak_managed = peak_managed.max(snap.committed.device_bytes);
        snapshots.push(
            json!({"phase":phase, "device_bytes":snap.committed.device_bytes,
            "host_pinned_bytes":snap.committed.host_bytes,
            "pending_device_bytes":snap.pending.device_bytes}),
        );
    };
    let resident = if resident_bytes > 0 {
        let buffer = gpu.hip.malloc(resident_bytes)?;
        gpu.hip.memset(&buffer, 0x5a, resident_bytes)?;
        Some(buffer)
    } else {
        None
    };
    let mut archive = PinnedKvArchive::new(layout)?;
    let mut window_owner = FixedKvWindow::new(std::sync::Arc::new(raw), layout)?;
    let mut source = vec![];
    for side in window_owner.sides() {
        gpu.hip.memset(side, 0xa5, layout.device_side_bytes())?;
        source.push(gpu.hip.malloc(BLOCK * ROW)?);
    }
    sample("fixed-window-allocated");
    // A cap refusal must happen in the ledger before HIP sees this oversized
    // request. A large physical card must not make the negative control pass.
    let refused = match gpu.hip.malloc(cap + 1) {
        Ok(buffer) => {
            gpu.hip.free(buffer)?;
            return Err("managed cap did not refuse".into());
        }
        Err(error) => error,
    };
    let negative_snapshot = emu_budget::snapshot(budget);
    if refused.code != 2
        || !refused
            .message
            .contains("emu capacity budget refused allocation")
        || negative_snapshot.pending.device_bytes != 0
    {
        return Err("negative control did not prove pre-allocation ledger refusal".into());
    }
    let mut growth = vec![];
    for start in (0..history).step_by(BLOCK) {
        for (side, src) in source.iter().enumerate() {
            gpu.hip.memcpy_htod(src, &encoded(start, BLOCK, side))?;
        }
        archive.append_from_device(&gpu.hip, &source.iter().collect::<Vec<_>>(), 0)?;
        if (start + BLOCK).is_multiple_of(8192) {
            sample("archive-grow");
            growth.push(
                json!({"history_tokens":start+BLOCK,"host_bytes":archive.host_bytes(),
                "fixed_device_kv_bytes":layout.device_capacity_bytes()}),
            );
            eprintln!(
                "archived {} tokens, pinned {} bytes",
                start + BLOCK,
                archive.host_bytes()
            );
        }
    }
    let starts: Vec<_> = (0..window / BLOCK).map(|i| i * BLOCK).collect();
    let first_bytes = archive.stage(&mut window_owner, &starts)?;
    let all_prefix = verify(&mut gpu, &window_owner, &starts, layout, false)?;
    let repeat_bytes = archive.stage(&mut window_owner, &starts)?;
    if repeat_bytes != 0 {
        return Err("unchanged selection retransferred history".into());
    }
    let mut sparse = starts.clone();
    *sparse.last_mut().unwrap() = history - BLOCK;
    let changed_bytes = archive.stage(&mut window_owner, &sparse)?;
    if changed_bytes != BLOCK * ROW * LAYERS * 2 {
        return Err("one-block delta copied wrong bytes".into());
    }
    let positions = layout.absolute_positions(window_owner.keys().ok_or("invalid map")?)?;
    if positions.last() != Some(&(history - 1)) {
        return Err("absolute position map lost suffix".into());
    }
    let sparse_result = verify(&mut gpu, &window_owner, &sparse, layout, false)?;
    // Refused metadata must leave published bytes/map intact.
    let before = window_owner.keys().unwrap().to_vec();
    if archive.stage(&mut window_owner, &[0, 0]).is_ok()
        || archive.stage(&mut window_owner, &[history]).is_ok()
        || window_owner.keys() != Some(before.as_slice())
    {
        return Err("invalid selection mutated live map".into());
    }
    if archive.truncate(history - 1, &mut window_owner).is_ok() {
        return Err("unaligned truncate accepted".into());
    }
    archive.truncate(history - BLOCK, &mut window_owner)?;
    if window_owner.keys().is_some() {
        return Err("truncate retained stale map".into());
    }
    // Republish this absolute block with DIFFERENT payload and a fresh revision.
    // Using identical bytes here would fail to detect stale suffix reuse.
    for (side, src) in source.iter().enumerate() {
        gpu.hip.memcpy_htod(src, &encoded(history, BLOCK, side))?;
    }
    let replacement =
        archive.append_from_device(&gpu.hip, &source.iter().collect::<Vec<_>>(), 0)?;
    let reloaded_bytes = archive.stage(&mut window_owner, &sparse)?;
    if reloaded_bytes != first_bytes || window_owner.keys().unwrap().last() != Some(&replacement) {
        return Err("divergent suffix did not fully reinstall invalidated window".into());
    }
    let restored_result = verify(&mut gpu, &window_owner, &sparse, layout, true)?;
    sample("attention-and-restage-complete");
    drop(window_owner);
    for src in source {
        gpu.hip.free(src)?;
    }
    let host_before_release = emu_budget::snapshot(budget).committed.host_bytes;
    let archive_bytes = archive.host_bytes() as u64;
    drop(archive);
    let host_after_release = emu_budget::snapshot(budget).committed.host_bytes;
    if host_before_release.checked_sub(host_after_release) != Some(archive_bytes) {
        return Err("pinned archive owner release delta mismatch".into());
    }
    if let Some(buffer) = resident {
        gpu.hip.free(buffer)?;
    }
    gpu.hip.device_synchronize()?;
    sample("owners-released");
    let snap = emu_budget::snapshot(budget);
    if snap.pending.device_bytes != 0 || snap.pending.host_bytes != 0 {
        return Err("pending capacity reservation leak".into());
    }
    let result = json!({"schema_version":1,"pass":true,"scope":"synthetic storage and attention only",
        "bdf":bdf,"arch":gpu.arch,"managed_limit_bytes":cap,"managed_peak_observed_bytes":peak_managed,
        "synthetic_resident_reserve_bytes":resident_bytes,"history_tokens":history,
        "window_tokens":window,"generation_reserve_tokens":8192,
        "host_archive_bytes":layout.host_capacity_bytes(),"device_kv_bytes":layout.device_capacity_bytes(),
        "released_archive_bytes":archive_bytes,"other_managed_host_bytes_after_release":host_after_release,
        "raw_vram_before":{"free":raw_before.0,"total":raw_before.1},
        "first_stage_bytes":first_bytes,"repeat_stage_bytes":repeat_bytes,
        "one_block_delta_bytes":changed_bytes,"reload_after_truncate_bytes":reloaded_bytes,
        "allocation_cap_negative_control":true,"invalid_selection_controls":true,
        "prefix":all_prefix,"sparse":sparse_result,"restored":restored_result,
        "growth":growth,"managed_snapshots":snapshots});
    Ok(result)
}
