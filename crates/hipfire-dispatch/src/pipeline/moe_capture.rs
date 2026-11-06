// SPDX-License-Identifier: MIT OR Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! Opt-in developer capture of real MoE prefill routes and activation rows
//! (PM/rip-loop slice 3: pinned judge inputs).
//!
//! Both hooks are off by default: with the variable unset every hook is one
//! `OnceLock` load that returns early — no allocation, no device
//! synchronization, no change to the launch sequence. When set, a hook
//! synchronizes the device and copies the named tensor to the host, so a
//! capture run is a diagnostic run only. A capture is **refused** (typed
//! error) while a hipGraph capture or Redline recording is active, and only
//! prefill calls with `rows > 1` are recorded (decode is never touched).
//!
//! # `HIPFIRE_MOE_ROUTE_DUMP=<file>` — route dump
//!
//! Binary, append-only. One record per MoE layer per prefill chunk with
//! `rows > 1`: four little-endian `u32` (`layer`, `rows`, `topk`, `position`
//! = sequence position of the chunk's first row) then `rows * topk`
//! little-endian `i32` routed expert ids in slot order (`slot = row * topk +
//! j`): the device top-k index buffer read right after the router.
//!
//! # `HIPFIRE_MOE_ACT_CAPTURE=<file>` — activation capture
//!
//! Binary, append-only, capped at 64 MiB per file (the first record that
//! would cross the cap is dropped, appending stops, one warning is printed).
//! Per record: `u32 magic 0x41435431 ('ACT1')`, `u32 site_len`, `site` bytes
//! (UTF-8), `u32 layer`, `u32 rows`, `u32 k`, `u32 position`, then `rows * k`
//! little-endian `f32`. Sites and the exact tensor each reads:
//!
//! * `moe.gate_up.x` — `[rows, hidden]` F32: the layer's rotated (and
//!   AWQ-divided where the incumbent applies it) input basis the routed
//!   gate/up producer reads — `GroupedMoeScratch::x_rot` after
//!   `rotate_input_basis` in `execute_grouped_moe` (Raon, the generic grouped
//!   path; the same tensor feeds the A4/A8/FP8 quantizer or the F16 WMMA
//!   core), `MoePrefillParams::x_rot_batch` on the sealed prefill legacy
//!   route.
//! * `moe.down.x` — `[slots, down_k]` F32 with `slots = rows * topk`, K the
//!   physical (padded) intermediate width: the rotated SwiGLU activation
//!   rows the routed down producer reads — `GroupedMoeScratch::rot` after
//!   `rotate_x_mq_batched` in `execute_grouped_moe`,
//!   `MoePrefillParams::rot_batch` on the sealed prefill legacy route.
//!
//! The sealed prefill (qwen4 / A3B `qwen35`) route variants whose producer
//! does not leave the F32 rows in those buffers (fused front, expert IU4,
//! qt44/qt53 sealed-grouped, FP8, slab) are skipped with one warning per
//! site; route dumps are unaffected by that.
//!
//! Callers tag the current MoE layer with [`set_site`] (thread-local) before
//! running it; the hooks read it.

use crate::types::DispatchError;
use rdna_compute::{Gpu, GpuTensor};
use std::cell::Cell;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{LazyLock, Mutex};

const _: () = assert!(
    cfg!(target_endian = "little"),
    "MoE capture records are little-endian host copies"
);

/// Developer env of the route dump.
pub const ROUTE_DUMP_ENV: &str = "HIPFIRE_MOE_ROUTE_DUMP";
/// Developer env of the activation capture.
pub const ACT_CAPTURE_ENV: &str = "HIPFIRE_MOE_ACT_CAPTURE";
/// Activation record magic, `'ACT1'` as a little-endian `u32`.
pub const ACT_MAGIC: u32 = 0x4143_5431;
/// Per-file cap of the activation capture.
pub const ACT_CAP_BYTES: u64 = 64 << 20;
/// Site names of the activation capture.
pub const SITE_GATE_UP: &str = "moe.gate_up.x";
pub const SITE_DOWN: &str = "moe.down.x";

static ROUTE_PATH: LazyLock<Option<String>> = LazyLock::new(|| var(ROUTE_DUMP_ENV));
static ACT_PATH: LazyLock<Option<String>> = LazyLock::new(|| var(ACT_CAPTURE_ENV));

fn var(name: &str) -> Option<String> {
    hipfire_config::developer_var(name)
        .ok()
        .filter(|path| !path.is_empty())
}

struct Sink {
    file: File,
    /// Bytes in the file (pre-existing length plus what this process wrote).
    len: u64,
}

static ROUTE_SINK: Mutex<Option<Sink>> = Mutex::new(None);
static ACT_SINK: Mutex<Option<Sink>> = Mutex::new(None);
static ACT_CAPPED: AtomicBool = AtomicBool::new(false);
static ACT_SKIPPED: AtomicU64 = AtomicU64::new(0);

thread_local! {
    static SITE: Cell<(u32, u32)> = const { Cell::new((0, 0)) };
}

/// True when `HIPFIRE_MOE_ROUTE_DUMP` is set.
pub fn route_dump_enabled() -> bool {
    ROUTE_PATH.is_some()
}

/// True when `HIPFIRE_MOE_ACT_CAPTURE` is set.
pub fn act_capture_enabled() -> bool {
    ACT_PATH.is_some()
}

/// True when either hook is armed (callers gate their own bookkeeping on it).
pub fn enabled() -> bool {
    route_dump_enabled() || act_capture_enabled()
}

/// Tag the MoE layer about to run with its `layer` index and the sequence
/// `position` of its first row. A no-op unless a hook is armed.
pub fn set_site(layer: usize, position: usize) {
    if enabled() {
        SITE.with(|site| site.set((layer as u32, position as u32)));
    }
}

fn err(message: impl std::fmt::Display) -> DispatchError {
    DispatchError::Hip(format!("moe capture: {message}"))
}

/// One route record: `u32 layer, rows, topk, position` then `ids` (the
/// `rows * topk` i32 device bytes, little-endian, slot = `row * topk + j`).
pub fn encode_route_record(
    layer: u32,
    rows: u32,
    topk: u32,
    position: u32,
    ids: &[u8],
) -> Result<Vec<u8>, DispatchError> {
    let want = rows as usize * topk as usize * 4;
    if ids.len() != want {
        return Err(err(format_args!(
            "route record holds {} id bytes, {rows} rows x top-{topk} need {want}",
            ids.len()
        )));
    }
    let mut record = Vec::with_capacity(16 + ids.len());
    for word in [layer, rows, topk, position] {
        record.extend_from_slice(&word.to_le_bytes());
    }
    record.extend_from_slice(ids);
    Ok(record)
}

/// One activation record (`ACT1`): header then `rows * k` little-endian f32
/// (`data` is the raw device byte copy).
pub fn encode_act_record(
    site: &str,
    layer: u32,
    rows: u32,
    k: u32,
    position: u32,
    data: &[u8],
) -> Result<Vec<u8>, DispatchError> {
    let want = rows as usize * k as usize * 4;
    if data.len() != want {
        return Err(err(format_args!(
            "activation record holds {} bytes, {rows} rows x K={k} need {want}",
            data.len()
        )));
    }
    let mut record = Vec::with_capacity(28 + site.len() + data.len());
    record.extend_from_slice(&ACT_MAGIC.to_le_bytes());
    record.extend_from_slice(&(site.len() as u32).to_le_bytes());
    record.extend_from_slice(site.as_bytes());
    for word in [layer, rows, k, position] {
        record.extend_from_slice(&word.to_le_bytes());
    }
    record.extend_from_slice(data);
    Ok(record)
}

fn append(
    slot: &Mutex<Option<Sink>>,
    path: &str,
    env: &str,
    record: &[u8],
    cap: Option<u64>,
) -> Result<bool, DispatchError> {
    let mut guard = slot.lock().unwrap_or_else(|poison| poison.into_inner());
    if guard.is_none() {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map_err(|e| err(format_args!("{env}={path}: {e}")))?;
        let len = file
            .metadata()
            .map_err(|e| err(format_args!("{env}={path}: {e}")))?
            .len();
        *guard = Some(Sink { file, len });
    }
    let sink = guard.as_mut().expect("opened above");
    if let Some(cap) = cap {
        if sink.len + record.len() as u64 > cap {
            return Ok(false);
        }
    }
    sink.file
        .write_all(record)
        .map_err(|e| err(format_args!("{env}={path}: {e}")))?;
    sink.len += record.len() as u64;
    Ok(true)
}

/// Host copy of the first `bytes` of `tensor` after a device synchronize.
/// Refused while a hipGraph capture or a Redline recording is active.
fn read_device(
    gpu: &Gpu,
    env: &str,
    tensor: &GpuTensor,
    bytes: usize,
) -> Result<Vec<u8>, DispatchError> {
    if gpu.graphs.capture_mode || gpu.replay.is_recording() || gpu.replay.retained_body_active() {
        return Err(err(format_args!(
            "{env} is refused while a graph capture or replay recording is active"
        )));
    }
    if tensor.numel() * tensor.dtype.size() < bytes {
        return Err(err(format_args!(
            "{env}: tensor holds {} bytes, {bytes} requested",
            tensor.numel() * tensor.dtype.size()
        )));
    }
    let mut host = vec![0u8; bytes];
    gpu.bind_thread().map_err(|e| err(e))?;
    gpu.hip.device_synchronize().map_err(|e| err(e))?;
    gpu.hip
        .memcpy_dtoh(&mut host, &tensor.buf)
        .map_err(|e| err(e))?;
    Ok(host)
}

/// Append one route record of the current [`set_site`] layer: `rows * topk`
/// i32 expert ids read from the device top-k index buffer `topk_indices`.
/// No-op unless `HIPFIRE_MOE_ROUTE_DUMP` is set and `rows > 1`.
pub fn route_dump(
    gpu: &Gpu,
    topk_indices: &GpuTensor,
    rows: usize,
    topk: usize,
) -> Result<(), DispatchError> {
    let Some(path) = ROUTE_PATH.as_deref() else {
        return Ok(());
    };
    if rows <= 1 {
        return Ok(());
    }
    let (layer, position) = SITE.with(Cell::get);
    let ids = read_device(gpu, ROUTE_DUMP_ENV, topk_indices, rows * topk * 4)?;
    let record = encode_route_record(layer, rows as u32, topk as u32, position, &ids)?;
    append(&ROUTE_SINK, path, ROUTE_DUMP_ENV, &record, None)?;
    Ok(())
}

/// Append one activation record of `site` for the current [`set_site`]
/// layer: the first `rows * k` F32 of `x`. No-op unless
/// `HIPFIRE_MOE_ACT_CAPTURE` is set and `rows > 1`. Stops appending (one
/// warning) at [`ACT_CAP_BYTES`].
pub fn act_capture(
    gpu: &Gpu,
    site: &str,
    x: &GpuTensor,
    rows: usize,
    k: usize,
) -> Result<(), DispatchError> {
    let Some(path) = ACT_PATH.as_deref() else {
        return Ok(());
    };
    if rows <= 1 || ACT_CAPPED.load(Ordering::Relaxed) {
        return Ok(());
    }
    let (layer, position) = SITE.with(Cell::get);
    let data = read_device(gpu, ACT_CAPTURE_ENV, x, rows * k * 4)?;
    let record = encode_act_record(site, layer, rows as u32, k as u32, position, &data)?;
    if !append(
        &ACT_SINK,
        path,
        ACT_CAPTURE_ENV,
        &record,
        Some(ACT_CAP_BYTES),
    )? && !ACT_CAPPED.swap(true, Ordering::Relaxed)
    {
        eprintln!(
            "{ACT_CAPTURE_ENV}: {path} reached the {} MiB cap; no further activation rows are recorded",
            ACT_CAP_BYTES >> 20
        );
    }
    Ok(())
}

/// Report (once per process, not per call) that the sealed prefill route of
/// this call keeps its producer input out of the F32 basis buffers, so
/// `site` is not captured. No-op unless `HIPFIRE_MOE_ACT_CAPTURE` is set.
pub fn act_capture_unavailable(site: &str, route: &str) {
    if ACT_PATH.is_some() && ACT_SKIPPED.fetch_add(1, Ordering::Relaxed) == 0 {
        eprintln!(
            "{ACT_CAPTURE_ENV}: {site} is not captured on the sealed prefill route `{route}` \
             (its producer input is not an F32 basis buffer); only the generic grouped path \
             and the legacy sealed route record activations"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(bytes: &[u8]) -> Vec<u32> {
        bytes
            .chunks_exact(4)
            .map(|w| u32::from_le_bytes(w.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn route_record_layout_is_header_then_slot_ordered_ids() {
        let ids: Vec<u8> = [7i32, -1, 3, 9, 0, 255]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let record = encode_route_record(5, 3, 2, 4096, &ids).unwrap();
        assert_eq!(record.len(), 16 + 24);
        assert_eq!(words(&record[..16]), [5, 3, 2, 4096]);
        assert_eq!(&record[16..], ids.as_slice());
        let slots: Vec<i32> = record[16..]
            .chunks_exact(4)
            .map(|w| i32::from_le_bytes(w.try_into().unwrap()))
            .collect();
        // slot = row * topk + j
        assert_eq!(slots[2 * 2 + 1], 255);
    }

    #[test]
    fn route_record_refuses_a_short_or_long_id_block() {
        assert!(encode_route_record(0, 2, 2, 0, &[0u8; 15]).is_err());
        assert!(encode_route_record(0, 2, 2, 0, &[0u8; 17]).is_err());
        assert!(encode_route_record(0, 2, 2, 0, &[0u8; 16]).is_ok());
    }

    #[test]
    fn act_record_layout_matches_the_act1_contract() {
        let data: Vec<u8> = [1.5f32, -2.0, 0.25, 8.0, 0.0, -0.5]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let record = encode_act_record(SITE_DOWN, 11, 2, 3, 512, &data).unwrap();
        let site = SITE_DOWN.as_bytes();
        assert_eq!(
            u32::from_le_bytes(record[0..4].try_into().unwrap()),
            0x4143_5431
        );
        assert_eq!(&record[0..4], b"1TCA");
        assert_eq!(
            u32::from_le_bytes(record[4..8].try_into().unwrap()) as usize,
            site.len()
        );
        assert_eq!(&record[8..8 + site.len()], site);
        let tail = 8 + site.len();
        assert_eq!(words(&record[tail..tail + 16]), [11, 2, 3, 512]);
        assert_eq!(&record[tail + 16..], data.as_slice());
        assert_eq!(record.len(), 8 + site.len() + 16 + 24);
    }

    #[test]
    fn act_record_refuses_a_mismatched_payload() {
        assert!(encode_act_record(SITE_GATE_UP, 0, 2, 4, 0, &[0u8; 31]).is_err());
        assert!(encode_act_record(SITE_GATE_UP, 0, 2, 4, 0, &[0u8; 32]).is_ok());
    }

    #[test]
    fn append_stops_at_the_cap_without_a_partial_record() {
        let dir = std::env::temp_dir().join(format!("moe-capture-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("cap.bin");
        let _ = std::fs::remove_file(&path);
        let slot = Mutex::new(None);
        let path_str = path.to_str().unwrap();
        assert!(append(&slot, path_str, "T", &[1u8; 40], Some(100)).unwrap());
        assert!(append(&slot, path_str, "T", &[2u8; 40], Some(100)).unwrap());
        assert!(!append(&slot, path_str, "T", &[3u8; 40], Some(100)).unwrap());
        // A smaller record still fits under the cap.
        assert!(append(&slot, path_str, "T", &[4u8; 20], Some(100)).unwrap());
        drop(slot);
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(bytes.len(), 100);
        assert_eq!(&bytes[..40], &[1u8; 40]);
        assert_eq!(&bytes[40..80], &[2u8; 40]);
        assert_eq!(&bytes[80..], &[4u8; 20]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn append_counts_a_preexisting_file_toward_the_cap() {
        let dir = std::env::temp_dir().join(format!("moe-capture-pre-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("pre.bin");
        std::fs::write(&path, [9u8; 90]).unwrap();
        let slot = Mutex::new(None);
        assert!(!append(&slot, path.to_str().unwrap(), "T", &[1u8; 20], Some(100)).unwrap());
        drop(slot);
        assert_eq!(std::fs::read(&path).unwrap().len(), 90);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
