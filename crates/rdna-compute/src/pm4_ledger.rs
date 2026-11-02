// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 Kaden Schutt <kaden@hipfire.dev>

//! Retained-PM4 dispatch ledger: a developer-only, env-gated record of what a
//! retained Redline program executes, for `pm-census ledger`.
//!
//! Retained PM4 dispatches are submitted straight to a hardware queue, so
//! rocprofv3 never sees them and a rocprof census of a decode window sees
//! almost nothing. The ledger recovers the identity of those dispatches from
//! the one place that still knows it: the captured launch list the PM4 tape is
//! lowered from.
//!
//! # Env contract (both default off; `docs/env-vars.md`)
//!
//! * `HIPFIRE_REDLINE_LEDGER=<path>` — append JSONL to `<path>`. Unset, empty
//!   or `0` disables the ledger entirely (no file, no counters; every hook is
//!   an `Option` that is `None` or returns on [`enabled`]).
//! * `HIPFIRE_REDLINE_LEDGER_TIMING=1` — only with the ledger on: prepare the
//!   per-dispatch-timestamp tape (the `HIPFIRE_REDLINE_DISPATCH_PROFILE`
//!   instrument: one GPU-clock write after every dispatch, single queue,
//!   gfx12 dispatch pacing off) and accumulate per-dispatch GPU spans. This is
//!   a **perturbing** mode, recorded as such in the program row; run it
//!   separately from any run whose throughput you report.
//!
//! # What is written, and when
//!
//! * At retain (PM4 prepare) time, once per prepared program: one `program`
//!   row and one `dispatch` row per launch (kernel symbol, code-object
//!   identity — origin, module/path, image SHA-256, image bytes — grid, block,
//!   shared memory, grid binding). This happens off the replay hot path.
//! * At replay time, only counters: one `fetch_add` for the replay count and
//!   one for the whole-program GPU span the tape's own bracketing timestamps
//!   already produce (the value `gap_timing` reads). Nothing is written and no
//!   clock is read for the ledger. The default tape is unchanged, so the
//!   default-mode ledger has no effect on GPU timing.
//! * With timing on, the per-dispatch spans the instrumented tape already
//!   returns are summed per dispatch slot.
//! * At [`flush_all`] (controller shutdown, daemon exit), one cumulative
//!   `replay_summary` row per program. A process killed before flushing
//!   leaves no summary; the ingester refuses such a ledger.
//!
//! The row schema (`ledger_version` 1) is fixed by the golden fixture
//! `tests/fixtures/pm4_ledger_v1.jsonl`, which `pm-census` parses too.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex};

use serde_json::{json, Value};

use crate::code_object::{CodeObjectArtifact, CodeObjectOrigin};

/// Schema version written in every `program` row.
pub const LEDGER_VERSION: u64 = 1;

const ENV_PATH: &str = "HIPFIRE_REDLINE_LEDGER";
const ENV_TIMING: &str = "HIPFIRE_REDLINE_LEDGER_TIMING";

#[derive(Debug, PartialEq, Eq)]
struct Config {
    path: PathBuf,
    timing: bool,
}

fn config_from(path: Option<&str>, timing: Option<&str>) -> Option<Config> {
    let path = path.map(str::trim).filter(|p| !p.is_empty() && *p != "0");
    let timing = timing.map(str::trim) == Some("1");
    match path {
        Some(path) => Some(Config {
            path: PathBuf::from(path),
            timing,
        }),
        None => {
            if timing {
                eprintln!("[redline-ledger] {ENV_TIMING}=1 ignored: {ENV_PATH} is not set");
            }
            None
        }
    }
}

static CONFIG: LazyLock<Option<Config>> = LazyLock::new(|| {
    config_from(
        hipfire_config::process_value(ENV_PATH).as_deref(),
        hipfire_config::process_value(ENV_TIMING).as_deref(),
    )
});

/// True when `HIPFIRE_REDLINE_LEDGER` names a file.
pub fn enabled() -> bool {
    CONFIG.is_some()
}

/// True when the perturbing per-dispatch-timestamp mode was requested
/// (`HIPFIRE_REDLINE_LEDGER_TIMING=1` with the ledger on).
pub fn timing_requested() -> bool {
    CONFIG.as_ref().is_some_and(|config| config.timing)
}

/// Code-object identity of one launch: what `pm-census` joins to the PM
/// inventory (`image_sha256` against the embedded object's file SHA-256).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ArtifactRow {
    /// `native_embedded` (an `include_bytes!` PM image) | `hip_file` (a hipcc
    /// cache object).
    pub(crate) origin: &'static str,
    /// Embedded module name (native images).
    pub(crate) module: Option<String>,
    /// hipcc cache path (hip objects; provenance only).
    pub(crate) path: Option<String>,
    pub(crate) image_sha256: String,
    pub(crate) image_bytes: u64,
}

impl ArtifactRow {
    pub(crate) fn of(artifact: &CodeObjectArtifact) -> Self {
        let (origin, module, path) = match artifact.origin() {
            CodeObjectOrigin::NativeEmbedded(module) => {
                ("native_embedded", Some((*module).to_owned()), None)
            }
            CodeObjectOrigin::HipFile(path) => ("hip_file", None, Some(path.display().to_string())),
        };
        Self {
            origin,
            module,
            path,
            image_sha256: artifact.id().to_hex(),
            image_bytes: artifact.image().as_bytes().len() as u64,
        }
    }
}

/// `ReplayGridBinding::PositionCeilDiv` as written to the ledger.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GridBindingRow {
    pub(crate) axis: u8,
    pub(crate) addend: u32,
    pub(crate) divisor: u32,
}

/// One retained dispatch, in recorded order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct DispatchRow {
    /// Kernel symbol without the `.kd` suffix.
    pub(crate) kernel: String,
    pub(crate) artifact: ArtifactRow,
    pub(crate) grid: [u32; 3],
    pub(crate) block: [u32; 3],
    pub(crate) shared_mem: u32,
    pub(crate) grid_binding: Option<GridBindingRow>,
}

/// Program-level facts known at prepare time.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ProgramMeta {
    pub(crate) device_ordinal: usize,
    pub(crate) arch: String,
    pub(crate) pm4_architecture: String,
    pub(crate) command_dwords: u32,
    pub(crate) queue_id: u64,
    pub(crate) queues: usize,
    pub(crate) phases: usize,
    pub(crate) sequence_hash: u64,
    /// Execution slot -> recorded index when the single-IB reorder permuted
    /// the tape; `None` = recorded order.
    pub(crate) exec_order: Option<Vec<usize>>,
    /// The tape carries a GPU-clock write after every dispatch.
    pub(crate) timing: bool,
    pub(crate) pid: u32,
    pub(crate) prepared_unix_ms: u64,
}

fn program_row(id: u64, meta: &ProgramMeta, dispatches: usize) -> Value {
    json!({
        "kind": "program",
        "ledger_version": LEDGER_VERSION,
        "program": id,
        "pid": meta.pid,
        "device_ordinal": meta.device_ordinal,
        "arch": meta.arch,
        "pm4_architecture": meta.pm4_architecture,
        "dispatches": dispatches,
        "sequence_hash": format!("{:016x}", meta.sequence_hash),
        "command_dwords": meta.command_dwords,
        "queue_id": meta.queue_id,
        "queues": meta.queues,
        "phases": meta.phases,
        "exec_order": meta.exec_order,
        "timing_mode": meta.timing,
        "perturbing": meta.timing,
        "span_layout": if meta.timing { json!("per_dispatch") } else { Value::Null },
        "prepared_unix_ms": meta.prepared_unix_ms,
    })
}

fn dispatch_row(id: u64, pid: u32, index: usize, row: &DispatchRow) -> Value {
    json!({
        "kind": "dispatch",
        "program": id,
        "pid": pid,
        "index": index,
        "kernel": row.kernel,
        "artifact": {
            "origin": row.artifact.origin,
            "module": row.artifact.module,
            "path": row.artifact.path,
            "image_sha256": row.artifact.image_sha256,
            "image_bytes": row.artifact.image_bytes,
        },
        "grid": row.grid,
        "block": row.block,
        "shared_mem": row.shared_mem,
        "grid_binding": row.grid_binding.map(|binding| json!({
            "kind": "position_ceil_div",
            "axis": binding.axis,
            "addend": binding.addend,
            "divisor": binding.divisor,
        })),
    })
}

/// The retain-time rows of one program as JSONL (one trailing newline per row).
pub(crate) fn render_program(id: u64, meta: &ProgramMeta, dispatches: &[DispatchRow]) -> String {
    let mut out = String::new();
    out.push_str(&program_row(id, meta, dispatches.len()).to_string());
    out.push('\n');
    for (index, row) in dispatches.iter().enumerate() {
        out.push_str(&dispatch_row(id, meta.pid, index, row).to_string());
        out.push('\n');
    }
    out
}

/// One cumulative `replay_summary` row.
pub(crate) fn render_summary(
    id: u64,
    pid: u32,
    replays: u64,
    gpu_span_ns: u64,
    timing: Option<(u64, &[u64], u64)>,
) -> String {
    let mut line = json!({
        "kind": "replay_summary",
        "program": id,
        "pid": pid,
        "replays": replays,
        "gpu_span_ns": gpu_span_ns,
        "timing": timing.map(|(samples, sums, shape_mismatch)| json!({
            "samples": samples,
            "span_ns_sum": sums,
            "shape_mismatch": shape_mismatch,
        })),
    })
    .to_string();
    line.push('\n');
    line
}

#[derive(Default)]
struct SpanSums {
    samples: u64,
    /// Per execution slot, summed over `samples` replays.
    sum_ns: Vec<u64>,
    /// Replays whose span vector length disagreed with the first one.
    shape_mismatch: u64,
}

/// Live counters of one retained program. Held by the prepared replay object;
/// the process-wide registry keeps a second handle for [`flush_all`].
pub struct Program {
    id: u64,
    pid: u32,
    timing: bool,
    replays: AtomicU64,
    gpu_span_ns: AtomicU64,
    spans: Mutex<SpanSums>,
    /// Replay count at the last flush (`u64::MAX` = never flushed).
    flushed: AtomicU64,
}

impl Program {
    fn new(id: u64, pid: u32, timing: bool) -> Self {
        Self {
            id,
            pid,
            timing,
            replays: AtomicU64::new(0),
            gpu_span_ns: AtomicU64::new(0),
            spans: Mutex::new(SpanSums::default()),
            flushed: AtomicU64::new(u64::MAX),
        }
    }

    /// True when the instrumented tape's per-dispatch spans should be handed
    /// to [`Self::note_replay`].
    pub(crate) fn wants_spans(&self) -> bool {
        self.timing
    }

    /// Account one completed replay. `gpu_span_ns` is the whole-program span
    /// from the tape's bracketing timestamps; `spans` the per-dispatch spans
    /// (empty unless [`Self::wants_spans`]).
    pub(crate) fn note_replay(&self, gpu_span_ns: u64, spans: &[u64]) {
        self.replays.fetch_add(1, Ordering::Relaxed);
        self.gpu_span_ns.fetch_add(gpu_span_ns, Ordering::Relaxed);
        if !self.timing || spans.is_empty() {
            return;
        }
        let mut sums = self.spans.lock().unwrap_or_else(|e| e.into_inner());
        if sums.sum_ns.is_empty() {
            sums.sum_ns = vec![0; spans.len()];
        } else if sums.sum_ns.len() != spans.len() {
            sums.shape_mismatch += 1;
            return;
        }
        for (sum, span) in sums.sum_ns.iter_mut().zip(spans) {
            *sum = sum.saturating_add(*span);
        }
        sums.samples += 1;
    }

    fn summary_line(&self) -> String {
        let replays = self.replays.load(Ordering::Relaxed);
        let gpu_span_ns = self.gpu_span_ns.load(Ordering::Relaxed);
        if self.timing {
            let sums = self.spans.lock().unwrap_or_else(|e| e.into_inner());
            render_summary(
                self.id,
                self.pid,
                replays,
                gpu_span_ns,
                Some((sums.samples, &sums.sum_ns, sums.shape_mismatch)),
            )
        } else {
            render_summary(self.id, self.pid, replays, gpu_span_ns, None)
        }
    }
}

struct Sink {
    file: Option<File>,
    failed: bool,
    next_program: u64,
    programs: Vec<Arc<Program>>,
}

static SINK: Mutex<Sink> = Mutex::new(Sink {
    file: None,
    failed: false,
    next_program: 1,
    programs: Vec::new(),
});

impl Sink {
    fn write(&mut self, config: &Config, text: &str) {
        if self.failed {
            return;
        }
        if self.file.is_none() {
            match OpenOptions::new()
                .create(true)
                .append(true)
                .open(&config.path)
            {
                Ok(file) => self.file = Some(file),
                Err(error) => {
                    eprintln!(
                        "[redline-ledger] cannot open {}: {error}; ledger disabled for this process",
                        config.path.display()
                    );
                    self.failed = true;
                    return;
                }
            }
        }
        let file = self.file.as_mut().expect("opened above");
        if let Err(error) = file.write_all(text.as_bytes()).and_then(|()| file.flush()) {
            eprintln!(
                "[redline-ledger] write to {} failed: {error}; ledger is incomplete",
                config.path.display()
            );
            self.failed = true;
        }
    }
}

/// Retain-time hook: write the program and dispatch rows and hand back the
/// replay counters. Returns `None` without touching `dispatches` when the
/// ledger is off.
pub(crate) fn register_program(
    meta: ProgramMeta,
    dispatches: impl IntoIterator<Item = DispatchRow>,
) -> Option<Arc<Program>> {
    let config = CONFIG.as_ref()?;
    let dispatches: Vec<DispatchRow> = dispatches.into_iter().collect();
    let mut sink = SINK.lock().unwrap_or_else(|e| e.into_inner());
    let id = sink.next_program;
    sink.next_program += 1;
    let text = render_program(id, &meta, &dispatches);
    sink.write(config, &text);
    let program = Arc::new(Program::new(id, meta.pid, meta.timing));
    sink.programs.push(Arc::clone(&program));
    eprintln!(
        "[redline-ledger] program {id}: {} dispatches -> {} (timing={})",
        dispatches.len(),
        config.path.display(),
        meta.timing
    );
    Some(program)
}

/// Write one cumulative `replay_summary` row for every program whose replay
/// count changed since its last flush (or that was never flushed). Idempotent;
/// a no-op when the ledger is off.
pub fn flush_all() {
    let Some(config) = CONFIG.as_ref() else {
        return;
    };
    let mut sink = SINK.lock().unwrap_or_else(|e| e.into_inner());
    let mut text = String::new();
    for program in &sink.programs {
        let replays = program.replays.load(Ordering::Relaxed);
        if program.flushed.swap(replays, Ordering::Relaxed) == replays {
            continue;
        }
        text.push_str(&program.summary_line());
    }
    if !text.is_empty() {
        sink.write(config, &text);
    }
}

/// Milliseconds since the Unix epoch (ledger provenance only).
pub(crate) fn unix_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn meta(timing: bool) -> ProgramMeta {
        ProgramMeta {
            device_ordinal: 0,
            arch: "gfx1201".into(),
            pm4_architecture: "Gfx12".into(),
            command_dwords: 777,
            queue_id: 42,
            queues: 1,
            phases: 1,
            sequence_hash: 0xdead_beef_0123_4567,
            exec_order: None,
            timing,
            pid: 4242,
            prepared_unix_ms: 1_700_000_000_000,
        }
    }

    fn pm_art() -> ArtifactRow {
        ArtifactRow {
            origin: "native_embedded",
            module: Some("gemm_mq4g256v2_pm".into()),
            path: None,
            image_sha256: "a".repeat(64),
            image_bytes: 4096,
        }
    }

    fn hip_art() -> ArtifactRow {
        ArtifactRow {
            origin: "hip_file",
            module: None,
            path: Some("/cache/hipfire/rmsnorm.hsaco".into()),
            image_sha256: "b".repeat(64),
            image_bytes: 2048,
        }
    }

    fn rows() -> Vec<DispatchRow> {
        vec![
            DispatchRow {
                kernel: "rmsnorm_f32".into(),
                artifact: hip_art(),
                grid: [64, 1, 1],
                block: [256, 1, 1],
                shared_mem: 0,
                grid_binding: None,
            },
            DispatchRow {
                kernel: "gemm_mq4g256v2_gate_up_pm".into(),
                artifact: pm_art(),
                grid: [128, 8, 1],
                block: [32, 1, 1],
                shared_mem: 16384,
                grid_binding: None,
            },
            DispatchRow {
                kernel: "attn_decode_fp8".into(),
                artifact: hip_art(),
                grid: [1, 16, 1],
                block: [128, 1, 1],
                shared_mem: 512,
                grid_binding: Some(GridBindingRow {
                    axis: 0,
                    addend: 1,
                    divisor: 64,
                }),
            },
        ]
    }

    fn parse(text: &str) -> Vec<Value> {
        text.lines()
            .map(|l| serde_json::from_str(l).expect("every ledger row is JSON"))
            .collect()
    }

    /// The golden fixture `pm-census` also parses.
    fn golden() -> String {
        let mut out = render_program(1, &meta(false), &rows());
        out.push_str(&render_summary(1, 4242, 128, 9_000_000, None));
        out
    }

    #[test]
    fn program_and_dispatch_rows_carry_identity() {
        let rows_json = parse(&render_program(7, &meta(false), &rows()));
        assert_eq!(rows_json.len(), 4);
        let head = &rows_json[0];
        assert_eq!(head["kind"], "program");
        assert_eq!(head["ledger_version"], LEDGER_VERSION);
        assert_eq!(head["program"], 7);
        assert_eq!(head["dispatches"], 3);
        assert_eq!(head["arch"], "gfx1201");
        assert_eq!(head["sequence_hash"], "deadbeef01234567");
        assert_eq!(head["timing_mode"], false);
        assert_eq!(head["perturbing"], false);
        assert!(head["span_layout"].is_null());
        assert!(head["exec_order"].is_null());

        let pm = &rows_json[2];
        assert_eq!(pm["kind"], "dispatch");
        assert_eq!(pm["program"], 7);
        assert_eq!(pm["index"], 1);
        assert_eq!(pm["kernel"], "gemm_mq4g256v2_gate_up_pm");
        assert_eq!(pm["artifact"]["origin"], "native_embedded");
        assert_eq!(pm["artifact"]["module"], "gemm_mq4g256v2_pm");
        assert!(pm["artifact"]["path"].is_null());
        assert_eq!(pm["artifact"]["image_sha256"], "a".repeat(64));
        assert_eq!(pm["artifact"]["image_bytes"], 4096);
        assert_eq!(pm["grid"], json!([128, 8, 1]));
        assert_eq!(pm["block"], json!([32, 1, 1]));
        assert_eq!(pm["shared_mem"], 16384);
        assert!(pm["grid_binding"].is_null());

        let bound = &rows_json[3];
        assert_eq!(bound["artifact"]["origin"], "hip_file");
        assert_eq!(bound["artifact"]["path"], "/cache/hipfire/rmsnorm.hsaco");
        assert_eq!(
            bound["grid_binding"],
            json!({"kind": "position_ceil_div", "axis": 0, "addend": 1, "divisor": 64})
        );
    }

    #[test]
    fn timing_program_is_labelled_perturbing() {
        let head = &parse(&render_program(1, &meta(true), &rows()))[0];
        assert_eq!(head["timing_mode"], true);
        assert_eq!(head["perturbing"], true);
        assert_eq!(head["span_layout"], "per_dispatch");
    }

    #[test]
    fn exec_order_is_recorded_when_the_tape_was_reordered() {
        let mut m = meta(false);
        m.exec_order = Some(vec![1, 0, 2]);
        let head = &parse(&render_program(1, &m, &rows()))[0];
        assert_eq!(head["exec_order"], json!([1, 0, 2]));
    }

    #[test]
    fn artifact_row_reads_admitted_identity() {
        let image: Arc<[u8]> = Arc::from(&b"\x7fELF-not-really"[..]);
        let hip = CodeObjectArtifact::hip_file(Path::new("/tmp/none.hsaco"), image.clone());
        let row = ArtifactRow::of(&hip);
        assert_eq!(row.origin, "hip_file");
        assert_eq!(row.path.as_deref(), Some("/tmp/none.hsaco"));
        assert_eq!(row.module, None);
        assert_eq!(row.image_sha256, hip.id().to_hex());
        assert_eq!(row.image_bytes, image.len() as u64);

        let native = CodeObjectArtifact::native_embedded(
            "iu4_gemm_pm",
            redline_dispatch::aql::CodeObjectBytes::from(image.clone()),
            None,
        );
        let row = ArtifactRow::of(&native);
        assert_eq!(row.origin, "native_embedded");
        assert_eq!(row.module.as_deref(), Some("iu4_gemm_pm"));
        assert_eq!(row.path, None);
        assert_eq!(row.image_sha256, native.id().to_hex());
        assert_eq!(
            row.image_sha256,
            hip.id().to_hex(),
            "same bytes, same digest"
        );
    }

    #[test]
    fn summary_rows_are_cumulative_and_labelled() {
        let plain = parse(&render_summary(3, 4242, 12, 999, None));
        assert_eq!(plain[0]["kind"], "replay_summary");
        assert_eq!(plain[0]["replays"], 12);
        assert_eq!(plain[0]["gpu_span_ns"], 999);
        assert!(plain[0]["timing"].is_null());
        let timed = parse(&render_summary(3, 4242, 2, 30, Some((2, &[10, 20], 0))));
        assert_eq!(timed[0]["timing"]["samples"], 2);
        assert_eq!(timed[0]["timing"]["span_ns_sum"], json!([10, 20]));
    }

    #[test]
    fn replay_accounting_is_counters_only() {
        let program = Program::new(1, 4242, false);
        program.note_replay(100, &[]);
        program.note_replay(250, &[7, 8]);
        let summary = parse(&program.summary_line());
        assert_eq!(summary[0]["replays"], 2);
        assert_eq!(summary[0]["gpu_span_ns"], 350);
        assert!(
            summary[0]["timing"].is_null(),
            "spans ignored without timing"
        );

        let timed = Program::new(2, 4242, true);
        timed.note_replay(30, &[10, 20]);
        timed.note_replay(33, &[11, 22]);
        timed.note_replay(1, &[1, 2, 3]);
        let summary = parse(&timed.summary_line());
        assert_eq!(summary[0]["replays"], 3);
        assert_eq!(summary[0]["timing"]["samples"], 2);
        assert_eq!(summary[0]["timing"]["span_ns_sum"], json!([21, 42]));
        assert_eq!(summary[0]["timing"]["shape_mismatch"], 1);
    }

    #[test]
    fn off_by_default() {
        assert_eq!(config_from(None, None), None);
        assert_eq!(config_from(Some(""), Some("1")), None);
        assert_eq!(config_from(Some("0"), None), None);
        assert_eq!(
            config_from(None, Some("1")),
            None,
            "timing alone enables nothing"
        );
        assert_eq!(
            config_from(Some("/tmp/l.jsonl"), None),
            Some(Config {
                path: "/tmp/l.jsonl".into(),
                timing: false
            })
        );
        assert_eq!(
            config_from(Some(" /tmp/l.jsonl "), Some("1")),
            Some(Config {
                path: "/tmp/l.jsonl".into(),
                timing: true
            })
        );
        assert_eq!(
            config_from(Some("/tmp/l.jsonl"), Some("true")),
            Some(Config {
                path: "/tmp/l.jsonl".into(),
                timing: false
            }),
            "only exact `1` requests the perturbing mode"
        );
        // The test process sets neither variable.
        assert!(!enabled());
        assert!(!timing_requested());
        assert!(register_program(meta(false), rows()).is_none());
        flush_all();
    }

    #[test]
    fn golden_fixture_matches_the_serializer() {
        // Compared as parsed rows: serde_json key order depends on whether
        // `preserve_order` is unified into the build.
        let fixture = include_str!("../tests/fixtures/pm4_ledger_v1.jsonl");
        assert_eq!(
            parse(fixture),
            parse(&golden()),
            "pm4_ledger_v1.jsonl is the schema contract shared with pm-census; \
             update both together"
        );
    }
}
