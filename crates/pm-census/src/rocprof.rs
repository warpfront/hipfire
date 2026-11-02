// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! rocprofv3 trace reader: `trace_results.json` (the `rocprofiler-sdk-tool` document) or, as a
//! fallback, the `*kernel_trace.csv` / `*memory_copy_trace.csv` / `*agent_info.csv` set.
//!
//! Everything returned is owned data; the raw JSON document is never kept. Field meanings as
//! observed on real traces (r9700 / 7900 XTX / Strix Halo captures):
//!
//! * `dispatch_info.grid_size` is in **work-items**; [`Dispatch::grid_workgroups`] converts.
//! * Kernel symbol names carry a `.kd` suffix; it is stripped.
//! * `code_objects[].memory_size` is the byte length of the loaded **device ELF** (the AMDGPU
//!   ELF inside an offload bundle, not the bundle) — see `inventory::Inventory::attribute`.
//! * `memory_copy` operation codes index `strings.buffer_records[kind = MEMORY_COPY].operations`
//!   (`MEMORY_COPY_HOST_TO_DEVICE`, ...); the `MEMORY_COPY_` prefix is dropped from
//!   [`MemCopy::kind`] (`HOST_TO_DEVICE`), identically for CSV `Direction`.
//! * Agent `type` 1 is a CPU and 2 a GPU; the GPU `name` is the gfx target (`gfx1201`).
//!
//! CSV traces carry no code-object ids ([`KernelSymbol::code_object_id`] is `None`), no HIP API
//! records ([`Dispatch::hip_api`] is `None`) and no `memory_size`. `memory_copy_trace.csv` has no
//! byte-count column: [`MemCopy::bytes`] is read from an optional `Bytes` column and is `0`
//! when the file has none (the copy still counts, its size is unknown).

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

/// Prefix of rocprof memory-copy operation names.
const COPY_PREFIX: &str = "MEMORY_COPY_";

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TraceSource {
    pub path: String,
    /// SHA-256 (lowercase hex) of the file actually parsed (`trace_results.json`, or the
    /// `*kernel_trace.csv` for a CSV set).
    pub sha256: String,
    /// `"rocprofv3-json"` | `"rocprofv3-csv"`.
    pub format: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Agent {
    pub id: u64,
    /// `"gpu"` | `"cpu"`.
    pub kind: String,
    /// gfx target for GPUs (`gfx1201`), CPU model string otherwise.
    pub name: String,
    /// Index among agents of the same kind.
    pub ordinal: Option<u32>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CodeObject {
    pub id: u64,
    pub agent: u64,
    pub uri: String,
    /// Byte length of the loaded device ELF (`memory_size`); absent in some records.
    pub memory_size: Option<u64>,
    /// Page-rounded size of the loaded image (`load_size`), not an ELF length.
    pub load_size: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct KernelSymbol {
    pub kernel_id: u64,
    /// `None` for CSV traces.
    pub code_object_id: Option<u64>,
    /// Kernel symbol name without the `.kd` suffix.
    pub name: String,
    pub kernarg_segment_size: Option<u32>,
    pub group_segment_size: Option<u32>,
    pub private_segment_size: Option<u32>,
    pub sgpr_count: Option<u32>,
    pub vgpr_count: Option<u32>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Dispatch {
    pub kernel_id: u64,
    pub agent: u64,
    pub queue: u64,
    pub correlation: u64,
    pub start_ns: u64,
    pub end_ns: u64,
    /// Workgroup size in work-items.
    pub workgroup: [u32; 3],
    /// Grid size in work-items, as rocprof reports it.
    pub grid_items: [u32; 3],
    pub group_segment_size: u32,
    pub private_segment_size: u32,
    pub graph_exec_id: u64,
    /// The dispatch's correlation id matches a HIP API record (false = submitted outside HIP:
    /// Redline/railgun PM4 or direct HSA; None = trace has no API records).
    pub hip_api: Option<bool>,
}

impl Dispatch {
    /// Grid in workgroups: `ceil(grid_items / workgroup)` per axis.
    pub fn grid_workgroups(&self) -> [u32; 3] {
        let axis = |i: usize| self.grid_items[i].div_ceil(self.workgroup[i].max(1));
        [axis(0), axis(1), axis(2)]
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemCopy {
    pub start_ns: u64,
    pub end_ns: u64,
    /// `0` when the source (CSV without a `Bytes` column) does not carry a size.
    pub bytes: u64,
    /// `HOST_TO_DEVICE` | `DEVICE_TO_HOST` | `DEVICE_TO_DEVICE` | `HOST_TO_HOST` | ...
    pub kind: String,
    pub src_agent: u64,
    pub dst_agent: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Marker {
    pub name: String,
    pub start_ns: u64,
    pub end_ns: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Trace {
    pub source: TraceSource,
    pub agents: Vec<Agent>,
    pub code_objects: Vec<CodeObject>,
    pub kernels: Vec<KernelSymbol>,
    pub dispatches: Vec<Dispatch>,
    pub copies: Vec<MemCopy>,
    pub markers: Vec<Marker>,
}

impl Trace {
    pub fn kernel(&self, id: u64) -> Option<&KernelSymbol> {
        self.kernels.iter().find(|k| k.kernel_id == id)
    }

    pub fn code_object(&self, id: u64) -> Option<&CodeObject> {
        self.code_objects.iter().find(|c| c.id == id)
    }

    pub fn agent(&self, id: u64) -> Option<&Agent> {
        self.agents.iter().find(|a| a.id == id)
    }

    pub fn gpu_agents(&self) -> impl Iterator<Item = &Agent> {
        self.agents.iter().filter(|a| a.kind == "gpu")
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn read_file(path: &Path) -> Result<Vec<u8>, String> {
    std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))
}

/// Loads a trace from a directory (prefers `trace_results.json`, then a single `*_results.json`,
/// else the `*kernel_trace.csv` set) or from a file (`*.json`, or `*kernel_trace.csv`).
pub fn load(path: &Path) -> Result<Trace, String> {
    let meta = std::fs::metadata(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if meta.is_dir() {
        return load_dir(path);
    }
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| format!("{}: file name is not UTF-8", path.display()))?;
    if name.ends_with(".json") {
        load_json(path)
    } else if name.ends_with("kernel_trace.csv") {
        load_csv(path)
    } else {
        Err(format!(
            "{}: not a rocprofv3 *.json results file or *kernel_trace.csv",
            path.display()
        ))
    }
}

fn load_dir(dir: &Path) -> Result<Trace, String> {
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .map_err(|e| format!("{}: {e}", dir.display()))?
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().map(|t| t.is_file()).unwrap_or(false))
        .filter_map(|e| e.file_name().into_string().ok())
        .collect();
    names.sort();
    if names.iter().any(|n| n == "trace_results.json") {
        return load_json(&dir.join("trace_results.json"));
    }
    let pick = |suffix: &str| -> Result<Option<PathBuf>, String> {
        let hits: Vec<&String> = names.iter().filter(|n| n.ends_with(suffix)).collect();
        match hits.as_slice() {
            [] => Ok(None),
            [one] => Ok(Some(dir.join(one))),
            many => Err(format!(
                "{}: {} files match *{suffix}: {}",
                dir.display(),
                many.len(),
                many.iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            )),
        }
    };
    if let Some(json) = pick("_results.json")? {
        return load_json(&json);
    }
    if let Some(csv) = pick("kernel_trace.csv")? {
        return load_csv(&csv);
    }
    Err(format!(
        "{}: no trace_results.json, *_results.json or *kernel_trace.csv",
        dir.display()
    ))
}

// ---------------------------------------------------------------------------------------
// JSON
// ---------------------------------------------------------------------------------------

#[derive(Deserialize)]
struct RawFile {
    #[serde(rename = "rocprofiler-sdk-tool")]
    tool: Vec<RawTool>,
}

#[derive(Deserialize, Default)]
#[serde(default)]
struct RawTool {
    agents: Vec<RawAgent>,
    code_objects: Vec<RawCodeObject>,
    kernel_symbols: Vec<RawSymbol>,
    buffer_records: RawBuffers,
    strings: RawStrings,
}

#[derive(Deserialize, Default, Clone, Copy)]
#[serde(default)]
struct RawHandle {
    handle: u64,
}

#[derive(Deserialize, Default, Clone, Copy)]
#[serde(default)]
struct RawCorr {
    internal: u64,
}

#[derive(Deserialize, Default, Clone, Copy)]
#[serde(default)]
struct RawDim {
    x: u32,
    y: u32,
    z: u32,
}

#[derive(Deserialize, Default)]
#[serde(default)]
struct RawAgent {
    id: RawHandle,
    #[serde(rename = "type")]
    kind: u32,
    name: String,
    logical_node_type_id: Option<u32>,
}

#[derive(Deserialize, Default)]
#[serde(default)]
struct RawCodeObject {
    code_object_id: u64,
    agent_id: RawHandle,
    uri: String,
    load_size: u64,
    memory_size: Option<u64>,
}

#[derive(Deserialize, Default)]
#[serde(default)]
struct RawSymbol {
    kernel_id: u64,
    code_object_id: u64,
    kernel_name: String,
    kernarg_segment_size: Option<u32>,
    group_segment_size: Option<u32>,
    private_segment_size: Option<u32>,
    sgpr_count: Option<u32>,
    arch_vgpr_count: Option<u32>,
}

#[derive(Deserialize, Default)]
#[serde(default)]
struct RawBuffers {
    kernel_dispatch: Vec<RawDispatch>,
    memory_copy: Vec<RawCopy>,
    hip_api: Vec<RawApi>,
    marker_api: Vec<RawMarker>,
}

#[derive(Deserialize, Default)]
#[serde(default)]
struct RawApi {
    correlation_id: RawCorr,
}

#[derive(Deserialize, Default)]
#[serde(default)]
struct RawInfo {
    agent_id: RawHandle,
    queue_id: RawHandle,
    kernel_id: u64,
    private_segment_size: u32,
    group_segment_size: u32,
    workgroup_size: RawDim,
    grid_size: RawDim,
}

#[derive(Deserialize, Default)]
#[serde(default)]
struct RawDispatch {
    correlation_id: RawCorr,
    start_timestamp: u64,
    end_timestamp: u64,
    dispatch_info: RawInfo,
    graph_exec_id: u64,
}

#[derive(Deserialize, Default)]
#[serde(default)]
struct RawCopy {
    kind: u32,
    operation: u32,
    start_timestamp: u64,
    end_timestamp: u64,
    bytes: u64,
    src_agent_id: RawHandle,
    dst_agent_id: RawHandle,
}

#[derive(Deserialize, Default)]
#[serde(default)]
struct RawMarker {
    kind: u32,
    operation: u32,
    correlation_id: RawCorr,
    start_timestamp: u64,
    end_timestamp: u64,
    message: Option<String>,
    name: Option<String>,
}

#[derive(Deserialize, Default)]
#[serde(default)]
struct RawStrings {
    /// Indexed by record `kind`: the enum-name table of each tracing domain.
    buffer_records: Vec<RawKindOps>,
    marker_api: Value,
}

#[derive(Deserialize, Default)]
#[serde(default)]
struct RawKindOps {
    kind: String,
    operations: Vec<String>,
}

impl RawStrings {
    /// `strings.buffer_records[kind].operations[op]`.
    fn op_name(&self, kind: u32, op: u32) -> Option<&str> {
        self.buffer_records
            .get(kind as usize)
            .and_then(|k| k.operations.get(op as usize))
            .map(String::as_str)
    }

    /// Marker messages keyed by correlation id: `strings.marker_api` as `[{key, value}]` (the
    /// cereal map encoding) or as an object keyed by the id.
    fn marker_messages(&self) -> HashMap<u64, String> {
        let mut out = HashMap::new();
        let id_of = |v: &Value| match v {
            Value::Number(n) => n.as_u64(),
            Value::String(s) => s.parse().ok(),
            _ => None,
        };
        match &self.marker_api {
            Value::Array(items) => {
                for item in items {
                    if let (Some(key), Some(Value::String(value))) =
                        (item.get("key").and_then(id_of), item.get("value"))
                    {
                        out.insert(key, value.clone());
                    }
                }
            }
            Value::Object(map) => {
                for (key, value) in map {
                    if let (Ok(key), Value::String(value)) = (key.parse::<u64>(), value) {
                        out.insert(key, value.clone());
                    }
                }
            }
            _ => {}
        }
        out
    }
}

fn strip_kd(name: &str) -> String {
    name.strip_suffix(".kd").unwrap_or(name).to_string()
}

fn load_json(path: &Path) -> Result<Trace, String> {
    let bytes = read_file(path)?;
    let source = TraceSource {
        path: path.display().to_string(),
        sha256: sha256_hex(&bytes),
        format: "rocprofv3-json".into(),
    };
    let file: RawFile = serde_json::from_slice(&bytes)
        .map_err(|e| format!("{}: not a rocprofv3 results document: {e}", path.display()))?;
    drop(bytes);
    let mut tools = file.tool;
    if tools.len() != 1 {
        return Err(format!(
            "{}: {} rocprofiler-sdk-tool entries, expected exactly 1",
            path.display(),
            tools.len()
        ));
    }
    Ok(from_raw(tools.remove(0), source))
}

fn from_raw(tool: RawTool, source: TraceSource) -> Trace {
    let strings = &tool.strings;

    let agents = tool
        .agents
        .iter()
        .map(|a| Agent {
            id: a.id.handle,
            kind: match a.kind {
                1 => "cpu".to_string(),
                2 => "gpu".to_string(),
                n => format!("type_{n}"),
            },
            name: a.name.clone(),
            ordinal: a.logical_node_type_id,
        })
        .collect();

    let code_objects = tool
        .code_objects
        .iter()
        // id 0 with an empty uri is rocprofiler's placeholder entry.
        .filter(|c| !(c.code_object_id == 0 && c.uri.is_empty()))
        .map(|c| CodeObject {
            id: c.code_object_id,
            agent: c.agent_id.handle,
            uri: c.uri.clone(),
            memory_size: c.memory_size,
            load_size: c.load_size,
        })
        .collect();

    let kernels = tool
        .kernel_symbols
        .iter()
        .filter(|k| k.kernel_id != 0)
        .map(|k| KernelSymbol {
            kernel_id: k.kernel_id,
            code_object_id: Some(k.code_object_id),
            name: strip_kd(&k.kernel_name),
            kernarg_segment_size: k.kernarg_segment_size,
            group_segment_size: k.group_segment_size,
            private_segment_size: k.private_segment_size,
            sgpr_count: k.sgpr_count,
            vgpr_count: k.arch_vgpr_count,
        })
        .collect();

    let buffers = &tool.buffer_records;
    let hip_ids: HashSet<u64> = buffers
        .hip_api
        .iter()
        .map(|r| r.correlation_id.internal)
        .collect();
    let have_hip = !buffers.hip_api.is_empty();
    let dispatches = buffers
        .kernel_dispatch
        .iter()
        .map(|d| {
            let i = &d.dispatch_info;
            Dispatch {
                kernel_id: i.kernel_id,
                agent: i.agent_id.handle,
                queue: i.queue_id.handle,
                correlation: d.correlation_id.internal,
                start_ns: d.start_timestamp,
                end_ns: d.end_timestamp,
                workgroup: [i.workgroup_size.x, i.workgroup_size.y, i.workgroup_size.z],
                grid_items: [i.grid_size.x, i.grid_size.y, i.grid_size.z],
                group_segment_size: i.group_segment_size,
                private_segment_size: i.private_segment_size,
                graph_exec_id: d.graph_exec_id,
                hip_api: have_hip.then(|| hip_ids.contains(&d.correlation_id.internal)),
            }
        })
        .collect();

    let copies = buffers
        .memory_copy
        .iter()
        .map(|c| {
            let kind = match strings.op_name(c.kind, c.operation) {
                Some(name) => name.strip_prefix(COPY_PREFIX).unwrap_or(name).to_string(),
                None => format!("operation_{}", c.operation),
            };
            MemCopy {
                start_ns: c.start_timestamp,
                end_ns: c.end_timestamp,
                bytes: c.bytes,
                kind,
                src_agent: c.src_agent_id.handle,
                dst_agent: c.dst_agent_id.handle,
            }
        })
        .collect();

    let messages = strings.marker_messages();
    let markers = buffers
        .marker_api
        .iter()
        .map(|m| {
            let name = m
                .message
                .clone()
                .or_else(|| m.name.clone())
                .or_else(|| messages.get(&m.correlation_id.internal).cloned())
                .or_else(|| strings.op_name(m.kind, m.operation).map(str::to_string))
                .unwrap_or_else(|| format!("marker_{}", m.operation));
            Marker {
                name,
                start_ns: m.start_timestamp,
                end_ns: m.end_timestamp,
            }
        })
        .collect();

    Trace {
        source,
        agents,
        code_objects,
        kernels,
        dispatches,
        copies,
        markers,
    }
}

// ---------------------------------------------------------------------------------------
// CSV
// ---------------------------------------------------------------------------------------

/// RFC 4180 records: quoted fields may hold commas, newlines and `""` escapes. Blank lines are
/// dropped.
fn parse_csv(text: &str) -> Vec<Vec<String>> {
    let mut records = Vec::new();
    let mut record: Vec<String> = Vec::new();
    let mut field = String::new();
    let mut quoted = false;
    let mut chars = text
        .strip_prefix('\u{feff}')
        .unwrap_or(text)
        .chars()
        .peekable();
    while let Some(c) = chars.next() {
        if quoted {
            if c == '"' {
                if chars.peek() == Some(&'"') {
                    field.push('"');
                    chars.next();
                } else {
                    quoted = false;
                }
            } else {
                field.push(c);
            }
            continue;
        }
        match c {
            '"' => quoted = true,
            ',' => record.push(std::mem::take(&mut field)),
            '\r' => {}
            '\n' => {
                record.push(std::mem::take(&mut field));
                if !(record.len() == 1 && record[0].is_empty()) {
                    records.push(std::mem::take(&mut record));
                } else {
                    record.clear();
                }
            }
            c => field.push(c),
        }
    }
    if !field.is_empty() || !record.is_empty() {
        record.push(field);
        records.push(record);
    }
    records
}

struct Csv {
    path: PathBuf,
    header: HashMap<String, usize>,
    rows: Vec<Vec<String>>,
}

impl Csv {
    fn read(path: &Path) -> Result<Csv, String> {
        let bytes = read_file(path)?;
        let text = std::str::from_utf8(&bytes)
            .map_err(|e| format!("{}: not UTF-8: {e}", path.display()))?;
        let mut records = parse_csv(text).into_iter();
        let header = records
            .next()
            .ok_or_else(|| format!("{}: empty CSV", path.display()))?
            .into_iter()
            .enumerate()
            .map(|(i, name)| (name, i))
            .collect();
        Ok(Csv {
            path: path.to_path_buf(),
            header,
            rows: records.collect(),
        })
    }

    fn col(&self, name: &str) -> Result<usize, String> {
        self.header
            .get(name)
            .copied()
            .ok_or_else(|| format!("{}: missing column {name:?}", self.path.display()))
    }

    fn opt_col(&self, name: &str) -> Option<usize> {
        self.header.get(name).copied()
    }

    fn cell<'a>(&self, row: &'a [String], col: usize, line: usize) -> Result<&'a str, String> {
        row.get(col).map(String::as_str).ok_or_else(|| {
            format!(
                "{}: line {line} has {} fields, column {col} is out of range",
                self.path.display(),
                row.len()
            )
        })
    }

    fn num<T: std::str::FromStr>(
        &self,
        row: &[String],
        col: usize,
        line: usize,
    ) -> Result<T, String> {
        let s = self.cell(row, col, line)?.trim();
        s.parse().map_err(|_| {
            format!(
                "{}: line {line}: {s:?} is not a valid number",
                self.path.display()
            )
        })
    }

    /// `"Agent 4"` or `4`.
    fn agent_id(&self, row: &[String], col: usize, line: usize) -> Result<u64, String> {
        let s = self.cell(row, col, line)?.trim();
        let digits = s.strip_prefix("Agent").map(str::trim).unwrap_or(s);
        digits.parse().map_err(|_| {
            format!(
                "{}: line {line}: {s:?} is not an agent id",
                self.path.display()
            )
        })
    }
}

fn load_csv(kernel_csv: &Path) -> Result<Trace, String> {
    let name = kernel_csv
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| format!("{}: file name is not UTF-8", kernel_csv.display()))?;
    let prefix = name
        .strip_suffix("kernel_trace.csv")
        .ok_or_else(|| format!("{}: not a *kernel_trace.csv", kernel_csv.display()))?;
    let dir = kernel_csv.parent().unwrap_or_else(|| Path::new("."));
    let sibling = |suffix: &str| dir.join(format!("{prefix}{suffix}"));

    let bytes = read_file(kernel_csv)?;
    let source = TraceSource {
        path: kernel_csv.display().to_string(),
        sha256: sha256_hex(&bytes),
        format: "rocprofv3-csv".into(),
    };
    drop(bytes);

    let (kernels, dispatches) = csv_kernels(&Csv::read(kernel_csv)?)?;
    let copies_path = sibling("memory_copy_trace.csv");
    let copies = if copies_path.is_file() {
        csv_copies(&Csv::read(&copies_path)?)?
    } else {
        Vec::new()
    };
    let agents_path = sibling("agent_info.csv");
    let agents = if agents_path.is_file() {
        csv_agents(&Csv::read(&agents_path)?)?
    } else {
        Vec::new()
    };

    Ok(Trace {
        source,
        agents,
        code_objects: Vec::new(),
        kernels,
        dispatches,
        copies,
        markers: Vec::new(),
    })
}

fn csv_kernels(csv: &Csv) -> Result<(Vec<KernelSymbol>, Vec<Dispatch>), String> {
    let c_agent = csv.col("Agent_Id")?;
    let c_queue = csv.col("Queue_Id")?;
    let c_kernel = csv.col("Kernel_Id")?;
    let c_name = csv.col("Kernel_Name")?;
    let c_corr = csv.col("Correlation_Id")?;
    let c_start = csv.col("Start_Timestamp")?;
    let c_end = csv.col("End_Timestamp")?;
    let c_lds = csv.col("LDS_Block_Size")?;
    let c_scratch = csv.col("Scratch_Size")?;
    let c_vgpr = csv.col("VGPR_Count")?;
    let c_sgpr = csv.col("SGPR_Count")?;
    let c_wg = [
        csv.col("Workgroup_Size_X")?,
        csv.col("Workgroup_Size_Y")?,
        csv.col("Workgroup_Size_Z")?,
    ];
    let c_grid = [
        csv.col("Grid_Size_X")?,
        csv.col("Grid_Size_Y")?,
        csv.col("Grid_Size_Z")?,
    ];

    let mut kernels: BTreeMap<u64, KernelSymbol> = BTreeMap::new();
    let mut dispatches = Vec::with_capacity(csv.rows.len());
    for (i, row) in csv.rows.iter().enumerate() {
        let line = i + 2;
        let kernel_id: u64 = csv.num(row, c_kernel, line)?;
        let name = strip_kd(csv.cell(row, c_name, line)?);
        let lds: u32 = csv.num(row, c_lds, line)?;
        let scratch: u32 = csv.num(row, c_scratch, line)?;
        match kernels.get(&kernel_id) {
            Some(existing) if existing.name != name => {
                return Err(format!(
                    "{}: line {line}: kernel id {kernel_id} is named both {:?} and {name:?}",
                    csv.path.display(),
                    existing.name
                ));
            }
            Some(_) => {}
            None => {
                kernels.insert(
                    kernel_id,
                    KernelSymbol {
                        kernel_id,
                        code_object_id: None,
                        name,
                        kernarg_segment_size: None,
                        group_segment_size: Some(lds),
                        private_segment_size: Some(scratch),
                        sgpr_count: Some(csv.num(row, c_sgpr, line)?),
                        vgpr_count: Some(csv.num(row, c_vgpr, line)?),
                    },
                );
            }
        }
        let mut workgroup = [0u32; 3];
        let mut grid_items = [0u32; 3];
        for axis in 0..3 {
            workgroup[axis] = csv.num(row, c_wg[axis], line)?;
            grid_items[axis] = csv.num(row, c_grid[axis], line)?;
        }
        dispatches.push(Dispatch {
            kernel_id,
            agent: csv.agent_id(row, c_agent, line)?,
            queue: csv.num(row, c_queue, line)?,
            correlation: csv.num(row, c_corr, line)?,
            start_ns: csv.num(row, c_start, line)?,
            end_ns: csv.num(row, c_end, line)?,
            workgroup,
            grid_items,
            group_segment_size: lds,
            private_segment_size: scratch,
            graph_exec_id: 0,
            hip_api: None,
        });
    }
    Ok((kernels.into_values().collect(), dispatches))
}

fn csv_copies(csv: &Csv) -> Result<Vec<MemCopy>, String> {
    let c_dir = csv.col("Direction")?;
    let c_src = csv.col("Source_Agent_Id")?;
    let c_dst = csv.col("Destination_Agent_Id")?;
    let c_start = csv.col("Start_Timestamp")?;
    let c_end = csv.col("End_Timestamp")?;
    let c_bytes = csv.opt_col("Bytes");
    let mut out = Vec::with_capacity(csv.rows.len());
    for (i, row) in csv.rows.iter().enumerate() {
        let line = i + 2;
        let direction = csv.cell(row, c_dir, line)?;
        out.push(MemCopy {
            start_ns: csv.num(row, c_start, line)?,
            end_ns: csv.num(row, c_end, line)?,
            bytes: match c_bytes {
                Some(c) => csv.num(row, c, line)?,
                None => 0,
            },
            kind: direction
                .strip_prefix(COPY_PREFIX)
                .unwrap_or(direction)
                .to_string(),
            src_agent: csv.agent_id(row, c_src, line)?,
            dst_agent: csv.agent_id(row, c_dst, line)?,
        });
    }
    Ok(out)
}

fn csv_agents(csv: &Csv) -> Result<Vec<Agent>, String> {
    let c_id = csv
        .opt_col("Logical_Node_Id")
        .map_or_else(|| csv.col("Node_Id"), Ok)?;
    let c_type = csv.col("Agent_Type")?;
    let c_name = csv.col("Name")?;
    let mut per_kind: HashMap<String, u32> = HashMap::new();
    let mut out = Vec::with_capacity(csv.rows.len());
    for (i, row) in csv.rows.iter().enumerate() {
        let line = i + 2;
        let kind = csv.cell(row, c_type, line)?.trim().to_ascii_lowercase();
        let ordinal = per_kind.entry(kind.clone()).or_insert(0);
        let this = *ordinal;
        *ordinal += 1;
        out.push(Agent {
            id: csv.num(row, c_id, line)?,
            kind,
            name: csv.cell(row, c_name, line)?.to_string(),
            ordinal: Some(this),
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two agents (cpu 100, gpu 200 gfx1201), two code objects, three symbols, three
    /// dispatches (correlations 1, 2, 7), two HIP API records (1, 2), copies and markers.
    const FIXTURE: &str = r#"{
  "rocprofiler-sdk-tool": [{
    "metadata": {"pid": 1},
    "agents": [
      {"id": {"handle": 100}, "type": 1, "name": "AMD Ryzen", "logical_node_type_id": 0},
      {"id": {"handle": 200}, "type": 2, "name": "gfx1201", "logical_node_type_id": 0}
    ],
    "strings": {
      "buffer_records": [
        {"kind": "NONE", "operations": []},
        {"kind": "MEMORY_COPY", "operations": ["MEMORY_COPY_NONE", "MEMORY_COPY_HOST_TO_HOST", "MEMORY_COPY_HOST_TO_DEVICE", "MEMORY_COPY_DEVICE_TO_HOST"]},
        {"kind": "MARKER_CORE_API", "operations": ["roctxMarkA", "roctxRangePushA"]}
      ],
      "marker_api": [{"key": 9, "value": "prefill-roi"}]
    },
    "code_objects": [
      {"size": 0, "code_object_id": 0, "agent_id": {"handle": 0}, "uri": "", "load_size": 0},
      {"size": 88, "code_object_id": 5, "agent_id": {"handle": 200}, "uri": "memory://1#offset=0x1&size=46792", "load_size": 57344, "memory_size": 46792},
      {"size": 88, "code_object_id": 6, "agent_id": {"handle": 200}, "uri": "memory://1#offset=0x2&size=100", "load_size": 4096}
    ],
    "kernel_symbols": [
      {"kernel_id": 0, "code_object_id": 0, "kernel_name": ""},
      {"kernel_id": 3, "code_object_id": 5, "kernel_name": "gemm_pm_gfx1201.kd", "kernarg_segment_size": 80, "group_segment_size": 16384, "private_segment_size": 0, "sgpr_count": 128, "arch_vgpr_count": 96},
      {"kernel_id": 4, "code_object_id": 6, "kernel_name": "add_f32.kd", "kernarg_segment_size": 32, "group_segment_size": 0, "private_segment_size": 0, "sgpr_count": 128, "arch_vgpr_count": 8}
    ],
    "buffer_records": {
      "kernel_dispatch": [
        {"kind": 2, "operation": 2, "correlation_id": {"internal": 1, "external": 0}, "start_timestamp": 1000, "end_timestamp": 1500,
         "dispatch_info": {"agent_id": {"handle": 200}, "queue_id": {"handle": 1}, "kernel_id": 3, "dispatch_id": 1, "private_segment_size": 0, "group_segment_size": 16384,
                           "workgroup_size": {"x": 256, "y": 1, "z": 1}, "grid_size": {"x": 1024, "y": 4, "z": 1}},
         "graph_exec_id": 0},
        {"kind": 2, "operation": 2, "correlation_id": {"internal": 2, "external": 0}, "start_timestamp": 2000, "end_timestamp": 2100,
         "dispatch_info": {"agent_id": {"handle": 200}, "queue_id": {"handle": 1}, "kernel_id": 4, "dispatch_id": 2, "private_segment_size": 0, "group_segment_size": 0,
                           "workgroup_size": {"x": 256, "y": 1, "z": 1}, "grid_size": {"x": 300, "y": 1, "z": 1}},
         "graph_exec_id": 7},
        {"kind": 2, "operation": 2, "correlation_id": {"internal": 7, "external": 0}, "start_timestamp": 3000, "end_timestamp": 3300,
         "dispatch_info": {"agent_id": {"handle": 200}, "queue_id": {"handle": 2}, "kernel_id": 3, "dispatch_id": 3, "private_segment_size": 0, "group_segment_size": 16384,
                           "workgroup_size": {"x": 256, "y": 1, "z": 1}, "grid_size": {"x": 256, "y": 2, "z": 3}},
         "graph_exec_id": 0}
      ],
      "hip_api": [
        {"kind": 5, "operation": 1, "correlation_id": {"internal": 1, "external": 0, "ancestor": 0}, "start_timestamp": 900, "end_timestamp": 950},
        {"kind": 5, "operation": 1, "correlation_id": {"internal": 2, "external": 0, "ancestor": 0}, "start_timestamp": 1900, "end_timestamp": 1950}
      ],
      "memory_copy": [
        {"kind": 1, "operation": 2, "correlation_id": {"internal": 11}, "start_timestamp": 10, "end_timestamp": 20, "dst_agent_id": {"handle": 200}, "src_agent_id": {"handle": 100}, "bytes": 4096},
        {"kind": 1, "operation": 3, "correlation_id": {"internal": 12}, "start_timestamp": 30, "end_timestamp": 50, "dst_agent_id": {"handle": 100}, "src_agent_id": {"handle": 200}, "bytes": 8}
      ],
      "marker_api": [
        {"kind": 2, "operation": 0, "correlation_id": {"internal": 9}, "start_timestamp": 5, "end_timestamp": 6},
        {"kind": 2, "operation": 1, "correlation_id": {"internal": 10}, "start_timestamp": 7, "end_timestamp": 8}
      ]
    }
  }]
}"#;

    fn scratch_dir(tag: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("pm-census-rocprof-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write(dir: &Path, name: &str, body: &str) -> PathBuf {
        let path = dir.join(name);
        std::fs::write(&path, body).unwrap();
        path
    }

    #[test]
    fn json_fixture_grid_kd_strip_hip_api_copies_markers() {
        let dir = scratch_dir("json");
        let path = write(&dir, "trace_results.json", FIXTURE);
        let trace = load(&path).unwrap();
        assert_eq!(trace.source.format, "rocprofv3-json");
        assert_eq!(trace.source.sha256, sha256_hex(FIXTURE.as_bytes()));

        // agents
        assert_eq!(trace.agents.len(), 2);
        let gpus: Vec<_> = trace.gpu_agents().collect();
        assert_eq!(gpus.len(), 1);
        assert_eq!(gpus[0].name, "gfx1201");
        assert_eq!(gpus[0].id, 200);
        assert_eq!(trace.agent(100).unwrap().kind, "cpu");

        // placeholder entries are dropped, .kd is stripped, counts come through
        assert_eq!(trace.code_objects.len(), 2);
        assert_eq!(trace.code_object(5).unwrap().memory_size, Some(46792));
        assert_eq!(trace.code_object(5).unwrap().load_size, 57344);
        assert_eq!(trace.code_object(6).unwrap().memory_size, None);
        assert_eq!(trace.kernels.len(), 2);
        let k = trace.kernel(3).unwrap();
        assert_eq!(k.name, "gemm_pm_gfx1201");
        assert_eq!(k.code_object_id, Some(5));
        assert_eq!(k.kernarg_segment_size, Some(80));
        assert_eq!(k.group_segment_size, Some(16384));
        assert_eq!(k.vgpr_count, Some(96));
        assert_eq!(k.sgpr_count, Some(128));
        assert!(trace.kernel(0).is_none());

        // grid: work-items -> workgroups (ceil per axis)
        assert_eq!(trace.dispatches.len(), 3);
        let d0 = &trace.dispatches[0];
        assert_eq!(d0.grid_items, [1024, 4, 1]);
        assert_eq!(d0.workgroup, [256, 1, 1]);
        assert_eq!(d0.grid_workgroups(), [4, 4, 1]);
        assert_eq!(trace.dispatches[1].grid_workgroups(), [2, 1, 1]);
        assert_eq!(trace.dispatches[1].graph_exec_id, 7);
        assert_eq!(trace.dispatches[2].grid_workgroups(), [1, 2, 3]);
        assert_eq!(d0.queue, 1);
        assert_eq!(d0.agent, 200);
        assert_eq!((d0.start_ns, d0.end_ns), (1000, 1500));
        assert_eq!(d0.group_segment_size, 16384);

        // hip_api correlation: 1 and 2 are HIP launches, 7 is not
        assert_eq!(trace.dispatches[0].hip_api, Some(true));
        assert_eq!(trace.dispatches[1].hip_api, Some(true));
        assert_eq!(trace.dispatches[2].hip_api, Some(false));

        // copies: operation code -> name from the strings table, prefix dropped
        assert_eq!(trace.copies.len(), 2);
        assert_eq!(trace.copies[0].kind, "HOST_TO_DEVICE");
        assert_eq!(trace.copies[0].bytes, 4096);
        assert_eq!(
            (trace.copies[0].src_agent, trace.copies[0].dst_agent),
            (100, 200)
        );
        assert_eq!(trace.copies[1].kind, "DEVICE_TO_HOST");

        // markers: message table by correlation id, else the operation name
        assert_eq!(trace.markers.len(), 2);
        assert_eq!(trace.markers[0].name, "prefill-roi");
        assert_eq!((trace.markers[0].start_ns, trace.markers[0].end_ns), (5, 6));
        assert_eq!(trace.markers[1].name, "roctxRangePushA");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn json_without_hip_api_records_leaves_hip_api_unknown() {
        let dir = scratch_dir("nohip");
        let doc = r#"{"rocprofiler-sdk-tool": [{
            "agents": [{"id": {"handle": 1}, "type": 2, "name": "gfx1100"}],
            "kernel_symbols": [{"kernel_id": 2, "code_object_id": 1, "kernel_name": "k.kd"}],
            "buffer_records": {"kernel_dispatch": [
              {"correlation_id": {"internal": 5}, "start_timestamp": 1, "end_timestamp": 2,
               "dispatch_info": {"agent_id": {"handle": 1}, "queue_id": {"handle": 1}, "kernel_id": 2,
                 "workgroup_size": {"x": 64, "y": 1, "z": 1}, "grid_size": {"x": 64, "y": 1, "z": 1}}}
            ]}
        }]}"#;
        let path = write(&dir, "x_results.json", doc);
        let trace = load(&dir).unwrap();
        assert_eq!(trace.source.path, path.display().to_string());
        assert_eq!(trace.dispatches[0].hip_api, None);
        assert_eq!(trace.dispatches[0].grid_workgroups(), [1, 1, 1]);
        assert!(trace.copies.is_empty() && trace.markers.is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn json_with_two_tool_entries_is_rejected() {
        let dir = scratch_dir("twotool");
        let path = write(
            &dir,
            "trace_results.json",
            r#"{"rocprofiler-sdk-tool": [{}, {}]}"#,
        );
        assert!(load(&path).unwrap_err().contains("expected exactly 1"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    // Columns deliberately permuted: the reader must go by header name.
    const KERNEL_CSV: &str = "\"Kernel_Name\",\"Kind\",\"Correlation_Id\",\"Agent_Id\",\"Queue_Id\",\"Stream_Id\",\"Thread_Id\",\"Dispatch_Id\",\"Kernel_Id\",\"Start_Timestamp\",\"End_Timestamp\",\"LDS_Block_Size\",\"Scratch_Size\",\"VGPR_Count\",\"Accum_VGPR_Count\",\"SGPR_Count\",\"Workgroup_Size_X\",\"Workgroup_Size_Y\",\"Workgroup_Size_Z\",\"Grid_Size_X\",\"Grid_Size_Y\",\"Grid_Size_Z\"\n\
\"gemm_pm_gfx1201.kd\",\"KERNEL_DISPATCH\",10,\"Agent 4\",2,0,99,1,14,100,160,16384,0,96,0,128,256,1,1,1024,4,1\n\
\"tmpl<int, float>\",\"KERNEL_DISPATCH\",11,\"Agent 4\",2,0,99,2,15,200,250,0,32,8,0,128,64,1,1,100,1,1\n\
\"gemm_pm_gfx1201\",\"KERNEL_DISPATCH\",12,\"Agent 4\",2,0,99,3,14,300,330,16384,0,96,0,128,256,1,1,256,1,1\n";

    const COPY_CSV: &str = "\"Kind\",\"Direction\",\"Stream_Id\",\"Source_Agent_Id\",\"Destination_Agent_Id\",\"Correlation_Id\",\"Start_Timestamp\",\"End_Timestamp\"\n\
\"MEMORY_COPY\",\"MEMORY_COPY_HOST_TO_DEVICE\",1,\"Agent 0\",\"Agent 4\",2,10,40\n\
\"MEMORY_COPY\",\"MEMORY_COPY_DEVICE_TO_HOST\",1,\"Agent 4\",\"Agent 0\",3,50,60\n";

    const AGENT_CSV: &str = "\"Node_Id\",\"Logical_Node_Id\",\"Agent_Type\",\"Name\"\n\
0,0,\"CPU\",\"AMD Ryzen Threadripper\"\n\
1,1,\"GPU\",\"gfx1100\"\n\
2,4,\"GPU\",\"gfx1201\"\n";

    #[test]
    fn csv_set_reads_columns_by_name() {
        let dir = scratch_dir("csv");
        let kernel_path = write(&dir, "trace_kernel_trace.csv", KERNEL_CSV);
        write(&dir, "trace_memory_copy_trace.csv", COPY_CSV);
        write(&dir, "trace_agent_info.csv", AGENT_CSV);

        for entry in [dir.clone(), kernel_path.clone()] {
            let trace = load(&entry).unwrap();
            assert_eq!(trace.source.format, "rocprofv3-csv");
            assert_eq!(trace.source.sha256, sha256_hex(KERNEL_CSV.as_bytes()));
            assert!(trace.code_objects.is_empty());

            // one symbol per kernel id, .kd stripped, quoted comma kept, no code objects
            assert_eq!(trace.kernels.len(), 2);
            let k = trace.kernel(14).unwrap();
            assert_eq!(k.name, "gemm_pm_gfx1201");
            assert_eq!(k.code_object_id, None);
            assert_eq!(k.kernarg_segment_size, None);
            assert_eq!(k.group_segment_size, Some(16384));
            assert_eq!(k.vgpr_count, Some(96));
            assert_eq!(k.sgpr_count, Some(128));
            assert_eq!(trace.kernel(15).unwrap().name, "tmpl<int, float>");
            assert_eq!(trace.kernel(15).unwrap().private_segment_size, Some(32));

            assert_eq!(trace.dispatches.len(), 3);
            let d = &trace.dispatches[0];
            assert_eq!(
                (d.kernel_id, d.agent, d.queue, d.correlation),
                (14, 4, 2, 10)
            );
            assert_eq!((d.start_ns, d.end_ns), (100, 160));
            assert_eq!(d.workgroup, [256, 1, 1]);
            assert_eq!(d.grid_items, [1024, 4, 1]);
            assert_eq!(d.grid_workgroups(), [4, 4, 1]);
            assert_eq!(d.hip_api, None);
            assert_eq!(d.graph_exec_id, 0);
            assert_eq!(trace.dispatches[1].grid_workgroups(), [2, 1, 1]);

            assert_eq!(trace.copies.len(), 2);
            assert_eq!(trace.copies[0].kind, "HOST_TO_DEVICE");
            assert_eq!(
                (trace.copies[0].src_agent, trace.copies[0].dst_agent),
                (0, 4)
            );
            assert_eq!(trace.copies[0].bytes, 0);
            assert_eq!(trace.copies[1].kind, "DEVICE_TO_HOST");

            let gpus: Vec<_> = trace.gpu_agents().collect();
            assert_eq!(gpus.len(), 2);
            assert_eq!(
                (gpus[1].id, gpus[1].name.as_str(), gpus[1].ordinal),
                (4, "gfx1201", Some(1))
            );
            assert_eq!(trace.agent(0).unwrap().kind, "cpu");
            assert!(trace.markers.is_empty());
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn csv_kernel_id_with_two_names_is_rejected() {
        let dir = scratch_dir("csvdup");
        let bad = KERNEL_CSV.replace(
            "\"gemm_pm_gfx1201\",\"KERNEL_DISPATCH\",12",
            "\"other\",\"KERNEL_DISPATCH\",12",
        );
        let path = write(&dir, "trace_kernel_trace.csv", &bad);
        assert!(load(&path).unwrap_err().contains("kernel id 14"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn csv_missing_column_is_named() {
        let dir = scratch_dir("csvmissing");
        let bad = KERNEL_CSV.replace("\"Grid_Size_Y\"", "\"Grid_Y\"");
        let path = write(&dir, "trace_kernel_trace.csv", &bad);
        assert!(load(&path).unwrap_err().contains("Grid_Size_Y"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn csv_parser_handles_quotes_and_blank_lines() {
        let rows = parse_csv("a,\"b,\"\"c\"\"\",d\r\n\n1,2,3");
        assert_eq!(
            rows,
            vec![
                vec!["a".to_string(), "b,\"c\"".to_string(), "d".to_string()],
                vec!["1".to_string(), "2".to_string(), "3".to_string()],
            ]
        );
    }

    #[test]
    fn dir_prefers_trace_results_json_over_csv() {
        let dir = scratch_dir("prefer");
        write(&dir, "trace_kernel_trace.csv", KERNEL_CSV);
        write(&dir, "trace_results.json", FIXTURE);
        assert_eq!(load(&dir).unwrap().source.format, "rocprofv3-json");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn empty_dir_and_unknown_file_are_errors() {
        let dir = scratch_dir("empty");
        assert!(load(&dir).unwrap_err().contains("no trace_results.json"));
        let other = write(&dir, "notes.txt", "x");
        assert!(load(&other).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Real r9700 capture (only when the file exists on this machine).
    #[test]
    fn real_r9700_trace_if_present() {
        let path = Path::new(
            "/home/kaden/qcal/release-0.4.3/fn-prefill-profile-002/r9700/prof/trace_results.json",
        );
        if !path.exists() {
            return;
        }
        let trace = load(path).unwrap();
        assert!(!trace.dispatches.is_empty());
        assert!(!trace.copies.is_empty());
        assert!(trace.gpu_agents().any(|a| a.name == "gfx1201"));
        assert!(trace.kernels.iter().all(|k| !k.name.ends_with(".kd")));
        // A PM object in this trace is loaded from the daemon binary at the device ELF size.
        let mq6 = trace
            .kernels
            .iter()
            .find(|k| k.name == "gemm_mq6g256v2_mmq_i8_pm_set_gfx1201")
            .expect("MQ6 PM kernel in the trace");
        let co = trace.code_object(mq6.code_object_id.unwrap()).unwrap();
        assert_eq!(co.memory_size, Some(46792));
        assert!(trace.dispatches.iter().all(|d| d.hip_api.is_none()));
        // every dispatch resolves to a kernel symbol
        assert!(trace
            .dispatches
            .iter()
            .all(|d| trace.kernel(d.kernel_id).is_some()));
    }

    /// Real traces: the CSV set agrees with the JSON on dispatch count (same capture).
    #[test]
    fn real_r9700_csv_matches_json_if_present() {
        let dir = Path::new("/home/kaden/qcal/release-0.4.3/fn-prefill-profile-002/r9700/prof");
        let csv = dir.join("trace_kernel_trace.csv");
        if !dir.join("trace_results.json").exists() || !csv.exists() {
            return;
        }
        let json = load(&dir.join("trace_results.json")).unwrap();
        let csv = load(&csv).unwrap();
        assert_eq!(json.dispatches.len(), csv.dispatches.len());
        assert_eq!(json.copies.len(), csv.copies.len());
        assert_eq!(
            json.dispatches.first().map(|d| d.grid_workgroups()),
            csv.dispatches.first().map(|d| d.grid_workgroups())
        );
    }
}
