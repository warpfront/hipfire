//! CPU-only execution of `.rip` kernels with a strict JSON case.
//!
//! `run FILE [--case FILE.json] [--arch LIST|all] [--variant LIST] [--json]
//! [--dump OUT]` uses the same target discovery as `check`; the default case
//! path is FILE with its extension replaced by `.json`. Each target is compiled,
//! natively assembled, lifted and admitted by `Emulator::new` before execution.
//! No GPU or ROCm tools are involved. Workgroups run with x varying fastest,
//! then y, then z; barriers rendezvous within a workgroup, not across the grid.
//!
//! # Case schema
//!
//! ```json
//! {"name":"n=2","block":[64,1,1],"grid":[1,1,1],"dynamic_lds":0,
//!  "buffers":[{"name":"a","dtype":"f32","values":[0.5,-1.5]},
//!             {"name":"c","dtype":"f32","expect":[0.5,-1.5],"tolerance":0.0}],
//!  "kernargs":{"size":32,"fields":[{"offset":0,"ptr":"a"},
//!                                      {"offset":24,"u32":2}]}}
//! ```
//!
//! `name`, `block`, `grid`, `buffers`, and `kernargs` are required;
//! `dynamic_lds` defaults to zero. Dimensions are three positive u32 values,
//! and the block contains at most 1024 threads. Buffers require a unique name,
//! dtype, and `values` or `expect`. Dtypes are `u8/u16/u32/u64`,
//! `i8/i16/i32/i64`, `f16/bf16/f32/f64` (numeric arrays) and `bytes` (an
//! even-length hexadecimal string without a prefix). All encoding is little
//! endian. Half types use the existing emulator's round-to-nearest-even
//! conversions; half is already present in the workspace dependency graph.
//! Integer values must fit their declared types and floats must remain finite
//! after conversion. `values` and `expect`, when both given, must have identical
//! byte lengths. An output (a buffer with `expect`) without `values` starts as
//! 0xAA bytes. Every output has a 256-byte 0xAA guard; inputs must remain unchanged.
//! Floating output tolerance defaults to exact bytes; an explicit nonnegative
//! finite `tolerance` permits `abs(actual - expected) <= tolerance * max(1,
//! abs(expected))`, with finite actual values only. Tolerance is for floats only.
//!
//! `kernargs.size` and `kernargs.fields` are required. Each field has `offset`
//! and exactly one of `ptr` (a buffer name), `u8/u16/u32/u64/i32/i64/f32/f64`,
//! or `bytes` (hex). Fields may not overlap or exceed size; gaps are zeroed.
//! Unknown keys at every level, invalid references and length mismatches are
//! rejected before any kernel runs. Root `source` and `description` are reserved
//! for dumps: source, if supplied, must equal `"dump"`, and description is text.
//!
//! # Reports and dumps
//!
//! A single target produces schema 1 `{file, arch, variant, case, ok, buffers,
//! error}` (plus `schema:1`). `buffers` maps names to `{kind:"input", changed,
//! first_mismatch?}` (byte offset), or `{kind:"output", first_mismatch?,
//! max_abs_diff?, guard_ok}` (element index, byte index for `bytes`).
//! `max_abs_diff` is present for floating outputs and null for nonfinite results.
//! Errors are null or `{stage,message}`; emulator refusal messages retain their
//! mnemonic. Multiple targets use `{schema:1,ok,results:[reports...]}`. Human
//! output has one line per buffer and a final verdict. Exit 0 requires all pass.
//!
//! `--dump OUT` requires one target and writes a replayable case after successful
//! execution, even if expectations fail. Output buffers become exact `bytes`
//! with their pre-run `values` and observed post-run `expect`, without tolerance.
//! The file has `source:"dump"` and a warning: **a dump captures behavior, not
//! correctness**. A replay compares the captured bytes exactly. Guard damage or
//! changed inputs still fails; dumping does not hide memory corruption.

use peacemaker_ir::emu::{convert, Emulator, Launch, Memory};
use peacemaker_ir::inst::Frontend;
use peacemaker_isa::Arch;
use serde_json::{json, Map, Value};

const GUARD: usize = 256;
const KERNARG_BASE: u64 = 0x8000_0000;
const BUFFER_STRIDE: u64 = 0x1000_0000;

/// The report and, when execution completed, its behavior-capture case.
pub struct Outcome {
    pub report: Value,
    pub dump: Option<Value>,
}

/// Construct a schema-1 failure, including failures before target discovery.
pub fn failure(
    file: &str,
    arch: Option<&str>,
    variant: Option<&str>,
    case: Option<&str>,
    stage: &str,
    message: &str,
) -> Value {
    json!({"schema":1,"file":file,"arch":arch,"variant":variant,"case":case,
           "ok":false,"buffers":{},"error":{"stage":stage,"message":message}})
}

#[derive(Clone, Copy)]
enum Dtype {
    U8,
    U16,
    U32,
    U64,
    I8,
    I16,
    I32,
    I64,
    F16,
    Bf16,
    F32,
    F64,
    Bytes,
}
impl Dtype {
    fn parse(name: &str) -> Result<Self, String> {
        Ok(match name {
            "u8" => Self::U8,
            "u16" => Self::U16,
            "u32" => Self::U32,
            "u64" => Self::U64,
            "i8" => Self::I8,
            "i16" => Self::I16,
            "i32" => Self::I32,
            "i64" => Self::I64,
            "f16" => Self::F16,
            "bf16" => Self::Bf16,
            "f32" => Self::F32,
            "f64" => Self::F64,
            "bytes" => Self::Bytes,
            _ => return Err(format!("unknown dtype `{name}`")),
        })
    }
    fn width(self) -> usize {
        match self {
            Self::U8 | Self::I8 | Self::Bytes => 1,
            Self::U16 | Self::I16 | Self::F16 | Self::Bf16 => 2,
            Self::U32 | Self::I32 | Self::F32 => 4,
            Self::U64 | Self::I64 | Self::F64 => 8,
        }
    }
    fn float(self) -> bool {
        matches!(self, Self::F16 | Self::Bf16 | Self::F32 | Self::F64)
    }
    fn float_value(self, bytes: &[u8]) -> f64 {
        match self {
            Self::F16 => f32::from_bits(convert::f16_to_f32(u16::from_le_bytes(
                bytes.try_into().unwrap(),
            ))) as f64,
            Self::Bf16 => {
                f32::from_bits(u32::from(u16::from_le_bytes(bytes.try_into().unwrap())) << 16)
                    as f64
            }
            Self::F32 => f32::from_le_bytes(bytes.try_into().unwrap()) as f64,
            Self::F64 => f64::from_le_bytes(bytes.try_into().unwrap()),
            _ => unreachable!("only floating dtypes are decoded as floats"),
        }
    }
    fn encode(self, value: &Value) -> Result<Vec<u8>, String> {
        if matches!(self, Self::Bytes) {
            return unhex(value);
        }
        let values = value.as_array().ok_or("typed values must be an array")?;
        self.encode_values(values)
    }
    fn encode_values(self, values: &[Value]) -> Result<Vec<u8>, String> {
        let mut bytes = Vec::with_capacity(values.len() * self.width());
        for v in values {
            macro_rules! unsigned {
                ($ty:ty) => {{
                    let n = v.as_u64().ok_or("expected unsigned integer")?;
                    let n = <$ty>::try_from(n).map_err(|_| "unsigned integer out of range")?;
                    bytes.extend_from_slice(&n.to_le_bytes());
                }};
            }
            macro_rules! signed {
                ($ty:ty) => {{
                    let n = v.as_i64().ok_or("expected signed integer")?;
                    let n = <$ty>::try_from(n).map_err(|_| "signed integer out of range")?;
                    bytes.extend_from_slice(&n.to_le_bytes());
                }};
            }
            match self {
                Self::U8 => unsigned!(u8),
                Self::U16 => unsigned!(u16),
                Self::U32 => unsigned!(u32),
                Self::U64 => unsigned!(u64),
                Self::I8 => signed!(i8),
                Self::I16 => signed!(i16),
                Self::I32 => signed!(i32),
                Self::I64 => signed!(i64),
                Self::F16 | Self::Bf16 | Self::F32 | Self::F64 => {
                    let n = v
                        .as_f64()
                        .filter(|n| n.is_finite())
                        .ok_or("expected finite float")?;
                    let start = bytes.len();
                    match self {
                        Self::F16 => bytes.extend_from_slice(
                            &convert::f32_to_f16((n as f32).to_bits()).to_le_bytes(),
                        ),
                        Self::Bf16 => bytes.extend_from_slice(
                            &convert::f32_to_bf16((n as f32).to_bits()).to_le_bytes(),
                        ),
                        Self::F32 => bytes.extend_from_slice(&(n as f32).to_le_bytes()),
                        Self::F64 => bytes.extend_from_slice(&n.to_le_bytes()),
                        _ => unreachable!(),
                    }
                    if !self.float_value(&bytes[start..]).is_finite() {
                        return Err("float out of range for dtype".into());
                    }
                }
                Self::Bytes => unreachable!(),
            }
        }
        Ok(bytes)
    }
}

/// A validated buffer; dtype, expectations and tolerance remain case-owned.
pub struct Buffer {
    pub name: String,
    dtype: Dtype,
    initial: Vec<u8>,
    expect: Option<Vec<u8>>,
    tolerance: Option<f64>,
}
impl Buffer {
    /// Initial contents, including the 256-byte 0xAA guard for outputs.
    pub fn initial_bytes(&self) -> Vec<u8> {
        let guard = if self.expect.is_some() { GUARD } else { 0 };
        let mut bytes = Vec::with_capacity(self.initial.len() + guard);
        bytes.extend_from_slice(&self.initial);
        if self.expect.is_some() {
            bytes.extend_from_slice(&[0xAA; GUARD]);
        }
        bytes
    }
}

/// The strict `run` case, shared with the hardware differential example.
pub struct Case {
    name: String,
    pub block: [u32; 3],
    pub grid: [u32; 3],
    pub dynamic_lds: u32,
    pub buffers: Vec<Buffer>,
    kernarg: Vec<u8>,
    pointers: Vec<(usize, usize)>,
    document: Value,
}

fn object<'a>(v: &'a Value, keys: &[&str]) -> Result<&'a Map<String, Value>, String> {
    let m = v.as_object().ok_or("expected an object")?;
    for key in m.keys() {
        if !keys.contains(&key.as_str()) {
            return Err(format!("unknown field `{key}`"));
        }
    }
    Ok(m)
}
fn required<'a>(m: &'a Map<String, Value>, key: &str) -> Result<&'a Value, String> {
    m.get(key).ok_or_else(|| format!("missing field `{key}`"))
}
fn text(v: &Value) -> Result<&str, String> {
    v.as_str().ok_or_else(|| "expected a string".into())
}
fn array(v: &Value) -> Result<&[Value], String> {
    v.as_array()
        .map(Vec::as_slice)
        .ok_or_else(|| "expected an array".into())
}
fn uint(v: &Value) -> Result<u32, String> {
    v.as_u64()
        .and_then(|n| u32::try_from(n).ok())
        .ok_or_else(|| "expected a u32".into())
}
fn size(v: &Value) -> Result<usize, String> {
    v.as_u64()
        .and_then(|n| usize::try_from(n).ok())
        .ok_or_else(|| "expected a byte size/offset".into())
}
fn dimensions(v: &Value) -> Result<[u32; 3], String> {
    let a = array(v)?;
    if a.len() != 3 {
        return Err("dimensions must contain exactly three values".into());
    }
    let dims = [uint(&a[0])?, uint(&a[1])?, uint(&a[2])?];
    if dims.contains(&0) {
        return Err("dimensions must be positive".into());
    }
    Ok(dims)
}
fn unhex(v: &Value) -> Result<Vec<u8>, String> {
    let s = text(v)?;
    if s.len() % 2 != 0 {
        return Err("hex string must have even length".into());
    }
    fn digit(b: u8) -> Result<u8, String> {
        match b {
            b'0'..=b'9' => Ok(b - b'0'),
            b'a'..=b'f' => Ok(b - b'a' + 10),
            b'A'..=b'F' => Ok(b - b'A' + 10),
            _ => Err("invalid hexadecimal digit".into()),
        }
    }
    s.as_bytes()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| Ok(digit(b[0])? * 16 + digit(b[1])?))
        .collect()
}
fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        s.push(DIGITS[(b >> 4) as usize] as char);
        s.push(DIGITS[(b & 15) as usize] as char);
    }
    s
}
fn base_of(index: usize) -> Result<u64, String> {
    u64::try_from(index)
        .ok()
        .and_then(|i| i.checked_mul(BUFFER_STRIDE))
        .and_then(|n| n.checked_add(0x1_0000_0000))
        .ok_or_else(|| "buffer address overflow".into())
}

impl Case {
    /// Validate and encode a case using the same schema as `run`.
    pub fn parse(document: Value) -> Result<Self, String> {
        let root = object(
            &document,
            &[
                "name",
                "block",
                "grid",
                "dynamic_lds",
                "buffers",
                "kernargs",
                "source",
                "description",
            ],
        )?;
        if let Some(source) = root.get("source") {
            if text(source)? != "dump" {
                return Err("source must be `dump`".into());
            }
        }
        if let Some(description) = root.get("description") {
            text(description)?;
        }
        let name = text(required(root, "name")?)?.to_string();
        let block = dimensions(required(root, "block")?)?;
        let grid = dimensions(required(root, "grid")?)?;
        let threads = block.iter().try_fold(1u32, |n, d| n.checked_mul(*d));
        if threads.is_none_or(|n| n > 1024) {
            return Err("block must contain 1..1024 threads".into());
        }
        let dynamic_lds = root.get("dynamic_lds").map(uint).transpose()?.unwrap_or(0);
        let mut buffers: Vec<Buffer> = Vec::new();
        for entry in array(required(root, "buffers")?)? {
            let m = object(entry, &["name", "dtype", "values", "expect", "tolerance"])?;
            let name = text(required(m, "name")?)?.to_string();
            if buffers.iter().any(|b| b.name == name) {
                return Err(format!("duplicate buffer name `{name}`"));
            }
            let dtype_name = text(required(m, "dtype")?)?;
            let dtype = Dtype::parse(dtype_name)?;
            let encode = |v| {
                dtype
                    .encode(v)
                    .map_err(|e| format!("buffer `{name}` ({dtype_name}): {e}"))
            };
            let expect = m.get("expect").map(encode).transpose()?;
            let initial = match m.get("values") {
                Some(v) => encode(v)?,
                None => vec![
                    0xAA;
                    expect
                        .as_ref()
                        .ok_or_else(|| format!("buffer `{name}` requires values or expect"))?
                        .len()
                ],
            };
            if expect.as_ref().is_some_and(|e| e.len() != initial.len()) {
                return Err(format!("buffer `{name}` values/expect length mismatch"));
            }
            if initial
                .len()
                .checked_add(GUARD)
                .is_none_or(|n| n as u64 >= BUFFER_STRIDE)
            {
                return Err(format!("buffer `{name}` exceeds address slot"));
            }
            let tolerance = m
                .get("tolerance")
                .map(|v| {
                    v.as_f64()
                        .filter(|t| t.is_finite() && *t >= 0.0)
                        .ok_or("tolerance must be finite and nonnegative")
                })
                .transpose()?;
            if tolerance.is_some() && (!dtype.float() || expect.is_none()) {
                return Err(format!(
                    "buffer `{name}` tolerance requires a floating output"
                ));
            }
            buffers.push(Buffer {
                name,
                dtype,
                initial,
                expect,
                tolerance,
            });
        }
        let args = object(required(root, "kernargs")?, &["size", "fields"])?;
        let arg_size = size(required(args, "size")?)?;
        // Validate every field before allocating or copying the argument block.
        let mut fields: Vec<(usize, Vec<u8>)> = Vec::new();
        let mut pointers = Vec::new();
        for entry in array(required(args, "fields")?)? {
            let m = object(
                entry,
                &[
                    "offset", "ptr", "u32", "u64", "i32", "i64", "u16", "u8", "f32", "f64", "bytes",
                ],
            )?;
            let offset = size(required(m, "offset")?)?;
            if m.len() != 2 {
                return Err("kernarg field requires exactly one payload".into());
            }
            let (kind, v) = m.iter().find(|(key, _)| key.as_str() != "offset").unwrap();
            let bytes = if kind == "ptr" {
                let target = text(v)?;
                let index = buffers
                    .iter()
                    .position(|b| b.name == target)
                    .ok_or_else(|| format!("unknown ptr target `{target}`"))?;
                pointers.push((offset, index));
                base_of(index)?.to_le_bytes().to_vec()
            } else if kind == "bytes" {
                unhex(v)?
            } else {
                Dtype::parse(kind)?.encode_values(std::slice::from_ref(v))?
            };
            let end = offset
                .checked_add(bytes.len())
                .filter(|end| *end <= arg_size)
                .ok_or("kernarg field exceeds size")?;
            if fields
                .iter()
                .any(|(start, b)| offset < *start + b.len() && *start < end)
            {
                return Err("overlapping kernarg fields".into());
            }
            fields.push((offset, bytes));
        }
        let mut kernarg = Vec::new();
        kernarg
            .try_reserve_exact(arg_size)
            .map_err(|e| format!("kernarg size: {e}"))?;
        kernarg.resize(arg_size, 0);
        for (offset, bytes) in fields {
            kernarg[offset..offset + bytes.len()].copy_from_slice(&bytes);
        }
        Ok(Self {
            name,
            block,
            grid,
            dynamic_lds,
            buffers,
            kernarg,
            pointers,
            document,
        })
    }
    /// Relocate declared pointer fields only; scalar bytes and gaps are retained.
    /// Addresses are in `buffers` order, including buffers without a pointer field.
    pub fn kernarg_with_addresses(&self, addresses: &[u64]) -> Result<Vec<u8>, String> {
        if addresses.len() != self.buffers.len() {
            return Err("device address count must equal buffer count".into());
        }
        let mut blob = self.kernarg.clone();
        for &(offset, index) in &self.pointers {
            blob[offset..offset + 8].copy_from_slice(&addresses[index].to_le_bytes());
        }
        Ok(blob)
    }

    /// Execute with exactly `run`'s memory layout and workgroup order.
    /// Returned buffers retain output guards and are in case order.
    pub fn emulate(&self, emulator: &Emulator<'_>) -> Result<Vec<Vec<u8>>, String> {
        let memory = self
            .execute_memory(emulator, self.kernarg.clone())
            .map_err(|(_, message)| message)?;
        Ok(memory
            .regions
            .into_iter()
            .take(self.buffers.len())
            .map(|region| region.bytes)
            .collect())
    }

    fn execute_memory(
        &self,
        emulator: &Emulator<'_>,
        kernarg: Vec<u8>,
    ) -> Result<Memory, (&'static str, String)> {
        let mut memory = Memory::default();
        for (index, buffer) in self.buffers.iter().enumerate() {
            let base = base_of(index).map_err(|e| ("memory", e))?;
            memory
                .map(buffer.name.clone(), base, buffer.initial_bytes())
                .map_err(|e| ("memory", e))?;
        }
        memory
            .map("__rip_kernarg".into(), KERNARG_BASE, kernarg)
            .map_err(|e| ("memory", e))?;
        for z in 0..self.grid[2] {
            for y in 0..self.grid[1] {
                for x in 0..self.grid[0] {
                    let launch = Launch {
                        kernarg_ptr: KERNARG_BASE,
                        workgroup: [x, y, z],
                        block: self.block,
                        dynamic_lds: self.dynamic_lds,
                    };
                    emulator
                        .run_workgroup(&launch, &mut memory)
                        .map_err(|e| ("emulator", format!("workgroup [{x},{y},{z}]: {e}")))?;
                }
            }
        }
        Ok(memory)
    }

    /// Compare an execution's buffers against case expectations and guards.
    pub fn compare(&self, actual: &[Vec<u8>]) -> Value {
        let mut ok = actual.len() == self.buffers.len();
        let mut reports = Map::new();
        for (buffer, bytes) in self.buffers.iter().zip(actual) {
            let (passes, mut report) = compare(buffer, bytes);
            ok &= passes;
            let fields = report.as_object_mut().unwrap();
            fields.entry("first_mismatch").or_insert(Value::Null);
            fields.entry("max_abs_diff").or_insert(Value::Null);
            fields.entry("guard_ok").or_insert(json!(true));
            reports.insert(buffer.name.clone(), report);
        }
        json!({"ok":ok,"buffers":reports})
    }

    /// Compare two executions directly using each output's case tolerance.
    /// Integer/byte buffers and inputs are exact; output guards must be intact.
    pub fn agrees(&self, emulator: &[Vec<u8>], hardware: &[Vec<u8>]) -> bool {
        emulator.len() == self.buffers.len()
            && hardware.len() == self.buffers.len()
            && self.buffers.iter().zip(emulator).zip(hardware).all(
                |((buffer, emulated), actual)| {
                    if buffer.expect.is_none() {
                        return emulated == actual && actual.len() == buffer.initial.len();
                    }
                    let size = buffer.initial.len();
                    emulated.len() == size + GUARD
                        && actual.len() == size + GUARD
                        && emulated[size..].iter().all(|b| *b == 0xAA)
                        && compare_expected(buffer, actual, Some(&emulated[..size])).0
                },
            )
    }
}

fn compare(buffer: &Buffer, actual: &[u8]) -> (bool, Value) {
    compare_expected(buffer, actual, buffer.expect.as_deref())
}

fn compare_expected(buffer: &Buffer, actual: &[u8], expected: Option<&[u8]>) -> (bool, Value) {
    let Some(expected) = expected else {
        let first = actual
            .iter()
            .zip(&buffer.initial)
            .position(|(a, b)| a != b)
            .or_else(|| {
                (actual.len() != buffer.initial.len())
                    .then_some(actual.len().min(buffer.initial.len()))
            });
        let mut report = json!({"kind":"input", "changed":first.is_some()});
        if let Some(offset) = first {
            report["first_mismatch"] = json!(offset);
        }
        return (first.is_none(), report);
    };
    let guard_ok = actual.len() == expected.len() + GUARD
        && actual[expected.len()..].iter().all(|b| *b == 0xAA);
    let mut first = (actual.len() < expected.len()).then_some(actual.len() / buffer.dtype.width());
    let mut max_diff = 0.0f64;
    let mut finite = true;
    for (i, (a, b)) in actual[..actual.len().min(expected.len())]
        .chunks_exact(buffer.dtype.width())
        .zip(expected.chunks_exact(buffer.dtype.width()))
        .enumerate()
    {
        let matches = if buffer.dtype.float() {
            let got = buffer.dtype.float_value(a);
            let want = buffer.dtype.float_value(b);
            let diff = (got - want).abs();
            finite &= diff.is_finite();
            max_diff = max_diff.max(diff);
            match buffer.tolerance {
                Some(t) => got.is_finite() && diff <= t * want.abs().max(1.0),
                None => a == b,
            }
        } else {
            a == b
        };
        if !matches && first.is_none() {
            first = Some(i);
        }
    }
    let mut report = json!({"kind":"output", "guard_ok":guard_ok});
    if let Some(i) = first {
        report["first_mismatch"] = json!(i);
    }
    if buffer.dtype.float() {
        report["max_abs_diff"] = if finite { json!(max_diff) } else { Value::Null };
    }
    (first.is_none() && guard_ok, report)
}

/// Execute one selected target; every failure is returned as a structured report.
/// `capture_dump` requests an exact behavior capture, not a correctness oracle.
pub fn execute(
    source: &str,
    file: &str,
    arch: Arch,
    variant: &str,
    case_json: &str,
    capture_dump: bool,
) -> Outcome {
    let parsed = serde_json::from_str::<Value>(case_json)
        .map_err(|e| e.to_string())
        .and_then(Case::parse);
    let mut case = match parsed {
        Ok(case) => case,
        Err(message) => {
            return Outcome {
                report: failure(
                    file,
                    Some(arch.name()),
                    Some(variant),
                    None,
                    "case",
                    &message,
                ),
                dump: None,
            }
        }
    };
    let failed = |stage: &str, message: String| Outcome {
        report: failure(
            file,
            Some(arch.name()),
            Some(variant),
            Some(&case.name),
            stage,
            &message,
        ),
        dump: None,
    };
    let emitted = match crate::compile_named(source, file, arch, variant) {
        Ok(v) => v,
        Err(e) => return failed("compile", e),
    };
    let symbol = match crate::symbol_of(&emitted) {
        Ok(v) => v,
        Err(e) => return failed("symbol", e),
    };
    let elf = match peacemaker_isa::native::assemble(&emitted.s_text, arch) {
        Ok(v) => v,
        Err(e) => return failed("assemble", e),
    };
    let lifted = match peacemaker_lift::lift_object(
        &elf,
        peacemaker_lift::Options {
            frontend: Frontend::Builder,
        },
    ) {
        Ok(v) => v,
        Err(e) => return failed("lift", format!("{e:?}")),
    };
    let emulator = match Emulator::new(&lifted.program, &symbol) {
        Ok(v) => v,
        Err(e) => return failed("emulator", e),
    };
    let kernarg = std::mem::take(&mut case.kernarg);
    let memory = match case.execute_memory(&emulator, kernarg) {
        Ok(memory) => memory,
        Err((stage, message)) => return failed(stage, message),
    };
    let mut ok = true;
    let mut reports = Map::new();
    let mut dump = capture_dump.then_some(case.document);
    if let Some(document) = &mut dump {
        document["source"] = json!("dump");
        document["description"] = json!("Captures observed behavior, not correctness. Output expectations are exact post-run bytes.");
    }
    for (index, buffer) in case.buffers.iter().enumerate() {
        let actual = &memory.regions[index].bytes;
        let (passes, report) = compare(buffer, actual);
        ok &= passes;
        reports.insert(buffer.name.clone(), report);
        if let Some(document) = dump.as_mut().filter(|_| buffer.expect.is_some()) {
            let entry = &mut document["buffers"][index];
            entry["dtype"] = json!("bytes");
            entry["values"] = json!(hex(&buffer.initial));
            entry["expect"] = json!(hex(&actual[..buffer.initial.len()]));
            entry.as_object_mut().unwrap().remove("tolerance");
        }
    }
    Outcome {
        report: json!({"schema":1,"file":file,"arch":arch.name(),"variant":variant,"case":case.name,"ok":ok,"buffers":reports,"error":null}),
        dump,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_guard_damage_fails_without_an_element_mismatch() {
        let buffer = Buffer {
            name: "out".into(),
            dtype: Dtype::U32,
            initial: vec![0xAA; 4],
            expect: Some(7u32.to_le_bytes().to_vec()),
            tolerance: None,
        };
        let mut actual = 7u32.to_le_bytes().to_vec();
        actual.extend_from_slice(&[0xAA; GUARD]);
        actual[4 + 19] = 0;
        let (ok, report) = compare(&buffer, &actual);
        assert!(!ok);
        assert_eq!(report["guard_ok"], false);
        assert!(report.get("first_mismatch").is_none());
    }

    #[test]
    fn dtype_roundtrips_and_range_checks() {
        for name in [
            "u8", "u16", "u32", "u64", "i8", "i16", "i32", "i64", "f16", "bf16", "f32", "f64",
        ] {
            let dtype = Dtype::parse(name).unwrap();
            let bytes = dtype.encode(&json!([1])).unwrap();
            assert_eq!(bytes.len(), dtype.width());
            if dtype.float() {
                assert_eq!(dtype.float_value(&bytes), 1.0);
            }
        }
        assert!(Dtype::U8.encode(&json!([256])).is_err());
        assert!(Dtype::I8.encode(&json!([-129])).is_err());
        assert!(Dtype::F16.encode(&json!([1e30])).is_err());
    }
}
