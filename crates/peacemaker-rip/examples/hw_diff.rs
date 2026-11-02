//! Emulator-vs-hardware differential for one `.rip` kernel and one case.
//!
//! WARNING: this example drives a real GPU. Hold the GPU lock first:
//!
//! ```text
//! source scripts/gpu-lock.sh && gpu_acquire "rip-hw-diff" && {
//!   cargo run -p peacemaker-rip --features hw-diff --example hw_diff -- KERNEL.rip \
//!     --case CASE.json --variant VARIANT --arch gfx1151 --device 0
//!   gpu_release
//! }
//! ```
//!
//! ```text
//! hw_diff SOURCE.rip --case CASE.json --variant VARIANT --arch gfxXXXX --device N [--json]
//! ```
//!
//! The kernel is compiled, assembled and lifted once. The CPU emulator runs
//! first via `Case::emulate` and must already pass `Case::compare`; only
//! then is the *same* ELF loaded on hardware (`module_load_data` /
//! `module_get_function`), every buffer's `initial_bytes` uploaded, the
//! pointer-patched kernarg launched once with `launch_kernel_blob`, and all
//! buffers (including output guards) downloaded. Device buffers are freed on
//! every path, including errors. Preflight refusals (launch bounds, device
//! arch mismatch) exit 2; runtime failures exit 1; exit 0 requires the
//! hardware side to pass expectations and agree with the emulator.
//!
//! The `#[cfg(test)]` module is GPU-free: it exercises only the pure bound /
//! arch checks and the public `Case` API.

use hip_bridge::HipRuntime;
use peacemaker_isa::Arch;
use peacemaker_rip::run::Case;
use serde_json::{json, Value};
use std::process::ExitCode;

const USAGE: &str = "usage: hw_diff SOURCE.rip --case CASE.json --variant VARIANT --arch gfxXXXX --device N [--json]";

const BLOCK_THREAD_LIMIT: u128 = 1024;
const GRID_DIM_LIMIT: u32 = 65536;
const TOTAL_THREAD_LIMIT: u128 = 10_000_000;

const BLOCK_BOUND_MESSAGE: &str = "block must contain 1..1024 threads";
const GRID_BOUND_MESSAGE: &str = "grid dimensions must not exceed 65536";
const TOTAL_BOUND_MESSAGE: &str = "total threads must not exceed 10000000";

/// What is known when a failure is rendered.
#[derive(Clone, Default)]
struct Partial {
    source: Option<String>,
    case: Option<String>,
    arch: Option<String>,
    device: Option<i32>,
    emulator: Option<Value>,
    device_arch: Option<String>,
    grid: Option<[u32; 3]>,
    block: Option<[u32; 3]>,
    dynamic_lds: Option<u32>,
}

struct Failure {
    exit: i32,
    stage: &'static str,
    message: String,
    partial: Box<Partial>,
}

impl Failure {
    fn usage(message: impl Into<String>) -> Self {
        Self {
            exit: 2,
            stage: "usage",
            message: message.into(),
            partial: Box::default(),
        }
    }
    fn refuse(stage: &'static str, message: String, partial: &Partial) -> Self {
        Self {
            exit: 2,
            stage,
            message,
            partial: Box::new(partial.clone()),
        }
    }
    fn runtime(stage: &'static str, message: String, partial: &Partial) -> Self {
        Self {
            exit: 1,
            stage,
            message,
            partial: Box::new(partial.clone()),
        }
    }
}

#[derive(Debug)]
struct Config {
    source: String,
    case: String,
    variant: String,
    arch: Arch,
    device: i32,
    json_mode: bool,
}

fn take_flag(args: &mut Vec<String>, flag: &str) -> bool {
    let before = args.len();
    args.retain(|a| a != flag);
    args.len() != before
}

/// `--flag VALUE` or `--flag=VALUE`.
fn take_value(args: &mut Vec<String>, flag: &str) -> Result<Option<String>, String> {
    let prefix = format!("{flag}=");
    let Some(i) = args
        .iter()
        .position(|a| a == flag || a.starts_with(&prefix))
    else {
        return Ok(None);
    };
    let arg = args.remove(i);
    if let Some(v) = arg.strip_prefix(&prefix) {
        return Ok(Some(v.to_string()));
    }
    if i < args.len() {
        return Ok(Some(args.remove(i)));
    }
    Err(format!("`{flag}` needs a value"))
}

fn parse_args(raw: &[String]) -> Result<Config, Failure> {
    let mut args = raw.to_vec();
    let json_mode = take_flag(&mut args, "--json");
    let case = take_value(&mut args, "--case").map_err(Failure::usage)?;
    let variant = take_value(&mut args, "--variant").map_err(Failure::usage)?;
    let arch = take_value(&mut args, "--arch").map_err(Failure::usage)?;
    let device = take_value(&mut args, "--device").map_err(Failure::usage)?;
    if let Some(a) = args.iter().find(|a| a.starts_with("--")) {
        return Err(Failure::usage(format!("unknown option `{a}`")));
    }
    let positionals: Vec<String> = args.into_iter().filter(|a| !a.starts_with("--")).collect();
    let [source] = positionals.as_slice() else {
        return Err(Failure::usage("expected exactly one SOURCE.rip"));
    };
    let case = case.ok_or_else(|| Failure::usage("missing `--case CASE.json`"))?;
    let variant = variant.ok_or_else(|| Failure::usage("missing `--variant VARIANT`"))?;
    if variant.is_empty() {
        return Err(Failure::usage("`--variant` must not be empty"));
    }
    let arch = arch.ok_or_else(|| Failure::usage("missing `--arch gfxXXXX`"))?;
    let arch =
        peacemaker_rip::parse_arch(&arch).map_err(|e| Failure::usage(format!("{e} ({USAGE})")))?;
    let device = device.ok_or_else(|| Failure::usage("missing `--device N`"))?;
    let device: i32 = device
        .parse()
        .map_err(|_| Failure::usage("`--device` must be a non-negative integer"))?;
    if device < 0 {
        return Err(Failure::usage("`--device` must be a non-negative integer"));
    }
    Ok(Config {
        source: source.clone(),
        case,
        variant,
        arch,
        device,
        json_mode,
    })
}

/// Preflight launch bounds; every product uses u128 so u32 inputs cannot overflow.
fn launch_bounds(block: [u32; 3], grid: [u32; 3]) -> Result<u128, String> {
    let threads = (block[0] as u128) * (block[1] as u128) * (block[2] as u128);
    if !(1..=BLOCK_THREAD_LIMIT).contains(&threads) {
        return Err(BLOCK_BOUND_MESSAGE.into());
    }
    if grid.iter().any(|d| *d > GRID_DIM_LIMIT) {
        return Err(GRID_BOUND_MESSAGE.into());
    }
    let groups = (grid[0] as u128) * (grid[1] as u128) * (grid[2] as u128);
    let total = threads * groups;
    if total > TOTAL_THREAD_LIMIT {
        return Err(TOTAL_BOUND_MESSAGE.into());
    }
    Ok(total)
}

fn check_device_arch(device: i32, actual: &str, expected: &str) -> Result<(), String> {
    if actual != expected {
        return Err(format!(
            "device {device} arch `{actual}` does not match `--arch {expected}`"
        ));
    }
    Ok(())
}

/// Run the already-validated case on hardware with the same ELF the emulator
/// used. Every allocated device buffer is freed before returning, including
/// on error. The module handle is intentionally left loaded: this bridge
/// exposes no unload, matching the existing smoke example.
fn run_hardware(
    hip: &HipRuntime,
    elf: &[u8],
    symbol: &str,
    case: &Case,
) -> Result<Vec<Vec<u8>>, String> {
    let module = hip
        .module_load_data(elf)
        .map_err(|e| format!("module_load_data: {e}"))?;
    let function = hip
        .module_get_function(&module, symbol)
        .map_err(|e| format!("module_get_function: {e}"))?;
    let mut devices: Vec<hip_bridge::DeviceBuffer> = Vec::with_capacity(case.buffers.len());
    let result = (|| {
        let mut addresses = Vec::with_capacity(case.buffers.len());
        let mut outputs = Vec::with_capacity(case.buffers.len());
        for buffer in &case.buffers {
            let initial = buffer.initial_bytes();
            let device = hip
                .malloc(initial.len().max(1))
                .map_err(|e| format!("malloc `{}`: {e}", buffer.name))?;
            devices.push(device);
            let device = devices.last().unwrap();
            if !initial.is_empty() {
                hip.memcpy_htod(device, &initial)
                    .map_err(|e| format!("upload `{}`: {e}", buffer.name))?;
            }
            addresses.push(device.as_ptr() as u64);
            outputs.push(initial);
        }
        let mut kernarg = case.kernarg_with_addresses(&addresses)?;
        // SAFETY: `kernarg` is the case's full argument block with declared
        // pointer fields relocated to live device allocations that outlive
        // this call; grid/block/LDS passed through from the same case.
        unsafe {
            hip.launch_kernel_blob(
                &function,
                case.grid,
                case.block,
                case.dynamic_lds,
                None,
                &mut kernarg,
            )
        }
        .map_err(|e| format!("launch: {e}"))?;
        hip.device_synchronize()
            .map_err(|e| format!("synchronize: {e}"))?;
        for ((buffer, device), host) in case.buffers.iter().zip(&devices).zip(&mut outputs) {
            if !host.is_empty() {
                hip.memcpy_dtoh(host, device)
                    .map_err(|e| format!("download `{}`: {e}", buffer.name))?;
            }
        }
        Ok(outputs)
    })();
    for device in devices {
        let _ = hip.free(device);
    }
    result
}

fn opt_string(value: &Option<String>) -> Value {
    value.clone().map(Value::String).unwrap_or(Value::Null)
}

fn opt_dims(value: &Option<[u32; 3]>) -> Value {
    value
        .map(|d| json!([d[0], d[1], d[2]]))
        .unwrap_or(Value::Null)
}

fn verdict(ok: bool) -> &'static str {
    if ok {
        "PASS"
    } else {
        "FAIL"
    }
}

fn buffer_details(side: &str, report: &Value) -> String {
    let mut out = String::new();
    let empty = serde_json::Map::new();
    let buffers = report
        .get("buffers")
        .and_then(Value::as_object)
        .unwrap_or(&empty);
    for (name, entry) in buffers {
        let compact = serde_json::to_string(entry).expect("a JSON value serializes");
        out.push_str(&format!("\n  {side} {name}: {compact}"));
    }
    out
}

fn dims(d: [u32; 3]) -> String {
    format!("[{}, {}, {}]", d[0], d[1], d[2])
}

fn json_failure(f: &Failure) -> Value {
    let p = &f.partial;
    json!({
        "schema": 1,
        "source": opt_string(&p.source),
        "case": opt_string(&p.case),
        "arch": opt_string(&p.arch),
        "device": p.device,
        "emulator": p.emulator.clone().unwrap_or(Value::Null),
        "hardware": Value::Null,
        "agree": false,
        "error": {"stage": f.stage, "message": f.message},
        "receipt": {
            "device_arch": opt_string(&p.device_arch),
            "grid": opt_dims(&p.grid),
            "block": opt_dims(&p.block),
            "dynamic_lds": p.dynamic_lds,
        },
    })
}

fn human_failure(f: &Failure) -> String {
    let p = &f.partial;
    let mut out = format!(
        "hw_diff {}\ncase: {}\narch: {} device: {} (device arch {})\ngrid: {} block: {} dynamic_lds: {}",
        p.source.as_deref().unwrap_or("?"),
        p.case.as_deref().unwrap_or("?"),
        p.arch.as_deref().unwrap_or("?"),
        p.device.map(|d| d.to_string()).unwrap_or("?".into()),
        p.device_arch.as_deref().unwrap_or("?"),
        p.grid.map(dims).unwrap_or("?".into()),
        p.block.map(dims).unwrap_or("?".into()),
        p.dynamic_lds.map(|n| n.to_string()).unwrap_or("?".into()),
    );
    match &p.emulator {
        Some(report) => {
            let ok = report.get("ok") == Some(&json!(true));
            out.push_str(&format!("\nemulator: {}", verdict(ok)));
            out.push_str(&buffer_details("emulator", report));
        }
        None => out.push_str("\nemulator: SKIPPED"),
    }
    out.push_str("\nhardware: SKIPPED\nagree: NO");
    out.push_str(&format!("\nerror {}: {}", f.stage, f.message));
    out
}

struct Done {
    report: Value,
    human: String,
    pass: bool,
}

fn run(raw: &[String]) -> Result<(Done, bool), Failure> {
    let cfg = parse_args(raw)?;
    let mut partial = Partial {
        source: Some(cfg.source.clone()),
        case: Some(cfg.case.clone()),
        arch: Some(cfg.arch.name().to_string()),
        device: Some(cfg.device),
        ..Partial::default()
    };
    let source = std::fs::read_to_string(&cfg.source)
        .map_err(|e| Failure::runtime("io", format!("read {}: {e}", cfg.source), &partial))?;
    let case_json = std::fs::read_to_string(&cfg.case)
        .map_err(|e| Failure::runtime("io", format!("read {}: {e}", cfg.case), &partial))?;
    let document: Value = serde_json::from_str(&case_json)
        .map_err(|e| Failure::runtime("case", format!("parse {}: {e}", cfg.case), &partial))?;
    let case = Case::parse(document).map_err(|e| Failure::refuse("case", e, &partial))?;
    partial.grid = Some(case.grid);
    partial.block = Some(case.block);
    partial.dynamic_lds = Some(case.dynamic_lds);
    launch_bounds(case.block, case.grid)
        .map_err(|message| Failure::refuse("bounds", message, &partial))?;

    // Compile, assemble and lift once; the same ELF runs on hardware below.
    let emitted = peacemaker_rip::compile_named(&source, &cfg.source, cfg.arch, &cfg.variant)
        .map_err(|e| Failure::runtime("compile", e, &partial))?;
    let symbol =
        peacemaker_rip::symbol_of(&emitted).map_err(|e| Failure::runtime("symbol", e, &partial))?;
    let elf = peacemaker_isa::native::assemble(&emitted.s_text, cfg.arch)
        .map_err(|e| Failure::runtime("assemble", e, &partial))?;
    let lifted = peacemaker_lift::lift_object(
        &elf,
        peacemaker_lift::Options {
            frontend: peacemaker_ir::inst::Frontend::Builder,
        },
    )
    .map_err(|e| Failure::runtime("lift", format!("{e:?}"), &partial))?;
    let emulator = peacemaker_ir::emu::Emulator::new(&lifted.program, &symbol)
        .map_err(|e| Failure::runtime("emulator", e.to_string(), &partial))?;
    let emulated = case
        .emulate(&emulator)
        .map_err(|e| Failure::runtime("emulator", e, &partial))?;
    let emulator_report = case.compare(&emulated);
    partial.emulator = Some(emulator_report.clone());
    if emulator_report.get("ok") != Some(&json!(true)) {
        return Err(Failure::runtime(
            "emulator",
            "emulator run did not pass expectations; hardware launch refused".into(),
            &partial,
        ));
    }

    let hip =
        HipRuntime::load().map_err(|e| Failure::runtime("hip", format!("load: {e}"), &partial))?;
    hip.set_device(cfg.device)
        .map_err(|e| Failure::runtime("hip", format!("set_device: {e}"), &partial))?;
    let device_arch = hip
        .get_arch(cfg.device)
        .map_err(|e| Failure::runtime("hip", format!("get_arch: {e}"), &partial))?;
    partial.device_arch = Some(device_arch.clone());
    check_device_arch(cfg.device, &device_arch, cfg.arch.name())
        .map_err(|message| Failure::refuse("arch", message, &partial))?;

    let hardware = run_hardware(&hip, &elf, &symbol, &case)
        .map_err(|e| Failure::runtime("hardware", e, &partial))?;
    let hardware_report = case.compare(&hardware);
    let hardware_ok = hardware_report.get("ok") == Some(&json!(true));
    let agree = case.agrees(&emulated, &hardware);
    let pass = hardware_ok && agree;

    let report = json!({
        "schema": 1,
        "source": cfg.source,
        "case": cfg.case,
        "arch": cfg.arch.name(),
        "device": cfg.device,
        "emulator": emulator_report,
        "hardware": hardware_report.clone(),
        "agree": agree,
        "error": if pass { Value::Null } else if !hardware_ok {
            json!({"stage": "hardware", "message": "hardware run did not pass expectations"})
        } else {
            json!({"stage": "agree", "message": "hardware and emulator disagree"})
        },
        "receipt": {
            "device_arch": device_arch,
            "grid": [case.grid[0], case.grid[1], case.grid[2]],
            "block": [case.block[0], case.block[1], case.block[2]],
            "dynamic_lds": case.dynamic_lds,
        },
    });
    let mut human = format!(
        "hw_diff {}\ncase: {}\narch: {} device: {} (device arch {})\ngrid: {} block: {} dynamic_lds: {}\nemulator: {}\nhardware: {}\nagree: {}",
        partial.source.as_deref().unwrap_or("?"),
        partial.case.as_deref().unwrap_or("?"),
        cfg.arch.name(),
        cfg.device,
        device_arch,
        dims(case.grid),
        dims(case.block),
        case.dynamic_lds,
        verdict(true),
        verdict(hardware_ok),
        if agree { "YES" } else { "NO" },
    );
    human.push_str(&buffer_details(
        "emulator",
        partial.emulator.as_ref().unwrap_or(&Value::Null),
    ));
    human.push_str(&buffer_details("hardware", &hardware_report));
    if !hardware_ok {
        human.push_str("\nerror hardware: hardware run did not pass expectations");
    } else if !agree {
        human.push_str("\nerror agree: hardware and emulator disagree");
    }
    Ok((
        Done {
            report,
            human,
            pass,
        },
        cfg.json_mode,
    ))
}

fn print_json(value: &Value) {
    println!(
        "{}",
        serde_json::to_string_pretty(value).expect("a JSON value serializes")
    );
}

fn main() -> ExitCode {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let json_mode = raw.iter().any(|a| a == "--json");
    match run(&raw) {
        Ok((done, json)) => {
            if json {
                print_json(&done.report);
            } else {
                println!("{}", done.human);
            }
            ExitCode::from(u8::from(!done.pass))
        }
        Err(failure) => {
            if json_mode {
                print_json(&json_failure(&failure));
            } else {
                eprintln!("{}", human_failure(&failure));
                eprintln!("\n{USAGE}");
            }
            ExitCode::from(failure.exit as u8)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn pointer_case() -> Case {
        Case::parse(json!({
            "name": "t",
            "block": [1, 1, 1],
            "grid": [1, 1, 1],
            "buffers": [
                {"name": "a", "dtype": "u32", "values": [1, 2], "expect": [1, 2]},
                {"name": "s", "dtype": "u32", "values": [9]},
            ],
            "kernargs": {"size": 24, "fields": [
                {"offset": 0, "ptr": "a"},
                {"offset": 8, "u32": 7},
                {"offset": 12, "bytes": "deadbeef"},
                {"offset": 16, "ptr": "s"},
            ]},
        }))
        .expect("test case parses")
    }

    fn float_case() -> Case {
        Case::parse(json!({
            "name": "f",
            "block": [1, 1, 1],
            "grid": [1, 1, 1],
            "buffers": [
                {"name": "o", "dtype": "f32", "values": [0.0, 0.0],
                 "expect": [1.0, 2.0], "tolerance": 0.01},
            ],
            "kernargs": {"size": 8, "fields": [{"offset": 0, "ptr": "o"}]},
        }))
        .expect("float case parses")
    }

    fn guarded_f32(values: [f32; 2]) -> Vec<u8> {
        let mut bytes = Vec::new();
        for v in values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        bytes.extend_from_slice(&[0xAA; 256]);
        bytes
    }

    #[test]
    fn kernarg_pointers_patch_only_declared_offsets() {
        let case = pointer_case();
        let blob = case
            .kernarg_with_addresses(&[0x1111_2222_3333_4444, 0x5555_6666_7777_8888])
            .expect("addresses patch");
        assert_eq!(&blob[0..8], &0x1111_2222_3333_4444u64.to_le_bytes());
        // Untouched scalar field stays byte-exact.
        assert_eq!(&blob[8..12], &7u32.to_le_bytes());
        // Bytes field stays byte-exact.
        assert_eq!(&blob[12..16], &[0xde, 0xad, 0xbe, 0xef]);
        assert_eq!(&blob[16..24], &0x5555_6666_7777_8888u64.to_le_bytes());
    }

    #[test]
    fn kernarg_address_count_must_equal_buffer_count() {
        let case = pointer_case();
        let error = case.kernarg_with_addresses(&[1]).unwrap_err();
        assert!(
            error.contains("address"),
            "count validation must name addresses: {error}"
        );
    }

    #[test]
    fn launch_bound_block_threads_exact() {
        let error = launch_bounds([64, 4, 5], [1, 1, 1]).unwrap_err();
        assert_eq!(error, "block must contain 1..1024 threads");
        let failure = Failure::refuse("bounds", error, &Partial::default());
        assert_eq!(failure.exit, 2);
        // u32 products cannot overflow the bound check.
        let error = launch_bounds([u32::MAX, u32::MAX, u32::MAX], [1, 1, 1]).unwrap_err();
        assert_eq!(error, "block must contain 1..1024 threads");
        // Exact boundary passes.
        assert_eq!(launch_bounds([32, 32, 1], [1, 1, 1]), Ok(1024));
    }

    #[test]
    fn launch_bound_grid_dims_exact() {
        for axis in 0..3 {
            let mut grid = [1, 1, 1];
            grid[axis] = 65537;
            let error = launch_bounds([1, 1, 1], grid).unwrap_err();
            assert_eq!(
                error, "grid dimensions must not exceed 65536",
                "axis {axis}"
            );
            let failure = Failure::refuse("bounds", error, &Partial::default());
            assert_eq!(failure.exit, 2);
        }
        assert_eq!(launch_bounds([1, 1, 1], [65536, 1, 1]), Ok(65536));
    }

    #[test]
    fn launch_bound_total_threads_exact() {
        let error = launch_bounds([1024, 1, 1], [100, 100, 1]).unwrap_err();
        assert_eq!(error, "total threads must not exceed 10000000");
        let failure = Failure::refuse("bounds", error, &Partial::default());
        assert_eq!(failure.exit, 2);
        // Large grid products cannot overflow either.
        let error = launch_bounds([1, 1, 1], [65536, 65536, 65536]).unwrap_err();
        assert_eq!(error, "total threads must not exceed 10000000");
        // Exact boundary passes.
        assert_eq!(launch_bounds([125, 8, 1], [1000, 10, 1]), Ok(10_000_000));
    }

    #[test]
    fn arch_mismatch_refusal_exact() {
        let message = check_device_arch(0, "gfx1151", "gfx1201").unwrap_err();
        assert_eq!(
            message,
            "device 0 arch `gfx1151` does not match `--arch gfx1201`"
        );
        let failure = Failure::refuse("arch", message, &Partial::default());
        assert_eq!(failure.exit, 2);
        assert!(check_device_arch(0, "gfx1151", "gfx1151").is_ok());
    }

    #[test]
    fn usage_errors_exit_two() {
        let failure = parse_args(&[]).unwrap_err();
        assert_eq!(failure.exit, 2);
        assert_eq!(failure.message, "expected exactly one SOURCE.rip");
        let failure = parse_args(&["k.rip".to_string()]).unwrap_err();
        assert_eq!(failure.exit, 2);
        assert_eq!(failure.message, "missing `--case CASE.json`");
    }

    #[test]
    fn float_tolerance_agrees_within_and_mismatches_beyond() {
        let case = float_case();
        let emulated = vec![guarded_f32([1.0, 2.0])];
        let close = vec![guarded_f32([1.005, 2.0])];
        assert!(case.agrees(&emulated, &close));
        let compared = case.compare(&close);
        assert_eq!(compared["ok"], json!(true));
        assert_eq!(compared["buffers"]["o"]["guard_ok"], json!(true));
        let far = vec![guarded_f32([2.0, 2.0])];
        assert!(!case.agrees(&emulated, &far));
        let compared = case.compare(&far);
        assert_eq!(compared["ok"], json!(false));
        assert_eq!(compared["buffers"]["o"]["first_mismatch"], json!(0));
    }

    #[test]
    fn guard_corruption_disagrees_without_element_mismatch() {
        let case = float_case();
        let emulated = vec![guarded_f32([1.0, 2.0])];
        let mut damaged = guarded_f32([1.0, 2.0]);
        damaged[8 + 19] = 0x00;
        let damaged = vec![damaged];
        assert!(!case.agrees(&emulated, &damaged));
        let compared = case.compare(&damaged);
        assert_eq!(compared["ok"], json!(false));
        assert_eq!(compared["buffers"]["o"]["guard_ok"], json!(false));
        assert_eq!(compared["buffers"]["o"]["first_mismatch"], Value::Null);
    }

    #[test]
    fn input_change_disagrees_and_reports_byte_offset() {
        let case = pointer_case();
        let mut out = [1u32.to_le_bytes(), 2u32.to_le_bytes()].concat();
        out.extend_from_slice(&[0xAA; 256]);
        let emulated = vec![out, 9u32.to_le_bytes().to_vec()];
        assert!(case.agrees(&emulated, &emulated));
        let mut changed = emulated.clone();
        changed[1][0] ^= 0xff;
        assert!(!case.agrees(&emulated, &changed));
        let compared = case.compare(&changed);
        assert_eq!(compared["ok"], json!(false));
        assert_eq!(compared["buffers"]["s"]["changed"], json!(true));
        assert_eq!(compared["buffers"]["s"]["first_mismatch"], json!(0));
        assert_eq!(compared["buffers"]["s"]["guard_ok"], json!(true));
    }
}
