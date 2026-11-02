// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Runs the arbitrary-operand gfx1100 divide probe (`gfx1100_divide_cases_probe`, peacemaker-isa
//! example) on caller-supplied `{numerator_bits, denominator_bits}` pairs and writes the complete
//! copied divide-DAG result for each (B4-C divide audit). It is independent of the gate-domain
//! probe `gfx1100_epilogue_intrinsic_observe`, whose code object, manifest and schema it leaves alone.
//!
//! Generate: cargo run -p peacemaker-isa --release --features lab --example gfx1100_divide_cases_probe -- --out-dir DIR
//! Build:    cargo build -p hipfire-runtime --release --features lab --example gfx1100_divide_cases_observe
//! Run on the locked xtx (gfx1100) through the fabric scheduler:
//!   gfx1100_divide_cases_observe --codeobject DIR/gfx1100_divide_cases_probe.co \
//!     --manifest DIR/gfx1100_divide_cases_probe.manifest.json --cases CASES.json --card xtx \
//!     --expect-uuid GPU-43390a851e296ee5 [--expect-bdf 0000:67:00.0] --out OUT.json [--repeats 3]
//!
//! `CASES.json` is an array (or `{"cases": [...]}`) of `{"numerator_bits": N, "denominator_bits": D}`; each
//! value is a JSON number or a string (`"0x3f800000"`, decimal). Any f32 bit pattern is allowed (NaN,
//! inf, zero, subnormal, any finite pair). Cases are run in order, duplicates included.
//!
//! Output (schema `consumer24.gfx1100.divide-cases.v1`):
//!  * `divide_extras`: `[{divide_num, divide_den, divide_result}]`, one per case in input order, the
//!    same key/result triple as the primary table's `divide_*` fields, for merging into the divide hook table.
//!  * `trace`: every divide-DAG intermediate per case (see the generator's word table).
//!  * `host_ieee_compare`: Rust `f32` division (correctly-rounded RNE, the `__fdiv_rn` result for non-NaN)
//!    against the device DAG, per case (`differs`), with totals and every differing case.
//!  * `selfcheck`: marker/echo/repeat-run/duplicate-case consistency; failure exits 1 after the file is written.
//! Refuses (exit 1, nothing launched) on `HSA_CU_MASK` / `ROC_GLOBAL_CU_MASK`, any manifest hash/ABI
//! disagreement, a non-gfx1100 device, a UUID/BDF mismatch, or `--card xtx` not resolving to the known xtx.

use hip_bridge::HipRuntime;
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::HashMap;

type Result<T, E = String> = std::result::Result<T, E>;
fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}
fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

const SCHEMA: &str = "consumer24.gfx1100.divide-cases.v1";
const ARCH: &str = "gfx1100";
const SYMBOL: &str = "c24_divide_cases_probe_gfx1100";
const KERNARG: usize = 24;
const WORDS: usize = 16;
const RECORD_BYTES: usize = WORDS * 4;
const MARKER: u32 = 0x4449_5650;
const POISON: u8 = 0xa5;
const XTX: (&str, &str, &str) = ("xtx", "GPU-43390a851e296ee5", "0000:67:00.0");
const LINK_CAVEAT: &str = "hipx xtx sits behind a PCIe 4.0 x4 root port (00:03.1); this probe measures arithmetic bits only, no speed or bandwidth";
const USAGE: &str = "usage: gfx1100_divide_cases_observe --codeobject PATH --manifest PATH --cases PATH --card LABEL --expect-uuid UUID [--expect-bdf BDF] --out PATH [--repeats N]";
const TRACE_FIELDS: [&str; 14] = [
    "numerator_bits",
    "denominator_bits",
    "div_scale_den",
    "div_scale_num",
    "rcp",
    "fma_one_minus_d_r",
    "fmac_r_refine",
    "mul_q0",
    "fma_residual0",
    "fmac_q1",
    "fma_residual1",
    "div_fmas",
    "divide_result",
    "numerator_scale_carry",
];

type Rec = [u32; WORDS];

fn bits(v: &Value, what: &str) -> Result<u32> {
    let n = match v {
        Value::Number(n) => n
            .as_u64()
            .ok_or(format!("{what}: not an unsigned integer"))?,
        Value::String(s) => {
            let s = s.trim();
            let parsed = match s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
                Some(h) => u64::from_str_radix(h, 16),
                None => s.parse::<u64>(),
            };
            parsed.map_err(|e| format!("{what}: {e}"))?
        }
        _ => return Err(format!("{what}: expected number or string")),
    };
    u32::try_from(n).map_err(|_| format!("{what}: {n:#x} exceeds 32 bits"))
}

fn run() -> Result<()> {
    let (
        mut codeobject,
        mut manifest_path,
        mut cases_path,
        mut card,
        mut expect_uuid,
        mut expect_bdf,
        mut out_path,
    ) = (None, None, None, None, None, None, None);
    let mut repeats = 3usize;
    let argv: Vec<String> = std::env::args().collect();
    let mut args = argv.iter().skip(1);
    while let Some(flag) = args.next() {
        let value = args.next().ok_or(USAGE)?.clone();
        match flag.as_str() {
            "--codeobject" => codeobject = Some(value),
            "--manifest" => manifest_path = Some(value),
            "--cases" => cases_path = Some(value),
            "--card" => card = Some(value),
            "--expect-uuid" => expect_uuid = Some(value),
            "--expect-bdf" => expect_bdf = Some(value.to_ascii_lowercase()),
            "--out" => out_path = Some(value),
            "--repeats" => repeats = value.parse().map_err(err)?,
            _ => return Err(USAGE.into()),
        }
    }
    let (codeobject, manifest_path, cases_path, card, expect_uuid, out_path) = (
        codeobject.ok_or(USAGE)?,
        manifest_path.ok_or(USAGE)?,
        cases_path.ok_or(USAGE)?,
        card.ok_or(USAGE)?,
        expect_uuid.ok_or(USAGE)?,
        out_path.ok_or(USAGE)?,
    );
    if repeats == 0 {
        return Err("--repeats must be at least 1".into());
    }
    for var in ["HSA_CU_MASK", "ROC_GLOBAL_CU_MASK"] {
        if std::env::var_os(var).is_some() {
            return Err(format!(
                "{var} is set: CU-mask variables are forbidden on RDNA devices"
            ));
        }
    }

    // Inputs and bindings.
    let image = std::fs::read(&codeobject).map_err(|e| format!("{codeobject}: {e}"))?;
    let manifest_bytes =
        std::fs::read(&manifest_path).map_err(|e| format!("{manifest_path}: {e}"))?;
    let manifest: Value = serde_json::from_slice(&manifest_bytes).map_err(err)?;
    let m_str = |k: &str| manifest[k].as_str().ok_or(format!("manifest: missing {k}"));
    let m_u64 = |k: &str| manifest[k].as_u64().ok_or(format!("manifest: missing {k}"));
    if m_str("schema")? != "consumer24.gfx1100.divide-cases-build.v1"
        || m_str("probe_schema")? != SCHEMA
    {
        return Err("manifest: schema mismatch".into());
    }
    if m_str("symbol")? != SYMBOL || m_str("arch")? != ARCH {
        return Err("manifest: symbol/arch mismatch".into());
    }
    if m_u64("kernarg_bytes")? as usize != KERNARG
        || m_u64("record_words")? as usize != WORDS
        || m_u64("marker")? as u32 != MARKER
    {
        return Err("manifest: kernarg/record ABI mismatch".into());
    }
    let co_sha = sha(&image);
    if m_str("co_sha256")? != co_sha {
        return Err(format!(
            "code object sha256 {co_sha} differs from the manifest's {}",
            m_str("co_sha256")?
        ));
    }
    let cases_bytes = std::fs::read(&cases_path).map_err(|e| format!("{cases_path}: {e}"))?;
    let cases_doc: Value = serde_json::from_slice(&cases_bytes).map_err(err)?;
    let list = cases_doc
        .as_array()
        .or_else(|| cases_doc["cases"].as_array())
        .ok_or("cases: expected an array or {\"cases\": [...]}")?;
    if list.is_empty() {
        return Err("cases: empty".into());
    }
    let mut pairs: Vec<(u32, u32)> = Vec::with_capacity(list.len());
    for (i, c) in list.iter().enumerate() {
        pairs.push((
            bits(&c["numerator_bits"], &format!("cases[{i}].numerator_bits"))?,
            bits(
                &c["denominator_bits"],
                &format!("cases[{i}].denominator_bits"),
            )?,
        ));
    }
    let real = pairs.len();

    // Device identity.
    let hip = HipRuntime::load().map_err(err)?;
    let id = hip.current_device().map_err(err)?;
    let uuid = hip.device_uuid(id).map_err(err)?;
    let bdf = hip.device_pci_bus_id(id).map_err(err)?.to_ascii_lowercase();
    let arch = hip.get_arch(id).map_err(err)?;
    let card_known = card != XTX.0 || (uuid == XTX.1 && bdf == XTX.2);
    let identity_match = arch == ARCH
        && uuid == expect_uuid
        && expect_bdf.as_deref().is_none_or(|b| b == bdf)
        && card_known;
    let device = json!({"card": card, "uuid": uuid, "bdf": bdf, "arch": arch, "expected_arch": ARCH, "expected_uuid": expect_uuid,
                        "expected_bdf": expect_bdf, "card_known_identity_checked": card == XTX.0, "identity_match": identity_match});
    if !identity_match {
        println!(
            "{}",
            json!({"schema": SCHEMA, "device": device, "error": "gfx1100 divide probe device identity mismatch"})
        );
        return Err("gfx1100 divide probe device identity mismatch".into());
    }

    // Launch: the cases padded to a multiple of 32 with (1.0, 1.0).
    let n = real.div_ceil(32) * 32;
    let mut padded = pairs.clone();
    padded.resize(n, (0x3f80_0000, 0x3f80_0000));
    let case_bytes: Vec<u8> = padded
        .iter()
        .flat_map(|&(a, b)| a.to_le_bytes().into_iter().chain(b.to_le_bytes()))
        .collect();
    let module = hip.module_load_data(&image).map_err(err)?;
    let f = hip.module_get_function(&module, SYMBOL).map_err(err)?;
    let (dcases, dout) = (
        hip.malloc(n * 8).map_err(err)?,
        hip.malloc(n * RECORD_BYTES).map_err(err)?,
    );
    hip.memcpy_htod(&dcases, &case_bytes).map_err(err)?;
    let mut runs: Vec<Vec<u8>> = Vec::new();
    for _ in 0..repeats {
        hip.memset(&dout, i32::from(POISON), n * RECORD_BYTES)
            .map_err(err)?;
        let mut kernarg = [0u8; KERNARG];
        kernarg[0..8].copy_from_slice(&(dcases.as_ptr() as u64).to_le_bytes());
        kernarg[8..16].copy_from_slice(&(dout.as_ptr() as u64).to_le_bytes());
        kernarg[16..20].copy_from_slice(&(n as u32).to_le_bytes());
        // SAFETY: exact 24-byte ABI (`cases, out, n`); both allocations outlive the synchronize.
        unsafe {
            hip.launch_kernel_blob(
                &f,
                [(n / 32) as u32, 1, 1],
                [32, 1, 1],
                0,
                None,
                &mut kernarg,
            )
        }
        .map_err(err)?;
        hip.device_synchronize().map_err(err)?;
        let mut bytes = vec![0u8; n * RECORD_BYTES];
        hip.memcpy_dtoh(&mut bytes, &dout).map_err(err)?;
        runs.push(bytes);
    }
    let recs: Vec<Rec> = runs[0]
        .chunks_exact(RECORD_BYTES)
        .map(|c| {
            let mut r = [0u32; WORDS];
            for (i, w) in c.chunks_exact(4).enumerate() {
                r[i] = u32::from_le_bytes(w.try_into().unwrap());
            }
            r
        })
        .collect();

    // Checks and comparison.
    let repeat_identical = runs.iter().all(|r| *r == runs[0]);
    let (mut marker_missing, mut echo_mismatch) = (0u64, 0u64);
    let mut first_bad: Vec<Value> = Vec::new();
    for (i, r) in recs.iter().enumerate() {
        if r[15] != MARKER {
            marker_missing += 1;
            if first_bad.len() < 16 {
                first_bad.push(json!({"index": i, "problem": "marker"}));
            }
        }
        if (r[0], r[1]) != padded[i] {
            echo_mismatch += 1;
            if first_bad.len() < 16 {
                first_bad.push(json!({"index": i, "problem": "echo"}));
            }
        }
    }
    let mut first_index: HashMap<(u32, u32), usize> = HashMap::new();
    let (mut duplicate_cases, mut duplicate_inconsistent) = (0u64, 0u64);
    for i in 0..real {
        match first_index.get(&pairs[i]) {
            Some(&j) => {
                duplicate_cases += 1;
                if recs[i] != recs[j] {
                    duplicate_inconsistent += 1;
                }
            }
            None => {
                first_index.insert(pairs[i], i);
            }
        }
    }
    let mut divide_extras = Vec::with_capacity(real);
    let mut trace = Vec::with_capacity(real);
    let mut compare = Vec::with_capacity(real);
    let mut differing = Vec::new();
    for (i, r) in recs[..real].iter().enumerate() {
        divide_extras.push(json!({"divide_num": r[0], "divide_den": r[1], "divide_result": r[12]}));
        let mut m = Map::new();
        for (k, name) in TRACE_FIELDS.iter().enumerate() {
            m.insert((*name).into(), json!(r[k]));
        }
        trace.push(Value::Object(m));
        let host = (f32::from_bits(r[0]) / f32::from_bits(r[1])).to_bits();
        let differs = host != r[12];
        let row = json!({"index": i, "numerator_bits": r[0], "denominator_bits": r[1], "device_result": r[12], "host_ieee_result": host, "differs": differs});
        if differs {
            differing.push(row.clone());
        }
        compare.push(row);
    }
    let records_sha = sha(&runs[0][..real * RECORD_BYTES]);
    let checks_ok = marker_missing == 0
        && echo_mismatch == 0
        && repeat_identical
        && duplicate_inconsistent == 0;
    let selfcheck = json!({
        "ok": checks_ok, "launches": repeats, "repeat_identical": repeat_identical, "marker_missing": marker_missing, "echo_mismatch": echo_mismatch,
        "cases_listed_twice": duplicate_cases, "cases_listed_twice_inconsistent": duplicate_inconsistent, "first_problems": first_bad,
    });
    let doc = json!({
        "schema": SCHEMA,
        "divide_extras": divide_extras,
        "trace_fields": TRACE_FIELDS, "trace": trace,
        "host_ieee_compare": {"cases": real, "differs": differing.len(), "differing_cases": differing, "per_case": compare,
                              "host_semantics": "Rust f32 division (correctly rounded, round-to-nearest-even); NaN payloads follow the host"},
        "device": device,
        "probe": {"symbol": SYMBOL, "codeobject": codeobject, "co_sha256": co_sha, "manifest": manifest_path, "manifest_sha256": sha(&manifest_bytes),
                  "cases": cases_path, "cases_file_sha256": sha(&cases_bytes), "kernarg_bytes": KERNARG, "record_words": WORDS,
                  "grid_x": n / 32, "block_x": 32, "padded_cases": n, "denorm": manifest["denorm"],
                  "dag_source": manifest["dag_source"], "dag_source_sha256": manifest["dag_source_sha256"], "listing_instructions": manifest["instructions"]},
        "records_sha256": records_sha,
        "selfcheck": selfcheck,
        "scope": "gfx1100 divide DAG bits for the supplied operand pairs only; no candidate K2 code participated",
        "link_caveat": LINK_CAVEAT,
        "command": argv.join(" "),
    });
    let out_bytes = serde_json::to_vec(&doc).map_err(err)?;
    std::fs::write(&out_path, &out_bytes).map_err(|e| format!("{out_path}: {e}"))?;
    println!(
        "{}",
        json!({
            "schema": SCHEMA, "out": out_path, "out_sha256": sha(&out_bytes), "co_sha256": co_sha, "manifest_sha256": sha(&manifest_bytes),
            "cases_file_sha256": sha(&cases_bytes), "records_sha256": records_sha, "device": device, "cases": real,
            "selfcheck_ok": checks_ok, "host_ieee_differs": differing.len(), "link_caveat": LINK_CAVEAT,
        })
    );
    if !checks_ok {
        return Err("divide probe self-checks failed (see selfcheck in the output file)".into());
    }
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("gfx1100_divide_cases_observe: {e}");
        std::process::exit(1);
    }
}
