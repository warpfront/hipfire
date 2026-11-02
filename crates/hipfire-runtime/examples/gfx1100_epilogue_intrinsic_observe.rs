// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Runs the independent gfx1100 epilogue-intrinsic probe code object and writes the observation
//! table for `mq_i4_reference::EpilogueIntrinsics::{exp2, ldexp, divide}` (B4-C). Never takes
//! part in runtime dispatch; the candidate K2 kernel never populates any table.
//!
//! Generate the probe on hiptrx (no GPU, native assembler/linker, `lab` feature only, no `toolchain` feature):
//!   cargo run -p peacemaker-isa --release --features lab --example gfx1100_epilogue_intrinsic_probe -- --out-dir DIR
//! which writes `DIR/gfx1100_epilogue_intrinsic_probe.{s,proof.json,co,hxaco,manifest.json}`.
//!
//! Build the observer (no GPU needed to build):
//!   cargo build -p hipfire-runtime --release --features lab --example gfx1100_epilogue_intrinsic_observe
//! Run on the locked xtx (gfx1100) through the fabric scheduler, e.g.
//!   gfx1100_epilogue_intrinsic_observe --codeobject DIR/gfx1100_epilogue_intrinsic_probe.co \
//!     --manifest DIR/gfx1100_epilogue_intrinsic_probe.manifest.json --card xtx \
//!     --expect-uuid GPU-43390a851e296ee5 [--expect-bdf 0000:67:00.0] --out OBS.json [--repeats 3]
//!
//! The run refuses (exit 1, nothing launched) when: `HSA_CU_MASK` / `ROC_GLOBAL_CU_MASK` is set; the
//! manifest's code-object sha256, symbol, ABI or gate-set hashes disagree with the inputs; the device
//! arch is not gfx1100; the device UUID differs from `--expect-uuid` (or `--expect-bdf`); or `--card xtx`
//! does not resolve to the known xtx UUID/BDF (GPU-43390a851e296ee5, 0000:67:00.0).
//!
//! Output `OBS.json` (schema `consumer24.gfx1100.epilogue-probe.v1`):
//!  * `observations` (65,280 finite-BF16 gates, ascending) and `exceptions` (full-F32 NaN/inf/zero/
//!    subnormal/max cases, see `gates.rs`), each entry exactly
//!    `{gate_bits, exp2_arg, exp2_result, ldexp_arg, ldexp_exponent(i32), ldexp_result, divide_num,
//!    divide_den, divide_result}` — all u32 bit patterns. `ldexp_arg == exp2_result`, `divide_num == gate_bits`.
//!  * `observation_trace` / `exception_trace`: parallel arrays with every intermediate of the DAG
//!    (`ph, pl, e_rndne, ph_minus_e, exp2_arg, exp2_result, ldexp_exponent, ldexp_result,
//!    r_after_underflow_select, r_after_overflow_select, divide_den, divide_result, final_mul_up1,
//!    silu_dense_reference`), for assessing any divergence of the CPU twin's operands.
//!  * `cpu_comparison`: for each set, the CPU twin's operands (`silu_mul` with `f32` host arithmetic and
//!    `e as i32`) recomputed per gate and compared bit-for-bit to the device, per stage, with the first
//!    mismatches. `key_missing_*` counts CPU operand keys absent from the device table; the diagnostic
//!    chain then continues from the device value so later stages are still isolated (this is analysis
//!    only: the table-backed hook must error on a missing key, never fall back).
//!  * `selfcheck`: every launch's record is complete (marker), the tapped DAG equals the real
//!    `Epilogue::silu_dense` bit-for-bit, repeated launches and the gates listed in both sets agree, and
//!    no operand key maps to two results. Any failure exits 1 after the file is written.
//! The stdout line is the machine-readable receipt (hashes, counts, identity, exit-affecting checks).

use hip_bridge::HipRuntime;
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap};

#[path = "../../peacemaker-isa/examples/gfx1100_epilogue_probe/gates.rs"]
mod gates;

type Result<T, E = String> = std::result::Result<T, E>;
fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}
fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

const ARCH: &str = "gfx1100";
const SYMBOL: &str = "c24_epilogue_intrinsic_probe_gfx1100";
const KERNARG: usize = 24;
const WORDS: usize = 16;
const RECORD_BYTES: usize = WORDS * 4;
const MARKER: u32 = 0x5052_4f42;
const POISON: u8 = 0xa5;
const XTX: (&str, &str, &str) = ("xtx", "GPU-43390a851e296ee5", "0000:67:00.0");
const LINK_CAVEAT: &str = "hipx xtx sits behind a PCIe 4.0 x4 root port (00:03.1); this probe measures arithmetic bits only, no speed or bandwidth";
const USAGE: &str = "usage: gfx1100_epilogue_intrinsic_observe --codeobject PATH --manifest PATH --card LABEL --expect-uuid UUID [--expect-bdf BDF] --out PATH [--repeats N]";
const TRACE_FIELDS: [&str; 14] = [
    "ph",
    "pl",
    "e_rndne",
    "ph_minus_e",
    "exp2_arg",
    "exp2_result",
    "ldexp_exponent",
    "ldexp_result",
    "r_after_underflow_select",
    "r_after_overflow_select",
    "divide_den",
    "divide_result",
    "final_mul_up1",
    "silu_dense_reference",
];
// Record word of each trace field (word 0 = gate, 15 = marker).
const TRACE_WORDS: [usize; 14] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14];

type Rec = [u32; WORDS];

fn observation(r: &Rec) -> Value {
    json!({"gate_bits": r[0], "exp2_arg": r[5], "exp2_result": r[6], "ldexp_arg": r[6], "ldexp_exponent": r[7] as i32,
           "ldexp_result": r[8], "divide_num": r[0], "divide_den": r[11], "divide_result": r[12]})
}
fn trace(r: &Rec) -> Value {
    let mut m = Map::new();
    m.insert("gate_bits".into(), json!(r[0]));
    for (name, w) in TRACE_FIELDS.iter().zip(TRACE_WORDS) {
        m.insert((*name).into(), json!(r[w]));
    }
    Value::Object(m)
}

/// Device operand tables: key bits -> result bits, with conflicts counted.
#[derive(Default)]
struct Tables {
    exp2: BTreeMap<u32, u32>,
    ldexp: BTreeMap<(u32, u32), u32>,
    divide: BTreeMap<(u32, u32), u32>,
    conflicts: Vec<Value>,
}
impl Tables {
    fn add(&mut self, r: &Rec) {
        fn put<K: Ord + Copy + std::fmt::Debug>(
            map: &mut BTreeMap<K, u32>,
            k: K,
            v: u32,
            name: &str,
            gate: u32,
            out: &mut Vec<Value>,
        ) {
            match map.get(&k) {
                Some(&old) if old != v => out.push(json!({"table": name, "key": format!("{k:?}"), "first": old, "second": v, "gate_bits": gate})),
                Some(_) => {}
                None => { map.insert(k, v); }
            }
        }
        put(
            &mut self.exp2,
            r[5],
            r[6],
            "exp2",
            r[0],
            &mut self.conflicts,
        );
        put(
            &mut self.ldexp,
            (r[6], r[7]),
            r[8],
            "ldexp",
            r[0],
            &mut self.conflicts,
        );
        put(
            &mut self.divide,
            (r[0], r[11]),
            r[12],
            "divide",
            r[0],
            &mut self.conflicts,
        );
    }
}

struct Stage {
    name: &'static str,
    mismatches: u64,
    first: Vec<Value>,
}
impl Stage {
    fn check(&mut self, gate: u32, cpu: u32, dev: u32) -> bool {
        if cpu == dev {
            return false;
        }
        self.mismatches += 1;
        if self.first.len() < 8 {
            self.first
                .push(json!({"gate_bits": gate, "cpu": cpu, "device": dev}));
        }
        true
    }
}
const STAGES: [&str; 13] = [
    "ph",
    "pl",
    "e_rndne",
    "ph_minus_e",
    "exp2_arg",
    "ldexp_exponent",
    "r_after_underflow_select",
    "r_after_overflow_select",
    "divide_den",
    "final_mul_up1",
    "exp2_result_by_cpu_key",
    "ldexp_result_by_cpu_key",
    "divide_result_by_cpu_key",
];
struct SetCmp {
    stages: Vec<Stage>,
    missing: [u64; 3],
    first_missing: Vec<Value>,
    gates: u64,
    gates_with_discrepancy: u64,
}
impl SetCmp {
    fn new() -> Self {
        Self {
            stages: STAGES
                .iter()
                .map(|&name| Stage {
                    name,
                    mismatches: 0,
                    first: vec![],
                })
                .collect(),
            missing: [0; 3],
            first_missing: vec![],
            gates: 0,
            gates_with_discrepancy: 0,
        }
    }
    /// Recompute the CPU twin's `silu_mul` operands for this gate (host f32 arithmetic) and compare each
    /// stage to the device record. A missing hook key is counted and the chain continues from the device value.
    fn compare(&mut self, r: &Rec, t: &Tables) {
        self.gates += 1;
        let gate = r[0];
        let g = f32::from_bits(gate);
        let neg_log2e = f32::from_bits(0xbfb8_aa3b);
        let neg_log2e_lo = f32::from_bits(0xb2a5_705f);
        let ph = neg_log2e * g;
        let mut pl = neg_log2e.mul_add(g, -ph);
        pl = neg_log2e_lo.mul_add(g, pl);
        let e = ph.round_ties_even();
        let sub = ph - e;
        let a = sub + pl;
        let e_int = e as i32;
        let mut bad = [
            self.stages[0].check(gate, ph.to_bits(), r[1]),
            self.stages[1].check(gate, pl.to_bits(), r[2]),
            self.stages[2].check(gate, e.to_bits(), r[3]),
            self.stages[3].check(gate, sub.to_bits(), r[4]),
            self.stages[4].check(gate, a.to_bits(), r[5]),
            self.stages[5].check(gate, e_int as u32, r[7]),
        ]
        .iter()
        .any(|&b| b);
        let miss = |idx: usize, key: String, me: &mut Self| {
            me.missing[idx] += 1;
            let table: &str = ["exp2", "ldexp", "divide"][idx];
            if me.first_missing.len() < 16 {
                me.first_missing
                    .push(json!({"gate_bits": gate, "table": table, "cpu_key": key}));
            }
        };
        let exp2_res = match t.exp2.get(&a.to_bits()) {
            Some(&x) => {
                bad |= self.stages[10].check(gate, x, r[6]);
                x
            }
            None => {
                miss(0, format!("{:#010x}", a.to_bits()), self);
                bad = true;
                r[6]
            }
        };
        let ldexp_res = match t.ldexp.get(&(exp2_res, e_int as u32)) {
            Some(&x) => {
                bad |= self.stages[11].check(gate, x, r[8]);
                x
            }
            None => {
                miss(1, format!("({exp2_res:#010x}, {e_int})"), self);
                bad = true;
                r[8]
            }
        };
        let r1 = if f32::from_bits(0x42ce_8ed0) < g {
            0.0f32.to_bits()
        } else {
            ldexp_res
        };
        let r2 = if g < f32::from_bits(0xc2b1_7218) {
            f32::INFINITY.to_bits()
        } else {
            r1
        };
        bad |= self.stages[6].check(gate, r1, r[9]);
        bad |= self.stages[7].check(gate, r2, r[10]);
        let den = 1.0f32 + f32::from_bits(r2);
        bad |= self.stages[8].check(gate, den.to_bits(), r[11]);
        let q = match t.divide.get(&(gate, den.to_bits())) {
            Some(&x) => {
                bad |= self.stages[12].check(gate, x, r[12]);
                x
            }
            None => {
                miss(2, format!("({gate:#010x}, {:#010x})", den.to_bits()), self);
                bad = true;
                r[12]
            }
        };
        let h = f32::from_bits(q) * std::hint::black_box(1.0f32);
        bad |= self.stages[9].check(gate, h.to_bits(), r[13]);
        if bad {
            self.gates_with_discrepancy += 1;
        }
    }
    fn json(&self) -> Value {
        let stages: Map<String, Value> = self
            .stages
            .iter()
            .map(|s| {
                (
                    s.name.to_string(),
                    json!({"mismatches": s.mismatches, "first": s.first}),
                )
            })
            .collect();
        json!({"gates": self.gates, "gates_with_any_discrepancy": self.gates_with_discrepancy, "stages": stages,
               "key_missing_exp2": self.missing[0], "key_missing_ldexp": self.missing[1], "key_missing_divide": self.missing[2],
               "first_missing_keys": self.first_missing})
    }
}

fn parse_records(bytes: &[u8]) -> Vec<Rec> {
    bytes
        .chunks_exact(RECORD_BYTES)
        .map(|c| {
            let mut r = [0u32; WORDS];
            for (i, w) in c.chunks_exact(4).enumerate() {
                r[i] = u32::from_le_bytes(w.try_into().unwrap());
            }
            r
        })
        .collect()
}

fn run() -> Result<()> {
    let (
        mut codeobject,
        mut manifest_path,
        mut card,
        mut expect_uuid,
        mut expect_bdf,
        mut out_path,
    ) = (None, None, None, None, None, None);
    let mut repeats = 3usize;
    let argv: Vec<String> = std::env::args().collect();
    let mut args = argv.iter().skip(1);
    while let Some(flag) = args.next() {
        let value = args.next().ok_or(USAGE)?.clone();
        match flag.as_str() {
            "--codeobject" => codeobject = Some(value),
            "--manifest" => manifest_path = Some(value),
            "--card" => card = Some(value),
            "--expect-uuid" => expect_uuid = Some(value),
            "--expect-bdf" => expect_bdf = Some(value.to_ascii_lowercase()),
            "--out" => out_path = Some(value),
            "--repeats" => repeats = value.parse().map_err(err)?,
            _ => return Err(USAGE.into()),
        }
    }
    let (codeobject, manifest_path, card, expect_uuid, out_path) = (
        codeobject.ok_or(USAGE)?,
        manifest_path.ok_or(USAGE)?,
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

    // Inputs and their bindings.
    let image = std::fs::read(&codeobject).map_err(|e| format!("{codeobject}: {e}"))?;
    let manifest_bytes =
        std::fs::read(&manifest_path).map_err(|e| format!("{manifest_path}: {e}"))?;
    let manifest: Value = serde_json::from_slice(&manifest_bytes).map_err(err)?;
    let m_str = |k: &str| manifest[k].as_str().ok_or(format!("manifest: missing {k}"));
    let m_u64 = |k: &str| manifest[k].as_u64().ok_or(format!("manifest: missing {k}"));
    if m_str("schema")? != "consumer24.gfx1100.epilogue-probe-build.v1"
        || m_str("probe_schema")? != gates::SCHEMA
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
    let (finite, exceptions) = (gates::finite_bf16(), gates::exceptional());
    let (finite_sha, exc_sha) = (gates::list_sha256(&finite), gates::list_sha256(&exceptions));
    let sets = &manifest["gate_sets"];
    if sets["finite_bf16"]["sha256_le_u32"].as_str() != Some(finite_sha.as_str())
        || sets["finite_bf16"]["count"].as_u64() != Some(finite.len() as u64)
        || sets["exceptions"]["sha256_le_u32"].as_str() != Some(exc_sha.as_str())
        || sets["exceptions"]["count"].as_u64() != Some(exceptions.len() as u64)
    {
        return Err(
            "manifest: gate-set hashes/counts differ from this observer's input sets".into(),
        );
    }

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
            json!({"schema": gates::SCHEMA, "device": device, "error": "gfx1100 probe device identity mismatch"})
        );
        return Err("gfx1100 probe device identity mismatch".into());
    }

    // Launch: real gates, then zero padding to a multiple of 32.
    let real = finite.len() + exceptions.len();
    let n = real.div_ceil(32) * 32;
    let mut all: Vec<u32> = finite.iter().chain(&exceptions).copied().collect();
    all.resize(n, 0);
    let gate_bytes: Vec<u8> = all.iter().flat_map(|g| g.to_le_bytes()).collect();
    let module = hip.module_load_data(&image).map_err(err)?;
    let f = hip.module_get_function(&module, SYMBOL).map_err(err)?;
    let (dgates, dout) = (
        hip.malloc(n * 4).map_err(err)?,
        hip.malloc(n * RECORD_BYTES).map_err(err)?,
    );
    hip.memcpy_htod(&dgates, &gate_bytes).map_err(err)?;
    let mut runs: Vec<Vec<u8>> = Vec::new();
    for _ in 0..repeats {
        hip.memset(&dout, i32::from(POISON), n * RECORD_BYTES)
            .map_err(err)?;
        let mut kernarg = [0u8; KERNARG];
        kernarg[0..8].copy_from_slice(&(dgates.as_ptr() as u64).to_le_bytes());
        kernarg[8..16].copy_from_slice(&(dout.as_ptr() as u64).to_le_bytes());
        kernarg[16..20].copy_from_slice(&(n as u32).to_le_bytes());
        // SAFETY: exact 24-byte ABI (`gates, out, n`); both allocations outlive the synchronize.
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
    let recs = parse_records(&runs[0]);

    // Self checks.
    let repeat_identical = runs.iter().all(|r| *r == runs[0]);
    let (mut marker_missing, mut echo_mismatch, mut tapped_vs_reference, mut poisoned_records) =
        (0u64, 0u64, 0u64, 0u64);
    let mut first_bad: Vec<Value> = Vec::new();
    for (i, r) in recs.iter().enumerate() {
        let mut note = |what: &str| {
            if first_bad.len() < 16 {
                first_bad.push(json!({"index": i, "gate_bits": all[i], "problem": what}));
            }
        };
        if r[15] != MARKER {
            marker_missing += 1;
            note("marker");
        }
        if r[0] != all[i] {
            echo_mismatch += 1;
            note("echo");
        }
        if r.iter()
            .take(15)
            .all(|&w| w == u32::from_ne_bytes([POISON; 4]))
        {
            poisoned_records += 1;
        }
        if i < real && r[13] != r[14] {
            tapped_vs_reference += 1;
            note("tapped DAG != Epilogue::silu_dense");
        }
    }
    let mut first_index: HashMap<u32, usize> = HashMap::new();
    let mut duplicate_gates = 0u64;
    let mut duplicate_inconsistent = 0u64;
    for i in 0..real {
        match first_index.get(&all[i]) {
            Some(&j) => {
                duplicate_gates += 1;
                if recs[i] != recs[j] {
                    duplicate_inconsistent += 1;
                }
            }
            None => {
                first_index.insert(all[i], i);
            }
        }
    }
    let mut tables = Tables::default();
    for r in &recs[..real] {
        tables.add(r);
    }

    // CPU-twin operand comparison, per set.
    let (mut cmp_finite, mut cmp_exc) = (SetCmp::new(), SetCmp::new());
    for r in &recs[..finite.len()] {
        cmp_finite.compare(r, &tables);
    }
    for r in &recs[finite.len()..real] {
        cmp_exc.compare(r, &tables);
    }

    let observations: Vec<Value> = recs[..finite.len()].iter().map(observation).collect();
    let exception_obs: Vec<Value> = recs[finite.len()..real].iter().map(observation).collect();
    let observation_trace: Vec<Value> = recs[..finite.len()].iter().map(trace).collect();
    let exception_trace: Vec<Value> = recs[finite.len()..real].iter().map(trace).collect();
    let records_sha = sha(&runs[0][..real * RECORD_BYTES]);
    let checks_ok = marker_missing == 0
        && echo_mismatch == 0
        && tapped_vs_reference == 0
        && poisoned_records == 0
        && repeat_identical
        && duplicate_inconsistent == 0
        && tables.conflicts.is_empty();
    let selfcheck = json!({
        "ok": checks_ok, "launches": repeats, "repeat_identical": repeat_identical, "marker_missing": marker_missing,
        "echo_mismatch": echo_mismatch, "poisoned_records": poisoned_records, "tapped_dag_vs_silu_dense_mismatch": tapped_vs_reference,
        "gates_listed_twice": duplicate_gates, "gates_listed_twice_inconsistent": duplicate_inconsistent,
        "operand_key_conflicts": tables.conflicts.len(), "first_conflicts": tables.conflicts.iter().take(8).collect::<Vec<_>>(), "first_problems": first_bad,
    });
    let doc = json!({
        "schema": gates::SCHEMA,
        "observations": observations, "exceptions": exception_obs,
        "trace_fields": TRACE_FIELDS, "observation_trace": observation_trace, "exception_trace": exception_trace,
        "device": device,
        "probe": {"symbol": SYMBOL, "codeobject": codeobject, "co_sha256": co_sha, "manifest": manifest_path, "manifest_sha256": sha(&manifest_bytes),
                  "kernarg_bytes": KERNARG, "record_words": WORDS, "grid_x": n / 32, "block_x": 32, "padded_gates": n,
                  "denorm": manifest["denorm"], "dag_source": manifest["dag_source"], "dag_source_sha256": manifest["dag_source_sha256"],
                  "listing_instructions": manifest["instructions"]},
        "gate_sets": {"finite_bf16": {"count": finite.len(), "sha256_le_u32": finite_sha}, "exceptions": {"count": exceptions.len(), "sha256_le_u32": exc_sha}},
        "counts": {"observations": finite.len(), "exceptions": exceptions.len(), "exp2_keys": tables.exp2.len(), "ldexp_keys": tables.ldexp.len(), "divide_keys": tables.divide.len()},
        "records_sha256": records_sha,
        "selfcheck": selfcheck,
        "cpu_comparison": {"finite_bf16": cmp_finite.json(), "exceptions": cmp_exc.json()},
        "scope": "gfx1100 intrinsic operand/result bits only; no candidate K2 code participated",
        "link_caveat": LINK_CAVEAT,
        "command": argv.join(" "),
    });
    let out_bytes = serde_json::to_vec(&doc).map_err(err)?;
    std::fs::write(&out_path, &out_bytes).map_err(|e| format!("{out_path}: {e}"))?;
    println!(
        "{}",
        json!({
            "schema": gates::SCHEMA, "out": out_path, "out_sha256": sha(&out_bytes), "co_sha256": co_sha, "manifest_sha256": sha(&manifest_bytes),
            "records_sha256": records_sha, "device": device, "observations": finite.len(), "exceptions": exceptions.len(),
            "selfcheck_ok": checks_ok,
            "cpu_discrepant_gates": {"finite_bf16": cmp_finite.gates_with_discrepancy, "exceptions": cmp_exc.gates_with_discrepancy},
            "cpu_key_missing": {"finite_bf16": [cmp_finite.missing[0], cmp_finite.missing[1], cmp_finite.missing[2]], "exceptions": [cmp_exc.missing[0], cmp_exc.missing[1], cmp_exc.missing[2]]},
            "link_caveat": LINK_CAVEAT,
        })
    );
    if !checks_ok {
        return Err("probe self-checks failed (see selfcheck in the output file)".into());
    }
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("gfx1100_epilogue_intrinsic_observe: {e}");
        std::process::exit(1);
    }
}
