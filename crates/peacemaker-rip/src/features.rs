//! Per-target feature report for one `.rip` source: resources, assembly hash,
//! static schedule cost and emulator semantic coverage.
//!
//! Authority for executability is `peacemaker-ir/isa/semantics.tbl` through
//! [`peacemaker_ir::emu::semantic_class`]; a table/handler disagreement is an
//! emulator bug and is never masked here. Classification uses
//! `semantic_class` only, never handler introspection.

use serde_json::{json, Value};

use peacemaker_isa::Arch;

#[cfg(feature = "certify")]
use super::symbol_of;
use super::{check, wave_size_of};
use crate::diag::Severity;
#[cfg(feature = "certify")]
use std::collections::BTreeMap;

// ---------------------------------------------------------------------------
// SHA-256 (bytes hashed in Rust; no shell-out, no new dependency).
// Streaming: full 64-byte blocks hash straight from the input slice; only the
// padded tail (at most two blocks) lives in a stack buffer.
// ---------------------------------------------------------------------------

fn compress(h: &mut [u32; 8], chunk: &[u8]) {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut w = [0u32; 64];
    for i in 0..16 {
        w[i] = u32::from_be_bytes([
            chunk[4 * i],
            chunk[4 * i + 1],
            chunk[4 * i + 2],
            chunk[4 * i + 3],
        ]);
    }
    for i in 16..64 {
        let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
        let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16]
            .wrapping_add(s0)
            .wrapping_add(w[i - 7])
            .wrapping_add(s1);
    }
    let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh) =
        (h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);
    for i in 0..64 {
        let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
        let ch = (e & f) ^ ((!e) & g);
        let t1 = hh
            .wrapping_add(s1)
            .wrapping_add(ch)
            .wrapping_add(K[i])
            .wrapping_add(w[i]);
        let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
        let maj = (a & b) ^ (a & c) ^ (b & c);
        let t2 = s0.wrapping_add(maj);
        hh = g;
        g = f;
        f = e;
        e = d.wrapping_add(t1);
        d = c;
        c = b;
        b = a;
        a = t1.wrapping_add(t2);
    }
    h[0] = h[0].wrapping_add(a);
    h[1] = h[1].wrapping_add(b);
    h[2] = h[2].wrapping_add(c);
    h[3] = h[3].wrapping_add(d);
    h[4] = h[4].wrapping_add(e);
    h[5] = h[5].wrapping_add(f);
    h[6] = h[6].wrapping_add(g);
    h[7] = h[7].wrapping_add(hh);
}

fn sha256(bytes: &[u8]) -> [u8; 32] {
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let (chunks, rem) = bytes.as_chunks::<64>();
    for chunk in chunks {
        compress(&mut h, chunk);
    }
    let bit_len = (bytes.len() as u64).wrapping_mul(8);
    // Remainder plus the 0x80 byte plus the 8 length bytes pads out to one or
    // two blocks; the buffer is zeroed, so unwritten bytes are the zero pad.
    let mut tail = [0u8; 128];
    tail[..rem.len()].copy_from_slice(rem);
    tail[rem.len()] = 0x80;
    let padded = (rem.len() + 1 + 8).div_ceil(64) * 64;
    tail[padded - 8..padded].copy_from_slice(&bit_len.to_be_bytes());
    for block in tail[..padded].as_chunks::<64>().0 {
        compress(&mut h, block);
    }
    let mut out = [0u8; 32];
    for (i, v) in h.iter().enumerate() {
        out[4 * i..4 * i + 4].copy_from_slice(&v.to_be_bytes());
    }
    out
}

fn hex(bytes: &[u8]) -> String {
    const H: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push(H[(b >> 4) as usize] as char);
        s.push(H[(b & 15) as usize] as char);
    }
    s
}

/// Hex SHA-256 of the `.rip` source bytes.
pub fn source_sha256(source: &str) -> String {
    hex(&sha256(source.as_bytes()))
}

// ---------------------------------------------------------------------------
// Routes (parsed per target, validated even when no loop or symbol matches).
// ---------------------------------------------------------------------------

#[cfg(feature = "certify")]
type Routes = BTreeMap<String, peacemaker_isa::cost_lint::Route>;

#[cfg(feature = "certify")]
fn parse_routes(text: &str) -> Result<Routes, String> {
    let map: Routes =
        serde_json::from_str(text).map_err(|e| format!("routes: invalid JSON: {e}"))?;
    for (symbol, route) in &map {
        route
            .padding_fraction()
            .map_err(|e| format!("routes[{symbol}]: {e}"))?;
        match route.k128_epochs {
            Some(0) => {
                return Err(format!("routes[{symbol}]: k128_epochs must be positive"));
            }
            Some(n) => {
                for starts in route.epoch_starts.values() {
                    if starts.len() != n {
                        return Err(format!(
                            "routes[{symbol}]: epoch starts must match k128_epochs"
                        ));
                    }
                }
            }
            None => {
                if !route.epoch_starts.is_empty() {
                    return Err(format!(
                        "routes[{symbol}]: epoch starts without k128_epochs"
                    ));
                }
            }
        }
    }
    Ok(map)
}

// ---------------------------------------------------------------------------
// Linked build: the same `toolchain::build` path `--certify` uses.
// ---------------------------------------------------------------------------

/// Assemble and link `s_text` for `arch` and return the linked ELF bytes
/// (`build.elf`). Temp files are removed best-effort; their content is never
/// part of the report.
#[cfg(feature = "certify")]
fn linked_object(arch: Arch, variant: &str, s_text: &str) -> Result<Vec<u8>, String> {
    use peacemaker_isa::toolchain::{build, Toolchain};
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let mut tag = String::with_capacity(variant.len());
    for ch in variant.chars() {
        tag.push(if ch.is_ascii_alphanumeric() { ch } else { '_' });
    }
    let stem = format!(
        "rip-features-{}-{}-{}-{tag}",
        std::process::id(),
        SEQ.fetch_add(1, Ordering::Relaxed),
        arch.name(),
    );
    let dir = std::env::temp_dir();
    let s_path = dir.join(format!("{stem}.s"));
    let hsaco_path = dir.join(format!("{stem}.hsaco"));
    let cleanup = |elf: Option<&std::path::Path>| {
        let _ = std::fs::remove_file(&s_path);
        let _ = std::fs::remove_file(&hsaco_path);
        if let Some(elf) = elf {
            let _ = std::fs::remove_file(elf);
        }
    };
    std::fs::write(&s_path, s_text).map_err(|e| format!("{}: {e}", s_path.display()))?;
    let elf = match build(&Toolchain::default(), &s_path, &hsaco_path, arch.name()) {
        Ok(out) => out.elf,
        Err(e) => {
            cleanup(None);
            return Err(format!("linked build: {e}"));
        }
    };
    let bytes = std::fs::read(&elf).map_err(|e| format!("{}: {e}", elf.display()));
    cleanup(Some(&elf));
    bytes
}

// ---------------------------------------------------------------------------
// Emulator semantic coverage.
// ---------------------------------------------------------------------------

/// `{modeled, instructions, unmodeled: [{mnemonic, count}]}` over every lifted
/// instruction slot of `symbol` (VOPD X and Y halves count separately),
/// classified by `semantic_class` only. `unmodeled` is sorted by mnemonic.
#[cfg(feature = "certify")]
fn emulator_of(program: &peacemaker_ir::Program, symbol: &str) -> Result<Value, String> {
    use peacemaker_ir::{emu::semantic_class, inst::FormFields, isa};
    let arch = program.target.arch;
    let kernel = program
        .kernels
        .iter()
        .find(|k| k.symbol.0 == symbol)
        .ok_or_else(|| format!("{symbol}: lifted object has no such kernel"))?;
    let mut modeled = 0usize;
    let mut total = 0usize;
    let mut unmodeled: BTreeMap<String, usize> = BTreeMap::new();
    let mut count = |name: &str| {
        total += 1;
        if semantic_class(name).is_some() {
            modeled += 1;
        } else {
            *unmodeled.entry(name.to_string()).or_default() += 1;
        }
    };
    for id in &kernel.body.layout {
        let inst = kernel
            .body
            .insts
            .get(*id)
            .ok_or_else(|| format!("{symbol}: dangling instruction"))?;
        let x = isa::lookup(arch, inst.op, inst.form)
            .map(|row| row.name)
            .or_else(|| inst.op.name(arch))
            .unwrap_or("unknown");
        count(x);
        if let FormFields::Vopd { y_op, .. } = &inst.fields {
            count(y_op.name(arch).unwrap_or("unknown"));
        }
    }
    let unmodeled: Vec<Value> = unmodeled
        .into_iter()
        .map(|(mnemonic, n)| json!({"mnemonic": mnemonic, "count": n}))
        .collect();
    Ok(json!({"modeled": modeled, "instructions": total, "unmodeled": unmodeled}))
}

// ---------------------------------------------------------------------------
// Report.
// ---------------------------------------------------------------------------

fn fail(stage: &str, message: String) -> Value {
    json!({"stage": stage, "message": message})
}

fn target_error(error: &Option<Value>) -> Value {
    error.clone().unwrap_or(Value::Null)
}

fn finish(
    arch: Arch,
    variant: &str,
    resources: Value,
    assembly_sha256: Value,
    cost: Value,
    emulator: Value,
    error: &Option<Value>,
) -> Value {
    json!({
        "arch": arch.name(),
        "variant": variant,
        "ok": error.is_none(),
        "resources": resources,
        "assembly_sha256": assembly_sha256,
        "cost": cost,
        "emulator": emulator,
        "error": target_error(error),
    })
}

fn collect_one_inner(
    source: &str,
    name: &str,
    arch: Arch,
    variant: &str,
    routes_text: Option<&str>,
) -> Value {
    let mut error: Option<Value> = None;
    let mut resources = Value::Null;
    let mut assembly_sha256 = Value::Null;
    let mut cost = Value::Null;
    let mut emulator = Value::Null;

    // 1. Compile and validate.
    let checked = check(source, name, arch, variant);
    let Some(emitted) = checked.emitted else {
        let message = checked
            .diagnostics
            .iter()
            .find(|d| d.severity == Severity::Error)
            .map(|d| d.message.clone())
            .unwrap_or_else(|| "check failed without diagnostics".to_string());
        error = Some(fail("check", message));
        return finish(
            arch,
            variant,
            resources,
            assembly_sha256,
            cost,
            emulator,
            &error,
        );
    };

    // 2. Resources; wave size comes from the emitted descriptor, never the arch.
    match (&checked.resources, wave_size_of(&emitted)) {
        (Some(r), Some(w)) => {
            resources = json!({
                "vgprs": r.vgprs,
                "sgprs": r.sgprs,
                "lds_bytes": r.lds_bytes,
                "wave_size": w,
                "instructions": r.instructions,
            });
        }
        (None, _) => {
            error = Some(fail(
                "resources",
                "emitted kernel descriptor lacks resource fields".to_string(),
            ));
        }
        (_, None) => {
            error = Some(fail(
                "resources",
                "emitted kernel lacks a valid `.amdhsa_wavefront_size32`".to_string(),
            ));
        }
    }

    // 3. The assembly hash covers the emitted assembly bytes, never the object.
    assembly_sha256 = json!(hex(&sha256(emitted.s_text.as_bytes())));

    #[cfg(feature = "certify")]
    {
        // 4. Routes, validated even when no loop or symbol matches.
        let valid_routes: Option<Routes> = match routes_text {
            None => Some(BTreeMap::new()),
            Some(text) => match parse_routes(text) {
                Ok(map) => Some(map),
                Err(e) => {
                    if error.is_none() {
                        error = Some(fail("routes", e));
                    }
                    None
                }
            },
        };

        // 5. Assemble and link on the same `toolchain::build` path `--certify`
        // uses, then lift the linked `build.elf`. A lift failure keeps the
        // resources and hash above.
        let program = match linked_object(arch, variant, &emitted.s_text) {
            Ok(bytes) => {
                let symbol = match symbol_of(&emitted) {
                    Ok(symbol) => symbol,
                    Err(e) => {
                        if error.is_none() {
                            error = Some(fail("lift", e));
                        }
                        return finish(
                            arch,
                            variant,
                            resources,
                            assembly_sha256,
                            cost,
                            emulator,
                            &error,
                        );
                    }
                };
                match peacemaker_lift::lift_object(
                    &bytes,
                    peacemaker_lift::Options {
                        frontend: peacemaker_ir::inst::Frontend::Builder,
                    },
                ) {
                    Ok(lifted) => Some((lifted.program, symbol)),
                    Err(e) => {
                        if error.is_none() {
                            error = Some(fail("lift", e.to_string()));
                        }
                        None
                    }
                }
            }
            Err(e) => {
                if error.is_none() {
                    error = Some(fail("assemble", e));
                }
                None
            }
        };

        // 6. Emulator coverage and static cost are independent given the
        // program; both run so a cost failure still preserves emulator data.
        if let Some((program, symbol)) = &program {
            match emulator_of(program, symbol) {
                Ok(v) => emulator = v,
                Err(e) => {
                    if error.is_none() {
                        error = Some(fail("emulator", e));
                    }
                }
            }
            if let Some(map) = &valid_routes {
                let report = peacemaker_isa::cost_lint::analyze(program, map);
                let value = report
                    .as_ref()
                    .map_err(|e| e.clone())
                    .and_then(|r| serde_json::to_value(r).map_err(|e| e.to_string()));
                match value {
                    Ok(mut v) => {
                        // `drain_pressure` is a paired-object comparison, never
                        // a single-target measurement, so it is always null.
                        v["drain_pressure"] = Value::Null;
                        cost = v;
                    }
                    Err(e) => {
                        if error.is_none() {
                            error = Some(fail("cost", e));
                        }
                    }
                }
            }
        }
    }

    #[cfg(not(feature = "certify"))]
    {
        let _ = routes_text;
        // Without the toolchain there is no linked object to lift, cost or
        // model; fail closed but keep the resources and hash above.
        if error.is_none() {
            error = Some(fail(
                "toolchain",
                "feature `certify` disabled: linked build, lift, emulator and cost require peacemaker-isa/toolchain".to_string(),
            ));
        }
    }

    finish(
        arch,
        variant,
        resources,
        assembly_sha256,
        cost,
        emulator,
        &error,
    )
}

/// Collect the stable (schema 1) feature report for every `arch × variant` of
/// `source`: typed resources, assembly hash, static cost and emulator
/// coverage. Failures are fail-closed per-target `{stage, message}` values
/// that preserve whatever resources and hashes are available. `routes` is the
/// raw routes JSON text, passed unchanged to `cost_lint`; it is validated even
/// when no loop or symbol matches. Human rendering is owned by the parent.
pub fn collect(
    source: &str,
    name: &str,
    arches: &[Arch],
    variants: &[String],
    routes: Option<&str>,
) -> Value {
    let mut targets = serde_json::Map::new();
    let mut ok = true;
    for arch in arches {
        for variant in variants {
            let target = collect_one_inner(source, name, *arch, variant, routes);
            ok &= target["ok"].as_bool().unwrap_or(false);
            targets.insert(format!("{}/{}", arch.name(), variant), target);
        }
    }
    json!({
        "schema": 1,
        "file": name,
        "source_sha256": source_sha256(source),
        "ok": ok,
        "routes_supplied": routes.is_some(),
        "targets": targets,
        "error": null,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// FIPS vectors: empty, `abc`, and 56 bytes requiring two padded blocks.
    #[test]
    fn sha256_vectors() {
        assert_eq!(
            hex(&sha256(b"")),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            source_sha256("abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            hex(&sha256(
                b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"
            )),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
    }
}
