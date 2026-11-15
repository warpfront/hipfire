// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Narrow gfx942 S_NOR_B64 hardware/interpreter oracle. Three destination wirings,
//! 4096 edge/random input pairs, alternating incoming SCC, two output poisons and
//! allocation guards. No dry-run or numerical tolerance. Run only in a fabric job:
//! `cargo run -p peacemaker-isa --features lab --example cdna_nor_hw_oracle -- OUT_DIR`.
use hip_bridge::HipRuntime;
use peacemaker_cdna::enc::sop::{Sop2, Sop2Op};
use peacemaker_cdna::sim::{exec_one, Config, Memory, Wave};
use peacemaker_cdna::{Arch, Inst, Src};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

const N: usize = 4096;
const GUARD: usize = 64;
const NAMES: [&str; 3] = ["nor_separate", "nor_alias0", "nor_alias1"];
const SEED: u64 = 0x17d9_3b64_ac80_f125;

fn sha(b: &[u8]) -> String {
    format!("{:x}", Sha256::digest(b))
}
fn source() -> String {
    let mut s = String::from("#include <hip/hip_runtime.h>\n");
    for (k, asm) in [
        "asm volatile(\"s_cmp_lg_u32 %4, 0\\n\\ts_nor_b64 %0, %2, %3\\n\\ts_cselect_b32 %1, 1, 0\" : \"=&s\"(r), \"=&s\"(cc) : \"s\"(a), \"s\"(b), \"s\"(prime) : \"scc\");",
        "asm volatile(\"s_cmp_lg_u32 %3, 0\\n\\ts_nor_b64 %0, %0, %2\\n\\ts_cselect_b32 %1, 1, 0\" : \"+&s\"(a), \"=&s\"(cc) : \"s\"(b), \"s\"(prime) : \"scc\"); r = a;",
        "asm volatile(\"s_cmp_lg_u32 %3, 0\\n\\ts_nor_b64 %0, %2, %0\\n\\ts_cselect_b32 %1, 1, 0\" : \"+&s\"(b), \"=&s\"(cc) : \"s\"(a), \"s\"(prime) : \"scc\"); r = b;",
    ].iter().enumerate() {
        s.push_str(&format!(
            "extern \"C\" __global__ void {}(const unsigned long long* A, const unsigned long long* B, unsigned* O) {{\nunsigned i = blockIdx.x; unsigned long long a = A[i], b = B[i], r; unsigned cc, prime = i & 1;\n{}\nif (threadIdx.x == 0) {{ O[3*i] = (unsigned)r; O[3*i+1] = (unsigned)(r >> 32); O[3*i+2] = cc; }}\n}}\n",
            NAMES[k], asm
        ));
    }
    s
}
fn inputs() -> (Vec<u64>, Vec<u64>) {
    let edges = [
        0,
        1,
        2,
        u64::MAX,
        u64::MAX - 1,
        1 << 63,
        (1 << 63) - 1,
        0xffff_ffff,
        0xffff_ffff_0000_0000,
        0xaaaa_aaaa_5555_5555,
        0x5555_5555_aaaa_aaaa,
        1 << 32,
        (1 << 32) + 1,
        1 << 31,
        1 << 62,
        0x0123_4567_89ab_cdef,
    ];
    let mut state = SEED;
    let mut next = || {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        state.wrapping_mul(0x2545_f491_4f6c_dd1d)
    };
    let (mut a, mut b) = (Vec::with_capacity(N), Vec::with_capacity(N));
    for i in 0..N {
        if i < edges.len() * edges.len() {
            a.push(edges[i / edges.len()]);
            b.push(edges[i % edges.len()]);
        } else {
            a.push(next());
            b.push(next());
        }
    }
    (a, b)
}
fn bytes(x: &[u64]) -> Vec<u8> {
    x.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn expected(a: &[u64], b: &[u64], dst: u8) -> Result<Vec<u8>, String> {
    let inst = Inst::Sop2(Sop2 {
        op: Sop2Op::NorB64,
        sdst: Src::Sgpr(dst),
        ssrc0: Src::Sgpr(0),
        ssrc1: Src::Sgpr(2),
    });
    let mut out = Vec::with_capacity(N * 12);
    for (i, (&a, &b)) in a.iter().zip(b).enumerate() {
        let mut w = Wave::new(Arch::Gfx942, 0, 0);
        w.s[0] = a as u32;
        w.s[1] = (a >> 32) as u32;
        w.s[2] = b as u32;
        w.s[3] = (b >> 32) as u32;
        w.scc = i & 1 != 0;
        exec_one(
            &mut w,
            &inst,
            &mut [],
            &mut Memory::default(),
            &Config::default(),
        )
        .map_err(|e| format!("interpreter {i}: {e}"))?;
        for v in [
            w.s[usize::from(dst)],
            w.s[usize::from(dst) + 1],
            u32::from(w.scc),
        ] {
            out.extend(v.to_le_bytes());
        }
    }
    Ok(out)
}
fn run(dir: &Path) -> Result<Value, String> {
    std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    let src = source();
    let src_path = dir.join("nor.hip");
    let co_path = dir.join("nor.hsaco");
    std::fs::write(&src_path, &src).map_err(|e| e.to_string())?;
    let cc = Command::new("/opt/rocm/bin/hipcc")
        .args([
            "--genco",
            "--offload-arch=gfx942",
            "-O3",
            "--no-offload-compress",
            "-fuse-cuid=none",
            "-o",
            co_path.to_str().ok_or("non-UTF8 output path")?,
            src_path.to_str().ok_or("non-UTF8 source path")?,
        ])
        .output()
        .map_err(|e| e.to_string())?;
    if !cc.status.success() {
        return Err(format!("hipcc: {}", String::from_utf8_lossy(&cc.stderr)));
    }
    let code = std::fs::read(&co_path).map_err(|e| e.to_string())?;
    let rt = HipRuntime::load().map_err(|e| e.to_string())?;
    rt.set_device(0).map_err(|e| e.to_string())?;
    let arch = rt.get_arch(0).map_err(|e| e.to_string())?;
    if arch != "gfx942" {
        return Err(format!("requires gfx942, got {arch}"));
    }
    let module = rt.module_load_data(&code).map_err(|e| e.to_string())?;
    let (a, b) = inputs();
    let (ab, bb) = (bytes(&a), bytes(&b));
    let da = rt.malloc(ab.len()).map_err(|e| e.to_string())?;
    let db = rt.malloc(bb.len()).map_err(|e| e.to_string())?;
    let total = GUARD * 2 + N * 12;
    let dout = rt.malloc(total).map_err(|e| e.to_string())?;
    rt.memcpy_htod(&da, &ab).map_err(|e| e.to_string())?;
    rt.memcpy_htod(&db, &bb).map_err(|e| e.to_string())?;
    let mut rows = Vec::new();
    let mut controls = 0;
    for (k, dst) in [4, 0, 2].into_iter().enumerate() {
        let mut want = expected(&a, &b, dst)?;
        let f = rt
            .module_get_function(&module, NAMES[k])
            .map_err(|e| e.to_string())?;
        for poison in [0x5a, 0xa5] {
            rt.memset(&dout, i32::from(poison), total)
                .map_err(|e| e.to_string())?;
            let mut blob = Vec::new();
            for p in [
                da.as_ptr() as usize as u64,
                db.as_ptr() as usize as u64,
                dout.as_ptr() as usize as u64 + GUARD as u64,
            ] {
                blob.extend(p.to_le_bytes());
            }
            // Same explicit/hidden dispatch layout as cdna_mix_hw_oracle; three aligned pointers.
            for x in [N as u32, 1, 1] {
                blob.extend(x.to_le_bytes());
            }
            for x in [64u16, 1, 1] {
                blob.extend(x.to_le_bytes());
            }
            blob.resize(256, 0);
            // SAFETY: three live allocations, one wave per valid input pair, full hidden argument storage.
            unsafe { rt.launch_kernel_blob(&f, [N as u32, 1, 1], [64, 1, 1], 0, None, &mut blob) }
                .map_err(|e| e.to_string())?;
            rt.device_synchronize().map_err(|e| e.to_string())?;
            let mut got = vec![0; total];
            rt.memcpy_dtoh(&mut got, &dout).map_err(|e| e.to_string())?;
            let guards = got[..GUARD]
                .iter()
                .chain(&got[total - GUARD..])
                .all(|&v| v == poison);
            let payload = &got[GUARD..total - GUARD];
            let bad = payload.iter().zip(&want).filter(|(a, b)| a != b).count();
            rows.push(json!({"kernel":NAMES[k], "poison":poison, "cases":N,
                "guards_ok":guards, "bad_bytes":bad, "hardware_sha256":sha(payload), "interpreter_sha256":sha(&want)}));
            if !guards || bad != 0 {
                return Err(format!(
                    "{} poison {poison}: guards={guards}, bad_bytes={bad}",
                    NAMES[k]
                ));
            }
            // Detect result-bit and SCC corruption independently, not a no-op comparison.
            for j in [0, 8] {
                want[j] ^= 1;
                let detected = payload != want;
                want[j] ^= 1;
                if detected {
                    controls += 1;
                } else {
                    return Err("missed corruption control".into());
                }
            }
        }
    }
    for d in [da, db, dout] {
        rt.free(d).map_err(|e| e.to_string())?;
    }
    Ok(
        json!({"status":"PASS", "oracle":"cdna_nor_hw_oracle", "arch":arch,
        "cases":N*6, "controls_detected":controls, "seed":SEED, "inputs_a_sha256":sha(&ab),
        "inputs_b_sha256":sha(&bb), "hip_source_sha256":sha(src.as_bytes()), "code_object_sha256":sha(&code),
        "oracle_source_sha256":sha(include_bytes!("cdna_nor_hw_oracle.rs")),
        "interpreter_source_sha256":sha(include_bytes!("../../peacemaker-cdna/src/sim/salu.rs")), "rows":rows}),
    )
}
fn main() -> ExitCode {
    let mut args = std::env::args_os().skip(1);
    let Some(dir) = args.next().map(PathBuf::from) else {
        eprintln!("usage: cdna_nor_hw_oracle OUT_DIR");
        return ExitCode::FAILURE;
    };
    if args.next().is_some() {
        eprintln!("unexpected argument");
        return ExitCode::FAILURE;
    }
    let result = run(&dir);
    let v = match &result {
        Ok(v) => v.clone(),
        Err(e) => json!({"status":"FAIL", "error":e}),
    };
    let path = dir.join("receipt.json");
    let saved = serde_json::to_vec_pretty(&v)
        .map_err(|e| e.to_string())
        .and_then(|b| std::fs::write(&path, b).map_err(|e| e.to_string()));
    println!("{v}");
    if result.is_ok() && saved.is_ok() {
        ExitCode::SUCCESS
    } else {
        eprintln!("oracle result: {result:?}; receipt write: {saved:?}");
        ExitCode::FAILURE
    }
}
