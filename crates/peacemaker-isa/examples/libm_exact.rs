// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! GPU exactness receipts of `peacemaker_isa::libm` against the hipcc objects (lab; run through the fabric).
//!
//! ```text
//! libm_exact <gfx1100|gfx1151|gfx1201> <libm_exact_ref.<arch>.co> <suite> [--chunk-log2 N] [--lo-log2 N]
//!            [--count N] [--seed S] [--flip-last] [--out FILE.json]
//! ```
//! Suites:
//!  * `sin`, `cos`: EVERY f32 bit pattern (2^32), `sinf` / `cosf` built through the libm module (both
//!    large-argument policies) vs the hipcc kernel, bitwise (NaN payloads included).
//!  * `sincos`: the shared-reduction `sincosf` kernel (both policies), sine vs `sinf` and cosine vs `cosf`.
//!  * `pow-random`: `--count` (default 2^28) mixed-distribution (x, y) pairs generated on the device.
//!  * `pow-edge`: the edge grid (IEEE classes, NaN payloads, subnormals, +-0, +-inf, integer / half-integer /
//!    large y, x near 1, the overflow / underflow boundary scans) and the RoPE domain (bases x exponents).
//!  * `negative`: controls that must show mismatches (PM sine vs hipcc cosine; PM `powf` vs rotated-exponent hipcc).
//!  * `--flip-last`: control; flips one output bit of the last element of the first variant (exactly one mismatch
//!    must be reported, and the exit code is then 1).
//!  * `all`: every suite above.
//! Exit 0 iff every comparison has zero mismatches.
use hip_bridge::{DeviceBuffer, Function, HipRuntime};
use peacemaker_isa::kernels::libm_exact::{emit_elf, Fun};
use peacemaker_isa::Arch;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::process::ExitCode;
use std::time::Instant;

struct Card {
    rt: HipRuntime,
}

impl Card {
    fn alloc(&self, bytes: usize) -> Result<DeviceBuffer, String> {
        self.rt
            .malloc(bytes.max(4))
            .map_err(|e| format!("hipMalloc: {e}"))
    }
    fn upload(&self, bytes: &[u8]) -> Result<DeviceBuffer, String> {
        let d = self.alloc(bytes.len())?;
        self.rt
            .memcpy_htod(&d, bytes)
            .map_err(|e| format!("upload: {e}"))?;
        Ok(d)
    }
    fn zero(&self, d: &DeviceBuffer, bytes: usize) -> Result<(), String> {
        self.rt
            .memset(d, 0, bytes)
            .map_err(|e| format!("memset: {e}"))
    }
    fn read(&self, d: &DeviceBuffer, n: usize) -> Result<Vec<u8>, String> {
        let mut v = vec![0u8; n];
        self.rt
            .memcpy_dtoh(&mut v, d)
            .map_err(|e| format!("readback: {e}"))?;
        Ok(v)
    }
    fn function(&self, elf: &[u8], symbol: &str) -> Result<Function, String> {
        let m = self
            .rt
            .module_load_data(elf)
            .map_err(|e| format!("hipModuleLoadData({symbol}): {e}"))?;
        let f = self
            .rt
            .module_get_function(&m, symbol)
            .map_err(|e| format!("hipModuleGetFunction({symbol}): {e}"))?;
        std::mem::forget(m);
        Ok(f)
    }
    fn launch(&self, f: &Function, grid: u32, block: u32, blob: &mut [u8]) -> Result<(), String> {
        // SAFETY: `blob` is the kernel's full kernarg image; every pointer in it is a live allocation of this
        // harness sized for the launch.
        unsafe {
            self.rt
                .launch_kernel_blob(f, [grid, 1, 1], [block, 1, 1], 0, None, blob)
        }
        .map_err(|e| format!("launch: {e}"))?;
        self.rt
            .device_synchronize()
            .map_err(|e| format!("synchronize: {e}"))
    }
}

fn ptr(d: &DeviceBuffer) -> u64 {
    d.as_ptr() as u64
}

/// hipcc kernarg image: the explicit arguments (already packed, natural alignment), then the code-object-v5
/// hidden arguments the kernels read (`hidden_block_count_*` u32 and `hidden_group_size_*` u16) at the next
/// 8-byte boundary, zero elsewhere.
fn hip_blob(explicit: &[u8], grid: u32, block: u32) -> Vec<u8> {
    let mut v = explicit.to_vec();
    while v.len() % 8 != 0 {
        v.push(0);
    }
    for x in [grid, 1, 1] {
        v.extend(x.to_le_bytes());
    }
    for x in [block as u16, 1, 1] {
        v.extend(x.to_le_bytes());
    }
    v.resize(v.len().max(256), 0);
    v
}

fn args(parts: &[(u64, usize)]) -> Vec<u8> {
    let mut v = Vec::new();
    for &(x, size) in parts {
        while v.len() % size != 0 {
            v.push(0);
        }
        v.extend(&x.to_le_bytes()[..size]);
    }
    v
}

/// PM kernel kernarg image: y @0, y2 @8, x @16, e @24, n @32.
fn pm_blob(y: u64, y2: u64, x: u64, e: u64, n: u32) -> Vec<u8> {
    let mut v = Vec::new();
    for p in [y, y2, x, e] {
        v.extend(p.to_le_bytes());
    }
    v.extend(n.to_le_bytes());
    v
}

struct Harness<'a> {
    card: &'a Card,
    arch: Arch,
    ref_sin: Function,
    ref_cos: Function,
    ref_pow: Function,
    iota: Function,
    gen_pow: Function,
    cmp: Function,
    flip: Function,
    flip_last: bool,
    cnt: DeviceBuffer,
    pm: std::collections::BTreeMap<String, (Function, String)>,
}

const BLOCK: u32 = 256;

impl<'a> Harness<'a> {
    fn pm_fn(&mut self, fun: Fun, masked: bool) -> Result<String, String> {
        let key = format!("{}{}", fun.name(), if masked { "_masked" } else { "" });
        if !self.pm.contains_key(&key) {
            let elf = emit_elf(self.arch, fun, masked)?;
            let sha = format!("{:x}", Sha256::digest(&elf));
            let sym = peacemaker_isa::kernels::libm_exact::symbol(self.arch, fun, masked);
            let f = self.card.function(&elf, &sym)?;
            self.pm.insert(key.clone(), (f, sha));
        }
        Ok(key)
    }

    fn grid(n: u32) -> u32 {
        n.div_ceil(BLOCK)
    }

    /// Mismatches between two f32 buffers of `n` words, bitwise: `(count, first up to 8 (index, ref, got))`.
    fn compare(
        &self,
        a: &DeviceBuffer,
        b: &DeviceBuffer,
        n: u32,
    ) -> Result<(u64, Vec<[u32; 3]>), String> {
        self.card.zero(&self.cnt, 4 * 32)?;
        let mut blob = hip_blob(
            &args(&[(ptr(a), 8), (ptr(b), 8), (n as u64, 4), (ptr(&self.cnt), 8)]),
            Self::grid(n),
            BLOCK,
        );
        self.card
            .launch(&self.cmp, Self::grid(n), BLOCK, &mut blob)?;
        let raw = self.card.read(&self.cnt, 4 * 32)?;
        let w: Vec<u32> = raw
            .chunks(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let first = (0..(w[0].min(8) as usize))
            .map(|k| [w[1 + 3 * k], w[2 + 3 * k], w[3 + 3 * k]])
            .collect();
        Ok((w[0] as u64, first))
    }

    fn ref_unary(
        &self,
        sin: bool,
        x: &DeviceBuffer,
        y: &DeviceBuffer,
        n: u32,
    ) -> Result<(), String> {
        let f = if sin { &self.ref_sin } else { &self.ref_cos };
        let mut blob = hip_blob(
            &args(&[(ptr(x), 8), (ptr(y), 8), (n as u64, 4)]),
            Self::grid(n),
            BLOCK,
        );
        self.card.launch(f, Self::grid(n), BLOCK, &mut blob)
    }

    fn pm_launch(
        &self,
        key: &str,
        y: &DeviceBuffer,
        y2: &DeviceBuffer,
        x: &DeviceBuffer,
        e: &DeviceBuffer,
        n: u32,
    ) -> Result<(), String> {
        let mut blob = pm_blob(ptr(y), ptr(y2), ptr(x), ptr(e), n);
        self.card
            .launch(&self.pm[key].0, Self::grid(n), BLOCK, &mut blob)
    }

    /// Exhaustive sin / cos / sincos over `[lo, lo + count)` bit patterns in chunks of `1 << chunk_log2`.
    fn trig(
        &mut self,
        suite: &str,
        lo: u64,
        count: u64,
        chunk_log2: u32,
    ) -> Result<Vec<Value>, String> {
        let chunk = 1u64 << chunk_log2;
        let words = chunk as usize * 4;
        let (x, yr_s, yr_c, yp, yp2, dummy_e) = (
            self.card.alloc(words)?,
            self.card.alloc(words)?,
            self.card.alloc(words)?,
            self.card.alloc(words)?,
            self.card.alloc(words)?,
            self.card.alloc(4)?,
        );
        let variants: Vec<(Fun, bool)> = match suite {
            "sin" => vec![(Fun::Sinf, false), (Fun::Sinf, true)],
            "cos" => vec![(Fun::Cosf, false), (Fun::Cosf, true)],
            _ => vec![(Fun::Sincosf, false), (Fun::Sincosf, true)],
        };
        let fns: Vec<String> = variants
            .iter()
            .map(|&(f, m)| self.pm_fn(f, m))
            .collect::<Result<_, _>>()?;
        let mut mism = vec![0u64; variants.len() * 2];
        let mut firsts: Vec<Vec<[u32; 3]>> = vec![vec![]; variants.len() * 2];
        let mut done = 0u64;
        let t0 = Instant::now();
        while done < count {
            let n = chunk.min(count - done) as u32;
            let base = (lo + done) as u32;
            let mut blob = hip_blob(
                &args(&[(ptr(&x), 8), (base as u64, 4), (n as u64, 4)]),
                Self::grid(n),
                BLOCK,
            );
            self.card
                .launch(&self.iota, Self::grid(n), BLOCK, &mut blob)?;
            self.ref_unary(true, &x, &yr_s, n)?;
            self.ref_unary(false, &x, &yr_c, n)?;
            for (i, ((fun, _), f)) in variants.iter().zip(&fns).enumerate() {
                self.card.zero(&yp, words)?;
                self.card.zero(&yp2, words)?;
                self.pm_launch(f, &yp, &yp2, &x, &dummy_e, n)?;
                if self.flip_last && done + n as u64 == count && i == 0 {
                    // control: one flipped output bit in the very last element must be the one reported mismatch.
                    let mut blob = hip_blob(&args(&[(ptr(&yp), 8), (n as u64 - 1, 4)]), 1, 1);
                    self.card.launch(&self.flip, 1, 1, &mut blob)?;
                }
                let mut check =
                    |slot: usize, r: &DeviceBuffer, g: &DeviceBuffer| -> Result<(), String> {
                        let (c, first) = self.compare(r, g, n)?;
                        if c > 0 {
                            for mut t in first {
                                t[0] = t[0].wrapping_add(base);
                                if firsts[slot].len() < 8 {
                                    firsts[slot].push(t);
                                }
                            }
                        }
                        mism[slot] += c;
                        Ok(())
                    };
                match fun {
                    Fun::Sinf => check(2 * i, &yr_s, &yp)?,
                    Fun::Cosf => check(2 * i, &yr_c, &yp)?,
                    _ => {
                        check(2 * i, &yr_s, &yp)?;
                        check(2 * i + 1, &yr_c, &yp2)?;
                    }
                }
            }
            done += n as u64;
        }
        let secs = t0.elapsed().as_secs_f64();
        let mut out = Vec::new();
        for (i, (fun, masked)) in variants.iter().enumerate() {
            let slots: Vec<(usize, &str)> = match fun {
                Fun::Sincosf => vec![(2 * i, "sin_out"), (2 * i + 1, "cos_out")],
                _ => vec![(2 * i, "out")],
            };
            for (slot, what) in slots {
                out.push(json!({
                    "suite": suite, "fun": fun.name(), "masked": masked, "output": what,
                    "inputs": count, "lo": lo, "mismatches": mism[slot],
                    "first_mismatches_idx_ref_got": firsts[slot],
                    "pm_elf_sha256": self.pm[&format!("{}{}", fun.name(), if *masked { "_masked" } else { "" })].1,
                    "seconds": secs,
                }));
            }
        }
        Ok(out)
    }

    /// Negative controls (the harness must be able to see a difference): the PM sine against the hipcc cosine, and
    /// the PM `powf` against the hipcc `powf` of the exponent buffer rotated by one element. Both must show
    /// mismatches; `mismatches` carries the count and `expected_nonzero` marks the row.
    fn negative(&mut self) -> Result<Vec<Value>, String> {
        let n = 1u32 << 22;
        let words = n as usize * 4;
        let (x, yr_c, yp, yp2, e, e2, yr, yq) = (
            self.card.alloc(words)?,
            self.card.alloc(words)?,
            self.card.alloc(words)?,
            self.card.alloc(words)?,
            self.card.alloc(words)?,
            self.card.alloc(words)?,
            self.card.alloc(words)?,
            self.card.alloc(words)?,
        );
        // x: iota from the exponent-1 region so sin / cos are non-trivial.
        let mut blob = hip_blob(
            &args(&[(ptr(&x), 8), (0x3f00_0000u64, 4), (n as u64, 4)]),
            Self::grid(n),
            BLOCK,
        );
        self.card
            .launch(&self.iota, Self::grid(n), BLOCK, &mut blob)?;
        self.ref_unary(false, &x, &yr_c, n)?;
        let key = self.pm_fn(Fun::Sinf, false)?;
        self.pm_launch(&key, &yp, &yp2, &x, &e, n)?;
        let (sin_vs_cos, _) = self.compare(&yr_c, &yp, n)?;
        // powf: exponent buffer and a rotated copy.
        let mut blob = hip_blob(
            &args(&[(ptr(&x), 8), (ptr(&e), 8), (0x1234u64, 8), (n as u64, 4)]),
            Self::grid(n),
            BLOCK,
        );
        self.card
            .launch(&self.gen_pow, Self::grid(n), BLOCK, &mut blob)?;
        let eb = self.card.read(&e, words)?;
        let mut rot = eb.clone();
        rot.rotate_left(4);
        self.card
            .rt
            .memcpy_htod(&e2, &rot)
            .map_err(|e| format!("upload: {e}"))?;
        let mut blob = hip_blob(
            &args(&[(ptr(&x), 8), (ptr(&e2), 8), (ptr(&yr), 8), (n as u64, 4)]),
            Self::grid(n),
            BLOCK,
        );
        self.card
            .launch(&self.ref_pow, Self::grid(n), BLOCK, &mut blob)?;
        let key = self.pm_fn(Fun::Powf, false)?;
        self.pm_launch(&key, &yq, &yp2, &x, &e, n)?;
        let (pow_rot, _) = self.compare(&yr, &yq, n)?;
        Ok(vec![
            json!({"suite": "negative", "set": "pm_sinf_vs_hipcc_cosf", "inputs": n, "mismatches": sin_vs_cos, "expected_nonzero": true}),
            json!({"suite": "negative", "set": "pm_powf_vs_hipcc_powf_rotated_exponents", "inputs": n, "mismatches": pow_rot, "expected_nonzero": true}),
        ])
    }

    /// `powf` over host-provided pairs (chunked).
    fn pow_pairs(&mut self, label: &str, xs: &[u32], es: &[u32]) -> Result<Value, String> {
        let n_total = xs.len();
        let chunk = 1usize << 24;
        let f = self.pm_fn(Fun::Powf, false)?;
        let (dx, de, yr, yp, y2) = (
            self.card.alloc(chunk * 4)?,
            self.card.alloc(chunk * 4)?,
            self.card.alloc(chunk * 4)?,
            self.card.alloc(chunk * 4)?,
            self.card.alloc(4)?,
        );
        let (mut mism, mut firsts) = (0u64, Vec::<[u32; 4]>::new());
        let mut done = 0usize;
        while done < n_total {
            let n = chunk.min(n_total - done);
            let bx: Vec<u8> = xs[done..done + n]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            let be: Vec<u8> = es[done..done + n]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            self.card
                .rt
                .memcpy_htod(&dx, &bx)
                .map_err(|e| format!("upload: {e}"))?;
            self.card
                .rt
                .memcpy_htod(&de, &be)
                .map_err(|e| format!("upload: {e}"))?;
            let (c, first) = self.pow_chunk(&f, &dx, &de, &yr, &yp, &y2, n as u32)?;
            for t in first {
                if firsts.len() < 8 {
                    let i = t[0] as usize + done;
                    firsts.push([xs[i], es[i], t[1], t[2]]);
                }
            }
            mism += c;
            done += n;
        }
        Ok(json!({
            "suite": "pow", "set": label, "inputs": n_total, "mismatches": mism,
            "first_mismatches_x_y_ref_got": firsts,
            "pm_elf_sha256": self.pm["powf"].1,
        }))
    }

    #[allow(clippy::too_many_arguments)]
    fn pow_chunk(
        &self,
        f: &str,
        dx: &DeviceBuffer,
        de: &DeviceBuffer,
        yr: &DeviceBuffer,
        yp: &DeviceBuffer,
        y2: &DeviceBuffer,
        n: u32,
    ) -> Result<(u64, Vec<[u32; 3]>), String> {
        let mut blob = hip_blob(
            &args(&[(ptr(dx), 8), (ptr(de), 8), (ptr(yr), 8), (n as u64, 4)]),
            Self::grid(n),
            BLOCK,
        );
        self.card
            .launch(&self.ref_pow, Self::grid(n), BLOCK, &mut blob)?;
        self.card.zero(yp, n as usize * 4)?;
        self.pm_launch(f, yp, y2, dx, de, n)?;
        self.compare(yr, yp, n)
    }

    /// `count` random pairs generated on the device (the hipcc `k_gen_pow` distribution mix).
    fn pow_random(&mut self, count: u64, seed: u64) -> Result<Value, String> {
        let chunk = 1u64 << 24;
        let f = self.pm_fn(Fun::Powf, false)?;
        let (dx, de, yr, yp, y2) = (
            self.card.alloc(chunk as usize * 4)?,
            self.card.alloc(chunk as usize * 4)?,
            self.card.alloc(chunk as usize * 4)?,
            self.card.alloc(chunk as usize * 4)?,
            self.card.alloc(4)?,
        );
        let (mut mism, mut firsts, mut done) = (0u64, Vec::<[u32; 4]>::new(), 0u64);
        let t0 = Instant::now();
        while done < count {
            let n = chunk.min(count - done) as u32;
            let s = seed ^ (done.wrapping_mul(0x9e37_79b9_7f4a_7c15));
            let mut blob = hip_blob(
                &args(&[(ptr(&dx), 8), (ptr(&de), 8), (s, 8), (n as u64, 4)]),
                Self::grid(n),
                BLOCK,
            );
            self.card
                .launch(&self.gen_pow, Self::grid(n), BLOCK, &mut blob)?;
            let (c, first) = self.pow_chunk(&f, &dx, &de, &yr, &yp, &y2, n)?;
            if !first.is_empty() && firsts.len() < 8 {
                let xs = self.card.read(&dx, n as usize * 4)?;
                let es = self.card.read(&de, n as usize * 4)?;
                let w = |b: &[u8], i: usize| {
                    u32::from_le_bytes(b[4 * i..4 * i + 4].try_into().unwrap())
                };
                for t in first.iter().take(8 - firsts.len()) {
                    firsts.push([w(&xs, t[0] as usize), w(&es, t[0] as usize), t[1], t[2]]);
                }
            }
            mism += c;
            done += n as u64;
        }
        Ok(json!({
            "suite": "pow", "set": "random", "inputs": count, "seed": seed, "mismatches": mism,
            "first_mismatches_x_y_ref_got": firsts, "seconds": t0.elapsed().as_secs_f64(),
            "pm_elf_sha256": self.pm["powf"].1,
        }))
    }
}

// ------------------------------------------------------------------ edge grid

fn f(b: u32) -> f32 {
    f32::from_bits(b)
}

fn specials() -> Vec<u32> {
    let mut s: Vec<u32> = vec![
        0x0000_0000,
        0x8000_0000,
        0x7f80_0000,
        0xff80_0000, // +-0, +-inf
        0x7fc0_0000,
        0xffc0_0000,
        0x7f80_0001,
        0xff80_0001,
        0x7fff_ffff,
        0xffff_ffff,
        0x7fc1_2345,
        0xffc5_4321,
        0x7fa0_0000, // NaNs
        0x0000_0001,
        0x0000_0002,
        0x0000_0100,
        0x0040_0000,
        0x007f_ffff,
        0x8000_0001,
        0x8040_0000,
        0x807f_ffff, // subnormals
        0x0080_0000,
        0x0080_0001,
        0x8080_0000,
        0x7f7f_ffff,
        0xff7f_ffff,
        0x7f7f_fffe, // min normal / max
        0x3f80_0000,
        0xbf80_0000,
        0x3f80_0001,
        0x3f7f_ffff,
        0xbf80_0001,
        0xbf7f_ffff, // +-1, 1 +- ulp
    ];
    for v in [
        2.0f32,
        0.5,
        3.0,
        0.25,
        10.0,
        1e4,
        1e6,
        5e6,
        1e7,
        1e5,
        5e5,
        1e3,
        1e8,
        2.5,
        3.5,
        1.5,
        0.1,
        0.7,
        100.0,
        1000.0,
        65504.0,
        1e-10,
        1e10,
        1e20,
        1e-20,
        1e30,
        1e-30,
        1e38,
        1e-38,
        16777216.0,
        8388608.0,
        8388609.0,
        2147483648.0,
        9.223372e18,
        1.8446744e19,
        std::f32::consts::E,
        std::f32::consts::PI,
        std::f32::consts::LN_2,
        std::f32::consts::SQRT_2,
        0.999_999_94,
        1.000_000_1,
        88.72284,
        87.33654,
        128.0,
        127.0,
        126.0,
        149.0,
        150.0,
        1.0 / 3.0,
        2.0 / 3.0,
    ] {
        s.push(v.to_bits());
        s.push((-v).to_bits());
    }
    s.sort_unstable();
    s.dedup();
    s
}

fn edge_pairs() -> (Vec<u32>, Vec<u32>) {
    let (mut xs, mut es) = (Vec::new(), Vec::new());
    let mut push = |x: u32, y: u32| {
        xs.push(x);
        es.push(y);
    };
    let sp = specials();
    // special x special
    for &x in &sp {
        for &y in &sp {
            push(x, y);
        }
    }
    // special x integer and half-integer y in [-300, 300], and 1/3-ish fractions
    let mut ys: Vec<u32> = Vec::new();
    for k in -600..=600 {
        ys.push((k as f32 * 0.5).to_bits());
    }
    for k in [-7, -5, -3, -2, 2, 3, 5, 7, 11, 13] {
        ys.push((1.0f32 / k as f32).to_bits());
        ys.push((k as f32 / 3.0).to_bits());
    }
    for &x in &sp {
        for &y in &ys {
            push(x, y);
        }
    }
    // x in a dense grid of positive and negative magnitudes x integer y
    for e in (0u32..255).step_by(3) {
        for m in [
            0u32, 1, 0x1234, 0x400000, 0x7ffffe, 0x7fffff, 0x2aaaab, 0x555555,
        ] {
            for sign in [0u32, 0x8000_0000] {
                let x = sign | (e << 23) | m;
                for y in [
                    -3.0f32,
                    -2.0,
                    -1.0,
                    0.5,
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                    -0.5,
                    1.5,
                    0.333_333_34,
                    10.0,
                    127.0,
                    -127.0,
                    0.0,
                    -0.0,
                ] {
                    push(x, y.to_bits());
                }
            }
        }
    }
    // x near 1 (+-4096 ulp) x big y
    for d in -4096i32..=4096 {
        let x = (0x3f80_0000i32 + d) as u32;
        for y in [
            1e3f32, -1e3, 1e5, -1e5, 1e6, 3e7, -3e7, 16777216.0, 33554432.0, 0.5, 2.0, -2.0, 1e9,
            3e9, 1e38,
        ] {
            push(x, y.to_bits());
        }
    }
    // overflow / underflow boundary scans: x ~ 2^(128/y), 2^(-126/y), 2^(-149/y), 2^(-150/y) for several y, +-2048 ulps
    for y in [
        2.0f32,
        3.0,
        4.0,
        0.5,
        1.5,
        10.0,
        100.0,
        0.1,
        -1.0,
        -2.0,
        -0.5,
        -10.0,
        -100.0,
        1.0 / 3.0,
        7.0,
        31.0,
        64.0,
        127.0,
        128.0,
        -127.0,
        -128.0,
        -149.0,
        -150.0,
        0.25,
        8.0,
    ] {
        for t in [128.0f64, -126.0, -149.0, -150.0, 127.99999, -127.0] {
            let center = (t / y as f64).exp2() as f32;
            if !center.is_finite() || center == 0.0 {
                continue;
            }
            let c = center.to_bits() as i64;
            for d in -2048i64..=2048 {
                let x = (c + d) as u32;
                push(x, y.to_bits());
                push(x | 0x8000_0000, y.to_bits());
            }
        }
    }
    (xs, es)
}

/// RoPE bases x `(float)(2 i) / (float)n_rot` exponents, both signs.
fn rope_pairs() -> (Vec<u32>, Vec<u32>) {
    let (mut xs, mut es) = (Vec::new(), Vec::new());
    for base in [
        1e4f32,
        1e6,
        5e6,
        1e7,
        1e5,
        5e5,
        1e3,
        1e8,
        10000.0,
        500000.0,
        1_000_000.0,
        10_000_000.0,
        2.0,
        100.0,
    ] {
        for n in [16u32, 32, 48, 64, 80, 96, 112, 128, 192, 256, 512] {
            for i in 0..n {
                let e = (2 * i) as f32 / n as f32;
                for y in [e, -e] {
                    xs.push(base.to_bits());
                    es.push(y.to_bits());
                }
            }
        }
    }
    (xs, es)
}

fn main() -> ExitCode {
    match run() {
        Ok(ok) => ExitCode::from(if ok { 0 } else { 1 }),
        Err(e) => {
            eprintln!("libm_exact: {e}");
            ExitCode::from(2)
        }
    }
}

fn run() -> Result<bool, String> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 4 {
        return Err("usage: libm_exact <arch> <ref.co> <sin|cos|sincos|pow-random|pow-edge|all> [--chunk-log2 N] [--lo-log2 N] [--count N] [--seed S] [--out FILE]".into());
    }
    let arch: Arch = a[1].parse().map_err(|e: String| e)?;
    let co = std::fs::read(&a[2]).map_err(|e| format!("{}: {e}", a[2]))?;
    let suite = a[3].as_str();
    let opt = |name: &str| {
        a.iter()
            .position(|x| x == name)
            .and_then(|i| a.get(i + 1))
            .cloned()
    };
    let chunk_log2: u32 =
        opt("--chunk-log2").map_or(Ok(26), |v| v.parse().map_err(|_| "bad --chunk-log2"))?;
    let count: Option<u64> = opt("--count")
        .map(|v| v.parse().map_err(|_| "bad --count"))
        .transpose()?;
    let seed: u64 = opt("--seed").map_or(Ok(0x5eed_1ba5_e11e_7777), |v| {
        v.parse().map_err(|_| "bad --seed")
    })?;
    let lo: u64 = opt("--lo").map_or(Ok(0), |v| v.parse().map_err(|_| "bad --lo"))?;
    let rt = HipRuntime::load().map_err(|e| format!("hip: {e}"))?;
    let card = Card { rt };
    let mut h = Harness {
        arch,
        ref_sin: card.function(&co, "k_sinf")?,
        ref_cos: card.function(&co, "k_cosf")?,
        ref_pow: card.function(&co, "k_powf")?,
        iota: card.function(&co, "k_iota")?,
        gen_pow: card.function(&co, "k_gen_pow")?,
        cmp: card.function(&co, "k_cmp")?,
        flip: card.function(&co, "k_flip")?,
        flip_last: a.iter().any(|x| x == "--flip-last"),
        cnt: card.alloc(4 * 32)?,
        pm: Default::default(),
        card: &card,
    };
    let t0 = Instant::now();
    let mut rows: Vec<Value> = Vec::new();
    let total_trig = count.unwrap_or(1u64 << 32);
    let suites: Vec<&str> = if suite == "all" {
        vec!["negative", "sin", "cos", "sincos", "pow-edge", "pow-random"]
    } else {
        vec![suite]
    };
    for s in suites {
        match s {
            "sin" | "cos" | "sincos" => rows.extend(h.trig(s, lo, total_trig, chunk_log2)?),
            "pow-random" => rows.push(h.pow_random(count.unwrap_or(1 << 28), seed)?),
            "pow-edge" => {
                let (xs, es) = edge_pairs();
                rows.push(h.pow_pairs("edge", &xs, &es)?);
                let (xs, es) = rope_pairs();
                rows.push(h.pow_pairs("rope", &xs, &es)?);
            }
            "negative" => rows.extend(h.negative()?),
            other => return Err(format!("unknown suite {other}")),
        }
    }
    let ok = rows.iter().all(|r| {
        if r.get("expected_nonzero").is_some() {
            r["mismatches"].as_u64().unwrap_or(0) > 0
        } else {
            r["mismatches"] == 0
        }
    });
    let report = json!({
        "arch": arch.name(), "ref_co": a[2], "ref_co_sha256": format!("{:x}", Sha256::digest(&co)),
        "seconds": t0.elapsed().as_secs_f64(), "all_zero_mismatches": ok, "rows": rows,
    });
    let text = serde_json::to_string_pretty(&report).unwrap();
    if let Some(out) = opt("--out") {
        std::fs::write(&out, &text).map_err(|e| format!("{out}: {e}"))?;
    }
    println!("{text}");
    Ok(ok)
}
