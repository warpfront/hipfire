// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! MI300X (gfx942) GPU bit oracle: the PeaceMaker twin of `kernels/src/gemm_bf16_mfma.gfx942.hip`'s few-row entries
//! (`gemm_bf16_mfma_gfx942_rows16`, `gemm_bf16_mfma_gfx942_rows16_x2`) against the hipcc code object, on the card
//! (lab; run through the fabric on the MI300X lock). Both images are loaded with `hipModuleLoadData` and launched with
//! `hipModuleLaunchKernel` (hip-bridge `launch_kernel_blob`, the kernarg image the production launcher
//! `Gpu::launch_gemm_bf16_mfma_rows16` builds), on the same device inputs, and the outputs are compared as RAW BYTES.
//!
//! ```text
//! gemm_bf16_rows16_oracle --ref <hipcc.co> --pm <twin.hxaco> [--out FILE.json] [--seed S] [--device N]
//!                         [--only SUBSTR] [--skip-big] [--expect-pm-sha256 HEX] [--expect-ref-sha256 HEX]
//! ```
//! Typical invocation (repo root; `kernels/` holds the sealed twin and its hipcc anchor):
//!
//! ```text
//! cargo build --release -p peacemaker-isa --features lab --example gemm_bf16_rows16_oracle
//! target/release/examples/gemm_bf16_rows16_oracle \
//!   --ref kernels/gemm_bf16_mfma_gfx942_rows16_hipcc_gfx942.co \
//!   --pm  kernels/gemm_bf16_mfma_gfx942_rows16_pm_gfx942.hxaco \
//!   --expect-pm-sha256 "$(cat kernels/gemm_bf16_mfma_gfx942_rows16_pm_gfx942.hxaco.sha256)" \
//!   --out <receipt-dir>/gemm_bf16_mfma_gfx942_rows16.gpu_oracle.json
//! ```
//!
//! * `--ref`: the hipcc object (bare device ELF as `hipcc --genco` writes it, or an uncompressed clang offload bundle).
//! * `--pm`: the committed twin `.hxaco` (clang offload bundle); its `hipv4-amdgcn-amd-amdhsa--gfx942` entry is the
//!   device ELF that is loaded. A bare ELF is accepted too. The report's `input_hxaco_sha256` is the sha256 of the
//!   `--pm` FILE exactly as read; `--expect-pm-sha256` / `--expect-ref-sha256` abort (exit 2) when the file differs.
//! * `--device N`: HIP device index (default 0); the device must report `gfx942`.
//! * `--only SUBSTR`: run only the shapes whose name contains SUBSTR (the committed receipt must be a full run).
//! * `--skip-big`: skip the shapes whose weights exceed 256 MiB (`lm_head`); a smoke option, not a receipt.
//!
//! Matrix (per shape x data profile; every combination is a case):
//!  * rows 1, 7, 8, 9, 16; X as F32 or BF16; each member's D as F32 or BF16 (d_bf16_mask: none, all, alternating).
//!  * kernels: `rows16` (BF16 X at every row count; F32 X at rows <= 8) and `rows16_x2` (every combination; it is the
//!    production route for F32 X at rows > 8). `rows16` with F32 X at rows > 8 is outside the kernel's contract and
//!    is not launched.
//!  * shapes: every Flash-Next decode projection (M x K of the r16bench table, incl. the 248320-row lm_head), the
//!    in-proj groups (4 members), a 2-member and a 3-member group, tail K (K % 128 / % 256 != 0, K = 8 .. 16384) and
//!    tail M (M % 16 != 0, M = 1 ..), groups 1..=4 members with unused (M = 0, null) trailing members; `packed_*`
//!    shapes place every weight (and X) at 16-byte, non-256-byte aligned offsets of one slab.
//!  * data profiles (deterministic counter-based generator keyed by `--seed`; weights and X are streamed to the
//!    device in 16 MiB-bounded chunks, never pooled): `random`; `edge` (+-0, subnormals, +-max finite, exact
//!    round-to-nearest-even ties and +-1 ulp neighbours, +-1, 2^+-100, the value that rounds up to inf);
//!    `nan_sparse` (+-inf, quiet / signalling NaN payloads at 5/65536); `nan_dense` (the same at 5/256). Shapes
//!    with more than 256 MiB of weights (`lm_head`) run `random` + `nan_sparse` only.
//!
//! Per case, each arm is launched twice into D buffers poisoned with 0xA5 and 0x5A. Required, per member:
//! ref bytes == pm bytes for the whole `rows * M` output (both poison runs, NaN payloads included); each arm's two
//! poison runs identical (every output element written, deterministic); the 256-byte guard past the output untouched.
//! Before the matrix, built-in controls must fire (pm on a perturbed X differs from ref; a single flipped host byte
//! is detected) or the harness reports itself blind (exit 2).
//!
//! Tiled comparator: the shipped tiled kernel `gemm_bf16_mfma_gfx942` of the hipcc anchor image is launched on the
//! same weights for every member and every (rows, X dtype), into F32 (two poison runs that must agree). F32 X is first
//! staged to BF16 with the route's `convert_f32_to_bf16` rule (RNE; NaN -> sign | 0x7fc0, sign taken from the F32
//! bits); BF16 X is passed as is. A member whose `d_bf16_mask` bit is set expects each tiled F32 word rounded with that
//! same rule (sign of a NaN from the tiled F32 bits, as the rows16 kernel does). Both the hipcc rows16 output and the PM
//! rows16 output must equal the expected bytes over the whole `rows * M` output (`hipcc_ne_tiled` / `pm_ne_tiled`).
//! The tiled kernel is not taken from the PM image.
//!
//! The receipt records, per case, the sha256 of the exact input image (profile, seed, shape, rows, dtypes, mask and
//! the sha256 of every streamed weight and X byte) and of each arm's output; and the sha256 of the hxaco file, of both
//! device ELFs, and of the concatenation of all case input digests.
//!
//! Exit code: 0 iff every comparison is byte-equal; 1 on any mismatch; 2 on a harness / runtime error.
use hip_bridge::{DeviceBuffer, Function, HipRuntime, KernargBlob, Module};
use peacemaker_ir::envelope::Bundle;
use peacemaker_lift::bundle::BundleCodec;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::ffi::c_void;
use std::process::ExitCode;
use std::time::Instant;

const SYM_R16: &str = "gemm_bf16_mfma_gfx942_rows16";
const SYM_X2: &str = "gemm_bf16_mfma_gfx942_rows16_x2";
const SYM_TILED: &str = "gemm_bf16_mfma_gfx942";
const DEVICE_TRIPLE: &str = "hipv4-amdgcn-amd-amdhsa--gfx942";
const EF_AMDGPU_MACH_GFX942: u32 = 0x4c;
/// `GEMM_BF16_MFMA_ROWS16_LDS_PITCH` / `R16_LD` (bf16 elements).
const R16_LD: usize = 260;
const BLOCK: u32 = 576;
const ROWS: [usize; 5] = [1, 7, 8, 9, 16];
/// Bytes of poisoned guard compared past every member's output.
const GUARD: usize = 256;
const POISON: [u8; 2] = [0xA5, 0x5A];
/// Elements per streamed upload chunk (F32: 16 MiB).
const CHUNK: usize = 1 << 22;
/// Weights above this many bytes run the reduced profile set.
const BIG_BYTES: u64 = 256 << 20;
const STREAM_A: u64 = 0x1000;
const STREAM_X: u64 = 0x2000;
const MAX_FAILURES_KEPT: usize = 64;

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn sha_hex(b: &[u8]) -> String {
    hex(&Sha256::digest(b))
}

// ───────────────────────────── device images ─────────────────────────────

/// The gfx942 device ELF of `bytes`: the bytes themselves when they are a bare ELF, else the single
/// `hipv4-amdgcn-amd-amdhsa--gfx942` entry of the offload bundle.
fn device_image<'a>(bytes: &'a [u8], label: &str) -> Result<&'a [u8], String> {
    let elf: &[u8] = if Bundle::is_bundle(bytes) {
        let (bundle, payloads) =
            Bundle::read(bytes).map_err(|e| format!("{label}: bundle: {e}"))?;
        let hits: Vec<usize> = bundle
            .entries
            .iter()
            .enumerate()
            .filter(|(_, e)| {
                e.triple
                    .strip_prefix(DEVICE_TRIPLE)
                    .is_some_and(|rest| rest.is_empty() || rest.starts_with(':'))
            })
            .map(|(i, _)| i)
            .collect();
        match hits[..] {
            [i] => payloads[i],
            _ => {
                return Err(format!(
                    "{label}: bundle has {} `{DEVICE_TRIPLE}` entries (want 1); entries {:?}",
                    hits.len(),
                    bundle.entries.iter().map(|e| &e.triple).collect::<Vec<_>>()
                ))
            }
        }
    } else {
        bytes
    };
    if elf.len() < 64 || &elf[..4] != b"\x7fELF" || elf[4] != 2 || elf[5] != 1 {
        return Err(format!(
            "{label}: device image is not a 64-bit little-endian ELF"
        ));
    }
    let machine = u16::from_le_bytes([elf[18], elf[19]]);
    let flags = u32::from_le_bytes([elf[48], elf[49], elf[50], elf[51]]);
    if machine != 224 || flags & 0xff != EF_AMDGPU_MACH_GFX942 {
        return Err(format!(
            "{label}: device image is not an AMDGPU gfx942 code object (e_machine {machine}, e_flags {flags:#x})"
        ));
    }
    Ok(elf)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Sym {
    R16,
    X2,
}

impl Sym {
    fn name(self) -> &'static str {
        match self {
            Sym::R16 => SYM_R16,
            Sym::X2 => SYM_X2,
        }
    }
}

struct Image {
    file_sha: String,
    elf_sha: String,
    elf_bytes: usize,
    module: Module,
    has_tiled: bool,
}

struct Card {
    rt: HipRuntime,
}

impl Card {
    fn alloc(&self, bytes: usize) -> Result<DeviceBuffer, String> {
        self.rt
            .malloc(bytes.max(16))
            .map_err(|e| format!("hipMalloc({bytes}): {e}"))
    }
    fn upload_at(&self, d: &DeviceBuffer, off: usize, b: &[u8]) -> Result<(), String> {
        self.rt
            .memcpy_htod_offset(d, off, b)
            .map_err(|e| format!("upload: {e}"))
    }
    fn memset(&self, d: &DeviceBuffer, value: u8, bytes: usize) -> Result<(), String> {
        self.rt
            .memset(d, i32::from(value), bytes)
            .map_err(|e| format!("memset: {e}"))
    }
    fn read(&self, d: &DeviceBuffer, dst: &mut [u8]) -> Result<(), String> {
        self.rt
            .memcpy_dtoh(dst, d)
            .map_err(|e| format!("readback: {e}"))
    }
    fn sync(&self) -> Result<(), String> {
        self.rt
            .device_synchronize()
            .map_err(|e| format!("synchronize: {e}"))
    }
    fn free(&self, d: DeviceBuffer) -> Result<(), String> {
        self.rt.free(d).map_err(|e| format!("hipFree: {e}"))
    }

    fn load(&self, path: &str, label: &str, expect_sha: Option<&str>) -> Result<Image, String> {
        let raw = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
        let file_sha = sha_hex(&raw);
        if let Some(want) = expect_sha {
            if !want.eq_ignore_ascii_case(&file_sha) {
                return Err(format!(
                    "{label} {path}: sha256 {file_sha} != expected {want}"
                ));
            }
        }
        let elf = device_image(&raw, label)?;
        let module = self
            .rt
            .module_load_data(elf)
            .map_err(|e| format!("{label}: hipModuleLoadData: {e}"))?;
        self.rt
            .module_get_function(&module, SYM_R16)
            .map_err(|e| format!("{label}: hipModuleGetFunction({SYM_R16}): {e}"))?;
        self.rt
            .module_get_function(&module, SYM_X2)
            .map_err(|e| format!("{label}: hipModuleGetFunction({SYM_X2}): {e}"))?;
        let has_tiled = self.rt.module_get_function(&module, SYM_TILED).is_ok();
        Ok(Image {
            file_sha,
            elf_sha: sha_hex(elf),
            elf_bytes: elf.len(),
            module,
            has_tiled,
        })
    }
}

fn ptr(d: &DeviceBuffer) -> u64 {
    d.as_ptr() as u64
}

// ───────────────────────────── host RAM guard ─────────────────────────────

fn meminfo() -> Result<(u64, u64), String> {
    let text =
        std::fs::read_to_string("/proc/meminfo").map_err(|e| format!("/proc/meminfo: {e}"))?;
    let field = |name: &str| -> Result<u64, String> {
        text.lines()
            .find_map(|l| l.strip_prefix(name))
            .and_then(|r| r.trim().trim_end_matches("kB").trim().parse::<u64>().ok())
            .map(|kb| kb * 1024)
            .ok_or_else(|| format!("/proc/meminfo: no {name}"))
    };
    Ok((field("MemTotal:")?, field("MemAvailable:")?))
}

/// Refuse to proceed when `planned` more host bytes would leave less than 25% of host RAM available.
fn host_ram_check(planned: u64) -> Result<(), String> {
    let (total, avail) = meminfo()?;
    if avail.saturating_sub(planned) < total / 4 {
        return Err(format!(
            "host RAM guard: {} MiB available of {} MiB, {} MiB planned; at least 25% must stay free",
            avail >> 20,
            total >> 20,
            planned >> 20
        ));
    }
    Ok(())
}

// ───────────────────────────── deterministic data ─────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq)]
enum Profile {
    Random,
    Edge,
    NanSparse,
    NanDense,
}

impl Profile {
    fn name(self) -> &'static str {
        match self {
            Profile::Random => "random",
            Profile::Edge => "edge",
            Profile::NanSparse => "nan_sparse",
            Profile::NanDense => "nan_dense",
        }
    }
}

fn mix(z: u64) -> u64 {
    let mut z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// F32 bit pattern of element draw `r` under `p`; `scale` is the width of the uniform base distribution.
fn f32_bits(p: Profile, r: u64, scale: f32) -> u32 {
    let u = ((r as u32) >> 8) as f32 / 16_777_216.0;
    let base = ((u - 0.5) * scale).to_bits();
    let sign = ((r >> 63) as u32) << 31;
    let class = ((r >> 40) & 0xffff) as u32;
    match p {
        Profile::Random => base,
        Profile::Edge => match class & 15 {
            0 => 0,
            1 => 0x8000_0000,
            2 => sign | ((r >> 8) as u32 & 0x007f_ffff),
            3 => sign | 0x7f7f_ffff,
            // exact bf16 round-to-nearest-even ties (even / odd low bit) and +-1 neighbours
            4 => (base & 0xfffe_0000) | 0x8000,
            5 => (base & 0xffff_0000) | 0x1_8000,
            6 => (base & 0xffff_0000) | 0x8001,
            7 => (base & 0xffff_0000) | 0x7fff,
            8 => sign | 0x3f80_0000,
            9 => sign | 0x7180_0000,
            10 => sign | 0x0d80_0000,
            // max finite + half ulp: rounds up to infinity
            11 => sign | 0x7f7f_8000,
            _ => base,
        },
        Profile::NanSparse | Profile::NanDense => {
            let c = if p == Profile::NanDense {
                class & 0xff
            } else {
                class
            };
            match c {
                0 => 0x7f80_0000,
                1 => 0xff80_0000,
                2 => sign | 0x7fc0_1234,
                3 => sign | 0x7f80_0001,
                4 => sign | 0x7fff_ffff,
                _ => base,
            }
        }
    }
}

/// RNE of an F32 bit pattern to BF16 bits (finite and infinite inputs; the kernel's rule).
fn bf16_rne(bits: u32) -> u16 {
    let lsb = (bits >> 16) & 1;
    let round = (bits >> 15) & 1;
    let sticky = u32::from(bits & 0x7fff != 0);
    ((bits >> 16) + (round & (sticky | lsb))) as u16
}

/// BF16 storage of an F32 draw: RNE, NaN kept as a quiet NaN with its sign and high payload (weights and BF16 X: every
/// arm reads the same bits).
fn bf16_store(bits: u32) -> u16 {
    if bits & 0x7fff_ffff > 0x7f80_0000 {
        return ((bits >> 16) as u16) | 0x0040;
    }
    bf16_rne(bits)
}

/// `convert_f32_to_bf16` / `r16_f32_bits_to_bf16_bits`: RNE, NaN -> sign | 0x7fc0 (sign from the F32 bits).
fn bf16_rule(bits: u32) -> u16 {
    if bits & 0x7fff_ffff > 0x7f80_0000 {
        return (((bits >> 16) & 0x8000) | 0x7fc0) as u16;
    }
    bf16_rne(bits)
}

/// Element encoding of a streamed upload.
#[derive(Clone, Copy)]
enum Elem {
    /// The draw's F32 bits.
    F32,
    /// `bf16_store` of the draw.
    Bf16,
    /// `bf16_rule` of the draw: the staged BF16 copy of F32 X.
    Bf16Stage,
}

impl Elem {
    fn size(self) -> usize {
        match self {
            Elem::F32 => 4,
            Elem::Bf16 | Elem::Bf16Stage => 2,
        }
    }
}

/// Elements `[start, start + n)` of stream `stream`, little-endian, into `out`.
#[allow(clippy::too_many_arguments)]
fn fill(
    out: &mut Vec<u8>,
    p: Profile,
    seed: u64,
    stream: u64,
    scale: f32,
    elem: Elem,
    start: usize,
    n: usize,
) {
    let key = mix(seed ^ mix(stream.wrapping_mul(0xd134_2543_de82_ef95)));
    out.clear();
    for i in start..start + n {
        let r = mix(key ^ (i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15));
        let bits = f32_bits(p, r, scale);
        match elem {
            Elem::F32 => out.extend_from_slice(&bits.to_le_bytes()),
            Elem::Bf16 => out.extend_from_slice(&bf16_store(bits).to_le_bytes()),
            Elem::Bf16Stage => out.extend_from_slice(&bf16_rule(bits).to_le_bytes()),
        }
    }
}

/// Stream `n` elements to `d` at byte offset `off`; returns the sha256 of the streamed bytes.
#[allow(clippy::too_many_arguments)]
fn upload_stream(
    card: &Card,
    d: &DeviceBuffer,
    off: usize,
    p: Profile,
    seed: u64,
    stream: u64,
    scale: f32,
    elem: Elem,
    n: usize,
    scratch: &mut Vec<u8>,
) -> Result<String, String> {
    let esz = elem.size();
    let mut sha = Sha256::new();
    let mut done = 0;
    while done < n {
        let c = CHUNK.min(n - done);
        fill(scratch, p, seed, stream, scale, elem, done, c);
        sha.update(&scratch[..]);
        card.upload_at(d, off + done * esz, scratch)?;
        done += c;
    }
    Ok(hex(&sha.finalize()))
}

// ───────────────────────────── shapes ─────────────────────────────

struct Shape {
    name: String,
    ms: Vec<usize>,
    k: usize,
    packed: bool,
}

impl Shape {
    fn new(name: &str, ms: &[usize], k: usize, packed: bool) -> Shape {
        assert!(!ms.is_empty() && ms.len() <= 4 && k % 8 == 0 && ms.iter().all(|&m| m > 0));
        Shape {
            name: name.to_string(),
            ms: ms.to_vec(),
            k,
            packed,
        }
    }
    fn weight_bytes(&self) -> u64 {
        (self.ms.iter().sum::<usize>() * self.k * 2) as u64
    }
}

fn shapes() -> Vec<Shape> {
    let mut v = Vec::new();
    // r16bench.cpp Flash-Next decode projections (name, M, K).
    for (name, m, k) in [
        ("gdn_qkv", 10240, 2560),
        ("gdn_z", 6144, 2560),
        ("gdn_a", 48, 2560),
        ("gdn_b", 48, 2560),
        ("out_proj", 2560, 6144),
        ("qsa_q", 12288, 2560),
        ("qsa_k", 512, 2560),
        ("qsa_v", 512, 2560),
        ("qsa_idx", 640, 2560),
        ("hc_down", 320, 10240),
        ("hc_up", 10240, 320),
        ("hc_inject", 4, 10240),
        ("sh_gate", 640, 2560),
        ("sh_up", 640, 2560),
        ("sh_down", 2560, 640),
        ("router", 512, 2560),
        ("sh_egate", 1, 2560),
        ("lm_head", 248_320, 2560),
        ("ple_key", 10240, 2560),
        ("ple_val", 2560, 2560),
    ] {
        v.push(Shape::new(name, &[m], k, false));
    }
    // r16bench.cpp projection groups (shared X and K).
    v.push(Shape::new(
        "grp_gdn_in",
        &[10240, 6144, 48, 48],
        2560,
        false,
    ));
    v.push(Shape::new(
        "grp_qsa_in",
        &[640, 12288, 512, 512],
        2560,
        false,
    ));
    v.push(Shape::new("grp_sh_gu", &[640, 640], 2560, false));
    v.push(Shape::new("grp_moe_in", &[512, 640, 640, 1], 2560, false));
    v.push(Shape::new(
        "packed_gdn_in",
        &[10240, 6144, 48, 48],
        2560,
        true,
    ));
    v.push(Shape::new("packed_sh_gu", &[640, 640], 2560, true));
    // Tail K: K % 128 / K % 256 != 0 and the one-chunk / odd-half-chunk walks.
    for (name, m, k) in [
        ("tailk_8", 48, 8),
        ("tailk_16", 33, 16),
        ("tailk_128", 64, 128),
        ("tailk_136", 31, 136),
        ("tailk_248", 17, 248),
        ("tailk_256", 16, 256),
        ("tailk_264", 129, 264),
        ("tailk_384", 33, 384),
        ("tailk_392", 16, 392),
        ("tailk_520", 47, 520),
        ("tailk_2568", 129, 2568),
        ("tailk_4104", 513, 4104),
        ("tailk_16384", 20, 16384),
    ] {
        v.push(Shape::new(name, &[m], k, false));
    }
    // Tail M: M % 16 != 0 and single partial tiles.
    for (name, m, k) in [
        ("tailm_1", 1, 2560),
        ("tailm_7", 7, 2560),
        ("tailm_15", 15, 256),
        ("tailm_17", 17, 2560),
        ("tailm_31", 31, 640),
        ("tailm_33", 33, 2560),
        ("tailm_4097", 4097, 2056),
    ] {
        v.push(Shape::new(name, &[m], k, false));
    }
    // Groups of 2..=4 members with ragged M / K (3 members; unused trailing members are null with M = 0).
    v.push(Shape::new("tailg_2", &[17, 33], 264, false));
    v.push(Shape::new("tailg_3", &[1, 16, 31], 136, true));
    v.push(Shape::new("tailg_4", &[15, 17, 129, 1], 2568, true));
    v
}

fn profiles(sh: &Shape) -> Vec<Profile> {
    if sh.weight_bytes() > BIG_BYTES {
        vec![Profile::Random, Profile::NanSparse]
    } else {
        vec![
            Profile::Random,
            Profile::Edge,
            Profile::NanSparse,
            Profile::NanDense,
        ]
    }
}

/// d_bf16_mask values per group size: none, all, and the alternating splits.
fn masks(n: usize) -> Vec<i32> {
    let full = (1i32 << n) - 1;
    let mut v = vec![0, full];
    if n >= 2 {
        v.push(0b0101 & full);
        v.push(0b1010 & full);
    }
    v.sort_unstable();
    v.dedup();
    v
}

/// Kernels exercised at (rows, X dtype): `(symbol, is the production route)`.
fn variants(rows: usize, xf32: bool) -> Vec<(Sym, bool)> {
    let mut v = Vec::new();
    if !(xf32 && rows > 8) {
        v.push((Sym::R16, true));
    }
    v.push((Sym::X2, xf32 && rows > 8));
    v
}

// ───────────────────────────── launch ─────────────────────────────

struct Geom<'a> {
    ms: &'a [usize],
    k: usize,
    rows: usize,
    xf32: bool,
    mask: i32,
}

impl Geom<'_> {
    fn esz(&self, j: usize) -> usize {
        if (self.mask >> j) & 1 == 1 {
            2
        } else {
            4
        }
    }
    fn valid_bytes(&self, j: usize) -> usize {
        self.rows * self.ms[j] * self.esz(j)
    }
}

/// The production kernarg image of `Gpu::launch_gemm_bf16_mfma_rows16`: X, A0..A3, D0..D3, M0..M3, K, batch, x_f32,
/// d_bf16_mask (104 bytes; hidden arguments are HIP's).
fn launch(
    card: &Card,
    f: &Function,
    g: &Geom,
    x: u64,
    a: &[u64; 4],
    d: &[u64; 4],
) -> Result<(), String> {
    let mut m = [0i32; 4];
    for (j, &v) in g.ms.iter().enumerate() {
        m[j] = v as i32;
    }
    let tiles: usize = g.ms.iter().map(|&v| v.div_ceil(16)).sum();
    let mut blob = KernargBlob::new();
    blob.push_ptr(x as usize as *const c_void);
    blob.push_ptr(a[0] as usize as *const c_void);
    blob.push_ptr(a[1] as usize as *const c_void);
    blob.push_ptr(a[2] as usize as *const c_void);
    blob.push_ptr(a[3] as usize as *const c_void);
    blob.push_ptr(d[0] as usize as *const c_void);
    blob.push_ptr(d[1] as usize as *const c_void);
    blob.push_ptr(d[2] as usize as *const c_void);
    blob.push_ptr(d[3] as usize as *const c_void);
    blob.push_i32(m[0]);
    blob.push_i32(m[1]);
    blob.push_i32(m[2]);
    blob.push_i32(m[3]);
    blob.push_i32(g.k as i32);
    blob.push_i32(g.rows as i32);
    blob.push_i32(i32::from(g.xf32));
    blob.push_i32(g.mask);
    let lds = (2 * (16 + g.rows) * R16_LD * 2) as u32;
    // SAFETY: `blob` is the kernel's full explicit kernarg image; every pointer in it is a live allocation of this
    // harness (or null for an unused member, never dereferenced) sized for this launch.
    unsafe {
        card.rt.launch_kernel_blob(
            f,
            [tiles as u32, 1, 1],
            [BLOCK, 1, 1],
            lds,
            None,
            blob.as_mut_slice(),
        )
    }
    .map_err(|e| format!("launch: {e}"))
}

/// `(first differing byte, differing byte count)`, or `None` when equal.
fn first_diff(a: &[u8], b: &[u8]) -> Option<(usize, usize)> {
    if a == b {
        return None;
    }
    let mut first = None;
    let mut count = 0;
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        if x != y {
            first.get_or_insert(i);
            count += 1;
        }
    }
    first.map(|f| (f, count))
}

// ───────────────────────────── cases ─────────────────────────────

struct Report {
    cases: Vec<Value>,
    failures: Vec<Value>,
    n_cases: u64,
    n_failed: u64,
    n_pm_ne_hipcc: u64,
    n_hipcc_ne_tiled: u64,
    n_pm_ne_tiled: u64,
    by_kernel: std::collections::BTreeMap<String, u64>,
    all_inputs: Sha256,
}

struct Env<'a> {
    card: &'a Card,
    refi: &'a Image,
    pmi: &'a Image,
    seed: u64,
}

/// The shipped tiled kernel's kernarg image: A, B (BF16 [batch, K]), D (F32 [batch, M]), M, K, batch.
#[allow(clippy::too_many_arguments)]
fn launch_tiled(
    card: &Card,
    f: &Function,
    a: u64,
    b: u64,
    d: u64,
    m: usize,
    k: usize,
    rows: usize,
) -> Result<(), String> {
    let mut blob = KernargBlob::new();
    blob.push_ptr(a as usize as *const c_void);
    blob.push_ptr(b as usize as *const c_void);
    blob.push_ptr(d as usize as *const c_void);
    blob.push_i32(m as i32);
    blob.push_i32(k as i32);
    blob.push_i32(rows as i32);
    // SAFETY: `blob` is the kernel's full explicit kernarg image (A, B, D, M, K, batch_size); the pointers are live
    // allocations of this harness sized for this launch.
    unsafe {
        card.rt.launch_kernel_blob(
            f,
            [m.div_ceil(32) as u32, rows.div_ceil(32) as u32, 1],
            [256, 1, 1],
            0,
            None,
            blob.as_mut_slice(),
        )
    }
    .map_err(|e| format!("launch tiled: {e}"))
}

struct CaseOut {
    ref_sha: String,
    pm_sha: String,
    tiled_sha: String,
    pm_eq_hipcc: bool,
    hipcc_eq_tiled: bool,
    pm_eq_tiled: bool,
    failures: Vec<Value>,
}

/// One launch of `sym` per arm and poison into `dbufs[arm][poison][member]`; byte comparisons per member, between the
/// arms and against `tiled[member]` (the tiled kernel's F32 output; rounded here for BF16 members).
/// `host[0..4]` are readback scratch (`arm * 2 + poison`), `host[4]` the expected-bytes scratch.
#[allow(clippy::too_many_arguments)]
fn run_case(
    env: &Env,
    g: &Geom,
    sym: Sym,
    x_ptr: u64,
    a_ptr: &[u64; 4],
    dbufs: &[Vec<Vec<DeviceBuffer>>],
    tiled: &[Vec<u8>],
    host: &mut [Vec<u8>; 5],
) -> Result<CaseOut, String> {
    let card = env.card;
    let n = g.ms.len();
    for arm in dbufs {
        for (p, set) in arm.iter().enumerate() {
            for j in 0..n {
                card.memset(&set[j], POISON[p], g.valid_bytes(j) + GUARD)?;
            }
        }
    }
    for (arm, img) in [env.refi, env.pmi].into_iter().enumerate() {
        for p in 0..2 {
            let mut d = [0u64; 4];
            for j in 0..n {
                d[j] = ptr(&dbufs[arm][p][j]);
            }
            let name = match sym {
                Sym::R16 => SYM_R16,
                Sym::X2 => SYM_X2,
            };
            let f = card
                .rt
                .module_get_function(&img.module, name)
                .map_err(|e| format!("hipModuleGetFunction({name}): {e}"))?;
            launch(card, &f, g, x_ptr, a_ptr, &d)?;
        }
    }
    card.sync()?;
    let mut ref_sha = Sha256::new();
    let mut pm_sha = Sha256::new();
    let mut tiled_sha = Sha256::new();
    let mut pm_eq_hipcc = true;
    let mut hipcc_eq_tiled = true;
    let mut pm_eq_tiled = true;
    let mut failures = Vec::new();
    for j in 0..n {
        let valid = g.valid_bytes(j);
        let used = valid + GUARD;
        // host[arm * 2 + poison]
        for arm in 0..2 {
            for p in 0..2 {
                let h = &mut host[arm * 2 + p];
                h.resize(used, 0);
                card.read(&dbufs[arm][p][j], &mut h[..used])?;
            }
        }
        let esz = g.esz(j);
        let m = g.ms[j];
        let exp: &[u8] = if esz == 4 {
            &tiled[j]
        } else {
            let e = &mut host[4];
            e.clear();
            for w in tiled[j].chunks_exact(4) {
                let bits = u32::from_le_bytes([w[0], w[1], w[2], w[3]]);
                e.extend_from_slice(&bf16_rule(bits).to_le_bytes());
            }
            &host[4]
        };
        if exp.len() != valid {
            return Err(format!(
                "expected-output length {} != valid output length {valid} (member {j})",
                exp.len()
            ));
        }
        let mut report = |kind: &str, at: (usize, usize), a: &[u8], b: &[u8]| {
            let e = at.0 / esz;
            let lo = (at.0 / 4) * 4;
            let word = |s: &[u8]| hex(&s[lo..(lo + 4).min(s.len())]);
            failures.push(json!({
                "member": j, "kind": kind, "byte_offset": at.0, "differing_bytes": at.1,
                "row": e / m, "m": e % m, "d_bytes_per_element": esz,
                "word_a": word(a), "word_b": word(b),
            }));
        };
        for p in 0..2 {
            let (r, q) = (&host[p], &host[2 + p]);
            if let Some(at) = first_diff(&r[..valid], &q[..valid]) {
                pm_eq_hipcc = false;
                report(
                    &format!("pm_ne_hipcc_poison_{:02x}", POISON[p]),
                    at,
                    &r[..valid],
                    &q[..valid],
                );
            }
        }
        // word_a = the arm's output, word_b = the tiled-derived expectation.
        for arm in 0..2 {
            let out = &host[arm * 2][..valid];
            let d = first_diff(out, &exp[..valid]);
            if let Some(at) = d {
                report(
                    if arm == 0 {
                        "hipcc_ne_tiled"
                    } else {
                        "pm_ne_tiled"
                    },
                    at,
                    out,
                    &exp[..valid],
                );
            }
            if arm == 0 {
                hipcc_eq_tiled &= d.is_none();
            } else {
                pm_eq_tiled &= d.is_none();
            }
        }
        for arm in 0..2 {
            let (a, b) = (&host[arm * 2], &host[arm * 2 + 1]);
            if let Some(at) = first_diff(&a[..valid], &b[..valid]) {
                report(
                    if arm == 0 {
                        "hipcc_output_depends_on_poison_or_is_nondeterministic"
                    } else {
                        "pm_output_depends_on_poison_or_is_nondeterministic"
                    },
                    at,
                    &a[..valid],
                    &b[..valid],
                );
            }
            for p in 0..2 {
                let h = &host[arm * 2 + p];
                if let Some(i) = h[valid..used].iter().position(|&x| x != POISON[p]) {
                    let at = (
                        valid + i,
                        h[valid..used].iter().filter(|&&x| x != POISON[p]).count(),
                    );
                    report(
                        if arm == 0 {
                            "hipcc_guard_written"
                        } else {
                            "pm_guard_written"
                        },
                        at,
                        &h[..used],
                        &h[..used],
                    );
                }
            }
        }
        ref_sha.update(&host[0][..valid]);
        pm_sha.update(&host[2][..valid]);
        tiled_sha.update(exp);
    }
    Ok(CaseOut {
        ref_sha: hex(&ref_sha.finalize()),
        pm_sha: hex(&pm_sha.finalize()),
        tiled_sha: hex(&tiled_sha.finalize()),
        pm_eq_hipcc,
        hipcc_eq_tiled,
        pm_eq_tiled,
        failures,
    })
}

fn run_shape(
    env: &Env,
    sh: &Shape,
    prof: Profile,
    scratch: &mut Vec<u8>,
    host: &mut [Vec<u8>; 5],
    rep: &mut Report,
) -> Result<(), String> {
    let card = env.card;
    let n = sh.ms.len();
    let k = sh.k;
    let mmax = *sh.ms.iter().max().expect("non-empty");
    if !env.refi.has_tiled {
        return Err(format!("image has no `{SYM_TILED}` (the tiled comparator)"));
    }
    let tiled_fn = &card
        .rt
        .module_get_function(&env.refi.module, SYM_TILED)
        .map_err(|e| format!("hipModuleGetFunction({SYM_TILED}): {e}"))?;
    host_ram_check((CHUNK * 4 + 10 * (16 * mmax * 4 + GUARD)) as u64 + (256 << 20))?;

    // Weights: one buffer per member, or one slab at 16-byte (non-256-byte) offsets.
    let mut a_bufs: Vec<DeviceBuffer> = Vec::new();
    let mut a_ptr = [0u64; 4];
    let mut a_sha: Vec<String> = Vec::new();
    let slab_base = 16usize;
    if sh.packed {
        let mut offs = Vec::new();
        let mut off = 0usize;
        for &m in &sh.ms {
            offs.push(off);
            off += m * k;
            off = off.div_ceil(8) * 8;
        }
        let buf = card.alloc(slab_base + off * 2)?;
        card.memset(&buf, 0, buf.size())?;
        for (j, &m) in sh.ms.iter().enumerate() {
            let byte = slab_base + offs[j] * 2;
            a_ptr[j] = ptr(&buf) + byte as u64;
            a_sha.push(upload_stream(
                card,
                &buf,
                byte,
                prof,
                env.seed,
                STREAM_A + j as u64,
                0.04,
                Elem::Bf16,
                m * k,
                scratch,
            )?);
        }
        a_bufs.push(buf);
    } else {
        for (j, &m) in sh.ms.iter().enumerate() {
            let buf = card.alloc(m * k * 2)?;
            a_ptr[j] = ptr(&buf);
            a_sha.push(upload_stream(
                card,
                &buf,
                0,
                prof,
                env.seed,
                STREAM_A + j as u64,
                0.04,
                Elem::Bf16,
                m * k,
                scratch,
            )?);
            a_bufs.push(buf);
        }
    }
    // Activations: up to 16 rows of F32 (xbuf) or BF16 (xbuf); xbf = the staged BF16 copy of F32 X for the tiled kernel.
    let x_off = if sh.packed { 16 } else { 0 };
    let xbuf = card.alloc(x_off + 16 * k * 4)?;
    card.memset(&xbuf, 0, xbuf.size())?;
    let x_ptr = ptr(&xbuf) + x_off as u64;
    let xbf = card.alloc(16 * k * 2)?;
    card.memset(&xbf, 0, xbf.size())?;
    // dbufs[arm][poison][member]: 16 rows of F32 + guard.
    let mut dbufs: Vec<Vec<Vec<DeviceBuffer>>> = Vec::new();
    for _ in 0..2 {
        let mut arm = Vec::new();
        for _ in 0..2 {
            let mut set = Vec::new();
            for &m in &sh.ms {
                set.push(card.alloc(16 * m * 4 + GUARD)?);
            }
            arm.push(set);
        }
        dbufs.push(arm);
    }
    // tbufs[poison][member]: the tiled kernel's F32 [rows, M] outputs.
    let mut tbufs: Vec<Vec<DeviceBuffer>> = Vec::new();
    for _ in 0..2 {
        let mut set = Vec::new();
        for &m in &sh.ms {
            set.push(card.alloc(16 * m * 4)?);
        }
        tbufs.push(set);
    }
    let mut thost: Vec<Vec<u8>> = vec![Vec::new(); n];
    let mut tchk: Vec<u8> = Vec::new();

    for &rows in &ROWS {
        for xf32 in [true, false] {
            let x_sha = upload_stream(
                card,
                &xbuf,
                x_off,
                prof,
                env.seed,
                STREAM_X,
                1.0,
                if xf32 { Elem::F32 } else { Elem::Bf16 },
                rows * k,
                scratch,
            )?;
            let stage_sha = if xf32 {
                upload_stream(
                    card,
                    &xbf,
                    0,
                    prof,
                    env.seed,
                    STREAM_X,
                    1.0,
                    Elem::Bf16Stage,
                    rows * k,
                    scratch,
                )?
            } else {
                String::new()
            };
            // The tiled kernel's F32 outputs of this (rows, X dtype): independent of the mask and the rows16 entry.
            let b_ptr = if xf32 { ptr(&xbf) } else { x_ptr };
            for j in 0..n {
                let bytes = rows * sh.ms[j] * 4;
                for p in 0..2 {
                    card.memset(&tbufs[p][j], POISON[p], bytes)?;
                    launch_tiled(
                        card,
                        tiled_fn,
                        a_ptr[j],
                        b_ptr,
                        ptr(&tbufs[p][j]),
                        sh.ms[j],
                        k,
                        rows,
                    )?;
                }
            }
            card.sync()?;
            for j in 0..n {
                let bytes = rows * sh.ms[j] * 4;
                thost[j].resize(bytes, 0);
                card.read(&tbufs[0][j], &mut thost[j][..])?;
                tchk.resize(bytes, 0);
                card.read(&tbufs[1][j], &mut tchk[..])?;
                if thost[j] != tchk {
                    return Err(format!(
                        "tiled comparator is nondeterministic or leaves outputs unwritten: {} {} rows={rows} x={} member {j}",
                        sh.name,
                        prof.name(),
                        if xf32 { "f32" } else { "bf16" }
                    ));
                }
            }
            for mask in masks(n) {
                let g = Geom {
                    ms: &sh.ms,
                    k,
                    rows,
                    xf32,
                    mask,
                };
                let mut h = Sha256::new();
                h.update(
                    format!(
                        "rows16-oracle-input-v2|profile={}|seed={}|k={}|ms={:?}|packed={}|rows={}|x={}|d_bf16_mask={}",
                        prof.name(),
                        env.seed,
                        k,
                        sh.ms,
                        sh.packed,
                        rows,
                        if xf32 { "f32" } else { "bf16" },
                        mask
                    )
                    .as_bytes(),
                );
                for s in &a_sha {
                    h.update(s.as_bytes());
                }
                h.update(x_sha.as_bytes());
                h.update(stage_sha.as_bytes());
                let input_sha = hex(&h.finalize());
                for (sym, route) in variants(rows, xf32) {
                    let out = run_case(env, &g, sym, x_ptr, &a_ptr, &dbufs, &thost, host)?;
                    rep.n_cases += 1;
                    rep.all_inputs.update(input_sha.as_bytes());
                    *rep.by_kernel.entry(sym.name().to_string()).or_insert(0) += 1;
                    rep.n_pm_ne_hipcc += u64::from(!out.pm_eq_hipcc);
                    rep.n_hipcc_ne_tiled += u64::from(!out.hipcc_eq_tiled);
                    rep.n_pm_ne_tiled += u64::from(!out.pm_eq_tiled);
                    let ok = out.failures.is_empty();
                    let id = rep.n_cases;
                    let desc = format!(
                        "{} {} {} rows={} x={} d_mask={:#x}",
                        sh.name,
                        prof.name(),
                        sym.name(),
                        rows,
                        if xf32 { "f32" } else { "bf16" },
                        mask
                    );
                    if !ok {
                        rep.n_failed += 1;
                        eprintln!("FAIL case {id}: {desc}: {}", out.failures[0]);
                        if rep.failures.len() < MAX_FAILURES_KEPT {
                            rep.failures.push(json!({
                                "case": id, "desc": desc, "input_sha256": input_sha,
                                "first": out.failures.iter().take(8).collect::<Vec<_>>(),
                                "findings": out.failures.len(),
                            }));
                        }
                    }
                    rep.cases.push(json!({
                        "id": id, "shape": sh.name, "profile": prof.name(), "kernel": sym.name(),
                        "production_route": route, "rows": rows,
                        "x": if xf32 { "f32" } else { "bf16" }, "d_bf16_mask": mask,
                        "ms": sh.ms, "k": k, "packed": sh.packed,
                        "input_sha256": input_sha, "hipcc_output_sha256": out.ref_sha,
                        "pm_output_sha256": out.pm_sha, "tiled_expected_output_sha256": out.tiled_sha,
                        "pm_equals_hipcc": out.pm_eq_hipcc, "hipcc_equals_tiled": out.hipcc_eq_tiled,
                        "pm_equals_tiled": out.pm_eq_tiled, "equal": ok,
                    }));
                }
            }
        }
    }

    for b in a_bufs {
        card.free(b)?;
    }
    card.free(xbuf)?;
    card.free(xbf)?;
    for arm in dbufs {
        for set in arm {
            for b in set {
                card.free(b)?;
            }
        }
    }
    for set in tbufs {
        for b in set {
            card.free(b)?;
        }
    }
    Ok(())
}

/// The harness must be able to see a difference: pm on a perturbed X must differ from hipcc on X, pm on X must equal
/// hipcc on X, and one flipped host byte must be reported as exactly one differing byte.
fn controls(env: &Env, scratch: &mut Vec<u8>) -> Result<Value, String> {
    let card = env.card;
    let (m, k, rows) = (48usize, 256usize, 2usize);
    let ms = [m];
    let g = Geom {
        ms: &ms,
        k,
        rows,
        xf32: true,
        mask: 0,
    };
    let abuf = card.alloc(m * k * 2)?;
    upload_stream(
        card,
        &abuf,
        0,
        Profile::Random,
        env.seed,
        STREAM_A,
        0.04,
        Elem::Bf16,
        m * k,
        scratch,
    )?;
    let x1 = card.alloc(rows * k * 4)?;
    upload_stream(
        card,
        &x1,
        0,
        Profile::Random,
        env.seed,
        STREAM_X,
        1.0,
        Elem::F32,
        rows * k,
        scratch,
    )?;
    // X2 = X1 with the exponent MSB of element 0 flipped (the weights are random and non-zero).
    let mut bytes = vec![0u8; rows * k * 4];
    card.read(&x1, &mut bytes)?;
    bytes[3] ^= 0x40;
    let x2 = card.alloc(rows * k * 4)?;
    card.upload_at(&x2, 0, &bytes)?;
    let out = rows * m * 4;
    let d_ref = card.alloc(out)?;
    let d_pm = card.alloc(out)?;
    let d_pm2 = card.alloc(out)?;
    let a = [ptr(&abuf), 0, 0, 0];
    let run = |img: &Image, x: &DeviceBuffer, d: &DeviceBuffer| -> Result<Vec<u8>, String> {
        card.memset(d, 0xA5, out)?;
        let f = card
            .rt
            .module_get_function(&img.module, SYM_R16)
            .map_err(|e| format!("hipModuleGetFunction({SYM_R16}): {e}"))?;
        launch(card, &f, &g, ptr(x), &a, &[ptr(d), 0, 0, 0])?;
        card.sync()?;
        let mut v = vec![0u8; out];
        card.read(d, &mut v)?;
        Ok(v)
    };
    let r = run(env.refi, &x1, &d_ref)?;
    let p = run(env.pmi, &x1, &d_pm)?;
    let p2 = run(env.pmi, &x2, &d_pm2)?;
    let same = first_diff(&r, &p).is_none();
    let sees_perturbation = first_diff(&r, &p2).is_some();
    let mut flipped = r.clone();
    let last = flipped.len() - 1;
    flipped[last] ^= 1;
    let sees_flip = first_diff(&r, &flipped) == Some((last, 1));
    for b in [abuf, x1, x2, d_ref, d_pm, d_pm2] {
        card.free(b)?;
    }
    if !same {
        return Err(
            "control: pm differs from hipcc on the control input (see the matrix for details)"
                .into(),
        );
    }
    if !sees_perturbation || !sees_flip {
        return Err(format!(
            "harness is blind: perturbed-input difference seen {sees_perturbation}, single-byte flip seen {sees_flip}"
        ));
    }
    Ok(json!({
        "pm_equals_hipcc_on_control_input": same,
        "pm_on_perturbed_input_differs_from_hipcc": sees_perturbation,
        "single_flipped_host_byte_detected": sees_flip,
        "control_shape": {"m": m, "k": k, "rows": rows, "x": "f32", "d": "f32"},
    }))
}

// ───────────────────────────── main ─────────────────────────────

fn main() -> ExitCode {
    match run() {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::from(1),
        Err(e) => {
            eprintln!("gemm_bf16_rows16_oracle: {e}");
            ExitCode::from(2)
        }
    }
}

fn run() -> Result<bool, String> {
    let a: Vec<String> = std::env::args().collect();
    let opt = |name: &str| -> Option<String> {
        a.iter()
            .position(|x| x == name)
            .and_then(|i| a.get(i + 1))
            .cloned()
    };
    let flag = |name: &str| a.iter().any(|x| x == name);
    let (Some(ref_path), Some(pm_path)) = (opt("--ref"), opt("--pm")) else {
        return Err("usage: gemm_bf16_rows16_oracle --ref <hipcc.co> --pm <twin.hxaco> [--out FILE.json] [--seed S] [--device N] [--only SUBSTR] [--skip-big] [--expect-pm-sha256 HEX] [--expect-ref-sha256 HEX]".into());
    };
    let seed: u64 = opt("--seed").map_or(Ok(0x7203_1604_f1a5_4e11), |v| {
        v.parse().map_err(|_| "bad --seed".to_string())
    })?;
    let device: i32 =
        opt("--device").map_or(Ok(0), |v| v.parse().map_err(|_| "bad --device".to_string()))?;
    let only = opt("--only");
    let skip_big = flag("--skip-big");
    let t0 = Instant::now();
    let (mem_total, mem_avail) = meminfo()?;
    host_ram_check(512 << 20)?;

    let rt = HipRuntime::load().map_err(|e| format!("hip: {e}"))?;
    rt.set_device(device)
        .map_err(|e| format!("hipSetDevice({device}): {e}"))?;
    let arch = rt
        .get_arch(device)
        .map_err(|e| format!("device arch: {e}"))?;
    if !arch.starts_with("gfx942") {
        return Err(format!("device {device} is {arch}, not gfx942"));
    }
    let pci = rt.device_pci_bus_id(device).unwrap_or_default();
    let card = Card { rt };
    let refi = card.load(&ref_path, "hipcc", opt("--expect-ref-sha256").as_deref())?;
    let pmi = card.load(&pm_path, "pm", opt("--expect-pm-sha256").as_deref())?;
    let env = Env {
        card: &card,
        refi: &refi,
        pmi: &pmi,
        seed,
    };

    let mut scratch: Vec<u8> = Vec::with_capacity(CHUNK * 4);
    let ctl = controls(&env, &mut scratch)?;
    eprintln!("controls: {ctl}");

    let mut host: [Vec<u8>; 5] = Default::default();
    let mut rep = Report {
        cases: Vec::new(),
        failures: Vec::new(),
        n_cases: 0,
        n_failed: 0,
        n_pm_ne_hipcc: 0,
        n_hipcc_ne_tiled: 0,
        n_pm_ne_tiled: 0,
        by_kernel: Default::default(),
        all_inputs: Sha256::new(),
    };
    let mut ran: Vec<String> = Vec::new();
    for sh in shapes() {
        if only.as_deref().is_some_and(|o| !sh.name.contains(o)) {
            continue;
        }
        if skip_big && sh.weight_bytes() > BIG_BYTES {
            continue;
        }
        let t = Instant::now();
        let before = rep.n_cases;
        for prof in profiles(&sh) {
            run_shape(&env, &sh, prof, &mut scratch, &mut host, &mut rep)?;
        }
        eprintln!(
            "{:<14} {:>6} cases, {} failed so far, {:.1}s",
            sh.name,
            rep.n_cases - before,
            rep.n_failed,
            t.elapsed().as_secs_f64()
        );
        ran.push(sh.name.clone());
    }
    if ran.is_empty() {
        return Err("no shape selected".into());
    }
    let ok = rep.n_failed == 0 && rep.n_cases > 0;
    let full = only.is_none() && !skip_big;
    let summary = json!({
        "tool": "peacemaker-isa example gemm_bf16_rows16_oracle",
        "status": if ok { "PASS" } else { "FAIL" },
        "target": "gfx942",
        "method": "hipModuleLoadData of the hipcc device ELF and of the gfx942 device ELF of the twin hxaco; identical device inputs; hipModuleLaunchKernel via hip-bridge with the production kernarg image; raw byte compare of every output, two poison patterns per arm; every output also compared with the shipped tiled kernel gemm_bf16_mfma_gfx942 of the hipcc image (F32 X staged with convert_f32_to_bf16's rule, BF16 members rounded with the same rule)",
        "full_matrix": full,
        "input_hxaco_sha256": pmi.file_sha,
        "pm_device_elf_sha256": pmi.elf_sha,
        "pm_device_elf_bytes": pmi.elf_bytes,
        "hipcc_image_sha256": refi.file_sha,
        "hipcc_device_elf_sha256": refi.elf_sha,
        "hipcc_device_elf_bytes": refi.elf_bytes,
        "device": {"index": device, "arch": arch, "pci_bus_id": pci},
        "host_ram": {"mem_total_bytes": mem_total, "mem_available_at_start_bytes": mem_avail, "min_free_fraction": 0.25},
        "seed": seed,
        "case_count": rep.n_cases,
        "failed_cases": rep.n_failed,
        "cases_by_kernel": rep.by_kernel,
        "rows": ROWS,
        "x_types": ["f32", "bf16"],
        "shapes": ran,
        "all_case_inputs_sha256": hex(&rep.all_inputs.finalize_reset()),
        "controls": ctl,
        "tiled_comparison": {
            "kernel": SYM_TILED,
            "image_device_elf_sha256": refi.elf_sha,
            "f32_x_staging_rule": "RNE; NaN -> sign | 0x7fc0 (sign from the F32 bits), convert_f32_to_bf16.hip",
            "bf16_output_rule": "same rule applied to the tiled F32 output words",
            "cases_hipcc_ne_tiled": rep.n_hipcc_ne_tiled,
            "cases_pm_ne_tiled": rep.n_pm_ne_tiled,
        },
        "results": {
            "raw_outputs_pm_equals_hipcc": rep.n_pm_ne_hipcc == 0 && rep.n_cases > 0,
            "raw_outputs_hipcc_equals_tiled": rep.n_hipcc_ne_tiled == 0 && rep.n_cases > 0,
            "raw_outputs_pm_equals_tiled": rep.n_pm_ne_tiled == 0 && rep.n_cases > 0,
            "all_checks_pass": ok,
        },
        "seconds": t0.elapsed().as_secs_f64(),
        "failures": rep.failures,
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&summary).map_err(|e| e.to_string())?
    );
    if let Some(out) = opt("--out") {
        let mut full_doc = summary;
        full_doc["cases"] = Value::Array(std::mem::take(&mut rep.cases));
        let text = serde_json::to_string_pretty(&full_doc).map_err(|e| e.to_string())?;
        std::fs::write(&out, text + "\n").map_err(|e| format!("{out}: {e}"))?;
    }
    Ok(ok)
}
