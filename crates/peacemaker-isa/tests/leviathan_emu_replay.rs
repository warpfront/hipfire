// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Differential execution of the UNCHANGED committed Leviathan objects in the PeaceMaker CPU emulator against GPU
//! results (item 21): `kernels/lane_leviathan_hipcc_gfx120{0,1}.co` and the PM twin
//! `kernels/lane_leviathan_pm_gfx1201.hxaco`, kernel `lane_leviathan_verify` in modes 0 (full row), 1 (shard select)
//! and 2 (merged), including its out-of-line helper `llv_tail` (`s_swappc_b64` call / `s_setpc_b64` return, 272-byte
//! per-lane scratch frame, one returning `flat_atomic_add_u32`).
//!
//! The GPU side is `leviathan_emu_capture` in `crates/hipfire-runtime/tests/x1_lane_draw_shard_gpu.rs` (fabric, exact
//! gfx1201; its doc comment is the capture format). It runs the X1 twin matrix through the committed hipcc object and
//! records, for every launch, the seven kernarg values and every device buffer the launch can touch (region, p table,
//! candidate tables, each lane's logits) at its real device address, byte-exact before and after. The replay maps each
//! buffer at its captured address with its BEFORE bytes, builds the kernarg segment from the lifted kernel's own HSA
//! metadata, runs workgroups `x = 0..rows` of 512 threads, and requires every buffer to equal its AFTER bytes
//! byte-for-byte. Each launch also runs in reverse workgroup order on fresh memory and must equal the forward run
//! (the verdict is taken by whichever block of a lane takes the last ticket, so the schedule changes who computes it).
//!
//! ```text
//! HIPFIRE_LEV_EMU_CAPTURE=<dir written by leviathan_emu_capture> \
//!   cargo test -p peacemaker-isa --release --test leviathan_emu_replay -- --ignored --nocapture
//! ```
//!
//! # Arms (each replays the same captures; none is a claim beyond what it names)
//! - `hipcc-gfx1201`: `lane_leviathan_hipcc_gfx1201.co` with gfx1201 numerics, against the gfx1201 GPU results of the
//!   same object. The only arm that is hardware-measured end to end.
//! - `hipcc-gfx1200-as-gfx1201`: `lane_leviathan_hipcc_gfx1200.co` under `Emulator::gfx1200_as_gfx1201`: shared gfx12
//!   ISA reasoning, NOT a gfx1200 hardware measurement (the fabric has no gfx1200 device). A pass says the gfx1200
//!   object's instruction stream computes the gfx1201 object's results under gfx1201 numerics; it does not say what a
//!   gfx1200 GPU does.
//! - `pm-twin-gfx1201`: the committed `.hxaco` bundle. Its device ELF is byte-identical to the hipcc object's
//!   (asserted below), so this arm is the same instruction stream lifted through the PM bundle path; it checks that the
//!   committed PM artifact itself executes, not an independent re-implementation.
//!
//! # Negative controls (run before the matrix; a control that does not trip fails the test)
//! 1. `Emulator::new` (kernel only, no helpers) must fail at the first call (`s_swappc_b64`) with the unresolved
//!    call-target error, not run past it.
//! 2. One mutation: the logit of lane 0's first GPU-emitted token is replaced by -inf before replaying a mode 0
//!    launch; the emulator's out record / p table must then DIFFER from the GPU's after-bytes (an input the kernel
//!    reads must matter, otherwise the equality above could be vacuous).
//! 3. The gfx1200 object without `gfx1200_as_gfx1201` must refuse to run, naming the opt-in.
//!
//! # What the captured X1 matrix covers
//! The matrix is `TWIN_MATRIX` in `x1_lane_draw_shard_gpu.rs`: V = 248,320 (12 launches, seed 0x510a), 5,000 (24,
//! 0x510b) and 4,098 (24, 0x510c); 6 lanes per launch; lane generator `make_lane` plus `plant_edges` (one edge lane per
//! launch). Each launch is captured in modes 0, 1 (shard A, shard B) and 2. All paths below are in
//! `crates/hipfire-runtime/tests/x1_lane_draw_shard_gpu.rs` unless stated.
//! - Temperatures including greedy: `TEMPS = [0.0, 0.7, 1.0]`, `temperature: TEMPS[lane_global % 3]`; temperature 0.0
//!   is the greedy (point) path. Top-k / top-p / min-p: `TOPK = [None, Some(0), Some(1), Some(20), Some(21), Some(64)]`
//!   at `(lane_global / 3) % 6`, `TOPP = [0.95, 1.0]` at `(lane_global / 18) % 2`, `MINP = [None, Some(0.05)]` at
//!   `(lane_global / 36) % 2`. Those four indices are injective over `lane_global` in 0..72, so the 72-lane Flash
//!   sweep visits each of the 72 combinations once and each 144-lane small-vocab sweep twice. `None` is packed as
//!   top-k 20 (`pack` in `x1_lane_draw_twin/mod.rs`) and the kernel widens its candidate list to 64 only for top-k > 20
//!   (`G` in `kernels/src/lane_leviathan.hip`), so k = 21 and 64 exercise the wide list. Top-p 1.0 and min-p `None`
//!   are the unfiltered settings.
//! - Penalties: `REP = [1.0, 1.3, 2.0]` at `(lane_global / 2) % 3`, `PRES = [0.0, 0.5, -0.5]` at `/ 5`,
//!   `FREQ = [0.0, 0.3, -0.3]` at `/ 7`, `WINDOW = [8, 64, 200]` at `/ 11`; history lengths 0 / 3 / 20 / 80 whose tokens
//!   are 70% hot ids (`pick_tok`), drafts 80% hot. Both signs of presence and frequency are generated, but the kernel
//!   applies `v -= freq * count + pres` only when `pres > 0 || freq > 0` and the penalty path is active
//!   (`(rep > 1 || pres > 0 || freq > 0) && pen_len > 0`), so a negative term is effective only in a mixed-sign lane
//!   (e.g. `pres = 0.5`, `freq = -0.3`); a lane whose only non-zero terms are negative with `rep == 1.0` has no
//!   penalty path at all. The repeat penalty has no sign here: only `{1.0, 1.3, 2.0}` are generated, no `rep < 1`.
//!   (Offline replay of the generator, not asserted by this test: mixed-sign lanes with the penalty active are 13 of
//!   72 / 33 of 144 / 35 of 144 for V = 248,320 / 5,000 / 4,098.)
//! - Non-finite values, signed zero, NaN payloads (all as INPUTS; the kernel never writes a logit): `make_lane` puts
//!   up to three canonical NaN (`f32::NAN`, 0x7fc00000) and up to three -inf at random non-hot positions in 25% of rows;
//!   `plant_edges` (lane `launch % 6` of every launch, every row) adds +inf (even rows, both halves), -0.0 at `V/2 - 2`,
//!   +0.0 at `V/2 + 1`, random +0.0 / -0.0, the NaN payload words `0x7fc00000, 0xffc12345 (negative), 0x7f800001
//!   (signalling), 0x7fffffff` alternating between halves, and one -inf. `plant_edges` does not protect the hot ids, so a
//!   planted value can overwrite one by chance. The replay censuses the captured mode 0 logits and requires every
//!   payload above to be present. NOT produced: subnormal logits (every non-special logit is a normal `unit()`-derived
//!   value).
//! - Boundary and mass ties: `make_lane` sets `row[V/2 - 1] = row[V/2] = 9.0` in every row (an exact tie across the shard
//!   boundary, A's last and B's first element) and, in 40% of rows (`all_tied`), the six other hot ids to 9.0 as well
//!   (an 8-way exact tie at the maximum, which straddles the cut for top-k 1); 50% of rows are BF16-rounded (`bf16`),
//!   producing dense exact ties. The mass-tie lane (`mass_tie`: lane 0 of every launch with `launch % 3 == 0`)
//!   sets the first `min(2200, V/2)` values of EACH half to 7.0 over a floor of `-2 - |x|`: 2,200 equal values per shard
//!   for V = 248,320 and 5,000, 2,049 (the whole shard) for V = 4,098.
//! - LDS-overflow tile path: the kernel enters it when more than `LLV_CAND_CAP = 2048` candidates reach `tau`
//!   (`nc > LLV_CAND_CAP`, "Tile reduction"). The mass-tie rows are BUILT to overflow it in whole-row mode 0 (4,400 tied
//!   values for V = 248,320 and 5,000; 4,098 for V = 4,098) and, for V = 248,320 and 5,000, in every shard row of
//!   mode 1 (2,200 tied values per shard, minus at most a few dozen positions the hot ids and planted special values
//!   overwrite: far above 2,048). V = 4,098 is the borderline case: a shard holds at most 2,049 tied values, one over the
//!   cap, so whether a given shard row enters the tile path depends on which positions its hot ids and special values
//!   overwrote (an all-tied row with none overwritten has 2,049 candidates, a row with one hot id below 7.0 in the zone
//!   has 2,048); mode 1 at V = 4,098 is not a guaranteed tile-path case. The emulator reports instruction counts only:
//!   this replay does not observe which kernel branch ran, so it claims inputs built to overflow LDS (arithmetic above),
//!   not branch coverage.
//! - Also generated: blocked tokens (hot ids, up to 3 per lane), scope tokens in both halves, EOS, drafts with and
//!   without q records, single-row (AR) lanes, and the three vocab alignments (aligned shards; V = 4,098 has an odd
//!   2,049-wide shard, so rows past the first take the unaligned scalar path).
//!
//! A pass establishes emulator fidelity for exactly these captured complete-kernel inputs, including the helper call and
//! scratch. It does not discharge the M7 all-path obligations and is not a proof for every input.

use peacemaker_ir::emu::{Emulator, Launch, Memory};
use peacemaker_ir::envelope::Bundle;
use peacemaker_ir::helpers::{load_helpers, Helper};
use peacemaker_ir::inst::{Abi, Arch, Frontend, Kernel, Program};
use peacemaker_lift::bundle::BundleCodec;
use peacemaker_lift::{lift_object, Options};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

const SYMBOL: &str = "lane_leviathan_verify";
const BLOCK: u32 = 512;
const CAPTURE_ENV: &str = "HIPFIRE_LEV_EMU_CAPTURE";
const CAPTURE_FORMAT: &str = "leviathan_emu_capture/1";

/// `include_bytes!` of the committed objects: the replay runs these bytes and nothing else.
const HIPCC_1201: &[u8] = include_bytes!("../../../kernels/lane_leviathan_hipcc_gfx1201.co");
const HIPCC_1200: &[u8] = include_bytes!("../../../kernels/lane_leviathan_hipcc_gfx1200.co");
const PM_1201: &[u8] = include_bytes!("../../../kernels/lane_leviathan_pm_gfx1201.hxaco");

const OPTS: Options = Options {
    frontend: Frontend::Hipcc,
};

/// Index of the gfx1200 arm in [`ARMS`] (negative control 3).
const GFX1200_ARM: usize = 1;

struct ArmSpec {
    name: &'static str,
    /// What a pass of this arm does and does not establish.
    claim: &'static str,
    object: &'static [u8],
    arch: Arch,
    /// Run under `Emulator::gfx1200_as_gfx1201`.
    as_gfx1201: bool,
}

const ARMS: [ArmSpec; 3] = [
    ArmSpec {
        name: "hipcc-gfx1201",
        claim: "committed hipcc .co, gfx1201 numerics vs gfx1201 GPU results of the same object",
        object: HIPCC_1201,
        arch: Arch::Gfx1201,
        as_gfx1201: false,
    },
    ArmSpec {
        name: "hipcc-gfx1200-as-gfx1201",
        claim: "committed gfx1200 hipcc .co under gfx1201 numerics: shared-gfx12-ISA reasoning, no gfx1200 hardware",
        object: HIPCC_1200,
        arch: Arch::Gfx1200,
        as_gfx1201: true,
    },
    ArmSpec {
        name: "pm-twin-gfx1201",
        claim: "committed PM .hxaco bundle (device ELF byte-identical to the hipcc object), gfx1201 numerics",
        object: PM_1201,
        arch: Arch::Gfx1201,
        as_gfx1201: false,
    },
];

fn hex(b: &[u8]) -> String {
    format!("{:x}", Sha256::digest(b))
}

fn clip(s: &str, n: usize) -> String {
    match s.char_indices().nth(n) {
        Some((at, _)) => format!("{}…", &s[..at]),
        None => s.to_string(),
    }
}

// ---------------------------------------------------------------------------------------------- lifting

struct Lifted {
    program: Program,
    helpers: Vec<Helper>,
}

fn lift(spec: &ArmSpec) -> Lifted {
    let lifted =
        lift_object(spec.object, OPTS).unwrap_or_else(|e| panic!("{}: lift: {e}", spec.name));
    let program = lifted.program;
    assert_eq!(
        program.target.arch, spec.arch,
        "{}: lifted target arch",
        spec.name
    );
    let helpers =
        load_helpers(&program).unwrap_or_else(|e| panic!("{}: load_helpers: {e}", spec.name));
    assert!(
        helpers
            .iter()
            .any(|h| h.names.iter().any(|n| n.contains("llv_tail"))),
        "{}: the out-of-line llv_tail helper was not loaded (helpers: {:?})",
        spec.name,
        helpers.iter().map(|h| &h.names).collect::<Vec<_>>()
    );
    Lifted { program, helpers }
}

fn kernel(p: &Program) -> &Kernel {
    p.kernels
        .iter()
        .find(|k| k.symbol.0 == SYMBOL)
        .unwrap_or_else(|| panic!("kernel {SYMBOL} is not in the lifted program"))
}

/// One arm ready to run.
struct Arm<'a> {
    name: &'static str,
    claim: &'static str,
    kernel: &'a Kernel,
    emu: Emulator<'a>,
}

fn emulator<'a>(spec: &ArmSpec, l: &'a Lifted) -> Emulator<'a> {
    let emu = Emulator::with_helpers(&l.program, SYMBOL, &l.helpers)
        .unwrap_or_else(|e| panic!("{}: Emulator::with_helpers: {e}", spec.name));
    if spec.as_gfx1201 {
        emu.gfx1200_as_gfx1201()
    } else {
        emu
    }
}

fn arm<'a>(spec: &ArmSpec, l: &'a Lifted) -> Arm<'a> {
    Arm {
        name: spec.name,
        claim: spec.claim,
        kernel: kernel(&l.program),
        emu: emulator(spec, l),
    }
}

/// The device ELF of a `__CLANG_OFFLOAD_BUNDLE__` for `arch`.
fn device_payload(bytes: &[u8], arch: Arch) -> &[u8] {
    let (bundle, payloads) = Bundle::read(bytes).unwrap_or_else(|e| panic!("bundle: {e}"));
    let i = bundle
        .device_entry(arch)
        .unwrap_or_else(|e| panic!("bundle device entry: {e}"));
    payloads[i]
}

/// The kernarg segment of `lane_leviathan_verify`: the seven explicit args at the offsets the lifted kernel's HSA
/// metadata gives, hidden args zero, `kernarg_segment_size` bytes. The object reads only offsets 0x0..0x2f: the
/// explicit args must end at 48.
fn kernarg_image(k: &Kernel, values: &[u64; 7]) -> Vec<u8> {
    let Abi::Hsa { metadata, .. } = &k.abi else {
        panic!("{SYMBOL}: not an HSA kernel");
    };
    let m = &metadata.parsed;
    let explicit: Vec<(u32, u32)> = m
        .args
        .iter()
        .filter(|a| !a.value_kind.starts_with("hidden_"))
        .map(|a| (a.offset, a.size))
        .collect();
    let want: [(u32, u32); 7] = [(0, 8), (8, 4), (12, 4), (16, 8), (24, 4), (32, 8), (40, 8)];
    assert_eq!(
        explicit, want,
        "explicit kernarg layout of {SYMBOL}: region, lanes_off, rows_off, ptab, mode, cand_local, cand_peer"
    );
    let end = explicit.iter().map(|(o, s)| o + s).max().unwrap();
    assert_eq!(end, 48, "metadata: the explicit kernargs must end at 48");
    assert!(
        m.kernarg_segment_size >= 48,
        "kernarg_segment_size {} is smaller than the explicit args",
        m.kernarg_segment_size
    );
    let mut image = vec![0u8; m.kernarg_segment_size as usize];
    for (&(off, size), &v) in explicit.iter().zip(values) {
        assert!(
            size == 8 || v <= u64::from(u32::MAX),
            "kernarg value {v:#x} does not fit {size} bytes"
        );
        image[off as usize..(off + size) as usize]
            .copy_from_slice(&v.to_le_bytes()[..size as usize]);
    }
    image
}

// ------------------------------------------------------------------------------------- capture parsing

struct Blob {
    file: String,
    bytes: usize,
    sha256: String,
}

struct BufHdr {
    role: String,
    lane: Option<u64>,
    addr: u64,
    bytes: usize,
    before: Blob,
    /// `None`: asserted unchanged at capture time (logits); the expected after-bytes are the before-bytes.
    after: Option<Blob>,
}

impl BufHdr {
    fn name(&self) -> String {
        match self.lane {
            Some(lane) => format!("{}[lane {lane}]", self.role),
            None => self.role.clone(),
        }
    }
}

struct Hdr {
    n: usize,
    vocab: u64,
    launch: u64,
    mode: u64,
    label: String,
    rows: u32,
    /// `region_ptr, lanes_off, rows_off, ptab_ptr, mode, cand_local, cand_peer`.
    kernargs: [u64; 7],
    bufs: Vec<BufHdr>,
}

impl Hdr {
    fn what(&self) -> String {
        format!(
            "launch #{} (V {} set {} {} mode {})",
            self.n, self.vocab, self.launch, self.label, self.mode
        )
    }
}

struct Index {
    dir: PathBuf,
    hdrs: Vec<Hdr>,
}

const KERNARG_ORDER: [&str; 7] = [
    "region_ptr",
    "lanes_off",
    "rows_off",
    "ptab_ptr",
    "mode",
    "cand_local",
    "cand_peer",
];

fn read_json(path: &Path) -> Value {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    serde_json::from_slice(&bytes).unwrap_or_else(|e| panic!("parse {}: {e}", path.display()))
}

fn field<'a>(v: &'a Value, key: &str, what: &str) -> &'a Value {
    v.get(key)
        .unwrap_or_else(|| panic!("{what}: missing field `{key}`"))
}

fn num(v: &Value, key: &str, what: &str) -> u64 {
    field(v, key, what)
        .as_u64()
        .unwrap_or_else(|| panic!("{what}: field `{key}` is not an unsigned integer"))
}

fn text<'a>(v: &'a Value, key: &str, what: &str) -> &'a str {
    field(v, key, what)
        .as_str()
        .unwrap_or_else(|| panic!("{what}: field `{key}` is not a string"))
}

fn parse_blob(v: &Value, what: &str) -> Blob {
    Blob {
        file: text(v, "file", what).to_string(),
        bytes: num(v, "bytes", what) as usize,
        sha256: text(v, "sha256", what).to_string(),
    }
}

/// Generic flat-address apertures of the emulator (`SHARED_APERTURE_HI`, `PRIVATE_APERTURE_HI`): no buffer may overlap
/// their 4 GiB windows, or a flat access to it would be taken for LDS / scratch.
const APERTURE_HI: [u64; 2] = [2, 3];

fn parse_hdr(dir: &Path, n: usize, entry: &Value) -> Hdr {
    let rel = text(entry, "header", "index entry");
    let h = read_json(&dir.join(rel));
    let what = rel.to_string();
    assert_eq!(text(&h, "format", &what), CAPTURE_FORMAT, "{what}: format");
    assert_eq!(text(&h, "symbol", &what), SYMBOL, "{what}: symbol");
    let mode = num(&h, "mode", &what);
    let label = text(&h, "label", &what).to_string();
    let want_label: &[&str] = match mode {
        0 => &["mode0_full"],
        1 => &["mode1_shard_a", "mode1_shard_b"],
        2 => &["mode2_merged"],
        m => panic!("{what}: mode {m} is not 0, 1 or 2"),
    };
    assert!(
        want_label.contains(&label.as_str()),
        "{what}: label {label} for mode {mode}"
    );
    let rows = num(&h, "rows", &what) as u32;
    let grid: Vec<u64> = field(&h, "grid", &what)
        .as_array()
        .unwrap_or_else(|| panic!("{what}: grid is not an array"))
        .iter()
        .map(|g| g.as_u64().unwrap_or_else(|| panic!("{what}: grid entry")))
        .collect();
    assert_eq!(
        grid,
        [u64::from(rows), 1, 1],
        "{what}: grid is one workgroup per row"
    );
    let block: Vec<u64> = field(&h, "block", &what)
        .as_array()
        .unwrap_or_else(|| panic!("{what}: block is not an array"))
        .iter()
        .map(|g| g.as_u64().unwrap_or_else(|| panic!("{what}: block entry")))
        .collect();
    assert_eq!(block, [u64::from(BLOCK), 1, 1], "{what}: block");
    let ka = field(&h, "kernargs", &what);
    let mut kernargs = [0u64; 7];
    for (slot, name) in kernargs.iter_mut().zip(KERNARG_ORDER) {
        *slot = num(ka, name, &format!("{what} kernargs"));
    }
    assert_eq!(kernargs[4], mode, "{what}: kernarg mode");
    let mut bufs = Vec::new();
    for b in field(&h, "buffers", &what)
        .as_array()
        .unwrap_or_else(|| panic!("{what}: buffers is not an array"))
    {
        let role = text(b, "role", &what).to_string();
        let lane = b.get("lane").and_then(Value::as_u64);
        let bwhat = format!("{what} buffer {role}");
        let after = field(b, "after", &bwhat);
        let buf = BufHdr {
            role,
            lane,
            addr: num(b, "addr", &bwhat),
            bytes: num(b, "bytes", &bwhat) as usize,
            before: parse_blob(field(b, "before", &bwhat), &bwhat),
            after: (!after.is_null()).then(|| parse_blob(after, &bwhat)),
        };
        assert_eq!(buf.before.bytes, buf.bytes, "{bwhat}: before blob size");
        assert!(
            buf.bytes % 4 == 0 && buf.bytes > 0,
            "{bwhat}: size {}",
            buf.bytes
        );
        assert_eq!(
            buf.role == "logits",
            buf.after.is_none(),
            "{bwhat}: only logits buffers may omit the after blob"
        );
        if let Some(a) = &buf.after {
            assert_eq!(a.bytes, buf.bytes, "{bwhat}: after blob size");
        }
        let (first, last) = (buf.addr >> 32, (buf.addr + buf.bytes as u64 - 1) >> 32);
        assert!(
            !APERTURE_HI.iter().any(|a| (first..=last).contains(a)),
            "{bwhat}: captured address {:#x}+{:#x} overlaps a flat aperture window (high dword 2 or 3)",
            buf.addr,
            buf.bytes
        );
        bufs.push(buf);
    }
    // Every kernarg pointer names exactly the captured buffer of its role (0 = absent).
    let single = |role: &str| -> Option<u64> {
        let mut it = bufs.iter().filter(|b| b.role == role);
        let first = it.next().map(|b| b.addr);
        assert!(it.next().is_none(), "{what}: more than one {role} buffer");
        first
    };
    assert_eq!(single("region"), Some(kernargs[0]), "{what}: region_ptr");
    assert_eq!(single("ptab").unwrap_or(0), kernargs[3], "{what}: ptab_ptr");
    assert_eq!(
        single("cand_local").unwrap_or(0),
        kernargs[5],
        "{what}: cand_local"
    );
    assert_eq!(
        single("cand_peer").unwrap_or(0),
        kernargs[6],
        "{what}: cand_peer"
    );
    let lanes = num(&h, "lanes", &what);
    let logits = bufs.iter().filter(|b| b.role == "logits").count() as u64;
    assert_eq!(logits, lanes, "{what}: one logits buffer per lane");
    Hdr {
        n,
        vocab: num(&h, "vocab", &what),
        launch: num(&h, "launch", &what),
        mode,
        label,
        rows,
        kernargs,
        bufs,
    }
}

fn load_index(dir: &Path) -> Index {
    let index = read_json(&dir.join("index.json"));
    assert_eq!(
        text(&index, "format", "index"),
        CAPTURE_FORMAT,
        "index format"
    );
    assert_eq!(text(&index, "symbol", "index"), SYMBOL, "index symbol");
    assert_eq!(text(&index, "object", "index"), "hipcc", "index object");
    assert_eq!(
        text(&index, "arch", "index"),
        "gfx1201",
        "the replay compares against gfx1201 hardware results; this capture is of another arch"
    );
    assert_eq!(
        text(&index, "object_sha256", "index"),
        hex(HIPCC_1201),
        "the capture was taken with a different object than the committed lane_leviathan_hipcc_gfx1201.co"
    );
    assert_eq!(
        num(&index, "block", "index"),
        u64::from(BLOCK),
        "index block"
    );
    let order: Vec<&str> = field(&index, "kernarg_order", "index")
        .as_array()
        .expect("kernarg_order array")
        .iter()
        .map(|v| v.as_str().expect("kernarg_order entry"))
        .collect();
    assert_eq!(order, KERNARG_ORDER, "index kernarg_order");
    let hdrs: Vec<Hdr> = field(&index, "launches", "index")
        .as_array()
        .expect("launches array")
        .iter()
        .enumerate()
        .map(|(n, entry)| parse_hdr(dir, n, entry))
        .collect();
    assert!(!hdrs.is_empty(), "the capture holds no launches");
    for mode in 0..3 {
        assert!(
            hdrs.iter().any(|h| h.mode == mode),
            "the capture holds no mode {mode} launch"
        );
    }
    Index {
        dir: dir.to_path_buf(),
        hdrs,
    }
}

fn read_blob(dir: &Path, b: &Blob, what: &str) -> Vec<u8> {
    let bytes =
        std::fs::read(dir.join(&b.file)).unwrap_or_else(|e| panic!("{what}: read {}: {e}", b.file));
    assert_eq!(bytes.len(), b.bytes, "{what}: {} size", b.file);
    assert_eq!(hex(&bytes), b.sha256, "{what}: {} sha256", b.file);
    bytes
}

/// One launch's buffers read from disk (sizes and sha256 verified).
struct Loaded<'h> {
    hdr: &'h Hdr,
    before: Vec<Vec<u8>>,
    /// The GPU's after-bytes; `None` for a buffer the launch must leave unchanged.
    after: Vec<Option<Vec<u8>>>,
}

impl Loaded<'_> {
    /// The bytes buffer `i` must hold after the launch.
    fn want(&self, i: usize) -> &[u8] {
        self.after[i].as_deref().unwrap_or(&self.before[i])
    }
}

/// Words of the region's row records (`rdna_compute::lane_leviathan`: `LLV_ROW_WORDS`, `LR_PTR_LO`, `LR_PTR_HI`,
/// `LR_LANE`; this crate cannot depend on rdna-compute, so a layout drift fails here, closed).
const ROW_WORDS: usize = 6;

fn load<'h>(dir: &Path, hdr: &'h Hdr) -> Loaded<'h> {
    let what = hdr.what();
    let before: Vec<Vec<u8>> = hdr
        .bufs
        .iter()
        .map(|b| read_blob(dir, &b.before, &format!("{what} {}", b.name())))
        .collect();
    let after = hdr
        .bufs
        .iter()
        .map(|b| {
            b.after
                .as_ref()
                .map(|a| read_blob(dir, a, &format!("{what} {} after", b.name())))
        })
        .collect();
    // Every row record must point inside the captured logits of its lane: the capture maps everything the kernel
    // can dereference.
    let ri = hdr
        .bufs
        .iter()
        .position(|b| b.role == "region")
        .unwrap_or_else(|| panic!("{what}: no region buffer"));
    let region = &before[ri];
    let w = |i: usize| u32::from_le_bytes(region[i * 4..i * 4 + 4].try_into().unwrap());
    let rows_off = hdr.kernargs[2] as usize;
    for r in 0..hdr.rows as usize {
        let rec = rows_off + r * ROW_WORDS;
        let ptr = u64::from(w(rec)) | u64::from(w(rec + 1)) << 32;
        let lane = u64::from(w(rec + 2));
        assert!(
            hdr.bufs.iter().any(|b| b.role == "logits"
                && b.lane == Some(lane)
                && b.addr <= ptr
                && ptr < b.addr + b.bytes as u64),
            "{what}: row {r} points at {ptr:#x}, outside the captured logits of lane {lane}"
        );
    }
    Loaded { hdr, before, after }
}

// ------------------------------------------------------------------------------------------- execution

/// An address for the kernarg segment that collides with no captured buffer and whose high dword is neither aperture.
fn kernarg_base(bufs: &[BufHdr], size: u64) -> u64 {
    let mut base = 1u64 << 32;
    while let Some(b) = bufs
        .iter()
        .find(|b| base < b.addr + b.bytes as u64 && b.addr < base + size)
    {
        base = (b.addr + b.bytes as u64 + 0xfff) & !0xfff;
    }
    let (first, last) = (base >> 32, (base + size - 1) >> 32);
    assert!(
        !APERTURE_HI.iter().any(|a| (first..=last).contains(a)),
        "no kernarg address clear of the captured buffers and the flat apertures"
    );
    base
}

fn map_launch(l: &Loaded<'_>, kernarg_addr: u64, image: &[u8]) -> Memory {
    let mut mem = Memory::default();
    for (b, bytes) in l.hdr.bufs.iter().zip(&l.before) {
        mem.map(b.name(), b.addr, bytes.clone())
            .unwrap_or_else(|e| panic!("{}: map {}: {e}", l.hdr.what(), b.name()));
    }
    mem.map("kernarg".into(), kernarg_addr, image.to_vec())
        .unwrap_or_else(|e| panic!("{}: map kernarg: {e}", l.hdr.what()));
    mem
}

/// Every workgroup of the grid in forward or reverse order on `mem`; the emulated instruction total.
fn run_all(
    emu: &Emulator<'_>,
    kernarg_ptr: u64,
    rows: u32,
    reverse: bool,
    mem: &mut Memory,
) -> Result<u64, String> {
    let order: Vec<u32> = if reverse {
        (0..rows).rev().collect()
    } else {
        (0..rows).collect()
    };
    let mut total = 0u64;
    for x in order {
        total += emu
            .run_workgroup_counted(
                &Launch {
                    kernarg_ptr,
                    workgroup: [x, 0, 0],
                    block: [BLOCK, 1, 1],
                    dynamic_lds: 0,
                },
                mem,
            )
            .map_err(|e| format!("workgroup {x}: {e}"))?;
    }
    Ok(total)
}

/// First differing word, the two words, and the number of differing words of two equal-length byte buffers.
fn word_diff(got: &[u8], want: &[u8]) -> Option<(usize, u32, u32, usize)> {
    assert_eq!(got.len(), want.len(), "buffer length");
    if got == want {
        return None;
    }
    let word = |c: &[u8]| u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
    let mut first = None;
    let mut count = 0;
    for (i, (g, w)) in got.chunks_exact(4).zip(want.chunks_exact(4)).enumerate() {
        if g != w {
            count += 1;
            first.get_or_insert((i, word(g), word(w)));
        }
    }
    first.map(|(i, g, w)| (i, g, w, count))
}

fn region_bytes<'m>(mem: &'m Memory, b: &BufHdr, what: &str) -> &'m [u8] {
    &mem.regions
        .iter()
        .find(|r| r.base == b.addr)
        .unwrap_or_else(|| panic!("{what}: buffer {} is not mapped", b.name()))
        .bytes
}

/// Byte-exact comparison of every buffer of `mem` with the GPU's after-bytes; one line per differing buffer.
/// `skip_logits`: leave the logits out (the mutation control changes one on purpose).
fn diff_to_gpu(l: &Loaded<'_>, mem: &Memory, skip_logits: bool) -> Vec<String> {
    let what = l.hdr.what();
    let mut out = Vec::new();
    for (i, b) in l.hdr.bufs.iter().enumerate() {
        if skip_logits && b.role == "logits" {
            continue;
        }
        let got = region_bytes(mem, b, &what);
        if let Some((first, g, w, count)) = word_diff(got, l.want(i)) {
            out.push(format!(
                "buffer {} @{:#x}: first differing word {first} (byte {:#x}): emulator {g:#010x}, gpu {w:#010x}; {count} of {} words differ",
                b.name(),
                b.addr,
                first * 4,
                b.bytes / 4
            ));
        }
    }
    out
}

/// Byte-exact comparison of every buffer of `a` with `b`.
fn diff_mem(l: &Loaded<'_>, a: &Memory, b: &Memory) -> Vec<String> {
    let what = l.hdr.what();
    let mut out = Vec::new();
    for buf in &l.hdr.bufs {
        let (ga, gb) = (region_bytes(a, buf, &what), region_bytes(b, buf, &what));
        if let Some((first, x, y, count)) = word_diff(ga, gb) {
            out.push(format!(
                "buffer {} @{:#x}: first differing word {first}: {x:#010x} vs {y:#010x}; {count} of {} words differ",
                buf.name(),
                buf.addr,
                buf.bytes / 4
            ));
        }
    }
    out
}

/// What one arm did with one launch.
#[derive(Default)]
struct ArmOut {
    workgroups: u64,
    instructions: u64,
    reverse_instructions: u64,
    failures: Vec<String>,
}

fn run_arm(a: &Arm<'_>, l: &Loaded<'_>) -> ArmOut {
    let h = l.hdr;
    let what = format!("{} {}", a.name, h.what());
    let image = kernarg_image(a.kernel, &h.kernargs);
    let kbase = kernarg_base(&h.bufs, image.len() as u64);
    let mut out = ArmOut::default();

    let mut fwd = map_launch(l, kbase, &image);
    match run_all(&a.emu, kbase, h.rows, false, &mut fwd) {
        Ok(n) => {
            out.workgroups = u64::from(h.rows);
            out.instructions = n;
        }
        Err(e) => {
            out.failures
                .push(format!("{what}: emulator error: {}", clip(&e, 700)));
            return out;
        }
    }
    for d in diff_to_gpu(l, &fwd, false) {
        out.failures.push(format!("{what}: {d}"));
    }

    let mut rev = map_launch(l, kbase, &image);
    match run_all(&a.emu, kbase, h.rows, true, &mut rev) {
        Ok(n) => {
            out.reverse_instructions = n;
            for d in diff_mem(l, &rev, &fwd) {
                out.failures.push(format!(
                    "{what}: reverse workgroup order differs from forward order: {d}"
                ));
            }
        }
        Err(e) => out.failures.push(format!(
            "{what}: emulator error in reverse workgroup order: {}",
            clip(&e, 700)
        )),
    }
    out
}

// ----------------------------------------------------------------------------------------- the census

/// Special-value census of captured mode 0 logits (each row appears once there).
#[derive(Default)]
struct Census {
    words: u64,
    pos_inf: u64,
    neg_inf: u64,
    pos_zero: u64,
    neg_zero: u64,
    nan: BTreeMap<u32, u64>,
}

impl Census {
    fn of(l: &Loaded<'_>) -> Census {
        let mut c = Census::default();
        for (b, bytes) in l.hdr.bufs.iter().zip(&l.before) {
            if b.role != "logits" {
                continue;
            }
            for w in bytes.chunks_exact(4) {
                let bits = u32::from_le_bytes([w[0], w[1], w[2], w[3]]);
                c.words += 1;
                match bits {
                    0x7f80_0000 => c.pos_inf += 1,
                    0xff80_0000 => c.neg_inf += 1,
                    0x0000_0000 => c.pos_zero += 1,
                    0x8000_0000 => c.neg_zero += 1,
                    b if b & 0x7f80_0000 == 0x7f80_0000 => *c.nan.entry(b).or_default() += 1,
                    _ => {}
                }
            }
        }
        c
    }

    fn add(&mut self, o: Census) {
        self.words += o.words;
        self.pos_inf += o.pos_inf;
        self.neg_inf += o.neg_inf;
        self.pos_zero += o.pos_zero;
        self.neg_zero += o.neg_zero;
        for (k, v) in o.nan {
            *self.nan.entry(k).or_default() += v;
        }
    }
}

/// NaN payload words `make_lane` (`f32::NAN`) and `plant_edges` (`NANS`) put into the matrix.
const PLANTED_NANS: [u32; 4] = [0x7fc0_0000, 0xffc1_2345, 0x7f80_0001, 0x7fff_ffff];

struct LaunchOut {
    arms: Vec<ArmOut>,
    census: Option<Census>,
}

fn replay_launch(index: &Index, hdr: &Hdr, arms: &[Arm<'_>]) -> LaunchOut {
    let l = load(&index.dir, hdr);
    let census = (hdr.mode == 0).then(|| Census::of(&l));
    LaunchOut {
        arms: arms.iter().map(|a| run_arm(a, &l)).collect(),
        census,
    }
}

/// Bytes of logits one launch maps: the work scales with it, so the largest launches go first.
fn cost(h: &Hdr) -> usize {
    h.bufs
        .iter()
        .filter(|b| b.role == "logits")
        .map(|b| b.bytes)
        .sum()
}

// ---------------------------------------------------------------------------------- negative controls

/// The first mode 0 launch of the smallest vocab: cheap, and its kernel always reaches the helper call.
fn control_launch(index: &Index) -> &Hdr {
    index
        .hdrs
        .iter()
        .filter(|h| h.mode == 0)
        .min_by_key(|h| (h.vocab, h.n))
        .expect("a mode 0 launch")
}

/// Control 1: the kernel-only emulator refuses the helper call.
fn control_no_helpers(index: &Index, lifted: &Lifted) {
    let hdr = control_launch(index);
    let l = load(&index.dir, hdr);
    let emu = Emulator::new(&lifted.program, SYMBOL).expect("kernel-only emulator builds");
    let image = kernarg_image(kernel(&lifted.program), &hdr.kernargs);
    let kbase = kernarg_base(&hdr.bufs, image.len() as u64);
    let mut mem = map_launch(&l, kbase, &image);
    match run_all(&emu, kbase, hdr.rows, false, &mut mem) {
        Ok(_) => panic!(
            "negative control 1: Emulator::new (no helpers) ran {} to completion; the helper call was not refused",
            hdr.what()
        ),
        Err(e) => {
            assert!(
                e.contains("s_swappc_b64") && e.contains("target"),
                "negative control 1: expected the unresolved call-target error at s_swappc_b64, got: {}",
                clip(&e, 700)
            );
            eprintln!(
                "control 1 ok: Emulator::new without helpers fails at the first call: {}",
                clip(&e, 220)
            );
        }
    }
}

/// Control 2: a flipped logit the kernel reads must change the emulator's output.
fn control_mutation(index: &Index, a: &Arm<'_>) {
    let hdr = control_launch(index);
    let mut l = load(&index.dir, hdr);
    let region = hdr.bufs.iter().position(|b| b.role == "region").unwrap();
    let word = |i: usize| {
        let want = l.want(region);
        u32::from_le_bytes(want[i * 4..i * 4 + 4].try_into().unwrap())
    };
    // Lane 0's out record in the GPU's after-bytes: [ticket, done, accepted, ntok, rng, _, _, _, tokens...].
    assert!(
        word(1) == 1 && word(3) >= 1,
        "control 2: lane 0 has no finished out record in {}",
        hdr.what()
    );
    let token = word(8) as usize;
    assert!(
        token < hdr.vocab as usize,
        "control 2: token {token} outside the vocab"
    );
    let logits = hdr
        .bufs
        .iter()
        .position(|b| b.role == "logits" && b.lane == Some(0))
        .expect("lane 0 logits");
    // Row 0 of lane 0 starts at word 0 of its logits (mode 0: base 0, full rows).
    const NEG_INF: u32 = 0xff80_0000;
    let at = token * 4;
    let old = u32::from_le_bytes(l.before[logits][at..at + 4].try_into().unwrap());
    assert_ne!(
        old, NEG_INF,
        "control 2: the emitted token's logit is already -inf"
    );
    l.before[logits][at..at + 4].copy_from_slice(&NEG_INF.to_le_bytes());

    let image = kernarg_image(a.kernel, &hdr.kernargs);
    let kbase = kernarg_base(&hdr.bufs, image.len() as u64);
    let mut mem = map_launch(&l, kbase, &image);
    if let Err(e) = run_all(&a.emu, kbase, hdr.rows, false, &mut mem) {
        panic!(
            "control 2: the mutated launch failed to execute instead of changing the result: {}",
            clip(&e, 700)
        );
    }
    let diffs = diff_to_gpu(&l, &mem, true);
    assert!(
        !diffs.is_empty(),
        "control 2: replacing lane 0's emitted token {token}'s logit {old:#010x} with -inf left every output buffer equal to the GPU's: the replay would not notice a wrong input"
    );
    eprintln!(
        "control 2 ok: lane 0 token {token} logit {old:#010x} -> -inf changes {} output buffer(s) of {}: {}",
        diffs.len(),
        hdr.what(),
        clip(&diffs[0], 200)
    );
}

/// Control 3: the gfx1200 object without the opt-in refuses to run and names it.
fn control_gfx1200_opt_in(index: &Index, lifted: &Lifted) {
    let hdr = control_launch(index);
    let l = load(&index.dir, hdr);
    let emu = Emulator::with_helpers(&lifted.program, SYMBOL, &lifted.helpers)
        .expect("a gfx1200 program is accepted at construction");
    let image = kernarg_image(kernel(&lifted.program), &hdr.kernargs);
    let kbase = kernarg_base(&hdr.bufs, image.len() as u64);
    let mut mem = map_launch(&l, kbase, &image);
    match run_all(&emu, kbase, 1, false, &mut mem) {
        Ok(_) => panic!("negative control 3: the gfx1200 object ran without gfx1200_as_gfx1201"),
        Err(e) => {
            assert!(
                e.contains("gfx1200_as_gfx1201"),
                "negative control 3: the refusal does not name the opt-in: {}",
                clip(&e, 700)
            );
            eprintln!(
                "control 3 ok: gfx1200 without the opt-in refuses: {}",
                clip(&e, 220)
            );
        }
    }
}

// -------------------------------------------------------------------------------------------- tests

/// CPU-only: the committed objects lift, their helpers load, the kernarg ABI is the seven explicit args, the PM bundle's
/// device ELF is the hipcc object's, and an emulator builds for every arm (so every opcode of the kernel and the
/// helper has semantics).
#[test]
fn committed_leviathan_objects_load_in_the_emulator() {
    assert_eq!(
        device_payload(HIPCC_1201, Arch::Gfx1201),
        device_payload(PM_1201, Arch::Gfx1201),
        "the PM twin's device ELF is not byte-identical to the hipcc object's"
    );
    for spec in &ARMS {
        let l = lift(spec);
        let _ = emulator(spec, &l);
        let image = kernarg_image(kernel(&l.program), &[0; 7]);
        assert!(image.len() >= 48, "{}: kernarg image", spec.name);
        let helper = l
            .helpers
            .iter()
            .find(|h| h.names.iter().any(|n| n.contains("llv_tail")))
            .unwrap();
        assert!(
            !helper.insts.is_empty() && helper.size > 0,
            "{}: llv_tail body",
            spec.name
        );
        eprintln!(
            "{}: llv_tail @{:#x} size {:#x}, {} instructions; kernarg segment {} bytes",
            spec.name,
            helper.entry_va,
            helper.size,
            helper.insts.len(),
            image.len()
        );
    }
}

#[test]
fn kernarg_base_avoids_every_buffer_and_both_apertures() {
    let blob = || Blob {
        file: String::new(),
        bytes: 0,
        sha256: String::new(),
    };
    let buf = |addr: u64, bytes: usize| BufHdr {
        role: "region".into(),
        lane: None,
        addr,
        bytes,
        before: blob(),
        after: None,
    };
    // Nothing at 4 GiB: the first candidate.
    assert_eq!(kernarg_base(&[buf(0x7f00_0000_0000, 4096)], 104), 1 << 32);
    // A buffer on the first candidate pushes the segment past it, page aligned.
    let at = (1u64 << 32) + 8;
    let base = kernarg_base(&[buf(at, 4096)], 104);
    assert!(base >= at + 4096 && base % 4096 == 0, "{base:#x}");
    // A buffer that runs from the first candidate past 8 GiB leaves only the aperture windows (high dwords 2 and 3):
    // refused, not placed.
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        kernarg_base(&[buf(1 << 32, (1usize << 32) + 4096)], 104)
    }));
    assert!(
        r.is_err(),
        "a kernarg segment inside an aperture window must be refused"
    );
}

#[test]
fn word_diff_reports_first_word_and_count() {
    let a = [1u8, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0];
    assert_eq!(word_diff(&a, &a), None);
    let mut b = a;
    b[4] = 9;
    b[8] = 7;
    assert_eq!(word_diff(&b, &a), Some((1, 9, 2, 2)));
}

/// The differential execution. See the file doc comment.
#[test]
#[ignore = "needs HIPFIRE_LEV_EMU_CAPTURE (a leviathan_emu_capture directory)"]
fn leviathan_emulator_equals_gpu_capture() {
    let dir = match std::env::var(CAPTURE_ENV) {
        Ok(d) if !d.is_empty() => PathBuf::from(d),
        _ => panic!("{CAPTURE_ENV} must name a leviathan_emu_capture directory"),
    };
    let t0 = Instant::now();
    let index = load_index(&dir);
    assert_eq!(
        device_payload(HIPCC_1201, Arch::Gfx1201),
        device_payload(PM_1201, Arch::Gfx1201),
        "the PM twin's device ELF is not byte-identical to the hipcc object's"
    );
    let lifted: Vec<Lifted> = ARMS.iter().map(lift).collect();
    let arms: Vec<Arm<'_>> = ARMS.iter().zip(&lifted).map(|(s, l)| arm(s, l)).collect();
    eprintln!(
        "replay: {} launches from {} ({} MiB of logits), {} arms",
        index.hdrs.len(),
        dir.display(),
        index.hdrs.iter().map(cost).sum::<usize>() / (1024 * 1024),
        arms.len()
    );

    control_no_helpers(&index, &lifted[0]);
    control_mutation(&index, &arms[0]);
    control_gfx1200_opt_in(&index, &lifted[GFX1200_ARM]);

    // The matrix: launches in parallel, largest first; each launch is loaded once and run by every arm.
    let mut order: Vec<usize> = (0..index.hdrs.len()).collect();
    order.sort_by_key(|&i| std::cmp::Reverse(cost(&index.hdrs[i])));
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .min(order.len());
    let next = AtomicUsize::new(0);
    let done = AtomicUsize::new(0);
    let results: Vec<Mutex<Option<LaunchOut>>> =
        (0..index.hdrs.len()).map(|_| Mutex::new(None)).collect();
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| loop {
                let k = next.fetch_add(1, Ordering::Relaxed);
                let Some(&i) = order.get(k) else { return };
                let out = replay_launch(&index, &index.hdrs[i], &arms);
                *results[i].lock().unwrap() = Some(out);
                let d = done.fetch_add(1, Ordering::Relaxed) + 1;
                if d % 20 == 0 || d == order.len() {
                    eprintln!("replay: {d}/{} launches", order.len());
                }
            });
        }
    });

    // Receipts.
    #[derive(Default)]
    struct Tally {
        launches: u64,
        workgroups: u64,
        instructions: u64,
        reverse_instructions: u64,
        equal: u64,
        different: u64,
    }
    let mut tally: BTreeMap<(usize, u64), Tally> = BTreeMap::new();
    let mut failures: Vec<String> = Vec::new();
    let mut census = Census::default();
    for (h, slot) in index.hdrs.iter().zip(results) {
        let out = slot
            .into_inner()
            .unwrap()
            .expect("every launch was replayed");
        if let Some(c) = out.census {
            census.add(c);
        }
        for (ai, o) in out.arms.into_iter().enumerate() {
            let t = tally.entry((ai, h.mode)).or_default();
            t.launches += 1;
            t.workgroups += o.workgroups;
            t.instructions += o.instructions;
            t.reverse_instructions += o.reverse_instructions;
            if o.failures.is_empty() {
                t.equal += 1;
            } else {
                t.different += 1;
                failures.extend(o.failures);
            }
        }
    }
    eprintln!(
        "== leviathan emulator replay receipt ({:.1} s, {threads} threads)",
        t0.elapsed().as_secs_f64()
    );
    for (ai, a) in arms.iter().enumerate() {
        eprintln!("-- arm {}: {}", a.name, a.claim);
        for mode in 0..3u64 {
            let t = &tally[&(ai, mode)];
            eprintln!(
                "   mode {mode}: launches {} workgroups {} emulated instructions {} (reverse order {}) equal {} different {}",
                t.launches, t.workgroups, t.instructions, t.reverse_instructions, t.equal, t.different
            );
        }
    }
    eprintln!(
        "-- captured mode 0 logits census: {} words, +inf {}, -inf {}, +0.0 {}, -0.0 {}, {} distinct NaN payloads; planted payloads present: {:?}",
        census.words,
        census.pos_inf,
        census.neg_inf,
        census.pos_zero,
        census.neg_zero,
        census.nan.len(),
        PLANTED_NANS.map(|p| (format!("{p:#010x}"), census.nan.get(&p).copied().unwrap_or(0)))
    );
    for p in PLANTED_NANS {
        assert!(
            census.nan.contains_key(&p),
            "the capture holds no logit with NaN payload {p:#010x}: it is not the X1 twin matrix"
        );
    }
    assert!(
        census.pos_inf > 0 && census.neg_inf > 0 && census.pos_zero > 0 && census.neg_zero > 0,
        "the capture lacks +inf / -inf / +0.0 / -0.0 logits: it is not the X1 twin matrix"
    );

    if !failures.is_empty() {
        for f in failures.iter().take(25) {
            eprintln!("DIFFERENCE: {f}");
        }
        panic!(
            "emulator != GPU capture: {} failing launch/arm result(s) ({} message(s); first 25 above)",
            tally.values().map(|t| t.different).sum::<u64>(),
            failures.len()
        );
    }
    eprintln!("all arms, all modes: every buffer byte-identical to the GPU capture, forward and reverse workgroup order");
}
