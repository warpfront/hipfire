//! Byte oracle + timing for the Qwen4 top-10 MoE scatter on exact gfx1201:
//! the block-wide prefix scan (`moe_scatter_fused_top10`, default,
//! `HIPFIRE_QWEN4_MOE_SCATTER_SCAN`) against the serial thread-0 walk
//! (`moe_scatter_fused_top10_serial`, the pre-change kernel and the kill switch).
//!
//! Both arms run through the product launcher `Gpu::moe_scatter_fused_top10`
//! with the `qwen4_moe_scatter_scan` feature flag flipped between launches, so
//! the oracle exercises the same launch path, kernel argument blob and module
//! the daemon uses (one module, two symbols).
//!
//! Per route (rows x top-10 over 512 experts, `grouped_rows` = the production
//! worst-case bound) every output the kernel writes is poisoned (0x5A) with a
//! guard tail (0xA5), launched `--reps` times per arm, and compared as BYTES
//! over the whole buffer, guard included:
//!
//! * `counts`, `offsets`, `tile ids`: must be byte-identical between the arms
//!   and equal to the host model (integer sums, no arrival-order dependence).
//! * `sorted slot index`, `inverse perm`: written in LDS-atomic arrival order
//!   inside one workgroup (phase 3b, unchanged by the scan), so the order of the
//!   slots WITHIN one expert is not part of the contract (production compares
//!   it per-run canonical too). They are checked canonically: per-expert slot
//!   multisets equal the host model's AND the other arm's, `sorted[inverse[i]] == i`
//!   for every routed slot, `-1` for every unrouted one. The strict byte
//!   comparison against the serial arm is printed as an informational counter
//!   and never fails.
//! * the downstream chain (`--no-chain` skips it): for seeded decode routes
//!   (rows 1..=8, uniform / Zipf / same-row) both arms feed the REAL chain
//!   rotate -> `gemm_mq4g256v2_moe_grouped_top10` ->
//!   `moe_gate_up_unscatter_silu_rotate128_top10` ->
//!   `gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded` ->
//!   `moe_down_combine_top10_batched` on synthetic finite MQ4 experts in the real
//!   layout (64 distinct images, expert e -> image e % 64). The unscattered
//!   activations, the expanded down rows and the combine output must be
//!   byte-identical between the arms whatever the within-expert order
//!   (`chain_routes`, `chain_bad_bytes`).
//! * guard words and every word past the live region keep their poison
//!   (sorted / tile padding is the `-1` sentinel).
//!
//! Routes: uniform (distinct experts per row), Zipf-skewed, every row to the
//! same ten experts (ties on every count), all rows to experts {0..9}, edge
//! experts {0, 1, 510, 511}, counts that land exactly on 16 (no padding), and
//! bad ids (-1, 512, INT_MIN, INT_MAX) mixed into valid routes. Rows 1..=8
//! (the decode lanes) plus 16, 33, 64, 300, 512 (prefill chunks).
//!
//! Timing (`--time`, default on): both arms, 200 replays between two events,
//! eager (host-launch inclusive) and as one captured HIP graph (GPU time
//! only), median of `--rounds` alternating rounds, microseconds per launch.
//!
//! Run (exact gfx1201 only):
//!
//! ```text
//! cargo run --release -p rdna-compute --features lab --example moe_scatter_scan_oracle -- \
//!     [--seeds 24] [--reps 3] [--replays 200] [--rounds 7] [--quick] [--no-time] [--no-chain]
//! ```
//!
//! Exit status 0 only when every check passes; one line per failing route.

use rdna_compute::{DType, Gpu, GpuTensor};
use std::sync::Arc;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

const E: usize = 512;
const K: usize = 10;
const BLOCK_M: usize = 16;
const GUARD: usize = 64;
const POISON: u8 = 0x5A;
const GUARD_BYTE: u8 = 0xA5;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// `K` distinct experts out of `E` weighted by `weight(e)` (rejection on the
/// cumulative table; routes never repeat an expert inside one row).
fn pick_row(rng: &mut Rng, cdf: &[f64], out: &mut Vec<i32>) {
    let start = out.len();
    while out.len() - start < K {
        let u = rng.unit() * cdf[E - 1];
        let e = cdf.partition_point(|&c| c < u).min(E - 1) as i32;
        if !out[start..].contains(&e) {
            out.push(e);
        }
    }
}

fn cdf_of(weight: impl Fn(usize) -> f64) -> Vec<f64> {
    let mut acc = 0.0;
    (0..E)
        .map(|e| {
            acc += weight(e);
            acc
        })
        .collect()
}

#[derive(Clone, Copy, Debug)]
enum Kind {
    Uniform,
    Zipf,
    SameRow,
    FirstTen,
    EdgeExperts,
    ExactTiles,
    BadIds,
}

const KINDS: [Kind; 7] = [
    Kind::Uniform,
    Kind::Zipf,
    Kind::SameRow,
    Kind::FirstTen,
    Kind::EdgeExperts,
    Kind::ExactTiles,
    Kind::BadIds,
];

const BAD: [i32; 5] = [-1, E as i32, i32::MIN, i32::MAX, -7];

fn route(kind: Kind, rows: usize, seed: u64) -> Vec<i32> {
    let mut rng = Rng(seed ^ (rows as u64) << 32 ^ 0xC0FF_EE00);
    let mut v = Vec::with_capacity(rows * K);
    match kind {
        Kind::Uniform => {
            let cdf = cdf_of(|_| 1.0);
            for _ in 0..rows {
                pick_row(&mut rng, &cdf, &mut v);
            }
        }
        Kind::Zipf => {
            let cdf = cdf_of(|e| 1.0 / ((e + 1) as f64).powf(1.1));
            for _ in 0..rows {
                pick_row(&mut rng, &cdf, &mut v);
            }
        }
        Kind::SameRow => {
            let cdf = cdf_of(|_| 1.0);
            let mut row = Vec::new();
            pick_row(&mut rng, &cdf, &mut row);
            for _ in 0..rows {
                v.extend_from_slice(&row);
            }
        }
        Kind::FirstTen => {
            for _ in 0..rows {
                v.extend(0..K as i32);
            }
        }
        Kind::EdgeExperts => {
            // Experts 0, 1, 510, 511 plus six random per row: the scan's first
            // and last thread ranges and the M_total cell.
            let cdf = cdf_of(|_| 1.0);
            for _ in 0..rows {
                let start = v.len();
                v.extend([0, 1, 510, 511]);
                while v.len() - start < K {
                    let e = (rng.next() % E as u64) as i32;
                    if !v[start..].contains(&e) {
                        v.push(e);
                    }
                }
                let _ = &cdf;
            }
        }
        Kind::ExactTiles => {
            // 16 rows per expert group: slot counts land exactly on the tile.
            // Row r takes experts 10*(r/16 % 51) .. +10, so every touched expert
            // holds a multiple of 16 slots when rows is a multiple of 16
            // (and a ragged remainder otherwise).
            for r in 0..rows {
                let base = 10 * ((r / 16) % 51);
                v.extend((0..K).map(|k| (base + k) as i32));
            }
        }
        Kind::BadIds => {
            let cdf = cdf_of(|_| 1.0);
            for _ in 0..rows {
                pick_row(&mut rng, &cdf, &mut v);
            }
            let n = (rows * K / 7).max(1);
            for _ in 0..n {
                let at = rng.below(v.len());
                v[at] = BAD[rng.below(BAD.len())];
            }
        }
    }
    v
}

fn is_valid(id: i32) -> bool {
    (0..E as i32).contains(&id)
}

fn grouped_rows_bound(slots: usize) -> usize {
    (slots + (BLOCK_M - 1) * slots.min(E)).next_multiple_of(BLOCK_M)
}

/// Host model of the order-independent outputs.
struct Model {
    counts: Vec<i32>,
    offsets: Vec<i32>,
    tiles: Vec<i32>,
    raw: Vec<usize>,
    /// Canonical sorted index: each expert's run ascending, `-1` padding.
    sorted: Vec<i32>,
}

fn model(topk: &[i32], gr: usize) -> Model {
    let mut raw = vec![0usize; E];
    for &id in topk {
        if is_valid(id) {
            raw[id as usize] += 1;
        }
    }
    let mut counts = vec![0i32; E];
    let mut offsets = vec![0i32; E + 1];
    let mut acc = 0i32;
    for e in 0..E {
        let padded = raw[e].div_ceil(BLOCK_M) * BLOCK_M;
        counts[e] = padded as i32;
        offsets[e] = acc;
        acc += padded as i32;
    }
    offsets[E] = acc;
    let mut tiles = vec![-1i32; gr / BLOCK_M];
    let mut sorted = vec![-1i32; gr];
    let mut fill = vec![0usize; E];
    for (slot, &id) in topk.iter().enumerate() {
        if is_valid(id) {
            let e = id as usize;
            sorted[offsets[e] as usize + fill[e]] = slot as i32;
            fill[e] += 1;
        }
    }
    for e in 0..E {
        for t in (offsets[e] as usize / BLOCK_M)..(offsets[e + 1] as usize / BLOCK_M) {
            tiles[t] = e as i32;
        }
    }
    Model {
        counts,
        offsets,
        tiles,
        raw,
        sorted,
    }
}

struct Bufs {
    topk: GpuTensor,
    counts: GpuTensor,
    offsets: GpuTensor,
    sorted: GpuTensor,
    tiles: GpuTensor,
    inverse: GpuTensor,
    slots: usize,
    gr: usize,
}

/// Word counts of the six outputs for a call of `slots` and `gr`.
fn lens(slots: usize, gr: usize) -> [usize; 5] {
    [E, E + 1, gr, gr / BLOCK_M, slots]
}

fn alloc_bytes(gpu: &mut Gpu, bytes: usize) -> Res<GpuTensor> {
    Ok(gpu.zeros(&[bytes.div_ceil(4)], DType::F32)?)
}

fn bufs(gpu: &mut Gpu, topk: &[i32]) -> Res<Bufs> {
    let slots = topk.len();
    let gr = grouped_rows_bound(slots);
    let bytes: Vec<u8> = topk.iter().flat_map(|v| v.to_le_bytes()).collect();
    let t_topk = gpu.upload_raw(&bytes, &[slots])?;
    let l = lens(slots, gr);
    let mut out = Vec::new();
    for words in l {
        out.push(alloc_bytes(gpu, words * 4 + GUARD)?);
    }
    let mut it = out.into_iter();
    Ok(Bufs {
        topk: t_topk,
        counts: it.next().unwrap(),
        offsets: it.next().unwrap(),
        sorted: it.next().unwrap(),
        tiles: it.next().unwrap(),
        inverse: it.next().unwrap(),
        slots,
        gr,
    })
}

impl Bufs {
    fn outs(&self) -> [&GpuTensor; 5] {
        [
            &self.counts,
            &self.offsets,
            &self.sorted,
            &self.tiles,
            &self.inverse,
        ]
    }

    fn poison(&self, gpu: &Gpu) -> Res<()> {
        for (t, words) in self.outs().into_iter().zip(lens(self.slots, self.gr)) {
            let mut v = vec![POISON; words * 4];
            v.extend(std::iter::repeat(GUARD_BYTE).take(GUARD));
            gpu.hip.memcpy_htod(&t.buf, &v)?;
        }
        Ok(())
    }

    /// The five buffers as bytes, guard included.
    fn download(&self, gpu: &Gpu) -> Res<[Vec<u8>; 5]> {
        let mut res: [Vec<u8>; 5] = Default::default();
        for (i, (t, words)) in self
            .outs()
            .into_iter()
            .zip(lens(self.slots, self.gr))
            .enumerate()
        {
            let mut v = vec![0u8; words * 4 + GUARD];
            gpu.hip.memcpy_dtoh(&mut v, &t.buf)?;
            res[i] = v;
        }
        Ok(res)
    }

    fn launch(&self, gpu: &mut Gpu) -> Res<()> {
        gpu.moe_scatter_fused_top10(
            &self.topk,
            &self.counts,
            &self.offsets,
            &self.sorted,
            &self.tiles,
            &self.inverse,
            self.slots,
            E,
            self.gr,
            BLOCK_M,
        )?;
        Ok(())
    }

    fn free(self, gpu: &mut Gpu) -> Res<()> {
        for t in [
            self.topk,
            self.counts,
            self.offsets,
            self.sorted,
            self.tiles,
            self.inverse,
        ] {
            gpu.free_tensor(t)?;
        }
        Ok(())
    }
}

fn set_scan(gpu: &mut Gpu, scan: bool) {
    Arc::make_mut(&mut gpu.flags).qwen4_moe_scatter_scan = scan;
}

fn run_arm(gpu: &mut Gpu, b: &Bufs, scan: bool) -> Res<[Vec<u8>; 5]> {
    set_scan(gpu, scan);
    b.poison(gpu)?;
    gpu.hip.device_synchronize()?;
    b.launch(gpu)?;
    gpu.hip.device_synchronize()?;
    b.download(gpu)
}

fn words(b: &[u8]) -> Vec<i32> {
    b.chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

const NAMES: [&str; 5] = ["counts", "offsets", "sorted", "tiles", "inverse"];

/// Failure classes, in report order.
const CLASSES: [&str; 9] = [
    "counts",
    "offsets",
    "tiles",
    "sorted",
    "inverse",
    "guard",
    "poison",
    "model-mismatch",
    "chain",
];

#[derive(Default)]
struct Tally {
    routes: usize,
    launches: usize,
    /// Routes whose serial arm was byte-stable across reps (strict sorted and
    /// inverse byte comparison applied).
    strict_routes: usize,
    /// Routes where the serial arm itself reordered a run between reps.
    order_nondeterministic: usize,
    /// Routes whose serial arm was byte-stable and whose scan arm had the same
    /// sorted/inverse BYTES (informational).
    strict_same: usize,
    /// Routes whose serial arm was byte-stable and whose scan arm ordered a run
    /// differently (informational: within-expert order is not a contract).
    strict_differs: usize,
    /// Downstream-chain routes run, bytes that differed between the arms, and
    /// bytes that differed between the serial arm's own reps.
    chain_routes: usize,
    chain_bad_bytes: usize,
    chain_serial_unstable_bytes: usize,
    /// Route kind the checks currently run for (set by the driver).
    kind: String,
    total: usize,
    per_class: std::collections::BTreeMap<&'static str, usize>,
    per_kind: std::collections::BTreeMap<String, std::collections::BTreeMap<&'static str, usize>>,
    /// The first failure of each class, full detail.
    first: std::collections::BTreeMap<&'static str, String>,
}

impl Tally {
    fn fail(&mut self, class: &'static str, msg: String) {
        self.total += 1;
        *self.per_class.entry(class).or_default() += 1;
        *self
            .per_kind
            .entry(self.kind.clone())
            .or_default()
            .entry(class)
            .or_default() += 1;
        self.first.entry(class).or_insert(msg);
    }

    fn report(&self) {
        if self.total == 0 {
            return;
        }
        println!("FAILURES: {} total, by class:", self.total);
        for class in CLASSES {
            if let Some(n) = self.per_class.get(class) {
                println!("  {class:<15} {n}");
            }
        }
        println!("by route kind (class: count):");
        for (kind, classes) in &self.per_kind {
            let line: Vec<String> = CLASSES
                .iter()
                .filter_map(|c| classes.get(c).map(|n| format!("{c}: {n}")))
                .collect();
            println!("  {kind:<12} {}", line.join(", "));
        }
        println!("first failure of each class:");
        for class in CLASSES {
            if let Some(msg) = self.first.get(class) {
                println!("  [{class}] {msg}");
            }
        }
    }
}

/// Semantic check of a sorted/inverse pair against the model.
fn check_order(label: &str, topk: &[i32], m: &Model, out: &[Vec<u8>; 5], t: &mut Tally) {
    let sorted = words(&out[2][..out[2].len() - GUARD]);
    let inverse = words(&out[4][..out[4].len() - GUARD]);
    // Canonical per-run order equals the model.
    let mut canon = sorted.clone();
    for e in 0..E {
        let (lo, n) = (m.offsets[e] as usize, m.raw[e]);
        canon[lo..lo + n].sort_unstable();
    }
    if canon != m.sorted {
        t.fail(
            "model-mismatch",
            format!("{label}: per-expert canonical sorted differs from the model"),
        );
    }
    for (i, &id) in topk.iter().enumerate() {
        let pos = inverse[i];
        if is_valid(id) {
            let (lo, n) = (m.offsets[id as usize], m.raw[id as usize] as i32);
            let ok = pos >= lo
                && pos < lo + n
                && (pos as usize) < sorted.len()
                && sorted[pos as usize] == i as i32;
            if !ok {
                t.fail(
                    "model-mismatch",
                    format!("{label}: inverse[{i}] = {pos} does not point back to slot {i}"),
                );
                return;
            }
        } else if pos != -1 {
            t.fail(
                "model-mismatch",
                format!("{label}: unrouted slot {i} (id {id}) has inverse {pos}, want -1"),
            );
            return;
        }
    }
}

/// The word every output buffer is poisoned with. No legitimate output word
/// can equal it (counts, offsets, tile ids, sorted slots and inverse positions
/// are small non-negative ints or -1), while single 0x5A BYTES do occur in
/// legitimate data (tile id 90, count 0x5A...), so the check is per i32 word.
const POISON_WORD: i32 = i32::from_le_bytes([POISON; 4]);

/// Index of the first word equal to [`POISON_WORD`] among the live words
/// (`live` = how many leading words the kernel must have written).
fn first_poison_word(body: &[u8], live: usize) -> Option<usize> {
    words(body)
        .into_iter()
        .take(live)
        .position(|w| w == POISON_WORD)
}

fn guards_intact(label: &str, out: &[Vec<u8>; 5], slots: usize, gr: usize, t: &mut Tally) {
    // Words the kernel writes in full: counts E, offsets E+1, sorted gr (-1
    // padded to the bound), tile ids gr/16 (-1 beyond the live tiles), inverse slots.
    let live = lens(slots, gr);
    for (i, buf) in out.iter().enumerate() {
        let (body, guard) = buf.split_at(buf.len() - GUARD);
        if let Some(at) = guard.iter().position(|&b| b != GUARD_BYTE) {
            t.fail(
                "guard",
                format!(
                    "{label}: {} guard byte {at} is {:#04x}, want {GUARD_BYTE:#04x}",
                    NAMES[i], guard[at]
                ),
            );
        }
        if let Some(at) = first_poison_word(body, live[i]) {
            t.fail(
                "poison",
                format!(
                    "{label}: {} word {at} of {} still holds the poison word {POISON_WORD:#010x}",
                    NAMES[i], live[i]
                ),
            );
        }
    }
}

fn check_route(gpu: &mut Gpu, label: &str, topk: &[i32], reps: usize, t: &mut Tally) -> Res<()> {
    let b = bufs(gpu, topk)?;
    let m = model(topk, b.gr);
    let mut serial = Vec::new();
    let mut scan = Vec::new();
    // Alternate the arms so a stale-buffer artifact would show in both.
    for _ in 0..reps {
        serial.push(run_arm(gpu, &b, false)?);
        scan.push(run_arm(gpu, &b, true)?);
        t.launches += 2;
    }
    let (b_slots, b_gr) = (b.slots, b.gr);
    b.free(gpu)?;
    t.routes += 1;

    // Order independent outputs: byte-identical everywhere, equal to the model.
    for (i, want) in [(0usize, &m.counts), (1, &m.offsets), (3, &m.tiles)] {
        for (arm, outs) in [("serial", &serial), ("scan", &scan)] {
            for (r, out) in outs.iter().enumerate() {
                if &words(&out[i][..out[i].len() - GUARD]) != want {
                    t.fail(
                        CLASSES[i.min(2)],
                        format!(
                            "{label}: {arm} rep {r} {} differs from the model: {}",
                            NAMES[i],
                            first_diff(&words(&out[i][..out[i].len() - GUARD]), want)
                        ),
                    );
                }
            }
        }
        for r in 0..reps {
            if serial[r][i] != scan[r][i] {
                t.fail(
                    CLASSES[i.min(2)],
                    format!(
                        "{label}: rep {r} {} bytes differ scan vs serial: {}",
                        NAMES[i],
                        first_diff(&words(&scan[r][i]), &words(&serial[r][i]))
                    ),
                );
            }
        }
    }
    for outs in [&serial, &scan] {
        for out in outs.iter() {
            guards_intact(label, out, b_slots, b_gr, t);
        }
    }
    // Sorted / inverse: canonical comparison (the within-expert order is LDS arrival
    // order, not a contract): per-expert slot sets of the scan arm equal the serial
    // arm's, rep by rep. The model and inverse consistency are checked below.
    for (r, (s, c)) in serial.iter().zip(scan.iter()).enumerate() {
        let cs = canonical_sorted(&m, &words(&s[2][..s[2].len() - GUARD]));
        let cc = canonical_sorted(&m, &words(&c[2][..c[2].len() - GUARD]));
        if cs != cc {
            t.fail(
                "sorted",
                format!(
                    "{label}: rep {r} per-expert slot sets differ scan vs serial: {}",
                    first_diff(&cc, &cs)
                ),
            );
        }
    }
    // Informational strict byte comparison, only where the serial arm is itself byte-stable.
    let stable = serial
        .iter()
        .all(|o| o[2] == serial[0][2] && o[4] == serial[0][4]);
    if stable {
        t.strict_routes += 1;
        if scan
            .iter()
            .all(|o| o[2] == serial[0][2] && o[4] == serial[0][4])
        {
            t.strict_same += 1;
        } else {
            t.strict_differs += 1;
        }
    } else {
        t.order_nondeterministic += 1;
    }
    for out in serial.iter().chain(scan.iter()) {
        check_order(label, topk, &m, out, t);
    }
    Ok(())
}

/// Each expert's run of `sorted` put in ascending order (padding and tail untouched).
fn canonical_sorted(m: &Model, sorted: &[i32]) -> Vec<i32> {
    let mut canon = sorted.to_vec();
    for e in 0..E {
        let (lo, n) = (m.offsets[e] as usize, m.raw[e]);
        canon[lo..lo + n].sort_unstable();
    }
    canon
}

// ------------------------------------------------------------- downstream chain

const H: usize = 2560;
const MI: usize = 640;
/// MQ4G256V2 gate/up expert: [2 * 640 rows, K 2560 / 256 groups] x 136 B.
const GU_BYTES: usize = 2 * MI * (H / 256) * 136;
/// MQ4G128V2 down expert: [2560 rows, K 640 / 128 groups] x 68 B.
const DN_BYTES: usize = H * (MI / 128) * 68;
/// Distinct synthetic expert images; expert `e` reads image `e % IMAGES`.
const IMAGES: usize = 64;
const CHAIN_MAX_ROWS: usize = 8;

/// A finite F16 with exponent in `[lo, hi]` and random sign / mantissa.
fn half(r: &mut Rng, lo: usize, hi: usize) -> u16 {
    let sign = (r.below(2) as u16) << 15;
    let e = (lo + r.below(hi - lo + 1)) as u16;
    sign | (e << 10) | r.below(1024) as u16
}

/// One routed expert in its real layout: `rows * groups` groups of `group_bytes`, each
/// `pairs` finite F16 (scale, zero) headers and random 4-bit payload.
fn expert_bytes(
    rows: usize,
    groups: usize,
    group_bytes: usize,
    pairs: usize,
    seed: u64,
) -> Vec<u8> {
    let mut r = Rng(seed);
    let mut v = vec![0u8; rows * groups * group_bytes];
    for g in v.chunks_exact_mut(group_bytes) {
        let mut off = 0;
        for _ in 0..pairs {
            let sc = half(&mut r, 6, 10);
            let e = ((sc >> 10) & 31) as usize;
            let zp = half(&mut r, e + 3, e + 6);
            g[off..off + 2].copy_from_slice(&sc.to_le_bytes());
            g[off + 2..off + 4].copy_from_slice(&zp.to_le_bytes());
            off += 4;
        }
        for c in g[off..].chunks_mut(8) {
            let w = r.next().to_le_bytes();
            c.copy_from_slice(&w[..c.len()]);
        }
    }
    v
}

struct Chain {
    _gu: GpuTensor,
    _dn: GpuTensor,
    gu_tab: GpuTensor,
    dn_tab: GpuTensor,
    xn_rot: GpuTensor,
    counts: GpuTensor,
    offsets: GpuTensor,
    sorted: GpuTensor,
    tiles: GpuTensor,
    inv: GpuTensor,
    y_gu: GpuTensor,
    rot_b: GpuTensor,
    down_x: GpuTensor,
    ffn_p: GpuTensor,
}

fn raw_zeros(gpu: &mut Gpu, bytes: usize) -> Res<GpuTensor> {
    Ok(gpu.zeros(&[bytes], DType::Raw)?)
}

fn f32_zeros(gpu: &mut Gpu, n: usize) -> Res<GpuTensor> {
    Ok(gpu.zeros(&[n], DType::F32)?)
}

fn build_chain(gpu: &mut Gpu) -> Res<Chain> {
    let mut gu = Vec::with_capacity(IMAGES * GU_BYTES);
    let mut dn = Vec::with_capacity(IMAGES * DN_BYTES);
    for i in 0..IMAGES as u64 {
        gu.extend(expert_bytes(2 * MI, H / 256, 136, 2, 0x6d71_3401 + 977 * i));
        dn.extend(expert_bytes(H, MI / 128, 68, 1, 0x6d71_3402 + 977 * i));
    }
    let gu_t = gpu.upload_raw(&gu, &[gu.len()])?;
    let dn_t = gpu.upload_raw(&dn, &[dn.len()])?;
    let table = |base: u64, stride: usize| -> Vec<u8> {
        (0..E)
            .flat_map(|e| (base + ((e % IMAGES) * stride) as u64).to_le_bytes())
            .collect()
    };
    let gtab = table(gu_t.buf.as_ptr() as u64, GU_BYTES);
    let dtab = table(dn_t.buf.as_ptr() as u64, DN_BYTES);
    let slots = CHAIN_MAX_ROWS * K;
    let gr = grouped_rows_bound(slots);
    let c = Chain {
        gu_tab: gpu.upload_raw(&gtab, &[gtab.len()])?,
        dn_tab: gpu.upload_raw(&dtab, &[dtab.len()])?,
        _gu: gu_t,
        _dn: dn_t,
        xn_rot: f32_zeros(gpu, CHAIN_MAX_ROWS * H)?,
        counts: raw_zeros(gpu, 4 * E)?,
        offsets: raw_zeros(gpu, 4 * (E + 1))?,
        sorted: raw_zeros(gpu, 4 * gr)?,
        tiles: raw_zeros(gpu, 4 * (gr / BLOCK_M))?,
        inv: raw_zeros(gpu, 4 * slots)?,
        y_gu: f32_zeros(gpu, gr * 2 * MI)?,
        rot_b: f32_zeros(gpu, slots * MI)?,
        down_x: f32_zeros(gpu, slots * H)?,
        ffn_p: f32_zeros(gpu, CHAIN_MAX_ROWS * H)?,
    };
    gpu.hip.device_synchronize()?;
    Ok(c)
}

impl Chain {
    fn free(self, gpu: &mut Gpu) -> Res<()> {
        for t in [
            self._gu,
            self._dn,
            self.gu_tab,
            self.dn_tab,
            self.xn_rot,
            self.counts,
            self.offsets,
            self.sorted,
            self.tiles,
            self.inv,
            self.y_gu,
            self.rot_b,
            self.down_x,
            self.ffn_p,
        ] {
            gpu.free_tensor(t)?;
        }
        Ok(())
    }
}

fn upload_i32(gpu: &Gpu, v: &[i32]) -> Res<GpuTensor> {
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    Ok(gpu.upload_raw(&bytes, &[bytes.len()])?)
}

/// Run the real chain once with the scatter arm `scan`; returns the unscattered
/// activations, the expanded down rows and the combine output, as bytes.
#[allow(clippy::too_many_arguments)]
fn run_chain(
    gpu: &mut Gpu,
    c: &Chain,
    ids: &GpuTensor,
    xn: &GpuTensor,
    w: &GpuTensor,
    n: usize,
    scan: bool,
) -> Res<[Vec<u8>; 3]> {
    let slots = n * K;
    let gr = grouped_rows_bound(slots);
    set_scan(gpu, scan);
    let poison = |t: &GpuTensor, bytes: usize| -> Res<()> {
        gpu.hip.memcpy_htod(&t.buf, &vec![POISON; bytes])?;
        Ok(())
    };
    poison(&c.y_gu, gr * 2 * MI * 4)?;
    poison(&c.rot_b, slots * MI * 4)?;
    poison(&c.down_x, slots * H * 4)?;
    gpu.hip.memcpy_htod(&c.ffn_p.buf, &vec![0u8; n * H * 4])?;
    gpu.hip.device_synchronize()?;
    gpu.rotate_x_mq_batched(xn, &c.xn_rot, H, n)?;
    gpu.moe_scatter_fused_top10(
        ids, &c.counts, &c.offsets, &c.sorted, &c.tiles, &c.inv, slots, E, gr, BLOCK_M,
    )?;
    gpu.gemm_mq4g256v2_moe_grouped_top10(
        &c.gu_tab,
        &c.tiles,
        &c.sorted,
        &c.xn_rot,
        &c.y_gu,
        2 * MI,
        H,
        K,
        gr,
        n,
    )?;
    gpu.moe_gate_up_unscatter_silu_rotate128_top10(
        &c.y_gu, &c.sorted, &c.rot_b, MI, gr, true, None,
    )?;
    gpu.gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded(
        &c.dn_tab, ids, &c.rot_b, &c.down_x, H, MI, n, E, K,
    )?;
    gpu.moe_down_combine_top10_batched(&c.down_x, ids, w, &c.ffn_p, H, n, E)?;
    gpu.hip.device_synchronize()?;
    let mut out: [Vec<u8>; 3] = Default::default();
    for (o, (t, bytes)) in out.iter_mut().zip([
        (&c.rot_b, slots * MI * 4),
        (&c.down_x, slots * H * 4),
        (&c.ffn_p, n * H * 4),
    ]) {
        let mut v = vec![0u8; bytes];
        gpu.hip.memcpy_dtoh(&mut v, &t.buf)?;
        *o = v;
    }
    Ok(out)
}

const CHAIN_NAMES: [&str; 3] = [
    "unscattered activations (rot_b)",
    "down_expanded rows",
    "combine output",
];

fn check_chain(
    gpu: &mut Gpu,
    c: &Chain,
    label: &str,
    topk: &[i32],
    seed: u64,
    reps: usize,
    t: &mut Tally,
) -> Res<()> {
    let n = topk.len() / K;
    let mut r = Rng(seed ^ 0xC4A1_0000);
    let x: Vec<f32> = (0..n * H).map(|_| (r.unit() as f32 - 0.5) * 2.0).collect();
    let wv: Vec<f32> = (0..n * K).map(|_| 0.02 + r.unit() as f32 * 0.18).collect();
    let ids = upload_i32(gpu, topk)?;
    let xn = gpu.upload_f32(&x, &[x.len()])?;
    let w = gpu.upload_f32(&wv, &[wv.len()])?;
    let mut serial = Vec::new();
    let mut scan = Vec::new();
    for _ in 0..reps {
        serial.push(run_chain(gpu, c, &ids, &xn, &w, n, false)?);
        scan.push(run_chain(gpu, c, &ids, &xn, &w, n, true)?);
        t.launches += 2;
    }
    gpu.free_tensor(ids)?;
    gpu.free_tensor(xn)?;
    gpu.free_tensor(w)?;
    t.chain_routes += 1;
    for rep in 0..reps {
        for k in 0..3 {
            let bad = scan[rep][k]
                .iter()
                .zip(&serial[rep][k])
                .filter(|(a, b)| a != b)
                .count();
            if bad != 0 {
                t.chain_bad_bytes += bad;
                t.fail(
                    "chain",
                    format!(
                        "{label}: rep {rep} {} differs scan vs serial in {bad} bytes: {}",
                        CHAIN_NAMES[k],
                        first_diff(&words(&scan[rep][k]), &words(&serial[rep][k]))
                    ),
                );
            }
            t.chain_serial_unstable_bytes += serial[rep][k]
                .iter()
                .zip(&serial[0][k])
                .filter(|(a, b)| a != b)
                .count();
        }
    }
    // The comparison must not be vacuous: the unscatter wrote every live slot, the
    // combine produced something.
    if first_poison_word(&serial[0][0], n * K * MI).is_some() {
        t.fail(
            "chain",
            format!("{label}: unscattered activations still hold poison"),
        );
    }
    if serial[0][2].iter().all(|&b| b == 0) {
        t.fail(
            "chain",
            format!("{label}: combine output is all zero (vacuous comparison)"),
        );
    }
    Ok(())
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

/// Microseconds per launch, eager and graph, of both arms on `topk`.
fn time_route(gpu: &mut Gpu, topk: &[i32], replays: usize, rounds: usize) -> Res<[[f64; 2]; 2]> {
    let b = bufs(gpu, topk)?;
    gpu.ensure_capture_stream()?;
    let (e0, e1) = (gpu.hip.event_create()?, gpu.hip.event_create()?);
    // [arm][eager, graph]; arm 0 = serial, arm 1 = scan.
    let mut samples: [[Vec<f64>; 2]; 2] = Default::default();
    for arm in 0..2 {
        set_scan(gpu, arm == 1);
        b.launch(gpu)?; // warm: compile and load
    }
    gpu.hip.device_synchronize()?;
    // One graph per arm: `replays` launches of the product launcher captured.
    let mut graphs = Vec::new();
    for arm in 0..2 {
        set_scan(gpu, arm == 1);
        gpu.hip.device_synchronize()?;
        gpu.begin_stream_capture()?;
        let mut captured = Ok(());
        for _ in 0..replays {
            captured = b.launch(gpu);
            if captured.is_err() {
                break;
            }
        }
        let g = gpu.end_stream_capture()?;
        captured?;
        let exec = gpu.hip.graph_instantiate(&g)?;
        graphs.push((g, exec));
    }
    for round in 0..rounds {
        let order: [usize; 2] = if round % 2 == 0 { [0, 1] } else { [1, 0] };
        for arm in order {
            set_scan(gpu, arm == 1);
            // Eager: `replays` launches between two events on the launch stream.
            gpu.hip.event_record(&e0, gpu.active_stream.as_ref())?;
            for _ in 0..replays {
                b.launch(gpu)?;
            }
            gpu.hip.event_record(&e1, gpu.active_stream.as_ref())?;
            gpu.hip.event_synchronize(&e1)?;
            samples[arm][0].push(gpu.hip.event_elapsed_ms(&e0, &e1)? as f64 * 1e3 / replays as f64);
            // Graph: the same launches, one submission.
            gpu.hip.event_record(&e0, gpu.active_stream.as_ref())?;
            gpu.launch_graph(&graphs[arm].1)?;
            gpu.hip.event_record(&e1, gpu.active_stream.as_ref())?;
            gpu.hip.event_synchronize(&e1)?;
            samples[arm][1].push(gpu.hip.event_elapsed_ms(&e0, &e1)? as f64 * 1e3 / replays as f64);
        }
    }
    for (g, exec) in graphs {
        gpu.hip.graph_exec_destroy(exec)?;
        gpu.hip.graph_destroy(g)?;
    }
    gpu.hip.device_synchronize()?;
    b.free(gpu)?;
    let mut out = [[0.0; 2]; 2];
    for arm in 0..2 {
        for mode in 0..2 {
            out[arm][mode] = median(&mut samples[arm][mode]);
        }
    }
    Ok(out)
}

fn arg<T: std::str::FromStr>(args: &[String], name: &str, default: T) -> Res<T> {
    match args.iter().position(|a| a == name) {
        None => Ok(default),
        Some(i) => args
            .get(i + 1)
            .and_then(|v| v.parse().ok())
            .ok_or_else(|| format!("{name} needs a value").into()),
    }
}

/// First differing word of two word vectors, with both values.
fn first_diff(got: &[i32], want: &[i32]) -> String {
    match got.iter().zip(want).position(|(g, w)| g != w) {
        Some(at) => format!(
            "first word {at}: got {} ({:#010x}), want {} ({:#010x}); {} vs {} words",
            got[at],
            got[at],
            want[at],
            want[at],
            got.len(),
            want.len()
        ),
        None => format!("same prefix, {} vs {} words", got.len(), want.len()),
    }
}

fn main() -> Res<()> {
    let args: Vec<String> = std::env::args().collect();
    // `--quick`: a smoke run (2 seeds, 2 reps, 3 timing rounds of 50 replays); every
    // explicit `--seeds/--reps/--replays/--rounds` still wins.
    let quick = args.iter().any(|a| a == "--quick");
    let seeds: u64 = arg(&args, "--seeds", if quick { 2 } else { 24 })?;
    let reps: usize = arg(&args, "--reps", if quick { 2 } else { 3 })?;
    let replays: usize = arg(&args, "--replays", if quick { 50 } else { 200 })?;
    let rounds: usize = arg(&args, "--rounds", if quick { 3 } else { 7 })?;
    let time = !args.iter().any(|a| a == "--no-time");
    let chain = !args.iter().any(|a| a == "--no-chain");
    if reps == 0 || replays == 0 || rounds == 0 || seeds == 0 {
        return Err("--seeds, --reps, --replays and --rounds must be > 0".into());
    }
    let mut gpu = Gpu::init()?;
    if gpu.arch != "gfx1201" {
        return Err(format!(
            "the scan-vs-serial arms differ only on exact gfx1201 (device is {})",
            gpu.arch
        )
        .into());
    }
    let saved = gpu.flags.qwen4_moe_scatter_scan;
    println!(
        "moe_scatter_scan_oracle: arch {}, flag default {}, seeds {seeds}, reps {reps}",
        gpu.arch, saved
    );
    let mut tally = Tally::default();
    let row_counts = [1usize, 2, 3, 4, 5, 6, 7, 8, 16, 33, 64, 300, 512];
    for &kind in &KINDS {
        // Seed-less kinds are deterministic routes: one seed is enough.
        let kind_seeds = match kind {
            Kind::FirstTen | Kind::ExactTiles => 1,
            _ => seeds,
        };
        tally.kind = format!("{kind:?}");
        let before = tally.total;
        let routes_before = tally.routes;
        for &rows in &row_counts {
            for seed in 0..kind_seeds {
                let topk = route(kind, rows, 0x5CA7_0000 + seed);
                let label = format!("{kind:?} rows={rows} seed={seed}");
                check_route(&mut gpu, &label, &topk, reps, &mut tally)?;
            }
        }
        println!(
            "  {kind:?}: {} routes, {} failures",
            tally.routes - routes_before,
            tally.total - before
        );
    }
    if chain {
        println!("downstream chain: synthetic MQ4 experts ({IMAGES} images), rows 1..=8, {seeds} seeds x {reps} reps");
        let c = build_chain(&mut gpu)?;
        for kind in [Kind::Uniform, Kind::Zipf, Kind::SameRow] {
            tally.kind = format!("chain {kind:?}");
            let before = tally.total;
            let routes_before = tally.chain_routes;
            for rows in 1..=CHAIN_MAX_ROWS {
                for seed in 0..seeds {
                    let topk = route(kind, rows, 0x5CA7_C000 + seed);
                    let label = format!("chain {kind:?} rows={rows} seed={seed}");
                    check_chain(&mut gpu, &c, &label, &topk, seed, reps, &mut tally)?;
                }
            }
            println!(
                "  chain {kind:?}: {} routes, {} failures",
                tally.chain_routes - routes_before,
                tally.total - before
            );
        }
        c.free(&mut gpu)?;
    }
    set_scan(&mut gpu, saved);
    println!(
        "routes {}, launches {}, order-nondeterministic routes (canonical check) {}",
        tally.routes, tally.launches, tally.order_nondeterministic
    );
    println!(
        "strict sorted/inverse byte comparison (informational, serial byte-stable routes {}): \
         scan bytes equal {}, within-expert order differs {}",
        tally.strict_routes, tally.strict_same, tally.strict_differs
    );
    println!(
        "chain_routes {}, chain_bad_bytes {}, chain_serial_unstable_bytes {}",
        tally.chain_routes, tally.chain_bad_bytes, tally.chain_serial_unstable_bytes
    );

    if time {
        println!("timing (us per launch, median of {rounds} alternating rounds, {replays} replays between events):");
        println!(
            "  rows  slots   eager serial   eager scan   graph serial   graph scan   graph delta"
        );
        for rows in [1usize, 2, 4, 8, 64, 512] {
            for kind in [Kind::Uniform, Kind::Zipf] {
                let topk = route(kind, rows, 0x71E0);
                let t = time_route(&mut gpu, &topk, replays, rounds)?;
                println!(
                    "  {rows:4} {:6}   {:>12.2} {:>12.2} {:>14.2} {:>12.2}   {:>+8.2} ({kind:?})",
                    topk.len(),
                    t[0][0],
                    t[1][0],
                    t[0][1],
                    t[1][1],
                    t[1][1] - t[0][1]
                );
            }
        }
        set_scan(&mut gpu, saved);
    }

    tally.report();
    if tally.total == 0 {
        println!("PASS");
        Ok(())
    } else {
        Err(format!("{} failing checks", tally.total).into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn le(words: &[i32]) -> Vec<u8> {
        words.iter().flat_map(|w| w.to_le_bytes()).collect()
    }

    /// Legitimate words containing the byte 0x5A (tile id 90, count 0x5A00, ...)
    /// must not look like poison; the poison word itself must.
    #[test]
    fn poison_check_is_word_level() {
        let legit = le(&[90, 0x5A00, 0x005A_0000, 0x5A5A, -1, 0x5A5A_5A00]);
        assert_eq!(first_poison_word(&legit, 6), None);
        let bad = le(&[90, 3, POISON_WORD, 7]);
        assert_eq!(first_poison_word(&bad, 4), Some(2));
        // Words past the live count are not checked.
        assert_eq!(first_poison_word(&bad, 2), None);
        assert_eq!(POISON_WORD, 0x5A5A_5A5A);
    }

    /// The host model against hand-computed layouts.
    #[test]
    fn model_matches_a_hand_layout() {
        // One row to experts 0..9: each expert holds 1 slot, padded to 16.
        let topk: Vec<i32> = (0..K as i32).collect();
        let gr = grouped_rows_bound(topk.len());
        let m = model(&topk, gr);
        for e in 0..K {
            assert_eq!(m.counts[e], 16);
            assert_eq!(m.offsets[e], 16 * e as i32);
            assert_eq!(m.sorted[16 * e], e as i32);
            assert_eq!(m.tiles[e], e as i32);
        }
        assert_eq!(m.offsets[K], 16 * K as i32);
        assert_eq!(m.counts[K], 0);
        // 10 slots bound to 160 grouped rows = 10 tiles, all live.
        assert_eq!(m.tiles.len(), K);
        assert_eq!(m.sorted.len(), gr);
        // A bad id is unrouted and takes no space.
        let mut bad = topk.clone();
        bad[3] = -1;
        assert_eq!(model(&bad, gr).counts[3], 0);
    }

    /// Every route generator yields `rows * K` slots; distinct-per-row kinds
    /// never repeat an expert within a row, and bad-id routes contain bad ids.
    #[test]
    fn routes_have_the_advertised_shape() {
        for &kind in &KINDS {
            for rows in [1usize, 8, 33] {
                let v = route(kind, rows, 7);
                assert_eq!(v.len(), rows * K, "{kind:?} rows={rows}");
                if !matches!(kind, Kind::BadIds) {
                    assert!(v.iter().all(|&id| is_valid(id)), "{kind:?}");
                    for row in v.chunks(K) {
                        let mut r = row.to_vec();
                        r.sort_unstable();
                        r.dedup();
                        assert_eq!(r.len(), K, "{kind:?} repeats an expert in a row");
                    }
                }
            }
        }
        assert!(route(Kind::BadIds, 33, 7).iter().any(|&id| !is_valid(id)));
    }

    /// Failures are counted per class and per kind; the first of each class keeps its detail.
    #[test]
    fn tally_reports_per_class_and_keeps_the_first_detail() {
        let mut t = Tally::default();
        t.kind = "Uniform".into();
        t.fail("tiles", "first".into());
        t.fail("tiles", "second".into());
        t.kind = "Zipf".into();
        t.fail("guard", "g".into());
        assert_eq!(t.total, 3);
        assert_eq!(t.per_class["tiles"], 2);
        assert_eq!(t.first["tiles"], "first");
        assert_eq!(t.per_kind["Zipf"]["guard"], 1);
        assert!(!t.per_kind["Uniform"].contains_key("guard"));
    }

    /// Two within-expert orders of the same multiset are canonically equal; a
    /// different slot set is not; padding stays in place.
    #[test]
    fn canonical_sorted_ignores_within_expert_order_only() {
        let topk: Vec<i32> = vec![5, 5, 5, 7, 7, 5];
        let gr = grouped_rows_bound(topk.len());
        let m = model(&topk, gr);
        let (o5, o7) = (m.offsets[5] as usize, m.offsets[7] as usize);
        let mut a = vec![-1i32; gr];
        let mut b = a.clone();
        a[o5..o5 + 4].copy_from_slice(&[0, 1, 2, 5]);
        b[o5..o5 + 4].copy_from_slice(&[5, 2, 0, 1]);
        a[o7..o7 + 2].copy_from_slice(&[3, 4]);
        b[o7..o7 + 2].copy_from_slice(&[4, 3]);
        assert_eq!(canonical_sorted(&m, &a), canonical_sorted(&m, &b));
        assert_eq!(canonical_sorted(&m, &a), m.sorted);
        b[o7] = 0; // a slot of another expert: the set differs
        assert_ne!(canonical_sorted(&m, &a), canonical_sorted(&m, &b));
    }

    /// The synthetic experts are finite F16 headers in the real layout sizes.
    #[test]
    fn synthetic_experts_have_the_real_layout_and_finite_headers() {
        let gu = expert_bytes(2 * MI, H / 256, 136, 2, 11);
        let dn = expert_bytes(H, MI / 128, 68, 1, 12);
        assert_eq!((gu.len(), dn.len()), (GU_BYTES, DN_BYTES));
        for (bytes, group, pairs) in [(&gu, 136usize, 2usize), (&dn, 68, 1)] {
            for g in bytes.chunks_exact(group) {
                for p in 0..2 * pairs {
                    let h = u16::from_le_bytes([g[2 * p], g[2 * p + 1]]);
                    assert_ne!((h >> 10) & 31, 31, "non-finite F16 header");
                }
            }
        }
    }
}
