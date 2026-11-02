// SPDX-License-Identifier: Apache-2.0
//! Raw PM dense-row K1 byte oracle; never participates in runtime dispatch.
//! Build: cargo build -p hipfire-runtime --release --features lab --example mq_i4_rows_byte_oracle
//! Run on the locked expected card: target/release/examples/mq_i4_rows_byte_oracle --codeobject /absolute/path/module.co
//! Every mismatch is fatal, including signed-zero and nonfinite bit differences.
use hip_bridge::{Function, HipRuntime};
use hipfire_runtime::{
    mq_i4::{self, MqI4Codec as Codec, MqI4Profile as Profile, MqI4Rotation as Rotation},
    mq_i4_reference,
};
use serde_json::{json, Value};

type Result<T, E = String> = std::result::Result<T, E>;
fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}
const UUID: &str = "GPU-6109a4cb5f833235";
const BDF: &str = "0000:9a:00.0";
const ARCH: &str = "gfx1201";

struct Case {
    name: String,
    m: usize,
    k: usize,
    codes: Vec<u8>,
    scales: Vec<u16>,
    x: Vec<f32>,
}
fn random_case(name: &str, p: Profile, m: usize, k: usize) -> Case {
    // Fixed integer PRNG, independent of platform and external RNG versions.
    let mut state = 0x4b31_1201u32;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        state
    };
    let codes = (0..m * k)
        .map(|_| (next() as u8) & p.codec.levels())
        .collect();
    let scales = (0..m * (k / 128))
        .map(|_| [0x3000, 0x3800, 0x3c00, 0x4000][next() as usize % 4])
        .collect();
    let x = (0..k)
        .map(|_| ((next() % 8193) as i32 - 4096) as f32 / 1024.0)
        .collect();
    Case {
        name: name.into(),
        m,
        k,
        codes,
        scales,
        x,
    }
}
fn cases(p: Profile) -> Vec<Case> {
    let r = p.rotation.width();
    let (m, k, name) = if r == 256 {
        (1280, 2560, "fn_gate_up")
    } else {
        (2560, 640, "fn_down")
    };
    let mut out = vec![
        random_case(name, p, m, k),
        random_case("small_k256", p, 3, 256),
    ];
    if r == 128 {
        out.push(random_case("small_k128", p, 3, 128));
    }
    // Isolate each of the four code positions in each of the 32 lanes,
    // for every legal value. All other positions retain midpoint codes.
    let m = 128 * (p.codec.levels() as usize + 1);
    let mut c = random_case("exhaustive_codes_every_lane_position", p, m, r);
    c.codes.fill(p.codec.midpoint());
    c.scales.fill(0x3c00);
    for pos in 0..128 {
        for code in 0..=p.codec.levels() {
            let row = pos * (p.codec.levels() as usize + 1) + code as usize;
            for group in 0..r / 128 {
                c.codes[row * r + group * 128 + pos] = code;
            }
        }
    }
    out.push(c);
    let mut c = random_case("zero_scales_midpoint_codes", p, 3, r);
    c.codes.fill(p.codec.midpoint());
    c.scales.fill(0);
    out.push(c);
    for (name, scale) in [("f16_max65504", 0x7bff), ("f16_smallest_subnormal", 0x0001)] {
        let mut c = random_case(name, p, 3, r);
        c.scales.fill(scale);
        // Finite even for max scale, while preserving very small products.
        for (i, x) in c.x.iter_mut().enumerate() {
            *x = [0.0, -0.0, 1.0, -1.0, 1.0 / 1048576.0, -1.0 / 1048576.0][i % 6];
        }
        out.push(c);
    }
    let mut c = random_case("signed_zero_large_small_x", p, 3, r);
    c.scales.fill(0x3c00);
    for (i, x) in c.x.iter_mut().enumerate() {
        *x = [0.0, -0.0, 16777216.0, -16777216.0, 1.0e-30, -1.0e-30][i % 6];
    }
    out.push(c);
    let mut c = random_case(
        "catastrophic_cancellation",
        p,
        3,
        if r == 256 { 1024 } else { 640 },
    );
    // Constant +1 weights: each lane sees [2^24, 1, -2^24, 1].
    // Group chains and the shuffle tree must retain the frozen rounding DAG.
    c.codes.fill(p.codec.midpoint());
    c.scales.fill(0x3c00);
    for (i, x) in c.x.iter_mut().enumerate() {
        *x = [16777216.0, 1.0, -16777216.0, 1.0][i % 4];
    }
    out.push(c);
    out
}
fn validate(p: Profile, c: &Case) -> Result<()> {
    if c.m == 0 || c.k == 0 || c.k % 128 != 0 || c.k % p.rotation.width() != 0 {
        return Err("MQ-I4 rows requires M>0, K>0, K%128=0 and K%R=0".into());
    }
    let n = c.m.checked_mul(c.k).ok_or("MQ-I4 rows span overflow")?;
    // Conservatively bound every intermediate byte/product index to u32.
    let bits = n
        .checked_mul(p.codec.bits())
        .ok_or("MQ-I4 rows span overflow")?;
    let total = mq_i4::tensor_bytes(p, c.m, c.k).map_err(err)?;
    if [
        c.m,
        c.k,
        n,
        bits,
        total,
        c.k.checked_mul(4).ok_or("x span overflow")?,
        c.m.checked_mul(4).ok_or("y span overflow")?,
    ]
    .iter()
    .any(|&v| v > u32::MAX as usize)
    {
        return Err("MQ-I4 rows span exceeds u32".into());
    }
    if c.x.len() != c.k {
        return Err("MQ-I4 rows x length mismatch".into());
    }
    Ok(())
}
fn run_case(hip: &HipRuntime, f: &Function, p: Profile, c: &Case) -> Result<Value> {
    validate(p, c)?;
    let bytes = mq_i4::encode_tensor(p, c.m, c.k, &c.codes, &c.scales).map_err(err)?;
    let twin = mq_i4_reference::decode_gemv_f32(p, c.m, c.k, &bytes, &c.x);
    let xbytes: Vec<u8> = c.x.iter().flat_map(|x| x.to_le_bytes()).collect();
    let codes = hip.malloc(bytes.len()).map_err(err)?;
    let x = hip.malloc(xbytes.len()).map_err(err)?;
    let y = hip.malloc(c.m * 4).map_err(err)?;
    hip.memcpy_htod(&codes, &bytes).map_err(err)?;
    hip.memcpy_htod(&x, &xbytes).map_err(err)?;
    hip.memset(&y, 0xa5, c.m * 4).map_err(err)?;
    let mut args = [0u8; 32];
    for (off, ptr) in [(0, codes.as_ptr()), (8, x.as_ptr()), (16, y.as_ptr())] {
        args[off..off + 8].copy_from_slice(&(ptr as u64).to_le_bytes());
    }
    args[24..28].copy_from_slice(&(c.m as u32).to_le_bytes());
    args[28..32].copy_from_slice(&(c.k as u32).to_le_bytes());
    // SAFETY: exact 32-byte ABI; all allocations remain live through sync.
    unsafe { hip.launch_kernel_blob(f, [c.m as u32, 1, 1], [32, 1, 1], 0, None, &mut args) }
        .map_err(err)?;
    hip.device_synchronize().map_err(err)?;
    let mut gpu = vec![0u8; c.m * 4];
    hip.memcpy_dtoh(&mut gpu, &y).map_err(err)?;
    let mut mismatches = 0usize;
    let mut first = None;
    for (index, (b, t)) in gpu.chunks_exact(4).zip(&twin).enumerate() {
        let bits = u32::from_le_bytes(b.try_into().unwrap());
        if bits != t.to_bits() {
            mismatches += 1;
            if first.is_none() {
                first = Some(
                    json!({"index":index,"gpu_bits":format!("0x{bits:08x}"),"twin_bits":format!("0x{:08x}",t.to_bits())}),
                );
            }
        }
    }
    Ok(json!({"case":c.name,"m":c.m,"k":c.k,"mismatch_count":mismatches,"first":first}))
}
fn run() -> Result<()> {
    let mut args = std::env::args().skip(1);
    if args.next().as_deref() != Some("--codeobject") {
        return Err("usage: mq_i4_rows_byte_oracle --codeobject PATH".into());
    }
    let path = args
        .next()
        .ok_or("usage: mq_i4_rows_byte_oracle --codeobject PATH")?;
    if args.next().is_some() {
        return Err("usage: mq_i4_rows_byte_oracle --codeobject PATH".into());
    }
    let hip = HipRuntime::load().map_err(err)?;
    let id = hip.current_device().map_err(err)?;
    let uuid = hip.device_uuid(id).map_err(err)?;
    let bdf = hip.device_pci_bus_id(id).map_err(err)?.to_ascii_lowercase();
    let arch = hip.get_arch(id).map_err(err)?;
    let device = json!({"uuid":uuid,"bdf":bdf,"arch":arch,"expected_uuid":UUID,"expected_bdf":BDF,"expected_arch":ARCH,"identity_match":uuid==UUID && bdf==BDF && arch==ARCH});
    if uuid != UUID || bdf != BDF || arch != ARCH {
        println!(
            "{}",
            json!({"schema":"mq-i4-rows-byte-oracle-v1","device":device,"error":"MQ-I4 rows device identity mismatch"})
        );
        return Err("MQ-I4 rows device identity mismatch".into());
    }
    let bytes = std::fs::read(&path).map_err(err)?;
    let module = hip.module_load_data(&bytes).map_err(err)?;
    let mut results = Vec::new();
    let mut total = 0u64;
    for (codec, tag) in [(Codec::W3, "w3"), (Codec::W2, "w2")] {
        for rotation in [Rotation::R256, Rotation::R128] {
            let p = Profile { codec, rotation };
            let symbol = format!("mq_i4_rows_{tag}_r{}", rotation.width());
            let f = hip.module_get_function(&module, &symbol).map_err(err)?;
            let mut rows = Vec::new();
            for c in cases(p) {
                let row = run_case(&hip, &f, p, &c)?;
                total += row["mismatch_count"].as_u64().unwrap();
                rows.push(row);
            }
            results.push(json!({"symbol":symbol,"cases":rows}));
        }
    }
    println!(
        "{}",
        json!({"schema":"mq-i4-rows-byte-oracle-v1","module":"mq_i4_rows_pm_gfx1201","codeobject":path,"device":device,"kernarg_bytes":32,"symbols":results,"mismatch_count":total,"pass":total==0})
    );
    if total != 0 {
        return Err("MQ-I4 rows byte oracle mismatch".into());
    }
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
