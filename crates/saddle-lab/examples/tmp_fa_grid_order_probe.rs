// SPDX-License-Identifier: Apache-2.0
// Raster3 untracked FA grid-order timing probe (device-resident, timing only).
//
// Builds one prefill-8192-scale packet-FA problem (batch 8192, ctx 8192,
// H24/KV4/D256, fp8 KV, contiguous positions) and launches the production
// `attention_fp8_e4m3_fa2_gqa_packet_gfx1201` launcher in a loop. The grid
// order comes from HIPFIRE_FA_GRID_ORDER (same binary for every order).
// Numerical output is meaningless; only device time is measured.
//
// Usage:
//   HIPFIRE_FA_GRID_ORDER=<0..4> kx3... no:
//   HIPFIRE_FA_GRID_ORDER=<0..4> ./target/release/examples/tmp_fa_grid_order_probe
// Under rocprofv3 --kernel-trace the per-order attention sum is exact.

use rdna_compute::{DType, Gpu};

const NH: usize = 24;
const NKV: usize = 4;
const HD: usize = 256;
const ROWB: usize = 1032;
const BATCH: usize = 8192;
const CTX: usize = 8192;
const WARMUP: usize = 5;
const SAMPLES: usize = 20;
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0 = self.0.wrapping_mul(0x2545_f491_4f6c_dd1d);
        self.0
    }
}

fn main() {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let mut gpu = Gpu::init().expect("gpu init");
    assert_eq!(gpu.arch, "gfx1201");

    // Q f32: batch * heads * dim random in [-4, 4].
    let qn = BATCH * NH * HD;
    let mut qh = vec![0f32; qn];
    for x in &mut qh {
        *x = ((rng.next() % 20001) as f32 / 10000.0 - 1.0) * 4.0;
    }
    let q = gpu.upload_f32(&qh, &[qn]).expect("upload Q");
    let out = gpu.zeros(&[qn], DType::F32).expect("out");

    // KV: random codes (avoid 0x7f NaN-ish code), deterministic headers.
    let kv_bytes = CTX * ROWB;
    let mut kh = vec![0x56u8; kv_bytes];
    let mut vh = vec![0x56u8; kv_bytes];
    for row in 0..CTX {
        let base = row * ROWB;
        for d in 0..1024 {
            let mut kc = rng.next() as u8;
            let mut vc = rng.next() as u8;
            if kc & 0x7f == 0x7f {
                kc ^= 1;
            }
            if vc & 0x7f == 0x7f {
                vc ^= 1;
            }
            kh[base + d] = kc;
            vh[base + d] = vc;
        }
        for h in 0..NKV {
            let ks = 0x3000u16 + ((row * 37 + h * 211) & 0x0fff) as u16;
            let vs = 0x2c00u16 + ((row * 53 + h * 173) & 0x0fff) as u16;
            kh[base + 1024 + 2 * h..base + 1026 + 2 * h].copy_from_slice(&ks.to_le_bytes());
            vh[base + 1024 + 2 * h..base + 1026 + 2 * h].copy_from_slice(&vs.to_le_bytes());
        }
    }
    let k = gpu.upload_raw(&kh, &[kv_bytes]).expect("upload K");
    let v = gpu.upload_raw(&vh, &[kv_bytes]).expect("upload V");

    // Contiguous causal positions.
    let ph: Vec<i32> = (0..BATCH as i32).collect();
    let pos = gpu
        .upload_raw(
            &ph.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>(),
            &[BATCH * 4],
        )
        .expect("upload pos");

    for _ in 0..WARMUP {
        gpu.attention_fp8_e4m3_fa2_gqa_packet_gfx1201(&q, &k, &v, &out, &pos, NH, NKV, HD, CTX, BATCH)
            .expect("packet launch");
    }
    gpu.hip.device_synchronize().expect("warmup sync");

    let mut pairs = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        let start = gpu.hip.event_create().expect("start");
        let stop = gpu.hip.event_create().expect("stop");
        gpu.hip.event_record(&start, gpu.active_stream.as_ref()).expect("rec start");
        gpu.attention_fp8_e4m3_fa2_gqa_packet_gfx1201(&q, &k, &v, &out, &pos, NH, NKV, HD, CTX, BATCH)
            .expect("packet launch");
        gpu.hip.event_record(&stop, gpu.active_stream.as_ref()).expect("rec stop");
        pairs.push((start, stop));
    }
    gpu.hip.event_synchronize(&pairs.last().unwrap().1).expect("sync");
    let mut us: Vec<f64> = pairs
        .iter()
        .map(|(s, e)| gpu.hip.event_elapsed_ms(s, e).expect("elapsed") as f64 * 1000.0)
        .collect();
    for (s, e) in pairs {
        gpu.hip.event_destroy(s).expect("d start");
        gpu.hip.event_destroy(e).expect("d stop");
    }
    us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = (us[SAMPLES / 2 - 1] + us[SAMPLES / 2]) * 0.5;
    println!("RESULT,fa_packet_prefill8192,{med:.3}");
}
