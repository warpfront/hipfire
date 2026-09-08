//! Bare-PM4 A/B of `gemv_mq4g256v2` code objects: no HIP runtime in the loop.
//!
//! `poc_gemv_ab <elf-a> <elf-b> [MxK,...] [render-node]` loads both ELFs,
//! builds one IB with N back-to-back dispatches (each followed by the
//! redline compute barrier, like a retained tape), and reports GPU-side
//! microseconds per dispatch as (t(N) - t(1)) / (N - 1) over the
//! submit-and-wait wall clock, plus a bit-compare of y between the two.
//!
//! Layout / cache-honest extensions (env, defaults preserve original CLI):
//! - `GEMV_LAYOUT_B=soa|aos` (default `aos`): arm A always AoS; B optional
//!   GLOBAL SoA planes (`payload=A+n*128`, `header=A+N*128+n*8`, `N=M*gpr`).
//! - `GEMV_WORKING_SET_MIB=N` (default `0`): slots =
//!   `max(1, ceil(N<<20 / weight_bytes))`; round-robin distinct weight
//!   allocations; iters rounded up to whole rotations; warm one full
//!   rotation set before timing.
//! - `GEMV_READ_ONLY=1`: RO checksum kernels + CPU RO oracle (uint32 bits
//!   through y; x unused).
//! - `GEMV_REVERSE=1`: reverse timed arm order (B then A).
//! - `GEMV_ITERS=N` (default 200), `GEMV_SEED` (default 1; per-slot
//!   `seed.wrapping_add(slot)`).
//! - Existing `GEMV_SYMBOL_{A,B}`, `GEMV_BLOCK_{A,B}`, `GEMV_GRID_DIV_{A,B}`.
//!
//! Timing: three raw samples each for t1 and tN (no best-of/min); median
//! used for marginal us/dispatch. Logical-byte rate is analytical only —
//! not a DRAM claim. Production barrier dwords are unchanged; never call
//! `CommandBuffer::barrier`.
//!
//! Warning: gfx1100 repeated-submission A/A warmup produced an illegal PM4
//! command / ring reset in this research probe, so its throughput is not
//! accepted evidence pending root cause; use the HIP probe for this layout
//! experiment.
use redline::device::{Device, GpuBuffer};
use redline::dispatch::{CommandBuffer, DispatchQueue, KernargBuilder, Kernel, LoadedModule};
use std::time::Instant;

fn xorshift(s: &mut u64) -> u64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    *s
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn env_u64(key: &str, default: u64) -> u64 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn env_flag(key: &str) -> bool {
    match std::env::var(key) {
        Ok(v) => matches!(
            v.as_str(),
            "1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON"
        ),
        Err(_) => false,
    }
}

fn fnv1a64(data: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

fn hex64(h: u64) -> String {
    format!("{h:016x}")
}

fn median3(mut v: [f64; 3]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v[1]
}

/// GLOBAL SoA: payload plane first (`N*128`), then header plane (`N*8`).
/// AoS group `n` at `A+n*136` (header +0, payload +8) → SoA payload `A+n*128`,
/// header `A+N*128+n*8`. Byte-preserving permutation; total still `136*N`.
fn aos_to_soa(aos: &[u8], m: usize, gpr: usize) -> Vec<u8> {
    let n_total = m * gpr;
    debug_assert_eq!(aos.len(), n_total * 136);
    let mut soa = vec![0u8; n_total * 136];
    let hdr_base = n_total * 128;
    for n in 0..n_total {
        let src = n * 136;
        soa[n * 128..n * 128 + 128].copy_from_slice(&aos[src + 8..src + 136]);
        let hd = hdr_base + n * 8;
        soa[hd..hd + 8].copy_from_slice(&aos[src..src + 8]);
    }
    soa
}

fn soa_to_aos(soa: &[u8], m: usize, gpr: usize) -> Vec<u8> {
    let n_total = m * gpr;
    debug_assert_eq!(soa.len(), n_total * 136);
    let mut aos = vec![0u8; n_total * 136];
    let hdr_base = n_total * 128;
    for n in 0..n_total {
        let dst = n * 136;
        let hd = hdr_base + n * 8;
        aos[dst..dst + 8].copy_from_slice(&soa[hd..hd + 8]);
        aos[dst + 8..dst + 136].copy_from_slice(&soa[n * 128..n * 128 + 128]);
    }
    aos
}

fn next_unit(s: &mut u64) -> f32 {
    // Uniform [0, 1) from xorshift high bits (2^-53 scale).
    const INV: f64 = 1.0 / 9_007_199_254_740_992.0;
    ((xorshift(s) >> 11) as f64 * INV) as f32
}

/// f32 → f16 bits, round-to-nearest-even (normal-range scales/zps only in practice).
fn f32_to_f16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7F_FFFF;
    if exp == 255 {
        return sign | 0x7C00 | if mant != 0 { 0x0200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 31 {
        return sign | 0x7C00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = mant | 0x80_0000;
        let shift = (14 - e) as u32;
        let low = m & ((1u32 << shift) - 1);
        let half = 1u32 << (shift - 1);
        let mut q = (m >> shift) as u16;
        if low > half || (low == half && (q & 1) == 1) {
            q += 1;
        }
        return sign | q;
    }
    let low = mant & 0x1FFF;
    let mut q = (mant >> 13) as u16;
    if low > 0x1000 || (low == 0x1000 && (q & 1) == 1) {
        q += 1;
        if q == 0x0400 {
            let e2 = e + 1;
            if e2 >= 31 {
                return sign | 0x7C00;
            }
            return sign | ((e2 as u16) << 10);
        }
    }
    sign | ((e as u16) << 10) | (q & 0x03FF)
}

fn gen_weights_aos(m: usize, gpr: usize, mut seed: u64) -> Vec<u8> {
    let n_total = m * gpr;
    let mut w = vec![0u8; n_total * 136];
    for g in 0..n_total {
        let o = g * 136;
        // Distinct finite f16 sc/zp per half (hA vs hB) and group/row so
        // header-half swaps and wrong-index bugs are visible to oracles.
        for h in 0..2 {
            let sc = 0.001 + next_unit(&mut seed) * 0.049;
            let zp = -0.4 + next_unit(&mut seed) * 0.4;
            let packed = f32_to_f16_bits(sc) as u32 | ((f32_to_f16_bits(zp) as u32) << 16);
            w[o + 4 * h..o + 4 * h + 4].copy_from_slice(&packed.to_le_bytes());
        }
        for c in w[o + 8..o + 136].chunks_mut(8) {
            c.copy_from_slice(&xorshift(&mut seed).to_le_bytes());
        }
    }
    w
}

fn gen_x(k: u32, mut seed: u64) -> Vec<f32> {
    (0..k)
        .map(|_| (xorshift(&mut seed) % 2000) as f32 / 1000.0 - 1.0)
        .collect()
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x03FF) as u32;
    let b = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            let mut e: i32 = 127 - 14;
            let mut m = mant;
            while (m & 0x0400) == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | ((e as u32) << 23) | ((m & 0x03FF) << 13)
        }
    } else if exp == 31 {
        sign | (0xFF << 23) | (mant << 13)
    } else {
        sign | ((exp + 112) << 23) | (mant << 13)
    };
    f32::from_bits(b)
}

/// Plain GEMV CPU 4-acc row oracle on AoS layout (DOG left-to-right, acc[g%4],
/// combine (a0+a1)+(a2+a3), shfl_down 16..1). Lab check for rows 0 / M-1;
/// full-shape plain correctness also gated by all-row A/B + parent HIP CPU.
fn cpu_gemv_row_aos(w: &[u8], x: &[f32], row: usize, k: usize) -> f32 {
    let gpr = k / 256;
    let quads = gpr >> 2;
    let tail = gpr & 3;
    let mut acc = [[0f32; 4]; 32];
    let dot_group = |gp: usize, g: usize, slot: usize, acc: &mut [[f32; 4]; 32]| {
        let ha = u32::from_le_bytes([w[gp], w[gp + 1], w[gp + 2], w[gp + 3]]);
        let hb = u32::from_le_bytes([w[gp + 4], w[gp + 5], w[gp + 6], w[gp + 7]]);
        for (tid, lane) in acc.iter_mut().enumerate() {
            let hs = if tid < 16 { ha } else { hb };
            let sc = f16_to_f32((hs & 0xFFFF) as u16);
            let zp = f16_to_f32((hs >> 16) as u16);
            let off = gp + 8 + tid * 4;
            let pk = u32::from_le_bytes([w[off], w[off + 1], w[off + 2], w[off + 3]]);
            let base = g * 256 + tid * 8;
            let mut sum = (sc * ((pk & 0xF) as f32) + zp) * x[base];
            for i in 1..8 {
                let term = (sc * (((pk >> (4 * i)) & 0xF) as f32) + zp) * x[base + i];
                sum += term;
            }
            lane[slot] += sum;
        }
    };
    for q in 0..quads {
        let g = q << 2;
        for s in 0..4 {
            let n = row * gpr + g + s;
            dot_group(n * 136, g + s, s, &mut acc);
        }
    }
    for t in 0..tail {
        let g = (quads << 2) + t;
        let n = row * gpr + g;
        // tail uses acc[g%4] == acc[t] for g = quads*4+t
        dot_group(n * 136, g, g % 4, &mut acc);
    }
    let mut lane = [0f32; 32];
    for (tid, lane_acc) in acc.iter().enumerate() {
        lane[tid] = (lane_acc[0] + lane_acc[1]) + (lane_acc[2] + lane_acc[3]);
    }
    for offset in [16usize, 8, 4, 2, 1] {
        let prev = lane;
        for tid in 0..32 {
            lane[tid] = if tid + offset < 32 {
                prev[tid] + prev[tid + offset]
            } else {
                prev[tid]
            };
        }
    }
    lane[0]
}

/// RO counterfactual checksum on AoS bytes (mirrors kernel contract):
/// per-lane `acc += pk` (wrapping); XOR-shfl reduce 16..1; lane0 adds
/// `header_acc = sum(hA + rotl32(hB, 16))` over groups.
fn cpu_ro_row_aos(w: &[u8], row: usize, gpr: usize) -> u32 {
    let mut acc = [0u32; 32];
    let mut header_acc = 0u32;
    for g in 0..gpr {
        let n = row * gpr + g;
        let gp = n * 136;
        let ha = u32::from_le_bytes([w[gp], w[gp + 1], w[gp + 2], w[gp + 3]]);
        let hb = u32::from_le_bytes([w[gp + 4], w[gp + 5], w[gp + 6], w[gp + 7]]);
        header_acc = header_acc.wrapping_add(ha.wrapping_add(hb.rotate_left(16)));
        for tid in 0..32 {
            let off = gp + 8 + tid * 4;
            let pk = u32::from_le_bytes([w[off], w[off + 1], w[off + 2], w[off + 3]]);
            acc[tid] = acc[tid].wrapping_add(pk);
        }
    }
    for offset in [16usize, 8, 4, 2, 1] {
        let prev = acc;
        for tid in 0..32 {
            acc[tid] = if tid + offset < 32 {
                prev[tid] ^ prev[tid + offset]
            } else {
                prev[tid]
            };
        }
    }
    acc[0].wrapping_add(header_acc)
}

fn cpu_ro_row_soa(soa: &[u8], row: usize, gpr: usize, m: usize) -> u32 {
    let n_total = m * gpr;
    let hdr_base = n_total * 128;
    let mut acc = [0u32; 32];
    let mut header_acc = 0u32;
    for g in 0..gpr {
        let n = row * gpr + g;
        let pay = n * 128;
        let hd = hdr_base + n * 8;
        let ha = u32::from_le_bytes([soa[hd], soa[hd + 1], soa[hd + 2], soa[hd + 3]]);
        let hb = u32::from_le_bytes([soa[hd + 4], soa[hd + 5], soa[hd + 6], soa[hd + 7]]);
        header_acc = header_acc.wrapping_add(ha.wrapping_add(hb.rotate_left(16)));
        for tid in 0..32 {
            let off = pay + tid * 4;
            let pk = u32::from_le_bytes([soa[off], soa[off + 1], soa[off + 2], soa[off + 3]]);
            acc[tid] = acc[tid].wrapping_add(pk);
        }
    }
    for offset in [16usize, 8, 4, 2, 1] {
        let prev = acc;
        for tid in 0..32 {
            acc[tid] = if tid + offset < 32 {
                prev[tid] ^ prev[tid + offset]
            } else {
                prev[tid]
            };
        }
    }
    acc[0].wrapping_add(header_acc)
}

struct SlotArm {
    w: GpuBuffer,
    y: GpuBuffer,
    ka: GpuBuffer,
}

fn push_prod_barrier(cb: &mut CommandBuffer) {
    // Production gfx12 tape inter-node barrier (redline-rocr pm4.rs).
    // EXACT dwords — do not call CommandBuffer::barrier (known ~100us trap).
    cb.push_raw(&[0xc000_4600, 0x407]);
    cb.push_raw(&[
        0xc006_5800,
        0,
        u32::MAX,
        0x00ff_ffff,
        0,
        0,
        0x0000_000a,
        0x10180,
    ]);
}

fn build_cb(
    kern: &Kernel,
    slots: &[SlotArm],
    n: usize,
    m: u32,
    block: u32,
    grid_div: u32,
    gran: u32,
) -> CommandBuffer {
    let grid = [(m + grid_div - 1) / grid_div, 1, 1];
    let blk = [block, 1, 1];
    let nslots = slots.len().max(1);
    let mut cb = CommandBuffer::new();
    for i in 0..n {
        let s = &slots[i % nslots];
        cb.dispatch_lds(kern, grid, blk, s.ka.gpu_addr, 0, gran);
        push_prod_barrier(&mut cb);
    }
    cb
}

fn arm_bos<'a>(
    module: &'a LoadedModule,
    slots: &'a [SlotArm],
    d_x: &'a GpuBuffer,
    fence: &'a GpuBuffer,
) -> Vec<&'a GpuBuffer> {
    let mut bos: Vec<&GpuBuffer> = Vec::with_capacity(3 + slots.len() * 3);
    bos.push(&module.code_buf);
    bos.push(d_x);
    bos.push(fence);
    for s in slots {
        bos.push(&s.w);
        bos.push(&s.y);
        bos.push(&s.ka);
    }
    bos
}

fn y_u32_vec(y: &[u8]) -> Vec<u32> {
    y.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        eprintln!("usage: poc_gemv_ab <elf-a> <elf-b> [MxK,...] [render-node]");
        std::process::exit(2);
    }
    let shapes: Vec<(u32, u32)> = args
        .get(2)
        .map(|s| {
            s.split(',')
                .map(|p| {
                    let (m, k) = p.split_once('x').unwrap();
                    (m.parse().unwrap(), k.parse().unwrap())
                })
                .collect()
        })
        .unwrap_or_else(|| vec![(4096, 1280), (16384, 1280), (5120, 5120), (248320, 5120)]);

    let layout_b = std::env::var("GEMV_LAYOUT_B").unwrap_or_else(|_| "aos".into());
    let layout_b_soa = matches!(layout_b.as_str(), "soa" | "SOA" | "SoA");
    let ws_mib = env_usize("GEMV_WORKING_SET_MIB", 0);
    let read_only = env_flag("GEMV_READ_ONLY");
    let reverse = env_flag("GEMV_REVERSE");
    let base_iters = env_usize("GEMV_ITERS", 200);
    let base_seed = env_u64("GEMV_SEED", 1);

    let default_sym_a = if read_only {
        "gemv_mq4g256v2_ro"
    } else {
        "gemv_mq4g256v2"
    };
    let default_sym_b = if read_only {
        if layout_b_soa {
            "gemv_mq4g256v2_soa_ro"
        } else {
            "gemv_mq4g256v2_ro"
        }
    } else if layout_b_soa {
        "gemv_mq4g256v2_soa"
    } else {
        "gemv_mq4g256v2"
    };

    let sym_a = std::env::var("GEMV_SYMBOL_A").unwrap_or_else(|_| default_sym_a.into());
    let sym_b = std::env::var("GEMV_SYMBOL_B").unwrap_or_else(|_| default_sym_b.into());
    let block_a: u32 = std::env::var("GEMV_BLOCK_A")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(32);
    let block_b: u32 = std::env::var("GEMV_BLOCK_B")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(32);
    let grid_div_a: u32 = std::env::var("GEMV_GRID_DIV_A")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    let grid_div_b: u32 = std::env::var("GEMV_GRID_DIV_B")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);

    let elf_a_bytes = std::fs::read(&args[0]).unwrap_or_else(|e| panic!("read {}: {e}", args[0]));
    let elf_b_bytes = std::fs::read(&args[1]).unwrap_or_else(|e| panic!("read {}: {e}", args[1]));
    let elf_a_hash = hex64(fnv1a64(&elf_a_bytes));
    let elf_b_hash = hex64(fnv1a64(&elf_b_bytes));

    let dev = Device::open(args.get(3).map(|s| s.as_str())).unwrap();
    let dq = DispatchQueue::new(&dev).unwrap();
    let gran = redline::dispatch::lds_granularity(&dev.info.gfx_arch);
    eprintln!(
        "gfx_arch={} gran={} layout_b={} soa_b={} ws_mib={} ro={} reverse={} iters={} seed={} route=pm4-prod-barrier",
        dev.info.gfx_arch,
        gran,
        layout_b,
        layout_b_soa,
        ws_mib,
        read_only as u8,
        reverse as u8,
        base_iters,
        base_seed
    );
    eprintln!(
        "elf_a={} hash={} sym={} block={} grid_div={}",
        args[0], elf_a_hash, sym_a, block_a, grid_div_a
    );
    eprintln!(
        "elf_b={} hash={} sym={} block={} grid_div={}",
        args[1], elf_b_hash, sym_b, block_b, grid_div_b
    );

    let mod_a = dev.load_module_file(&args[0]).unwrap();
    let mod_b = dev.load_module_file(&args[1]).unwrap();
    let kern_a = Kernel::find(&mod_a, &sym_a).unwrap_or_else(|| panic!("missing symbol {sym_a}"));
    let kern_b = Kernel::find(&mod_b, &sym_b).unwrap_or_else(|| panic!("missing symbol {sym_b}"));
    let fence = dev.alloc_vram(4096).unwrap();

    for &(m, k) in &shapes {
        assert!(k % 256 == 0, "K={k} must be multiple of 256");
        let gpr = (k / 256) as usize;
        let m_usz = m as usize;
        let wbytes = m_usz * gpr * 136;
        let slots = if ws_mib == 0 {
            1usize
        } else {
            let need = (ws_mib as u64) << 20;
            let per = wbytes.max(1) as u64;
            ((need + per - 1) / per).max(1) as usize
        };
        // Round iters up to whole rotations so each timed sample covers full WS.
        let mut n_iters = base_iters;
        if slots > 1 {
            let rem = n_iters % slots;
            if rem != 0 {
                n_iters += slots - rem;
            }
        }
        let ws_bytes = wbytes * slots;
        eprintln!(
            "M={m} K={k} gpr={gpr} weight_bytes={wbytes} slots={slots} working_set_bytes={ws_bytes} seed={base_seed}"
        );

        // Shared x (not slot-dependent); seed stream separate from weight slots.
        let x = gen_x(k, base_seed ^ 0xA5A5_5A5A_C3C3_3C3C);
        let x_hash = hex64(fnv1a64(unsafe {
            std::slice::from_raw_parts(x.as_ptr() as *const u8, x.len() * 4)
        }));
        let d_x = dev.alloc_vram(k as u64 * 4).unwrap();
        dev.upload(&d_x, unsafe {
            std::slice::from_raw_parts(x.as_ptr() as *const u8, k as usize * 4)
        })
        .unwrap();

        let mut host_aos: Vec<Vec<u8>> = Vec::with_capacity(slots);
        let mut host_b: Vec<Vec<u8>> = Vec::with_capacity(slots);
        let mut slots_a: Vec<SlotArm> = Vec::with_capacity(slots);
        let mut slots_b: Vec<SlotArm> = Vec::with_capacity(slots);

        for slot in 0..slots {
            let slot_seed = base_seed.wrapping_add(slot as u64);
            let aos = gen_weights_aos(m_usz, gpr, slot_seed);
            let aos_hash = hex64(fnv1a64(&aos));

            // Inverse byte permutation gate (SoA ↔ AoS) before any GPU work.
            let soa = aos_to_soa(&aos, m_usz, gpr);
            let back = soa_to_aos(&soa, m_usz, gpr);
            if back != aos {
                eprintln!("FAIL M={m} K={k} slot={slot}: inverse SoA layout mismatch");
                std::process::exit(1);
            }
            let b_bytes = if layout_b_soa { soa } else { aos.clone() };
            let b_hash = hex64(fnv1a64(&b_bytes));
            eprintln!(
                "  slot={slot} seed={slot_seed} aos_hash={aos_hash} b_hash={b_hash} x_hash={x_hash}"
            );

            let d_w_a = dev.alloc_vram(wbytes as u64).unwrap();
            dev.upload(&d_w_a, &aos).unwrap();
            let d_y_a = dev.alloc_vram(m as u64 * 4).unwrap();
            // zero y so RO/plain stores are observable
            dev.upload(&d_y_a, &vec![0u8; m_usz * 4]).unwrap();
            let d_ka_a = dev.alloc_vram(4096).unwrap();
            {
                let mut ka = KernargBuilder::new(32);
                ka.write_ptr(0, d_w_a.gpu_addr)
                    .write_ptr(8, d_x.gpu_addr)
                    .write_ptr(16, d_y_a.gpu_addr)
                    .write_u32(24, m)
                    .write_u32(28, k);
                dev.upload(&d_ka_a, ka.as_bytes()).unwrap();
            }

            let d_w_b = dev.alloc_vram(wbytes as u64).unwrap();
            dev.upload(&d_w_b, &b_bytes).unwrap();
            let d_y_b = dev.alloc_vram(m as u64 * 4).unwrap();
            dev.upload(&d_y_b, &vec![0u8; m_usz * 4]).unwrap();
            let d_ka_b = dev.alloc_vram(4096).unwrap();
            {
                let mut ka = KernargBuilder::new(32);
                ka.write_ptr(0, d_w_b.gpu_addr)
                    .write_ptr(8, d_x.gpu_addr)
                    .write_ptr(16, d_y_b.gpu_addr)
                    .write_u32(24, m)
                    .write_u32(28, k);
                dev.upload(&d_ka_b, ka.as_bytes()).unwrap();
            }

            host_aos.push(aos);
            host_b.push(b_bytes);
            slots_a.push(SlotArm {
                w: d_w_a,
                y: d_y_a,
                ka: d_ka_a,
            });
            slots_b.push(SlotArm {
                w: d_w_b,
                y: d_y_b,
                ka: d_ka_b,
            });
        }

        // ——— parity: every slot, all-row A/B + CPU oracle ———
        let bos_a = arm_bos(&mod_a, &slots_a, &d_x, &fence);
        let bos_b = arm_bos(&mod_b, &slots_b, &d_x, &fence);

        for slot in 0..slots {
            // Clear y then single-dispatch each arm on this slot's kernarg.
            dev.upload(&slots_a[slot].y, &vec![0u8; m_usz * 4]).unwrap();
            dev.upload(&slots_b[slot].y, &vec![0u8; m_usz * 4]).unwrap();

            let mut cb = CommandBuffer::new();
            cb.dispatch_lds(
                &kern_a,
                [(m + grid_div_a - 1) / grid_div_a, 1, 1],
                [block_a, 1, 1],
                slots_a[slot].ka.gpu_addr,
                0,
                gran,
            );
            push_prod_barrier(&mut cb);
            dq.submit(&dev, &cb, &bos_a).unwrap();

            let mut cb = CommandBuffer::new();
            cb.dispatch_lds(
                &kern_b,
                [(m + grid_div_b - 1) / grid_div_b, 1, 1],
                [block_b, 1, 1],
                slots_b[slot].ka.gpu_addr,
                0,
                gran,
            );
            push_prod_barrier(&mut cb);
            dq.submit(&dev, &cb, &bos_b).unwrap();

            let mut ya = vec![0u8; m_usz * 4];
            let mut yb = vec![0u8; m_usz * 4];
            dev.download(&slots_a[slot].y, &mut ya).unwrap();
            dev.download(&slots_b[slot].y, &mut yb).unwrap();

            let diff = ya
                .chunks(4)
                .zip(yb.chunks(4))
                .filter(|(a, b)| a != b)
                .count();
            if diff != 0 {
                eprintln!("FAIL M={m} K={k} slot={slot}: A/B y-bit-diff rows={diff} (all-row)");
                for (row, (a, b)) in ya
                    .chunks_exact(4)
                    .zip(yb.chunks_exact(4))
                    .enumerate()
                    .filter(|(_, (a, b))| a != b)
                    .take(4)
                {
                    let a = u32::from_le_bytes(a.try_into().unwrap());
                    let b = u32::from_le_bytes(b.try_into().unwrap());
                    eprintln!(
                        "  row={row} A={a:08x} ({}) B={b:08x} ({})",
                        f32::from_bits(a),
                        f32::from_bits(b)
                    );
                }
                std::process::exit(1);
            }
            eprintln!("  slot={slot} A/B all-row y-bit-diff=0");

            if read_only {
                // RO CPU oracle: all rows when M is modest; else 0 and M-1.
                let check_all = m_usz <= 8192;
                let rows: Vec<usize> = if check_all {
                    (0..m_usz).collect()
                } else {
                    vec![0, m_usz - 1]
                };
                let ya32 = y_u32_vec(&ya);
                let yb32 = y_u32_vec(&yb);
                for &row in &rows {
                    let exp_a = cpu_ro_row_aos(&host_aos[slot], row, gpr);
                    let exp_b = if layout_b_soa {
                        cpu_ro_row_soa(&host_b[slot], row, gpr, m_usz)
                    } else {
                        cpu_ro_row_aos(&host_b[slot], row, gpr)
                    };
                    if exp_a != exp_b {
                        eprintln!(
                            "FAIL M={m} K={k} slot={slot} row={row}: RO CPU A/B layout diverge {exp_a:08x} vs {exp_b:08x}"
                        );
                        std::process::exit(1);
                    }
                    if ya32[row] != exp_a {
                        eprintln!(
                            "FAIL M={m} K={k} slot={slot} row={row}: RO GPU A {:08x} != CPU {:08x}",
                            ya32[row], exp_a
                        );
                        std::process::exit(1);
                    }
                    if yb32[row] != exp_b {
                        eprintln!(
                            "FAIL M={m} K={k} slot={slot} row={row}: RO GPU B {:08x} != CPU {:08x}",
                            yb32[row], exp_b
                        );
                        std::process::exit(1);
                    }
                }
                eprintln!(
                    "  slot={slot} RO CPU oracle ok rows={} (check_all={check_all})",
                    rows.len()
                );
            } else {
                // Plain GEMV: CPU rows 0 and M-1 on AoS arm A; label residual
                // reliance on all-row A/B + parent HIP CPU gate.
                for &row in &[0usize, m_usz - 1] {
                    let exp = cpu_gemv_row_aos(&host_aos[slot], &x, row, k as usize);
                    let got = f32::from_le_bytes(ya[row * 4..row * 4 + 4].try_into().unwrap());
                    let err = (got - exp).abs();
                    let rel = (err as f64) / (exp.abs() as f64).max(1e-30);
                    if !(got.is_finite() && exp.is_finite() && err.is_finite() && rel <= 1e-4) {
                        eprintln!(
                            "FAIL M={m} K={k} slot={slot} row={row}: plain GPU A {got} vs CPU {exp} rel={rel:.3e}"
                        );
                        std::process::exit(1);
                    }
                }
                eprintln!(
                    "  slot={slot} plain CPU oracle rows 0/M-1 ok (full plain also: all-row A/B + parent HIP CPU gate)"
                );
            }
        }

        // ——— timing: warm full rotations, raw 3 samples, median ———
        let logical_bytes = wbytes as f64 + k as f64 * 4.0 + m as f64 * 4.0;
        let mut per = [0f64; 2];
        let timed_order: [usize; 2] = if reverse { [1, 0] } else { [0, 1] };

        for &mi in &timed_order {
            let (label, kern, slots_arm, block, grid_div, module) = if mi == 0 {
                ("A", kern_a, slots_a.as_slice(), block_a, grid_div_a, &mod_a)
            } else {
                ("B", kern_b, slots_b.as_slice(), block_b, grid_div_b, &mod_b)
            };
            let bos = arm_bos(module, slots_arm, &d_x, &fence);
            let cb1 = build_cb(kern, slots_arm, 1, m, block, grid_div, gran);
            let cbn = build_cb(kern, slots_arm, n_iters, m, block, grid_div, gran);

            // CPU input construction can let DPM fall idle. Warm each arm for
            // at least 250 ms, not merely one short rotation, before sampling.
            let warm_start = Instant::now();
            loop {
                dq.submit(&dev, &cbn, &bos).unwrap();
                if warm_start.elapsed() >= std::time::Duration::from_millis(250) {
                    break;
                }
            }
            eprintln!(
                "  arm={label} warmup_ms={:.1}",
                warm_start.elapsed().as_secs_f64() * 1e3
            );

            let mut t1_samples = [0f64; 3];
            for s in &mut t1_samples {
                let t = Instant::now();
                dq.submit(&dev, &cb1, &bos).unwrap();
                *s = t.elapsed().as_secs_f64() * 1e6;
            }
            let mut tn_samples = [0f64; 3];
            for s in &mut tn_samples {
                let t = Instant::now();
                dq.submit(&dev, &cbn, &bos).unwrap();
                *s = t.elapsed().as_secs_f64() * 1e6;
            }
            let t1 = median3(t1_samples);
            let tn = median3(tn_samples);
            let us = if n_iters > 1 {
                (tn - t1) / (n_iters as f64 - 1.0)
            } else {
                t1
            };
            per[mi] = us;

            // Marginal samples from paired raw t1/tN (report all).
            let mut marg = [0f64; 3];
            for i in 0..3 {
                marg[i] = if n_iters > 1 {
                    (tn_samples[i] - t1_samples[i]) / (n_iters as f64 - 1.0)
                } else {
                    t1_samples[i]
                };
            }

            let elf_name = args[mi].rsplit('/').next().unwrap();
            eprintln!(
                "M={m} K={k} arm{mi}/{label} {elf_name}: {us:.2} us/dispatch  logical {:.0} GB/s  (median t1={t1:.0}us tN={tn:.0}us iters={n_iters} slots={slots})",
                logical_bytes / us / 1e3
            );
            eprintln!(
                "  raw t1_us=[{:.3},{:.3},{:.3}] tN_us=[{:.3},{:.3},{:.3}] marginal_us=[{:.3},{:.3},{:.3}] order={}",
                t1_samples[0],
                t1_samples[1],
                t1_samples[2],
                tn_samples[0],
                tn_samples[1],
                tn_samples[2],
                marg[0],
                marg[1],
                marg[2],
                if reverse { "B,A" } else { "A,B" }
            );
            eprintln!(
                "  note: logical GB/s = analytical bytes/us; not a DRAM bandwidth claim. working_set_bytes={}",
                ws_bytes
            );
            let _ = slots_arm; // y residual after rotation not used as parity gate
        }

        eprintln!(
            "M={m} K={k} ratio b/a={:.3} (parity gated pre-timing; timed order={})",
            per[1] / per[0],
            if reverse { "B,A" } else { "A,B" }
        );
    }
}
