// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Instruction fields derived from Xilinx/llvm-aie AIE2PGenInstrInfo.td and
// MCTargetDesc/aie2p/AIE2PMCCodeEmitterGen.inc (Apache-2.0 WITH LLVM-exception).
//! Signed int8 GEMM: an 8x8 output microtile, reduction in 64-element DMA chunks.
//! Host matrices are row-major; DMA streams contain row-major 8x8 submatrices.
use crate::{
    dma::{self, BdLocks},
    isa::{self, Inst, Program, Reg},
};
use std::sync::LazyLock;

fn descriptor(name: &str) -> &'static isa::gen::Encoding {
    isa::gen::ENCODINGS
        .iter()
        .find(|e| e.name == name)
        .expect("generated AIE2P kernel instruction")
}
fn encoded(encoding: &isa::gen::Encoding, operands: &[(&str, u64)]) -> u64 {
    isa::bundle::encode_slot(encoding, operands, 0).expect("valid kernel operands")
}

pub const MICRO: usize = 8;
pub const CHUNK_K: usize = 64;
pub const CHUNK_BYTES: usize = MICRO * CHUNK_K;
pub const A_ADDR: [u32; 2] = [0x0000, 0x1000];
pub const B_ADDR: [u32; 2] = [0x2000, 0x3000];
pub const C_ADDR: u32 = 0x4000;
/// amode=0, bmode=1: 8x8 by 8x8, signed X/Y, 64 acc32 lanes.
/// aie2p_vmult.h compute_control + mul/mac_8x8_8x8 signed overloads;
/// vendor 032eb944... matmul_i8_i32 initializes r2 to this exact value.
pub const SIGNED_8X8: u32 = 0x308;

/// VMAC_vmul_cm_core_X_X (TableGen lines 8787..8797).
/// `acc=None` emits VMUL_vmul_cm_core_X_X (lines 9628..9637).
pub fn matrix_mac(dst: u8, acc: Option<u8>, a: u8, b: u8, conf: u8) -> Inst {
    assert!(dst < 8 && acc.is_none_or(|x| x < 7) && a < 16 && b < 16 && conf < 32);
    static VMAC: LazyLock<&'static isa::gen::Encoding> =
        LazyLock::new(|| descriptor("VMAC_vmul_cm_core_X_X"));
    static VMUL: LazyLock<&'static isa::gen::Encoding> =
        LazyLock::new(|| descriptor("VMUL_vmul_cm_core_X_X"));
    let common = [
        ("dst", dst as u64),
        ("s1", a as u64),
        ("s2", b as u64),
        ("acc", conf as u64),
    ];
    Inst::Vec(match acc {
        Some(acc) => encoded(
            &VMAC,
            &[
                common[0],
                common[1],
                common[2],
                common[3],
                ("acc1", acc as u64),
            ],
        ),
        None => encoded(&VMUL, &common),
    })
}

/// VLDA_dmx_lda_x_pstm_nrm_imm (lines 7980..7988), step=64 bytes.
pub fn load_x(x: u8, ptr: u8) -> Inst {
    assert!(x < 16 && ptr < 8);
    static LOAD: LazyLock<&'static isa::gen::Encoding> =
        LazyLock::new(|| descriptor("VLDA_dmx_lda_x_pstm_nrm_imm"));
    Inst::Lda(encoded(
        &LOAD,
        &[("dst", x as u64), ("ptr", ptr as u64), ("imm", 1)],
    ))
}

/// VST_dmx_sts_bm_pstm_nrm_imm (lines 11481..11489), step=64 bytes.
/// getmBMsOpValue: quarter 0=LL,1=LH,2=HL,3=HH; 16 i32 lanes each.
pub fn store_acc_quarter(dm: u8, quarter: u8, ptr: u8) -> Inst {
    assert!(dm < 8 && quarter < 4 && ptr < 8);
    static STORE: LazyLock<&'static isa::gen::Encoding> =
        LazyLock::new(|| descriptor("VST_dmx_sts_bm_pstm_nrm_imm"));
    Inst::St(encoded(
        &STORE,
        &[
            ("src", (dm * 4 + quarter) as u64),
            ("ptr", ptr as u64),
            ("imm", 1),
        ],
    ))
}

fn lock(p: &mut Program, acquire: bool, id: u32) {
    p.push(if acquire {
        isa::acq((48 + id) as u64, Reg::R(0))
    } else {
        isa::rel((48 + id) as u64, Reg::R(1))
    })
    .nops(7);
}

fn job_body(p: &mut Program, k: usize, starting_slot: usize) {
    lock(p, true, 8); // C empty: no overwrite while output DMA drains.
    for chunk in 0..k / CHUNK_K {
        let slot = (starting_slot + chunk) % 2;
        lock(p, true, (slot * 4 + 1) as u32);
        lock(p, true, (slot * 4 + 3) as u32);
        p.push(isa::movxm(Reg::P(0), 0x70000 + A_ADDR[slot]));
        p.push(isa::movxm(Reg::P(1), 0x70000 + B_ADDR[slot]))
            .nops(7);
        for block in 0..CHUNK_K / MICRO {
            p.push(load_x(0, 0)).push(load_x(1, 1)).nops(7);
            p.push(matrix_mac(
                0,
                if chunk == 0 && block == 0 {
                    None
                } else {
                    Some(0)
                },
                0,
                1,
                2,
            ));
        }
        // Last VMAC has consumed both sources before the producer gets its buffer.
        lock(p, false, (slot * 4) as u32);
        lock(p, false, (slot * 4 + 2) as u32);
    }
    p.nops(7)
        .push(isa::movxm(Reg::P(2), 0x70000 + C_ADDR))
        .nops(7);
    for quarter in 0..4 {
        p.push(store_acc_quarter(0, quarter, 2));
    }
    p.nops(7); // Store memory stage is cycle 5; publish only after completion.
    lock(p, false, 9);
}

/// Largest single-core job count (256^3 = 1024 microtiles; 4 MiB input stream).
pub const MAX_JOBS: usize = 1024;

/// Core program consumes `jobs` consecutive packed jobs; no initial C is read.
/// C is 8x8 row-major i32. The input BD rings stay alternating across jobs.
/// Conservative standalone schedule: load latency7 and VMAC result latency6
/// from AIE2PGenSchedule.td; no VLIW overlap or peak throughput claim.
pub fn core_program(k: usize, jobs: usize) -> Program {
    assert!(k > 0 && k % CHUNK_K == 0 && k <= 256 && jobs > 0 && jobs <= MAX_JOBS);
    let mut p = Program::new();
    p.push(isa::movxm(Reg::R(0), (-1i32) as u32));
    p.push(isa::movxm(Reg::R(1), 1));
    p.push(isa::movxm(Reg::R(2), SIGNED_8X8)).nops(7);
    if jobs >= 2 {
        p.push(isa::movxm(Reg::R(3), (jobs / 2) as u32)).nops(7);
        let target = p.pc() as u64;
        job_body(&mut p, k, 0);
        job_body(&mut p, k, (k / CHUNK_K) % 2);
        p.push(isa::add_ri(Reg::R(3), Reg::R(3), -1)).nops(7);
        p.push(isa::jnz(Reg::R(3), target)).nops(7);
    }
    if jobs % 2 != 0 {
        job_body(&mut p, k, 0);
    }
    p.push(isa::done()).nops(7);
    p
}

/// S2MM0 ring BDs0..3=A0,B0,A1,B1; BD4=C MM2S0 ring.
/// DMA own-tile lock IDs are used here, not core48+IDs.
pub fn tile_bds() -> [[u32; 6]; 5] {
    let mut bds = [[0; 6]; 5];
    for slot in 0..2 {
        let base = (slot * 4) as u32;
        bds[slot * 2] = dma::tile_bd(
            A_ADDR[slot],
            (CHUNK_BYTES / 4) as u32,
            BdLocks {
                acq: Some((base, -1)),
                rel: Some((base + 1, 1)),
            },
            Some((slot * 2 + 1) as u32),
        );
        bds[slot * 2 + 1] = dma::tile_bd(
            B_ADDR[slot],
            (CHUNK_BYTES / 4) as u32,
            BdLocks {
                acq: Some((base + 2, -1)),
                rel: Some((base + 3, 1)),
            },
            Some(((slot * 2 + 2) % 4) as u32),
        );
    }
    bds[4] = dma::tile_bd(
        C_ADDR,
        64,
        BdLocks {
            acq: Some((9, -1)),
            rel: Some((8, 1)),
        },
        Some(4),
    );
    bds
}

pub fn initial_locks() -> [i32; 16] {
    [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0]
}

pub struct PackedJob {
    pub a: Vec<u8>,
    pub b: Vec<u8>,
}

/// Pack a single output microtile's complete reduction as consecutive 512-byte
/// chunks, each consisting of eight 8x8 row-major matrices. `mb/nb` are block indices.
pub fn pack_job(
    a: &[i8],
    b: &[i8],
    m: usize,
    n: usize,
    k: usize,
    mb: usize,
    nb: usize,
) -> PackedJob {
    assert!(m % 8 == 0 && n % 8 == 0 && k > 0 && k % 64 == 0);
    assert!(a.len() == m * k && b.len() == k * n && mb < m / 8 && nb < n / 8);
    let mut out = PackedJob {
        a: Vec::with_capacity(8 * k),
        b: Vec::with_capacity(8 * k),
    };
    for kb in (0..k).step_by(8) {
        for row in 0..8 {
            for kk in 0..8 {
                out.a.push(a[(mb * 8 + row) * k + kb + kk] as u8);
            }
        }
        for kk in 0..8 {
            for col in 0..8 {
                out.b.push(b[(kb + kk) * n + nb * 8 + col] as u8);
            }
        }
    }
    out
}

fn append_job_stream(
    stream: &mut Vec<u8>,
    a: &[i8],
    b: &[i8],
    n: usize,
    k: usize,
    mb: usize,
    nb: usize,
) {
    for chunk in (0..k).step_by(CHUNK_K) {
        for kb in (chunk..chunk + CHUNK_K).step_by(8) {
            for r in 0..8 {
                for kk in 0..8 {
                    stream.push(a[(mb * 8 + r) * k + kb + kk] as u8);
                }
            }
        }
        for kb in (chunk..chunk + CHUNK_K).step_by(8) {
            for kk in 0..8 {
                for c in 0..8 {
                    stream.push(b[(kb + kk) * n + nb * 8 + c] as u8);
                }
            }
        }
    }
}

/// Exact wrapping int8×int8→int32 reference. Row-parallel over all host threads with a k-outer,
/// contiguous-column inner loop (wrapping adds are order-independent), so expert-shape references
/// (e.g. 4096×1280×2560) take well under a second inside a hardware window.
pub fn cpu_reference(a: &[i8], b: &[i8], m: usize, n: usize, k: usize) -> Vec<i32> {
    assert!(a.len() == m * k && b.len() == k * n);
    let mut c = vec![0i32; m * n];
    if m == 0 || n == 0 {
        return c;
    }
    let threads = std::thread::available_parallelism()
        .map_or(1, |t| t.get())
        .min(m);
    let rows_per_thread = m.div_ceil(threads);
    std::thread::scope(|s| {
        for (part, rows) in c.chunks_mut(rows_per_thread * n).enumerate() {
            s.spawn(move || {
                for (r, out) in rows.chunks_mut(n).enumerate() {
                    let row = part * rows_per_thread + r;
                    for (kk, &x) in a[row * k..(row + 1) * k].iter().enumerate() {
                        let x = x as i32;
                        for (o, &y) in out.iter_mut().zip(&b[kk * n..(kk + 1) * n]) {
                            *o = o.wrapping_add(x * y as i32);
                        }
                    }
                }
            });
        }
    });
    c
}

/// Ideal hardware ceiling, not this conservatively scheduled kernel's speed.
/// One VMAC = 8*8*8 = 512 MACs; 32 cores; 2 operations per MAC.
pub fn peak_ops_per_second(clock_hz: u64) -> u64 {
    512 * 32 * 2 * clock_hz
}

/// Shape supported by the full 8-column, 4-core-row plan; no padded edge tiles.
#[derive(Clone, Copy, Debug)]
pub struct ArrayShape {
    pub m: usize,
    pub n: usize,
    pub k: usize,
}
impl ArrayShape {
    pub fn waves(self) -> usize {
        assert!(self.m > 0 && self.m % 32 == 0 && self.n > 0 && self.n % 64 == 0);
        assert!(self.k > 0 && self.k % 64 == 0 && self.k <= 256);
        let waves = (self.m / 32) * (self.n / 64);
        assert!(waves <= 256);
        waves
    }
    /// Block index assigned to one physical core in this wave.
    pub fn microtile(self, wave: usize, column: usize, row: usize) -> (usize, usize) {
        assert!(wave < self.waves() && column < 8 && row < 4);
        (
            (wave / (self.n / 64)) * 4 + row,
            (wave % (self.n / 64)) * 8 + column,
        )
    }
}

/// Column streams: wave-major, then physical core row, then K64 chunk.
/// Each chunk is A512 followed by B512. No runtime vendor packing is needed.
pub fn pack_columns(a: &[i8], b: &[i8], shape: ArrayShape) -> [Vec<u8>; 8] {
    let waves = shape.waves();
    assert!(a.len() == shape.m * shape.k && b.len() == shape.k * shape.n);
    std::array::from_fn(|col| {
        let mut stream = Vec::with_capacity(waves * 4 * 16 * shape.k);
        for wave in 0..waves {
            for row in 0..4 {
                let (mb, nb) = shape.microtile(wave, col, row);
                append_job_stream(&mut stream, a, b, shape.n, shape.k, mb, nb);
            }
        }
        stream
    })
}

/// Gather shim C streams (wave-major, row-major microtiles) into host row-major C.
pub fn unpack_columns(columns: [&[i32]; 8], shape: ArrayShape) -> Vec<i32> {
    let waves = shape.waves();
    assert!(columns.iter().all(|c| c.len() == waves * 4 * 64));
    let mut c = vec![0; shape.m * shape.n];
    for (col, stream) in columns.iter().enumerate() {
        for wave in 0..waves {
            for row in 0..4 {
                let (mb, nb) = shape.microtile(wave, col, row);
                let base = (wave * 4 + row) * 64;
                for r in 0..8 {
                    c[(mb * 8 + r) * shape.n + nb * 8..(mb * 8 + r) * shape.n + nb * 8 + 8]
                        .copy_from_slice(&stream[base + r * 8..base + r * 8 + 8]);
                }
            }
        }
    }
    c
}

#[derive(Clone, Copy, Debug)]
pub struct CycleEstimate {
    /// Perfect one-VMAC-per-cycle array utilization; excludes all data movement.
    pub ideal_mac_cycles: u64,
    /// One standalone instruction issue/cycle, including explicit NOPs/loops.
    /// Excludes DMA, lock waits, instruction fetch and any additional hazards.
    pub standalone_issue_cycles: u64,
    pub peak_ops_per_second: u64,
}
pub fn estimate(shape: ArrayShape, clock_hz: u64) -> CycleEstimate {
    let jobs = shape.waves() as u64;
    let pairs = jobs / 2;
    CycleEstimate {
        ideal_mac_cycles: shape.m as u64 * shape.n as u64 * shape.k as u64 / (512 * 32),
        standalone_issue_cycles: 18
            + if pairs > 0 { 8 } else { 0 }
            + jobs * (42 + 121 * (shape.k as u64 / 64))
            + pairs * 16,
        peak_ops_per_second: peak_ops_per_second(clock_hz),
    }
}

/// One BD assignment in the physical array. Memtile addresses/lock IDs are
/// explicitly in the DMA own-memory view (+0x80000 / +64).
#[derive(Clone, Debug)]
pub struct Descriptor {
    pub tile: dma::Location,
    pub id: u32,
    pub bd: dma::Bd,
}
#[derive(Clone, Copy, Debug)]
pub struct DmaTask {
    pub tile: dma::Location,
    pub task: dma::Task,
}
pub struct ArrayPlan {
    pub shape: ArrayShape,
    pub program: Program,
    pub descriptors: Vec<Descriptor>,
    pub tasks: Vec<DmaTask>,
    pub circuits: Vec<crate::route::Circuit>,
    pub shim_routes: Vec<crate::route::ShimDma>,
    /// Own-module lock register writes (not DMA neighbor-view lock IDs).
    pub locks: Vec<(u32, u32)>,
}

impl ArrayPlan {
    /// Host addresses are per-column shim input/output DMA addresses. An offline
    /// caller can supply offsets and use firmware DDR_PATCH before starting.
    pub fn new(shape: ArrayShape, input: [u64; 8], output: [u64; 8]) -> Self {
        use crate::route::{Circuit, Port, ShimDma};
        use dma::{
            Bd,
            Direction::{Mm2s, S2mm},
            Location,
        };
        let waves = shape.waves();
        let mut plan = Self {
            shape,
            program: core_program(shape.k, waves),
            descriptors: Vec::new(),
            tasks: Vec::new(),
            circuits: Vec::new(),
            shim_routes: Vec::new(),
            locks: Vec::new(),
        };
        assert!(plan.program.bytes.len() <= 16 * 1024);
        for col in 0..8 {
            let shim = Location::new(col, 0);
            let mem = Location::new(col, 1);
            let mm = ShimDma {
                tile: shim,
                direction: Mm2s,
                channel: 0,
            };
            let sm = ShimDma {
                tile: shim,
                direction: S2mm,
                channel: 0,
            };
            plan.circuits.extend([
                Circuit {
                    tile: shim,
                    slave: mm.port(),
                    master: Port::North(0),
                },
                Circuit {
                    tile: shim,
                    slave: Port::North(0),
                    master: sm.port(),
                },
                Circuit {
                    tile: mem,
                    slave: Port::South(0),
                    master: Port::Dma(0),
                },
                Circuit {
                    tile: mem,
                    slave: Port::Dma(0),
                    master: Port::South(0),
                },
            ]);
            plan.shim_routes.extend([mm, sm]);
            plan.descriptors.extend([
                Descriptor {
                    tile: shim,
                    id: 0,
                    bd: Bd::new(input[col as usize], (waves * 16 * shape.k) as u32),
                },
                Descriptor {
                    tile: shim,
                    id: 1,
                    bd: Bd::new(output[col as usize], (waves * 256) as u32),
                },
            ]);
            // Shim output is finite and carries the command-completion token.
            plan.tasks.push(DmaTask {
                tile: shim,
                task: dma::Task {
                    direction: S2mm,
                    channel: 0,
                    bd: 1,
                    repeat: 1,
                    issue_token: true,
                },
            });
            for row in 0..4 {
                let core = Location::new(col, row + 2);
                let lane = row as u8;
                plan.circuits.extend([
                    Circuit {
                        tile: mem,
                        slave: Port::Dma(lane + 2),
                        master: Port::North(lane),
                    },
                    Circuit {
                        tile: mem,
                        slave: Port::North(lane),
                        master: Port::Dma(lane + 2),
                    },
                    Circuit {
                        tile: core,
                        slave: Port::South(lane),
                        master: Port::Dma(0),
                    },
                    Circuit {
                        tile: core,
                        slave: Port::Dma(0),
                        master: Port::South(lane),
                    },
                ]);
                for upper in row + 1..4 {
                    plan.circuits.extend([
                        Circuit {
                            tile: core,
                            slave: Port::South(upper as u8),
                            master: Port::North(upper as u8),
                        },
                        Circuit {
                            tile: core,
                            slave: Port::North(upper as u8),
                            master: Port::South(upper as u8),
                        },
                    ]);
                }
                for (id, value) in initial_locks().into_iter().enumerate() {
                    plan.locks
                        .push(dma::lock_write(core, id as u32, value as u32));
                }
                for slot in 0..2 {
                    let base = slot * 4;
                    for (id, addr, empty) in [
                        (slot * 2, A_ADDR[slot], base),
                        (slot * 2 + 1, B_ADDR[slot], base + 2),
                    ] {
                        let mut bd = Bd::new(addr as u64, 128);
                        bd.locks = BdLocks {
                            acq: Some((empty as u32, -1)),
                            rel: Some((empty as u32 + 1, 1)),
                        };
                        bd.next = Some(((id + 1) % 4) as u32);
                        plan.descriptors.push(Descriptor {
                            tile: core,
                            id: id as u32,
                            bd,
                        });
                    }
                }
                let mut c = Bd::new(C_ADDR as u64, 64);
                c.locks = BdLocks {
                    acq: Some((9, -1)),
                    rel: Some((8, 1)),
                };
                c.next = Some(4);
                plan.descriptors.push(Descriptor {
                    tile: core,
                    id: 4,
                    bd: c,
                });
                plan.tasks.extend([
                    DmaTask {
                        tile: core,
                        task: dma::Task {
                            direction: Mm2s,
                            channel: 0,
                            bd: 4,
                            repeat: 1,
                            issue_token: false,
                        },
                    },
                    DmaTask {
                        tile: core,
                        task: dma::Task {
                            direction: S2mm,
                            channel: 0,
                            bd: 0,
                            repeat: 1,
                            issue_token: false,
                        },
                    },
                ]);
                // Even channels2,4 use BD bank0; odd channels3,5 use bank24.
                // Channel-specific AB output BDs0/24/2/26 and C input4/28/6/30.
                let bank = if row % 2 == 0 { 0 } else { 24 };
                let ab_out = bank + (row / 2) * 2;
                let c_in = ab_out + 4;
                for slot in 0..2u32 {
                    let index = slot * 4 + row;
                    let ab_addr = 0x80000 + row * 0x2000 + slot * 0x1000;
                    let c_addr = 0x90000 + row * 0x2000 + slot * 0x1000;
                    let ab_lock = row * 4 + slot * 2;
                    let c_lock = 16 + ab_lock;
                    for id in [ab_lock, c_lock] {
                        plan.locks.push(dma::lock_write(mem, id, 1));
                        plan.locks.push(dma::lock_write(mem, id + 1, 0));
                    }
                    for (id, addr, words, empty, producer, next) in [
                        (
                            8 + index,
                            ab_addr,
                            (shape.k * 4) as u32,
                            ab_lock,
                            true,
                            8 + (index + 1) % 8,
                        ),
                        (
                            ab_out + slot,
                            ab_addr,
                            (shape.k * 4) as u32,
                            ab_lock,
                            false,
                            ab_out + (slot + 1) % 2,
                        ),
                        (c_in + slot, c_addr, 64, c_lock, true, c_in + (slot + 1) % 2),
                        (16 + index, c_addr, 64, c_lock, false, 16 + (index + 1) % 8),
                    ] {
                        let mut bd = Bd::new(addr as u64, words);
                        bd.next = Some(next);
                        bd.locks = if producer {
                            BdLocks {
                                acq: Some((64 + empty, -1)),
                                rel: Some((65 + empty, 1)),
                            }
                        } else {
                            BdLocks {
                                acq: Some((65 + empty, -1)),
                                rel: Some((64 + empty, 1)),
                            }
                        };
                        plan.descriptors.push(Descriptor { tile: mem, id, bd });
                    }
                }
                plan.tasks.extend([
                    DmaTask {
                        tile: mem,
                        task: dma::Task {
                            direction: Mm2s,
                            channel: row + 2,
                            bd: ab_out,
                            repeat: 1,
                            issue_token: false,
                        },
                    },
                    DmaTask {
                        tile: mem,
                        task: dma::Task {
                            direction: S2mm,
                            channel: row + 2,
                            bd: c_in,
                            repeat: 1,
                            issue_token: false,
                        },
                    },
                ]);
            }
            plan.tasks.extend([
                DmaTask {
                    tile: mem,
                    task: dma::Task {
                        direction: Mm2s,
                        channel: 0,
                        bd: 16,
                        repeat: 1,
                        issue_token: false,
                    },
                },
                DmaTask {
                    tile: mem,
                    task: dma::Task {
                        direction: S2mm,
                        channel: 0,
                        bd: 8,
                        repeat: 1,
                        issue_token: false,
                    },
                },
                DmaTask {
                    tile: shim,
                    task: dma::Task {
                        direction: Mm2s,
                        channel: 0,
                        bd: 0,
                        repeat: 1,
                        issue_token: false,
                    },
                },
            ]);
        }
        plan
    }

    /// DMA/routing/lock configuration for a reset partition. Program memory
    /// loading and core reset/enable remain the integration caller's responsibility.
    pub fn emit_cdo(&self, cdo: &mut crate::cdo::Cdo) {
        for &(addr, value) in &self.locks {
            cdo.write(addr, value);
        }
        for &circuit in &self.circuits {
            circuit.emit_cdo(cdo);
        }
        for &route in &self.shim_routes {
            route.emit_cdo(cdo);
        }
        for entry in &self.descriptors {
            entry.bd.emit_cdo(entry.tile, entry.id, cdo);
        }
        for col in 0..8 {
            for (addr, value) in crate::regs::shim_token_route(col) {
                cdo.write(addr, value);
            }
        }
    }

    /// Enqueue persistent tile/memtile rings and finite shim tasks. After the
    /// caller enables cores, a sync for shim S2MM0 over8columns waits for all C.
    pub fn start_txn(&self, txn: &mut crate::txn::Txn) {
        for col in 0..8 {
            txn.mask_write(
                dma::Location::new(col, 0).address(crate::regs::shim::DMA_S2MM_0_CTRL),
                0xf00,
                0x1f00,
            );
        }
        for entry in &self.tasks {
            entry.task.emit_txn(entry.tile, txn);
            if entry.tile.kind() != dma::TileType::Shim {
                entry.task.enable_txn(entry.tile, txn);
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArgKind {
    In,
    Out,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArgSpec {
    pub bytes: usize,
    pub kind: ArgKind,
}

/// Runnable, two-argument single-core baseline; the larger ArrayPlan is separate.
pub struct Design {
    pub pdi: Vec<u8>,
    pub insts: Vec<u8>,
    pub args: Vec<ArgSpec>,
    pub col: u32,
    m: usize,
    n: usize,
    k: usize,
    /// Shim AXI attributes currently encoded in `insts`.
    shim_axi: dma::ShimAxi,
}
impl Design {
    /// Return buffers for In arguments only, in argument order (one combined AB
    /// buffer). The caller separately allocates the Out buffer from `args[1]`.
    pub fn pack_in(&self, a: &[i8], b: &[i8]) -> Vec<Vec<u8>> {
        assert!(a.len() == self.m * self.k && b.len() == self.k * self.n);
        let mut stream = Vec::with_capacity(self.args[0].bytes);
        for mb in 0..self.m / 8 {
            for nb in 0..self.n / 8 {
                append_job_stream(&mut stream, a, b, self.n, self.k, mb, nb);
            }
        }
        vec![stream]
    }
    pub fn unpack_out(&self, output: &[u8]) -> Vec<i32> {
        assert_eq!(output.len(), self.args[1].bytes);
        let mut c = vec![0; self.m * self.n];
        for mb in 0..self.m / 8 {
            for nb in 0..self.n / 8 {
                let base = (mb * (self.n / 8) + nb) * 256;
                for r in 0..8 {
                    for col in 0..8 {
                        let off = base + (r * 8 + col) * 4;
                        c[(mb * 8 + r) * self.n + nb * 8 + col] =
                            i32::from_le_bytes(output[off..off + 4].try_into().unwrap());
                    }
                }
            }
        }
        c
    }
    /// Set burst length, AxCACHE and AxQoS of both shim BDs in place in [`Design::insts`] (default
    /// [`dma::ShimAxi::default`]: burst 3, cache 2, qos 0). Same contract as `ArrayDesign::set_shim_axi`:
    /// absolute value, atomic, no allocation, only the burst/cache/qos bits change, `pdi` unchanged (shim BDs are
    /// rewritten by the TXN on every submit); call before configuring/submitting. No simulator performance or
    /// coherence guarantee. Non-vendor word-5 attributes (AxCACHE != 2 or AxQoS != 0) are rejected.
    pub fn set_shim_axi(&mut self, axi: dma::ShimAxi) -> Result<(), String> {
        axi.validate_vendor_word5()?;
        if axi == self.shim_axi {
            return Ok(());
        }
        set_shim_axi_txn(&mut self.insts, axi)?;
        self.shim_axi = axi;
        Ok(())
    }
}

/// Build the fresh-context, single-core deployment image and command stream.
/// No NPU access occurs here. Exactly two DDR-patched arguments are required:
/// combined AB input and packed C output. Shapes are multiples8, K64..256,
/// with 1..=MAX_JOBS output microtiles. The first Halo gate should use [0],64,64,64.
pub fn design(cols: &[u32], m: usize, n: usize, k: usize) -> Design {
    use crate::{
        cdo::Cdo,
        regs,
        route::{Circuit, Port, ShimDma},
        txn::Txn,
    };
    use dma::{
        Direction::{Mm2s, S2mm},
        Location, Task,
    };
    assert!(
        cols.len() == 1 && cols[0] < 8,
        "single-core deployment requires one column"
    );
    assert!(m > 0 && m % 8 == 0 && n > 0 && n % 8 == 0);
    let jobs = (m / 8) * (n / 8);
    let program = core_program(k, jobs).finish();
    assert!(program.len() <= 16 * 1024);
    let col = cols[0];
    let core = Location::new(col, 2);
    let mem = Location::new(col, 1);
    let shim = Location::new(col, 0);
    let input_bytes = jobs * 16 * k;
    let output_bytes = jobs * 256;
    let mut cdo = Cdo::new();
    cdo.mask_write(core.address(regs::core::CORE_CONTROL), 1, 0);
    for off in [regs::core::DMA_S2MM_0_CTRL, regs::core::DMA_MM2S_0_CTRL] {
        cdo.mask_write(core.address(off), 2, 2);
    }
    let words: Vec<u32> = program
        .chunks(4)
        .map(|chunk| {
            let mut word = [0; 4];
            word[..chunk.len()].copy_from_slice(chunk);
            u32::from_le_bytes(word)
        })
        .collect();
    cdo.dma_write(core.address(regs::core::PROGRAM_MEMORY), &words);
    for off in [regs::core::DMA_S2MM_0_CTRL, regs::core::DMA_MM2S_0_CTRL] {
        cdo.mask_write(core.address(off), 2, 0);
    }
    cdo.mask_write(core.address(regs::core::CORE_CONTROL), 2, 2)
        .mask_write(core.address(regs::core::CORE_CONTROL), 2, 0)
        .write(core.address(regs::core::CORE_PC), 0);
    for (id, value) in initial_locks().into_iter().enumerate() {
        let (addr, value) = dma::lock_write(core, id as u32, value as u32);
        cdo.write(addr, value);
    }
    for (id, bd) in tile_bds().iter().enumerate() {
        cdo.dma_write(dma::Bd::address(core, id as u32), bd);
    }
    let mm = ShimDma {
        tile: shim,
        direction: Mm2s,
        channel: 0,
    };
    let sm = ShimDma {
        tile: shim,
        direction: S2mm,
        channel: 0,
    };
    for circuit in [
        Circuit {
            tile: shim,
            slave: mm.port(),
            master: Port::North(0),
        },
        Circuit {
            tile: mem,
            slave: Port::South(0),
            master: Port::North(0),
        },
        Circuit {
            tile: core,
            slave: Port::South(0),
            master: Port::Dma(0),
        },
        Circuit {
            tile: core,
            slave: Port::Dma(0),
            master: Port::South(0),
        },
        Circuit {
            tile: mem,
            slave: Port::North(0),
            master: Port::South(0),
        },
        Circuit {
            tile: shim,
            slave: Port::North(0),
            master: sm.port(),
        },
    ] {
        circuit.emit_cdo(&mut cdo);
    }
    mm.emit_cdo(&mut cdo);
    sm.emit_cdo(&mut cdo);
    for (direction, bd) in [(Mm2s, 4), (S2mm, 0)] {
        let task = Task {
            direction,
            channel: 0,
            bd,
            repeat: 1,
            issue_token: false,
        };
        task.emit_cdo(core, &mut cdo);
        task.enable_cdo(core, &mut cdo);
    }
    for (addr, value) in regs::shim_token_route(col) {
        cdo.write(addr, value);
    }
    cdo.mask_write(core.address(regs::core::CORE_CONTROL), 1, 1);
    let mut txn = Txn::aie2p_8col();
    for (id, words, arg) in [(0, input_bytes / 4, 0), (1, output_bytes / 4, 1)] {
        dma::Bd::new(0, words as u32).emit_txn(shim, id, &mut txn);
        txn.ddr_patch(dma::Bd::address(shim, id) + 4, arg, 0);
    }
    txn.mask_write(shim.address(regs::shim::DMA_S2MM_0_CTRL), 0xf00, 0x1f00);
    for (direction, bd, token) in [(S2mm, 1, true), (Mm2s, 0, false)] {
        Task {
            direction,
            channel: 0,
            bd,
            repeat: 1,
            issue_token: token,
        }
        .emit_txn(shim, &mut txn);
    }
    txn.sync(col, 0, 0, 0, 1, 1);
    Design {
        pdi: crate::pdi::build(&cdo.to_words()),
        insts: txn.to_bytes(),
        args: vec![
            ArgSpec {
                bytes: input_bytes,
                kind: ArgKind::In,
            },
            ArgSpec {
                bytes: output_bytes,
                kind: ArgKind::Out,
            },
        ],
        col,
        m,
        n,
        k,
        shim_axi: dma::ShimAxi::default(),
    }
}

/// Set burst/AxCACHE/AxQoS in every shim BD descriptor of the TXN `txn` (layout: `crate::txn`), in place. Returns
/// the number of descriptors patched. Pass 1 validates the entire stream (header, every op's size, and that each
/// BLOCKWRITE touching the row-0 shim BD file `0x1d000..0x1d200` is exactly one aligned 8-word descriptor); only
/// then does pass 2 rewrite word 4 bits `0xc000_0000` (burst) and word 5 bits `0x0ff0_0000` (cache, qos). On
/// `Err` nothing was modified. A stream without any shim descriptor is an error.
pub(crate) fn set_shim_axi_txn(txn: &mut [u8], axi: dma::ShimAxi) -> Result<usize, String> {
    axi.validate_vendor_word5()?;
    walk_shim_bds(txn, None)?;
    match walk_shim_bds(txn, Some(axi))? {
        0 => Err("TXN contains no shim BD descriptor to patch".into()),
        n => Ok(n),
    }
}

fn walk_shim_bds(txn: &mut [u8], patch: Option<dma::ShimAxi>) -> Result<usize, String> {
    use crate::regs::shim::DMA_BD0;
    const BD_BYTES: u32 = 0x20;
    const BD_END: u32 = DMA_BD0 + 16 * BD_BYTES;
    const ROW_MASK: u32 = 0x01f0_0000;
    fn word(b: &[u8], at: usize) -> u32 {
        u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
    }
    let total = txn.len();
    if total < 16 || total % 4 != 0 {
        return Err(format!(
            "malformed TXN: {total} bytes is not a header plus whole words"
        ));
    }
    if txn[0] != 0 || txn[1] != 1 {
        return Err(format!("unsupported TXN version {}.{}", txn[0], txn[1]));
    }
    if word(txn, 12) as usize != total {
        return Err(format!(
            "malformed TXN: header size {} != {total} bytes",
            word(txn, 12)
        ));
    }
    let (mut at, mut ops, mut patched) = (16usize, 0u32, 0usize);
    while at < total {
        let bad = |why: &str| format!("malformed TXN op {ops} at byte {at}: {why}");
        if total - at < 16 {
            return Err(bad("truncated op"));
        }
        let (size, size_word) = match word(txn, at) {
            0 => (24, Some(at + 20)),
            1 => (word(txn, at + 12) as usize, None),
            3 | 4 => (28, Some(at + 24)),
            0x80 => (16, Some(at + 4)),
            0x81 => (48, Some(at + 4)),
            op => return Err(bad(&format!("unsupported opcode {op:#x}"))),
        };
        if size < 16 || size % 4 != 0 || size > total - at {
            return Err(bad(&format!("size {size} invalid or past end of stream")));
        }
        if let Some(w) = size_word {
            if word(txn, w) as usize != size {
                return Err(bad(&format!("embedded size {} != {size}", word(txn, w))));
            }
        }
        if word(txn, at) == 1 {
            let (addr, n) = (word(txn, at + 8), (size - 16) / 4);
            let off = addr & 0x000f_ffff;
            if addr & ROW_MASK == 0
                && n != 0
                && off < BD_END
                && off as usize + 4 * n > DMA_BD0 as usize
            {
                if off < DMA_BD0 || (off - DMA_BD0) % BD_BYTES != 0 || n != 8 {
                    return Err(bad(&format!(
                        "shim BLOCKWRITE at {addr:#x} of {n} words is not one aligned 8-word BD"
                    )));
                }
                if let Some(a) = patch {
                    let (w4, w5) = (at + 32, at + 36);
                    let v4 = word(txn, w4) & !0xc000_0000 | a.burst << 30;
                    let v5 = word(txn, w5) & !0x0ff0_0000 | a.cache << 24 | a.qos << 20;
                    txn[w4..w4 + 4].copy_from_slice(&v4.to_le_bytes());
                    txn[w5..w5 + 4].copy_from_slice(&v5.to_le_bytes());
                }
                patched += 1;
            }
        }
        at += size;
        ops += 1;
    }
    if ops != word(txn, 8) {
        return Err(format!(
            "malformed TXN: header says {} ops, found {ops}",
            word(txn, 8)
        ));
    }
    Ok(patched)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn gemm_i8_oracle_bytes() {
        // Vendor 032eb944... core0,2 .text 0x6a0 and 0x690, independently disassembled.
        assert_eq!(
            matrix_mac(0, Some(0), 8, 1, 2).encode(),
            [0x08, 0x21, 0x10, 0x10]
        );
        assert_eq!(
            matrix_mac(4, Some(4), 8, 10, 2).encode(),
            [0x08, 0x41, 0x91, 0x14]
        );
        assert_eq!(
            store_acc_quarter(2, 0, 6).encode(),
            [0x98, 0x06, 0x1d, 0x0e]
        );
        let Some(corpus) = crate::vendor_corpus() else {
            return;
        };
        let path =
            corpus.join("cache/032eb944d6cdbf2df2d4b8d9/elfs_main_core_0_2/elfs_main_core_0_2.elf");
        let Ok(bytes) = std::fs::read(&path) else {
            eprintln!("skip optional corpus check: {} absent", path.display());
            return;
        };
        for needle in [
            matrix_mac(0, Some(0), 8, 1, 2).encode(),
            store_acc_quarter(2, 0, 6).encode(),
        ] {
            assert!(bytes.windows(needle.len()).any(|w| w == needle));
        }
    }
    #[test]
    fn gemm_i8_signed_packing() {
        let a: Vec<i8> = (0..64 * 64).map(|i| (i * 17 + 128) as i8).collect();
        let b: Vec<i8> = (0..64 * 64).map(|i| (i * 31 + 255) as i8).collect();
        let packed = pack_job(&a, &b, 64, 64, 64, 3, 5);
        let expected = cpu_reference(&a, &b, 64, 64, 64);
        for row in 0..8 {
            for col in 0..8 {
                let mut acc = 0i32;
                for block in 0..8 {
                    for kk in 0..8 {
                        acc += packed.a[block * 64 + row * 8 + kk] as i8 as i32
                            * packed.b[block * 64 + kk * 8 + col] as i8 as i32;
                    }
                }
                assert_eq!(acc, expected[(24 + row) * 64 + 40 + col]);
            }
        }
    }
    #[test]
    fn gemm_i8_peak_estimate() {
        assert_eq!(peak_ops_per_second(1_000_000_000), 32_768_000_000_000);
        eprintln!("ESTIMATE: 512 MAC/core/cycle x32 cores x1GHz = 16.384 TMAC/s = 32.768 TOP/s; DMA/stalls excluded");
    }
}
