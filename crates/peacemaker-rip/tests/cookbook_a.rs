//! Execute the actual cookbook Markdown kernels, never a copied implementation.
#[path = "support/emu.rs"]
mod emu;
use emu::{f32s, u32s, Buffer, Case, Karg, Kernel, ARCHES};

#[test]
fn minimal_fill() {
    for arch in ARCHES {
        let kernel = Kernel::recipe("01-minimal-kernel.md", arch);
        for n in [0u32, 1, 31, 32, 33, 63, 64, 65, 127, 129, 257] {
            for (k, c) in [(0u32, 7u32), (3, 19), (u32::MAX, 0xffff_ff80)] {
                let expected: Vec<u32> =
                    (0..n).map(|i| i.wrapping_mul(k).wrapping_add(c)).collect();
                kernel.run(&Case {
                    name: format!("n={n},k={k},c={c}"),
                    bufs: vec![Buffer::output("out", u32s(&expected))],
                    kernarg: Karg::new(24)
                        .ptr(0, 0)
                        .u32(8, n)
                        .u32(12, k)
                        .u32(16, c)
                        .finish(),
                    // Even n=0 launches a wave: no lane may write the guard.
                    grid: [n.div_ceil(64).max(1), 1, 1],
                    block: [64, 1, 1],
                });
            }
        }
    }
}

#[test]
fn elementwise_descriptor_bounds() {
    for arch in ARCHES {
        let kernel = Kernel::recipe("02-elementwise-bounds.md", arch);
        for n in [0u32, 1, 7, 31, 32, 33, 63, 64, 65, 130, 257] {
            // Exact binary fractions keep the Rust and ISA subtraction bitwise.
            let a: Vec<f32> = (0..n).map(|i| (i as i32 % 23 - 11) as f32 * 0.5).collect();
            let b: Vec<f32> = (0..n).map(|i| (i as i32 % 17 - 8) as f32 * 0.25).collect();
            let expected: Vec<f32> = a.iter().zip(&b).map(|(a, b)| a - b).collect();
            kernel.run(&Case {
                name: format!("n={n}"),
                bufs: vec![
                    Buffer::input("a", f32s(&a)),
                    Buffer::input("b", f32s(&b)),
                    Buffer::output("c", f32s(&expected)),
                ],
                kernarg: Karg::new(32)
                    .ptr(0, 0)
                    .ptr(8, 1)
                    .ptr(16, 2)
                    .u32(24, n)
                    .finish(),
                // Extra workgroup exercises a wholly out-of-range descriptor access.
                grid: [n.div_ceil(64) + 1, 1, 1],
                block: [64, 1, 1],
            });
        }
    }
}

#[test]
fn two_dimensional_workgroup_ids() {
    for arch in ARCHES {
        let kernel = Kernel::recipe("03-workgroup-ids.md", arch);
        for (gx, gy) in [(1u32, 1u32), (5, 1), (1, 4), (3, 5), (7, 3)] {
            let expected: Vec<u32> = (0..gy)
                .flat_map(|y| (0..gx).map(move |x| x | (y << 16)))
                .collect();
            kernel.run(&Case {
                name: format!("gx={gx},gy={gy}"),
                bufs: vec![Buffer::output("out", u32s(&expected))],
                kernarg: Karg::new(16).ptr(0, 0).u32(8, gx).finish(),
                grid: [gx, gy, 1],
                block: [32, 1, 1],
            });
        }
    }
}

#[test]
fn wave_unsigned_minimum() {
    for arch in ARCHES {
        let kernel = Kernel::recipe("04-wave-reduction.md", arch);
        for groups in [1u32, 2, 7] {
            for minimum_lane in [0usize, 1, 7, 15, 16, 23, 31] {
                let input: Vec<u32> = (0..groups as usize)
                    .flat_map(|group| {
                        (0..32usize).map(move |lane| {
                            if lane == minimum_lane {
                                17 + group as u32
                            } else {
                                0x8000_0000 + (group * 97 + (31 - lane) * 13) as u32
                            }
                        })
                    })
                    .collect();
                let expected: Vec<u32> = input
                    .chunks_exact(32)
                    .map(|wave| *wave.iter().min().unwrap())
                    .collect();
                kernel.run(&Case {
                    name: format!("groups={groups},min_lane={minimum_lane}"),
                    bufs: vec![
                        Buffer::input("input", u32s(&input)),
                        Buffer::output("output", u32s(&expected)),
                    ],
                    kernarg: Karg::new(16).ptr(0, 0).ptr(8, 1).finish(),
                    grid: [groups, 1, 1],
                    block: [32, 1, 1],
                });
            }
            // Values all above INT_MAX catch a signed reduction masquerading as u32.
            let input: Vec<u32> = (0..groups * 32).map(|i| u32::MAX - i * 3).collect();
            let expected: Vec<u32> = input
                .chunks_exact(32)
                .map(|wave| *wave.iter().min().unwrap())
                .collect();
            kernel.run(&Case {
                name: format!("groups={groups},high_bits"),
                bufs: vec![
                    Buffer::input("input", u32s(&input)),
                    Buffer::output("output", u32s(&expected)),
                ],
                kernarg: Karg::new(16).ptr(0, 0).ptr(8, 1).finish(),
                grid: [groups, 1, 1],
                block: [32, 1, 1],
            });
        }
    }
}

#[test]
fn two_wave_lds_reverse() {
    for arch in ARCHES {
        let kernel = Kernel::recipe("05-lds-handoff.md", arch);
        assert!(kernel.assembly.contains("ds_store_b32"));
        assert!(kernel.assembly.contains("ds_load_b32"));
        assert!(kernel.assembly.contains("s_barrier"));
        for groups in [1u32, 2, 5] {
            let input: Vec<u32> = (0..groups * 64)
                .map(|i| i.wrapping_mul(0x9e37_79b9) ^ 0xa513_76cd)
                .collect();
            let expected: Vec<u32> = input
                .chunks_exact(64)
                .flat_map(|block| block.iter().rev().copied())
                .collect();
            kernel.run(&Case {
                name: format!("groups={groups}"),
                bufs: vec![
                    Buffer::input("input", u32s(&input)),
                    Buffer::output("output", u32s(&expected)),
                ],
                kernarg: Karg::new(16).ptr(0, 0).ptr(8, 1).finish(),
                grid: [groups, 1, 1],
                block: [64, 1, 1],
            });
        }
    }
}
