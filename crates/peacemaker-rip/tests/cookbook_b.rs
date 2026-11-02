//! Execute the actual Markdown kernels; no duplicate .rip fixtures.
#[path = "support/emu.rs"]
mod emu;
use emu::{f32s, u16s, u32s, Buffer, Case, Karg, Kernel, ARCHES};

#[test]
fn device_loop_runtime_chunks_and_carried() {
    let (source, variant) = emu::recipe("06-device-loop.md");
    for arch in ARCHES {
        for variant in [variant.as_str(), "copy_carried"] {
            let kernel = Kernel::compile(&source, variant, arch);
            for n in [0, 1, 17, 31, 32, 33, 65, 127, 1025, 100_000] {
                let input: Vec<u32> = (0..n)
                    .map(|i: u32| i.wrapping_mul(0x9e37_79b9) ^ 0x1234_5678)
                    .collect();
                kernel.run(&Case {
                    name: format!("{} {variant} n={n}", arch.name()),
                    bufs: vec![
                        Buffer::input("a", u32s(&input)),
                        Buffer::output("out", u32s(&input)),
                    ],
                    kernarg: Karg::new(24).ptr(0, 0).ptr(8, 1).u32(16, n).finish(),
                    grid: [1, 1, 1],
                    block: [32, 1, 1],
                });
            }
        }
    }
}

#[test]
fn uniform_branch_zero_nonzero_flag_and_early_exit() {
    for arch in ARCHES {
        let kernel = Kernel::recipe("07-uniform-branch.md", arch);
        for n in [0, 1, 17, 31, 32] {
            for flag in [0, 1, 7, u32::MAX] {
                let a: Vec<u32> = (0..n).map(|i: u32| u32::MAX.wrapping_sub(i * 13)).collect();
                let b: Vec<u32> = (0..n).map(|i| 31 + i * 17).collect();
                let expected: Vec<u32> = a
                    .iter()
                    .zip(&b)
                    .map(|(&a, &b)| if flag != 0 { a.wrapping_add(b) } else { a })
                    .collect();
                kernel.run(&Case {
                    name: format!("{} n={n} flag={flag}", arch.name()),
                    bufs: vec![
                        Buffer::input("a", u32s(&a)),
                        Buffer::input("b", u32s(&b)),
                        Buffer::output("out", u32s(&expected)),
                    ],
                    kernarg: Karg::new(32)
                        .ptr(0, 0)
                        .ptr(8, 1)
                        .ptr(16, 2)
                        .u32(24, n)
                        .u32(28, flag)
                        .finish(),
                    grid: [1, 1, 1],
                    block: [32, 1, 1],
                });
            }
        }
    }
}

#[test]
fn bf16_wmma_column_major_a_row_major_b_chain_and_raw() {
    let (source, variant) = emu::recipe("08-wmma.md");
    for arch in ARCHES {
        for variant in [variant.as_str(), "bf16_raw"] {
            let kernel = Kernel::compile(&source, variant, arch);
            assert!(kernel.assembly.contains("v_wmma_f32_16x16x16_bf16"));
            for pattern in 0..4 {
                let mut a = vec![0f32; 256];
                let mut b = vec![0f32; 256];
                for r in 0..16 {
                    for k in 0..16 {
                        a[r + 16 * k] = match pattern {
                            0 => {
                                if r == k {
                                    1.0
                                } else {
                                    0.0
                                }
                            }
                            1 => ((r * 7 + k * 3) % 17) as f32 - 8.0,
                            2 => (((r * 5 + k * 11) % 19) as f32 - 9.0) * 0.25,
                            _ => {
                                if (r + k) % 5 == 0 {
                                    -0.5
                                } else {
                                    0.0
                                }
                            }
                        };
                        b[k * 16 + r] = match pattern {
                            0 => ((k * 13 + r * 5) % 23) as f32 - 11.0,
                            1 => ((k * 11 + r * 3) % 13) as f32 - 6.0,
                            2 => (((k * 7 + r * 3) % 17) as f32 - 8.0) * 0.5,
                            _ => {
                                if (k * 3 + r) % 7 == 0 {
                                    2.0
                                } else {
                                    0.0
                                }
                            }
                        };
                    }
                }
                let mut expected = vec![0f32; 256];
                for r in 0..16 {
                    for c in 0..16 {
                        for k in 0..16 {
                            expected[r * 16 + c] += a[r + 16 * k] * b[k * 16 + c];
                        }
                    }
                }
                // All inputs above have exact BF16 representations.
                let bf16 = |values: &[f32]| {
                    u16s(
                        &values
                            .iter()
                            .map(|v| (v.to_bits() >> 16) as u16)
                            .collect::<Vec<_>>(),
                    )
                };
                let output = if arch == emu::Arch::Gfx1151 {
                    Buffer::near_output("d", f32s(&expected), 1e-3)
                } else {
                    Buffer::output("d", f32s(&expected))
                };
                kernel.run(&Case {
                    name: format!("{} {variant} pattern={pattern}", arch.name()),
                    bufs: vec![
                        Buffer::input("a", bf16(&a)),
                        Buffer::input("b", bf16(&b)),
                        output,
                    ],
                    kernarg: Karg::new(24).ptr(0, 0).ptr(8, 1).ptr(16, 2).finish(),
                    grid: [1, 1, 1],
                    block: [32, 1, 1],
                });
            }
        }
    }
}

#[test]
fn kernarg_owners_automatic_readiness_and_explicit_store_ordering() {
    for arch in ARCHES {
        let kernel = Kernel::recipe("09-kernarg-loads-and-waits.md", arch);
        let (smem_wait, vmem_wait, store_wait) = if arch.gfx12() {
            (
                "s_wait_kmcnt 0x0",
                "s_wait_loadcnt 0x0",
                "s_wait_storecnt 0x0",
            )
        } else {
            (
                "s_waitcnt lgkmcnt(0)",
                "s_waitcnt vmcnt(0)",
                "s_waitcnt_vscnt null, 0x0",
            )
        };
        let assembly = &kernel.assembly;
        let load_args = assembly.find("s_load_b256").unwrap();
        let smem_ready = assembly.find(smem_wait).unwrap();
        let use_args = assembly.find("s_lshl_b32 s5, s14, 2").unwrap();
        assert!(load_args < smem_ready && smem_ready < use_args);
        let load_data = assembly.find("buffer_load_b32").unwrap();
        let vmem_ready = assembly.find(vmem_wait).unwrap();
        let use_data = assembly.find("v_mul_f32_e32 v3, s4, v3").unwrap();
        assert!(load_data < vmem_ready && vmem_ready < use_data);
        assert!(assembly.find("buffer_store_b32").unwrap() < assembly.find(store_wait).unwrap());
        for (n, first) in [
            (0u32, 0u32),
            (1, 0),
            (17, 0),
            (32, 0),
            (33, 0),
            (129, 0),
            (103, 5),
            (77, 77),
        ] {
            for alpha in [0.0f32, -1.5, 0.25, 2.0] {
                let a: Vec<f32> = (0..n).map(|i| (i as f32 - 20.0) * 0.25).collect();
                let expected: Vec<f32> = a
                    .iter()
                    .enumerate()
                    .map(|(i, &a)| {
                        if (i as u32) < first {
                            -999.0
                        } else {
                            a * alpha + i as f32
                        }
                    })
                    .collect();
                let mut output = Buffer::output("out", f32s(&expected));
                output.bytes = f32s(&vec![-999.0; n as usize]);
                kernel.run(&Case {
                    name: format!("{} n={n} first={first} alpha={alpha}", arch.name()),
                    bufs: vec![
                        Buffer::input("a", f32s(&a)),
                        output,
                        Buffer::input("reserved", u32s(&[0x8765_4321, 0xfedc_ba98])),
                    ],
                    kernarg: Karg::new(40)
                        .ptr(0, 0)
                        .ptr(8, 1)
                        .ptr(16, 2)
                        .u32(24, n)
                        .u32(28, first)
                        .f32(32, alpha)
                        .finish(),
                    grid: [((n - first + 31) / 32).max(1), 1, 1],
                    block: [32, 1, 1],
                });
            }
        }
    }
}
