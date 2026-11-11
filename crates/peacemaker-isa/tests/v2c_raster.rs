// SPDX-License-Identifier: Apache-2.0
//! CPU-only interpretation of the emitted workgroup remapping instructions.
//! This checks the actual generator, not a duplicate production mapping helper.
use peacemaker_isa::kernels::iu4_v2c::{emit, Epi, Spec};
use peacemaker_isa::Arch;

#[derive(Clone, Copy)]
enum Operand {
    Register(usize),
    Immediate(u32),
}

impl Operand {
    fn parse(text: &str) -> Self {
        if let Some(index) = text.strip_prefix('s') {
            Self::Register(index.parse().unwrap())
        } else if let Some(hex) = text.strip_prefix("0x") {
            Self::Immediate(u32::from_str_radix(hex, 16).unwrap())
        } else {
            Self::Immediate(text.parse::<i64>().unwrap() as u32)
        }
    }

    fn read(self, registers: &[u32; 64]) -> u32 {
        match self {
            Self::Register(index) => registers[index],
            Self::Immediate(value) => value,
        }
    }
}

/// Workgroup ids: gfx1100 pads the two user SGPRs to `user + system (X, Y) = 16` (LLVM `FeatureUserSGPRInit16Bug`).
const WGX: usize = Arch::Gfx1100.declared_user_sgprs(2, 2) as usize;
const WGY: usize = WGX + 1;

struct Mapping(Vec<(String, Vec<Operand>)>, usize);

impl Mapping {
    fn emitted(epi: Epi) -> Self {
        let text = emit(Spec {
            arch: Arch::Gfx1100,
            epi,
        })
        .unwrap()
        .s_text;
        let start = if epi == Epi::Silu {
            "s_and_b32 s4, s30,"
        } else {
            "s_and_b32 s4, s62,"
        };
        let lines: Vec<_> = text.lines().map(str::trim).collect();
        let first = lines
            .iter()
            .position(|line| line.starts_with(start))
            .unwrap();
        let last = lines[first..]
            .iter()
            .position(|line| *line == format!("s_cselect_b32 s{WGY}, s6, s{WGY}"))
            .unwrap()
            + first;
        Self(
            lines[first..=last]
                .iter()
                .map(|line| {
                    let (op, args) = line.split_once(' ').unwrap();
                    (
                        op.to_owned(),
                        args.split(", ").map(Operand::parse).collect(),
                    )
                })
                .collect(),
            if epi == Epi::Silu { 30 } else { 62 },
        )
    }

    fn run(&self, x: u32, y: u32, m: u32, n: u32) -> (u32, u32) {
        let mut registers = std::array::from_fn(|index| 0xa500_0000 + index as u32);
        registers[WGX] = x;
        registers[WGY] = y;
        registers[self.1] = m;
        registers[16] = n;
        let original = registers;
        let mut scc = None;
        for (op, args) in &self.0 {
            if op == "s_cmp_lt_u32" || op == "s_cmp_eq_u32" {
                let (a, b) = (args[0].read(&registers), args[1].read(&registers));
                scc = Some(if op == "s_cmp_lt_u32" { a < b } else { a == b });
                continue;
            }
            let Operand::Register(destination) = args[0] else {
                panic!("non-register destination");
            };
            let (a, b) = (args[1].read(&registers), args[2].read(&registers));
            registers[destination] = match op.as_str() {
                "s_and_b32" => a & b,
                "s_or_b32" => a | b,
                "s_cselect_b32" => {
                    if scc.unwrap() {
                        a
                    } else {
                        b
                    }
                }
                _ => panic!("unexpected mapping instruction: {op}"),
            };
        }
        // Only workgroup IDs and the four already-reserved temporaries change.
        assert_eq!(&registers[..2], &original[..2]);
        assert_eq!(&registers[8..WGX], &original[8..WGX]);
        assert_eq!(&registers[WGY + 1..], &original[WGY + 1..]);
        (registers[WGX], registers[WGY])
    }
}

#[test]
fn emitted_raster_is_a_bijection_and_preserves_fallback_grids() {
    for epi in Epi::ALL {
        let mapping = Mapping::emitted(epi);
        let row_tile = if epi == Epi::Silu { 64 } else { 128 };
        for m in [128, 256, 384, 512, 640, 6272, 10240, 12288, 17408] {
            for n in [128, 512, 2048, 3584, 3968, 4096, 4224, 4480, 4608, 8192] {
                let (cols, rows) = (n / 128, m / row_tile);
                let active = n >= 4096 && cols % 4 == 0 && rows % 4 == 0;
                let mut seen = vec![false; (cols * rows) as usize];
                for y in 0..rows {
                    for x in 0..cols {
                        let (mx, my) = mapping.run(x, y, m, n);
                        let expected = if active {
                            ((x / 4) * 4 + y % 4, (y / 4) * 4 + x % 4)
                        } else {
                            (x, y)
                        };
                        assert_eq!((mx, my), expected, "{epi:?} M{m} N{n} ({x},{y})");
                        assert!(mx < cols && my < rows);
                        let index = (my * cols + mx) as usize;
                        assert!(!seen[index], "duplicate output tile");
                        seen[index] = true;
                        assert_eq!(mapping.run(mx, my, m, n), (x, y), "not involutive");
                    }
                }
                assert!(
                    seen.into_iter().all(|visited| visited),
                    "missing output tile"
                );
            }
        }
    }
}
