#![allow(dead_code)]
//! CPU-only cookbook harness: compile the Markdown source, assemble, lift, execute.
use peacemaker_ir::emu::{Emulator, Launch, Memory};
use peacemaker_ir::inst::{Frontend, Program};
pub use peacemaker_isa::Arch;

pub const ARCHES: [Arch; 2] = [Arch::Gfx1151, Arch::Gfx1201];
pub const KERNARG_BASE: u64 = 0x8000_0000;
const GUARD: usize = 256;
pub fn base_of(index: usize) -> u64 {
    0x1_0000_0000 + index as u64 * 0x1000_0000
}
pub fn u32s(values: &[u32]) -> Vec<u8> {
    values.iter().flat_map(|x| x.to_le_bytes()).collect()
}
pub fn f32s(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|x| x.to_le_bytes()).collect()
}
pub fn u16s(values: &[u16]) -> Vec<u8> {
    values.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Extract the single complete kernel from a named cookbook file; no copied fixtures.
pub fn recipe(file: &str) -> (String, String) {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("docs/cookbook")
        .join(file);
    let markdown =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut blocks = Vec::new();
    let mut lines = markdown.lines();
    while let Some(line) = lines.next() {
        if let Some(info) = line.trim().strip_prefix("```rip") {
            let variant = info
                .split_whitespace()
                .find_map(|w| w.strip_prefix("variant="))
                .expect("recipe must declare variant=NAME")
                .to_string();
            let mut source = String::new();
            let mut closed = false;
            for body in lines.by_ref() {
                if body.trim() == "```" {
                    closed = true;
                    break;
                }
                source.push_str(body);
                source.push('\n');
            }
            assert!(closed, "{}: unclosed rip block", path.display());
            blocks.push((source, variant));
        }
    }
    assert_eq!(
        blocks.len(),
        1,
        "{}: expected exactly one complete rip kernel",
        path.display()
    );
    blocks.pop().unwrap()
}

pub struct Kernel {
    program: Program,
    symbol: String,
    pub assembly: String,
}
impl Kernel {
    pub fn compile(source: &str, variant: &str, arch: Arch) -> Self {
        let emitted = peacemaker_rip::compile(source, arch, variant)
            .unwrap_or_else(|e| panic!("{} {variant}: compile: {e}", arch.name()));
        let symbol = peacemaker_rip::symbol_of(&emitted).unwrap();
        let elf = peacemaker_isa::native::assemble(&emitted.s_text, arch)
            .unwrap_or_else(|e| panic!("{} {variant}: assemble: {e}", arch.name()));
        let program = peacemaker_lift::lift_object(
            &elf,
            peacemaker_lift::Options {
                frontend: Frontend::Builder,
            },
        )
        .unwrap_or_else(|e| panic!("{} {variant}: lift: {e:?}", arch.name()))
        .program;
        Self {
            program,
            symbol,
            assembly: emitted.s_text,
        }
    }
    pub fn recipe(file: &str, arch: Arch) -> Self {
        let (source, variant) = recipe(file);
        Self::compile(&source, &variant, arch)
    }
    pub fn run(&self, case: &Case) {
        let emulator = Emulator::new(&self.program, &self.symbol).unwrap();
        let mut memory = Memory::default();
        for (index, buffer) in case.bufs.iter().enumerate() {
            let mut bytes = buffer.bytes.clone();
            if buffer.expected.is_some() {
                bytes.extend_from_slice(&[0xAA; GUARD]);
            }
            memory
                .map(buffer.name.to_string(), base_of(index), bytes)
                .unwrap();
        }
        memory
            .map("kernarg".into(), KERNARG_BASE, case.kernarg.clone())
            .unwrap();
        for z in 0..case.grid[2] {
            for y in 0..case.grid[1] {
                for x in 0..case.grid[0] {
                    let launch = Launch {
                        kernarg_ptr: KERNARG_BASE,
                        workgroup: [x, y, z],
                        block: case.block,
                        dynamic_lds: 0,
                    };
                    emulator
                        .run_workgroup(&launch, &mut memory)
                        .unwrap_or_else(|e| {
                            panic!("{} case {} wg [{x},{y},{z}]: {e}", self.symbol, case.name)
                        });
                }
            }
        }
        for (index, buffer) in case.bufs.iter().enumerate() {
            let actual = &memory.regions[index].bytes;
            match &buffer.expected {
                None => assert_eq!(
                    actual, &buffer.bytes,
                    "{} {}: input {} changed",
                    self.symbol, case.name, buffer.name
                ),
                Some(expected) => {
                    assert_eq!(
                        expected.len(),
                        buffer.bytes.len(),
                        "output expected length differs"
                    );
                    let output = &actual[..expected.len()];
                    if let Some(tolerance) = buffer.tolerance {
                        assert_eq!(expected.len() % 4, 0);
                        for (element, (a, b)) in output
                            .chunks_exact(4)
                            .zip(expected.chunks_exact(4))
                            .enumerate()
                        {
                            let got = f32::from_le_bytes(a.try_into().unwrap());
                            let want = f32::from_le_bytes(b.try_into().unwrap());
                            assert!(
                                got.is_finite()
                                    && (got - want).abs() <= tolerance * want.abs().max(1.0),
                                "{} {}: {}[{element}] got {got}, expected {want}",
                                self.symbol,
                                case.name,
                                buffer.name
                            );
                        }
                    } else {
                        assert_eq!(
                            output, expected,
                            "{} {}: output {} mismatch",
                            self.symbol, case.name, buffer.name
                        );
                    }
                    assert!(
                        actual[expected.len()..].iter().all(|&b| b == 0xAA),
                        "{} {}: output {} guard overwritten",
                        self.symbol,
                        case.name,
                        buffer.name
                    );
                }
            }
        }
    }
}

pub struct Buffer {
    pub name: &'static str,
    pub bytes: Vec<u8>,
    pub expected: Option<Vec<u8>>,
    pub tolerance: Option<f32>,
}
impl Buffer {
    pub fn input(name: &'static str, bytes: Vec<u8>) -> Self {
        Self {
            name,
            bytes,
            expected: None,
            tolerance: None,
        }
    }
    pub fn output(name: &'static str, expected: Vec<u8>) -> Self {
        Self {
            name,
            bytes: vec![0xAA; expected.len()],
            expected: Some(expected),
            tolerance: None,
        }
    }
    pub fn near_output(name: &'static str, expected: Vec<u8>, tolerance: f32) -> Self {
        Self {
            tolerance: Some(tolerance),
            ..Self::output(name, expected)
        }
    }
}
pub struct Case {
    pub name: String,
    pub bufs: Vec<Buffer>,
    pub kernarg: Vec<u8>,
    pub grid: [u32; 3],
    pub block: [u32; 3],
}
pub struct Karg {
    bytes: Vec<u8>,
}
impl Karg {
    pub fn new(size: usize) -> Self {
        Self {
            bytes: vec![0; size],
        }
    }
    fn put(mut self, offset: usize, bytes: &[u8]) -> Self {
        self.bytes[offset..offset + bytes.len()].copy_from_slice(bytes);
        self
    }
    pub fn ptr(self, offset: usize, index: usize) -> Self {
        self.put(offset, &base_of(index).to_le_bytes())
    }
    pub fn u32(self, offset: usize, value: u32) -> Self {
        self.put(offset, &value.to_le_bytes())
    }
    pub fn f32(self, offset: usize, value: f32) -> Self {
        self.put(offset, &value.to_le_bytes())
    }
    pub fn finish(self) -> Vec<u8> {
        self.bytes
    }
}
