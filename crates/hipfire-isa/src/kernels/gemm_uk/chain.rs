use crate::{Arch, Builder, V, insn::{Instruction, Wmma}};
use peacemaker_author::{Gfx1100, Gfx1151, Gfx1201, MmaIu4, Wave};
use std::marker::PhantomData;

mod sealed { pub trait Kind {} }
/// Operand shape and lowering of an integer MMA; independent of chain count.
pub trait MmaKind<T: MmaIu4>: sealed::Kind {
    type Input: Copy;
    fn instruction(arch: Arch, dst: V<8>, a: Self::Input, b: Self::Input, seed: V<8>) -> Instruction;
}
pub enum Iu4 {}
pub enum Iu8 {}
impl sealed::Kind for Iu4 {}
impl sealed::Kind for Iu8 {}
impl<T: MmaIu4> MmaKind<T> for Iu4 {
    type Input = V<2>;
    fn instruction(arch: Arch, dst: V<8>, a: V<2>, b: V<2>, seed: V<8>) -> Instruction {
        Wmma::iu4(arch, dst, a, b, Some(seed))
    }
}
macro_rules! iu8 {
    ($target:ty, $width:literal) => {
        impl MmaKind<$target> for Iu8 {
            type Input = V<$width>;
            fn instruction(_: Arch, dst: V<8>, a: Self::Input, b: Self::Input, seed: V<8>) -> Instruction {
                Instruction::new(format!("v_wmma_i32_16x16x16_iu8 {}, {}, {}, {} neg_lo:[1,1,0]", dst.reg(), a.reg(), b.reg(), seed.reg()),
                    vec![dst.reg()], vec![a.reg(), b.reg(), seed.reg()])
            }
        }
    };
}
iu8!(Gfx1100, 4);
iu8!(Gfx1151, 4);
iu8!(Gfx1201, 2);
/// N disjoint output fragments, seeded at each epoch's first K step.
/// Steps always accumulate each output in caller K order, never reassociate.
pub struct Chain<const N: usize, K: sealed::Kind = Iu4> {
    outputs: [V<8>; N], seed: V<8>, kind: PhantomData<K>,
}
impl<const N: usize, K: sealed::Kind> Chain<N, K> {
    pub fn new(outputs: [V<8>; N], seed: V<8>) -> Result<Self, String> {
        if N == 0 { return Err("a chain needs an output".into()) }
        if u16::from(seed.base()) + 8 > 256 { return Err("chain seed exceeds VGPR file".into()) }
        for (i, output) in outputs.iter().enumerate() {
            let start = u16::from(output.base());
            if start + 8 > 256 { return Err("chain output exceeds VGPR file".into()) }
            if N > 1 && output.reg().overlaps(seed.reg()) { return Err("shared chain seed overlaps an output".into()) }
            if outputs[..i].iter().any(|other| {
                let o = u16::from(other.base());
                start < o + 8 && o < start + 8
            }) { return Err("independent chains overlap".into()) }
        }
        Ok(Self { outputs, seed, kind: PhantomData })
    }
    pub fn step<T: MmaIu4>(&self, w: &mut Wave<T, Builder>, a: <K as MmaKind<T>>::Input,
        b: [<K as MmaKind<T>>::Input; N], first: bool) -> Result<(), String> where K: MmaKind<T> {
        let builder = w.isa();
        for (dst, b) in self.outputs.iter().copied().zip(b) {
            builder.push(<K as MmaKind<T>>::instruction(builder.spec.arch, dst, a, b, if first { self.seed } else { dst }))?;
        }
        Ok(())
    }
    /// Same bank-preserving fold DAG as the dense IU4 kernels.
    pub fn fold(&self, b: &mut Builder, sums: [u8; N], scales: u8, factors: [u8; N], tmp: u8) -> Result<(), String> {
        for ((dst, sum), factor) in self.outputs.iter().zip(sums).zip(factors) {
            crate::kernels::iu4_fold::fold_pass(b, dst.base(), sum, scales, factor, tmp)?;
        }
        Ok(())
    }
}
