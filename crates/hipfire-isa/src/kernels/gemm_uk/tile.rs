use crate::{Builder, insn::{Instruction, MemoryClass}};
use peacemaker_author::{LdsRegion, MmaIu4, Published, Ring, State, Wave};
use std::marker::PhantomData;

/// Fragment-major LDS layout: K slices within row panels, then lane rows.
#[derive(Clone, Copy, Debug)]
pub struct FragmentLayout { pub rows: u32, pub k_slices: u32, pub row_bytes: u32 }
impl FragmentLayout {
    pub fn offset(self, row: u32, slice: u32) -> Result<u32, String> {
        if row >= self.rows || slice >= self.k_slices { return Err("fragment coordinate outside tile".into()) }
        let element = (u64::from(row / 16) * u64::from(self.k_slices) + u64::from(slice)) * 16 + u64::from(row % 16);
        element.checked_mul(u64::from(self.row_bytes)).and_then(|n| u32::try_from(n).ok()).ok_or_else(|| "fragment offset overflow".into())
    }
}
pub struct RegisterDirect;
pub struct Lds<const D: usize>;
/// Loader policy and fragment layout without register allocation or scheduling.
pub struct Tile<T: MmaIu4, L> { pub layout: FragmentLayout, policy: PhantomData<(T, L)> }
impl<T: MmaIu4, L> Tile<T, L> {
    pub fn new(layout: FragmentLayout) -> Self { Self { layout, policy: PhantomData } }
}
impl<T: MmaIu4> Tile<T, RegisterDirect> {
    pub fn load(w: &mut Wave<T, Builder>, instruction: Instruction) -> Result<(), String> {
        if instruction.memory != Some(MemoryClass::VmemLoad) { return Err("register-direct tile needs a global load".into()) }
        w.isa().push(instruction)
    }
}
impl<T: MmaIu4, const D: usize> Tile<T, Lds<D>> {
    /// A depth-D ring uses one independently published region per slot.
    /// Publication/retirement stay under the author's barrier typestate.
    pub fn load_slot<R: 'static>(w: &mut Wave<T, Builder>, slots: &[LdsRegion<R, Published>; D], epoch: usize, instruction: Instruction) -> Result<(), String> {
        if D == 0 { return Err("LDS ring depth must be positive".into()) }
        w.ds_load(&slots[epoch % D], instruction)
    }
}
impl<T: MmaIu4> Tile<T, Lds<2>> {
    /// Native two-role author ring; carries pending writes independently.
    pub fn load_cur<R: 'static, N: State>(w: &mut Wave<T, Builder>, ring: &Ring<R, Published, N>, instruction: Instruction) -> Result<(), String> {
        w.ds_load_cur(ring, instruction)
    }
}
