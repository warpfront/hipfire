use crate::{Builder, ledger::Counter};
use peacemaker_author::{Free, LdsWrite, MmaIu4, Pending, Published, Ring, Workgroup, Writing, rotate};

/// Epoch distance and outstanding-load wait depth are part of the schedule type.
pub struct Prefetch<const D: usize, const WAIT: u8 = 0>;
/// A packet cannot be consumed before its declared epoch distance.
pub struct InFlight<const D: usize, P> { epoch: usize, packet: P }
impl<const D: usize, const WAIT: u8> Prefetch<D, WAIT> {
    pub fn issue<P>(epoch: usize, packet: P) -> InFlight<D, P> { InFlight { epoch, packet } }
    pub fn consume<P>(epoch: usize, packet: InFlight<D, P>) -> Result<P, String> {
        if epoch.checked_sub(packet.epoch) != Some(D) { return Err("prefetch epoch distance differs from schedule type".into()) }
        Ok(packet.packet)
    }
    pub fn wait_load(b: &mut Builder) -> Result<(), String> {
        b.wait(if b.spec.arch.gfx12() { Counter::Load } else { Counter::Vm }, WAIT)
    }
}
impl<const WAIT: u8> Prefetch<1, WAIT> {
    /// Finish next-slot publication and rotate exactly one epoch ahead.
    /// Uses the existing store drain and barrier, with no extra instruction.
    pub fn rotate<T: MmaIu4, A: 'static, X: 'static>(
        wg: &mut Workgroup<T, Builder>,
        a: (Ring<A, Published, Writing>, Pending<T::Waits, LdsWrite<A>>),
        x: (Ring<X, Published, Writing>, Pending<T::Waits, LdsWrite<X>>),
    ) -> Result<(Ring<A, Published, Free>, Ring<X, Published, Free>), String> {
        let ((ra, pa), (rx, px)) = (a, x);
        let (da, dx) = wg.wait_all((pa, px))?;
        wg.barrier((rotate(ra, da), rotate(rx, dx)))
    }
}
