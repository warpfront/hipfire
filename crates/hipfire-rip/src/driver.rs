//! Object-safe face of `peacemaker_author::runtime::Driver`.
//!
//! The driver is generic over its target and borrows the builder, and its
//! control-flow methods hand callbacks a fresh driver over the re-entered
//! backend. [`DynDriver`] erases the target and the borrow so the evaluator
//! can run script closures as those callbacks; [`drive`] opens the driver
//! scope (the rest of the block that created the workgroup).

use crate::interp::{Cx, R};
use hipfire_isa::insn::Instruction;
use hipfire_isa::{Arch, Builder};
use peacemaker_author::runtime::{Arrived, Breaks, Cond, Driver, Exit, Forward, Phase, Place, RegionId, RingId, Transition, WgCond};
use peacemaker_author::{Gfx1100, Gfx1151, Gfx1201, Target};

type Res<T> = Result<T, String>;

pub trait DynDriver {
    fn isa(&mut self) -> &mut Builder;
    fn position(&self) -> usize;
    fn lds(&mut self, name: &str, base: u32, len: u32) -> Res<RegionId>;
    fn relayout(&mut self) -> Res<()>;
    fn join(&mut self, parts: &[RegionId]) -> Res<RegionId>;
    fn split(&mut self, r: RegionId, first: usize) -> Res<(RegionId, RegionId)>;
    fn ring(&mut self, name: &str, first: RegionId, second: RegionId) -> Res<RingId>;
    fn ring_steady(&mut self, g: RingId) -> Res<()>;
    fn region_phase(&self, r: RegionId) -> Res<Phase>;
    fn ring_phases(&self, g: RingId) -> Res<(Phase, Phase)>;
    fn cur_index(&self, g: RingId) -> Res<usize>;
    fn next_index(&self, g: RingId) -> Res<usize>;
    fn begin_write(&mut self, to: Place) -> Res<()>;
    fn ds_store(&mut self, to: Place, insn: Instruction) -> Res<()>;
    fn ds_store2(&mut self, x: Place, y: Place, insn: Instruction) -> Res<()>;
    fn ds_load(&mut self, r: RegionId, insn: Instruction) -> Res<()>;
    fn ds_load_cur(&mut self, g: RingId, insn: Instruction) -> Res<()>;
    fn wait(&mut self, p: Place) -> Res<()>;
    fn wait_all(&mut self, ps: &[Place]) -> Res<()>;
    fn barrier(&mut self, ts: &[Transition]) -> Res<()>;
    fn signal(&mut self, ts: &[Transition]) -> Res<Arrived>;
    fn wait_arrived(&mut self, a: Arrived) -> Res<()>;
    fn scmp(&mut self, insn: Instruction) -> Res<Cond>;
    fn scmp_wg_uniform(&mut self, insn: Instruction) -> Res<WgCond>;
    fn label(&mut self, name: &str) -> Res<()>;
    fn skip_if(&mut self, cond: Cond, target: &str, body: &mut dyn FnMut(&mut dyn DynDriver) -> Res<()>) -> Res<()>;
    fn skip_unless(&mut self, cond: Cond, target: &str, body: &mut dyn FnMut(&mut dyn DynDriver) -> Res<()>) -> Res<()>;
    fn wg_skip_if(&mut self, cond: WgCond, target: &str, body: &mut dyn FnMut(&mut dyn DynDriver) -> Res<()>) -> Res<()>;
    fn exec_if(&mut self, cond: Cond, body: &mut dyn FnMut(&mut dyn DynDriver) -> Res<()>) -> Res<()>;
    fn forward(&mut self, body: &mut dyn FnMut(&mut dyn DynDriver, &mut Forward<Builder>) -> Res<()>) -> Res<()>;
    fn fwd_branch_if(&mut self, f: &mut Forward<Builder>, cond: Cond, label: &str) -> Res<()>;
    fn fwd_branch_unless(&mut self, f: &mut Forward<Builder>, cond: Cond, label: &str) -> Res<()>;
    fn fwd_goto(&mut self, f: &mut Forward<Builder>, label: &str) -> Res<()>;
    fn fwd_place(&mut self, f: &mut Forward<Builder>, label: &str) -> Res<()>;
    fn if_else(&mut self, cond: Cond, else_label: &str, join: &str, then: &mut dyn FnMut(&mut dyn DynDriver) -> Res<()>, els: &mut dyn FnMut(&mut dyn DynDriver) -> Res<()>) -> Res<()>;
    fn loop_until(&mut self, head: &str, exit: &str, body: &dyn Fn(&mut dyn DynDriver, &Breaks<Builder>) -> Res<()>) -> Res<()>;
    fn break_if(&mut self, cond: Cond, exit: &Breaks<Builder>) -> Res<()>;
    fn loop_carried(&mut self, head: &str, body: &dyn Fn(&mut dyn DynDriver) -> Res<WgCond>) -> Res<()>;
    fn exit(&mut self, label: &str) -> Res<Exit>;
    fn exit_if(&mut self, cond: WgCond, end: &Exit) -> Res<()>;
    fn wave_exit_unless(&mut self, cond: Cond, end: &Exit) -> Res<()>;
    fn wg_exit_unless(&mut self, cond: WgCond, target: &str, end: &Exit, body: &mut dyn FnMut(&mut dyn DynDriver) -> Res<()>) -> Res<()>;
    fn end(&mut self, end: Exit) -> Res<()>;
    fn end_with(&mut self, end: Exit, tail: &mut dyn FnMut(&mut dyn DynDriver) -> Res<()>) -> Res<()>;
    fn handoff(
        &mut self,
        readers: Cond,
        reader_label: &str,
        end: Exit,
        region: RegionId,
        write: &mut dyn FnMut(&mut dyn DynDriver, RegionId) -> Res<()>,
        read: &mut dyn FnMut(&mut dyn DynDriver, RegionId, &Exit) -> Res<()>,
    ) -> Res<()>;
}

/// Targets with and without the split barrier (`signal` / `wait_arrived`).
pub trait SplitOps: Target {
    fn sig(d: &mut Driver<'_, Self, Builder>, ts: &[Transition]) -> Res<Arrived>;
    fn wait_arr(d: &mut Driver<'_, Self, Builder>, a: Arrived) -> Res<()>;
}
macro_rules! no_split {
    ($($t:ty),*) => {$(
        impl SplitOps for $t {
            fn sig(_: &mut Driver<'_, Self, Builder>, _: &[Transition]) -> Res<Arrived> {
                Err(format!("{} has one s_barrier: use barrier(..), not signal(..)", <$t as Target>::NAME))
            }
            fn wait_arr(_: &mut Driver<'_, Self, Builder>, _: Arrived) -> Res<()> {
                Err(format!("{} has one s_barrier: no split barrier to wait", <$t as Target>::NAME))
            }
        }
    )*};
}
no_split!(Gfx1100, Gfx1151);
impl SplitOps for Gfx1201 {
    fn sig(d: &mut Driver<'_, Self, Builder>, ts: &[Transition]) -> Res<Arrived> {
        d.signal(ts)
    }
    fn wait_arr(d: &mut Driver<'_, Self, Builder>, a: Arrived) -> Res<()> {
        d.wait_arrived(a)
    }
}

impl<'b, T: SplitOps> DynDriver for Driver<'b, T, Builder> {
    fn isa(&mut self) -> &mut Builder {
        Driver::isa(self)
    }
    fn position(&self) -> usize {
        Driver::position(self)
    }
    fn lds(&mut self, name: &str, base: u32, len: u32) -> Res<RegionId> {
        Driver::lds(self, name, base, len)
    }
    fn relayout(&mut self) -> Res<()> {
        Driver::relayout(self)
    }
    fn join(&mut self, parts: &[RegionId]) -> Res<RegionId> {
        Driver::join(self, parts)
    }
    fn split(&mut self, r: RegionId, first: usize) -> Res<(RegionId, RegionId)> {
        Driver::split(self, r, first)
    }
    fn ring(&mut self, name: &str, first: RegionId, second: RegionId) -> Res<RingId> {
        Driver::ring(self, name, first, second)
    }
    fn ring_steady(&mut self, g: RingId) -> Res<()> {
        Driver::ring_steady(self, g)
    }
    fn region_phase(&self, r: RegionId) -> Res<Phase> {
        Driver::region_phase(self, r)
    }
    fn ring_phases(&self, g: RingId) -> Res<(Phase, Phase)> {
        Driver::ring_phases(self, g)
    }
    fn cur_index(&self, g: RingId) -> Res<usize> {
        Driver::cur_index(self, g)
    }
    fn next_index(&self, g: RingId) -> Res<usize> {
        Driver::next_index(self, g)
    }
    fn begin_write(&mut self, to: Place) -> Res<()> {
        Driver::begin_write(self, to)
    }
    fn ds_store(&mut self, to: Place, insn: Instruction) -> Res<()> {
        Driver::ds_store(self, to, insn)
    }
    fn ds_store2(&mut self, x: Place, y: Place, insn: Instruction) -> Res<()> {
        Driver::ds_store2(self, x, y, insn)
    }
    fn ds_load(&mut self, r: RegionId, insn: Instruction) -> Res<()> {
        Driver::ds_load(self, r, insn)
    }
    fn ds_load_cur(&mut self, g: RingId, insn: Instruction) -> Res<()> {
        Driver::ds_load_cur(self, g, insn)
    }
    fn wait(&mut self, p: Place) -> Res<()> {
        Driver::wait(self, p)
    }
    fn wait_all(&mut self, ps: &[Place]) -> Res<()> {
        Driver::wait_all(self, ps)
    }
    fn barrier(&mut self, ts: &[Transition]) -> Res<()> {
        Driver::barrier(self, ts)
    }
    fn signal(&mut self, ts: &[Transition]) -> Res<Arrived> {
        T::sig(self, ts)
    }
    fn wait_arrived(&mut self, a: Arrived) -> Res<()> {
        T::wait_arr(self, a)
    }
    fn scmp(&mut self, insn: Instruction) -> Res<Cond> {
        Driver::scmp(self, insn)
    }
    fn scmp_wg_uniform(&mut self, insn: Instruction) -> Res<WgCond> {
        Driver::scmp_wg_uniform(self, insn)
    }
    fn label(&mut self, name: &str) -> Res<()> {
        Driver::label(self, name)
    }
    fn skip_if(&mut self, cond: Cond, target: &str, body: &mut dyn FnMut(&mut dyn DynDriver) -> Res<()>) -> Res<()> {
        Driver::skip_if(self, cond, target, |d| body(d))
    }
    fn skip_unless(&mut self, cond: Cond, target: &str, body: &mut dyn FnMut(&mut dyn DynDriver) -> Res<()>) -> Res<()> {
        Driver::skip_unless(self, cond, target, |d| body(d))
    }
    fn wg_skip_if(&mut self, cond: WgCond, target: &str, body: &mut dyn FnMut(&mut dyn DynDriver) -> Res<()>) -> Res<()> {
        Driver::wg_skip_if(self, cond, target, |d| body(d))
    }
    fn exec_if(&mut self, cond: Cond, body: &mut dyn FnMut(&mut dyn DynDriver) -> Res<()>) -> Res<()> {
        Driver::exec_if(self, cond, |d| body(d))
    }
    fn forward(&mut self, body: &mut dyn FnMut(&mut dyn DynDriver, &mut Forward<Builder>) -> Res<()>) -> Res<()> {
        Driver::forward(self, |d, f| body(d, f))
    }
    fn fwd_branch_if(&mut self, f: &mut Forward<Builder>, cond: Cond, label: &str) -> Res<()> {
        f.branch_if(self, cond, label)
    }
    fn fwd_branch_unless(&mut self, f: &mut Forward<Builder>, cond: Cond, label: &str) -> Res<()> {
        f.branch_unless(self, cond, label)
    }
    fn fwd_goto(&mut self, f: &mut Forward<Builder>, label: &str) -> Res<()> {
        f.goto(self, label)
    }
    fn fwd_place(&mut self, f: &mut Forward<Builder>, label: &str) -> Res<()> {
        f.place(self, label)
    }
    fn if_else(&mut self, cond: Cond, else_label: &str, join: &str, then: &mut dyn FnMut(&mut dyn DynDriver) -> Res<()>, els: &mut dyn FnMut(&mut dyn DynDriver) -> Res<()>) -> Res<()> {
        Driver::if_else(self, cond, else_label, join, |d| then(d), |d| els(d))
    }
    fn loop_until(&mut self, head: &str, exit: &str, body: &dyn Fn(&mut dyn DynDriver, &Breaks<Builder>) -> Res<()>) -> Res<()> {
        Driver::loop_until(self, head, exit, |d, b| body(d, b))
    }
    fn break_if(&mut self, cond: Cond, exit: &Breaks<Builder>) -> Res<()> {
        Driver::break_if(self, cond, exit)
    }
    fn loop_carried(&mut self, head: &str, body: &dyn Fn(&mut dyn DynDriver) -> Res<WgCond>) -> Res<()> {
        Driver::loop_carried(self, head, |d| body(d))
    }
    fn exit(&mut self, label: &str) -> Res<Exit> {
        Driver::exit(self, label)
    }
    fn exit_if(&mut self, cond: WgCond, end: &Exit) -> Res<()> {
        Driver::exit_if(self, cond, end)
    }
    fn wave_exit_unless(&mut self, cond: Cond, end: &Exit) -> Res<()> {
        Driver::wave_exit_unless(self, cond, end)
    }
    fn wg_exit_unless(&mut self, cond: WgCond, target: &str, end: &Exit, body: &mut dyn FnMut(&mut dyn DynDriver) -> Res<()>) -> Res<()> {
        Driver::wg_exit_unless(self, cond, target, end, |d| body(d))
    }
    fn end(&mut self, end: Exit) -> Res<()> {
        Driver::end(self, end)
    }
    fn end_with(&mut self, end: Exit, tail: &mut dyn FnMut(&mut dyn DynDriver) -> Res<()>) -> Res<()> {
        Driver::end_with(self, end, |d| tail(d))
    }
    fn handoff(
        &mut self,
        readers: Cond,
        reader_label: &str,
        end: Exit,
        region: RegionId,
        write: &mut dyn FnMut(&mut dyn DynDriver, RegionId) -> Res<()>,
        read: &mut dyn FnMut(&mut dyn DynDriver, RegionId, &Exit) -> Res<()>,
    ) -> Res<()> {
        Driver::handoff(self, readers, reader_label, end, region, |d, r| write(d, r), |d, r, e| read(d, r, e))
    }
}

fn drive_t<'a, T: SplitOps>(bld: &mut Builder, body: &mut dyn FnMut(&mut Cx<'_>) -> R<'a>) -> Res<R<'a>> {
    let mut d = Driver::<T, Builder>::new(bld)?;
    let mut cx = Cx { d: Some(&mut d), fwd: None, brk: None };
    Ok(body(&mut cx))
}

/// Seal `bld` under a new driver for `arch` and run `body` with it as the
/// current scope. The driver ends with `body`: the builder is free again.
pub fn drive<'a>(arch: Arch, bld: &mut Builder, body: &mut dyn FnMut(&mut Cx<'_>) -> R<'a>) -> Res<R<'a>> {
    match arch {
        Arch::Gfx1151 => drive_t::<Gfx1151>(bld, body),
        Arch::Gfx1201 => drive_t::<Gfx1201>(bld, body),
        Arch::Gfx1100 => drive_t::<Gfx1100>(bld, body),
    }
}
