#![forbid(unsafe_code)]
//! PM kernel language, milestone M0: the typed core
//! (`/home/kaden/qcal/release-0.4.1/pm-kernel-lang-design.md` §4, §12 M0).
//!
//! Kernel source is ordinary Rust run at build time. This crate puts types
//! in front of a lowering backend (`hipfire-isa`'s checked `Builder`, which
//! keeps its wait ledger, LDS slot machine, hazard model and loop fixpoint
//! as the second evaluator):
//! - targets and capabilities as types (`target`);
//! - LDS regions with per-phase ownership, rings and barrier transitions (`lds`);
//! - `Pending` / `Drained` wait tokens (`wait`);
//! - workgroup / wave / lane scopes, condition handles and `loop_carried` (`scope`).
//!
//! The types emit nothing of their own: every instruction is one the
//! backend would emit for the same request, so a kernel ported onto this
//! core re-emits byte-identical assembly. The Halo VerifyAttn `lgkmcnt`
//! race and the gfx11 FA2 mailbox race are compile errors here
//! (`tests/ui`).
//!
//! Scheduling, instruction selection and register allocation are this
//! layer's job (decision D2); the certifier core stays non-scheduling.

pub mod backend;
pub mod lds;
pub mod scope;
pub mod target;
pub mod trace;
pub mod wait;

pub use backend::{Auth, Backend, EventId, Seal, SlotTransition};
pub use lds::{
    join, prime, ready, retire, retire_cur, rotate, AllFree, Free, JoinPart, LdsRegion, Published, Ring, State, StoreTarget,
    Transition, Transitions, Writing,
};
pub use scope::{Arrived, Carried, End, Forward, LoopExit, Scc, Uniform, Wave, WgUniform, Workgroup};
pub use target::{
    BarrierModel, Full, Gfx11Waits, Gfx1100, Gfx1151, Gfx1201, Gfx12Waits, LdsCounter, MmaIu4, Split, SplitBarrier, Target,
    WaitModel,
};
pub use wait::{Drained, Event, LdsWrite, Pending, Pendings};
