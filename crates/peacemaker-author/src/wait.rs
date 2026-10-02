//! Wait tokens. A memory operation returns a `Pending` token; `Wave::wait`
//! turns it into `Drained` evidence, emitting the counter wait the backend's
//! ledger still needs (none if an earlier wait already retired it). Rules
//! that publish memory demand `Drained`, so "forgot to drain before the
//! barrier" is a type error. Tokens are zero-cost: they never reach the GPU.

use crate::backend::EventId;
use crate::target::WaitModel;
use std::marker::PhantomData;

/// A kind of memory event a token tracks.
pub trait Event {}

/// The LDS stores into region `R`.
pub struct LdsWrite<R>(PhantomData<fn() -> R>);
impl<R> Event for LdsWrite<R> {}

/// An issued, not yet awaited memory operation on counter model `W`.
#[must_use = "a memory operation is in flight; consume it with `wait` before the barrier that publishes it"]
pub struct Pending<W: WaitModel, E: Event> {
    /// The youngest event the token covers; `None` before the first store.
    pub(crate) last: Option<EventId>,
    _p: PhantomData<fn() -> (W, E)>,
}

/// Evidence that every operation of a `Pending` token has completed.
#[must_use = "drain evidence is only useful when a transition consumes it"]
pub struct Drained<E: Event> {
    pub(crate) last: Option<EventId>,
    _p: PhantomData<fn() -> E>,
}

impl<W: WaitModel, E: Event> Pending<W, E> {
    pub(crate) fn new(last: Option<EventId>) -> Self {
        Self { last, _p: PhantomData }
    }
}
impl<E: Event> Drained<E> {
    pub(crate) fn new(last: Option<EventId>) -> Self {
        Self { last, _p: PhantomData }
    }
}

/// A set of pending tokens drained together by `Wave::wait_all`.
pub trait Pendings<W: WaitModel> {
    type Drained;
    #[doc(hidden)]
    fn each_event(&self, f: &mut dyn FnMut(EventId));
    #[doc(hidden)]
    fn drained(self) -> Self::Drained;
}
impl<W: WaitModel, E: Event> Pendings<W> for Pending<W, E> {
    type Drained = Drained<E>;
    fn each_event(&self, f: &mut dyn FnMut(EventId)) {
        if let Some(e) = self.last { f(e) }
    }
    fn drained(self) -> Drained<E> {
        Drained::new(self.last)
    }
}
macro_rules! pendings_tuple {
    ($($p:ident),+) => {
        impl<W: WaitModel, $($p: Pendings<W>),+> Pendings<W> for ($($p,)+) {
            type Drained = ($($p::Drained,)+);
            #[allow(non_snake_case)]
            fn each_event(&self, f: &mut dyn FnMut(EventId)) {
                let ($($p,)+) = self;
                $($p.each_event(f);)+
            }
            #[allow(non_snake_case)]
            fn drained(self) -> Self::Drained {
                let ($($p,)+) = self;
                ($($p.drained(),)+)
            }
        }
    };
}
pendings_tuple!(A);
pendings_tuple!(A, B);
pendings_tuple!(A, B, C);
pendings_tuple!(A, B, C, D);
