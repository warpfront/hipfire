//! Submission authority for retained programs that bypass the call that
//! recorded them.
//!
//! A retained tape (recorded-HIP replay, PM4, AQL) dereferences the device
//! tables it captured without re-running the sealed call that validated them.
//! For a binding whose placement is mutable, that stable binding identity is not
//! permission to submit: the executing token must also hold a live lease on the
//! current image of the same owner generation. The sealed call registers the
//! authority while the tape is recorded; every replay entry of
//! [`crate::replay::ReplayController`] (and the recorded-HIP replay of `Gpu`)
//! verifies the whole set before any work is submitted. A tape with no
//! registered authority (every static binding) is unaffected.

use std::sync::Arc;

/// An owner of a mutable placement image that can vouch for one submission.
pub trait LaunchAuthority: Send + Sync {
    /// Process-unique identity of this authority (not of its contents).
    fn authority_id(&self) -> u64;

    /// Verify, before a submission that does not re-run the sealed call, that
    /// the image currently published for `owner_generation` may be read: a live
    /// generation lease at the current image epoch. `Err` names the refusal.
    fn authorize_launch(&self, owner_generation: u64) -> Result<(), String>;
}

/// The authorities a retained tape was recorded against, with the owner
/// generation each one recorded.
#[derive(Default)]
pub struct LaunchAuthorities {
    entries: Vec<(Arc<dyn LaunchAuthority>, u64)>,
}

impl LaunchAuthorities {
    /// Latch `authority` at `owner_generation`. Re-noting the same authority at
    /// the same generation is a no-op; a different generation for the same
    /// authority means the tape names two owners and is refused. `sealed`: the
    /// tape's authority set is final, so an authority first seen now guarded
    /// nothing the tape validated.
    pub fn note<A>(
        &mut self,
        authority: &A,
        owner_generation: u64,
        sealed: bool,
    ) -> Result<(), String>
    where
        A: LaunchAuthority + Clone + 'static,
    {
        let id = authority.authority_id();
        if let Some((_, recorded)) = self
            .entries
            .iter()
            .find(|(known, _)| known.authority_id() == id)
        {
            if *recorded == owner_generation {
                return Ok(());
            }
            return Err(format!(
                "retained launch authority {id:#x} recorded owner generation {recorded}, \
                 observed {owner_generation}"
            ));
        }
        if sealed {
            return Err(format!(
                "retained launch authority {id:#x} (owner generation {owner_generation}) first \
                 appeared after the tape was sealed"
            ));
        }
        self.entries
            .push((Arc::new(authority.clone()), owner_generation));
        Ok(())
    }

    /// Verify every registered authority; stops at the first refusal.
    pub fn authorize_all(&self) -> Result<(), String> {
        for (authority, generation) in &self.entries {
            authority.authorize_launch(*generation)?;
        }
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};

    #[derive(Clone)]
    struct Fake {
        id: u64,
        open: Arc<AtomicBool>,
    }
    impl LaunchAuthority for Fake {
        fn authority_id(&self) -> u64 {
            self.id
        }
        fn authorize_launch(&self, owner_generation: u64) -> Result<(), String> {
            if owner_generation != 4 {
                return Err(format!("generation {owner_generation} is not leased"));
            }
            if self.open.load(Ordering::SeqCst) {
                Ok(())
            } else {
                Err("no lease".into())
            }
        }
    }

    #[test]
    fn empty_set_authorizes_and_registered_set_follows_the_authority() {
        let mut set = LaunchAuthorities::default();
        assert!(set.is_empty());
        set.authorize_all().unwrap();
        let open = Arc::new(AtomicBool::new(false));
        let fake = Fake {
            id: 1,
            open: open.clone(),
        };
        set.note(&fake, 4, false).unwrap();
        set.note(&fake, 4, false).unwrap();
        assert_eq!(set.len(), 1);
        assert_eq!(set.authorize_all().unwrap_err(), "no lease");
        open.store(true, Ordering::SeqCst);
        set.authorize_all().unwrap();
        set.clear();
        assert!(set.is_empty());
    }

    #[test]
    fn generation_change_and_late_authority_are_refused() {
        let mut set = LaunchAuthorities::default();
        let fake = Fake {
            id: 1,
            open: Arc::new(AtomicBool::new(true)),
        };
        set.note(&fake, 4, false).unwrap();
        assert!(set.note(&fake, 5, false).is_err());
        let late = Fake {
            id: 2,
            open: Arc::new(AtomicBool::new(true)),
        };
        assert!(set.note(&late, 4, true).is_err());
        // An already latched authority still matches after sealing.
        set.note(&fake, 4, true).unwrap();
        // A recorded generation the authority no longer leases refuses.
        let mut stale = LaunchAuthorities::default();
        stale.note(&fake, 9, false).unwrap();
        assert!(stale.authorize_all().is_err());
    }
}
