//! Process-wide quarantine of devices whose persistent-program cancel could
//! not be proven safe.
//!
//! A device is quarantined when a railgun persistent-window teardown
//! (`WindowQueue::teardown`) ends without retirement evidence: either
//! `hsa_queue_inactivate` returned but the end-of-pipe completion signal never
//! reached zero within the observation window, or the inactivation itself
//! failed. In-tree, `redline-rocr/src/runtime.rs:1621-1626` documents only that
//! inactivation aborts pending executions and has no wall-clock bound; it does
//! not say every wave is retired when the call returns. Waves of the cancelled
//! persistent kernel may therefore still be running against memory the owner
//! is about to reuse, so every later railgun lowering, submit and pinned-object
//! registration on that device is refused with
//! `railgun::window::FailureKind::DeviceQuarantined`.
//!
//! A retained replay with unknown quiescence also quarantines its owning
//! `Gpu`: HIP stream/device synchronization is not retirement evidence for an
//! independent PM4/AQL queue. Its owners are retained and later GPU use/init is
//! refused through this same registry.
//!
//! Quarantine is cleared only by restarting the process: no device-reset API
//! exists in `hip-bridge` or `redline-rocr` (a search for `hipDeviceReset` /
//! `device_reset` found nothing). The first recorded reason for a device wins.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

/// Identity of a device for quarantine: `pci:<domain:bus:dev.fn>` when the PCI
/// address resolves, else `hip:<ordinal>`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DeviceKey(String);

impl DeviceKey {
    pub fn from_pci(address: &str) -> Self {
        Self(format!("pci:{address}"))
    }

    pub fn from_ordinal(ordinal: i32) -> Self {
        Self(format!("hip:{ordinal}"))
    }
}

impl fmt::Display for DeviceKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Device key -> first quarantine reason.
static REGISTRY: Mutex<BTreeMap<String, String>> = Mutex::new(BTreeMap::new());
/// Keep the ordinary GPU bind path lock-free before the first quarantine.
static ANY_QUARANTINED: AtomicBool = AtomicBool::new(false);

/// Per-GPU sticky refusal and the device identities used at re-initialization.
/// CPU-testable without constructing a GPU or invoking HIP.
pub(crate) struct RetainedReplayQuarantine {
    device: DeviceKey,
    ordinal: DeviceKey,
    reason: Option<String>,
}

impl RetainedReplayQuarantine {
    pub(crate) fn for_device(device: DeviceKey, ordinal: i32) -> Result<Self, String> {
        let ordinal = DeviceKey::from_ordinal(ordinal);
        refuse_if_quarantined(&device)?;
        refuse_if_quarantined(&ordinal)?;
        Ok(Self {
            device,
            ordinal,
            reason: None,
        })
    }

    pub(crate) fn mark(&mut self, reason: impl Into<String>) {
        if self.reason.is_some() {
            return;
        }
        let reason = reason.into();
        quarantine(&self.device, reason.clone());
        quarantine(&self.ordinal, reason.clone());
        self.reason = quarantine_reason(&self.device).or(Some(reason));
    }

    #[inline]
    pub(crate) fn check(&self) -> Result<(), String> {
        if let Some(reason) = self.reason.as_ref() {
            return Err(format!(
                "device {} retained replay quiescence is unknown; GPU owners retained and reuse refused until process restart: {reason}",
                self.device
            ));
        }
        if ANY_QUARANTINED.load(Ordering::Acquire) {
            refuse_if_quarantined(&self.device)?;
            refuse_if_quarantined(&self.ordinal)?;
        }
        Ok(())
    }
}

/// Quarantine `key` until process restart. The first reason wins.
pub(crate) fn quarantine(key: &DeviceKey, reason: impl Into<String>) {
    let reason = reason.into();
    let mut registry = REGISTRY.lock().unwrap_or_else(|e| e.into_inner());
    if registry.contains_key(&key.0) {
        return;
    }
    eprintln!("[railgun] device {key} quarantined until process restart: {reason}");
    registry.insert(key.0.clone(), reason);
    ANY_QUARANTINED.store(true, Ordering::Release);
}

/// Why `key` is quarantined, or `None` when it is not.
pub fn quarantine_reason(key: &DeviceKey) -> Option<String> {
    REGISTRY
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get(&key.0)
        .cloned()
}

/// `Err(message)` when `key` is quarantined.
pub(crate) fn refuse_if_quarantined(key: &DeviceKey) -> Result<(), String> {
    match quarantine_reason(key) {
        None => Ok(()),
        Some(reason) => Err(format!(
            "device {key} is quarantined for GPU work until process restart (no device reset API exists in hip-bridge/redline-rocr): {reason}"
        )),
    }
}

#[cfg(test)]
pub(crate) fn clear_for_test(key: &DeviceKey) {
    let mut registry = REGISTRY.lock().unwrap_or_else(|e| e.into_inner());
    registry.remove(&key.0);
    ANY_QUARANTINED.store(!registry.is_empty(), Ordering::Release);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retained_replay_refuses_reuse_and_reinit_with_first_reason() {
        let device = DeviceKey::from_pci("test:retained-gpu");
        let ordinal = DeviceKey::from_ordinal(19001);
        let mut state = RetainedReplayQuarantine::for_device(device.clone(), 19001).unwrap();
        assert!(state.check().is_ok());
        state.mark("independent PM4 wait lost retirement evidence");
        state.mark("later HIP sync succeeded");
        let error = state.check().unwrap_err();
        assert!(error.contains("independent PM4 wait lost retirement evidence"));
        assert!(!error.contains("later HIP sync succeeded"));
        // This is the exact admission policy used by Gpu::init_with_device.
        let replacement = RetainedReplayQuarantine::for_device(device.clone(), 19001);
        assert!(replacement.is_err());
        assert!(refuse_if_quarantined(&ordinal).is_err());
        clear_for_test(&device);
        clear_for_test(&ordinal);
        // Registry clearing exists only under cfg(test). Even that does not
        // reset the already-marked object's sticky refusal.
        assert!(state.check().is_err());
    }

    #[test]
    fn retained_replay_quarantine_blocks_other_live_owner_of_same_device() {
        let device = DeviceKey::from_pci("test:retained-alias");
        let ordinal = DeviceKey::from_ordinal(19002);
        let mut first = RetainedReplayQuarantine::for_device(device.clone(), 19002).unwrap();
        let second = RetainedReplayQuarantine::for_device(device.clone(), 19002).unwrap();
        let healthy_key = DeviceKey::from_pci("test:retained-healthy");
        let healthy = RetainedReplayQuarantine::for_device(healthy_key, 19003).unwrap();
        first.mark("unknown AQL completion");
        assert!(first.check().is_err());
        assert!(second.check().is_err());
        assert!(healthy.check().is_ok());
        clear_for_test(&device);
        clear_for_test(&ordinal);
    }

    #[test]
    fn quarantine_then_refuse_carries_reason() {
        let key = DeviceKey::from_pci("test:q1");
        assert!(refuse_if_quarantined(&key).is_ok());
        quarantine(&key, "no retirement evidence");
        let err = refuse_if_quarantined(&key).unwrap_err();
        assert!(err.contains("no retirement evidence"), "{err}");
        assert!(err.contains("pci:test:q1"), "{err}");
        clear_for_test(&key);
    }

    #[test]
    fn other_key_unaffected() {
        let a = DeviceKey::from_ordinal(9001);
        let b = DeviceKey::from_ordinal(9002);
        quarantine(&a, "x");
        assert!(refuse_if_quarantined(&b).is_ok());
        assert_eq!(quarantine_reason(&b), None);
        clear_for_test(&a);
    }

    #[test]
    fn first_reason_wins() {
        let key = DeviceKey::from_pci("test:q3");
        quarantine(&key, "first");
        quarantine(&key, "second");
        assert_eq!(quarantine_reason(&key).as_deref(), Some("first"));
        clear_for_test(&key);
    }
}
