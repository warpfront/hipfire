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
//! Quarantine is cleared only by restarting the process: no device-reset API
//! exists in `hip-bridge` or `redline-rocr` (a search for `hipDeviceReset` /
//! `device_reset` found nothing). The first recorded reason for a device wins.

use std::collections::BTreeMap;
use std::fmt;
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

/// Quarantine `key` until process restart. The first reason wins.
pub(crate) fn quarantine(key: &DeviceKey, reason: impl Into<String>) {
    let reason = reason.into();
    let mut registry = REGISTRY.lock().unwrap_or_else(|e| e.into_inner());
    if registry.contains_key(&key.0) {
        return;
    }
    eprintln!("[railgun] device {key} quarantined until process restart: {reason}");
    registry.insert(key.0.clone(), reason);
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
            "device {key} is quarantined for railgun submits until process restart (no device reset API exists in hip-bridge/redline-rocr): {reason}"
        )),
    }
}

#[cfg(test)]
pub(crate) fn clear_for_test(key: &DeviceKey) {
    REGISTRY
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .remove(&key.0);
}

#[cfg(test)]
mod tests {
    use super::*;

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
