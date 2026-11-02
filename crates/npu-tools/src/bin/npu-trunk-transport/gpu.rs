// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Alex Mittell
// hipfire — see LICENSE and NOTICE in the project root.
//! The transport oracle can issue GPU work through either side of the HIP handoff.
use hip_bridge as serving;
use railgun::npu::hip_runtime as relay;
use std::{
    cell::{Cell, RefCell},
    ffi::c_void,
    ops::Deref,
    rc::Rc,
    time::{Duration, Instant},
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) enum Backend {
    #[default]
    Relay,
    Serving,
}
impl Backend {
    pub(super) fn parse(value: &str) -> Result<Self, String> {
        match value {
            "relay" => Ok(Self::Relay),
            "serving" => Ok(Self::Serving),
            _ => Err(format!(
                "unknown GPU runtime {value:?}; expected relay or serving"
            )),
        }
    }
    pub(super) fn load(self, pci: &str) -> Result<Option<Serving>, String> {
        match self {
            Self::Relay => Ok(None),
            Self::Serving => Serving::load(pci).map(Some),
        }
    }
}

pub(super) struct Serving {
    hip: Rc<serving::HipRuntime>,
    device: i32,
    cleanup_errors: Rc<RefCell<Vec<String>>>,
}
impl Serving {
    fn load(pci: &str) -> Result<Self, String> {
        let hip = Rc::new(serving::HipRuntime::load().map_err(error)?);
        let mut devices = Vec::new();
        for device in 0..hip.device_count().map_err(error)? {
            let id = hip.device_pci_bus_id(device).map_err(error)?;
            if id.eq_ignore_ascii_case(pci) {
                hip.set_device(device).map_err(error)?;
                return Ok(Self {
                    hip,
                    device,
                    cleanup_errors: Rc::new(RefCell::new(Vec::new())),
                });
            }
            devices.push(id);
        }
        Err(format!(
            "serving runtime has no HIP device at PCI {pci}; found {devices:?}"
        ))
    }

    /// Called after resource wrappers drop; teardown is part of gate success.
    pub(super) fn check_cleanup(&self) -> Result<(), String> {
        cleanup_result(&self.cleanup_errors)
    }
}

fn cleanup_result(errors: &RefCell<Vec<String>>) -> Result<(), String> {
    let errors = std::mem::take(&mut *errors.borrow_mut());
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors.join("; "))
    }
}

/// Explicit ownership for hip-bridge's handles, which do not free themselves.
pub(super) struct Owned<T> {
    hip: Rc<serving::HipRuntime>,
    drained: Rc<Cell<bool>>,
    cleanup_errors: Rc<RefCell<Vec<String>>>,
    value: Option<T>,
    destroy: fn(&serving::HipRuntime, T) -> serving::HipResult<()>,
}
impl<T> Owned<T> {
    fn new(
        hip: &Rc<serving::HipRuntime>,
        drained: &Rc<Cell<bool>>,
        cleanup_errors: &Rc<RefCell<Vec<String>>>,
        value: T,
        destroy: fn(&serving::HipRuntime, T) -> serving::HipResult<()>,
    ) -> Self {
        Self {
            hip: Rc::clone(hip),
            drained: Rc::clone(drained),
            cleanup_errors: Rc::clone(cleanup_errors),
            value: Some(value),
            destroy,
        }
    }
}
impl<T> Deref for Owned<T> {
    type Target = T;
    fn deref(&self) -> &T {
        self.value.as_ref().expect("live gate handle")
    }
}
impl<T> Drop for Owned<T> {
    fn drop(&mut self) {
        if let Some(value) = self.value.take() {
            if let Err(e) = release_or_quarantine(
                value,
                Rc::clone(&self.hip),
                self.drained.get(),
                self.destroy,
            ) {
                self.cleanup_errors.borrow_mut().push(e.to_string());
            }
        }
    }
}
fn release_or_quarantine<T, H, E>(
    value: T,
    owner: Rc<H>,
    drained: bool,
    destroy: impl FnOnce(&H, T) -> Result<(), E>,
) -> Result<(), E> {
    if drained {
        destroy(&owner, value)
    } else {
        // A failed/bounded-out wait does not make hipFree safe: it may block
        // on the stream that runtime.close must release. Keep the handle and
        // runtime mapping until this diagnostic process exits.
        std::mem::forget((value, owner));
        Ok(())
    }
}
fn error(e: serving::HipError) -> String {
    e.to_string()
}
fn mismatch<T>() -> Result<T, String> {
    Err("transport resource belongs to a different GPU backend".into())
}

pub(super) enum Buffer {
    Relay(relay::HipBuffer),
    Serving(Owned<serving::DeviceBuffer>),
}
impl Buffer {
    pub(super) fn ptr(&self) -> u64 {
        match self {
            Self::Relay(b) => b.ptr(),
            Self::Serving(b) => b.as_ptr() as usize as u64,
        }
    }
}
pub(super) enum Stream {
    Relay(relay::HipStream),
    Serving(Owned<serving::Stream>),
}
impl Stream {
    pub(super) fn raw(&self) -> *mut c_void {
        match self {
            Self::Relay(s) => s.raw(),
            Self::Serving(s) => s.as_raw(),
        }
    }
}
pub(super) enum Event {
    Relay(relay::HipEvent),
    Serving(Owned<serving::Event>),
}
pub(super) enum Module {
    Relay(relay::HipModule),
    // hip-bridge keeps modules resident for the process lifetime, as serving does.
    Serving(Rc<serving::HipRuntime>, serving::Module),
}
pub(super) enum Function {
    Relay(relay::HipFunction),
    Serving(serving::Function),
}
impl Module {
    pub(super) fn function(&self, name: &str) -> Result<Function, String> {
        match self {
            Self::Relay(m) => m.function(name).map(Function::Relay),
            Self::Serving(h, m) => h
                .module_get_function(m, name)
                .map(Function::Serving)
                .map_err(error),
        }
    }
}

pub(super) enum Gpu<'a> {
    Relay(&'a relay::HipRuntime),
    Serving(ScopedServing<'a>),
}
pub(super) struct ScopedServing<'a> {
    owner: &'a Serving,
    // Each positive/watchdog invocation owns a fresh token. A previous passing
    // invocation must never authorize frees after a later failed wait.
    drained: Rc<Cell<bool>>,
}
impl Deref for ScopedServing<'_> {
    type Target = Serving;
    fn deref(&self) -> &Serving {
        self.owner
    }
}
impl<'a> Gpu<'a> {
    pub(super) fn new(
        relay: &'a relay::HipRuntime,
        serving: Option<&'a Serving>,
    ) -> Result<Self, String> {
        if let Some(serving) = serving {
            let relay_address = relay.event_record_address();
            let serving_address = serving.hip.event_record_address();
            if !same_runtime(relay_address, serving_address) {
                return Err(format!("two HIP runtimes: relay {:?} (hipEventRecord={relay_address:#x}), serving {:?} (hipEventRecord={serving_address:#x}); use one ROCm install", relay.mapped_hip_library(), serving.hip.mapped_hip_library()));
            }
            println!("gpu_runtime=serving relay_library={:?} serving_library={:?} same_hipEventRecord=true", relay.mapped_hip_library(), serving.hip.mapped_hip_library());
            Ok(Self::Serving(ScopedServing {
                owner: serving,
                drained: Rc::new(Cell::new(false)),
            }))
        } else {
            println!(
                "gpu_runtime=relay relay_library={:?}",
                relay.mapped_hip_library()
            );
            Ok(Self::Relay(relay))
        }
    }
    /// Call only after bounded completion of every stream used by this scope.
    pub(super) fn allow_cleanup(&self) {
        if let Self::Serving(s) = self {
            s.drained.set(true);
        }
    }
    pub(super) fn allocate(&self, bytes: usize) -> Result<Buffer, String> {
        match self {
            Self::Relay(h) => h.allocate(bytes, None).map(Buffer::Relay),
            Self::Serving(s) => s
                .hip
                .malloc(bytes)
                .map(|b| {
                    Buffer::Serving(Owned::new(
                        &s.hip,
                        &s.drained,
                        &s.cleanup_errors,
                        b,
                        serving::HipRuntime::free,
                    ))
                })
                .map_err(error),
        }
    }
    pub(super) fn upload(&self, buffer: &Buffer, bytes: &[u8]) -> Result<(), String> {
        match (self, buffer) {
            (Self::Relay(h), Buffer::Relay(b)) => h.upload(b, bytes),
            (Self::Serving(s), Buffer::Serving(b)) => s.hip.memcpy_htod(b, bytes).map_err(error),
            _ => mismatch(),
        }
    }
    pub(super) fn download(&self, buffer: &Buffer, bytes: &mut [u8]) -> Result<(), String> {
        match (self, buffer) {
            (Self::Relay(h), Buffer::Relay(b)) => h.download(b, bytes),
            (Self::Serving(s), Buffer::Serving(b)) => s.hip.memcpy_dtoh(bytes, b).map_err(error),
            _ => mismatch(),
        }
    }
    pub(super) fn memset(&self, buffer: &Buffer, value: u8) -> Result<(), String> {
        match (self, buffer) {
            (Self::Relay(h), Buffer::Relay(b)) => h.memset(b, value),
            (Self::Serving(s), Buffer::Serving(b)) => {
                s.hip.memset(b, i32::from(value), b.size()).map_err(error)
            }
            _ => mismatch(),
        }
    }
    pub(super) fn load_module(&self, bytes: &[u8]) -> Result<Module, String> {
        match self {
            Self::Relay(h) => h.load_module(bytes).map(Module::Relay),
            Self::Serving(s) => s
                .hip
                .module_load_data(bytes)
                .map(|m| Module::Serving(Rc::clone(&s.hip), m))
                .map_err(error),
        }
    }
    pub(super) fn compute_units(&self) -> Result<u32, String> {
        match self {
            Self::Relay(h) => h.compute_units(),
            Self::Serving(s) => {
                // hipDeviceAttributeMultiprocessorCount, same attribute as the relay arm.
                let n = s.hip.get_device_attribute(63, s.device).map_err(error)?;
                u32::try_from(n).map_err(|_| format!("negative compute unit count {n}"))
            }
        }
    }
    pub(super) fn stream(&self) -> Result<Stream, String> {
        match self {
            Self::Relay(h) => h.stream().map(Stream::Relay),
            Self::Serving(s) => s
                .hip
                .stream_create_non_blocking()
                .map(|v| {
                    Stream::Serving(Owned::new(
                        &s.hip,
                        &s.drained,
                        &s.cleanup_errors,
                        v,
                        serving::HipRuntime::stream_destroy,
                    ))
                })
                .map_err(error),
        }
    }
    pub(super) fn event(&self) -> Result<Event, String> {
        self.create_event(false)
    }
    pub(super) fn system_event(&self) -> Result<Event, String> {
        self.create_event(true)
    }
    fn create_event(&self, system: bool) -> Result<Event, String> {
        match self {
            Self::Relay(h) => if system { h.system_event() } else { h.event() }.map(Event::Relay),
            Self::Serving(s) => {
                let event = if system {
                    s.hip.event_create_with_flags(
                        serving::HIP_EVENT_RELEASE_TO_SYSTEM | serving::HIP_EVENT_DISABLE_TIMING,
                    )
                } else {
                    s.hip.event_create()
                };
                event
                    .map(|e| {
                        Event::Serving(Owned::new(
                            &s.hip,
                            &s.drained,
                            &s.cleanup_errors,
                            e,
                            serving::HipRuntime::event_destroy,
                        ))
                    })
                    .map_err(error)
            }
        }
    }
    pub(super) fn record_on(&self, event: &Event, stream: &Stream) -> Result<(), String> {
        match (self, event, stream) {
            (Self::Relay(h), Event::Relay(e), Stream::Relay(st)) => h.record_on(e, Some(st)),
            (Self::Serving(s), Event::Serving(e), Stream::Serving(st)) => {
                s.hip.event_record(e, Some(st)).map_err(error)
            }
            _ => mismatch(),
        }
    }
    pub(super) fn wait_event_timeout(
        &self,
        event: &Event,
        timeout: Duration,
    ) -> Result<(), String> {
        match (self, event) {
            (Self::Relay(h), Event::Relay(e)) => h.wait_event_timeout(e, timeout),
            (Self::Serving(s), Event::Serving(e)) => {
                let deadline = Instant::now() + timeout;
                loop {
                    if s.hip.event_query(e).map_err(error)? {
                        return Ok(());
                    }
                    if Instant::now() >= deadline {
                        return Err("HIP event timed out".into());
                    }
                    std::thread::sleep(Duration::from_micros(20));
                }
            }
            _ => mismatch(),
        }
    }
    pub(super) fn elapsed_ms(&self, start: &Event, end: &Event) -> Result<f32, String> {
        // The gate has already bounded consumer completion before reading timing.
        match (self, start, end) {
            (Self::Relay(h), Event::Relay(a), Event::Relay(b)) => h.elapsed_ms(a, b),
            (Self::Serving(s), Event::Serving(a), Event::Serving(b)) => {
                s.hip.event_elapsed_ms(a, b).map_err(error)
            }
            _ => mismatch(),
        }
    }
    pub(super) fn launch_on(
        &self,
        stream: &Stream,
        function: &Function,
        grid: u32,
        block: u32,
        args: &mut [u8],
    ) -> Result<(), String> {
        match (self, stream, function) {
            (Self::Relay(h), Stream::Relay(st), Function::Relay(f)) => {
                h.launch_on(Some(st), f, grid, block, args)
            }
            (Self::Serving(s), Stream::Serving(st), Function::Serving(f)) => {
                // SAFETY: the caller passes coop_gpu::copy_args for the module's COPY
                // kernel; its buffers and runtime-registered arenas outlive the drain.
                unsafe {
                    s.hip
                        .launch_kernel_blob(f, [grid, 1, 1], [block, 1, 1], 0, Some(st), args)
                }
                .map_err(error)
            }
            _ => mismatch(),
        }
    }
}

fn same_runtime(relay: usize, serving: usize) -> bool {
    relay != 0 && relay == serving
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn backend_selection_preserves_relay_default_and_refuses_typos() {
        assert_eq!(Backend::default(), Backend::Relay);
        assert_eq!(Backend::parse("relay").unwrap(), Backend::Relay);
        assert_eq!(Backend::parse("serving").unwrap(), Backend::Serving);
        assert!(Backend::parse("serivng").is_err());
    }
    #[test]
    fn serving_handoff_requires_identical_nonzero_live_symbol() {
        assert!(same_runtime(0x1234, 0x1234));
        assert!(!same_runtime(0x1234, 0x5678));
        assert!(!same_runtime(0, 0));
        assert!(!same_runtime(0, 0x1234));
        assert!(!same_runtime(0x1234, 0));
    }

    #[test]
    fn uncertain_completion_retains_handle_and_runtime_without_destroy() {
        let owner = Rc::new(());
        let destroyed = Cell::new(false);
        release_or_quarantine((), Rc::clone(&owner), false, |_, _| {
            destroyed.set(true);
            Ok::<_, ()>(())
        })
        .unwrap();
        assert!(!destroyed.get());
        assert_eq!(Rc::strong_count(&owner), 2, "runtime must remain mapped");
    }

    #[test]
    fn completed_scope_releases_handles_and_reports_destroy_failure() {
        let owner = Rc::new(());
        let destroyed = Cell::new(false);
        let result = release_or_quarantine((), Rc::clone(&owner), true, |_, _| {
            destroyed.set(true);
            Err("destroy failed")
        });
        assert!(destroyed.get());
        assert_eq!(result, Err("destroy failed"));
        assert_eq!(Rc::strong_count(&owner), 1);
    }

    #[test]
    fn resource_teardown_errors_fail_the_gate_after_scope_drop() {
        let errors = RefCell::new(vec![
            "hipFree failed".into(),
            "hipStreamDestroy failed".into(),
        ]);
        assert_eq!(
            cleanup_result(&errors),
            Err("hipFree failed; hipStreamDestroy failed".into())
        );
        assert!(errors.borrow().is_empty());
        assert!(cleanup_result(&errors).is_ok());
    }
}
