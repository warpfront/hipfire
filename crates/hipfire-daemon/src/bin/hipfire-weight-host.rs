// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `hipfire-weight-host`: shared-weight host v1 (docs/CONFIG.md, "Shared-weight
//! host").
//!
//! Loads one pinned dense Qwen3.5 HFQ with the incumbent loader, promotes every
//! unique weight owner into an exportable read-only VMM allocation, and leases
//! the allocation FDs of that one generation (random 128-bit epoch) to admitted
//! tenants over a private Unix socket. It never runs a forward and allocates no
//! KV, state or scratch.
//!
//! Lifecycle: `Starting -> Loading -> Publishing -> Ready -> Draining -> Closed`.
//! SIGTERM/SIGINT/SIGHUP stop admission (the socket path is unlinked), half-close
//! every lease so tenants see the host going away, then wait until every lease
//! is released or its tenant process (pid + start time) is gone. Only then are
//! the exported owners freed. If the drain deadline passes, the host exits 3
//! with the generation still occupied: tenants keep their own references, and
//! nothing is freed on their behalf.

#[cfg(target_os = "linux")]
#[path = "../gpu_lock.rs"]
#[allow(dead_code)]
mod gpu_lock;

#[cfg(target_os = "linux")]
fn main() {
    std::process::exit(host::run());
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("hipfire-weight-host: Linux only");
    std::process::exit(1);
}

#[cfg(target_os = "linux")]
mod host {
    use super::gpu_lock::{fabric, GpuLocks};
    use hipfire_arch_qwen35::carrier as qcarrier;
    use hipfire_arch_qwen35::shared_weights;
    use hipfire_runtime::arch::Architecture;
    use hipfire_runtime::weight_host::{
        certify_source, Epoch, HostConnection, ReleasePoll, ServeOptions, WeightPublication,
    };
    use parking_lot::Mutex;
    use std::collections::BTreeMap;
    use std::os::fd::AsRawFd;
    use std::os::unix::ffi::OsStrExt;
    use std::os::unix::fs::{FileTypeExt, MetadataExt, PermissionsExt};
    use std::os::unix::net::{UnixListener, UnixStream};
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    /// Exit codes: 0 closed cleanly, 1 failed before Ready (or teardown
    /// failed), 2 usage, 3 drain deadline passed with leases still occupied.
    const EXIT_FAILED: i32 = 1;
    const EXIT_USAGE: i32 = 2;
    const EXIT_DRAIN_TIMEOUT: i32 = 3;

    const USAGE: &str =
        "usage: hipfire-weight-host --model <dense-qwen35.hfq> --socket <absolute path> \
[--drain-timeout-secs N (600)] [--handshake-timeout-ms N (10000)]";

    struct Args {
        model: PathBuf,
        socket: PathBuf,
        drain_timeout: Duration,
        handshake_timeout: Duration,
    }

    fn parse_args() -> Result<Args, String> {
        let mut model = None;
        let mut socket = None;
        let mut drain_timeout = Duration::from_secs(600);
        let mut handshake_timeout = Duration::from_millis(10_000);
        let mut it = std::env::args_os().skip(1);
        while let Some(flag) = it.next() {
            let mut value = || {
                it.next()
                    .ok_or_else(|| format!("{} needs a value", flag.to_string_lossy()))
            };
            match flag.to_str() {
                Some("--model") => model = Some(PathBuf::from(value()?)),
                Some("--socket") => socket = Some(PathBuf::from(value()?)),
                Some("--drain-timeout-secs") => {
                    let v = value()?;
                    drain_timeout = Duration::from_secs(
                        v.to_str()
                            .and_then(|s| s.parse().ok())
                            .ok_or("--drain-timeout-secs: integer")?,
                    );
                }
                Some("--handshake-timeout-ms") => {
                    let v = value()?;
                    let ms: u64 = v
                        .to_str()
                        .and_then(|s| s.parse().ok())
                        .ok_or("--handshake-timeout-ms: integer")?;
                    if ms == 0 {
                        return Err("--handshake-timeout-ms must be > 0".into());
                    }
                    handshake_timeout = Duration::from_millis(ms);
                }
                _ => return Err(format!("unknown argument {}", flag.to_string_lossy())),
            }
        }
        let model = model.ok_or("--model is required")?;
        let socket = socket.ok_or("--socket is required")?;
        if !socket.is_absolute() {
            return Err("--socket must be an absolute path".into());
        }
        Ok(Args {
            model,
            socket,
            drain_timeout,
            handshake_timeout,
        })
    }

    // ── Signals: one sigwait thread, everything else blocks them ─────────

    static DRAIN: AtomicBool = AtomicBool::new(false);
    static LISTENER_FD: AtomicI32 = AtomicI32::new(-1);

    fn signal_set() -> libc::sigset_t {
        unsafe {
            let mut set: libc::sigset_t = std::mem::zeroed();
            libc::sigemptyset(&mut set);
            for sig in [libc::SIGTERM, libc::SIGINT, libc::SIGHUP] {
                libc::sigaddset(&mut set, sig);
            }
            set
        }
    }

    /// Block the drain signals in this thread before any other thread exists
    /// (HIP spawns its own), then hand them to one waiter thread.
    fn install_signal_waiter() -> Result<(), String> {
        let set = signal_set();
        let rc = unsafe { libc::pthread_sigmask(libc::SIG_BLOCK, &set, std::ptr::null_mut()) };
        if rc != 0 {
            return Err(format!("pthread_sigmask: errno {rc}"));
        }
        std::thread::Builder::new()
            .name("weight-host-signals".into())
            .spawn(move || loop {
                let mut sig = 0;
                if unsafe { libc::sigwait(&set, &mut sig) } != 0 {
                    continue;
                }
                if !DRAIN.swap(true, Ordering::SeqCst) {
                    eprintln!("[weight-host] signal {sig}: draining (admission stopped)");
                }
                let fd = LISTENER_FD.load(Ordering::SeqCst);
                if fd >= 0 {
                    // Wakes the blocking accept with an error; the fd stays owned
                    // by the listener.
                    unsafe { libc::shutdown(fd, libc::SHUT_RDWR) };
                }
            })
            .map(|_| ())
            .map_err(|e| format!("spawn signal thread: {e}"))
    }

    fn draining() -> bool {
        DRAIN.load(Ordering::SeqCst)
    }

    // ── Lease registry ───────────────────────────────────────────────────

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum LeaseState {
        /// Admitted; FDs may be in flight.
        Pending,
        /// ImportAck received.
        Active,
        /// Closed without Release: charged until the process is gone.
        Orphaned,
    }

    #[derive(Clone, Copy, Debug)]
    struct Lease {
        pid: u32,
        start: u64,
        state: LeaseState,
    }

    #[derive(Default)]
    struct Registry {
        leases: Mutex<BTreeMap<u64, Lease>>,
    }

    impl Registry {
        fn insert(&self, id: u64, lease: Lease) {
            self.leases.lock().insert(id, lease);
        }
        fn set(&self, id: u64, state: LeaseState) {
            if let Some(lease) = self.leases.lock().get_mut(&id) {
                lease.state = state;
            }
        }
        fn remove(&self, id: u64) {
            self.leases.lock().remove(&id);
        }
        /// Drop leases whose process (pid + start time) is gone; PID reuse
        /// cannot release another lease. Returns the remaining ones.
        fn reconcile(&self) -> Vec<(u64, Lease)> {
            let mut leases = self.leases.lock();
            leases.retain(|id, lease| {
                let alive = fabric::proc_start(lease.pid) == Some(lease.start);
                if !alive {
                    eprintln!(
                        "[weight-host] lease {id} pid {} ({:?}) process gone: references released by exit",
                        lease.pid, lease.state
                    );
                }
                alive
            });
            leases.iter().map(|(id, lease)| (*id, *lease)).collect()
        }
    }

    // ── Socket publication ───────────────────────────────────────────────

    /// The socket directory is the generation's private publication area:
    /// it must exist, belong to this user and be mode 0700.
    fn check_socket_dir(socket: &Path) -> Result<(), String> {
        let dir = socket
            .parent()
            .ok_or("socket path has no parent directory")?;
        let md = std::fs::symlink_metadata(dir)
            .map_err(|e| format!("socket directory {}: {e}", dir.display()))?;
        let uid = unsafe { libc::geteuid() };
        if !md.is_dir() || md.uid() != uid || md.mode() & 0o777 != 0o700 {
            return Err(format!(
                "socket directory {} must be a directory owned by uid {uid} with mode 0700 (is {:o})",
                dir.display(),
                md.mode() & 0o7777
            ));
        }
        Ok(())
    }

    /// Refuse to start while a live host publishes on `socket`; a socket
    /// inode nobody listens on is a dead generation's rendezvous and is removed.
    fn refuse_live_or_clear_stale(socket: &Path) -> Result<(), String> {
        let md = match std::fs::symlink_metadata(socket) {
            Ok(md) => md,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(e) => return Err(format!("{}: {e}", socket.display())),
        };
        if !md.file_type().is_socket() {
            return Err(format!(
                "{} exists and is not a socket; refusing to replace it",
                socket.display()
            ));
        }
        match UnixStream::connect(socket) {
            Ok(_) => Err(format!(
                "a live weight host already publishes on {}",
                socket.display()
            )),
            Err(e) if e.raw_os_error() == Some(libc::ECONNREFUSED) => {
                eprintln!(
                    "[weight-host] removing stale socket {} (no listener)",
                    socket.display()
                );
                std::fs::remove_file(socket)
                    .map_err(|e| format!("remove stale {}: {e}", socket.display()))
            }
            Err(e) => Err(format!("probe {}: {e}", socket.display())),
        }
    }

    /// Bind under a temporary name, set 0600, then publish atomically with
    /// `renameat2(RENAME_NOREPLACE)`: a concurrent host is never overwritten.
    fn publish_socket(socket: &Path) -> Result<UnixListener, String> {
        let dir = socket
            .parent()
            .ok_or("socket path has no parent directory")?;
        let name = socket.file_name().ok_or("socket path has no file name")?;
        let mut tmp_name = std::ffi::OsString::from(".");
        tmp_name.push(name);
        tmp_name.push(format!(".{}.tmp", std::process::id()));
        let tmp = dir.join(tmp_name);
        let listener =
            UnixListener::bind(&tmp).map_err(|e| format!("bind {}: {e}", tmp.display()))?;
        let publish = (|| -> Result<(), String> {
            std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o600))
                .map_err(|e| format!("chmod {}: {e}", tmp.display()))?;
            let from =
                std::ffi::CString::new(tmp.as_os_str().as_bytes()).map_err(|e| e.to_string())?;
            let to =
                std::ffi::CString::new(socket.as_os_str().as_bytes()).map_err(|e| e.to_string())?;
            let rc = unsafe {
                libc::renameat2(
                    libc::AT_FDCWD,
                    from.as_ptr(),
                    libc::AT_FDCWD,
                    to.as_ptr(),
                    libc::RENAME_NOREPLACE,
                )
            };
            if rc != 0 {
                return Err(format!(
                    "publish {}: {}",
                    socket.display(),
                    std::io::Error::last_os_error()
                ));
            }
            Ok(())
        })();
        if let Err(e) = publish {
            let _ = std::fs::remove_file(&tmp);
            return Err(e);
        }
        Ok(listener)
    }

    // ── Device claim ─────────────────────────────────────────────────────

    /// Shared fabric claim on exactly one visible card (the fabric pins
    /// `ROCR_VISIBLE_DEVICES`). Returns the claim (held until exit) and the
    /// verified host admission.
    fn claim_card() -> Result<(GpuLocks, fabric::FabricAdmission), String> {
        use hipfire_config::devices::{DeviceRoot, IdentitySource, ObservedDevice};
        let token = std::env::var(fabric::ADMISSION_ENV).map_err(|_| {
            format!(
                "{} is unset: run the host under fabric-run.sh --class correct",
                fabric::ADMISSION_ENV
            )
        })?;
        let admission = fabric::FabricAdmission::verify_for_pid(&token, std::process::id())?;
        let mut locks = GpuLocks::open_shared(&token)?;
        let root = DeviceRoot::system();
        let with_uuid = root.identity_source() == IdentitySource::Hip;
        let hip = hip_bridge::HipRuntime::load().map_err(|e| e.to_string())?;
        let count = hip.device_count().map_err(|e| e.to_string())?;
        if count != 1 {
            return Err(format!(
                "weight host v1 needs exactly one visible device, HIP exposes {count}"
            ));
        }
        let observed = vec![ObservedDevice {
            logical: 0,
            arch: hip.get_arch(0).map_err(|e| e.to_string())?,
            pci_bus_id: hip.device_pci_bus_id(0).map_err(|e| e.to_string())?,
            uuid: with_uuid
                .then(|| hip.device_uuid(0).map_err(|e| e.to_string()))
                .transpose()?,
        }];
        let devices = hipfire_config::devices::identify_observed(&root, &observed)?;
        locks.claim_all(&devices)?;
        Ok((locks, admission))
    }

    fn raise_nofile_limit() -> Result<u64, String> {
        let mut lim = libc::rlimit {
            rlim_cur: 0,
            rlim_max: 0,
        };
        if unsafe { libc::getrlimit(libc::RLIMIT_NOFILE, &mut lim) } != 0 {
            return Err(format!("getrlimit: {}", std::io::Error::last_os_error()));
        }
        if lim.rlim_cur < lim.rlim_max {
            lim.rlim_cur = lim.rlim_max;
            if unsafe { libc::setrlimit(libc::RLIMIT_NOFILE, &lim) } != 0 {
                return Err(format!("setrlimit: {}", std::io::Error::last_os_error()));
            }
        }
        Ok(lim.rlim_cur)
    }

    fn ms(since: Instant) -> f64 {
        since.elapsed().as_secs_f64() * 1e3
    }

    // ── Tenant sessions ──────────────────────────────────────────────────

    struct Shared {
        publication: Arc<WeightPublication>,
        options: ServeOptions,
        admission: fabric::FabricAdmission,
        registry: Registry,
    }

    fn serve_tenant(id: u64, stream: UnixStream, shared: Arc<Shared>) {
        if let Err(e) = serve_tenant_inner(id, stream, &shared) {
            eprintln!("[weight-host] connection {id}: {e}");
        }
    }

    fn serve_tenant_inner(id: u64, stream: UnixStream, shared: &Shared) -> Result<(), String> {
        // Peer credentials, Hello framing and identity (card, source, loader
        // ABI, recipe) are checked here; a mismatch is rejected before any FD.
        let conn = HostConnection::from_stream(stream, shared.publication.clone(), shared.options)
            .map_err(|e| format!("handshake refused: {e}"))?;
        let pid = conn.peer().pid;
        if draining() {
            conn.reject("weight host is draining");
            return Err(format!("pid {pid}: refused, draining"));
        }
        let tenant =
            match fabric::FabricAdmission::verify_for_pid(&conn.hello().admission_token, pid) {
                Ok(tenant) => tenant,
                Err(e) => {
                    conn.reject(&e);
                    return Err(format!("pid {pid}: {e}"));
                }
            };
        if tenant.state_path != shared.admission.state_path {
            let reason = format!(
                "fabric admission {} is for another card than the host's {}",
                tenant.state_path.display(),
                shared.admission.state_path.display()
            );
            conn.reject(&reason);
            return Err(format!("pid {pid}: {reason}"));
        }
        let start = fabric::proc_start(pid)
            .ok_or_else(|| format!("pid {pid} vanished during handshake"))?;
        let mut session = conn
            .admit(|_| {
                shared.registry.insert(
                    id,
                    Lease {
                        pid,
                        start,
                        state: LeaseState::Pending,
                    },
                );
                Ok(())
            })
            .map_err(|e| format!("pid {pid}: admission: {e}"))?;
        let handshake = (|| {
            session.send_allocations()?;
            session.wait_import_ack(Some(shared.options.handshake_timeout))
        })();
        let ack = match handshake {
            Ok(ack) => ack,
            Err(e) => {
                if session.fds_exposed() == 0 {
                    shared.registry.remove(id);
                } else {
                    shared.registry.set(id, LeaseState::Orphaned);
                }
                return Err(format!("pid {pid}: import handshake: {e}"));
            }
        };
        shared.registry.set(id, LeaseState::Active);
        eprintln!(
            "WEIGHT_HOST_LEASE id={id} pid={pid} epoch={} imported_bytes={} fds={} state=active",
            ack.epoch,
            ack.imported_bytes,
            session.fd_count()
        );
        let mut drain_sent = false;
        loop {
            if draining() && !drain_sent {
                if let Err(e) = session.begin_drain() {
                    eprintln!("[weight-host] lease {id} pid {pid}: begin_drain: {e}");
                }
                drain_sent = true;
            }
            match session.poll_release() {
                Ok(ReleasePoll::Pending) => std::thread::sleep(Duration::from_millis(100)),
                Ok(ReleasePoll::Released) => {
                    shared.registry.remove(id);
                    eprintln!("WEIGHT_HOST_LEASE id={id} pid={pid} state=released");
                    return Ok(());
                }
                Ok(ReleasePoll::PeerClosed) => {
                    shared.registry.set(id, LeaseState::Orphaned);
                    eprintln!("WEIGHT_HOST_LEASE id={id} pid={pid} state=closed-without-release");
                    return Ok(());
                }
                Err(e) => {
                    shared.registry.set(id, LeaseState::Orphaned);
                    return Err(format!("pid {pid}: lease: {e}"));
                }
            }
        }
    }

    // ── Main ─────────────────────────────────────────────────────────────

    pub fn run() -> i32 {
        let args = match parse_args() {
            Ok(args) => args,
            Err(e) => {
                eprintln!("hipfire-weight-host: {e}\n{USAGE}");
                return EXIT_USAGE;
            }
        };
        match serve(&args) {
            Ok(code) => code,
            Err(e) => {
                eprintln!("FATAL: hipfire-weight-host: {e}");
                EXIT_FAILED
            }
        }
    }

    fn serve(args: &Args) -> Result<i32, String> {
        // Bootstrap read: `0` is the feature kill switch for hosts too.
        if std::env::var_os("HIPFIRE_WEIGHT_HOST").is_some_and(|v| v == "0") {
            return Err("HIPFIRE_WEIGHT_HOST=0: shared-weight host disabled".into());
        }
        install_signal_waiter()?;
        let nofile = raise_nofile_limit()?;
        check_socket_dir(&args.socket)?;
        refuse_live_or_clear_stale(&args.socket)?;

        // Starting: shared fabric claim, then the GPU.
        let (_locks, admission) = claim_card()?;
        let mut gpu = rdna_compute::Gpu::init().map_err(|e| format!("GPU init: {e}"))?;
        if gpu.arch != shared_weights::SUPPORTED_DEVICE_ARCH {
            return Err(format!(
                "weight host v1 supports {} only, card is {}",
                shared_weights::SUPPORTED_DEVICE_ARCH,
                gpu.arch
            ));
        }
        let card = qcarrier::weight_host_card_identity(&gpu)?;
        eprintln!(
            "[weight-host] starting pid={} card={} pci={} arch={} fabric_entry={} nofile={nofile}",
            std::process::id(),
            card.uuid,
            card.pci_bus_id,
            card.device_arch,
            admission.entry_id
        );

        // Loading: one full-content hash of the read-only source (kept open
        // for the host's lifetime), then the incumbent loader.
        let hash_started = Instant::now();
        let (_source_file, cert) = certify_source(&args.model, "base")
            .map_err(|e| format!("certify {}: {e}", args.model.display()))?;
        let source_sha256 = qcarrier::weight_host_source_sha256(&cert)?;
        let hash_ms = ms(hash_started);
        let load_started = Instant::now();
        let mut hfq = hipfire_runtime::hfq::HfqFile::open(&args.model)
            .map_err(|e| format!("open {}: {e}", args.model.display()))?;
        let config = <hipfire_arch_qwen35::Qwen35 as Architecture>::config_from_hfq(&hfq)
            .map_err(|e| e.to_string())?;
        shared_weights::check_dense_config(&config).map_err(|e| e.to_string())?;
        shared_weights::apply_source_dispatch_flags(&mut gpu, &hfq);
        let weights = <hipfire_arch_qwen35::Qwen35 as Architecture>::load_weights(
            &mut hfq, &config, &mut gpu,
        )?;
        let source_load_ms = ms(load_started);
        if draining() {
            weights.free_gpu(&mut gpu);
            return Err("signalled while loading; nothing published".into());
        }
        let convert_started = Instant::now();
        let recipe =
            shared_weights::resolve_dense_recipe(&gpu, &config).map_err(|e| e.to_string())?;
        let mut exported =
            shared_weights::export_dense_weights(&mut gpu, weights, &config, &recipe)
                .map_err(|e| format!("export: {e}"))?;
        let convert_ms = ms(convert_started);
        let fds = std::mem::take(&mut exported.allocation_fds);
        if fds.len() as u64 + 256 > nofile {
            let _ = exported.free_gpu(&mut gpu);
            return Err(format!(
                "RLIMIT_NOFILE {nofile} too low for {} allocation FDs",
                fds.len()
            ));
        }

        // Publishing: seal, certificate, then the socket last.
        let unpublish = |gpu: &mut rdna_compute::Gpu,
                         exported: shared_weights::ExportedDenseWeights,
                         e: String| {
            match exported.free_gpu(gpu) {
                Ok(()) => e,
                Err(c) => format!("{e}; teardown also failed: {c}"),
            }
        };
        let epoch = match Epoch::random() {
            Ok(epoch) => epoch,
            Err(e) => return Err(unpublish(&mut gpu, exported, format!("epoch: {e}"))),
        };
        let abi = match qcarrier::loader_abi().sha256() {
            Ok(abi) => abi,
            Err(e) => return Err(unpublish(&mut gpu, exported, format!("loader ABI: {e}"))),
        };
        let manifest = exported.manifest.clone();
        let publication =
            match WeightPublication::new(epoch, card, source_sha256, abi, recipe, manifest, fds) {
                Ok(p) => Arc::new(p),
                Err(e) => return Err(unpublish(&mut gpu, exported, format!("publication: {e}"))),
            };
        let cert_path = qcarrier::weight_host_source_certificate_path(&args.socket);
        if let Err(e) = cert.write_to(&cert_path) {
            drop(publication);
            return Err(unpublish(
                &mut gpu,
                exported,
                format!("certificate {}: {e}", cert_path.display()),
            ));
        }
        let listener = match publish_socket(&args.socket) {
            Ok(listener) => listener,
            Err(e) => {
                drop(publication);
                let _ = std::fs::remove_file(&cert_path);
                return Err(unpublish(&mut gpu, exported, e));
            }
        };
        LISTENER_FD.store(listener.as_raw_fd(), Ordering::SeqCst);
        eprintln!(
            "WEIGHT_HOST_READY epoch={epoch} layout_sha256={} unique_bytes={} fds={} source_sha256={source_sha256} loader_abi_sha256={abi} source_hash_ms={hash_ms:.3} source_load_ms={source_load_ms:.3} convert_ms={convert_ms:.3} socket={} pid={}",
            publication.seals().layout_sha256,
            publication.unique_bytes(),
            publication.manifest().allocations.len(),
            args.socket.display(),
            std::process::id(),
        );

        // Ready: accept until a drain signal shuts the listener down.
        let shared = Arc::new(Shared {
            publication,
            options: ServeOptions {
                handshake_timeout: args.handshake_timeout,
                expected_peer_pid: None,
            },
            admission,
            registry: Registry::default(),
        });
        let mut sessions = Vec::new();
        let mut next_id = 0u64;
        if !draining() {
            for stream in listener.incoming() {
                if draining() {
                    break;
                }
                match stream {
                    Ok(stream) => {
                        next_id += 1;
                        let (id, shared) = (next_id, shared.clone());
                        match std::thread::Builder::new()
                            .name(format!("weight-host-lease-{id}"))
                            .spawn(move || serve_tenant(id, stream, shared))
                        {
                            Ok(handle) => sessions.push(handle),
                            Err(e) => eprintln!("[weight-host] connection {id}: spawn: {e}"),
                        }
                    }
                    Err(e) => eprintln!("[weight-host] accept: {e}"),
                }
            }
        }

        // Draining: no new tenants; leases half-closed by their threads.
        let _ = std::fs::remove_file(&args.socket);
        LISTENER_FD.store(-1, Ordering::SeqCst);
        drop(listener);
        let drain_started = Instant::now();
        loop {
            let remaining = shared.registry.reconcile();
            if remaining.is_empty() {
                break;
            }
            if drain_started.elapsed() >= args.drain_timeout {
                let held = remaining
                    .iter()
                    .map(|(id, l)| format!("{id}:pid{}:{:?}", l.pid, l.state))
                    .collect::<Vec<_>>()
                    .join(",");
                eprintln!(
                    "WEIGHT_HOST_DRAIN_TIMEOUT epoch={} occupied_by=[{held}] (generation stays charged until those processes exit)",
                    shared.publication.epoch()
                );
                return Ok(EXIT_DRAIN_TIMEOUT);
            }
            std::thread::sleep(Duration::from_millis(250));
        }
        for handle in sessions {
            let _ = handle.join();
        }
        let Ok(shared) = Arc::try_unwrap(shared) else {
            return Err("internal: lease threads still reference the publication".into());
        };
        let epoch = shared.publication.epoch();
        drop(shared);
        let _ = std::fs::remove_file(&cert_path);

        // Closed: every lease is gone; free the exported owners.
        let teardown = exported
            .free_gpu(&mut gpu)
            .map_err(|e| e.to_string())
            .and_then(|()| gpu.ensure_vmm_cleaned().map_err(|e| e.to_string()));
        match teardown {
            Ok(()) => {
                eprintln!(
                    "WEIGHT_HOST_CLOSED epoch={epoch} drain_ms={:.3}",
                    ms(drain_started)
                );
                Ok(0)
            }
            Err(e) => Err(format!("teardown of epoch {epoch}: {e}")),
        }
    }
}
