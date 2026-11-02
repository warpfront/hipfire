//! Cross-platform process identity facts.
//!
//! Every fact here is either *proven* by the OS or reported as unavailable
//! (`false` / `None` / `Err`). Nothing is guessed:
//!
//! * Linux reads `/proc/<pid>/...`.
//! * Windows uses hand-written `kernel32` / `iphlpapi` bindings.
//! * Any other OS reports unavailable, except bare PID liveness on non-Linux
//!   Unix (`kill(pid, 0)`).
//!
//! `process_start_time` is an opaque per-platform tick count (Linux: clock
//! ticks since boot from `/proc/<pid>/stat`; Windows: the process creation
//! `FILETIME`, 100 ns ticks since 1601). It is only meaningful compared against
//! a value recorded on the same platform.

use std::path::Path;

#[cfg(target_os = "linux")]
use std::path::PathBuf;

/// A process captured by [`scan_processes`].
pub(crate) struct ProcessSnapshot {
    pub(crate) pid: u32,
    pub(crate) rss_mb: u64,
    pub(crate) command: String,
}

/// The process exists (an access-denied process on Windows still exists).
pub(crate) fn pid_alive(pid: u32) -> bool {
    platform::pid_alive(pid)
}

/// Preserve the Unix daemon-pidfile signal-0 probe (including permissions).
pub(crate) fn pid_signalable(pid: i32) -> bool {
    #[cfg(unix)]
    {
        // SAFETY: signal 0 probes only existence and permission.
        pid > 0 && unsafe { libc::kill(pid, 0) } == 0
    }
    #[cfg(not(unix))]
    {
        u32::try_from(pid).is_ok_and(pid_alive)
    }
}

/// Opaque creation stamp of `pid`; `None` when it cannot be read.
pub(crate) fn process_start_time(pid: u32) -> Option<u64> {
    platform::process_start_time(pid)
}

/// The live process `pid` runs the executable `expected`.
///
/// Linux compares `argv[0]` (canonicalised) with `expected`. Windows compares
/// the full image path with `expected`, which must resolve to an absolute path.
/// Unreadable or unsupported ⇒ `false`.
pub(crate) fn process_image_matches(pid: u32, expected: &Path) -> bool {
    platform::process_image_matches(pid, expected)
}

/// The live process `pid` looks like a `hipfire serve`.
///
/// Linux requires `hipfire` and `serve` in the command line. Windows has no
/// permission-free command-line read, so it requires the image basename
/// `hipfire` / `hipfire.exe`; the port and instance token are what prove the
/// rest. Unreadable or unsupported ⇒ `false`.
pub(crate) fn serve_process_matches(pid: u32) -> bool {
    platform::serve_process_matches(pid)
}

/// Whether `pid` owns a TCP listener on `port`; `None` when not provable.
pub(crate) fn pid_owns_listen_port(pid: u32, port: u16) -> Option<bool> {
    platform::pid_owns_listen_port(pid, port)
}

/// The process has exited, or is a zombie awaiting its parent.
pub(crate) fn process_gone(pid: u32) -> bool {
    platform::process_gone(pid)
}

/// Every other process whose command line satisfies `keep`.
///
/// `Err` explains why the enumeration is unavailable on this OS.
pub(crate) fn scan_processes(
    keep: impl FnMut(&str) -> bool,
) -> Result<Vec<ProcessSnapshot>, String> {
    platform::scan_processes(keep)
}

/// Canonical-path equality; any unresolvable side is unequal.
pub(crate) fn same_file_path(a: &Path, b: &Path) -> bool {
    match (std::fs::canonicalize(a), std::fs::canonicalize(b)) {
        (Ok(a), Ok(b)) => a == b,
        _ => false,
    }
}

/// The hipfire home a process runs under, read from its environment with the
/// `ConfigPaths::discover` rule (`HIPFIRE_HOME`, else `$HOME/.hipfire`).
/// `None` when the environment is unreadable (another user's process).
#[cfg(target_os = "linux")]
pub(crate) fn process_hipfire_home(pid: u32) -> Option<PathBuf> {
    use std::os::unix::ffi::OsStrExt;
    let environ = std::fs::read(format!("/proc/{pid}/environ")).ok()?;
    let mut hipfire_home = None;
    let mut home = None;
    for var in environ.split(|byte| *byte == 0) {
        if let Some(value) = var.strip_prefix(b"HIPFIRE_HOME=") {
            hipfire_home = Some(PathBuf::from(std::ffi::OsStr::from_bytes(value)));
        } else if let Some(value) = var.strip_prefix(b"HOME=") {
            home = Some(PathBuf::from(std::ffi::OsStr::from_bytes(value)));
        }
    }
    hipfire_home.or_else(|| home.map(|home| home.join(".hipfire")))
}

/// Direct children of `pid`, from every `/proc/<pid>/task/*/children`.
#[cfg(target_os = "linux")]
pub(crate) fn child_pids(pid: u32) -> Vec<u32> {
    let mut children = Vec::new();
    if let Ok(tasks) = std::fs::read_dir(format!("/proc/{pid}/task")) {
        for task in tasks.flatten() {
            if let Ok(raw) = std::fs::read_to_string(task.path().join("children")) {
                children.extend(
                    raw.split_whitespace()
                        .filter_map(|pid| pid.parse::<u32>().ok()),
                );
            }
        }
    }
    children
}

/// First NUL-terminated field of a raw `/proc/<pid>/cmdline`.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn cmdline_argv0(cmdline: &[u8]) -> &[u8] {
    cmdline.split(|byte| *byte == 0).next().unwrap_or_default()
}

/// The `hipfire serve` command-line rule of `validate_serve_pid`.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn cmdline_is_serve(cmdline: &[u8]) -> bool {
    let cmdline = String::from_utf8_lossy(cmdline).replace('\0', " ");
    cmdline.contains("hipfire") && cmdline.contains("serve")
}

/// `/proc/<pid>/stat` state `Z` (zombie). The state follows the parenthesised
/// command name.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn stat_is_zombie(stat: &str) -> bool {
    stat.rsplit_once(") ")
        .is_some_and(|(_, rest)| rest.starts_with('Z'))
}

/// A Windows path in comparison form: `/` → `\`, `\\?\` / `\\?\UNC\` verbatim
/// prefixes removed, lower-cased.
#[cfg_attr(not(windows), allow(dead_code))]
fn normalize_windows_path(path: &str) -> String {
    let path = path.replace('/', "\\");
    let path = if let Some(rest) = path.strip_prefix("\\\\?\\UNC\\") {
        format!("\\\\{rest}")
    } else if let Some(rest) = path.strip_prefix("\\\\?\\") {
        rest.to_owned()
    } else {
        path
    };
    path.to_lowercase()
}

/// `normalized` is a drive-absolute (`c:\...`) or UNC (`\\server\...`) path.
#[cfg_attr(not(windows), allow(dead_code))]
fn normalized_windows_path_is_absolute(normalized: &str) -> bool {
    let bytes = normalized.as_bytes();
    (bytes.len() >= 3 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':' && bytes[2] == b'\\')
        || (normalized.starts_with("\\\\") && normalized.len() > 2)
}

/// The image at `actual` is exactly the absolute executable `expected`.
/// A relative `expected` never matches.
#[cfg_attr(not(windows), allow(dead_code))]
fn windows_image_matches(actual: &str, expected: &str) -> bool {
    let actual = normalize_windows_path(actual);
    let expected = normalize_windows_path(expected);
    normalized_windows_path_is_absolute(&expected)
        && normalized_windows_path_is_absolute(&actual)
        && actual == expected
}

/// The image basename is `hipfire` or `hipfire.exe`.
#[cfg_attr(not(windows), allow(dead_code))]
fn windows_image_is_hipfire(image: &str) -> bool {
    let name = image.rsplit(['\\', '/']).next().unwrap_or_default();
    name.eq_ignore_ascii_case("hipfire.exe") || name.eq_ignore_ascii_case("hipfire")
}

/// Whether a process whose `GetExitCodeProcess` is `code` is still running.
/// `STILL_ACTIVE` (259) is also a legal exit code, so a recorded exit time
/// (`exit_time_zero == Some(false)`) overrides it. An unreadable exit time is
/// not proof of exit.
#[cfg_attr(not(windows), allow(dead_code))]
fn exit_state_is_running(code: u32, exit_time_zero: Option<bool>) -> bool {
    const STILL_ACTIVE: u32 = 259;
    code == STILL_ACTIVE && exit_time_zero != Some(false)
}

/// Port, in network byte order in the low 16 bits of a `DWORD`, to host order.
#[cfg_attr(not(windows), allow(dead_code))]
fn dword_port(raw: [u8; 4]) -> u16 {
    ((u32::from_le_bytes(raw) & 0xFFFF) as u16).swap_bytes()
}

/// Owning PIDs of the listeners on `port` in a `GetExtendedTcpTable`
/// `TCP_TABLE_OWNER_PID_LISTENER` blob (`DWORD` entry count, then rows of
/// `row_size` bytes). `None` when the blob is shorter than it claims.
#[cfg_attr(not(windows), allow(dead_code))]
fn listener_owners(
    table: &[u8],
    row_size: usize,
    port_offset: usize,
    pid_offset: usize,
    port: u16,
) -> Option<Vec<u32>> {
    let count = u32::from_le_bytes(table.get(..4)?.try_into().ok()?) as usize;
    let rows = table.get(4..)?;
    if rows.len() < count.checked_mul(row_size)? {
        return None;
    }
    let mut owners = Vec::new();
    for row in rows.chunks_exact(row_size).take(count) {
        let row_port = dword_port(row[port_offset..port_offset + 4].try_into().ok()?);
        if row_port == port {
            owners.push(u32::from_le_bytes(
                row[pid_offset..pid_offset + 4].try_into().ok()?,
            ));
        }
    }
    Some(owners)
}

/// `MIB_TCPROW_OWNER_PID`: state, local addr, local port, remote addr, remote
/// port, owning pid (6 × `DWORD`).
#[cfg_attr(not(windows), allow(dead_code))]
fn listener_owners_v4(table: &[u8], port: u16) -> Option<Vec<u32>> {
    listener_owners(table, 24, 8, 20, port)
}

/// `MIB_TCP6ROW_OWNER_PID`: local addr[16], scope id, local port, remote
/// addr[16], scope id, remote port, state, owning pid (56 bytes).
#[cfg_attr(not(windows), allow(dead_code))]
fn listener_owners_v6(table: &[u8], port: u16) -> Option<Vec<u32>> {
    listener_owners(table, 56, 20, 52, port)
}

/// Merge per-family listener owner lists. A listener owned by `pid` proves
/// ownership even if another family could not be read; absence is only proven
/// when every family was read.
#[cfg_attr(not(windows), allow(dead_code))]
fn combine_listener_results(pid: u32, families: &[Option<Vec<u32>>]) -> Option<bool> {
    if families
        .iter()
        .flatten()
        .any(|owners| owners.contains(&pid))
    {
        return Some(true);
    }
    families.iter().all(Option::is_some).then_some(false)
}

#[cfg(target_os = "linux")]
mod platform {
    use super::*;
    use std::collections::BTreeSet;
    use std::fs;

    pub(super) fn pid_alive(pid: u32) -> bool {
        Path::new(&format!("/proc/{pid}")).exists()
    }

    pub(super) fn process_start_time(pid: u32) -> Option<u64> {
        let stat = fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
        let after_comm = stat.rsplit_once(") ")?.1;
        after_comm.split_whitespace().nth(19)?.parse().ok()
    }

    pub(super) fn process_image_matches(pid: u32, expected: &Path) -> bool {
        use std::os::unix::ffi::OsStrExt;
        let Ok(cmdline) = fs::read(format!("/proc/{pid}/cmdline")) else {
            return false;
        };
        let argv0 = cmdline_argv0(&cmdline);
        !argv0.is_empty() && same_file_path(Path::new(std::ffi::OsStr::from_bytes(argv0)), expected)
    }

    pub(super) fn serve_process_matches(pid: u32) -> bool {
        let cmdline = fs::read(format!("/proc/{pid}/cmdline")).unwrap_or_default();
        cmdline_is_serve(&cmdline)
    }

    pub(super) fn pid_owns_listen_port(pid: u32, port: u16) -> Option<bool> {
        let mut listen_inodes = BTreeSet::new();
        let port_hex = format!("{port:04X}");
        let mut read_any = false;
        for table in ["/proc/net/tcp", "/proc/net/tcp6"] {
            let Ok(raw) = fs::read_to_string(table) else {
                continue;
            };
            read_any = true;
            for line in raw.lines().skip(1) {
                let columns = line.split_whitespace().collect::<Vec<_>>();
                if columns.len() < 10 || columns[3] != "0A" {
                    continue;
                }
                let Some((_, local_port)) = columns[1].rsplit_once(':') else {
                    continue;
                };
                if local_port.eq_ignore_ascii_case(&port_hex) {
                    listen_inodes.insert(columns[9].to_owned());
                }
            }
        }
        if !read_any {
            return None;
        }
        if listen_inodes.is_empty() {
            return Some(false);
        }
        let entries = fs::read_dir(format!("/proc/{pid}/fd")).ok()?;
        for entry in entries.flatten() {
            let Ok(target) = fs::read_link(entry.path()) else {
                continue;
            };
            let target = target.to_string_lossy();
            if let Some(inode) = target
                .strip_prefix("socket:[")
                .and_then(|value| value.strip_suffix(']'))
            {
                if listen_inodes.contains(inode) {
                    return Some(true);
                }
            }
        }
        Some(false)
    }

    pub(super) fn process_gone(pid: u32) -> bool {
        match fs::read_to_string(format!("/proc/{pid}/stat")) {
            Ok(stat) => stat_is_zombie(&stat),
            Err(_) => true,
        }
    }

    pub(super) fn scan_processes(
        mut keep: impl FnMut(&str) -> bool,
    ) -> Result<Vec<ProcessSnapshot>, String> {
        let mut found = Vec::new();
        let Ok(entries) = fs::read_dir("/proc") else {
            return Ok(found);
        };
        for entry in entries.flatten() {
            let Some(pid) = entry
                .file_name()
                .to_str()
                .and_then(|name| name.parse::<u32>().ok())
            else {
                continue;
            };
            if pid == std::process::id() {
                continue;
            }
            let Ok(raw) = fs::read(entry.path().join("cmdline")) else {
                continue;
            };
            let command = String::from_utf8_lossy(&raw)
                .replace('\0', " ")
                .trim()
                .to_owned();
            if command.is_empty() || !keep(&command) {
                continue;
            }
            let rss_mb = fs::read_to_string(entry.path().join("status"))
                .ok()
                .and_then(|status| {
                    status.lines().find_map(|line| {
                        line.strip_prefix("VmRSS:")?
                            .split_whitespace()
                            .next()?
                            .parse::<u64>()
                            .ok()
                    })
                })
                .unwrap_or(0)
                / 1024;
            found.push(ProcessSnapshot {
                pid,
                rss_mb,
                command,
            });
        }
        Ok(found)
    }
}

#[cfg(windows)]
mod platform {
    use super::*;
    use std::ffi::c_void;

    type Handle = *mut c_void;

    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    struct FileTime {
        low: u32,
        high: u32,
    }

    impl FileTime {
        fn ticks(self) -> u64 {
            (u64::from(self.high) << 32) | u64::from(self.low)
        }
    }

    #[link(name = "kernel32")]
    extern "system" {
        fn OpenProcess(desired_access: u32, inherit_handle: i32, process_id: u32) -> Handle;
        fn CloseHandle(object: Handle) -> i32;
        fn GetExitCodeProcess(process: Handle, exit_code: *mut u32) -> i32;
        fn GetProcessTimes(
            process: Handle,
            creation: *mut FileTime,
            exit: *mut FileTime,
            kernel: *mut FileTime,
            user: *mut FileTime,
        ) -> i32;
        fn QueryFullProcessImageNameW(
            process: Handle,
            flags: u32,
            exe_name: *mut u16,
            size: *mut u32,
        ) -> i32;
        fn GetLastError() -> u32;
    }

    #[link(name = "iphlpapi")]
    extern "system" {
        fn GetExtendedTcpTable(
            table: *mut c_void,
            size: *mut u32,
            order: i32,
            family: u32,
            class: i32,
            reserved: u32,
        ) -> u32;
    }

    const PROCESS_QUERY_LIMITED_INFORMATION: u32 = 0x1000;
    const ERROR_ACCESS_DENIED: u32 = 5;
    const ERROR_INSUFFICIENT_BUFFER: u32 = 122;
    const NO_ERROR: u32 = 0;
    const AF_INET: u32 = 2;
    const AF_INET6: u32 = 23;
    const TCP_TABLE_OWNER_PID_LISTENER: i32 = 3;

    /// Closes the handle on drop.
    struct OwnedHandle(Handle);

    impl Drop for OwnedHandle {
        fn drop(&mut self) {
            // SAFETY: the handle came from a successful OpenProcess and is
            // closed exactly once.
            unsafe { CloseHandle(self.0) };
        }
    }

    /// `Err` carries the Win32 error code.
    fn open_query(pid: u32) -> Result<OwnedHandle, u32> {
        // SAFETY: plain value arguments; a null return is a failure.
        let handle = unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid) };
        if handle.is_null() {
            // SAFETY: reads the calling thread's last-error value.
            Err(unsafe { GetLastError() })
        } else {
            Ok(OwnedHandle(handle))
        }
    }

    /// `(creation, exit)` times of the process.
    fn process_times(handle: &OwnedHandle) -> Option<(FileTime, FileTime)> {
        let mut creation = FileTime::default();
        let mut exit = FileTime::default();
        let mut kernel = FileTime::default();
        let mut user = FileTime::default();
        // SAFETY: the handle is live and every out-pointer is a valid FILETIME.
        let ok =
            unsafe { GetProcessTimes(handle.0, &mut creation, &mut exit, &mut kernel, &mut user) };
        (ok != 0).then_some((creation, exit))
    }

    /// Running unless the process is proven exited. A failed query keeps the
    /// process "running": an existing handle is not proof of exit.
    fn handle_running(handle: &OwnedHandle) -> bool {
        let mut code = 0u32;
        // SAFETY: the handle is live and `code` is a valid out-pointer.
        if unsafe { GetExitCodeProcess(handle.0, &mut code) } == 0 {
            return true;
        }
        let exit_time_zero = process_times(handle).map(|(_, exit)| exit.ticks() == 0);
        exit_state_is_running(code, exit_time_zero)
    }

    fn image_path(handle: &OwnedHandle) -> Option<String> {
        let mut buffer = vec![0u16; 32768];
        let mut size = buffer.len() as u32;
        // SAFETY: the handle is live; `buffer` holds `size` UTF-16 units.
        let ok = unsafe { QueryFullProcessImageNameW(handle.0, 0, buffer.as_mut_ptr(), &mut size) };
        if ok == 0 || size == 0 {
            return None;
        }
        Some(String::from_utf16_lossy(&buffer[..size as usize]))
    }

    pub(super) fn pid_alive(pid: u32) -> bool {
        match open_query(pid) {
            Ok(handle) => handle_running(&handle),
            // Access denied means the process exists but is not ours to query.
            Err(error) => error == ERROR_ACCESS_DENIED,
        }
    }

    pub(super) fn process_start_time(pid: u32) -> Option<u64> {
        let handle = open_query(pid).ok()?;
        let (creation, _) = process_times(&handle)?;
        let ticks = creation.ticks();
        (ticks != 0).then_some(ticks)
    }

    pub(super) fn process_image_matches(pid: u32, expected: &Path) -> bool {
        let expected = std::fs::canonicalize(expected).unwrap_or_else(|_| expected.to_path_buf());
        let Ok(handle) = open_query(pid) else {
            return false;
        };
        handle_running(&handle)
            && image_path(&handle)
                .is_some_and(|image| windows_image_matches(&image, &expected.to_string_lossy()))
    }

    pub(super) fn serve_process_matches(pid: u32) -> bool {
        let Ok(handle) = open_query(pid) else {
            return false;
        };
        handle_running(&handle)
            && image_path(&handle).is_some_and(|image| windows_image_is_hipfire(&image))
    }

    /// Listener owners on `port` for one address family; `None` when the table
    /// cannot be read.
    fn listener_table(family: u32, port: u16) -> Option<Vec<u32>> {
        let mut size = 0u32;
        let mut buffer: Vec<u32> = Vec::new();
        // The table can grow between the sizing call and the read.
        for _ in 0..8 {
            let table = if buffer.is_empty() {
                std::ptr::null_mut()
            } else {
                buffer.as_mut_ptr().cast::<c_void>()
            };
            // SAFETY: `table` is null with `size == 0`, or a writable
            // DWORD-aligned buffer of at least `size` bytes.
            let status = unsafe {
                GetExtendedTcpTable(table, &mut size, 0, family, TCP_TABLE_OWNER_PID_LISTENER, 0)
            };
            match status {
                NO_ERROR => {
                    let bytes = (size as usize).min(buffer.len() * 4);
                    // SAFETY: `buffer` owns at least `bytes` initialised bytes.
                    let table =
                        unsafe { std::slice::from_raw_parts(buffer.as_ptr().cast::<u8>(), bytes) };
                    return if family == AF_INET6 {
                        listener_owners_v6(table, port)
                    } else {
                        listener_owners_v4(table, port)
                    };
                }
                ERROR_INSUFFICIENT_BUFFER => {
                    buffer = vec![0u32; (size as usize).div_ceil(4).max(1)];
                }
                _ => return None,
            }
        }
        None
    }

    pub(super) fn pid_owns_listen_port(pid: u32, port: u16) -> Option<bool> {
        combine_listener_results(
            pid,
            &[
                listener_table(AF_INET, port),
                listener_table(AF_INET6, port),
            ],
        )
    }

    pub(super) fn process_gone(pid: u32) -> bool {
        !pid_alive(pid)
    }

    pub(super) fn scan_processes(
        _keep: impl FnMut(&str) -> bool,
    ) -> Result<Vec<ProcessSnapshot>, String> {
        Err(unavailable_scan_reason())
    }
}

#[cfg(not(any(target_os = "linux", windows)))]
mod platform {
    use super::*;

    #[cfg(unix)]
    pub(super) fn pid_alive(pid: u32) -> bool {
        let Ok(pid) = i32::try_from(pid) else {
            return false;
        };
        // SAFETY: signal 0 only probes for existence and permission.
        pid > 0
            && (unsafe { libc::kill(pid, 0) } == 0
                || std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM))
    }

    #[cfg(not(unix))]
    pub(super) fn pid_alive(_pid: u32) -> bool {
        false
    }

    pub(super) fn process_start_time(_pid: u32) -> Option<u64> {
        None
    }

    pub(super) fn process_image_matches(_pid: u32, _expected: &Path) -> bool {
        false
    }

    pub(super) fn serve_process_matches(_pid: u32) -> bool {
        false
    }

    pub(super) fn pid_owns_listen_port(_pid: u32, _port: u16) -> Option<bool> {
        None
    }

    pub(super) fn process_gone(pid: u32) -> bool {
        !pid_alive(pid)
    }

    pub(super) fn scan_processes(
        _keep: impl FnMut(&str) -> bool,
    ) -> Result<Vec<ProcessSnapshot>, String> {
        Err(unavailable_scan_reason())
    }
}

#[cfg(not(target_os = "linux"))]
fn unavailable_scan_reason() -> String {
    format!(
        "process command lines cannot be read on {}; quantize and HF upload processes are not listed",
        std::env::consts::OS
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table(rows: &[Vec<u8>]) -> Vec<u8> {
        let mut blob = (rows.len() as u32).to_le_bytes().to_vec();
        for row in rows {
            blob.extend_from_slice(row);
        }
        blob
    }

    fn v4_row(port: u16, pid: u32) -> Vec<u8> {
        let mut row = Vec::new();
        row.extend_from_slice(&2u32.to_le_bytes()); // MIB_TCP_STATE_LISTEN
        row.extend_from_slice(&0u32.to_le_bytes()); // local addr
        row.extend_from_slice(&u32::from(port.swap_bytes()).to_le_bytes());
        row.extend_from_slice(&0u32.to_le_bytes()); // remote addr
        row.extend_from_slice(&0u32.to_le_bytes()); // remote port
        row.extend_from_slice(&pid.to_le_bytes());
        row
    }

    fn v6_row(port: u16, pid: u32) -> Vec<u8> {
        let mut row = vec![0u8; 16]; // local addr
        row.extend_from_slice(&0u32.to_le_bytes()); // scope id
        row.extend_from_slice(&u32::from(port.swap_bytes()).to_le_bytes());
        row.extend_from_slice(&[0u8; 16]); // remote addr
        row.extend_from_slice(&0u32.to_le_bytes()); // scope id
        row.extend_from_slice(&0u32.to_le_bytes()); // remote port
        row.extend_from_slice(&2u32.to_le_bytes()); // state
        row.extend_from_slice(&pid.to_le_bytes());
        row
    }

    #[test]
    fn listener_table_v4_selects_owners_of_the_port() {
        let blob = table(&[v4_row(8080, 41), v4_row(9000, 42), v4_row(8080, 43)]);
        assert_eq!(listener_owners_v4(&blob, 8080), Some(vec![41, 43]));
        assert_eq!(listener_owners_v4(&blob, 9000), Some(vec![42]));
        assert_eq!(listener_owners_v4(&blob, 1), Some(Vec::new()));
    }

    #[test]
    fn listener_table_v6_selects_owners_of_the_port() {
        let blob = table(&[v6_row(8080, 7), v6_row(8081, 8)]);
        assert_eq!(listener_owners_v6(&blob, 8081), Some(vec![8]));
        assert_eq!(listener_owners_v6(&blob, 8080), Some(vec![7]));
    }

    #[test]
    fn listener_table_rejects_truncated_blobs() {
        let mut blob = table(&[v4_row(80, 1), v4_row(80, 2)]);
        blob.truncate(blob.len() - 1);
        assert_eq!(listener_owners_v4(&blob, 80), None);
        assert_eq!(listener_owners_v4(&[1, 0], 80), None);
        assert_eq!(listener_owners_v6(&table(&[v4_row(80, 1)]), 80), None);
        // An empty table is a proven absence, not a failure.
        assert_eq!(listener_owners_v4(&table(&[]), 80), Some(Vec::new()));
    }

    #[test]
    fn dword_port_is_network_order_in_the_low_half() {
        assert_eq!(dword_port(0x5000u32.to_le_bytes()), 80);
        // The high half is padding and ignored.
        assert_eq!(dword_port(0xABCD_1F90u32.to_le_bytes()), 0x901F);
        assert_eq!(
            dword_port(u32::from(8080u16.swap_bytes()).to_le_bytes()),
            8080
        );
    }

    #[test]
    fn listener_results_prove_ownership_conservatively() {
        // Positive proof survives an unreadable family.
        assert_eq!(
            combine_listener_results(5, &[Some(vec![5]), None]),
            Some(true)
        );
        // Another owner on every family is a proven "not ours".
        assert_eq!(
            combine_listener_results(5, &[Some(vec![6]), Some(Vec::new())]),
            Some(false)
        );
        // Absence on one family with the other unreadable is unproven.
        assert_eq!(combine_listener_results(5, &[Some(vec![6]), None]), None);
        assert_eq!(combine_listener_results(5, &[None, None]), None);
    }

    #[test]
    fn windows_image_match_is_exact_absolute_and_case_insensitive() {
        let daemon = r"C:\Users\kaden\.hipfire\bin\daemon.exe";
        assert!(windows_image_matches(daemon, daemon));
        assert!(windows_image_matches(
            r"c:\users\KADEN\.hipfire\bin\Daemon.EXE",
            daemon
        ));
        assert!(windows_image_matches(
            r"C:\Users\kaden\.hipfire\bin\daemon.exe",
            r"\\?\C:\Users\kaden\.hipfire\bin\daemon.exe"
        ));
        assert!(windows_image_matches(
            r"C:\Users\kaden\.hipfire\bin\daemon.exe",
            "C:/Users/kaden/.hipfire/bin/daemon.exe"
        ));
        assert!(windows_image_matches(
            r"\\srv\share\bin\daemon.exe",
            r"\\?\UNC\srv\share\bin\daemon.exe"
        ));
        // Same basename in another directory is not this daemon.
        assert!(!windows_image_matches(r"C:\Other\bin\daemon.exe", daemon));
        // A relative expectation never matches, even textually equal.
        assert!(!windows_image_matches(r"bin\daemon.exe", r"bin\daemon.exe"));
        assert!(!windows_image_matches("daemon.exe", "daemon.exe"));
    }

    #[test]
    fn windows_serve_image_requires_hipfire_basename() {
        assert!(windows_image_is_hipfire(
            r"C:\Users\k\.hipfire\bin\hipfire.exe"
        ));
        assert!(windows_image_is_hipfire(r"C:\x\HIPFIRE.EXE"));
        assert!(windows_image_is_hipfire("/opt/hipfire"));
        assert!(!windows_image_is_hipfire(r"C:\x\hipfire-tui.exe"));
        assert!(!windows_image_is_hipfire(r"C:\x\daemon.exe"));
        assert!(!windows_image_is_hipfire(r"C:\hipfire.exe\daemon.exe"));
        assert!(!windows_image_is_hipfire(""));
    }

    #[test]
    fn exit_state_treats_still_active_with_exit_time_as_exited() {
        assert!(exit_state_is_running(259, Some(true)));
        assert!(exit_state_is_running(259, None));
        assert!(!exit_state_is_running(259, Some(false)));
        assert!(!exit_state_is_running(0, Some(true)));
        assert!(!exit_state_is_running(1, None));
    }

    #[test]
    fn cmdline_rules_match_the_serve_validation_condition() {
        assert!(cmdline_is_serve(
            b"/home/u/.hipfire/bin/hipfire\0serve\0-d\0"
        ));
        assert!(!cmdline_is_serve(b"/home/u/.hipfire/bin/hipfire\0ps\0"));
        assert!(!cmdline_is_serve(b"/usr/bin/nginx\0serve\0"));
        assert!(!cmdline_is_serve(b""));
        assert_eq!(cmdline_argv0(b"/bin/daemon\0--flag\0"), b"/bin/daemon");
        assert_eq!(cmdline_argv0(b""), b"");
        assert_eq!(cmdline_argv0(b"\0x"), b"");
    }

    #[test]
    fn stat_zombie_state_follows_the_command_name() {
        assert!(stat_is_zombie("12 (daemon) Z 1 12"));
        assert!(stat_is_zombie("12 (we) ird) (name) Z 1"));
        assert!(!stat_is_zombie("12 (daemon) S 1 12"));
        assert!(!stat_is_zombie("12 (Z) S 1 12"));
        assert!(!stat_is_zombie("garbage"));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn linux_own_process_facts_are_readable() {
        let me = std::process::id();
        assert!(pid_alive(me));
        assert!(!process_gone(me));
        assert!(process_start_time(me).is_some());
        assert_eq!(process_start_time(me), process_start_time(me));
        assert!(!process_image_matches(me, Path::new("/")));
        assert!(!process_image_matches(me, Path::new("/nonexistent/daemon")));
        assert!(child_pids(me).iter().all(|&pid| pid != me));
        assert!(process_hipfire_home(me).is_some() || std::env::var_os("HOME").is_none());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn linux_absent_pid_has_no_identity() {
        let absent = u32::MAX - 1;
        assert!(!pid_alive(absent));
        assert!(process_gone(absent));
        assert_eq!(process_start_time(absent), None);
        assert!(!process_image_matches(absent, Path::new("/bin/sh")));
        assert!(!serve_process_matches(absent));
        assert!(matches!(
            pid_owns_listen_port(absent, 1),
            Some(false) | None
        ));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn linux_listener_ownership_is_proven_for_a_bound_socket() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        assert_eq!(pid_owns_listen_port(std::process::id(), port), Some(true));
        drop(listener);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn linux_scan_filters_by_command_and_skips_self() {
        let all = scan_processes(|_| true).unwrap();
        assert!(all.iter().all(|process| process.pid != std::process::id()));
        let none = scan_processes(|_| false).unwrap();
        assert!(none.is_empty());
    }

    #[cfg(not(target_os = "linux"))]
    #[test]
    fn scan_unavailable_names_the_reason() {
        let reason = scan_processes(|_| true).err().unwrap();
        assert!(reason.contains(std::env::consts::OS));
        assert!(reason.contains("not listed"));
    }
}
