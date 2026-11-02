//! Always-on, process-global per-phase model-load timers.
//!
//! Measurement only: buckets are touched per tensor / per load phase, never
//! per kernel launch, and nothing here synchronises the GPU. A bucket that was
//! never recorded is absent from the summary (an uninstrumented path is
//! distinguishable from a measured zero).
//!
//! Naming: a dotted name `parent.child` is a child of `parent`. Children are
//! printed nested and are excluded from the top-level sum (`other` = total −
//! Σ top-level, clamped at 0). A parent that was never recorded itself is
//! synthesised from the sum of its children. Buckets recorded with
//! [`add_overlapped`] run concurrently with the load and are flagged and
//! excluded from the sum.
//!
//! Scope: [`begin_load`] bumps a generation and activates recording;
//! [`finish_load`] (called right before the `loaded` ACK) deactivates it, so
//! records between an ACK and the next `begin_load` (inference-time uploads)
//! are dropped. A thread that outlives its load records through
//! [`add_overlapped_for`] with the generation it captured at spawn.
//!
//! Caveats: the process `storage.read_bytes` / major-fault delta covers the
//! whole process and so includes background threads (PLE fill and prewarm,
//! JIT workers), not just the loader thread. The per-thread
//! `*.storage_read` children carry thread storage bytes against the whole
//! phase's wall time and are flagged `phase_wall` (amortized). Parallel
//! EP-rank loads record into the same buckets, so their calls, bytes and
//! times sum across ranks.

use std::sync::Mutex;
use std::time::Instant;

pub const SCHEMA: &str = "hipfire.load_phases.v1";

pub const RATE_PHASE_WALL: &str = "phase_wall";
pub const RATE_THREAD_WALL: &str = "thread_wall_incl_pauses";

#[derive(Clone, Debug)]
struct Bucket {
    name: &'static str,
    ns: u64,
    calls: u64,
    bytes: u64,
    overlapped: bool,
    rate_basis: Option<&'static str>,
}

#[derive(Clone, Copy, Debug)]
struct IoSnapshot {
    read_bytes: u64,
    major_faults: u64,
}

struct State {
    started: Option<Instant>,
    io0: Option<IoSnapshot>,
    generation: u64,
    active: bool,
    buckets: Vec<Bucket>,
    notes: Vec<(&'static str, String)>,
}

static STATE: Mutex<State> = Mutex::new(State {
    started: None,
    io0: None,
    generation: 0,
    active: false,
    buckets: Vec::new(),
    notes: Vec::new(),
});

fn lock() -> std::sync::MutexGuard<'static, State> {
    STATE.lock().unwrap_or_else(|e| e.into_inner())
}

fn read_io_field(path: &str, key: &str) -> Option<u64> {
    let text = std::fs::read_to_string(path).ok()?;
    text.lines()
        .find_map(|l| l.strip_prefix(key))
        .and_then(|v| v.trim().parse().ok())
}

/// Field 12 (`majflt`) of a `/proc/.../stat` file. The comm field may contain
/// spaces and parentheses, so parse after the last `)`; the next token is
/// field 3.
fn read_majflt(path: &str) -> Option<u64> {
    let text = std::fs::read_to_string(path).ok()?;
    let rest = &text[text.rfind(')')? + 1..];
    rest.split_whitespace().nth(12 - 3)?.parse().ok()
}

fn process_io() -> Option<IoSnapshot> {
    Some(IoSnapshot {
        read_bytes: read_io_field("/proc/self/io", "read_bytes:")?,
        major_faults: read_majflt("/proc/self/stat")?,
    })
}

/// Calling thread's `(storage read_bytes, major faults)`, `None` when
/// `/proc/thread-self` is unavailable (non-Linux, restricted procfs).
pub fn thread_io() -> Option<(u64, u64)> {
    Some((
        read_io_field("/proc/thread-self/io", "read_bytes:")?,
        read_majflt("/proc/thread-self/stat")?,
    ))
}

/// Reset every bucket and note, bump the load generation, mark the registry
/// active and start the load clock.
pub fn begin_load() {
    let io0 = process_io();
    let mut st = lock();
    st.generation = st.generation.wrapping_add(1);
    st.active = true;
    st.started = Some(Instant::now());
    st.io0 = io0;
    st.buckets.clear();
    st.notes.clear();
}

/// The current load generation (bumped by every [`begin_load`]). A thread
/// that outlives its load captures it at spawn and records through
/// [`add_overlapped_for`], so a late record cannot land in a newer load.
pub fn generation() -> u64 {
    lock().generation
}

fn record(
    name: &'static str,
    ns: u64,
    bytes: u64,
    overlapped: bool,
    rate_basis: Option<&'static str>,
    for_generation: Option<u64>,
) {
    let mut st = lock();
    // Between a finished load and the next begin_load (inference-time
    // uploads) and for a late record from an older load: drop.
    if !st.active || for_generation.is_some_and(|g| g != st.generation) {
        return;
    }
    if let Some(b) = st.buckets.iter_mut().find(|b| b.name == name) {
        b.ns = b.ns.saturating_add(ns);
        b.calls += 1;
        b.bytes = b.bytes.saturating_add(bytes);
        b.overlapped |= overlapped;
        b.rate_basis = b.rate_basis.or(rate_basis);
    } else {
        st.buckets.push(Bucket {
            name,
            ns,
            calls: 1,
            bytes,
            overlapped,
            rate_basis,
        });
    }
}

/// Accumulate `ns` and `bytes` (one call) into bucket `name`.
pub fn add(name: &'static str, ns: u64, bytes: u64) {
    record(name, ns, bytes, false, None, None);
}

/// Like [`add`] for a bucket whose bytes were measured over a longer scope
/// than `ns` describes (a thread's storage bytes against the whole phase's
/// wall time): the GB/s is amortized, flagged `rate_basis: "phase_wall"`.
pub fn add_amortized(name: &'static str, ns: u64, bytes: u64) {
    record(name, ns, bytes, false, Some(RATE_PHASE_WALL), None);
}

/// Like [`add`], for work that runs concurrently with the load (its own
/// thread, never awaited by ready): printed flagged, excluded from the sum.
pub fn add_overlapped(name: &'static str, ns: u64, bytes: u64) {
    record(name, ns, bytes, true, None, None);
}

/// [`add_overlapped`] from a thread spawned during load `generation` (see
/// [`generation`]): dropped when a newer load has begun or the load ended.
/// Its rate is bytes over the thread's wall time including its own throttle
/// pauses (`rate_basis: "thread_wall_incl_pauses"`).
pub fn add_overlapped_for(generation: u64, name: &'static str, ns: u64, bytes: u64) {
    record(
        name,
        ns,
        bytes,
        true,
        Some(RATE_THREAD_WALL),
        Some(generation),
    );
}

/// Record a categorical fact (`kernels=lazy`); a repeated name replaces.
/// Dropped while the registry is inactive.
pub fn note(name: &'static str, value: impl Into<String>) {
    let value = value.into();
    let mut st = lock();
    if !st.active {
        return;
    }
    if let Some(n) = st.notes.iter_mut().find(|n| n.0 == name) {
        n.1 = value;
    } else {
        st.notes.push((name, value));
    }
}

/// Adds the elapsed time (and optional bytes) of its scope to a bucket on drop.
pub struct SpanGuard {
    name: &'static str,
    started: Instant,
    bytes: u64,
}

impl SpanGuard {
    pub fn set_bytes(&mut self, bytes: u64) {
        self.bytes = bytes;
    }
}

impl Drop for SpanGuard {
    fn drop(&mut self) {
        add(
            self.name,
            self.started.elapsed().as_nanos().min(u64::MAX as u128) as u64,
            self.bytes,
        );
    }
}

pub fn span(name: &'static str) -> SpanGuard {
    SpanGuard {
        name,
        started: Instant::now(),
        bytes: 0,
    }
}

/// Elapsed ns since `started`, saturating.
pub fn ns_since(started: Instant) -> u64 {
    started.elapsed().as_nanos().min(u64::MAX as u128) as u64
}

struct Node {
    name: String,
    ns: u64,
    calls: u64,
    bytes: u64,
    overlapped: bool,
    derived: bool,
    rate_basis: Option<&'static str>,
    children: Vec<Bucket>,
}

struct Snapshot {
    total_ns: u64,
    top: Vec<Node>,
    other_ns: u64,
    io_delta: Option<(u64, u64)>,
    notes: Vec<(&'static str, String)>,
}

fn parent_of(name: &str) -> &str {
    name.split_once('.').map_or(name, |(p, _)| p)
}

fn snapshot() -> Snapshot {
    let io1 = process_io();
    let st = lock();
    let total_ns = st.started.map_or(0, ns_since);
    let mut order: Vec<&str> = Vec::new();
    for b in &st.buckets {
        let p = parent_of(b.name);
        if !order.contains(&p) {
            order.push(p);
        }
    }
    let mut top = Vec::with_capacity(order.len());
    for p in order {
        let own = st.buckets.iter().find(|b| b.name == p);
        let children: Vec<Bucket> = st
            .buckets
            .iter()
            .filter(|b| b.name != p && parent_of(b.name) == p)
            .cloned()
            .collect();
        let node = match own {
            Some(b) => Node {
                name: p.to_string(),
                ns: b.ns,
                calls: b.calls,
                bytes: b.bytes,
                overlapped: b.overlapped,
                derived: false,
                rate_basis: b.rate_basis,
                children,
            },
            None => Node {
                name: p.to_string(),
                ns: children
                    .iter()
                    .filter(|c| !c.overlapped)
                    .map(|c| c.ns)
                    .sum(),
                calls: 0,
                bytes: 0,
                overlapped: false,
                derived: true,
                rate_basis: None,
                children,
            },
        };
        top.push(node);
    }
    let sum: u64 = top
        .iter()
        .filter(|n| !n.overlapped)
        .map(|n| n.ns)
        .fold(0u64, |a, b| a.saturating_add(b));
    let io_delta = match (st.io0, io1) {
        (Some(a), Some(b)) => Some((
            b.read_bytes.saturating_sub(a.read_bytes),
            b.major_faults.saturating_sub(a.major_faults),
        )),
        _ => None,
    };
    Snapshot {
        total_ns,
        top,
        other_ns: total_ns.saturating_sub(sum),
        io_delta,
        notes: st.notes.clone(),
    }
}

fn ms(ns: u64) -> f64 {
    ns as f64 / 1e6
}

fn gbps(bytes: u64, ns: u64) -> Option<f64> {
    (bytes > 0 && ns > 0).then(|| bytes as f64 / ns as f64)
}

/// Human tag for a bucket's flags: `(overlapped)`, `(sum)`, `(phase-wall)`,
/// `(thread-wall)`, comma-joined inside one pair of parentheses.
fn tag_of(overlapped: bool, derived: bool, basis: Option<&'static str>) -> String {
    let mut parts: Vec<&str> = Vec::new();
    if overlapped {
        parts.push("overlapped");
    }
    if derived {
        parts.push("sum");
    }
    match basis {
        Some(RATE_PHASE_WALL) => parts.push("phase-wall"),
        Some(RATE_THREAD_WALL) => parts.push("thread-wall"),
        Some(_) => parts.push("rate-basis"),
        None => {}
    }
    if parts.is_empty() {
        String::new()
    } else {
        format!("({})", parts.join(","))
    }
}

fn fmt_bucket(name: &str, ns: u64, calls: u64, bytes: u64, tag: &str) -> String {
    let mut s = format!("{name}={:.1}ms/{calls}", ms(ns));
    if bytes > 0 {
        s.push_str(&format!("/{:.3}GB", bytes as f64 / 1e9));
        if let Some(g) = gbps(bytes, ns) {
            s.push_str(&format!("/{g:.2}GB/s"));
        }
    }
    s.push_str(tag);
    s
}

fn render_line(snap: &Snapshot) -> String {
    let mut out = format!("total={:.1}ms", ms(snap.total_ns));
    for n in &snap.top {
        out.push(' ');
        let tag = tag_of(n.overlapped, n.derived, n.rate_basis);
        out.push_str(&fmt_bucket(&n.name, n.ns, n.calls, n.bytes, &tag));
        if !n.children.is_empty() {
            let kids: Vec<String> = n
                .children
                .iter()
                .map(|c| {
                    let child = c.name.split_once('.').map_or(c.name, |(_, k)| k);
                    let tag = tag_of(c.overlapped, false, c.rate_basis);
                    fmt_bucket(child, c.ns, c.calls, c.bytes, &tag)
                })
                .collect();
            out.push_str(&format!("{{{}}}", kids.join(",")));
        }
    }
    out.push_str(&format!(" other={:.1}ms", ms(snap.other_ns)));
    if let Some((rb, mf)) = snap.io_delta {
        out.push_str(&format!(
            " storage_read={:.3}GB major_faults={mf}",
            rb as f64 / 1e9
        ));
    }
    for (k, v) in &snap.notes {
        out.push_str(&format!(" {k}={v}"));
    }
    out
}

/// One-line human summary, e.g.
/// `total=77123.4ms open_admission=1200.3ms/1 weights=60000.0ms/1{range_read=40000.0ms/4096/11.900GB/0.30GB/s,h2d=15000.0ms/4096/11.900GB/0.79GB/s,storage_read=60000.0ms/1/12.100GB/0.20GB/s(phase-wall)} ple_load_fill=5000.0ms/1/3.000GB/0.60GB/s(overlapped,thread-wall) other=3000.0ms storage_read=12.100GB major_faults=123 kernels=lazy`
/// Each bucket is `name=<ms>/<calls>[/<GB>/<GB/s>]`; a parent synthesised from
/// its children is tagged `(sum)`. A GB/s tagged `(phase-wall)` divides thread
/// storage bytes by the whole phase's wall time (amortized, not a device
/// rate); `(thread-wall)` includes the thread's own throttle pauses.
pub fn summary_line() -> String {
    render_line(&snapshot())
}

fn bucket_json(
    name: &str,
    ns: u64,
    calls: u64,
    bytes: u64,
    overlapped: bool,
    rate_basis: Option<&'static str>,
) -> serde_json::Value {
    let mut o = serde_json::Map::new();
    o.insert("name".into(), name.into());
    o.insert("ms".into(), ms(ns).into());
    o.insert("calls".into(), calls.into());
    if bytes > 0 {
        o.insert("bytes".into(), bytes.into());
        if let Some(g) = gbps(bytes, ns) {
            o.insert("gbps".into(), g.into());
        }
    }
    if let Some(basis) = rate_basis {
        o.insert("rate_basis".into(), basis.into());
    }
    if overlapped {
        o.insert("overlapped".into(), true.into());
    }
    o.into()
}

fn render_json(snap: &Snapshot) -> serde_json::Value {
    let phases: Vec<serde_json::Value> = snap
        .top
        .iter()
        .map(|n| {
            let mut v = bucket_json(&n.name, n.ns, n.calls, n.bytes, n.overlapped, n.rate_basis);
            let o = v.as_object_mut().expect("object");
            if n.derived {
                o.insert("derived".into(), true.into());
            }
            if !n.children.is_empty() {
                let kids: Vec<serde_json::Value> = n
                    .children
                    .iter()
                    .map(|c| {
                        let child = c.name.split_once('.').map_or(c.name, |(_, k)| k);
                        bucket_json(child, c.ns, c.calls, c.bytes, c.overlapped, c.rate_basis)
                    })
                    .collect();
                o.insert("children".into(), kids.into());
            }
            v
        })
        .collect();
    let mut root = serde_json::Map::new();
    root.insert("schema".into(), SCHEMA.into());
    root.insert("total_ms".into(), ms(snap.total_ns).into());
    root.insert("phases".into(), phases.into());
    root.insert("other_ms".into(), ms(snap.other_ns).into());
    if let Some((rb, mf)) = snap.io_delta {
        root.insert(
            "storage".into(),
            serde_json::json!({"read_bytes": rb, "major_faults": mf}),
        );
    }
    let notes: serde_json::Map<String, serde_json::Value> = snap
        .notes
        .iter()
        .map(|(k, v)| ((*k).to_string(), v.clone().into()))
        .collect();
    root.insert("notes".into(), notes.into());
    root.into()
}

/// Machine-readable summary (`hipfire.load_phases.v1`).
pub fn summary_json() -> serde_json::Value {
    render_json(&snapshot())
}

/// End the load: take ONE snapshot, mark the registry inactive (later
/// records, e.g. inference-time uploads, are dropped until the next
/// [`begin_load`]) and return `(summary_line, summary_json)` rendered from it.
pub fn finish_load() -> (String, serde_json::Value) {
    let snap = snapshot();
    lock().active = false;
    (render_line(&snap), render_json(&snap))
}

/// End load `generation` without a receipt (failed, refused or rejected
/// load): mark the registry inactive if that load is still the current one.
/// A newer load (different generation) is never touched.
pub fn abort_load(generation: u64) {
    let mut st = lock();
    if st.generation == generation {
        st.active = false;
    }
}

/// One load's active scope, held for the whole daemon `load` arm: created by
/// [`begin_load_scope`] (which calls [`begin_load`]), it ends the scope on
/// EVERY exit of the arm — success after [`finish_load`] (then a no-op),
/// error ACK, refusal `continue`, panic unwind — so a rejected or failed load
/// never leaves recording on while existing requests continue.
#[must_use = "dropping the scope ends the load's recording immediately"]
pub struct LoadScope {
    generation: u64,
}

/// Start a load ([`begin_load`]) and return the guard that ends it.
pub fn begin_load_scope() -> LoadScope {
    begin_load();
    LoadScope {
        generation: generation(),
    }
}

impl Drop for LoadScope {
    fn drop(&mut self) {
        abort_load(self.generation);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex as TestMutex;

    // The registry is process-global; serialise the tests that reset it.
    static GUARD: TestMutex<()> = TestMutex::new(());

    fn phase<'a>(v: &'a serde_json::Value, name: &str) -> Option<&'a serde_json::Value> {
        v["phases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|p| p["name"] == name)
    }

    #[test]
    fn nesting_sum_and_other() {
        let _g = GUARD.lock().unwrap_or_else(|e| e.into_inner());
        begin_load();
        add("a", 3_000_000, 0);
        add("weights", 10_000_000, 0);
        add("weights.h2d", 4_000_000, 2_000_000);
        add("weights.h2d", 4_000_000, 2_000_000);
        add_overlapped("fill", 50_000_000, 1);
        std::thread::sleep(std::time::Duration::from_millis(30));
        let j = summary_json();
        let total = j["total_ms"].as_f64().unwrap();
        let other = j["other_ms"].as_f64().unwrap();
        // Top-level sum is a + weights = 13 ms; child and overlapped excluded.
        assert!((total - 13.0 - other).abs() < 1e-6, "{total} {other}");
        assert!(other > 10.0);
        let w = phase(&j, "weights").unwrap();
        let kids = w["children"].as_array().unwrap();
        assert_eq!(kids.len(), 1);
        assert_eq!(kids[0]["name"], "h2d");
        assert_eq!(kids[0]["calls"], 2);
        assert_eq!(kids[0]["bytes"], 4_000_000);
        assert!(kids[0]["gbps"].as_f64().unwrap() > 0.0);
        assert!(phase(&j, "h2d").is_none(), "child must not be top-level");
        assert_eq!(phase(&j, "fill").unwrap()["overlapped"], true);
        let line = summary_line();
        assert!(line.contains("weights="), "{line}");
        assert!(line.contains("{h2d="), "{line}");
        assert!(
            line.contains("fill=") && line.contains("(overlapped)"),
            "{line}"
        );
    }

    #[test]
    fn derived_parent_and_other_clamped() {
        let _g = GUARD.lock().unwrap_or_else(|e| e.into_inner());
        begin_load();
        add("kernels.route_begin", 5_000_000, 0);
        add("kernels.route_wait", 7_000_000, 0);
        add("huge", 3_600_000_000_000, 0);
        let j = summary_json();
        let k = phase(&j, "kernels").unwrap();
        assert_eq!(k["derived"], true);
        assert!((k["ms"].as_f64().unwrap() - 12.0).abs() < 1e-9);
        assert_eq!(j["other_ms"].as_f64().unwrap(), 0.0);
        assert!(summary_line().contains("kernels=12.0ms/0(sum){route_begin="));
    }

    #[test]
    fn absent_versus_zero_and_ordering() {
        let _g = GUARD.lock().unwrap_or_else(|e| e.into_inner());
        begin_load();
        add("zeroed", 0, 0);
        add("first", 1, 0);
        add("zeroed", 0, 0);
        let j = summary_json();
        assert!(phase(&j, "never_recorded").is_none());
        let z = phase(&j, "zeroed").unwrap();
        assert_eq!(z["ms"].as_f64().unwrap(), 0.0);
        assert_eq!(z["calls"], 2);
        assert!(z.get("bytes").is_none(), "no bytes key when none recorded");
        let names: Vec<_> = j["phases"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| p["name"].as_str().unwrap().to_string())
            .collect();
        assert_eq!(names, ["zeroed", "first"]);
        begin_load();
        assert!(summary_json()["phases"].as_array().unwrap().is_empty());
    }

    #[test]
    fn json_keys_notes_and_span() {
        let _g = GUARD.lock().unwrap_or_else(|e| e.into_inner());
        begin_load();
        {
            let mut s = span("spanned");
            s.set_bytes(7);
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        note("kernels", "lazy");
        note("kernels", "route_plan");
        let j = summary_json();
        assert_eq!(j["schema"], SCHEMA);
        for key in ["total_ms", "phases", "other_ms", "notes"] {
            assert!(j.get(key).is_some(), "missing {key}");
        }
        assert_eq!(j["notes"]["kernels"], "route_plan");
        let p = phase(&j, "spanned").unwrap();
        assert_eq!(p["calls"], 1);
        assert_eq!(p["bytes"], 7);
        assert!(p["ms"].as_f64().unwrap() >= 2.0);
        assert!(summary_line().contains(" kernels=route_plan"));
    }

    #[test]
    fn inactive_registry_drops_records_and_notes() {
        let _g = GUARD.lock().unwrap_or_else(|e| e.into_inner());
        begin_load();
        add("kept", 1, 0);
        let (line, json) = finish_load();
        assert!(line.contains("kept="));
        assert!(phase(&json, "kept").is_some());
        // Between the ACK and the next begin_load (inference-time uploads).
        add("weights.h2d", 5, 5);
        add_amortized("weights.storage_read", 5, 5);
        add_overlapped("fill", 5, 5);
        note("kernels", "lazy");
        let j = summary_json();
        assert!(phase(&j, "weights").is_none());
        assert!(phase(&j, "fill").is_none());
        assert!(j["notes"].as_object().unwrap().is_empty());
        assert!(phase(&j, "kept").is_some(), "finish_load keeps the receipt");
    }

    #[test]
    fn late_old_generation_record_is_dropped() {
        let _g = GUARD.lock().unwrap_or_else(|e| e.into_inner());
        begin_load();
        let old = generation();
        begin_load();
        assert_ne!(old, generation());
        add_overlapped_for(old, "ple_load_fill", 7, 7);
        assert!(phase(&summary_json(), "ple_load_fill").is_none());
        // The old generation stays dropped after the new load finished too.
        let _ = finish_load();
        add_overlapped_for(old, "ple_load_fill", 7, 7);
        assert!(phase(&summary_json(), "ple_load_fill").is_none());
    }

    #[test]
    fn current_generation_overlapped_record_is_kept_and_flagged() {
        let _g = GUARD.lock().unwrap_or_else(|e| e.into_inner());
        begin_load();
        let current = generation();
        add_overlapped_for(current, "ple_load_fill", 2_000_000_000, 1_000_000_000);
        add_amortized("weights.storage_read", 1_000_000_000, 500_000_000);
        let j = summary_json();
        let fill = phase(&j, "ple_load_fill").unwrap();
        assert_eq!(fill["overlapped"], true);
        assert_eq!(fill["rate_basis"], RATE_THREAD_WALL);
        assert_eq!(fill["bytes"], 1_000_000_000u64);
        let w = phase(&j, "weights").unwrap();
        assert_eq!(w["children"][0]["name"], "storage_read");
        assert_eq!(w["children"][0]["rate_basis"], "phase_wall");
        let line = summary_line();
        assert!(
            line.contains("ple_load_fill=2000.0ms/1/1.000GB/0.50GB/s(overlapped,thread-wall)"),
            "{line}"
        );
        assert!(
            line.contains("storage_read=1000.0ms/1/0.500GB/0.50GB/s(phase-wall)"),
            "{line}"
        );
        // Overlapped records stay out of the sum.
        assert!(j["other_ms"].as_f64().unwrap() >= 0.0);
    }

    #[test]
    fn begin_load_resets_buckets_notes_and_reactivates() {
        let _g = GUARD.lock().unwrap_or_else(|e| e.into_inner());
        begin_load();
        add("a", 1, 0);
        note("k", "v");
        let _ = finish_load();
        begin_load();
        let j = summary_json();
        assert!(j["phases"].as_array().unwrap().is_empty());
        assert!(j["notes"].as_object().unwrap().is_empty());
        add("b", 1, 0);
        assert!(phase(&summary_json(), "b").is_some(), "active again");
    }

    #[test]
    fn dropped_scope_without_finish_stops_recording() {
        let _g = GUARD.lock().unwrap_or_else(|e| e.into_inner());
        {
            let _scope = begin_load_scope();
            add("weights", 5, 0);
            // error ACK / refusal `continue`: the arm exits without finish_load
        }
        add("late", 1, 0);
        note("late_note", "x");
        let j = summary_json();
        assert!(
            phase(&j, "late").is_none(),
            "recorded after an aborted load"
        );
        assert!(j["notes"].get("late_note").is_none());
    }

    #[test]
    fn scope_drop_after_finish_is_a_noop_and_never_touches_a_newer_load() {
        let _g = GUARD.lock().unwrap_or_else(|e| e.into_inner());
        let old = begin_load_scope();
        add("a", 1, 0);
        let (line, json) = finish_load();
        assert!(line.contains("a=") && phase(&json, "a").is_some());
        drop(old);
        let newer = begin_load_scope();
        // a stale guard of the previous generation ending late
        abort_load(newer.generation.wrapping_sub(1));
        add("b", 1, 0);
        assert!(
            phase(&summary_json(), "b").is_some(),
            "newer load still active"
        );
        drop(newer);
        add("c", 1, 0);
        assert!(phase(&summary_json(), "c").is_none());
    }

    #[test]
    fn proc_readers_do_not_panic() {
        // Linux CI has procfs; elsewhere the readers must report None.
        if let Some((rb, _mf)) = thread_io() {
            let _ = rb;
        }
        let _ = process_io();
    }
}
