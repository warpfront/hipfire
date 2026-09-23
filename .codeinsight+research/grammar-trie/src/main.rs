//! CPU parity + performance harness for `adapter::AfterOpenTrie`.
//! Parent runs: cargo run --release --manifest-path …/Cargo.toml -- <model.mq4-xt>

use hipfire_grammar_trie_experiment::adapter::AfterOpenTrie;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::tokenizer::Tokenizer;
use saddle_core::grammar::json::{Matcher, State, ToolSchema};
use serde_json::{json, Value};
use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

// ── counting allocator (only windows with COUNTING=true) ───────────────────

struct CountingAlloc;

static COUNTING: AtomicBool = AtomicBool::new(false);
static ALLOC_COUNT: AtomicU64 = AtomicU64::new(0);
static ALLOC_BYTES: AtomicU64 = AtomicU64::new(0);

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
            ALLOC_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        }
        System.alloc(layout)
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
            ALLOC_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        }
        System.alloc_zeroed(layout)
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
            let grow = new_size.saturating_sub(layout.size()) as u64;
            ALLOC_BYTES.fetch_add(grow.max(new_size as u64), Ordering::Relaxed);
        }
        System.realloc(ptr, layout, new_size)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
}

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

fn alloc_reset() {
    ALLOC_COUNT.store(0, Ordering::Relaxed);
    ALLOC_BYTES.store(0, Ordering::Relaxed);
}

fn alloc_snapshot() -> (u64, u64) {
    (
        ALLOC_COUNT.load(Ordering::Relaxed),
        ALLOC_BYTES.load(Ordering::Relaxed),
    )
}

fn with_counting<T>(f: impl FnOnce() -> T) -> (T, u64, u64) {
    alloc_reset();
    COUNTING.store(true, Ordering::SeqCst);
    let out = f();
    COUNTING.store(false, Ordering::SeqCst);
    let (c, b) = alloc_snapshot();
    (out, c, b)
}

/// Wall-clock only — COUNTING stays false so alloc instrumentation cannot skew ns.
fn timed_ns_per(repeats: u32, mut f: impl FnMut()) -> f64 {
    COUNTING.store(false, Ordering::SeqCst);
    let t0 = Instant::now();
    for _ in 0..repeats {
        f();
    }
    t0.elapsed().as_secs_f64() * 1e9 / repeats as f64
}

/// Separate allocation window (not combined with wall-clock timing).
fn count_allocs_per(repeats: u32, mut f: impl FnMut()) -> (f64, f64) {
    let (_, ac, ab) = with_counting(|| {
        for _ in 0..repeats {
            f();
        }
    });
    (ac as f64 / repeats as f64, ab as f64 / repeats as f64)
}

// ── fixtures ───────────────────────────────────────────────────────────────

fn tools_basic() -> Vec<ToolSchema> {
    vec![
        ToolSchema {
            name: "bash".into(),
            required: vec!["command".into()],
        },
        ToolSchema {
            name: "read".into(),
            required: vec!["path".into()],
        },
        ToolSchema {
            name: "write".into(),
            required: vec!["path".into(), "content".into()],
        },
        // Unicode tool name — multi-byte UTF-8 boundaries matter.
        ToolSchema {
            name: "读文件".into(),
            required: vec!["路径".into()],
        },
        ToolSchema {
            name: "café_tool".into(),
            required: vec!["arg".into()],
        },
    ]
}

fn tools_alt() -> Vec<ToolSchema> {
    // Different schema set for reuse / tool-change exactness.
    vec![
        ToolSchema {
            name: "grep".into(),
            required: vec!["pattern".into()],
        },
        ToolSchema {
            name: "bash".into(),
            required: Vec::new(),
        },
    ]
}

fn header_for(name: &str) -> String {
    format!("\n{{\"name\": \"{name}\", \"arguments\": ")
}

struct Case {
    name: String,
    kind: &'static str, // free | AfterOpen | InArgs
    matcher: Matcher,
}

fn expect_state(m: &Matcher, want: &State, label: &str) {
    let got = m.state();
    if got != want {
        eprintln!(
            "FATAL fixture state: {label}: expected {want:?}, got {got:?}, partial={:?}",
            m.partial()
        );
        std::process::exit(2);
    }
}

fn build_cases() -> Vec<Case> {
    let mut out = Vec::new();
    let tools = tools_basic();

    // ── free / Out (partial markers stay free) ──
    {
        let m = Matcher::new(tools.clone());
        expect_state(&m, &State::Out, "free_out");
        assert!(m.is_free());
        out.push(Case {
            name: "free_out".into(),
            kind: "free",
            matcher: m,
        });
    }
    for (label, prefix) in [
        ("free_partial_lt", "<"),
        ("free_partial_tool", "<tool"),
        ("free_partial_tool_call", "<tool_call"),
        ("free_prose_then_lt", "hello world <"),
    ] {
        let mut m = Matcher::new(tools.clone());
        m.advance(prefix);
        expect_state(&m, &State::Out, label);
        assert!(m.is_free(), "{label} must remain free");
        out.push(Case {
            name: label.into(),
            kind: "free",
            matcher: m,
        });
    }

    // ── AfterOpen empty partial ──
    {
        let mut m = Matcher::new(tools.clone());
        m.advance("<tool_call>");
        expect_state(&m, &State::AfterOpen, "after_open_empty");
        assert!(!m.is_free());
        assert!(m.partial().is_empty());
        out.push(Case {
            name: "after_open_empty".into(),
            kind: "AfterOpen",
            matcher: m,
        });
    }

    // AfterOpen at every UTF-8 boundary of each header prefix.
    for schema in &tools {
        let header = header_for(&schema.name);
        let bytes = header.as_bytes();
        let mut i = 0;
        while i <= bytes.len() {
            if i < bytes.len() && (bytes[i] & 0xC0) == 0x80 {
                // skip continuation bytes — only char boundaries
                i += 1;
                continue;
            }
            let prefix = std::str::from_utf8(&bytes[..i]).expect("utf8 boundary");
            let mut m = Matcher::new(tools.clone());
            m.advance("<tool_call>");
            if !prefix.is_empty() {
                m.advance(prefix);
            }
            // Full header completes → InArgs; only keep AfterOpen slices.
            if matches!(m.state(), State::AfterOpen) {
                let name = format!("after_open_{}_b{i}", schema.name);
                out.push(Case {
                    name,
                    kind: "AfterOpen",
                    matcher: m,
                });
            } else if i == bytes.len() {
                // exact full header → InArgs with empty leftover
                expect_state(&m, &State::InArgs, &format!("header_complete_{}", schema.name));
            }
            i += 1;
        }
    }

    // Merged header+args token: advance a chunk that finishes the header and
    // spills into the args body in one go.
    {
        let mut m = Matcher::new(tools.clone());
        m.advance("<tool_call>");
        let merged = format!(
            "{}\n{{\"command\": \"ls\"}}",
            header_for("bash")
        );
        // Advance in one shot so transition + leftover args land together.
        m.advance(&merged);
        expect_state(&m, &State::InArgs, "merged_header_args");
        out.push(Case {
            name: "merged_header_args".into(),
            kind: "InArgs",
            matcher: m,
        });
    }

    // ── InArgs gates ──

    // required missing, empty body
    {
        let mut m = Matcher::new(tools.clone());
        m.advance("<tool_call>");
        m.advance(&header_for("bash"));
        expect_state(&m, &State::InArgs, "inargs_required_missing");
        assert!(!m.is_free(), "required missing must constrain");
        out.push(Case {
            name: "inargs_required_missing".into(),
            kind: "InArgs",
            matcher: m,
        });
    }

    // nested object, required still missing deeper
    {
        let mut m = Matcher::new(tools.clone());
        m.advance("<tool_call>");
        m.advance(&header_for("write"));
        m.advance("{\"meta\": {\"x\": 1");
        expect_state(&m, &State::InArgs, "inargs_nested");
        out.push(Case {
            name: "inargs_nested".into(),
            kind: "InArgs",
            matcher: m,
        });
    }

    // escaped quote inside string value
    {
        let mut m = Matcher::new(tools.clone());
        m.advance("<tool_call>");
        m.advance(&header_for("bash"));
        m.advance("{\"command\": \"echo \\\"hi\\\"");
        expect_state(&m, &State::InArgs, "inargs_escaped");
        out.push(Case {
            name: "inargs_escaped".into(),
            kind: "InArgs",
            matcher: m,
        });
    }

    // required present, inner brace closed, outer still open (close gated)
    {
        let mut m = Matcher::new(tools.clone());
        m.advance("<tool_call>");
        m.advance(&header_for("read"));
        m.advance("{\"path\": \"/tmp/x\"}");
        expect_state(&m, &State::InArgs, "inargs_close_outer_open");
        assert!(
            !m.is_token_allowed("</tool_call>"),
            "close must be blocked while outer brace open"
        );
        out.push(Case {
            name: "inargs_close_outer_open".into(),
            kind: "InArgs",
            matcher: m,
        });
    }

    // outer closed → close marker allowed (still InArgs until marker advances)
    {
        let mut m = Matcher::new(tools.clone());
        m.advance("<tool_call>");
        m.advance(&header_for("read"));
        m.advance("{\"path\": \"/tmp/x\"}}");
        expect_state(&m, &State::InArgs, "inargs_close_ready");
        assert!(
            m.is_token_allowed("</tool_call>"),
            "close must be allowed once outer balanced"
        );
        out.push(Case {
            name: "inargs_close_ready".into(),
            kind: "InArgs",
            matcher: m,
        });
    }

    // long args body (for weighted workload realism)
    {
        let mut m = Matcher::new(tools.clone());
        m.advance("<tool_call>");
        m.advance(&header_for("write"));
        let mut body = String::from("{\"path\": \"/tmp/big.rs\", \"content\": \"");
        for _ in 0..200 {
            body.push_str("fn demo() { let x = 1 + 2; }\\n");
        }
        // leave string open — deep InArgs
        m.advance(&body);
        expect_state(&m, &State::InArgs, "inargs_long_body");
        out.push(Case {
            name: "inargs_long_body".into(),
            kind: "InArgs",
            matcher: m,
        });
    }

    // close-prefix forming
    {
        let mut m = Matcher::new(tools.clone());
        m.advance("<tool_call>");
        m.advance(&header_for("bash"));
        m.advance("{\"command\": \"ls\"}}");
        m.advance("\n</tool");
        expect_state(&m, &State::InArgs, "inargs_close_prefix");
        out.push(Case {
            name: "inargs_close_prefix".into(),
            kind: "InArgs",
            matcher: m,
        });
    }

    out
}

// ── parity ─────────────────────────────────────────────────────────────────

fn mask_density(mask: &[bool]) -> f64 {
    if mask.is_empty() {
        return 0.0;
    }
    let ones = mask.iter().filter(|&&b| b).count();
    ones as f64 / mask.len() as f64
}

fn compare_masks(
    label: &str,
    matcher: &Matcher,
    vocab: &[String],
    oracle: &[bool],
    adapter: &[bool],
) -> Result<(), String> {
    let n = vocab.len();
    if oracle.len() < n || adapter.len() < n {
        return Err(format!("{label}: mask len short oracle={} adapter={} n={n}", oracle.len(), adapter.len()));
    }
    for id in 0..n {
        if oracle[id] != adapter[id] {
            let text = &vocab[id];
            let preview: String = text.chars().take(80).collect();
            return Err(format!(
                "mismatch state={label} id={id} oracle={} adapter={} text={preview:?} \
                 matcher_state={:?} partial={:?} is_free={}",
                oracle[id],
                adapter[id],
                matcher.state(),
                matcher.partial(),
                matcher.is_free()
            ));
        }
    }
    Ok(())
}

fn run_parity_on_vocab(
    label_prefix: &str,
    cases: &[Case],
    vocab: &[String],
    trie: &mut AfterOpenTrie,
) -> Result<(u64, Vec<Value>), String> {
    let n = vocab.len();
    let mut oracle = vec![false; n];
    let mut adapted = vec![false; n];
    let mut compared: u64 = 0;
    let mut per = Vec::new();

    for case in cases {
        // Oracle first — independent; matcher must not advance during fill.
        let partial_before = case.matcher.partial().to_string();
        let state_before = case.matcher.state().clone();
        case.matcher.token_mask(vocab, &mut oracle);
        trie.fill(&case.matcher, vocab, &mut adapted);
        // Matcher integrity: fill must not mutate oracle matcher.
        if case.matcher.partial() != partial_before || case.matcher.state() != &state_before {
            return Err(format!(
                "{label_prefix}/{}: matcher mutated during fill",
                case.name
            ));
        }
        compare_masks(
            &format!("{label_prefix}/{}", case.name),
            &case.matcher,
            vocab,
            &oracle,
            &adapted,
        )?;
        compared += n as u64;
        per.push(json!({
            "name": case.name,
            "kind": case.kind,
            "state": format!("{:?}", case.matcher.state()),
            "partial_len": case.matcher.partial().len(),
            "is_free": case.matcher.is_free(),
            "density": mask_density(&oracle[..n]),
            "vocab": n,
        }));
    }
    Ok((compared, per))
}

/// Semantic guards that are not mere wrapper echoes.
fn semantic_guards(vocab: &[String], trie: &mut AfterOpenTrie) -> Result<Value, String> {
    let tools = tools_basic();
    let n = vocab.len();
    let mut oracle = vec![false; n];
    let mut adapted = vec![false; n];
    let mut notes = Vec::new();

    // 1) Empty ids always allowed under AfterOpen (oracle invariant).
    {
        let mut m = Matcher::new(tools.clone());
        m.advance("<tool_call>");
        expect_state(&m, &State::AfterOpen, "guard_empty");
        m.token_mask(vocab, &mut oracle);
        trie.fill(&m, vocab, &mut adapted);
        compare_masks("guard_empty_ids", &m, vocab, &oracle, &adapted)?;
        let mut empty_ids = 0u64;
        for (id, t) in vocab.iter().enumerate() {
            if t.is_empty() {
                empty_ids += 1;
                if !oracle[id] {
                    return Err(format!("empty id {id} oracle-disallowed under AfterOpen"));
                }
                if !adapted[id] {
                    return Err(format!("empty id {id} adapter-disallowed under AfterOpen"));
                }
            }
        }
        notes.push(json!({"check": "empty_ids_allowed", "count": empty_ids, "ok": true}));
    }

    // 2) Duplicate decoded strings share allow bits (string oracle).
    {
        let mut m = Matcher::new(tools.clone());
        m.advance("<tool_call>");
        m.advance("\n");
        m.token_mask(vocab, &mut oracle);
        trie.fill(&m, vocab, &mut adapted);
        compare_masks("guard_dup_bits", &m, vocab, &oracle, &adapted)?;
        // Group by text; all ids with same text must agree.
        use std::collections::HashMap;
        let mut groups: HashMap<&str, Vec<usize>> = HashMap::new();
        for (id, t) in vocab.iter().enumerate() {
            groups.entry(t.as_str()).or_default().push(id);
        }
        let mut dup_groups = 0u64;
        for (text, ids) in &groups {
            if ids.len() < 2 {
                continue;
            }
            dup_groups += 1;
            let bit = oracle[ids[0]];
            for &id in ids {
                if oracle[id] != bit || adapted[id] != bit {
                    return Err(format!(
                        "duplicate text {text:?} disagree id={} oracle={} adapter={} expect={bit}",
                        id, oracle[id], adapted[id]
                    ));
                }
            }
        }
        notes.push(json!({"check": "duplicate_ids_agree", "dup_groups": dup_groups, "ok": true}));
    }

    // 3) Absorbing header completion: a token that finishes the header and
    //    continues into args must be allowed at the corresponding partial.
    {
        let header = header_for("bash");
        // partial = all but last 3 chars of header; token = last 3 + args spill
        let split = header.len().saturating_sub(3);
        let partial = &header[..split];
        let spill = format!("{}{{\"command\":\"x\"}}", &header[split..]);
        let mut m = Matcher::new(tools.clone());
        m.advance("<tool_call>");
        m.advance(partial);
        expect_state(&m, &State::AfterOpen, "guard_absorb");
        if !m.is_token_allowed(&spill) {
            return Err(format!(
                "absorbing completion: oracle rejects spill partial={partial:?} spill={spill:?}"
            ));
        }
        // Inject spill as synthetic last vocab entry for mask compare path.
        let mut v = vocab.to_vec();
        v.push(spill.clone());
        // Rebuild trie over extended vocab for this one check.
        let mut t2 = AfterOpenTrie::new(&v);
        let mut o = vec![false; v.len()];
        let mut a = vec![false; v.len()];
        m.token_mask(&v, &mut o);
        t2.fill(&m, &v, &mut a);
        let id = v.len() - 1;
        if !o[id] || !a[id] {
            return Err(format!(
                "absorbing completion mask: oracle={} adapter={} id={id}",
                o[id], a[id]
            ));
        }
        compare_masks("guard_absorb_full", &m, &v, &o, &a)?;
        notes.push(json!({"check": "absorbing_header_completion", "ok": true}));
    }

    // 4) Tool schema change on reused adapter must not leave stale headers.
    {
        let mut m1 = Matcher::new(tools_basic());
        m1.advance("<tool_call>");
        trie.fill(&m1, vocab, &mut adapted);
        let dens1 = mask_density(&adapted[..n]);

        let mut m2 = Matcher::new(tools_alt());
        m2.advance("<tool_call>");
        m2.token_mask(vocab, &mut oracle);
        trie.fill(&m2, vocab, &mut adapted);
        compare_masks("guard_tool_change", &m2, vocab, &oracle, &adapted)?;
        // alt tools: "grep" header must be allowed; "读文件" must not (no stale headers).
        let grep_ok = m2.is_token_allowed("\n{\"name\": \"grep\"");
        let uni_ok = m2.is_token_allowed("\n{\"name\": \"读文件\"");
        if !grep_ok {
            return Err("tool change: grep header prefix rejected".into());
        }
        if uni_ok {
            return Err("tool change: stale 读文件 header still allowed after schema swap".into());
        }
        notes.push(json!({
            "check": "tool_schema_change_no_stale",
            "density_basic_after_open": dens1,
            "density_alt_after_open": mask_density(&adapted[..n]),
            "ok": true
        }));
    }

    // 5) Repeated fill on same matcher is bit-identical (reuse scratch).
    {
        let mut m = Matcher::new(tools_basic());
        m.advance("<tool_call>");
        m.advance("\n{\"name\": \"");
        let mut a1 = vec![false; n];
        let mut a2 = vec![false; n];
        trie.fill(&m, vocab, &mut a1);
        trie.fill(&m, vocab, &mut a2);
        if a1 != a2 {
            return Err("repeated fill on same state produced different masks".into());
        }
        m.token_mask(vocab, &mut oracle);
        compare_masks("guard_repeated_fill", &m, vocab, &oracle, &a1)?;
        notes.push(json!({"check": "repeated_fill_identical", "ok": true}));
    }

    Ok(json!(notes))
}


fn synthetic_vocab_parity(trie_factory: impl Fn(&[String]) -> AfterOpenTrie) -> Result<Value, String> {
    // empty / duplicate / over-300-byte tokens.
    // Critical: one ACCEPTED long token = full header + 512-byte body suffix so the
    // recognizer must traverse >300 accepted bytes (not reject at first byte).
    let header = header_for("bash");
    let mut long_accepted = header.clone();
    long_accepted.push_str(&"B".repeat(512));
    if long_accepted.len() <= 300 {
        return Err(format!(
            "syn long_accepted len {} not >300",
            long_accepted.len()
        ));
    }

    let mut vocab = vec![
        "".to_string(),
        "".to_string(), // duplicate empty
        "\n".to_string(),
        "\n".to_string(), // duplicate newline
        "{\"name\"".to_string(),
        "hello".to_string(),
        "hello".to_string(), // duplicate
        "x".repeat(301),     // over 300, rejected non-prefix
        "y".repeat(512),     // rejected non-prefix
        header.clone(),
        long_accepted.clone(), // ACCEPTED: header complete + absorbing body
        format!("{}EXTRA", header), // completes header + short spill
        "<|im_start|>".to_string(),
        "读".to_string(),
    ];
    for i in 0..32 {
        vocab.push(format!("tok{i}"));
    }

    let mut trie = trie_factory(&vocab);
    let tools = tools_basic();
    let mut cases = Vec::new();

    {
        let mut m = Matcher::new(tools.clone());
        m.advance("<tool_call>");
        cases.push(Case {
            name: "syn_after_open".into(),
            kind: "AfterOpen",
            matcher: m,
        });
    }
    {
        let mut m = Matcher::new(tools.clone());
        m.advance("<tool_call>");
        m.advance("\n{\"name\": \"");
        cases.push(Case {
            name: "syn_after_open_mid".into(),
            kind: "AfterOpen",
            matcher: m,
        });
    }
    {
        let mut m = Matcher::new(tools.clone());
        m.advance("<tool_call>");
        m.advance(&header_for("bash"));
        cases.push(Case {
            name: "syn_inargs".into(),
            kind: "InArgs",
            matcher: m,
        });
    }
    {
        let m = Matcher::new(tools.clone());
        cases.push(Case {
            name: "syn_free".into(),
            kind: "free",
            matcher: m,
        });
    }

    let (bits, per) = run_parity_on_vocab("synthetic", &cases, &vocab, &mut trie)?;

    // Empty always allowed; reject non-prefix long; ACCEPT header+512 body under AfterOpen.
    let mut m = Matcher::new(tools);
    m.advance("<tool_call>");
    if !m.is_token_allowed("") {
        return Err("syn empty token disallowed under AfterOpen".into());
    }
    if m.is_token_allowed(&"x".repeat(301)) {
        return Err("syn 301-byte non-prefix token unexpectedly allowed".into());
    }
    if !m.is_token_allowed(&long_accepted) {
        return Err(format!(
            "syn long ACCEPTED token (header+512 body, {} bytes) rejected by oracle under AfterOpen",
            long_accepted.len()
        ));
    }
    // Mask path must also allow the long accepted id (forces >300-byte trie walk).
    let long_id = vocab
        .iter()
        .position(|t| t == &long_accepted)
        .expect("long_accepted in vocab");
    let mut o = vec![false; vocab.len()];
    let mut a = vec![false; vocab.len()];
    m.token_mask(&vocab, &mut o);
    trie.fill(&m, &vocab, &mut a);
    if !o[long_id] || !a[long_id] {
        return Err(format!(
            "syn long_accepted id={long_id} mask oracle={} adapter={} len={}",
            o[long_id],
            a[long_id],
            long_accepted.len()
        ));
    }
    compare_masks("syn_long_accepted_mask", &m, &vocab, &o, &a)?;

    for (id, t) in vocab.iter().enumerate() {
        if t.is_empty() && !m.is_token_allowed(t) {
            return Err(format!("syn empty id {id} disallowed"));
        }
    }

    Ok(json!({
        "compared_bits": bits,
        "cases": per,
        "vocab_len": vocab.len(),
        "max_token_bytes": trie.max_token_bytes(),
        "long_accepted_bytes": long_accepted.len(),
        "long_accepted_id": long_id,
    }))
}

// ── bench ──────────────────────────────────────────────────────────────────

struct BenchRow {
    name: String,
    kind: String,
    oracle_ns: f64,
    adapter_ns: f64,
    oracle_allocs: f64,
    adapter_allocs: f64,
    oracle_alloc_bytes: f64,
    adapter_alloc_bytes: f64,
    density: f64,
    repeats: u32,
}

fn bench_case(
    case: &Case,
    vocab: &[String],
    trie: &mut AfterOpenTrie,
    warmups: u32,
    repeats: u32,
) -> BenchRow {
    let n = vocab.len();
    let mut oracle = vec![false; n];
    let mut adapted = vec![false; n];

    COUNTING.store(false, Ordering::SeqCst);
    for _ in 0..warmups {
        case.matcher.token_mask(vocab, &mut oracle);
        trie.fill(&case.matcher, vocab, &mut adapted);
        black_box(&oracle);
        black_box(&adapted);
    }

    // Wall-clock with COUNTING=false.
    let oracle_ns = timed_ns_per(repeats, || {
        case.matcher.token_mask(vocab, black_box(&mut oracle));
        black_box(&oracle);
    });
    let adapter_ns = timed_ns_per(repeats, || {
        trie.fill(&case.matcher, vocab, black_box(&mut adapted));
        black_box(&adapted);
    });

    // Separate allocation windows (not timed together).
    let (o_allocs, o_bytes) = count_allocs_per(repeats, || {
        case.matcher.token_mask(vocab, black_box(&mut oracle));
        black_box(&oracle);
    });
    let (a_allocs, a_bytes) = count_allocs_per(repeats, || {
        trie.fill(&case.matcher, vocab, black_box(&mut adapted));
        black_box(&adapted);
    });

    case.matcher.token_mask(vocab, &mut oracle);

    BenchRow {
        name: case.name.clone(),
        kind: case.kind.to_string(),
        oracle_ns,
        adapter_ns,
        oracle_allocs: o_allocs,
        adapter_allocs: a_allocs,
        oracle_alloc_bytes: o_bytes,
        adapter_alloc_bytes: a_bytes,
        density: mask_density(&oracle),
        repeats,
    }
}

/// Realistic tool-call text: short header + modest args, grammar-safe close.
///
/// **Observed old Matcher limitation (do not alter grammar):** `ngram_history`
/// is a 256-byte rolling window used by `required_fields_satisfied`. A long
/// args body can evict the `"command"` key from the window → `req_satisfied=
/// false` → brace-close tokens rejected. Keep total args body under ~200
/// bytes so the required key remains visible. Closure is whitespace-separated
/// (`" } }`) so encode does not emit a merged `"}}` token.
fn realistic_tool_call_text() -> String {
    // 4 lines × ~20 chars → well under the 256-byte required-key window.
    let cmd = "echo a1b2c3d4e5f6g7\necho h8i9j0k1l2m3n4\necho o5p6q7r8s9t0u1\necho v2w3x4y5z6a7b8";
    debug_assert!(cmd.len() < 120, "cmd payload must stay short for ngram window");
    let cmd_esc = cmd
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n");
    // After the string value: close-quote, space, args-`}, space, outer-`}`.
    format!(
        "<tool_call>\n{{\"name\": \"bash\", \"arguments\": {{\"command\": \"{cmd_esc}\" }} }}\n</tool_call>"
    )
}

struct TraceBench {
    snapshots: Vec<Case>,
    token_count: usize,
    nonfree_count: usize,
    kind_counts: std::collections::BTreeMap<&'static str, u64>,
    tool_seen: Option<usize>,
    final_state: String,
    text_bytes: usize,
}

/// Encode a real tool-call, advance token-by-token via decode(&[id]), snapshot
/// every non-free pre-token matcher state. Asserts Out + bash tool completion.
fn build_encode_trace(tok: &Tokenizer, tools: Vec<ToolSchema>) -> Result<TraceBench, String> {
    let text = realistic_tool_call_text();
    let ids = tok.encode(&text);
    if ids.is_empty() {
        return Err("encode produced empty id list".into());
    }

    let mut m = Matcher::new(tools);
    let mut snapshots = Vec::new();
    let mut kind_counts = std::collections::BTreeMap::<&'static str, u64>::new();
    let mut tool_seen: Option<usize> = None;
    let mut nonfree = 0usize;

    for (step, &id) in ids.iter().enumerate() {
        let piece = tok.decode(&[id]);
        if !m.is_free() {
            nonfree += 1;
            let kind: &'static str = match m.state() {
                State::AfterOpen => "AfterOpen",
                State::InArgs => "InArgs",
                State::Out => "free",
            };
            *kind_counts.entry(kind).or_default() += 1;
            if let Some(t) = m.current_tool() {
                tool_seen = Some(t);
            }
            // Sample-time invariant: committed piece must be allowed.
            // Do not weaken the predicate — if rejected, the fixture text is wrong.
            if !m.is_token_allowed(&piece) {
                return Err(format!(
                    "trace step={step} id={id} piece={:?} rejected in {:?} partial={:?} \
                     is_free={} current_tool={:?} attractor={} close_dbg={}",
                    piece.chars().take(60).collect::<String>(),
                    m.state(),
                    m.partial(),
                    m.is_free(),
                    m.current_tool(),
                    m.attractor_detected(),
                    m.debug_close_reject()
                ));
            }
            snapshots.push(Case {
                name: format!("trace_{step}_{kind}"),
                kind,
                matcher: m.clone(),
            });
        }
        m.advance(&piece);
        if let Some(t) = m.current_tool() {
            tool_seen = Some(t);
        }
    }

    if !matches!(m.state(), State::Out) {
        return Err(format!(
            "trace did not finish in Out; state={:?} partial={:?}",
            m.state(),
            m.partial()
        ));
    }
    // bash is index 0 in tools_basic.
    match tool_seen {
        Some(0) => {}
        Some(other) => {
            return Err(format!(
                "trace completed wrong tool index={other}, expected 0 (bash)"
            ))
        }
        None => return Err("trace never entered a tool (current_tool never set)".into()),
    }
    if nonfree == 0 {
        return Err("trace produced zero non-free steps".into());
    }

    Ok(TraceBench {
        snapshots,
        token_count: ids.len(),
        nonfree_count: nonfree,
        kind_counts,
        tool_seen,
        final_state: format!("{:?}", m.state()),
        text_bytes: text.len(),
    })
}

/// Measure the encode-trace non-free snapshot distribution (body-heavy, not
/// every header-byte fixture). Timing and allocs are separate windows.
fn measure_trace_workload(
    snapshots: &[Case],
    vocab: &[String],
    trie: &mut AfterOpenTrie,
    warmups: u32,
    repeats: u32,
) -> Result<Value, String> {
    if snapshots.is_empty() {
        return Err("empty trace snapshots".into());
    }
    let n = vocab.len();
    let mut oracle = vec![false; n];
    let mut adapted = vec![false; n];

    // Parity on every snapshot first.
    let mut compared = 0u64;
    for s in snapshots {
        let before_p = s.matcher.partial().to_string();
        let before_st = s.matcher.state().clone();
        s.matcher.token_mask(vocab, &mut oracle);
        trie.fill(&s.matcher, vocab, &mut adapted);
        if s.matcher.partial() != before_p || s.matcher.state() != &before_st {
            return Err(format!("{}: matcher mutated during fill", s.name));
        }
        compare_masks(&s.name, &s.matcher, vocab, &oracle, &adapted)?;
        compared += n as u64;
    }

    COUNTING.store(false, Ordering::SeqCst);
    for _ in 0..warmups {
        for s in snapshots {
            s.matcher.token_mask(vocab, &mut oracle);
            trie.fill(&s.matcher, vocab, &mut adapted);
        }
    }

    let fills_per_rep = snapshots.len() as f64;

    let oracle_ns = {
        let ns = timed_ns_per(repeats, || {
            for s in snapshots {
                s.matcher.token_mask(vocab, black_box(&mut oracle));
                black_box(&oracle);
            }
        });
        ns / fills_per_rep
    };
    let adapter_ns = {
        let ns = timed_ns_per(repeats, || {
            for s in snapshots {
                trie.fill(&s.matcher, vocab, black_box(&mut adapted));
                black_box(&adapted);
            }
        });
        ns / fills_per_rep
    };

    let (o_a, o_b) = {
        let (a, b) = count_allocs_per(repeats, || {
            for s in snapshots {
                s.matcher.token_mask(vocab, black_box(&mut oracle));
                black_box(&oracle);
            }
        });
        (a / fills_per_rep, b / fills_per_rep)
    };
    let (a_a, a_b) = {
        let (a, b) = count_allocs_per(repeats, || {
            for s in snapshots {
                trie.fill(&s.matcher, vocab, black_box(&mut adapted));
                black_box(&adapted);
            }
        });
        (a / fills_per_rep, b / fills_per_rep)
    };

    // Density sample: first / mid / last snapshot.
    let dens = |s: &Case| {
        let mut msk = vec![false; n];
        s.matcher.token_mask(vocab, &mut msk);
        mask_density(&msk)
    };
    let d0 = dens(&snapshots[0]);
    let dmid = dens(&snapshots[snapshots.len() / 2]);
    let dlast = dens(&snapshots[snapshots.len() - 1]);

    Ok(json!({
        "schedule_len": snapshots.len(),
        "repeats": repeats,
        "compared_bits": compared,
        "oracle_ns_per_fill": oracle_ns,
        "adapter_ns_per_fill": adapter_ns,
        "oracle_allocs_per_fill": o_a,
        "adapter_allocs_per_fill": a_a,
        "oracle_alloc_bytes_per_fill": o_b,
        "adapter_alloc_bytes_per_fill": a_b,
        "density_first": d0,
        "density_mid": dmid,
        "density_last": dlast,
        "note": "encode-trace nonfree pre-token snapshots; body-heavy, not header-byte grid",
    }))
}

// ── main ───────────────────────────────────────────────────────────────────

fn main() {
    let mut argv = std::env::args().skip(1);
    let model = argv
        .next()
        .unwrap_or_else(|| "/home/kaden/.hipfire/models/qwen3.8-27b.mq4-xt".into());
    let out_path = argv.next(); // optional JSON sink path
    let path = Path::new(&model);

    let t_load = Instant::now();
    let hfq = match HfqFile::open(path) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("HfqFile::open({model}): {e}");
            std::process::exit(2);
        }
    };
    let tok = match Tokenizer::from_hfq_metadata(&hfq.metadata_json) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Tokenizer::from_hfq_metadata: {e}");
            std::process::exit(2);
        }
    };
    let load_ms = t_load.elapsed().as_secs_f64() * 1e3;

    // Production decoded_vocab construction — EXACTLY decode(&[id]).
    let n = tok.vocab_size();
    let t_vocab = Instant::now();
    let vocab: Vec<String> = (0..n).map(|id| tok.decode(&[id as u32])).collect();
    let vocab_ms = t_vocab.elapsed().as_secs_f64() * 1e3;

    let t_trie = Instant::now();
    let (mut trie, trie_allocs, trie_alloc_bytes) = with_counting(|| AfterOpenTrie::new(&vocab));
    let trie_build_ns = t_trie.elapsed().as_secs_f64() * 1e9;
    let max_token_bytes = trie.max_token_bytes();

    let cases = build_cases();
    let mut kind_counts = std::collections::BTreeMap::<&str, u64>::new();
    for c in &cases {
        *kind_counts.entry(c.kind).or_default() += 1;
    }

    // Full-vocab parity across boundary fixtures.
    let parity = match run_parity_on_vocab("real", &cases, &vocab, &mut trie) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("PARITY FAIL: {e}");
            std::process::exit(1);
        }
    };

    let guards = match semantic_guards(&vocab, &mut trie) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("SEMANTIC GUARD FAIL: {e}");
            std::process::exit(1);
        }
    };

    let synthetic = match synthetic_vocab_parity(AfterOpenTrie::new) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("SYNTHETIC PARITY FAIL: {e}");
            std::process::exit(1);
        }
    };

    // Encode-trace snapshots for realistic weighted mixed measurement.
    let trace = match build_encode_trace(&tok, tools_basic()) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("TRACE FAIL: {e}");
            std::process::exit(1);
        }
    };

    const WARMUPS: u32 = 3;
    const REPEATS: u32 = 8;

    let mixed = match measure_trace_workload(&trace.snapshots, &vocab, &mut trie, WARMUPS, REPEATS)
    {
        Ok(v) => v,
        Err(e) => {
            eprintln!("TRACE WORKLOAD FAIL: {e}");
            std::process::exit(1);
        }
    };

    // Per-state benches: free + AfterOpen empty + key InArgs + a few trace samples.
    let bench_names: &[&str] = &[
        "free_out",
        "after_open_empty",
        "inargs_required_missing",
        "inargs_close_outer_open",
        "inargs_close_ready",
        "inargs_long_body",
        "merged_header_args",
    ];
    let mut bench_cases: Vec<&Case> = cases
        .iter()
        .filter(|c| bench_names.contains(&c.name.as_str()))
        .collect();
    // A couple of mid-header AfterOpen boundaries (not the full grid).
    let mut ao_extra = 0;
    for c in &cases {
        if c.kind == "AfterOpen" && c.name.contains("bash_b") && ao_extra < 3 {
            if !bench_cases.iter().any(|x| x.name == c.name) {
                bench_cases.push(c);
                ao_extra += 1;
            }
        }
    }
    // Trace samples: first AfterOpen, mid InArgs, last nonfree.
    if let Some(c) = trace.snapshots.iter().find(|c| c.kind == "AfterOpen") {
        bench_cases.push(c);
    }
    let inargs_n = trace.snapshots.iter().filter(|c| c.kind == "InArgs").count();
    if let Some(c) = trace
        .snapshots
        .iter()
        .filter(|c| c.kind == "InArgs")
        .nth(inargs_n / 2)
    {
        bench_cases.push(c);
    }
    if let Some(c) = trace.snapshots.last() {
        bench_cases.push(c);
    }

    let mut benches = Vec::new();
    for c in &bench_cases {
        let row = bench_case(c, &vocab, &mut trie, WARMUPS, REPEATS);
        benches.push(json!({
            "name": row.name,
            "kind": row.kind,
            "oracle_ns_per_fill": row.oracle_ns,
            "adapter_ns_per_fill": row.adapter_ns,
            "oracle_allocs_per_fill": row.oracle_allocs,
            "adapter_allocs_per_fill": row.adapter_allocs,
            "oracle_alloc_bytes_per_fill": row.oracle_alloc_bytes,
            "adapter_alloc_bytes_per_fill": row.adapter_alloc_bytes,
            "density": row.density,
            "repeats": row.repeats,
        }));
    }

    // Final reuse pass: alt tools on reused adapter.
    {
        let mut m = Matcher::new(tools_alt());
        m.advance("<tool_call>");
        let mut o = vec![false; n];
        let mut a = vec![false; n];
        m.token_mask(&vocab, &mut o);
        trie.fill(&m, &vocab, &mut a);
        if let Err(e) = compare_masks("final_alt_tools", &m, &vocab, &o, &a) {
            eprintln!("PARITY FAIL: {e}");
            std::process::exit(1);
        }
    }

    let report = json!({
        "ok": true,
        "model": model,
        "load_ms": load_ms,
        "vocab_build_ms": vocab_ms,
        "vocab_size": n,
        "trie_build_ns": trie_build_ns,
        "trie_build_allocs": trie_allocs,
        "trie_build_alloc_bytes": trie_alloc_bytes,
        "max_token_bytes": max_token_bytes,
        "fixture_counts": kind_counts,
        "fixture_total": cases.len(),
        "compared_bits_real": parity.0,
        "fixtures": parity.1,
        "semantic_guards": guards,
        "synthetic": synthetic,
        "encode_trace": {
            "token_count": trace.token_count,
            "nonfree_count": trace.nonfree_count,
            "kind_counts": trace.kind_counts,
            "tool_seen": trace.tool_seen,
            "final_state": trace.final_state,
            "text_bytes": trace.text_bytes,
            "snapshot_count": trace.snapshots.len(),
            "old_matcher_limitation": "ngram_history is a 256-byte rolling window; required_fields_satisfied looks for \"command\" inside it. Args bodies that push the required key out of the window yield req_satisfied=false and reject brace-close tokens. Trace body kept <200B args so the key remains. Grammar unchanged.",
            "body_shape": "4 lines ~20 chars each; whitespace-separated close quote/braces",
        },
        "bench": {
            "warmups": WARMUPS,
            "repeats": REPEATS,
            "timing_note": "wallclock with COUNTING=false; allocs in separate windows",
            "per_state": benches,
            "weighted_mixed": mixed,
        },
    });

    let s = serde_json::to_string(&report).expect("json");
    println!("{s}");
    if let Some(p) = out_path {
        if let Err(e) = std::fs::write(&p, &s) {
            eprintln!("write {p}: {e}");
            std::process::exit(2);
        }
    }
}
