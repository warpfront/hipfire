// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `vl_vcn_lease_race`: contention proof for the shared VCN session lease.
//!
//! Two distinct same-geometry JPEGs share one pool key
//! (`rt_format`, width, height ⇒ one pooled surface). The main thread holds
//! image A's [`va_bridge::SharedVcnLease`] (via product [`vcn_decode`]) while a
//! contender thread attempts image B:
//!
//! 1. contender `try_shared_decode_jpeg(B)` ⇒ `Ok(None)` — deterministic,
//!    timing-free proof that B cannot reuse the surface early;
//! 2. contender then parks in blocking `shared_decode_jpeg_lease(B)`;
//! 3. main consumes A through product [`vcn_to_patches`] (both preprocess
//!    kernels + checked terminal stream sync — the release boundary). Only
//!    then does the contender's decode return;
//! 4. contender captures a **scalar receipt** (width/height/fourcc/pool key),
//!    drops the lease on the contender thread (lease is not `Send`), and
//!    sends the receipt to main — never the lease, never a raw pointer;
//! 5. main re-decodes B under its own lease via product [`vcn_decode`] +
//!    [`vcn_to_patches`], then compares each output against its OWN CPU
//!    oracle (`load_and_preprocess_from_bytes` + `extract_patches`), plus a
//!    cross-check that A's output is not B's image;
//! 6. **last** (process-lifetime poison): main acquires a fresh A lease,
//!    calls [`va_bridge::VaSession::quarantine_shared`] with **no GPU work**
//!    (explicit simulated completion failure — not a hang reproduction),
//!    then asserts both `try_shared_decode_jpeg(B)` and blocking
//!    `shared_decode_jpeg_lease(B)` return `Err` naming quarantine — not
//!    `Ok(None)`, not a wait, not a decode.
//!
//! Every barrier step uses `recv_timeout`: the probe is bounded, never
//! hangs, and fails closed (nonzero exit) on VA/HIP absence, unsuitable
//! fixtures, exclusion violation, deadlock, or parity breach. All HIP work
//! runs on the main thread; the contender only touches the shared session
//! (libva calls never overlap in time — the mutex serializes them).
//!
//! Run (device 0, GPU lock held by the caller):
//! ```sh
//! exec 9>/tmp/hipfire-gpu.lock; flock -w 300 9
//! HIPFIRE_IMAGE_DECODE=vcn cargo run --release -p saddle-lab --features vcn-jpeg --example vl_vcn_lease_race -- \
//!   --a=benchmarks/vision/images/doge.jpeg \
//!   --b=crates/saddle-lab/examples/fixtures/lease-race-red-537x529.jpg
//! ```
//!
//! `--timeout-s=N` bounds each barrier step (default 120).
use hipfire_arch_qwen35_vl::image::{
    extract_patches, load_and_preprocess_from_bytes, smart_resize, vcn_decode, vcn_to_patches,
};
use rdna_compute::Gpu;
use std::time::{Duration, Instant};

const PATCH: usize = 16;
const TEMPORAL: usize = 2; // vision_config_from_hfq default (qwen35_vl.rs:52-55)
const SMS: usize = 2;
const FACTOR: usize = PATCH * SMS;
const MIN_PX: usize = 65_536;
const MAX_PX: usize = 2_000_000; // VISION_MAX_PIXELS default (image.rs:67)

/// Acceptance: per-image rel-L1 within ~1e-2 (the `vl_vcn_parity` bound).
const PARITY_BOUND: f64 = 1e-2;

fn rel_l1(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b.iter()) {
        num += (*x as f64 - *y as f64).abs();
        den += (*x as f64).abs();
    }
    num / den.max(1e-30)
}

/// Contender → main protocol. Channel order is the barrier: `Contended`
/// arrives while A is held, `Decoded` only after A is released. The
/// `Decoded` arm carries only `Copy` scalars — never a lease or pointer
/// (`SharedVcnLease` embeds a `MutexGuard` and is not `Send`).
enum ContenderEvent {
    /// `try_shared_decode_jpeg(B)` returned `Ok(None)` while A was held:
    /// exclusion proven without timing.
    Contended,
    /// `try_shared_decode_jpeg(B)` decoded while A was held: the race the
    /// lease exists to prevent. Fails the probe loudly.
    Violated,
    /// Blocking decode returned after A released. Contender captured the
    /// scalar receipt and dropped the lease on its own thread.
    Decoded {
        /// Source width from the contender-local frame.
        width: u32,
        /// Source height from the contender-local frame.
        height: u32,
        /// Surface fourcc (format).
        fourcc: u32,
        /// Pool key half shared with fixtures: `rt_format` from the JPEG
        /// parse (with width/height forms the full pool key).
        rt_format: u32,
        returned_at: Instant,
    },
    /// B unsuitable for the probe (unsupported stream / decode error).
    /// Fail closed, not a race verdict.
    Unusable(String),
}

fn usage() -> ! {
    eprintln!(
        "usage: vl_vcn_lease_race --a=PATH --b=PATH [--timeout-s=N]\n\
         --a/--b must be distinct baseline 4:2:0 JPEGs with identical geometry (one pool key)"
    );
    std::process::exit(2);
}

fn main() {
    // Product path defaults to CPU; force VCN for this contention probe.
    // Must precede the first `process_value` / `vcn_decode` read.
    // SAFETY: single-threaded startup; no other threads observe env yet.
    unsafe {
        std::env::set_var("HIPFIRE_IMAGE_DECODE", "vcn");
    }

    let (mut a_path, mut b_path) = (None::<String>, None::<String>);
    let mut timeout_s = 120u64;
    for arg in std::env::args().skip(1) {
        if let Some(v) = arg.strip_prefix("--a=") {
            a_path = Some(v.to_string());
        } else if let Some(v) = arg.strip_prefix("--b=") {
            b_path = Some(v.to_string());
        } else if let Some(v) = arg.strip_prefix("--timeout-s=") {
            timeout_s = v.parse().unwrap_or_else(|_| usage());
        } else {
            usage();
        }
    }
    let (a_path, b_path) = match (a_path, b_path) {
        (Some(a), Some(b)) => (a, b),
        _ => usage(),
    };
    let bound = Duration::from_secs(timeout_s);

    let a_bytes = std::fs::read(&a_path).unwrap_or_else(|e| {
        eprintln!("[lease-race] --a {a_path}: {e}");
        std::process::exit(1);
    });
    let b_bytes = std::fs::read(&b_path).unwrap_or_else(|e| {
        eprintln!("[lease-race] --b {b_path}: {e}");
        std::process::exit(1);
    });
    assert_ne!(a_bytes, b_bytes, "probe images must be distinct JPEGs");

    // Same pool key ⇒ one shared surface: the contention is real, not
    // two independent pool slots.
    let pa = va_bridge::parse_for_va(&a_bytes).unwrap_or_else(|e| {
        eprintln!("[lease-race] --a parses for VA: {e}");
        std::process::exit(1);
    });
    let pb = va_bridge::parse_for_va(&b_bytes).unwrap_or_else(|e| {
        eprintln!("[lease-race] --b parses for VA: {e}");
        std::process::exit(1);
    });
    assert_eq!(
        (pa.rt_format, pa.width, pa.height),
        (pb.rt_format, pb.width, pb.height),
        "probe images must share one pool key (rt_format, width, height)"
    );
    println!(
        "[lease-race] shared pool key: rt_format=0x{:x} {}x{}",
        pa.rt_format, pa.width, pa.height
    );
    let pool_key = (pa.rt_format, u32::from(pa.width), u32::from(pa.height));

    // CPU oracles (repo path, upfront — also validates both fixtures on CPU).
    let cpu_patches = |name: &str, bytes: &[u8]| -> (Vec<f32>, usize, usize) {
        let (pixels, img_h, img_w) = load_and_preprocess_from_bytes(bytes, PATCH, SMS)
            .unwrap_or_else(|e| {
                eprintln!("[lease-race] {name}: CPU preprocess failed: {e}");
                std::process::exit(1);
            });
        let patches = extract_patches(&pixels, 3, img_h, img_w, PATCH, TEMPORAL, SMS);
        (patches, img_h, img_w)
    };
    let (oracle_a, img_h_a, img_w_a) = cpu_patches("--a", &a_bytes);
    let (oracle_b, img_h_b, img_w_b) = cpu_patches("--b", &b_bytes);
    assert_eq!(
        (img_h_a, img_w_a),
        (img_h_b, img_w_b),
        "resized targets must agree (same geometry ⇒ same smart_resize)"
    );
    assert!(
        !oracle_a.is_empty() && !oracle_b.is_empty(),
        "CPU oracles must be nonempty"
    );
    // Distinct fixtures must produce distinct CPU patch tensors — otherwise
    // the cross-identity self-check is vacuous.
    assert!(
        rel_l1(&oracle_a, &oracle_b) > PARITY_BOUND,
        "fixtures are too similar: cross-identity check would be vacuous"
    );

    // ——— HIP via product Gpu (vcn_to_patches owns the kernel path) ———
    let mut gpu = Gpu::init().unwrap_or_else(|e| {
        eprintln!("[lease-race] Gpu::init: {e}");
        std::process::exit(1);
    });
    println!("[lease-race] hip arch={}", gpu.arch);

    // ——— race: hold A, contender attempts B ———
    // Decode A on main (product path). Lease stays on main until
    // vcn_to_patches completes its terminal sync.
    let dec_a = match vcn_decode(&a_bytes, PATCH, SMS) {
        Some(d) => d,
        None => {
            eprintln!("[lease-race] --a vcn_decode returned None (fixture/mode unsuitable)");
            std::process::exit(1);
        }
    };
    assert_eq!(
        (dec_a.img_h, dec_a.img_w),
        (img_h_a, img_w_a),
        "A VCN resize vs CPU oracle"
    );
    {
        let f = dec_a.frame();
        assert_eq!(
            (f.width, f.height),
            (u32::from(pa.width), u32::from(pa.height)),
            "A frame geometry vs parse"
        );
    }
    println!("[lease-race] A decoded under lease (surface held)");

    let (tx_held, rx_held) = std::sync::mpsc::channel::<()>();
    let (tx_ev, rx_ev) = std::sync::mpsc::channel::<ContenderEvent>();
    let b_bytes_c = b_bytes.clone();
    let pb_rt = pb.rt_format;
    let contender = std::thread::spawn(move || {
        if rx_held.recv_timeout(bound).is_err() {
            panic!("contender: no A-held barrier within {bound:?}");
        }
        // Timing-free exclusion proof: the session is leased, so the
        // non-blocking attempt must observe contention — never a frame.
        match va_bridge::VaSession::try_shared_decode_jpeg(&b_bytes_c) {
            Ok(None) => tx_ev.send(ContenderEvent::Contended).unwrap(),
            Ok(Some(_)) => {
                tx_ev.send(ContenderEvent::Violated).unwrap();
                return;
            }
            Err(e) => {
                tx_ev
                    .send(ContenderEvent::Unusable(format!("try decode: {e}")))
                    .unwrap();
                return;
            }
        }
        // Park in the blocking decode: returns only after main releases A
        // (terminal sync inside vcn_to_patches drops the lease).
        match va_bridge::VaSession::shared_decode_jpeg_lease(&b_bytes_c) {
            Ok(va_bridge::SharedDecodeOutcome::Decoded(lease)) => {
                let returned_at = Instant::now();
                // Scalar receipt only — copy geometry/format out, then drop
                // the lease on THIS thread. Never send the lease (not Send).
                let (width, height, fourcc) = {
                    let f = lease.frame();
                    (f.width, f.height, f.fourcc)
                };
                drop(lease);
                tx_ev
                    .send(ContenderEvent::Decoded {
                        width,
                        height,
                        fourcc,
                        rt_format: pb_rt,
                        returned_at,
                    })
                    .unwrap();
            }
            Ok(va_bridge::SharedDecodeOutcome::Unsupported(reason)) => {
                tx_ev
                    .send(ContenderEvent::Unusable(format!("unsupported ({reason})")))
                    .unwrap();
            }
            Err(e) => {
                tx_ev
                    .send(ContenderEvent::Unusable(format!("decode: {e}")))
                    .unwrap();
            }
        }
    });

    tx_held.send(()).unwrap();
    match rx_ev.recv_timeout(bound) {
        Ok(ContenderEvent::Contended) => {
            println!(
                "[lease-race] PROOF 1/3: try_decode(B) while A held ⇒ contended (no early reuse)"
            );
        }
        Ok(ContenderEvent::Violated) => {
            eprintln!("[lease-race] RACE DETECTED: B decoded while A was leased");
            std::process::exit(1);
        }
        Ok(ContenderEvent::Unusable(msg)) => {
            eprintln!("[lease-race] B unusable: {msg}");
            std::process::exit(1);
        }
        Ok(ContenderEvent::Decoded { .. }) => {
            eprintln!("[lease-race] B decode returned before A terminal sync");
            std::process::exit(1);
        }
        Err(_) => {
            eprintln!("[lease-race] barrier timeout waiting for contention proof");
            std::process::exit(1);
        }
    }

    // Consume A under the lease via product path: kernels + checked
    // terminal sync, then the lease drops inside vcn_to_patches. Contender
    // is still parked on the blocking B decode.
    // VcnDecoded is non-Copy: move it in by value.
    //
    // Handoff proof is channel order, not Instant compare: the lease drops
    // *inside* vcn_to_patches (before it returns), so the contender may stamp
    // `returned_at` before main observes the function return. We only start
    // waiting for Decoded after this call returns — that is the barrier.
    let patches_a = vcn_to_patches(&mut gpu, dec_a, PATCH, TEMPORAL, SMS).unwrap_or_else(|e| {
        eprintln!("[lease-race] A vcn_to_patches: {e}");
        std::process::exit(1);
    });
    println!("[lease-race] A consumed + synced, lease released");
    let out_a = gpu.download_f32(&patches_a.patches).unwrap_or_else(|e| {
        eprintln!("[lease-race] A download_f32: {e}");
        std::process::exit(1);
    });
    let _ = gpu.free_tensor(patches_a.patches);

    let b_receipt = match rx_ev.recv_timeout(bound) {
        Ok(ContenderEvent::Decoded {
            width,
            height,
            fourcc,
            rt_format,
            returned_at,
        }) => {
            // Channel barrier: this arm is only reachable after A terminal
            // sync released the lease (we waited on Decoded only after
            // vcn_to_patches returned). Log contender stamp for diagnostics.
            let _ = returned_at;
            assert_eq!(
                (rt_format, width, height),
                pool_key,
                "B receipt pool key must match fixtures"
            );
            assert_eq!(
                fourcc,
                va_bridge::VA_FOURCC_NV12,
                "probe pins the NV12 lane (both fixtures are 4:2:0)"
            );
            println!(
                "[lease-race] PROOF 2/3: B blocking decode returned only after A release \
                 (receipt {width}x{height} fourcc=0x{fourcc:08x} key=0x{rt_format:x}; \
                 channel-ordered after vcn_to_patches)"
            );
            (width, height, fourcc, rt_format)
        }
        Ok(ContenderEvent::Violated) => {
            eprintln!("[lease-race] RACE DETECTED late");
            std::process::exit(1);
        }
        Ok(ContenderEvent::Unusable(msg)) => {
            eprintln!("[lease-race] B unusable: {msg}");
            std::process::exit(1);
        }
        Ok(ContenderEvent::Contended) | Err(_) => {
            eprintln!("[lease-race] barrier timeout waiting for B handoff (deadlock?)");
            std::process::exit(1);
        }
    };
    contender.join().unwrap_or_else(|e| {
        eprintln!("[lease-race] contender panicked: {e:?}");
        std::process::exit(1);
    });
    // Silence unused-if-assert-elided; receipt already checked above.
    let _ = b_receipt;

    // Main re-decodes B under its own lease and consumes through product GPU.
    // Contender already dropped its B lease; the pool is free.
    let dec_b = match vcn_decode(&b_bytes, PATCH, SMS) {
        Some(d) => d,
        None => {
            eprintln!("[lease-race] --b vcn_decode returned None after handoff");
            std::process::exit(1);
        }
    };
    assert_eq!(
        (dec_b.img_h, dec_b.img_w),
        (img_h_b, img_w_b),
        "B VCN resize vs CPU oracle"
    );
    // smart_resize cross-check (geometry scalars only — no frame borrow held
    // across the by-value vcn_to_patches move).
    {
        let f = dec_b.frame();
        let (t_h, t_w) = smart_resize(f.height as usize, f.width as usize, FACTOR, MIN_PX, MAX_PX);
        assert_eq!((t_h, t_w), (img_h_b, img_w_b));
        assert_eq!(
            (f.width, f.height),
            (u32::from(pb.width), u32::from(pb.height))
        );
    }
    let patches_b = vcn_to_patches(&mut gpu, dec_b, PATCH, TEMPORAL, SMS).unwrap_or_else(|e| {
        eprintln!("[lease-race] B vcn_to_patches: {e}");
        std::process::exit(1);
    });
    let out_b = gpu.download_f32(&patches_b.patches).unwrap_or_else(|e| {
        eprintln!("[lease-race] B download_f32: {e}");
        std::process::exit(1);
    });
    let _ = gpu.free_tensor(patches_b.patches);

    assert!(
        !out_a.is_empty() && !out_b.is_empty(),
        "GPU patch tensors must be nonempty"
    );
    assert_eq!(out_a.len(), oracle_a.len(), "A patch count vs oracle");
    assert_eq!(out_b.len(), oracle_b.len(), "B patch count vs oracle");

    // Each output against its OWN oracle, plus the cross self-check.
    let r_a = rel_l1(&out_a, &oracle_a);
    let r_b = rel_l1(&out_b, &oracle_b);
    let r_cross = rel_l1(&out_a, &oracle_b);
    let r_ab = rel_l1(&out_a, &out_b);
    println!(
        "[lease-race] rel-L1 A-vs-own={r_a:.3e} B-vs-own={r_b:.3e} \
         A-vs-B-oracle={r_cross:.3e} A-vs-B-out={r_ab:.3e}"
    );
    assert!(
        r_a <= PARITY_BOUND,
        "A output drifted from its own oracle (surface clobbered?)"
    );
    assert!(
        r_b <= PARITY_BOUND,
        "B output drifted from its own oracle (surface clobbered?)"
    );
    assert!(
        r_cross > PARITY_BOUND,
        "A output matches B's oracle: harness buffer mixup suspected"
    );
    assert!(
        r_ab > PARITY_BOUND,
        "A and B GPU outputs match: fixtures not distinct under VCN path"
    );
    println!("[lease-race] PROOF 3/3: each output matches its own oracle — PASS");

    // ——— PROOF 4/4: simulated completion-failure quarantine (LAST) ———
    // Process-lifetime poison of the shared session. No GPU hang: we lease
    // A, mark quarantine_shared without launching kernels, and prove both
    // try and blocking B paths fail closed with Err(quarantine) — never
    // Ok(None) contention, never a successful decode, never a park.
    println!(
        "[lease-race] SIMULATED-FAULT: quarantine_shared(A) without GPU work \
         (explicit completion-failure stand-in — not a hang reproduction)"
    );
    let q_lease = match va_bridge::VaSession::shared_decode_jpeg_lease(&a_bytes) {
        Ok(va_bridge::SharedDecodeOutcome::Decoded(l)) => l,
        Ok(va_bridge::SharedDecodeOutcome::Unsupported(reason)) => {
            eprintln!("[lease-race] quarantine setup: A unsupported ({reason})");
            std::process::exit(1);
        }
        Err(e) => {
            eprintln!("[lease-race] quarantine setup: A lease failed ({e})");
            std::process::exit(1);
        }
    };
    va_bridge::VaSession::quarantine_shared(q_lease);
    println!("[lease-race] shared session marked quarantined (SIMULATED-FAULT)");

    match va_bridge::VaSession::try_shared_decode_jpeg(&b_bytes) {
        Err(e) => {
            let msg = e.to_string();
            assert!(
                msg.to_ascii_lowercase().contains("quarantine"),
                "try_shared_decode_jpeg(B) after quarantine must name quarantine, got: {msg}"
            );
            println!(
                "[lease-race] PROOF 4a/4: try_shared_decode_jpeg(B) ⇒ Err(quarantine) \
                 (not Ok(None)/decode)"
            );
        }
        Ok(None) => {
            eprintln!(
                "[lease-race] quarantine miss: try_shared_decode_jpeg(B) returned Ok(None) \
                 (contention) — expected Err(quarantine)"
            );
            std::process::exit(1);
        }
        Ok(Some(_)) => {
            eprintln!(
                "[lease-race] quarantine miss: try_shared_decode_jpeg(B) decoded after poison"
            );
            std::process::exit(1);
        }
    }

    match va_bridge::VaSession::shared_decode_jpeg_lease(&b_bytes) {
        Err(e) => {
            let msg = e.to_string();
            assert!(
                msg.to_ascii_lowercase().contains("quarantine"),
                "shared_decode_jpeg_lease(B) after quarantine must name quarantine, got: {msg}"
            );
            println!(
                "[lease-race] PROOF 4b/4: shared_decode_jpeg_lease(B) ⇒ Err(quarantine) \
                 (not wait/decode) — PASS"
            );
        }
        Ok(va_bridge::SharedDecodeOutcome::Decoded(_)) => {
            eprintln!(
                "[lease-race] quarantine miss: blocking shared_decode_jpeg_lease(B) decoded \
                 after poison"
            );
            std::process::exit(1);
        }
        Ok(va_bridge::SharedDecodeOutcome::Unsupported(reason)) => {
            eprintln!(
                "[lease-race] quarantine miss: blocking path Unsupported({reason}) — \
                 expected Err(quarantine)"
            );
            std::process::exit(1);
        }
    }
}
