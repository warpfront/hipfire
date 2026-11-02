// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! Production-size two-bank transport gate. No HIP VMM export, no SYNC_BO.
#[path = "npu-trunk-transport/gpu.rs"]
mod gpu;
use npu_tools::coop_gpu;
use peacemaker_npu::kernels::{gemm_g80, gemm_g80_n512, w8a8_epi::bf16_direct};
use railgun::npu::{
    clflush,
    trunk::{
        monotonic_ns, LayerPanels, NpuTrunkRuntime, OpenOptions, PanelProgram, Seq, Shape,
        CHAIN_WATCHDOG, OUT_PANEL_K, OUT_PANEL_N, PANEL_K, PANEL_N, WAIT_TIMEOUT,
    },
    Bo, Device, HwCtx, ERT_STATE_COMPLETED,
};
use std::time::{Duration, Instant};
const ROWS: usize = 8192;
const CLASSES: usize = 16;
const DEFAULT_ROUNDS: usize = 20;
const POISON: u8 = 0xa5;
fn a_code(row: usize, k: usize, round: usize) -> i8 {
    ((k * 5 + (row % CLASSES) * 7 + round * 11) % 23) as i8 - 11
}
fn b_code(panel: usize, k: usize, col: usize) -> i8 {
    ((k * 7 + (col % CLASSES) * 3 + panel * 5) % 19) as i8 - 9
}
fn dm(row: usize, round: usize) -> (u16, i16) {
    if row % CLASSES == 0 {
        (0, 0)
    } else {
        (
            16384 + (row % CLASSES) as u16 * 127 + round as u16 * 17,
            -20 + (row % 3) as i16,
        )
    }
}
fn sm(panel: usize, col: usize) -> (u16, i16) {
    (
        16384 + (col % CLASSES) as u16 * 151 + panel as u16 * 31,
        -19 + (panel % 3) as i16,
    )
}
fn percentile(values: &[f64], p: f64) -> f64 {
    values[((values.len() - 1) as f64 * p).ceil() as usize]
}
fn at<T>(round: Option<usize>, stage: &str, result: Result<T, String>) -> Result<T, String> {
    result.map_err(|e| {
        format!(
            "stage={stage} round={}: {e}",
            round.map_or_else(|| "none".into(), |r| r.to_string())
        )
    })
}
fn resume_latency_ns(anchor_ns: u64, elapsed_ms: f32, done_ns: u64) -> i128 {
    i128::from(anchor_ns) + (f64::from(elapsed_ms) * 1e6) as i128 - i128::from(done_ns)
}
fn consumer_columns(shape: Shape) -> Vec<std::ops::Range<usize>> {
    match shape {
        Shape::GdnIn | Shape::DenseGdnIn => {
            vec![0..10240, 10240..16384, 16384..16432, 16432..16480]
        }
        Shape::QsaIn => vec![0..12288, 12288..12800, 12800..13312, 13312..13952],
        // The out projection feeds the whole 2560-wide residual: quarters stand in for the consumer windows.
        Shape::GdnOut | Shape::QsaO => vec![0..640, 640..1280, 1280..1920, 1920..2560],
        // 27B gated q, k, v: 28 panels exactly, no padding.
        Shape::DenseFaIn => vec![0..12288, 12288..13312, 13312..14336],
    }
}

/// GPU contention for `--corun`: 64 device-to-device copies of 256 MiB (16 GiB read + 16 GiB written per round) queued on a
/// second stream right before every kick, so the NPU chain runs against a memory-bound GPU.
const CORUN_BYTES: usize = 256 << 20;
const CORUN_COPIES: usize = 64;

fn run(
    pci: &str,
    shape: Shape,
    rounds: usize,
    corun: bool,
    backend: gpu::Backend,
) -> Result<(), String> {
    // Match serving's load order: hip-bridge first, then the NPU relay. Keep
    // the serving owner alive until checked NPU teardown and its watchdog test.
    let serving = at(None, "open-serving-GPU", backend.load(pci))?;
    let (kdim, pn) = (shape.k(), shape.panel_n());
    let panels_count = shape.panels();
    let columns = consumer_columns(shape);
    let live = columns.last().unwrap().end;
    // The consumer launch geometry uses precisely these live windows, not a
    // panel-rounded width. Diagnostic full-C downloads below also check padding.
    // (The N512 shapes need no padding: out = five 512-column panels, dense FA = 28; dense GDN pads 16480 to 16896.)
    if columns[0].start != 0
        || columns.windows(2).any(|w| w[0].end != w[1].start)
        || columns.iter().any(|r| r.is_empty() || r.end > live)
        || live > shape.pitch()
        || (live == shape.pitch() && !shape.is_n512())
    {
        return at(
            None,
            "consumer-padding-bounds",
            Err("consumer reads padded columns".into()),
        );
    }
    let design = at(
        None,
        "build-panel",
        if shape.is_n512() {
            gemm_g80_n512::build_panel_program_n512(ROWS, shape.pitch(), 0, kdim)
        } else {
            gemm_g80::build_panel_program_rowmajor(ROWS, shape.pitch(), 0)
        },
    )?;
    let mut panels = Vec::with_capacity(panels_count);
    for panel in 0..panels_count {
        let b: Vec<i8> = (0..kdim)
            .flat_map(|k| {
                (0..pn).map(move |n| {
                    if panel * pn + n < live {
                        b_code(panel, k, n)
                    } else {
                        0
                    }
                })
            })
            .collect();
        let scales: Vec<_> = (0..pn)
            .map(|n| {
                if panel * pn + n < live {
                    sm(panel, n)
                } else {
                    (0, 0)
                }
            })
            .collect();
        panels.push(gemm_g80::pack_b_bf16(&design, &b, &scales));
    }
    let program = PanelProgram {
        shape,
        pdi: design.pdi.clone(),
        txn: design.insts.clone(),
        padded_rows: ROWS,
        pitch: shape.pitch(),
    };
    let runtime = at(
        None,
        "open-transport",
        NpuTrunkRuntime::open_with_options(
            vec![program],
            vec![LayerPanels {
                layer: 0,
                shape,
                panels,
            }],
            2,
            ROWS,
            pci,
            OpenOptions {
                cpu_reads: true,
                ..OpenOptions::default()
            },
        ),
    )?;
    let outcome = (|| {
        let hip = at(
            None,
            "GPU-runtime-identity",
            gpu::Gpu::new(runtime.hip(), serving.as_ref()),
        )?;
        let layout = runtime.layout();
        let a_bytes = layout.a_bytes_for(shape);
        let c_bytes = layout.c_bytes(shape);
        let module = at(
            None,
            "load-copy-module",
            hip.load_module(coop_gpu::CODE_OBJECT),
        )?;
        let copy = at(None, "get-copy-kernel", module.function(coop_gpu::COPY))?;
        let stream = at(None, "create-stream", hip.stream())?;
        let load_stream = at(None, "create-load-stream", hip.stream())?;
        let load_end = at(None, "create-load-event", hip.event())?;
        let load = if corun {
            Some((
                at(None, "allocate-load-src", hip.allocate(CORUN_BYTES))?,
                at(None, "allocate-load-dst", hip.allocate(CORUN_BYTES))?,
            ))
        } else {
            None
        };
        let end = at(None, "create-system-event", hip.system_event())?;
        let anchor = at(None, "create-timing-anchor", hip.event())?;
        let resumed = at(None, "create-resume-event", hip.event())?;
        // Map the HIP event clock into the relay's CLOCK_MONOTONIC domain.
        // Report the bracket width rather than pretending these are exact clocks.
        at(None, "warm-anchor", hip.record_on(&anchor, &stream))?;
        at(
            None,
            "warm-anchor-completion",
            hip.wait_event_timeout(&anchor, WAIT_TIMEOUT),
        )?;
        let anchor_before = monotonic_ns();
        at(None, "record-anchor", hip.record_on(&anchor, &stream))?;
        at(
            None,
            "anchor-completion",
            hip.wait_event_timeout(&anchor, WAIT_TIMEOUT),
        )?;
        let anchor_after = monotonic_ns();
        let anchor_ns = anchor_before + (anchor_after - anchor_before) / 2;
        let anchor_uncertainty_ns = anchor_after - anchor_before;
        let input = at(None, "allocate-input", hip.allocate(a_bytes))?;
        let output = at(None, "allocate-output", hip.allocate(c_bytes))?;
        let poison = at(None, "allocate-poison", hip.allocate(c_bytes))?;
        at(None, "initialize-poison", hip.memset(&poison, POISON))?;
        let blocks = at(None, "query-compute-units", hip.compute_units())? * 8;
        let gpu_copy = |dst: u64, src: u64, bytes: usize| -> Result<(), String> {
            if bytes % 16 != 0 {
                return Err("copy size not aligned".into());
            }
            let mut args = coop_gpu::copy_args(
                dst,
                src,
                (bytes / 16) as u64,
                u64::from(blocks * coop_gpu::COPY_BLOCK),
            );
            hip.launch_on(&stream, &copy, blocks, coop_gpu::COPY_BLOCK, &mut args)
        };
        let drain = || -> Result<(), String> {
            hip.record_on(&end, &stream)?;
            hip.wait_event_timeout(&end, WAIT_TIMEOUT)
        };
        let mut observed = vec![0u8; c_bytes];
        let mut a = vec![0i8; ROWS * kdim];
        let mut scales = vec![(0u16, 0i16); ROWS];
        let mut reference = vec![0u16; panels_count * CLASSES * CLASSES * 3];
        let mut latencies = Vec::with_capacity(rounds);
        let mut busies = Vec::with_capacity(rounds);
        let mut resumes = Vec::with_capacity(rounds);
        let mut busy = Duration::ZERO;
        let mut last_seq = [None; 2];
        let wall = Instant::now();
        println!("transport shape={shape:?} layout=rowmajor pitch={} live={live} banks=2 panels={panels_count} rounds={rounds} corun={corun} A={} C={c_bytes} bytes/bank ready_timeout={WAIT_TIMEOUT:?} chain_watchdog={CHAIN_WATCHDOG:?} shim_axi=default consumer_windows={columns:?} anchor_uncertainty_ns={anchor_uncertainty_ns}", shape.pitch(), a_bytes);
        for round in 0..rounds {
            let bank = round % 2;
            for row in 0..ROWS {
                scales[row] = dm(row, round);
                for k in 0..kdim {
                    a[row * kdim + k] = a_code(row, k, round);
                }
            }
            let packed = gemm_g80::pack_a_bf16(&design, &a, &scales);
            if packed.len() != a_bytes {
                return at(Some(round), "pack-A", Err("A sizing mismatch".into()));
            }
            at(Some(round), "upload-packed-A", hip.upload(&input, &packed))?;
            // Independent CPU dot products for EVERY class combination. The
            // operand is periodic in rows/columns, but comparisons cover EVERY
            // byte of the fused row-major C arena, including zero padding.
            for panel in 0..panels_count {
                for r in 0..CLASSES {
                    for n in 0..CLASSES {
                        let dot: i32 = (0..kdim)
                            .map(|k| {
                                i32::from(a_code(r, k, round)) * i32::from(b_code(panel, k, n))
                            })
                            .sum();
                        let (sm, se) = sm(panel, n);
                        for rem in 0..3 {
                            let (dm, _) = dm(r, round);
                            reference[((panel * CLASSES + r) * CLASSES + n) * 3 + rem] =
                                bf16_direct(dot, sm, se, dm, -20 + rem as i16);
                        }
                    }
                }
            }
            let av = at(
                Some(round),
                "A-arena",
                if shape.is_n512() {
                    runtime.a_out_arena(bank)
                } else {
                    runtime.a_arena(bank)
                },
            )?;
            let cv = at(Some(round), "C-arena", runtime.c_arena(bank, shape))?;
            at(Some(round), "prepare-bank", unsafe {
                runtime.prepare_bank(bank, stream.raw())
            })?;
            at(
                Some(round),
                "poison-C",
                gpu_copy(cv.device, poison.ptr(), c_bytes),
            )?;
            at(
                Some(round),
                "GPU-write-A",
                gpu_copy(av.device, input.ptr(), a_bytes),
            )?;
            if let Some((src, dst)) = &load {
                for _ in 0..CORUN_COPIES {
                    let mut args = coop_gpu::copy_args(
                        dst.ptr(),
                        src.ptr(),
                        (CORUN_BYTES / 16) as u64,
                        u64::from(blocks * coop_gpu::COPY_BLOCK),
                    );
                    at(
                        Some(round),
                        "queue-load",
                        hip.launch_on(&load_stream, &copy, blocks, coop_gpu::COPY_BLOCK, &mut args),
                    )?;
                }
                at(
                    Some(round),
                    "record-load-end",
                    hip.record_on(&load_end, &load_stream),
                )?;
            }
            let seq = at(Some(round), "kick", unsafe {
                runtime.kick(0, shape, bank, ROWS, stream.raw())
            })?;
            at(Some(round), "wait-GPU-enqueue-ahead", unsafe {
                runtime.wait_gpu(seq, stream.raw())
            })?;
            let wait_enqueue_ns = monotonic_ns();
            at(
                Some(round),
                "record-GPU-resume",
                hip.record_on(&resumed, &stream),
            )?;
            at(
                Some(round),
                "GPU-read-C",
                gpu_copy(output.ptr(), cv.device, c_bytes),
            )?;
            at(Some(round), "retire-bank", unsafe {
                runtime.retire_bank(seq, stream.raw())
            })?;
            last_seq[bank] = Some(seq);
            at(Some(round), "consumer-completion", drain())?;
            if load.is_some() {
                at(
                    Some(round),
                    "load-completion",
                    hip.wait_event_timeout(&load_end, Duration::from_secs(30)),
                )?;
            }
            let completion = at(
                Some(round),
                "check-terminal-completion",
                runtime.wait_host(seq),
            )?;
            at(Some(round), "check-runtime-error", runtime.check_error())?;
            if wait_enqueue_ns >= completion.done_ns {
                return at(
                    Some(round),
                    "enqueue-ahead-order",
                    Err("GPU wait was not enqueued before NPU completion".into()),
                );
            }
            let resume_ns = resume_latency_ns(
                anchor_ns,
                at(
                    Some(round),
                    "GPU-resume-time",
                    hip.elapsed_ms(&anchor, &resumed),
                )?,
                completion.done_ns,
            );
            resumes.push(resume_ns as f64 / 1000.0);
            at(
                Some(round),
                "download-C",
                hip.download(&output, &mut observed),
            )?;
            // The oracle intentionally reads C's CPU alias; open opted into
            // per-completion invalidation before terminal publication.
            if unsafe { core::slice::from_raw_parts(cv.host, cv.len) } != &observed[..] {
                return at(
                    Some(round),
                    "compare-C-alias",
                    Err(format!("bank {bank}: GPU C/userptr alias mismatch")),
                );
            }
            // Also prove the entire GPU-written A aliases the anonymous pages.
            unsafe {
                clflush(av.host, av.len);
            }
            if unsafe { core::slice::from_raw_parts(av.host, av.len) } != &packed[..] {
                return at(
                    Some(round),
                    "compare-A-alias",
                    Err(format!("bank {bank}: GPU A/userptr alias mismatch")),
                );
            }
            for panel in 0..panels_count {
                for row in 0..ROWS {
                    for col in 0..pn {
                        let fused_col = panel * pn + col;
                        let offset = 2 * (row * shape.pitch() + fused_col);
                        let actual = u16::from_le_bytes([observed[offset], observed[offset + 1]]);
                        let expected = if fused_col < live {
                            reference[((panel * CLASSES + row % CLASSES) * CLASSES + col % CLASSES)
                                * 3
                                + row % 3]
                        } else {
                            0
                        };
                        if actual != expected {
                            return at(Some(round), "compare-C", Err(format!("bank {bank} panel {panel} row {row} col {col}: {actual:04x} != {expected:04x}")));
                        }
                    }
                }
            }
            let ms = completion.kick_to_done.as_secs_f64() * 1000.0;
            busy += completion.npu_busy;
            latencies.push(ms);
            busies.push(completion.npu_busy.as_secs_f64() * 1000.0);
            println!("round={round:02} bank={bank} seq={} exact enqueue_ahead=true gpu_wait_enqueue_ns={wait_enqueue_ns} npu_done_ns={} kick->done={ms:.3} ms npu_submit_wait={:.3} ms npu_done->gpu_resume={:.3} us", seq.value, completion.done_ns, completion.npu_busy.as_secs_f64() * 1000.0, resume_ns as f64 / 1000.0);
        }
        latencies.sort_by(f64::total_cmp);
        busies.sort_by(f64::total_cmp);
        println!("EXACT rounds={rounds} p50={:.3} p99={:.3} max={:.3} ms npu_busy_envelope={:.3} ms wall={:.3} ms (busy=submit-to-timeline host envelope, wall includes CPU oracle/copies)",
            percentile(&latencies, 0.5), percentile(&latencies, 0.99), latencies[rounds - 1], busy.as_secs_f64() * 1000.0, wall.elapsed().as_secs_f64() * 1000.0);
        // Useful work counts the live columns only (padding columns are computed but not useful).
        let ops = 2.0 * (ROWS * live * kdim) as f64;
        let tops = |ms: f64| ops / (ms * 1e-3) / 1e12;
        let (busy50, wall50) = (percentile(&busies, 0.5), percentile(&latencies, 0.5));
        println!("RATE shape={shape:?} m={ROWS} n_live={live} n_padded={} k={kdim} corun={corun} busy_p50={busy50:.3} ms busy_TOPS={:.2} kick->done_p50={wall50:.3} ms wall_TOPS={:.2} (measured, live columns)",
            shape.pitch(), tops(busy50), tops(wall50));
        // Negative controls run only AFTER every exact round has reported.
        // This generation was never submitted; Shared::wait rejects it in
        // process, without a DRM wait, relay request or GPU wait.
        let invalid_start = Instant::now();
        let unsubmitted = Seq {
            bank: 0,
            value: last_seq[0].unwrap().value + 1,
        };
        match runtime.wait_host(unsubmitted) {
            Ok(_) => {
                return at(
                    None,
                    "negative-unsubmitted-seq",
                    Err("unsubmitted sequence unexpectedly succeeded".into()),
                )
            }
            Err(e) if e == "unknown/stale sequence" => println!(
                "negative unsubmitted-seq: error ({e}), {:.3} ms",
                invalid_start.elapsed().as_secs_f64() * 1000.0
            ),
            Err(e) => {
                return at(
                    None,
                    "negative-unsubmitted-seq",
                    Err(format!("unexpected error: {e}")),
                )
            }
        }
        // Deterministic missing-wait negative: read freshly poisoned C without
        // submitting a replacement NPU job. A racing read could finish after
        // NPU completion and accidentally pass, so it is not used here.
        let cv = at(None, "negative-C-arena", runtime.c_arena(0, shape))?;
        at(
            None,
            "negative-poison-C",
            gpu_copy(cv.device, poison.ptr(), c_bytes),
        )?;
        at(
            None,
            "negative-GPU-read-C",
            gpu_copy(output.ptr(), cv.device, c_bytes),
        )?;
        at(None, "negative-retire-bank", unsafe {
            runtime.retire_bank(last_seq[0].unwrap(), stream.raw())
        })?;
        at(None, "negative-consumer-completion", drain())?;
        at(
            None,
            "negative-download-C",
            hip.download(&output, &mut observed),
        )?;
        if observed.iter().any(|&b| b != POISON) {
            return at(
                None,
                "negative-missing-wait",
                Err("poison negative failed".into()),
            );
        }
        if reference
            .iter()
            .all(|&x| x == u16::from_le_bytes([POISON; 2]))
        {
            return at(
                None,
                "negative-missing-wait",
                Err("negative not distinguishable from reference".into()),
            );
        }
        println!(
            "negative missing-wait: detected poisoned C without replacement submission (all bytes)"
        );
        println!("EXACT_CONTROLS shape={shape:?} rounds={rounds} negatives=2 consumer_padding_bounds=checked");
        // Every round drained the producer/consumer and optional co-run stream;
        // the poison control drained the final consumer as well.
        hip.allow_cleanup();
        Ok(())
    })();
    let teardown = at(None, "teardown", runtime.close());
    let cleanup = at(
        None,
        "GPU-resource-teardown",
        serving.as_ref().map_or(Ok(()), gpu::Serving::check_cleanup),
    );
    outcome.and(teardown).and(cleanup)?;
    // A separate runtime is intentional: a real timeout permanently poisons the
    // relay, and can never be followed by success on that same transport.
    watchdog_negative(pci, shape, design.pdi, design.insts, serving.as_ref())?;
    println!("PASS shape={shape:?} rounds={rounds} negatives=3 consumer_padding_bounds=checked");
    Ok(())
}
fn watchdog_negative(
    pci: &str,
    shape: Shape,
    pdi: Vec<u8>,
    txn: Vec<u8>,
    serving: Option<&gpu::Serving>,
) -> Result<(), String> {
    let layout = railgun::npu::trunk::ArenaLayout::new(ROWS)?;
    // Valid zero operands/program, but a deliberately impossible chain budget:
    // EXEC_CMD must return before its 1 ns deadline can be checked. This exercises
    // production watchdog cancellation, not an environment-dependent test shim.
    let panels = (0..shape.panels())
        .map(|_| vec![0; layout.b_bytes_for(shape)])
        .collect();
    let runtime = at(
        None,
        "negative-timeout-open",
        NpuTrunkRuntime::open_with_options(
            vec![PanelProgram {
                shape,
                pdi,
                txn,
                padded_rows: ROWS,
                pitch: shape.pitch(),
            }],
            vec![LayerPanels {
                layer: 0,
                shape,
                panels,
            }],
            1,
            ROWS,
            pci,
            OpenOptions {
                watchdog: Duration::from_nanos(1),
                cpu_reads: true,
                ..OpenOptions::default()
            },
        ),
    )?;
    let outcome = (|| {
        let hip = at(
            None,
            "negative-timeout-GPU-runtime-identity",
            gpu::Gpu::new(runtime.hip(), serving),
        )?;
        let stream = at(None, "negative-timeout-stream", hip.stream())?;
        let resumed = at(None, "negative-timeout-resume-event", hip.system_event())?;
        let start = Instant::now();
        let seq = at(None, "negative-timeout-kick", unsafe {
            runtime.kick(0, shape, 0, ROWS, stream.raw())
        })?;
        at(None, "negative-timeout-wait-enqueue", unsafe {
            runtime.wait_gpu(seq, stream.raw())
        })?;
        at(
            None,
            "negative-timeout-resume-record",
            hip.record_on(&resumed, &stream),
        )?;
        // Record retirement even when a very fast timeout is already visible.
        let retirement = unsafe { runtime.retire_bank(seq, stream.raw()) };
        at(
            None,
            "negative-timeout-GPU-released",
            hip.wait_event_timeout(&resumed, CHAIN_WATCHDOG),
        )?;
        let elapsed = start.elapsed();
        if elapsed > CHAIN_WATCHDOG {
            return Err(format!(
                "timeout GPU release exceeded watchdog: {elapsed:?}"
            ));
        }
        let error = runtime
            .wait_host(seq)
            .err()
            .ok_or("forced watchdog unexpectedly succeeded")?;
        if !error.contains("watchdog") {
            return Err(format!("unexpected forced-timeout error: {error}"));
        }
        let sticky = runtime
            .check_error()
            .err()
            .ok_or("timeout did not poison runtime")?;
        if !sticky.contains("watchdog") {
            return Err(format!("missing sticky watchdog error: {sticky}"));
        }
        if let Err(e) = retirement {
            if !e.contains("watchdog") {
                return Err(format!("unexpected retirement error: {e}"));
            }
        }
        let retire_error = unsafe { runtime.retire_bank(seq, stream.raw()) }
            .err()
            .ok_or("retirement did not surface terminal timeout")?;
        if !retire_error.contains("watchdog") {
            return Err(format!("missing retirement watchdog error: {retire_error}"));
        }
        // Cover the retirement event too, which follows the earlier resume
        // marker. Uncertain completion quarantines serving handles on exit.
        at(
            None,
            "negative-timeout-final-record",
            hip.record_on(&resumed, &stream),
        )?;
        at(
            None,
            "negative-timeout-final-drain",
            hip.wait_event_timeout(&resumed, CHAIN_WATCHDOG),
        )?;
        hip.allow_cleanup();
        println!("negative forced-NPU-watchdog: GPU resumed in {:.3} ms (limit {:.3} ms); surfaced error ({error}); failed C never read", elapsed.as_secs_f64()*1000.0, CHAIN_WATCHDOG.as_secs_f64()*1000.0);
        Ok(())
    })();
    // A failed runtime reports the sticky job error at checked teardown, too.
    let teardown = runtime.close();
    let cleanup = at(
        None,
        "negative-timeout-GPU-resource-teardown",
        serving.map_or(Ok(()), gpu::Serving::check_cleanup),
    );
    outcome.and(cleanup)?;
    match teardown {
        Err(e) if e.contains("watchdog") => Ok(()),
        Err(e) => Err(format!("negative-timeout teardown: {e}")),
        Ok(()) => Err("negative-timeout teardown lost sticky error".into()),
    }
}
/// Independent class reference of panel `panel` at `round` for a K-deep operand: `[(r_class * 16 + n_class) * 3 + row % 3]`
/// holds `bf16_direct` of the exact whole-K int32 dot product.
fn class_reference(kdim: usize, panel: usize, round: usize) -> Vec<u16> {
    let mut reference = vec![0u16; CLASSES * CLASSES * 3];
    for r in 0..CLASSES {
        for n in 0..CLASSES {
            let dot: i32 = (0..kdim)
                .map(|k| i32::from(a_code(r, k, round)) * i32::from(b_code(panel, k, n)))
                .sum();
            let (sm, se) = sm(panel, n);
            for rem in 0..3 {
                reference[(r * CLASSES + n) * 3 + rem] =
                    bf16_direct(dot, sm, se, dm(r, round).0, -20 + rem as i16);
            }
        }
    }
    reference
}

/// One hardware context with one design's buffers. Field order is the teardown order: the context before its BOs.
struct Side<'d> {
    ctx: HwCtx<'d>,
    _pdi: Bo,
    insts: Bo,
    bos: Vec<Bo>,
    cmd: Bo,
    rows: usize,
    pitch: usize,
    pn: usize,
    kdim: usize,
}

/// Create + configure one context for `design` (K-deep periodic operands, panel 0): `(side, create_ms, config_cu_ms)`.
fn side<'d>(
    dev: &'d Device,
    design: &peacemaker_npu::kernels::gemm_array::ArrayDesign,
    kdim: usize,
    pn: usize,
    pitch: usize,
    rows: usize,
) -> Result<(Side<'d>, f64, f64), String> {
    let t = Instant::now();
    let ctx = HwCtx::create(dev, 32, 1)?;
    let create_ms = t.elapsed().as_secs_f64() * 1e3;
    let pdi = dev.dev_bo(&design.pdi)?;
    let t = Instant::now();
    ctx.config_cu(&pdi)?;
    let config_ms = t.elapsed().as_secs_f64() * 1e3;
    let insts = dev.dev_bo(&design.insts)?;
    let a: Vec<i8> = (0..rows * kdim)
        .map(|i| a_code(i / kdim, i % kdim, 0))
        .collect();
    let scales: Vec<_> = (0..rows).map(|r| dm(r, 0)).collect();
    let b: Vec<i8> = (0..kdim)
        .flat_map(|k| (0..pn).map(move |n| b_code(0, k, n)))
        .collect();
    let sscales: Vec<_> = (0..pn).map(|n| sm(0, n)).collect();
    let inputs = [
        gemm_g80::pack_a_bf16(design, &a, &scales),
        gemm_g80::pack_b_bf16(design, &b, &sscales),
    ];
    let mut bos = Vec::new();
    for (i, spec) in design.args.iter().enumerate() {
        let mut bo = dev.shmem_bo(spec.bytes)?;
        if i < 2 {
            bo.as_mut_slice().copy_from_slice(&inputs[i]);
        } else {
            bo.as_mut_slice().fill(POISON);
        }
        bo.flush();
        bos.push(bo);
    }
    let cmd = dev.cmd_bo()?;
    Ok((
        Side {
            ctx,
            _pdi: pdi,
            insts,
            bos,
            cmd,
            rows,
            pitch,
            pn,
            kdim,
        },
        create_ms,
        config_ms,
    ))
}

impl Side<'_> {
    /// Poison C, submit, wait: `(submit -> completion ms)`. `check` compares every element with the class reference.
    fn run(&mut self, check: bool) -> Result<f64, String> {
        {
            let c = &mut self.bos[2];
            c.as_mut_slice().fill(POISON);
            c.flush();
        }
        let refs: Vec<&Bo> = self.bos.iter().collect();
        let t = Instant::now();
        let seq = self.ctx.submit(&mut self.cmd, &self.insts, &refs)?;
        let state = self.ctx.wait(&self.cmd, seq, 5000)?;
        let ms = t.elapsed().as_secs_f64() * 1e3;
        if state != ERT_STATE_COMPLETED {
            return Err(format!("chain state {state}"));
        }
        if check {
            self.bos[2].flush();
            let reference = class_reference(self.kdim, 0, 0);
            let c = self.bos[2].as_slice();
            for row in 0..self.rows {
                for col in 0..self.pn {
                    let off = 2 * (row * self.pitch + col);
                    let actual = u16::from_le_bytes([c[off], c[off + 1]]);
                    let expected =
                        reference[((row % CLASSES) * CLASSES + col % CLASSES) * 3 + row % 3];
                    if actual != expected {
                        return Err(format!(
                            "row {row} col {col}: {actual:04x} != {expected:04x}"
                        ));
                    }
                }
            }
        }
        Ok(ms)
    }
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

/// H3 second-context probe. Creates the input-shape context (K2560 N1280, 512 rows) and, while it is alive, the out-shape
/// context (K6144 N512, 512 rows): reports `create_hwctx` / `config_cu` cost of each, then submit-to-completion of each
/// design repeated on its own context against the same designs alternating between the two contexts (the context-switch
/// cost), every run checked element-exact. A failed second context creation is the "no coexistence" kill.
fn coexistence(reps: usize) -> Result<(), String> {
    let mut dev = Device::open()?;
    dev.map_heap(64 << 20)?;
    let rows = 512;
    let d_in = at(
        None,
        "build-in",
        gemm_g80::build_panel_program_rowmajor(rows, PANEL_N, 0),
    )?;
    let d_out = at(
        None,
        "build-out",
        gemm_g80_n512::build_panel_program_n512(rows, OUT_PANEL_N, 0, OUT_PANEL_K),
    )?;
    let (mut a, create_in, config_in) = at(
        None,
        "context-in",
        side(&dev, &d_in, PANEL_K, PANEL_N, PANEL_N, rows),
    )?;
    println!(
        "COEXIST ctx_in create_hwctx={create_in:.3} ms config_cu={config_in:.3} ms pdi={} B",
        d_in.pdi.len()
    );
    let first_in = at(None, "run-in-first", a.run(true))?;
    let mut same_in: Vec<f64> = (0..reps)
        .map(|_| a.run(false))
        .collect::<Result<_, _>>()
        .map_err(|e| format!("stage=run-in-same: {e}"))?;
    let second = side(&dev, &d_out, OUT_PANEL_K, OUT_PANEL_N, OUT_PANEL_N, rows);
    let (mut b, create_out, config_out) = match second {
        Ok(s) => s,
        Err(e) => {
            println!("COEXIST ctx_out FAILED while ctx_in is alive: {e}");
            println!("H3_KILL no-coexistence: the out-shape HwCtx cannot be created next to the input-shape HwCtx");
            return Err(format!("stage=context-out: {e}"));
        }
    };
    println!("COEXIST ctx_out create_hwctx={create_out:.3} ms config_cu={config_out:.3} ms pdi={} B (created with ctx_in alive)", d_out.pdi.len());
    let first_out = at(None, "run-out-first", b.run(true))?;
    let mut same_out: Vec<f64> = (0..reps)
        .map(|_| b.run(false))
        .collect::<Result<_, _>>()
        .map_err(|e| format!("stage=run-out-same: {e}"))?;
    let (mut alt_in, mut alt_out) = (Vec::new(), Vec::new());
    for round in 0..reps {
        let check = round + 1 == reps;
        alt_in.push(at(Some(round), "alternate-in", a.run(check))?);
        alt_out.push(at(Some(round), "alternate-out", b.run(check))?);
    }
    let (si, so, ai, ao) = (
        median(&mut same_in),
        median(&mut same_out),
        median(&mut alt_in),
        median(&mut alt_out),
    );
    let switch = (ai - si).max(ao - so);
    println!("COEXIST first submit in={first_in:.3} ms out={first_out:.3} ms (exact)");
    println!("COEXIST p50 submit->done same-context in={si:.3} out={so:.3} ms; alternating in={ai:.3} out={ao:.3} ms; context-switch cost in={:.3} out={:.3} ms (reps={reps}, exact after alternation)", ai - si, ao - so);
    let config = config_in.max(config_out);
    let kill = config > 2.0 || switch > 2.0;
    println!("H3_COEXIST create_hwctx_ms={create_in:.3}/{create_out:.3} config_cu_ms={config_in:.3}/{config_out:.3} switch_ms={switch:.3} kill_threshold_ms=2.0 {}",
        if kill { "H3_KILL config_cu-or-switch>2ms" } else { "ok" });
    Ok(())
}

const USAGE: &str = "npu-trunk-transport [--pci 0000:c0:00.0] [--shape gdn_in|qsa_in|gdn_out|qsa_o|dense_gdn_in|dense_fa_in]... [--rounds N] [--corun] [--gpu-runtime relay|serving] [--coexist [--reps N]]\n\
Default (no --shape/--coexist): 8192 rows, row-major GdnIn + QsaIn (13/11 panels), 2 banks, 20 enqueue-ahead exact rounds/shape FIRST; negatives AFTER include forced chain timeout and bounded GPU release. Prints npu_done->GPU resume event latency; 3 s ready waits, 300 ms chain watchdog, default shim AXI.\n\
--shape selects shapes (repeatable): gdn_out / qsa_o are the K=6144 out-projection programs (N512 panels, 5 panels/chain, own PDI); dense_gdn_in / dense_fa_in are the qwen3.8-27b K=5120 in-projections (N512 panels, 33 / 28 panels/chain, own PDI). Oracle = whole-K int32 sum + bf16_direct, exact on both banks every round.\n\
--corun: queue 16 GiB of GPU copies per round concurrently with every NPU chain (co-run rate).\n\
--gpu-runtime: relay (default) uses railgun's GPU resources; serving loads hip-bridge first and uses its streams, kernels, allocations and events for producers/consumers, including co-run and watchdog controls. Different HIP implementations refuse before any cross-runtime work.\n\
--coexist: second-HwCtx probe (create + config_cu cost, context-switch cost, no transport/HIP).";
fn usage(why: &str) -> ! {
    eprintln!("FAIL: stage=arguments round=none: {why}\n{USAGE}");
    std::process::exit(1)
}
fn main() {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let mut backend = gpu::Backend::default();
    let (mut pci, mut shapes, mut rounds, mut coexist, mut reps, mut corun) = (
        "0000:c0:00.0".to_string(),
        Vec::new(),
        DEFAULT_ROUNDS,
        false,
        10usize,
        false,
    );
    let mut it = args.iter();
    while let Some(flag) = it.next() {
        let mut value = |name: &str| {
            it.next()
                .cloned()
                .unwrap_or_else(|| usage(&format!("{name} needs a value")))
        };
        match flag.as_str() {
            "--help" => {
                println!("{USAGE}");
                return;
            }
            "--pci" => pci = value("--pci"),
            "--gpu-runtime" => {
                backend = gpu::Backend::parse(&value("--gpu-runtime")).unwrap_or_else(|e| usage(&e))
            }
            "--rounds" => {
                rounds = value("--rounds")
                    .parse()
                    .ok()
                    .filter(|&r| r >= 2)
                    .unwrap_or_else(|| usage("--rounds needs an integer >= 2 (both banks)"))
            }
            "--reps" => {
                reps = value("--reps")
                    .parse()
                    .ok()
                    .filter(|&r| r >= 1)
                    .unwrap_or_else(|| usage("--reps needs an integer >= 1"))
            }
            "--coexist" => coexist = true,
            "--corun" => corun = true,
            "--shape" => shapes.push(match value("--shape").as_str() {
                "gdn_in" => Shape::GdnIn,
                "qsa_in" => Shape::QsaIn,
                "gdn_out" => Shape::GdnOut,
                "qsa_o" => Shape::QsaO,
                "dense_gdn_in" => Shape::DenseGdnIn,
                "dense_fa_in" => Shape::DenseFaIn,
                other => usage(&format!("unknown shape {other:?}")),
            }),
            other => usage(&format!("unknown argument {other:?}")),
        }
    }
    if shapes.is_empty() && !coexist {
        shapes = vec![Shape::GdnIn, Shape::QsaIn];
    }
    if shapes.is_empty() && backend == gpu::Backend::Serving {
        usage("--gpu-runtime serving requires transport shapes; --coexist alone does not exercise HIP");
    }
    for shape in shapes {
        if let Err(e) = run(&pci, shape, rounds, corun, backend) {
            eprintln!("FAIL shape={shape:?}: {e}");
            std::process::exit(1);
        }
    }
    if coexist {
        if let Err(e) = coexistence(reps) {
            eprintln!("FAIL coexistence: {e}");
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn resume_event_clock_conversion_preserves_signed_latency() {
        assert_eq!(resume_latency_ns(1_000_000, 2.0, 2_900_000), 100_000);
        assert_eq!(resume_latency_ns(1_000_000, 2.0, 3_100_000), -100_000);
    }

    #[test]
    fn consumer_windows_exclude_panel_padding() {
        for shape in [Shape::GdnIn, Shape::QsaIn] {
            let columns = consumer_columns(shape);
            assert_eq!(columns[0].start, 0);
            for pair in columns.windows(2) {
                assert_eq!(pair[0].end, pair[1].start);
            }
            let live = columns[3].end;
            assert!(live < shape.pitch());
            for window in columns {
                for row in [0, ROWS - 1] {
                    let end = 2 * (row * shape.pitch() + window.end);
                    assert!(end <= 2 * (row * shape.pitch() + live));
                }
            }
        }
    }

    #[test]
    fn out_shapes_fill_the_arena_exactly_and_use_k6144() {
        for shape in [Shape::GdnOut, Shape::QsaO] {
            let columns = consumer_columns(shape);
            assert_eq!((columns[0].start, columns[3].end), (0, shape.pitch()));
            for pair in columns.windows(2) {
                assert_eq!(pair[0].end, pair[1].start);
            }
            assert_eq!((shape.k(), shape.panel_n(), shape.panels()), (6144, 512, 5));
        }
        assert_eq!((Shape::GdnIn.k(), Shape::QsaIn.k()), (2560, 2560));
        // 27B dense in-projections: K5120, 16480 live of 16896 (GDN) and 14336 of 14336 (FA).
        for (shape, live) in [(Shape::DenseGdnIn, 16480), (Shape::DenseFaIn, 14336)] {
            assert_eq!(
                (shape.k(), consumer_columns(shape).last().unwrap().end),
                (5120, live)
            );
        }
    }

    #[test]
    fn class_reference_is_the_whole_k_twin_of_one_class_pair() {
        // Whole-K dot of class (r, n), twin per row % 3 exponent: recompute one entry independently.
        for kdim in [2560usize, 5120, 6144] {
            let reference = class_reference(kdim, 2, 3);
            let (r, n, rem) = (5usize, 9usize, 1usize);
            let mut dot = 0i32;
            for k in 0..kdim {
                dot += i32::from(a_code(r, k, 3)) * i32::from(b_code(2, k, n));
            }
            let (sm, se) = sm(2, n);
            assert_eq!(
                reference[(r * CLASSES + n) * 3 + rem],
                bf16_direct(dot, sm, se, dm(r, 3).0, -20 + rem as i16)
            );
        }
    }
}
