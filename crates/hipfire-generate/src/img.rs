// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Image generation — the `img_generate` wire body.
//!
//! Per-architecture generation body lifted verbatim from
//! `crates/hipfire-daemon/src/main.rs` (imggen port). See `lib.rs` for
//! layering rationale: the daemon dispatches (id parse, no-model and
//! non-diffusion refusals via `hipfire_loader` only) and this module owns
//! everything that names `hipfire_arch_diffusion`.

use base64::Engine;
use hipfire_engine::emit::emit_uncorrelated_error;
use hipfire_loader::LoadedModel;
use std::io::Write;

/// Serve one `img_generate` request: reference-image validation, sampler /
/// backend / geometry fail-closed checks, denoise (GPU default, CPU oracle
/// on explicit `backend: "cpu"`), then exactly one `img_done` or one error
/// envelope.
///
/// Contract (byte-for-byte the daemon's): monotonic `img_progress` 0..steps,
/// exactly one `img_done`; fail-closed validation errors for
/// width/height/steps/seed/sampler/non-diffusion load; the T5-host-fallback
/// path and the `generate_img_prompt_gpu` vs `generate_img_prompt` split are
/// preserved.
pub fn generate_img(
    m: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    stdout: &mut impl Write,
    id: &str,
    req: &serde_json::Value,
) {
    // Reference-image edit: validate `images[]`
    // fully before any GPU work — wrong route, too many images,
    // or an unreadable path must fail closed here, not deep
    // inside the denoise loop.
    let image_paths: Vec<String> = req
        .get("images")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|x| x.as_str().map(str::to_owned))
                .collect()
        })
        .unwrap_or_default();
    if !image_paths.is_empty()
        && hipfire_loader::img_route(m.arch_id) != hipfire_loader::ImgRoute::Flux2
    {
        emit_uncorrelated_error(
            stdout,
            Some(id),
            "reference images need a FLUX.2 Klein pipe (arch 45)",
            "validation",
            false,
            false,
        );
        let _ = stdout.flush();
        return;
    }
    if image_paths.len() > 4 {
        emit_uncorrelated_error(
            stdout,
            Some(id),
            "at most 4 reference images",
            "validation",
            false,
            false,
        );
        let _ = stdout.flush();
        return;
    }
    // Each entry is the image's bytes, base64 (a `data:...;base64,`
    // prefix is tolerated). The daemon never opens a client-named
    // path: that would let any HTTP client read any file the
    // daemon can read.
    let mut references = Vec::with_capacity(image_paths.len());
    let mut ref_err = None;
    for (i, entry) in image_paths.iter().enumerate() {
        let payload = entry
            .rsplit_once(";base64,")
            .map_or(entry.as_str(), |(_, b)| b);
        let max_b64 = hipfire_arch_diffusion::refimg::MAX_REFERENCE_BYTES / 3 * 4 + 4;
        let decoded = if payload.len() > max_b64 {
            Err(format!(
                "images[{i}]: {} base64 chars exceed the {} byte reference cap",
                payload.len(),
                hipfire_arch_diffusion::refimg::MAX_REFERENCE_BYTES
            ))
        } else {
            base64::engine::general_purpose::STANDARD
                .decode(payload)
                .map_err(|e| format!("images[{i}]: not base64 image bytes: {e}"))
                .and_then(|bytes| {
                    hipfire_arch_diffusion::refimg::decode_reference(&bytes)
                        .map_err(|e| format!("images[{i}]: {e}"))
                })
        };
        match decoded {
            Ok(r) => references.push(r),
            Err(e) => {
                ref_err = Some(e);
                break;
            }
        }
    }
    if let Some(e) = ref_err {
        emit_uncorrelated_error(stdout, Some(id), &e, "validation", false, false);
        let _ = stdout.flush();
        return;
    }
    // Hoist the model name before the mutable `flux_pipe_mut`
    // borrow so the denoise path can take `&mut pipe.bundle`.
    let model_name = m.model_path.clone();
    let pipe = match m.flux_pipe_mut() {
        Some(p) => p,
        None => {
            emit_uncorrelated_error(
                stdout,
                Some(id),
                "img_generate refused: arch 40 model carries no flux pipe bundle",
                "internal",
                false,
                false,
            );
            let _ = stdout.flush();
            return;
        }
    };
    // Sampler/scheduler: Flow-Match Euler only.
    // Anything else fails closed rather than silently aliasing
    // to euler.
    let sampler = req
        .get("sampler")
        .or_else(|| req.get("scheduler"))
        .and_then(|v| v.as_str())
        .unwrap_or("euler");
    if !matches!(
        sampler,
        "euler" | "flow-match" | "flow_match_euler" | "flowmatch"
    ) {
        emit_uncorrelated_error(
            stdout,
            Some(id),
            &format!("unsupported sampler {sampler:?}: only euler (flow-match) is supported"),
            "validation",
            false,
            false,
        );
        let _ = stdout.flush();
        return;
    }
    // Guidance-distilled (schnell): no negative prompts, no CFG.
    if req
        .get("negative_prompt")
        .and_then(|v| v.as_str())
        .is_some_and(|s| !s.is_empty())
    {
        emit_uncorrelated_error(
            stdout,
            Some(id),
            "negative_prompt unsupported: guidance-distilled FLUX has no CFG dual pass",
            "validation",
            false,
            false,
        );
        let _ = stdout.flush();
        return;
    }
    // Backend selection: `gpu` (default — fp32 MMDiT forward
    // lifted to HIP, weights upload once on the first gpu
    // request and stay resident for the session) or `cpu`
    // (Phase-1 oracle fallback). No re-probe needed here: `main`
    // already called `rdna_compute::Gpu::init()` unconditionally
    // at startup and exited on failure, so a running daemon
    // always has a GPU — the "cpu" branch is reachable only via
    // an explicit `backend` override below, which always wins.
    let backend = req.get("backend").and_then(|v| v.as_str()).unwrap_or("gpu");
    if !matches!(backend, "cpu" | "gpu") {
        emit_uncorrelated_error(
            stdout,
            Some(id),
            &format!("unsupported backend {backend:?}: expected \"cpu\" or \"gpu\""),
            "validation",
            false,
            false,
        );
        let _ = stdout.flush();
        return;
    }
    let prompt = req.get("prompt").and_then(|v| v.as_str()).unwrap_or("");
    // width/height are `Option<usize>` so a reference-edit
    // request can omit them and default to the reference image's
    // own size (`generate_img_prompt`/`_gpu` resolves that). FLUX.1
    // txt2img keeps its unconditional 1024x1024 default.
    let msg_width = req
        .get("width")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize);
    let msg_height = req
        .get("height")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize);
    let (width, height) = if image_paths.is_empty() {
        (
            Some(msg_width.unwrap_or(1024)),
            Some(msg_height.unwrap_or(1024)),
        )
    } else {
        (msg_width, msg_height)
    };
    let steps = match req.get("steps").and_then(|v| v.as_u64()) {
        // Absent → the architecture default (4 for step-distilled
        // schnell, 28 for guidance-distilled dev). An explicit 0
        // is a client bug, refused like every other bad field.
        None => pipe.bundle.transformer_cfg.default_steps() as usize,
        Some(0) => {
            emit_uncorrelated_error(
                stdout,
                Some(id),
                "steps must be >= 1 (omit it for the architecture default)",
                "validation",
                false,
                false,
            );
            let _ = stdout.flush();
            return;
        }
        Some(s) => s as usize,
    };
    let seed = req.get("seed").and_then(|v| v.as_u64()).unwrap_or(0);
    let started = std::time::Instant::now();
    let mut last_step = 0usize;
    let bundle = &mut pipe.bundle;
    // The GPU path now serves BOTH families and the
    // reference-edit route (FLUX.1 txt2img runs the same code it
    // always did, with an empty reference list). Only an explicit
    // `backend: "cpu"` takes the host oracle, which at real
    // geometry is minutes per step.
    let use_gpu = backend == "gpu";
    let result = if use_gpu {
        if let Err(e) = bundle.ensure_gpu(&mut *gpu) {
            emit_uncorrelated_error(
                stdout,
                Some(id),
                &format!("img_generate gpu upload failed: {e}"),
                "internal",
                false,
                false,
            );
            let _ = stdout.flush();
            return;
        }
        hipfire_arch_diffusion::pipeline::generate_img_prompt_gpu(
            bundle,
            &mut *gpu,
            prompt,
            width,
            height,
            steps,
            seed,
            &references,
            &mut |step, total| {
                // Monotonic 1..=steps progress events, flushed per
                // step so a client can render a live progress bar.
                if step >= last_step {
                    let _ = writeln!(
                        stdout,
                        r#"{{"type":"img_progress","id":"{}","step":{},"total":{}}}"#,
                        id, step, total
                    );
                    let _ = stdout.flush();
                    last_step = step;
                }
            },
        )
    } else {
        hipfire_arch_diffusion::pipeline::generate_img_prompt(
            &pipe.bundle,
            prompt,
            width,
            height,
            steps,
            seed,
            &references,
            &mut |step, total| {
                // Monotonic 1..=steps progress events, flushed per
                // step so a client can render a live progress bar.
                if step >= last_step {
                    let _ = writeln!(
                        stdout,
                        r#"{{"type":"img_progress","id":"{}","step":{},"total":{}}}"#,
                        id, step, total
                    );
                    let _ = stdout.flush();
                    last_step = step;
                }
            },
        )
    };
    // Conditioning-cache state, read after the generation borrow
    // ends. The cache itself lives in the bundle (so it is
    // daemon-lifetime by construction and needs no daemon-side
    // storage); these two fields exist so a client — or the
    // serve harness — can tell a cached prompt from a re-encoded
    // one without timing it. `cond_cached` counts resident
    // `(prompt, t5_seq)` entries; it is 0 whenever
    // `HIPFIRE_IMG_COND_CACHE=0`, which clears after each run.
    let cond_cached = pipe.bundle.cond_cache.len();
    let cond_cache_on = pipe.bundle.cond_cache.is_enabled();
    match result {
        Ok(out) => {
            let png_b64 = base64::engine::general_purpose::STANDARD.encode(&out.png);
            let (ow, oh) = out.image_shape;
            let done = serde_json::json!({
                "type": "img_done",
                "id": id,
                "png_b64": png_b64,
                "model": model_name,
                "seed": seed,
                "width": ow,
                "height": oh,
                "steps": steps,
                "backend": backend,
                "cond_cache": cond_cache_on,
                "cond_cached": cond_cached,
                "references": references.len(),
                "ms": started.elapsed().as_millis() as u64,
            });
            let _ = writeln!(stdout, "{done}");
            let _ = stdout.flush();
        }
        Err(e) => {
            emit_uncorrelated_error(
                stdout,
                Some(id),
                &format!("img_generate failed: {e}"),
                "validation",
                false,
                false,
            );
            let _ = stdout.flush();
        }
    }
}
