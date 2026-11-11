# MTP parity baseline: hipfire EP2 singleton MTP on beta 35461ed399, fixture C1 cells — 2026-11-01

**Lifecycle:** `historical`

**Disposition:** registration of the first hipfire baseline on the `mtp-parity-v1` fixture (see
`2026-11-01-mtp-parity-ratchet-registration.md`, which pre-registered the rule). It is the baseline the
ratchet (`benchmarks/mtp-parity/ratchet.json`) compares against for the 12 C1 cells; it is a measurement under
this exact fixture, binary and route, not a shipped-performance claim. The 16 C>1 cells are un-gated: native-MTP
lanes are not on beta, so no per-position count exists there.

## Fixture and method

- Binary: beta `35461ed399` release (`hipfire` sha256 `af66311b1e197bab4299e91c0b4937459e6b02780b1b3779c832a8eb073c6d60`,
  `daemon` `120fbfc2d76c96847200ef18fd030ed7bbe79798a7baf953e5724c66776c208b`), `hipfire serve --spec mtp --max-seq 16384`,
  EP2 on two R9700 (cards C,B), `qwen3.8-flash-next-base.mq4-xts`, product defaults (singleton native MTP with the
  adaptive depth chooser, K up to 7, n-gram copy lane on), `HIPFIRE_MTP_TRACE=1`, session cache off.
- Fabric job j6268 (class timing, P1), 3 fresh serve processes one after another (one rep each), each driven by
  `scripts/mtp-parity-drive.py --cells c1 --rounds 3 --warm 2`: the 12 C1 cells, 3 requests per cell per rep, max_tokens 256,
  thinking off, no nonce, T=1 = 1.0 / top_p 0.95 / top_k 20 (seed 1000*rep + 100*round + stream), greedy = 0.
  Elapsed 245 / 252 / 247 s per rep. `GPU_MAX_HW_QUEUES` unset (product default queue policy; recorded in `pre.txt`).
- Per-position counts come from the serve's `QWEN4_MTP_TRACE` window events sliced per request
  (`check-mtp-parity-ratchet.py from-driver --trace`): native-head (`source: mtp`) windows only for `a_d`; `yield` is
  committed tokens per window over all windows (n-gram takeover windows included); `window_ms` is the client's
  (wall - first token) divided by the trace's window count, so it includes host overheads and takeover windows and is
  **not** a server step timer. Records: `benchmarks/mtp-parity/receipts/base-beta-35461ed399-c1/` (36 files).
- Definitions: `a_d = accepted_d / reached_d`; `reached_d` = windows that offered >= d drafts and accepted the first d-1. The
  depth chooser and margin prune decide how many drafts a window offers, so `reached_d` is a confidence-selected subset
  and the rates are not an unconditional per-position probability.

## Baseline (3 reps pooled; cv across reps in brackets)

| cell | a1 / a2 / a3 | reached d1 / d2 / d3 | yield (cv) | window ms (cv) |
|---|---|---|---|---|
| story t1 | 0.666 / 0.568 / 0.617 | 1006 / 618 / 180 | 2.26 (2.4%) | 32.1 (2.2%) |
| story greedy | 0.780 / 0.668 / 0.685 | 901 / 666 / 251 | 2.54 (0.5%) | 32.7 (0.5%) |
| code t1 | 1.000 / 0.955 / 0.914 | 112 / 112 / 105 | 6.77 (2.3%) | 48.6 (0.9%) |
| code greedy | 1.000 / 1.000 / 0.917 | 108 / 108 / 108 | 6.97 (0.8%) | 48.7 (0.4%) |
| prose_long t1 | 0.716 / 0.659 / 0.663 | 859 / 587 / 273 | 2.65 (1.4%) | 34.3 (1.9%) |
| prose_long greedy | 0.792 / 0.695 / 0.638 | 897 / 683 / 287 | 2.54 (1.1%) | 32.7 (1.4%) |
| agentic t1 | 0.844 / 0.896 / 0.890 | 493 / 412 / 353 | 4.63 (5.9%) | 48.4 (4.5%) |
| agentic greedy | 0.924 / 0.864 / 0.899 | 537 / 447 / 347 | 4.25 (3.2%) | 43.4 (2.6%) |
| math t1 | 0.891 / 0.872 / 0.846 | 487 / 431 / 364 | 4.67 (4.1%) | 44.0 (3.2%) |
| math greedy | 0.900 / 0.906 / 0.944 | 519 / 467 / 411 | 4.40 (1.5%) | 42.0 (1.1%) |
| multilingual t1 | 0.647 / 0.581 / 0.624 | 1001 / 582 / 133 | 2.16 (5.8%) | 30.8 (2.4%) |
| multilingual greedy | 0.659 / 0.639 / 0.696 | 1031 / 623 / 224 | 2.21 (1.3%) | 31.1 (0.7%) |

Per-process check (Main's HWQ rule: flag any process > 5% off the median): 1 of 12 cells has a rep beyond 5%
(`agentic-C1-t1` window_ms 50.7 vs median 48.2 = +5.1%); no bimodal pattern in the other cells.

## Registered tolerances

From the pre-registered rule (z 3, abs floor 0.02, yield floor 3%, window floor 5%): e.g. story t1 allows `a_1` to fall by
0.063, yield by 0.16 (7%), window ms to rise by 2.1 ms (6.6%); the noisiest cells (agentic t1, multilingual t1, yield cv
5.8-5.9%) allow yield drops of 17-18%. The full per-cell values are in `ratchet.json`.

## Same-bytes comparison with Radiance 1.11.0 stock (their cells: `mgmtp/parity-mtp/radiance/`, depth 3, T=1, 3 reps)

Equal-depth-3 tokens per window from the per-position rates (`1 + a1 + a1 a2 + a1 a2 a3`) against their measured yield; ours is
confidence-selected (see above), theirs always drafts 3.

| prompt (T=1) | hipfire a1/a2/a3 | hipfire depth-3 yield | hipfire measured yield (K<=7) | Radiance a1/a2/a3 | Radiance yield |
|---|---|---|---|---|---|
| story | 0.666/0.568/0.617 | 2.28 | 2.26 | 0.664/0.528/0.444 | 2.17 |
| code | 1.000/0.955/0.914 | 3.83 | 6.77 | 1.000/0.992/0.977 | 3.98 |
| prose_long | 0.716/0.659/0.663 | 2.50 | 2.65 | 0.702/0.677/0.653 | 2.49 |
| math | 0.891/0.872/0.846 | 3.33 | 4.67 | 0.876/0.887/0.834 | 3.30 |
| multilingual | 0.647/0.581/0.624 | 2.26 | 2.16 | 0.551/0.447/0.406 | 1.90 |

Window cost is the gap: 30.8-48.7 ms per window here against their 14.1-16.8 ms engine step at C1 (2.0-3.4x).

## Not covered

C>1 cells (no per-position count on beta); the chooser's selection effect on `a_d`; a server-side window timer (the
singleton trace has no `window_us` on the batched path); a second card pair or a second binary.
