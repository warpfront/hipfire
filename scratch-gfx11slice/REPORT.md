# gfx1100 pp8192 projection slicing investigation

All measurements here used hipx HIP ordinal 0 (gfx1100), the absolute-path 14,987,185,152-byte Qwen3.8-27B MQ4V2 model, typed C kernel knobs, and explicit q8 KV. `HIPFIRE_GRAPH=1` enables graph support but does not imply the fused producer ran in a graph: `Gpu::iu4_producer_sidecar_active` excludes capture and replay, and the traced fused `_i4` producer symbols establish the eager route in these requests. See `pbs-audit.txt` for the complete per-buffer inventory, consumers, and exact row-byte accounting.

## Cause of the original seven-way split

The original C trace on contiguous KV issued 1,904 LF16 SET and 896 LF16 ADD calls for an 8192 prefill, compared with 272 SET and 128 ADD at 512. Attention and GDN held their expected 512-row recurrent stride; the sevenfold projection and fused producer increase came from admission reducing the projection chunk after the first request, not a gfx1100 LF16 grid/N cap or fixed graph replay. `ordinary_prefill_chunk_limit` (`qwen35/prefill.rs`) projects a widened PBS allocation plus projection scratch and 1 GiB headroom against current free VRAM. A widened PBS retained from the previous request consumes free VRAM, yet without the optional credit the same PBS bytes are charged again to the next request's admission. The actual projection then runs the admitted chunks, while the printed first warm-up receipt (`requested=8192 admitted=8192 commit_stride=512`) is not a receipt for later requests.

## PBS inventory and experimental lean allocation

Full dense tape-free PBS at 8192: **5,942,378,752 B**, 725,388 B/row plus 256 B. The ordinary fused route does not consume four verify-only F16 sidecars, a tree-only `rope_positions`, and full-row versions of five F32 fallback buffers. Default-off typed `kernel.gfx11_lean_pbs=true` caps those fallback buffers at 64 rows, omits the verify/tree HIP allocations, and gates the lean route to exact eligible gfx11 dense MQ4V2 paths before launch. Lean PBS at 8192: **4,040,229,120 B**, saving **1,902,149,632 B**. Absent sidecars have zero-length borrowed sentinels so existing non-lean ownership and aliases remain intact; a lean PBS rejects verify and unsupported widened fallback routes before any chunk kernel.

VMM plus the lean allocation reaches the full 8192-row admission on repeated requests without any admission credit. The experimental `prefill.credit_reusable_scratch` knob and subtraction have therefore been removed; admission conservatively projects the full PBS bytes against free VRAM, even when it may be reused.

## Contiguous backend proof (superseded for product throughput)

With **lean=true, credit=false, contiguous q8**, three cold pp8192 requests in one manual server process had prompt_tokens=8192 and cached_tokens=0 each. Observed peaks were 21,340,028,928 / 21,537,492,992 / 21,537,492,992 B; unprofiled prefill 3,813.5 / 3,805.8 / 3,810.1 ms (2,148.1 / 2,152.5 / 2,150.1 tok/s). The rocprof CSV `lean-trace/daemon_kernel_trace.csv` decisively shows request 1 **272 SET + 128 ADD, gridX=2048** (one 8192 block), but requests 2 and 3 each **544 SET + 256 ADD, gridX=1024** (two 4096 blocks). Producer `_i4` counts correspond 64+128 on request 1 and 128+256 on each subsequent request. Thus lean alone does not eliminate *all* slicing on the old contiguous KV backend, even though its two-block throughput was nearly identical.

## KV backend discovery / ledger

Passing an absolute model path bypasses the `qwen3.8:27b` tag policy, so all earlier serve/bench runs used contiguous KV (`KV cache: q8 (16/64 layers carry KV, others placeholder)`). The product-policy comparator must explicitly pass `--kv-backend vmm` and confirm `KV cache: Q8 vmm` in the serving process log. A VMM-C server was loaded but stopped at Main's request before any prefill request; its own log read: `KV cache: Q8 vmm (16/64 layers carry KV; K 272B/head + V Q8 272B/head; mapped_prefix=1927 / physical_cap=32768 / max_seq=32768)`. Four KV heads and 16 FA layers imply 34,816 bytes per token, **67,090,432 B initial mapped VMM** versus **1,140,850,688 B contiguous at physical cap**: a **1,073,760,256 B** difference at initial load, separate from the 1,902,149,632 B PBS saving. These values are mathematical from the logged geometry, not a live allocator/pool attribution. No VMM C/lean request-level results have been measured here yet.

The `dflash_state_bulk_copy_gfx1100` seen in ordinary spec-off traces is from DeltaNet prefix-cache checkpoint save (`hipfire-generate/src/ar.rs` calling `take_dn_checkpoint`), not evidence of DFlash verification or use of verify PBS sidecars.

## VMM lean three-request grid proof

With **lean=true, credit=false, explicit `--kv-mode q8 --kv-backend vmm`**, the serve log asserted `KV cache: Q8 vmm (... mapped_prefix=1927 / physical_cap=32768 / max_seq=32768)`. Three cold pp8192 requests in one process each had prompt_tokens=8192 and cached_tokens=0. Per-request rocprof (`vmm-lean-trace/daemon_kernel_trace.csv`, timestamps joined to `vmm-lean-{1,2,3}.json`) recorded **272 LF16 SET and 128 LF16 ADD, every call gridX=2048**, for requests 1, 2, and 3. The fused producers were 128 RMSNorm `_i4` plus 64 SiLU `_i4` each request. Observed VRAM peaks were **20,535,128,064 / 20,535,128,064 / 20,581,494,784 B**, and profiled prefill rates **2,135.5 / 2,143.5 / 2,147.5 tok/s**. This directly proves one projection block per repeated request; do not substitute these profiled timings for the unprofiled matrix ship gate.

## Ship gate status

The VMM lean grid proof passed, but the **first ship gate failed**, so odd-fill, WT2, and battery were correctly not run. Fresh-process VMM q8 ABBA using the exact matrix command (`--pp 512,8192 --ctx 128 --tg 128 --spec off --runs 3 --warmups 1`) gave these per-process medians:

| Arm | lean | pp512 tok/s | pp8192 tok/s | tg128@128 tok/s |
|---|---:|---:|---:|---:|
| A1 | false | 2150.9 | 2120.8 | 49.1082 |
| B1 | true | 2133.1 | 2144.2 | 49.1574 |
| B2 | true | 2129.6 | 2138.3 | 49.0714 |
| A2 | false | 2124.6 | 2109.9 | 48.9841 |

Every process's JSON asserted `kv_mode: q8` and stderr asserted `KV cache: Q8 vmm`; no contiguous log appeared. The arm means are C 2137.75 / 2115.35 / 49.0462 versus lean 2131.35 / 2141.25 / 49.1144: **pp512 −0.299%, pp8192 +1.224%, decode +0.139%**. Both lean pp8192 arms beat both C arms; no row regressed >1%, and decode was within 1%. This missed the original ≥1.5% throughput-only bar, but Main explicitly reclassified it as a **defect fix** because it removes 1,902,149,632 B of dead GPU scratch and restores single-block admission on every VMM request. **Verdict: SHIP**, after the exactness, odd-fill, and battery gates below all passed. Raw per-arm data: `vmm-matrix-{A1,B1,B2,A2}.json` and `.stderr`. Main's independent VMM C ABBA baseline was pp512 2,133.3 / pp8192 2,112.8 / decode 49.02 tok/s (`/home/kaden/gt-kvbackend`), consistent with this control.

## Correctness gates after defect-fix throughput override

The VMM q8 TTFT-511 prompt md5 was `7423e8940920082c6fa11576d23bc9a2` with exactly 511 tokens in all four fresh-process arms (`--runs 4 --warmups 2 --spec off`). Every serve stderr asserted `KV cache: Q8 vmm`. Per-process median TTFT: A1 291.2017 ms, B1 291.0168 ms, B2 291.4554 ms, A2 291.5503 ms. The arm means are 291.3760 → 291.2361 ms (**−0.048% latency**), within the 1% no-regression requirement. Raw files: `vmm-ttft-{A1,B1,B2,A2}.json` and `.stderr`.

WT2 q8/q8 prefill-scoring c24 evaluated once per arm with `HIPFIRE_GRAPH=0 HIPFIRE_NORMALIZE_PROMPT=0`, scoring 24,552 tokens each. A (lean=false) admitted 4096; B (lean=true) admitted 8192. Both gave **KLD 0.076879**, mean NLL 1.882926 and PPL 6.5727. The complete `.kldseq` outputs compared **bit-for-bit identical** (`cmp` exit 0; both SHA256 `8f2b94bb904603bc53d17f7877c306ee4aacd0afc7c71acbf80a60bea0b1e2b2`), so changed chunk scheduling did not perturb scored quality. Raw `wt2-{A,B}.kldseq` and `.stderr` are retained locally; the example evaluator was rebuilt from this branch immediately before the run.

Manual serve with explicit `--kv-mode q8 --kv-backend vmm` logged `KV cache: Q8 vmm`, with C kernel knobs, lean=true and speculation off. `scripts/serve_harness.py --no-spawn --mode battery --thinking off --kv q8 --kv-backend vmm --port 11741` returned **5/5** coherent decoded responses (code merge function, distance arithmetic 210 miles, seasons, lighthouse vignette, programming advice), each `finish=stop`, no empty/runaway/attractor/stream error. Raw `vmm-battery.json` contains all five decoded `assistant_content` values and request fingerprints. The owned server and daemon were terminated; pre/post process inspection distinguished the concurrently running foreign gfx1151 evaluator by `/proc/<pid>/environ` and confirmed no gfx1100 process remained.
