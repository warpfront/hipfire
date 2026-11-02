# Native GPU–NPU prefill/decode investigation — 2026-10-21

This is a local Strix Halo investigation, not a model admission or a throughput floor.
Worktree: `/home/husrcf/Code/hipfire-native-pd-20261021`, branch
`experiment/halo-native-pd-20261021`, base `fbcd2a2ffdee5fb648c03f15fca80f83f05ee374`.
The previous full-drafter resident experiment was stopped and archived separately;
it is not part of this change. No push is authorized.

## Fixture and evidence

Hardware: Nahida, gfx1151, GPU PCI `0000:c5:00.0`, HIP 7.16, kernel
`7.0.0-38-generic`, NPU firmware `1.1.2.65`. Model:
`/home/husrcf/.hipfire/models/qwen3.8-flash-next.mq4-xts` (125288544792 bytes),
with its existing Q4 PLE sidecar and admitted `gfx1151-zab-v1` trunk overlay.
The runtime log resolves QSA KV to q8/VMM and GDN recurrent state to F32.
After all hardware timing, the model was freshly hashed: MD5
`be007fc3219e9f6cdb1d4dfa8380625f`, SHA-256
`8b15b6fede7d7c5bfed0db4720a8295bedda51bc93e545fa242bd50d0f200972`.
The adjacent compressed receipt records full model/PLE/overlay MD5 and SHA-256,
size and mtime pins; reuse them with matching file metadata on subsequent cycles.
The expert hot-set sidecar is present but not applied on this arch by default.
Expert placement is fully GPU resident. NPU trunk is `auto`, all input families
armed, direct BF16 results in the existing shared arenas; output projections
and decode remain on the GPU. No shared-memory interface was newly introduced.

Artifacts on Nahida are under the worktree's
`.codeinsight+research/halo-native-pd-20261021/` (called `R` below).
Every run has a fresh directory. Baseline and candidate executables are frozen
under `R/{baseline,candidate}-binaries-v1/`; validation command records contain
their MD5 and SHA-256, source patch, exact prompt identity, and environment.

## Profiling

`R/baseline-long-auto-trace-v1` is the formal serve path, AR, q8 KV, VMM,
max sequence 32768, original committed long-code prompt
`benchmarks/prompts/qwen38_issue693_longcode_20676.txt`, MD5
`b4d0b63cddcac872648ddf3cdd92cac2`. The actual prompt was 21582 tokens;
21573 were uncached in this run. Prefill was 10035.4 ms and decode 31.7 tok/s.
The 512-token answer was coherent but truncated: this run is diagnostic evidence,
not a generation quality pass.

The 96 bank/layer trace rows sum to 4881.4 ms of NPU chains, 5934.4 ms of GPU
layer remainders, 631.5 ms of preparation, 34.7 ms of packing, and 1.4 ms of
GPU wait. GPU layer remainder includes Core, output projection, HC and MoE;
it is not an attention-only measurement. The runtime's pair sample reports
4881.7 ms NPU, 6482.6 ms overlap window and 0.9 ms exposed wait. These are
different counters with different boundaries. They establish that NPU computation
is largely hidden on this long-input fixture.

`R/baseline-prefill-ipc-rocprof-v2` uses the same frozen daemon through ordinary
JSON IPC, normal stdin EOF shutdown and no HSA lifetime shim. The load ACK
confirms q8/VMM, sequence 32768, MTP off. The request has 21582 prompt tokens
and one output token, intentionally isolating prefill; it finished with `length`
after correct commit negotiation. Its instrumented prefill was 9971.5 ms.
Main GPU kernel duration sums (not end-to-end savings):

| Kernel | Sum, ms |
|---|---:|
| MoE symmetric IU4 gate/up + SiLU | 1325.6 |
| gathered indexed attention WMMA | 1080.1 |
| MQ6 output projection + HC write | 965.5 |
| MoE symmetric IU4 down | 891.3 |
| F16 WMMA `160_64_32_64_k64_p` | 741.1 |
| HC read up + mix | 609.3 |
| HC row fold + next norm/gates | 599.2 |

The initial short decode trace (`baseline-decode-ipc-rocprof-v1`) computed all
43 prime-answer tokens, but its diagnostic client incorrectly waited for `done`
before acknowledging `commit_ready`. The terminal transaction was cancelled;
do not count it as a completed generation quality pass. It still supplied kernel
attribution. Its 355.5 ms of MQ6 GEMV consists of 93.5 ms vocabulary-sized
launches, 142.3 ms at M=10240 and 119.6 ms at M=2560. MoE gate/up and down
add 175.0 and 96.7 ms. Loader symmetry-check kernels are excluded from decode
hotspot interpretation. A corrected client acknowledges `commit_ready` first.

### Profiler failures retained

The serve wrapper's SIGKILL cleanup lost profiler output. Optional SIGINT
cleanup reached an SDK `hsa_amd_profiling_get_dispatch_time` failure after HSA
shutdown. A profiling-only extra HSA init reference then hit a shutdown SIGSEGV.
These attempts are diagnostic failures, not timing or correctness evidence.
A minimal normal-exit HIP program produced CSV with and without the reference;
ordinary daemon stdin EOF then produced CSV successfully. This narrows the
problem to the shutdown path; it does not establish the SDK's root cause.
No production shutdown, driver, firmware or HSA behavior was changed.

## Rejected kernel experiments

These are deterministic synthetic microbenchmarks, with alternating order,
per-cell warmups, three rounds, raw values retained, and all output elements
compared against an independently compiled original body. They are not model
speed claims. All compared outputs were byte-identical and finite.

* `R/mq6-workgroup-sweep-v1`: nine workgroup/load configurations over nine
  shapes, including M=248320/K=2560 and ragged M/K group tails. Packing 2/4/8
  independent row waves per workgroup did not provide a useful stable gain.
  The vocabulary shape's best point was about 0.6% below the original's
  2094.2 us; smaller shapes were mostly slower. Three aligned 16-bit payload
  loads instead of six byte loads were also generally slower. All rejected.
* `R/hc-batch-sweep-v1`: HC gate-weight prefetch batch 10 versus 1/2/4/5,
  rows 128/1024/8192. At 8192 rows: 4528.8 / 5802.6 / 5458.6 / 4569.8 /
  4555.1 us respectively. Lower register pressure alone did not help. All
  rejected; the original batch stays 10.
* `R/hc-batch-sweep-v2`: reload BF16 values from LDS after RMS reduction
  instead of retaining the raw array, and/or finish the unchanged reduction
  tree with wave shuffles. At 8192 rows: original 4532.6 us, reload 4585.0,
  reload/batch5 4572.3, shuffle 4523.9, both 4574.4. No useful stable gain;
  all rejected. Streams, normalized F16 rows and gate logits were compared
  separately and all matched.

The tuning hypotheses were checked against AMD's
[RDNA performance guide](https://gpuopen.com/learn/rdna-performance-guide/) and
[occupancy explanation](https://gpuopen.com/learn/occupancy-explained/).
Those explain workgroup/occupancy tradeoffs; they do not predict that any of
these configurations must win. No CodeGraph tool, executable or index was
available in this checkout; focused source inspection was used.

## Adopted work removal

`NpuTrunkRuntime::prepare_bank` waits for the previous NPU generation and queues
the GPU retirement dependency before setting `prepared`. `kick_panels` formerly
repeated the same host completion query. It now skips that second query only
when `prepared` is true. An unprepared kick still checks GPU retirement and
waits for host completion. Sticky errors are still checked at kick entry;
retired ownership, sequence exhaustion and pending state remain checked; every
new kick clears `prepared`. This preserves the existing buffer lifetime and
terminal-failure contract while removing one mutex/completion lookup per
prepared bank reuse. It does not remove GPU retirement or ready/acquire fences.

The adoption criterion is proved work removal plus correctness/lifetime
validation. No measurable speedup is assumed for this small change.

## Validation and timing

CPU NPU trunk tests: 19 passed. Release build and full
`scripts/pre-commit-ci.sh origin/beta` passed. No GPU kernel or dispatch geometry
change from the rejected experiments is included.

The source change only removes a host runtime completion query. Serve is the
applicable lifetime/semantic gate; the discarded standalone kernel variants do
not change the product and are not promoted through a replay kernel gate.

### Matched native bench (NPU `auto`, speculation `off`)

`validation-01..04-*-bench-v1`, fresh processes in declared A/B/B/A order,
one matching throwaway request then one measurement per process, same original
committed prompt MD5, 21582 prompt tokens, 128 output budget, q8/VMM,
sequence 32768, greedy, DPM warmup 10 s, session/disk caches disabled.
The path is native `hipfire bench --backend noslots --workload stateless`;
there is no profiler attached. Each command record pins the actual daemon.
Directory `off` here denotes **speculation off**, not NPU off.

| Order | Binary | TTFT, ms | AR decode, tok/s | Fast/slow PPT residency, % |
|---|---|---:|---:|---:|
| A | baseline | 9689.0 | 33.7 | 30.5 / 45.8 |
| B | candidate | 9673.5 | 33.8 | 29.8 / 48.9 |
| B | candidate | 9691.3 | 33.8 | 29.8 / 50.4 |
| A | baseline | 9678.2 | 33.7 | 29.3 / 49.5 |

Median TTFT is 9683.6 -> 9682.4 ms; decode is 33.7 -> 33.8 tok/s.
These are effectively unchanged, not an established speedup. All arms were
power limited (no thermal throttle recorded); full power/clock/memory fields
are retained. Automatic PLE prewarm budgets also varied with free RAM. The
matching throwaway request warms the same prompt's lookups, but this remains a
limitation of any finer-than-noise timing interpretation. Adoption is justified
by work removal and correctness, not this point estimate.

### Serve path proof limitation

`validation-05-baseline-off-serve-v1` stopped **before any prompt** because
`serve_harness.py` does not recognize Qwen4's native VMM marker for an explicit
`--kv-backend vmm` request. The Qwen4 log says `QSA q8 ... (vmm context storage)`
and `QSA context: vmm storage`; the IPC loaded ACK independently reports VMM.
The harness was not modified. Fresh `validation-05..08-*-serve-v2` runs omit
the explicit allocator argument and use the same automatic VMM result, checked
from their Qwen4 logs. The harness reports allocator/KV mode UNOBSERVED; do not
misrepresent that as an observed loaded ACK from the harness.

Their prompts are the original committed long-code bytes plus the exact literal
suffix in `R/prepare_validation.py` requesting a <=80-word summary, and the
committed prime-number prompt. Both arms use byte-identical materialized prompt
batteries (long prompt MD5 `062a0fea1fd67af2c2322830fb52fe0b`, 21607 tokens;
short MD5 `0508eec29a44323f62e70fa77d92b834`, 24 tokens). Greedy, thinking off,
512 output cap, sequence 32768, q8 KV. Candidate uses `--compare-transcript`
against the corresponding original binary in both AR and MTP.

All four serve processes and eight turns passed: 97-token long summaries and
43-token prime lists, all `finish=stop`, no attractor/empty/runaway. The decoded
text was read. Both AR and native MTP candidate transcripts are byte-identical
to their original-binary controls (`91b4a733973456d5657d3b30218bd1d8`). MTP was
observed active on both turns (`mtp=true`); these are not AR fallback tests.
MTP's per-turn rates are semantics-check context, not a claimed speedup.

### Cold-tail outlier investigation

The candidate AR serve long turn took 11114.0 ms against the original's
9740.5 ms: a 14.1% isolated cold-request loss. This was investigated rather than
dismissed. Their measured bank pairs are 7032.0 and 7023.0 ms, so the additional
1.37 s is outside the changed two-bank segment, in the remaining tail/request
work. Its exact cause is not proven; do not label it a known compiler or driver
bug.

`tail-recheck-01..06-*-v1` repeats that exact long-summary prompt, three new
processes per binary, order A B B A A B, one matching throwaway request followed
by one measurement. PLE prewarm and load-fill budgets are fixed at 8 GiB in
both arms; other settings match the original bench. Raw TTFT values:

* Original: 9705.7, 9799.2, 9654.1 ms (median 9705.7).
* Candidate: 9709.0, 9722.2, 9798.6 ms (median 9722.2, +0.17%).
* AR decode: original 33.8/33.8/33.9, candidate 33.8/33.8/33.8 tok/s.

The >=5% loss did not persist after matching-shape warmup and budget control.
This closes the stable-regression check, not the exact cold-outlier root cause.
Power limitations remain in all samples; one original sample also records 2.3%
core thermal-limiter residency. No stable speedup is claimed.

### Corrected decode profiler

`baseline-decode-ipc-rocprof-v2` correctly acknowledges `commit_ready`, ends
with `finish_reason=stop`, 43 output tokens, normal EOF shutdown and nonempty
CSV. It independently reproduces the initial hotspot ranking: MQ6 GEMV 355.7
ms (93.7 ms vocabulary-sized, 142.4 ms M=10240, 119.6 ms M=2560), MoE gate/up
175.2 ms, MoE down 96.9 ms. These instrumented durations include the request's
short prefill/head setup; they are attribution evidence, not pure per-token
latency or model throughput. No signal cleanup or HSA shim is involved.

## Next investigation

The priority is GPU MoE and output-projection work in prefill, plus a separate
native-MTP decode profile. Most of the long-input NPU chain is already hidden.
The existing symmetric MoE down row-repeat knob
`HIPFIRE_QWEN4_MOE_SYM_DOWN_RR=1|2|4` is a concrete data-reuse candidate.
Its production kernels are embedded peacemaker builder images, not the hipcc
reference body: `crates/rdna-compute/src/gemm.rs` selects the symbols and
`kernels/qwen4_moe_iu4_sym_pm_gfx1151.hxaco` supplies the image. Do not tune only
the reference HIP text and assume production changed.

`moe-rr-sweep-01-rr2-v1` and `02-rr4-v1` use the same candidate, long-summary
prompt, fixed 8 GiB PLE budgets, fresh process and one matching warmup, changing
only the existing RR override. RR2 measured TTFT 9712.8 ms / AR 33.9 tok/s;
RR4 measured 9823.9 ms / AR 33.9 tok/s. RR1's three candidate controls above have
median 9722.2 ms. These are one-process exploratory points for RR2/RR4, not
promotion evidence or a stable ranking. No useful win was established, so the
shipping RR1 choice is retained. This knob changes prefill, not decode; its
small AR differences are a control, not a decode optimization.

The next pass should profile native MTP independently and investigate the
actual builder MoE gate/up epoch/load schedule against its independent hipcc
anchor; do not repeat the rejected generic workgroup/prefetch sweeps. The
ongoing optimization goal remains active; this checkpoint does not claim that
native PD optimization is complete.
