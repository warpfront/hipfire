# Native MTP MoE inverse launch — 2026-10-21

Local Halo optimization on `experiment/halo-native-pd-20261021`, parent
`7bcc4188306a5348a9ba5bca4dde3d4a4700baee`. The full NPU drafter direction
remains stopped. The native GPU–NPU prefill/decode goal remains active.
No push. Evidence root on Nahida:
`/home/husrcf/Code/hipfire-native-pd-20261021/.codeinsight+research/halo-native-pd-20261021/mtp-moe-cycle-v1` (`R`).

## Fixture

gfx1151, GPU PCI `0000:c5:00.0`, HIP 7.16. Existing monolithic
`/home/husrcf/.hipfire/models/qwen3.8-flash-next.mq4-xts`, MD5
`be007fc3219e9f6cdb1d4dfa8380625f`, Q4 PLE and `gfx1151-zab-v1` trunk overlay.
Full model/sidecar pins from the previous cycle were reused after checking
size/mtime; the 125 GB model was not rehashed. Fully resident experts;
q8/VMM, max sequence 32768, native MTP (including its default n-gram lane;
standalone `ngram_draft` is off), NPU trunk `auto`, fixed 8 GiB PLE
prewarm/load-fill, session/disk caches off, DPM warmup 10 s.

Common CLI: the previous cycle's frozen candidate CLI. Original daemon is
the source tree corresponding to `7bcc41883`, MD5
`588270213b5d40ac753c597a7fdd07f7`. Inverse daemon, built by
`candidate-build-v2.log`, MD5 `08c0a555aaa965da40aac19e5092104f`, SHA-256
`d2cdf498c108b9f753f378fae96975941dc533f92e86451bc4cc12b50fe8a28e`.
The original binary was built before that source checkpoint was committed;
the current candidate is that parent plus this change. Command records contain
the comparison workspace's source file pins; those are not baseline build
identities. The Redline baseline records have an identity supplement correcting
copied template binary fields; their actual `--daemon` and report paths always
identify the original frozen binary.

## What the native MTP profile showed

`native-mtp-long-profile-v1`: ordinary daemon IPC, correct commit negotiation,
normal stdin EOF, rocprof kernel trace, no shutdown shim. Actual loaded ACK:
q8, VMM, MTP on. Long-summary prompt 21607 tokens, 97 output tokens, stop,
MTP active, 43 windows. Instrumented prefill 9994.5 ms, decode 40.8 tok/s.
These are diagnostic timings, not performance admission.

After the first vocabulary head, main GPU duration sums were:

| Kernel family | Sum, ms | Share of post-head kernel duration |
|---|---:|---:|
| F32 QT44 few-row grouped MoE gate/up | 418.2 | 19.9% |
| QT53 indexed MoE down | 242.6 | 11.6% |
| MQ6 shared-weight row-4 projections/head | 154.8 | 7.4% |
| MQ6 shared-weight row-3 projections/head | 140.8 | 6.7% |

The first-head boundary agrees approximately with the reported decode interval;
the trace is attribution evidence, not an isolated microbenchmark. Native MTP
chooses different window widths as timing changes.

## Adopted change

The few-row QT44 F32 gate/up used O4xR2 subtiles over the whole pad16 grouped
capacity. Most subtiles exit without math. The current scatter already writes
`inverse_perm[flat_slot] = grouped_row` (including `-1` for invalid routes).
Its expert runs have contiguous live prefixes and pad16 tails. Therefore each
live two-row pair has exactly one valid even inverse row as its leader.

The new launch visits flat slots, reads this inverse map, and skips negative,
out-of-range and odd positions before entering the shared R2 arithmetic body.
No allocation, copy, extra GPU pass, atomic or host synchronization is added.
The exact gfx1151, M=1280/K=2560/top-10, 1..=8-row F32 pipeline arm selects it.
Other dimensions, architectures and larger batches keep their existing paths.
The original R2 entry remains available as the mapping/ABI oracle.

For the production few-row grouping bound `grouped_rows = 16 * flat_slots`,
grid.y changes from `grouped_rows / 2` to `flat_slots`: eight times fewer CTAs.
For four source rows this is 25600 -> 3200 CTAs. The removed CTAs already
skipped matrix arithmetic; weight/activation traffic and FLOPs are not reduced
eightfold. This distinction explains why the full-model gain can be small.

Implementation: `kernels/src/gemm_mq4g256v2_moe_grouped_top10_o4_r8_x4.gfx1151.hip`,
`crates/rdna-compute/src/moe_few.rs`, kernel registry and
`hipfire-dispatch/src/pipeline/qt44_qt53_prefill.rs`. The launcher retains
ordinary profiling and the captured-kernarg funnel. The arithmetic body and
per-output reduction order are shared unchanged; the oracle compares independent
row-grid and inverse-grid mappings, not two independent implementations of the
same arithmetic.

## Kernel and capture validation

`inverse-gate-probe-v1`: 20 synthetic cases (1/2/3/4/8 source rows, disjoint,
shared, mixed and invalid routing), independent original launch vs inverse
launch, full output including unwritten padding and prefix/suffix guards,
finite live values, and a missing-leader negative control per case. All match;
every negative control is detected. Three alternating warm rounds per arm:

| Rows / routing | Original -> inverse median, us |
|---|---:|
| 2 / shared | 39.05 -> 31.10 |
| 3 / mixed | 96.20 -> 82.18 |
| 4 / shared | 85.60 -> 59.59 |
| 4 / mixed | 187.04 -> 162.56 |
| 4 / disjoint | 328.63 -> 328.85 |

These are synthetic kernel results, not model throughput. The committed GPU
test executes the production Rust launcher on 20 further cases with asymmetric
headers, exact buffer/guard comparison, a missing-leader negative control and
2/3/4-row HIP capture/replay. `gpu-oracle-v2.log`: actual gfx1151, 1 test passed,
not skipped. Release build and full pre-commit CI passed.

## Native service validation

Four fresh serve processes: original/candidate x AR/MTP, four prompts each
(long summary, primes, merge-sort code, LRU method completion). All 16 turns
finish with stop, lengths 97/43/156/48, no runaway/empty/attractor. Both candidate
transcripts are byte-identical to their controls, MD5
`aec69e55a2b43dfaab86e586272c0e3d`; decoded text was read. MTP is observed active.
Qwen4 logs and IPC ACK establish q8/VMM. The serve harness still reports its
known KV marker-parser UNOBSERVED; no harness parser was modified.

The first benchmark attempt was refused before load by the native device lock:
the immediately preceding serve child was still retiring. The recorded holder
PID was subsequently absent and no lock was held. The failed attempt has no
timing data. Fresh `bench-*-v2` samples retain A B B A A B order; a tooling
wrapper waits up to 30 s for the real device lock to retire under the outer GPU
lock, closes/releases it, then execs the native command. Neither lock is removed.

## Redline gate scope and retained failures

Default `redline-*-v1` failed identically on original and candidate: same 1177
launches/33 kernels/sequence hash, same state/logit differences. The new inverse
entry is absent (MTP is off in this gate). On gfx1151, eager F32 QSA split-K
and recorded per-head attention are different arithmetic by design; see
`qsa_decode_attention_eager_differs_from_recorded` and `Qwen4ArLaunch`.

Fresh `redline-baseline-v2` / `redline-candidate-v2` set only
`HIPFIRE_QWEN4_QSA_DECODE_SPLITK=0` for this retained-route check, making the HIP
reference use the same per-head arithmetic. Both pass: stable capture, 33 valid
AQL contracts, full state/logits byte parity at 513/514/515, and failure/recovery
checks. No tolerance or oracle criterion was weakened. The harness requires
BF16 KV for its Qwen4 mode. This certifies that route, not default split-K or
MTP. The real q8 MTP serve/performance runs keep default attention; the changed
entry itself has the direct eager/capture oracle above. All v1 failures remain.

## Matched native performance — no stable speedup claim

Six fresh processes, order A B B A A B. Each runs one identical throwaway request
then one measured request via `hipfire bench --backend noslots --workload stateless`.
Committed original long-code prompt MD5 `b4d0b63cddcac872648ddf3cdd92cac2`, actual
21582 prompt tokens, 256 output budget, greedy, NPU auto/native MTP/q8/VMM.
The CLI reports `spec_route=mtp+ngram` with `mtp_effective=on`; both arms keep
the same default built-in n-gram lane. These are the normal native speculative
route, not a forced native-only ablation.
No profiler or concurrent CPU build/audit. Raw samples:

| Arm | Decode, tok/s | TTFT, ms |
|---|---|---|
| Original | 51.2 / 53.1 / 51.7 | 9885.1 / 9893.7 / 9872.3 |
| Inverse | 52.2 / 51.2 / 52.4 | 9861.0 / 9867.2 / 10003.7 |
| Median | **51.7 -> 52.2 (+0.97%)** | **9885.1 -> 9867.2 (-0.18%)** |

Ranges overlap, and adaptive MTP tau varies (original 2.07/2.07/1.97;
candidate 2.00/1.71/1.83). All runs have PPT limitations; one original sample
has 2.8% core thermal-limiter residency. Full power/clock/memory fields are
archived. There is no established stable end-to-end speedup. Adoption rests on
verified work removal, unchanged output, and no measured sustained regression.

## Actual route profile and next work

`native-mtp-inverse-profile-v1`: loaded q8/VMM/MTP on, 97 tokens/stop,
44 windows. The new entry executes 2016 times; old R2 executes zero times.
Both use 112 VGPRs, 128 allocated SGPRs, 2048 B LDS, zero scratch/spills.
Observed per-call medians by source-row count:

| Rows | Original -> inverse, us |
|---|---:|
| 2 | 151.935 -> 151.375 |
| 3 | 201.334 -> 200.974 |
| 4 | 259.611 -> 253.352 |

These profiles have different adaptive window mixes, not paired input tensors;
do not turn their total durations into a speedup claim. Empty dispatches are a
small part of real-model gate/up cost. Next quantify weight/activation traffic
and execution stalls, and investigate exact shared-weight MoE down across
rows and the actual prefill builder's independent WMMA chains/leader schedule.
Preserve the full native prefill/decode scope and do not resume the stopped
full-NPU drafter direction.
