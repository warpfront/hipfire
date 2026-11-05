# tools/hetero

Probes for the Qwen4 (Flash-Next) heterogeneous expert split: a gfx1201 R9700 runs the
trunk and the hot routed experts; the gfx1151 Strix Halo iGPU computes the cold routed
experts out of its own (unified) memory. The design is in
`docs/superpowers/specs/2026-11-04-qwen4-hetero-expert-server-design.md`.

Tested machine: Framework Desktop (Strix Halo, 128 GB, BIOS 3.06), R9700 in a USB4
enclosure (256 MB BAR), kernel 7.0.1 in-tree amdgpu, ROCm 7.2.2.

## Proof A: the iGPU reads R9700-pinned host memory (`cross_pinned.hip`)

```
hipcc -O2 --offload-arch=gfx1151 --offload-arch=gfx1201 cross_pinned.hip -o cross_pinned
./cross_pinned 2147483648
```

The probe pins 2 GiB through the gfx1201 device and fills it on the CPU. The gfx1151 iGPU
then reads it through its own `hipHostGetDevicePointer`, verifying a seeded pattern on
every pass (5 timed passes).

| buffer | mismatches | median read |
|---|---|---|
| `hipHostMalloc(Mapped)` on gfx1201 (hipfire's host-mapped tier flags) | 0 | 229.3 GB/s |
| `hipHostMalloc(Mapped \| Portable)` on gfx1201 | 0 | 230.1 GB/s |
| control: `hipMalloc` on gfx1151 | 0 | 223.0 GB/s |

**Proof A: PASS (Mapped-only works).** The iGPU reads the existing host-mapped expert tier
in place at its full local bandwidth. No copy, and no allocation-flag change, is needed for
the cold arm.

## Proof B: CPU-driven iGPU responder latency (`pingpong.hip`)

```
hipcc -O2 -std=c++20 --offload-arch=gfx1151 --offload-arch=gfx1201 pingpong.hip -o pingpong
for i in 1 2 3; do ./pingpong 20000 5120 ABC; done
```

The R9700 posts a 5 KB payload (one Flash-Next hidden row) plus a flag into pinned
coherent host memory; the Strix side answers with 5 KB plus a flag. Every spin is bounded.
Plain loads and stores only (no PCIe atomics on this link). Three runs, all
`payload errors a=0 b=0`, no kernel faults:

| mode | round trip |
|---|---|
| A: persistent kernels on both GPUs | 21.0-22.4 us |
| C: a Strix host thread spins on the flag, launches one captured hipGraph per request; its kernel answers and stores the flag | 23.0-23.2 us |
| B: host-mediated (kernel + hipMemcpy + sync on each side) | 204-205 us |

**Proof B: CPU-driven responder.** C is within 1.1x of A, against the 2x decision
threshold. The decode arm launches captured gfx1151 graphs of the existing MoE kernels from
a host thread. That reuses kernels unchanged and never pins iGPU CUs while idle. At 48
layers per token, the transport costs about 1.1 ms per token.

(A throwaway predecessor of this probe measured mode A at 14.8 us. The committed version
differs in flag access (`__hip_atomic` system-scope on both sides) and in the responder
block size. Both modes here are measured with the same code.)
