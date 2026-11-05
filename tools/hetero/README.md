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
