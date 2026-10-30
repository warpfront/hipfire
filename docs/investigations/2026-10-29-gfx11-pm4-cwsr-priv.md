# gfx11 retained-PM4 dispatches need `COMPUTE_PGM_RSRC1.PRIV` under CWSR

Status: root cause identified and fixed (`40d9d0401`). This is the triage note
behind the `HIPFIRE_DEBUG_PM4_NO_CWSR_PRIV` negative control in
[`../env-vars.md`](../env-vars.md) and behind the `vram` retained-kernarg
default that the fix restored.

## Symptom

A long retained-PM4 soak on gfx1100 (multi-slot, VMM-paged KV) intermittently
wedged the device. Every fault had the same signature, in steady-state decode,
never on a prepare:

```
amdgpu: sq_intr: ... (wgp 0, waves 0/4/5/6)
amdgpu: gfxhub page fault (vmid 8, addr 0x0, client SQC (data),
        read, PERMISSION_FAULTS 0x3, STATUS 0x00801431)
amdgpu: MES failed to respond to msg=REMOVE_QUEUE / SUSPEND
amdgpu: GPU reset(3) … succeeded
```

The daemon side shows `HIPFIRE_REDLINE_QUEUE_TIMEOUT` with
`PM4_PROGRESS end_tick=0` on the live slots tape. A null scalar (SQC) read
means a dispatch loaded a pointer that was zero where it expected a kernarg
or descriptor — but the tape, the kernargs and the block tables were all
intact, and the same tape replayed correctly thousands of times before and
after each fault.

## Cause

On GFX11 the AQL dispatch microcode sets `COMPUTE_PGM_RSRC1.PRIV` (bit 20)
implicitly, and CWSR (compute wave save/restore) depends on it. Redline's
raw-PM4 encoder wrote the code object's `rsrc1` unchanged, and the code
object does not carry that bit, so a retained PM4 dispatch ran with PRIV
clear. A KFD queue preemption (eviction / MES preemption) during such a
dispatch then faults, and the MES cannot remove or suspend the queue
afterwards — the reset is the only exit.

The windows are therefore rare and preemption-driven, which is why the fault
rate tracked eviction and soak length rather than request shape, and why a
20-iteration A/B could not discriminate configurations.

## Fix

`Gfx10Pm4CommandBuffer::with_gfx11_cwsr_priv()` sets the bit; the replay
builder enables it for `Pm4Architecture::Gfx11` only. GFX10 and GFX12
encodings are unchanged byte for byte.

## Evidence

gfx1100, `cwsr_enable=1`, forced KFD evictions every 10 ms
(`HIPFIRE_DEBUG_FORCE_EVICT_MS=10`), 120 s runs from an idle start:

| arm | result |
|---|---|
| `HIPFIRE_DEBUG_PM4_NO_CWSR_PRIV=1` | 7/7 runs page-faulted (same signature) |
| PRIV set (default) | 5/5 runs clean, 0 kernel faults |

The negative control exists so the fault class stays reproducible without a
driver-level eviction harness: never set it outside this repro.

## Ruled out first

The fault was originally attributed to the retained-kernarg storage, because
a host-memory kernarg pool never faulted and a device-VRAM pool did. Each
hypothesis below was measured against the same soak and rejected:

| hypothesis | test | outcome |
|---|---|---|
| host BAR writes not visible | `HDP_MEM_FLUSH_CNTL` flush + readback on publish | still faults |
| PM4 SH-register elision | full SH emission (`REPLAY_PM4_STATEFUL=legacy`) | still faults |
| VRAM kernarg VA rebound to a new page | never-freed recycle pool | still faults (19 iterations) |
| kernarg pool location | host pool | clean, but ~13-15 % slower decode |

The host pool was adopted as a workaround at that point (`129ab13ee`) and
reverted (`8f9368f3d`) once PRIV explained the fault class: the pool
location was correlated with the fault, not its cause.

## References

- `40d9d0401` — the fix.
- `4034e3264` — the two debug knobs (`HIPFIRE_DEBUG_PM4_NO_CWSR_PRIV`,
  `HIPFIRE_DEBUG_FORCE_EVICT_MS`), default off.
- `8f9368f3d` — revert of the host-kernarg-pool workaround.
- [`2026-10-30-vmm-paged-kv-fixlog-and-revalidation.md`](2026-10-30-vmm-paged-kv-fixlog-and-revalidation.md)
  — the chronological log of the soaks that produced these numbers.
