# AQL User-Mode Queue Research

## System State
- `/dev/kfd` exists, user has `render` group access
- KFD topology: node 1 = navi10, gpu_id=38217, gfx_target_version=100100
- wave_front_size=32, 80 SIMDs, drm_render_minor=128
- Kernel 6.17.0, kfd_ioctl.h available at /usr/include/linux/kfd_ioctl.h

## Chosen Approach: KFD AQL Queue (Option A)

The `/dev/kfd` path is what HIP uses internally. It provides:
1. User-mode ring buffer (no ioctl per dispatch)
2. Doorbell writes (MMIO, ~1 PCIe write)
3. AQL packet format (64-byte packets, hardware-parsed)

## KFD Ioctl Sequence

```
1. open("/dev/kfd") → kfd_fd
2. AMDKFD_IOC_GET_VERSION → verify KFD version
3. AMDKFD_IOC_GET_PROCESS_APERTURES_NEW → discover gpu_id (38217)
4. AMDKFD_IOC_ACQUIRE_VM(drm_fd=renderD128, gpu_id=38217) → bridge address spaces
5. Allocate: ring buffer (64KB GTT), EOP buffer (4KB), optional CWSR
6. AMDKFD_IOC_CREATE_QUEUE(queue_type=AQL=0x2) → get doorbell_offset, queue_id
7. mmap(kfd_fd, doorbell_offset, 8192) → doorbell page in userspace
```

## AQL Dispatch Packet (64 bytes)
```
[0:1]   header: type=2 (dispatch), barrier=1, acquire=2 (agent), release=2 (agent)
[2:3]   setup: ndims
[4:9]   workgroup_size_x/y/z (u16 each)
[12:23] grid_size_x/y/z (u32 each) — total work items, NOT groups
[24:27] private_segment_size
[28:31] group_segment_size (LDS bytes)
[32:39] kernel_object → GPU VA of kernel DESCRIPTOR (not code entry!)
[40:47] kernarg_address → GPU VA of kernarg buffer
[56:63] completion_signal (0 = no signal)
```

## Key Difference from PM4 Path
- PM4: kernel_object = code entry address (code_va)
- AQL: kernel_object = kernel descriptor address (kd_va)
- The hardware reads the KD to find the code entry, RSRC1/2, etc.

## Completion
- For now: poll a flag in VRAM (write 0→1 via completion_signal)
- Future: proper HSA signals via KFD_IOC_CREATE_EVENT

## Memory Interop
- After ACQUIRE_VM, libdrm-allocated buffers (amdgpu_bo_alloc) are accessible
  from KFD queues — same GPU VA space
- Ring buffer + EOP: allocate via KFD ioctls (required by CREATE_QUEUE validation)
- Kernel code + kernarg + data: can use existing libdrm allocations

## FINDING: KFD AQL Not Feasible on gfx1010

**KFD_IOC_CREATE_QUEUE returns EINVAL for ALL queue types (PM4 and AQL).**

Root cause: KFD topology reports `local_mem_size=0` for gfx1010 (Navi 10).
This means the KFD thinks the discrete GPU has no VRAM. Queue creation fails
because the MQD (Message Queue Descriptor) must be allocated in VRAM.

This is the same fundamental limitation that prevents ROCm from working on
consumer RDNA1 hardware. The KFD kernel module has gfx1010 support gated
behind VRAM detection, which doesn't work for this class of GPU.

Verified with both Rust and C programs. Tested:
- PM4 compute queue (type=0) → EINVAL
- AQL compute queue (type=2, no CWSR) → EINVAL
- AQL with 16MB CWSR buffer → EINVAL

The KFD path requires hardware officially supported by AMD's compute stack.
For gfx1010, the only compute dispatch path is libdrm's amdgpu_cs_submit.

## Optimized Ioctl Path (Fallback)

Since AQL is blocked, optimize the existing amdgpu_cs_submit path:
1. Zero-copy kernarg: map once, update in place (eliminates memcpy)
2. Pre-built PM4 template: patch only kernarg VA + grid dims per dispatch
3. Persistent BO list: create once, reuse
4. IB reuse: keep the IB buffer mapped, overwrite per dispatch
