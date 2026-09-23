# Compute Barrier Research — GFX10 (gfx1010)

## Source: Linux kernel gfx_v10_0.c

The ground truth is `gfx_v10_0_ring_emit_fence()` (line 8703) and
`gfx_v10_0_wait_reg_mem()` (line 4011) in the amdgpu kernel driver.

## What Previous Attempts Got Wrong

1. **CS_PARTIAL_FLUSH**: This is an EVENT_WRITE packet, not a fence. It tells
   the ME/MEC to flush, but doesn't WAIT for completion. On MEC (compute),
   it doesn't drain the compute pipeline.

2. **ACQUIRE_MEM GCR_CNTL**: The GCR_CNTL bit layout has 2-bit GLI_INV at [1:0],
   which shifted all other bits. We had GL2_WB at bit 3 instead of bit 4.
   More importantly, ACQUIRE_MEM initiates cache ops but doesn't wait for
   outstanding compute dispatches to finish. Also, the ACQUIRE_MEM with
   non-zero GCR_CNTL hangs on gfx1010 MEC for unknown reasons.

3. **The correct approach**: RELEASE_MEM (writes fence + flushes caches after
   all prior work completes) + WAIT_REG_MEM (polls fence value until written).

## RELEASE_MEM Packet (GFX10 Compute)

```
Opcode: 0x49 (PACKET3_RELEASE_MEM)
Body: 7 dwords (header count=6)

Header: 0xC0064902
  [31:30] = 3 (type 3 packet)
  [29:16] = 6 (count)
  [15:8]  = 0x49 (opcode)
  [1]     = 1 (SHADER_TYPE = compute)

DW1: Event + GCR flags = 0x06198514
  [5:0]   EVENT_TYPE = 0x14 (CACHE_FLUSH_AND_INV_TS_EVENT)
  [11:8]  EVENT_INDEX = 5
  [12]    GCR_GLV_INV = 0
  [13]    GCR_GL1_INV = 0
  [14]    GCR_GL2_INV = 0
  [15]    GCR_GL2_WB = 1  ← writeback L2
  [16]    GCR_SEQ = 1     ← sequential ordering
  [18:17] GCR_GL2_RANGE = 0 (all)
  [19]    GCR_GLM_WB = 1  ← writeback metadata cache
  [20]    GCR_GLM_INV = 1 ← invalidate metadata cache
  [26:25] CACHE_POLICY = 3

DW2: Data/interrupt select = 0x20000000
  [28:26] DATA_SEL = 1 (write 32-bit immediate)
  [31:29] INT_SEL = 0 (no interrupt)

DW3: addr_lo (GPU VA of fence, dword-aligned)
DW4: addr_hi
DW5: data_lo (fence value to write)
DW6: data_hi (0 for 32-bit write)
DW7: 0 (reserved)
```

## WAIT_REG_MEM Packet

```
Opcode: 0x3C (PACKET3_WAIT_REG_MEM)
Body: 6 dwords (header count=5)

Header: 0xC0053C02
  [31:30] = 3 (type 3)
  [29:16] = 5 (count)
  [15:8]  = 0x3C (opcode)
  [1]     = 1 (SHADER_TYPE = compute)

DW1: Control = 0x00000013
  [2:0] FUNCTION = 3 (equal)
  [4]   MEM_SPACE = 1 (memory, not register)
  [6]   OPERATION = 0 (wait until condition met)
  [8]   ENGINE = 0 (ME/MEC)

DW2: addr_lo (same GPU VA as RELEASE_MEM fence)
DW3: addr_hi
DW4: reference (fence value to compare against)
DW5: mask = 0xFFFFFFFF (compare all bits)
DW6: poll_interval = 4
```

## Barrier Sequence

```
Dispatch A
  ↓
RELEASE_MEM → waits for A to finish, flushes L2, writes fence=N to fence_va
  ↓
WAIT_REG_MEM → polls fence_va until value == N
  ↓
Dispatch B (guaranteed to see A's writes)
```
