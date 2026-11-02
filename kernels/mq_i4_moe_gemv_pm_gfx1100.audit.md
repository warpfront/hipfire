# Peacemaker ISA audit

Input: `/home/kaden/qcal/release-0.4.3/consumer24-k1/kernels/mq_i4_moe_gemv_pm_gfx1100.co`; arch: `gfx1100`. ISA evidence lines refer to the disassembly named in each profile.

## default (`/home/kaden/qcal/release-0.4.3/consumer24-k1/kernels/mq_i4_moe_gemv_pm_gfx1100.co`; [ISA](/home/kaden/qcal/release-0.4.3/consumer24/rcpt/b2.k1.audit.default.isa.txt))

| Kernel | VGPR | Private B/thread | Hot loop WMMA/SWMMAC | VALU slots | VOPD fold/all/pairable | s_wait_alu | Findings |
|---|---:|---:|---:|---:|---:|---:|---|
| `mq_i4_moe_gemv_mq2i4_r256_gate_up_pm_gfx1100` | 104 | 0 | 0/0 | 288 | 0/0/88 | 0 | 16 |
| `mq_i4_moe_gemv_mq2i4_r128_down_pm_gfx1100` | 80 | 0 | 0/0 | 72 | 0/0/22 | 0 | 4 |
| `mq_i4_moe_gemv_mq4sym_r256_gate_up_pm_gfx1100` | 144 | 0 | 0/0 | 440 | 0/0/144 | 0 | 32 |
| `mq_i4_moe_gemv_mq4sym_r128_down_pm_gfx1100` | 80 | 0 | 0/0 | 66 | 0/0/20 | 0 | 4 |

### `mq_i4_moe_gemv_mq2i4_r256_gate_up_pm_gfx1100`

- Loop 0 ISA lines 183–852 (PC 0x324c–0x3db0): WMMA 0, SWMMAC 0, VALU slots 288, VOPD 0 fold/0 all/88 bank-pairable, waits {"s_waitcnt": 4}, DS 0, VMEM 40, barriers 0.
- **loop_invariant_rematerialization** ISA lines 227–227 (PC 0x3328–0x3328): v_bfe_u32 reads no register defined in this loop.
- **loop_invariant_rematerialization** ISA lines 236–236 (PC 0x3350–0x3350): v_bfe_u32 reads no register defined in this loop.
- **loop_invariant_rematerialization** ISA lines 245–245 (PC 0x3378–0x3378): v_bfe_u32 reads no register defined in this loop.
- **loop_invariant_rematerialization** ISA lines 254–254 (PC 0x33a0–0x33a0): v_bfe_u32 reads no register defined in this loop.
- **loop_invariant_rematerialization** ISA lines 394–394 (PC 0x3600–0x3600): v_bfe_u32 reads no register defined in this loop.
- **loop_invariant_rematerialization** ISA lines 403–403 (PC 0x3628–0x3628): v_bfe_u32 reads no register defined in this loop.
- **loop_invariant_rematerialization** ISA lines 412–412 (PC 0x3650–0x3650): v_bfe_u32 reads no register defined in this loop.
- **loop_invariant_rematerialization** ISA lines 421–421 (PC 0x3678–0x3678): v_bfe_u32 reads no register defined in this loop.
- **loop_invariant_rematerialization** ISA lines 561–561 (PC 0x38d8–0x38d8): v_bfe_u32 reads no register defined in this loop.
- **loop_invariant_rematerialization** ISA lines 570–570 (PC 0x3900–0x3900): v_bfe_u32 reads no register defined in this loop.
- **loop_invariant_rematerialization** ISA lines 579–579 (PC 0x3928–0x3928): v_bfe_u32 reads no register defined in this loop.
- **loop_invariant_rematerialization** ISA lines 588–588 (PC 0x3950–0x3950): v_bfe_u32 reads no register defined in this loop.
- **loop_invariant_rematerialization** ISA lines 728–728 (PC 0x3bb0–0x3bb0): v_bfe_u32 reads no register defined in this loop.
- **loop_invariant_rematerialization** ISA lines 737–737 (PC 0x3bd8–0x3bd8): v_bfe_u32 reads no register defined in this loop.
- **loop_invariant_rematerialization** ISA lines 746–746 (PC 0x3c00–0x3c00): v_bfe_u32 reads no register defined in this loop.
- **loop_invariant_rematerialization** ISA lines 755–755 (PC 0x3c28–0x3c28): v_bfe_u32 reads no register defined in this loop.

### `mq_i4_moe_gemv_mq2i4_r128_down_pm_gfx1100`

- Loop 0 ISA lines 1897–2065 (PC 0x50dc–0x53b8): WMMA 0, SWMMAC 0, VALU slots 72, VOPD 0 fold/0 all/22 bank-pairable, waits {"s_waitcnt": 1}, DS 0, VMEM 10, barriers 0.
- **loop_invariant_rematerialization** ISA lines 1941–1941 (PC 0x51b8–0x51b8): v_bfe_u32 reads no register defined in this loop.
- **loop_invariant_rematerialization** ISA lines 1950–1950 (PC 0x51e0–0x51e0): v_bfe_u32 reads no register defined in this loop.
- **loop_invariant_rematerialization** ISA lines 1959–1959 (PC 0x5208–0x5208): v_bfe_u32 reads no register defined in this loop.
- **loop_invariant_rematerialization** ISA lines 1968–1968 (PC 0x5230–0x5230): v_bfe_u32 reads no register defined in this loop.

### `mq_i4_moe_gemv_mq4sym_r256_gate_up_pm_gfx1100`

- Loop 0 ISA lines 2538–3519 (PC 0x5d44–0x6ee8): WMMA 0, SWMMAC 0, VALU slots 440, VOPD 0 fold/0 all/144 bank-pairable, waits {"s_waitcnt": 4}, DS 0, VMEM 72, barriers 0.
- **loop_invariant_rematerialization** ISA lines 2592–2592 (PC 0x5e70–0x5e70): v_bfe_u32 reads no register defined in this loop.
- **loop_invariant_rematerialization** ISA lines 2597–2597 (PC 0x5e8c–0x5e8c): v_bfe_u32 reads no register defined in this loop.
- **loop_invariant_rematerialization** ISA lines 2602–2602 (PC 0x5ea8–0x5ea8): v_bfe_u32 reads no register defined in this loop.
- **loop_invariant_rematerialization** ISA lines 2607–2607 (PC 0x5ec4–0x5ec4): v_bfe_u32 reads no register defined in this loop.
- **loop_invariant_rematerialization** ISA lines 2612–2612 (PC 0x5ee0–0x5ee0): v_bfe_u32 reads no register defined in this loop.
- **loop_invariant_rematerialization** ISA lines 2617–2617 (PC 0x5efc–0x5efc): v_bfe_u32 reads no register defined in this loop.
- **loop_invariant_rematerialization** ISA lines 2622–2622 (PC 0x5f18–0x5f18): v_bfe_u32 reads no register defined in this loop.
- **loop_invariant_rematerialization** ISA lines 2627–2627 (PC 0x5f34–0x5f34): v_bfe_u32 reads no register defined in this loop.
- **loop_invariant_rematerialization** ISA lines 2837–2837 (PC 0x62d8–0x62d8): v_bfe_u32 reads no register defined in this loop.
- **loop_invariant_rematerialization** ISA lines 2842–2842 (PC 0x62f4–0x62f4): v_bfe_u32 reads no register defined in this loop.
- **loop_invariant_rematerialization** ISA lines 2847–2847 (PC 0x6310–0x6310): v_bfe_u32 reads no register defined in this loop.
- **loop_invariant_rematerialization** ISA lines 2852–2852 (PC 0x632c–0x632c): v_bfe_u32 reads no register defined in this loop.
- **loop_invariant_rematerialization** ISA lines 2857–2857 (PC 0x6348–0x6348): v_bfe_u32 reads no register defined in this loop.
- **loop_invariant_rematerialization** ISA lines 2862–2862 (PC 0x6364–0x6364): v_bfe_u32 reads no register defined in this loop.
- **loop_invariant_rematerialization** ISA lines 2867–2867 (PC 0x6380–0x6380): v_bfe_u32 reads no register defined in this loop.
- **loop_invariant_rematerialization** ISA lines 2872–2872 (PC 0x639c–0x639c): v_bfe_u32 reads no register defined in this loop.
- **loop_invariant_rematerialization** ISA lines 3082–3082 (PC 0x6740–0x6740): v_bfe_u32 reads no register defined in this loop.
- **loop_invariant_rematerialization** ISA lines 3087–3087 (PC 0x675c–0x675c): v_bfe_u32 reads no register defined in this loop.
- **loop_invariant_rematerialization** ISA lines 3092–3092 (PC 0x6778–0x6778): v_bfe_u32 reads no register defined in this loop.
- **loop_invariant_rematerialization** ISA lines 3097–3097 (PC 0x6794–0x6794): v_bfe_u32 reads no register defined in this loop.
- **loop_invariant_rematerialization** ISA lines 3102–3102 (PC 0x67b0–0x67b0): v_bfe_u32 reads no register defined in this loop.
- **loop_invariant_rematerialization** ISA lines 3107–3107 (PC 0x67cc–0x67cc): v_bfe_u32 reads no register defined in this loop.
- **loop_invariant_rematerialization** ISA lines 3112–3112 (PC 0x67e8–0x67e8): v_bfe_u32 reads no register defined in this loop.
- **loop_invariant_rematerialization** ISA lines 3117–3117 (PC 0x6804–0x6804): v_bfe_u32 reads no register defined in this loop.
- **loop_invariant_rematerialization** ISA lines 3327–3327 (PC 0x6ba8–0x6ba8): v_bfe_u32 reads no register defined in this loop.
- **loop_invariant_rematerialization** ISA lines 3332–3332 (PC 0x6bc4–0x6bc4): v_bfe_u32 reads no register defined in this loop.
- **loop_invariant_rematerialization** ISA lines 3337–3337 (PC 0x6be0–0x6be0): v_bfe_u32 reads no register defined in this loop.
- **loop_invariant_rematerialization** ISA lines 3342–3342 (PC 0x6bfc–0x6bfc): v_bfe_u32 reads no register defined in this loop.
- **loop_invariant_rematerialization** ISA lines 3347–3347 (PC 0x6c18–0x6c18): v_bfe_u32 reads no register defined in this loop.
- **loop_invariant_rematerialization** ISA lines 3352–3352 (PC 0x6c34–0x6c34): v_bfe_u32 reads no register defined in this loop.
- **loop_invariant_rematerialization** ISA lines 3357–3357 (PC 0x6c50–0x6c50): v_bfe_u32 reads no register defined in this loop.
- **loop_invariant_rematerialization** ISA lines 3362–3362 (PC 0x6c6c–0x6c6c): v_bfe_u32 reads no register defined in this loop.

### `mq_i4_moe_gemv_mq4sym_r128_down_pm_gfx1100`

- Loop 0 ISA lines 4798–4952 (PC 0x86c8–0x8984): WMMA 0, SWMMAC 0, VALU slots 66, VOPD 0 fold/0 all/20 bank-pairable, waits {"s_waitcnt": 1}, DS 0, VMEM 10, barriers 0.
- **loop_invariant_rematerialization** ISA lines 4844–4844 (PC 0x87b4–0x87b4): v_bfe_u32 reads no register defined in this loop.
- **loop_invariant_rematerialization** ISA lines 4849–4849 (PC 0x87d0–0x87d0): v_bfe_u32 reads no register defined in this loop.
- **loop_invariant_rematerialization** ISA lines 4854–4854 (PC 0x87ec–0x87ec): v_bfe_u32 reads no register defined in this loop.
- **loop_invariant_rematerialization** ISA lines 4859–4859 (PC 0x8808–0x8808): v_bfe_u32 reads no register defined in this loop.

