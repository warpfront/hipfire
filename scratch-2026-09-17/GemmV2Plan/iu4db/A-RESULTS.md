# Change A — double-buffered slab staging (W-double / A-single, 3 WG/CU)

## Design (deviation from the brief, per parent note 2026-09-18)
The brief specified two full A/W slab buffers (~22 KiB, 1 barrier/slab).
Built and measured exactly that first (A-full: A0/A1 + W0/W1 + DS + SZ ping,
24576 B, 2 barriers/block, rolled h-loop, VGPR 219, 0 spill, oracle 9/9):
gate 744 / down ~712 at 2 WG/CU — SLOWER than v2 (680/702 at 3 WG/CU).
Occupancy 3->2 cost dominates the overlap gain. Trimmed per the parent note
(double-buffer W only) and measured both points: A-full loses, A3 wins.
Change A (committed) = A3.

A3: W slab double-buffered per 64-K slab (W0/W1); A slab single-buffered (A)
with v2's publish fence; DS + SZ ping-pong per half-block (plane[h],
kb = 2g+h); token (d,s) re-read from the DS plane at fold time (same values,
same DAG; frees 8 VGPRs). LDS 19456 B = 19 KiB ->
3 x 19456 = 58368 < 65536/CU on gfx1201 -> 3 WG/CU (OCC-probe measured).

Steady state per 128-K block (4 barriers, vs 5 in v2):
1. prefetch slab 1 (A->A_pf, W->W_pf), sched-pinned above WMMAs;
   compute slab 0 from (A, W0);
2. B0 (slab-0 A-reads done: A publish overwrites slab 0);
   publish (A_pf->A, W_pf->W1); refill A_pf with next A slab 0;
3. B1 (publish visible); compute slab 1 from (A, W1);
   stage next W0/DS[h^1]/SZ[h^1] straight to LDS (above the fold);
4. fold (IU4_FOLD_RN untouched); B2a (slab-1 A-reads done);
   publish A_pf->A; B2b (visible).
h-loop rolled (unroll_count(1)): the unrolled h x 2 body kept block k's
staging+fold temporaries live across block k+1 and spilled at 256 VGPR;
rolled, only cacc/acc carry. 220 VGPR, 0 spill/scratch.

Race-freedom: B0 fences A-overwrite vs slab-0 reads; B1 publish
visibility; W0-next/DS-next/SZ-next target buffers no in-flight reader
touches (W1/A/SZ[h]/DS[h] being read); B2a fences A-republish vs slab-1
reads; B2b next-block visibility. Oracle 9/9 bit-identical proves it
empirically (integer MMA + same fold DAG).

## ISA (tu-changeA3.o, gfx1201, A3dis.txt)
- slab-1 prefetch globals (offset:40) issue above slab-0 WMMAs (FIRST `, 0`);
  next-block staging globals issue above the fold bpermutes.
- fold bpermutes intact; K-step = ds_reads + WMMAs only (bundle unchanged).
- 4 s_barrier pairs/block as designed (B0/B1/B2a/B2b).

## Gates
1. Oracle 9/9 PASS, cpu_bitwise=OK mism=0, repeat_identical=OK (A-oracle.log).
2. Metadata (JIT e9175e1eb2e29f4f + A-occ.log):
   VGPR 220 (all 3 symbols) <= 256 | spill 0/0 | scratch 0 |
   LDS 19456 B | WG/CU 3 (measured OCC probe, block 256).
3. TIME N=512, same session (A-time.log; v2 + s2bt8 same-session controls
   in interleave-v2-A.log):
   gate/set 679.9 (__TIME medians below__), down/add 660.2,
   qkvza/set 661.8, qkv/set 560.6.
   vs v2 same-session (~681/~702, launcher harness): gate -0.2%, res -6.0%.
   vs A-full (24576 B, 2 WG/CU): gate 744->680 (-8.6%), down ~712->660 (-7.3%).
   Incumbent s2bt8 same-session: gate_up fused 1679.3, residual 850.1.

## TIME medians, N=512 (isolated GEMM us/call)
| case | M | K | N | mode | A3 us | v2 same-session | s2bt8 same-session |
|---|---|---|---|---|---|---|---|
| gate | 17408 | 5120 | 512 | set | 679.9 | ~681 | 1679.3 (fused, 2 matrices) |
| gate | 17408 | 5120 | 512 | add | 684.6 | — | — |
| down (residual) | 5120 | 17408 | 512 | add | 660.2 | ~702 | 850.1 |
| qkvza-synth | 16480 | 5120 | 512 | set | 661.8 | — | — |
| qkv-synth | 14336 | 5120 | 512 | set | 560.6 | — | — |

No 20% on gate_up (gate -0.2%): gate-4 KLD/bench/serve phase NOT triggered
by A alone. Change B follows on top.
