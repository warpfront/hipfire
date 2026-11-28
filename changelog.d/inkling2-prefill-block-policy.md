### Performance: Inkling-2-Mini prefill block size follows the load's VRAM (gfx1201)

- **Inkling-2-Mini blocked prefill (arch 24, gfx1201 typed-XTS loads) runs
  2048-row blocks where the VRAM allows.** At load, after the weights, KV and
  conv are resident, the prefill takes the widest of 2048, 1024 and 512 rows
  whose persistent scratch, pooled working set and a 1 GiB headroom fit the
  free VRAM (`PrefillBlockPlan`). A load that misses the widest rung logs
  `inkling2 prefill: 2048-row blocks need … GiB …: falling back to N-row
  blocks`. The load summary now prints the scratch at the chosen rung and the
  working set that the first prefill will allocate, with the free VRAM the
  choice was made against: an R9700 at `max_seq` 32768 has 17.0 GiB free and
  takes 2048 rows (scratch 0.527 GiB, working set 0.969 GiB). The working-set
  model (128 MiB plus 48 `[rows, n_embd]` F32 streams) is an upper bound on the
  measured pool growth: 308, 512 and 919 MB at 512, 1024 and 2048 rows, and the
  same 919 MB at pp8192.
- `generate_inkling` prefills in chunks of the load's block rows
  (`prefill_chunk_rows`), and `forward_prefill_gpu` blocks at the same size.
  Prefill and decode logits are bitwise unchanged: the last-row logits SHA at
  pp2048 and pp8192 is the same at every rung, and the WT2 and code24 KLD
  sequences (prefill and decode-blocked routes) are byte-identical to beta's.
- Daemon prefill on R9700, 4 fresh loads per arm (ABBA BAAB), 2048-row blocks
  vs beta's 512: pp2048 goes from 4,416 to 5,637 tok/s (+27.6%), and an
  8,127-token prompt from 4,234 to 5,388 tok/s (+27.2%). pp512 is unchanged
  (4,288 vs 4,293), and so is decode. An 8K prompt plus decode at
  `max_seq` 32768 peaks 888 MiB above beta's VRAM (16.2 GiB in all).
- The daemon's `hipfire load summary` prefill facet for Inkling-2 now names the
  route that runs (`blocked, 2048-row blocks (widest of [2048, 1024, 512] the
  free VRAM admits)`). Before this change it read `single-row` even on
  blocked loads.
- The scored-rows prefill head (evaluation only) reuses one 64-row logits
  buffer. Before this change it kept every chunk of the block allocated.
- The blocked-prefill oracle (`tests/prefill_parity.rs`) gains
  `blocked_prefill_oracle_rungs` (2,600 rows). It runs row-exact (bitwise) and
  gated blocked arms at 1024 and 2048 rows, plus blocked/512. The wide arms
  prefill in two calls, so one 2048-row block starts mid-ring (`[300, 2348)`,
  wider than the 1024-row SWA ring), and both end on a partial tail. Probes
  cover positions 1023 to 1025 and every block boundary.
- `inkling_kld bench` records the load's rung and plan, and the pooled working
  set (`prefill_pool_bytes`).
