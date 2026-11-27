### Tests: Qwen4 EF oracle expects BF16-rounded conv rows

- **Fixed: the Qwen4 Q8Ef snapshot/restore oracle (`state::ef_oracle::q8ef_snapshot_restore_rollback_continue_is_byte_exact_hw`) expects the conv rows that #1199 keeps.** Since `f044e7cce` (#1199), `rollback_rows_retain` copies the kept conv rows through `copy_regions_bf16`, so the GPU conv history is BF16-rounded. The oracle still expected raw F32 rows and failed at `state_ef_oracle.rs:232` (`keep 1: conv`) on gfx1151. `conv_after` now rounds each kept input row through `half::bf16`. The rollback code is unchanged.
