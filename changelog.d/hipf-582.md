### Fixes: Qwen4 state tests

- **The Q8Ef state lifecycle oracle expects the BF16-rounded conv rows the rollback stores (HIPF-582).** Since #1199 (`f044e7cce`), `rollback_rows_retain` copies kept conv rows BF16-rounded, as the conv kernels store them. `q8ef_snapshot_restore_rollback_continue_is_byte_exact_hw` still expected raw F32 rows and failed on GPU at `keep 1: conv` (it skips without a GPU). Its `conv_after` now rounds the same way. Test only.
