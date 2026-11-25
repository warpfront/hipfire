### Changed: tier C arch binds (#1130 D2/D3)

- **Changed:** `HIPFIRE_QWEN4_PLE_FUSE` and `HIPFIRE_QWEN4_MOE_DOWN_R4` (`kernel.qwen4_ple_fuse`, `kernel.qwen4_moe_down_r4`) are now on by default on exact gfx1100 too. Kaden's gfx1100 runs on #1130 show byte-identical output at +0.8 % and +0.9 % decode. `0` stays the kill switch.
- **Changed:** the three `HIPFIRE_VERIFY_GRAPH` default-off quarantines (gfx1100 + MQ*G256V2, gfx1100 + MQ4G256, gfx1201 + MQ4G256) are now rows of the typed table `VERIFY_GRAPH_QUARANTINES` in `hipfire-arch-qwen35`. Each row records its receipt; only gfx1100 + MQ4G256 has one (#695). The behaviour does not change.
