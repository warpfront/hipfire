### New model: Inkling-2-Mini (arch 24)

- **Added: Inkling-2-Mini text generation (arch 24, #1192, raufaser).** New crate
  `hipfire-arch-inkling2` with the `Inkling2Carrier`. It loads the MQ4 (`qt=44`)
  artifact from `Raufaser/Inkling-2-Mini-hipfire-MQ4` (`inkling2-mini-mq4.hfq`, about
  14.6 GiB resident) and serves single-GPU, single-request text through the new
  `InklingAr` route (`hipfire run`, `hipfire serve`, `hipfire bench --backend noslots`).
  The quantizer ingests the checkpoint's NVFP4/MXFP8 packed experts and leaves the MTP
  sidecar (arch id 25, reserved) out of the trunk. Out of scope for now, and refused
  cleanly: speculative decoding (`--spec mtp|dflash`), batching and slots, PP/EP/TP,
  and vision. Reasoning is streamed as visible text (the reasoning contract is
  `Unsupported`). Prefill runs one token at a time, so long prompts are slow, and a
  single prefill of about 3k tokens can run out of VRAM.
