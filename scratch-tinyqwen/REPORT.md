# Tiny Qwen Q8 VMM decode crash (gfx1201)

## Reproducer

`/home/kaden/.hipfire/models/qwen3.5-0.8b.mq4` is the small model. Its HFQ metadata identifies `qwen3_5` text, 24 layers, 8 attention heads, 2 KV heads, head dimension 256, and a 262,144-token context. Card-A (`GPU-9eb7aeda51c88ffd`) loaded it with `--kv-mode q8 --kv-backend vmm`, `HIPFIRE_GRAPH=1`, and a private HOME/kernel cache. The process log confirmed `KV cache: Q8 vmm (` and `physical_cap=262144 / max_seq=262144`.

On unmodified `e5f944a3a`, `hipfire run ... --max-tokens 40 --kv-mode q8 --kv-backend vmm --spec off 'Explain why the sky appears blue in two sentences.'` reached the first decode and the GPU queue aborted in `attention_flash_q8_0_reduce` with `HSA_STATUS_ERROR_INVALID_ALLOCATION`. The HSA hang analysis reported `group_seg_size=65544`: 65,536 bytes requested dynamic LDS plus 8 bytes static LDS. The CLI did not exit after the daemon's GPU fault and was terminated by the 300-second command timeout.

## Root cause and fix

`crates/rdna-compute/src/attention.rs:58-86` selected tile size 16 for gfx12/H8/Hkv2/D256 regardless of the KV reservation. With a 262,144-token physical cache, `max_tiles=ceil(262144/16)=16384`; the plain reducer launch at `crates/rdna-compute/src/attention.rs:7308-7312` allocates `max_tiles * 4 = 65536` dynamic LDS bytes. Its two static shared floats (`kernels/src/attention_flash_q8_0_reduce.hip:48-50`) bring the total to 65,544 bytes, exceeding gfx1201's 64 KiB/workgroup limit. The gated reducer additionally allocates `head_dim` floats (`crates/rdna-compute/src/attention.rs:10317`). This is a reservation/geometry mismatch, not a zero-sized KV allocation or malformed output.

`q8_flash_tile_size` now raises any preferred tile size (including an explicit tuning override) just enough that the maximum graph/replay-stable tile count plus gated-reducer head-dimension scratch fits in 32 KiB LDS. For this cache, it chooses tile 64: 4,096 correction floats (16 KiB plain, 17 KiB gated). Short-context tile16 choices remain unchanged; ordinary larger-head models retain their preferred tile when already safe. The scratch allocation derives from the same resolved tile-size policy, optionally using a smaller tile for conservative capacity, so it cannot underallocate for this launcher. A focused CPU test checks the 2K, 262K, and 1M reservation boundaries.

## Observed post-fix behavior

The native release build and focused tile-capacity test passed. With `reasoning.mode=off` configured in its private HOME, `hipfire run ... --max-tokens 96 --kv-mode q8 --kv-backend vmm --spec off 'Explain why the sky appears blue in two sentences.'` completed successfully on card-A. Its visible text was: “Sunlight travels through the atmosphere, but Rayleigh scattering causes shorter wavelengths of blue light to scatter more easily than longer red ones, which is why the sky turns blue.” The process log confirmed `GPU dev 0: gfx1201`, `KV cache: Q8 vmm (`, and the 262,144-token reservation.

A card-A OpenAI chat completion with `enable_thinking=false` independently returned HTTP 200, `finish_reason=stop`, and a coherent explanation of Rayleigh scattering. The original one-shot attempts without thinking disabled no longer triggered the GPU fault but hit the unrelated open-`<think>` validation at both 40 and 256 tokens; setting the existing `reasoning.mode=off` preference resolved that prompt-framing issue without changing the model, KV configuration, or source code.

A card-A synthetic matrix with `--pp 128,8192 --ctx 128,8192 --tg 16 --runs 1 --warmups 1 --kv-mode q8 --kv-backend vmm --spec off --json` completed: prefill 10,314.60/10,427.50 tok/s at 128/8192, decode 468.58/426.69 tok/s at context 128/8192. JSON reported `kv_backend=vmm`, `kv_mode=q8`, and `max_seq=262144`; process stderr independently logged `KV cache: Q8 vmm (`.

## Card-B 27B baseline guard

`python3 scripts/guard_gfx1201_baseline.py --output-dir scratch-tinyqwen/guard --home /home/kaden/.hipfire-homes/tinyqwen-guard` passed on `GPU-e475645fe0200397`. The pinned 27B fixture was 14,987,185,152 bytes with MD5 `2cfe88923b3671ca16a8de6ec1122fde`; the script verified the actual fp8 VMM backend in each process log. After a discarded fresh-process warmup, three independent pp8192 process medians were 3665.10, 3640.30, and 3636.70 tok/s (all at least the 3620 floor). Their decode medians were 36.5254, 36.5190, and 36.5073 tok/s, all within 1% of 36.5. Raw logs, fixture, and summary reside in the untracked `scratch-tinyqwen/guard/` evidence directory.
