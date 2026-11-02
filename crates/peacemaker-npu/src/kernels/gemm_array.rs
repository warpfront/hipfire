// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//! Whole-array (8 columns x 4 core rows) signed int8 GEMM deployment.
//!
//! Output tile of core (col c, row r) in wave `(mw, nw)` is C rows `(mw*4+r)*128 ..+128`, columns
//! `(nw*8+c)*64 ..+64`; a wave therefore covers 512 x 512. `waves = (Mp/512)*(Np/512)`, wave order
//! `w = mw*NW + nw`. Host M, N are zero padded to 512; padded C is discarded by [`ArrayDesign::unpack_out`].
//!
//! Host arguments (firmware DDR-patches arg0..arg2):
//! * arg0 = A: 4 physical-row segments (`waves*kc*8192` B each), inside a segment wave, K chunk, then
//!   `(mb 0..16, kb 0..8)` 8x8 row-major blocks. The segment is repeated for every `nw` of the same `mw`
//!   (A is not replicated across columns: column `c < 4` memtile fetches segment `c` only).
//! * arg1 = B: 8 column segments (`waves*kc*4096` B each), wave, K chunk, `(kb 0..8, nb 0..8)` blocks
//!   (not replicated across rows).
//! * arg2 = C: 8 column segments (`waves*4*32768` B each), wave, core row, `(mb 0..16, nb 0..8)` blocks of
//!   8x8 row-major little-endian i32.
//!
//! Data movement (deviation from the vendor `032eb944...` design, which uses one memtile channel per A/B
//! stream with multi-dimensional BDs and different port assignment): every stream is a contiguous
//! 1-D BD; the host layout above does the reordering. Switch circuits (all circuit-switched):
//! * shim: `Mm2s0` (B) -> `North0`, `North0` -> `S2mm0` (C); columns `<4` also `Mm2s1` (A) -> `North1`.
//! * memtile: `South0` -> `Dma4` (B S2MM), `South1` -> `Dma5` (A S2MM, col<4), `Dma1` -> `North4` (B),
//!   `Dma2` -> `North5` (A, col<4), `Dma0` -> `South0` (C), `North(r)` -> `Dma(r)` (C from core row r).
//! * core row r: B `South4` -> `Dma1` and `North4` (not the top row); C `Dma0` -> `South(r)` and
//!   `North(s)` -> `South(s)` for s>r; A for row r is injected by column r (`North5` of its memtile), rides
//!   `South5` -> `North5` up column r to row r, is received on `South5` in column r, `East0` in columns <r,
//!   `West0` in columns >r, and is broadcast to `Dma0` plus `West0` (col>0, col<=r) and `East0` (col<7, col>=r).
//!   No switch master is driven by two slaves.
//!
//! Memtile (DMA own view `0x80000 + off`, locks `64 + id`): A ping 0 / pong 0x2000, B ping 0x4000 / pong
//! 0x5000, C `0x6000 + slot*0x20000 + row*0x8000` (<= 0x46000). A/B locks use the core numbering
//! (`gemm_core::A_EMPTY/A_FULL/B_EMPTY/B_FULL`), C slot/row locks are `8 + slot*8 + row*2` (empty, full+1).
//! Channel/BD banks (even channel BD<24, odd >=24): S2MM0 C0 {0,1}, S2MM1 C1 {24,25}, S2MM2 C2 {2,3}, S2MM3
//! C3 {26,27}, S2MM4 B {4,5}, S2MM5 A {28,29}; MM2S0 C chain {6..13}, MM2S1 B {30,31}, MM2S2 A {14,15}.
//! S2MM Finish-on-TLAST (`FoT_MODE`, CTRL bits16..17) remains reset-default 0 / `DMA_FoT_DISABLED`
//! (`ref/aie-rt/driver/src/global/{xaie2pgbl_params.h,xaiegbl.h}`). Completion is length-driven, so the
//! intermediate TLASTs from the joined C descriptors do not terminate the long shim output BD.
//!
//! ## V8 data movement ([`Variant::V8`], [`design_v8`]): vertical pair A-sharing, int8 SRS epilogue
//! V6 gives every core its own 8 KiB A chunk on one 32-bit stream (2048 cycles) against 1067 compute cycles
//! (A-stream bound). In V8 the cores of a vertical pair (core rows 0/1 and 2/3, physical rows 2/3 and 4/5) share
//! one 128-row A block through neighbour data memory, so every input stream of every core and every shim MM2S
//! channel carries exactly 4 KiB (1024 cycles) per K64 chunk.
//! * Geometry: column `c`, core row `rho = 2q + h` (pair `q`, half `h`: lower core `h=0`, upper core `h=1`)
//!   computes C rows `mw*512 + (2q + c%2)*128 ..+128`, columns `nw*512 + (c/2)*128 + h*64 ..+64`. A block
//!   `2q + c%2` is shared by the pair; B block `c/2` is split into its two 64-column halves, one per core of the
//!   pair. A *half* is the 8-row blocks of one parity (E: even `mb`, O: odd `mb`; `mb = 2*mb_local + parity`), per
//!   chunk 8 `mb_local` x 8 `kb` blocks of 8x8 int8 = 4 KiB. The lower core's DMA receives E, the upper core's
//!   DMA receives O; each reads the other half from its neighbour's data memory (the core program, see
//!   [`gemm_core::program_pair`]). The peer lock protocol (`PEER_FULL` / `PEER_EMPTY`) is in `gemm_core`.
//! * Host arguments: arg0 = A, 8 segments `waves*kc*4096` B by *injecting column* `c`
//!   (`(q, h, par) = (c/4, (c/2)%2, c%2)`: the half `h` of A block `2q + par`), each `(wave, chunk, 4096 B)`,
//!   repeated for every `nw` of one `mw`. arg1 = B, 8 segments `waves*kc*4096` B by column (the V6 layout:
//!   segment `c` is the 64-column group `c` = B block `c/2`, half `c%2`), repeated for every `mw`. arg2 = C, 8
//!   column segments, each the shim S2MM0 stream (core rows 0,1: `(wave, rho%2)` tiles) followed by the S2MM1
//!   stream (core rows 2,3); a tile is `(mb 0..16, nb 0..8)` 8x8 blocks of little-endian i32 (`Epilogue::I32`,
//!   32 KiB) or int8 (`Epilogue::Int8`, 8 KiB; saturated `clamp(floor(c / 2^shift), -128, 127)`).
//!   [`ArrayDesign::unpack_out`] sign extends the int8 tiles; [`ArrayDesign::reference`] is the CPU result
//!   with the epilogue applied ([`apply_epilogue`]).
//! * Switch circuits (all circuit-switched, no master has two slaves; lanes are kept through every tile):
//!   - shim: `Mm2s0` (B half) -> `North0`, `Mm2s1` (A half) -> `North1` in all 8 columns; C returns on `North0`
//!     -> `S2mm0` (core rows 0,1) and `North1` -> `S2mm1` (core rows 2,3).
//!   - memtile: `South0 -> Dma4` (B S2MM), `South1 -> Dma5` (A S2MM), `Dma1 -> North4` (B), `Dma2 -> North5` (A),
//!     `North(r) -> Dma(r)` (C of core row r), and two C channels: `Dma0 -> South0` (rows 0,1), `Dma3 -> South1`
//!     (rows 2,3).
//!   - A: column `c` injects on `North5` up its column to core row `rho = c/2`, where `South5` feeds that core's
//!     `Dma0` and broadcasts along the row on horizontal lane `par = c%2` (`East(par)` / `West(par)`); in row
//!     `rho` the lane-`par` stream is taken into `Dma0` by every column `x` with `x%2 == par` and passed through
//!     by the others (tile `(x, rho)` selects `South5` if `x == 2*rho + par`, else `East(par)` for smaller `x`,
//!     `West(par)` for larger `x`). Rows below `rho` pass `South5 -> North5`.
//!   - B: column `c = 2j + h` injects B block `j` half `h` on `North4`; its rows `h` and `h+2` take it into `Dma1`
//!     (`North4` is passed up to row `h+2`). At row `h` it also crosses on horizontal lane 2 (`East2` if `h=0`,
//!     `West2` if `h=1`) into the neighbour column `2j + (1-h)`, which takes it at row `h` into `Dma1` and up
//!     lane 3 (`North3` at row `h`, `South3 -> North3` at row `h+1`, `South3 -> Dma1` at row `h+2`).
//!   - C: core `Dma0 -> South(rho)` down the column (as V6) into memtile S2MM `rho`.
//! * Memtile (offsets as V6): A ping 0 / pong 0x2000 (4 KiB used), B ping 0x4000 / pong 0x5000, C
//!   `0x6000 + slot*0x20000 + row*0x8000` (2 slots; an int8 row buffer is 8 KiB of that 32 KiB stride). Channel / BD
//!   banks (even channel BD<24, odd >=24): S2MM0..3 C rows {0,1}/{24,25}/{2,3}/{26,27}, S2MM4 B {4,5}, S2MM5 A {28,29};
//!   MM2S0 C rows 0,1 chain {6..9} (slot 0 rows 0,1, slot 1 rows 0,1), MM2S3 C rows 2,3 chain {32..35}, MM2S1 B {30,31},
//!   MM2S2 A {14,15}. 24 BDs per memtile.
//! * Shim (per column): BD0 B (arg1) on MM2S0, BD1 C stream 0 (arg2) on S2MM0, BD2 A half (arg0) on MM2S1,
//!   BD3 C stream 1 (arg2, `+ waves*2*tile` bytes) on S2MM1; both S2MM tasks issue a token (controller id set in
//!   both S2MM CTRL registers). The TXN waits for the tokens of both channels of all 8 columns: one aggregate SYNC per
//!   channel (`ncol = 8`). Submit / reset strategy is V6's (dynamic TXN, context reusable). The C drain of the
//!   core is overlapped with the next wave for `Epilogue::Int8` (separate `Cout` buffer) and serialized for
//!   `Epilogue::I32`; see [`ArrayDesign::estimate`] (V8 model: [`gemm_core::PAIR_CHUNK_CYCLES`] per chunk).
//!
//! ## V9 data movement ([`Variant::V9`], [`design_v9`]): resident B, N-wave outer
//! V8 re-reads B from DDR for every wave (`waves*kc*4096` B per column) although B depends on `nw` only. V9 keeps the
//! B tile of one N-wave (`kc*4096` B per column) resident in the memtile and replays it to the cores for all `MW`
//! M-waves of that `nw`; waves therefore run N-wave outer: `w = nw*MW + mw` (`Geometry::wave_coords`). The core
//! programs, core BDs/locks, switch circuits, shim channel assignment and C layout per wave are exactly V8's
//! (`Epilogue::Int8` only, the epilogue the C buffers below are sized for); only the memtile and the host layout change.
//! * Host layout: arg0 = A as V8 but in wave order (per injecting column `(wave, chunk, 4096 B)`, `waves*kc*4096` B);
//!   arg1 = B, per column one segment of `NW*kc*4096` B, `(nw, chunk, (kb, nb) blocks)`, each B tile **once**; arg2 = C as
//!   V8 (two streams of `waves*2*COUT_BYTES` per column) indexed by the wave number `w`. DDR read per submit is
//!   `8*4096*kc*(MW*NW + NW)` B (A `MW*NW`, B `NW` tiles per column) against V8's `8*4096*kc*2*MW*NW`.
//! * Limits: `MW <= 16` (one replay BD per `mw`, 16 free odd-bank BD ids), `NW <= 8`, `kc <= 54` i.e. `K <= 3456`
//!   (memory, below); `Epilogue::Int8` only.
//! * Memtile memory (bytes): A ping 0 / pong 0x2000; C compact `0x4000 + slot*0x8000 + row*0x2000` (8 KiB int8 tile
//!   per row, two slots x four rows = 64 KiB, ends at 0x14000); B `0x14000 + slot*kc*4096` (two slots). The last byte
//!   is `0x14000 + 2*kc*4096 <= 0x80000`, hence `kc <= (0x80000-0x14000)/(2*4096) = 54`. All replay/fill lengths are
//!   `kc*1024` words (<= 55296 < 2^17), one full B slot per BD.
//! * BDs (memtile; even channel BD < 24, odd >= 24): B producer S2MM4 two cyclic BDs 4 / 5, slot 0 / slot 1, each acquiring
//!   `B_EMPTY[0]` (-1) and releasing `B_FULL[0]` (+1). B replay MM2S1: a cyclic chain of `MW` BDs with ids from
//!   `[25, 27, 30, 31, 36..=47]` (the odd-bank ids not used by C producers 24/26, A producer 28/29 and C drain
//!   32..35), first id 25. Every replay BD reads slot 0 with an iteration `step = kc*1024` words (one slot stride /4),
//!   `wrap = 2`, `current = 0`: each time a BD is loaded the hardware advances `current`, so the slot toggles once per
//!   traversal of the whole chain (all BDs advance in lockstep). The first BD acquires `B_FULL[0]` (-1) only, the
//!   last releases `B_EMPTY[0]` (+1) only, the others have no lock action; with `MW = 1` the single BD does both.
//!   C producers: S2MM0/2 (rows 0, 2) keep V8's two-BD rings with distinct slot locks; S2MM1/3 (rows 1, 3) use one BD
//!   (ids 24 / 26) at slot 0 with iteration `step = 0x8000/4`, `wrap = 2`, `next = self` and the row-pair lock
//!   `c_lock(0, row)` (empty, full = empty+1). Their drains (MM2S0 BDs 6..9, MM2S3 BDs 32..35, slot 0 rows `2g,2g+1`
//!   then slot 1) use that same pair for both slots. A rings, core BDs and the A / C drain channels are V8's.
//! * Locks (memtile, initial): `B_EMPTY[0] = 2`, `B_FULL[0] = 0` (slot 1 pair unused/omitted); rows 1, 3
//!   `c_lock(0, row)` empty 2, full 0; rows 0, 2 per-slot pairs empty 1, full 0; A pairs empty 1, full 0.
//! * Ordered-semaphore argument: fill `j` goes to slot `j%2`, replay traversal `t` reads slot `t%2`; fills and
//!   traversals each complete in strict order (one channel each, in-order BD chains). `B_FULL` equals fills completed
//!   minus traversals started, so traversal `t` starts only when fills `0..=t` are complete; `B_EMPTY` equals
//!   `2 - fills started + traversals completed`, so fill `j` starts only when traversal `j-2` (the previous reader
//!   of the same slot) has completed (its last BD, which released `B_EMPTY`, has finished reading). The initial
//!   empty of 2 allows both slots to be prefilled. The same argument covers the single-BD C rows: a shared
//!   empty/full pair is correct because producer slots and drain slots both alternate strictly.
//! * Iteration reset: the iteration `current` lives in the BD word 6 and advances at every BD load; a DMA channel
//!   reset does not clear it. Every dynamic TXN therefore rewrites **all** resident memtile descriptors after the
//!   channel reset and before any queue is written, and the CDO carries the same descriptors. The shim channels
//!   are V8's (B: one BD of `NW*kc*4096` B).
//!
//! ## V10 data movement ([`Variant::V10`], [`design_v10`]): resident A and B, M-wave outer
//! V9 still reads A from DDR once per wave (`MW*NW` A tiles per column). V10 keeps **both** operands resident: each A tile
//! (`kc*4096` B per injecting column, depends on `mw` only) and the whole B segment of the column (`NW*kc*4096` B, one
//! tile per `nw`) are fetched from DDR exactly once per submit. Waves run in V8's order `w = mw*NW + nw`
//! (`Geometry::wave_coords`): for one `mw` the cores see the same A tile `NW` times and the B tiles `nw = 0..NW` in order.
//! Core programs, core BDs/locks, switch circuits, shim channels and the per-wave C layout are V8's / V9's
//! (`Epilogue::Int8` only); the distinct topology mode (`ResidentMode::MwOuter`) changes only the memtile and the host layout,
//! V8 / V9 descriptors, layouts and wave order are unchanged.
//! * Host layout: arg0 = A, per injecting column `(mw, chunk, 4096 B)`, `MW*kc*4096` B per column (`8*MW*kc*4096` total);
//!   arg1 = B, per column `(nw, chunk, (kb, nb) blocks)`, `NW*kc*4096` B (`8*NW*kc*4096` total), each B tile once; arg2 = C as V8
//!   (two streams of `waves*2*COUT_BYTES` per column) indexed by `w = mw*NW + nw`. DDR read per submit (model, bytes):
//!   `8*4096*kc*(MW + NW)` against V9's `8*4096*kc*(MW*NW + NW)` and V8's `8*4096*kc*2*MW*NW`.
//! * Limits: `Epilogue::Int8` only, `MW <= 16`, `NW <= 8` (as V9) and the memory bound `kc*(2 + NW) <= 112`
//!   (`V10_MAX_SLOTS`, below). `K <= 3456` (`kc <= 54`) is not a V10 limit by itself, e.g. `NW = 5` allows `kc <= 16`
//!   (`K <= 1024`); `NW = 1` allows `kc <= 37`. Larger shapes are rejected by [`design_v10`] with an explanatory panic.
//! * Memtile memory (byte offsets): compact C `slot*0x8000 + row*0x2000` (as V9, base 0, two slots x four rows x 8 KiB,
//!   ends `0x10000`); A slot 0 `0x10000`, A slot 1 `0x10000 + kc*4096`; B `0x10000 + 2*kc*4096`, `NW*kc*4096` B.
//!   The last byte is `0x10000 + (2 + NW)*kc*4096 <= 0x80000`, which is exactly `kc*(2 + NW) <= 112`. By the same bound
//!   the whole B segment is `NW*kc*1024 < 112*1024 = 114688` words, below the memtile BD maximum of 131071 words, so one BD
//!   always covers it (the A slot BDs are `kc*1024 <= 37*1024` words).
//! * BDs (memtile; even channel BD < 24, odd >= 24): B fill S2MM4: **one finite BD 4** (no `next`) of `NW*kc*1024` words,
//!   no acquire, release `B_FULL[0]` by `+MW` when the whole segment has landed. B replay MM2S1: **one cyclic BD 30**
//!   (`next = 30`) of the same `NW*kc*1024` words, acquire `B_FULL[0]` (-1), no release: one pass streams `nw = 0..NW` in
//!   core wave order and the lock holds exactly `MW` credits, so the channel makes exactly `MW` passes (the `MW` limit
//!   fits the signed 7-bit release value). A fill S2MM5: ring BDs 28 / 29 (slot 0 / 1) of `kc*1024` words each, acquire
//!   `A_EMPTY[0]` (-1), release `A_FULL[0]` (+1). A replay MM2S2: a cyclic chain of `NW` BDs 14..14+NW (even bank), every BD
//!   reads one full slot (`kc*1024` words) starting at slot 0 with iteration `step = kc*1024` words, `wrap = 2`,
//!   `current = 0`, so the slot toggles once per chain traversal (all BDs advance in lockstep, one load each); the first
//!   BD acquires `A_FULL[0]` (-1), the last releases `A_EMPTY[0]` (+1) (one BD: both). C producers (S2MM0..3: rings 0/1,
//!   2/3 and single iteration BDs 24, 26 with the row-pair lock) and C drains (MM2S0 BDs 6..9, MM2S3 BDs 32..35) are V9's
//!   descriptors with base 0.
//! * Locks (memtile, initial): `A_EMPTY[0] = 2`, `A_FULL[0] = 0`, `B_FULL[0] = 0` (B empty and the slot-1 A pair are
//!   unused); C locks as V9 (rows 1, 3 pair empty 2; rows 0, 2 per slot empty 1).
//! * Ordered-semaphore argument for A is V9's for B: fill `j` goes to slot `j%2`, traversal `t` reads slot `t%2`;
//!   `A_FULL` = fills completed - traversals started, `A_EMPTY` = 2 - fills started + traversals completed. A traversal is
//!   `NW` consecutive BDs of one chain that read the same slot without further synchronisation: the slot is only released
//!   after the last BD finished reading it, so the fill two steps later cannot overwrite it. B is written once before the
//!   first pass and read-only afterwards.
//! * DDR / shim: shim B BD 0 is one BD of `NW*kc*4096` B, shim A BD 2 one BD of `MW*kc*4096` B, both finite and read
//!   once; the A fill stalls the A lane when both slots are full, B and C use other lanes. B replay starts only after the
//!   whole B segment is resident, so the first core wave cannot begin before the B fill completed (modelled startup in
//!   [`ArrayDesign::estimate`]).
//! * Reset: BD word 6 `current` of the A replay chain advances at every load and a channel reset does not clear it, and
//!   the finite fill BD 4 has no ring to restart; every dynamic TXN therefore rewrites **all** resident memtile
//!   descriptors after the channel reset and before any queue is written (`build_txn_dynamic`), the CDO carries the same
//!   descriptors. The persistent channels (incl. the cyclic B replay left blocked on `B_FULL` after its `MW` passes) are
//!   reset by the TXN at the next submit, which also re-initialises every memtile lock.
//!
//! ## Opt-in V10 B prefix ([`design_v10_prefix`])
//! Retained prefix: BD 4 / 5 fill and guarded BD 30 / 31 read the B segment chunk by chunk in groups of at most 64 chunks;
//! `B_FULL` is credited +1 per chunk by the fill and -1 per chunk by the guarded read, and above 63 chunks a `B_EMPTY`
//! credit window of 63 bounds the `B_FULL` count. After the guarded pass a lockfree whole-segment BD 36 replays B
//! `MW - 1` times, queued only when `MW > 1`, and every submit rewrites all of it. It changes only the memtile
//! descriptors, the memtile locks and the queued memtile tasks of a resident design; core programs, core BDs / locks, shim
//! BDs, switch circuits, host layouts, wave order and `Program::finish` are unchanged. `design_v10` (`b_prefix: false`)
//! emits exactly the descriptors, locks and tasks described above. CDO and TXN take the same descriptors
//! (`memtile_descriptors`), locks (`memtile_locks`) and tasks (`ring_tasks`); the TXN rewrites every descriptor after the
//! channel reset, so a resubmit never depends on iteration `current`, a stopped finite BD or a lock left by the last run.
//! * V10 B prefix: the B fill / first replay pass is chunk granular instead of one whole-segment BD, so the first wave
//!   starts when the first B chunk landed. The segment has `T = NW*kc` chunks (`T <= 88` under `kc*(2 + NW) <= 112`; the
//!   code accepts `T <= 128`) of `1024` words each, split into groups of at most 64 chunks (DMA iteration wrap limit). Group
//!   `g` (first chunk `f`, length `L`, `g = 0, 1`): finite producer BD `4 + g` (S2MM4) and finite guarded reader BD
//!   `30 + g` (MM2S1), each `1024` words at `B + f*4096`, iteration `step = 1024`, `wrap = L`, `current = 0`, no `next`;
//!   queued tasks `repeat = L`, group order, producer 1..=2 tasks. Producer releases `B_FULL[0]` (+1) per chunk, reader
//!   acquires `B_FULL[0]` (-1), so the reader can only read landed chunks. Then one lockfree finite BD 36 of `T*1024` words
//!   (< 131072 by the memory bound) is queued with `repeat = MW - 1` only when `MW > 1`; it follows the guarded tasks in the same
//!   channel queue so it starts after the last chunk was read, B is immutable afterwards and needs no
//!   lock. Queue depth: AIE2P memtile `StartQSizeMax = 4`, `MaxRepeatCount = 256`
//!   (`ref/aie-rt/driver/src/global/xaie2pgbl_reginit.c`); at most 2 S2MM and 3 MM2S tasks are queued, asserted at design time
//!   (`groups + (MW > 1) <= 4`) because the simulator queue is unbounded.
//!   A lock holds at most 63, so for `T > 63` only: `B_EMPTY[0] = 63` initially, the producer acquires it (-1) and the
//!   guarded reader releases it (+1), bounding `B_FULL` credits to 63. For `T <= 63` the producer has no acquire and the reader
//!   no release. A and C are V10's. `used_channels` deduplicates channels that carry more than one queued task, the reset
//!   order is unchanged.
//!
//! ## G80 ([`Variant::G80`], [`gemm_g80`]): 128x80 core tile, N = 1280 exact
//! Vertical-pair A sharing as V8 on the 128x80 core of [`gemm_core_g80`]: a wave is 256 rows x 1280 columns (core row `r`
//! of column `c` computes M tile `r/2`, N tile `2c + r%2` of 80 columns); only columns 0..4 inject A. This file only
//! dispatches (`Kind::G80`): circuits, memtile BDs / locks / queued tasks, host packing and the estimate live in
//! `gemm_g80`; CDO, TXN, shim BDs and the `emit_dynamic_body` reset / requeue sequence are shared with V8..V10, and every
//! submit rewrites all G80 memtile BDs after the channel reset. `Epilogue::I32` (raw 40,960 B tile per core, host unpack
//! row-major) uses six core DMA BDs (two 20,480 B stripes) and one 20,480 B memtile C stage per row drained as two stripes,
//! with finite C producer / drain tasks (`gemm_g80` module header).
//!
//! ## IEF15 ([`Variant::Ief15`], [`iu4_ief15`]): 64x32 core tile, 256 x 256 wave
//! The V8 pair structure unchanged (circuits [`pair_column_circuits`], memtile map, BD ids, locks, tasks, lean parity plan with
//! `J = epochs * waves`, `W = waves`) on the [`iu4_ief15_core`] tile: one A half / B chunk of 2176 B per K128 epoch (`Kind::Ief15`
//! with `kc = epochs`), f32 `Cout` tile of 8192 B (the V8 int8 `COUT_BYTES`, so `pair_c_offset` is unchanged). Only the ring BD
//! lengths (`iu4_ief15::memtile_descriptors`), the core BDs (`iu4_ief15_core::tile_bds`), the host segment sizes and the core
//! programs differ; packing, unpacking and the reference are `iu4_ief15`'s, `Geometry::pack_a` / `pack_b` / `unpack_c` reject it.
//!
//! ## Submit / context reuse strategy
//! The PDI (CDO) only loads state that survives a run: core program memory, all tile/memtile BDs, switch
//! routes, shim NOC mux and token route. It does NOT enqueue any task, initialize any lock, or enable a core.
//! Every submit (TXN) starts from the same state, so a context can be resubmitted any number of times,
//! including odd `kc*waves` (where the A/B ping-pong rings would otherwise stop on the pong slot):
//! 1. halt + reset every core (`CORE_CONTROL` mask 3 <- reset);
//! 2. assert then deassert reset (CTRL bit1, ref/aie-rt `xaie2pgbl_params.h`) on every compute-tile and
//!    memtile persistent-ring channel - this clears queued BDs, the active BD and the ring position, so rings
//!    restart at ping / slot 0 / BD6. Shim channels are NOT reset (aie-rt rejects shim reset; shim CTRL bit1 is
//!    pause-mem): shim queues are finite and have drained before the C token that ended the previous submit;
//! 3. initialize every core and memtile lock to its initial value (no DMA is active at this point);
//! 4. deassert core reset, `CORE_PC <- 0`;
//! 5. queue core and memtile rings (cyclic BD chains, repeat 1), enabling compute channels only (memtile
//!    queues start on enqueue); set the shim token controller id, rewrite finite BDs, DDR-patch and queue;
//! 6. enable all cores last, then wait for the C token of all 8 shim S2MM0 channels.
//! Assumes the previous submit completed (its sync returned), so nothing is in flight in the switches.
//!
//! ## Deviations from the vendor whole-array design (`032eb944...`), see also module header
//! * The vendor streams full-K operands with multi-dimensional BDs and `ObjectFifo` repeat; here every L3/L2
//!   transfer is a plain contiguous 1-D BD and the host pre-packs A/B ([`ArrayDesign::pack_in`]) into the exact
//!   order the cores consume, so the shim reads each byte exactly once per use and the cores stream K in
//!   64-element chunk buffers (contract local buffers: A 8 KiB x2, B 4 KiB x2, C resident 32 KiB).
//! * Routes are regular (same B/C lanes in every column, A on a dedicated lane 5 and per-row East/West
//!   broadcast) rather than the vendor's placement-driven port assignment.
//! * The persistent rings are not started from the CDO; every submit restarts them from the TXN so a context
//!   can be resubmitted regardless of the parity of `kc*waves`.
//! * The existing `dma::vendor_gate` and route structural oracle gates validate the BD/task and switch
//!   encoders against the vendor artifacts; they do not assert equality of this changed deployment.
//!
//! ## Hardware bisect variants ([`Variant`], [`design_variant`])
//! Hardware run 1 of the whole-array deployment (V6) timed out; the variants narrow it down. All use the same
//! core program, the same three arguments (A, B, C) and the same `pack_in`/`unpack_out` API; the wave is the
//! variant's own tile set (`rows*128 x cols*64`), M/N are padded to it, and `output_tile_ranges` reports the
//! packed-C byte ranges written by each active physical tile.
//!
//! | variant | shape | active tiles | data movement | configuration |
//! |---|---|---|---|---|
//! | V1 | 128x64 | core (0,2) | shim <-> memtile switch pass-through <-> core, same lane through the memtile (AIE2P North/South pass-through must keep the lane): B `South0->North0->South0->Dma1`, A `South1->North1->South1->Dma0`, C `Dma0->South0->North0->shim` | static CDO, shim-only TXN |
//! | V2 | 128x64 | core (0,2) | as the array but 1 row/1 column: memtile DMA rings for A, B, C (2 C slots, 2-BD C drain chain) | static |
//! | V3 | 512x64 | column 0, rows 2..5 | one memtile, no helper columns (below) | static |
//! | V4 | 128x512 | row 2, columns 0..7 | array routes restricted to row 0: A injected by column 0 and broadcast East, B per column, 2-BD C drain chain | static |
//! | V5 | 512x512 | full 8x4 | as V6 | static CDO, shim-only TXN |
//! | V6 | 512x512 | full 8x4 | as designed above | dynamic TXN (resets + requeues), one aggregate SYNC |
//! | V6b | 512x512 | full 8x4 | as V6 | dynamic TXN, 8 separate SYNCs |
//! | V8 | 512x512 | full 8x4 | V8 data movement (above), not in [`Variant::ALL`] | dynamic TXN, 2 aggregate SYNCs, `design_v8` |
//! | V9 | 512x512 | full 8x4 | V9 data movement (above): resident B, N-wave outer, not in [`Variant::ALL`] | dynamic TXN, 2 aggregate SYNCs, `design_v9` |
//! | V10 | 512x512 | full 8x4 | V10 data movement (above): resident A and B, M-wave outer, not in [`Variant::ALL`] | dynamic TXN, 2 aggregate SYNCs, `design_v10` |
//! | G80 | 256x1280 | full 8x4 | G80 data movement ([`gemm_g80`], N = 1280, `K <= 2560`), not in [`Variant::ALL`] | dynamic TXN, 2 aggregate SYNCs, `gemm_g80::design_g80` |
//! | G80 N512 | 512x512 | full 8x4 | V8 data movement on the 128x64 BF16 core with the whole B panel resident ([`gemm_g80_n512`], N = 512, `K <= 6336`), `Kind::G80 { n512: true }`, not in [`Variant::ALL`] | dynamic TXN, 2 aggregate SYNCs, `gemm_g80_n512::design_g80_n512` |
//!
//! *Static* (V1..V5): the CDO additionally loads core/memtile locks, queues and enables every core and memtile
//! ring, releases core reset, sets PC 0 and enables the cores last (as `gemm_i8::design` does); the TXN only
//! has the shim token controller id, shim BDs + DDR patches, shim tasks and one `SYNC(col, ncol=1, nrow=1)`
//! per active column (V6: one `SYNC(0, ncol=8)`). Static designs are valid for a fresh context only (a second
//! submit would find drained, non-restarted rings).
//!
//! V3 one-column layout (36 memtile BDs, 36 locks, 328 KiB): shim MM2S0 streams A interleaved as
//! `(wave, chunk, core row 0..3)` into memtile S2MM4, an 8-BD producer chain (slot 0 rows 0..3, slot 1 rows
//! 0..3, A at `slot*0x8000 + row*0x2000`, locks `slot*8+row*2`); memtile MM2S0..3 are the per-row A consumers
//! (2-BD rings) onto switch lanes `North(row)`, which core tiles pass up `South(r) -> North(r)` until row `r`
//! (`South(r) -> Dma0`). Shim MM2S1 streams B into S2MM5 (ping `0x10000`/pong `0x11000`, locks 16..19), MM2S4
//! broadcasts it on `North4` up the column. Core row `r` C returns on lane `South(r)` into S2MM r (C at
//! `0x12000 + slot*0x20000 + row*0x8000`, locks `20 + slot*8 + row*2`); MM2S5 joins slot 0 rows 0..3, slot 1
//! rows 0..3 (8-BD chain) onto `South0` to shim S2MM0. DMA ports are duplex (A out `Dma(r)` vs C in `Dma(r)`,
//! A in `Dma4` vs B out `Dma4`, B in `Dma5` vs C join `Dma5`). BDs: even channels `0..8` A producer (ch4),
//! `8/9` C row 0, `10/11` C row 2, `12/13` A row 0, `14/15` A row 2, `16/17` B consumer (ch4); odd channels
//! `24/25` B producer (ch5), `26/27` C row 1, `28/29` C row 3, `30/31` A row 1, `32/33` A row 3, `34..41` C join.
//! The V3 A stream is one argument segment (not four row segments): `pack_in` interleaves chunk then row.
//!
//! ## Core program variants ([`CoreVariant`], [`design_variant_core`])
//! Second hardware bisect: on hardware every [`Variant`] timed out with the compute cores finished (`CoreDone`)
//! but the C-full lock never released. [`design_variant_core`] swaps only the program loaded into the active cores
//! ([`gemm_core::program_variant`]); DMA rings, locks, switches, host layouts, the CDO/TXN structure and
//! [`ArrayDesign::output_tile_ranges`] are those of the [`Variant`]. [`design_variant`] / [`design`] always use
//! [`CoreVariant::Fast`] (the default program).
//! * `Fast`: the default optimized program, exact GEMM.
//! * `Serial`: exact GEMM (identical C to `Fast`) with a serialized program; [`ArrayDesign::unpack_out`] applies.
//! * `LockOnly`: no multiply. Per physical tile and wave it acquires/releases every A/B chunk and then the full C
//!   buffer, and fills the first 64 little-endian `i32` words of that tile's packed C frame (the range reported by
//!   [`ArrayDesign::output_tile_ranges`]) with `0xC0DE0000 + wave*0x100 + word`, where `wave` is the zero-based
//!   wave of the design. The remaining words of the frame are unspecified. The output is NOT a GEMM, so
//!   `unpack_out` is meaningless; check the per-tile frame prefix instead.
use super::gemm_core::{self, Control, CoreVariant, Epilogue, PairRole};
use super::gemm_i8::{peak_ops_per_second, ArgKind, ArgSpec};
use super::{gemm_core_g80, gemm_g80, gemm_g80_n512, iu4_ief15, iu4_ief15_core};
use crate::{
    cdo::Cdo,
    dma::{
        self, Bd, BdLocks,
        Direction::{Mm2s, S2mm},
        Iteration, Location, Task, TileType,
    },
    regs,
    route::{Circuit, Port, ShimDma},
    txn::Txn,
};

pub const COLS: usize = 8;
pub const ROWS: usize = 4;
pub const TM: usize = gemm_core::TM as usize;
pub const TN: usize = gemm_core::TN as usize;
pub const CHUNK_K: usize = gemm_core::CHUNK_K as usize;
pub const A_BYTES: usize = gemm_core::A_BYTES as usize;
pub const B_BYTES: usize = gemm_core::B_BYTES as usize;
pub const C_BYTES: usize = gemm_core::C_BYTES as usize;
/// Rows / columns covered by one wave.
pub const WAVE_M: usize = ROWS * TM;
pub const WAVE_N: usize = COLS * TN;
pub const MAX_WAVES: usize = 256;
pub const MAX_KC: usize = gemm_core::MAX_KC;
const MB: usize = TM / 8;
const NB: usize = TN / 8;
const KB: usize = CHUNK_K / 8;
const _: () = assert!(A_BYTES == TM * CHUNK_K && B_BYTES == CHUNK_K * TN && C_BYTES == TM * TN * 4);
const _: () = assert!(TM % 8 == 0 && TN % 8 == 0 && CHUNK_K % 8 == 0);
/// G80 wave: two 128-row M tiles x sixteen 80-column N tiles; A is injected by columns `0..G80_A_COLS` only.
pub(crate) const G80_WAVE_M: usize = 2 * gemm_core_g80::TM;
pub(crate) const G80_WAVE_N: usize = 2 * COLS * gemm_core_g80::TN;
pub(crate) const G80_A_COLS: usize = 4;
/// G80 N512 wave ([`gemm_g80_n512`]): the V8 geometry on the 128x64 BF16 core, four 128-row A blocks x eight 64-column
/// B halves per column pair; every column injects an A half and a B half (V8 data movement, B resident).
pub(crate) const G80_N512_WAVE_M: usize = ROWS * gemm_core_g80::TM;
pub(crate) const G80_N512_WAVE_N: usize = COLS * TN;
/// IEF15 wave: 4 A blocks of 64 tokens x 4 B blocks of 64 features ([`iu4_ief15`], core tile 64 x 32); every column injects A.
pub(crate) const IEF15_WAVE: usize = 4 * iu4_ief15_core::TM;

/// Memtile DMA own-memory view base.
pub(crate) const MEM_BASE: u32 = 0x80000;
/// Memtile lock ids are 64 + id in a BD.
pub(crate) const MEM_LOCK: u32 = 64;
const A_OFF: [u32; 2] = [0x0000, 0x2000];
const B_OFF: [u32; 2] = [0x4000, 0x5000];
const C_OFF: u32 = 0x6000;
const C_SLOT_STRIDE: u32 = 0x20000;
const C_ROW_STRIDE: u32 = 0x8000;
const C_LOCK0: u32 = 8;
const A_S2MM_BD: [u32; 2] = [28, 29];
const A_MM2S_BD: [u32; 2] = [14, 15];
const B_S2MM_BD: [u32; 2] = [4, 5];
const B_MM2S_BD: [u32; 2] = [30, 31];
const C_S2MM_BD: [[u32; 2]; ROWS] = [[0, 1], [24, 25], [2, 3], [26, 27]];
const C_MM2S_BD0: u32 = 6;
/// Memtile channels: S2MM0..3 C rows, S2MM4 B, S2MM5 A; MM2S0 C, MM2S1 B, MM2S2 A.
const B_S2MM_CH: u32 = 4;
const A_S2MM_CH: u32 = 5;
const B_MM2S_CH: u32 = 1;
const A_MM2S_CH: u32 = 2;

// V8 (vertical pair A-sharing): A half per K64 chunk, memtile->shim C channels.
pub const A_HALF_BYTES: usize = gemm_core::A_HALF_BYTES;
/// C bytes of one core tile with the int8 epilogue (`Cout`).
pub const COUT_BYTES: usize = gemm_core::COUT_BYTES;
/// V8 memtile C drain: MM2S0 drains core rows 0,1 (BDs 6..9), MM2S3 core rows 2,3 (BDs 32..35); each cyclic
/// chain visits slot 0 rows `2g, 2g+1` then slot 1 rows `2g, 2g+1`.
const PAIR_C_MM2S_CH: [u32; 2] = [0, 3];
const PAIR_C_MM2S_BD0: [u32; 2] = [6, 32];
const _: () = assert!(A_HALF_BYTES == TM * CHUNK_K / 2 && COUT_BYTES == TM * TN);

// V9 (resident B, N-wave outer): memtile layout.
/// Compact C buffers (int8 tile per row), `slot*0x8000 + row*0x2000`.
const R_C_OFF: u32 = 0x4000;
const R_C_SLOT_STRIDE: u32 = 0x8000;
const R_C_ROW_STRIDE: u32 = 0x2000;
/// Two resident B slots of `kc*4096` B follow the C buffers.
const R_B_OFF: u32 = 0x14000;
/// Largest `kc` of V9: two B slots must fit the 512 KiB memtile (`(0x80000 - 0x14000) / (2*4096)`).
pub const V9_MAX_KC: usize = (0x80000 - R_B_OFF as usize) / (2 * B_BYTES);
/// Largest number of M-waves / N-waves of V9.
pub const V9_MAX_MW: usize = 16;
pub const V9_MAX_NW: usize = 8;
/// B replay chain BD ids (odd bank, MM2S1): free ids of 24..47 after C producers 24/26, A producer 28/29 and
/// C drain 32..35.
const R_B_REPLAY_BD: [u32; V9_MAX_MW] = [
    25, 27, 30, 31, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
];
const _: () = assert!(V9_MAX_KC == 54 && COUT_BYTES == R_C_ROW_STRIDE as usize);
const _: () = assert!(
    R_C_OFF + 2 * R_C_SLOT_STRIDE == R_B_OFF && R_C_SLOT_STRIDE == ROWS as u32 * R_C_ROW_STRIDE
);
fn resident_c_offset(base: u32, slot: usize, row: usize) -> u32 {
    base + slot as u32 * R_C_SLOT_STRIDE + row as u32 * R_C_ROW_STRIDE
}
fn resident_b_offset(kc: usize, slot: usize) -> u32 {
    R_B_OFF + (slot * kc * B_BYTES) as u32
}

// V10 (resident A and B, M-wave outer): memtile layout. C uses the V9 compact buffers at base 0 (ends 0x10000).
const V10_C_OFF: u32 = 0;
/// A slot 0; slot 1 follows at `+kc*4096`, then the whole B segment.
const V10_A_OFF: u32 = 0x10000;
/// Memory bound of V10: `kc*(2 + NW)` slots of 4096 B between `V10_A_OFF` and the memtile end.
pub const V10_MAX_SLOTS: usize = (0x80000 - V10_A_OFF as usize) / B_BYTES;
pub const V10_MAX_MW: usize = 16;
pub const V10_MAX_NW: usize = 8;
/// First BD id of the A replay chain (`NW` consecutive even-bank ids 14..=21).
const V10_A_CHAIN_BD0: u32 = A_MM2S_BD[0];
const _: () = assert!(V10_C_OFF + 2 * R_C_SLOT_STRIDE == V10_A_OFF && V10_MAX_SLOTS == 112);
const _: () = assert!(A_HALF_BYTES == B_BYTES && V10_A_CHAIN_BD0 as usize + V10_MAX_NW <= 24);
fn mw_outer_a_offset(kc: usize, slot: usize) -> u32 {
    V10_A_OFF + (slot * kc * B_BYTES) as u32
}
fn mw_outer_b_offset(kc: usize) -> u32 {
    V10_A_OFF + (2 * kc * B_BYTES) as u32
}
// Opt-in V10 B prefix: finite chunk-granular producer / guarded reader groups (iteration wrap <= 64) then a lockfree replay.
const V10_B_GROUP_MAX: usize = 64;
/// Largest memtile lock value; `B_FULL` credits stay at or below it when the prefix has more chunks.
const V10_B_CREDIT_MAX: u32 = 63;
/// Whole-segment lockfree B replay BD (odd bank, MM2S1; free in V10).
const V10_B_LOCKFREE_BD: u32 = 36;
/// Memtile channel start-queue depth (aie-rt `StartQSizeMax` of the AIE2P memtile).
const V10_B_QUEUE_MAX: usize = 4;
const _: () = assert!(V10_B_GROUP_MAX <= 64 && V10_B_LOCKFREE_BD >= 24 && V10_B_LOCKFREE_BD < 48);
pub(crate) fn core_loc(col: u32, row: usize) -> Location {
    Location::new(col, row as u32 + 2)
}
pub(crate) fn mem_loc(col: u32) -> Location {
    Location::new(col, 1)
}
pub(crate) fn shim_loc(col: u32) -> Location {
    Location::new(col, 0)
}
fn c_lock(slot: usize, row: usize) -> u32 {
    C_LOCK0 + (slot * 8 + row * 2) as u32
}
fn c_offset(slot: usize, row: usize) -> u32 {
    C_OFF + slot as u32 * C_SLOT_STRIDE + row as u32 * C_ROW_STRIDE
}

// One-column layout (V3): all four core rows of column 0 behind a single memtile.
// Memtile memory: A `slot*0x8000 + row*0x2000`, B ping/pong, C `0x12000 + slot*0x20000 + row*0x8000`.
const OC_B_OFF: [u32; 2] = [0x10000, 0x11000];
const OC_C_OFF: u32 = 0x12000;
/// Memtile channels: S2MM0..3 C rows, S2MM4 interleaved A, S2MM5 B; MM2S0..3 per-row A, MM2S4 B, MM2S5 C join.
const OC_A_PROD_CH: u32 = 4;
const OC_B_PROD_CH: u32 = 5;
const OC_B_CONS_CH: u32 = 4;
const OC_C_JOIN_CH: u32 = 5;
/// Even channels use BD<24, odd >=24 (A producer chain 0..8, C join chain 34..42).
const OC_A_PROD_BD0: u32 = 0;
const OC_C_PROD_BD: [[u32; 2]; ROWS] = [[8, 9], [26, 27], [10, 11], [28, 29]];
const OC_A_CONS_BD: [[u32; 2]; ROWS] = [[12, 13], [30, 31], [14, 15], [32, 33]];
const OC_B_PROD_BD: [u32; 2] = [24, 25];
const OC_B_CONS_BD: [u32; 2] = [16, 17];
const OC_C_JOIN_BD0: u32 = 34;
fn oc_a_offset(slot: usize, row: usize) -> u32 {
    slot as u32 * 0x8000 + row as u32 * 0x2000
}
fn oc_a_lock(slot: usize, row: usize) -> u32 {
    (slot * 8 + row * 2) as u32
}
const OC_B_EMPTY: [u32; 2] = [16, 18];
const OC_B_FULL: [u32; 2] = [17, 19];
fn oc_c_offset(slot: usize, row: usize) -> u32 {
    OC_C_OFF + slot as u32 * C_SLOT_STRIDE + row as u32 * C_ROW_STRIDE
}
fn oc_c_lock(slot: usize, row: usize) -> u32 {
    20 + (slot * 8 + row * 2) as u32
}

/// Deployment variant of the hardware bisect (see the module header). `V6` is the deployment that
/// timed out on hardware (dynamic TXN, aggregate SYNC); the others narrow it down.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Variant {
    V1,
    V2,
    V3,
    V4,
    V5,
    V6,
    V6b,
    V8,
    V9,
    V10,
    G80,
    Ief15,
}

impl Variant {
    /// The V1..V6b bisect set. [`Variant::V8`], [`Variant::V9`] and [`Variant::V10`] are the pair-sharing production designs
    /// and are deliberately not part of this list (they need an epilogue and take a different core program).
    pub const ALL: [Variant; 7] = [
        Variant::V1,
        Variant::V2,
        Variant::V3,
        Variant::V4,
        Variant::V5,
        Variant::V6,
        Variant::V6b,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Variant::V1 => "V1",
            Variant::V2 => "V2",
            Variant::V3 => "V3",
            Variant::V4 => "V4",
            Variant::V5 => "V5",
            Variant::V6 => "V6",
            Variant::V6b => "V6b",
            Variant::V8 => "V8",
            Variant::V9 => "V9",
            Variant::V10 => "V10",
            Variant::G80 => "G80",
            Variant::Ief15 => "IEF15",
        }
    }

    /// Exact `(m, n)` the variant is designed for (one wave); `design_variant` also accepts other shapes
    /// (more waves, padding to the variant's own wave size).
    pub fn shape(self) -> (usize, usize) {
        let t = self.topo();
        (t.wave_m(), t.wave_n())
    }

    fn topo(self) -> Topo {
        match self {
            Variant::V1 => Topo {
                rows: 1,
                cols: 1,
                kind: Kind::Pass,
            },
            Variant::V2 => Topo {
                rows: 1,
                cols: 1,
                kind: Kind::Staged,
            },
            Variant::V3 => Topo {
                rows: ROWS,
                cols: 1,
                kind: Kind::Column,
            },
            Variant::V4 => Topo {
                rows: 1,
                cols: COLS,
                kind: Kind::Staged,
            },
            Variant::V5 | Variant::V6 | Variant::V6b => Topo::FULL,
            Variant::V8 | Variant::V9 | Variant::V10 => Topo::pair(Epilogue::I32),
            Variant::G80 => Topo::g80(V8_DEFAULT_EPILOGUE, Control::Fast, 1, 1),
            Variant::Ief15 => Topo::ief15(Control::Fast, 1, 1, 1),
        }
    }

    /// Everything (core programs, DMA rings, locks, core enables) is configured by the CDO and the TXN only
    /// drives the shim; valid for a fresh context only.
    fn is_static(self) -> bool {
        !matches!(
            self,
            Variant::V6
                | Variant::V6b
                | Variant::V8
                | Variant::V9
                | Variant::V10
                | Variant::G80
                | Variant::Ief15
        )
    }

    /// One `SYNC(col, ncol=1, nrow=1)` per active column instead of one aggregate `ncol=cols` SYNC.
    fn per_column_sync(self) -> bool {
        !matches!(
            self,
            Variant::V6 | Variant::V8 | Variant::V9 | Variant::V10 | Variant::G80 | Variant::Ief15
        )
    }
}

impl std::fmt::Display for Variant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

impl std::str::FromStr for Variant {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        Variant::ALL
            .into_iter()
            .chain([Variant::V8, Variant::V9, Variant::V10, Variant::G80])
            .find(|v| v.name().eq_ignore_ascii_case(s))
            .ok_or_else(|| {
                format!(
                    "unknown variant {s:?}, expected one of V1 V2 V3 V4 V5 V6 V6b V8 V9 V10 G80"
                )
            })
    }
}

/// Memtile-resident operand data movement of a pair topology (distinct from the V8 streaming topology).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ResidentMode {
    /// V9: B per N-wave resident, N-wave outer wave order `w = nw*MW + mw`.
    NwOuter,
    /// V10: A per M-wave and the whole B segment resident, M-wave outer wave order `w = mw*NW + nw`.
    MwOuter,
}

/// Resident-operand parameters carried by the topology: `kc` K64 chunks per tile, `mw` M-waves, `nw` N-waves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Resident {
    mode: ResidentMode,
    kc: usize,
    mw: usize,
    nw: usize,
    b_prefix: bool,
}

impl Resident {
    /// Memtile offset of the compact C buffers.
    fn c_base(self) -> u32 {
        if self.mode == ResidentMode::MwOuter {
            V10_C_OFF
        } else {
            R_C_OFF
        }
    }
    /// Chunk groups `(first chunk, length)` of the V10 B prefix: at most `V10_B_GROUP_MAX` chunks each, one per BD pair.
    fn b_prefix_groups(self) -> impl Iterator<Item = (usize, usize)> {
        let total = self.nw * self.kc;
        let count = total.div_ceil(V10_B_GROUP_MAX);
        assert!(
            count <= B_S2MM_BD.len(),
            "B prefix of {total} chunks needs {count} groups, only {} BD pairs",
            B_S2MM_BD.len()
        );
        (0..total)
            .step_by(V10_B_GROUP_MAX)
            .map(move |first| (first, (total - first).min(V10_B_GROUP_MAX)))
    }
    /// Number of B prefix chunk groups (before the `B_S2MM_BD` limit is checked).
    fn b_prefix_group_count(self) -> usize {
        (self.nw * self.kc).div_ceil(V10_B_GROUP_MAX)
    }
    /// The B prefix holds more chunks than a memtile lock can count, so `B_EMPTY[0]` bounds the `B_FULL` credits.
    fn b_prefix_backpressure(self) -> bool {
        self.nw * self.kc > V10_B_CREDIT_MAX as usize
    }
}

/// Data-movement structure of a topology.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kind {
    /// Shim <-> core pass-through through the memtile switch, no memtile DMA (single core only).
    Pass,
    /// Memtile-staged rings, A of core row `r` injected by column `r` (`rows <= cols`).
    Staged,
    /// Staged, one physical column: A of all rows interleaved on one shim channel (`cols == 1`).
    Column,
    /// V8: vertical core pairs (rows 0/1, 2/3) share one A block through neighbour data memory; every column
    /// injects an A half and a B half; two memtile->shim C channels. `epi` fixes the C tile size; `ctl` fixes the
    /// core control discipline and data-memory layout (`Control::probe().layout`, see `gemm_core::PairLayout`);
    /// `resident` is the V9 / V10 memtile-resident data movement.
    Pair {
        epi: Epilogue,
        ctl: Control,
        resident: Option<Resident>,
    },
    /// G80: the V8 pair structure on the 128x80 core ([`gemm_core_g80`], int8 or i32 epilogue). `kc` K64 chunks per tile and
    /// `mw` M-waves fix the memtile B segments and replay counts (N is always one wave of 1280 columns). `n512`: the N=512
    /// BF16 geometry of [`gemm_g80_n512`] (V8 circuits and wave of 512 x 512, 128x64 tiles, resident K<=6336 B panel).
    G80 {
        epi: Epilogue,
        ctl: Control,
        kc: usize,
        mw: usize,
        bf16_layout: gemm_g80::Bf16Layout,
        n512: bool,
    },
    /// IEF15 ([`iu4_ief15`]): the V8 pair structure and memtile map on the 64x32 [`iu4_ief15_core`] tile (A/B chunks of 2176 B
    /// per K128 epoch, f32 `Cout` of 8192 B). `epochs` K128 epochs and `mw` x `nw` waves of 256 x 256 fix the host segments.
    Ief15 {
        ctl: Control,
        epochs: usize,
        mw: usize,
        nw: usize,
    },
}

/// Active tile set: core rows `0..rows` x columns `0..cols` (physical rows `2..2+rows`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Topo {
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) kind: Kind,
}

impl Topo {
    const FULL: Topo = Topo {
        rows: ROWS,
        cols: COLS,
        kind: Kind::Staged,
    };
    fn pair(epi: Epilogue) -> Topo {
        Topo::pair_control(epi, Control::Fast)
    }
    fn pair_control(epi: Epilogue, ctl: Control) -> Topo {
        Topo {
            rows: ROWS,
            cols: COLS,
            kind: Kind::Pair {
                epi,
                ctl,
                resident: None,
            },
        }
    }
    /// V9 / V10: the pair topology with operands resident in the memtile.
    fn with_resident(epi: Epilogue, ctl: Control, resident: Resident) -> Topo {
        Topo {
            rows: ROWS,
            cols: COLS,
            kind: Kind::Pair {
                epi,
                ctl,
                resident: Some(resident),
            },
        }
    }
    /// G80 topology: `kc` K64 chunks, `mw` M-waves of 256 rows; one N wave of 1280 columns.
    pub(crate) fn g80(epi: Epilogue, ctl: Control, kc: usize, mw: usize) -> Topo {
        Topo {
            rows: ROWS,
            cols: COLS,
            kind: Kind::G80 {
                epi,
                ctl,
                kc,
                mw,
                bf16_layout: gemm_g80::Bf16Layout::NativeTiles,
                n512: false,
            },
        }
    }
    /// G80 N512 BF16 topology: `kc` K64 chunks, `mw` M-waves of 512 rows; one N wave of 512 columns.
    pub(crate) fn g80_n512(
        ctl: Control,
        kc: usize,
        mw: usize,
        layout: gemm_g80::Bf16Layout,
    ) -> Topo {
        Topo {
            rows: ROWS,
            cols: COLS,
            kind: Kind::G80 {
                epi: Epilogue::Bf16,
                ctl,
                kc,
                mw,
                bf16_layout: layout,
                n512: true,
            },
        }
    }
    pub(crate) fn is_g80_n512(self) -> bool {
        matches!(self.kind, Kind::G80 { n512: true, .. })
    }
    pub(crate) fn is_g80(self) -> bool {
        matches!(self.kind, Kind::G80 { .. })
    }
    /// G80 with the raw i32 epilogue (finite two-stripe C tasks, no C wave-parity state).
    pub(crate) fn is_g80_i32(self) -> bool {
        matches!(
            self.kind,
            Kind::G80 {
                epi: Epilogue::I32,
                ..
            }
        )
    }
    pub(crate) fn is_g80_bf16(self) -> bool {
        matches!(
            self.kind,
            Kind::G80 {
                epi: Epilogue::Bf16,
                ..
            }
        )
    }
    pub(crate) fn bf16_rowmajor(self) -> Option<(usize, usize)> {
        match self.kind {
            Kind::G80 {
                epi: Epilogue::Bf16,
                bf16_layout:
                    gemm_g80::Bf16Layout::RowMajor {
                        pitch_elems,
                        col0_elems,
                    },
                ..
            } => Some((pitch_elems, col0_elems)),
            _ => None,
        }
    }
    /// IEF15 topology: `epochs` K128 epochs, `mw` x `nw` waves of 256 tokens x 256 features.
    pub(crate) fn ief15(ctl: Control, epochs: usize, mw: usize, nw: usize) -> Topo {
        Topo {
            rows: ROWS,
            cols: COLS,
            kind: Kind::Ief15 {
                ctl,
                epochs,
                mw,
                nw,
            },
        }
    }
    pub(crate) fn is_ief15(self) -> bool {
        matches!(self.kind, Kind::Ief15 { .. })
    }
    /// Vertical-pair A sharing: V8 / V9 / V10, G80 and IEF15.
    pub(crate) fn is_pair(self) -> bool {
        matches!(
            self.kind,
            Kind::Pair { .. } | Kind::G80 { .. } | Kind::Ief15 { .. }
        )
    }
    /// V9 / V10 memtile-resident parameters; G80 has its own memtile layout and reports `None`.
    fn resident(self) -> Option<Resident> {
        match self.kind {
            Kind::Pair { resident, .. } => resident,
            _ => None,
        }
    }
    /// V10: A per M-wave and B resident, `w = mw*NW + nw`.
    fn mw_outer(self) -> bool {
        self.resident()
            .is_some_and(|r| r.mode == ResidentMode::MwOuter)
    }
    /// Bytes of one core C tile as it reaches the host (i32 `C_BYTES`, int8 `COUT_BYTES`; G80 128x80: 40,960 B i32,
    /// 10,240 B int8).
    pub(crate) fn c_tile_bytes(self) -> usize {
        match self.kind {
            Kind::Pair {
                epi: Epilogue::Int8 { .. },
                ..
            } => COUT_BYTES,
            Kind::G80 {
                epi: Epilogue::I32, ..
            } => gemm_core_g80::C_BYTES,
            Kind::G80 {
                epi: Epilogue::Bf16,
                n512: true,
                ..
            } => gemm_core_g80::bf16_n64_tile_bytes(),
            Kind::G80 {
                epi: Epilogue::Bf16,
                ..
            } => gemm_core_g80::C_STRIPE_BYTES,
            Kind::G80 { .. } => gemm_core_g80::COUT_BYTES,
            Kind::Ief15 { .. } => iu4_ief15_core::COUT_BYTES,
            _ => C_BYTES,
        }
    }
    pub(crate) fn wave_m(self) -> usize {
        if self.is_g80_n512() {
            G80_N512_WAVE_M
        } else if self.is_g80() {
            G80_WAVE_M
        } else if self.is_ief15() {
            IEF15_WAVE
        } else {
            self.rows * TM
        }
    }
    pub(crate) fn wave_n(self) -> usize {
        if self.is_g80_n512() {
            G80_N512_WAVE_N
        } else if self.is_g80() {
            G80_WAVE_N
        } else if self.is_ief15() {
            IEF15_WAVE
        } else {
            self.cols * TN
        }
    }
    /// Column injects an A stream: every column of a pair design, columns `< G80_A_COLS` of G80, the column of
    /// core row `col` (Pass/Staged kinds).
    pub(crate) fn has_a(self, col: u32) -> bool {
        match self.kind {
            Kind::Pair { .. } | Kind::Ief15 { .. } => true,
            Kind::G80 { n512: true, .. } => true,
            Kind::G80 { .. } => (col as usize) < G80_A_COLS,
            _ => (col as usize) < self.rows,
        }
    }
    pub(crate) fn cores(self) -> impl Iterator<Item = Location> {
        (0..self.cols as u32).flat_map(move |c| (0..self.rows).map(move |r| core_loc(c, r)))
    }
}

/// Padded problem geometry shared by packing, unpacking, descriptors and the TXN.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Geometry {
    pub(crate) topo: Topo,
    pub(crate) m: usize,
    pub(crate) n: usize,
    pub(crate) k: usize,
    pub(crate) kc: usize,
    pub(crate) mw: usize,
    pub(crate) nw: usize,
    pub(crate) waves: usize,
    /// V9 ([`ArrayDesign::with_a_repeat`]): host A holds each `mw` tile once; the shim A task repeats it `nw` times.
    pub(crate) a_repeat: bool,
}

impl Geometry {
    /// G80 geometry: `n` is exactly one 1280-column wave, `k` a multiple of 64 up to `gemm_core_g80::MAX_KC` chunks,
    /// `m` padded to 256-row M-waves (`waves = mw`, `nw = 1`). Int8 or i32 epilogue.
    pub(crate) fn g80(m: usize, n: usize, k: usize, epi: Epilogue, ctl: Control) -> Self {
        assert!(m > 0, "empty GEMM");
        assert_eq!(n, G80_WAVE_N, "G80 needs N = {G80_WAVE_N}, got {n}");
        assert!(
            k % CHUNK_K == 0 && (CHUNK_K..=gemm_core_g80::MAX_KC * CHUNK_K).contains(&k),
            "G80 K must be a multiple of {CHUNK_K} in {CHUNK_K}..={}, got {k}",
            gemm_core_g80::MAX_KC * CHUNK_K
        );
        let (kc, mw) = (k / CHUNK_K, m.div_ceil(G80_WAVE_M));
        assert!(mw <= MAX_WAVES, "{mw} waves exceed {MAX_WAVES}");
        Self {
            topo: Topo::g80(epi, ctl, kc, mw),
            m,
            n,
            k,
            kc,
            mw,
            nw: 1,
            waves: mw,
            a_repeat: false,
        }
    }
    /// G80 N512 geometry ([`gemm_g80_n512`]): `n` is exactly one 512-column wave, `k` a multiple of 64 up to
    /// `gemm_g80_n512::MAX_KC` chunks (resident B panel), `m` padded to 512-row waves (`waves = mw`, `nw = 1`).
    pub(crate) fn g80_n512(
        m: usize,
        n: usize,
        k: usize,
        ctl: Control,
        layout: gemm_g80::Bf16Layout,
    ) -> Self {
        assert!(m > 0, "empty GEMM");
        assert_eq!(
            n, G80_N512_WAVE_N,
            "G80 N512 needs N = {G80_N512_WAVE_N}, got {n}"
        );
        assert!(
            k % CHUNK_K == 0 && (CHUNK_K..=gemm_g80_n512::MAX_KC * CHUNK_K).contains(&k),
            "G80 N512 K must be a multiple of {CHUNK_K} in {CHUNK_K}..={}, got {k}",
            gemm_g80_n512::MAX_KC * CHUNK_K
        );
        let (kc, mw) = (k / CHUNK_K, m.div_ceil(G80_N512_WAVE_M));
        assert!(mw <= MAX_WAVES, "{mw} waves exceed {MAX_WAVES}");
        Self {
            topo: Topo::g80_n512(ctl, kc, mw, layout),
            m,
            n,
            k,
            kc,
            mw,
            nw: 1,
            waves: mw,
            a_repeat: false,
        }
    }
    /// IEF15 geometry: `m` tokens x `n` features x `k` (K128 epochs, `kc = k / 128`), 256 x 256 waves in the order
    /// `w = mw*NW + nw`, `waves <= 256`.
    pub(crate) fn ief15(m: usize, n: usize, k: usize, ctl: Control) -> Self {
        assert!(m > 0 && n > 0, "empty GEMM");
        assert!(
            k % iu4_ief15_core::EPOCH_K == 0
                && (1..=iu4_ief15_core::MAX_EPOCHS).contains(&(k / iu4_ief15_core::EPOCH_K)),
            "IEF15 K must be a multiple of {} in {}..={}",
            iu4_ief15_core::EPOCH_K,
            iu4_ief15_core::EPOCH_K,
            iu4_ief15_core::MAX_EPOCHS * iu4_ief15_core::EPOCH_K
        );
        let (epochs, mw, nw) = (
            k / iu4_ief15_core::EPOCH_K,
            m.div_ceil(IEF15_WAVE),
            n.div_ceil(IEF15_WAVE),
        );
        assert!(mw * nw <= MAX_WAVES, "{} waves exceed {MAX_WAVES}", mw * nw);
        Self {
            topo: Topo::ief15(ctl, epochs, mw, nw),
            m,
            n,
            k,
            kc: epochs,
            mw,
            nw,
            waves: mw * nw,
            a_repeat: false,
        }
    }
    fn new(topo: Topo, m: usize, n: usize, k: usize) -> Self {
        assert!(m > 0 && n > 0, "empty GEMM");
        assert!(
            k % CHUNK_K == 0 && (CHUNK_K..=MAX_KC * CHUNK_K).contains(&k),
            "K must be a multiple of {CHUNK_K} in {CHUNK_K}..={}",
            MAX_KC * CHUNK_K
        );
        let (mw, nw) = (m.div_ceil(topo.wave_m()), n.div_ceil(topo.wave_n()));
        let waves = mw * nw;
        assert!(waves <= MAX_WAVES, "{waves} waves exceed {MAX_WAVES}");
        Self {
            topo,
            m,
            n,
            k,
            kc: k / CHUNK_K,
            mw,
            nw,
            waves,
            a_repeat: false,
        }
    }
    /// Bytes of one A host segment: a core-row block (`A_BYTES` per chunk) or, for V8, one injecting
    /// column's A half (`A_HALF_BYTES` per chunk) per wave; V10 and `a_repeat` V9 hold each `mw` tile once (`mw` tiles).
    pub(crate) fn a_segment_bytes(&self) -> usize {
        if self.topo.is_g80() {
            return self.waves * self.kc * gemm_g80::a_chunk_bytes(self.topo);
        }
        if self.topo.is_ief15() {
            return self.waves * self.kc * iu4_ief15_core::A_HALF_BYTES;
        }
        let tiles = if self.topo.mw_outer() || self.a_repeat {
            self.mw
        } else {
            self.waves
        };
        tiles
            * self.kc
            * if self.topo.is_pair() {
                A_HALF_BYTES
            } else {
                A_BYTES
            }
    }
    /// Bytes of one B host segment: every wave's B tile (`waves*kc*B_BYTES`) or, for V9 / V10, each `nw` tile once.
    pub(crate) fn b_segment_bytes(&self) -> usize {
        // G80: per chunk both 80-column N tiles of a column, `[even tile 5120 B, odd tile 5120 B]`.
        if self.topo.is_g80_n512() {
            return self.kc * gemm_g80::b_chunk_bytes(self.topo);
        }
        if self.topo.is_g80() {
            return self.kc * 2 * gemm_g80::b_chunk_bytes(self.topo);
        }
        if self.topo.is_ief15() {
            return self.waves * self.kc * iu4_ief15_core::B_BYTES;
        }
        (if self.topo.resident().is_some() {
            self.nw
        } else {
            self.waves
        }) * self.kc
            * B_BYTES
    }
    /// `(mw, nw)` of wave `w`: `w = mw*NW + nw`, for V9 (N-wave outer) `w = nw*MW + mw`.
    pub(crate) fn wave_coords(&self, w: usize) -> (usize, usize) {
        if self
            .topo
            .resident()
            .is_some_and(|r| r.mode == ResidentMode::NwOuter)
        {
            (w % self.mw, w / self.mw)
        } else {
            (w / self.nw, w % self.nw)
        }
    }
    pub(crate) fn c_segment_bytes(&self) -> usize {
        self.waves * self.topo.rows * self.topo.c_tile_bytes()
    }
    pub(crate) fn a_bytes(&self) -> usize {
        let a_cols = if self.topo.is_g80_n512() {
            self.topo.cols
        } else if self.topo.is_g80() {
            G80_A_COLS
        } else if self.topo.is_pair() {
            self.topo.cols
        } else {
            self.topo.rows
        };
        a_cols * self.a_segment_bytes()
    }
    pub(crate) fn b_bytes(&self) -> usize {
        self.topo.cols * self.b_segment_bytes()
    }
    pub(crate) fn c_bytes(&self) -> usize {
        if let Some((pitch, _)) = self.topo.bf16_rowmajor() {
            self.waves * self.topo.wave_m() * pitch * 2
        } else {
            self.topo.cols * self.c_segment_bytes()
        }
    }

    /// V8 tile `(m offset, n offset)` inside a wave of core `(col, rho)`: A block `2*(rho/2) + col%2`, B block
    /// `col/2`, 64-column half `rho%2`.
    fn pair_tile(col: usize, rho: usize) -> (usize, usize) {
        (
            (2 * (rho / 2) + col % 2) * TM,
            (col / 2) * 2 * TN + (rho % 2) * TN,
        )
    }

    /// V8 C argument layout: per column a segment of two shim-S2MM streams (`rho / 2`), each
    /// `(wave, rho % 2)` tiles of `c_tile_bytes`. Byte offset of the tile of core `(col, rho)` in wave `w`.
    pub(crate) fn pair_c_offset(&self, col: usize, w: usize, rho: usize) -> usize {
        let cb = self.topo.c_tile_bytes();
        col * self.c_segment_bytes() + (rho / 2) * self.waves * 2 * cb + (w * 2 + rho % 2) * cb
    }

    /// One 128x64 A half chunk: the 8-row blocks `mb = 2*mbl + h` of the 128-row block at `row0`, stored
    /// `(mbl 0..8, kb 0..8)`, zero padded below `m`.
    pub(crate) fn pack_a_half_chunk(
        &self,
        a: &[i8],
        row0: usize,
        h: usize,
        ch: usize,
        dst_chunk: &mut [u8],
    ) {
        for mbl in 0..MB / 2 {
            for kb in 0..KB {
                for i in 0..8 {
                    let row = row0 + (2 * mbl + h) * 8 + i;
                    let dst = (mbl * KB + kb) * 64 + i * 8;
                    let d = &mut dst_chunk[dst..dst + 8];
                    if row < self.m {
                        let s = row * self.k + ch * CHUNK_K + kb * 8;
                        for (x, y) in d.iter_mut().zip(&a[s..s + 8]) {
                            *x = *y as u8;
                        }
                    } else {
                        d.fill(0);
                    }
                }
            }
        }
    }

    /// V8 A stream: column `c` injects the half `h = (c/2)%2` (even / odd 8-row blocks) of A block
    /// `2*(c/4) + c%2`. Each `(column, mw)` tile of `kc` chunks is repeated for every wave of that `mw`, in wave order.
    fn pack_a_pair(&self, a: &[i8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.a_bytes());
        let tile_bytes = self.kc * A_HALF_BYTES;
        let mut tiles = vec![0u8; self.mw * tile_bytes];
        for col in 0..self.topo.cols {
            let (q, h, par) = (col / 4, (col / 2) % 2, col % 2);
            for (mw, tile) in tiles.chunks_mut(tile_bytes).enumerate() {
                for ch in 0..self.kc {
                    let row0 = mw * self.topo.wave_m() + (2 * q + par) * TM;
                    self.pack_a_half_chunk(
                        a,
                        row0,
                        h,
                        ch,
                        &mut tile[ch * A_HALF_BYTES..(ch + 1) * A_HALF_BYTES],
                    );
                }
            }
            if self.topo.mw_outer() || self.a_repeat {
                // V10: every `mw` tile once, in `mw` order (the cores see it `NW` times from the memtile); `a_repeat`
                // V9: the same layout, the shim task replays it `NW` times (nw-outer wave order).
                out.extend_from_slice(&tiles);
            } else {
                for w in 0..self.waves {
                    let mw = self.wave_coords(w).0;
                    out.extend_from_slice(&tiles[mw * tile_bytes..(mw + 1) * tile_bytes]);
                }
            }
        }
        out
    }

    /// One 128x64 A chunk (`ch`) of the rows starting at `row0`, zero padded below `m`.
    fn pack_a_chunk(&self, a: &[i8], row0: usize, ch: usize, dst_chunk: &mut [u8]) {
        for mb in 0..MB {
            for kb in 0..KB {
                for i in 0..8 {
                    let row = row0 + mb * 8 + i;
                    let dst = (mb * KB + kb) * 64 + i * 8;
                    let d = &mut dst_chunk[dst..dst + 8];
                    if row < self.m {
                        let s = row * self.k + ch * CHUNK_K + kb * 8;
                        for (x, y) in d.iter_mut().zip(&a[s..s + 8]) {
                            *x = *y as u8;
                        }
                    } else {
                        d.fill(0);
                    }
                }
            }
        }
    }

    /// Row-segmented A stream (Pass/Staged): each `(row, mw)` segment is written once and repeated for `nw`.
    /// One-column (V3): per `mw`, chunk-major with the core rows interleaved `(ch, row)`, repeated for `nw`.
    fn pack_a(&self, a: &[i8]) -> Vec<u8> {
        if self.topo.is_g80() {
            return gemm_g80::pack_a(self, a);
        }
        assert!(
            !self.topo.is_ief15(),
            "IEF15 operands are packed by iu4_ief15::Ief15Gemm::pack_in"
        );
        assert_eq!(a.len(), self.m * self.k, "A must be m*k");
        let rows = self.topo.rows;
        let mut out = Vec::with_capacity(self.a_bytes());
        if self.topo.is_pair() {
            return self.pack_a_pair(a);
        }
        if self.topo.kind == Kind::Column {
            let mut seg = vec![0u8; self.kc * rows * A_BYTES];
            for mw in 0..self.mw {
                for ch in 0..self.kc {
                    for r in 0..rows {
                        let at = (ch * rows + r) * A_BYTES;
                        self.pack_a_chunk(a, (mw * rows + r) * TM, ch, &mut seg[at..at + A_BYTES]);
                    }
                }
                for _ in 0..self.nw {
                    out.extend_from_slice(&seg);
                }
            }
            return out;
        }
        let mut tile = vec![0u8; self.kc * A_BYTES];
        for r in 0..rows {
            for mw in 0..self.mw {
                for ch in 0..self.kc {
                    self.pack_a_chunk(
                        a,
                        (mw * rows + r) * TM,
                        ch,
                        &mut tile[ch * A_BYTES..(ch + 1) * A_BYTES],
                    );
                }
                for _ in 0..self.nw {
                    out.extend_from_slice(&tile);
                }
            }
        }
        out
    }

    /// Column-segmented B stream; each `(col, nw)` segment is built once and reused for every `mw`.
    fn pack_b(&self, b: &[i8]) -> Vec<u8> {
        if self.topo.is_g80() {
            return gemm_g80::pack_b(self, b);
        }
        assert!(
            !self.topo.is_ief15(),
            "IEF15 operands are packed by iu4_ief15::Ief15Gemm::pack_in"
        );
        assert_eq!(b.len(), self.k * self.n, "B must be k*n");
        let cols = self.topo.cols;
        let mut out = Vec::with_capacity(self.b_bytes());
        let mut tiles = vec![vec![0u8; self.kc * B_BYTES]; self.nw];
        for col in 0..cols {
            for (nw, tile) in tiles.iter_mut().enumerate() {
                self.pack_b_tile(b, col, nw, tile);
            }
            if self.topo.resident().is_some() {
                for tile in &tiles {
                    out.extend_from_slice(tile);
                }
            } else {
                for w in 0..self.waves {
                    out.extend_from_slice(&tiles[w % self.nw]);
                }
            }
        }
        out
    }

    /// The `kc`-chunk B tile of column `col` in N-wave `nw` (`(chunk, (kb, nb) blocks)`), zero padded right of `n`.
    fn pack_b_tile(&self, b: &[i8], col: usize, nw: usize, tile: &mut [u8]) {
        tile.fill(0);
        let n0 = (nw * self.topo.cols + col) * TN;
        for ch in 0..self.kc {
            for kb in 0..KB {
                for nb in 0..NB {
                    for kk in 0..8 {
                        let c0 = n0 + nb * 8;
                        if c0 >= self.n {
                            continue;
                        }
                        let width = (self.n - c0).min(8);
                        let row = ch * CHUNK_K + kb * 8 + kk;
                        let dst = ch * B_BYTES + (kb * NB + nb) * 64 + kk * 8;
                        let s = row * self.n + c0;
                        for (x, y) in tile[dst..dst + width].iter_mut().zip(&b[s..s + width]) {
                            *x = *y as u8;
                        }
                    }
                }
            }
        }
    }

    /// Streaming-topology B stream whose B depends on the M-wave: wave `w = (mw, nw)` carries tile `nw` of the
    /// `k*n` matrix `b_of_mw(mw)`. This is the batched GEMM `C[mw block] = A[mw block] * B(mw)`; the device-side
    /// design is unchanged because a streaming topology reads every wave's B tile from its own host bytes.
    fn pack_b_per_mw<'a>(&self, b_of_mw: &dyn Fn(usize) -> &'a [i8]) -> Vec<u8> {
        assert!(
            self.topo.resident().is_none(),
            "per-M-wave B needs a streaming (non-resident) topology"
        );
        let mut out = Vec::with_capacity(self.b_bytes());
        let mut tile = vec![0u8; self.kc * B_BYTES];
        for col in 0..self.topo.cols {
            for w in 0..self.waves {
                let (mw, nw) = self.wave_coords(w);
                let b = b_of_mw(mw);
                assert_eq!(b.len(), self.k * self.n, "B of M-wave {mw} must be k*n");
                self.pack_b_tile(b, col, nw, &mut tile);
                out.extend_from_slice(&tile);
            }
        }
        out
    }

    /// Row-major `m*n` C from the packed C stream; int8 tiles (V8 `Epilogue::Int8`) are sign extended.
    fn unpack_c(&self, output: &[u8]) -> Vec<i32> {
        if self.topo.is_g80() {
            return gemm_g80::unpack_c(self, output);
        }
        assert!(
            !self.topo.is_ief15(),
            "IEF15 output is unpacked by iu4_ief15::Ief15Gemm::unpack_out"
        );
        assert_eq!(output.len(), self.c_bytes(), "C stream size");
        let (rows, cols) = (self.topo.rows, self.topo.cols);
        let int8 = matches!(
            self.topo.kind,
            Kind::Pair {
                epi: Epilogue::Int8 { .. },
                ..
            }
        );
        let mut c = vec![0i32; self.m * self.n];
        for col in 0..cols {
            for w in 0..self.waves {
                for r in 0..rows {
                    let (mw, nw) = self.wave_coords(w);
                    let (m0, n0, base) = if self.topo.is_pair() {
                        let (dm, dn) = Self::pair_tile(col, r);
                        (
                            mw * self.topo.wave_m() + dm,
                            nw * self.topo.wave_n() + dn,
                            self.pair_c_offset(col, w, r),
                        )
                    } else {
                        (
                            (mw * rows + r) * TM,
                            (nw * cols + col) * TN,
                            ((col * self.waves + w) * rows + r) * C_BYTES,
                        )
                    };
                    if m0 >= self.m || n0 >= self.n {
                        continue;
                    }
                    for mb in 0..MB {
                        for nb in 0..NB {
                            for i in 0..8 {
                                let row = m0 + mb * 8 + i;
                                if row >= self.m {
                                    break;
                                }
                                for j in 0..8 {
                                    let cc = n0 + nb * 8 + j;
                                    if cc >= self.n {
                                        break;
                                    }
                                    let at = (mb * NB + nb) * 64 + i * 8 + j;
                                    c[row * self.n + cc] = if int8 {
                                        output[base + at] as i8 as i32
                                    } else {
                                        let off = base + at * 4;
                                        i32::from_le_bytes(output[off..off + 4].try_into().unwrap())
                                    };
                                }
                            }
                        }
                    }
                }
            }
        }
        c
    }

    /// Shim stream per column: `(A offset (columns with A), B offset, C offset)` into the three args.
    fn column_offsets(&self, col: usize) -> (usize, usize, usize) {
        let a = if self.topo.has_a(col as u32) {
            col * self.a_segment_bytes()
        } else {
            0
        };
        (
            a,
            col * self.b_segment_bytes(),
            col * self.c_segment_bytes(),
        )
    }
}

/// ESTIMATE only (never a measurement): serialized input/compute and output phases, with ideal VMAC issue
/// and 4 bytes/cycle per DMA channel. Ignores inter-wave overlap, pipeline fill/drain, lock waits, start-up
/// and DDR/route contention; consequently this is neither a measured time nor a strict lower/upper bound.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CycleEstimate {
    /// One 128x64x64 chunk is 1024 VMAC issues per core; all active cores run in parallel.
    pub ideal_mac_cycles: u64,
    /// A chunk (8 KiB) over the core's A S2MM channel: 2048 cycles per chunk per wave.
    pub input_dma_cycles: u64,
    /// The core C tiles of a column (4 for the full array) share one shim S2MM0: 32768 cycles per row per wave.
    pub output_dma_cycles: u64,
    /// `max(input, ideal) + output`: aggregate output is modeled as a separate, non-overlapped phase.
    pub estimated_cycles: u64,
    pub estimated_seconds: f64,
    /// Useful (unpadded) 2*m*n*k ops per estimated second, in TOPS.
    pub useful_tops: f64,
    /// 512 MAC x 32 cores x 2 ops x clock.
    pub peak_ops_per_second: u64,
}

/// Switch circuits of the one-column variant (shim, memtile, 4 core tiles of column 0).
fn one_column_circuits() -> Vec<Circuit> {
    let mut v = Vec::new();
    let (shim, mem) = (shim_loc(0), mem_loc(0));
    let mm0 = ShimDma {
        tile: shim,
        direction: Mm2s,
        channel: 0,
    };
    let mm1 = ShimDma {
        tile: shim,
        direction: Mm2s,
        channel: 1,
    };
    let sm0 = ShimDma {
        tile: shim,
        direction: S2mm,
        channel: 0,
    };
    // A (interleaved) on shim MM2S0 -> memtile S2MM4; B on MM2S1 -> S2MM5; C join MM2S5 -> shim S2MM0.
    v.push(Circuit {
        tile: shim,
        slave: mm0.port(),
        master: Port::North(0),
    });
    v.push(Circuit {
        tile: shim,
        slave: mm1.port(),
        master: Port::North(1),
    });
    v.push(Circuit {
        tile: shim,
        slave: Port::North(0),
        master: sm0.port(),
    });
    v.push(Circuit {
        tile: mem,
        slave: Port::South(0),
        master: Port::Dma(OC_A_PROD_CH as u8),
    });
    v.push(Circuit {
        tile: mem,
        slave: Port::South(1),
        master: Port::Dma(OC_B_PROD_CH as u8),
    });
    for r in 0..ROWS as u8 {
        // Per-row A consumer MM2S r up lane r; C of core row r back on lane r into S2MM r.
        v.push(Circuit {
            tile: mem,
            slave: Port::Dma(r),
            master: Port::North(r),
        });
        v.push(Circuit {
            tile: mem,
            slave: Port::North(r),
            master: Port::Dma(r),
        });
    }
    v.push(Circuit {
        tile: mem,
        slave: Port::Dma(OC_B_CONS_CH as u8),
        master: Port::North(4),
    });
    v.push(Circuit {
        tile: mem,
        slave: Port::Dma(OC_C_JOIN_CH as u8),
        master: Port::South(0),
    });
    for r in 0..ROWS {
        let tile = core_loc(0, r);
        v.push(Circuit {
            tile,
            slave: Port::South(4),
            master: Port::Dma(1),
        });
        if r + 1 < ROWS {
            v.push(Circuit {
                tile,
                slave: Port::South(4),
                master: Port::North(4),
            });
        }
        v.push(Circuit {
            tile,
            slave: Port::Dma(0),
            master: Port::South(r as u8),
        });
        for s in r + 1..ROWS {
            v.push(Circuit {
                tile,
                slave: Port::North(s as u8),
                master: Port::South(s as u8),
            });
        }
        // Lane r ends at row r; higher lanes pass through.
        v.push(Circuit {
            tile,
            slave: Port::South(r as u8),
            master: Port::Dma(0),
        });
        for s in r + 1..ROWS {
            v.push(Circuit {
                tile,
                slave: Port::South(s as u8),
                master: Port::North(s as u8),
            });
        }
    }
    v
}

/// V8 switch circuits of one column (see the module header, "V8 data movement").
pub(crate) fn pair_column_circuits(col: u32) -> Vec<Circuit> {
    let mut v = Vec::new();
    let (shim, mem) = (shim_loc(col), mem_loc(col));
    let sh = |direction, channel| ShimDma {
        tile: shim,
        direction,
        channel,
    };
    // Shim: B on MM2S0 / North0, A half on MM2S1 / North1; C of memtile MM2S0 returns on North0 -> S2MM0,
    // of MM2S3 on North1 -> S2MM1.
    v.push(Circuit {
        tile: shim,
        slave: sh(Mm2s, 0).port(),
        master: Port::North(0),
    });
    v.push(Circuit {
        tile: shim,
        slave: sh(Mm2s, 1).port(),
        master: Port::North(1),
    });
    v.push(Circuit {
        tile: shim,
        slave: Port::North(0),
        master: sh(S2mm, 0).port(),
    });
    v.push(Circuit {
        tile: shim,
        slave: Port::North(1),
        master: sh(S2mm, 1).port(),
    });
    v.push(Circuit {
        tile: mem,
        slave: Port::South(0),
        master: Port::Dma(B_S2MM_CH as u8),
    });
    v.push(Circuit {
        tile: mem,
        slave: Port::South(1),
        master: Port::Dma(A_S2MM_CH as u8),
    });
    v.push(Circuit {
        tile: mem,
        slave: Port::Dma(B_MM2S_CH as u8),
        master: Port::North(4),
    });
    v.push(Circuit {
        tile: mem,
        slave: Port::Dma(A_MM2S_CH as u8),
        master: Port::North(5),
    });
    v.push(Circuit {
        tile: mem,
        slave: Port::Dma(PAIR_C_MM2S_CH[0] as u8),
        master: Port::South(0),
    });
    v.push(Circuit {
        tile: mem,
        slave: Port::Dma(PAIR_C_MM2S_CH[1] as u8),
        master: Port::South(1),
    });
    for r in 0..ROWS as u8 {
        v.push(Circuit {
            tile: mem,
            slave: Port::North(r),
            master: Port::Dma(r),
        });
    }
    let c = col as usize;
    // B half injected by this column is half `par` = c%2 (B block j = c/2): it feeds rows par and par+2 of
    // this column and crosses at row par on lane 2 to the neighbour column, which takes it at that same row
    // (Dma1 + North3) and at row par+2 (South3 -> Dma1). So this column receives the neighbour's half at
    // rows x = 1-par (the rows that are not its own parity) and x+2.
    let par = c % 2;
    let (cross_out, cross_in) = if par == 0 {
        (Port::East(2), Port::East(2))
    } else {
        (Port::West(2), Port::West(2))
    };
    for r in 0..ROWS {
        let tile = core_loc(col, r);
        // C down as V6.
        v.push(Circuit {
            tile,
            slave: Port::Dma(0),
            master: Port::South(r as u8),
        });
        for s in r + 1..ROWS {
            v.push(Circuit {
                tile,
                slave: Port::North(s as u8),
                master: Port::South(s as u8),
            });
        }
        // B own half.
        if r % 2 == par {
            v.push(Circuit {
                tile,
                slave: Port::South(4),
                master: Port::Dma(1),
            });
        }
        if r < par + 2 {
            v.push(Circuit {
                tile,
                slave: Port::South(4),
                master: Port::North(4),
            });
        }
        if r == par {
            v.push(Circuit {
                tile,
                slave: Port::South(4),
                master: cross_out,
            });
        }
        // B other half from the neighbour column.
        let x = 1 - par;
        if r == x {
            v.push(Circuit {
                tile,
                slave: cross_in,
                master: Port::Dma(1),
            });
            v.push(Circuit {
                tile,
                slave: cross_in,
                master: Port::North(3),
            });
        }
        if r == x + 1 {
            v.push(Circuit {
                tile,
                slave: Port::South(3),
                master: Port::North(3),
            });
        }
        if r == x + 2 {
            v.push(Circuit {
                tile,
                slave: Port::South(3),
                master: Port::Dma(1),
            });
        }
        // A: row r receives block-parity lane `l` from the injecting column c0 = 2r + l, along the row on
        // horizontal lane l; columns with c%2 == l take it into Dma0, the others pass it through. The lane is
        // only forwarded towards a side that still has a consumer column (x%2 == l, x < c resp. x > c), because
        // a master driving a neighbour slave without a circuit would stall the whole broadcast.
        for lane in 0..2usize {
            let c0 = 2 * r + lane;
            let slave = match c.cmp(&c0) {
                std::cmp::Ordering::Equal => Port::South(5),
                std::cmp::Ordering::Less => Port::East(lane as u8),
                std::cmp::Ordering::Greater => Port::West(lane as u8),
            };
            if c % 2 == lane {
                v.push(Circuit {
                    tile,
                    slave,
                    master: Port::Dma(0),
                });
            }
            if c > lane && c <= c0 {
                v.push(Circuit {
                    tile,
                    slave,
                    master: Port::West(lane as u8),
                });
            }
            if c < 6 + lane && c >= c0 {
                v.push(Circuit {
                    tile,
                    slave,
                    master: Port::East(lane as u8),
                });
            }
        }
        // Column c injects A for row c/2; rows below it pass the stream up lane 5.
        if r < c / 2 {
            v.push(Circuit {
                tile,
                slave: Port::South(5),
                master: Port::North(5),
            });
        }
    }
    v
}

/// Static stream-switch circuits for one column (shim, memtile, core tiles) of `topo`.
fn column_circuits(topo: Topo, col: u32) -> Vec<Circuit> {
    if topo.is_g80() {
        return gemm_g80::column_circuits(topo, col);
    }
    if topo.is_pair() {
        return pair_column_circuits(col);
    }
    if topo.kind == Kind::Column {
        return one_column_circuits();
    }
    assert!(topo.rows <= topo.cols);
    let mut v = Vec::new();
    let (shim, mem) = (shim_loc(col), mem_loc(col));
    let mm0 = ShimDma {
        tile: shim,
        direction: Mm2s,
        channel: 0,
    };
    let mm1 = ShimDma {
        tile: shim,
        direction: Mm2s,
        channel: 1,
    };
    let sm0 = ShimDma {
        tile: shim,
        direction: S2mm,
        channel: 0,
    };
    v.push(Circuit {
        tile: shim,
        slave: mm0.port(),
        master: Port::North(0),
    });
    v.push(Circuit {
        tile: shim,
        slave: Port::North(0),
        master: sm0.port(),
    });
    if topo.kind == Kind::Pass {
        // Pass-through (single core, no memtile DMA). The AIE2P stream switch only passes a memtile
        // North/South slave to the North/South master of the SAME lane (aie-rt `xaie_ss_aieml.c`, port
        // verification `xaie2pgbl_reginit.c`), so B rides lane 0 and A lane 1 through the memtile.
        v.push(Circuit {
            tile: shim,
            slave: mm1.port(),
            master: Port::North(1),
        });
        v.push(Circuit {
            tile: mem,
            slave: Port::South(0),
            master: Port::North(0),
        });
        v.push(Circuit {
            tile: mem,
            slave: Port::South(1),
            master: Port::North(1),
        });
        v.push(Circuit {
            tile: mem,
            slave: Port::North(0),
            master: Port::South(0),
        });
        let tile = core_loc(col, 0);
        v.push(Circuit {
            tile,
            slave: Port::South(0),
            master: Port::Dma(1),
        });
        v.push(Circuit {
            tile,
            slave: Port::South(1),
            master: Port::Dma(0),
        });
        v.push(Circuit {
            tile,
            slave: Port::Dma(0),
            master: Port::South(0),
        });
        return v;
    }
    // Staged.
    v.push(Circuit {
        tile: mem,
        slave: Port::South(0),
        master: Port::Dma(B_S2MM_CH as u8),
    });
    v.push(Circuit {
        tile: mem,
        slave: Port::Dma(B_MM2S_CH as u8),
        master: Port::North(4),
    });
    v.push(Circuit {
        tile: mem,
        slave: Port::Dma(0),
        master: Port::South(0),
    });
    for r in 0..topo.rows as u8 {
        v.push(Circuit {
            tile: mem,
            slave: Port::North(r),
            master: Port::Dma(r),
        });
    }
    if topo.has_a(col) {
        v.push(Circuit {
            tile: shim,
            slave: mm1.port(),
            master: Port::North(1),
        });
        v.push(Circuit {
            tile: mem,
            slave: Port::South(1),
            master: Port::Dma(A_S2MM_CH as u8),
        });
        v.push(Circuit {
            tile: mem,
            slave: Port::Dma(A_MM2S_CH as u8),
            master: Port::North(5),
        });
    }
    let (c, last) = (col as usize, topo.cols - 1);
    for r in 0..topo.rows {
        let tile = core_loc(col, r);
        v.push(Circuit {
            tile,
            slave: Port::South(4),
            master: Port::Dma(1),
        });
        if r + 1 < topo.rows {
            v.push(Circuit {
                tile,
                slave: Port::South(4),
                master: Port::North(4),
            });
        }
        v.push(Circuit {
            tile,
            slave: Port::Dma(0),
            master: Port::South(r as u8),
        });
        for s in r + 1..topo.rows {
            v.push(Circuit {
                tile,
                slave: Port::North(s as u8),
                master: Port::South(s as u8),
            });
        }
        // Row r's A is injected by column r and flows along the row to the last column.
        let a_slave = match c.cmp(&r) {
            std::cmp::Ordering::Equal => Port::South(5),
            std::cmp::Ordering::Less => Port::East(0),
            std::cmp::Ordering::Greater => Port::West(0),
        };
        v.push(Circuit {
            tile,
            slave: a_slave,
            master: Port::Dma(0),
        });
        if c > 0 && c <= r {
            v.push(Circuit {
                tile,
                slave: a_slave,
                master: Port::West(0),
            });
        }
        if c < last && c >= r {
            v.push(Circuit {
                tile,
                slave: a_slave,
                master: Port::East(0),
            });
        }
        if c < topo.rows && r < c {
            v.push(Circuit {
                tile,
                slave: Port::South(5),
                master: Port::North(5),
            });
        }
    }
    v
}

fn circuits(topo: Topo) -> Vec<Circuit> {
    (0..topo.cols as u32)
        .flat_map(|col| column_circuits(topo, col))
        .collect()
}

fn shim_routes(topo: Topo, col: u32) -> Vec<ShimDma> {
    let shim = shim_loc(col);
    let mut v = vec![
        ShimDma {
            tile: shim,
            direction: Mm2s,
            channel: 0,
        },
        ShimDma {
            tile: shim,
            direction: S2mm,
            channel: 0,
        },
    ];
    if topo.kind == Kind::Column || topo.has_a(col) {
        v.push(ShimDma {
            tile: shim,
            direction: Mm2s,
            channel: 1,
        });
    }
    if topo.is_pair() {
        v.push(ShimDma {
            tile: shim,
            direction: S2mm,
            channel: 1,
        });
    }
    v
}

/// Producer ring = S2MM (acquire empty, release full); consumer ring = MM2S (acquire full, release empty).
pub(crate) fn ring(
    ids: [u32; 2],
    offs: [u32; 2],
    words: u32,
    empty: [u32; 2],
    full: [u32; 2],
    producer: bool,
) -> Vec<(u32, Bd)> {
    (0..2)
        .map(|i| {
            let mut bd = Bd::new((MEM_BASE + offs[i]) as u64, words);
            let (acq, rel) = if producer {
                (empty[i], full[i])
            } else {
                (full[i], empty[i])
            };
            bd.locks = BdLocks {
                acq: Some((MEM_LOCK + acq, -1)),
                rel: Some((MEM_LOCK + rel, 1)),
            };
            bd.next = Some(ids[1 - i]);
            (ids[i], bd)
        })
        .collect()
}

fn ab_locks() -> (([u32; 2], [u32; 2]), ([u32; 2], [u32; 2])) {
    let a = (
        [gemm_core::A_EMPTY[0] as u32, gemm_core::A_EMPTY[1] as u32],
        [gemm_core::A_FULL[0] as u32, gemm_core::A_FULL[1] as u32],
    );
    let b = (
        [gemm_core::B_EMPTY[0] as u32, gemm_core::B_EMPTY[1] as u32],
        [gemm_core::B_FULL[0] as u32, gemm_core::B_FULL[1] as u32],
    );
    (a, b)
}

/// One memtile BD in a cyclic chain: `idx` of `len`, ids from `first`, acquiring `acq` and releasing `rel`
/// (memtile-local lock ids).
fn chain_bd(
    first: u32,
    idx: usize,
    len: usize,
    addr: u32,
    words: u32,
    acq: u32,
    rel: u32,
) -> (u32, Bd) {
    let mut bd = Bd::new((MEM_BASE + addr) as u64, words);
    bd.locks = BdLocks {
        acq: Some((MEM_LOCK + acq, -1)),
        rel: Some((MEM_LOCK + rel, 1)),
    };
    bd.next = Some(first + ((idx + 1) % len) as u32);
    (first + idx as u32, bd)
}

/// Memtile descriptors of the V9 / V10 resident designs (see the module header): the operand descriptors of the
/// mode followed by the shared compact C descriptors (single-BD C producers on rows 1 and 3).
fn resident_descriptors(res: Resident) -> Vec<(u32, Bd)> {
    let mut v = match res.mode {
        ResidentMode::NwOuter => nw_outer_operand_descriptors(res),
        ResidentMode::MwOuter => mw_outer_operand_descriptors(res),
    };
    v.extend(resident_c_descriptors(res.c_base()));
    v
}

/// V9 operand descriptors: resident B producer + replay chain, V8's A rings.
fn nw_outer_operand_descriptors(rb: Resident) -> Vec<(u32, Bd)> {
    let mut v = Vec::new();
    let ((a_empty, a_full), (b_empty, b_full)) = ab_locks();
    let (b_empty, b_full) = (b_empty[0], b_full[0]);
    let b_words = (rb.kc * B_BYTES / 4) as u32;
    let b_slots = [resident_b_offset(rb.kc, 0), resident_b_offset(rb.kc, 1)];
    v.extend(ring(
        B_S2MM_BD,
        b_slots,
        b_words,
        [b_empty; 2],
        [b_full; 2],
        true,
    ));
    // Replay chain: every BD reads slot 0 advanced by `current * slot stride`; `current` toggles at each load.
    let step = (b_slots[1] - b_slots[0]) / 4;
    for i in 0..rb.mw {
        let mut bd = Bd::new((MEM_BASE + b_slots[0]) as u64, b_words);
        bd.iteration = Iteration {
            step,
            wrap: 2,
            current: 0,
        };
        bd.locks = BdLocks {
            acq: (i == 0).then_some((MEM_LOCK + b_full, -1)),
            rel: (i == rb.mw - 1).then_some((MEM_LOCK + b_empty, 1)),
        };
        bd.next = Some(R_B_REPLAY_BD[(i + 1) % rb.mw]);
        v.push((R_B_REPLAY_BD[i], bd));
    }
    v.extend(ring(
        A_S2MM_BD,
        A_OFF,
        (A_HALF_BYTES / 4) as u32,
        a_empty,
        a_full,
        true,
    ));
    v.extend(ring(
        A_MM2S_BD,
        A_OFF,
        (A_HALF_BYTES / 4) as u32,
        a_empty,
        a_full,
        false,
    ));
    v
}

/// V10 operand descriptors: finite whole-segment B fill + cyclic B replay under one `B_FULL` semaphore (`MW`
/// credits), 2-slot A fill ring + `NW`-BD A replay chain with a slot-toggling iteration under `A_EMPTY` / `A_FULL`.
fn mw_outer_operand_descriptors(res: Resident) -> Vec<(u32, Bd)> {
    let mut v = Vec::new();
    let ((a_empty, a_full), (_, b_full)) = ab_locks();
    let (a_empty, a_full, b_full) = (a_empty[0], a_full[0], b_full[0]);
    let a_words = (res.kc * B_BYTES / 4) as u32;
    if res.b_prefix {
        v.extend(b_prefix_descriptors(res));
    } else {
        let b_words = (res.nw * res.kc * B_BYTES / 4) as u32;
        let b_off = mw_outer_b_offset(res.kc);
        let mut fill = Bd::new((MEM_BASE + b_off) as u64, b_words);
        fill.locks = BdLocks {
            acq: None,
            rel: Some((MEM_LOCK + b_full, res.mw as i32)),
        };
        v.push((B_S2MM_BD[0], fill));
        let mut replay = Bd::new((MEM_BASE + b_off) as u64, b_words);
        replay.locks = BdLocks {
            acq: Some((MEM_LOCK + b_full, -1)),
            rel: None,
        };
        replay.next = Some(B_MM2S_BD[0]);
        v.push((B_MM2S_BD[0], replay));
    }
    let slots = [mw_outer_a_offset(res.kc, 0), mw_outer_a_offset(res.kc, 1)];
    v.extend(ring(
        A_S2MM_BD,
        slots,
        a_words,
        [a_empty; 2],
        [a_full; 2],
        true,
    ));
    for i in 0..res.nw {
        let mut bd = Bd::new((MEM_BASE + slots[0]) as u64, a_words);
        bd.iteration = Iteration {
            step: a_words,
            wrap: 2,
            current: 0,
        };
        bd.locks = BdLocks {
            acq: (i == 0).then_some((MEM_LOCK + a_full, -1)),
            rel: (i == res.nw - 1).then_some((MEM_LOCK + a_empty, 1)),
        };
        bd.next = Some(V10_A_CHAIN_BD0 + ((i + 1) % res.nw) as u32);
        v.push((V10_A_CHAIN_BD0 + i as u32, bd));
    }
    v
}

/// Opt-in V10 B prefix: per chunk group a finite producer BD (`B_S2MM_BD[g]`, releases `B_FULL` +1) and a finite guarded
/// reader BD (`B_MM2S_BD[g]`, acquires `B_FULL` -1), both one chunk long with a wrapping iteration over the group, then the
/// whole-segment lockfree replay BD. With more than 63 chunks `B_EMPTY` additionally bounds the `B_FULL` credits.
fn b_prefix_descriptors(res: Resident) -> Vec<(u32, Bd)> {
    let (_, (b_empty, b_full)) = ab_locks();
    let (b_empty, b_full) = (b_empty[0], b_full[0]);
    let backpressure = res.b_prefix_backpressure();
    let chunk_words = (B_BYTES / 4) as u32;
    let base = mw_outer_b_offset(res.kc);
    let mut v = Vec::new();
    for (g, (first, len)) in res.b_prefix_groups().enumerate() {
        let addr = (MEM_BASE + base) as u64 + (first * B_BYTES) as u64;
        let iteration = Iteration {
            step: chunk_words,
            wrap: len as u32,
            current: 0,
        };
        let mut fill = Bd::new(addr, chunk_words);
        fill.iteration = iteration;
        fill.locks = BdLocks {
            acq: backpressure.then_some((MEM_LOCK + b_empty, -1)),
            rel: Some((MEM_LOCK + b_full, 1)),
        };
        v.push((B_S2MM_BD[g], fill));
        let mut read = Bd::new(addr, chunk_words);
        read.iteration = iteration;
        read.locks = BdLocks {
            acq: Some((MEM_LOCK + b_full, -1)),
            rel: backpressure.then_some((MEM_LOCK + b_empty, 1)),
        };
        v.push((B_MM2S_BD[g], read));
    }
    v.push((
        V10_B_LOCKFREE_BD,
        Bd::new(
            (MEM_BASE + base) as u64,
            (res.nw * res.kc * B_BYTES / 4) as u32,
        ),
    ));
    v
}

/// Compact C descriptors of the resident designs at memtile offset `base`: S2MM0/2 (rows 0, 2) two-BD rings with
/// distinct slot locks, S2MM1/3 (rows 1, 3) one iteration BD with the row-pair lock, and V8's two C drain chains.
fn resident_c_descriptors(base: u32) -> Vec<(u32, Bd)> {
    let mut v = Vec::new();
    let words = (COUT_BYTES / 4) as u32;
    for r in 0..ROWS {
        if r % 2 == 1 {
            let id = C_S2MM_BD[r][0];
            let lock = c_lock(0, r);
            let mut bd = Bd::new((MEM_BASE + resident_c_offset(base, 0, r)) as u64, words);
            bd.iteration = Iteration {
                step: R_C_SLOT_STRIDE / 4,
                wrap: 2,
                current: 0,
            };
            bd.locks = BdLocks {
                acq: Some((MEM_LOCK + lock, -1)),
                rel: Some((MEM_LOCK + lock + 1, 1)),
            };
            bd.next = Some(id);
            v.push((id, bd));
        } else {
            let offs = [resident_c_offset(base, 0, r), resident_c_offset(base, 1, r)];
            let empty = [c_lock(0, r), c_lock(1, r)];
            v.extend(ring(
                C_S2MM_BD[r],
                offs,
                words,
                empty,
                [empty[0] + 1, empty[1] + 1],
                true,
            ));
        }
    }
    // Two cyclic drain chains: channel g drains rows 2g, 2g+1 of slot 0, then slot 1; odd rows share one lock pair.
    for g in 0..2 {
        for slot in 0..2 {
            for rr in 0..2 {
                let row = 2 * g + rr;
                let lock = if row % 2 == 1 {
                    c_lock(0, row)
                } else {
                    c_lock(slot, row)
                };
                v.push(chain_bd(
                    PAIR_C_MM2S_BD0[g],
                    slot * 2 + rr,
                    4,
                    resident_c_offset(base, slot, row),
                    words,
                    lock + 1,
                    lock,
                ));
            }
        }
    }
    v
}

/// All memtile BDs of a column `(id, bd)`; independent of the problem size. Empty for pass-through.
fn memtile_descriptors(topo: Topo, col: u32) -> Vec<(u32, Bd)> {
    let mut v = Vec::new();
    let words = (topo.c_tile_bytes() / 4) as u32;
    match topo.kind {
        Kind::Pass => {}
        Kind::Column => {
            let a_words = (A_BYTES / 4) as u32;
            // A producer: one 8-BD chain, slot 0 rows 0..3 then slot 1 rows 0..3 (host chunk order).
            for i in 0..2 * ROWS {
                let (slot, row) = (i / ROWS, i % ROWS);
                v.push(chain_bd(
                    OC_A_PROD_BD0,
                    i,
                    2 * ROWS,
                    oc_a_offset(slot, row),
                    a_words,
                    oc_a_lock(slot, row),
                    oc_a_lock(slot, row) + 1,
                ));
            }
            for r in 0..ROWS {
                let empty = [oc_a_lock(0, r), oc_a_lock(1, r)];
                v.extend(ring(
                    OC_A_CONS_BD[r],
                    [oc_a_offset(0, r), oc_a_offset(1, r)],
                    a_words,
                    empty,
                    [empty[0] + 1, empty[1] + 1],
                    false,
                ));
                let empty = [oc_c_lock(0, r), oc_c_lock(1, r)];
                v.extend(ring(
                    OC_C_PROD_BD[r],
                    [oc_c_offset(0, r), oc_c_offset(1, r)],
                    words,
                    empty,
                    [empty[0] + 1, empty[1] + 1],
                    true,
                ));
            }
            let b_words = (B_BYTES / 4) as u32;
            v.extend(ring(
                OC_B_PROD_BD,
                OC_B_OFF,
                b_words,
                OC_B_EMPTY,
                OC_B_FULL,
                true,
            ));
            v.extend(ring(
                OC_B_CONS_BD,
                OC_B_OFF,
                b_words,
                OC_B_EMPTY,
                OC_B_FULL,
                false,
            ));
            for i in 0..2 * ROWS {
                let (slot, row) = (i / ROWS, i % ROWS);
                v.push(chain_bd(
                    OC_C_JOIN_BD0,
                    i,
                    2 * ROWS,
                    oc_c_offset(slot, row),
                    words,
                    oc_c_lock(slot, row) + 1,
                    oc_c_lock(slot, row),
                ));
            }
        }
        Kind::Staged => {
            let ((a_empty, a_full), (b_empty, b_full)) = ab_locks();
            v.extend(ring(
                B_S2MM_BD,
                B_OFF,
                (B_BYTES / 4) as u32,
                b_empty,
                b_full,
                true,
            ));
            v.extend(ring(
                B_MM2S_BD,
                B_OFF,
                (B_BYTES / 4) as u32,
                b_empty,
                b_full,
                false,
            ));
            if topo.has_a(col) {
                v.extend(ring(
                    A_S2MM_BD,
                    A_OFF,
                    (A_BYTES / 4) as u32,
                    a_empty,
                    a_full,
                    true,
                ));
                v.extend(ring(
                    A_MM2S_BD,
                    A_OFF,
                    (A_BYTES / 4) as u32,
                    a_empty,
                    a_full,
                    false,
                ));
            }
            let rows = topo.rows;
            for r in 0..rows {
                let offs = [c_offset(0, r), c_offset(1, r)];
                let empty = [c_lock(0, r), c_lock(1, r)];
                let full = [c_lock(0, r) + 1, c_lock(1, r) + 1];
                v.extend(ring(C_S2MM_BD[r], offs, words, empty, full, true));
            }
            // One cyclic MM2S chain drains slot 0 rows 0..rows then slot 1 rows 0..rows.
            for slot in 0..2 {
                for r in 0..rows {
                    v.push(chain_bd(
                        C_MM2S_BD0,
                        slot * rows + r,
                        2 * rows,
                        c_offset(slot, r),
                        words,
                        c_lock(slot, r) + 1,
                        c_lock(slot, r),
                    ));
                }
            }
        }
        Kind::Pair {
            resident: Some(res),
            ..
        } => v.extend(resident_descriptors(res)),
        Kind::Pair { resident: None, .. } => {
            let ((a_empty, a_full), (b_empty, b_full)) = ab_locks();
            v.extend(ring(
                B_S2MM_BD,
                B_OFF,
                (B_BYTES / 4) as u32,
                b_empty,
                b_full,
                true,
            ));
            v.extend(ring(
                B_MM2S_BD,
                B_OFF,
                (B_BYTES / 4) as u32,
                b_empty,
                b_full,
                false,
            ));
            v.extend(ring(
                A_S2MM_BD,
                A_OFF,
                (A_HALF_BYTES / 4) as u32,
                a_empty,
                a_full,
                true,
            ));
            v.extend(ring(
                A_MM2S_BD,
                A_OFF,
                (A_HALF_BYTES / 4) as u32,
                a_empty,
                a_full,
                false,
            ));
            for r in 0..ROWS {
                let offs = [c_offset(0, r), c_offset(1, r)];
                let empty = [c_lock(0, r), c_lock(1, r)];
                let full = [c_lock(0, r) + 1, c_lock(1, r) + 1];
                v.extend(ring(C_S2MM_BD[r], offs, words, empty, full, true));
            }
            // Two cyclic MM2S chains: channel g drains core rows 2g, 2g+1 of slot 0, then of slot 1.
            for g in 0..2 {
                for slot in 0..2 {
                    for rr in 0..2 {
                        let row = 2 * g + rr;
                        v.push(chain_bd(
                            PAIR_C_MM2S_BD0[g],
                            slot * 2 + rr,
                            4,
                            c_offset(slot, row),
                            words,
                            c_lock(slot, row) + 1,
                            c_lock(slot, row),
                        ));
                    }
                }
            }
        }
        Kind::G80 { .. } => return gemm_g80::memtile_descriptors(topo, col),
        Kind::Ief15 { .. } => return iu4_ief15::memtile_descriptors(col),
    }
    v
}

/// The V8 streaming-pair memtile descriptors of a column (int8 epilogue: C tile 8192 B), the map [`iu4_ief15`] reuses with
/// its own A / B ring lengths.
pub(crate) fn v8_pair_memtile_descriptors(col: u32) -> Vec<(u32, Bd)> {
    memtile_descriptors(Topo::pair(V8_DEFAULT_EPILOGUE), col)
}

/// BD ids of the A and B memtile rings (fill and replay): `(A, B)`.
pub(crate) fn pair_ab_ring_bds() -> ([u32; 4], [u32; 4]) {
    (
        [A_S2MM_BD[0], A_S2MM_BD[1], A_MM2S_BD[0], A_MM2S_BD[1]],
        [B_S2MM_BD[0], B_S2MM_BD[1], B_MM2S_BD[0], B_MM2S_BD[1]],
    )
}

/// Initial memtile lock values (empty = 1, full = 0) for every lock a column uses.
fn memtile_locks(topo: Topo, col: u32) -> Vec<(u32, u32)> {
    let mut v = Vec::new();
    match topo.kind {
        Kind::Pass => {}
        Kind::Column => {
            for slot in 0..2 {
                for r in 0..ROWS {
                    v.extend([(oc_a_lock(slot, r), 1), (oc_a_lock(slot, r) + 1, 0)]);
                    v.extend([(oc_c_lock(slot, r), 1), (oc_c_lock(slot, r) + 1, 0)]);
                }
            }
            for i in 0..2 {
                v.extend([(OC_B_EMPTY[i], 1), (OC_B_FULL[i], 0)]);
            }
        }
        Kind::Staged | Kind::Pair { resident: None, .. } | Kind::Ief15 { .. } => {
            let ((a_empty, a_full), (b_empty, b_full)) = ab_locks();
            for i in 0..2 {
                v.extend([(b_empty[i], 1), (b_full[i], 0)]);
                if topo.has_a(col) {
                    v.extend([(a_empty[i], 1), (a_full[i], 0)]);
                }
            }
            for slot in 0..2 {
                for r in 0..topo.rows {
                    v.extend([(c_lock(slot, r), 1), (c_lock(slot, r) + 1, 0)]);
                }
            }
        }
        Kind::Pair {
            resident: Some(res),
            ..
        } => {
            let ((a_empty, a_full), (b_empty, b_full)) = ab_locks();
            match res.mode {
                ResidentMode::NwOuter => {
                    v.extend([(b_empty[0], 2), (b_full[0], 0)]);
                    for i in 0..2 {
                        v.extend([(a_empty[i], 1), (a_full[i], 0)]);
                    }
                }
                ResidentMode::MwOuter => {
                    v.extend([(a_empty[0], 2), (a_full[0], 0), (b_full[0], 0)]);
                    if res.b_prefix && res.b_prefix_backpressure() {
                        v.push((b_empty[0], V10_B_CREDIT_MAX));
                    }
                }
            }
            for r in 0..ROWS {
                if r % 2 == 1 {
                    v.extend([(c_lock(0, r), 2), (c_lock(0, r) + 1, 0)]);
                } else {
                    for slot in 0..2 {
                        v.extend([(c_lock(slot, r), 1), (c_lock(slot, r) + 1, 0)]);
                    }
                }
            }
        }
        Kind::G80 { .. } => return gemm_g80::memtile_locks(topo, col),
    }
    v
}

/// Core and memtile ring tasks of a column (not the shim; those depend on the problem size).
fn ring_tasks(topo: Topo, col: u32) -> Vec<(Location, Task)> {
    // G80 queues its own core and memtile tasks (same shape as below, widths of the 128x80 core).
    if topo.is_g80() {
        return gemm_g80::ring_tasks(topo, col);
    }
    let mut v = Vec::new();
    let rep = |direction, channel, bd, repeat| Task {
        direction,
        channel,
        bd,
        repeat,
        issue_token: false,
    };
    let task = |direction, channel, bd| rep(direction, channel, bd, 1);
    for r in 0..topo.rows {
        let tile = core_loc(col, r);
        v.push((tile, task(S2mm, gemm_core::A_CHANNEL, gemm_core::A_BD[0])));
        v.push((tile, task(S2mm, gemm_core::B_CHANNEL, gemm_core::B_BD[0])));
        v.push((tile, task(Mm2s, gemm_core::C_CHANNEL, gemm_core::C_BD)));
    }
    let mem = mem_loc(col);
    match topo.kind {
        Kind::Pass => {}
        Kind::Column => {
            for r in 0..ROWS {
                v.push((mem, task(S2mm, r as u32, OC_C_PROD_BD[r][0])));
                v.push((mem, task(Mm2s, r as u32, OC_A_CONS_BD[r][0])));
            }
            v.push((mem, task(S2mm, OC_A_PROD_CH, OC_A_PROD_BD0)));
            v.push((mem, task(S2mm, OC_B_PROD_CH, OC_B_PROD_BD[0])));
            v.push((mem, task(Mm2s, OC_B_CONS_CH, OC_B_CONS_BD[0])));
            v.push((mem, task(Mm2s, OC_C_JOIN_CH, OC_C_JOIN_BD0)));
        }
        Kind::Staged => {
            for r in 0..topo.rows {
                v.push((mem, task(S2mm, r as u32, C_S2MM_BD[r][0])));
            }
            v.push((mem, task(S2mm, B_S2MM_CH, B_S2MM_BD[0])));
            v.push((mem, task(Mm2s, 0, C_MM2S_BD0)));
            v.push((mem, task(Mm2s, B_MM2S_CH, B_MM2S_BD[0])));
            if topo.has_a(col) {
                v.push((mem, task(S2mm, A_S2MM_CH, A_S2MM_BD[0])));
                v.push((mem, task(Mm2s, A_MM2S_CH, A_MM2S_BD[0])));
            }
        }
        Kind::Pair { .. } | Kind::Ief15 { .. } => {
            let prefix = topo.resident().filter(|r| r.b_prefix);
            for r in 0..ROWS {
                v.push((mem, task(S2mm, r as u32, C_S2MM_BD[r][0])));
            }
            match prefix {
                None => v.push((mem, task(S2mm, B_S2MM_CH, B_S2MM_BD[0]))),
                Some(res) => {
                    for (g, (_, len)) in res.b_prefix_groups().enumerate() {
                        v.push((mem, rep(S2mm, B_S2MM_CH, B_S2MM_BD[g], len as u32)));
                    }
                }
            }
            v.push((mem, task(S2mm, A_S2MM_CH, A_S2MM_BD[0])));
            for g in 0..2 {
                v.push((mem, task(Mm2s, PAIR_C_MM2S_CH[g], PAIR_C_MM2S_BD0[g])));
            }
            match prefix {
                None => {
                    let nw_outer = topo
                        .resident()
                        .is_some_and(|r| r.mode == ResidentMode::NwOuter);
                    let b_mm2s_bd = if nw_outer {
                        R_B_REPLAY_BD[0]
                    } else {
                        B_MM2S_BD[0]
                    };
                    v.push((mem, task(Mm2s, B_MM2S_CH, b_mm2s_bd)));
                }
                Some(res) => {
                    for (g, (_, len)) in res.b_prefix_groups().enumerate() {
                        v.push((mem, rep(Mm2s, B_MM2S_CH, B_MM2S_BD[g], len as u32)));
                    }
                    if res.mw > 1 {
                        v.push((
                            mem,
                            rep(Mm2s, B_MM2S_CH, V10_B_LOCKFREE_BD, (res.mw - 1) as u32),
                        ));
                    }
                }
            }
            v.push((mem, task(Mm2s, A_MM2S_CH, A_MM2S_BD[0])));
        }
        Kind::G80 { .. } => unreachable!("G80 ring tasks are built by gemm_g80"),
    }
    v
}

/// Persistent-ring DMA channels reset at every dynamic submit: `(tile, direction, channel)`. Shim channels
/// are excluded (see `build_txn_dynamic`).
fn used_channels(topo: Topo, col: u32) -> Vec<(Location, dma::Direction, u32)> {
    let mut v: Vec<_> = ring_tasks(topo, col)
        .into_iter()
        .map(|(t, task)| (t, task.direction, task.channel))
        .collect();
    v.sort_by_key(|(t, d, c)| (t.row, *d == Mm2s, *c));
    v.dedup();
    v
}

/// Channel CTRL register (bit1 reset, bit0 enable for tile/memtile; bits 8..15 token controller id on shim).
fn channel_ctrl(loc: Location, direction: dma::Direction, channel: u32) -> u32 {
    let bd = if loc.kind() == TileType::Memtile && channel % 2 == 1 {
        24
    } else {
        0
    };
    Task {
        direction,
        channel,
        bd,
        repeat: 1,
        issue_token: false,
    }
    .write(loc)
    .0 - 4
}

/// Per-core DMA descriptors as a fixed six-slot backing array (indexed by BD id) plus the valid count (G80 `Epilogue::I32` has
/// six, every other design five; emit `bds[..len]`) and initial lock values of `topo` (V8 uses the pair kernel's).
fn core_config(topo: Topo) -> (([[u32; 6]; 6], usize), [i32; 16]) {
    let five = |b: [[u32; 6]; 5]| {
        let mut bds = [[0; 6]; 6];
        bds[..5].copy_from_slice(&b);
        (bds, 5)
    };
    match topo.kind {
        Kind::G80 { n512: true, .. } => (
            gemm_core_g80::tile_bds_pair_n64(),
            gemm_core_g80::initial_locks(),
        ),
        Kind::G80 { epi, ctl, .. } => (
            gemm_core_g80::tile_bds_pair(epi, ctl),
            gemm_core_g80::initial_locks(),
        ),
        Kind::Ief15 { .. } => (
            five(iu4_ief15_core::tile_bds()),
            gemm_core::initial_locks_pair(),
        ),
        Kind::Pair { epi, ctl, .. } => (
            five(gemm_core::tile_bds_pair(epi, ctl)),
            gemm_core::initial_locks_pair(),
        ),
        _ => (five(gemm_core::tile_bds()), gemm_core::initial_locks()),
    }
}

/// Deployment image. `start = false` (V6, V6b, V8): program memory, BDs, routes and token route only; the TXN
/// resets and starts everything on every submit. `start = true` (V1..V5): additionally core reset release,
/// PC, every core/memtile lock, the compute/memtile ring tasks (queued and enabled) and the core enables
/// (last), as `gemm_i8::design` does; the TXN then only drives the shim. `programs` are the two core programs
/// (index 0 for every core except V8 upper-pair cores, which are the odd core rows and load index 1).
pub(crate) fn build_cdo(topo: Topo, programs: [&[u8]; 2], start: bool) -> Cdo {
    let mut cdo = Cdo::new();
    let words: Vec<Vec<u32>> = programs
        .iter()
        .map(|program| {
            program
                .chunks(4)
                .map(|chunk| {
                    let mut word = [0; 4];
                    word[..chunk.len()].copy_from_slice(chunk);
                    u32::from_le_bytes(word)
                })
                .collect()
        })
        .collect();
    let ((tile_bds, bd_count), core_locks) = core_config(topo);
    for col in 0..topo.cols as u32 {
        for r in 0..topo.rows {
            let tile = core_loc(col, r);
            // Halt + reset, quiesce the core DMAs while program memory is written.
            cdo.mask_write(tile.address(regs::core::CORE_CONTROL), 3, 2);
            let ctrls = [
                regs::core::DMA_S2MM_0_CTRL,
                regs::core::DMA_S2MM_1_CTRL,
                regs::core::DMA_MM2S_0_CTRL,
            ];
            for off in ctrls {
                cdo.mask_write(tile.address(off), 2, 2);
            }
            let words = &words[if topo.is_pair() { r % 2 } else { 0 }];
            cdo.dma_write(tile.address(regs::core::PROGRAM_MEMORY), words);
            for off in ctrls {
                cdo.mask_write(tile.address(off), 2, 0);
            }
            if start {
                cdo.mask_write(tile.address(regs::core::CORE_CONTROL), 2, 0)
                    .write(tile.address(regs::core::CORE_PC), 0);
                for (id, value) in core_locks.iter().enumerate() {
                    let (addr, value) = dma::lock_write(tile, id as u32, *value as u32);
                    cdo.write(addr, value);
                }
            }
            for (id, bd) in tile_bds[..bd_count].iter().enumerate() {
                cdo.dma_write(Bd::address(tile, id as u32), bd);
            }
        }
        for (id, bd) in memtile_descriptors(topo, col) {
            bd.emit_cdo(mem_loc(col), id, &mut cdo);
        }
        if start {
            for (id, value) in memtile_locks(topo, col) {
                let (addr, value) = dma::lock_write(mem_loc(col), id, value);
                cdo.write(addr, value);
            }
        }
    }
    for circuit in circuits(topo) {
        circuit.emit_cdo(&mut cdo);
    }
    for col in 0..topo.cols as u32 {
        for route in shim_routes(topo, col) {
            route.emit_cdo(&mut cdo);
        }
        for (addr, value) in regs::shim_token_route(col) {
            cdo.write(addr, value);
        }
    }
    if start {
        for col in 0..topo.cols as u32 {
            for (tile, task) in ring_tasks(topo, col) {
                task.emit_cdo(tile, &mut cdo);
                // AIE2P memtile CTRL has no ENABLE bit; its queue starts on enqueue.
                if tile.kind() == TileType::Compute {
                    task.enable_cdo(tile, &mut cdo);
                }
            }
        }
        for tile in topo.cores() {
            cdo.mask_write(tile.address(regs::core::CORE_CONTROL), 1, 1);
        }
    }
    cdo
}

/// Shim BDs (DDR-patched), token controller id and finite shim tasks for every shim of the topology.
fn emit_shim(txn: &mut Txn, geo: &Geometry) {
    emit_shim_at(txn, geo, [0; 3], true, false)
}

/// [`emit_shim`] with every DDR patch advanced by `base[arg]` bytes (the arena offset of that run in args 0..=2).
/// `!bd_words`: the shim BD words and S2MM controller ids written by an earlier body of the SAME design are still in
/// place, so only the DDR patches and tasks are emitted (the patch rewrites the one address word that differs).
/// `skip_b`: no B stream at all (B resident in the memtile, G80 B-resident lean body).
fn emit_shim_at(txn: &mut Txn, geo: &Geometry, base: [u64; 3], bd_words: bool, skip_b: bool) {
    let topo = geo.topo;
    let kc = geo.kc;
    for col in 0..topo.cols {
        let shim = shim_loc(col as u32);
        let (a_off, b_off, c_off) = geo.column_offsets(col);
        if bd_words {
            txn.mask_write(shim.address(regs::shim::DMA_S2MM_0_CTRL), 0xf00, 0x1f00);
            if topo.is_pair() {
                txn.mask_write(shim.address(regs::shim::DMA_S2MM_0_CTRL + 8), 0xf00, 0x1f00);
            }
        }
        // (bd id, argument, byte offset, words, direction, channel, token)
        let streams = if topo.is_pair() {
            // B MM2S0 (BD0), C rows 0,1 S2MM0 (BD1), A half MM2S1 (BD2), C rows 2,3 S2MM1 (BD3): the C argument
            // of a column is the S2MM0 stream followed by the S2MM1 stream, each `(wave, rho % 2)` tiles.
            let cb = topo.c_tile_bytes();
            let c_words = geo.waves * 2 * cb / 4;
            let mut s = vec![
                (0, 1, b_off, geo.b_segment_bytes() / 4, Mm2s, 0, false),
                (1, 2, c_off, c_words, S2mm, 0, true),
            ];
            // G80 injects A in columns 0..4 only.
            if topo.has_a(col as u32) {
                s.push((2, 0, a_off, geo.a_segment_bytes() / 4, Mm2s, 1, false));
            }
            s.push((3, 2, c_off + geo.waves * 2 * cb, c_words, S2mm, 1, true));
            s
        } else if topo.kind == Kind::Column {
            // MM2S0 = chunk/row interleaved A, MM2S1 = B, S2MM0 = joined C.
            vec![
                (
                    0,
                    0,
                    a_off,
                    geo.waves * kc * ROWS * A_BYTES / 4,
                    Mm2s,
                    0,
                    false,
                ),
                (1, 1, b_off, geo.waves * kc * B_BYTES / 4, Mm2s, 1, false),
                (2, 2, c_off, geo.waves * ROWS * C_BYTES / 4, S2mm, 0, true),
            ]
        } else {
            // (bd id, argument, byte offset, words, direction, channel, token)
            let mut s = vec![
                (0, 1, b_off, geo.b_segment_bytes() / 4, Mm2s, 0, false),
                (
                    1,
                    2,
                    c_off,
                    geo.waves * topo.rows * C_BYTES / 4,
                    S2mm,
                    0,
                    true,
                ),
            ];
            if topo.has_a(col as u32) {
                s.push((2, 0, a_off, geo.waves * kc * A_BYTES / 4, Mm2s, 1, false));
            }
            s
        };
        let streams: Vec<_> = streams
            .into_iter()
            .filter(|s| !(skip_b && s.1 == 1 || topo.bf16_rowmajor().is_some() && s.1 == 2))
            .collect();
        for &(id, arg, offset, words, ..) in &streams {
            if bd_words {
                Bd::new(0, words as u32).emit_txn(shim, id, txn);
            }
            txn.ddr_patch(
                Bd::address(shim, id) + 4,
                arg,
                offset as u64 + base[arg as usize],
            );
        }
        for &(bd, arg, _, _, direction, channel, token) in streams.iter().rev() {
            // `a_repeat` (pair designs): the host A segment is the `mw` tiles once; replay it for every `nw`.
            let repeat = if geo.a_repeat && arg == 0 {
                geo.nw as u32
            } else {
                1
            };
            Task {
                direction,
                channel,
                bd,
                repeat,
                issue_token: token,
            }
            .emit_txn(shim, txn);
        }
    }
    if geo.topo.bf16_rowmajor().is_some() {
        emit_rowmajor_batch(txn, geo, base[2], 0);
    }
}

/// The shim has 16 BDs TOTAL (not per channel). B=0 and A=2 stay live;
/// fourteen C BDs hold seven waves, seven per channel. One BD scatters both
/// 128x80 parity tiles: 40 words per row, 128 rows, then +40 words to parity1.
/// No wave stride is encoded: every wave has an absolute DDR patch.
const ROWMAJOR_BATCH_WAVES: usize = 7;
const ROWMAJOR_C_BDS: [[u32; 7]; 2] = [[1, 4, 6, 7, 8, 9, 10], [3, 5, 11, 12, 13, 14, 15]];

/// One finite chain per C channel, one completion token per batch. After
/// both SYNCs, only addresses need refreshing: iteration wrap=1 leaves
/// `current` at zero. A partial final batch cuts its last link only.
/// Each body's first batch rewrites all active C words, also restoring links
/// cut by the preceding run (including same-design lean/ring bodies).
fn emit_rowmajor_batch(txn: &mut Txn, geo: &Geometry, base: u64, wave: usize) {
    let (pitch, col0) = geo.topo.bf16_rowmajor().unwrap();
    let n512 = geo.topo.is_g80_n512();
    let count = (geo.waves - wave).min(ROWMAJOR_BATCH_WAVES);
    for col in 0..geo.topo.cols {
        let shim = shim_loc(col as u32);
        for (pair, ids) in ROWMAJOR_C_BDS.iter().enumerate() {
            for index in 0..count {
                let id = ids[index];
                // 128x80: both 80-column parity tiles of a channel adjoin (40 words each) in the same 128 rows;
                // 128x64 (N512): both 64-column halves of a channel adjoin (32 words each) in the same 128 rows.
                let (words, tile_words) = if n512 { (8192, 32) } else { (10240, 40) };
                let mut bd = Bd::new(0, words);
                bd.dims[0] = dma::Dim {
                    step: 1,
                    wrap: tile_words,
                };
                bd.dims[1] = dma::Dim {
                    step: (pitch / 2) as u32,
                    wrap: 128,
                };
                bd.dims[2] = dma::Dim {
                    step: tile_words,
                    wrap: 0,
                };
                bd.next = (index + 1 < count).then(|| ids[index + 1]);
                if wave == 0 {
                    bd.emit_txn(shim, id, txn);
                } else if index + 1 == count && count < ROWMAJOR_BATCH_WAVES {
                    txn.write32(Bd::address(shim, id) + 28, bd.shim_words()[7]);
                }
                let (row, column) = if n512 {
                    (
                        (wave + index) * G80_N512_WAVE_M + (2 * pair + col % 2) * 128,
                        col0 + col / 2 * 128,
                    )
                } else {
                    (
                        (wave + index) * G80_WAVE_M + pair * 128,
                        col0 + 2 * col * 80,
                    )
                };
                txn.ddr_patch(
                    Bd::address(shim, id) + 4,
                    2,
                    base + (2 * (row * pitch + column)) as u64,
                );
            }
            Task {
                direction: S2mm,
                channel: pair as u32,
                bd: ids[0],
                repeat: 1,
                issue_token: true,
            }
            .emit_txn(shim, txn);
        }
    }
}

fn emit_output_tail(txn: &mut Txn, geo: &Geometry, per_column: bool, base: u64) {
    emit_sync(txn, geo.topo, per_column);
    if geo.topo.bf16_rowmajor().is_some() {
        for wave in (ROWMAJOR_BATCH_WAVES..geo.waves).step_by(ROWMAJOR_BATCH_WAVES) {
            emit_rowmajor_batch(txn, geo, base, wave);
            emit_sync(txn, geo.topo, per_column);
        }
    }
}

/// Wait for the C token of every active column: one SYNC each (`ncol = 1`) or one aggregate SYNC. V8 waits for
/// the tokens of both shim S2MM channels of all columns (one aggregate SYNC per channel).
fn emit_sync(txn: &mut Txn, topo: Topo, per_column: bool) {
    if topo.is_pair() {
        for chan in 0..2 {
            txn.sync(0, 0, 0, chan, topo.cols as u32, 1);
        }
    } else if per_column {
        for col in 0..topo.cols as u32 {
            txn.sync(col, 0, 0, 0, 1, 1);
        }
    } else {
        txn.sync(0, 0, 0, 0, topo.cols as u32, 1);
    }
}

/// Static variants: shim configuration, tasks and syncs only.
fn build_txn_static(geo: &Geometry, per_column_sync: bool) -> Txn {
    let mut txn = Txn::aie2p_8col();
    emit_shim(&mut txn, geo);
    emit_sync(&mut txn, geo.topo, per_column_sync);
    txn
}

/// Dynamic variants (V6, V6b): reset and restart every core/ring/lock on each submit.
pub(crate) fn build_txn_dynamic(geo: &Geometry, per_column_sync: bool) -> Txn {
    let mut txn = Txn::aie2p_8col();
    emit_dynamic_body(&mut txn, geo, per_column_sync, [0; 3]);
    txn
}

/// The reset-and-restart body of [`build_txn_dynamic`] appended to `txn`, with every shim DDR patch advanced by
/// `base[arg]` (the arena offset of this run in args 0..=2). Ends with the C token SYNC(s).
fn emit_dynamic_body(txn: &mut Txn, geo: &Geometry, per_column_sync: bool, base: [u64; 3]) {
    let topo = geo.topo;
    let cores: Vec<Location> = topo.cores().collect();
    let channels: Vec<_> = (0..topo.cols as u32)
        .flat_map(|c| used_channels(topo, c))
        .collect();
    // 1. halt + reset cores.
    for &tile in &cores {
        txn.mask_write(tile.address(regs::core::CORE_CONTROL), 2, 3);
    }
    // 2. reset compute/memtile DMA channels (CTRL bit1 RESET, ref/aie-rt xaie2pgbl_params.h), then release it.
    // Shim channels are not reset (aie-rt rejects shim reset; bit1 there is pause-mem): their finite
    // queues drained before the C token that ended the previous submit.
    for &(loc, dir, ch) in &channels {
        txn.mask_write(channel_ctrl(loc, dir, ch), 2, 2);
    }
    for &(loc, dir, ch) in &channels {
        txn.mask_write(channel_ctrl(loc, dir, ch), 0, 2);
    }
    // V9 / V10 / G80: the iteration `current` of a memtile BD advances at every load and is not cleared by a channel reset, so
    // every submit rewrites all resident (G80: all) descriptors (after the reset, before any queue).
    if topo.resident().is_some() || topo.is_g80() {
        for col in 0..topo.cols as u32 {
            for (id, bd) in memtile_descriptors(topo, col) {
                bd.emit_txn(mem_loc(col), id, txn);
            }
        }
    }
    // 3. locks.
    let core_locks = core_config(topo).1;
    for &tile in &cores {
        for (id, value) in core_locks.iter().enumerate() {
            let (addr, value) = dma::lock_write(tile, id as u32, *value as u32);
            txn.write32(addr, value);
        }
    }
    for col in 0..topo.cols as u32 {
        for (id, value) in memtile_locks(topo, col) {
            let (addr, value) = dma::lock_write(mem_loc(col), id, value);
            txn.write32(addr, value);
        }
    }
    // 4. release core reset, PC = 0.
    for &tile in &cores {
        txn.mask_write(tile.address(regs::core::CORE_CONTROL), 0, 2);
        txn.write32(tile.address(regs::core::CORE_PC), 0);
    }
    // 5. rings, then shim.
    for col in 0..topo.cols as u32 {
        for (tile, task) in ring_tasks(topo, col) {
            task.emit_txn(tile, txn);
            // AIE2P memtile CTRL has no ENABLE bit; its queue starts on enqueue.
            if tile.kind() == TileType::Compute {
                task.enable_txn(tile, txn);
            }
        }
    }
    emit_shim_at(txn, geo, base, true, false);
    // 6. cores last, then wait for the C token(s).
    for &tile in &cores {
        txn.mask_write(tile.address(regs::core::CORE_CONTROL), 1, 1);
    }
    emit_output_tail(txn, geo, per_column_sync, base[2]);
}

/// Built deployment image + command stream; no NPU access occurs here.
pub struct ArrayDesign {
    pub pdi: Vec<u8>,
    pub insts: Vec<u8>,
    /// arg0 = packed A (In), arg1 = packed B (In), arg2 = packed C (Out).
    pub args: Vec<ArgSpec>,
    pub(crate) geo: Geometry,
    pub(crate) variant: Variant,
    /// Core program of every core (V8: the `Lower` program of the even core rows).
    pub(crate) program: Vec<u8>,
    /// V8 `Upper` program of the odd core rows (empty for every other variant).
    pub(crate) upper_program: Vec<u8>,
    pub(crate) core: CoreVariant,
    /// V8 / G80 epilogue and control discipline.
    pub(crate) v8: Option<(Epilogue, Control)>,
    /// Shim AXI attributes currently encoded in `insts`.
    pub(crate) shim_axi: dma::ShimAxi,
}

impl ArrayDesign {
    /// Set burst length, AxCACHE and AxQoS of every shim BD descriptor, in place in [`ArrayDesign::insts`]
    /// (default [`dma::ShimAxi::default`]: burst 3, cache 2, qos 0). Only each shim descriptor's word 4 burst bits
    /// and word 5 cache/qos bits are rewritten; dimensions, DDR patches, queues and all other bytes are preserved,
    /// nothing is allocated, and the value is absolute (setting a new value replaces the previous one, including
    /// back to the default). Atomic: an invalid `axi` or a malformed/unsupported TXN returns `Err` with `insts`
    /// untouched. [`ArrayDesign::pdi`] is unchanged because shim BDs are rewritten by the TXN on every submit.
    /// Call before configuring/submitting. The attributes are hints to the memory system; they carry no
    /// simulator performance or coherence guarantee. Non-vendor word-5 attributes are rejected.
    pub fn set_shim_axi(&mut self, axi: dma::ShimAxi) -> Result<(), String> {
        axi.validate_vendor_word5()?;
        if axi == self.shim_axi {
            return Ok(());
        }
        super::gemm_i8::set_shim_axi_txn(&mut self.insts, axi)?;
        self.shim_axi = axi;
        Ok(())
    }
    pub fn waves(&self) -> usize {
        self.geo.waves
    }
    pub fn variant(&self) -> Variant {
        self.variant
    }
    /// Active core rows / columns (the variant's wave is `rows*128 x cols*64`).
    pub fn active_rows(&self) -> usize {
        self.geo.topo.rows
    }
    pub fn active_cols(&self) -> usize {
        self.geo.topo.cols
    }
    /// Exact core program bytes loaded into every active core (PC 0 = byte 0).
    pub fn core_program(&self) -> &[u8] {
        &self.program
    }
    /// V8 only: the `Upper` pair program loaded into the odd core rows (1, 3); `core_program` is `Lower`.
    pub fn core_program_upper(&self) -> Option<&[u8]> {
        (!self.upper_program.is_empty()).then_some(&self.upper_program[..])
    }
    /// V8 epilogue (`None` for every other variant).
    pub fn epilogue(&self) -> Option<Epilogue> {
        self.v8.map(|(e, _)| e)
    }
    /// V8 core control discipline (`None` for every other variant).
    pub fn control(&self) -> Option<Control> {
        self.v8.map(|(_, c)| c)
    }
    /// Core program variant the image was built with ([`CoreVariant::Fast`] unless made by
    /// [`design_variant_core`]).
    pub fn core_variant(&self) -> CoreVariant {
        self.core
    }
    /// True iff the CDO holds every core/ring/lock start (V1..V5): the TXN only drives the shim, so the
    /// design is valid for a fresh hardware context only (one submit).
    pub fn static_config(&self) -> bool {
        self.variant.is_static()
    }
    /// Real (unpadded) operations: 2*m*n*k.
    pub fn useful_ops(&self) -> u64 {
        2 * self.geo.m as u64 * self.geo.n as u64 * self.geo.k as u64
    }
    /// Bytes the shim DMAs read from DDR (packed A + packed B, including replication/padding).
    pub fn host_bytes_read(&self) -> u64 {
        (self.geo.a_bytes() + self.geo.b_bytes()) as u64
    }
    /// Bytes the shim DMAs write to DDR (padded packed C).
    pub fn host_bytes_written(&self) -> u64 {
        (self.geo.topo.cols * self.geo.c_segment_bytes()) as u64
    }

    /// Active compute tiles `(col, physical row)` (physical row = core row + 2), column-major.
    pub fn output_tiles(&self) -> Vec<(u32, u32)> {
        self.geo.topo.cores().map(|t| (t.col, t.row)).collect()
    }

    /// Per active compute tile `(col, physical row)`: the packed-C (arg2) byte range of each wave, in wave
    /// order. Native ranges cover arg2 exactly. Row-major BF16 returns one
    /// contiguous 80-element row range per tile row (padding columns excluded).
    pub fn output_tile_ranges(&self) -> Vec<((u32, u32), Vec<std::ops::Range<usize>>)> {
        let g = &self.geo;
        let len = g.topo.c_tile_bytes();
        g.topo
            .cores()
            .map(|t| {
                let (col, r) = (t.col as usize, t.row as usize - 2);
                if let Some((pitch, col0)) = g.topo.bf16_rowmajor() {
                    let n512 = g.topo.is_g80_n512();
                    let ranges = (0..g.waves)
                        .flat_map(|w| {
                            (0..128).map(move |i| {
                                if n512 {
                                    let row =
                                        w * G80_N512_WAVE_M + (2 * (r / 2) + col % 2) * 128 + i;
                                    let base =
                                        2 * (row * pitch + col0 + col / 2 * 128 + (r % 2) * 64);
                                    return base..base + 128;
                                }
                                let row = w * G80_WAVE_M + r / 2 * 128 + i;
                                let base = 2 * (row * pitch + col0 + (2 * col + r % 2) * 80);
                                base..base + 160
                            })
                        })
                        .collect();
                    return ((t.col, t.row), ranges);
                }
                let ranges = (0..g.waves)
                    .map(|w| {
                        let base = if g.topo.is_pair() {
                            g.pair_c_offset(col, w, r)
                        } else {
                            ((col * g.waves + w) * g.topo.rows + r) * C_BYTES
                        };
                        base..base + len
                    })
                    .collect();
                ((t.col, t.row), ranges)
            })
            .collect()
    }

    /// Packed `[A, B]` for arg0 and arg1; `a` is row-major m*k, `b` row-major k*n.
    pub fn pack_in(&self, a: &[i8], b: &[i8]) -> [Vec<u8>; 2] {
        assert!(
            !self.geo.topo.is_g80_bf16(),
            "BF16 inputs require explicit scale sidecars"
        );
        [self.geo.pack_a(a), self.geo.pack_b(b)]
    }
    /// Pack only arg0, without allocating the unused weight stream.
    pub fn pack_a(&self, a: &[i8]) -> Vec<u8> {
        assert!(
            !self.geo.topo.is_g80_bf16(),
            "BF16 activations require explicit scale sidecars"
        );
        self.geo.pack_a(a)
    }
    /// Pack only arg1, without allocating the unused activation stream.
    pub fn pack_b(&self, b: &[i8]) -> Vec<u8> {
        assert!(
            !self.geo.topo.is_g80_bf16(),
            "BF16 weights require explicit scale sidecars"
        );
        self.geo.pack_b(b)
    }
    /// V8 only (streaming B): packed `[A, B]` for the batched GEMM in which the rows of M-wave `mw` (512 rows of
    /// the row-major m*k `a`) are multiplied by the row-major k*n matrix `b_of_mw(mw)`. Output layout and
    /// [`ArrayDesign::unpack_out`] are those of [`ArrayDesign::pack_in`].
    pub fn pack_in_per_mw<'a>(
        &self,
        a: &[i8],
        b_of_mw: &dyn Fn(usize) -> &'a [i8],
    ) -> [Vec<u8>; 2] {
        assert_eq!(
            self.variant,
            Variant::V8,
            "per-M-wave B needs the streaming V8 design"
        );
        assert_eq!(a.len(), self.geo.m * self.geo.k, "A must be m*k");
        [self.geo.pack_a(a), self.geo.pack_b_per_mw(b_of_mw)]
    }
    /// Rows of one M-wave.
    pub fn wave_m(&self) -> usize {
        self.geo.topo.wave_m()
    }
    /// Row-major m*n C from the packed arg2 buffer (padding discarded). V8 with `Epilogue::Int8`: the int8
    /// results sign extended to `i32`.
    pub fn unpack_out(&self, output: &[u8]) -> Vec<i32> {
        self.geo.unpack_c(output)
    }
    /// Exact CPU result in the form [`ArrayDesign::unpack_out`] returns: the plain int32 GEMM, for V8 with the
    /// design's epilogue applied ([`apply_epilogue`]).
    pub fn reference(&self, a: &[i8], b: &[i8]) -> Vec<i32> {
        let c = super::gemm_i8::cpu_reference(a, b, self.geo.m, self.geo.n, self.geo.k);
        match self.v8 {
            Some((epi, _)) => c.into_iter().map(|x| apply_epilogue(x, epi)).collect(),
            None => c,
        }
    }

    /// ESTIMATE (phase-sum model, see [`CycleEstimate`]; V8 uses the pipelined model of [`estimate_v8`]).
    pub fn estimate(&self, clock_hz: u64) -> CycleEstimate {
        assert!(clock_hz > 0);
        assert!(
            !self.geo.topo.is_ief15(),
            "no cycle model for IEF15: it is measured, not estimated"
        );
        if self.geo.topo.is_g80() {
            let ctl = self.v8.map_or(Control::Fast, |(_, ctl)| ctl);
            return gemm_g80::estimate(&self.geo, ctl, clock_hz);
        }
        if let Some((epi, ctl)) = self.v8 {
            return self.estimate_v8(clock_hz, epi, ctl);
        }
        let chunks = (self.geo.kc * self.geo.waves) as u64;
        let ideal_mac_cycles = (TM * TN * CHUNK_K / 512) as u64 * chunks;
        let input_dma_cycles = (A_BYTES / 4) as u64 * chunks;
        let output_dma_cycles = (self.geo.topo.rows * C_BYTES / 4) as u64 * self.geo.waves as u64;
        let estimated_cycles = input_dma_cycles.max(ideal_mac_cycles) + output_dma_cycles;
        let estimated_seconds = estimated_cycles as f64 / clock_hz as f64;
        CycleEstimate {
            ideal_mac_cycles,
            input_dma_cycles,
            output_dma_cycles,
            estimated_cycles,
            estimated_seconds,
            useful_tops: self.useful_ops() as f64 / estimated_seconds / 1e12,
            peak_ops_per_second: peak_ops_per_second(clock_hz),
        }
    }

    /// V8 ESTIMATE (never a measurement). Per K64 chunk a core needs `max(PAIR_CHUNK_CYCLES, 1024)`: the kernel's
    /// issue cycles including lock/control overhead ([`gemm_core::PAIR_CHUNK_CYCLES`], ping and pong slot
    /// averaged, per control discipline) against the 1024-cycle A-half and B input streams (4 KiB at 4 B/cycle
    /// each). Per wave the core additionally pays the C epilogue: `Epilogue::I32` the 8192-cycle drain of the
    /// resident 32 KiB C (the next wave's first chunk waits for `C_EMPTY`), `Epilogue::Int8` the conversion
    /// ([`gemm_core::PAIR_EPILOGUE_CYCLES`]; the Cout drain overlaps the next wave). The memtile->shim C transfer
    /// (2 channels x 4 B/cycle, each carrying two core tiles) overlaps the next wave through the 2 memtile
    /// slots, so a wave period is `max(core wave cycles, shim C cycles)` and the last wave's shim transfer is
    /// exposed. Ignores pipeline fill, lock waits beyond this, DDR contention and start-up.
    /// `output_dma_cycles` reports the per-wave core epilogue cycles summed over waves.
    /// V9 reuses this model unchanged (explicit V9 model choice): the per-core chunk period is the same compute /
    /// 1024-cycle A and B stream bound, because V9 only changes how often the shim reads B from DDR, not the
    /// per-core stream rates; DDR contention is not modelled, so the V9 gain shows only in
    /// [`ArrayDesign::host_bytes_read`] (actual packed bytes), not in `estimated_cycles`.
    /// V10 uses the same steady-state model plus an explicit startup of `NW*kc*1024` cycles (model): B replay (and so
    /// the first core wave) waits for the whole B segment (`NW*kc*4096` B at 4 B/cycle, which exceeds the first A slot
    /// fill `kc*1024`); the DDR saving only shows in `host_bytes_read`, other pipeline fill and contention are ignored.
    fn estimate_v8(&self, clock_hz: u64, epi: Epilogue, ctl: Control) -> CycleEstimate {
        let g = &self.geo;
        let (chunks, waves) = ((g.kc * g.waves) as u64, g.waves as u64);
        let input_chunk = (A_HALF_BYTES / 4) as u64;
        let c = usize::from(ctl.is_slow());
        let [ping, pong] = gemm_core::PAIR_CHUNK_CYCLES[c].map(|x| x.max(input_chunk));
        let (core_epilogue, shim_c) = match epi {
            Epilogue::I32 => ((C_BYTES / 4) as u64, (2 * C_BYTES / 4) as u64),
            Epilogue::Int8 { .. } => (
                gemm_core::PAIR_EPILOGUE_CYCLES[c],
                (2 * COUT_BYTES / 4) as u64,
            ),
            Epilogue::Bf16 => unreachable!("direct BF16 is a G80-only epilogue"),
        };
        let wave_core = (g.kc as u64 * (ping + pong)).div_ceil(2) + core_epilogue;
        let startup = if self.variant == Variant::V10 {
            (g.nw * g.kc * B_BYTES / 4) as u64
        } else {
            0
        };
        let estimated_cycles = startup + (waves - 1) * wave_core.max(shim_c) + wave_core + shim_c;
        let estimated_seconds = estimated_cycles as f64 / clock_hz as f64;
        CycleEstimate {
            ideal_mac_cycles: (TM * TN * CHUNK_K / 512) as u64 * chunks,
            input_dma_cycles: input_chunk * chunks,
            output_dma_cycles: core_epilogue * waves,
            estimated_cycles,
            estimated_seconds,
            useful_tops: self.useful_ops() as f64 / estimated_seconds / 1e12,
            peak_ops_per_second: peak_ops_per_second(clock_hz),
        }
    }
}

/// Build the whole-array design for a row-major `m x k` by `k x n` int8 GEMM.
/// M and N are padded to 512, `k` is a multiple of 64 in 64..=10240 (`MAX_KC`), at most 256 waves.
/// This is the deployment [`Variant::V6`] with the [`CoreVariant::Fast`] core program.
pub fn design(m: usize, n: usize, k: usize) -> ArrayDesign {
    design_variant(m, n, k, Variant::V6)
}

/// Build `variant` (hardware bisect, see the module header) with the default [`CoreVariant::Fast`] core
/// program. M and N are padded to the variant's wave (`Variant::shape`); the arguments are always A, B, C.
pub fn design_variant(m: usize, n: usize, k: usize, variant: Variant) -> ArrayDesign {
    design_variant_core(m, n, k, variant, CoreVariant::Fast)
}

/// Build `variant` with an explicit core program ([`gemm_core::program_variant`]). Only the program loaded into
/// every active core differs: the DMA rings, locks, switches, host layouts and TXN are those of `variant`.
/// `Serial` computes the same exact GEMM (output identical to `Fast`); `LockOnly` skips the multiply
/// and fills the first 64 `i32` words of every tile's C frame (see the module header) - its output is
/// NOT a GEMM and [`ArrayDesign::unpack_out`] of it is meaningless.
pub fn design_variant_core(
    m: usize,
    n: usize,
    k: usize,
    variant: Variant,
    core: CoreVariant,
) -> ArrayDesign {
    design_variant_probe(m, n, k, variant, core, 0)
}

/// [`design_variant_core`] with the `ClockProbe` iteration count (`probe_iters` per tile; see
/// [`gemm_core::program_probe`]). `probe_iters` is only meaningful (and must be 0 otherwise) for
/// [`CoreVariant::ClockProbe`], whose output is the `LockOnly` pattern.
pub fn design_variant_probe(
    m: usize,
    n: usize,
    k: usize,
    variant: Variant,
    core: CoreVariant,
    probe_iters: u32,
) -> ArrayDesign {
    assert!(
        core == CoreVariant::ClockProbe || probe_iters == 0,
        "probe_iters needs the ClockProbe core"
    );
    assert!(
        variant != Variant::Ief15,
        "IEF15 is built by iu4_ief15::Ief15Gemm::new, not design_variant"
    );
    if matches!(
        variant,
        Variant::V8 | Variant::V9 | Variant::V10 | Variant::G80
    ) {
        let ctl = match core {
            CoreVariant::Fast => Control::Fast,
            CoreVariant::FastSlowCtl => Control::Slow,
            other => {
                panic!("{variant} supports the Fast and FastSlowCtl cores only, not {other:?}")
            }
        };
        return match variant {
            Variant::G80 => gemm_g80::design_g80(m, n, k, V8_DEFAULT_EPILOGUE, ctl),
            Variant::V10 => design_v10(m, n, k, V8_DEFAULT_EPILOGUE, ctl),
            Variant::V9 => design_v9(m, n, k, V8_DEFAULT_EPILOGUE, ctl),
            _ => design_v8(m, n, k, V8_DEFAULT_EPILOGUE, ctl),
        };
    }
    let topo = variant.topo();
    let geo = Geometry::new(topo, m, n, k);
    let program = if core == CoreVariant::ClockProbe {
        gemm_core::program_probe(geo.kc, geo.waves, probe_iters)
    } else {
        gemm_core::program_variant(geo.kc, geo.waves, core)
    }
    .finish();
    assert!(
        program.len() <= 16 * 1024,
        "core program {} B exceeds 16 KiB",
        program.len()
    );
    let insts = if variant.is_static() {
        build_txn_static(&geo, variant.per_column_sync())
    } else {
        build_txn_dynamic(&geo, variant.per_column_sync())
    };
    ArrayDesign {
        pdi: crate::pdi::build(
            &build_cdo(topo, [&program, &program], variant.is_static()).to_words(),
        ),
        insts: insts.to_bytes(),
        args: vec![
            ArgSpec {
                bytes: geo.a_bytes(),
                kind: ArgKind::In,
            },
            ArgSpec {
                bytes: geo.b_bytes(),
                kind: ArgKind::In,
            },
            ArgSpec {
                bytes: geo.c_bytes(),
                kind: ArgKind::Out,
            },
        ],
        geo,
        variant,
        program,
        upper_program: Vec::new(),
        core,
        v8: None,
        shim_axi: dma::ShimAxi::default(),
    }
}

/// The C epilogue [`Variant::V8`] gets from [`design_variant`] / [`design_variant_core`].
pub const V8_DEFAULT_EPILOGUE: Epilogue = Epilogue::Int8 { shift: 12 };

/// `Epilogue::Int8 { shift }` reference: `clamp(floor(c / 2^shift), -128, 127)` (arithmetic shift, saturate);
/// identity for `Epilogue::I32`.
pub fn apply_epilogue(c: i32, epi: Epilogue) -> i32 {
    match epi {
        Epilogue::I32 => c,
        Epilogue::Int8 { shift } => (i64::from(c) >> shift.min(63)).clamp(-128, 127) as i32,
        Epilogue::Bf16 => panic!("direct BF16 requires per-feature and per-token scales"),
    }
}

/// Build the V8 whole-array design (vertical pair A-sharing, see the module header): M and N padded to 512,
/// `k` a multiple of 64 in 64..=10240 (`MAX_KC`), at most 256 waves. The C argument holds int32 tiles for
/// `Epilogue::I32` and int8 tiles for `Epilogue::Int8`; `ctl` is the core control discipline.
pub fn design_v8(m: usize, n: usize, k: usize, epi: Epilogue, ctl: Control) -> ArrayDesign {
    design_pair(
        Variant::V8,
        Geometry::new(Topo::pair_control(epi, ctl), m, n, k),
        epi,
        ctl,
    )
}

/// Build the V9 whole-array design (resident B, N-wave outer, see the module header "V9 data movement"): the V8
/// core programs and C layout with B kept in the memtile per N-wave. `Epilogue::Int8` only, `MW <= 16`,
/// `NW <= 8`, `kc <= 54` (`K <= 3456`). The B argument holds each `nw` tile once (`NW*kc*4096` B per column).
pub fn design_v9(m: usize, n: usize, k: usize, epi: Epilogue, ctl: Control) -> ArrayDesign {
    assert!(
        matches!(epi, Epilogue::Int8 { .. }),
        "V9 supports the int8 epilogue only, got {epi:?}"
    );
    let base = Geometry::new(Topo::pair_control(epi, ctl), m, n, k);
    assert!(
        base.kc <= V9_MAX_KC,
        "V9 K={k} needs kc={} > {V9_MAX_KC} (two resident B slots must fit the memtile; max K {})",
        base.kc,
        V9_MAX_KC * CHUNK_K
    );
    assert!(
        base.mw <= V9_MAX_MW,
        "V9 MW={} exceeds {V9_MAX_MW} (one replay BD per M-wave)",
        base.mw
    );
    assert!(
        base.nw <= V9_MAX_NW,
        "V9 NW={} exceeds {V9_MAX_NW}",
        base.nw
    );
    let res = Resident {
        mode: ResidentMode::NwOuter,
        kc: base.kc,
        mw: base.mw,
        nw: base.nw,
        b_prefix: false,
    };
    let geo = Geometry {
        topo: Topo::with_resident(epi, ctl, res),
        ..base
    };
    design_pair(Variant::V9, geo, epi, ctl)
}

/// Build the V10 whole-array design (resident A and B, M-wave outer, see the module header "V10 data movement"): the
/// V8 core programs and C layout with every A tile and the whole B segment read from DDR once. `Epilogue::Int8` only,
/// `MW <= 16`, `NW <= 8` and the memtile bound `kc*(2 + NW) <= 112`. The A argument holds each `mw` tile once
/// (`MW*kc*4096` B per column), the B argument each `nw` tile once (`NW*kc*4096` B per column).
pub fn design_v10(m: usize, n: usize, k: usize, epi: Epilogue, ctl: Control) -> ArrayDesign {
    design_v10_with(m, n, k, epi, ctl, false)
}

/// V10 with the opt-in chunk-granular B prefix (module header "Opt-in V10 B prefix"): the same shape
/// domain, host layout, locks for A and C as [`design_v10`].
pub fn design_v10_prefix(m: usize, n: usize, k: usize, epi: Epilogue, ctl: Control) -> ArrayDesign {
    design_v10_with(m, n, k, epi, ctl, true)
}

fn design_v10_with(
    m: usize,
    n: usize,
    k: usize,
    epi: Epilogue,
    ctl: Control,
    b_prefix: bool,
) -> ArrayDesign {
    assert!(
        matches!(epi, Epilogue::Int8 { .. }),
        "V10 supports the int8 epilogue only, got {epi:?}"
    );
    let base = Geometry::new(Topo::pair_control(epi, ctl), m, n, k);
    assert!(
        base.mw <= V10_MAX_MW,
        "V10 MW={} exceeds {V10_MAX_MW}",
        base.mw
    );
    assert!(
        base.nw <= V10_MAX_NW,
        "V10 NW={} exceeds {V10_MAX_NW}",
        base.nw
    );
    assert!(base.kc * (2 + base.nw) <= V10_MAX_SLOTS,
        "V10 memtile overflow: kc={} (K={k}) x (2 A slots + NW={} B tiles) = {} 4 KiB slots exceeds {V10_MAX_SLOTS} \
         (max kc for NW={} is {})",
        base.kc, base.nw, base.kc * (2 + base.nw), base.nw, V10_MAX_SLOTS / (2 + base.nw));
    let res = Resident {
        mode: ResidentMode::MwOuter,
        kc: base.kc,
        mw: base.mw,
        nw: base.nw,
        b_prefix,
    };
    if b_prefix {
        let queued = res.b_prefix_group_count() + usize::from(res.mw > 1);
        assert!(
            queued <= V10_B_QUEUE_MAX,
            "V10 B prefix queues {queued} MM2S tasks, memtile queue holds {V10_B_QUEUE_MAX}"
        );
    }
    let geo = Geometry {
        topo: Topo::with_resident(epi, ctl, res),
        ..base
    };
    design_pair(Variant::V10, geo, epi, ctl)
}

/// Shared V8 / V9 / V10 pair design builder: identical core programs and `v8` metadata; `geo.topo` selects the memtile
/// data movement.
fn design_pair(variant: Variant, geo: Geometry, epi: Epilogue, ctl: Control) -> ArrayDesign {
    let topo = geo.topo;
    let program = |role| gemm_core::program_pair(geo.kc, geo.waves, role, epi, ctl).finish();
    let (lower, upper) = (program(PairRole::Lower), program(PairRole::Upper));
    for p in [&lower, &upper] {
        assert!(
            p.len() <= 16 * 1024,
            "core program {} B exceeds 16 KiB",
            p.len()
        );
    }
    let insts = build_txn_dynamic(&geo, false);
    ArrayDesign {
        pdi: crate::pdi::build(&build_cdo(topo, [&lower, &upper], false).to_words()),
        insts: insts.to_bytes(),
        args: vec![
            ArgSpec {
                bytes: geo.a_bytes(),
                kind: ArgKind::In,
            },
            ArgSpec {
                bytes: geo.b_bytes(),
                kind: ArgKind::In,
            },
            ArgSpec {
                bytes: geo.c_bytes(),
                kind: ArgKind::Out,
            },
        ],
        geo,
        variant,
        program: lower,
        upper_program: upper,
        core: if ctl.is_slow() {
            CoreVariant::FastSlowCtl
        } else {
            CoreVariant::Fast
        },
        v8: Some((epi, ctl)),
        shim_axi: dma::ShimAxi::default(),
    }
}

// ---- Persistent-ring support (`kernels::ring`): additive APIs only; the deployment encodings above are unchanged. ----

/// Shim BD ids `0..PAIR_SHIM_BDS_USED` are used by every pair design (B, C rows 0/1, A, C rows 2/3); higher ids are free.
pub const PAIR_SHIM_BDS_USED: u32 = 4;
/// DMA address of memtile byte offset 0 (own memory view) as used in memtile BD addresses.
pub const MEMTILE_DMA_BASE: u32 = MEM_BASE;

impl ArrayDesign {
    /// Append one complete reset-and-restart run of this dynamic design to `txn` (every core/ring/lock restart, the
    /// resident descriptor rewrite, shim BDs and tasks, and the C token SYNCs), with each shim DDR patch of args
    /// 0..=2 advanced by `arena_base[arg]` bytes. `arena_base = [0; 3]` appends exactly the body of [`ArrayDesign::insts`].
    /// The caller guarantees no earlier submit of this design is still in flight.
    pub fn append_run_body(&self, txn: &mut Txn, arena_base: [u64; 3]) {
        assert!(
            !self.variant.is_static(),
            "{} is a static design: its TXN cannot be re-run",
            self.variant
        );
        emit_dynamic_body(txn, &self.geo, self.variant.per_column_sync(), arena_base);
    }

    /// Memtile BD `id` is used by this design's descriptors in some column.
    pub(crate) fn uses_memtile_bd(&self, id: u32) -> bool {
        (0..self.geo.topo.cols as u32).any(|col| {
            memtile_descriptors(self.geo.topo, col)
                .iter()
                .any(|&(b, _)| b == id)
        })
    }

    /// V9 only: host A holds each `mw` tile once (`1/NW` of the default packing, which replicates A for every N-wave)
    /// and the shim A task replays it `NW` times. The NPU receives the identical A stream, so C, the PDI and every
    /// memtile / core descriptor are unchanged; `args[0]`, [`ArrayDesign::pack_in`] and the shim tasks change.
    pub fn with_a_repeat(mut self) -> Self {
        assert_eq!(
            self.variant,
            Variant::V9,
            "A repeat needs the nw-outer V9 wave order"
        );
        assert!(self.geo.nw <= 256, "shim task repeat is at most 256");
        self.geo.a_repeat = true;
        self.args[0].bytes = self.geo.a_bytes();
        self.insts = build_txn_dynamic(&self.geo, false).to_bytes();
        self
    }
}

#[path = "gemm_array_lean.rs"]
mod lean;

/// Byte ranges (offsets in the memtile) the V9 memtile DMA descriptors of shape `(kc, mw, nw)` can touch: each BD's
/// `len` plus every iteration step. Everything outside these ranges is never accessed by a V9 run.
pub fn v9_memtile_footprint(kc: usize, mw: usize, nw: usize) -> Vec<std::ops::Range<u32>> {
    assert!(
        (1..=V9_MAX_KC).contains(&kc)
            && (1..=V9_MAX_MW).contains(&mw)
            && (1..=V9_MAX_NW).contains(&nw)
    );
    let res = Resident {
        mode: ResidentMode::NwOuter,
        kc,
        mw,
        nw,
        b_prefix: false,
    };
    let topo = Topo::with_resident(V8_DEFAULT_EPILOGUE, Control::Fast, res);
    memtile_descriptors(topo, 0)
        .into_iter()
        .map(|(_, bd)| {
            let start = bd.addr as u32 - MEM_BASE;
            start..start + (bd.iteration.step * (bd.iteration.wrap - 1) + bd.len_words) * 4
        })
        .collect()
}

/// Memtile BD ids of one bank (`odd_bank`: ids 24..48 for odd channels, else 0..24 for even channels) that no V9
/// descriptor uses at the maximal topology (`mw = V9_MAX_MW`, `nw = V9_MAX_NW`, `kc = V9_MAX_KC`), ascending.
pub fn v9_spare_memtile_bds(odd_bank: bool) -> Vec<u32> {
    let res = Resident {
        mode: ResidentMode::NwOuter,
        kc: V9_MAX_KC,
        mw: V9_MAX_MW,
        nw: V9_MAX_NW,
        b_prefix: false,
    };
    let topo = Topo::with_resident(V8_DEFAULT_EPILOGUE, Control::Fast, res);
    let used: Vec<u32> = memtile_descriptors(topo, 0)
        .into_iter()
        .map(|(id, _)| id)
        .collect();
    (0..48)
        .filter(|id| (*id >= 24) == odd_bank && !used.contains(id))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{HashMap, HashSet};

    fn lcg_i8(n: usize, seed: u64) -> Vec<i8> {
        let mut s = seed;
        (0..n)
            .map(|i| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                match i % 97 {
                    0 => -128,
                    1 => 127,
                    _ => (s >> 56) as i8,
                }
            })
            .collect()
    }

    fn reference(a: &[i8], b: &[i8], m: usize, n: usize, k: usize) -> Vec<i32> {
        let mut c = vec![0i32; m * n];
        for i in 0..m {
            for kk in 0..k {
                let av = a[i * k + kk] as i32;
                for j in 0..n {
                    c[i * n + j] = c[i * n + j].wrapping_add(av.wrapping_mul(b[kk * n + j] as i32));
                }
            }
        }
        c
    }

    /// Emulates one core on the packed streams using the documented core layouts only
    /// (contract: A chunk block (mb,kb) at (mb*8+kb)*64, B (kb,nb) at (kb*8+nb)*64, C (mb,nb) at (mb*8+nb)*256).
    fn emulate_core(a_chunks: &[u8], b_chunks: &[u8], kc: usize, out: &mut [u8]) {
        let mut acc = vec![0i32; TM * TN];
        for ch in 0..kc {
            let a = &a_chunks[ch * A_BYTES..(ch + 1) * A_BYTES];
            let b = &b_chunks[ch * B_BYTES..(ch + 1) * B_BYTES];
            for row in 0..TM {
                for kk in 0..CHUNK_K {
                    let av = a[((row / 8) * 8 + kk / 8) * 64 + (row % 8) * 8 + kk % 8] as i8 as i32;
                    if av == 0 {
                        continue;
                    }
                    for col in 0..TN {
                        let bv =
                            b[((kk / 8) * 8 + col / 8) * 64 + (kk % 8) * 8 + col % 8] as i8 as i32;
                        acc[row * TN + col] = acc[row * TN + col].wrapping_add(av.wrapping_mul(bv));
                    }
                }
            }
        }
        for row in 0..TM {
            for col in 0..TN {
                let off = (((row / 8) * 8 + col / 8) * 64 + (row % 8) * 8 + col % 8) * 4;
                out[off..off + 4].copy_from_slice(&acc[row * TN + col].to_le_bytes());
            }
        }
    }

    /// Run every core tile whose output intersects the real matrix on the packed args; padding-only
    /// tiles must be fed all-zero operands (checked) and are left zero.
    fn emulate_array(d: &ArrayDesign, a_pack: &[u8], b_pack: &[u8]) -> Vec<u8> {
        let g = &d.geo;
        let (rows, cols) = (g.topo.rows, g.topo.cols);
        assert_eq!(a_pack.len(), d.args[0].bytes);
        assert_eq!(b_pack.len(), d.args[1].bytes);
        let mut out = vec![0u8; d.args[2].bytes];
        for col in 0..cols {
            for w in 0..g.waves {
                for r in 0..rows {
                    let (mw, nw) = (w / g.nw, w % g.nw);
                    let (m0, n0) = ((mw * rows + r) * TM, (nw * cols + col) * TN);
                    // Core (col, r)'s A chunks: row-segmented, or chunk/row interleaved for the one-column variant.
                    let a: Vec<u8> = (0..g.kc)
                        .flat_map(|ch| {
                            let at = if g.topo.kind == Kind::Column {
                                ((w * g.kc + ch) * rows + r) * A_BYTES
                            } else {
                                r * g.a_segment_bytes() + (w * g.kc + ch) * A_BYTES
                            };
                            a_pack[at..at + A_BYTES].iter().copied()
                        })
                        .collect();
                    let b0 = col * g.b_segment_bytes() + w * g.kc * B_BYTES;
                    let b = &b_pack[b0..b0 + g.kc * B_BYTES];
                    if m0 >= g.m {
                        assert!(a.iter().all(|&x| x == 0), "padded A rows not zero");
                        continue;
                    }
                    if n0 >= g.n {
                        assert!(b.iter().all(|&x| x == 0), "padded B cols not zero");
                        continue;
                    }
                    let o = ((col * g.waves + w) * rows + r) * C_BYTES;
                    emulate_core(&a, b, g.kc, &mut out[o..o + C_BYTES]);
                }
            }
        }
        out
    }

    fn check_variant(m: usize, n: usize, k: usize, seed: u64, v: Variant) {
        let d = design_variant(m, n, k, v);
        let (a, b) = (lcg_i8(m * k, seed), lcg_i8(k * n, seed ^ 0x9e37));
        let [pa, pb] = d.pack_in(&a, &b);
        let c_stream = emulate_array(&d, &pa, &pb);
        assert_eq!(
            d.unpack_out(&c_stream),
            reference(&a, &b, m, n, k),
            "{v} {m}x{n}x{k}"
        );
    }

    fn check_exact(m: usize, n: usize, k: usize, seed: u64) {
        check_variant(m, n, k, seed, Variant::V6)
    }

    #[test]
    fn exact_with_padding_single_wave() {
        check_exact(129, 65, 64, 1);
    }

    #[test]
    fn exact_multi_wave_odd_chunk_count() {
        // 2x2 waves (mw-major order, nw replication of A, mw replication of B), kc=3 (odd).
        check_exact(513, 513, 192, 2);
    }

    #[test]
    fn exact_wide_n_only_waves() {
        check_exact(100, 1030, 64, 3);
    }

    #[test]
    fn packed_stream_indexing_across_waves_and_carry() {
        let (m, n, k) = (1024, 1024, 192);
        let d = design(m, n, k);
        let (a, b) = (lcg_i8(m * k, 4), lcg_i8(k * n, 5));
        let [pa, pb] = d.pack_in(&a, &b);
        let g = d.geo;
        assert_eq!((g.waves, g.kc), (4, 3));
        assert_eq!((pa.len(), pb.len()), (4 * 4 * 3 * 8192, 8 * 4 * 3 * 4096));
        // Spot elements by definition: A[row, kcol] for wave (mw,nw), core row r.
        for &(mw, nw, r, ch, mb, kb, i, kk) in &[
            (0, 0, 0, 0, 0, 0, 0, 0),
            (1, 1, 3, 2, 15, 7, 7, 7),
            (1, 0, 2, 1, 9, 3, 5, 2),
            (0, 1, 1, 2, 4, 6, 3, 1),
        ] {
            let w = mw * g.nw + nw;
            let off = r * g.a_segment_bytes()
                + (w * g.kc + ch) * A_BYTES
                + (mb * 8 + kb) * 64
                + i * 8
                + kk;
            let row = (mw * 4 + r) * 128 + mb * 8 + i;
            assert_eq!(pa[off] as i8, a[row * k + ch * 64 + kb * 8 + kk]);
        }
        for &(mw, nw, c, ch, kb, nb, kk, j) in &[
            (0, 0, 0, 0, 0, 0, 0, 0),
            (1, 1, 7, 2, 7, 7, 7, 7),
            (1, 0, 5, 1, 3, 2, 4, 6),
            (0, 1, 3, 2, 6, 5, 1, 3),
        ] {
            let w = mw * g.nw + nw;
            let off = c * g.b_segment_bytes()
                + (w * g.kc + ch) * B_BYTES
                + (kb * 8 + nb) * 64
                + kk * 8
                + j;
            let col = (nw * 8 + c) * 64 + nb * 8 + j;
            assert_eq!(pb[off] as i8, b[(ch * 64 + kb * 8 + kk) * n + col]);
        }
        // Replication: A identical across nw, B identical across mw.
        let a_wave = |r: usize, w: usize| {
            &pa[r * g.a_segment_bytes() + w * g.kc * A_BYTES..][..g.kc * A_BYTES]
        };
        let b_wave = |c: usize, w: usize| {
            &pb[c * g.b_segment_bytes() + w * g.kc * B_BYTES..][..g.kc * B_BYTES]
        };
        assert_eq!(a_wave(2, 0), a_wave(2, 1));
        assert_ne!(a_wave(2, 0), a_wave(2, 2));
        assert_eq!(b_wave(6, 0), b_wave(6, 2));
        assert_ne!(b_wave(6, 0), b_wave(6, 1));
    }

    #[test]
    fn unpack_uses_wave_row_column_segments_and_drops_padding() {
        let d = design(600, 520, 64);
        let g = d.geo;
        let mut out = vec![0u8; d.args[2].bytes];
        // Tag each core tile's first i32 with a unique value derived from (col,wave,row).
        for col in 0..COLS {
            for w in 0..g.waves {
                for r in 0..ROWS {
                    let o = ((col * g.waves + w) * ROWS + r) * C_BYTES;
                    out[o..o + 4]
                        .copy_from_slice(&(1 + (col * 1000 + w * 10 + r) as i32).to_le_bytes());
                }
            }
        }
        let c = d.unpack_out(&out);
        assert_eq!(c.len(), 600 * 520);
        for (w, mw, nw) in [(0, 0, 0), (1, 0, 1), (2, 1, 0), (3, 1, 1)] {
            for col in 0..COLS {
                for r in 0..ROWS {
                    let (row, cc) = ((mw * 4 + r) * 128, (nw * 8 + col) * 64);
                    let want = 1 + (col * 1000 + w * 10 + r) as i32;
                    if row < 600 && cc < 520 {
                        assert_eq!(c[row * 520 + cc], want, "w{w} c{col} r{r}");
                    }
                }
            }
        }
    }

    #[test]
    fn geometry_args_and_estimate() {
        let d = design(512, 512, 64);
        assert_eq!(d.waves(), 1);
        assert_eq!(
            d.args.iter().map(|a| a.bytes).collect::<Vec<_>>(),
            [4 * 8192, 8 * 4096, 8 * 4 * 32768]
        );
        assert_eq!(
            d.args.iter().map(|a| a.kind).collect::<Vec<_>>(),
            [ArgKind::In, ArgKind::In, ArgKind::Out]
        );
        assert_eq!(d.useful_ops(), 2 * 512 * 512 * 64);
        assert_eq!(
            (d.host_bytes_read(), d.host_bytes_written()),
            ((4 * 8192 + 8 * 4096) as u64, 8 * 4 * 32768)
        );
        let d = design(129, 65, 64 * 5);
        let e = d.estimate(1_000_000_000);
        assert_eq!(
            (e.ideal_mac_cycles, e.input_dma_cycles, e.output_dma_cycles),
            (1024 * 5, 2048 * 5, 32768)
        );
        assert_eq!(e.estimated_cycles, 2048 * 5 + 32768);
        assert_eq!(e.peak_ops_per_second, 32_768_000_000_000);
        assert!((e.estimated_seconds - e.estimated_cycles as f64 * 1e-9).abs() < 1e-15);
        assert!((e.useful_tops - d.useful_ops() as f64 / e.estimated_seconds / 1e12).abs() < 1e-9);
    }

    #[test]
    fn all_expert_shapes_fit_limits() {
        let mut shapes = vec![(512, 512, 64), (512, 512, 256)];
        for m in [512, 1024, 2048, 4096] {
            shapes.push((m, 1280, 2560));
            shapes.push((m, 2560, 640));
        }
        for (m, n, k) in shapes {
            let d = design(m, n, k);
            assert_eq!(d.args.len(), 3, "{m}x{n}x{k}");
            assert!(d.waves() <= 256);
            assert_eq!(d.args[0].bytes, 4 * d.geo.waves * (k / 64) * 8192);
        }
        let program = gemm_core::program(MAX_KC, MAX_WAVES).finish();
        assert!(program.len() <= 16 * 1024, "{}", program.len());
    }

    #[test]
    #[should_panic]
    fn rejects_k_not_multiple_of_64() {
        design(512, 512, 96);
    }
    #[test]
    #[should_panic]
    fn rejects_k_beyond_max() {
        design(512, 512, (MAX_KC + 1) * CHUNK_K);
    }
    #[test]
    #[should_panic]
    fn rejects_more_than_256_waves() {
        design(512 * 17, 512 * 16, 64);
    }

    // ---------------------------------------------------------------- routing

    fn topos() -> Vec<Topo> {
        [
            Variant::V1,
            Variant::V2,
            Variant::V3,
            Variant::V4,
            Variant::V5,
        ]
        .iter()
        .map(|v| v.topo())
        .collect()
    }

    #[test]
    fn no_master_driven_twice_and_ports_valid() {
        for topo in topos() {
            let mut driven: HashSet<(u32, u32, String)> = HashSet::new();
            for c in circuits(topo) {
                // writes() resolves port indices and panics on an absent/out-of-range port.
                let w = c.writes();
                assert!(w[0].0 != w[1].0);
                let key = (c.tile.col, c.tile.row, format!("{:?}", c.master));
                assert!(
                    driven.insert(key.clone()),
                    "master driven twice: {key:?} in {topo:?}"
                );
            }
        }
    }

    /// Follows every circuit across tile boundaries; returns the DMA endpoints reached from `start`.
    fn reach(all: &[Circuit], start: (Location, Port)) -> HashSet<(u32, u32, String)> {
        let mut by_slave: HashMap<(u32, u32, String), Vec<Circuit>> = HashMap::new();
        for &c in all {
            by_slave
                .entry((c.tile.col, c.tile.row, format!("{:?}", c.slave)))
                .or_default()
                .push(c);
        }
        let mut seen = HashSet::new();
        let mut ends = HashSet::new();
        let mut stack = vec![start];
        while let Some((tile, slave)) = stack.pop() {
            if !seen.insert((tile.col, tile.row, format!("{slave:?}"))) {
                continue;
            }
            for c in by_slave
                .get(&(tile.col, tile.row, format!("{slave:?}")))
                .into_iter()
                .flatten()
            {
                let next = match c.master {
                    Port::North(n) => Some((Location::new(tile.col, tile.row + 1), Port::South(n))),
                    Port::South(n) if tile.row > 0 => {
                        Some((Location::new(tile.col, tile.row - 1), Port::North(n)))
                    }
                    Port::East(n) => Some((Location::new(tile.col + 1, tile.row), Port::West(n))),
                    Port::West(n) => Some((Location::new(tile.col - 1, tile.row), Port::East(n))),
                    _ => None,
                };
                match next {
                    Some(n) => stack.push(n),
                    None => {
                        ends.insert((tile.col, tile.row, format!("{:?}", c.master)));
                    }
                }
            }
        }
        ends
    }

    #[test]
    fn streams_reach_exactly_the_documented_endpoints() {
        let dma = |n: u8| format!("{:?}", Port::Dma(n));
        for topo in topos() {
            let all = circuits(topo);
            let rows = topo.rows as u32;
            for col in 0..topo.cols as u32 {
                let shim = shim_loc(col);
                let port = |direction, channel| {
                    ShimDma {
                        tile: shim,
                        direction,
                        channel,
                    }
                    .port()
                };
                let shim_c = HashSet::from([(col, 0, format!("{:?}", port(S2mm, 0)))]);
                match topo.kind {
                    Kind::Pass => {
                        // Pass-through: B and A land directly on core DMA1 / DMA0; C returns to shim S2MM0.
                        assert_eq!(
                            reach(&all, (shim, port(Mm2s, 0))),
                            HashSet::from([(0, 2, dma(1))])
                        );
                        assert_eq!(
                            reach(&all, (shim, port(Mm2s, 1))),
                            HashSet::from([(0, 2, dma(0))])
                        );
                        assert_eq!(reach(&all, (core_loc(0, 0), Port::Dma(0))), shim_c);
                        // The memtile only has switch pass-through circuits and each keeps its lane (hardware cannot remap
                        // South<->North lanes through a memtile); the core receives B on South0 and A on South1.
                        let mem_circuits: Vec<_> =
                            all.iter().filter(|c| c.tile == mem_loc(0)).collect();
                        for c in mem_circuits {
                            match (&c.slave, &c.master) {
                                (Port::South(a), Port::North(b))
                                | (Port::North(a), Port::South(b)) => assert_eq!(a, b, "{c:?}"),
                                other => panic!("non pass-through memtile circuit {other:?}"),
                            }
                        }
                    }
                    Kind::Column => {
                        // A (interleaved) shim Mm2s0 -> memtile S2MM4; B shim Mm2s1 -> S2MM5.
                        assert_eq!(
                            reach(&all, (shim, port(Mm2s, 0))),
                            HashSet::from([(0, 1, dma(OC_A_PROD_CH as u8))])
                        );
                        assert_eq!(
                            reach(&all, (shim, port(Mm2s, 1))),
                            HashSet::from([(0, 1, dma(OC_B_PROD_CH as u8))])
                        );
                        // Memtile MM2S r carries only the A of core row r; MM2S4 is B broadcast to all 4 cores.
                        for r in 0..rows {
                            assert_eq!(
                                reach(&all, (mem_loc(0), Port::Dma(r as u8))),
                                HashSet::from([(0, r + 2, dma(0))]),
                                "A row {r}"
                            );
                            assert_eq!(
                                reach(&all, (core_loc(0, r as usize), Port::Dma(0))),
                                HashSet::from([(0, 1, dma(r as u8))]),
                                "C row {r}"
                            );
                        }
                        let cores: HashSet<_> = (0..rows).map(|r| (0, r + 2, dma(1))).collect();
                        assert_eq!(
                            reach(&all, (mem_loc(0), Port::Dma(OC_B_CONS_CH as u8))),
                            cores
                        );
                        assert_eq!(
                            reach(&all, (mem_loc(0), Port::Dma(OC_C_JOIN_CH as u8))),
                            shim_c
                        );
                    }
                    Kind::Pair { .. } | Kind::G80 { .. } | Kind::Ief15 { .. } => {
                        unreachable!("V8 has its own tests")
                    }
                    Kind::Staged => {
                        // B: shim Mm2s0 lands in memtile S2MM4; memtile MM2S1 feeds S2MM1 of the cores of the column.
                        assert_eq!(
                            reach(&all, (shim, port(Mm2s, 0))),
                            HashSet::from([(col, 1, dma(B_S2MM_CH as u8))]),
                            "B in col {col}"
                        );
                        let cores: HashSet<_> = (0..rows).map(|r| (col, r + 2, dma(1))).collect();
                        assert_eq!(
                            reach(&all, (mem_loc(col), Port::Dma(B_MM2S_CH as u8))),
                            cores,
                            "B out col {col}"
                        );
                        // A: column r (<rows) shim Mm2s1 lands in memtile S2MM5; memtile MM2S2 feeds S2MM0 of every core of row r.
                        if topo.has_a(col) {
                            assert_eq!(
                                reach(&all, (shim, port(Mm2s, 1))),
                                HashSet::from([(col, 1, dma(A_S2MM_CH as u8))]),
                                "A in col {col}"
                            );
                            let row: HashSet<_> = (0..topo.cols as u32)
                                .map(|c| (c, col + 2, dma(0)))
                                .collect();
                            assert_eq!(
                                reach(&all, (mem_loc(col), Port::Dma(A_MM2S_CH as u8))),
                                row,
                                "A out col {col}"
                            );
                        }
                        // C: core row r reaches only memtile S2MM r; memtile MM2S0 reaches the shim S2MM0 port.
                        for r in 0..rows {
                            let got = reach(&all, (core_loc(col, r as usize), Port::Dma(0)));
                            assert_eq!(
                                got,
                                HashSet::from([(col, 1, dma(r as u8))]),
                                "C col {col} row {r}"
                            );
                        }
                        assert_eq!(reach(&all, (mem_loc(col), Port::Dma(0))), shim_c);
                    }
                }
            }
        }
    }

    // ------------------------------------------------------ addresses / locks

    fn decode_tile_bd(words: &[u32; 6]) -> (u32, u32) {
        ((words[0] >> 14) * 4, (words[0] & 0x3fff) * 4)
    }

    fn disjoint_or_equal(ranges: &[(u32, u32)]) {
        let uniq: HashSet<_> = ranges.iter().copied().collect();
        let v: Vec<_> = uniq.into_iter().collect();
        for (i, a) in v.iter().enumerate() {
            for b in &v[i + 1..] {
                assert!(
                    a.0 + a.1 <= b.0 || b.0 + b.1 <= a.0,
                    "overlapping buffers {a:?} {b:?}"
                );
            }
        }
    }

    #[test]
    fn core_memory_footprint() {
        let ranges: Vec<_> = gemm_core::tile_bds().iter().map(decode_tile_bd).collect();
        assert!(ranges.iter().all(|&(a, l)| a + l <= 64 * 1024));
        disjoint_or_equal(&ranges);
        assert_eq!(
            ranges
                .iter()
                .map(|r| r.1)
                .filter(|&l| l == A_BYTES as u32)
                .count(),
            2
        );
    }

    #[test]
    fn memtile_footprint_ids_and_locks() {
        for topo in topos() {
            for col in 0..topo.cols as u32 {
                let bds = memtile_descriptors(topo, col);
                if topo.kind == Kind::Pass {
                    assert!(bds.is_empty() && memtile_locks(topo, col).is_empty());
                    continue;
                }
                let ids: Vec<u32> = bds.iter().map(|b| b.0).collect();
                assert_eq!(
                    ids.iter().collect::<HashSet<_>>().len(),
                    ids.len(),
                    "duplicate BD id"
                );
                assert!(ids.iter().all(|&i| i < 48));
                let expected = match topo.kind {
                    Kind::Column => 36,
                    _ => 4 + 4 * topo.rows + if topo.has_a(col) { 4 } else { 0 },
                };
                assert_eq!(ids.len(), expected, "{topo:?} col {col}");
                let mut ranges = Vec::new();
                for (_, bd) in &bds {
                    let start = bd.addr as u32 - MEM_BASE;
                    ranges.push((start, bd.len_words * 4));
                    assert!(start + bd.len_words * 4 <= 512 * 1024);
                    assert!(
                        bd.next.is_some_and(|n| ids.contains(&n)),
                        "chain leaves descriptor set"
                    );
                }
                disjoint_or_equal(&ranges);
                let top = match topo.kind {
                    Kind::Column => oc_c_offset(1, ROWS - 1),
                    _ => c_offset(1, topo.rows - 1),
                } + C_BYTES as u32;
                assert_eq!(ranges.iter().map(|r| r.0 + r.1).max().unwrap(), top);
                // Every lock is consumed by exactly the matching releases: net delta over the BDs is zero.
                let mut net: HashMap<u32, i32> = HashMap::new();
                for (_, bd) in &bds {
                    for (id, v) in [bd.locks.acq, bd.locks.rel].into_iter().flatten() {
                        *net.entry(id).or_default() += v;
                    }
                }
                assert!(net.values().all(|&v| v == 0), "{net:?}");
                let init: HashMap<u32, u32> = memtile_locks(topo, col).into_iter().collect();
                assert!(
                    net.keys().all(|id| init.contains_key(&(id - MEM_LOCK)))
                        && init.keys().all(|&i| i < 64)
                );
                assert!(init.iter().all(|(&id, &v)| v
                    == if net.contains_key(&(id + MEM_LOCK)) && is_empty_lock(topo, id) {
                        1
                    } else {
                        0
                    }));
                // Rings are cycles of the documented length and visit slot 0 rows then slot 1 rows.
                let next: HashMap<u32, u32> =
                    bds.iter().map(|(i, b)| (*i, b.next.unwrap())).collect();
                let addr: HashMap<u32, u32> = bds
                    .iter()
                    .map(|(i, b)| (*i, b.addr as u32 - MEM_BASE))
                    .collect();
                let cycle = |first: u32, len: usize| -> Vec<u32> {
                    let mut at = first;
                    let seq: Vec<u32> = (0..len)
                        .map(|_| {
                            let cur = at;
                            at = next[&at];
                            cur
                        })
                        .collect();
                    assert_eq!(at, first, "chain does not cycle after {len}");
                    seq
                };
                match topo.kind {
                    Kind::Column => {
                        let offsets: [(u32, fn(usize, usize) -> u32); 2] =
                            [(OC_A_PROD_BD0, oc_a_offset), (OC_C_JOIN_BD0, oc_c_offset)];
                        for (first, offset) in offsets {
                            let seq = cycle(first, 2 * ROWS);
                            assert_eq!(seq, (first..first + 8).collect::<Vec<_>>());
                            for (i, id) in seq.iter().enumerate() {
                                assert_eq!(addr[id], offset(i / ROWS, i % ROWS));
                            }
                        }
                    }
                    _ => {
                        let seq = cycle(C_MM2S_BD0, 2 * topo.rows);
                        assert_eq!(seq, (6..6 + 2 * topo.rows as u32).collect::<Vec<_>>());
                        for (i, id) in seq.iter().enumerate() {
                            assert_eq!(addr[id], c_offset(i / topo.rows, i % topo.rows));
                        }
                    }
                }
            }
        }
    }
    fn is_empty_lock(topo: Topo, id: u32) -> bool {
        if topo.kind == Kind::Column {
            return (id < 16 && id % 2 == 0)
                || OC_B_EMPTY.contains(&id)
                || (id >= 20 && id % 2 == 0);
        }
        let ((ae, _), (be, _)) = ab_locks();
        ae.contains(&id) || be.contains(&id) || (id >= C_LOCK0 && (id - C_LOCK0) % 2 == 0)
    }

    #[test]
    fn task_banks_match_channels_and_reset_set_is_unique() {
        for topo in topos() {
            for col in 0..topo.cols as u32 {
                for (tile, task) in ring_tasks(topo, col) {
                    // Task::write asserts the memtile even-channel/BD<24 rule.
                    let _ = task.write(tile);
                }
                let chans = used_channels(topo, col);
                assert!(
                    chans.iter().all(|(l, ..)| l.kind() != TileType::Shim),
                    "shim channels have no reset"
                );
                let uniq: HashSet<_> = chans
                    .iter()
                    .map(|(l, d, c)| (l.row, *d == S2mm, *c))
                    .collect();
                assert_eq!(uniq.len(), chans.len());
                // 3 channels per core; memtile: one-column 6 S2MM + 6 MM2S, staged (rows C + B) S2MM + (C chain + B) MM2S
                // (+1 S2MM, +1 MM2S carrying A in columns < rows), pass-through none.
                let mem = match topo.kind {
                    Kind::Pass => 0,
                    Kind::Column => 12,
                    Kind::Staged => topo.rows + 1 + 2 + if topo.has_a(col) { 2 } else { 0 },
                    Kind::Pair { .. } | Kind::G80 { .. } | Kind::Ief15 { .. } => {
                        unreachable!("V8 has its own tests")
                    }
                };
                assert_eq!(chans.len(), topo.rows * 3 + mem, "{topo:?} col {col}");
            }
        }
    }

    // ------------------------------------------------------------ TXN / CDO

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct Op {
        kind: u32,
        addr: u32,
        value: u32,
        mask: u32,
        plus: u64,
    }

    fn txn_ops(bytes: &[u8]) -> Vec<Op> {
        let w: Vec<u32> = bytes
            .chunks(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let (count, mut i) = (w[2], 4);
        let mut ops = Vec::new();
        for _ in 0..count {
            let (op, len) = match w[i] {
                0 => (
                    Op {
                        kind: 0,
                        addr: w[i + 2],
                        value: w[i + 4],
                        mask: u32::MAX,
                        plus: 0,
                    },
                    6,
                ),
                1 => (
                    Op {
                        kind: 1,
                        addr: w[i + 2],
                        value: 0,
                        mask: 0,
                        plus: 0,
                    },
                    (w[i + 3] / 4) as usize,
                ),
                3 => (
                    Op {
                        kind: 3,
                        addr: w[i + 2],
                        value: w[i + 4],
                        mask: w[i + 5],
                        plus: 0,
                    },
                    7,
                ),
                0x80 => (
                    Op {
                        kind: 0x80,
                        addr: w[i + 2],
                        value: w[i + 3],
                        mask: 0,
                        plus: 0,
                    },
                    4,
                ),
                0x81 => (
                    Op {
                        kind: 0x81,
                        addr: w[i + 6],
                        value: w[i + 8],
                        mask: 0,
                        plus: w[i + 10] as u64 | (w[i + 11] as u64) << 32,
                    },
                    12,
                ),
                other => panic!("unknown txn op {other:#x}"),
            };
            ops.push(op);
            i += len;
        }
        assert_eq!(i, w.len());
        ops
    }

    #[test]
    fn ddr_patches_partition_each_argument_by_column() {
        let d = design(1024, 1024, 192);
        let ops = txn_ops(&d.insts);
        let patches: Vec<_> = ops.iter().filter(|o| o.kind == 0x81).collect();
        assert_eq!(patches.len(), 8 + 8 + 4);
        let mut per_arg: HashMap<u32, Vec<(u32, u64)>> = HashMap::new();
        for p in &patches {
            let col = (p.addr >> regs::COL_SHIFT) & 7;
            assert_eq!(
                (p.addr >> regs::ROW_SHIFT) & 0x1f,
                0,
                "patch must target shim BD"
            );
            per_arg.entry(p.value).or_default().push((col, p.plus));
        }
        let seg = |arg: usize| d.args[arg].bytes / [4, 8, 8][arg];
        assert!(per_arg.keys().all(|&a| a < 3));
        for (arg, cols) in [(0u32, 4u32), (1, 8), (2, 8)] {
            let mut v = per_arg[&arg].clone();
            v.sort();
            assert_eq!(
                v,
                (0..cols)
                    .map(|c| (c, (c as usize * seg(arg as usize)) as u64))
                    .collect::<Vec<_>>()
            );
        }
        assert_eq!(ops.last().unwrap().kind, 0x80);
    }

    #[test]
    fn submit_resets_then_starts_everything_in_txn() {
        let d = design(512, 512, 64);
        let ops = txn_ops(&d.insts);
        let pos = |f: &dyn Fn(&Op) -> bool| -> Vec<usize> {
            ops.iter()
                .enumerate()
                .filter(|(_, o)| f(o))
                .map(|(i, _)| i)
                .collect()
        };
        for col in 0..8u32 {
            for r in 0..4 {
                let t = core_loc(col, r);
                let ctrl = t.address(regs::core::CORE_CONTROL);
                let halt = pos(&|o| o.kind == 3 && o.addr == ctrl && o.value == 2 && o.mask == 3);
                let release =
                    pos(&|o| o.kind == 3 && o.addr == ctrl && o.value == 0 && o.mask == 2);
                let enable = pos(&|o| o.kind == 3 && o.addr == ctrl && o.value == 1 && o.mask == 1);
                let pc = pos(&|o| {
                    o.kind == 0 && o.addr == t.address(regs::core::CORE_PC) && o.value == 0
                });
                let lock0 = pos(&|o| o.kind == 0 && o.addr == dma::lock_write(t, 0, 0).0);
                let s2mm0 = regs::core::DMA_S2MM_0_CTRL;
                let assert_reset =
                    pos(&|o| o.kind == 3 && o.addr == t.address(s2mm0) && o.value == 2);
                let deassert = pos(&|o| {
                    o.kind == 3 && o.addr == t.address(s2mm0) && o.value == 0 && o.mask == 2
                });
                let queue = pos(&|o| o.kind == 0 && o.addr == t.address(s2mm0 + 4));
                let dma_enable = pos(&|o| {
                    o.kind == 3 && o.addr == t.address(s2mm0) && o.value == 1 && o.mask == 1
                });
                for v in [
                    &halt,
                    &release,
                    &enable,
                    &pc,
                    &lock0,
                    &assert_reset,
                    &deassert,
                    &queue,
                    &dma_enable,
                ] {
                    assert_eq!(v.len(), 1, "core {col},{r}");
                }
                assert!(
                    halt[0] < assert_reset[0]
                        && assert_reset[0] < deassert[0]
                        && deassert[0] < lock0[0]
                );
                assert!(lock0[0] < release[0] && release[0] < pc[0] && pc[0] < queue[0]);
                assert!(queue[0] < dma_enable[0] && dma_enable[0] < enable[0]);
            }
        }
        // No op may assert reset (bit1) on a shim DMA channel CTRL register: aie-rt rejects it.
        for col in 0..8 {
            let shim = shim_loc(col);
            let ctrls = [
                regs::shim::DMA_S2MM_0_CTRL,
                regs::shim::DMA_MM2S_0_CTRL,
                regs::shim::DMA_MM2S_0_CTRL + 8,
            ];
            for off in ctrls {
                assert!(
                    pos(&|o| o.kind == 3 && o.addr == shim.address(off) && o.mask & 2 != 0)
                        .is_empty(),
                    "shim channel reset"
                );
            }
        }
        let shim_queue =
            pos(&|o| o.kind == 0 && o.addr == shim_loc(0).address(regs::shim::DMA_S2MM_0_CTRL + 4));
        let sync = pos(&|o| o.kind == 0x80);
        let last_core_enable = pos(&|o| {
            o.kind == 3
                && o.addr & 0x000f_ffff == regs::core::CORE_CONTROL
                && o.value == 1
                && o.mask == 1
        });
        assert_eq!(
            (last_core_enable.len(), sync.len(), shim_queue.len()),
            (32, 1, 1)
        );
        assert!(
            shim_queue[0] < last_core_enable[0],
            "core enable must follow all shim tasks"
        );
        assert!(last_core_enable[31] + 1 == sync[0]);
    }

    // ------------------------------------------------------------ variants

    #[test]
    fn variant_names_parse_and_shapes() {
        for v in Variant::ALL {
            assert_eq!(v.name().parse::<Variant>(), Ok(v));
            assert_eq!(v.to_string(), v.name());
        }
        assert_eq!("v6b".parse::<Variant>(), Ok(Variant::V6b));
        assert!("V7".parse::<Variant>().is_err());
        assert_eq!(
            Variant::ALL.map(|v| v.shape()),
            [
                (128, 64),
                (128, 64),
                (512, 64),
                (128, 512),
                (512, 512),
                (512, 512),
                (512, 512)
            ]
        );
        assert_eq!(design(512, 512, 64).variant(), Variant::V6);
    }

    #[test]
    fn every_variant_is_cpu_exact_at_its_shape() {
        for v in Variant::ALL {
            let (m, n) = v.shape();
            let ks: &[usize] = if v.topo().rows * v.topo().cols == 32 {
                &[64, 192]
            } else {
                &[64, 192, 256]
            };
            for &k in ks {
                check_variant(m, n, k, 7 + k as u64, v);
            }
        }
    }

    #[test]
    fn reduced_variants_handle_padding_and_multiple_waves() {
        for (v, m, n, k) in [
            (Variant::V1, 129, 65, 64),
            (Variant::V2, 200, 130, 128),
            (Variant::V3, 513, 65, 192),
            (Variant::V3, 100, 130, 64),
            (Variant::V4, 129, 513, 64),
            (Variant::V4, 128, 1100, 64),
        ] {
            check_variant(m, n, k, 11, v);
        }
    }

    #[test]
    fn variant_args_and_output_tile_ranges_follow_the_active_tiles() {
        for v in Variant::ALL {
            let (rows, cols) = (v.topo().rows, v.topo().cols);
            let (m, n) = v.shape();
            let d = design_variant(2 * m, n, 64, v);
            assert_eq!(d.waves(), 2);
            // Exactly the active tiles: A has `rows` segments (one interleaved stream for V3), B and C `cols`.
            assert_eq!(
                d.args.iter().map(|a| a.bytes).collect::<Vec<_>>(),
                [rows * 2 * 8192, cols * 2 * 4096, cols * 2 * rows * 32768],
                "{v}"
            );
            let tiles = d.output_tiles();
            assert_eq!(tiles.len(), rows * cols);
            assert!(tiles
                .iter()
                .all(|&(c, r)| (c as usize) < cols && (2..2 + rows as u32).contains(&r)));
            let ranges = d.output_tile_ranges();
            assert_eq!(ranges.iter().map(|t| t.0).collect::<Vec<_>>(), tiles);
            // Ranges tile the C argument exactly and map to the matching C block of the unpacked matrix.
            let mut sorted: Vec<_> = ranges
                .iter()
                .flat_map(|(_, rs)| rs.iter().cloned())
                .collect();
            sorted.sort_by_key(|r| r.start);
            let mut at = 0;
            for r in &sorted {
                assert_eq!((r.start, r.len()), (at, C_BYTES), "{v}");
                at = r.end;
            }
            assert_eq!(at, d.args[2].bytes);
            let mut packed = vec![0u8; at];
            for (tag, (_, rs)) in ranges.iter().enumerate() {
                assert_eq!(rs.len(), 2);
                for (w, r) in rs.iter().enumerate() {
                    packed[r.start..r.start + 4]
                        .copy_from_slice(&(1 + 10 * tag as i32 + w as i32).to_le_bytes());
                }
            }
            let c = d.unpack_out(&packed);
            for (tag, ((col, row), _)) in ranges.iter().enumerate() {
                for w in 0..2usize {
                    let (m0, n0) = ((w * rows + (*row as usize - 2)) * TM, *col as usize * TN);
                    assert_eq!(
                        c[m0 * n + n0],
                        1 + 10 * tag as i32 + w as i32,
                        "{v} tile ({col},{row}) wave {w}"
                    );
                }
            }
        }
    }

    #[test]
    fn static_variants_start_everything_in_cdo_and_dynamic_variants_do_not() {
        let program = gemm_core::program(1, 1).finish();
        for v in Variant::ALL {
            let topo = v.topo();
            let cmds =
                crate::cdo::parse(&build_cdo(topo, [&program, &program], v.is_static()).to_words())
                    .unwrap();
            let count = |c: crate::cdo::Cmd| cmds.iter().filter(|x| **x == c).count();
            let at = |c: crate::cdo::Cmd| cmds.iter().position(|x| *x == c).unwrap();
            let want = usize::from(v.is_static());
            let ctrl = regs::core::CORE_CONTROL;
            let cores: Vec<Location> = topo.cores().collect();
            for &t in &cores {
                let enable = crate::cdo::Cmd::MaskWrite(t.address(ctrl), 1, 1);
                let pc = crate::cdo::Cmd::Write(t.address(regs::core::CORE_PC), 0);
                let dma_enable =
                    crate::cdo::Cmd::MaskWrite(t.address(regs::core::DMA_S2MM_0_CTRL), 1, 1);
                let queue = crate::cdo::Cmd::Write(
                    t.address(regs::core::DMA_S2MM_0_CTRL + 4),
                    gemm_core::A_BD[0],
                );
                for c in [enable.clone(), pc, dma_enable.clone(), queue] {
                    assert_eq!(count(c), want, "{v} {t:?}");
                }
                if v.is_static() {
                    assert!(at(dma_enable) < at(enable));
                }
            }
            if v.is_static() {
                // Core enables are the very last commands.
                let enables: Vec<_> = cores
                    .iter()
                    .map(|t| crate::cdo::Cmd::MaskWrite(t.address(ctrl), 1, 1))
                    .collect();
                assert_eq!(&cmds[cmds.len() - cores.len()..], &enables[..], "{v}");
            }
            // Memtile queues and locks follow the same rule; shim tasks are never in the CDO.
            for col in 0..topo.cols as u32 {
                for (tile, task) in ring_tasks(topo, col)
                    .into_iter()
                    .filter(|(t, _)| t.kind() == TileType::Memtile)
                {
                    let (addr, value) = task.write(tile);
                    assert_eq!(
                        count(crate::cdo::Cmd::Write(addr, value)),
                        want,
                        "{v} memtile task"
                    );
                }
                for (id, value) in memtile_locks(topo, col) {
                    let (addr, value) = dma::lock_write(mem_loc(col), id, value);
                    assert_eq!(
                        count(crate::cdo::Cmd::Write(addr, value)),
                        want,
                        "{v} memtile lock {id}"
                    );
                }
                let shim = shim_loc(col);
                for off in [
                    regs::shim::DMA_S2MM_0_CTRL + 4,
                    regs::shim::DMA_MM2S_0_CTRL + 4,
                    regs::shim::DMA_MM2S_0_CTRL + 12,
                ] {
                    assert!(!cmds.iter().any(
                        |c| matches!(c, crate::cdo::Cmd::Write(a, _) if *a == shim.address(off))
                    ));
                }
            }
        }
    }

    #[test]
    fn variant_txn_shapes_syncs_and_patches() {
        for v in Variant::ALL {
            let topo = v.topo();
            let (m, n) = v.shape();
            let d = design_variant(m, n, 64, v);
            let ops = txn_ops(&d.insts);
            let syncs: Vec<_> = ops.iter().filter(|o| o.kind == 0x80).collect();
            assert_eq!(ops.last().unwrap().kind, 0x80);
            if v == Variant::V6 {
                // One aggregate SYNC over all columns.
                assert_eq!(syncs.len(), 1);
                assert_eq!(
                    (syncs[0].value >> 16 & 0xff, syncs[0].value >> 8 & 0xff),
                    (COLS as u32, 1)
                );
            } else {
                // One SYNC per active column, ncol = nrow = 1, in column order.
                assert_eq!(syncs.len(), topo.cols, "{v}");
                for (c, s) in syncs.iter().enumerate() {
                    assert_eq!(
                        (
                            s.addr >> 16 & 0xff,
                            s.value >> 16 & 0xff,
                            s.value >> 8 & 0xff
                        ),
                        (c as u32, 1, 1),
                        "{v}"
                    );
                }
            }
            let core_ctrl =
                |o: &&Op| o.kind == 3 && o.addr & 0x000f_ffff == regs::core::CORE_CONTROL;
            let core_ops = ops.iter().filter(core_ctrl).count();
            if v.is_static() {
                // Shim-only: every op targets a row 0 (shim) register.
                for o in ops.iter().filter(|o| o.kind != 0x80) {
                    assert_eq!(
                        (o.addr >> regs::ROW_SHIFT) & 0x1f,
                        0,
                        "{v}: non-shim op {o:?}"
                    );
                }
                assert_eq!(core_ops, 0);
            } else {
                assert_eq!(core_ops, 3 * topo.rows * topo.cols);
            }
            // DDR patches: A once per injection column (one interleaved stream for V3), B and C per column.
            let mut per_arg: HashMap<u32, Vec<(u32, u64)>> = HashMap::new();
            for p in ops.iter().filter(|o| o.kind == 0x81) {
                per_arg
                    .entry(p.value)
                    .or_default()
                    .push(((p.addr >> regs::COL_SHIFT) & 7, p.plus));
            }
            let a_cols = if topo.kind == Kind::Column {
                1
            } else {
                topo.rows
            };
            for (arg, count, seg) in [
                (0u32, a_cols, d.args[0].bytes / a_cols),
                (1, topo.cols, d.args[1].bytes / topo.cols),
                (2, topo.cols, d.args[2].bytes / topo.cols),
            ] {
                let mut got = per_arg[&arg].clone();
                got.sort();
                assert_eq!(
                    got,
                    (0..count)
                        .map(|c| (c as u32, (c * seg) as u64))
                        .collect::<Vec<_>>(),
                    "{v} arg{arg}"
                );
            }
        }
    }

    #[test]
    fn one_column_variant_interleaves_a_by_chunk_then_row() {
        let (m, n, k) = (1024, 64, 192);
        let d = design_variant(m, n, k, Variant::V3);
        let a = lcg_i8(m * k, 21);
        let [pa, _] = d.pack_in(&a, &lcg_i8(k * n, 22));
        let g = d.geo;
        assert_eq!((g.waves, g.kc), (2, 3));
        // Stream tile index (wave w, chunk ch, row r) = (w*kc + ch)*4 + r; no per-row segments.
        for &(w, ch, r, mb, kb, i, kk) in &[
            (0, 0, 0, 0, 0, 0, 0),
            (1, 2, 3, 15, 7, 7, 7),
            (1, 0, 2, 9, 3, 5, 2),
            (0, 1, 1, 4, 6, 3, 1),
        ] {
            let off = ((w * g.kc + ch) * ROWS + r) * A_BYTES + (mb * 8 + kb) * 64 + i * 8 + kk;
            let row = (w * ROWS + r) * 128 + mb * 8 + i;
            assert_eq!(pa[off] as i8, a[row * k + ch * 64 + kb * 8 + kk]);
        }
    }

    // ------------------------------------------------------------------ V8

    const V8_EPIS: [Epilogue; 2] = [Epilogue::I32, Epilogue::Int8 { shift: 12 }];

    /// Emulates V8 on the packed streams from the documented layouts only: each core's A chunk is rebuilt from
    /// the E half injected by column `4q + par` and the O half injected by column `4q + 2 + par`, B comes from
    /// its own column segment, C is written in the V8 C layout (int32 or int8 blocks).
    fn emulate_v8(d: &ArrayDesign, a_pack: &[u8], b_pack: &[u8]) -> Vec<u8> {
        let g = &d.geo;
        let epi = d.epilogue().unwrap();
        let cb = g.topo.c_tile_bytes();
        let mut out = vec![0u8; d.args[2].bytes];
        for col in 0..COLS {
            for w in 0..g.waves {
                for rho in 0..ROWS {
                    let (q, par) = (rho / 2, col % 2);
                    let half = |h: usize, ch: usize| -> &[u8] {
                        let inj = 4 * q + 2 * h + par;
                        let at = inj * g.a_segment_bytes() + (w * g.kc + ch) * A_HALF_BYTES;
                        &a_pack[at..at + A_HALF_BYTES]
                    };
                    let mut a = vec![0u8; g.kc * A_BYTES];
                    for ch in 0..g.kc {
                        for mb in 0..MB {
                            for kb in 0..KB {
                                let src = (mb / 2 * KB + kb) * 64;
                                let dst = ch * A_BYTES + (mb * KB + kb) * 64;
                                a[dst..dst + 64].copy_from_slice(&half(mb % 2, ch)[src..src + 64]);
                            }
                        }
                    }
                    // B block j = col/2, 64-column half rho%2: injected by column 2j + rho%2.
                    let b0 = (2 * (col / 2) + rho % 2) * g.b_segment_bytes() + w * g.kc * B_BYTES;
                    let mut c32 = vec![0u8; C_BYTES];
                    emulate_core(&a, &b_pack[b0..b0 + g.kc * B_BYTES], g.kc, &mut c32);
                    let o = g.pair_c_offset(col, w, rho);
                    match epi {
                        Epilogue::I32 => out[o..o + cb].copy_from_slice(&c32),
                        Epilogue::Int8 { .. } => {
                            for (i, x) in c32.chunks(4).enumerate() {
                                out[o + i] =
                                    apply_epilogue(i32::from_le_bytes(x.try_into().unwrap()), epi)
                                        as i8 as u8;
                            }
                        }
                        Epilogue::Bf16 => unreachable!("direct BF16 is a G80-only epilogue"),
                    }
                }
            }
        }
        out
    }

    #[test]
    fn v8_apply_epilogue_floors_and_saturates() {
        let e = Epilogue::Int8 { shift: 12 };
        for (c, want) in [
            (0, 0),
            (4095, 0),
            (4096, 1),
            (-1, -1),
            (-4096, -1),
            (-4097, -2),
            (127 << 12, 127),
            ((127 << 12) + 4095, 127),
            (128 << 12, 127),
            (i32::MAX, 127),
            (-(128 << 12), -128),
            (-(129 << 12), -128),
            (i32::MIN, -128),
        ] {
            assert_eq!(apply_epilogue(c, e), want, "{c}");
        }
        assert_eq!(apply_epilogue(-7, Epilogue::I32), -7);
        assert_eq!(apply_epilogue(300, Epilogue::Int8 { shift: 0 }), 127);
        assert_eq!(apply_epilogue(-3, Epilogue::Int8 { shift: 1 }), -2);
    }

    #[test]
    fn v8_variant_and_defaults() {
        assert_eq!("v8".parse::<Variant>(), Ok(Variant::V8));
        assert_eq!(Variant::V8.to_string(), "V8");
        assert!(!Variant::ALL.contains(&Variant::V8));
        assert_eq!(Variant::V8.shape(), (512, 512));
        let d = design_variant(512, 512, 64, Variant::V8);
        assert_eq!(
            (d.variant(), d.epilogue(), d.control()),
            (Variant::V8, Some(V8_DEFAULT_EPILOGUE), Some(Control::Fast))
        );
        let d = design_variant_core(512, 512, 64, Variant::V8, CoreVariant::FastSlowCtl);
        assert_eq!(d.control(), Some(Control::Slow));
        assert!(!d.static_config());
    }

    #[test]
    #[should_panic]
    fn v8_rejects_other_core_programs() {
        design_variant_core(512, 512, 64, Variant::V8, CoreVariant::Serial);
    }

    #[test]
    fn v8_args_and_host_bytes() {
        for (epi, cb) in [
            (Epilogue::I32, 32768usize),
            (Epilogue::Int8 { shift: 5 }, 8192),
        ] {
            let d = design_v8(1024, 512, 192, epi, Control::Fast);
            assert_eq!(d.waves(), 2);
            assert_eq!(
                d.args.iter().map(|a| a.bytes).collect::<Vec<_>>(),
                [8 * 2 * 3 * 4096, 8 * 2 * 3 * 4096, 8 * 2 * 4 * cb]
            );
            assert_eq!(
                d.args.iter().map(|a| a.kind).collect::<Vec<_>>(),
                [ArgKind::In, ArgKind::In, ArgKind::Out]
            );
            assert_eq!(
                (d.host_bytes_read(), d.host_bytes_written()),
                ((2 * 8 * 2 * 3 * 4096) as u64, (8 * 2 * 4 * cb) as u64)
            );
            assert_eq!((d.active_rows(), d.active_cols()), (4, 8));
        }
    }

    #[test]
    fn v8_cpu_exact_through_documented_layouts() {
        // Padding in M and N, two waves in each direction, odd chunk count; every epilogue.
        for epi in V8_EPIS {
            for (m, n, k) in [
                (129, 65, 64),
                (513, 513, 192),
                (100, 1030, 64),
                (512, 512, 256),
            ] {
                let d = design_v8(m, n, k, epi, Control::Fast);
                let (a, b) = (lcg_i8(m * k, 31), lcg_i8(k * n, 32));
                let [pa, pb] = d.pack_in(&a, &b);
                // Large |c| so the int8 epilogue exercises both saturation directions and the floor.
                let got = d.unpack_out(&emulate_v8(&d, &pa, &pb));
                assert_eq!(got, d.reference(&a, &b), "{epi:?} {m}x{n}x{k}");
                assert_eq!(
                    got,
                    reference(&a, &b, m, n, k)
                        .into_iter()
                        .map(|x| apply_epilogue(x, epi))
                        .collect::<Vec<_>>()
                );
            }
        }
    }

    #[test]
    fn v8_pack_a_half_layout_and_replication() {
        let (m, n, k) = (1024, 1024, 128);
        let d = design_v8(m, n, k, Epilogue::I32, Control::Fast);
        let g = d.geo;
        let (a, b) = (lcg_i8(m * k, 41), lcg_i8(k * n, 42));
        let [pa, pb] = d.pack_in(&a, &b);
        assert_eq!((g.waves, g.kc, g.nw, g.mw), (4, 2, 2, 2));
        // Column c, wave (mw,nw), chunk ch, local 8-row block mbl, kb, i, kk: A[row, kcol] by definition.
        for &(c, mw, nw, ch, mbl, kb, i, kk) in &[
            (0, 0, 0, 0, 0, 0, 0, 0),
            (7, 1, 1, 1, 7, 7, 7, 7),
            (3, 1, 0, 1, 4, 2, 5, 6),
            (5, 0, 1, 0, 2, 6, 1, 3),
            (2, 1, 1, 0, 3, 1, 4, 5),
            (6, 0, 0, 1, 6, 3, 2, 0),
        ] {
            let (q, h, par) = (c / 4, (c / 2) % 2, c % 2);
            let w = mw * g.nw + nw;
            let off = c * g.a_segment_bytes()
                + (w * g.kc + ch) * A_HALF_BYTES
                + (mbl * 8 + kb) * 64
                + i * 8
                + kk;
            let row = mw * 512 + (2 * q + par) * 128 + (2 * mbl + h) * 8 + i;
            assert_eq!(
                pa[off] as i8,
                a[row * k + ch * 64 + kb * 8 + kk],
                "c{c} w{w}"
            );
        }
        // A is replicated across nw only, never across columns.
        let seg = g.a_segment_bytes();
        assert_eq!(pa.len(), 8 * seg);
        let tile =
            |c: usize, w: usize| &pa[c * seg + w * g.kc * A_HALF_BYTES..][..g.kc * A_HALF_BYTES];
        assert_eq!(tile(5, 0), tile(5, 1));
        assert_ne!(tile(5, 0), tile(5, 2));
        assert_ne!(tile(4, 0), tile(5, 0));
        // B is exactly the V6 column-segmented layout (B block j = c/2, half c%2 = 64-column group c).
        let v6 = Geometry::new(Topo::FULL, m, n, k);
        assert_eq!(pb, v6.pack_b(&b));
    }

    #[test]
    fn v8_unpack_uses_pair_tiles_and_output_tile_ranges() {
        for epi in V8_EPIS {
            let d = design_v8(600, 1100, 64, epi, Control::Fast);
            let g = d.geo;
            let cb = g.topo.c_tile_bytes();
            let mut out = vec![0u8; d.args[2].bytes];
            let ranges = d.output_tile_ranges();
            assert_eq!(ranges.len(), 32);
            // Ranges are disjoint, C-tile sized and cover the argument exactly.
            let mut sorted: Vec<_> = ranges
                .iter()
                .flat_map(|(_, rs)| rs.iter().cloned())
                .collect();
            sorted.sort_by_key(|r| r.start);
            let mut at = 0;
            for r in &sorted {
                assert_eq!((r.start, r.len()), (at, cb));
                at = r.end;
            }
            assert_eq!(at, d.args[2].bytes);
            for (tag, (_, rs)) in ranges.iter().enumerate() {
                for (w, r) in rs.iter().enumerate() {
                    // First element of the tile's first block: value tags the tile (within int8 range).
                    out[r.start] = (1 + tag + 32 * w) as u8 % 100;
                }
            }
            let c = d.unpack_out(&out);
            for (tag, ((col, row), rs)) in ranges.iter().enumerate() {
                for w in 0..rs.len() {
                    let (mw, nw) = (w / g.nw, w % g.nw);
                    let (dm, dn) = Geometry::pair_tile(*col as usize, *row as usize - 2);
                    let (r0, c0) = (mw * 512 + dm, nw * 512 + dn);
                    if r0 < 600 && c0 < 1100 {
                        assert_eq!(
                            c[r0 * 1100 + c0],
                            ((1 + tag + 32 * w) as u8 % 100) as i32,
                            "{epi:?} tile {col},{row} w{w}"
                        );
                    }
                }
            }
        }
        // The documented core-to-C-block map: A block 2q+c%2, B block c/2, half rho%2.
        assert_eq!(Geometry::pair_tile(0, 0), (0, 0));
        assert_eq!(Geometry::pair_tile(1, 1), (128, 64));
        assert_eq!(Geometry::pair_tile(6, 3), (256, 3 * 128 + 64));
        assert_eq!(Geometry::pair_tile(7, 2), (3 * 128, 3 * 128));
    }

    fn pair() -> Topo {
        Topo::pair(Epilogue::Int8 { shift: 12 })
    }

    #[test]
    fn v8_no_master_driven_twice_and_ports_valid() {
        let mut driven: HashSet<(u32, u32, String)> = HashSet::new();
        for c in circuits(pair()) {
            let w = c.writes();
            assert!(w[0].0 != w[1].0);
            let key = (c.tile.col, c.tile.row, format!("{:?}", c.master));
            assert!(driven.insert(key.clone()), "master driven twice: {key:?}");
        }
    }

    #[test]
    fn v8_streams_reach_exactly_the_documented_endpoints() {
        let topo = pair();
        let all = circuits(topo);
        let dma = |n: u8| format!("{:?}", Port::Dma(n));
        for col in 0..COLS as u32 {
            let shim = shim_loc(col);
            let port = |direction, channel| {
                ShimDma {
                    tile: shim,
                    direction,
                    channel,
                }
                .port()
            };
            let c = col as usize;
            let (q, h, par) = (c / 4, (c / 2) % 2, c % 2);
            // Shim -> memtile.
            assert_eq!(
                reach(&all, (shim, port(Mm2s, 0))),
                HashSet::from([(col, 1, dma(B_S2MM_CH as u8))]),
                "B in {col}"
            );
            assert_eq!(
                reach(&all, (shim, port(Mm2s, 1))),
                HashSet::from([(col, 1, dma(A_S2MM_CH as u8))]),
                "A in {col}"
            );
            // A half: core row 2q+h, Dma0 of every column with the same parity (itself included), nowhere else.
            let rho = 2 * q + h;
            let want: HashSet<_> = (0..COLS as u32)
                .filter(|x| *x as usize % 2 == par)
                .map(|x| (x, rho as u32 + 2, dma(0)))
                .collect();
            assert_eq!(
                reach(&all, (mem_loc(col), Port::Dma(A_MM2S_CH as u8))),
                want,
                "A out col {col}"
            );
            // B half: injected half `par` feeds rows par, par+2 of this column and, through the lane-2 crossing,
            // the same rows of the neighbour column (whose own half is the other one).
            let nb = (c ^ 1) as u32;
            let want: HashSet<_> = [(col, par), (col, par + 2), (nb, par), (nb, par + 2)]
                .into_iter()
                .map(|(x, r)| (x, r as u32 + 2, dma(1)))
                .collect();
            assert_eq!(
                reach(&all, (mem_loc(col), Port::Dma(B_MM2S_CH as u8))),
                want,
                "B out col {col}"
            );
            // C: each core row reaches only its memtile S2MM; rows 0,1 leave on MM2S0 to shim S2MM0, rows 2,3 on MM2S3 to S2MM1.
            for r in 0..ROWS {
                assert_eq!(
                    reach(&all, (core_loc(col, r), Port::Dma(0))),
                    HashSet::from([(col, 1, dma(r as u8))]),
                    "C col {col} row {r}"
                );
            }
            assert_eq!(
                reach(&all, (mem_loc(col), Port::Dma(PAIR_C_MM2S_CH[0] as u8))),
                HashSet::from([(col, 0, format!("{:?}", port(S2mm, 0)))])
            );
            assert_eq!(
                reach(&all, (mem_loc(col), Port::Dma(PAIR_C_MM2S_CH[1] as u8))),
                HashSet::from([(col, 0, format!("{:?}", port(S2mm, 1)))])
            );
        }
    }

    #[test]
    fn v8_every_core_dma_input_has_exactly_one_source_per_wave_stream() {
        // Each core tile's Dma0 / Dma1 masters are driven by exactly one circuit.
        let topo = pair();
        let all = circuits(topo);
        for col in 0..COLS as u32 {
            for r in 0..ROWS {
                for n in 0..2u8 {
                    let drivers = all
                        .iter()
                        .filter(|c| c.tile == core_loc(col, r) && c.master == Port::Dma(n))
                        .count();
                    assert_eq!(drivers, 1, "core ({col},{r}) Dma{n}");
                }
            }
        }
    }

    #[test]
    fn v8_memtile_descriptors_banks_locks_and_chains() {
        for epi in V8_EPIS {
            let topo = Topo::pair(epi);
            for col in 0..COLS as u32 {
                let bds = memtile_descriptors(topo, col);
                let ids: Vec<u32> = bds.iter().map(|b| b.0).collect();
                assert_eq!(ids.len(), 24);
                assert_eq!(
                    ids.iter().collect::<HashSet<_>>().len(),
                    ids.len(),
                    "duplicate BD id"
                );
                assert!(ids.iter().all(|&i| i < 48));
                let mut ranges = Vec::new();
                for (_, bd) in &bds {
                    let start = bd.addr as u32 - MEM_BASE;
                    ranges.push((start, bd.len_words * 4));
                    assert!(start + bd.len_words * 4 <= 512 * 1024);
                    assert!(
                        bd.next.is_some_and(|n| ids.contains(&n)),
                        "chain leaves descriptor set"
                    );
                }
                disjoint_or_equal(&ranges);
                // Bank rule: a channel's BDs live in its bank (checked by Task::write for the first BD of each ring/chain);
                // every BD of a chain stays in the bank of its first BD.
                let next: HashMap<u32, u32> =
                    bds.iter().map(|(i, b)| (*i, b.next.unwrap())).collect();
                let addr: HashMap<u32, u32> = bds
                    .iter()
                    .map(|(i, b)| (*i, b.addr as u32 - MEM_BASE))
                    .collect();
                let words: HashMap<u32, u32> = bds.iter().map(|(i, b)| (*i, b.len_words)).collect();
                let cycle = |first: u32, len: usize| -> Vec<u32> {
                    let mut at = first;
                    let seq: Vec<u32> = (0..len)
                        .map(|_| {
                            let cur = at;
                            at = next[&at];
                            cur
                        })
                        .collect();
                    assert_eq!(at, first);
                    assert!(
                        seq.iter().all(|&b| (b >= 24) == (first >= 24)),
                        "chain crosses the BD bank"
                    );
                    seq
                };
                for g in 0..2 {
                    let seq = cycle(PAIR_C_MM2S_BD0[g], 4);
                    for (i, id) in seq.iter().enumerate() {
                        assert_eq!(addr[id], c_offset(i / 2, 2 * g + i % 2));
                        assert_eq!(words[id] as usize * 4, topo.c_tile_bytes());
                    }
                }
                for (first, a0, len) in [
                    (B_S2MM_BD[0], B_OFF[0], B_BYTES),
                    (B_MM2S_BD[0], B_OFF[0], B_BYTES),
                    (A_S2MM_BD[0], A_OFF[0], A_HALF_BYTES),
                    (A_MM2S_BD[0], A_OFF[0], A_HALF_BYTES),
                ] {
                    let seq = cycle(first, 2);
                    assert_eq!((addr[&seq[0]], words[&seq[0]] as usize * 4), (a0, len));
                }
                for r in 0..ROWS {
                    let seq = cycle(C_S2MM_BD[r][0], 2);
                    assert_eq!(
                        seq.iter().map(|b| addr[b]).collect::<Vec<_>>(),
                        [c_offset(0, r), c_offset(1, r)]
                    );
                }
                // Locks: every one is released as often as it is acquired over its ring.
                let mut net: HashMap<u32, i32> = HashMap::new();
                for (_, bd) in &bds {
                    for (id, v) in [bd.locks.acq, bd.locks.rel].into_iter().flatten() {
                        *net.entry(id).or_default() += v;
                    }
                }
                assert!(net.values().all(|&v| v == 0), "{net:?}");
                let init: HashMap<u32, u32> = memtile_locks(topo, col).into_iter().collect();
                assert!(net.keys().all(|id| init.contains_key(&(id - MEM_LOCK))));
                assert_eq!(init.len(), net.len());
                assert!(
                    init.iter()
                        .all(|(&id, &v)| v == u32::from(is_empty_lock(Topo::FULL, id))),
                    "{init:?}"
                );
            }
        }
    }

    #[test]
    fn v8_ring_tasks_and_reset_channels() {
        let topo = pair();
        for col in 0..COLS as u32 {
            for (tile, task) in ring_tasks(topo, col) {
                let _ = task.write(tile);
            }
            let chans = used_channels(topo, col);
            assert!(chans.iter().all(|(l, ..)| l.kind() != TileType::Shim));
            let uniq: HashSet<_> = chans
                .iter()
                .map(|(l, d, c)| (l.row, *d == S2mm, *c))
                .collect();
            assert_eq!(uniq.len(), chans.len());
            // 3 per core (12) + memtile S2MM: 4 C rows + B + A; MM2S: 2 C chains + B + A.
            assert_eq!(chans.len(), 12 + 6 + 4);
            let mem: Vec<_> = chans
                .iter()
                .filter(|(l, ..)| l.kind() == TileType::Memtile)
                .collect();
            assert_eq!(
                mem.iter().filter(|(_, d, c)| *d == Mm2s && *c == 3).count(),
                1
            );
        }
    }

    #[test]
    fn v8_txn_syncs_patches_and_resubmit_structure() {
        for epi in V8_EPIS {
            let d = design_v8(1024, 512, 192, epi, Control::Fast);
            let g = d.geo;
            let ops = txn_ops(&d.insts);
            // Both shim S2MM channels of all 8 columns, one aggregate SYNC each, last in the stream.
            let syncs: Vec<_> = ops.iter().filter(|o| o.kind == 0x80).collect();
            assert_eq!(syncs.len(), 2);
            assert_eq!(
                ops.iter().rposition(|o| o.kind == 0x80),
                Some(ops.len() - 1)
            );
            for (chan, s) in syncs.iter().enumerate() {
                assert_eq!(
                    (s.addr >> 16 & 0xff, s.addr >> 8 & 0xff, s.addr & 0xff),
                    (0, 0, 0)
                );
                assert_eq!(
                    (s.value >> 24, s.value >> 16 & 0xff, s.value >> 8 & 0xff),
                    (chan as u32, COLS as u32, 1)
                );
            }
            // DDR patches: A and B by column; C twice per column (S2MM0 stream, S2MM1 stream after it).
            let mut per_arg: HashMap<u32, Vec<(u32, u64)>> = HashMap::new();
            for p in ops.iter().filter(|o| o.kind == 0x81) {
                per_arg
                    .entry(p.value)
                    .or_default()
                    .push(((p.addr >> regs::COL_SHIFT) & 7, p.plus));
            }
            let cs = g.c_segment_bytes();
            let half = g.waves * 2 * g.topo.c_tile_bytes();
            for (arg, seg) in [(0u32, g.a_segment_bytes()), (1, g.b_segment_bytes())] {
                let mut got = per_arg[&arg].clone();
                got.sort();
                assert_eq!(
                    got,
                    (0..COLS)
                        .map(|c| (c as u32, (c * seg) as u64))
                        .collect::<Vec<_>>(),
                    "arg{arg}"
                );
            }
            let mut got = per_arg[&2].clone();
            got.sort();
            let want: Vec<_> = (0..COLS)
                .flat_map(|c| {
                    [
                        (c as u32, (c * cs) as u64),
                        (c as u32, (c * cs + half) as u64),
                    ]
                })
                .collect();
            assert_eq!(got, want);
            assert_eq!(2 * half, cs);
            // Reset + restart of every core in the TXN, no shim channel reset.
            let core_ctrl =
                |o: &&Op| o.kind == 3 && o.addr & 0x000f_ffff == regs::core::CORE_CONTROL;
            assert_eq!(ops.iter().filter(core_ctrl).count(), 3 * 32);
            for col in 0..COLS as u32 {
                let shim = shim_loc(col);
                for off in [
                    regs::shim::DMA_S2MM_0_CTRL,
                    regs::shim::DMA_S2MM_0_CTRL + 8,
                    regs::shim::DMA_MM2S_0_CTRL,
                    regs::shim::DMA_MM2S_0_CTRL + 8,
                ] {
                    assert!(
                        !ops.iter()
                            .any(|o| o.kind == 3 && o.addr == shim.address(off) && o.mask & 2 != 0),
                        "shim channel reset"
                    );
                }
                // The token controller id is set for both S2MM channels and both shim tasks issue a token.
                for off in [regs::shim::DMA_S2MM_0_CTRL, regs::shim::DMA_S2MM_0_CTRL + 8] {
                    assert_eq!(
                        ops.iter()
                            .filter(|o| o.kind == 3
                                && o.addr == shim.address(off)
                                && o.mask == 0x1f00)
                            .count(),
                        1
                    );
                }
                for off in [
                    regs::shim::DMA_S2MM_0_CTRL + 4,
                    regs::shim::DMA_S2MM_0_CTRL + 12,
                ] {
                    let q: Vec<_> = ops
                        .iter()
                        .filter(|o| o.kind == 0 && o.addr == shim.address(off))
                        .collect();
                    assert_eq!(q.len(), 1);
                    assert_eq!(q[0].value >> 31, 1, "S2MM token");
                }
            }
        }
    }

    #[test]
    fn v8_pdi_loads_lower_and_upper_programs_into_the_right_rows() {
        let d = design_v8(512, 512, 64, Epilogue::Int8 { shift: 12 }, Control::Fast);
        let lower = d.core_program();
        let upper = d.core_program_upper().unwrap();
        assert!(lower.len() <= 16 * 1024 && upper.len() <= 16 * 1024);
        let words = |p: &[u8]| -> Vec<u32> {
            p.chunks(4)
                .map(|c| {
                    let mut w = [0; 4];
                    w[..c.len()].copy_from_slice(c);
                    u32::from_le_bytes(w)
                })
                .collect()
        };
        let cmds =
            crate::cdo::parse(&build_cdo(d.geo.topo, [lower, upper], false).to_words()).unwrap();
        for col in 0..COLS as u32 {
            for r in 0..ROWS {
                let addr = u64::from(core_loc(col, r).address(regs::core::PROGRAM_MEMORY));
                let writes: Vec<_> = cmds
                    .iter()
                    .filter(|c| matches!(c, crate::cdo::Cmd::DmaWrite(a, _) if *a == addr))
                    .collect();
                assert_eq!(writes.len(), 1, "core ({col},{r})");
                let want = words(if r % 2 == 0 { lower } else { upper });
                assert!(
                    matches!(writes[0], crate::cdo::Cmd::DmaWrite(_, w) if *w == want),
                    "core ({col},{r}) program role"
                );
            }
        }
    }

    #[test]
    fn v8_estimate_model() {
        let clk = 1_800_000_000;
        let [fast_ping, fast_pong] = gemm_core::PAIR_CHUNK_CYCLES[0];
        // kc=40, one wave, Int8: 20 ping + 20 pong chunks + conversion, then the exposed 2-channel shim C
        // transfer (2 tiles x 8 KiB / 4 B).
        let d = design_v8(512, 512, 2560, Epilogue::Int8 { shift: 12 }, Control::Fast);
        let e = d.estimate(clk);
        assert_eq!(
            (e.ideal_mac_cycles, e.input_dma_cycles),
            (1024 * 40, 1024 * 40)
        );
        assert_eq!(e.output_dma_cycles, gemm_core::PAIR_EPILOGUE_CYCLES[0]);
        assert_eq!(
            e.estimated_cycles,
            20 * (fast_ping + fast_pong) + gemm_core::PAIR_EPILOGUE_CYCLES[0] + 4096
        );
        // The Slow discipline uses its own chunk and epilogue cycles.
        let [slow_ping, slow_pong] = gemm_core::PAIR_CHUNK_CYCLES[1];
        let d = design_v8(512, 512, 2560, Epilogue::Int8 { shift: 12 }, Control::Slow);
        assert_eq!(
            d.estimate(clk).estimated_cycles,
            20 * (slow_ping + slow_pong) + gemm_core::PAIR_EPILOGUE_CYCLES[1] + 4096
        );
        // I32 adds the 8192-cycle C drain per wave and a 16384-cycle shim transfer; waves overlap it.
        let d = design_v8(1024, 1024, 128, Epilogue::I32, Control::Fast);
        let e = d.estimate(clk);
        let wave = fast_ping + fast_pong + 8192;
        assert_eq!(e.output_dma_cycles, 4 * 8192);
        assert_eq!(e.estimated_cycles, 3 * wave.max(16384) + wave + 16384);
        assert!((e.estimated_seconds - e.estimated_cycles as f64 / clk as f64).abs() < 1e-15);
        assert_eq!(e.peak_ops_per_second, peak_ops_per_second(clk));
    }

    /// Every circuit that leaves a tile towards a neighbour must be received by a circuit of that neighbour
    /// (otherwise the stream stalls on a slave without a route), and every neighbour-facing slave of a circuit must be
    /// fed by a circuit of that neighbour.
    #[test]
    fn v8_no_dangling_stream_links() {
        let all = circuits(pair());
        let has = |tile: Location, slave: Option<Port>, master: Option<Port>| {
            all.iter().any(|c| {
                c.tile == tile
                    && slave.map_or(true, |s| c.slave == s)
                    && master.map_or(true, |m| c.master == m)
            })
        };
        for c in &all {
            let (col, row) = (c.tile.col as i32, c.tile.row as i32);
            let at = |dc: i32, dr: i32| Location::new((col + dc) as u32, (row + dr) as u32);
            // Shim (row 0) South ports are the shim DMAs, not a neighbour.
            let next = match c.master {
                Port::North(l) => Some((at(0, 1), Port::South(l))),
                Port::South(l) if row > 0 => Some((at(0, -1), Port::North(l))),
                Port::East(l) => Some((at(1, 0), Port::West(l))),
                Port::West(l) => Some((at(-1, 0), Port::East(l))),
                _ => None,
            };
            if let Some((tile, slave)) = next {
                assert!(has(tile, Some(slave), None), "{c:?} has no receiver");
            }
            let prev = match c.slave {
                Port::South(l) if row > 0 => Some((at(0, -1), Port::North(l))),
                Port::North(l) => Some((at(0, 1), Port::South(l))),
                Port::East(l) => Some((at(1, 0), Port::West(l))),
                Port::West(l) => Some((at(-1, 0), Port::East(l))),
                _ => None,
            };
            if let Some((tile, master)) = prev {
                assert!(has(tile, None, Some(master)), "{c:?} has no feeder");
            }
        }
    }

    #[test]
    fn v8_expert_shapes_fit_limits() {
        let mut shapes = vec![(512, 512, 64), (512, 512, 256)];
        for m in [512, 1024, 2048, 4096] {
            shapes.push((m, 1280, 2560));
            shapes.push((m, 2560, 640));
        }
        for epi in V8_EPIS {
            for (m, n, k) in shapes.iter().copied() {
                let d = design_v8(m, n, k, epi, Control::Fast);
                assert!(d.waves() <= MAX_WAVES, "{m}x{n}x{k}");
                let cb = d.geo.topo.c_tile_bytes();
                assert_eq!(
                    d.args.iter().map(|a| a.bytes).collect::<Vec<_>>(),
                    [
                        8 * d.waves() * (k / 64) * 4096,
                        8 * d.waves() * (k / 64) * 4096,
                        8 * d.waves() * 4 * cb
                    ]
                );
            }
        }
        // Largest K and wave counts keep both pair programs inside the 16 KiB core program memory, both disciplines.
        for ctl in [Control::Fast, Control::Slow] {
            for role in [PairRole::Lower, PairRole::Upper] {
                for epi in V8_EPIS {
                    let p = gemm_core::program_pair(MAX_KC, MAX_WAVES, role, epi, ctl).finish();
                    assert!(
                        p.len() <= 16 * 1024,
                        "{role:?} {epi:?} {ctl:?}: {}",
                        p.len()
                    );
                }
            }
        }
    }

    #[test]
    fn v10_memory_boundary_and_down_shapes() {
        let epi = Epilogue::Int8 { shift: 12 };
        // kc*(2+NW) = 112 exactly fits (NW 5, kc 16); down shapes (kc 10) fit.
        design_v10(512, 2560, 1024, epi, Control::Fast);
        for (m, n) in [(2048, 2560), (4096, 2560), (4096, 1280)] {
            let d = design_v10(m, n, 640, epi, Control::Fast);
            assert_eq!(d.variant(), Variant::V10);
        }
        // Model host read of the down shape: 8*4096*10*(MW 8 + NW 5) = 4,259,840 B.
        let d = design_v10(4096, 2560, 640, epi, Control::Slow);
        assert_eq!((d.geo.mw, d.geo.nw, d.geo.kc), (8, 5, 10));
        assert_eq!(
            d.args.iter().map(|a| a.bytes).collect::<Vec<_>>(),
            [
                8 * 8 * 10 * 4096,
                8 * 5 * 10 * 4096,
                8 * d.waves() * 4 * COUT_BYTES
            ]
        );
        assert_eq!(d.host_bytes_read(), 4_259_840);
    }

    #[test]
    #[should_panic(expected = "V10 memtile overflow")]
    fn v10_rejects_one_chunk_over_the_memory_bound() {
        design_v10(512, 2560, 1088, Epilogue::Int8 { shift: 12 }, Control::Fast);
    }

    #[test]
    #[should_panic(expected = "V10 memtile overflow")]
    fn v10_rejects_gate_up_kc40() {
        design_v10(
            4096,
            1280,
            2560,
            Epilogue::Int8 { shift: 12 },
            Control::Fast,
        );
    }

    #[test]
    #[should_panic(expected = "V10 supports the int8 epilogue only")]
    fn v10_rejects_i32_epilogue() {
        design_v10(512, 512, 64, Epilogue::I32, Control::Fast);
    }

    #[test]
    fn v10_descriptors_stay_inside_memtile_and_bd_length_limit() {
        let epi = Epilogue::Int8 { shift: 12 };
        // Shapes on the `kc*(2+NW) <= 112` bound for several NW, MW 16 and NW 8 limits.
        for (mw, nw, kc) in [(1, 5, 16), (16, 1, 37), (2, 8, 11), (3, 3, 22), (16, 8, 1)] {
            let res = Resident {
                mode: ResidentMode::MwOuter,
                kc,
                mw,
                nw,
                b_prefix: false,
            };
            let bds: HashMap<u32, Bd> =
                memtile_descriptors(Topo::with_resident(epi, Control::Fast, res), 0)
                    .into_iter()
                    .collect();
            let slot = (kc * 4096) as u32;
            let range = |id: u32| {
                let b = &bds[&id];
                let s = b.addr as u32 - MEM_BASE;
                (s, s + b.len_words * 4)
            };
            // Compact C ends at 0x10000, A slot 0/1 and B follow contiguously and end inside the memtile.
            for id in [
                PAIR_C_MM2S_BD0[0],
                PAIR_C_MM2S_BD0[0] + 3,
                PAIR_C_MM2S_BD0[1],
                PAIR_C_MM2S_BD0[1] + 3,
            ] {
                assert!(range(id).1 <= 0x10000, "C BD {id}");
            }
            assert_eq!(range(A_S2MM_BD[0]), (0x10000, 0x10000 + slot));
            assert_eq!(range(A_S2MM_BD[1]), (0x10000 + slot, 0x10000 + 2 * slot));
            let b = range(B_S2MM_BD[0]);
            assert_eq!(b, range(B_MM2S_BD[0]));
            assert_eq!(b.0, 0x10000 + 2 * slot);
            assert!(
                b.1 <= 0x80000 && bds[&B_S2MM_BD[0]].len_words <= 131071,
                "mw={mw} nw={nw} kc={kc}"
            );
            assert_eq!(b.1 == 0x80000, kc * (2 + nw) == V10_MAX_SLOTS);
            // The A replay chain covers one slot per BD and toggles through the next one.
            let chain: Vec<_> = (0..nw as u32)
                .map(|i| &bds[&(V10_A_CHAIN_BD0 + i)])
                .collect();
            assert!(chain.iter().all(|bd| bd.iteration.step * 4 == slot
                && bd.len_words * 4 == slot
                && bd.iteration.wrap == 2));
        }
    }
}
