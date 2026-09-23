# Slice 2 — ACCEPT

## Kernel A/B/A

Ordinal 2; immutable `99c44a96b` baseline A versus Slice 2 candidate B. Each arm used one untimed post-JIT launch, 5 warmups, and 20 measured HIP events.

- gate-up M4096/K5120/N34816: A1 7101.385 us, B 6856.243 us, A2 7112.504 us, A mean 7106.945 us, candidate +3.657%
- residual M4096/K17408/N5120: A1 3174.478 us, B 3030.876 us, A2 3168.878 us, A mean 3171.678 us, candidate +4.646%

Evidence: `kernel-aba-stable.log`.

## ISA and resources

- IU4 WMMAs per steady K128: 32
- barrier signal/wait pairs per steady K128: 2
- pair 1: `0xE994/0xE9E4`
- pair 2: `0xEC98/0xEC9C`
- prologue pair: `0xE180/0xE184`
- steady LOAD waits: `loadcnt 0` before pair 1; `loadcnt 3,2,1,0` before pair 2
- full SET/ADD VGPR: 202
- full SET/ADD SGPR: 40
- wrapper SGPR: 43
- scratch: 0 B/lane
- VGPR spills: 0
- SGPR spills: 0
- compiler occupancy: 7 waves/SIMD
- dynamic LDS: 20,480 B/block
- runtime occupancy: 3 blocks/CU

Evidence: `objdump.txt`, `resource.log`, `kernel-aba-stable.log`.

## Exactness

Device-resident shipped-versus-candidate comparison on identical resident buffers: 9/9 PASS, 0 device bit mismatches, 0 repeat mismatches, 0 canary mismatches, 0 CPU-reference mismatches. Cases: gate SET/ADD, down SET/ADD, M48 tail SET, N80 tail SET, K256 SET/seeded ADD, M100/N100 K512 SET.

Evidence: `oracle-full.log`.

## Pinned serving A/B

- baseline CLI MD5: `19c832084440f7142181a54ea1607c8f`
- candidate CLI MD5: `a338e40e625a0ab21536048288046ad6`
- baseline daemon MD5: `4564127b36d1e217ab21706f58a35d3a`
- candidate daemon MD5: `b5543670fd1743fc7253c08f7853d34d`
- baseline IU4 cache key: `bd779269c415f5a0`
- candidate IU4 cache key: `62be8413199d7e5c`
- baseline IU4 HSACO MD5: `f5771416a6f2d54651027ea3138dfa38`
- candidate IU4 HSACO MD5: `104224043106de96f6f42c17bf9be093`

Rows:

- pp512: 2285.0 -> 2334.6 tok/s, +2.170678%
- pp2048: 2359.8 -> 2408.4 tok/s, +2.059497%
- pp8192: 2268.6 -> 2307.9 tok/s, +1.732346%
- pp32768: 1902.2 -> 1936.0 tok/s, +1.776890%
- tg64@ctx128: 36.5173368385 -> 36.4778357623 tok/s, -0.108171%

Evidence: `clean-pinned-B.log`, `clean-pinned-C.log`.

## Attribution

Three traced pp8192 passes:

- IU4 combined sum: 6219.046440 -> 5855.491459 ms, -5.845832%
- other-kernel sum: 4483.291498 -> 4471.332080 ms, -0.266755%
- total kernel sum: 10702.337938 -> 10326.823539 ms, -3.508714%
- traced pp8192 serving: 2283.3 -> 2365.8 tok/s, +3.613191%

Evidence: `pp8192-trace-attribution.json`.

## Outcome

ACCEPT. Commit `c2eb81fcd53f4bac97a92c3aec9f1e811ea34c24`.
