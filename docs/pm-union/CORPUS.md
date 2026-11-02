# PM union U0 corpus freeze: MQ4 residual four-arm corpus

Corpus id: `pm-union-u0-mq4-residual-4cell-v1` - frozen 2026-10-11. Machine-readable twin: [`corpus.json`](corpus.json) (same directory; this prose and the JSON are published together and must be identical to `<release-workdir>/pm-union/{CORPUS.md,corpus.json}`). Slice ownership: [`OWNERS.md`](OWNERS.md).

**Status (revision 2): definition plus observed pilot export pins.** The only computation in this document is CPU hashing/reading with `sha256sum` and a one-pass CPU crop/scan of the fixture bytes. The pilot has since **exported** the V0 SPIR-V and the ACO/RADV raw binary, ABI sidecar and rawspec (section 4.1); this is an export only. There is **no** raw lift, HSA adapter, PM-HIP object, Vulkan reference receipt, binary replay, or equality/parity result, and none is claimed. Hpm and Vpm do not exist, no result digest exists for any arm. The pilot (ACO pilot F1-F4) is untouched; its files are cited, not edited. Rev 2 changes only section 4.1, section 4 arm table, section 10 and `corpus.json` `exported_artifacts`/`arms`/`limits`/`status`; the four cells, fixture bytes, `fixtures_sha256` and all section 8 source pins are unchanged from rev 1.

Plan source: `pm-programs/PM-UNION.md` section 5 (`:168-208`), section 7 (`:263-317`), section 8 U0 (`:323-327`). Source paths: `H/` = the repository root (paths below are repository-relative; the U0 worktree has the same bytes for every H file pinned here, HEAD `2bb96700`), `M/` = `<peacemaker-workdir>/mesa/` (Mesa `mesa-26.0.8`, commit `60e95b787857afbc9a00b693b91c0d9c8923a430`, `<peacemaker-workdir>/aco-shim/pin.json:2-8`), `Q/` = `<release-workdir>/` (placeholder for the local release working directory; evidence under `Q/`, `M/` and `P/` is local-only and not tracked in this repository), `P/` = `Q/worktrees/pm-aco-pilot-20261011/` (pilot, base `d7ae3672`, files below are uncommitted working-tree files at freeze).

## 1. Kernel under test

`gemv_mq4g256v2_residual` (module `gemv_hfq4g256_residual_mq4v2`, gfx1201, wave32, block `[32,1,1]`, no LDS/scratch): `y[row] += A[row] . x`, MQ4-G256 v2 weights, 136 bytes per 256-weight group (two f16 scale/zero headers + 128 packed nibble bytes), x/y f32. Kernarg ABI, 32 bytes little-endian: A ptr @0, x ptr @8, y ptr @16, M i32 @24, K i32 @28. One workgroup per row pair (`row0 = wg.x << 1`, exits when `row0 >= M`), launcher rule grid = M.

Recipe (not re-derived here; arms are measured against it): four independent group streams per row, `acc = (a0+a1)+(a2+a3)`, `shfl_down` 16/8/4/2/1 tree, epilogue `y.x = acc + y.x`, `y.y = bcc + y.y` (odd last row `y = y + acc`), second-row dot starts `x1` then `x0`. Source: `H/kernels/src/gemv_mq4g256v2_residual.hip:43-258` (signature `:43-48`, row pair `:49-52`, stride `:61-69`, quad loop `:114-193`, tails `:195-236`, reduce `:242-247`, epilogue `:249-258`); SPIR-V source twin `P/crates/peacemaker-lift/tests/fixtures/aco/src/gemv_mq4g256v2_residual.comp:119-181`; builder twin `H/crates/hipfire-isa/src/kernels/pm_decode/gemv_hfq4g256_residual_mq4v2.rs:1-50`. The incumbent object is compiled from the registry snapshot (preamble + kernel), whose text differs from `kernels/src/...hip` by a 55-line weight-buffer preamble; this corpus binds the incumbent to the snapshot + object digest and does not assert text equality (`corpus.json` `limits`).

## 2. Four cells

| Cell | Origin | M | K | groups | quads+tail | grid | A bytes | x bytes | y bytes |
|---|---|---|---|---|---|---|---|---|---|
| `C1_occ0_full` | real capture occ0 | 5120 | 6144 | 24 | 6+0 | [5120,1,1] | 16711680 | 24576 | 20480 |
| `C2_occ127_full` | real capture occ127 | 5120 | 17408 | 68 | 17+0 | [5120,1,1] | 47349760 | 69632 | 20480 |
| `C3_odd_m5119_k5888` | real-prefix crop of occ0 | 5119 | 5888 | 23 | 5+3 | [5119,1,1] | 16012232 | 23552 | 20476 |
| `C4_odd_m5119_k17152` | real-prefix crop of occ127 | 5119 | 17152 | 67 | 16+3 | [5119,1,1] | 46644328 | 68608 | 20476 |

Cells C1/C2 are the real 27B decode-callsite captures (`pmdt-capture-v1`, gfx1201, context 512, launch 6 and 751 of 755, occurrence 0 and 127 of 128). C3/C4 are deterministic crops of those same bytes. Tail groups 1 and 2 are not covered. The pilot's additional minimal-grid (`ceil(M/2)`) geometry is not part of this corpus (the V0 Vulkan reference runs only the captured launcher rule, `pm_aco_oracle.rs:380`).

### 2.1 Fixture bytes (actual `sha256sum`, run once, not a test)

Directories `Q/pm-decode-twins/oracle-inputs/gemv_mq4g256v2_residual/{occ0,occ127}/`. Each `.bin` was also length-checked: `M*(K/256)*136`, `K*4`, `M*4`.

| File | Bytes | sha256 |
|---|---|---|
| `occ0/arg0.pre.bin` | 16711680 | `81399d75f09fb226030f0dc836b0bb46bfc2c955d4008ffedf8cabf42386ac0a` |
| `occ0/arg0.post.bin` | 16711680 | `81399d75f09fb226030f0dc836b0bb46bfc2c955d4008ffedf8cabf42386ac0a` |
| `occ0/arg8.pre.bin` | 24576 | `afa96e019383b5f252d7a1eb77e634dba1328898ea0974a2efebff85351b4621` |
| `occ0/arg8.post.bin` | 24576 | `afa96e019383b5f252d7a1eb77e634dba1328898ea0974a2efebff85351b4621` |
| `occ0/arg16.pre.bin` | 20480 | `33bea66ece8108a06b4cdeb3409d20400b62794e0d13857cccb6ed3510eaec73` |
| `occ0/arg16.post.bin` | 20480 | `8c48e1e308ff86fd903baf1a72f516abeb0a21d0d505ed903466236711484827` |
| `occ0/meta.json` | - | `22dead9e3a1887d15230d0653281724f52ac445ba155e206d224e6da831de0b6` |
| `occ127/arg0.pre.bin` | 47349760 | `ee203d35f31437865ba9dd311db8182a6d3d0438471133eba2ee56b4f1c25bbb` |
| `occ127/arg0.post.bin` | 47349760 | `ee203d35f31437865ba9dd311db8182a6d3d0438471133eba2ee56b4f1c25bbb` |
| `occ127/arg8.pre.bin` | 69632 | `e3d5e5cdf8e3e2ad1fca72c26b6422449598be83e7c153429848d05d0a714100` |
| `occ127/arg8.post.bin` | 69632 | `e3d5e5cdf8e3e2ad1fca72c26b6422449598be83e7c153429848d05d0a714100` |
| `occ127/arg16.pre.bin` | 20480 | `a6ca8432d95c2c99674b43244a7eb6670fdfaa69cf0ef974b3b8425b98aeeefc` |
| `occ127/arg16.post.bin` | 20480 | `4fb1147e09c3c306f91e8ae3c153b96a81403f833d1eb5ab9424424fb23b0ada` |
| `occ127/meta.json` | - | `85062964edbd75e896f995628225c10c6be4a20a1c8f55785288598f357870af` |

A and x post bytes equal pre bytes (identical digests) for both captures. `arg16.post.bin` is the captured callsite result, recorded only as evidence; it is not any arm's output.

### 2.2 Deterministic row-crop recipe and derived digests

For a real capture with `G = K/256` groups per row: `M_out = M-1`, `groups_out = G-1`, `K_out = 256*(G-1)`; `A_out = concat_{r<M_out} A_pre[r*G*136 : r*G*136 + (G-1)*136]`; `x_out = x_pre[0:(G-1)*1024]`; `y_out = y_pre[0:M_out*4]`; kernarg pointer slots zero, `M_out`@24, `K_out`@28; grid `[M_out,1,1]`. Reference implementation: `P/crates/rdna-compute/examples/pm_aco_oracle.rs:342-383` (A row-crop `:356-361`). Digests below were computed on the CPU from the actual fixture bytes by this recipe.

| Cell | Buffer | Payload sha256 (unframed pre) | Framed pre sha256 (4096 x 0xA5 + payload + 4096 x 0xA5) |
|---|---|---|---|
| `C1_occ0_full` | A | `81399d75f09fb226030f0dc836b0bb46bfc2c955d4008ffedf8cabf42386ac0a` | `2be2589ca6432837d15868da4e46fdad77b10df831c2302406a20f90ee156516` |
| `C1_occ0_full` | x | `afa96e019383b5f252d7a1eb77e634dba1328898ea0974a2efebff85351b4621` | `04427dda6ddcfe6ac168a8c2adca890d7c31b31464143baabafc987ff779c308` |
| `C1_occ0_full` | y | `33bea66ece8108a06b4cdeb3409d20400b62794e0d13857cccb6ed3510eaec73` | `6ea68012a8ee9dab118913d2d07368017ed37852728778dbcdb588da11856c33` |
| `C2_occ127_full` | A | `ee203d35f31437865ba9dd311db8182a6d3d0438471133eba2ee56b4f1c25bbb` | `22948db30c1dd77bbe2ebfe94a0669a891cd334fa78cc9614c8e9d81d079a2c1` |
| `C2_occ127_full` | x | `e3d5e5cdf8e3e2ad1fca72c26b6422449598be83e7c153429848d05d0a714100` | `458e1916e82b6bb917ffc332052fbcab5b6578979c2752cec84a1fa34fb0e31a` |
| `C2_occ127_full` | y | `a6ca8432d95c2c99674b43244a7eb6670fdfaa69cf0ef974b3b8425b98aeeefc` | `e5bb54175a7c43e735a8778afd6cd8a780017b192c467d6fc4e160fda166c5df` |
| `C3_odd_m5119_k5888` | A | `4b55bc387c5682bc3b4ed88e7aaa13b2d4978fcfe66af6be093d8e010c5e9b31` | `0c64acb8997269d45fd1a5b0da9b556c0f74a0df04f3e5266d47944312595e01` |
| `C3_odd_m5119_k5888` | x | `a0aff443a398c7217d400deb8ea5fc02b1744d6c5f4fedb058eb51491f536c55` | `85aa9791745e2fec562c3009f86c60c65a3f0ba3c0a10484b39ea2e450c24b7b` |
| `C3_odd_m5119_k5888` | y | `29db2d8a14bce4fad2dd4b5ec2a540a80177686f9acd71f0366f95b9245db1a6` | `2b84bfecc64bffccd32d476dd241fc63d690783d8b515fc1dde5f9b40dbb1012` |
| `C4_odd_m5119_k17152` | A | `fa5e77962b875fc792912c3ac83da0b87fbe33f9e376d9dc6dd47165d55c0c4b` | `259e91812289a8652934e9ce0a6d9e205a3b36c0e3930460bd747d095f89193b` |
| `C4_odd_m5119_k17152` | x | `fde65a7ef0fd661fc5bc53aac0f5a7bc86979ceea3f14dd02681ceeb56e22231` | `6a38a9f23e0138aecafa5f9ec2a12415c1fb978c72e61a77ec5bb0d938e92eb0` |
| `C4_odd_m5119_k17152` | y | `dbab410b135081a905360076f273f2e64738aae988b80e1807a0b96b983b939d` | `0767835f063c2a969075cfc79148fe5f21f0fffd736126e2652d8f5b347191a1` |

Kernarg templates (pointer slots zero, hex / sha256):
- `C1_occ0_full`: `0000000000000000000000000000000000000000000000000014000000180000` / `29cde049d8c0034138760e96e252bea5bf8092dd6e4e431e990979e1122fc3a2`
- `C2_occ127_full`: `0000000000000000000000000000000000000000000000000014000000440000` / `5de2801046e49faf9beb785c1d9fa65ee84d1c3e332ddd785b4e07484de7148a`
- `C3_odd_m5119_k5888`: `000000000000000000000000000000000000000000000000ff13000000170000` / `54fa1b74b91443848b712be4bb56dbc0bb7279513d51809b0be9e07570bfef75`
- `C4_odd_m5119_k17152`: `000000000000000000000000000000000000000000000000ff13000000430000` / `67a9a77e55e0ebdce4a1be936c87e9653220b390183c99629ae58910cc6bf92f`

Corpus fixture digest: `fixtures_sha256 = b44ee95d2bc5588cfb38f463dac9f9ec9dcc212d9be66fdf87ab8126800587ef` = sha256 of the concatenated lines `<cell_id>/<buf>.pre <payload_bytes> <payload_sha256>\n` in cell order C1..C4, buffer order A,x,y (lines are listed in `corpus.json` `recipe.fixtures_sha256_manifest_lines`).

### 2.3 Framing, guards, independent restored preimages

Every pointer argument gets its own allocation framed by 4096 guard bytes of `0xA5` on each side, padded with `0xA5` to the device VMM granule (granularity is a hardware property recorded in each receipt as `vmm_granularity_bytes`, not frozen here); the kernel receives `frame_base + 4096` (`pm_aco_oracle.rs:25-28,66-67,395-399,702-734`). Each arm x mode x cell x repeat run allocates fresh frames and uploads the pristine pre-image frames; no run reads another run's output, and graph replays reset all frames before every launch (`:793-818`). The uploaded frames' sha256 must equal the framed digests above before launch.

## 3. Supported finite domain (and what is not promoted)

Scanned on the full fixtures (CPU): x f32, y-pre f32 and all four f16 header halves (scale/zero, lane<16 and lane>=16) of every group contain **zero** non-finite values, zero non-zero subnormals and zero negative zeros (positive zeros: 33 in occ0 y-pre only); nibbles span 0..15. Ranges: occ0 |x| <= 0.0963, y-pre in [-0.0659, 0.0566], scale in [0.00157, 0.2374], zero in [-1.8994, -0.01259]; occ127 |x| <= 6.5194, y-pre in [-50.53, 305.59], scale in [0.00167, 0.04453], zero in [-0.3562, -0.01337]. Full scan record: `corpus.json` `supported_domain.observed_scan`. C3/C4 are strict byte subsets of these.

The supported domain is exactly the fixture bytes of C1-C4 plus the island preimage pairing of section 7. A finite oracle is not a universal proof. **Not promoted** (neither measured nor promotable under this corpus id): NaN/Inf in any input; synthetic, random or constructed tensors; subnormal / signed-zero / scale-extreme cases not present in the scan; intermediate overflow; K not a multiple of 256; M/K other than the four cells; tail groups 1 and 2; other qtypes; gfx1100/gfx1151 (separate cells); other drivers. Extending the domain requires a new corpus id.

## 4. Four arms and envelope requirements

Identical logical inputs with backend-correct pointers (plan `PM-UNION.md:189-194`):

| Arm | Role | Exists now | Pinned now |
|---|---|---|---|
| V0 | original SPIR-V pipeline under the producing Vulkan driver | SPIR-V and ACO binary exported (4.1); no executed reference result | GLSL source (section 8); exported SPIR-V, binary, sidecar, driver build (4.1) |
| Hpm | ACO-origin PM raw lift + HSA adapter through HIP | no | none |
| Vpm | PM-certified original Vulkan binary replay (private-import candidates are a later separate arm with their own envelope namespace) | no | none |
| Hllvm | tuned hipcc incumbent | yes | registry snapshot source, `.hsaco`, inventory device ELF digest |
| Hbuilder | accepted builder twin, additional incumbent | yes | `.co`, `.hxaco`, source |

Claims stay separate (none run): K1 `V0 == Hpm`; K2 `V0 == Vpm`; K3 `Hpm == Hllvm` and `Vpm == Hllvm` (Hbuilder as additional incumbent); K4 eager == graph == retained == Vulkan replay == mixed island per mode claimed; K5 island dispatch == singleton dispatch. Comparison is raw byte equality, zero differing bytes, no tolerance, on byte arrays (NaN payloads and -0 cannot hide). If V0 matches Hpm/Vpm but not hipcc, the result is faithful import, not exact replacement (`PM-UNION.md:206`).

Every arm result record MUST carry (machine-readable list in `corpus.json` `arm_envelope_requirements.fields`): arm id; artifact kind; producer kind/build/options; source hashes; arch, wave, block; kernarg ABI hash; raw executable/data/config/entry hash; static certificate and ABI-adapter digests; envelope kind; blob hash; compatibility/global key; loaded-executable binding; corpus id, `fixtures_sha256`, cell id; execution mode; state-surface digest; paired receipt ids; device UUID; driver/runtime build; negative-control results; differing bytes. Fields are real digests or the arm is not runnable; absent/mismatching fields are refused, never inferred, and a baseline fallback cannot satisfy a candidate receipt (`PM-UNION.md:179`). The generic `OracleReceipt` (`H/crates/radiowave/src/lib.rs:480-487`) alone does not identify this matrix. `corpus.json` `arms[*].required_before_use` lists, per arm, the identities not yet existing.


### 4.1 Observed pilot export (read from actual files; export only)

Directory `<release-workdir>/pm-aco-pilot/artifacts/gemv_mq4g256v2_residual/` (files dated 2026-10-11 07:27, pilot-owned, `artifact.json` schema `pm-aco-artifact-v1`). Status: **exported; no lift, no equality, no replay proof; no `aco-vulkan-ref-v1` receipt and no `aco-pm-candidate-v1` manifest exist.** Every sha256 below was produced by `sha256sum` and agrees with the pilot's own `artifact.json` file table.

| File | Bytes | sha256 |
|---|---|---|
| `artifact.json` | 13425 | `79214366483647be3b79700cdf9e02ac9e92c9e8a86ae835311eafdc65168cbd` |
| `shader.spv` | 18220 | `a551573300dc9db3771ebe9417d8503fbd1e720942c4835ee7cd0cc2a618e0fc` |
| `shader.bin` | 5476 | `ff89b0579505459c566330ab4b259c293a8c825e324648d57a80fced81372670` |
| `shader.abi.json` | 7516 | `ffcce5e67ddf4435fc1ffc34719e3d544b1136d37f5a0bfd12883cdc8010f7dd` |
| `shader.rawspec.json` | 1548 | `b785dae525c27dadb5b3bd7cd664a9c77eb00525d20ce8b9a0d2aeb69a163413` |
| `shader.aco.dump` | 38085 | `1315fbca1e7d2ad5b174ee120c88a6f2ae45b6eaa61c7d946eb8a69439c575b2` |
| `capture/capture.json` | 6492 | `975ce74211e0f632e07d39a1699c78fc0eb941a503feb014fd0c4ee4242938ef` |
| `capture/identity.json` | 446 | `8bfcdba291d95458de7cf49df6bce40b5185469da92b9b7faf7d38a2a14a5ef1` |
| `capture/aco.disasm.txt` | 67214 | `b841a8492aabf9c7495dc7dd1f7bd2a905c1aac93d60e501daa0379080b657a4` |
| `capture/final.nir.txt` | 54146 | `bab2ff5d8532e498a0ac00d53f0d127a4cf1ac4d8042eaf14957545d808bb344` |
| `capture/host-stats.txt` | 477 | `ee131477bcb39587e5f1e1297d7ebdbae2ea3e5fc2fea5f08a5418caaedec83e` |
| `capture/host.stderr.txt` | 746 | `4a7a9935ff78436aba3ae10a027d5c782c893967c3cee4465b8999c6f742ef7a` |

- **Binary** `shader.bin` (5476 B): executable range offset 0, length 5456, sha256 `d860d885a18e49b748fbb4659735024d1f04fa090da6fb68cf25ecf97374a680`; trailing range offset 5456, length 20 (five `0x9fbf0000` words), sha256 `9754d4a15268314011364b6b4844cd5feab9e8c38efff00feba5290d39249f72`. Both range hashes were recomputed on the file bytes. No relocations, no symbols. This is the ACO callback code binary, **not** a RADV-serialized pipeline binary: the export holds no pipeline key, global key or `vkGetPipelineBinaryData` blob, so the Vpm blob/key identity is still unfulfilled.
- **Rawspec** `shader.rawspec.json` (`pm-raw-kernel-spec-v1`): binds `shader.bin` sha256 `ff89b0579505459c566330ab4b259c293a8c825e324648d57a80fced81372670`, exec range 0..5456 sha256 `d860d885a18e49b748fbb4659735024d1f04fa090da6fb68cf25ecf97374a680`, sidecar `shader.abi.json` sha256 `ffcce5e67ddf4435fc1ffc34719e3d544b1136d37f5a0bfd12883cdc8010f7dd`, 9 entry-fact records, wave 32, target gfx1201 (sramecc/xnack any).
- **ABI sidecar** `shader.abi.json` (`pm-aco-abi-v1`): push block A@0, x@8, y@16 (u64), M@24, K@28 (i32), matching the corpus kernarg ABI; it states ttmp8 and other undeclared ttmp reads are not declared and that the lift rejects an undefined read.
- **SPIR-V** `shader.spv` (18220 B) sha256 `a551573300dc9db3771ebe9417d8503fbd1e720942c4835ee7cd0cc2a618e0fc`, built by glslang (`bfcc2479c85bdab08c7f7e20425f35d8b9aad176ab5fbbbad3427166c7a259a5`) with `-V --target-env vulkan1.3 -S comp -e main` from the GLSL source whose sha256 `9d8b98b85de1b42d6836fa07949f40b071c1b2846c9dd69aaa7302c129ab90cb` equals the section 8 `.comp` pin (matches: True); float controls DenormPreserve 32 and SignedZeroInfNanPreserve 32; BDA addressing, no descriptor sets. Pipeline: layout sha256 `990d3f894605cb220cf98b0e54ff61e06463bad48df1f8ffb267cc5c913d1b37`, specialization (none) sha256 `6d1cd599435b59d038f9ad163f55470b4c77cb17bb15c8aac52ab96daabe5558`, push range 0..32 COMPUTE, local size 32x1x1, require-full-subgroups with required size 32.
- **Config**: wave 32, exec size 5456, code size 5476, VGPRs 96, SGPRs 108 (actual 107), LDS 0, scratch 0, rsrc1 `0x200f000b`, rsrc2 `0x00000094`, rsrc3 `0x00000000`, float_mode 240.
- **Compiler/driver**: RADV with ACO (no LLVM in the compiler; capture exists only because the ACO callback ran), exporter `aco-export` 0.1.0, driver library `<release-workdir>/pm-aco-pilot/mesa-build/src/amd/vulkan/libvulkan_radeon.so` sha256 `65bd3163bf8b1351237d59e5fc451b7a69cb24efab4a8861075a46469ec18118` (re-verified), ICD json sha256 `b6230d88bdf42e25332ac7fb4dacb17c54b93099040776a969800a67e49885b1`, host `<release-workdir>/pm-aco-pilot/bin/vk_aco_host` sha256 `f82dde11c4568afec55467d8599879a372f2fec7a2bb5bfe8833781836288021`. The tree is Mesa `mesa-26.0.8` `60e95b78` **plus an export-hook patch** `tools/aco-export/mesa/0001-radv-aco-export-hook.patch` sha256 `c258fbc927941534af7a444d41173aa97ae330b5d130156a0913e09e039812ee`; the source pins of section 8 are the unpatched files. Env: `MESA_SHADER_CACHE_DISABLE=true`, `RADV_DEBUG=nocache`. Disassembly (`shader.aco.dump`, `capture/aco.disasm.txt`) used ROCm 10.1 llvm-mc `9ff7bd4cf95c171098c2c2f616b65152cb94812a11e86106d3afd0910a37f4cf`.
- **Device (compile only, not an execution receipt)**: AMD Radeon AI PRO R9700 (RADV GFX1201), PCI `<pci-bdf>` (selected by BDF; host enumerated 4 devices), device UUID `<device-uuid>`, driver UUID `414d442d4d4553412d44525600000000`, `Mesa 26.0.8 (git-60e95b7878)`, API 1.4.335, subgroup sizes 32-64.
- **Driver statistics** (`capture/host-stats.txt`, as printed; no performance claim): pipeline hash 754289669937981336, 985 instructions, VALU 614, SALU 124, VMEM 32, SMEM 14, VOPD 39.
- **Cross-check against the pilot's prepared inputs**: `<release-workdir>/pm-aco-pilot/prepare/prepare.json` (sha256 `86f1787d52c5e6a93782e51dc9e8f1b0287e96c08eb964a9bc369ac09c20fbc4`) lists framed pre-image digests for `fixture_m5120_k6144` and `odd_tail_m5119_k5888` (captured and minimal grid); all full 64-hex digests equal this corpus's C1 and C3 framed digests (section 2.2). The occ127 cells (C2, C4) were not prepared by the pilot at observation time.

Still absent for other arms: the Vulkan reference receipt (framed post buffers) for any cell, a candidate manifest/PM-HIP object (Hpm), a RADV pipeline blob with pipeline/global key (Vpm), raw round-trip, static certificate, adapter digest, loaded-executable binding.



## 5. Surface lists

- **Output**: y payload, frame bytes `[4096, 4096+4M)` (read-modify-write; its pre-image participates).
- **Input (must be unchanged)**: A payload, x payload, kernarg blob (32 bytes padded to 16).
- **Guards**: front 4096 B, back 4096 B and granule tail of each of the A, x, y frames, all `0xA5`. For odd M the last workgroup stores only `y[M-1]`; the first back-guard byte must stay `0xA5`.
- **Full state**: whole mapped A, x, y frames and kernarg blob (island: shared A/x frames once, y1, y2, both kernarg blobs). Reference: `pm_aco_oracle.rs:743-766`; methodology `H/crates/rdna-compute/src/railgun_check.rs:6-38`.
- **Model state**: none. No KV, recurrent, conv, PLE, RNG, token or cursor state exists in this corpus; it claims nothing about them.

## 6. Negative controls, repeat sequence, execution modes

Negative controls (each is a single deliberate change the comparison MUST report in exactly that surface; a miss invalidates the receipt; reference `pm_aco_oracle.rs:889-899,1000-1073`): y[0] low bit; y[M-1] sign bit; unchanged A byte; unchanged x byte; y front-guard byte; A back-guard last byte (all cells). Wrong inputs (C1/C2; adapted to last row/last group on C3/C4): A byte 8 ^0x10, row 0 group 0 nibbles inverted, row M/2 group G/2 inverted, last row last group inverted, x byte 0 ^0x10, x last byte ^0x10. Cross-arm: tampered reference y[1] vs candidate. Before promotion (owned by U1/U-F2/U-F4 milestones, frozen here as required): swapped M/K, wrong pointer order, wrong artifact digest, incompatible global/pipeline key, disabled candidate not executing nor falling back under a candidate receipt, stale/old-epoch preimage.

Repeat sequence S1: `C1_occ0_full -> C2_occ127_full -> C1_occ0_full` in one process, per arm and per mode; the loaded executable is created once and reused (no reload/recompile), kernargs rewritten per launch, fresh restored preimages every launch. Launch 3 must be byte-identical to launch 1 and each launch to its own cell reference/singleton. No odd-cell sequence is frozen.

Execution modes are admitted **separately** (none is admitted by U0): `hip_eager` (`pm_aco_oracle.rs:780-788`), `hip_graph_capture_replay_x2` (`:793-818`), `railgun_retained` (only if claimed; `railgun_check.rs:6-38`), `vulkan_cmdbuf_replay`, `mixed_host_fenced_island` (no semaphore/async claim). Passing one mode never admits another; graph/retained/mixed results compare to eager for the same cell.

## 7. Independent two-dispatch island (plan only)

Island `I1_two_dispatch_duplicate_kernel`, geometry of C1: one frozen arm artifact loaded twice into two independent executable handles D1/D2 (identical bytes, separate module/pipeline instances, distinct kernarg blobs). Shared **immutable** A and x (C1 payloads, single frames, hashed whole before D1 and after D2, must be byte-identical). **Disjoint** y1 (preimage occ0 `arg16.pre.bin`, `{c['island']['disjoint']['y1']['sha256']}`) and y2 (preimage occ127 `arg16.pre.bin`, M=5120, length 20480, `{c['island']['disjoint']['y2']['sha256']}`), each with its own guard frame and mapped tail; D2's preimage is never D1's output. Host-fenced between dispatches; both orders D1->D2 and D2->D1 must give identical bytes. Expected: y1/y2 post equal the same-arm/same-mode singleton results for their own preimages; A/x/guards unchanged. Island negative controls: flipped y1 post byte; swapped y pointers; D2 omitted; D1 rerun without restoring y1; A/x frame-hash change. Non-goals: no tensor/model-state transfer, no VMM/semaphore interop (U-F3), no routing/scheduling (U-F4), no concurrency claim. This is a plan; no GPU result exists.

## 8. Actual source and template pins (`sha256sum`)

| Role | Path | sha256 |
|---|---|---|
| hipcc incumbent source (compile input of the pinned object: weight-buffer preamble + kernel) | `<release-workdir>/parallel-packs/registry/gfx1201/gemv_hfq4g256_residual_mq4v2.hip` | `869f4a4bdd33adc3e5a3aa9a6faa7584b53d50b0930f1d9e737e19a77975dc0e` |
| tree kernel source (kernel body only; text differs from registry snapshot by the 55-line preamble) | `kernels/src/gemv_mq4g256v2_residual.hip` | `be1bc095db85d98b9665421093e0c06cd0b320610e4a58100e8f6c82d14094c2` |
| hipcc incumbent object (Hllvm) | `<release-workdir>/jit-peacemaker/hipcc/gfx1201/gemv_hfq4g256_residual_mq4v2/rep-1/kernel.hsaco` | `bd87c3e5a11ae57644a3e462af3398adea5da973cfc95789230d0e6d79b742ca` |
| hipcc incumbent build command | `<release-workdir>/jit-peacemaker/hipcc/gfx1201/gemv_hfq4g256_residual_mq4v2/rep-1/command.json` | `a0d1d53faef63691a0e00312e7360082fd57d2de60551381c4dc1edbbb78b19d` |
| decode-twin inventory | `<release-workdir>/pm-decode-twins/inventory.json` | `b17f6b13e0c740f40fd9a3090fa3e84603d378afb622671eb487583a70f85f10` |
| accepted builder twin source (additional incumbent) | `crates/peacemaker-isa/src/kernels/pm_decode/gemv_hfq4g256_residual_mq4v2.rs` | `ebdbde12690c2542a5862d9fa1d5513cfa9193cc7364fed8ecc12e452596743d` |
| accepted builder twin asm | `kernels/pm-decode/gfx1201/gemv_hfq4g256_residual_mq4v2.s` | `463cad60ff98ae9eec7ec9d52632745cd8d16acef672b057029cf8c7b42b28dd` |
| accepted builder twin code object | `kernels/pm-decode/gfx1201/gemv_hfq4g256_residual_mq4v2.co` | `1214242d3c152aa7919182150c96c4b1033aadc312731171679948c50658c3f0` |
| accepted builder twin hxaco | `kernels/pm-decode/gfx1201/gemv_hfq4g256_residual_mq4v2.hxaco` | `7c98a6a0071b28817224deeec6a28ffac1aeccab2286382377902e47e8df7f2d` |
| accepted builder twin radiowave record | `kernels/pm-decode/gfx1201/gemv_hfq4g256_residual_mq4v2.radiowave.json` | `8ae44ce1fa2fef8a1cd9b3ee60c9b5b521b31951afa3d579d1eee26381f2a413` |
| accepted builder twin shape record | `kernels/pm-decode/gfx1201/gemv_hfq4g256_residual_mq4v2.shape.json` | `16dd862bd1ba296f48baaea43c97d90a003a326851237362cad45134e6f23847` |
| accepted builder twin proof record | `kernels/pm-decode/gfx1201/gemv_hfq4g256_residual_mq4v2.proof.json` | `7841824b22f266904466041d1b8a2317358717baea2630dd60a55d894ea78cb4` |
| accepted builder twin m7 record | `kernels/pm-decode/gfx1201/gemv_hfq4g256_residual_mq4v2.m7.json` | `8e036da174efbc71e40468ba2b292a7711300ef2402536611ff87fd32650ba41` |
| SPIR-V source twin template (V0/Hpm/Vpm producer input; pilot-owned, uncommitted working tree) | `<release-workdir>/worktrees/pm-aco-pilot-20261011/crates/peacemaker-lift/tests/fixtures/aco/src/gemv_mq4g256v2_residual.comp` | `9d8b98b85de1b42d6836fa07949f40b071c1b2846c9dd69aaa7302c129ab90cb` |
| pilot four-arm oracle harness (pilot-owned, uncommitted working tree) | `<release-workdir>/worktrees/pm-aco-pilot-20261011/crates/rdna-compute/examples/pm_aco_oracle.rs` | `0b5e98bd947d1f486841f6b81328ffcaf83590b7e925868cc5cb1fd232fd7ead` |
| pilot Vulkan reference receipt consumer (pilot-owned, uncommitted working tree) | `<release-workdir>/worktrees/pm-aco-pilot-20261011/tools/aco-export/src/reference.rs` | `c642bef439335978f8a1c7aa6c1943174be24f66e95de963cb1be6a0f6a10125` |
| Mesa pin record | `<peacemaker-workdir>/aco-shim/pin.json` | `c6c6942f845f75a306d8370d4864339113f2c41283d9ef70bcb1405f46fdeab7` |
| pinned Mesa file (also recorded in pin.json files_sha256) | `<peacemaker-workdir>/mesa/src/amd/compiler/aco_opcodes.py` | `d0ac25b0b52185c807b32acb751a4d96be9d5792e1ec4773835b64b191010e7e` |
| pinned Mesa file (also recorded in pin.json files_sha256) | `<peacemaker-workdir>/mesa/src/amd/compiler/aco_assembler.cpp` | `a2a35b4716206714ba0518e23985d7d51bad8d884c4106c969f0b23d7122d784` |
| pinned Mesa file (also recorded in pin.json files_sha256) | `<peacemaker-workdir>/mesa/src/amd/compiler/aco_insert_waitcnt.cpp` | `e84a9fb537fc05983b0bc5ce4282c2dc2b6fa0e2494ab9e6cd841abdc021ef87` |
| pinned Mesa plan source (tracked, unmodified vs commit 60e95b78 per git status at freeze) | `<peacemaker-workdir>/mesa/src/amd/compiler/aco_interface.cpp` | `a19f1d12a9824cd39ba5a297c60fe3191dd3aa848b7117d566ddd5346d2dd428` |
| pinned Mesa plan source (tracked, unmodified vs commit 60e95b78 per git status at freeze) | `<peacemaker-workdir>/mesa/src/amd/vulkan/radv_pipeline_compute.c` | `5fd82d7674abc36bc740e5049a95d6aaaf4e2ade41fd5ea12b2cc9d691662052` |
| pinned Mesa plan source (tracked, unmodified vs commit 60e95b78 per git status at freeze) | `<peacemaker-workdir>/mesa/src/amd/vulkan/radv_shader.c` | `db5b23f445c9229535537b3fedfbdd3550c9d0720058ff53de7be03e0f856706` |
| pinned Mesa plan source (tracked, unmodified vs commit 60e95b78 per git status at freeze) | `<peacemaker-workdir>/mesa/src/amd/vulkan/radv_pipeline_binary.c` | `81c652fe0613b0e567a86f9113ef8bec99891da899842c5d43447f5d76271599` |
| pinned Mesa plan source (tracked, unmodified vs commit 60e95b78 per git status at freeze) | `<peacemaker-workdir>/mesa/src/amd/vulkan/radv_pipeline_cache.c` | `9dff38382af8be6f4f9c393c82b2d7282d0f53674a1cb4848a4641d007cb1b1f` |
| pinned Mesa plan source (tracked, unmodified vs commit 60e95b78 per git status at freeze) | `<peacemaker-workdir>/mesa/src/amd/vulkan/radv_shader_args.c` | `e9165b77f105f19192dba05fa478c069de01556d4bb658a89c02b0ed196e39a7` |
| pinned Mesa plan source (tracked, unmodified vs commit 60e95b78 per git status at freeze) | `<peacemaker-workdir>/mesa/src/amd/vulkan/radv_device_memory.c` | `e3d12ae5bbf93073e4cfc732a153d591b9e64c1c34c6c6f7df4eae1c2cb0e053` |
| pinned Mesa plan source (tracked, unmodified vs commit 60e95b78 per git status at freeze) | `<peacemaker-workdir>/mesa/src/amd/vulkan/radv_physical_device.c` | `9a4fbe36d5426aabb967410a06cb05a1f7387465a8917c72c09cb8def097d26d` |
| H source: whole-state snapshot/twin/restore + negative-control methodology | `crates/rdna-compute/src/railgun_check.rs` | `93f44ce2d6790d9520cc06c16f49371cd41805569d263d9ee9e27a3a5dbd2af3` |
| H source: PeacemakerRecord/OracleReceipt vocabulary | `crates/radiowave/src/lib.rs` | `302b8fbfe1390da0af10b0aec74600609aa2bf76e0466e0c15f62e477d552d1f` |
| H source: ReplayQuiescence | `crates/rdna-compute/src/replay.rs` | `2711fd12c1c5daa4478345bb443518bb03344056084f131d3b83ecdbac7dab4b` |
| H source: launch_maybe_blob_bound capture boundary | `crates/rdna-compute/src/dispatch.rs` | `bc27840992fb02b05e4b3550797dce797ead42cbb45af428077f77085c013e48` |
| H source: load_kernel_module/load_embedded_module funnel | `crates/rdna-compute/src/scratch.rs` | `2830287ff2144e4f90bc74f2aaa0683089c57e99fa6bf2aff5ad7691b53ce769` |
| H source: m7 static check | `crates/peacemaker-isa/src/pm_check.rs` | `33b12c3d37a7b2d4715f7a177273259e5ce1609c49cf31d55eec6fd90e2a04e2` |

**Code-end tail fill (0.4.3): the builder-twin rows above (`.s`, `.co`, `.hxaco`, `.radiowave.json`, `.proof.json`) are the regenerated objects the 0.4.3 tree ships, not the objects this corpus was frozen against on 2026-10-11.** The freeze-time digests were `.s` `b320c9d07cebea702fc4bb206d3637505b363de9826fc739ce05ec16979c991f`, `.co` `bca8ec25b5fdf77147df2ac7a0b925d688fc550ce50090ba0bd8d20616585c7d`, `.hxaco` `8e2cb8e6bbefeca5d996d274e06e37532140cf42ca65606a565404f6e3790757`, radiowave record `9d5924ff7e7699493c48d610c0d9ce23bbf496001f5591a9081195a8de02bacf`, proof record `1b71bd5d8f82ce3bd74e47bb8cd67d7f45be951b367c8dfaba8f36b6e6b23a98`; the shape record (`16dd862b...`) and m7 record (`8e036da1...`) are byte-identical and unchanged. The regeneration (builder `19fc7e2c`) changed only the code end: `.s` gains a trailing `.text` block (`.p2alignl 7, 3214868480` then `.fill 96, 4, 3214868480`, the `s_code_end` word), and the shipped `.text` is the freeze-time `.text` followed by those words (4296 -> 4736 bytes). The kernel body, descriptors, symbols and metadata are unchanged; only the added tail, the ELF layout fields derived from the new `.text` size (section addresses/offsets, `.relro_padding`) and the `_DYNAMIC` symbol value in `.symtab` differ. Equivalence is a byte-level fact about the two objects, not a new measurement: **no GPU hardware oracle, timing or Vulkan measurement was repeated on the regenerated objects, and none is claimed.** The 0.4.3 CPU compiler, static, emulator and RIP oracle checks that were run on the new objects are not hardware remeasurement. The oracle acceptance, perf and timing records under `kernels/pm-decode/gfx1201/` (`*.acceptance.json`, `*.oracle.json`, `*.timing*.json`, `*.perf-*.json`, `*.receipt.json`, `*.g0.*`) and every 2026-10-11 pilot result remain bound to the freeze-time objects by their recorded digests; `crates/rdna-compute/src/pm_decode_twins.rs` pins the regenerated image and ELF and lists the freeze-time pairs.

Pilot-owned files (comp, oracle, reference.rs) are uncommitted working-tree files at freeze: consumers must refuse a different digest rather than follow the live file. Mesa tree has local modifications at freeze (`src/amd/compiler/meson.build`, untracked `aco-shim`); none of the pinned Mesa files above is modified. The three `pin.json` entries match `pin.json:5-9`.

## 9. Citations to inspected source

- Pilot oracle `P/crates/rdna-compute/examples/pm_aco_oracle.rs`: purpose/modes `:4-47`, constants/ABI `:62-76`, length checks `:215-219`, shapes `:285-383`, framing `:395-399,702-734`, guard readback `:743-766`, eager `:780-788`, graph `:793-818`, controls `:889-899,1000-1073`.
- Radiowave `H/crates/radiowave/src/lib.rs:429-438,480-487`; `H/crates/rdna-compute/src/replay.rs:41-50`; `dispatch.rs:3021`; `scratch.rs:564,615`; `H/crates/hipfire-isa/src/pm_check.rs:35`.
- Inventory `Q/pm-decode-twins/inventory.json:43136` (module entry), `:43159` source sha, `:43194` object sha, `:43195` device ELF sha.
- Pinned Mesa `60e95b78`: `src/amd/compiler/aco_interface.cpp:102-174,247-270`; `src/amd/vulkan/radv_pipeline_compute.c:93-132,248-307`; `radv_shader.c:3284-3325`; `radv_pipeline_binary.c:73-96,130-156,160-181`; `radv_pipeline_cache.c:66-81,95-119`; `radv_shader_args.c:68-111,540-595`; `radv_device_memory.c:173-182`; `radv_physical_device.c:1739-1747`. These identify the producing/import/ABI boundaries each arm's identity record must bind; line numbers describe the inspected tree.

## 10. Limits

This corpus freezes inputs, surfaces, controls and identity requirements. It does not show that any arm is exact, that ACO/Vulkan artifacts exist, or that the island runs; it does not admit any execution mode. Estimates and results are for later milestones (ACO pilot, U1, U3). The observed export is not evidence of lift, equality or replay; the export was built by a patched Mesa (hook patch), so the Mesa plan-source line numbers in section 9 refer to the unpatched pinned files and may differ in the patched tree, and the V0 reference arm must be executed on the exact exported driver build; the pilot files pinned in section 8 remain the frozen snapshot and were not re-pinned.

