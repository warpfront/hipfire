# First-request JIT inventory: Flash-Next, 27B and A3B artifacts x gfx1201/gfx1100/gfx1151 (baseline)

Worktree `/home/kaden/qcal/release-0.4.3/first-request-jit` at `3866d41b18df` (branch `first-request-jit`), generated 2026-10-18T18:20:32Z. Machine-readable twin: `/home/kaden/qcal/release-0.4.3/first-request-jit-evidence/inventory.json` (same data, full per-artifact records).

This is a **baseline** report. It states what the checked-in source, plans and fixtures establish today. Measured after-state results are appended below this document by the parent, one section per gate, each naming the artifact sha256, arch and commit; the baseline sections above are not rewritten. Nothing here was produced by a build, test, formatter or GPU run: facts are read from source/fixtures or from local file hashes recomputed for this report (no model payload was downloaded). Failures cited from earlier fixtures, and any failures supplied by the task owner, are observations and were not rerun.

## 1. Scope and method

- Source: `registry/v1.json` (sha256 `c70d3adcb927f85caa9026dccbfb64d32573a4b6ae194512493b6f7f73a70897`, generated 2026-10-17T01:19:29Z; byte-identical to `first-request-jit-evidence/registry-v1.json`). It is generated from `registry/models.json`.
- Selection (deterministic): `.models|to_entries|map(select(.key|test("flash-next|27b|35b-a3b")))|group_by([.value.repo,.value.file])` gives **51 tags in 47 distinct `repo/file` artifacts** (every group has one sha256 and one size). One file may carry several tags (Flash-Next base has 4, Qwen3.6-35B-A3B mq4p has 2).
- Classes: **32 core**, **8 drafts** (arch_id 20 DFlash), **7 derivatives** (Bonsai x2, Carnice x2, Qwopus x3; all dense 27B). Core = 18 Qwen3.5/3.6/3.8 27B dense files, 10 35B-A3B MoE files (Qwen3.5 x1, Qwen3.6 x7, **Ornith 1.5 x2**) and 4 Flash-Next files. Per Main, Ornith A3B is part of the core walks, not a derivative. Topology totals: 25 dense-27B (18 core + 7 derivative), 10 MoE-A3B, 4 Flash-Next, 8 draft files.
- Default rung = the file carries a family default tag (alias-resolved or unsuffixed base): `qwen3.5:27b`, `qwen3.5:35b-a3b` (legacy default), `qwen3.6:27b`, `qwen3.6:35b-a3b` (= mq4p), `ornith-1.5:35b-a3b`, `qwen3.8:27b`, `qwen3.8:flash-next` (= the base mq4-xts file, also tagged gptq3/mq4/mq4-xts); draft defaults `qwen3.5:27b-draft`, `qwen3.6:27b-draft`, `qwen3.8:27b-draft-mq4`; derivative defaults `carnice:27b`, `qwopus:27b`, `qwopus3.6:27b-coder`, `bonsai:27b-tq2`. 14 files.
- Walk scope (Main, verbatim): core = "registry DEFAULT tag of each model family plus its listed XTS/mq4r/mq2/mq3 rungs for Flash-Next, A3B (Qwen3.6 + Ornith) and the 27B". This report applies the predicate "file suffix contains `xts`, `mq4r`, `mq2` or `mq3`" to find the listed rungs; the parent owns the final list. Drafts are tested with their trunk; the 7 non-Ornith derivatives have no walk of their own and rely on the predictive fallback of their base family (itself unproven where the base has no walk).
  Result (core, total 32): default_family_tag 7, listed_variant 13, out_of_stated_walk_scope 12; tested_with_trunk 8; predictive_fallback 7. Hardware priority is the new A3B generation (Qwen3.6 + Ornith); the Qwen3.5 A3B legacy default stays in the inventory as core; the default 27B tags qwen3.5/.6/.8 are kept by the parent unless Main clarifies. `inventory.json` carries `hardware_priority` per row.
- Baseline status per arch uses four values: **W** walked plan (checked-in `route_plans/*.tsv`); **SX** static closure, exact (`qwen36_27b_route` applies and this artifact's file+header identity is proven against the frozen fixtures); **SC** static-selector candidate, unverified (the code path exists for model_arch 5 on gfx1201 but this artifact's header was never fingerprinted: **not a closure, unknown**); **N** none (lazy). A status is never inferred from a different artifact, and **SC rows must never be counted as closure achieved**.

## 2. The 47 distinct artifacts

Identity columns come straight from the registry (HF LFS sha256/size). Repo prefix is `hipfire-models/`.

| # | class | repo/file | bytes | sha256 | tags |
|---:|---|---|---:|---|---|
| 1 | derivative | bonsai-27b/bonsai-27b.bq1 | 3,795,196,928 | `83dc56409efa0503e171d91fffcad25c8a9f000120f61cf9cbba589a09fbbb07` | `bonsai:27b-bq1` |
| 2 | derivative | bonsai-27b/bonsai-27b.tq2 | 7,156,866,048 | `b2ea3f7f2160fe961d5ed256f474a4428b0f88e3a923e70fa80d3bf3bde5098e` | `bonsai:27b-tq2` |
| 3 | derivative | carnice-27b/carnice-27b.mq4 | 14,987,717,632 | `d82234a0755b973de48e62fa9dd531ac5ec407514d9728f97cee05c39dd7f191` | `carnice:27b` |
| 4 | derivative | carnice-27b/carnice-27b.mq6 | 21,393,697,792 | `1a96237e7b52024eced5215fd0eaca4334bc37e3dfd07c6b972cdef2fd86e92b` | `carnice:27b-mq6` |
| 5 | core | ornith-1.5-35b-a3b/ornith-1.5-35b-a3b.mq4 | 19,020,296,510 | `f329b497eb8684d889c184a0dcf1388ab6171f99a1c72052d7e2a3bee4f5934b` | `ornith-1.5:35b-a3b` |
| 6 | core | ornith-1.5-35b-a3b/ornith-1.5-35b-a3b.mq4r | 18,700,570,368 | `84103fcc8ade42aa2ac8ec01176df7a4ead5e94810597c9fae2f6763152a3ac6` | `ornith-1.5:35b-a3b-mq4r` |
| 7 | core | qwen3.5-27b/qwen3.5-27b.mq3 | 11,784,735,744 | `58fdaf54ec3c4be9372bd44b4754e198e7ff6d7129c6abdaec76c1ac70fb092b` | `qwen3.5:27b-mq3` |
| 8 | core | qwen3.5-27b/qwen3.5-27b.mq4 | 14,984,158,208 | `ea615949ddf6a180eee03ff6fde39f7e51148f153b1b05f82258b9953088576e` | `qwen3.5:27b` |
| 9 | core | qwen3.5-27b/qwen3.5-27b.mq6 | 21,385,292,800 | `08215e22b04dbe19706d9783c35e54ea92154090c0eca84aeaa71e2041eedd5a` | `qwen3.5:27b-mq6` |
| 10 | draft | qwen3.5-27b/qwen35-27b-dflash-mq3.hfq | 703,132,672 | `b3526f2f8ffce5a2483f6d74fd1e78af2b8eebe9249e5b55177ed0e0c58f53fc` | `qwen3.5:27b-draft-mq3` |
| 11 | draft | qwen3.5-27b/qwen35-27b-dflash-mq4.hfq | 919,401,472 | `3d428b97c1911a9ad815cc52fbee080306852c1dafad6b1b17bb70bd68010301` | `qwen3.5:27b-draft` |
| 12 | core | qwen3.5-35b-a3b/qwen3.5-35b-a3b.mq4 | 19,663,624,448 | `953c570cb242857c4de4d6118d5d532a06c3d8fd218e5b4d4b0835161189081e` | `qwen3.5:35b-a3b` |
| 13 | core | qwen3.6-27b/qwen3.6-27b.mq3 | 11,776,322,560 | `6b650a7d6cb4e2447dd67da7a87cb5e57956e98eff676a498a7586e3b3cc9298` | `qwen3.6:27b-mq3` |
| 14 | core | qwen3.6-27b/qwen3.6-27b.mq4 | 14,984,158,208 | `86a5f80fd29d545abb1093dead242725ced6d68b8607c6d566d897b1a82442dc` | `qwen3.6:27b` |
| 15 | draft | qwen3.6-27b/qwen36-27b-dflash-mq3.hfq | 703,132,672 | `f5fc4bef4c15229f940f8ead6c04723b04308321c4a9584c3318d364cadf1728` | `qwen3.6:27b-draft-mq3` |
| 16 | draft | qwen3.6-27b/qwen36-27b-dflash-mq4.hfq | 919,401,472 | `bd8c4f07ae80fe1385bf2606af9a7ba0daa18ca8daec50916f2a489054c44e70` | `qwen3.6:27b-draft` |
| 17 | core | qwen3.6-35b-a3b/qwen3.6-35b-a3b.mfp4 | 20,181,686,528 | `6bdade4e5602dccb012fce2dfddd7a27d169f08fd47c5c243d1cb06878fdc48a` | `qwen3.6:35b-a3b-mfp4` |
| 18 | core | qwen3.6-35b-a3b/qwen3.6-35b-a3b.mq2 | 11,610,560,768 | `48b3f84614c46eb8b5ffb494f7a75c15216664afcbb47c3e78dd80c4ce7eb0a3` | `qwen3.6:35b-a3b-mq2` |
| 19 | core | qwen3.6-35b-a3b/qwen3.6-35b-a3b.mq3p | 17,241,413,888 | `4ef8b5c5fcc37082e0b3c7496fcb980bb1dcabd21487bd7c8be16ac60e731953` | `qwen3.6:35b-a3b-mq3p` |
| 20 | core | qwen3.6-35b-a3b/qwen3.6-35b-a3b.mq4p | 19,757,996,288 | `8eb3be6912aa9db2dcc89c3233b05ccdfe81a84a8d6ef43202f2098d7f2fc78f` | `qwen3.6:35b-a3b`, `qwen3.6:35b-a3b-mq4p` |
| 21 | core | qwen3.6-35b-a3b/qwen3.6-35b-a3b.mq4r | 18,700,048,128 | `4685c140c46b1a6f31a0fd9053bf09d5faf1d2529d715b84794249b66cde0428` | `qwen3.6:35b-a3b-mq4r` |
| 22 | core | qwen3.6-35b-a3b/qwen3.6-35b-a3b.mq5 | 23,690,156,288 | `c7e168b8fc54b6a047cb8ea3890b2c944cebcf00f95ded10a58a00420dfdada9` | `qwen3.6:35b-a3b-mq5` |
| 23 | core | qwen3.6-35b-a3b/qwen3.6-35b-a3b.mq6 | 27,716,688,128 | `ae2a1240fc4a93c71ec6f75de4457e7132d932689b228de6a0de49c3f8fce1a6` | `qwen3.6:35b-a3b-mq6` |
| 24 | core | qwen3.8-27b/qwen3.8-27b.mq3 | 12,618,796,032 | `09c3544690aceca29e1822d79adab6ffcc8fd9e4b58359fe8dfb185ef49811c9` | `qwen3.8:27b-mq3` |
| 25 | core | qwen3.8-27b/qwen3.8-27b.mq3-pro | 13,184,433,152 | `394c50966bf4f68172df8eb34cd7ded8f9d0576c9ef24ea6ba639a88c184f795` | `qwen3.8:27b-mq3-pro` |
| 26 | core | qwen3.8-27b/qwen3.8-27b.mq3-xt | 11,777,616,896 | `3e04fc8db80bda557b965ec60ac876cf2500fced7f340624f3fcbeae134af5c5` | `qwen3.8:27b-mq3-xt` |
| 27 | core | qwen3.8-27b/qwen3.8-27b.mq4 | 15,662,615,552 | `5bb556a6cc84035234995c017c9791aa3951ad1eae4cf8c8172b0eaef399e507` | `qwen3.8:27b` |
| 28 | core | qwen3.8-27b/qwen3.8-27b.mq4-pro | 16,464,182,272 | `e6f2ac87042b9e314c323f00bc499a6304cd5624021ae501ce90b12a3a7ea3fa` | `qwen3.8:27b-mq4-pro` |
| 29 | core | qwen3.8-27b/qwen3.8-27b.mq4-xt | 14,987,185,152 | `80e7c624424fd1d363ba86681d3dc1e5ac5534e0e064306a32be204c4843d0f3` | `qwen3.8:27b-mq4-xt` |
| 30 | core | qwen3.8-27b/qwen3.8-27b.mq4-xts | 14,987,185,152 | `3e38ccbae3776470eb5a89344d300e9279d6b9ab6c31fd40ca1758c4f7c6f8ae` | `qwen3.8:27b-mq4-xts` |
| 31 | core | qwen3.8-27b/qwen3.8-27b.mq5 | 18,706,435,072 | `c018a0d7510bffbb3788844d5d8f72694464244e16f128b7e34804045324cf25` | `qwen3.8:27b-mq5` |
| 32 | core | qwen3.8-27b/qwen3.8-27b.mq5-pro | 19,319,258,112 | `7a46204f5ce16b260ebb028359a945c8c02f5abe87c040df063db782a99ee7cd` | `qwen3.8:27b-mq5-pro` |
| 33 | core | qwen3.8-27b/qwen3.8-27b.mq5-xt | 18,183,105,536 | `f4760c159f80d5d3ca237593a191dd553790cba0f89de59827f430b23109b39b` | `qwen3.8:27b-mq5-xt` |
| 34 | core | qwen3.8-27b/qwen3.8-27b.mq6 | 21,750,254,592 | `b798ea1166fc03a568f6daf8090b20b7d6314af7429951c03d86540385568db8` | `qwen3.8:27b-mq6` |
| 35 | core | qwen3.8-27b/qwen3.8-27b.mq6-pro | 22,174,333,952 | `58ac3ee645ede2bad2f6f833db1d4abbfdc1850047fdadef036d35a023bb9401` | `qwen3.8:27b-mq6-pro` |
| 36 | core | qwen3.8-27b/qwen3.8-27b.mq6-xt | 21,385,849,856 | `9d472ddc5b4e11a1986bfc83c54c0dac979a8e1d7186613dd7c7436f69ec8b2f` | `qwen3.8:27b-mq6-xt` |
| 37 | draft | qwen3.8-27b/qwen38-27b-dflash-mq3.hfq | 984,978,432 | `dd335fb2634c371d120077c2fe754643c91b27f32d0c32668278fe4b8f1ca319` | `qwen3.8:27b-draft-mq3` |
| 38 | draft | qwen3.8-27b/qwen38-27b-dflash-mq4.hfq | 1,209,603,072 | `d0a74a232a0e2166d889f823e91e0fbf778d21dd9668d7de055cdecb065401bc` | `qwen3.8:27b-draft-mq4` |
| 39 | draft | qwen3.8-27b/qwen38-27b-dflash-mq5.hfq | 1,434,227,712 | `8a8d3daeaa3788743ef9aedfa1a6cd9961395ecbd88e9b5e2eb981e0506a861f` | `qwen3.8:27b-draft-mq5` |
| 40 | draft | qwen3.8-27b/qwen38-27b-dflash-mq6.hfq | 1,658,852,352 | `d190ef2faa953252ac40e9706cf2c6763d095f7817c2c7a0466792fd6dfa9015` | `qwen3.8:27b-draft-mq6` |
| 41 | core | qwen3.8-flash-next/qwen3.8-flash-next-base.mq2-xts | 46,895,167,000 | `e2ba747485190da92cd7a36d3f56809e6db31a3ec328f5ab23e000275d798e57` | `qwen3.8:flash-next-mq2-xts` |
| 42 | core | qwen3.8-flash-next/qwen3.8-flash-next-base.mq2-xts-4500 | 44,537,253,400 | `9087d7d8f67629a3256c26bfeb705e7ed22dd1622bb91e074035543d6caea5c2` | `qwen3.8:flash-next-mq2-xts-4500` |
| 43 | core | qwen3.8-flash-next/qwen3.8-flash-next-base.mq3-xts | 59,109,362,200 | `d35896ca69b87d3e93176847a036db3e2cb1554931f130a842b13718b486238b` | `qwen3.8:flash-next-mq3-xts` |
| 44 | core | qwen3.8-flash-next/qwen3.8-flash-next-base.mq4-xts | 70,888,270,360 | `eafaff94fb9c6bb51282942e0ea6441cbdc266b634bb003b5913cfe2ccb90f45` | `qwen3.8:flash-next`, `qwen3.8:flash-next-mq4`, `qwen3.8:flash-next-gptq3`, `qwen3.8:flash-next-mq4-xts` |
| 45 | derivative | qwopus-27b/qwopus-27b.mq4 | 14,987,725,824 | `eb5202847bd4c07e067182fba04d06ef1d08bd584ebd0dc378c8ac8efd22a1b1` | `qwopus:27b` |
| 46 | derivative | qwopus-27b/qwopus-27b.mq6 | 21,393,705,984 | `563b0900399b3f0bbc3cc6e26a689d64a0d98bf3e327ba1d5fd8ff4056e94009` | `qwopus:27b-mq6` |
| 47 | derivative | qwopus3.6-27b-coder/qwopus3.6-27b-coder.mq4 | 14,992,575,488 | `808cdd6cf75179f15a921ec58179ff87f70f7e83eda2229582f52cc2f2d354c8` | `qwopus3.6:27b-coder` |

PLE sidecars declared by Flash-Next rows (registry `ple`): `qwen3.8-flash-next.ple-q4` (q4, 29,120,146,432 B, sha256 `fc78d30b9d2376db2b713732e80de2069f1cd1d6e1e40b50c716927dcc18ee1e`); `qwen3.8-flash-next.ple-q8` (q8, 54,400,269,312 B, sha256 `a655edc25402d0dfbef276cb3c5ca57724fa1074976d8a9029371a3a52eca89c`) for the base mq4-xts row; the mq2-xts, mq2-xts-4500 and mq3-xts rows carry their own `ple` entries in `inventory.json`.

## 3. Baseline matrix (three arches)

W = walked plan; SX = exact static closure (identity proven); SC = static-selector candidate, header unverified (22 files, NOT closure achieved); N = none (lazy). Walk scope: D default tag, L listed variant (xts/mq4r/mq2/mq3), O outside stated scope, T tested with trunk, P predictive fallback. Ornith A3B rows are core (D/L) per Main.

| # | file | class | default | scope | gfx1201 | gfx1100 | gfx1151 |
|---:|---|---|---|---|---|---|---|
| 1 | bonsai-27b.bq1 | derivative |  | P | N | N | N |
| 2 | bonsai-27b.tq2 | derivative | yes | P | N | N | N |
| 3 | carnice-27b.mq4 | derivative | yes | P | SC | N | N |
| 4 | carnice-27b.mq6 | derivative |  | P | SC | N | N |
| 5 | ornith-1.5-35b-a3b.mq4 | core | yes | D | N | N | N |
| 6 | ornith-1.5-35b-a3b.mq4r | core |  | L | N | N | N |
| 7 | qwen3.5-27b.mq3 | core |  | L | SC | N | N |
| 8 | qwen3.5-27b.mq4 | core | yes | D | SC | N | N |
| 9 | qwen3.5-27b.mq6 | core |  | O | SC | N | N |
| 10 | qwen35-27b-dflash-mq3.hfq | draft |  | T | N | N | N |
| 11 | qwen35-27b-dflash-mq4.hfq | draft | yes | T | N | N | N |
| 12 | qwen3.5-35b-a3b.mq4 | core | yes | D | N | N | N |
| 13 | qwen3.6-27b.mq3 | core |  | L | SC | N | N |
| 14 | qwen3.6-27b.mq4 | core | yes | D | SC | N | N |
| 15 | qwen36-27b-dflash-mq3.hfq | draft |  | T | N | N | N |
| 16 | qwen36-27b-dflash-mq4.hfq | draft | yes | T | N | N | N |
| 17 | qwen3.6-35b-a3b.mfp4 | core |  | O | N | N | N |
| 18 | qwen3.6-35b-a3b.mq2 | core |  | L | N | N | N |
| 19 | qwen3.6-35b-a3b.mq3p | core |  | L | N | N | N |
| 20 | qwen3.6-35b-a3b.mq4p | core | yes | D | N | N | N |
| 21 | qwen3.6-35b-a3b.mq4r | core |  | L | N | N | N |
| 22 | qwen3.6-35b-a3b.mq5 | core |  | O | N | N | N |
| 23 | qwen3.6-35b-a3b.mq6 | core |  | O | N | N | N |
| 24 | qwen3.8-27b.mq3 | core |  | L | SC | N | N |
| 25 | qwen3.8-27b.mq3-pro | core |  | L | SC | N | N |
| 26 | qwen3.8-27b.mq3-xt | core |  | L | SC | N | N |
| 27 | qwen3.8-27b.mq4 | core | yes | D | SC | N | N |
| 28 | qwen3.8-27b.mq4-pro | core |  | O | SC | N | N |
| 29 | qwen3.8-27b.mq4-xt | core |  | O | SC | N | N |
| 30 | qwen3.8-27b.mq4-xts | core |  | L | SX | N | N |
| 31 | qwen3.8-27b.mq5 | core |  | O | SC | N | N |
| 32 | qwen3.8-27b.mq5-pro | core |  | O | SC | N | N |
| 33 | qwen3.8-27b.mq5-xt | core |  | O | SC | N | N |
| 34 | qwen3.8-27b.mq6 | core |  | O | SC | N | N |
| 35 | qwen3.8-27b.mq6-pro | core |  | O | SC | N | N |
| 36 | qwen3.8-27b.mq6-xt | core |  | O | SC | N | N |
| 37 | qwen38-27b-dflash-mq3.hfq | draft |  | T | N | N | N |
| 38 | qwen38-27b-dflash-mq4.hfq | draft | yes | T | N | N | N |
| 39 | qwen38-27b-dflash-mq5.hfq | draft |  | T | N | N | N |
| 40 | qwen38-27b-dflash-mq6.hfq | draft |  | T | N | N | N |
| 41 | qwen3.8-flash-next-base.mq2-xts | core |  | L | N | N | N |
| 42 | qwen3.8-flash-next-base.mq2-xts-4500 | core |  | L | N | N | N |
| 43 | qwen3.8-flash-next-base.mq3-xts | core |  | L | N | N | N |
| 44 | qwen3.8-flash-next-base.mq4-xts | core | yes | D | W | N | N |
| 45 | qwopus-27b.mq4 | derivative | yes | P | SC | N | N |
| 46 | qwopus-27b.mq6 | derivative |  | P | SC | N | N |
| 47 | qwopus3.6-27b-coder.mq4 | derivative | yes | P | SC | N | N |

Totals (gfx1201): W 1, SX 1 (exact), **SC 22 (static-selector candidates, unverified, not closed)**, N 23; gfx1100 N 47/47; gfx1151 N 47/47. Proven closure on gfx1201 is therefore 2 artifacts (1 walked, 1 exact static) of 47; the other 22 are candidates only.

Reasons and conditions (source-derived; line anchors drift with sibling edits, function names are the selectors; `inventory.json` `artifacts[].baseline.<arch>.conditions` carries the per-row citations):

- **W on gfx1201, one file** `qwen3.8-flash-next-base.mq4-xts` (tags flash-next, flash-next-gptq3, flash-next-mq4, flash-next-mq4-xts, alias `qwen3.8:flash`). Conditions: `walked_route_entries` (kernel_registry.rs) needs model_arch 16 + exact arch gfx1201, header layout `65811e8a...`, `max_seq` <= 131072, default `kernel.pm_decode`, axes qsa=fp8, kv_backend=vmm, gdn=q8, qsa_f16_arena=off, qsa_pooled_bf16=off; `resolve_qwen4_route_plan` (hipfire-loader lib.rs) needs pp==1, ep<=1, no drafter/CASK sidecar, no adaptive KV, no retained Redline, default MTP K, no trunk overlay. Open policy: preload, not sealed. Identity: the walked file `mq4split-stage/split-plain/qwen3.8-flash-next-base.mq4-xts` (one inode with `split-hotset/`) re-hashes to the registry sha256 `eafaff94...0f45`.
- **Flash-Next mq2-xts, mq2-xts-4500, mq3-xts**: N. Plans are keyed on the header layout hash; neither walked layout was shown to equal these headers (headers not read), so `walked_route_entries` refuses and the load stays lazy. This also holds on gfx1201.
- **SX, one file** `qwen3.8-27b.mq4-xts`. Identity: local `~/.hipfire/models/qwen3.8-27b.mq4-xts` sha256 = registry `3e38ccba...f8ae`; its first 14,430,208 bytes (HFQM header) hash to `2a4f495e...5de0`, the header recorded in the jit-g0/jit-a3 fixtures the A3 tests plan against. Conditions: `qwen36_27b_route` needs gfx1201, model_arch 5, heads 24/4/256 and the layer/projection/shape fingerprint, then KV fp8/fp8, state q8, `max_seq` 2..=262144, prefill chunk 512..=8192, builtin flags; native MTP additionally needs the MQ4G256 sidecar head; `resolve_route_plan` refuses pp>1, CASK, adaptive KV, any DFlash draft, n-gram, `-dspark` sibling, vision; sealed after load. `qwen3.8-27b.mq4-xt` has the same size (14,987,185,152) but a different sha256 (`80e7c624...d0f3`): size is not identity, so it is SC.
- **SC, 22 files** (every other arch-5 dense file except Bonsai; includes Carnice/Qwopus). `qwen36_27b_route` would apply only if the header satisfies the same fingerprint (all 464 projections MQ4G256V2, 64 layers, 5120/17408/248320 dims) and the policy conditions above; no header was read, so whether any of them closes is unknown. Registry quant labels other than mq4 make a match unlikely. A non-match falls to lazy.
- **N on gfx1201 otherwise**: Bonsai bq1/tq2 (passthrough projections cannot satisfy the MQ4G256V2 fingerprint); all 10 A3B files including Ornith (`resolve_route_plan` refuses `model arch 6` before `route_entries`); all 8 drafts (`route_entries` refuses drafter routes without a separately admitted header; attaching a draft forces the trunk lazy via the "DFlash draft" refusal).
- **gfx1100 and gfx1151: N for all 47.** `qwen36_27b_route` is exact-`gfx1201` only and `walked_route_entries` filters on exact arch; the only checked-in plans are gfx1201. Historical observed-point inventories exist (section 7) but they are not plans (`is_closed_ready_plan: false`).

### 3a. Architecture admission exceptions (N is not universally "lazy")

An N cell means "no closed kernel plan". It does not mean the load runs lazily: a load can be refused before any kernel is requested. Per-expert slab codec admission for Flash-Next `qwen4.expert_slabs` v1 artifacts is decided by `expert_slab_plan` (`crates/hipfire-arch-qwen4/src/artifact.rs:328-366`, called from the artifact load at `:1450`), which refuses (does not compile lazily) when the architecture lacks a codec. The single capability table is `crates/rdna-compute/src/expert_capabilities.rs` (`slab_serves_with`), projected by `crates/rdna-compute/src/slab_admission.rs` (arch x codec table at lines 8-30; `SLAB_ROUTE_GFX1151_E8P_RECEIPTS_LANDED = true` at line 66); cold K1 kernels exist only for the pairs listed in `slab_moe.rs::slab_cold_k1_metadata` (`:461`: gfx1201 W3/E8P, gfx1100 E8P, gfx1151 W3/E8P).

| arch | MQ4 (0) | W3 (1, MQ3-I4) | W2 (2, MQ2-I4) | E8P (3, E8P-I4) |
|---|---|---|---|---|
| gfx1201 | admit | admit | refuse | admit |
| gfx1151 | admit | admit | refuse | admit (receipts constant true) |
| gfx1100 | admit | **refuse: no W3 K1/K2 kernels** | refuse | admit |
| other | refuse | refuse | refuse | refuse |

Consequences, each conditional on the artifact's slab metadata (not read for any registry file in this report):

1. **mq3 on gfx1100 is unsupported (load refused, not lazily compiled) if the published `qwen3.8-flash-next-base.mq3-xts` header carries a cold-W3 expert codec** (W3 expert tags in the slab table, `counts.w3 > 0`). If its header uses another codec this does not apply. The header was not inspected here, so this report states only the conditional caveat; the parent's current-header staging check decides it.
2. A cold codec of E8P is admitted on all three arches by this table (gfx1151 via the receipts constant); a header carrying W2 is refused on every arch (`ExpertSlabKernelsUnavailable`).
3. A layer holds at most one cold codec; a W3+E8P layer is refused on every arch.
4. Tier E8P/W3 hold cells do not gate the slab split (as reported by the parent from the capability source; not independently re-derived here).
5. The gfx1100 Flash-Next trunk also has the separate host-memory inadmissibility recorded in `docs/jit-g0-inventory.md`.

So for Flash-Next rows, "N on gfx1100/gfx1151" means lazy per-kernel compilation only when the load is admitted at all; where the codec table refuses, no JIT occurs because the load fails first.

## 4. Existing walked plans

`WALKED_ROUTE_PLAN_FILES` (crates/rdna-compute/src/kernel_registry.rs:5458) lists exactly two files; `walked_route_entries` (crates/rdna-compute/src/kernel_registry.rs:5633) selects by `model_arch` + exact arch, then by header layout sha256, then enforces `max_seq`, `kernel.pm_decode` default, and every `#! axis` value.

### `crates/rdna-compute/src/route_plans/qwen3.8-flash-next.gfx1201.tsv`

- canonical monolithic Flash-Next header (gptq3.mq4, 1325 tensors); grandfathered, not shipped by the registry. family `qwen3.8-flash-next`, model_arch 16, arch gfx1201, layout `a45ee786d88bea97949b78fed1fbcdd8a5957033ff467aea0df54d5af03cce90`, max_seq 131072.
- Axes (source selectors resolved by the loader in `resolve_qwen4_route_plan`): `gdn`=q8, `kv_backend`=vmm, `qsa`=fp8, `qsa_f16_arena`=off, `qsa_pooled_bf16`=off. Spec route: `ar` or `native_mtp` (default K only).
- Rows: AR 52 HIP + 3 embedded (139 symbols); native MTP 58 HIP + 3 embedded (165 symbols; includes every AR row); 61 distinct modules (including the fused HC-down + activation module `qwen4_hc_down_fused`, source digest refreshed by `scripts/route_plan_hc_down_digest.py`, the HC up+mix module `hyper_read_up_wmma_gfx1201` whose planned symbols are the retiled `_t128` and the persistent `_v2` entries, symbol list and source digest refreshed by `scripts/route_plan_hc_up_digest.py`, and the exact-INT8 MQ6 prefill trio `gemm_mq6g256v2_mmq_i8_gfx12`, its A8 quantizer module `gemm_mq4g256v2_residual_mmq_i8_gfx12` and the one-pass rotate + A8 producer `mq_rotate_x_i8_gfx12`, added by hand when `HIPFIRE_MQ6_I8` became the gfx1201 default; their digests are the current effective sources, not a walk capture). Embedded: `qsa_gather_pm_gfx1201`, `qwen4_moe_iu4_sym_pm_gfx1201`, `qwen4_mq6_x4_pm_gfx1201`.
- Walks: ar `mono-hotset-q4-ar` (637 requests); ar `mono-plain-ar` (637 requests); native_mtp `mono-hotset-q4-mtp` (230 requests); native_mtp `mono-plain-mtp` (230 requests).
- Registry identity: **unshipped**. walked model qwen3.8-flash-next-gptq3.mq4 = 125,288,544,792-byte monolith (jit-g0 header-flash-next.json header sha256 8703bb1626ff48b976a84d5f4f7d18b4aad8e9d1d9b780a52f963d73cb83c013, canonical model sha256 8b15b6fede7d7c5bfed0db4720a8295bedda51bc93e545fa242bd50d0f200972 per docs/jit-g0-inventory.md); no registry row has that size; the registry tag flash-next-gptq3 resolves to the split base file instead.

### `crates/rdna-compute/src/route_plans/qwen3.8-flash-next-base.mq4-xts.gfx1201.tsv`

- split Flash-Next header (canonical minus 128 PLE ngram shards, 1197 tensors). family `qwen3.8-flash-next`, model_arch 16, arch gfx1201, layout `65811e8a696c6b68d25a0c5bf9f971cef40caf60f47a83498a41297a15597c5c`, max_seq 131072.
- Axes (source selectors resolved by the loader in `resolve_qwen4_route_plan`): `gdn`=q8, `kv_backend`=vmm, `qsa`=fp8, `qsa_f16_arena`=off, `qsa_pooled_bf16`=off. Spec route: `ar` or `native_mtp` (default K only).
- Rows: AR 52 HIP + 3 embedded (139 symbols); native MTP 58 HIP + 3 embedded (165 symbols; includes every AR row); 61 distinct modules (including the fused HC-down + activation module `qwen4_hc_down_fused`, source digest refreshed by `scripts/route_plan_hc_down_digest.py`, the HC up+mix module `hyper_read_up_wmma_gfx1201` whose planned symbols are the retiled `_t128` and the persistent `_v2` entries, symbol list and source digest refreshed by `scripts/route_plan_hc_up_digest.py`, and the exact-INT8 MQ6 prefill trio `gemm_mq6g256v2_mmq_i8_gfx12`, its A8 quantizer module `gemm_mq4g256v2_residual_mmq_i8_gfx12` and the one-pass rotate + A8 producer `mq_rotate_x_i8_gfx12`, added by hand when `HIPFIRE_MQ6_I8` became the gfx1201 default; their digests are the current effective sources, not a walk capture). Embedded: `qsa_gather_pm_gfx1201`, `qwen4_moe_iu4_sym_pm_gfx1201`, `qwen4_mq6_x4_pm_gfx1201`.
- Walks: ar `split-plain-ar` (637 requests); ar `split-hotset-ar` (637 requests); native_mtp `split-plain-mtp` (230 requests); native_mtp `split-hotset-mtp` (230 requests).
- Registry identity: **sha256-verified**. walked paths mq4split-stage/split-plain and split-hotset are one inode (36450532); sha256 of that file recomputed this session = registry sha256 eafaff94...; a present-day content hash of the walked path, not a receipt captured at walk time.

Proof obligations that exist in source (not run here): `fn_route_plan.rs::every_walked_kernel_is_planned` (walk subset of plan, source/image sha256 equal), `::every_planned_hip_module_is_in_the_release_pack` (plan subset of the gfx1201 pack with equal source, flags, profile, symbols), `::unwalked_flash_next_policies_keep_lazy_loading`, `::unknown_header_layout_has_no_walked_route`. The test file states that older Flash-Next traces (jit-g0, kernel-trace) ran other policies and are not a closure oracle for this route.

## 5. Static 27B closure (gfx1201 only)

`qwen36_27b_route` (crates/rdna-compute/src/kernel_registry.rs:5303) returns a route only if: gpu arch `gfx1201`, model_arch 5, heads/kv/head_dim 24/4/256, 64 layers, 464 projection tensors all MQ4G256V2, lm_head MQ4G256V2 [248320,5120], embed Q8_0 [248320,5120], and three anchor shapes (mlp gate L0 [17408,5120], in_proj_qkv L0 [10240,5120], q_proj L3 [12288,5120]). It then requires KV fp8/fp8, state q8, max_seq 2..=262144, prefill chunk 512..=8192, spec Ar or NativeMtp (native MTP needs the MQ4G256 sidecar head tensors, otherwise refuse), builtin feature flags except `pm_decode`, no host-mapped experts. The loader additionally refuses pp>1, CASK sidecar, adaptive KV, any DFlash draft, n-gram drafter, `-dspark` sibling, vision tower, and `HIPFIRE_ROUTE_PLAN=0` (crates/hipfire-loader/src/lib.rs:2973).

Frozen manifest (`jit-a3/27b-xts-gfx1201.a3.json`, hipcc recipe with `pm_decode=false`): AR 72 entries / 120 symbols; native MTP 101 entries / 149 symbols (29 MTP-only modules). Default `pm_decode=true` replaces accepted HIP entries with embedded twins (test `pm_decode_twins_replace_their_hip_route_entries`). Tests (not run here): `jit_a3_route_plan.rs::qwen36_27b_gfx1201_plans_are_exact_frozen_manifests`, `::every_observed_route_symbol_is_planned_with_its_observed_identity`, `::explicit_drafter_and_unclosed_routes_refuse`.

Exact-identity proof exists for one artifact only, `qwen3.8-27b.mq4-xts` (SX in section 3). The other 22 arch-5 dense files are static-selector candidates (SC): the closure code path would apply only if their unread headers satisfy the fingerprint and policy conditions above, so none of them counts as a closed route. `docs/jit-g0-inventory.md` and `A3-closure-audit.md` record that the G0 point observations are lower bounds, not an oracle for all-row/tail/max-context/graph shapes.

## 6. Runtime policy: what "plan-open" means today

- **Sealed (27B static closure)**: `begin_route_kernels` passes `open = None`; after weights load `finish_route_kernel_load` (crates/rdna-compute/src/dispatch.rs:3985) seals the route and an unplanned kernel request returns an error rather than compiling (crates/rdna-compute/src/scratch.rs:488).
- **Open (Flash-Next walked plans)**: `begin_route_kernels` (crates/hipfire-loader/src/lib.rs:3293) sets `open = Some("Flash-Next <arch> <spec>")` when `hfq.arch_id == hipfire_arch_qwen4::ARCH_ID` (crates/hipfire-loader/src/lib.rs:3309). The planned modules are compiled/loaded before the first dispatch, but the route is **not sealed**: a kernel the walks did not bind loads lazily with one `kernels: WARNING unplanned kernel ...` line and is counted. Preload is therefore a performance guarantee for walked kernels only, not a no-JIT guarantee.
- **Lazy (everything else)**: when the load is admitted, it logs `kernels: compiling kernels on first use ... [no closed route plan: <reason>]` and each kernel compiles at first launch. Loads refused earlier (for example a slab-codec architecture refusal, section 3a) never reach this path. On this baseline the lazy path covers all A3B files, all drafts, every admitted gfx1100/gfx1151 load, the three non-base Flash-Next files, and any 27B load outside the conditions of section 5; for Flash-Next base it also covers EP, pp>1, drafter/CASK, adaptive KV, trunk overlay, retained Redline, non-default MTP K, unwalked axes, max_seq above 131072 and non-default `pm_decode`/flags.
- Consequence: a first request can still trigger JIT on every admitted lazy route, and on the open Flash-Next route for unwalked kernels. Unsupported routes are lazy by construction, not failing, unless an earlier admission gate refuses the load (section 3a); the loader returns `None` from route planning and continues.

## 7. Official kernel pack and trace-fixture coverage

- Pack inventory = `kernel_registry.rs::entries(arch, flags)` (built by `scripts/build-kernel-pack.sh` / the official builder). **Current observed packs (v2, local, not published):** freshly built from source commit `4e1a1ebc75eb1f78336d0787fce3f5a549fff3c3`, tag `first-request-jit-local-v2`, ROCm 10.1 (rocm_version 10.1.0, HIP 7.16.26385-0000000), code object version 6, ROCm range [10.1, 11.0), flags `-DIU4_A4_CANDIDATES=2` on all three arches, kernel cache ABI 6. Module counts, read from the generated `first-request-jit-evidence/packs-v2/<arch>/manifest.json`: **gfx1201 252, gfx1100 238, gfx1151 232**. Against the earlier packs the registry gained `tape_copy_batched` (portable, all five arches) and `conv1d_silu_split_f32_lanes` (gfx1201, actual caller only), per the CPU-gate owner. Per the same owner: zero compiler failures and every module's object, hash and index file present. Logs: `packs-v2/<arch>.log`, `packs-v2/DONE`, `packs-v2/tree_hashes.txt`. **This is a local verification, not a published release pack.**
- *Historical, superseded, kept immutable:* the earlier packs-v1 (`first-request-jit-evidence/packs/`, commit `ee7e16f59b700a033eed2ae3084d82274f4f30fb`, tag `first-request-jit-local`, ROCm 10.1) had 250 / 237 / 231 modules for gfx1201 / gfx1100 / gfx1151. They remain valid proof of what was built at that commit but are not the current counts.
- **Registry after first-request discovery (this change; packs-v2 predate it):** `entries(arch, "")` is **gfx1201 369, gfx1100 339, gfx1151 333** modules, each name unique per arch. The additions come from fresh-kernel-cache holds on the `hipfire serve` default load projection (daemon `567fc823296c`; per-cell receipts in `first-request-jit-evidence/discovery-cells/`, observed rows in `crates/rdna-compute/tests/fixtures/kernel-trace-first-request.tsv`, test `first_request_discovery_is_packaged`) and from the static scan below. Holes the holds found: Flash-Next `bf16_round_trip` (all arches), gfx1151 `indexed_attention_gathered_wmma` and `gemm_mq4g256v2_residual_wmma_gfx1151_bt` (a sampled request below 64 rows); on gfx1201 the A3B and 27B MQ3/MQ4/MQ6 prefill GEMMs, decode GEMVs and FA-prep modules (cells R4/R5/R7/R8/R11/R13/R15/R16; R17 needed nothing beyond R15). The 27B MQ4-xts DFlash cell R12 compiled nothing new. A static audit of the allowlist then registered the remaining MQ4V2 batch-tile prefill GEMMs that launches below 64 rows reach on the default config (gfx1100/gfx1151 `gemm_{qkvza,qkv,gate_up}_mq4g256v2_wmma_*_bt` and `gemm_mq4g256v2_residual_wmma_*_bt`, gfx1201 `gemm_qkv_mq4g256v2_wmma_gfx1201_bt`), because the IU4 route takes only 64 rows and up.
- **Beta carry adaptation:** the counts above describe the original FRJ inventory, not a remeasurement of later beta. The carry retains beta's slow-link and native A3B modules, declares their shared-source exports and the later `hyper_write_bf16x8_gfx1201` export, and retains the sealed gfx1100 fixture unchanged as historical evidence. The dynamic-site census now includes three additional beta descriptor-forwarding sites (`attention_flash_decode_gqa_tile`, `qwen4_moe_sym_check`, `qwen4_moe_sym_gemm`), for 86 dynamic rows; none is a runtime-value proof.
- **Static completeness gate** `crates/rdna-compute/tests/kernel_registry_coverage.rs`: every module argument of a runtime HIP-module sink in `crates/*/src` must be registered for each arch its name claims or carry a categorized, `file:line`-cited row in the shrink-only `tests/fixtures/lazy-kernel-allowlist.tsv` (1204 module rows: codec-outside-core-rungs, config-opt-in, developer-opt-in, no-caller, non-core-family, non-target-arch; 86 dynamic-site rows, including 38 dynamic-lazy rows). Dynamic sites (a `format!`ed or caller-forwarded module) are only listed, not proved: the gate cannot see which runtime values they take, so fresh-cache holds remain the evidence for those.
- Embedded native images (`kind: embedded`: e.g. `qsa_gather_pm_gfx1201`, `qwen4_moe_iu4_sym_pm_gfx1201`, `qwen4_mq6_x4_pm_gfx1201`) are code objects compiled into the runtime ELF with `include_bytes!`; they are not pack members and need no pack entry.
- Contract for any walked row: HIP rows must name one current pack module with every walked symbol; beta rebinds them to that module's current source/cache identity while preserving the historical source sha256 in the fixture. Embedded rows must still resolve to the runtime image with the walked image sha256. This matches `walked_route_entries` and `kernel_registry::tests::every_walked_plan_row_resolves_with_its_walked_identity`; for the two header-fixture layouts, `fn_route_plan.rs` also checks admission.
- **Checked-in open plan `crates/rdna-compute/src/route_plans/qwen3.8-flash-next-base.mq2-xts.gfx1201.tsv`** (registry `qwen3.8:flash-next-mq2-xts`, sha256 `e2ba7474...8e57`; layout `92d2c376d98535639df92432b6255c8de577ac43f57a5e9f32fbc71c970d471a`, same five axes and max_seq 131072 as the other two plans). Generated by `scripts/route_plan_build.py` from three audited walks at baseline `3866d41b` (`HIPFIRE_ROUTE_PLAN=0`, empty private kernel cache): AR shards `frj-w-mq2-1201-ar-01` (618 requests) + `-02` (19), the full 637-request AR plan, and native MTP `frj-w-mq2-1201-mtp-01`, the full 230-request MTP plan, all `ok`, 0 errors. Rows: AR 52 HIP + 9 embedded (147 symbols); native MTP 60 HIP + 9 embedded (176 symbols; adds `gemv_mq2g256v2`, `qwen4_gemv_q8_0`, `requant_g256`, `topk8_rescore_mq6g256v2`, `gemm_bf16_xf32_multirow_pto2`, the batched top-10 MoE gate_up/down and `gemv_q8_0`). The plan is **open** (no `#! sealed`).
  The walks found one HIP module outside the 4e1a1ebc7 inventory, `bf16_round_trip` (symbol `bf16_round_trip_f32`, source `kernels/src/bf16_round_trip.hip`, sha256 `b1d52caa...1964`; gfx1100/gfx1151 only carried the different strided module `qwen4_bf16_round_trip`). It is now registered on every supported arch, so packs-v2 predate it. The 9 embedded rows (`e8p_i4_moe_gemv_pm_gfx1201`, `ep_tile_mask_pm_gfx1201`, `gemv_hfq4g256_multirow_default_mq4v2`, `mq4_rows_x4_pm_gfx1201`, `mq_i4_moe_gemv_pm_gfx1201`, `qsa_gather_pm_gfx1201`, `qwen4_moe_iu4_sym_e8p_pm_gfx1201`, `qwen4_moe_iu4_sym_pm_gfx1201`, `qwen4_mq6_x4_pm_gfx1201`) resolve by image sha256 through `walked_embedded_image` (the `pm_decode` twin table or the exact-arch image catalog); none needed a catalog addition.
- **What the pack counts do and do not prove.** They prove the pack is complete for the registry inventory at `4e1a1ebc75`. They do not prove closure for any artifact on an arch without a walked plan: walks for gfx1100, gfx1151, A3B, the other 27B files and drafts are still absent, so no identity-complete artifact-to-module closure proof exists for them, and the section 3 baseline matrix is unchanged. Exact official-pack recipe counts for new walked routes remain to be established once those plans are generated and the parent final gates run.

Existing trace fixtures (`assert_trace_packaged`, crates/rdna-compute/src/kernel_registry.rs:7328: every traced `(arch, module, symbol)` is a member of `entries(arch)`; tests not run here). Rows per arch mix every model in the fixture, so they bound what was observed, not any one artifact:

| fixture | arch | traced modules | traced symbols | models traced on that arch |
|---|---|---:|---:|---|
| kernel-trace-qwen35.tsv | gfx1100 | 111 | 124 | ornith-1.5-35b-a3b.mq4 (MoE), qwen3.8-27b.mq4-xts (dense, load + AR) |
| kernel-trace-qwen35.tsv | gfx1151 | 99 | 111 | same |
| kernel-trace-qwen35.tsv | gfx1201 | 115 | 120 | same, plus 27B DFlash cold additions |
| kernel-trace-qwen4-flash-next.tsv | gfx1100 | 55 | 129 | XTX route walk of the canonical mq4-xts header (`route_plan_walk.py --profile xtx-fn-bounded`, AR + native MTP), not a cold-cache trace |
| kernel-trace-qwen4-flash-next.tsv | gfx1151 | 55 | 132 | flash-next-gptq3.mq4, then-shipped flash-next.mq4, cold-cache pack runs |
| kernel-trace-qwen4-flash-next.tsv | gfx1201 | 52 | 113 | gptq3 (host-mapped experts), flash-next.mq4, cold pack runs |

Overlap with the split-base plan: 50 of 54 planned HIP modules appear in the gfx1201 Flash-Next trace; not in the trace: `gemm_bf16_xf32_multirow_pto2`, `gemm_mq6g256v2_residual_wmma_gfx12_bt4_mq5v2`, `qsa_bf16_scores`, `qsa_filter_select` (these four are plan-only; the plan-subset-of-pack test, not the trace, is what covers them). Trace file names (`qwen3.8-flash-next.mq4`, `ornith-1.5-35b-a3b.mq4`) are name matches only; no sha256 binds those trace models to registry rows, so the traces give family-level, per-arch observed lower bounds, not artifact identity.

Historical G0 fixtures (`crates/rdna-compute/tests/fixtures/jit-g0/*.routes.json`, none a closed plan; `is_closed_ready_plan: false` in all six):

| fixture | arch | status | modules (HIP+embedded) | default symbol lists | HIP modules also in matching trace |
|---|---|---|---:|---:|---:|
| 27b-xts | gfx1201 | observed-point-routes | 45 (44+1) | 51 | 41/44 |
| 27b-xts | gfx1100 | observed-point-routes | 25 (24+1) | 30 | 22/24 |
| 27b-xts | gfx1151 | observed-point-routes | 27 (25+2) | 31 | 22/25 |
| flash-next | gfx1201 | observed-point-routes | 42 (40+2) | 107 | 40/40 |
| flash-next | gfx1100 | source-predicted | 29 (29+0) | 67 | no trace on this arch |
| flash-next | gfx1151 | failed-default-request-observation | 17 (16+1) | 28 | 16/16 |

`docs/jit-g0-inventory.md` tabulates 106 default symbols for flash-next-gfx1201; the fixture `default_entries` symbol lists sum to 107 here. The FN fixtures were produced from the 125,288,544,792-byte monolithic `gptq3.mq4` (not a registry file; header sha256 `8703bb16...3c013`), so they are not evidence for any of the 47 registry artifacts. flash-next-gfx1100 is source-predicted and excluded (24 GiB VRAM + ~30 GiB host cannot admit it); flash-next-gfx1151 records requests before a failed native prefill (qwen4_moe_group_ranks) and is a pre-failure lower bound. The 27B gfx1100/gfx1151 fixtures exclude the ordinary-decode runs that were reset. These are observations recorded in the repo; none was rerun.

## 8. PLE format on Flash-Next mq2 versus mq4: availability fallback, not policy

Conclusion: intended availability fallback, not a quant-tier policy; no PLE policy change needed.

- direct runs: env.json child_selected lists no HIPFIRE_PLE_FORMAT; isolated HOME (parent-verified)
- j1440 direct mq2 logs show q4 rows; direct mq4 shows q8 (parent-verified)
- cpu-preflight preflight.json (competitive/strata-020-evidence/cpu-preflight/preflight.json): mq2 artifact has only a .ple-q4 symlink (-> /home/kaden/qcal/models/qwen3.8-flash-next-gptq3.mq4.ple-q4, 29,120,146,432 B); mq4 artifact has .ple-q4 (29,120,146,432 B), .expert-hotset, .trunk-overlay
- current stat (not a historical receipt): mq2 .ple-q8 absent; mq4 .ple-q8 present, 54,400,269,312 B (parent-verified)
- external MATCH sections 2 and 5 and the direct env/log receipts are cited as supplied by parent and were not re-read here
- Source: crates/hipfire-config/src/lib.rs:9281 (Auto -> Q4 on gfx1151, Q8 elsewhere); `resolve_and_attach_ple_rows` (crates/hipfire-runtime/src/ple_sidecar.rs:881) honors rows already attached under `auto` and, for a split base, picks the preferred format and otherwise the other with a note (parent-cited ranges ~888-906 and ~958-987).

Caveat that must stay explicit: the j1440 mq2 file was the OLD artifact d3d536c0...a949 of 47,038,936,600 B (preflight claimed_sha256_not_rehashed); the registry mq2-xts row is e2ba7474...8e57 base file of 46,895,167,000 B. They are different files and are not interchangeable evidence.

## 9. Evidence performed for this report

- jq/python deterministic parse of registry/v1.json (47 distinct repo+file groups from 51 tags)
- sha256sum /home/kaden/.hipfire/models/qwen3.8-27b.mq4-xts == registry qwen3.8:27b-mq4-xts (local read)
- sha256 of first 14430208 bytes of that file == 2a4f495e...5de0 (jit-g0 27B header sha256)
- sha256sum mq4split-stage/split-plain/qwen3.8-flash-next-base.mq4-xts == registry qwen3.8:flash-next (eafaff94...)
- no model payload downloaded; the inventory facts above were produced without cargo build/test, formatter or GPU runs
- CPU gate evidence reported by the CPU-gate owner for commit `ee7e16f59` (`first-request-jit-evidence/cpu/rc.txt`, logs in `cpu/`): release and debug builds rc 0; `pre-commit-ci.sh origin/beta` rc 0 (PASS); tests `fn_route_plan` 5, `jit_a3_route_plan` 10, `jit_g0_route_data` 7, `kernel_registry` 31, loader 178, Python 12, all PASS. Only the three pack manifests were re-read for this report; the test counts are cited, not re-run here.
- Incremental CPU gates for commit `4e1a1ebc75` as reported by the CPU-gate owner (not re-run here; only the three v2 manifests were re-read): kernel registry 32 tests, pack 5, kernel-cache ABI 60, railgun corpus 2 + 10, precommit PASS, release build PASS. The hardware walk table (section 3) is unchanged until real new route-plan TSVs exist.

Line anchors above are as of generation while siblings edit `kernel_registry.rs` and the loader; the function and test names are the stable selectors.
