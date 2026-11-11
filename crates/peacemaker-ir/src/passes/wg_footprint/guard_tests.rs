// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Synthetic kernels for the guard-aware prover (see [`super::guard`]).
//!
//! Kernargs: `out` (pointer, offset 0), `n` (i32, offset 8), `m` (i32, offset 12).
//! The output index is `row * n + col`, stored at `out + 4 * index`.

use super::*;
use crate::cfg::Body;
use crate::descriptor::{
    KernargPreload, KernelCodeProperties, KernelDescriptor, Rsrc1, Rsrc2, Rsrc3,
};
use crate::inst::{KernelOrigin, SymbolId, Wave};
use crate::metadata::{HsaKernelMetadata, Kernarg, KernelMeta};

/// A gfx1201 wave32 HSA kernel from llvm-mc words: pointer `out` at 0, i32 `n`
/// at 8, i32 `m` at 12 (kernarg pointer in s[0:1], x work-item id in v0).
fn kernel(words: &[u32]) -> Kernel {
    let mut body = Body::default();
    let mut at = 0;
    while at < words.len() {
        let (inst, used) = crate::codec::gfx12::decode(&words[at..]).expect("decodes");
        let id = body.insts.insert(inst);
        body.layout.push(id);
        at += used;
    }
    crate::passes::cfg::build_blocks(&mut body, Arch::Gfx1201).expect("blocks");
    let descriptor = KernelDescriptor {
        group_segment_fixed_size: 0,
        private_segment_fixed_size: 0,
        kernarg_size: 16,
        kernel_code_entry_byte_offset: 0,
        compute_pgm_rsrc3: Rsrc3(0),
        compute_pgm_rsrc1: Rsrc1(0x1f),
        compute_pgm_rsrc2: Rsrc2(0x384),
        kernel_code_properties: KernelCodeProperties(0x408),
        kernarg_preload: KernargPreload(0),
        reserved: [0; 28],
    };
    let scalar = |name: &str, offset: u32| Kernarg {
        name: name.into(),
        offset,
        size: 4,
        value_kind: "by_value".into(),
        address_space: None,
    };
    let parsed = KernelMeta {
        name: "k".into(),
        symbol: "k.kd".into(),
        kernarg_segment_size: 16,
        kernarg_segment_align: 8,
        wavefront_size: 32,
        args: vec![
            Kernarg {
                name: "out".into(),
                offset: 0,
                size: 8,
                value_kind: "global_buffer".into(),
                address_space: Some("global".into()),
            },
            scalar("n", 8),
            scalar("m", 12),
        ],
        ..Default::default()
    };
    Kernel {
        symbol: SymbolId("k".into()),
        wave: Wave::Wave32,
        abi: Abi::Hsa {
            descriptor,
            metadata: HsaKernelMetadata {
                raw_msgpack: Vec::new(),
                parsed,
            },
        },
        body,
        origin: KernelOrigin::Authored {
            builder_crate: "test".into(),
            version: "0".into(),
            git: "0".into(),
        },
    }
}

fn class(words: &[u32], block_x: u32) -> ArgWgClass {
    let r = analyze(&kernel(words), Arch::Gfx1201, false, WgLaunch { block_x });
    assert_eq!(r.applicable, Ok(()));
    r.arg(0).expect("pointer arg").clone()
}

fn disjoint(words: &[u32], block_x: u32) -> WgDisjoint {
    match class(words, block_x) {
        ArgWgClass::Disjoint(d) => d,
        other => panic!("expected Disjoint, got {other:?}"),
    }
}

fn unproven(words: &[u32], block_x: u32) {
    match class(words, block_x) {
        ArgWgClass::Unproven(why) => assert!(!why.is_empty()),
        other => panic!("expected Unproven, got {other:?}"),
    }
}

fn req(offset: u32, mask: u32, value: u32, op: WgReqOp) -> WgRequirement {
    WgRequirement {
        offset,
        mask,
        value,
        op,
    }
}

// P1 (llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx1201 -filetype=obj; llvm-objcopy -O binary --only-section=.text):
//   s_load_b64 s[2:3], s[0:1], 0x0
//   s_load_b32 s4, s[0:1], 0x8
//   s_wait_kmcnt 0x0
//   s_ashr_i32 s5, s4, 31
//   s_mov_b32 s8, ttmp7
//   s_ashr_i32 s9, s8, 31
//   s_mul_u64 s[10:11], s[4:5], s[8:9]
//   s_lshl_b64 s[10:11], s[10:11], 2
//   s_lshl_b32 s6, ttmp9, 4
//   s_ashr_i32 s7, s6, 31
//   s_lshl_b64 s[12:13], s[6:7], 2
//   s_cmp_ge_i32 s6, s4
//   s_cbranch_scc1 END
//   s_add_nc_u64 s[2:3], s[2:3], s[10:11]
//   s_add_nc_u64 s[2:3], s[2:3], s[12:13]
//   v_lshlrev_b32 v1, 2, v0
//   v_mov_b32 v2, 0
//   global_store_b32 v1, v2, s[2:3]
// END:
//   s_endpgm
const P1: [u32; 23] = [
    0xf4002080, 0xf8000000, 0xf4000100, 0xf8000008, 0xbfc70000, 0x86059f04, 0xbe880073, 0x86099f08,
    0xaa8a0804, 0x848a820a, 0x84068475, 0x86079f06, 0x848c8206, 0xbf030406, 0xbfa20007, 0xa9820a02,
    0xa9820c02, 0x30020082, 0x7e040280, 0xee068002, 0x01000000, 0x00000001, 0xbfb00000,
];
// P2 (llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx1201 -filetype=obj; llvm-objcopy -O binary --only-section=.text):
//   s_load_b64 s[2:3], s[0:1], 0x0
//   s_load_b32 s4, s[0:1], 0x8
//   s_wait_kmcnt 0x0
//   s_ashr_i32 s5, s4, 31
//   s_mov_b32 s8, ttmp7
//   s_ashr_i32 s9, s8, 31
//   s_mul_u64 s[10:11], s[4:5], s[8:9]
//   s_lshl_b64 s[10:11], s[10:11], 2
//   s_lshl_b32 s6, ttmp9, 4
//   s_ashr_i32 s7, s6, 31
//   s_lshl_b64 s[12:13], s[6:7], 2
//   s_cmp_ge_i32 s6, s4
//   s_cbranch_scc1 END
//   s_add_nc_u64 s[2:3], s[2:3], s[10:11]
//   s_add_nc_u64 s[2:3], s[2:3], s[12:13]
//   v_mov_b32 v1, 0
//   v_mov_b32 v2, 0
//   global_store_b32 v1, v2, s[2:3]
// END:
//   s_endpgm
const P2: [u32; 23] = [
    0xf4002080, 0xf8000000, 0xf4000100, 0xf8000008, 0xbfc70000, 0x86059f04, 0xbe880073, 0x86099f08,
    0xaa8a0804, 0x848a820a, 0x84068475, 0x86079f06, 0x848c8206, 0xbf030406, 0xbfa20007, 0xa9820a02,
    0xa9820c02, 0x7e020280, 0x7e040280, 0xee068002, 0x01000000, 0x00000001, 0xbfb00000,
];
// P3 (llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx1201 -filetype=obj; llvm-objcopy -O binary --only-section=.text):
//   s_load_b64 s[2:3], s[0:1], 0x0
//   s_load_b32 s4, s[0:1], 0x8
//   s_wait_kmcnt 0x0
//   s_ashr_i32 s5, s4, 31
//   s_mov_b32 s8, ttmp7
//   s_ashr_i32 s9, s8, 31
//   s_mul_u64 s[10:11], s[4:5], s[8:9]
//   s_lshl_b64 s[10:11], s[10:11], 2
//   s_lshl_b32 s6, ttmp9, 4
//   s_ashr_i32 s7, s6, 31
//   s_lshl_b64 s[12:13], s[6:7], 2
//   s_cmp_ge_i32 s6, s4
//   s_cbranch_scc1 END
//   s_add_nc_u64 s[2:3], s[2:3], s[10:11]
//   s_add_nc_u64 s[2:3], s[2:3], s[12:13]
//   v_cmpx_gt_u32 8, v0
//   v_lshlrev_b32 v1, 2, v0
//   v_mov_b32 v2, 0
//   global_store_b32 v1, v2, s[2:3]
// END:
//   s_endpgm
const P3: [u32; 24] = [
    0xf4002080, 0xf8000000, 0xf4000100, 0xf8000008, 0xbfc70000, 0x86059f04, 0xbe880073, 0x86099f08,
    0xaa8a0804, 0x848a820a, 0x84068475, 0x86079f06, 0x848c8206, 0xbf030406, 0xbfa20008, 0xa9820a02,
    0xa9820c02, 0x7d980088, 0x30020082, 0x7e040280, 0xee068002, 0x01000000, 0x00000001, 0xbfb00000,
];
// P4_STRIDE2 (llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx1201 -filetype=obj; llvm-objcopy -O binary --only-section=.text):
//   s_load_b64 s[2:3], s[0:1], 0x0
//   s_load_b32 s4, s[0:1], 0x8
//   s_wait_kmcnt 0x0
//   s_ashr_i32 s5, s4, 31
//   s_lshl_b32 s8, ttmp7, 1
//   s_ashr_i32 s9, s8, 31
//   s_mul_u64 s[10:11], s[4:5], s[8:9]
//   s_lshl_b64 s[10:11], s[10:11], 2
//   s_lshl_b32 s6, ttmp9, 4
//   s_ashr_i32 s7, s6, 31
//   s_lshl_b64 s[12:13], s[6:7], 2
//   s_cmp_ge_i32 s6, s4
//   s_cbranch_scc1 END
//   s_add_nc_u64 s[2:3], s[2:3], s[10:11]
//   s_add_nc_u64 s[2:3], s[2:3], s[12:13]
//   v_mov_b32 v1, 0
//   v_mov_b32 v2, 0
//   global_store_b32 v1, v2, s[2:3]
//   s_lshl_b64 s[14:15], s[4:5], 2
//   s_add_nc_u64 s[2:3], s[2:3], s[14:15]
//   global_store_b32 v1, v2, s[2:3]
// END:
//   s_endpgm
const P4_STRIDE2: [u32; 28] = [
    0xf4002080, 0xf8000000, 0xf4000100, 0xf8000008, 0xbfc70000, 0x86059f04, 0x84088173, 0x86099f08,
    0xaa8a0804, 0x848a820a, 0x84068475, 0x86079f06, 0x848c8206, 0xbf030406, 0xbfa2000c, 0xa9820a02,
    0xa9820c02, 0x7e020280, 0x7e040280, 0xee068002, 0x01000000, 0x00000001, 0x848e8204, 0xa9820e02,
    0xee068002, 0x01000000, 0x00000001, 0xbfb00000,
];
// P4_STRIDE1 (llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx1201 -filetype=obj; llvm-objcopy -O binary --only-section=.text):
//   s_load_b64 s[2:3], s[0:1], 0x0
//   s_load_b32 s4, s[0:1], 0x8
//   s_wait_kmcnt 0x0
//   s_ashr_i32 s5, s4, 31
//   s_mov_b32 s8, ttmp7
//   s_ashr_i32 s9, s8, 31
//   s_mul_u64 s[10:11], s[4:5], s[8:9]
//   s_lshl_b64 s[10:11], s[10:11], 2
//   s_lshl_b32 s6, ttmp9, 4
//   s_ashr_i32 s7, s6, 31
//   s_lshl_b64 s[12:13], s[6:7], 2
//   s_cmp_ge_i32 s6, s4
//   s_cbranch_scc1 END
//   s_add_nc_u64 s[2:3], s[2:3], s[10:11]
//   s_add_nc_u64 s[2:3], s[2:3], s[12:13]
//   v_mov_b32 v1, 0
//   v_mov_b32 v2, 0
//   global_store_b32 v1, v2, s[2:3]
//   s_lshl_b64 s[14:15], s[4:5], 2
//   s_add_nc_u64 s[2:3], s[2:3], s[14:15]
//   global_store_b32 v1, v2, s[2:3]
// END:
//   s_endpgm
const P4_STRIDE1: [u32; 28] = [
    0xf4002080, 0xf8000000, 0xf4000100, 0xf8000008, 0xbfc70000, 0x86059f04, 0xbe880073, 0x86099f08,
    0xaa8a0804, 0x848a820a, 0x84068475, 0x86079f06, 0x848c8206, 0xbf030406, 0xbfa2000c, 0xa9820a02,
    0xa9820c02, 0x7e020280, 0x7e040280, 0xee068002, 0x01000000, 0x00000001, 0x848e8204, 0xa9820e02,
    0xee068002, 0x01000000, 0x00000001, 0xbfb00000,
];
// P5 (llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx1201 -filetype=obj; llvm-objcopy -O binary --only-section=.text):
//   s_load_b64 s[2:3], s[0:1], 0x0
//   s_load_b32 s4, s[0:1], 0x8
//   s_load_b32 s12, s[0:1], 0xc
//   s_wait_kmcnt 0x0
//   s_ashr_i32 s5, s4, 31
//   s_lshr_b32 s13, ttmp7, 16
//   s_and_b32 s14, ttmp7, 0xffff
//   s_mul_i32 s15, s12, s13
//   s_add_co_i32 s8, s15, s14
//   s_ashr_i32 s9, s8, 31
//   s_mul_u64 s[10:11], s[4:5], s[8:9]
//   s_lshl_b64 s[10:11], s[10:11], 2
//   s_lshl_b32 s6, ttmp9, 4
//   s_ashr_i32 s7, s6, 31
//   s_lshl_b64 s[16:17], s[6:7], 2
//   s_cmp_ge_i32 s14, s12
//   s_cbranch_scc1 END
//   s_cmp_ge_i32 s6, s4
//   s_cbranch_scc1 END
//   s_add_nc_u64 s[2:3], s[2:3], s[10:11]
//   s_add_nc_u64 s[2:3], s[2:3], s[16:17]
//   v_mov_b32 v1, 0
//   v_mov_b32 v2, 0
//   global_store_b32 v1, v2, s[2:3]
// END:
//   s_endpgm
const P5: [u32; 31] = [
    0xf4002080, 0xf8000000, 0xf4000100, 0xf8000008, 0xf4000300, 0xf800000c, 0xbfc70000, 0x86059f04,
    0x850d9073, 0x8b0eff73, 0x0000ffff, 0x960f0d0c, 0x81080e0f, 0x86099f08, 0xaa8a0804, 0x848a820a,
    0x84068475, 0x86079f06, 0x84908206, 0xbf030c0e, 0xbfa20009, 0xbf030406, 0xbfa20007, 0xa9820a02,
    0xa9821002, 0x7e020280, 0x7e040280, 0xee068002, 0x01000000, 0x00000001, 0xbfb00000,
];
// N1 (llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx1201 -filetype=obj; llvm-objcopy -O binary --only-section=.text):
//   s_load_b64 s[2:3], s[0:1], 0x0
//   s_load_b32 s4, s[0:1], 0x8
//   s_wait_kmcnt 0x0
//   s_ashr_i32 s5, s4, 31
//   s_mov_b32 s8, ttmp7
//   s_ashr_i32 s9, s8, 31
//   s_mul_u64 s[10:11], s[4:5], s[8:9]
//   s_lshl_b64 s[10:11], s[10:11], 2
//   s_lshl_b32 s6, ttmp9, 4
//   s_ashr_i32 s7, s6, 31
//   s_lshl_b64 s[12:13], s[6:7], 2
//   s_cmp_ge_i32 s6, s4
//   s_cbranch_scc1 L
//   s_nop 0
// L:
//   s_add_nc_u64 s[2:3], s[2:3], s[10:11]
//   s_add_nc_u64 s[2:3], s[2:3], s[12:13]
//   v_lshlrev_b32 v1, 2, v0
//   v_mov_b32 v2, 0
//   global_store_b32 v1, v2, s[2:3]
// END:
//   s_endpgm
const N1: [u32; 24] = [
    0xf4002080, 0xf8000000, 0xf4000100, 0xf8000008, 0xbfc70000, 0x86059f04, 0xbe880073, 0x86099f08,
    0xaa8a0804, 0x848a820a, 0x84068475, 0x86079f06, 0x848c8206, 0xbf030406, 0xbfa20001, 0xbf800000,
    0xa9820a02, 0xa9820c02, 0x30020082, 0x7e040280, 0xee068002, 0x01000000, 0x00000001, 0xbfb00000,
];
// N2_WGX (llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx1201 -filetype=obj; llvm-objcopy -O binary --only-section=.text):
//   s_load_b64 s[2:3], s[0:1], 0x0
//   s_load_b32 s4, s[0:1], 0x8
//   s_wait_kmcnt 0x0
//   s_ashr_i32 s5, s4, 31
//   s_mov_b32 s8, ttmp7
//   s_ashr_i32 s9, s8, 31
//   s_mul_u64 s[10:11], s[4:5], s[8:9]
//   s_lshl_b64 s[10:11], s[10:11], 2
//   s_lshl_b32 s6, ttmp9, 4
//   s_ashr_i32 s7, s6, 31
//   s_lshl_b64 s[12:13], s[6:7], 2
//   s_cmp_ge_i32 ttmp9, s4
//   s_cbranch_scc1 END
//   s_add_nc_u64 s[2:3], s[2:3], s[10:11]
//   s_add_nc_u64 s[2:3], s[2:3], s[12:13]
//   v_lshlrev_b32 v1, 2, v0
//   v_mov_b32 v2, 0
//   global_store_b32 v1, v2, s[2:3]
// END:
//   s_endpgm
const N2_WGX: [u32; 23] = [
    0xf4002080, 0xf8000000, 0xf4000100, 0xf8000008, 0xbfc70000, 0x86059f04, 0xbe880073, 0x86099f08,
    0xaa8a0804, 0x848a820a, 0x84068475, 0x86079f06, 0x848c8206, 0xbf030475, 0xbfa20007, 0xa9820a02,
    0xa9820c02, 0x30020082, 0x7e040280, 0xee068002, 0x01000000, 0x00000001, 0xbfb00000,
];
// N2_M (llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx1201 -filetype=obj; llvm-objcopy -O binary --only-section=.text):
//   s_load_b64 s[2:3], s[0:1], 0x0
//   s_load_b32 s4, s[0:1], 0x8
//   s_load_b32 s12, s[0:1], 0xc
//   s_wait_kmcnt 0x0
//   s_ashr_i32 s5, s4, 31
//   s_mov_b32 s8, ttmp7
//   s_ashr_i32 s9, s8, 31
//   s_mul_u64 s[10:11], s[4:5], s[8:9]
//   s_lshl_b64 s[10:11], s[10:11], 2
//   s_lshl_b32 s6, ttmp9, 4
//   s_ashr_i32 s7, s6, 31
//   s_lshl_b64 s[12:13], s[6:7], 2
//   s_cmp_ge_i32 s6, s12
//   s_cbranch_scc1 END
//   s_add_nc_u64 s[2:3], s[2:3], s[10:11]
//   s_add_nc_u64 s[2:3], s[2:3], s[12:13]
//   v_lshlrev_b32 v1, 2, v0
//   v_mov_b32 v2, 0
//   global_store_b32 v1, v2, s[2:3]
// END:
//   s_endpgm
const N2_M: [u32; 25] = [
    0xf4002080, 0xf8000000, 0xf4000100, 0xf8000008, 0xf4000300, 0xf800000c, 0xbfc70000, 0x86059f04,
    0xbe880073, 0x86099f08, 0xaa8a0804, 0x848a820a, 0x84068475, 0x86079f06, 0x848c8206, 0xbf030c06,
    0xbfa20007, 0xa9820a02, 0xa9820c02, 0x30020082, 0x7e040280, 0xee068002, 0x01000000, 0x00000001,
    0xbfb00000,
];
// N3 (llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx1201 -filetype=obj; llvm-objcopy -O binary --only-section=.text):
//   s_load_b64 s[2:3], s[0:1], 0x0
//   s_load_b32 s4, s[0:1], 0x8
//   s_wait_kmcnt 0x0
//   s_ashr_i32 s5, s4, 31
//   s_mov_b32 s8, ttmp7
//   s_ashr_i32 s9, s8, 31
//   s_mul_u64 s[10:11], s[4:5], s[8:9]
//   s_lshl_b64 s[10:11], s[10:11], 2
//   s_lshl_b32 s6, ttmp9, 3
//   s_ashr_i32 s7, s6, 31
//   s_lshl_b64 s[12:13], s[6:7], 2
//   s_cmp_ge_i32 s6, s4
//   s_cbranch_scc1 END
//   s_add_nc_u64 s[2:3], s[2:3], s[10:11]
//   s_add_nc_u64 s[2:3], s[2:3], s[12:13]
//   v_lshlrev_b32 v1, 2, v0
//   v_mov_b32 v2, 0
//   global_store_b32 v1, v2, s[2:3]
// END:
//   s_endpgm
const N3: [u32; 23] = [
    0xf4002080, 0xf8000000, 0xf4000100, 0xf8000008, 0xbfc70000, 0x86059f04, 0xbe880073, 0x86099f08,
    0xaa8a0804, 0x848a820a, 0x84068375, 0x86079f06, 0x848c8206, 0xbf030406, 0xbfa20007, 0xa9820a02,
    0xa9820c02, 0x30020082, 0x7e040280, 0xee068002, 0x01000000, 0x00000001, 0xbfb00000,
];
// N4 (llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx1201 -filetype=obj; llvm-objcopy -O binary --only-section=.text):
//   s_load_b64 s[2:3], s[0:1], 0x0
//   s_load_b32 s4, s[0:1], 0x8
//   s_wait_kmcnt 0x0
//   s_ashr_i32 s5, s4, 31
//   s_mov_b32 s8, ttmp7
//   s_ashr_i32 s9, s8, 31
//   s_mul_u64 s[10:11], s[4:5], s[8:9]
//   s_lshl_b64 s[10:11], s[10:11], 2
//   s_lshl_b32 s6, ttmp9, 4
//   s_ashr_i32 s7, s6, 31
//   s_lshl_b64 s[12:13], s[6:7], 2
//   s_add_nc_u64 s[2:3], s[2:3], s[10:11]
//   s_add_nc_u64 s[2:3], s[2:3], s[12:13]
//   v_lshlrev_b32 v1, 2, v0
//   v_mov_b32 v2, 0
//   global_store_b32 v1, v2, s[2:3]
// END:
//   s_endpgm
const N4: [u32; 21] = [
    0xf4002080, 0xf8000000, 0xf4000100, 0xf8000008, 0xbfc70000, 0x86059f04, 0xbe880073, 0x86099f08,
    0xaa8a0804, 0x848a820a, 0x84068475, 0x86079f06, 0x848c8206, 0xa9820a02, 0xa9820c02, 0x30020082,
    0x7e040280, 0xee068002, 0x01000000, 0x00000001, 0xbfb00000,
];
// N5 (llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx1201 -filetype=obj; llvm-objcopy -O binary --only-section=.text):
//   s_load_b64 s[2:3], s[0:1], 0x0
//   s_load_b32 s4, s[0:1], 0x8
//   s_wait_kmcnt 0x0
//   s_ashr_i32 s5, s4, 31
//   s_mov_b32 s8, ttmp7
//   s_ashr_i32 s9, s8, 31
//   s_mul_u64 s[10:11], s[4:5], s[8:9]
//   s_lshl_b64 s[10:11], s[10:11], 2
//   s_lshl_b32 s6, ttmp9, 4
//   s_ashr_i32 s7, s6, 31
//   s_lshl_b64 s[12:13], s[6:7], 2
//   s_cmp_ge_u32 s6, s4
//   s_cbranch_scc1 END
//   s_add_nc_u64 s[2:3], s[2:3], s[10:11]
//   s_add_nc_u64 s[2:3], s[2:3], s[12:13]
//   v_lshlrev_b32 v1, 2, v0
//   v_mov_b32 v2, 0
//   global_store_b32 v1, v2, s[2:3]
// END:
//   s_endpgm
const N5: [u32; 23] = [
    0xf4002080, 0xf8000000, 0xf4000100, 0xf8000008, 0xbfc70000, 0x86059f04, 0xbe880073, 0x86099f08,
    0xaa8a0804, 0x848a820a, 0x84068475, 0x86079f06, 0x848c8206, 0xbf090406, 0xbfa20007, 0xa9820a02,
    0xa9820c02, 0x30020082, 0x7e040280, 0xee068002, 0x01000000, 0x00000001, 0xbfb00000,
];
// N6 (llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx1201 -filetype=obj; llvm-objcopy -O binary --only-section=.text):
//   s_load_b64 s[2:3], s[0:1], 0x0
//   s_load_b32 s4, s[0:1], 0x8
//   s_load_b32 s12, s[0:1], 0xc
//   s_wait_kmcnt 0x0
//   s_ashr_i32 s5, s4, 31
//   s_lshr_b32 s13, ttmp7, 16
//   s_and_b32 s14, ttmp7, 0xffff
//   s_mul_i32 s15, s12, s13
//   s_add_co_i32 s8, s15, s14
//   s_ashr_i32 s9, s8, 31
//   s_mul_u64 s[10:11], s[4:5], s[8:9]
//   s_lshl_b64 s[10:11], s[10:11], 2
//   s_lshl_b32 s6, ttmp9, 4
//   s_ashr_i32 s7, s6, 31
//   s_lshl_b64 s[16:17], s[6:7], 2
//   s_cmp_ge_i32 s6, s4
//   s_cbranch_scc1 END
//   s_add_nc_u64 s[2:3], s[2:3], s[10:11]
//   s_add_nc_u64 s[2:3], s[2:3], s[16:17]
//   v_mov_b32 v1, 0
//   v_mov_b32 v2, 0
//   global_store_b32 v1, v2, s[2:3]
// END:
//   s_endpgm
const N6: [u32; 29] = [
    0xf4002080, 0xf8000000, 0xf4000100, 0xf8000008, 0xf4000300, 0xf800000c, 0xbfc70000, 0x86059f04,
    0x850d9073, 0x8b0eff73, 0x0000ffff, 0x960f0d0c, 0x81080e0f, 0x86099f08, 0xaa8a0804, 0x848a820a,
    0x84068475, 0x86079f06, 0x84908206, 0xbf030406, 0xbfa20007, 0xa9820a02, 0xa9821002, 0x7e020280,
    0x7e040280, 0xee068002, 0x01000000, 0x00000001, 0xbfb00000,
];

// P6 (llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx1201 -filetype=obj; llvm-objcopy -O binary --only-section=.text):
//   s_load_b64 s[2:3], s[0:1], 0x0
//   s_load_b32 s4, s[0:1], 0x8
//   s_wait_kmcnt 0x0
//   s_ashr_i32 s5, s4, 31
//   s_mov_b32 s8, ttmp7
//   s_ashr_i32 s9, s8, 31
//   s_mul_u64 s[10:11], s[4:5], s[8:9]
//   s_lshl_b64 s[10:11], s[10:11], 2
//   s_lshl_b32 s6, ttmp9, 3
//   s_ashr_i32 s7, s6, 31
//   s_lshl_b64 s[12:13], s[6:7], 2
//   s_add_nc_u64 s[2:3], s[2:3], s[10:11]
//   s_add_nc_u64 s[2:3], s[2:3], s[12:13]
//   v_lshlrev_b32 v1, 2, v0
//   v_mov_b32 v2, 0
//   global_store_b32 v1, v2, s[2:3]
// END:
//   s_endpgm
const P6: [u32; 21] = [
    0xf4002080, 0xf8000000, 0xf4000100, 0xf8000008, 0xbfc70000, 0x86059f04, 0xbe880073, 0x86099f08,
    0xaa8a0804, 0x848a820a, 0x84068375, 0x86079f06, 0x848c8206, 0xa9820a02, 0xa9820c02, 0x30020082,
    0x7e040280, 0xee068002, 0x01000000, 0x00000001, 0xbfb00000,
];

#[test]
fn p1_origin_guard_with_lane_window() {
    let d = disjoint(&P1, 16);
    assert_eq!(d.axes, [0, 1, 2]);
    assert_eq!(d.sites, 1);
    assert_eq!(d.requires, [req(8, 15, 0, WgReqOp::MaskEq)]);
    assert!(d.bounds[0] >= 1 << 16);
}

#[test]
fn p2_single_lane_origin_guard() {
    let d = disjoint(&P2, 1);
    assert_eq!(d.axes, [0, 1, 2]);
    assert_eq!(d.sites, 1);
    assert!(
        d.requires.iter().all(|r| r.op != WgReqOp::MaskEq),
        "{:?}",
        d.requires
    );
}

#[test]
fn p3_exec_predicate_window() {
    let d = disjoint(&P3, 16);
    assert_eq!(d.axes, [0, 1, 2]);
    assert_eq!(d.sites, 1);
    assert!(
        d.requires.contains(&req(8, 7, 0, WgReqOp::MaskEq)),
        "{:?}",
        d.requires
    );
}

#[test]
fn p4_two_stores_row_stride_two_is_disjoint() {
    let d = disjoint(&P4_STRIDE2, 1);
    assert_eq!(d.axes, [0, 1, 2]);
    assert_eq!(d.sites, 2);
}

#[test]
fn p4_two_stores_row_stride_one_overlaps() {
    unproven(&P4_STRIDE1, 1);
}

#[test]
fn p5_nested_radix_row() {
    let d = disjoint(&P5, 1);
    assert_eq!(d.axes, [0, 1, 2]);
    assert_eq!(d.sites, 1);
    assert!(
        d.requires
            .iter()
            .any(|r| r.offset == 12 && r.op == WgReqOp::LeU && r.value > 0),
        "{:?}",
        d.requires
    );
}

#[test]
fn n1_guard_does_not_dominate() {
    let d = disjoint(&N1, 16);
    assert_eq!(d.axes, [0, 1, 2]);
    assert_eq!(d.sites, 1);
    assert_eq!(d.requires, [req(8, 0, 16, WgReqOp::GridLe)]);
}

#[test]
fn n2_guard_on_workgroup_id_not_element_index() {
    let d = disjoint(&N2_WGX, 16);
    assert_eq!(d.axes, [0, 1, 2]);
    assert_eq!(d.sites, 1);
    assert_eq!(d.requires, [req(8, 0, 16, WgReqOp::GridLe)]);
}

#[test]
fn n2_guard_against_wrong_dword() {
    let d = disjoint(&N2_M, 16);
    assert_eq!(d.axes, [0, 1, 2]);
    assert_eq!(d.sites, 1);
    assert_eq!(d.requires, [req(8, 0, 16, WgReqOp::GridLe)]);
}

#[test]
fn n3_origin_stride_smaller_than_lane_window() {
    unproven(&N3, 16);
}

#[test]
fn n4_missing_guard() {
    let d = disjoint(&N4, 16);
    assert_eq!(d.axes, [0, 1, 2]);
    assert_eq!(d.sites, 1);
    assert_eq!(d.requires, [req(8, 0, 16, WgReqOp::GridLe)]);
}

#[test]
fn n5_unsigned_compare_of_sign_extended_stride() {
    // The unsigned guard is not accepted as a signed bound; the proof falls
    // back to the GridLe precondition alone (observed).
    let d = disjoint(&N5, 16);
    assert_eq!(d.requires, [req(8, 0, 16, WgReqOp::GridLe)], "{d:?}");
}

#[test]
fn n6_nested_radix_without_row_guard() {
    unproven(&N6, 1);
}

#[test]
fn p6_grid_precondition_without_any_guard() {
    let d = disjoint(&P6, 8);
    assert_eq!(d.axes, [0, 1, 2]);
    assert_eq!(d.sites, 1);
    assert_eq!(d.requires, [req(8, 0, 8, WgReqOp::GridLe)]);
}
