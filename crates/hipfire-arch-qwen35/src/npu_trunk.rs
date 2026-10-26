//! The qwen3.8-27b dense co-processor stage (NPU trunk): its switch, its op-level stage map and the
//! load-time gate. The prefill driver and the 27B weight packer are not in this crate yet, so every
//! non-off mode is refused at load. The map is the validated shape the driver will consume.

use hipfire_runtime::coproc::{
    EpilogueContract, FamilyDecl, GroupDecl, LayerKindDecl, Mode, StageMap, Switch,
};

/// Model width: the reduction dimension of every input projection of the 27B trunk.
const K: usize = 5120;

/// Developer switch of the 27B stage; `fusions.qwen35_npu_trunk` is its config key.
pub const SWITCH: Switch = Switch {
    env: "HIPFIRE_QWEN35_NPU_TRUNK",
    config_key: "fusions.qwen35_npu_trunk",
    tag: "qwen35-npu",
    name: "qwen35_npu_trunk",
};

/// Refuses every non-off mode at load: the 27B prefill driver is not in this build.
pub fn check_mode() -> Result<(), String> {
    refuse_unless_off(Mode::from_config(SWITCH, Mode::Off)?)
}

fn refuse_unless_off(mode: Mode) -> Result<(), String> {
    if mode.is_off() {
        return Ok(());
    }
    Err(format!(
        "{}={mode}: the qwen3.8-27b NPU prefill driver is not in this build; keep the switch off",
        SWITCH.env
    ))
}

/// The input projections the NPU may own, grouped by kind: GDN qkv, z, a, b, then FA q, k, v.
/// Columns are output rows: GDN 10240 + 6144 + 48 + 48, FA gated q 12288 + k 1024 + v 1024.
const FAMILIES: [FamilyDecl; 7] = [
    FamilyDecl {
        name: "qkv",
        kind: 0,
        cols: 10240,
    },
    FamilyDecl {
        name: "z",
        kind: 0,
        cols: 6144,
    },
    FamilyDecl {
        name: "a",
        kind: 0,
        cols: 48,
    },
    FamilyDecl {
        name: "b",
        kind: 0,
        cols: 48,
    },
    FamilyDecl {
        name: "q",
        kind: 1,
        cols: 12288,
    },
    FamilyDecl {
        name: "k",
        kind: 1,
        cols: 1024,
    },
    FamilyDecl {
        name: "v",
        kind: 1,
        cols: 1024,
    },
];

// Layer counts are the checkpoint's `layer_types` (qwen3.8-27b.mq4: 48 linear_attention,
// 16 full_attention); `hidden_size` is 5120.
// ponytail: out-family names mirror qwen4 and are checked only for distinctness; the driver PR
// calibrates the real families and runs `check_model` on the loaded layer kinds before any plan.
pub static STAGE_MAP: StageMap = StageMap {
    tag: "qwen35-npu",
    kinds: [
        LayerKindDecl {
            name: "gdn",
            layers: 48,
            in_k: K,
            out_family: "gdn.out",
        },
        LayerKindDecl {
            name: "fa",
            layers: 16,
            in_k: K,
            out_family: "fa.o",
        },
    ],
    families: &FAMILIES,
    groups: [
        GroupDecl {
            name: "qkv",
            kind: 0,
            families: 0x01,
        },
        GroupDecl {
            name: "zab",
            kind: 0,
            families: 0x0e,
        },
        GroupDecl {
            name: "q",
            kind: 1,
            families: 0x10,
        },
        GroupDecl {
            name: "kv",
            kind: 1,
            families: 0x60,
        },
    ],
    contract: EpilogueContract::Bf16Direct,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_is_valid_and_matches_the_27b_dims() {
        STAGE_MAP.validate().unwrap();
        let kinds: Vec<usize> = (0..64).map(|l| usize::from(l % 4 == 3)).collect();
        STAGE_MAP.check_model(&kinds, 5120).unwrap();
    }

    #[test]
    fn only_off_passes_the_load_gate() {
        assert!(refuse_unless_off(Mode::Off).is_ok());
        for mode in [Mode::Auto, Mode::On, Mode::Emulate] {
            let err = refuse_unless_off(mode).unwrap_err();
            assert!(err.starts_with("HIPFIRE_QWEN35_NPU_TRUNK="), "{err}");
        }
    }
}
