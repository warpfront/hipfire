// SPDX-License-Identifier: Apache-2.0
//! Weight preparation, program selection and thin GPU glue for the opt-in
//! Flash-Next NPU trunk. References live in peacemaker-npu; no device opens here.

pub(crate) use calibration::runtime_key_with_firmware;
#[cfg(target_os = "linux")]
pub use calibration::{bounded_memlock_bytes, trunk_pin_bytes};
pub use calibration::{host_facts, npu_firmware, HostFacts};
#[cfg(target_os = "linux")]
pub(crate) use calibration::{TRUNK_BANKS, TRUNK_ROWS_MAX};
#[cfg(target_os = "linux")]
pub use railgun::npu::fclk;
mod calibration;
mod config;
pub(crate) mod containment;
mod gpu;
mod packing;
mod programs;
pub mod route;
mod shapes;

pub(crate) use packing::pack_matrix_direct;
pub(crate) use programs::build_rowmajor_programs_for_families;

pub use calibration::{
    builtin_record, load_calibration, model_index_identity, model_trunk_index_sha256,
    runtime_calibration_key,
};
pub use config::{arch_admits_auto, NPU_TRUNK_DAEMON_KEY, NPU_TRUNK_ENV, SWITCH};
pub use packing::{
    pack_matrix, pack_out, pack_out_layers, pack_shape, pack_shape_with_contract, pack_trunk,
    pack_trunk_with_contract, LayerTrunkRefs, MatrixRef, PackedB, PackedOut, PackedTrunk,
};
pub use programs::{
    build_out_programs, build_program_for_rows, build_programs, build_programs_for_families,
    wave_padded_rows, NpuOutPrograms, NpuPrograms, ShapeProgram,
};
pub use route::{diagnostic_gpu_route, set_diagnostic_gpu_route};
pub use shapes::{
    ColumnMap, Family, FamilyMask, Shape, FLASH_NEXT, OUT_COLS, OUT_K, OUT_PANEL_COLS, PANEL_COLS,
    TRUNK_K,
};

#[cfg(test)]
mod plan_tests;
#[cfg(test)]
mod rates_tests;
