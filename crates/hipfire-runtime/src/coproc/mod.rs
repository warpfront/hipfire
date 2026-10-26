// SPDX-License-Identifier: Apache-2.0
//! A co-processor (XDNA2 NPU) as a heterogeneous pipeline stage beside the GPU.
//!
//! One module per concern:
//! - [`stage`]: the op-level stage map an architecture declares (which projection families the co-processor
//!   may own, per layer kind) and the ownership table ([`FamilySet`]) over it, plus the stage numerics
//!   contract ([`EpilogueContract`]);
//! - [`calibration`]: the stored measurements that price one map on one host;
//! - [`plan`] / [`rates`]: the deterministic planner that picks the ownership and the chunk-pair micro-batches,
//!   and the live rate scales that refine its prior under `auto`;
//! - [`schedule`]: the layer-major order of a chunk pair and the dependency edges it respects;
//! - [`mode`]: the per-stage switch (`off`, `auto`, `on`, `emulate`) and its resolution;
//! - [`admission`]: the checks that run before any weight or device work, and the settlement of a refusal;
//! - [`containment`]: the fault latch and the GPU-only rerun of a faulted request;
//! - [`pair`]: the bank-pair geometry (where the session snapshot cuts a group, what each bank launches);
//!
//! Architecture crates keep only their shape description (the [`StageMap`] value), weight packing and the hooks
//! that run one (layer, bank) visit.

pub mod admission;
pub mod calibration;
pub mod containment;
pub mod mode;
pub mod pair;
pub mod plan;
pub mod rates;
pub mod schedule;
pub mod stage;

pub use calibration::{
    calibration_build_identity, calibration_key, CalKey, Calibration, ContentionCalibration,
    DeviceKey, FamilyCalibration, GlueCalibration, GpuCalibration, KernelKey, NpuCalibration,
    OutFamily, QsaOrder, CALIBRATION_SCHEMA_VERSION,
};
pub use mode::{Mode, Switch};
pub use plan::{
    admit_wave_budget, auto_admission, bank_segments, bank_segments_moved, contention_extra,
    crossover, executable_plan, executable_plan_for, gpu_chunk, load_log_line, npu_ms, pair_time,
    pair_time_moved, plan, plan_executable, plan_moved, planned_pair, planned_pair_rows,
    request_log_line, select_families, BankSegments, GpuChunk, LoadLogKey, Moved, PairTime, Plan,
    SelectionReport, SelectionRow, Step, CHUNK, SELECTION_ROWS,
};
pub use rates::{
    choose_families, pair_model, scaled_calibration, PairModel, RateEstimator, RateSample,
    RateSnapshot, APPLY_EPS, EWMA_ALPHA, FREEZE_AFTER, FREEZE_EPS, FREEZE_WINDOW, MIN_MODEL_MS,
    REOPEN_CONFIRM, REOPEN_EPS, SCALE_MAX, SCALE_MIN,
};
pub use schedule::{pair_edges, pair_schedule, Edge, PairOp};
pub use stage::{
    EpilogueContract, FamilyDecl, FamilySet, GroupDecl, LayerKindDecl, OpSlot, StageDevice,
    StageMap, GROUPS, KINDS,
};
