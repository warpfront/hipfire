use hipfire_dispatch::pipeline::sealed_moe::{
    ContractAssignment, ContractAxis, ContractCollectiveHint, ContractCollectiveRow,
    ContractParallelism, ExpertExecutionContract, ROOT_ROUTED_EP_EXECUTION,
};
use hipfire_runtime::ep::{RootRoutedEpReduction, RootRoutedEpSchedule};

fn row(name: &str, layer: usize, axis: ContractAxis) -> ContractCollectiveRow {
    ContractCollectiveRow {
        name: name.into(),
        layer,
        hint: ContractCollectiveHint::AllReduce { kind: axis },
    }
}

fn contract(
    layer: usize,
    physical_devices: Vec<i32>,
    rows: Vec<ContractCollectiveRow>,
) -> ExpertExecutionContract {
    ExpertExecutionContract::new(
        "ffn",
        Some(layer),
        "source-v1",
        7,
        physical_devices,
        ContractParallelism::ExpertParallel,
        ContractAssignment::Stride,
        vec![0, 1, 0, 1],
        vec![0, 0, 1, 1],
        ROOT_ROUTED_EP_EXECUTION,
        rows,
    )
    .expect("test execution contract is internally well-formed")
}

fn valid_contract() -> ExpertExecutionContract {
    contract(
        3,
        vec![3, 5],
        vec![
            row("attention", 3, ContractAxis::Ep),
            row("moe", 3, ContractAxis::Ep),
        ],
    )
}

fn derive(
    contract: &ExpertExecutionContract,
    n_ranks: usize,
    layer: usize,
    route_count: usize,
    partial_bytes: usize,
    reduce_count: usize,
    reduction: RootRoutedEpReduction,
) -> Result<RootRoutedEpSchedule, hipfire_dispatch::types::DispatchError> {
    RootRoutedEpSchedule::derive(
        contract,
        n_ranks,
        layer,
        route_count,
        partial_bytes,
        reduce_count,
        route_count * reduce_count,
        reduce_count,
        reduction,
    )
}

#[test]
fn accepts_actual_moe_collective_position_and_reuses_contract_identity() {
    let contract = valid_contract();
    let decode = derive(&contract, 2, 3, 16, 128, 32, RootRoutedEpReduction::Decode)
        .expect("valid root-routed contract must derive");
    let prefill = derive(&contract, 2, 3, 16, 128, 32, RootRoutedEpReduction::Prefill)
        .expect("same contract and geometry must serve prefill");

    assert_eq!(decode.contract_id(), contract.contract_id());
    assert_eq!(decode.contract_id(), prefill.contract_id());
    assert_eq!(decode.layer(), 3);
    assert_eq!(decode.rank_count(), 2);
    assert_eq!(decode.route_count(), 16);
    assert_eq!(decode.partial_bytes(), 128);
    assert_eq!(decode.reduce_count(), 32);
    assert_eq!(decode.contribution_count(), 512);
    assert_eq!(decode.contribution_chunk(), 32);
}

#[test]
fn rejects_duplicate_moe_collective_rows() {
    let contract = contract(
        3,
        vec![3, 5],
        vec![
            row("moe", 3, ContractAxis::Ep),
            row("moe", 3, ContractAxis::Ep),
        ],
    );
    let error = derive(&contract, 2, 3, 16, 128, 32, RootRoutedEpReduction::Decode)
        .expect_err("duplicate named rows must be refused");
    assert!(error.to_string().contains("duplicate moe collective"));
}

#[test]
fn rejects_missing_moe_collective_row() {
    let contract = contract(3, vec![3, 5], vec![row("attention", 3, ContractAxis::Ep)]);
    let error = derive(&contract, 2, 3, 16, 128, 32, RootRoutedEpReduction::Decode)
        .expect_err("a schedule without its named moe row must be refused");
    assert!(error.to_string().contains("no moe EP all-reduce row"));
}

#[test]
fn rejects_wrong_layer_moe_collective_row() {
    let contract = contract(3, vec![3, 5], vec![row("moe", 2, ContractAxis::Ep)]);
    let error = derive(&contract, 2, 3, 16, 128, 32, RootRoutedEpReduction::Decode)
        .expect_err("a row at another layer must not be reused");
    assert!(error.to_string().contains("no moe EP all-reduce row"));
}

#[test]
fn rejects_wrong_axis_moe_collective_row() {
    let contract = contract(3, vec![3, 5], vec![row("moe", 3, ContractAxis::Tp)]);
    let error = derive(&contract, 2, 3, 16, 128, 32, RootRoutedEpReduction::Decode)
        .expect_err("a TP row must not authorize an EP schedule");
    assert!(error
        .to_string()
        .contains("admitted root-routed EP contract"));
}

#[test]
fn rejects_mismatched_mesh_and_activation_geometry() {
    let contract = valid_contract();

    assert!(derive(&contract, 3, 3, 16, 128, 32, RootRoutedEpReduction::Decode).is_err());
    assert!(derive(&contract, 2, 3, 0, 128, 32, RootRoutedEpReduction::Decode).is_err());
    assert!(derive(&contract, 2, 3, 16, 6, 1, RootRoutedEpReduction::Decode).is_err());
    assert!(derive(&contract, 2, 3, 16, 128, 33, RootRoutedEpReduction::Decode).is_err());
}

#[test]
fn rejects_contract_identity_encoded_by_wrong_layer() {
    let wrong_layer = contract(2, vec![3, 5], vec![row("moe", 2, ContractAxis::Ep)]);
    assert!(derive(
        &wrong_layer,
        2,
        3,
        16,
        128,
        32,
        RootRoutedEpReduction::Decode,
    )
    .is_err());
}

// The executor's effect ordering requires live Gpu/Gpus resources and is
// covered by Main's hardware validation. These CPU tests deliberately exercise
// only the real contract/geometry admission boundary; they do not duplicate
// the executor with a fake effect model.
