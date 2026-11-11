//! Fail-closed collector diagnostics and cleanup aggregation.
//! This exact helper is shared by the GPU tool and standalone CPU controls.

pub(crate) const INCOMPLETE_IDS: &str =
    "exact-cache route collection incomplete or invalid expert id";

pub(crate) enum CollectionFailure<'a> {
    InvalidInput(&'a str),
    Hip(&'a str),
    Other(&'a str),
    UnexpectedSuccess,
}

pub(crate) fn require_missing_layer(
    failure: CollectionFailure<'_>,
) -> Result<&'static str, String> {
    match failure {
        CollectionFailure::InvalidInput(INCOMPLETE_IDS) => Ok(INCOMPLETE_IDS),
        CollectionFailure::InvalidInput(message) => Err(format!(
            "missing-layer control returned an unrelated validation error: {message}"
        )),
        CollectionFailure::Hip(message) => Err(format!(
            "missing-layer control failed in HIP/readback/wait, not the intended ID check: {message}"
        )),
        CollectionFailure::Other(message) => Err(format!(
            "missing-layer control returned an unrelated error: {message}"
        )),
        CollectionFailure::UnexpectedSuccess => {
            Err("unwritten collector layer unexpectedly returned success".into())
        }
    }
}

pub(crate) fn merge_cleanup(
    primary: Result<(), String>,
    cleanup: Vec<String>,
) -> Result<(), String> {
    match (primary, cleanup.is_empty()) {
        (result, true) => result,
        (Ok(()), false) => Err(format!("collector cleanup: {}", cleanup.join("; "))),
        (Err(primary), false) => Err(format!(
            "{primary}; collector cleanup: {}",
            cleanup.join("; ")
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_the_named_invalid_id_check_proves_the_negative() {
        assert_eq!(
            require_missing_layer(CollectionFailure::InvalidInput(INCOMPLETE_IDS)).unwrap(),
            INCOMPLETE_IDS
        );
        for message in [
            "cache route readback has no active batch",
            "event wait failed",
            "",
        ] {
            assert!(require_missing_layer(CollectionFailure::InvalidInput(message)).is_err());
        }
        assert!(require_missing_layer(CollectionFailure::UnexpectedSuccess).is_err());
    }

    #[test]
    fn transport_errors_cannot_masquerade_as_expected_validation() {
        // Even an identical diagnostic string must fail when its typed source
        // is HIP (or another subsystem), rather than Invalid input.
        for message in [
            INCOMPLETE_IDS,
            "HIP readback failed",
            "event_synchronize failed",
        ] {
            assert!(require_missing_layer(CollectionFailure::Hip(message)).is_err());
            assert!(require_missing_layer(CollectionFailure::Other(message)).is_err());
        }
    }

    #[test]
    fn a_cleanup_failure_cannot_erase_the_exercise_failure() {
        let error = merge_cleanup(
            Err("primary: collector readback failed".into()),
            vec![
                "HIP fence failed".into(),
                "pinned-buffer free failed".into(),
                "fixture release failed".into(),
            ],
        )
        .unwrap_err();
        for cause in [
            "primary: collector readback failed",
            "HIP fence failed",
            "pinned-buffer free failed",
            "fixture release failed",
        ] {
            assert!(error.contains(cause), "lost failure cause {cause}: {error}");
        }
        assert!(error.starts_with("primary: collector readback failed"));
    }

    #[test]
    fn cleanup_only_failure_and_unchanged_primary_are_reported() {
        assert!(merge_cleanup(Ok(()), vec!["fixture release failed".into()])
            .unwrap_err()
            .contains("fixture release failed"));
        assert_eq!(
            merge_cleanup(Err("primary".into()), vec![]).unwrap_err(),
            "primary"
        );
        merge_cleanup(Ok(()), vec![]).unwrap();
    }
}
