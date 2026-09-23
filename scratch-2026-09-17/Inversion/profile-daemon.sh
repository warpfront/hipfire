#!/bin/sh
set -eu
: "${HIPFIRE_PROFILE_DIR:?}"
: "${HIPFIRE_REAL_DAEMON:?}"
mkdir -p "$HIPFIRE_PROFILE_DIR"
case "${HIPFIRE_PROFILE_MODE:-trace}" in
  trace)
    exec rocprofv3 --kernel-trace --stats --output-format csv json \
      --output-directory "$HIPFIRE_PROFILE_DIR" --output-file profile -- \
      "$HIPFIRE_REAL_DAEMON" "$@"
    ;;
  att)
    : "${HIPFIRE_PROFILE_REGEX:?}"
    exec rocprofv3 --att --kernel-include-regex "$HIPFIRE_PROFILE_REGEX" \
      --kernel-iteration-range "${HIPFIRE_PROFILE_ITER:-1}" --att-consecutive-kernels 1 \
      --output-format csv --output-directory "$HIPFIRE_PROFILE_DIR" \
      --output-file profile -- "$HIPFIRE_REAL_DAEMON" "$@"
    ;;
  *)
    echo "unknown HIPFIRE_PROFILE_MODE=$HIPFIRE_PROFILE_MODE" >&2
    exit 2
    ;;
esac
