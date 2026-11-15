#!/usr/bin/env bash
# Verify a model path IS the pinned canonical dense fixture before any number
# is taken on it. The pin is pin:model/qwen3.8-27b.mq4-xts in fixtures/pins.toml
# (size + sha256). The asymmetric qwen3.8-27b.mq4-xt and the retired mq4-xts
# test upload have exactly the pinned size, and lookalikes have turned up under
# the canonical filename (e.g. a local qwen3.8-27b.mq4-xt symlinked to a
# symmetric file), so the sha256 is always verified (~10 s). A size or digest
# that belongs to another pin (e.g. the discarded non-AWQ mq4-xt upload,
# pin:model/qwen3.8-27b.mq4-xt-no-awq) is named in the failure.
#
#   scripts/check_fixture.sh [PATH]   (default: ~/.hipfire/models/qwen3.8-27b.mq4-xts)
set -euo pipefail
path="${1:-$HOME/.hipfire/models/qwen3.8-27b.mq4-xts}"
if ! python3 "$(dirname "${BASH_SOURCE[0]}")/check-pins.py" verify model/qwen3.8-27b.mq4-xts "$path"; then
  echo "FAIL: not the pinned canonical dense fixture; numbers measured on it are not comparable." >&2
  exit 2
fi
