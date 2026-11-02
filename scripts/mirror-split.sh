#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Sync the PeaceMaker sub-project to its standalone public repository.
#
#   scripts/mirror-split.sh peacemaker [destination]
#
#   peacemaker  history-preserving filter of the committed source tree, branch `main`
#
# Reads the committed tree and history of the source checkout (HEAD, or
# MIRROR_SOURCE_REV) and never touches it. The destination (default
# mirror-preview/peacemaker under the source checkout) must not exist. It never
# pushes; scripts/mirror-push.sh gates and pushes.
#
# Environment: MIRROR_SOURCE_REV, MIRROR_FILTER_REPO (path of git-filter-repo if
# it is not on PATH).
# Requires: git, python3 >= 3.11, git-filter-repo 2.47.0, cargo (settles the
# generated Cargo.lock).
# Author and committer metadata is normalized by the checked-in
# scripts/mirror.mailmap on every run (it is not part of the mirrored tree).
# See scripts/mirror_split.py for the exact rules.
set -euo pipefail

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec python3 "$here/mirror_split.py" "$@"
