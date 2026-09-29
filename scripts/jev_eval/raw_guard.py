# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Nick Woolmer
# hipfire — see LICENSE and NOTICE in the project root.

"""The one guard against collecting a calibrated answer into held-out or
reported data (spec §13.2, §13.6). heldout.py `run`, run_jevbench.py and
run_calibration.py all assume the model they drive is raw: no
`decide.calibration.*` configured for it. If it is not, every downstream
number — the fit (calibrate.py), the report (report.py), the reported rows
themselves — would be built from already-calibrated answers.

`x-hipfire-timing` carries a `calibration` entry only when serve applied a
non-identity calibration (hipfire-generate decide.rs `note_calibration`;
spec §13.2 "Timing header"). Its absence is not proof positive that the
model is uncalibrated on every question type — an absent T defaults to 1 —
but its *presence* is proof positive that it is not raw, which is what
these collectors must refuse."""
import json


def check_raw(get_header):
    """`get_header`: a callable taking a header name and returning its value
    or None/""/absent, e.g. `http.client.HTTPResponse.headers.get` or a
    plain dict's `.get`. Raises RuntimeError if `x-hipfire-timing` carries a
    `calibration` entry."""
    raw = get_header("x-hipfire-timing")
    if not raw:
        return
    timing = json.loads(raw)
    cal = timing.get("calibration")
    if cal is not None:
        raise RuntimeError(
            f"got a calibrated answer (x-hipfire-timing.calibration={cal}); this collector "
            "requires raw answers only. Unset decide.calibration.* for this model "
            "(docs/CONFIG.md) before running it.")
