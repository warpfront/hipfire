# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import sys

if "--source-reference" in sys.argv or "--compare" in sys.argv:
    from .quality import main
else:
    from .generate_fixtures import main

raise SystemExit(main())
