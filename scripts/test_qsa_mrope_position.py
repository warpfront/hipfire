#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# Copyright (c) 2026 HUSRCF
# hipfire — see LICENSE and NOTICE in the project root.
"""GPU-free test of the production QSA position helper, including two mutants.

Requires a host C++ compiler, never HIP/ROCm. CXX defaults to c++ (cl on
Windows). A missing compiler is a failure, not a skipped correctness check.
"""

import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    compiler = shlex.split(os.environ.get("CXX", "cl" if os.name == "nt" else "c++"))
    if not compiler or shutil.which(compiler[0]) is None:
        raise RuntimeError(f"required host C++ compiler not found: {compiler!r}")
    msvc = Path(compiler[-1]).stem.lower() == "cl"
    header = (ROOT / "kernels/src/qsa_mrope_position.h").read_text()
    original = "int first_token = block * compress;"
    if header.count(original) != 1:
        raise RuntimeError("position helper mutation anchor is absent or ambiguous")
    cases = [
        ("production", header, True),
        ("wrong_block_index", header.replace(original, "int first_token = block;"), False),
        ("wrong_last_token", header.replace(original, "int first_token = block * compress + compress - 1;"), False),
    ]
    with tempfile.TemporaryDirectory(prefix="hipfire-qsa-position-") as temporary:
        work = Path(temporary)
        source = work / "qsa_mrope_position_host.cpp"
        source.write_bytes((ROOT / "tests/qsa_mrope_position_host.cpp").read_bytes())
        for name, contents, should_pass in cases:
            (work / "qsa_mrope_position.h").write_text(contents)
            binary = work / (name + (".exe" if os.name == "nt" else ""))
            if msvc:
                flags = ["/nologo", "/std:c++14", "/EHsc", str(source), f"/Fe:{binary}"]
            else:
                flags = ["-std=c++14", "-Wall", "-Wextra", str(source), "-o", str(binary)]
            built = subprocess.run(compiler + flags, cwd=work, capture_output=True, text=True)
            if built.returncode != 0:
                raise RuntimeError(f"{name} host compilation failed:\n{built.stdout}{built.stderr}")
            result = subprocess.run([str(binary)], capture_output=True, text=True)
            if (result.returncode == 0) != should_pass:
                raise RuntimeError(f"{name}: unexpected exit {result.returncode}:\n{result.stdout}{result.stderr}")
            if not should_pass and result.returncode != 1:
                raise RuntimeError(f"{name}: mutant did not fail a checked expectation: {result.returncode}")
            print(f"{name}: {'PASS' if should_pass else 'REJECTED'}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError) as error:
        print(f"QSA mRoPE host test failed: {error}", file=sys.stderr)
        sys.exit(1)
