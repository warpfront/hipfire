#!/usr/bin/env python3
"""Extract every concat!() constant named on the command line from
crates/rdna-compute/src/kernels.rs and expand it exactly as rustc does
(string literals + include_str! of kernels/src files), writing <out>/<CONST>.hip.

usage: gen_prod.py <worktree> <out_dir> CONST [CONST ...]
"""
import pathlib
import re
import sys

wt = pathlib.Path(sys.argv[1])
out = pathlib.Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
src = (wt / "crates/rdna-compute/src/kernels.rs").read_text()
for name in sys.argv[3:]:
    m = re.search(r"pub const " + name + r": &str = concat!\((.*?)\n\);", src, re.S)
    if not m:
        sys.exit(f"{name}: not found")
    text = ""
    for tok in re.finditer(r'include_str!\("([^"]+)"\)|"((?:[^"\\]|\\.)*)"', m.group(1)):
        if tok.group(1):
            rel = tok.group(1).replace("../../../", "")
            text += (wt / rel).read_text()
        else:
            text += tok.group(2).encode().decode("unicode_escape")
    (out / f"{name}.hip").write_text(text)
    print(name, len(text))
