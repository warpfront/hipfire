#!/usr/bin/env python3
import sys
from pathlib import Path

if len(sys.argv) != 3:
    raise SystemExit(f"usage: {sys.argv[0]} STDERR EXPECT_ARCH")
path = Path(sys.argv[1])
expected_arch = sys.argv[2]
text = path.read_text(errors="replace")
if f"arch={expected_arch}" not in text:
    raise SystemExit(f"ARCH_ASSERT expected={expected_arch} log={path}")
if "effective_n_chunk = 24/" not in text:
    raise SystemExit(f"C24_ASSERT missing effective_n_chunk=24 log={path}")
print(f"validated {path.name}: arch={expected_arch} c24")
