import json
import re
import sys
from collections import Counter
from pathlib import Path

name = "attention_fp8_e4m3_fa2_gqa_packet_gfx1201"

def census(path):
    text = Path(path).read_text()
    start = text.index(name + ":")
    end_match = re.search(r"^\.Lfunc_end\d+:\s*$", text[start:], re.M)
    if end_match is None:
        raise RuntimeError(f"function end not found in {path}")
    body = text[start:start + end_match.start()]
    ops = []
    for line in body.splitlines():
        match = re.match(r"^\s+((?:v_|s_|ds_|global_)[A-Za-z0-9_]+)", line)
        if match:
            ops.append(match.group(1))
    counts = Counter(ops)
    return {
        "assembly": str(Path(path).resolve()),
        "recognized_instruction_lines": len(ops),
        "wmma": sum(v for k, v in counts.items() if k.startswith("v_wmma_")),
        "fp8_pack": sum(v for k, v in counts.items() if k.startswith("v_cvt_pk_fp8")),
        "global": sum(v for k, v in counts.items() if k.startswith("global_")),
        "lds": sum(v for k, v in counts.items() if k.startswith("ds_")),
        "wait": sum(v for k, v in counts.items() if k.startswith("s_wait")),
        "barrier": sum(v for k, v in counts.items() if k.startswith("s_barrier")),
        "branch": sum(v for k, v in counts.items() if k.startswith("s_branch") or k.startswith("s_cbranch")),
        "ops": dict(sorted(counts.items())),
    }

result = {
    "kernel": name,
    "scope": "complete direct symbol; one recognized ISA line is one packet",
    "baseline": census(sys.argv[1]),
    "candidate": census(sys.argv[2]),
}
Path(sys.argv[3]).write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({k: {f: v for f, v in row.items() if f not in ("assembly", "ops")} for k, row in result.items() if isinstance(row, dict)}, indent=2))
