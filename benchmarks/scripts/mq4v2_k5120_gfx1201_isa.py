#!/usr/bin/env python3
"""Archive the actual capture-bound code objects, metadata and disassembly."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--llvm", type=Path, required=True)
    args = parser.parse_args()
    out = args.root / "isa"
    out.mkdir(exist_ok=True)
    identities = []
    for flag in (0, 1):
        report = json.loads((args.root / f"replay-{flag}.json").read_text())
        sequence = report["decode"]["captures"][0]["redline_capture"]["sequence"]
        symbol = "fused_gate_up_mq4g256v2" + ("_k5120_gfx1201" if flag else "")
        paths = {launch["artifact"] for launch in sequence if launch["kernel"] == symbol}
        if len(paths) != 1 or None in paths:
            raise RuntimeError(f"expected exactly one captured object for {symbol}: {paths}")
        source = Path(paths.pop())
        obj = out / f"arm-{flag}.hsaco"
        shutil.copyfile(source, obj)
        index = source.with_suffix(".index.json")
        if index.is_file():
            shutil.copyfile(index, out / f"arm-{flag}.index.json")
        elf = out / f"arm-{flag}.elf"
        if obj.read_bytes()[:4] == b"\x7fELF":
            shutil.copyfile(obj, elf)
        else:
            subprocess.run([str(args.llvm / "clang-offload-bundler"), "--unbundle", "--type=o",
                            "--targets=hipv4-amdgcn-amd-amdhsa--gfx1201",
                            f"--input={obj}", f"--output={elf}"], check=True)
        for tool, option, suffix in (("llvm-readobj", "--notes", "notes"), ("llvm-objdump", "-d", "isa")):
            result = subprocess.check_output([str(args.llvm / tool), option, str(elf)], text=True)
            (out / f"arm-{flag}.{suffix}").write_text(result)
        identities.append(dict(flag=flag, symbol=symbol, source=str(source),
                               sha256=hashlib.sha256(obj.read_bytes()).hexdigest()))
    (out / "objects.json").write_text(json.dumps(identities, indent=2) + "\n")
    print(json.dumps(identities, indent=2))


if __name__ == "__main__":
    main()
