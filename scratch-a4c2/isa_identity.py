#!/usr/bin/env python3
"""Prove every pre-existing kernels.rs concat!() source that includes a file
this branch touched compiles to instruction-identical code objects at the base
commit and at HEAD for: gfx1201 with no flag (HIPFIRE_G12_A4C2=0, RTN) and
gfx1100/gfx1151 with -DIU4_A4_CANDIDATES=2 (production gfx11 default) and =8.
gfx1201 with -DIU4_A4_CANDIDATES=2 is reported separately: only sources that
call emit_iu4_sidecar_from_producer8 may differ there.

usage: isa_identity.py <worktree> <base_tree> <work_dir>
"""
import concurrent.futures as cf
import pathlib
import re
import subprocess
import sys

wt, base, work = map(pathlib.Path, sys.argv[1:4])
work.mkdir(parents=True, exist_ok=True)
TOUCHED = ("block_i4_128_quant.hip",)
ARCHES = {"gfx1201": ("gfx1201", []),
          "gfx1100": ("gfx1100", ["-DIU4_A4_CANDIDATES=2"]),
          "gfx1151": ("gfx1151", ["-DIU4_A4_CANDIDATES=2"]),
          "gfx1100/c8": ("gfx1100", ["-DIU4_A4_CANDIDATES=8"]),
          "gfx1151/c8": ("gfx1151", ["-DIU4_A4_CANDIDATES=8"]),
          "gfx1201/c2": ("gfx1201", ["-DIU4_A4_CANDIDATES=2"])}
INFO_ONLY = {"gfx1201/c2"}


def consts(tree):
    src = (tree / "crates/rdna-compute/src/kernels.rs").read_text()
    out = {}
    for m in re.finditer(r"pub const (\w+): &str =\s*(concat!\((.*?)\n\);|include_str!\(\"([^\"]+)\"\);)", src, re.S):
        name, body, single = m.group(1), m.group(3), m.group(4)
        text = ""
        files = []
        toks = re.finditer(r'include_str!\("([^"]+)"\)|"((?:[^"\\]|\\.)*)"', body) if body else \
            iter([re.match(r'include_str!\("([^"]+)"\)', f'include_str!("{single}")')])
        for tok in toks:
            if tok.group(1):
                rel = tok.group(1).replace("../../../", "")
                files.append(rel)
                text += (tree / rel).read_text()
            else:
                text += tok.group(2).encode().decode("unicode_escape")
        if any(f.endswith(t) for f in files for t in TOUCHED):
            out[name] = text
    return out


def isa(tag, name, cfg, text):
    arch, flags = ARCHES[cfg]
    d = work / tag / cfg.replace("/", "_")
    d.mkdir(parents=True, exist_ok=True)
    hip = d / f"{name}.{arch}.hip"
    hip.write_text(text)
    co = d / f"{name}.{arch}.hsaco"
    r = subprocess.run(["hipcc", "--genco", f"--offload-arch={arch}", "-O3", "--no-offload-compress",
                        "--rocm-path=/opt/rocm", "--hip-path=/opt/rocm", "-I/opt/rocm/include",
                        *flags, "-o", str(co), str(hip)], capture_output=True, text=True)
    if r.returncode:
        return "COMPILE_FAIL"
    elf = d / f"{name}.{arch}.elf"
    subprocess.run(["/opt/rocm/llvm/bin/clang-offload-bundler", "--type=o", "--unbundle", f"--input={co}",
                    f"--output={elf}", f"--targets=hipv4-amdgcn-amd-amdhsa--{arch}"], check=True)
    dis = subprocess.run(["/opt/rocm/llvm/bin/llvm-objdump", "-d", f"--mcpu={arch}", str(elf)],
                         capture_output=True, text=True, check=True).stdout
    return "\n".join(re.sub(r"\s*//.*$", "", l) for l in dis.splitlines()[3:])


b, h = consts(base), consts(wt)
print(f"base consts touching edited files: {len(b)}; head: {len(h)}; new at head: {sorted(set(h) - set(b))}")
jobs = {}
with cf.ThreadPoolExecutor(16) as ex:
    for name in sorted(b):
        if name not in h:
            print(f"MISSING_AT_HEAD {name}")
            continue
        for arch in ARCHES:
            jobs[(name, arch)] = (ex.submit(isa, "base", name, arch, b[name]),
                                  ex.submit(isa, "head", name, arch, h[name]))
bad = 0
info = []
for (name, arch), (fb, fh) in sorted(jobs.items()):
    rb, rh = fb.result(), fh.result()
    if rb == "COMPILE_FAIL" and rh == "COMPILE_FAIL":
        status = "both-noncompile"
    elif rb == rh:
        status = "IDENTICAL"
    else:
        status = "DIFFERENT"
        if arch in INFO_ONLY:
            info.append(name)
        else:
            bad += 1
    print(f"{status:16s} {arch} {name}")
print("gfx1201/c2 differing (expected: producer sources only):", ", ".join(info))
print("ISA_IDENTITY_PASS" if bad == 0 else f"ISA_IDENTITY_FAIL {bad}")
