#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""Direction ratchets: shrink-only counts of three couplings the design is moving away from.

Emits three metrics for `scripts/leanup-ratchets.sh`, whose ceilings live in
`scripts/leanup-thresholds.txt` (raising one needs a RATCHET-RAISE line,
scripts/ratchet-diff.sh):

    direction_bundle_downcasts   `downcast_ref::<X>` / `downcast_mut::<X>` to a
                                 concrete type in the shared crates (hipfire-generate,
                                 -loader, -daemon, -engine, -runtime). The shared
                                 crates should ask the model through a feature trait
                                 on the carrier, not reach into one arch's bundle.
    direction_gfx_string_matches `== "gfx…"`, `!= "gfx…"`, `matches!(…, "gfx…")` and
                                 `starts_with("gfx` outside the arch-capability owners
                                 (rdna-compute/src/arch_caps.rs and the ISA/toolchain
                                 crates). Ask `ArchCaps` (`has_*` / `is_rdna*`) instead.
    direction_raw_hipfire_env    `std::env::var[_os]("HIPFIRE_…")` (literal, or a
                                 same-file `const X: &str = "HIPFIRE_…"`) outside the
                                 config crate's sanctioned readers (the
                                 CENTRAL_CONFIG_READERS of scripts/check-env-docs.py).
                                 Add a typed key to hipfire-config instead.

Emits a fourth shrink-only metric, `scripts_top_level_files` (files directly in `scripts/`,
not directories): a new script goes into `cargo xtask` as a subcommand, or under a
subdirectory that is archived, never into the top level. Its ceiling is in the same file.

Also emits six informational gauges (`gauge_*`, no ceiling line, never fail; they
feed the Monday digest): `gauge_generate_entry_fns` (`pub fn generate_*` in
hipfire-generate/src), `gauge_generation_routes` (variants of `GenerationRoute`),
`gauge_engine_routes` (entries of `ENGINE_ROUTES`), `gauge_map_md_files` (committed
`crates/**/map.md`), `gauge_typed_arch_compares` (`== GfxArch::X`, `matches!(.., GfxArch::..)`,
`.is_gfx*()` lines outside the arch-capability owners: arch coupling that the string
ratchet stops seeing once strings become typed), `gauge_ci_gate_steps` (`run` lines in scripts/ci-gates.sh)
and `gauge_scripts_total` (every file under `scripts/`, so that moving files into subdirectories is visible).

Only non-test code counts: `crates/*/src/**/*.rs`, excluding build.rs, comments,
`#[cfg(test)]` items and out-of-line `#[cfg(test)] mod x;` files.

Usage:
    scripts/check-direction-ratchets.py --metrics       `name value` lines (leanup-ratchets.sh)
    scripts/check-direction-ratchets.py --lower         rewrite each direction_* / scripts_top_level_files ceiling in
                                                        scripts/leanup-thresholds.txt that sits above
                                                        its measured value down to that value; never raises
    scripts/check-direction-ratchets.py --list METRIC   every counted site, file:line
    scripts/check-direction-ratchets.py --explain [METRIC...]  guidance + per-file counts
"""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location("flags_golden_extract", ROOT / "scripts/flags-golden-extract.py")
_fge = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fge)
match_close = _fge.match_close

# Tokens that can hide a comment opener: raw strings, strings, char literals, then comments.
_LEXEME = re.compile(r'(?<!\w)r(#*)"[\s\S]*?"\1|"(?:\\[\s\S]|[^"\\])*"|\'(?:\\.[^\']*|[^\\\'])\'|//[^\n]*|/\*[\s\S]*?\*/')


def strip_comments(src: str) -> str:
    """Blank `//` and `/* */` comments (newlines kept); literals are left intact."""
    def blank(m: re.Match) -> str:
        text = m.group(0)
        return re.sub(r"[^\n]", " ", text) if text[0] == "/" else text
    return _LEXEME.sub(blank, src)


SHARED_CRATES = ("hipfire-generate", "hipfire-loader", "hipfire-daemon", "hipfire-engine", "hipfire-runtime")
ARCH_OWNER_FILES = {"crates/rdna-compute/src/arch_caps.rs"}
# ISA / toolchain crates: an arch string there names an instruction-set target, which is their job.
ARCH_OWNER_CRATES = re.compile(r"^(peacemaker-.*|railgun.*|redline.*|pm-census|hipfire-detect)$")
CENTRAL_CONFIG_READERS = {
    "crates/hipfire-config/src/lib.rs",
    "crates/hipfire-runtime/src/config.rs",
    "crates/rdna-compute/src/feature_flags.rs",
}

GUIDANCE = {
    "direction_bundle_downcasts": (
        "a shared crate downcasts to a concrete model bundle",
        "Do not add `downcast_ref::<SomeBundle>()` in hipfire-generate/-loader/-daemon/-engine/-runtime.\n"
        "    Put the behaviour behind a feature trait on the carrier (the way `Carrier::topology_caps`\n"
        "    replaced arch-id matches) and call it through the trait object.",
    ),
    "direction_gfx_string_matches": (
        "a GPU-arch string is matched outside the arch-capability owner",
        "Do not compare against \"gfx…\" strings. Query `ArchCaps` (crates/rdna-compute/src/arch_caps.rs):\n"
        "    a `has_<feature>()` capability, or an `is_rdna*`/`is_cdna*` family. A new arch fact goes\n"
        "    into arch_caps.rs once, as an atom/capability, and every caller asks for it by name.",
    ),
    "direction_raw_hipfire_env": (
        "a raw std::env::var(\"HIPFIRE_…\") read outside hipfire-config",
        "Do not read HIPFIRE_* from the environment directly. Add a typed key to the\n"
        "    hipfire-config schema (crates/hipfire-config/src/lib.rs) and read it from ProcessConfig /\n"
        "    FeatureFlags; a developer-only knob uses hipfire_config::developer_var/developer_bool.\n"
        "    Then run scripts/regen-generated.sh to refresh the golden and the env-vars inventory.",
    ),
    "scripts_top_level_files": (
        "a file was added directly under scripts/",
        "Do not add a new top-level script. Add a `cargo xtask` subcommand instead (xtask/), or put a\n"
        "    one-off under a subdirectory that is archived (scripts/analysis/, scripts/lab/). Moving\n"
        "    files out of the top level is the goal: the ceiling only goes down, and\n"
        "    gauge_scripts_total shows whether files merely moved into subdirectories.",
    ),
}

DOWNCAST = re.compile(r"\bdowncast_(?:ref|mut)\s*::\s*<\s*([A-Za-z_][\w:]*)")
GFX_CMP = re.compile(r'(?:==|!=)\s*"gfx|"gfx[^"\n]*"\s*(?:==|!=)|starts_with\(\s*"gfx')
TYPED_ARCH_CMP = re.compile(r"(?:==|!=)\s*GfxArch::\w+|GfxArch::\w+\s*(?:==|!=)|\.is_gfx\w*\(\)")
MATCHES = re.compile(r"\bmatches!\s*\(")
ENV_LIT = re.compile(r'\benv::var(?:_os)?\(\s*"HIPFIRE_[A-Z0-9_]+"\s*\)')
ENV_CALL = re.compile(r"\benv::var(?:_os)?\(\s*([A-Z][A-Z0-9_]*)\s*\)")
ENV_NAME_CONST = re.compile(r'\b(?:const|static)\s+([A-Z][A-Z0-9_]*)\s*:\s*&(?:\'static\s+)?str\s*=\s*"HIPFIRE_')
CFG_TEST = re.compile(r"#\[cfg\((?:test|all\([^\]]*\btest\b[^\]]*\))\)\]\s*(?:#\[[^\]]*\]\s*)*")
OUTLINE_MOD = re.compile(r"(?:pub(?:\([a-z]+\))?\s+)?mod\s+(\w+)\s*;")


def production_text(path: Path, raw: str, test_files: set[Path]) -> str:
    """Source with comments and `#[cfg(test)]` items blanked (newlines kept, so line numbers hold)."""
    src = strip_comments(raw)
    out, pos = [], 0
    for m in CFG_TEST.finditer(src):
        if m.start() < pos:
            continue
        rest = src[m.end():]
        mod = OUTLINE_MOD.match(rest)
        if mod:
            base = path.parent if path.name in ("lib.rs", "main.rs", "mod.rs") else path.with_suffix("")
            test_files.update({base / f"{mod.group(1)}.rs", base / mod.group(1) / "mod.rs"})
            end = m.end() + mod.end()
        else:
            brace, semi = src.find("{", m.end()), src.find(";", m.end())
            if brace < 0 and semi < 0:
                continue
            end = semi + 1 if brace < 0 or (0 <= semi < brace) else match_close(src, brace)
        out.append(src[pos:m.start()])
        out.append(re.sub(r"[^\n]", " ", src[m.start():end]))
        pos = end
    out.append(src[pos:])
    return "".join(out)


def sources() -> list[tuple[str, str]]:
    test_files: set[Path] = set()
    texts = {}
    for path in sorted(ROOT.glob("crates/*/src/**/*.rs")):
        if path.name == "build.rs":
            continue
        raw = path.read_text(encoding="utf-8", errors="ignore")
        if any(token in raw for token in ("downcast_", "gfx", "HIPFIRE_", "fn generate_", "GfxArch::", ".is_gfx", "enum GenerationRoute", "ENGINE_ROUTES")):
            texts[path] = production_text(path, raw, test_files)
        elif "cfg(" in raw:  # only the out-of-line test modules matter here
            for m in CFG_TEST.finditer(raw):
                mod = OUTLINE_MOD.match(raw, m.end())
                if mod:
                    base = path.parent if path.name in ("lib.rs", "main.rs", "mod.rs") else path.with_suffix("")
                    test_files.update({base / f"{mod.group(1)}.rs", base / mod.group(1) / "mod.rs"})
    return [(p.relative_to(ROOT).as_posix(), t) for p, t in texts.items() if p not in test_files]


def line_of(text: str, idx: int) -> int:
    return text.count("\n", 0, idx) + 1


def sites(texts: list[tuple[str, str]] | None = None) -> dict[str, list[str]]:
    found = {name: [] for name in GUIDANCE}
    found["scripts_top_level_files"] = sorted(
        f"scripts/{p.name}:1" for p in (ROOT / "scripts").iterdir() if p.is_file()
    )
    for rel, text in (texts if texts is not None else sources()):
        crate = rel.split("/")[1]
        if crate in SHARED_CRATES:
            for m in DOWNCAST.finditer(text):
                if not re.fullmatch(r"[A-Z]", m.group(1)):  # a bare generic parameter is not a bundle
                    found["direction_bundle_downcasts"].append(f"{rel}:{line_of(text, m.start())}")
        if '"gfx' in text and rel not in ARCH_OWNER_FILES and not ARCH_OWNER_CRATES.match(crate):
            hits = {line_of(text, m.start()) for m in GFX_CMP.finditer(text)}
            for m in MATCHES.finditer(text):
                if '"gfx' in text[m.end():match_close(text, m.end() - 1)]:
                    hits.add(line_of(text, m.start()))
            found["direction_gfx_string_matches"] += [f"{rel}:{n}" for n in sorted(hits)]
        if rel not in CENTRAL_CONFIG_READERS:
            consts = set(ENV_NAME_CONST.findall(text))
            for m in ENV_LIT.finditer(text):
                found["direction_raw_hipfire_env"].append(f"{rel}:{line_of(text, m.start())}")
            for m in ENV_CALL.finditer(text):
                if m.group(1) in consts:
                    found["direction_raw_hipfire_env"].append(f"{rel}:{line_of(text, m.start())}")
    return found


def gauges(texts: list[tuple[str, str]]) -> dict[str, int]:
    """Informational counts; `texts` is `sources()`."""
    entry_fns = 0
    routes = 0
    engine_routes = 0
    for rel, text in texts:
        if rel.startswith("crates/hipfire-generate/src/"):
            entry_fns += len(re.findall(r"\bpub(?:\([a-z]+\))?\s+fn\s+generate_\w+", text))
        if rel == "crates/hipfire-generate/src/ar.rs":
            m = re.search(r"\benum\s+GenerationRoute\s*\{", text)
            if m:
                body = text[m.end():match_close(text, m.end() - 1)]
                routes = len(re.findall(r"^\s*[A-Z]\w*\s*(?:=[^,]*)?,", body, re.M))
        if rel == "crates/hipfire-generate/src/engine/mod.rs":
            m = re.search(r"\bconst\s+ENGINE_ROUTES\b[^=]*=\s*&\[", text)
            if m:
                body = text[m.end():text.index("]", m.end())]
                engine_routes = len(re.findall(r"GenerationRoute::\w+", body))
    typed = 0
    for rel, text in texts:
        if rel in ARCH_OWNER_FILES or ARCH_OWNER_CRATES.match(rel.split("/")[1]):
            continue
        hits = {line_of(text, m.start()) for m in TYPED_ARCH_CMP.finditer(text)}
        for m in MATCHES.finditer(text):
            if "GfxArch::" in text[m.end():match_close(text, m.end() - 1)]:
                hits.add(line_of(text, m.start()))
        typed += len(hits)
    ci = (ROOT / "scripts/ci-gates.sh").read_text(encoding="utf-8", errors="ignore")
    return {
        "gauge_generate_entry_fns": entry_fns,
        "gauge_generation_routes": routes,
        "gauge_engine_routes": engine_routes,
        "gauge_map_md_files": sum(1 for p in ROOT.glob("crates/**/map.md") if "target" not in p.relative_to(ROOT).parts),
        "gauge_typed_arch_compares": typed,
        "gauge_ci_gate_steps": len(re.findall(r"^\s*run\s", ci, re.M)),
        "gauge_scripts_total": sum(1 for p in (ROOT / "scripts").rglob("*") if p.is_file() and "__pycache__" not in p.parts),
    }


def lower(found: dict[str, list[str]]) -> int:
    """Rewrite `direction_* / scripts_top_level_files <= N` ceilings above the measured count down to it. Never raises."""
    path = ROOT / "scripts/leanup-thresholds.txt"
    out, changed = [], 0
    for line in path.read_text(encoding="utf-8").splitlines(keepends=True):
        m = re.match(r"((?:direction_\w+|scripts_top_level_files))(\s+<=\s+)(\d+)(\s*)$", line.rstrip("\n"))
        if m and m.group(1) in found and len(found[m.group(1)]) < int(m.group(3)):
            now = len(found[m.group(1)])
            print(f"{m.group(1)}: {m.group(3)} -> {now}")
            line = f"{m.group(1)}{m.group(2)}{now}\n"
            changed += 1
        out.append(line)
    if changed:
        path.write_text("".join(out), encoding="utf-8")
    else:
        print("nothing to lower")
    return 0


def main(argv: list[str]) -> int:
    if argv == ["--metrics"]:
        texts = sources()
        found = sites(texts)
        for name, where in found.items():
            print(f"{name} {len(where)}")
        for name, n in gauges(texts).items():
            print(f"{name} {n}")
        return 0
    found = sites()
    if argv == ["--lower"]:
        return lower(found)
    if len(argv) == 2 and argv[0] == "--list" and argv[1] in found:
        print("\n".join(found[argv[1]]))
        return 0
    if argv[:1] == ["--explain"] and all(name in found for name in argv[1:]):
        for name in argv[1:] or list(found):
            where = found[name]
            what, fix = GUIDANCE[name]
            print(f"{name} = {len(where)}: {what}.")
            print(f"    {fix}")
            per_file: dict[str, int] = {}
            for site in where:
                f = site.rsplit(":", 1)[0]
                per_file[f] = per_file.get(f, 0) + 1
            top = sorted(per_file.items(), key=lambda kv: (-kv[1], kv[0]))[:8]
            print("    most sites: " + ", ".join(f"{f} ({n})" for f, n in top))
            print(f"    every site: scripts/check-direction-ratchets.py --list {name}")
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
