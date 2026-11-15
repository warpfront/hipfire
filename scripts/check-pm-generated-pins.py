#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# hipfire — see LICENSE and NOTICE in the project root.
"""Check the provenance pins of the committed CDNA tables without the AMD XML, and the generated-file lists.

`crates/peacemaker-cdna/src/generated_cdna{3,4}.rs` are written by
`crates/peacemaker-cdna/scripts/generate-cdna-tables.py` from AMD's ISA XML (not in the repo)
and the committed LLVM receipts `tools/cdna{3,4}-llvm.json`. Each table pins its inputs twice,
in the `// ...` header and in `TABLE.provenance`. A generator or receipt edited without
regenerating leaves a stale pin, and this check catches that:

  * generator sha256 (header and TABLE) == sha256 of the current generator script;
  * receipt path and sha256 (header and TABLE) == the arch's receipt file and its sha256;
  * XML sha256 (header and TABLE) == the generator's pin for that XML == the receipt's xml_sha256;
  * the header records the sha256 of the (uncommitted) row manifest;
  * the repo `.ignore` search exclusions equal the `linguist-generated=true` paths of
    `.gitattributes`, and every listed path is an existing file.

The full byte check needs the XML: `generate-cdna-tables.py --check --xml-dir DIR`.

usage: check-pm-generated-pins.py [--self-test]
exit 0 = pins current, 1 = stale or malformed pins (each printed).
"""
import hashlib
import importlib.util
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
CRATE = ROOT / "crates/peacemaker-cdna"
GENERATOR = CRATE / "scripts/generate-cdna-tables.py"
SHA = "[0-9a-f]{64}"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def load_generator():
    spec = importlib.util.spec_from_file_location("generate_cdna_tables", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def one(pattern, text, what, errors):
    found = re.findall(pattern, text, re.M)
    if len(found) != 1:
        errors.append(f"{what}: expected exactly one match of {pattern!r}, found {len(found)}")
        return None
    return found[0]


def check_table(name, rust, arch, xml_pin, generator_sha, receipt_bytes):
    """Errors for one generated table (`rust` text) against its current inputs."""
    errors = []
    where = f"src/{name}.rs"
    head_xml = one(rf"^// XML sha256  : ({SHA})$", rust, f"{where} header XML sha256", errors)
    head_receipt = one(rf"^// Receipt     : (\S+) \(sha256 ({SHA})\)$", rust, f"{where} header receipt", errors)
    head_gen = one(rf"^// Generator   : \S+ sha256 ({SHA})$", rust, f"{where} header generator", errors)
    one(rf"^// Manifest    : xml-manifest-{arch['key']}\.json sha256 ({SHA}) ", rust, f"{where} header manifest", errors)
    table_xml = one(rf'^        xml_sha256: "({SHA})",$', rust, f"{where} TABLE xml_sha256", errors)
    table_file = one(r'^        receipt_file: "([^"]+)",$', rust, f"{where} TABLE receipt_file", errors)
    table_receipt = one(rf'^        receipt_sha256: "({SHA})",$', rust, f"{where} TABLE receipt_sha256", errors)
    table_gen = one(rf'^        generator_sha256: "({SHA})",$', rust, f"{where} TABLE generator_sha256", errors)
    if errors:
        return errors
    receipt_sha = sha256(receipt_bytes)
    try:
        receipt_xml = json.loads(receipt_bytes)["xml_sha256"]
    except (ValueError, KeyError, TypeError) as err:
        return [f"{arch['receipt']}: unreadable receipt ({err})"]
    expect = [
        ("generator sha256 (header)", head_gen, generator_sha),
        ("generator sha256 (TABLE)", table_gen, generator_sha),
        ("receipt path (header)", head_receipt[0], arch["receipt"]),
        ("receipt path (TABLE)", table_file, arch["receipt"]),
        ("receipt sha256 (header)", head_receipt[1], receipt_sha),
        ("receipt sha256 (TABLE)", table_receipt, receipt_sha),
        ("XML sha256 (header)", head_xml, xml_pin),
        ("XML sha256 (TABLE)", table_xml, xml_pin),
        (f"{arch['receipt']} xml_sha256", receipt_xml, xml_pin),
    ]
    for what, got, want in expect:
        if got != want:
            errors.append(f"{where}: {what} is {got}, current input is {want}: regenerate with "
                          f"crates/peacemaker-cdna/scripts/generate-cdna-tables.py --xml-dir <AMD XML dir>")
    return errors


def check_lists(ignore_text, attributes_text, exists):
    """Errors unless .ignore's paths equal .gitattributes' linguist-generated paths, all existing."""
    ignored = {line.lstrip("/") for line in ignore_text.splitlines() if line.strip() and not line.startswith("#")}
    generated = {line.split()[0] for line in attributes_text.splitlines()
                 if not line.startswith("#") and "linguist-generated=true" in line.split()[1:]}
    errors = [f".ignore lists {p} but .gitattributes does not mark it linguist-generated"
              for p in sorted(ignored - generated)]
    errors += [f".gitattributes marks {p} linguist-generated but .ignore does not exclude it"
               for p in sorted(generated - ignored)]
    errors += [f"{p}: listed as generated but not a file" for p in sorted(ignored | generated) if not exists(p)]
    return errors


def check_tree():
    generator = load_generator()
    generator_sha = sha256(GENERATOR.read_bytes())
    errors = []
    for arch in generator.ARCHES:
        rust = (CRATE / "src" / f"{arch['module']}.rs").read_text(encoding="utf-8")
        receipt = (CRATE / arch["receipt"]).read_bytes()
        errors += check_table(arch["module"], rust, arch, generator.PINS[arch["xml"]], generator_sha, receipt)
    errors += check_lists((ROOT / ".ignore").read_text(encoding="utf-8"),
                          (ROOT / ".gitattributes").read_text(encoding="utf-8"),
                          lambda p: (ROOT / p).is_file())
    return errors


def self_test():
    """Each planted defect must be reported; the committed tree must pass."""
    clean = check_tree()
    if clean:
        print("self-test: the committed tree does not pass:", *clean, sep="\n  ")
        return 1
    generator = load_generator()
    generator_sha = sha256(GENERATOR.read_bytes())
    arch = generator.ARCHES[0]
    rust = (CRATE / "src" / f"{arch['module']}.rs").read_text(encoding="utf-8")
    receipt = (CRATE / arch["receipt"]).read_bytes()
    pin = generator.PINS[arch["xml"]]
    other = "0" * 64
    plants = {
        "edited generator": (rust, other, receipt),
        "edited receipt": (rust, generator_sha, receipt + b" "),
        "stale TABLE generator pin": (rust.replace(f'generator_sha256: "{generator_sha}"',
                                                   f'generator_sha256: "{other}"'), generator_sha, receipt),
        "stale header receipt pin": (rust.replace(f"(sha256 {sha256(receipt)})", f"(sha256 {other})"),
                                     generator_sha, receipt),
        "missing manifest pin": (re.sub(r"^// Manifest .*\n", "", rust, flags=re.M), generator_sha, receipt),
    }
    failed = 0
    for name, (text, gen_sha, rec) in plants.items():
        if not check_table(arch["module"], text, arch, pin, gen_sha, rec):
            print(f"self-test: planted defect not caught: {name}")
            failed = 1
    ignore = (ROOT / ".ignore").read_text(encoding="utf-8")
    attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    first = next(line for line in ignore.splitlines() if line.startswith("/"))
    list_plants = {
        "path excluded from search but not marked generated": (ignore + "/crates/x.rs\n", attributes),
        "path marked generated but still searched": (ignore.replace(first + "\n", ""), attributes),
        "listed path that does not exist": (ignore + "/crates/x.rs\n", attributes + "crates/x.rs linguist-generated=true\n"),
    }
    for name, (ig, attr) in list_plants.items():
        if not check_lists(ig, attr, lambda p: (ROOT / p).is_file()):
            print(f"self-test: planted defect not caught: {name}")
            failed = 1
    print("self-test: ok" if not failed else "self-test: FAILED")
    return failed


def main():
    if sys.argv[1:] == ["--self-test"]:
        return self_test()
    if sys.argv[1:]:
        print(__doc__)
        return 2
    errors = check_tree()
    if errors:
        print("stale generated-file pins or lists:", *errors, sep="\n  ")
        return 1
    print("CDNA table pins current (generator, receipts, XML); .ignore == linguist-generated list.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
