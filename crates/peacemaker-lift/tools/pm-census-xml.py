#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

"""Export pinned RDNA3, RDNA3.5 or RDNA4 XML instruction encodings for offline census.

Usage: python3 crates/peacemaker-lift/tools/pm-census-xml.py <amdgpu_isa_*.xml> <output.tsv>
The exported header records and verifies the architecture-specific XML SHA.
"""
import hashlib
import pathlib
import sys
import xml.etree.ElementTree as ET

PINS = {
    "amdgpu_isa_rdna3.xml": "6eee5f8737172adf08c0e7d5994ea916e9555b9c23ccb9ec80d9ee38678733b4",
    "amdgpu_isa_rdna3_5.xml": "c36b6d79b1e940d74107221c985f5a7fde248025da251d2c6ef756c4cd31391a",
    "amdgpu_isa_rdna4.xml": "f8a290c8471e26a1071b08b61a33e4d9efa46ec6cedcdb8e5ba57d7969b60692",
}


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: pm-census-xml.py <amdgpu_isa_*.xml> <output.tsv>")
    source, output = map(pathlib.Path, sys.argv[1:])
    data = source.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != PINS.get(source.name):
        raise SystemExit(f"MR-ISA XML {source.name} SHA-256 {digest} does not equal its pinned digest")
    root = ET.fromstring(data)
    encodings = {}
    for encoding in root.findall("ISA/Encodings/Encoding"):
        name = encoding.findtext("EncodingName")
        encodings[name] = (
            int(encoding.findtext("BitCount")),
            int(encoding.findtext("EncodingIdentifierMask"), 2),
            sorted({int(node.text, 2) for node in encoding.findall("EncodingIdentifiers/EncodingIdentifier")}),
        )
    rows = [f"# xml_sha256={digest}", "encoding\topcode\tbits\tspelling\tid_mask\tids"]
    for instruction in root.findall("ISA/Instructions/Instruction"):
        name = instruction.findtext("InstructionName").lower()
        for variant in instruction.findall("InstructionEncodings/InstructionEncoding"):
            encoding = variant.findtext("EncodingName")
            bits, mask, identifiers = encodings[encoding]
            fields = (encoding, variant.findtext("Opcode"), str(bits), name,
                      f"{mask:x}", ",".join(f"{identifier:x}" for identifier in identifiers))
            rows.append("\t".join(fields))
    output.write_text("\n".join(rows) + "\n")
    print(f"{len(rows)-2} XML instruction-encoding entries -> {output}")


if __name__ == "__main__":
    main()
