#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

"""Generate the complete CDNA3/CDNA4 instruction tables from AMD's ISA XML.

Inputs (all pinned by SHA-256, a mismatch is a hard error):
  * ``amdgpu_isa_cdna3.xml`` / ``amdgpu_isa_cdna4.xml`` (AMD machine-readable
    ISA, MIT; GPUOpen ``amdgpu_isa`` release 2026-02-20, schema 1.1.1).
  * ``tools/cdna3-llvm.json`` / ``tools/cdna4-llvm.json``: the per-row LLVM
    cross-check receipts written by ``tools/crosscheck_xml.py``. A row is
    ``Verification::Verified`` only if its receipt row says so; anything else
    stays in the table as ``Unverified(<receipt reason>)`` (decoded, never
    encoded).

Outputs:
  * ``src/generated_cdna3.rs`` / ``src/generated_cdna4.rs``: every encoding,
    field (with padding), condition, instruction, operand type, data format,
    form (instruction x encoding x condition) and operand of the XML, as Rust
    ``static`` tables consumed by ``src/xml.rs``.
  * ``tools/xml-manifest-cdna3.json`` / ``tools/xml-manifest-cdna4.json``:
    machine-readable row manifest (row id/name/encoding/condition/opcode/
    bit_count/identifier/fields/operands/verification) with a structured seed
    (named field values and little-endian bytes) that satisfies the row's XML
    condition and decodes back to that row.

Also usable as a module: ``load_model(xml_path)`` returns the parsed model and
``py_decode(model, words)`` the reference decoder the Rust runtime mirrors.

Run from anywhere:  scripts/generate-cdna-tables.py --xml-dir <dir with the
two XMLs>   (``--check`` regenerates in memory and fails if a file differs).

Structural cases the generator refuses (instead of faking support) are listed
in ``Unsupported``; none occurs in the pinned XMLs except as reported in the
run summary (e.g. the CDNA4 ``ENC_VOP3PX2`` second-half ``ENCODING`` field,
which no XML identifier pins).
"""
import argparse
import hashlib
import json
import os
import pathlib
import re
import sys
import xml.etree.ElementTree as ET

CRATE = pathlib.Path(__file__).resolve().parent.parent
GENERATOR = "scripts/generate-cdna-tables.py"

PINS = {
    "amdgpu_isa_cdna3.xml": "af24feb95c9230a87f694c814b0035e7448a57f8503b059f0282dba1e5f20d9c",
    "amdgpu_isa_cdna4.xml": "bb2a96f3db5c67d7c9b90e34559045c272f3536acdb03d16bc455f2f5b0a1597",
}

ARCHES = [
    dict(key="cdna3", xml="amdgpu_isa_cdna3.xml", isa="Cdna3", gfx="gfx942",
         module="generated_cdna3", receipt="tools/cdna3-llvm.json"),
    dict(key="cdna4", xml="amdgpu_isa_cdna4.xml", isa="Cdna4", gfx="gfx950",
         module="generated_cdna4", receipt="tools/cdna4-llvm.json"),
]

MAX_FIELDS = 24  # Decoded::values in src/xml.rs
MAX_DWORDS = 4


class Unsupported(Exception):
    """An XML structure the Rust runtime cannot represent (never faked)."""


# --------------------------------------------------------------- XML parsing

def num(text):
    text = text.strip()
    return int(text, 16) if text.lower().startswith("0x") else int(text, 10)


def radix_num(node):
    return int(node.text.strip(), int(node.get("Radix", "10")))


def text(node, tag, default=""):
    child = node.find(tag)
    if child is None or child.text is None:
        return default
    return child.text.strip()


class Field:
    def __init__(self, node):
        self.name = text(node, "FieldName")
        self.description = text(node, "Description")
        if node.get("IsConditional", "false").lower() == "true":
            raise Unsupported(f"conditional field {self.name}: per-field conditions are not in the schema this generator knows")
        ranges = []
        pad_bits = 0
        pad_value = 0
        for rg in sorted(node.findall("BitLayout/Range"), key=lambda r: int(r.get("Order", "0"))):
            ranges.append((num(rg.find("BitOffset").text), num(rg.find("BitCount").text)))
            pad = rg.find("Padding")
            if pad is not None:
                if pad_bits:
                    raise Unsupported(f"field {self.name}: padding on several ranges")
                pad_bits = num(pad.find("BitCount").text)
                pad_value = radix_num(pad.find("Value"))
        self.ranges = ranges
        self.bits = sum(c for _, c in ranges)
        self.pad_bits = pad_bits
        self.pad_value = pad_value
        self.predefined = [
            (text(p, "Name"), text(p, "Description"), num(p.find("Value").text))
            for p in node.findall("FieldPredefinedValues/PredefinedValue")
        ]
        self.mask = 0
        for off, cnt in ranges:
            m = ((1 << cnt) - 1) << off
            if self.mask & m:
                raise Unsupported(f"field {self.name}: overlapping ranges")
            self.mask |= m
        if self.bits > 32 or self.pad_bits + self.bits > 32:
            raise Unsupported(f"field {self.name}: wider than 32 bits")

    def extract(self, word):
        v = 0
        shift = 0
        for off, cnt in self.ranges:
            v |= ((word >> off) & ((1 << cnt) - 1)) << shift
            shift += cnt
        return v

    def place(self, value):
        word = 0
        for off, cnt in self.ranges:
            word |= (value & ((1 << cnt) - 1)) << off
            value >>= cnt
        return word


class Encoding:
    def __init__(self, node, index):
        self.index = index
        self.name = text(node, "EncodingName")
        self.order = int(node.get("Order", "0"))
        self.description = text(node, "Description")
        self.bit_count = num(node.find("BitCount").text)
        if self.bit_count % 32 or self.bit_count // 32 > MAX_DWORDS:
            raise Unsupported(f"{self.name}: bit count {self.bit_count}")
        self.dwords = self.bit_count // 32
        self.id_mask = radix_num(node.find("EncodingIdentifierMask"))
        self.ids = sorted({radix_num(i) for i in node.findall("EncodingIdentifiers/EncodingIdentifier")})
        self.xml_id_mask = self.id_mask
        self.xml_ids = list(self.ids)
        self.correction = ""
        if self.name == "ENC_VOP3PX2":
            # The pinned CDNA4 XML places the second-half MFMA identifier in
            # word 0 despite placing ENCODING/OP at bits 87/80. LLVM decodes
            # the actual 128-bit instructions with the d3ac extension prefix.
            # Decode-only correction; both rows remain receipt-unverified.
            # Independent byte/text evidence: tools/cdna4-vop3px2-llvm.json.
            if self.id_mask != 0xffff0000 or self.ids != [0xd3ad0000, 0xd3ae0000]:
                raise Unsupported("ENC_VOP3PX2 XML discrepancy changed; revisit the decode correction")
            self.id_mask = (self.xml_id_mask << 64) | 0xffff0000
            self.ids = [(i << 64) | 0xd3ac0000 for i in self.xml_ids]
            self.correction = "CDNA4 identifier shifted to bits 80..95; d3ac prefix at bits 16..31 (LLVM decode receipt, still unverified)"
        self.fields = [Field(f) for f in node.findall("MicrocodeFormat/BitMap/Field")]
        if len(self.fields) > MAX_FIELDS:
            raise Unsupported(f"{self.name}: {len(self.fields)} fields > {MAX_FIELDS}")
        names = [f.name for f in self.fields]
        if len(set(names)) != len(names):
            raise Unsupported(f"{self.name}: duplicate field names")
        self.field_index = {f.name: i for i, f in enumerate(self.fields)}
        self.covered = 0
        for f in self.fields:
            if self.covered & f.mask:
                raise Unsupported(f"{self.name}: overlapping fields")
            self.covered |= f.mask
        full = (1 << self.bit_count) - 1
        if self.covered & ~full:
            raise Unsupported(f"{self.name}: field beyond bit count")
        self.reserved = full & ~self.covered
        if self.id_mask & ~self.covered:
            raise Unsupported(f"{self.name}: identifier mask outside the field layout")
        for i in self.ids:
            if i & ~self.id_mask:
                raise Unsupported(f"{self.name}: identifier bits outside its mask")
        if "OP" not in self.field_index:
            raise Unsupported(f"{self.name}: no OP field")
        op = self.fields[self.field_index["OP"]]
        if len(op.ranges) != 1:
            raise Unsupported(f"{self.name}: split OP field")
        self.op_offset, self.op_width = op.ranges[0]
        self.node = node
        self.conditions = []  # filled by Model (needs all encodings)


class Model:
    pass


# Condition expressions -----------------------------------------------------

def compile_cond(model, enc, node):
    """XML ``CondtionExpression`` -> tuple tree over this encoding's fields."""
    def field_ref(dd):
        ids = [i.get("val") for i in dd.findall("id")]
        sub = [c for c in dd if c.tag == "op"]
        if sub:  # nested (.fieldderef (.fieldderef INST X) FIELD)
            inner = [i.get("val") for i in sub[0].findall("id")]
            if inner[0] != "INST" or len(inner) != 2 or len(ids) != 1:
                raise Unsupported(f"{enc.name}: unrecognised field reference")
            owner, fname = inner[1], ids[0]
        else:
            raise Unsupported(f"{enc.name}: unrecognised field reference")
        owner_enc = model.by_name.get("ENC_" + owner) or model.by_name.get(owner)
        if owner_enc is None or fname not in owner_enc.field_index:
            raise Unsupported(f"{enc.name}: reference INST.{owner}.{fname} resolves to no field")
        target = owner_enc.fields[owner_enc.field_index[fname]]
        for i, f in enumerate(enc.fields):
            if f.ranges == target.ranges:
                return i
        raise Unsupported(f"{enc.name}: referenced field {owner}.{fname} has no equal-layout field in this encoding")

    def operands(n):
        return [c for c in n if c.tag != "ty"]

    def value_of(n):
        if n.tag == "lit":
            return ("lit", num(n.get("val")))
        if n.tag == "op" and n.get("type") == ".fieldderef":
            return ("field", field_ref(n))
        if n.tag == "op" and n.get("type") == ".cons_array":
            return ("array", [num(c.get("val")) for c in operands(n)])
        raise Unsupported(f"{enc.name}: unsupported condition operand {n.tag} {n.get('type')}")

    def comp(n):
        if n.tag == "lit":
            return ("lit", num(n.get("val")))
        if n.tag != "op":
            raise Unsupported(f"{enc.name}: unsupported condition node {n.tag}")
        kind = n.get("type")
        args = operands(n)
        if kind in ("==", "!="):
            a, b = value_of(args[0]), value_of(args[1])
            if a[0] == "lit" and b[0] == "field":
                a, b = b, a
            if a[0] != "field" or b[0] != "lit":
                raise Unsupported(f"{enc.name}: {kind} needs field vs literal")
            return ("eq" if kind == "==" else "ne", a[1], b[1])
        if kind == "&&":
            out = comp(args[0])
            for extra in args[1:]:
                out = ("and", out, comp(extra))
            return out
        if kind == ".notwithin":
            a, b = value_of(args[0]), value_of(args[1])
            if a[0] != "field" or b[0] != "array":
                raise Unsupported(f"{enc.name}: .notwithin needs field vs array")
            return ("notwithin", a[1], b[1])
        raise Unsupported(f"{enc.name}: unsupported condition operator {kind!r} (extend Expr in src/xml.rs and this generator)")

    return comp(node[0])


def holds(expr, values):
    kind = expr[0]
    if kind == "lit":
        return expr[1] != 0
    if kind == "eq":
        return values[expr[1]] == expr[2]
    if kind == "ne":
        return values[expr[1]] != expr[2]
    if kind == "and":
        return holds(expr[1], values) and holds(expr[2], values)
    if kind == "notwithin":
        return values[expr[1]] not in expr[2]
    raise AssertionError(kind)


def expr_fields(expr):
    kind = expr[0]
    if kind in ("eq", "ne", "notwithin"):
        return {expr[1]}
    if kind == "and":
        return expr_fields(expr[1]) | expr_fields(expr[2])
    return set()


def expr_rust(expr):
    kind = expr[0]
    if kind == "lit":
        return f"Expr::Lit({expr[1]})"
    if kind == "eq":
        return f"Expr::Eq({expr[1]}, {expr[2]})"
    if kind == "ne":
        return f"Expr::Ne({expr[1]}, {expr[2]})"
    if kind == "and":
        return f"Expr::Both(&{expr_rust(expr[1])}, &{expr_rust(expr[2])})"
    if kind == "notwithin":
        return f"Expr::NotWithin({expr[1]}, &[{', '.join(str(v) for v in expr[2])}])"
    raise AssertionError(kind)


def expr_text(expr):
    kind = expr[0]
    if kind == "lit":
        return str(expr[1])
    if kind == "eq":
        return f"F{expr[1]} == {expr[2]}"
    if kind == "ne":
        return f"F{expr[1]} != {expr[2]}"
    if kind == "and":
        return f"({expr_text(expr[1])}) && ({expr_text(expr[2])})"
    if kind == "notwithin":
        return f"F{expr[1]} not in {expr[2]}"
    raise AssertionError(kind)


# Operand types -------------------------------------------------------------

class OperandType:
    def __init__(self, node):
        self.name = text(node, "OperandTypeName")
        self.description = text(node, "Description")
        self.partitioned = node.get("IsPartitioned", "false").lower() == "true"
        self.subtypes = [s.text.strip() for s in node.findall("Subtypes/Subtype")]
        self.values = [
            (text(p, "Name"), text(p, "Description"), num(p.find("Value").text))
            for p in node.findall("OperandPredefinedValues/PredefinedValue")
        ]
        self.fields = [Field(f) for f in node.findall("MicrocodeFormat/BitMap/Field")]
        self.ranges = []
        for v in sorted({v for _, _, v in self.values}):
            if self.ranges and self.ranges[-1][1] + 1 == v:
                self.ranges[-1][1] = v
            else:
                self.ranges.append([v, v])
        self.ranges = [tuple(r) for r in self.ranges]

    def accepts(self, value, modulo):
        """XML value set membership of ``value`` projected through ``modulo``."""
        value %= modulo
        for lo, hi in self.ranges:
            if lo + (value + modulo - lo % modulo) % modulo <= hi:
                return True
        return False


def variant(name):
    assert name.startswith("OPR_"), name
    return "".join(w.capitalize() for w in name[4:].split("_"))


# ---------------------------------------------------------------- model load

def load_model(xml_path, pin=True):
    data = pathlib.Path(xml_path).read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    name = pathlib.Path(xml_path).name
    if pin and PINS.get(name) != sha:
        raise SystemExit(f"{name}: SHA-256 {sha} does not equal its pinned digest {PINS.get(name)}")
    root = ET.fromstring(data)
    m = Model()
    m.sha256 = sha
    m.xml_file = name
    doc = root.find("Document")
    arch = root.find("ISA/Architecture")
    m.schema = text(doc, "SchemaVersion")
    m.release = text(doc, "ReleaseDate")
    m.arch_name = text(arch, "ArchitectureName")
    m.arch_id = num(arch.find("ArchitectureId").text)
    m.encodings = [Encoding(e, i) for i, e in enumerate(root.findall("ISA/Encodings/Encoding"))]
    m.by_name = {}
    for e in m.encodings:
        if e.name in m.by_name:
            raise Unsupported(f"duplicate encoding {e.name}")
        m.by_name[e.name] = e
    for e in m.encodings:
        conds = []
        for ci, c in enumerate(e.node.findall("EncodingConditions/EncodingCondition")):
            conds.append((text(c, "ConditionName"), compile_cond(m, e, c.find("CondtionExpression"))))
        if not conds:
            # ENC_FLAT_GLBL / ENC_FLAT_SCRATCH declare no condition block at
            # all; every row of theirs says `default`: unconditional.
            conds = [("default", ("lit", 1))]
        e.conditions = conds
    m.operand_types = [OperandType(o) for o in root.findall("ISA/OperandTypes/OperandType")]
    m.operand_type = {o.name: o for o in m.operand_types}
    m.data_formats = []
    for d in root.findall("ISA/DataFormats/DataFormat"):
        comps = []
        for bm in d.findall("DataAttributes/BitMap"):
            fields = []
            for f in bm.findall("Field"):
                rs = sorted(f.findall("BitLayout/Range"), key=lambda r: int(r.get("Order", "0")))
                fields.append((text(f, "FieldName"), f.get("Signedness", ""),
                               [(num(r.find("BitOffset").text), num(r.find("BitCount").text)) for r in rs]))
            comps.append((int(bm.get("Order", "0")), fields))
        m.data_formats.append(dict(
            name=text(d, "DataFormatName"), description=text(d, "Description"),
            data_type=text(d, "DataType"), bit_count=num(d.find("BitCount").text),
            component_count=num(d.find("ComponentCount").text), components=comps))
    m.format_names = {d["name"] for d in m.data_formats}
    m.groups = [(text(g, "Name"), text(g, "Description")) for g in root.findall("ISA/FunctionalGroups/FunctionalGroup")]
    m.subgroups = [text(g, "Name") for g in root.findall("ISA/FunctionalSubgroups/FunctionalSubgroup")]
    m.instructions = []
    m.rows = []
    for ii, inode in enumerate(root.findall("ISA/Instructions/Instruction")):
        flags = {k: text(inode, f"InstructionFlags/{k}").upper() == "TRUE"
                 for k in ("IsBranch", "IsConditionalBranch", "IsIndirectBranch",
                           "IsProgramTerminator", "IsImmediatelyExecuted")}
        group = text(inode, "FunctionalGroup/Name")
        subgroups = [s.text.strip() for s in inode.findall("FunctionalGroup/FunctionalSubgroups/Subgroup")]
        aliases = [a.text.strip() for a in inode.findall("AliasedInstructionNames/InstructionName")]
        inst = dict(index=ii, name=text(inode, "InstructionName"), description=text(inode, "Description"),
                    flags=flags, group=group, subgroups=subgroups, aliases=aliases)
        m.instructions.append(inst)
        for rnode in inode.findall("InstructionEncodings/InstructionEncoding"):
            enc = m.by_name[text(rnode, "EncodingName")]
            cname = text(rnode, "EncodingCondition")
            cidx = [i for i, (n, _) in enumerate(enc.conditions) if n == cname]
            if not cidx:
                raise Unsupported(f"{inst['name']}: unknown condition {cname} on {enc.name}")
            # Several equally-named conditions (ENC_FLAT) are one logical
            # condition per identical expression: use the first.
            ops = []
            for o in sorted(rnode.findall("Operands/Operand"), key=lambda o: int(o.get("Order"))):
                fname = o.findtext("FieldName")
                fidx = None
                if fname is not None:
                    fname = fname.strip()
                    if fname not in enc.field_index:
                        raise Unsupported(f"{inst['name']}/{enc.name}: operand field {fname} not in the encoding")
                    fidx = enc.field_index[fname]
                otype = text(o, "OperandType")
                if otype not in m.operand_type:
                    raise Unsupported(f"{inst['name']}: unknown operand type {otype}")
                dfmt = text(o, "DataFormatName")
                if dfmt not in m.format_names:
                    raise Unsupported(f"{inst['name']}: unknown data format {dfmt}")
                ops.append(dict(
                    order=int(o.get("Order")), field=fname, field_index=fidx, type=otype,
                    data_format=dfmt, size=num(o.find("OperandSize").text),
                    input=o.get("Input", "false").lower() == "true",
                    output=o.get("Output", "false").lower() == "true",
                    implicit=o.get("IsImplicit", "false").lower() == "true",
                    required=o.get("IsBinaryMicrocodeRequired", "false").lower() == "true"))
            m.rows.append(dict(
                row=len(m.rows), inst=ii, name=inst["name"], enc=enc.index,
                condition=cidx[0], condition_name=cname, opcode=radix_num(rnode.find("Opcode")),
                operands=ops))
    m.keys = {}
    for r in m.rows:
        key = (r["enc"], r["condition"], r["opcode"])
        if key in m.keys:
            raise Unsupported(f"duplicate (encoding, condition, opcode) {key}: {r['name']} vs {m.rows[m.keys[key]]['name']}")
        m.keys[key] = r["row"]
    return m


# ------------------------------------------------------ reference decode (Rust mirror)

class DecodeError(ValueError):
    pass


def py_decode(model, words, verified=None):
    """Reference decoder; semantics mirror ``xml::decode`` exactly.

    ``verified[row]`` is the receipt standing of each row (a verified row
    beats an unverified one on an identifier collision)."""
    if not words:
        raise DecodeError("no words to decode")
    word = 0
    for i, w in enumerate(words[:MAX_DWORDS]):
        word |= (w & 0xFFFFFFFF) << (32 * i)
    hit = None
    for enc in model.encodings:
        if (word & enc.id_mask) not in enc.ids:
            continue
        masked = word & ((1 << enc.bit_count) - 1)
        values = [f.extract(masked) for f in enc.fields]
        opcode = (masked >> enc.op_offset) & ((1 << enc.op_width) - 1)
        found = None
        for ci, (_, expr) in enumerate(enc.conditions):
            if not holds(expr, values):
                continue
            row = model.keys.get((enc.index, ci, opcode))
            if row is not None:
                found = row
                break
        if found is None:
            continue
        if hit is not None:
            vp = bool(verified and verified[hit])
            vf = bool(verified and verified[found])
            if vp and not vf:
                continue
            if not (vf and not vp):
                raise DecodeError(f"ambiguous decode: {model.rows[hit]['name']} and {model.rows[found]['name']}")
        hit = found
    if hit is None:
        raise DecodeError(f"no encoding matches {words[0]:#x}")
    enc = model.encodings[model.rows[hit]["enc"]]
    if len(words) < enc.dwords:
        raise DecodeError(f"{model.rows[hit]['name']} needs {enc.dwords} dwords, only {len(words)} present")
    if word & enc.reserved:
        raise DecodeError(f"{model.rows[hit]['name']} sets reserved bits {word & enc.reserved:#x}")
    masked = word & ((1 << enc.bit_count) - 1)
    return hit, [f.extract(masked) for f in enc.fields], enc.dwords


def py_assemble(model, row, values):
    """Reference assembler; mirrors ``Decoded::assemble`` (minus re-decode)."""
    r = model.rows[row]
    enc = model.encodings[r["enc"]]
    word = 0
    for f, v in zip(enc.fields, values):
        if v >> f.bits:
            raise DecodeError(f"field {f.name} value {v:#x} too wide")
        word |= f.place(v)
    if (word & enc.id_mask) not in enc.ids:
        raise DecodeError("identifier mismatch")
    if not holds(enc.conditions[r["condition"]][1], [f.extract(word) for f in enc.fields]):
        raise DecodeError("condition violated")
    if ((word >> enc.op_offset) & ((1 << enc.op_width) - 1)) != r["opcode"]:
        raise DecodeError("opcode mismatch")
    if word & enc.reserved:
        raise DecodeError("reserved bits")
    return [(word >> (32 * i)) & 0xFFFFFFFF for i in range(enc.dwords)]


def operand_valid_raw(model, op, field):
    """Smallest raw value of ``field`` the operand's XML value set accepts."""
    otype = model.operand_type[op["type"]]
    if not otype.values or otype.partitioned or field.bits > 12:
        return 0
    modulo = 1 << (field.bits + field.pad_bits)
    for raw in range(1 << field.bits):
        if otype.accepts((raw << field.pad_bits) | field.pad_value, modulo):
            return raw
    return None


def make_seed(model, row):
    """Structured default values satisfying the row's identifier, opcode,
    condition and (where the XML value sets allow) operand value sets."""
    r = model.rows[row]
    enc = model.encodings[r["enc"]]
    op_field = enc.fields[enc.field_index["OP"]]
    ids = enc.ids
    if op_field.mask & ~enc.id_mask == 0:
        ids = [i for i in ids if op_field.extract(i) == r["opcode"]]
    if not ids:
        raise Unsupported(f"{r['name']}: no identifier for opcode {r['opcode']}")
    ident = ids[0]
    values = [0] * len(enc.fields)
    for i, f in enumerate(enc.fields):
        if f.mask & ~enc.id_mask == 0:
            values[i] = f.extract(ident)
    values[enc.field_index["OP"]] = r["opcode"]
    cond = enc.conditions[r["condition"]][1]
    refs = sorted(expr_fields(cond))
    pinned = set(refs) | {enc.field_index["OP"]}
    for i, f in enumerate(enc.fields):
        if f.mask & ~enc.id_mask == 0:
            pinned.add(i)
    for o in r["operands"]:
        if o["field_index"] is not None and o["field_index"] not in pinned:
            raw = operand_valid_raw(model, o, enc.fields[o["field_index"]])
            if raw is not None:
                values[o["field_index"]] = raw
    import itertools
    choices = [0, 1, 128, 249, 250, 255, 256]
    for combo in itertools.product(choices, repeat=len(refs)):
        trial = list(values)
        ok = True
        for i, v in zip(refs, combo):
            if v >> enc.fields[i].bits:
                ok = False
                break
            trial[i] = v
        if ok and holds(cond, trial):
            values = trial
            break
    else:
        raise Unsupported(f"{r['name']}: no seed satisfies the condition of {enc.name}")
    return values


# ---------------------------------------------------------------- receipts

def load_receipt(model, arch, path):
    path = pathlib.Path(path)
    if not path.exists():
        raise SystemExit(f"{path}: LLVM receipt missing (run tools/crosscheck_xml.py first)")
    raw = path.read_bytes()
    doc = json.loads(raw)
    if doc["xml_sha256"] != model.sha256:
        raise SystemExit(f"{path}: receipt is for XML {doc['xml_sha256']}, not {model.sha256}")
    if doc["arch"] != arch["gfx"]:
        raise SystemExit(f"{path}: receipt arch {doc['arch']} != {arch['gfx']}")
    rows = doc["rows"]
    if len(rows) != len(model.rows):
        raise SystemExit(f"{path}: {len(rows)} receipt rows vs {len(model.rows)} XML rows")
    for mr, rr in zip(model.rows, rows):
        enc = model.encodings[mr["enc"]]
        got = (rr["index"], rr["name"], rr["encoding"], rr["condition"], rr["opcode"])
        want = (mr["row"], mr["name"], enc.name, mr["condition_name"], mr["opcode"])
        if got != want:
            raise SystemExit(f"{path}: row {mr['row']} mismatch: receipt {got} vs XML {want}")
    out = []
    for rr in rows:
        if rr["status"] == "verified":
            out.append((True, "", bytes.fromhex(rr["bytes"])))
        else:
            reason = rr.get("verification") or f"unverified({rr['status']})"
            out.append((False, " ".join(reason.split()), None))
    version = doc["llvm_version"].strip().splitlines()[0]
    return dict(sha256=hashlib.sha256(raw).hexdigest(), version=version, rows=out,
                counts=doc["counts"], path=str(path.relative_to(CRATE)) if path.is_relative_to(CRATE) else str(path))


# -------------------------------------------------------------- Rust emission

def rs(s):
    out = ['"']
    for ch in s:
        if ch == "\\":
            out.append("\\\\")
        elif ch == '"':
            out.append('\\"')
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\r":
            out.append("\\r")
        elif ch == "\t":
            out.append("\\t")
        elif ord(ch) < 0x20 or ord(ch) == 0x7F:
            out.append("\\u{%x}" % ord(ch))
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


def rs_u128(v):
    return f"0x{v:x}u128"


def rs_ranges(rs_):
    return "&[" + ", ".join(f"({o}, {c})" for o, c in rs_) + "]"


def rs_bool(b):
    return "true" if b else "false"


def rs_field(f):
    pre = ""
    if f.predefined:
        pre = "&[" + ", ".join(
            f"PredefinedValue {{ name: {rs(n)}, description: {rs(d)}, value: {v} }}" for n, d, v in f.predefined) + "]"
    else:
        pre = "&[]"
    return (f"Field {{ name: {rs(f.name)}, description: {rs(f.description)}, ranges: {rs_ranges(f.ranges)}, "
            f"bits: {f.bits}, pad_bits: {f.pad_bits}, pad_value: {f.pad_value}, predefined: {pre} }}")


def check_operand_kinds(models):
    src = (CRATE / "src" / "xml.rs").read_text()
    match = re.search(r"// BEGIN operand-kinds.*?// END operand-kinds", src, re.S)
    if not match:
        raise SystemExit("src/xml.rs: operand-kinds block missing")
    known = dict(re.findall(r"OperandKind::(\w+) => \"(OPR_\w+)\"", match.group(0)))
    inv = {v: k for k, v in known.items()}
    names = set()
    for m in models:
        names |= {o.name for o in m.operand_types}
        for r in m.rows:
            names |= {o["type"] for o in r["operands"]}
    for n in sorted(names):
        if inv.get(n) != variant(n):
            raise SystemExit(f"src/xml.rs OperandKind lacks {variant(n)} => \"{n}\"")
    extra = set(inv) - names
    if extra:
        raise SystemExit(f"src/xml.rs OperandKind has kinds absent from both XMLs: {sorted(extra)}")


def emit_rust(model, arch, receipt, generator_sha):
    L = []
    w = L.append
    w("// SPDX-License-Identifier: Apache-2.0")
    w("// Copyright (c) 2026 Kaden Schutt")
    w("// hipfire — see LICENSE and NOTICE in the project root.")
    w("")
    w(f"// @generated by {GENERATOR}; DO NOT EDIT. Regenerate with:")
    w(f"//   {GENERATOR} --xml-dir <dir with {model.xml_file}>")
    w(f"// Source XML  : {model.xml_file}")
    w(f"// XML sha256  : {model.sha256}")
    w(f"// XML version : schema {model.schema}, released {model.release}, {model.arch_name} (id {model.arch_id})")
    w(f"// Receipt     : {receipt['path']} (sha256 {receipt['sha256']})")
    w(f"// LLVM        : {receipt['version']}")
    w(f"// Receipt rows: {json.dumps(receipt['counts'], sort_keys=True)}")
    w(f"// Generator   : {GENERATOR} sha256 {generator_sha}")
    w(f"// Tables      : {len(model.encodings)} encodings, {len(model.instructions)} instructions, "
      f"{len(model.rows)} forms, {len(model.operand_types)} operand types, {len(model.data_formats)} data formats")
    w("")
    w("#![allow(clippy::all, clippy::pedantic, clippy::nursery, unused_imports)]")
    w("")
    w("use crate::xml::*;")
    w("")
    for enc in model.encodings:
        n = f"E{enc.index:02}"
        w(f"static {n}_FIELDS: [Field; {len(enc.fields)}] = [")
        for f in enc.fields:
            w(f"    {rs_field(f)},")
        w("];")
        w(f"static {n}: Encoding = Encoding {{")
        w(f"    index: {enc.index},")
        w(f"    name: {rs(enc.name)},")
        w(f"    order: {enc.order},")
        w(f"    description: {rs(enc.description)},")
        w(f"    dwords: {enc.dwords},")
        w(f"    bit_count: {enc.bit_count},")
        w(f"    id_mask: {rs_u128(enc.id_mask)},")
        w("    id_values: &[" + ", ".join(rs_u128(i) for i in enc.ids) + "],")
        w("    conditions: &[" + ", ".join(
            f"EncodingCondition {{ index: {ci}, name: {rs(cn)}, expr: {expr_rust(ex)} }}"
            for ci, (cn, ex) in enumerate(enc.conditions)) + "],")
        w(f"    fields: &{n}_FIELDS,")
        w(f"    op_offset: {enc.op_offset},")
        w(f"    op_width: {enc.op_width},")
        w(f"    covered: {rs_u128(enc.covered)},")
        w(f"    reserved: {rs_u128(enc.reserved)},")
        w("};")
        w("")
    w(f"static ENCODINGS: [&Encoding; {len(model.encodings)}] = [")
    for enc in model.encodings:
        w(f"    &E{enc.index:02},")
    w("];")
    w("")
    w(f"static INSTRUCTIONS: [InstructionMeta; {len(model.instructions)}] = [")
    for inst in model.instructions:
        fl = inst["flags"]
        w("    InstructionMeta {")
        w(f"        name: {rs(inst['name'])},")
        w(f"        description: {rs(inst['description'])},")
        w(f"        is_branch: {rs_bool(fl['IsBranch'])},")
        w(f"        is_conditional_branch: {rs_bool(fl['IsConditionalBranch'])},")
        w(f"        is_indirect_branch: {rs_bool(fl['IsIndirectBranch'])},")
        w(f"        is_program_terminator: {rs_bool(fl['IsProgramTerminator'])},")
        w(f"        is_immediately_executed: {rs_bool(fl['IsImmediatelyExecuted'])},")
        w(f"        group: {rs(inst['group'])},")
        w("        subgroups: &[" + ", ".join(rs(s) for s in inst["subgroups"]) + "],")
        w("        aliases: &[" + ", ".join(rs(s) for s in inst["aliases"]) + "],")
        w("    },")
    w("];")
    w("")
    w(f"static FORMS: [Form; {len(model.rows)}] = [")
    for r in model.rows:
        enc = model.encodings[r["enc"]]
        inst = model.instructions[r["inst"]]
        fl = inst["flags"]
        ok, reason, sample = receipt["rows"][r["row"]]
        ver = "Verification::Verified" if ok else f"Verification::Unverified({rs(reason)})"
        if ok and (not sample or len(sample) % 4):
            raise SystemExit(f"row {r['row']} {inst['name']}: receipt bytes {sample!r} are not whole dwords")
        words = [int.from_bytes(sample[i:i + 4], "little") for i in range(0, len(sample or b""), 4)]
        w("    Form {")
        w(f"        isa: Isa::{arch['isa']},")
        w(f"        row: {r['row']},")
        w(f"        inst_index: {r['inst']},")
        w(f"        name: {rs(inst['name'])},")
        w(f"        mnemonic: {rs(inst['name'].lower())},")
        w(f"        encoding: {rs(enc.name)},")
        w(f"        encoding_index: {enc.index},")
        w(f"        condition: {rs(r['condition_name'])},")
        w(f"        condition_index: {r['condition']},")
        w(f"        opcode: {r['opcode']},")
        w("        operands: &[")
        for o in r["operands"]:
            field = f"Some({rs(o['field'])})" if o["field"] is not None else "None"
            fidx = f"Some({o['field_index']})" if o["field_index"] is not None else "None"
            w(f"            Operand {{ order: {o['order']}, field: {field}, field_index: {fidx}, "
              f"kind: OperandKind::{variant(o['type'])}, data_format: {rs(o['data_format'])}, size_bits: {o['size']}, "
              f"input: {rs_bool(o['input'])}, output: {rs_bool(o['output'])}, implicit: {rs_bool(o['implicit'])}, "
              f"microcode_required: {rs_bool(o['required'])} }},")
        w("        ],")
        w(f"        fields: &E{enc.index:02}_FIELDS,")
        w(f"        is_branch: {rs_bool(fl['IsBranch'])},")
        w(f"        is_conditional_branch: {rs_bool(fl['IsConditionalBranch'])},")
        w(f"        is_indirect_branch: {rs_bool(fl['IsIndirectBranch'])},")
        w(f"        is_program_terminator: {rs_bool(fl['IsProgramTerminator'])},")
        w(f"        is_immediately_executed: {rs_bool(fl['IsImmediatelyExecuted'])},")
        w(f"        group: {rs(inst['group'])},")
        w("        subgroups: &[" + ", ".join(rs(s) for s in inst["subgroups"]) + "],")
        w(f"        verification: {ver},")
        w("        sample: &[" + ", ".join(f"0x{x:08x}" for x in words) + "],")
        w(f"        enc: &E{enc.index:02},")
        w("    },")
    w("];")
    w("")
    keys = sorted(model.keys.items())
    w(f"static FORM_KEYS: [(u16, u16, u32); {len(keys)}] = [")
    for i in range(0, len(keys), 6):
        w("    " + ", ".join(f"({k[0]}, {k[1]}, {k[2]})" for k, _ in keys[i:i + 6]) + ",")
    w("];")
    w(f"static FORM_ROWS: [u32; {len(keys)}] = [")
    for i in range(0, len(keys), 16):
        w("    " + ", ".join(str(row) for _, row in keys[i:i + 16]) + ",")
    w("];")
    w("")
    w(f"static OPERAND_TYPES: [OperandTypeMeta; {len(model.operand_types)}] = [")
    for o in model.operand_types:
        w("    OperandTypeMeta {")
        w(f"        name: {rs(o.name)},")
        w(f"        description: {rs(o.description)},")
        w(f"        partitioned: {rs_bool(o.partitioned)},")
        w("        subtypes: &[" + ", ".join(rs(s) for s in o.subtypes) + "],")
        w("        values: &[")
        for n, d, v in o.values:
            w(f"            PredefinedValue {{ name: {rs(n)}, description: {rs(d)}, value: {v} }},")
        w("        ],")
        w("        ranges: &[" + ", ".join(f"({lo}, {hi})" for lo, hi in o.ranges) + "],")
        w("        fields: &[")
        for f in o.fields:
            w(f"            {rs_field(f)},")
        w("        ],")
        w("    },")
    w("];")
    w("")
    w(f"static DATA_FORMATS: [DataFormat; {len(model.data_formats)}] = [")
    for d in model.data_formats:
        w("    DataFormat {")
        w(f"        name: {rs(d['name'])},")
        w(f"        description: {rs(d['description'])},")
        w(f"        data_type: {rs(d['data_type'])},")
        w(f"        bit_count: {d['bit_count']},")
        w(f"        component_count: {d['component_count']},")
        w("        components: &[")
        for order, fields in d["components"]:
            w(f"            DataComponent {{ order: {order}, fields: &[" + ", ".join(
                f"DataField {{ name: {rs(n)}, signedness: {rs(s)}, ranges: {rs_ranges(rg)} }}" for n, s, rg in fields) + "] },")
        w("        ],")
        w("    },")
    w("];")
    w("")
    w(f"static FUNCTIONAL_GROUPS: [FunctionalGroup; {len(model.groups)}] = [")
    for n, d in model.groups:
        w(f"    FunctionalGroup {{ name: {rs(n)}, description: {rs(d)} }},")
    w("];")
    w(f"static FUNCTIONAL_SUBGROUPS: [&str; {len(model.subgroups)}] = [")
    for n in model.subgroups:
        w(f"    {rs(n)},")
    w("];")
    w("")
    w("pub static TABLE: IsaTable = IsaTable {")
    w(f"    isa: Isa::{arch['isa']},")
    w(f"    gfx: {rs(arch['gfx'])},")
    w("    provenance: Provenance {")
    w(f"        xml_file: {rs(model.xml_file)},")
    w(f"        xml_sha256: {rs(model.sha256)},")
    w(f"        xml_version: {rs(f'schema {model.schema} {model.release} {model.arch_name}')},")
    w(f"        arch_id: {model.arch_id},")
    w(f"        rows: {len(model.rows)},")
    w(f"        encodings: {len(model.encodings)},")
    w(f"        receipt_file: {rs(receipt['path'])},")
    w(f"        receipt_sha256: {rs(receipt['sha256'])},")
    w(f"        llvm_version: {rs(receipt['version'])},")
    w(f"        generator: {rs(GENERATOR)},")
    w(f"        generator_sha256: {rs(generator_sha)},")
    w("    },")
    w("    encodings: &ENCODINGS,")
    w("    instructions: &INSTRUCTIONS,")
    w("    forms: &FORMS,")
    w("    form_keys: &FORM_KEYS,")
    w("    form_rows: &FORM_ROWS,")
    w("    operand_types: &OPERAND_TYPES,")
    w("    data_formats: &DATA_FORMATS,")
    w("    functional_groups: &FUNCTIONAL_GROUPS,")
    w("    functional_subgroups: &FUNCTIONAL_SUBGROUPS,")
    w("};")
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------ manifest

def build_manifest(model, arch, receipt, generator_sha):
    standing = [ok for ok, _, _ in receipt["rows"]]
    encs = []
    for e in model.encodings:
        encs.append(dict(
            index=e.index, name=e.name, bit_count=e.bit_count, dwords=e.dwords,
            id_mask=f"0x{e.id_mask:x}", identifiers=[f"0x{i:x}" for i in e.ids],
            xml_id_mask=f"0x{e.xml_id_mask:x}", xml_identifiers=[f"0x{i:x}" for i in e.xml_ids],
            decode_correction=e.correction,
            reserved_mask=f"0x{e.reserved:x}", op=dict(offset=e.op_offset, width=e.op_width),
            conditions=[dict(index=i, name=n, expr=expr_text(x)) for i, (n, x) in enumerate(e.conditions)],
            fields=[dict(name=f.name, ranges=[list(r) for r in f.ranges], bits=f.bits,
                         pad_bits=f.pad_bits, pad_value=f.pad_value) for f in e.fields]))
    rows = []
    problems = []
    for r in model.rows:
        enc = model.encodings[r["enc"]]
        ok, reason, llvm_bytes = receipt["rows"][r["row"]]
        entry = dict(
            row=r["row"], name=r["name"], mnemonic=r["name"].lower(), encoding=enc.name,
            encoding_index=enc.index, condition=r["condition_name"], condition_index=r["condition"],
            opcode=r["opcode"], bit_count=enc.bit_count, mask=f"0x{enc.id_mask:x}",
            verified=ok, verification="verified" if ok else reason,
            operands=[dict(order=o["order"], field=o["field"], type=o["type"], data_format=o["data_format"],
                           size_bits=o["size"], input=o["input"], output=o["output"], implicit=o["implicit"],
                           microcode_required=o["required"]) for o in r["operands"]])
        try:
            values = make_seed(model, r["row"])
            words = py_assemble(model, r["row"], values)
            entry["value"] = f"0x{sum(w << (32 * i) for i, w in enumerate(words)):x}"
            entry["seed"] = dict(
                fields={f.name: v for f, v in zip(enc.fields, values)},
                words=[f"0x{w:08x}" for w in words],
                bytes="".join(w.to_bytes(4, "little").hex() for w in words))
            back, _, _ = py_decode(model, words, standing)
            if back != r["row"]:
                raise DecodeError(f"seed satisfies the XML for this row but decodes as {model.rows[back]['name']} (identifier collision)")
        except (DecodeError, Unsupported) as err:
            entry.setdefault("seed", None)
            entry["seed_error"] = str(err)
            problems.append((r["row"], r["name"], enc.name, str(err)))
        if llvm_bytes is not None:
            entry["llvm_bytes"] = llvm_bytes.hex()
        rows.append(entry)
    doc = dict(
        xml_file=model.xml_file, xml_sha256=model.sha256, schema=model.schema, release=model.release,
        architecture=model.arch_name, gfx=arch["gfx"], receipt=receipt["path"], receipt_sha256=receipt["sha256"],
        llvm_version=receipt["version"], generator=GENERATOR, generator_sha256=generator_sha,
        counts=dict(rows=len(rows), encodings=len(encs), verified=sum(1 for e in rows if e["verified"])),
        encodings=encs, rows=rows)
    return doc, problems


def receipt_selfcheck(model, receipt):
    """LLVM-verified bytes must decode to their own row; returns the misses."""
    misses = []
    for r in model.rows:
        ok, _, raw = receipt["rows"][r["row"]]
        if not ok:
            continue
        words = [int.from_bytes(raw[i:i + 4].ljust(4, b"\0"), "little") for i in range(0, len(raw), 4)]
        try:
            row, _, n = py_decode(model, words, [ok for ok, _, _ in receipt["rows"]])
            if row != r["row"]:
                misses.append((r["row"], r["name"], f"decodes as row {row} {model.rows[row]['name']}"))
            elif n * 4 != len(raw):
                misses.append((r["row"], r["name"], f"length {n * 4} != LLVM {len(raw)}"))
        except DecodeError as err:
            misses.append((r["row"], r["name"], str(err)))
    return misses


# ----------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xml-dir", type=pathlib.Path,
                    default=os.environ.get("AMDGPU_ISA_XML_DIR"),
                    help="directory holding amdgpu_isa_cdna3.xml and amdgpu_isa_cdna4.xml "
                         "(or $AMDGPU_ISA_XML_DIR)")
    ap.add_argument("--src-dir", type=pathlib.Path, default=CRATE / "src")
    ap.add_argument("--tools-dir", type=pathlib.Path, default=CRATE / "tools")
    ap.add_argument("--check", action="store_true", help="fail if any generated file would change")
    args = ap.parse_args()
    if args.xml_dir is None:
        raise SystemExit("--xml-dir (or $AMDGPU_ISA_XML_DIR) required")
    generator_sha = hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()
    models = []
    for arch in ARCHES:
        models.append(load_model(pathlib.Path(args.xml_dir) / arch["xml"]))
    check_operand_kinds(models)
    stale = []
    for arch, model in zip(ARCHES, models):
        receipt = load_receipt(model, arch, CRATE / arch["receipt"])
        misses = receipt_selfcheck(model, receipt)
        manifest, problems = build_manifest(model, arch, receipt, generator_sha)
        rust = emit_rust(model, arch, receipt, generator_sha)
        outputs = {
            args.src_dir / f"{arch['module']}.rs": rust,
            args.tools_dir / f"xml-manifest-{arch['key']}.json": json.dumps(manifest, indent=1) + "\n",
        }
        for path, body in outputs.items():
            if args.check:
                if not path.exists() or path.read_text() != body:
                    stale.append(str(path))
            else:
                path.write_text(body)
        print(f"{arch['gfx']}: {len(model.encodings)} encodings, {len(model.instructions)} instructions, "
              f"{len(model.rows)} forms ({manifest['counts']['verified']} receipt-verified), "
              f"{sum(len(o.values) for o in model.operand_types)} operand values -> {arch['module']}.rs")
        if misses:
            print(f"  RECEIPT SELF-CHECK: {len(misses)} LLVM-verified rows do not decode to themselves:")
            for row, name, why in misses:
                print(f"    row {row} {name}: {why}")
        if problems:
            print(f"  SEED PROBLEMS: {len(problems)} rows without a decodable seed:")
            for row, name, enc, why in problems:
                print(f"    row {row} {name} ({enc}): {why}")
    if stale:
        print("STALE:", *stale, sep="\n  ")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
