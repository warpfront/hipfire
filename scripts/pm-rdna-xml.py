#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
"""Generate lossless RDNA encoding metadata without rewriting census exceptions.

The key includes the original instruction-local encoding ordinal: AMD repeats
some (instruction, encoding, condition, opcode) tuples, sometimes with distinct
operands (for example the X and Y halves of VOPD).
LLVM is an independent oracle, not the source of the instruction inventory.
Failed oracle candidates remain present with an explicit diagnostic; they are
never labelled verified or silently discarded. Numeric semantics are not inferred.

SOP1_OVERRIDES names every ENC_SOP1 row whose XML condition
(Nothas_lit_0_Nothas_lit_1) is undefined, pinned to each XML SHA-256. With
--sop1-evidence, llvm-mc disassemble/reassemble proves each row in register,
inline-constant and literal form and writes sop1-overrides.json: corrected
predicates (SSRC0 != 255, or SSRC0 == 0 for source-less rows) only where the
probes prove them, explicit named unavailable rows otherwise. The catalogs keep
the original XML condition; evidence is never merged into catalog rows.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import itertools
import json
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET

TARGETS = {'gfx1100': 'rdna3', 'gfx1151': 'rdna3_5', 'gfx12': 'rdna4'}
GRAPHICS_ENCODINGS = frozenset({
    'ENC_EXP', 'ENC_VEXPORT', 'ENC_VINTERP', 'ENC_MIMG', 'MIMG_NSA1',
    'ENC_VIMAGE', 'ENC_VSAMPLE',
})

SOP1_EVIDENCE = 'sop1-overrides.json'
# SSRC0 forms probed per row: a plain SGPR, the inline constant 0, and the literal marker.
SOP1_FORMS = (('register', 0), ('inline', 128), ('literal', 255))
SOP1_LITERAL = 0x41234567
SOP1_ENCODING_BASE = 0xbe800000
SOP1_PREDICATES = {
    # The XML name means "no literal on source 0": exclude the literal marker.
    'ssrc0-not-literal': {'field': 'SSRC0', 'op': '!=', 'value': 255},
    # Source-less rows (s_getpc_b64): LLVM only round-trips an all-zero field.
    'ssrc0-zero': {'field': 'SSRC0', 'op': '==', 'value': 0},
    # Rows whose accepted sources are a strict LLVM-derived subset: the row's own
    # `domain_ranges` (sorted inclusive [lo, hi] SSRC0 ranges proven by llvm-mc) applies.
    'ssrc0-in-domain': {'field': 'SSRC0', 'op': 'in-ranges'},
}
# SSRC0 selector classes for per-form domain evidence (inclusive ranges).
SOP1_FORM_RANGES = {'register': (0, 127), 'inline': (128, 254), 'literal': (255, 255)}
SOP1_CONDITION = 'Nothas_lit_0_Nothas_lit_1'
# Every ENC_SOP1 row whose XML EncodingCondition is the undefined name above, by XML SHA-256.
SOP1_OVERRIDES = {
    'gfx1100': {
        'xml_sha256': '6eee5f8737172adf08c0e7d5994ea916e9555b9c23ccb9ec80d9ee38678733b4',
        'rows': (
            's_mov_b32', 's_mov_b64', 's_cmov_b32', 's_cmov_b64', 's_brev_b32', 's_brev_b64',
            's_ctz_i32_b32', 's_ctz_i32_b64', 's_clz_i32_u32', 's_clz_i32_u64', 's_cls_i32',
            's_cls_i32_i64', 's_sext_i32_i8', 's_sext_i32_i16', 's_bitset0_b32',
            's_bitset0_b64', 's_bitset1_b32', 's_bitset1_b64', 's_bitreplicate_b64_b32',
            's_abs_i32', 's_bcnt0_i32_b32', 's_bcnt0_i32_b64', 's_bcnt1_i32_b32',
            's_bcnt1_i32_b64', 's_quadmask_b32', 's_quadmask_b64', 's_wqm_b32', 's_wqm_b64',
            's_not_b32', 's_not_b64', 's_and_saveexec_b32', 's_and_saveexec_b64',
            's_or_saveexec_b32', 's_or_saveexec_b64', 's_xor_saveexec_b32',
            's_xor_saveexec_b64', 's_nand_saveexec_b32', 's_nand_saveexec_b64',
            's_nor_saveexec_b32', 's_nor_saveexec_b64', 's_xnor_saveexec_b32',
            's_xnor_saveexec_b64', 's_and_not0_saveexec_b32', 's_and_not0_saveexec_b64',
            's_or_not0_saveexec_b32', 's_or_not0_saveexec_b64', 's_and_not1_saveexec_b32',
            's_and_not1_saveexec_b64', 's_or_not1_saveexec_b32', 's_or_not1_saveexec_b64',
            's_and_not0_wrexec_b32', 's_and_not0_wrexec_b64', 's_and_not1_wrexec_b32',
            's_and_not1_wrexec_b64', 's_movrels_b32', 's_movrels_b64', 's_movreld_b32',
            's_movreld_b64', 's_movrelsd_2_b32', 's_getpc_b64', 's_setpc_b64', 's_swappc_b64',
            's_rfe_b64',
        )},
    'gfx1151': {
        'xml_sha256': 'c36b6d79b1e940d74107221c985f5a7fde248025da251d2c6ef756c4cd31391a',
        'rows': (
            's_mov_b32', 's_mov_b64', 's_cmov_b32', 's_cmov_b64', 's_brev_b32', 's_brev_b64',
            's_ctz_i32_b32', 's_ctz_i32_b64', 's_clz_i32_u32', 's_clz_i32_u64', 's_cls_i32',
            's_cls_i32_i64', 's_sext_i32_i8', 's_sext_i32_i16', 's_bitset0_b32',
            's_bitset0_b64', 's_bitset1_b32', 's_bitset1_b64', 's_bitreplicate_b64_b32',
            's_abs_i32', 's_bcnt0_i32_b32', 's_bcnt0_i32_b64', 's_bcnt1_i32_b32',
            's_bcnt1_i32_b64', 's_quadmask_b32', 's_quadmask_b64', 's_wqm_b32', 's_wqm_b64',
            's_not_b32', 's_not_b64', 's_and_saveexec_b32', 's_and_saveexec_b64',
            's_or_saveexec_b32', 's_or_saveexec_b64', 's_xor_saveexec_b32',
            's_xor_saveexec_b64', 's_nand_saveexec_b32', 's_nand_saveexec_b64',
            's_nor_saveexec_b32', 's_nor_saveexec_b64', 's_xnor_saveexec_b32',
            's_xnor_saveexec_b64', 's_and_not0_saveexec_b32', 's_and_not0_saveexec_b64',
            's_or_not0_saveexec_b32', 's_or_not0_saveexec_b64', 's_and_not1_saveexec_b32',
            's_and_not1_saveexec_b64', 's_or_not1_saveexec_b32', 's_or_not1_saveexec_b64',
            's_and_not0_wrexec_b32', 's_and_not0_wrexec_b64', 's_and_not1_wrexec_b32',
            's_and_not1_wrexec_b64', 's_movrels_b32', 's_movrels_b64', 's_movreld_b32',
            's_movreld_b64', 's_movrelsd_2_b32', 's_getpc_b64', 's_setpc_b64', 's_swappc_b64',
            's_rfe_b64', 's_ceil_f32', 's_floor_f32', 's_trunc_f32', 's_rndne_f32',
            's_cvt_f32_i32', 's_cvt_f32_u32', 's_cvt_i32_f32', 's_cvt_u32_f32', 's_cvt_f16_f32',
            's_cvt_f32_f16', 's_cvt_hi_f32_f16', 's_ceil_f16', 's_floor_f16', 's_trunc_f16',
            's_rndne_f16',
        )},
    'gfx12': {
        'xml_sha256': 'f8a290c8471e26a1071b08b61a33e4d9efa46ec6cedcdb8e5ba57d7969b60692',
        'rows': (
            's_mov_b32', 's_mov_b64', 's_cmov_b32', 's_cmov_b64', 's_brev_b32', 's_brev_b64',
            's_ctz_i32_b32', 's_ctz_i32_b64', 's_clz_i32_u32', 's_clz_i32_u64', 's_cls_i32',
            's_cls_i32_i64', 's_sext_i32_i8', 's_sext_i32_i16', 's_bitset0_b32',
            's_bitset0_b64', 's_bitset1_b32', 's_bitset1_b64', 's_bitreplicate_b64_b32',
            's_abs_i32', 's_bcnt0_i32_b32', 's_bcnt0_i32_b64', 's_bcnt1_i32_b32',
            's_bcnt1_i32_b64', 's_quadmask_b32', 's_quadmask_b64', 's_wqm_b32', 's_wqm_b64',
            's_not_b32', 's_not_b64', 's_and_saveexec_b32', 's_and_saveexec_b64',
            's_or_saveexec_b32', 's_or_saveexec_b64', 's_xor_saveexec_b32',
            's_xor_saveexec_b64', 's_nand_saveexec_b32', 's_nand_saveexec_b64',
            's_nor_saveexec_b32', 's_nor_saveexec_b64', 's_xnor_saveexec_b32',
            's_xnor_saveexec_b64', 's_and_not0_saveexec_b32', 's_and_not0_saveexec_b64',
            's_or_not0_saveexec_b32', 's_or_not0_saveexec_b64', 's_and_not1_saveexec_b32',
            's_and_not1_saveexec_b64', 's_or_not1_saveexec_b32', 's_or_not1_saveexec_b64',
            's_and_not0_wrexec_b32', 's_and_not0_wrexec_b64', 's_and_not1_wrexec_b32',
            's_and_not1_wrexec_b64', 's_movrels_b32', 's_movrels_b64', 's_movreld_b32',
            's_movreld_b64', 's_movrelsd_2_b32', 's_getpc_b64', 's_setpc_b64', 's_swappc_b64',
            's_rfe_b64', 's_barrier_signal', 's_barrier_signal_isfirst', 's_alloc_vgpr',
            's_sleep_var', 's_ceil_f32', 's_floor_f32', 's_trunc_f32', 's_rndne_f32',
            's_cvt_f32_i32', 's_cvt_f32_u32', 's_cvt_i32_f32', 's_cvt_u32_f32', 's_cvt_f16_f32',
            's_cvt_f32_f16', 's_cvt_hi_f32_f16', 's_ceil_f16', 's_floor_f16', 's_trunc_f16',
            's_rndne_f16',
        )},
}


def sop1_probe(mc, cpu, row, ssrc0, literal):
    """Disassemble then reassemble one SOP1 word; never infer success from partial output."""
    word = SOP1_ENCODING_BASE | row['opcode'] << 8 | ssrc0
    data = word.to_bytes(4, 'little') + (SOP1_LITERAL.to_bytes(4, 'little') if literal else b'')
    args = [str(mc), '--triple=amdgcn-amd-amdhsa', f'--mcpu={cpu}']
    result = {'ssrc0': ssrc0, 'candidate': data.hex()}
    dec = subprocess.run(args + ['--disassemble'], input=' '.join(f'0x{x:02x}' for x in data) + '\n',
                         text=True, capture_output=True, check=False)
    lines = [x.strip() for x in dec.stdout.splitlines() if x.strip() and x.strip() != '.text']
    result['decoded'] = lines
    if dec.returncode or dec.stderr.strip() or len(lines) != 1:
        return result | {'status': 'decode-rejected', 'diagnostic': dec.stderr.strip() or 'not exactly one instruction'}
    if '/*' in lines[0]:
        return result | {'status': 'decode-invalid', 'diagnostic': lines[0]}
    names = {row['name'], *row['aliases']}
    if lines[0].split()[0] not in names:
        return result | {'status': 'name-mismatch'}
    enc = subprocess.run(args + ['--show-encoding'], input=lines[0] + '\n', text=True,
                         capture_output=True, check=False)
    chunks = re.findall(r'encoding: \[([^]]+)\]', enc.stdout)
    if enc.returncode or not chunks:
        return result | {'status': 'encode-rejected', 'diagnostic': enc.stderr.strip()}
    encoded = bytes(int(v.strip(), 16) for c in chunks for v in c.split(',')
                    if re.fullmatch(r'0x[0-9a-fA-F]{2}', v.strip()))
    return result | {'status': 'verified' if encoded == data else 'byte-mismatch',
                     'reencoded': encoded.hex()}


def sop1_decide(proofs):
    """Derive the corrected predicate solely from the observed llvm-mc outcomes."""
    statuses = {name: proofs[name]['status'] for name, _ in SOP1_FORMS}
    if statuses['register'] != 'verified':
        # LLVM-unknown only when no form decodes to an instruction at all.
        if all(not proofs[name]['decoded'] for name in statuses):
            return None, 'llvm-unknown'
        return None, f"register-form-{statuses['register']}"
    if statuses['inline'] == 'verified':
        return 'ssrc0-not-literal', None
    # Source-less: the field is ignored on decode (inline re-encodes to zero) and the
    # literal dword is never consumed as part of the instruction.
    register = proofs['register']['decoded']
    if (statuses['inline'] == 'byte-mismatch' and proofs['inline']['decoded'] == register
            and statuses['literal'] != 'verified' and proofs['literal']['decoded'][:1] == register):
        return 'ssrc0-zero', None
    return None, f"inline-constant-{statuses['inline']}"


def sop1_ranges(values):
    ranges = []
    for value in sorted(values):
        if ranges and ranges[-1][1] == value - 1:
            ranges[-1][1] = value
        else:
            ranges.append([value, value])
    return ranges


def sop1_domain(mc, cpu, row):
    """Probe all 256 SSRC0 selectors (selector 255 carries a literal dword)."""
    outcomes = {s: sop1_probe(mc, cpu, row, s, s == 255) for s in range(256)}
    by_form = {}
    for form, (low, high) in SOP1_FORM_RANGES.items():
        refused = {}
        for s in range(low, high + 1):
            if outcomes[s]['status'] != 'verified':
                refused.setdefault(outcomes[s]['status'], []).append(s)
        by_form[form] = {'selectors': sop1_ranges(s for s in range(low, high + 1)
                                                  if outcomes[s]['status'] == 'verified'),
                         'refused': {k: sop1_ranges(v) for k, v in sorted(refused.items())}}
    return {'probed': 256, 'selectors': sop1_ranges(s for s, o in outcomes.items()
                                                    if o['status'] == 'verified'),
            'by_form': by_form}


def sop1_evidence(mc, isa, jobs):
    """Probe every named override row and return the evidence document."""
    document = {'schema': 1, 'xml_condition': SOP1_CONDITION, 'predicates': SOP1_PREDICATES,
                'forms': {name: value for name, value in SOP1_FORMS},
                'literal': SOP1_LITERAL,
                'llvm': subprocess.check_output([str(mc), '--version'], text=True).splitlines()[0],
                'llvm_sha256': hashlib.sha256(mc.read_bytes()).hexdigest(), 'targets': {}}
    for target, table in SOP1_OVERRIDES.items():
        header, *catalog = [json.loads(x) for x in (isa / f'{target}.xml.jsonl').read_text().splitlines()]
        if header['sha256'] != table['xml_sha256']:
            raise ValueError(f'{target}: catalog XML hash differs from SOP1_OVERRIDES')
        found = {r['name']: r for r in catalog
                 if r['encoding'] == 'ENC_SOP1' and r['condition'] == SOP1_CONDITION}
        if list(found) != list(table['rows']):
            raise ValueError(f'{target}: SOP1_OVERRIDES rows differ from catalog')
        cpu = 'gfx1201' if target == 'gfx12' else target
        def probe(row):
            proofs = {name: sop1_probe(mc, cpu, row, value, name == 'literal')
                      for name, value in SOP1_FORMS}
            predicate, reason = sop1_decide(proofs)
            key = {k: row[k] for k in header['key']}
            domain = None
            if predicate is None:
                domain = sop1_domain(mc, cpu, row)
                if domain['selectors']:
                    predicate, reason = 'ssrc0-in-domain', None
            result = {'name': row['name'], 'key': key, 'xml_condition': row['condition'],
                      'xml_predicate': header['encodings']['ENC_SOP1']['conditions'][row['condition']],
                      'status': 'corrected' if predicate else 'unavailable',
                      'corrected_predicate': predicate, 'unavailable_reason': reason,
                      'proofs': proofs}
            if predicate == 'ssrc0-in-domain':
                result['domain_ranges'] = domain['selectors']
            if domain is not None:
                result['domain'] = domain
            return result
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            rows = list(pool.map(probe, found.values()))
        document['targets'][target] = {'xml': header['xml'], 'xml_sha256': header['sha256'],
                                       'cpu': cpu, 'rows': rows}
    return document



def number(node):
    return int(node.text, int(node.get('Radix', '10')))


def tree(node):
    """Retain expression structure and attributes, excluding formatting whitespace."""
    return {'tag': node.tag, 'attributes': dict(sorted(node.attrib.items())),
            'text': (node.text or '').strip(), 'children': [tree(c) for c in node]}


def predicate(node, fields):
    if node.tag == 'lit':
        return int(node.get('val'), 0)
    if node.tag == 'id':
        return node.get('val')
    args = [c for c in node if c.tag != 'ty']
    op = node.get('type')
    if op == '.fieldderef':
        return fields[args[1].get('val')]
    values = [predicate(c, fields) for c in args]
    if op == '.cons_array':
        return values
    if op == '&&':
        return all(values)
    if op == '==':
        return values[0] == values[1]
    if op == '!=':
        return values[0] != values[1]
    if op == '.within':
        return values[0] in values[1]
    if op == '.notwithin':
        return values[0] not in values[1]
    raise ValueError(f'unsupported XML predicate {op}')


def put(bits, ranges, value):
    for offset, width in ranges:
        mask = (1 << width) - 1
        bits = (bits & ~(mask << offset)) | ((value & mask) << offset)
        value >>= width
    if value:
        raise ValueError('field value exceeds XML bit layout')
    return bits


def candidate(row, encoding, condition):
    fields = row['fields']
    values = {name: 0 for name in fields}
    for name in ('SRC0', 'SRC1', 'SRC2'):
        if name in values:
            values[name] = 128
    if 'LITERAL' in values:
        values['LITERAL'] = 0x41234567
    # VGPR sources avoid scalar-bus constraints. Even register bases also work
    # for paired operands; scalar selector zero avoids unsupported inline values.
    for operand in row['operands']:
        name = operand.get('FieldName')
        if name in fields and name != 'LITERAL' and operand.get('OperandType') in ('OPR_SRC', 'OPR_SRC_VGPR'):
            values[name] = 256
    # Solve the actual XML condition rather than guessing from its human name.
    expression = list(condition)[0]
    refs = sorted({x[1].get('val') for x in expression.iter('op')
                   if x.get('type') == '.fieldderef' and x[1].get('val') in fields})
    literals = sorted({int(x.get('val'), 0) for x in expression.iter('lit')})
    domains = [[values[n]] + [x for x in literals if x != values[n]
                             and x < 1 << sum(w for _, w in fields[n])]
               for n in refs]
    for combination in itertools.product(*domains):
        chosen = values | dict(zip(refs, combination))
        if predicate(expression, chosen):
            values = chosen
            break
    else:
        raise ValueError('no satisfying candidate for XML condition')
    mask = number(encoding.find('EncodingIdentifierMask'))
    ids = [number(x) for x in encoding.findall('EncodingIdentifiers/EncodingIdentifier')]
    opfield = 'OP' if 'OP' in fields else 'OPX' if 'OPX' in fields else None
    matches = ids
    if opfield:
        opmask = put(0, fields[opfield], (1 << sum(w for _, w in fields[opfield])) - 1)
        opbits = put(0, fields[opfield], row['opcode'])
        matches = [x for x in ids if x & opmask == opbits & mask & opmask]
        values[opfield] = row['opcode']
    if not matches:
        raise ValueError('opcode has no XML encoding identifier')
    bits = matches[0]
    for name, value in values.items():
        if name != 'ENCODING':
            bits = put(bits, fields[name], value)
    if bits & mask not in ids:
        raise ValueError('candidate does not match XML encoding identifier')
    return bits.to_bytes(row['bits'] // 8, 'little')


def oracle(mc, cpu, row, data):
    args = [str(mc), '--triple=amdgcn-amd-amdhsa', f'--mcpu={cpu}']
    raw = ' '.join(f'0x{x:02x}' for x in data) + '\n'
    dec = subprocess.run(args + ['--disassemble'], input=raw, text=True,
                         capture_output=True, check=False)
    lines = [x.strip() for x in dec.stdout.splitlines()
             if x.strip() and not x.strip().startswith('.')]
    result = {'candidate': data.hex(), 'decoded': lines}
    names = {row['name'], *row['aliases']}
    normalized = lambda s: re.sub(r'(?:_e(?:32|64))?(?:_dpp)?$', '', s)
    if dec.returncode or dec.stderr.strip() or len(lines) != 1:
        return result | {'status': 'decode-rejected', 'diagnostic': dec.stderr.strip()}
    if normalized(lines[0].split()[0]) not in names:
        return result | {'status': 'name-mismatch'}
    enc = subprocess.run(args + ['--show-encoding'], input=lines[0] + '\n',
                         text=True, capture_output=True, check=False)
    chunks = re.findall(r'encoding: \[([^]]+)\]', enc.stdout)
    encoded = bytes(int(v.strip(), 16) for c in chunks for v in c.split(',')
                    if re.fullmatch(r'0x[0-9a-fA-F]{2}', v.strip()))
    if enc.returncode or not chunks:
        return result | {'status': 'encode-rejected', 'diagnostic': enc.stderr.strip()}
    return result | {'status': 'verified' if encoded == data else 'byte-mismatch',
                     'reencoded': encoded.hex()}


def generate(source, mc, target, xmlname, jobs):
    raw = (source / f'amdgpu_isa_{xmlname}.xml').read_bytes()
    root = ET.fromstring(raw)
    encodings = {x.findtext('EncodingName'): x for x in root.findall('./ISA/Encodings/Encoding')}
    rows = []
    for inst in root.findall('./ISA/Instructions/Instruction'):
        for variant, spec in enumerate(inst.findall('InstructionEncodings/InstructionEncoding')):
            enc = encodings[spec.findtext('EncodingName')]
            condname = spec.findtext('EncodingCondition')
            cond = next((x.find('CondtionExpression') for x in enc.findall('EncodingConditions/EncodingCondition')
                         if x.findtext('ConditionName') == condname), None)
            operands = [dict(x.attrib) | {c.tag: c.text for c in x} for x in spec.findall('Operands/Operand')]
            fields = {x.findtext('FieldName'): [[int(r.findtext('BitOffset')), int(r.findtext('BitCount'))]
                      for r in sorted(x.findall('BitLayout/Range'), key=lambda r: int(r.get('Order')))]
                      for x in enc.findall('MicrocodeFormat/BitMap/Field')}
            memory = [x for x in operands if x.get('OperandType') in ('OPR_GPUMEM', 'OPR_DSMEM')]
            row = {'name': inst.findtext('InstructionName').lower(),
                   'aliases': [x.text.lower() for x in inst.findall('AliasedInstructionNames/InstructionName')],
                   'encoding': spec.findtext('EncodingName'), 'condition': condname,
                   'variant': variant,
                   'codec_status': ('unsupported-graphics' if spec.findtext('EncodingName')
                                    in GRAPHICS_ENCODINGS else 'metadata-only'),
                   'opcode': number(spec.find('Opcode')), 'bits': int(enc.findtext('BitCount')),
                   'identifier_mask': hex(number(enc.find('EncodingIdentifierMask'))),
                   'identifiers': [hex(number(x)) for x in enc.findall('EncodingIdentifiers/EncodingIdentifier')],
                   'predicate': tree(cond) if cond is not None else None,
                   'fields': fields, 'operands': operands,
                   'defs': [i for i, x in enumerate(operands) if x.get('Output') == 'true'],
                   'uses': [i for i, x in enumerate(operands) if x.get('Input') == 'true'],
                   'implicit': [i for i, x in enumerate(operands) if x.get('IsImplicit') == 'true'],
                   'memory': memory,
                   'group': inst.findtext('FunctionalGroup/Name'),
                   'subgroups': [x.text for x in inst.findall('FunctionalGroup/FunctionalSubgroups/Subgroup')],
                   'flags': {x.tag: x.text == 'TRUE' for x in inst.findall('InstructionFlags/*')}}
            try:
                if cond is None:
                    raise ValueError(f'XML references undefined condition {condname}')
                data = candidate(row, enc, cond)
            except ValueError as error:
                data = None
                row['oracle'] = {'status': 'candidate-unavailable', 'diagnostic': str(error)}
            rows.append((row, data))
    cpu = 'gfx1201' if target == 'gfx12' else target
    def check(pair):
        row, data = pair
        if data is not None:
            row['oracle'] = oracle(mc, cpu, row, data)
        return row
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        checked = list(pool.map(check, rows))
    header = {'schema': 1, 'target': target, 'xml': f'amdgpu_isa_{xmlname}.xml',
              'sha256': hashlib.sha256(raw).hexdigest(),
              'llvm': subprocess.check_output([str(mc), '--version'], text=True).splitlines()[0],
              'llvm_sha256': hashlib.sha256(mc.read_bytes()).hexdigest(),
              'key': ['name', 'encoding', 'condition', 'opcode', 'variant'],
              'instructions': len(root.findall('./ISA/Instructions/Instruction')), 'rows': len(rows)}
    layouts = {}
    for row in checked:
        name = row['encoding']
        layout = {key: row.pop(key) for key in
                  ('bits', 'identifier_mask', 'identifiers', 'fields')}
        expression = row.pop('predicate')
        if name not in layouts:
            layouts[name] = layout | {'conditions': {}}
        elif any(layouts[name][key] != value for key, value in layout.items()):
            raise ValueError('inconsistent XML encoding layout')
        layouts[name]['conditions'][row['condition']] = expression
    header['encodings'] = layouts
    keys = [tuple(r[x] for x in header['key']) for r in checked]
    if len(set(keys)) != len(keys):
        raise ValueError('duplicate full XML encoding key')
    return header, checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--xml-dir', type=Path)
    parser.add_argument('--llvm-mc', required=True, type=Path)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parents[1] / 'crates/peacemaker-ir/isa')
    parser.add_argument('--target', choices=TARGETS, action='append')
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--sop1-evidence', action='store_true',
                        help='write sop1-overrides.json from the committed catalogs only')
    args = parser.parse_args()
    if args.sop1_evidence:
        document = sop1_evidence(args.llvm_mc, args.output_dir, args.jobs)
        (args.output_dir / SOP1_EVIDENCE).write_text(
            json.dumps(document, sort_keys=True, indent=1) + '\n')
        for target, entry in document['targets'].items():
            print(target, dict(Counter((r['status'], r['unavailable_reason']) for r in entry['rows'])))
        return
    if args.xml_dir is None:
        parser.error('--xml-dir is required unless --sop1-evidence is given')
    for target in args.target or TARGETS:
        header, rows = generate(args.xml_dir, args.llvm_mc, target, TARGETS[target], args.jobs)
        path = args.output_dir / f'{target}.xml.jsonl'
        with path.open('w') as stream:
            for row in [header, *rows]:
                stream.write(json.dumps(row, sort_keys=True, separators=(',', ':')) + '\n')
        print(target, header['rows'], dict(Counter(r['oracle']['status'] for r in rows)))


if __name__ == '__main__':
    main()
