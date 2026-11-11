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
"""
import argparse
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
    parser.add_argument('--xml-dir', required=True, type=Path)
    parser.add_argument('--llvm-mc', required=True, type=Path)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parents[1] / 'crates/peacemaker-ir/isa')
    parser.add_argument('--target', choices=TARGETS, action='append')
    parser.add_argument('--jobs', type=int, default=4)
    args = parser.parse_args()
    for target in args.target or TARGETS:
        header, rows = generate(args.xml_dir, args.llvm_mc, target, TARGETS[target], args.jobs)
        path = args.output_dir / f'{target}.xml.jsonl'
        with path.open('w') as stream:
            for row in [header, *rows]:
                stream.write(json.dumps(row, sort_keys=True, separators=(',', ':')) + '\n')
        from collections import Counter
        print(target, header['rows'], dict(Counter(r['oracle']['status'] for r in rows)))


if __name__ == '__main__':
    main()
