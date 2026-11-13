#!/usr/bin/env python3
"""Cross-check every XML instruction encoding against LLVM's independent codec.

Produces one JSON receipt per XML row, including failures; never silently drops rows.
XML-only means LLVM explicitly rejects the mnemonic, not merely a bad seed.
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


def number(node):
    return int(node.text, int(node.get('Radix', '10')))


def fields(encoding):
    return {f.findtext('FieldName'): [(int(b.findtext('BitOffset')), int(b.findtext('BitCount'))) for b in f.findall('./BitLayout/Range')] for f in encoding.findall('./MicrocodeFormat/BitMap/Field')}


def insert(bits, layout, value):
    for offset, width in layout:
        mask = (1 << width) - 1
        bits = (bits & ~(mask << offset)) | ((value & mask) << offset)
        value >>= width
    return bits


def evaluate(node, values):
    if node.tag == 'lit':
        return int(node.get('val'), 0)
    args = [c for c in node if c.tag != 'ty']
    op = node.get('type')
    if op == '.fieldderef':
        return values.get(args[1].get('val'), 0)
    vals = [evaluate(c, values) for c in args]
    if op == '==':
        return vals[0] == vals[1]
    if op == '!=':
        return vals[0] != vals[1]
    if op == '&&':
        return all(vals)
    if op == '.cons_array':
        return vals
    if op == '.notwithin':
        return vals[0] not in vals[1]
    raise ValueError(f'unsupported XML expression {op}')


def candidates(row, enc):
    fs = fields(enc)
    opcode = int(row.findtext('Opcode'))
    mask = number(enc.find('EncodingIdentifierMask'))
    ids = [number(e) for e in enc.findall('./EncodingIdentifiers/EncodingIdentifier')]
    base = insert(0, fs.get('OP', []), opcode)
    ids = [v for v in ids if (v & sum(((1 << w)-1) << o for o,w in fs.get('OP', []))) == base]
    if not ids:
        raise ValueError(f'no identifier for opcode {opcode}')
    condition = row.findtext('EncodingCondition')
    exprs = [c.find('CondtionExpression')[0] for c in enc.findall('./EncodingConditions/EncodingCondition') if c.findtext('ConditionName') == condition]
    names = sorted({n.get('val') for expr in exprs for n in expr.findall('.//op[@type=".fieldderef"]/id')} & fs.keys())
    choices = [0, 1, 128, 249, 250, 255, 256]
    used = {o.findtext('FieldName') for o in row.findall('./Operands/Operand')}
    for variant in itertools.product(choices, repeat=len(names)):
        vals = dict(zip(names, variant))
        if exprs and not any(evaluate(e, vals) for e in exprs):
            continue
        for seed, gds in itertools.product((0, 256), (0, 1)):
            bits = ids[0]
            for name, layout in fs.items():
                value = vals.get(name, 0)
                if name.startswith('SRC') and name not in vals and name in used:
                    operand = next(o for o in row.findall('./Operands/Operand') if o.findtext('FieldName') == name)
                    ty = operand.findtext('OperandType')
                    value = 0 if ty.startswith('OPR_SREG') or ty.startswith('OPR_SSRC') else seed
                if name == 'GDS':
                    value = gds
                if name == 'DPP_CTRL':
                    value = 0x150
                if name == 'OP_SEL_HI':
                    value = 7
                if name == 'SEG' and enc.findtext('EncodingName') == 'ENC_FLAT_GLBL':
                    value = 2
                if name == 'SEG' and enc.findtext('EncodingName') == 'ENC_FLAT_SCRATCH':
                    value = 1
                bits = insert(bits, layout, value) if name not in ('ENCODING', 'OP') else bits
            bits = (bits & ~mask) | ids[0]
            yield bits.to_bytes((int(enc.findtext('BitCount')) + 7)//8, 'little')


def run(mc, arch, data, disassemble=False):
    args = [mc, '-triple=amdgcn-amd-amdhsa', f'-mcpu={arch}', '--show-encoding']
    if disassemble:
        args.append('--disassemble')
    return subprocess.run(args, input=data, text=True, capture_output=True)


def check(job):
    mc, arch, name, row, enc, index = job
    receipt = dict(index=index, name=name, encoding=row.findtext('EncodingName'), condition=row.findtext('EncodingCondition'), opcode=int(row.findtext('Opcode')))
    diagnostic = ''
    try:
        for raw in candidates(row, enc):
            dis = run(mc, arch, ' '.join(f'0x{b:02x}' for b in raw)+'\n', True)
            lines = [l.strip().split(';')[0].strip() for l in dis.stdout.splitlines() if l.strip() and not l.strip().startswith('.')]
            if dis.returncode or 'warning:' in dis.stderr or len(lines) != 1:
                diagnostic = dis.stderr.strip() or dis.stdout.strip()
                continue
            text = lines[0]
            mnemonic = text.split()[0]
            # LLVM aliases are recorded, not silently treated as the requested row.
            if re.sub(r'_(e32|e64|sdwa|dpp)$', '', mnemonic) != name.lower() and mnemonic != name.lower() + '_b32':
                diagnostic = f'XML mnemonic {name} decoded as {text}'
                continue
            asm = run(mc, arch, text+'\n')
            match = re.search(r'encoding:\s*\[([^]]+)\]', asm.stdout)
            if asm.returncode or not match:
                diagnostic = asm.stderr.strip() or asm.stdout.strip()
                continue
            encoded = bytes(int(b.strip(), 16) for b in match.group(1).split(','))
            if encoded != raw:
                encoded_bits = int.from_bytes(encoded, 'little')
                raw_bits = int.from_bytes(raw, 'little')
                if len(encoded) != len(raw) or (encoded_bits & number(enc.find('EncodingIdentifierMask'))) != (raw_bits & number(enc.find('EncodingIdentifierMask'))):
                    diagnostic = f'form/opcode mismatch: {raw.hex()} != {encoded.hex()} ({text})'
                    continue
                # LLVM canonicalizes ignored fields; verify its canonical seed,
                # retaining the original seed in the receipt for audit.
                redis = run(mc, arch, ' '.join(f'0x{b:02x}' for b in encoded)+'\n', True)
                relines = [l.strip().split(';')[0].strip() for l in redis.stdout.splitlines() if l.strip() and not l.strip().startswith('.')]
                if redis.returncode or 'warning:' in redis.stderr or relines != [text]:
                    diagnostic = f'canonical disassembly mismatch: {redis.stdout}'
                    continue
                receipt['initial_bytes'] = raw.hex()
                raw = encoded
            receipt.update(status='verified', bytes=raw.hex(), assembly=text)
            return receipt
        probe = run(mc, arch, name.lower()+'\n')
        status = 'xml-only' if 'invalid instruction' in probe.stderr or 'instruction not supported' in probe.stderr else 'unresolved'
        receipt.update(status=status, diagnostic=diagnostic, mnemonic_probe=probe.stderr.strip())
    except (ValueError, TypeError, OverflowError) as error:
        receipt.update(status='unresolved', diagnostic=str(error))
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--xml', type=Path, required=True)
    p.add_argument('--arch', choices=['gfx942', 'gfx950'], required=True)
    p.add_argument('--llvm-mc', default='/opt/rocm/llvm/bin/llvm-mc')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--jobs', type=int, default=4)
    args = p.parse_args()
    root = ET.parse(args.xml).getroot()
    encs = {e.findtext('EncodingName'): e for e in root.findall('./ISA/Encodings/Encoding')}
    jobs = [(args.llvm_mc, args.arch, i.findtext('InstructionName'), row, encs[row.findtext('EncodingName')], n) for n,(i,row) in enumerate((i,row) for i in root.findall('./ISA/Instructions/Instruction') for row in i.findall('./InstructionEncodings/InstructionEncoding'))]
    version = subprocess.run([args.llvm_mc, '--version'], text=True, capture_output=True, check=True).stdout
    with ThreadPoolExecutor(args.jobs) as pool:
        rows = list(pool.map(check, jobs))
    for row in rows:
        row['verified'] = row['status'] == 'verified'
        if not row['verified']:
            row['verification'] = 'unverified(' + row['status'] + ': ' + row.get('diagnostic', '') + ')'
    counts = {status:sum(r['status']==status for r in rows) for status in ('verified','xml-only','unresolved')}
    output = dict(xml_sha256=hashlib.sha256(args.xml.read_bytes()).hexdigest(), arch=args.arch, llvm_version=version, counts=counts, rows=rows)
    args.output.write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps(counts))
    return bool(counts['unresolved'])


if __name__ == '__main__':
    raise SystemExit(main())
