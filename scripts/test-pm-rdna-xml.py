#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
"""Offline integrity checks for committed RDNA XML catalogs (no ROCm required)."""
import hashlib
import json
from pathlib import Path
import unittest

ISA = Path(__file__).resolve().parents[1] / 'crates/peacemaker-ir/isa'
PINS = {
    'gfx1100': (4929, '6eee5f8737172adf08c0e7d5994ea916e9555b9c23ccb9ec80d9ee38678733b4',
                '0d0c0634725c98f02180e07236dfb94660ff66c9d8aace1577d203444d1d7aff'),
    'gfx1151': (5056, 'c36b6d79b1e940d74107221c985f5a7fde248025da251d2c6ef756c4cd31391a',
                'eeb3919edb4731629ddf90399e76f09da95f9a5d4a60fdf92363a15b2b0caeb9'),
    'gfx12': (5031, 'f8a290c8471e26a1071b08b61a33e4d9efa46ec6cedcdb8e5ba57d7969b60692',
              '05c5d1a0828d78501298109a902afa61092ec3a7bfa0d8a5c623af73920f2484'),
}


class CatalogIntegrity(unittest.TestCase):
    def test_census_rows_remain_byte_identical(self):
        for target, (_, xml_hash, census_hash) in PINS.items():
            with self.subTest(target=target):
                data = (ISA / f'{target}.tbl').read_bytes()
                self.assertIn(xml_hash.encode(), data.splitlines()[0])
                rows = b''.join(line for line in data.splitlines(keepends=True)
                                if line.strip() and not line.startswith(b'#'))
                self.assertEqual(hashlib.sha256(rows).hexdigest(), census_hash)

    def test_complete_catalog_structure(self):
        for target, (count, xml_hash, _) in PINS.items():
            with self.subTest(target=target):
                header, *rows = [json.loads(line) for line in
                                 (ISA / f'{target}.xml.jsonl').read_text().splitlines()]
                self.assertEqual(header['sha256'], xml_hash)
                self.assertEqual(header['rows'], count)
                self.assertEqual(len(rows), count)
                keys = {tuple(row[k] for k in header['key']) for row in rows}
                self.assertEqual(len(keys), count)
                for row in rows:
                    layout = header['encodings'][row['encoding']]
                    self.assertIn(row['condition'], layout['conditions'])
                    operands = row['operands']
                    for attribute, key in [('Output', 'defs'), ('Input', 'uses'),
                                           ('IsImplicit', 'implicit')]:
                        self.assertEqual(row[key], [i for i, op in enumerate(operands)
                                                    if op[attribute] == 'true'])
                    self.assertEqual(row['memory'], [op for op in operands if
                                     op['OperandType'] in ('OPR_GPUMEM', 'OPR_DSMEM')])
                    result = row['oracle']
                    if result['status'] == 'verified':
                        self.assertEqual(result['candidate'], result['reencoded'])
                        self.assertEqual(len(bytes.fromhex(result['candidate'])) * 8,
                                         layout['bits'])
                        self.assertEqual(len(result['decoded']), 1)
                    if layout['conditions'][row['condition']] is None:
                        self.assertEqual(result['status'], 'candidate-unavailable')
                        self.assertIn('undefined condition', result['diagnostic'])


if __name__ == '__main__':
    unittest.main()
