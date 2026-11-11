#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
"""Offline integrity checks for committed RDNA XML catalogs (no ROCm required)."""
import hashlib
import json
import importlib.util
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

SOP1_COUNTS = {'gfx1100': 63, 'gfx1151': 78, 'gfx12': 82}
SOP1_DOMAIN = {'s_movrels_b32', 's_movrels_b64', 's_setpc_b64', 's_rfe_b64'}
SOP1_DOMAIN_GFX12 = SOP1_DOMAIN | {'s_barrier_signal', 's_barrier_signal_isfirst'}
SOP1_PAIR_ONLY = {'s_movrels_b64', 's_setpc_b64', 's_rfe_b64'}
SPEC = importlib.util.spec_from_file_location('pm_rdna_xml', Path(__file__).with_name('pm-rdna-xml.py'))
GENERATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GENERATOR)


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


class Sop1Overrides(unittest.TestCase):
    def setUp(self):
        self.document = json.loads((ISA / 'sop1-overrides.json').read_text())

    def test_table_enumerates_every_undefined_condition_row(self):
        self.assertEqual(set(GENERATOR.SOP1_OVERRIDES), set(PINS))
        for target, (_, xml_hash, _) in PINS.items():
            with self.subTest(target=target):
                table = GENERATOR.SOP1_OVERRIDES[target]
                header, *catalog = [json.loads(line) for line in
                                    (ISA / f'{target}.xml.jsonl').read_text().splitlines()]
                self.assertEqual(table['xml_sha256'], xml_hash)
                self.assertEqual(self.document['targets'][target]['xml_sha256'], xml_hash)
                expected = [r['name'] for r in catalog if r['encoding'] == 'ENC_SOP1'
                            and r['condition'] == GENERATOR.SOP1_CONDITION]
                self.assertEqual(len(expected), SOP1_COUNTS[target])
                self.assertEqual(list(table['rows']), expected)
                self.assertEqual([r['name'] for r in self.document['targets'][target]['rows']], expected)
                # The source catalog keeps the original undefined condition untouched.
                layout = header['encodings']['ENC_SOP1']
                self.assertIsNone(layout['conditions'][GENERATOR.SOP1_CONDITION])
                for row in catalog:
                    if row['name'] in expected and row['encoding'] == 'ENC_SOP1' \
                            and row['condition'] == GENERATOR.SOP1_CONDITION:
                        self.assertEqual(row['oracle']['status'], 'candidate-unavailable')

    def test_predicates_follow_llvm_proofs(self):
        predicates = self.document['predicates']
        for target, entry in self.document['targets'].items():
            in_domain = SOP1_DOMAIN_GFX12 if target == 'gfx12' else SOP1_DOMAIN
            for row in entry['rows']:
                with self.subTest(target=target, row=row['name']):
                    proofs = row['proofs']
                    self.assertEqual(set(proofs), {'register', 'inline', 'literal'})
                    self.assertEqual(row['xml_condition'], GENERATOR.SOP1_CONDITION)
                    self.assertIsNone(row['xml_predicate'])
                    for proof in proofs.values():
                        if proof['status'] == 'verified':
                            self.assertEqual(proof['candidate'], proof['reencoded'])
                            self.assertEqual(len(proof['decoded']), 1)
                    self.assertEqual(row['status'], 'corrected')
                    self.assertIsNone(row['unavailable_reason'])
                    if row['name'] in in_domain:
                        domain = row['domain']
                        self.assertEqual(row['corrected_predicate'], 'ssrc0-in-domain')
                        self.assertEqual(domain['probed'], 256)
                        self.assertTrue(domain['selectors'])
                        self.assertEqual(row['domain_ranges'], domain['selectors'])
                        self.assertTrue(all(0 <= lo <= hi <= 511 for lo, hi in row['domain_ranges']))
                        self.assertTrue(all(a[1] < b[0] for a, b in zip(row['domain_ranges'],
                                                                         row['domain_ranges'][1:])))
                        self.assertEqual(predicates['ssrc0-in-domain'],
                                         {'field': 'SSRC0', 'op': 'in-ranges'})
                        union = sorted(x for form in domain['by_form'].values()
                                       for lo, hi in form['selectors'] for x in range(lo, hi + 1))
                        self.assertEqual(union, [x for lo, hi in domain['selectors']
                                                 for x in range(lo, hi + 1)])
                        for proof in proofs.values():
                            allowed = any(lo <= proof['ssrc0'] <= hi for lo, hi in domain['selectors'])
                            self.assertEqual(proof['status'] == 'verified', allowed)
                        if row['name'] in SOP1_PAIR_ONLY:
                            self.assertEqual(domain['by_form']['inline']['selectors'], [])
                            self.assertEqual(domain['by_form']['literal']['selectors'], [])
                            self.assertTrue(all(lo == hi and lo % 2 == 0
                                                for lo, hi in domain['selectors']))
                        self.assertIn('ssrc0-in-domain', predicates)
                        continue
                    self.assertNotIn('domain', row)
                    self.assertNotIn('domain_ranges', row)
                    self.assertEqual(proofs['register']['status'], 'verified')
                    if row['name'] == 's_getpc_b64':
                        self.assertEqual(row['corrected_predicate'], 'ssrc0-zero')
                        self.assertNotEqual(proofs['inline']['status'], 'verified')
                        self.assertNotEqual(proofs['literal']['status'], 'verified')
                    else:
                        self.assertEqual(row['corrected_predicate'], 'ssrc0-not-literal')
                        self.assertEqual(proofs['inline']['status'], 'verified')
                    self.assertIn(row['corrected_predicate'], predicates)

    def test_ordinary_rows_prove_literal_form(self):
        for target, entry in self.document['targets'].items():
            for row in entry['rows']:
                if row['corrected_predicate'] == 'ssrc0-not-literal':
                    with self.subTest(target=target, row=row['name']):
                        self.assertEqual(row['proofs']['literal']['status'], 'verified')
                        self.assertEqual(len(row['proofs']['literal']['candidate']), 16)


if __name__ == '__main__':
    unittest.main()
