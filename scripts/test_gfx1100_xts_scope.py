# SPDX-License-Identifier: Apache-2.0
"""CPU-only source-contract checks for the gfx1100 HIN and half-V changes.

Read the production sources, not an experimental source transformer or cache.
These static checks and address/order models do NOT establish GPU numerical
parity, synchronization correctness, or performance. Run the GPU validation
route separately. No compiler, HIP runtime, model, or third-party Python is used.

Run: python3 scripts/test_gfx1100_xts_scope.py
"""

import ast
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
HIN = "FUSED_SILU_MUL_MQ_ROTATE_AWQ_I4_HIN_SRC"
ONEPASS = "HIPFIRE_IU4_C2_ONEPASS"


def compact(source):
    """Ignore comments and formatting in HIP source-contract assertions."""
    source = re.sub(r"/\*.*?\*/|//[^\n]*", "", source, flags=re.S)
    return " ".join(source.split())


class HinScopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = (ROOT / "crates/rdna-compute/src/kernels.rs").read_text()
        matches = re.findall(
            r"pub const " + HIN + r": &str = concat!\((.*?)\n\);",
            cls.source,
            re.S,
        )
        if len(matches) != 1:
            raise AssertionError("expected exactly one production HIN constant")
        cls.body = matches[0]

    def test_production_hin_prefix_admits_only_explicit_gfx1100_c2(self):
        first = re.match(r'\s*("(?:[^"\\]|\\.)*")\s*,', self.body)
        self.assertIsNotNone(first, "HIN must begin with its local admission gate")
        prefix = ast.literal_eval(first.group(1))
        self.assertEqual(
            prefix,
            "#if defined(__gfx1100__) && defined(IU4_A4_CANDIDATES) "
            "&& IU4_A4_CANDIDATES == 2\n"
            "#define HIPFIRE_IU4_C2_ONEPASS 1\n#endif\n",
        )
        # Admission precedes the shared quantizer; its default C8 must not
        # accidentally become C2 when the host omits its compile flag.
        self.assertLess(
            self.body.index(ONEPASS), self.body.index("block_i4_128_quant.hip")
        )
        self.assertIn('"#define HIPFIRE_SILU_HIN 1\\n"', self.body)
        self.assertIn(
            "fused_silu_mul_mq_rotate_awq_i4_hin\\n", self.body
        )

    def test_onepass_optin_does_not_leak_to_other_production_constants(self):
        self.assertEqual(self.body.count(ONEPASS), 1)
        self.assertNotIn(ONEPASS, self.source.replace(self.body, "", 1))
        self.assertNotRegex(self.body, r"#define\s+__gfx\w+")

    def test_shared_quantizer_does_not_globally_admit_gfx1100(self):
        source = (ROOT / "kernels/src/block_i4_128_quant.hip").read_text()
        gates = re.findall(
            r"(#if [^\n]*(?:\\\n[^\n]*)*)\n#define " + ONEPASS + r" 1\n#endif",
            source,
        )
        self.assertEqual(len(gates), 1)
        self.assertEqual(
            compact(gates[0].replace("\\\n", " ")),
            "#if IU4_A4_CANDIDATES == 2 && (defined(__gfx1201__) "
            "|| (defined(HIPFIRE_IU4_ONEPASS) && defined(__gfx1151__)))",
        )


class HalfVScopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = (
            ROOT / "kernels/src/attention_flash_q8_0_tile_gqa.gfx1100.hip"
        ).read_text()
        cls.code = compact(cls.source)

    def test_production_slab_layout_and_partial_chunk_guards(self):
        for contract in (
            "#define GQA_TILE 128",
            "#define GQA_V_TILE 64",
            "#define GQA_HD 256",
            "#define GQA_WAVES 8",
            "#define GQA_THREADS (GQA_WAVES * 32)",
            "#define GQA_BLK 34",
            "#define GQA_SEG (GQA_HD / 32 * GQA_BLK)",
            "#define GQA_V_CHUNKS ((GQA_V_TILE * GQA_SEG / 8 + GQA_THREADS - 1) / GQA_THREADS)",
            "unsigned int v_lds[GQA_V_TILE * GQA_SEG / 4];",
        ):
            self.assertIn(contract, self.code)
        # Both the initial load/store and the refill need to mask the
        # half-full ninth chunk; an unguarded path reads uninitialized vr.
        self.assertEqual(
            self.code.count("if (c >= GQA_V_TILE * GQA_SEG / 8) continue;"), 2
        )
        self.assertEqual(
            self.code.count(
                "if (c < GQA_V_TILE * GQA_SEG / 8) *(uint2*)(v_lds + 2 * c) = vr[i];"
            ),
            2,
        )

    def test_production_refill_barriers_and_increasing_key_order(self):
        for contract in (
            "for (int vstart = 0; vstart < tile_len; vstart += GQA_V_TILE)",
            "const int vend = min(tile_len, vstart + GQA_V_TILE);",
            "if (vstart != 0) { __syncthreads();",
            "(unsigned)min(seg, tile_len - 1) * row_stride + head_off",
            "(unsigned)(vstart + min(seg, tile_len - vstart - 1)) * row_stride + head_off",
            "__syncthreads(); } int t = vstart;",
            "for (; t + 3 < vend; t += 4)",
            "gqa_accum_key(vb + (t - vstart + k) * GQA_SEG, soff, coff, w, acc);",
            "for (; t < vend; ++t)",
            "gqa_accum_key(vb + (t - vstart) * GQA_SEG, soff, coff, w, acc);",
            "if (tile_id != (int)blockIdx.y) __syncthreads();",
        ):
            self.assertIn(contract, self.code)

    def test_all_tile_tails_cover_each_key_once_with_bounded_slab_addresses(self):
        # The source-contract tests above pin the constants and expressions
        # modeled here; this remains a CPU address model, not HIP execution.
        slab, segment_bytes, threads = 64, 272, 256
        count = slab * segment_bytes // 8
        chunks = (count + threads - 1) // threads
        addresses = [
            tid + threads * i
            for tid in range(threads)
            for i in range(chunks)
            if tid + threads * i < count
        ]
        self.assertEqual(chunks, 9)
        self.assertEqual(sorted(addresses), list(range(count)))
        for length in range(1, 129):
            visited = []
            for start in range(0, length, slab):
                end = min(length, start + slab)
                t = start
                while t + 3 < end:
                    for k in range(4):
                        self.assertLess(t - start + k, slab)
                        visited.append(t + k)
                    t += 4
                while t < end:
                    self.assertLess(t - start, slab)
                    visited.append(t)
                    t += 1
                for c in addresses:
                    seg, col = divmod(c, segment_bytes // 8)
                    row = start + min(seg, length - start - 1)
                    self.assertTrue(start <= row < length)
                    self.assertLess(col * 8 + 7, segment_bytes)
            self.assertEqual(visited, list(range(length)))


if __name__ == "__main__":
    unittest.main()
