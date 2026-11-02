#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU-only regression tests for local-link validation in
scripts/check-crate-maps.py, run against throwaway crates in a temp dir."""

import contextlib
import importlib.util
import io
import tempfile
import unittest
import unittest.mock
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "check-crate-maps.py"
_spec = importlib.util.spec_from_file_location("check_crate_maps", SCRIPT)
cm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cm)

HAND_WRITTEN_HEAD = "# demo — map\n\n## Purpose\n\nFacade over the engine.\n\n"
HAND_WRITTEN_TAIL = "\n## Gotchas\n\nNothing yet.\n"


class CrateMapLinkTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.crates = self.root / "crates"
        self.crate = self.crates / "demo"
        (self.crate / "src").mkdir(parents=True)
        (self.crate / "Cargo.toml").write_text(
            '[package]\nname = "demo"\n\n[dependencies]\nserde = "1"\n'
        )
        (self.crate / "src" / "lib.rs").write_text("pub fn facade() {}\n")
        (self.crate / "src" / "engine.rs").write_text("pub struct Engine;\n")
        (self.root / "docs").mkdir()
        (self.root / "docs" / "ARCH.md").write_text("# arch\n")

        for name, value in (("ROOT", self.root), ("CRATES", self.crates)):
            patcher = unittest.mock.patch.object(cm, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def write_map(self, head: str = HAND_WRITTEN_HEAD, tail: str = HAND_WRITTEN_TAIL,
                  block: str | None = None) -> None:
        block = cm.generated_block(self.crate) if block is None else block
        (self.crate / "map.md").write_text(head + block + "\n" + tail)

    def run_check(self) -> tuple[int, str]:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = cm.main(["--check"])
        return code, out.getvalue()

    def test_fresh_map_with_valid_handwritten_links_passes(self) -> None:
        self.write_map(head=HAND_WRITTEN_HEAD + "See [facade](src/lib.rs) and "
                       "[arch](../../docs/ARCH.md).\n\n")
        self.assertEqual(cm.check_crate(self.crate), [])
        code, _ = self.run_check()
        self.assertEqual(code, 0)

    def test_fresh_block_with_broken_handwritten_link_fails(self) -> None:
        # The generated block is byte-for-byte fresh; only prose outside the
        # markers carries the dead link (a `llama.rs` that does not exist).
        self.write_map(head=HAND_WRITTEN_HEAD + "Backend: [llama](src/llama.rs).\n\n")
        text = (self.crate / "map.md").read_text()
        old_block = text[text.index(cm.BEGIN):text.index(cm.END) + len(cm.END)]
        self.assertEqual(old_block, cm.generated_block(self.crate))

        problems = cm.check_crate(self.crate)
        self.assertEqual(len(problems), 1, problems)
        self.assertIn("demo", problems[0])
        self.assertIn("map.md:7", problems[0])
        self.assertIn("src/llama.rs", problems[0])

        code, out = self.run_check()
        self.assertEqual(code, 1)
        self.assertIn("src/llama.rs", out)

    def test_handwritten_content_after_markers_is_checked_too(self) -> None:
        self.write_map(tail=HAND_WRITTEN_TAIL + "\nSee [gone](src/gone.rs).\n")
        problems = cm.check_crate(self.crate)
        self.assertEqual(len(problems), 1, problems)
        self.assertIn("src/gone.rs", problems[0])

    def test_altered_generated_href_with_valid_label_fails(self) -> None:
        block = cm.generated_block(self.crate)
        tampered = block.replace("](src/engine.rs)", "](src/missing.rs)")
        self.assertNotEqual(block, tampered)
        self.write_map(block=tampered)

        problems = cm.check_crate(self.crate)
        joined = "\n".join(problems)
        self.assertIn("src/missing.rs", joined)
        self.assertIn("src/engine.rs", joined)
        code, _ = self.run_check()
        self.assertEqual(code, 1)

    def test_altered_generated_href_pointing_at_existing_file_fails(self) -> None:
        block = cm.generated_block(self.crate)
        tampered = block.replace("](src/engine.rs)", "](src/lib.rs)")
        self.write_map(block=tampered)
        problems = cm.check_crate(self.crate)
        self.assertTrue(problems)
        self.assertTrue(any("src/engine.rs" in p and "src/lib.rs" in p for p in problems))

    def test_valid_facade_lib_rs_link_passes(self) -> None:
        self.write_map(head=HAND_WRITTEN_HEAD + "Docs live in [`src/lib.rs`](src/lib.rs).\n\n")
        self.assertEqual(cm.check_crate(self.crate), [])

    def test_anchors_fragments_and_external_links(self) -> None:
        head = (
            HAND_WRITTEN_HEAD
            + "[anchor](src/lib.rs#L10) [heading](../../docs/ARCH.md#layers)\n"
            + "[query](src/lib.rs?plain=1#L3) [self](#purpose)\n"
            + "[web](https://example.com/missing.md) [mail](mailto:a@example.com)\n"
            + "[proto](//example.com/x.md) [ftp](ftp://example.com/x.md#frag)\n"
            + "\n"
        )
        self.write_map(head=head)
        self.assertEqual(cm.check_crate(self.crate), [])

        # A missing file is still caught once the anchor is stripped.
        self.write_map(head=head + "[dead](src/gone.rs#section)\n\n")
        problems = cm.check_crate(self.crate)
        self.assertEqual(len(problems), 1, problems)
        self.assertIn("src/gone.rs#section", problems[0])

    def test_links_in_code_are_ignored(self) -> None:
        head = (
            HAND_WRITTEN_HEAD
            + "Inline `[x](src/nope.rs)` is an example.\n\n"
            + "```md\n[y](src/nope2.rs)\n```\n\n"
        )
        self.write_map(head=head)
        self.assertEqual(cm.check_crate(self.crate), [])

    def test_inventory_lists_only_actual_files(self) -> None:
        (self.crate / "src" / "dir.rs").mkdir()
        (self.crate / "src" / "dir.rs" / "inner.rs").write_text("pub fn inner() {}\n")
        rels = [rel for rel, _ in cm.module_inventory(self.crate)]
        self.assertEqual(
            rels, ["src/dir.rs/inner.rs", "src/engine.rs", "src/lib.rs"]
        )


if __name__ == "__main__":
    unittest.main()
