#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU-only regression tests for scripts/check-crate-maps.py (committed form of
map.md, local-link validation, and the uncommitted map.generated.md), run
against throwaway crates in a temp dir."""

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


class CrateMapTests(unittest.TestCase):
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
        for name in ("GLOSSARY.md", "ARCHITECTURE.md"):
            (self.root / "docs" / name).write_text("# skeleton link target\n")

        for name, value in (("ROOT", self.root), ("CRATES", self.crates)):
            patcher = unittest.mock.patch.object(cm, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def write_map(self, head: str = HAND_WRITTEN_HEAD, tail: str = HAND_WRITTEN_TAIL,
                  block: str | None = None) -> None:
        block = cm.STUB if block is None else block
        (self.crate / "map.md").write_text(head + block + "\n" + tail)

    def run_check(self) -> tuple[int, str]:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = cm.main(["--check"])
        return code, out.getvalue()

    def test_committed_form_with_valid_handwritten_links_passes(self) -> None:
        self.write_map(head=HAND_WRITTEN_HEAD + "See [facade](src/lib.rs) and "
                       "[arch](../../docs/ARCH.md).\n\n")
        self.assertEqual(cm.check_crate(self.crate), [])
        code, _ = self.run_check()
        self.assertEqual(code, 0)

    def test_check_does_not_need_or_read_the_generated_file(self) -> None:
        self.write_map()
        self.assertFalse((self.crate / cm.GENERATED_NAME).exists())
        self.assertEqual(cm.check_crate(self.crate), [])
        (self.crate / cm.GENERATED_NAME).write_text("stale, uncommitted, never compared\n")
        self.assertEqual(cm.check_crate(self.crate), [])

    def test_committed_generated_block_fails_and_names_the_fix(self) -> None:
        old_block = (
            cm.BEGIN + "\n\n### Modules and exported symbols\n\n- [`src/lib.rs`](src/lib.rs)\n"
            "  - `facade`\n\n" + cm.END
        )
        self.write_map(block=old_block)
        problems = cm.check_crate(self.crate)
        self.assertEqual(len(problems), 1, problems)
        self.assertIn("cargo xtask regen", problems[0])
        code, out = self.run_check()
        self.assertEqual(code, 1)
        self.assertIn("demo", out)

    def test_handedit_between_markers_fails(self) -> None:
        self.write_map(block=cm.STUB.replace("generated, not committed", "hand-edited"))
        self.assertTrue(cm.check_crate(self.crate))

    def test_missing_markers_fail(self) -> None:
        (self.crate / "map.md").write_text(HAND_WRITTEN_HEAD)
        problems = cm.check_crate(self.crate)
        self.assertEqual(len(problems), 1, problems)
        self.assertIn("markers", problems[0])

    def test_pointer_to_the_uncommitted_file_is_not_a_link(self) -> None:
        # The stub names map.generated.md in code spans only; the file is absent in a
        # clean checkout and must not count as a broken link.
        self.write_map()
        self.assertFalse((self.crate / cm.GENERATED_NAME).exists())
        self.assertEqual(cm.check_links(self.crate, (self.crate / "map.md").read_text()), [])

    def test_broken_handwritten_link_fails(self) -> None:
        # Only prose outside the markers carries the dead link (a `llama.rs` that
        # does not exist).
        self.write_map(head=HAND_WRITTEN_HEAD + "Backend: [llama](src/llama.rs).\n\n")
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

    def test_generate_writes_the_uncommitted_file_and_restores_the_pointer(self) -> None:
        old_block = cm.BEGIN + "\n\nold generated content\n\n" + cm.END
        self.write_map(head=HAND_WRITTEN_HEAD + "Prose stays.\n\n", block=old_block)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(cm.generate_crate(self.crate))
        text = (self.crate / "map.md").read_text()
        self.assertEqual(
            text, HAND_WRITTEN_HEAD + "Prose stays.\n\n" + cm.STUB + "\n" + HAND_WRITTEN_TAIL
        )
        generated = (self.crate / cm.GENERATED_NAME).read_text()
        self.assertIn("### Modules and exported symbols", generated)
        self.assertIn("- [`src/engine.rs`](src/engine.rs)", generated)
        self.assertIn("  - `Engine`", generated)
        self.assertIn("- `serde`", generated)
        self.assertEqual(cm.check_crate(self.crate), [])

    def test_generate_is_idempotent_and_refreshes_a_stale_file(self) -> None:
        self.write_map()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            cm.generate_crate(self.crate)
            first = (self.crate / cm.GENERATED_NAME).read_text()
            self.assertIn("wrote", out.getvalue())
            out.seek(0)
            out.truncate()
            cm.generate_crate(self.crate)
            self.assertEqual(out.getvalue().strip(), "demo: already up to date")
            (self.crate / "src" / "extra.rs").write_text("pub fn extra() {}\n")
            cm.generate_crate(self.crate)
        second = (self.crate / cm.GENERATED_NAME).read_text()
        self.assertNotEqual(first, second)
        self.assertIn("`src/extra.rs`", second)

    def test_new_crate_gets_a_skeleton_with_the_pointer(self) -> None:
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(cm.generate_crate(self.crate))
        text = (self.crate / "map.md").read_text()
        self.assertIn(cm.STUB, text)
        self.assertEqual(cm.check_crate(self.crate), [])
        self.assertTrue((self.crate / cm.GENERATED_NAME).is_file())


if __name__ == "__main__":
    unittest.main()
