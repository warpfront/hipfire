"""CPU tests for heldout.py (spec §13.5, §13.7). No network.
Run: python3 -m unittest discover -s scripts/jev_eval -p 'test_*.py'"""
import json
import tempfile
import unittest
from pathlib import Path

import heldout as h


class StateHash(unittest.TestCase):
    def test_normalisation(self):
        self.assertEqual(h.state_hash(" hi \n"), h.state_hash("hi"))
        self.assertEqual(h.state_hash({"b": 1, "a": "é"}), h.state_hash({"a": "é", "b": 1}))
        self.assertNotEqual(h.state_hash({"a": 1}), h.state_hash('{"a": 1}'))
        self.assertNotEqual(h.state_hash("Hi"), h.state_hash("hi"))


class Keys(unittest.TestCase):
    def test_keys_follow_the_decide_answer_order(self):
        self.assertEqual(h.keys_for({"type": "choice", "criteria": {"b": "x", "a": "y"}}), ["b", "a"])
        self.assertEqual(h.keys_for({"type": "score", "criteria": ["lo", "mid", "hi"]}), ["0", "1", "2"])
        self.assertEqual(h.keys_for({"type": "noul", "instructions": "i"}), ["true", "false"])


class Select(unittest.TestCase):
    def test_drops_reported_and_repeats_and_caps(self):
        cands = [{"state": s} for s in ["a", "b", "a", "c", "d", "e"]]
        kept, excluded, repeats = h.select(cands, {h.state_hash("c")}, cap=3)
        self.assertEqual([c["state"] for c in kept], ["a", "b", "d"])
        self.assertEqual((excluded, repeats), (1, 1))

    def test_object_states_match_regardless_of_key_order(self):
        kept, excluded, _ = h.select([{"state": {"x": 1, "y": 2}}], {h.state_hash({"y": 2, "x": 1})})
        self.assertEqual((kept, excluded), ([], 1))


class Redirect(unittest.TestCase):
    def test_task_loader_is_redirected_then_restored(self):
        calls = []

        class JB:
            pass

        jb = JB()

        def hf_rows(dataset, config, split, n, seed=0, page=100):
            calls.append((dataset, split, n, seed))
            return [{"x": 1}], []

        def get(url):
            calls.append(url)
            return b""

        def task(n):  # shaped like jevbench's task constructors
            jb.hf_rows("ds", "cfg", "test", 3000)
            jb._get("https://x/banking_data/test.csv")
            return [{"state": "s", "label": "l"}], {"type": "noul"}

        jb.hf_rows, jb._get, jb.TASKS = hf_rows, get, {"t": (task, "", "")}
        ex = h.jevbench_candidates(jb, "t", "validation")
        self.assertEqual(ex, [{"state": "s", "label": "l"}])
        self.assertEqual(calls, [("ds", "validation", h.DRAW, h.HELDOUT_SEED),
                                 "https://x/banking_data/train.csv"])
        self.assertIs(jb.hf_rows, hf_rows)
        self.assertIs(jb._get, get)


class Verify(unittest.TestCase):
    def test_catches_an_overlap_and_a_changed_examples_file(self):
        with tempfile.TemporaryDirectory() as d:
            out, work = Path(d) / "out", Path(d) / "work"
            (work / "examples").mkdir(parents=True)
            out.mkdir()
            ex = work / "examples/src.jsonl"
            ex.write_text(json.dumps({"state": "fresh"}) + "\n")
            (out / "manifest.json").write_text(json.dumps({"src": {"kept": 1, "sha256": h.sha256_file(ex)}}))
            hashes = out / "reported_state_sha256.txt"
            hashes.write_text(h.state_hash("old") + "\n")
            h.verify_dirs(out, work)
            hashes.write_text(h.state_hash("fresh") + "\n")
            with self.assertRaises(AssertionError):
                h.verify_dirs(out, work)
            hashes.write_text(h.state_hash("old") + "\n")
            ex.write_text(json.dumps({"state": "changed"}) + "\n")
            with self.assertRaises(AssertionError):
                h.verify_dirs(out, work)


class AnswerRow(unittest.TestCase):
    def test_noul_and_choice_rows(self):
        n = h.answer_row("sms-spam", "noul", ["true", "false"], 1, {"type": "noul", "noul": 0.25})
        self.assertEqual((n["probs"], n["gold"]), ([0.25, 0.75], 1))
        c = h.answer_row("ag-news", "choice", ["b", "a"], 0,
                         {"type": "choice", "probabilities": {"a": 0.4, "b": 0.6}, "confidence": 0.2})
        self.assertEqual(c["probs"], [0.6, 0.4])
        self.assertNotIn("state", c)


if __name__ == "__main__":
    unittest.main()
