"""Add an excluded AR warmup to the official serving harness."""
import importlib.util
import json
import signal
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("serve_harness", root / "scripts/serve_harness.py")
harness = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = harness
spec.loader.exec_module(harness)
original_run = harness.run


def warm_run(cfg, args):
    Path(str(args.out) + ".config.json").write_text(json.dumps(cfg, indent=2))
    warm = harness.send(cfg, [{"role": "user", "content":
        "Write a detailed story about a mountain expedition."}], max_tokens=128)
    Path(str(args.out) + ".warmup.json").write_text(json.dumps(warm, indent=2))
    if warm.get("stream_error") or not warm.get("saw_done") or warm.get("gen") != 128:
        raise RuntimeError("warmup must complete exactly 128 tokens without stream errors")
    return original_run(cfg, args)


if __name__ == "__main__":
    # SystemExit preserves the official harness's atexit daemon cleanup.
    def stop(signum, _frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    harness.run = warm_run
    harness.main()
