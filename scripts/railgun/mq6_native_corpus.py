#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Pack a certified, pinned native MQ6 image for explicit railgun probes.

Run railgun-cert on the committed HIP bundle first. This creates a small native
fixture corpus, with no hipcc recipe or default-program admission. It does not
certify objects itself and is not a replacement for the default corpus builder.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import socket
import subprocess

BUNDLE = "bc34ee389bd990386bdfc93d94b66752affcd3fbd33bec212a6b4edcaa51a963"
DEVICE = "b0a702507aa5420261ac90742e3665403e47636c24e034918507997bb14c53dc"
MODULE = "qwen4_lane_rows_pm"
SYMBOLS = {"gemv_mq6g256v2_xbatch_pm", "gemv_mq6g256v2_x4_xbatch_pm"}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    repo, out = args.repo.resolve(), args.out.absolute()
    bundle = repo / "kernels/qwen4_lane_rows_pm_gfx1151.hxaco"
    raw = args.receipt.read_bytes()
    receipt = json.loads(raw)
    if not (receipt.get("schema_version") == 9 and receipt.get("arch") == "gfx1151"
            and receipt.get("object_sha256") == sha(bundle.read_bytes()) == BUNDLE
            and receipt.get("device_elf_sha256") == DEVICE):
        parser.error("schema-9 receipt must bind the pinned gfx1151 MQ6 bundle/device ELF")
    kernels = receipt["railgun"]["kernels"]
    selected = [k for k in kernels if k["symbol"] in SYMBOLS]
    if not (len(selected) == 2 and {k["symbol"] for k in selected} == SYMBOLS
            and all(k["status"] == "proven" and k["differential"]["status"] == "pass"
                    for k in selected)):
        parser.error("both MQ6 kernels require actual Proven/G8-pass receipt records")
    record = dict(arch="gfx1151", artifact_sha256=BUNDLE, module=MODULE, origin="pm_native",
        symbols=[k["symbol"] for k in kernels], source_sha256="", extra_flags="", flags=[],
        scheduler_profile="", packaging_key="", jit_cache_key="", argv=[],
        object=f"native/gfx1151/{MODULE}.hxaco",
        receipt=f"receipts/gfx1151/{BUNDLE}.receipt.json", receipt_sha256=sha(raw),
        kernels=[dict(symbol=k["symbol"], status=k["status"],
                      differential=k["differential"]["status"]) for k in kernels])
    tool_id = "pm-native fixture; no hipcc compilation"
    host = dict(hostname=socket.gethostname(), hipcc="not invoked: prebuilt native fixture",
        rocm_env_root=None, clang_sha256=sha(b""), header_set_sha256=sha(b""), build_dir=str(repo))
    host["key"] = sha(json.dumps(host, sort_keys=True).encode())
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    index = dict(schema="railgun-jit-corpus", version=2, source_commit=head, build_host=host,
        toolchain=dict(id=tool_id, pin=sha(tool_id.encode()), version_sha256=sha((tool_id+"\n").encode())),
        records=[record], rejected=[], programs=[])
    out.mkdir(parents=True, exist_ok=False)
    for relative, original in [(record["object"], bundle), (record["receipt"], args.receipt)]:
        path = out / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, path)
    (out / "index.json").write_text(json.dumps(index, indent=2)+"\n")
    print(out)


if __name__ == "__main__":
    main()
