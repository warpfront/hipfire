#!/usr/bin/env python3
"""Build the prompt/completion corpus and held-out KLD references for Qwen3.8 run 6."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

SCHEMA = "hipfire.qat.capture.v1"
SEED = 20260921
SEQ_LEN = 8192
REF_CTX = 2048
REF_CHUNKS = 24
REF_TOP_K = 256
MAX_PROMPT_TOKENS = 2048
MAX_NEW_TOKENS = 6000
EOS_IDS = (248046, 248044)
TULU_AGENT_SOURCES = {
    "ai2-adapt-dev/tulu_v3.9_table_gpt_5k",
    "ai2-adapt-dev/tulu_v3.9_sciriff_10k",
}
TULU_GENERAL_SOURCES = {
    "ai2-adapt-dev/tulu_v3.9_wildchat_100k",
    "ai2-adapt-dev/no_robots_converted",
    "ai2-adapt-dev/oasst1_converted",
}


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path, chunk: int = 32 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while data := handle.read(chunk):
            digest.update(data)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def prefix_before_assistant(messages: Iterable[dict[str, Any]], legacy: bool = False) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    mapping = {"human": "user", "gpt": "assistant"} if legacy else {}
    for item in messages:
        role = mapping.get(str(item.get("from", item.get("role", ""))), str(item.get("role", "")))
        content = str(item.get("value", item.get("content", ""))).strip()
        if role == "assistant":
            break
        if role not in {"system", "user"} or not content:
            continue
        if result and role == result[-1]["role"]:
            result[-1]["content"] += "\n\n" + content
        else:
            result.append({"role": role, "content": content})
    if not result or result[-1]["role"] != "user":
        return []
    return result


def prompt_ids(tokenizer: Any, row: dict[str, Any]) -> list[int]:
    return [
        int(token)
        for token in tokenizer.apply_chat_template(
            row["messages"],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=bool(row["enable_thinking"]),
        )
    ]


def dataset_sha(repo: str) -> str | None:
    try:
        from huggingface_hub import HfApi

        return HfApi().dataset_info(repo).sha
    except Exception:
        return None


def select(args: argparse.Namespace) -> None:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    started = time.monotonic()
    args.out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.source, trust_remote_code=False, use_fast=True)
    selected: dict[str, list[dict[str, Any]]] = {
        "agent_hermes": [],
        "agent_tulu": [],
        "code_feedback": [],
        "code_mbpp_commitpack": [],
        "reason_opus": [],
        "reason_qwen": [],
        "chat_ultra": [],
        "chat_tulu": [],
    }
    seen: set[str] = set()

    def offer(
        slice_name: str,
        dataset_id: str,
        dataset_split: str,
        row_index: int,
        messages: list[dict[str, str]],
        *,
        category: str,
        enable_thinking: bool,
        dataset_config: str | None = None,
        subset: str | None = None,
        limit: int = 40,
    ) -> None:
        if len(selected[slice_name]) >= limit or not messages:
            return
        row = {
            "slice": slice_name,
            "category": category,
            "dataset_id": dataset_id,
            "dataset_config": dataset_config,
            "dataset_split": dataset_split,
            "dataset_row_index": int(row_index),
            "dataset_subset": subset,
            "messages": messages,
            "enable_thinking": enable_thinking,
        }
        ids = prompt_ids(tokenizer, row)
        digest = sha_bytes(np.asarray(ids, dtype="<u4").tobytes())
        if not ids or len(ids) > MAX_PROMPT_TOKENS or digest in seen:
            return
        seen.add(digest)
        row["prompt_token_count"] = len(ids)
        row["prompt_token_sha256"] = digest
        selected[slice_name].append(row)

    hermes = load_dataset("NousResearch/hermes-function-calling-v1", split="train", streaming=True)
    for index, row in enumerate(hermes):
        offer(
            "agent_hermes",
            "NousResearch/hermes-function-calling-v1",
            "train",
            index,
            prefix_before_assistant(row["conversations"], legacy=True),
            category="agentic",
            enable_thinking=False,
            subset=str(row.get("category", "")),
        )
        if len(selected["agent_hermes"]) == 40:
            break

    tulu = load_dataset("allenai/tulu-3-sft-mixture", split="train", streaming=True)
    for index, row in enumerate(tulu):
        source = str(row["source"])
        messages = prefix_before_assistant(row["messages"])
        if source in TULU_AGENT_SOURCES:
            offer(
                "agent_tulu",
                "allenai/tulu-3-sft-mixture",
                "train",
                index,
                messages,
                category="agentic",
                enable_thinking=False,
                subset=source,
            )
        if source in TULU_GENERAL_SOURCES:
            offer(
                "chat_tulu",
                "allenai/tulu-3-sft-mixture",
                "train",
                index,
                messages,
                category="chat",
                enable_thinking=False,
                subset=source,
            )
        if len(selected["agent_tulu"]) == 40 and len(selected["chat_tulu"]) == 40:
            break

    feedback = load_dataset("m-a-p/CodeFeedback-Filtered-Instruction", split="train", streaming=True)
    for index, row in enumerate(feedback):
        offer(
            "code_feedback",
            "m-a-p/CodeFeedback-Filtered-Instruction",
            "train",
            index,
            [{"role": "user", "content": str(row["query"]).strip()}],
            category="code",
            enable_thinking=False,
            subset=str(row.get("lang", "")),
        )
        if len(selected["code_feedback"]) == 40:
            break

    mbpp_plan = (("train", 8), ("validation", 6), ("prompt", 6))
    for split, count in mbpp_plan:
        dataset = load_dataset("google-research-datasets/mbpp", "full", split=split, streaming=True)
        before = len(selected["code_mbpp_commitpack"])
        for index, row in enumerate(dataset):
            offer(
                "code_mbpp_commitpack",
                "google-research-datasets/mbpp",
                split,
                index,
                [{"role": "user", "content": str(row["text"]).strip()}],
                category="code",
                enable_thinking=False,
                dataset_config="full",
                subset="mbpp",
            )
            if len(selected["code_mbpp_commitpack"]) == before + count:
                break

    commit_url = "hf://datasets/bigcode/commitpackft/data/python/data.jsonl"
    commitpack = load_dataset("json", data_files=commit_url, split="train", streaming=True)
    for index, row in enumerate(commitpack):
        old = str(row.get("old_contents", ""))
        instruction = str(row.get("subject", row.get("message", ""))).strip()
        if not old.strip() or not instruction or len(old) > 12_000:
            continue
        content = (
            f"Edit `{row.get('old_file', 'the file')}` to make this change:\n{instruction}\n\n"
            f"Current file:\n```{row.get('lang', '')}\n{old}\n```\nReturn the complete updated file."
        )
        offer(
            "code_mbpp_commitpack",
            "bigcode/commitpackft",
            "train",
            index,
            [{"role": "user", "content": content}],
            category="code",
            enable_thinking=False,
            dataset_config="python",
            subset=str(row.get("lang", "Python")),
        )
        if len(selected["code_mbpp_commitpack"]) == 40:
            break

    for slice_name, repo in (
        ("reason_opus", "angrygiraffe/claude-opus-4.6-4.7-reasoning-8.7k"),
        ("reason_qwen", "r0b0tlab/qwen3.8-max-distillation-50k"),
    ):
        dataset = load_dataset(repo, split="train", streaming=True)
        for index, row in enumerate(dataset):
            offer(
                slice_name,
                repo,
                "train",
                index,
                prefix_before_assistant(row["messages"]),
                category="reasoning",
                enable_thinking=True,
                subset=str(row.get("category", row.get("domain", ""))),
            )
            if len(selected[slice_name]) == 40:
                break

    ultra = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
    for index, row in enumerate(ultra):
        offer(
            "chat_ultra",
            "HuggingFaceH4/ultrachat_200k",
            "train_sft",
            index,
            prefix_before_assistant(row["messages"]),
            category="chat",
            enable_thinking=False,
        )
        if len(selected["chat_ultra"]) == 40:
            break

    bad = {name: len(rows) for name, rows in selected.items() if len(rows) != 40}
    if bad:
        raise RuntimeError(f"incomplete prompt slices: {bad}")

    rows: list[dict[str, Any]] = []
    for slice_name, slice_rows in selected.items():
        heldout = {
            id(row)
            for row in sorted(
                slice_rows,
                key=lambda value: hashlib.sha256(
                    f"{SEED}:{value['dataset_id']}:{value['dataset_split']}:{value['dataset_row_index']}".encode()
                ).digest(),
            )[:6]
        }
        for row in slice_rows:
            row["split"] = "heldout" if id(row) in heldout else "train"
            rows.append(row)
    rows.sort(key=lambda row: (row["slice"], row["dataset_id"], row["dataset_split"], row["dataset_row_index"]))
    for index, row in enumerate(rows):
        row["prompt_index"] = index
    if len(rows) != 320:
        raise AssertionError(len(rows))
    counts = Counter((row["slice"], row["split"]) for row in rows)
    if any(counts[(name, "train")] != 34 or counts[(name, "heldout")] != 6 for name in selected):
        raise AssertionError(counts)

    prompt_path = args.out / "prompts.jsonl"
    atomic_jsonl(prompt_path, rows)
    repos = sorted({row["dataset_id"] for row in rows})
    receipt = {
        "schema": "hipfire.qat.run6.prompts.v1",
        "seed": SEED,
        "total_prompts": len(rows),
        "heldout_fraction_per_slice": 0.15,
        "slice_counts": {
            name: {split: counts[(name, split)] for split in ("train", "heldout")}
            for name in selected
        },
        "datasets": {repo: {"revision": dataset_sha(repo)} for repo in repos},
        "tulu_agent_source_filter": sorted(TULU_AGENT_SOURCES),
        "tulu_general_source_filter": sorted(TULU_GENERAL_SOURCES),
        "prompt_file": str(prompt_path),
        "prompt_file_sha256": sha_file(prompt_path),
        "max_prompt_tokens": MAX_PROMPT_TOKENS,
        "tokenizer_json_sha256": sha_file(args.source / "tokenizer.json"),
        "wall_clock_seconds": time.monotonic() - started,
    }
    atomic_json(args.out / "prompt_manifest.json", receipt)
    print(json.dumps(receipt, indent=2), flush=True)


def load_teacher(source: Path) -> tuple[Any, Any, Any]:
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from lora_qat import load_text_model
    from transformers import AutoTokenizer

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=False, use_fast=True)
    tokenizer.padding_side = "left"
    model = load_text_model(source, device)
    model.eval()
    model.config.use_cache = True
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return tokenizer, model, device


def generate_completions(args: argparse.Namespace, tokenizer: Any, model: Any, device: Any) -> list[dict[str, Any]]:
    import torch

    prompts = read_jsonl(args.out / "prompts.jsonl")
    if len(prompts) != 320:
        raise RuntimeError("select must produce exactly 320 prompts")
    completion_path = args.out / "teacher_completions.jsonl"
    existing = {int(row["prompt_index"]): row for row in read_jsonl(completion_path)}
    pending: list[tuple[dict[str, Any], list[int]]] = []
    for row in prompts:
        if int(row["prompt_index"]) not in existing:
            ids = prompt_ids(tokenizer, row)
            if len(ids) > MAX_PROMPT_TOKENS:
                raise AssertionError((row["prompt_index"], len(ids)))
            pending.append((row, ids))
    pending.sort(key=lambda item: (item[0]["enable_thinking"], len(item[1])))
    eos = list(EOS_IDS)
    torch.manual_seed(SEED)
    for start in range(0, len(pending), args.batch_size):
        batch = pending[start : start + args.batch_size]
        width = max(len(ids) for _, ids in batch)
        input_ids = torch.full((len(batch), width), tokenizer.pad_token_id, dtype=torch.long, device=device)
        attention = torch.zeros_like(input_ids)
        for index, (_, ids) in enumerate(batch):
            input_ids[index, width - len(ids) :] = torch.tensor(ids, dtype=torch.long, device=device)
            attention[index, width - len(ids) :] = 1
        with torch.inference_mode():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention,
                do_sample=False,
                use_cache=True,
                max_new_tokens=MAX_NEW_TOKENS,
                eos_token_id=eos,
                pad_token_id=tokenizer.pad_token_id,
            )
        for index, (prompt, ids) in enumerate(batch):
            output = [int(value) for value in generated[index, width:].detach().cpu().tolist()]
            while output and output[-1] == tokenizer.pad_token_id:
                output.pop()
            stop = next((position + 1 for position, value in enumerate(output) if value in eos), len(output))
            output = output[:stop]
            if not output or output[-1] not in eos:
                output.append(int(tokenizer.eos_token_id))
            if len(ids) + len(output) > SEQ_LEN:
                raise AssertionError((prompt["prompt_index"], len(ids), len(output)))
            existing[int(prompt["prompt_index"])] = {
                "prompt_index": int(prompt["prompt_index"]),
                "prompt_token_ids": ids,
                "completion_token_ids": output,
                "completion_text": tokenizer.decode(output, skip_special_tokens=False),
                "completion_token_count": len(output),
                "completion_token_sha256": sha_bytes(np.asarray(output, dtype="<u4").tobytes()),
                "teacher_dtype": "bfloat16",
                "generation": {
                    "do_sample": False,
                    "max_new_tokens": MAX_NEW_TOKENS,
                    "eos_token_ids": eos,
                    "enable_thinking": bool(prompt["enable_thinking"]),
                },
            }
        atomic_jsonl(completion_path, [existing[index] for index in sorted(existing)])
        print(f"teacher completions {len(existing):03d}/320", flush=True)
        del generated, input_ids, attention
    rows = [existing[index] for index in range(320)]
    receipt = {
        "schema": "hipfire.qat.run6.completions.v1",
        "count": len(rows),
        "teacher": str(args.source),
        "teacher_dtype": "bfloat16",
        "stored_teacher_logits": False,
        "total_completion_tokens": sum(row["completion_token_count"] for row in rows),
        "min_completion_tokens": min(row["completion_token_count"] for row in rows),
        "max_completion_tokens": max(row["completion_token_count"] for row in rows),
        "file": str(completion_path),
        "sha256": sha_file(completion_path),
    }
    atomic_json(args.out / "completion_receipt.json", receipt)
    print(json.dumps(receipt, indent=2), flush=True)
    return rows


def pack_corpus(args: argparse.Namespace, tokenizer: Any, completions: list[dict[str, Any]]) -> dict[str, Any]:
    prompts = {int(row["prompt_index"]): row for row in read_jsonl(args.out / "prompts.jsonl")}
    examples = []
    for completion in completions:
        prompt = prompts[int(completion["prompt_index"])]
        pids = [int(value) for value in completion["prompt_token_ids"]]
        cids = [int(value) for value in completion["completion_token_ids"]]
        examples.append(
            {
                "prompt": prompt,
                "tokens": pids + cids,
                "mask": [0] * len(pids) + [1] * len(cids),
            }
        )
    examples.sort(
        key=lambda item: (
            0 if item["prompt"]["split"] == "train" else 1,
            hashlib.sha256(f"{SEED}:pack:{item['prompt']['prompt_index']}".encode()).digest(),
        )
    )
    packed: list[dict[str, Any]] = []
    pad = int(tokenizer.pad_token_id)
    for split in ("train", "heldout"):
        current_tokens: list[int] = []
        current_mask: list[int] = []
        segments: list[dict[str, Any]] = []

        def flush() -> None:
            nonlocal current_tokens, current_mask, segments
            if not current_tokens:
                return
            real = len(current_tokens)
            current_tokens.extend([pad] * (SEQ_LEN - real))
            current_mask.extend([0] * (SEQ_LEN - real))
            packed.append(
                {
                    "split": split,
                    "tokens": current_tokens,
                    "loss_mask": current_mask,
                    "valid_tokens": real,
                    "segments": segments,
                }
            )
            current_tokens, current_mask, segments = [], [], []

        for example in (item for item in examples if item["prompt"]["split"] == split):
            if len(current_tokens) + len(example["tokens"]) > SEQ_LEN:
                flush()
            offset = len(current_tokens)
            current_tokens.extend(example["tokens"])
            current_mask.extend(example["mask"])
            prompt = example["prompt"]
            segments.append(
                {
                    "prompt_index": prompt["prompt_index"],
                    "category": prompt["category"],
                    "slice": prompt["slice"],
                    "dataset_id": prompt["dataset_id"],
                    "dataset_split": prompt["dataset_split"],
                    "dataset_row_index": prompt["dataset_row_index"],
                    "token_offset": offset,
                    "prompt_tokens": prompt["prompt_token_count"],
                    "completion_tokens": len(example["tokens"]) - prompt["prompt_token_count"],
                }
            )
            if len(current_tokens) == SEQ_LEN:
                flush()
        flush()

    train_count = sum(row["split"] == "train" for row in packed)
    heldout_count = len(packed) - train_count
    if packed[:train_count] and any(row["split"] != "train" for row in packed[:train_count]):
        raise AssertionError("train sequences must precede heldout")
    tokens = np.asarray([row.pop("tokens") for row in packed], dtype="<u4")
    masks = np.asarray([row.pop("loss_mask") for row in packed], dtype=np.uint8)
    attention = np.zeros_like(masks)
    for index, row in enumerate(packed):
        attention[index, : row["valid_tokens"]] = 1
    positions = np.broadcast_to(np.arange(SEQ_LEN, dtype="<u4"), tokens.shape).copy()
    corpus_dir = args.out / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    token_path = corpus_dir / "tokens.u32"
    loss_path = corpus_dir / "loss_mask.u8"
    attention_path = corpus_dir / "attention_mask.u8"
    position_path = corpus_dir / "positions.u32"
    token_path.write_bytes(tokens.tobytes())
    loss_path.write_bytes(masks.tobytes())
    attention_path.write_bytes(attention.tobytes())
    position_path.write_bytes(positions.tobytes())

    sequence_rows = []
    for sequence_id, row in enumerate(packed):
        token_sha = sha_bytes(tokens[sequence_id].tobytes())
        prompt_hashes = [prompts[int(segment["prompt_index"])]["prompt_token_sha256"] for segment in row["segments"]]
        document_sha = sha_bytes(canonical(prompt_hashes))
        sequence_rows.append(
            {
                "sequence_id": sequence_id,
                "source": "run6-agentic-prompt-mixture",
                "split": row["split"],
                "dataset_document_id": f"packed:run6:{document_sha[:16]}",
                "document_sha256": document_sha,
                "document_bytes": 0,
                "document_token_count": row["valid_tokens"],
                "token_offset": 0,
                "token_count": SEQ_LEN,
                "valid_token_count": row["valid_tokens"],
                "completion_token_count": int(masks[sequence_id].sum()),
                "token_sha256": token_sha,
                "state_reset_id": f"zero:{document_sha}:{token_sha}",
                "token_file_offset_bytes": sequence_id * SEQ_LEN * 4,
                "prompt_segments": row["segments"],
            }
        )

    old_manifest = json.loads(args.source_manifest.read_text())
    source_identity = old_manifest["source"]
    prompt_manifest = json.loads((args.out / "prompt_manifest.json").read_text())
    completion_receipt = json.loads((args.out / "completion_receipt.json").read_text())
    prompt_mix = Counter((row["category"], row["split"]) for row in prompts.values())

    def meta(path: Path, dtype: str, semantics: str) -> dict[str, Any]:
        return {
            "file": str(path.relative_to(args.out)),
            "dtype": dtype,
            "axes": ["sequence", "token"],
            "shape": list(tokens.shape),
            "sha256": sha_file(path),
            "semantics": semantics,
        }

    manifest = {
        "schema_version": SCHEMA,
        "created_unix": time.time(),
        "source": source_identity,
        "corpus": {
            "seed": SEED,
            "sequence_length": SEQ_LEN,
            "train_sequences": train_count,
            "heldout_sequences": heldout_count,
            "total_sequences": len(packed),
            "total_tokens": int(tokens.size),
            "valid_tokens": int(attention.sum()),
            "completion_loss_tokens": int(masks.sum()),
            "prompt_count": len(prompts),
            "mixture": {
                split: {category: prompt_mix[(category, split)] for category in ("agentic", "code", "reasoning", "chat")}
                for split in ("train", "heldout")
            },
            "packing": {
                "enabled": True,
                "strategy": "deterministic whole-example greedy packing; no prompt/completion crosses an 8192-token boundary",
                "padding_token_id": pad,
                "prompt_loss_masked": True,
            },
            "datasets": prompt_manifest["datasets"],
            "prompt_manifest": {
                "file": "prompt_manifest.json",
                "sha256": sha_file(args.out / "prompt_manifest.json"),
                "prompts_file": "prompts.jsonl",
                "prompts_sha256": sha_file(args.out / "prompts.jsonl"),
                "dataset_id_and_row_index_recorded": True,
            },
            "teacher_completions": completion_receipt,
            "tokenizer": {
                "path": str(args.source.resolve()),
                "tokenizer_json_sha256": sha_file(args.source / "tokenizer.json"),
                "class": tokenizer.__class__.__name__,
                "chat_template": "Qwen source tokenizer; reasoning slices enable_thinking=True, all others False",
            },
            "token_stream": meta(token_path, "uint32-le", "Qwen chat-template prompt plus BF16-teacher completion"),
            "attention_mask": meta(attention_path, "uint8", "1 for real packed tokens, 0 for terminal padding"),
            "positions": meta(position_path, "uint32-le", f"0..{SEQ_LEN - 1} independently per sequence"),
            "loss_mask": meta(loss_path, "uint8", "1 only on BF16-teacher completion tokens; prompts and padding are zero"),
            "state": {
                "reset": "zero at each packed sequence",
                "carry": "within an 8192-token packed sequence",
                "kv": "empty",
                "convolution": "zero",
                "recurrent": "zero",
            },
            "sequences": sequence_rows,
        },
        "capture": {"status": "corpus_ready", "blocks": {}, "tensor_shards": [], "validation": {}},
    }
    atomic_json(args.out / "manifest.json", manifest)
    receipt = {
        "manifest": str(args.out / "manifest.json"),
        "manifest_sha256": sha_file(args.out / "manifest.json"),
        "train_sequences": train_count,
        "heldout_sequences": heldout_count,
        "sequence_length": SEQ_LEN,
        "valid_tokens": int(attention.sum()),
        "completion_loss_tokens": int(masks.sum()),
    }
    atomic_json(args.out / "corpus_receipt.json", receipt)
    print(json.dumps(receipt, indent=2), flush=True)
    return manifest


def reference_windows(prompts: dict[int, dict[str, Any]], completions: list[dict[str, Any]], category: str) -> tuple[np.ndarray, list[int], list[int]]:
    by_index = {int(row["prompt_index"]): row for row in completions}
    prompt_indexes = sorted(
        index for index, row in prompts.items() if row["split"] == "heldout" and row["category"] == category
    )
    stream: list[int] = []
    for index in prompt_indexes:
        row = by_index[index]
        stream.extend(int(value) for value in row["prompt_token_ids"])
        stream.extend(int(value) for value in row["completion_token_ids"])
    if len(stream) < REF_CTX:
        raise RuntimeError(f"heldout {category} stream has only {len(stream)} tokens")
    span = len(stream) - REF_CTX
    offsets = [int(round(index * span / (REF_CHUNKS - 1))) for index in range(REF_CHUNKS)]
    windows = np.asarray([stream[offset : offset + REF_CTX] for offset in offsets], dtype="<u4")
    return windows, offsets, prompt_indexes


def build_reference(path: Path, tokens: np.ndarray, model: Any, device: Any, tile_vocab: int) -> None:
    import torch
    import torch.nn.functional as functional

    vocab = int(model.config.vocab_size)
    header = struct.pack("<8sIIIIHHI", b"HFKLDR\0\0", 1, REF_CTX, vocab, REF_CHUNKS, REF_TOP_K, 0, 0)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(header)
        handle.write(tokens.tobytes())
        for chunk_index, chunk in enumerate(tokens):
            input_ids = torch.from_numpy(chunk.astype(np.int64)).view(1, REF_CTX).to(device)
            with torch.inference_mode():
                hidden = model.model(input_ids=input_ids, use_cache=False, return_dict=True).last_hidden_state[0, REF_CTX // 2 : -1]
                lse = torch.full((hidden.shape[0],), -torch.inf, dtype=torch.float32, device=device)
                best_values = None
                best_indices = None
                for start in range(0, vocab, tile_vocab):
                    stop = min(start + tile_vocab, vocab)
                    logits = functional.linear(hidden, model.lm_head.weight[start:stop]).float()
                    lse = torch.logaddexp(lse, torch.logsumexp(logits, dim=-1))
                    count = min(REF_TOP_K, stop - start)
                    values, indices = torch.topk(logits, count, dim=-1, largest=True, sorted=True)
                    indices = indices + start
                    if best_values is None:
                        best_values, best_indices = values, indices
                    else:
                        merged_values = torch.cat((best_values, values), dim=-1)
                        merged_indices = torch.cat((best_indices, indices), dim=-1)
                        best_values, order = torch.topk(merged_values, REF_TOP_K, dim=-1, largest=True, sorted=True)
                        best_indices = torch.gather(merged_indices, -1, order)
                if best_values is None or best_indices is None:
                    raise AssertionError("empty vocabulary")
                log_probs = best_values - lse.unsqueeze(-1)
                residual = (1.0 - torch.exp(log_probs).sum(dim=-1)).clamp_(0.0, 1.0)
            index_array = best_indices.to(torch.int64).cpu().numpy().astype("<u4", copy=False)
            log_array = log_probs.cpu().numpy().astype("<f4", copy=False)
            residual_array = residual.cpu().numpy().astype("<f4", copy=False)
            for position in range(index_array.shape[0]):
                handle.write(index_array[position].tobytes())
                handle.write(log_array[position].tobytes())
                handle.write(struct.pack("<ff", float(residual_array[position]), 0.0))
            print(f"{path.name}: chunk {chunk_index + 1:02d}/{REF_CHUNKS}", flush=True)
            del input_ids, hidden, lse, best_values, best_indices, log_probs, residual
    os.replace(temporary, path)
    expected = 32 + REF_CHUNKS * REF_CTX * 4 + REF_CHUNKS * (REF_CTX - 1 - REF_CTX // 2) * (8 + 8 * REF_TOP_K)
    if path.stat().st_size != expected:
        raise AssertionError((path.stat().st_size, expected))


def build_references(args: argparse.Namespace, model: Any, device: Any, completions: list[dict[str, Any]]) -> dict[str, Any]:
    prompts = {int(row["prompt_index"]): row for row in read_jsonl(args.out / "prompts.jsonl")}
    refs_dir = args.out / "refs"
    refs_dir.mkdir(parents=True, exist_ok=True)
    receipt: dict[str, Any] = {
        "schema": "hipfire.qat.run6.refs.v1",
        "teacher": str(args.source),
        "teacher_dtype": "bfloat16",
        "n_ctx": REF_CTX,
        "n_chunk": REF_CHUNKS,
        "top_k": REF_TOP_K,
        "stored_dense_teacher_logits": False,
        "refs": {},
    }
    for category, filename in (
        ("agentic", "ref_agent.bin"),
        ("code", "ref_code.bin"),
        ("reasoning", "ref_reason.bin"),
    ):
        tokens, offsets, prompt_indexes = reference_windows(prompts, completions, category)
        path = refs_dir / filename
        build_reference(path, tokens, model, device, args.tile_vocab)
        receipt["refs"][category] = {
            "path": str(path),
            "sha256": sha_file(path),
            "bytes": path.stat().st_size,
            "token_stream_sha256": sha_bytes(tokens.tobytes()),
            "heldout_prompt_indexes": prompt_indexes,
            "window_offsets": offsets,
        }
    atomic_json(refs_dir / "receipt.json", receipt)
    print(json.dumps(receipt, indent=2), flush=True)
    return receipt


def materialize(args: argparse.Namespace) -> None:
    tokenizer, model, device = load_teacher(args.source)
    completions = generate_completions(args, tokenizer, model, device)
    pack_corpus(args, tokenizer, completions)
    build_references(args, model, device, completions)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--source", type=Path, default=Path("/root/qat/parents/qwen3.8-27b"))
    common.add_argument("--out", type=Path, default=Path("/root/qat/l3/run6-capture"))
    choose = sub.add_parser("select", parents=[common])
    choose.set_defaults(func=select)
    make = sub.add_parser("materialize", parents=[common])
    make.add_argument("--source-manifest", type=Path, default=Path("/root/qat/l3/manifest.json"))
    make.add_argument("--batch-size", type=int, default=4)
    make.add_argument("--tile-vocab", type=int, default=8192)
    make.set_defaults(func=materialize)
    return root


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
