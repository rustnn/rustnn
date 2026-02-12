#!/usr/bin/env python3
"""
Generate text from a WebNN graph by converting it to ONNX with rustnn, then running ONNX Runtime.

This demo uses the WebNN artifact as input (`.webnn` + `.weights/.manifest`) and performs
token-by-token KV-cache generation for GPT-2 style cached decode graphs.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Dict, List

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GPT-2 generation from WebNN via rustnn ONNX conversion")
    p.add_argument("--webnn", type=Path, default=Path("/tmp/gpt2_cached64.webnn"))
    p.add_argument("--onnx", type=Path, default=Path("/tmp/gpt2_cached64.from_webnn.onnx"))
    p.add_argument("--prompt", type=str, default="Hello")
    p.add_argument("--max-new-tokens", type=int, default=20)
    p.add_argument("--kv-len", type=int, default=64)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--heads", type=int, default=12)
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--tensor-limit", type=int, default=500_000_000_000)
    return p.parse_args()


def convert_webnn_to_onnx(webnn_path: Path, onnx_path: Path, tensor_limit: int) -> None:
    cmd = [
        "cargo",
        "run",
        "--features",
        "onnx-runtime",
        "--",
        str(webnn_path),
        "--tensor-limit",
        str(tensor_limit),
        "--convert",
        "onnx",
        "--convert-output",
        str(onnx_path),
    ]
    subprocess.run(cmd, check=True)


def init_kv_cache(layers: int, heads: int, kv_len: int, head_dim: int) -> Dict[str, np.ndarray]:
    kv = {}
    shape = (1, heads, kv_len, head_dim)
    for i in range(layers):
        kv[f"past_key_values_{i}_key"] = np.zeros(shape, dtype=np.float32)
        kv[f"past_key_values_{i}_value"] = np.zeros(shape, dtype=np.float32)
    return kv


def update_kv_cache(
    kv_cache: Dict[str, np.ndarray],
    outputs: Dict[str, np.ndarray],
    layers: int,
    pos: int,
) -> None:
    for i in range(layers):
        pk = outputs[f"present_{i}_key"]
        pv = outputs[f"present_{i}_value"]
        kv_cache[f"past_key_values_{i}_key"][:, :, pos : pos + 1, :] = pk[:, :, pos : pos + 1, :]
        kv_cache[f"past_key_values_{i}_value"][:, :, pos : pos + 1, :] = pv[:, :, pos : pos + 1, :]


def run_generation(args: argparse.Namespace) -> None:
    if not args.webnn.exists():
        raise FileNotFoundError(f"Missing WebNN graph: {args.webnn}")

    print(f"[INFO] Converting WebNN to ONNX: {args.webnn} -> {args.onnx}")
    convert_webnn_to_onnx(args.webnn, args.onnx, args.tensor_limit)

    print("[INFO] Loading tokenizer: distilgpt2")
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    eos_token_id = tokenizer.eos_token_id

    print("[INFO] Loading ONNX Runtime session")
    sess = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    output_names = [o.name for o in sess.get_outputs()]

    prompt_ids: List[int] = tokenizer.encode(args.prompt, add_special_tokens=False)
    if not prompt_ids:
        raise ValueError("Prompt tokenization produced no tokens")
    if len(prompt_ids) >= args.kv_len:
        raise ValueError(f"Prompt too long ({len(prompt_ids)}), must be < kv-len ({args.kv_len})")

    attention_mask = np.zeros((1, args.kv_len + 1), dtype=np.int64)
    kv_cache = init_kv_cache(args.layers, args.heads, args.kv_len, args.head_dim)

    outputs: Dict[str, np.ndarray] = {}
    for pos, token_id in enumerate(prompt_ids):
        attention_mask[0, pos] = 1
        feed = {
            "input_ids": np.array([[token_id]], dtype=np.int64),
            "position_ids": np.array([[pos]], dtype=np.int64),
            "attention_mask": attention_mask,
            **kv_cache,
        }
        raw = sess.run(output_names, feed)
        outputs = dict(zip(output_names, raw))
        update_kv_cache(kv_cache, outputs, args.layers, pos)

    logits = outputs["logits"]
    next_token = int(np.argmax(logits[0, 0, :]))
    generated = [next_token]

    current_len = len(prompt_ids)
    for _ in range(args.max_new_tokens - 1):
        if eos_token_id is not None and next_token == eos_token_id:
            break
        if current_len >= args.kv_len:
            break

        attention_mask[0, current_len] = 1
        feed = {
            "input_ids": np.array([[next_token]], dtype=np.int64),
            "position_ids": np.array([[current_len]], dtype=np.int64),
            "attention_mask": attention_mask,
            **kv_cache,
        }
        raw = sess.run(output_names, feed)
        outputs = dict(zip(output_names, raw))
        update_kv_cache(kv_cache, outputs, args.layers, current_len)

        logits = outputs["logits"]
        next_token = int(np.argmax(logits[0, 0, :]))
        generated.append(next_token)
        current_len += 1

    text = tokenizer.decode(generated, skip_special_tokens=True)
    print(f"Prompt: {args.prompt}")
    print(f"Prompt tokens ({len(prompt_ids)}): {prompt_ids}")
    print(f"Generated tokens ({len(generated)}): {generated}")
    print(f"Generated text: {text}")


def main() -> int:
    run_generation(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
