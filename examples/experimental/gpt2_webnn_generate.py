#!/usr/bin/env python3
"""
Token-by-token GPT-2 generation demo using converted WebNN artifacts.

Expected model files (defaults point to /tmp):
  - gpt2_cached64.webnn
  - gpt2_cached64.weights
  - gpt2_cached64.manifest.json

The graph is compiled for:
  - sequence_length = 1
  - past_sequence_length = 64
  - attention_mask length = 65
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

try:
    import numpy as np
except ImportError:
    np = None

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None


@dataclass
class Gpt2Config:
    num_layers: int = 6
    num_heads: int = 12
    head_dim: int = 64
    kv_cache_len: int = 64


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="GPT-2 WebNN token generation demo")
    p.add_argument("--model", type=Path, default=Path("/tmp/gpt2_cached64.webnn"))
    p.add_argument("--weights", type=Path, default=Path("/tmp/gpt2_cached64.weights"))
    p.add_argument("--manifest", type=Path, default=Path("/tmp/gpt2_cached64.manifest.json"))
    p.add_argument("--prompt", type=str, default="Once upon a time")
    p.add_argument("--max-new-tokens", type=int, default=20)
    p.add_argument("--device", type=str, default="cpu")
    return p


def require_files(paths: List[Path]) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n  - " + "\n  - ".join(missing))


def init_kv_cache(cfg: Gpt2Config) -> Dict[str, np.ndarray]:
    kv = {}
    shape = (1, cfg.num_heads, cfg.kv_cache_len, cfg.head_dim)
    for i in range(cfg.num_layers):
        kv[f"past_key_values_{i}_key"] = np.zeros(shape, dtype=np.float32)
        kv[f"past_key_values_{i}_value"] = np.zeros(shape, dtype=np.float32)
    return kv


def update_kv_cache_from_result(
    cfg: Gpt2Config,
    kv_cache: Dict[str, np.ndarray],
    result: Dict[str, np.ndarray],
    pos: int,
) -> None:
    for i in range(cfg.num_layers):
        present_k = np.asarray(result[f"present_{i}_key"], dtype=np.float32)
        present_v = np.asarray(result[f"present_{i}_value"], dtype=np.float32)
        kv_cache[f"past_key_values_{i}_key"][:, :, pos : pos + 1, :] = present_k[
            :, :, pos : pos + 1, :
        ]
        kv_cache[f"past_key_values_{i}_value"][:, :, pos : pos + 1, :] = present_v[
            :, :, pos : pos + 1, :
        ]


def run_generation(args: argparse.Namespace) -> None:
    if np is None or AutoTokenizer is None:
        raise RuntimeError(
            "This demo requires numpy and transformers. Install with: pip install numpy transformers pywebnn"
        )

    try:
        import webnn
    except ImportError as exc:
        raise RuntimeError(
            "pywebnn is required to run this demo. Install with: pip install pywebnn"
        ) from exc

    cfg = Gpt2Config()
    require_files([args.model, args.weights, args.manifest])

    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    eos_token_id = tokenizer.eos_token_id

    ml = webnn.ML()
    context = ml.create_context(device_type=args.device)
    graph = webnn.MLGraph.load(
        str(args.model), manifest_path=str(args.manifest), weights_path=str(args.weights)
    )

    prompt_ids = tokenizer.encode(args.prompt, add_special_tokens=False)
    if not prompt_ids:
        raise ValueError("Prompt tokenization produced no tokens")
    if len(prompt_ids) >= cfg.kv_cache_len:
        raise ValueError(
            f"Prompt too long ({len(prompt_ids)}), must be < {cfg.kv_cache_len} tokens"
        )

    kv_cache = init_kv_cache(cfg)
    attention_mask = np.zeros((1, cfg.kv_cache_len + 1), dtype=np.int64)

    # Prefill: process prompt token-by-token to build cache.
    result = None
    for pos, token_id in enumerate(prompt_ids):
        attention_mask[0, pos] = 1
        feed = {
            "input_ids": np.array([[token_id]], dtype=np.int64),
            "position_ids": np.array([[pos]], dtype=np.int64),
            "attention_mask": attention_mask,
            **kv_cache,
        }
        result = context.compute(graph, feed)
        update_kv_cache_from_result(cfg, kv_cache, result, pos)

    assert result is not None
    logits = np.asarray(result["logits"], dtype=np.float32)
    next_token = int(np.argmax(logits[0, 0, :]))
    generated: List[int] = [next_token]

    current_len = len(prompt_ids)
    for _ in range(args.max_new_tokens - 1):
        if eos_token_id is not None and next_token == eos_token_id:
            break
        if current_len >= cfg.kv_cache_len:
            break

        attention_mask[0, current_len] = 1
        feed = {
            "input_ids": np.array([[next_token]], dtype=np.int64),
            "position_ids": np.array([[current_len]], dtype=np.int64),
            "attention_mask": attention_mask,
            **kv_cache,
        }
        result = context.compute(graph, feed)
        update_kv_cache_from_result(cfg, kv_cache, result, current_len)

        logits = np.asarray(result["logits"], dtype=np.float32)
        next_token = int(np.argmax(logits[0, 0, :]))
        generated.append(next_token)
        current_len += 1

    generated_text = tokenizer.decode(generated, skip_special_tokens=True)

    print(f"Prompt: {args.prompt}")
    print(f"Prompt tokens ({len(prompt_ids)}): {prompt_ids}")
    print(f"Generated tokens ({len(generated)}): {generated}")
    print(f"Generated text: {generated_text}")


def main() -> int:
    args = build_parser().parse_args()
    run_generation(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
