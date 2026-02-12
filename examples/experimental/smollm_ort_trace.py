#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Trace ORT decode loop for SmolLM ONNX")
    p.add_argument("--onnx", default="/tmp/smollm135/from_webnn.onnx")
    p.add_argument("--tokenizer", default="/tmp/smollm135/tokenizer.json")
    p.add_argument("--prompt", default="Once upon a time")
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--trace-file", default="/tmp/smollm135/py.trace")
    return p.parse_args()


def argmax(x: np.ndarray) -> int:
    return int(np.argmax(x))


def main() -> None:
    args = parse_args()
    sess_opts = ort.SessionOptions()
    sess_opts.log_severity_level = 3
    sess = ort.InferenceSession(args.onnx, sess_options=sess_opts, providers=["CPUExecutionProvider"])

    tokenizer = Tokenizer.from_file(args.tokenizer)
    enc = tokenizer.encode(args.prompt, add_special_tokens=False)
    prompt_ids = enc.ids

    num_layers = 0
    for i in sess.get_inputs():
        if i.name.startswith("past_key_values_") and i.name.endswith("_key"):
            layer = int(i.name.split("_")[3])
            num_layers = max(num_layers, layer + 1)

    # Static export used by this demo: [1, 3, 64, 64] cache and [1,65] mask.
    cache_len = 64
    num_heads = 3
    head_dim = 64
    attention_mask = np.zeros((1, cache_len + 1), dtype=np.int64)
    cache = {}
    for layer in range(num_layers):
        cache[f"past_key_values_{layer}_key"] = np.zeros((1, num_heads, cache_len, head_dim), dtype=np.float32)
        cache[f"past_key_values_{layer}_value"] = np.zeros((1, num_heads, cache_len, head_dim), dtype=np.float32)

    trace_lines = []
    current_pos = 0
    last_outputs = None

    def run_step(token_id: int):
        nonlocal current_pos, last_outputs
        attention_mask[0, current_pos] = 1
        mask_ones = int(attention_mask.sum())
        trace_lines.append(
            f"TRACE phase=prefill pos={current_pos} token_in={token_id} position_id={current_pos} mask_ones={mask_ones}"
        )
        feed = {
            "input_ids": np.array([[token_id]], dtype=np.int64),
            "position_ids": np.array([[current_pos]], dtype=np.int64),
            "attention_mask": attention_mask.copy(),
        }
        feed.update(cache)
        outputs = sess.run(None, feed)
        logits = outputs[0][0, 0, :]
        trace_lines.append(f"TRACE phase=prefill pos={current_pos} logits_argmax={argmax(logits)}")

        for layer in range(num_layers):
            pk = outputs[1 + layer * 2]
            pv = outputs[1 + layer * 2 + 1]
            cache[f"past_key_values_{layer}_key"][:, :, current_pos, :] = pk[:, :, current_pos, :]
            cache[f"past_key_values_{layer}_value"][:, :, current_pos, :] = pv[:, :, current_pos, :]
        current_pos += 1
        last_outputs = outputs

    for t in prompt_ids:
        run_step(int(t))

    generated = []
    for _ in range(args.max_new_tokens):
        assert last_outputs is not None
        next_id = argmax(last_outputs[0][0, 0, :])
        generated.append(next_id)
        trace_lines.append(f"TRACE phase=decode_select pos={current_pos} selected_token={next_id}")

        if current_pos >= cache_len:
            break

        attention_mask[0, current_pos] = 1
        mask_ones = int(attention_mask.sum())
        trace_lines.append(
            f"TRACE phase=decode_run pos={current_pos} token_in={next_id} position_id={current_pos} mask_ones={mask_ones}"
        )
        feed = {
            "input_ids": np.array([[next_id]], dtype=np.int64),
            "position_ids": np.array([[current_pos]], dtype=np.int64),
            "attention_mask": attention_mask.copy(),
        }
        feed.update(cache)
        outputs = sess.run(None, feed)
        logits = outputs[0][0, 0, :]
        trace_lines.append(f"TRACE phase=decode_run pos={current_pos} logits_argmax={argmax(logits)}")

        for layer in range(num_layers):
            pk = outputs[1 + layer * 2]
            pv = outputs[1 + layer * 2 + 1]
            cache[f"past_key_values_{layer}_key"][:, :, current_pos, :] = pk[:, :, current_pos, :]
            cache[f"past_key_values_{layer}_value"][:, :, current_pos, :] = pv[:, :, current_pos, :]
        current_pos += 1
        last_outputs = outputs

    out = Path(args.trace_file)
    out.write_text("\n".join(trace_lines) + "\n", encoding="utf-8")
    print(f"Prompt token ids: {prompt_ids}")
    print(f"Generated token ids: {generated}")
    print(f"Trace file: {out}")


if __name__ == "__main__":
    main()
