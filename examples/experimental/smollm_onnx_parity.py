#!/usr/bin/env python3
import argparse
import re
from dataclasses import dataclass

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


LAYER_KEY_RE = re.compile(r"past_key_values[._](\d+)[._]key$")
LAYER_VALUE_RE = re.compile(r"past_key_values[._](\d+)[._]value$")
PRESENT_KEY_RE = re.compile(r"present[._](\d+)[._]key$")
PRESENT_VALUE_RE = re.compile(r"present[._](\d+)[._]value$")


def argmax(x: np.ndarray) -> int:
    return int(np.argmax(x))


def max_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a.astype(np.float32) - b.astype(np.float32))))


def mean_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))


@dataclass
class ModelIO:
    input_ids: str
    attention_mask: str
    position_ids: str
    logits: str
    past_key_inputs: dict[int, str]
    past_value_inputs: dict[int, str]
    present_key_outputs: dict[int, str]
    present_value_outputs: dict[int, str]


def inspect_io(sess: ort.InferenceSession) -> ModelIO:
    in_names = [x.name for x in sess.get_inputs()]
    out_names = [x.name for x in sess.get_outputs()]

    def find_exact(name: str, names: list[str]) -> str:
        if name in names:
            return name
        raise RuntimeError(f"missing required tensor: {name}")

    input_ids = find_exact("input_ids", in_names)
    attention_mask = find_exact("attention_mask", in_names)
    position_ids = find_exact("position_ids", in_names)
    logits = "logits" if "logits" in out_names else out_names[0]

    past_key_inputs: dict[int, str] = {}
    past_value_inputs: dict[int, str] = {}
    present_key_outputs: dict[int, str] = {}
    present_value_outputs: dict[int, str] = {}

    for name in in_names:
        m = LAYER_KEY_RE.search(name)
        if m:
            past_key_inputs[int(m.group(1))] = name
            continue
        m = LAYER_VALUE_RE.search(name)
        if m:
            past_value_inputs[int(m.group(1))] = name

    for name in out_names:
        m = PRESENT_KEY_RE.search(name)
        if m:
            present_key_outputs[int(m.group(1))] = name
            continue
        m = PRESENT_VALUE_RE.search(name)
        if m:
            present_value_outputs[int(m.group(1))] = name

    return ModelIO(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        logits=logits,
        past_key_inputs=past_key_inputs,
        past_value_inputs=past_value_inputs,
        present_key_outputs=present_key_outputs,
        present_value_outputs=present_value_outputs,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare logits/past cache between source ONNX and rustnn-exported ONNX")
    p.add_argument("--source-onnx", default="/Users/tarek/Downloads/smol_hf.onnx")
    p.add_argument("--converted-onnx", default="/tmp/smol_from_rustnn_dyn.onnx")
    p.add_argument("--tokenizer", default="/tmp/smol_hf_dyn/tokenizer.json")
    p.add_argument("--prompt", default="Once upon a time")
    p.add_argument("--decode-steps", type=int, default=8)
    p.add_argument("--topk", type=int, default=10)
    return p.parse_args()


def build_zero_cache(io: ModelIO, layers: list[int], past_len: int) -> dict[str, np.ndarray]:
    cache: dict[str, np.ndarray] = {}
    for layer in layers:
        cache[io.past_key_inputs[layer]] = np.zeros((1, 3, past_len, 64), dtype=np.float32)
        cache[io.past_value_inputs[layer]] = np.zeros((1, 3, past_len, 64), dtype=np.float32)
    return cache


def run_step(
    sess: ort.InferenceSession,
    io: ModelIO,
    layers: list[int],
    cache: dict[str, np.ndarray],
    token_id: int,
    pos: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    mask = np.ones((1, pos + 1), dtype=np.int64)
    feed = {
        io.input_ids: np.array([[token_id]], dtype=np.int64),
        io.position_ids: np.array([[pos]], dtype=np.int64),
        io.attention_mask: mask,
    }
    feed.update(cache)
    out = sess.run(None, feed)
    out_names = [x.name for x in sess.get_outputs()]
    out_map = {name: arr for name, arr in zip(out_names, out)}
    logits = out_map[io.logits][0, 0, :].astype(np.float32)
    new_cache: dict[str, np.ndarray] = {}
    for layer in layers:
        new_cache[io.past_key_inputs[layer]] = out_map[io.present_key_outputs[layer]].astype(np.float32)
        new_cache[io.past_value_inputs[layer]] = out_map[io.present_value_outputs[layer]].astype(np.float32)
    return logits, new_cache


def topk_ids(x: np.ndarray, k: int) -> list[int]:
    idx = np.argpartition(x, -k)[-k:]
    idx = idx[np.argsort(x[idx])[::-1]]
    return idx.astype(int).tolist()


def main() -> None:
    args = parse_args()
    source_sess = ort.InferenceSession(args.source_onnx, providers=["CPUExecutionProvider"])
    conv_sess = ort.InferenceSession(args.converted_onnx, providers=["CPUExecutionProvider"])
    source_io = inspect_io(source_sess)
    conv_io = inspect_io(conv_sess)

    source_layers = sorted(set(source_io.past_key_inputs) & set(source_io.past_value_inputs))
    conv_layers = sorted(set(conv_io.past_key_inputs) & set(conv_io.past_value_inputs))
    common_layers = sorted(set(source_layers) & set(conv_layers))
    if not common_layers:
        raise RuntimeError("No common cache layers found between source and converted models")

    tokenizer = Tokenizer.from_file(args.tokenizer)
    prompt_ids = tokenizer.encode(args.prompt, add_special_tokens=False).ids
    if not prompt_ids:
        raise RuntimeError("Prompt encoded to empty token sequence")

    source_cache = build_zero_cache(source_io, common_layers, past_len=0)
    conv_cache = build_zero_cache(conv_io, common_layers, past_len=0)

    # Use source model's greedy path as the canonical token stream.
    token_stream = [int(x) for x in prompt_ids]
    source_generated: list[int] = []

    print(f"Prompt token ids: {token_stream}")
    print(f"Common layers: {len(common_layers)} (0..{common_layers[-1]})")

    step = 0
    for token_id in token_stream:
        src_logits, source_cache = run_step(source_sess, source_io, common_layers, source_cache, token_id, step)
        conv_logits, conv_cache = run_step(conv_sess, conv_io, common_layers, conv_cache, token_id, step)
        src_top1 = argmax(src_logits)
        conv_top1 = argmax(conv_logits)
        print(
            f"prefill step={step} token_in={token_id} src_top1={src_top1} conv_top1={conv_top1} "
            f"max_abs={max_abs(src_logits, conv_logits):.6f} mean_abs={mean_abs(src_logits, conv_logits):.6f}"
        )
        if step == len(token_stream) - 1:
            for _ in range(args.decode_steps):
                next_token = argmax(src_logits)
                source_generated.append(next_token)
                step += 1
                src_logits, source_cache = run_step(
                    source_sess, source_io, common_layers, source_cache, next_token, step
                )
                conv_logits, conv_cache = run_step(conv_sess, conv_io, common_layers, conv_cache, next_token, step)
                src_top1 = argmax(src_logits)
                conv_top1 = argmax(conv_logits)
                src_topk = topk_ids(src_logits, args.topk)
                conv_topk = topk_ids(conv_logits, args.topk)
                overlap = len(set(src_topk) & set(conv_topk))
                print(
                    f"decode step={step} token_in={next_token} src_top1={src_top1} conv_top1={conv_top1} "
                    f"top{args.topk}_overlap={overlap}/{args.topk} "
                    f"max_abs={max_abs(src_logits, conv_logits):.6f} mean_abs={mean_abs(src_logits, conv_logits):.6f}"
                )
            break
        step += 1

    print(f"Source greedy generated ids: {source_generated}")


if __name__ == "__main__":
    main()
