# SmolLM2 on Apple devices

`examples/smollm2_coreml.rs` runs `HuggingFaceTB/SmolLM2-135M-Instruct` through `MLContext` on
Core ML, starting from the published checkpoint. This page is the whole path: download, author
the graph, run it on a Mac, then on an iPhone and an Apple Watch, and compare your receipt with
the reference runs at the end.

Nothing here needs Python packages, an ONNX export or a prebuilt model. The example reads
`model.safetensors`, quantizes the weights and records the graph with `MLGraphBuilder`.

## What the example builds

A single-token decode graph with a fixed key/value window (512 positions by default, newest
last). Each dispatch takes one token, the rotary `cos`/`sin` row and the mask row of its
position, and returns the logits.

- Projection weights are 8-bit per block of 32 elements with float16 scales, stored as uint8
  with zero point 128. The token embedding, which is also the output head, is int8 per block
  of 64. The CoreML converter keeps both compressed
  ([Quantized constant weights](../integration/coreml.md#quantized-constant-weights)).
- Projections, attention, the MLP products and the key/value window are float16. The residual
  stream, the RMSNorm statistics and the logits are float32.
- The key/value window is carried inside the model as an `MLState` (`--carry state`, the
  default) or bound as tensors (`--carry tensors`); see
  [Carrying recurrent state inside the model](../integration/coreml.md#carrying-recurrent-state-inside-the-model).

`examples/smollm2_coreml/author.rs` has the graph and the quantization recipe, with unit tests.

## 1. Get the checkpoint

The reference runs use revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`.

```bash
mkdir -p smollm2 && cd smollm2
base=https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/resolve/12fd25f77366fa6b3b4b768ec3050bf629380bac
curl -L -O $base/model.safetensors -O $base/tokenizer.json
shasum -a 256 model.safetensors
# 5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c
```

## 2. Author the bundle and run it on a Mac

```bash
cargo run --release --no-default-features \
  --features native-examples,coreml-runtime,coreml-export,dynamic-inputs \
  --example smollm2_coreml -- --checkpoint smollm2 --bundle smollm2-bundle
```

`--checkpoint` authors `smollm2-bundle/model.webnn`, `model.safetensors` (142 MB) and a copy of
`tokenizer.json`, then decodes the default prompt. Later runs only need `--bundle`. The
receipt is printed as JSON and, with `--receipt FILE`, written to a file:

| Field | Meaning |
|---|---|
| `buildSeconds` | `MLContext::rustnn_build_graph`: conversion, compile or export, and load |
| `tokensPerSecond` | Decode rate over the generated tokens, one dispatch per token |
| `peakResidentBytes` | `ru_maxrss` of the process at the end of the run |
| `footprintBytes`, `peakFootprintBytes` | Physical footprint now and its lifetime peak, the figure iOS and watchOS hold an app to; also recorded after the graph load and after the build |
| `tokenHash` | FNV-1a over the prompt and generated token ids |
| `text` | The decoded reply |

To compare with the reference runs, use the fixed prompt and plain greedy decoding:

```bash
... --example smollm2_coreml -- --bundle smollm2-bundle \
  --prompt-ids 1,9690,198,2683,359,253,5356,29,11181,351,4226 \
  --repetition-penalty 1.0 --ignore-end-of-turn --max-new-tokens 64
```

Other switches: `--carry state|tensors`, `--units cpu|gpu|npu`, `--route auto|native|export`,
`--cache DIR` (compiled-bundle cache), `--window N` and `--layers N` (authoring only),
`--prompt TEXT`, `--repetition-penalty`. `--help` lists them.

## 3. Run it on an iPhone

An app must run the system's application loop, so on a device the example is linked as a
static library into a small SwiftUI host. `scripts/apple_device_app.py` does the packaging
and the device round trip. You need Xcode, a development certificate, and a provisioning
profile that covers your bundle identifier and device.

```bash
rustup target add aarch64-apple-ios
IPHONEOS_DEPLOYMENT_TARGET=18.0 cargo build --release --target aarch64-apple-ios \
  --no-default-features --features native-examples,coreml-runtime,coreml-export,dynamic-inputs \
  --example smollm2_coreml_device

python3 scripts/apple_device_app.py package --platform ios \
  --library target/aarch64-apple-ios/release/examples/libsmollm2_coreml_device.a \
  --data smollm2-bundle --bundle-id com.example.smollm2 \
  --team TEAMID --identity "Apple Development: Your Name (XXXXXXXXXX)" \
  --profile your.mobileprovision --out build/ios

xcrun devicectl list devices
python3 scripts/apple_device_app.py run --device DEVICE-ID --app build/ios/Example.app \
  --receipt iphone.json -- --bundle model --receipt @RECEIPT@
```

`run` installs the app, launches it with the arguments after `--`, waits for the receipt in
the app's `Library/Caches` and prints it. The bundle is inside the app as `model`. The phone
must be unlocked.

## 4. Run it on an Apple Watch

The same two commands with the watchOS target and `--platform watchos`:

```bash
rustup target add aarch64-apple-watchos
WATCHOS_DEPLOYMENT_TARGET=11.0 CFLAGS_aarch64_apple_watchos=-DCFG_TARGET_OS_ios \
  cargo build --release --target aarch64-apple-watchos \
  --no-default-features --features native-examples,coreml-runtime,coreml-export,dynamic-inputs \
  --example smollm2_coreml_device
```

- `CFLAGS_aarch64_apple_watchos=-DCFG_TARGET_OS_ios` is for the `psm` crate, a dependency of
  the `.webnn` parser. Its assembly only knows macOS and iOS as Mach-O targets and otherwise
  emits ELF directives; the define selects the Mach-O form.
- `coreml-export` is required: watchOS has no model compiler, so the bundle is written by the
  pure-Rust exporter ([Compile routes](../integration/coreml.md#compile-routes)).
- `aarch64-apple-watchos` is the arm64 slice, which the Apple Watch Series 10 on watchOS 27.2
  runs. Older watches such as the Apple Watch SE 2nd generation are `arm64_32-apple-watchos`,
  a tier 3 target that needs `-Zbuild-std` and was not tried with this example.
- watchOS suspends an app whose screen is asleep. Keep the watch awake during a run, or
  ignore the timing.

**The full model does not fit an Apple Watch yet.** On the Apple Watch Series 10 (watchOS
27.2) the process is killed while the model loads. An app there was granted 299 MiB when
measured on 2026-10-30; no jetsam report was pulled to confirm the reason. The footprint on
the Mac shows why:

| Phase (M4 Max, state, auto) | Full model | `--layers 2` |
|---|---|---|
| After the graph is loaded | 304 MiB | 97 MiB |
| Peak during the build | 626 MiB | 156 MiB |
| While decoding | 319 MiB | 157 MiB |

The bundle holds 142 MB of weights. Loading it keeps two copies on the heap for a moment
(the file and the per-tensor constants), and the build adds a copy of the graph and the
weights blob. Core ML itself maps its weights file and adds almost nothing: while decoding,
9 MB of the heap is in use and the rest is freed memory the allocator has not returned. So
the limit is rustnn's transient copies, not the model. Keeping the constants file-backed
through load, conversion and export is the fix, and it is not in this repository yet.

A two-layer cut of the model (`--layers 2` when authoring) runs on the watch end to end and
returns the Mac's tokens, which shows the build, the export route and the state carry work
there.

## Reference runs

2026-11-17, CPU, the fixed prompt above, 64 greedy tokens, one run per row unless a range is
given. Rates move with thermal state; the token hash is what should match.

| Device | Carry, route | Build | Rate | Peak resident | Token hash |
|---|---|---|---|---|---|
| MacBook Pro M4 Max, macOS 27.3 | state, auto | 1.1 s | 374 tok/s | 890 MiB | `2db53a2d95685f0b` |
| | tensors, export | 0.8 s | 287 tok/s | 880 MiB | `2db53a2d95685f0b` |
| | tensors, auto | 5.8 s | 55 tok/s | 994 MiB | `e2b450fbb2d994c6` |
| iPhone SE 2nd gen (A13), iOS 26.7.1 | state, auto | 4.6 to 5.0 s | 41 to 52 tok/s (2 runs) | 672 to 819 MiB | `2db53a2d95685f0b` |
| | tensors, export | 4.2 s | 43 tok/s | 745 MiB | `2db53a2d95685f0b` |
| | tensors, auto | 8.7 s | 23 tok/s | 843 MiB | `e2b450fbb2d994c6` |
| iPhone XS Max (A12), iOS 18.7.10 | tensors, export | 3.4 s | 23 tok/s | 724 MiB | `2db53a2d95685f0b` |
| | tensors, auto | 9.9 s | 19 tok/s | 841 MiB | `2db53a2d95685f0b` |
| Apple Watch Series 10, watchOS 27.2 | state, auto, `--layers 2`, 16 tokens | 0.7 s | not timed | 215 MiB | `deb9a7079c2fa62e`, as on the Mac |

What to expect when your numbers differ:

- `tensors, auto` loads a stateless graph in memory on the Mac and the A13. That route is
  slower and rounds float16 differently, hence the second hash
  ([How a graph runs](../integration/coreml.md#how-a-graph-runs)).
- With `--carry state` the model fails to load on iOS 18.7.10 (Core ML error -14 on the A12
  above, also for the two-layer cut). Use `--carry tensors` there. The cause is not known.
- With the default chat prompt and repetition penalty, the Mac and the A13 produce different
  replies after the first tokens. Both are coherent. Float16 kernels differ between
  processors, and a penalized near-tie can go either way; the fixed prompt above agrees on
  every device listed.
- GPU and Neural Engine requests (`--units gpu|npu`) are accepted but were not measured here.
