<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Burn CPU

Build with `--features burn-runtime-cpu` to execute graphs on the host using Burn's
NdArray backend. No ONNX Runtime, GPU, or external native runtime library is required.
Automatic selection tries Burn CPU last, after every other eligible backend, including
ONNX Runtime CPU. It can also serve accelerated requests when no preferred backend is
available. An explicit hint for another backend does not fall back to Burn CPU.

```rust,ignore
use rustnn::mlcontext::{Backend, MLContext, MLContextOptions, MLPowerPreference};

let options = MLContextOptions::new(MLPowerPreference::Default, false)
    .with_rustnn_backend_hint(Backend::BurnCpu);
let mut context = MLContext::create(&options)?;
```

Use the normal builder, tensor and dispatch workflow from the
[Rust API reference](https://rustnn.github.io/rustnn/api/rustnn/).
Graph building lowers typed operations and embedded constants into an in-memory execution
plan for JIT preparation. The `burn-runtime-cpu` feature includes both conversion and
execution; no separate plan-generation feature is needed.
The builder owns operation and shape validation. Burn preparation consumes that validated
graph, checks its tensor representation limits, and initializes constant tensors once;
it does not repeat operation arity, datatype or structural validation.
Dispatch reuses that prepared state without decoding the serialized plan or rebuilding
constants. Intermediate tensors and their exact integer payloads are released after their
last consumer; graph outputs stay available until they are collected. Each dispatch clears
its remaining activations on success or error, so a failed run does not contaminate the next.
The standalone serialized-plan execution functions validate externally supplied plan
structure and metadata and prepare a fresh runtime on each call. These checks do not run
on the MLContext builder path. Tensor payload checks and kernel-result checks remain at
their storage and publication boundaries.
Dispatch replays typed operations with Burn tensors; operations without a device implementation
use host kernels. Float16 values are represented by float32 tensors with explicit rounding.
Tensors may have at most eight dimensions. Quantization and dequantization support packed
`int4` and `uint4` tensors, including scalar parameters, odd element counts, broadcasting
and block quantization. Tensor storage places even indices in the low nibble and odd indices
in the high nibble. Decoding ignores unused padding; output packing zeroes the unused high
nibble for odd lengths.
Packed 4-bit quantization is a rustnn extension outside the generated spec type
tables. Recording it requires
`MLContextOptions::with_rustnn_allow_int4_uint4(true)`. Other data type rules
remain enforced. To reload such a saved graph, pass the same
options to `loader::load_graph_from_path_with_options`.
Integer arithmetic and comparisons use exact signed or unsigned host payloads, including
32-bit and 64-bit values. Shape changes, selection, padding, gather/scatter and other data
movement preserve those payloads. Integer indices never pass through float32. Arithmetic
wraps to the output integer width, and integer-to-integer casts retain the low bits using
two's complement for signed types. Float-to-integer casts truncate and saturate to the target
range (NaN becomes zero). Clamp and padding options retain integer precision.
Quantization clips to the target integer range, and dequantization subtracts integer zero
points before conversion to floating point. Unsupported integer operations return an error
instead of silently using a floating-point kernel.

Each live operand has one authoritative value: a Burn float tensor or an exact signed or
unsigned integer buffer. Storage is private and the datatype is derived from its variant
and restricted integer kind. Checked constructors enforce shape, element count and integer
range before a complete value is inserted or replaced. Kernels borrow read-only views;
they cannot mutate the environment map or construct unchecked storage.
Integer operands do not retain float mirrors. The standalone executor's public output
object generates its float view only when results are collected. Every operation publishes
its results through one checked output boundary. It requires declared datatypes, rejects
missing or duplicate outputs, and checks graph output binding shapes before any results
of that operation become visible. Float16 operations may accumulate in float32, but every
operation output is rounded to float16 before a following node consumes it. Shared slice
and reduction descriptors keep shape and coordinate handling independent of datatype.
Closed layout, padding, interpolation and recurrent choices use enums inside the runtime.
Legacy string options retain their documented empty-string defaults; unknown nonempty
values return errors instead of selecting a fallback activation or layout.
The generic rank adapter covers the plan's ranks through eight. Burn NdArray's native
reshape currently supports only ranks through six, which also limits native paths such as
batched float matmul and singleton expansion. Failures in those paths return runtime errors.
Shared movement kernels support both float and exact integer tensors through rank eight.
Resampling derives scaled output dimensions with `floor(input size * scale)`;
explicit sizes take precedence. Sampling uses the resulting output dimensions for
half-pixel coordinates, including noninteger scaling.
Instance and layer normalization accumulate centered moments in float64 to retain small
variance when input values have a large common offset.

Serialized plans are checked for version, binding names, operand availability, output arity,
type metadata and constant sizes. Runtime inputs must match their bindings and element
counts. Exact integer payloads must match the declared datatype and element count; the
standalone API requires them for 32-bit and 64-bit integer inputs. The builder path relies
on the builder's operation and shape validation. Imported exact integer values outside
the declared width are rejected. Narrow integer inputs supplied through the standalone
API's float field must be finite, integral and in range. Arithmetic wrapping and cast
conversion are applied explicitly by kernels before constructing validated results.
Kernel assertion failures during execution are returned as runtime errors, with temporary
state cleared before reuse.

See the [generated operator report](https://rustnn.github.io/rustnn/development/backend-operator-support/)
for interpreter coverage. Support in that report does not imply conformance for every data type.

Run conformance against the cached WPT corpus (requires Node.js):

```bash
make test-wpt-burn-cpu
make wpt-sync-burn-cpu
```

For a focused run, use `make test-wpt-burn-cpu TEST_FILTER=relu`.
The expected-failure list is `tests/wpt_conformance/burn-cpu_expected_failures.txt`.
