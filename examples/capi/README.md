# rustnn C and C++ examples

## TL;DL

Run from repo root

```bash
cargo cinstall \
  --features capi,trtx-runtime \
  --prefix="$(pwd)/target/rustnn-capi-install"
cmake -GNinja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -S examples/capi/ -B build \
 -DCMAKE_PREFIX_PATH=$(pwd)/target/rustnn-capi-install
cmake --build build
```

- [`main.c`](main.c) uses the generated C API directly.
- [`main.cpp`](main.cpp) uses the header-only C++ RAII wrapper.

## API stability

The C ABI and C++ API are experimental and are not stable. They are subject to
change without backward compatibility, like the public Rust APIs. Pin the
rustnn release or Git commit used by your application and rebuild your consumer
when updating rustnn.

The runtime-facing parts of these APIs are expected to evolve in particular:

- The ownership and lifetime rules for host pointers and externally owned
  tensor buffers are not finalized.
- GPU-to-host and host-to-GPU synchronization semantics are not finalized.
- Operations that are synchronous today may become asynchronous or gain
  asynchronous variants, completion objects, or callbacks.

Do not treat the present pointer, synchronization, or blocking dispatch
behavior as a long-term ABI contract.

## Prerequisites

- A Rust toolchain compatible with this repository
- [cargo-c](https://github.com/lu-zero/cargo-c)
- CMake 3.16 or newer
- A C11 and C++17 compiler
- At least one rustnn runtime backend and its native dependencies

Install cargo-c if it is not already available:

```bash
cargo install cargo-c
```

## Build and install the C package

From the rustnn repository root, install into a local prefix:

```bash
cargo cinstall \
  --features capi,onnx-runtime \
  --prefix="$PWD/target/rustnn-capi-install"
```

Replace `onnx-runtime` with another supported backend feature when appropriate,
such as `trtx-runtime`, `coreml-runtime`, or `litert-runtime`. The selected
backend's native runtime must also be available when the examples run. For ONNX
Runtime development builds, `make onnxruntime-download` downloads the version
expected by this repository; set `ORT_DYLIB_PATH` to the downloaded shared
library when running the examples if it is not discoverable automatically.

The installed package contains:

```text
include/rustnn/rustnn.h       Generated C API
include/rustnn/rustnn.hpp     Header-only C++ wrapper
lib/.../librustnn.so          Shared library on Linux
lib/.../librustnn.a           Static library on Linux
lib/.../pkgconfig/rustnn.pc   pkg-config metadata
share/rustnn/cmake/           CMake package configuration
```

Library names and extensions vary by platform.

## Build and run both examples

Configure the examples against the local installation:

```bash
cmake \
  -S examples/capi \
  -B target/examples/capi \
  -DCMAKE_PREFIX_PATH="$PWD/target/rustnn-capi-install"
cmake --build target/examples/capi --parallel
```

Run the raw C example:

```bash
./target/examples/capi/rustnn_c_example
```

Run the C++ wrapper example:

```bash
./target/examples/capi/rustnn_cpp_example
```

Both programs print the inferred output shape, the same serialized graph, and:

```text
dispatch output: 4 0 10 0
```

If a native backend or the installed rustnn library is not discoverable, set
its documented loader variable. Depending on the platform and backend this can
include `ORT_DYLIB_PATH`, `LD_LIBRARY_PATH`, `DYLD_LIBRARY_PATH`, or `PATH`.

## Using the C API

Include the generated header and link the imported CMake target:

```c
#include <rustnn/rustnn.h>
```

The C API follows these conventions:

- Call `rustnn_init_logger()` after setting `RUST_LOG` to install
  `pretty_env_logger`. Initialization is idempotent and has no effect if the
  process has already installed another global Rust logger.
- Functions return `RustnnStatus`. Compare it with `RustnnStatus_Success`.
- `rustnn_last_error_message()` describes the most recent failure on the
  calling thread. The returned pointer is borrowed and is only valid until the
  next C API call on that thread.
- Handles are opaque. Destroy each owned handle exactly once with its matching
  `rustnn_*_destroy` function.
- A runtime-backed builder temporarily borrows its context. Do not call any
  context function or destroy the context until the builder is built or
  destroyed. `rustnn_graph_builder_build()` consumes the builder and sets its
  handle to null, including when compilation reports an error.
- The originating context must outlive its compiled graphs and tensors.
- Functions that return an object write it through an output pointer. The
  output remains null on failure.
- `RustnnOperatorOptions` supplies the WebNN operator label. Operations are
  available both with default options, such as `rustnn_graph_builder_add()`,
  and explicitly as `rustnn_graph_builder_add_with_options()`.
- `rustnn_graph_builder_unary()`, `rustnn_graph_builder_binary()`, and
  `rustnn_graph_builder_ternary()` expose all operations whose required inputs
  are only operands. Their operation enums include activation, normalization,
  convolution, pooling, reduction, gather/scatter, and matrix operations.
- Operations with additional required arguments use typed entry points such as
  `rustnn_graph_builder_arg_max()`, `rustnn_graph_builder_cast()`,
  `rustnn_graph_builder_concat()`, `rustnn_graph_builder_pad()`,
  `rustnn_graph_builder_split()`, and `rustnn_graph_builder_lstm()`.
- The C ABI covers all 101 operation variants available from Rust
  `MLGraphBuilder`; it invokes those builder methods directly and does not use
  a JSON operation bridge.

Initialize `RustnnContextOptions` with `rustnn_context_options_default()` so
backend-specific settings retain the Rust API defaults. `backend_hint` selects
a backend while still letting that backend select its device:

```c
RustnnContextOptions options = rustnn_context_options_default();
options.accelerated = true;
options.backend_hint = RustnnBackend_Trtx;
RustnnContext *context = NULL;
RustnnStatus status = rustnn_context_create(&options, &context);
```

For an exact device, set `has_device_hint` and fill `device_hint`. ONNX uses
`device_index` as its execution-provider device index and TensorRT uses it as
the CUDA device index. CoreML, LiteRT, and CANN currently require index zero.
Set `has_rustnn_options` to override the RustNN-specific defaults in
`rustnn_options`; currently only TensorRT has non-empty backend options.

A minimal concrete operation looks like this:

```c
const RustnnOperatorOptions options = {"add bias"};
RustnnOperand *sum = NULL;
RustnnStatus status = rustnn_graph_builder_add_with_options(
    builder, input, bias, &options, &sum);
if (status != RustnnStatus_Success) {
  fprintf(stderr, "%s\n", rustnn_last_error_message());
}
```

MLTensor execution uses `rustnn_context_create_tensor()`,
`rustnn_context_write_tensor()`, `rustnn_context_dispatch()`, and
`rustnn_context_read_tensor()`. See [`main.c`](main.c) for complete graph
compilation, error handling, dispatch, and cleanup.

## Using the C++ API

Include the wrapper header and link the same CMake target:

```cpp
#include <rustnn/rustnn.hpp>
```

The wrapper provides move-only RAII classes. Contexts, graphs, tensors,
descriptors, operands, and graph builders release their C handles automatically.
Failures are reported as `rustnn::Error` exceptions.

Call `rustnn::initializeLogger()` after setting `RUST_LOG` to initialize logging
through the C API.

```cpp
rustnn::MLContext context;
rustnn::MLGraphBuilder builder(context);
rustnn::MLOperandDescriptor matrix(rustnn::MLOperandDataType::Float32, {2, 2});
auto input = builder.input("input", matrix);
auto bias = builder.constant<float>(matrix, {1.0f, 2.0f, 3.0f, 4.0f});
auto sum = builder.add(input, bias, rustnn::MLOperatorOptions{"add bias"});
auto graph = builder.build({{"sum", &sum}});
```

Backend and device selection are also available through `rustnn::MLContextOptions`:

```cpp
rustnn::MLContextOptions options;
options.accelerated = true;
options.backend_hint = rustnn::Backend::Trtx;
options.device_hint = rustnn::BackendDevice{
    rustnn::Backend::Trtx, rustnn::DeviceType::Gpu, 0};
rustnn::MLContext context(options);
```

The generic `unary()` and `binary()` methods remain available when an operation
is selected dynamically. Named methods cover all 101 Rust `MLGraphBuilder`
operation variants, including typed overloads for parameterized operations and
`std::vector<rustnn::MLOperand>` results for split and recurrent operations.
`MLContext::createTensor()`, `writeTensor()`, `dispatch()`, and `readTensor()`
wrap the corresponding C execution APIs. `operandShape()` queries an operand's
inferred dimensions, while `shape()` creates the WebNN shape operation. See
[`main.cpp`](main.cpp) for the complete example.

## Consuming rustnn from another CMake project

Point `CMAKE_PREFIX_PATH` at the cargo-c installation and import the package:

```cmake
cmake_minimum_required(VERSION 3.16)
project(my_rustnn_app LANGUAGES C CXX)

find_package(rustnn CONFIG REQUIRED)

add_executable(my_rustnn_app main.cpp)
target_compile_features(my_rustnn_app PRIVATE cxx_std_17)
target_link_libraries(my_rustnn_app PRIVATE rustnn::rustnn)
```

Configure it with:

```bash
cmake -S . -B build \
  -DCMAKE_PREFIX_PATH=/absolute/path/to/rustnn-capi-install
cmake --build build
```
For example, to install with TensorRT and build the samples with Ninja from the
repository root:

```bash
cargo cinstall \
  --features capi,trtx-runtime \
  --prefix="$PWD/target/rustnn-capi-install"
cmake -GNinja -DCMAKE_BUILD_TYPE=RelWithDebInfo -S examples/capi/ -B build \
 -DCMAKE_PREFIX_PATH=$(pwd)/target/rustnn-capi-install
cmake --build build
```
