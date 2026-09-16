# rustnn C and C++ examples

These examples build the same small WebNN graph through the installed rustnn C
package:

```text
output = relu((input + bias) * scale)
```

They inspect its inferred `2 x 2` output shape and print its serialized `.webnn`
representation. They intentionally do not select a runtime backend, dispatch a
graph, or run a neural network.

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

The current C package focuses on backend-agnostic graph construction and
serialization. Do not treat its present pointer or synchronization behavior as
a long-term ABI contract.

## Prerequisites

- A Rust toolchain compatible with this repository
- [cargo-c](https://github.com/lu-zero/cargo-c)
- CMake 3.16 or newer
- A C11 and C++17 compiler

Install cargo-c if it is not already available:

```bash
cargo install cargo-c
```

## Build and install the C package

From the rustnn repository root, install into a local prefix:

```bash
cargo cinstall \
  --features capi \
  --prefix="$PWD/target/rustnn-capi-install"
```

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

Both programs print the inferred output shape followed by the same serialized
graph. If a platform does not embed the installation directory in the
executable's runtime search path, add the installed library directory to
`LD_LIBRARY_PATH`, `DYLD_LIBRARY_PATH`, or `PATH`, as appropriate.

## Using the C API

Include the generated header and link the imported CMake target:

```c
#include <rustnn/rustnn.h>
```

The C API follows these conventions:

- Functions return `RustnnStatus`. Compare it with `RustnnStatus_Success`.
- `rustnn_last_error_message()` describes the most recent failure on the
  calling thread. The returned pointer is borrowed and is only valid until the
  next C API call on that thread.
- Handles are opaque. Destroy each owned handle exactly once with its matching
  `rustnn_*_destroy` function.
- Functions that return an object write it through an output pointer. The
  output remains null on failure.
- `RustnnOperatorOptions` supplies the WebNN operator label. Operations are
  available both with default options, such as `rustnn_graph_builder_add()`,
  and explicitly as `rustnn_graph_builder_add_with_options()`.
- `rustnn_graph_builder_unary()` and `rustnn_graph_builder_binary()` are useful
  for dynamically selected operations; named functions are preferable when
  the operation is known at compile time.

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

See [`main.c`](main.c) for complete error handling and cleanup.

## Using the C++ API

Include the wrapper header and link the same CMake target:

```cpp
#include <rustnn/rustnn.hpp>
```

The wrapper provides move-only RAII classes. Descriptors, operands, and graph
builders release their C handles automatically. Failures are reported as
`rustnn::Error` exceptions.

```cpp
rustnn::GraphBuilder builder;
rustnn::OperandDescriptor matrix(rustnn::DataType::Float32, {2, 2});
auto input = builder.input("input", matrix);
auto bias = builder.constant<float>(matrix, {1.0f, 2.0f, 3.0f, 4.0f});
auto sum = builder.add(input, bias, rustnn::OperatorOptions{"add bias"});
```

The generic `unary()` and `binary()` methods remain available when an operation
is selected dynamically. See [`main.cpp`](main.cpp) for the complete example.

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


E.g. from the root of this repository (please select at least one runtime feature `onnx-runtime`,`trtx-runtime`, `coreml-runtime`, `litert-runtime`)

```bash
cargo cinstall \
  --features capi,trtx-runtime \
  --prefix="$PWD/target/rustnn-capi-install"
cmake -GNinja -DCMAKE_BUILD_TYPE=RelWithDebInfo -S examples/capi/ -B build \
 -DCMAKE_PREFIX_PATH=$(pwd)/target/rustnn-capi-install
cmake --build build
```
