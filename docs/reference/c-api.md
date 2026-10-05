# C and C++ API

rustnn exposes a C ABI in `rustnn.h` and a header-only C++17 wrapper in
`rustnn.hpp`. Both APIs are experimental and can change without backward
compatibility. Pin the release or Git commit used by your application.

Browse the [C and C++ API reference](https://rustnn.github.io/rustnn/c-api/)
for the generated declarations and source browser.

## Build the reference

Install [Doxygen](https://www.doxygen.nl/) with your system package manager and
cbindgen with `cargo install cbindgen --locked`, then run from the repository root:

```bash
make docs-capi
```

Open `target/doxygen/html/index.html` to browse the C functions, structures and
enums alongside the `rustnn` namespace and C++ classes. The target regenerates
`target/doxygen/include/rustnn/rustnn.h` directly from `src/capi.rs` using
`cbindgen.toml` before running Doxygen. It needs no backend SDK or library build.
The generated header and HTML stay under the ignored `target/` directory.

C documentation comes from Rust doc comments in `src/capi.rs`; C++ documentation
comes from `assets/capi/include/rustnn.hpp`. Doxygen expands the C++ operation
macros to include the named methods and their overloads.

The website build (`make docs-build` or `make ci-docs`, also `mkdocs build`
directly) runs the same target and embeds the HTML at `site/c-api/index.html`.
Serve `site/` after building to browse both references together. The Doxygen
pages have their own search index; MkDocs search covers the guide pages.

## C++ usage

Include `<rustnn/rustnn.hpp>` and link against the installed rustnn library.
The wrapper owns its handles, releases them in destructors, and supports moving
them. Handles cannot be copied. Calls that return a failing C status throw
`rustnn::Error`.

```cpp
#include <rustnn/rustnn.hpp>

int main() {
  try {
    rustnn::MLContextOptions options;
    options.accelerated = false;
    rustnn::MLContext context(options);
    rustnn::MLGraphBuilder builder(context);
    rustnn::MLOperandDescriptor descriptor(rustnn::MLOperandDataType::Float32, {2});
    auto input = builder.input("input", descriptor);
    auto output = builder.relu(input);
    auto graph = builder.build({{"output", &output}});

    rustnn::MLTensorDescriptor input_descriptor(
        rustnn::MLOperandDataType::Float32, {2}, false, true);
    rustnn::MLTensorDescriptor output_descriptor(
        rustnn::MLOperandDataType::Float32, {2}, true, false);
    auto input_tensor = context.createTensor(input_descriptor);
    auto output_tensor = context.createTensor(output_descriptor);
    context.writeTensor<float>(input_tensor, {-1.0f, 2.0f});
    context.dispatch(graph, {{"input", &input_tensor}}, {{"output", &output_tensor}});
    auto values = context.readTensor<float>(output_tensor, 2);
    return values == std::vector<float>{0.0f, 2.0f} ? 0 : 1;
  } catch (const rustnn::Error &error) {
    std::cerr << "rustnn error: " << error.what() << '\n';
    return 1;
  }
}
```

Keep the context alive while using its builder, graphs and tensors. A default
`rustnn::MLGraphBuilder` supports graph construction and text export without a
runtime; compiling with `build` requires a builder created with a context.
Building consumes the builder. Subsequent builder operations are invalid.

## C ownership and errors

Include `<rustnn/rustnn.h>`. Check each `RustnnStatus` result against
`RustnnStatus_Success`; `rustnn_last_error_message()` supplies thread-local failure
details. Its returned pointer is borrowed and must not be freed.

Release every owned handle exactly once with its matching `rustnn_*_destroy`
function. A successful `rustnn_graph_builder_build` consumes the builder and sets
the supplied builder pointer to null. Release text returned by
`rustnn_graph_builder_webnn_text` with `rustnn_string_destroy`.

Pointers must refer to live values of the declared type for the duration of each
call. Arrays and byte buffers must contain at least the supplied number of
elements or bytes. Tensor data is passed as raw bytes; use the descriptor's
element type, shape and packing when sizing buffers. Host reads and writes require
the tensor's `readable` and `writable` flags respectively.

For installation, CMake integration and complete C/C++ dispatch examples, see the
[C and C++ examples](https://github.com/rustnn/rustnn/tree/main/examples/capi).
The [WebNN specification](https://www.w3.org/TR/webnn/) describes the graph
operations; available operations and options follow the declarations in the
headers.

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

You can create cargo-c build of RustNN and configure it with your preferred runtimes (e.g. with onnx-runtime and trtx-runtime)

```bash
cargo cinstall --features capi,trtx-runtime,onnx-runtime
```
