# Browser WebNN (wasm32)

With the `webnn-runtime` feature and the `wasm32-unknown-unknown` target, rustnn compiles to
WebAssembly and can hand its graphs to the browser's own WebNN implementation. This is the
starting point for running the same Rust application natively and in a web page.

## What exists

- `src/backends/webnn/api_generated/`: `wasm-bindgen` bindings for the WebNN IDL (`Ml`,
  `MlContext`, `MlGraphBuilder`, `MlOperand`, `MlTensor`, all option dictionaries and enums).
  They are generated from the specification's IDL with the `web-sys` tooling and vendored until
  they land upstream.
- `src/converters/webnn.rs`: replays a `GraphInfo` through an `MlGraphBuilder` of the browser
  (`convert_async`), mapping every `Operation` variant and its options to the JavaScript calls.
- `tests/webnn_wpt.rs`: a `wasm-bindgen-test` that embeds the WPT conformance corpus at build
  time (`webnn-wpt-tests` feature, `build.rs` runs the Node bridge) and checks that every case
  compiles to a `GraphInfo` and, when `navigator.ml` exists, builds in the browser.
  `tests/wpt_conformance/webnn_chrome_expected_failures.rs` lists the cases Chrome rejects.

Not there yet: a `BackendDevice`/`MLBackendContext` for the browser, so `MLContext::create`
cannot select this backend and tensors and dispatch are not wired. The generated methods are
not marked `catch`, so a browser exception traps out of WebAssembly instead of returning an
error.

## Building

```bash
rustup target add wasm32-unknown-unknown
RUSTFLAGS="-C target-feature=+reference-types --cfg=web_sys_unstable_apis" \
  cargo check --target wasm32-unknown-unknown --no-default-features -F webnn-runtime
```

`web_sys_unstable_apis` is required because the WebNN bindings are unstable in `web-sys`;
`reference-types` matches what the generated bindings expect. CI runs this check plus the
all-targets variant with `webnn-wpt-tests`.

## Running the WPT graph-build tests in Chrome

```bash
make fetch-wpt                    # the corpus is embedded into the test binary
make webnn-chromedriver           # downloads a ChromeDriver matching the installed Chrome
make test-webnn-wpt-chrome        # or test-webnn-wpt-chrome-headless
```

The targets need `wasm-pack`, Node.js, `curl` and `unzip`. `webdriver.json` enables Chrome's
WebNN flag (`--enable-features=WebMachineLearningNeuralNetwork`). Results are logged to the
browser console as `[WEBNN-WPT PASS|FAIL|SKIP] operation::case` lines followed by a summary.
