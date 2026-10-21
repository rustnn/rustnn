/// Generated files come from following the instructions in
/// https://github.com/wasm-bindgen/wasm-bindgen/tree/main/crates/web-sys
/// on the WebNN bikeshed file.
///
/// The `MLGraphBuilder` operation methods (all but `build`) were then marked `catch` and return
/// `Result<_, JsValue>` by hand: the IDL does not declare that they throw, but browsers reject
/// unsupported ranks and data types with a `TypeError`. Reapply this after regenerating.
pub mod api_generated;
