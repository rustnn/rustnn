#![allow(unused_imports)]
#![allow(clippy::all)]
use wasm_bindgen::prelude::*;
#[cfg(web_sys_unstable_apis)]
#[wasm_bindgen]
#[doc = "The `MlOperandDataType` enum."]
#[doc = ""]
#[doc = "*This API requires the following crate features to be activated: `MlOperandDataType`*"]
#[doc = ""]
#[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
#[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MlOperandDataType {
    Float32 = "float32",
    Float16 = "float16",
    Int32 = "int32",
    Uint32 = "uint32",
    Int64 = "int64",
    Uint64 = "uint64",
    Int8 = "int8",
    Uint8 = "uint8",
}
