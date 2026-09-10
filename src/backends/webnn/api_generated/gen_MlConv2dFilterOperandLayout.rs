#![allow(unused_imports)]
#![allow(clippy::all)]
use wasm_bindgen::prelude::*;
#[cfg(web_sys_unstable_apis)]
#[wasm_bindgen]
#[doc = "The `MlConv2dFilterOperandLayout` enum."]
#[doc = ""]
#[doc = "*This API requires the following crate features to be activated: `MlConv2dFilterOperandLayout`*"]
#[doc = ""]
#[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
#[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MlConv2dFilterOperandLayout {
    Oihw = "oihw",
    Hwio = "hwio",
    Ohwi = "ohwi",
    Ihwo = "ihwo",
}
