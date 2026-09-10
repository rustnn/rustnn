#![allow(unused_imports)]
#![allow(clippy::all)]
use wasm_bindgen::prelude::*;
#[cfg(web_sys_unstable_apis)]
#[wasm_bindgen]
#[doc = "The `MlConvTranspose2dFilterOperandLayout` enum."]
#[doc = ""]
#[doc = "*This API requires the following crate features to be activated: `MlConvTranspose2dFilterOperandLayout`*"]
#[doc = ""]
#[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
#[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MlConvTranspose2dFilterOperandLayout {
    Iohw = "iohw",
    Hwoi = "hwoi",
    Ohwi = "ohwi",
}
