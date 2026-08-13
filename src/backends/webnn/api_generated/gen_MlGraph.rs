#![allow(unused_imports)]
#![allow(clippy::all)]
use super::*;
use wasm_bindgen::prelude::*;
#[cfg(web_sys_unstable_apis)]
#[wasm_bindgen]
extern "C" {
    # [wasm_bindgen (extends = :: js_sys :: Object , js_name = MLGraph , typescript_type = "MLGraph")]
    #[derive(Debug, Clone, PartialEq, Eq)]
    #[doc = "The `MlGraph` class."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraph)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraph`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub type MlGraph;
    #[cfg(web_sys_unstable_apis)]
    # [wasm_bindgen (method , structural , js_class = "MLGraph" , js_name = destroy)]
    #[doc = "The `destroy()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraph/destroy)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraph`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn destroy(this: &MlGraph);
}
