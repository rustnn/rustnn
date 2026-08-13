#![allow(unused_imports)]
#![allow(clippy::all)]
use super::*;
use wasm_bindgen::prelude::*;
#[cfg(web_sys_unstable_apis)]
#[wasm_bindgen]
extern "C" {
    # [wasm_bindgen (extends = :: js_sys :: Object , js_name = MLRankRange)]
    #[derive(Debug, Clone, PartialEq, Eq)]
    #[doc = "The `MlRankRange` dictionary."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlRankRange`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub type MlRankRange;
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Get the `max` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlRankRange`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "max")]
    pub fn get_max(this: &MlRankRange) -> Option<u32>;
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Change the `max` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlRankRange`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "max")]
    pub fn set_max(this: &MlRankRange, val: u32);
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Get the `min` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlRankRange`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "min")]
    pub fn get_min(this: &MlRankRange) -> Option<u32>;
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Change the `min` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlRankRange`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "min")]
    pub fn set_min(this: &MlRankRange, val: u32);
}
#[cfg(web_sys_unstable_apis)]
impl MlRankRange {
    #[doc = "Construct a new `MlRankRange`."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlRankRange`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn new() -> Self {
        #[allow(unused_mut)]
        let mut ret: Self = ::wasm_bindgen::JsCast::unchecked_into(::js_sys::Object::new());
        ret
    }
    #[cfg(web_sys_unstable_apis)]
    #[deprecated = "Use `set_max()` instead."]
    pub fn max(&mut self, val: u32) -> &mut Self {
        self.set_max(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[deprecated = "Use `set_min()` instead."]
    pub fn min(&mut self, val: u32) -> &mut Self {
        self.set_min(val);
        self
    }
}
#[cfg(web_sys_unstable_apis)]
impl Default for MlRankRange {
    fn default() -> Self {
        Self::new()
    }
}
