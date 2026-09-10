#![allow(unused_imports)]
#![allow(clippy::all)]
use super::*;
use wasm_bindgen::prelude::*;
#[cfg(web_sys_unstable_apis)]
#[wasm_bindgen]
extern "C" {
    # [wasm_bindgen (extends = :: js_sys :: Object , js_name = MLTensorLimits)]
    #[derive(Debug, Clone, PartialEq, Eq)]
    #[doc = "The `MlTensorLimits` dictionary."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlTensorLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub type MlTensorLimits;
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Get the `dataTypes` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlTensorLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "dataTypes")]
    pub fn get_data_types(this: &MlTensorLimits) -> Option<::js_sys::Array<::js_sys::JsString>>;
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Change the `dataTypes` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlTensorLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "dataTypes")]
    pub fn set_data_types(this: &MlTensorLimits, val: &[::js_sys::JsString]);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlRankRange")]
    #[doc = "Get the `rankRange` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlRankRange`, `MlTensorLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "rankRange")]
    pub fn get_rank_range(this: &MlTensorLimits) -> Option<MlRankRange>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlRankRange")]
    #[doc = "Change the `rankRange` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlRankRange`, `MlTensorLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "rankRange")]
    pub fn set_rank_range(this: &MlTensorLimits, val: &MlRankRange);
}
#[cfg(web_sys_unstable_apis)]
impl MlTensorLimits {
    #[doc = "Construct a new `MlTensorLimits`."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlTensorLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn new() -> Self {
        #[allow(unused_mut)]
        let mut ret: Self = ::wasm_bindgen::JsCast::unchecked_into(::js_sys::Object::new());
        ret
    }
    #[cfg(web_sys_unstable_apis)]
    #[deprecated = "Use `set_data_types()` instead."]
    pub fn data_types(&mut self, val: &[::js_sys::JsString]) -> &mut Self {
        self.set_data_types(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlRankRange")]
    #[deprecated = "Use `set_rank_range()` instead."]
    pub fn rank_range(&mut self, val: &MlRankRange) -> &mut Self {
        self.set_rank_range(val);
        self
    }
}
#[cfg(web_sys_unstable_apis)]
impl Default for MlTensorLimits {
    fn default() -> Self {
        Self::new()
    }
}
