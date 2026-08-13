#![allow(unused_imports)]
#![allow(clippy::all)]
use super::*;
use wasm_bindgen::prelude::*;
#[cfg(web_sys_unstable_apis)]
#[wasm_bindgen]
extern "C" {
    # [wasm_bindgen (extends = :: js_sys :: Object , js_name = MLReduceOptions)]
    #[derive(Debug, Clone, PartialEq, Eq)]
    #[doc = "The `MlReduceOptions` dictionary."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlReduceOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub type MlReduceOptions;
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Get the `label` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlReduceOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "label")]
    pub fn get_label(this: &MlReduceOptions) -> Option<::alloc::string::String>;
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Change the `label` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlReduceOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "label")]
    pub fn set_label(this: &MlReduceOptions, val: &str);
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Get the `axes` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlReduceOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "axes")]
    pub fn get_axes(this: &MlReduceOptions) -> Option<::js_sys::Array<::js_sys::Number>>;
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Change the `axes` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlReduceOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "axes")]
    pub fn set_axes(this: &MlReduceOptions, val: &[::js_sys::Number]);
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Get the `keepDimensions` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlReduceOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "keepDimensions")]
    pub fn get_keep_dimensions(this: &MlReduceOptions) -> Option<bool>;
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Change the `keepDimensions` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlReduceOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "keepDimensions")]
    pub fn set_keep_dimensions(this: &MlReduceOptions, val: bool);
}
#[cfg(web_sys_unstable_apis)]
impl MlReduceOptions {
    #[doc = "Construct a new `MlReduceOptions`."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlReduceOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn new() -> Self {
        #[allow(unused_mut)]
        let mut ret: Self = ::wasm_bindgen::JsCast::unchecked_into(::js_sys::Object::new());
        ret
    }
    #[cfg(web_sys_unstable_apis)]
    #[deprecated = "Use `set_label()` instead."]
    pub fn label(&mut self, val: &str) -> &mut Self {
        self.set_label(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[deprecated = "Use `set_axes()` instead."]
    pub fn axes(&mut self, val: &[::js_sys::Number]) -> &mut Self {
        self.set_axes(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[deprecated = "Use `set_keep_dimensions()` instead."]
    pub fn keep_dimensions(&mut self, val: bool) -> &mut Self {
        self.set_keep_dimensions(val);
        self
    }
}
#[cfg(web_sys_unstable_apis)]
impl Default for MlReduceOptions {
    fn default() -> Self {
        Self::new()
    }
}
