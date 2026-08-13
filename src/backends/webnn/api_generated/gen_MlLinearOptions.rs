#![allow(unused_imports)]
#![allow(clippy::all)]
use super::*;
use wasm_bindgen::prelude::*;
#[cfg(web_sys_unstable_apis)]
#[wasm_bindgen]
extern "C" {
    # [wasm_bindgen (extends = :: js_sys :: Object , js_name = MLLinearOptions)]
    #[derive(Debug, Clone, PartialEq, Eq)]
    #[doc = "The `MlLinearOptions` dictionary."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlLinearOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub type MlLinearOptions;
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Get the `label` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlLinearOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "label")]
    pub fn get_label(this: &MlLinearOptions) -> Option<::alloc::string::String>;
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Change the `label` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlLinearOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "label")]
    pub fn set_label(this: &MlLinearOptions, val: &str);
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Get the `alpha` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlLinearOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "alpha")]
    pub fn get_alpha(this: &MlLinearOptions) -> Option<f64>;
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Change the `alpha` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlLinearOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "alpha")]
    pub fn set_alpha(this: &MlLinearOptions, val: f64);
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Get the `beta` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlLinearOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "beta")]
    pub fn get_beta(this: &MlLinearOptions) -> Option<f64>;
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Change the `beta` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlLinearOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "beta")]
    pub fn set_beta(this: &MlLinearOptions, val: f64);
}
#[cfg(web_sys_unstable_apis)]
impl MlLinearOptions {
    #[doc = "Construct a new `MlLinearOptions`."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlLinearOptions`*"]
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
    #[deprecated = "Use `set_alpha()` instead."]
    pub fn alpha(&mut self, val: f64) -> &mut Self {
        self.set_alpha(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[deprecated = "Use `set_beta()` instead."]
    pub fn beta(&mut self, val: f64) -> &mut Self {
        self.set_beta(val);
        self
    }
}
#[cfg(web_sys_unstable_apis)]
impl Default for MlLinearOptions {
    fn default() -> Self {
        Self::new()
    }
}
