#![allow(unused_imports)]
#![allow(clippy::all)]
use super::*;
use wasm_bindgen::prelude::*;
#[cfg(web_sys_unstable_apis)]
#[wasm_bindgen]
extern "C" {
    # [wasm_bindgen (extends = :: js_sys :: Object , js_name = MLGemmOptions)]
    #[derive(Debug, Clone, PartialEq, Eq)]
    #[doc = "The `MlGemmOptions` dictionary."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGemmOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub type MlGemmOptions;
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Get the `label` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGemmOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "label")]
    pub fn get_label(this: &MlGemmOptions) -> Option<::alloc::string::String>;
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Change the `label` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGemmOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "label")]
    pub fn set_label(this: &MlGemmOptions, val: &str);
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Get the `aTranspose` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGemmOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "aTranspose")]
    pub fn get_a_transpose(this: &MlGemmOptions) -> Option<bool>;
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Change the `aTranspose` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGemmOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "aTranspose")]
    pub fn set_a_transpose(this: &MlGemmOptions, val: bool);
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Get the `alpha` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGemmOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "alpha")]
    pub fn get_alpha(this: &MlGemmOptions) -> Option<f64>;
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Change the `alpha` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGemmOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "alpha")]
    pub fn set_alpha(this: &MlGemmOptions, val: f64);
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Get the `bTranspose` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGemmOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "bTranspose")]
    pub fn get_b_transpose(this: &MlGemmOptions) -> Option<bool>;
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Change the `bTranspose` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGemmOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "bTranspose")]
    pub fn set_b_transpose(this: &MlGemmOptions, val: bool);
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Get the `beta` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGemmOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "beta")]
    pub fn get_beta(this: &MlGemmOptions) -> Option<f64>;
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Change the `beta` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGemmOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "beta")]
    pub fn set_beta(this: &MlGemmOptions, val: f64);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    #[doc = "Get the `c` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGemmOptions`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "c")]
    pub fn get_c(this: &MlGemmOptions) -> Option<MlOperand>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    #[doc = "Change the `c` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGemmOptions`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "c")]
    pub fn set_c(this: &MlGemmOptions, val: &MlOperand);
}
#[cfg(web_sys_unstable_apis)]
impl MlGemmOptions {
    #[doc = "Construct a new `MlGemmOptions`."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGemmOptions`*"]
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
    #[deprecated = "Use `set_a_transpose()` instead."]
    pub fn a_transpose(&mut self, val: bool) -> &mut Self {
        self.set_a_transpose(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[deprecated = "Use `set_alpha()` instead."]
    pub fn alpha(&mut self, val: f64) -> &mut Self {
        self.set_alpha(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[deprecated = "Use `set_b_transpose()` instead."]
    pub fn b_transpose(&mut self, val: bool) -> &mut Self {
        self.set_b_transpose(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[deprecated = "Use `set_beta()` instead."]
    pub fn beta(&mut self, val: f64) -> &mut Self {
        self.set_beta(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    #[deprecated = "Use `set_c()` instead."]
    pub fn c(&mut self, val: &MlOperand) -> &mut Self {
        self.set_c(val);
        self
    }
}
#[cfg(web_sys_unstable_apis)]
impl Default for MlGemmOptions {
    fn default() -> Self {
        Self::new()
    }
}
