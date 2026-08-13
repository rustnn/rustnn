#![allow(unused_imports)]
#![allow(clippy::all)]
use super::*;
use wasm_bindgen::prelude::*;
#[cfg(web_sys_unstable_apis)]
#[wasm_bindgen]
extern "C" {
    # [wasm_bindgen (extends = :: js_sys :: Object , js_name = MLLayerNormalizationOptions)]
    #[derive(Debug, Clone, PartialEq, Eq)]
    #[doc = "The `MlLayerNormalizationOptions` dictionary."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlLayerNormalizationOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub type MlLayerNormalizationOptions;
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Get the `label` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlLayerNormalizationOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "label")]
    pub fn get_label(this: &MlLayerNormalizationOptions) -> Option<::alloc::string::String>;
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Change the `label` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlLayerNormalizationOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "label")]
    pub fn set_label(this: &MlLayerNormalizationOptions, val: &str);
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Get the `axes` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlLayerNormalizationOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "axes")]
    pub fn get_axes(
        this: &MlLayerNormalizationOptions,
    ) -> Option<::js_sys::Array<::js_sys::Number>>;
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Change the `axes` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlLayerNormalizationOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "axes")]
    pub fn set_axes(this: &MlLayerNormalizationOptions, val: &[::js_sys::Number]);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    #[doc = "Get the `bias` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlLayerNormalizationOptions`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "bias")]
    pub fn get_bias(this: &MlLayerNormalizationOptions) -> Option<MlOperand>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    #[doc = "Change the `bias` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlLayerNormalizationOptions`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "bias")]
    pub fn set_bias(this: &MlLayerNormalizationOptions, val: &MlOperand);
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Get the `epsilon` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlLayerNormalizationOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "epsilon")]
    pub fn get_epsilon(this: &MlLayerNormalizationOptions) -> Option<f64>;
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Change the `epsilon` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlLayerNormalizationOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "epsilon")]
    pub fn set_epsilon(this: &MlLayerNormalizationOptions, val: f64);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    #[doc = "Get the `scale` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlLayerNormalizationOptions`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "scale")]
    pub fn get_scale(this: &MlLayerNormalizationOptions) -> Option<MlOperand>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    #[doc = "Change the `scale` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlLayerNormalizationOptions`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "scale")]
    pub fn set_scale(this: &MlLayerNormalizationOptions, val: &MlOperand);
}
#[cfg(web_sys_unstable_apis)]
impl MlLayerNormalizationOptions {
    #[doc = "Construct a new `MlLayerNormalizationOptions`."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlLayerNormalizationOptions`*"]
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
    #[cfg(feature = "MlOperand")]
    #[deprecated = "Use `set_bias()` instead."]
    pub fn bias(&mut self, val: &MlOperand) -> &mut Self {
        self.set_bias(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[deprecated = "Use `set_epsilon()` instead."]
    pub fn epsilon(&mut self, val: f64) -> &mut Self {
        self.set_epsilon(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    #[deprecated = "Use `set_scale()` instead."]
    pub fn scale(&mut self, val: &MlOperand) -> &mut Self {
        self.set_scale(val);
        self
    }
}
#[cfg(web_sys_unstable_apis)]
impl Default for MlLayerNormalizationOptions {
    fn default() -> Self {
        Self::new()
    }
}
