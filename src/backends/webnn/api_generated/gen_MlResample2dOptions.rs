#![allow(unused_imports)]
#![allow(clippy::all)]
use super::*;
use wasm_bindgen::prelude::*;
#[cfg(web_sys_unstable_apis)]
#[wasm_bindgen]
extern "C" {
    # [wasm_bindgen (extends = :: js_sys :: Object , js_name = MLResample2dOptions)]
    #[derive(Debug, Clone, PartialEq, Eq)]
    #[doc = "The `MlResample2dOptions` dictionary."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlResample2dOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub type MlResample2dOptions;
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Get the `label` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlResample2dOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "label")]
    pub fn get_label(this: &MlResample2dOptions) -> Option<::alloc::string::String>;
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Change the `label` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlResample2dOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "label")]
    pub fn set_label(this: &MlResample2dOptions, val: &str);
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Get the `axes` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlResample2dOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "axes")]
    pub fn get_axes(this: &MlResample2dOptions) -> Option<::js_sys::Array<::js_sys::Number>>;
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Change the `axes` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlResample2dOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "axes")]
    pub fn set_axes(this: &MlResample2dOptions, val: &[::js_sys::Number]);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlInterpolationMode")]
    #[doc = "Get the `mode` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlInterpolationMode`, `MlResample2dOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "mode")]
    pub fn get_mode(this: &MlResample2dOptions) -> Option<MlInterpolationMode>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlInterpolationMode")]
    #[doc = "Change the `mode` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlInterpolationMode`, `MlResample2dOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "mode")]
    pub fn set_mode(this: &MlResample2dOptions, val: MlInterpolationMode);
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Get the `scales` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlResample2dOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "scales")]
    pub fn get_scales(this: &MlResample2dOptions) -> Option<::js_sys::Array<::js_sys::Number>>;
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Change the `scales` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlResample2dOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "scales")]
    pub fn set_scales(this: &MlResample2dOptions, val: &[::js_sys::Number]);
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Get the `sizes` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlResample2dOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "sizes")]
    pub fn get_sizes(this: &MlResample2dOptions) -> Option<::js_sys::Array<::js_sys::Number>>;
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Change the `sizes` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlResample2dOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "sizes")]
    pub fn set_sizes(this: &MlResample2dOptions, val: &[::js_sys::Number]);
}
#[cfg(web_sys_unstable_apis)]
impl MlResample2dOptions {
    #[doc = "Construct a new `MlResample2dOptions`."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlResample2dOptions`*"]
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
    #[cfg(feature = "MlInterpolationMode")]
    #[deprecated = "Use `set_mode()` instead."]
    pub fn mode(&mut self, val: MlInterpolationMode) -> &mut Self {
        self.set_mode(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[deprecated = "Use `set_scales()` instead."]
    pub fn scales(&mut self, val: &[::js_sys::Number]) -> &mut Self {
        self.set_scales(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[deprecated = "Use `set_sizes()` instead."]
    pub fn sizes(&mut self, val: &[::js_sys::Number]) -> &mut Self {
        self.set_sizes(val);
        self
    }
}
#[cfg(web_sys_unstable_apis)]
impl Default for MlResample2dOptions {
    fn default() -> Self {
        Self::new()
    }
}
