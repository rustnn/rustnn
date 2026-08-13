#![allow(unused_imports)]
#![allow(clippy::all)]
use super::*;
use wasm_bindgen::prelude::*;
#[cfg(web_sys_unstable_apis)]
#[wasm_bindgen]
extern "C" {
    # [wasm_bindgen (extends = :: js_sys :: Object , js_name = MLGruCellOptions)]
    #[derive(Debug, Clone, PartialEq, Eq)]
    #[doc = "The `MlGruCellOptions` dictionary."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGruCellOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub type MlGruCellOptions;
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Get the `label` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGruCellOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "label")]
    pub fn get_label(this: &MlGruCellOptions) -> Option<::alloc::string::String>;
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Change the `label` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGruCellOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "label")]
    pub fn set_label(this: &MlGruCellOptions, val: &str);
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Get the `activations` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGruCellOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "activations")]
    pub fn get_activations(this: &MlGruCellOptions) -> Option<::js_sys::Array<::js_sys::JsString>>;
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Change the `activations` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGruCellOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "activations")]
    pub fn set_activations(this: &MlGruCellOptions, val: &[::js_sys::JsString]);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    #[doc = "Get the `bias` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGruCellOptions`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "bias")]
    pub fn get_bias(this: &MlGruCellOptions) -> Option<MlOperand>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    #[doc = "Change the `bias` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGruCellOptions`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "bias")]
    pub fn set_bias(this: &MlGruCellOptions, val: &MlOperand);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlGruWeightLayout")]
    #[doc = "Get the `layout` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGruCellOptions`, `MlGruWeightLayout`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "layout")]
    pub fn get_layout(this: &MlGruCellOptions) -> Option<MlGruWeightLayout>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlGruWeightLayout")]
    #[doc = "Change the `layout` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGruCellOptions`, `MlGruWeightLayout`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "layout")]
    pub fn set_layout(this: &MlGruCellOptions, val: MlGruWeightLayout);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    #[doc = "Get the `recurrentBias` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGruCellOptions`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "recurrentBias")]
    pub fn get_recurrent_bias(this: &MlGruCellOptions) -> Option<MlOperand>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    #[doc = "Change the `recurrentBias` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGruCellOptions`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "recurrentBias")]
    pub fn set_recurrent_bias(this: &MlGruCellOptions, val: &MlOperand);
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Get the `resetAfter` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGruCellOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "resetAfter")]
    pub fn get_reset_after(this: &MlGruCellOptions) -> Option<bool>;
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Change the `resetAfter` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGruCellOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "resetAfter")]
    pub fn set_reset_after(this: &MlGruCellOptions, val: bool);
}
#[cfg(web_sys_unstable_apis)]
impl MlGruCellOptions {
    #[doc = "Construct a new `MlGruCellOptions`."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGruCellOptions`*"]
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
    #[deprecated = "Use `set_activations()` instead."]
    pub fn activations(&mut self, val: &[::js_sys::JsString]) -> &mut Self {
        self.set_activations(val);
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
    #[cfg(feature = "MlGruWeightLayout")]
    #[deprecated = "Use `set_layout()` instead."]
    pub fn layout(&mut self, val: MlGruWeightLayout) -> &mut Self {
        self.set_layout(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    #[deprecated = "Use `set_recurrent_bias()` instead."]
    pub fn recurrent_bias(&mut self, val: &MlOperand) -> &mut Self {
        self.set_recurrent_bias(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[deprecated = "Use `set_reset_after()` instead."]
    pub fn reset_after(&mut self, val: bool) -> &mut Self {
        self.set_reset_after(val);
        self
    }
}
#[cfg(web_sys_unstable_apis)]
impl Default for MlGruCellOptions {
    fn default() -> Self {
        Self::new()
    }
}
