#![allow(unused_imports)]
#![allow(clippy::all)]
use super::*;
use wasm_bindgen::prelude::*;
#[cfg(web_sys_unstable_apis)]
#[wasm_bindgen]
extern "C" {
    # [wasm_bindgen (extends = :: js_sys :: Object , js_name = MLConv2dOptions)]
    #[derive(Debug, Clone, PartialEq, Eq)]
    #[doc = "The `MlConv2dOptions` dictionary."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlConv2dOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub type MlConv2dOptions;
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Get the `label` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlConv2dOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "label")]
    pub fn get_label(this: &MlConv2dOptions) -> Option<::alloc::string::String>;
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Change the `label` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlConv2dOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "label")]
    pub fn set_label(this: &MlConv2dOptions, val: &str);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    #[doc = "Get the `bias` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlConv2dOptions`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "bias")]
    pub fn get_bias(this: &MlConv2dOptions) -> Option<MlOperand>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    #[doc = "Change the `bias` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlConv2dOptions`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "bias")]
    pub fn set_bias(this: &MlConv2dOptions, val: &MlOperand);
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Get the `dilations` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlConv2dOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "dilations")]
    pub fn get_dilations(this: &MlConv2dOptions) -> Option<::js_sys::Array<::js_sys::Number>>;
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Change the `dilations` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlConv2dOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "dilations")]
    pub fn set_dilations(this: &MlConv2dOptions, val: &[::js_sys::Number]);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlConv2dFilterOperandLayout")]
    #[doc = "Get the `filterLayout` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlConv2dFilterOperandLayout`, `MlConv2dOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "filterLayout")]
    pub fn get_filter_layout(this: &MlConv2dOptions) -> Option<MlConv2dFilterOperandLayout>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlConv2dFilterOperandLayout")]
    #[doc = "Change the `filterLayout` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlConv2dFilterOperandLayout`, `MlConv2dOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "filterLayout")]
    pub fn set_filter_layout(this: &MlConv2dOptions, val: MlConv2dFilterOperandLayout);
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Get the `groups` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlConv2dOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "groups")]
    pub fn get_groups(this: &MlConv2dOptions) -> Option<u32>;
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Change the `groups` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlConv2dOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "groups")]
    pub fn set_groups(this: &MlConv2dOptions, val: u32);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlInputOperandLayout")]
    #[doc = "Get the `inputLayout` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlConv2dOptions`, `MlInputOperandLayout`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "inputLayout")]
    pub fn get_input_layout(this: &MlConv2dOptions) -> Option<MlInputOperandLayout>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlInputOperandLayout")]
    #[doc = "Change the `inputLayout` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlConv2dOptions`, `MlInputOperandLayout`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "inputLayout")]
    pub fn set_input_layout(this: &MlConv2dOptions, val: MlInputOperandLayout);
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Get the `padding` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlConv2dOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "padding")]
    pub fn get_padding(this: &MlConv2dOptions) -> Option<::js_sys::Array<::js_sys::Number>>;
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Change the `padding` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlConv2dOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "padding")]
    pub fn set_padding(this: &MlConv2dOptions, val: &[::js_sys::Number]);
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Get the `strides` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlConv2dOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "strides")]
    pub fn get_strides(this: &MlConv2dOptions) -> Option<::js_sys::Array<::js_sys::Number>>;
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Change the `strides` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlConv2dOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "strides")]
    pub fn set_strides(this: &MlConv2dOptions, val: &[::js_sys::Number]);
}
#[cfg(web_sys_unstable_apis)]
impl MlConv2dOptions {
    #[doc = "Construct a new `MlConv2dOptions`."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlConv2dOptions`*"]
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
    #[cfg(feature = "MlOperand")]
    #[deprecated = "Use `set_bias()` instead."]
    pub fn bias(&mut self, val: &MlOperand) -> &mut Self {
        self.set_bias(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[deprecated = "Use `set_dilations()` instead."]
    pub fn dilations(&mut self, val: &[::js_sys::Number]) -> &mut Self {
        self.set_dilations(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlConv2dFilterOperandLayout")]
    #[deprecated = "Use `set_filter_layout()` instead."]
    pub fn filter_layout(&mut self, val: MlConv2dFilterOperandLayout) -> &mut Self {
        self.set_filter_layout(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[deprecated = "Use `set_groups()` instead."]
    pub fn groups(&mut self, val: u32) -> &mut Self {
        self.set_groups(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlInputOperandLayout")]
    #[deprecated = "Use `set_input_layout()` instead."]
    pub fn input_layout(&mut self, val: MlInputOperandLayout) -> &mut Self {
        self.set_input_layout(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[deprecated = "Use `set_padding()` instead."]
    pub fn padding(&mut self, val: &[::js_sys::Number]) -> &mut Self {
        self.set_padding(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[deprecated = "Use `set_strides()` instead."]
    pub fn strides(&mut self, val: &[::js_sys::Number]) -> &mut Self {
        self.set_strides(val);
        self
    }
}
#[cfg(web_sys_unstable_apis)]
impl Default for MlConv2dOptions {
    fn default() -> Self {
        Self::new()
    }
}
