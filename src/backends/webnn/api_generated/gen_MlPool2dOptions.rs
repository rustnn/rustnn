#![allow(unused_imports)]
#![allow(clippy::all)]
use super::*;
use wasm_bindgen::prelude::*;
#[cfg(web_sys_unstable_apis)]
#[wasm_bindgen]
extern "C" {
    # [wasm_bindgen (extends = :: js_sys :: Object , js_name = MLPool2dOptions)]
    #[derive(Debug, Clone, PartialEq, Eq)]
    #[doc = "The `MlPool2dOptions` dictionary."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlPool2dOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub type MlPool2dOptions;
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Get the `label` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlPool2dOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "label")]
    pub fn get_label(this: &MlPool2dOptions) -> Option<::alloc::string::String>;
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Change the `label` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlPool2dOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "label")]
    pub fn set_label(this: &MlPool2dOptions, val: &str);
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Get the `dilations` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlPool2dOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "dilations")]
    pub fn get_dilations(this: &MlPool2dOptions) -> Option<::js_sys::Array<::js_sys::Number>>;
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Change the `dilations` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlPool2dOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "dilations")]
    pub fn set_dilations(this: &MlPool2dOptions, val: &[::js_sys::Number]);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlInputOperandLayout")]
    #[doc = "Get the `layout` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlInputOperandLayout`, `MlPool2dOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "layout")]
    pub fn get_layout(this: &MlPool2dOptions) -> Option<MlInputOperandLayout>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlInputOperandLayout")]
    #[doc = "Change the `layout` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlInputOperandLayout`, `MlPool2dOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "layout")]
    pub fn set_layout(this: &MlPool2dOptions, val: MlInputOperandLayout);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlRoundingType")]
    #[doc = "Get the `outputShapeRounding` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlPool2dOptions`, `MlRoundingType`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "outputShapeRounding")]
    pub fn get_output_shape_rounding(this: &MlPool2dOptions) -> Option<MlRoundingType>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlRoundingType")]
    #[doc = "Change the `outputShapeRounding` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlPool2dOptions`, `MlRoundingType`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "outputShapeRounding")]
    pub fn set_output_shape_rounding(this: &MlPool2dOptions, val: MlRoundingType);
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Get the `outputSizes` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlPool2dOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "outputSizes")]
    pub fn get_output_sizes(this: &MlPool2dOptions) -> Option<::js_sys::Array<::js_sys::Number>>;
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Change the `outputSizes` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlPool2dOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "outputSizes")]
    pub fn set_output_sizes(this: &MlPool2dOptions, val: &[::js_sys::Number]);
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Get the `padding` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlPool2dOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "padding")]
    pub fn get_padding(this: &MlPool2dOptions) -> Option<::js_sys::Array<::js_sys::Number>>;
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Change the `padding` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlPool2dOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "padding")]
    pub fn set_padding(this: &MlPool2dOptions, val: &[::js_sys::Number]);
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Get the `strides` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlPool2dOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "strides")]
    pub fn get_strides(this: &MlPool2dOptions) -> Option<::js_sys::Array<::js_sys::Number>>;
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Change the `strides` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlPool2dOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "strides")]
    pub fn set_strides(this: &MlPool2dOptions, val: &[::js_sys::Number]);
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Get the `windowDimensions` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlPool2dOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "windowDimensions")]
    pub fn get_window_dimensions(
        this: &MlPool2dOptions,
    ) -> Option<::js_sys::Array<::js_sys::Number>>;
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Change the `windowDimensions` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlPool2dOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "windowDimensions")]
    pub fn set_window_dimensions(this: &MlPool2dOptions, val: &[::js_sys::Number]);
}
#[cfg(web_sys_unstable_apis)]
impl MlPool2dOptions {
    #[doc = "Construct a new `MlPool2dOptions`."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlPool2dOptions`*"]
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
    #[deprecated = "Use `set_dilations()` instead."]
    pub fn dilations(&mut self, val: &[::js_sys::Number]) -> &mut Self {
        self.set_dilations(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlInputOperandLayout")]
    #[deprecated = "Use `set_layout()` instead."]
    pub fn layout(&mut self, val: MlInputOperandLayout) -> &mut Self {
        self.set_layout(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlRoundingType")]
    #[deprecated = "Use `set_output_shape_rounding()` instead."]
    pub fn output_shape_rounding(&mut self, val: MlRoundingType) -> &mut Self {
        self.set_output_shape_rounding(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[deprecated = "Use `set_output_sizes()` instead."]
    pub fn output_sizes(&mut self, val: &[::js_sys::Number]) -> &mut Self {
        self.set_output_sizes(val);
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
    #[cfg(web_sys_unstable_apis)]
    #[deprecated = "Use `set_window_dimensions()` instead."]
    pub fn window_dimensions(&mut self, val: &[::js_sys::Number]) -> &mut Self {
        self.set_window_dimensions(val);
        self
    }
}
#[cfg(web_sys_unstable_apis)]
impl Default for MlPool2dOptions {
    fn default() -> Self {
        Self::new()
    }
}
