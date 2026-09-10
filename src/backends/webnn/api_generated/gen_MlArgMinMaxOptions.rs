#![allow(unused_imports)]
#![allow(clippy::all)]
use super::*;
use wasm_bindgen::prelude::*;
#[cfg(web_sys_unstable_apis)]
#[wasm_bindgen]
extern "C" {
    # [wasm_bindgen (extends = :: js_sys :: Object , js_name = MLArgMinMaxOptions)]
    #[derive(Debug, Clone, PartialEq, Eq)]
    #[doc = "The `MlArgMinMaxOptions` dictionary."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlArgMinMaxOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub type MlArgMinMaxOptions;
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Get the `label` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlArgMinMaxOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "label")]
    pub fn get_label(this: &MlArgMinMaxOptions) -> Option<::alloc::string::String>;
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Change the `label` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlArgMinMaxOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "label")]
    pub fn set_label(this: &MlArgMinMaxOptions, val: &str);
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Get the `keepDimensions` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlArgMinMaxOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "keepDimensions")]
    pub fn get_keep_dimensions(this: &MlArgMinMaxOptions) -> Option<bool>;
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Change the `keepDimensions` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlArgMinMaxOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "keepDimensions")]
    pub fn set_keep_dimensions(this: &MlArgMinMaxOptions, val: bool);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperandDataType")]
    #[doc = "Get the `outputDataType` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlArgMinMaxOptions`, `MlOperandDataType`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "outputDataType")]
    pub fn get_output_data_type(this: &MlArgMinMaxOptions) -> Option<MlOperandDataType>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperandDataType")]
    #[doc = "Change the `outputDataType` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlArgMinMaxOptions`, `MlOperandDataType`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "outputDataType")]
    pub fn set_output_data_type(this: &MlArgMinMaxOptions, val: MlOperandDataType);
}
#[cfg(web_sys_unstable_apis)]
impl MlArgMinMaxOptions {
    #[doc = "Construct a new `MlArgMinMaxOptions`."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlArgMinMaxOptions`*"]
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
    #[deprecated = "Use `set_keep_dimensions()` instead."]
    pub fn keep_dimensions(&mut self, val: bool) -> &mut Self {
        self.set_keep_dimensions(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperandDataType")]
    #[deprecated = "Use `set_output_data_type()` instead."]
    pub fn output_data_type(&mut self, val: MlOperandDataType) -> &mut Self {
        self.set_output_data_type(val);
        self
    }
}
#[cfg(web_sys_unstable_apis)]
impl Default for MlArgMinMaxOptions {
    fn default() -> Self {
        Self::new()
    }
}
