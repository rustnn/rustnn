#![allow(unused_imports)]
#![allow(clippy::all)]
use super::*;
use wasm_bindgen::prelude::*;
#[cfg(web_sys_unstable_apis)]
#[wasm_bindgen]
extern "C" {
    # [wasm_bindgen (extends = :: js_sys :: Object , js_name = MLOperandDescriptor)]
    #[derive(Debug, Clone, PartialEq, Eq)]
    #[doc = "The `MlOperandDescriptor` dictionary."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOperandDescriptor`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub type MlOperandDescriptor;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperandDataType")]
    #[doc = "Get the `dataType` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOperandDataType`, `MlOperandDescriptor`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "dataType")]
    pub fn get_data_type(this: &MlOperandDescriptor) -> MlOperandDataType;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperandDataType")]
    #[doc = "Change the `dataType` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOperandDataType`, `MlOperandDescriptor`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "dataType")]
    pub fn set_data_type(this: &MlOperandDescriptor, val: MlOperandDataType);
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Get the `shape` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOperandDescriptor`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "shape")]
    pub fn get_shape(this: &MlOperandDescriptor) -> ::js_sys::Array<::js_sys::Number>;
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Change the `shape` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOperandDescriptor`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "shape")]
    pub fn set_shape(this: &MlOperandDescriptor, val: &[::js_sys::Number]);
}
#[cfg(web_sys_unstable_apis)]
impl MlOperandDescriptor {
    #[cfg(feature = "MlOperandDataType")]
    #[doc = "Construct a new `MlOperandDescriptor`."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOperandDataType`, `MlOperandDescriptor`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn new(data_type: MlOperandDataType, shape: &[::js_sys::Number]) -> Self {
        #[allow(unused_mut)]
        let mut ret: Self = ::wasm_bindgen::JsCast::unchecked_into(::js_sys::Object::new());
        ret.set_data_type(data_type);
        ret.set_shape(shape);
        ret
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperandDataType")]
    #[deprecated = "Use `set_data_type()` instead."]
    pub fn data_type(&mut self, val: MlOperandDataType) -> &mut Self {
        self.set_data_type(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[deprecated = "Use `set_shape()` instead."]
    pub fn shape(&mut self, val: &[::js_sys::Number]) -> &mut Self {
        self.set_shape(val);
        self
    }
}
