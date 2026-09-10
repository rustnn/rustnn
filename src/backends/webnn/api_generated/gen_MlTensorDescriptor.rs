#![allow(unused_imports)]
#![allow(clippy::all)]
use super::*;
use wasm_bindgen::prelude::*;
#[cfg(web_sys_unstable_apis)]
#[wasm_bindgen]
extern "C" {
    # [wasm_bindgen (extends = :: js_sys :: Object , js_name = MLTensorDescriptor)]
    #[derive(Debug, Clone, PartialEq, Eq)]
    #[doc = "The `MlTensorDescriptor` dictionary."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlTensorDescriptor`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub type MlTensorDescriptor;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperandDataType")]
    #[doc = "Get the `dataType` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOperandDataType`, `MlTensorDescriptor`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "dataType")]
    pub fn get_data_type(this: &MlTensorDescriptor) -> MlOperandDataType;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperandDataType")]
    #[doc = "Change the `dataType` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOperandDataType`, `MlTensorDescriptor`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "dataType")]
    pub fn set_data_type(this: &MlTensorDescriptor, val: MlOperandDataType);
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Get the `shape` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlTensorDescriptor`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "shape")]
    pub fn get_shape(this: &MlTensorDescriptor) -> ::js_sys::Array<::js_sys::Number>;
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Change the `shape` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlTensorDescriptor`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "shape")]
    pub fn set_shape(this: &MlTensorDescriptor, val: &[::js_sys::Number]);
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Get the `readable` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlTensorDescriptor`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "readable")]
    pub fn get_readable(this: &MlTensorDescriptor) -> Option<bool>;
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Change the `readable` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlTensorDescriptor`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "readable")]
    pub fn set_readable(this: &MlTensorDescriptor, val: bool);
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Get the `writable` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlTensorDescriptor`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "writable")]
    pub fn get_writable(this: &MlTensorDescriptor) -> Option<bool>;
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Change the `writable` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlTensorDescriptor`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "writable")]
    pub fn set_writable(this: &MlTensorDescriptor, val: bool);
}
#[cfg(web_sys_unstable_apis)]
impl MlTensorDescriptor {
    #[cfg(feature = "MlOperandDataType")]
    #[doc = "Construct a new `MlTensorDescriptor`."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOperandDataType`, `MlTensorDescriptor`*"]
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
    #[cfg(web_sys_unstable_apis)]
    #[deprecated = "Use `set_readable()` instead."]
    pub fn readable(&mut self, val: bool) -> &mut Self {
        self.set_readable(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[deprecated = "Use `set_writable()` instead."]
    pub fn writable(&mut self, val: bool) -> &mut Self {
        self.set_writable(val);
        self
    }
}
