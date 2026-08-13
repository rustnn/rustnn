#![allow(unused_imports)]
#![allow(clippy::all)]
use super::*;
use wasm_bindgen::prelude::*;
#[cfg(web_sys_unstable_apis)]
#[wasm_bindgen]
extern "C" {
    # [wasm_bindgen (extends = :: js_sys :: Object , js_name = MLLogicalNotSupportLimits)]
    #[derive(Debug, Clone, PartialEq, Eq)]
    #[doc = "The `MlLogicalNotSupportLimits` dictionary."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlLogicalNotSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub type MlLogicalNotSupportLimits;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlTensorLimits")]
    #[doc = "Get the `a` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlLogicalNotSupportLimits`, `MlTensorLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "a")]
    pub fn get_a(this: &MlLogicalNotSupportLimits) -> Option<MlTensorLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlTensorLimits")]
    #[doc = "Change the `a` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlLogicalNotSupportLimits`, `MlTensorLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "a")]
    pub fn set_a(this: &MlLogicalNotSupportLimits, val: &MlTensorLimits);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlTensorLimits")]
    #[doc = "Get the `output` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlLogicalNotSupportLimits`, `MlTensorLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "output")]
    pub fn get_output(this: &MlLogicalNotSupportLimits) -> Option<MlTensorLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlTensorLimits")]
    #[doc = "Change the `output` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlLogicalNotSupportLimits`, `MlTensorLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "output")]
    pub fn set_output(this: &MlLogicalNotSupportLimits, val: &MlTensorLimits);
}
#[cfg(web_sys_unstable_apis)]
impl MlLogicalNotSupportLimits {
    #[doc = "Construct a new `MlLogicalNotSupportLimits`."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlLogicalNotSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn new() -> Self {
        #[allow(unused_mut)]
        let mut ret: Self = ::wasm_bindgen::JsCast::unchecked_into(::js_sys::Object::new());
        ret
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlTensorLimits")]
    #[deprecated = "Use `set_a()` instead."]
    pub fn a(&mut self, val: &MlTensorLimits) -> &mut Self {
        self.set_a(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlTensorLimits")]
    #[deprecated = "Use `set_output()` instead."]
    pub fn output(&mut self, val: &MlTensorLimits) -> &mut Self {
        self.set_output(val);
        self
    }
}
#[cfg(web_sys_unstable_apis)]
impl Default for MlLogicalNotSupportLimits {
    fn default() -> Self {
        Self::new()
    }
}
