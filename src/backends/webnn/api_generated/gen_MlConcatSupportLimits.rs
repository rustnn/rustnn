#![allow(unused_imports)]
#![allow(clippy::all)]
use super::*;
use wasm_bindgen::prelude::*;
#[cfg(web_sys_unstable_apis)]
#[wasm_bindgen]
extern "C" {
    # [wasm_bindgen (extends = :: js_sys :: Object , js_name = MLConcatSupportLimits)]
    #[derive(Debug, Clone, PartialEq, Eq)]
    #[doc = "The `MlConcatSupportLimits` dictionary."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlConcatSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub type MlConcatSupportLimits;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlTensorLimits")]
    #[doc = "Get the `inputs` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlConcatSupportLimits`, `MlTensorLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "inputs")]
    pub fn get_inputs(this: &MlConcatSupportLimits) -> Option<MlTensorLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlTensorLimits")]
    #[doc = "Change the `inputs` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlConcatSupportLimits`, `MlTensorLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "inputs")]
    pub fn set_inputs(this: &MlConcatSupportLimits, val: &MlTensorLimits);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlTensorLimits")]
    #[doc = "Get the `output` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlConcatSupportLimits`, `MlTensorLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "output")]
    pub fn get_output(this: &MlConcatSupportLimits) -> Option<MlTensorLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlTensorLimits")]
    #[doc = "Change the `output` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlConcatSupportLimits`, `MlTensorLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "output")]
    pub fn set_output(this: &MlConcatSupportLimits, val: &MlTensorLimits);
}
#[cfg(web_sys_unstable_apis)]
impl MlConcatSupportLimits {
    #[doc = "Construct a new `MlConcatSupportLimits`."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlConcatSupportLimits`*"]
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
    #[deprecated = "Use `set_inputs()` instead."]
    pub fn inputs(&mut self, val: &MlTensorLimits) -> &mut Self {
        self.set_inputs(val);
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
impl Default for MlConcatSupportLimits {
    fn default() -> Self {
        Self::new()
    }
}
