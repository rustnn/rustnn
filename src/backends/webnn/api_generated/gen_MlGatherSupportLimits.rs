#![allow(unused_imports)]
#![allow(clippy::all)]
use super::*;
use wasm_bindgen::prelude::*;
#[cfg(web_sys_unstable_apis)]
#[wasm_bindgen]
extern "C" {
    # [wasm_bindgen (extends = :: js_sys :: Object , js_name = MLGatherSupportLimits)]
    #[derive(Debug, Clone, PartialEq, Eq)]
    #[doc = "The `MlGatherSupportLimits` dictionary."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGatherSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub type MlGatherSupportLimits;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlTensorLimits")]
    #[doc = "Get the `indices` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGatherSupportLimits`, `MlTensorLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "indices")]
    pub fn get_indices(this: &MlGatherSupportLimits) -> Option<MlTensorLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlTensorLimits")]
    #[doc = "Change the `indices` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGatherSupportLimits`, `MlTensorLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "indices")]
    pub fn set_indices(this: &MlGatherSupportLimits, val: &MlTensorLimits);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlTensorLimits")]
    #[doc = "Get the `input` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGatherSupportLimits`, `MlTensorLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "input")]
    pub fn get_input(this: &MlGatherSupportLimits) -> Option<MlTensorLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlTensorLimits")]
    #[doc = "Change the `input` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGatherSupportLimits`, `MlTensorLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "input")]
    pub fn set_input(this: &MlGatherSupportLimits, val: &MlTensorLimits);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlTensorLimits")]
    #[doc = "Get the `output` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGatherSupportLimits`, `MlTensorLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "output")]
    pub fn get_output(this: &MlGatherSupportLimits) -> Option<MlTensorLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlTensorLimits")]
    #[doc = "Change the `output` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGatherSupportLimits`, `MlTensorLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "output")]
    pub fn set_output(this: &MlGatherSupportLimits, val: &MlTensorLimits);
}
#[cfg(web_sys_unstable_apis)]
impl MlGatherSupportLimits {
    #[doc = "Construct a new `MlGatherSupportLimits`."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGatherSupportLimits`*"]
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
    #[deprecated = "Use `set_indices()` instead."]
    pub fn indices(&mut self, val: &MlTensorLimits) -> &mut Self {
        self.set_indices(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlTensorLimits")]
    #[deprecated = "Use `set_input()` instead."]
    pub fn input(&mut self, val: &MlTensorLimits) -> &mut Self {
        self.set_input(val);
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
impl Default for MlGatherSupportLimits {
    fn default() -> Self {
        Self::new()
    }
}
