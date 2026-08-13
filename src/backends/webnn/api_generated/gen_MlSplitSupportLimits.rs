#![allow(unused_imports)]
#![allow(clippy::all)]
use super::*;
use wasm_bindgen::prelude::*;
#[cfg(web_sys_unstable_apis)]
#[wasm_bindgen]
extern "C" {
    # [wasm_bindgen (extends = :: js_sys :: Object , js_name = MLSplitSupportLimits)]
    #[derive(Debug, Clone, PartialEq, Eq)]
    #[doc = "The `MlSplitSupportLimits` dictionary."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlSplitSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub type MlSplitSupportLimits;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlTensorLimits")]
    #[doc = "Get the `input` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlSplitSupportLimits`, `MlTensorLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "input")]
    pub fn get_input(this: &MlSplitSupportLimits) -> Option<MlTensorLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlTensorLimits")]
    #[doc = "Change the `input` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlSplitSupportLimits`, `MlTensorLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "input")]
    pub fn set_input(this: &MlSplitSupportLimits, val: &MlTensorLimits);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlTensorLimits")]
    #[doc = "Get the `outputs` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlSplitSupportLimits`, `MlTensorLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "outputs")]
    pub fn get_outputs(this: &MlSplitSupportLimits) -> Option<MlTensorLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlTensorLimits")]
    #[doc = "Change the `outputs` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlSplitSupportLimits`, `MlTensorLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "outputs")]
    pub fn set_outputs(this: &MlSplitSupportLimits, val: &MlTensorLimits);
}
#[cfg(web_sys_unstable_apis)]
impl MlSplitSupportLimits {
    #[doc = "Construct a new `MlSplitSupportLimits`."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlSplitSupportLimits`*"]
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
    #[deprecated = "Use `set_input()` instead."]
    pub fn input(&mut self, val: &MlTensorLimits) -> &mut Self {
        self.set_input(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlTensorLimits")]
    #[deprecated = "Use `set_outputs()` instead."]
    pub fn outputs(&mut self, val: &MlTensorLimits) -> &mut Self {
        self.set_outputs(val);
        self
    }
}
#[cfg(web_sys_unstable_apis)]
impl Default for MlSplitSupportLimits {
    fn default() -> Self {
        Self::new()
    }
}
