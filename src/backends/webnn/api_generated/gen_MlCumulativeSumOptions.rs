#![allow(unused_imports)]
#![allow(clippy::all)]
use super::*;
use wasm_bindgen::prelude::*;
#[cfg(web_sys_unstable_apis)]
#[wasm_bindgen]
extern "C" {
    # [wasm_bindgen (extends = :: js_sys :: Object , js_name = MLCumulativeSumOptions)]
    #[derive(Debug, Clone, PartialEq, Eq)]
    #[doc = "The `MlCumulativeSumOptions` dictionary."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlCumulativeSumOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub type MlCumulativeSumOptions;
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Get the `label` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlCumulativeSumOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "label")]
    pub fn get_label(this: &MlCumulativeSumOptions) -> Option<::alloc::string::String>;
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Change the `label` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlCumulativeSumOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "label")]
    pub fn set_label(this: &MlCumulativeSumOptions, val: &str);
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Get the `exclusive` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlCumulativeSumOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "exclusive")]
    pub fn get_exclusive(this: &MlCumulativeSumOptions) -> Option<bool>;
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Change the `exclusive` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlCumulativeSumOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "exclusive")]
    pub fn set_exclusive(this: &MlCumulativeSumOptions, val: bool);
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Get the `reversed` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlCumulativeSumOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "reversed")]
    pub fn get_reversed(this: &MlCumulativeSumOptions) -> Option<bool>;
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Change the `reversed` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlCumulativeSumOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "reversed")]
    pub fn set_reversed(this: &MlCumulativeSumOptions, val: bool);
}
#[cfg(web_sys_unstable_apis)]
impl MlCumulativeSumOptions {
    #[doc = "Construct a new `MlCumulativeSumOptions`."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlCumulativeSumOptions`*"]
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
    #[deprecated = "Use `set_exclusive()` instead."]
    pub fn exclusive(&mut self, val: bool) -> &mut Self {
        self.set_exclusive(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[deprecated = "Use `set_reversed()` instead."]
    pub fn reversed(&mut self, val: bool) -> &mut Self {
        self.set_reversed(val);
        self
    }
}
#[cfg(web_sys_unstable_apis)]
impl Default for MlCumulativeSumOptions {
    fn default() -> Self {
        Self::new()
    }
}
