#![allow(unused_imports)]
#![allow(clippy::all)]
use super::*;
use wasm_bindgen::prelude::*;
#[cfg(web_sys_unstable_apis)]
#[wasm_bindgen]
extern "C" {
    # [wasm_bindgen (extends = :: js_sys :: Object , js_name = MLContextLostInfo)]
    #[derive(Debug, Clone, PartialEq, Eq)]
    #[doc = "The `MlContextLostInfo` dictionary."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlContextLostInfo`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub type MlContextLostInfo;
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Get the `message` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlContextLostInfo`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "message")]
    pub fn get_message(this: &MlContextLostInfo) -> Option<::alloc::string::String>;
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Change the `message` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlContextLostInfo`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "message")]
    pub fn set_message(this: &MlContextLostInfo, val: &str);
}
#[cfg(web_sys_unstable_apis)]
impl MlContextLostInfo {
    #[doc = "Construct a new `MlContextLostInfo`."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlContextLostInfo`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn new() -> Self {
        #[allow(unused_mut)]
        let mut ret: Self = ::wasm_bindgen::JsCast::unchecked_into(::js_sys::Object::new());
        ret
    }
    #[cfg(web_sys_unstable_apis)]
    #[deprecated = "Use `set_message()` instead."]
    pub fn message(&mut self, val: &str) -> &mut Self {
        self.set_message(val);
        self
    }
}
#[cfg(web_sys_unstable_apis)]
impl Default for MlContextLostInfo {
    fn default() -> Self {
        Self::new()
    }
}
