#![allow(unused_imports)]
#![allow(clippy::all)]
use super::*;
use wasm_bindgen::prelude::*;
#[cfg(web_sys_unstable_apis)]
#[wasm_bindgen]
extern "C" {
    # [wasm_bindgen (extends = :: js_sys :: Object , js_name = MLContextOptions)]
    #[derive(Debug, Clone, PartialEq, Eq)]
    #[doc = "The `MlContextOptions` dictionary."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlContextOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub type MlContextOptions;
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Get the `accelerated` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlContextOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "accelerated")]
    pub fn get_accelerated(this: &MlContextOptions) -> Option<bool>;
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Change the `accelerated` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlContextOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "accelerated")]
    pub fn set_accelerated(this: &MlContextOptions, val: bool);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlPowerPreference")]
    #[doc = "Get the `powerPreference` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlContextOptions`, `MlPowerPreference`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "powerPreference")]
    pub fn get_power_preference(this: &MlContextOptions) -> Option<MlPowerPreference>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlPowerPreference")]
    #[doc = "Change the `powerPreference` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlContextOptions`, `MlPowerPreference`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "powerPreference")]
    pub fn set_power_preference(this: &MlContextOptions, val: MlPowerPreference);
}
#[cfg(web_sys_unstable_apis)]
impl MlContextOptions {
    #[doc = "Construct a new `MlContextOptions`."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlContextOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn new() -> Self {
        #[allow(unused_mut)]
        let mut ret: Self = ::wasm_bindgen::JsCast::unchecked_into(::js_sys::Object::new());
        ret
    }
    #[cfg(web_sys_unstable_apis)]
    #[deprecated = "Use `set_accelerated()` instead."]
    pub fn accelerated(&mut self, val: bool) -> &mut Self {
        self.set_accelerated(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlPowerPreference")]
    #[deprecated = "Use `set_power_preference()` instead."]
    pub fn power_preference(&mut self, val: MlPowerPreference) -> &mut Self {
        self.set_power_preference(val);
        self
    }
}
#[cfg(web_sys_unstable_apis)]
impl Default for MlContextOptions {
    fn default() -> Self {
        Self::new()
    }
}
