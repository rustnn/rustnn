#![allow(unused_imports)]
#![allow(clippy::all)]
use super::*;
use wasm_bindgen::prelude::*;
#[cfg(web_sys_unstable_apis)]
#[wasm_bindgen]
extern "C" {
    # [wasm_bindgen (extends = :: js_sys :: Object , js_name = MLTriangularOptions)]
    #[derive(Debug, Clone, PartialEq, Eq)]
    #[doc = "The `MlTriangularOptions` dictionary."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlTriangularOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub type MlTriangularOptions;
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Get the `label` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlTriangularOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "label")]
    pub fn get_label(this: &MlTriangularOptions) -> Option<::alloc::string::String>;
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Change the `label` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlTriangularOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "label")]
    pub fn set_label(this: &MlTriangularOptions, val: &str);
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Get the `diagonal` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlTriangularOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "diagonal")]
    pub fn get_diagonal(this: &MlTriangularOptions) -> Option<i32>;
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Change the `diagonal` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlTriangularOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "diagonal")]
    pub fn set_diagonal(this: &MlTriangularOptions, val: i32);
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Get the `upper` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlTriangularOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "upper")]
    pub fn get_upper(this: &MlTriangularOptions) -> Option<bool>;
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Change the `upper` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlTriangularOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "upper")]
    pub fn set_upper(this: &MlTriangularOptions, val: bool);
}
#[cfg(web_sys_unstable_apis)]
impl MlTriangularOptions {
    #[doc = "Construct a new `MlTriangularOptions`."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlTriangularOptions`*"]
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
    #[deprecated = "Use `set_diagonal()` instead."]
    pub fn diagonal(&mut self, val: i32) -> &mut Self {
        self.set_diagonal(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[deprecated = "Use `set_upper()` instead."]
    pub fn upper(&mut self, val: bool) -> &mut Self {
        self.set_upper(val);
        self
    }
}
#[cfg(web_sys_unstable_apis)]
impl Default for MlTriangularOptions {
    fn default() -> Self {
        Self::new()
    }
}
