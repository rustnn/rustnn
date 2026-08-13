#![allow(unused_imports)]
#![allow(clippy::all)]
use super::*;
use wasm_bindgen::prelude::*;
#[cfg(web_sys_unstable_apis)]
#[wasm_bindgen]
extern "C" {
    # [wasm_bindgen (extends = :: js_sys :: Object , js_name = ML , typescript_type = "ML")]
    #[derive(Debug, Clone, PartialEq, Eq)]
    #[doc = "The `Ml` class."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/ML)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `Ml`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub type Ml;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlContext")]
    # [wasm_bindgen (method , structural , js_class = "ML" , js_name = createContext)]
    #[doc = "The `createContext()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/ML/createContext)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `Ml`, `MlContext`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn create_context(this: &Ml) -> ::js_sys::Promise<MlContext>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlContext", feature = "MlContextOptions",))]
    # [wasm_bindgen (method , structural , js_class = "ML" , js_name = createContext)]
    #[doc = "The `createContext()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/ML/createContext)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `Ml`, `MlContext`, `MlContextOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn create_context_with_ml_context_options(
        this: &Ml,
        options: &MlContextOptions,
    ) -> ::js_sys::Promise<MlContext>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "GpuDevice", feature = "MlContext",))]
    # [wasm_bindgen (method , structural , js_class = "ML" , js_name = createContext)]
    #[doc = "The `createContext()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/ML/createContext)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `GpuDevice`, `Ml`, `MlContext`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn create_context_with_gpu_device(
        this: &Ml,
        gpu_device: &GpuDevice,
    ) -> ::js_sys::Promise<MlContext>;
}
