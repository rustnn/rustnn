#![allow(unused_imports)]
#![allow(clippy::all)]
use super::*;
use wasm_bindgen::prelude::*;
#[cfg(web_sys_unstable_apis)]
#[wasm_bindgen]
extern "C" {
    # [wasm_bindgen (extends = :: js_sys :: Object , js_name = MLClampOptions)]
    #[derive(Debug, Clone, PartialEq, Eq)]
    #[doc = "The `MlClampOptions` dictionary."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlClampOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub type MlClampOptions;
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Get the `label` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlClampOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "label")]
    pub fn get_label(this: &MlClampOptions) -> Option<::alloc::string::String>;
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Change the `label` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlClampOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "label")]
    pub fn set_label(this: &MlClampOptions, val: &str);
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Get the `maxValue` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlClampOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "maxValue")]
    pub fn get_max_value(this: &MlClampOptions) -> ::wasm_bindgen::JsValue;
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Change the `maxValue` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlClampOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "maxValue")]
    pub fn set_max_value(this: &MlClampOptions, val: &::js_sys::BigInt);
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Change the `maxValue` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlClampOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "maxValue")]
    pub fn set_max_value_f64(this: &MlClampOptions, val: f64);
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Get the `minValue` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlClampOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "minValue")]
    pub fn get_min_value(this: &MlClampOptions) -> ::wasm_bindgen::JsValue;
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Change the `minValue` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlClampOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "minValue")]
    pub fn set_min_value(this: &MlClampOptions, val: &::js_sys::BigInt);
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Change the `minValue` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlClampOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "minValue")]
    pub fn set_min_value_f64(this: &MlClampOptions, val: f64);
}
#[cfg(web_sys_unstable_apis)]
impl MlClampOptions {
    #[doc = "Construct a new `MlClampOptions`."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlClampOptions`*"]
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
    #[deprecated = "Use `set_max_value()` instead."]
    pub fn max_value(&mut self, val: &::js_sys::BigInt) -> &mut Self {
        self.set_max_value(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[deprecated = "Use `set_min_value()` instead."]
    pub fn min_value(&mut self, val: &::js_sys::BigInt) -> &mut Self {
        self.set_min_value(val);
        self
    }
}
#[cfg(web_sys_unstable_apis)]
impl Default for MlClampOptions {
    fn default() -> Self {
        Self::new()
    }
}
