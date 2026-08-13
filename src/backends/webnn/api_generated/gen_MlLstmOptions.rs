#![allow(unused_imports)]
#![allow(clippy::all)]
use super::*;
use wasm_bindgen::prelude::*;
#[cfg(web_sys_unstable_apis)]
#[wasm_bindgen]
extern "C" {
    # [wasm_bindgen (extends = :: js_sys :: Object , js_name = MLLstmOptions)]
    #[derive(Debug, Clone, PartialEq, Eq)]
    #[doc = "The `MlLstmOptions` dictionary."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlLstmOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub type MlLstmOptions;
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Get the `label` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlLstmOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "label")]
    pub fn get_label(this: &MlLstmOptions) -> Option<::alloc::string::String>;
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Change the `label` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlLstmOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "label")]
    pub fn set_label(this: &MlLstmOptions, val: &str);
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Get the `activations` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlLstmOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "activations")]
    pub fn get_activations(this: &MlLstmOptions) -> Option<::js_sys::Array<::js_sys::JsString>>;
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Change the `activations` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlLstmOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "activations")]
    pub fn set_activations(this: &MlLstmOptions, val: &[::js_sys::JsString]);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    #[doc = "Get the `bias` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlLstmOptions`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "bias")]
    pub fn get_bias(this: &MlLstmOptions) -> Option<MlOperand>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    #[doc = "Change the `bias` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlLstmOptions`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "bias")]
    pub fn set_bias(this: &MlLstmOptions, val: &MlOperand);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlRecurrentNetworkDirection")]
    #[doc = "Get the `direction` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlLstmOptions`, `MlRecurrentNetworkDirection`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "direction")]
    pub fn get_direction(this: &MlLstmOptions) -> Option<MlRecurrentNetworkDirection>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlRecurrentNetworkDirection")]
    #[doc = "Change the `direction` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlLstmOptions`, `MlRecurrentNetworkDirection`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "direction")]
    pub fn set_direction(this: &MlLstmOptions, val: MlRecurrentNetworkDirection);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    #[doc = "Get the `initialCellState` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlLstmOptions`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "initialCellState")]
    pub fn get_initial_cell_state(this: &MlLstmOptions) -> Option<MlOperand>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    #[doc = "Change the `initialCellState` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlLstmOptions`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "initialCellState")]
    pub fn set_initial_cell_state(this: &MlLstmOptions, val: &MlOperand);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    #[doc = "Get the `initialHiddenState` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlLstmOptions`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "initialHiddenState")]
    pub fn get_initial_hidden_state(this: &MlLstmOptions) -> Option<MlOperand>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    #[doc = "Change the `initialHiddenState` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlLstmOptions`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "initialHiddenState")]
    pub fn set_initial_hidden_state(this: &MlLstmOptions, val: &MlOperand);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlLstmWeightLayout")]
    #[doc = "Get the `layout` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlLstmOptions`, `MlLstmWeightLayout`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "layout")]
    pub fn get_layout(this: &MlLstmOptions) -> Option<MlLstmWeightLayout>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlLstmWeightLayout")]
    #[doc = "Change the `layout` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlLstmOptions`, `MlLstmWeightLayout`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "layout")]
    pub fn set_layout(this: &MlLstmOptions, val: MlLstmWeightLayout);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    #[doc = "Get the `peepholeWeight` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlLstmOptions`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "peepholeWeight")]
    pub fn get_peephole_weight(this: &MlLstmOptions) -> Option<MlOperand>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    #[doc = "Change the `peepholeWeight` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlLstmOptions`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "peepholeWeight")]
    pub fn set_peephole_weight(this: &MlLstmOptions, val: &MlOperand);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    #[doc = "Get the `recurrentBias` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlLstmOptions`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "recurrentBias")]
    pub fn get_recurrent_bias(this: &MlLstmOptions) -> Option<MlOperand>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    #[doc = "Change the `recurrentBias` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlLstmOptions`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "recurrentBias")]
    pub fn set_recurrent_bias(this: &MlLstmOptions, val: &MlOperand);
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Get the `returnSequence` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlLstmOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "returnSequence")]
    pub fn get_return_sequence(this: &MlLstmOptions) -> Option<bool>;
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Change the `returnSequence` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlLstmOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "returnSequence")]
    pub fn set_return_sequence(this: &MlLstmOptions, val: bool);
}
#[cfg(web_sys_unstable_apis)]
impl MlLstmOptions {
    #[doc = "Construct a new `MlLstmOptions`."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlLstmOptions`*"]
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
    #[deprecated = "Use `set_activations()` instead."]
    pub fn activations(&mut self, val: &[::js_sys::JsString]) -> &mut Self {
        self.set_activations(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    #[deprecated = "Use `set_bias()` instead."]
    pub fn bias(&mut self, val: &MlOperand) -> &mut Self {
        self.set_bias(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlRecurrentNetworkDirection")]
    #[deprecated = "Use `set_direction()` instead."]
    pub fn direction(&mut self, val: MlRecurrentNetworkDirection) -> &mut Self {
        self.set_direction(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    #[deprecated = "Use `set_initial_cell_state()` instead."]
    pub fn initial_cell_state(&mut self, val: &MlOperand) -> &mut Self {
        self.set_initial_cell_state(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    #[deprecated = "Use `set_initial_hidden_state()` instead."]
    pub fn initial_hidden_state(&mut self, val: &MlOperand) -> &mut Self {
        self.set_initial_hidden_state(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlLstmWeightLayout")]
    #[deprecated = "Use `set_layout()` instead."]
    pub fn layout(&mut self, val: MlLstmWeightLayout) -> &mut Self {
        self.set_layout(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    #[deprecated = "Use `set_peephole_weight()` instead."]
    pub fn peephole_weight(&mut self, val: &MlOperand) -> &mut Self {
        self.set_peephole_weight(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    #[deprecated = "Use `set_recurrent_bias()` instead."]
    pub fn recurrent_bias(&mut self, val: &MlOperand) -> &mut Self {
        self.set_recurrent_bias(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[deprecated = "Use `set_return_sequence()` instead."]
    pub fn return_sequence(&mut self, val: bool) -> &mut Self {
        self.set_return_sequence(val);
        self
    }
}
#[cfg(web_sys_unstable_apis)]
impl Default for MlLstmOptions {
    fn default() -> Self {
        Self::new()
    }
}
