#![allow(unused_imports)]
#![allow(clippy::all)]
use super::*;
use wasm_bindgen::prelude::*;
#[cfg(web_sys_unstable_apis)]
#[wasm_bindgen]
extern "C" {
    # [wasm_bindgen (extends = :: js_sys :: Object , js_name = MLContext , typescript_type = "MLContext")]
    #[derive(Debug, Clone, PartialEq, Eq)]
    #[doc = "The `MlContext` class."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLContext)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlContext`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub type MlContext;
    #[cfg(web_sys_unstable_apis)]
    # [wasm_bindgen (structural , method , getter , js_class = "MLContext" , js_name = accelerated)]
    #[doc = "Getter for the `accelerated` field of this object."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLContext/accelerated)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlContext`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn accelerated(this: &MlContext) -> bool;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlContextLostInfo")]
    # [wasm_bindgen (structural , method , getter , js_class = "MLContext" , js_name = lost)]
    #[doc = "Getter for the `lost` field of this object."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLContext/lost)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlContext`, `MlContextLostInfo`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn lost(this: &MlContext) -> ::js_sys::Promise<MlContextLostInfo>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlOperandDescriptor", feature = "MlTensor",))]
    # [wasm_bindgen (method , structural , js_class = "MLContext" , js_name = createConstantTensor)]
    #[doc = "The `createConstantTensor()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLContext/createConstantTensor)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlContext`, `MlOperandDescriptor`, `MlTensor`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn create_constant_tensor_with_buffer_source(
        this: &MlContext,
        descriptor: &MlOperandDescriptor,
        input_data: &::js_sys::Object,
    ) -> ::js_sys::Promise<MlTensor>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlOperandDescriptor", feature = "MlTensor",))]
    # [wasm_bindgen (method , structural , js_class = "MLContext" , js_name = createConstantTensor)]
    #[doc = "The `createConstantTensor()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLContext/createConstantTensor)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlContext`, `MlOperandDescriptor`, `MlTensor`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn create_constant_tensor_with_u8_slice(
        this: &MlContext,
        descriptor: &MlOperandDescriptor,
        input_data: &mut [u8],
    ) -> ::js_sys::Promise<MlTensor>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlOperandDescriptor", feature = "MlTensor",))]
    # [wasm_bindgen (method , structural , js_class = "MLContext" , js_name = createConstantTensor)]
    #[doc = "The `createConstantTensor()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLContext/createConstantTensor)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlContext`, `MlOperandDescriptor`, `MlTensor`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn create_constant_tensor_with_u8_array(
        this: &MlContext,
        descriptor: &MlOperandDescriptor,
        input_data: &::js_sys::Uint8Array,
    ) -> ::js_sys::Promise<MlTensor>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlTensor", feature = "MlTensorDescriptor",))]
    # [wasm_bindgen (method , structural , js_class = "MLContext" , js_name = createTensor)]
    #[doc = "The `createTensor()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLContext/createTensor)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlContext`, `MlTensor`, `MlTensorDescriptor`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn create_tensor(
        this: &MlContext,
        descriptor: &MlTensorDescriptor,
    ) -> ::js_sys::Promise<MlTensor>;
    #[cfg(web_sys_unstable_apis)]
    # [wasm_bindgen (method , structural , js_class = "MLContext" , js_name = destroy)]
    #[doc = "The `destroy()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLContext/destroy)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlContext`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn destroy(this: &MlContext);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlGraph", feature = "MlTensor",))]
    # [wasm_bindgen (method , structural , js_class = "MLContext" , js_name = dispatch)]
    #[doc = "The `dispatch()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLContext/dispatch)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlContext`, `MlGraph`, `MlTensor`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn dispatch(
        this: &MlContext,
        graph: &MlGraph,
        inputs: &::js_sys::Object<MlTensor>,
        outputs: &::js_sys::Object<MlTensor>,
    );
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOpSupportLimits")]
    # [wasm_bindgen (method , structural , js_class = "MLContext" , js_name = opSupportLimits)]
    #[doc = "The `opSupportLimits()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLContext/opSupportLimits)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlContext`, `MlOpSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn op_support_limits(this: &MlContext) -> MlOpSupportLimits;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlTensor")]
    # [wasm_bindgen (method , structural , js_class = "MLContext" , js_name = readTensor)]
    #[doc = "The `readTensor()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLContext/readTensor)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlContext`, `MlTensor`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn read_tensor(
        this: &MlContext,
        tensor: &MlTensor,
    ) -> ::js_sys::Promise<::js_sys::ArrayBuffer>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlTensor")]
    # [wasm_bindgen (method , structural , js_class = "MLContext" , js_name = readTensor)]
    #[doc = "The `readTensor()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLContext/readTensor)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlContext`, `MlTensor`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn read_tensor_with_buffer_source(
        this: &MlContext,
        tensor: &MlTensor,
        output_data: &::js_sys::Object,
    ) -> ::js_sys::Promise<::js_sys::Undefined>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlTensor")]
    # [wasm_bindgen (method , structural , js_class = "MLContext" , js_name = readTensor)]
    #[doc = "The `readTensor()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLContext/readTensor)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlContext`, `MlTensor`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn read_tensor_with_u8_slice(
        this: &MlContext,
        tensor: &MlTensor,
        output_data: &mut [u8],
    ) -> ::js_sys::Promise<::js_sys::Undefined>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlTensor")]
    # [wasm_bindgen (method , structural , js_class = "MLContext" , js_name = readTensor)]
    #[doc = "The `readTensor()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLContext/readTensor)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlContext`, `MlTensor`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn read_tensor_with_u8_array(
        this: &MlContext,
        tensor: &MlTensor,
        output_data: &::js_sys::Uint8Array,
    ) -> ::js_sys::Promise<::js_sys::Undefined>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlTensor")]
    # [wasm_bindgen (method , structural , js_class = "MLContext" , js_name = writeTensor)]
    #[doc = "The `writeTensor()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLContext/writeTensor)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlContext`, `MlTensor`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn write_tensor_with_buffer_source(
        this: &MlContext,
        tensor: &MlTensor,
        input_data: &::js_sys::Object,
    );
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlTensor")]
    # [wasm_bindgen (method , structural , js_class = "MLContext" , js_name = writeTensor)]
    #[doc = "The `writeTensor()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLContext/writeTensor)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlContext`, `MlTensor`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn write_tensor_with_u8_slice(this: &MlContext, tensor: &MlTensor, input_data: &mut [u8]);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlTensor")]
    # [wasm_bindgen (method , structural , js_class = "MLContext" , js_name = writeTensor)]
    #[doc = "The `writeTensor()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLContext/writeTensor)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlContext`, `MlTensor`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn write_tensor_with_u8_array(
        this: &MlContext,
        tensor: &MlTensor,
        input_data: &::js_sys::Uint8Array,
    );
}
