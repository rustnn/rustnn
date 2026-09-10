#![allow(unused_imports)]
#![allow(clippy::all)]
use super::*;
use wasm_bindgen::prelude::*;
#[cfg(web_sys_unstable_apis)]
#[wasm_bindgen]
extern "C" {
    # [wasm_bindgen (extends = :: js_sys :: Object , js_name = MLGraphBuilder , typescript_type = "MLGraphBuilder")]
    #[derive(Debug, Clone, PartialEq, Eq)]
    #[doc = "The `MlGraphBuilder` class."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub type MlGraphBuilder;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlContext")]
    #[wasm_bindgen(catch, constructor, js_class = "MLGraphBuilder")]
    #[doc = "The `new MlGraphBuilder(..)` constructor, creating a new instance of `MlGraphBuilder`."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/MLGraphBuilder)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlContext`, `MlGraphBuilder`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn new(context: &MlContext) -> Result<MlGraphBuilder, JsValue>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = abs)]
    #[doc = "The `abs()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/abs)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn abs(this: &MlGraphBuilder, input: &MlOperand) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlOperand", feature = "MlOperatorOptions",))]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = abs)]
    #[doc = "The `abs()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/abs)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`, `MlOperatorOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn abs_with_options(
        this: &MlGraphBuilder,
        input: &MlOperand,
        options: &MlOperatorOptions,
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = add)]
    #[doc = "The `add()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/add)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn add(this: &MlGraphBuilder, a: &MlOperand, b: &MlOperand) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlOperand", feature = "MlOperatorOptions",))]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = add)]
    #[doc = "The `add()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/add)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`, `MlOperatorOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn add_with_options(
        this: &MlGraphBuilder,
        a: &MlOperand,
        b: &MlOperand,
        options: &MlOperatorOptions,
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = argMax)]
    #[doc = "The `argMax()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/argMax)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn arg_max(this: &MlGraphBuilder, input: &MlOperand, axis: u32) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlArgMinMaxOptions", feature = "MlOperand",))]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = argMax)]
    #[doc = "The `argMax()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/argMax)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlArgMinMaxOptions`, `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn arg_max_with_options(
        this: &MlGraphBuilder,
        input: &MlOperand,
        axis: u32,
        options: &MlArgMinMaxOptions,
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = argMin)]
    #[doc = "The `argMin()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/argMin)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn arg_min(this: &MlGraphBuilder, input: &MlOperand, axis: u32) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlArgMinMaxOptions", feature = "MlOperand",))]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = argMin)]
    #[doc = "The `argMin()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/argMin)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlArgMinMaxOptions`, `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn arg_min_with_options(
        this: &MlGraphBuilder,
        input: &MlOperand,
        axis: u32,
        options: &MlArgMinMaxOptions,
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = averagePool2d)]
    #[doc = "The `averagePool2d()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/averagePool2d)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn average_pool2d(this: &MlGraphBuilder, input: &MlOperand) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlOperand", feature = "MlPool2dOptions",))]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = averagePool2d)]
    #[doc = "The `averagePool2d()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/averagePool2d)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`, `MlPool2dOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn average_pool2d_with_options(
        this: &MlGraphBuilder,
        input: &MlOperand,
        options: &MlPool2dOptions,
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = batchNormalization)]
    #[doc = "The `batchNormalization()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/batchNormalization)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn batch_normalization(
        this: &MlGraphBuilder,
        input: &MlOperand,
        mean: &MlOperand,
        variance: &MlOperand,
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlBatchNormalizationOptions", feature = "MlOperand",))]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = batchNormalization)]
    #[doc = "The `batchNormalization()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/batchNormalization)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlBatchNormalizationOptions`, `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn batch_normalization_with_options(
        this: &MlGraphBuilder,
        input: &MlOperand,
        mean: &MlOperand,
        variance: &MlOperand,
        options: &MlBatchNormalizationOptions,
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlGraph", feature = "MlOperand",))]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = build)]
    #[doc = "The `build()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/build)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraph`, `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn build(
        this: &MlGraphBuilder,
        outputs: &::js_sys::Object<MlOperand>,
    ) -> ::js_sys::Promise<MlGraph>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlOperand", feature = "MlOperandDataType",))]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = cast)]
    #[doc = "The `cast()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/cast)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`, `MlOperandDataType`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn cast(
        this: &MlGraphBuilder,
        input: &MlOperand,
        data_type: MlOperandDataType,
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(
        feature = "MlOperand",
        feature = "MlOperandDataType",
        feature = "MlOperatorOptions",
    ))]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = cast)]
    #[doc = "The `cast()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/cast)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`, `MlOperandDataType`, `MlOperatorOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn cast_with_options(
        this: &MlGraphBuilder,
        input: &MlOperand,
        data_type: MlOperandDataType,
        options: &MlOperatorOptions,
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = ceil)]
    #[doc = "The `ceil()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/ceil)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn ceil(this: &MlGraphBuilder, input: &MlOperand) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlOperand", feature = "MlOperatorOptions",))]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = ceil)]
    #[doc = "The `ceil()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/ceil)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`, `MlOperatorOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn ceil_with_options(
        this: &MlGraphBuilder,
        input: &MlOperand,
        options: &MlOperatorOptions,
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = clamp)]
    #[doc = "The `clamp()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/clamp)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn clamp(this: &MlGraphBuilder, input: &MlOperand) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlClampOptions", feature = "MlOperand",))]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = clamp)]
    #[doc = "The `clamp()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/clamp)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlClampOptions`, `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn clamp_with_options(
        this: &MlGraphBuilder,
        input: &MlOperand,
        options: &MlClampOptions,
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = concat)]
    #[doc = "The `concat()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/concat)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn concat(this: &MlGraphBuilder, inputs: &[MlOperand], axis: u32) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlOperand", feature = "MlOperatorOptions",))]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = concat)]
    #[doc = "The `concat()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/concat)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`, `MlOperatorOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn concat_with_options(
        this: &MlGraphBuilder,
        inputs: &[MlOperand],
        axis: u32,
        options: &MlOperatorOptions,
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlOperand", feature = "MlOperandDescriptor",))]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = constant)]
    #[doc = "The `constant()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/constant)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`, `MlOperandDescriptor`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn constant_with_ml_operand_descriptor_and_buffer_source(
        this: &MlGraphBuilder,
        descriptor: &MlOperandDescriptor,
        buffer: &::js_sys::Object,
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlOperand", feature = "MlOperandDescriptor",))]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = constant)]
    #[doc = "The `constant()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/constant)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`, `MlOperandDescriptor`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn constant_with_ml_operand_descriptor_and_u8_slice(
        this: &MlGraphBuilder,
        descriptor: &MlOperandDescriptor,
        buffer: &mut [u8],
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlOperand", feature = "MlOperandDescriptor",))]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = constant)]
    #[doc = "The `constant()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/constant)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`, `MlOperandDescriptor`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn constant_with_ml_operand_descriptor_and_u8_array(
        this: &MlGraphBuilder,
        descriptor: &MlOperandDescriptor,
        buffer: &::js_sys::Uint8Array,
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlOperand", feature = "MlOperandDataType",))]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = constant)]
    #[doc = "The `constant()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/constant)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`, `MlOperandDataType`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn constant_with_ml_operand_data_type_and_big_int(
        this: &MlGraphBuilder,
        data_type: MlOperandDataType,
        value: &::js_sys::BigInt,
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlOperand", feature = "MlOperandDataType",))]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = constant)]
    #[doc = "The `constant()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/constant)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`, `MlOperandDataType`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn constant_with_ml_operand_data_type_and_f64(
        this: &MlGraphBuilder,
        data_type: MlOperandDataType,
        value: f64,
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlOperand", feature = "MlTensor",))]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = constant)]
    #[doc = "The `constant()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/constant)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`, `MlTensor`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn constant_with_tensor(this: &MlGraphBuilder, tensor: &MlTensor) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = conv2d)]
    #[doc = "The `conv2d()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/conv2d)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn conv2d(this: &MlGraphBuilder, input: &MlOperand, filter: &MlOperand) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlConv2dOptions", feature = "MlOperand",))]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = conv2d)]
    #[doc = "The `conv2d()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/conv2d)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlConv2dOptions`, `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn conv2d_with_options(
        this: &MlGraphBuilder,
        input: &MlOperand,
        filter: &MlOperand,
        options: &MlConv2dOptions,
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = convTranspose2d)]
    #[doc = "The `convTranspose2d()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/convTranspose2d)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn conv_transpose2d(
        this: &MlGraphBuilder,
        input: &MlOperand,
        filter: &MlOperand,
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlConvTranspose2dOptions", feature = "MlOperand",))]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = convTranspose2d)]
    #[doc = "The `convTranspose2d()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/convTranspose2d)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlConvTranspose2dOptions`, `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn conv_transpose2d_with_options(
        this: &MlGraphBuilder,
        input: &MlOperand,
        filter: &MlOperand,
        options: &MlConvTranspose2dOptions,
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = cos)]
    #[doc = "The `cos()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/cos)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn cos(this: &MlGraphBuilder, input: &MlOperand) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlOperand", feature = "MlOperatorOptions",))]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = cos)]
    #[doc = "The `cos()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/cos)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`, `MlOperatorOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn cos_with_options(
        this: &MlGraphBuilder,
        input: &MlOperand,
        options: &MlOperatorOptions,
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = cumulativeSum)]
    #[doc = "The `cumulativeSum()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/cumulativeSum)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn cumulative_sum(this: &MlGraphBuilder, input: &MlOperand, axis: u32) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlCumulativeSumOptions", feature = "MlOperand",))]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = cumulativeSum)]
    #[doc = "The `cumulativeSum()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/cumulativeSum)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlCumulativeSumOptions`, `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn cumulative_sum_with_options(
        this: &MlGraphBuilder,
        input: &MlOperand,
        axis: u32,
        options: &MlCumulativeSumOptions,
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = dequantizeLinear)]
    #[doc = "The `dequantizeLinear()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/dequantizeLinear)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn dequantize_linear(
        this: &MlGraphBuilder,
        input: &MlOperand,
        scale: &MlOperand,
        zero_point: &MlOperand,
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlOperand", feature = "MlOperatorOptions",))]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = dequantizeLinear)]
    #[doc = "The `dequantizeLinear()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/dequantizeLinear)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`, `MlOperatorOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn dequantize_linear_with_options(
        this: &MlGraphBuilder,
        input: &MlOperand,
        scale: &MlOperand,
        zero_point: &MlOperand,
        options: &MlOperatorOptions,
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = div)]
    #[doc = "The `div()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/div)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn div(this: &MlGraphBuilder, a: &MlOperand, b: &MlOperand) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlOperand", feature = "MlOperatorOptions",))]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = div)]
    #[doc = "The `div()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/div)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`, `MlOperatorOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn div_with_options(
        this: &MlGraphBuilder,
        a: &MlOperand,
        b: &MlOperand,
        options: &MlOperatorOptions,
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = elu)]
    #[doc = "The `elu()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/elu)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn elu(this: &MlGraphBuilder, input: &MlOperand) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlEluOptions", feature = "MlOperand",))]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = elu)]
    #[doc = "The `elu()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/elu)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlEluOptions`, `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn elu_with_options(
        this: &MlGraphBuilder,
        input: &MlOperand,
        options: &MlEluOptions,
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = equal)]
    #[doc = "The `equal()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/equal)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn equal(this: &MlGraphBuilder, a: &MlOperand, b: &MlOperand) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlOperand", feature = "MlOperatorOptions",))]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = equal)]
    #[doc = "The `equal()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/equal)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`, `MlOperatorOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn equal_with_options(
        this: &MlGraphBuilder,
        a: &MlOperand,
        b: &MlOperand,
        options: &MlOperatorOptions,
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = erf)]
    #[doc = "The `erf()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/erf)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn erf(this: &MlGraphBuilder, input: &MlOperand) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlOperand", feature = "MlOperatorOptions",))]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = erf)]
    #[doc = "The `erf()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/erf)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`, `MlOperatorOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn erf_with_options(
        this: &MlGraphBuilder,
        input: &MlOperand,
        options: &MlOperatorOptions,
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = exp)]
    #[doc = "The `exp()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/exp)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn exp(this: &MlGraphBuilder, input: &MlOperand) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlOperand", feature = "MlOperatorOptions",))]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = exp)]
    #[doc = "The `exp()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/exp)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`, `MlOperatorOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn exp_with_options(
        this: &MlGraphBuilder,
        input: &MlOperand,
        options: &MlOperatorOptions,
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = expand)]
    #[doc = "The `expand()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/expand)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn expand(
        this: &MlGraphBuilder,
        input: &MlOperand,
        new_shape: &[::js_sys::Number],
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlOperand", feature = "MlOperatorOptions",))]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = expand)]
    #[doc = "The `expand()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/expand)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`, `MlOperatorOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn expand_with_options(
        this: &MlGraphBuilder,
        input: &MlOperand,
        new_shape: &[::js_sys::Number],
        options: &MlOperatorOptions,
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = floor)]
    #[doc = "The `floor()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/floor)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn floor(this: &MlGraphBuilder, input: &MlOperand) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlOperand", feature = "MlOperatorOptions",))]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = floor)]
    #[doc = "The `floor()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/floor)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`, `MlOperatorOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn floor_with_options(
        this: &MlGraphBuilder,
        input: &MlOperand,
        options: &MlOperatorOptions,
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = gather)]
    #[doc = "The `gather()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/gather)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn gather(this: &MlGraphBuilder, input: &MlOperand, indices: &MlOperand) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlGatherOptions", feature = "MlOperand",))]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = gather)]
    #[doc = "The `gather()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/gather)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGatherOptions`, `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn gather_with_options(
        this: &MlGraphBuilder,
        input: &MlOperand,
        indices: &MlOperand,
        options: &MlGatherOptions,
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = gatherElements)]
    #[doc = "The `gatherElements()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/gatherElements)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn gather_elements(
        this: &MlGraphBuilder,
        input: &MlOperand,
        indices: &MlOperand,
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlGatherOptions", feature = "MlOperand",))]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = gatherElements)]
    #[doc = "The `gatherElements()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/gatherElements)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGatherOptions`, `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn gather_elements_with_options(
        this: &MlGraphBuilder,
        input: &MlOperand,
        indices: &MlOperand,
        options: &MlGatherOptions,
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = gatherND)]
    #[doc = "The `gatherND()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/gatherND)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn gather_nd(this: &MlGraphBuilder, input: &MlOperand, indices: &MlOperand) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlOperand", feature = "MlOperatorOptions",))]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = gatherND)]
    #[doc = "The `gatherND()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/gatherND)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`, `MlOperatorOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn gather_nd_with_options(
        this: &MlGraphBuilder,
        input: &MlOperand,
        indices: &MlOperand,
        options: &MlOperatorOptions,
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = gelu)]
    #[doc = "The `gelu()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/gelu)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn gelu(this: &MlGraphBuilder, input: &MlOperand) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlOperand", feature = "MlOperatorOptions",))]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = gelu)]
    #[doc = "The `gelu()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/gelu)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`, `MlOperatorOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn gelu_with_options(
        this: &MlGraphBuilder,
        input: &MlOperand,
        options: &MlOperatorOptions,
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = gemm)]
    #[doc = "The `gemm()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/gemm)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn gemm(this: &MlGraphBuilder, a: &MlOperand, b: &MlOperand) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlGemmOptions", feature = "MlOperand",))]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = gemm)]
    #[doc = "The `gemm()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/gemm)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGemmOptions`, `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn gemm_with_options(
        this: &MlGraphBuilder,
        a: &MlOperand,
        b: &MlOperand,
        options: &MlGemmOptions,
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = greater)]
    #[doc = "The `greater()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/greater)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn greater(this: &MlGraphBuilder, a: &MlOperand, b: &MlOperand) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlOperand", feature = "MlOperatorOptions",))]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = greater)]
    #[doc = "The `greater()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/greater)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`, `MlOperatorOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn greater_with_options(
        this: &MlGraphBuilder,
        a: &MlOperand,
        b: &MlOperand,
        options: &MlOperatorOptions,
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = greaterOrEqual)]
    #[doc = "The `greaterOrEqual()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/greaterOrEqual)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn greater_or_equal(this: &MlGraphBuilder, a: &MlOperand, b: &MlOperand) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlOperand", feature = "MlOperatorOptions",))]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = greaterOrEqual)]
    #[doc = "The `greaterOrEqual()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/greaterOrEqual)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`, `MlOperatorOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn greater_or_equal_with_options(
        this: &MlGraphBuilder,
        a: &MlOperand,
        b: &MlOperand,
        options: &MlOperatorOptions,
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = gru)]
    #[doc = "The `gru()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/gru)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn gru(
        this: &MlGraphBuilder,
        input: &MlOperand,
        weight: &MlOperand,
        recurrent_weight: &MlOperand,
        steps: u32,
        hidden_size: u32,
    ) -> ::js_sys::Array<MlOperand>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlGruOptions", feature = "MlOperand",))]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = gru)]
    #[doc = "The `gru()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/gru)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlGruOptions`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn gru_with_options(
        this: &MlGraphBuilder,
        input: &MlOperand,
        weight: &MlOperand,
        recurrent_weight: &MlOperand,
        steps: u32,
        hidden_size: u32,
        options: &MlGruOptions,
    ) -> ::js_sys::Array<MlOperand>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = gruCell)]
    #[doc = "The `gruCell()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/gruCell)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn gru_cell(
        this: &MlGraphBuilder,
        input: &MlOperand,
        weight: &MlOperand,
        recurrent_weight: &MlOperand,
        hidden_state: &MlOperand,
        hidden_size: u32,
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlGruCellOptions", feature = "MlOperand",))]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = gruCell)]
    #[doc = "The `gruCell()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/gruCell)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlGruCellOptions`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn gru_cell_with_options(
        this: &MlGraphBuilder,
        input: &MlOperand,
        weight: &MlOperand,
        recurrent_weight: &MlOperand,
        hidden_state: &MlOperand,
        hidden_size: u32,
        options: &MlGruCellOptions,
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = hardSigmoid)]
    #[doc = "The `hardSigmoid()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/hardSigmoid)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn hard_sigmoid(this: &MlGraphBuilder, input: &MlOperand) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlHardSigmoidOptions", feature = "MlOperand",))]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = hardSigmoid)]
    #[doc = "The `hardSigmoid()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/hardSigmoid)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlHardSigmoidOptions`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn hard_sigmoid_with_options(
        this: &MlGraphBuilder,
        input: &MlOperand,
        options: &MlHardSigmoidOptions,
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = hardSwish)]
    #[doc = "The `hardSwish()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/hardSwish)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn hard_swish(this: &MlGraphBuilder, input: &MlOperand) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlOperand", feature = "MlOperatorOptions",))]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = hardSwish)]
    #[doc = "The `hardSwish()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/hardSwish)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`, `MlOperatorOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn hard_swish_with_options(
        this: &MlGraphBuilder,
        input: &MlOperand,
        options: &MlOperatorOptions,
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = identity)]
    #[doc = "The `identity()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/identity)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn identity(this: &MlGraphBuilder, input: &MlOperand) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlOperand", feature = "MlOperatorOptions",))]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = identity)]
    #[doc = "The `identity()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/identity)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`, `MlOperatorOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn identity_with_options(
        this: &MlGraphBuilder,
        input: &MlOperand,
        options: &MlOperatorOptions,
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlOperand", feature = "MlOperandDescriptor",))]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = input)]
    #[doc = "The `input()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/input)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`, `MlOperandDescriptor`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn input(this: &MlGraphBuilder, name: &str, descriptor: &MlOperandDescriptor) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = instanceNormalization)]
    #[doc = "The `instanceNormalization()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/instanceNormalization)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn instance_normalization(this: &MlGraphBuilder, input: &MlOperand) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlInstanceNormalizationOptions", feature = "MlOperand",))]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = instanceNormalization)]
    #[doc = "The `instanceNormalization()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/instanceNormalization)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlInstanceNormalizationOptions`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn instance_normalization_with_options(
        this: &MlGraphBuilder,
        input: &MlOperand,
        options: &MlInstanceNormalizationOptions,
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = isInfinite)]
    #[doc = "The `isInfinite()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/isInfinite)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn is_infinite(this: &MlGraphBuilder, a: &MlOperand) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlOperand", feature = "MlOperatorOptions",))]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = isInfinite)]
    #[doc = "The `isInfinite()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/isInfinite)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`, `MlOperatorOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn is_infinite_with_options(
        this: &MlGraphBuilder,
        a: &MlOperand,
        options: &MlOperatorOptions,
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = isNaN)]
    #[doc = "The `isNaN()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/isNaN)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn is_na_n(this: &MlGraphBuilder, a: &MlOperand) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlOperand", feature = "MlOperatorOptions",))]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = isNaN)]
    #[doc = "The `isNaN()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/isNaN)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`, `MlOperatorOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn is_na_n_with_options(
        this: &MlGraphBuilder,
        a: &MlOperand,
        options: &MlOperatorOptions,
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = l2Pool2d)]
    #[doc = "The `l2Pool2d()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/l2Pool2d)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn l2_pool2d(this: &MlGraphBuilder, input: &MlOperand) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlOperand", feature = "MlPool2dOptions",))]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = l2Pool2d)]
    #[doc = "The `l2Pool2d()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/l2Pool2d)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`, `MlPool2dOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn l2_pool2d_with_options(
        this: &MlGraphBuilder,
        input: &MlOperand,
        options: &MlPool2dOptions,
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = layerNormalization)]
    #[doc = "The `layerNormalization()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/layerNormalization)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn layer_normalization(this: &MlGraphBuilder, input: &MlOperand) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlLayerNormalizationOptions", feature = "MlOperand",))]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = layerNormalization)]
    #[doc = "The `layerNormalization()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/layerNormalization)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlLayerNormalizationOptions`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn layer_normalization_with_options(
        this: &MlGraphBuilder,
        input: &MlOperand,
        options: &MlLayerNormalizationOptions,
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = leakyRelu)]
    #[doc = "The `leakyRelu()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/leakyRelu)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn leaky_relu(this: &MlGraphBuilder, input: &MlOperand) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlLeakyReluOptions", feature = "MlOperand",))]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = leakyRelu)]
    #[doc = "The `leakyRelu()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/leakyRelu)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlLeakyReluOptions`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn leaky_relu_with_options(
        this: &MlGraphBuilder,
        input: &MlOperand,
        options: &MlLeakyReluOptions,
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = lesser)]
    #[doc = "The `lesser()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/lesser)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn lesser(this: &MlGraphBuilder, a: &MlOperand, b: &MlOperand) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlOperand", feature = "MlOperatorOptions",))]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = lesser)]
    #[doc = "The `lesser()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/lesser)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`, `MlOperatorOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn lesser_with_options(
        this: &MlGraphBuilder,
        a: &MlOperand,
        b: &MlOperand,
        options: &MlOperatorOptions,
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = lesserOrEqual)]
    #[doc = "The `lesserOrEqual()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/lesserOrEqual)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn lesser_or_equal(this: &MlGraphBuilder, a: &MlOperand, b: &MlOperand) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlOperand", feature = "MlOperatorOptions",))]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = lesserOrEqual)]
    #[doc = "The `lesserOrEqual()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/lesserOrEqual)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`, `MlOperatorOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn lesser_or_equal_with_options(
        this: &MlGraphBuilder,
        a: &MlOperand,
        b: &MlOperand,
        options: &MlOperatorOptions,
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = linear)]
    #[doc = "The `linear()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/linear)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn linear(this: &MlGraphBuilder, input: &MlOperand) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlLinearOptions", feature = "MlOperand",))]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = linear)]
    #[doc = "The `linear()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/linear)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlLinearOptions`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn linear_with_options(
        this: &MlGraphBuilder,
        input: &MlOperand,
        options: &MlLinearOptions,
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = log)]
    #[doc = "The `log()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/log)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn log(this: &MlGraphBuilder, input: &MlOperand) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlOperand", feature = "MlOperatorOptions",))]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = log)]
    #[doc = "The `log()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/log)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`, `MlOperatorOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn log_with_options(
        this: &MlGraphBuilder,
        input: &MlOperand,
        options: &MlOperatorOptions,
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = logicalAnd)]
    #[doc = "The `logicalAnd()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/logicalAnd)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn logical_and(this: &MlGraphBuilder, a: &MlOperand, b: &MlOperand) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlOperand", feature = "MlOperatorOptions",))]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = logicalAnd)]
    #[doc = "The `logicalAnd()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/logicalAnd)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`, `MlOperatorOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn logical_and_with_options(
        this: &MlGraphBuilder,
        a: &MlOperand,
        b: &MlOperand,
        options: &MlOperatorOptions,
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = logicalNot)]
    #[doc = "The `logicalNot()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/logicalNot)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn logical_not(this: &MlGraphBuilder, a: &MlOperand) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlOperand", feature = "MlOperatorOptions",))]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = logicalNot)]
    #[doc = "The `logicalNot()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/logicalNot)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`, `MlOperatorOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn logical_not_with_options(
        this: &MlGraphBuilder,
        a: &MlOperand,
        options: &MlOperatorOptions,
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = logicalOr)]
    #[doc = "The `logicalOr()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/logicalOr)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn logical_or(this: &MlGraphBuilder, a: &MlOperand, b: &MlOperand) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlOperand", feature = "MlOperatorOptions",))]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = logicalOr)]
    #[doc = "The `logicalOr()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/logicalOr)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`, `MlOperatorOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn logical_or_with_options(
        this: &MlGraphBuilder,
        a: &MlOperand,
        b: &MlOperand,
        options: &MlOperatorOptions,
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = logicalXor)]
    #[doc = "The `logicalXor()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/logicalXor)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn logical_xor(this: &MlGraphBuilder, a: &MlOperand, b: &MlOperand) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlOperand", feature = "MlOperatorOptions",))]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = logicalXor)]
    #[doc = "The `logicalXor()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/logicalXor)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`, `MlOperatorOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn logical_xor_with_options(
        this: &MlGraphBuilder,
        a: &MlOperand,
        b: &MlOperand,
        options: &MlOperatorOptions,
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = lstm)]
    #[doc = "The `lstm()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/lstm)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn lstm(
        this: &MlGraphBuilder,
        input: &MlOperand,
        weight: &MlOperand,
        recurrent_weight: &MlOperand,
        steps: u32,
        hidden_size: u32,
    ) -> ::js_sys::Array<MlOperand>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlLstmOptions", feature = "MlOperand",))]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = lstm)]
    #[doc = "The `lstm()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/lstm)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlLstmOptions`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn lstm_with_options(
        this: &MlGraphBuilder,
        input: &MlOperand,
        weight: &MlOperand,
        recurrent_weight: &MlOperand,
        steps: u32,
        hidden_size: u32,
        options: &MlLstmOptions,
    ) -> ::js_sys::Array<MlOperand>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = lstmCell)]
    #[doc = "The `lstmCell()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/lstmCell)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn lstm_cell(
        this: &MlGraphBuilder,
        input: &MlOperand,
        weight: &MlOperand,
        recurrent_weight: &MlOperand,
        hidden_state: &MlOperand,
        cell_state: &MlOperand,
        hidden_size: u32,
    ) -> ::js_sys::Array<MlOperand>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlLstmCellOptions", feature = "MlOperand",))]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = lstmCell)]
    #[doc = "The `lstmCell()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/lstmCell)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlLstmCellOptions`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn lstm_cell_with_options(
        this: &MlGraphBuilder,
        input: &MlOperand,
        weight: &MlOperand,
        recurrent_weight: &MlOperand,
        hidden_state: &MlOperand,
        cell_state: &MlOperand,
        hidden_size: u32,
        options: &MlLstmCellOptions,
    ) -> ::js_sys::Array<MlOperand>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = matmul)]
    #[doc = "The `matmul()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/matmul)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn matmul(this: &MlGraphBuilder, a: &MlOperand, b: &MlOperand) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlOperand", feature = "MlOperatorOptions",))]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = matmul)]
    #[doc = "The `matmul()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/matmul)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`, `MlOperatorOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn matmul_with_options(
        this: &MlGraphBuilder,
        a: &MlOperand,
        b: &MlOperand,
        options: &MlOperatorOptions,
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = max)]
    #[doc = "The `max()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/max)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn max(this: &MlGraphBuilder, a: &MlOperand, b: &MlOperand) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlOperand", feature = "MlOperatorOptions",))]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = max)]
    #[doc = "The `max()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/max)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`, `MlOperatorOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn max_with_options(
        this: &MlGraphBuilder,
        a: &MlOperand,
        b: &MlOperand,
        options: &MlOperatorOptions,
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = maxPool2d)]
    #[doc = "The `maxPool2d()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/maxPool2d)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn max_pool2d(this: &MlGraphBuilder, input: &MlOperand) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlOperand", feature = "MlPool2dOptions",))]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = maxPool2d)]
    #[doc = "The `maxPool2d()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/maxPool2d)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`, `MlPool2dOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn max_pool2d_with_options(
        this: &MlGraphBuilder,
        input: &MlOperand,
        options: &MlPool2dOptions,
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = min)]
    #[doc = "The `min()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/min)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn min(this: &MlGraphBuilder, a: &MlOperand, b: &MlOperand) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlOperand", feature = "MlOperatorOptions",))]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = min)]
    #[doc = "The `min()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/min)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`, `MlOperatorOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn min_with_options(
        this: &MlGraphBuilder,
        a: &MlOperand,
        b: &MlOperand,
        options: &MlOperatorOptions,
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = mul)]
    #[doc = "The `mul()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/mul)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn mul(this: &MlGraphBuilder, a: &MlOperand, b: &MlOperand) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlOperand", feature = "MlOperatorOptions",))]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = mul)]
    #[doc = "The `mul()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/mul)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`, `MlOperatorOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn mul_with_options(
        this: &MlGraphBuilder,
        a: &MlOperand,
        b: &MlOperand,
        options: &MlOperatorOptions,
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = neg)]
    #[doc = "The `neg()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/neg)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn neg(this: &MlGraphBuilder, input: &MlOperand) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlOperand", feature = "MlOperatorOptions",))]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = neg)]
    #[doc = "The `neg()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/neg)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`, `MlOperatorOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn neg_with_options(
        this: &MlGraphBuilder,
        input: &MlOperand,
        options: &MlOperatorOptions,
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = notEqual)]
    #[doc = "The `notEqual()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/notEqual)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn not_equal(this: &MlGraphBuilder, a: &MlOperand, b: &MlOperand) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlOperand", feature = "MlOperatorOptions",))]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = notEqual)]
    #[doc = "The `notEqual()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/notEqual)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`, `MlOperatorOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn not_equal_with_options(
        this: &MlGraphBuilder,
        a: &MlOperand,
        b: &MlOperand,
        options: &MlOperatorOptions,
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = pad)]
    #[doc = "The `pad()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/pad)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn pad(
        this: &MlGraphBuilder,
        input: &MlOperand,
        beginning_padding: &[::js_sys::Number],
        ending_padding: &[::js_sys::Number],
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlOperand", feature = "MlPadOptions",))]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = pad)]
    #[doc = "The `pad()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/pad)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`, `MlPadOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn pad_with_options(
        this: &MlGraphBuilder,
        input: &MlOperand,
        beginning_padding: &[::js_sys::Number],
        ending_padding: &[::js_sys::Number],
        options: &MlPadOptions,
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = pow)]
    #[doc = "The `pow()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/pow)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn pow(this: &MlGraphBuilder, a: &MlOperand, b: &MlOperand) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlOperand", feature = "MlOperatorOptions",))]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = pow)]
    #[doc = "The `pow()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/pow)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`, `MlOperatorOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn pow_with_options(
        this: &MlGraphBuilder,
        a: &MlOperand,
        b: &MlOperand,
        options: &MlOperatorOptions,
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = prelu)]
    #[doc = "The `prelu()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/prelu)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn prelu(this: &MlGraphBuilder, input: &MlOperand, slope: &MlOperand) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlOperand", feature = "MlOperatorOptions",))]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = prelu)]
    #[doc = "The `prelu()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/prelu)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`, `MlOperatorOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn prelu_with_options(
        this: &MlGraphBuilder,
        input: &MlOperand,
        slope: &MlOperand,
        options: &MlOperatorOptions,
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = quantizeLinear)]
    #[doc = "The `quantizeLinear()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/quantizeLinear)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn quantize_linear(
        this: &MlGraphBuilder,
        input: &MlOperand,
        scale: &MlOperand,
        zero_point: &MlOperand,
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlOperand", feature = "MlOperatorOptions",))]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = quantizeLinear)]
    #[doc = "The `quantizeLinear()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/quantizeLinear)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`, `MlOperatorOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn quantize_linear_with_options(
        this: &MlGraphBuilder,
        input: &MlOperand,
        scale: &MlOperand,
        zero_point: &MlOperand,
        options: &MlOperatorOptions,
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = reciprocal)]
    #[doc = "The `reciprocal()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/reciprocal)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn reciprocal(this: &MlGraphBuilder, input: &MlOperand) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlOperand", feature = "MlOperatorOptions",))]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = reciprocal)]
    #[doc = "The `reciprocal()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/reciprocal)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`, `MlOperatorOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn reciprocal_with_options(
        this: &MlGraphBuilder,
        input: &MlOperand,
        options: &MlOperatorOptions,
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = reduceL1)]
    #[doc = "The `reduceL1()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/reduceL1)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn reduce_l1(this: &MlGraphBuilder, input: &MlOperand) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlOperand", feature = "MlReduceOptions",))]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = reduceL1)]
    #[doc = "The `reduceL1()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/reduceL1)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`, `MlReduceOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn reduce_l1_with_options(
        this: &MlGraphBuilder,
        input: &MlOperand,
        options: &MlReduceOptions,
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = reduceL2)]
    #[doc = "The `reduceL2()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/reduceL2)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn reduce_l2(this: &MlGraphBuilder, input: &MlOperand) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlOperand", feature = "MlReduceOptions",))]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = reduceL2)]
    #[doc = "The `reduceL2()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/reduceL2)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`, `MlReduceOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn reduce_l2_with_options(
        this: &MlGraphBuilder,
        input: &MlOperand,
        options: &MlReduceOptions,
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = reduceLogSum)]
    #[doc = "The `reduceLogSum()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/reduceLogSum)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn reduce_log_sum(this: &MlGraphBuilder, input: &MlOperand) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlOperand", feature = "MlReduceOptions",))]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = reduceLogSum)]
    #[doc = "The `reduceLogSum()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/reduceLogSum)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`, `MlReduceOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn reduce_log_sum_with_options(
        this: &MlGraphBuilder,
        input: &MlOperand,
        options: &MlReduceOptions,
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = reduceLogSumExp)]
    #[doc = "The `reduceLogSumExp()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/reduceLogSumExp)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn reduce_log_sum_exp(this: &MlGraphBuilder, input: &MlOperand) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlOperand", feature = "MlReduceOptions",))]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = reduceLogSumExp)]
    #[doc = "The `reduceLogSumExp()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/reduceLogSumExp)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`, `MlReduceOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn reduce_log_sum_exp_with_options(
        this: &MlGraphBuilder,
        input: &MlOperand,
        options: &MlReduceOptions,
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = reduceMax)]
    #[doc = "The `reduceMax()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/reduceMax)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn reduce_max(this: &MlGraphBuilder, input: &MlOperand) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlOperand", feature = "MlReduceOptions",))]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = reduceMax)]
    #[doc = "The `reduceMax()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/reduceMax)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`, `MlReduceOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn reduce_max_with_options(
        this: &MlGraphBuilder,
        input: &MlOperand,
        options: &MlReduceOptions,
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = reduceMean)]
    #[doc = "The `reduceMean()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/reduceMean)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn reduce_mean(this: &MlGraphBuilder, input: &MlOperand) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlOperand", feature = "MlReduceOptions",))]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = reduceMean)]
    #[doc = "The `reduceMean()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/reduceMean)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`, `MlReduceOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn reduce_mean_with_options(
        this: &MlGraphBuilder,
        input: &MlOperand,
        options: &MlReduceOptions,
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = reduceMin)]
    #[doc = "The `reduceMin()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/reduceMin)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn reduce_min(this: &MlGraphBuilder, input: &MlOperand) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlOperand", feature = "MlReduceOptions",))]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = reduceMin)]
    #[doc = "The `reduceMin()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/reduceMin)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`, `MlReduceOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn reduce_min_with_options(
        this: &MlGraphBuilder,
        input: &MlOperand,
        options: &MlReduceOptions,
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = reduceProduct)]
    #[doc = "The `reduceProduct()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/reduceProduct)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn reduce_product(this: &MlGraphBuilder, input: &MlOperand) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlOperand", feature = "MlReduceOptions",))]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = reduceProduct)]
    #[doc = "The `reduceProduct()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/reduceProduct)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`, `MlReduceOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn reduce_product_with_options(
        this: &MlGraphBuilder,
        input: &MlOperand,
        options: &MlReduceOptions,
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = reduceSum)]
    #[doc = "The `reduceSum()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/reduceSum)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn reduce_sum(this: &MlGraphBuilder, input: &MlOperand) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlOperand", feature = "MlReduceOptions",))]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = reduceSum)]
    #[doc = "The `reduceSum()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/reduceSum)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`, `MlReduceOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn reduce_sum_with_options(
        this: &MlGraphBuilder,
        input: &MlOperand,
        options: &MlReduceOptions,
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = reduceSumSquare)]
    #[doc = "The `reduceSumSquare()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/reduceSumSquare)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn reduce_sum_square(this: &MlGraphBuilder, input: &MlOperand) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlOperand", feature = "MlReduceOptions",))]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = reduceSumSquare)]
    #[doc = "The `reduceSumSquare()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/reduceSumSquare)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`, `MlReduceOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn reduce_sum_square_with_options(
        this: &MlGraphBuilder,
        input: &MlOperand,
        options: &MlReduceOptions,
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = relu)]
    #[doc = "The `relu()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/relu)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn relu(this: &MlGraphBuilder, input: &MlOperand) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlOperand", feature = "MlOperatorOptions",))]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = relu)]
    #[doc = "The `relu()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/relu)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`, `MlOperatorOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn relu_with_options(
        this: &MlGraphBuilder,
        input: &MlOperand,
        options: &MlOperatorOptions,
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = resample2d)]
    #[doc = "The `resample2d()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/resample2d)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn resample2d(this: &MlGraphBuilder, input: &MlOperand) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlOperand", feature = "MlResample2dOptions",))]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = resample2d)]
    #[doc = "The `resample2d()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/resample2d)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`, `MlResample2dOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn resample2d_with_options(
        this: &MlGraphBuilder,
        input: &MlOperand,
        options: &MlResample2dOptions,
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = reshape)]
    #[doc = "The `reshape()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/reshape)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn reshape(
        this: &MlGraphBuilder,
        input: &MlOperand,
        new_shape: &[::js_sys::Number],
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlOperand", feature = "MlOperatorOptions",))]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = reshape)]
    #[doc = "The `reshape()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/reshape)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`, `MlOperatorOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn reshape_with_options(
        this: &MlGraphBuilder,
        input: &MlOperand,
        new_shape: &[::js_sys::Number],
        options: &MlOperatorOptions,
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = reverse)]
    #[doc = "The `reverse()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/reverse)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn reverse(this: &MlGraphBuilder, input: &MlOperand) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlOperand", feature = "MlReverseOptions",))]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = reverse)]
    #[doc = "The `reverse()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/reverse)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`, `MlReverseOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn reverse_with_options(
        this: &MlGraphBuilder,
        input: &MlOperand,
        options: &MlReverseOptions,
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = roundEven)]
    #[doc = "The `roundEven()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/roundEven)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn round_even(this: &MlGraphBuilder, input: &MlOperand) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlOperand", feature = "MlOperatorOptions",))]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = roundEven)]
    #[doc = "The `roundEven()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/roundEven)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`, `MlOperatorOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn round_even_with_options(
        this: &MlGraphBuilder,
        input: &MlOperand,
        options: &MlOperatorOptions,
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = scatterElements)]
    #[doc = "The `scatterElements()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/scatterElements)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn scatter_elements(
        this: &MlGraphBuilder,
        input: &MlOperand,
        indices: &MlOperand,
        updates: &MlOperand,
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlOperand", feature = "MlScatterOptions",))]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = scatterElements)]
    #[doc = "The `scatterElements()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/scatterElements)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`, `MlScatterOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn scatter_elements_with_options(
        this: &MlGraphBuilder,
        input: &MlOperand,
        indices: &MlOperand,
        updates: &MlOperand,
        options: &MlScatterOptions,
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = scatterND)]
    #[doc = "The `scatterND()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/scatterND)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn scatter_nd(
        this: &MlGraphBuilder,
        input: &MlOperand,
        indices: &MlOperand,
        updates: &MlOperand,
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlOperand", feature = "MlOperatorOptions",))]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = scatterND)]
    #[doc = "The `scatterND()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/scatterND)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`, `MlOperatorOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn scatter_nd_with_options(
        this: &MlGraphBuilder,
        input: &MlOperand,
        indices: &MlOperand,
        updates: &MlOperand,
        options: &MlOperatorOptions,
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = sigmoid)]
    #[doc = "The `sigmoid()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/sigmoid)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn sigmoid(this: &MlGraphBuilder, input: &MlOperand) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlOperand", feature = "MlOperatorOptions",))]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = sigmoid)]
    #[doc = "The `sigmoid()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/sigmoid)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`, `MlOperatorOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn sigmoid_with_options(
        this: &MlGraphBuilder,
        input: &MlOperand,
        options: &MlOperatorOptions,
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = sign)]
    #[doc = "The `sign()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/sign)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn sign(this: &MlGraphBuilder, input: &MlOperand) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlOperand", feature = "MlOperatorOptions",))]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = sign)]
    #[doc = "The `sign()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/sign)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`, `MlOperatorOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn sign_with_options(
        this: &MlGraphBuilder,
        input: &MlOperand,
        options: &MlOperatorOptions,
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = sin)]
    #[doc = "The `sin()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/sin)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn sin(this: &MlGraphBuilder, input: &MlOperand) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlOperand", feature = "MlOperatorOptions",))]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = sin)]
    #[doc = "The `sin()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/sin)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`, `MlOperatorOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn sin_with_options(
        this: &MlGraphBuilder,
        input: &MlOperand,
        options: &MlOperatorOptions,
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = slice)]
    #[doc = "The `slice()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/slice)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn slice(
        this: &MlGraphBuilder,
        input: &MlOperand,
        starts: &[::js_sys::Number],
        sizes: &[::js_sys::Number],
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlOperand", feature = "MlSliceOptions",))]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = slice)]
    #[doc = "The `slice()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/slice)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`, `MlSliceOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn slice_with_options(
        this: &MlGraphBuilder,
        input: &MlOperand,
        starts: &[::js_sys::Number],
        sizes: &[::js_sys::Number],
        options: &MlSliceOptions,
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = softmax)]
    #[doc = "The `softmax()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/softmax)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn softmax(this: &MlGraphBuilder, input: &MlOperand, axis: u32) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlOperand", feature = "MlOperatorOptions",))]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = softmax)]
    #[doc = "The `softmax()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/softmax)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`, `MlOperatorOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn softmax_with_options(
        this: &MlGraphBuilder,
        input: &MlOperand,
        axis: u32,
        options: &MlOperatorOptions,
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = softplus)]
    #[doc = "The `softplus()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/softplus)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn softplus(this: &MlGraphBuilder, input: &MlOperand) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlOperand", feature = "MlOperatorOptions",))]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = softplus)]
    #[doc = "The `softplus()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/softplus)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`, `MlOperatorOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn softplus_with_options(
        this: &MlGraphBuilder,
        input: &MlOperand,
        options: &MlOperatorOptions,
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = softsign)]
    #[doc = "The `softsign()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/softsign)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn softsign(this: &MlGraphBuilder, input: &MlOperand) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlOperand", feature = "MlOperatorOptions",))]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = softsign)]
    #[doc = "The `softsign()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/softsign)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`, `MlOperatorOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn softsign_with_options(
        this: &MlGraphBuilder,
        input: &MlOperand,
        options: &MlOperatorOptions,
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = split)]
    #[doc = "The `split()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/split)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn split_with_u32(
        this: &MlGraphBuilder,
        input: &MlOperand,
        splits: u32,
    ) -> ::js_sys::Array<MlOperand>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = split)]
    #[doc = "The `split()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/split)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn split_with_u32_sequence(
        this: &MlGraphBuilder,
        input: &MlOperand,
        splits: &[::js_sys::Number],
    ) -> ::js_sys::Array<MlOperand>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlOperand", feature = "MlSplitOptions",))]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = split)]
    #[doc = "The `split()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/split)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`, `MlSplitOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn split_with_u32_and_options(
        this: &MlGraphBuilder,
        input: &MlOperand,
        splits: u32,
        options: &MlSplitOptions,
    ) -> ::js_sys::Array<MlOperand>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlOperand", feature = "MlSplitOptions",))]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = split)]
    #[doc = "The `split()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/split)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`, `MlSplitOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn split_with_u32_sequence_and_options(
        this: &MlGraphBuilder,
        input: &MlOperand,
        splits: &[::js_sys::Number],
        options: &MlSplitOptions,
    ) -> ::js_sys::Array<MlOperand>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = sqrt)]
    #[doc = "The `sqrt()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/sqrt)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn sqrt(this: &MlGraphBuilder, input: &MlOperand) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlOperand", feature = "MlOperatorOptions",))]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = sqrt)]
    #[doc = "The `sqrt()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/sqrt)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`, `MlOperatorOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn sqrt_with_options(
        this: &MlGraphBuilder,
        input: &MlOperand,
        options: &MlOperatorOptions,
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = sub)]
    #[doc = "The `sub()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/sub)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn sub(this: &MlGraphBuilder, a: &MlOperand, b: &MlOperand) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlOperand", feature = "MlOperatorOptions",))]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = sub)]
    #[doc = "The `sub()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/sub)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`, `MlOperatorOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn sub_with_options(
        this: &MlGraphBuilder,
        a: &MlOperand,
        b: &MlOperand,
        options: &MlOperatorOptions,
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = tan)]
    #[doc = "The `tan()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/tan)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn tan(this: &MlGraphBuilder, input: &MlOperand) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlOperand", feature = "MlOperatorOptions",))]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = tan)]
    #[doc = "The `tan()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/tan)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`, `MlOperatorOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn tan_with_options(
        this: &MlGraphBuilder,
        input: &MlOperand,
        options: &MlOperatorOptions,
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = tanh)]
    #[doc = "The `tanh()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/tanh)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn tanh(this: &MlGraphBuilder, input: &MlOperand) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlOperand", feature = "MlOperatorOptions",))]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = tanh)]
    #[doc = "The `tanh()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/tanh)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`, `MlOperatorOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn tanh_with_options(
        this: &MlGraphBuilder,
        input: &MlOperand,
        options: &MlOperatorOptions,
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = tile)]
    #[doc = "The `tile()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/tile)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn tile(
        this: &MlGraphBuilder,
        input: &MlOperand,
        repetitions: &[::js_sys::Number],
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlOperand", feature = "MlOperatorOptions",))]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = tile)]
    #[doc = "The `tile()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/tile)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`, `MlOperatorOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn tile_with_options(
        this: &MlGraphBuilder,
        input: &MlOperand,
        repetitions: &[::js_sys::Number],
        options: &MlOperatorOptions,
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = transpose)]
    #[doc = "The `transpose()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/transpose)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn transpose(this: &MlGraphBuilder, input: &MlOperand) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlOperand", feature = "MlTransposeOptions",))]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = transpose)]
    #[doc = "The `transpose()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/transpose)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`, `MlTransposeOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn transpose_with_options(
        this: &MlGraphBuilder,
        input: &MlOperand,
        options: &MlTransposeOptions,
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = triangular)]
    #[doc = "The `triangular()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/triangular)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn triangular(this: &MlGraphBuilder, input: &MlOperand) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlOperand", feature = "MlTriangularOptions",))]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = triangular)]
    #[doc = "The `triangular()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/triangular)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`, `MlTriangularOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn triangular_with_options(
        this: &MlGraphBuilder,
        input: &MlOperand,
        options: &MlTriangularOptions,
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlOperand")]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = where)]
    #[doc = "The `where()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/where)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn where_(
        this: &MlGraphBuilder,
        condition: &MlOperand,
        true_value: &MlOperand,
        false_value: &MlOperand,
    ) -> MlOperand;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(all(feature = "MlOperand", feature = "MlOperatorOptions",))]
    # [wasm_bindgen (method , structural , js_class = "MLGraphBuilder" , js_name = where)]
    #[doc = "The `where()` method."]
    #[doc = ""]
    #[doc = "[MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/API/MLGraphBuilder/where)"]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGraphBuilder`, `MlOperand`, `MlOperatorOptions`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn where_with_options(
        this: &MlGraphBuilder,
        condition: &MlOperand,
        true_value: &MlOperand,
        false_value: &MlOperand,
        options: &MlOperatorOptions,
    ) -> MlOperand;
}
