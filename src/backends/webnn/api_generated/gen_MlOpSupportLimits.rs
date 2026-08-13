#![allow(unused_imports)]
#![allow(clippy::all)]
use super::*;
use wasm_bindgen::prelude::*;
#[cfg(web_sys_unstable_apis)]
#[wasm_bindgen]
extern "C" {
    # [wasm_bindgen (extends = :: js_sys :: Object , js_name = MLOpSupportLimits)]
    #[derive(Debug, Clone, PartialEq, Eq)]
    #[doc = "The `MlOpSupportLimits` dictionary."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub type MlOpSupportLimits;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Get the `abs` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "abs")]
    pub fn get_abs(this: &MlOpSupportLimits) -> Option<MlSingleInputSupportLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Change the `abs` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "abs")]
    pub fn set_abs(this: &MlOpSupportLimits, val: &MlSingleInputSupportLimits);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlBinarySupportLimits")]
    #[doc = "Get the `add` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlBinarySupportLimits`, `MlOpSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "add")]
    pub fn get_add(this: &MlOpSupportLimits) -> Option<MlBinarySupportLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlBinarySupportLimits")]
    #[doc = "Change the `add` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlBinarySupportLimits`, `MlOpSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "add")]
    pub fn set_add(this: &MlOpSupportLimits, val: &MlBinarySupportLimits);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Get the `argMax` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "argMax")]
    pub fn get_arg_max(this: &MlOpSupportLimits) -> Option<MlSingleInputSupportLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Change the `argMax` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "argMax")]
    pub fn set_arg_max(this: &MlOpSupportLimits, val: &MlSingleInputSupportLimits);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Get the `argMin` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "argMin")]
    pub fn get_arg_min(this: &MlOpSupportLimits) -> Option<MlSingleInputSupportLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Change the `argMin` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "argMin")]
    pub fn set_arg_min(this: &MlOpSupportLimits, val: &MlSingleInputSupportLimits);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Get the `averagePool2d` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "averagePool2d")]
    pub fn get_average_pool2d(this: &MlOpSupportLimits) -> Option<MlSingleInputSupportLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Change the `averagePool2d` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "averagePool2d")]
    pub fn set_average_pool2d(this: &MlOpSupportLimits, val: &MlSingleInputSupportLimits);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlBatchNormalizationSupportLimits")]
    #[doc = "Get the `batchNormalization` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlBatchNormalizationSupportLimits`, `MlOpSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "batchNormalization")]
    pub fn get_batch_normalization(
        this: &MlOpSupportLimits,
    ) -> Option<MlBatchNormalizationSupportLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlBatchNormalizationSupportLimits")]
    #[doc = "Change the `batchNormalization` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlBatchNormalizationSupportLimits`, `MlOpSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "batchNormalization")]
    pub fn set_batch_normalization(
        this: &MlOpSupportLimits,
        val: &MlBatchNormalizationSupportLimits,
    );
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Get the `cast` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "cast")]
    pub fn get_cast(this: &MlOpSupportLimits) -> Option<MlSingleInputSupportLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Change the `cast` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "cast")]
    pub fn set_cast(this: &MlOpSupportLimits, val: &MlSingleInputSupportLimits);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Get the `ceil` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "ceil")]
    pub fn get_ceil(this: &MlOpSupportLimits) -> Option<MlSingleInputSupportLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Change the `ceil` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "ceil")]
    pub fn set_ceil(this: &MlOpSupportLimits, val: &MlSingleInputSupportLimits);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Get the `clamp` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "clamp")]
    pub fn get_clamp(this: &MlOpSupportLimits) -> Option<MlSingleInputSupportLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Change the `clamp` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "clamp")]
    pub fn set_clamp(this: &MlOpSupportLimits, val: &MlSingleInputSupportLimits);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlConcatSupportLimits")]
    #[doc = "Get the `concat` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlConcatSupportLimits`, `MlOpSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "concat")]
    pub fn get_concat(this: &MlOpSupportLimits) -> Option<MlConcatSupportLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlConcatSupportLimits")]
    #[doc = "Change the `concat` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlConcatSupportLimits`, `MlOpSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "concat")]
    pub fn set_concat(this: &MlOpSupportLimits, val: &MlConcatSupportLimits);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlTensorLimits")]
    #[doc = "Get the `constant` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlTensorLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "constant")]
    pub fn get_constant(this: &MlOpSupportLimits) -> Option<MlTensorLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlTensorLimits")]
    #[doc = "Change the `constant` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlTensorLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "constant")]
    pub fn set_constant(this: &MlOpSupportLimits, val: &MlTensorLimits);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlConv2dSupportLimits")]
    #[doc = "Get the `conv2d` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlConv2dSupportLimits`, `MlOpSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "conv2d")]
    pub fn get_conv2d(this: &MlOpSupportLimits) -> Option<MlConv2dSupportLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlConv2dSupportLimits")]
    #[doc = "Change the `conv2d` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlConv2dSupportLimits`, `MlOpSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "conv2d")]
    pub fn set_conv2d(this: &MlOpSupportLimits, val: &MlConv2dSupportLimits);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlConv2dSupportLimits")]
    #[doc = "Get the `convTranspose2d` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlConv2dSupportLimits`, `MlOpSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "convTranspose2d")]
    pub fn get_conv_transpose2d(this: &MlOpSupportLimits) -> Option<MlConv2dSupportLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlConv2dSupportLimits")]
    #[doc = "Change the `convTranspose2d` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlConv2dSupportLimits`, `MlOpSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "convTranspose2d")]
    pub fn set_conv_transpose2d(this: &MlOpSupportLimits, val: &MlConv2dSupportLimits);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Get the `cos` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "cos")]
    pub fn get_cos(this: &MlOpSupportLimits) -> Option<MlSingleInputSupportLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Change the `cos` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "cos")]
    pub fn set_cos(this: &MlOpSupportLimits, val: &MlSingleInputSupportLimits);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Get the `cumulativeSum` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "cumulativeSum")]
    pub fn get_cumulative_sum(this: &MlOpSupportLimits) -> Option<MlSingleInputSupportLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Change the `cumulativeSum` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "cumulativeSum")]
    pub fn set_cumulative_sum(this: &MlOpSupportLimits, val: &MlSingleInputSupportLimits);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlQuantizeDequantizeLinearSupportLimits")]
    #[doc = "Get the `dequantizeLinear` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlQuantizeDequantizeLinearSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "dequantizeLinear")]
    pub fn get_dequantize_linear(
        this: &MlOpSupportLimits,
    ) -> Option<MlQuantizeDequantizeLinearSupportLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlQuantizeDequantizeLinearSupportLimits")]
    #[doc = "Change the `dequantizeLinear` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlQuantizeDequantizeLinearSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "dequantizeLinear")]
    pub fn set_dequantize_linear(
        this: &MlOpSupportLimits,
        val: &MlQuantizeDequantizeLinearSupportLimits,
    );
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlBinarySupportLimits")]
    #[doc = "Get the `div` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlBinarySupportLimits`, `MlOpSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "div")]
    pub fn get_div(this: &MlOpSupportLimits) -> Option<MlBinarySupportLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlBinarySupportLimits")]
    #[doc = "Change the `div` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlBinarySupportLimits`, `MlOpSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "div")]
    pub fn set_div(this: &MlOpSupportLimits, val: &MlBinarySupportLimits);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Get the `elu` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "elu")]
    pub fn get_elu(this: &MlOpSupportLimits) -> Option<MlSingleInputSupportLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Change the `elu` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "elu")]
    pub fn set_elu(this: &MlOpSupportLimits, val: &MlSingleInputSupportLimits);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlBinarySupportLimits")]
    #[doc = "Get the `equal` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlBinarySupportLimits`, `MlOpSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "equal")]
    pub fn get_equal(this: &MlOpSupportLimits) -> Option<MlBinarySupportLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlBinarySupportLimits")]
    #[doc = "Change the `equal` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlBinarySupportLimits`, `MlOpSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "equal")]
    pub fn set_equal(this: &MlOpSupportLimits, val: &MlBinarySupportLimits);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Get the `erf` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "erf")]
    pub fn get_erf(this: &MlOpSupportLimits) -> Option<MlSingleInputSupportLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Change the `erf` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "erf")]
    pub fn set_erf(this: &MlOpSupportLimits, val: &MlSingleInputSupportLimits);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Get the `exp` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "exp")]
    pub fn get_exp(this: &MlOpSupportLimits) -> Option<MlSingleInputSupportLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Change the `exp` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "exp")]
    pub fn set_exp(this: &MlOpSupportLimits, val: &MlSingleInputSupportLimits);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Get the `expand` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "expand")]
    pub fn get_expand(this: &MlOpSupportLimits) -> Option<MlSingleInputSupportLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Change the `expand` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "expand")]
    pub fn set_expand(this: &MlOpSupportLimits, val: &MlSingleInputSupportLimits);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Get the `floor` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "floor")]
    pub fn get_floor(this: &MlOpSupportLimits) -> Option<MlSingleInputSupportLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Change the `floor` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "floor")]
    pub fn set_floor(this: &MlOpSupportLimits, val: &MlSingleInputSupportLimits);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlGatherSupportLimits")]
    #[doc = "Get the `gather` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGatherSupportLimits`, `MlOpSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "gather")]
    pub fn get_gather(this: &MlOpSupportLimits) -> Option<MlGatherSupportLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlGatherSupportLimits")]
    #[doc = "Change the `gather` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGatherSupportLimits`, `MlOpSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "gather")]
    pub fn set_gather(this: &MlOpSupportLimits, val: &MlGatherSupportLimits);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlGatherSupportLimits")]
    #[doc = "Get the `gatherElements` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGatherSupportLimits`, `MlOpSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "gatherElements")]
    pub fn get_gather_elements(this: &MlOpSupportLimits) -> Option<MlGatherSupportLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlGatherSupportLimits")]
    #[doc = "Change the `gatherElements` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGatherSupportLimits`, `MlOpSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "gatherElements")]
    pub fn set_gather_elements(this: &MlOpSupportLimits, val: &MlGatherSupportLimits);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlGatherSupportLimits")]
    #[doc = "Get the `gatherND` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGatherSupportLimits`, `MlOpSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "gatherND")]
    pub fn get_gather_nd(this: &MlOpSupportLimits) -> Option<MlGatherSupportLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlGatherSupportLimits")]
    #[doc = "Change the `gatherND` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGatherSupportLimits`, `MlOpSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "gatherND")]
    pub fn set_gather_nd(this: &MlOpSupportLimits, val: &MlGatherSupportLimits);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Get the `gelu` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "gelu")]
    pub fn get_gelu(this: &MlOpSupportLimits) -> Option<MlSingleInputSupportLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Change the `gelu` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "gelu")]
    pub fn set_gelu(this: &MlOpSupportLimits, val: &MlSingleInputSupportLimits);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlGemmSupportLimits")]
    #[doc = "Get the `gemm` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGemmSupportLimits`, `MlOpSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "gemm")]
    pub fn get_gemm(this: &MlOpSupportLimits) -> Option<MlGemmSupportLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlGemmSupportLimits")]
    #[doc = "Change the `gemm` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGemmSupportLimits`, `MlOpSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "gemm")]
    pub fn set_gemm(this: &MlOpSupportLimits, val: &MlGemmSupportLimits);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlBinarySupportLimits")]
    #[doc = "Get the `greater` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlBinarySupportLimits`, `MlOpSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "greater")]
    pub fn get_greater(this: &MlOpSupportLimits) -> Option<MlBinarySupportLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlBinarySupportLimits")]
    #[doc = "Change the `greater` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlBinarySupportLimits`, `MlOpSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "greater")]
    pub fn set_greater(this: &MlOpSupportLimits, val: &MlBinarySupportLimits);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlBinarySupportLimits")]
    #[doc = "Get the `greaterOrEqual` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlBinarySupportLimits`, `MlOpSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "greaterOrEqual")]
    pub fn get_greater_or_equal(this: &MlOpSupportLimits) -> Option<MlBinarySupportLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlBinarySupportLimits")]
    #[doc = "Change the `greaterOrEqual` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlBinarySupportLimits`, `MlOpSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "greaterOrEqual")]
    pub fn set_greater_or_equal(this: &MlOpSupportLimits, val: &MlBinarySupportLimits);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlGruSupportLimits")]
    #[doc = "Get the `gru` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGruSupportLimits`, `MlOpSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "gru")]
    pub fn get_gru(this: &MlOpSupportLimits) -> Option<MlGruSupportLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlGruSupportLimits")]
    #[doc = "Change the `gru` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGruSupportLimits`, `MlOpSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "gru")]
    pub fn set_gru(this: &MlOpSupportLimits, val: &MlGruSupportLimits);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlGruCellSupportLimits")]
    #[doc = "Get the `gruCell` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGruCellSupportLimits`, `MlOpSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "gruCell")]
    pub fn get_gru_cell(this: &MlOpSupportLimits) -> Option<MlGruCellSupportLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlGruCellSupportLimits")]
    #[doc = "Change the `gruCell` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlGruCellSupportLimits`, `MlOpSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "gruCell")]
    pub fn set_gru_cell(this: &MlOpSupportLimits, val: &MlGruCellSupportLimits);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Get the `hardSigmoid` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "hardSigmoid")]
    pub fn get_hard_sigmoid(this: &MlOpSupportLimits) -> Option<MlSingleInputSupportLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Change the `hardSigmoid` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "hardSigmoid")]
    pub fn set_hard_sigmoid(this: &MlOpSupportLimits, val: &MlSingleInputSupportLimits);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Get the `hardSwish` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "hardSwish")]
    pub fn get_hard_swish(this: &MlOpSupportLimits) -> Option<MlSingleInputSupportLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Change the `hardSwish` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "hardSwish")]
    pub fn set_hard_swish(this: &MlOpSupportLimits, val: &MlSingleInputSupportLimits);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Get the `identity` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "identity")]
    pub fn get_identity(this: &MlOpSupportLimits) -> Option<MlSingleInputSupportLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Change the `identity` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "identity")]
    pub fn set_identity(this: &MlOpSupportLimits, val: &MlSingleInputSupportLimits);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlTensorLimits")]
    #[doc = "Get the `input` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlTensorLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "input")]
    pub fn get_input(this: &MlOpSupportLimits) -> Option<MlTensorLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlTensorLimits")]
    #[doc = "Change the `input` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlTensorLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "input")]
    pub fn set_input(this: &MlOpSupportLimits, val: &MlTensorLimits);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlNormalizationSupportLimits")]
    #[doc = "Get the `instanceNormalization` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlNormalizationSupportLimits`, `MlOpSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "instanceNormalization")]
    pub fn get_instance_normalization(
        this: &MlOpSupportLimits,
    ) -> Option<MlNormalizationSupportLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlNormalizationSupportLimits")]
    #[doc = "Change the `instanceNormalization` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlNormalizationSupportLimits`, `MlOpSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "instanceNormalization")]
    pub fn set_instance_normalization(this: &MlOpSupportLimits, val: &MlNormalizationSupportLimits);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlLogicalNotSupportLimits")]
    #[doc = "Get the `isInfinite` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlLogicalNotSupportLimits`, `MlOpSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "isInfinite")]
    pub fn get_is_infinite(this: &MlOpSupportLimits) -> Option<MlLogicalNotSupportLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlLogicalNotSupportLimits")]
    #[doc = "Change the `isInfinite` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlLogicalNotSupportLimits`, `MlOpSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "isInfinite")]
    pub fn set_is_infinite(this: &MlOpSupportLimits, val: &MlLogicalNotSupportLimits);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlLogicalNotSupportLimits")]
    #[doc = "Get the `isNaN` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlLogicalNotSupportLimits`, `MlOpSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "isNaN")]
    pub fn get_is_na_n(this: &MlOpSupportLimits) -> Option<MlLogicalNotSupportLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlLogicalNotSupportLimits")]
    #[doc = "Change the `isNaN` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlLogicalNotSupportLimits`, `MlOpSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "isNaN")]
    pub fn set_is_na_n(this: &MlOpSupportLimits, val: &MlLogicalNotSupportLimits);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Get the `l2Pool2d` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "l2Pool2d")]
    pub fn get_l2_pool2d(this: &MlOpSupportLimits) -> Option<MlSingleInputSupportLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Change the `l2Pool2d` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "l2Pool2d")]
    pub fn set_l2_pool2d(this: &MlOpSupportLimits, val: &MlSingleInputSupportLimits);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlNormalizationSupportLimits")]
    #[doc = "Get the `layerNormalization` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlNormalizationSupportLimits`, `MlOpSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "layerNormalization")]
    pub fn get_layer_normalization(
        this: &MlOpSupportLimits,
    ) -> Option<MlNormalizationSupportLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlNormalizationSupportLimits")]
    #[doc = "Change the `layerNormalization` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlNormalizationSupportLimits`, `MlOpSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "layerNormalization")]
    pub fn set_layer_normalization(this: &MlOpSupportLimits, val: &MlNormalizationSupportLimits);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Get the `leakyRelu` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "leakyRelu")]
    pub fn get_leaky_relu(this: &MlOpSupportLimits) -> Option<MlSingleInputSupportLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Change the `leakyRelu` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "leakyRelu")]
    pub fn set_leaky_relu(this: &MlOpSupportLimits, val: &MlSingleInputSupportLimits);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlBinarySupportLimits")]
    #[doc = "Get the `lesser` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlBinarySupportLimits`, `MlOpSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "lesser")]
    pub fn get_lesser(this: &MlOpSupportLimits) -> Option<MlBinarySupportLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlBinarySupportLimits")]
    #[doc = "Change the `lesser` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlBinarySupportLimits`, `MlOpSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "lesser")]
    pub fn set_lesser(this: &MlOpSupportLimits, val: &MlBinarySupportLimits);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlBinarySupportLimits")]
    #[doc = "Get the `lesserOrEqual` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlBinarySupportLimits`, `MlOpSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "lesserOrEqual")]
    pub fn get_lesser_or_equal(this: &MlOpSupportLimits) -> Option<MlBinarySupportLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlBinarySupportLimits")]
    #[doc = "Change the `lesserOrEqual` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlBinarySupportLimits`, `MlOpSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "lesserOrEqual")]
    pub fn set_lesser_or_equal(this: &MlOpSupportLimits, val: &MlBinarySupportLimits);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Get the `linear` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "linear")]
    pub fn get_linear(this: &MlOpSupportLimits) -> Option<MlSingleInputSupportLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Change the `linear` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "linear")]
    pub fn set_linear(this: &MlOpSupportLimits, val: &MlSingleInputSupportLimits);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Get the `log` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "log")]
    pub fn get_log(this: &MlOpSupportLimits) -> Option<MlSingleInputSupportLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Change the `log` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "log")]
    pub fn set_log(this: &MlOpSupportLimits, val: &MlSingleInputSupportLimits);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlBinarySupportLimits")]
    #[doc = "Get the `logicalAnd` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlBinarySupportLimits`, `MlOpSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "logicalAnd")]
    pub fn get_logical_and(this: &MlOpSupportLimits) -> Option<MlBinarySupportLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlBinarySupportLimits")]
    #[doc = "Change the `logicalAnd` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlBinarySupportLimits`, `MlOpSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "logicalAnd")]
    pub fn set_logical_and(this: &MlOpSupportLimits, val: &MlBinarySupportLimits);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlLogicalNotSupportLimits")]
    #[doc = "Get the `logicalNot` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlLogicalNotSupportLimits`, `MlOpSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "logicalNot")]
    pub fn get_logical_not(this: &MlOpSupportLimits) -> Option<MlLogicalNotSupportLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlLogicalNotSupportLimits")]
    #[doc = "Change the `logicalNot` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlLogicalNotSupportLimits`, `MlOpSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "logicalNot")]
    pub fn set_logical_not(this: &MlOpSupportLimits, val: &MlLogicalNotSupportLimits);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlBinarySupportLimits")]
    #[doc = "Get the `logicalOr` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlBinarySupportLimits`, `MlOpSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "logicalOr")]
    pub fn get_logical_or(this: &MlOpSupportLimits) -> Option<MlBinarySupportLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlBinarySupportLimits")]
    #[doc = "Change the `logicalOr` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlBinarySupportLimits`, `MlOpSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "logicalOr")]
    pub fn set_logical_or(this: &MlOpSupportLimits, val: &MlBinarySupportLimits);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlBinarySupportLimits")]
    #[doc = "Get the `logicalXor` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlBinarySupportLimits`, `MlOpSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "logicalXor")]
    pub fn get_logical_xor(this: &MlOpSupportLimits) -> Option<MlBinarySupportLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlBinarySupportLimits")]
    #[doc = "Change the `logicalXor` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlBinarySupportLimits`, `MlOpSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "logicalXor")]
    pub fn set_logical_xor(this: &MlOpSupportLimits, val: &MlBinarySupportLimits);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlLstmSupportLimits")]
    #[doc = "Get the `lstm` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlLstmSupportLimits`, `MlOpSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "lstm")]
    pub fn get_lstm(this: &MlOpSupportLimits) -> Option<MlLstmSupportLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlLstmSupportLimits")]
    #[doc = "Change the `lstm` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlLstmSupportLimits`, `MlOpSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "lstm")]
    pub fn set_lstm(this: &MlOpSupportLimits, val: &MlLstmSupportLimits);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlLstmCellSupportLimits")]
    #[doc = "Get the `lstmCell` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlLstmCellSupportLimits`, `MlOpSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "lstmCell")]
    pub fn get_lstm_cell(this: &MlOpSupportLimits) -> Option<MlLstmCellSupportLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlLstmCellSupportLimits")]
    #[doc = "Change the `lstmCell` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlLstmCellSupportLimits`, `MlOpSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "lstmCell")]
    pub fn set_lstm_cell(this: &MlOpSupportLimits, val: &MlLstmCellSupportLimits);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlBinarySupportLimits")]
    #[doc = "Get the `matmul` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlBinarySupportLimits`, `MlOpSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "matmul")]
    pub fn get_matmul(this: &MlOpSupportLimits) -> Option<MlBinarySupportLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlBinarySupportLimits")]
    #[doc = "Change the `matmul` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlBinarySupportLimits`, `MlOpSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "matmul")]
    pub fn set_matmul(this: &MlOpSupportLimits, val: &MlBinarySupportLimits);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlBinarySupportLimits")]
    #[doc = "Get the `max` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlBinarySupportLimits`, `MlOpSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "max")]
    pub fn get_max(this: &MlOpSupportLimits) -> Option<MlBinarySupportLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlBinarySupportLimits")]
    #[doc = "Change the `max` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlBinarySupportLimits`, `MlOpSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "max")]
    pub fn set_max(this: &MlOpSupportLimits, val: &MlBinarySupportLimits);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Get the `maxPool2d` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "maxPool2d")]
    pub fn get_max_pool2d(this: &MlOpSupportLimits) -> Option<MlSingleInputSupportLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Change the `maxPool2d` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "maxPool2d")]
    pub fn set_max_pool2d(this: &MlOpSupportLimits, val: &MlSingleInputSupportLimits);
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Get the `maxTensorByteLength` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "maxTensorByteLength")]
    pub fn get_max_tensor_byte_length(this: &MlOpSupportLimits) -> Option<f64>;
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Change the `maxTensorByteLength` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "maxTensorByteLength")]
    pub fn set_max_tensor_byte_length(this: &MlOpSupportLimits, val: u32);
    #[cfg(web_sys_unstable_apis)]
    #[doc = "Change the `maxTensorByteLength` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "maxTensorByteLength")]
    pub fn set_max_tensor_byte_length_f64(this: &MlOpSupportLimits, val: f64);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlBinarySupportLimits")]
    #[doc = "Get the `min` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlBinarySupportLimits`, `MlOpSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "min")]
    pub fn get_min(this: &MlOpSupportLimits) -> Option<MlBinarySupportLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlBinarySupportLimits")]
    #[doc = "Change the `min` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlBinarySupportLimits`, `MlOpSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "min")]
    pub fn set_min(this: &MlOpSupportLimits, val: &MlBinarySupportLimits);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlBinarySupportLimits")]
    #[doc = "Get the `mul` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlBinarySupportLimits`, `MlOpSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "mul")]
    pub fn get_mul(this: &MlOpSupportLimits) -> Option<MlBinarySupportLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlBinarySupportLimits")]
    #[doc = "Change the `mul` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlBinarySupportLimits`, `MlOpSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "mul")]
    pub fn set_mul(this: &MlOpSupportLimits, val: &MlBinarySupportLimits);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Get the `neg` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "neg")]
    pub fn get_neg(this: &MlOpSupportLimits) -> Option<MlSingleInputSupportLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Change the `neg` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "neg")]
    pub fn set_neg(this: &MlOpSupportLimits, val: &MlSingleInputSupportLimits);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlBinarySupportLimits")]
    #[doc = "Get the `notEqual` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlBinarySupportLimits`, `MlOpSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "notEqual")]
    pub fn get_not_equal(this: &MlOpSupportLimits) -> Option<MlBinarySupportLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlBinarySupportLimits")]
    #[doc = "Change the `notEqual` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlBinarySupportLimits`, `MlOpSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "notEqual")]
    pub fn set_not_equal(this: &MlOpSupportLimits, val: &MlBinarySupportLimits);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlTensorLimits")]
    #[doc = "Get the `output` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlTensorLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "output")]
    pub fn get_output(this: &MlOpSupportLimits) -> Option<MlTensorLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlTensorLimits")]
    #[doc = "Change the `output` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlTensorLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "output")]
    pub fn set_output(this: &MlOpSupportLimits, val: &MlTensorLimits);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Get the `pad` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "pad")]
    pub fn get_pad(this: &MlOpSupportLimits) -> Option<MlSingleInputSupportLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Change the `pad` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "pad")]
    pub fn set_pad(this: &MlOpSupportLimits, val: &MlSingleInputSupportLimits);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlBinarySupportLimits")]
    #[doc = "Get the `pow` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlBinarySupportLimits`, `MlOpSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "pow")]
    pub fn get_pow(this: &MlOpSupportLimits) -> Option<MlBinarySupportLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlBinarySupportLimits")]
    #[doc = "Change the `pow` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlBinarySupportLimits`, `MlOpSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "pow")]
    pub fn set_pow(this: &MlOpSupportLimits, val: &MlBinarySupportLimits);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlInputOperandLayout")]
    #[doc = "Get the `preferredInputLayout` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlInputOperandLayout`, `MlOpSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "preferredInputLayout")]
    pub fn get_preferred_input_layout(this: &MlOpSupportLimits) -> Option<MlInputOperandLayout>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlInputOperandLayout")]
    #[doc = "Change the `preferredInputLayout` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlInputOperandLayout`, `MlOpSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "preferredInputLayout")]
    pub fn set_preferred_input_layout(this: &MlOpSupportLimits, val: MlInputOperandLayout);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlPreluSupportLimits")]
    #[doc = "Get the `prelu` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlPreluSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "prelu")]
    pub fn get_prelu(this: &MlOpSupportLimits) -> Option<MlPreluSupportLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlPreluSupportLimits")]
    #[doc = "Change the `prelu` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlPreluSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "prelu")]
    pub fn set_prelu(this: &MlOpSupportLimits, val: &MlPreluSupportLimits);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlQuantizeDequantizeLinearSupportLimits")]
    #[doc = "Get the `quantizeLinear` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlQuantizeDequantizeLinearSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "quantizeLinear")]
    pub fn get_quantize_linear(
        this: &MlOpSupportLimits,
    ) -> Option<MlQuantizeDequantizeLinearSupportLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlQuantizeDequantizeLinearSupportLimits")]
    #[doc = "Change the `quantizeLinear` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlQuantizeDequantizeLinearSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "quantizeLinear")]
    pub fn set_quantize_linear(
        this: &MlOpSupportLimits,
        val: &MlQuantizeDequantizeLinearSupportLimits,
    );
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Get the `reciprocal` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "reciprocal")]
    pub fn get_reciprocal(this: &MlOpSupportLimits) -> Option<MlSingleInputSupportLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Change the `reciprocal` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "reciprocal")]
    pub fn set_reciprocal(this: &MlOpSupportLimits, val: &MlSingleInputSupportLimits);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Get the `reduceL1` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "reduceL1")]
    pub fn get_reduce_l1(this: &MlOpSupportLimits) -> Option<MlSingleInputSupportLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Change the `reduceL1` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "reduceL1")]
    pub fn set_reduce_l1(this: &MlOpSupportLimits, val: &MlSingleInputSupportLimits);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Get the `reduceL2` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "reduceL2")]
    pub fn get_reduce_l2(this: &MlOpSupportLimits) -> Option<MlSingleInputSupportLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Change the `reduceL2` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "reduceL2")]
    pub fn set_reduce_l2(this: &MlOpSupportLimits, val: &MlSingleInputSupportLimits);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Get the `reduceLogSum` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "reduceLogSum")]
    pub fn get_reduce_log_sum(this: &MlOpSupportLimits) -> Option<MlSingleInputSupportLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Change the `reduceLogSum` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "reduceLogSum")]
    pub fn set_reduce_log_sum(this: &MlOpSupportLimits, val: &MlSingleInputSupportLimits);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Get the `reduceLogSumExp` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "reduceLogSumExp")]
    pub fn get_reduce_log_sum_exp(this: &MlOpSupportLimits) -> Option<MlSingleInputSupportLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Change the `reduceLogSumExp` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "reduceLogSumExp")]
    pub fn set_reduce_log_sum_exp(this: &MlOpSupportLimits, val: &MlSingleInputSupportLimits);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Get the `reduceMax` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "reduceMax")]
    pub fn get_reduce_max(this: &MlOpSupportLimits) -> Option<MlSingleInputSupportLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Change the `reduceMax` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "reduceMax")]
    pub fn set_reduce_max(this: &MlOpSupportLimits, val: &MlSingleInputSupportLimits);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Get the `reduceMean` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "reduceMean")]
    pub fn get_reduce_mean(this: &MlOpSupportLimits) -> Option<MlSingleInputSupportLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Change the `reduceMean` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "reduceMean")]
    pub fn set_reduce_mean(this: &MlOpSupportLimits, val: &MlSingleInputSupportLimits);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Get the `reduceMin` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "reduceMin")]
    pub fn get_reduce_min(this: &MlOpSupportLimits) -> Option<MlSingleInputSupportLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Change the `reduceMin` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "reduceMin")]
    pub fn set_reduce_min(this: &MlOpSupportLimits, val: &MlSingleInputSupportLimits);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Get the `reduceProduct` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "reduceProduct")]
    pub fn get_reduce_product(this: &MlOpSupportLimits) -> Option<MlSingleInputSupportLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Change the `reduceProduct` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "reduceProduct")]
    pub fn set_reduce_product(this: &MlOpSupportLimits, val: &MlSingleInputSupportLimits);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Get the `reduceSum` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "reduceSum")]
    pub fn get_reduce_sum(this: &MlOpSupportLimits) -> Option<MlSingleInputSupportLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Change the `reduceSum` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "reduceSum")]
    pub fn set_reduce_sum(this: &MlOpSupportLimits, val: &MlSingleInputSupportLimits);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Get the `reduceSumSquare` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "reduceSumSquare")]
    pub fn get_reduce_sum_square(this: &MlOpSupportLimits) -> Option<MlSingleInputSupportLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Change the `reduceSumSquare` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "reduceSumSquare")]
    pub fn set_reduce_sum_square(this: &MlOpSupportLimits, val: &MlSingleInputSupportLimits);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Get the `relu` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "relu")]
    pub fn get_relu(this: &MlOpSupportLimits) -> Option<MlSingleInputSupportLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Change the `relu` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "relu")]
    pub fn set_relu(this: &MlOpSupportLimits, val: &MlSingleInputSupportLimits);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Get the `resample2d` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "resample2d")]
    pub fn get_resample2d(this: &MlOpSupportLimits) -> Option<MlSingleInputSupportLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Change the `resample2d` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "resample2d")]
    pub fn set_resample2d(this: &MlOpSupportLimits, val: &MlSingleInputSupportLimits);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Get the `reshape` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "reshape")]
    pub fn get_reshape(this: &MlOpSupportLimits) -> Option<MlSingleInputSupportLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Change the `reshape` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "reshape")]
    pub fn set_reshape(this: &MlOpSupportLimits, val: &MlSingleInputSupportLimits);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Get the `reverse` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "reverse")]
    pub fn get_reverse(this: &MlOpSupportLimits) -> Option<MlSingleInputSupportLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Change the `reverse` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "reverse")]
    pub fn set_reverse(this: &MlOpSupportLimits, val: &MlSingleInputSupportLimits);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Get the `roundEven` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "roundEven")]
    pub fn get_round_even(this: &MlOpSupportLimits) -> Option<MlSingleInputSupportLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Change the `roundEven` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "roundEven")]
    pub fn set_round_even(this: &MlOpSupportLimits, val: &MlSingleInputSupportLimits);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlScatterSupportLimits")]
    #[doc = "Get the `scatterElements` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlScatterSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "scatterElements")]
    pub fn get_scatter_elements(this: &MlOpSupportLimits) -> Option<MlScatterSupportLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlScatterSupportLimits")]
    #[doc = "Change the `scatterElements` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlScatterSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "scatterElements")]
    pub fn set_scatter_elements(this: &MlOpSupportLimits, val: &MlScatterSupportLimits);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlScatterSupportLimits")]
    #[doc = "Get the `scatterND` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlScatterSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "scatterND")]
    pub fn get_scatter_nd(this: &MlOpSupportLimits) -> Option<MlScatterSupportLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlScatterSupportLimits")]
    #[doc = "Change the `scatterND` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlScatterSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "scatterND")]
    pub fn set_scatter_nd(this: &MlOpSupportLimits, val: &MlScatterSupportLimits);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Get the `sigmoid` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "sigmoid")]
    pub fn get_sigmoid(this: &MlOpSupportLimits) -> Option<MlSingleInputSupportLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Change the `sigmoid` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "sigmoid")]
    pub fn set_sigmoid(this: &MlOpSupportLimits, val: &MlSingleInputSupportLimits);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Get the `sign` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "sign")]
    pub fn get_sign(this: &MlOpSupportLimits) -> Option<MlSingleInputSupportLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Change the `sign` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "sign")]
    pub fn set_sign(this: &MlOpSupportLimits, val: &MlSingleInputSupportLimits);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Get the `sin` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "sin")]
    pub fn get_sin(this: &MlOpSupportLimits) -> Option<MlSingleInputSupportLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Change the `sin` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "sin")]
    pub fn set_sin(this: &MlOpSupportLimits, val: &MlSingleInputSupportLimits);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Get the `slice` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "slice")]
    pub fn get_slice(this: &MlOpSupportLimits) -> Option<MlSingleInputSupportLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Change the `slice` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "slice")]
    pub fn set_slice(this: &MlOpSupportLimits, val: &MlSingleInputSupportLimits);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Get the `softmax` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "softmax")]
    pub fn get_softmax(this: &MlOpSupportLimits) -> Option<MlSingleInputSupportLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Change the `softmax` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "softmax")]
    pub fn set_softmax(this: &MlOpSupportLimits, val: &MlSingleInputSupportLimits);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Get the `softplus` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "softplus")]
    pub fn get_softplus(this: &MlOpSupportLimits) -> Option<MlSingleInputSupportLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Change the `softplus` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "softplus")]
    pub fn set_softplus(this: &MlOpSupportLimits, val: &MlSingleInputSupportLimits);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Get the `softsign` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "softsign")]
    pub fn get_softsign(this: &MlOpSupportLimits) -> Option<MlSingleInputSupportLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Change the `softsign` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "softsign")]
    pub fn set_softsign(this: &MlOpSupportLimits, val: &MlSingleInputSupportLimits);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSplitSupportLimits")]
    #[doc = "Get the `split` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSplitSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "split")]
    pub fn get_split(this: &MlOpSupportLimits) -> Option<MlSplitSupportLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSplitSupportLimits")]
    #[doc = "Change the `split` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSplitSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "split")]
    pub fn set_split(this: &MlOpSupportLimits, val: &MlSplitSupportLimits);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Get the `sqrt` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "sqrt")]
    pub fn get_sqrt(this: &MlOpSupportLimits) -> Option<MlSingleInputSupportLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Change the `sqrt` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "sqrt")]
    pub fn set_sqrt(this: &MlOpSupportLimits, val: &MlSingleInputSupportLimits);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlBinarySupportLimits")]
    #[doc = "Get the `sub` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlBinarySupportLimits`, `MlOpSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "sub")]
    pub fn get_sub(this: &MlOpSupportLimits) -> Option<MlBinarySupportLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlBinarySupportLimits")]
    #[doc = "Change the `sub` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlBinarySupportLimits`, `MlOpSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "sub")]
    pub fn set_sub(this: &MlOpSupportLimits, val: &MlBinarySupportLimits);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Get the `tan` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "tan")]
    pub fn get_tan(this: &MlOpSupportLimits) -> Option<MlSingleInputSupportLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Change the `tan` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "tan")]
    pub fn set_tan(this: &MlOpSupportLimits, val: &MlSingleInputSupportLimits);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Get the `tanh` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "tanh")]
    pub fn get_tanh(this: &MlOpSupportLimits) -> Option<MlSingleInputSupportLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Change the `tanh` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "tanh")]
    pub fn set_tanh(this: &MlOpSupportLimits, val: &MlSingleInputSupportLimits);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Get the `tile` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "tile")]
    pub fn get_tile(this: &MlOpSupportLimits) -> Option<MlSingleInputSupportLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Change the `tile` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "tile")]
    pub fn set_tile(this: &MlOpSupportLimits, val: &MlSingleInputSupportLimits);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Get the `transpose` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "transpose")]
    pub fn get_transpose(this: &MlOpSupportLimits) -> Option<MlSingleInputSupportLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Change the `transpose` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "transpose")]
    pub fn set_transpose(this: &MlOpSupportLimits, val: &MlSingleInputSupportLimits);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Get the `triangular` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "triangular")]
    pub fn get_triangular(this: &MlOpSupportLimits) -> Option<MlSingleInputSupportLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[doc = "Change the `triangular` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlSingleInputSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "triangular")]
    pub fn set_triangular(this: &MlOpSupportLimits, val: &MlSingleInputSupportLimits);
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlWhereSupportLimits")]
    #[doc = "Get the `where` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlWhereSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, getter = "where")]
    pub fn get_where(this: &MlOpSupportLimits) -> Option<MlWhereSupportLimits>;
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlWhereSupportLimits")]
    #[doc = "Change the `where` field of this object."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`, `MlWhereSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    #[wasm_bindgen(method, setter = "where")]
    pub fn set_where(this: &MlOpSupportLimits, val: &MlWhereSupportLimits);
}
#[cfg(web_sys_unstable_apis)]
impl MlOpSupportLimits {
    #[doc = "Construct a new `MlOpSupportLimits`."]
    #[doc = ""]
    #[doc = "*This API requires the following crate features to be activated: `MlOpSupportLimits`*"]
    #[doc = ""]
    #[doc = "*This API is unstable and requires `--cfg=web_sys_unstable_apis` to be activated, as"]
    #[doc = "[described in the `wasm-bindgen` guide](https://wasm-bindgen.github.io/wasm-bindgen/web-sys/unstable-apis.html)*"]
    pub fn new() -> Self {
        #[allow(unused_mut)]
        let mut ret: Self = ::wasm_bindgen::JsCast::unchecked_into(::js_sys::Object::new());
        ret
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[deprecated = "Use `set_abs()` instead."]
    pub fn abs(&mut self, val: &MlSingleInputSupportLimits) -> &mut Self {
        self.set_abs(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlBinarySupportLimits")]
    #[deprecated = "Use `set_add()` instead."]
    pub fn add(&mut self, val: &MlBinarySupportLimits) -> &mut Self {
        self.set_add(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[deprecated = "Use `set_arg_max()` instead."]
    pub fn arg_max(&mut self, val: &MlSingleInputSupportLimits) -> &mut Self {
        self.set_arg_max(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[deprecated = "Use `set_arg_min()` instead."]
    pub fn arg_min(&mut self, val: &MlSingleInputSupportLimits) -> &mut Self {
        self.set_arg_min(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[deprecated = "Use `set_average_pool2d()` instead."]
    pub fn average_pool2d(&mut self, val: &MlSingleInputSupportLimits) -> &mut Self {
        self.set_average_pool2d(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlBatchNormalizationSupportLimits")]
    #[deprecated = "Use `set_batch_normalization()` instead."]
    pub fn batch_normalization(&mut self, val: &MlBatchNormalizationSupportLimits) -> &mut Self {
        self.set_batch_normalization(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[deprecated = "Use `set_cast()` instead."]
    pub fn cast(&mut self, val: &MlSingleInputSupportLimits) -> &mut Self {
        self.set_cast(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[deprecated = "Use `set_ceil()` instead."]
    pub fn ceil(&mut self, val: &MlSingleInputSupportLimits) -> &mut Self {
        self.set_ceil(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[deprecated = "Use `set_clamp()` instead."]
    pub fn clamp(&mut self, val: &MlSingleInputSupportLimits) -> &mut Self {
        self.set_clamp(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlConcatSupportLimits")]
    #[deprecated = "Use `set_concat()` instead."]
    pub fn concat(&mut self, val: &MlConcatSupportLimits) -> &mut Self {
        self.set_concat(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlTensorLimits")]
    #[deprecated = "Use `set_constant()` instead."]
    pub fn constant(&mut self, val: &MlTensorLimits) -> &mut Self {
        self.set_constant(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlConv2dSupportLimits")]
    #[deprecated = "Use `set_conv2d()` instead."]
    pub fn conv2d(&mut self, val: &MlConv2dSupportLimits) -> &mut Self {
        self.set_conv2d(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlConv2dSupportLimits")]
    #[deprecated = "Use `set_conv_transpose2d()` instead."]
    pub fn conv_transpose2d(&mut self, val: &MlConv2dSupportLimits) -> &mut Self {
        self.set_conv_transpose2d(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[deprecated = "Use `set_cos()` instead."]
    pub fn cos(&mut self, val: &MlSingleInputSupportLimits) -> &mut Self {
        self.set_cos(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[deprecated = "Use `set_cumulative_sum()` instead."]
    pub fn cumulative_sum(&mut self, val: &MlSingleInputSupportLimits) -> &mut Self {
        self.set_cumulative_sum(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlQuantizeDequantizeLinearSupportLimits")]
    #[deprecated = "Use `set_dequantize_linear()` instead."]
    pub fn dequantize_linear(
        &mut self,
        val: &MlQuantizeDequantizeLinearSupportLimits,
    ) -> &mut Self {
        self.set_dequantize_linear(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlBinarySupportLimits")]
    #[deprecated = "Use `set_div()` instead."]
    pub fn div(&mut self, val: &MlBinarySupportLimits) -> &mut Self {
        self.set_div(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[deprecated = "Use `set_elu()` instead."]
    pub fn elu(&mut self, val: &MlSingleInputSupportLimits) -> &mut Self {
        self.set_elu(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlBinarySupportLimits")]
    #[deprecated = "Use `set_equal()` instead."]
    pub fn equal(&mut self, val: &MlBinarySupportLimits) -> &mut Self {
        self.set_equal(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[deprecated = "Use `set_erf()` instead."]
    pub fn erf(&mut self, val: &MlSingleInputSupportLimits) -> &mut Self {
        self.set_erf(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[deprecated = "Use `set_exp()` instead."]
    pub fn exp(&mut self, val: &MlSingleInputSupportLimits) -> &mut Self {
        self.set_exp(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[deprecated = "Use `set_expand()` instead."]
    pub fn expand(&mut self, val: &MlSingleInputSupportLimits) -> &mut Self {
        self.set_expand(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[deprecated = "Use `set_floor()` instead."]
    pub fn floor(&mut self, val: &MlSingleInputSupportLimits) -> &mut Self {
        self.set_floor(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlGatherSupportLimits")]
    #[deprecated = "Use `set_gather()` instead."]
    pub fn gather(&mut self, val: &MlGatherSupportLimits) -> &mut Self {
        self.set_gather(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlGatherSupportLimits")]
    #[deprecated = "Use `set_gather_elements()` instead."]
    pub fn gather_elements(&mut self, val: &MlGatherSupportLimits) -> &mut Self {
        self.set_gather_elements(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlGatherSupportLimits")]
    #[deprecated = "Use `set_gather_nd()` instead."]
    pub fn gather_nd(&mut self, val: &MlGatherSupportLimits) -> &mut Self {
        self.set_gather_nd(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[deprecated = "Use `set_gelu()` instead."]
    pub fn gelu(&mut self, val: &MlSingleInputSupportLimits) -> &mut Self {
        self.set_gelu(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlGemmSupportLimits")]
    #[deprecated = "Use `set_gemm()` instead."]
    pub fn gemm(&mut self, val: &MlGemmSupportLimits) -> &mut Self {
        self.set_gemm(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlBinarySupportLimits")]
    #[deprecated = "Use `set_greater()` instead."]
    pub fn greater(&mut self, val: &MlBinarySupportLimits) -> &mut Self {
        self.set_greater(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlBinarySupportLimits")]
    #[deprecated = "Use `set_greater_or_equal()` instead."]
    pub fn greater_or_equal(&mut self, val: &MlBinarySupportLimits) -> &mut Self {
        self.set_greater_or_equal(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlGruSupportLimits")]
    #[deprecated = "Use `set_gru()` instead."]
    pub fn gru(&mut self, val: &MlGruSupportLimits) -> &mut Self {
        self.set_gru(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlGruCellSupportLimits")]
    #[deprecated = "Use `set_gru_cell()` instead."]
    pub fn gru_cell(&mut self, val: &MlGruCellSupportLimits) -> &mut Self {
        self.set_gru_cell(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[deprecated = "Use `set_hard_sigmoid()` instead."]
    pub fn hard_sigmoid(&mut self, val: &MlSingleInputSupportLimits) -> &mut Self {
        self.set_hard_sigmoid(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[deprecated = "Use `set_hard_swish()` instead."]
    pub fn hard_swish(&mut self, val: &MlSingleInputSupportLimits) -> &mut Self {
        self.set_hard_swish(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[deprecated = "Use `set_identity()` instead."]
    pub fn identity(&mut self, val: &MlSingleInputSupportLimits) -> &mut Self {
        self.set_identity(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlTensorLimits")]
    #[deprecated = "Use `set_input()` instead."]
    pub fn input(&mut self, val: &MlTensorLimits) -> &mut Self {
        self.set_input(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlNormalizationSupportLimits")]
    #[deprecated = "Use `set_instance_normalization()` instead."]
    pub fn instance_normalization(&mut self, val: &MlNormalizationSupportLimits) -> &mut Self {
        self.set_instance_normalization(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlLogicalNotSupportLimits")]
    #[deprecated = "Use `set_is_infinite()` instead."]
    pub fn is_infinite(&mut self, val: &MlLogicalNotSupportLimits) -> &mut Self {
        self.set_is_infinite(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlLogicalNotSupportLimits")]
    #[deprecated = "Use `set_is_na_n()` instead."]
    pub fn is_na_n(&mut self, val: &MlLogicalNotSupportLimits) -> &mut Self {
        self.set_is_na_n(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[deprecated = "Use `set_l2_pool2d()` instead."]
    pub fn l2_pool2d(&mut self, val: &MlSingleInputSupportLimits) -> &mut Self {
        self.set_l2_pool2d(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlNormalizationSupportLimits")]
    #[deprecated = "Use `set_layer_normalization()` instead."]
    pub fn layer_normalization(&mut self, val: &MlNormalizationSupportLimits) -> &mut Self {
        self.set_layer_normalization(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[deprecated = "Use `set_leaky_relu()` instead."]
    pub fn leaky_relu(&mut self, val: &MlSingleInputSupportLimits) -> &mut Self {
        self.set_leaky_relu(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlBinarySupportLimits")]
    #[deprecated = "Use `set_lesser()` instead."]
    pub fn lesser(&mut self, val: &MlBinarySupportLimits) -> &mut Self {
        self.set_lesser(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlBinarySupportLimits")]
    #[deprecated = "Use `set_lesser_or_equal()` instead."]
    pub fn lesser_or_equal(&mut self, val: &MlBinarySupportLimits) -> &mut Self {
        self.set_lesser_or_equal(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[deprecated = "Use `set_linear()` instead."]
    pub fn linear(&mut self, val: &MlSingleInputSupportLimits) -> &mut Self {
        self.set_linear(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[deprecated = "Use `set_log()` instead."]
    pub fn log(&mut self, val: &MlSingleInputSupportLimits) -> &mut Self {
        self.set_log(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlBinarySupportLimits")]
    #[deprecated = "Use `set_logical_and()` instead."]
    pub fn logical_and(&mut self, val: &MlBinarySupportLimits) -> &mut Self {
        self.set_logical_and(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlLogicalNotSupportLimits")]
    #[deprecated = "Use `set_logical_not()` instead."]
    pub fn logical_not(&mut self, val: &MlLogicalNotSupportLimits) -> &mut Self {
        self.set_logical_not(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlBinarySupportLimits")]
    #[deprecated = "Use `set_logical_or()` instead."]
    pub fn logical_or(&mut self, val: &MlBinarySupportLimits) -> &mut Self {
        self.set_logical_or(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlBinarySupportLimits")]
    #[deprecated = "Use `set_logical_xor()` instead."]
    pub fn logical_xor(&mut self, val: &MlBinarySupportLimits) -> &mut Self {
        self.set_logical_xor(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlLstmSupportLimits")]
    #[deprecated = "Use `set_lstm()` instead."]
    pub fn lstm(&mut self, val: &MlLstmSupportLimits) -> &mut Self {
        self.set_lstm(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlLstmCellSupportLimits")]
    #[deprecated = "Use `set_lstm_cell()` instead."]
    pub fn lstm_cell(&mut self, val: &MlLstmCellSupportLimits) -> &mut Self {
        self.set_lstm_cell(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlBinarySupportLimits")]
    #[deprecated = "Use `set_matmul()` instead."]
    pub fn matmul(&mut self, val: &MlBinarySupportLimits) -> &mut Self {
        self.set_matmul(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlBinarySupportLimits")]
    #[deprecated = "Use `set_max()` instead."]
    pub fn max(&mut self, val: &MlBinarySupportLimits) -> &mut Self {
        self.set_max(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[deprecated = "Use `set_max_pool2d()` instead."]
    pub fn max_pool2d(&mut self, val: &MlSingleInputSupportLimits) -> &mut Self {
        self.set_max_pool2d(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[deprecated = "Use `set_max_tensor_byte_length()` instead."]
    pub fn max_tensor_byte_length(&mut self, val: u32) -> &mut Self {
        self.set_max_tensor_byte_length(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlBinarySupportLimits")]
    #[deprecated = "Use `set_min()` instead."]
    pub fn min(&mut self, val: &MlBinarySupportLimits) -> &mut Self {
        self.set_min(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlBinarySupportLimits")]
    #[deprecated = "Use `set_mul()` instead."]
    pub fn mul(&mut self, val: &MlBinarySupportLimits) -> &mut Self {
        self.set_mul(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[deprecated = "Use `set_neg()` instead."]
    pub fn neg(&mut self, val: &MlSingleInputSupportLimits) -> &mut Self {
        self.set_neg(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlBinarySupportLimits")]
    #[deprecated = "Use `set_not_equal()` instead."]
    pub fn not_equal(&mut self, val: &MlBinarySupportLimits) -> &mut Self {
        self.set_not_equal(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlTensorLimits")]
    #[deprecated = "Use `set_output()` instead."]
    pub fn output(&mut self, val: &MlTensorLimits) -> &mut Self {
        self.set_output(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[deprecated = "Use `set_pad()` instead."]
    pub fn pad(&mut self, val: &MlSingleInputSupportLimits) -> &mut Self {
        self.set_pad(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlBinarySupportLimits")]
    #[deprecated = "Use `set_pow()` instead."]
    pub fn pow(&mut self, val: &MlBinarySupportLimits) -> &mut Self {
        self.set_pow(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlInputOperandLayout")]
    #[deprecated = "Use `set_preferred_input_layout()` instead."]
    pub fn preferred_input_layout(&mut self, val: MlInputOperandLayout) -> &mut Self {
        self.set_preferred_input_layout(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlPreluSupportLimits")]
    #[deprecated = "Use `set_prelu()` instead."]
    pub fn prelu(&mut self, val: &MlPreluSupportLimits) -> &mut Self {
        self.set_prelu(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlQuantizeDequantizeLinearSupportLimits")]
    #[deprecated = "Use `set_quantize_linear()` instead."]
    pub fn quantize_linear(&mut self, val: &MlQuantizeDequantizeLinearSupportLimits) -> &mut Self {
        self.set_quantize_linear(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[deprecated = "Use `set_reciprocal()` instead."]
    pub fn reciprocal(&mut self, val: &MlSingleInputSupportLimits) -> &mut Self {
        self.set_reciprocal(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[deprecated = "Use `set_reduce_l1()` instead."]
    pub fn reduce_l1(&mut self, val: &MlSingleInputSupportLimits) -> &mut Self {
        self.set_reduce_l1(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[deprecated = "Use `set_reduce_l2()` instead."]
    pub fn reduce_l2(&mut self, val: &MlSingleInputSupportLimits) -> &mut Self {
        self.set_reduce_l2(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[deprecated = "Use `set_reduce_log_sum()` instead."]
    pub fn reduce_log_sum(&mut self, val: &MlSingleInputSupportLimits) -> &mut Self {
        self.set_reduce_log_sum(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[deprecated = "Use `set_reduce_log_sum_exp()` instead."]
    pub fn reduce_log_sum_exp(&mut self, val: &MlSingleInputSupportLimits) -> &mut Self {
        self.set_reduce_log_sum_exp(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[deprecated = "Use `set_reduce_max()` instead."]
    pub fn reduce_max(&mut self, val: &MlSingleInputSupportLimits) -> &mut Self {
        self.set_reduce_max(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[deprecated = "Use `set_reduce_mean()` instead."]
    pub fn reduce_mean(&mut self, val: &MlSingleInputSupportLimits) -> &mut Self {
        self.set_reduce_mean(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[deprecated = "Use `set_reduce_min()` instead."]
    pub fn reduce_min(&mut self, val: &MlSingleInputSupportLimits) -> &mut Self {
        self.set_reduce_min(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[deprecated = "Use `set_reduce_product()` instead."]
    pub fn reduce_product(&mut self, val: &MlSingleInputSupportLimits) -> &mut Self {
        self.set_reduce_product(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[deprecated = "Use `set_reduce_sum()` instead."]
    pub fn reduce_sum(&mut self, val: &MlSingleInputSupportLimits) -> &mut Self {
        self.set_reduce_sum(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[deprecated = "Use `set_reduce_sum_square()` instead."]
    pub fn reduce_sum_square(&mut self, val: &MlSingleInputSupportLimits) -> &mut Self {
        self.set_reduce_sum_square(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[deprecated = "Use `set_relu()` instead."]
    pub fn relu(&mut self, val: &MlSingleInputSupportLimits) -> &mut Self {
        self.set_relu(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[deprecated = "Use `set_resample2d()` instead."]
    pub fn resample2d(&mut self, val: &MlSingleInputSupportLimits) -> &mut Self {
        self.set_resample2d(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[deprecated = "Use `set_reshape()` instead."]
    pub fn reshape(&mut self, val: &MlSingleInputSupportLimits) -> &mut Self {
        self.set_reshape(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[deprecated = "Use `set_reverse()` instead."]
    pub fn reverse(&mut self, val: &MlSingleInputSupportLimits) -> &mut Self {
        self.set_reverse(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[deprecated = "Use `set_round_even()` instead."]
    pub fn round_even(&mut self, val: &MlSingleInputSupportLimits) -> &mut Self {
        self.set_round_even(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlScatterSupportLimits")]
    #[deprecated = "Use `set_scatter_elements()` instead."]
    pub fn scatter_elements(&mut self, val: &MlScatterSupportLimits) -> &mut Self {
        self.set_scatter_elements(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlScatterSupportLimits")]
    #[deprecated = "Use `set_scatter_nd()` instead."]
    pub fn scatter_nd(&mut self, val: &MlScatterSupportLimits) -> &mut Self {
        self.set_scatter_nd(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[deprecated = "Use `set_sigmoid()` instead."]
    pub fn sigmoid(&mut self, val: &MlSingleInputSupportLimits) -> &mut Self {
        self.set_sigmoid(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[deprecated = "Use `set_sign()` instead."]
    pub fn sign(&mut self, val: &MlSingleInputSupportLimits) -> &mut Self {
        self.set_sign(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[deprecated = "Use `set_sin()` instead."]
    pub fn sin(&mut self, val: &MlSingleInputSupportLimits) -> &mut Self {
        self.set_sin(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[deprecated = "Use `set_slice()` instead."]
    pub fn slice(&mut self, val: &MlSingleInputSupportLimits) -> &mut Self {
        self.set_slice(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[deprecated = "Use `set_softmax()` instead."]
    pub fn softmax(&mut self, val: &MlSingleInputSupportLimits) -> &mut Self {
        self.set_softmax(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[deprecated = "Use `set_softplus()` instead."]
    pub fn softplus(&mut self, val: &MlSingleInputSupportLimits) -> &mut Self {
        self.set_softplus(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[deprecated = "Use `set_softsign()` instead."]
    pub fn softsign(&mut self, val: &MlSingleInputSupportLimits) -> &mut Self {
        self.set_softsign(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSplitSupportLimits")]
    #[deprecated = "Use `set_split()` instead."]
    pub fn split(&mut self, val: &MlSplitSupportLimits) -> &mut Self {
        self.set_split(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[deprecated = "Use `set_sqrt()` instead."]
    pub fn sqrt(&mut self, val: &MlSingleInputSupportLimits) -> &mut Self {
        self.set_sqrt(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlBinarySupportLimits")]
    #[deprecated = "Use `set_sub()` instead."]
    pub fn sub(&mut self, val: &MlBinarySupportLimits) -> &mut Self {
        self.set_sub(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[deprecated = "Use `set_tan()` instead."]
    pub fn tan(&mut self, val: &MlSingleInputSupportLimits) -> &mut Self {
        self.set_tan(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[deprecated = "Use `set_tanh()` instead."]
    pub fn tanh(&mut self, val: &MlSingleInputSupportLimits) -> &mut Self {
        self.set_tanh(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[deprecated = "Use `set_tile()` instead."]
    pub fn tile(&mut self, val: &MlSingleInputSupportLimits) -> &mut Self {
        self.set_tile(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[deprecated = "Use `set_transpose()` instead."]
    pub fn transpose(&mut self, val: &MlSingleInputSupportLimits) -> &mut Self {
        self.set_transpose(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlSingleInputSupportLimits")]
    #[deprecated = "Use `set_triangular()` instead."]
    pub fn triangular(&mut self, val: &MlSingleInputSupportLimits) -> &mut Self {
        self.set_triangular(val);
        self
    }
    #[cfg(web_sys_unstable_apis)]
    #[cfg(feature = "MlWhereSupportLimits")]
    #[deprecated = "Use `set_where()` instead."]
    pub fn where_(&mut self, val: &MlWhereSupportLimits) -> &mut Self {
        self.set_where(val);
        self
    }
}
#[cfg(web_sys_unstable_apis)]
impl Default for MlOpSupportLimits {
    fn default() -> Self {
        Self::new()
    }
}
