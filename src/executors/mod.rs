#[cfg(any(feature = "cann-runtime", feature = "cann-runtime-mock"))]
pub mod cann_shim;
#[cfg(any(feature = "cann-runtime", feature = "cann-runtime-mock"))]
pub mod cann_shim_types;

#[cfg(feature = "coreml-runtime")]
pub mod coreml;
#[cfg(feature = "onnx-runtime")]
pub mod onnx;
#[cfg(any(feature = "trtx-runtime-mock", feature = "trtx-runtime"))]
pub mod trtx;
