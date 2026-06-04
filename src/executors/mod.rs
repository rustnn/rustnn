#[cfg(any(feature = "burn-runtime-cpu", feature = "burn-runtime-webgpu"))]
pub mod burn;
#[cfg(all(target_os = "macos", feature = "coreml-runtime"))]
pub mod coreml;
#[cfg(feature = "onnx-runtime")]
pub mod onnx;
#[cfg(any(feature = "trtx-runtime-mock", feature = "trtx-runtime"))]
pub mod trtx;
