//! Legacy one-shot executors for converted models.
//!
//! These functions take the bytes produced by a converter plus host input buffers, load the
//! model, run it once and return host outputs. They predate [`crate::mlcontext::MLContext`]:
//! there are no device tensors and the model is deserialized on every call. They remain for
//! older examples and standalone model execution; new code should
//! use `MLContext` and the backends in [`crate::backends`].

#[cfg(feature = "burn-cpu-runtime")]
pub mod burn;
#[cfg(any(feature = "cann-runtime", feature = "cann-runtime-mock"))]
pub mod cann;
#[cfg(feature = "coreml-runtime")]
pub mod coreml;
#[cfg(feature = "coreml-runtime")]
mod coreml_dtype;
#[cfg(feature = "onnx-runtime")]
pub mod onnx;
