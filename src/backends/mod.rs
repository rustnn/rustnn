//! Execution backends behind [`crate::mlcontext::MLContext`].
//!
//! A backend implements the crate-private `MLBackendContext` and `MLBackendBuilder` traits:
//! it owns device tensors, converts a graph with its converter from [`crate::converters`],
//! compiles it once into an `MLGraph`, and executes `dispatch` calls against persistent
//! tensors. Backends that are not compiled in are replaced by `DisabledContext` aliases so the
//! selection code in [`crate::backend_selection`] type-checks under every feature set.
//!
//! | Module | Feature | Notes |
//! |---|---|---|
//! | `ort` | `onnx-runtime` | ONNX Runtime sessions; lists CPU/GPU/NPU execution-provider devices |
//! | `trtx` | `trtx-runtime`, `trtx-runtime-mock` | TensorRT-RTX engines with refittable weights, engine and runtime caches, CUDA graphs |
//! | `coreml` | `coreml-runtime` | CoreML MLProgram compiled on macOS; failing shims elsewhere |
//! | `litert` | `litert-runtime` | LiteRT interpreter over a TFLite flatbuffer |
//! | `cann` | `cann-runtime`, `cann-runtime-mock` | Huawei HiAI/CANN on OpenHarmony |
//! | `webnn` | `webnn-runtime` | Browser WebNN through generated bindings (`wasm32`) |

use crate::mlcontext::{self, RustNNOptions};

pub mod caching;

#[cfg(feature = "coreml-runtime")]
pub mod coreml;

#[cfg(feature = "onnx-runtime")]
pub mod ort;

#[cfg(any(feature = "trtx-runtime", feature = "trtx-runtime-mock"))]
pub mod trtx;

#[cfg(feature = "litert-runtime")]
pub mod litert;

#[cfg(any(feature = "cann-runtime", feature = "cann-runtime-mock"))]
pub mod cann;

#[cfg(feature = "webnn-runtime")]
pub mod webnn;

#[derive(Debug)]
pub(crate) struct DisabledContext {}

impl DisabledContext {
    // Shared constructor for the device-typed disabled backends (CoreML, LiteRT).
    // Both alias `DisabledContext`, so this must be defined exactly once to avoid
    // duplicate-definition errors when more than one such backend is disabled.
    #[allow(dead_code)]
    pub(crate) fn new_from_device_type(
        _device_type: crate::backend_selection::DeviceType,
        _options: Option<&RustNNOptions>,
    ) -> crate::error::Result<Self> {
        panic!("Tried to create a disabled device-typed backend");
    }
}

impl<'context> mlcontext::MLBackendContext<'context> for DisabledContext {
    fn accelerated(&self) -> bool {
        panic!("RustNN is expected to never use a disabled backend")
    }

    fn create_builder<'builder>(
        &mut self,
    ) -> crate::error::Result<Box<dyn mlcontext::MLBackendBuilder<'context, 'builder> + 'builder>>
    where
        'context: 'builder,
    {
        panic!("RustNN is expected to never use a disabled backend")
    }

    fn create_tensor(
        &mut self,
        _descriptor: &mlcontext::MLTensorDescriptor,
    ) -> crate::error::Result<mlcontext::MLTensor> {
        panic!("RustNN is expected to never use a disabled backend")
    }

    fn create_constant_tensor(
        &mut self,
        _descriptor: &mlcontext::MLTensorDescriptor,
        _input_data: &[u8],
    ) -> crate::error::Result<mlcontext::MLTensor> {
        panic!("RustNN is expected to never use a disabled backend")
    }

    fn read_tensor(
        &mut self,
        _tensor: &mlcontext::MLTensor,
        _array: &mut [u8],
    ) -> crate::error::Result<()> {
        panic!("RustNN is expected to never use a disabled backend")
    }

    fn write_tensor(
        &mut self,
        _tensor: &mlcontext::MLTensor,
        _array: &[u8],
    ) -> crate::error::Result<()> {
        panic!("RustNN is expected to never use a disabled backend")
    }

    fn dispatch(
        &mut self,
        _graph: &mut mlcontext::MLGraph,
        _inputs: &mlcontext::MLNamedTensors,
        _outputs: &mlcontext::MLNamedTensors,
    ) -> crate::error::Result<()> {
        panic!("RustNN is expected to never use a disabled backend")
    }

    fn rustnn_resize_tensor(
        &mut self,
        _tensor: &mut mlcontext::MLTensor,
        _new_shape: &[u64],
    ) -> crate::error::Result<()> {
        panic!("RustNN is expected to never use a disabled backend")
    }

    fn rustnn_set_tensor_capacity(
        &mut self,
        _tensor: &mut mlcontext::MLTensor,
        _max_shape: &[u64],
    ) -> crate::error::Result<()> {
        panic!("RustNN is expected to never use a disabled backend")
    }
}

#[cfg(not(feature = "onnx-runtime"))]
pub mod ort {

    pub(crate) use crate::backends::DisabledContext as OrtContext;
    use crate::mlcontext::RustNNOptions;

    impl OrtContext {
        pub(crate) fn new_from_ep_idx(
            _device_idx: usize,
            _options: Option<&RustNNOptions>,
        ) -> crate::error::Result<Self> {
            panic!("Tried to create disabled ONNX backend");
        }
    }
}

#[cfg(not(any(feature = "trtx-runtime", feature = "trtx-runtime-mock")))]
pub mod trtx {
    pub(crate) use crate::backends::DisabledContext as TrtxContext;
    use crate::mlcontext::RustNNOptions;

    impl TrtxContext {
        pub(crate) fn new(
            _cuda_device_idx: u32,
            _options: Option<&RustNNOptions>,
        ) -> crate::error::Result<Self> {
            panic!("Tried to create disabled Trtx backend");
        }
    }
}

#[cfg(not(any(feature = "cann-runtime", feature = "cann-runtime-mock")))]
pub mod cann {
    pub(crate) use crate::backends::DisabledContext as CannContext;
}

#[cfg(not(feature = "coreml-runtime"))]
pub mod coreml {
    pub(crate) use crate::backends::DisabledContext as CoremlContext;
}
#[cfg(not(feature = "litert-runtime"))]
pub mod litert {
    pub(crate) use crate::backends::DisabledContext as LiteRtContext;
}
