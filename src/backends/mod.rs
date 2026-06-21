use crate::mlcontext;

pub mod caching;

#[cfg(all(target_os = "macos", feature = "coreml-runtime"))]
pub mod coreml;

#[cfg(feature = "onnx-runtime")]
pub mod ort;

#[cfg(any(feature = "trtx-runtime", feature = "trtx-runtime-mock"))]
pub mod trtx;

#[cfg(feature = "litert-runtime")]
pub mod litert;

#[derive(Debug)]
pub(crate) struct DisabledContext {}

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
        _inputs: &std::collections::HashMap<&str, &mlcontext::MLTensor>,
        _outputs: &std::collections::HashMap<&str, &mlcontext::MLTensor>,
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

    impl OrtContext {
        pub(crate) fn new_from_ep_idx(_device_idx: usize) -> crate::error::Result<Self> {
            panic!("Tried to create disabled ONNX backend");
        }
    }
}

#[cfg(not(any(feature = "trtx-runtime", feature = "trtx-runtime-mock")))]
pub mod trtx {
    pub(crate) use crate::backends::DisabledContext as TrtxContext;

    impl TrtxContext {
        pub(crate) fn new(_cuda_device_idx: u32) -> crate::error::Result<Self> {
            panic!("Tried to create disabled Trtx backend");
        }
    }
}

#[cfg(not(all(target_os = "macos", feature = "coreml-runtime")))]
pub mod coreml {
    pub(crate) use crate::backends::DisabledContext as CoremlContext;

    impl CoremlContext {
        pub(crate) fn new_from_device_type(
            _device_type: crate::backend_selection::DeviceType,
        ) -> crate::error::Result<Self> {
            panic!("Tried to create disabled CoreML backend");
        }
    }
}
#[cfg(not(feature = "litert-runtime"))]
pub mod litert {
    pub(crate) use crate::backends::DisabledContext as LiteRtContext;
}
