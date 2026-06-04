use crate::mlcontext;

#[cfg(feature = "onnx-runtime")]
pub mod ort;

#[cfg(any(feature = "trtx-runtime", feature = "trtx-runtime-mock"))]
pub mod trtx;

#[cfg(any(feature = "burn-runtime-cpu", feature = "burn-runtime-webgpu"))]
pub mod burn;

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

macro_rules! impl_disabled_backend_delegate {
    ($context:ty) => {
        impl<'context> crate::mlcontext::MLBackendContext<'context> for $context {
            fn accelerated(&self) -> bool {
                self.0.accelerated()
            }

            fn create_builder<'builder>(
                &mut self,
            ) -> crate::error::Result<
                Box<dyn crate::mlcontext::MLBackendBuilder<'context, 'builder> + 'builder>,
            >
            where
                'context: 'builder,
            {
                self.0.create_builder()
            }

            fn create_tensor(
                &mut self,
                descriptor: &crate::mlcontext::MLTensorDescriptor,
            ) -> crate::error::Result<crate::mlcontext::MLTensor> {
                self.0.create_tensor(descriptor)
            }

            fn create_constant_tensor(
                &mut self,
                descriptor: &crate::mlcontext::MLTensorDescriptor,
                input_data: &[u8],
            ) -> crate::error::Result<crate::mlcontext::MLTensor> {
                self.0.create_constant_tensor(descriptor, input_data)
            }

            fn read_tensor(
                &mut self,
                tensor: &crate::mlcontext::MLTensor,
                array: &mut [u8],
            ) -> crate::error::Result<()> {
                self.0.read_tensor(tensor, array)
            }

            fn write_tensor(
                &mut self,
                tensor: &crate::mlcontext::MLTensor,
                array: &[u8],
            ) -> crate::error::Result<()> {
                self.0.write_tensor(tensor, array)
            }

            fn dispatch(
                &mut self,
                graph: &mut crate::mlcontext::MLGraph,
                inputs: &std::collections::HashMap<&str, &crate::mlcontext::MLTensor>,
                outputs: &std::collections::HashMap<&str, &crate::mlcontext::MLTensor>,
            ) -> crate::error::Result<()> {
                self.0.dispatch(graph, inputs, outputs)
            }

            fn rustnn_resize_tensor(
                &mut self,
                tensor: &mut crate::mlcontext::MLTensor,
                new_shape: &[u64],
            ) -> crate::error::Result<()> {
                self.0.rustnn_resize_tensor(tensor, new_shape)
            }

            fn rustnn_set_tensor_capacity(
                &mut self,
                tensor: &mut crate::mlcontext::MLTensor,
                max_shape: &[u64],
            ) -> crate::error::Result<()> {
                self.0.rustnn_set_tensor_capacity(tensor, max_shape)
            }
        }
    };
}

#[cfg(not(any(feature = "burn-runtime-cpu", feature = "burn-runtime-webgpu")))]
pub mod burn {
    use super::DisabledContext;

    #[derive(Debug)]
    pub(crate) struct BurnContext(DisabledContext);

    impl_disabled_backend_delegate!(BurnContext);

    impl BurnContext {
        pub(crate) fn new(_kind: ()) -> Self {
            panic!("Tried to create disabled Burn backend");
        }
    }
}

#[cfg(not(feature = "onnx-runtime"))]
pub mod ort {
    use super::DisabledContext;

    #[derive(Debug)]
    pub(crate) struct OrtContext(DisabledContext);

    impl_disabled_backend_delegate!(OrtContext);

    impl OrtContext {
        pub(crate) fn new_from_ep_idx(_device_idx: usize) -> crate::error::Result<Self> {
            panic!("Tried to create disabled ONNX backend");
        }
    }
}

#[cfg(not(any(feature = "trtx-runtime", feature = "trtx-runtime-mock")))]
pub mod trtx {
    use super::DisabledContext;

    #[derive(Debug)]
    pub(crate) struct TrtxContext(DisabledContext);

    impl_disabled_backend_delegate!(TrtxContext);

    impl TrtxContext {
        pub(crate) fn new(_cuda_device_idx: u32) -> crate::error::Result<Self> {
            panic!("Tried to create disabled Trtx backend");
        }
    }
}
