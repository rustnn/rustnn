#[cfg(feature = "onnx-runtime")]
pub mod ort;

#[cfg(any(feature = "trtx-runtime", feature = "trtx-runtime-mock"))]
pub mod trtx;

#[cfg(not(feature = "onnx-runtime"))]
pub mod ort {
    use crate::mlcontext;

    #[derive(Debug)]
    pub(crate) struct OrtContext {}

    impl OrtContext {
        pub(crate) fn new_from_ep_idx(_device_idx: usize) -> crate::error::Result<Self> {
            panic!("Tried to create disabled ONNX backend");
        }
    }
    impl<'context> mlcontext::MLBackendContext<'context> for OrtContext {
        fn accelerated(&self) -> bool {
            todo!()
        }

        fn create_builder(
            &mut self,
        ) -> crate::error::Result<Box<dyn mlcontext::MLBackendBuilder<'context> + 'context>>
        {
            todo!()
        }

        fn create_tensor(
            &mut self,
            _descriptor: &mlcontext::MLTensorDescriptor,
        ) -> crate::error::Result<mlcontext::MLTensor> {
            todo!()
        }

        fn create_constant_tensor(
            &mut self,
            _descriptor: &mlcontext::MLTensorDescriptor,
            _input_data: &[u8],
        ) -> crate::error::Result<mlcontext::MLTensor> {
            todo!()
        }

        fn read_tensor(
            &mut self,
            _tensor: &mlcontext::MLTensor,
            _array: &mut [u8],
        ) -> crate::error::Result<()> {
            todo!()
        }

        fn write_tensor(
            &mut self,
            _tensor: &mlcontext::MLTensor,
            _array: &[u8],
        ) -> crate::error::Result<()> {
            todo!()
        }

        fn dispatch(
            &mut self,
            _graph: &mut mlcontext::MLGraph,
            _inputs: &std::collections::HashMap<&str, &mlcontext::MLTensor>,
            _outputs: &std::collections::HashMap<&str, &mlcontext::MLTensor>,
        ) -> crate::error::Result<()> {
            todo!()
        }
    }
}

#[cfg(not(any(feature = "trtx-runtime", feature = "trtx-runtime-mock")))]
pub mod trtx {
    use crate::mlcontext;
    #[derive(Debug)]
    pub(crate) struct TrtxContext {}

    impl TrtxContext {
        pub(crate) fn new(_cuda_device_idx: u32) -> crate::error::Result<Self> {
            panic!("Tried to create disabled Trtx backend");
        }
    }
    impl<'context> mlcontext::MLBackendContext<'context> for TrtxContext {
        fn accelerated(&self) -> bool {
            todo!()
        }

        fn create_builder(
            &mut self,
        ) -> crate::error::Result<Box<dyn mlcontext::MLBackendBuilder<'context> + 'context>>
        {
            todo!()
        }

        fn create_tensor(
            &mut self,
            _descriptor: &mlcontext::MLTensorDescriptor,
        ) -> crate::error::Result<mlcontext::MLTensor> {
            todo!()
        }

        fn create_constant_tensor(
            &mut self,
            _descriptor: &mlcontext::MLTensorDescriptor,
            _input_data: &[u8],
        ) -> crate::error::Result<mlcontext::MLTensor> {
            todo!()
        }

        fn read_tensor(
            &mut self,
            _tensor: &mlcontext::MLTensor,
            _array: &mut [u8],
        ) -> crate::error::Result<()> {
            todo!()
        }

        fn write_tensor(
            &mut self,
            _tensor: &mlcontext::MLTensor,
            _array: &[u8],
        ) -> crate::error::Result<()> {
            todo!()
        }

        fn dispatch(
            &mut self,
            _graph: &mut mlcontext::MLGraph,
            _inputs: &std::collections::HashMap<&str, &mlcontext::MLTensor>,
            _outputs: &std::collections::HashMap<&str, &mlcontext::MLTensor>,
        ) -> crate::error::Result<()> {
            todo!()
        }
    }
}
