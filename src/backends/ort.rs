use std::sync::Arc;

use log::debug;
use ort::{device::Device, environment::Environment, memory::DeviceType, sys::OrtHardwareDevice};

use crate::{
    backend_selection::BackendDevice,
    error::Error,
    executors::onnx::ensure_ort_initialized,
    mlcontext::{ListDevices, MLBackendContext},
};

pub(crate) struct OrtContext {
    env: Arc<Environment>,
    device_idx: usize,
}

impl OrtContext {
    pub(crate) fn new_from_ty(
        device_type: crate::backend_selection::DeviceType,
    ) -> crate::error::Result<Self> {
        let env =
            Environment::current().map_err(|e| Error::ContextCreationError { source: e.into() })?;
        //env.devices() has wrong lifetimes, the device should have the lifetime from env, but not hold on
        //the borrow

        let selected = env
            .devices()
            .inspect(|d| {
                debug!(
                    "Saw ONNX device {:?}",
                    &(&d.vendor(), &d.ep_vendor(), &d.id(), &d.ty(),)
                )
            })
            .position(|d| device_type == d.ty().into())
            .ok_or(Error::NoDeviceAvailable)?;

        Ok(Self {
            env,
            device_idx: selected,
        })
    }
}

impl std::fmt::Debug for OrtContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let device = self.env.devices().nth(self.device_idx).unwrap();
        f.debug_struct("OrtContext")
            .field(
                "device",
                &(
                    &device.vendor(),
                    &device.ep_vendor(),
                    &device.id(),
                    &device.ty(),
                ),
            )
            //.field("ep_device", &self.ep_device)
            //.field("hw_device", &self.hw_device)
            .finish()
    }
}

impl ListDevices for OrtContext {
    fn list_devices() -> Vec<crate::backend_selection::BackendDevice> {
        if ensure_ort_initialized().is_err() {
            return vec![];
        }

        let Ok(env) =
            Environment::current().map_err(|e| Error::ContextCreationError { source: e.into() })
        else {
            return vec![];
        };

        let mut rtn = vec![];
        for d in env.devices() {
            debug!(
                "Saw ONNX device {:?}",
                &(&d.vendor(), &d.ep_vendor(), &d.id(), &d.ty(),)
            );
            rtn.push(BackendDevice::OnnxDevice {
                device_type: d.ty().into(),
            })
        }

        rtn
    }
}

//let env = Environment::current()?;
//let _ = env.register_ep_library("CUDA", "/path/to/onnxruntime_providers_cuda.dll");

impl<'context> MLBackendContext<'context> for OrtContext {
    fn accelerated(&self) -> bool {
        let device = self.env.devices().nth(self.device_idx).unwrap();
        device.ty() != DeviceType::CPU
    }

    fn create_builder(
        &mut self,
    ) -> crate::error::Result<Box<dyn crate::mlcontext::MLBackendBuilder<'context> + 'context>>
    {
        todo!()
    }

    fn create_tensor(
        &mut self,
        descriptor: &crate::mlcontext::MLTensorDescriptor,
    ) -> crate::error::Result<crate::mlcontext::MLTensor> {
        todo!()
    }

    fn create_constant_tensor(
        &mut self,
        descriptor: &crate::mlcontext::MLTensorDescriptor,
        input_data: &[u8],
    ) -> crate::error::Result<crate::mlcontext::MLTensor> {
        todo!()
    }

    fn read_tensor(
        &mut self,
        tensor: &crate::mlcontext::MLTensor,
        array: &mut [u8],
    ) -> crate::error::Result<()> {
        todo!()
    }

    fn write_tensor(
        &mut self,
        tensor: &crate::mlcontext::MLTensor,
        array: &[u8],
    ) -> crate::error::Result<()> {
        todo!()
    }

    fn dispatch(
        &mut self,
        graph: &mut crate::mlcontext::MLGraph,
        inputs: &std::collections::HashMap<&str, crate::mlcontext::MLTensor>,
        outputs: &std::collections::HashMap<&str, crate::mlcontext::MLTensor>,
    ) -> crate::error::Result<()> {
        todo!()
    }
}
