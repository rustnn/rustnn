#[cfg(feature = "onnx-runtime")]
use crate::executors::onnx::ensure_ort_initialized;
use crate::{
    error::Result,
    mlcontext::{GpuDevice, MLContextOptions, MLPowerPreference},
};

#[cfg(feature = "litert-runtime")]
use crate::backends::litert::LiteRtContext;
#[cfg(feature = "onnx-runtime")]
use crate::backends::ort::OrtContext;
#[cfg(any(feature = "trtx-runtime", feature = "trtx-runtime-mock"))]
use crate::backends::trtx::TrtxContext;

#[allow(unused_imports)]
use crate::mlcontext::ListDevices;

// this is a concept of pywebnn
#[derive(Debug, Eq, PartialEq, Copy, Clone)]
pub enum DeviceType {
    Cpu,
    Gpu,
    Npu,
}

#[cfg(feature = "onnx-runtime")]
impl From<ort::memory::DeviceType> for DeviceType {
    fn from(value: ort::memory::DeviceType) -> Self {
        match value {
            ort::memory::DeviceType::CPU => Self::Cpu,
            ort::memory::DeviceType::GPU => Self::Gpu,
            ort::memory::DeviceType::NPU => Self::Gpu,
        }
    }
}

#[derive(Debug, Eq, PartialEq, Copy, Clone)]
pub enum Backend {
    Onnx,
    Trtx,
    Coreml,
    Litert,
}

/// we currently only consider internal backends,
/// might allow to register external backends in future
/// like with converter registry
#[derive(Debug, Eq, PartialEq, Copy, Clone)]
pub enum BackendDevice {
    Onnx {
        ep_device_idx: usize,
        device_type: DeviceType,
    },
    Trtx {
        cuda_device_idx: u32,
    },
    Coreml {
        //device_idx: u64,
        device_type: DeviceType,
    },
    LiteRt {
        device_type: DeviceType,
    },
    //WebNN {
    //options: MLContextOptions,
    //},
    //ExternalBackend,
}

impl BackendDevice {
    pub fn backend(&self) -> Backend {
        match self {
            BackendDevice::Onnx { .. } => Backend::Onnx,
            BackendDevice::Trtx { .. } => Backend::Trtx,
            BackendDevice::Coreml { .. } => Backend::Coreml,
            BackendDevice::LiteRt { .. } => Backend::Litert,
        }
    }

    pub fn device_type(&self) -> DeviceType {
        match self {
            BackendDevice::Trtx { .. } => DeviceType::Gpu,
            BackendDevice::Onnx { device_type, .. }
            | BackendDevice::Coreml { device_type }
            | BackendDevice::LiteRt { device_type } => *device_type,
        }
    }

    pub fn is_npu(&self) -> bool {
        self.device_type() == DeviceType::Npu
    }

    pub fn is_gpu(&self) -> bool {
        self.device_type() == DeviceType::Gpu
    }

    pub fn is_cpu(&self) -> bool {
        self.device_type() == DeviceType::Cpu
    }

    #[cfg(feature = "trtx-runtime")]
    #[allow(dead_code)]
    pub(crate) fn as_trtx_device(&self) -> Option<&u32> {
        if let Self::Trtx { cuda_device_idx } = self {
            Some(cuda_device_idx)
        } else {
            None
        }
    }
}

pub(crate) fn select_backend(options: &MLContextOptions) -> Result<BackendDevice> {
    #[cfg(any(feature = "trtx-runtime", feature = "trtx-runtime-mock"))]
    let have_trtx = cfg!(any(feature = "trtx-runtime", feature = "trtx-runtime-mock"));
    #[cfg(any(feature = "trtx-runtime", feature = "trtx-runtime-mock"))]
    let want_trtx = options.backend_hint.is_none() || options.backend_hint == Some(Backend::Trtx);
    #[cfg(any(feature = "trtx-runtime", feature = "trtx-runtime-mock"))]
    let trtx_devices = TrtxContext::list_devices();

    #[cfg(feature = "onnx-runtime")]
    let have_onnx = cfg!(feature = "onnx-runtime");
    #[cfg(feature = "onnx-runtime")]
    let want_onnx = options.backend_hint.is_none() || options.backend_hint == Some(Backend::Onnx);

    let have_coreml = cfg!(all(target_os = "macos", feature = "coreml-runtime"));
    let want_coreml =
        options.backend_hint.is_none() || options.backend_hint == Some(Backend::Coreml);

    #[cfg(feature = "litert-runtime")]
    let have_litert = cfg!(feature = "litert-runtime");
    #[cfg(feature = "litert-runtime")]
    let want_litert = false;
    #[cfg(feature = "litert-runtime")]
    let litert_devices = LiteRtContext::list_devices();

    if let Some(device_hint) = options.device_hint {
        // No fallbacks for now. We could check if device is available
        return Ok(device_hint);
    }

    Ok(match (options.power_preference, options.accelerated) {
        // Trtx
        #[cfg(any(feature = "trtx-runtime", feature = "trtx-runtime-mock"))]
        (MLPowerPreference::Default | MLPowerPreference::HighPerformance, true)
            if have_trtx
                && want_trtx
                && let [first, ..] = trtx_devices.as_slice()
                && trtx::dynamically_load_tensorrt(None::<String>).is_ok() =>
        {
            *first
        }

        // CoreML
        (MLPowerPreference::Default | MLPowerPreference::HighPerformance, true)
            if have_coreml && want_coreml =>
        {
            BackendDevice::Coreml {
                device_type: DeviceType::Gpu,
            }
        }
        (MLPowerPreference::LowPower, true) if have_coreml && want_coreml => {
            BackendDevice::Coreml {
                device_type: DeviceType::Npu,
            }
        }
        (_, false) if have_coreml && want_coreml => BackendDevice::Coreml {
            device_type: DeviceType::Cpu,
        },
        // LiteRT
        #[cfg(feature = "litert-runtime")]
        (MLPowerPreference::Default | MLPowerPreference::HighPerformance, true)
            if have_litert
                && want_litert
                && let Some(first) = litert_devices.iter().find(|device| device.is_gpu()) =>
        {
            *first
        }
        #[cfg(feature = "litert-runtime")]
        (MLPowerPreference::LowPower, true)
            if have_litert
                && want_litert
                && let Some(first) = litert_devices.iter().find(|device| device.is_npu()) =>
        {
            *first
        }
        #[cfg(feature = "litert-runtime")]
        (_, false)
            if have_litert
                && want_litert
                && let Some(first) = litert_devices.iter().find(|device| device.is_cpu()) =>
        {
            *first
        }
        // ORT
        #[cfg(feature = "onnx-runtime")]
        (MLPowerPreference::Default | MLPowerPreference::HighPerformance, true)
            if have_onnx
                && want_onnx
                && ensure_ort_initialized().is_ok()
                && let Some(first) = OrtContext::list_devices().iter().find(|d| d.is_gpu()) =>
        {
            *first
        }
        #[cfg(feature = "onnx-runtime")]
        (MLPowerPreference::Default | MLPowerPreference::LowPower, true)
            if have_onnx
                && want_onnx
                && ensure_ort_initialized().is_ok()
                && let Some(first) = OrtContext::list_devices().iter().find(|d| d.is_npu()) =>
        {
            *first
        }
        // TODO: confirm whether we are allowed to return CPU, if user wanted accelerated (IRC
        // chrome did have this behavior in browser)
        #[cfg(feature = "onnx-runtime")]
        (_, _)
            if have_onnx
                && want_onnx
                && ensure_ort_initialized().is_ok()
                && let Some(first) = OrtContext::list_devices().iter().find(|d| d.is_cpu()) =>
        {
            *first
        }
        _ => return Err(crate::error::Error::NoBackendAvailable),
    })
}

pub(crate) fn select_backend_by_gpu(_gpu_device: &GpuDevice) -> Result<BackendDevice> {
    todo!()
}
