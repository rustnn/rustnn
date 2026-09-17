//! Backend and device selection for [`crate::mlcontext::MLContext::create`].
//!
//! Selection follows the
//! [WebNN device selection explainer](https://github.com/webmachinelearning/webnn/blob/main/device-selection-explainer.md):
//! the caller passes hints and the implementation picks the device. rustnn resolves the hints
//! in this order, skipping backends that are not compiled in or report no device:
//!
//! 1. An explicit device hint ([`crate::mlcontext::MLContextOptions::with_rustnn_device_hint`])
//!    is used as is.
//! 2. `accelerated` with `Default` or `HighPerformance` power preference: TensorRT-RTX (first
//!    CUDA device), then CoreML (GPU), then LiteRT (GPU), then ONNX Runtime (GPU, then NPU).
//! 3. `accelerated` with `LowPower`: CoreML (Neural Engine), then LiteRT (NPU), then ONNX
//!    Runtime (NPU).
//! 4. Not accelerated: CoreML (CPU), then LiteRT (CPU), then ONNX Runtime (CPU).
//! 5. CANN is only selected when requested with a backend hint.
//!
//! ONNX Runtime CPU also serves as the last resort for accelerated requests. When nothing
//! matches, creation fails with [`crate::error::Error::NoBackendAvailable`] (or the
//! `ForBackendHint` variant), which lists the wanted and compiled backends.

#[cfg(feature = "onnx-runtime")]
use crate::executors::onnx::ensure_ort_initialized;
use crate::{
    error::Result,
    mlcontext::{GpuDevice, MLContextOptions, MLPowerPreference},
};

#[cfg(any(feature = "cann-runtime", feature = "cann-runtime-mock"))]
use crate::backends::cann::CannContext;
#[cfg(feature = "litert-runtime")]
use crate::backends::litert::LiteRtContext;
#[cfg(feature = "onnx-runtime")]
use crate::backends::ort::OrtContext;
#[cfg(any(feature = "trtx-runtime", feature = "trtx-runtime-mock"))]
use crate::backends::trtx::TrtxContext;

#[allow(unused_imports)]
use crate::mlcontext::ListDevices;

/// Device class of a [`BackendDevice`].
// this is a concept of pywebnn
#[derive(Debug, Eq, PartialEq, Copy, Clone)]
pub enum DeviceType {
    Cpu,
    Gpu,
    Npu,
}

/// Execution backend. Each variant is compiled in by the matching Cargo feature
/// (`onnx-runtime`, `trtx-runtime`, `coreml-runtime`, `litert-runtime`, `cann-runtime`).
#[derive(Debug, Eq, PartialEq, Copy, Clone)]
pub enum Backend {
    Onnx,
    Trtx,
    Coreml,
    Litert,
    Cann,
}

/// A concrete device of a [`Backend`], as returned by
/// [`crate::mlcontext::MLContext::rustnn_device`] or passed as a device hint.
///
/// Only internal backends are represented; registering external backends (as the converter
/// registry allows) may come later.
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
    Cann {
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
            BackendDevice::Cann { .. } => Backend::Cann,
        }
    }

    pub fn device_type(&self) -> DeviceType {
        match self {
            BackendDevice::Trtx { .. } => DeviceType::Gpu,
            BackendDevice::Onnx { device_type, .. }
            | BackendDevice::Coreml { device_type }
            | BackendDevice::LiteRt { device_type }
            | BackendDevice::Cann { device_type } => *device_type,
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
    let have_trtx = cfg!(any(feature = "trtx-runtime", feature = "trtx-runtime-mock"));
    let want_trtx = options.backend_hint.is_none() || options.backend_hint == Some(Backend::Trtx);
    #[cfg(any(feature = "trtx-runtime", feature = "trtx-runtime-mock"))]
    let trtx_devices = TrtxContext::list_devices();

    #[cfg(any(feature = "cann-runtime", feature = "cann-runtime-mock"))]
    let have_cann = cfg!(any(feature = "cann-runtime", feature = "cann-runtime-mock"));
    #[cfg(not(any(feature = "cann-runtime", feature = "cann-runtime-mock")))]
    let have_cann = false;
    let want_cann = options.backend_hint == Some(Backend::Cann);
    #[cfg(any(feature = "cann-runtime", feature = "cann-runtime-mock"))]
    let cann_devices = CannContext::list_devices();

    let have_onnx = cfg!(feature = "onnx-runtime");
    let want_onnx = options.backend_hint.is_none() || options.backend_hint == Some(Backend::Onnx);

    let have_coreml = cfg!(all(target_os = "macos", feature = "coreml-runtime"));
    let want_coreml =
        options.backend_hint.is_none() || options.backend_hint == Some(Backend::Coreml);

    let have_litert = cfg!(feature = "litert-runtime");
    let want_litert =
        options.backend_hint.is_none() || options.backend_hint == Some(Backend::Litert);
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

        // CANN (Huawei Ascend NPU)
        #[cfg(any(feature = "cann-runtime", feature = "cann-runtime-mock"))]
        (_, _)
            if have_cann
                && want_cann
                && let [first, ..] = cann_devices.as_slice() =>
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
        (_, _) if let Some(backend_hint) = options.backend_hint => {
            return Err(crate::error::Error::NoBackendAvailableForBackendHint {
                backend_hint,
                want_trtx,
                have_trtx,
                want_onnx,
                have_onnx,
                want_coreml,
                have_coreml,
                want_litert,
                have_litert,
                want_cann,
                have_cann,
            });
        }
        _ => {
            return Err(crate::error::Error::NoBackendAvailable {
                want_trtx,
                have_trtx,
                want_onnx,
                have_onnx,
                want_coreml,
                have_coreml,
                want_litert,
                have_litert,
                want_cann,
                have_cann,
            });
        }
    })
}

pub(crate) fn select_backend_by_gpu(_gpu_device: &GpuDevice) -> Result<BackendDevice> {
    todo!()
}
