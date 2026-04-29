use crate::{
    error::Result,
    executors::trtx::TrtxContext,
    mlcontext::{GpuDevice, ListDevices, MLContextOptions, MLPowerPreference},
};

// this is a concept of pywebnn
#[derive(Debug, Eq, PartialEq, Copy, Clone)]
pub(crate) enum DeviceType {
    Cpu,
    Gpu,
    Npu,
}

/// we currently only consider internal backends,
/// might allow to register external backends in future
/// like with converter registry
#[derive(Debug, Eq, PartialEq, Copy, Clone)]
#[allow(dead_code)]
pub(crate) enum BackendDevice {
    OnnxDevice {
        //ep_device_idx: u64,
        device_type: DeviceType,
    },
    TrtxDevice {
        cuda_device_idx: u32,
    },
    CoremlDevice {
        //device_idx: u64,
        device_type: DeviceType,
    },
    WebNN,
    ExternalBackend,
}

impl BackendDevice {
    #[allow(dead_code)]
    pub(crate) fn as_trtx_device(&self) -> Option<&u32> {
        if let Self::TrtxDevice { cuda_device_idx } = self {
            Some(cuda_device_idx)
        } else {
            None
        }
    }
}

// TODO: pywebnn has device_type, and we could have backend preference for user to overwrite
// autoselection
pub(crate) fn select_backend(options: &MLContextOptions) -> Result<BackendDevice> {
    let have_trtx = cfg!(any(feature = "trtx-runtime", feature = "trtx-runtime-mock"));
    let want_trtx = true;
    let trtx_devices = TrtxContext::list_devices();

    let have_onnx = cfg!(feature = "onnx-runtime");
    let want_onnx = true;

    let have_coreml = cfg!(all(target_os = "macos", feature = "coreml-runtime"));
    let want_coreml = false;

    // not merged yet
    //let have_webnn = cfg!(all(feature = "web", target_arch = "wasm32"));
    let have_webnn = false;
    let want_webnn = true;

    // TODO: also check whether WebNN is available
    //#[cfg(target_arch = "wasm32")]
    //{
    //let window = window().expect("no global `window` exists");
    //let navigator = window.navigator();
    //let ml = navigator.ml();
    //}

    Ok(match (options.power_preference, options.accelerated) {
        // Trtx
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
            BackendDevice::CoremlDevice {
                device_type: DeviceType::Gpu,
            }
        }
        (MLPowerPreference::LowPower, true) if have_coreml && want_coreml => {
            BackendDevice::CoremlDevice {
                device_type: DeviceType::Npu,
            }
        }
        (_, false) if have_coreml && want_coreml => BackendDevice::CoremlDevice {
            device_type: DeviceType::Cpu,
        },
        // ORT
        (MLPowerPreference::Default | MLPowerPreference::HighPerformance, true)
            if have_onnx && want_onnx =>
        {
            BackendDevice::OnnxDevice {
                device_type: DeviceType::Gpu,
            }
        }
        (MLPowerPreference::LowPower, true) if have_onnx && want_onnx => {
            BackendDevice::OnnxDevice {
                device_type: DeviceType::Npu,
            }
        }
        (_, false) if have_onnx && want_onnx => BackendDevice::CoremlDevice {
            device_type: DeviceType::Cpu,
        },
        // WebNN
        (_, _) if have_webnn && want_webnn => BackendDevice::WebNN,
        _ => return Err(crate::error::Error::NoBackendAvialable),
    })
}

pub(crate) fn select_backend_by_gpu(_gpu_device: &GpuDevice) -> Result<BackendDevice> {
    todo!()
}
