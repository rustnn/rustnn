//! Backend selection for the WPT harness ([`MLContext`] trial runners).

use std::hash::{Hash, Hasher};

use rustnn::backend_selection::{Backend, BackendDevice, DeviceType};
use rustnn::mlcontext::{MLContext, MLContextOptions, MLPowerPreference, RustNNOptions};

/// One WPT trial backend: a stable name prefix plus [`MLContextOptions`] with backend hints.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WptBackend {
    prefix: &'static str,
    options: MLContextOptions,
}

impl Hash for WptBackend {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.prefix.hash(state);
    }
}

impl WptBackend {
    fn new(prefix: &'static str, options: MLContextOptions) -> Self {
        // The pinned WPT corpus includes packed 4-bit quantization extensions.
        Self {
            prefix,
            options: options.with_rustnn_allow_int4_uint4(true),
        }
    }

    pub fn trial_prefix(&self) -> &'static str {
        self.prefix
    }

    pub fn context_options(&self) -> &MLContextOptions {
        &self.options
    }

    fn with_coreml_device(mut self, device_type: DeviceType) -> Self {
        self.options.set_accelerated(device_type != DeviceType::Cpu);
        self.options = self
            .options
            .with_rustnn_device_hint(BackendDevice::Coreml { device_type });
        self
    }

    /// Candidate backends for WPT trials (before availability probing).
    pub fn all() -> Vec<Self> {
        let mut coreml = RustNNOptions::default();
        match std::env::var("WPT_COREML_TENSOR_MODE").as_deref() {
            Err(_) | Ok("baseline") => {}
            Ok("persistent") => {
                coreml.coreml.reuse_tensor_storage = true;
                coreml.coreml.output_backings = false;
            }
            Ok("backings") => {
                coreml.coreml.reuse_tensor_storage = true;
                coreml.coreml.output_backings = true;
            }
            Ok(mode) => panic!(
                "invalid WPT_COREML_TENSOR_MODE={mode}; expected baseline, persistent or backings"
            ),
        }
        coreml.coreml.float16_precision = super::wpt_config::coreml_float16_precision();
        coreml.coreml.compile_route = super::wpt_config::coreml_compile_route();
        coreml.coreml.compiled_bundle_cache = super::wpt_config::coreml_bundle_cache();
        vec![
            Self::new(
                "burn-cpu",
                MLContextOptions::new(MLPowerPreference::Default, false)
                    .with_rustnn_backend_hint(Backend::BurnCpu),
            ),
            Self::new(
                "onnx",
                MLContextOptions::new(MLPowerPreference::Default, false)
                    .with_rustnn_backend_hint(Backend::Onnx),
            ),
            Self::new(
                "trtx",
                MLContextOptions::new(MLPowerPreference::HighPerformance, true)
                    .with_rustnn_backend_hint(Backend::Trtx),
            ),
            Self::new(
                "litert",
                MLContextOptions::new(MLPowerPreference::Default, false)
                    .with_rustnn_backend_hint(Backend::Litert),
            ),
            // CoreML (macOS only). Filtered out by `probe` unless the binary is built
            // with `coreml-runtime` on macOS, so this is a no-op on other platforms/builds.
            // `accelerated = false` picks CoreML's CPU device for deterministic results.
            Self::new(
                "coreml",
                MLContextOptions::new(MLPowerPreference::Default, false)
                    .with_rustnn_backend_hint(Backend::Coreml)
                    .with_rustnn_options(coreml),
            ),
            // CANN/HiAI (Huawei Ascend NPU). Filtered out by `probe` unless the
            // binary is built with `cann-runtime`; availability is still probed at startup
            // (requires a device).
            Self::new(
                "cann",
                MLContextOptions::new(MLPowerPreference::Default, true)
                    .with_rustnn_backend_hint(Backend::Cann),
            ),
        ]
    }

    /// Creates an [`MLContext`] for this backend's options; the error says why the backend
    /// cannot run (not compiled in, or its runtime failed to start).
    pub fn probe(&self) -> Result<(), rustnn::error::Error> {
        MLContext::create(self.context_options()).map(drop)
    }

    pub fn parse_name(s: &str) -> Option<Self> {
        Self::all()
            .into_iter()
            .find(|backend| backend.prefix.eq_ignore_ascii_case(s.trim()))
            .or_else(|| match s.trim().to_ascii_lowercase().as_str() {
                "ort" | "cpu" | "onnx-cpu" | "ort-cpu" => Self::all().into_iter().next(),
                "tensorrt" | "trt" => Self::all().into_iter().find(|b| b.prefix == "trtx"),
                "litert" | "tflite" => Self::all().into_iter().find(|b| b.prefix == "litert"),
                "core-ml" | "mlprogram" => Self::all().into_iter().find(|b| b.prefix == "coreml"),
                "cann" => Self::all().into_iter().find(|b| b.prefix == "cann"),
                _ => None,
            })
    }

    /// Backends to register as trials. Honors `WPT_BACKEND` when set; only returns backends
    /// that pass [`Self::probe`].
    pub fn selected() -> Vec<Self> {
        let requested = std::env::var("WPT_BACKEND")
            .ok()
            .and_then(|raw| match Self::parse_name(&raw) {
                Some(backend) => Some(backend),
                None => {
                    eprintln!(
                        "[WPT] warning: invalid WPT_BACKEND={raw} (expected onnx, trtx, coreml, litert, cann or burn-cpu); using all backends"
                    );
                    None
                }
            });
        let explicit = requested.is_some();
        let candidates = requested.map_or_else(Self::all, |backend| vec![backend]);

        let mut available = Vec::new();
        for mut backend in candidates {
            if backend.prefix == "coreml"
                && cfg!(all(feature = "coreml-runtime", target_os = "macos"))
            {
                let requested = super::wpt_config::coreml_requested_device();
                let device_type = match requested {
                    "cpu" => DeviceType::Cpu,
                    "gpu" => DeviceType::Gpu,
                    "npu" => DeviceType::Npu,
                    _ => unreachable!("validated CoreML policy"),
                };
                backend = backend.with_coreml_device(device_type);
                eprintln!("[WPT] CoreML requested device: {requested} (not measured placement)");
            }
            match backend.probe() {
                Ok(()) => available.push(backend),
                // A backend the run asked for by name must say why it cannot start; without
                // `WPT_BACKEND`, backends that are not compiled in are expected to drop out.
                Err(error) if explicit => eprintln!(
                    "[WPT] backend {} is unavailable: {error}",
                    backend.trial_prefix()
                ),
                Err(error) => log::warn!(
                    "[WPT] skipping unavailable backend {}: {error}",
                    backend.trial_prefix()
                ),
            }
        }
        available
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn coreml_policy_preserves_each_tensor_storage_mode() {
        use super::*;

        for mode in ["persistent", "backings", "baseline"] {
            let mut tuning = RustNNOptions::default();
            if mode != "baseline" {
                tuning.coreml.reuse_tensor_storage = true;
                tuning.coreml.output_backings = mode == "backings";
            }
            for device_type in [DeviceType::Cpu, DeviceType::Gpu, DeviceType::Npu] {
                let backend = WptBackend::new(
                    "coreml",
                    MLContextOptions::new(MLPowerPreference::Default, false)
                        .with_rustnn_backend_hint(Backend::Coreml)
                        .with_rustnn_options(tuning.clone()),
                )
                .with_coreml_device(device_type);
                let expected = MLContextOptions::new(
                    MLPowerPreference::Default,
                    device_type != DeviceType::Cpu,
                )
                .with_rustnn_backend_hint(Backend::Coreml)
                .with_rustnn_options(tuning.clone())
                .with_rustnn_allow_int4_uint4(true)
                .with_rustnn_device_hint(BackendDevice::Coreml { device_type });
                assert_eq!(
                    backend.context_options(),
                    &expected,
                    "{mode} storage under {device_type:?} permission"
                );
            }
        }
    }
}
