//! Backend selection for the WPT harness ([`MLContext`] trial runners).

use rustnn::mlcontext::{MLContextOptions, MLPowerPreference};

/// Execution backend for WPT conformance trials.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WptBackend {
    #[cfg(feature = "onnx-runtime")]
    /// ONNX Runtime CPU (`accelerated = false`).
    OnnxCpu,
    #[cfg(feature = "onnx-runtime")]
    /// ONNX Runtime GPU when available (`accelerated = true`, high performance).
    OnnxGpu,
    #[cfg(any(feature = "trtx-runtime", feature = "trtx-runtime-mock"))]
    /// TensorRT via TRTX (`accelerated = true`).
    Trtx,
}

impl WptBackend {
    pub fn trial_prefix(self) -> &'static str {
        match self {
            #[cfg(feature = "onnx-runtime")]
            Self::OnnxCpu => "onnx",
            #[cfg(feature = "onnx-runtime")]
            Self::OnnxGpu => "onnx-gpu",
            #[cfg(any(feature = "trtx-runtime", feature = "trtx-runtime-mock"))]
            Self::Trtx => "trtx",
        }
    }

    pub fn context_options(self) -> MLContextOptions {
        match self {
            #[cfg(feature = "onnx-runtime")]
            Self::OnnxCpu => MLContextOptions::new(MLPowerPreference::Default, false),
            #[cfg(feature = "onnx-runtime")]
            Self::OnnxGpu => MLContextOptions::new(MLPowerPreference::HighPerformance, true),
            #[cfg(any(feature = "trtx-runtime", feature = "trtx-runtime-mock"))]
            Self::Trtx => MLContextOptions::new(MLPowerPreference::HighPerformance, true),
        }
    }

    pub fn available() -> Vec<Self> {
        let mut backends = Vec::new();
        #[cfg(feature = "onnx-runtime")]
        backends.push(Self::OnnxCpu);
        #[cfg(any(feature = "trtx-runtime", feature = "trtx-runtime-mock"))]
        backends.push(Self::Trtx);
        backends
    }

    pub fn parse_name(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            #[cfg(feature = "onnx-runtime")]
            "onnx" | "ort" | "cpu" | "onnx-cpu" | "ort-cpu" => Some(Self::OnnxCpu),
            #[cfg(feature = "onnx-runtime")]
            "onnx-gpu" | "ort-gpu" | "gpu" => Some(Self::OnnxGpu),
            #[cfg(any(feature = "trtx-runtime", feature = "trtx-runtime-mock"))]
            "trtx" | "tensorrt" | "trt" => Some(Self::Trtx),
            _ => None,
        }
    }

    /// Backends to register as trials. Honors `WPT_BACKEND` when set.
    pub fn selected() -> Vec<Self> {
        let available = Self::available();
        #[cfg(feature = "onnx-runtime")]
        let selectable: Vec<Self> = {
            let mut v = available.clone();
            if !v.contains(&Self::OnnxGpu) {
                v.push(Self::OnnxGpu);
            }
            v
        };
        #[cfg(not(feature = "onnx-runtime"))]
        let selectable = available;

        if let Ok(raw) = std::env::var("WPT_BACKEND") {
            if let Some(backend) = Self::parse_name(&raw) {
                if selectable.contains(&backend) {
                    return vec![backend];
                }
                eprintln!(
                    "[WPT] warning: WPT_BACKEND={} is not available with current features; using all enabled backends",
                    raw
                );
            } else {
                eprintln!(
                    "[WPT] warning: invalid WPT_BACKEND={} (expected onnx, onnx-gpu, or trtx); using all enabled backends",
                    raw
                );
            }
        }
        available
    }
}
