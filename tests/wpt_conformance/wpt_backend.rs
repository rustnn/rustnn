//! Backend selection for the WPT harness ([`MLContext`] trial runners).

use std::hash::{Hash, Hasher};

use rustnn::backend_selection::Backend;
use rustnn::mlcontext::{MLContext, MLContextOptions, MLPowerPreference};

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
        Self { prefix, options }
    }

    pub fn trial_prefix(&self) -> &'static str {
        self.prefix
    }

    pub fn context_options(&self) -> &MLContextOptions {
        &self.options
    }

    /// Candidate backends for WPT trials (before availability probing).
    pub fn all() -> Vec<Self> {
        vec![
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
        ]
    }

    /// Whether [`MLContext::create`] succeeds for this backend's options.
    pub fn is_available(&self) -> bool {
        MLContext::create(self.context_options()).is_ok()
    }

    pub fn parse_name(s: &str) -> Option<Self> {
        Self::all()
            .into_iter()
            .find(|backend| backend.prefix.eq_ignore_ascii_case(s.trim()))
            .or_else(|| match s.trim().to_ascii_lowercase().as_str() {
                "ort" | "cpu" | "onnx-cpu" | "ort-cpu" => Self::all().into_iter().next(),
                "tensorrt" | "trt" => Self::all().into_iter().find(|b| b.prefix == "trtx"),
                "litert" | "tflite" => Self::all().into_iter().find(|b| b.prefix == "litert"),
                _ => None,
            })
    }

    /// Backends to register as trials. Honors `WPT_BACKEND` when set; only returns backends
    /// that pass [`Self::is_available`].
    pub fn selected() -> Vec<Self> {
        let candidates = if let Ok(raw) = std::env::var("WPT_BACKEND") {
            match Self::parse_name(&raw) {
                Some(backend) => vec![backend],
                None => {
                    eprintln!(
                        "[WPT] warning: invalid WPT_BACKEND={raw} (expected onnx, trtx, or litert); using all backends"
                    );
                    Self::all()
                }
            }
        } else {
            Self::all()
        };

        let mut available = Vec::new();
        for backend in candidates {
            if backend.is_available() {
                available.push(backend);
            } else {
                eprintln!(
                    "[WPT] skipping unavailable backend: {}",
                    backend.trial_prefix()
                );
            }
        }
        available
    }
}
