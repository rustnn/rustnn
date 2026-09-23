// SPDX-FileCopyrightText: 2026 Nvidia
//
// SPDX-License-Identifier: Apache-2

//! Options for [`crate::mlcontext::MLContext::create`].
//!
//! [`MLContextOptions`] carries the two WebNN hints (`powerPreference`, `accelerated`) plus
//! rustnn extensions: a backend or device hint that overrides automatic selection, and
//! [`RustNNOptions`] with per-backend tuning such as [`TrtxOptions`].

use crate::mlcontext::{Backend, BackendDevice};

/// WebNN power preference hint. <https://www.w3.org/TR/webnn/#enumdef-mlpowerpreference>
#[derive(Debug, Default, PartialEq, Eq, Copy, Clone)]
pub enum MLPowerPreference {
    /// No preference; GPU-class devices are tried first when `accelerated` is set.
    #[default]
    Default,
    /// Prefer the fastest device (GPU before NPU).
    HighPerformance,
    /// Prefer the most efficient device (NPU before GPU).
    LowPower,
}

/// <https://www.w3.org/TR/webnn/#dictdef-mlcontextoptions>
/// <https://www.w3.org/TR/webnn/#api-ml>
///
/// From specs: Note: MLContextOptions is under active development, and the design is expected to change,
#[derive(Debug, Eq, PartialEq, Clone)]
pub struct MLContextOptions {
    // WebNN options
    pub(crate) power_preference: MLPowerPreference,
    pub(crate) accelerated: bool,

    // RustNN specific options
    pub(crate) device_hint: Option<BackendDevice>,
    pub(crate) backend_hint: Option<Backend>,
    pub(crate) rustnn_options: RustNNOptions,
}

impl MLContextOptions {
    /// Options with the WebNN hints only; backend selection is automatic.
    pub fn new(power_preference: MLPowerPreference, accelerated: bool) -> Self {
        Self {
            power_preference,
            accelerated,
            device_hint: None,
            backend_hint: None,
            rustnn_options: RustNNOptions::default(),
        }
    }

    /// The `powerPreference` hint.
    pub fn power_preference(&self) -> MLPowerPreference {
        self.power_preference
    }

    /// Set the `powerPreference` hint.
    pub fn set_power_preference(&mut self, power_preference: MLPowerPreference) {
        self.power_preference = power_preference;
    }

    /// The `accelerated` hint: request a GPU or NPU instead of the CPU.
    pub fn accelerated(&self) -> bool {
        self.accelerated
    }

    /// Set the `accelerated` hint.
    pub fn set_accelerated(&mut self, accelerated: bool) {
        self.accelerated = accelerated;
    }

    /// Restrict selection to one backend; creation fails with
    /// [`crate::error::Error::NoBackendAvailableForBackendHint`] if it cannot serve the hints.
    pub fn with_rustnn_backend_hint(mut self, backend: Backend) -> Self {
        self.backend_hint = Some(backend);
        self
    }

    /// Use exactly this device, skipping selection (no availability check, no fallback).
    pub fn with_rustnn_device_hint(mut self, device: BackendDevice) -> Self {
        self.device_hint = Some(device);
        self
    }

    /// Attach backend-specific tuning options.
    pub fn with_rustnn_options(mut self, options: RustNNOptions) -> Self {
        self.rustnn_options = options;
        self
    }
}

/// Backend-specific tuning options (rustnn extension, subject to change).
///
/// The structs are `#[non_exhaustive]`: start from [`RustNNOptions::default`] and set fields.
#[derive(PartialEq, Eq, Clone, Debug, Default)]
#[non_exhaustive]
pub struct RustNNOptions {
    /// CoreML backend options (none yet).
    pub coreml: CoremlOptions,
    /// LiteRT backend options (none yet).
    pub litert: LiteRtOptions,
    /// ONNX Runtime backend options (none yet).
    pub ort: OrtOptions,
    /// TensorRT-RTX backend options.
    pub trtx: TrtxOptions,
}

/// Tuning of the TensorRT-RTX backend; see `docs/integration/tensorrt.md`.
#[derive(PartialEq, Eq, Clone, Debug)]
#[non_exhaustive]
pub struct TrtxOptions {
    /// Caches built engines from GraphInfos to skip compilation for already built engines
    pub engine_caching: bool,
    /// Uses a global runtime cache to speed up engine loading
    pub runtime_cache: bool,
    /// Create a MLContext that does not build TRT engines and only uses already pre-built
    /// AOT engines from cache or user provided. Useful for debugging cache or for AOT only
    /// workflows
    pub fail_on_cache_miss: bool,
    /// Lower CPU overhead using CUDA graphs
    pub cuda_graphs: bool,
}

#[allow(clippy::derivable_impls)]
impl Default for TrtxOptions {
    fn default() -> Self {
        Self {
            engine_caching: true,
            runtime_cache: true,
            cuda_graphs: true,
            fail_on_cache_miss: false,
        }
    }
}

/// LiteRT backend options; no fields yet.
#[derive(PartialEq, Eq, Clone, Debug)]
#[non_exhaustive]
pub struct LiteRtOptions {}

#[allow(clippy::derivable_impls)]
impl Default for LiteRtOptions {
    fn default() -> Self {
        Self {}
    }
}

/// ONNX Runtime backend options; no fields yet.
#[derive(PartialEq, Eq, Clone, Debug)]
#[non_exhaustive]
pub struct OrtOptions {}

#[allow(clippy::derivable_impls)]
impl Default for OrtOptions {
    fn default() -> Self {
        Self {}
    }
}

/// CoreML backend options; no fields yet.
#[derive(PartialEq, Eq, Clone, Debug)]
#[non_exhaustive]
pub struct CoremlOptions {}

#[allow(clippy::derivable_impls)]
impl Default for CoremlOptions {
    fn default() -> Self {
        Self {}
    }
}
