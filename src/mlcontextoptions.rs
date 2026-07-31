// SPDX-FileCopyrightText: 2026 Nvidia
//
// SPDX-License-Identifier: Apache-2

use crate::mlcontext::{Backend, BackendDevice};

#[derive(Debug, Default, PartialEq, Eq, Copy, Clone)]
pub enum MLPowerPreference {
    #[default]
    Default,
    HighPerformance,
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
    pub fn new(power_preference: MLPowerPreference, accelerated: bool) -> Self {
        Self {
            power_preference,
            accelerated,
            device_hint: None,
            backend_hint: None,
            rustnn_options: RustNNOptions::default(),
        }
    }

    pub fn power_preference(&self) -> MLPowerPreference {
        self.power_preference
    }

    pub fn set_power_preference(&mut self, power_preference: MLPowerPreference) {
        self.power_preference = power_preference;
    }

    pub fn accelerated(&self) -> bool {
        self.accelerated
    }

    pub fn set_accelerated(&mut self, accelerated: bool) {
        self.accelerated = accelerated;
    }

    pub fn with_rustnn_backend_hint(mut self, backend: Backend) -> Self {
        self.backend_hint = Some(backend);
        self
    }

    pub fn with_rustnn_device_hint(mut self, device: BackendDevice) -> Self {
        self.device_hint = Some(device);
        self
    }

    pub fn with_rustnn_options(mut self, options: RustNNOptions) -> Self {
        self.rustnn_options = options;
        self
    }
}

/// Options that steer backend or RustNN internals, experiments
/// Could be replaced later by a proper API for RustNN options, internals and for backends
#[derive(PartialEq, Eq, Clone, Debug, Default)]
#[non_exhaustive]
pub struct RustNNOptions {
    pub coreml: CoremlOptions,
    pub litert: LiteRtOptions,
    pub ort: OrtOptions,
    pub trtx: TrtxOptions,
}

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

#[derive(PartialEq, Eq, Clone, Debug)]
#[non_exhaustive]
pub struct LiteRtOptions {}

#[allow(clippy::derivable_impls)]
impl Default for LiteRtOptions {
    fn default() -> Self {
        Self {}
    }
}

#[derive(PartialEq, Eq, Clone, Debug)]
#[non_exhaustive]
pub struct OrtOptions {}

#[allow(clippy::derivable_impls)]
impl Default for OrtOptions {
    fn default() -> Self {
        Self {}
    }
}

#[derive(PartialEq, Eq, Clone, Debug)]
#[non_exhaustive]
pub struct CoremlOptions {}

#[allow(clippy::derivable_impls)]
impl Default for CoremlOptions {
    fn default() -> Self {
        Self {}
    }
}
