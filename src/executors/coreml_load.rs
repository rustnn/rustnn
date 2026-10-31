//! Load-policy diagnostics, independent of Objective-C and prediction.

use crate::backend_selection::DeviceType;
use crate::error::GraphError;

/// The CoreML loading route, not an execution device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoremlLoadRoute {
    /// Load an in-memory `MLModelAsset`.
    InMemoryAsset,
    /// Compile a source URL and load the resulting `.mlmodelc`.
    CompiledUrl,
}

/// An unsuccessful attempt preceding a successful model load.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CoremlLoadFailure {
    /// Route on which the failure occurred.
    pub route: CoremlLoadRoute,
    /// Requested policy for this attempt, or `None` if route preparation failed.
    pub compute_units: Option<&'static str>,
    /// Original CoreML or preparation error; no tensor contents are retained.
    pub reason: String,
}

/// What the compiled-bundle cache did for one load (rustnn extension).
///
/// Enabled with `CoremlOptions::compiled_bundle_cache`. The cache never decides whether a
/// build succeeds: on `Error` the graph was compiled as without the cache.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum CacheOutcome {
    /// No cache directory was configured, or the graph loaded in memory, where no bundle
    /// exists to keep.
    Disabled,
    /// A complete entry for the key was loaded; `MLModel.compileModel` did not run.
    Hit,
    /// No complete entry existed; the graph was compiled and the bundle was published.
    Miss,
    /// The cache was enabled but could not serve or store the entry (an unreadable or
    /// unloadable entry, a key collision, a failed copy or publish). The message names the
    /// step; the graph was compiled as without the cache, and a stale entry was removed.
    Error(String),
}

/// Why and how a graph was successfully loaded by CoreML.
///
/// Compute units are permissions, not measured CPU/GPU/Neural Engine placement.
/// An accelerator-enabled load can still schedule every operation on the CPU.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CoremlLoadDiagnostics {
    /// Policy selected for the requested backend device before any fallback.
    pub requested_compute_units: &'static str,
    /// Policy of the successful load.
    pub loaded_compute_units: &'static str,
    /// Route of the successful load.
    pub route: CoremlLoadRoute,
    /// Earlier failures in attempt order, including successful CPU/URL fallback.
    pub failures: Vec<CoremlLoadFailure>,
    /// What the compiled-bundle cache did; `Disabled` unless the option is set.
    pub cache: CacheOutcome,
}

pub(super) fn compute_unit_for_device(device: DeviceType) -> (i64, &'static str) {
    match device {
        DeviceType::Npu => (3, "CPU_AND_NE"),
        DeviceType::Gpu => (1, "CPU_AND_GPU"),
        DeviceType::Cpu => (0, "CPU_ONLY"),
    }
}

pub(super) struct LoadTrace {
    preferred: (i64, &'static str),
    failures: Vec<CoremlLoadFailure>,
}

impl LoadTrace {
    pub(super) fn new(device: DeviceType) -> Self {
        Self {
            preferred: compute_unit_for_device(device),
            failures: Vec::new(),
        }
    }

    pub(super) fn prepare<T>(
        &mut self,
        route: CoremlLoadRoute,
        prepare: impl FnOnce() -> Result<T, GraphError>,
    ) -> Result<T, GraphError> {
        prepare().inspect_err(|error| {
            self.failures.push(CoremlLoadFailure {
                route,
                compute_units: None,
                reason: error.to_string(),
            });
        })
    }

    pub(super) fn policies<T>(
        &mut self,
        route: CoremlLoadRoute,
        mut load: impl FnMut(i64) -> Result<T, String>,
    ) -> Result<(T, &'static str), GraphError> {
        let candidates = [self.preferred, (0, "CPU_ONLY")];
        let count = if self.preferred.0 == 0 { 1 } else { 2 };
        let mut last_error = String::from("MLModel load failed");
        for &(code, name) in &candidates[..count] {
            match load(code) {
                Ok(value) => return Ok((value, name)),
                Err(reason) => {
                    self.failures.push(CoremlLoadFailure {
                        route,
                        compute_units: Some(name),
                        reason: reason.clone(),
                    });
                    last_error = reason;
                }
            }
        }
        Err(GraphError::CoremlRuntimeFailed { reason: last_error })
    }

    pub(super) fn routes<T>(
        &mut self,
        use_memory: bool,
        mut load: impl FnMut(&mut Self, CoremlLoadRoute) -> Result<T, GraphError>,
    ) -> Result<T, GraphError> {
        if !use_memory {
            return load(self, CoremlLoadRoute::CompiledUrl);
        }
        load(self, CoremlLoadRoute::InMemoryAsset).or_else(|memory_error| {
            load(self, CoremlLoadRoute::CompiledUrl).map_err(|url_error| {
                GraphError::CoremlRuntimeFailed {
                    reason: format!(
                        "in-memory model load failed ({memory_error}); URL fallback failed ({url_error})"
                    ),
                }
            })
        })
    }

    pub(super) fn finish(
        &self,
        route: CoremlLoadRoute,
        loaded_compute_units: &'static str,
    ) -> CoremlLoadDiagnostics {
        let diagnostics = CoremlLoadDiagnostics {
            requested_compute_units: self.preferred.1,
            loaded_compute_units,
            route,
            failures: self.failures.clone(),
            cache: CacheOutcome::Disabled,
        };
        log::debug!("CoreML model load: {diagnostics:?}");
        diagnostics
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simulate(
        device: DeviceType,
        memory: bool,
        successful: (CoremlLoadRoute, i64),
    ) -> (CoremlLoadDiagnostics, Vec<(CoremlLoadRoute, i64)>) {
        let mut trace = LoadTrace::new(device);
        let mut attempts = Vec::new();
        let diagnostics = trace
            .routes(memory, |trace, route| {
                let (_, name) = trace.policies(route, |code| {
                    attempts.push((route, code));
                    if (route, code) == successful {
                        Ok(())
                    } else {
                        Err(format!("rejected {route:?}/{code}"))
                    }
                })?;
                Ok(trace.finish(route, name))
            })
            .unwrap();
        (diagnostics, attempts)
    }

    #[test]
    fn cpu_fallback_retains_original_policy_and_error() {
        for device in [DeviceType::Gpu, DeviceType::Npu] {
            let route = CoremlLoadRoute::InMemoryAsset;
            let (result, attempts) = simulate(device, true, (route, 0));
            let (code, name) = compute_unit_for_device(device);
            assert_eq!(attempts, [(route, code), (route, 0)]);
            assert_eq!(result.requested_compute_units, name);
            assert_eq!(result.loaded_compute_units, "CPU_ONLY");
            assert_eq!(result.failures.len(), 1);
            assert_eq!(result.failures[0].compute_units, Some(name));
            assert_eq!(
                result.failures[0].reason,
                format!("rejected {route:?}/{code}")
            );
        }
    }

    #[test]
    fn url_success_retains_both_memory_failures_in_order() {
        let memory = CoremlLoadRoute::InMemoryAsset;
        let url = CoremlLoadRoute::CompiledUrl;
        let (result, attempts) = simulate(DeviceType::Gpu, true, (url, 1));
        assert_eq!(attempts, [(memory, 1), (memory, 0), (url, 1)]);
        assert_eq!(result.route, url);
        assert_eq!(result.loaded_compute_units, "CPU_AND_GPU");
        assert_eq!(result.failures.len(), 2);
        assert_eq!(result.failures[0].compute_units, Some("CPU_AND_GPU"));
        assert_eq!(result.failures[1].compute_units, Some("CPU_ONLY"));
        assert!(
            result
                .failures
                .iter()
                .all(|failure| failure.route == memory)
        );
    }

    #[test]
    fn url_cpu_success_retains_all_three_earlier_failures() {
        let url = CoremlLoadRoute::CompiledUrl;
        let (result, attempts) = simulate(DeviceType::Gpu, true, (url, 0));
        assert_eq!(attempts.len(), 4);
        assert_eq!(result.failures.len(), 3);
        assert_eq!(result.failures[2].route, url);
        assert_eq!(result.loaded_compute_units, "CPU_ONLY");
    }

    #[test]
    fn direct_success_and_explicit_url_policy_do_not_invent_fallback() {
        for device in [DeviceType::Cpu, DeviceType::Gpu, DeviceType::Npu] {
            for memory in [false, true] {
                let route = if memory {
                    CoremlLoadRoute::InMemoryAsset
                } else {
                    CoremlLoadRoute::CompiledUrl
                };
                let (code, name) = compute_unit_for_device(device);
                let (result, attempts) = simulate(device, memory, (route, code));
                assert_eq!(attempts, [(route, code)]);
                assert_eq!(result.requested_compute_units, name);
                assert_eq!(result.loaded_compute_units, name);
                assert!(result.failures.is_empty());
            }
        }
    }

    #[test]
    fn preparation_error_survives_successful_url_load() {
        let mut trace = LoadTrace::new(DeviceType::Cpu);
        let result = trace
            .routes(true, |trace, route| {
                trace.prepare(route, || {
                    if route == CoremlLoadRoute::InMemoryAsset {
                        Err(GraphError::CoremlRuntimeFailed {
                            reason: "asset unavailable".into(),
                        })
                    } else {
                        Ok(())
                    }
                })?;
                let (_, name) = trace.policies(route, |_| Ok(()))?;
                Ok(trace.finish(route, name))
            })
            .unwrap();
        assert_eq!(result.failures.len(), 1);
        assert_eq!(result.failures[0].compute_units, None);
        assert!(result.failures[0].reason.contains("asset unavailable"));
    }

    #[test]
    fn all_failed_routes_remain_errors_without_duplicate_cpu_attempts() {
        let mut trace = LoadTrace::new(DeviceType::Cpu);
        let mut attempts = Vec::new();
        let error = trace
            .routes(true, |trace, route| {
                trace.policies::<()>(route, |code| {
                    attempts.push((route, code));
                    Err(format!("failed {route:?}"))
                })
            })
            .unwrap_err();
        assert_eq!(attempts.len(), 2);
        assert_eq!(trace.failures.len(), 2);
        assert!(error.to_string().contains("failed InMemoryAsset"));
        assert!(error.to_string().contains("failed CompiledUrl"));
    }
}
