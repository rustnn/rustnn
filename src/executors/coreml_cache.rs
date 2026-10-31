//! Compiled-bundle cache of the `MLGraphBuilder::build` path.
//!
//! `MLModel.compileModel` is the slow step of a CoreML build on small devices (tens of seconds
//! for a language model on an A12 or A13). With `CoremlOptions::compiled_bundle_cache` set,
//! the `.mlmodelc` it produces is copied into `<dir>/<key>/bundle.mlmodelc` next to a
//! `manifest.json`, and the next build of the same graph loads that bundle with
//! `MLModel(contentsOf:configuration:)` instead of compiling. The directory entries come from
//! [`crate::backends::caching::BundleCacheEntry`]; this module computes the key, loads a hit
//! and publishes a miss.
//!
//! The cache is advisory: whatever goes wrong with it is logged at warn level and recorded in
//! [`CacheOutcome`], and the graph is compiled as without a cache.

use std::path::Path;
use std::ptr;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use objc::rc::autoreleasepool;
use objc::runtime::Object;
use objc::{class, msg_send, sel, sel_impl};

use super::load::{CacheOutcome, CoremlLoadRoute, compute_unit_for_device};
use super::{
    CompiledCoremlModel, CoremlModelBacking, CoremlWeightStorage, LoadTrace, ReleaseOnDrop,
    compile_model, has_single_program_output_scope, model_aliases, nsurl_from_path,
    rustnn_coreml_load, shim_error_to_string,
};
use crate::backend_selection::DeviceType;
use crate::backends::caching::{BundleCacheEntry, BundleManifest, copy_directory};
use crate::error::GraphError;
use crate::mlcontextoptions::CoremlFloat16Precision;

/// SeaHash of the CoreML converter sources and the compile shim, written by `build.rs`.
const CONVERTER_SOURCE_HASH: &str =
    include_str!(concat!(env!("OUT_DIR"), "/coreml_source_hash.txt"));

/// Name of the bundle directory inside an entry; CoreML loads only `.mlmodelc` paths.
const BUNDLE_NAME: &str = "bundle.mlmodelc";

/// What the builder asks of the cache for one graph.
#[derive(Debug, Clone)]
pub(crate) struct BundleCacheRequest<'a> {
    /// The cache directory from `CoremlOptions::compiled_bundle_cache`.
    pub(crate) directory: &'a Path,
    /// `GraphInfo::hash_identifier` over operands, operations and every constant's bytes.
    /// The emitted MIL protobuf cannot serve as the key input: its map fields encode from
    /// `HashMap`s, so two conversions of one graph differ in byte order.
    pub(crate) graph_hash: String,
    /// The compile route that produced the bundle, part of the key (`"native"` today).
    pub(crate) route: &'a str,
    /// The float16 lowering the program was emitted with, part of the key.
    pub(crate) float16_precision: CoremlFloat16Precision,
}

/// The inputs of a cache key, named so the manifest can record them and a hit can verify
/// them (a 64-bit key alone would make a collision load another graph's bundle).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CacheKeyInputs {
    graph_hash: String,
    program_len: usize,
    weights_hash: Option<u64>,
    weights_len: usize,
    compute_units: &'static str,
    float16_precision: String,
    route: String,
    crate_version: &'static str,
    converter_source_hash: &'static str,
    os_build: String,
}

impl CacheKeyInputs {
    pub(crate) fn new(
        model_bytes: &[u8],
        weights_data: Option<&[u8]>,
        device_type: DeviceType,
        request: &BundleCacheRequest<'_>,
    ) -> Self {
        Self::with_environment(
            model_bytes,
            weights_data,
            device_type,
            request,
            env!("CARGO_PKG_VERSION"),
            CONVERTER_SOURCE_HASH,
            os_build_string(),
        )
    }

    fn with_environment(
        model_bytes: &[u8],
        weights_data: Option<&[u8]>,
        device_type: DeviceType,
        request: &BundleCacheRequest<'_>,
        crate_version: &'static str,
        converter_source_hash: &'static str,
        os_build: String,
    ) -> Self {
        Self {
            graph_hash: request.graph_hash.clone(),
            program_len: model_bytes.len(),
            weights_hash: weights_data.map(seahash::hash),
            weights_len: weights_data.map_or(0, <[u8]>::len),
            compute_units: compute_unit_for_device(device_type).1,
            float16_precision: format!("{:?}", request.float16_precision),
            route: request.route.to_owned(),
            crate_version,
            converter_source_hash,
            os_build,
        }
    }

    /// The inputs as `(name, value)` pairs, in hashing order; stored in the manifest.
    pub(crate) fn named(&self) -> Vec<(String, String)> {
        let text = |value: &dyn std::fmt::Display| value.to_string();
        vec![
            ("graph_hash".into(), self.graph_hash.clone()),
            ("program_len".into(), text(&self.program_len)),
            (
                "weights_hash".into(),
                self.weights_hash
                    .map_or_else(|| "none".to_owned(), |hash| format!("{hash:016x}")),
            ),
            ("weights_len".into(), text(&self.weights_len)),
            ("compute_units".into(), self.compute_units.to_owned()),
            ("float16_precision".into(), self.float16_precision.clone()),
            ("route".into(), self.route.clone()),
            ("crate_version".into(), self.crate_version.to_owned()),
            (
                "converter_source_hash".into(),
                self.converter_source_hash.to_owned(),
            ),
            ("os_build".into(), self.os_build.clone()),
        ]
    }

    /// The entry key: SeaHash over the named inputs, as 16 hex digits.
    pub(crate) fn key(&self) -> String {
        use std::hash::Hasher;
        let mut hasher = seahash::SeaHasher::new();
        for (name, value) in self.named() {
            hasher.write(name.as_bytes());
            hasher.write_u8(b'=');
            hasher.write(value.as_bytes());
            hasher.write_u8(b'\n');
        }
        format!("{:016x}", hasher.finish())
    }

    fn manifest(&self) -> BundleManifest {
        BundleManifest {
            key_inputs: self.named(),
            crate_version: self.crate_version.to_owned(),
            created_unix_seconds: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |elapsed| elapsed.as_secs()),
            os_build: self.os_build.clone(),
            bundle_bytes: 0,
        }
    }
}

/// The operating system build, as `NSProcessInfo` reports it (for example
/// `Version 27.2 (Build 25C56)`); the target OS name on hosts without Foundation.
fn os_build_string() -> String {
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        autoreleasepool(|| unsafe {
            let info: *mut Object = msg_send![class!(NSProcessInfo), processInfo];
            if info.is_null() {
                return std::env::consts::OS.to_owned();
            }
            let version: *mut Object = msg_send![info, operatingSystemVersionString];
            if version.is_null() {
                return std::env::consts::OS.to_owned();
            }
            super::nsstring_to_string(version)
        })
    }
    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    {
        std::env::consts::OS.to_owned()
    }
}

/// [`compile_model`] with the compiled-bundle cache in front of the URL route.
///
/// Without a request, or for an in-memory load (which produces no bundle), this is
/// `compile_model` and the diagnostics report [`CacheOutcome::Disabled`]. Otherwise a
/// complete entry for the key is loaded (`Hit`); a missing or partial entry is compiled as
/// usual and its bundle published (`Miss`); an entry that cannot be read, fails to load or
/// belongs to another key is removed and rebuilt once, and a failed publish is reported, in
/// both cases as `Error` with the model compiled as without the cache.
pub(crate) fn compile_model_cached(
    model_bytes: Vec<u8>,
    weights_data: Option<Vec<u8>>,
    device_type: DeviceType,
    use_in_memory_asset: bool,
    cache: Option<BundleCacheRequest<'_>>,
) -> Result<CompiledCoremlModel, GraphError> {
    let Some(request) = cache.filter(|_| !use_in_memory_asset) else {
        return compile_model(model_bytes, weights_data, device_type, use_in_memory_asset);
    };
    let inputs = CacheKeyInputs::new(&model_bytes, weights_data.as_deref(), device_type, &request);
    let key = inputs.key();
    let entry = match BundleCacheEntry::new(request.directory, &key, BUNDLE_NAME) {
        Ok(entry) => entry,
        Err(error) => {
            log::warn!("CoreML compiled-bundle cache is unusable: {error}");
            let mut model = compile_model(model_bytes, weights_data, device_type, false)?;
            model.diagnostics.cache = CacheOutcome::Error(format!("cache unusable: {error}"));
            return Ok(model);
        }
    };
    let weights_data = weights_data.map(Arc::new);
    let mut outcome = CacheOutcome::Miss;
    if entry.is_complete() {
        match load_entry(
            &entry,
            &inputs,
            &model_bytes,
            weights_data.as_ref(),
            device_type,
        ) {
            Ok(mut model) => {
                log::debug!(
                    "CoreML compiled-bundle cache hit: {}",
                    entry.dir().display()
                );
                model.diagnostics.cache = CacheOutcome::Hit;
                return Ok(model);
            }
            Err(error) => {
                log::warn!(
                    "cached CoreML bundle {} is unusable and is rebuilt: {error}",
                    entry.dir().display()
                );
                if let Err(remove_error) = entry.remove() {
                    log::warn!(
                        "could not remove cached CoreML bundle {}: {remove_error}",
                        entry.dir().display()
                    );
                }
                outcome = CacheOutcome::Error(format!("cached bundle rebuilt: {error}"));
            }
        }
    } else {
        log::debug!(
            "CoreML compiled-bundle cache miss: {}",
            entry.dir().display()
        );
    }
    // After a failed load the model that borrowed the weights is gone, so the Arc is unique.
    let weights_data = weights_data.map(|weights| {
        Arc::try_unwrap(weights).unwrap_or_else(|shared| shared.as_slice().to_vec())
    });
    let mut model = compile_model(model_bytes, weights_data, device_type, false)?;
    match &model.backing {
        CoremlModelBacking::OnDisk { compiled_dir, .. } => {
            let published = entry.publish(inputs.manifest(), |staged| {
                copy_directory(compiled_dir, staged)
            });
            if let Err(error) = published {
                log::warn!(
                    "could not publish CoreML compiled bundle to {}: {error}",
                    entry.dir().display()
                );
                outcome = CacheOutcome::Error(format!("publish failed: {error}"));
            }
        }
        _ => {
            outcome = CacheOutcome::Error("compiled model has no on-disk bundle".into());
        }
    }
    model.diagnostics.cache = outcome;
    Ok(model)
}

/// Load the bundle of a complete entry after checking that its manifest was written for the
/// same key inputs. The entry outlives the returned model.
fn load_entry(
    entry: &BundleCacheEntry,
    inputs: &CacheKeyInputs,
    model_bytes: &[u8],
    weights_data: Option<&Arc<Vec<u8>>>,
    device_type: DeviceType,
) -> Result<CompiledCoremlModel, GraphError> {
    let manifest = entry
        .manifest()
        .map_err(|error| GraphError::CoremlRuntimeFailed {
            reason: format!("unreadable cache manifest: {error}"),
        })?;
    if manifest.key_inputs != inputs.named() {
        return Err(GraphError::CoremlRuntimeFailed {
            reason: "cache key collision: the manifest records other key inputs".into(),
        });
    }
    let bundle = entry.bundle();
    let route = CoremlLoadRoute::CompiledUrl;
    let mut trace = LoadTrace::new(device_type);
    autoreleasepool(|| unsafe {
        // `fileURLWithPath:` returns an autoreleased URL; the pool releases it.
        let compiled_url = trace.prepare(route, || nsurl_from_path(&bundle))?;
        let (model, name) = trace.policies(route, |code| load_compiled_url(compiled_url, code))?;
        let weights = weights_data.map(|bytes| CoremlWeightStorage::Shared(Arc::clone(bytes)));
        let mut aliases = match model_aliases(model, weights.as_ref()) {
            Ok(aliases) => aliases,
            Err(error) => {
                let _: () = msg_send![model, release];
                return Err(error);
            }
        };
        aliases.compact_constant_storage();
        Ok(CompiledCoremlModel {
            model,
            compute_unit: name,
            diagnostics: trace.finish(route, name),
            aliases: Box::new(aliases),
            has_single_program_output_scope: has_single_program_output_scope(model_bytes),
            backing: CoremlModelBacking::Persistent,
        })
    })
}

/// `MLModel(contentsOf:configuration:)` through the exception shim; the returned model is
/// owned (+1) by the caller.
unsafe fn load_compiled_url(compiled_url: *mut Object, code: i64) -> Result<*mut Object, String> {
    let config: *mut Object = unsafe { msg_send![class!(MLModelConfiguration), new] };
    let _config_guard = ReleaseOnDrop(config);
    let () = unsafe { msg_send![config, setComputeUnits: code] };
    let mut model: *mut Object = ptr::null_mut();
    let mut error = [0u8; 1024];
    let status = unsafe {
        rustnn_coreml_load(
            compiled_url,
            config,
            &mut model,
            error.as_mut_ptr().cast(),
            error.len(),
        )
    };
    if status != 0 || model.is_null() {
        return Err(format!(
            "MLModel load failed: {}",
            shim_error_to_string(&error)
        ));
    }
    Ok(model)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request<'a>(directory: &'a Path, route: &'a str) -> BundleCacheRequest<'a> {
        BundleCacheRequest {
            directory,
            graph_hash: "graph-1".to_owned(),
            route,
            float16_precision: CoremlFloat16Precision::Native,
        }
    }

    fn inputs(
        program: &[u8],
        weights: Option<&[u8]>,
        device: DeviceType,
        request: &BundleCacheRequest<'_>,
        crate_version: &'static str,
        converter_hash: &'static str,
        os_build: &str,
    ) -> CacheKeyInputs {
        CacheKeyInputs::with_environment(
            program,
            weights,
            device,
            request,
            crate_version,
            converter_hash,
            os_build.to_owned(),
        )
    }

    #[test]
    fn the_same_inputs_give_the_same_key() {
        let dir = Path::new("/cache");
        let a = inputs(
            b"program",
            Some(b"weights"),
            DeviceType::Cpu,
            &request(dir, "native"),
            "1.0.0",
            "abc",
            "Build 1",
        );
        let b = inputs(
            b"program",
            Some(b"weights"),
            DeviceType::Cpu,
            &request(Path::new("/elsewhere"), "native"),
            "1.0.0",
            "abc",
            "Build 1",
        );
        assert_eq!(a, b, "the directory is not part of the key");
        assert_eq!(a.key(), b.key());
        assert_eq!(a.key().len(), 16);
        crate::backends::caching::validate_cache_key(&a.key()).unwrap();
        assert_eq!(a.named(), a.manifest().key_inputs);
    }

    #[test]
    fn every_input_changes_the_key() {
        let dir = Path::new("/cache");
        let baseline = inputs(
            b"program",
            Some(b"weights"),
            DeviceType::Cpu,
            &request(dir, "native"),
            "1.0.0",
            "abc",
            "Build 1",
        );
        let exact = BundleCacheRequest {
            float16_precision: CoremlFloat16Precision::Exact,
            ..request(dir, "native")
        };
        let other_graph = BundleCacheRequest {
            graph_hash: "graph-2".to_owned(),
            ..request(dir, "native")
        };
        let variants = [
            (
                "graph hash",
                inputs(
                    b"program",
                    Some(b"weights"),
                    DeviceType::Cpu,
                    &other_graph,
                    "1.0.0",
                    "abc",
                    "Build 1",
                ),
            ),
            (
                "program length",
                inputs(
                    b"programs",
                    Some(b"weights"),
                    DeviceType::Cpu,
                    &request(dir, "native"),
                    "1.0.0",
                    "abc",
                    "Build 1",
                ),
            ),
            (
                "weights bytes",
                inputs(
                    b"program",
                    Some(b"weightz"),
                    DeviceType::Cpu,
                    &request(dir, "native"),
                    "1.0.0",
                    "abc",
                    "Build 1",
                ),
            ),
            (
                "no weights",
                inputs(
                    b"program",
                    None,
                    DeviceType::Cpu,
                    &request(dir, "native"),
                    "1.0.0",
                    "abc",
                    "Build 1",
                ),
            ),
            (
                "compute units",
                inputs(
                    b"program",
                    Some(b"weights"),
                    DeviceType::Gpu,
                    &request(dir, "native"),
                    "1.0.0",
                    "abc",
                    "Build 1",
                ),
            ),
            (
                "float16 precision",
                inputs(
                    b"program",
                    Some(b"weights"),
                    DeviceType::Cpu,
                    &exact,
                    "1.0.0",
                    "abc",
                    "Build 1",
                ),
            ),
            (
                "route",
                inputs(
                    b"program",
                    Some(b"weights"),
                    DeviceType::Cpu,
                    &request(dir, "rust-export"),
                    "1.0.0",
                    "abc",
                    "Build 1",
                ),
            ),
            (
                "crate version",
                inputs(
                    b"program",
                    Some(b"weights"),
                    DeviceType::Cpu,
                    &request(dir, "native"),
                    "1.0.1",
                    "abc",
                    "Build 1",
                ),
            ),
            (
                "converter source hash",
                inputs(
                    b"program",
                    Some(b"weights"),
                    DeviceType::Cpu,
                    &request(dir, "native"),
                    "1.0.0",
                    "abd",
                    "Build 1",
                ),
            ),
            (
                "OS build",
                inputs(
                    b"program",
                    Some(b"weights"),
                    DeviceType::Cpu,
                    &request(dir, "native"),
                    "1.0.0",
                    "abc",
                    "Build 2",
                ),
            ),
        ];
        let mut keys = vec![baseline.key()];
        for (what, variant) in &variants {
            let key = variant.key();
            assert_ne!(key, baseline.key(), "{what} must change the key");
            assert_ne!(
                variant.named(),
                baseline.named(),
                "{what} is in the manifest"
            );
            keys.push(key);
        }
        keys.sort();
        keys.dedup();
        assert_eq!(keys.len(), variants.len() + 1, "all variants are distinct");
    }

    #[test]
    fn the_embedded_environment_is_present() {
        assert_eq!(CONVERTER_SOURCE_HASH.len(), 16, "{CONVERTER_SOURCE_HASH:?}");
        assert!(!os_build_string().is_empty());
        let inputs = CacheKeyInputs::new(
            b"program",
            None,
            DeviceType::Cpu,
            &request(Path::new("/cache"), "native"),
        );
        assert_eq!(inputs.crate_version, env!("CARGO_PKG_VERSION"));
        assert_eq!(inputs.converter_source_hash, CONVERTER_SOURCE_HASH);
    }
}
