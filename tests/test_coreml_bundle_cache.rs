//! The compiled-bundle cache serves the second build of a graph from disk, survives graph
//! drops, rebuilds unusable entries once and never fails a build.
#![cfg(all(target_os = "macos", feature = "coreml-runtime"))]

use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};

use rustnn::backend_selection::{BackendDevice, DeviceType};
use rustnn::executors::coreml::{CacheOutcome, CoremlLoadDiagnostics, CoremlLoadRoute};
use rustnn::mlcontext::{
    Backend, CoremlFloat16Precision, LoadDiagnostics, MLContext, MLContextOptions, MLGraph,
    MLGraphBuilder, MLNamedOperands, MLNamedTensors, MLOperandDescriptor, MLPowerPreference,
    MLTensorDescriptor, RustNNOptions,
};
use rustnn::operator_enums::MLOperandDataType;

const INPUT: [f32; 4] = [1.0, -2.0, 3.5, -0.25];
const BIAS: [f32; 4] = [0.5, 0.5, -4.0, 0.0];

fn context<'a>(cache: Option<&Path>, precision: CoremlFloat16Precision) -> MLContext<'a> {
    context_on(cache, precision, DeviceType::Cpu)
}

fn context_on<'a>(
    cache: Option<&Path>,
    precision: CoremlFloat16Precision,
    device_type: DeviceType,
) -> MLContext<'a> {
    let mut options = RustNNOptions::default();
    options.coreml.compiled_bundle_cache = cache.map(Path::to_path_buf);
    options.coreml.float16_precision = precision;
    MLContext::create(
        &MLContextOptions::new(MLPowerPreference::Default, device_type != DeviceType::Cpu)
            .with_rustnn_backend_hint(Backend::Coreml)
            .with_rustnn_device_hint(BackendDevice::Coreml { device_type })
            .with_rustnn_options(options),
    )
    .unwrap()
}

fn diagnostics(graph: &MLGraph<'_>) -> CoremlLoadDiagnostics {
    let Some(LoadDiagnostics::Coreml(diagnostics)) = graph.rustnn_load_diagnostics() else {
        panic!("CoreML graph did not report CoreML load diagnostics");
    };
    diagnostics
}

/// `relu(input + bias)` with a non-scalar float32 constant, so the program has a weights
/// blob and the entry records a weights hash.
fn build<'a>(context: &mut MLContext<'a>) -> (MLGraph<'a>, CoremlLoadDiagnostics) {
    build_with_bias(context, &BIAS)
}

fn build_with_bias<'a>(
    context: &mut MLContext<'a>,
    bias: &[f32; 4],
) -> (MLGraph<'a>, CoremlLoadDiagnostics) {
    let mut builder = MLGraphBuilder::new(context).unwrap();
    let descriptor = MLOperandDescriptor::new(MLOperandDataType::Float32, vec![4]);
    let input = builder.input("input", &descriptor).unwrap();
    let bias = builder.constant_from_slice(&descriptor, bias).unwrap();
    let sum = builder.add(input, bias).unwrap();
    let output = builder.relu(sum).unwrap();
    let graph = builder
        .build(&MLNamedOperands::from([("output", output)]))
        .unwrap();
    let diagnostics = diagnostics(&graph);
    (graph, diagnostics)
}

fn run(context: &mut MLContext<'_>, graph: &mut MLGraph<'_>) -> [f32; 4] {
    let descriptor = MLTensorDescriptor::new(MLOperandDataType::Float32, vec![4])
        .to_readable()
        .to_writable();
    let input = context.create_tensor(&descriptor).unwrap();
    let output = context.create_tensor(&descriptor).unwrap();
    context.write_tensor(&input, &INPUT).unwrap();
    context
        .dispatch(
            graph,
            &MLNamedTensors::from([("input", &input)]),
            &MLNamedTensors::from([("output", &output)]),
        )
        .unwrap();
    let mut actual = [0f32; 4];
    context.read_tensor(&output, &mut actual).unwrap();
    actual
}

fn expected() -> [f32; 4] {
    let mut values = [0f32; 4];
    for (value, (input, bias)) in values.iter_mut().zip(INPUT.iter().zip(BIAS.iter())) {
        *value = (input + bias).max(0.0);
    }
    values
}

/// The one complete entry of the cache directory.
fn single_entry(cache: &Path) -> PathBuf {
    let names: Vec<_> = std::fs::read_dir(cache)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    assert_eq!(names.len(), 1, "exactly one entry, no staging: {names:?}");
    assert!(
        !names[0].contains(".partial-"),
        "no staging directory left: {names:?}"
    );
    let entry = cache.join(&names[0]);
    assert!(entry.join("manifest.json").is_file(), "{entry:?}");
    assert!(entry.join("bundle.mlmodelc").is_dir(), "{entry:?}");
    entry
}

#[test]
fn the_second_build_loads_the_cached_bundle_and_computes_the_same_values() {
    let cache = tempfile::tempdir().unwrap();
    let mut context = context(Some(cache.path()), CoremlFloat16Precision::Native);

    let (mut first, diagnostics) = build(&mut context);
    assert_eq!(diagnostics.cache, CacheOutcome::Miss, "{diagnostics:?}");
    assert_eq!(diagnostics.route, CoremlLoadRoute::CompiledUrl);
    assert_eq!(run(&mut context, &mut first), expected());

    let entry = single_entry(cache.path());
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(entry.join("manifest.json")).unwrap()).unwrap();
    assert_eq!(manifest["crate_version"], env!("CARGO_PKG_VERSION"));
    assert!(manifest["bundle_bytes"].as_u64().unwrap() > 0, "{manifest}");
    assert!(manifest["created_unix_seconds"].as_u64().unwrap() > 0);
    assert!(!manifest["os_build"].as_str().unwrap().is_empty());
    let key_inputs = manifest["key_inputs"].as_array().unwrap();
    let input = |name: &str| {
        key_inputs
            .iter()
            .find(|pair| pair[0] == name)
            .unwrap_or_else(|| panic!("{name} missing from {manifest}"))[1]
            .as_str()
            .unwrap()
            .to_owned()
    };
    assert_eq!(input("route"), "native");
    assert_eq!(input("compute_units"), "CPU_ONLY");
    assert_eq!(input("float16_precision"), "Native");
    assert!(input("graph_hash").ends_with("_coreml"), "{manifest}");
    assert_ne!(
        input("weights_hash"),
        "none",
        "the bias constant is a weight"
    );
    assert_eq!(input("converter_source_hash").len(), 16);
    let entry_inode = std::fs::metadata(&entry).unwrap().ino();
    let manifest_bytes = std::fs::read(entry.join("manifest.json")).unwrap();

    // The second build of the same graph reports a hit and leaves the entry as it was.
    // That alone does not show that nothing was compiled, because a publish never replaces
    // a complete entry; `a_hit_runs_the_bundle_on_disk` shows it.
    let (mut second, diagnostics) = build(&mut context);
    assert_eq!(diagnostics.cache, CacheOutcome::Hit, "{diagnostics:?}");
    assert_eq!(diagnostics.route, CoremlLoadRoute::CompiledUrl);
    assert_eq!(diagnostics.loaded_compute_units, "CPU_ONLY");
    assert!(diagnostics.failures.is_empty(), "{diagnostics:?}");
    assert_eq!(run(&mut context, &mut second), expected());
    assert_eq!(single_entry(cache.path()), entry);
    assert_eq!(std::fs::metadata(&entry).unwrap().ino(), entry_inode);
    assert_eq!(
        std::fs::read(entry.join("manifest.json")).unwrap(),
        manifest_bytes
    );

    // The entry belongs to the cache directory, not to the graphs that used it.
    drop(first);
    drop(second);
    assert_eq!(single_entry(cache.path()), entry);
    assert_eq!(std::fs::metadata(&entry).unwrap().ino(), entry_inode);

    // The same graph in another context with the same cache directory hits as well.
    let mut other = context_with_same_cache(cache.path());
    let (mut third, diagnostics) = build(&mut other);
    assert_eq!(diagnostics.cache, CacheOutcome::Hit, "{diagnostics:?}");
    assert_eq!(run(&mut other, &mut third), expected());

    // Without the option, nothing changes: in-memory load, cache reported as disabled.
    let mut plain = self::context(None, CoremlFloat16Precision::Native);
    let (mut uncached, diagnostics) = build(&mut plain);
    assert_eq!(diagnostics.cache, CacheOutcome::Disabled, "{diagnostics:?}");
    assert_eq!(diagnostics.route, CoremlLoadRoute::InMemoryAsset);
    assert_eq!(run(&mut plain, &mut uncached), expected());
}

fn context_with_same_cache<'a>(cache: &Path) -> MLContext<'a> {
    context(Some(cache), CoremlFloat16Precision::Native)
}

/// A hit must run the bundle that is on disk. The entry of one graph gets the compiled
/// bundle of another (same shapes, another bias) while its manifest stays: the next build
/// of the first graph then computes the other graph's values, which a build that compiled
/// its own program could not.
#[test]
fn a_hit_runs_the_bundle_on_disk() {
    const OTHER_BIAS: [f32; 4] = [8.0, 8.0, 8.0, 8.0];
    let cache = tempfile::tempdir().unwrap();
    let mut context = context(Some(cache.path()), CoremlFloat16Precision::Native);
    let (mut graph, diagnostics) = build(&mut context);
    assert_eq!(diagnostics.cache, CacheOutcome::Miss, "{diagnostics:?}");
    assert_eq!(run(&mut context, &mut graph), expected());
    drop(graph);
    let entry = single_entry(cache.path());

    let other_cache = tempfile::tempdir().unwrap();
    let mut other = self::context(Some(other_cache.path()), CoremlFloat16Precision::Native);
    let (other_graph, diagnostics) = build_with_bias(&mut other, &OTHER_BIAS);
    assert_eq!(diagnostics.cache, CacheOutcome::Miss, "{diagnostics:?}");
    drop(other_graph);
    let other_entry = single_entry(other_cache.path());
    std::fs::remove_dir_all(entry.join("bundle.mlmodelc")).unwrap();
    std::fs::rename(
        other_entry.join("bundle.mlmodelc"),
        entry.join("bundle.mlmodelc"),
    )
    .unwrap();

    let (mut graph, diagnostics) = build(&mut context);
    assert_eq!(diagnostics.cache, CacheOutcome::Hit, "{diagnostics:?}");
    let swapped: Vec<f32> = INPUT
        .iter()
        .zip(OTHER_BIAS)
        .map(|(input, bias)| (input + bias).max(0.0))
        .collect();
    assert_ne!(swapped, expected());
    assert_eq!(run(&mut context, &mut graph).to_vec(), swapped);
}

/// The weights blob is a key input, so it has to be the same bytes on every conversion of
/// one graph. With several constants that depends on the order the converter emits them.
#[test]
fn a_graph_with_many_constants_misses_once_and_then_hits() {
    let cache = tempfile::tempdir().unwrap();
    let outcomes: Vec<_> = (0..6)
        .map(|_| {
            let mut context = context(Some(cache.path()), CoremlFloat16Precision::Native);
            let mut builder = MLGraphBuilder::new(&mut context).unwrap();
            let descriptor = MLOperandDescriptor::new(MLOperandDataType::Float32, vec![64]);
            let mut sum = builder.input("input", &descriptor).unwrap();
            for constant in 0..8 {
                let values: Vec<f32> = (0..64).map(|i| (constant * 64 + i) as f32).collect();
                let operand = builder.constant_from_slice(&descriptor, &values).unwrap();
                sum = builder.add(sum, operand).unwrap();
            }
            let graph = builder
                .build(&MLNamedOperands::from([("output", sum)]))
                .unwrap();
            diagnostics(&graph).cache
        })
        .collect();
    assert_eq!(outcomes[0], CacheOutcome::Miss);
    assert!(
        outcomes[1..]
            .iter()
            .all(|outcome| *outcome == CacheOutcome::Hit),
        "{outcomes:?}"
    );
    single_entry(cache.path());
}

#[test]
fn each_float16_lowering_and_policy_has_its_own_entry() {
    let cache = tempfile::tempdir().unwrap();
    let configurations = [
        (CoremlFloat16Precision::Native, DeviceType::Cpu),
        (CoremlFloat16Precision::Exact, DeviceType::Cpu),
        (CoremlFloat16Precision::Native, DeviceType::Gpu),
        (CoremlFloat16Precision::Native, DeviceType::Npu),
    ];
    for (count, &(precision, device)) in configurations.iter().enumerate() {
        let mut context = context_on(Some(cache.path()), precision, device);
        let (_, diagnostics) = build(&mut context);
        assert_eq!(
            diagnostics.cache,
            CacheOutcome::Miss,
            "{precision:?} {device:?}: {diagnostics:?}"
        );
        let entries = std::fs::read_dir(cache.path()).unwrap().count();
        assert_eq!(entries, count + 1, "one entry per lowering and policy");
    }
    for (precision, device) in configurations {
        let mut context = context_on(Some(cache.path()), precision, device);
        let (_, diagnostics) = build(&mut context);
        assert_eq!(
            diagnostics.cache,
            CacheOutcome::Hit,
            "{precision:?} {device:?}: {diagnostics:?}"
        );
    }
    assert_eq!(std::fs::read_dir(cache.path()).unwrap().count(), 4);
}

#[test]
fn partial_corrupt_and_foreign_entries_are_rebuilt_once() {
    let cache = tempfile::tempdir().unwrap();
    let mut context = context(Some(cache.path()), CoremlFloat16Precision::Native);
    let (_, diagnostics) = build(&mut context);
    assert_eq!(diagnostics.cache, CacheOutcome::Miss);
    let entry = single_entry(cache.path());
    let manifest = std::fs::read(entry.join("manifest.json")).unwrap();

    // A partial entry (a writer that died before the manifest) is a miss and is replaced.
    std::fs::remove_file(entry.join("manifest.json")).unwrap();
    let (mut graph, diagnostics) = build(&mut context);
    assert_eq!(diagnostics.cache, CacheOutcome::Miss, "{diagnostics:?}");
    assert_eq!(single_entry(cache.path()), entry);
    assert_eq!(run(&mut context, &mut graph), expected());

    // A complete entry whose bundle does not load is removed, rebuilt and then hits again.
    std::fs::remove_dir_all(entry.join("bundle.mlmodelc")).unwrap();
    std::fs::create_dir(entry.join("bundle.mlmodelc")).unwrap();
    let (mut graph, diagnostics) = build(&mut context);
    let CacheOutcome::Error(reason) = &diagnostics.cache else {
        panic!("an unloadable entry is reported: {diagnostics:?}");
    };
    assert!(reason.contains("rebuilt"), "{reason}");
    assert_eq!(single_entry(cache.path()), entry);
    assert!(
        entry
            .join("bundle.mlmodelc")
            .join("coremldata.bin")
            .exists()
    );
    assert_eq!(run(&mut context, &mut graph), expected());
    let (_, diagnostics) = build(&mut context);
    assert_eq!(diagnostics.cache, CacheOutcome::Hit, "{diagnostics:?}");

    // A complete entry written for other key inputs (a key collision) is never loaded.
    let foreign = String::from_utf8(manifest)
        .unwrap()
        .replace("\"native\"", "\"other-route\"");
    assert!(foreign.contains("other-route"), "{foreign}");
    std::fs::write(entry.join("manifest.json"), foreign).unwrap();
    let (mut graph, diagnostics) = build(&mut context);
    let CacheOutcome::Error(reason) = &diagnostics.cache else {
        panic!("a collision is reported: {diagnostics:?}");
    };
    assert!(reason.contains("collision"), "{reason}");
    assert_eq!(run(&mut context, &mut graph), expected());
    let (_, diagnostics) = build(&mut context);
    assert_eq!(diagnostics.cache, CacheOutcome::Hit, "{diagnostics:?}");
}

#[test]
fn an_unwritable_cache_directory_does_not_fail_the_build() {
    use std::os::unix::fs::PermissionsExt;
    let parent = tempfile::tempdir().unwrap();
    std::fs::set_permissions(parent.path(), std::fs::Permissions::from_mode(0o555)).unwrap();
    let cache = parent.path().join("cache");
    let mut context = context(Some(&cache), CoremlFloat16Precision::Native);
    let (mut graph, diagnostics) = build(&mut context);
    std::fs::set_permissions(parent.path(), std::fs::Permissions::from_mode(0o755)).unwrap();
    let CacheOutcome::Error(reason) = &diagnostics.cache else {
        panic!("a failed publish is reported: {diagnostics:?}");
    };
    assert!(reason.contains("publish failed"), "{reason}");
    assert_eq!(diagnostics.route, CoremlLoadRoute::CompiledUrl);
    assert_eq!(run(&mut context, &mut graph), expected());
    assert!(!cache.exists());
}

/// Input-free programs of proven constant copies keep the in-memory route (the URL compiler
/// can crash in BNNS on them), so there is no bundle to cache.
#[test]
fn input_free_constant_copies_bypass_the_cache() {
    let cache = tempfile::tempdir().unwrap();
    let mut context = context(Some(cache.path()), CoremlFloat16Precision::Native);
    let mut builder = MLGraphBuilder::new(&mut context).unwrap();
    let descriptor = MLOperandDescriptor::new(MLOperandDataType::Float32, vec![4]);
    let constant = builder.constant_from_slice(&descriptor, &BIAS).unwrap();
    let output = builder.identity(constant).unwrap();
    let graph = builder
        .build(&MLNamedOperands::from([("output", output)]))
        .unwrap();
    let diagnostics = diagnostics(&graph);
    assert_eq!(diagnostics.route, CoremlLoadRoute::InMemoryAsset);
    assert_eq!(diagnostics.cache, CacheOutcome::Disabled, "{diagnostics:?}");
    assert_eq!(std::fs::read_dir(cache.path()).unwrap().count(), 0);
}

/// An input-free program that is not a constant copy keeps its route as well: through the
/// URL compiler, iOS and tvOS 26 and 27 return 0 for this saturating uint8 clamp.
#[test]
fn input_free_programs_keep_their_route() {
    use rustnn::operator_options::MLClampOptions;
    let cache = tempfile::tempdir().unwrap();
    for cached in [false, true] {
        let mut context = context(
            cached.then_some(cache.path()),
            CoremlFloat16Precision::Native,
        );
        let mut builder = MLGraphBuilder::new(&mut context).unwrap();
        let descriptor = MLOperandDescriptor::new(MLOperandDataType::Uint8, vec![4]);
        let constant = builder
            .constant_from_slice(&descriptor, &[0u8, 100, 200, 255])
            .unwrap();
        let options = MLClampOptions {
            min_value: Some(serde_json::json!(150)),
            max_value: Some(serde_json::json!(255)),
            ..Default::default()
        };
        let output = builder.clamp_with_options(constant, options).unwrap();
        let graph = builder
            .build(&MLNamedOperands::from([("output", output)]))
            .unwrap();
        let diagnostics = diagnostics(&graph);
        assert_eq!(
            diagnostics.route,
            CoremlLoadRoute::InMemoryAsset,
            "{cached}"
        );
        assert_eq!(diagnostics.cache, CacheOutcome::Disabled, "{cached}");
    }
    assert_eq!(std::fs::read_dir(cache.path()).unwrap().count(), 0);
}

/// Under the native lowering a float16 graph keeps the in-memory route, whose values the
/// URL compiler does not always reproduce; the exact lowering compiles from a URL with or
/// without the cache, so there the graph is cached.
#[test]
fn native_float16_graphs_keep_their_route_and_exact_ones_are_cached() {
    let cache = tempfile::tempdir().unwrap();
    let build = |precision| {
        let mut context = context(Some(cache.path()), precision);
        let mut builder = MLGraphBuilder::new(&mut context).unwrap();
        let descriptor = MLOperandDescriptor::new(MLOperandDataType::Float16, vec![4]);
        let input = builder.input("input", &descriptor).unwrap();
        let output = builder.relu(input).unwrap();
        let graph = builder
            .build(&MLNamedOperands::from([("output", output)]))
            .unwrap();
        diagnostics(&graph)
    };
    let native = build(CoremlFloat16Precision::Native);
    assert_eq!(native.route, CoremlLoadRoute::InMemoryAsset, "{native:?}");
    assert_eq!(native.cache, CacheOutcome::Disabled, "{native:?}");
    assert_eq!(std::fs::read_dir(cache.path()).unwrap().count(), 0);

    let exact = build(CoremlFloat16Precision::Exact);
    assert_eq!(exact.route, CoremlLoadRoute::CompiledUrl, "{exact:?}");
    assert_eq!(exact.cache, CacheOutcome::Miss, "{exact:?}");
    assert_eq!(
        build(CoremlFloat16Precision::Exact).cache,
        CacheOutcome::Hit
    );
    single_entry(cache.path());
}
