/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//! URL constants (`MLGraphBuilder::rustnn_constant_from_url`): the pull model for weights,
//! prototype of <https://github.com/webmachinelearning/webnn/issues/901>.
//!
//! Every test builds the same graph twice, once from host buffers and once from byte ranges
//! of a resource, and requires bit-identical outputs. The TensorRT-RTX tests return early
//! when no device is available.

#![cfg(not(target_arch = "wasm32"))]

#[path = "support/range_server.rs"]
mod range_server;

use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use rustnn::mlcontext::{
    Backend, MLContext, MLContextOptions, MLGraph, MLGraphBuilder, MLNamedOperands, MLNamedTensors,
    MLOperand, MLOperandDescriptor, MLPowerPreference, MLTensorDescriptor, RustNNOptions,
};
use rustnn::operator_enums::MLOperandDataType;
use rustnn::operator_options::MLDimension;
use rustnn::weights::{DefaultWeightsFetcher, UrlRef, WeightsFetcher};

/// Where the constants of a test graph come from.
enum Source {
    /// Host buffers: `constant_from_bytes`.
    Pointer,
    /// Byte ranges of the resource at this URL: `rustnn_constant_from_url`.
    Url(String),
}

/// Creates constants from one of the two sources and collects the bytes of the URL resource.
struct Constants {
    source: Source,
    /// Content of the resource the URL constants point into.
    blob: Vec<u8>,
    seed: u32,
}

impl Constants {
    fn new(source: Source, seed: u32) -> Self {
        Self {
            source,
            // A header in front of the first tensor, so that no range starts at offset 0.
            blob: b"RUSTNN-URL-WEIGHTS-TEST\0".to_vec(),
            seed,
        }
    }

    fn next_f32(&mut self) -> f32 {
        // xorshift32; values in [-0.5, 0.5)
        self.seed ^= self.seed << 13;
        self.seed ^= self.seed >> 17;
        self.seed ^= self.seed << 5;
        (self.seed >> 8) as f32 / (1u32 << 24) as f32 - 0.5
    }

    fn f32(&mut self, builder: &mut MLGraphBuilder, shape: &[u64]) -> MLOperand {
        let count: u64 = shape.iter().product();
        let bytes: Vec<u8> = (0..count)
            .flat_map(|_| self.next_f32().to_le_bytes())
            .collect();
        self.bytes(builder, MLOperandDataType::Float32, shape, bytes)
    }

    fn bytes(
        &mut self,
        builder: &mut MLGraphBuilder,
        data_type: MLOperandDataType,
        shape: &[u64],
        bytes: Vec<u8>,
    ) -> MLOperand {
        let descriptor = MLOperandDescriptor::new(data_type, shape.to_vec());
        match &self.source {
            Source::Pointer => builder.constant_from_bytes(&descriptor, bytes).unwrap(),
            Source::Url(url) => {
                let offset = self.blob.len() as u64;
                let len = bytes.len() as u64;
                self.blob.extend_from_slice(&bytes);
                // Padding between tensors: ranges are not adjacent.
                self.blob.extend_from_slice(&[0xAB; 3]);
                builder
                    .rustnn_constant_from_url(&descriptor, UrlRef::new(url.clone(), offset, len))
                    .unwrap()
            }
        }
    }

    /// Write the resource. Called after the graph was recorded and before `build`:
    /// recording URL constants performs no I/O.
    fn write_blob(&self, path: &Path) {
        if matches!(self.source, Source::Url(_)) {
            std::fs::write(path, &self.blob).unwrap();
        }
    }
}

const INPUT_SHAPE: [u64; 4] = [1, 3, 8, 8];
const OUTPUT_LEN: usize = 16;

/// conv2d -> add -> reshape -> matmul -> add, with four weight constants. There is no
/// activation: every weight element contributes to the output, so a corrupted range shows.
fn record_conv_mlp(builder: &mut MLGraphBuilder, constants: &mut Constants) -> MLOperand {
    let x = builder
        .input(
            "x",
            &MLOperandDescriptor::new(MLOperandDataType::Float32, INPUT_SHAPE.to_vec()),
        )
        .unwrap();
    let filter = constants.f32(builder, &[4, 3, 3, 3]);
    let conv_bias = constants.f32(builder, &[1, 4, 1, 1]);
    let weight = constants.f32(builder, &[144, OUTPUT_LEN as u64]);
    let bias = constants.f32(builder, &[OUTPUT_LEN as u64]);

    let conv = builder.conv2d(x, filter).unwrap();
    let conv = builder.add(conv, conv_bias).unwrap();
    let flat = builder
        .reshape(conv, vec![MLDimension::Static(1), MLDimension::Static(144)])
        .unwrap();
    let product = builder.matmul(flat, weight).unwrap();
    builder.add(product, bias).unwrap()
}

fn input_values() -> Vec<f32> {
    // No zeros: every filter element is multiplied by a non-zero input.
    (0..INPUT_SHAPE.iter().product::<u64>())
        .map(|i| ((i % 17) as f32 + 1.0) / 8.0)
        .collect()
}

fn context(backend: Backend, options: RustNNOptions) -> Option<MLContext<'static>> {
    let options = MLContextOptions::new(MLPowerPreference::HighPerformance, true)
        .with_rustnn_backend_hint(backend)
        .with_rustnn_options(options);
    match MLContext::create(&options) {
        Ok(context) => Some(context),
        Err(error) => {
            // Set RUSTNN_TEST_REQUIRE_BACKENDS=1 where the backends must be present, so
            // that a missing device fails the run instead of skipping it.
            assert!(
                std::env::var_os("RUSTNN_TEST_REQUIRE_BACKENDS").is_none(),
                "no {backend:?} context: {error}"
            );
            eprintln!("SKIP: no {backend:?} context: {error}");
            None
        }
    }
}

fn dispatch(
    context: &mut MLContext,
    graph: &mut MLGraph,
    input: (&str, &[u64], &[f32]),
    output: (&str, &[u64]),
) -> Vec<f32> {
    let mut input_descriptor =
        MLTensorDescriptor::new(MLOperandDataType::Float32, input.1.to_vec());
    input_descriptor.set_writable(true);
    let input_tensor = context.create_tensor(&input_descriptor).unwrap();
    let mut output_descriptor =
        MLTensorDescriptor::new(MLOperandDataType::Float32, output.1.to_vec());
    output_descriptor.set_readable(true);
    let output_tensor = context.create_tensor(&output_descriptor).unwrap();

    context.write_tensor(&input_tensor, input.2).unwrap();
    let inputs = MLNamedTensors::from([(input.0, &input_tensor)]);
    let outputs = MLNamedTensors::from([(output.0, &output_tensor)]);
    context.dispatch(graph, &inputs, &outputs).unwrap();
    let mut values = vec![0f32; output.1.iter().product::<u64>() as usize];
    context.read_tensor(&output_tensor, &mut values).unwrap();
    values
}

/// Build the conv/MLP graph from `source` on `context` and run it once.
fn run_conv_mlp(
    context: &mut MLContext,
    source: Source,
    seed: u32,
    blob_path: &Path,
    configure: impl FnOnce(&mut MLGraphBuilder),
) -> rustnn::error::Result<Vec<f32>> {
    let mut builder = MLGraphBuilder::new(context)?;
    configure(&mut builder);
    let mut constants = Constants::new(source, seed);
    let y = record_conv_mlp(&mut builder, &mut constants);
    constants.write_blob(blob_path);
    let mut graph = builder.build(&MLNamedOperands::from([("y", y)]))?;
    drop(builder);
    Ok(dispatch(
        context,
        &mut graph,
        ("x", &INPUT_SHAPE, &input_values()),
        ("y", &[1, OUTPUT_LEN as u64]),
    ))
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

fn temp_blob() -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model weights.bin");
    (dir, path)
}

fn file_url(path: &Path) -> String {
    let path = path
        .to_str()
        .unwrap()
        .replace('\\', "/")
        .replace(' ', "%20");
    if path.starts_with('/') {
        format!("file://{path}")
    } else {
        format!("file:///{path}")
    }
}

/// Counts the ranges it serves and delegates to the default fetcher.
#[derive(Debug, Default)]
struct CountingFetcher {
    ranges: std::sync::Mutex<Vec<(u64, u64)>>,
    calls: AtomicUsize,
}

impl WeightsFetcher for CountingFetcher {
    fn fetch_into(&self, source: &UrlRef, dst: &mut [u8]) -> io::Result<()> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.ranges
            .lock()
            .unwrap()
            .push((source.offset, source.len));
        DefaultWeightsFetcher.fetch_into(source, dst)
    }
}

// ---------------------------------------------------------------------------------------
// Backend-independent rules and the fallback (Burn CPU, a default feature).
// ---------------------------------------------------------------------------------------

#[test]
fn creating_a_url_constant_checks_the_length_and_does_no_io() {
    let mut builder = MLGraphBuilder::new_uncompiled();
    let descriptor = MLOperandDescriptor::new(MLOperandDataType::Float32, vec![2, 3]);
    // 24 bytes are required.
    let error = builder
        .rustnn_constant_from_url(&descriptor, UrlRef::new("does-not-exist.bin", 0, 23))
        .unwrap_err();
    assert!(
        format!("{error:?}").contains("WrongConstantSize"),
        "{error:?}"
    );
    // The resource does not exist; creation still succeeds.
    builder
        .rustnn_constant_from_url(&descriptor, UrlRef::new("does-not-exist.bin", 0, 24))
        .unwrap();
}

#[test]
fn finish_graph_info_fetches_url_constants() {
    let (_dir, path) = temp_blob();
    let mut builder = MLGraphBuilder::new_uncompiled();
    let mut constants = Constants::new(Source::Url(file_url(&path)), 7);
    let y = record_conv_mlp(&mut builder, &mut constants);
    constants.write_blob(&path);
    let graph = builder
        .finish_graph_info(&MLNamedOperands::from([("y", y)]))
        .unwrap();
    // Every constant operand has its bytes, equal to the range of the resource.
    assert_eq!(graph.constant_operand_ids_to_handles.len(), 4);
    let header = b"RUSTNN-URL-WEIGHTS-TEST\0".len();
    let filter = &graph.constant_operand_ids_to_handles[&1].data;
    assert_eq!(filter.len(), 4 * 3 * 3 * 3 * 4);
    assert_eq!(filter[..], constants.blob[header..header + filter.len()]);
}

#[test]
fn exports_reject_graphs_with_url_constants() {
    let (dir, path) = temp_blob();
    let mut builder = MLGraphBuilder::new_uncompiled();
    let mut constants = Constants::new(Source::Url(file_url(&path)), 7);
    let y = record_conv_mlp(&mut builder, &mut constants);
    let outputs = MLNamedOperands::from([("y", y)]);
    let error = builder
        .rustnn_save_webnn(&outputs, dir.path().join("model.webnn"))
        .unwrap_err();
    assert!(
        format!("{error:?}").contains("UrlConstantsNotMaterialized"),
        "{error:?}"
    );
    assert!(builder.rustnn_webnn_text_for_outputs(&outputs).is_none());
}

#[test]
fn fallback_backend_fetches_everything_and_matches_pointer_constants() {
    let Some(mut context) = context(Backend::BurnCpu, RustNNOptions::default()) else {
        return;
    };
    let (_dir, path) = temp_blob();
    let pointer = run_conv_mlp(&mut context, Source::Pointer, 11, &path, |_| {}).unwrap();

    let fetcher = Arc::new(CountingFetcher::default());
    let url = run_conv_mlp(&mut context, Source::Url(file_url(&path)), 11, &path, |b| {
        b.rustnn_set_weights_fetcher(fetcher.clone())
    })
    .unwrap();
    assert_eq!(bits(&pointer), bits(&url));
    // Four constants, each range requested exactly once.
    let mut ranges = fetcher.ranges.lock().unwrap().clone();
    ranges.sort_unstable();
    ranges.dedup();
    assert_eq!(ranges.len(), 4);
    assert_eq!(fetcher.calls.load(Ordering::SeqCst), 4);
}

/// Records how many fetches ran at the same time.
#[derive(Debug, Default)]
struct OverlapFetcher {
    running: AtomicUsize,
    peak: AtomicUsize,
}

impl WeightsFetcher for OverlapFetcher {
    fn fetch_into(&self, source: &UrlRef, dst: &mut [u8]) -> io::Result<()> {
        let running = self.running.fetch_add(1, Ordering::SeqCst) + 1;
        self.peak.fetch_max(running, Ordering::SeqCst);
        // Long enough for the other workers to start a fetch, if there are any.
        std::thread::sleep(std::time::Duration::from_millis(20));
        let result = DefaultWeightsFetcher.fetch_into(source, dst);
        self.running.fetch_sub(1, Ordering::SeqCst);
        result
    }
}

#[test]
fn context_options_are_the_fetch_options_of_its_builders() {
    // The default is four fetch threads, one for each constant of this graph. With one
    // thread from `RustNNOptions::url_fetch` no two fetches can overlap.
    let mut options = RustNNOptions::default();
    options.url_fetch.threads = Some(1);
    let Some(mut context) = context(Backend::BurnCpu, options) else {
        return;
    };
    {
        let builder = MLGraphBuilder::new(&mut context).unwrap();
        assert_eq!(builder.rustnn_url_fetch_options().threads, Some(1));
    }
    let (_dir, path) = temp_blob();
    let fetcher = Arc::new(OverlapFetcher::default());
    run_conv_mlp(&mut context, Source::Url(file_url(&path)), 5, &path, |b| {
        b.rustnn_set_weights_fetcher(fetcher.clone())
    })
    .unwrap();
    assert_eq!(fetcher.peak.load(Ordering::SeqCst), 1);
}

#[test]
fn missing_resource_fails_the_build_with_operand_and_url() {
    let Some(mut context) = context(Backend::BurnCpu, RustNNOptions::default()) else {
        return;
    };
    let (_dir, path) = temp_blob();
    let mut builder = MLGraphBuilder::new(&mut context).unwrap();
    let mut constants = Constants::new(Source::Url(file_url(&path)), 3);
    let y = record_conv_mlp(&mut builder, &mut constants);
    // The blob is never written.
    let error = builder
        .build(&MLNamedOperands::from([("y", y)]))
        .unwrap_err();
    let text = format!("{error:?}");
    assert!(text.contains("UnresolvedConstant"), "{text}");
    assert!(text.contains("model%20weights.bin"), "{text}");
}

#[test]
fn short_resource_fails_the_build() {
    let Some(mut context) = context(Backend::BurnCpu, RustNNOptions::default()) else {
        return;
    };
    let (_dir, path) = temp_blob();
    let mut builder = MLGraphBuilder::new(&mut context).unwrap();
    let mut constants = Constants::new(Source::Url(file_url(&path)), 3);
    let y = record_conv_mlp(&mut builder, &mut constants);
    // Truncated resource: the last range ends past the end of the file.
    std::fs::write(&path, &constants.blob[..constants.blob.len() - 40]).unwrap();
    let error = builder
        .build(&MLNamedOperands::from([("y", y)]))
        .unwrap_err();
    assert!(
        format!("{error:?}").contains("UnresolvedConstant"),
        "{error:?}"
    );
}

#[cfg(feature = "url-weights-http")]
#[test]
fn http_range_requests_feed_the_fallback_backend() {
    let Some(mut context) = context(Backend::BurnCpu, RustNNOptions::default()) else {
        return;
    };
    let (_dir, path) = temp_blob();
    let pointer = run_conv_mlp(&mut context, Source::Pointer, 23, &path, |_| {}).unwrap();
    // The server reads the file per request; it is written before `build`.
    let server = range_server::RangeServer::start(&path, 0).unwrap();
    let url = run_conv_mlp(
        &mut context,
        Source::Url(server.url("model.bin")),
        23,
        &path,
        |_| {},
    )
    .unwrap();
    assert_eq!(bits(&pointer), bits(&url));
    assert_eq!(server.requests(), 4);
    // Only the four ranges travelled, not the header or the padding.
    assert_eq!(server.bytes_sent(), (432 + 16 + 144 * 16 * 4 + 64) as u64);
}

#[cfg(feature = "url-weights-http")]
#[test]
fn http_entity_tag_mismatch_fails_the_fetch() {
    let (_dir, path) = temp_blob();
    std::fs::write(&path, [1u8, 2, 3, 4, 5, 6, 7, 8]).unwrap();
    let server = range_server::RangeServer::start(&path, 0).unwrap();
    let mut dst = [0u8; 4];
    DefaultWeightsFetcher
        .fetch_into(
            &UrlRef::new(server.url("m"), 2, 4).with_etag(range_server::ETAG),
            &mut dst,
        )
        .unwrap();
    assert_eq!(dst, [3, 4, 5, 6]);
    let error = DefaultWeightsFetcher
        .fetch_into(
            &UrlRef::new(server.url("m"), 2, 4).with_etag("\"other\""),
            &mut dst,
        )
        .unwrap_err();
    assert!(error.to_string().contains("412"), "{error}");
    // A range past the end of the resource is an error too.
    DefaultWeightsFetcher
        .fetch_into(&UrlRef::new(server.url("m"), 6, 4), &mut dst)
        .unwrap_err();
}

// ---------------------------------------------------------------------------------------
// TensorRT-RTX: placeholder engine, concurrent fetch, refit from the fetched buffers.
// ---------------------------------------------------------------------------------------

#[cfg(feature = "trtx-runtime")]
mod trtx {
    use super::*;
    use rustnn::operator_options::MLLayerNormalizationOptions;
    use rustnn::weights::UrlFetchOptions;

    fn trtx_context(engine_caching: bool, fail_on_cache_miss: bool) -> Option<MLContext<'static>> {
        let mut options = RustNNOptions::default();
        options.trtx.engine_caching = engine_caching;
        options.trtx.fail_on_cache_miss = fail_on_cache_miss;
        context(Backend::Trtx, options)
    }

    #[test]
    fn url_constants_match_pointer_constants() {
        let Some(mut context) = trtx_context(false, false) else {
            return;
        };
        let (_dir, path) = temp_blob();
        let pointer = run_conv_mlp(&mut context, Source::Pointer, 101, &path, |_| {}).unwrap();

        // One refit after everything was fetched.
        let fetcher = Arc::new(CountingFetcher::default());
        let url = run_conv_mlp(
            &mut context,
            Source::Url(file_url(&path)),
            101,
            &path,
            |b| {
                b.rustnn_set_weights_fetcher(fetcher.clone());
                b.rustnn_set_url_fetch_options(UrlFetchOptions::default());
            },
        )
        .unwrap();
        assert_eq!(bits(&pointer), bits(&url));
        assert_eq!(fetcher.calls.load(Ordering::SeqCst), 4);

        // A plain path instead of a file:// URL.
        let plain = run_conv_mlp(
            &mut context,
            Source::Url(path.to_str().unwrap().to_string()),
            101,
            &path,
            |_| {},
        )
        .unwrap();
        assert_eq!(bits(&pointer), bits(&plain));
    }

    #[test]
    fn batched_refit_with_a_byte_budget_matches() {
        let Some(mut context) = trtx_context(false, false) else {
            return;
        };
        let (_dir, path) = temp_blob();
        let pointer = run_conv_mlp(&mut context, Source::Pointer, 55, &path, |_| {}).unwrap();
        // (max in-flight bytes, bind batch bytes, threads)
        let cases: [(Option<usize>, Option<usize>, usize); 4] = [
            // Refit after every constant.
            (None, Some(1), 1),
            // Budget smaller than every constant: each one is fetched alone and the consumer
            // must release budget for the workers to continue.
            (Some(8), None, 4),
            // Budget that fits the small constants but not the matmul weight.
            (Some(1024), Some(512), 2),
            // Batch larger than the budget: it is clamped to half of the budget.
            (Some(4096), Some(1 << 30), 3),
        ];
        for (budget, batch, threads) in cases {
            let mut options = UrlFetchOptions::default().with_threads(threads);
            options.max_inflight_bytes = budget;
            options.bind_batch_bytes = batch;
            let url = run_conv_mlp(&mut context, Source::Url(file_url(&path)), 55, &path, |b| {
                b.rustnn_set_url_fetch_options(options.clone())
            })
            .unwrap();
            assert_eq!(bits(&pointer), bits(&url), "options {options:?}");
        }
    }

    #[test]
    fn placeholder_engine_is_cached_independent_of_the_weights() {
        // The engine cache is keyed by topology: a second model with other weights at another
        // URL reuses the engine, and only the refit differs.
        let (_dir, path) = temp_blob();
        {
            let Some(mut context) = trtx_context(true, false) else {
                return;
            };
            run_conv_mlp(&mut context, Source::Url(file_url(&path)), 1, &path, |_| {}).unwrap();
        }
        let pointer = {
            let mut context = trtx_context(false, false).unwrap();
            run_conv_mlp(&mut context, Source::Pointer, 2, &path, |_| {}).unwrap()
        };
        let (_dir2, path2) = temp_blob();
        // `fail_on_cache_miss`: the build fails unless the engine comes from the cache.
        let mut context = trtx_context(true, true).unwrap();
        let url = run_conv_mlp(
            &mut context,
            Source::Url(file_url(&path2)),
            2,
            &path2,
            |_| {},
        )
        .unwrap();
        assert_eq!(bits(&pointer), bits(&url));
    }

    #[test]
    fn failed_fetch_fails_the_build_without_hanging() {
        let Some(mut context) = trtx_context(false, false) else {
            return;
        };
        let (_dir, path) = temp_blob();
        let mut builder = MLGraphBuilder::new(&mut context).unwrap();
        let mut constants = Constants::new(Source::Url(file_url(&path)), 3);
        let y = record_conv_mlp(&mut builder, &mut constants);
        // Truncated resource: the first ranges succeed, the last one fails.
        std::fs::write(&path, &constants.blob[..constants.blob.len() - 40]).unwrap();
        let error = builder
            .build(&MLNamedOperands::from([("y", y)]))
            .unwrap_err();
        let text = format!("{error:?}");
        assert!(text.contains("UnresolvedConstant"), "{text}");
    }

    /// gather with constant int64 indices: TensorRT bakes the indices into the engine, so
    /// their values are needed before the network is built (eager fetch); the table is
    /// refittable and deferred. An unused constant is never fetched.
    fn run_gather(
        context: &mut MLContext,
        source: Source,
        blob_path: &Path,
        fetcher: Option<Arc<CountingFetcher>>,
    ) -> Vec<f32> {
        let mut builder = MLGraphBuilder::new(context).unwrap();
        if let Some(fetcher) = fetcher {
            builder.rustnn_set_weights_fetcher(fetcher);
        }
        let mut constants = Constants::new(source, 77);
        let x = builder
            .input(
                "x",
                &MLOperandDescriptor::new(MLOperandDataType::Float32, vec![2, 3]),
            )
            .unwrap();
        let table = constants.f32(&mut builder, &[5, 3]);
        let indices = constants.bytes(
            &mut builder,
            MLOperandDataType::Int64,
            &[2],
            [3i64, 1].iter().flat_map(|v| v.to_le_bytes()).collect(),
        );
        let rows = builder.gather(table, indices).unwrap();
        let y = builder.add(rows, x).unwrap();
        constants.write_blob(blob_path);
        if matches!(constants.source, Source::Url(_)) {
            // A constant nobody consumes, pointing at a resource that does not exist.
            builder
                .rustnn_constant_from_url(
                    &MLOperandDescriptor::new(MLOperandDataType::Float32, vec![4]),
                    UrlRef::new("no-such-directory/unused.bin", 0, 16),
                )
                .unwrap();
        }
        let mut graph = builder.build(&MLNamedOperands::from([("y", y)])).unwrap();
        drop(builder);
        dispatch(
            context,
            &mut graph,
            ("x", &[2, 3], &[0.5, -1.0, 2.0, 4.0, 0.25, -3.0]),
            ("y", &[2, 3]),
        )
    }

    #[test]
    fn baked_constants_are_fetched_before_the_build_and_unused_ones_never() {
        let Some(mut context) = trtx_context(false, false) else {
            return;
        };
        let (_dir, path) = temp_blob();
        let pointer = run_gather(&mut context, Source::Pointer, &path, None);
        let fetcher = Arc::new(CountingFetcher::default());
        let url = run_gather(
            &mut context,
            Source::Url(file_url(&path)),
            &path,
            Some(fetcher.clone()),
        );
        assert_eq!(bits(&pointer), bits(&url));
        // The table and the indices; not the unused constant.
        assert_eq!(fetcher.calls.load(Ordering::SeqCst), 2);
    }

    /// layerNormalization with `axes = []` normalizes nothing, so the output is the bias.
    /// The lowering reads a constant bias at conversion time when its bytes are present and
    /// uses the bias tensor otherwise; a URL bias takes the tensor branch.
    ///
    /// With host buffers this graph does not build on TensorRT-RTX today (the lowering copies
    /// the bias into an unnamed constant, TensorRT drops the original constant layer and its
    /// refit is rejected), so the expected values are computed here instead of compared.
    #[test]
    fn layer_normalization_without_axes_takes_the_tensor_branch_for_a_url_bias() {
        let Some(mut context) = trtx_context(false, false) else {
            return;
        };
        let (_dir, path) = temp_blob();
        let mut builder = MLGraphBuilder::new(&mut context).unwrap();
        let mut constants = Constants::new(Source::Url(file_url(&path)), 9);
        let x = builder
            .input(
                "x",
                &MLOperandDescriptor::new(MLOperandDataType::Float32, vec![2, 3]),
            )
            .unwrap();
        let bias = constants.f32(&mut builder, &[]);
        let y = builder
            .layer_normalization_with_options(
                x,
                MLLayerNormalizationOptions {
                    bias: Some(bias.into()),
                    axes: Some(vec![]),
                    ..Default::default()
                },
            )
            .unwrap();
        constants.write_blob(&path);
        let mut graph = builder.build(&MLNamedOperands::from([("y", y)])).unwrap();
        drop(builder);
        let output = dispatch(
            &mut context,
            &mut graph,
            ("x", &[2, 3], &[0.5, -1.0, 2.0, 4.0, 0.25, -3.0]),
            ("y", &[2, 3]),
        );
        let header = b"RUSTNN-URL-WEIGHTS-TEST\0".len();
        let bias_value = f32::from_le_bytes(constants.blob[header..header + 4].try_into().unwrap());
        assert_eq!(bits(&output), bits(&[bias_value; 6]));
    }

    #[cfg(feature = "url-weights-http")]
    #[test]
    fn http_range_requests_feed_the_refit() {
        let Some(mut context) = trtx_context(false, false) else {
            return;
        };
        let (_dir, path) = temp_blob();
        let pointer = run_conv_mlp(&mut context, Source::Pointer, 31, &path, |_| {}).unwrap();
        let server = range_server::RangeServer::start(&path, 0).unwrap();
        let url = run_conv_mlp(
            &mut context,
            Source::Url(server.url("model.bin")),
            31,
            &path,
            |b| {
                b.rustnn_set_url_fetch_options(
                    UrlFetchOptions::default().with_max_inflight_bytes(2048),
                )
            },
        )
        .unwrap();
        assert_eq!(bits(&pointer), bits(&url));
        assert_eq!(server.requests(), 4);
    }
}
