// SPDX-FileCopyrightText: 2026 Nvidia
//
// SPDX-License-Identifier: Apache-2

#![cfg(not(target_arch = "wasm32"))]

use criterion::{Criterion, criterion_group, criterion_main};
use log::warn;
use rand::Rng;
use rustnn::{
    GraphInfo,
    loader::sanitize_webnn_identifiers,
    mlcontext::{
        Backend, MLContext, MLContextOptions, MLGraphBuilder, MLPowerPreference, RustNNOptions,
    },
};
use safetensors::{Dtype, tensor::TensorView};
use std::path::{Path, PathBuf};
use webnn_graph::ast::{ConstInit, DataType, GraphJson};

const RESNET50_OPSET16: &str = include_str!("resnet50_Opset16.webnn");
const FLUX_KEIN_DECODER: &str = include_str!("flux_klein_decoder.webnn");

fn parse_model(webnn_string: &str) -> GraphJson {
    let sanitized = sanitize_webnn_identifiers(webnn_string);
    webnn_graph::parser::parse_wg_text(&sanitized).unwrap()
}

fn load_model(webnn_string: &str) -> GraphInfo {
    rustnn::webnn_json::from_graph_json(&parse_model(webnn_string)).unwrap()
}

fn load_model_with_weights(
    webnn_string: &str,
    weights_path: &Path,
    resolve_weights: bool,
) -> GraphInfo {
    let mut graph_json = parse_model(webnn_string);
    if resolve_weights {
        webnn_graph::external_weights::resolve_external_weights(
            &mut graph_json,
            Path::new("benchmark.webnn"),
            Some(weights_path.to_str().unwrap()),
            None,
        )
        .unwrap();
    }
    rustnn::webnn_json::from_graph_json(&graph_json).unwrap()
}

fn safetensors_dtype(data_type: &DataType) -> Dtype {
    match data_type {
        DataType::Float32 => Dtype::F32,
        DataType::Float16 => Dtype::F16,
        DataType::Int32 => Dtype::I32,
        DataType::Uint32 => Dtype::U32,
        DataType::Int64 => Dtype::I64,
        DataType::Uint64 => Dtype::U64,
        DataType::Int8 => Dtype::I8,
        DataType::Uint8 => Dtype::U8,
        DataType::Int4 | DataType::Uint4 => {
            panic!("safetensors does not support WebNN int4 weights")
        }
    }
}

fn prepare_weights(webnn_string: &str, filename: &str) -> PathBuf {
    let cache_path = dirs::cache_dir()
        .expect("could not determine the cache directory")
        .join("rustnn")
        .join("weights_io")
        .join(filename);
    if cache_path.exists() {
        return cache_path;
    }

    std::fs::create_dir_all(cache_path.parent().unwrap()).unwrap();
    let graph_json = parse_model(webnn_string);
    let mut rng = rand::rng();
    let tensors: Vec<_> = graph_json
        .consts
        .values()
        .filter_map(|constant| {
            let ConstInit::Weights { r#ref } = &constant.init else {
                return None;
            };
            let dtype = safetensors_dtype(&constant.data_type);
            let shape: Vec<_> = constant.shape.iter().map(|&dim| dim as usize).collect();
            let element_count = shape
                .iter()
                .try_fold(1usize, |count, &dim| count.checked_mul(dim))?;
            let byte_length = element_count
                .checked_mul(dtype.bitsize())
                .map(|bits| bits.div_ceil(8))?;
            let mut data = vec![0; byte_length];
            rng.fill_bytes(&mut data);
            Some((r#ref.clone(), dtype, shape, data))
        })
        .collect();
    let views: Vec<_> = tensors
        .iter()
        .map(|(name, dtype, shape, data)| {
            (
                name.as_str(),
                TensorView::new(*dtype, shape.clone(), data).unwrap(),
            )
        })
        .collect();
    safetensors::serialize_to_file(views, None, &cache_path).unwrap();
    warn!(
        "Wrote fake .safetensors file to {cache_path:?} for benchmark. Feel free to remove when no longer needed"
    );
    cache_path
}

fn compile_model(
    webnn_string: &str,
    weights_path: &Path,
    context_options: &MLContextOptions,
    resolve_weights: bool,
) {
    let mut graph_info = load_model_with_weights(webnn_string, weights_path, resolve_weights);
    if !resolve_weights {
        graph_info
            .resolve_safetensor_references(weights_path)
            .unwrap();
    }
    let mut context = MLContext::create(&context_options).unwrap();
    let mut builder = MLGraphBuilder::new(&mut context).unwrap();
    builder.build_graph_info(graph_info).unwrap();
}

fn create_context_options(backend: Backend) -> MLContextOptions {
    let (power_preference, accelerated) = match backend {
        Backend::Trtx => (MLPowerPreference::HighPerformance, true),
        _ => (MLPowerPreference::Default, false),
    };

    MLContextOptions::new(power_preference, accelerated).with_rustnn_backend_hint(backend)
}

fn parse_webnn(c: &mut Criterion) {
    c.bench_function("parse resnet 50 ", |b| {
        b.iter(|| load_model(RESNET50_OPSET16))
    });
    c.bench_function("parse flux_kein_decoder", |b| {
        b.iter(|| load_model(FLUX_KEIN_DECODER))
    });
}

fn compile_webnn(c: &mut Criterion) {
    let resnet50_weights = prepare_weights(RESNET50_OPSET16, "resnet50_Opset16.safetensors");
    let flux_weights = prepare_weights(FLUX_KEIN_DECODER, "flux_klein_decoder.safetensors");

    let mut no_trtx_caching = RustNNOptions::default();
    no_trtx_caching.trtx.engine_caching = false;

    for (test_name, options) in [
        ("ort", create_context_options(Backend::Onnx)),
        ("trtx with caching", create_context_options(Backend::Trtx)),
        (
            "trtx without caching",
            create_context_options(Backend::Trtx).with_rustnn_options(no_trtx_caching),
        ),
        ("litert", create_context_options(Backend::Litert)),
        ("coreml", create_context_options(Backend::Coreml)),
    ]
    .iter()
    {
        let backend = options.rustnn_backend_hint();
        if let Err(err) = MLContext::create(&options) {
            warn!(
                "Skipping {test_name:?} ({backend:?} with options {options:?}). Can't create context: {err}"
            );
            continue;
        }

        c.bench_function(
            &format!("compile resnet 50 {test_name}, options {options:?}"),
            |b| b.iter(|| compile_model(RESNET50_OPSET16, &resnet50_weights, options, false)),
        );
        c.bench_function(
            &format!("compile flux_kein_decoder {test_name}, options {options:?}"),
            |b| b.iter(|| compile_model(FLUX_KEIN_DECODER, &flux_weights, options, false)),
        );
    }
}

criterion_group!(benches, parse_webnn);
criterion_group! {
    name = compile_graph;
    config = Criterion::default().sample_size(10);
    targets = compile_webnn
}
criterion_main!(benches, compile_graph);
