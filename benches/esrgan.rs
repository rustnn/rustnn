// SPDX-FileCopyrightText: 2026 NVIDIA
// SPDX-License-Identifier: Apache-2.0

use anyhow::{Context, Result, ensure};
use clap::Parser;
use rustnn::{
    GraphInfo,
    backend_selection::BackendDevice,
    graph::to_symbolic_shape,
    mlcontext::{
        MLContext, MLContextOptions, MLNamedTensors, MLPowerPreference, MLTensorDescriptor,
    },
    mlcontextoptions::RustNNOptions,
};
use std::{collections::BTreeMap, path::PathBuf, time::Instant};

#[derive(Parser)]
struct Args {
    /// Network topology; external weights are replaced with deterministic synthetic FP16 values.
    #[arg(long)]
    model: Option<PathBuf>,
    #[arg(long, default_value_t = 0)]
    device: u32,
    #[arg(long, default_value_t = 1)]
    compile_samples: usize,
    #[arg(long, default_value_t = 5)]
    infer_samples: usize,
    #[arg(long, default_value_t = 3)]
    warmup: usize,
    /// Dynamic profile minimum height (must be even and no larger than the 1080p input).
    #[arg(long, default_value_t = 2)]
    min_height: u32,
    /// Dynamic profile minimum width (must be even and no larger than the 1080p input).
    #[arg(long, default_value_t = 2)]
    min_width: u32,
    /// Run one configuration (static-1080p, dynamic-1080p, dynamic-2k).
    #[arg(long)]
    case: Option<String>,
    /// Parse and validate all configurations without loading CUDA or compiling engines.
    #[arg(long)]
    validate_only: bool,
    // Cargo passes this flag to harness=false benchmarks.
    #[arg(long, hide = true)]
    bench: bool,
}

fn model(text: &str, height: u32, width: u32, dynamic: bool) -> Result<GraphInfo> {
    let mut json = serde_json::to_value(webnn_graph::parser::parse_wg_text(
        &rustnn::loader::sanitize_webnn_identifiers(text),
    )?)?;
    // Update both declarations and reshape attributes before import/shape inference.
    fn dimensions(value: &mut serde_json::Value, h: u32, w: u32, dynamic: bool) {
        match value {
            serde_json::Value::Object(object) => {
                if let Some(name) = object.get("name").and_then(|v| v.as_str()) {
                    let size = match name {
                        "h" => Some(h),
                        "w" => Some(w),
                        "h / 2" => Some(h / 2),
                        "w / 2" => Some(w / 2),
                        _ => None,
                    };
                    if let Some(size) = size {
                        if dynamic {
                            object.insert("maxSize".into(), size.into());
                        } else {
                            *value = size.into();
                        }
                        return;
                    }
                }
                for value in object.values_mut() {
                    dimensions(value, h, w, dynamic);
                }
            }
            serde_json::Value::Array(array) => {
                for value in array {
                    dimensions(value, h, w, dynamic);
                }
            }
            _ => {}
        }
    }
    dimensions(&mut json, height, width, dynamic);
    let mut graph = rustnn::webnn_json::from_graph_json_owned(serde_json::from_value(json)?)?;
    for (&id, constant) in &mut graph.constant_operand_ids_to_handles {
        let operand = &graph.operands[id as usize];
        let bytes = operand
            .descriptor
            .byte_length()
            .context("non-static constant")?;
        // Nonzero, finite, reproducible values. No weight file or output-quality claim.
        let scalar = if bytes == 2 { 0.2 } else { 0.001 };
        let bits = half::f16::from_f32(scalar).to_le_bytes();
        constant.data = bits.into_iter().cycle().take(bytes).collect();
    }
    ensure!(
        graph.has_dynamic_dimensions() == dynamic,
        "unexpected shape configuration"
    );
    Ok(graph)
}

fn concrete_shape(graph: &GraphInfo, id: u32) -> Result<Vec<u64>> {
    let symbolic = to_symbolic_shape(&graph.operands[id as usize].descriptor.shape)?;
    let context = shapeinfer_symbolic::Context::new(&Default::default());
    let shapes = context
        .compute_shapes_cel_comcrete(&[symbolic], &BTreeMap::from([("h", 1080), ("w", 1920)]))?;
    Ok(shapes[0].iter().map(|&size| u64::from(size)).collect())
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        args.compile_samples > 0 && args.infer_samples > 0,
        "sample counts must be positive"
    );
    ensure!(
        args.min_height >= 2
            && args.min_height <= 1080
            && args.min_height.is_multiple_of(2)
            && args.min_width >= 2
            && args.min_width <= 1920
            && args.min_width.is_multiple_of(2),
        "dynamic minima must be even and between 2x2 and 1080x1920"
    );
    pretty_env_logger::init();
    let path = args.model.clone().unwrap_or_else(|| {
        dirs::home_dir()
            .unwrap()
            .join("Downloads/esr_gan_fp16_fixed/real_esrgan_x2_fp16_resize_fixed.webnn")
    });
    let text =
        std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
    let cases = [
        ("static-1080p", 1080, 1920, false),
        ("dynamic-1080p", 1080, 1920, true),
        ("dynamic-2k", 1440, 2560, true),
    ];
    if let Some(case) = &args.case {
        ensure!(
            cases.iter().any(|entry| entry.0 == case),
            "unknown case {case}"
        );
    }
    println!(
        "case,sample,compile_ms,first_infer_ms,infer_mean_ms,infer_min_ms,infer_max_ms,min_height,min_width"
    );
    for (name, height, width, dynamic) in cases {
        if args.case.as_deref().is_some_and(|case| case != name) {
            continue;
        }
        let prepared =
            model(&text, height, width, dynamic).with_context(|| format!("preparing {name}"))?;
        let input_id = prepared.input_operands[0];
        let output_id = prepared.output_operands[0];
        ensure!(concrete_shape(&prepared, input_id)? == [1, 3, 1080, 1920]);
        ensure!(
            concrete_shape(&prepared, output_id)? == [1, 3, 2160, 3840],
            "{name}: output {:?}",
            concrete_shape(&prepared, output_id)?
        );
        if args.validate_only {
            eprintln!(
                "{name}: shapes validated; {} operations",
                prepared.operations.len()
            );
            continue;
        }
        for sample in 0..args.compile_samples {
            let mut options = RustNNOptions::default();
            options.trtx.engine_caching = false;
            options.trtx.runtime_cache = false;
            options.trtx.cuda_graphs = false;
            if dynamic {
                options.trtx.input_profiles.insert(
                    "input".into(),
                    rustnn::mlcontextoptions::TrtxInputProfile {
                        min: vec![1, 3, args.min_height, args.min_width],
                        optimal: vec![1, 3, 1080, 1920],
                    },
                );
            }
            let options = MLContextOptions::new(MLPowerPreference::HighPerformance, true)
                .with_rustnn_device_hint(BackendDevice::Trtx {
                    cuda_device_idx: args.device,
                })
                .with_rustnn_options(options);
            let mut context = MLContext::create(&options)?;
            let cuda = cudarc::driver::CudaContext::new(args.device as usize)?;
            let input_shape = concrete_shape(&prepared, input_id)?;
            let output_shape = concrete_shape(&prepared, output_id)?;
            let graph_info = prepared.clone();
            eprintln!("{name}: compiling sample {}", sample + 1);
            cuda.synchronize()?;
            let start = Instant::now();
            let mut graph = context.rustnn_build_graph(graph_info)?;
            cuda.synchronize()?;
            let compile_ms = start.elapsed().as_secs_f64() * 1000.0;
            let mut input_desc = MLTensorDescriptor::new(
                rustnn::operator_enums::MLOperandDataType::Float16,
                input_shape,
            );
            input_desc.set_writable(true);
            let input = context.create_tensor(&input_desc)?;
            let output = context.create_tensor(&MLTensorDescriptor::new(
                rustnn::operator_enums::MLOperandDataType::Float16,
                output_shape,
            ))?;
            let input_data = vec![half::f16::from_f32(0.5).to_bits(); 3 * 1080 * 1920];
            context.write_tensor(&input, &input_data)?;
            let inputs = MLNamedTensors::from([("input", &input)]);
            let outputs = MLNamedTensors::from([("output", &output)]);
            let mut run = || -> Result<f64> {
                cuda.synchronize()?;
                let start = Instant::now();
                context.dispatch(&mut graph, &inputs, &outputs)?;
                cuda.synchronize()?;
                Ok(start.elapsed().as_secs_f64() * 1000.0)
            };
            let first_ms = run()?;
            for _ in 0..args.warmup {
                run()?;
            }
            let times = (0..args.infer_samples)
                .map(|_| run())
                .collect::<Result<Vec<_>>>()?;
            let mean = times.iter().sum::<f64>() / times.len() as f64;
            let min = times.iter().copied().fold(f64::INFINITY, f64::min);
            let max = times.iter().copied().fold(0.0, f64::max);
            let (min_height, min_width) = if dynamic {
                (args.min_height, args.min_width)
            } else {
                (1080, 1920)
            };
            println!(
                "{name},{sample},{compile_ms:.3},{first_ms:.3},{mean:.3},{min:.3},{max:.3},{min_height},{min_width}"
            );
        }
    }
    Ok(())
}
