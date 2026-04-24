#[cfg(any(
    feature = "onnx-runtime",
    feature = "trtx-runtime-mock",
    feature = "trtx-runtime"
))]
use std::io::Write;
use std::path::PathBuf;

use clap::Parser;
#[cfg(any(feature = "trtx-runtime-mock", feature = "trtx-runtime"))]
use rustnn::converters::TrtxConverter;
#[cfg(any(
    feature = "onnx-runtime",
    feature = "trtx-runtime-mock",
    feature = "trtx-runtime"
))]
use rustnn::graph::get_static_or_max_size;
#[cfg(any(
    feature = "onnx-runtime",
    feature = "trtx-runtime-mock",
    feature = "trtx-runtime"
))]
use rustnn::{ContextProperties, GraphError, GraphValidator, graph_to_dot, load_graph_from_path};

#[derive(Parser, Debug)]
#[command(author, version, about = "Validate WebNN graph descriptions", long_about = None)]
struct Cli {
    /// Path to a WebNN graph file (.webnn text or .json)
    graph: PathBuf,
    /// Optional override for the tensor byte length limit.
    #[arg(long)]
    tensor_limit: Option<usize>,
    /// Optional path to write a Graphviz DOT export of the graph.
    #[arg(long)]
    export_dot: Option<PathBuf>,
    /// Convert the graph to a different format (e.g. `onnx`).
    #[arg(long)]
    convert: Option<String>,
    /// Path to write the converted graph (stdout if omitted).
    #[arg(long)]
    convert_output: Option<PathBuf>,
    /// Execute the converted CoreML graph with zeroed inputs (macOS only).
    #[cfg(all(target_os = "macos", feature = "coreml-runtime"))]
    #[arg(long, requires = "convert")]
    run_coreml: bool,
    /// Optional path to store/load the compiled .mlmodelc bundle for CoreML execution.
    #[cfg(all(target_os = "macos", feature = "coreml-runtime"))]
    #[arg(long, requires = "run_coreml")]
    coreml_compiled_output: Option<PathBuf>,
    /// Execute the converted ONNX graph with zeroed inputs (requires `onnx-runtime` feature).
    #[cfg(feature = "onnx-runtime")]
    #[arg(long, requires = "convert")]
    run_onnx: bool,
    #[cfg(any(feature = "trtx-runtime-mock", feature = "trtx-runtime"))]
    #[arg(long, requires = "convert")]
    run_trtx: bool,
}

#[cfg(any(
    feature = "onnx-runtime",
    feature = "trtx-runtime-mock",
    feature = "trtx-runtime"
))]
fn run() -> Result<(), GraphError> {
    let cli = Cli::parse();
    let graph = load_graph_from_path(&cli.graph)?;
    let mut context = ContextProperties::default();
    if let Some(limit) = cli.tensor_limit {
        context.tensor_byte_length_limit = limit;
    }
    let validator = GraphValidator::new(&graph, context);
    let artifacts = validator.validate()?;

    println!(
        "Validated graph from `{}` with {} operands and {} operations.",
        cli.graph.display(),
        graph.operands.len(),
        graph.operations.len()
    );
    println!("Inputs:");
    for (name, descriptor) in artifacts.input_names_to_descriptors.iter() {
        println!(
            "  - {}: {:?} {:?}",
            name, descriptor.data_type, descriptor.shape
        );
    }
    println!("Outputs:");
    for (name, descriptor) in artifacts.output_names_to_descriptors.iter() {
        println!(
            "  - {}: {:?} {:?}",
            name, descriptor.data_type, descriptor.shape
        );
    }
    println!("Dependency fan-out:");
    for (operand, deps) in artifacts.operand_to_dependent_operations.iter() {
        println!("  - operand {} -> {}", operand, deps.join(", "));
    }

    if let Some(dot_path) = cli.export_dot {
        let dot = graph_to_dot(&graph);
        std::fs::write(&dot_path, dot).map_err(|err| GraphError::export(dot_path.clone(), err))?;
        println!("Exported Graphviz DOT to `{}`.", dot_path.display());
    }

    if let Some(format) = cli.convert {
        let converted = rustnn::ConverterRegistry::with_defaults().convert(&format, &graph)?;
        if let Some(ref path) = cli.convert_output {
            std::fs::write(path, &converted.data)
                .map_err(|err| GraphError::export(path.clone(), err))?;
            if let Some(ref weights) = converted.weights_data {
                let sidecar = path
                    .parent()
                    .unwrap_or_else(|| std::path::Path::new("."))
                    .join(rustnn::ONNX_EXTERNAL_WEIGHTS_FILENAME);
                std::fs::write(&sidecar, weights)
                    .map_err(|err| GraphError::export(sidecar.clone(), err))?;
                println!(
                    "Wrote ONNX external weights to `{}` (same directory as the model).",
                    sidecar.display()
                );
            }
            println!(
                "Converted graph to `{}` format at `{}` (type {}).",
                converted.format,
                path.display(),
                converted.content_type
            );
        }
        // Check if execution is requested (skip stdout write if so)
        let execution_requested = [
            false, // default to false
            #[cfg(all(target_os = "macos", feature = "coreml-runtime"))]
            cli.run_coreml,
            #[cfg(feature = "onnx-runtime")]
            cli.run_onnx,
            #[cfg(any(feature = "trtx-runtime-mock", feature = "trtx-runtime"))]
            cli.run_trtx,
        ]
        .iter()
        .any(|x| *x);

        if cli.convert_output.is_none() {
            if execution_requested {
                println!(
                    "Converted graph to `{}` in-memory (skipping stdout because execution is requested).",
                    converted.format
                );
            } else {
                std::io::stdout()
                    .write_all(&converted.data)
                    .map_err(|err| GraphError::ConversionFailed {
                        format: converted.format.to_string(),
                        reason: err.to_string(),
                    })?;
            }
        }

        #[cfg(all(target_os = "macos", feature = "coreml-runtime"))]
        if cli.run_coreml {
            if converted.format != "coreml" {
                return Err(GraphError::UnsupportedRuntimeFormat {
                    format: converted.format.to_string(),
                });
            }
            let attempts = rustnn::run_coreml_zeroed_cached(
                &converted.data,
                &artifacts.input_names_to_descriptors,
                cli.coreml_compiled_output.as_deref(),
            )?;
            println!("Executed CoreML model with zeroed inputs:");
            for attempt in attempts {
                match attempt.result {
                    Ok(outputs) => {
                        println!("  - {} succeeded:", attempt.compute_unit);
                        for out in outputs {
                            println!(
                                "      {}: shape={:?} type_code={}",
                                out.name, out.shape, out.data_type_code
                            );
                        }
                    }
                    Err(err) => {
                        println!("  - {} failed: {}", attempt.compute_unit, err);
                    }
                }
            }
        }

        #[cfg(feature = "onnx-runtime")]
        if cli.run_onnx {
            if converted.format != "onnx" {
                return Err(GraphError::UnsupportedRuntimeFormat {
                    format: converted.format.to_string(),
                });
            }
            // Build zeroed inputs
            let inputs: Vec<rustnn::OnnxInput> = artifacts
                .input_names_to_descriptors
                .iter()
                .map(|(name, desc)| {
                    let shape: Vec<usize> = desc
                        .shape
                        .iter()
                        .map(|dim| get_static_or_max_size(dim) as usize)
                        .collect();
                    let total: usize = shape.iter().product();
                    rustnn::OnnxInput {
                        name: name.clone(),
                        shape,
                        data: rustnn::TensorData::Float32(vec![0f32; total.max(1)]),
                    }
                })
                .collect();

            let outputs = rustnn::run_onnx_with_inputs(
                &converted.data,
                converted.weights_data.as_deref(),
                inputs,
            )?;
            println!("Executed ONNX model with zeroed inputs (CPU):");
            for out in outputs {
                println!("  - {}: shape={:?}", out.name, out.shape);
            }
        }

        #[cfg(any(feature = "trtx-runtime-mock", feature = "trtx-runtime"))]
        if cli.run_trtx {
            // Support both ONNX format (parsed by TensorRT) and native trtx format (pre-built engine)
            if converted.format != "onnx" && converted.format != "trtx" {
                return Err(GraphError::UnsupportedRuntimeFormat {
                    format: converted.format.to_string(),
                });
            }
            // Native WebNN->TRT engines use `TrtxConverter::engine_binding_name(operand_id)` so TRT's
            // QDQ optimizer does not match user names (e.g. WPT `quantizeLinearZeroPoint`). ONNX models
            // keep protobuf I/O names from `artifacts`.
            let inputs: Vec<rustnn::TrtxInput> = if converted.format == "trtx" {
                let mut v = Vec::with_capacity(graph.input_operands.len());
                for &op_id in &graph.input_operands {
                    let logical = graph
                        .operand(op_id)
                        .and_then(|o| o.name.as_deref())
                        .ok_or_else(|| GraphError::ConversionFailed {
                            format: "trtx".to_string(),
                            reason: format!("input operand {op_id} has no name"),
                        })?;
                    let desc = artifacts
                        .input_names_to_descriptors
                        .get(logical)
                        .ok_or_else(|| GraphError::ConversionFailed {
                            format: "trtx".to_string(),
                            reason: format!("missing descriptor for input `{logical}`"),
                        })?;
                    let total: usize = desc
                        .shape
                        .iter()
                        .map(|dim| get_static_or_max_size(dim) as usize)
                        .product::<usize>()
                        .max(1);
                    let byte_len = total * desc.data_type.bytes_per_element();
                    v.push(rustnn::TrtxInput {
                        name: TrtxConverter::engine_binding_name(op_id),
                        data: vec![0u8; byte_len],
                    });
                }
                v
            } else {
                artifacts
                    .input_names_to_descriptors
                    .iter()
                    .map(|(name, desc)| {
                        let total: usize = desc
                            .shape
                            .iter()
                            .map(|dim| get_static_or_max_size(dim) as usize)
                            .product::<usize>()
                            .max(1);
                        let byte_len = total * desc.data_type.bytes_per_element();
                        rustnn::TrtxInput {
                            name: name.clone(),
                            data: vec![0u8; byte_len],
                        }
                    })
                    .collect()
            };

            let outputs = rustnn::run_trtx_with_inputs(&converted.data, inputs)?;

            let model_type = if converted.format == "trtx" {
                "TensorRT engine"
            } else {
                "ONNX model"
            };
            println!("Executed {} with zeroed inputs (TRT-RTX):", model_type);
            for out in &outputs {
                println!("  - {}: shape={:?}", out.name, out.shape);
            }
        }
    }
    Ok(())
}

fn main() {
    #[cfg(any(
        feature = "onnx-runtime",
        feature = "trtx-runtime-mock",
        feature = "trtx-runtime"
    ))]
    {
        if let Err(err) = run() {
            eprintln!("error: {}", err);
            std::process::exit(1);
        }
    }
    #[cfg(not(any(
        feature = "onnx-runtime",
        feature = "trtx-runtime-mock",
        feature = "trtx-runtime"
    )))]
    {
        eprintln!(
            "rustnn CLI requires a runtime feature. Build with --features onnx-runtime or \
             --features trtx-runtime-mock (or trtx-runtime)."
        );
        std::process::exit(1);
    }
}
