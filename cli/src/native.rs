use std::collections::BTreeMap;
use std::io::{IsTerminal, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use clap::{CommandFactory, FromArgMatches, Parser, builder::PossibleValuesParser};
use rustnn::mlcontext::{
    Backend, CoremlCompileRoute, MLContext, MLContextOptions, MLNamedTensors, MLPowerPreference,
    MLTensor, MLTensorDescriptor, RustNNOptions,
};
use rustnn::{ConvertedGraph, ConverterRegistry, DataType, GraphInfo};

#[derive(Debug, Parser)]
#[command(version, about = "Inspect, convert and execute WebNN graphs")]
struct Cli {
    /// WebNN graph file (.webnn or .json).
    #[arg(required_unless_present = "list")]
    graph: Option<PathBuf>,
    /// List compiled converters and probe runtime availability.
    #[arg(long, conflicts_with_all = ["graph", "convert", "run", "export_dot", "convert_output"])]
    list: bool,
    /// Write a Graphviz DOT graph.
    #[arg(long)]
    export_dot: Option<PathBuf>,
    /// Convert to an enabled model format (see --list).
    #[arg(long)]
    convert: Option<String>,
    /// Write the converted model here; otherwise write binary bytes to stdout.
    #[arg(long, requires = "convert")]
    convert_output: Option<PathBuf>,
    /// With `--convert coreml`: also write a runnable `.mlmodelc` bundle to this directory
    /// in pure Rust (needs the `coreml-export` feature; works on any host).
    #[arg(long, requires = "convert", value_name = "DIR")]
    coreml_compiled_output: Option<PathBuf>,
    /// CoreML compile step for --run: Apple's compiler, the Rust exporter, or automatic.
    #[arg(long, requires = "run", default_value = "auto", value_parser = ["auto", "native", "rust-export"])]
    coreml_compile_route: String,
    /// Execute with zero inputs using this runtime, or 'auto' for normal selection.
    #[arg(long)]
    run: Option<String>,
    /// Request a GPU or NPU for execution.
    #[arg(long, requires = "run")]
    accelerated: bool,
    /// Power preference for execution.
    #[arg(long, default_value = "default", value_parser = ["default", "high-performance", "low-power"])]
    power_preference: String,
}

fn command(registry: &ConverterRegistry) -> clap::Command {
    let runtimes =
        std::iter::once("auto").chain(Backend::compiled().into_iter().map(Backend::into));
    Cli::command()
        .mut_arg("convert", |arg| {
            arg.value_parser(PossibleValuesParser::new(registry.available_formats()))
        })
        .mut_arg("run", |arg| {
            arg.value_parser(PossibleValuesParser::new(runtimes))
        })
}

fn context_options(cli: &Cli) -> MLContextOptions {
    let power = match cli.power_preference.as_str() {
        "high-performance" => MLPowerPreference::HighPerformance,
        "low-power" => MLPowerPreference::LowPower,
        _ => MLPowerPreference::Default,
    };
    let mut rustnn_options = RustNNOptions::default();
    rustnn_options.coreml.compile_route = match cli.coreml_compile_route.as_str() {
        "native" => CoremlCompileRoute::Native,
        "rust-export" => CoremlCompileRoute::RustExport,
        _ => CoremlCompileRoute::Auto,
    };
    let options = MLContextOptions::new(power, cli.accelerated).with_rustnn_options(rustnn_options);
    match Backend::compiled()
        .into_iter()
        .find(|backend| Some(backend.as_ref()) == cli.run.as_deref())
    {
        Some(backend) => options.with_rustnn_backend_hint(backend),
        None => options,
    }
}

fn list(registry: &ConverterRegistry) {
    println!("Converters: {}", registry.available_formats().join(", "));
    println!("Runtimes:");
    let backends = Backend::compiled();
    if backends.is_empty() {
        println!("  (none compiled)");
    }
    for backend in backends {
        let mut devices = Vec::new();
        let mut last_error = None;
        for (power, accelerated) in [
            (MLPowerPreference::Default, false),
            (MLPowerPreference::HighPerformance, true),
            (MLPowerPreference::LowPower, true),
        ] {
            let options =
                MLContextOptions::new(power, accelerated).with_rustnn_backend_hint(backend);
            match MLContext::create(&options) {
                Ok(context) => {
                    let device = format!("{:?}", context.rustnn_device());
                    if !devices.contains(&device) {
                        devices.push(device);
                    }
                }
                Err(error) => last_error = Some(error),
            }
        }
        if devices.is_empty() {
            println!(
                "  {}: unavailable ({})",
                backend.as_ref(),
                last_error.unwrap()
            );
        } else {
            println!("  {}: {}", backend.as_ref(), devices.join(", "));
        }
    }
}

fn create_tensors(
    context: &mut MLContext<'_>,
    descriptors: &std::collections::HashMap<String, rustnn::OperandDescriptor>,
    input: bool,
) -> Result<BTreeMap<String, MLTensor>> {
    descriptors
        .iter()
        .map(|(name, descriptor)| {
            let shape = descriptor.static_or_max_shape();
            let descriptor = MLTensorDescriptor::new(
                descriptor.data_type.try_into()?,
                shape.into_iter().map(u64::from).collect(),
            );
            let tensor = context.create_tensor(&if input {
                descriptor.to_writable()
            } else {
                descriptor.to_readable()
            })?;
            if input {
                context.write_tensor(&tensor, &vec![0u8; tensor.rustnn_required_bytes()])?;
            }
            Ok((name.clone(), tensor))
        })
        .collect()
}

fn output_preview(data_type: DataType, bytes: &[u8], elements: usize) -> String {
    let limit = elements.min(16);
    macro_rules! values {
        ($ty:ty) => {
            bytes
                .chunks_exact(size_of::<$ty>())
                .take(limit)
                .map(|chunk| <$ty>::from_le_bytes(chunk.try_into().unwrap()).to_string())
                .collect::<Vec<_>>()
                .join(", ")
        };
    }
    match data_type {
        DataType::Float32 => values!(f32),
        DataType::Float16 => bytes
            .chunks_exact(2)
            .take(limit)
            .map(|v| half::f16::from_le_bytes(v.try_into().unwrap()).to_string())
            .collect::<Vec<_>>()
            .join(", "),
        DataType::Int64 => values!(i64),
        DataType::Uint64 => values!(u64),
        DataType::Int32 => values!(i32),
        DataType::Uint32 => values!(u32),
        DataType::Int8 => values!(i8),
        DataType::Uint8 => values!(u8),
        DataType::Int4 | DataType::Uint4 => {
            let signed = data_type == DataType::Int4;
            bytes
                .iter()
                .flat_map(|byte| [byte & 15, byte >> 4])
                .take(limit)
                .map(|v| {
                    if signed && v >= 8 {
                        (i16::from(v) - 16).to_string()
                    } else {
                        v.to_string()
                    }
                })
                .collect::<Vec<_>>()
                .join(", ")
        }
    }
}

fn bindings(tensors: &BTreeMap<String, MLTensor>) -> MLNamedTensors<'_> {
    tensors
        .iter()
        .map(|(name, tensor)| (name.as_str(), tensor))
        .collect()
}

fn execute(cli: &Cli, graph: GraphInfo) -> Result<()> {
    let mut context =
        MLContext::create(&context_options(cli)).context("creating execution context")?;
    eprintln!(
        "Runtime: {} ({:?})",
        context.rustnn_backend().as_ref(),
        context.rustnn_device()
    );
    let mut graph = context
        .rustnn_build_graph(graph)
        .context("building graph")?;
    let inputs = create_tensors(&mut context, &graph.input_descriptors, true)?;
    let outputs = create_tensors(&mut context, &graph.output_descriptors, false)?;
    context.dispatch(&mut graph, &bindings(&inputs), &bindings(&outputs))?;
    for (name, tensor) in &outputs {
        let descriptor = &graph.output_descriptors[name];
        let mut data = vec![0u8; tensor.rustnn_required_bytes()];
        context.read_tensor(tensor, &mut data)?;
        eprintln!(
            "  {name}: shape={:?} type={:?} data=[{}] (first 16 values)",
            tensor.shape(),
            descriptor.data_type,
            output_preview(
                descriptor.data_type,
                &data,
                descriptor.element_count().unwrap()
            )
        );
    }
    Ok(())
}

/// Write the CoreML conversion as a `.mlmodelc` bundle with the pure-Rust exporter.
#[cfg(feature = "coreml-export")]
fn write_coreml_bundle(model: &ConvertedGraph, directory: &Path) -> Result<()> {
    let stats = model
        .write_coreml_bundle(directory)
        .context("writing .mlmodelc bundle")?;
    eprintln!(
        "Compiled bundle: {} ({} MIL operations, {} constants, {} bytes of MIL)",
        directory.display(),
        stats.operation_count,
        stats.const_count,
        stats.output_bytes
    );
    Ok(())
}

#[cfg(not(feature = "coreml-export"))]
fn write_coreml_bundle(_model: &ConvertedGraph, _directory: &Path) -> Result<()> {
    bail!(
        "--coreml-compiled-output needs the `coreml-export` feature \
         (cargo run -p rustnn-cli --features coreml-export -- ...)"
    )
}

fn convert(
    registry: &ConverterRegistry,
    format: &str,
    graph: &GraphInfo,
    path: Option<&Path>,
    compiled_output: Option<&Path>,
) -> Result<()> {
    let model = registry.convert(format, graph)?;
    if let Some(directory) = compiled_output {
        if model.format != "coreml" {
            bail!("--coreml-compiled-output requires --convert coreml");
        }
        write_coreml_bundle(&model, directory)?;
        if path.is_none() {
            return Ok(());
        }
    }
    if let Some(path) = path {
        if let Some(weights) = model.weights_data {
            let parent = path.parent().unwrap_or_else(|| Path::new("."));
            let sidecar = if model.format == "coreml" {
                parent.join("weights/weights.bin")
            } else {
                parent.join(rustnn::ONNX_EXTERNAL_WEIGHTS_FILENAME)
            };
            if sidecar == path {
                bail!("model path conflicts with its weights sidecar");
            }
            std::fs::create_dir_all(sidecar.parent().unwrap())?;
            std::fs::write(&sidecar, weights).context("writing weights sidecar")?;
            eprintln!("Weights: {}", sidecar.display());
        }
        std::fs::write(path, model.data).context("writing converted model")?;
        eprintln!("Converted {}: {}", model.format, path.display());
    } else {
        if model.weights_data.is_some() {
            bail!("this model has external weights; provide --convert-output to save its sidecar");
        }
        std::io::stdout().lock().write_all(&model.data)?;
    }
    Ok(())
}

fn run() -> Result<()> {
    let registry = ConverterRegistry::with_defaults();
    let cli = Cli::from_arg_matches(&command(&registry).get_matches())?;
    if cli.list {
        list(&registry);
    }
    let Some(path) = &cli.graph else {
        return Ok(());
    };
    let graph = rustnn::load_graph_from_path(path)?;
    let (inputs, outputs) = graph.io_binding_maps()?;
    for (kind, descriptors) in [("Inputs", &inputs), ("Outputs", &outputs)] {
        eprintln!("{kind}:");
        for (name, descriptor) in descriptors.iter().collect::<BTreeMap<_, _>>() {
            eprintln!(
                "  {name}: {:?} {:?}",
                descriptor.data_type, descriptor.shape
            );
        }
    }
    if let Some(path) = &cli.export_dot {
        std::fs::write(path, rustnn::graph_to_dot(&graph))?;
    }
    if let Some(format) = &cli.convert {
        convert(
            &registry,
            format,
            &graph,
            cli.convert_output.as_deref(),
            cli.coreml_compiled_output.as_deref(),
        )?;
    }
    if cli.run.is_some() {
        execute(&cli, graph)?;
    }
    Ok(())
}

pub(super) fn main() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::builder()
                .with_default_directive(tracing_subscriber::filter::LevelFilter::INFO.into())
                .from_env_lossy(),
        )
        .with_span_events(tracing_subscriber::fmt::format::FmtSpan::CLOSE)
        .with_writer(std::io::stderr)
        .with_ansi(std::io::stderr().is_terminal())
        .try_init();
    if let Err(error) = run() {
        eprintln!("error: {error:#}");
        std::process::exit(1);
    }
}
