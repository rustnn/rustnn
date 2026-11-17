//! Shared by the command-line example and the device app: arguments, the decode loop and
//! the receipt.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, bail, ensure};
use clap::{Parser, ValueEnum};
use rustnn::backend_selection::DeviceType;
use rustnn::graph::{DataType, Dimension, OperandDescriptor};
use rustnn::mlcontext::{
    BackendDevice, MLContext, MLContextOptions, MLGraphBuilder, MLNamedTensors, MLPowerPreference,
    MLTensor, MLTensorDescriptor,
};
use rustnn::mlcontextoptions::{CoremlCompileRoute, RustNNOptions};
use rustnn::operator_enums::MLOperandDataType;
use serde_json::{Value, json};

#[path = "author.rs"]
mod author;

const SYSTEM: &str = "You are a helpful AI assistant named SmolLM, trained by Hugging Face";
const END_OF_TURN: u32 = 2; // <|im_end|>

#[derive(Clone, Copy, Debug, PartialEq, ValueEnum)]
enum Carry {
    /// Key/value window inside the model as an `MLState`.
    State,
    /// Key/value window bound as input and output tensors.
    Tensors,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Units {
    Cpu,
    Gpu,
    Npu,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Route {
    /// The backend's default: Apple's compiler, or the in-memory load for stateless graphs.
    Auto,
    /// Apple's `MLModel.compileModel`.
    Native,
    /// The pure-Rust `.mlmodelc` writer (`coreml-export` feature).
    Export,
}

#[derive(Parser, Debug, Clone)]
#[command(about = "SmolLM2-135M-Instruct through MLContext on Core ML")]
struct Args {
    /// Directory with `model.webnn`, its weights and `tokenizer.json`.
    #[arg(long)]
    bundle: PathBuf,
    /// Hugging Face checkpoint directory (`model.safetensors`, `tokenizer.json`). When given,
    /// the graph is authored into `--bundle` first.
    #[arg(long)]
    checkpoint: Option<PathBuf>,
    /// Stop after authoring the bundle.
    #[arg(long)]
    author_only: bool,
    /// Key/value positions kept by the graph (authoring only).
    #[arg(long, default_value_t = 512)]
    window: usize,
    /// Author only the first layers of the model (authoring only). The output is not
    /// meaningful; this is for bringing up a device that cannot hold the whole model.
    #[arg(long, default_value_t = author::LAYERS)]
    layers: usize,
    #[arg(long, value_enum, default_value = "state")]
    carry: Carry,
    #[arg(long, value_enum, default_value = "cpu")]
    units: Units,
    #[arg(long, value_enum, default_value = "auto")]
    route: Route,
    /// Directory for the compiled-bundle cache; a second run with it skips compilation.
    #[arg(long)]
    cache: Option<PathBuf>,
    /// User message, wrapped in the model's chat template.
    #[arg(long, default_value = "Tell me a short story about a small robot.")]
    prompt: String,
    /// Comma-separated token ids used as the whole prompt instead of `--prompt`.
    #[arg(long)]
    prompt_ids: Option<String>,
    #[arg(long, default_value_t = 64)]
    max_new_tokens: usize,
    /// Divides positive and multiplies negative logits of tokens already in the context.
    /// 1.0 is plain greedy decoding, which this model tends to loop under.
    #[arg(long, default_value_t = 1.1)]
    repetition_penalty: f32,
    /// Keep decoding past the end-of-turn token (fixed-length timing runs).
    #[arg(long)]
    ignore_end_of_turn: bool,
    /// Also write the receipt to this file.
    #[arg(long)]
    receipt: Option<PathBuf>,
}

fn peak_resident_bytes() -> u64 {
    let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
    unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) };
    // Bytes on Apple platforms, kilobytes elsewhere.
    usage.ru_maxrss as u64
        * if cfg!(target_vendor = "apple") {
            1
        } else {
            1024
        }
}

fn elements(descriptor: &OperandDescriptor) -> Result<usize> {
    descriptor
        .shape
        .iter()
        .map(|d| match d {
            Dimension::Static(n) => Ok(*n as usize),
            _ => bail!("the decode graph is static"),
        })
        .product()
}

fn zeroed_tensor(context: &mut MLContext<'_>, d: &OperandDescriptor) -> Result<MLTensor> {
    let data_type = match d.data_type {
        DataType::Float32 => MLOperandDataType::Float32,
        DataType::Float16 => MLOperandDataType::Float16,
        DataType::Int32 => MLOperandDataType::Int32,
        other => bail!("unsupported binding type {other:?}"),
    };
    let count = elements(d)?;
    let shape = d
        .shape
        .iter()
        .map(|d| match d {
            Dimension::Static(n) => u64::from(*n),
            Dimension::Dynamic(d) => u64::from(d.max_size),
        })
        .collect();
    let tensor = context.create_tensor(
        &MLTensorDescriptor::new(data_type, shape)
            .to_readable()
            .to_writable(),
    )?;
    match d.data_type {
        DataType::Float16 => context.write_tensor(&tensor, &vec![0u16; count])?,
        DataType::Float32 => context.write_tensor(&tensor, &vec![0f32; count])?,
        _ => context.write_tensor(&tensor, &vec![0i32; count])?,
    }
    Ok(tensor)
}

fn context_options(args: &Args, pairs: &[(String, String)]) -> MLContextOptions {
    let mut options = RustNNOptions::default();
    options.coreml.reuse_tensor_storage = true;
    options.coreml.output_backings = true;
    options.coreml.compile_route = match args.route {
        Route::Auto => CoremlCompileRoute::Auto,
        Route::Native => CoremlCompileRoute::Native,
        Route::Export => CoremlCompileRoute::RustExport,
    };
    options.coreml.compiled_bundle_cache = args.cache.clone();
    if args.carry == Carry::State {
        options.coreml.carried_state_pairs = pairs.to_vec();
    }
    let device_type = match args.units {
        Units::Cpu => DeviceType::Cpu,
        Units::Gpu => DeviceType::Gpu,
        Units::Npu => DeviceType::Npu,
    };
    MLContextOptions::new(MLPowerPreference::Default, false)
        .with_rustnn_device_hint(BackendDevice::Coreml { device_type })
        .with_rustnn_options(options)
}

fn prompt_tokens(args: &Args, tokenizer: &tokenizers::Tokenizer) -> Result<Vec<u32>> {
    if let Some(ids) = &args.prompt_ids {
        return ids
            .split(',')
            .map(|id| id.trim().parse().context("--prompt-ids"))
            .collect();
    }
    let text = format!(
        "<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
        args.prompt
    );
    let encoding = tokenizer
        .encode(text, false)
        .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
    Ok(encoding.get_ids().to_vec())
}

/// Highest logit after the Hugging Face repetition penalty over the tokens seen so far.
fn select(scores: &mut [f32], seen: &[u32], penalty: f32) -> u32 {
    if penalty != 1.0 {
        let mut done = std::collections::HashSet::new();
        for &token in seen {
            if done.insert(token) {
                let score = &mut scores[token as usize];
                *score = if *score > 0.0 {
                    *score / penalty
                } else {
                    *score * penalty
                };
            }
        }
    }
    let mut best = 0;
    for (index, score) in scores.iter().enumerate() {
        if *score > scores[best] {
            best = index;
        }
    }
    best as u32
}

fn run(args: &Args) -> Result<Value> {
    if let Some(checkpoint) = &args.checkpoint {
        let started = Instant::now();
        let mut context = MLContext::create(&context_options(args, &[]))?;
        let mut builder = MLGraphBuilder::new(&mut context)?;
        author::save(
            &mut builder,
            checkpoint,
            args.window,
            args.layers,
            &args.bundle,
        )?;
        eprintln!(
            "authored {} in {:.1} s",
            args.bundle.display(),
            started.elapsed().as_secs_f64()
        );
        if args.author_only {
            return Ok(json!({"bundle": args.bundle, "window": args.window}));
        }
    }

    let tokenizer = tokenizers::Tokenizer::from_file(args.bundle.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("tokenizer.json: {e}"))?;
    let prompt = prompt_tokens(args, &tokenizer)?;
    ensure!(!prompt.is_empty(), "the prompt is empty");

    let started = Instant::now();
    let graph_info = rustnn::load_graph_from_path(args.bundle.join("model.webnn"))?;
    let graph_load_seconds = started.elapsed().as_secs_f64();
    let (inputs, outputs) = graph_info.io_binding_maps().context("graph I/O")?;
    let window = elements(inputs.get("mask").context("mask input")?)?;
    let mut pairs: Vec<(String, String)> = inputs
        .keys()
        .filter(|name| name.starts_with("key_") || name.starts_with("value_"))
        .map(|name| (name.clone(), format!("next_{name}")))
        .collect();
    pairs.sort();
    ensure!(
        pairs.iter().all(|(_, output)| outputs.contains_key(output)),
        "every key/value input needs its next_ output"
    );

    let mut context = MLContext::create(&context_options(args, &pairs))?;
    let started = Instant::now();
    let mut graph = context.rustnn_build_graph(graph_info)?;
    let build_seconds = started.elapsed().as_secs_f64();
    let diagnostics = format!("{:?}", graph.rustnn_load_diagnostics());
    let peak_after_build = peak_resident_bytes();

    let stateful = args.carry == Carry::State;
    let mut fixed = BTreeMap::new();
    let mut banks = [BTreeMap::new(), BTreeMap::new()];
    for (name, descriptor) in &inputs {
        if pairs.iter().any(|(input, _)| input == name) {
            if !stateful {
                banks[0].insert(name.clone(), zeroed_tensor(&mut context, descriptor)?);
                banks[1].insert(name.clone(), zeroed_tensor(&mut context, descriptor)?);
            }
        } else {
            fixed.insert(name.clone(), zeroed_tensor(&mut context, descriptor)?);
        }
    }
    let logits = zeroed_tensor(&mut context, &outputs["logits"])?;
    let mut scores = vec![0f32; elements(&outputs["logits"])?];

    let mut tokens = prompt.clone();
    let mut step_seconds = Vec::new();
    let mut position = 0;
    while position < tokens.len() {
        let started = Instant::now();
        let (cos, sin) = author::rotary_row(position);
        context.write_tensor(&fixed["token"], &[tokens[position] as i32])?;
        context.write_tensor(&fixed["cos"], &cos)?;
        context.write_tensor(&fixed["sin"], &sin)?;
        context.write_tensor(&fixed["mask"], &author::mask_row(position, window))?;
        let mut input: MLNamedTensors = fixed.iter().map(|(k, v)| (k.as_str(), v)).collect();
        let mut output = MLNamedTensors::from([("logits", &logits)]);
        if !stateful {
            for (key, next) in &pairs {
                input.insert(key.as_str(), &banks[position % 2][key]);
                output.insert(next.as_str(), &banks[(position + 1) % 2][key]);
            }
        }
        context.dispatch(&mut graph, &input, &output)?;
        position += 1;
        let generated = tokens.len() - prompt.len();
        if position == tokens.len() && generated < args.max_new_tokens {
            context.read_tensor(&logits, &mut scores)?;
            ensure!(
                scores.iter().all(|s| s.is_finite()),
                "non-finite logits at position {position}"
            );
            let next = select(&mut scores, &tokens, args.repetition_penalty);
            if next != END_OF_TURN || args.ignore_end_of_turn {
                tokens.push(next);
            }
        }
        step_seconds.push(started.elapsed().as_secs_f64());
    }

    let generated = &tokens[prompt.len()..];
    // The first generated token includes one-time setup; the rate covers the rest.
    let decode = &step_seconds[prompt.len().min(step_seconds.len())..];
    let tokens_per_second =
        (!decode.is_empty()).then(|| decode.len() as f64 / decode.iter().sum::<f64>());
    let mut hash = 0xcbf29ce484222325u64;
    for token in &tokens {
        hash = (hash ^ u64::from(*token)).wrapping_mul(0x100000001b3);
    }
    let text = tokenizer
        .decode(generated, true)
        .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
    Ok(json!({
        "target": format!("{}-{}", std::env::consts::ARCH, std::env::consts::OS),
        "carry": format!("{:?}", args.carry).to_lowercase(),
        "units": format!("{:?}", args.units).to_lowercase(),
        "route": format!("{:?}", args.route).to_lowercase(),
        "window": window,
        "repetitionPenalty": args.repetition_penalty,
        "graphLoadSeconds": graph_load_seconds,
        "buildSeconds": build_seconds,
        "promptTokens": prompt.len(),
        "generatedTokens": generated.len(),
        "promptSeconds": step_seconds[..prompt.len()].iter().sum::<f64>(),
        "tokensPerSecond": tokens_per_second,
        "peakResidentBytesAfterBuild": peak_after_build,
        "peakResidentBytes": peak_resident_bytes(),
        "tokenHash": format!("{hash:016x}"),
        "tokens": tokens,
        "text": text,
        "diagnostics": diagnostics,
    }))
}

/// Parses the arguments, runs the decode and reports the receipt.
pub fn entry() {
    let mut args = Args::parse();
    let on_device = cfg!(any(target_os = "ios", target_os = "watchos"));
    if on_device {
        // The bundle ships inside the app; the cache and the receipt go to the app's caches.
        let app = std::env::current_exe().unwrap();
        let caches = PathBuf::from(std::env::var("HOME").unwrap()).join("Library/Caches");
        args.bundle = app.parent().unwrap().join(&args.bundle);
        args.cache = args.cache.map(|path| caches.join(path));
        args.receipt = args.receipt.map(|path| caches.join(path));
    }
    let result = run(&args).unwrap_or_else(|e| json!({"error": format!("{e:#}")}));
    let text = serde_json::to_string_pretty(&result).unwrap();
    println!("{text}");
    if let Some(path) = &args.receipt {
        std::fs::write(path, &text).expect("write the receipt");
    }
    if !on_device && result.get("error").is_some() {
        std::process::exit(1);
    }
}
