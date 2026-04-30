use std::{path::PathBuf, process::Command};

use anyhow::{Context, anyhow, bail};
use clap::Parser;
use log::info;
use rustnn::{
    ContextProperties, ConverterRegistry, GraphValidator, load_graph_from_path,
    mlcontext::{MLContext, MLContextOptions, MLGraphBuilder, MLPowerPreference},
};
use tokenizers::Tokenizer;

#[derive(Parser, Debug)]
#[command(about = "Pure Rust SmolLM generation from .webnn + tokenizer.json")]
struct Args {
    #[arg(long)]
    model: Option<PathBuf>,
    #[arg(long)]
    tokenizer: Option<PathBuf>,
    #[arg(long, default_value = "Once upon a time")]
    prompt: String,
    #[arg(long, default_value_t = 16)]
    max_new_tokens: usize,
    #[arg(long, default_value_t = 500_000_000_000usize)]
    tensor_limit: usize,
    #[arg(long)]
    trace: bool,
    #[arg(long)]
    trace_file: Option<PathBuf>,
}

fn main() -> anyhow::Result<()> {
    if !std::env::var("RUST_LOG").is_ok() {
        unsafe { std::env::set_var("RUST_LOG", "info") };
    }
    pretty_env_logger::init();

    let args = Args::parse();
    let mut model_path = args.model;
    let mut tokenizer_path = args.tokenizer;

    let default_model_path = dirs::cache_dir()
        .ok_or_else(|| anyhow!("Could not get cache dir"))?
        .join("SmolLM-135M-webnn");

    if model_path.is_none() {
        if !default_model_path.is_dir() {
            info!(
                "Default model path {default_model_path:?} does not exist. Cloning it via git..."
            );
            let mut child = Command::new("git")
                .args([
                    "clone",
                    "https://huggingface.co/tarekziade/SmolLM-135M-webnn",
                    &default_model_path.to_string_lossy(),
                ])
                .spawn()?;
            let result = child.wait()?;
            if !result.success() {
                let _ = std::fs::remove_dir(default_model_path);
                bail!("Failed to git clone!");
            }
            info!("Finished git clone");
        } else {
            info!("Default model path exists! Using {default_model_path:?}");
        }
        model_path = Some(default_model_path.join("model.webnn"));
        tokenizer_path = Some(default_model_path.join("tokenizer.json"));
    }

    let model_path = model_path.ok_or_else(|| anyhow!("No model path available! Provide via --model (obtain from https://huggingface.co/tarekziade/SmolLM-135M-webnn)"))?;
    let tokenizer_path = tokenizer_path.ok_or_else(|| anyhow!("No tokenizer path available! Provide via --tokenizer (obtain from https://huggingface.co/tarekziade/SmolLM-135M-webnn)"))?;

    info!("Loading graph: {model_path:?}");
    let graph_info = load_graph_from_path(&model_path)
        .with_context(|| format!("Failed to load {model_path:?}"))?;

    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow!("load tokenizer {}: {e}", tokenizer_path.display()))?;
    let enc = tokenizer
        .encode(args.prompt.clone(), false)
        .map_err(|e| anyhow!("tokenize prompt: {e}"))?;
    let prompt_ids = enc.get_ids().to_vec();
    if prompt_ids.is_empty() {
        bail!("prompt produced zero tokens");
    }
    //if prompt_ids.len() >= layout.max_cache_len {
    //bail!(
    //"prompt too long: {} tokens (must be < {})",
    //prompt_ids.len(),
    //layout.max_cache_len
    //);
    //}

    let context = ContextProperties {
        tensor_byte_length_limit: args.tensor_limit,
        ..Default::default()
    };

    let mut context = MLContext::create(&MLContextOptions {
        power_preference: MLPowerPreference::Default,
        accelerated: true,
    })
    .map_err(|e| anyhow!("Failed to create MLContext: {e:?}"))?;

    let converted = ConverterRegistry::with_defaults()
        .convert("onnx", &graph_info)
        .map_err(|e| anyhow!("convert to onnx: {e}"))?;

    let mut builder = MLGraphBuilder::new(&mut context)
        .map_err(|e| anyhow!("Failed to create MLGraphBuilder:\n{e}"))?;
    let mut graph = builder
        .build_graph_info(&graph_info)
        .map_err(|e| anyhow!("Failed to build graph:\n{e}"))?;

    Ok(())
}
