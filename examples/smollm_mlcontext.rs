use std::{path::PathBuf, process::Command};

use anyhow::{Context, anyhow, bail};
use clap::Parser;
use log::info;
use rustnn::{
    load_graph_from_path,
    mlcontext::{MLContext, MLContextOptions, MLPowerPreference},
};

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
        info!("Default model path {default_model_path:?} does not exist. Cloning it via git...");
        if !default_model_path.is_dir() {
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

    info!("Loading graph");
    let graph = load_graph_from_path(&model_path)
        .with_context(|| format!("Failed to load {model_path:?}"))?;

    let mut context = MLContext::create(&MLContextOptions {
        power_preference: MLPowerPreference::Default,
        accelerated: true,
    })
    .map_err(|e| anyhow!("Failed to create MLContext: {e:?}"))?;

    Ok(())
}
