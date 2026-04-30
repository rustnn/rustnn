use std::{collections::HashMap, path::PathBuf, process::Command};

use anyhow::{Context, anyhow, bail};
use clap::Parser;
use log::info;
use rustnn::{
    ContextProperties, GraphValidator, load_graph_from_path,
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

#[derive(Debug, Clone)]
struct Layout {
    num_layers: usize,
    num_heads: usize,
    max_cache_len: usize,
    head_dim: usize,
    logits_name: String,
}

#[derive(Debug, Clone)]
struct StepState {
    cache: HashMap<String, Vec<f32>>,
    current_pos: usize,
}

fn dim_to_usize(dim: &rustnn::graph::Dimension) -> usize {
    match dim {
        rustnn::graph::Dimension::Static(v) => *v as usize,
        rustnn::graph::Dimension::Dynamic(d) => d.max_size as usize,
    }
}

fn argmax(values: &[f32]) -> usize {
    let mut best_idx = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (i, v) in values.iter().enumerate() {
        if *v > best_val {
            best_val = *v;
            best_idx = i;
        }
    }
    best_idx
}

fn detect_layout(artifacts: &rustnn::ValidationArtifacts) -> anyhow::Result<Layout> {
    let mut num_layers = 0usize;
    let mut num_heads = None;
    let mut max_cache_len = None;
    let mut head_dim = None;
    let mut logits_name = None;

    for (name, desc) in &artifacts.input_names_to_descriptors {
        if let Some(rest) = name.strip_prefix("past_key_values_") {
            let parts: Vec<&str> = rest.split('_').collect();
            if parts.len() >= 2
                && let Ok(layer_idx) = parts[0].parse::<usize>()
            {
                num_layers = num_layers.max(layer_idx + 1);
            }
            if desc.shape.len() == 4 {
                num_heads = Some(dim_to_usize(&desc.shape[1]));
                max_cache_len = Some(dim_to_usize(&desc.shape[2]));
                head_dim = Some(dim_to_usize(&desc.shape[3]));
            }
        }
    }

    for name in artifacts.output_names_to_descriptors.keys() {
        if name == "logits" || name.contains("logits") {
            logits_name = Some(name.clone());
            break;
        }
    }

    Ok(Layout {
        num_layers,
        num_heads: num_heads.ok_or_else(|| anyhow!("failed to detect num_heads"))?,
        max_cache_len: max_cache_len.ok_or_else(|| anyhow!("failed to detect cache_len"))?,
        head_dim: head_dim.ok_or_else(|| anyhow!("failed to detect head_dim"))?,
        logits_name: logits_name.ok_or_else(|| anyhow!("failed to detect logits output"))?,
    })
}

fn init_state(layout: &Layout) -> StepState {
    let mut cache = HashMap::new();
    let elems = layout.num_heads * layout.max_cache_len * layout.head_dim;
    for layer in 0..layout.num_layers {
        cache.insert(
            format!("past_key_values_{}_key", layer),
            vec![0.0_f32; elems],
        );
        cache.insert(
            format!("past_key_values_{}_value", layer),
            vec![0.0_f32; elems],
        );
    }
    StepState {
        cache,
        current_pos: 0,
    }
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

    let mut context = MLContext::create(&MLContextOptions {
        power_preference: MLPowerPreference::Default,
        accelerated: true,
    })
    .map_err(|e| anyhow!("Failed to create MLContext: {e:?}"))?;

    let mut builder = MLGraphBuilder::new(&mut context)
        .map_err(|e| anyhow!("Failed to create MLGraphBuilder:\n{e}"))?;
    info!("Building graph...");
    let mut graph = builder
        .build_graph_info(&graph_info)
        .map_err(|e| anyhow!("Failed to build graph:\n{e}"))?;
    info!("Finished");

    let context_properties = ContextProperties {
        tensor_byte_length_limit: args.tensor_limit,
        ..Default::default()
    };
    let artifacts = GraphValidator::new(&graph_info, context_properties)
        .validate()
        .map_err(|e| anyhow!("validate graph: {e}"))?;
    let layout = detect_layout(&artifacts)?;
    let mut state = init_state(&layout);

    Ok(())
}
