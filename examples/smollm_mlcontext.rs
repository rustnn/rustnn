use std::{collections::HashMap, path::PathBuf, process::Command};

use anyhow::{Context, anyhow, bail};
use clap::Parser;
use log::info;
use rustnn::{
    ContextProperties, GraphValidator, ValidationArtifacts, load_graph_from_path,
    mlcontext::{
        MLContext, MLContextOptions, MLGraph, MLGraphBuilder, MLPowerPreference, MLTensor,
        MLTensorDescriptor,
    },
    operator_enums::MLOperandDataType,
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
}

#[derive(Debug, Clone)]
struct Layout {
    num_layers: usize,
    num_heads: usize,
    max_cache_len: usize,
    head_dim: usize,
    logits_name: String,
    vocab_size: usize,
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

fn detect_layout(artifacts: &ValidationArtifacts) -> anyhow::Result<Layout> {
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

    let logits_name = logits_name.ok_or_else(|| anyhow!("failed to detect logits output"))?;
    let logits_desc = artifacts
        .output_names_to_descriptors
        .get(&logits_name)
        .ok_or_else(|| anyhow!("missing logits descriptor"))?;
    let vocab_size = dim_to_usize(
        logits_desc
            .shape
            .last()
            .ok_or_else(|| anyhow!("empty logits shape"))?,
    );

    Ok(Layout {
        num_layers,
        num_heads: num_heads.ok_or_else(|| anyhow!("failed to detect num_heads"))?,
        max_cache_len: max_cache_len.ok_or_else(|| anyhow!("failed to detect cache_len"))?,
        head_dim: head_dim.ok_or_else(|| anyhow!("failed to detect head_dim"))?,
        logits_name,
        vocab_size,
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

fn make_tensor(
    context: &mut MLContext,
    dtype: MLOperandDataType,
    shape: Vec<u64>,
    writable: bool,
    readable: bool,
) -> anyhow::Result<MLTensor> {
    let mut td = MLTensorDescriptor::new(dtype, shape);
    td.set_writable(writable);
    td.set_readable(readable);
    context
        .create_tensor(&td)
        .map_err(|e| anyhow!("create tensor: {e:?}"))
}

// Copies compacted (contiguous) past tokens from the padded CPU cache into a flat buffer.
fn compact_kv(
    state: &StepState,
    layout: &Layout,
    layer: usize,
    kv: &str,
    past_len: usize,
) -> Vec<f32> {
    if past_len == 0 {
        return Vec::new();
    }
    let cache = state
        .cache
        .get(&format!("past_key_values_{}_{}", layer, kv))
        .unwrap();
    let mut out = vec![0.0f32; layout.num_heads * past_len * layout.head_dim];
    for h in 0..layout.num_heads {
        for t in 0..past_len {
            let src = (h * layout.max_cache_len + t) * layout.head_dim;
            let dst = (h * past_len + t) * layout.head_dim;
            out[dst..dst + layout.head_dim].copy_from_slice(&cache[src..src + layout.head_dim]);
        }
    }
    out
}

// Stores the last token's KV slice from the present output into the padded CPU cache at
// current_pos. Mirrors smollm_webnn_rust's update_kv_cache logic.
fn store_present(
    state: &mut StepState,
    layout: &Layout,
    layer: usize,
    kv: &str,
    present: &[f32],
    seq_len: usize,
) {
    let cache = state
        .cache
        .get_mut(&format!("past_key_values_{}_{}", layer, kv))
        .unwrap();
    for h in 0..layout.num_heads {
        let dst = (h * layout.max_cache_len + state.current_pos) * layout.head_dim;
        let src = (h * seq_len + (seq_len - 1)) * layout.head_dim;
        cache[dst..dst + layout.head_dim].copy_from_slice(&present[src..src + layout.head_dim]);
    }
}

// Creates tensors with the correct dynamic shapes for each step:
// - attention_mask: [1, current_pos+1]
// - past KV: [1, heads, current_pos, head_dim]
// - present KV: [1, heads, current_pos+1, head_dim]
// This mirrors how smollm_webnn_rust provides varying-shape OnnxInputs each step.
fn run_step(
    context: &mut MLContext,
    graph: &mut MLGraph,
    layout: &Layout,
    state: &mut StepState,
    token_id: i64,
    logits_tensor: &MLTensor,
) -> anyhow::Result<usize> {
    let past_len = state.current_pos;
    let seq_len = past_len + 1;
    let h = layout.num_heads as u64;
    let d = layout.head_dim as u64;

    let t_input_ids = make_tensor(context, MLOperandDataType::Int64, vec![1, 1], true, false)?;
    context
        .write_tensor(&t_input_ids, &[token_id])
        .map_err(|e| anyhow!("write input_ids: {e:?}"))?;

    let t_position_ids = make_tensor(context, MLOperandDataType::Int64, vec![1, 1], true, false)?;
    context
        .write_tensor(&t_position_ids, &[past_len as i64])
        .map_err(|e| anyhow!("write position_ids: {e:?}"))?;

    let t_attn_mask = make_tensor(
        context,
        MLOperandDataType::Int64,
        vec![1, seq_len as u64],
        true,
        false,
    )?;
    context
        .write_tensor(&t_attn_mask, &vec![1i64; seq_len])
        .map_err(|e| anyhow!("write attention_mask: {e:?}"))?;

    let mut past_k = Vec::with_capacity(layout.num_layers);
    let mut past_v = Vec::with_capacity(layout.num_layers);
    for layer in 0..layout.num_layers {
        let k_data = compact_kv(state, layout, layer, "key", past_len);
        let t_k = make_tensor(
            context,
            MLOperandDataType::Float32,
            vec![1, h, past_len as u64, d],
            true,
            false,
        )?;
        if !k_data.is_empty() {
            context
                .write_tensor(&t_k, &k_data)
                .map_err(|e| anyhow!("write past_key {layer}: {e:?}"))?;
        }
        past_k.push(t_k);

        let v_data = compact_kv(state, layout, layer, "value", past_len);
        let t_v = make_tensor(
            context,
            MLOperandDataType::Float32,
            vec![1, h, past_len as u64, d],
            true,
            false,
        )?;
        if !v_data.is_empty() {
            context
                .write_tensor(&t_v, &v_data)
                .map_err(|e| anyhow!("write past_val {layer}: {e:?}"))?;
        }
        past_v.push(t_v);
    }

    let mut pres_k = Vec::with_capacity(layout.num_layers);
    let mut pres_v = Vec::with_capacity(layout.num_layers);
    for _ in 0..layout.num_layers {
        pres_k.push(make_tensor(
            context,
            MLOperandDataType::Float32,
            vec![1, h, seq_len as u64, d],
            false,
            true,
        )?);
        pres_v.push(make_tensor(
            context,
            MLOperandDataType::Float32,
            vec![1, h, seq_len as u64, d],
            false,
            true,
        )?);
    }

    // Name strings must outlive the HashMap borrows below.
    let past_k_names: Vec<String> = (0..layout.num_layers)
        .map(|l| format!("past_key_values_{l}_key"))
        .collect();
    let past_v_names: Vec<String> = (0..layout.num_layers)
        .map(|l| format!("past_key_values_{l}_value"))
        .collect();
    let pres_k_names: Vec<String> = (0..layout.num_layers)
        .map(|l| format!("present_{l}_key"))
        .collect();
    let pres_v_names: Vec<String> = (0..layout.num_layers)
        .map(|l| format!("present_{l}_value"))
        .collect();

    let mut inputs: HashMap<&str, &MLTensor> = HashMap::new();
    inputs.insert("input_ids", &t_input_ids);
    inputs.insert("position_ids", &t_position_ids);
    inputs.insert("attention_mask", &t_attn_mask);
    for i in 0..layout.num_layers {
        inputs.insert(&past_k_names[i], &past_k[i]);
        inputs.insert(&past_v_names[i], &past_v[i]);
    }

    let mut outputs: HashMap<&str, &MLTensor> = HashMap::new();
    outputs.insert(&layout.logits_name, logits_tensor);
    for i in 0..layout.num_layers {
        outputs.insert(&pres_k_names[i], &pres_k[i]);
        outputs.insert(&pres_v_names[i], &pres_v[i]);
    }

    context
        .dispatch(graph, &inputs, &outputs)
        .map_err(|e| anyhow!("dispatch at pos={}: {e:?}", state.current_pos))?;

    let kv_elems = layout.num_heads * seq_len * layout.head_dim;
    for layer in 0..layout.num_layers {
        let mut k = vec![0f32; kv_elems];
        context
            .read_tensor(&pres_k[layer], &mut k)
            .map_err(|e| anyhow!("read present_key {layer}: {e:?}"))?;
        store_present(state, layout, layer, "key", &k, seq_len);

        let mut v = vec![0f32; kv_elems];
        context
            .read_tensor(&pres_v[layer], &mut v)
            .map_err(|e| anyhow!("read present_val {layer}: {e:?}"))?;
        store_present(state, layout, layer, "value", &v, seq_len);
    }

    let mut logits = vec![0f32; layout.vocab_size];
    context
        .read_tensor(logits_tensor, &mut logits)
        .map_err(|e| anyhow!("read logits: {e:?}"))?;

    state.current_pos += 1;
    Ok(argmax(&logits))
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

    let context_properties = ContextProperties {
        tensor_byte_length_limit: args.tensor_limit,
        ..Default::default()
    };
    let artifacts = GraphValidator::new(&graph_info, context_properties)
        .validate()
        .map_err(|e| anyhow!("validate graph: {e}"))?;
    let layout = detect_layout(&artifacts)?;
    info!(
        "Layout: {} layers, {} heads, cache_len={}, head_dim={}, vocab={}",
        layout.num_layers,
        layout.num_heads,
        layout.max_cache_len,
        layout.head_dim,
        layout.vocab_size
    );

    if prompt_ids.len() >= layout.max_cache_len {
        bail!(
            "prompt too long: {} tokens (must be < {})",
            prompt_ids.len(),
            layout.max_cache_len
        );
    }

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
    info!("Graph built");

    // Logits output is always [1, 1, vocab_size] — create it once and reuse across all steps.
    let logits_ndim = artifacts
        .output_names_to_descriptors
        .get(&layout.logits_name)
        .map(|d| d.shape.len())
        .unwrap_or(3);
    let logits_shape: Vec<u64> = (0..logits_ndim)
        .map(|i| {
            if i + 1 == logits_ndim {
                layout.vocab_size as u64
            } else {
                1u64
            }
        })
        .collect();
    let mut logits_td = MLTensorDescriptor::new(MLOperandDataType::Float32, logits_shape);
    logits_td.set_readable(true);
    let logits_tensor = context
        .create_tensor(&logits_td)
        .map_err(|e| anyhow!("create logits tensor: {e:?}"))?;

    let mut state = init_state(&layout);
    let mut last_token = 0usize;

    info!("Prefill ({} tokens)...", prompt_ids.len());
    for token_id in &prompt_ids {
        last_token = run_step(
            &mut context,
            &mut graph,
            &layout,
            &mut state,
            *token_id as i64,
            &logits_tensor,
        )?;
    }

    info!("Decoding (max {} tokens)...", args.max_new_tokens);
    let mut generated = Vec::new();
    for _ in 0..args.max_new_tokens {
        generated.push(last_token as u32);
        if state.current_pos >= layout.max_cache_len {
            break;
        }
        last_token = run_step(
            &mut context,
            &mut graph,
            &layout,
            &mut state,
            last_token as i64,
            &logits_tensor,
        )?;
    }

    let generated_text = tokenizer
        .decode(&generated, false)
        .map_err(|e| anyhow!("decode generated text: {e}"))?;

    println!("Prompt: {}", args.prompt);
    println!("Prompt token ids: {:?}", prompt_ids);
    println!("Generated token ids: {:?}", generated);
    println!("Generated text: {}", generated_text);

    Ok(())
}
