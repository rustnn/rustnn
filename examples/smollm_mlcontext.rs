use std::{collections::HashMap, path::PathBuf, process::Command};

use anyhow::{Context, anyhow, bail};
use clap::Parser;
use log::info;
use rustnn::{
    ContextProperties, DataType, GraphValidator, ValidationArtifacts, load_graph_from_path,
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

fn datatype_to_ml(dt: DataType) -> anyhow::Result<MLOperandDataType> {
    match dt {
        DataType::Float32 => Ok(MLOperandDataType::Float32),
        DataType::Float16 => Ok(MLOperandDataType::Float16),
        DataType::Int32 => Ok(MLOperandDataType::Int32),
        DataType::Uint32 => Ok(MLOperandDataType::Uint32),
        DataType::Int64 => Ok(MLOperandDataType::Int64),
        DataType::Uint64 => Ok(MLOperandDataType::Uint64),
        DataType::Int8 => Ok(MLOperandDataType::Int8),
        DataType::Uint8 => Ok(MLOperandDataType::Uint8),
        DataType::Int4 | DataType::Uint4 => bail!("Int4/Uint4 not supported in MLContext"),
    }
}

fn make_tensor(
    context: &mut MLContext,
    desc: &rustnn::OperandDescriptor,
    readable: bool,
    writable: bool,
) -> anyhow::Result<MLTensor> {
    let shape: Vec<u64> = desc.shape.iter().map(|d| dim_to_usize(d) as u64).collect();
    let data_type = datatype_to_ml(desc.data_type)?;
    let mut td = MLTensorDescriptor::new(data_type, shape);
    td.set_readable(readable);
    td.set_writable(writable);
    context
        .create_tensor(&td)
        .map_err(|e| anyhow!("create tensor: {e:?}"))
}

fn create_tensors(
    context: &mut MLContext,
    artifacts: &ValidationArtifacts,
) -> anyhow::Result<(HashMap<String, MLTensor>, HashMap<String, MLTensor>)> {
    let mut inputs = HashMap::new();
    for (name, desc) in &artifacts.input_names_to_descriptors {
        inputs.insert(name.clone(), make_tensor(context, desc, false, true)?);
    }
    let mut outputs = HashMap::new();
    for (name, desc) in &artifacts.output_names_to_descriptors {
        outputs.insert(name.clone(), make_tensor(context, desc, true, false)?);
    }
    Ok((inputs, outputs))
}

fn write_inputs(
    context: &mut MLContext,
    inputs: &HashMap<String, MLTensor>,
    layout: &Layout,
    state: &StepState,
    token_id: i64,
) -> anyhow::Result<()> {
    for (name, tensor) in inputs {
        if name == "input_ids" {
            context
                .write_tensor(tensor, &[token_id])
                .map_err(|e| anyhow!("write input_ids: {e:?}"))?;
        } else if name == "position_ids" {
            context
                .write_tensor(tensor, &[state.current_pos as i64])
                .map_err(|e| anyhow!("write position_ids: {e:?}"))?;
        } else if name == "attention_mask" {
            let total = tensor.shape().iter().product::<u64>() as usize;
            let mut mask = vec![0i64; total];
            let fill = (state.current_pos + 1).min(total);
            for i in 0..fill {
                mask[i] = 1;
            }
            context
                .write_tensor(tensor, &mask)
                .map_err(|e| anyhow!("write attention_mask: {e:?}"))?;
        } else if name.starts_with("past_key_values_") {
            let data = state
                .cache
                .get(name)
                .ok_or_else(|| anyhow!("missing cache entry: {name}"))?;
            context
                .write_tensor(tensor, data.as_slice())
                .map_err(|e| anyhow!("write {name}: {e:?}"))?;
        } else {
            bail!("unknown input: {name}");
        }
    }
    Ok(())
}

// Reads present_{layer}_{key,value} outputs back into the CPU cache buffers,
// then reads and returns the logits vector.
//
// The model uses fixed-size KV cache tensors ([1, heads, max_cache_len, head_dim]),
// so the full updated cache is available as a present output each step.
fn read_outputs_and_update_cache(
    context: &mut MLContext,
    outputs: &HashMap<String, MLTensor>,
    layout: &Layout,
    state: &mut StepState,
) -> anyhow::Result<Vec<f32>> {
    for layer in 0..layout.num_layers {
        for kv in ["key", "value"] {
            let present_name = format!("present_{layer}_{kv}");
            let past_name = format!("past_key_values_{layer}_{kv}");
            if let Some(tensor) = outputs.get(&present_name) {
                let cache = state
                    .cache
                    .get_mut(&past_name)
                    .ok_or_else(|| anyhow!("missing cache: {past_name}"))?;
                context
                    .read_tensor(tensor, cache.as_mut_slice())
                    .map_err(|e| anyhow!("read {present_name}: {e:?}"))?;
            }
        }
    }

    let logits_tensor = outputs
        .get(&layout.logits_name)
        .ok_or_else(|| anyhow!("missing logits: {}", layout.logits_name))?;
    let len = logits_tensor.shape().iter().product::<u64>() as usize;
    let mut logits = vec![0f32; len];
    context
        .read_tensor(logits_tensor, &mut logits)
        .map_err(|e| anyhow!("read logits: {e:?}"))?;
    Ok(logits)
}

fn run_step(
    context: &mut MLContext,
    graph: &mut MLGraph,
    input_tensors: &HashMap<String, MLTensor>,
    output_tensors: &HashMap<String, MLTensor>,
    layout: &Layout,
    state: &mut StepState,
    token_id: i64,
) -> anyhow::Result<usize> {
    write_inputs(context, input_tensors, layout, state, token_id)?;

    let inputs: HashMap<&str, &MLTensor> =
        input_tensors.iter().map(|(k, v)| (k.as_str(), v)).collect();
    let outputs: HashMap<&str, &MLTensor> =
        output_tensors.iter().map(|(k, v)| (k.as_str(), v)).collect();
    context
        .dispatch(graph, &inputs, &outputs)
        .map_err(|e| anyhow!("dispatch at pos={}: {e:?}", state.current_pos))?;

    let logits = read_outputs_and_update_cache(context, output_tensors, layout, state)?;
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
        "Layout: {} layers, {} heads, cache_len={}, head_dim={}",
        layout.num_layers, layout.num_heads, layout.max_cache_len, layout.head_dim
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

    let (input_tensors, output_tensors) = create_tensors(&mut context, &artifacts)?;
    info!(
        "Created {} input tensors, {} output tensors",
        input_tensors.len(),
        output_tensors.len()
    );

    let mut state = init_state(&layout);
    let mut last_token = 0usize;

    info!("Prefill ({} tokens)...", prompt_ids.len());
    for token_id in &prompt_ids {
        last_token = run_step(
            &mut context,
            &mut graph,
            &input_tensors,
            &output_tensors,
            &layout,
            &mut state,
            *token_id as i64,
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
            &input_tensors,
            &output_tensors,
            &layout,
            &mut state,
            last_token as i64,
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
