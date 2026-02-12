#[cfg(not(feature = "onnx-runtime"))]
fn main() {
    eprintln!("This example requires `--features onnx-runtime`.");
    eprintln!("Run: cargo run --features onnx-runtime --example smollm_webnn_rust -- --help");
}

#[cfg(feature = "onnx-runtime")]
mod app {
    use std::collections::HashMap;
    use std::fs::File;
    use std::io::{BufWriter, Write};
    use std::path::PathBuf;

    use clap::Parser;
    use prost::Message;
    use rustnn::executors::onnx::{
        OnnxInput, OnnxOutputWithData, TensorData, run_onnx_with_inputs,
    };
    use rustnn::protos::onnx::ModelProto;
    use rustnn::{ContextProperties, ConverterRegistry, GraphValidator, load_graph_from_path};
    use tokenizers::Tokenizer;

    #[derive(Parser, Debug)]
    #[command(about = "Pure Rust SmolLM generation from .webnn + tokenizer.json")]
    struct Args {
        #[arg(long, default_value = "/tmp/smollm135/smollm.webnn")]
        model: PathBuf,
        #[arg(long, default_value = "/tmp/smollm135/tokenizer.json")]
        tokenizer: PathBuf,
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

    struct TraceLogger {
        enabled: bool,
        writer: Option<BufWriter<File>>,
    }

    impl TraceLogger {
        fn new(args: &Args) -> Result<Self, String> {
            if !args.trace {
                return Ok(Self {
                    enabled: false,
                    writer: None,
                });
            }
            let writer = if let Some(path) = &args.trace_file {
                Some(BufWriter::new(
                    File::create(path).map_err(|e| format!("create trace file: {e}"))?,
                ))
            } else {
                None
            };
            Ok(Self {
                enabled: true,
                writer,
            })
        }

        fn log(&mut self, line: &str) -> Result<(), String> {
            if !self.enabled {
                return Ok(());
            }
            println!("{line}");
            if let Some(w) = self.writer.as_mut() {
                writeln!(w, "{line}").map_err(|e| format!("write trace: {e}"))?;
            }
            Ok(())
        }
    }

    fn dim_to_usize(dim: &rustnn::graph::Dimension) -> usize {
        match dim {
            rustnn::graph::Dimension::Static(v) => *v as usize,
            rustnn::graph::Dimension::Dynamic(d) => d.max_size as usize,
        }
    }

    fn model_input_order(onnx_bytes: &[u8]) -> Result<Vec<String>, String> {
        let model = ModelProto::decode(onnx_bytes).map_err(|e| format!("decode onnx: {e}"))?;
        let graph = model
            .graph
            .ok_or_else(|| "onnx model missing graph".to_string())?;
        Ok(graph.input.into_iter().map(|v| v.name).collect())
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

    fn detect_layout(artifacts: &rustnn::ValidationArtifacts) -> Result<Layout, String> {
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
            num_heads: num_heads.ok_or_else(|| "failed to detect num_heads".to_string())?,
            max_cache_len: max_cache_len.ok_or_else(|| "failed to detect cache_len".to_string())?,
            head_dim: head_dim.ok_or_else(|| "failed to detect head_dim".to_string())?,
            logits_name: logits_name.ok_or_else(|| "failed to detect logits output".to_string())?,
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

    fn update_kv_cache(
        layout: &Layout,
        state: &mut StepState,
        outputs: &[OnnxOutputWithData],
    ) -> Result<(), String> {
        for layer in 0..layout.num_layers {
            let present_k_name = format!("present_{}_key", layer);
            let present_v_name = format!("present_{}_value", layer);

            let present_k = outputs
                .iter()
                .find(|o| o.name == present_k_name)
                .ok_or_else(|| format!("missing output: {present_k_name}"))?;
            let present_v = outputs
                .iter()
                .find(|o| o.name == present_v_name)
                .ok_or_else(|| format!("missing output: {present_v_name}"))?;

            if present_k.shape.len() != 4 {
                return Err(format!(
                    "unexpected rank for {}: {:?}",
                    present_k_name, present_k.shape
                ));
            }
            if present_k.shape[1] != layout.num_heads || present_k.shape[3] != layout.head_dim {
                return Err(format!(
                    "unexpected shape for {}: {:?}",
                    present_k_name, present_k.shape
                ));
            }
            let seq_len = present_k.shape[2];
            let cache_k_name = format!("past_key_values_{}_key", layer);
            let cache_v_name = format!("past_key_values_{}_value", layer);
            {
                let cache_k = state
                    .cache
                    .get_mut(&cache_k_name)
                    .ok_or_else(|| format!("missing cache tensor: {cache_k_name}"))?;
                let cache_len = layout.max_cache_len;
                for h in 0..layout.num_heads {
                    let cache_base = (h * cache_len + state.current_pos) * layout.head_dim;
                    let present_base = (h * seq_len + (seq_len - 1)) * layout.head_dim;
                    cache_k[cache_base..cache_base + layout.head_dim].copy_from_slice(
                        &present_k.data[present_base..present_base + layout.head_dim],
                    );
                }
            }
            {
                let cache_v = state
                    .cache
                    .get_mut(&cache_v_name)
                    .ok_or_else(|| format!("missing cache tensor: {cache_v_name}"))?;
                let cache_len = layout.max_cache_len;
                for h in 0..layout.num_heads {
                    let cache_base = (h * cache_len + state.current_pos) * layout.head_dim;
                    let present_base = (h * seq_len + (seq_len - 1)) * layout.head_dim;
                    cache_v[cache_base..cache_base + layout.head_dim].copy_from_slice(
                        &present_v.data[present_base..present_base + layout.head_dim],
                    );
                }
            }
        }
        Ok(())
    }

    fn build_inputs(
        input_order: &[String],
        layout: &Layout,
        state: &StepState,
        token_id: i64,
    ) -> Result<Vec<OnnxInput>, String> {
        let mut onnx_inputs = Vec::with_capacity(input_order.len());
        for name in input_order {
            if name == "input_ids" {
                onnx_inputs.push(OnnxInput {
                    name: name.clone(),
                    shape: vec![1, 1],
                    data: TensorData::Int64(vec![token_id]),
                });
            } else if name == "position_ids" {
                onnx_inputs.push(OnnxInput {
                    name: name.clone(),
                    shape: vec![1, 1],
                    data: TensorData::Int64(vec![state.current_pos as i64]),
                });
            } else if name == "attention_mask" {
                let mask_len = state.current_pos + 1;
                let attention_mask = vec![1_i64; mask_len];
                onnx_inputs.push(OnnxInput {
                    name: name.clone(),
                    shape: vec![1, mask_len],
                    data: TensorData::Int64(attention_mask),
                });
            } else if name.starts_with("past_key_values_") {
                let tensor = state
                    .cache
                    .get(name)
                    .ok_or_else(|| format!("missing cache input required by ONNX: {name}"))?
                    .clone();
                let past_len = state.current_pos;
                let mut compact = vec![0.0_f32; layout.num_heads * past_len * layout.head_dim];
                for h in 0..layout.num_heads {
                    for t in 0..past_len {
                        let src = (h * layout.max_cache_len + t) * layout.head_dim;
                        let dst = (h * past_len + t) * layout.head_dim;
                        compact[dst..dst + layout.head_dim]
                            .copy_from_slice(&tensor[src..src + layout.head_dim]);
                    }
                }
                onnx_inputs.push(OnnxInput {
                    name: name.clone(),
                    shape: vec![1, layout.num_heads, past_len, layout.head_dim],
                    data: TensorData::Float32(compact),
                });
            } else {
                return Err(format!("unsupported/unknown model input: {name}"));
            }
        }
        Ok(onnx_inputs)
    }

    fn run() -> Result<(), String> {
        let args = Args::parse();
        let mut trace = TraceLogger::new(&args)?;

        let graph = load_graph_from_path(&args.model).map_err(|e| format!("load graph: {e}"))?;
        let mut context = ContextProperties::default();
        context.tensor_byte_length_limit = args.tensor_limit;
        let artifacts = GraphValidator::new(&graph, context)
            .validate()
            .map_err(|e| format!("validate graph: {e}"))?;

        let converted = ConverterRegistry::with_defaults()
            .convert("onnx", &graph)
            .map_err(|e| format!("convert to onnx: {e}"))?;
        let input_order = model_input_order(&converted.data)?;
        let layout = detect_layout(&artifacts)?;

        let tokenizer = Tokenizer::from_file(&args.tokenizer)
            .map_err(|e| format!("load tokenizer {}: {e}", args.tokenizer.display()))?;
        let enc = tokenizer
            .encode(args.prompt.clone(), false)
            .map_err(|e| format!("tokenize prompt: {e}"))?;
        let prompt_ids = enc.get_ids().to_vec();
        if prompt_ids.is_empty() {
            return Err("prompt produced zero tokens".to_string());
        }
        if prompt_ids.len() >= layout.max_cache_len {
            return Err(format!(
                "prompt too long: {} tokens (must be < {})",
                prompt_ids.len(),
                layout.max_cache_len
            ));
        }

        let mut state = init_state(&layout);
        let mut last_outputs = Vec::new();

        for token_id in &prompt_ids {
            let pos_before = state.current_pos;
            let mask_ones = state.current_pos + 1;
            trace.log(&format!(
                "TRACE phase=prefill pos={} token_in={} position_id={} mask_ones={}",
                pos_before, token_id, pos_before, mask_ones
            ))?;
            let onnx_inputs = build_inputs(&input_order, &layout, &state, *token_id as i64)?;
            let outputs = run_onnx_with_inputs(&converted.data, onnx_inputs)
                .map_err(|e| format!("onnx run (prefill pos={}): {e}", state.current_pos))?;
            let logits = outputs
                .iter()
                .find(|o| o.name == layout.logits_name)
                .ok_or_else(|| format!("missing logits output: {}", layout.logits_name))?;
            let argmax_after = argmax(&logits.data);
            trace.log(&format!(
                "TRACE phase=prefill pos={} logits_argmax={}",
                pos_before, argmax_after
            ))?;
            update_kv_cache(&layout, &mut state, &outputs)?;
            state.current_pos += 1;
            last_outputs = outputs;
        }

        let mut generated = Vec::new();
        for _ in 0..args.max_new_tokens {
            let logits = last_outputs
                .iter()
                .find(|o| o.name == layout.logits_name)
                .ok_or_else(|| format!("missing logits output: {}", layout.logits_name))?;
            let next_id = argmax(&logits.data) as u32;
            generated.push(next_id);
            trace.log(&format!(
                "TRACE phase=decode_select pos={} selected_token={}",
                state.current_pos, next_id
            ))?;

            if state.current_pos >= layout.max_cache_len {
                break;
            }

            let pos_before = state.current_pos;
            let mask_ones = state.current_pos + 1;
            trace.log(&format!(
                "TRACE phase=decode_run pos={} token_in={} position_id={} mask_ones={}",
                pos_before, next_id, pos_before, mask_ones
            ))?;
            let onnx_inputs = build_inputs(&input_order, &layout, &state, next_id as i64)?;
            let outputs = run_onnx_with_inputs(&converted.data, onnx_inputs)
                .map_err(|e| format!("onnx run (decode pos={}): {e}", state.current_pos))?;
            let logits_after = outputs
                .iter()
                .find(|o| o.name == layout.logits_name)
                .ok_or_else(|| format!("missing logits output: {}", layout.logits_name))?;
            let argmax_after = argmax(&logits_after.data);
            trace.log(&format!(
                "TRACE phase=decode_run pos={} logits_argmax={}",
                pos_before, argmax_after
            ))?;
            update_kv_cache(&layout, &mut state, &outputs)?;
            state.current_pos += 1;
            last_outputs = outputs;
        }

        let generated_text = tokenizer
            .decode(&generated, false)
            .map_err(|e| format!("decode generated text: {e}"))?;

        println!("Prompt: {}", args.prompt);
        println!("Prompt token ids: {:?}", prompt_ids);
        println!("Generated token ids: {:?}", generated);
        println!("Generated text: {}", generated_text);
        Ok(())
    }

    pub fn main() {
        if let Err(err) = run() {
            eprintln!("error: {err}");
            std::process::exit(1);
        }
    }
}

#[cfg(feature = "onnx-runtime")]
fn main() {
    app::main();
}
