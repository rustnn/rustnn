#[cfg(not(feature = "onnx-runtime"))]
fn main() {
    eprintln!("This example requires `--features onnx-runtime`.");
    eprintln!("Run: cargo run --features onnx-runtime --example gpt2_webnn_rust -- --help");
}

#[cfg(feature = "onnx-runtime")]
mod app {
    use std::collections::HashMap;
    use std::path::PathBuf;

    use clap::Parser;
    use prost::Message;
    use rustnn::executors::onnx::{
        OnnxInput, OnnxOutputWithData, TensorData, run_onnx_with_inputs_checked,
    };
    use rustnn::protos::onnx::ModelProto;
    use rustnn::{ContextProperties, ConverterRegistry, GraphValidator, load_graph_from_path};
    use serde_json::Value;

    #[derive(Parser, Debug)]
    #[command(about = "Pure Rust GPT-2 generation from .webnn + weights/manifest")]
    struct Args {
        #[arg(long, default_value = "/tmp/gpt2_cached64.webnn")]
        model: PathBuf,
        #[arg(
            long,
            default_value = "7454,2402,257,640",
            help = "Comma-separated GPT-2 token IDs for the prompt"
        )]
        prompt_ids: String,
        #[arg(long, default_value_t = 20)]
        max_new_tokens: usize,
        #[arg(
            long,
            default_value = "/tmp/gpt2_vocab.json",
            help = "Path to GPT-2 vocab.json"
        )]
        vocab: PathBuf,
        #[arg(long, default_value_t = 500_000_000_000usize)]
        tensor_limit: usize,
    }

    #[derive(Debug, Clone)]
    struct Gpt2Layout {
        num_layers: usize,
        num_heads: usize,
        cache_len: usize,
        head_dim: usize,
        logits_name: String,
    }

    #[derive(Debug, Clone)]
    struct InputsByName {
        i64_inputs: HashMap<String, Vec<i64>>,
        f32_inputs: HashMap<String, Vec<f32>>,
        shapes: HashMap<String, Vec<usize>>,
    }

    fn dim_to_usize(dim: &rustnn::graph::Dimension) -> usize {
        match dim {
            rustnn::graph::Dimension::Static(v) => *v as usize,
            rustnn::graph::Dimension::Dynamic(d) => d.max_size as usize,
        }
    }

    fn detect_layout(artifacts: &rustnn::ValidationArtifacts) -> Result<Gpt2Layout, String> {
        let mut num_layers = 0usize;
        let mut num_heads = None;
        let mut cache_len = None;
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
                    cache_len = Some(dim_to_usize(&desc.shape[2]));
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

        let layout = Gpt2Layout {
            num_layers,
            num_heads: num_heads.ok_or_else(|| "failed to detect num_heads".to_string())?,
            cache_len: cache_len.ok_or_else(|| "failed to detect cache_len".to_string())?,
            head_dim: head_dim.ok_or_else(|| "failed to detect head_dim".to_string())?,
            logits_name: logits_name.ok_or_else(|| "failed to detect logits output".to_string())?,
        };
        Ok(layout)
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

    fn parse_prompt_ids(raw: &str) -> Result<Vec<u32>, String> {
        let ids: Result<Vec<u32>, _> = raw
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(|s| s.parse::<u32>())
            .collect();
        let ids = ids.map_err(|e| format!("invalid --prompt-ids: {e}"))?;
        if ids.is_empty() {
            return Err("prompt ids are empty".to_string());
        }
        Ok(ids)
    }

    fn gpt2_bytes_to_unicode() -> HashMap<u8, char> {
        let mut bs: Vec<u8> = (33u8..=126u8)
            .chain(161u8..=172u8)
            .chain(174u8..=255u8)
            .collect();
        let mut cs: Vec<u32> = bs.iter().map(|&b| b as u32).collect();
        let mut n = 0u32;
        for b in 0u8..=255u8 {
            if !bs.contains(&b) {
                bs.push(b);
                cs.push(256 + n);
                n += 1;
            }
        }
        bs.into_iter()
            .zip(cs)
            .map(|(b, c)| (b, char::from_u32(c).unwrap_or('\u{FFFD}')))
            .collect()
    }

    fn load_reverse_vocab(vocab_path: &PathBuf) -> Result<HashMap<u32, String>, String> {
        let text = std::fs::read_to_string(vocab_path)
            .map_err(|e| format!("read vocab {}: {e}", vocab_path.display()))?;
        let value: Value =
            serde_json::from_str(&text).map_err(|e| format!("parse vocab json: {e}"))?;
        let obj = value
            .as_object()
            .ok_or_else(|| "vocab.json is not a JSON object".to_string())?;
        let mut rev = HashMap::with_capacity(obj.len());
        for (token, id_val) in obj {
            let id = id_val
                .as_u64()
                .ok_or_else(|| format!("non-integer token id in vocab for token `{token}`"))?
                as u32;
            rev.insert(id, token.clone());
        }
        Ok(rev)
    }

    fn decode_gpt2_tokens(ids: &[u32], rev_vocab: &HashMap<u32, String>) -> String {
        let byte_to_unicode = gpt2_bytes_to_unicode();
        let unicode_to_byte: HashMap<char, u8> =
            byte_to_unicode.into_iter().map(|(b, ch)| (ch, b)).collect();
        let mut bytes = Vec::new();
        for id in ids {
            if let Some(token) = rev_vocab.get(id) {
                for ch in token.chars() {
                    if let Some(b) = unicode_to_byte.get(&ch) {
                        bytes.push(*b);
                    }
                }
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    fn build_step_inputs(layout: &Gpt2Layout, token_id: i64, pos: usize) -> InputsByName {
        let mut i64_inputs = HashMap::new();
        i64_inputs.insert("input_ids".to_string(), vec![token_id]);
        i64_inputs.insert("position_ids".to_string(), vec![pos as i64]);

        let mut mask = vec![0_i64; layout.cache_len + 1];
        for m in mask.iter_mut().take(pos + 1) {
            *m = 1;
        }
        i64_inputs.insert("attention_mask".to_string(), mask);

        let mut f32_inputs = HashMap::new();
        let mut shapes = HashMap::new();
        shapes.insert("input_ids".to_string(), vec![1, 1]);
        shapes.insert("position_ids".to_string(), vec![1, 1]);
        shapes.insert("attention_mask".to_string(), vec![1, layout.cache_len + 1]);

        let cache_elems = layout.num_heads * layout.cache_len * layout.head_dim;
        for layer in 0..layout.num_layers {
            let k = format!("past_key_values_{}_key", layer);
            let v = format!("past_key_values_{}_value", layer);
            f32_inputs.insert(k.clone(), vec![0.0_f32; cache_elems]);
            f32_inputs.insert(v.clone(), vec![0.0_f32; cache_elems]);
            let shape = vec![1, layout.num_heads, layout.cache_len, layout.head_dim];
            shapes.insert(k, shape.clone());
            shapes.insert(v, shape);
        }

        InputsByName {
            i64_inputs,
            f32_inputs,
            shapes,
        }
    }

    fn write_cache_slice(
        cache: &mut [f32],
        present: &[f32],
        num_heads: usize,
        cache_len: usize,
        seq_len: usize,
        head_dim: usize,
        pos: usize,
    ) {
        for h in 0..num_heads {
            let cache_base = (h * cache_len + pos) * head_dim;
            let present_base = (h * seq_len + pos) * head_dim;
            cache[cache_base..cache_base + head_dim]
                .copy_from_slice(&present[present_base..present_base + head_dim]);
        }
    }

    fn update_kv_cache(
        layout: &Gpt2Layout,
        step_inputs: &mut InputsByName,
        outputs: &[OnnxOutputWithData],
        pos: usize,
    ) -> Result<(), String> {
        for layer in 0..layout.num_layers {
            let key_name = format!("present_{}_key", layer);
            let value_name = format!("present_{}_value", layer);

            let present_k = outputs
                .iter()
                .find(|o| o.name == key_name)
                .ok_or_else(|| format!("missing output: {key_name}"))?;
            let present_v = outputs
                .iter()
                .find(|o| o.name == value_name)
                .ok_or_else(|| format!("missing output: {value_name}"))?;

            if present_k.shape.len() != 4 {
                return Err(format!(
                    "unexpected present_k rank for {key_name}: {:?}",
                    present_k.shape
                ));
            }
            let seq_len = present_k.shape[2];
            let cache_k_name = format!("past_key_values_{}_key", layer);
            let cache_v_name = format!("past_key_values_{}_value", layer);

            {
                let cache_k = step_inputs
                    .f32_inputs
                    .get_mut(&cache_k_name)
                    .ok_or_else(|| format!("missing cache input: {cache_k_name}"))?;
                write_cache_slice(
                    cache_k,
                    &present_k.data,
                    layout.num_heads,
                    layout.cache_len,
                    seq_len,
                    layout.head_dim,
                    pos,
                );
            }
            {
                let cache_v = step_inputs
                    .f32_inputs
                    .get_mut(&cache_v_name)
                    .ok_or_else(|| format!("missing cache input: {cache_v_name}"))?;
                write_cache_slice(
                    cache_v,
                    &present_v.data,
                    layout.num_heads,
                    layout.cache_len,
                    seq_len,
                    layout.head_dim,
                    pos,
                );
            }
        }
        Ok(())
    }

    fn to_onnx_inputs(order: &[String], by_name: &InputsByName) -> Result<Vec<OnnxInput>, String> {
        let mut out = Vec::with_capacity(order.len());
        for name in order {
            let shape = by_name
                .shapes
                .get(name)
                .ok_or_else(|| format!("missing shape for input {name}"))?
                .clone();
            if let Some(v) = by_name.i64_inputs.get(name) {
                out.push(OnnxInput {
                    name: name.clone(),
                    shape,
                    data: TensorData::Int64(v.clone()),
                });
            } else if let Some(v) = by_name.f32_inputs.get(name) {
                out.push(OnnxInput {
                    name: name.clone(),
                    shape,
                    data: TensorData::Float32(v.clone()),
                });
            } else {
                return Err(format!("missing input value for {name}"));
            }
        }
        Ok(out)
    }

    fn run() -> Result<(), String> {
        let args = Args::parse();

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
        let prompt_ids = parse_prompt_ids(&args.prompt_ids)?;
        let reverse_vocab = load_reverse_vocab(&args.vocab)?;
        if prompt_ids.is_empty() {
            return Err("prompt produced zero tokens".to_string());
        }
        if prompt_ids.len() >= layout.cache_len {
            return Err(format!(
                "prompt too long: {} tokens (max < {})",
                prompt_ids.len(),
                layout.cache_len
            ));
        }

        let mut state = build_step_inputs(&layout, prompt_ids[0] as i64, 0);
        let mut current_pos = 0usize;
        let mut last_outputs = Vec::new();

        for token_id in &prompt_ids {
            state
                .i64_inputs
                .insert("input_ids".to_string(), vec![*token_id as i64]);
            state
                .i64_inputs
                .insert("position_ids".to_string(), vec![current_pos as i64]);
            if let Some(mask) = state.i64_inputs.get_mut("attention_mask") {
                mask[current_pos] = 1;
            }

            let inputs = to_onnx_inputs(&input_order, &state)?;
            let outputs = run_onnx_with_inputs_checked(
                &converted.data,
                inputs,
                &artifacts.input_names_to_descriptors,
                &artifacts.output_names_to_descriptors,
            )
            .map_err(|e| format!("onnx run (prefill pos={current_pos}): {e}"))?;
            update_kv_cache(&layout, &mut state, &outputs, current_pos)?;
            last_outputs = outputs;
            current_pos += 1;
        }

        let mut generated_ids = Vec::new();
        for _ in 0..args.max_new_tokens {
            let logits = last_outputs
                .iter()
                .find(|o| o.name == layout.logits_name)
                .ok_or_else(|| format!("missing logits output: {}", layout.logits_name))?;
            let next_id = argmax(&logits.data) as u32;
            generated_ids.push(next_id);

            if current_pos >= layout.cache_len {
                break;
            }

            state
                .i64_inputs
                .insert("input_ids".to_string(), vec![next_id as i64]);
            state
                .i64_inputs
                .insert("position_ids".to_string(), vec![current_pos as i64]);
            if let Some(mask) = state.i64_inputs.get_mut("attention_mask") {
                mask[current_pos] = 1;
            }

            let inputs = to_onnx_inputs(&input_order, &state)?;
            let outputs = run_onnx_with_inputs_checked(
                &converted.data,
                inputs,
                &artifacts.input_names_to_descriptors,
                &artifacts.output_names_to_descriptors,
            )
            .map_err(|e| format!("onnx run (decode pos={current_pos}): {e}"))?;
            update_kv_cache(&layout, &mut state, &outputs, current_pos)?;
            last_outputs = outputs;
            current_pos += 1;
        }

        let prompt_text = decode_gpt2_tokens(&prompt_ids, &reverse_vocab);
        let generated_text = decode_gpt2_tokens(&generated_ids, &reverse_vocab);

        println!("Prompt ids: {:?}", prompt_ids);
        println!("Prompt text: {}", prompt_text);
        println!("Generated token ids: {:?}", generated_ids);
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
