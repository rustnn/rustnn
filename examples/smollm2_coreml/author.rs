//! Authors the SmolLM2-135M single-token decode graph from the Hugging Face checkpoint.
//!
//! The graph takes one token and the key/value window of the previous step and returns the
//! logits and the next window. Everything is static: the window keeps the newest `window`
//! positions (newest last), and the caller supplies the rotary `cos`/`sin` row and the
//! attention mask row of the current position.
//!
//! Precision is declared in the graph. Dequantized weights, projections, attention, the MLP
//! products and the key/value window are float16. The residual stream, the RMSNorm statistics
//! and the logits are float32, with a `cast` at every boundary.
//!
//! Weights are quantized here, from the checkpoint's own values:
//!
//! - Projections: signed 8-bit per block of 32 input elements, `scale = max|w| / 127` stored
//!   as float16, round to nearest even, clamp to ±127, stored as uint8 with zero point 128.
//! - Token embedding (shared with the output head): signed 8-bit per block of 64 elements,
//!   float32 `scale = max|w| / 127` for rounding, stored as float16.
//!
//! Both chains are the constant dequantization shape the CoreML converter keeps compressed.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use rustnn::mlcontext::{MLGraphBuilder, MLOperand, MLOperandDescriptor};
use rustnn::operator_enums::MLOperandDataType;
use rustnn::operator_options::{MLDimension, MLReduceOptions, MLTransposeOptions};

/// Architecture constants of `HuggingFaceTB/SmolLM2-135M-Instruct` (`config.json`).
pub const LAYERS: usize = 30;
pub const HIDDEN: usize = 576;
pub const INTERMEDIATE: usize = 1536;
pub const QUERY_HEADS: usize = 9;
pub const KV_HEADS: usize = 3;
pub const HEAD: usize = 64;
pub const VOCABULARY: usize = 49152;
pub const ROPE_THETA: f64 = 100_000.0;
const RMS_EPSILON: f32 = 1e-5;
const PROJECTION_BLOCK: usize = 32;
const EMBEDDING_BLOCK: usize = 64;

const H: MLOperandDataType = MLOperandDataType::Float16;
const F: MLOperandDataType = MLOperandDataType::Float32;

/// Float32 values of one checkpoint tensor. The published checkpoint stores bfloat16.
pub fn tensor_f32(checkpoint: &safetensors::SafeTensors<'_>, name: &str) -> Result<Vec<f32>> {
    let view = checkpoint
        .tensor(name)
        .with_context(|| format!("checkpoint tensor {name}"))?;
    let bytes = view.data();
    Ok(match view.dtype() {
        safetensors::Dtype::BF16 => bytes
            .chunks_exact(2)
            .map(|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16))
            .collect(),
        safetensors::Dtype::F16 => bytes
            .chunks_exact(2)
            .map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32())
            .collect(),
        safetensors::Dtype::F32 => bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
        other => bail!("checkpoint tensor {name} has unsupported type {other:?}"),
    })
}

/// uint8 values (zero point 128) and one float16 scale per block of 32 elements.
pub fn quantize_projection(weight: &[f32]) -> (Vec<u8>, Vec<half::f16>) {
    let mut values = Vec::with_capacity(weight.len());
    let mut scales = Vec::with_capacity(weight.len() / PROJECTION_BLOCK);
    for block in weight.chunks_exact(PROJECTION_BLOCK) {
        let maximum = block.iter().fold(0f32, |m, v| m.max(v.abs()));
        let mut scale = half::f16::from_f32(maximum / 127.0);
        if maximum == 0.0 {
            scale = half::f16::ONE;
        } else if scale == half::f16::ZERO {
            scale = half::f16::from_f32(6.1e-5);
        }
        let divisor = scale.to_f32();
        values.extend(block.iter().map(|v| {
            let q = (v / divisor).round_ties_even().clamp(-127.0, 127.0);
            (q as i16 + 128) as u8
        }));
        scales.push(scale);
    }
    (values, scales)
}

/// int8 values and one float16 scale per block of 64 elements. Rounding uses the float32
/// scale; the graph multiplies by its float16 rounding.
pub fn quantize_embedding(weight: &[f32]) -> (Vec<i8>, Vec<half::f16>) {
    let mut values = Vec::with_capacity(weight.len());
    let mut scales = Vec::with_capacity(weight.len() / EMBEDDING_BLOCK);
    for block in weight.chunks_exact(EMBEDDING_BLOCK) {
        let maximum = block.iter().fold(0f32, |m, v| m.max(v.abs()));
        let scale = if maximum == 0.0 { 1.0 } else { maximum / 127.0 };
        values.extend(
            block
                .iter()
                .map(|v| (v / scale).round_ties_even().clamp(-127.0, 127.0) as i8),
        );
        scales.push(half::f16::from_f32(scale));
    }
    (values, scales)
}

/// Rotary `cos` and `sin` rows of one position: 32 frequencies, repeated for both halves.
pub fn rotary_row(position: usize) -> ([f32; HEAD], [f32; HEAD]) {
    let (mut cos, mut sin) = ([0f32; HEAD], [0f32; HEAD]);
    for i in 0..HEAD / 2 {
        let frequency = (ROPE_THETA.powf(-((2 * i) as f64) / HEAD as f64)) as f32;
        let angle = position as f32 * frequency;
        (cos[i], sin[i]) = (angle.cos(), angle.sin());
        (cos[i + HEAD / 2], sin[i + HEAD / 2]) = (cos[i], sin[i]);
    }
    (cos, sin)
}

/// Mask row of one position. The newest entry is the last column, so position `p` may read the
/// last `p + 1` columns.
pub fn mask_row(position: usize, window: usize) -> Vec<f32> {
    let first = window.saturating_sub(position + 1);
    (0..window)
        .map(|column| if column >= first { 0.0 } else { f32::MIN })
        .collect()
}

fn dimensions(shape: &[usize]) -> Vec<MLDimension> {
    shape
        .iter()
        .map(|&d| MLDimension::Static(d as u32))
        .collect()
}

fn descriptor(data_type: MLOperandDataType, shape: &[usize]) -> MLOperandDescriptor {
    MLOperandDescriptor::new(data_type, shape.iter().map(|&d| d as u64).collect())
}

struct Author<'a, 'b, 'c, 'd> {
    builder: &'a mut MLGraphBuilder<'b, 'c>,
    checkpoint: &'a safetensors::SafeTensors<'d>,
    zero_point: MLOperand,
    epsilon: MLOperand,
}

impl Author<'_, '_, '_, '_> {
    fn reshape(&mut self, x: MLOperand, shape: &[usize]) -> Result<MLOperand> {
        Ok(self.builder.reshape(x, dimensions(shape))?)
    }

    fn permute(&mut self, x: MLOperand, permutation: &[u32]) -> Result<MLOperand> {
        let options = MLTransposeOptions {
            permutation: permutation.to_vec(),
            ..Default::default()
        };
        Ok(self.builder.transpose_with_options(x, options)?)
    }

    /// `x @ W^T` for a checkpoint matrix `[rows, columns]`, in float16.
    fn linear(
        &mut self,
        x: MLOperand,
        name: &str,
        rows: usize,
        columns: usize,
    ) -> Result<MLOperand> {
        let weight = tensor_f32(self.checkpoint, name)?;
        ensure!(weight.len() == rows * columns, "{name}: unexpected size");
        let (values, scales) = quantize_projection(&weight);
        let blocks = columns / PROJECTION_BLOCK;
        let b = &mut *self.builder;
        let stored = b.constant_from_bytes(
            &descriptor(MLOperandDataType::Uint8, &[rows, columns]),
            values,
        )?;
        let scale = b.constant_from_slice(&descriptor(H, &[rows, blocks, 1]), &scales)?;
        let centered = b.cast(stored, H)?;
        let centered = b.sub(centered, self.zero_point)?;
        let grouped = b.reshape(centered, dimensions(&[rows, blocks, PROJECTION_BLOCK]))?;
        let scaled = b.mul(grouped, scale)?;
        let matrix = b.reshape(scaled, dimensions(&[rows, columns]))?;
        let transposed = self.permute(matrix, &[1, 0])?;
        Ok(self.builder.matmul(x, transposed)?)
    }

    /// RMSNorm in float32: `x / sqrt(mean(x^2) + eps) * weight`.
    fn rms_norm(&mut self, x: MLOperand, name: &str) -> Result<MLOperand> {
        let weight = tensor_f32(self.checkpoint, name)?;
        let b = &mut *self.builder;
        let weight = b.constant_from_slice(&descriptor(F, &[HIDDEN]), &weight)?;
        let squared = b.mul(x, x)?;
        let options = MLReduceOptions {
            axes: Some(vec![2]),
            keep_dimensions: true,
            ..Default::default()
        };
        let mean = b.reduce_mean_with_options(squared, options)?;
        let shifted = b.add(mean, self.epsilon)?;
        let root = b.sqrt(shifted)?;
        let normalized = b.div(x, root)?;
        Ok(b.mul(normalized, weight)?)
    }

    /// Rotates `[1, heads, 1, 64]` by the position's `cos`/`sin` rows.
    fn rotate(
        &mut self,
        x: MLOperand,
        heads: usize,
        cos: MLOperand,
        sin: MLOperand,
    ) -> Result<MLOperand> {
        let b = &mut *self.builder;
        let half = dimensions(&[1, heads, 1, HEAD / 2]);
        let low = b.slice(x, &[0, 0, 0, 0], &half)?;
        let high = b.slice(x, &[0, 0, 0, (HEAD / 2) as u32], &half)?;
        let negated = b.neg(high)?;
        let rotated = b.concat(&[negated, low], 3)?;
        let direct = b.mul(x, cos)?;
        let crossed = b.mul(rotated, sin)?;
        Ok(b.add(direct, crossed)?)
    }
}

/// Records the decode graph in `builder` and returns its outputs by name: `logits` and one
/// `next_key_N` / `next_value_N` per layer for the `key_N` / `value_N` inputs.
pub fn author(
    builder: &mut MLGraphBuilder<'_, '_>,
    checkpoint: &safetensors::SafeTensors<'_>,
    window: usize,
    layers: usize,
) -> Result<BTreeMap<String, MLOperand>> {
    ensure!(window >= 2, "the window needs at least two positions");
    ensure!(
        (1..=LAYERS).contains(&layers),
        "the model has {LAYERS} layers"
    );
    let token = builder.input("token", &descriptor(MLOperandDataType::Int32, &[1, 1]))?;
    let cos = builder.input("cos", &descriptor(F, &[1, 1, 1, HEAD]))?;
    let sin = builder.input("sin", &descriptor(F, &[1, 1, 1, HEAD]))?;
    let mask = builder.input("mask", &descriptor(F, &[1, 1, 1, window]))?;
    let cos = builder.cast(cos, H)?;
    let sin = builder.cast(sin, H)?;
    let mask = builder.cast(mask, H)?;
    let attention_scale =
        builder.constant_from_slice(&descriptor(H, &[]), &[half::f16::from_f32(0.125)])?;
    let zero_point =
        builder.constant_from_slice(&descriptor(H, &[]), &[half::f16::from_f32(128.0)])?;
    let epsilon = builder.constant_from_slice(&descriptor(F, &[]), &[RMS_EPSILON])?;

    // Token embedding, also the output head.
    let embedding = tensor_f32(checkpoint, "model.embed_tokens.weight")?;
    ensure!(
        embedding.len() == VOCABULARY * HIDDEN,
        "unexpected embedding size"
    );
    let (values, scales) = quantize_embedding(&embedding);
    drop(embedding);
    let blocks = HIDDEN / EMBEDDING_BLOCK;
    let stored = builder.constant_from_slice(
        &descriptor(MLOperandDataType::Int8, &[VOCABULARY, HIDDEN]),
        &values,
    )?;
    let scale = builder.constant_from_slice(&descriptor(H, &[VOCABULARY, blocks, 1]), &scales)?;
    let widened = builder.cast(stored, H)?;
    let grouped = builder.reshape(widened, dimensions(&[VOCABULARY, blocks, EMBEDDING_BLOCK]))?;
    let scaled = builder.mul(grouped, scale)?;
    let embedding = builder.reshape(scaled, dimensions(&[VOCABULARY, HIDDEN]))?;

    let mut a = Author {
        builder,
        checkpoint,
        zero_point,
        epsilon,
    };
    let row = a.builder.gather(embedding, token)?;
    let mut residual = a.builder.cast(row, F)?;
    let mut outputs = BTreeMap::new();

    for layer in 0..layers {
        let prefix = format!("model.layers.{layer}");
        let normalized = a.rms_norm(residual, &format!("{prefix}.input_layernorm.weight"))?;
        let normalized = a.builder.cast(normalized, H)?;

        let query = a.linear(
            normalized,
            &format!("{prefix}.self_attn.q_proj.weight"),
            HIDDEN,
            HIDDEN,
        )?;
        let query = a.reshape(query, &[1, 1, QUERY_HEADS, HEAD])?;
        let query = a.permute(query, &[0, 2, 1, 3])?;
        let query = a.rotate(query, QUERY_HEADS, cos, sin)?;

        let kv_width = KV_HEADS * HEAD;
        let key = a.linear(
            normalized,
            &format!("{prefix}.self_attn.k_proj.weight"),
            kv_width,
            HIDDEN,
        )?;
        let key = a.reshape(key, &[1, 1, KV_HEADS, HEAD])?;
        let key = a.permute(key, &[0, 2, 1, 3])?;
        let key = a.rotate(key, KV_HEADS, cos, sin)?;

        let value = a.linear(
            normalized,
            &format!("{prefix}.self_attn.v_proj.weight"),
            kv_width,
            HIDDEN,
        )?;
        let value = a.reshape(value, &[1, 1, KV_HEADS, HEAD])?;
        let value = a.permute(value, &[0, 2, 1, 3])?;

        // Slide the window: drop the oldest position, append this one.
        let shape = descriptor(H, &[1, KV_HEADS, window, HEAD]);
        let kept = dimensions(&[1, KV_HEADS, window - 1, HEAD]);
        let (key_name, value_name) = (format!("key_{layer}"), format!("value_{layer}"));
        let past_key = a.builder.input(&key_name, &shape)?;
        let past_value = a.builder.input(&value_name, &shape)?;
        let past_key = a.builder.slice(past_key, &[0, 0, 1, 0], &kept)?;
        let past_value = a.builder.slice(past_value, &[0, 0, 1, 0], &kept)?;
        let next_key = a.builder.concat(&[past_key, key], 2)?;
        let next_value = a.builder.concat(&[past_value, value], 2)?;
        outputs.insert(format!("next_{key_name}"), next_key);
        outputs.insert(format!("next_{value_name}"), next_value);

        // Grouped-query attention: each key/value head serves three query heads.
        let group = QUERY_HEADS / KV_HEADS;
        let keys = a.reshape(next_key, &[1, KV_HEADS, 1, window, HEAD])?;
        let keys = a.permute(keys, &[0, 1, 2, 4, 3])?;
        let values = a.reshape(next_value, &[1, KV_HEADS, 1, window, HEAD])?;
        let queries = a.reshape(query, &[1, KV_HEADS, group, 1, HEAD])?;
        let scores = a.builder.matmul(queries, keys)?;
        let scores = a.builder.mul(scores, attention_scale)?;
        let scores = a.builder.add(scores, mask)?;
        let weights = a.builder.softmax(scores, 4)?;
        let attended = a.builder.matmul(weights, values)?;
        let attended = a.reshape(attended, &[1, QUERY_HEADS, 1, HEAD])?;
        let attended = a.permute(attended, &[0, 2, 1, 3])?;
        let attended = a.reshape(attended, &[1, 1, HIDDEN])?;
        let projected = a.linear(
            attended,
            &format!("{prefix}.self_attn.o_proj.weight"),
            HIDDEN,
            HIDDEN,
        )?;
        let projected = a.builder.cast(projected, F)?;
        residual = a.builder.add(residual, projected)?;

        let normalized = a.rms_norm(
            residual,
            &format!("{prefix}.post_attention_layernorm.weight"),
        )?;
        let normalized = a.builder.cast(normalized, H)?;
        let gate = a.linear(
            normalized,
            &format!("{prefix}.mlp.gate_proj.weight"),
            INTERMEDIATE,
            HIDDEN,
        )?;
        let up = a.linear(
            normalized,
            &format!("{prefix}.mlp.up_proj.weight"),
            INTERMEDIATE,
            HIDDEN,
        )?;
        let gate_sigmoid = a.builder.sigmoid(gate)?;
        let activated = a.builder.mul(gate, gate_sigmoid)?;
        let activated = a.builder.cast(activated, F)?;
        let up = a.builder.cast(up, F)?;
        let product = a.builder.mul(activated, up)?;
        let product = a.builder.cast(product, H)?;
        let down = a.linear(
            product,
            &format!("{prefix}.mlp.down_proj.weight"),
            HIDDEN,
            INTERMEDIATE,
        )?;
        let down = a.builder.cast(down, F)?;
        residual = a.builder.add(residual, down)?;
    }

    let normalized = a.rms_norm(residual, "model.norm.weight")?;
    let normalized = a.builder.cast(normalized, H)?;
    let head = a.permute(embedding, &[1, 0])?;
    let logits = a.builder.matmul(normalized, head)?;
    let logits = a.builder.cast(logits, F)?;
    outputs.insert("logits".to_owned(), logits);
    Ok(outputs)
}

/// Reads the checkpoint, authors the graph and writes `model.webnn` with its weights.
pub fn save(
    builder: &mut MLGraphBuilder<'_, '_>,
    checkpoint: &Path,
    window: usize,
    layers: usize,
    bundle: &Path,
) -> Result<()> {
    let bytes = std::fs::read(checkpoint.join("model.safetensors"))
        .with_context(|| format!("{}/model.safetensors", checkpoint.display()))?;
    let tensors = safetensors::SafeTensors::deserialize(&bytes)?;
    let authored = author(builder, &tensors, window, layers)?;
    std::fs::create_dir_all(bundle)?;
    let outputs = authored
        .iter()
        .map(|(name, operand)| (name.as_str(), *operand))
        .collect();
    builder.rustnn_save_webnn(&outputs, bundle.join("model.webnn"))?;
    std::fs::copy(
        checkpoint.join("tokenizer.json"),
        bundle.join("tokenizer.json"),
    )
    .with_context(|| format!("{}/tokenizer.json", checkpoint.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn projection_blocks_round_to_nearest_even_around_128() {
        let mut weight = vec![0f32; 2 * PROJECTION_BLOCK];
        weight[0] = 1.0;
        weight[1] = -1.0;
        weight[2] = 0.5;
        let (values, scales) = quantize_projection(&weight);
        assert_eq!(scales[0], half::f16::from_f32(1.0 / 127.0));
        assert_eq!(&values[..4], &[255, 1, 192, 128]);
        // An all-zero block keeps scale one and the zero point.
        assert_eq!(scales[1], half::f16::ONE);
        assert!(values[PROJECTION_BLOCK..].iter().all(|&v| v == 128));
    }

    #[test]
    fn embedding_blocks_use_the_float32_scale_for_rounding() {
        let mut weight = vec![0f32; EMBEDDING_BLOCK];
        weight[0] = -0.3;
        weight[1] = 0.1;
        let (values, scales) = quantize_embedding(&weight);
        assert_eq!(&values[..3], &[-127, 42, 0]);
        assert_eq!(scales, [half::f16::from_f32(0.3 / 127.0)]);
    }

    #[test]
    fn position_rows_follow_the_sliding_window() {
        assert_eq!(mask_row(0, 4), [f32::MIN, f32::MIN, f32::MIN, 0.0]);
        assert_eq!(mask_row(2, 4), [f32::MIN, 0.0, 0.0, 0.0]);
        assert_eq!(mask_row(9, 4), [0.0; 4]);
        let (cos, sin) = rotary_row(0);
        assert!(cos.iter().all(|&c| c == 1.0) && sin.iter().all(|&s| s == 0.0));
        let (cos, sin) = rotary_row(3);
        assert_eq!((cos[0], sin[0]), (3f32.cos(), 3f32.sin()));
        assert_eq!((cos[32], sin[32]), (cos[0], sin[0]));
    }
}
