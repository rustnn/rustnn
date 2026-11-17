//! Fixed-window attention with an independent host oracle; no external weights.

use super::*;
use rustnn::mlcontext::{MLGraphBuilder, MLNamedOperands, MLOperandDescriptor};
use rustnn::operator_options::{MLDimension, MLTransposeOptions};

const HEADS: usize = 4;
const WINDOW: usize = 128;
const WIDTH: usize = 16;
const STEPS: usize = 256;
const CACHE_ELEMENTS: usize = HEADS * WINDOW * WIDTH;

fn query() -> Vec<f32> {
    (0..HEADS * WIDTH)
        .map(|i| (i as i32 % 17 - 8) as f32 / 64.0)
        .collect()
}

fn advance(cache: &mut [f32], token: &[f32]) {
    for head in 0..HEADS {
        let start = head * WINDOW * WIDTH;
        cache.copy_within(start + WIDTH..start + WINDOW * WIDTH, start);
        cache[start + (WINDOW - 1) * WIDTH..start + WINDOW * WIDTH]
            .copy_from_slice(&token[head * WIDTH..(head + 1) * WIDTH]);
    }
}

fn oracle(key: &[f32], value: &[f32], query: &[f32]) -> Vec<f32> {
    let mut result = vec![0f32; HEADS * WIDTH];
    for head in 0..HEADS {
        let mut scores = vec![0.0f64; WINDOW];
        for (position, score) in scores.iter_mut().enumerate() {
            *score = (0..WIDTH)
                .map(|i| {
                    f64::from(query[head * WIDTH + i])
                        * f64::from(key[(head * WINDOW + position) * WIDTH + i])
                })
                .sum();
        }
        let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let weights: Vec<_> = scores.iter().map(|score| (score - maximum).exp()).collect();
        let denominator: f64 = weights.iter().sum();
        for i in 0..WIDTH {
            result[head * WIDTH + i] = (weights
                .iter()
                .enumerate()
                .map(|(position, weight)| {
                    weight * f64::from(value[(head * WINDOW + position) * WIDTH + i])
                })
                .sum::<f64>()
                / denominator) as f32;
        }
    }
    result
}

fn tensor(context: &mut MLContext<'_>, shape: &[u64]) -> Result<MLTensor> {
    Ok(context.create_tensor(
        &MLTensorDescriptor::new(MLOperandDataType::Float32, shape.to_vec())
            .to_readable()
            .to_writable(),
    )?)
}

fn run_mode(mode: &str) -> Result<Value> {
    let mut context = MLContext::create(
        &MLContextOptions::new(MLPowerPreference::Default, false)
            .with_rustnn_device_hint(BackendDevice::Coreml {
                device_type: DeviceType::Cpu,
            })
            .with_rustnn_options(storage_options(mode)?),
    )?;
    let cache_shape = [1, HEADS as u64, WINDOW as u64, WIDTH as u64];
    let token_shape = [1, HEADS as u64, 1, WIDTH as u64];
    let mut builder = MLGraphBuilder::new(&mut context)?;
    let cache_descriptor =
        MLOperandDescriptor::new(MLOperandDataType::Float32, cache_shape.to_vec());
    let token_descriptor =
        MLOperandDescriptor::new(MLOperandDataType::Float32, token_shape.to_vec());
    let key = builder.input("past_key", &cache_descriptor)?;
    let value = builder.input("past_value", &cache_descriptor)?;
    let new_key = builder.input("new_key", &token_descriptor)?;
    let new_value = builder.input("new_value", &token_descriptor)?;
    let q = builder.input("query", &token_descriptor)?;
    let sizes = [1, HEADS as u32, (WINDOW - 1) as u32, WIDTH as u32].map(MLDimension::Static);
    let key_tail = builder.slice(key, &[0, 0, 1, 0], &sizes)?;
    let value_tail = builder.slice(value, &[0, 0, 1, 0], &sizes)?;
    let key = builder.concat(&[key_tail, new_key], 2)?;
    let value = builder.concat(&[value_tail, new_value], 2)?;
    let transposed = builder.transpose_with_options(
        key,
        MLTransposeOptions {
            permutation: vec![0, 1, 3, 2],
            ..Default::default()
        },
    )?;
    let scores = builder.matmul(q, transposed)?;
    let weights = builder.softmax(scores, 3)?;
    let attended = builder.matmul(weights, value)?;
    let mut graph = builder.build(&MLNamedOperands::from([
        ("present_key", key),
        ("present_value", value),
        ("attended", attended),
    ]))?;
    drop(builder);
    let keys = [
        tensor(&mut context, &cache_shape)?,
        tensor(&mut context, &cache_shape)?,
    ];
    let values = [
        tensor(&mut context, &cache_shape)?,
        tensor(&mut context, &cache_shape)?,
    ];
    let new_key = tensor(&mut context, &token_shape)?;
    let new_value = tensor(&mut context, &token_shape)?;
    let q = tensor(&mut context, &token_shape)?;
    let attended = tensor(&mut context, &token_shape)?;
    let query = query();
    context.write_tensor(&q, &query)?;
    // `state` keeps both caches inside the model as an MLState: no cache tensor is bound,
    // and the cache cannot be read back, so only the attention output is checked. Every
    // cache value here is a multiple of 1/128 below 2, exact in the float16 state storage.
    let stateful = mode == "state";
    let mut runs = Vec::new();
    // Oracle checking is outside measured dispatch intervals. A separate pass
    // measures cache chaining with no cache inspection or reference computation.
    for verify in [true, false] {
        let mut key_reference = vec![0.0f32; CACHE_ELEMENTS];
        let mut value_reference = key_reference.clone();
        let mut scratch = key_reference.clone();
        context.write_tensor(&keys[0], &key_reference)?;
        context.write_tensor(&values[0], &value_reference)?;
        graph.rustnn_reset_state()?;
        let before = statistics(&context)?;
        let mut seconds = 0.0;
        let mut max_abs = 0.0f64;
        for step in 0..STEPS {
            let previous = step % 2;
            let next = 1 - previous;
            let key_token: Vec<_> = (0..HEADS * WIDTH)
                .map(|i| ((step * 17 + i * 7) % 251) as f32 / 128.0 - 125.0 / 128.0)
                .collect();
            let value_token: Vec<_> = (0..HEADS * WIDTH)
                .map(|i| ((step + i * 5) % 127) as f32 / 64.0 - 63.0 / 64.0)
                .collect();
            let expected = if verify {
                advance(&mut key_reference, &key_token);
                advance(&mut value_reference, &value_token);
                Some(oracle(&key_reference, &value_reference, &query))
            } else {
                None
            };
            let start = Instant::now();
            context.write_tensor(&new_key, &key_token)?;
            context.write_tensor(&new_value, &value_token)?;
            // Bind the previous outputs directly in every storage mode. The
            // baseline's backend copies are reported separately, not inflated
            // by an additional application-level cache read/write round trip.
            let mut inputs = MLNamedTensors::from([
                ("new_key", &new_key),
                ("new_value", &new_value),
                ("query", &q),
            ]);
            let mut outputs = MLNamedTensors::from([("attended", &attended)]);
            if !stateful {
                inputs.insert("past_key", &keys[previous]);
                inputs.insert("past_value", &values[previous]);
                outputs.insert("present_key", &keys[next]);
                outputs.insert("present_value", &values[next]);
            }
            context.dispatch(&mut graph, &inputs, &outputs)?;
            let mut actual = vec![0f32; HEADS * WIDTH];
            context.read_tensor(&attended, &mut actual)?;
            seconds += start.elapsed().as_secs_f64();
            ensure!(
                actual.iter().all(|value| value.is_finite()),
                "nonfinite attention output"
            );
            if let Some(expected) = expected {
                let comparison = compare(&actual, &expected, 1e-5, 1e-5)?;
                ensure!(
                    comparison["failures"] == 0,
                    "{mode} step {step}: {comparison}"
                );
                max_abs = max_abs.max(comparison["max_abs"].as_f64().unwrap());
                let caches = [
                    (&keys[next], &key_reference),
                    (&values[next], &value_reference),
                ];
                for (cache, expected) in caches.into_iter().filter(|_| !stateful) {
                    context.read_tensor(cache, &mut scratch)?;
                    ensure!(
                        scratch == *expected,
                        "{mode} step {step}: cache contents differ"
                    );
                }
            }
        }
        let io = statistics_delta(before, statistics(&context)?)?;
        if !verify {
            let attention_bytes = (STEPS * HEADS * WIDTH * 4) as u64;
            let token_bytes = 2 * attention_bytes;
            ensure!(
                io.host_read_bytes == attention_bytes,
                "cache chaining must not read KV tensors to the host"
            );
            ensure!(
                io.host_write_bytes == token_bytes,
                "cache chaining must not rewrite KV tensors from the host"
            );
            if mode == "baseline" {
                let cache_bytes = (STEPS * 2 * CACHE_ELEMENTS * 4) as u64;
                ensure!(
                    io.input_copy_bytes == cache_bytes + 3 * attention_bytes,
                    "baseline must report backend input copies"
                );
                ensure!(
                    io.native_input_bindings == 0,
                    "baseline bound native inputs"
                );
            } else {
                ensure!(io.input_copy_bytes == 0, "retained inputs were copied");
                let bound = if stateful { 3 } else { 5 };
                ensure!(
                    io.native_input_bindings == (STEPS * bound) as u64,
                    "retained tensors must bind directly on every dispatch"
                );
            }
        }
        runs.push(json!({"phase":if verify {"verification"} else {"measured"},
            "steps":STEPS,"seconds":seconds,"steps_per_second":STEPS as f64/seconds,
            "max_abs":if verify {Some(max_abs)} else {None},"io":statistics_json(io)}));
    }
    Ok(json!({"mode":mode,"runs":runs}))
}

pub(super) fn run() -> Result<Value> {
    let mut results = Vec::new();
    for mode in ["baseline", "persistent", "backings", "state"] {
        eprintln!("Fixed-window attention/cache benchmark: {mode}");
        results.push(run_mode(mode)?);
    }
    Ok(
        json!({"workload":"synthetic fixed-window FP32 attention, not a language model",
        "policy":"cpuOnly","cache_shape":[1,HEADS,WINDOW,WIDTH],
        "timing":"token input updates, dispatch and attended output read; previous KV outputs bind directly in every mode; excludes oracle/cache validation and model compilation",
        "reference":"independent f64 attention and exact sliding-cache checks at every step",
        "results":results}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(all(target_os = "macos", feature = "coreml-runtime"))]
    fn all_storage_modes_preserve_attention_and_cache_contents() {
        let report = run().unwrap();
        assert_eq!(report["results"].as_array().unwrap().len(), 4);
        for result in report["results"].as_array().unwrap() {
            let measured = &result["runs"][1];
            assert_eq!(measured["phase"], "measured");
            assert_eq!(measured["io"]["host_read_bytes"], STEPS * HEADS * WIDTH * 4);
            assert_eq!(
                measured["io"]["host_write_bytes"],
                STEPS * HEADS * WIDTH * 4 * 2
            );
        }
    }

    #[test]
    fn host_oracle_shifts_each_head_independently() {
        let mut cache = vec![0f32; CACHE_ELEMENTS];
        let token: Vec<_> = (0..HEADS * WIDTH).map(|i| i as f32).collect();
        advance(&mut cache, &token);
        advance(&mut cache, &token);
        for head in 0..HEADS {
            let start = head * WINDOW * WIDTH;
            assert!(
                cache[start..start + (WINDOW - 2) * WIDTH]
                    .iter()
                    .all(|&value| value == 0.0)
            );
            assert_eq!(
                &cache[start + (WINDOW - 2) * WIDTH..start + (WINDOW - 1) * WIDTH],
                &token[head * WIDTH..(head + 1) * WIDTH]
            );
        }
        let attended = oracle(&vec![0.0; CACHE_ELEMENTS], &cache, &query());
        for (actual, value) in attended.iter().zip(token) {
            assert_eq!(*actual, value * 2.0 / WINDOW as f32);
        }
    }
}
