//! CoreML attention and KV-cache probes through the real RustNN tensor API.
//!
//! The default fixed-window attention benchmark needs no fixture. External model
//! graphs and typed references can also be supplied; neither mode is a tokenizer
//! or a held-out language-quality benchmark.

use std::collections::BTreeMap;
use std::fs;
use std::hash::Hasher;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, ensure};
use rustnn::backend_selection::DeviceType;
use rustnn::graph::{DataType, GraphInfo, OperandDescriptor};
use rustnn::mlcontext::{
    BackendDevice, MLContext, MLContextOptions, MLGraph, MLNamedTensors, MLPowerPreference,
    MLTensor, MLTensorDescriptor,
};
use rustnn::mlcontextoptions::{BackendStatistics, CoremlTensorStatistics, RustNNOptions};
use rustnn::operator_enums::MLOperandDataType;
use rustnn::{ContextProperties, GraphValidator};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

#[path = "coreml_kv_benchmark/self_contained.rs"]
mod self_contained;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct BenchmarkConfig {
    pub fixture_root: PathBuf,
    pub output: PathBuf,
    pub policy: String,
    pub mode: String,
    pub streams: Vec<String>,
    pub steps: Option<usize>,
    pub warmups: usize,
    pub repeats: usize,
    #[serde(default)]
    pub min_measured_seconds: Option<f64>,
    #[serde(default)]
    pub smollm_model: Option<PathBuf>,
}

#[derive(Clone, Deserialize)]
struct Step {
    tokens: Vec<i32>,
}

#[derive(Clone, Deserialize)]
struct Stream {
    name: String,
    steps: Vec<Step>,
}

#[derive(Deserialize)]
struct Fixture {
    vocab: usize,
    atol: f64,
    rtol: f64,
    streams: Vec<Stream>,
}

fn graph(path: &Path) -> Result<GraphInfo> {
    let graph: GraphInfo = serde_json::from_slice(&fs::read(path)?)
        .with_context(|| format!("deserialize {}", path.display()))?;
    GraphValidator::new(&graph, ContextProperties::default())
        .validate()
        .with_context(|| format!("validate {}", path.display()))?;
    Ok(graph)
}

fn dtype(value: DataType) -> Result<MLOperandDataType> {
    Ok(match value {
        DataType::Float32 => MLOperandDataType::Float32,
        DataType::Int32 => MLOperandDataType::Int32,
        DataType::Int64 => MLOperandDataType::Int64,
        _ => anyhow::bail!("fixture dtype {value:?} is not supported by this probe"),
    })
}

fn allocate(context: &mut MLContext<'_>, descriptor: &OperandDescriptor) -> Result<MLTensor> {
    let max: Vec<u64> = descriptor
        .static_or_max_shape()
        .into_iter()
        .map(u64::from)
        .collect();
    let initial: Vec<u64> = descriptor
        .shape
        .iter()
        .map(|d| match d {
            rustnn::graph::Dimension::Static(n) => u64::from(*n),
            rustnn::graph::Dimension::Dynamic(_) => 1,
        })
        .collect();
    let mut tensor = context.create_tensor(
        &MLTensorDescriptor::new(dtype(descriptor.data_type)?, initial)
            .to_readable()
            .to_writable(),
    )?;
    context.rustnn_set_tensor_capacity(&mut tensor, &max)?;
    Ok(tensor)
}

fn tensors(
    context: &mut MLContext<'_>,
    graph: &GraphInfo,
    ids: &[u32],
) -> Result<BTreeMap<String, MLTensor>> {
    ids.iter()
        .map(|&id| {
            let operand = &graph.operands[id as usize];
            let name = operand.name.clone().context("unnamed graph binding")?;
            Ok((name, allocate(context, &operand.descriptor)?))
        })
        .collect()
}

fn compare(actual: &[f32], expected: &[f32], atol: f64, rtol: f64) -> Result<Value> {
    ensure!(
        actual.len() == expected.len(),
        "reference element count mismatch"
    );
    let mut failures = 0;
    let mut maximum = 0.0_f64;
    for (&a, &e) in actual.iter().zip(expected) {
        ensure!(a.is_finite() && e.is_finite(), "nonfinite result/reference");
        let delta = (f64::from(a) - f64::from(e)).abs();
        maximum = maximum.max(delta);
        failures += usize::from(delta > atol + rtol * f64::from(e).abs());
    }
    Ok(json!({"failures": failures, "max_abs": maximum, "elements": actual.len()}))
}

fn floats(path: &Path) -> Result<Vec<f32>> {
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let (chunks, remainder) = bytes.as_chunks::<4>();
    ensure!(remainder.is_empty(), "unaligned float reference");
    Ok(chunks
        .iter()
        .map(|bytes| f32::from_le_bytes(*bytes))
        .collect())
}

fn argmax(values: &[f32]) -> usize {
    let mut best = 0;
    for index in 1..values.len() {
        if values[index] > values[best] {
            best = index;
        }
    }
    best
}

fn resize_write<T: bytemuck::Pod>(
    context: &mut MLContext<'_>,
    tensor: &mut MLTensor,
    shape: &[u64],
    contents: &[T],
) -> Result<()> {
    context.rustnn_resize_tensor(tensor, shape)?;
    context.write_tensor(tensor, contents)?;
    Ok(())
}

struct Runner<'a> {
    context: MLContext<'static>,
    prefill: MLGraph<'static>,
    decode: MLGraph<'static>,
    inputs: BTreeMap<String, MLTensor>,
    outputs: [BTreeMap<String, MLTensor>; 2],
    root: &'a Path,
    vocab: usize,
    atol: f64,
    rtol: f64,
    mode: String,
    report_path: &'a Path,
}

fn statistics(context: &MLContext<'_>) -> Result<CoremlTensorStatistics> {
    match context.rustnn_backend_statistics() {
        Some(BackendStatistics::Coreml(statistics)) => Ok(statistics),
        _ => anyhow::bail!("CoreML counters unavailable"),
    }
}

fn statistics_json(s: CoremlTensorStatistics) -> Value {
    json!({"last_compute_units":s.last_compute_units,"native_allocations":s.native_allocations,
        "host_read_bytes":s.host_read_bytes,"host_write_bytes":s.host_write_bytes,
        "native_input_bindings":s.native_input_bindings,"input_copy_bytes":s.input_copy_bytes,
        "output_backings_requested":s.output_backings_requested,"output_backings_accepted":s.output_backings_accepted,
        "output_copy_bytes":s.output_copy_bytes})
}

fn statistics_delta(
    before: CoremlTensorStatistics,
    after: CoremlTensorStatistics,
) -> Result<CoremlTensorStatistics> {
    let mut delta = CoremlTensorStatistics::default();
    macro_rules! subtract {
        ($($name:ident),*) => { $(delta.$name = after.$name.checked_sub(before.$name).context(concat!("counter decreased: ",stringify!($name)))?;)* };
    }
    subtract!(
        native_allocations,
        host_read_bytes,
        host_write_bytes,
        native_input_bindings,
        input_copy_bytes,
        output_backings_requested,
        output_backings_accepted,
        output_copy_bytes
    );
    delta.last_compute_units = after.last_compute_units;
    Ok(delta)
}

fn add_statistics(total: &mut CoremlTensorStatistics, delta: CoremlTensorStatistics) -> Result<()> {
    macro_rules! add {
        ($($name:ident),*) => { $(total.$name = total.$name.checked_add(delta.$name).context(concat!("counter overflow: ",stringify!($name)))?;)* };
    }
    add!(
        native_allocations,
        host_read_bytes,
        host_write_bytes,
        native_input_bindings,
        input_copy_bytes,
        output_backings_requested,
        output_backings_accepted,
        output_copy_bytes
    );
    total.last_compute_units = delta.last_compute_units;
    Ok(())
}

#[derive(Default, Serialize)]
struct LogicalPayload {
    host_inputs: u64,
    cache_inputs: u64,
    logits_output: u64,
    cache_outputs: u64,
}

impl LogicalPayload {
    fn step(sequence: usize, prefix: usize, cached: bool, vocab: usize) -> Self {
        Self {
            host_inputs: (8 * sequence + 4 * sequence * prefix + 4) as u64,
            cache_inputs: if cached {
                4096 * (prefix - 1) as u64
            } else {
                0
            },
            logits_output: 4 * vocab as u64,
            cache_outputs: 4096 * prefix as u64,
        }
    }

    fn add(&mut self, other: &Self) {
        self.host_inputs += other.host_inputs;
        self.cache_inputs += other.cache_inputs;
        self.logits_output += other.logits_output;
        self.cache_outputs += other.cache_outputs;
    }
}

fn check_copy_accounting(
    mode: &str,
    stats: CoremlTensorStatistics,
    payload: &LogicalPayload,
    cached: bool,
) -> Result<()> {
    ensure!(
        stats.host_write_bytes == payload.host_inputs,
        "unexpected host input/cache writes"
    );
    ensure!(
        stats.host_read_bytes == payload.logits_output,
        "timed path read a cache or omitted logits"
    );
    ensure!(
        stats.native_allocations == 0,
        "preallocated tensor storage grew during the stream"
    );
    let outputs = payload.logits_output + payload.cache_outputs;
    match mode {
        "baseline" => {
            ensure!(
                stats.input_copy_bytes == payload.host_inputs + payload.cache_inputs,
                "baseline input materialization mismatch"
            );
            ensure!(
                stats.output_copy_bytes == outputs,
                "baseline output materialization mismatch"
            );
            ensure!(
                stats.native_input_bindings == 0
                    && stats.output_backings_requested == 0
                    && stats.output_backings_accepted == 0,
                "baseline used optimized storage"
            );
        }
        "persistent" | "backings" => {
            ensure!(
                stats.input_copy_bytes == 0,
                "compatible persistent inputs were materialized"
            );
            ensure!(
                stats.native_input_bindings == if cached { 20 } else { 4 },
                "persistent input binding count mismatch"
            );
            ensure!(
                stats.output_backings_accepted <= stats.output_backings_requested
                    && stats.output_backings_requested <= 17,
                "invalid output backing counters"
            );
            ensure!(
                stats.output_copy_bytes <= outputs,
                "output materialization exceeds logical payload"
            );
            if mode == "persistent" {
                ensure!(
                    stats.output_backings_requested == 0
                        && stats.output_backings_accepted == 0
                        && stats.output_copy_bytes == outputs,
                    "persistent copy-output path skipped a required copy"
                );
            } else {
                // All sixteen cache outputs have one size, while the logits have
                // another. Derive which returned objects were accepted without
                // confusing native object identity with a matching data pointer.
                let saved = outputs - stats.output_copy_bytes;
                let cache_bytes = payload.cache_outputs / 16;
                let accepted = stats.output_backings_accepted;
                let cache_only = accepted <= 16 && saved == accepted * cache_bytes;
                let with_logits = (1..=17).contains(&accepted)
                    && saved == payload.logits_output + (accepted - 1) * cache_bytes;
                ensure!(
                    cache_only || with_logits,
                    "copy reduction is not explained by accepted output objects"
                );
            }
        }
        _ => anyhow::bail!("unknown storage mode"),
    }
    Ok(())
}

fn check_tensor_independence(
    context: &mut MLContext<'_>,
    sets: &[BTreeMap<String, MLTensor>; 2],
    latest: usize,
) -> Result<usize> {
    let mut snapshots = Vec::new();
    for (set_index, set) in sets.iter().enumerate() {
        for (name, tensor) in set {
            let mut bytes = vec![0_u8; tensor.rustnn_required_bytes()];
            context.read_tensor(tensor, &mut bytes)?;
            snapshots.push((set_index, name, tensor, bytes));
        }
    }
    // Mutate each current cache output and prove both logical tensor sets retain
    // independent storage. Restore every mutation before the next dispatch.
    let mut mutations = 0;
    for (set_index, name, target, original) in &snapshots {
        if *set_index != latest || *name == "logits" {
            continue;
        }
        let mut sentinel = original.clone();
        for bytes in sentinel.as_chunks_mut::<4>().0 {
            let bits = u32::from_le_bytes(*bytes) ^ 0x8000_0000;
            *bytes = bits.to_le_bytes();
        }
        context.write_tensor(target, &sentinel)?;
        for (other_set, other_name, other, expected) in &snapshots {
            if other_set == set_index && other_name == name {
                continue;
            }
            let mut actual = vec![0_u8; expected.len()];
            context.read_tensor(other, &mut actual)?;
            ensure!(
                actual == *expected,
                "cache output {name} aliases {other_set}:{other_name}"
            );
        }
        context.write_tensor(target, original)?;
        mutations += 1;
    }
    Ok(mutations)
}

impl Runner<'_> {
    fn stream(&mut self, stream: &Stream, verify: bool) -> Result<Value> {
        let raw_path = self.report_path.with_file_name(format!(
            "{}-{}-logits.f32",
            self.report_path
                .file_stem()
                .and_then(|name| name.to_str())
                .unwrap_or("result"),
            stream.name
        ));
        let mut raw = if verify {
            if let Some(parent) = raw_path.parent() {
                fs::create_dir_all(parent)?;
            }
            Some(fs::File::create(&raw_path)?)
        } else {
            None
        };
        let run_start = statistics(&self.context)?;
        let mut execution_io = CoremlTensorStatistics::default();
        let mut decode_io = CoremlTensorStatistics::default();
        let mut prefill_io = CoremlTensorStatistics::default();
        let mut execution_payload = LogicalPayload::default();
        let mut decode_payload = LogicalPayload::default();
        let mut seconds = Vec::new();
        let mut rows = Vec::new();
        let mut digest = seahash::SeaHasher::new();
        let mut total_failures = 0;
        let mut greedy_mismatches = 0;
        let mut logits = vec![0_f32; self.vocab];
        for (index, step) in stream.steps.iter().enumerate() {
            let prefix = step.tokens.len();
            let destination = index % 2;
            let step_start = statistics(&self.context)?;
            let start = Instant::now();
            let tokens = if index == 0 {
                step.tokens.as_slice()
            } else {
                &step.tokens[prefix - 1..]
            };
            let positions: Vec<i32> = if index == 0 {
                (0..prefix as i32).collect()
            } else {
                vec![prefix as i32 - 1]
            };
            let sequence = tokens.len();
            let mut mask = vec![0_f32; sequence * prefix];
            if index == 0 {
                for row in 0..sequence {
                    for column in row + 1..prefix {
                        // Exactly the same finite mask as the frozen references.
                        mask[row * prefix + column] = -10000.0;
                    }
                }
            }
            resize_write(
                &mut self.context,
                self.inputs.get_mut("input_ids").unwrap(),
                &[1, sequence as u64],
                tokens,
            )?;
            resize_write(
                &mut self.context,
                self.inputs.get_mut("position_ids").unwrap(),
                &[1, sequence as u64],
                &positions,
            )?;
            resize_write(
                &mut self.context,
                self.inputs.get_mut("causal_mask").unwrap(),
                &[1, 1, sequence as u64, prefix as u64],
                &mask,
            )?;
            resize_write(
                &mut self.context,
                self.inputs.get_mut("last_token_index").unwrap(),
                &[1],
                &[sequence as i32 - 1],
            )?;
            for (name, tensor) in &mut self.outputs[destination] {
                if name != "logits" {
                    self.context
                        .rustnn_resize_tensor(tensor, &[1, 16, prefix as u64, 4])?;
                }
            }
            let cache_names: Vec<_> = self.outputs[1 - destination]
                .keys()
                .filter(|name| name.as_str() != "logits")
                .map(|name| (name.clone(), name.replacen("present_", "past_", 1)))
                .collect();
            let mut input = MLNamedTensors::new();
            for (name, tensor) in &self.inputs {
                input.insert(name, tensor);
            }
            if index > 0 {
                for (present, past) in &cache_names {
                    input.insert(past, &self.outputs[1 - destination][present]);
                }
            }
            let output: MLNamedTensors = self.outputs[destination]
                .iter()
                .map(|(name, tensor)| (name.as_str(), tensor))
                .collect();
            let graph = if index == 0 {
                &mut self.prefill
            } else {
                &mut self.decode
            };
            self.context
                .dispatch(graph, &input, &output)
                .with_context(|| format!("{} step {index}, prefix {prefix}", stream.name))?;
            self.context
                .read_tensor(&self.outputs[destination]["logits"], &mut logits)?;
            let token = argmax(&logits);
            // Cache ownership advances by alternating these two distinct sets.
            // No cache readback, repacking or writeback is needed for chaining.
            let elapsed = start.elapsed().as_secs_f64();
            let step_io = statistics_delta(step_start, statistics(&self.context)?)?;
            let payload = LogicalPayload::step(sequence, prefix, index > 0, self.vocab);
            check_copy_accounting(&self.mode, step_io, &payload, index > 0)
                .with_context(|| format!("copy accounting {} step {index}", stream.name))?;
            add_statistics(&mut execution_io, step_io)?;
            execution_payload.add(&payload);
            if index > 0 {
                seconds.push(elapsed);
                add_statistics(&mut decode_io, step_io)?;
                decode_payload.add(&payload);
            } else {
                prefill_io = step_io;
            }
            ensure!(logits.iter().all(|x| x.is_finite()), "nonfinite logits");
            digest.write(bytemuck::cast_slice(&logits));
            if verify {
                raw.as_mut()
                    .unwrap()
                    .write_all(bytemuck::cast_slice(&logits))?;
                let reference = floats(&self.root.join(format!(
                    "references/fp32_cache/{}/{index:03}.f32",
                    stream.name
                )))?;
                let metric = compare(&logits, &reference, self.atol, self.rtol)?;
                total_failures += metric["failures"].as_u64().unwrap();
                greedy_mismatches += usize::from(token != argmax(&reference));
                let mut cache_failures = 0;
                for (name, tensor) in &self.outputs[destination] {
                    if name == "logits" {
                        continue;
                    }
                    ensure!(
                        tensor.shape() == [1, 16, prefix as u64, 4],
                        "cache binding shape"
                    );
                    let past = name.replacen("present_", "past_", 1);
                    let path = self.root.join(format!(
                        "cache-references/{}/{index:03}/{past}.f32",
                        stream.name
                    ));
                    if path.exists() {
                        let mut contents = vec![0_f32; 64 * prefix];
                        self.context.read_tensor(tensor, &mut contents)?;
                        let metric = compare(&contents, &floats(&path)?, self.atol, self.rtol)?;
                        cache_failures += metric["failures"].as_u64().unwrap();
                    }
                }
                total_failures += cache_failures;
                rows.push(json!({"step":index,"prefix":prefix,"logits":metric,"cache_failures":cache_failures,"top_token":token}));
            }
        }
        let alias_mutations = if verify {
            check_tensor_independence(
                &mut self.context,
                &self.outputs,
                (stream.steps.len() - 1) % 2,
            )?
        } else {
            0
        };
        let total_io = statistics_delta(run_start, statistics(&self.context)?)?;
        let verification_io = statistics_delta(execution_io, total_io)?;
        let elapsed: f64 = seconds.iter().sum();
        let windows: Vec<_> = seconds
            .chunks(32)
            .enumerate()
            .map(|(window, timings)| {
                let sum: f64 = timings.iter().sum();
                json!({"first_decode_step":window*32+1,"tokens":timings.len(),"seconds":sum,
                "tokens_per_second":timings.len() as f64/sum})
            })
            .collect();
        let mut sorted = seconds.clone();
        sorted.sort_by(f64::total_cmp);
        let percentile = |fraction: f64| {
            sorted
                .get(((sorted.len().saturating_sub(1)) as f64 * fraction).round() as usize)
                .copied()
        };
        Ok(
            json!({"stream":stream.name,"steps":stream.steps.len(),"decode_tokens":seconds.len(),
            "seconds":elapsed,"tokens_per_second":if elapsed>0.0 {seconds.len() as f64/elapsed} else {0.0},
            "p50_seconds":percentile(0.5),"p95_seconds":percentile(0.95),"step_seconds":seconds,
            "windows":windows,
            "prefill_io":statistics_json(prefill_io),"decode_io":statistics_json(decode_io),
            "execution_io":statistics_json(execution_io),"validation_io":statistics_json(verification_io),
            "execution_logical_payload":execution_payload,"decode_logical_payload":decode_payload,
            "alias_independence_mutations":alias_mutations,
            "validation_logits_file":if verify {Some(&raw_path)} else {None},
            "logit_seahash":format!("{:016x}",digest.finish()),"verified":verify,
            "failures":total_failures,"greedy_mismatches":greedy_mismatches,"checks":rows}),
        )
    }
}

pub fn run_configuration(config: &BenchmarkConfig) -> Result<Value> {
    if let Some(minimum) = config.min_measured_seconds {
        ensure!(
            minimum.is_finite() && minimum >= 0.0,
            "invalid minimum measurement duration"
        );
    }
    let device_type = match config.policy.as_str() {
        "cpuOnly" => DeviceType::Cpu,
        "cpuAndGPU" => DeviceType::Gpu,
        "cpuAndNeuralEngine" => DeviceType::Npu,
        _ => anyhow::bail!("unknown policy {}", config.policy),
    };
    if let Some(path) = &config.smollm_model {
        return smollm_probe(config, path, device_type);
    }
    let resources = config.fixture_root.join("Resources");
    let mut fixture: Fixture = serde_json::from_slice(&fs::read(resources.join("streams.json"))?)?;
    let boundary: Fixture = serde_json::from_slice(&fs::read(resources.join("boundary.json"))?)?;
    fixture.streams.extend(boundary.streams);
    let prefill_info = graph(&config.fixture_root.join("graphs/fp32_prefill.json"))?;
    let decode_info = graph(&config.fixture_root.join("graphs/fp32_decode.json"))?;
    let options = MLContextOptions::new(MLPowerPreference::Default, device_type != DeviceType::Cpu)
        .with_rustnn_device_hint(BackendDevice::Coreml { device_type })
        .with_rustnn_options(storage_options(&config.mode)?);
    let mut context = MLContext::create(&options)?;
    let inputs = tensors(&mut context, &prefill_info, &prefill_info.input_operands)?;
    let outputs = [
        tensors(&mut context, &prefill_info, &prefill_info.output_operands)?,
        tensors(&mut context, &decode_info, &decode_info.output_operands)?,
    ];
    let build = Instant::now();
    eprintln!("Building FP32 prefill with real MLContext");
    let prefill = context
        .rustnn_build_graph(prefill_info)
        .context("build prefill")?;
    eprintln!("Building FP32 decode with real MLContext");
    let decode = context
        .rustnn_build_graph(decode_info)
        .context("build decode")?;
    let build_seconds = build.elapsed().as_secs_f64();
    let mut runner = Runner {
        context,
        prefill,
        decode,
        inputs,
        outputs,
        root: &resources,
        vocab: fixture.vocab,
        atol: fixture.atol,
        rtol: fixture.rtol,
        mode: config.mode.clone(),
        report_path: &config.output,
    };
    let mut runs = Vec::new();
    for name in &config.streams {
        let mut stream = fixture
            .streams
            .iter()
            .find(|s| &s.name == name)
            .cloned()
            .context("unknown stream")?;
        if let Some(steps) = config.steps {
            stream.steps.truncate(steps);
        }
        ensure!(!stream.steps.is_empty(), "empty stream");
        ensure!(
            config.min_measured_seconds.unwrap_or(0.0) == 0.0 || stream.steps.len() > 1,
            "a timed sustained run needs at least one token after prefill"
        );
        eprintln!("Verifying {name}");
        let verification = runner.stream(&stream, true)?;
        ensure!(
            verification["failures"] == 0,
            "numerical gate failed: {verification}"
        );
        ensure!(
            verification["greedy_mismatches"] == 0,
            "greedy token gate failed: {verification}"
        );
        runs.push(json!({"phase":"verification","result":verification}));
        for warmup in 0..config.warmups {
            runner.stream(&stream, false)?;
            eprintln!("WARMUP {} {name} {}", config.mode, warmup + 1);
        }
        let mut repetition = 0;
        let mut measured_seconds = 0.0;
        while repetition < config.repeats
            || (config.repeats > 0 && measured_seconds < config.min_measured_seconds.unwrap_or(0.0))
        {
            let result = runner.stream(&stream, false)?;
            ensure!(
                result["logit_seahash"] == verification["logit_seahash"],
                "repeated/reset logit mismatch"
            );
            eprintln!(
                "MEASURED {} {name} {} tokens/s={}",
                config.mode,
                repetition + 1,
                result["tokens_per_second"]
            );
            runs.push(json!({"phase":"measured","repetition":repetition,"result":result}));
            measured_seconds += result["seconds"].as_f64().unwrap();
            repetition += 1;
        }
    }
    Ok(
        json!({"status":"complete","config":config,"build_seconds":build_seconds,
        "runtime":"RustNN MLContext dispatch","cache_chaining":"distinct tensor-set ping-pong",
        "timing":"after first delivered token, includes input updates, dispatch, logits read and argmax; excludes oracle and disk IO",
        "atol":fixture.atol,"rtol":fixture.rtol,"runs":runs}),
    )
}

fn smollm_probe(config: &BenchmarkConfig, path: &Path, device_type: DeviceType) -> Result<Value> {
    eprintln!("Importing supplied SmolLM source {}", path.display());
    let graph = rustnn::load_graph_from_path(path).context("SmolLM import")?;
    let properties = ContextProperties {
        tensor_byte_length_limit: usize::try_from(500_000_000_000_u64).unwrap_or(usize::MAX),
        ..ContextProperties::default()
    };
    GraphValidator::new(&graph, properties)
        .validate()
        .context("SmolLM validation")?;
    let options = MLContextOptions::new(MLPowerPreference::Default, device_type != DeviceType::Cpu)
        .with_rustnn_device_hint(BackendDevice::Coreml { device_type })
        .with_rustnn_options(storage_options(&config.mode)?);
    let mut context = MLContext::create(&options)?;
    let mut inputs = BTreeMap::new();
    let mut outputs = BTreeMap::new();
    for (ids, result) in [
        (&graph.input_operands, &mut inputs),
        (&graph.output_operands, &mut outputs),
    ] {
        for &id in ids {
            let operand = &graph.operands[id as usize];
            let name = operand.name.clone().context("unnamed SmolLM binding")?;
            let shape = operand
                .descriptor
                .shape
                .iter()
                .enumerate()
                .map(|(axis, dimension)| match dimension {
                    rustnn::graph::Dimension::Static(n) => Ok(u64::from(*n)),
                    rustnn::graph::Dimension::Dynamic(d) => match d.name.as_str() {
                        "batch_size" | "sequence_length" | "past_sequence_length" => Ok(1),
                        "past_sequence_length + 1" => Ok(2),
                        "" if name.starts_with("present_") && axis == 2 => Ok(2),
                        "" if name == "logits" && axis < 2 => Ok(1),
                        other => anyhow::bail!(
                            "unhandled SmolLM dimension {other:?}, {name} axis {axis}, max {}",
                            d.max_size
                        ),
                    },
                })
                .collect::<Result<Vec<_>>>()?;
            let tensor = context.create_tensor(
                &MLTensorDescriptor::new(dtype(operand.descriptor.data_type)?, shape)
                    .to_readable()
                    .to_writable(),
            )?;
            result.insert(name, tensor);
        }
    }
    for (name, tensor) in &inputs {
        let count = tensor.shape().iter().product::<u64>() as usize;
        match tensor.data_type() {
            MLOperandDataType::Int64 => {
                let value = if name.starts_with("past_") { 0 } else { 1 };
                context.write_tensor(tensor, &vec![value as i64; count])?;
            }
            MLOperandDataType::Int32 => context.write_tensor(tensor, &vec![1_i32; count])?,
            MLOperandDataType::Float32 => context.write_tensor(tensor, &vec![0_f32; count])?,
            other => anyhow::bail!("unexpected SmolLM tensor dtype {other:?}"),
        }
    }
    eprintln!("Building supplied SmolLM with real MLContext");
    let start = Instant::now();
    let mut compiled = context.rustnn_build_graph(graph).context("SmolLM build")?;
    let build_seconds = start.elapsed().as_secs_f64();
    let named_inputs = inputs
        .iter()
        .map(|(name, tensor)| (name.as_str(), tensor))
        .collect();
    let named_outputs = outputs
        .iter()
        .map(|(name, tensor)| (name.as_str(), tensor))
        .collect();
    eprintln!("Dispatching SmolLM with batch=1, query=1, nonempty past=1");
    context
        .dispatch(&mut compiled, &named_inputs, &named_outputs)
        .context("SmolLM nonempty-cache dispatch")?;
    let mut result = Vec::new();
    for (name, tensor) in outputs {
        let count = tensor.shape().iter().product::<u64>() as usize;
        let mut values = vec![0_f32; count];
        context.read_tensor(&tensor, &mut values)?;
        ensure!(
            values.iter().all(|x| x.is_finite()),
            "nonfinite SmolLM {name}"
        );
        result.push(json!({"name":name,"shape":tensor.shape(),"elements":count}));
    }
    Ok(
        json!({"status":"smoke_complete","config":config,"build_seconds":build_seconds,
        "numerical_reference_validated":false,"note":"Readiness probe only: synthetic nonempty cache, not model-quality or throughput qualification", "outputs":result}),
    )
}

#[allow(dead_code)]
fn main() -> Result<()> {
    let path = std::env::args()
        .nth(1)
        .context("usage: coreml_kv_benchmark --self-contained | CONFIG.json")?;
    if path == "--self-contained" {
        println!("{}", serde_json::to_string_pretty(&self_contained::run()?)?);
        return Ok(());
    }
    let config: BenchmarkConfig = serde_json::from_slice(&fs::read(path)?)?;
    let result = run_configuration(&config);
    let report = match &result {
        Ok(report) => report.clone(),
        Err(error) => json!({"status":"error","config":config,"error":format!("{error:#}")}),
    };
    if let Some(parent) = config.output.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&config.output, serde_json::to_vec_pretty(&report)?)?;
    result.map(|_| ())
}

fn storage_options(mode: &str) -> Result<RustNNOptions> {
    let mut options = RustNNOptions::default();
    match mode {
        "baseline" => {
            options.coreml.reuse_tensor_storage = false;
            options.coreml.output_backings = false;
        }
        "persistent" => {
            options.coreml.reuse_tensor_storage = true;
            options.coreml.output_backings = false;
        }
        "backings" => {
            options.coreml.reuse_tensor_storage = true;
            options.coreml.output_backings = true;
        }
        _ => anyhow::bail!("unknown storage mode {mode}"),
    }
    Ok(options)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_ties_choose_first_index() {
        assert_eq!(argmax(&[-1.0, 3.0, 3.0, 2.0]), 1);
    }

    #[test]
    fn numerical_gate_is_not_execution_only() {
        let result = compare(&[1.0, 2.01], &[1.0, 2.0], 0.0005, 0.0001).unwrap();
        assert_eq!(result["failures"], 1);
        assert!(compare(&[f32::NAN], &[1.0], 0.0005, 0.0001).is_err());
        assert!(compare(&[1.0], &[1.0, 2.0], 0.0005, 0.0001).is_err());
    }

    #[test]
    fn storage_modes_only_change_backend_flags() {
        let baseline = storage_options("baseline").unwrap();
        let persistent = storage_options("persistent").unwrap();
        let backings = storage_options("backings").unwrap();
        assert!(!baseline.coreml.reuse_tensor_storage && !baseline.coreml.output_backings);
        assert!(persistent.coreml.reuse_tensor_storage && !persistent.coreml.output_backings);
        assert!(backings.coreml.reuse_tensor_storage && backings.coreml.output_backings);
        assert!(storage_options("unknown").is_err());
    }

    #[test]
    fn copy_accounting_rejects_hidden_cache_round_trips() {
        let payload = LogicalPayload::step(1, 5, true, 50257);
        let mut stats = CoremlTensorStatistics {
            host_read_bytes: 201028,
            host_write_bytes: 32,
            native_input_bindings: 20,
            output_copy_bytes: 221508,
            ..CoremlTensorStatistics::default()
        };
        check_copy_accounting("persistent", stats, &payload, true).unwrap();
        stats.host_read_bytes += 16384;
        assert!(check_copy_accounting("persistent", stats, &payload, true).is_err());
    }

    #[test]
    fn copy_reduction_requires_an_accepted_output_object() {
        let payload = LogicalPayload::step(1, 5, true, 50257);
        let mut stats = CoremlTensorStatistics {
            host_read_bytes: 201028,
            host_write_bytes: 32,
            native_input_bindings: 20,
            output_backings_requested: 17,
            output_copy_bytes: 0,
            ..CoremlTensorStatistics::default()
        };
        assert!(check_copy_accounting("backings", stats, &payload, true).is_err());
        stats.output_backings_accepted = 17;
        check_copy_accounting("backings", stats, &payload, true).unwrap();
        assert!(statistics_delta(stats, CoremlTensorStatistics::default()).is_err());
    }
}
