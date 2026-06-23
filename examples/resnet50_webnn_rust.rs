use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use clap::Parser;
use rustnn::mlcontext::{
    MLContext, MLContextOptions, MLGraphBuilder, MLPowerPreference, MLTensorDescriptor,
};
use rustnn::operator_enums::MLOperandDataType;
use rustnn::{ContextProperties, GraphValidator, load_graph_from_path};

#[derive(Parser, Debug)]
#[command(about = "Pure Rust ResNet-50 inference benchmark from .webnn + weights/manifest")]
struct Args {
    #[arg(long, default_value = "resnet50_Opset16.webnn")]
    model: PathBuf,
    #[arg(
        long,
        help = "Optional input: .jpg/.png (preprocessed with ImageNet mean/std) or raw f32 LE (1*3*224*224 floats). Synthetic if omitted."
    )]
    input: Option<PathBuf>,
    #[arg(long, help = "Optional ImageNet labels file (one per line)")]
    labels: Option<PathBuf>,
    #[arg(long, default_value_t = 5)]
    top_k: usize,
    #[arg(long, help = "Run a warmup + timed loop and report latency statistics")]
    bench: bool,
    #[arg(
        long,
        default_value_t = 5,
        requires = "bench",
        help = "Untimed warmup iterations (--bench)"
    )]
    warmup: usize,
    #[arg(
        long,
        default_value_t = 50,
        requires = "bench",
        help = "Timed iterations (--bench)"
    )]
    iters: usize,
}

const N: usize = 1;
const C: usize = 3;
const H: usize = 224;
const W: usize = 224;
const NUM_ELEMS: usize = N * C * H * W;

fn load_input(args: &Args) -> Result<Vec<f32>, String> {
    let Some(path) = args.input.as_ref() else {
        return Ok((0..NUM_ELEMS)
            .map(|i| ((i % 256) as f32 / 255.0 - 0.5) / 0.5)
            .collect());
    };
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .map(str::to_ascii_lowercase)
        .unwrap_or_default();
    if matches!(ext.as_str(), "jpg" | "jpeg" | "png") {
        return load_image(path);
    }
    let bytes = fs::read(path).map_err(|e| format!("read input {}: {e}", path.display()))?;
    let expected = NUM_ELEMS * 4;
    if bytes.len() != expected {
        return Err(format!(
            "input file size {} bytes, expected {} ({} * f32)",
            bytes.len(),
            expected,
            NUM_ELEMS
        ));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// Standard ImageNet preprocessing: resize shorter side to 256, center-crop 224x224,
/// scale to [0,1], normalize with ImageNet mean/std, emit NCHW float32.
fn load_image(path: &std::path::Path) -> Result<Vec<f32>, String> {
    const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
    const STD: [f32; 3] = [0.229, 0.224, 0.225];

    let img = image::open(path)
        .map_err(|e| format!("decode image {}: {e}", path.display()))?
        .to_rgb8();
    let (src_w, src_h) = (img.width() as f32, img.height() as f32);
    let scale = 256.0 / src_w.min(src_h);
    let new_w = (src_w * scale).round() as u32;
    let new_h = (src_h * scale).round() as u32;
    let resized =
        image::imageops::resize(&img, new_w, new_h, image::imageops::FilterType::Triangle);
    let crop_x = (new_w - W as u32) / 2;
    let crop_y = (new_h - H as u32) / 2;
    let cropped =
        image::imageops::crop_imm(&resized, crop_x, crop_y, W as u32, H as u32).to_image();

    let plane = H * W;
    let mut out = vec![0.0f32; C * plane];
    for y in 0..H {
        for x in 0..W {
            let p = cropped.get_pixel(x as u32, y as u32);
            let idx = y * W + x;
            for c in 0..C {
                out[c * plane + idx] = (p[c] as f32 / 255.0 - MEAN[c]) / STD[c];
            }
        }
    }
    Ok(out)
}

fn resolve_labels_path(args: &Args) -> Option<PathBuf> {
    if let Some(p) = &args.labels {
        return Some(p.clone());
    }
    let default = PathBuf::from("examples/imagenet_classes.txt");
    if default.exists() {
        Some(default)
    } else {
        None
    }
}

fn load_labels(path: &PathBuf) -> Option<Vec<String>> {
    Some(
        fs::read_to_string(path)
            .ok()?
            .lines()
            .map(str::to_string)
            .collect(),
    )
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut exps: Vec<f32> = logits.iter().map(|v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum > 0.0 {
        for v in &mut exps {
            *v /= sum;
        }
    }
    exps
}

fn top_k(values: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut indexed: Vec<(usize, f32)> = values.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.truncate(k.min(values.len()));
    indexed
}

fn percentile(sorted_us: &[u128], p: f64) -> f64 {
    if sorted_us.is_empty() {
        return 0.0;
    }
    let rank = (p / 100.0) * (sorted_us.len() as f64 - 1.0);
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    if lo == hi {
        sorted_us[lo] as f64
    } else {
        let frac = rank - lo as f64;
        sorted_us[lo] as f64 * (1.0 - frac) + sorted_us[hi] as f64 * frac
    }
}

fn fmt_us(us: f64) -> String {
    if us >= 1000.0 {
        format!("{:>8.3} ms", us / 1000.0)
    } else {
        format!("{:>8.1} us", us)
    }
}

fn report_stats(samples: &[Duration]) {
    let mut us: Vec<u128> = samples.iter().map(|d| d.as_micros()).collect();
    us.sort_unstable();
    let n = us.len() as f64;
    let mean = us.iter().sum::<u128>() as f64 / n;
    let var = us
        .iter()
        .map(|&v| {
            let d = v as f64 - mean;
            d * d
        })
        .sum::<f64>()
        / n;
    let stddev = var.sqrt();
    let min = *us.first().unwrap() as f64;
    let max = *us.last().unwrap() as f64;
    let p50 = percentile(&us, 50.0);
    let p95 = percentile(&us, 95.0);
    let p99 = percentile(&us, 99.0);
    let throughput = 1_000_000.0 / mean;

    println!("\nBenchmark results:");
    println!("  iters:        {}", samples.len());
    println!("  min:        {}", fmt_us(min));
    println!("  mean:       {}  +/- {}", fmt_us(mean), fmt_us(stddev));
    println!("  median:     {}", fmt_us(p50));
    println!("  p95:        {}", fmt_us(p95));
    println!("  p99:        {}", fmt_us(p99));
    println!("  max:        {}", fmt_us(max));
    println!("  throughput: {:>8.2} infer/s", throughput);
}

fn print_predictions(args: &Args, shape: &[usize], logits: &[f32]) {
    let probs = softmax(logits);
    let top = top_k(&probs, args.top_k);
    let labels = resolve_labels_path(args).and_then(|p| load_labels(&p));

    println!("Output shape: {:?}", shape);
    println!("Top-{} predictions:", args.top_k);
    for (idx, prob) in top {
        let label = labels
            .as_ref()
            .and_then(|l| l.get(idx))
            .map(String::as_str)
            .unwrap_or("(no label)");
        println!("  class {:>4}  {:>7.3}%  {}", idx, prob * 100.0, label);
    }
}

fn run() -> Result<(), String> {
    let args = Args::parse();
    if args.iters == 0 {
        return Err("--iters must be >= 1".to_string());
    }

    println!("Loading graph: {}", args.model.display());
    let graph = load_graph_from_path(&args.model).map_err(|e| format!("load graph: {e}"))?;

    let artifacts = GraphValidator::new(&graph, ContextProperties::default())
        .validate()
        .map_err(|e| format!("validate graph: {e}"))?;
    let input_name = artifacts
        .input_names_to_descriptors
        .keys()
        .next()
        .ok_or("graph has no inputs")?
        .clone();
    let output_name = artifacts
        .output_names_to_descriptors
        .keys()
        .next()
        .ok_or("graph has no outputs")?
        .clone();
    println!("Graph input `{input_name}`, output `{output_name}`");

    let input_data = load_input(&args)?;

    let options = MLContextOptions::new(MLPowerPreference::HighPerformance, true);
    println!("Creating MLContext...");
    let mut context =
        MLContext::create(&options).map_err(|e| format!("create MLContext: {e:?}"))?;

    println!("Building graph...");
    let build_start = Instant::now();
    let mut builder =
        MLGraphBuilder::new(&mut context).map_err(|e| format!("create MLGraphBuilder: {e}"))?;
    let mut ml_graph = builder
        .build_graph_info(graph)
        .map_err(|e| format!("build graph: {e}"))?;
    println!("Graph build took {:?}", build_start.elapsed());

    let input_shape = vec![N as u64, C as u64, H as u64, W as u64];
    let mut input_td = MLTensorDescriptor::new(MLOperandDataType::Float32, input_shape);
    input_td.set_writable(true);
    let input_tensor = context
        .create_tensor(&input_td)
        .map_err(|e| format!("create input tensor: {e}"))?;

    let output_shape_u64 = vec![N as u64, 1000u64];
    let mut output_td =
        MLTensorDescriptor::new(MLOperandDataType::Float32, output_shape_u64.clone());
    output_td.set_readable(true);
    let output_tensor = context
        .create_tensor(&output_td)
        .map_err(|e| format!("create output tensor: {e}"))?;

    let output_shape: Vec<usize> = output_shape_u64.iter().map(|&d| d as usize).collect();
    let num_classes: usize = output_shape.iter().product();

    let dispatch_once = |context: &mut MLContext,
                         ml_graph: &mut rustnn::mlcontext::MLGraph,
                         input_data: &[f32]|
     -> Result<Vec<f32>, String> {
        context
            .write_tensor(&input_tensor, input_data)
            .map_err(|e| format!("write_tensor: {e}"))?;
        let inputs = HashMap::from([(input_name.as_str(), &input_tensor)]);
        let outputs = HashMap::from([(output_name.as_str(), &output_tensor)]);
        context
            .dispatch(ml_graph, &inputs, &outputs)
            .map_err(|e| format!("dispatch: {e:?}"))?;
        let mut logits = vec![0.0f32; num_classes];
        context
            .read_tensor(&output_tensor, &mut logits)
            .map_err(|e| format!("read_tensor: {e}"))?;
        Ok(logits)
    };

    if args.bench {
        println!("Warmup ({} iters)...", args.warmup);
        for _ in 0..args.warmup {
            dispatch_once(&mut context, &mut ml_graph, &input_data)?;
        }
        println!("Timed ({} iters)...", args.iters);
        let mut samples = Vec::with_capacity(args.iters);
        let mut last_logits = Vec::new();
        for _ in 0..args.iters {
            context
                .write_tensor(&input_tensor, &input_data)
                .map_err(|e| format!("write_tensor: {e}"))?;
            let inputs = HashMap::from([(input_name.as_str(), &input_tensor)]);
            let outputs = HashMap::from([(output_name.as_str(), &output_tensor)]);
            let start = Instant::now();
            context
                .dispatch(&mut ml_graph, &inputs, &outputs)
                .map_err(|e| format!("dispatch: {e:?}"))?;
            samples.push(start.elapsed());
            let mut logits = vec![0.0f32; num_classes];
            context
                .read_tensor(&output_tensor, &mut logits)
                .map_err(|e| format!("read_tensor: {e}"))?;
            last_logits = logits;
        }
        report_stats(&samples);
        println!();
        print_predictions(&args, &output_shape, &last_logits);
    } else {
        let start = Instant::now();
        let logits = dispatch_once(&mut context, &mut ml_graph, &input_data)?;
        println!("Inference time: {:?}", start.elapsed());
        print_predictions(&args, &output_shape, &logits);
    }

    Ok(())
}

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}
