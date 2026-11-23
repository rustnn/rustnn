/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//! Push model against pull model for weights on TensorRT-RTX
//! (<https://github.com/webmachinelearning/webnn/issues/901>).
//!
//! One process measures one sample and prints one JSON line: peak host memory and the time to
//! the first inference for a stack of `matmul` layers whose weights live in one blob file.
//!
//! ```text
//! cargo run --release --example url_weights_bench --features trtx-runtime,url-weights-http -- \
//!     --mode url --source http --size-mib 512 --cache off --throttle-mib-s 200
//! ```
//!
//! Modes:
//! - `pointer-whole`: the application loads the whole resource, copies each tensor out of it
//!   into `constant_from_bytes`, then builds. This is what a `fetch()` + `ArrayBuffer` page or
//!   the `.webnn` loader does today.
//! - `pointer-range`: the application loads each tensor range into its own buffer and moves
//!   it into `constant_from_bytes`, then builds. The ranges are read one after another on
//!   one thread.
//! - `url`: `rustnn_constant_from_url`; the backend pulls. `--budget-mib` bounds the fetched
//!   bytes held in host memory (batched refit).

#[path = "../tests/support/range_server.rs"]
mod range_server;

use std::alloc::{GlobalAlloc, Layout, System};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use rustnn::mlcontext::{
    Backend, MLContext, MLContextOptions, MLGraphBuilder, MLNamedOperands, MLNamedTensors,
    MLOperandDescriptor, MLPowerPreference, MLTensorDescriptor, RustNNOptions,
};
use rustnn::operator_enums::MLOperandDataType;
use rustnn::weights::{DefaultWeightsFetcher, UrlFetchOptions, UrlRef, WeightsFetcher};

/// Counts live and peak bytes of the Rust heap: the buffers rustnn and this program own.
/// TensorRT and CUDA allocate outside of it; the OS counters below include them.
struct CountingAllocator;
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            let live = LIVE.fetch_add(layout.size(), Ordering::Relaxed) + layout.size();
            PEAK.fetch_max(live, Ordering::Relaxed);
        }
        ptr
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() {
            let live = LIVE.fetch_add(layout.size(), Ordering::Relaxed) + layout.size();
            PEAK.fetch_max(live, Ordering::Relaxed);
        }
        ptr
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new_ptr = unsafe { System.realloc(ptr, layout, new_size) };
        if !new_ptr.is_null() {
            if new_size >= layout.size() {
                let grow = new_size - layout.size();
                let live = LIVE.fetch_add(grow, Ordering::Relaxed) + grow;
                PEAK.fetch_max(live, Ordering::Relaxed);
            } else {
                LIVE.fetch_sub(layout.size() - new_size, Ordering::Relaxed);
            }
        }
        new_ptr
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

/// (peak working set, peak commit charge) of this process in bytes.
#[cfg(windows)]
fn os_peak_memory() -> (u64, u64) {
    #[repr(C)]
    #[derive(Default)]
    struct ProcessMemoryCounters {
        cb: u32,
        page_fault_count: u32,
        peak_working_set_size: usize,
        working_set_size: usize,
        quota_peak_paged_pool_usage: usize,
        quota_paged_pool_usage: usize,
        quota_peak_non_paged_pool_usage: usize,
        quota_non_paged_pool_usage: usize,
        pagefile_usage: usize,
        peak_pagefile_usage: usize,
    }
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetCurrentProcess() -> *mut std::ffi::c_void;
        fn K32GetProcessMemoryInfo(
            process: *mut std::ffi::c_void,
            counters: *mut ProcessMemoryCounters,
            cb: u32,
        ) -> i32;
    }
    let mut counters = ProcessMemoryCounters {
        cb: std::mem::size_of::<ProcessMemoryCounters>() as u32,
        ..Default::default()
    };
    let ok = unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, counters.cb) };
    if ok == 0 {
        return (0, 0);
    }
    (
        counters.peak_working_set_size as u64,
        counters.peak_pagefile_usage as u64,
    )
}

/// (peak resident set, peak virtual size) from `/proc/self/status`, or zeros.
#[cfg(not(windows))]
fn os_peak_memory() -> (u64, u64) {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    let field = |name: &str| {
        status
            .lines()
            .find_map(|line| line.strip_prefix(name))
            .and_then(|rest| rest.split_whitespace().next()?.parse::<u64>().ok())
            .map_or(0, |kib| kib * 1024)
    };
    (field("VmHWM:"), field("VmPeak:"))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    PointerWhole,
    PointerRange,
    Url,
}

struct Args {
    mode: Mode,
    http: bool,
    size_mib: usize,
    layers: usize,
    cache: bool,
    runtime_cache: bool,
    budget_mib: Option<usize>,
    batch_mib: Option<usize>,
    threads: Option<usize>,
    throttle_mib_s: u64,
    blob: Option<PathBuf>,
    serve: Option<PathBuf>,
    weightless_build: bool,
}

fn parse_args() -> Args {
    let mut args = Args {
        mode: Mode::Url,
        http: false,
        size_mib: 256,
        layers: 4,
        cache: false,
        runtime_cache: false,
        budget_mib: None,
        batch_mib: None,
        threads: None,
        throttle_mib_s: 0,
        blob: None,
        serve: None,
        weightless_build: false,
    };
    let mut iter = std::env::args().skip(1);
    while let Some(flag) = iter.next() {
        let mut value = || {
            iter.next()
                .unwrap_or_else(|| panic!("{flag} needs a value"))
        };
        match flag.as_str() {
            "--mode" => {
                args.mode = match value().as_str() {
                    "pointer-whole" => Mode::PointerWhole,
                    "pointer-range" => Mode::PointerRange,
                    "url" => Mode::Url,
                    other => panic!("unknown mode {other}"),
                }
            }
            "--source" => args.http = value() == "http",
            "--size-mib" => args.size_mib = value().parse().unwrap(),
            "--layers" => args.layers = value().parse().unwrap(),
            "--cache" => args.cache = value() == "on",
            // The TensorRT JIT cache is read into one Rust allocation of tens of MiB; it is off
            // by default here so that the Rust heap peak shows weight buffers only.
            "--runtime-cache" => args.runtime_cache = value() == "on",
            "--budget-mib" => args.budget_mib = Some(value().parse().unwrap()),
            "--batch-mib" => args.batch_mib = Some(value().parse().unwrap()),
            "--threads" => args.threads = Some(value().parse().unwrap()),
            "--throttle-mib-s" => args.throttle_mib_s = value().parse().unwrap(),
            "--blob" => args.blob = Some(PathBuf::from(value())),
            "--serve" => args.serve = Some(PathBuf::from(value())),
            // Pointer modes: build the network from placeholders too (experiment switch of
            // the backend), which separates the weightless build from the pull.
            "--weightless-build" => args.weightless_build = value() == "on",
            other => panic!("unknown argument {other}"),
        }
    }
    args
}

/// Square layer width such that `layers` float32 matrices take about `size_mib`.
fn layer_width(size_mib: usize, layers: usize) -> usize {
    let per_layer = size_mib * 1024 * 1024 / layers / 4;
    let mut width = (per_layer as f64).sqrt() as usize;
    width -= width % 64;
    width.max(64)
}

/// Create the weight blob once: a 4096 byte header, then `layers` matrices of small float32
/// values (the product of the layers stays finite).
fn ensure_blob(path: &Path, width: usize, layers: usize) {
    let expected = 4096 + (width * width * 4 * layers) as u64;
    if std::fs::metadata(path).is_ok_and(|m| m.len() == expected) {
        return;
    }
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let mut file = std::io::BufWriter::with_capacity(1 << 22, std::fs::File::create(path).unwrap());
    file.write_all(&[0u8; 4096]).unwrap();
    let scale = 1.0 / (width as f32).sqrt();
    let mut seed = 0x9E37_79B9u32;
    let mut row = vec![0u8; width * 4];
    for _ in 0..layers * width {
        for cell in row.chunks_exact_mut(4) {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            let value = ((seed >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * 2.0 * scale;
            cell.copy_from_slice(&value.to_le_bytes());
        }
        file.write_all(&row).unwrap();
    }
    file.flush().unwrap();
}

/// `--serve`: run the range server in this process until stdin closes.
fn serve(path: &Path, throttle_mib_s: u64) {
    let server =
        range_server::RangeServer::start(path, throttle_mib_s * 1024 * 1024).expect("server");
    println!("{}", server.port);
    std::io::stdout().flush().unwrap();
    let mut line = String::new();
    while std::io::stdin().read_line(&mut line).is_ok_and(|n| n > 0) {
        line.clear();
    }
}

fn main() {
    let args = parse_args();
    if let Some(path) = &args.serve {
        serve(path, args.throttle_mib_s);
        return;
    }
    if args.weightless_build {
        // SAFETY: single-threaded at this point.
        unsafe { std::env::set_var("RUSTNN_TRTX_WEIGHTLESS_BUILD", "1") };
    }
    let width = layer_width(args.size_mib, args.layers);
    let layer_bytes = width * width * 4;
    let blob = args.blob.clone().unwrap_or_else(|| {
        PathBuf::from(format!(
            "target/url_weights_bench/weights_{}x{}.bin",
            args.layers, width
        ))
    });
    ensure_blob(&blob, width, args.layers);
    let blob = std::fs::canonicalize(&blob).unwrap();
    let total_bytes = layer_bytes * args.layers;

    // The server is a separate process so that its memory does not count.
    let mut server = None;
    let url = if args.http {
        let mut child = std::process::Command::new(std::env::current_exe().unwrap())
            .arg("--serve")
            .arg(&blob)
            .arg("--throttle-mib-s")
            .arg(args.throttle_mib_s.to_string())
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .spawn()
            .expect("spawn server");
        let mut port = String::new();
        BufReader::new(child.stdout.take().unwrap())
            .read_line(&mut port)
            .unwrap();
        server = Some(child);
        format!("http://127.0.0.1:{}/weights.bin", port.trim())
    } else {
        blob.to_str()
            .unwrap()
            .trim_start_matches(r"\\?\")
            .to_string()
    };

    // Baseline after process start-up and blob creation.
    PEAK.store(LIVE.load(Ordering::Relaxed), Ordering::Relaxed);
    let start = Instant::now();

    let mut rustnn_options = RustNNOptions::default();
    rustnn_options.trtx.engine_caching = args.cache;
    rustnn_options.trtx.runtime_cache = args.runtime_cache;
    let options = MLContextOptions::new(MLPowerPreference::HighPerformance, true)
        .with_rustnn_backend_hint(Backend::Trtx)
        .with_rustnn_options(rustnn_options);
    let mut context = MLContext::create(&options).expect("TensorRT-RTX context");
    let context_ms = start.elapsed().as_secs_f64() * 1e3;
    let (context_ws, context_commit) = os_peak_memory();

    let mut builder = MLGraphBuilder::new(&mut context).unwrap();
    if args.mode == Mode::Url {
        let mut fetch = UrlFetchOptions::from_env();
        if let Some(mib) = args.budget_mib {
            fetch.max_inflight_bytes = Some(mib << 20);
        }
        if let Some(mib) = args.batch_mib {
            fetch.bind_batch_bytes = Some(mib << 20);
        }
        if let Some(threads) = args.threads {
            fetch.threads = Some(threads);
        }
        builder.rustnn_set_url_fetch_options(fetch);
    }
    let vector = MLOperandDescriptor::new(MLOperandDataType::Float32, vec![1, width as u64]);
    let matrix =
        MLOperandDescriptor::new(MLOperandDataType::Float32, vec![width as u64, width as u64]);
    let range = |layer: usize| {
        UrlRef::new(
            url.clone(),
            (4096 + layer * layer_bytes) as u64,
            layer_bytes as u64,
        )
    };

    // The whole resource, for `pointer-whole` only.
    let whole: Option<Vec<u8>> = (args.mode == Mode::PointerWhole).then(|| {
        let mut data = vec![0u8; 4096 + total_bytes];
        DefaultWeightsFetcher
            .fetch_into(&UrlRef::new(url.clone(), 0, data.len() as u64), &mut data)
            .expect("load whole resource");
        data
    });

    let mut x = builder.input("x", &vector).unwrap();
    for layer in 0..args.layers {
        let weight = match args.mode {
            Mode::Url => builder
                .rustnn_constant_from_url(&matrix, range(layer))
                .unwrap(),
            Mode::PointerRange => {
                let mut data = vec![0u8; layer_bytes];
                DefaultWeightsFetcher
                    .fetch_into(&range(layer), &mut data)
                    .expect("load range");
                builder.constant_from_bytes(&matrix, data).unwrap()
            }
            Mode::PointerWhole => {
                let offset = 4096 + layer * layer_bytes;
                let data = whole.as_ref().unwrap()[offset..offset + layer_bytes].to_vec();
                builder.constant_from_bytes(&matrix, data).unwrap()
            }
        };
        // No activation: every weight element contributes to the output checksum.
        x = builder.matmul(x, weight).unwrap();
    }
    drop(whole);
    let constants_ms = start.elapsed().as_secs_f64() * 1e3;

    let mut graph = builder
        .build(&MLNamedOperands::from([("y", x)]))
        .expect("build");
    drop(builder);
    let build_ms = start.elapsed().as_secs_f64() * 1e3;
    let (build_ws, build_commit) = os_peak_memory();
    let build_heap = PEAK.load(Ordering::Relaxed);

    let mut input_descriptor =
        MLTensorDescriptor::new(MLOperandDataType::Float32, vec![1, width as u64]);
    input_descriptor.set_writable(true);
    let input = context.create_tensor(&input_descriptor).unwrap();
    let mut output_descriptor =
        MLTensorDescriptor::new(MLOperandDataType::Float32, vec![1, width as u64]);
    output_descriptor.set_readable(true);
    let output = context.create_tensor(&output_descriptor).unwrap();
    let values: Vec<f32> = (0..width).map(|i| ((i % 13) as f32 + 1.0) / 13.0).collect();
    context.write_tensor(&input, &values).unwrap();
    context
        .dispatch(
            &mut graph,
            &MLNamedTensors::from([("x", &input)]),
            &MLNamedTensors::from([("y", &output)]),
        )
        .unwrap();
    let mut result = vec![0f32; width];
    context.read_tensor(&output, &mut result).unwrap();
    let first_inference_ms = start.elapsed().as_secs_f64() * 1e3;
    let (end_ws, end_commit) = os_peak_memory();

    // FNV-1a over the output bits: equal across modes when the weights arrived intact.
    let checksum = result.iter().fold(0xcbf2_9ce4_8422_2325u64, |hash, v| {
        (hash ^ u64::from(v.to_bits())).wrapping_mul(0x0000_0100_0000_01b3)
    });
    let nonzero = result.iter().filter(|v| **v != 0.0).count();

    if let Some(mut child) = server {
        drop(child.stdin.take());
        let _ = child.kill();
        let _ = child.wait();
    }

    let mib = |bytes: u64| bytes as f64 / (1024.0 * 1024.0);
    println!(
        "{{\"mode\":\"{:?}\",\"source\":\"{}\",\"cache\":{},\"runtime_cache\":{},\"weightless_build\":{},\"threads\":{},\"weights_mib\":{:.1},\"layers\":{},\"width\":{},\"budget_mib\":{},\"batch_mib\":{},\"throttle_mib_s\":{},\"context_ms\":{:.0},\"constants_ms\":{:.0},\"build_ms\":{:.0},\"first_inference_ms\":{:.0},\"peak_rust_heap_mib\":{:.1},\"context_peak_ws_mib\":{:.1},\"context_peak_commit_mib\":{:.1},\"build_peak_ws_mib\":{:.1},\"build_peak_commit_mib\":{:.1},\"end_peak_ws_mib\":{:.1},\"end_peak_commit_mib\":{:.1},\"output_nonzero\":{},\"checksum\":\"{:016x}\"}}",
        args.mode,
        if args.http { "http" } else { "file" },
        args.cache,
        args.runtime_cache,
        args.weightless_build,
        args.threads.map_or("null".to_string(), |v| v.to_string()),
        mib(total_bytes as u64),
        args.layers,
        width,
        args.budget_mib
            .map_or("null".to_string(), |v| v.to_string()),
        args.batch_mib.map_or("null".to_string(), |v| v.to_string()),
        args.throttle_mib_s,
        context_ms,
        constants_ms,
        build_ms,
        first_inference_ms,
        mib(build_heap as u64),
        mib(context_ws),
        mib(context_commit),
        mib(build_ws),
        mib(build_commit),
        mib(end_ws),
        mib(end_commit),
        nonzero,
        checksum
    );
}
