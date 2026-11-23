//! URL-backed constants: the pull model for weights.
//!
//! Prototype of the pull model from the discussion in
//! <https://github.com/webmachinelearning/webnn/issues/901>. Instead of pushing a host buffer
//! into [`crate::mlgraphbuilder::MLGraphBuilder`], the application names where the bytes of a
//! constant live ([`UrlRef`]: a URL plus a byte range) and the implementation decides when and
//! how to load them. The sketch in the issue discussion is
//!
//! ```text
//! builder.constant(new Request("model.bin", {headers: {"Range": "bytes=o-e"}}))
//! ```
//!
//! and the rustnn analogue, which takes the operand descriptor as well, is
//! [`MLGraphBuilder::rustnn_constant_from_url`](crate::mlgraphbuilder::MLGraphBuilder::rustnn_constant_from_url).
//!
//! # Semantics of the prototype
//!
//! 1. Creating a URL constant performs no I/O. It checks that the byte range has the length
//!    the descriptor requires and that `offset + len` does not overflow.
//! 2. `build()` is the pull point: it returns once every constant a backend consumes is bound.
//!    A failed fetch is a build error naming the operand and the URL. `finish_graph_info`
//!    pulls too: it fetches every URL constant into host memory.
//! 3. The bytes are used as they are: little-endian, packed like the bytes given to
//!    `constant_from_bytes`, no transcoding.
//! 4. A backend without late binding gets the default behaviour: every URL constant is fetched
//!    into host memory before the backend sees the graph, also constants without a consumer,
//!    and the byte bound of [`UrlFetchOptions`] has no effect. TensorRT-RTX overrides this: it
//!    builds (or loads from its cache) an engine with placeholder weights while worker threads
//!    fetch, and refits the engine from the fetched buffers.
//!
//! # What is stubbed
//!
//! There is no on-disk weight cache, no request coalescing, no retry, no "hollow" graph that
//! can be dispatched before all weights arrived, no C ABI entry and no loader integration.
//! Targets without threads (`wasm32`) are not supported: the fetch needs worker threads, so
//! building a graph with URL constants fails there with `UnresolvedConstant`.

use std::collections::{HashMap, VecDeque};
use std::io::{self, Read, Seek, SeekFrom};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender, channel};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::error::GraphBuilderError;
use crate::graph::{ConstantData, GraphInfo};

/// Environment variable: number of worker threads that fetch URL constants (default 4).
pub const RUSTNN_WEIGHTS_FETCH_THREADS_ENV_VAR: &str = "RUSTNN_WEIGHTS_FETCH_THREADS";
/// Environment variable: upper bound in bytes for fetched weights that are held in host memory
/// and not yet handed to the device (default: unlimited, or twice the bind batch when that is
/// set). A single constant larger than the bound is still fetched, alone. Only TensorRT-RTX
/// applies the bound.
pub const RUSTNN_WEIGHTS_MAX_INFLIGHT_BYTES_ENV_VAR: &str = "RUSTNN_WEIGHTS_MAX_INFLIGHT_BYTES";
/// Environment variable: hand fetched weights to the device whenever this many bytes are held
/// in host memory, then free them (default: half of the in-flight bound when that is set,
/// otherwise once, after everything was fetched). Only TensorRT-RTX uses it.
pub const RUSTNN_WEIGHTS_BIND_BATCH_BYTES_ENV_VAR: &str = "RUSTNN_WEIGHTS_BIND_BATCH_BYTES";

/// How the URL constants of one graph are fetched and bound.
///
/// [`UrlFetchOptions::from_env`] is the default of every builder.
/// [`MLGraphBuilder::rustnn_set_url_fetch_options`](crate::mlgraphbuilder::MLGraphBuilder::rustnn_set_url_fetch_options)
/// replaces all of it: start from `from_env()` to keep the environment values of the fields
/// that are not set, or from `default()` to ignore the environment.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct UrlFetchOptions {
    /// Worker threads that fetch ranges (default 4, clamped to 1..=64 and to the number of
    /// constants to fetch).
    pub threads: Option<usize>,
    /// Upper bound in bytes for fetched data held in host memory before it reaches the
    /// device. `None` is unbounded: every deferred constant may be in host memory at once.
    /// Only TensorRT-RTX applies the bound; the fallback of the other backends and
    /// `finish_graph_info` fetch everything into host memory.
    ///
    /// The bound is best effort in two cases: a single constant larger than the bound is
    /// fetched alone, and a backend that can only bind weights in groups (TensorRT-RTX
    /// refits combined weights together) keeps the fetched members of an incomplete group
    /// while the fetch goes on.
    pub max_inflight_bytes: Option<usize>,
    /// Bind fetched data to the device whenever this many bytes are held, then free it.
    /// `None` is half of `max_inflight_bytes` when that is set, otherwise one bind at the end.
    /// A batch is at most half of `max_inflight_bytes`; a batch without `max_inflight_bytes`
    /// implies a bound of twice the batch.
    pub bind_batch_bytes: Option<usize>,
}

impl UrlFetchOptions {
    /// Options from `RUSTNN_WEIGHTS_FETCH_THREADS`, `RUSTNN_WEIGHTS_MAX_INFLIGHT_BYTES` and
    /// `RUSTNN_WEIGHTS_BIND_BATCH_BYTES`; unset, unparsable and zero values mean `None`.
    pub fn from_env() -> Self {
        let read = |name| env_usize(name).filter(|&n| n > 0);
        Self {
            threads: read(RUSTNN_WEIGHTS_FETCH_THREADS_ENV_VAR),
            max_inflight_bytes: read(RUSTNN_WEIGHTS_MAX_INFLIGHT_BYTES_ENV_VAR),
            bind_batch_bytes: read(RUSTNN_WEIGHTS_BIND_BATCH_BYTES_ENV_VAR),
        }
    }

    /// Set the number of fetch threads.
    pub fn with_threads(mut self, threads: usize) -> Self {
        self.threads = Some(threads);
        self
    }

    /// Bound the fetched bytes held in host memory.
    pub fn with_max_inflight_bytes(mut self, bytes: usize) -> Self {
        self.max_inflight_bytes = Some(bytes);
        self
    }

    /// Bind to the device in batches of this many bytes.
    pub fn with_bind_batch_bytes(mut self, bytes: usize) -> Self {
        self.bind_batch_bytes = Some(bytes);
        self
    }

    /// Batch size after applying the default rule; `usize::MAX` means one bind at the end.
    ///
    /// With a byte bound the batch is at most half of it: the fetch workers stop at the bound,
    /// so a larger batch would never fill.
    #[cfg_attr(not(feature = "trtx-runtime"), allow(dead_code))]
    pub(crate) fn effective_bind_batch_bytes(&self) -> usize {
        let half_bound = self.max_inflight_bytes.map(|b| (b / 2).max(1));
        match (self.bind_batch_bytes.filter(|&n| n > 0), half_bound) {
            (Some(batch), Some(half)) => batch.min(half),
            (Some(batch), None) => batch,
            (None, Some(half)) => half,
            (None, None) => usize::MAX,
        }
    }

    /// Byte bound for the fetch workers; `usize::MAX` is unbounded.
    ///
    /// A batch size without a bound implies a bound of twice the batch: binding in batches
    /// frees nothing if the workers may fetch the whole model ahead of the consumer.
    #[cfg_attr(not(feature = "trtx-runtime"), allow(dead_code))]
    pub(crate) fn effective_max_inflight_bytes(&self) -> usize {
        self.max_inflight_bytes
            .filter(|&n| n > 0)
            .or(self
                .bind_batch_bytes
                .filter(|&n| n > 0)
                .map(|batch| batch.saturating_mul(2)))
            .unwrap_or(usize::MAX)
    }
}

/// Location of the bytes of one constant: a URL and a byte range inside the resource.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct UrlRef {
    /// `http://`, `https://` or `file://` URL, or a plain file system path.
    pub url: String,
    /// Offset of the first byte of the constant inside the resource.
    pub offset: u64,
    /// Number of bytes; must equal the byte length required by the operand descriptor.
    pub len: u64,
    /// Optional entity tag (a strong validator, quotes included). HTTP requests send it as
    /// `If-Match`, so a changed resource fails the fetch instead of yielding bytes from a
    /// different file version. Ignored for file sources.
    #[serde(default)]
    pub etag: Option<String>,
    /// Optional label for diagnostics.
    #[serde(default)]
    pub label: Option<String>,
}

impl UrlRef {
    /// Byte range `offset..offset + len` of the resource at `url`.
    pub fn new(url: impl Into<String>, offset: u64, len: u64) -> Self {
        Self {
            url: url.into(),
            offset,
            len,
            etag: None,
            label: None,
        }
    }

    /// Set the entity tag used to validate the resource version.
    pub fn with_etag(mut self, etag: impl Into<String>) -> Self {
        self.etag = Some(etag.into());
        self
    }

    /// Set a label for diagnostics.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }
}

/// Loads the bytes of a [`UrlRef`] into a destination chosen by the implementation.
///
/// The destination is provided by the caller so that a backend can hand out staging memory it
/// owns (pinned, or device-addressable on unified-memory systems) instead of receiving a buffer
/// allocated by the fetch layer.
pub trait WeightsFetcher: Send + Sync + std::fmt::Debug {
    /// Fill `dst` (`dst.len() == source.len`) with the bytes of `source`.
    fn fetch_into(&self, source: &UrlRef, dst: &mut [u8]) -> io::Result<()>;
}

/// Fetcher for plain paths and `file://` URLs, and for `http://` / `https://` URLs when the
/// crate is built with the `url-weights-http` feature.
#[derive(Debug, Default, Clone, Copy)]
pub struct DefaultWeightsFetcher;

impl WeightsFetcher for DefaultWeightsFetcher {
    fn fetch_into(&self, source: &UrlRef, dst: &mut [u8]) -> io::Result<()> {
        if dst.len() as u64 != source.len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "destination has {} bytes, the range has {}",
                    dst.len(),
                    source.len
                ),
            ));
        }
        let scheme = url_scheme(&source.url);
        match scheme.as_deref() {
            Some("http") | Some("https") => fetch_http(source, dst),
            Some("file") | None => fetch_file(&file_url_to_path(&source.url)?, source, dst),
            Some(other) => Err(io::Error::new(
                io::ErrorKind::Unsupported,
                format!("unsupported URL scheme {other:?}"),
            )),
        }
    }
}

/// Lower-cased scheme of `url`, or `None` for a plain path (including `C:\...`).
fn url_scheme(url: &str) -> Option<String> {
    let (scheme, _) = url.split_once("://")?;
    let valid = scheme.len() > 1
        && scheme
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '+' | '-' | '.'));
    valid.then(|| scheme.to_ascii_lowercase())
}

/// Convert a `file://` URL or a plain path into a path.
fn file_url_to_path(url: &str) -> io::Result<std::path::PathBuf> {
    // The scheme is case-insensitive; `url_scheme` is the single place that recognizes it.
    let rest = match url_scheme(url).as_deref() {
        Some("file") => &url["file://".len()..],
        _ => return Ok(std::path::PathBuf::from(url)),
    };
    // file://host/path is not supported, only file:///path and file://localhost/path.
    let rest = rest.strip_prefix("localhost").unwrap_or(rest);
    if !rest.starts_with('/') {
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            format!("file URL with a host is not supported: {url}"),
        ));
    }
    let decoded = percent_decode(rest)?;
    // file:///C:/dir/file -> C:/dir/file
    let bytes = decoded.as_bytes();
    let has_drive = bytes.len() >= 3 && bytes[1].is_ascii_alphabetic() && bytes[2] == b':';
    Ok(std::path::PathBuf::from(if has_drive {
        &decoded[1..]
    } else {
        &decoded[..]
    }))
}

fn percent_decode(input: &str) -> io::Result<String> {
    let bytes = input.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' {
            let hex = bytes
                .get(i + 1..i + 3)
                .and_then(|h| std::str::from_utf8(h).ok())
                .and_then(|h| u8::from_str_radix(h, 16).ok())
                .ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidInput, "invalid percent escape")
                })?;
            out.push(hex);
            i += 3;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8(out).map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))
}

fn fetch_file(path: &std::path::Path, source: &UrlRef, dst: &mut [u8]) -> io::Result<()> {
    let mut file = std::fs::File::open(path)?;
    file.seek(SeekFrom::Start(source.offset))?;
    file.read_exact(dst)
}

#[cfg(feature = "url-weights-http")]
fn fetch_http(source: &UrlRef, dst: &mut [u8]) -> io::Result<()> {
    use std::sync::LazyLock;
    // One agent for the process: connections are pooled and reused across ranges.
    // The read timeout bounds every single read, not the transfer: a stalled peer fails the
    // fetch (and with it `build`) instead of blocking it forever.
    static AGENT: LazyLock<ureq::Agent> = LazyLock::new(|| {
        ureq::AgentBuilder::new()
            .timeout_connect(Duration::from_secs(30))
            .timeout_read(Duration::from_secs(60))
            .timeout_write(Duration::from_secs(60))
            .build()
    });

    if dst.is_empty() {
        return Ok(());
    }
    let last = source
        .offset
        .checked_add(source.len - 1)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "byte range overflows"))?;
    let mut request = AGENT
        .get(&source.url)
        .set("Range", &format!("bytes={}-{last}", source.offset));
    if let Some(etag) = &source.etag {
        request = request.set("If-Match", etag);
    }
    let response = request
        .call()
        .map_err(|e| io::Error::other(format!("GET {}: {e}", source.url)))?;
    let status = response.status();
    let content_length = response
        .header("Content-Length")
        .and_then(|v| v.parse::<u64>().ok());
    // 206 is the expected answer. 200 means the server ignored the range; that is only
    // acceptable when the range is the whole resource.
    let whole_resource = status == 200 && source.offset == 0 && content_length == Some(source.len);
    if status != 206 && !whole_resource {
        return Err(io::Error::other(format!(
            "GET {} with Range bytes={}-{last}: expected 206 Partial Content, got {status}",
            source.url, source.offset
        )));
    }
    if let Some(n) = content_length
        && n != source.len
    {
        return Err(io::Error::other(format!(
            "GET {}: server returned {n} bytes for a {} byte range",
            source.url, source.len
        )));
    }
    response.into_reader().read_exact(dst)
}

#[cfg(not(feature = "url-weights-http"))]
fn fetch_http(source: &UrlRef, _dst: &mut [u8]) -> io::Result<()> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        format!(
            "{}: http(s) URLs need the `url-weights-http` feature",
            source.url
        ),
    ))
}

/// The URL constants of one graph and the fetcher that resolves them.
///
/// Handed to [`crate::mlcontext::MLBackendBuilder::build_with_url_constants`] next to a
/// `GraphInfo` whose `constant_operand_ids_to_handles` has no entry for these operands.
#[derive(Debug)]
pub(crate) struct UrlConstants {
    pub(crate) sources: HashMap<u32, UrlRef>,
    pub(crate) fetcher: Arc<dyn WeightsFetcher>,
    pub(crate) options: UrlFetchOptions,
}

impl UrlConstants {
    pub(crate) fn unresolved(
        &self,
        operand: u32,
        reason: impl std::fmt::Display,
    ) -> GraphBuilderError {
        GraphBuilderError::UnresolvedConstant {
            operand,
            url: self
                .sources
                .get(&operand)
                .map(|s| s.url.clone())
                .unwrap_or_default(),
            reason: reason.to_string(),
        }
    }

    /// Fetch one constant on the calling thread.
    #[cfg_attr(not(feature = "trtx-runtime"), allow(dead_code))]
    pub(crate) fn fetch_one(&self, operand: u32) -> Result<Vec<u8>, GraphBuilderError> {
        let source = self
            .sources
            .get(&operand)
            .ok_or_else(|| self.unresolved(operand, "operand is not a URL constant"))?;
        let mut data = vec![0u8; source.len as usize];
        self.fetcher
            .fetch_into(source, &mut data)
            .map_err(|e| self.unresolved(operand, e))?;
        Ok(data)
    }

    /// Fallback for backends without late binding: fetch every URL constant into host memory
    /// and insert it as an ordinary constant.
    pub(crate) fn materialize_into(self, graph: &mut GraphInfo) -> Result<(), GraphBuilderError> {
        // Every buffer ends up in the graph, so a byte budget would only stall the fetch.
        let jobs: Vec<u32> = self.sources.keys().copied().collect();
        let mut stream = spawn_fetch(&self, jobs, usize::MAX);
        while let Some((operand, result)) = stream.recv() {
            let data = result.map_err(|e| self.unresolved(operand, e))?;
            stream.release(data.len());
            let label = self.sources.get(&operand).and_then(|s| s.label.clone());
            graph
                .constant_operand_ids_to_handles
                .insert(operand, ConstantData { data, label });
        }
        Ok(())
    }
}

/// Byte budget shared by the fetch workers and the consumer of a [`FetchStream`].
#[derive(Debug)]
struct Budget {
    max_bytes: usize,
    in_flight: Mutex<usize>,
    changed: Condvar,
    /// Number of workers blocked until the consumer releases budget.
    waiting: AtomicUsize,
}

/// One finished fetch: the operand id and its bytes or the failure.
pub(crate) type FetchResult = (u32, io::Result<Vec<u8>>);

/// Results of background fetches, in completion order.
///
/// Dropping the stream cancels the outstanding work and joins the workers.
pub(crate) struct FetchStream {
    receiver: Receiver<FetchResult>,
    budget: Arc<Budget>,
    cancelled: Arc<AtomicBool>,
    workers: Vec<std::thread::JoinHandle<()>>,
    remaining: usize,
}

impl FetchStream {
    /// Number of results not yet received.
    #[allow(dead_code)]
    pub(crate) fn remaining(&self) -> usize {
        self.remaining
    }

    /// Next finished fetch, blocking. `None` once every job was delivered.
    pub(crate) fn recv(&mut self) -> Option<FetchResult> {
        if self.remaining == 0 {
            return None;
        }
        match self.receiver.recv() {
            Ok(item) => {
                self.remaining -= 1;
                Some(item)
            }
            Err(_) => {
                // Every worker exited without delivering all jobs: a worker panicked.
                let remaining = std::mem::take(&mut self.remaining);
                Some((
                    u32::MAX,
                    Err(io::Error::other(format!(
                        "weight fetch workers stopped with {remaining} constants outstanding"
                    ))),
                ))
            }
        }
    }

    /// Like [`Self::recv`], but gives up after `timeout`. `Ok(None)` means every job was
    /// delivered, `Err(())` means the timeout expired.
    #[cfg_attr(not(feature = "trtx-runtime"), allow(dead_code))]
    #[allow(clippy::result_unit_err)]
    pub(crate) fn recv_timeout(&mut self, timeout: Duration) -> Result<Option<FetchResult>, ()> {
        if self.remaining == 0 {
            return Ok(None);
        }
        match self.receiver.recv_timeout(timeout) {
            Ok(item) => {
                self.remaining -= 1;
                Ok(Some(item))
            }
            Err(RecvTimeoutError::Timeout) => Err(()),
            Err(RecvTimeoutError::Disconnected) => {
                let remaining = std::mem::take(&mut self.remaining);
                Ok(Some((
                    u32::MAX,
                    Err(io::Error::other(format!(
                        "weight fetch workers stopped with {remaining} constants outstanding"
                    ))),
                )))
            }
        }
    }

    /// True when at least one worker is blocked until [`Self::release`] frees budget.
    #[cfg_attr(not(feature = "trtx-runtime"), allow(dead_code))]
    pub(crate) fn workers_waiting(&self) -> bool {
        self.budget.waiting.load(Ordering::SeqCst) > 0
    }

    /// Tell the workers that `bytes` of fetched data left host memory (or are no longer
    /// counted against the budget), so that further fetches may start.
    pub(crate) fn release(&self, bytes: usize) {
        let mut in_flight = self.budget.in_flight.lock().unwrap();
        *in_flight = in_flight.saturating_sub(bytes);
        self.budget.changed.notify_all();
    }
}

impl Drop for FetchStream {
    fn drop(&mut self) {
        self.cancelled.store(true, Ordering::SeqCst);
        // Wake workers that wait for budget; the lock orders the store against their check.
        drop(self.budget.in_flight.lock().unwrap());
        self.budget.changed.notify_all();
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

fn env_usize(name: &str) -> Option<usize> {
    std::env::var(name).ok()?.trim().parse().ok()
}

/// Fetch `operands` on worker threads, ordered by `(url, offset)` so that ranges of one
/// resource are requested front to back.
///
/// At most `max_inflight_bytes` of fetched data are outstanding at a time: a worker waits
/// before allocating the destination of its next job until the consumer called
/// [`FetchStream::release`]. A job larger than the whole budget runs when nothing else is
/// outstanding.
pub(crate) fn spawn_fetch(
    urls: &UrlConstants,
    operands: Vec<u32>,
    max_inflight_bytes: usize,
) -> FetchStream {
    let mut jobs: Vec<(u32, UrlRef)> = operands
        .into_iter()
        .filter_map(|id| urls.sources.get(&id).map(|s| (id, s.clone())))
        .collect();
    jobs.sort_by(|a, b| (&a.1.url, a.1.offset, a.0).cmp(&(&b.1.url, b.1.offset, b.0)));
    let remaining = jobs.len();

    let queue = Arc::new(Mutex::new(VecDeque::from(jobs)));
    let budget = Arc::new(Budget {
        max_bytes: max_inflight_bytes.max(1),
        in_flight: Mutex::new(0),
        changed: Condvar::new(),
        waiting: AtomicUsize::new(0),
    });
    let cancelled = Arc::new(AtomicBool::new(false));
    let (sender, receiver) = channel();

    let threads = urls
        .options
        .threads
        .unwrap_or(4)
        .clamp(1, 64)
        .min(remaining.max(1));
    let workers = (0..if remaining == 0 { 0 } else { threads })
        .map(|index| {
            let queue = Arc::clone(&queue);
            let budget = Arc::clone(&budget);
            let cancelled = Arc::clone(&cancelled);
            let fetcher = Arc::clone(&urls.fetcher);
            let sender: Sender<FetchResult> = sender.clone();
            std::thread::Builder::new()
                .name(format!("rustnn-weights-{index}"))
                .spawn(move || fetch_worker(&queue, &budget, &cancelled, &*fetcher, &sender))
        })
        .collect::<Vec<_>>();
    let mut handles = Vec::new();
    let mut spawn_error = None;
    for worker in workers {
        match worker {
            Ok(handle) => handles.push(handle),
            Err(error) => spawn_error = Some(error),
        }
    }
    // Without any worker (for example on wasm32, which has no threads) nobody would deliver
    // the jobs: report that as a failed fetch instead of waiting forever.
    if handles.is_empty()
        && let Some(error) = spawn_error
    {
        let _ = sender.send((
            u32::MAX,
            Err(io::Error::other(format!(
                "cannot start a weight fetch thread: {error}"
            ))),
        ));
    }

    FetchStream {
        receiver,
        budget,
        cancelled,
        workers: handles,
        remaining,
    }
}

fn fetch_worker(
    queue: &Mutex<VecDeque<(u32, UrlRef)>>,
    budget: &Budget,
    cancelled: &AtomicBool,
    fetcher: &dyn WeightsFetcher,
    sender: &Sender<FetchResult>,
) {
    loop {
        // Reserve budget before taking a job, so that jobs keep their (url, offset) order.
        let (operand, source) = {
            let mut in_flight = budget.in_flight.lock().unwrap();
            loop {
                if cancelled.load(Ordering::SeqCst) {
                    return;
                }
                let mut queue = queue.lock().unwrap();
                let Some((_, next)) = queue.front() else {
                    return;
                };
                let len = next.len as usize;
                if *in_flight == 0 || in_flight.saturating_add(len) <= budget.max_bytes {
                    *in_flight = in_flight.saturating_add(len);
                    break queue.pop_front().expect("front() was Some");
                }
                drop(queue);
                budget.waiting.fetch_add(1, Ordering::SeqCst);
                in_flight = budget
                    .changed
                    .wait_timeout(in_flight, Duration::from_millis(100))
                    .unwrap()
                    .0;
                budget.waiting.fetch_sub(1, Ordering::SeqCst);
            }
        };
        // A panic in an embedder's fetcher must not lose the job: the consumer would wait
        // for it forever while the reservation keeps the other workers blocked.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut data = vec![0u8; source.len as usize];
            fetcher.fetch_into(&source, &mut data).map(|()| data)
        }))
        .unwrap_or_else(|_| Err(io::Error::other("the weights fetcher panicked")));
        if sender.send((operand, result)).is_err() {
            return;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[derive(Debug, Default)]
    struct PatternFetcher {
        calls: AtomicUsize,
        peak_in_flight: AtomicUsize,
        current: AtomicUsize,
    }

    impl WeightsFetcher for PatternFetcher {
        fn fetch_into(&self, source: &UrlRef, dst: &mut [u8]) -> io::Result<()> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let now = self.current.fetch_add(dst.len(), Ordering::SeqCst) + dst.len();
            self.peak_in_flight.fetch_max(now, Ordering::SeqCst);
            if source.url == "fail" {
                return Err(io::Error::other("boom"));
            }
            if source.url == "panic" {
                panic!("fetcher bug");
            }
            for (i, byte) in dst.iter_mut().enumerate() {
                *byte = (source.offset as usize + i) as u8;
            }
            Ok(())
        }
    }

    fn constants(fetcher: Arc<dyn WeightsFetcher>, refs: &[(u32, UrlRef)]) -> UrlConstants {
        UrlConstants {
            sources: refs.iter().cloned().collect(),
            fetcher,
            options: UrlFetchOptions::default(),
        }
    }

    #[test]
    fn scheme_detection_keeps_windows_paths() {
        assert_eq!(url_scheme("https://a/b").as_deref(), Some("https"));
        assert_eq!(url_scheme("FILE:///c:/x").as_deref(), Some("file"));
        assert_eq!(url_scheme("C:\\weights\\model.bin"), None);
        assert_eq!(url_scheme("relative/model.bin"), None);
    }

    #[test]
    fn file_urls_map_to_paths() {
        assert_eq!(
            file_url_to_path("file:///C:/a%20b/model.bin").unwrap(),
            std::path::PathBuf::from("C:/a b/model.bin")
        );
        assert_eq!(
            file_url_to_path("file:///tmp/model.bin").unwrap(),
            std::path::PathBuf::from("/tmp/model.bin")
        );
        assert_eq!(
            file_url_to_path("file://localhost/tmp/model.bin").unwrap(),
            std::path::PathBuf::from("/tmp/model.bin")
        );
        assert!(file_url_to_path("file://server/share/model.bin").is_err());
        // The scheme is case-insensitive.
        assert_eq!(
            file_url_to_path("File:///tmp/model.bin").unwrap(),
            std::path::PathBuf::from("/tmp/model.bin")
        );
        // Not a URL: a plain path is returned unchanged.
        assert_eq!(
            file_url_to_path("C:\\weights\\model.bin").unwrap(),
            std::path::PathBuf::from("C:\\weights\\model.bin")
        );
    }

    #[test]
    fn default_fetcher_reads_a_file_range() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        let bytes: Vec<u8> = (0..=255u8).collect();
        file.write_all(&bytes).unwrap();
        file.flush().unwrap();
        let path = file.path().to_str().unwrap().to_string();

        let mut dst = [0u8; 16];
        DefaultWeightsFetcher
            .fetch_into(&UrlRef::new(path.clone(), 32, 16), &mut dst)
            .unwrap();
        assert_eq!(&dst[..], &bytes[32..48]);

        // A range past the end of the file is an error, not a short read.
        let mut dst = [0u8; 16];
        let error = DefaultWeightsFetcher
            .fetch_into(&UrlRef::new(path, 250, 16), &mut dst)
            .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::UnexpectedEof);
    }

    #[test]
    fn unsupported_scheme_is_rejected() {
        let mut dst = [0u8; 4];
        let error = DefaultWeightsFetcher
            .fetch_into(&UrlRef::new("ftp://host/model.bin", 0, 4), &mut dst)
            .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::Unsupported);
    }

    #[test]
    fn stream_delivers_every_job_once() {
        let fetcher = Arc::new(PatternFetcher::default());
        let refs: Vec<(u32, UrlRef)> = (0..17u32)
            .map(|i| (i, UrlRef::new("blob", u64::from(i) * 8, 8)))
            .collect();
        let urls = constants(fetcher.clone(), &refs);
        let mut stream = spawn_fetch(&urls, refs.iter().map(|r| r.0).collect(), usize::MAX);
        let mut seen = HashMap::new();
        while let Some((id, result)) = stream.recv() {
            let data = result.unwrap();
            stream.release(data.len());
            assert!(seen.insert(id, data).is_none());
        }
        assert_eq!(seen.len(), 17);
        assert_eq!(fetcher.calls.load(Ordering::SeqCst), 17);
        assert_eq!(seen[&3], (24u8..32).collect::<Vec<u8>>());
    }

    #[test]
    fn budget_bounds_outstanding_bytes() {
        let fetcher = Arc::new(PatternFetcher::default());
        let refs: Vec<(u32, UrlRef)> = (0..32u32)
            .map(|i| (i, UrlRef::new("blob", u64::from(i) * 100, 100)))
            .collect();
        let urls = constants(fetcher.clone(), &refs);
        let mut stream = spawn_fetch(&urls, refs.iter().map(|r| r.0).collect(), 250);
        let mut count = 0;
        while let Some((_, result)) = stream.recv() {
            let data = result.unwrap();
            // The consumer holds one buffer at a time and releases it before the next recv.
            fetcher.current.fetch_sub(data.len(), Ordering::SeqCst);
            stream.release(data.len());
            count += 1;
        }
        assert_eq!(count, 32);
        // Two 100 byte jobs fit in 250 bytes; a third must wait.
        assert!(fetcher.peak_in_flight.load(Ordering::SeqCst) <= 200);
    }

    #[test]
    fn oversized_job_runs_alone() {
        let fetcher = Arc::new(PatternFetcher::default());
        let refs = vec![
            (0u32, UrlRef::new("blob", 0, 1000)),
            (1u32, UrlRef::new("blob", 1000, 10)),
        ];
        let urls = constants(fetcher, &refs);
        let mut stream = spawn_fetch(&urls, vec![0, 1], 100);
        let mut total = 0;
        while let Some((_, result)) = stream.recv() {
            let data = result.unwrap();
            total += data.len();
            stream.release(data.len());
        }
        assert_eq!(total, 1010);
    }

    #[test]
    fn panicking_fetcher_is_reported_under_a_budget() {
        // The first job panics while the others wait for the budget it reserved. The
        // consumer must get an error for it instead of waiting forever.
        let fetcher = Arc::new(PatternFetcher::default());
        let mut refs = vec![(0u32, UrlRef::new("panic", 0, 60))];
        refs.extend((1..4u32).map(|i| (i, UrlRef::new("zz", u64::from(i) * 60, 60))));
        let urls = constants(fetcher, &refs);
        let mut stream = spawn_fetch(&urls, vec![0, 1, 2, 3], 100);
        let (id, result) = stream.recv().unwrap();
        assert_eq!(id, 0);
        assert!(result.unwrap_err().to_string().contains("panicked"));
    }

    #[test]
    fn bind_batch_is_clamped_to_half_of_the_bound() {
        let options = UrlFetchOptions::default();
        assert_eq!(options.effective_bind_batch_bytes(), usize::MAX);
        assert_eq!(options.effective_max_inflight_bytes(), usize::MAX);
        let options = UrlFetchOptions::default().with_max_inflight_bytes(100);
        assert_eq!(options.effective_bind_batch_bytes(), 50);
        let options = options.with_bind_batch_bytes(400);
        assert_eq!(options.effective_bind_batch_bytes(), 50);
        // A batch alone implies a bound, otherwise nothing would be freed early.
        let options = UrlFetchOptions::default().with_bind_batch_bytes(64);
        assert_eq!(options.effective_bind_batch_bytes(), 64);
        assert_eq!(options.effective_max_inflight_bytes(), 128);
    }

    #[test]
    fn dropping_the_stream_stops_waiting_workers() {
        let fetcher = Arc::new(PatternFetcher::default());
        let refs: Vec<(u32, UrlRef)> = (0..8u32)
            .map(|i| (i, UrlRef::new("blob", u64::from(i) * 100, 100)))
            .collect();
        let urls = constants(fetcher.clone(), &refs);
        let mut stream = spawn_fetch(&urls, refs.iter().map(|r| r.0).collect(), 100);
        // Take one result and never release it: the other workers wait for budget.
        let _ = stream.recv().unwrap();
        drop(stream); // must not hang
        assert!(fetcher.calls.load(Ordering::SeqCst) < 8);
    }

    #[test]
    fn materialize_inserts_constants_and_reports_failures() {
        let fetcher = Arc::new(PatternFetcher::default());
        let urls = constants(
            fetcher,
            &[
                (0, UrlRef::new("blob", 4, 4).with_label("w")),
                (1, UrlRef::new("blob", 0, 2)),
            ],
        );
        let mut graph = GraphInfo::default();
        urls.materialize_into(&mut graph).unwrap();
        assert_eq!(graph.constant_operand_ids_to_handles[&0].data, [4, 5, 6, 7]);
        assert_eq!(
            graph.constant_operand_ids_to_handles[&0].label.as_deref(),
            Some("w")
        );
        assert_eq!(graph.constant_operand_ids_to_handles[&1].data, [0, 1]);

        let urls = constants(
            Arc::new(PatternFetcher::default()),
            &[(7, UrlRef::new("fail", 0, 2))],
        );
        let error = urls
            .materialize_into(&mut GraphInfo::default())
            .unwrap_err();
        match error {
            GraphBuilderError::UnresolvedConstant {
                operand,
                url,
                reason,
            } => {
                assert_eq!(operand, 7);
                assert_eq!(url, "fail");
                assert!(reason.contains("boom"));
            }
            other => panic!("unexpected error {other:?}"),
        }
    }
}
