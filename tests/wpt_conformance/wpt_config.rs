//! WPT harness compile-time options.

/// Reuse one [`MLContext`](rustnn::mlcontext::MLContext) per [`WptBackend`](super::wpt_backend::WptBackend) per test thread.
///
/// Default is `false`: benchmarks showed no ONNX CPU speedup, `OrtContext` retains tensors across
/// trials when reused, and backends are not validated for concurrent use across libtest threads.
/// When `true`, trials on the same thread share a context via `wpt_context_pool` (thread-local).
pub const REUSE_ML_CONTEXT: bool = true;
