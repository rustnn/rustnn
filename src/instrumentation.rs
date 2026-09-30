//! Optional span instrumentation (Cargo feature `tracing`).
//!
//! This module owns every span the crate emits from its backend-agnostic layer: names, levels,
//! targets and field keys are defined here and nowhere else, so a call site is a single
//! `#[cfg(feature = "tracing")]` line that imports nothing from `tracing`. Module and call
//! sites are compiled only with the feature, so a default build carries neither the dependency
//! nor any instrumentation code. Each constructor below documents its own level, target and
//! fields.
//!
//! `INFO` spans are emitted once per context, graph or file, `DEBUG` spans once per call. Field
//! values are borrowed wrappers over slices, enums and integers, so nothing is formatted unless a
//! subscriber is listening. Fields that only exist after the span was created start as
//! [`tracing::field::Empty`] and are filled in by a `record_*` function of this module.
//!
//! # Call sites
//!
//! Bind the guard — `let _ = …entered();` drops it on the spot and records nothing:
//!
//! ```ignore
//! #[cfg(feature = "tracing")]
//! let _span = crate::instrumentation::dispatch_span(&self.device, inputs, outputs).entered();
//! ```
//!
//! A span with late fields is entered with [`tracing::Span::enter`], which borrows it, so the
//! `record_*` call can still follow the operation it describes; `Span::entered` would consume
//! the span and leave nothing to record into.
//!
//! A call site is either cfg-gated (`#[cfg(feature = "tracing")] let _span = …entered();`) or,
//! where the whole operation is one expression, a `with_*_span` helper beside it that holds the
//! cfg lines — `backends::caching` has the two of those. Every cfg is on the feature, never on
//! its absence. Refer to this module by its full path; no call site imports `tracing`.
//!
//! # Why explicit spans rather than `#[instrument]`
//!
//! The attribute names a span after its function and targets it at the module path, so a rename
//! or a file move silently changes what subscribers filter on, and it declares only the fields
//! known on entry — most of what these spans report (`outcome`, `bytes`, `tensor_id`, the
//! finalized graph sizes) is known only after the work. It would also pull `tracing-attributes`
//! and a `syn` built with `full` into the build. Where a function *is* the phase, takes
//! `Debug`-cheap arguments and has no late fields, the attribute fits — the converter and
//! executor internals, not this layer.

use std::fmt;
use std::path::Path;

use tracing::{Span, field};

use crate::backend_selection::BackendDevice;
use crate::backends::caching::{CacheError, CacheResult};
use crate::graph::GraphInfo;
use crate::mlcontext::{MLNamedOperands, MLNamedTensors, MLTensor, MLTensorDescriptor};
use crate::mlcontextoptions::MLContextOptions;

/// Span for the backend selection in `MLContext::create`. `backend` and `device_type` stay
/// empty when selection fails; [`record_selected_backend`] fills them in otherwise.
pub(crate) fn select_backend_span(options: &MLContextOptions) -> Span {
    tracing::info_span!(
        target: "rustnn::backend_selection",
        "select_backend",
        accelerated = options.accelerated(),
        power_preference = ?options.power_preference(),
        backend_hint = ?options.backend_hint,
        device_hint = ?options.device_hint,
        backend = field::Empty,
        device_type = field::Empty,
    )
}

/// Record the device `select_backend` chose on the span from [`select_backend_span`].
pub(crate) fn record_selected_backend(span: &Span, device: &BackendDevice) {
    span.record("backend", field::debug(device.backend()));
    span.record("device_type", field::debug(device.device_type()));
}

/// Span for one `MLGraphBuilder::build`; [`record_graph_info`] fills in what the finalized
/// graph contains.
pub(crate) fn build_span(outputs: &MLNamedOperands<'_>) -> Span {
    tracing::info_span!(
        target: "rustnn::mlgraphbuilder",
        "build",
        output_names = ?Names(outputs),
        operands = field::Empty,
        operations = field::Empty,
    )
}

/// Record the size of the graph `MLGraphBuilder::build` finalized.
pub(crate) fn record_graph_info(span: &Span, graph: &GraphInfo) {
    span.record("operands", graph.operands.len());
    span.record("operations", graph.operations.len());
}

/// Span for one `MLGraphBuilder::rustnn_save_webnn`; the recorder already holds the finalized
/// graph, so its size is known up front.
pub(crate) fn save_graph_span(
    path: &Path,
    outputs: &MLNamedOperands<'_>,
    graph: &GraphInfo,
) -> Span {
    tracing::info_span!(
        target: "rustnn::mlgraphbuilder",
        "save_graph",
        path = %path.display(),
        output_names = ?Names(outputs),
        operands = graph.operands.len(),
        operations = graph.operations.len(),
    )
}

/// Span for one `load_graph_from_path`: the file, the format it is parsed as, and the size of
/// the graph that came out of it ([`record_graph_info`]).
pub(crate) fn load_graph_span(path: &Path) -> Span {
    tracing::info_span!(
        target: "rustnn::loader",
        "load_graph",
        path = %path.display(),
        format = %graph_format(path),
        operands = field::Empty,
        operations = field::Empty,
    )
}

/// Span for one `GraphValidator::validate` of an already loaded graph.
pub(crate) fn validate_span(graph: &GraphInfo) -> Span {
    tracing::info_span!(
        target: "rustnn::validator",
        "validate",
        operands = graph.operands.len(),
        operations = graph.operations.len(),
        dynamic = graph.has_dynamic_dimensions(),
    )
}

/// Span for one `RuntimeShapeState::validate_named_shapes`, the binding check `dispatch` runs
/// before the backend sees the tensors.
pub(crate) fn validate_shapes_span(
    kind: crate::runtime_checks::TensorKind,
    tensors: usize,
) -> Span {
    tracing::debug_span!(
        target: "rustnn::runtime_checks",
        "validate_shapes",
        kind = ?kind,
        tensors = tensors,
    )
}

/// Span for one `validate_shape_data_length`, the element-count check the executors run on a
/// bound buffer.
pub(crate) fn validate_data_length_span(name: &str, shape: &[usize], elements: usize) -> Span {
    tracing::debug_span!(
        target: "rustnn::runtime_checks",
        "validate_data_length",
        name = %name,
        shape = ?shape,
        elements = elements,
    )
}

/// The format `load_graph_from_path` parses a path as, from its extension.
fn graph_format(path: &Path) -> &'static str {
    match path.extension().and_then(|ext| ext.to_str()) {
        Some("webnn") => "webnn",
        Some("json") => "json",
        _ => "unknown",
    }
}

/// Span for one `MLContext::create_tensor`; [`record_tensor_id`] fills in the id the backend
/// assigned.
pub(crate) fn create_tensor_span(descriptor: &MLTensorDescriptor) -> Span {
    tracing::debug_span!(
        target: "rustnn::mlcontext",
        "create_tensor",
        data_type = ?descriptor.data_type(),
        shape = ?descriptor.shape(),
        tensor_id = field::Empty,
    )
}

/// Record the id the backend assigned to a tensor from [`create_tensor_span`].
pub(crate) fn record_tensor_id(span: &Span, tensor: &MLTensor) {
    span.record("tensor_id", tensor.id);
}

/// Span for one `MLContext::rustnn_resize_tensor`, carrying the shape it moves away from.
pub(crate) fn resize_tensor_span(tensor: &MLTensor, new_shape: &[u64]) -> Span {
    tracing::debug_span!(
        target: "rustnn::mlcontext",
        "resize_tensor",
        tensor_id = tensor.id,
        from = ?tensor.shape(),
        to = ?new_shape,
    )
}

/// Span for one `MLContext::rustnn_set_tensor_capacity`.
pub(crate) fn set_tensor_capacity_span(tensor: &MLTensor, max_shape: &[u64]) -> Span {
    tracing::debug_span!(
        target: "rustnn::mlcontext",
        "set_tensor_capacity",
        tensor_id = tensor.id,
        capacity = ?max_shape,
    )
}

/// Span for one `MLContext::dispatch`, carrying the device and the bound tensors.
pub(crate) fn dispatch_span(
    device: &BackendDevice,
    inputs: &MLNamedTensors<'_>,
    outputs: &MLNamedTensors<'_>,
) -> Span {
    tracing::debug_span!(
        target: "rustnn::mlcontext",
        "dispatch",
        backend = ?device.backend(),
        device_type = ?device.device_type(),
        inputs = ?TensorBindings(inputs),
        outputs = ?TensorBindings(outputs),
        input_count = inputs.len(),
        output_count = outputs.len(),
    )
}

/// Span for one `MLContext::read_tensor`.
pub(crate) fn read_tensor_span(tensor: &MLTensor, bytes: usize) -> Span {
    tracing::debug_span!(
        target: "rustnn::mlcontext",
        "read_tensor",
        tensor_id = tensor.id,
        data_type = ?tensor.data_type(),
        shape = ?tensor.shape(),
        bytes = bytes,
    )
}

/// Span for one `MLContext::write_tensor`.
pub(crate) fn write_tensor_span(tensor: &MLTensor, bytes: usize) -> Span {
    tracing::debug_span!(
        target: "rustnn::mlcontext",
        "write_tensor",
        tensor_id = tensor.id,
        data_type = ?tensor.data_type(),
        shape = ?tensor.shape(),
        bytes = bytes,
    )
}

/// Span for one cache lookup; [`record_cache_result`] fills in `outcome` and `bytes`.
pub(crate) fn cache_get_span(root_path: &Path, key: &str, compressed: bool) -> Span {
    tracing::debug_span!(
        target: "rustnn::backends::caching",
        "get",
        category = %cache_category(root_path),
        key = %key,
        compressed = compressed,
        outcome = field::Empty,
        bytes = field::Empty,
    )
}

/// Span for one cache entry write.
pub(crate) fn cache_set_span(root_path: &Path, key: &str, bytes: usize, compressed: bool) -> Span {
    tracing::debug_span!(
        target: "rustnn::backends::caching",
        "set",
        category = %cache_category(root_path),
        key = %key,
        bytes = bytes,
        compressed = compressed,
    )
}

/// Record what a `PersistentCache::get` found: a hit, a missing entry (an `Err` with
/// `NotFound`, the case the trait's `Result` cannot name) or an I/O error.
pub(crate) fn record_cache_result(span: &Span, result: &CacheResult<Vec<u8>>) {
    match result {
        Ok(data) => {
            span.record("outcome", "hit");
            span.record("bytes", data.len());
        }
        Err(CacheError::FailedToReadCacheFile { source, .. })
            if source.kind() == std::io::ErrorKind::NotFound =>
        {
            span.record("outcome", "miss");
        }
        Err(_) => {
            span.record("outcome", "error");
        }
    }
}

/// The cache category, which is the last component of `<cache dir>/rustnn/<category>`.
fn cache_category(root_path: &Path) -> impl fmt::Display + '_ {
    root_path
        .file_name()
        .map(|name| Path::new(name).display())
        .unwrap_or_else(|| Path::new("unknown").display())
}

/// `{lhs: Float32[2, 2], rhs: Float32[2, 2]}`: the tensors bound to a dispatch.
struct TensorBindings<'a, 'names>(&'a MLNamedTensors<'names>);

impl<'a, 'names> fmt::Debug for TensorBindings<'a, 'names> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("{")?;
        for (index, (name, tensor)) in self.0.iter().enumerate() {
            if index > 0 {
                f.write_str(", ")?;
            }
            // `Float32[2, 2]`: written straight into the formatter, so nothing allocates.
            write!(f, "{name}: {:?}{:?}", tensor.data_type(), tensor.shape())?;
        }
        f.write_str("}")
    }
}

/// `["sum"]`: the output names of a build.
struct Names<'a, 'names>(&'a MLNamedOperands<'names>);

impl fmt::Debug for Names<'_, '_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.0.keys()).finish()
    }
}
