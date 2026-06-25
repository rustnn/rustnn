//! Per-thread [`MLContext`] cache for WPT trials.

#[cfg(any(
    feature = "onnx-runtime",
    feature = "trtx-runtime",
    feature = "trtx-runtime-mock"
))]
use std::cell::RefCell;
#[cfg(any(
    feature = "onnx-runtime",
    feature = "trtx-runtime",
    feature = "trtx-runtime-mock"
))]
use std::collections::HashMap;

#[cfg(any(
    feature = "onnx-runtime",
    feature = "trtx-runtime",
    feature = "trtx-runtime-mock"
))]
use rustnn::mlcontext::MLContext;

#[cfg(any(
    feature = "onnx-runtime",
    feature = "trtx-runtime",
    feature = "trtx-runtime-mock"
))]
use super::wpt_backend::WptBackend;
#[cfg(any(
    feature = "onnx-runtime",
    feature = "trtx-runtime",
    feature = "trtx-runtime-mock"
))]
use super::wpt_config;

#[cfg(any(
    feature = "onnx-runtime",
    feature = "trtx-runtime",
    feature = "trtx-runtime-mock"
))]
thread_local! {
    static CONTEXTS: RefCell<HashMap<WptBackend, MLContext<'static>>> =
        RefCell::new(HashMap::new());
}

#[cfg(any(
    feature = "onnx-runtime",
    feature = "trtx-runtime",
    feature = "trtx-runtime-mock"
))]
fn create_context(backend: WptBackend) -> Result<MLContext<'static>, String> {
    MLContext::create(&backend.context_options()).map_err(|e| e.to_string())
}

/// Run `f` with an [`MLContext`] for `backend`, reusing a per-thread instance when enabled.
#[cfg(any(
    feature = "onnx-runtime",
    feature = "trtx-runtime",
    feature = "trtx-runtime-mock"
))]
pub fn with_context<R>(
    backend: WptBackend,
    f: impl FnOnce(&mut MLContext<'_>) -> Result<R, String>,
) -> Result<R, String> {
    if wpt_config::REUSE_ML_CONTEXT {
        CONTEXTS.with(|cell| {
            let mut map = cell.borrow_mut();
            if !map.contains_key(&backend) {
                map.insert(backend, create_context(backend)?);
            }
            f(map.get_mut(&backend).unwrap())
        })
    } else {
        let mut context = create_context(backend)?;
        f(&mut context)
    }
}
