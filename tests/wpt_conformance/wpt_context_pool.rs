//! Per-thread [`MLContext`] cache for WPT trials.

use std::cell::RefCell;
use std::collections::HashMap;

use rustnn::mlcontext::MLContext;

use super::wpt_backend::WptBackend;
use super::wpt_config;

thread_local! {
    static CONTEXTS: RefCell<HashMap<WptBackend, MLContext<'static>>> =
        RefCell::new(HashMap::new());
}

fn create_context(backend: &WptBackend) -> Result<MLContext<'static>, String> {
    MLContext::create(backend.context_options()).map_err(|e| e.to_string())
}

/// Run `f` with an [`MLContext`] for `backend`, reusing a per-thread instance when enabled.
pub fn with_context<R>(
    backend: &WptBackend,
    f: impl FnOnce(&mut MLContext<'_>) -> Result<R, String>,
) -> Result<R, String> {
    if wpt_config::REUSE_ML_CONTEXT {
        CONTEXTS.with(|cell| {
            let mut map = cell.borrow_mut();
            if !map.contains_key(backend) {
                map.insert(backend.clone(), create_context(backend)?);
            }
            f(map.get_mut(backend).unwrap())
        })
    } else {
        let mut context = create_context(backend)?;
        f(&mut context)
    }
}
