// SPDX-FileCopyrightText: 2026 Shubham Gupta <shubhamg13.work@gmail.com>
//
// SPDX-License-Identifier: Apache-2

//! CANN shim -- loads the adapter library `libcann_shim.so` at runtime.
//!
//! Fails gracefully on platforms where it's not available.
//!
//! The adapter (src/executors/cann_shim/*.cc) compiles to
//! `libcann_shim.so` on OHOS and wraps the HiAI DDK's C++ API as plain C.
#![allow(unused)]
use std::sync::LazyLock;

use libloading::Library;

use crate::error::{Error, Result};

/// Global shim instance, loaded once per process.
static SHIM: LazyLock<Result<CannShim>> = LazyLock::new(|| CannShim::load());

/// Get the global shim, if the adapter library was successfully loaded.
pub(crate) fn get_shim() -> Option<&'static CannShim> {
    SHIM.as_ref().ok()
}

pub(crate) struct CannShim {
    _lib: Library,
}

impl CannShim {
    fn load() -> Result<Self> {
        let lib_paths = [
            std::env::var("CANN_SHIM_PATH").ok(),
            Some("libcann_shim.so".into()),
        ];

        let mut last_err = String::new();
        for path in lib_paths.iter().flatten() {
            match unsafe { Library::new(path) } {
                Ok(lib) => return Ok(Self { _lib: lib }),
                Err(e) => last_err = format!("{path}: {e}"),
            }
        }

        Err(Error::GraphDispatchError {
            source: last_err.into(),
        })
    }
}
