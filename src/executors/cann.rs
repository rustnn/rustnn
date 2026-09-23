/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Shubham Gupta <shubhamg13.work@gmail.com>
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

//! CANN NPU runtime: the loaded model session and its dispatch I/O views.
//!
//! With `cann-runtime` the session loads a compiled model onto the NPU;
//! with `cann-runtime-mock` it is a no-op that rejects dispatch.

use crate::error::GraphError;
use crate::graph::GraphInfo;
use crate::operator_enums::MLOperandDataType;

/// Host-side input view for a NPU dispatch.
pub struct CannInput<'a> {
    pub data: &'a [u8],
    pub shape: Vec<u32>,
    pub dtype: MLOperandDataType,
}

/// Host-side output view for a NPU dispatch. `actual_len` is filled in by the
/// runtime with the number of bytes the NPU produced.
pub struct CannOutput<'a> {
    pub data: &'a mut [u8],
    pub shape: Vec<u32>,
    pub dtype: MLOperandDataType,
    pub actual_len: usize,
}

/// A model loaded onto the NPU.
#[cfg(feature = "cann-runtime")]
#[derive(Debug)]
pub(crate) struct CannSession {
    // `hiai_rs::Session` is `Send` but not `Sync`, so it lives behind a `Mutex`.
    session: std::sync::Mutex<hiai_rs::Session>,
}

/// Readable message from a caught panic payload.
#[cfg(feature = "cann-runtime")]
fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic".to_string()
    }
}

#[cfg(feature = "cann-runtime")]
impl CannSession {
    /// Compile `graph` (via `encode`) and load it onto the NPU. Pre-creates the
    /// DDK IO tensors when every I/O shape is static, so the per-dispatch ION
    /// allocation is paid once; otherwise dispatch creates them lazily.
    pub(crate) fn compile(
        graph: &GraphInfo,
        encode: impl FnOnce(&GraphInfo) -> Result<Vec<u8>, GraphError>,
    ) -> Result<Self, GraphError> {
        // A panic while encoding must surface as an error, not unwind into the
        // caller. (A native abort inside the DDK cannot be caught here.)
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            let model_bytes = encode(graph)?;
            let mut session =
                hiai_rs::Session::load(&model_bytes).map_err(|e| GraphError::ConversionFailed {
                    format: "cann".into(),
                    reason: format!("session load failed: {e}"),
                })?;

            let static_slots = |ids: &[u32]| -> Option<Vec<(Vec<u32>, i32)>> {
                ids.iter()
                    .map(|&id| {
                        let desc = &graph.operands[id as usize].descriptor;
                        let shape = desc.static_shape()?;
                        let dtype = MLOperandDataType::try_from(desc.data_type)
                            .map(ml_operand_to_cann_dtype)
                            .unwrap_or(0);
                        Some((shape, dtype))
                    })
                    .collect()
            };
            if let (Some(input_slots), Some(output_slots)) = (
                static_slots(&graph.input_operands),
                static_slots(&graph.output_operands),
            ) {
                session
                    .prepare_io(&input_slots, &output_slots)
                    .map_err(|e| GraphError::ConversionFailed {
                        format: "cann".into(),
                        reason: format!("prepare_io failed: {e}"),
                    })?;
            }

            Ok(Self {
                session: std::sync::Mutex::new(session),
            })
        })) {
            Ok(result) => result,
            Err(payload) => Err(GraphError::ConversionFailed {
                format: "cann".into(),
                reason: format!("CANN build panicked: {}", panic_message(&*payload)),
            }),
        }
    }

    /// Run one inference. Inputs are read in place; outputs are written in place
    /// with `actual_len` updated from the runtime.
    pub(crate) fn dispatch(
        &self,
        inputs: &[CannInput<'_>],
        outputs: &mut [CannOutput<'_>],
    ) -> Result<(), GraphError> {
        // Keep panics inside the runtime from unwinding into the caller.
        // (A native abort inside the DDK cannot be caught here.)
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let input_descs: Vec<hiai_rs::InputDesc<'_>> = inputs
                .iter()
                .map(|i| hiai_rs::InputDesc {
                    data: i.data,
                    shape: i.shape.clone(),
                    dtype: ml_operand_to_cann_dtype(i.dtype),
                })
                .collect();
            let mut output_descs: Vec<hiai_rs::OutputDesc<'_>> = outputs
                .iter_mut()
                .map(|o| hiai_rs::OutputDesc {
                    data: &mut *o.data,
                    shape: o.shape.clone(),
                    dtype: ml_operand_to_cann_dtype(o.dtype),
                    actual_len: 0,
                })
                .collect();

            let lock_start = std::time::Instant::now();
            let mut guard = self.session.lock().expect("session mutex poisoned");
            let lock_ms = lock_start.elapsed().as_secs_f64() * 1e3;
            let call_start = std::time::Instant::now();
            let result = guard.dispatch(&input_descs, &mut output_descs);
            let call_ms = call_start.elapsed().as_secs_f64() * 1e3;

            let actual_lens: Vec<usize> = output_descs.iter().map(|d| d.actual_len).collect();
            drop(output_descs);
            for (o, actual) in outputs.iter_mut().zip(actual_lens) {
                o.actual_len = actual;
            }

            if timing_enabled() {
                log::debug!("[cann-timing] lock={lock_ms:.3}ms call={call_ms:.2}ms");
            }

            result.map_err(|e| GraphError::ConversionFailed {
                format: "cann".into(),
                reason: format!("dispatch failed: {e}"),
            })
        })) {
            Ok(result) => result,
            Err(payload) => Err(GraphError::ConversionFailed {
                format: "cann".into(),
                reason: format!("CANN dispatch panicked: {}", panic_message(&*payload)),
            }),
        }
    }
}

/// Enable per-dispatch `[cann-timing]` logs. Off by default; set `CANN_TIMING=1`.
#[cfg(feature = "cann-runtime")]
fn timing_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("CANN_TIMING")
            .map(|v| v == "1")
            .unwrap_or(false)
    })
}

/// Map WebNN operand data type to the CANN adapter enum.
#[cfg(feature = "cann-runtime")]
fn ml_operand_to_cann_dtype(data_type: MLOperandDataType) -> i32 {
    match data_type {
        MLOperandDataType::Float32 => 0, // CANN_DT_FLOAT
        MLOperandDataType::Float16 => 1, // CANN_DT_FLOAT16
        MLOperandDataType::Int32 => 3,   // CANN_DT_INT32
        MLOperandDataType::Int8 => 2,    // CANN_DT_INT8
        MLOperandDataType::Uint8 => 4,   // CANN_DT_UINT8
        MLOperandDataType::Int64 => 9,   // CANN_DT_INT64
        MLOperandDataType::Uint32 => 8,  // CANN_DT_UINT32
        _ => 0,                          // default CANN_DT_FLOAT
    }
}

/// Mock runtime: no compilation, dispatch is unavailable.
#[cfg(not(feature = "cann-runtime"))]
#[derive(Debug)]
pub(crate) struct CannSession;

#[cfg(not(feature = "cann-runtime"))]
impl CannSession {
    pub(crate) fn compile(
        _graph: &GraphInfo,
        _encode: impl FnOnce(&GraphInfo) -> Result<Vec<u8>, GraphError>,
    ) -> Result<Self, GraphError> {
        Ok(Self)
    }

    pub(crate) fn dispatch(
        &self,
        _inputs: &[CannInput<'_>],
        _outputs: &mut [CannOutput<'_>],
    ) -> Result<(), GraphError> {
        Err(GraphError::ConversionFailed {
            format: "cann".into(),
            reason: "CANN shim not available (mock mode)".into(),
        })
    }
}
