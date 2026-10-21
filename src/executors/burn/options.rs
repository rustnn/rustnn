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

//! Parse legacy string options once, at the boundary of typed kernels.
use crate::error::GraphError;
use std::{fmt::Display, str::FromStr};

pub(super) fn parse_option<T: FromStr>(value: &str, field: &str) -> Result<T, GraphError>
where
    T::Err: Display,
{
    value
        .to_ascii_lowercase()
        .parse()
        .map_err(|error| GraphError::BurnRuntimeFailed {
            reason: format!("invalid {field} '{value}': {error}"),
        })
}

/// Empty strings in the shared option structs encode the documented IDL default.
pub(super) fn parse_option_or_default<T: FromStr + Default>(
    value: &str,
    field: &str,
) -> Result<T, GraphError>
where
    T::Err: Display,
{
    if value.is_empty() {
        Ok(T::default())
    } else {
        parse_option(value, field)
    }
}

// JSON encodes non-finite MLNumber values as strings (or null for NaN).
pub(super) fn float_number(v: &serde_json::Value) -> Option<f32> {
    if v.is_null() {
        return Some(f32::NAN);
    }
    if let Some(s) = v.as_str() {
        return match s {
            "Infinity" => Some(f32::INFINITY),
            "-Infinity" => Some(f32::NEG_INFINITY),
            "NaN" => Some(f32::NAN),
            _ => s.parse::<f64>().ok().map(|f| f as f32),
        };
    }
    v.as_f64()
        .map(|f| f as f32)
        .or_else(|| v.as_i64().map(|i| i as f32))
}
