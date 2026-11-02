// SPDX-FileCopyrightText: 2026 NVIDIA
//
// SPDX-License-Identifier: Apache-2.0

pub mod cel;
#[cfg(feature = "z3")]
pub mod z3;

pub mod context;
pub mod error;
pub mod expression;
pub mod program;
pub mod shape_ops;
mod shared;

pub use context::Context;
pub use error::{Error, Result};
pub use expression::{Expression, Shape};
pub use program::Program;

#[cfg(feature = "smallvec")]
type Vec<T> = smallvec::SmallVec<[T; 8]>;
#[cfg(not(feature = "smallvec"))]
type Vec<T> = std::vec::Vec<T>;
