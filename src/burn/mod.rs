//! Burn backend intermediate representation shared by converter and executor.

#![cfg(feature = "burn-plan")]

pub mod plan;

pub use plan::{BURN_PLAN_VERSION, BurnGraphPlan, ConstantSlot, IOBinding};
