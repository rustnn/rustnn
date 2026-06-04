#![cfg(any(feature = "burn-runtime-cpu", feature = "burn-runtime-webgpu"))]

#[cfg(feature = "burn-runtime-cpu")]
mod cpu;
mod device_ops;
mod execute;
mod f16;
mod host_array;
mod host_ops_extra;
mod interpreter;
mod tensor_env;
#[cfg(feature = "burn-runtime-webgpu")]
mod webgpu;
#[cfg(feature = "burn-runtime-webgpu")]
mod webgpu_device;

pub use interpreter::{BurnInput, BurnOutput, BurnOutputWithData, execute_plan};

#[cfg(feature = "burn-runtime-cpu")]
pub use cpu::{run_burn_cpu_with_inputs, run_burn_cpu_zeroed};

#[cfg(feature = "burn-runtime-webgpu")]
pub use webgpu::{
    exceeds_webgpu_tensor_binding_limit, max_storage_buffer_binding_bytes,
    run_burn_webgpu_with_inputs, run_burn_webgpu_zeroed,
};
