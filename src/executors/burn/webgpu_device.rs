#![cfg(feature = "burn-runtime-webgpu")]

//! Burn WGPU device setup with raised [`maxStorageBufferBindingSize`](https://www.w3.org/TR/webgpu/#limits).
//!
//! Burn/cubecl's default path already passes `adapter.limits()` to `request_device`, but the WebGPU
//! spec minimum for storage bindings is 128 MiB while large-tensor cases may need more. We request
//! up to [`DESIRED_STORAGE_BUFFER_BINDING_BYTES`] capped by the adapter so MLContext and WPT paths
//! can use larger single-buffer bindings when the GPU allows.

use std::sync::OnceLock;

use burn_wgpu::graphics::{AutoGraphicsApi, GraphicsApi};
use burn_wgpu::{RuntimeOptions, WgpuDevice, WgpuSetup, init_device};
use futures_lite::future;
use wgpu::Features;

use crate::error::GraphError;

use super::interpreter::WEBGPU_MIN_STORAGE_BUFFER_BINDING_BYTES;

/// Target storage binding size (2 GiB); capped by adapter `max_storage_buffer_binding_size`.
pub const DESIRED_STORAGE_BUFFER_BINDING_BYTES: u64 = 2 * 1024 * 1024 * 1024;

static ELEVATED_DEVICE: OnceLock<Result<WgpuDevice, String>> = OnceLock::new();
static EFFECTIVE_STORAGE_BINDING_BYTES: OnceLock<usize> = OnceLock::new();

/// Burn [`WgpuDevice`] initialized once with elevated buffer limits.
pub fn elevated_wgpu_device() -> Result<WgpuDevice, GraphError> {
    ELEVATED_DEVICE
        .get_or_init(init_elevated_wgpu_device)
        .clone()
        .map_err(|reason| GraphError::BurnRuntimeFailed { reason })
}

/// Effective `max_storage_buffer_binding_size` of the elevated device (queried after init).
pub fn effective_storage_buffer_binding_bytes() -> usize {
    if let Some(&bytes) = EFFECTIVE_STORAGE_BINDING_BYTES.get() {
        return bytes;
    }
    if elevated_wgpu_device().is_err() {
        return WEBGPU_MIN_STORAGE_BUFFER_BINDING_BYTES;
    }
    EFFECTIVE_STORAGE_BINDING_BYTES
        .get()
        .copied()
        .unwrap_or(WEBGPU_MIN_STORAGE_BUFFER_BINDING_BYTES)
}

fn init_elevated_wgpu_device() -> Result<WgpuDevice, String> {
    let setup = create_elevated_wgpu_setup()?;
    let binding_limit = setup.device.limits().max_storage_buffer_binding_size as usize;
    let _ = EFFECTIVE_STORAGE_BINDING_BYTES.set(binding_limit);
    Ok(init_device(setup, RuntimeOptions::default()))
}

/// Build `required_limits` for `request_device`: adapter maximum, at least [`DESIRED_STORAGE_BUFFER_BINDING_BYTES`].
fn elevated_device_limits(adapter: &wgpu::Adapter) -> wgpu::Limits {
    let supported = adapter.limits();
    let desired_binding = DESIRED_STORAGE_BUFFER_BINDING_BYTES
        .max(WEBGPU_MIN_STORAGE_BUFFER_BINDING_BYTES as u64 + 1)
        .min(supported.max_storage_buffer_binding_size);
    let desired_buffer = desired_binding.min(supported.max_buffer_size);

    let mut limits = supported;
    limits.max_storage_buffer_binding_size = desired_binding;
    limits.max_buffer_size = desired_buffer.max(limits.max_buffer_size);
    limits
}

async fn create_elevated_wgpu_setup_async() -> Result<WgpuSetup, String> {
    let backend = AutoGraphicsApi::backend();
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::from(backend),
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        })
        .await
        .map_err(|err| format!("wgpu request_adapter failed: {err}"))?;

    let limits = elevated_device_limits(&adapter);
    let features = adapter
        .features()
        .difference(Features::MAPPABLE_PRIMARY_BUFFERS);

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("rustnn-burn-elevated-limits"),
            required_features: features,
            required_limits: limits,
            memory_hints: wgpu::MemoryHints::MemoryUsage,
            trace: wgpu::Trace::Off,
            experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
        })
        .await
        .map_err(|err| format!("wgpu request_device failed: {err:?}"))?;

    Ok(WgpuSetup {
        instance,
        adapter,
        device,
        queue,
        backend,
    })
}

fn create_elevated_wgpu_setup() -> Result<WgpuSetup, String> {
    future::block_on(create_elevated_wgpu_setup_async())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn desired_binding_exceeds_webgpu_spec_minimum() {
        assert!(
            DESIRED_STORAGE_BUFFER_BINDING_BYTES > WEBGPU_MIN_STORAGE_BUFFER_BINDING_BYTES as u64
        );
    }
}
