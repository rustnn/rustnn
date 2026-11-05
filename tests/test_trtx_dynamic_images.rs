// SPDX-FileCopyrightText: 2026 NVIDIA
// SPDX-License-Identifier: Apache-2.0

#![cfg(all(feature = "trtx-runtime", feature = "dynamic-inputs"))]

use rustnn::{
    backend_selection::BackendDevice,
    mlcontext::{
        MLContext, MLContextOptions, MLNamedTensors, MLPowerPreference, MLTensorDescriptor,
    },
    mlcontextoptions::{RustNNOptions, TrtxInputProfile},
    operator_enums::MLOperandDataType,
};

// Pixel unshuffle, channel reduction, scalar broadcast and upsampling exercise the
// ESRGAN lowering paths with values that expose any use of maximum instead of active shapes.
#[test]
#[ignore = "requires an NVIDIA GPU and TensorRT-RTX"]
fn dynamic_pixel_unshuffle_resize_and_repeated_builds() -> anyhow::Result<()> {
    exercise_dynamic_images(false)
}

#[test]
#[ignore = "requires an NVIDIA GPU and TensorRT-RTX"]
fn placeholders_refit_dynamic_pixel_unshuffle() -> anyhow::Result<()> {
    exercise_dynamic_images(true)
}

fn exercise_dynamic_images(weight_placeholders: bool) -> anyhow::Result<()> {
    let text = r#"webnn_graph "dynamic_image" v2 {
        inputs { input: f32[1, 1, dyn("h", 6), dyn("w", 8)]; }
        consts {
            dead: f32[2] @scalar(1.0);
            filter: f32[1, 4, 1, 1] @weights("filter");
            scale: f32[] @scalar(0.5);
        }
        nodes {
            [u] = unsqueeze(input, axes=[3]);
            [a] = reshape(u, newShape=[1, 1, {"name":"h / 2","maxSize":3}, 2, {"name":"w","maxSize":8}]);
            [b] = reshape(a, newShape=[1, 1, {"name":"h / 2","maxSize":3}, 2, {"name":"w / 2","maxSize":4}, 2]);
            [c] = transpose(b, permutation=[0, 1, 3, 5, 2, 4]);
            [d] = reshape(c, newShape=[1, 4, {"name":"h / 2","maxSize":3}, {"name":"w / 2","maxSize":4}]);
            [e] = conv2d(d, filter);
            [f] = mul(e, scale);
            [output] = resample2d(f, axes=[2, 3], scales=[2.0, 2.0], mode="nearest-neighbor");
        }
        outputs { output; }
    }"#;
    let json = webnn_graph::parser::parse_wg_text(text)?;
    let mut info = rustnn::webnn_json::from_graph_json(&json)?;
    let filter = info
        .operands
        .iter()
        .position(|operand| operand.name.as_deref() == Some("filter"))
        .unwrap() as u32;
    info.constant_operand_ids_to_handles
        .get_mut(&filter)
        .unwrap()
        .data = [1_f32, 10., 100., 1000.]
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    let mut options = RustNNOptions::default();
    options.trtx.engine_caching = false;
    options.trtx.weight_placeholders = weight_placeholders;
    options.trtx.runtime_cache = false;
    options.trtx.input_profiles.insert(
        "input".into(),
        TrtxInputProfile {
            min: vec![1, 1, 2, 2],
            optimal: vec![1, 1, 4, 4],
        },
    );
    let options = MLContextOptions::new(MLPowerPreference::HighPerformance, true)
        .with_rustnn_device_hint(BackendDevice::Trtx { cuda_device_idx: 0 })
        .with_rustnn_options(options);
    let mut context = MLContext::create(&options)?;
    // Building twice on the same context must not accumulate optimization profiles.
    let _first = context.rustnn_build_graph(info.clone())?;
    let mut graph = context.rustnn_build_graph(info)?;
    for (height, width) in [(4, 4), (6, 8), (2, 6), (4, 4)] {
        let shape = vec![1, 1, height as u64, width as u64];
        let mut input_desc = MLTensorDescriptor::new(MLOperandDataType::Float32, shape.clone());
        input_desc.set_writable(true);
        let mut output_desc = MLTensorDescriptor::new(MLOperandDataType::Float32, shape);
        output_desc.set_readable(true);
        let input = context.create_tensor(&input_desc)?;
        let output = context.create_tensor(&output_desc)?;
        let data: Vec<f32> = (0..height * width).map(|value| value as f32).collect();
        context.write_tensor(&input, &data)?;
        context.dispatch(
            &mut graph,
            &MLNamedTensors::from([("input", &input)]),
            &MLNamedTensors::from([("output", &output)]),
        )?;
        let mut actual = vec![0_f32; data.len()];
        context.read_tensor(&output, &mut actual)?;
        for y in 0..height {
            for x in 0..width {
                let base = (y / 2 * 2) * width + x / 2 * 2;
                let expected = (data[base]
                    + 10. * data[base + 1]
                    + 100. * data[base + width]
                    + 1000. * data[base + width + 1])
                    * 0.5;
                assert_eq!(
                    actual[y * width + x],
                    expected,
                    "{height}x{width} at ({y},{x})"
                );
            }
        }
    }
    Ok(())
}
