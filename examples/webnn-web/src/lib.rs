use anyhow::anyhow;
use js_sys::{Float32Array, Reflect};
use rustnn::{converters::WebNNConverter, webnn_json::from_graph_json};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use web_sys::{
    MlContext, MlContextOptions, MlOperandDataType, MlPowerPreference, MlTensorDescriptor, window,
};

async fn run_inference() -> anyhow::Result<()> {
    let contents = r#"
webnn_graph "sample_graph" v1 {
  inputs {
    lhs: f32[2, 2];
  }

  consts {
    rhs: f32[2, 2] @scalar(1.0);
  }

  nodes {
    sum = add(lhs, rhs);
  }

  outputs { sum; }
}"#;

    let sanitized = rustnn::loader::sanitize_webnn_identifiers(contents);
    // Parse .webnn text format
    let graph_json = webnn_graph::parser::parse_wg_text(&sanitized)?;

    let graph_info = from_graph_json(&graph_json)?;
    web_sys::console::log_1(&format!("Parsed successfully: {graph_info:?}").into());

    // will generate for internal MlContext
    //let converted = WebNNConverter::default().convert_async(&graph_info).await?;

    // Let's create for own context
    let window = window().expect("no global `window` exists");
    let navigator = window.navigator();
    let ml = navigator.ml();

    web_sys::console::log_1(&"Requesting WebNN Context...".into());
    let options = MlContextOptions::new();
    options.set_accelerated(true);
    options.set_power_preference(MlPowerPreference::HighPerformance);
    let promise = ml.create_context_with_ml_context_options(&options);
    let result = JsFuture::from(promise)
        .await
        .map_err(|e| anyhow!("Failed to get mlcontext: {e:?}"))?;
    let context: MlContext = result
        .dyn_into()
        .map_err(|e| anyhow!("Failed to get mlcontext: {e:?}"))?;
    let converted = WebNNConverter::default()
        .convert_async(&context, &graph_info)
        .await?;

    let graph = converted
        .graph
        .ok_or_else(|| anyhow!("No MlGraph on conversion result"))?;
    web_sys::console::log_1(&graph);

    let array = Float32Array::new_from_slice(&[1., 1., 1., 1.]);
    let desc = MlTensorDescriptor::new(MlOperandDataType::Float32, &[2.into(), 2.into()]);
    desc.set_writable(true);
    let input = JsFuture::from(context.create_tensor(&desc))
        .await
        .map_err(|e| anyhow!("Failed to create tensor: {e:?}"))?;
    context.write_tensor_with_buffer_source(&input, &array);
    let desc = MlTensorDescriptor::new(MlOperandDataType::Float32, &[2.into(), 2.into()]);
    desc.set_readable(true);
    let output = JsFuture::from(context.create_tensor(&desc))
        .await
        .map_err(|e| anyhow!("Failed to create tensor: {e:?}"))?;

    let inputs = js_sys::Object::new_typed();
    Reflect::set(&inputs, &"lhs".into(), &input).unwrap();
    let outputs = js_sys::Object::new_typed();
    Reflect::set(&outputs, &"sum".into(), &output).unwrap();
    web_sys::console::log_2(&"Before dispatch".into(), &array);
    context.dispatch(&graph, &inputs, &outputs);
    JsFuture::from(context.read_tensor_with_buffer_source(&output, &array))
        .await
        .map_err(|e| anyhow!("Failed to read tensor: {e:?}"))?;
    web_sys::console::log_2(&"After dispatch".into(), &array);

    Ok(())
}

#[wasm_bindgen(start)]
async fn start() -> Result<(), JsValue> {
    run_inference()
        .await
        .map_err(|e| -> JsValue { format!("{e}").into() })
}
