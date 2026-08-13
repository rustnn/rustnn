//! Browser WebNN cases skipped during graph construction.
//!
//! Generated bindings currently do not mark builder operations as `catch`, so a browser
//! validation error traps the Wasm test. Keep Chrome validation failures skipped until bindings
//! become fallible.

const EXPECTED_FAILURES: &[(&str, &str)] = &[
    (
        "l2Pool2d float16 4D constant tensor all positive default options",
        "Chrome WebNN's TFLite backend does not support this float16 constant l2Pool2d configuration",
    ),
    (
        "l2Pool2d float32 4D tensor options.dilations with options.strides",
        "Chrome WebNN's TFLite backend does not support this l2Pool2d dilation/stride configuration",
    ),
    (
        "isInfinite float16 2D tensor",
        "Chrome WebNN does not support float16 isInfinite for this tensor shape",
    ),
    (
        "isNaN float32 special values",
        "Chrome WebNN does not support float32 special values for isNaN",
    ),
    (
        "l2Pool2d float32 4D tensor all positive default options",
        "Chrome WebNN's TFLite backend does not support this float32 l2Pool2d configuration",
    ),
    (
        "l2Pool2d float16 4D tensor options.windowDimensions",
        "Chrome WebNN's TFLite backend does not support this float16 l2Pool2d window configuration",
    ),
    (
        "isInfinite float32 large finite values",
        "Chrome WebNN does not correctly handle large finite float32 values for isInfinite",
    ),
    (
        "isNaN float16 special values",
        "Chrome WebNN does not support float16 special values for isNaN",
    ),
    (
        "layerNormalization float32 4D tensor options.scale and options.axes=[0, 2]",
        "Chrome WebNN's TFLite backend can leave this layerNormalization scale/axes build pending",
    ),
    (
        "prelu int64 2D constant tensors",
        "Chrome WebNN PReLU supports only float32 and float16",
    ),
    (
        "prelu float32 broadcast 5D x 5D slope with expanded output shape",
        "Chrome WebNN rejects this PReLU broadcast shape",
    ),
    (
        "convTranspose2d float32 4D input and filter tensors options.groups",
        "Chrome WebNN's TFLite backend does not support convTranspose2d groups",
    ),
    (
        "convTranspose2d float32 4D input and filter tensors options.groups=2 options.strides=[2, 2]",
        "Chrome WebNN's TFLite backend does not support convTranspose2d groups",
    ),
    (
        "convTranspose2d float32 4D input and filter tensors options.padding",
        "Chrome WebNN's TFLite backend does not support explicit convTranspose2d padding",
    ),
    (
        "convTranspose2d float32 4D input and filter tensors options.dilations",
        "Chrome WebNN's TFLite backend does not support convTranspose2d dilations",
    ),
    (
        "convTranspose2d same output size different padding (padding=2, outputPadding=2))",
        "Chrome WebNN's TFLite backend does not support explicit convTranspose2d padding",
    ),
    (
        "convTranspose2d float16 4D input and filter tensors options.padding",
        "Chrome WebNN's TFLite backend does not support explicit convTranspose2d padding",
    ),
    (
        "convTranspose2d float16 4D input and filter tensors options.dilations",
        "Chrome WebNN's TFLite backend does not support convTranspose2d dilations",
    ),
    (
        "dequantizeLinear uint4 1D tensor of even size with float32 1D scale",
        "the Rust WebNN converter does not support uint4 operands",
    ),
    (
        "dequantizeLinear uint4 1D tensor of odd size with float32 1D scale",
        "the Rust WebNN converter does not support uint4 operands",
    ),
    (
        "dequantizeLinear uint4 4D tensor with broadcasting float32 4D scale and uint4 4D zeroPoint",
        "the Rust WebNN converter does not support uint4 operands",
    ),
    (
        "dequantizeLinear uint4 3D tensor with float32 3D scale, block_size = [1, 1, 2]",
        "the Rust WebNN converter does not support uint4 operands",
    ),
    (
        "dequantizeLinear int4 1D tensor of even size with float32 1D scale",
        "the Rust WebNN converter does not support int4 operands",
    ),
    (
        "dequantizeLinear int4 1D tensor of odd size with float32 1D scale",
        "the Rust WebNN converter does not support int4 operands",
    ),
    (
        "per-tensor dequantizeLinear for int4 4D tensor with float32 4D scale",
        "the Rust WebNN converter does not support int4 operands",
    ),
    (
        "dequantizeLinear uint4 1D tensor of even size with float16 1D scale",
        "the Rust WebNN converter does not support uint4 operands",
    ),
    (
        "dequantizeLinear uint4 1D tensor of odd size with float16 1D scale",
        "the Rust WebNN converter does not support uint4 operands",
    ),
    (
        "dequantizeLinear uint4 4D tensor with broadcasting float16 4D scale and uint4 4D zeroPoint",
        "the Rust WebNN converter does not support uint4 operands",
    ),
    (
        "dequantizeLinear uint4 3D tensor with float16 3D scale, block_size = [1, 1, 2]",
        "the Rust WebNN converter does not support uint4 operands",
    ),
    (
        "dequantizeLinear int4 1D tensor of even size with float16 1D scale",
        "the Rust WebNN converter does not support int4 operands",
    ),
    (
        "dequantizeLinear int4 1D tensor of odd size with float16 1D scale",
        "the Rust WebNN converter does not support int4 operands",
    ),
    (
        "per-tensor dequantizeLinear for int4 4D tensor with float16 4D scale",
        "the Rust WebNN converter does not support int4 operands",
    ),
    (
        "gru float32 tensors steps=1 with options.bias, options.recurrentBias and options.activations=['relu', 'relu']",
        "the Rust WebNN converter does not supply the default GRU direction",
    ),
    (
        "gru float32 tensors steps=1 with options.bias, options.recurrentBias and options.activations=['relu', 'relu'] and reset_after=true",
        "the Rust WebNN converter does not supply the default GRU direction",
    ),
    (
        "gru float32 tensors steps=1 with options.bias, options.recurrentBias, options.activations=['relu', 'relu'] and explicit options.layout='zrn'",
        "the Rust WebNN converter does not supply the default GRU direction",
    ),
    (
        "gru float32 tensors steps=1 with options.bias, options.recurrentBias, options.activations=['relu', 'relu'] and options.layout='rzn'",
        "the Rust WebNN converter does not supply the default GRU direction",
    ),
    (
        "gru float32 tensors steps=1 with options.bias, options.recurrentBias, options.activations=['relu', 'relu'] and options.initialHiddenState",
        "the Rust WebNN converter does not supply the default GRU direction",
    ),
    (
        "gru float16 tensors steps=1 with options.bias, options.recurrentBias and options.activations=['relu', 'relu']",
        "the Rust WebNN converter does not supply the default GRU direction",
    ),
    (
        "gru float16 tensors steps=1 with options.bias, options.recurrentBias and options.activations=['relu', 'relu'] and resetAfter=true",
        "the Rust WebNN converter does not supply the default GRU direction",
    ),
    (
        "gru float16 tensors steps=1 with options.bias, options.recurrentBias, options.activations=['relu', 'relu'] and explicit options.layout='zrn'",
        "the Rust WebNN converter does not supply the default GRU direction",
    ),
    (
        "gru float16 tensors steps=1 with options.bias, options.recurrentBias, options.activations=['relu', 'relu'] and options.layout='rzn'",
        "the Rust WebNN converter does not supply the default GRU direction",
    ),
    (
        "gru float16 tensors steps=1 with options.bias, options.recurrentBias, options.activations=['relu', 'relu'] and options.initialHiddenState",
        "the Rust WebNN converter does not supply the default GRU direction",
    ),
    (
        "averagePool2d float32 4D tensor options.dilations",
        "Chrome WebNN's TFLite backend does not support Pool2d dilations",
    ),
    (
        "averagePool2d float16 4D tensor options.dilations",
        "Chrome WebNN's TFLite backend does not support Pool2d dilations",
    ),
    (
        "l2Pool2d float32 4D tensor options.dilations",
        "Chrome WebNN's TFLite backend does not support Pool2d dilations",
    ),
    (
        "l2Pool2d float16 4D tensor options.dilations",
        "Chrome WebNN's TFLite backend does not support Pool2d dilations",
    ),
    (
        "maxPool2d float32 4D tensor options.dilations",
        "Chrome WebNN's TFLite backend does not support Pool2d dilations",
    ),
    (
        "maxPool2d float16 4D tensor options.dilations",
        "Chrome WebNN's TFLite backend does not support Pool2d dilations",
    ),
    (
        "greater float32 5D tensors",
        "Chrome WebNN binary comparison supports ranks 0 through 4 only",
    ),
    (
        "greater float16 5D tensors",
        "Chrome WebNN binary comparison supports ranks 0 through 4 only",
    ),
    (
        "neg int8 4D tensor",
        "Chrome WebNN neg does not support int8",
    ),
    (
        "reduceL1 uint32 4D tensor options.axes with options.keepDimensions=false",
        "Chrome WebNN reduceL1 supports only float32, float16, and int32",
    ),
    (
        "abs int8 4D tensor",
        "Chrome WebNN abs supports only float32, float16, and int32",
    ),
    (
        "abs int64 4D tensor",
        "Chrome WebNN abs supports only float32, float16, and int32",
    ),
    (
        "clamp int8 1D tensor",
        "Chrome WebNN clamp supports only float32 and float16",
    ),
    (
        "clamp uint8 1D tensor",
        "Chrome WebNN clamp supports only float32 and float16",
    ),
    (
        "clamp int32 1D tensor",
        "Chrome WebNN clamp supports only float32 and float16",
    ),
    (
        "clamp uint32 1D tensor",
        "Chrome WebNN clamp supports only float32 and float16",
    ),
    (
        "clamp int64 1D tensor",
        "Chrome WebNN clamp supports only float32 and float16",
    ),
    (
        "clamp uint64 1D tensor",
        "Chrome WebNN clamp supports only float32 and float16",
    ),
    (
        "clamp int64 1D tensor with bigint max",
        "Chrome WebNN clamp supports only float32 and float16",
    ),
    (
        "clamp uint64 1D tensor with bigint max",
        "Chrome WebNN clamp supports only float32 and float16",
    ),
    (
        "clamp uint64 1D tensor with Number min and max",
        "Chrome WebNN clamp supports only float32 and float16",
    ),
    (
        "equal float32 5D tensors",
        "Chrome WebNN binary comparison supports ranks 0 through 4 only",
    ),
    (
        "equal float16 5D tensors",
        "Chrome WebNN binary comparison supports ranks 0 through 4 only",
    ),
    (
        "greaterOrEqual float32 5D tensors",
        "Chrome WebNN binary comparison supports ranks 0 through 4 only",
    ),
    (
        "greaterOrEqual float16 5D tensors",
        "Chrome WebNN binary comparison supports ranks 0 through 4 only",
    ),
    (
        "lesserOrEqual float32 5D tensors",
        "Chrome WebNN binary comparison supports ranks 0 through 4 only",
    ),
    (
        "lesserOrEqual float16 5D tensors",
        "Chrome WebNN binary comparison supports ranks 0 through 4 only",
    ),
    (
        "lesser float32 5D tensors",
        "Chrome WebNN binary comparison supports ranks 0 through 4 only",
    ),
    (
        "lesser float16 5D tensors",
        "Chrome WebNN binary comparison supports ranks 0 through 4 only",
    ),
    (
        "pow float32 5D base tensor and 5D integer exponent tensor",
        "Chrome WebNN pow supports ranks 0 through 4 only",
    ),
    (
        "pow float16 5D base tensor and 5D integer exponent tensor",
        "Chrome WebNN pow supports ranks 0 through 4 only",
    ),
    (
        "sub int8 4D tensors",
        "Chrome WebNN sub does not support int8",
    ),
    (
        "sub uint8 4D tensors",
        "Chrome WebNN sub does not support uint8",
    ),
    (
        "sub uint32 4D tensors",
        "Chrome WebNN sub does not support uint32",
    ),
    (
        "sub uint64 4D tensors",
        "Chrome WebNN sub does not support uint64",
    ),
    (
        "argMin uint32 4D tensor, axis=1, all options",
        "Chrome WebNN argMin does not support uint32",
    ),
    (
        "argMin int64 4D tensor, axis=0, all options",
        "Chrome WebNN argMin does not support int64",
    ),
    (
        "argMin uint64 4D tensor, axis=1, all options",
        "Chrome WebNN argMin does not support uint64",
    ),
    (
        "argMax uint32 4D tensor, axis=1, all options",
        "Chrome WebNN argMax does not support uint32",
    ),
    (
        "argMax int64 4D tensor, axis=0, all options",
        "Chrome WebNN argMax does not support int64",
    ),
    (
        "argMax uint64 4D tensor, axis=1, all options",
        "Chrome WebNN argMax does not support uint64",
    ),
    (
        "layer_normalization::layerNormalization float32 2D tensor axes=[]",
        "hang",
    ),
    (
        "is_infinite::isInfinite float32 positive infinity only",
        "hang",
    ),
    (
        "leaky_relu::leakyRelu float16 1D constant tensor default options",
        "hang",
    ),
    (
        "l2Pool2d::l2Pool2d float32 4D tensor options.outputShapeRounding=floor",
        "hang",
    ),
    (
        "l2Pool2d::l2Pool2d float16 4D tensor options.dilations with options.strides",
        "hang",
    ),
    (
        "l2Pool2d::l2Pool2d float16 4D tensor options.dilations with options.strides",
        "hang",
    ),
    ("is_nan::isNaN float16 2D tensor", "hang"),
    ("is_nan::isNaN float16 positive 0D scalar", "hang"),
    (
        "l2Pool2d::l2Pool2d float32 4D tensor options.outputSizes ignores options.outputShapeRounding=ceil",
        "hang",
    ),
    (
        "layer_normalization::layerNormalization float32 2D tensor axes=[]",
        "hang",
    ),
    (
        "l2Pool2d::l2Pool2d float32 4D tensor options.outputSizes ignores options.outputShapeRounding=ceil",
        "hang",
    ),
    (
        "leaky_relu::leakyRelu float16 2D tensor default options",
        "hang",
    ),
    (
        "layer_normalization::layerNormalization float32 4D tensor options.scale",
        "hang",
    ),
    (
        "leaky_relu::leakyRelu float16 2D tensor positive options.alpha",
        "hang",
    ),
    (
        "l2Pool2d::l2Pool2d float16 4D tensor options.layout=nchw",
        "hang",
    ),
    ("is_nan::isNaN float16 1D tensor", "hang"),
    (
        "l2Pool2d::l2Pool2d float16 4D tensor options.outputSizes ignores options.outputShapeRounding=ceil",
        "hang",
    ),
    (
        "l2Pool2d::l2Pool2d float16 4D tensor options.strides",
        "hang",
    ),
    (
        "l2Pool2d::l2Pool2d float32 4D tensor options.outputShapeRounding=ceil",
        "hang",
    ),
    (
        "l2Pool2d::l2Pool2d float32 4D tensor options.outputShapeRounding=ceil with asymmetric window",
        "hang",
    ),
    (
        "l2Pool2d::l2Pool2d float32 4D tensor options.outputShapeRounding=floor",
        "hang",
    ),
    (
        "l2Pool2d::l2Pool2d float16 4D tensor all negative default options",
        "hang",
    ),
    (
        "layer_normalization::layerNormalization float32 4D tensor options.epsilon",
        "hang",
    ),
    (
        "l2Pool2d::l2Pool2d float32 4D tensor options.layout=nhwc",
        "hang",
    ),
    (
        "l2Pool2d::l2Pool2d float32 4D tensor options.outputShapeRounding=ceil with asymmetric window",
        "hang",
    ),
    ("is_infinite::isInfinite float16 1D tensor", "hang"),
];

pub fn reason(operation: &str, name: &str) -> Option<&'static str> {
    if operation == "mlNumber" {
        return Some(
            "Chrome WebNN's clamp implementation does not support the 64-bit integer helper graphs in mlNumber",
        );
    }
    if operation == "layer_normalization"
        && name.starts_with("layerNormalization float16 ")
        && (name.contains("3D") || name.contains("4D") || name.contains("5D"))
    {
        return Some(
            "Chrome WebNN's TFLite backend can leave float16 rank-3-or-higher layerNormalization builds pending",
        );
    }

    EXPECTED_FAILURES
        .iter()
        .find_map(|(expected_name, reason)| (*expected_name == name).then_some(*reason))
}
