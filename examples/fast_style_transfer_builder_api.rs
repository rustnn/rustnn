// Port of https://d3i5xkfad89fac.cloudfront.net/test-data/models/fast_style_transfer_nchw/weights
// (Apache 2.0)
use std::{
    collections::HashMap,
    fmt,
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::{Context, Result, bail, ensure};
use clap::{Parser, ValueEnum};
use futures_util::future::join_all;
use rustnn::mlcontext::{
    MLContext, MLContextOptions, MLGraphBuilder, MLOperand, MLOperandDescriptor, MLPowerPreference,
    MLTensorDescriptor,
};
use rustnn::operator_enums::MLOperandDataType;
use rustnn::operator_options::{
    MLConv2dOptions, MLConvTranspose2dOptions, MLPadOptions, MLReduceOptions,
};
use tokio::sync::Mutex;

const DEFAULT_WEIGHTS_BASE_URL: &str = "https://raw.githubusercontent.com/webmachinelearning/test-data/0495fc5b5e4ccf77f745b747aa43e12a71a30cff/models/fast_style_transfer_nchw/weights/";

const BATCH: usize = 1;
const CHANNELS: usize = 3;
const HEIGHT: usize = 540;
const WIDTH: usize = 540;
const INPUT_SHAPE: [u64; 4] = [BATCH as u64, CHANNELS as u64, HEIGHT as u64, WIDTH as u64];
const OUTPUT_SHAPE: [u64; 4] = INPUT_SHAPE;
const IMAGE_ELEMENT_COUNT: usize = BATCH * CHANNELS * HEIGHT * WIDTH;
const PIPELINE_DEPTH: usize = 2;

const WEIGHTS: &[WeightSpec] = &[
    WeightSpec::new("weightConv0", "Variable_read__0__cf__0_0.npy"),
    WeightSpec::new("variableAdd0", "Variable_1_read__1__cf__1_0.npy"),
    WeightSpec::new("variableMul0", "Variable_2_read__12__cf__12_0.npy"),
    WeightSpec::new("weightConv1", "Variable_3_read__23__cf__23_0.npy"),
    WeightSpec::new("variableAdd1", "Variable_4_read__34__cf__34_0.npy"),
    WeightSpec::new("variableMul1", "Variable_5_read__43__cf__43_0.npy"),
    WeightSpec::new("weightConv2", "Variable_6_read__44__cf__44_0.npy"),
    WeightSpec::new("variableAdd2", "Variable_7_read__45__cf__45_0.npy"),
    WeightSpec::new("variableMul2", "Variable_8_read__46__cf__46_0.npy"),
    WeightSpec::new("weightConv3", "Variable_9_read__47__cf__47_0.npy"),
    WeightSpec::new("variableAdd3", "Variable_10_read__2__cf__2_0.npy"),
    WeightSpec::new("variableMul3", "Variable_11_read__3__cf__3_0.npy"),
    WeightSpec::new("weightConv4", "Variable_12_read__4__cf__4_0.npy"),
    WeightSpec::new("variableAdd4", "Variable_13_read__5__cf__5_0.npy"),
    WeightSpec::new("variableMul4", "Variable_14_read__6__cf__6_0.npy"),
    WeightSpec::new("weightConv5", "Variable_15_read__7__cf__7_0.npy"),
    WeightSpec::new("variableAdd5", "Variable_16_read__8__cf__8_0.npy"),
    WeightSpec::new("variableMul5", "Variable_17_read__9__cf__9_0.npy"),
    WeightSpec::new("weightConv6", "Variable_18_read__10__cf__10_0.npy"),
    WeightSpec::new("variableAdd6", "Variable_19_read__11__cf__11_0.npy"),
    WeightSpec::new("variableMul6", "Variable_20_read__13__cf__13_0.npy"),
    WeightSpec::new("weightConv7", "Variable_21_read__14__cf__14_0.npy"),
    WeightSpec::new("variableAdd7", "Variable_22_read__15__cf__15_0.npy"),
    WeightSpec::new("variableMul7", "Variable_23_read__16__cf__16_0.npy"),
    WeightSpec::new("weightConv8", "Variable_24_read__17__cf__17_0.npy"),
    WeightSpec::new("variableAdd8", "Variable_25_read__18__cf__18_0.npy"),
    WeightSpec::new("variableMul8", "Variable_26_read__19__cf__19_0.npy"),
    WeightSpec::new("weightConv9", "Variable_27_read__20__cf__20_0.npy"),
    WeightSpec::new("variableAdd9", "Variable_28_read__21__cf__21_0.npy"),
    WeightSpec::new("variableMul9", "Variable_29_read__22__cf__22_0.npy"),
    WeightSpec::new("weightConv10", "Variable_30_read__24__cf__24_0.npy"),
    WeightSpec::new("variableAdd10", "Variable_31_read__25__cf__25_0.npy"),
    WeightSpec::new("variableMul10", "Variable_32_read__26__cf__26_0.npy"),
    WeightSpec::new("weightConv11", "Variable_33_read__27__cf__27_0.npy"),
    WeightSpec::new("variableAdd11", "Variable_34_read__28__cf__28_0.npy"),
    WeightSpec::new("variableMul11", "Variable_35_read__29__cf__29_0.npy"),
    WeightSpec::new("weightConv12", "Variable_36_read__30__cf__30_0.npy"),
    WeightSpec::new("variableAdd12", "Variable_37_read__31__cf__31_0.npy"),
    WeightSpec::new("variableMul12", "Variable_38_read__32__cf__32_0.npy"),
    WeightSpec::new("weightConvTranspose0", "Variable_39_read__33__cf__33_0.npy"),
    WeightSpec::new("variableAdd13", "Variable_40_read__35__cf__35_0.npy"),
    WeightSpec::new("variableMul13", "Variable_41_read__36__cf__36_0.npy"),
    WeightSpec::new("weightConvTranspose1", "Variable_42_read__37__cf__37_0.npy"),
    WeightSpec::new("variableAdd14", "Variable_43_read__38__cf__38_0.npy"),
    WeightSpec::new("variableMul14", "Variable_44_read__39__cf__39_0.npy"),
    WeightSpec::new("weightConv13", "Variable_45_read__40__cf__40_0.npy"),
    WeightSpec::new("variableAdd15", "Variable_46_read__41__cf__41_0.npy"),
    WeightSpec::new("variableMul15", "Variable_47_read__42__cf__42_0.npy"),
];

#[derive(Parser, Debug)]
#[command(about = "Load Fast Style Transfer weights into RustNN")]
struct Args {
    #[arg(short, long, help = "Input image path")]
    input: PathBuf,
    #[arg(short, long, help = "Output image path")]
    output: PathBuf,
    #[arg(long, default_value_t = 1, help = "Number of inferences to run")]
    num_inferences: usize,
    #[arg(
        long,
        help = "Disable pipelined tensor buffering and read each result before dispatching the next inference"
    )]
    disable_pipelining: bool,
    #[arg(long, value_enum, default_value_t = StyleModel::StarryNight, visible_alias = "model-id")]
    model: StyleModel,
    #[arg(long, default_value = DEFAULT_WEIGHTS_BASE_URL)]
    weights_base_url: String,
    #[arg(long)]
    sequential_weights: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum StyleModel {
    #[value(name = "starry-night")]
    StarryNight,
    #[value(name = "self-portrait")]
    SelfPortrait,
    #[value(name = "bedroom")]
    Bedroom,
    #[value(name = "sunflowers-bew")]
    SunflowersBew,
    #[value(name = "red-vineyards")]
    RedVineyards,
    #[value(name = "sien-with-a-cigar")]
    SienWithACigar,
    #[value(name = "la-campesinos")]
    LaCampesinos,
    #[value(name = "soup-distribution")]
    SoupDistribution,
    #[value(name = "wheatfield_with_crows")]
    WheatfieldWithCrows,
}

impl StyleModel {
    const fn id(self) -> &'static str {
        match self {
            Self::StarryNight => "starry-night",
            Self::SelfPortrait => "self-portrait",
            Self::Bedroom => "bedroom",
            Self::SunflowersBew => "sunflowers-bew",
            Self::RedVineyards => "red-vineyards",
            Self::SienWithACigar => "sien-with-a-cigar",
            Self::LaCampesinos => "la-campesinos",
            Self::SoupDistribution => "soup-distribution",
            Self::WheatfieldWithCrows => "wheatfield_with_crows",
        }
    }

    const fn title(self) -> &'static str {
        match self {
            Self::StarryNight => "The starry night",
            Self::SelfPortrait => "Self-Portrait",
            Self::Bedroom => "Vincent's Bedroom in Arles",
            Self::SunflowersBew => "Sunflowers (1889)",
            Self::RedVineyards => "The Red Vineyard",
            Self::SienWithACigar => "Sien with a cigar",
            Self::LaCampesinos => "Rest from Work",
            Self::SoupDistribution => "Soup Distribution in a Public Soup Kitchen",
            Self::WheatfieldWithCrows => "Wheatfield with Crows",
        }
    }
}

impl fmt::Display for StyleModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.id())
    }
}

#[derive(Debug, Clone, Copy)]
struct WeightSpec {
    name: &'static str,
    file_name: &'static str,
}

impl WeightSpec {
    const fn new(name: &'static str, file_name: &'static str) -> Self {
        Self { name, file_name }
    }
}

#[derive(Debug)]
struct LoadedWeight {
    name: &'static str,
    operand: MLOperand,
    data_type: NpyDataType,
    shape: Vec<u64>,
    byte_len: usize,
}

struct NetworkWeights<'a> {
    weights: &'a [LoadedWeight],
}

impl<'a> NetworkWeights<'a> {
    fn new(weights: &'a [LoadedWeight]) -> Result<Self> {
        for spec in WEIGHTS {
            ensure!(
                weights.iter().any(|weight| weight.name == spec.name),
                "missing loaded weight `{}`",
                spec.name
            );
        }
        Ok(Self { weights })
    }

    fn get(&self, name: &str) -> Result<MLOperand> {
        self.weights
            .iter()
            .find(|weight| weight.name == name)
            .map(|weight| weight.operand)
            .with_context(|| format!("missing loaded weight `{name}`"))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NpyDataType {
    Float16,
    Float32,
    Int32,
    Uint32,
    Int64,
    Uint64,
    Int8,
    Uint8,
}

impl NpyDataType {
    fn from_descr(descr: &str) -> Result<Self> {
        let byte_order = descr
            .chars()
            .next()
            .context("empty NumPy dtype descriptor")?;
        ensure!(
            matches!(byte_order, '<' | '|' | '='),
            "only little-endian or byte-order independent NumPy arrays are supported, got {descr}"
        );
        match descr.trim_start_matches(['<', '|', '=']) {
            "f2" => Ok(Self::Float16),
            "f4" => Ok(Self::Float32),
            "i4" => Ok(Self::Int32),
            "u4" => Ok(Self::Uint32),
            "i8" => Ok(Self::Int64),
            "u8" => Ok(Self::Uint64),
            "i1" => Ok(Self::Int8),
            "u1" => Ok(Self::Uint8),
            _ => bail!("unsupported NumPy dtype descriptor {descr}"),
        }
    }

    const fn operand_data_type(self) -> MLOperandDataType {
        match self {
            Self::Float16 => MLOperandDataType::Float16,
            Self::Float32 => MLOperandDataType::Float32,
            Self::Int32 => MLOperandDataType::Int32,
            Self::Uint32 => MLOperandDataType::Uint32,
            Self::Int64 => MLOperandDataType::Int64,
            Self::Uint64 => MLOperandDataType::Uint64,
            Self::Int8 => MLOperandDataType::Int8,
            Self::Uint8 => MLOperandDataType::Uint8,
        }
    }

    const fn byte_width(self) -> usize {
        match self {
            Self::Float16 => 2,
            Self::Float32 | Self::Int32 | Self::Uint32 => 4,
            Self::Int64 | Self::Uint64 => 8,
            Self::Int8 | Self::Uint8 => 1,
        }
    }
}

#[derive(Debug)]
struct NpyArray {
    data_type: NpyDataType,
    shape: Vec<u64>,
    data: Vec<u8>,
}

fn parse_npy(bytes: &[u8]) -> Result<NpyArray> {
    ensure!(bytes.len() >= 10, "file is too small to be a .npy array");
    ensure!(&bytes[..6] == b"\x93NUMPY", "missing .npy magic header");

    let major = bytes[6];
    let header_len_start = 8usize;
    let (header_len, header_start) = match major {
        1 => (
            u16::from_le_bytes([bytes[8], bytes[9]]) as usize,
            header_len_start + 2,
        ),
        2 | 3 => {
            ensure!(bytes.len() >= 12, "truncated NumPy v{major} header");
            (
                u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
                header_len_start + 4,
            )
        }
        _ => bail!("unsupported NumPy format version {major}"),
    };
    let data_start = header_start
        .checked_add(header_len)
        .context("NumPy header length overflow")?;
    ensure!(data_start <= bytes.len(), "truncated NumPy header");

    let header = std::str::from_utf8(&bytes[header_start..data_start])
        .context("NumPy header is not valid UTF-8")?;
    let descr = parse_header_string(header, "descr")?;
    let fortran_order = parse_header_bool(header, "fortran_order")?;
    ensure!(
        !fortran_order,
        "Fortran-order NumPy arrays are not supported by this loader"
    );
    let shape = parse_header_shape(header)?;
    let data_type = NpyDataType::from_descr(&descr)?;
    let element_count = element_count(&shape)?;
    let expected_data_len = element_count
        .checked_mul(data_type.byte_width())
        .context("NumPy data length overflow")?;
    ensure!(
        bytes.len() >= data_start + expected_data_len,
        "NumPy data is truncated: expected {expected_data_len} data bytes"
    );

    Ok(NpyArray {
        data_type,
        shape,
        data: bytes[data_start..data_start + expected_data_len].to_vec(),
    })
}

fn parse_header_string(header: &str, key: &str) -> Result<String> {
    let key_pos = header
        .find(&format!("'{key}'"))
        .or_else(|| header.find(&format!("\"{key}\"")))
        .with_context(|| format!("NumPy header is missing `{key}`"))?;
    let after_key = &header[key_pos..];
    let colon = after_key
        .find(':')
        .with_context(|| format!("NumPy header key `{key}` is missing `:`"))?;
    let value = after_key[colon + 1..].trim_start();
    let quote = value
        .chars()
        .next()
        .with_context(|| format!("NumPy header key `{key}` has empty value"))?;
    ensure!(
        quote == '\'' || quote == '"',
        "NumPy header key `{key}` is not a string"
    );
    let value = &value[quote.len_utf8()..];
    let end = value
        .find(quote)
        .with_context(|| format!("NumPy header key `{key}` has unterminated string value"))?;
    Ok(value[..end].to_string())
}

fn parse_header_bool(header: &str, key: &str) -> Result<bool> {
    let key_pos = header
        .find(&format!("'{key}'"))
        .or_else(|| header.find(&format!("\"{key}\"")))
        .with_context(|| format!("NumPy header is missing `{key}`"))?;
    let after_key = &header[key_pos..];
    let colon = after_key
        .find(':')
        .with_context(|| format!("NumPy header key `{key}` is missing `:`"))?;
    let value = after_key[colon + 1..].trim_start();
    if value.starts_with("True") {
        Ok(true)
    } else if value.starts_with("False") {
        Ok(false)
    } else {
        bail!("NumPy header key `{key}` is not a bool")
    }
}

fn parse_header_shape(header: &str) -> Result<Vec<u64>> {
    let key_pos = header
        .find("'shape'")
        .or_else(|| header.find("\"shape\""))
        .context("NumPy header is missing `shape`")?;
    let after_key = &header[key_pos..];
    let colon = after_key
        .find(':')
        .context("NumPy header key `shape` is missing `:`")?;
    let value = after_key[colon + 1..].trim_start();
    let open = value
        .find('(')
        .context("NumPy header shape is missing `(`")?;
    let close = value[open + 1..]
        .find(')')
        .context("NumPy header shape is missing `)`")?
        + open
        + 1;
    let body = &value[open + 1..close];
    if body.trim().is_empty() {
        return Ok(Vec::new());
    }
    body.split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(|part| {
            part.parse::<u64>()
                .with_context(|| format!("invalid NumPy shape dimension `{part}`"))
        })
        .collect()
}

fn element_count(shape: &[u64]) -> Result<usize> {
    shape.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim as usize)
            .context("NumPy shape element count overflow")
    })
}

fn f32_values(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("exact chunk size")))
        .collect()
}

fn u16_values(data: &[u8]) -> Vec<u16> {
    data.chunks_exact(2)
        .map(|chunk| u16::from_le_bytes(chunk.try_into().expect("exact chunk size")))
        .collect()
}

fn i32_values(data: &[u8]) -> Vec<i32> {
    data.chunks_exact(4)
        .map(|chunk| i32::from_le_bytes(chunk.try_into().expect("exact chunk size")))
        .collect()
}

fn u32_values(data: &[u8]) -> Vec<u32> {
    data.chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().expect("exact chunk size")))
        .collect()
}

fn i64_values(data: &[u8]) -> Vec<i64> {
    data.chunks_exact(8)
        .map(|chunk| i64::from_le_bytes(chunk.try_into().expect("exact chunk size")))
        .collect()
}

fn u64_values(data: &[u8]) -> Vec<u64> {
    data.chunks_exact(8)
        .map(|chunk| u64::from_le_bytes(chunk.try_into().expect("exact chunk size")))
        .collect()
}

fn add_npy_constant(
    builder: &mut MLGraphBuilder<'_, '_>,
    spec: WeightSpec,
    array: NpyArray,
) -> Result<LoadedWeight> {
    let descriptor =
        MLOperandDescriptor::new(array.data_type.operand_data_type(), array.shape.clone());
    let operand = match array.data_type {
        NpyDataType::Float16 => {
            let values = u16_values(&array.data);
            builder.constant_from_slice(&descriptor, &values)
        }
        NpyDataType::Float32 => {
            let values = f32_values(&array.data);
            builder.constant_from_slice(&descriptor, &values)
        }
        NpyDataType::Int32 => {
            let values = i32_values(&array.data);
            builder.constant_from_slice(&descriptor, &values)
        }
        NpyDataType::Uint32 => {
            let values = u32_values(&array.data);
            builder.constant_from_slice(&descriptor, &values)
        }
        NpyDataType::Int64 => {
            let values = i64_values(&array.data);
            builder.constant_from_slice(&descriptor, &values)
        }
        NpyDataType::Uint64 => {
            let values = u64_values(&array.data);
            builder.constant_from_slice(&descriptor, &values)
        }
        NpyDataType::Int8 => builder.constant_from_slice(&descriptor, &array.data),
        NpyDataType::Uint8 => builder.constant_from_slice(&descriptor, &array.data),
    }
    .with_context(|| format!("add `{}` to MLGraphBuilder", spec.name))?;

    Ok(LoadedWeight {
        name: spec.name,
        operand,
        data_type: array.data_type,
        shape: array.shape,
        byte_len: array.data.len(),
    })
}

fn scalar_constant(
    builder: &mut MLGraphBuilder<'_, '_>,
    label: &str,
    value: f32,
) -> Result<MLOperand> {
    let descriptor = MLOperandDescriptor::new(MLOperandDataType::Float32, vec![1]);
    builder
        .constant_from_slice(&descriptor, &[value])
        .with_context(|| format!("add scalar constant `{label}`"))
}

fn reflection_pad(
    builder: &mut MLGraphBuilder<'_, '_>,
    input: MLOperand,
    amount: u32,
    label: &str,
) -> Result<MLOperand> {
    let padding = vec![0, 0, amount, amount];
    let options = MLPadOptions {
        label: label.to_string(),
        mode: "reflection".to_string(),
        ..Default::default()
    };
    builder
        .pad_with_options(input, padding.clone(), padding, options)
        .with_context(|| label.to_string())
}

fn conv2d(
    builder: &mut MLGraphBuilder<'_, '_>,
    input: MLOperand,
    filter: MLOperand,
    strides: Option<[u32; 2]>,
    label: &str,
) -> Result<MLOperand> {
    let options = MLConv2dOptions {
        label: label.to_string(),
        strides: strides.map(Vec::from).unwrap_or_default(),
        input_layout: "nchw".to_string(),
        filter_layout: "oihw".to_string(),
        ..Default::default()
    };
    builder
        .conv2_with_options(input, filter, options)
        .with_context(|| label.to_string())
}

fn conv_transpose2d(
    builder: &mut MLGraphBuilder<'_, '_>,
    input: MLOperand,
    filter: MLOperand,
    output_sizes: [u32; 2],
    label: &str,
) -> Result<MLOperand> {
    let options = MLConvTranspose2dOptions {
        label: label.to_string(),
        padding: vec![0, 1, 0, 1],
        strides: vec![2, 2],
        output_sizes: Some(Vec::from(output_sizes)),
        input_layout: "nchw".to_string(),
        filter_layout: "iohw".to_string(),
        ..Default::default()
    };
    builder
        .conv_transpose2d_with_options(input, filter, options)
        .with_context(|| label.to_string())
}

fn instance_normalization_fallback(
    builder: &mut MLGraphBuilder<'_, '_>,
    input: MLOperand,
    scale: MLOperand,
    bias: MLOperand,
    epsilon: MLOperand,
    exponent: MLOperand,
    label: &str,
) -> Result<MLOperand> {
    let reduce_options = MLReduceOptions {
        label: format!("{label}/reduce_mean"),
        axes: Some(vec![2, 3]),
        keep_dimensions: true,
    };
    let mean = builder
        .reduce_mean_with_options(input, reduce_options.clone())
        .with_context(|| format!("{label}/mean"))?;
    let centered = builder
        .sub(input, mean)
        .with_context(|| format!("{label}/center"))?;
    let squared = builder
        .mul(centered, centered)
        .with_context(|| format!("{label}/square"))?;
    let variance = builder
        .reduce_mean_with_options(squared, reduce_options)
        .with_context(|| format!("{label}/variance"))?;
    let stabilized = builder
        .add(variance, epsilon)
        .with_context(|| format!("{label}/epsilon"))?;
    let stddev = builder
        .pow(stabilized, exponent)
        .with_context(|| format!("{label}/sqrt"))?;
    let normalized = builder
        .div(centered, stddev)
        .with_context(|| format!("{label}/normalize"))?;
    let scaled = builder
        .mul(scale, normalized)
        .with_context(|| format!("{label}/scale"))?;
    builder
        .add(scaled, bias)
        .with_context(|| format!("{label}/bias"))
}

fn conv_norm_relu(
    builder: &mut MLGraphBuilder<'_, '_>,
    input: MLOperand,
    weights: &NetworkWeights<'_>,
    index: usize,
    pad: u32,
    strides: Option<[u32; 2]>,
    norm_constants: (MLOperand, MLOperand),
) -> Result<MLOperand> {
    let padded = reflection_pad(builder, input, pad, &format!("pad{index}"))?;
    let conv = conv2d(
        builder,
        padded,
        weights.get(&format!("weightConv{index}"))?,
        strides,
        &format!("conv2D{index}"),
    )?;
    let normalized = instance_normalization_fallback(
        builder,
        conv,
        weights.get(&format!("variableMul{index}"))?,
        weights.get(&format!("variableAdd{index}"))?,
        norm_constants.0,
        norm_constants.1,
        &format!("instance_norm{index}"),
    )?;
    builder
        .relu(normalized)
        .with_context(|| format!("relu{index}"))
}

fn conv_norm(
    builder: &mut MLGraphBuilder<'_, '_>,
    input: MLOperand,
    weights: &NetworkWeights<'_>,
    index: usize,
    pad: u32,
    norm_constants: (MLOperand, MLOperand),
) -> Result<MLOperand> {
    conv_norm_with_indices(builder, input, weights, index, index, pad, norm_constants)
}

fn conv_norm_with_indices(
    builder: &mut MLGraphBuilder<'_, '_>,
    input: MLOperand,
    weights: &NetworkWeights<'_>,
    weight_index: usize,
    norm_index: usize,
    pad: u32,
    norm_constants: (MLOperand, MLOperand),
) -> Result<MLOperand> {
    let padded = reflection_pad(builder, input, pad, &format!("pad{weight_index}"))?;
    let conv = conv2d(
        builder,
        padded,
        weights.get(&format!("weightConv{weight_index}"))?,
        None,
        &format!("conv2D{weight_index}"),
    )?;
    instance_normalization_fallback(
        builder,
        conv,
        weights.get(&format!("variableMul{norm_index}"))?,
        weights.get(&format!("variableAdd{norm_index}"))?,
        norm_constants.0,
        norm_constants.1,
        &format!("instance_norm{norm_index}"),
    )
}

fn conv_transpose_norm_relu(
    builder: &mut MLGraphBuilder<'_, '_>,
    input: MLOperand,
    weights: &NetworkWeights<'_>,
    transpose_index: usize,
    norm_index: usize,
    output_sizes: [u32; 2],
    norm_constants: (MLOperand, MLOperand),
) -> Result<MLOperand> {
    let conv = conv_transpose2d(
        builder,
        input,
        weights.get(&format!("weightConvTranspose{transpose_index}"))?,
        output_sizes,
        &format!("convTranspose{transpose_index}"),
    )?;
    let normalized = instance_normalization_fallback(
        builder,
        conv,
        weights.get(&format!("variableMul{norm_index}"))?,
        weights.get(&format!("variableAdd{norm_index}"))?,
        norm_constants.0,
        norm_constants.1,
        &format!("instance_norm{norm_index}"),
    )?;
    builder
        .relu(normalized)
        .with_context(|| format!("relu{norm_index}"))
}

fn build_fast_style_transfer_graph(
    builder: &mut MLGraphBuilder<'_, '_>,
    loaded_weights: &[LoadedWeight],
) -> Result<MLOperand> {
    let weights = NetworkWeights::new(loaded_weights)?;
    let norm_epsilon = scalar_constant(builder, "norm_epsilon", 1.0e-9)?;
    let norm_exponent = scalar_constant(builder, "norm_exponent", 0.5)?;
    let output_scale = scalar_constant(builder, "output_scale", 150.0)?;
    let output_offset = scalar_constant(builder, "output_offset", 127.5)?;

    let input_descriptor =
        MLOperandDescriptor::new(MLOperandDataType::Float32, INPUT_SHAPE.to_vec());
    let input = builder
        .input("input", &input_descriptor)
        .context("create input operand")?;
    let norm_constants = (norm_epsilon, norm_exponent);

    let relu0 = conv_norm_relu(builder, input, &weights, 0, 4, None, norm_constants)?;
    let relu1 = conv_norm_relu(builder, relu0, &weights, 1, 1, Some([2, 2]), norm_constants)?;
    let relu2 = conv_norm_relu(builder, relu1, &weights, 2, 1, Some([2, 2]), norm_constants)?;

    let add4 = conv_norm(builder, relu2, &weights, 3, 1, norm_constants)?;
    let relu3 = builder.relu(add4).context("relu3")?;
    let add4 = conv_norm(builder, relu3, &weights, 4, 1, norm_constants)?;
    let add5 = builder.add(relu2, add4).context("add5")?;

    let add6 = conv_norm(builder, add5, &weights, 5, 1, norm_constants)?;
    let relu4 = builder.relu(add6).context("relu4")?;
    let add7 = conv_norm(builder, relu4, &weights, 6, 1, norm_constants)?;
    let add8 = builder.add(add5, add7).context("add8")?;

    let add9 = conv_norm(builder, add8, &weights, 7, 1, norm_constants)?;
    let relu5 = builder.relu(add9).context("relu5")?;
    let add10 = conv_norm(builder, relu5, &weights, 8, 1, norm_constants)?;
    let add11 = builder.add(add8, add10).context("add11")?;

    let add12 = conv_norm(builder, add11, &weights, 9, 1, norm_constants)?;
    let relu6 = builder.relu(add12).context("relu6")?;
    let add13 = conv_norm(builder, relu6, &weights, 10, 1, norm_constants)?;
    let add14 = builder.add(add11, add13).context("add14")?;

    let add15 = conv_norm(builder, add14, &weights, 11, 1, norm_constants)?;
    let relu7 = builder.relu(add15).context("relu7")?;
    let add16 = conv_norm(builder, relu7, &weights, 12, 1, norm_constants)?;
    let add17 = builder.add(add14, add16).context("add17")?;

    let relu8 =
        conv_transpose_norm_relu(builder, add17, &weights, 0, 13, [270, 270], norm_constants)?;
    let relu9 =
        conv_transpose_norm_relu(builder, relu8, &weights, 1, 14, [540, 540], norm_constants)?;

    let add20 = conv_norm_with_indices(builder, relu9, &weights, 13, 15, 4, norm_constants)?;
    let tanh = builder.tanh(add20).context("tanh_output")?;
    let scaled = builder.mul(tanh, output_scale).context("scale_output")?;
    builder.add(scaled, output_offset).context("offset_output")
}

fn image_to_tensor(image: &image::RgbImage) -> Vec<f32> {
    let resized = image::imageops::resize(
        image,
        WIDTH as u32,
        HEIGHT as u32,
        image::imageops::FilterType::Triangle,
    );
    let plane = HEIGHT * WIDTH;
    let mut tensor = vec![0.0f32; IMAGE_ELEMENT_COUNT];
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let pixel = resized.get_pixel(x as u32, y as u32);
            let idx = y * WIDTH + x;
            for channel in 0..CHANNELS {
                tensor[channel * plane + idx] = pixel[channel] as f32;
            }
        }
    }
    tensor
}

fn load_input_image(path: &Path) -> Result<Vec<f32>> {
    let image = image::open(path)
        .with_context(|| format!("decode input image {}", path.display()))?
        .to_rgb8();
    Ok(image_to_tensor(&image))
}

fn save_output_image(path: &PathBuf, output: &[f32]) -> Result<()> {
    ensure!(
        output.len() == IMAGE_ELEMENT_COUNT,
        "output tensor length {}, expected {}",
        output.len(),
        IMAGE_ELEMENT_COUNT
    );
    let plane = HEIGHT * WIDTH;
    let mut image = image::RgbImage::new(WIDTH as u32, HEIGHT as u32);
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let idx = y * WIDTH + x;
            let pixel = image::Rgb([
                output[idx].round().clamp(0.0, 255.0) as u8,
                output[plane + idx].round().clamp(0.0, 255.0) as u8,
                output[2 * plane + idx].round().clamp(0.0, 255.0) as u8,
            ]);
            image.put_pixel(x as u32, y as u32, pixel);
        }
    }
    image
        .save(path)
        .with_context(|| format!("write output image {}", path.display()))
}

fn run_inferences(
    context: &mut MLContext<'_>,
    graph: &mut rustnn::mlcontext::MLGraph<'_>,
    input_data: &[f32],
    num_inferences: usize,
    pipelined: bool,
) -> Result<Vec<f32>> {
    let mut input_descriptor =
        MLTensorDescriptor::new(MLOperandDataType::Float32, INPUT_SHAPE.to_vec());
    input_descriptor.set_writable(true);
    let mut output_descriptor =
        MLTensorDescriptor::new(MLOperandDataType::Float32, OUTPUT_SHAPE.to_vec());
    output_descriptor.set_readable(true);

    let tensor_count = if pipelined { PIPELINE_DEPTH } else { 1 };
    let mut input_tensors = Vec::with_capacity(tensor_count);
    let mut output_tensors = Vec::with_capacity(tensor_count);
    for _ in 0..tensor_count {
        input_tensors.push(
            context
                .create_tensor(&input_descriptor)
                .context("create input tensor")?,
        );
        output_tensors.push(
            context
                .create_tensor(&output_descriptor)
                .context("create output tensor")?,
        );
    }

    let mut output_data = vec![0.0f32; IMAGE_ELEMENT_COUNT];
    let mut starts = vec![None; tensor_count];
    let mut next_to_read = 0;

    for i in 0..num_inferences {
        let slot = i % tensor_count;
        context
            .write_tensor(&input_tensors[slot], input_data)
            .context("write input tensor")?;
        starts[slot] = Some(Instant::now());
        let inputs = HashMap::from([("input", &input_tensors[slot])]);
        let outputs = HashMap::from([("output", &output_tensors[slot])]);
        context
            .dispatch(graph, &inputs, &outputs)
            .context("dispatch graph")?;

        if i + 1 >= tensor_count {
            let completed_slot = next_to_read % tensor_count;
            context
                .read_tensor(&output_tensors[completed_slot], &mut output_data)
                .context("read output tensor")?;
            next_to_read += 1;
        }
    }

    while next_to_read < num_inferences {
        let completed_slot = next_to_read % tensor_count;
        context
            .read_tensor(&output_tensors[completed_slot], &mut output_data)
            .context("drain output tensor")?;
        println!(
            "inference {} took {:?}",
            next_to_read + 1,
            starts[completed_slot]
                .take()
                .expect("inference start time must be recorded")
                .elapsed()
        );
        next_to_read += 1;
    }

    Ok(output_data)
}

async fn download_weight(
    client: &reqwest::Client,
    builder: &Mutex<MLGraphBuilder<'_, '_>>,
    base_url: &str,
    spec: WeightSpec,
) -> Result<LoadedWeight> {
    let url = format!("{}/{}", base_url.trim_end_matches('/'), spec.file_name);
    let response = client
        .get(&url)
        .send()
        .await
        .with_context(|| format!("download `{}` from {url}", spec.name))?
        .error_for_status()
        .with_context(|| format!("download `{}` from {url}", spec.name))?;
    let bytes = response
        .bytes()
        .await
        .with_context(|| format!("read `{}` response body", spec.name))?;
    let array = parse_npy(&bytes).with_context(|| format!("parse `{}`", spec.file_name))?;

    let mut builder = builder.lock().await;
    add_npy_constant(&mut builder, spec, array)
}

async fn load_weights(
    client: &reqwest::Client,
    builder: &Mutex<MLGraphBuilder<'_, '_>>,
    base_url: &str,
    sequential_loading: bool,
) -> Result<Vec<LoadedWeight>> {
    let futures = WEIGHTS
        .iter()
        .copied()
        .map(|spec| download_weight(client, builder, base_url, spec));
    let results = if sequential_loading {
        let mut results = Vec::new();
        for fut in futures {
            results.push(fut.await);
        }
        results
    } else {
        join_all(futures).await
    };

    let mut loaded = Vec::with_capacity(WEIGHTS.len());
    let mut errors = Vec::new();
    for result in results {
        match result {
            Ok(weight) => {
                println!(
                    "loaded {:<22} {:?} {:?} ({} bytes)",
                    weight.name, weight.data_type, weight.shape, weight.byte_len
                );
                loaded.push(weight);
            }
            Err(error) => errors.push(format!("{error:#}")),
        }
    }

    if !errors.is_empty() {
        bail!(
            "failed to load {} Fast Style Transfer weight tensors:\n{}",
            errors.len(),
            errors.join("\n")
        );
    }

    Ok(loaded)
}

#[tokio::main]
async fn main() -> Result<()> {
    pretty_env_logger::init();
    let args = Args::parse();
    ensure!(args.num_inferences >= 1, "--num-inferences must be >= 1");

    let input = &args.input;
    println!("Input image: {}", input.display());
    println!("Output image: {}", args.output.display());
    println!("Requested inferences: {}", args.num_inferences);
    println!(
        "Inference tensor buffering: {}",
        if args.disable_pipelining {
            "disabled"
        } else {
            "pipeline depth 3"
        }
    );
    println!("Model: {} ({})", args.model.id(), args.model.title());
    println!("Input shape: {:?}", INPUT_SHAPE);
    println!("Output shape: {:?}", OUTPUT_SHAPE);

    let options = MLContextOptions::new(MLPowerPreference::HighPerformance, true);
    let mut context = MLContext::create(&options).context(
        "create MLContext; run this example with an enabled RustNN backend feature enabled and make sure the runtime libraries can be found",
    )?;
    let builder = Mutex::new(MLGraphBuilder::new(&mut context).context("create MLGraphBuilder")?);
    let client = reqwest::Client::new();
    let model_base_url = format!(
        "{}/{}",
        args.weights_base_url.trim_end_matches('/'),
        args.model.id()
    );

    let loaded = load_weights(&client, &builder, &model_base_url, args.sequential_weights).await?;
    let total_bytes: usize = loaded.iter().map(|weight| weight.byte_len).sum();

    println!(
        "Loaded {} Fast Style Transfer weight tensors ({} bytes total).",
        loaded.len(),
        total_bytes
    );

    println!("Building Fast Style Transfer graph...");
    let mut graph = {
        let mut builder = builder.lock().await;
        let output_operand = build_fast_style_transfer_graph(&mut builder, &loaded)?;
        let inferred_output_shape = builder
            .rustnn_operand_shape(output_operand)
            .context("infer output shape")?;
        ensure!(
            inferred_output_shape == OUTPUT_SHAPE,
            "inferred output shape {:?}, expected {:?}",
            inferred_output_shape,
            OUTPUT_SHAPE
        );
        let outputs = HashMap::from([("output", output_operand)]);
        builder.build(&outputs).context("build MLGraph")?
    };

    let input_data = load_input_image(input)?;
    let output_data = run_inferences(
        &mut context,
        &mut graph,
        &input_data,
        args.num_inferences,
        !args.disable_pipelining,
    )?;
    save_output_image(&args.output, &output_data)?;
    println!("Wrote output image: {}", args.output.display());

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_npy() -> Vec<u8> {
        let header = "{'descr': '<f4', 'fortran_order': False, 'shape': (2, 2), }";
        let base_len = 10 + header.len() + 1;
        let padding = (16 - (base_len % 16)) % 16;
        let mut full_header = String::from(header);
        full_header.extend(std::iter::repeat_n(' ', padding));
        full_header.push('\n');

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"\x93NUMPY");
        bytes.push(1);
        bytes.push(0);
        bytes.extend_from_slice(&(full_header.len() as u16).to_le_bytes());
        bytes.extend_from_slice(full_header.as_bytes());
        for value in [1.0f32, 2.0, 3.0, 4.0] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }

    #[test]
    fn parses_basic_float32_npy() {
        let array = parse_npy(&sample_npy()).unwrap();
        assert_eq!(array.data_type, NpyDataType::Float32);
        assert_eq!(array.shape, vec![2, 2]);
        assert_eq!(f32_values(&array.data), vec![1.0, 2.0, 3.0, 4.0]);
    }
}
