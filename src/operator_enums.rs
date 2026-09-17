//! Enumerations of the WebNN IDL (`MLOperandDataType`, layouts, rounding, padding modes).
//!
//! Serde uses the kebab-case spelling of the specification (`"float32"`,
//! `"nearest-neighbor"`); `as_str` returns the same strings for converters. [`MLOperandDataType`]
//! converts to and from the graph-level [`DataType`].

use crate::{DataType, error::GraphBuilderError};
use serde::{Deserialize, Serialize};

/// Operand data type of the WebNN API. <https://www.w3.org/TR/webnn/#enumdef-mloperanddatatype>
///
/// `int4` and `uint4` are rustnn extensions that follow the WPT test data.
#[derive(Default, Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize, Hash)]
#[serde(rename_all = "kebab-case")]
pub enum MLOperandDataType {
    #[default]
    /// IEEE 754 binary32.
    Float32,
    /// IEEE 754 binary16.
    Float16,
    /// Signed 32-bit integer.
    Int32,
    /// Unsigned 32-bit integer.
    Uint32,
    /// Signed 64-bit integer.
    Int64,
    /// Unsigned 64-bit integer.
    Uint64,
    /// Signed 8-bit integer.
    Int8,
    /// Unsigned 8-bit integer; also the boolean type.
    Uint8,
    /// Signed 4-bit integer (rustnn extension).
    Int4,
    /// Unsigned 4-bit integer (rustnn extension).
    Uint4,
}

impl TryFrom<DataType> for MLOperandDataType {
    type Error = GraphBuilderError;
    fn try_from(value: DataType) -> Result<Self, Self::Error> {
        Ok(match value {
            DataType::Float32 => Self::Float32,
            DataType::Float16 => Self::Float16,
            DataType::Int32 => Self::Int32,
            DataType::Uint32 => Self::Uint32,
            DataType::Int64 => Self::Int64,
            DataType::Uint64 => Self::Uint64,
            DataType::Int8 => Self::Int8,
            DataType::Uint8 => Self::Uint8,
            DataType::Int4 => Self::Int4,
            DataType::Uint4 => Self::Uint4,
        })
    }
}

impl From<MLOperandDataType> for DataType {
    fn from(val: MLOperandDataType) -> Self {
        match val {
            MLOperandDataType::Float32 => DataType::Float32,
            MLOperandDataType::Float16 => DataType::Float16,
            MLOperandDataType::Int32 => DataType::Int32,
            MLOperandDataType::Uint32 => DataType::Uint32,
            MLOperandDataType::Int64 => DataType::Int64,
            MLOperandDataType::Uint64 => DataType::Uint64,
            MLOperandDataType::Int8 => DataType::Int8,
            MLOperandDataType::Uint8 => DataType::Uint8,
            MLOperandDataType::Int4 => DataType::Int4,
            MLOperandDataType::Uint4 => DataType::Uint4,
        }
    }
}

impl MLOperandDataType {
    /// Bits per element (4 for the packed 4-bit types).
    pub const fn rustnn_element_size_bits(self) -> usize {
        match self {
            MLOperandDataType::Float32 | MLOperandDataType::Int32 | MLOperandDataType::Uint32 => 32,
            MLOperandDataType::Float16 => 16,
            MLOperandDataType::Int64 | MLOperandDataType::Uint64 => 64,
            MLOperandDataType::Int8 | MLOperandDataType::Uint8 => 8,
            MLOperandDataType::Int4 | MLOperandDataType::Uint4 => 4,
        }
    }

    /// Host storage bytes for `elements` values, rounding 4-bit types up to whole bytes.
    pub const fn rustnn_storage_byte_length(self, elements: usize) -> usize {
        let bits = self.rustnn_element_size_bits();
        (bits * elements).div_ceil(8)
    }
}

/// Gate order of LSTM weights. <https://www.w3.org/TR/webnn/#enumdef-mllstmweightlayout>
#[derive(Default, Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum MLLstmWeightLayout {
    #[default]
    /// Input, output, forget, cell gate order.
    Iofg,
    /// Input, forget, cell, output gate order.
    Ifgo,
}

/// Rounding of pooling output sizes. <https://www.w3.org/TR/webnn/#enumdef-mlroundingtype>
#[derive(Default, Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum MLRoundingType {
    #[default]
    /// Round the output size down.
    Floor,
    /// Round the output size up.
    Ceil,
}

/// Interpolation of `resample2d`. <https://www.w3.org/TR/webnn/#enumdef-mlinterpolationmode>
#[derive(Default, Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum MLInterpolationMode {
    #[default]
    /// Nearest-neighbor sampling.
    NearestNeighbor,
    /// Bilinear interpolation.
    Linear,
}

/// Processing direction of `gru` and `lstm`. <https://www.w3.org/TR/webnn/#enumdef-mlrecurrentnetworkdirection>
#[derive(Default, Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum MLRecurrentNetworkDirection {
    #[default]
    /// Process the sequence from the first to the last step.
    Forward,
    /// Process the sequence from the last to the first step.
    Backward,
    /// Run both directions and stack the results.
    Both,
}

/// Filter layout of `conv2d`. <https://www.w3.org/TR/webnn/#enumdef-mlconv2dfilteroperandlayout>
#[derive(Default, Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum MLConv2dFilterOperandLayout {
    #[default]
    /// Output channels, input channels, height, width.
    Oihw,
    /// Height, width, input channels, output channels.
    Hwio,
    /// Output channels, height, width, input channels.
    Ohwi,
    /// Input channels, height, width, output channels.
    Ihwo,
}

/// Filter layout of `convTranspose2d`. <https://www.w3.org/TR/webnn/#enumdef-mlconvtranspose2dfilteroperandlayout>
#[derive(Default, Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum MLConvTranspose2dFilterOperandLayout {
    #[default]
    /// Input channels, output channels, height, width.
    Iohw,
    /// Height, width, output channels, input channels.
    Hwoi,
    /// Output channels, height, width, input channels.
    Ohwi,
}

/// Gate activation of the recurrent operations. <https://www.w3.org/TR/webnn/#enumdef-mlrecurrentnetworkactivation>
#[derive(Default, Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum MLRecurrentNetworkActivation {
    #[default]
    /// Rectified linear unit.
    Relu,
    /// Logistic sigmoid.
    Sigmoid,
    /// Hyperbolic tangent.
    Tanh,
}

/// Gate order of GRU weights. <https://www.w3.org/TR/webnn/#enumdef-mlgruweightlayout>
#[derive(Default, Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum MLGruWeightLayout {
    #[default]
    /// Update, reset, new gate order.
    Zrn,
    /// Reset, update, new gate order.
    Rzn,
}

/// Input layout of convolution, pooling and normalization. <https://www.w3.org/TR/webnn/#enumdef-mlinputoperandlayout>
#[derive(Default, Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum MLInputOperandLayout {
    #[default]
    /// Batch, channels, height, width.
    Nchw,
    /// Batch, height, width, channels.
    Nhwc,
}

/// Padding mode of `pad`. <https://www.w3.org/TR/webnn/#enumdef-mlpaddingmode>
#[derive(Default, Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum MLPaddingMode {
    #[default]
    /// Fill with a constant value.
    Constant,
    /// Repeat the edge value.
    Edge,
    /// Mirror the values next to the edge.
    Reflection,
}

// ---------------------------------------------------------------------------
// Stable WebNN JSON / IDL string forms (kebab-case / lowercase) for converters
// ---------------------------------------------------------------------------

impl MLOperandDataType {
    /// The specification spelling, as used in JSON and by the converters.
    pub fn as_str(self) -> &'static str {
        match self {
            MLOperandDataType::Float32 => "float32",
            MLOperandDataType::Float16 => "float16",
            MLOperandDataType::Int32 => "int32",
            MLOperandDataType::Uint32 => "uint32",
            MLOperandDataType::Int64 => "int64",
            MLOperandDataType::Uint64 => "uint64",
            MLOperandDataType::Int8 => "int8",
            MLOperandDataType::Uint8 => "uint8",
            MLOperandDataType::Int4 => "int4",
            MLOperandDataType::Uint4 => "uint4",
        }
    }
}

impl MLRoundingType {
    /// The specification spelling, as used in JSON and by the converters.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Floor => "floor",
            Self::Ceil => "ceil",
        }
    }
}

impl MLInterpolationMode {
    /// The specification spelling, as used in JSON and by the converters.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::NearestNeighbor => "nearest-neighbor",
            Self::Linear => "linear",
        }
    }
}

impl MLRecurrentNetworkDirection {
    /// The specification spelling, as used in JSON and by the converters.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Forward => "forward",
            Self::Backward => "backward",
            Self::Both => "both",
        }
    }
}

impl MLConv2dFilterOperandLayout {
    /// The specification spelling, as used in JSON and by the converters.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Oihw => "oihw",
            Self::Hwio => "hwio",
            Self::Ohwi => "ohwi",
            Self::Ihwo => "ihwo",
        }
    }
}

impl MLConvTranspose2dFilterOperandLayout {
    /// The specification spelling, as used in JSON and by the converters.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Iohw => "iohw",
            Self::Hwoi => "hwoi",
            Self::Ohwi => "ohwi",
        }
    }
}

impl MLRecurrentNetworkActivation {
    /// The specification spelling, as used in JSON and by the converters.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Relu => "relu",
            Self::Sigmoid => "sigmoid",
            Self::Tanh => "tanh",
        }
    }
}

impl MLGruWeightLayout {
    /// The specification spelling, as used in JSON and by the converters.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Zrn => "zrn",
            Self::Rzn => "rzn",
        }
    }
}

impl MLInputOperandLayout {
    /// The specification spelling, as used in JSON and by the converters.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Nchw => "nchw",
            Self::Nhwc => "nhwc",
        }
    }
}

impl MLPaddingMode {
    /// The specification spelling, as used in JSON and by the converters.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Constant => "constant",
            Self::Edge => "edge",
            Self::Reflection => "reflection",
        }
    }
}
