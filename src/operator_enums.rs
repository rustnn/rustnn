//! Enumerations of the WebNN IDL (`MLOperandDataType`, layouts, rounding, padding modes).

use crate::{DataType, error::GraphBuilderError};
use serde::{Deserialize, Serialize};
use strum::{AsRefStr, EnumString, IntoStaticStr};

/// Operand data type of the WebNN API. <https://www.w3.org/TR/webnn/#enumdef-mloperanddatatype>
///
/// `int4` and `uint4` are rustnn extensions that follow the WPT test data.
#[derive(
    Default,
    Clone,
    Copy,
    Debug,
    PartialEq,
    Eq,
    Deserialize,
    Serialize,
    Hash,
    EnumString,
    AsRefStr,
    IntoStaticStr,
)]
#[serde(rename_all = "kebab-case")]
#[strum(serialize_all = "kebab-case")]
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
#[derive(
    Default,
    Clone,
    Copy,
    Debug,
    PartialEq,
    Eq,
    Deserialize,
    Serialize,
    EnumString,
    AsRefStr,
    IntoStaticStr,
)]
#[serde(rename_all = "kebab-case")]
#[strum(serialize_all = "kebab-case")]
pub enum MLLstmWeightLayout {
    #[default]
    /// Input, output, forget, cell gate order.
    Iofg,
    /// Input, forget, cell, output gate order.
    Ifgo,
}

/// Rounding of pooling output sizes. <https://www.w3.org/TR/webnn/#enumdef-mlroundingtype>
#[derive(
    Default,
    Clone,
    Copy,
    Debug,
    PartialEq,
    Eq,
    Deserialize,
    Serialize,
    EnumString,
    AsRefStr,
    IntoStaticStr,
)]
#[serde(rename_all = "kebab-case")]
#[strum(serialize_all = "kebab-case")]
pub enum MLRoundingType {
    #[default]
    /// Round the output size down.
    Floor,
    /// Round the output size up.
    Ceil,
}

/// Interpolation of `resample2d`. <https://www.w3.org/TR/webnn/#enumdef-mlinterpolationmode>
#[derive(
    Default,
    Clone,
    Copy,
    Debug,
    PartialEq,
    Eq,
    Deserialize,
    Serialize,
    EnumString,
    AsRefStr,
    IntoStaticStr,
)]
#[serde(rename_all = "kebab-case")]
#[strum(serialize_all = "kebab-case")]
pub enum MLInterpolationMode {
    #[default]
    /// Nearest-neighbor sampling.
    NearestNeighbor,
    /// Bilinear interpolation.
    Linear,
}

/// Processing direction of `gru` and `lstm`. <https://www.w3.org/TR/webnn/#enumdef-mlrecurrentnetworkdirection>
#[derive(
    Default,
    Clone,
    Copy,
    Debug,
    PartialEq,
    Eq,
    Deserialize,
    Serialize,
    EnumString,
    AsRefStr,
    IntoStaticStr,
)]
#[serde(rename_all = "kebab-case")]
#[strum(serialize_all = "kebab-case")]
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
#[derive(
    Default,
    Clone,
    Copy,
    Debug,
    PartialEq,
    Eq,
    Hash,
    Deserialize,
    Serialize,
    EnumString,
    AsRefStr,
    IntoStaticStr,
)]
#[serde(rename_all = "kebab-case")]
#[strum(serialize_all = "kebab-case")]
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
#[derive(
    Default,
    Clone,
    Copy,
    Debug,
    PartialEq,
    Eq,
    Hash,
    Deserialize,
    Serialize,
    EnumString,
    AsRefStr,
    IntoStaticStr,
)]
#[serde(rename_all = "kebab-case")]
#[strum(serialize_all = "kebab-case")]
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
#[derive(
    Default,
    Clone,
    Copy,
    Debug,
    PartialEq,
    Eq,
    Deserialize,
    Serialize,
    EnumString,
    AsRefStr,
    IntoStaticStr,
)]
#[serde(rename_all = "kebab-case")]
#[strum(serialize_all = "kebab-case")]
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
#[derive(
    Default,
    Clone,
    Copy,
    Debug,
    PartialEq,
    Eq,
    Deserialize,
    Serialize,
    EnumString,
    AsRefStr,
    IntoStaticStr,
)]
#[serde(rename_all = "kebab-case")]
#[strum(serialize_all = "kebab-case")]
pub enum MLGruWeightLayout {
    #[default]
    /// Update, reset, new gate order.
    Zrn,
    /// Reset, update, new gate order.
    Rzn,
}

/// Input layout of convolution, pooling and normalization. <https://www.w3.org/TR/webnn/#enumdef-mlinputoperandlayout>
#[derive(
    Default,
    Clone,
    Copy,
    Debug,
    PartialEq,
    Eq,
    Hash,
    Deserialize,
    Serialize,
    EnumString,
    AsRefStr,
    IntoStaticStr,
)]
#[serde(rename_all = "kebab-case")]
#[strum(serialize_all = "kebab-case")]
pub enum MLInputOperandLayout {
    #[default]
    /// Batch, channels, height, width.
    Nchw,
    /// Batch, height, width, channels.
    Nhwc,
}

/// Padding mode of `pad`. <https://www.w3.org/TR/webnn/#enumdef-mlpaddingmode>
#[derive(
    Default,
    Clone,
    Copy,
    Debug,
    PartialEq,
    Eq,
    Hash,
    Deserialize,
    Serialize,
    EnumString,
    AsRefStr,
    IntoStaticStr,
)]
#[serde(rename_all = "kebab-case")]
#[strum(serialize_all = "kebab-case")]
pub enum MLPaddingMode {
    #[default]
    /// Fill with a constant value.
    Constant,
    /// Repeat the edge value.
    Edge,
    /// Mirror the values next to the edge.
    Reflection,
}

// Stable WebNN JSON / IDL string forms for converters.

impl MLOperandDataType {
    /// The specification spelling, as used in JSON and by the converters.
    pub fn as_str(self) -> &'static str {
        self.into()
    }
}

impl MLLstmWeightLayout {
    /// The specification spelling, as used in JSON and by the converters.
    pub fn as_str(self) -> &'static str {
        self.into()
    }
}

impl MLRoundingType {
    /// The specification spelling, as used in JSON and by the converters.
    pub fn as_str(self) -> &'static str {
        self.into()
    }
}

impl MLInterpolationMode {
    /// The specification spelling, as used in JSON and by the converters.
    pub fn as_str(self) -> &'static str {
        self.into()
    }
}

impl MLRecurrentNetworkDirection {
    /// The specification spelling, as used in JSON and by the converters.
    pub fn as_str(self) -> &'static str {
        self.into()
    }
}

impl MLConv2dFilterOperandLayout {
    /// The specification spelling, as used in JSON and by the converters.
    pub fn as_str(self) -> &'static str {
        self.into()
    }
}

impl MLConvTranspose2dFilterOperandLayout {
    /// The specification spelling, as used in JSON and by the converters.
    pub fn as_str(self) -> &'static str {
        self.into()
    }
}

impl MLRecurrentNetworkActivation {
    /// The specification spelling, as used in JSON and by the converters.
    pub fn as_str(self) -> &'static str {
        self.into()
    }
}

impl MLGruWeightLayout {
    /// The specification spelling, as used in JSON and by the converters.
    pub fn as_str(self) -> &'static str {
        self.into()
    }
}

impl MLInputOperandLayout {
    /// The specification spelling, as used in JSON and by the converters.
    pub fn as_str(self) -> &'static str {
        self.into()
    }
}

impl MLPaddingMode {
    /// The specification spelling, as used in JSON and by the converters.
    pub fn as_str(self) -> &'static str {
        self.into()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn specification_string_conversions() {
        macro_rules! check_strings {
            ($enum:ident { $($variant:ident => $spelling:literal),+ $(,)? }) => {
                $(
                    let value = $enum::$variant;
                    assert_eq!(value.as_ref(), $spelling);
                    assert_eq!(value.as_str(), $spelling);
                    assert_eq!($spelling.parse::<$enum>().unwrap(), value);
                    assert_eq!($enum::try_from($spelling).unwrap(), value);
                    assert_eq!(serde_json::to_value(value).unwrap(), $spelling);
                    assert_eq!(
                        serde_json::from_value::<$enum>(serde_json::json!($spelling)).unwrap(),
                        value
                    );
                    assert!($spelling.to_uppercase().parse::<$enum>().is_err());
                )+
                assert_eq!(
                    "invalid".parse::<$enum>(),
                    Err(strum::ParseError::VariantNotFound)
                );
                assert_eq!(
                    $enum::try_from("invalid"),
                    Err(strum::ParseError::VariantNotFound)
                );
            };
        }

        check_strings!(MLOperandDataType {
            Float32 => "float32", Float16 => "float16", Int32 => "int32",
            Uint32 => "uint32", Int64 => "int64", Uint64 => "uint64",
            Int8 => "int8", Uint8 => "uint8", Int4 => "int4", Uint4 => "uint4",
        });
        check_strings!(MLLstmWeightLayout { Iofg => "iofg", Ifgo => "ifgo" });
        check_strings!(MLRoundingType { Floor => "floor", Ceil => "ceil" });
        check_strings!(MLInterpolationMode {
            NearestNeighbor => "nearest-neighbor", Linear => "linear",
        });
        check_strings!(MLRecurrentNetworkDirection {
            Forward => "forward", Backward => "backward", Both => "both",
        });
        check_strings!(MLConv2dFilterOperandLayout {
            Oihw => "oihw", Hwio => "hwio", Ohwi => "ohwi", Ihwo => "ihwo",
        });
        check_strings!(MLConvTranspose2dFilterOperandLayout {
            Iohw => "iohw", Hwoi => "hwoi", Ohwi => "ohwi",
        });
        check_strings!(MLRecurrentNetworkActivation {
            Relu => "relu", Sigmoid => "sigmoid", Tanh => "tanh",
        });
        check_strings!(MLGruWeightLayout { Zrn => "zrn", Rzn => "rzn" });
        check_strings!(MLInputOperandLayout { Nchw => "nchw", Nhwc => "nhwc" });
        check_strings!(MLPaddingMode {
            Constant => "constant", Edge => "edge", Reflection => "reflection",
        });
    }

    #[test]
    fn conv_transpose2d_filter_layout_roundtrip() {
        for layout in [
            MLConvTranspose2dFilterOperandLayout::Iohw,
            MLConvTranspose2dFilterOperandLayout::Hwoi,
            MLConvTranspose2dFilterOperandLayout::Ohwi,
        ] {
            assert_eq!(
                MLConvTranspose2dFilterOperandLayout::try_from(layout.as_str()).unwrap(),
                layout
            );
        }
        assert!(MLConvTranspose2dFilterOperandLayout::try_from("hwio").is_err());
    }
}
