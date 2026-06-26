use crate::{DataType, error::GraphBuilderError};
use serde::{Deserialize, Serialize};

#[derive(Default, Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize, Hash)]
#[serde(rename_all = "kebab-case")]
pub enum MLOperandDataType {
    #[default]
    Float32,
    Float16,
    Int32,
    Uint32,
    Int64,
    Uint64,
    Int8,
    Uint8,
    Int4,
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
    pub const fn rustnn_element_size_bits(self) -> usize {
        match self {
            MLOperandDataType::Float32 | MLOperandDataType::Int32 | MLOperandDataType::Uint32 => 32,
            MLOperandDataType::Float16 => 16,
            MLOperandDataType::Int64 | MLOperandDataType::Uint64 => 64,
            MLOperandDataType::Int8 | MLOperandDataType::Uint8 => 8,
            MLOperandDataType::Int4 | MLOperandDataType::Uint4 => 4,
        }
    }

    pub const fn rustnn_storage_byte_length(self, elements: usize) -> usize {
        let bits = self.rustnn_element_size_bits();
        (bits * elements).div_ceil(8)
    }
}

#[derive(Default, Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum MLLstmWeightLayout {
    #[default]
    Iofg,
    Ifgo,
}

#[derive(Default, Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum MLRoundingType {
    #[default]
    Floor,
    Ceil,
}

#[derive(Default, Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum MLInterpolationMode {
    #[default]
    NearestNeighbor,
    Linear,
}

#[derive(Default, Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum MLRecurrentNetworkDirection {
    #[default]
    Forward,
    Backward,
    Both,
}

#[derive(Default, Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum MLConv2dFilterOperandLayout {
    #[default]
    Oihw,
    Hwio,
    Ohwi,
    Ihwo,
}

#[derive(Default, Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum MLConvTranspose2dFilterOperandLayout {
    #[default]
    Iohw,
    Hwoi,
    Ohwi,
}

#[derive(Default, Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum MLRecurrentNetworkActivation {
    #[default]
    Relu,
    Sigmoid,
    Tanh,
}

#[derive(Default, Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum MLGruWeightLayout {
    #[default]
    Zrn,
    Rzn,
}

#[derive(Default, Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum MLInputOperandLayout {
    #[default]
    Nchw,
    Nhwc,
}

#[derive(Default, Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum MLPaddingMode {
    #[default]
    Constant,
    Edge,
    Reflection,
}

// ---------------------------------------------------------------------------
// Stable WebNN JSON / IDL string forms (kebab-case / lowercase) for converters
// ---------------------------------------------------------------------------

impl MLOperandDataType {
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
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Floor => "floor",
            Self::Ceil => "ceil",
        }
    }
}

impl MLInterpolationMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::NearestNeighbor => "nearest-neighbor",
            Self::Linear => "linear",
        }
    }
}

impl MLRecurrentNetworkDirection {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Forward => "forward",
            Self::Backward => "backward",
            Self::Both => "both",
        }
    }
}

impl MLConv2dFilterOperandLayout {
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
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Iohw => "iohw",
            Self::Hwoi => "hwoi",
            Self::Ohwi => "ohwi",
        }
    }
}

impl MLRecurrentNetworkActivation {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Relu => "relu",
            Self::Sigmoid => "sigmoid",
            Self::Tanh => "tanh",
        }
    }
}

impl MLGruWeightLayout {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Zrn => "zrn",
            Self::Rzn => "rzn",
        }
    }
}

impl MLInputOperandLayout {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Nchw => "nchw",
            Self::Nhwc => "nhwc",
        }
    }
}

impl MLPaddingMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Constant => "constant",
            Self::Edge => "edge",
            Self::Reflection => "reflection",
        }
    }
}
