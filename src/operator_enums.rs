use serde::{Deserialize, Serialize};

#[derive(Default, Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum MlOperandDataType {
    #[default]
    Float32,
    Float16,
    Int32,
    Uint32,
    Int64,
    Uint64,
    Int8,
    Uint8,
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
