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
