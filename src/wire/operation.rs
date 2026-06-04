/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::operators::Operation;

/// Legacy graph JSON used a top-level `"label"` on each operation; WebNN places `label` on options.
/// When deserializing, merge top-level label into `attributes` if `attributes.label` is absent or empty.
fn merge_top_level_label_into_attributes(
    mut attributes: serde_json::Value,
    top_level_label: Option<String>,
) -> serde_json::Value {
    let Some(s) = top_level_label.filter(|x| !x.is_empty()) else {
        return attributes;
    };
    if attributes.is_null() {
        attributes = serde_json::json!({});
    }
    if let Some(obj) = attributes.as_object_mut() {
        let has_nonempty = obj
            .get("label")
            .and_then(|v| v.as_str())
            .map(|t| !t.is_empty())
            .unwrap_or(false);
        if !has_nonempty {
            obj.insert("label".to_string(), serde_json::Value::String(s));
        }
    }
    attributes
}

impl Serialize for Operation {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::SerializeStruct;
        let (op_type, input_operands, _) = self.to_legacy();
        let attributes = self.attributes_json_value();
        let outs = self.outputs();
        let output_operands: Vec<u32> = outs.to_vec();
        let output_operand = outs.first().copied();
        let mut st = serializer.serialize_struct("Operation", 5)?;
        st.serialize_field("type", &op_type)?;
        st.serialize_field("input_operands", &input_operands)?;
        st.serialize_field("attributes", &attributes)?;
        st.serialize_field("output_operand", &output_operand)?;
        st.serialize_field("output_operands", &output_operands)?;
        st.end()
    }
}

impl<'de> Deserialize<'de> for Operation {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct OperationHelper {
            #[serde(rename = "type")]
            op_type: String,
            #[serde(default)]
            input_operands: Vec<u32>,
            #[serde(default)]
            output_operand: Option<u32>,
            #[serde(default)]
            output_operands: Vec<u32>,
            #[serde(default)]
            attributes: serde_json::Value,
            #[serde(default)]
            label: Option<String>,
        }
        let h = OperationHelper::deserialize(deserializer)?;
        let attributes_value = merge_top_level_label_into_attributes(h.attributes, h.label);
        let output_ids: Vec<u32> = if !h.output_operands.is_empty() {
            h.output_operands.clone()
        } else if let Some(o) = h.output_operand {
            vec![o]
        } else {
            Vec::new()
        };
        Operation::from_json_attributes(
            &h.op_type,
            &h.input_operands,
            &output_ids,
            &attributes_value,
        )
        .ok_or_else(|| {
            serde::de::Error::custom(format!("unknown or invalid op_type: {}", h.op_type))
        })
    }
}
