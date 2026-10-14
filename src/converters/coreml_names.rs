//! Reversible escaping at the WebNN name / MIL identifier boundary.

use std::borrow::Cow;

pub(crate) const METADATA_KEY: &str = "rustnn.coreml.name_encoding";
pub(crate) const METADATA_VALUE: &str = "hex-v1";
const PREFIX: &str = "rustnn_escaped_";

// Matches the reserved names in coremltools' NameSanitizer. Escaping, rather
// than replacing punctuation with underscores, preserves distinct WebNN names.
const RESERVED: &[&str] = &[
    "any", "bool", "program", "func", "tensor", "list", "dict", "tuple", "true", "false", "string",
    "bf16", "fp16", "fp32", "fp64", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32",
    "uint64", "state", "function", "block", "return", "none",
];

pub(crate) fn encode(name: &str) -> Cow<'_, str> {
    let valid_start = name.starts_with(|c: char| c.is_ascii_alphabetic() || c == '_');
    if valid_start
        && name.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'_')
        && !RESERVED.contains(&name)
        && !name.starts_with(PREFIX)
    {
        return Cow::Borrowed(name);
    }
    const HEX: &[u8] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(PREFIX.len() + name.len() * 2);
    encoded.push_str(PREFIX);
    for byte in name.bytes() {
        encoded.push(HEX[usize::from(byte >> 4)] as char);
        encoded.push(HEX[usize::from(byte & 15)] as char);
    }
    Cow::Owned(encoded)
}

#[cfg(test)]
pub(crate) fn decode(name: &str) -> Cow<'_, str> {
    let Some(hex) = name.strip_prefix(PREFIX) else {
        return Cow::Borrowed(name);
    };
    let (pairs, remainder) = hex.as_bytes().as_chunks::<2>();
    if !remainder.is_empty() {
        return Cow::Borrowed(name);
    }
    let bytes: Option<Vec<_>> = pairs
        .iter()
        .map(|&[hi, lo]| Some(((hi as char).to_digit(16)? * 16 + (lo as char).to_digit(16)?) as u8))
        .collect();
    bytes
        .and_then(|bytes| String::from_utf8(bytes).ok())
        .map_or(Cow::Borrowed(name), Cow::Owned)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn escape_is_reversible_and_collision_free() {
        let mut names = RESERVED.to_vec();
        names.extend([
            "",
            "0cache",
            "a.b",
            "a-b",
            "a_b",
            "缓存",
            "state_workaround",
            "rustnn_escaped_7374617465",
        ]);
        let encoded: std::collections::HashSet<_> =
            names.iter().map(|name| encode(name).into_owned()).collect();
        assert_eq!(encoded.len(), names.len());
        for name in names {
            assert_eq!(decode(&encode(name)), name);
        }
        assert_eq!(encode("ordinary_input"), "ordinary_input");
    }

    #[test]
    fn malformed_encoded_names_are_not_decoded() {
        for name in [
            "rustnn_escaped_0",
            "rustnn_escaped_zz",
            "rustnn_escaped_ff",
            "ordinary",
        ] {
            assert_eq!(decode(name), name);
        }
    }
}
