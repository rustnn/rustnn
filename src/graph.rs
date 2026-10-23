//! Backend-agnostic graph model: [`GraphInfo`], [`Operand`], [`OperandDescriptor`] and
//! [`DataType`].
//!
//! Operands are referenced by their index in [`GraphInfo::operands`]; operations are
//! [`crate::operators::Operation`] variants with named operand fields. Constant data is kept
//! in [`GraphInfo::constant_operand_ids_to_handles`] as raw little-endian bytes (4-bit types
//! are nibble-packed, see [`pack_int4`]). Dynamic shapes use [`Dimension::Dynamic`] with an
//! upper bound and require the `dynamic-inputs` feature at runtime.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use serde_with::{base64::Base64, serde_as};

use crate::error::GraphError;
use std::hash::{Hash, Hasher};

use crate::operator_options::{MLDimension, MLDynamicDimension};
use crate::operators::Operation;

/// A dimension whose size is only known at dispatch time, bounded by `max_size`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[serde(rename_all = "camelCase")]
pub struct DynamicDimension {
    /// Dimensions with the same name must have the same size at dispatch time.
    pub name: String,
    /// Upper bound; storage is allocated for this size.
    pub max_size: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[serde(untagged)]
/// One entry of an operand shape, either static or bounded dynamic.
pub enum Dimension {
    /// Fixed size.
    Static(u32),
    /// Bounded dynamic size; requires the `dynamic-inputs` feature.
    Dynamic(DynamicDimension),
}

impl Dimension {
    /// The static size, or `max_size` for a dynamic dimension.
    pub fn get_static_or_max_size(&self) -> u32 {
        match self {
            Self::Static(value) => *value,
            Self::Dynamic(dimension) => dimension.max_size,
        }
    }
}

/// Wraps a static shape as [`Dimension::Static`] entries.
pub fn to_dimension_vector(shape: &[u32]) -> Vec<Dimension> {
    shape.iter().copied().map(Dimension::Static).collect()
}

/// Free-function form of [`Dimension::get_static_or_max_size`], handy in iterator chains.
pub fn get_static_or_max_size(dim: &Dimension) -> u32 {
    dim.get_static_or_max_size()
}

impl From<MLDimension> for Dimension {
    fn from(m: MLDimension) -> Self {
        match m {
            MLDimension::Static(n) => Dimension::Static(n),
            MLDimension::Dynamic(d) => Dimension::Dynamic(DynamicDimension {
                name: d.name,
                max_size: d.max_size,
            }),
        }
    }
}

impl From<MLDynamicDimension> for DynamicDimension {
    fn from(d: MLDynamicDimension) -> Self {
        DynamicDimension {
            name: d.name,
            max_size: d.max_size,
        }
    }
}

impl From<Dimension> for MLDimension {
    fn from(d: Dimension) -> Self {
        match d {
            Dimension::Static(n) => MLDimension::Static(n),
            Dimension::Dynamic(d) => MLDimension::Dynamic(MLDynamicDimension {
                name: d.name,
                max_size: d.max_size,
            }),
        }
    }
}

impl From<DynamicDimension> for MLDynamicDimension {
    fn from(d: DynamicDimension) -> Self {
        MLDynamicDimension {
            name: d.name,
            max_size: d.max_size,
        }
    }
}

/// Whether the crate was built with the `dynamic-inputs` feature.
pub fn dynamic_inputs_enabled() -> bool {
    cfg!(feature = "dynamic-inputs")
}

/// Element type of an operand in the graph model; converts to and from
/// [`crate::operator_enums::MLOperandDataType`]. Serialized in snake_case (`"float32"`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DataType {
    /// Signed 4-bit integer, two per byte.
    Int4,
    /// Unsigned 4-bit integer, two per byte.
    Uint4,
    /// IEEE 754 binary16.
    Float16,
    /// IEEE 754 binary32.
    Float32,
    /// Signed 32-bit integer.
    Int32,
    /// Unsigned 32-bit integer.
    Uint32,
    /// Signed 8-bit integer.
    Int8,
    /// Unsigned 8-bit integer; also carries boolean results.
    Uint8,
    /// Signed 64-bit integer.
    Int64,
    /// Unsigned 64-bit integer.
    Uint64,
}

impl DataType {
    /// Bits per element (4 for the packed 4-bit types).
    pub const fn bits_per_element(self) -> usize {
        match self {
            DataType::Int4 | DataType::Uint4 => 4,
            DataType::Int8 | DataType::Uint8 => 8,
            DataType::Float16 => 16,
            DataType::Float32 | DataType::Int32 | DataType::Uint32 => 32,
            DataType::Int64 | DataType::Uint64 => 64,
        }
    }

    /// Host storage bytes for `elements` values (always `ceil(elements * bits / 8)`).
    pub fn storage_byte_length(self, elements: usize) -> Option<usize> {
        let total_bits = elements.checked_mul(self.bits_per_element())?;
        Some(total_bits.div_ceil(8))
    }

    /// Byte size of one element when the type is whole-byte aligned (`bits % 8 == 0`).
    pub fn bytes_per_element(self) -> usize {
        let bits = self.bits_per_element();
        debug_assert!(
            bits.is_multiple_of(8),
            "bytes_per_element is only defined for byte-aligned types; use storage_byte_length for int4/uint4"
        );
        bits / 8
    }
}

/// Packed int4/uint4 layout: even logical indices in the low nibble, odd in the high (ONNX/WebNN
/// convention, i.e. element `2*i` occupies the least-significant 4 bits of byte `i`).
pub fn unpack_int4(data: &[u8], element_count: usize) -> Vec<i32> {
    let mut out = Vec::with_capacity(element_count);
    for i in 0..element_count {
        let byte = data[i / 2];
        let nibble = if i % 2 == 0 {
            byte & 0x0F
        } else {
            (byte >> 4) & 0x0F
        };
        out.push(if nibble >= 8 {
            nibble as i32 - 16
        } else {
            nibble as i32
        });
    }
    out
}

/// Packs signed 4-bit values (clamped to `-8..=7`) into nibbles; see [`unpack_int4`] for the layout.
pub fn pack_int4(values: &[i32]) -> Vec<u8> {
    let byte_len = DataType::Int4
        .storage_byte_length(values.len())
        .unwrap_or(0);
    let mut out = vec![0u8; byte_len];
    for (i, &v) in values.iter().enumerate() {
        let nibble = ((v.clamp(-8, 7) as i8) as u8) & 0x0F;
        if i % 2 == 0 {
            out[i / 2] = nibble;
        } else {
            out[i / 2] |= nibble << 4;
        }
    }
    out
}

/// Unpacks nibbles into unsigned 4-bit values; same layout as [`unpack_int4`].
pub fn unpack_uint4(data: &[u8], element_count: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(element_count);
    for i in 0..element_count {
        let byte = data[i / 2];
        let nibble = if i % 2 == 0 {
            byte & 0x0F
        } else {
            (byte >> 4) & 0x0F
        };
        out.push(nibble);
    }
    out
}

/// Packs unsigned 4-bit values (low nibble of each byte is used) into nibbles.
pub fn pack_uint4(values: &[u8]) -> Vec<u8> {
    let byte_len = DataType::Uint4
        .storage_byte_length(values.len())
        .unwrap_or(0);
    let mut out = vec![0u8; byte_len];
    for (i, &v) in values.iter().enumerate() {
        let nibble = v & 0x0F;
        if i % 2 == 0 {
            out[i / 2] = nibble;
        } else {
            out[i / 2] |= nibble << 4;
        }
    }
    out
}

/// [`pack_uint4`] for `i32` inputs, clamped to `0..=15`.
pub fn pack_uint4_from_i32(values: &[i32]) -> Vec<u8> {
    pack_uint4(
        &values
            .iter()
            .map(|&v| v.clamp(0, 15) as u8)
            .collect::<Vec<_>>(),
    )
}

/// Data type and shape of an operand in the graph model.
#[derive(Debug, Clone, Serialize, Deserialize, Hash)]
pub struct OperandDescriptor {
    /// Element type.
    pub data_type: DataType,
    /// Known dimensions, outermost first. An empty vector is a rank-0 scalar, never an unknown shape.
    pub shape: Vec<Dimension>,
    /// Layout permutation a converter still has to apply (internal bookkeeping, normally empty).
    #[serde(default)]
    pub pending_permutation: Vec<u32>,
}

impl OperandDescriptor {
    /// Whether any dimension is [`Dimension::Dynamic`].
    pub fn has_dynamic_dimensions(&self) -> bool {
        self.shape
            .iter()
            .any(|dim| matches!(dim, Dimension::Dynamic(_)))
    }

    /// The shape as plain sizes, or `None` if any dimension is dynamic.
    pub fn static_shape(&self) -> Option<Vec<u32>> {
        let mut shape = Vec::with_capacity(self.shape.len());
        for dim in &self.shape {
            match dim {
                Dimension::Static(v) => shape.push(*v),
                Dimension::Dynamic(_) => return None,
            }
        }
        Some(shape)
    }

    /// The shape with dynamic dimensions replaced by their maximum size.
    pub fn static_or_max_shape(&self) -> Vec<u32> {
        self.shape.iter().map(get_static_or_max_size).collect()
    }

    /// Number of elements at the maximum shape; `None` on overflow.
    pub fn element_count(&self) -> Option<usize> {
        if self.shape.is_empty() {
            return Some(1);
        }
        let mut count = 1usize;
        for dim in &self.shape {
            let size = get_static_or_max_size(dim) as usize;
            count = count.checked_mul(size)?;
        }
        Some(count)
    }

    /// Storage bytes at the maximum shape (4-bit types packed); `None` on overflow.
    pub fn byte_length(&self) -> Option<usize> {
        let elements = self.element_count()?;
        self.data_type.storage_byte_length(elements)
    }
}

/// Role of an operand in the graph.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum OperandKind {
    /// Named graph input bound at dispatch.
    Input,
    /// Constant with data in [`GraphInfo::constant_operand_ids_to_handles`].
    Constant,
    /// Named graph output bound at dispatch.
    Output,
    /// Result of an operation that is consumed inside the graph.
    // optional operand type, at the moment not required in graphs, but useful for validation and
    // incremental shape inference
    Intermediate,
}

/// A graph operand: its role, descriptor and optional name.
#[derive(Debug, Clone, Serialize, Deserialize, Hash)]
pub struct Operand {
    /// Role in the graph.
    pub kind: OperandKind,
    /// Data type and shape.
    pub descriptor: OperandDescriptor,
    /// Binding name for inputs and outputs; label for other operands.
    #[serde(default)]
    pub name: Option<String>,
}

/// Resolves constant references to bytes for graph conversion.
pub trait WeightsContext<'context> {
    /// Resolves the reference for a constant operand.
    fn resolve(
        &mut self,
        constant_ref: &'context ConstantReference,
        descriptor: &OperandDescriptor,
        id: u32,
    ) -> crate::error::Result<&'context [u8]>;
}
/// Resolves inline constant data without performing I/O.
pub struct DefaultWeightsContext {}

impl<'context> WeightsContext<'context> for DefaultWeightsContext {
    fn resolve(
        &mut self,
        constant: &'context ConstantReference,
        _descriptor: &OperandDescriptor,
        id: u32,
    ) -> crate::error::Result<&'context [u8]> {
        match constant {
            ConstantReference::OwnedData(ConstantData { data, .. }) => Ok(data),
            ConstantReference::UrlRef(_)
            | ConstantReference::FileRef(_)
            | ConstantReference::OffsetRef(_)
            | ConstantReference::StringRef(_)
            | ConstantReference::IdRef(_)
            | ConstantReference::Missing => {
                Err(crate::error::GraphBuilderError::UnresolvedConstant {
                    id,
                    constant_ref: constant.clone(),
                }
                .into())
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Hash)]
/// An external tensor byte range addressed by URL.
pub struct UrlRef {
    /// Entity tag identifying the remote content.
    pub etag: String,
    /// Source URL.
    pub url: String,
    /// Byte offset into the source.
    pub offset: u64,
    /// Length of the byte range.
    pub len: u64,
    #[serde(default)]
    /// Optional source label.
    pub label: Option<String>,
}

#[serde_as]
#[derive(Debug, Clone, Serialize, Deserialize, Hash)]
/// Raw little-endian constant bytes, with nibble-packed 4-bit values.
pub struct ConstantData {
    /// The bytes; base64 in JSON.
    #[serde_as(as = "Base64")]
    pub data: Vec<u8>,
    /// Optional label carried into exported graphs.
    #[serde(default)]
    /// Optional source label.
    pub label: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Hash)]
/// A byte range in an external tensor file.
pub struct FileRef {
    /// File containing the tensor bytes.
    pub path: PathBuf,
    /// Byte offset into the source.
    pub offset: usize,
    /// Length of the byte range.
    pub len: usize,
    #[serde(default)]
    /// Optional source label.
    pub label: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Hash)]
/// A byte range in an externally supplied buffer.
pub struct OffsetRef {
    /// Byte offset into the source.
    pub offset: usize,
    /// Length of the byte range.
    pub len: usize,
    #[serde(default)]
    /// Optional source label.
    pub label: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Hash)]
/// A tensor handle in the graph context.
pub struct IdRef {
    /// Context-local tensor identifier.
    pub id: u64,
    #[serde(default)]
    /// Optional source label.
    pub label: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Hash)]
/// A named tensor to resolve from an external weights file.
pub struct StringRef {
    #[serde(default)]
    /// Optional source label.
    pub label: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Hash)]
#[serde(untagged)]
/// Inline constant data or a reference to externally managed tensor storage.
pub enum ConstantReference {
    /// Bytes owned by the graph.
    OwnedData(ConstantData),
    /// Remote tensor bytes.
    UrlRef(UrlRef),
    /// Tensor bytes in a local file.
    FileRef(FileRef),
    /// Tensor bytes in an external buffer.
    OffsetRef(OffsetRef),
    /// Named external tensor awaiting resolution.
    StringRef(StringRef),
    /// Tensor allocated by the graph context.
    IdRef(IdRef),
    /// No data source has been supplied.
    Missing,
}

impl From<ConstantData> for ConstantReference {
    fn from(data: ConstantData) -> Self {
        Self::OwnedData(data)
    }
}

impl ConstantReference {
    /// Returns the inline data, if present.
    pub fn as_owned_data(&self) -> Option<&ConstantData> {
        match self {
            Self::OwnedData(data) => Some(data),
            _ => None,
        }
    }

    /// Returns the tensor handle reference, if present.
    pub fn as_id_ref(&self) -> Option<&IdRef> {
        if let Self::IdRef(v) = self {
            Some(v)
        } else {
            None
        }
    }

    /// Returns `true` if the constant reference is [`IdRef`].
    ///
    /// [`IdRef`]: ConstantReference::IdRef
    #[must_use]
    pub fn is_id_ref(&self) -> bool {
        matches!(self, Self::IdRef(..))
    }
}

#[derive(Debug, Serialize, Clone, Deserialize, Default)]
/// The complete backend-agnostic graph.
pub struct GraphInfo {
    /// All operands; operations refer to them by index.
    pub operands: Vec<Operand>,
    /// Indices of the [`OperandKind::Input`] operands, in declaration order.
    #[serde(default)]
    pub input_operands: Vec<u32>,
    /// Indices of the [`OperandKind::Output`] operands, sorted.
    #[serde(default)]
    pub output_operands: Vec<u32>,
    /// Operations in topological (recording) order.
    #[serde(default)]
    pub operations: Vec<Operation>,
    /// Constant data by operand index.
    #[serde(default)]
    pub constant_operand_ids_to_handles: HashMap<u32, ConstantReference>,
    /// Whether the graph carries quantized constants (affects the `.webnn` export).
    #[serde(default)]
    pub quantized: bool,
}

impl GraphInfo {
    /// The operand with index `id`, if it exists.
    pub fn constant_data(&self, id: u32) -> crate::error::Result<&[u8]> {
        let constant = self
            .constant_operand_ids_to_handles
            .get(&id)
            .ok_or(crate::error::GraphBuilderError::MissingConstantId { id })?;
        let descriptor = &self
            .operands
            .get(id as usize)
            .ok_or(crate::error::GraphBuilderError::MissingConstantId { id })?
            .descriptor;
        DefaultWeightsContext {}.resolve(constant, descriptor, id)
    }

    /// Resolves all [`ConstantReference::StringRef`] entries against a safetensors file.
    ///
    /// Only the safetensors length prefix and JSON header are read. Tensor offsets in the
    /// header are relative to the data section, so the resulting [`FileRef`] offsets include
    /// both the eight-byte length prefix and the header itself.
    pub fn resolve_safetensor_references(
        &mut self,
        path: impl AsRef<std::path::Path>,
    ) -> crate::error::Result<()> {
        use std::io::Read;

        const LENGTH_PREFIX_SIZE: usize = size_of::<u64>();
        const MAX_HEADER_SIZE: usize = 100_000_000;

        let string_refs: Vec<(u32, String)> = self
            .constant_operand_ids_to_handles
            .iter()
            .filter_map(|(&id, reference)| match reference {
                ConstantReference::StringRef(string_ref) => Some((id, string_ref.label.clone())),
                _ => None,
            })
            .collect();

        if string_refs.is_empty() {
            return Ok(());
        }

        let path = path.as_ref();
        let mut file = std::fs::File::open(path).map_err(|error| GraphError::io(path, error))?;
        let mut header_len_bytes = [0_u8; LENGTH_PREFIX_SIZE];
        file.read_exact(&mut header_len_bytes)
            .map_err(|error| GraphError::io(path, error))?;

        let header_len = usize::try_from(u64::from_le_bytes(header_len_bytes)).map_err(|_| {
            GraphError::ConversionFailed {
                format: "safetensors".to_string(),
                reason: "header length does not fit in usize".to_string(),
            }
        })?;
        if header_len > MAX_HEADER_SIZE {
            return Err(GraphError::ConversionFailed {
                format: "safetensors".to_string(),
                reason: format!(
                    "header length {header_len} exceeds the {MAX_HEADER_SIZE}-byte limit"
                ),
            }
            .into());
        }

        let mut header = vec![0_u8; header_len];
        file.read_exact(&mut header)
            .map_err(|error| GraphError::io(path, error))?;
        let metadata: safetensors::tensor::Metadata =
            serde_json::from_slice(&header).map_err(|error| GraphError::ConversionFailed {
                format: "safetensors".to_string(),
                reason: format!("invalid header JSON: {error}"),
            })?;
        let data_start = LENGTH_PREFIX_SIZE.checked_add(header_len).ok_or_else(|| {
            GraphError::ConversionFailed {
                format: "safetensors".to_string(),
                reason: "data section offset overflowed usize".to_string(),
            }
        })?;

        // Resolve every reference before mutating the graph, leaving it unchanged on error.
        let mut replacements = Vec::with_capacity(string_refs.len());
        for (id, label) in string_refs {
            let tensor = metadata
                .info(&label)
                .ok_or_else(|| GraphError::ConversionFailed {
                    format: "safetensors".to_string(),
                    reason: format!("tensor `{label}` is missing from the header"),
                })?;
            let (relative_start, relative_end) = tensor.data_offsets;
            let offset = data_start.checked_add(relative_start).ok_or_else(|| {
                GraphError::ConversionFailed {
                    format: "safetensors".to_string(),
                    reason: format!("tensor `{label}` file offset overflowed usize"),
                }
            })?;

            replacements.push((
                id,
                ConstantReference::FileRef(FileRef {
                    path: path.to_path_buf(),
                    offset,
                    len: relative_end - relative_start,
                    label: Some(label),
                }),
            ));
        }

        for (id, reference) in replacements {
            self.constant_operand_ids_to_handles.insert(id, reference);
        }

        Ok(())
    }

    /// Appends a constant operand backed by the supplied reference.
    pub fn add_constant_reference(
        &mut self,
        descriptor: OperandDescriptor,
        reference: ConstantReference,
        name: Option<String>,
    ) -> crate::error::Result<u32> {
        let id = self
            .operands
            .len()
            .try_into()
            .map_err(|_e| GraphError::TooManyOperands {
                count: self.operands.len(),
            })?;

        self.operands.push(Operand {
            kind: OperandKind::Constant,
            descriptor,
            name,
        });
        self.constant_operand_ids_to_handles.insert(id, reference);

        Ok(id)
    }

    /// Resolves a constant using a caller-supplied context.
    pub fn resolve_constant<'context>(
        &'context self,
        id: u32,
        context: &mut (impl WeightsContext<'context> + 'context),
    ) -> crate::error::Result<&'context [u8]> {
        let constant = self.constant_operand_ids_to_handles.get(&id);
        let Some(constant) = constant else {
            return Err(crate::error::GraphBuilderError::MissingConstantId { id }.into());
        };
        let Some(operand) = self.operands.get(id as usize) else {
            return Err(crate::error::GraphBuilderError::MissingConstantId { id }.into());
        };
        context.resolve(constant, &operand.descriptor, id)
    }

    /// Returns the operand with this index, if present.
    pub fn operand(&self, id: u32) -> Option<&Operand> {
        self.operands.get(id as usize)
    }

    /// Whether any operand has a dynamic dimension.
    pub fn has_dynamic_dimensions(&self) -> bool {
        self.operands
            .iter()
            .any(|operand| operand.descriptor.has_dynamic_dimensions())
    }

    /// Ensures `input_operands` / `output_operands` match operands tagged as graph I/O.
    pub fn validate_io_operand_lists(&self) -> Result<(), GraphError> {
        let mut derived_inputs = Vec::new();
        let mut derived_outputs = Vec::new();

        for (idx, operand) in self.operands.iter().enumerate() {
            match operand.kind {
                OperandKind::Input => derived_inputs.push(idx as u32),
                OperandKind::Output => derived_outputs.push(idx as u32),
                OperandKind::Constant | OperandKind::Intermediate => {}
            }
        }

        if derived_inputs != self.input_operands {
            return Err(GraphError::InputIdListMismatch {
                input_ids: self.input_operands.clone(),
                input_ids_in_operands: derived_inputs,
            });
        }
        if derived_outputs != self.output_operands {
            return Err(GraphError::OutputIdListMismatch {
                output_ids: self.output_operands.clone(),
                output_ids_in_operands: derived_outputs,
            });
        }

        Ok(())
    }
}

/// Which constant operands [`GraphInfo::hash_identifier`] includes in a cache key.
pub enum WeightsToHash<'a> {
    /// Topology only; weights are refitted after loading.
    None,
    /// Every constant.
    All,
    /// The listed constant operand indices.
    Some(&'a HashSet<u32>),
}

/// Named input/output operands for `MLGraph` dispatch.
pub type IoBindingMaps = (
    HashMap<String, OperandDescriptor>,
    HashMap<String, OperandDescriptor>,
);

impl GraphInfo {
    /// Named input/output operands for `MLGraph` dispatch, after list consistency checks.
    #[allow(clippy::type_complexity)]
    pub fn io_binding_maps(
        &self,
    ) -> Result<
        (
            HashMap<String, OperandDescriptor>,
            HashMap<String, OperandDescriptor>,
        ),
        GraphError,
    > {
        self.validate_io_operand_lists()?;

        let mut inputs = HashMap::new();
        for &id in &self.input_operands {
            let operand = self.operand(id).expect("validated above");
            let name = operand
                .name
                .as_ref()
                .ok_or(GraphError::MissingInputName { operand: id })?;
            if name.is_empty() {
                return Err(GraphError::MissingInputName { operand: id });
            }
            if inputs
                .insert(name.clone(), operand.descriptor.clone())
                .is_some()
            {
                return Err(GraphError::DuplicateInputName { name: name.clone() });
            }
        }

        let mut outputs = HashMap::new();
        for &id in &self.output_operands {
            let operand = self.operand(id).expect("validated above");
            let name = operand
                .name
                .as_ref()
                .ok_or(GraphError::MissingOutputName { operand: id })?;
            if name.is_empty() {
                return Err(GraphError::MissingOutputName { operand: id });
            }
            if outputs
                .insert(name.clone(), operand.descriptor.clone())
                .is_some()
            {
                return Err(GraphError::DuplicateOutputName { name: name.clone() });
            }
        }

        Ok((inputs, outputs))
    }

    /// Stable hash of the graph (operands, operations, selected weights) plus `suffix`, used as
    /// the key of the backend engine caches.
    pub fn hash_identifier<'a>(&self, suffix: &str, weights_to_hash: WeightsToHash<'a>) -> String {
        let mut hasher = seahash::SeaHasher::new();
        self.input_operands.hash(&mut hasher);
        self.output_operands.hash(&mut hasher);
        self.operands.hash(&mut hasher);
        self.operations.hash(&mut hasher);
        match weights_to_hash {
            WeightsToHash::None => (),
            WeightsToHash::All => {
                for (constant_id, _) in self
                    .operands
                    .iter()
                    .enumerate()
                    .filter(|(_id, op)| op.kind == OperandKind::Constant)
                {
                    self.constant_operand_ids_to_handles
                        .get(&(constant_id as u32))
                        .hash(&mut hasher);
                }
            }
            WeightsToHash::Some(items) => {
                let mut items: Vec<u32> = items.iter().copied().collect();
                items.sort_unstable();
                for constant_id in items.iter() {
                    self.constant_operand_ids_to_handles
                        .get(constant_id)
                        .hash(&mut hasher);
                }
            }
        }
        let reproduciable_hash_64bit = hasher.finish();
        format!(
            "{reproduciable_hash_64bit:x}_{}_{suffix}",
            self.operands.len()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::GraphError;

    fn write_safetensors_header(path: &std::path::Path, header: &serde_json::Value) -> usize {
        let header = serde_json::to_vec(header).unwrap();
        let mut contents = Vec::with_capacity(size_of::<u64>() + header.len());
        contents.extend_from_slice(&(header.len() as u64).to_le_bytes());
        contents.extend_from_slice(&header);
        std::fs::write(path, contents).unwrap();
        size_of::<u64>() + header.len()
    }

    #[test]
    fn resolve_safetensor_references_uses_header_offsets() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("weights.safetensors");
        let data_start = write_safetensors_header(
            &path,
            &serde_json::json!({
                "unused": {
                    "dtype": "F32",
                    "shape": [1],
                    "data_offsets": [0, 4]
                },
                "weight": {
                    "dtype": "F32",
                    "shape": [1],
                    "data_offsets": [4, 8]
                }
            }),
        );
        let mut graph = GraphInfo::default();
        graph.constant_operand_ids_to_handles.insert(
            4,
            ConstantReference::StringRef(StringRef {
                label: "weight".to_string(),
            }),
        );

        graph.resolve_safetensor_references(&path).unwrap();

        let ConstantReference::FileRef(file_ref) = &graph.constant_operand_ids_to_handles[&4]
        else {
            panic!("StringRef was not replaced with FileRef");
        };
        assert_eq!(file_ref.path, path);
        assert_eq!(file_ref.offset, data_start + 4);
        assert_eq!(file_ref.len, 4);
        assert_eq!(file_ref.label.as_deref(), Some("weight"));
    }

    #[test]
    fn resolve_safetensor_references_is_transactional() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("weights.safetensors");
        write_safetensors_header(
            &path,
            &serde_json::json!({
                "present": {
                    "dtype": "F32",
                    "shape": [1],
                    "data_offsets": [0, 4]
                }
            }),
        );
        let mut graph = GraphInfo::default();
        graph.constant_operand_ids_to_handles.insert(
            1,
            ConstantReference::StringRef(StringRef {
                label: "present".to_string(),
            }),
        );
        graph.constant_operand_ids_to_handles.insert(
            2,
            ConstantReference::StringRef(StringRef {
                label: "missing".to_string(),
            }),
        );

        assert!(graph.resolve_safetensor_references(&path).is_err());
        assert!(matches!(
            graph.constant_operand_ids_to_handles[&1],
            ConstantReference::StringRef(_)
        ));
        assert!(matches!(
            graph.constant_operand_ids_to_handles[&2],
            ConstantReference::StringRef(_)
        ));
    }

    #[test]
    fn test_data_type_bits_per_element() {
        assert_eq!(DataType::Int4.bits_per_element(), 4);
        assert_eq!(DataType::Uint4.bits_per_element(), 4);
        assert_eq!(DataType::Float16.bits_per_element(), 16);
        assert_eq!(DataType::Float32.bits_per_element(), 32);
        assert_eq!(DataType::Int32.bits_per_element(), 32);
        assert_eq!(DataType::Uint32.bits_per_element(), 32);
        assert_eq!(DataType::Int8.bits_per_element(), 8);
        assert_eq!(DataType::Uint8.bits_per_element(), 8);
        assert_eq!(DataType::Int64.bits_per_element(), 64);
        assert_eq!(DataType::Uint64.bits_per_element(), 64);
    }

    #[test]
    fn test_data_type_bytes_per_element() {
        assert_eq!(DataType::Float16.bytes_per_element(), 2);
        assert_eq!(DataType::Float32.bytes_per_element(), 4);
        assert_eq!(DataType::Int32.bytes_per_element(), 4);
        assert_eq!(DataType::Uint32.bytes_per_element(), 4);
        assert_eq!(DataType::Int8.bytes_per_element(), 1);
        assert_eq!(DataType::Uint8.bytes_per_element(), 1);
        assert_eq!(DataType::Int64.bytes_per_element(), 8);
        assert_eq!(DataType::Uint64.bytes_per_element(), 8);
    }

    #[test]
    fn test_data_type_storage_byte_length_int4() {
        assert_eq!(DataType::Int4.storage_byte_length(0), Some(0));
        assert_eq!(DataType::Int4.storage_byte_length(1), Some(1));
        assert_eq!(DataType::Int4.storage_byte_length(2), Some(1));
        assert_eq!(DataType::Int4.storage_byte_length(3), Some(2));
        assert_eq!(DataType::Int4.storage_byte_length(100), Some(50));
    }

    #[test]
    fn test_pack_unpack_int4() {
        let values = vec![-8_i32, 7, 0, -1];
        let packed = pack_int4(&values);
        // Low-nibble-first: byte0 = 0x8 | (0x7 << 4), byte1 = 0x0 | (0xF << 4).
        assert_eq!(packed, vec![0x78, 0xF0]);
        assert_eq!(unpack_int4(&packed, values.len()), values);
    }

    #[test]
    fn test_pack_unpack_uint4() {
        let values = vec![0_u8, 15, 7, 1];
        let packed = pack_uint4(&values);
        // Low-nibble-first: byte0 = 0x0 | (0xF << 4), byte1 = 0x7 | (0x1 << 4).
        assert_eq!(packed, vec![0xF0, 0x17]);
        assert_eq!(unpack_uint4(&packed, values.len()), values);
    }

    #[test]
    fn test_data_type_serialization() {
        assert_eq!(serde_json::to_string(&DataType::Int4).unwrap(), "\"int4\"");
        assert_eq!(
            serde_json::to_string(&DataType::Uint4).unwrap(),
            "\"uint4\""
        );
        assert_eq!(
            serde_json::to_string(&DataType::Float32).unwrap(),
            "\"float32\""
        );
    }

    #[test]
    fn test_data_type_deserialization() {
        assert_eq!(
            serde_json::from_str::<DataType>("\"int4\"").unwrap(),
            DataType::Int4
        );
        assert_eq!(
            serde_json::from_str::<DataType>("\"uint4\"").unwrap(),
            DataType::Uint4
        );
        assert_eq!(
            serde_json::from_str::<DataType>("\"float32\"").unwrap(),
            DataType::Float32
        );
    }

    #[test]
    fn test_operand_descriptor_element_count() {
        let desc = OperandDescriptor {
            data_type: DataType::Int4,
            shape: to_dimension_vector(&[2, 3, 4]),
            pending_permutation: vec![],
        };
        assert_eq!(desc.element_count(), Some(24));
    }

    #[test]
    fn test_operand_descriptor_byte_length_int4() {
        let desc = OperandDescriptor {
            data_type: DataType::Int4,
            shape: to_dimension_vector(&[10, 10]),
            pending_permutation: vec![],
        };
        assert_eq!(desc.byte_length(), Some(50));
    }

    #[test]
    fn test_operand_descriptor_byte_length_uint4() {
        let desc = OperandDescriptor {
            data_type: DataType::Uint4,
            shape: to_dimension_vector(&[8, 16]),
            pending_permutation: vec![],
        };
        assert_eq!(desc.byte_length(), Some(64));
    }

    #[test]
    fn test_operand_descriptor_byte_length_float32() {
        let desc = OperandDescriptor {
            data_type: DataType::Float32,
            shape: to_dimension_vector(&[4, 4]),
            pending_permutation: vec![],
        };
        assert_eq!(desc.byte_length(), Some(64));
    }

    #[test]
    fn operand_descriptor_requires_shape_but_accepts_explicit_scalar() {
        let scalar: OperandDescriptor =
            serde_json::from_str(r#"{"data_type":"float32","shape":[]}"#).unwrap();
        assert!(scalar.shape.is_empty());

        let missing = serde_json::from_str::<OperandDescriptor>(r#"{"data_type":"float32"}"#);
        assert!(missing.is_err());
    }

    #[test]
    fn test_graph_info_quantized_field_default() {
        let json =
            r#"{"operands": [], "input_operands": [], "output_operands": [], "operations": []}"#;
        let graph: GraphInfo = serde_json::from_str(json).unwrap();
        assert!(!graph.quantized);
    }

    #[test]
    fn test_graph_info_quantized_field_true() {
        let json = r#"{"operands": [], "input_operands": [], "output_operands": [], "operations": [], "quantized": true}"#;
        let graph: GraphInfo = serde_json::from_str(json).unwrap();
        assert!(graph.quantized);
    }

    #[test]
    fn test_graph_info_quantized_field_serialization() {
        let graph = GraphInfo {
            operands: vec![],
            input_operands: vec![],
            output_operands: vec![],
            operations: vec![],
            constant_operand_ids_to_handles: HashMap::new(),
            quantized: true,
        };
        let json = serde_json::to_string(&graph).unwrap();
        assert!(json.contains("\"quantized\":true"));
    }

    #[test]
    fn test_graph_info_with_int4_operand() {
        let operand = Operand {
            kind: OperandKind::Input,
            descriptor: OperandDescriptor {
                data_type: DataType::Int4,
                shape: to_dimension_vector(&[1, 3, 224, 224]),
                pending_permutation: vec![],
            },
            name: Some("input".to_string()),
        };

        let graph = GraphInfo {
            operands: vec![operand],
            input_operands: vec![0],
            output_operands: vec![],
            operations: vec![],
            constant_operand_ids_to_handles: HashMap::new(),
            quantized: true,
        };

        let json = serde_json::to_string(&graph).unwrap();
        assert!(json.contains("\"int4\""));
        assert!(json.contains("\"quantized\":true"));

        let deserialized: GraphInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(
            deserialized.operands[0].descriptor.data_type,
            DataType::Int4
        );
        assert!(deserialized.quantized);
    }

    #[test]
    fn test_graph_info_with_uint4_operand() {
        let operand = Operand {
            kind: OperandKind::Constant,
            descriptor: OperandDescriptor {
                data_type: DataType::Uint4,
                shape: to_dimension_vector(&[64, 64]),
                pending_permutation: vec![],
            },
            name: Some("weight".to_string()),
        };

        let graph = GraphInfo {
            operands: vec![operand],
            input_operands: vec![],
            output_operands: vec![],
            operations: vec![],
            constant_operand_ids_to_handles: HashMap::new(),
            quantized: true,
        };

        let json = serde_json::to_string(&graph).unwrap();
        assert!(json.contains("\"uint4\""));

        let deserialized: GraphInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(
            deserialized.operands[0].descriptor.data_type,
            DataType::Uint4
        );
    }

    fn sample_io_graph() -> GraphInfo {
        GraphInfo {
            operands: vec![
                Operand {
                    kind: OperandKind::Input,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: to_dimension_vector(&[2, 2]),
                        pending_permutation: vec![],
                    },
                    name: Some("x".to_string()),
                },
                Operand {
                    kind: OperandKind::Output,
                    descriptor: OperandDescriptor {
                        data_type: DataType::Float32,
                        shape: to_dimension_vector(&[2, 2]),
                        pending_permutation: vec![],
                    },
                    name: Some("y".to_string()),
                },
            ],
            input_operands: vec![0],
            output_operands: vec![1],
            operations: vec![],
            constant_operand_ids_to_handles: HashMap::new(),
            quantized: false,
        }
    }

    #[test]
    fn validate_io_operand_lists_accepts_consistent_graph() {
        sample_io_graph().validate_io_operand_lists().unwrap();
        sample_io_graph().io_binding_maps().unwrap();
    }

    #[test]
    fn validate_io_operand_lists_rejects_extra_input_operand_id() {
        let mut graph = sample_io_graph();
        graph.input_operands.push(99);
        std::assert_matches!(
            graph.validate_io_operand_lists(),
            Err(GraphError::InputIdListMismatch { .. })
        );
    }

    #[test]
    fn validate_io_operand_lists_rejects_invalid_operand_id_in_list() {
        let mut graph = sample_io_graph();
        graph.input_operands = vec![0, 99];
        graph.operands.push(Operand {
            kind: OperandKind::Input,
            descriptor: OperandDescriptor {
                data_type: DataType::Float32,
                shape: to_dimension_vector(&[1]),
                pending_permutation: vec![],
            },
            name: Some("z".to_string()),
        });
        std::assert_matches!(
            graph.validate_io_operand_lists(),
            Err(GraphError::InputIdListMismatch { .. })
        );
    }

    #[test]
    fn validate_io_operand_lists_rejects_missing_input_in_list() {
        let mut graph = sample_io_graph();
        graph.operands.push(Operand {
            kind: OperandKind::Input,
            descriptor: OperandDescriptor {
                data_type: DataType::Float32,
                shape: to_dimension_vector(&[1]),
                pending_permutation: vec![],
            },
            name: Some("z".to_string()),
        });
        std::assert_matches!(
            graph.validate_io_operand_lists(),
            Err(GraphError::InputIdListMismatch { .. })
        );
    }

    #[test]
    fn validate_io_operand_lists_rejects_wrong_kind_in_input_list() {
        let mut graph = sample_io_graph();
        graph.input_operands = vec![1];
        std::assert_matches!(
            graph.validate_io_operand_lists(),
            Err(GraphError::InputIdListMismatch { .. })
        );
    }

    #[test]
    fn validate_io_operand_lists_rejects_output_kind_not_in_list() {
        let mut graph = sample_io_graph();
        graph.operands[0].kind = OperandKind::Output;
        graph.output_operands.push(0);
        std::assert_matches!(
            graph.validate_io_operand_lists(),
            Err(GraphError::InputIdListMismatch { .. })
        );
    }
}
