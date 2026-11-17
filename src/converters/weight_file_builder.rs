use crate::error::GraphError;
use std::collections::HashMap;

const SENTINEL: u32 = 0xDEADBEEF;
const ALIGNMENT: usize = 64;
const FILE_VERSION: u32 = 2;

/// BlobDataType values used in the per-entry metadata.
/// Matches the enum from Chromium's graph_builder_coreml.cc.
pub mod blob_data_type {
    pub const FLOAT16: u32 = 1;
    pub const FLOAT32: u32 = 2;
    pub const UINT8: u32 = 3;
    pub const INT8: u32 = 4;
}

/// Builds a CoreML MLProgram blob weight file (format version 2).
///
/// Structure:
///
/// Global file header — exactly 64 bytes at offset 0:
///   [0-3]   u32  entry count (written as 0, patched at finalize)
///   [4-7]   u32  version = 2
///   [8-63]  u8[] zeros
///
/// Per-entry — at a 64-byte aligned offset:
///   WeightMetadata block — 64 bytes:
///     [0-3]   u32  sentinel = 0xDEADBEEF
///     [4-7]   u32  mil_data_type (BlobDataType enum)
///     [8-15]  u64  size_in_bytes (byte length of payload)
///     [16-23] u64  absolute file offset of payload (= metadata offset + 64)
///     [24-63] u8[] zeros
///   Raw payload — size_in_bytes bytes
///   Padding — zeros to next 64-byte boundary
///
/// BlobFileValue.offset points to the WeightMetadata block (not the payload).
///
/// Reference: Chromium services/webnn/coreml/graph_builder_coreml.cc
pub struct WeightFileBuilder {
    sink: Sink,
    offsets: HashMap<u32, u64>,
    original_offsets: HashMap<u32, Vec<u64>>,
    entry_count: u32,
}

/// Where the file is assembled. The offsets and the bytes are the same for both.
enum Sink {
    Memory(Vec<u8>),
    /// An anonymous temporary file and its length so far. A language model's weights are
    /// then never on the heap a second time while the graph still holds them.
    File(std::io::BufWriter<std::fs::File>, u64),
}

/// A finished weight file.
pub(crate) enum WeightBlob {
    Memory(Vec<u8>),
    File(std::fs::File, u64),
}

impl WeightBlob {
    /// The file's bytes. Drop the graph's constants before calling this for a file-backed
    /// blob, or both are on the heap at once.
    pub(crate) fn into_bytes(self) -> Result<Vec<u8>, GraphError> {
        match self {
            WeightBlob::Memory(bytes) => Ok(bytes),
            WeightBlob::File(mut file, length) => {
                use std::io::{Read, Seek, SeekFrom};
                let mut bytes = Vec::new();
                bytes
                    .try_reserve_exact(length as usize)
                    .map_err(|e| io_error(std::io::Error::other(e)))?;
                file.seek(SeekFrom::Start(0)).map_err(io_error)?;
                file.read_to_end(&mut bytes).map_err(io_error)?;
                Ok(bytes)
            }
        }
    }
}

fn io_error(error: std::io::Error) -> GraphError {
    GraphError::ConversionFailed {
        format: "coreml_mlprogram".to_string(),
        reason: format!("weight file: {error}"),
    }
}

impl Sink {
    fn len(&self) -> u64 {
        match self {
            Sink::Memory(data) => data.len() as u64,
            Sink::File(_, length) => *length,
        }
    }

    fn write(&mut self, bytes: &[u8]) -> Result<(), GraphError> {
        match self {
            Sink::Memory(data) => data.extend_from_slice(bytes),
            Sink::File(file, length) => {
                std::io::Write::write_all(file, bytes).map_err(io_error)?;
                *length += bytes.len() as u64;
            }
        }
        Ok(())
    }

    /// Zero-fills up to `length`.
    fn pad_to(&mut self, length: u64) -> Result<(), GraphError> {
        let padding = length.saturating_sub(self.len()) as usize;
        self.write(&[0u8; ALIGNMENT][..padding.min(ALIGNMENT)])?;
        debug_assert_eq!(
            self.len(),
            length,
            "padding is less than one alignment unit"
        );
        Ok(())
    }

    /// The `length` bytes at `offset`.
    fn read(&mut self, offset: u64, length: usize) -> Result<Vec<u8>, GraphError> {
        match self {
            Sink::Memory(data) => Ok(data[offset as usize..offset as usize + length].to_vec()),
            Sink::File(file, end) => {
                use std::io::{Read, Seek, SeekFrom, Write};
                file.flush().map_err(io_error)?;
                let file = file.get_mut();
                let mut bytes = vec![0; length];
                file.seek(SeekFrom::Start(offset)).map_err(io_error)?;
                file.read_exact(&mut bytes).map_err(io_error)?;
                file.seek(SeekFrom::Start(*end)).map_err(io_error)?;
                Ok(bytes)
            }
        }
    }
}

impl WeightFileBuilder {
    pub fn new() -> Self {
        Self::with_sink(Sink::Memory(Vec::new())).expect("writing to memory cannot fail")
    }

    /// Assembles the file in an anonymous temporary file instead of on the heap. Fails
    /// where the process cannot create one (a sandbox, a full disk); use [`Self::new`] then.
    pub(crate) fn file_backed() -> Result<Self, GraphError> {
        let file = tempfile::tempfile().map_err(io_error)?;
        Self::with_sink(Sink::File(std::io::BufWriter::new(file), 0))
    }

    fn with_sink(sink: Sink) -> Result<Self, GraphError> {
        let mut builder = Self {
            sink,
            offsets: HashMap::new(),
            original_offsets: HashMap::new(),
            entry_count: 0,
        };
        // Write the 64-byte file header (count = 0 for now; patched at finalize)
        builder.write_file_header(0)?;
        Ok(builder)
    }

    /// Adds a weight entry. Returns the absolute file offset of the WeightMetadata
    /// block (this value goes into BlobFileValue.offset).
    ///
    /// `mil_data_type` is a BlobDataType enum value (e.g. blob_data_type::FLOAT16).
    pub fn add_weight(
        &mut self,
        operand_id: u32,
        mil_data_type: u32,
        data: &[u8],
    ) -> Result<u64, GraphError> {
        if self.offsets.contains_key(&operand_id) {
            return Err(GraphError::ConversionFailed {
                format: "coreml_mlprogram".to_string(),
                reason: format!("Duplicate weight for operand {}", operand_id),
            });
        }

        let offset = self.append_weight(mil_data_type, data)?;
        self.offsets.insert(operand_id, offset);
        Ok(offset)
    }

    /// Reuse an unchanged MIL weight, or retain one raw original record when
    /// lowering has coerced the value or stored it as a MIL immediate.
    pub(crate) fn original_weight(
        &mut self,
        operand_id: u32,
        bytes: &[u8],
    ) -> Result<u64, GraphError> {
        let candidates: Vec<u64> = self
            .offsets
            .get(&operand_id)
            .into_iter()
            .chain(self.original_offsets.get(&operand_id).into_iter().flatten())
            .copied()
            .collect();
        for offset in candidates {
            let header = self.sink.read(offset + 8, 8)?;
            let length = u64::from_le_bytes(header.try_into().expect("eight bytes")) as usize;
            if length == bytes.len() && self.sink.read(offset + ALIGNMENT as u64, length)? == *bytes
            {
                return Ok(offset);
            }
        }
        let offset = self.append_weight(blob_data_type::UINT8, bytes)?;
        self.original_offsets
            .entry(operand_id)
            .or_default()
            .push(offset);
        Ok(offset)
    }

    fn append_weight(&mut self, mil_data_type: u32, data: &[u8]) -> Result<u64, GraphError> {
        // Align to 64-byte boundary before writing the metadata block
        let metadata_offset = align_to(self.sink.len() as usize, ALIGNMENT) as u64;
        self.sink.pad_to(metadata_offset)?;

        let size_in_bytes = data.len() as u64;
        // Payload starts immediately after the 64-byte metadata block
        let payload_offset = metadata_offset + ALIGNMENT as u64;

        // 64-byte WeightMetadata block; `metadata_offset` is the BlobFileValue.offset.
        let mut metadata = [0u8; ALIGNMENT];
        metadata[0..4].copy_from_slice(&SENTINEL.to_le_bytes());
        metadata[4..8].copy_from_slice(&mil_data_type.to_le_bytes());
        metadata[8..16].copy_from_slice(&size_in_bytes.to_le_bytes());
        metadata[16..24].copy_from_slice(&payload_offset.to_le_bytes());
        self.sink.write(&metadata)?;

        // Write raw payload
        self.sink.write(data)?;

        self.entry_count += 1;
        Ok(metadata_offset)
    }

    /// Pre-allocate for the expected payload volume. Weight files reach
    /// hundreds of MB; growing by amortized doubling briefly holds ~3x the
    /// final size during the last realloc.
    pub fn reserve(&mut self, additional: usize) {
        if let Sink::Memory(data) = &mut self.sink {
            data.reserve(additional);
        }
    }

    /// Returns the file offset for a previously added weight.
    #[allow(dead_code)]
    pub fn get_offset(&self, operand_id: u32) -> Option<u64> {
        self.offsets.get(&operand_id).copied()
    }

    /// Finalizes an in-memory file: patches the entry count in the header and pads to
    /// alignment.
    #[cfg(test)]
    pub fn finalize(self) -> Vec<u8> {
        self.finish()
            .and_then(WeightBlob::into_bytes)
            .expect("an in-memory weight file cannot fail; use `finish` for a file-backed one")
    }

    /// [`Self::finalize`] without reading a file-backed blob back into memory.
    pub(crate) fn finish(mut self) -> Result<WeightBlob, GraphError> {
        // Pad to 64-byte boundary
        let aligned = align_to(self.sink.len() as usize, ALIGNMENT) as u64;
        self.sink.pad_to(aligned)?;

        // Patch entry count at offset 0
        let count_bytes = self.entry_count.to_le_bytes();
        match self.sink {
            Sink::Memory(mut data) => {
                data[0..4].copy_from_slice(&count_bytes);
                Ok(WeightBlob::Memory(data))
            }
            Sink::File(file, length) => {
                use std::io::{Seek, SeekFrom, Write};
                let mut file = file.into_inner().map_err(|e| io_error(e.into_error()))?;
                file.seek(SeekFrom::Start(0)).map_err(io_error)?;
                file.write_all(&count_bytes).map_err(io_error)?;
                Ok(WeightBlob::File(file, length))
            }
        }
    }

    pub fn has_weights(&self) -> bool {
        self.entry_count > 0
    }

    #[allow(dead_code)]
    pub fn size(&self) -> usize {
        self.sink.len() as usize
    }

    fn write_file_header(&mut self, count: u32) -> Result<(), GraphError> {
        let mut header = [0u8; ALIGNMENT];
        header[0..4].copy_from_slice(&count.to_le_bytes());
        header[4..8].copy_from_slice(&FILE_VERSION.to_le_bytes());
        self.sink.write(&header)
    }
}

/// Index complete records before accepting metadata byte references. An aligned
/// address inside another payload is not a record, even if it resembles a header.
#[cfg(all(feature = "coreml-runtime", any(target_vendor = "apple", test)))]
pub(crate) fn weight_ranges(
    bytes: &[u8],
) -> Result<HashMap<u64, std::ops::Range<usize>>, GraphError> {
    let invalid = || GraphError::CoremlRuntimeFailed {
        reason: "invalid CoreML constant-copy weight file".into(),
    };
    let header = bytes.get(..ALIGNMENT).ok_or_else(invalid)?;
    let count = u32::from_le_bytes(header[..4].try_into().unwrap());
    if u32::from_le_bytes(header[4..8].try_into().unwrap()) != FILE_VERSION
        || header[8..].iter().any(|&byte| byte != 0)
        || count as usize > bytes.len() / ALIGNMENT
    {
        return Err(invalid());
    }
    let mut ranges = HashMap::new();
    let mut offset = ALIGNMENT;
    for _ in 0..count {
        let end = offset.checked_add(ALIGNMENT).ok_or_else(invalid)?;
        let header = bytes.get(offset..end).ok_or_else(invalid)?;
        let length = usize::try_from(u64::from_le_bytes(header[8..16].try_into().unwrap()))
            .map_err(|_| invalid())?;
        let start = usize::try_from(u64::from_le_bytes(header[16..24].try_into().unwrap()))
            .map_err(|_| invalid())?;
        if u32::from_le_bytes(header[..4].try_into().unwrap()) != SENTINEL
            || !matches!(u32::from_le_bytes(header[4..8].try_into().unwrap()), 1..=4)
            || start != end
            || header[24..].iter().any(|&byte| byte != 0)
        {
            return Err(invalid());
        }
        let end = start.checked_add(length).ok_or_else(invalid)?;
        bytes.get(start..end).ok_or_else(invalid)?;
        ranges.insert(offset as u64, start..end);
        offset = end.checked_add(ALIGNMENT - 1).ok_or_else(invalid)? & !(ALIGNMENT - 1);
    }
    if offset != bytes.len() {
        return Err(invalid());
    }
    Ok(ranges)
}

fn align_to(offset: usize, alignment: usize) -> usize {
    (offset + (alignment - 1)) & !(alignment - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_file_header_written_on_new() {
        let builder = WeightFileBuilder::new();
        assert_eq!(builder.size(), 64, "Header must be exactly 64 bytes");
        // version at bytes [4-7]
        let data = builder.finalize();
        assert_eq!(&data[4..8], &2u32.to_le_bytes(), "File version must be 2");
    }

    #[test]
    fn test_empty_builder_has_only_header() {
        let builder = WeightFileBuilder::new();
        assert!(!builder.has_weights());
        let data = builder.finalize();
        // Entry count = 0
        assert_eq!(&data[0..4], &0u32.to_le_bytes());
        // Version = 2
        assert_eq!(&data[4..8], &2u32.to_le_bytes());
        // File is 64 bytes (header only, aligned)
        assert_eq!(data.len(), 64);
    }

    #[test]
    fn test_single_weight_layout() {
        let mut builder = WeightFileBuilder::new();
        let payload = vec![0x00u8, 0x3C, 0x00, 0x40, 0x00, 0x42]; // f16: 1.0, 2.0, 3.0
        let offset = builder
            .add_weight(0, blob_data_type::FLOAT16, &payload)
            .unwrap();

        // Metadata block starts at byte 64 (right after header)
        assert_eq!(offset, 64);
        assert!(builder.has_weights());

        let data = builder.finalize();

        // Header: entry count = 1
        assert_eq!(&data[0..4], &1u32.to_le_bytes());
        assert_eq!(&data[4..8], &2u32.to_le_bytes()); // version

        // Metadata block at offset 64:
        // [64-67]  sentinel
        assert_eq!(&data[64..68], &SENTINEL.to_le_bytes());
        // [68-71]  mil_data_type = FLOAT16 = 1
        assert_eq!(&data[68..72], &blob_data_type::FLOAT16.to_le_bytes());
        // [72-79]  size_in_bytes = 6
        assert_eq!(&data[72..80], &6u64.to_le_bytes());
        // [80-87]  payload absolute offset = 64 + 64 = 128
        assert_eq!(&data[80..88], &128u64.to_le_bytes());
        // [88-127] zeros
        assert!(data[88..128].iter().all(|&b| b == 0));

        // Payload at offset 128
        assert_eq!(&data[128..134], &payload[..]);
    }

    #[test]
    fn test_multiple_weights() {
        let mut builder = WeightFileBuilder::new();

        let d1 = vec![0xAAu8, 0xBB]; // 2 bytes
        let off1 = builder.add_weight(0, blob_data_type::FLOAT16, &d1).unwrap();
        assert_eq!(off1, 64); // right after 64-byte header

        let d2 = vec![0x11u8, 0x22, 0x33, 0x44]; // 4 bytes
        let off2 = builder.add_weight(1, blob_data_type::FLOAT16, &d2).unwrap();

        // d1 metadata at 64, payload at 128 (2 bytes), padded to 192.
        // d2 metadata at 192.
        assert_eq!(off2, 192);

        let data = builder.finalize();
        assert_eq!(&data[0..4], &2u32.to_le_bytes()); // 2 entries
        // d2 payload offset = 192 + 64 = 256
        assert_eq!(&data[256..260], &d2[..]);
    }

    #[test]
    fn test_duplicate_operand_error() {
        let mut builder = WeightFileBuilder::new();
        let d = vec![0x00u8, 0x01];
        builder.add_weight(0, blob_data_type::FLOAT16, &d).unwrap();
        assert!(builder.add_weight(0, blob_data_type::FLOAT16, &d).is_err());
    }

    #[test]
    fn a_file_backed_builder_writes_the_same_file() {
        let build = |mut builder: WeightFileBuilder| {
            let first = builder
                .add_weight(1, blob_data_type::FLOAT16, &[1, 2, 3])
                .unwrap();
            // Reads back what it wrote, then keeps appending at the end.
            assert_eq!(builder.original_weight(1, &[1, 2, 3]).unwrap(), first);
            let original = builder.original_weight(1, &[9; 70]).unwrap();
            let second = builder
                .add_weight(2, blob_data_type::INT8, &[4; 65])
                .unwrap();
            assert_eq!(builder.get_offset(2), Some(second));
            assert_eq!(builder.original_weight(1, &[9; 70]).unwrap(), original);
            let size = builder.size();
            let bytes = builder.finish().unwrap().into_bytes().unwrap();
            assert!(bytes.len() >= size && bytes.len().is_multiple_of(ALIGNMENT));
            (first, original, second, bytes)
        };
        let memory = build(WeightFileBuilder::new());
        let file = build(WeightFileBuilder::file_backed().unwrap());
        assert_eq!(memory, file);
        assert_eq!(u32::from_le_bytes(file.3[..4].try_into().unwrap()), 3);
    }

    #[test]
    fn original_weight_reuses_equal_bytes_but_keeps_coerced_bytes_separate() {
        let mut builder = WeightFileBuilder::new();
        let offset = builder
            .add_weight(7, blob_data_type::UINT8, &[1, 2])
            .unwrap();
        assert_eq!(builder.original_weight(7, &[1, 2]).unwrap(), offset);
        let original = builder.original_weight(7, &[1, 0, 2, 0]).unwrap();
        assert_ne!(original, offset);
        assert_eq!(builder.original_weight(7, &[1, 0, 2, 0]).unwrap(), original);
        let weights = builder.finalize();
        assert_eq!(u32::from_le_bytes(weights[..4].try_into().unwrap()), 2);
        assert_eq!(
            &weights[original as usize + 64..original as usize + 68],
            &[1, 0, 2, 0]
        );
    }

    #[test]
    fn original_weight_checks_bytes_when_reusing_an_operand_id() {
        let mut builder = WeightFileBuilder::new();
        let first = builder.original_weight(7, &[1, 0, 2, 0]).unwrap();
        let changed = builder.original_weight(7, &[1, 0, 3, 0]).unwrap();
        assert_ne!(first, changed);
        assert_eq!(builder.original_weight(7, &[1, 0, 2, 0]).unwrap(), first);
        assert_eq!(builder.original_weight(7, &[1, 0, 3, 0]).unwrap(), changed);
        let weights = builder.finalize();
        assert_eq!(u32::from_le_bytes(weights[..4].try_into().unwrap()), 2);
    }

    #[test]
    fn test_large_weight_alignment() {
        let mut builder = WeightFileBuilder::new();
        // 100 float16 values = 200 bytes
        let payload = vec![0xABu8; 200];
        let offset = builder
            .add_weight(0, blob_data_type::FLOAT16, &payload)
            .unwrap();
        assert_eq!(offset, 64);

        let data = builder.finalize();
        // metadata at 64, payload at 128 (200 bytes), padded to 64-byte boundary
        // 128 + 200 = 328 -> aligned to 384
        assert_eq!(data.len(), 384);
        assert_eq!(&data[128..328], &payload[..]);
    }
}
