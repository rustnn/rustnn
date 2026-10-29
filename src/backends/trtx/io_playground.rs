//! File-backed constant upload algorithms for the TensorRT-RTX backend.
//!
//! Implements memory mapping, buffered file reads, io_uring and cuFile uploads. The
//! configuration and implementation details remain internal to the TensorRT backend.

use std::collections::HashMap;
#[cfg(all(target_os = "linux", feature = "io-uring"))]
use std::os::fd::AsRawFd;
use std::sync::Arc;

#[cfg(all(target_os = "linux", feature = "cufile"))]
use cudarc::cufile::safe::Cufile;
use cudarc::driver::{CudaContext, CudaEvent, CudaStream};
#[cfg(all(target_os = "linux", feature = "cufile"))]
use log::debug;
use log::warn;

#[cfg(all(target_os = "linux", feature = "cufile"))]
use super::SharedCudaTensor;
use super::{HostTensor, TrtxTensor, TrtxTensorStorage, graph_build_error};
use crate::error::GraphBuilderError;
use crate::graph::FileRef;

#[derive(Debug, Eq, PartialEq, Default)]
#[allow(dead_code)]
enum IoAlgorithm {
    #[default]
    ReadDirect,
    MemMap,
    #[cfg(all(target_os = "linux", feature = "io-uring"))]
    IoUring {
        uring_entries: u32,
    },
    #[cfg(all(target_os = "linux", feature = "cufile"))]
    CuFile,
    #[cfg(all(target_os = "linux", feature = "cufile"))]
    CuFilePerConstant,
}

#[derive(Debug, Eq, PartialEq)]
pub(super) struct IoConfig {
    algorithm: IoAlgorithm,
    num_upload_buffers: usize,
    upload_buffer_size: usize,
}

impl Default for IoConfig {
    fn default() -> Self {
        // TODO: determine good defaults
        Self {
            algorithm: Default::default(),
            num_upload_buffers: 4,
            upload_buffer_size: 16 * 1024 * 1024,
        }
    }
}

#[cfg(all(target_os = "linux", feature = "io-uring"))]
#[derive(Clone, Copy)]
struct FileReadChunk {
    destination_offset: usize,
    len: usize,
}

pub(super) fn upload_file_ref_constants(
    cuda_context: &Arc<CudaContext>,
    file_refs: &[(u32, FileRef, Option<crate::OperandDescriptor>)],
    io_config: &IoConfig,
) -> crate::error::Result<Vec<(u32, TrtxTensorStorage)>> {
    match io_config.algorithm {
        IoAlgorithm::ReadDirect => {
            upload_file_ref_constants_read_direct(cuda_context, file_refs, io_config)
        }
        IoAlgorithm::MemMap => upload_file_ref_constants_memmap(cuda_context, file_refs),
        #[cfg(all(target_os = "linux", feature = "io-uring"))]
        IoAlgorithm::IoUring { uring_entries } => upload_file_ref_constants_io_uring(
            cuda_context,
            file_refs,
            uring_entries,
            io_config.num_upload_buffers,
            io_config.upload_buffer_size,
        ),
        #[cfg(all(target_os = "linux", feature = "cufile"))]
        IoAlgorithm::CuFile => upload_file_ref_constants_cufile(cuda_context, file_refs),
        #[cfg(all(target_os = "linux", feature = "cufile"))]
        IoAlgorithm::CuFilePerConstant => {
            upload_file_ref_constants_cufile_per_constant(cuda_context, file_refs)
        }
    }
}

// Cache handles and metadata once per path, rather than once per constant. Validate all
// ranges before allocating destinations or queuing any copies.
struct FileUploadSource {
    file: std::fs::File,
    len: usize,
}

fn open_file_upload_sources(
    file_refs: &[(u32, FileRef, Option<crate::OperandDescriptor>)],
    direct: bool,
) -> crate::error::Result<HashMap<std::path::PathBuf, FileUploadSource>> {
    let mut sources = HashMap::new();
    for entry @ (_, file_ref, _) in file_refs {
        if let std::collections::hash_map::Entry::Vacant(slot) =
            sources.entry(file_ref.path.clone())
        {
            let mut options = std::fs::OpenOptions::new();
            options.read(true);
            #[cfg(target_os = "linux")]
            if direct {
                use std::os::unix::fs::OpenOptionsExt;
                options.custom_flags(libc::O_DIRECT);
            }
            #[cfg(not(target_os = "linux"))]
            let _ = direct;
            let file = options
                .open(&file_ref.path)
                .map_err(|error| crate::error::GraphError::io(&file_ref.path, error))?;
            let len = file
                .metadata()
                .map_err(|error| crate::error::GraphError::io(&file_ref.path, error))?
                .len();
            slot.insert(FileUploadSource {
                file,
                len: usize::try_from(len).map_err(graph_build_error)?,
            });
        }
        validate_file_ref_with_len(entry, sources[&file_ref.path].len as u64)?;
    }
    Ok(sources)
}

fn file_upload_order(file_refs: &[(u32, FileRef, Option<crate::OperandDescriptor>)]) -> Vec<usize> {
    let mut order = (0..file_refs.len()).collect::<Vec<_>>();
    order.sort_unstable_by(|&a, &b| {
        let a = &file_refs[a].1;
        let b = &file_refs[b].1;
        a.path.cmp(&b.path).then_with(|| a.offset.cmp(&b.offset))
    });
    order
}

// Declared after mappings / host buffers so error cleanup drains copies before releasing
// their source memory. On success there is exactly one final stream synchronization.
struct FileUploadStream {
    stream: Arc<CudaStream>,
    drained: bool,
}

impl FileUploadStream {
    fn finish(&mut self) -> crate::error::Result<()> {
        self.stream.synchronize().map_err(graph_build_error)?;
        self.drained = true;
        Ok(())
    }
}

impl Drop for FileUploadStream {
    fn drop(&mut self) {
        if !self.drained
            && let Err(error) = self.stream.synchronize()
        {
            warn!("Failed to drain constant upload stream: {error}");
        }
    }
}

fn allocate_file_upload_tensors(
    stream: &Arc<CudaStream>,
    file_refs: &[(u32, FileRef, Option<crate::OperandDescriptor>)],
) -> crate::error::Result<Vec<(u32, TrtxTensorStorage)>> {
    // Allocate before uploading: synchronous allocation on older CUDA devices must not
    // interrupt an in-flight batch. All allocations and copies use the same stream.
    file_refs
        .iter()
        .map(|(constant_id, file_ref, _)| {
            // SAFETY: The upload paths initialize every byte before returning these tensors.
            let memory = unsafe { stream.alloc::<u8>(file_ref.len) }.map_err(graph_build_error)?;
            Ok((
                *constant_id,
                TrtxTensorStorage::TrtxTensor(TrtxTensor {
                    memory,
                    stream: Arc::clone(stream),
                }),
            ))
        })
        .collect()
}

fn upload_file_ref_constants_memmap(
    cuda_context: &Arc<CudaContext>,
    file_refs: &[(u32, FileRef, Option<crate::OperandDescriptor>)],
) -> crate::error::Result<Vec<(u32, TrtxTensorStorage)>> {
    if file_refs.is_empty() {
        return Ok(Vec::new());
    }
    let sources = open_file_upload_sources(file_refs, false)?;
    let mut mappings = HashMap::with_capacity(sources.len());
    for (path, source) in &sources {
        if source.len != 0 {
            // SAFETY: FileRef sources must remain unchanged during graph building. Mappings
            // outlive the upload guard, including during error cleanup.
            let mapping = unsafe { memmap2::MmapOptions::new().map(&source.file) }
                .map_err(|error| crate::error::GraphError::io(path, error))?;
            mappings.insert(path, mapping);
        }
    }
    let mut upload = FileUploadStream {
        stream: cuda_context.new_stream().map_err(graph_build_error)?,
        drained: false,
    };
    let mut uploaded = allocate_file_upload_tensors(&upload.stream, file_refs)?;
    for index in file_upload_order(file_refs) {
        let file_ref = &file_refs[index].1;
        if file_ref.len == 0 {
            continue;
        }
        let source = &mappings[&file_ref.path][file_ref.offset..file_ref.offset + file_ref.len];
        let TrtxTensorStorage::TrtxTensor(tensor) = &mut uploaded[index].1 else {
            unreachable!("file uploads allocate CUDA tensors");
        };
        upload
            .stream
            .memcpy_htod(source, &mut tensor.memory)
            .map_err(graph_build_error)?;
    }
    upload.finish()?;
    Ok(uploaded)
}

struct FileUploadBuffer {
    _storage: TrtxTensorStorage,
    // Aligned pointer into the pinned allocation owned by _storage. Access is protected
    // by completion, recorded after the last copy using this read window.
    ptr: *mut u8,
    completion: CudaEvent,
}

struct FileReadWindow<'a> {
    path: &'a std::path::Path,
    offset: usize,
    len: usize,
    prefix: usize,
    buffer_index: usize,
}

const DIRECT_READ_ALIGNMENT: usize = 4096;

fn direct_read_range(offset: usize, len: usize) -> std::io::Result<(usize, usize, usize)> {
    let prefix = offset % DIRECT_READ_ALIGNMENT;
    let read_len = prefix
        .checked_add(len)
        .and_then(|size| size.checked_next_multiple_of(DIRECT_READ_ALIGNMENT))
        .ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidInput, "read size overflow")
        })?;
    Ok((offset - prefix, prefix, read_len))
}

fn upload_file_ref_constants_read_direct(
    cuda_context: &Arc<CudaContext>,
    file_refs: &[(u32, FileRef, Option<crate::OperandDescriptor>)],
    config: &IoConfig,
) -> crate::error::Result<Vec<(u32, TrtxTensorStorage)>> {
    use std::io::{Read, Seek, SeekFrom};

    if config.num_upload_buffers == 0 || config.upload_buffer_size == 0 {
        return Err(graph_build_error(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "num_upload_buffers and upload_buffer_size must be greater than zero",
        )));
    }
    if file_refs.is_empty() {
        return Ok(Vec::new());
    }
    // Allow both an unaligned file offset and an unaligned pinned allocation address.
    let allocation_size = direct_read_range(DIRECT_READ_ALIGNMENT - 1, config.upload_buffer_size)
        .map_err(graph_build_error)?
        .2
        .checked_add(DIRECT_READ_ALIGNMENT - 1)
        .ok_or_else(|| {
            graph_build_error(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "upload buffer size overflow",
            ))
        })?;
    let mut sources = open_file_upload_sources(file_refs, true)?;
    let stream = cuda_context.new_stream().map_err(graph_build_error)?;
    let mut buffers = Vec::with_capacity(config.num_upload_buffers);
    for _ in 0..config.num_upload_buffers {
        let mut host =
            HostTensor::new_with_stream(cuda_context, allocation_size, Arc::clone(&stream))
                .map_err(graph_build_error)?;
        // Obtain the pointer once, before any upload is queued. PinnedHostSlice accessors
        // synchronize its internal event, which is unnecessary for each subsequent window.
        let ptr = host.memory.as_mut_ptr().map_err(graph_build_error)?;
        let ptr = ptr.wrapping_add(ptr.align_offset(DIRECT_READ_ALIGNMENT));
        buffers.push(FileUploadBuffer {
            _storage: TrtxTensorStorage::HostTensor(host),
            ptr,
            completion: cuda_context.new_event(None).map_err(graph_build_error)?,
        });
    }
    let mut upload = FileUploadStream {
        stream,
        drained: false,
    };
    let mut uploaded = allocate_file_upload_tensors(&upload.stream, file_refs)?;
    let mut window: Option<FileReadWindow<'_>> = None;
    let mut next_buffer = 0;
    for index in file_upload_order(file_refs) {
        let file_ref = &file_refs[index].1;
        let source_file = sources
            .get_mut(&file_ref.path)
            .expect("validated upload source");
        let TrtxTensorStorage::TrtxTensor(tensor) = &mut uploaded[index].1 else {
            unreachable!("file uploads allocate CUDA tensors");
        };
        let mut offset = 0;
        while offset < file_ref.len {
            let file_offset = file_ref.offset + offset; // validated range cannot overflow
            if !window.as_ref().is_some_and(|cached| {
                cached.path == file_ref.path
                    && file_offset >= cached.offset
                    && file_offset - cached.offset < cached.len
            }) {
                if let Some(previous) = window.take() {
                    // One event per read window, even when it supplies many tensors.
                    buffers[previous.buffer_index]
                        .completion
                        .record(&upload.stream)
                        .map_err(graph_build_error)?;
                }
                let buffer = &buffers[next_buffer];
                if !buffer
                    .completion
                    .try_is_complete()
                    .map_err(graph_build_error)?
                {
                    // Wait only for this buffer's last use, not later work on the stream.
                    buffer.completion.synchronize().map_err(graph_build_error)?;
                }
                let len = config.upload_buffer_size.min(source_file.len - file_offset);
                let (read_offset, prefix, read_len) = if cfg!(target_os = "linux") {
                    direct_read_range(file_offset, len).map_err(graph_build_error)?
                } else {
                    (file_offset, 0, len)
                };
                // SAFETY: The allocation includes alignment padding and read_len bytes.
                // The completion event ensures CUDA no longer reads this buffer.
                let target = unsafe { std::slice::from_raw_parts_mut(buffer.ptr, read_len) };
                source_file
                    .file
                    .seek(SeekFrom::Start(read_offset as u64))
                    .map_err(|error| crate::error::GraphError::io(&file_ref.path, error))?;
                let bytes_read = if cfg!(target_os = "linux") {
                    // Padding may extend beyond EOF, but the payload must be complete.
                    loop {
                        match source_file.file.read(target) {
                            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {
                                continue;
                            }
                            result => break result,
                        }
                    }
                    .map_err(|error| crate::error::GraphError::io(&file_ref.path, error))?
                } else {
                    source_file
                        .file
                        .read_exact(target)
                        .map_err(|error| crate::error::GraphError::io(&file_ref.path, error))?;
                    read_len
                };
                if bytes_read < prefix + len {
                    return Err(crate::error::GraphError::io(
                        &file_ref.path,
                        std::io::Error::new(
                            std::io::ErrorKind::UnexpectedEof,
                            "direct read did not contain the requested tensor bytes",
                        ),
                    )
                    .into());
                }
                // Read ahead across constant boundaries and also reuse any alignment tail.
                window = Some(FileReadWindow {
                    path: &file_ref.path,
                    offset: file_offset,
                    len: bytes_read - prefix,
                    prefix,
                    buffer_index: next_buffer,
                });
                next_buffer = (next_buffer + 1) % buffers.len();
            }
            let cached = window.as_ref().expect("read window covers this offset");
            let source_offset = file_offset - cached.offset;
            let len = (file_ref.len - offset).min(cached.len - source_offset);
            // SAFETY: This range contains initialized bytes from the cached read. The buffer
            // is immutable until its completion event permits reuse; the guard drains all
            // outstanding copies before the pinned allocation can be dropped.
            let source = unsafe {
                std::slice::from_raw_parts(
                    buffers[cached.buffer_index]
                        .ptr
                        .add(cached.prefix + source_offset),
                    len,
                )
            };
            let mut destination = tensor.memory.slice_mut(offset..offset + len);
            upload
                .stream
                .memcpy_htod(source, &mut destination)
                .map_err(graph_build_error)?;
            offset += len;
        }
    }
    upload.finish()?;
    Ok(uploaded)
}

#[cfg(all(target_os = "linux", feature = "io-uring"))]
fn upload_file_ref_constants_io_uring(
    cuda_context: &Arc<CudaContext>,
    file_refs: &[(u32, FileRef, Option<crate::OperandDescriptor>)],
    uring_entries: u32,
    num_upload_buffers: usize,
    upload_buffer_size: usize,
) -> crate::error::Result<Vec<(u32, TrtxTensorStorage)>> {
    if uring_entries == 0 {
        return Err(graph_build_error(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "uring_entries must be greater than zero",
        )));
    }
    if num_upload_buffers == 0 || num_upload_buffers > uring_entries as usize {
        return Err(graph_build_error(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "num_upload_buffers must be in 1..=uring_entries",
        )));
    }
    if upload_buffer_size == 0 || upload_buffer_size > u32::MAX as usize {
        return Err(graph_build_error(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "upload_buffer_size must be in 1..=u32::MAX",
        )));
    }
    if file_refs.is_empty() {
        return Ok(Vec::new());
    }

    cuda_context.bind_to_thread().map_err(graph_build_error)?;
    let mut ring = io_uring::IoUring::new(uring_entries).map_err(graph_build_error)?;
    let mut upload_buffers = (0..uring_entries)
        .map(|_| HostTensor::new(cuda_context, upload_buffer_size).map_err(graph_build_error))
        .collect::<crate::error::Result<Vec<_>>>()?;
    let mut uploaded_tensors = Vec::with_capacity(file_refs.len());

    for (constant_id, file_ref, descriptor) in file_refs {
        let descriptor =
            descriptor
                .as_ref()
                .ok_or_else(|| GraphBuilderError::InconsistentGraphInfo {
                    message: format!("FileRef constant {constant_id} has no operand descriptor"),
                })?;
        let expected_bytes = descriptor.byte_length().ok_or_else(|| {
            GraphBuilderError::RequestedConstantDataForDynamicallyShapedConstant {
                id: *constant_id,
                desc: descriptor.clone(),
            }
        })?;
        if file_ref.len != expected_bytes {
            return Err(GraphBuilderError::InconsistentGraphInfo {
                message: format!(
                    "FileRef for constant {constant_id} has {} bytes, expected {expected_bytes}",
                    file_ref.len
                ),
            }
            .into());
        }

        let file = std::fs::File::open(&file_ref.path)
            .map_err(|error| crate::error::GraphError::io(&file_ref.path, error))?;
        let mut tensor =
            TrtxTensor::new(cuda_context, expected_bytes).map_err(graph_build_error)?;
        let mut chunks = vec![None; uring_entries as usize];
        let mut next_offset = 0usize;
        let mut in_flight = 0usize;

        loop {
            for buffer_index in 0..upload_buffers.len() {
                if next_offset == file_ref.len || chunks[buffer_index].is_some() {
                    continue;
                }

                // The buffer may still be the source of an asynchronous CUDA copy from the
                // previous batch. It must not be handed back to the kernel until that copy ends.
                upload_buffers[buffer_index]
                    .stream
                    .synchronize()
                    .map_err(graph_build_error)?;

                let len = upload_buffer_size.min(file_ref.len - next_offset);
                let file_offset = file_ref.offset.checked_add(next_offset).ok_or_else(|| {
                    graph_build_error(std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        format!("file offset overflow for {:?}", file_ref.path),
                    ))
                })?;
                let buffer_ptr = upload_buffers[buffer_index]
                    .memory
                    .as_mut_ptr()
                    .map_err(graph_build_error)?;
                let entry = io_uring::opcode::Read::new(
                    io_uring::types::Fd(file.as_raw_fd()),
                    buffer_ptr,
                    len as u32,
                )
                .offset(file_offset as u64)
                .build()
                .user_data(buffer_index as u64);
                unsafe {
                    ring.submission().push(&entry).map_err(|_| {
                        graph_build_error(std::io::Error::other(
                            "io_uring submission queue is full",
                        ))
                    })?;
                }
                chunks[buffer_index] = Some(FileReadChunk {
                    destination_offset: next_offset,
                    len,
                });
                next_offset += len;
                in_flight += 1;
            }

            if in_flight == 0 {
                break;
            }

            let completions_to_wait_for = num_upload_buffers.min(in_flight);
            ring.submit_and_wait(completions_to_wait_for)
                .map_err(graph_build_error)?;
            for _ in 0..completions_to_wait_for {
                let completion = ring.completion().next().ok_or_else(|| {
                    graph_build_error(std::io::Error::other(
                        "io_uring returned fewer completions than requested",
                    ))
                })?;
                let buffer_index = completion.user_data() as usize;
                let chunk = chunks
                    .get_mut(buffer_index)
                    .and_then(Option::take)
                    .ok_or_else(|| {
                        graph_build_error(std::io::Error::other(format!(
                            "io_uring returned invalid buffer index {buffer_index}"
                        )))
                    })?;
                let read_result = completion.result();
                if read_result < 0 {
                    return Err(crate::error::GraphError::io(
                        &file_ref.path,
                        std::io::Error::from_raw_os_error(-read_result),
                    )
                    .into());
                }
                if read_result as usize != chunk.len {
                    return Err(crate::error::GraphError::io(
                        &file_ref.path,
                        std::io::Error::new(
                            std::io::ErrorKind::UnexpectedEof,
                            format!(
                                "read {} bytes at offset {}, expected {}",
                                read_result,
                                file_ref.offset + chunk.destination_offset,
                                chunk.len
                            ),
                        ),
                    )
                    .into());
                }

                let source = &upload_buffers[buffer_index]
                    .memory
                    .as_slice()
                    .map_err(graph_build_error)?[..chunk.len];
                let mut destination = tensor
                    .memory
                    .slice_mut(chunk.destination_offset..chunk.destination_offset + chunk.len);
                upload_buffers[buffer_index]
                    .stream
                    .memcpy_htod(source, &mut destination)
                    .map_err(graph_build_error)?;
                in_flight -= 1;
            }
        }

        for buffer in &upload_buffers {
            buffer.stream.synchronize().map_err(graph_build_error)?;
        }
        uploaded_tensors.push((*constant_id, TrtxTensorStorage::TrtxTensor(tensor)));
    }

    Ok(uploaded_tensors)
}

#[cfg(all(target_os = "linux", feature = "cufile"))]
fn upload_file_ref_constants_cufile(
    cuda_context: &Arc<CudaContext>,
    file_refs: &[(u32, FileRef, Option<crate::OperandDescriptor>)],
) -> crate::error::Result<Vec<(u32, TrtxTensorStorage)>> {
    use std::collections::BTreeMap;
    use std::os::unix::fs::OpenOptionsExt;

    if file_refs.is_empty() {
        return Ok(Vec::new());
    }

    cuda_context.bind_to_thread().map_err(graph_build_error)?;
    let cufile = Cufile::new().map_err(graph_build_error)?;
    let mut refs_by_path = BTreeMap::<_, Vec<_>>::new();
    for file_ref in file_refs {
        validate_file_ref(file_ref)?;
        refs_by_path
            .entry(file_ref.1.path.clone())
            .or_default()
            .push(file_ref);
    }

    let mut uploaded_tensors = Vec::with_capacity(file_refs.len());
    for (path, refs) in refs_by_path {
        let file = std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_DIRECT)
            .open(&path)
            .map_err(|error| crate::error::GraphError::io(&path, error))?;
        let file_len = usize::try_from(
            file.metadata()
                .map_err(|error| crate::error::GraphError::io(&path, error))?
                .len(),
        )
        .map_err(graph_build_error)?;
        let first_offset = refs
            .iter()
            .map(|(_, file_ref, _)| file_ref.offset)
            .min()
            .expect("a grouped path always has at least one FileRef");
        let allocation_len = file_len.checked_sub(first_offset).ok_or_else(|| {
            graph_build_error(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                format!(
                    "FileRef offset {first_offset} exceeds the size of {path:?} ({file_len} bytes)"
                ),
            ))
        })?;
        for (_, file_ref, _) in &refs {
            let end = file_ref.offset.checked_add(file_ref.len).ok_or_else(|| {
                graph_build_error(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!("FileRef range overflows for {:?}", file_ref.path),
                ))
            })?;
            if end > file_len {
                return Err(crate::error::GraphError::io(
                    &path,
                    std::io::Error::new(
                        std::io::ErrorKind::UnexpectedEof,
                        format!(
                            "FileRef range {}..{end} exceeds the file size {file_len}",
                            file_ref.offset
                        ),
                    ),
                )
                .into());
            }
        }

        debug!(
            "Allocating {allocation_len} CUDA bytes for cuFile upload from {path:?} at offset {first_offset}"
        );
        let stream = cuda_context.new_stream().map_err(graph_build_error)?;
        // SAFETY: cuFile initializes the complete allocation before any tensor view is exposed.
        let mut allocation =
            unsafe { stream.alloc::<u8>(allocation_len) }.map_err(graph_build_error)?;
        let handle = cufile.register(file).map_err(graph_build_error)?;
        cufile
            .buf_register(&allocation)
            .map_err(graph_build_error)?;
        let read_result = i64::try_from(first_offset)
            .map_err(graph_build_error)
            .and_then(|offset| {
                handle
                    .sync_read(offset, &mut allocation)
                    .map_err(graph_build_error)
            });
        let deregister_result = cufile
            .buf_deregister(&allocation)
            .map_err(graph_build_error);
        let bytes_read = read_result?;
        deregister_result?;
        if bytes_read != allocation_len as isize {
            return Err(crate::error::GraphError::io(
                &path,
                std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    format!("cuFile read {bytes_read} bytes, expected {allocation_len}"),
                ),
            )
            .into());
        }

        let allocation = Arc::new(allocation);
        for (constant_id, file_ref, _) in refs {
            uploaded_tensors.push((
                *constant_id,
                TrtxTensorStorage::SharedCudaTensor(SharedCudaTensor {
                    memory: Arc::clone(&allocation),
                    offset: file_ref.offset - first_offset,
                    len: file_ref.len,
                    stream: Arc::clone(&stream),
                }),
            ));
        }
    }

    Ok(uploaded_tensors)
}

#[cfg(all(target_os = "linux", feature = "cufile"))]
fn upload_file_ref_constants_cufile_per_constant(
    cuda_context: &Arc<CudaContext>,
    file_refs: &[(u32, FileRef, Option<crate::OperandDescriptor>)],
) -> crate::error::Result<Vec<(u32, TrtxTensorStorage)>> {
    use std::os::unix::fs::OpenOptionsExt;

    if file_refs.is_empty() {
        return Ok(Vec::new());
    }

    cuda_context.bind_to_thread().map_err(graph_build_error)?;
    let cufile = Cufile::new().map_err(graph_build_error)?;
    let mut uploaded_tensors = Vec::with_capacity(file_refs.len());
    for file_ref in file_refs {
        validate_file_ref(file_ref)?;
        let (constant_id, file_ref, _) = file_ref;
        let file = std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_DIRECT)
            .open(&file_ref.path)
            .map_err(|error| crate::error::GraphError::io(&file_ref.path, error))?;
        let handle = cufile.register(file).map_err(graph_build_error)?;
        let stream = cuda_context.new_stream().map_err(graph_build_error)?;
        // SAFETY: cuFile initializes the complete allocation before it is returned.
        let memory = unsafe { stream.alloc::<u8>(file_ref.len) }.map_err(graph_build_error)?;
        let mut tensor = TrtxTensor { memory, stream };
        cufile
            .buf_register(&tensor.memory)
            .map_err(graph_build_error)?;
        let read_result = i64::try_from(file_ref.offset)
            .map_err(graph_build_error)
            .and_then(|offset| {
                handle
                    .sync_read(offset, &mut tensor.memory)
                    .map_err(graph_build_error)
            });
        let deregister_result = cufile
            .buf_deregister(&tensor.memory)
            .map_err(graph_build_error);
        let bytes_read = read_result?;
        deregister_result?;
        if bytes_read != file_ref.len as isize {
            return Err(crate::error::GraphError::io(
                &file_ref.path,
                std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    format!("cuFile read {bytes_read} bytes, expected {}", file_ref.len),
                ),
            )
            .into());
        }
        uploaded_tensors.push((*constant_id, TrtxTensorStorage::TrtxTensor(tensor)));
    }

    Ok(uploaded_tensors)
}

#[cfg(any(test, all(target_os = "linux", feature = "cufile")))]
fn validate_file_ref(
    entry: &(u32, FileRef, Option<crate::OperandDescriptor>),
) -> crate::error::Result<()> {
    let file_len = std::fs::metadata(&entry.1.path)
        .map_err(|error| crate::error::GraphError::io(&entry.1.path, error))?
        .len();
    validate_file_ref_with_len(entry, file_len)
}

fn validate_file_ref_with_len(
    (constant_id, file_ref, descriptor): &(u32, FileRef, Option<crate::OperandDescriptor>),
    file_len: u64,
) -> crate::error::Result<()> {
    let descriptor =
        descriptor
            .as_ref()
            .ok_or_else(|| GraphBuilderError::InconsistentGraphInfo {
                message: format!("FileRef constant {constant_id} has no operand descriptor"),
            })?;
    let expected_bytes = descriptor.byte_length().ok_or_else(|| {
        GraphBuilderError::RequestedConstantDataForDynamicallyShapedConstant {
            id: *constant_id,
            desc: descriptor.clone(),
        }
    })?;
    if file_ref.len != expected_bytes {
        return Err(GraphBuilderError::InconsistentGraphInfo {
            message: format!(
                "FileRef for constant {constant_id} has {} bytes, expected {expected_bytes}",
                file_ref.len
            ),
        }
        .into());
    }
    let end = file_ref.offset.checked_add(file_ref.len).ok_or_else(|| {
        graph_build_error(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "FileRef range overflow",
        ))
    })?;
    if end as u64 > file_len {
        return Err(crate::error::GraphError::io(
            &file_ref.path,
            std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "FileRef exceeds file size",
            ),
        )
        .into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{backends::trtx::TrtxContext, mlcontext::ListDevices};

    fn file_entry(
        path: &std::path::Path,
        offset: usize,
        len: usize,
    ) -> (u32, FileRef, Option<crate::OperandDescriptor>) {
        (
            0,
            FileRef {
                path: path.to_owned(),
                offset,
                len,
                label: None,
            },
            Some(crate::OperandDescriptor {
                data_type: crate::graph::DataType::Uint8,
                shape: vec![crate::graph::Dimension::Static(len as u32)],
                pending_permutation: Vec::new(),
            }),
        )
    }

    #[test]
    fn file_upload_rejects_invalid_ranges() {
        let file = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(file.path(), [0u8; 8]).unwrap();
        assert!(validate_file_ref(&file_entry(file.path(), 7, 2)).is_err());
        assert!(validate_file_ref(&file_entry(file.path(), usize::MAX, 2)).is_err());
        let mut entry = file_entry(file.path(), 0, 4);
        entry.1.len = 3;
        assert!(validate_file_ref(&entry).is_err());
        entry.2 = None;
        assert!(validate_file_ref(&entry).is_err());
        assert!(direct_read_range(1, usize::MAX).is_err());
    }

    #[test]
    fn file_upload_algorithms_preserve_bytes() {
        let devices = TrtxContext::list_devices();
        let Some(device) = devices.first() else {
            eprintln!("Skipping file upload test: no CUDA device available");
            return;
        };
        let cuda = CudaContext::new(*device.as_trtx_device().unwrap() as usize).unwrap();
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("weights.bin");
        let other = directory.path().join("other.bin");
        let bytes = (0..20003).map(|i| (i % 251) as u8).collect::<Vec<_>>();
        std::fs::write(&path, &bytes).unwrap();
        std::fs::write(&other, &bytes).unwrap();
        let mut entries = vec![
            file_entry(&path, 3, 9001),
            file_entry(&path, 8191, bytes.len() - 8191),
            file_entry(&other, 17, 41),
        ];
        // Adjacent constants should share a read-ahead window. Scramble their order to
        // verify that sequential file processing preserves the caller's result order.
        for index in (0..40).rev() {
            entries.push(file_entry(&path, 10003 + index * 31, 31));
        }
        for (id, entry) in entries.iter_mut().enumerate() {
            entry.0 = id as u32;
        }
        for algorithm in [IoAlgorithm::MemMap, IoAlgorithm::ReadDirect] {
            for num_upload_buffers in [1, 3] {
                for upload_buffer_size in [1, 1023, 65536] {
                    let config = IoConfig {
                        algorithm: if algorithm == IoAlgorithm::MemMap {
                            IoAlgorithm::MemMap
                        } else {
                            IoAlgorithm::ReadDirect
                        },
                        num_upload_buffers,
                        upload_buffer_size,
                    };
                    let uploaded = upload_file_ref_constants(&cuda, &entries, &config).unwrap();
                    assert_eq!(uploaded.len(), entries.len());
                    for ((id, storage), (expected_id, file_ref, _)) in uploaded.iter().zip(&entries)
                    {
                        assert_eq!(id, expected_id);
                        let TrtxTensorStorage::TrtxTensor(tensor) = storage else {
                            panic!("expected CUDA tensor");
                        };
                        let actual = tensor.stream.clone_dtoh(&tensor.memory).unwrap();
                        assert_eq!(
                            actual,
                            bytes[file_ref.offset..file_ref.offset + file_ref.len]
                        );
                    }
                }
            }
        }
        for (num_upload_buffers, upload_buffer_size) in [(0, 1024), (2, 0)] {
            let config = IoConfig {
                algorithm: IoAlgorithm::ReadDirect,
                num_upload_buffers,
                upload_buffer_size,
            };
            assert!(upload_file_ref_constants(&cuda, &entries, &config).is_err());
        }
    }
}
