# Opaque Device Tensors Implementation Plan

## Goal
Implement device-resident tensor I/O binding in rustnn to eliminate host round-trips for iterative GenAI workloads (KV cache). Tensors remain on GPU/NPU across decode steps, with explicit host transfers only when needed.

## Current Architecture Summary

### Tensor Storage (src/python/tensor.rs)
- `PyMLTensor`: Opaque tensor with `Arc<Mutex<Vec<f32>>>` backing
- Descriptor flags: `readable`, `writable`, `exportable_to_gpu`
- **Limitation**: All data stored as CPU vectors, no GPU backing

### Compute Flow (src/python/context.rs)
- **Synchronous compute()**: numpy dict → backend → numpy dict
- **Async dispatch()**: MLTensor → read to numpy → compute → write to MLTensor
- **Current bottleneck**: Every dispatch() reads/writes through numpy, causing host round-trips

### Backend Execution
- **ONNX Runtime** (executors/onnx.rs): Supports 8 input dtypes, returns f32 outputs
- **CoreML** (executors/coreml.rs): macOS Neural Engine, f32 only
- **TensorRT** (executors/trtx.rs): NVIDIA GPU via trtx crate

### Key Issue
Current `dispatch()` implementation (context.rs:150-177):
```rust
// Every call does this:
for input in inputs {
    numpy_inputs[name] = read_tensor(input)  // Device→CPU copy
}
result = compute(graph, numpy_inputs)         // Execute
for output in outputs {
    write_tensor(output, result[name])        // CPU→Device copy
}
```
This defeats the purpose of persistent tensors for KV cache.

## Implementation Plan

### Phase 1: Core Rust Infrastructure

#### 1.1 Add TensorValue Enum (src/tensor.rs - NEW FILE)
**Purpose**: Unified representation for host and device tensors

```rust
// src/tensor.rs
pub enum TensorValue {
    Host(HostTensor),
    Device(DeviceTensorHandle),
}

pub struct HostTensor {
    data: Vec<f32>,  // Start with f32, expand to TensorData later
    shape: Vec<usize>,
    dtype: DataType,
}

pub struct DeviceTensorHandle {
    inner: Box<dyn DeviceTensorBackend>,
    dtype: DataType,
    shape: Vec<usize>,
}

pub trait DeviceTensorBackend: Send + Sync {
    fn dtype(&self) -> DataType;
    fn shape(&self) -> &[usize];
    fn device_kind(&self) -> DeviceKind;
    fn backend_kind(&self) -> BackendKind;

    // Host transfers
    fn read_to_host(&self) -> Result<Vec<f32>, GraphError>;
    fn write_from_host(&mut self, data: &[f32]) -> Result<(), GraphError>;
}

pub enum DeviceKind {
    Cpu,
    Cuda,
    DirectML,
    CoreML,
}

pub enum BackendKind {
    OnnxCpu,
    OnnxGpu,
    CoreML,
    TensorRT,
}
```

**Files to create:**
- `src/tensor.rs` - Core tensor abstractions
- Add `pub mod tensor;` to `src/lib.rs`

#### 1.2 ONNX Runtime Device Tensors (src/executors/onnx.rs)
**Purpose**: Implement DeviceTensorBackend using ONNX Runtime IoBinding

**Key Implementation:**
```rust
// Add to src/executors/onnx.rs

pub struct OrtDeviceTensor {
    value: Value<'static>,      // OrtValue on device
    session: Arc<Session>,      // Keep session alive
    dtype: DataType,
    shape: Vec<usize>,
    device: DeviceKind,
}

impl DeviceTensorBackend for OrtDeviceTensor {
    fn read_to_host(&self) -> Result<Vec<f32>, GraphError> {
        // Extract from OrtValue to host buffer
        match self.dtype {
            DataType::Float32 => {
                let arr = self.value.try_extract_tensor::<f32>()?;
                Ok(arr.as_slice()?.to_vec())
            }
            // ... other types
        }
    }

    fn write_from_host(&mut self, data: &[f32]) -> Result<(), GraphError> {
        // Create new OrtValue from host data
        // This requires recreating the Value - ORT doesn't support in-place writes
        let array = ndarray::Array::from_shape_vec(
            IxDyn(&self.shape),
            data.to_vec()
        )?;
        self.value = Value::from_array(&array)?;
        Ok(())
    }
}

// New function: Create device tensor
pub fn create_ort_device_tensor(
    session: Arc<Session>,
    shape: &[usize],
    dtype: DataType,
    device: DeviceKind,
) -> Result<OrtDeviceTensor, GraphError> {
    // Allocate OrtValue on device
    let array = ndarray::Array::zeros(IxDyn(shape));
    let value = Value::from_array(&array)?;

    Ok(OrtDeviceTensor {
        value,
        session,
        dtype,
        shape: shape.to_vec(),
        device,
    })
}

// New function: Execute with device tensor bindings
pub fn run_onnx_with_bindings(
    session: &Session,
    input_bindings: Vec<(&str, &OrtDeviceTensor)>,
    output_bindings: Vec<(&str, &mut OrtDeviceTensor)>,
) -> Result<(), GraphError> {
    // Use ORT IoBinding API
    let io_binding = session.create_io_binding()?;

    // Bind inputs
    for (name, tensor) in input_bindings {
        io_binding.bind_input(name, &tensor.value)?;
    }

    // Bind outputs
    for (name, tensor) in output_bindings {
        io_binding.bind_output(name, &mut tensor.value)?;
    }

    // Run with bindings (no host copies)
    session.run_with_io_binding(&io_binding)?;

    Ok(())
}
```

**Files to modify:**
- `src/executors/onnx.rs` - Add OrtDeviceTensor and run_onnx_with_bindings()
- May need to update `ort` crate usage for IoBinding API

#### 1.3 Update Context Backend Management (src/python/context.rs)
**Purpose**: Store session handle for device tensor creation

**Changes needed:**
```rust
pub struct PyMLContext {
    backend: Backend,
    power_preference: String,
    accelerated_available: bool,

    // NEW: Store session for device tensor creation
    onnx_session: Arc<Mutex<Option<Arc<Session>>>>,
}

impl PyMLContext {
    // NEW: Get or create ONNX session
    fn get_onnx_session(&self, graph: &PyMLGraph) -> Result<Arc<Session>, GraphError> {
        let mut session_guard = self.onnx_session.lock().unwrap();

        if let Some(session) = session_guard.as_ref() {
            return Ok(Arc::clone(session));
        }

        // Create session
        let converted = convert_graph_to_onnx(&graph.graph_info)?;
        let session = create_onnx_session(&converted.data)?;
        let session_arc = Arc::new(session);
        *session_guard = Some(Arc::clone(&session_arc));

        Ok(session_arc)
    }
}
```

**Files to modify:**
- `src/python/context.rs` - Add session caching

### Phase 2: Python API

#### 2.1 Expose DeviceTensor to Python (src/python/tensor.rs)
**Purpose**: New Python class for device-resident tensors

**Implementation:**
```rust
// Modify src/python/tensor.rs

#[pyclass(name = "MLDeviceTensor")]
pub struct PyMLDeviceTensor {
    handle: DeviceTensorHandle,
    destroyed: Arc<Mutex<bool>>,
}

#[pymethods]
impl PyMLDeviceTensor {
    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.handle.shape().to_vec()
    }

    #[getter]
    fn data_type(&self) -> String {
        format!("{:?}", self.handle.dtype()).to_lowercase()
    }

    #[getter]
    fn device(&self) -> String {
        format!("{:?}", self.handle.device_kind()).to_lowercase()
    }

    #[getter]
    fn backend(&self) -> String {
        format!("{:?}", self.handle.backend_kind()).to_lowercase()
    }

    fn read(&self, py: Python) -> PyResult<PyObject> {
        // Device → CPU transfer
        let destroyed = self.destroyed.lock().unwrap();
        if *destroyed {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Tensor has been destroyed"
            ));
        }
        drop(destroyed);

        let data = self.handle.inner.read_to_host()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let numpy = py.import("numpy")?;
        let array = numpy.call_method1("array", (data,))?;
        let reshaped = array.call_method1("reshape", (self.handle.shape.clone(),))?;
        Ok(reshaped.to_object(py))
    }

    fn write(&mut self, py: Python, array: PyObject) -> PyResult<()> {
        // CPU → Device transfer
        let destroyed = self.destroyed.lock().unwrap();
        if *destroyed {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Tensor has been destroyed"
            ));
        }
        drop(destroyed);

        let numpy_array = extract_numpy_array(py, array)?;
        self.handle.inner.write_from_host(&numpy_array)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(())
    }

    fn destroy(&mut self) {
        let mut destroyed = self.destroyed.lock().unwrap();
        *destroyed = true;
    }
}
```

**Files to modify:**
- `src/python/tensor.rs` - Add PyMLDeviceTensor class
- `src/python/mod.rs` - Export new class

#### 2.2 Add Context Methods (src/python/context.rs)
**Purpose**: Create device tensors from context

**New methods:**
```rust
#[pymethods]
impl PyMLContext {
    fn create_device_tensor(
        &mut self,
        shape: Vec<usize>,
        dtype: String,
        device: Option<String>,
    ) -> PyResult<PyMLDeviceTensor> {
        // Parse dtype
        let data_type = parse_dtype(&dtype)?;

        // Determine device from backend
        let device_kind = match self.backend {
            Backend::OnnxGpu => DeviceKind::Cuda,
            Backend::CoreML => DeviceKind::CoreML,
            Backend::TensorRT => DeviceKind::Cuda,
            _ => DeviceKind::Cpu,
        };

        // Create backend-specific tensor
        let handle = match self.backend {
            Backend::OnnxCpu | Backend::OnnxGpu => {
                // Need session - this is tricky, might need lazy creation
                // For now, error if no session yet
                return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    "Cannot create device tensor before building a graph"
                ));
            }
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    "Device tensors not supported for this backend yet"
                ));
            }
        };

        Ok(PyMLDeviceTensor {
            handle,
            destroyed: Arc::new(Mutex::new(false)),
        })
    }

    fn read_device_tensor(&self, py: Python, tensor: &PyMLDeviceTensor) -> PyResult<PyObject> {
        tensor.read(py)
    }

    fn write_device_tensor(&self, py: Python, tensor: &mut PyMLDeviceTensor, array: PyObject) -> PyResult<()> {
        tensor.write(py, array)
    }
}
```

**Files to modify:**
- `src/python/context.rs` - Add create_device_tensor(), read_device_tensor(), write_device_tensor()

#### 2.3 Extend dispatch() Signature (src/python/context.rs)
**Purpose**: Accept mix of numpy arrays and DeviceTensors

**New implementation:**
```rust
#[pymethods]
impl PyMLContext {
    fn dispatch(
        &mut self,
        py: Python,
        graph: &PyMLGraph,
        inputs: &PyDict,
        outputs: Option<&PyDict>,
    ) -> PyResult<Option<PyObject>> {
        // Parse inputs: either numpy arrays or DeviceTensors
        let mut numpy_inputs: HashMap<String, PyObject> = HashMap::new();
        let mut device_inputs: HashMap<String, &PyMLDeviceTensor> = HashMap::new();

        for (key, value) in inputs.iter() {
            let name: String = key.extract()?;

            // Try to extract as DeviceTensor
            if let Ok(device_tensor) = value.extract::<PyRef<PyMLDeviceTensor>>() {
                device_inputs.insert(name, &*device_tensor);
            } else {
                // Assume numpy array
                numpy_inputs.insert(name, value.to_object(py));
            }
        }

        // Parse outputs: dict of {name: DeviceTensor or "host"}
        let output_spec = if let Some(outputs_dict) = outputs {
            let mut spec = HashMap::new();
            for (key, value) in outputs_dict.iter() {
                let name: String = key.extract()?;

                // Try DeviceTensor
                if let Ok(device_tensor) = value.extract::<PyRef<PyMLDeviceTensor>>() {
                    spec.insert(name, OutputSpec::Device(&*device_tensor));
                } else if let Ok(s) = value.extract::<String>() {
                    if s == "host" {
                        spec.insert(name, OutputSpec::Host);
                    }
                }
            }
            Some(spec)
        } else {
            None
        };

        // Execute based on input types
        if device_inputs.is_empty() {
            // All host inputs - use existing compute() path
            let result = self.compute(py, graph, inputs)?;

            if let Some(spec) = output_spec {
                // Write to output tensors
                let result_dict: &PyDict = result.extract(py)?;
                for (name, output_type) in spec {
                    match output_type {
                        OutputSpec::Device(tensor) => {
                            let array = result_dict.get_item(&name)?.unwrap();
                            tensor.write(py, array.to_object(py))?;
                        }
                        OutputSpec::Host => {
                            // Already in result dict
                        }
                    }
                }
                Ok(None)  // No return when outputs specified
            } else {
                Ok(Some(result))
            }
        } else {
            // Has device inputs - use device path
            self.dispatch_with_device_tensors(
                py,
                graph,
                device_inputs,
                numpy_inputs,
                output_spec,
            )
        }
    }

    fn dispatch_with_device_tensors(
        &mut self,
        py: Python,
        graph: &PyMLGraph,
        device_inputs: HashMap<String, &PyMLDeviceTensor>,
        numpy_inputs: HashMap<String, PyObject>,
        output_spec: Option<HashMap<String, OutputSpec>>,
    ) -> PyResult<Option<PyObject>> {
        // THIS IS THE KEY OPTIMIZATION:
        // Call backend with device tensor bindings, no host copies

        match self.backend {
            Backend::OnnxCpu | Backend::OnnxGpu => {
                // Use run_onnx_with_bindings()
                // TODO: Extract OrtDeviceTensor from DeviceTensorHandle
                // TODO: Call executors::onnx::run_onnx_with_bindings()
                unimplemented!("Device tensor execution for ONNX")
            }
            _ => {
                // Fallback: read device tensors to host, execute normally
                for (name, tensor) in device_inputs {
                    numpy_inputs.insert(name, tensor.read(py)?);
                }

                let result = self.compute(py, graph, &numpy_to_pydict(py, numpy_inputs)?)?;

                if let Some(spec) = output_spec {
                    // Write outputs
                    // ... similar to above
                }

                Ok(None)
            }
        }
    }
}

enum OutputSpec<'a> {
    Device(&'a PyMLDeviceTensor),
    Host,
}
```

**Files to modify:**
- `src/python/context.rs` - Rewrite dispatch() to support device tensors

### Phase 3: Backend Integration

#### 3.1 Complete ONNX Runtime IoBinding (src/executors/onnx.rs)
**Priority**: This is the critical path for KV cache optimization

**Implementation steps:**
1. Research ONNX Runtime IoBinding API in `ort` crate
2. Implement `run_onnx_with_bindings()` using SessionInputValue and SessionOutputValue
3. Handle mixed device/host inputs
4. Add error handling for device mismatch
5. Add logging for profiling (count host→device transfers)

**Files to modify:**
- `src/executors/onnx.rs` - Complete IoBinding implementation

#### 3.2 CoreML Device Tensors (src/executors/coreml.rs)
**Note**: CoreML may not support persistent device tensors via Objective-C FFI

**Fallback strategy:**
- Store device tensors as host buffers
- Still expose DeviceTensor API to Python
- Copy on execute (no worse than current state)

**Files to modify:**
- `src/executors/coreml.rs` - Add CoreMLDeviceTensor (backed by Vec<f32>)

#### 3.3 TensorRT Device Tensors (src/executors/trtx.rs)
**Note**: Depends on `trtx` crate capabilities

**Research needed:**
- Check if trtx supports persistent CUDA buffers
- Investigate `trtx::Tensor` API for device memory

**Files to modify:**
- `src/executors/trtx.rs` - Add TrtxDeviceTensor if supported

### Phase 4: Testing & Validation

#### 4.1 Rust Unit Tests
**Files**: `src/tensor.rs`, `src/executors/onnx.rs`

```rust
#[cfg(test)]
mod tests {
    #[test]
    fn test_device_tensor_create_read_write() {
        // Create device tensor
        // Write host data
        // Read back
        // Assert equality
    }

    #[test]
    fn test_onnx_device_execution() {
        // Load tiny model (add operation)
        // Create device input tensors
        // Execute with bindings
        // Read device output
        // Assert correct result
    }
}
```

#### 4.2 Python Integration Tests
**Files**: `tests/test_device_tensors.py` (NEW)

```python
def test_create_device_tensor():
    ml = webnn.ML()
    ctx = ml.create_context(device_type="gpu")

    tensor = ctx.create_device_tensor([2, 3], "float32")
    assert tensor.shape == [2, 3]
    assert tensor.data_type == "float32"

def test_device_tensor_read_write():
    ml = webnn.ML()
    ctx = ml.create_context(device_type="gpu")

    tensor = ctx.create_device_tensor([2, 3], "float32")
    data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)

    tensor.write(data)
    result = tensor.read()
    np.testing.assert_array_equal(result, data)

def test_dispatch_with_device_tensors():
    ml = webnn.ML()
    ctx = ml.create_context(device_type="gpu")
    builder = ctx.create_graph_builder()

    # Build graph: y = x + 1
    x = builder.input("x", [2, 3], "float32")
    one = builder.constant([2, 3], "float32", np.ones((2, 3)))
    y = builder.add(x, one)
    graph = builder.build({"y": y})

    # Create device tensors
    x_device = ctx.create_device_tensor([2, 3], "float32")
    y_device = ctx.create_device_tensor([2, 3], "float32")

    # Write input
    x_data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    x_device.write(x_data)

    # Execute with device tensors (NO HOST COPIES)
    ctx.dispatch(graph, {"x": x_device}, {"y": y_device})

    # Read output
    result = y_device.read()
    expected = x_data + 1
    np.testing.assert_array_almost_equal(result, expected)
```

#### 4.3 KV Cache Performance Test
**Files**: `examples/kv_cache_device_tensors.py` (NEW)

```python
def test_kv_cache_no_copies():
    """Simulated decoder loop: measure host transfer count"""

    ml = webnn.ML()
    ctx = ml.create_context(device_type="gpu")
    builder = ctx.create_graph_builder()

    # Build decoder graph with KV cache inputs/outputs
    # (simplified: just pass-through for now)
    past_k = builder.input("past_k", [1, 8, seq_len, 64], "float32")
    past_v = builder.input("past_v", [1, 8, seq_len, 64], "float32")
    # ... decode logic
    present_k = builder.identity(past_k)  # Placeholder
    present_v = builder.identity(past_v)  # Placeholder

    graph = builder.build({
        "present_k": present_k,
        "present_v": present_v,
    })

    # Create persistent device tensors
    kv_shape = [1, 8, 128, 64]
    past_k_device = ctx.create_device_tensor(kv_shape, "float32")
    past_v_device = ctx.create_device_tensor(kv_shape, "float32")
    present_k_device = ctx.create_device_tensor(kv_shape, "float32")
    present_v_device = ctx.create_device_tensor(kv_shape, "float32")

    # Initialize
    past_k_device.write(np.zeros(kv_shape, dtype=np.float32))
    past_v_device.write(np.zeros(kv_shape, dtype=np.float32))

    # Run 100 decode steps
    for step in range(100):
        ctx.dispatch(
            graph,
            {"past_k": past_k_device, "past_v": past_v_device},
            {"present_k": present_k_device, "present_v": present_v_device},
        )

        # Swap references (no copies!)
        past_k_device, present_k_device = present_k_device, past_k_device
        past_v_device, present_v_device = present_v_device, past_v_device

    # Only read once at the end
    final_k = past_k_device.read()
    print(f"Completed 100 steps with only 1 host transfer")
```

### Phase 5: Documentation & Examples

#### 5.1 Update API Reference
**Files**: `docs/user-guide/api-reference.md`

Add sections:
- MLDeviceTensor class
- create_device_tensor() method
- Extended dispatch() signature
- Memory management best practices

#### 5.2 Add KV Cache Example
**Files**: `examples/kv_cache_example.py`

Show:
- Creating device tensors
- Running decode loop
- Ping-pong buffer pattern
- Performance comparison (with/without device tensors)

#### 5.3 Update Architecture Docs
**Files**: `docs/architecture/overview.md`

Add section on device tensors:
- TensorValue abstraction
- Backend capabilities
- Memory management
- Performance characteristics

## Implementation Order

### Week 1: Core Infrastructure
1. Create `src/tensor.rs` with TensorValue enum and traits
2. Implement OrtDeviceTensor in `src/executors/onnx.rs`
3. Add session caching to `src/python/context.rs`

### Week 2: Python API
4. Add PyMLDeviceTensor to `src/python/tensor.rs`
5. Add create_device_tensor() to `src/python/context.rs`
6. Implement basic read/write methods

### Week 3: Dispatch Integration
7. Extend dispatch() to accept DeviceTensors
8. Implement device tensor execution path
9. Add run_onnx_with_bindings() for ONNX backend

### Week 4: Testing & Polish
10. Write Rust unit tests
11. Write Python integration tests
12. Add KV cache performance example
13. Update documentation

## Critical Files to Modify

**New Files:**
- `src/tensor.rs` - Core tensor abstractions
- `tests/test_device_tensors.py` - Python tests
- `examples/kv_cache_device_tensors.py` - Performance example

**Modified Files:**
- `src/lib.rs` - Export new module
- `src/python/tensor.rs` - Add PyMLDeviceTensor
- `src/python/context.rs` - Add create/read/write methods, extend dispatch()
- `src/python/mod.rs` - Export new class
- `src/executors/onnx.rs` - Add OrtDeviceTensor and IoBinding
- `src/error.rs` - Add device tensor error types
- `docs/user-guide/api-reference.md` - Document new API

## Risk Mitigation

### Risk 1: ONNX Runtime IoBinding API Complexity
**Mitigation**: Start with simple test case (single input/output), expand gradually

### Risk 2: Session Lifetime Management
**Mitigation**: Use Arc<Session> in DeviceTensorHandle to keep session alive

### Risk 3: Breaking Existing API
**Mitigation**: Keep existing compute() unchanged, only extend dispatch()

### Risk 4: Backend Compatibility
**Mitigation**: Implement for ONNX first, fallback to host tensors for others

## Success Criteria

1. **Functional**: Device tensors can be created, read, written
2. **Performance**: 100-step decode loop with no intermediate host transfers
3. **Compatible**: Existing compute() API unchanged and working
4. **Tested**: 20+ tests covering device tensor operations
5. **Documented**: API reference and KV cache example complete

## Open Questions

1. **Session creation timing**: When to create ONNX session for device tensors?
   - Option A: Require build() first, then create_device_tensor()
   - Option B: Lazy session creation on first create_device_tensor()
   - **Recommendation**: Option A - explicit build() before device tensors

2. **Multi-dtype support**: Start with f32 only or support all types?
   - **Recommendation**: Start with f32, add types incrementally

3. **Device selection**: Explicit device parameter or infer from backend?
   - **Recommendation**: Infer from backend, optional override later

4. **Output allocation**: Pre-allocate or auto-allocate?
   - **Recommendation**: Support both - pre-allocated for KV cache, auto for others
