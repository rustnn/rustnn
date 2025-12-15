# IPC Design Notes

## Overview

This document outlines the design considerations for adding Inter-Process Communication (IPC) support to this WebNN implementation, drawing from Chromium's architecture.

## Current Architecture (Single-Process)

### Intermediate Representation

**Format:** Rust structs with JSON attributes
```rust
pub struct Operation {
    pub op_type: String,              // e.g., "conv2d"
    pub input_operands: Vec<u32>,     // operand IDs
    pub output_operand: Option<u32>,
    pub attributes: serde_json::Value, // Flexible JSON
    pub label: Option<String>,
}
```

**Benefits:**
- Simple: no code generation
- Flexible: easy to add operations
- Debuggable: human-readable JSON
- Serializable: can save/load graphs
- Cross-language: works with Python/Rust/CLI

**Limitations for IPC:**
- JSON parsing overhead on every access
- No structured validation at serialization boundaries
- String-based keys prone to typos
- Runtime-only validation

## Chromium's Architecture (Multi-Process)

### Process Model

```
Browser Process (JavaScript)
    ↓ Mojo IPC
Service Process (C++ WebNN)
    ↓ Platform APIs
GPU Process / ML Hardware
```

### Intermediate Representation

**Format:** [Mojo IDL](https://chromium.googlesource.com/chromium/src/+/lkgr/services/webnn/public/mojom/webnn_graph.mojom) - strongly-typed structs

**Example Operation:**
```mojo
struct Conv2d {
  OperandId input_operand_id;
  OperandId filter_operand_id;
  OperandId? bias_operand_id;  // Optional
  Padding2d padding;
  Size2d strides;
  Size2d dilations;
  uint32 groups;
  InputOperandLayout input_layout;
  Conv2dKind kind;
};

union Operation {
  Conv2d conv2d;
  ElementWiseBinary elementwise_binary;
  Reduce reduce;
  // ... 50+ operation types
};

struct GraphInfo {
  array<Operand> operands;
  array<Operation> operations;  // Sorted topologically
  map<uint64, ConstantOperandData> constant_operand_data;
};
```

**Benefits:**
- Type safety at compile time
- Binary serialization for efficient IPC
- Structured validation by Mojo compiler
- Auto-generated bindings (C++/JavaScript/etc.)
- Versioned interfaces for compatibility

**Drawbacks:**
- Requires Mojo build system
- Less flexible - changes require IDL updates
- Browser-specific infrastructure
- More complex build process

**Reference:**
- Mojo interface: `services/webnn/public/mojom/webnn_graph.mojom`
- Implementation: `services/webnn/webnn_graph_impl.{h,cc}`

## Design Options for IPC Support

### Option 1: Cap'n Proto (Recommended)

[Cap'n Proto](https://capnproto.org/) is a modern, efficient serialization format similar to Mojo but platform-agnostic.

**Architecture:**
```rust
// Define schema in schema.capnp
struct Conv2d {
  inputOperandId @0 :UInt32;
  filterOperandId @1 :UInt32;
  biasOperandId @2 :UInt32;  # 0 = none
  strides @3 :List(UInt32);
  dilations @4 :List(UInt32);
  pads @5 :List(UInt32);
  groups @6 :UInt32;
  inputLayout @7 :Text;
}

struct Operation {
  union {
    conv2d @0 :Conv2d;
    add @1 :ElementWiseBinary;
    # ... more operations
  }
}

struct GraphInfo {
  operands @0 :List(Operand);
  operations @1 :List(Operation);
  constantData @2 :List(ConstantOperandData);
}
```

**Benefits:**
- **Zero-copy deserialization** - directly reference serialized data
- **Rust native** - excellent Rust support via `capnp` crate
- **No runtime dependencies** - generated code is pure Rust
- **Versioning built-in** - forward/backward compatibility
- **Faster than protobuf** - no parsing step
- **Type safe** - compile-time validation

**Implementation Path:**
1. Define Cap'n Proto schema (`schema/webnn.capnp`)
2. Generate Rust bindings at build time (`build.rs`)
3. Implement conversion: `GraphInfo` → Cap'n Proto → `GraphInfo`
4. Add IPC transport layer (Unix sockets, pipes, or TCP)
5. Keep JSON format as optional human-readable export

**Gradual Migration:**
- Phase 1: Add Cap'n Proto as parallel format (JSON still works)
- Phase 2: Use Cap'n Proto for internal IPC
- Phase 3: Optional - deprecate JSON for IPC (keep for debugging)

### Option 2: Protocol Buffers

Similar to ONNX protobuf but for graph IR.

**Benefits:**
- Already using protobuf for ONNX/CoreML conversion
- Well-known format
- Good tooling

**Drawbacks:**
- Parsing overhead (not zero-copy)
- More verbose than Cap'n Proto
- Requires protobuf runtime

**Would reuse existing infrastructure:**
```rust
// Already in build.rs for ONNX
prost_build::compile_protos(&["protos/webnn/graph.proto"], &["protos/"])?;
```

### Option 3: Typed Rust Enums (No Serialization)

Replace JSON with strongly-typed Rust enums in-process.

**Architecture:**
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Operation {
    Conv2d {
        input_operand_id: u32,
        filter_operand_id: u32,
        bias_operand_id: Option<u32>,
        strides: Vec<u32>,
        dilations: Vec<u32>,
        pads: Vec<u32>,
        groups: u32,
        input_layout: String,
    },
    Add {
        lhs_operand_id: u32,
        rhs_operand_id: u32,
    },
    // ... 50+ variants
}
```

**Benefits:**
- Type safety in-process
- No serialization overhead
- Exhaustive pattern matching
- Can still use Serde for JSON export

**Drawbacks:**
- No IPC support
- Large enum (50+ variants)
- Doesn't solve cross-process problem

## Recommended Approach: Cap'n Proto

For future IPC support, **Cap'n Proto is recommended** because:

1. **Rust-first design** - excellent Rust integration
2. **Zero-copy** - critical for large models
3. **Type safety** - structured validation
4. **No runtime** - pure generated code
5. **Platform agnostic** - not tied to Chromium

### Migration Strategy

**Phase 1: Parallel Format (Backwards Compatible)**
```rust
pub struct GraphInfo {
    // Current fields remain
    pub operands: Vec<Operand>,
    pub operations: Vec<Operation>,  // Still uses JSON attributes
    // ...
}

impl GraphInfo {
    // New: serialize to Cap'n Proto
    pub fn to_capnp(&self) -> capnp::message::Builder<capnp::message::HeapAllocator> {
        // Convert to Cap'n Proto format
    }

    // New: deserialize from Cap'n Proto
    pub fn from_capnp(reader: capnp::message::Reader) -> Result<Self, GraphError> {
        // Convert from Cap'n Proto format
    }

    // Existing JSON support unchanged
    pub fn to_json(&self) -> Result<String, GraphError> { ... }
    pub fn from_json(s: &str) -> Result<Self, GraphError> { ... }
}
```

**Phase 2: IPC Transport Layer**
```rust
// New module: src/ipc/mod.rs
pub struct GraphService {
    // Unix socket, pipe, or TCP listener
}

impl GraphService {
    pub fn serve(&self) -> Result<(), GraphError> {
        // Accept connections
        // Receive Cap'n Proto messages
        // Deserialize to GraphInfo
        // Execute operations
        // Send results back
    }
}

// Client side
pub struct GraphClient {
    // Connection to service
}

impl GraphClient {
    pub fn build_graph(&self, info: &GraphInfo) -> Result<GraphHandle, GraphError> {
        // Serialize to Cap'n Proto
        // Send over IPC
        // Receive handle
    }

    pub fn compute(&self, handle: GraphHandle, inputs: &[Tensor]) -> Result<Vec<Tensor>, GraphError> {
        // Send compute request over IPC
        // Receive results
    }
}
```

**Phase 3: Optional JSON Deprecation**
- Keep JSON for debugging and CLI tools
- Use Cap'n Proto exclusively for IPC
- Document migration path for users

## Process Model Options

### Option A: Separate Service Process (Chromium-like)

```
Client Process (Python/Rust)
    ↓ Cap'n Proto IPC
Service Process (Rust WebNN)
    ↓ Direct FFI
Backend (ONNX Runtime / CoreML / TensorRT)
```

**Benefits:**
- Isolates GPU/ML hardware failures
- Sandboxing possible
- Multiple clients can share service
- Resource pooling

**Use Cases:**
- Web browser integration
- Multi-tenant ML serving
- Fault isolation

### Option B: Worker Thread Pool (Simpler)

```
Main Thread (Python/Rust)
    ↓ Channel/Queue
Worker Thread Pool
    ↓ Direct calls
Backend (ONNX Runtime / CoreML / TensorRT)
```

**Benefits:**
- Simpler than multi-process
- Lower overhead
- Shared memory (no serialization within process)

**Use Cases:**
- Desktop applications
- ML tools/libraries
- Lower latency critical

### Option C: Hybrid (Flexible)

Support both in-process and IPC:
```rust
pub enum GraphExecutor {
    InProcess(DirectExecutor),      // Current implementation
    Worker(ThreadPoolExecutor),     // Thread pool
    Service(IpcExecutor),           // Separate process via Cap'n Proto
}
```

User chooses at runtime:
```rust
let executor = GraphExecutor::new_service()?; // IPC
let executor = GraphExecutor::new_worker(4)?; // 4 worker threads
let executor = GraphExecutor::new_direct()?;  // Current behavior
```

## Implementation Checklist

When adding IPC support:

- [ ] Choose serialization format (Cap'n Proto recommended)
- [ ] Define schema for all operations
  - [ ] Conv2d, ConvTranspose2d
  - [ ] Pool2d (Average, Max)
  - [ ] Normalization (Batch, Instance, Layer)
  - [ ] Element-wise operations
  - [ ] Reduction operations
  - [ ] Activation functions
  - [ ] Shape operations (Reshape, Transpose, etc.)
  - [ ] All other WebNN operations (50+ total)
- [ ] Add schema compilation to build.rs
- [ ] Implement GraphInfo ↔ Schema conversions
- [ ] Add transport layer (sockets/pipes)
- [ ] Implement service/client split
- [ ] Add authentication/security (if multi-user)
- [ ] Add resource limits and quotas
- [ ] Test serialization performance vs JSON
- [ ] Update Python bindings to support IPC mode
- [ ] Add IPC mode examples
- [ ] Document IPC setup and usage

## Performance Considerations

### Serialization Overhead

| Format | Serialize | Deserialize | Size | Zero-Copy |
|--------|-----------|-------------|------|-----------|
| JSON | ~1-5ms | ~2-10ms | Large | No |
| Protobuf | ~0.5-2ms | ~1-3ms | Medium | No |
| Cap'n Proto | ~0.1-0.5ms | ~0ms | Small | Yes |

*Estimates for typical WebNN graph (100 ops, 1MB constants)*

### When to Use IPC

**IPC is beneficial when:**
- Need process isolation (security/stability)
- Multiple clients sharing resources
- Different privilege levels required
- Large model sizes (reduces memory copies)

**In-process is better when:**
- Single user application
- Low latency critical (<1ms)
- Simple deployment requirements
- Development/debugging

## Security Considerations

If implementing IPC for multi-user scenarios:

1. **Authentication**
   - Token-based client authentication
   - Per-client resource quotas

2. **Sandboxing**
   - Run service with minimal privileges
   - Use seccomp/pledge to restrict syscalls

3. **Validation**
   - Validate all inputs at service boundary
   - Enforce memory limits on graphs
   - Rate limit requests

4. **Constant Data**
   - Validate constant operand sizes
   - Prevent memory exhaustion attacks
   - Consider shared memory for large constants

## References

- **Chromium WebNN Implementation:**
  - Mojo Interface: https://chromium.googlesource.com/chromium/src/+/lkgr/services/webnn/public/mojom/webnn_graph.mojom
  - Graph Implementation: https://chromium.googlesource.com/chromium/src/+/lkgr/services/webnn/webnn_graph_impl.h
  - Builder Implementation: https://chromium.googlesource.com/chromium/src/+/lkgr/services/webnn/webnn_graph_builder_impl.h

- **Cap'n Proto:**
  - Official Site: https://capnproto.org/
  - Rust Crate: https://crates.io/crates/capnp
  - Schema Language: https://capnproto.org/language.html

- **W3C WebNN Specification:**
  - Main Spec: https://www.w3.org/TR/webnn/
  - Device Selection: https://github.com/webmachinelearning/webnn/blob/main/device-selection-explainer.md

## Future Work

1. **Benchmark serialization formats** (JSON vs Protobuf vs Cap'n Proto)
2. **Design Cap'n Proto schema** for WebNN operations
3. **Implement parallel format support** (keep JSON, add Cap'n Proto)
4. **Add IPC transport layer** (Unix sockets for POSIX, named pipes for Windows)
5. **Update Python bindings** to support IPC mode
6. **Add service/client examples**
7. **Document migration path** for users
8. **Consider WebAssembly** integration (WASI sockets)

## Status

**Current:** Single-process with JSON attributes (adequate for current use cases)

**Future:** Multi-process with Cap'n Proto when IPC becomes necessary

This document will be updated as IPC requirements become clearer.
