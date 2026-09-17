#pragma once

#include <rustnn/rustnn.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace rustnn {

inline void initializeLogger() { rustnn_init_logger(); }

class Error : public std::runtime_error {
public:
  explicit Error(const std::string &message) : std::runtime_error(message) {}
};

inline void check(RustnnStatus status) {
  if (status != RustnnStatus_Success) {
    const char *message = rustnn_last_error_message();
    throw Error(message != nullptr ? message : "unknown rustnn error");
  }
}

enum class DataType {
  Float32 = RustnnDataType_Float32,
  Float16 = RustnnDataType_Float16,
  Int32 = RustnnDataType_Int32,
  Uint32 = RustnnDataType_Uint32,
  Int64 = RustnnDataType_Int64,
  Uint64 = RustnnDataType_Uint64,
  Int8 = RustnnDataType_Int8,
  Uint8 = RustnnDataType_Uint8,
  Int4 = RustnnDataType_Int4,
  Uint4 = RustnnDataType_Uint4,
};

enum class PowerPreference {
  Default = RustnnPowerPreference_Default,
  HighPerformance = RustnnPowerPreference_HighPerformance,
  LowPower = RustnnPowerPreference_LowPower,
};

enum class Backend {
  Automatic = RustnnBackend_Automatic,
  Onnx = RustnnBackend_Onnx,
  Trtx = RustnnBackend_Trtx,
  Coreml = RustnnBackend_Coreml,
  Litert = RustnnBackend_Litert,
  Cann = RustnnBackend_Cann,
};

enum class DeviceType {
  Cpu = RustnnDeviceType_Cpu,
  Gpu = RustnnDeviceType_Gpu,
  Npu = RustnnDeviceType_Npu,
};

struct BackendDevice {
  Backend backend;
  DeviceType device_type;
  std::size_t device_index = 0;
};

struct TrtxOptions {
  bool engine_caching = true;
  bool runtime_cache = true;
  bool fail_on_cache_miss = false;
  bool cuda_graphs = true;
};

struct RustNNOptions {
  TrtxOptions trtx;
};

struct ContextOptions {
  PowerPreference power_preference = PowerPreference::Default;
  bool accelerated = true;
  Backend backend_hint = Backend::Automatic;
  std::optional<BackendDevice> device_hint;
  std::optional<RustNNOptions> rustnn_options;
};

struct OperatorOptions {
  std::string label;
};

class OperandDescriptor {
public:
  OperandDescriptor(DataType data_type, const std::vector<std::uint64_t> &shape) {
    check(rustnn_operand_descriptor_create(
        static_cast<RustnnDataType>(data_type), shape.data(), shape.size(), &value_));
  }

  ~OperandDescriptor() { rustnn_operand_descriptor_destroy(value_); }
  OperandDescriptor(const OperandDescriptor &) = delete;
  OperandDescriptor &operator=(const OperandDescriptor &) = delete;

  OperandDescriptor(OperandDescriptor &&other) noexcept
      : value_(std::exchange(other.value_, nullptr)) {}

  OperandDescriptor &operator=(OperandDescriptor &&other) noexcept {
    if (this != &other) {
      rustnn_operand_descriptor_destroy(value_);
      value_ = std::exchange(other.value_, nullptr);
    }
    return *this;
  }

private:
  friend class GraphBuilder;
  RustnnOperandDescriptor *value_ = nullptr;
};

class TensorDescriptor {
public:
  TensorDescriptor(DataType data_type, const std::vector<std::uint64_t> &shape,
                   bool readable, bool writable) {
    check(rustnn_tensor_descriptor_create(static_cast<RustnnDataType>(data_type),
                                          shape.data(), shape.size(), readable,
                                          writable, &value_));
  }

  ~TensorDescriptor() { rustnn_tensor_descriptor_destroy(value_); }
  TensorDescriptor(const TensorDescriptor &) = delete;
  TensorDescriptor &operator=(const TensorDescriptor &) = delete;

  TensorDescriptor(TensorDescriptor &&other) noexcept
      : value_(std::exchange(other.value_, nullptr)) {}

  TensorDescriptor &operator=(TensorDescriptor &&other) noexcept {
    if (this != &other) {
      rustnn_tensor_descriptor_destroy(value_);
      value_ = std::exchange(other.value_, nullptr);
    }
    return *this;
  }

private:
  friend class Context;
  RustnnTensorDescriptor *value_ = nullptr;
};

class Operand {
public:
  ~Operand() { rustnn_operand_destroy(value_); }
  Operand(const Operand &) = delete;
  Operand &operator=(const Operand &) = delete;

  Operand(Operand &&other) noexcept : value_(std::exchange(other.value_, nullptr)) {}

  Operand &operator=(Operand &&other) noexcept {
    if (this != &other) {
      rustnn_operand_destroy(value_);
      value_ = std::exchange(other.value_, nullptr);
    }
    return *this;
  }

private:
  friend class GraphBuilder;
  explicit Operand(RustnnOperand *value) : value_(value) {}
  RustnnOperand *value_ = nullptr;
};

class Graph {
public:
  ~Graph() { rustnn_graph_destroy(value_); }
  Graph(const Graph &) = delete;
  Graph &operator=(const Graph &) = delete;

  Graph(Graph &&other) noexcept : value_(std::exchange(other.value_, nullptr)) {}

  Graph &operator=(Graph &&other) noexcept {
    if (this != &other) {
      rustnn_graph_destroy(value_);
      value_ = std::exchange(other.value_, nullptr);
    }
    return *this;
  }

private:
  friend class Context;
  friend class GraphBuilder;
  explicit Graph(RustnnGraph *value) : value_(value) {}
  RustnnGraph *value_ = nullptr;
};

class Tensor {
public:
  ~Tensor() { rustnn_tensor_destroy(value_); }
  Tensor(const Tensor &) = delete;
  Tensor &operator=(const Tensor &) = delete;

  Tensor(Tensor &&other) noexcept : value_(std::exchange(other.value_, nullptr)) {}

  Tensor &operator=(Tensor &&other) noexcept {
    if (this != &other) {
      rustnn_tensor_destroy(value_);
      value_ = std::exchange(other.value_, nullptr);
    }
    return *this;
  }

private:
  friend class Context;
  explicit Tensor(RustnnTensor *value) : value_(value) {}
  RustnnTensor *value_ = nullptr;
};

class Context {
public:
  explicit Context(const ContextOptions &options = {}) {
    const RustnnBackendDevice raw_device_hint{
        options.device_hint.has_value()
            ? static_cast<RustnnBackend>(options.device_hint->backend)
            : RustnnBackend_Automatic,
        options.device_hint.has_value()
            ? static_cast<RustnnDeviceType>(options.device_hint->device_type)
            : RustnnDeviceType_Cpu,
        options.device_hint.has_value() ? options.device_hint->device_index : 0};
    const TrtxOptions trtx = options.rustnn_options.has_value()
                                  ? options.rustnn_options->trtx
                                  : TrtxOptions{};
    const RustnnOptions raw_rustnn_options{{
        trtx.engine_caching,
        trtx.runtime_cache,
        trtx.fail_on_cache_miss,
        trtx.cuda_graphs,
    }};
    const RustnnContextOptions raw_options{
        static_cast<RustnnPowerPreference>(options.power_preference),
        options.accelerated,
        static_cast<RustnnBackend>(options.backend_hint),
        options.device_hint.has_value(),
        raw_device_hint,
        options.rustnn_options.has_value(),
        raw_rustnn_options};
    check(rustnn_context_create(&raw_options, &value_));
  }

  ~Context() { rustnn_context_destroy(value_); }
  Context(const Context &) = delete;
  Context &operator=(const Context &) = delete;

  Context(Context &&other) noexcept : value_(std::exchange(other.value_, nullptr)) {}

  Context &operator=(Context &&other) noexcept {
    if (this != &other) {
      rustnn_context_destroy(value_);
      value_ = std::exchange(other.value_, nullptr);
    }
    return *this;
  }

  Tensor createTensor(const TensorDescriptor &descriptor) {
    RustnnTensor *tensor = nullptr;
    check(rustnn_context_create_tensor(value_, descriptor.value_, &tensor));
    return Tensor(tensor);
  }

  template <typename T>
  void writeTensor(const Tensor &tensor, const std::vector<T> &data) {
    static_assert(std::is_trivially_copyable_v<T>, "tensor elements must be plain data");
    check(rustnn_context_write_tensor(value_, tensor.value_, data.data(),
                                      data.size() * sizeof(T)));
  }

  template <typename T>
  std::vector<T> readTensor(const Tensor &tensor, std::size_t element_count) {
    static_assert(std::is_trivially_copyable_v<T>, "tensor elements must be plain data");
    std::vector<T> data(element_count);
    check(rustnn_context_read_tensor(value_, tensor.value_, data.data(),
                                     data.size() * sizeof(T)));
    return data;
  }

  void dispatch(Graph &graph,
                const std::vector<std::pair<std::string, const Tensor *>> &inputs,
                const std::vector<std::pair<std::string, const Tensor *>> &outputs) {
    std::vector<RustnnNamedTensor> named_inputs;
    named_inputs.reserve(inputs.size());
    for (const auto &[name, tensor] : inputs) {
      named_inputs.push_back({name.c_str(), tensor != nullptr ? tensor->value_ : nullptr});
    }
    std::vector<RustnnNamedTensor> named_outputs;
    named_outputs.reserve(outputs.size());
    for (const auto &[name, tensor] : outputs) {
      named_outputs.push_back({name.c_str(), tensor != nullptr ? tensor->value_ : nullptr});
    }
    check(rustnn_context_dispatch(value_, graph.value_, named_inputs.data(),
                                  named_inputs.size(), named_outputs.data(),
                                  named_outputs.size()));
  }

private:
  friend class GraphBuilder;
  RustnnContext *value_ = nullptr;
};

class GraphBuilder {
public:
  GraphBuilder() { check(rustnn_graph_builder_create_uncompiled(&value_)); }
  explicit GraphBuilder(Context &context) {
    check(rustnn_graph_builder_create(context.value_, &value_));
  }
  ~GraphBuilder() { rustnn_graph_builder_destroy(value_); }
  GraphBuilder(const GraphBuilder &) = delete;
  GraphBuilder &operator=(const GraphBuilder &) = delete;

  GraphBuilder(GraphBuilder &&other) noexcept
      : value_(std::exchange(other.value_, nullptr)) {}

  GraphBuilder &operator=(GraphBuilder &&other) noexcept {
    if (this != &other) {
      rustnn_graph_builder_destroy(value_);
      value_ = std::exchange(other.value_, nullptr);
    }
    return *this;
  }

  Operand input(const std::string &name, const OperandDescriptor &descriptor) {
    RustnnOperand *output = nullptr;
    check(rustnn_graph_builder_input(value_, name.c_str(), descriptor.value_, &output));
    return Operand(output);
  }

  template <typename T>
  Operand constant(const OperandDescriptor &descriptor, const std::vector<T> &data) {
    static_assert(std::is_trivially_copyable_v<T>, "tensor elements must be plain data");
    RustnnOperand *output = nullptr;
    check(rustnn_graph_builder_constant(value_, descriptor.value_, data.data(),
                                        data.size() * sizeof(T), &output));
    return Operand(output);
  }

  Operand unary(RustnnUnaryOperation operation, const Operand &input) {
    RustnnOperand *output = nullptr;
    check(rustnn_graph_builder_unary(value_, operation, input.value_, &output));
    return Operand(output);
  }

  Operand unary(RustnnUnaryOperation operation, const Operand &input,
                const OperatorOptions &options) {
    RustnnOperand *output = nullptr;
    const RustnnOperatorOptions raw_options{options.label.c_str()};
    check(rustnn_graph_builder_unary_with_options(
        value_, operation, input.value_, &raw_options, &output));
    return Operand(output);
  }

  Operand binary(RustnnBinaryOperation operation, const Operand &lhs,
                 const Operand &rhs) {
    RustnnOperand *output = nullptr;
    check(rustnn_graph_builder_binary(value_, operation, lhs.value_, rhs.value_, &output));
    return Operand(output);
  }

  Operand binary(RustnnBinaryOperation operation, const Operand &lhs,
                 const Operand &rhs, const OperatorOptions &options) {
    RustnnOperand *output = nullptr;
    const RustnnOperatorOptions raw_options{options.label.c_str()};
    check(rustnn_graph_builder_binary_with_options(
        value_, operation, lhs.value_, rhs.value_, &raw_options, &output));
    return Operand(output);
  }

  Operand add(const Operand &lhs, const Operand &rhs) {
    RustnnOperand *output = nullptr;
    check(rustnn_graph_builder_add(value_, lhs.value_, rhs.value_, &output));
    return Operand(output);
  }

  Operand add(const Operand &lhs, const Operand &rhs,
              const OperatorOptions &options) {
    RustnnOperand *output = nullptr;
    const RustnnOperatorOptions raw_options{options.label.c_str()};
    check(rustnn_graph_builder_add_with_options(value_, lhs.value_, rhs.value_,
                                                &raw_options, &output));
    return Operand(output);
  }

  Operand subtract(const Operand &lhs, const Operand &rhs) {
    RustnnOperand *output = nullptr;
    check(rustnn_graph_builder_subtract(value_, lhs.value_, rhs.value_, &output));
    return Operand(output);
  }

  Operand subtract(const Operand &lhs, const Operand &rhs,
                   const OperatorOptions &options) {
    RustnnOperand *output = nullptr;
    const RustnnOperatorOptions raw_options{options.label.c_str()};
    check(rustnn_graph_builder_subtract_with_options(
        value_, lhs.value_, rhs.value_, &raw_options, &output));
    return Operand(output);
  }

  Operand multiply(const Operand &lhs, const Operand &rhs) {
    RustnnOperand *output = nullptr;
    check(rustnn_graph_builder_multiply(value_, lhs.value_, rhs.value_, &output));
    return Operand(output);
  }

  Operand multiply(const Operand &lhs, const Operand &rhs,
                   const OperatorOptions &options) {
    RustnnOperand *output = nullptr;
    const RustnnOperatorOptions raw_options{options.label.c_str()};
    check(rustnn_graph_builder_multiply_with_options(
        value_, lhs.value_, rhs.value_, &raw_options, &output));
    return Operand(output);
  }

  Operand divide(const Operand &lhs, const Operand &rhs) {
    RustnnOperand *output = nullptr;
    check(rustnn_graph_builder_divide(value_, lhs.value_, rhs.value_, &output));
    return Operand(output);
  }

  Operand divide(const Operand &lhs, const Operand &rhs,
                 const OperatorOptions &options) {
    RustnnOperand *output = nullptr;
    const RustnnOperatorOptions raw_options{options.label.c_str()};
    check(rustnn_graph_builder_divide_with_options(
        value_, lhs.value_, rhs.value_, &raw_options, &output));
    return Operand(output);
  }

  Operand relu(const Operand &input) {
    RustnnOperand *output = nullptr;
    check(rustnn_graph_builder_relu(value_, input.value_, &output));
    return Operand(output);
  }

  Operand relu(const Operand &input, const OperatorOptions &options) {
    RustnnOperand *output = nullptr;
    const RustnnOperatorOptions raw_options{options.label.c_str()};
    check(rustnn_graph_builder_relu_with_options(value_, input.value_, &raw_options,
                                                 &output));
    return Operand(output);
  }

  std::vector<std::uint64_t> shape(const Operand &operand) {
    std::size_t rank = 0;
    check(rustnn_graph_builder_operand_shape(value_, operand.value_, nullptr, 0, &rank));
    std::vector<std::uint64_t> dimensions(rank);
    check(rustnn_graph_builder_operand_shape(value_, operand.value_, dimensions.data(),
                                             dimensions.size(), &rank));
    return dimensions;
  }

  DataType dataType(const Operand &operand) {
    RustnnDataType data_type = RustnnDataType_Float32;
    check(rustnn_graph_builder_operand_data_type(value_, operand.value_, &data_type));
    return static_cast<DataType>(data_type);
  }

  std::string webnnText(
      const std::vector<std::pair<std::string, const Operand *>> &outputs) const {
    std::vector<RustnnNamedOperand> named;
    named.reserve(outputs.size());
    for (const auto &[name, operand] : outputs) {
      named.push_back({name.c_str(), operand != nullptr ? operand->value_ : nullptr});
    }
    char *text = nullptr;
    check(rustnn_graph_builder_webnn_text(value_, named.data(), named.size(), &text));
    std::string result(text);
    rustnn_string_destroy(text);
    return result;
  }

  Graph build(const std::vector<std::pair<std::string, const Operand *>> &outputs) {
    std::vector<RustnnNamedOperand> named;
    named.reserve(outputs.size());
    for (const auto &[name, operand] : outputs) {
      named.push_back({name.c_str(), operand != nullptr ? operand->value_ : nullptr});
    }
    RustnnGraph *graph = nullptr;
    RustnnStatus status =
        rustnn_graph_builder_build(&value_, named.data(), named.size(), &graph);
    check(status);
    return Graph(graph);
  }

private:
  RustnnGraphBuilder *value_ = nullptr;
};

} // namespace rustnn
