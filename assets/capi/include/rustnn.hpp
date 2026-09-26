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

enum class MLOperandDataType {
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

enum class MLPowerPreference {
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

struct MLContextOptions {
  MLPowerPreference power_preference = MLPowerPreference::Default;
  bool accelerated = true;
  Backend backend_hint = Backend::Automatic;
  std::optional<BackendDevice> device_hint;
  std::optional<RustNNOptions> rustnn_options;
};

struct MLOperatorOptions {
  std::string label;
};

class MLOperandDescriptor {
public:
  MLOperandDescriptor(MLOperandDataType data_type, const std::vector<std::uint64_t> &shape) {
    check(rustnn_operand_descriptor_create(
        static_cast<RustnnDataType>(data_type), shape.data(), shape.size(), &value_));
  }

  ~MLOperandDescriptor() { rustnn_operand_descriptor_destroy(value_); }
  MLOperandDescriptor(const MLOperandDescriptor &) = delete;
  MLOperandDescriptor &operator=(const MLOperandDescriptor &) = delete;

  MLOperandDescriptor(MLOperandDescriptor &&other) noexcept
      : value_(std::exchange(other.value_, nullptr)) {}

  MLOperandDescriptor &operator=(MLOperandDescriptor &&other) noexcept {
    if (this != &other) {
      rustnn_operand_descriptor_destroy(value_);
      value_ = std::exchange(other.value_, nullptr);
    }
    return *this;
  }

private:
  friend class MLGraphBuilder;
  RustnnOperandDescriptor *value_ = nullptr;
};

class MLTensorDescriptor {
public:
  MLTensorDescriptor(MLOperandDataType data_type, const std::vector<std::uint64_t> &shape,
                   bool readable, bool writable) {
    check(rustnn_tensor_descriptor_create(static_cast<RustnnDataType>(data_type),
                                          shape.data(), shape.size(), readable,
                                          writable, &value_));
  }

  ~MLTensorDescriptor() { rustnn_tensor_descriptor_destroy(value_); }
  MLTensorDescriptor(const MLTensorDescriptor &) = delete;
  MLTensorDescriptor &operator=(const MLTensorDescriptor &) = delete;

  MLTensorDescriptor(MLTensorDescriptor &&other) noexcept
      : value_(std::exchange(other.value_, nullptr)) {}

  MLTensorDescriptor &operator=(MLTensorDescriptor &&other) noexcept {
    if (this != &other) {
      rustnn_tensor_descriptor_destroy(value_);
      value_ = std::exchange(other.value_, nullptr);
    }
    return *this;
  }

private:
  friend class MLContext;
  RustnnTensorDescriptor *value_ = nullptr;
};

class MLOperand {
public:
  ~MLOperand() { rustnn_operand_destroy(value_); }
  MLOperand(const MLOperand &) = delete;
  MLOperand &operator=(const MLOperand &) = delete;

  MLOperand(MLOperand &&other) noexcept : value_(std::exchange(other.value_, nullptr)) {}

  MLOperand &operator=(MLOperand &&other) noexcept {
    if (this != &other) {
      rustnn_operand_destroy(value_);
      value_ = std::exchange(other.value_, nullptr);
    }
    return *this;
  }

private:
  friend class MLGraphBuilder;
  explicit MLOperand(RustnnOperand *value) : value_(value) {}
  RustnnOperand *value_ = nullptr;
};

class MLGraph {
public:
  ~MLGraph() { rustnn_graph_destroy(value_); }
  MLGraph(const MLGraph &) = delete;
  MLGraph &operator=(const MLGraph &) = delete;

  MLGraph(MLGraph &&other) noexcept : value_(std::exchange(other.value_, nullptr)) {}

  MLGraph &operator=(MLGraph &&other) noexcept {
    if (this != &other) {
      rustnn_graph_destroy(value_);
      value_ = std::exchange(other.value_, nullptr);
    }
    return *this;
  }

private:
  friend class MLContext;
  friend class MLGraphBuilder;
  explicit MLGraph(RustnnGraph *value) : value_(value) {}
  RustnnGraph *value_ = nullptr;
};

class MLTensor {
public:
  ~MLTensor() { rustnn_tensor_destroy(value_); }
  MLTensor(const MLTensor &) = delete;
  MLTensor &operator=(const MLTensor &) = delete;

  MLTensor(MLTensor &&other) noexcept : value_(std::exchange(other.value_, nullptr)) {}

  MLTensor &operator=(MLTensor &&other) noexcept {
    if (this != &other) {
      rustnn_tensor_destroy(value_);
      value_ = std::exchange(other.value_, nullptr);
    }
    return *this;
  }

private:
  friend class MLContext;
  explicit MLTensor(RustnnTensor *value) : value_(value) {}
  RustnnTensor *value_ = nullptr;
};

class MLContext {
public:
  explicit MLContext(const MLContextOptions &options = {}) {
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

  ~MLContext() { rustnn_context_destroy(value_); }
  MLContext(const MLContext &) = delete;
  MLContext &operator=(const MLContext &) = delete;

  MLContext(MLContext &&other) noexcept : value_(std::exchange(other.value_, nullptr)) {}

  MLContext &operator=(MLContext &&other) noexcept {
    if (this != &other) {
      rustnn_context_destroy(value_);
      value_ = std::exchange(other.value_, nullptr);
    }
    return *this;
  }

  MLTensor createTensor(const MLTensorDescriptor &descriptor) {
    RustnnTensor *tensor = nullptr;
    check(rustnn_context_create_tensor(value_, descriptor.value_, &tensor));
    return MLTensor(tensor);
  }

  template <typename T>
  void writeTensor(const MLTensor &tensor, const std::vector<T> &data) {
    static_assert(std::is_trivially_copyable_v<T>, "tensor elements must be plain data");
    check(rustnn_context_write_tensor(value_, tensor.value_, data.data(),
                                      data.size() * sizeof(T)));
  }

  template <typename T>
  std::vector<T> readTensor(const MLTensor &tensor, std::size_t element_count) {
    static_assert(std::is_trivially_copyable_v<T>, "tensor elements must be plain data");
    std::vector<T> data(element_count);
    check(rustnn_context_read_tensor(value_, tensor.value_, data.data(),
                                     data.size() * sizeof(T)));
    return data;
  }

  void dispatch(MLGraph &graph,
                const std::vector<std::pair<std::string, const MLTensor *>> &inputs,
                const std::vector<std::pair<std::string, const MLTensor *>> &outputs) {
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
  friend class MLGraphBuilder;
  RustnnContext *value_ = nullptr;
};

#define RUSTNN_ML_UNARY_METHOD(method, function)                              \
  MLOperand method(const MLOperand &input) {                                  \
    RustnnOperand *output = nullptr;                                        \
    check(function(value_, input.value_, &output));                           \
    return MLOperand(output);                                                 \
  }

#define RUSTNN_ML_UNARY_METHOD_WITH_OPTIONS(method, function)                 \
  RUSTNN_ML_UNARY_METHOD(method, function)                                    \
  MLOperand method(const MLOperand &input,                                    \
                   const MLOperatorOptions &options) {                        \
    RustnnOperand *output = nullptr;                                        \
    const RustnnOperatorOptions raw_options{options.label.c_str()};          \
    check(function##_with_options(value_, input.value_, &raw_options,         \
                                  &output));                                  \
    return MLOperand(output);                                                 \
  }

#define RUSTNN_ML_BINARY_METHOD(method, function)                             \
  MLOperand method(const MLOperand &lhs, const MLOperand &rhs) {              \
    RustnnOperand *output = nullptr;                                        \
    check(function(value_, lhs.value_, rhs.value_, &output));                 \
    return MLOperand(output);                                                 \
  }

#define RUSTNN_ML_BINARY_METHOD_WITH_OPTIONS(method, function)                \
  RUSTNN_ML_BINARY_METHOD(method, function)                                   \
  MLOperand method(const MLOperand &lhs, const MLOperand &rhs,                \
                   const MLOperatorOptions &options) {                        \
    RustnnOperand *output = nullptr;                                        \
    const RustnnOperatorOptions raw_options{options.label.c_str()};          \
    check(function##_with_options(value_, lhs.value_, rhs.value_,             \
                                  &raw_options, &output));                     \
    return MLOperand(output);                                                 \
  }

#define RUSTNN_ML_TERNARY_METHOD(method, function)                            \
  MLOperand method(const MLOperand &first, const MLOperand &second,           \
                   const MLOperand &third) {                                  \
    RustnnOperand *output = nullptr;                                        \
    check(function(value_, first.value_, second.value_, third.value_,         \
                   &output));                                                 \
    return MLOperand(output);                                                 \
  }

class MLGraphBuilder {
public:
  MLGraphBuilder() { check(rustnn_graph_builder_create_uncompiled(&value_)); }
  explicit MLGraphBuilder(MLContext &context) {
    check(rustnn_graph_builder_create(context.value_, &value_));
  }
  ~MLGraphBuilder() { rustnn_graph_builder_destroy(value_); }
  MLGraphBuilder(const MLGraphBuilder &) = delete;
  MLGraphBuilder &operator=(const MLGraphBuilder &) = delete;

  MLGraphBuilder(MLGraphBuilder &&other) noexcept
      : value_(std::exchange(other.value_, nullptr)) {}

  MLGraphBuilder &operator=(MLGraphBuilder &&other) noexcept {
    if (this != &other) {
      rustnn_graph_builder_destroy(value_);
      value_ = std::exchange(other.value_, nullptr);
    }
    return *this;
  }

  MLOperand input(const std::string &name, const MLOperandDescriptor &descriptor) {
    RustnnOperand *output = nullptr;
    check(rustnn_graph_builder_input(value_, name.c_str(), descriptor.value_, &output));
    return MLOperand(output);
  }

  template <typename T>
  MLOperand constant(const MLOperandDescriptor &descriptor, const std::vector<T> &data) {
    static_assert(std::is_trivially_copyable_v<T>, "tensor elements must be plain data");
    RustnnOperand *output = nullptr;
    check(rustnn_graph_builder_constant(value_, descriptor.value_, data.data(),
                                        data.size() * sizeof(T), &output));
    return MLOperand(output);
  }

  MLOperand unary(RustnnUnaryOperation operation, const MLOperand &input) {
    RustnnOperand *output = nullptr;
    check(rustnn_graph_builder_unary(value_, operation, input.value_, &output));
    return MLOperand(output);
  }

  MLOperand unary(RustnnUnaryOperation operation, const MLOperand &input,
                const MLOperatorOptions &options) {
    RustnnOperand *output = nullptr;
    const RustnnOperatorOptions raw_options{options.label.c_str()};
    check(rustnn_graph_builder_unary_with_options(
        value_, operation, input.value_, &raw_options, &output));
    return MLOperand(output);
  }

  MLOperand binary(RustnnBinaryOperation operation, const MLOperand &lhs,
                 const MLOperand &rhs) {
    RustnnOperand *output = nullptr;
    check(rustnn_graph_builder_binary(value_, operation, lhs.value_, rhs.value_, &output));
    return MLOperand(output);
  }

  MLOperand binary(RustnnBinaryOperation operation, const MLOperand &lhs,
                 const MLOperand &rhs, const MLOperatorOptions &options) {
    RustnnOperand *output = nullptr;
    const RustnnOperatorOptions raw_options{options.label.c_str()};
    check(rustnn_graph_builder_binary_with_options(
        value_, operation, lhs.value_, rhs.value_, &raw_options, &output));
    return MLOperand(output);
  }

  RUSTNN_ML_BINARY_METHOD_WITH_OPTIONS(add, rustnn_graph_builder_add)
  RUSTNN_ML_BINARY_METHOD_WITH_OPTIONS(subtract, rustnn_graph_builder_subtract)
  RUSTNN_ML_BINARY_METHOD_WITH_OPTIONS(multiply, rustnn_graph_builder_multiply)
  RUSTNN_ML_BINARY_METHOD_WITH_OPTIONS(divide, rustnn_graph_builder_divide)
  RUSTNN_ML_BINARY_METHOD_WITH_OPTIONS(power, rustnn_graph_builder_power)
  RUSTNN_ML_BINARY_METHOD_WITH_OPTIONS(maximum, rustnn_graph_builder_maximum)
  RUSTNN_ML_BINARY_METHOD_WITH_OPTIONS(minimum, rustnn_graph_builder_minimum)
  RUSTNN_ML_BINARY_METHOD_WITH_OPTIONS(equal, rustnn_graph_builder_equal)
  RUSTNN_ML_BINARY_METHOD_WITH_OPTIONS(greater, rustnn_graph_builder_greater)
  RUSTNN_ML_BINARY_METHOD_WITH_OPTIONS(greaterOrEqual,
                                       rustnn_graph_builder_greater_or_equal)
  RUSTNN_ML_BINARY_METHOD_WITH_OPTIONS(lesser, rustnn_graph_builder_lesser)
  RUSTNN_ML_BINARY_METHOD_WITH_OPTIONS(lesserOrEqual,
                                       rustnn_graph_builder_lesser_or_equal)
  RUSTNN_ML_BINARY_METHOD_WITH_OPTIONS(notEqual, rustnn_graph_builder_not_equal)
  RUSTNN_ML_BINARY_METHOD_WITH_OPTIONS(logicalAnd,
                                       rustnn_graph_builder_logical_and)
  RUSTNN_ML_BINARY_METHOD_WITH_OPTIONS(logicalOr,
                                       rustnn_graph_builder_logical_or)
  RUSTNN_ML_BINARY_METHOD_WITH_OPTIONS(logicalXor,
                                       rustnn_graph_builder_logical_xor)
  RUSTNN_ML_BINARY_METHOD_WITH_OPTIONS(matmul, rustnn_graph_builder_matmul)

  RUSTNN_ML_UNARY_METHOD_WITH_OPTIONS(abs, rustnn_graph_builder_abs)
  RUSTNN_ML_UNARY_METHOD_WITH_OPTIONS(roundEven, rustnn_graph_builder_round_even)
  RUSTNN_ML_UNARY_METHOD_WITH_OPTIONS(ceil, rustnn_graph_builder_ceil)
  RUSTNN_ML_UNARY_METHOD_WITH_OPTIONS(cos, rustnn_graph_builder_cos)
  RUSTNN_ML_UNARY_METHOD_WITH_OPTIONS(exp, rustnn_graph_builder_exp)
  RUSTNN_ML_UNARY_METHOD_WITH_OPTIONS(floor, rustnn_graph_builder_floor)
  RUSTNN_ML_UNARY_METHOD_WITH_OPTIONS(gelu, rustnn_graph_builder_gelu)
  RUSTNN_ML_UNARY_METHOD_WITH_OPTIONS(log, rustnn_graph_builder_log)
  RUSTNN_ML_UNARY_METHOD_WITH_OPTIONS(negate, rustnn_graph_builder_negate)
  RUSTNN_ML_UNARY_METHOD_WITH_OPTIONS(relu, rustnn_graph_builder_relu)
  RUSTNN_ML_UNARY_METHOD_WITH_OPTIONS(sigmoid, rustnn_graph_builder_sigmoid)
  RUSTNN_ML_UNARY_METHOD_WITH_OPTIONS(sin, rustnn_graph_builder_sin)
  RUSTNN_ML_UNARY_METHOD_WITH_OPTIONS(sqrt, rustnn_graph_builder_sqrt)
  RUSTNN_ML_UNARY_METHOD_WITH_OPTIONS(tan, rustnn_graph_builder_tan)
  RUSTNN_ML_UNARY_METHOD_WITH_OPTIONS(tanh, rustnn_graph_builder_tanh)
  RUSTNN_ML_UNARY_METHOD_WITH_OPTIONS(erf, rustnn_graph_builder_erf)
  RUSTNN_ML_UNARY_METHOD_WITH_OPTIONS(reciprocal,
                                      rustnn_graph_builder_reciprocal)
  RUSTNN_ML_UNARY_METHOD_WITH_OPTIONS(sign, rustnn_graph_builder_sign)
  RUSTNN_ML_UNARY_METHOD_WITH_OPTIONS(logicalNot,
                                      rustnn_graph_builder_logical_not)
  RUSTNN_ML_UNARY_METHOD_WITH_OPTIONS(identity, rustnn_graph_builder_identity)
  RUSTNN_ML_UNARY_METHOD_WITH_OPTIONS(softplus, rustnn_graph_builder_softplus)
  RUSTNN_ML_UNARY_METHOD_WITH_OPTIONS(softsign, rustnn_graph_builder_softsign)
  RUSTNN_ML_UNARY_METHOD_WITH_OPTIONS(isNaN, rustnn_graph_builder_is_nan)
  RUSTNN_ML_UNARY_METHOD_WITH_OPTIONS(isInfinite,
                                      rustnn_graph_builder_is_infinite)
  RUSTNN_ML_UNARY_METHOD_WITH_OPTIONS(shape, rustnn_graph_builder_shape)

  RUSTNN_ML_UNARY_METHOD(elu, rustnn_graph_builder_elu)
  RUSTNN_ML_UNARY_METHOD(hardSigmoid, rustnn_graph_builder_hard_sigmoid)
  RUSTNN_ML_UNARY_METHOD(hardSwish, rustnn_graph_builder_hard_swish)
  RUSTNN_ML_UNARY_METHOD(leakyRelu, rustnn_graph_builder_leaky_relu)
  RUSTNN_ML_UNARY_METHOD(linear, rustnn_graph_builder_linear)
  RUSTNN_ML_UNARY_METHOD(clamp, rustnn_graph_builder_clamp)
  RUSTNN_ML_UNARY_METHOD(instanceNormalization,
                         rustnn_graph_builder_instance_normalization)
  RUSTNN_ML_UNARY_METHOD(layerNormalization,
                         rustnn_graph_builder_layer_normalization)
  RUSTNN_ML_UNARY_METHOD(resample2d, rustnn_graph_builder_resample2d)
  RUSTNN_ML_UNARY_METHOD(reverse, rustnn_graph_builder_reverse)
  RUSTNN_ML_UNARY_METHOD(triangular, rustnn_graph_builder_triangular)
  RUSTNN_ML_UNARY_METHOD(averagePool2d, rustnn_graph_builder_average_pool2d)
  RUSTNN_ML_UNARY_METHOD(maxPool2d, rustnn_graph_builder_max_pool2d)
  RUSTNN_ML_UNARY_METHOD(l2Pool2d, rustnn_graph_builder_l2_pool2d)
  RUSTNN_ML_UNARY_METHOD(globalAveragePool,
                         rustnn_graph_builder_global_average_pool)
  RUSTNN_ML_UNARY_METHOD(globalMaxPool, rustnn_graph_builder_global_max_pool)
  RUSTNN_ML_UNARY_METHOD(reduceSum, rustnn_graph_builder_reduce_sum)
  RUSTNN_ML_UNARY_METHOD(reduceMean, rustnn_graph_builder_reduce_mean)
  RUSTNN_ML_UNARY_METHOD(reduceMax, rustnn_graph_builder_reduce_max)
  RUSTNN_ML_UNARY_METHOD(reduceMin, rustnn_graph_builder_reduce_min)
  RUSTNN_ML_UNARY_METHOD(reduceProduct, rustnn_graph_builder_reduce_product)
  RUSTNN_ML_UNARY_METHOD(reduceL1, rustnn_graph_builder_reduce_l1)
  RUSTNN_ML_UNARY_METHOD(reduceL2, rustnn_graph_builder_reduce_l2)
  RUSTNN_ML_UNARY_METHOD(reduceLogSum, rustnn_graph_builder_reduce_log_sum)
  RUSTNN_ML_UNARY_METHOD(reduceLogSumExp,
                         rustnn_graph_builder_reduce_log_sum_exp)
  RUSTNN_ML_UNARY_METHOD(reduceSumSquare,
                         rustnn_graph_builder_reduce_sum_square)
  RUSTNN_ML_UNARY_METHOD(transpose, rustnn_graph_builder_transpose)
  RUSTNN_ML_UNARY_METHOD(squeeze, rustnn_graph_builder_squeeze)
  RUSTNN_ML_UNARY_METHOD(unsqueeze, rustnn_graph_builder_unsqueeze)

  RUSTNN_ML_BINARY_METHOD(gemm, rustnn_graph_builder_gemm)
  RUSTNN_ML_BINARY_METHOD(conv2d, rustnn_graph_builder_conv2d)
  RUSTNN_ML_BINARY_METHOD(convTranspose2d,
                          rustnn_graph_builder_conv_transpose2d)
  RUSTNN_ML_BINARY_METHOD(gather, rustnn_graph_builder_gather)
  RUSTNN_ML_BINARY_METHOD(gatherElements,
                          rustnn_graph_builder_gather_elements)
  RUSTNN_ML_BINARY_METHOD(gatherND, rustnn_graph_builder_gather_nd)
  RUSTNN_ML_BINARY_METHOD(prelu, rustnn_graph_builder_prelu)

  RUSTNN_ML_TERNARY_METHOD(batchNormalization,
                           rustnn_graph_builder_batch_normalization)
  RUSTNN_ML_TERNARY_METHOD(where, rustnn_graph_builder_where)
  RUSTNN_ML_TERNARY_METHOD(scatterElements,
                           rustnn_graph_builder_scatter_elements)
  RUSTNN_ML_TERNARY_METHOD(scatterND, rustnn_graph_builder_scatter_nd)

  MLOperand argMin(const MLOperand &input, std::uint32_t axis) {
    RustnnOperand *output = nullptr;
    check(rustnn_graph_builder_arg_min(value_, input.value_, axis, &output));
    return MLOperand(output);
  }

  MLOperand argMax(const MLOperand &input, std::uint32_t axis) {
    RustnnOperand *output = nullptr;
    check(rustnn_graph_builder_arg_max(value_, input.value_, axis, &output));
    return MLOperand(output);
  }

  MLOperand cast(const MLOperand &input, MLOperandDataType data_type) {
    RustnnOperand *output = nullptr;
    check(rustnn_graph_builder_cast(value_, input.value_,
                                    static_cast<RustnnDataType>(data_type), &output));
    return MLOperand(output);
  }

  MLOperand cumulativeSum(const MLOperand &input, std::uint32_t axis) {
    RustnnOperand *output = nullptr;
    check(rustnn_graph_builder_cumulative_sum(value_, input.value_, axis, &output));
    return MLOperand(output);
  }

  MLOperand softmax(const MLOperand &input, std::uint32_t axis) {
    RustnnOperand *output = nullptr;
    check(rustnn_graph_builder_softmax(value_, input.value_, axis, &output));
    return MLOperand(output);
  }

  MLOperand expand(const MLOperand &input,
                   const std::vector<std::uint32_t> &new_shape) {
    RustnnOperand *output = nullptr;
    check(rustnn_graph_builder_expand(value_, input.value_, new_shape.data(),
                                      new_shape.size(), &output));
    return MLOperand(output);
  }

  MLOperand reshape(const MLOperand &input,
                    const std::vector<std::uint32_t> &new_shape) {
    RustnnOperand *output = nullptr;
    check(rustnn_graph_builder_reshape(value_, input.value_, new_shape.data(),
                                       new_shape.size(), &output));
    return MLOperand(output);
  }

  MLOperand tile(const MLOperand &input,
                 const std::vector<std::uint32_t> &repetitions) {
    RustnnOperand *output = nullptr;
    check(rustnn_graph_builder_tile(value_, input.value_, repetitions.data(),
                                    repetitions.size(), &output));
    return MLOperand(output);
  }

  MLOperand pad(const MLOperand &input,
                const std::vector<std::uint32_t> &beginning_padding,
                const std::vector<std::uint32_t> &ending_padding) {
    RustnnOperand *output = nullptr;
    check(rustnn_graph_builder_pad(
        value_, input.value_, beginning_padding.data(), beginning_padding.size(),
        ending_padding.data(), ending_padding.size(), &output));
    return MLOperand(output);
  }

  MLOperand slice(const MLOperand &input,
                  const std::vector<std::uint32_t> &starts,
                  const std::vector<std::uint32_t> &sizes) {
    RustnnOperand *output = nullptr;
    check(rustnn_graph_builder_slice(value_, input.value_, starts.data(),
                                     starts.size(), sizes.data(), sizes.size(),
                                     &output));
    return MLOperand(output);
  }

  MLOperand concat(const std::vector<const MLOperand *> &inputs,
                   std::uint32_t axis) {
    std::vector<const RustnnOperand *> raw_inputs;
    raw_inputs.reserve(inputs.size());
    for (const MLOperand *input : inputs) {
      raw_inputs.push_back(input != nullptr ? input->value_ : nullptr);
    }
    RustnnOperand *output = nullptr;
    check(rustnn_graph_builder_concat(value_, raw_inputs.data(), raw_inputs.size(),
                                      axis, &output));
    return MLOperand(output);
  }

  MLOperand quantizeLinear(const MLOperand &input, const MLOperand &scale,
                           const MLOperand *zero_point = nullptr) {
    RustnnOperand *output = nullptr;
    check(rustnn_graph_builder_quantize_linear(
        value_, input.value_, scale.value_,
        zero_point != nullptr ? zero_point->value_ : nullptr, &output));
    return MLOperand(output);
  }

  MLOperand dequantizeLinear(const MLOperand &input, const MLOperand &scale,
                             const MLOperand *zero_point = nullptr) {
    RustnnOperand *output = nullptr;
    check(rustnn_graph_builder_dequantize_linear(
        value_, input.value_, scale.value_,
        zero_point != nullptr ? zero_point->value_ : nullptr, &output));
    return MLOperand(output);
  }

  std::vector<MLOperand> split(const MLOperand &input,
                               const std::vector<std::uint32_t> &splits) {
    std::vector<RustnnOperand *> raw_outputs(splits.size());
    std::size_t outputs_len = 0;
    check(rustnn_graph_builder_split(value_, input.value_, splits.data(), splits.size(),
                                     raw_outputs.data(), raw_outputs.size(),
                                     &outputs_len));
    return wrapOperands(raw_outputs, outputs_len);
  }

  std::vector<MLOperand> split(const MLOperand &input,
                               std::uint32_t num_splits) {
    std::vector<RustnnOperand *> raw_outputs(num_splits);
    std::size_t outputs_len = 0;
    check(rustnn_graph_builder_split_equal(
        value_, input.value_, num_splits, raw_outputs.data(), raw_outputs.size(),
        &outputs_len));
    return wrapOperands(raw_outputs, outputs_len);
  }

  std::vector<MLOperand> gru(const MLOperand &input, const MLOperand &weight,
                             const MLOperand &recurrent_weight,
                             std::uint32_t steps, std::uint32_t hidden_size) {
    std::vector<RustnnOperand *> raw_outputs(1);
    std::size_t outputs_len = 0;
    check(rustnn_graph_builder_gru(
        value_, input.value_, weight.value_, recurrent_weight.value_, steps,
        hidden_size, raw_outputs.data(), raw_outputs.size(), &outputs_len));
    return wrapOperands(raw_outputs, outputs_len);
  }

  MLOperand gruCell(const MLOperand &input, const MLOperand &weight,
                    const MLOperand &recurrent_weight,
                    const MLOperand &hidden_state, std::uint32_t hidden_size) {
    RustnnOperand *output = nullptr;
    check(rustnn_graph_builder_gru_cell(
        value_, input.value_, weight.value_, recurrent_weight.value_,
        hidden_state.value_, hidden_size, &output));
    return MLOperand(output);
  }

  std::vector<MLOperand> lstm(const MLOperand &input, const MLOperand &weight,
                              const MLOperand &recurrent_weight,
                              std::uint32_t steps, std::uint32_t hidden_size) {
    std::vector<RustnnOperand *> raw_outputs(2);
    std::size_t outputs_len = 0;
    check(rustnn_graph_builder_lstm(
        value_, input.value_, weight.value_, recurrent_weight.value_, steps,
        hidden_size, raw_outputs.data(), raw_outputs.size(), &outputs_len));
    return wrapOperands(raw_outputs, outputs_len);
  }

  std::vector<MLOperand>
  lstmCell(const MLOperand &input, const MLOperand &weight,
           const MLOperand &recurrent_weight, const MLOperand &hidden_state,
           const MLOperand &cell_state, std::uint32_t hidden_size) {
    std::vector<RustnnOperand *> raw_outputs(2);
    std::size_t outputs_len = 0;
    check(rustnn_graph_builder_lstm_cell(
        value_, input.value_, weight.value_, recurrent_weight.value_,
        hidden_state.value_, cell_state.value_, hidden_size, raw_outputs.data(),
        raw_outputs.size(), &outputs_len));
    return wrapOperands(raw_outputs, outputs_len);
  }

  std::vector<std::uint64_t> operandShape(const MLOperand &operand) {
    std::size_t rank = 0;
    check(rustnn_graph_builder_operand_shape(value_, operand.value_, nullptr, 0, &rank));
    std::vector<std::uint64_t> dimensions(rank);
    check(rustnn_graph_builder_operand_shape(value_, operand.value_, dimensions.data(),
                                             dimensions.size(), &rank));
    return dimensions;
  }

  MLOperandDataType dataType(const MLOperand &operand) {
    RustnnDataType data_type = RustnnDataType_Float32;
    check(rustnn_graph_builder_operand_data_type(value_, operand.value_, &data_type));
    return static_cast<MLOperandDataType>(data_type);
  }

  std::string webnnText(
      const std::vector<std::pair<std::string, const MLOperand *>> &outputs) const {
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

  MLGraph build(const std::vector<std::pair<std::string, const MLOperand *>> &outputs) {
    std::vector<RustnnNamedOperand> named;
    named.reserve(outputs.size());
    for (const auto &[name, operand] : outputs) {
      named.push_back({name.c_str(), operand != nullptr ? operand->value_ : nullptr});
    }
    RustnnGraph *graph = nullptr;
    RustnnStatus status =
        rustnn_graph_builder_build(&value_, named.data(), named.size(), &graph);
    check(status);
    return MLGraph(graph);
  }

private:
  static std::vector<MLOperand>
  wrapOperands(const std::vector<RustnnOperand *> &raw_operands,
               std::size_t length) {
    std::vector<MLOperand> operands;
    operands.reserve(length);
    for (std::size_t index = 0; index < length; ++index) {
      operands.push_back(MLOperand(raw_operands[index]));
    }
    return operands;
  }

  RustnnGraphBuilder *value_ = nullptr;
};

#undef RUSTNN_ML_UNARY_METHOD
#undef RUSTNN_ML_UNARY_METHOD_WITH_OPTIONS
#undef RUSTNN_ML_BINARY_METHOD
#undef RUSTNN_ML_BINARY_METHOD_WITH_OPTIONS
#undef RUSTNN_ML_TERNARY_METHOD

} // namespace rustnn
