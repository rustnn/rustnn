#pragma once

#include <rustnn/rustnn.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace rustnn {

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

class GraphBuilder {
public:
  GraphBuilder() { check(rustnn_graph_builder_create_uncompiled(&value_)); }
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

private:
  RustnnGraphBuilder *value_ = nullptr;
};

} // namespace rustnn
