/**
 * CANN Adapter Layer - Operator Implementation & Factory Functions
 */

#include "operator_adapter.h"
#include "adapter_internal.h"

#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "graph/attr_value.h"
#include "graph/cann/all_ops.h"
#include "graph/operator.h"
#include "graph/op/all_ops.h"
#include "graph/tensor.h"
#include "op_tensor_adapter.h"

namespace ddk {
extern "C" {

/* ── Base Operator Lifecycle ──────────────────────────────────────────── */

CannOperatorHandle cann_operator_create(const char* type, const char* name) {
    if (!type || !name) return nullptr;
    auto* op = new ge::Operator(std::string(name), std::string(type));
    return FromGeOp(op);
}

/// Generic operator factory that creates the correct typed subclass
/// instead of the base ge::Operator. 
/// Needed because subclasses register type-specific inputs/outputs via
/// HIAI_REG_OP macros that the base class skips.
CannOperatorHandle cann_operator_create_registered(const char* op_type_name, const char* op_name) {
    if (!op_type_name || !op_name) return nullptr;
    std::string type(op_type_name);
    std::string name(op_name);

    // ── Element-wise binary ──────────────────────────────────────
    if (type == "Add")    return FromGeOp(new hiai::op::Add(name));
    if (type == "Sub")    return FromGeOp(new hiai::op::Sub(name));
    if (type == "Mul")    return FromGeOp(new hiai::op::Mul(name));
    if (type == "Div")    return FromGeOp(new hiai::op::RealDiv(name));
    if (type == "Pow")    return FromGeOp(new hiai::op::Pow(name));
    if (type == "Max")    return FromGeOp(new hiai::op::Maximum(name));
    if (type == "Min")    return FromGeOp(new hiai::op::Minimum(name));

    // ── Comparison ───────────────────────────────────────────────
    if (type == "Equal")           return FromGeOp(new hiai::op::Equal(name));
    if (type == "Greater")         return FromGeOp(new hiai::op::Greater(name));
    if (type == "GreaterOrEqual")  return FromGeOp(new hiai::op::GreaterEqual(name));
    if (type == "Lesser")          return FromGeOp(new hiai::op::Less(name));
    if (type == "LesserOrEqual")   return FromGeOp(new hiai::op::LessEqual(name));
    if (type == "NotEqual")        return FromGeOp(new hiai::op::NotEqual(name));

    // ── Logical ──────────────────────────────────────────────────
    if (type == "LogicalAnd")  return FromGeOp(new hiai::op::LogicalAnd(name));
    if (type == "LogicalOr")   return FromGeOp(new hiai::op::LogicalOr(name));
    if (type == "LogicalXor")  return FromGeOp(new hiai::op::LogicalXor(name));
    if (type == "LogicalNot")  return FromGeOp(new hiai::op::LogicalNot(name));

    // ── Unary math ───────────────────────────────────────────────
    if (type == "Abs")   return FromGeOp(new hiai::op::Activation(name));
    if (type == "Neg")   return FromGeOp(new hiai::op::Neg(name));
    if (type == "Exp")   return FromGeOp(new hiai::op::Exp(name));
    if (type == "Log")   return FromGeOp(new hiai::op::Log(name));
    if (type == "Sin")   return FromGeOp(new hiai::op::Sin(name));
    if (type == "Cos")   return FromGeOp(new hiai::op::Cos(name));
    if (type == "Tan")   return FromGeOp(new hiai::op::Tan(name));
    if (type == "Sqrt")  return FromGeOp(new hiai::op::Sqrt(name));
    if (type == "Ceil")  return FromGeOp(new hiai::op::Ceil(name));
    if (type == "Floor") return FromGeOp(new hiai::op::Floor(name));
    if (type == "Sign")  return FromGeOp(new hiai::op::Sign(name));
    if (type == "Erf")   return FromGeOp(new hiai::op::Erf(name));
    if (type == "Reciprocal") return FromGeOp(new hiai::op::Reciprocal(name));

    // ── Activations ──────────────────────────────────────────────
    // Activation modes: 0=Sigmoid, 1=ReLU, 2=Tanh, 3=GELU, 5=LeakyReLU,
    // 6=ELU, 8=Softplus, 9=Softsign, 10=HardSigmoid
    if (type == "ReLU" || type == "Sigmoid" || type == "Tanh" ||
        type == "ELU" || type == "GELU" || type == "LeakyRelu" ||
        type == "HardSigmoid" || type == "Softplus" || type == "Softsign")
        return FromGeOp(new hiai::op::Activation(name));
    if (type == "HardSwish") return FromGeOp(new hiai::op::HardSwish(name));

    // ── Neural network ───────────────────────────────────────────
    if (type == "Conv2D")        return FromGeOp(new hiai::op::Convolution(name));
    if (type == "ConvTranspose") return FromGeOp(new hiai::op::ConvTranspose(name));
    if (type == "MaxPool")       return FromGeOp(new hiai::op::PoolingD(name));
    if (type == "AvgPool")       return FromGeOp(new hiai::op::PoolingD(name));
    if (type == "MatMul")        return FromGeOp(new hiai::op::MatMul(name));
    if (type == "Gemm")          return FromGeOp(new hiai::op::GemmD(name));
    if (type == "Softmax")       return FromGeOp(new hiai::op::Softmax(name));
    if (type == "BatchNormalization") return FromGeOp(new hiai::op::BNInference(name));

    // ── Reductions ───────────────────────────────────────────────
    if (type == "ReduceSum")       return FromGeOp(new hiai::op::ReduceSum(name));
    if (type == "ReduceMean")      return FromGeOp(new hiai::op::ReduceMean(name));
    if (type == "ReduceMax")       return FromGeOp(new hiai::op::ReduceMax(name));
    if (type == "ReduceMin")       return FromGeOp(new hiai::op::ReduceMin(name));
    if (type == "ReduceProduct")   return FromGeOp(new hiai::op::ReduceProdD(name));
    if (type == "ReduceL2")        return FromGeOp(new hiai::op::ReduceL2D(name));
    if (type == "ReduceLogSumExp") return FromGeOp(new hiai::op::ReduceLogSumExp(name));
    if (type == "ArgMax")          return FromGeOp(new hiai::op::ArgMaxExt2(name));

    // ── Shape ops ────────────────────────────────────────────────
    if (type == "Reshape")    return FromGeOp(new hiai::op::Reshape(name));
    if (type == "Transpose")  return FromGeOp(new ge::op::Transpose(name));
    if (type == "Tile")       return FromGeOp(new hiai::op::Tile(name));
    if (type == "Slice")      return FromGeOp(new hiai::op::Slice(name));
    if (type == "Split")      return FromGeOp(new hiai::op::SplitD(name));
    if (type == "Concat")     return FromGeOp(new hiai::op::ConcatD(name));
    if (type == "Pad")        return FromGeOp(new hiai::op::Pad(name));
    if (type == "Squeeze")    return FromGeOp(new hiai::op::Squeeze(name));
    if (type == "Unsqueeze")  return FromGeOp(new hiai::op::ExpandDims(name));
    if (type == "Expand")     return FromGeOp(new hiai::op::BroadcastTo(name));
    if (type == "CumulativeSum") return FromGeOp(new ge::op::Cumsum(name));

    // ── Gather / Scatter / Where / Cast / Clamp ──────────────────
    if (type == "Gather")     return FromGeOp(new hiai::op::GatherV2D(name));
    if (type == "GatherND")   return FromGeOp(new hiai::op::GatherNd(name));
    if (type == "ScatterND")  return FromGeOp(new hiai::op::ScatterNdUpdate(name));
    if (type == "Where")      return FromGeOp(new hiai::op::Select(name));
    if (type == "Cast")       return FromGeOp(new hiai::op::CastT(name));
    if (type == "Identity")   return FromGeOp(new hiai::op::Squeeze(name));
    if (type == "Clamp")      return FromGeOp(new hiai::op::ClipByValue(name));

    // ── Other ────────────────────────────────────────────────────
    if (type == "Resample2D")  return FromGeOp(new hiai::op::ResizeBilinear(name));
    if (type == "Constant")    return FromGeOp(new hiai::op::Const(name));
    if (type == "Shape")       return FromGeOp(new hiai::op::Shape(name));
    if (type == "QuantizeLinear") return FromGeOp(new hiai::op::QuantizeV2(name));

    return nullptr;
}

void cann_operator_destroy(CannOperatorHandle op) {
    delete ToGeOp(op);
}

CannOperatorHandle cann_operator_clone(CannOperatorHandle op) {
    if (!op) return nullptr;
    auto* new_op = new ge::Operator(*ToGeOp(op));
    return FromGeOp(new_op);
}

/* ── Input/Output Connections ─────────────────────────────────────────── */

CannStatus cann_operator_set_input(CannOperatorHandle op,
                                    const char* name,
                                    CannOperatorHandle input_op) {
    if (!op || !name || !input_op) return kInvalidPtr;
    ToGeOp(op)->SetInput(std::string(name), *ToGeOp(input_op));
    return kSuccess;
}

CannStatus cann_operator_set_input_by_index(CannOperatorHandle op,
                                              int32_t index,
                                              CannOperatorHandle input_op,
                                              int32_t input_index) {
    if (!op || !input_op || index < 0) return kInvalidPara;
    ToGeOp(op)->SetInput(index, *ToGeOp(input_op), input_index);
    return kSuccess;
}

/* ── Attributes ──────────────────────────────────────────────────────── */

CannStatus cann_operator_set_attr_int64(CannOperatorHandle op,
                                          const char* name,
                                          int64_t value) {
    if (!op || !name) return kInvalidPtr;
    ToGeOp(op)->SetAttr(std::string(name), value);
    return kSuccess;
}

CannStatus cann_operator_set_attr_float(CannOperatorHandle op,
                                          const char* name,
                                          float value) {
    if (!op || !name) return kInvalidPtr;
    ToGeOp(op)->SetAttr(std::string(name), value);
    return kSuccess;
}

CannStatus cann_operator_set_attr_string(CannOperatorHandle op,
                                           const char* name,
                                           const char* value) {
    if (!op || !name || !value) return kInvalidPtr;
    ToGeOp(op)->SetAttr(std::string(name), std::string(value));
    return kSuccess;
}

CannStatus cann_operator_set_attr_int64_list(CannOperatorHandle op,
                                               const char* name,
                                               const int64_t* values,
                                               int32_t count) {
    if (!op || !name || !values || count <= 0) return kInvalidPara;
    std::vector<int64_t> v(values, values + count);
    ToGeOp(op)->SetAttr(std::string(name), v);
    return kSuccess;
}

CannStatus cann_operator_set_attr_float_list(CannOperatorHandle op,
                                               const char* name,
                                               const float* values,
                                               int32_t count) {
    if (!op || !name || !values || count <= 0) return kInvalidPara;
    std::vector<float> v(values, values + count);
    ToGeOp(op)->SetAttr(std::string(name), v);
    return kSuccess;
}

CannStatus cann_operator_set_attr_tensor(CannOperatorHandle op,
                                          const char* name,
                                          CannOpTensorHandle tensor) {
    if (!op || !name || !tensor) return kInvalidPtr;
    ge::AttrValue attr_value = ge::AttrValue();
    // WARN: Memory model of tensor is unknown. // Here perhaps the tensor is recreated.
    attr_value.SetTensor(std::make_shared<ge::Tensor>(static_cast<CannOpTensorImpl*>(tensor)->tensor));
    ToGeOp(op)->SetAttr(std::string(name), std::move(attr_value));
    return kSuccess;
}

// Todo:: There are two types of tensor in Cann
// 1. ge::Tensor for Operator Tensor Attribute.
// 2. hiai::AiTensor for model I/O dispatch.
// Here we need type 1, but it is not supported yet.
CannStatus cann_operator_set_attr_tensor_raw(CannOperatorHandle op,
                                          const char* name,
                                          const void* data,
                                          uint32_t size,
                                          const int64_t* shape,
                                          int32_t shape_count,
                                          CannDataType dtype) {
    if (!op || !name || !data || size == 0 || !shape || shape_count <= 0)
        return kInvalidPara;
    std::vector<int64_t> shape_vec(shape, shape + shape_count);
    ge::Shape geShape(shape_vec);
    ge::TensorDesc tensor_desc(geShape, ge::FORMAT_ND, static_cast<ge::DataType>(dtype));
    auto tensor = std::make_shared<ge::Tensor>(tensor_desc);
    tensor->SetData(static_cast<const uint8_t*>(data), static_cast<size_t>(size));
    ge::AttrValue attr_value;
    attr_value.SetTensor(tensor);
    ToGeOp(op)->SetAttr(std::string(name), std::move(attr_value));
    return kSuccess;
}

CannStatus cann_operator_set_attr_tensor_raw_format(CannOperatorHandle op,
                                          const char* name,
                                          const void* data,
                                          uint32_t size,
                                          const int64_t* shape,
                                          int32_t shape_count,
                                          CannDataType dtype,
                                          int32_t format) {
    if (!op || !name || !data || size == 0 || !shape || shape_count <= 0)
        return kInvalidPara;
    std::vector<int64_t> shape_vec(shape, shape + shape_count);
    ge::Shape geShape(shape_vec);
    ge::TensorDesc tensor_desc(geShape, static_cast<ge::Format>(format), static_cast<ge::DataType>(dtype));
    auto tensor = std::make_shared<ge::Tensor>(tensor_desc);
    tensor->SetData(static_cast<const uint8_t*>(data), static_cast<size_t>(size));
    ge::AttrValue attr_value;
    attr_value.SetTensor(tensor);
    ToGeOp(op)->SetAttr(std::string(name), std::move(attr_value));
    return kSuccess;
}

CannStatus cann_operator_set_attr_bool(CannOperatorHandle op,
                                        const char* name,
                                        int32_t value) {
    if (!op || !name) return kInvalidPtr;
    ToGeOp(op)->SetAttr(std::string(name), static_cast<bool>(value));
    return kSuccess;
}

/* ── Tensor Descriptor ───────────────────────────────────────────────── */

CannOpTensorDescHandle cann_operator_get_input_desc(CannOperatorHandle op,
                                                   uint32_t index) {
    if (!op) return nullptr;
    ge::TensorDesc desc = ToGeOp(op)->GetInputDesc(index);
    auto* wrap = new CannOpTensorDescImpl{desc};
    return reinterpret_cast<CannOpTensorDescHandle>(wrap);
}

CannOpTensorDescHandle cann_operator_get_output_desc(CannOperatorHandle op,
                                                    uint32_t index) {
    if (!op) return nullptr;
    ge::TensorDesc desc = ToGeOp(op)->GetOutputDesc(index);
    auto* wrap = new CannOpTensorDescImpl{desc};
    return reinterpret_cast<CannOpTensorDescHandle>(wrap);
}

CannStatus cann_operator_update_input_desc(CannOperatorHandle op,
                                              const char* name,
                                              CannOpTensorDescHandle desc) {
    if (!op || !name || !desc) return kInvalidPtr;
    ToGeOp(op)->UpdateInputDesc(std::string(name), desc->desc);
    return kSuccess;
}

/* ── Operator Info ────────────────────────────────────────────────────── */

const char* cann_operator_get_name(CannOperatorHandle op) {
    if (!op) return nullptr;
    std::string name = ToGeOp(op)->GetName();
    char* buf = static_cast<char*>(std::malloc(name.size() + 1));
    if (!buf) return nullptr;
    std::memcpy(buf, name.c_str(), name.size() + 1);
    return buf;
}

const char* cann_operator_get_type(CannOperatorHandle op) {
    if (!op) return nullptr;
    std::string type = ToGeOp(op)->GetType();
    char* buf = static_cast<char*>(std::malloc(type.size() + 1));
    if (!buf) return nullptr;
    std::memcpy(buf, type.c_str(), type.size() + 1);
    return buf;
}

/* ── Dynamic Input / Output ──────────────────────────────────────────── */

CannStatus cann_operator_create_dynamic_input(CannOperatorHandle op,
                                               const char* name,
                                               uint32_t num) {
    if (!op || !name) return kInvalidPtr;
    ToGeOp(op)->DynamicInputRegister(std::string(name), num);
    return kSuccess;
}

CannStatus cann_operator_create_dynamic_output(CannOperatorHandle op,
                                                const char* name,
                                                uint32_t num) {
    if (!op || !name) return kInvalidPtr;
    ToGeOp(op)->DynamicOutputRegister(std::string(name), num);
    return kSuccess;
}

CannStatus cann_operator_set_dynamic_input_by_index(CannOperatorHandle op, const char* name, uint32_t index, CannOperatorHandle input_op) {
    if (!op || !name || !input_op) return kInvalidPtr;
    ToGeOp(op)->SetDynamicInput(std::string(name), index, *ToGeOp(input_op));
    return kSuccess;
}

CannStatus cann_operator_set_dynamic_input_by_index_by_output(CannOperatorHandle op,
                                                const char* name,
                                                uint32_t index,
                                                CannOperatorHandle input_op,
                                                uint32_t output_index) {
    if (!op || !name || !input_op) return kInvalidPtr;
    ToGeOp(op)->SetDynamicInput(std::string(name), index, ToGeOp(input_op)->GetOutput(output_index));
    return kSuccess;
}


CannStatus cann_operator_set_input_by_output(CannOperatorHandle op,
                                              const char* name,
                                              CannOperatorHandle input_op,
                                              uint32_t output_index) {
    if (!op || !name || !input_op) return kInvalidPtr;
    ToGeOp(op)->SetInput(std::string(name), ToGeOp(input_op)->GetOutput(output_index));
    return kSuccess;
}

/* ══════════════════════════════════════════════════════════════════════════
 * Operator Factory Functions
 *
 * Each factory creates a typed operator with the given instance name.
 * Inputs and attributes must be set via the generic setters after creation.
 * ══════════════════════════════════════════════════════════════════════════ */

/* ── Data / Constants ──────────────────────────────────────────────────── */

CannOperatorHandle cann_op_data(void) {
    auto* op = new hiai::op::Data();
    return FromGeOp(op);
}
CannOperatorHandle cann_op_data_with_name(const char* name) {
    if (!name) return nullptr;
    auto* op = new hiai::op::Data(std::string(name));
    return FromGeOp(op);
}
CannOperatorHandle cann_op_const(void) {
    auto* op = new hiai::op::Const();
    return FromGeOp(op);
}
CannOperatorHandle cann_op_const_with_name(const char* name) {
    if (!name) return nullptr;
    auto* op = new hiai::op::Const(std::string(name));
    return FromGeOp(op);
}
/* ── Element-wise Arithmetic ───────────────────────────────────────────── */

CannOperatorHandle cann_op_add(void) {
    auto* op = new hiai::op::Add();
    return FromGeOp(op);
}
CannOperatorHandle cann_op_add_with_name(const char* name) {
    if (!name) return nullptr;
    auto* op = new hiai::op::Add(std::string(name));
    return FromGeOp(op);
}
CannOperatorHandle cann_op_sub(void) {
    auto* op = new hiai::op::Sub();
    return FromGeOp(op);
}
CannOperatorHandle cann_op_sub_with_name(const char* name) {
    if (!name) return nullptr;
    auto* op = new hiai::op::Sub(std::string(name));
    return FromGeOp(op);
}
CannOperatorHandle cann_op_mul(void) {
    auto* op = new hiai::op::Mul();
    return FromGeOp(op);
}
CannOperatorHandle cann_op_mul_with_name(const char* name) {
    if (!name) return nullptr;
    auto* op = new hiai::op::Mul(std::string(name));
    return FromGeOp(op);
}
CannOperatorHandle cann_op_real_div(void) {
    auto* op = new hiai::op::RealDiv();
    return FromGeOp(op);
}
CannOperatorHandle cann_op_real_div_with_name(const char* name) {
    if (!name) return nullptr;
    auto* op = new hiai::op::RealDiv(std::string(name));
    return FromGeOp(op);
}
CannOperatorHandle cann_op_maximum(void) {
    auto* op = new hiai::op::Maximum();
    return FromGeOp(op);
}
CannOperatorHandle cann_op_maximum_with_name(const char* name) {
    if (!name) return nullptr;
    auto* op = new hiai::op::Maximum(std::string(name));
    return FromGeOp(op);
}
CannOperatorHandle cann_op_minimum(void) {
    auto* op = new hiai::op::Minimum();
    return FromGeOp(op);
}
CannOperatorHandle cann_op_minimum_with_name(const char* name) {
    if (!name) return nullptr;
    auto* op = new hiai::op::Minimum(std::string(name));
    return FromGeOp(op);
}
CannOperatorHandle cann_op_pow(void) {
    auto* op = new hiai::op::Pow();
    return FromGeOp(op);
}
CannOperatorHandle cann_op_pow_with_name(const char* name) {
    if (!name) return nullptr;
    auto* op = new hiai::op::Pow(std::string(name));
    return FromGeOp(op);
}
/* ── Comparisons ──────────────────────────────────────────────────────── */

CannOperatorHandle cann_op_greater(void) {
    auto* op = new hiai::op::Greater();
    return FromGeOp(op);
}
CannOperatorHandle cann_op_greater_with_name(const char* name) {
    if (!name) return nullptr;
    auto* op = new hiai::op::Greater(std::string(name));
    return FromGeOp(op);
}
CannOperatorHandle cann_op_greater_equal(void) {
    auto* op = new hiai::op::GreaterEqual();
    return FromGeOp(op);
}
CannOperatorHandle cann_op_greater_equal_with_name(const char* name) {
    if (!name) return nullptr;
    auto* op = new hiai::op::GreaterEqual(std::string(name));
    return FromGeOp(op);
}
CannOperatorHandle cann_op_less(void) {
    auto* op = new hiai::op::Less();
    return FromGeOp(op);
}
CannOperatorHandle cann_op_less_with_name(const char* name) {
    if (!name) return nullptr;
    auto* op = new hiai::op::Less(std::string(name));
    return FromGeOp(op);
}
CannOperatorHandle cann_op_less_equal(void) {
    auto* op = new hiai::op::LessEqual();
    return FromGeOp(op);
}
CannOperatorHandle cann_op_less_equal_with_name(const char* name) {
    if (!name) return nullptr;
    auto* op = new hiai::op::LessEqual(std::string(name));
    return FromGeOp(op);
}
CannOperatorHandle cann_op_equal(void) {
    auto* op = new hiai::op::Equal();
    return FromGeOp(op);
}
CannOperatorHandle cann_op_equal_with_name(const char* name) {
    if (!name) return nullptr;
    auto* op = new hiai::op::Equal(std::string(name));
    return FromGeOp(op);
}
CannOperatorHandle cann_op_not_equal(void) {
    auto* op = new hiai::op::NotEqual();
    return FromGeOp(op);
}
CannOperatorHandle cann_op_not_equal_with_name(const char* name) {
    if (!name) return nullptr;
    auto* op = new hiai::op::NotEqual(std::string(name));
    return FromGeOp(op);
}
/* ── Logical ──────────────────────────────────────────────────────────── */

CannOperatorHandle cann_op_logical_or(void) {
    auto* op = new hiai::op::LogicalOr();
    return FromGeOp(op);
}
CannOperatorHandle cann_op_logical_or_with_name(const char* name) {
    if (!name) return nullptr;
    auto* op = new hiai::op::LogicalOr(std::string(name));
    return FromGeOp(op);
}
CannOperatorHandle cann_op_logical_xor(void) {
    auto* op = new hiai::op::LogicalXor();
    return FromGeOp(op);
}
CannOperatorHandle cann_op_logical_xor_with_name(const char* name) {
    if (!name) return nullptr;
    auto* op = new hiai::op::LogicalXor(std::string(name));
    return FromGeOp(op);
}
CannOperatorHandle cann_op_logical_not(void) {
    auto* op = new hiai::op::LogicalNot();
    return FromGeOp(op);
}
CannOperatorHandle cann_op_logical_not_with_name(const char* name) {
    if (!name) return nullptr;
    auto* op = new hiai::op::LogicalNot(std::string(name));
    return FromGeOp(op);
}
CannOperatorHandle cann_op_logical_and(void) {
    auto* op = new hiai::op::LogicalAnd();
    return FromGeOp(op);
}
CannOperatorHandle cann_op_logical_and_with_name(const char* name) {
    if (!name) return nullptr;
    auto* op = new hiai::op::LogicalAnd(std::string(name));
    return FromGeOp(op);
}
/* ── Activations ──────────────────────────────────────────────────────── */

CannOperatorHandle cann_op_activation(void) {
    auto* op = new hiai::op::Activation();
    return FromGeOp(op);
}
CannOperatorHandle cann_op_activation_with_name(const char* name) {
    if (!name) return nullptr;
    auto* op = new hiai::op::Activation(std::string(name));
    return FromGeOp(op);
}
CannOperatorHandle cann_op_hard_swish(void) {
    auto* op = new hiai::op::HardSwish();
    return FromGeOp(op);
}
CannOperatorHandle cann_op_hard_swish_with_name(const char* name) {
    if (!name) return nullptr;
    auto* op = new hiai::op::HardSwish(std::string(name));
    return FromGeOp(op);
}
/* ── Unary Math ──────────────────────────────────────────────────────── */

CannOperatorHandle cann_op_neg(void) {
    auto* op = new hiai::op::Neg();
    return FromGeOp(op);
}
CannOperatorHandle cann_op_neg_with_name(const char* name) {
    if (!name) return nullptr;
    auto* op = new hiai::op::Neg(std::string(name));
    return FromGeOp(op);
}
CannOperatorHandle cann_op_ceil(void) {
    auto* op = new hiai::op::Ceil();
    return FromGeOp(op);
}
CannOperatorHandle cann_op_ceil_with_name(const char* name) {
    if (!name) return nullptr;
    auto* op = new hiai::op::Ceil(std::string(name));
    return FromGeOp(op);
}
CannOperatorHandle cann_op_cos(void) {
    auto* op = new hiai::op::Cos();
    return FromGeOp(op);
}
CannOperatorHandle cann_op_cos_with_name(const char* name) {
    if (!name) return nullptr;
    auto* op = new hiai::op::Cos(std::string(name));
    return FromGeOp(op);
}
CannOperatorHandle cann_op_exp(void) {
    auto* op = new hiai::op::Exp();
    return FromGeOp(op);
}
CannOperatorHandle cann_op_exp_with_name(const char* name) {
    if (!name) return nullptr;
    auto* op = new hiai::op::Exp(std::string(name));
    return FromGeOp(op);
}
CannOperatorHandle cann_op_floor(void) {
    auto* op = new hiai::op::Floor();
    return FromGeOp(op);
}
CannOperatorHandle cann_op_floor_with_name(const char* name) {
    if (!name) return nullptr;
    auto* op = new hiai::op::Floor(std::string(name));
    return FromGeOp(op);
}
CannOperatorHandle cann_op_log(void) {
    auto* op = new hiai::op::Log();
    return FromGeOp(op);
}
CannOperatorHandle cann_op_log_with_name(const char* name) {
    if (!name) return nullptr;
    auto* op = new hiai::op::Log(std::string(name));
    return FromGeOp(op);
}
CannOperatorHandle cann_op_sign(void) {
    auto* op = new hiai::op::Sign();
    return FromGeOp(op);
}
CannOperatorHandle cann_op_sign_with_name(const char* name) {
    if (!name) return nullptr;
    auto* op = new hiai::op::Sign(std::string(name));
    return FromGeOp(op);
}
CannOperatorHandle cann_op_round(void) {
    auto* op = new hiai::op::Round();
    return FromGeOp(op);
}
CannOperatorHandle cann_op_round_with_name(const char* name) {
    if (!name) return nullptr;
    auto* op = new hiai::op::Round(std::string(name));
    return FromGeOp(op);
}
CannOperatorHandle cann_op_sin(void) {
    auto* op = new hiai::op::Sin();
    return FromGeOp(op);
}
CannOperatorHandle cann_op_sin_with_name(const char* name) {
    if (!name) return nullptr;
    auto* op = new hiai::op::Sin(std::string(name));
    return FromGeOp(op);
}
CannOperatorHandle cann_op_tan(void) {
    auto* op = new hiai::op::Tan();
    return FromGeOp(op);
}
CannOperatorHandle cann_op_tan_with_name(const char* name) {
    if (!name) return nullptr;
    auto* op = new hiai::op::Tan(std::string(name));
    return FromGeOp(op);
}
CannOperatorHandle cann_op_sqrt(void) {
    auto* op = new hiai::op::Sqrt();
    return FromGeOp(op);
}
CannOperatorHandle cann_op_sqrt_with_name(const char* name) {
    if (!name) return nullptr;
    auto* op = new hiai::op::Sqrt(std::string(name));
    return FromGeOp(op);
}
CannOperatorHandle cann_op_erf(void) {
    auto* op = new hiai::op::Erf();
    return FromGeOp(op);
}
CannOperatorHandle cann_op_erf_with_name(const char* name) {
    if (!name) return nullptr;
    auto* op = new hiai::op::Erf(std::string(name));
    return FromGeOp(op);
}
CannOperatorHandle cann_op_reciprocal(void) {
    auto* op = new hiai::op::Reciprocal();
    return FromGeOp(op);
}
CannOperatorHandle cann_op_reciprocal_with_name(const char* name) {
    if (!name) return nullptr;
    auto* op = new hiai::op::Reciprocal(std::string(name));
    return FromGeOp(op);
}
CannOperatorHandle cann_op_square(void) {
    auto* op = new hiai::op::Square();
    return FromGeOp(op);
}
CannOperatorHandle cann_op_square_with_name(const char* name) {
    if (!name) return nullptr;
    auto* op = new hiai::op::Square(std::string(name));
    return FromGeOp(op);
}
/* ── Neural Network ──────────────────────────────────────────────────── */

CannOperatorHandle cann_op_conv2d(void) {
    auto* op = new ge::op::Conv2D();
    return FromGeOp(op);
}
CannOperatorHandle cann_op_conv2d_with_name(const char* name) {
    if (!name) return nullptr;
    auto* op = new ge::op::Conv2D(std::string(name));
    return FromGeOp(op);
}
CannOperatorHandle cann_op_matmul(void) {
    auto* op = new hiai::op::MatMul();
    return FromGeOp(op);
}
CannOperatorHandle cann_op_matmul_with_name(const char* name) {
    if (!name) return nullptr;
    auto* op = new hiai::op::MatMul(std::string(name));
    return FromGeOp(op);
}
CannOperatorHandle cann_op_pool2d(void) {
    auto* op = new ge::op::Pooling();
    return FromGeOp(op);
}
CannOperatorHandle cann_op_pool2d_with_name(const char* name) {
    if (!name) return nullptr;
    auto* op = new ge::op::Pooling(std::string(name));
    return FromGeOp(op);
}
CannOperatorHandle cann_op_softmax(void) {
    auto* op = new hiai::op::Softmax();
    return FromGeOp(op);
}
CannOperatorHandle cann_op_softmax_with_name(const char* name) {
    if (!name) return nullptr;
    auto* op = new hiai::op::Softmax(std::string(name));
    return FromGeOp(op);
}
CannOperatorHandle cann_op_batch_norm(void) {
    auto* op = new ge::op::BNInference();
    return FromGeOp(op);
}
CannOperatorHandle cann_op_batch_norm_with_name(const char* name) {
    if (!name) return nullptr;
    auto* op = new ge::op::BNInference(std::string(name));
    return FromGeOp(op);
}
CannOperatorHandle cann_op_bn_inference(void) {
    auto* op = new hiai::op::BNInference();
    return FromGeOp(op);
}
CannOperatorHandle cann_op_bn_inference_with_name(const char* name) {
    if (!name) return nullptr;
    auto* op = new hiai::op::BNInference(std::string(name));
    return FromGeOp(op);
}
CannOperatorHandle cann_op_convolution(void) {
    auto* op = new hiai::op::Convolution();
    return FromGeOp(op);
}
CannOperatorHandle cann_op_convolution_with_name(const char* name) {
    if (!name) return nullptr;
    auto* op = new hiai::op::Convolution(std::string(name));
    return FromGeOp(op);
}
CannOperatorHandle cann_op_pool2d_d(void) {
    auto* op = new hiai::op::PoolingD();
    return FromGeOp(op);
}
CannOperatorHandle cann_op_pool2d_d_with_name(const char* name) {
    if (!name) return nullptr;
    auto* op = new hiai::op::PoolingD(std::string(name));
    return FromGeOp(op);
}
CannOperatorHandle cann_op_gemm_d(void) {
    auto* op = new hiai::op::GemmD();
    return FromGeOp(op);
}
CannOperatorHandle cann_op_gemm_d_with_name(const char* name) {
    if (!name) return nullptr;
    auto* op = new hiai::op::GemmD(std::string(name));
    return FromGeOp(op);
}
/* ── Shape / Transform ───────────────────────────────────────────────── */

CannOperatorHandle cann_op_reshape(void) {
    auto* op = new hiai::op::Reshape();
    return FromGeOp(op);
}
CannOperatorHandle cann_op_reshape_with_name(const char* name) {
    if (!name) return nullptr;
    auto* op = new hiai::op::Reshape(std::string(name));
    return FromGeOp(op);
}
CannOperatorHandle cann_op_transpose(void) {
    auto* op = new ge::op::Transpose();
    return FromGeOp(op);
}
CannOperatorHandle cann_op_transpose_with_name(const char* name) {
    if (!name) return nullptr;
    auto* op = new ge::op::Transpose(std::string(name));
    return FromGeOp(op);
}
CannOperatorHandle cann_op_concat(void) {
    auto* op = new hiai::op::ConcatD();
    return FromGeOp(op);
}
CannOperatorHandle cann_op_concat_with_name(const char* name) {
    if (!name) return nullptr;
    auto* op = new hiai::op::ConcatD(std::string(name));
    return FromGeOp(op);
}
CannOperatorHandle cann_op_split(void) {
    auto* op = new hiai::op::SplitD();
    return FromGeOp(op);
}
CannOperatorHandle cann_op_split_with_name(const char* name) {
    if (!name) return nullptr;
    auto* op = new hiai::op::SplitD(std::string(name));
    return FromGeOp(op);
}
CannOperatorHandle cann_op_slice(void) {
    auto* op = new hiai::op::Slice();
    return FromGeOp(op);
}
CannOperatorHandle cann_op_slice_with_name(const char* name) {
    if (!name) return nullptr;
    auto* op = new hiai::op::Slice(std::string(name));
    return FromGeOp(op);
}
CannOperatorHandle cann_op_squeeze(void) {
    auto* op = new hiai::op::Squeeze();
    return FromGeOp(op);
}
CannOperatorHandle cann_op_squeeze_with_name(const char* name) {
    if (!name) return nullptr;
    auto* op = new hiai::op::Squeeze(std::string(name));
    return FromGeOp(op);
}
CannOperatorHandle cann_op_expand_dims(void) {
    auto* op = new hiai::op::ExpandDims();
    return FromGeOp(op);
}
CannOperatorHandle cann_op_expand_dims_with_name(const char* name) {
    if (!name) return nullptr;
    auto* op = new hiai::op::ExpandDims(std::string(name));
    return FromGeOp(op);
}
/* ── Array ───────────────────────────────────────────────────────────── */

CannOperatorHandle cann_op_broadcast_to(void) {
    auto* op = new hiai::op::BroadcastTo();
    return FromGeOp(op);
}
CannOperatorHandle cann_op_broadcast_to_with_name(const char* name) {
    if (!name) return nullptr;
    auto* op = new hiai::op::BroadcastTo(std::string(name));
    return FromGeOp(op);
}
CannOperatorHandle cann_op_shape(void) {
    auto* op = new hiai::op::Shape();
    return FromGeOp(op);
}
CannOperatorHandle cann_op_shape_with_name(const char* name) {
    if (!name) return nullptr;
    auto* op = new hiai::op::Shape(std::string(name));
    return FromGeOp(op);
}
CannOperatorHandle cann_op_gather_nd(void) {
    auto* op = new hiai::op::GatherNd();
    return FromGeOp(op);
}
CannOperatorHandle cann_op_gather_nd_with_name(const char* name) {
    if (!name) return nullptr;
    auto* op = new hiai::op::GatherNd(std::string(name));
    return FromGeOp(op);
}
CannOperatorHandle cann_op_gather_v2d(void) {
    auto* op = new hiai::op::GatherV2D();
    return FromGeOp(op);
}
CannOperatorHandle cann_op_gather_v2d_with_name(const char* name) {
    if (!name) return nullptr;
    auto* op = new hiai::op::GatherV2D(std::string(name));
    return FromGeOp(op);
}
CannOperatorHandle cann_op_pad(void) {
    auto* op = new hiai::op::Pad();
    return FromGeOp(op);
}
CannOperatorHandle cann_op_pad_with_name(const char* name) {
    if (!name) return nullptr;
    auto* op = new hiai::op::Pad(std::string(name));
    return FromGeOp(op);
}
CannOperatorHandle cann_op_tile(void) {
    auto* op = new hiai::op::Tile();
    return FromGeOp(op);
}
CannOperatorHandle cann_op_tile_with_name(const char* name) {
    if (!name) return nullptr;
    auto* op = new hiai::op::Tile(std::string(name));
    return FromGeOp(op);
}
CannOperatorHandle cann_op_select(void) {
    auto* op = new hiai::op::Select();
    return FromGeOp(op);
}
CannOperatorHandle cann_op_select_with_name(const char* name) {
    if (!name) return nullptr;
    auto* op = new hiai::op::Select(std::string(name));
    return FromGeOp(op);
}
CannOperatorHandle cann_op_strided_slice_v2(void) {
    auto* op = new hiai::op::StridedSliceV2();
    return FromGeOp(op);
}
CannOperatorHandle cann_op_strided_slice_v2_with_name(const char* name) {
    if (!name) return nullptr;
    auto* op = new hiai::op::StridedSliceV2(std::string(name));
    return FromGeOp(op);
}
CannOperatorHandle cann_op_scatter_nd_update(void) {
    auto* op = new hiai::op::ScatterNdUpdate();
    return FromGeOp(op);
}
CannOperatorHandle cann_op_scatter_nd_update_with_name(const char* name) {
    if (!name) return nullptr;
    auto* op = new hiai::op::ScatterNdUpdate(std::string(name));
    return FromGeOp(op);
}
CannOperatorHandle cann_op_clip_by_value(void) {
    auto* op = new hiai::op::ClipByValue();
    return FromGeOp(op);
}
CannOperatorHandle cann_op_clip_by_value_with_name(const char* name) {
    if (!name) return nullptr;
    auto* op = new hiai::op::ClipByValue(std::string(name));
    return FromGeOp(op);
}
CannOperatorHandle cann_op_arg_max_ext2(void) {
    auto* op = new hiai::op::ArgMaxExt2();
    return FromGeOp(op);
}
CannOperatorHandle cann_op_arg_max_ext2_with_name(const char* name) {
    if (!name) return nullptr;
    auto* op = new hiai::op::ArgMaxExt2(std::string(name));
    return FromGeOp(op);
}
CannOperatorHandle cann_op_cast_t(void) {
    auto* op = new hiai::op::CastT();
    return FromGeOp(op);
}
CannOperatorHandle cann_op_cast_t_with_name(const char* name) {
    if (!name) return nullptr;
    auto* op = new hiai::op::CastT(std::string(name));
    return FromGeOp(op);
}
/* ── Image ───────────────────────────────────────────────────────────── */

CannOperatorHandle cann_op_resize_nearest_neighbor(void) {
    auto* op = new hiai::op::ResizeNearestNeighbor();
    return FromGeOp(op);
}
CannOperatorHandle cann_op_resize_nearest_neighbor_with_name(const char* name) {
    if (!name) return nullptr;
    auto* op = new hiai::op::ResizeNearestNeighbor(std::string(name));
    return FromGeOp(op);
}
CannOperatorHandle cann_op_resize_bilinear(void) {
    auto* op = new hiai::op::ResizeBilinear();
    return FromGeOp(op);
}
CannOperatorHandle cann_op_resize_bilinear_with_name(const char* name) {
    if (!name) return nullptr;
    auto* op = new hiai::op::ResizeBilinear(std::string(name));
    return FromGeOp(op);
}
/* ── Quantization ────────────────────────────────────────────────────── */

CannOperatorHandle cann_op_quantize_v2(void) {
    auto* op = new hiai::op::QuantizeV2();
    return FromGeOp(op);
}
CannOperatorHandle cann_op_quantize_v2_with_name(const char* name) {
    if (!name) return nullptr;
    auto* op = new hiai::op::QuantizeV2(std::string(name));
    return FromGeOp(op);
}
/* ── Reductions ──────────────────────────────────────────────────────── */

CannOperatorHandle cann_op_reduce_sum(void) {
    auto* op = new hiai::op::ReduceSum();
    return FromGeOp(op);
}
CannOperatorHandle cann_op_reduce_sum_with_name(const char* name) {
    if (!name) return nullptr;
    auto* op = new hiai::op::ReduceSum(std::string(name));
    return FromGeOp(op);
}
CannOperatorHandle cann_op_reduce_mean(void) {
    auto* op = new hiai::op::ReduceMean();
    return FromGeOp(op);
}
CannOperatorHandle cann_op_reduce_mean_with_name(const char* name) {
    if (!name) return nullptr;
    auto* op = new hiai::op::ReduceMean(std::string(name));
    return FromGeOp(op);
}
CannOperatorHandle cann_op_reduce_max(void) {
    auto* op = new hiai::op::ReduceMax();
    return FromGeOp(op);
}
CannOperatorHandle cann_op_reduce_max_with_name(const char* name) {
    if (!name) return nullptr;
    auto* op = new hiai::op::ReduceMax(std::string(name));
    return FromGeOp(op);
}
CannOperatorHandle cann_op_reduce_min(void) {
    auto* op = new hiai::op::ReduceMin();
    return FromGeOp(op);
}
CannOperatorHandle cann_op_reduce_min_with_name(const char* name) {
    if (!name) return nullptr;
    auto* op = new hiai::op::ReduceMin(std::string(name));
    return FromGeOp(op);
}
CannOperatorHandle cann_op_reduce_prod_d(void) {
    auto* op = new hiai::op::ReduceProdD();
    return FromGeOp(op);
}
CannOperatorHandle cann_op_reduce_prod_d_with_name(const char* name) {
    if (!name) return nullptr;
    auto* op = new hiai::op::ReduceProdD(std::string(name));
    return FromGeOp(op);
}
CannOperatorHandle cann_op_reduce_l2d(void) {
    auto* op = new hiai::op::ReduceL2D();
    return FromGeOp(op);
}
CannOperatorHandle cann_op_reduce_l2d_with_name(const char* name) {
    if (!name) return nullptr;
    auto* op = new hiai::op::ReduceL2D(std::string(name));
    return FromGeOp(op);
}
CannOperatorHandle cann_op_reduce_log_sum_exp(void) {
    auto* op = new hiai::op::ReduceLogSumExp();
    return FromGeOp(op);
}
CannOperatorHandle cann_op_reduce_log_sum_exp_with_name(const char* name) {
    if (!name) return nullptr;
    auto* op = new hiai::op::ReduceLogSumExp(std::string(name));
    return FromGeOp(op);
}
/* ── ge::op ──────────────────────────────────────────────────────────── */

CannOperatorHandle cann_op_cumsum(void) {
    auto* op = new ge::op::Cumsum();
    return FromGeOp(op);
}
CannOperatorHandle cann_op_cumsum_with_name(const char* name) {
    if (!name) return nullptr;
    auto* op = new ge::op::Cumsum(std::string(name));
    return FromGeOp(op);
}
/* ── NetOutput ────────────────────────────────────────────────────────── */

CannOperatorHandle cann_op_net_output(void) {
    auto* op = new hiai::op::NetOutput();
    return FromGeOp(op);
}
CannOperatorHandle cann_op_net_output_with_name(const char* name, int32_t input_count) {
    if (!name) return nullptr;
    auto* op = new hiai::op::NetOutput(std::string(name));
    (void)input_count;
    return FromGeOp(op);
}

}  // extern "C"
}  // namespace ddk