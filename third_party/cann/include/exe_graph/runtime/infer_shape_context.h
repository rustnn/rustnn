/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef METADEF_CXX_INC_EXE_GRAPH_RUNTIME_INFER_SHAPE_CONTEXT_H_
#define METADEF_CXX_INC_EXE_GRAPH_RUNTIME_INFER_SHAPE_CONTEXT_H_
#include <type_traits>
#include "shape.h"
#include "tensor.h"
#include "runtime_attrs.h"
#include "extended_kernel_context.h"
#include "graph/inference_context.h"
namespace gert {
/**
 * 在节点输入后的扩展输入的索引，若需要扩展，请新增枚举类型
 */
enum class InputExternLayout : uint32_t {
  kInferShapeFunc = 1,   // only exe runtime need infer shape func, compile stage need set to null
  kInferenceContext = 2, // only resource op in compile stage need inference context, exe runtime need set to null
};
/**
 * InferShape kernel的context
 */
class InferShapeContext : public ExtendedKernelContext {
 public:
  /**
   * 根据输入index，获取输入shape指针
   * @param index 输入index
   * @return 输入shape指针，index非法时，返回空指针
   */
  const Shape *GetInputShape(const size_t index) const {
    return GetInputPointer<Shape>(index);
  }
  /**
   * 根据输入index，获取输出tensor指针
   * **在`IMPL_OP`实现算子时， 如果输入没有被设置为数据依赖，调用此接口获取tensor时，
   *   只能在Tensor中获取到正确的shape, 无法获取到真实的地址（获取到的地址为nullptr）。**
   * @param index 输入index
   * @return 输入tensor指针，index非法时，返回空指针
   */
  const Tensor *GetInputTensor(const size_t index) const {
    return GetInputPointer<Tensor>(index);
  }

  /**
   * 基于算子IR原型定义，获取`OPTIONAL_INPUT`类型的输入tensor指针
   * **在`IMPL_OP`实现算子时， 如果输入没有被设置为数据依赖，调用此接口获取tensor时，
   *   只能在Tensor中获取到正确的shape, 无法获取到真实的地址（获取到的地址为nullptr）。**
   * @param ir_index IR原型定义中的index
   * @return tensor指针，index非法，或该INPUT没有实例化时，返回空指针
   */
  const Tensor *GetOptionalInputTensor(const size_t ir_index) const {
    return GetDynamicInputPointer<Tensor>(ir_index, 0);
  }

  /**
   * 基于算子IR原型定义，获取`OPTIONAL_INPUT`类型的输入shape指针
   * @param ir_index IR原型定义中的index
   * @return shape指针，index非法，或该INPUT没有实例化时，返回空指针
   */
  const Shape *GetOptionalInputShape(const size_t ir_index) const {
    return GetDynamicInputPointer<Shape>(ir_index, 0);
  }

  /**
   * 基于算子IR原型定义，获取`DYNAMIC_INPUT`类型的输入shape指针
   * @param ir_index IR原型定义中的index
   * @param relative_index 该输入实例化后的相对index，例如某个DYNAMIC_INPUT实例化了3个输入，那么relative_index的有效范围是[0,2]
   * @return shape指针，index或relative_index非法时，返回空指针
   */
  const Shape *GetDynamicInputShape(const size_t ir_index, const size_t relative_index) const {
    return GetDynamicInputPointer<Shape>(ir_index, relative_index);
  }

  /**
   * 基于算子IR原型定义，获取`DYNAMIC_INPUT`类型的输入tensor指针
   * **在`IMPL_OP`实现算子时， 如果输入没有被设置为数据依赖，调用此接口获取tensor时，
   *   只能在Tensor中获取到正确的shape, 无法获取到真实的地址（获取到的地址为nullptr）。**
   * @param ir_index IR原型定义中的index
   * @param relative_index 该输入实例化后的相对index，例如某个DYNAMIC_INPUT实例化了3个输入，那么relative_index的有效范围是[0,2]
   * @return tensor指针，index或relative_index非法时，返回空指针
   */
  const Tensor *GetDynamicInputTensor(const size_t ir_index, const size_t relative_index) const {
    return GetDynamicInputPointer<Tensor>(ir_index, relative_index);
  }

  /**
   * 基于算子IR原型定义，获取`REQUIRED_INPUT`类型的输入Tensor指针
   * **在`IMPL_OP`实现算子时， 如果输入没有被设置为数据依赖，调用此接口获取tensor时，
   *   只能在Tensor中获取到正确的shape, 无法获取到真实的地址（获取到的地址为nullptr）。**
   * @param ir_index IR原型定义中的index
   * @return Tensor指针，index非法时，返回空指针
   */
  const Tensor *GetRequiredInputTensor(const size_t ir_index) const {
    return GetDynamicInputPointer<Tensor>(ir_index, 0);
  }
  /**
   * 基于算子IR原型定义，获取`REQUIRED_INPUT`类型的输入shape指针，shape中包含了原始shape与运行时shape
   * @param ir_index IR原型定义中的index
   * @return shape指针，index非法，或该INPUT没有实例化时，返回空指针
   */
  const Shape *GetRequiredInputShape(const size_t ir_index) const {
    return GetDynamicInputPointer<Shape>(ir_index, 0);
  }

  /**
   * 根据输出index，获取输出shape指针
   * @param index 输出index
   * @return 输出shape指针，index非法时，返回空指针
   */
  Shape *GetOutputShape(const size_t index) {
    return GetOutputPointer<Shape>(index);
  }

  /**
   * 获取InferenceContext指针
   * @param NA
   * @return 输出InferenceContext指针 指针在节点输入地址后，仅编译态使用
   */
  const ge::InferenceContext *GetInferenceContextPtr() const {
    const auto compute_node_info = reinterpret_cast<const ComputeNodeInfo *>(GetContext()->compute_node_info);
    if (compute_node_info == nullptr) {
      return nullptr;
    }
    const size_t offset = compute_node_info->GetInputsNum() + static_cast<size_t>(InputExternLayout::kInferenceContext);
    if (GetContext()->input_size < offset) {
      return nullptr;
    }
    return GetInputPointer<ge::InferenceContext>(offset - 1UL);
  }
};
static_assert(std::is_standard_layout<InferShapeContext>::value, "The class InferShapeContext must be a POD");
}  // namespace gert
#endif  // METADEF_CXX_INC_EXE_GRAPH_RUNTIME_INFER_SHAPE_CONTEXT_H_
