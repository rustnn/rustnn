/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef METADEF_CXX_INC_EXE_GRAPH_TILING_CONTEXT_H_
#define METADEF_CXX_INC_EXE_GRAPH_TILING_CONTEXT_H_
#include "storage_shape.h"
#include "tensor.h"
#include "continuous_vector.h"
#include "extended_kernel_context.h"
#include "tiling_data.h"
#include "ge_common/ge_api_error_codes.h"

namespace fe {
class PlatFormInfos;
}  // namespace fe

namespace gert {
/**
 * tiling kernel的context
 */
class TilingContext : public ExtendedKernelContext {
 public:
  const void *GetCompileInfo() const {
    const auto compute_node_info = GetComputeNodeInfo();
    if (compute_node_info == nullptr) {
      return nullptr;
    }

    const size_t index = compute_node_info->GetInputsNum() + compute_node_info->GetOutputsNum();
    const auto av = GetInput(index);
    if (av == nullptr) {
      return nullptr;
    }
    return av->GetValue<void *>();
  }
  /**
   * 获取CompileInfo
   * @tparam T CompileInfo的类型
   * @return CompileInfo的指针
   */
  template<typename T>
  const T *GetCompileInfo() const {
    return reinterpret_cast<const T *>(GetCompileInfo());
  }
  /**
   * 获取输入shape，输入shape中包含了原始shape与运行时shape
   * @param index 输入index
   * @return 输入shape指针，index非法时返回空指针
   */
  const StorageShape *GetInputShape(const size_t index) const {
    const auto compute_node_info = GetComputeNodeInfo();
    if (compute_node_info == nullptr) {
      return nullptr;
    }

    if (index >= compute_node_info->GetInputsNum()) {
      return nullptr;
    }
    return GetInputPointer<StorageShape>(index);
  }
  /**
   * 获取输入tensor
   * **在`IMPL_OP`实现算子时， 如果输入没有被设置为数据依赖，调用此接口获取tensor时，
   *   只能在Tensor中获取到正确的shape, 无法获取到真实的地址（获取到的地址为nullptr）。**
   * @param index 输入index
   * @return 输入tensor指针，index非法时返回空指针
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
   * 基于算子IR原型定义，获取`DYNAMIC_INPUT`类型的输入Tensor指针
   * **在`IMPL_OP`实现算子时， 如果输入没有被设置为数据依赖，调用此接口获取tensor时，
   *   只能在Tensor中获取到正确的shape, 无法获取到真实的地址（获取到的地址为nullptr）。**
   * @param ir_index IR原型定义中的index
   * @param relative_index 该输入实例化后的相对index，例如某个DYNAMIC_INPUT实例化了3个输入，那么relative_index的有效范围是[0,2]
   * @return Tensor指针，index或relative_index非法时，返回空指针
   */
  const Tensor *GetDynamicInputTensor(const size_t ir_index, const size_t relative_index) const {
    return GetDynamicInputPointer<Tensor>(ir_index, relative_index);
  }
  /**
   * 基于算子IR原型定义，获取`OPTIONAL_INPUT`类型的输入shape指针，shape中包含了原始shape与运行时shape
   * @param ir_index IR原型定义中的index
   * @return shape指针，index非法，或该INPUT没有实例化时，返回空指针
   */
  const StorageShape *GetOptionalInputShape(const size_t ir_index) const {
    return GetDynamicInputPointer<StorageShape>(ir_index, 0);
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
  const StorageShape *GetRequiredInputShape(const size_t ir_index) const {
    return GetDynamicInputPointer<StorageShape>(ir_index, 0);
  }
  /**
   * 基于算子IR原型定义，获取`DYNAMIC_INPUT`类型的输入shape指针，shape中包含了原始shape与运行时shape
   * @param ir_index IR原型定义中的index
   * @param relative_index 该输入实例化后的相对index，例如某个DYNAMIC_INPUT实例化了3个输入，那么relative_index的有效范围是[0,2]
   * @return shape指针，index或relative_index非法时，返回空指针
   */
  const StorageShape *GetDynamicInputShape(const size_t ir_index, const size_t relative_index) const {
    return GetDynamicInputPointer<StorageShape>(ir_index, relative_index);
  }
  /**
   * 根据输出index，获取输出shape指针，shape中包含了原始shape与运行时shape
   * @param index 输出index
   * @return 输出shape指针，index非法时，返回空指针
   */
  const StorageShape *GetOutputShape(const size_t index) const {
    const auto compute_node_info = GetComputeNodeInfo();
    if (compute_node_info == nullptr) {
      return nullptr;
    }

    if (index >= compute_node_info->GetOutputsNum()) {
      return nullptr;
    }

    const size_t offset = compute_node_info->GetInputsNum();
    return GetInputPointer<StorageShape>(offset + index);
  }

  /*
   * outputs, tiling的outputs以如下顺序排列：
   * outputs[0]: tiling-key
   * outputs[1]: block-dim
   * outputs[2]: atomic-clean-flag
   * outputs[3]: tiling-data
   * outputs[4]: workspace sizes
   * outputs[5]: tiling condition
   * outputs[6]: schedule mode
   */
  enum TilingOutputIndex : uint32_t {
    kOutputTilingKey,
    kOutputBlockDim,
    kOutputAtomicCleanFlag,
    kOutputTilingData,
    kOutputWorkspace,
    kOutputTilingCond,
    kOutputScheduleMode,
    kOutputLocalMemorySize,
    // add new output definitions here
    kOutputNum
  };

  /*
  * outputs[0]: fallible tiling condition
  */
  enum FallibleTilingOutputIndex : uint32_t {
    kTilingStatus = TilingOutputIndex::kOutputNum,
    // add new output definitions here
    kFallibleOutputNum
  };

  /**
   * 设置tiling key
   * @param tiling_key tiling key
   * @return 成功时返回ge::GRAPH_SUCCESS
   */
  ge::graphStatus SetTilingKey(const uint64_t tiling_key) {
    const auto p = GetOutputPointer<uint64_t>(kOutputTilingKey);
    if (p == nullptr) {
      return ge::GRAPH_FAILED;
    }
    *p = tiling_key;
    return ge::GRAPH_SUCCESS;
  }
  /**
   * 获取tiling key
   * @return tiling key，获取失败时
   */
  uint64_t GetTilingKey() const {
    const auto p = GetOutputPointer<uint64_t>(kOutputTilingKey);
    if (p == nullptr) {
      return std::numeric_limits<uint64_t>::max();
    }
    return *p;
  }

  /**
   * 设置schedule_mode
   * @param schedule_mode schedule_mode
   * @return 成功时返回ge::GRAPH_SUCCESS
   */
  ge::graphStatus SetScheduleMode(const uint32_t schedule_mode) {
    const auto p = GetOutputPointer<uint32_t>(kOutputScheduleMode);
    if (p == nullptr) {
      return ge::GRAPH_FAILED;
    }
    *p = schedule_mode;
    return ge::GRAPH_SUCCESS;
  }
  /**
   * 获取设置schedule_mode
   * @return 设置schedule_mode，获取失败时
   */
  uint32_t GetScheduleMode() const {
    const auto p = GetOutputPointer<uint32_t>(kOutputScheduleMode);
    if (p == nullptr) {
      return 0U;
    }
    return *p;
  }

  /**
   * 设置block dim
   * @param block_dim block dim
   * @return 成功时返回ge::GRAPH_SUCCESS
   */
  ge::graphStatus SetBlockDim(const uint32_t block_dim) {
    const auto p = GetOutputPointer<uint32_t>(kOutputBlockDim);
    if (p == nullptr) {
      return ge::GRAPH_FAILED;
    }
    *p = block_dim;
    return ge::GRAPH_SUCCESS;
  }
  /**
   * 获取block dim
   * @return block dim
   */
  uint32_t GetBlockDim() const {
    const auto p = GetOutputPointer<uint32_t>(kOutputBlockDim);
    if (p == nullptr) {
      return std::numeric_limits<uint32_t>::max();
    }
    return *p;
  }
  /**
   * 设置tiling cond
   * @param tiling_cond tiling condition
   * @return 成功时返回ge::GRAPH_SUCCESS
   */
  ge::graphStatus SetTilingCond(int32_t tiling_cond) {
    const auto p = GetOutputPointer<int32_t>(kOutputTilingCond);
    if (p == nullptr) {
      return ge::GRAPH_FAILED;
    }
    *p = tiling_cond;
    return ge::GRAPH_SUCCESS;
  }
  /**
   * 获取tiling cond
   * @return tiling cond:有效的tiling_cond大于等于0，若该值无效返回-1
   */
  int32_t GetTilingCond() const {
    const auto p = GetOutputPointer<int32_t>(kOutputTilingCond);
    if (p == nullptr) {
      return -1;
    }
    return *p;
  }
  /**
   * 设置是否需要atomic clean
   * @param atomic true/false代表是否需要做atomic clean
   * @return 成功时返回ge::GRAPH_SUCCESS
   */
  ge::graphStatus SetNeedAtomic(const bool atomic) {
    const auto p = GetOutputPointer<bool>(kOutputAtomicCleanFlag);
    if (p == nullptr) {
      return ge::GRAPH_FAILED;
    }
    *p = atomic;
    return ge::GRAPH_SUCCESS;
  }
  /**
   * 获取是否需要atomic clean
   * @return true/false
   */
  bool NeedAtomic() const {
    const auto p = GetOutputPointer<bool>(kOutputAtomicCleanFlag);
    if (p == nullptr) {
      return false;
    }
    return *p;
  }
  /**
   * 获取有类型的tiling data指针
   * @tparam T tiling data类型，sizeof(T)不可以大于编译结果中指定的最大tiling data长度
   * @return tiling data指针，失败时返回空指针
   */
  template<typename T>
  auto GetTilingData() -> T* {
    auto tiling_data = GetRawTilingData();
    if (tiling_data == nullptr) {
      return nullptr;
    }
    if (tiling_data->GetCapacity() < sizeof(T)) {
      return nullptr;
    }
    tiling_data->SetDataSize(sizeof(T));
    return static_cast<T *>(tiling_data->GetData());
  }
  /**
   * 获取无类型的tiling data码流
   * @return tiling data指针，失败时返回空指针
   */
  TilingData *GetRawTilingData() {
    return *GetOutputPointer<TilingData *>(kOutputTilingData);
  }
  /**
   * 获取workspace sizes指针
   * @param workspace_count workspace的个数，传入的workspace个数不可以超过编译时指定的最大workspace个数
   * @return workspace sizes指针
   */
  size_t *GetWorkspaceSizes(const size_t workspace_count) {
    const auto workspace = GetOutputPointer<TypedContinuousVector<size_t>>(kOutputWorkspace);
    if (workspace == nullptr) {
      return nullptr;
    }
    if (workspace->SetSize(workspace_count) != ge::SUCCESS) {
      return nullptr;
    }
    return workspace->MutableData();
  }
   /**
   * 获取 workspace 个数
   * @return workspace 个数
   */
  size_t GetWorkspaceNum() const {
    const auto workspace = GetOutputPointer<TypedContinuousVector<size_t>>(kOutputWorkspace);
    if (workspace == nullptr) {
      return 0U;
    }
    return workspace->GetSize();
  }
  /**
   * 获取 fe::PlatFormInfos 指针
   * @return fe::PlatFormInfos 指针
   */
  fe::PlatFormInfos *GetPlatformInfo() const {
    const auto compute_node_info = GetComputeNodeInfo();
    if (compute_node_info == nullptr) {
      return nullptr;
    }

    const size_t index = compute_node_info->GetInputsNum() + compute_node_info->GetOutputsNum();
    const auto av = GetInput(index + 1U);
    if (av == nullptr) {
      return nullptr;
    }
    return av->GetValue<fe::PlatFormInfos *>();
  }

  /**
   * 获取 确定性计算变量
   * @return int32 变量
   */
  int32_t GetDeterministic() const {
    const auto compute_node_info = GetComputeNodeInfo();
    if (compute_node_info == nullptr) {
      return std::numeric_limits<int32_t>::max();
    }
    const size_t index = compute_node_info->GetInputsNum() + compute_node_info->GetOutputsNum();
    // 此处按照tiling内存排布，将确定性计算的字段添加在
    // inputshape outputshape compileinfo platform tiling_func之后
    const auto av = GetInput(index + 3U);
    if (av == nullptr) {
      return std::numeric_limits<int32_t>::max();
    }
    return av->GetValue<int32_t>();
  }

  /**
   * 设置 local memory size, 默认值为0
   * @param local_memory_size
   * @return 成功返回ge::GRAPH_SUCCESS
   */
  ge::graphStatus SetLocalMemorySize(const uint32_t local_memory_size) {
    const auto p = GetOutputPointer<uint32_t>(kOutputLocalMemorySize);
    if (p == nullptr) {
      return ge::GRAPH_FAILED;
    }
    *p = local_memory_size;
    return ge::GRAPH_SUCCESS;
  }
  /**
   * 获取 local memory size，默认值为0
   * @return local memory size， 失败返回 std::numeric_limits<uint32_t>::max()
   */
  uint32_t GetLocalMemorySize() const {
    const auto p = GetOutputPointer<uint32_t>(kOutputLocalMemorySize);
    if (p == nullptr) {
      return std::numeric_limits<uint32_t>::max();
    }
    return *p;
  }
};
static_assert(std::is_standard_layout<TilingContext>::value, "The class TilingContext must be a POD");
}  // namespace gert
#endif  // METADEF_CXX_INC_EXE_GRAPH_TILING_CONTEXT_H_
