/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef INC_EXTERNAL_REGISTER_OP_IMPL_REGISTRY_H_
#define INC_EXTERNAL_REGISTER_OP_IMPL_REGISTRY_H_
#include <initializer_list>
#include <string>
#include <map>
#include <unordered_set>
#include "graph/ge_error_codes.h"
#include "graph/types.h"
#include "graph/compiler_def.h"
#include "exe_graph/runtime/base_type.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/infer_shape_range_context.h"
#include "exe_graph/runtime/infer_datatype_context.h"
#include "exe_graph/runtime/tiling_context.h"
#include "exe_graph/runtime/tiling_parse_context.h"
#include "exe_graph/runtime/op_execute_context.h"

namespace ge {
class AnyValue;
}  // namespace ge

namespace gert {
enum TilingPlacement {
  TILING_ON_HOST = 0,
  TILING_ON_AICPU = 1,
};

class OpImplRegisterV2Impl;
class OpImplRegisterV2 {
 public:
  explicit OpImplRegisterV2(const ge::char_t *op_type);
  OpImplRegisterV2(OpImplRegisterV2 &&register_data) noexcept;
  OpImplRegisterV2(const OpImplRegisterV2 &register_data);
  OpImplRegisterV2 &operator=(const OpImplRegisterV2 &) = delete;
  OpImplRegisterV2 &operator=(OpImplRegisterV2 &&) = delete;
  ~OpImplRegisterV2();

  using InferShapeKernelFunc = UINT32 (*)(InferShapeContext *);
  using InferShapeRangeKernelFunc = UINT32 (*)(InferShapeRangeContext *);
  using TilingKernelFunc = UINT32 (*)(TilingContext *);
  using InferDataTypeKernelFunc = UINT32 (*)(InferDataTypeContext *);
  using GenSimplifiedKeyKernelFunc = UINT32 (*)(TilingContext *, ge::char_t *);
  // aclnn接口的原型，入参含义：
  // OpExecuteContext：保存算子的Input，Output，Attr信息
  using OpExecFunc = UINT32 (*)(OpExecuteContext *);
  using OpType = ge::AscendString;
  using PrivateAttrList = std::vector<std::pair<ge::AscendString, ge::AnyValue>>;
  using PrivateAttrSet = std::unordered_set<ge::AscendString>;
  using CompileInfoCreatorFunc = void *(*) ();
  using CompileInfoDeleterFunc = void (*)(void *);
  using KernelFunc = UINT32 (*)(KernelContext *context);
  using TilingParseFunc = UINT32 (*)(TilingParseContext *context);

 public:
  OpImplRegisterV2 &InferShape(InferShapeKernelFunc infer_shape_func);
  OpImplRegisterV2 &InferShapeRange(InferShapeRangeKernelFunc infer_shape_range_func);
  OpImplRegisterV2 &InferDataType(InferDataTypeKernelFunc infer_datatype_func);
  /*
   * 一种datatype推导规则，将第一个输入的datatype作为所有输出的datatype。
   * 使用方式：注册此规则，可以不用再注册自定义推导规则。若同时注册了InferDataType和InferOutDataTypeByFirstInput，将使能最后注册的规则。
   * 异常场景：若算子无输入或输入datatype为未定义，推导将报错。
   * */
  OpImplRegisterV2 &InferOutDataTypeSameWithFirstInput();
  OpImplRegisterV2 &Tiling(TilingKernelFunc tiling_func, size_t max_tiling_data_size = 2048);
  OpImplRegisterV2 &GenSimplifiedKey(GenSimplifiedKeyKernelFunc gen_simplifiedkey_func);
  OpImplRegisterV2 &PrivateAttr(const ge::char_t *private_attr);
  OpImplRegisterV2 &PrivateAttr(const ge::char_t *private_attr, int64_t private_attr_val);
  OpImplRegisterV2 &PrivateAttr(const ge::char_t *private_attr, const std::vector<int64_t> &private_attr_val);
  OpImplRegisterV2 &PrivateAttr(const ge::char_t *private_attr, const ge::char_t *private_attr_val);
  OpImplRegisterV2 &PrivateAttr(const ge::char_t *private_attr, ge::float32_t private_attr_val);
  OpImplRegisterV2 &PrivateAttr(const ge::char_t *private_attr, bool private_attr_val);
  OpImplRegisterV2 &PrivateAttr(const ge::char_t *private_attr, const std::vector<ge::float32_t> &private_attr_val);
  template<typename T>
  OpImplRegisterV2 &TilingParse(KernelFunc const tiling_parse_func) {
    return TilingParse(tiling_parse_func, CreateCompileInfo<T>, DeleteCompileInfo<T>);
  }
  template<typename T>
  OpImplRegisterV2 &TilingParse(TilingParseFunc const tiling_parse_func) {
    return TilingParse(reinterpret_cast<KernelFunc>(tiling_parse_func), CreateCompileInfo<T>,
                       DeleteCompileInfo<T>);
  }
  OpImplRegisterV2 &InputsDataDependency(std::initializer_list<int32_t> inputs);
  OpImplRegisterV2 &OpExecuteFunc(OpExecFunc op_execute_func);
  OpImplRegisterV2 &HostInputs(std::initializer_list<int32_t> inputs);
  OpImplRegisterV2 &TilingInputsDataDependency(std::initializer_list<int32_t> inputs);
  OpImplRegisterV2 &TilingInputsDataDependency(std::initializer_list<int32_t> inputs,
                    std::initializer_list<TilingPlacement> placements);
  OpImplRegisterV2 &OutputShapeDependOnCompute(std::initializer_list<int32_t> outputs);

 private:
  OpImplRegisterV2 &TilingParse(KernelFunc tiling_parse_func,
                                CompileInfoCreatorFunc creator_func,
                                CompileInfoDeleterFunc deleter_func);
  OpImplRegisterV2 &PrivateAttr(const ge::char_t *private_attr, ge::AnyValue private_attr_av);

  template<typename T, typename std::enable_if<(!std::is_array<T>::value), int32_t>::type = 0>
  static void *CreateCompileInfo() {
    return new T();
  }
  template<typename T>
  static void DeleteCompileInfo(void *const obj) {
    delete reinterpret_cast<T *>(obj);
  }
 private:
  std::unique_ptr<OpImplRegisterV2Impl> impl_;
};
}  // namespace gert

#define IMPL_OP_COUNTER(op_type, name, counter) \
  static gert::OpImplRegisterV2 VAR_UNUSED name##counter = gert::OpImplRegisterV2(#op_type)
#define IMPL_OP_COUNTER_NUMBER(op_type, name, counter) IMPL_OP_COUNTER(op_type, name, counter)
#define IMPL_OP(op_type) IMPL_OP_COUNTER_NUMBER(op_type, op_impl_register_##op_type, __COUNTER__)
#define IMPL_OP_DEFAULT() IMPL_OP(DefaultImpl)

#define IMPL_OP_INFERSHAPE(op_type) \
  gert::OpImplRegisterV2 VAR_UNUSED op_impl_register_infershape_##op_type = gert::OpImplRegisterV2(#op_type)
#define IMPL_OP_OPTILING(op_type) \
  gert::OpImplRegisterV2 VAR_UNUSED op_impl_register_optiling_##op_type = gert::OpImplRegisterV2(#op_type)

#endif  // INC_EXTERNAL_REGISTER_OP_IMPL_REGISTRY_H_
