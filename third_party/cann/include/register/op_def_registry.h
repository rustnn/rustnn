/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef OP_DEF_REGISTRY_H
#define OP_DEF_REGISTRY_H

#include "register/op_def.h"
#include "register/op_def_factory.h"

#if defined(OP_PROTO_LIB)

#define OP_ADD(opType, ...)                                                                                            \
  static int g_##opType##_added = [](const char *name) {                                                               \
    opType op(#opType);                                                                                                \
    gert::OpImplRegisterV2 impl(#opType);                                                                              \
    impl.InferShape(op.GetInferShape())                                                                                \
      .InferShapeRange(op.GetInferShapeRange())                                                                        \
      .InferDataType(op.GetInferDataType());                                                                           \
    gert::OpImplRegisterV2 implReg(impl);                                                                              \
    return 0;                                                                                                          \
  }(#opType)

#elif defined(OP_TILING_LIB)

#define OP_ADD(opType, ...)                                                                                            \
  struct OpAddCompilerInfoPlaceholder##opType {};                                                                      \
  static ge::graphStatus TilingPrepare##opType(gert::TilingParseContext *context) { return ge::GRAPH_SUCCESS; }        \
  static int g_##opType##_added = [](const char *name) {                                                               \
    opType op(#opType);                                                                                                \
    gert::OpImplRegisterV2 impl(#opType);                                                                              \
    impl.Tiling(op.AICore().GetTiling());                                                                              \
    impl.TilingParse<OpAddCompilerInfoPlaceholder##opType>(TilingPrepare##opType);                                     \
    optiling::OpCheckFuncHelper(FUNC_CHECK_SUPPORTED, #opType, op.AICore().GetCheckSupport());                         \
    optiling::OpCheckFuncHelper(FUNC_OP_SELECT_FORMAT, #opType, op.AICore().GetOpSelectFormat());                      \
    optiling::OpCheckFuncHelper(FUNC_GET_OP_SUPPORT_INFO, #opType, op.AICore().GetOpSupportInfo());                    \
    optiling::OpCheckFuncHelper(FUNC_GET_SPECIFIC_INFO, #opType, op.AICore().GetOpSpecInfo());                         \
    optiling::OpCheckFuncHelper(#opType, op.AICore().GetParamGeneralize());                                            \
    gert::OpImplRegisterV2 implReg(impl);                                                                              \
    return 0;                                                                                                          \
  }(#opType)

#else

#define OP_ADD(opType, ...)                                                                                            \
  static int g_##opType##_added =                                                                                      \
      ops::OpDefFactory::OpDefRegister(#opType, [](const char *name) { return opType(#opType); })

#endif
#endif
