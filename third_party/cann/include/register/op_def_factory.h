/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef OP_DEF_FACTORY_H
#define OP_DEF_FACTORY_H

#include "register/op_def.h"

namespace ops {
using OpDefCreator = std::function<OpDef(const char *)>;
class OpDefFactory {
public:
  static int OpDefRegister(const char *name, OpDefCreator creator);

private:
  friend class AclnnOpGenerator;
  friend class Generator;
  friend class OpProtoGenerator;
  friend class GeneratorFactory;
  friend class CfgGenerator;
  friend class OpDefUtils;

  static OpDef OpDefCreate(const char *name);
  static std::vector<ge::AscendString> &GetAllOp(void);
};
}  // namespace ops

#endif
