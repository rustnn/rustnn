/* Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef INC_EXTERNAL_GRAPH_GE_ERROR_CODES_H_
#define INC_EXTERNAL_GRAPH_GE_ERROR_CODES_H_

#include "graph/debug/ge_error_codes.h"

#include <cstdint>

namespace ge {
#if(defined(HOST_VISIBILITY)) && (defined(__GNUC__))
#define GE_FUNC_HOST_VISIBILITY __attribute__((visibility("default")))
#else
#define GE_FUNC_HOST_VISIBILITY
#endif
#ifdef __GNUC__
#ifdef NO_METADEF_ABI_COMPATIABLE
#define ATTRIBUTED_DEPRECATED(replacement)
#define ATTRIBUTED_NOT_SUPPORT()
#else
#define ATTRIBUTED_DEPRECATED(replacement) __attribute__((deprecated("Please use " #replacement " instead.")))
#define ATTRIBUTED_NOT_SUPPORT() __attribute__((deprecated("The method will not be supported in the future.")))
#endif
#else
#ifdef NO_METADEF_ABI_COMPATIABLE
#define ATTRIBUTED_DEPRECATED(replacement)
#define ATTRIBUTED_NOT_SUPPORT()
#else
#define ATTRIBUTED_DEPRECATED(replacement) __declspec(deprecated("Please use " #replacement " instead."))
#define ATTRIBUTED_NOT_SUPPORT() __declspec(deprecated("The method will not be supported in the future."))
#endif
#endif

using Status = uint32_t;
using graphStatus = uint32_t;
const graphStatus SUCCESS = 0;

const graphStatus GRAPH_PARAM_OUT_OF_RANGE = 50331644;

const graphStatus GRAPH_ADD_OVERFLOW = 50331428;
const graphStatus GRAPH_MUL_OVERFLOW = 50331427;
}  // namespace ge

#endif  // INC_EXTERNAL_GRAPH_GE_ERROR_CODES_H_
