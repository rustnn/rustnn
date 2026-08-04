/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef GE_ERROR_CODES_H
#define GE_ERROR_CODES_H

#include <cstdint>
namespace ge {
using GraphErrCodeStatus = uint32_t;
using graphStatus = uint32_t;
constexpr GraphErrCodeStatus GRAPH_FAILED = 1;
constexpr GraphErrCodeStatus GRAPH_SUCCESS = 0;
constexpr GraphErrCodeStatus GRAPH_PARAM_INVALID = 3;
} // namespace ge

#endif // GE_ERROR_CODES_H
