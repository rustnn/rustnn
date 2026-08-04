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
#ifndef FRAMEWORK_ERROR_TYPES_H
#define FRAMEWORK_ERROR_TYPES_H

#include <stdint.h>

namespace hiai {
using Status = uint32_t;

constexpr Status SUCCESS = 0;
constexpr Status FAILURE = 1;
constexpr Status UNINITIALIZED = 2;
constexpr Status INVALID_PARAM = 3;
constexpr Status TIMEOUT = 4;
constexpr Status UNSUPPORTED = 5;
constexpr Status MEMORY_EXCEPTION = 6;
constexpr Status INVALID_API = 7;
constexpr Status INVALID_POINTER = 8;
constexpr Status CALC_EXCEPTION = 9;
constexpr Status FILE_NOT_EXIST = 10;
constexpr Status COMM_EXCEPTION = 11;
constexpr Status DATA_OVERFLOW = 12;
constexpr Status CONNECT_EXCEPTION = 19;
} // namespace hiai
#endif // FRAMEWORK_ERROR_TYPES_H
