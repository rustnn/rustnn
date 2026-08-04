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
#ifndef HIAI_FRAMEWOKR_UTIL_BASE_TYPES_H
#define HIAI_FRAMEWOKR_UTIL_BASE_TYPES_H

#include <cstdint>

namespace hiai {
enum class DataType {
    UINT8,
    FLOAT32,
    FLOAT16,
    INT32,
    INT8,
    INT16,
    BOOL,
    INT64,
    UINT32,
    DOUBLE,
    RESERVED
};

enum class Format {
    NCHW = 0, /* NCHW */
    NHWC, /* NHWC */
    ND, /* Nd Tensor */
    RESERVED
};

struct NativeHandle {
    int fd;
    int size;
    int offset;
};

static constexpr uint32_t MEM_ALLOC_ALIGN = 512;
}; // namespace hiai
#endif
