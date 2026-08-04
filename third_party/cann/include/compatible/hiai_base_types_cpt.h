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
#ifndef FRAMEWORK_TENSOR_COMPATIBLE_HIAI_BASE_TYPES_CPT_H
#define FRAMEWORK_TENSOR_COMPATIBLE_HIAI_BASE_TYPES_CPT_H
#include <cstdint>

namespace hiai {
using AIStatus = int32_t;

/* Error Numbers */
static const AIStatus AI_SUCCESS = 0;
static const AIStatus AI_FAILED = 1;
static const AIStatus AI_NOT_INIT = 2;
static const AIStatus AI_INVALID_PARA = 3;
static const AIStatus AI_INVALID_API = 7;
static const AIStatus AI_INVALID_POINTER = 8;

enum HIAI_DataType {
    HIAI_DATATYPE_UINT8 = 0,
    HIAI_DATATYPE_FLOAT32 = 1,
    HIAI_DATATYPE_FLOAT16 = 2,
    HIAI_DATATYPE_INT32 = 3,
    HIAI_DATATYPE_INT8 = 4,
    HIAI_DATATYPE_INT16 = 5,
    HIAI_DATATYPE_BOOL = 6,
    HIAI_DATATYPE_INT64 = 7,
    HIAI_DATATYPE_UINT32 = 8,
    HIAI_DATATYPE_DOUBLE = 9,
};

enum AiTensorImage_Format {
    AiTensorImage_YUV420SP_U8 = 0,
    AiTensorImage_XRGB8888_U8,
    AiTensorImage_YUV400_U8,
    AiTensorImage_ARGB8888_U8,
    AiTensorImage_YUYV_U8,
    AiTensorImage_YUV422SP_U8,
    AiTensorImage_AYUV444_U8,
    AiTensorImage_RGB888_U8,
    AiTensorImage_BGR888_U8,
    AiTensorImage_YUV444SP_U8,
    AiTensorImage_YVU444SP_U8,
    AiTensorImage_INVALID = 255,
};

enum AiModelDescription_Frequency {
    AiModelDescription_Frequency_LOW = 1,
    AiModelDescription_Frequency_MEDIUM = 2,
    AiModelDescription_Frequency_HIGH = 3,
    AiModelDescription_Frequency_EXTREME = 4,
    AiModelDescription_Frequency_HIGH_COMPUTE_UNIT = 103,
    AiModelDescription_Frequency_MEDIUM_BAND_MEDIUM_COMPUTE_UNIT = 202,
};

enum AiModelDescription_DeviceType {
    AiModelDescription_DeviceType_NPU = 0,
    AiModelDescription_DeviceType_IPU = 1,
    AiModelDescription_DeviceType_MLU = 2,
    AiModelDescription_DeviceType_CPU = 3,
};

enum AiModelDescription_Framework {
    HIAI_FRAMEWORK_NONE = 0,
    HIAI_FRAMEWORK_TENSORFLOW,
    HIAI_FRAMEWORK_KALDI,
    HIAI_FRAMEWORK_CAFFE,
    HIAI_FRAMEWORK_TENSORFLOW_8BIT,
    HIAI_FRAMEWORK_CAFFE_8BIT,
};

enum AiModelDescription_ModelType { HIAI_MODELTYPE_ONLINE = 0, HIAI_MODELTYPE_OFFLINE };
} // namespace hiai
#endif // FRAMEWORK_TENSOR_COMPATIBLE_HIAI_BASE_TYPES_CPT_H
