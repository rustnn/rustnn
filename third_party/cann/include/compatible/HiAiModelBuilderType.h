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
#ifndef FRAMEWORK_BUILD_OPTION_TYPES_H
#define FRAMEWORK_BUILD_OPTION_TYPES_H

#include <string>
#include <vector>
#include <map>

namespace hiai {
enum WeightDataType { FP32, FP16 };

enum OpExecuteDevice {
    NPU_DEV = 0,
    CPU_DEV = 1,
};

enum ExecuteDeviceSelectMode {
    AUTO,
    CUSTOM,
};

enum class TuningStrategy { OFF = 0, ON_DEVICE_TUNING, ON_DEVICE_PREPROCESS_TUNING, ON_CLOUD_TUNING };

enum ModelPriority {
    PRIORITY_HIGH = 5,
    PRIORITY_MIDDLE = 6,
    PRIORITY_LOW = 7,
};

enum PrecisionMode { PRECISION_MODE_FP32 = 0, PRECISION_MODE_FP16 };

enum class ExecuteDevice { NPU = 0, CPU = 1, GPU = 2, ISP = 3, DEVICE_TYPE_RESERVED };

enum class DeviceMemoryReusePlan { UNSET = 0, LOW = 1, HIGH = 2 };

struct BuildOptions {
    bool useOriginFormat = false;
    ExecuteDeviceSelectMode mode = AUTO;
    WeightDataType weightDataType = FP32;
    std::map<std::string, std::vector<OpExecuteDevice>> opExecuteDeviceConfig;
    std::vector<std::vector<int64_t>> inputShapes;
    TuningStrategy tuningStrategy = TuningStrategy::OFF;
    std::vector<ExecuteDevice> modelDeviceOrder;
    DeviceMemoryReusePlan deviceMemoryReusePlan = DeviceMemoryReusePlan::UNSET;
    std::string quantizeConfig = "";
};

enum CacheMode { CACHE_BUILDED_MODEL, CACHE_LOADED_MODEL };
struct DynamicShapeConfig {
    bool enable = false; // mark whether dynamic shape enable.
    uint32_t maxCachedNum = 0; // max cache model
    CacheMode cacheMode = CacheMode::CACHE_BUILDED_MODEL;
};
} // namespace hiai

#endif