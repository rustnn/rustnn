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

#ifndef CLOUD_SERVICE_NEW_H
#define CLOUD_SERVICE_NEW_H
#include <atomic>

#ifdef HIAI_CLOUD_SERVICE_API_VISIABLE
#ifdef _MSC_VER
#define HIAI_CLOUD_SERVICE_API_EXPORT __declspec(dllexport)
#else
#define HIAI_CLOUD_SERVICE_API_EXPORT __attribute__((__visibility__("default")))
#endif
#else
#define HIAI_CLOUD_SERVICE_API_EXPORT
#endif

namespace hiai {
class HIAI_CLOUD_SERVICE_API_EXPORT CloudService {
public:
    static bool Enable(void* env, void* context);
    static const char* GetComputeCapabilityVersion();

private:
    CloudService() = default;
    ~CloudService() = default;
    static std::atomic<bool> isSoLoaded_;
};
} // end namespace hiai
#endif
