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
#ifndef FRAMEWORK_TENSOR_COMPATIBLE_AIPP_TENSOR_H
#define FRAMEWORK_TENSOR_COMPATIBLE_AIPP_TENSOR_H
#include <vector>
#include "tensor/tensor_api_export.h"
#include "HiAiAippPara.h"
#include "compatible/AiTensor.h"

namespace hiai {
class AippTensorImpl;
class HIAI_TENSOR_API_EXPORT AippTensor : public AiTensor {
public:
    AippTensor(std::shared_ptr<AiTensor> tensor, std::vector<std::shared_ptr<AippPara>> aippParas);

    ~AippTensor() override;

    void* GetBuffer() const override;

    uint32_t GetSize() const override;

    std::shared_ptr<AiTensor> GetAiTensor() const;

    std::vector<std::shared_ptr<AippPara>> GetAippParas() const;

    std::shared_ptr<AippPara> GetAippParas(uint32_t index) const;

private:
    using AiTensor::Init;

    std::shared_ptr<AippTensorImpl> impl_ = nullptr;
};

HIAI_TENSOR_API_EXPORT std::shared_ptr<AippTensor> HIAI_CreateAiPPTensorFromHandle(
    void* nativeHandle, const TensorDimension* dim, AiTensorImage_Format imageFormat = AiTensorImage_INVALID);
} // namespace hiai
#endif
