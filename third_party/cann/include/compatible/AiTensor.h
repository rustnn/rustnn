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
#ifndef FRAMEWORK_TENSOR_COMPATIBLE_AITENSOR_H
#define FRAMEWORK_TENSOR_COMPATIBLE_AITENSOR_H
#include <memory>
#include <vector>

#include "tensor/tensor_api_export.h"
#include "compatible/hiai_base_types_cpt.h"
#include "util/base_types.h"

namespace hiai {
class TensorDimensionImpl;
class HIAI_TENSOR_API_EXPORT TensorDimension {
public:
    TensorDimension();
    virtual ~TensorDimension();

    /*
     * @brief Tensor初始化接口
     * @param [in] number 输入的Tensor的数量
     * @param [in] channel 输入的Tensor的通道数
     * @param [in] height 输入的Tensor的高度
     * @param [in] width 输入的Tensor的宽度
     * @return 无
     */
    TensorDimension(uint32_t number, uint32_t channel, uint32_t height, uint32_t width);

    TensorDimension(const std::vector<uint32_t>& ndDims, Format format = Format::ND);

    void SetNumber(const uint32_t number);

    uint32_t GetNumber() const;

    void SetChannel(const uint32_t channel);

    uint32_t GetChannel() const;

    void SetHeight(const uint32_t height);

    uint32_t GetHeight() const;

    void SetWidth(const uint32_t width);

    uint32_t GetWidth() const;

    Format GetFormat() const;

    std::vector<uint32_t> GetDims() const;

    bool IsEqual(const TensorDimension& dim);
private:
    std::shared_ptr<TensorDimensionImpl> impl_ = nullptr;
};

class AiTensorImpl;
class HIAI_TENSOR_API_EXPORT AiTensor {
public:
    AiTensor();
    virtual ~AiTensor();

    /*
     * @brief 根据dim，默认DataType是FLOAT32，创建ION内存，完成Tensor初始化
     * @param [in] dim 输入tensor的尺寸结构信息
     * @return AIStatus::AI_SUCCESS 成功
     * @return Others 失败
     */
    AIStatus Init(const TensorDimension* dim);
    /*
     * @brief 根据dim和DataType创建ION内存，完成Tensor初始化
     * @param [in] dim 输入tensor的尺寸结构信息
     * @param [in] DataType数据类型
     * @return AIStatus::AI_SUCCESS 成功
     * @return Others 失败
     */
    AIStatus Init(const TensorDimension* dim, HIAI_DataType pdataType);

    /*
     * @brief 根据用户传入的data初始化Tensor，不对数据进行拷贝，需要用户保证数据释放晚于Tensor的生命周期
     * @param [in] data 用户传入的输入数据
     * @param [in] dim 输入tensor的尺寸结构信息
     * @param [in] pdataType 数据类型
     * @return AIStatus::AI_SUCCESS 成功
     * @return Others 失败
     */
    AIStatus Init(const void* data, const TensorDimension* dim, HIAI_DataType pdataType);

    /*
     * @brief 根据用户传入的ION内存handle初始化Tensor，不对数据进行拷贝，需要用户保证数据释放晚于Tensor的生命周期
     * @param [in] handle ION内存信息
     * @param [in] dim 输入tensor的尺寸结构信息
     * @param [in] DataType数据类型
     * @return AIStatus::AI_SUCCESS 成功
     * @return Others 失败
     */
    AIStatus Init(const NativeHandle& handle, const TensorDimension* dim, HIAI_DataType pdataType);
    /*
     * @brief 根据number、height、width、format创建ION内存，完成Tensor初始化
     * @param [in] number 输入的Tensor的数量
     * @param [in] height 输入的Tensor的高度
     * @param [in] width 输入的Tensor的宽度
     * @param [in] format 输入图片的格式AiTensorImage_Format类型
     * @return AIStatus::AI_SUCCESS 成功
     * @return Others 失败
     */
    AIStatus Init(uint32_t number, uint32_t height, uint32_t width, AiTensorImage_Format format);

    /*
     * @brief 获取Tensor buffer地址接口
     * @return void* tensor buffer地址
     */
    virtual void* GetBuffer() const;

    /*
     * @brief 获取Tensor buffer内存大小
     * @return void* tensor buffer内存大小
     */
    virtual uint32_t GetSize() const;

    AIStatus SetTensorDimension(const TensorDimension* dim);

    TensorDimension GetTensorDimension() const;
    void* GetTensorBuffer() const;

private:
    AiTensor(const AiTensor&) = delete;
    AiTensor& operator=(const AiTensor&) = delete;

private:
    friend class AiModelMngerClientImpl;
    std::shared_ptr<AiTensorImpl> impl_ = nullptr;
};
} // namespace hiai
#endif
