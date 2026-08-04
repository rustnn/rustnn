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
#ifndef FRAMEWORK_MODEL_MANAGER_COMPATIBLE_HIAI_MODEL_MANAGER_SERVICE_H
#define FRAMEWORK_MODEL_MANAGER_COMPATIBLE_HIAI_MODEL_MANAGER_SERVICE_H
#include <vector>
#include <string>
#include <memory>

#include "compatible/AiModelBuilder.h"
#include "compatible/HiAiModelBuilderType.h"
#include "HiAiAippPara.h"
#include "HiAiModelManagerType.h"

namespace hiai {
class AiModelDescriptionImpl;
class HIAI_MM_API_EXPORT AiModelDescription {
public:
    /*
     * @brief AiModelDescription初始化描述的构造函数
     * @param [in] pmodelName 模型名称
     * @param [in] frequency 算力要求等级：AiModelDescription_Frequency
     * @param [in] framework 模型平台类型：AiModelDescription_Framework
     * @param [in] pmodelType 模型类型：AiModelDescription_ModelType
     * @param [in] pdeviceType 设备类型：AiModelDescription_DeviceType
     */
    AiModelDescription(const std::string& pmodelName, const int32_t frequency, const int32_t framework,
        const int32_t pmodelType, const int32_t pdeviceType);

    virtual ~AiModelDescription();

    const std::string& GetName() const;
    void* GetModelBuffer() const;
    AIStatus SetModelBuffer(const void* data, uint32_t size);
    AIStatus SetModelPath(const std::string& modelPath);
    const std::string& GetModelPath() const;
    int32_t GetFrequency() const;
    int32_t GetFramework() const;
    int32_t GetModelType() const;
    int32_t GetDeviceType() const;
    uint32_t GetModelNetSize() const;
    AIStatus GetDynamicShapeConfig(DynamicShapeConfig& dynamicShape) const;
    AIStatus SetDynamicShapeConfig(const DynamicShapeConfig& dynamicShape);
    AIStatus GetInputDims(std::vector<TensorDimension>& inputDims) const;
    AIStatus SetInputDims(const std::vector<TensorDimension>& inputDims);
    AIStatus GetPrecisionMode(PrecisionMode& precisionMode) const;
    AIStatus SetPrecisionMode(const PrecisionMode& precisionMode);
    AIStatus SetTuningStrategy(const TuningStrategy& tuningStrategy);
    TuningStrategy GetTuningStrategy() const;
private:
    std::shared_ptr<AiModelDescriptionImpl> impl_ = nullptr;
};

class HIAI_MM_API_EXPORT AiModelManagerClientListener {
public:
    virtual ~AiModelManagerClientListener() = default;
    virtual void OnProcessDone(const AiContext& context, int32_t result,
        const std::vector<std::shared_ptr<AiTensor>>& poutTensor, int32_t piStamp) = 0;
    virtual void OnServiceDied() = 0;
};

class AiModelMngerClientImpl;
class HIAI_MM_API_EXPORT AiModelMngerClient {
public:
    AiModelMngerClient();
    virtual ~AiModelMngerClient();

    /*
     * @brief 初始化接口
     * @param [in] listener 监听接口, 设置为nullptr时为同步调用, 否则为异步
     * @return AIStatus::AI_SUCCESS 成功
     * @return Others 失败
     */
    AIStatus Init(std::shared_ptr<AiModelManagerClientListener> listener);

    /*
     * @brief 加载模型
     * @param [in] pmodelDesc 模型信息
     * @param [in] piStamp 异步返回标识，基于该标识和模型名称做回调索引
     * @return AIStatus::AI_SUCCESS 成功
     * @return AIStatus::AI_INVALID_API 失败，表示设备不支持NPU
     * @return Others 失败
     */
    AIStatus Load(std::vector<std::shared_ptr<AiModelDescription>>& pmodelDesc);

    /*
     * @brief 模型处理接口, 运行加载模型的模型推理
     * @param [in] context, 模型运行上下文, 必须带model_name字段
     * @param [in] pinputTensor, 模型输入节点tensor信息
     * @param [in] poutputTensor, 模型输出节点tensor信息
     * @param [in] timeout, 推理超时时间
     * @param [in] piStamp 异步返回标识，基于该标识和模型名称做回调索引
     * @return AIStatus::AI_SUCCESS 成功
     * @return Others 失败
     */
    AIStatus Process(AiContext& context, std::vector<std::shared_ptr<AiTensor>>& pinputTensor,
        std::vector<std::shared_ptr<AiTensor>>& poutputTensor, uint32_t timeout, int32_t& piStamp);

    /*
     * @brief 模型兼容性检查接口
     * @param [in] pmodelDesc, 模型描述
     * @param [out] pisModelCompatibility, 兼容性检查标识
     * @return AIStatus::AI_SUCCESS 兼容性检查通过
     * @return Others 兼容性检查失败
     */
    AIStatus CheckModelCompatibility(AiModelDescription& pmodelDesc, bool& pisModelCompatibility);

    /*
     * @brief 获取模型输入输出tensor信息
     * @param [in] pmodelName, 模型名
     * @param [out] pinputTensor 输出参数, 存储模型输入节点tensor信息
     * @param [out] poutputTensor 输出参数, 存储模型输出节点tensor信息
     * @return AIStatus::AI_SUCCESS 获取成功
     * @return Others 获取失败
     */
    AIStatus GetModelIOTensorDim(const std::string& pmodelName, std::vector<TensorDimension>& pinputTensor,
        std::vector<TensorDimension>& poutputTensor);

    /*
     * @brief 获取模型AIPP 配置信息
     * @param [in] pmodelName, 模型名
     * @param [out] aippPara 输出参数, 模型中的AIPP配置参数
     * @return AIStatus::AI_SUCCESS 获取成功
     * @return Others 获取失败
     */
    AIStatus GetModelAippPara(const std::string& modelName, std::vector<std::shared_ptr<AippPara>>& aippPara);

    /*
     * @brief 获取模型对应输入的AIPP 配置信息
     * @param [in] pmodelName, 模型名
     * @param [in] index, 输入下标
     * @param [out] aippPara 输出参数, 模型对应输入的AIPP配置参数
     * @return AIStatus::AI_SUCCESS 获取成功
     * @return Others 获取失败
     */
    AIStatus GetModelAippPara(
        const std::string& modelName, uint32_t index, std::vector<std::shared_ptr<AippPara>>& aippPara);

    /*
     * @brief 获取DDK版本
     * @return char * DDK版本
     * @return nullptr 获取失败
     */
    char* GetVersion();

    /*
     * @brief 卸载模型
     * @return AIStatus::AI_SUCCESS 卸载成功
     * @return Others 卸载失败
     */
    AIStatus UnLoadModel();

    /*
     * @brief 设置模型优先级
     * @param [in] modelName, 模型名
     * @param [in] modelPriority, 优先级
     * @return AIStatus::AI_SUCCESS 设置成功
     * @return Others 设置失败
     */
    AIStatus SetModelPriority(const std::string& modelName, ModelPriority modelPriority);

    /*
     * @brief 取消模型异步推理任务
     * @param [in] modelName, 模型名
     * @return void
     */
    void Cancel(const std::string& modelName);

private:
    std::shared_ptr<AiModelMngerClientImpl> impl_ = nullptr;
};
} // namespace hiai

#endif // FRAMEWORK_MODEL_MANAGER_COMPATIBLE_HIAI_MODEL_MANAGER_SERVICE_H
