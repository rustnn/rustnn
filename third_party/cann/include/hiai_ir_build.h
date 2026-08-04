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
#ifndef HIAI_API_HIAI_IR_BUILD_H
#define HIAI_API_HIAI_IR_BUILD_H

#include "graph/model.h"
#include "compatible/HiAiModelBuilderType.h"

namespace hiai {
class IBuiltModel;
using Status = uint32_t;
struct ModelBufferData {
    void* data = nullptr;
    uint32_t length = 0;
};

const static BuildOptions defaultBuildOptions;

class HiaiIrBuild {
public:
    /**
     * @ingroup domi_omg
     * @brief 创建模型Buffer, 默认分配200M
     * @param [in] irModel 输入模型数据
     * @param [out] output 输出离线模型Buffer
     * @return bool 执行结果
     */
    bool CreateModelBuff(ge::Model& irModel, ModelBufferData& output);

    /**
     * @ingroup domi_omg
     * @brief 创建模型Buffer
     * @param [in] irModel 输入模型数据
     * @param [out] output 输出离线模型Buffer
     * @param [in] customSize
     * 输入用户指定Buffer大小，单位字节，必须为非负整数，建议范围200M以内，当customSize设置为0时，接口内部根据irModel自动计算合适的Buffer大小
     * @return bool 执行结果
     */
    bool CreateModelBuff(ge::Model& irModel, ModelBufferData& output, uint32_t customSize);

    /*
     * @ingroup domi_omg
     * @brief 在线编译
     * @param [in] irModel 输入模型数据
     * @param [out] output 输出离线模型
     * @return bool 执行结果
     */
    bool BuildIRModel(ge::Model& irModel, ModelBufferData& output);

    /*
     * @ingroup domi_omg
     * @brief 在线编译
     * @param [in] options 输入编译配置参数
     * @param [in] irModel 输入模型数据
     * @param [out] output 输出离线模型
     * @return bool 执行结果
     */
    bool BuildIRModel(ge::Model& irModel, ModelBufferData& output, const BuildOptions& options);

    void ReleaseModelBuff(ModelBufferData& output);

private:
    /*
     * @ingroup domi_omg
     * @brief 在线编译生成模型
     * @return bool 成功返回true，失败返回false
     */
    bool BuildModelOnline(ge::Model& irModel, ModelBufferData& output, const BuildOptions& options);
    /*
     * @ingroup domi_omg
     * @brief 离线编译生成模型
     * @return bool 成功返回true，失败返回false
     */
    bool BuildModelOffline(ge::ComputeGraphPtr graph, ge::Model& irModel,
        ModelBufferData& output, const BuildOptions& options);
    /*
     * @ingroup domi_omg
     * @brief 检查Rom版本是否支持IR量化模型
     * @return bool 成功返回true，失败返回false
     */
    bool IsSupportQuantize(const std::string& hiaiVersion);
};
} // namespace hiai

namespace domi {
using hiai::BuildOptions;
using hiai::defaultBuildOptions;
using hiai::ExecuteDeviceSelectMode;
using hiai::HiaiIrBuild;
using hiai::ModelBufferData;
using hiai::OpExecuteDevice;
using hiai::WeightDataType;
} // namespace domi
#endif // HIAI_API_HIAI_IR_BUILD_H
