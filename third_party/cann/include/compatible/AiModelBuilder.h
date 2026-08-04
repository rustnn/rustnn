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
#ifndef FRAMEWORK_MODEL_MANAGER_COMPATIBLE_AIMODELBUILDER_H
#define FRAMEWORK_MODEL_MANAGER_COMPATIBLE_AIMODELBUILDER_H
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "model_manager/model_manager_api_export.h"
#include "compatible/hiai_base_types_cpt.h"
#include "compatible/MemBuffer.h"
#include "compatible/AiTensor.h"
#include "HiAiModelBuilderType.h"

namespace hiai {
class AiModelMngerClient;
class AiModelBuilderImpl;

class HIAI_MM_API_EXPORT AiModelBuilder {
public:
    AiModelBuilder(std::shared_ptr<AiModelMngerClient> client = nullptr);

    virtual ~AiModelBuilder();

    /*
     * @brief OM离线模型在线编译接口
     * @param [in] pinputMemBuffer 输入的OM离线模型buffer
     * @param [in] poutputModelBuffer 输出模型buffer
     * @param [out] poutputModelSize 输出模型大小
     * @return AIStatus::AI_SUCCESS 成功
     * @return Others 失败
     */
    AIStatus BuildModel(
        const std::vector<MemBuffer*>& pinputMemBuffer, MemBuffer* poutputModelBuffer, uint32_t& poutputModelSize);

    /*
     * @brief OM离线模型在线编译接口
     * @param [in] options 输入编译配置参数
     * @param [in] pinputMemBuffer 输入的OM离线模型buffer
     * @param [in] poutputModelBuffer 输出模型buffer
     * @param [out] poutputModelSize 输出模型大小
     * @return AIStatus::AI_SUCCESS 成功
     * @return Others 失败
     */
    AIStatus BuildModel(const std::vector<MemBuffer*>& pinputMemBuffer, MemBuffer* poutputModelBuffer,
        uint32_t& poutputModelSize, const BuildOptions& options);

    /*
     * @brief 从文件读取OM离线模型proto信息
     * @param [in] path, 模型文件路径
     * @return MemBuffer * proto信息存储地址
     * @return nullptr 获取失败
     */
    MemBuffer* ReadBinaryProto(const std::string path);

    /*
     * @brief 从内存读取OM离线模型proto信息
     * @param [in] data, OM离线模型内存地址
     * @param [in] size, OM离线模型内存存储大小
     * @return MemBuffer * proto信息存储地址
     * @return nullptr 获取失败
     */
    MemBuffer* ReadBinaryProto(void* data, uint32_t size);

    /*
     * @brief 为输入OM离线模型用户内存buffer创建通用MemBuffer
     * @param [in] data, 模型用户内存地址
     * @param [in] size, 模型内存存储大小
     * @return MemBuffer * proto信息存储地址
     * @return nullptr 获取失败
     */
    MemBuffer* InputMemBufferCreate(void* data, uint32_t size);

    /*
     * @brief 为输入OM离线模型从文件创建MemBuffer
     * @param [in] path 文件路径
     * @return MemBuffer * 创建的输入MemBuffer内存指针
     * @return nullptr 创建失败
     */
    MemBuffer* InputMemBufferCreate(const std::string path);

    /*
     * @brief 为在线编译输出模型创建MemBuffer
     * @param [in] framework 模型平台类型
     * @param [in] pinputMemBuffer 输入的OM离线模型buffer
     * @return MemBuffer * 创建的输出模型MemBuffer内存指针
     * @return nullptr 创建失败
     */
    MemBuffer* OutputMemBufferCreate(const int32_t framework, const std::vector<MemBuffer*>& pinputMemBuffer);

    /*
     * @brief 注销MemBuffer内存，通过上述MemBuffer申请的内存最终都需要由此接口进行释放
     * @param [in] membuf, 创建的MemBuffer内存
     * @return void
     */
    void MemBufferDestroy(MemBuffer* membuf);

    /*
     * @brief 将模型buffer导出到文件
     * @param [in] membuf, 存储离线模型信息内存指针
     * @param [in] pbuildSize, 离线模型大小
     * @param [in] pbuildPath, 离线模型导出文件存储路径
     * @return AIStatus::AI_SUCCESS 导出成功
     * @return Others 导出失败
     */
    AIStatus MemBufferExportFile(MemBuffer* membuf, const uint32_t pbuildSize, const std::string pbuildPath);

private:
    std::shared_ptr<AiModelBuilderImpl> impl_ = nullptr;
};
} // namespace hiai

#endif
