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
#ifndef FRAMEWORK_TENSOR_COMPATIBLE_HIAI_AIPP_PARA_H
#define FRAMEWORK_TENSOR_COMPATIBLE_HIAI_AIPP_PARA_H
#include <cstdint>
#include <memory>

#include "tensor/tensor_api_export.h"
#include "compatible/hiai_base_types_cpt.h"

namespace hiai {
struct AippInputShape {
    uint32_t srcImageSizeW = 0;
    uint32_t srcImageSizeH = 0;
};

struct AippCropPara {
    bool switch_ = false;
    uint32_t cropStartPosW = 0;
    uint32_t cropStartPosH = 0;
    uint32_t cropSizeW = 0;
    uint32_t cropSizeH = 0;
};

struct AippResizePara {
    bool switch_ = false;
    uint32_t resizeOutputSizeW = 0;
    uint32_t resizeOutputSizeH = 0;
};

struct AippPaddingPara {
    bool switch_ = false;
    uint32_t paddingSizeTop = 0;
    uint32_t paddingSizeBottom = 0;
    uint32_t paddingSizeLeft = 0;
    uint32_t paddingSizeRight = 0;
    float paddingValueChn0 = 0;
    float paddingValueChn1 = 0;
    float paddingValueChn2 = 0;
    float paddingValueChn3 = 0;
};

struct AippCscPara {
    bool switch_ = false;
    int32_t matrixR0C0 = 0;
    int32_t matrixR0C1 = 0;
    int32_t matrixR0C2 = 0;
    int32_t matrixR1C0 = 0;
    int32_t matrixR1C1 = 0;
    int32_t matrixR1C2 = 0;
    int32_t matrixR2C0 = 0;
    int32_t matrixR2C1 = 0;
    int32_t matrixR2C2 = 0;
    int32_t outputBias0 = 0;
    int32_t outputBias1 = 0;
    int32_t outputBias2 = 0;
    int32_t inputBias0 = 0;
    int32_t inputBias1 = 0;
    int32_t inputBias2 = 0;
};

struct AippDtcPara {
    int16_t pixelMeanChn0 = 0;
    int16_t pixelMeanChn1 = 0;
    int16_t pixelMeanChn2 = 0;
    int16_t pixelMeanChn3 = 0;
    float pixelMinChn0 = 0;
    float pixelMinChn1 = 0;
    float pixelMinChn2 = 0;
    float pixelMinChn3 = 0;
    float pixelVarReciChn0 = 1.0;
    float pixelVarReciChn1 = 1.0;
    float pixelVarReciChn2 = 1.0;
    float pixelVarReciChn3 = 1.0;
};

struct AippChannelSwapPara {
    bool rbuvSwapSwitch = false;
    bool axSwapSwitch = false;
};

enum ImageType {
    JPEG = 0,
    BT_601_NARROW,
    BT_601_FULL,
    BT_709_NARROW,
};

class IAIPPPara;
class AippParaImpl;
class HIAI_TENSOR_API_EXPORT AippPara {
public:
    AippPara();
    ~AippPara();
    AippPara(const AippPara& para) = delete;
    AippPara& operator=(const AippPara& para) = delete;

    /*
     * @brief 初始化AippPara对象
     * @param [in] batchCount 模型输入的Batch数，对应NCHW维度的N维，默认为1
     * @return AIStatus::AI_SUCCESS AippPara初始化成功
     * @return Others 失败
     */
    AIStatus Init(uint32_t batchCount = 1);
    uint32_t GetBatchCount();

    /*
     * @brief 设置AIPP的inputIndex参数
     * @param [in] inputIndex 用于标识此AIPP参数作用于模型的第几个输入
     * @return AIStatus::AI_SUCCESS 设置成功
     * @return Others 失败
     */
    AIStatus SetInputIndex(uint32_t inputIndex);
    int32_t GetInputIndex();

    /*
     * @brief 设置AIPP的inputAippIndex参数
     * @param [in] inputAippIndex 用于标识AIPP配置参数在输入Data有多个输出分支时作用于第几个分支
     * @return AIStatus::AI_SUCCESS 设置成功
     * @return Others 失败
     */
    AIStatus SetInputAippIndex(uint32_t inputAippIndex);
    int32_t GetInputAippIndex();

    /*
     * @brief 设置AIPP的inputShape参数，设置输入图片的长宽
     * @param [in] inputShape 输入图片尺寸
     * @return AIStatus::AI_SUCCESS 设置成功
     * @return Others 失败
     */
    AIStatus SetInputShape(AippInputShape inputShape);
    AippInputShape GetInputShape();

    /*
     * @brief 设置AIPP的inputFormat参数，设置输入图片的类型
     * @param [in] inputFormat 输入图片的类型
     * @return AIStatus::AI_SUCCESS 设置成功
     * @return Others 失败
     */
    AIStatus SetInputFormat(AiTensorImage_Format inputFormat);
    AiTensorImage_Format GetInputFormat();

    /*
     * @brief 设置AIPP的CSC色域转换相关参数
     * @param [in] targetFormat 转换后的图片类型，系统根据转换后的图片类型自动填充CSC相关参数
     * @param [in] imageType 图片格式，默认JPEG
     * @return AIStatus::AI_SUCCESS 设置成功
     * @return Others 失败
     */
    AIStatus SetCscPara(AiTensorImage_Format targetFormat, ImageType imageType = JPEG);

    /*
     * @brief 设置AIPP的CSC色域转换相关参数
     * @param [in] cscPara csc相关参数
     * @return AIStatus::AI_SUCCESS 设置成功
     * @return Others 失败
     */
    AIStatus SetCscPara(AippCscPara cscPara);
    AippCscPara GetCscPara();

    /*
     * @brief 设置AIPP的ChannelSwap通道交换相关参数
     * @param [in] channelSwapPara AIPP通道交换参数
     * @return AIStatus::AI_SUCCESS 设置成功
     * @return Others 失败
     */
    AIStatus SetChannelSwapPara(AippChannelSwapPara channelSwapPara);
    AippChannelSwapPara GetChannelSwapPara();

    /*
     * @brief 设置AIPP的Crop图片裁剪相关参数，为所有的Batch设置相同的Crop参数
     * @param [in] cropPara crop相关参数
     * @return AIStatus::AI_SUCCESS 设置成功
     * @return Others 失败
     */
    AIStatus SetCropPara(AippCropPara cropPara);

    /*
     * @brief 设置AIPP的Crop图片裁剪相关参数，为特定下标的batch设置Crop参数
     * @param [in] batchIndex batch下标，标识为第几个batch设置Crop参数
     * @param [in] cropPara crop相关参数
     * @return AIStatus::AI_SUCCESS 设置成功
     * @return Others 失败
     */
    AIStatus SetCropPara(uint32_t batchIndex, AippCropPara cropPara);
    AippCropPara GetCropPara(uint32_t batchIndex);

    /*
     * @brief 设置AIPP的Resize图片裁剪相关参数，为所有的Batch设置相同的Resize参数
     * @param [in] resizePara Resize相关参数
     * @return AIStatus::AI_SUCCESS 设置成功
     * @return Others 失败
     */
    AIStatus SetResizePara(AippResizePara resizePara);

    /*
     * @brief 设置AIPP的Resize图片裁剪相关参数，为特定下标的batch设置Resize参数
     * @param [in] batchIndex batch下标，标识为第几个batch设置Resize参数
     * @param [in] resizePara Resize相关参数
     * @return AIStatus::AI_SUCCESS 设置成功
     * @return Others 失败
     */
    AIStatus SetResizePara(uint32_t batchIndex, AippResizePara resizePara);
    AippResizePara GetResizePara(uint32_t batchIndex);

    /*
     * @brief 设置AIPP的Padding图片补边相关参数，为所有的Batch设置相同的Padding参数
     * @param [in] paddingPara Padding相关参数
     * @return AIStatus::AI_SUCCESS 设置成功
     * @return Others 失败
     */
    AIStatus SetPaddingPara(AippPaddingPara paddingPara);

    /*
     * @brief 设置AIPP的Padding图片补边相关参数，为特定下标的batch设置Padding相关参数
     * @param [in] batchIndex batch下标，标识为第几个batch设置Padding参数
     * @param [in] paddingPara Padding相关参数
     * @return AIStatus::AI_SUCCESS 设置成功
     * @return Others 失败
     */
    AIStatus SetPaddingPara(uint32_t batchIndex, AippPaddingPara paddingPara);
    AippPaddingPara GetPaddingPara(uint32_t batchIndex);

    /*
     * @brief 设置AIPP的DTC数据类型转换相关参数，为所有的Batch设置相同的DTC参数
     * @param [in] dtcPara DTC相关参数
     * @return AIStatus::AI_SUCCESS 设置成功
     * @return Others 失败
     */
    AIStatus SetDtcPara(AippDtcPara dtcPara);

    /*
     * @brief 设置AIPP的DTC数据类型转换相关参数，为特定下标的batch设置DTC相关参数
     * @param [in] batchIndex batch下标，标识为第几个batch设置DTC参数
     * @param [in] dtcPara DTC相关参数
     * @return AIStatus::AI_SUCCESS 设置成功
     * @return Others 失败
     */
    AIStatus SetDtcPara(uint32_t batchIndex, AippDtcPara dtcPara);
    AippDtcPara GetDtcPara(uint32_t batchIndex);

    std::shared_ptr<IAIPPPara> GetAIPPPara();
    void SetAIPPPara(std::shared_ptr<IAIPPPara>& para);

private:
    std::shared_ptr<AippParaImpl> impl_ = nullptr;
};
} // namespace hiai

#endif
