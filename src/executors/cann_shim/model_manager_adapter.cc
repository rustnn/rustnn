/**
 * CANN Adapter Layer - AiModelMngerClient Implementation
 */

#include "model_manager_adapter.h"
#include "adapter_internal.h"

#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "compatible/hiai_base_types_cpt.h"

namespace ddk {
extern "C" {

/* ── Model Description ──────────────────────────────────────────────────── */

CannModelDescHandle cann_model_desc_create(const char* name,
                                             int32_t frequency,
                                             int32_t framework,
                                             int32_t model_type,
                                             int32_t device_type) {
    if (!name) return nullptr;
    return new CannModelDescImpl(std::string(name), frequency, framework,
                                  model_type, device_type);
}

void cann_model_desc_destroy(CannModelDescHandle desc) {
    delete desc;
}

const char* cann_model_desc_get_name(CannModelDescHandle desc) {
    if (!desc) return nullptr;
    return desc->desc->GetName().c_str();
}

CannStatus cann_model_desc_set_model_buffer(CannModelDescHandle desc,
                                              const void* data,
                                              uint32_t size) {
    if (!desc || !data || size == 0) return kInvalidPara;
    hiai::AIStatus ret = desc->desc->SetModelBuffer(data, size);
    return (ret == hiai::AI_SUCCESS) ? kSuccess : kFailed;
}

CannStatus cann_model_desc_set_input_dims(CannModelDescHandle desc,
                                            CannIOTensorDimensionHandle* dims,
                                            int32_t dim_count) {
    if (!desc || !dims || dim_count <= 0) return kInvalidPara;
    std::vector<hiai::TensorDimension> hDims;
    hDims.reserve(static_cast<size_t>(dim_count));
    for (int32_t i = 0; i < dim_count; ++i) {
        if (!dims[i]) return kInvalidPtr;
        hDims.push_back(dims[i]->dim);
    }
    hiai::AIStatus ret = desc->desc->SetInputDims(hDims);
    return (ret == hiai::AI_SUCCESS) ? kSuccess : kFailed;
}

CannStatus cann_model_desc_set_dynamic_shape(CannModelDescHandle desc,
                                               int32_t enable,
                                               uint32_t max_cached_num) {
    if (!desc) return kInvalidPtr;
    hiai::DynamicShapeConfig config;
    config.enable = (enable != 0);
    config.maxCachedNum = max_cached_num;
    config.cacheMode = hiai::CACHE_BUILDED_MODEL;
    hiai::AIStatus ret = desc->desc->SetDynamicShapeConfig(config);
    return (ret == hiai::AI_SUCCESS) ? kSuccess : kFailed;
}

CannStatus cann_model_desc_get_dynamic_shape_config(CannModelDescHandle desc,
                                                      int32_t* out_enable,
                                                      uint32_t* out_max_cached_num) {
    if (!desc || !out_enable || !out_max_cached_num) return kInvalidPtr;
    hiai::DynamicShapeConfig config;
    hiai::AIStatus ret = desc->desc->GetDynamicShapeConfig(config);
    if (ret != hiai::AI_SUCCESS) return kFailed;
    *out_enable = config.enable ? 1 : 0;
    *out_max_cached_num = config.maxCachedNum;
    return kSuccess;
}

CannStatus cann_model_desc_get_input_dims(CannModelDescHandle desc,
                                            CannIOTensorDimensionHandle* dims,
                                            int32_t max_dims,
                                            int32_t* out_dim_count) {
    if (!desc || !dims || max_dims <= 0 || !out_dim_count) return kInvalidPara;
    std::vector<hiai::TensorDimension> hDims;
    hiai::AIStatus ret = desc->desc->GetInputDims(hDims);
    if (ret != hiai::AI_SUCCESS) return kFailed;
    *out_dim_count = static_cast<int32_t>(hDims.size());
    for (size_t i = 0; i < hDims.size() && static_cast<int32_t>(i) < max_dims; ++i) {
        dims[i] = new CannIOTensorDimImpl(hDims[i]);
    }
    return kSuccess;
}

CannStatus cann_model_desc_set_precision_mode(CannModelDescHandle desc,
                                                int32_t precision_mode) {
    if (!desc) return kInvalidPtr;
    hiai::AIStatus ret = desc->desc->SetPrecisionMode(
        static_cast<hiai::PrecisionMode>(precision_mode));
    return (ret == hiai::AI_SUCCESS) ? kSuccess : kFailed;
}

CannStatus cann_model_desc_get_precision_mode(CannModelDescHandle desc,
                                                int32_t* out_precision_mode) {
    if (!desc || !out_precision_mode) return kInvalidPtr;
    hiai::PrecisionMode mode;
    hiai::AIStatus ret = desc->desc->GetPrecisionMode(mode);
    if (ret != hiai::AI_SUCCESS) return kFailed;
    *out_precision_mode = static_cast<int32_t>(mode);
    return kSuccess;
}

CannStatus cann_model_desc_set_tuning_strategy(CannModelDescHandle desc,
                                                 int32_t strategy) {
    if (!desc) return kInvalidPtr;
    hiai::AIStatus ret = desc->desc->SetTuningStrategy(
        static_cast<hiai::TuningStrategy>(strategy));
    return (ret == hiai::AI_SUCCESS) ? kSuccess : kFailed;
}

CannStatus cann_model_desc_get_tuning_strategy(CannModelDescHandle desc,
                                                 int32_t* out_strategy) {
    if (!desc || !out_strategy) return kInvalidPtr;
    *out_strategy = static_cast<int32_t>(desc->desc->GetTuningStrategy());
    return kSuccess;
}

/* ── Model Manager Client ───────────────────────────────────────────────── */

CannModelManagerHandle cann_model_manager_create(void) {
    return reinterpret_cast<CannModelManagerHandle>(new CannModelMgrImpl());
}

void cann_model_manager_destroy(CannModelManagerHandle manager) {
    delete reinterpret_cast<CannModelMgrImpl*>(manager);
}

CannStatus cann_model_manager_init(CannModelManagerHandle manager) {
    if (!manager) return kInvalidPtr;
    hiai::AIStatus ret =
        reinterpret_cast<CannModelMgrImpl*>(manager)->client.Init(nullptr);
    return (ret == hiai::AI_SUCCESS) ? kSuccess : kFailed;
}

CannStatus cann_model_manager_load(CannModelManagerHandle manager,
                                     CannModelDescHandle* descs,
                                     int32_t desc_count) {
    if (!manager || !descs || desc_count <= 0) return kInvalidPara;
    auto* mgr = reinterpret_cast<CannModelMgrImpl*>(manager);
    std::vector<std::shared_ptr<hiai::AiModelDescription>> hDescs;
    hDescs.reserve(static_cast<size_t>(desc_count));
    for (int32_t i = 0; i < desc_count; ++i) {
        hDescs.push_back(descs[i]->desc);
    }
    hiai::AIStatus ret = mgr->client.Load(hDescs);
    return (ret == hiai::AI_SUCCESS) ? kSuccess : kFailed;
}

CannStatus cann_model_manager_process(
    CannModelManagerHandle manager,
    CannContextHandle context,
    CannIOTensorHandle* inputs,
    int32_t input_count,
    CannIOTensorHandle* outputs,
    int32_t output_count,
    uint32_t timeout,
    int32_t* out_stamp) {
    if (!manager || !context || !inputs || !outputs || !out_stamp)
        return kInvalidPtr;
    if (input_count <= 0 || output_count <= 0) return kInvalidPara;

    auto* mgr = reinterpret_cast<CannModelMgrImpl*>(manager);
    auto* ctxImpl = reinterpret_cast<CannContextImpl*>(context);

    /* Build hiai::AiTensor shared_ptrs for the Process call. */
    std::vector<std::shared_ptr<hiai::AiTensor>> inTensors;
    std::vector<std::shared_ptr<hiai::AiTensor>> outTensors;
    // TODO: We want as less data copying as possible.
    for (int32_t i = 0; i < input_count; ++i) {
        auto* t = reinterpret_cast<CannIOTensorImpl*>(inputs[i]);
        inTensors.push_back(t->tensor);  // Assuming CannIOTensorImpl::tensor is already a shared_ptr
    }

    for (int32_t i = 0; i < output_count; ++i) {
        auto* t = reinterpret_cast<CannIOTensorImpl*>(outputs[i]);
        outTensors.push_back(t->tensor);  // Assuming CannIOTensorImpl::tensor is already a shared_ptr
    }

    int32_t stamp = 0;
    hiai::AIStatus ret = mgr->client.Process(
        ctxImpl->context, inTensors, outTensors, timeout, stamp);
    *out_stamp = stamp;

    return (ret == hiai::AI_SUCCESS) ? kSuccess : kFailed;
}

CannStatus cann_model_manager_unload(CannModelManagerHandle manager) {
    if (!manager) return kInvalidPtr;
    hiai::AIStatus ret =
        reinterpret_cast<CannModelMgrImpl*>(manager)->client.UnLoadModel();
    return (ret == hiai::AI_SUCCESS) ? kSuccess : kFailed;
}

const char* cann_model_manager_get_version(CannModelManagerHandle manager) {
    if (!manager) return nullptr;
    char* version =
        reinterpret_cast<CannModelMgrImpl*>(manager)->client.GetVersion();
    return version;
}

CannStatus cann_model_manager_check_compatibility(
    CannModelManagerHandle manager,
    CannModelDescHandle desc,
    int32_t* out_compatible) {
    if (!manager || !desc || !out_compatible) return kInvalidPtr;
    auto* mgr = reinterpret_cast<CannModelMgrImpl*>(manager);
    bool compatible = false;
    hiai::AIStatus ret = mgr->client.CheckModelCompatibility(*(desc->desc), compatible);
    *out_compatible = compatible ? 1 : 0;
    return (ret == hiai::AI_SUCCESS) ? kSuccess : kFailed;
}

CannStatus cann_model_manager_get_model_io_tensor_dim(
    CannModelManagerHandle manager,
    const char* model_name,
    CannIOTensorDimensionHandle* input_dims,
    int32_t max_input_dims,
    int32_t* out_input_dim_count,
    CannIOTensorDimensionHandle* output_dims,
    int32_t max_output_dims,
    int32_t* out_output_dim_count) {
    if (!manager || !model_name || !out_input_dim_count || !out_output_dim_count)
        return kInvalidPtr;
    auto* mgr = reinterpret_cast<CannModelMgrImpl*>(manager);
    std::vector<hiai::TensorDimension> inDims;
    std::vector<hiai::TensorDimension> outDims;
    hiai::AIStatus ret = mgr->client.GetModelIOTensorDim(
        std::string(model_name), inDims, outDims);
    if (ret != hiai::AI_SUCCESS) return kFailed;

    *out_input_dim_count = static_cast<int32_t>(inDims.size());
    *out_output_dim_count = static_cast<int32_t>(outDims.size());
    if (input_dims) {
        for (size_t i = 0; i < inDims.size() && static_cast<int32_t>(i) < max_input_dims; ++i)
            input_dims[i] = new CannIOTensorDimImpl(inDims[i]);
    }
    if (output_dims) {
        for (size_t i = 0; i < outDims.size() && static_cast<int32_t>(i) < max_output_dims; ++i)
            output_dims[i] = new CannIOTensorDimImpl(outDims[i]);
    }
    return kSuccess;
}

CannStatus cann_model_manager_set_priority(
    CannModelManagerHandle manager,
    const char* model_name,
    int32_t priority) {
    if (!manager || !model_name) return kInvalidPtr;
    auto* mgr = reinterpret_cast<CannModelMgrImpl*>(manager);
    hiai::AIStatus ret = mgr->client.SetModelPriority(
        std::string(model_name), static_cast<hiai::ModelPriority>(priority));
    return (ret == hiai::AI_SUCCESS) ? kSuccess : kFailed;
}

}  // extern "C"
}  // namespace ddk
