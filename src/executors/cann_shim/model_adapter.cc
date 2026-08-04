/**
 * CANN Adapter Layer - Model Build Implementation
 *
 * Wraps ge::Model and hiai::HiaiIrBuild.
 */

#include "model_adapter.h"
#include "adapter_internal.h"

#include <string>

namespace ddk {
extern "C" {

/* ── Model lifecycle ────────────────────────────────────────────────── */

CannModelHandle cann_model_create() {
    return new CannModelImpl();
}

CannModelHandle cann_model_create_with_name(const char* name) {
    if (!name) return nullptr;
    return new CannModelImpl(std::string(name));
}

void cann_model_destroy(CannModelHandle model) {
    delete model;
}

/* ── Model / Graph ──────────────────────────────────────────────────── */

CannStatus cann_model_set_graph(CannModelHandle model, CannGraphHandle graph) {
    if (!model || !graph) return kInvalidPtr;
    model->model.SetGraph(graph->graph);
    return kSuccess;
}

CannGraphHandle cann_model_get_graph(CannModelHandle model) {
    if (!model) return nullptr;
    ge::Graph g = model->model.GetGraph();
    auto* impl = new CannGraphImpl();
    impl->graph = g;
    return reinterpret_cast<CannGraphHandle>(impl);
}

/* ── Build Options ──────────────────────────────────────────────────── */

CannBuildOptionsHandle cann_build_options_create(void) {
    return new CannBuildOptsImpl();
}

void cann_build_options_destroy(CannBuildOptionsHandle options) {
    delete options;
}

CannStatus cann_build_options_set_mode(CannBuildOptionsHandle options, int32_t mode) {
    if (!options) return kInvalidPtr;
    options->options.mode = (mode == 0) ? hiai::AUTO : hiai::CUSTOM;
    return kSuccess;
}

CannStatus cann_build_options_set_weight_data_type(CannBuildOptionsHandle options,
                                                      int32_t weight_dtype) {
    if (!options) return kInvalidPtr;
    options->options.weightDataType = (weight_dtype == 1)
        ? hiai::FP16 : hiai::FP32;
    return kSuccess;
}

CannStatus cann_build_options_set_device_order(CannBuildOptionsHandle options,
                                                  const int32_t* devices,
                                                  int32_t device_count) {
    if (!options || !devices || device_count <= 0) return kInvalidPara;
    options->options.modelDeviceOrder.clear();
    for (int32_t i = 0; i < device_count; ++i) {
        options->options.modelDeviceOrder.push_back(
            static_cast<hiai::ExecuteDevice>(devices[i]));
    }
    return kSuccess;
}

CannStatus cann_build_options_set_input_shapes(CannBuildOptionsHandle options,
                                                  const int64_t* const* shapes,
                                                  const int32_t* shape_counts,
                                                  int32_t num_inputs) {
    if (!options || !shapes || !shape_counts || num_inputs <= 0)
        return kInvalidPara;
    options->options.inputShapes.clear();
    for (int32_t i = 0; i < num_inputs; ++i) {
        if (!shapes[i]) return kInvalidPtr;
        std::vector<int64_t> s(shapes[i], shapes[i] + shape_counts[i]);
        options->options.inputShapes.push_back(s);
    }
    return kSuccess;
}

CannStatus cann_build_options_set_precision_mode(CannBuildOptionsHandle options,
                                                    int32_t precision_mode) {
    if (!options) return kInvalidPtr;
    /* PrecisionMode is set via AiModelDescription, not BuildOptions. */
    (void)precision_mode;
    return kSuccess;
}

CannStatus cann_build_options_set_quantize_config(CannBuildOptionsHandle options,
                                                    const char* config) {
    if (!options || !config) return kInvalidPtr;
    options->options.quantizeConfig = std::string(config);
    return kSuccess;
}

CannStatus cann_build_options_set_tuning_strategy(CannBuildOptionsHandle options,
                                                    int32_t strategy) {
    if (!options) return kInvalidPtr;
    options->options.tuningStrategy = static_cast<hiai::TuningStrategy>(strategy);
    return kSuccess;
}

/* ── Model Building ──────────────────────────────────────────────────── */

CannHiaiIrBuildHandle cann_hiai_ir_build_create() {
    auto* impl = new CannHiaiIrBuildImpl();
    return reinterpret_cast<CannHiaiIrBuildHandle>(impl);
}

CannStatus cann_hiai_ir_build_destroy(CannHiaiIrBuildHandle build) {
    if (!build) return kInvalidPtr;
    delete reinterpret_cast<CannHiaiIrBuildImpl*>(build);
    return kSuccess;
}

CannStatus cann_model_create_buff(CannHiaiIrBuildHandle build,
                                   CannModelHandle model,
                                   CannModelBuffer* output_buffer,
                                   uint32_t custom_size) {
    if (!build || !model || !output_buffer) return kInvalidPtr;

    hiai::HiaiIrBuild* builder = &reinterpret_cast<CannHiaiIrBuildImpl*>(build)->build;
    hiai::ModelBufferData outputData;
    bool success = builder->CreateModelBuff(model->model, outputData, custom_size);

    if (!success) return kFailed;

    output_buffer->data = outputData.data;
    output_buffer->length = outputData.length;
    return kSuccess;
}

CannStatus cann_model_create_buff_default(CannHiaiIrBuildHandle build,
                                           CannModelHandle model,
                                           CannModelBuffer* output_buffer) {
    if (!build || !model || !output_buffer) return kInvalidPtr;

    hiai::HiaiIrBuild* builder = &reinterpret_cast<CannHiaiIrBuildImpl*>(build)->build;
    hiai::ModelBufferData outputData;
    bool success = builder->CreateModelBuff(model->model, outputData);

    if (!success) return kFailed;

    output_buffer->data = outputData.data;
    output_buffer->length = outputData.length;
    return kSuccess;
}

CannStatus cann_build_model(CannHiaiIrBuildHandle build,
                               CannModelHandle model,
                               CannBuildOptionsHandle options,
                               CannModelBuffer* output_buffer) {
    if (!build || !model || !output_buffer) return kInvalidPtr;

    hiai::HiaiIrBuild* builder = &reinterpret_cast<CannHiaiIrBuildImpl*>(build)->build;
    hiai::ModelBufferData outputData;
    outputData.data = output_buffer->data;
    outputData.length = output_buffer->length;
    bool success = false;

    if (options) {
        success = builder->BuildIRModel(model->model, outputData, options->options);
    } else {
        success = builder->BuildIRModel(model->model, outputData);
    }

    if (!success) return kFailed;
    output_buffer->length = outputData.length;
    output_buffer->data = outputData.data;

    return kSuccess;
}

void cann_model_buffer_destroy(CannHiaiIrBuildHandle build, CannModelBuffer* buffer) {
    if (!build || !buffer) return;
    hiai::HiaiIrBuild* builder = &reinterpret_cast<CannHiaiIrBuildImpl*>(build)->build;
    hiai::ModelBufferData data;
    data.data = buffer->data;
    data.length = buffer->length;
    builder->ReleaseModelBuff(data);
    buffer->data = nullptr;
    buffer->length = 0;
}
}  // extern "C"
}  // namespace ddk
