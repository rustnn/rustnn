/**
 * CANN Adapter Layer - Model Build Wrapper
 *
 * Wraps ge::Model and hiai::HiaiIrBuild with a pure C interface.
 */

#ifndef CANN_MODEL_ADAPTER_H
#define CANN_MODEL_ADAPTER_H

#include "adapter_types.h"

namespace ddk {
extern "C" {

/* ── Model lifecycle ──────────────────────────────────────────────────── */

CANN_ADAPTER_EXPORT CannModelHandle cann_model_create();
CANN_ADAPTER_EXPORT CannModelHandle cann_model_create_with_name(const char* name);
CANN_ADAPTER_EXPORT void            cann_model_destroy(CannModelHandle model);

/* ── Model / Graph ────────────────────────────────────────────────────── */

CANN_ADAPTER_EXPORT CannStatus        cann_model_set_graph(CannModelHandle model, CannGraphHandle graph);
CANN_ADAPTER_EXPORT CannGraphHandle   cann_model_get_graph(CannModelHandle model);

/* ── Build Options ────────────────────────────────────────────────────── */

CANN_ADAPTER_EXPORT CannBuildOptionsHandle cann_build_options_create(void);
CANN_ADAPTER_EXPORT void                   cann_build_options_destroy(CannBuildOptionsHandle options);

CANN_ADAPTER_EXPORT CannStatus cann_build_options_set_mode(CannBuildOptionsHandle options, int32_t mode);

CANN_ADAPTER_EXPORT CannStatus cann_build_options_set_weight_data_type(CannBuildOptionsHandle options,
                                                      int32_t weight_dtype);

CANN_ADAPTER_EXPORT CannStatus cann_build_options_set_device_order(CannBuildOptionsHandle options,
                                                  const int32_t* devices,
                                                  int32_t device_count);

CANN_ADAPTER_EXPORT CannStatus cann_build_options_set_input_shapes(CannBuildOptionsHandle options,
                                                  const int64_t* const* shapes,
                                                  const int32_t* shape_counts,
                                                  int32_t num_inputs);

CANN_ADAPTER_EXPORT CannStatus cann_build_options_set_precision_mode(CannBuildOptionsHandle options,
                                                    int32_t precision_mode);

CANN_ADAPTER_EXPORT CannStatus cann_build_options_set_quantize_config(CannBuildOptionsHandle options,
                                                    const char* config);

CANN_ADAPTER_EXPORT CannStatus cann_build_options_set_tuning_strategy(CannBuildOptionsHandle options,
                                                    int32_t strategy);

/* ── Model Building ──────────────────────────────────────────────────── */

CANN_ADAPTER_EXPORT CannHiaiIrBuildHandle cann_hiai_ir_build_create();
CANN_ADAPTER_EXPORT CannStatus cann_hiai_ir_build_destroy(CannHiaiIrBuildHandle build);

CANN_ADAPTER_EXPORT CannStatus cann_model_create_buff(CannHiaiIrBuildHandle build, CannModelHandle model,
                                   CannModelBuffer* output_buffer,
                                   uint32_t custom_size);

CANN_ADAPTER_EXPORT CannStatus cann_model_create_buff_default(CannHiaiIrBuildHandle build, CannModelHandle model,
                                           CannModelBuffer* output_buffer);

CANN_ADAPTER_EXPORT CannStatus cann_build_model(CannHiaiIrBuildHandle build, CannModelHandle model,
                               CannBuildOptionsHandle options,
                               CannModelBuffer* output_buffer);

CANN_ADAPTER_EXPORT void cann_model_buffer_destroy(CannHiaiIrBuildHandle build, CannModelBuffer* buffer);

}  // extern "C"
}  // namespace ddk

#endif /* CANN_MODEL_ADAPTER_H */
