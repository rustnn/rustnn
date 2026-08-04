/**
 * CANN Adapter Layer - AiModelMngerClient Wrapper
 *
 * Wraps hiai::AiModelDescription and hiai::AiModelMngerClient
 * with a pure C interface.
 */

#ifndef CANN_MODEL_MANAGER_ADAPTER_H
#define CANN_MODEL_MANAGER_ADAPTER_H

#include "adapter_types.h"

namespace ddk {
extern "C" {

/* ── Model Description ────────────────────────────────────────────────── */

CANN_ADAPTER_EXPORT CannModelDescHandle cann_model_desc_create(const char* name,
                                             int32_t frequency,
                                             int32_t framework,
                                             int32_t model_type,
                                             int32_t device_type);

CANN_ADAPTER_EXPORT void          cann_model_desc_destroy(CannModelDescHandle desc);

CANN_ADAPTER_EXPORT const char*   cann_model_desc_get_name(CannModelDescHandle desc);

CANN_ADAPTER_EXPORT CannStatus    cann_model_desc_set_model_buffer(CannModelDescHandle desc,
                                                 const void* data,
                                                 uint32_t size);

CANN_ADAPTER_EXPORT CannStatus    cann_model_desc_set_input_dims(CannModelDescHandle desc,
                                               CannIOTensorDimensionHandle* dims,
                                               int32_t dim_count);

CANN_ADAPTER_EXPORT CannStatus    cann_model_desc_set_dynamic_shape(CannModelDescHandle desc,
                                                   int32_t enable,
                                                   uint32_t max_cached_num);

CANN_ADAPTER_EXPORT CannStatus    cann_model_desc_get_dynamic_shape_config(CannModelDescHandle desc,
                                                         int32_t* out_enable,
                                                         uint32_t* out_max_cached_num);

CANN_ADAPTER_EXPORT CannStatus    cann_model_desc_get_input_dims(CannModelDescHandle desc,
                                               CannIOTensorDimensionHandle* dims,
                                               int32_t max_dims,
                                               int32_t* out_dim_count);

CANN_ADAPTER_EXPORT CannStatus    cann_model_desc_set_precision_mode(CannModelDescHandle desc,
                                                   int32_t precision_mode);

CANN_ADAPTER_EXPORT CannStatus    cann_model_desc_get_precision_mode(CannModelDescHandle desc,
                                                   int32_t* out_precision_mode);

CANN_ADAPTER_EXPORT CannStatus    cann_model_desc_set_tuning_strategy(CannModelDescHandle desc,
                                                    int32_t strategy);

CANN_ADAPTER_EXPORT CannStatus    cann_model_desc_get_tuning_strategy(CannModelDescHandle desc,
                                                    int32_t* out_strategy);

/* ── Model Manager Client ─────────────────────────────────────────────── */

CANN_ADAPTER_EXPORT CannModelManagerHandle cann_model_manager_create(void);
CANN_ADAPTER_EXPORT void                   cann_model_manager_destroy(CannModelManagerHandle manager);

CANN_ADAPTER_EXPORT CannStatus cann_model_manager_init(CannModelManagerHandle manager);

CANN_ADAPTER_EXPORT CannStatus cann_model_manager_load(CannModelManagerHandle manager,
                                     CannModelDescHandle* descs,
                                     int32_t desc_count);

CANN_ADAPTER_EXPORT CannStatus cann_model_manager_process(
    CannModelManagerHandle manager,
    CannContextHandle context,
    CannIOTensorHandle* inputs,
    int32_t input_count,
    CannIOTensorHandle* outputs,
    int32_t output_count,
    uint32_t timeout,
    int32_t* out_stamp);

CANN_ADAPTER_EXPORT CannStatus cann_model_manager_unload(CannModelManagerHandle manager);

CANN_ADAPTER_EXPORT const char* cann_model_manager_get_version(CannModelManagerHandle manager);

CANN_ADAPTER_EXPORT CannStatus cann_model_manager_check_compatibility(
    CannModelManagerHandle manager,
    CannModelDescHandle desc,
    int32_t* out_compatible);

CANN_ADAPTER_EXPORT CannStatus cann_model_manager_get_model_io_tensor_dim(
    CannModelManagerHandle manager,
    const char* model_name,
    CannIOTensorDimensionHandle* input_dims,
    int32_t max_input_dims,
    int32_t* out_input_dim_count,
    CannIOTensorDimensionHandle* output_dims,
    int32_t max_output_dims,
    int32_t* out_output_dim_count);

CANN_ADAPTER_EXPORT CannStatus cann_model_manager_set_priority(
    CannModelManagerHandle manager,
    const char* model_name,
    int32_t priority);

}  // extern "C"
}  // namespace ddk

#endif /* CANN_MODEL_MANAGER_ADAPTER_H */
