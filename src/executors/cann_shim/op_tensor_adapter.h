/**
 * CANN Adapter Layer - Tensor & TensorDesc Wrappers
 *
 * Wraps ge::TensorDesc (low-level tensor description) and
 * ge::Tensor (high-level inference tensor) with a pure C interface.
 */

#ifndef CANN_OP_TENSOR_ADAPTER_H
#define CANN_OP_TENSOR_ADAPTER_H

#include <stdint.h>

#include "adapter_types.h"

namespace ddk {
extern "C" {


/* ── TensorDesc lifecycle (wraps ge::TensorDesc) ─────────────────────── */

CANN_ADAPTER_EXPORT CannOpTensorDescHandle cann_tensor_desc_create(CannShapeHandle shape,
                                              CannFormat format,
                                              CannDataType dtype);

CANN_ADAPTER_EXPORT void     cann_tensor_desc_destroy(CannOpTensorDescHandle desc);

/* ── TensorDesc property getters/setters ─────────────────────────────── */

CANN_ADAPTER_EXPORT CannStatus cann_tensor_desc_set_shape(CannOpTensorDescHandle desc,
                                       const int64_t* shape,
                                       int32_t shape_count);

CANN_ADAPTER_EXPORT CannStatus cann_tensor_desc_set_format(CannOpTensorDescHandle desc, CannFormat format);

CANN_ADAPTER_EXPORT CannStatus cann_tensor_desc_set_data_type(CannOpTensorDescHandle desc, CannDataType dtype);

CANN_ADAPTER_EXPORT const int64_t* cann_tensor_desc_get_shape(CannOpTensorDescHandle desc,
                                           int32_t* out_shape_count);

CANN_ADAPTER_EXPORT CannFormat    cann_tensor_desc_get_format(CannOpTensorDescHandle desc);
CANN_ADAPTER_EXPORT CannDataType  cann_tensor_desc_get_data_type(CannOpTensorDescHandle desc);

/* ── TensorDesc advanced accessors (Shape-based, origin props) ────────── */

CANN_ADAPTER_EXPORT CannShapeHandle cann_tensor_desc_get_shape_handle(CannOpTensorDescHandle desc);
CANN_ADAPTER_EXPORT CannStatus      cann_tensor_desc_set_shape_from_handle(CannOpTensorDescHandle desc,
                                                       CannShapeHandle shape);

CANN_ADAPTER_EXPORT CannShapeHandle cann_tensor_desc_get_origin_shape(CannOpTensorDescHandle desc);
CANN_ADAPTER_EXPORT CannStatus      cann_tensor_desc_set_origin_shape(CannOpTensorDescHandle desc,
                                                   CannShapeHandle shape);

CANN_ADAPTER_EXPORT CannFormat      cann_tensor_desc_get_origin_format(CannOpTensorDescHandle desc);
CANN_ADAPTER_EXPORT CannStatus      cann_tensor_desc_set_origin_format(CannOpTensorDescHandle desc,
                                                    CannFormat format);

CANN_ADAPTER_EXPORT CannDataType    cann_tensor_desc_get_origin_data_type(CannOpTensorDescHandle desc);
CANN_ADAPTER_EXPORT CannStatus      cann_tensor_desc_set_origin_data_type(CannOpTensorDescHandle desc,
                                                       CannDataType dtype);

/* ── Shape lifecycle (wraps ge::Shape) ───────────────────────────────── */

CANN_ADAPTER_EXPORT CannShapeHandle cann_shape_create(const int64_t* dims, int32_t dim_count);
CANN_ADAPTER_EXPORT CannShapeHandle cann_shape_create_default(void);
CANN_ADAPTER_EXPORT void cann_shape_destroy(CannShapeHandle shape);

CANN_ADAPTER_EXPORT int32_t  cann_shape_get_dim_num(CannShapeHandle shape);
CANN_ADAPTER_EXPORT int64_t  cann_shape_get_dim(CannShapeHandle shape, int32_t idx);
CANN_ADAPTER_EXPORT CannStatus cann_shape_set_dim(CannShapeHandle shape, int32_t idx, int64_t value);
CANN_ADAPTER_EXPORT const int64_t* cann_shape_get_dims(CannShapeHandle shape, int32_t* out_count);
CANN_ADAPTER_EXPORT int64_t  cann_shape_get_total_dim_num(CannShapeHandle shape);
CANN_ADAPTER_EXPORT uint32_t cann_shape_get_shape_size(CannShapeHandle shape);

/* ── Tensor lifecycle (wraps ge::Tensor) ───────────────────────────────── */

CANN_ADAPTER_EXPORT CannOpTensorHandle cann_op_tensor_create(CannOpTensorDescHandle desc);

CANN_ADAPTER_EXPORT void cann_op_tensor_destroy(CannOpTensorHandle tensor);

/* ── Tensor data access ──────────────────────────────────────────────── */

CANN_ADAPTER_EXPORT CannOpTensorDescHandle cann_op_tensor_get_desc(CannOpTensorHandle tensor);

CANN_ADAPTER_EXPORT CannStatus cann_op_tensor_set_data(CannOpTensorHandle tensor,
                                 const void* data,
                                 uint32_t size);


}  // extern "C"
}  // namespace ddk

#endif /* CANN_OP_TENSOR_ADAPTER_H */