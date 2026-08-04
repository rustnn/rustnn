/**
 * CANN Adapter Layer - Tensor & TensorDesc Wrappers
 *
 * hiai::AiTensor (high-level inference tensor) with a pure C interface.
 */

#ifndef CANN_IO_TENSOR_ADAPTER_H
#define CANN_IO_TENSOR_ADAPTER_H

#include <stdint.h>

#include "adapter_types.h"

namespace ddk {
extern "C" {


/* ── Tensor lifecycle (wraps hiai::AiTensor) ─────────────────────────── */

CANN_ADAPTER_EXPORT CannIOTensorHandle cann_io_tensor_create();

CANN_ADAPTER_EXPORT CannIOTensorHandle cann_io_tensor_create_with_data(const void* data,
                                               uint32_t size);

CANN_ADAPTER_EXPORT void     cann_io_tensor_destroy(CannIOTensorHandle tensor);

/* ── Tensor data access ──────────────────────────────────────────────── */

CANN_ADAPTER_EXPORT CannStatus cann_io_tensor_set_data(CannIOTensorHandle tensor,
                                 const void* data,
                                 uint32_t size);

CANN_ADAPTER_EXPORT void*      cann_io_tensor_get_buffer(CannIOTensorHandle tensor);
CANN_ADAPTER_EXPORT uint32_t   cann_io_tensor_get_size(CannIOTensorHandle tensor);

/* ── Tensor Init (wraps hiai::AiTensor::Init) ─────────────────────────── */

CANN_ADAPTER_EXPORT CannStatus cann_io_tensor_init(CannIOTensorHandle tensor,
                              CannIOTensorDimensionHandle dim,
                              CannDataType dtype);

CANN_ADAPTER_EXPORT CannStatus cann_io_tensor_init_with_data(CannIOTensorHandle tensor,
                                        const void* data,
                                        CannIOTensorDimensionHandle dim,
                                        CannDataType dtype);

/* ── Tensor Dimension access ──────────────────────────────────────────── */

CANN_ADAPTER_EXPORT CannStatus cann_io_tensor_set_tensor_dimension(CannIOTensorHandle tensor,
                                              CannIOTensorDimensionHandle dim);

CANN_ADAPTER_EXPORT CannIOTensorDimensionHandle cann_io_tensor_get_tensor_dimension(CannIOTensorHandle tensor);

/* ── TensorDimension lifecycle (wraps hiai::TensorDimension) ──────────── */

CANN_ADAPTER_EXPORT CannIOTensorDimensionHandle cann_io_tensor_dim_create(uint32_t n, uint32_t c,
                                                   uint32_t h, uint32_t w);

CANN_ADAPTER_EXPORT CannIOTensorDimensionHandle cann_io_tensor_dim_create_nd(const uint32_t* dims,
                                                      int32_t dim_count);

CANN_ADAPTER_EXPORT void     cann_io_tensor_dim_destroy(CannIOTensorDimensionHandle dim);

CANN_ADAPTER_EXPORT void     cann_io_tensor_dim_set_number(CannIOTensorDimensionHandle dim, uint32_t n);
CANN_ADAPTER_EXPORT uint32_t cann_io_tensor_dim_get_number(CannIOTensorDimensionHandle dim);

CANN_ADAPTER_EXPORT void     cann_io_tensor_dim_set_channel(CannIOTensorDimensionHandle dim, uint32_t c);
CANN_ADAPTER_EXPORT uint32_t cann_io_tensor_dim_get_channel(CannIOTensorDimensionHandle dim);

CANN_ADAPTER_EXPORT void     cann_io_tensor_dim_set_height(CannIOTensorDimensionHandle dim, uint32_t h);
CANN_ADAPTER_EXPORT uint32_t cann_io_tensor_dim_get_height(CannIOTensorDimensionHandle dim);

CANN_ADAPTER_EXPORT void     cann_io_tensor_dim_set_width(CannIOTensorDimensionHandle dim, uint32_t w);
CANN_ADAPTER_EXPORT uint32_t cann_io_tensor_dim_get_width(CannIOTensorDimensionHandle dim);

}  // extern "C"
}  // namespace ddk

#endif /* CANN_IO_TENSOR_ADAPTER_H */
