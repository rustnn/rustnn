/**
 * CANN Adapter Layer - Operator Wrapper & Factory Functions
 *
 * Wraps ge::Operator with a pure C interface and provides
 * factory functions for creating specific operator types.
 *
 * All factory functions accept a name only; inputs and attributes
 * are set via the generic setter functions after creation.
 */

#ifndef CANN_OPERATOR_ADAPTER_H
#define CANN_OPERATOR_ADAPTER_H

#include "adapter_types.h"

namespace ddk {
extern "C" {

/* ── Base Operator Lifecycle ─────────────────────────────────────────── */

CANN_ADAPTER_EXPORT CannOperatorHandle cann_operator_create(const char* type, const char* name);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_operator_create_registered(const char* type, const char* name);
CANN_ADAPTER_EXPORT void               cann_operator_destroy(CannOperatorHandle op);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_operator_clone(CannOperatorHandle op);

/* ── Input/Output Connections ────────────────────────────────────────── */

CANN_ADAPTER_EXPORT CannStatus cann_operator_set_input(CannOperatorHandle op,
                                    const char* name,
                                    CannOperatorHandle input_op);

CANN_ADAPTER_EXPORT CannStatus cann_operator_set_input_by_index(CannOperatorHandle op,
                                              int32_t index,
                                              CannOperatorHandle input_op,
                                              int32_t input_index);

/* ── Attributes ──────────────────────────────────────────────────────── */

CANN_ADAPTER_EXPORT CannStatus cann_operator_set_attr_int64(CannOperatorHandle op,
                                          const char* name,
                                          int64_t value);

CANN_ADAPTER_EXPORT CannStatus cann_operator_set_attr_float(CannOperatorHandle op,
                                          const char* name,
                                          float value);

CANN_ADAPTER_EXPORT CannStatus cann_operator_set_attr_string(CannOperatorHandle op,
                                           const char* name,
                                           const char* value);

CANN_ADAPTER_EXPORT CannStatus cann_operator_set_attr_int64_list(CannOperatorHandle op,
                                               const char* name,
                                               const int64_t* values,
                                               int32_t count);

CANN_ADAPTER_EXPORT CannStatus cann_operator_set_attr_float_list(CannOperatorHandle op,
                                               const char* name,
                                               const float* values,
                                               int32_t count);

CANN_ADAPTER_EXPORT CannStatus cann_operator_set_attr_tensor(CannOperatorHandle op,
                                          const char* name,
                                          CannOpTensorHandle tensor);

CANN_ADAPTER_EXPORT CannStatus cann_operator_set_attr_tensor_raw(CannOperatorHandle op,
                                          const char* name,
                                          const void* data,
                                          uint32_t size,
                                          const int64_t* shape,
                                          int32_t shape_count,
                                          CannDataType dtype);

CANN_ADAPTER_EXPORT CannStatus cann_operator_set_attr_tensor_raw_format(CannOperatorHandle op,
                                          const char* name,
                                          const void* data,
                                          uint32_t size,
                                          const int64_t* shape,
                                          int32_t shape_count,
                                          CannDataType dtype,
                                          int32_t format);

CANN_ADAPTER_EXPORT CannStatus cann_operator_set_attr_bool(CannOperatorHandle op,
                                        const char* name,
                                        int32_t value);

/* ── Tensor Descriptor ───────────────────────────────────────────────── */

CANN_ADAPTER_EXPORT CannOpTensorDescHandle cann_operator_get_input_desc(CannOperatorHandle op, uint32_t index);

CANN_ADAPTER_EXPORT CannOpTensorDescHandle cann_operator_get_output_desc(CannOperatorHandle op, uint32_t index);

CANN_ADAPTER_EXPORT CannStatus cann_operator_update_input_desc(CannOperatorHandle op,
                                              const char* name,
                                              CannOpTensorDescHandle desc);

/* ── Operator Info ───────────────────────────────────────────────────── */

CANN_ADAPTER_EXPORT const char* cann_operator_get_name(CannOperatorHandle op);
CANN_ADAPTER_EXPORT const char* cann_operator_get_type(CannOperatorHandle op);

/* ── Dynamic Input / Output ──────────────────────────────────────────── */

CANN_ADAPTER_EXPORT CannStatus cann_operator_create_dynamic_input(CannOperatorHandle op,
                                               const char* name,
                                               uint32_t num);

CANN_ADAPTER_EXPORT CannStatus cann_operator_create_dynamic_output(CannOperatorHandle op,
                                                const char* name,
                                                uint32_t num);

CANN_ADAPTER_EXPORT CannStatus cann_operator_set_dynamic_input_by_index(CannOperatorHandle op,
                                                const char* name,
                                                uint32_t index,
                                                CannOperatorHandle input_op);

CANN_ADAPTER_EXPORT CannStatus cann_operator_set_dynamic_input_by_index_by_output(CannOperatorHandle op,
                                                const char* name,
                                                uint32_t index,
                                                CannOperatorHandle input_op,
                                                uint32_t output_index);

CANN_ADAPTER_EXPORT CannStatus cann_operator_set_input_by_output(CannOperatorHandle op,
                                              const char* name,
                                              CannOperatorHandle input_op,
                                              uint32_t output_index);

/* ═════════════════════════════════════════════════════════════════════════
 * Operator Factory Functions
 *
 * Each creates a specific operator type by name.  Inputs, attributes,
 * and dynamic-input slots are set via the generic setters above.
 * ═══════════════════════════════════════════════════════════════════════ */

/* ── Data / Constants ────────────────────────────────────────────────── */

CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_data(void);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_data_with_name(const char* name);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_const(void);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_const_with_name(const char* name);

/* ── Element-wise Arithmetic ─────────────────────────────────────────── */

CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_add(void);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_add_with_name(const char* name);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_sub(void);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_sub_with_name(const char* name);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_mul(void);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_mul_with_name(const char* name);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_real_div(void);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_real_div_with_name(const char* name);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_maximum(void);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_maximum_with_name(const char* name);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_minimum(void);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_minimum_with_name(const char* name);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_pow(void);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_pow_with_name(const char* name);

/* ── Comparisons ──────────────────────────────────────────────────────── */

CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_greater(void);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_greater_with_name(const char* name);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_greater_equal(void);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_greater_equal_with_name(const char* name);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_less(void);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_less_with_name(const char* name);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_less_equal(void);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_less_equal_with_name(const char* name);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_equal(void);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_equal_with_name(const char* name);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_not_equal(void);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_not_equal_with_name(const char* name);

/* ── Logical ──────────────────────────────────────────────────────────── */

CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_logical_or(void);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_logical_or_with_name(const char* name);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_logical_xor(void);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_logical_xor_with_name(const char* name);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_logical_not(void);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_logical_not_with_name(const char* name);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_logical_and(void);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_logical_and_with_name(const char* name);

/* ── Activations ─────────────────────────────────────────────────────── */

CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_activation(void);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_activation_with_name(const char* name);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_hard_swish(void);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_hard_swish_with_name(const char* name);

/* ── Unary Math ──────────────────────────────────────────────────────── */

CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_neg(void);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_neg_with_name(const char* name);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_ceil(void);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_ceil_with_name(const char* name);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_cos(void);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_cos_with_name(const char* name);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_exp(void);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_exp_with_name(const char* name);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_floor(void);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_floor_with_name(const char* name);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_log(void);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_log_with_name(const char* name);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_sign(void);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_sign_with_name(const char* name);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_round(void);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_round_with_name(const char* name);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_sin(void);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_sin_with_name(const char* name);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_tan(void);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_tan_with_name(const char* name);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_sqrt(void);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_sqrt_with_name(const char* name);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_erf(void);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_erf_with_name(const char* name);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_reciprocal(void);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_reciprocal_with_name(const char* name);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_square(void);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_square_with_name(const char* name);

/* ── Neural Network ──────────────────────────────────────────────────── */

CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_conv2d(void);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_conv2d_with_name(const char* name);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_matmul(void);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_matmul_with_name(const char* name);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_pool2d(void);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_pool2d_with_name(const char* name);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_softmax(void);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_softmax_with_name(const char* name);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_batch_norm(void);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_batch_norm_with_name(const char* name);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_bn_inference(void);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_bn_inference_with_name(const char* name);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_convolution(void);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_convolution_with_name(const char* name);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_pool2d_d(void);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_pool2d_d_with_name(const char* name);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_gemm_d(void);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_gemm_d_with_name(const char* name);

/* ── Shape / Transform ───────────────────────────────────────────────── */

CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_reshape(void);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_reshape_with_name(const char* name);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_transpose(void);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_transpose_with_name(const char* name);

CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_concat(void);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_concat_with_name(const char* name);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_split(void);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_split_with_name(const char* name);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_slice(void);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_slice_with_name(const char* name);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_squeeze(void);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_squeeze_with_name(const char* name);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_expand_dims(void);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_expand_dims_with_name(const char* name);

/* ── Array ───────────────────────────────────────────────────────────── */

CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_broadcast_to(void);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_broadcast_to_with_name(const char* name);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_shape(void);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_shape_with_name(const char* name);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_gather_nd(void);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_gather_nd_with_name(const char* name);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_gather_v2d(void);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_gather_v2d_with_name(const char* name);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_pad(void);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_pad_with_name(const char* name);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_tile(void);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_tile_with_name(const char* name);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_select(void);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_select_with_name(const char* name);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_strided_slice_v2(void);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_strided_slice_v2_with_name(const char* name);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_scatter_nd_update(void);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_scatter_nd_update_with_name(const char* name);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_clip_by_value(void);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_clip_by_value_with_name(const char* name);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_arg_max_ext2(void);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_arg_max_ext2_with_name(const char* name);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_cast_t(void);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_cast_t_with_name(const char* name);

/* ── Image ───────────────────────────────────────────────────────────── */

CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_resize_nearest_neighbor(void);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_resize_nearest_neighbor_with_name(const char* name);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_resize_bilinear(void);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_resize_bilinear_with_name(const char* name);

/* ── Quantization ────────────────────────────────────────────────────── */

CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_quantize_v2(void);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_quantize_v2_with_name(const char* name);

/* ── Reductions ──────────────────────────────────────────────────────── */

CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_reduce_sum(void);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_reduce_sum_with_name(const char* name);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_reduce_mean(void);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_reduce_mean_with_name(const char* name);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_reduce_max(void);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_reduce_max_with_name(const char* name);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_reduce_min(void);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_reduce_min_with_name(const char* name);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_reduce_prod_d(void);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_reduce_prod_d_with_name(const char* name);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_reduce_l2d(void);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_reduce_l2d_with_name(const char* name);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_reduce_log_sum_exp(void);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_reduce_log_sum_exp_with_name(const char* name);

/* ── ge::op ──────────────────────────────────────────────────────────── */

CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_cumsum(void);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_cumsum_with_name(const char* name);

/* ── Utility ─────────────────────────────────────────────────────────── */

CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_net_output(void);
CANN_ADAPTER_EXPORT CannOperatorHandle cann_op_net_output_with_name(const char* name, int32_t input_count);

}  // extern "C"
}  // namespace ddk

#endif /* CANN_OPERATOR_ADAPTER_H */
