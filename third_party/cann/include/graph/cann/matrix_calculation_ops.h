/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

/*!
 * \file matrix_calculation_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_MATRIX_CALCULATION_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_MATRIX_CALCULATION_OPS_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {

/**
* @brief Multiplies matrix "a" by matrix "b", producing "a * b".
* @par Inputs:
* Four inputs, including:
* @li x1: A matrix Tensor. 2D. Must be one of the following types: float32,
* float16, int32, int8, int4, bfloat16, hifloat8. Has format [ND, NHWC, NCHW].
* @li x2: A matrix Tensor. 2D. Must be one of the following types: float32,
* float16, int32, int8, int4, bfloat16, hifloat8. Has format [ND, NHWC, NCHW].
* @li bias: A 1D Tensor. Must be one of the following types: float32,
* float16, int32, bfloat16. Has format [ND, NHWC, NCHW].
* @li offset_w: A Optional 1D Tensor for quantized inference. Type is int8, int4, bfloat16.
* Reserved.

* @par Attributes:
* @li transpose_x1: A bool. If True, changes the shape of "x1" from [K, M] to
* [M, K] before multiplication.
* @li transpose_x2: A bool. If True, changes the shape of "x2" from [N, K] to
* [K, N] before multiplication.
* @li offset_x: An optional integer for quantized MatMulV2.
* The negative offset added to the input x1 for int8 type. Ensure offset_x
* within the effective range of int8 [-128, 127]. Defaults to "0".

* @par Outputs:
* y: The result matrix Tensor. 2D. Must be one of the following types: float32,
* float16, int32, bfloat16, hifloat8. Has format [ND, NHWC, NCHW].

* @attention Constraints:
* if performances better in format NZ, please close
* "MatmulTransdataFusionPass" in fusion configuration.

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator MatMul.
*/
REG_OP(MatMulV2)
    .INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT8, DT_INT4, DT_BF16, DT_HIFLOAT8}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT8, DT_INT4, DT_BF16, DT_HIFLOAT8}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16, DT_HIFLOAT8}))
    .OPTIONAL_INPUT(offset_w, TensorType({DT_INT8, DT_INT4}))
    .ATTR(transpose_x1, Bool, false)
    .ATTR(transpose_x2, Bool, false)
    .ATTR(offset_x, Int, 0)
    .OP_END_FACTORY_REG(MatMulV2)


/**
* @brief Multiplies matrix "a" by matrix "b", producing "a * b" .
* @par Inputs:
* Four inputs, including:
* @li x1: A matrix Tensor. Must be one of the following types: float16,
* float32, int32, int8, int4, bfloat16, hifloat8. 2D-6D. Has format [ND, NHWC, NCHW].
* @li x2: A matrix Tensor. Must be one of the following types: float16,
* float32, int32, int8, int4, bfloat16, hifloat8. 2D-6D. Has format [ND, NHWC, NCHW].
* @li bias: A optional Tensor. Must be one of the following types:
* float16, float32, int32, bfloat16. Has format [ND, NHWC, NCHW].
* @li offset_w: A optional Tensor. Must be one of the following types:
* int8, int4. Has format [ND, NHWC, NCHW].

* @par Attributes:
* @li adj_x1: A bool. If True, changes the shape of "x1" from [B, M, K] to
* [B, K, M] before multiplication.
* @li adj_x2: A bool. If True, changes the shape of "x2" from [B, K, N] to
* [B, N, K] before multiplication.
* @li offset_x: An optional integer for quantized BatchMatMulV2.

* @par Outputs:
* y: The result matrix Tensor. Must be one of the following types: float16,
* float32, int32, bfloat16, hifloat8. 2D-6D. Has format [ND, NHWC]. Has the same shape
* length as "x1" and "x2".

* @attention Constraints:
* if performances better in format NZ, please close
* "MatmulTransdataFusionPass" in fusion configuration.

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator BatchMatmul.
*/

REG_OP(BatchMatMulV2)
    .INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT8, DT_INT4, DT_BF16, DT_HIFLOAT8}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT8, DT_INT4, DT_BF16, DT_HIFLOAT8}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .OPTIONAL_INPUT(offset_w, TensorType({DT_INT8, DT_INT4}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16, DT_HIFLOAT8}))
    .ATTR(adj_x1, Bool, false)
    .ATTR(adj_x2, Bool, false)
    .ATTR(offset_x, Int, 0)
    .OP_END_FACTORY_REG(BatchMatMulV2)

/**
* @brief Applies sparse "updates" to individual values or slices in a variable reference.

* @par Inputs:
* @li var: The rewritten tensor. An ND tensor. Support 1D ~ 8D. Must be one of the following types:
* complex128, complex64, double, float32, float16, int16, int32, int64, int8, qint16, qint32, qint8, quint16, quint8,
* uint16, uint32, uint64, uint8, bfloat16, bool.
* @li indices: The index tensor. An ND tensor. Support 1D ~ 8D. Must be one of the following types: int32, int64. The
* last dimension of "indices" represents that the first few dimensions of "var" are the batch dimensions.
* @li updates: The source tensor. An ND tensor. Support 1D ~ 8D. Shape should be equal to the shape of "indices" except
* for the last dimension concats the shape of "var" except for the batch dimensions. Must have the same type of "var".

* @par Attributes:
* use_locking: An optional bool. Defaults to "False". If "True", the operation will be protected by a lock.

* @par Outputs:
* var: An ND tensor. Support 1D ~ 8D. Must have the same type, shape and format as input "var".

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator ScatterNdUpdate.
*/
REG_OP(ScatterNdUpdate)
    .INPUT(var, TensorType({BasicType(), DT_BOOL}))
    .INPUT(indices, TensorType::IndexNumberType())
    .INPUT(updates, TensorType({BasicType(), DT_BOOL}))
    .OUTPUT(var,  TensorType({BasicType(), DT_BOOL}))
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(ScatterNdUpdate)

/**
* @brief Also known as a "fully-connected" layer, computes an inner product
* with a set of learned weights, and (optionally) adds biases.
* @par Inputs:
* Four inputs, including:
* @li x: A Tensor of type float16, int8, int4, bf16.
* @li w: A weight matrix of type float16, int8, int4, float32, bf16.
* @li b: An optional Tensor of type float16, int32, float32, bf16.
* @li offset_w: An optional Tensor of type int8, int4.
* Reserved. Only None Supported. \n

* @par Attributes:
* @li num_output: Required. An int, output neuron number. Reserved.
* @li transpose: A bool, specifying weight whether to transpose input w,
* either "true" or "false". Defaults to "false".
* @li axis: Optional. An int, 1 or 2, specifying which dimension the input
* "K" starts from. Defaults to 1.
* The product of the subsequent dimensions starting form first dimension
* or the second dimension is "K".
* @li offset_x: An optional integer for quantized FullyConnection.
* The negative offset added to the input image for int8 type. Ensure offset_x
* within the effective range of int8 [-128, 127]. Defaults to "0". \n

* @par Outputs:
* y: The result tensor of type float16, int32, float32, bf16. \n

* @par Third-party framework compatibility
* Compatible with the Caffe operator InnerProduct. \n

* @par Quantization supported or not
* Yes
*/
REG_OP(FullyConnection)
    .INPUT(x, TensorType({DT_FLOAT16, DT_INT8, DT_INT4, DT_FLOAT, DT_BF16}))
    .INPUT(w, TensorType({DT_FLOAT16, DT_INT8, DT_INT4, DT_FLOAT, DT_BF16}))
    .OPTIONAL_INPUT(b, TensorType({DT_FLOAT16, DT_INT32, DT_FLOAT, DT_BF16}))
    .OPTIONAL_INPUT(offset_w, TensorType({DT_INT8, DT_INT4}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_INT32, DT_FLOAT, DT_BF16}))
    .REQUIRED_ATTR(num_output, Int)
    .ATTR(transpose, Bool, false)
    .ATTR(axis, Int, 1)
    .ATTR(offset_x, Int, 0)
    .OP_END_FACTORY_REG(FullyConnection)
}  // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_INC_MATRIX_CALCULATION_OPS_H_
