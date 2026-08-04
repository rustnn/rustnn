/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
 * \file transformation_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_TRANSFORMATION_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_TRANSFORMATION_OPS_H_

#include "graph/operator_reg.h"

namespace ge {

/**
* @brief Permutes the dimensions according to perm.
         The returned tensor's dimension i will correspond to the input dimension perm[i].

* @par Inputs:
* Two inputs, including:
* @li x: A Tensor. Must be one of the following types:
* bfloat16, float16, float32, double, int64, int32, uint8, uint16, uint32, uint64, int8,
* int16, complex32, complex64, complex128, qint8, quint8, qint16, quint16, qint32, bool, hifloat8, float8_e5m2,
* float8_e4m3fn, and the maximum dimension should not exceed 8 dimensions,
* and the shape should be consistent with output.
* @li perm: A Tensor of type int32 or int64. A permutation of the dimensions of "x", the value
* should be within the range of [0, number of dimensions for self -1].

* @par Outputs:
* y: A Tensor. Has the same type as "x".

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator Transpose.
*/
REG_OP(Transpose)
    .INPUT(x, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT64, DT_INT32,
                          DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64, DT_INT8, DT_INT16,
                          DT_COMPLEX32, DT_COMPLEX64, DT_COMPLEX128, DT_QINT8, DT_QUINT8,
                          DT_QINT16, DT_QUINT16, DT_QINT32, DT_BOOL, DT_HIFLOAT8, DT_FLOAT8_E5M2,
                          DT_FLOAT8_E4M3FN}))
    .INPUT(perm, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT64, DT_INT32,
                          DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64, DT_INT8, DT_INT16,
                          DT_COMPLEX32, DT_COMPLEX64, DT_COMPLEX128, DT_QINT8, DT_QUINT8,
                          DT_QINT16, DT_QUINT16, DT_QINT32, DT_BOOL, DT_HIFLOAT8, DT_FLOAT8_E5M2,
                          DT_FLOAT8_E4M3FN}))
    .OP_END_FACTORY_REG(Transpose)

/**
* @brief Permutes the dimensions according to order.
        The returned tensor's dimension i will correspond to the input dimension order[i] . \n

* @par Inputs:
* x: A ND tensor. Support 4D. Must be one of the following types: float16, float32 . \n

* @par Attributes:
* order: A permutation of the dimensions of "x".Type must be int32.Support any axis transformation.Defaults to "{0}"

* @par Outputs:
* y: A ND tensor. Support 4D. Has the same type as "x".

* @attention Constraints:
* The Attributes order must ensure that the provided dimensions are unique,do not repeat, and cover all dimensions of "x". \n
*/
REG_OP(Permute)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(order, ListInt, {0})
    .OP_END_FACTORY_REG(Permute)

/**
* @brief Do format transfer for various data format.
* In general, the framework will insert it atomatically. \n

* @par Inputs:
* src: A Tensor. Support 2D ~ 6D. For all branches can be types: bfloat16, float16, float32, int32, int8, bool.
*      For branches without padding also can be types: int16, int64, uint8, uint16, uint32, uint64. \n

* @par Attributes:
* @li src_format: A string source data format, can be "NHWC", "NCHW" etc.
* @li dst_format: A string target data format, can be "NCHW" etc.
* @li src_subformat: A optional int32 for source sub-format, default value is 0.
* @li dst_subformat: A optional int32 for target sub-format, default value is 0.
* @li groups: A optional int32, the N axis must be divisible by "groups". Defaults to 1. \n

* @par Outputs:
* dst: A Tensor. Support 2D ~ 6D. Has the same type as "src".
*\n
*\n
* The following value ranges must be met.
* '<===>' indicates that format is bidirectionly supported, either input or output.
* '===>' indicates that format is unbidirectionly supported, and the input and
* output data types must be correspond to each other. \n
*\n
*\n
| src_format <===> dst_format | dtype                              | C0    | groups |\n
| :-------------------------: | :--------------------------------: |:-----:| :----: |\n
| NCHW <====> NC1HWC0         | float32, int32,uint32              | 8,16  | 1      |\n
| NCHW <====> NC1HWC0         | bfloat16, float16                  | 16    | 1      |\n
| NCHW <====> NC1HWC0         | int8, uint8                        | 32    | 1      |\n
| NHWC <====> NC1HWC0         | float32, int32,uint32              | 8,16  | 1      |\n
| NHWC <====> NC1HWC0         | bfloat16, float16                  | 16    | 1      |\n
| NHWC <====> NC1HWC0         | int8,  uint8                       | 32    | 1      |\n
| ND <====> FRACTAL_NZ        | float32, int32,uint32              | 16    | 1      |\n
| ND <====> FRACTAL_NZ        | bfloat16, float16                  | 16    | 1      |\n
| ND <====> FRACTAL_NZ        | int8, uint8                        | 32    | 1      |\n
| NCHW <====> FRACTAL_Z       | float32, int32,uint32              | 8,16  | 1      |\n
| NCHW <====> FRACTAL_Z       | bfloat16, float16                  | 16    | 1      |\n
| NCHW <====> FRACTAL_Z       | int8,  uint8                       | 32    | 1      |\n
| HWCN <====> FRACTAL_Z       | float32, int32,uint32              | 8,16  | 1      |\n
| HWCN <====> FRACTAL_Z       | bfloat16, float16                  | 16    | 1      |\n
| HWCN <====> FRACTAL_Z       | int8, uint8                        | 32    | 1      |\n
| NCDHW <====> NDC1HWC0       | float32, int32,uint32              | 8,16  | 1      |\n
| NCDHW <====> NDC1HWC0       | bfloat16, float16                  | 16    | 1      |\n
| NCDHW <====> NDC1HWC0       | int8, uint8                        | 32    | 1      |\n
| NDHWC <====> NDC1HWC0       | float32, int32,uint32              | 8,16  | 1      |\n
| NDHWC <====> NDC1HWC0       | bfloat16, float16                  | 16    | 1      |\n
| NDHWC <====> NDC1HWC0       | int8, uint8                        | 32    | 1      |\n
| NCDHW <====> FRACTAL_Z_3D   | float32, int32,uint32              | 8,16  | 1      |\n
| NCDHW <====> FRACTAL_Z_3D   | bfloat16, float16                  | 16    | 1      |\n
| NCDHW <====> FRACTAL_Z_3D   | int8, uint8                        | 32    | 1      |\n
| DHWCN <====> FRACTAL_Z_3D   | float32, int32,uint32              | 16    | 1      |\n
| DHWCN <====> FRACTAL_Z_3D   | bfloat16, float16                  | 16    | 1      |\n
| DHWCN <====> FRACTAL_Z_3D   | int8, uint8                        | 32    | 1      |\n
| NCHW <====> FRACTAL_Z       | float32, uint32, int32             | 8     | >1     |\n
| NCHW <====> FRACTAL_Z       | float16, bfloat16, uint16, int16   | 16    | >1     |\n
| HWCN ====> FRACTAL_Z        | float16, bfloat16, uint16, int16   | 16    | >1     |\n
| NCDHW <====> FRACTAL_Z_3D   | float32, uint32, int32             | 8     | >1     |\n
| NCDHW <====> FRACTAL_Z_3D   | float16, bfloat16, uint16, int16   | 16    | >1     |\n
| FRACTAL_Z_3D ====> DHWCN    | float32, uint32, int32             | 8     | >1     |\n
| FRACTAL_Z_3D ====> DHWCN    | float16, bfloat16, uint16, int16   | 16    | >1     |\n
| NCHW ====> FRACTAL_Z_C04    | float16, bfloat16                  | 16    | 1      |\n
| FRACTAL_Z_C04 ====> NCHW    | float32                            | 16    | 1      |\n
| ND ====> FRACTAL_NZ_C0_16   | float32, uint32, int32             | 16    | 1      |\n
*\n
*
*/
REG_OP(TransData)
    .INPUT(src, TensorType::BasicType())
    .OUTPUT(dst, TensorType::BasicType())
    .REQUIRED_ATTR(src_format, String)
    .REQUIRED_ATTR(dst_format, String)
    .ATTR(src_subformat, Int, 0)
    .ATTR(dst_subformat, Int, 0)
    .ATTR(groups, Int, 1)
    .OP_END_FACTORY_REG(TransData)

/**
* @brief Unpacks the given dimension of a rank-R Tensor "x" into rank-(R-1)
* tensors.

* @par Inputs:
* x: A rank-R tensor (R > 0) of type BasicType.(BasicType includes:
* complex128, complex64, double, float32, float16, int16, int32, int64, int8,
* qint16, qint32, qint8, quint16, quint8, uint16, uint32, uint64, uint8,
* bfloat16, complex32.) \n

* @par Attributes:
* @li num: A required int, specifying the number of tensors to be unpacked to.
* Defaults to "None".
* @li axis: An optional int, specifying the axis to unpack along. The value range
* is [-R, R). Defaults to "0". \n

* @par Outputs:
* y: Dynamic output. The list of Tensor objects unpacked from "x", of type BasicType . \n

* @attention Constraints:
* For the ND format, "axis" is in the range [-R, R). \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator Unstack.
*/

REG_OP(Unpack)
    .INPUT(x, TensorType::BasicType())
    .DYNAMIC_OUTPUT(y, TensorType::BasicType())
    .REQUIRED_ATTR(num, Int)
    .ATTR(axis, Int, 0)
    .OP_END_FACTORY_REG(Unpack)

/**
* @brief Flattens the inputs tensor into a 2D matrix. If input tensor has shape (d_0, d_1,..., d_n),
*        then the output will have shape (d_0 X d_1 ... d_(axis-1), d_axis X d_(axis + 1)...X d_n)\n

* @par Inputs:
* One input:
* x: A multi-dimensional tensor. All data types are supported.

* @par Outputs:
* y: A 2D flattened tensor with the contents of the input tensor, with input dimensions up to axis flattened
* to the outer dimension of the output and remaining input dimensions flattened into the inner dimension of the output.
* Has the same type as "x".

* @par Attributes:
* axis: A optional int32, default value is 1. Indicate up to which input dimensions (exclusive) should be flattened
* to the outer dimension of the output. The value for axis must be in the range [-r, r], where r is the rank of
* the input tensor. Negative value means counting dimensions from the back. When axis = 0, the shape of
* the output tensor is (1, (d_0 X d_1 ... d_n), where the shape of the input tensor is (d_0, d_1, ... d_n).

* @par Third-party framework compatibility
* Compatible with TensorFlow / ONNX operator Flatten.
*/
REG_OP(Flatten)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .ATTR(axis, Int, 1)
    .OP_END_FACTORY_REG(Flatten)

/**
* @brief Outputs a copy of the input tensor where values from the "height" and
* "width" dimensions are moved to the "depth" dimension . \n

* @par Inputs:
* x: A Tensor. The data type must be one of BasicType.
* The data format must be NCHW or NHWC and must be same as the attribute value data_format.

* @par Attributes:
* @li block_size: A required int, specifying the input block size.
* @li data_format: An optional string, specifying the data format. Must be
*     NCHW or NHWC, and be same as the data format of x. Defaults to "NHWC".

* @par Outputs:
* y: A Tensor. Has the same type as input "x".
* @par Third-party framework compatibility
* Compatible with the TensorFlow operator SpaceToDepth.
*/
REG_OP(SpaceToDepth)
  .INPUT(x, TensorType::BasicType())
  .OUTPUT(y, TensorType::BasicType())
  .REQUIRED_ATTR(block_size, Int)
  .ATTR(data_format, String, "NHWC")
  .OP_END_FACTORY_REG(SpaceToDepth)

/**
* @brief Rearranges data from depth into blocks of spatial data .

* @par Inputs:
* x: A Tensor. The data type must be one of BasicType.
* The data format must be NCHW or NHWC and must be same as the attribute value data_format.

* @par Attributes:
* Three attributes, including:
* @li block_size: An int >= 2, specifying the size of the spatial block.
* @li mode: An optional string, specifying the mode. Must be DCR(depth-column-row)
*     or CRD(column-row-depth). Defaults to "DCR".
* @li data_format: An optional string, specifying the data format. Must be
*     NCHW or NHWC, and be same as the data format of x. Defaults to "NHWC".

* @par Outputs:
* y: A Tensor of the same type as "x". \n

* @par Third-party framework compatibility:
* Compatible with TensorFlow operator DepthToSpace.
*/
REG_OP(DepthToSpace)
  .INPUT(x, TensorType::BasicType())
  .OUTPUT(y, TensorType::BasicType())
  .REQUIRED_ATTR(block_size, Int)
  .ATTR(mode, String, "DCR")
  .ATTR(data_format, String, "NHWC")
  .OP_END_FACTORY_REG(DepthToSpace)

/**
* @brief Zeros-pads and then permutes blocks of spatial data into batch.
* The values from the height and width dimensions are moved in spatial blocks to the batch dimension.
* After zeros-pads the height and width dimensions. \n

* Support formats are as follows:
* @code{.c}
    1.when ori_format is 'NHWC' or 'NCHW', input_format is 'NC1HWC0'

        for example:
            ori:
                x              shape = [16,16,16,16]           format = 'NHWC'
                block_shape    shape = [2,]                    format = 'ND'
                pads           shape = [2,2]                   format = 'ND'
                y              shape = [None,None,None,16]     format = 'NHWC'
            format transformer:
                x              shape = [16,1,16,16,16]         format = 'NC1HWC0'
                block_shape    shape = [2,]                    format = 'ND'
                pads           shape = [2,2]                   format = 'ND'
                y              shape = [None,1,None,None,16]   format = 'NC1HWC0'

    2.when ori_format is 'NDHWC' or 'NCDHW', input_format is 'NDC1HWC0'

        for example:
            ori:
                x              shape = [16,16,16,16,16]              format = 'NDHWC'
                block_shape    shape = [3,]                          format = 'ND'
                pads           shape = [3,2]                         format = 'ND'
                y              shape = [None,None,None,None,16]      format = 'NDHWC'
            format transformer:
                x              shape = [16,16,1,16,16,16]            format = 'NDC1HWC0'
                block_shape    shape = [3,]                          format = 'ND'
                pads           shape = [3,2]                         format = 'ND'
                y              shape = [None,None,1,None,None,16]    format = 'NDC1HWC0'
* @endcode

* @par Inputs:
* @li x: A ND tensor. Format is ND. Must be one of the following types:
* float16, float32, double, int64, int32, uint8, uint16, uint32, uint64, int8,
* int16, complex64, complex128, qint8, quint8, qint16, quint16, qint32, bfloat16.
* @li block_shape: A 1D tensor with shape [M]. Format is ND. Support int32 or int64.
* @li paddings: A 2D tensor with shape [M, 2]. Format is ND. Support int32 or int64. \n

* @par Outputs:
* y: A tensor, the same type and format as "x". \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator SpaceToBatchND.
*/
REG_OP(SpaceToBatchND)
    .INPUT(x, TensorType::BasicType())
    .INPUT(block_shape, TensorType::IndexNumberType())
    .INPUT(paddings, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::BasicType())
    .OP_END_FACTORY_REG(SpaceToBatchND)

/**
* @brief Permutes data from batch into blocks of spatial data and then prunes them.
* The values from the batch dimension are moved in spatial blocks to the height and width dimensions.
* And then prunes the height and width dimensions.

* @par Inputs:
* @li x: A ND tensor, must be one of the following types:
* float16, float32, double, int64, int32, uint8, uint16, uint32, uint64, int8,
* int16, complex64, complex128, qint8, quint8, qint16, quint16, qint32, bfloat16.
* @li block_shape: A 1D tensor with shape [M], support int32 or int64.
* @li crops: A 2D tensor with shape [M, 2], support int32 or int64. \n

* @par Outputs:
* y: A ND tensor, the same type as "x". \n

* @attention Constraints:
* If N is 4 and M is 2: \n
* The size of the first dimension of input "x" must be divisible by (block_size * block_size). \n
* "y" is a 4D shape [batch, height, width, depth], batch = x.shape[0] / block_shape[0] * block_shape[1],
* depth = x.shape[3], height = height_pad - crop_top - crop_bottom, width = width_pad - crop_left - crop_right
* where height_pad = x.shape[1] * block_shape[0], width_pad = x.shape[2] * block_shape[1],
* crop_top = crops[0][0], crop_bottom = crops[0][1], crop_left = crops[1][0], crop_left = crops[1][1]
*@par Third-party framework compatibility
* Compatible with the TensorFlow operator BatchToSpaceND.
*/
REG_OP(BatchToSpaceND)
    .INPUT(x, TensorType::BasicType())
    .INPUT(block_shape, TensorType::IndexNumberType())
    .INPUT(crops, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::BasicType())
    .OP_END_FACTORY_REG(BatchToSpaceND)
}  // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_INC_TRANSFORMATION_OPS_H_
