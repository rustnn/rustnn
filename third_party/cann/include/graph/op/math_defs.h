/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2018-2023. All rights reserved.
 * Description: math_defs
 */
#ifndef INC_API_GRAPH_OP_MATH_DEFS_H
#define INC_API_GRAPH_OP_MATH_DEFS_H
#include "graph/operator_hiai_reg.h"

// clang-format off
namespace hiai {
/*
 * Performs tensor addition.
 * Inputs 'x1' and 'x2' must meet the following constraints:
 *     1. The input shape can be 1D-4D.
 *         1D: 1 is added to the front and 2 is added in the back. For example, 5 is padded into: 1,5,1,1
 *         2D: 1 is added to the front and back. For example, 5,5 is padded into: 1,5,5,1
 *         3D: 1 is added to the front. For example, 5,5,5 is padded into: 1,5,5,5
 *     2. For the two inputs, the corresponding dimensions must have the same value, or one of them is 1.
 * <Input>
 *    x1 : First operand
 *    x2 : Second operand
 * <Output>
 *    y : Result of same element type as the two inputs
 * <Added in HiAI version>
 *    100.300.010.011
 * <Examples>
 *    TensorDesc xDesc1(Shape({4, 5, 6, 7}), FORMAT_NCHW, DT_FLOAT);
 *    hiai::op::Data x1 = hiai::op::Data("x1");
 *    x1.update_input_desc_x(xDesc1);
 *
 *    TensorDesc xdesc2(Shape({1, 5, 1, 1}), FORMAT_NCHW, DT_FLOAT);
 *    hiai::op::Data x2 = hiai::op::Data("x2");
 *    x2.update_input_desc_x(xdesc2);
 *
 *    auto add = hiai::op::Add("add")
 *               .set_input_x1(x1)
 *               .set_input_x2(x2);
 */
HIAI_REG_OP(Add)
.HIAI_INPUT(x1, TensorType({ DT_FLOAT, DT_INT32, DT_INT16 }))
.HIAI_INPUT(x2, TensorType({ DT_FLOAT, DT_INT32, DT_INT16 }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT, DT_INT32, DT_INT16 }))
.HIAI_OP_END()

/*
 * Performs tensor multiplication.
 * <Input>
 *    x1 : A Tensor
 *    x2 : A Tensor, Must have the same type and dimensions as x
 * <Output>
 *    y : Result of the same element type as the two inputs
 * <Added in HiAI version>
 *    100.300.010.011
 */
HIAI_REG_OP(Mul)
.HIAI_INPUT(x1, TensorType({ DT_FLOAT, DT_INT32, DT_INT16 }))
.HIAI_INPUT(x2, TensorType({ DT_FLOAT, DT_INT32, DT_INT16 }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT, DT_INT32, DT_INT16 }))
.HIAI_OP_END()

/*
 * Computes exp(x) - 1 element-wise. y = exp(x) - 1
 * <Input>
 *    x : A Tensor
 * <Output>
 *    y : A Tensor. Has the same type as x.
 * <Added in HiAI version>
 *    100.310.010.013
 */
HIAI_REG_OP(Expm1)
.HIAI_INPUT(x, TensorType({ DT_FLOAT }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT }))
.HIAI_OP_END()

/*
 * Computes smallest integer not less than x.
 * <Input>
 *    x : A Tensor
 * <Output>
 *    y : A Tensor. Has the same type as x.
 * <Added in HiAI version>
 *    100.310.010.013
 */
HIAI_REG_OP(Ceil)
.HIAI_INPUT(x, TensorType({ DT_FLOAT }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT }))
.HIAI_OP_END()

/*
 * Computes sin of 'x' element-wise.
 * <Input>
 *    x : A Tensor
 * <Output>
 *    y : A Tensor. Has the same type as x.
 * <Added in HiAI version>
 *    100.310.010.013
 */
HIAI_REG_OP(Sin)
.HIAI_INPUT(x, TensorType({ DT_FLOAT }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT }))
.HIAI_OP_END()

/*
 * Computes cos of 'x' element-wise.
 * <Input>
 *    x : A Tensor
 * <Output>
 *    y : A Tensor. Has the same type as x.
 * <Added in HiAI version>
 *    100.310.010.013
 */
HIAI_REG_OP(Cos)
.HIAI_INPUT(x, TensorType({ DT_FLOAT }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT }))
.HIAI_OP_END()

/*
 * Computes element-wise largest integer not greater than 'x'.
 * <Input>
 *    x : A Tensor
 * <Output>
 *    y : A Tensor. Has the same type as x.
 * <Added in HiAI version>
 *    100.310.010.013
 */
HIAI_REG_OP(Floor)
.HIAI_INPUT(x, TensorType({ DT_FLOAT }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT }))
.HIAI_OP_END()

/*
 * Computes natural logarithm of (1 + x) element-wise.
 * <Input>
 *    x : A Tensor
 * <Output>
 *    y : A Tensor. Has the same type as x.
 * <Added in HiAI version>
 *    100.310.010.013
 */
HIAI_REG_OP(Log1p)
.HIAI_INPUT(x, TensorType({ DT_FLOAT }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT }))
.HIAI_OP_END()

/*
 * Returns the logical and truth value of 'x1' and 'x2' element-wise.
 * <Input>
 *    x1 : First input operand
 *    x2 : Second input operand
 * <Output>
 *    y : Output tensor
 * <Added in HiAI version>
 *    100.310.010.013
 */
HIAI_REG_OP(LogicalAnd)
.HIAI_INPUT(x1, TensorType({ DT_BOOL }))
.HIAI_INPUT(x2, TensorType({ DT_BOOL }))
.HIAI_OUTPUT(y, TensorType({ DT_BOOL }))
.HIAI_OP_END()

/*
 * Returns the truth value of NOT 'x' element-wise.
 * <Input>
 *    x : Input tensor
 * <Output>
 *    y : Output tensor
 * <Added in HiAI version>
 *    100.310.010.013
 */
HIAI_REG_OP(LogicalNot)
.HIAI_INPUT(x, TensorType({ DT_BOOL }))
.HIAI_OUTPUT(y, TensorType({ DT_BOOL }))
.HIAI_OP_END()

/*
 * Computes the maximum of input tensors 'x1' and 'x2' element-wise.
 * The dimension of 'x1' and 'x2' must be the same unless 'x2' is a scalar.
 * <Input>
 *    x1 : Input tensor. Must be non const OP
 *    x2 : Input tensor. Must be non const OP
 * <Output>
 *    y : Maximum of 'x1' and 'x2'
 * <Added in HiAI version>
 *    100.310.010.013
 */
HIAI_REG_OP(Maximum)
.HIAI_INPUT(x1, TensorType({ DT_FLOAT, DT_INT16 }))
.HIAI_INPUT(x2, TensorType({ DT_FLOAT, DT_INT16 }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT, DT_INT16 }))
.HIAI_OP_END()

/*
 * Computes the minimum of input tensors 'x1' and 'x2' element-wise.
 * <Input>
 *    x1 : Input tensor
 *    x2 : Input tensor
 * <Output>
 *    y : Minimum of 'x1' and 'x2'
 * <Added in HiAI version>
 *    100.310.010.013
 * <Examples>
 *    TensorDesc xDesc1(Shape({4, 5, 6, 7}), FORMAT_NCHW, DT_FLOAT);
 *    hiai::op::Data x1 = hiai::op::Data("x1");
 *    x1.update_input_desc_x(xDesc1);
 *
 *    TensorDesc xDesc2(Shape({4, 5, 6, 7}), FORMAT_NCHW, DT_FLOAT);
 *    hiai::op::Data x2 = hiai::op::Data("x2");
 *    x2.update_input_desc_x(xDesc2);
 *
 *    auto minimum = hiai::op::Minimum("minimum")
 *                   .set_input_x1(x1)
 *                   .set_input_x2(x2);
 */
HIAI_REG_OP(Minimum)
.HIAI_INPUT(x1, TensorType({ DT_FLOAT, DT_INT32 }))
.HIAI_INPUT(x2, TensorType({ DT_FLOAT, DT_INT32 }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT, DT_INT32 }))
.HIAI_OP_END()

/*
 * Calculates whether the input tensors 'x1' and 'x2' are equal, if equal, returns the truth value element-wise.
 * <Input>
 *    x1 : First input operand
 *    x2 : Second input operand
 * <Output>
 *    y : Output tensor
 * <Added in HiAI version>
 *    100.310.010.013
 */
HIAI_REG_OP(Equal)
.HIAI_INPUT(x1, TensorType({ DT_UINT8, DT_FLOAT, DT_BOOL, DT_INT32 }))
.HIAI_INPUT(x2, TensorType({ DT_UINT8, DT_FLOAT, DT_BOOL, DT_INT32 }))
.HIAI_OUTPUT(y, TensorType({ DT_BOOL }))
.HIAI_OP_END()

/*
 * Computes the reciprocal of the input tensor 'x' element-wise.
 * <Input>
 *    x : Input tensor
 * <Output>
 *    y : Tensor of the same type as 'x'
 * <Added in HiAI version>
 *    100.310.010.013
 */
HIAI_REG_OP(Reciprocal)
.HIAI_INPUT(x, TensorType({ DT_FLOAT }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT }))
.HIAI_OP_END()

/*
 * Computes the square root of the input tensor 'x' element-wise.
 * <Input>
 *    x : Input tensor
 * <Output>
 *    y : Tensor of the same type as 'x'
 * <Added in HiAI version>
 *    100.310.010.013
 */
HIAI_REG_OP(Sqrt)
.HIAI_INPUT(x, TensorType({ DT_FLOAT }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT }))
.HIAI_OP_END()

/*
 * Computes the square of the input tensor 'x' element-wise.
 * <Input>
 *    x : Input tensor
 * <Output>
 *    y : Output tensor
 * <Added in HiAI version>
 *    100.310.010.013
 */
HIAI_REG_OP(Square)
.HIAI_INPUT(x, TensorType({ DT_FLOAT }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT }))
.HIAI_OP_END()

/*
 * Casts 'x' of input type 'src_dtype' to 'y' of 'dst_dtype'.
 * <Input>
 *    x : Input tensor to be cast. Must be non const OP.
 * <Output>
 *    y : Output tensor with the same shape as input
 * <Attr>
 *    src_dtype : Data type of the input tensor, same as x
 *                DT_FLOAT(0), DT_INT8(2), DT_INT32(3), DT_UINT8(4), DT_INT64(9), DT_BOOL(12)
 *    dst_dtype : Data type which will be cast to, same as y
 *                DT_FLOAT(0), DT_INT8(2), DT_INT32(3), DT_UINT8(4), DT_INT64(9), DT_BOOL(12)
 * <Added in HiAI version>
 *    100.310.010.013
 * <Examples>
 *    TensorDesc xDesc(Shape({4, 5, 6, 7}), FORMAT_NCHW, DT_FLOAT);
 *    hiai::op::Data x = hiai::op::Data("x");
 *    x.update_input_desc_x(xDesc);
 *
 *    auto castT = hiai::op::CastT("castT")
 *                 .set_input_x(x)
 *                 .set_attr_src_dtype(0)
 *                 .set_attr_dst_dtype(12);
 */
HIAI_REG_OP(CastT)
.HIAI_INPUT(x, TensorType({ DT_BOOL, DT_INT32, DT_UINT8, DT_FLOAT, DT_INT64, DT_INT8 }))
.HIAI_OUTPUT(y, TensorType({ DT_BOOL, DT_INT32, DT_UINT8, DT_FLOAT, DT_INT64, DT_INT8 }))
.HIAI_REQUIRED_ATTR(src_dtype, AttrValue::INT)
.HIAI_REQUIRED_ATTR(dst_dtype, AttrValue::INT)
.HIAI_OP_END()

/*
 * Returns an indication of the sign of a number.
 * <Input>
 *    x : A Tensor
 * <Output>
 *    y : Tensor of the same type as 'x'
 * <Added in HiAI version>
 *    100.310.010.013
 */
HIAI_REG_OP(Sign)
.HIAI_INPUT(x, TensorType({ DT_FLOAT }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT }))
.HIAI_OP_END()

/*
 * Computes exponential of 'x' element-wise. y = base^(scale*x+shift)
 * <Input>
 *    x : Input tensor
 * <Output>
 *    y : Output tensor
 * <Attr>
 *    base : Default -1 for a value of e
 *    scale : The scale
 *    shift : The shift
 * <Added in HiAI version>
 *    100.310.010.013
 */
HIAI_REG_OP(Exp)
.HIAI_INPUT(x, TensorType({ DT_FLOAT }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT }))
.HIAI_ATTR(base, AttrValue::FLOAT { -1.0 })
.HIAI_ATTR(scale, AttrValue::FLOAT { 1.0 })
.HIAI_ATTR(shift, AttrValue::FLOAT { 0.0 })
.HIAI_OP_END()

/*
 * Computes element-wise remainder of division, discarding decimal places in the negative infinity direction.
 * <Input>
 *    x1 : First input tensor
 *    x2 : Second input tensor
 * <Output>
 *    y : Output tensor
 * <Added in HiAI version>
 *    100.310.010.013
 */
HIAI_REG_OP(FloorMod)
.HIAI_INPUT(x1, TensorType({ DT_FLOAT }))
.HIAI_INPUT(x2, TensorType({ DT_FLOAT }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT }))
.HIAI_OP_END()

/*
 * Returns the truth value of (x1 >= x2) element-wise.
 * <Input>
 *    x1 : First input tensor
 *    x2 : Second input tensor
 * <Output>
 *    y : Output tensor
 * <Added in HiAI version>
 *    100.310.010.013
 */
HIAI_REG_OP(GreaterEqual)
.HIAI_INPUT(x1, TensorType({ DT_FLOAT, DT_INT32 }))
.HIAI_INPUT(x2, TensorType({ DT_FLOAT, DT_INT32 }))
.HIAI_OUTPUT(y, TensorType({ DT_BOOL }))
.HIAI_OP_END()

/*
 * Returns the truth value of (x1 > x2) element-wise.
 * <Input>
 *    x1 : First input tensor
 *    x2 : Second input tensor
 * <Output>
 *    y : Output tensor
 * <Added in HiAI version>
 *    100.500.010.010
 * <Examples>
 *    TensorDesc xDesc1(Shape({4, 5, 6, 7}), FORMAT_NCHW, DT_FLOAT);
 *    hiai::op::Data x1 = hiai::op::Data("x1");
 *    x1.update_input_desc_x(xDesc1);
 *
 *    TensorDesc xDesc2(Shape({4, 5, 6, 7}), FORMAT_NCHW, DT_FLOAT);
 *    hiai::op::Data x2 = hiai::op::Data("x2");
 *    x2.update_input_desc_x(xDesc2);
 *
 *    auto greater = hiai::op::Greater("greater")
 *                   .set_input_x1(x1)
 *                   .set_input_x2(x2);
 */
HIAI_REG_OP(Greater)
.HIAI_INPUT(x1, TensorType({ DT_FLOAT, DT_INT32, DT_INT8, DT_UINT8 }))
.HIAI_INPUT(x2, TensorType({ DT_FLOAT, DT_INT32, DT_INT8, DT_UINT8 }))
.HIAI_OUTPUT(y, TensorType({ DT_BOOL }))
.HIAI_OP_END()

/*
 * Returns the truth value of (x1 < x2) element-wise.
 * <Input>
 *    x1 : First input tensor
 *    x2 : Second input tensor
 * <Output>
 *    y : Output tensor
 * <Added in HiAI version>
 *    100.310.010.013
 */
HIAI_REG_OP(Less)
.HIAI_INPUT(x1, TensorType({ DT_FLOAT }))
.HIAI_INPUT(x2, TensorType({ DT_FLOAT }))
.HIAI_OUTPUT(y, TensorType({ DT_BOOL }))
.HIAI_OP_END()

/*
 * Multiplies matrix 'x1' by matrix 'x2'.
 * If both inputs are 2-D, the inner dimension of 'x1' (after being transposed if 'transpose_x1' is true)
 * must match the outer dimension of 'x2' (after being transposed if 'transposed_x2' is true).
 * If the first argument is 1-D, it is promoted to a matrix by prepending a 1 to its dimensions.
 * After matrix multiplication the prepended 1 is removed.
 * If the second argument is 1-D, it is promoted to a matrix by appending a 1 to its dimensions.
 * After matrix multiplication the appended 1 is removed.
 * <Input>
 *    x1 : First input tensor.
 *    x2 : Second input tensor.
 *    bias : Reserved - Optional input tensor, bias data
 * <Output>
 *    y : Output tensor
 * <Attr>
 *    transpose_x1 : If true, 'x1' is transposed before multiplication.
 *    transpose_x2 : If true, 'x2' is transposed before multiplication.
 * <Added in HiAI version>
 *    100.310.010.013
 * <Examples>
 *    TensorDesc xDesc1(Shape({4, 20}), FORMAT_NCHW, DT_FLOAT);
 *    hiai::op::Data x1 = hiai::op::Data("x1");
 *    x1.update_input_desc_x(xDesc1);
 *
 *    TensorDesc xDesc2(Shape({20, 2}), FORMAT_NCHW, DT_FLOAT);
 *    hiai::op::Data x2 = hiai::op::Data("x2");
 *    x2.update_input_desc_x(xDesc2);
 *
 *    auto matMul = hiai::op::MatMul("matMul")
 *                  .set_input_x1(x1)
 *                  .set_input_x2(x2)
 *                  .set_attr_transpose_x1(false)
 *                  .set_attr_transpose_x2(false);
 */
HIAI_REG_OP(MatMul)
.HIAI_INPUT(x1, TensorType({ DT_FLOAT, DT_UINT8 }))
.HIAI_INPUT(x2, TensorType({ DT_FLOAT, DT_INT8 }))
.HIAI_OPTIONAL_INPUT(bias, TensorType({ DT_FLOAT, DT_INT32 }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT, DT_INT32 }))
.HIAI_ATTR(transpose_x1, AttrValue::BOOL { false })
.HIAI_ATTR(transpose_x2, AttrValue::BOOL { false })
.HIAI_OP_END()

/*
 * Multiplies matrix 'x1' by matrix 'x2'.
 * The inputs must be two-dimensional matrices and the inner dimension of 'x1' (after being
 * transposed if 'transpose_x1' is true)
 * must match the outer dimension of 'x2' (after being transposed if 'transposed_x2' is true).
 * <Input>
 *    x1 : First input tensor. Must be non const OP
 *    x2 : Second input tensor. Must be const OP
 *    bias : Optional input tensor, bias data. If x2_quant_type = 1, shall use type 'tensor(int32)'.
 * <Output>
 *    y : Output tensor
 * <Attr>
 *    transpose_x1 : If true, 'x1' is transposed before multiplication.
 *    transpose_x2 : If true, 'x2' is transposed before multiplication.
 *    x1_quant_type : Either 0-No quant or 1-Quant8 (linear); Reserved: 2-Quant4 (linear), 3-Quant2 (linear)
 *    x2_quant_type : Either 0-No quant or 1-Quant8 (linear); Reserved: 2-Quant4 (linear), 3-Quant2 (linear)
 *    x1_quant_scale : If x1_quant_type > 0, this Attr is required.
 *    x1_quant_offset : x1 offset
 *    x2_quant_scales : If x2_quant_type > 0, this Attr is required. The number must be 1.
 * <Added in HiAI version>
 *    100.500.010.010
 */
HIAI_REG_OP(QuantizedMatMul)
.HIAI_INPUT(x1, TensorType({ DT_FLOAT }))
.HIAI_INPUT(x2, TensorType({ DT_FLOAT, DT_INT8 }))
.HIAI_OPTIONAL_INPUT(bias, TensorType({ DT_FLOAT, DT_INT32 }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT }))
.HIAI_ATTR(transpose_x1, AttrValue::BOOL { false })
.HIAI_ATTR(transpose_x2, AttrValue::BOOL { false })
.HIAI_ATTR(x1_quant_type, AttrValue::INT { 0 })
.HIAI_ATTR(x2_quant_type, AttrValue::INT { 0 })
.HIAI_ATTR(x1_quant_scale, AttrValue::FLOAT { 1.0 })
.HIAI_ATTR(x1_quant_offset, AttrValue::INT { 0 })
.HIAI_ATTR(x2_quant_scales, AttrValue::LIST_FLOAT ({}))
.HIAI_OP_END()

/*
 * Computes an inner product with an input tensor, a set of learned weights and adds biases.
 * if shape of a is M*K, shape of b is  K*N, shape of c must be N or (1,N).
 * <Input>
 *    a : Input tensor.
 *    b : Input tensor.
 *    c : Input tensor.
 * <Output>
 *    y : Output tensor
 * <Attr>
 *    alpha : Scalar multiplier for the product of input tensors a * b.
 *    beta : Scalar multiplier for input tensor c.
 *    transpose_a : Whether a should be transposed.
 *    transpose_b : Whether b should be transposed.
 * <Added in HiAI version>
 *    100.500.010.010
 */
HIAI_REG_OP(GemmD)
.HIAI_INPUT(a, TensorType({ DT_FLOAT, DT_UINT8 }))
.HIAI_INPUT(b, TensorType({ DT_FLOAT, DT_INT8 }))
.HIAI_OPTIONAL_INPUT(c, TensorType({ DT_FLOAT, DT_INT32 }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT, DT_INT32 }))
.HIAI_ATTR(alpha, AttrValue::FLOAT { 1.0f })
.HIAI_ATTR(beta, AttrValue::FLOAT { 1.0f })
.HIAI_ATTR(transpose_a, AttrValue::BOOL { false })
.HIAI_ATTR(transpose_b, AttrValue::BOOL { false })
.HIAI_OP_END()

/*
 * Returns x1/x2 element-wise for real types.
 * For the two inputs, the corresponding dimensions must have the same value, or one of them is 1.
 * <Input>
 *    x1 : First input tensor
 *    x2 : Second input tensor
 * <Output>
 *    y : Output tensor
 * <Added in HiAI version>
 *    100.310.010.013
 */
HIAI_REG_OP(RealDiv)
.HIAI_INPUT(x1, TensorType({ DT_FLOAT, DT_INT32 }))
.HIAI_INPUT(x2, TensorType({ DT_FLOAT, DT_INT32 }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT, DT_INT32 }))
.HIAI_OP_END()

/*
 * Finds and returns the integer closest to 'x', and if the result is between two representable values,
 * select an even representation.
 * <Input>
 *    x : Input tensor
 * <Output>
 *    y : Output tensor
 * <Added in HiAI version>
 *    100.310.010.013
 */
HIAI_REG_OP(Rint)
.HIAI_INPUT(x, TensorType({ DT_FLOAT }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT }))
.HIAI_OP_END()

/*
 * Rounds the value of tensor to the nearest integer by element.
 * <Input>
 *    x : Input tensor
 * <Output>
 *    y : Output tensor
 * <Added in HiAI version>
 *    100.310.010.013
 */
HIAI_REG_OP(Round)
.HIAI_INPUT(x, TensorType({ DT_FLOAT }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT }))
.HIAI_OP_END()

/*
 * Calculates the reciprocal of the square root for input 'x'.
 * <Input>
 *    x : Input tensor
 * <Output>
 *    y : the output tensor, has the same type as x
 * <Added in HiAI version>
 *    100.310.010.013
 */
HIAI_REG_OP(Rsqrt)
.HIAI_INPUT(x, TensorType({ DT_FLOAT }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT }))
.HIAI_OP_END()

/*
 * Returns x1 - x2.
 * For the two inputs, the corresponding dimensions must have the same value, or one of them is 1.
 * <Input>
 *    x1 : First input tensor
 *    x2 : Second input tensor
 * <Output>
 *    y : Output tensor
 * <Added in HiAI version>
 *    100.310.010.013
 */
HIAI_REG_OP(Sub)
.HIAI_INPUT(x1, TensorType({ DT_FLOAT, DT_INT32, DT_INT16 }))
.HIAI_INPUT(x2, TensorType({ DT_FLOAT, DT_INT32, DT_INT16 }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT, DT_INT32, DT_INT16 }))
.HIAI_OP_END()

/*
 * According to the input requirements, creates a sequence of numbers.
 * <Input>
 *    start : 0-D Tensor (scalar). Acts as first entry in the range
 *    limit : 0-D Tensor (scalar). Upper limit of sequence
 *    delta : 0-D Tensor (scalar). Number that increments 'start'.
 * <Output>
 *    y : Output tensor
 * <Added in HiAI version>
 *    100.310.010.013
 */
HIAI_REG_OP(Range)
.HIAI_INPUT(start, TensorType({ DT_FLOAT, DT_INT32 }))
.HIAI_INPUT(limit, TensorType({ DT_FLOAT, DT_INT32 }))
.HIAI_INPUT(delta, TensorType({ DT_FLOAT, DT_INT32 }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT, DT_INT32 }))
.HIAI_OP_END()

/*
 * Computes acos of input 'x'.
 * <Input>
 *    x : Input tensor, of range [-1, 1]
 * <Output>
 *    y : Output tensor of range [0, pi]. Has the same type as 'x'.
 * <Added in HiAI version>
 *    100.310.010.013
 */
HIAI_REG_OP(Acos)
.HIAI_INPUT(x, TensorType({ DT_FLOAT }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT }))
.HIAI_OP_END()

/*
 * Computes asin of input 'x'.
 * <Input>
 *    x : Input tensor, of range [-1, 1]
 * <Output>
 *    y : Output tensor of range [-pi/2, pi/2]. Has the same type as 'x'.
 * <Added in HiAI version>
 *    100.310.010.013
 */
HIAI_REG_OP(Asin)
.HIAI_INPUT(x, TensorType({ DT_FLOAT }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT }))
.HIAI_OP_END()

/*
 * LogLayer computes outputs y = log_base(shift + scale * x), for base > 0.
 * Or if base is set to the default (-1), base is set to e,
 * so y = ln(shift + scale * x) = log_e(shift + scale * x)
 * <Input>
 *    x : Input tensor.
 * <Output>
 *    y : Output Tensor.
 * <Attr>
 *    base : float, default is -1, which means set base to e
 *    scale : float, default is 1, which multiplies x
 *    shift : float, default is 0, which adds to scale*x
 * <Added in HiAI version>
 *    100.310.010.013
 */
HIAI_REG_OP(Log)
.HIAI_INPUT(x, TensorType({ DT_FLOAT }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT }))
.HIAI_ATTR(base, AttrValue::FLOAT { -1.0 })
.HIAI_ATTR(scale, AttrValue::FLOAT { 1.0 })
.HIAI_ATTR(shift, AttrValue::FLOAT { 0.0 })
.HIAI_OP_END()

/*
 * Computes natural logarithm of input 'x'.
 * <Input>
 *    x1 : Tensor of type bool
 *    x2 : Tensor of type bool
 * <Output>
 *    y : Tensor of type bool
 * <Added in HiAI version>
 *    100.310.010.013
 */
HIAI_REG_OP(LogicalOr)
.HIAI_INPUT(x1, TensorType({ DT_BOOL }))
.HIAI_INPUT(x2, TensorType({ DT_BOOL }))
.HIAI_OUTPUT(y, TensorType({ DT_BOOL }))
.HIAI_OP_END()

/*
 * Computes numerical negative value input.
 * <Input>
 *    x : Input tensor
 * <Output>
 *    y : Output tensor
 * <Added in HiAI version>
 *    100.310.010.013
 */
HIAI_REG_OP(Neg)
.HIAI_INPUT(x, TensorType({ DT_FLOAT }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT }))
.HIAI_OP_END()

/*
 * Returns the product of elements across dimensions of the input tensor.
 * <Input>
 *    x : Input tensor
 * <Output>
 *    y : Output tensor
 * <Attr>
 *    axes : Dimensions to reduce. Must be in the range [-rank(x), rank(x)).
 *    keep_dims : If true, retains reduced dimensions with length 1.
 * <Added in HiAI version>
 *    100.310.010.013
 */
HIAI_REG_OP(ReduceProdD)
.HIAI_INPUT(x, TensorType({ DT_FLOAT, DT_INT32, DT_UINT8 }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT, DT_INT32, DT_UINT8 }))
.HIAI_REQUIRED_ATTR(axes, AttrValue::LIST_INT)
.HIAI_ATTR(keep_dims, AttrValue::BOOL { false })
.HIAI_OP_END()

/*
 * Returns the sum of elements across dimensions of the input tensor.
 * <Input>
 *    x : Tensor to reduce
 *    axes : Dimensions to reduce
 * <Output>
 *    y : Output tensor
 * <Attr>
 *    keep_dims : If true, retains reduced dimensions with length 1.
 * <Added in HiAI version>
 *    100.310.010.013
 */
HIAI_REG_OP(ReduceSum)
.HIAI_INPUT(x, TensorType({ DT_FLOAT, DT_INT32, DT_UINT8 }))
.HIAI_INPUT(axes, TensorType({ DT_INT32 }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT, DT_INT32, DT_UINT8 }))
.HIAI_ATTR(keep_dims, AttrValue::BOOL { false })
.HIAI_OP_END()

/*
 * Computes tan of input 'x'.
 * <Input>
 *    x : Input tensor
 * <Output>
 *    y : Output tensor
 * <Added in HiAI version>
 *    100.310.010.013
 */
HIAI_REG_OP(Tan)
.HIAI_INPUT(x, TensorType({ DT_FLOAT }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT }))
.HIAI_OP_END()

/*
 * Computes y = (alpha*x + beta)^gamma, as specified by the scale:alpha, shift:beta, and power:gamma.
 * <Input>
 *    x : the input tensor, must be non const op.
 * <Output>
 *    y : the computed outputs y = (alpha*x + beta)^gamma.
 * <Attr>
 *    scale : (optional, default 1) the scale : alpha.
 *    shift : (optional, default 0) the shift : beta.
 *    power : (optional, default 1) the power : gamma.
 * <Added in HiAI version>
 *    100.320.010.010
 */
HIAI_REG_OP(Power)
.HIAI_INPUT(x, TensorType({ DT_FLOAT }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT }))
.HIAI_ATTR(scale, AttrValue::FLOAT { 1.0 })
.HIAI_ATTR(shift, AttrValue::FLOAT { 0.0 })
.HIAI_ATTR(power, AttrValue::FLOAT { 1.0 })
.HIAI_OP_END()

/*
 * Computes x1^x2 for corresponding elements in x1 and x2.
 * <Input>
 *    x1 : the input tensor.
 *    x2 : the exponent value, the datatype of x1 and x2 must be equal.
 * <Output>
 *    y : return a tensor for the result of x1^x2.
 * <Added in HiAI version>
 *    100.320.010.010
 */
HIAI_REG_OP(Pow)
.HIAI_INPUT(x1, TensorType({ DT_FLOAT, DT_INT32 }))
.HIAI_INPUT(x2, TensorType({ DT_FLOAT, DT_INT32 }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT, DT_INT32 }))
.HIAI_OP_END()

/*
 * Determine the largest value of inputs. Returns the index with the largest value across axis of a tensor.
 * <Input>
 *    x : First input tensor
 *    axis : Second input tensor, must be const op. Axis of the input tensor to reduce
 * <Output>
 *    y : Output tensor, max value index or max value
 * <Attr>
 *    output_type : Data type of output tensor
 *    keep_dims : If false,the rank of the tensor is reduced by 1 for each entry in axis.
 *                If true, the reduced dimensions are retained with length 1.
 *    outmaxval : If true, the max value is returned; if false, the max value index or value is returned.
 *                Only outmaxval = false is supported.
 *    topk : The number k of maximal items to output.
 * <Added in HiAI version>
 *    100.310.010.013
 */
HIAI_REG_OP(ArgMaxExt2)
.HIAI_INPUT(x, TensorType({ DT_FLOAT, DT_INT32 }))
.HIAI_INPUT(axis, TensorType({ DT_INT32 }))
.HIAI_OUTPUT(y, TensorType({ DT_INT32, DT_INT64 }))
.HIAI_ATTR(output_type, AttrValue::INT { 3 })
.HIAI_ATTR(keep_dims, AttrValue::BOOL { false })
.HIAI_ATTR(outmaxval, AttrValue::BOOL { false })
.HIAI_ATTR(topk, AttrValue::INT { 1 })
.HIAI_OP_END()

/*
 * Divides x1/x2 element-wise, rounding toward the most negative integer.
 * <Input>
 *    x1 : First input tensor
 *    x2 : Second input tensor
 * <Output>
 *    y : Output tensor
 * <Added in HiAI version>
 *    100.310.010.013
 */
HIAI_REG_OP(FloorDiv)
.HIAI_INPUT(x1, TensorType({ DT_FLOAT, DT_INT32 }))
.HIAI_INPUT(x2, TensorType({ DT_FLOAT, DT_INT32 }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT, DT_INT32 }))
.HIAI_OP_END()

/*
 * Returns the truth value of (x1 != x2) element-wise.
 * <Input>
 *    x1 : Tensor of type float.
 *    x2 : Tensor of the same type as 'x1'.
 * <Output>
 *    y : Tensor of type bool
 * <Added in HiAI version>
 *    100.320.010.010
 */
HIAI_REG_OP(NotEqual)
.HIAI_INPUT(x1, TensorType({ DT_FLOAT }))
.HIAI_INPUT(x2, TensorType({ DT_FLOAT }))
.HIAI_OUTPUT(y, TensorType({ DT_BOOL }))
.HIAI_OP_END()

/*
 * Returns the truth value of (x1 <= x2) element-wise.
 * <Input>
 *    x1 : Tensor of type float.
 *    x2 : Tensor of the same type as 'x1'.
 * <Output>
 *    y : Tensor of type bool.
 * <Added in HiAI version>
 *    100.320.010.010
 */
HIAI_REG_OP(LessEqual)
.HIAI_INPUT(x1, TensorType({ DT_FLOAT }))
.HIAI_INPUT(x2, TensorType({ DT_FLOAT }))
.HIAI_OUTPUT(y, TensorType({ DT_BOOL }))
.HIAI_OP_END()

/*
 * Returns a new tensor of (x1 - x2)(x1 - x2).
 * <Input>
 *    x1 : the first input tensor.
 *    x2 : the second input tensor.
 * <Output>
 *    y : the output tensor.
 * <Added in HiAI version>
 *    100.320.010.010
 */
HIAI_REG_OP(SquaredDifference)
.HIAI_INPUT(x1, TensorType({ DT_FLOAT }))
.HIAI_INPUT(x2, TensorType({ DT_FLOAT }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT }))
.HIAI_OP_END()

/*
 * Computes the trignometric inverse tangent of x input.
 * <Input>
 *    x : the input tensor.
 * <Output>
 *    y : the output tensor.
 * <Added in HiAI version>
 *    100.320.010.010
 */
HIAI_REG_OP(Atan)
.HIAI_INPUT(x, TensorType({ DT_FLOAT }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT }))
.HIAI_OP_END()

/*
 * Calculates the error function of the given input tensor, element-wise.
 * <Input>
 *    x : input tensor
 * <Output>
 *    y : Output tensor
 * <Added in HiAI version>
 *    100.500.010.010
 */
HIAI_REG_OP(Erf)
.HIAI_INPUT(x, TensorType({ DT_UINT8, DT_FLOAT, DT_INT32 }))
.HIAI_OUTPUT(y, TensorType({ DT_UINT8, DT_FLOAT, DT_INT32 }))
.HIAI_OP_END()

/*
 * Multiplies slices of two tensors in batches.
 * <Input>
 *    x1 : The input tensor
 *    x2 : The input tensor
 * <Output>
 *    y : The output tensor
 * <Attr>
 *    adj_x1 : adj_x1 is true, the input tensor x1  is  transposed, otherwise it will not be transposed.
 *             Default is false.
 *    adj_x2 : adj_x2 is true, the input tensor x2  is  transposed, otherwise it will not be transposed.
 *             Default is false.
 * <Added in HiAI version>
 *    100.320.010.010
 */
HIAI_REG_OP(BatchMatMul)
.HIAI_INPUT(x1, TensorType({ DT_FLOAT }))
.HIAI_INPUT(x2, TensorType({ DT_FLOAT }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT }))
.HIAI_ATTR(adj_x1, AttrValue::BOOL { false })
.HIAI_ATTR(adj_x2, AttrValue::BOOL { false })
.HIAI_OP_END()

/*
 * Given a tensor x, this operation returns a tensor of the same type and shape as x with its values clipped to
 * clip_value_min and clip_value_max. Any values less than clip_value_min are set to clip_value_min.
 * Any values greater than clip_value_max are set to clip_value_max.
 * <Input>
 *    x : A Tensor or IndexedSlices
 *    clip_value_min : The value is a 0-D (scalar) Tensor, or a Tensor with the same shape as x.
 *                     The minimum value to clip by.
 *    clip_value_max : The value is a 0-D (scalar) Tensor, or a Tensor with the same shape as x.
 *                     The maximum value to clip by.
 * <Output>
 *    y : the output tensor.
 * <Added in HiAI version>
 *    100.320.010.010
 */
HIAI_REG_OP(ClipByValue)
.HIAI_INPUT(x, TensorType({ DT_FLOAT }))
.HIAI_INPUT(clip_value_min, TensorType({ DT_FLOAT }))
.HIAI_INPUT(clip_value_max, TensorType({ DT_FLOAT }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT }))
.HIAI_OP_END()

/*
 * Standardize the specified dimension using the L2 norm.
 * <Input>
 *    x : the input tensor to be standardized, the input shape can be 1D-4D.
 * <Output>
 *    y : the output tensor with the same shape as x.
 * <Attr>
 *    axis : the dimension which to be standardized. Support { 1 }.
 *    eps : the lower limit of the standardization, when the calculated sum of the squares of the elements
 *          of x is less than eps, using eps instead of the sum of the elements of x.
 * <Added in HiAI version>
 *    100.320.010.010
 */
HIAI_REG_OP(L2Normalize)
.HIAI_INPUT(x, TensorType({ DT_FLOAT }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT }))
.HIAI_ATTR(axis, AttrValue::LIST_INT ({ 1 }))
.HIAI_ATTR(eps, AttrValue::FLOAT { 1e-4f })
.HIAI_OP_END()

/*
 * Computes the maximum of elements across dimensions of a tensor.
 * <Input>
 *    x : The tensor to reduce. Should have real numeric type.
 *    axes : The dimensions to reduce.Must be in the range [-rank(x), rank(x)). Const Op.
 * <Output>
 *    y : the output tensor.
 * <Attr>
 *    keep_dims : If true, retains reduced dimensions with length 1.
 * <Added in HiAI version>
 *    100.320.010.010
 * <Examples>
 *    TensorDesc xDesc(Shape({4, 5, 6, 7}), FORMAT_NCHW, DT_FLOAT);
 *    hiai::op::Data x = hiai::op::Data("x");
 *    x.update_input_desc_x(xDesc);
 *
 *    TensorDesc axesTensorDesc(Shape({2}), FORMAT_NCHW, DT_INT32);
 *    TensorPtr axesTensor = std::make_shared<hiai::Tensor>(axesTensorDesc);
 *    vector<int32_t> dataValue = {0, 1};
 *    axesTensor->SetData((uint8_t*)dataValue.data(), 2 * sizeof(int32_t));
 *    auto axes = hiai::op::Const("axes").set_attr_value(axesTensor);
 *
 *    auto reduceMax = hiai::op::ReduceMax("ReduceMax")
 *                     .set_input_x(x)
 *                     .set_input_axes(axes)
 *                     .set_attr_keep_dims(false);
 */
HIAI_REG_OP(ReduceMax)
.HIAI_INPUT(x, TensorType({ DT_FLOAT, DT_INT32, DT_UINT8 }))
.HIAI_INPUT(axes, TensorType({ DT_INT32 }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT, DT_INT32, DT_UINT8 }))
.HIAI_ATTR(keep_dims, AttrValue::BOOL { false })
.HIAI_OP_END()

/*
 * Computes the minimum of elements across dimensions of a tensor.
 * <Input>
 *    x : The tensor to reduce. Should have real numeric type.
 *    axes : The dimensions to reduce.Must be in the range [-rank(x), rank(x)). Const Op.
 * <Output>
 *    y : The output tensor.
 * <Attr>
 *    keep_dims : If true, retains reduced dimensions with length 1.
 * <Added in HiAI version>
 *    100.320.010.010
 * <Examples>
 *    TensorDesc xDesc(Shape({4, 5, 6, 7}), FORMAT_NCHW, DT_FLOAT);
 *    hiai::op::Data x = hiai::op::Data("x");
 *    x.update_input_desc_x(xDesc);
 *
 *    TensorDesc axesTensorDesc(Shape({1}), FORMAT_NCHW, DT_INT32);
 *    TensorPtr axesTensor = std::make_shared<hiai::Tensor>(axesTensorDesc);
 *    vector<int32_t> dataValue = {1};
 *    axesTensor->SetData((uint8_t*)dataValue.data(), 1 * sizeof(int32_t));
 *    auto axes = hiai::op::Const("axes").set_attr_value(axesTensor);
 *
 *    auto reduceMin = hiai::op::ReduceMin("ReduceMin")
 *                     .set_input_x(x)
 *                     .set_input_axes(axes)
 *                     .set_attr_keep_dims(false);
 */
HIAI_REG_OP(ReduceMin)
.HIAI_INPUT(x, TensorType({ DT_FLOAT }))
.HIAI_INPUT(axes, TensorType({ DT_INT32 }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT }))
.HIAI_ATTR(keep_dims, AttrValue::BOOL { false })
.HIAI_OP_END()

/*
 * The function of ReduceMean operator is computing the mean of elements across dimensions of a tensor.
 * <Input>
 *    x : The input tensor, with the type of Data.
 *    axes : Const tensor. Describes the dimensions to reduce.
 *           Must be in the range [-rank(input_tensor), rank(input_tensor)).
 * <Output>
 *    y : The reduced tensor.
 * <Attr>
 *    keep_dims : If true, retains reduced dimensions with length 1.
 * <Added in HiAI version>
 *    100.320.010.010
 * <Examples>
 *    TensorDesc xDesc(Shape({4, 5, 6, 7}), FORMAT_NCHW, DT_FLOAT);
 *    hiai::op::Data x = hiai::op::Data("x");
 *    x.update_input_desc_x(xDesc);
 *
 *    TensorDesc axesTensorDesc(Shape({1}), FORMAT_NCHW, DT_INT32);
 *    TensorPtr axesTensor = std::make_shared<hiai::Tensor>(axesTensorDesc);
 *    vector<int32_t> dataValue = {1};
 *    axesTensor->SetData((uint8_t*)dataValue.data(), 1 * sizeof(int32_t));
 *    auto axes = hiai::op::Const("axes").set_attr_value(axesTensor);
 *
 *    auto reduceMean = hiai::op::ReduceMean("reduceMean")
 *                      .set_input_x(x)
 *                      .set_input_axes(axes)
 *                      .set_attr_keep_dims(false);
 */
HIAI_REG_OP(ReduceMean)
.HIAI_INPUT(x, TensorType({ DT_FLOAT }))
.HIAI_INPUT(axes, TensorType({ DT_INT32 }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT }))
.HIAI_ATTR(keep_dims, AttrValue::BOOL { false })
.HIAI_OP_END()

/*
 * Computes log(sum(exp(elements across dimensions of a tensor))).
 * <Input>
 *    x : input tensor
 * <Output>
 *    y : output tensor
 * <Attr>
 *    axes : The dimensions to reduce.
 *           Must be in the range [-rank(x), rank(x)).
 *    keepdims : If false,the rank of the tensor is reduced by 1 for each entry in axis.
 *               If true, the reduced dimensions are retained with length 1.
 * <Added in HiAI version>
 *    100.500.010.011
 */
HIAI_REG_OP(ReduceLogSumExp)
.HIAI_INPUT(x, TensorType({ DT_FLOAT }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT }))
.HIAI_REQUIRED_ATTR(axes, AttrValue::LIST_INT)
.HIAI_ATTR(keepdims, AttrValue::BOOL { false })
.HIAI_OP_END()

/*
 * Computes the L2 norm of the input tensor's element along the provided axes.
 * <Input>
 *    x : input tensor
 * <Output>
 *    y : output tensor
 * <Attr>
 *    axes : The dimensions to reduce. Must be in the range [-rank(x), rank(x)).
 *    keep_dims : If keep_dims is false, the rank of the tensor is reduced by 1 for each entry in axis.
 *                If keep_dims is true, the reduced dimensions are retained with length 1.
 * <Added in HiAI version>
 *    100.500.010.011
 */
HIAI_REG_OP(ReduceL2D)
.HIAI_INPUT(x, TensorType({ DT_FLOAT }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT }))
.HIAI_REQUIRED_ATTR(axes, AttrValue::LIST_INT)
.HIAI_ATTR(keep_dims, AttrValue::BOOL { false })
.HIAI_OP_END()

/*
 * Use the sum or mean operation to apply the dimension specified by the input BLOB based on the given parameter.
 * (Generally speaking, the input feature graph is summed or averaged based on a given dimension.)
 * <Input>
 *    x : First input tensor
 * <Output>
 *    y : Output tensor
 * <Attr>
 *    operation : default = ASUM, enum ReductionOp { SUM = 1; ASUM = 2; SUMSQ = 3; MEAN = 4; }
 *    axis : Dimensions to reduce. If None (the default), reduces all dimensions.
 *           Must be in the range [-rank(x), rank(x)).
 *    coeff : default = 1.0 coefficient for output.
 * <Added in HiAI version>
 *    100.500.010.010
 */
HIAI_REG_OP(Reduction)
.HIAI_INPUT(x, TensorType({ DT_FLOAT }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT }))
.HIAI_ATTR(operation, AttrValue::STR { "ASUM" })
.HIAI_ATTR(axis, AttrValue::INT { 0 })
.HIAI_ATTR(coeff, AttrValue::FLOAT { 1.0 })
.HIAI_OP_END()

/*
 * Returns x1/x2 element-wise for integer types.
 * <Input>
 *    x1 : First input tensor.
 *    x2 : Second input tensor, must have the same type as x1.
 * <Output>
 *    y : Output tensor.
 * <Added in HiAI version>
 *    100.500.010.010
 */
HIAI_REG_OP(TruncateDiv)
.HIAI_INPUT(x1, TensorType({ DT_UINT8 }))
.HIAI_INPUT(x2, TensorType({ DT_UINT8 }))
.HIAI_OUTPUT(y, TensorType({ DT_UINT8 }))
.HIAI_OP_END()

/*
 * Computes x multiplied by the logarithm of y element-wise, if x == 0, returns 0.
 * <Input>
 *    x1 : First input tensor of type float32.
 *    x2 : Second input tensor, must have the same type as x1.
 * <Output>
 *    y : Output tensor of type float32.
 * <Added in HiAI version>
 *    100.500.010.010
 */
HIAI_REG_OP(Xlogy)
.HIAI_INPUT(x1, TensorType({ DT_FLOAT }))
.HIAI_INPUT(x2, TensorType({ DT_FLOAT }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT }))
.HIAI_OP_END()

/*
 * Converts a sparse representation into a dense tensor.
 * <Input>
 *    indices : A 0-D, 1-D, or 2-D Tensor of type int32.
 *    output_shape : A 1-D Tensor of type int32, must be a Const-OP.
 *    values : A 0-D or 1-D Tensor of type float32.
 *    default_value : A 0-D Tensor of type float32.
 * <Output>
 *    y : Tensor of shape output_shape.  Has the same type as values.
 * <Attr>
 *    validate_indices : Bool scalar, If True, indices are checked to make sure they are sorted in l
 *                       exicographic order and that there are no repeats. Now only supports False.
 * <Added in HiAI version>
 *    100.500.010.010
 */
HIAI_REG_OP(SparseToDense)
.HIAI_INPUT(indices, TensorType({ DT_INT32 }))
.HIAI_INPUT(output_shape, TensorType({ DT_INT32 }))
.HIAI_INPUT(values, TensorType({ DT_FLOAT }))
.HIAI_INPUT(default_value, TensorType({ DT_FLOAT }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT }))
.HIAI_ATTR(validate_indices, AttrValue::BOOL { false })
.HIAI_OP_END()
} // namespace hiai
// clang-format on

#endif