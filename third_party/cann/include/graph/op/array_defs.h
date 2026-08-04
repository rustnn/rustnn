/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2018-2023. All rights reserved.
 * Description: array_defs
 */
#ifndef INC_API_GRAPH_OP_ARRAY_DEFS_H
#define INC_API_GRAPH_OP_ARRAY_DEFS_H
#include "graph/operator_hiai_reg.h"

// clang-format off
namespace hiai {
/*
 * Data tensor
 * <Input>
 *    x : Input tensor
 * <Output>
 *    y : Output tensor
 * <Attr>
 *    index : Reserved. The index of input data in network, only support 0 currently.
 * <Added in HiAI version>
 *    100.300.010.011
 */
HIAI_REG_OP(Data)
.HIAI_INPUT(x, TensorType({ DT_FLOAT, DT_INT8, DT_UINT8, DT_INT32, DT_BOOL, DT_INT64, DT_DOUBLE }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT, DT_INT8, DT_UINT8, DT_INT32, DT_BOOL, DT_INT64, DT_DOUBLE }))
.HIAI_ATTR(index, AttrValue::INT { 0 })
.HIAI_OP_END()

/*
 * NetOutput tensor
 * <Input>
 *    x : Input tensor
 * <Output>
 *    y : Number of output tensor and number of input tensor are the same
 * <Attr>
 *    output_type : Specify output data type, DT_FLOAT(0), DT_UINT8(4), DT_INT8(2), DT_INT32(3), DT_FLOAT16(1).
 *                  The size of the output_type attribute must be the same as the number of output tensor.
 * <Added in HiAI version>
 *    100.520.020.100
 * <Examples>
 *    TensorDesc xDesc(Shape({4, 5, 6, 7}), FORMAT_NCHW, DT_FLOAT);
 *    hiai::op::Data x = hiai::op::Data("x");
 *    x.update_input_desc_x(xDesc);
 *
 *    auto activation = hiai::op::Activation("activation")
 *                  .set_input_x(x)
 *                  .set_attr_mode(0);
 *
 *    auto netoutput = hiai::op::NetOutput("netOutput")
 *                  .create_dynamic_input_x(1)
 *                  .set_input_x(activation)
 *                  .create_dynamic_output_y(1)
 *                  .set_attr_output_type({4});
 */
HIAI_REG_OP(NetOutput)
.HIAI_DYNAMIC_INPUT(x, TensorType({ DT_FLOAT, DT_INT8, DT_UINT8, DT_INT32 }))
.HIAI_DYNAMIC_OUTPUT(y, TensorType({ DT_FLOAT, DT_UINT8, DT_INT8, DT_INT32, DT_FLOAT16 }))
.HIAI_ATTR(output_type, AttrValue::LIST_INT ({}))
.HIAI_OP_END()

/*
 * Concatenates a list of tensors into a single tensor along one dimension.
 * The number of dimensions of the input tensors must match, and all dimensions except axis must be equal.
 * <Input>
 *    x : List of tensors for concatenation
 * <Output>
 *    y : Concatenated tensor
 * <Attr>
 *    concat_dim : Dimension along which to concatenate
 *    N : Number of inputs
 * <Added in HiAI version>
 *    100.300.010.011
 * <Examples>
 *    TensorDesc xDesc1(Shape({4, 5, 6, 7}), FORMAT_NCHW, DT_FLOAT);
 *    hiai::op::Data x1 = hiai::op::Data("x1");
 *    x1.update_input_desc_x(xDesc1);
 *
 *    TensorDesc xDesc2(Shape({4, 5, 6, 7}), FORMAT_NCHW, DT_FLOAT);
 *    hiai::op::Data x2 = hiai::op::Data("x2");
 *    x2.update_input_desc_x(xDesc2);
 *
 *    auto concatD = hiai::op::ConcatD("concatD")
 *                   .create_dynamic_input_x(2)
 *                   .set_dynamic_input_x(1, x1)
 *                   .set_dynamic_input_x(2, x2)
 *                   .set_attr_concat_dim(1)
 *                   .set_attr_N(2);
 */
HIAI_REG_OP(ConcatD)
.HIAI_DYNAMIC_INPUT(x, TensorType({ DT_FLOAT, DT_INT32, DT_UINT8, DT_INT8, DT_BOOL }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT, DT_INT32, DT_UINT8, DT_INT8, DT_BOOL }))
.HIAI_REQUIRED_ATTR(concat_dim, AttrValue::INT)
.HIAI_ATTR(N, AttrValue::INT { 1 })
.HIAI_OP_END()

/*
 * Fake-quantize the inputs tensor of type float via global float scalars.
 * <Input>
 *    x : A Tensor of type float32.
 *    min : 0D (scalar), A Tensor of type float32.
 *    max : 0D (scalar), A Tensor of type float32.
 * <Output>
 *    y : A Tensor of type float32.
 * <Attr>
 *    num_bits : An optional int. Defaults to 8. between 2 and 8, inclusive.
 *    narrow_range : An optional bool. Defaults to False.
 * <Added in HiAI version>
 *    100.500.010.011
 */
HIAI_REG_OP(FakeQuantWithMinMaxVars)
.HIAI_INPUT(x, TensorType({ DT_FLOAT }))
.HIAI_INPUT(min, TensorType({ DT_FLOAT }))
.HIAI_INPUT(max, TensorType({ DT_FLOAT }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT }))
.HIAI_ATTR(num_bits, AttrValue::INT { 8 })
.HIAI_ATTR(narrow_range, AttrValue::BOOL { false })
.HIAI_OP_END()

/*
 * Reshape the input tensor.
 * <Input>
 *    x : Input tensor, of the same type as Data
 *    shape : The shape to be resized, must be const op
 * <Output>
 *    y : Reshaped tensor that has the same values as Attr 'shape'
 * <Attr>
 *    axis : Dimension along which to reshape
 *    num_axes : Used to calculate the output shape
 *               When 'num_axes' is -1, output.size() = shape.size() + axis.
 *               When 'num_axes' is not -1, output.size() = shape.size() + tensor.size() - num_axes.
 * <Added in HiAI version>
 *    100.300.010.011
 */
HIAI_REG_OP(Reshape)
.HIAI_INPUT(x, TensorType({ DT_FLOAT, DT_INT32, DT_INT64, DT_BOOL }))
.HIAI_INPUT(shape, TensorType({ DT_INT32, DT_INT64 }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT, DT_INT32, DT_INT64, DT_BOOL }))
.HIAI_ATTR(axis, AttrValue::INT { 0 })
.HIAI_ATTR(num_axes, AttrValue::INT { -1 })
.HIAI_OP_END()

/*
 * Splits a tensor into a list of tensors along the split_dim.
 * <Input>
 *    x : Input tensor.
 * <Output>
 *    y : (Mandatory) splited tensors, whose number is specified by num_split
 * <Attr>
 *    split_dim : Which axis to split on, the value should be in range of [-RANK(X), RANK(X)).
 *    num_split : Number of outputs, x.shape[split_dim] % num_split should be 0.
 * <Added in HiAI version>
 *    100.300.010.011
 * <Examples>
 *    TensorDesc xDesc(Shape({4, 5, 6, 7}), FORMAT_NCHW, DT_FLOAT);
 *    hiai::op::Data x = hiai::op::Data("x");
 *    x.update_input_desc_x(xDesc);
 *
 *    auto splitD = hiai::op::SplitD("splitD")
 *                  .set_input_x(x)
 *                  .create_dynamic_output_y(3)
 *                  .set_attr_split_dim(2)
 *                  .set_attr_num_split(3);
 */
HIAI_REG_OP(SplitD)
.HIAI_INPUT(x, TensorType({ DT_FLOAT, DT_INT8, DT_INT32, DT_BOOL, DT_UINT8 }))
.HIAI_DYNAMIC_OUTPUT(y, TensorType({ DT_FLOAT, DT_INT8, DT_INT32, DT_BOOL, DT_UINT8 }))
.HIAI_REQUIRED_ATTR(split_dim, AttrValue::INT)
.HIAI_REQUIRED_ATTR(num_split, AttrValue::INT)
.HIAI_OP_END()

/*
 * Splits a tensor into 'num_split' tensors along one dimension.
 * <Input>
 *    x : Input tensor.
 *    size_splits : Tensor of type int32, must be const op.
 *                  Optional length of each output.
 *                  Sum of the values must be equal to the dim value at 'split_dim' specified.
 *    split_dim : Tensor of type int32, must be a scalar const op.
 *                Which axis to split on. A negative value means counting dimensions from the back.
 *                Accepted range is [-rank(x), rank(x)).
 * <Output>
 *    y : (Mandatory) spited tensors, whose number is specified by 'num_split'
 * <Attr>
 *    num_split : int (>= 1), number of outputs
 * <Added in HiAI version>
 *    100.310.010.013
 * <Examples>
 *    TensorDesc xDesc(Shape({4, 5, 6, 7}), FORMAT_NCHW, DT_FLOAT);
 *    hiai::op::Data x = hiai::op::Data("x");
 *    x.update_input_desc_x(xDesc);
 *
 *    TensorDesc sizeSplitsTensorDesc(Shape({3}), FORMAT_NCHW, DT_INT32);
 *    TensorPtr sizeSplitsTensor = std::make_shared<hiai::Tensor>(sizeSplitsTensorDesc);
 *    vector<int32_t> sizeSplitsDataValue = {1, 2, 3};
 *    sizeSplitsTensor->SetData((uint8_t*)sizeSplitsDataValue.data(), 3 * sizeof(int32_t));
 *    auto sizeSplits = hiai::op::Const("sizeSplits").set_attr_value(sizeSplitsTensor);
 *
 *    TensorDesc splitDimTensorDesc(Shape(), FORMAT_NCHW, DT_INT32);
 *    TensorPtr splitDimTensor = std::make_shared<hiai::Tensor>(splitDimTensorDesc);
 *    vector<int32_t> splitDimDataValue = {2};
 *    splitDimTensor->SetData((uint8_t*)splitDimDataValue.data(), 1 * sizeof(int32_t));
 *    auto splitDim = hiai::op::Const("splitDim").set_attr_value(splitDimTensor);
 *
 *    auto splitV = hiai::op::SplitV("splitV")
 *                  .set_input_x(x)
 *                  .set_input_size_splits(sizeSplits)
 *                  .set_input_split_dim(splitDim)
 *                  .set_attr_num_split(3)
 *                  .create_dynamic_output_y(3);
 */
HIAI_REG_OP(SplitV)
.HIAI_INPUT(x, TensorType({ DT_FLOAT }))
.HIAI_INPUT(size_splits, TensorType({ DT_INT32 }))
.HIAI_INPUT(split_dim, TensorType({ DT_INT32 }))
.HIAI_DYNAMIC_OUTPUT(y, TensorType({ DT_FLOAT }))
.HIAI_REQUIRED_ATTR(num_split, AttrValue::INT)
.HIAI_OP_END()

/*
 * Unpacks the given dimension of a rank-R tensor into rank-(R-1) tensors.
 * <Input>
 *    x : Input tensor
 * <Output>
 *    y : List of Tensor objects unstacked from x
 * <Attr>
 *    num : Output number of tensor after split.
 *          The value of num must be equal to x[axis].
 *    axis : Axis to unpack along
 * <Added in HiAI version>
 *    100.310.010.013
 * <Examples>
 *    TensorDesc xDesc(Shape({4, 5, 6, 7}), FORMAT_NCHW, DT_FLOAT);
 *    hiai::op::Data x = hiai::op::Data("x");
 *    x.update_input_desc_x(xDesc);
 *
 *    auto unpack = hiai::op::Unpack("unpack")
 *                  .set_input_x(x)
 *                  .set_attr_axis(2)
 *                  .set_attr_num(6)
 *                  .create_dynamic_output_y(6);
 */
HIAI_REG_OP(Unpack)
.HIAI_INPUT(x, TensorType({ DT_FLOAT, DT_INT32 }))
.HIAI_DYNAMIC_OUTPUT(y, TensorType({ DT_FLOAT, DT_INT32 }))
.HIAI_REQUIRED_ATTR(num, AttrValue::INT)
.HIAI_ATTR(axis, AttrValue::INT { 0 })
.HIAI_OP_END()

/*
 * Flattens the input tensor into a 2D matrix.
 * <Input>
 *    x : Tensor of rank
 * <Output>
 *    y : 2D tensor with the contents of the input tensor, with input dimensions up to axis flattened to
 *        the outer dimension of the output and remaining input dimensions flattened into the inner dimension
 *        of the output.
 * <Added in HiAI version>
 *    100.300.010.011
 */
HIAI_REG_OP(Flatten)
.HIAI_INPUT(x, TensorType({ DT_FLOAT }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT }))
.HIAI_OP_END()

/*
 * Flattens the input tensor into a 2D matrix.
 * <Input>
 *    x : Tensor of rank
 * <Output>
 *    y : 2D tensor with the contents of the input tensor, with input dimensions up to axis flattened to
 *        the outer dimension of the output and remaining input dimensions flattened into the inner dimension
 *        of the output.
 * <Attr>
 *    axis : An optional int32, specifying the first axis to flatten.
 *           All preceding axes are retained in the output. Defaults to "1".
 *    end_axis : An optional int32, specifying the last axis to flatten.
 *               All following axes are retained in the output. Defaults to "-1".
 * <Added in HiAI version>
 *    100.500.010.010
 */
HIAI_REG_OP(FlattenV2)
.HIAI_INPUT(x, TensorType({ DT_FLOAT }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT }))
.HIAI_ATTR(axis, AttrValue::INT { 1 })
.HIAI_ATTR(end_axis, AttrValue::INT { -1 })
.HIAI_OP_END()

/*
 * Extracts a slice of size 'size' from a tensor input starting at the location specified by 'offsets '.
 * The slice 'size' is represented as a tensor shape,
 * where size[i] is the number of elements of the ith dimension to slice.
 * The starting location (offsets) for the slice is represented as an offset in each dimension of input.
 * In other words, offsets[i] is the offset into the ith dimension of input to slice.
 * <Input>
 *    x : Input tensor
 *    offsets : Const, starting location for each dimension of input
 *    size : Const, number of elements for each dimension of input
 * <Output>
 *    y : Output tensor
 * <Added in HiAI version>
 *    100.310.010.013
 * <Examples>
 *    TensorDesc xDesc(Shape({4, 5, 6, 7}), FORMAT_NCHW, DT_FLOAT);
 *    hiai::op::Data x = hiai::op::Data("x");
 *    x.update_input_desc_x(xDesc);
 *
 *    TensorDesc offsetsTensorDesc(Shape({4}), FORMAT_NCHW, DT_INT32);
 *    TensorPtr offsetsTensor = std::make_shared<hiai::Tensor>(offsetsTensorDesc);
 *    vector<int32_t>offsetsDataValue = {0, 1, 0, 1};
 *    offsetsTensor->SetData((uint8_t*)offsetsDataValue.data(), 4 * sizeof(int32_t));
 *    auto offsets = hiai::op::Const("offsets").set_attr_value(offsetsTensor);
 *
 *    TensorDesc sizeTensorDesc(Shape({4}), FORMAT_NCHW, DT_INT32);
 *    TensorPtr sizeTensor = std::make_shared<hiai::Tensor>(sizeTensorDesc);
 *    vector<int32_t> sizeDataValue = {1, 3, 2, -1};
 *    sizeTensor->SetData((uint8_t*)sizeDataValue.data(), 4 * sizeof(int32_t));
 *    auto size = hiai::op::Const("size").set_attr_value(sizeTensor);
 *
 *    auto slice = hiai::op::Slice("slice")
 *                 .set_input_x(x)
 *                 .set_input_offsets(offsets)
 *                 .set_input_size(size);
 */
HIAI_REG_OP(Slice)
.HIAI_INPUT(x, TensorType({ DT_FLOAT, DT_INT32, DT_INT64, DT_UINT8, DT_BOOL }))
.HIAI_INPUT(offsets, TensorType({ DT_INT32 }))
.HIAI_INPUT(size, TensorType({ DT_INT32 }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT, DT_INT32, DT_UINT8, DT_BOOL }))
.HIAI_OP_END()

/*
 * Inserts a dimension of 1 into a tensor's shape.
 * This op support max realdims is 4 and support max padding dims is 3.
 * <Input>
 *    x : Input tensor
 *    axis : one element const, dimension index at which to expand the shape of input
 * <Output>
 *    y : Output tensor. The dimension of y not support bigger than 4.
 * <Added in HiAI version>
 *    100.310.010.013
 */
HIAI_REG_OP(ExpandDims)
.HIAI_INPUT(x, TensorType({ DT_FLOAT, DT_INT32, DT_UINT8, DT_BOOL }))
.HIAI_INPUT(axis, TensorType({ DT_INT32 }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT, DT_INT32, DT_UINT8, DT_BOOL }))
.HIAI_OP_END()

/*
 * Gathers slices from 'x' axis according to 'indices'.
 * <Input>
 *    x : Tensor from which to gather values
 *    indices : Input tensor
 * <Output>
 *    y : Output tensor
 * <Attr>
 *    axis : which axis to gather values. Accepted range is [-rank(x), rank(x)).
 * <Added in HiAI version>
 *    100.310.010.013
 * <Examples>
 *    TensorDesc xDesc(Shape({4, 5, 6, 7}), FORMAT_NCHW, DT_FLOAT);
 *    hiai::op::Data x = hiai::op::Data("x");
 *    x.update_input_desc_x(xDesc);
 *
 *    TensorDesc indicesDesc(Shape({4}), FORMAT_NCHW, DT_INT32);
 *    hiai::op::Data indices = hiai::op::Data("indices");
 *    indices.update_input_desc_x(indicesDesc);
 *
 *    auto gatherV2D = hiai::op::GatherV2D("gatherV2D")
 *                     .set_input_x(x)
 *                     .set_input_indices(indices)
 *                     .set_attr_axis(1);
 */
HIAI_REG_OP(GatherV2D)
.HIAI_INPUT(x, TensorType({ DT_FLOAT, DT_INT32 }))
.HIAI_INPUT(indices, TensorType({ DT_INT32 }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT, DT_INT32 }))
.HIAI_REQUIRED_ATTR(axis, AttrValue::INT)
.HIAI_OP_END()

/*
 * Gathers slices from 'x' into a Tensor with shape specified by 'indices'.
 * <Input>
 *    x : Tensor from which to gather values.
 *    indices : Input tensor.
 * <Output>
 *    y : Output tensor
 * <Added in HiAI version>
 *    100.310.010.013
 * <Examples>
 *    TensorDesc xDesc(Shape({4, 5, 6, 7}), FORMAT_NCHW, DT_FLOAT);
 *    hiai::op::Data x = hiai::op::Data("x");
 *    x.update_input_desc_x(xDesc);
 *
 *    TensorDesc indicesDesc(Shape({3}), FORMAT_NCHW, DT_FLOAT);
 *    hiai::op::Data indices = hiai::op::Data("indices");
 *    indices.update_input_desc_x(indicesDesc);
 *
 *    auto gatherNd = hiai::op::GatherNd("gatherNd")
 *                    .set_input_x(x)
 *                    .set_input_indices(indices);
 */
HIAI_REG_OP(GatherNd)
.HIAI_INPUT(x, TensorType({ DT_FLOAT, DT_INT32 }))
.HIAI_INPUT(indices, TensorType({ DT_FLOAT, DT_INT32 }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT, DT_INT32 }))
.HIAI_OP_END()

/*
 * Stacks a list of rank-R tensors into one rank-(R+1) tensor.
 * <Input>
 *    x : List of Tensor objects with the same shape and type
 * <Output>
 *    y : Output tensor. The dimension of y not support bigger than 4.
 * <Attr>
 *    axis : Axis to stack along
 *    N : Number of values
 * <Added in HiAI version>
 *    100.310.010.013
 * <Examples>
 *    TensorDesc xDesc1(Shape({4, 5, 6}), FORMAT_NCHW, DT_FLOAT);
 *    hiai::op::Data x1 = hiai::op::Data("x1");
 *    x1.update_input_desc_x(xDesc1);
 *
 *    TensorDesc xDesc2(Shape({4, 5, 6}), FORMAT_NCHW, DT_FLOAT);
 *    hiai::op::Data x2 = hiai::op::Data("x2");
 *    x2.update_input_desc_x(xDesc2);
 *
 *    auto pack = hiai::op::Pack("pack")
 *                .create_dynamic_input_x(2)
 *                .set_dynamic_input_x(1, x1)
 *                .set_dynamic_input_x(2, x2)
 *                .set_attr_N(2)
 *                .set_attr_axis(1);
 */
HIAI_REG_OP(Pack)
.HIAI_DYNAMIC_INPUT(x, TensorType({ DT_FLOAT, DT_INT32 }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT, DT_INT32 }))
.HIAI_ATTR(axis, AttrValue::INT { 0 })
.HIAI_REQUIRED_ATTR(N, AttrValue::INT)
.HIAI_OP_END()

/*
 * Rearranges blocks of spatial data, into depth. More specifically, this op outputs a copy of
 * the input tensor where values from the height and width dimensions are moved to the depth dimension.
 * The attr block_size indicates the input block size.
 * <Input>
 *    x : Input tensor with shape [n, c, h, w] or [n, h, w, c], where h and w must be divisible by 'block_size'
 * <Output>
 *    y : Output tensor
 * <Attr>
 *    block_size : int (>= 1), size of the spatial block
 *    data_format : Format of input, 'NCHW' or 'NHWC'. Default is 'NHWC'
 * <Added in HiAI version>
 *    100.310.010.013
 */
HIAI_REG_OP(SpaceToDepth)
.HIAI_INPUT(x, TensorType({ DT_FLOAT, DT_UINT8, DT_INT8 }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT, DT_UINT8, DT_INT8 }))
.HIAI_REQUIRED_ATTR(block_size, AttrValue::INT)
.HIAI_ATTR(data_format, AttrValue::STR { "NHWC" })
.HIAI_OP_END()

/*
 * Rearranges data from depth into blocks of spatial data.This is the reverse transformation of SpaceToDepth.
 * More specifically, this op outputs a copy of the input tensor where values from the depth dimension are moved
 * in spatial blocks to the height and width dimensions.
 * <Input>
 *    x : The input tensor with shape [ batch, height, width, channels ], channels
 *        must be divided by block_size*block_size.
 * <Output>
 *    y : The output tensor
 * <Attr>
 *    block_size : The size of the spatial block, must be greater than 0.
 *    mode : DCR (default) for depth-column-row order re-arrangement.
 *           CRD for column-row-depth order.
 *    data_format : Format of operator, 'NCHW' or 'NHWC'. Default is 'NHWC'
 * <Added in HiAI version>
 *    100.320.010.010
 */
HIAI_REG_OP(DepthToSpace)
.HIAI_INPUT(x, TensorType({ DT_FLOAT }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT }))
.HIAI_REQUIRED_ATTR(block_size, AttrValue::INT)
.HIAI_ATTR(mode, AttrValue::STR { "DCR" })
.HIAI_ATTR(data_format, AttrValue::STR { "NHWC" })
.HIAI_OP_END()

/*
 * Extracts a strided slice of a tensor
 * <Input>
 *    x : Input tensor
 *    begin : An int32 tensor, must be const op
 *    end : An int32 tensor, must be const op
 *    strides : An int32 tensor, must be const op
 * <Output>
 *    y : Output tensor
 * <Attr>
 *    begin_mask : An int32 mask, defaults to 0.
 *    end_mask : An int32 mask, defaults to 0.
 *    ellipsis_mask : An int32 mask, defaults to 0.
 *    new_axis_mask : An int32 mask, defaults to 0.
 *    shrink_axis_mask : Defaults to 0.
 * <Added in HiAI version>
 *    100.310.010.013
 * <Examples>
 *    TensorDesc xDesc(Shape({5, 6, 7, 8}), FORMAT_NCHW, DT_FLOAT);
 *    hiai::op::Data x = hiai::op::Data("x");
 *    x.update_input_desc_x(xDesc);
 *
 *    TensorDesc beginTensorDesc(Shape({4}), FORMAT_NCHW, DT_INT32);
 *    TensorPtr beginTensor = std::make_shared<hiai::Tensor>(beginTensorDesc);
 *    vector<int32_t> beginDataValue = {1, 3, 3, 5};
 *    beginTensor->SetData((uint8_t*)beginDataValue.data(), 4 * sizeof(int32_t));
 *    auto begin = hiai::op::Const("begin").set_attr_value(beginTensor);
 *
 *    TensorDesc endTensorDesc(Shape({4}), FORMAT_NCHW, DT_INT32);
 *    TensorPtr endTensor = std::make_shared<hiai::Tensor>(endTensorDesc);
 *    vector<int32_t> endDataValue = {-2, 7, 0, 6};
 *    endTensor->SetData((uint8_t*)endDataValue.data(), 4 * sizeof(int32_t));
 *    auto end = hiai::op::Const("end").set_attr_value(endTensor);
 *
 *    TensorDesc stridesTensorDesc(Shape({4}), FORMAT_NCHW, DT_INT32);
 *    TensorPtr stridesTensor = std::make_shared<hiai::Tensor>(stridesTensorDesc);
 *    vector<int32_t> stridesDataValue = {2, 2, 2, 2};
 *    stridesTensor->SetData((uint8_t*)stridesDataValue.data(), 4 * sizeof(int32_t));
 *    auto strides = hiai::op::Const("strides").set_attr_value(stridesTensor);
 *
 *    auto stridedSlice = hiai::op::StridedSlice("stridedSlice")
 *                        .set_input_x(x)
 *                        .set_input_begin(begin)
 *                        .set_input_end(end)
 *                        .set_input_strides(strides)
 *                        .set_attr_begin_mask(1)
 *                        .set_attr_end_mask(15)
 *                        .set_attr_ellipsis_mask(0)
 *                        .set_attr_new_axis_mask(0)
 *                        .set_attr_shrink_axis_mask(0);
 */
HIAI_REG_OP(StridedSlice)
.HIAI_INPUT(x, TensorType({ DT_FLOAT, DT_INT32, DT_UINT8, DT_BOOL }))
.HIAI_INPUT(begin, TensorType({ DT_INT32 }))
.HIAI_INPUT(end, TensorType({ DT_INT32 }))
.HIAI_INPUT(strides, TensorType({ DT_INT32 }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT, DT_INT32, DT_UINT8, DT_BOOL }))
.HIAI_ATTR(begin_mask, AttrValue::INT { 0 })
.HIAI_ATTR(end_mask, AttrValue::INT { 0 })
.HIAI_ATTR(ellipsis_mask, AttrValue::INT { 0 })
.HIAI_ATTR(new_axis_mask, AttrValue::INT { 0 })
.HIAI_ATTR(shrink_axis_mask, AttrValue::INT { 0 })
.HIAI_OP_END()

/*
 * Extracts a strided slice of a tensor.
 * <Input>
 *    x : Tensor of data to extract slices from.
 *    begin : 1-D tensor of starting indices of corresponding axis in `axes`. Only support Const op.
 *    end : 1-D tensor of ending indices (exclusive) of corresponding axis in `axes`. Only support Const op.
 *    axes : 1-D tensor of axes that `starts` and `ends` apply to. Only support Const op.
 *    strides : 1-D tensor of slice step of corresponding axis in `axes`.. Only support Const op.
 * <Output>
 *    y : Output tensor
 * <Attr>
 *    begin_mask : reserved, default 0
 *    end_mask : reserved, default 0
 *    ellipsis_mask : reserved, default 0
 *    new_axis_mask : reserved, default 0
 *    shrink_axis_mask : reserved, default 0
 * <Added in HiAI version>
 *    100.500.010.010
 * <Examples>
 *    TensorDesc xDesc(Shape({4, 5, 6, 7}), FORMAT_NCHW, DT_FLOAT);
 *    hiai::op::Data x = hiai::op::Data("x");
 *    x.update_input_desc_x(xDesc);
 *
 *    TensorDesc beginTensorDesc(Shape({4}), FORMAT_NCHW, DT_INT32);
 *    TensorPtr beginTensor = std::make_shared<hiai::Tensor>(beginTensorDesc);
 *    vector<int32_t>offsetsDataValue = {0, 1, 0, 1};
 *    beginTensor->SetData((uint8_t*)offsetsDataValue.data(), 4 * sizeof(int32_t));
 *    auto offsets = hiai::op::Const("offsets").set_attr_value(beginTensor);
 *
 *    TensorDesc endTensorDesc(Shape({4}), FORMAT_NCHW, DT_INT32);
 *    TensorPtr endTensor = std::make_shared<hiai::Tensor>(endTensorDesc);
 *    vector<int32_t> sizeDataValue = {1, 3, 2, -1};
 *    endTensor->SetData((uint8_t*)sizeDataValue.data(), 4 * sizeof(int32_t));
 *    auto size = hiai::op::Const("size").set_attr_value(endTensor);
 *
 *    auto StridedSliceV2 = hiai::op::StridedSliceV2("StridedSliceV2")
 *                         .set_input_x(x)
 *                         .set_input_offsets(begin)
 *                         .set_input_size(end);
 *                         .set_attr_begin_mask(0)
 *                         .set_attr_end_mask(0)
 *                         .set_attr_ellipsis_mask(0)
 *                         .set_attr_new_axis_mask(0)
 *                         .set_attr_shrink_axis_mask(0);
 */
HIAI_REG_OP(StridedSliceV2)
.HIAI_INPUT(x, TensorType({ DT_FLOAT, DT_INT32, DT_UINT8, DT_BOOL }))
.HIAI_INPUT(begin, TensorType({ DT_INT32 }))
.HIAI_INPUT(end, TensorType({ DT_INT32 }))
.HIAI_OPTIONAL_INPUT(axes, TensorType({ DT_INT32 }))
.HIAI_OPTIONAL_INPUT(strides, TensorType({ DT_INT32 }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT, DT_INT32, DT_UINT8, DT_BOOL }))
.HIAI_ATTR(begin_mask, AttrValue::INT { 0 })
.HIAI_ATTR(end_mask, AttrValue::INT { 0 })
.HIAI_ATTR(ellipsis_mask, AttrValue::INT { 0 })
.HIAI_ATTR(new_axis_mask, AttrValue::INT { 0 })
.HIAI_ATTR(shrink_axis_mask, AttrValue::INT { 0 })
.HIAI_OP_END()

/*
 * Zero-pads and then rearranges blocks of spatial data into batch.The operation outputs
 * a copy of the input tensor, in which the values from the height and width dimensions
 * are moved to the batch dimension.
 * <Input>
 *    x : 4-D tensor with shape [batch, depth, height, width]. Both height and width must be divisible
 *        by 'block_shape'.
 *    block_shape : Const OP, 1-D with shape [M], where all values must be >= 1
 *    paddings : Const OP, 2-D with shape [M, 2], where, all values must be >= 0
 *               paddings = [[pad_top, pad_bottom], [pad_left, pad_right]]
 *               The effective spatial dimensions of the zero-padded input tensor will be:
 *               height_pad = pad_top + height + pad_bottom
 *               width_pad = pad_left + width + pad_right
 *               Both 'height_pad' and 'width_pad' must be divisible by 'block_size'.
 * <Output>
 *    y : Tensor of the same type as 'x'
 * <Added in HiAI version>
 *    100.310.010.013
 * <Examples>
 *    TensorDesc xDesc(Shape({5, 6, 7, 8}), FORMAT_NCHW, DT_FLOAT);
 *    hiai::op::Data x = hiai::op::Data("x");
 *    x.update_input_desc_x(xDesc);
 *
 *    TensorDesc blockShapeTensorDesc(Shape({2}), FORMAT_NCHW, DT_INT32);
 *    TensorPtr blockShapeTensor = std::make_shared<hiai::Tensor>(blockShapeTensorDesc);
 *    vector<int32_t> blockShapeDataValue = {1, 2};
 *    blockShapeTensor->SetData((uint8_t*)blockShapeDataValue.data(), 2 * sizeof(int32_t));
 *    auto blockShape = hiai::op::Const("blockShape").set_attr_value(blockShapeTensor);
 *
 *    TensorDesc paddingsTensorDesc(Shape({2, 2}), FORMAT_NCHW, DT_INT32);
 *    TensorPtr paddingsTensor = std::make_shared<hiai::Tensor>(paddingsTensorDesc);
 *    vector<int32_t> paddingsDataValue = {1, 2, 2, 4};
 *    paddingsTensor->SetData((uint8_t*)paddingsDataValue.data(), 4 * sizeof(int32_t));
 *    auto paddings = hiai::op::Const("paddings").set_attr_value(paddingsTensor);
 *
 *    auto spaceToBatchND = hiai::op::SpaceToBatchND("spaceToBatchND")
 *                          .set_input_x(x)
 *                          .set_input_block_shape(blockShape)
 *                          .set_input_paddings(paddings);
 */
HIAI_REG_OP(SpaceToBatchND)
.HIAI_INPUT(x, TensorType({ DT_FLOAT }))
.HIAI_INPUT(block_shape, TensorType({ DT_INT32 }))
.HIAI_INPUT(paddings, TensorType({ DT_INT32 }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT }))
.HIAI_OP_END()

/*
 * This operation reshapes the "batch" dimension 0 into height and width that divided by block.
 * Interleaves these blocks back into the grid defined by the spatial dimensions [batch, depth, height, width],
 * to obtain a result with the same rank as the input. The spatial dimensions of this intermediate result
 * are then optionally cropped according to crops to produce the output.
 * <Input>
 *    x : 4-D tensor with shape [batch, depth, height, width]. It is required that the elements of x.dimension[0]
 *        must be divided by block_shape.dimension[0] * block_shape.dimension[1].
 *    block_shape : Const OP, 1-D with shape [M], where all values must be >= 1
 *    crops : Const OP, 2-D with shape [M, 2], where all values must be >= 0. crops[i] = [crop_start, crop_end]
 *            specifies the amount to crop from input dimension i + 1, which corresponds to spatial dimension i.
 *            It is required that crop_start[i] + crop_end[i] <= block_shape[i] * input_shape[i + 1].
 * <Output>
 *    y : Tensor of the same type as 'x'
 * <Added in HiAI version>
 *    100.310.010.013
 * <Examples>
 *    TensorDesc xDesc(Shape({4, 6, 7, 8}), FORMAT_NCHW, DT_FLOAT);
 *    hiai::op::Data x = hiai::op::Data("x");
 *    x.update_input_desc_x(xDesc);
 *
 *    TensorDesc blockShapeTensorDesc(Shape({2}), FORMAT_NCHW, DT_INT32);
 *    TensorPtr blockShapeTensor = std::make_shared<hiai::Tensor>(blockShapeTensorDesc);
 *    vector<int32_t> blockShapeDataValue = {1, 2};
 *    blockShapeTensor->SetData((uint8_t*)blockShapeDataValue.data(), 2 * sizeof(int32_t));
 *    auto blockShape = hiai::op::Const("blockShape").set_attr_value(blockShapeTensor);
 *
 *    TensorDesc cropsTensorDesc(Shape({2, 2}), FORMAT_NCHW, DT_INT32);
 *    TensorPtr cropsTensor = std::make_shared<hiai::Tensor>(cropsTensorDesc);
 *    vector<int32_t> cropsDataValue = {1, 2, 2, 4};
 *    cropsTensor->SetData((uint8_t*)cropsDataValue.data(), 4 * sizeof(int32_t));
 *    auto crops = hiai::op::Const("crops").set_attr_value(cropsTensor);
 *
 *    auto batchToSpaceND = hiai::op::BatchToSpaceND("batchToSpaceND")
 *                          .set_input_x(x)
 *                          .set_input_block_shape(blockShape)
 *                          .set_input_crops(crops);
 */
HIAI_REG_OP(BatchToSpaceND)
.HIAI_INPUT(x, TensorType({ DT_FLOAT }))
.HIAI_INPUT(block_shape, TensorType({ DT_INT32 }))
.HIAI_INPUT(crops, TensorType({ DT_INT32 }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT }))
.HIAI_OP_END()

/*
 * Constructs a tensor by tiling a given tensor.
 * <Input>
 *    x : Input tensor
 *    multiples : Const OP
 * <Output>
 *    y : Output tensor
 * <Added in HiAI version>
 *    100.310.010.013
 */
HIAI_REG_OP(Tile)
.HIAI_INPUT(x, TensorType({ DT_FLOAT, DT_INT32, DT_UINT8, DT_BOOL }))
.HIAI_INPUT(multiples, TensorType({ DT_INT32 }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT, DT_INT32, DT_UINT8, DT_BOOL }))
.HIAI_OP_END()

/*
 * Returns the size of a tensor.
 * <Input>
 *    x : Input tensor
 * <Output>
 *    y : Output tensor
 * <Attr>
 *    dtype : Reserved. data type of the output y.
 * <Added in HiAI version>
 *    100.310.010.013
 * <Examples>
 *    TensorDesc xDesc(Shape({1, 4, 1, 1}), FORMAT_NCHW, DT_FLOAT);
 *    hiai::op::Data x = hiai::op::Data("x");
 *    x.update_input_desc_x(xDesc);
 *
 *    auto size = hiai::op::Size("size")
 *               .set_input_x(x);
 *
 *    auto cast = hiai::op::CastT("cast")
 *               .set_input_x(size)
 *               .set_attr_src_dtype(3)
 *               .set_attr_dst_dtype(1);
 *
 *    auto add = hiai::op::Add("add")
 *               .set_input_x1(x)
 *               .set_input_x2(cast);
 */
HIAI_REG_OP(Size)
.HIAI_INPUT(x, TensorType({ DT_FLOAT, DT_INT32, DT_UINT8, DT_BOOL }))
.HIAI_OUTPUT(y, TensorType({ DT_INT32 }))
.HIAI_ATTR(dtype, AttrValue::INT { DT_INT32 })
.HIAI_OP_END()

/*
 * Creates a tensor of shape 'dims' and fills it with 'value'.
 * <Input>
 *    dims : Const OP, 1-D, shape of the output tensor
 *    value : 0-D, value to fill the returned tensor
 * <Output>
 *    y : Output tensor
 * <Added in HiAI version>
 *    100.310.010.013
 */
HIAI_REG_OP(Fill)
.HIAI_INPUT(dims, TensorType({ DT_INT32 }))
.HIAI_INPUT(value, TensorType({ DT_FLOAT, DT_BOOL, DT_INT32, DT_UINT8 }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT, DT_BOOL, DT_INT32, DT_UINT8 }))
.HIAI_OP_END()

/*
 * Performs tensor select.
 * <Input>
 *    condition : the condition of select. Only support first dimension broadcast.
 *    x1 : First operand.
 *    x2 : Second operand, has the same shape of x1.
 * <Output>
 *    y : Result, has same element type as two inputs.
 * <Added in HiAI version>
 *    100.320.010.010
 */
HIAI_REG_OP(Select)
.HIAI_INPUT(condition, TensorType({ DT_BOOL }))
.HIAI_INPUT(x1, TensorType({ DT_FLOAT, DT_INT32, DT_UINT8, DT_BOOL }))
.HIAI_INPUT(x2, TensorType({ DT_FLOAT, DT_INT32, DT_UINT8, DT_BOOL }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT, DT_INT32, DT_UINT8, DT_BOOL }))
.HIAI_OP_END()

/*
 * The operation pads input according to the paddings and constant_values.
 * <Input>
 *    x : The input tensor.
 *    paddings : The values of paddings, as a role of dimensions to be added on the input tensor x,
 *               must be a Const-OP.
 *    constant_values : A tensor of the same type as x, that indicates the value to use for padding input,
 *                      must be a Const-OP.
 * <Output>
 *    y : The output tensor.
 * <Added in HiAI version>
 *    100.320.010.010
 * <Examples>
 *    TensorDesc xDesc(Shape({4, 5, 6, 7}), FORMAT_NCHW, DT_FLOAT);
 *    hiai::op::Data x = hiai::op::Data("x");
 *    x.update_input_desc_x(xDesc);
 *
 *    TensorDesc biasTensorDesc(Shape({4, 2}), FORMAT_NCHW, DT_INT32);
 *    TensorPtr biasTensor = std::make_shared<hiai::Tensor>(biasTensorDesc);
 *    vector<int32_t> dataValue = {0, 1, 2, 3, 4, 5, 6, 7};
 *    biasTensor->SetData((uint8_t*)dataValue.data(), 8 * sizeof(int32_t));
 *    auto paddings = hiai::op::Const("paddings").set_attr_value(biasTensor);
 *
 *    TensorDesc constantValuesTensorDesc(Shape({1}), FORMAT_NCHW, DT_FLOAT);
 *    TensorPtr constantValuesTensor = std::make_shared<hiai::Tensor>(constantValuesTensorDesc);
 *    vector<float> constantValuesDataValue = {1.0};
 *    constantValuesTensor->SetData((uint8_t*)constantValuesDataValue.data(), 1 * sizeof(float));
 *    auto constantValues = hiai::op::Const("constantValues").set_attr_value(constantValuesTensor);
 *
 *    auto padV2 = hiai::op::PadV2("PadV2")
 *                 .set_input_x(x)
 *                 .set_input_paddings(paddings)
 *                 .set_input_constant_values(constantValues);
 */
HIAI_REG_OP(PadV2)
.HIAI_INPUT(x, TensorType({ DT_FLOAT, DT_INT32 }))
.HIAI_INPUT(paddings, TensorType({ DT_INT32 }))
.HIAI_INPUT(constant_values, TensorType({ DT_FLOAT, DT_INT32 }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT, DT_INT32 }))
.HIAI_OP_END()

/*
 * Given a tensor input, this operation returns a tensor of the same type with all dimensions of size 1 removed.
 * If you do not want to remove all size with 1 dimensions,
 * you can remove specific size 1 dimensions by specifying axis.
 * <Input>
 *    x : The input to be squeezed.
 * <Output>
 *    y : The output tensor.
 * <Attr>
 *    axis : The dimensions which to remove.
 * <Added in HiAI version>
 *    100.320.010.010
 */
HIAI_REG_OP(Squeeze)
.HIAI_INPUT(x, TensorType({ DT_FLOAT, DT_INT32, DT_BOOL, DT_UINT8 }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT, DT_INT32, DT_BOOL, DT_UINT8 }))
.HIAI_ATTR(axis, AttrValue::LIST_INT ({}))
.HIAI_OP_END()

/*
 * Pads a tensor.
 * <Input>
 *    x : The input tensor
 *    paddings : The values of padding, as a role of dimensions to be added on the input tensor x,
 *               with shape [D, 2], D is the rank of x.
 * <Output>
 *    y : The output tensor
 * <Added in HiAI version>
 *    100.320.010.010
 * <Examples>
 *    TensorDesc xDesc(Shape({4, 5, 6, 7}), FORMAT_NCHW, DT_FLOAT);
 *    hiai::op::Data x = hiai::op::Data("x");
 *    x.update_input_desc_x(xDesc);
 *
 *    TensorDesc biasTensorDesc(Shape({4, 2}), FORMAT_NCHW, DT_INT32);
 *    TensorPtr biasTensor = std::make_shared<hiai::Tensor>(biasTensorDesc);
 *    vector<int32_t> dataValue = {0, 1, 2, 3, 4, 5, 6, 7};
 *    biasTensor->SetData((uint8_t*)dataValue.data(), 8 * sizeof(int32_t));
 *    auto paddings = hiai::op::Const("paddings").set_attr_value(biasTensor);
 *
 *    auto pad = hiai::op::Pad("Pad")
 *               .set_input_x(x)
 *               .set_input_paddings(paddings);
 */
HIAI_REG_OP(Pad)
.HIAI_INPUT(x, TensorType({ DT_FLOAT, DT_INT32 }))
.HIAI_INPUT(paddings, TensorType({ DT_INT32 }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT, DT_INT32 }))
.HIAI_OP_END()

/*
 * Pads a tensor with mirrored values.
 * <Input>
 *    x : The input tensor
 *    paddings : The values of paddings, as a role of dimensions to be added on the input tensor x,
 *               must be a Const-OP.
 * <Output>
 *    y : The output tensor
 * <Attr>
 *    mode : REFLECT or SYMMETRIC.
 *           SYMMETRIC  paddings must be no greater than the x dimension size
 *           REFLECT  paddings must be less than the x dimension size
 * <Added in HiAI version>
 *    100.320.010.010
 * <Examples>
 *    TensorDesc xDesc(Shape({4, 5, 6, 7}), FORMAT_NCHW, DT_FLOAT);
 *    hiai::op::Data x = hiai::op::Data("x");
 *    x.update_input_desc_x(xDesc);
 *
 *    TensorDesc biasTensorDesc(Shape({4, 2}), FORMAT_NCHW, DT_INT32);
 *    TensorPtr biasTensor = std::make_shared<hiai::Tensor>(biasTensorDesc);
 *    vector<int32_t> dataValue = {0, 1, 2, 3, 4, 5, 6, 7};
 *    biasTensor->SetData((uint8_t*)dataValue.data(), 8 * sizeof(int32_t));
 *    auto paddings = hiai::op::Const("paddings").set_attr_value(biasTensor);
 *
 *    auto mirrorPad = hiai::op::MirrorPad("mirrorPad")
 *                    .set_input_x(x)
 *                    .set_input_paddings(paddings)
 *                    .set_attr_mode("SYMMETRIC");
 */
HIAI_REG_OP(MirrorPad)
.HIAI_INPUT(x, TensorType({ DT_FLOAT, DT_INT32, DT_BOOL, DT_INT64 }))
.HIAI_INPUT(paddings, TensorType({ DT_INT32, DT_INT64 }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT, DT_INT32, DT_BOOL, DT_INT64 }))
.HIAI_REQUIRED_ATTR(mode, AttrValue::STR)
.HIAI_OP_END()

/*
 * Return a one-hot tensor
 * <Input>
 *    x : A index tensor.
 *    depth : A scalar defining the depth of the one hot dimension.
 *    on_value : A scalar defining the value to fill in output when x[j] = i. (default: 1).
 *    off_value : A scalar defining the value to fill in output when x[j] != i. (default: 0).
 * <Output>
 *    y : Result, has element type as T. The dimension of y not support bigger than 4.
 * <Attr>
 *    axis : The axis to fill.(default: -1)
 * <Added in HiAI version>
 *    100.320.010.010
 * <Examples>
 *    TensorDesc xDesc(Shape({4}), FORMAT_NCHW, DT_INT32);
 *    hiai::op::Data x = hiai::op::Data("x");
 *    x.update_input_desc_x(xDesc);
 *
 *    TensorDesc depthTensorDesc(Shape(), FORMAT_NCHW, DT_INT32);
 *    TensorPtr depthTensor = std::make_shared<hiai::Tensor>(depthTensorDesc);
 *    vector<int32_t> depthDataValue = {1};
 *    depthTensor->SetData((uint8_t*)depthDataValue.data(), 1 * sizeof(int32_t));
 *    auto depth = hiai::op::Const("depth").set_attr_value(depthTensor);
 *
 *    TensorDesc onValueTensorDesc(Shape(), FORMAT_NCHW, DT_FLOAT);
 *    TensorPtr onValueTensor = std::make_shared<hiai::Tensor>(onValueTensorDesc);
 *    vector<float> onValueDataValue = {1.0};
 *    onValueTensor->SetData((uint8_t*)onValueDataValue.data(), 1 * sizeof(float));
 *    auto onValue = hiai::op::Const("onValue").set_attr_value(onValueTensor);
 *
 *    TensorDesc offValueTensorDesc(Shape(), FORMAT_NCHW, DT_FLOAT);
 *    TensorPtr offValueTensor = std::make_shared<hiai::Tensor>(offValueTensorDesc);
 *    vector<float> offValueDataValue = {1.0};
 *    offValueTensor->SetData((uint8_t*)offValueDataValue.data(), 1 * sizeof(float));
 *    auto offValue = hiai::op::Const("offValue").set_attr_value(offValueTensor);
 *
 *    auto oneHot = hiai::op::OneHot("oneHot")
 *                  .set_input_x(x)
 *                  .set_input_depth(depth)
 *                  .set_input_on_value(onValue)
 *                  .set_input_off_value(offValue)
 *                  .set_attr_axis(0);
 */
HIAI_REG_OP(OneHot)
.HIAI_INPUT(x, TensorType({ DT_INT32, DT_UINT8 }))
.HIAI_INPUT(depth, TensorType({ DT_INT32 }))
.HIAI_INPUT(on_value, TensorType({ DT_UINT8, DT_INT8, DT_FLOAT, DT_BOOL }))
.HIAI_INPUT(off_value, TensorType({ DT_UINT8, DT_INT8, DT_FLOAT, DT_BOOL }))
.HIAI_OUTPUT(y, TensorType({ DT_UINT8, DT_INT8, DT_FLOAT, DT_BOOL }))
.HIAI_ATTR(axis, AttrValue::INT { -1 })
.HIAI_OP_END()

/*
 * Obtain the shape of input tensor,  expressed in the form of one-dimensional integer tensor.
 * <Input>
 *    x : The input tensor.
 * <Output>
 *    y : The output tensor.
 * <Attr>
 *    dtype : Reserved. data type of the output y.
 * <Added in HiAI version>
 *    100.320.010.010
 * <Examples>
 *    TensorDesc xDesc(Shape({4, 5, 6, 7}), FORMAT_NCHW, DT_FLOAT);
 *    hiai::op::Data x = hiai::op::Data("x");
 *    x.update_input_desc_x(xDesc);
 *
 *    auto shape = hiai::op::Shape("shape")
 *               .set_input_x(x);
 */
HIAI_REG_OP(Shape)
.HIAI_INPUT(x, TensorType({ DT_FLOAT, DT_INT32, DT_BOOL, DT_UINT8 }))
.HIAI_OUTPUT(y, TensorType({ DT_INT32 }))
.HIAI_ATTR(dtype, AttrValue::INT { DT_INT32 })
.HIAI_OP_END()

/*
 * Dequantizes the 'input' tensor into a float tensor.
 * <Input>
 *    x : Tensor of type uint8.
 *    min_range : Tensor of type float, defining the minimum scalar value possibly produced for the input.
 *                Min need lessequal 0.
 *    max_range : Tensor of type float, defining the maximum scalar value possibly produced for the input.
 * <Output>
 *    y : Tensor of type float.
 * <Attr>
 *    mode : Only support MIN_COMBINED.
 * <Added in HiAI version>
 *    100.320.010.010
 */
HIAI_REG_OP(Dequantize)
.HIAI_INPUT(x, TensorType({ DT_UINT8 }))
.HIAI_INPUT(min_range, TensorType({ DT_FLOAT }))
.HIAI_INPUT(max_range, TensorType({ DT_FLOAT }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT }))
.HIAI_ATTR(mode, AttrValue::STR { "MIN_COMBINED" })
.HIAI_OP_END()

/*
 * Quantizes the 'input' tensor of type float to 'output' tensor of type uint8.
 * <Input>
 *    x : Tensor of type float.
 *    min_range : Tensor of type float, defining the minimum scalar value possibly produced for the input.
 *                Min need lessequal 0.
 *    max_range : Tensor of type float, defining the maximum scalar value possibly produced for the input.
 * <Output>
 *    y : Tensor of type uint8.
 * <Attr>
 *    mode : Only support MIN_COMBINED.
 * <Added in HiAI version>
 *    100.320.010.010
 */
HIAI_REG_OP(Quantize)
.HIAI_INPUT(x, TensorType({ DT_FLOAT }))
.HIAI_INPUT(min_range, TensorType({ DT_FLOAT }))
.HIAI_INPUT(max_range, TensorType({ DT_FLOAT }))
.HIAI_OUTPUT(y, TensorType({ DT_UINT8 }))
.HIAI_ATTR(mode, AttrValue::STR { "MIN_COMBINED" })
.HIAI_OP_END()

/*
 * Broadcasts an array for a compatible shape.
 * <Input>
 *    x : A Tensor to broadcast.
 *    shape : The shape of the desired output, must be const op.
 * <Output>
 *    y : Result, has the same type as x.
 * <Added in HiAI version>
 *    100.500.010.010
 */
HIAI_REG_OP(BroadcastTo)
.HIAI_INPUT(x, TensorType({ DT_FLOAT, DT_INT8, DT_UINT8, DT_BOOL, DT_INT32 }))
.HIAI_INPUT(shape, TensorType({ DT_INT32 }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT, DT_INT8, DT_UINT8, DT_BOOL, DT_INT32 }))
.HIAI_OP_END()

/*
 * Quantizes the 'input' tensor of type float to 'output' tensor of type specified.
 *     This op should be used before quantize op together with DequantizeV2 after quantize op.
 * <Input>
 *    x : Tensor of type float.
 * <Output>
 *    y : Tensor of type specified by dtype.
 * <Attr>
 *    scale : Quantize scale values. Currently only support Per Layer mode, scale value size is 1.
 *    offset : Quantize offset values. offset value nums is equal to scale values nums.
 *    dtype : Output data type after quantize.
 *    round_mode : Quantize round mode. Candidates:Round, Floor, Ceil.
 *                 Currently round_mode is not supported, param is reserved.
 *    sqrt_mode : Quantize with square root calculations. If true, sqrt the input data.
 *                Currently sqrt_mode is not supported, param is reserved.
 * <Added in HiAI version>
 *    100.515.020.100
 * <Examples>
 *    hiai::op::Data data("data");
 *    std::vector<float> inputScales = { 0.0023456 };
 *    std::vector<float> inputOffsets = { 120.00 };
 *    hiai::op::QuantizeV2 conv1Quantize = hiai::op::QuantizeV2("conv1_quantize")
 *                                              .set_input_x(data)
 *                                              .set_attr_scale(inputScales)
 *                                              .set_attr_offset(inputOffsets)
 *                                              .set_attr_dtype(ge::DT_UINT8);
 *    hiai::op::Convolution conv1 = hiai::op::Convolution1("conv1")
 *                                      .set_input_x(conv1Quantize);
 */
HIAI_REG_OP(QuantizeV2)
.HIAI_INPUT(x, TensorType({ DT_FLOAT }))
.HIAI_OUTPUT(y, TensorType({ DT_INT16, DT_UINT8, DT_INT8, DT_INT4 }))
.HIAI_REQUIRED_ATTR(scale, AttrValue::LIST_FLOAT)
.HIAI_REQUIRED_ATTR(offset, AttrValue::LIST_FLOAT)
.HIAI_REQUIRED_ATTR(dtype, AttrValue::INT)
.HIAI_ATTR(round_mode, AttrValue::STR { "Round" })
.HIAI_ATTR(sqrt_mode, AttrValue::BOOL { false })
.HIAI_ATTR(shape, AttrValue::LIST_INT ({}))
.HIAI_OP_END()

/*
 * Dequantizes the 'input' tensor of type int32 to 'output' tensor of float.
 *     This op should be used after quantize op together with QuantizeV2 before quantize op.
 * <Input>
 *    x : Tensor of type int32.
 * <Output>
 *    y : Tensor of type float.
 * <Attr>
 *    deq_scale : Dequantize scale values. This attr is not used currently, it can be calculated automatically.
 *    sqrt_mode : Quantize with square root calculations. If true, sqrt the input data.
 *                  Currently sqrt_mode is not supported, param is reserved.
 *    relu_flag : Specifing whether to perform ReLU.
 * <Added in HiAI version>
 *    100.515.020.100
 * <Examples>
 *    hiai::op::Convolution conv1 = hiai::op::Convolution("conv1");
 *    hiai::op::DequantizeV2 conv1Dequantize = hiai::op::DequantizeV2("conv1_dequantize")
 *                                   .set_input_x(conv1);
 */
HIAI_REG_OP(DequantizeV2)
.HIAI_INPUT(x, TensorType({ DT_INT32 }))
.HIAI_OUTPUT(y, TensorType({ DT_FLOAT }))
.HIAI_ATTR(deq_scale, AttrValue::LIST_FLOAT ({}))
.HIAI_ATTR(sqrt_mode, AttrValue::BOOL { false })
.HIAI_ATTR(relu_flag, AttrValue::BOOL { false })
.HIAI_ATTR(shape, AttrValue::LIST_INT ({}))
.HIAI_OP_END()
} // namespace hiai
// clang-format on

#endif