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
 * \file functional_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_FUNCTIONAL_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_FUNCTIONAL_OPS_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {

/**
 * @brief Select one of the subgraphs to pass the input tensors and return the output tensors.
 *       If "cond" means True, the selected subgraph is "then_branch".
 *       Otherwise, the selected subgraph is "else_branch" . \n

 * @par Inputs:
 * @li cond: A Tensor. If "cond" is not a scalar of boolean type,
 *          it will be converted to a boolean according to the following rule:
 *          if "cond" is a numerical scalar, non-zero means True and zero means False;
 *          if "cond" is a string scalar, non-empty means True and empty means False;
 *          if "cond" is not a scalar, non-empty means True and empty means False.
 * @li input: The input tensors . It's a dynamic input. \n

 * @par Graphs:
 * @li then_branch: A subgraph takes 'input' and returns a list of tensors,
 *                 whose types are the same as what else_branch returns.
 * @li else_branch: A subgraph takes 'input' and returns a list of tensors,
 *                 whose types are the same as what then_branch returns . \n

 * @par Outputs:
 * output: The output tensors returned by either then_branch(input) or else_branch(input).
 *        It's a dynamic output. \n

 * @par Third-party framework compatibility
 * Compatible with the TensorFlow operator If.
 */
REG_OP(If)
    .INPUT(cond, TensorType::ALL())
    .DYNAMIC_INPUT(input, TensorType::ALL())
    .DYNAMIC_OUTPUT(output, TensorType::ALL())
    .GRAPH(then_branch)
    .GRAPH(else_branch)
    .OP_END_FACTORY_REG(If)

/**
 * @brief Cyclic execute the "body" subgraph until the return tensor of "cond" subgraph means False . \n

 * @par Inputs:
 * input: The input tensors . It's a dynamic input. \n

 * @par Graphs:
 * @li cond: A subgraph takes 'input' and returns a tensor.
 *          If the tensor is not a scalar of boolean type,
 *          it will be converted to a boolean according to the following rule:
 *          if it is a numerical scalar, non-zero means True and zero means False;
 *          if it is a string scalar, non-empty means True and empty means False;
 *          if it is not a scalar, non-empty means True and empty means False.
 * @li body: A subgraph takes 'input' and returns another list of tensors . \n

 * @par Attributes:
 * parallel_iterations: An optional int, default as 10 . \n

 * @par Outputs:
 * output: The output tensors returned by "body". Has the same type as "input" . It's a dynamic output. \n

 * @par Third-party framework compatibility
 * Compatible with the TensorFlow operator While.
 */
REG_OP(While)
    .DYNAMIC_INPUT(input, TensorType::ALL())
    .DYNAMIC_OUTPUT(output, TensorType::ALL())
    .GRAPH(cond)
    .GRAPH(body)
    .ATTR(parallel_iterations, Int, 10)
    .OP_END_FACTORY_REG(While)
}  // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_INC_FUNCTIONAL_OPS_H_
