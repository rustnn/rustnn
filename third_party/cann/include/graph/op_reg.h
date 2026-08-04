/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#ifndef HIAI_OP_REG_COM_H
#define HIAI_OP_REG_COM_H

#include <memory>
#include <vector>

#include "graph/attr_value.h"
#include "graph/operator.h"
#include "graph/tensor.h"
#include "graph/types.h"

namespace ge {
using std::function;
using std::string;
using std::vector;
using TensorPtr = std::shared_ptr<Tensor>;

class OpDesc;

class OpReg {
public:
    OpReg& N()
    {
        return *this;
    }

    OpReg& ATTR()
    {
        return *this;
    }

    OpReg& REQUIRED_ATTR()
    {
        return *this;
    }

    OpReg& INPUT()
    {
        return *this;
    }

    OpReg& OPTIONAL_INPUT()
    {
        return *this;
    }

    OpReg& OUTPUT()
    {
        return *this;
    }

    OpReg& INFER_SHAPE_AND_TYPE()
    {
        return *this;
    }

    OpReg& REQUIRED_GRAPH()
    {
        return *this;
    }

    OpReg& GRAPH()
    {
        return *this;
    }
};

template <typename T>
struct GetType {
    using type = T;

    void assign(type& left, const T& right)
    {
        left = right;
    }
};

template <>
struct GetType<TensorPtr> {
    using type = ConstTensorPtr;

    void assign(type& left, const TensorPtr& right)
    {
        left = right;
    }
};

template <>
struct GetType<ComputeGraphPtr> {
    using type = ConstComputeGraphPtr;

    void assign(type& left, const ComputeGraphPtr& right)
    {
        left = right;
    }
};

template <>
struct GetType<vector<TensorPtr>> {
    using type = vector<ConstTensorPtr>;

    void assign(type& left, const vector<TensorPtr>& right)
    {
        for (auto& item : right) {
            left.push_back(item);
        }
    }
};

template <>
struct GetType<vector<ComputeGraphPtr>> {
    using type = vector<ConstComputeGraphPtr>;

    void assign(type& left, const vector<ComputeGraphPtr>& right)
    {
        for (auto& item : right) {
            left.push_back(item);
        }
    }
};

template <typename T>
using GraphGetType = typename GetType<T>::type;
} // namespace ge
#endif // HIAI_OP_REG_COM_H
