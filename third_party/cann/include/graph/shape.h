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

#ifndef GE_SHAPE_H
#define GE_SHAPE_H

#include <cstddef>
#include <vector>

#include "graph/debug/ge_error_codes.h"
#include "graph/attributes_holder.h"
#include "graph/graph_api_export.h"
#include "graph/types.h"

namespace hiai {
class IShapeDef;
}

namespace ge {
class GRAPH_API_EXPORT Shape {
public:
    Shape();
    Shape(hiai::IShapeDef* shapeDef, bool isOwner);
    explicit Shape(std::vector<int64_t> s);
    Shape(const Shape& other);
    Shape& operator=(const Shape& other);
    ~Shape();

    size_t GetDimNum() const;

    // if the idx is invalid, return 0
    int64_t GetDim(size_t idx) const;
    GraphErrCodeStatus SetDim(size_t idx, int64_t value);
    const std::vector<int64_t> GetDims() const;
    int64_t GetTotalDimNum() const;
    uint32_t GetShapeSize() const;
    bool operator==(const Shape& other) const;

private:
    hiai::IShapeDef* shapeDef_;
    bool isOwner_ {false};

    friend class TensorDesc;
    void RefTo(const Shape& shape);
};
} // namespace ge

#endif // GE_SHAPE_H
