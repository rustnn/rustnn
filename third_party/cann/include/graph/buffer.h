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

#ifndef GE_BUFFER_H
#define GE_BUFFER_H

#include <memory>
#include <string>
#include <vector>
#include <unordered_set>

#include "graph/graph_api_export.h"

namespace ge {
using std::shared_ptr;
using std::string;

class GRAPH_API_EXPORT Buffer {
public:
    Buffer();

    Buffer(std::string* buffer, bool isOwner);

    Buffer(const Buffer& other);

    explicit Buffer(std::size_t size, std::uint8_t defaultVal = 0);

    ~Buffer();

    Buffer& operator=(const Buffer& other);

    static void CopyFrom(const std::uint8_t* data, std::size_t bufferSize, Buffer& buffer);

    const std::uint8_t* GetData() const;

    std::uint8_t* MutableData();

    std::size_t GetSize() const;

    void Resize(std::size_t bufferSize);

    void Clear();

    inline const std::uint8_t* data() const
    {
        return GetData();
    }

    inline std::uint8_t* data()
    {
        return MutableData();
    }

    inline std::size_t size() const
    {
        return GetSize();
    }

private:
    std::string* buffer_;

    bool isOwner_{false};

    friend class Tensor;
    void RefTo(const Buffer& buffer);
};
} // namespace ge

#endif // GE_BUFFER_H
