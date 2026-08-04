/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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

#ifndef INC_EXTERNAL_GRAPH_ASCEND_STRING_H_
#define INC_EXTERNAL_GRAPH_ASCEND_STRING_H_

#include <string>
#include <memory>
#include <functional>

#include "graph/graph_api_export.h"

namespace ge {

using char_t = char;

class GRAPH_API_EXPORT AscendString {
public:
    AscendString() = default;

    ~AscendString() = default;

    AscendString(const char_t *const name);

    AscendString(const char_t *const name, size_t length);

    const char_t* GetString() const;

    size_t GetLength() const;

    size_t Find(const AscendString& ascend_string) const;

    size_t Hash() const;

    bool operator<(const AscendString& d) const;

    bool operator>(const AscendString& d) const;

    bool operator<=(const AscendString& d) const;

    bool operator>=(const AscendString& d) const;

    bool operator==(const AscendString& d) const;

    bool operator!=(const AscendString& d) const;

    bool operator==(const char_t *const d) const;

    bool operator!=(const char_t *const d) const;

private:
    std::shared_ptr<std::string> name_;
};
}  // namespace ge

namespace std {
template <>
struct hash<ge::AscendString> {
    size_t operator()(const ge::AscendString& name) const
    {
        return name.Hash();
    }
};
}  // namespace std
#endif  // INC_EXTERNAL_GRAPH_ASCEND_STRING_H_
