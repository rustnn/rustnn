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

#ifndef GE_ATTRIBUTES_HOLDER_H
#define GE_ATTRIBUTES_HOLDER_H

#include <map>
#include <memory>
#include <string>
#include <unordered_set>

#include "graph/debug/ge_error_codes.h"
#include "graph/graph_api_export.h"

namespace hiai {
class IAttrMapDef;
class IAttrDef;
}

namespace ge {
using std::string;

class AttrValue;
using AttrMap = std::map<std::string, AttrValue>;

class GRAPH_API_EXPORT AttrHolder {
public:
    GraphErrCodeStatus SetAttr(const string& name, const AttrValue& value);

    GraphErrCodeStatus GetAttr(const string& name, AttrValue& value) const;

    bool HasAttr(const string& name) const;

    GraphErrCodeStatus DelAttr(const string& name);

    hiai::IAttrDef* MutableAttr(const string& name);

    const hiai::IAttrDef* GetAttr(const string& name) const;

    GraphErrCodeStatus CopyAttrsFrom(const AttrHolder& other);

protected:
    AttrHolder() = default;

    virtual ~AttrHolder() = default;

    const std::map<string, AttrValue> GetAllAttrs() const;

    virtual const hiai::IAttrMapDef* GetAttrMapDef() const = 0;

    virtual hiai::IAttrMapDef* MutableAttrMapDef() = 0;
};

class GRAPH_API_EXPORT Base {
public:
    Base() = default;
    virtual ~Base() = default;
};
} // namespace ge

using BasePtr = std::shared_ptr<ge::Base>;

#endif // GE_ATTRIBUTES_HOLDER_H
