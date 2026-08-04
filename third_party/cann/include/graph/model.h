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

#ifndef GE_MODEL_H
#define GE_MODEL_H

#include <map>
#include <string>
#include <vector>

#include "graph/attr_value.h"
#include "graph/attributes_holder.h"
#include "graph/graph.h"

namespace hiai {
class IModelDef;
}

namespace ge {
using std::string;
using std::vector;

class GRAPH_API_EXPORT Model : public AttrHolder, public Base {
public:
    Model();

    Model(const string& name, const string& customVersion);

    ~Model() override;

    Model(const Model& other) = delete;

    Model& operator=(const Model& other) = delete;

    const string& GetName() const;

    void SetName(const string& name);

    uint32_t GetVersion() const;

    void SetVersion(uint32_t version);

    const string& GetPlatformVersion() const;

    void SetPlatformVersion(const std::string& version);

    Graph GetGraph() const;

    void SetGraph(const Graph& graph);

    GraphErrCodeStatus SerializeTo(hiai::IModelDef* modelDef) const;

    GraphErrCodeStatus Save(Buffer& buffer) const;

    GraphErrCodeStatus SaveFullModel(Buffer& buffer) const;

    GraphErrCodeStatus Load(const uint8_t* data, size_t len);

    GraphErrCodeStatus Dump(const std::string& outFile);

    bool IsValid() const;

    Buffer* CreateWeightBuffer(size_t size);

protected:
    const hiai::IAttrMapDef* GetAttrMapDef() const override;

    hiai::IAttrMapDef* MutableAttrMapDef() override;

private:
    Model(hiai::IModelDef* modelDef);

    hiai::IModelDef* modelDef_;

    Graph graph_;

    Buffer weightBuffer_;
};

} // namespace ge

#endif // GE_MODEL_H
