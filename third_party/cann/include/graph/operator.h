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

#ifndef GE_OPERATOR_H
#define GE_OPERATOR_H

#include <functional>
#include <map>
#include <memory>
#include <vector>

#include "graph/attr_value.h"
#include "graph/graph_api_export.h"
#include "graph/ascend_string.h"

namespace ops {
    class OpDefUtils;
}
namespace ge {
constexpr int CURRENT_OP_VERSION = 0;

class OperatorImpl;
using OperatorImplPtr = std::shared_ptr<OperatorImpl>;

class Graph;
using GraphBuilderFn = std::function<void(ge::Graph&)>;

using OpAnchor = std::pair<std::weak_ptr<OperatorImpl>, int32_t>;

class GRAPH_API_EXPORT Operator : public Base {
    friend class ops::OpDefUtils;
public:
    using OpInt = int64_t;
    using OpFloat = float32_t;
    using OpString = std::string;
    using OpBool = bool;
    using OpType = ge::DataType;
    using OpTensor = ge::TensorPtr;
    using OpListInt = std::vector<int64_t>;
    using OpListFloat = std::vector<float32_t>;
    using OpListString = std::vector<std::string>;
    using OpListBool = std::vector<bool>;
    using OpListListInt = std::vector<std::vector<int64_t>>;
    using OpListType = std::vector<ge::DataType>;
    using OpListTensor = std::vector<ge::TensorPtr>;

    Operator() = default;
    explicit Operator(const string& type);
    Operator(const string& name, const string& type, int version = CURRENT_OP_VERSION);
    explicit Operator(OperatorImplPtr&& opImpl);
    ~Operator() override = default;

    const OperatorImplPtr GetImpl() const;
    OpAnchor GetOutput(const uint32_t outIndex) const;
    OpAnchor GetOutput(const string& name, const uint32_t outIndex) const;

    Operator& SetInput(const string& name, const Operator& outOp);
    Operator& SetInput(const string& name, const Operator& outOp, const string& outName);
    Operator& SetInput(int32_t index, const Operator& outOp, int32_t outIndex);
    Operator& SetInput(const string& name, const ge::OpAnchor& opAnchor);

    Operator& SetDynamicInput(const string& name, int32_t index, const Operator& outOp);
    Operator& SetDynamicInput(const string& name, int32_t index, const Operator& outOp, const std::string& outName);
    Operator& SetDynamicInput(const string& name, int32_t index, const ge::OpAnchor& opAnchor);

    Operator& SetAttr(const string& name, AttrValue&& attrValue);
    Operator& SetAttr(const string& name, bool attrValue);
    Operator& SetAttr(const string& name, float attrValue);
    Operator& SetAttr(const string& name, int64_t attrValue);
    Operator& SetAttr(const string& name, std::string attrValue);
    Operator& SetAttr(const string& name, std::vector<bool> attrValue);
    Operator& SetAttr(const string& name, std::vector<float> attrValue);
    Operator& SetAttr(const string& name, std::vector<int64_t> attrValue);
    Operator& SetAttr(const string& name, std::vector<std::vector<int64_t>> attrValue);
    Operator& SetAttr(const string& name, std::vector<std::vector<float>> attrValue);
    GraphErrCodeStatus GetAttr(const string& name, ge::AscendString& attrValue) const;

    Operator& SetGraphBuilder(const string& name, const GraphBuilderFn& graphBuilderFn);

    GraphErrCodeStatus UpdateInputDesc(const string& name, const TensorDesc& tensorDesc);
    string GetName() const;
    string GetType() const;
    void SetName(const string& name);

    void DynamicInputRegister(const string& name, const unsigned int num);
    void DynamicOutputRegister(const string& name, const unsigned int num);

    TensorDesc GetInputDesc(uint32_t index) const;
    TensorDesc GetOutputDesc(uint32_t index) const;
    TensorDesc GetDynamicInputDesc(const std::string &name, uint32_t index) const;

protected:
    void InputRegister(const string& name);
    void OptionalInputRegister(const string& name);

    void SubgraphCountRegister(const string& name, const unsigned int num);
    void SetSubgraphBuilder(const string& name, const unsigned int index, const GraphBuilderFn& builder);
    GraphBuilderFn GetDynamicSubgraphBuilder(const string &name, const unsigned int index) const;

    void OutputRegister(const string& name);

    void AttrRegister(const string& name, AttrValue::ValueType type);
    void OptionalAttrRegister(const string& name, AttrValue&& attrValue);
    void SetType(const string& type);

private:
    std::vector<bool> GetOpIsInputConst() const;
    void SetOpIsInputConst(bool inputConst, uint32_t index);

private:
    OperatorImplPtr impl_ {nullptr};
};
} // namespace ge
#endif // GE_OPERATOR_H
