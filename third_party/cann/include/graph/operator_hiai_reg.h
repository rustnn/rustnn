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

#ifndef HIAI_OP_REG_H
#define HIAI_OP_REG_H

#include "graph/op_reg.h"

namespace hiai {
using namespace ge;

#define HIAI_REG_OP(x) \
    namespace op { \
    class x : public ge::Operator { \
        typedef x _THIS_TYPE; \
\
    public: \
        static constexpr const char* TYPE = #x; \
        explicit x(const string& name) : Operator(name, #x, 6) \
        { \
            __##x(); \
        } \
        explicit x() : Operator(#x) \
        { \
            __##x(); \
        } \
\
    private: \
        void __##x() \
        { \
            OpReg()

#define HIAI_ATTR(x, default_value) \
    N(); \
    __attr_##x(); \
    } \
\
public: \
    static constexpr const char* x = #x; \
    _THIS_TYPE& set_attr_##x(decltype(default_value) v) \
    { \
        auto attr = AttrValue::CreateFrom(v); \
        Operator::SetAttr(#x, std::move(attr)); \
        return *this; \
    } \
\
private: \
    void __attr_##x() \
    { \
        auto defaultAttr = AttrValue::CreateFrom(default_value); \
        Operator::OptionalAttrRegister(#x, std::move(defaultAttr)); \
        string attr_name(#x); \
        OpReg()

#define HIAI_REQUIRED_GRAPH(x) \
    N(); \
    __required_graph_##x(); \
    } \
\
public: \
    static constexpr const char* GRAPH_NAME_##x = #x; \
    _THIS_TYPE& set_attr_##x(AttrValue::STR v) \
    { \
        auto attr = AttrValue::CreateFrom(v); \
        Operator::SetAttr(#x, std::move(attr)); \
        return *this; \
    } \
\
    _THIS_TYPE& set_graph_builder_##x(const GraphBuilderFn& v) \
    { \
        Operator::SetGraphBuilder(#x, v); \
        return *this; \
    } \
\
private: \
    void __required_graph_##x() \
    { \
        Operator::AttrRegister(#x, AttrValue::ValueType::VT_STRING); \
        OpReg()

#define HIAI_REQUIRED_ATTR(x, type) \
    N(); \
    __required_attr_##x(); \
    } \
\
public: \
    static constexpr const char* x = #x; \
    _THIS_TYPE& set_attr_##x(type v) \
    { \
        auto attr = AttrValue::CreateFrom(v); \
        Operator::SetAttr(#x, std::move(attr)); \
        return *this; \
    } \
\
private: \
    void __required_attr_##x() \
    { \
        GraphGetType<type> ret {}; \
        auto attr = AttrValue::CreateFrom(ret); \
        Operator::AttrRegister(#x, attr.GetValueType()); \
        string attr_name(#x); \
        OpReg()

#define HIAI_INPUT(x, t) \
    N(); \
    __input_##x(); \
    } \
\
public: \
    static constexpr const char* INPUT_NAME_##x = #x; \
    _THIS_TYPE& set_input_##x(const Operator& v, const string& srcName) \
    { \
        Operator::SetInput(#x, v, srcName); \
        return *this; \
    } \
    _THIS_TYPE& set_input_##x(const Operator& v) \
    { \
        Operator::SetInput(#x, v); \
        return *this; \
    } \
\
    _THIS_TYPE& set_input_##x(const OpAnchor v) \
    { \
        Operator::SetInput(#x, v); \
        return *this; \
    } \
    GraphErrCodeStatus update_input_desc_##x(const TensorDesc& tensorDesc) \
    { \
        return Operator::UpdateInputDesc(#x, tensorDesc); \
    } \
\
private: \
    void __input_##x() \
    { \
        Operator::InputRegister(#x); \
        OpReg()

#define HIAI_OPTIONAL_INPUT(x, t) \
    N(); \
    __optional_input_##x(); \
    } \
\
public: \
    static constexpr const char* INPUT_NAME_##x = #x; \
    _THIS_TYPE& set_input_##x(const Operator& v) \
    { \
        Operator::SetInput(#x, v); \
        return *this; \
    } \
    _THIS_TYPE& set_input_##x(const Operator& v, const string& srcName) \
    { \
        Operator::SetInput(#x, v, srcName); \
        return *this; \
    } \
    GraphErrCodeStatus update_input_desc_##x(const TensorDesc& tensorDesc) \
    { \
        return Operator::UpdateInputDesc(#x, tensorDesc); \
    } \
    _THIS_TYPE& set_input_##x(const OpAnchor v) \
    { \
        Operator::SetInput(#x, v); \
        return *this; \
    } \
\
private: \
    void __optional_input_##x() \
    { \
        Operator::OptionalInputRegister(#x); \
        OpReg()

#define HIAI_OUTPUT(x, t) \
    N(); \
    __out_##x(); \
    } \
\
public: \
    static constexpr const char* OUTPUT_NAME_##x = #x; \
    /* deprecated function */ \
    GraphErrCodeStatus update_output_desc_##x(const TensorDesc& tensorDesc) const \
\
    { \
        (void)tensorDesc; \
        return GRAPH_SUCCESS; \
    } \
\
private: \
    void __out_##x() \
    { \
        Operator::OutputRegister(#x); \
        OpReg()

#define HIAI_OPTIONAL_OUTPUT(x, t) \
    N(); \
    __out_##x(); \
    } \
\
public: \
    static constexpr const char* OUTPUT_NAME_##x = #x; \
\
    /* deprecated function */ \
    GraphErrCodeStatus update_output_desc_##x(const TensorDesc& tensorDesc) const \
    { \
        (void)tensorDesc; \
        return GRAPH_SUCCESS; \
    } \
\
private: \
    void __out_##x() \
    { \
        OpReg()

#define HIAI_DYNAMIC_INPUT(x, t) \
    N(); \
    __dy_input_##x(); \
    } \
\
public: \
    static constexpr const char* INPUT_NAME_##x = #x; \
    _THIS_TYPE& create_dynamic_input_##x(unsigned int number) \
    { \
        Operator::DynamicInputRegister(#x, number); \
        return *this; \
    } \
    _THIS_TYPE& set_dynamic_input_##x(unsigned int dstIndex, const Operator& v) \
    { \
        Operator::SetDynamicInput(#x, dstIndex, v); \
        return *this; \
    } \
    _THIS_TYPE& set_dynamic_input_##x(unsigned int dstIndex, const Operator& v, const string& srcName) \
    { \
        Operator::SetDynamicInput(#x, dstIndex, v, srcName); \
        return *this; \
    } \
    _THIS_TYPE& set_dynamic_input_##x(unsigned int dstIndex, const OpAnchor v) \
    { \
        Operator::SetDynamicInput(#x, dstIndex, v); \
        return *this; \
    } \
    /* deprecated function */ \
    GraphErrCodeStatus update_dynamic_input_desc_##x(unsigned int index, const TensorDesc& tensorDesc) const \
    { \
        (void)index; \
        (void)tensorDesc; \
        return GRAPH_SUCCESS; \
    } \
\
private: \
    void __dy_input_##x() \
    { \
        Operator::DynamicInputRegister(#x, 0); \
        OpReg()

#define HIAI_DYNAMIC_GRAPH(x) \
    N(); \
    __graph_##x(); \
    } \
\
public: \
    static constexpr const char* GRAPH_NAME_BRANCHES = #x; \
    _THIS_TYPE& create_dynamic_subgraph_##x(unsigned int number) \
    { \
        Operator::SubgraphCountRegister(#x, number); \
        return *this; \
    } \
    _THIS_TYPE& set_dynamic_subgraph_builder_##x(unsigned int index, const GraphBuilderFn &v) \
    { \
        Operator::SetSubgraphBuilder(#x, index, v); \
        return *this; \
    } \
    GraphBuilderFn get_dynamic_subgraph_builder_##x(unsigned int index) const \
    { \
        return Operator::GetDynamicSubgraphBuilder(#x, index); \
    } \
\
private: \
    void __graph_##x() \
    { \
        OpReg()

#define HIAI_DYNAMIC_OUTPUT(x, t) \
    N(); \
    __dy_output_##x(); \
    } \
\
public: \
    static constexpr const char* OUTPUT_NAME_##x = #x; \
    _THIS_TYPE& create_dynamic_output_##x(unsigned int number) \
    { \
        Operator::DynamicOutputRegister(#x, number); \
        return *this; \
    } \
    /* deprecated function */ \
    GraphErrCodeStatus update_dynamic_output_desc_##x(unsigned int index, const TensorDesc& tensorDesc) const \
    { \
        (void)index; \
        (void)tensorDesc; \
        return GRAPH_SUCCESS; \
    } \
    OpAnchor get_output(unsigned int dstIndex) const \
    { \
        return Operator::GetOutput(dstIndex); \
    } \
\
private: \
    void __dy_output_##x() \
    { \
        Operator::DynamicOutputRegister(#x, 0); \
        OpReg()

#define HIAI_OP_END() \
    N(); \
    } \
    } \
    ; \
    }
} // namespace hiai
#endif // HIAI_OP_REG_H
