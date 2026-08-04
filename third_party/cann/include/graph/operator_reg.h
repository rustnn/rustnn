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

#ifndef GE_OP_REG2_H
#define GE_OP_REG2_H

#include "graph/op_reg.h"

namespace ge {
using namespace std;

#define SET_VALUE_Int(x) auto attrValue = (x)
#define SET_VALUE_Float(x) auto attrValue = (x)
#define SET_VALUE_Bool(x) auto attrValue = (x)
#define SET_VALUE_String(x) auto attrValue = (x)
#define SET_VALUE_Type(x) auto attrValue = (x)
#define SET_VALUE_Tensor(x) auto attrValue = AttrValue::TENSOR(new (std::nothrow) Tensor(TensorDesc()))
#define SET_VALUE_ListInt(x) auto attrValue = (x)
#define SET_VALUE_ListFloat(x) auto attrValue = (x)
#define SET_VALUE_ListBool(x) auto attrValue = (x)
#define SET_VALUE_ListString(x) auto attrValue = (x)
#define SET_VALUE_ListListInt(x) auto attrValue = (x)
#define SET_VALUE_ListType(x) auto attrValue = (x)

#define REG_OP(x) \
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

#define ATTR(x, Type, ...) \
    ATTR(); \
    __attr_##x(); \
    } \
\
public: \
    static constexpr const char* x = #x; \
    _THIS_TYPE& set_attr_##x(Op##Type v) \
    { \
        auto attr = AttrValue::CreateFrom(v); \
        Operator::SetAttr(#x, std::move(attr)); \
        return *this; \
    } \
\
private: \
    void __attr_##x() \
    { \
        SET_VALUE_##Type(Op##Type(__VA_ARGS__));  \
        auto attr = AttrValue::CreateFrom(attrValue); \
        Operator::OptionalAttrRegister(#x, std::move(attr)); \
        string attr_name(#x); \
        OpReg()

#define GRAPH(x) \
    GRAPH(); \
    __graph_##x(); \
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
    void __graph_##x() \
    { \
        Operator::AttrRegister(#x, AttrValue::ValueType::VT_STRING); \
        OpReg()

#define REQUIRED_GRAPH(x) \
    REQUIRED_GRAPH(); \
    __required_graph_##x(); \
    } \
\
public: \
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

#define REQUIRED_ATTR(x, type) \
    REQUIRED_ATTR(); \
    __required_attr_##x(); \
    } \
\
public: \
    static constexpr const char* x = #x; \
    _THIS_TYPE& set_attr_##x(Op##type v) \
    { \
        auto attr = AttrValue::CreateFrom(v); \
        Operator::SetAttr(#x, std::move(attr)); \
        return *this; \
    } \
\
private: \
    void __required_attr_##x() \
    { \
        GraphGetType<Op##type> ret {}; \
        auto attr = AttrValue::CreateFrom(ret); \
        Operator::AttrRegister(#x, attr.GetValueType()); \
        string attr_name(#x); \
        OpReg()

#define DATATYPE(x, t) \
    N(); \
    __datatype_##x(); \
    } \
\
private: \
  void __datatype_##x() { \
    (void) OpReg()

#define INPUT(x, t) \
    INPUT(); \
    __input_##x(); \
    } \
\
public: \
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

#define OPTIONAL_INPUT(x, t) \
    OPTIONAL_INPUT(); \
    __optional_input_##x(); \
    } \
\
public: \
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

#define OUTPUT(x, t) \
    OUTPUT(); \
    __out_##x(); \
    } \
\
public: \
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

#define OPTIONAL_OUTPUT(x, t) \
    N(); \
    __out_##x(); \
    } \
\
public: \
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

#define DYNAMIC_INPUT(x, t) \
    N(); \
    __dy_input_##x(); \
    } \
\
public: \
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

#define DYNAMIC_GRAPH(x) \
    N(); \
    __graph_##x(); \
    } \
\
public: \
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

#define DYNAMIC_OUTPUT(x, t) \
    N(); \
    __dy_output_##x(); \
    } \
\
public: \
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
    OpAnchor get_output_##x(unsigned int dstIndex) const \
    { \
        return Operator::GetOutput(#x, dstIndex); \
    } \
\
private: \
    void __dy_output_##x() \
    { \
        Operator::DynamicOutputRegister(#x, 0); \
        OpReg()

#define OP_END() \
    N(); \
    } \
    } \
    ; \
    }

#define OP_END_FACTORY_REG(x) OP_END()
} // namespace ge
#endif // GE_OP_REG_H
