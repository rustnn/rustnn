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

#ifndef GE_OP_REG_H
#define GE_OP_REG_H

#include "graph/op_reg.h"

namespace ge {
#define CPT_REG_OP(x) \
    namespace op { \
    class x : public ge::Operator { \
        typedef x _THIS_TYPE; \
\
    public: \
        explicit x(const string& name) : Operator(name, #x) \
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

#define CPT_ATTR(x, default_value) \
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

#define CPT_REQUIRED_ATTR(x, type) \
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
        ; \
        AttrValue attr = AttrValue::CreateFrom(ret); \
        Operator::AttrRegister(#x, attr.GetValueType()); \
        string attr_name(#x); \
        OpReg()

#define CPT_INPUT(x, t) \
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

#define CPT_OPTIONAL_INPUT(x, t) \
    N(); \
    __optional_input_##x(); \
    } \
\
public: \
    static constexpr const char* INPUT_NAME_##x = #x; \
    _THIS_TYPE& set_input_##x(const Operator& v) \
    { \
        Operator::OptionalInputRegister(#x); \
        Operator::SetInput(#x, v); \
        return *this; \
    } \
    _THIS_TYPE& set_input_##x(const Operator& v, const string& srcName) \
    { \
        Operator::SetInput(#x, v, srcName); \
        return *this; \
    } \
    _THIS_TYPE& set_input_##x(const OpAnchor v) \
    { \
        Operator::SetInput(#x, v); \
        return *this; \
    } \
    GraphErrCodeStatus update_input_desc_##x(const TensorDesc& tensorDesc) \
    { \
        return Operator::UpdateInputDesc(#x, tensorDesc); \
    } \
    /* deprecated function */ \
    GraphErrCodeStatus update_output_desc_##x(const TensorDesc& tensorDesc) const \
    { \
        (void)tensorDesc; \
        return GRAPH_SUCCESS; \
    } \
\
private: \
    void __optional_input_##x() \
    { \
        OpReg()

#define CPT_OUTPUT(x, t) \
    N(); \
    __out_##x(); \
    } \
\
public: \
    static constexpr const char* OUTPUT_NAME_##x = #x; \
\
private: \
    void __out_##x() \
    { \
        Operator::OutputRegister(#x); \
        OpReg()

#define CPT_DYNAMIC_INPUT(x, t) \
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
        OpReg()

#define CPT_DYNAMIC_OUTPUT(x, t) \
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
        OpReg()

#define CPT_OP_END() \
    N(); \
    } \
    } \
    ; \
    }
} // namespace ge
#endif // GE_OP_REG_H
